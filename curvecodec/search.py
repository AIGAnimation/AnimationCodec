"""The encoder's closed-loop search state: the decoded clip so far, its object-space transforms and its shell errors,
and the subtree evaluation of a candidate change of one bone's local transform under the mean-error budgets.

A candidate for bone b is accepted (`try_local`) when every bone c of b's subtree keeps
  sum_f e[f, c] <= mlim[c] x F      (its mean cap)
  max_f e[f, c] <= limits[c]        (its anti-pop guard)
and the subtree's added error fits what is left of the clip total. The numerics follow metric.fk_object exactly
(float64 chain of normalised quaternions from the float32 locals), so the final `err` is the shell error of the
decoded clip.
"""
import numpy as np

from . import fast as _fast
from .metric import SHELL_CM, fk_object, subtree_lists
from .rot import quat_cols, quat_mul, quat_normalize, rotate


class Search:
    """limits [B] = anti-pop guard (max), mlim [B] = per-bone mean caps, total = cap on sum_{f, c} e (cm x frames)."""

    def __init__(self, raw_local, parents, precision, mlim, guard, total=np.inf):
        self.parents = np.asarray(parents)
        self.F, self.B = raw_local.shape[:2]
        self.p = float(precision)
        self.limits = np.asarray(guard, dtype=np.float64)
        self.mlim = np.asarray(mlim, dtype=np.float64)
        self.total = float(total)
        self.shell = SHELL_CM
        self.subtrees = subtree_lists(self.parents)
        rr, rp = fk_object(raw_local, self.parents)
        self.raw_cols = np.ascontiguousarray(quat_cols(rr.reshape(-1, 4)).reshape(self.F, self.B, 3, 3))
        self.raw_rot = rr; self.raw_pos = rp
        self.n_allow = np.zeros(self.B)              # (an unused slot of the C state)
        self.cur_local = None                        # float32 [F, B, 7]
        self.obj_rot = self.obj_pos = None
        self.err = None                              # float64 [F, B]
        self.emax = self.cnt = self.esum = None
        self._pending = None                         # (b, tag, new_local_b, rot, pos, err, cnt) of the last passing candidate
        self._relax = None                           # (mlim, limits) while a track that cannot pass is searched

    def reset(self, cur_local):
        self.cur_local = np.array(cur_local, dtype=np.float32)
        self.obj_rot, self.obj_pos = fk_object(self.cur_local, self.parents)
        self.err = np.empty((self.F, self.B))
        for c in range(self.B):
            self.err[:, c] = self._bone_err(c, self.obj_rot[:, c], self.obj_pos[:, c])
        self.emax = self.err.max(axis=0)
        self.cnt = (self.err > self.p).sum(axis=0).astype(np.float64)
        self.esum = self.err.sum(axis=0)

    def refresh(self):
        self.emax = self.err.max(axis=0); self.esum = self.err.sum(axis=0)
        self.cnt = (self.err > self.p).sum(axis=0).astype(np.float64)

    def _bone_err(self, c, rot_c, pos_c):
        d = self.shell * (self.raw_cols[:, c] - quat_cols(rot_c)) + (self.raw_pos[:, c] - pos_c)[:, None, :]
        return np.sqrt(np.max(np.sum(d * d, axis=-1), axis=-1))

    def eval_subtree(self, b, new_local_b, early_exit=True):
        """Object transforms and errors of b's subtree with bone b's local replaced by new_local_b (float32 [F, 7]) ->
        (ok, rot, pos, err, cnt) dicts by bone; with early_exit the dicts stop at the first bone over a limit."""
        mlim, glim = (self.mlim, self.limits) if self._relax is None else self._relax
        r = _fast.eval_subtree_mean(self, b, new_local_b, early_exit, mlim, glim)
        if r is not None:
            return r
        sub = self.subtrees[b]
        t_rot, t_pos, t_err, cnt = {}, {}, {}, {}
        ok = True
        added = 0.0
        F = float(self.F)
        for c in sub:
            loc = new_local_b if c == b else self.cur_local[:, c]
            lq = quat_normalize(loc[:, :4].astype(np.float64)); lt = loc[:, 4:7].astype(np.float64)
            pc = int(self.parents[c])
            if pc < 0:
                rot_c, pos_c = lq, lt
            else:
                pr = t_rot[pc] if pc in t_rot else self.obj_rot[:, pc]
                pp = t_pos[pc] if pc in t_pos else self.obj_pos[:, pc]
                rot_c = quat_normalize(quat_mul(lq, pr)); pos_c = rotate(lt, pr) + pp
            e_c = self._bone_err(c, rot_c, pos_c)
            s_c = float(e_c.sum())
            if s_c > mlim[c] * F or e_c.max() > glim[c]:
                ok = False
                if early_exit:
                    return ok, t_rot, t_pos, t_err, cnt
            added += s_c - self.esum[c]
            t_rot[c] = rot_c; t_pos[c] = pos_c; t_err[c] = e_c; cnt[c] = float((e_c > self.p).sum())
        if np.isfinite(self.total) and added > self.total - float(self.esum.sum()) + 1e-9:
            ok = False
        return ok, t_rot, t_pos, t_err, cnt

    def try_local(self, b, new_local_b, tag=None):
        """Does the candidate pass? A passing candidate stays pending (with `tag`) until commit_pending()."""
        ok, t_rot, t_pos, t_err, cnt = self.eval_subtree(b, new_local_b)
        if ok:
            self._pending = (b, tag, new_local_b, t_rot, t_pos, t_err, cnt)
        return ok

    def pending_tag(self):
        return None if self._pending is None else (self._pending[0], self._pending[1])

    def drop_pending(self):
        self._pending = None

    def commit_pending(self):
        b = self._pending[0]
        if not _fast.commit_rows(self):
            _, _, new_local_b, t_rot, t_pos, t_err, cnt = self._pending
            self.cur_local[:, b] = new_local_b
            for c in self.subtrees[b]:
                self.obj_rot[:, c] = t_rot[c]; self.obj_pos[:, c] = t_pos[c]; self.err[:, c] = t_err[c]
                self.emax[c] = t_err[c].max(); self.cnt[c] = cnt[c]
            self._pending = None
        sub = self.subtrees[b]
        self.esum[sub] = self.err[:, sub].sum(axis=0)

    def force_local(self, b, new_local_b):
        """Install new_local_b for bone b whatever the budgets say."""
        _, t_rot, t_pos, t_err, cnt = self.eval_subtree(b, new_local_b, early_exit=False)
        self._pending = (b, 'force', new_local_b, t_rot, t_pos, t_err, cnt)
        self.commit_pending()

    def relax_to(self, b, new_local_b, ceil_mean, ceil_max, factor=1.02):
        """For the search of one track that cannot pass even unquantized: relax every subtree bone's mean cap and guard to
        `factor` x the error of the unquantized track (new_local_b), never above ceil_mean / ceil_max."""
        _, _, _, t_err, _ = self.eval_subtree(b, new_local_b, early_exit=False)
        ml = self.mlim.copy(); gl = self.limits.copy()
        for c in self.subtrees[b]:
            ml[c] = max(ml[c], min(factor * float(t_err[c].mean()), ceil_mean[c]))
            gl[c] = max(gl[c], min(factor * float(t_err[c].max()), ceil_max[c]))
        self._relax = (ml, gl)

    def clear_relax(self):
        self._relax = None


def eval_subtree_frames(S, b, local_b, idx):
    """Object transforms and errors of b's subtree at the frames idx with bone b's local replaced by local_b -> (rot,
    pos, err) dicts by bone (the other bones and the ancestors from the search state)."""
    r = _fast.eval_subtree_frames(S, b, local_b, idx)
    if r is not None:
        return r
    t_rot, t_pos, t_err = {}, {}, {}
    for c in S.subtrees[b]:
        loc = local_b[idx] if c == b else S.cur_local[idx, c]
        lq = quat_normalize(loc[:, :4].astype(np.float64)); lt = loc[:, 4:7].astype(np.float64)
        pc = int(S.parents[c])
        if pc < 0:
            rot_c, pos_c = lq, lt
        else:
            pr = t_rot[pc] if pc in t_rot else S.obj_rot[idx, pc]
            pp = t_pos[pc] if pc in t_pos else S.obj_pos[idx, pc]
            rot_c = quat_normalize(quat_mul(lq, pr)); pos_c = rotate(lt, pr) + pp
        d = S.shell * (S.raw_cols[idx, c] - quat_cols(rot_c)) + (S.raw_pos[idx, c] - pos_c)[:, None, :]
        t_err[c] = np.sqrt(np.max(np.sum(d * d, axis=-1), axis=-1))
        t_rot[c] = rot_c; t_pos[c] = pos_c
    return t_rot, t_pos, t_err
