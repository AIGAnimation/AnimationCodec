"""CurveCodec: the encoder (closed-loop quantization and key selection under the mean-error contract), the packer and
the decoder of the v1 blob.

Encoder, per clip and precision p (every state change is verified on the search state, so the encoder knows the exact
error of the decoded clip):
  1. folding     ACL's constant / default rule per track (quantize.fold_types); folded tracks on the root paths of
                 bones whose floor (constants + every other track unquantized) misses its cap are un-folded
  2. constants   the first sample of a constant rotation (log-map) at the step of p / 256 local error, of a constant
                 translation at p / 256 / sqrt(3), or the free rig offset
  3. steps       per animated track the largest step whose all-keys reconstruction keeps the subtree within alpha x
                 the per-bone budgets: analytic start, quarter-octave bracket search, snapped to the 16-per-octave
                 grid and searched again on the grid with a lookahead
  4. keys        the clip-wide RD ladder of key removals and step growths (keys.py) under the full mean caps; per
                 track the coded-cost proxy picks keyframes or all keys; the all-keys tracks then grow their step;
                 the ladder runs again with what the restored tracks gave back
  5. components  the keyframe tracks are thinned per component (keys.component_stage)
  6. repair      if the state still misses a cap: keys back / one grid unit finer along the violating root path
  7. pack        the predictor of every curve by its cost proxy; container.py's header; every residual coded under
                 the integer transformer (model.py) with features.py's alphabet, all curves in lockstep
"""
import numpy as np

from . import fast as _fast
from .container import (CT_MODE, CT_PID_ALL, CT_PID_KF, CT_TYPE_ROT, CT_TYPE_TRANS, CT_X0_ROT, CT_X0_TRANS, ESC, M_ALL, M_CK, M_KF,
                        M_STATIC, MAGIC, NGAP, NSYM, T_ROT_ANIM, T_ROT_CONST, T_ROT_DEF, T_TR_ANIM, T_TR_CONST, T_TR_DEF, T_TR_OFF,
                        VERSION, CT_FLAG, CT_GAP, Sink, cls_cost, get_mags, init_counts, leb, prec_code, read_leb,
                        read_prec, res_proxy)
from .contract import Contract
from .features import ALPHABET, STEP_COL, WIDTHS, CurveState, prec_index, symbol_freqs, symbol_index, symbol_value
from .keys import Tracks, ck_finish, component_stage, kf_cost, ladder, reconstruct, reconstruct_c
from .metric import bone_depth, depth_class, shell_error, shell_radii, subtree_lists
from .model import MODEL_FILE, PAD, IntTf
from .predict import IncPred, fixed_residuals, key_residuals, nlms_batch, nlms_residuals
from .quantize import SQRT3, Grid, const_rot, const_rot_ints, const_trans, fold_types, idx3, quantize
from .rans import BitReader, RansDecoder, model_forward, rans_encode, unzigzag, zigzag
from .rot import log_map, quat_cols, quat_normalize
from .search import Search

QUARTER = 2.0 ** 0.25          # step factor per unit of the stage-1 bracket search
GROW_SEQ = (1, 2, 4, 8, 16, 24, 32, 40)
SHRINK_SEQ = (-1, -2, -4, -8, -16, -32, -40)


def default_local(F, B, cv):
    """A clip with every track at its default (identity / zero) or at its constant value cv[(kind, bone)]."""
    loc = np.zeros((F, B, 7), dtype=np.float32); loc[:, :, 3] = 1.0
    for (kind, b), v in cv.items():
        if kind == 'rot':
            loc[:, b, :4] = v
        else:
            loc[:, b, 4:7] = v
    return loc


def bracket_search(passes_at, at0):
    """The largest exponent e (quarter octaves) with passes_at(e): grow through GROW_SEQ (or shrink through SHRINK_SEQ
    when e = 0 fails), then bisect. Returns the smallest shrink even if it fails."""
    if at0:
        lo, hi = 0, None
        for e in GROW_SEQ:
            if passes_at(e):
                lo = e
            else:
                hi = e; break
        if hi is None:
            return lo
    else:
        lo, hi = None, 0
        for e in SHRINK_SEQ:
            if passes_at(e):
                lo = e; break
            hi = e
        if lo is None:
            return SHRINK_SEQ[-1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes_at(mid):
            lo = mid
        else:
            hi = mid
    return lo


class CurveCodec:
    """encode(clip, precision, contract, margin) -> blob; decode(blob, parents, offsets, fps) -> float32 [F, B, 7]."""

    G = 16                     # animated-step grid: steps per octave
    LOOKAHEAD = 4              # grid lookahead, in eighths of an octave
    CFRAC = 1.0 / 256          # local error of a quantized constant, as a fraction of p
    ALPHA = 0.5                # share of the per-bone budget the step search may spend before the key ladder
    C0 = 0.5                   # per-symbol overhead of the cost proxies (bits)
    ID_BITS = 1.5              # cost proxy of one predictor id
    GBITS = 2.5                # calibration of the step-growth bit proxy

    def __init__(self, model_path=MODEL_FILE):
        self.grid = Grid(self.G)
        self.cgrid = Grid(8)
        self.la_units = self.LOOKAHEAD * self.grid.unit
        self.grow_sizes = tuple(int(2 ** k) for k in range(int(np.log2(self.la_units)) + 1))
        self.model = IntTf(model_path)
        self.init = init_counts()

    # ============================================================================================================
    # encoder
    # ============================================================================================================
    def encode(self, clip, precision, contract=None, margin=1.0, streams=None):
        """clip: .local float32 [F, B, 7], .parents, .offsets, .fps, .ref(p) = ACL's error statistics at p. A dict passed as
        `streams` receives the coded residual streams (the model's training data, see pack)."""
        contract = contract or Contract()
        p = float(precision); F, B = clip.local.shape[:2]; parents = np.asarray(clip.parents)
        R_dom = shell_radii(clip.local, parents)
        rt, tt = fold_types(clip.local, p, R_dom)
        m_p, L, bq, guard, t_enc = contract.budgets(clip.ref(p), B, p, margin)
        raw = clip.local.astype(np.float64)
        q_raw = quat_normalize(raw[:, :, :4])
        offsets = np.asarray(clip.offsets, dtype=np.float32)
        self._unfold_floor(clip.local, rt, tt, q_raw, raw, R_dom, L, guard, parents)
        cidx_rot, cint, cv, ctype = self._constants(clip.local, p, B, rt, tt, q_raw, raw, R_dom, offsets)
        tracks = [('rot', b) for b in range(B) if rt[b] == 2] + [('trans', b) for b in range(B) if tt[b] == 2]
        vals = {tr: log_map(q_raw[:, tr[1]]) if tr[0] == 'rot' else raw[:, tr[1], 4:7].copy() for tr in tracks}
        allK = np.arange(F, dtype=np.int64)
        S = Search(clip.local, parents, p, self.ALPHA * bq, guard)
        T = Tracks(S, vals)

        # stage 1: steps from the analytic start, the quarter-octave search, then the grid
        subs = subtree_lists(parents); depth = bone_depth(parents)
        chain = np.array([1 + int(depth[subs[b]].max()) for b in range(B)], dtype=np.float64)
        steps = {}
        for tr in tracks:
            bud = self.ALPHA * bq[tr[1]] / np.sqrt(chain[tr[1]])
            steps[tr] = np.float32(2.0 * bud / R_dom[tr[1]]) if tr[0] == 'rot' else np.float32(bud)
        loc0 = default_local(F, B, cv)
        for tr in tracks:
            ch = slice(0, 4) if tr[0] == 'rot' else slice(4, 7)
            loc0[:, tr[1], ch] = reconstruct(allK, quantize(vals[tr], steps[tr]), steps[tr], F, tr[0])
        S.reset(loc0)
        self._quant_search(S, T, tracks, steps, B, L, guard)
        idx = {tr: idx3(self.grid.floor(float(steps[tr]))) for tr in tracks}
        self._grid_search(S, T, tracks, idx)
        steps = {tr: self.grid.step(idx[tr]) for tr in tracks}
        S.refresh()

        # all-keys integers and the DPCM predictor costs
        X = {tr: quantize(vals[tr], steps[tr]) for tr in tracks}
        mode = {}; keys = {}; nks = {}; pid_all = {}; cost_all = {}; res_all = {}; res_nlms = {}

        def dpcm_costs(trs):
            if not trs:
                return
            Xall = np.concatenate([X[tr] for tr in trs], axis=1)
            cands = {0: fixed_residuals(Xall, 1), 1: fixed_residuals(Xall, 2), 2: nlms_residuals(Xall, Xall[0])}
            costs = np.zeros((3, 3 * len(trs)))
            for pid, R in cands.items():
                for c in range(3 * len(trs)):
                    costs[pid, c] = res_proxy(R[1:, c], self.C0)
            for j, tr in enumerate(trs):
                cols = slice(3 * j, 3 * j + 3)
                best = costs[:, cols].argmin(axis=0)
                pid_all[tr] = best
                cost_all[tr] = float(costs[:, cols].min(axis=0).sum()) + 3 * self.ID_BITS
                res_all[tr] = np.stack([cands[int(best[i])][:, 3 * j + i] for i in range(3)], axis=1)
                res_nlms[tr] = cands[2][:, cols].copy()
        dpcm_costs(tracks)
        for tr in tracks:
            mode[tr] = M_STATIC if bool(np.all(X[tr] == X[tr][0])) else M_ALL
            keys[tr] = allK

        # stage 2: the key ladder under the full caps and the clip total
        S.mlim = L.copy(); S.total = t_enc * B * m_p * F
        stats = dict(rounds=0, evals=0, n_acc=0)
        n_restored = 0
        kf_state = {}
        run_kf = F > 5

        def mode_decision(kf_state):
            nonlocal n_restored
            for tr in list(kf_state):
                K = kf_state[tr]['K']
                if len(K) == F:
                    continue
                nk = quantize(vals[tr][K], steps[tr])
                if kf_cost(K, nk, self.C0) >= cost_all[tr]:
                    if S.try_local(tr[1], T.local(tr, allK, steps[tr]), tag=(tr, 'restore')):
                        S.commit_pending(); n_restored += 1
                        mode[tr] = M_ALL; keys[tr] = allK; nks.pop(tr, None); kf_state.pop(tr)
                        continue
                    S.drop_pending()
                mode[tr] = M_KF; keys[tr] = K; nks[tr] = nk

        grow = dict(idx=idx, steps=steps, X=X, recost=dpcm_costs)
        if run_kf:
            kf_state = ladder(T, [tr for tr in tracks if mode[tr] != M_STATIC], self.grid, grow, m_p, stats, sizes=self.grow_sizes, gbits=self.GBITS)
            mode_decision(kf_state)
        S.esum = S.err.sum(axis=0)
        n_regrow = self._regrow_all(S, T, [tr for tr in tracks if mode[tr] == M_ALL], idx, steps, X, vals)
        if n_regrow:
            dpcm_costs([tr for tr in tracks if mode[tr] == M_ALL])
        if run_kf and kf_state and (n_restored or n_regrow) and S.total - float(S.esum.sum()) > 1e-3 * S.total:
            for st in kf_state.values():
                st['dirty'][:] = True; st['slope'][:] = np.inf
            ladder(T, list(kf_state), self.grid, grow, m_p, stats, state=kf_state, sizes=self.grow_sizes, gbits=self.GBITS)
            mode_decision(kf_state)
            S.esum = S.err.sum(axis=0)
        if run_kf:
            component_stage(T, tracks, mode, keys, nks, steps, m_p, self.C0, stats)
        if self._repair(S, T, tracks, mode, keys, nks, idx, steps, X, vals, L, guard, t_enc * m_p, parents):
            dpcm_costs([tr for tr in tracks if mode[tr] == M_ALL])

        # the predictors of the keyframe / per-component tracks
        pid_kf = {}; res_kf = {}; pid_ck = {}; res_ck = {}
        kf_tracks = [tr for tr in tracks if mode[tr] == M_KF]
        for tr, rn in zip(kf_tracks, nlms_batch([nks[tr] for tr in kf_tracks])):
            cands = {pid: key_residuals(nks[tr], pid, keys[tr]) for pid in range(3)}
            cands[3] = rn
            cost = {pid: res_proxy(r[1:], self.C0) for pid, r in cands.items()}
            pid_kf[tr] = min(cost, key=cost.get)
            res_kf[tr] = cands[pid_kf[tr]]
        for tr in tracks:
            if mode[tr] == M_CK:
                pid_ck[tr], res_ck[tr] = ck_finish(keys[tr], nks[tr], self.C0)
        rec = dict(F=F, B=B, p=p, fps=float(clip.fps), parents=parents, rt=rt, tt=tt, ctype=ctype, cidx_rot=cidx_rot, cint=cint,
                   tracks=tracks, idx=idx, X=X, mode=mode, keys=keys, nks=nks, pid_all=pid_all, pid_kf=pid_kf, pid_ck=pid_ck,
                   res_all=res_all, res_nlms=res_nlms, res_kf=res_kf, res_ck=res_ck)
        self.last_state = S.cur_local
        return self.pack(rec, streams)

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _unfold_floor(local, rt, tt, q_raw, raw, R_dom, L, guard, parents, max_unfold=64):
        """Un-fold folded tracks (greedily by their folding error, on the root paths of the violators) until the floor
        -- constants as sampled, animated tracks unquantized -- keeps every bone within its mean cap and guard."""
        F, B = raw.shape[:2]
        ident = np.eye(3)[None]

        def fold_err(b, kind):
            if kind == 'rot':
                cols = quat_cols(q_raw[:, b]); ref = ident if rt[b] == 0 else cols[:1]
                return float(R_dom[b] * np.sqrt(np.max(np.sum((cols - ref) ** 2, axis=-1), axis=-1)).max())
            t_ref = 0.0 if tt[b] == 0 else raw[:1, b, 4:7]
            return float(np.linalg.norm(raw[:, b, 4:7] - t_ref, axis=-1).max())

        def floor_local():
            loc = np.zeros((F, B, 7), dtype=np.float32); loc[:, :, 3] = 1.0
            for b in range(B):
                if rt[b] == 1:
                    q0 = local[0, b, :4].astype(np.float32); loc[:, b, :4] = q0 * (np.float32(-1.0) if q0[3] < 0 else np.float32(1.0))
                elif rt[b] == 2:
                    loc[:, b, :4] = local[:, b, :4]
                if tt[b] == 1:
                    loc[:, b, 4:7] = local[0, b, 4:7]
                elif tt[b] == 2:
                    loc[:, b, 4:7] = local[:, b, 4:7]
            return loc

        for _ in range(max_unfold):
            err = shell_error(local, floor_local(), parents)
            viol = [c for c in range(B) if err[:, c].mean() > L[c] or err[:, c].max() > guard[c]]
            if not viol:
                return
            path = set()
            for c in viol:
                b = c
                while b >= 0:
                    path.add(b); b = int(parents[b])
            cands = [(fold_err(b, 'rot'), b, 'rot') for b in path if rt[b] != 2] + [(fold_err(b, 'trans'), b, 'trans') for b in path if tt[b] != 2]
            cands = [c for c in cands if c[0] > 0.0]
            if not cands:
                return
            _, b, kind = max(cands)
            if kind == 'rot':
                rt[b] = 2
            else:
                tt[b] = 2

    def _constants(self, local, p, B, rt, tt, q_raw, raw, R_dom, offsets):
        """Quantized constant tracks: (rotation grid indices, integers, decoded values, type symbols)."""
        cg = self.cgrid
        cidx_rot = {}; cint = {}; cv = {}; ctype = {}
        st_ct = cg.step(cg.floor(p * self.CFRAC / SQRT3))
        for b in range(B):
            if rt[b] == 1:
                i_c = cg.floor(2.0 * p * self.CFRAC / (SQRT3 * R_dom[b]))
                st_c = cg.step(i_c)
                n = const_rot_ints(q_raw[0, b], st_c)
                cidx_rot[b] = i_c; cint[('rot', b)] = n; ctype[('rot', b)] = T_ROT_CONST
                cv[('rot', b)] = const_rot(n, st_c)
            if tt[b] == 1:
                if np.array_equal(local[0, b, 4:7].astype(np.float32), offsets[b]):
                    cv[('trans', b)] = offsets[b].copy(); ctype[('trans', b)] = T_TR_OFF
                else:
                    n = quantize(raw[0, b, 4:7][None], st_ct)[0]
                    cint[('trans', b)] = n; cv[('trans', b)] = const_trans(n, st_ct); ctype[('trans', b)] = T_TR_CONST
        return cidx_rot, cint, cv, ctype

    @staticmethod
    def _quant_search(S, T, tracks, steps, B, L, guard):
        """Per track (leaf-to-root, then root-to-leaf) the largest step, in quarter octaves around the analytic start, whose
        all-keys reconstruction passes the subtree test; a track whose subtree fails even unquantized is kept at its
        start in the first pass and searched against relaxed caps in the second."""
        F = S.F
        allK = np.arange(F, dtype=np.int64)
        by_bone = {}
        for tr in tracks:
            by_bone.setdefault(tr[1], []).append(tr)
        for pass_no, order in enumerate((range(B - 1, -1, -1), range(B))):
            for b in order:
                for tr in by_bone.get(b, []):
                    s0 = float(steps[tr])

                    def cand(e, s0=s0):
                        return np.float32(s0 * QUARTER ** e)

                    def passes_at(e, tr=tr, b=b, cand=cand):
                        return S.try_local(b, T.local(tr, allK, cand(e)), tag=(tr, e))

                    at0 = passes_at(0)
                    if not at0:
                        unq = T.local(tr, allK, np.float32(1e-9))
                        if not S.try_local(b, unq, tag=(tr, 'raw')):
                            S.drop_pending()
                            if pass_no == 0:
                                S.force_local(b, T.local(tr, allK, cand(0)))
                                continue
                            S.relax_to(b, unq, L, guard)
                            at0 = passes_at(0)
                        else:
                            S.drop_pending()
                    e = bracket_search(passes_at, at0)
                    if S.pending_tag() == (b, (tr, e)):
                        S.commit_pending()
                    else:
                        S.drop_pending(); S.force_local(b, T.local(tr, allK, cand(max(e, 0))))
                        e = max(e, 0)
                    S.clear_relax()
                    steps[tr] = cand(e)

    def _grow_bracket(self, passes_at, lo):
        """The largest passing offset above lo: doubling through caps_up, bisection, then the lookahead (the error is a
        sawtooth in the step, so the points up to la_units past the bracket are tried too)."""
        hi = None
        for e in self.grid.caps_up():
            if e <= lo:
                continue
            if passes_at(e):
                lo = e
            else:
                hi = e; break
        if hi is not None:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if passes_at(mid):
                    lo = mid
                else:
                    hi = mid
            for e in range(lo + 2, lo + self.la_units + 1):
                if passes_at(e):
                    lo = e
        return lo

    def _grid_search(self, S, T, tracks, idx):
        """Every track onto the grid: from the snapped-down index try 0 and +1, then grow (bracket + lookahead) or shrink;
        leaf-to-root, then root-to-leaf (0 only). A track for which nothing passes is forced to its snapped index."""
        grid = self.grid
        cap_down = grid.caps_down()
        allK = np.arange(S.F, dtype=np.int64)
        by_bone = {}
        for tr in tracks:
            by_bone.setdefault(tr[1], []).append(tr)
        on_grid = set()
        for pass_no, order in enumerate((range(S.B - 1, -1, -1), range(S.B))):
            for b in order:
                for tr in by_bone.get(b, []):
                    base = np.maximum(idx[tr], grid.idx_min)
                    e_min = int(grid.idx_min - base.min())

                    def passes_at(e, tr=tr, b=b, base=base):
                        return S.try_local(b, T.local(tr, allK, grid.step(base + e)), tag=(tr, e))

                    lo = None
                    for e in ((0, 1) if pass_no == 0 else (0,)):
                        if passes_at(e):
                            lo = e; break
                    if lo is None:
                        for e in cap_down:
                            if e < e_min:
                                break
                            if passes_at(e):
                                lo = e; break
                        if lo is None:
                            S.drop_pending()
                            if tr not in on_grid:
                                S.force_local(b, T.local(tr, allK, grid.step(base))); on_grid.add(tr)
                                idx[tr] = base.copy()
                            continue
                    else:
                        lo = self._grow_bracket(passes_at, lo)
                    assert S.pending_tag() == (b, (tr, lo))
                    S.commit_pending(); idx[tr] = base + lo; on_grid.add(tr)

    def _regrow_all(self, S, T, all_tr, idx, steps, X, vals):
        """The all-keys tracks (leaf to root) grow their step as far as the final caps allow."""
        allK = np.arange(S.F, dtype=np.int64)
        n = 0
        for tr in sorted(all_tr, key=lambda t: -t[1]):
            b = tr[1]; base = idx[tr].copy()
            S.drop_pending()

            def passes_at(e, tr=tr, b=b, base=base):
                return S.try_local(b, T.local(tr, allK, self.grid.step(base + e)), tag=(tr, e))

            lo = self._grow_bracket(passes_at, 0)
            if lo:
                assert S.pending_tag() == (b, (tr, lo))
                S.commit_pending(); idx[tr] = base + lo; steps[tr] = self.grid.step(idx[tr]); n += 1
                X[tr] = quantize(vals[tr], steps[tr])
            else:
                S.drop_pending()
        return n

    def _repair(self, S, T, tracks, mode, keys, nks, idx, steps, X, vals, L, guard, m_p, parents, max_iter=200):
        """Only when the state misses a cap or the clip total: the worst bone's root path, bone by bone upwards, one track
        at a time -- a keyframe track gets every key back, an all-keys track one grid unit finer."""
        grid = self.grid
        F, B = S.F, S.B
        allK = np.arange(F, dtype=np.int64)
        by_bone = {}
        for tr in tracks:
            by_bone.setdefault(tr[1], []).append(tr)
        n = 0; last = None; stall = 0
        for _ in range(max_iter * grid.unit):
            S.esum = S.err.sum(axis=0); S.emax = S.err.max(axis=0)
            over = np.maximum(S.esum / (L * F), S.emax / guard)
            tot = float(S.esum.sum()) / (B * m_p * F)
            if over.max() <= 1 + 1e-9 and tot <= 1 + 1e-9:
                return n
            c = int(np.argmax(over)) if over.max() > 1 + 1e-9 else int(np.argmax(S.esum / F / m_p))
            score = (float(over.max()), tot)
            if last is not None and score >= last:
                stall += 1
                if stall >= 3:
                    return n
            else:
                stall = 0
            last = score
            changed = False
            b = c
            while b >= 0 and not changed:
                for tr in by_bone.get(b, []):
                    if mode[tr] == M_STATIC:
                        continue
                    if mode[tr] in (M_KF, M_CK):
                        mode[tr] = M_ALL; keys[tr] = allK; nks.pop(tr, None)
                    elif idx[tr].min() > grid.idx_min:
                        idx[tr] = idx[tr] - 1; steps[tr] = grid.step(idx[tr]); X[tr] = quantize(vals[tr], steps[tr])
                    else:
                        continue
                    S.force_local(b, T.local(tr, allK, steps[tr])); changed = True; n += 1
                    break
                b = int(parents[b])
            if not changed:
                return n
        return n

    # ============================================================================================================
    # packer
    # ============================================================================================================
    @staticmethod
    def _columns(rec, use_ids):
        """The coded curves in coder order (all-keys tracks, then keyframe / per-component tracks), one column per
        component: (track, component, residuals [n], integers [n + 1], predictor id, keyframe flag, gaps [n], grid index)."""
        mode, keys, nks, X, idx = rec['mode'], rec['keys'], rec['nks'], rec['X'], rec['idx']
        order = [tr for tr in rec['tracks'] if mode[tr] == M_ALL] + [tr for tr in rec['tracks'] if mode[tr] in (M_KF, M_CK)]
        cols = []
        for tr in order:
            if mode[tr] == M_ALL:
                R = rec['res_all'][tr] if use_ids else rec['res_nlms'][tr]
                pid = rec['pid_all'][tr] if use_ids else np.full(3, 2, np.int64)
                if R.shape[0] > 1:
                    g = np.zeros(R.shape[0] - 1, np.int64)
                    for c in range(3):
                        cols.append((tr, c, np.asarray(R[1:, c], np.int64), np.asarray(X[tr][:, c], np.int64), int(pid[c]), 0, g, int(idx[tr][c])))
            elif mode[tr] == M_KF:
                R = rec['res_kf'][tr]
                if R.shape[0] > 1:
                    g = np.diff(keys[tr])
                    for c in range(3):
                        cols.append((tr, c, np.asarray(R[1:, c], np.int64), np.asarray(nks[tr][:, c], np.int64), int(rec['pid_kf'][tr]), 1, g,
                                     int(idx[tr][c])))
            else:
                for c in range(3):
                    r = np.asarray(rec['res_ck'][tr][c], np.int64)
                    if len(r) > 1:
                        cols.append((tr, c, r[1:], np.asarray(nks[tr][c], np.int64), int(rec['pid_ck'][tr][c]), 1, np.diff(keys[tr][c]),
                                     int(idx[tr][c])))
        return cols

    def _curve_state(self, kinds, bones, n, pid, kf, idx_c, gaps, depth, fps, p):
        dc = depth_class(depth[np.asarray(bones, np.int64)]) if len(bones) else np.zeros(0, np.int64)
        step_q8 = (256 * np.asarray(idx_c, np.int64)) // self.G
        return CurveState(kinds, dc, kf, pid, step_q8, int(round(fps)), prec_index(p), n, gaps)

    def pack(self, rec, streams=None):
        """The blob of an encoder state (container.py has the layout). `streams` (a dict) receives per coded curve what the
        model's features are computed from and the bits the residuals cost: n, kind, dc, kf, pid, step_q8, x0 [C], fps_i,
        p_idx, nmax, and curve-major r (residuals), dx (x_{t+1} - x_t), g (key gap before the residual, 0 for all-keys
        curves), bits (code length of every residual: symbol + raw low bits); header_bits (the adaptive symbols), raw_bits
        (every raw bit, the residuals' low bits included), prefix_bytes."""
        grid, cg = self.grid, self.cgrid
        F, B, p = rec['F'], rec['B'], rec['p']
        rt, tt, ctype = rec['rt'], rec['tt'], rec['ctype']
        tracks, mode, keys, X, idx = rec['tracks'], rec['mode'], rec['keys'], rec['X'], rec['idx']
        sink = Sink(); raw = sink.raw
        for b in range(B):
            sink.put(T_ROT_ANIM if rt[b] == 2 else (T_ROT_DEF if rt[b] == 0 else ctype[('rot', b)]), CT_TYPE_ROT)
        for b in range(B):
            sink.put(T_TR_ANIM if tt[b] == 2 else (T_TR_DEF if tt[b] == 0 else ctype[('trans', b)]), CT_TYPE_TRANS)
        for kind in ('rot', 'trans'):
            prev = grid.first[kind]
            for tr in tracks:
                if tr[0] == kind:
                    i0 = int(idx[tr][0])
                    raw.write_exp_golomb(int(zigzag(np.int64(i0 - prev))), grid.eg_k); prev = i0
        prev = cg.first['crot']
        for b in range(B):
            if rt[b] == 1:
                raw.write_exp_golomb(int(zigzag(np.int64(rec['cidx_rot'][b] - prev))), cg.eg_k); prev = rec['cidx_rot'][b]
        for b in range(B):
            if rt[b] == 1 and ctype[('rot', b)] == T_ROT_CONST:
                sink.put_mags(rec['cint'][('rot', b)], CT_X0_ROT)
        for b in range(B):
            if tt[b] == 1 and ctype[('trans', b)] == T_TR_CONST:
                sink.put_mags(rec['cint'][('trans', b)], CT_X0_TRANS)
        for tr in tracks:
            sink.put_mags(X[tr][0], CT_X0_ROT if tr[0] == 'rot' else CT_X0_TRANS)
        for tr in tracks:
            sink.put(mode[tr], CT_MODE)
        all_tracks = [tr for tr in tracks if mode[tr] == M_ALL]
        use_ids = False
        if all_tracks:                                      # per-curve ids only when they pay for themselves
            ids = np.concatenate([rec['pid_all'][tr] for tr in all_tracks])
            with_ids = sum(res_proxy(rec['res_all'][tr][1:], self.C0) for tr in all_tracks) + cls_cost(ids, 0.0) + 2.0
            nlms_only = sum(res_proxy(rec['res_nlms'][tr][1:], self.C0) for tr in all_tracks)
            use_ids = with_ids < nlms_only
            if use_ids:
                for tr in all_tracks:
                    sink.put_many(rec['pid_all'][tr], [CT_PID_ALL] * 3)
        for tr in tracks:
            if mode[tr] == M_KF:
                sink.put(rec['pid_kf'][tr], CT_PID_KF)
            elif mode[tr] == M_CK:
                sink.put_many(rec['pid_ck'][tr], [CT_PID_KF] * 3)
        for tr in tracks:
            if mode[tr] in (M_KF, M_CK):
                K = keys[tr] if mode[tr] == M_KF else np.unique(np.concatenate(keys[tr]))
                sink.put_gaps(K, tr[0])
                if mode[tr] == M_CK:
                    sink.put_flags(keys[tr], K)
        starts, freqs = model_forward(sink.sym, sink.ctx, NSYM, self.init)
        res_start, res_freq = self._pack_residuals(rec, self._columns(rec, use_ids), raw, streams)
        rans = rans_encode(np.concatenate([np.asarray(starts, np.int64), res_start]), np.concatenate([np.asarray(freqs, np.int64), res_freq]))
        rawb = raw.getvalue()
        prefix = MAGIC + bytes([VERSION, int(use_ids)]) + leb(F) + leb(B) + prec_code(p) + leb(len(rans)) + leb(len(rawb))
        if streams is not None:                             # the blob's budget: prefix + rANS (symbol bits + flush) + raw bits
            streams.update(header_bits=float((15.0 - np.log2(np.asarray(freqs, np.float64))).sum()), raw_bits=int(raw.bit_length),
                           prefix_bytes=len(prefix))
        return prefix + rans + rawb

    def _pack_residuals(self, rec, cols, raw, streams=None):
        """Every residual of the columns under the model, time-major: (start, freq) of its symbol; the low bits to `raw`."""
        if not cols:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        depth = bone_depth(rec['parents'])
        C = len(cols)
        n = np.array([len(c[2]) for c in cols], np.int64); n_max = int(n.max())
        G = np.zeros((n_max, C), np.int64); R = np.zeros((n_max, C), np.int64); Xl = np.zeros((n_max + 1, C), np.int64)
        for j, (tr, c, r, x, pj, kfj, g, ij) in enumerate(cols):
            R[:n[j], j] = r; Xl[:n[j] + 1, j] = x[:n[j] + 1]
            if kfj:
                G[:len(g), j] = g
        st = self._curve_state([0 if c[0][0] == 'rot' else 1 for c in cols], [c[0][1] for c in cols], n, [c[4] for c in cols],
                               [c[5] for c in cols], [c[7] for c in cols], G, depth, rec['fps'], rec['p'])
        st.start(Xl[0])
        model = self.model
        model.start(C, PAD)
        fb = _fast.tf_pack(model, st, n, n_max, G, R, Xl, PAD)
        if fb is None:
            Ws, Ms, Ss, ys = [], [], [], []
            for t in range(n_max):
                act = np.flatnonzero(n > t)
                lw, mu, ls = model.step(st.features(), act)
                Ws.append(lw[act]); Ms.append(mu[act]); Ss.append(ls[act]); ys.append(R[t, act])
                st.update(R[t], Xl[t + 1])
            fb = (np.concatenate(Ws), np.concatenate(Ms), np.concatenate(Ss), np.concatenate(ys))
        Lw, Mu, Ls, y = fb[:4]
        sym = symbol_index(y)
        wd = WIDTHS[sym]
        low = np.abs(y) & ((np.int64(1) << wd) - 1)
        N = len(y); start = np.zeros(N, np.int64); freq = np.zeros(N, np.int64)
        CH = 65536
        for i0 in range(0, N, CH):
            f = symbol_freqs(Lw[i0:i0 + CH], Mu[i0:i0 + CH], Ls[i0:i0 + CH], ALPHABET)
            cum = np.cumsum(f, axis=1) - f
            sl = np.arange(len(f))
            start[i0:i0 + CH] = cum[sl, sym[i0:i0 + CH]]; freq[i0:i0 + CH] = f[sl, sym[i0:i0 + CH]]
        for v, w in zip(low.tolist(), wd.tolist()):
            if w:
                raw.write(v, w)
        if streams is not None:
            col = np.concatenate([np.flatnonzero(n > t) for t in range(n_max)])          # the column of every symbol
            bits = 15.0 - np.log2(freq.astype(np.float64)) + wd
            order = np.argsort(col, kind='stable')                                       # time-major -> curve-major
            tt = np.concatenate([np.arange(k, dtype=np.int64) for k in n.tolist()])
            jj = np.repeat(np.arange(C, dtype=np.int64), n)
            streams.update(n=n, nmax=n_max, kind=st.base[:, 25] // 256, dc=np.argmax(st.base[:, 21:25], axis=1),
                           kf=st.base[:, 26] // 256, pid=np.array([c[4] for c in cols], np.int64), step_q8=st.base[:, STEP_COL].copy(),
                           x0=Xl[0].copy(), fps_i=int(round(rec['fps'])), p_idx=prec_index(rec['p']),
                           r=R[tt, jj], dx=Xl[tt + 1, jj] - Xl[tt, jj], g=G[tt, jj], bits=bits[order])
        return start, freq

    # ============================================================================================================
    # decoder
    # ============================================================================================================
    def decode(self, blob, parents, offsets, fps):
        """blob + skeleton (parents [B], rest offsets [B, 3] in cm) + frame rate -> local transforms float32 [F, B, 7]."""
        r = _fast.decode(self, blob, parents, offsets, fps)
        if r is not None:
            return r
        grid, cg = self.grid, self.cgrid
        if blob[:2] != MAGIC or blob[2] != VERSION:
            raise ValueError('not a CurveCodec v1 blob')
        flags = blob[3]; pos = 4
        F, pos = read_leb(blob, pos); B, pos = read_leb(blob, pos)
        p, pos = read_prec(blob, pos)
        n_rans, pos = read_leb(blob, pos); n_raw, pos = read_leb(blob, pos)
        rans = blob[pos:pos + n_rans]; pos += n_rans
        rawb = bytes(blob[pos:pos + n_raw])
        if flags & ~1:
            raise ValueError(f'unsupported blob flags {flags}')
        use_ids = bool(flags & 1)
        dec = RansDecoder(rans, self.init, NSYM); rd = BitReader(rawb)
        parents = np.asarray(parents); depth = bone_depth(parents)
        offsets = np.asarray(offsets, dtype=np.float32)
        rts = [dec.decode(CT_TYPE_ROT) for _ in range(B)]
        tts = [dec.decode(CT_TYPE_TRANS) for _ in range(B)]
        if any(t not in (T_ROT_DEF, T_ROT_CONST, T_ROT_ANIM) for t in rts) or any(t not in (T_TR_DEF, T_TR_OFF, T_TR_CONST, T_TR_ANIM) for t in tts):
            raise ValueError('unsupported track type')
        tracks = [('rot', b) for b in range(B) if rts[b] == T_ROT_ANIM] + [('trans', b) for b in range(B) if tts[b] == T_TR_ANIM]
        idxd = {}
        for kind in ('rot', 'trans'):
            prev = grid.first[kind]
            for tr in tracks:
                if tr[0] == kind:
                    prev += int(unzigzag(np.int64(rd.read_exp_golomb(grid.eg_k))))
                    idxd[tr] = idx3(prev)
        cidx = {}; prev = cg.first['crot']
        for b in range(B):
            if rts[b] == T_ROT_CONST:
                prev += int(unzigzag(np.int64(rd.read_exp_golomb(cg.eg_k)))); cidx[b] = prev
        st_ct = cg.step(cg.floor(p * self.CFRAC / SQRT3))
        cv = {}
        for b in range(B):
            if rts[b] == T_ROT_CONST:
                cv[('rot', b)] = const_rot(get_mags(dec, rd, 3, CT_X0_ROT), cg.step(cidx[b]))
        for b in range(B):
            if tts[b] == T_TR_CONST:
                cv[('trans', b)] = const_trans(get_mags(dec, rd, 3, CT_X0_TRANS), st_ct)
            elif tts[b] == T_TR_OFF:
                cv[('trans', b)] = offsets[b].copy()
        x0 = {tr: get_mags(dec, rd, 3, CT_X0_ROT if tr[0] == 'rot' else CT_X0_TRANS) for tr in tracks}
        mode = {tr: dec.decode(CT_MODE) for tr in tracks}
        pid = {}
        for tr in tracks:
            if mode[tr] == M_ALL:
                pid[tr] = [dec.decode(CT_PID_ALL) for _ in range(3)] if use_ids else [2, 2, 2]
        for tr in tracks:
            if mode[tr] == M_KF:
                pid[tr] = [dec.decode(CT_PID_KF)] * 3
            elif mode[tr] == M_CK:
                pid[tr] = [dec.decode(CT_PID_KF) for _ in range(3)]
        keys = {}
        allK = np.arange(F, dtype=np.int64)
        for tr in tracks:
            if mode[tr] in (M_KF, M_CK):
                K = [0]; pc = 0
                kbase = CT_GAP + (0 if tr[0] == 'rot' else NGAP)
                while K[-1] < F - 1:
                    s = dec.decode(kbase + pc)
                    gm = s if s < ESC else ESC + rd.read_exp_golomb(0)
                    K.append(K[-1] + gm + 1)
                    pc = 1 + min(int(gm).bit_length(), NGAP - 2)
                U = np.asarray(K, dtype=np.int64)
                if mode[tr] == M_KF:
                    keys[tr] = U
                else:
                    pres = np.ones((len(U), 3), dtype=bool)
                    prev = [1, 1, 1]
                    for j in range(1, len(U) - 1):
                        f0 = dec.decode(CT_FLAG + prev[0])
                        f1 = dec.decode(CT_FLAG + 4 + prev[1] * 2 + f0)
                        f2 = 1 if (f0 == 0 and f1 == 0) else dec.decode(CT_FLAG + 12 + prev[2] * 4 + f0 * 2 + f1)
                        pres[j] = (f0, f1, f2); prev = [f0, f1, f2]
                    keys[tr] = [U[pres[:, c]] for c in range(3)]
            elif mode[tr] == M_ALL:
                keys[tr] = allK
            else:
                keys[tr] = np.array([0, F - 1], np.int64) if F >= 2 else np.array([0], np.int64)
        # the coded columns: (track, component, n, predictor id, keyframe flag, gaps)
        order = [tr for tr in tracks if mode[tr] == M_ALL] + [tr for tr in tracks if mode[tr] in (M_KF, M_CK)]
        cols = []
        for tr in order:
            for c in range(3):
                Kc = keys[tr][c] if mode[tr] == M_CK else keys[tr]
                if len(Kc) > 1:
                    cols.append((tr, c, len(Kc) - 1, pid[tr][c], int(mode[tr] != M_ALL), np.diff(Kc)))
        Xcol = self._decode_residuals(dec, rd, cols, x0, idxd, depth, fps, p) if cols else {}
        loc = default_local(F, B, cv)
        for tr in tracks:
            kind, b = tr
            step = grid.step(idxd[tr])
            if mode[tr] == M_STATIC:
                r = reconstruct(keys[tr], np.repeat(x0[tr][None], len(keys[tr]), axis=0), step, F, kind)
            elif mode[tr] == M_CK:
                r = reconstruct_c(keys[tr], [Xcol.get((tr, c), np.array([x0[tr][c]], np.int64)) for c in range(3)], step, F, kind)
            elif (tr, 0) in Xcol:
                r = reconstruct(keys[tr], np.stack([Xcol[(tr, c)] for c in range(3)], axis=1), step, F, kind)
            else:
                r = reconstruct(keys[tr], x0[tr][None].copy(), step, F, kind)
            if kind == 'rot':
                loc[:, b, :4] = r
            else:
                loc[:, b, 4:7] = r
        return loc

    def _decode_residuals(self, dec, rd, cols, x0, idxd, depth, fps, p):
        """The lockstep residual loop: per step the model on every active column, one symbol + low bits per column, the
        predictors -> {(track, component): integers [n + 1]}."""
        lo, hi = ALPHABET
        C = len(cols)
        n = np.array([c[2] for c in cols], np.int64); n_max = int(n.max())
        G = np.zeros((n_max, C), np.int64)
        x0v = np.array([int(x0[c[0]][c[1]]) for c in cols], np.int64)
        for j, c in enumerate(cols):
            if c[4]:
                G[:len(c[5]), j] = c[5]
        pid = np.array([c[3] for c in cols], np.int64); kf = np.array([c[4] for c in cols], np.int64)
        st = self._curve_state([0 if c[0][0] == 'rot' else 1 for c in cols], [c[0][1] for c in cols], n, pid, kf,
                               [idxd[c[0]][c[1]] for c in cols], G, depth, fps, p)
        st.start(x0v)
        model = self.model
        model.start(C, PAD)
        pred = IncPred(pid, kf, x0v, G)
        Xl = np.zeros((n_max + 1, C), np.int64); Xl[0] = x0v
        for t in range(n_max):
            act = np.flatnonzero(n > t)
            lw, mu, ls = model.step(st.features(), act)
            f = symbol_freqs(lw[act], mu[act], ls[act], ALPHABET)
            cum = np.cumsum(f, axis=1)
            r_t = np.zeros(C, np.int64)
            for i, c in enumerate(act.tolist()):
                s = dec.decode_freqs(f[i], cum[i])
                w = int(WIDTHS[s])
                r_t[c] = symbol_value(s, rd.read(w) if w else 0, lo, hi)
            P = pred.predict()
            x_new = np.where(n > t, P + r_t, pred.x)
            Xl[t + 1] = x_new
            pred.update(P, x_new)
            st.update(r_t, x_new)
        return {(c[0], c[1]): Xl[:c[2] + 1, j].copy() for j, c in enumerate(cols)}


def compress(codec, clip, precision, contract=None, margins=None):
    """Encode under the contract, falling back to the margins 0.98 / 0.95 / 0.90 when the decoded clip misses it ->
    dict(blob, margin, ok, reasons, stats, ratios, decoded). When every margin misses, the last attempt is returned."""
    from .contract import MARGINS
    from .metric import error_stats
    contract = contract or Contract()
    p = float(precision)
    for m in (margins or MARGINS):
        blob = codec.encode(clip, p, contract, m)
        dec = codec.decode(blob, clip.parents, clip.offsets, clip.fps)
        st = error_stats(shell_error(clip.local, dec, clip.parents), p)
        ok, reasons, ratios = contract.check(st, clip.ref(p), p)
        if ok:
            break
    return dict(blob=blob, margin=m, ok=ok, reasons=reasons, stats=st, ratios=ratios, decoded=dec)
