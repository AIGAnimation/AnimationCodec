"""Key selection: which frames of an animated track are coded, and the interpolation that fills the frames in between.

A track is decoded from its keys by Catmull-Rom (cubic Hermite with central-difference tangents, linear with two keys)
of the dequantized values, rotations in the log-map domain (then exp_map). Keys are removed by a rate-distortion ladder
over the whole clip: a candidate key's cost D is the error its removal adds over the affected frames and the bone's
subtree (exact, in object space), its benefit is a proxy of its coded bits; a clip-wide lambda (m_p x 1e-4 per bit,
x sqrt(2) per level) accepts per track, in slope order, every candidate with D / bits <= lambda whose D fits every
subtree bone's remaining mean cap and the clip's remaining total; a removal pushing a bone over its guard is refused.
At every level each track is also offered a coarser grid step (`grow_step`) on the same slope test.

Per-component stage: the keys of a keyframe track are then thinned per component (x, y, z own key sets drawn from the
track's set) with the same ladder; the track keeps the cheapest of {track set, per-component sets, their union} by a
coded-cost proxy.
"""
import numpy as np

from . import fast as _fast
from .container import FLAG_BITS, M_CK, M_KF, gap_proxy, res_proxy
from .predict import key_residuals, nlms_residuals
from .quantize import dequantize, quantize
from .rot import exp_map
from .search import eval_subtree_frames

REACH = 2                   # keys on each side whose interpolation a removal changes
LAM0 = 1e-4                 # first lambda, in units of m_p per bit
LAM_RATIO = 2.0 ** 0.5
MAX_LEVELS = 120
GAP_BITS = 1.5              # bit proxy of one gap symbol


# ----------------------------------------------------------------------------------------------------------------
# interpolation
# ----------------------------------------------------------------------------------------------------------------
def interp_vectors(K, vk, F):
    """K int64 [n] sorted (K[0] = 0, K[-1] = F - 1), vk float32 [n, d] -> float64 [F, d], exact at the keys."""
    r = _fast.interp_vectors(K, vk, F)
    if r is not None:
        return r
    vk = vk.astype(np.float64)
    if len(K) == F:
        return vk.copy()
    f = np.arange(F)
    j = np.clip(np.searchsorted(K, f, side='right') - 1, 0, len(K) - 2)
    t = (f - K[j]) / (K[j + 1] - K[j]).astype(np.float64)
    a, b = vk[j], vk[j + 1]
    if len(K) >= 3:
        tk = K.astype(np.float64)
        m = np.empty_like(vk)
        m[1:-1] = (vk[2:] - vk[:-2]) / (tk[2:] - tk[:-2])[:, None]
        m[0] = (vk[1] - vk[0]) / (tk[1] - tk[0]); m[-1] = (vk[-1] - vk[-2]) / (tk[-1] - tk[-2])
        h = (tk[j + 1] - tk[j])[:, None]
        t2 = (t * t)[:, None]; t3 = (t2[:, 0] * t)[:, None]; tt = t[:, None]
        out = ((2 * t3 - 3 * t2 + 1) * a + (t3 - 2 * t2 + tt) * h * m[j] + (-2 * t3 + 3 * t2) * b + (t3 - t2) * h * m[j + 1])
    else:
        out = a + t[:, None] * (b - a)
    out[K] = vk
    return out


def reconstruct(K, nk, step, F, kind):
    """A track decoded from its keys K and key integers nk [n, 3]: rotation float32 [F, 4], translation float32 [F, 3]."""
    r = _fast.reconstruct(K, nk, step, F, kind)
    if r is not None:
        return r
    v = interp_vectors(K, dequantize(nk, step), F)
    return (exp_map(v) if kind == 'rot' else v).astype(np.float32)


def reconstruct_c(Ks, nks, step, F, kind):
    """A track decoded from per-component keys Ks = [K0, K1, K2] and integers nks = [n0, n1, n2] (step float32 [3])."""
    v = np.empty((F, 3), dtype=np.float64)
    for c in range(3):
        vk = dequantize(np.asarray(nks[c], dtype=np.int64)[:, None], step[c])
        v[:, c] = interp_vectors(np.asarray(Ks[c], dtype=np.int64), vk, F)[:, 0]
    return (exp_map(v) if kind == 'rot' else v).astype(np.float32)


def windows(K, pos, reach=REACH):
    """Frames strictly between K[pos - reach] and K[pos + reach] for every candidate -> (frames, candidate id per frame)."""
    lo = K[pos - reach] + 1; hi = K[pos + reach]
    lens = hi - lo
    starts = np.cumsum(lens) - lens
    idx = np.arange(int(lens.sum())) - np.repeat(starts, lens) + np.repeat(lo, lens)
    return idx, np.repeat(np.arange(len(pos)), lens)


class Tracks:
    """The animated tracks under search: candidate key sets -> the local transform of the bone (the other channel of the
    bone from the search state)."""

    def __init__(self, S, vals):
        self.S = S; self.vals = vals; self.F = S.F

    def local_keys(self, tr, K, nk, step):
        kind, b = tr
        loc = np.array(self.S.cur_local[:, b], dtype=np.float32)
        if kind == 'rot':
            loc[:, :4] = reconstruct(K, nk, step, self.F, 'rot')
        else:
            loc[:, 4:7] = reconstruct(K, nk, step, self.F, 'trans')
        return loc

    def local(self, tr, K, step):
        return self.local_keys(tr, K, quantize(self.vals[tr][K], step), step)

    def local_c(self, tr, Ks, nks, step):
        kind, b = tr
        loc = np.array(self.S.cur_local[:, b], dtype=np.float32)
        r = reconstruct_c(Ks, nks, step, self.F, kind)
        if kind == 'rot':
            loc[:, :4] = r
        else:
            loc[:, 4:7] = r
        return loc


# ----------------------------------------------------------------------------------------------------------------
# track keys: RD removal, step growth, the ladder
# ----------------------------------------------------------------------------------------------------------------
def key_bits(nk, K):
    """Bit proxy per key: sum over the components of log2(1 + 2 |d2t residual|) + 1.5 for the gap symbol."""
    r = key_residuals(nk, 2, K)
    return np.sum(np.log2(1.0 + 2.0 * np.abs(r).astype(np.float64)), axis=1) + GAP_BITS


def _accept(S, sub, D, bad, sl, lam, slope, pos):
    """Candidates in slope order: accept while slope <= lam and D fits the subtree caps and the clip total."""
    rem_sub = S.mlim[sub] * S.F - S.esum[sub]; rem_tot = S.total - float(S.esum.sum())
    acc = np.zeros(len(sl), dtype=bool)
    for w in np.argsort(sl, kind='stable').tolist():
        if bad[w]:
            continue
        if sl[w] > lam:
            break
        d = D[w]
        if np.all(d <= rem_sub + 1e-12) and d.sum() <= rem_tot + 1e-12:
            acc[w] = True; rem_sub = rem_sub - d; rem_tot -= d.sum()
        else:
            slope[pos[w]] = np.inf                       # blocked by a cap: re-opened when a neighbour changes
    return acc


def _candidates(dirty, slope, lam, n):
    """Eligible interior keys (dirty, or cached slope <= lam), spaced 2 x REACH apart."""
    elig = np.flatnonzero(dirty | (slope <= lam))
    elig = elig[(elig >= REACH) & (elig < n - REACH)]
    pos = []; last = -2 * REACH
    for e in elig.tolist():
        if e - last >= 2 * REACH:
            pos.append(e); last = e
    return np.asarray(pos, dtype=np.int64)


def _window_cost(S, b, sub, loc2, idx, wid, n_cand):
    """Errors of the candidates' windows -> (rows, E [frames, nsub], D [n_cand, nsub], bad [n_cand])."""
    t_rot, t_pos, t_err = eval_subtree_frames(S, b, loc2, idx)
    E = np.stack([t_err[c] for c in sub], axis=1)
    dE = E - S.err[np.ix_(idx, sub)]
    D = np.empty((n_cand, len(sub)))
    for j in range(len(sub)):
        D[:, j] = np.bincount(wid, dE[:, j], minlength=n_cand)
    viol = (E > S.limits[sub][None, :]).any(axis=1)
    bad = np.bincount(wid, viol.astype(np.float64), minlength=n_cand) > 0
    return (t_rot, t_pos, t_err), E, D, bad


def _commit_windows(S, b, sub, loc2, rows, idx, wid, acc, D, E):
    sel = acc[wid]; fidx = idx[sel]
    t_rot, t_pos, t_err = rows
    S.cur_local[fidx, b] = loc2[fidx]
    for c in sub.tolist():
        S.obj_rot[fidx, c] = t_rot[c][sel]; S.obj_pos[fidx, c] = t_pos[c][sel]; S.err[fidx, c] = t_err[c][sel]
    S.esum[sub] += D[acc].sum(axis=0)
    S.emax[sub] = np.maximum(S.emax[sub], E[sel].max(axis=0))
    return fidx


def remove_keys(tracks, tr, st, step, lam, stats):
    """RD removal of track tr's keys at one lambda (state st: K, dirty, slope; modified in place)."""
    S = tracks.S
    if _fast.remove_keys(S, tr, st, step, tracks.vals[tr], lam, REACH, stats):
        return
    b = tr[1]
    v = tracks.vals[tr]
    sub = np.asarray(S.subtrees[b])
    K, dirty, slope = st['K'], st['dirty'], st['slope']
    n_rounds = 0
    while len(K) > 2 * REACH:
        pos = _candidates(dirty, slope, lam, len(K))
        if len(pos) == 0:
            break
        n_rounds += 1
        keep = np.ones(len(K), dtype=bool); keep[pos] = False
        K2 = K[keep]
        idx, wid = windows(K, pos)
        loc2 = tracks.local_keys(tr, K2, quantize(v[K2], step), step)
        rows, E, D, bad = _window_cost(S, b, sub, loc2, idx, wid, len(pos))
        stats['evals'] += 1
        bits = key_bits(quantize(v[K], step), K)[pos]
        sl = D.sum(axis=1) / np.maximum(bits, 0.5)
        slope[pos] = sl; dirty[pos] = False
        slope[pos[bad]] = np.inf
        acc = _accept(S, sub, D, bad, sl, lam, slope, pos)
        if acc.any():
            _commit_windows(S, b, sub, loc2, rows, idx, wid, acc, D, E)
            stats['n_acc'] += int(acc.sum())
            for r in range(1, REACH + 1):
                dirty[np.clip(pos[acc] - r, 0, len(K) - 1)] = True; dirty[np.clip(pos[acc] + r, 0, len(K) - 1)] = True
            keep2 = np.ones(len(K), dtype=bool); keep2[pos[acc]] = False
            K = K[keep2]; dirty = dirty[keep2]; slope = slope[keep2]
            st['K'], st['dirty'], st['slope'] = K, dirty, slope; st['gdirty'] = True
    stats['rounds'] += n_rounds


def grow_step(tracks, tr, st, grid, grow, lam, sizes, gbits, max_offers=16):
    """Offer the track a coarser grid step (+1, 2, 4, 8 grid units: the first of the sizes that passes the caps and the
    guard is the offer), slope = D / bits saved (3 n_keys size / G x gbits); repeated while accepted."""
    S = tracks.S
    if _fast.grow_step(S, tr, st, tracks.vals[tr], grid, grow, lam, sizes, gbits, max_offers):
        return
    b = tr[1]
    v = tracks.vals[tr]
    sub = np.asarray(S.subtrees[b])
    K = st['K']
    idx, steps = grow['idx'], grow['steps']
    for _ in range(max_offers):
        if not (st['gdirty'] or st['gslope'] <= lam):
            return
        best = None
        st['gdirty'] = False; st['gslope'] = np.inf
        for size in sizes:
            i2 = idx[tr] + size
            step2 = grid.step(i2)
            loc2 = tracks.local(tr, K, step2)
            ok, t_rot, t_pos, t_err, cnt = S.eval_subtree(b, loc2, early_exit=False)
            D = np.array([float(t_err[c].sum()) - S.esum[c] for c in sub.tolist()])
            bits = size * 3.0 * len(K) / grid.G * gbits
            sl = float(D.sum() / bits)
            bad = any(float(t_err[c].max()) > S.limits[c] for c in sub.tolist())
            rem_sub = S.mlim[sub] * S.F - S.esum[sub]; rem_tot = S.total - float(S.esum.sum())
            fits = (not bad) and bool(np.all(D <= rem_sub + 1e-12)) and D.sum() <= rem_tot + 1e-12
            if bad or not fits:
                continue
            st['gslope'] = min(st['gslope'], sl)
            if sl <= lam:
                best = (size, i2, step2, loc2, t_rot, t_pos, t_err, cnt)
                break
        if best is None:
            return
        size, i2, step2, loc2, t_rot, t_pos, t_err, cnt = best
        S._pending = (b, (tr, 'grow'), loc2, t_rot, t_pos, t_err, cnt)
        S.commit_pending()
        idx[tr] = i2; steps[tr] = step2; grow['X'][tr] = quantize(v, step2)
        grow['recost']([tr])
        st['gdirty'] = True


def ladder(tracks, kf_tracks, grid, grow, m_p, stats, state=None, sizes=(1, 2, 4, 8), gbits=2.5):
    """The clip-wide lambda ladder over the tracks (key removal + step growth) -> {track: state}."""
    S = tracks.S; F = S.F
    if state is None:
        state = {tr: dict(K=np.arange(F, dtype=np.int64), dirty=np.ones(F, dtype=bool), slope=np.full(F, np.inf)) for tr in kf_tracks}
    for st in state.values():
        st.setdefault('gslope', -np.inf); st.setdefault('gdirty', True)
    kf_tracks = list(state)
    lam = LAM0 * m_p
    orders = (sorted(kf_tracks, key=lambda t: -t[1]), sorted(kf_tracks, key=lambda t: t[1]))
    for level in range(MAX_LEVELS):
        for tr in orders[level % 2]:
            remove_keys(tracks, tr, state[tr], grow['steps'][tr], lam, stats)
            grow_step(tracks, tr, state[tr], grid, grow, lam, sizes, gbits)
        nxt = np.inf; pending = False                     # next level: at least x ratio, at least the smallest open slope
        for st in state.values():
            s_in = st['slope'][REACH:max(REACH, len(st['slope']) - REACH)]
            if len(s_in):
                m = s_in[s_in > lam]
                if len(m):
                    nxt = min(nxt, float(m.min()))
                pending |= bool(st['dirty'][REACH:max(REACH, len(st['dirty']) - REACH)].any())
            gs = st['gslope']
            if np.isfinite(gs) and gs > lam:
                nxt = min(nxt, float(gs))
        if not np.isfinite(nxt) and not pending:
            break
        if S.total - float(S.esum.sum()) <= 1e-6 * S.total:
            break
        lam = max(lam * LAM_RATIO, nxt) if np.isfinite(nxt) else lam * LAM_RATIO
    return state


# ----------------------------------------------------------------------------------------------------------------
# per-component keys
# ----------------------------------------------------------------------------------------------------------------
def sub_ints(K, nk, Ks):
    """The integers of the component sets Ks[c] (subsets of K) from the track's key integers nk [n, 3]."""
    K = np.asarray(K, dtype=np.int64)
    return [np.asarray(nk, dtype=np.int64)[np.searchsorted(K, np.asarray(Ks[c], dtype=np.int64)), c].copy() for c in range(3)]


def _touch_other_components(st, c, F, changed):
    """Mark dirty the keys of the other components whose windows intersect the changed frames."""
    cs = np.concatenate([[0], np.cumsum(changed)])
    for c2 in range(3):
        if c2 == c:
            continue
        K2 = st['Ks'][c2]
        if len(K2) <= 2 * REACH:
            continue
        pos = np.arange(REACH, len(K2) - REACH)
        lo = K2[pos - REACH] + 1; hi = K2[pos + REACH]
        st['dirty'][c2][pos[(cs[hi] - cs[lo]) > 0]] = True


def remove_keys_c(tracks, tr, st, c, step, lam, stats, keys, nks):
    """RD removal of component c's keys of track tr at one lambda. A key saves its d2t residual bits; when it is the last
    key of the union at its frame it also saves the union's gap symbol and three presence flags."""
    S = tracks.S; F = S.F
    b = tr[1]
    v = tracks.vals[tr]
    sub = np.asarray(S.subtrees[b])
    K, dirty, slope = st['Ks'][c], st['dirty'][c], st['slope'][c]
    n_rounds = 0
    while len(K) > 2 * REACH:
        pos = _candidates(dirty, slope, lam, len(K))
        if len(pos) == 0:
            break
        n_rounds += 1
        keep = np.ones(len(K), dtype=bool); keep[pos] = False
        idx, wid = windows(K, pos)
        Ks2 = list(st['Ks']); Ks2[c] = K[keep]
        loc2 = tracks.local_c(tr, Ks2, sub_ints(keys[tr], nks[tr], Ks2), step)
        rows, E, D, bad = _window_cost(S, b, sub, loc2, idx, wid, len(pos))
        stats['evals'] += 1
        nk = quantize(v[K, c][:, None], step[c])[:, 0]
        others = np.concatenate([st['Ks'][c2] for c2 in range(3) if c2 != c])
        uniq = ~np.isin(K[pos], others)
        r = key_residuals(nk, 2, K)
        bits = np.log2(1.0 + 2.0 * np.abs(r).astype(np.float64))[pos] + uniq * (GAP_BITS + 3 * FLAG_BITS)
        sl = D.sum(axis=1) / np.maximum(bits, 0.5)
        slope[pos] = sl; dirty[pos] = False
        slope[pos[bad]] = np.inf
        acc = _accept(S, sub, D, bad, sl, lam, slope, pos)
        if acc.any():
            fidx = _commit_windows(S, b, sub, loc2, rows, idx, wid, acc, D, E)
            stats['n_acc'] += int(acc.sum())
            for r_ in range(1, REACH + 1):
                dirty[np.clip(pos[acc] - r_, 0, len(K) - 1)] = True; dirty[np.clip(pos[acc] + r_, 0, len(K) - 1)] = True
            keep2 = np.ones(len(K), dtype=bool); keep2[pos[acc]] = False
            K = K[keep2]; dirty = dirty[keep2]; slope = slope[keep2]
            st['Ks'][c], st['dirty'][c], st['slope'][c] = K, dirty, slope
            changed = np.zeros(F, dtype=bool); changed[fidx] = True
            _touch_other_components(st, c, F, changed)
    stats['rounds'] += n_rounds


def ladder_c(tracks, kf_tracks, steps, keys, nks, m_p, stats):
    """The lambda ladder over the components of the keyframe tracks (starting from their track sets) -> {track: state}."""
    S = tracks.S
    r = _fast.ladder_c(S, kf_tracks, steps, tracks.vals, keys, nks, m_p, stats, LAM0, LAM_RATIO, MAX_LEVELS, REACH, GAP_BITS, FLAG_BITS)
    if r is not None:
        return r
    state = {tr: dict(Ks=[np.asarray(keys[tr], dtype=np.int64).copy() for _ in range(3)],
                      dirty=[np.ones(len(keys[tr]), dtype=bool) for _ in range(3)],
                      slope=[np.full(len(keys[tr]), np.inf) for _ in range(3)]) for tr in kf_tracks}
    lam = LAM0 * m_p
    orders = (sorted(kf_tracks, key=lambda t: -t[1]), sorted(kf_tracks, key=lambda t: t[1]))
    for level in range(MAX_LEVELS):
        for tr in orders[level % 2]:
            for c in range(3):
                remove_keys_c(tracks, tr, state[tr], c, steps[tr], lam, stats, keys, nks)
        lam = _next_lambda(S, state, lam)
        if lam is None:
            break
    return state


def resume_ladder_c(S, kf_tracks, state, steps, vals, keys, nks, m_p, stats, level0, i0, c0, lam):
    """ladder_c continued from (level0, the i0-th track of the level's order, component c0, lam) -- after the C ladder
    met a NaN."""
    tracks = Tracks(S, vals)
    orders = (sorted(kf_tracks, key=lambda t: -t[1]), sorted(kf_tracks, key=lambda t: t[1]))
    for level in range(level0, MAX_LEVELS):
        for i, tr in enumerate(orders[level % 2]):
            for c in range(3):
                if level == level0 and (i < i0 or (i == i0 and c < c0)):
                    continue
                remove_keys_c(tracks, tr, state[tr], c, steps[tr], lam, stats, keys, nks)
        lam = _next_lambda(S, state, lam)
        if lam is None:
            break
    return state


def _next_lambda(S, state, lam):
    """The next level of the component ladder: at least x ratio, at least the smallest open slope; None = done."""
    nxt = np.inf; pending = False
    for st in state.values():
        for c in range(3):
            s_in = st['slope'][c][REACH:max(REACH, len(st['slope'][c]) - REACH)]
            if len(s_in):
                m = s_in[s_in > lam]
                if len(m):
                    nxt = min(nxt, float(m.min()))
                pending |= bool(st['dirty'][c][REACH:max(REACH, len(st['dirty'][c]) - REACH)].any())
    if not np.isfinite(nxt) and not pending:
        return None
    if S.total - float(S.esum.sum()) <= 1e-6 * S.total:
        return None
    return max(lam * LAM_RATIO, nxt) if np.isfinite(nxt) else lam * LAM_RATIO


def kf_cost(K, nk, c0):
    """Coded-cost proxy of a keyframe track: residuals under the best of d1 / d2 / d2t + the gap symbols + mode / id."""
    return min(res_proxy(key_residuals(nk, pid, K)[1:], c0) for pid in range(3)) + gap_proxy(np.diff(K), c0) + 2.0


def ck_cost(Ks, nks, c0):
    """Coded-cost proxy of a per-component track: per-component residuals + the union's gaps + the presence flags."""
    res = sum(min(res_proxy(key_residuals(nks[c], pid, Ks[c])[1:], c0) for pid in range(3)) for c in range(3))
    U = np.unique(np.concatenate(Ks))
    return res + gap_proxy(np.diff(U), c0) + 3 * FLAG_BITS * max(0, len(U) - 2) + 2.0


def ck_finish(Ks, nks, c0):
    """Per-component best predictor of {d1, d2, d2t, nlms} -> (ids, residual arrays)."""
    n = max(len(k) for k in nks)
    A = np.zeros((n, 3), dtype=np.int64)
    for c in range(3):
        A[:len(nks[c]), c] = nks[c]
    rn = nlms_residuals(A, A[0])
    pids = []; res = []
    for c in range(3):
        nk = np.asarray(nks[c], dtype=np.int64)
        cands = {pid: key_residuals(nk, pid, Ks[c]) for pid in range(3)}
        cands[3] = rn[:len(nk), c]
        costs = {pid: res_proxy(r[1:], c0) for pid, r in cands.items()}
        pid = min(costs, key=costs.get)
        pids.append(int(pid)); res.append(cands[pid])
    return pids, res


def component_stage(tracks, track_list, mode, keys, nks, steps, m_p, c0, stats):
    """Thin the keyframe tracks per component; per track keep the cheapest of the track set, the per-component sets and
    their union (by the coded-cost proxies; every state change verified by the search)."""
    S = tracks.S
    kf_tracks = [tr for tr in track_list if mode[tr] == M_KF]
    if not kf_tracks:
        return
    state = ladder_c(tracks, kf_tracks, steps, keys, nks, m_p, stats)
    for tr in kf_tracks:
        K = keys[tr]; nk = nks[tr]; b = tr[1]
        Ks = state[tr]['Ks']
        if all(len(k) == len(K) for k in Ks):
            continue
        nk_c = sub_ints(K, nk, Ks)
        cost_ck = ck_cost(Ks, nk_c, c0); cost_k = kf_cost(K, nk, c0)
        U = np.unique(np.concatenate(Ks))
        nkU = nk[np.searchsorted(K, U)]
        cost_u = kf_cost(U, nkU, c0) if len(U) < len(K) else np.inf
        best = min(cost_k, cost_ck, cost_u)
        if best == cost_u:
            if S.try_local(b, tracks.local_keys(tr, U, nkU, steps[tr]), tag=(tr, 'union')):
                S.commit_pending(); keys[tr] = U; nks[tr] = nkU
                continue
            S.drop_pending(); best = min(cost_k, cost_ck)
        if best == cost_k:
            if S.try_local(b, tracks.local_keys(tr, K, nk, steps[tr]), tag=(tr, 'restore')):
                S.commit_pending()
                continue
            S.drop_pending()
        mode[tr] = M_CK; keys[tr] = Ks; nks[tr] = nk_c
    S.esum = S.err.sum(axis=0)
