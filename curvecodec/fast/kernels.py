"""The C kernels of the encoder and of the shared numerics (fastkern.c + enckern.c, one library).

Every function mirrors the numpy code expression for expression (same float64 operation order, no FMA contraction,
numpy's pairwise summation written out, numpy's own sin / cos / log2 inner loops through nploops.py), so the
results are identical. Each wrapper checks dtypes / layouts and returns None (or False) when it does not apply.
"""
import ctypes

import numpy as np

from . import nploops
from ._build import build_lib

_VP = ctypes.c_void_p
_I64 = ctypes.c_int64
_INT = ctypes.c_int
_DBL = ctypes.c_double
_st = {'lib': None, 'tried': False, 'why': ''}
SEARCH = False                 # numpy's sin / cos / log2 loops are callable from C: the search kernels are on


def _p(a):
    return a.ctypes.data


def _declare(L):
    def d(name, res, args):
        f = getattr(L, name); f.restype = res; f.argtypes = args
    d('np_sum', _DBL, [_VP, _I64])
    d('set_sum_variant', None, [_INT])
    d('set_log_tab', None, [_VP])
    d('nlms_run', None, [_INT, _INT, _VP, _VP, _INT, _INT, _INT, _INT, _INT, _I64, _I64, _VP])
    d('model_forward', None, [_VP, _VP, _I64, _INT, _INT, _I64, _I64, _I64, _VP, _VP, _VP, _VP, _VP])
    d('features_batch', _I64, [_I64, _VP, _I64, _VP, _VP, _VP, _VP, _VP, _VP, _VP])
    d('window_features', _INT, [_I64, _I64, _VP, _VP, _VP, _VP, _I64, _I64, _VP])
    d('symbol_freqs', _INT, [_I64] * 3 + [_VP] * 5 + [_VP, _I64, _VP, _I64, _VP, _I64, _I64, _INT, _VP, _VP])
    d('interp_vectors', None, [_I64, _VP, _VP, _I64, _INT, _VP, _VP])
    d('ek_trig', None, [_VP, _VP, _VP, _VP])
    d('ek_trig_ok', _INT, [])
    d('ek_sincos', None, [_I64, _VP, _VP, _VP])
    d('ek_log2set', None, [_VP, _VP])
    d('ek_log2', None, [_I64, _VP, _VP])
    d('ek_subtree_eval', _INT, [_INT, _INT, _INT, _VP, _INT, _VP, _VP, _VP, _INT, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _DBL,
                                _INT, _VP, _VP, _DBL, _VP, _VP, _VP, _VP, _VP, _VP, _INT, _INT])
    d('ek_commit_rows', _INT, [_INT, _INT, _INT, _VP, _INT, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP])
    d('ek_reconstruct', _INT, [_VP, _I64, _VP, _VP, _I64, _INT, _INT, _VP])
    d('ek_rk_rd', _INT, [_VP] * 8 + [_DBL, _INT, _INT, _INT, _VP, _DBL, _VP, _VP])
    d('ek_ladder_c', _INT, [_VP, _VP, _INT, _VP, _VP, _VP, _VP, _DBL, _DBL, _INT, _DBL, _INT, _INT, _DBL, _DBL, _DBL, _VP, _VP])
    d('ek_sess_begin', _INT, [_VP, _VP])
    d('ek_sess_end', None, [])
    d('ek_sess_fexit', None, [_INT])
    d('ek_sess_eval', _INT, [_VP, _I64, _VP, _VP, _INT, _VP, _VP, _INT, _VP, _VP])
    d('ek_sess_stats', None, [_VP])
    d('ek_sess_keep', None, [])
    d('ek_sess_commit', _INT, [])
    d('ek_sess_refresh_b', None, [])


def _sum_selftest(L):
    """Pick the C summation that reproduces numpy's add.reduce on every test vector."""
    rng = np.random.default_rng(12345)
    vecs = []
    for n in list(range(1, 300)) + [511, 512, 513, 1000, 1024, 1025, 2047, 2400, 4096, 5566, 9000, 21549]:
        vecs.append(rng.random(n) * rng.choice([1e-3, 1.0, 1e3]))
        vecs.append(np.exp(rng.normal(0, 3, n)))
    for variant in (0, 1):
        L.set_sum_variant(variant)
        if all(L.np_sum(_p(v), len(v)) == float(v.sum()) for v in vecs):
            return
    raise RuntimeError('no C summation reproduces numpy add.reduce')


def _trig_selftest(L):
    rng = np.random.default_rng(20260928)
    for n in list(range(1, 67)) + [100, 255, 256, 257, 1000, 4097]:
        for off in range(4):
            x = np.concatenate([rng.random(n + off) * 3.3, [0.0, 1e-300, 5e-7, np.pi / 2, np.pi, 7.0, 40.0]])
            s = np.empty_like(x); c = np.empty_like(x)
            xs = x[off:]
            L.ek_sincos(len(xs), _p(x) + 8 * off, _p(s) + 8 * off, _p(c) + 8 * off)
            if not (np.array_equal(s[off:], np.sin(xs)) and np.array_equal(c[off:], np.cos(xs))):
                return False
    for n in list(range(1, 40)) + [100, 1000]:
        x = 1.0 + 2.0 * np.abs(np.round(rng.normal(0, 30, n)))
        y = np.empty_like(x)
        L.ek_log2(n, _p(x), _p(y))
        if not np.array_equal(y, np.log2(x)):
            return False
    return True


def lib():
    global SEARCH
    if _st['lib'] is None and not _st['tried']:
        _st['tried'] = True
        try:
            L = build_lib('kernels', ['fastkern.c', ('enckern.c', ['-fno-math-errno'])])
            _declare(L)
            _sum_selftest(L)
            from ..features import LOG_TAB
            _st['log_tab'] = np.ascontiguousarray(LOG_TAB, dtype=np.int64)       # the C side keeps the pointer
            L.set_log_tab(_p(_st['log_tab']))
            lp = nploops.loops(('sin', 'cos', 'log2'))
            if lp is not None:
                (fs, ds), (fc, dc), (fl, dl) = lp['sin'], lp['cos'], lp['log2']
                L.ek_trig(fs, ds or None, fc, dc or None)
                L.ek_log2set(fl, dl or None)
                SEARCH = bool(L.ek_trig_ok()) and _trig_selftest(L)
            if not SEARCH:
                L.ek_trig(None, None, None, None)
                _st['why'] = 'numpy sin / cos loops unavailable: the search kernels stay off'
            _st['lib'] = L
        except Exception as e:  # noqa: BLE001 -- any build / load problem: numpy
            _st['why'] = f'{type(e).__name__}: {e}'
            import warnings
            warnings.warn(f'curvecodec.fast: C kernels unavailable, using numpy ({_st["why"][:300]})')
    return _st['lib']


def ready():
    return lib() is not None


def status():
    return dict(loaded=lib() is not None, search=SEARCH, why=_st['why'])


# ================================================================================================================
# stateless kernels
# ================================================================================================================
def nlms_run(A, x0, order, mu_shift, decode, init):
    from ..predict import NLMS_CLAMP, NLMS_SH, NLMS_WMAX
    L = lib()
    if not isinstance(A, np.ndarray) or A.dtype != np.int64 or A.ndim != 2 or order > 64 or init not in ('zero', 'd2'):
        return None
    F, C = A.shape
    Ac = np.ascontiguousarray(A)
    x0 = np.ascontiguousarray(np.broadcast_to(np.asarray(x0, dtype=np.int64), (C,)))
    out = np.empty((F, C), dtype=np.int64)
    L.nlms_run(F, C, _p(Ac), _p(x0), int(order), int(mu_shift), int(bool(decode)), int(init == 'd2'), int(NLMS_SH), int(NLMS_CLAMP),
               int(NLMS_WMAX), _p(out))
    return out


def model_forward(coded, ctxs, nsym, nctx, adapt_rate, init):
    from ..rans import MODEL_INC, RANS_M
    L = lib()
    sym = np.asarray(coded, dtype=np.int64).ravel(); ctx = np.asarray(ctxs, dtype=np.int64).ravel()
    n = len(sym)
    if n != len(ctx) or (n and (int(sym.min()) < 0 or int(sym.max()) >= nsym or int(ctx.min()) < 0 or int(ctx.max()) >= nctx)):
        return None
    ini = np.ascontiguousarray(np.asarray(init, dtype=np.int64)[:nctx])
    if ini.shape != (nctx, nsym):
        return None
    starts = np.empty(n, np.int64); freqs = np.empty(n, np.int64)
    cnts = np.empty(nctx * nsym, np.int64); tots = np.empty(nctx, np.int64)
    L.model_forward(_p(sym), _p(ctx), n, int(nsym), int(nctx), int(RANS_M - nsym), int(MODEL_INC), int(nsym + (MODEL_INC << (adapt_rate + 5))),
                    _p(ini), _p(starts), _p(freqs), _p(cnts), _p(tots))
    return starts.tolist(), freqs.tolist()


def interp_vectors(K, vk, F):
    L = lib()
    if not isinstance(K, np.ndarray) or K.ndim != 1 or K.dtype != np.int64:
        return None
    nk = len(K)
    vk = np.ascontiguousarray(np.asarray(vk).astype(np.float64))
    if nk == F or nk < 2 or vk.shape != (nk, 3) or int(K[0]) != 0 or int(K[-1]) != F - 1 or bool(np.any(np.diff(K) <= 0)):
        return None
    K = np.ascontiguousarray(K)
    out = np.empty((F, 3)); m = np.empty((nk, 3))
    L.interp_vectors(nk, _p(K), _p(vk), int(F), 1, _p(m), _p(out))
    return out


_TABS = {}


def symbol_freqs(logw, mu, logs, alphabet):
    from .. import features as FT
    L = lib()
    lo, hi = alphabet
    lw = np.ascontiguousarray(logw, np.int64); m = np.ascontiguousarray(mu, np.int64); ls = np.ascontiguousarray(logs, np.int64)
    if lw.ndim != 2 or lw.shape != m.shape or lw.shape != ls.shape or lw.shape[1] > 64:
        return None
    N, K = lw.shape
    if N and (int(ls.min()) < FT.LOGS_LO or int(ls.max()) > FT.LOGS_HI):
        return None
    lo = np.ascontiguousarray(lo, np.int64); hi = np.ascontiguousarray(hi, np.int64); S = len(lo)
    t = _TABS.get('sf')
    if t is None:
        t = _TABS['sf'] = tuple(np.ascontiguousarray(a, np.int64) for a in (FT.EXP_TAB, FT.INVS_TAB, FT.SIG_TAB))
    out = np.empty((N, S), np.int64); work = np.empty(2 * K + 2 * S, np.int64)
    if L.symbol_freqs(N, K, S, _p(lw), _p(m), _p(ls), _p(lo), _p(hi), _p(t[0]), FT.EXP_LIM, _p(t[1]), FT.LOGS_LO, _p(t[2]), FT.SIG_LIM,
                      FT.MU_LIM, FT.FREQ_BITS, _p(out), _p(work)) < 0:
        return None
    return out


def features_batch(st, n, n_max, G, R, Xl):
    """CurveState.features() of every residual of the lockstep loop (time-major, active columns ascending) ->
    (X int64 [N, 36], y int64 [N], column int64 [N]), or None (a feature beyond int16)."""
    from ..features import L as Lq, NFEAT
    L = lib()
    n = np.asarray(n, np.int64); C = len(n)
    if C == 0 or st.base.shape != (C, NFEAT):
        return None
    lgM = Lq(np.maximum(np.asarray(G, np.int64), 1) - 1)
    tt = np.concatenate([np.arange(k, dtype=np.int64) for k in n.tolist()])
    jj = np.repeat(np.arange(C, dtype=np.int64), n)
    r = np.ascontiguousarray(R[tt, jj], np.int64)
    dx = np.ascontiguousarray(Xl[tt + 1, jj] - Xl[tt, jj], np.int64)
    lg = np.ascontiguousarray(lgM[tt, jj], np.int64)
    x0 = np.ascontiguousarray(Xl[0], np.int64)
    base = np.ascontiguousarray(st.base, np.int64)
    X = np.empty((len(tt), NFEAT), np.int16); xbuf = np.empty(int(n_max) + 2, np.int64)
    if L.features_batch(C, _p(n), int(n_max), _p(r), _p(dx), _p(x0), _p(lg), _p(base), _p(X), _p(xbuf)):
        return None
    order = np.lexsort((jj, tt))
    return X[order].astype(np.int64), r[order], jj[order]


def window_features(n, nmax, r, x, lg, base, T, t_lo, out):
    """Training rows t_lo .. T-1 of one curve into out (float32 [T - t_lo, 36]); False when a feature leaves int16."""
    L = lib()
    r = np.ascontiguousarray(r, np.int64); x = np.ascontiguousarray(x, np.int64); lg = np.ascontiguousarray(lg, np.int64)
    base = np.ascontiguousarray(base, np.int64)
    return L.window_features(int(n), int(nmax), _p(r), _p(x), _p(lg), _p(base), int(T), int(t_lo), _p(out)) == 0


def reconstruct(K, nk, step, F, kind):
    L = lib()
    K = np.asarray(K); nk = np.asarray(nk)
    if K.dtype != np.int64 or K.ndim != 1 or nk.dtype != np.int64 or nk.shape != (len(K), 3) or len(K) < 1:
        return None
    s3 = _step3(step)
    if s3 is None:
        return None
    if len(K) != F and (len(K) < 2 or int(K[0]) != 0 or int(K[-1]) != F - 1 or bool(np.any(np.diff(K) <= 0))):
        return None
    if len(K) == F and not np.array_equal(K, np.arange(F)):
        return None
    Kc = np.ascontiguousarray(K); nkc = np.ascontiguousarray(nk)
    out = np.empty((F, 4 if kind == 'rot' else 3), np.float32)
    if L.ek_reconstruct(_p(Kc), len(Kc), _p(nkc), _p(s3), int(F), 0 if kind == 'rot' else 1, 1, _p(out)) < 0:
        return None
    return out


def _step3(step):
    """A track's step(s) as float64 [3], or None."""
    s = np.asarray(step)
    if s.dtype != np.float32 or s.shape not in ((), (1,), (3,)):
        return None
    return np.ascontiguousarray(np.broadcast_to(s.astype(np.float64), (3,)))


# ================================================================================================================
# the search state seen from C
# ================================================================================================================
def _ok_f32(a, tail):
    return isinstance(a, np.ndarray) and a.dtype == np.float32 and a.flags.c_contiguous and a.shape[1:] == tail


def _ok_f64(a, shape):
    return isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags.c_contiguous and a.shape == shape


def _state_ok(S):
    F, B = S.F, S.B
    return (_ok_f32(S.cur_local, (B, 7)) and _ok_f64(S.obj_rot, (F, B, 4)) and _ok_f64(S.obj_pos, (F, B, 3))
            and _ok_f64(S.raw_cols, (F, B, 3, 3)) and _ok_f64(S.raw_pos, (F, B, 3)) and _ok_f64(S.raw_rot, (F, B, 4))
            and _ok_f64(S.err, (F, B)) and _ok_f64(S.emax, (B,)) and _ok_f64(S.cnt, (B,)) and _ok_f64(S.esum, (B,))
            and _ok_f64(S.limits, (B,)) and _ok_f64(S.mlim, (B,)) and _ok_f64(S.n_allow, (B,)))


def _cache(S):
    """Per search state: the subtree plans and the bone-major / structure-of-arrays copies of the constant raw arrays."""
    c = S.__dict__.get('_c')
    if c is None:
        rb = np.ascontiguousarray(S.raw_rot.transpose(1, 0, 2)); pb = np.ascontiguousarray(S.raw_pos.transpose(1, 0, 2))
        cols_s = np.ascontiguousarray(S.raw_cols.reshape(S.F, S.B, 9).transpose(1, 2, 0))
        pos_s = np.ascontiguousarray(S.raw_pos.transpose(1, 2, 0))
        raw_a = np.ascontiguousarray(np.concatenate([S.raw_cols.reshape(S.F, S.B, 9), S.raw_pos], axis=2).transpose(1, 0, 2))
        fexit = bool(np.isfinite(cols_s).all() and np.isfinite(pos_s).all() and np.isfinite(rb).all() and np.isfinite(pb).all())
        c = S._c = dict(plans={}, rot_bm=rb, pos_bm=pb, cols_s=cols_s, pos_s=pos_s, raw_a=raw_a, fexit=fexit)
    return c


def _plan(S, b):
    """(sub list, sub int32, index of the parent within sub when it precedes it else -1, parent int32)."""
    plans = _cache(S)['plans']
    pl = plans.get(b)
    if pl is None:
        sub = [int(c) for c in S.subtrees[b]]
        pos = {}; rel = []
        par = [int(S.parents[c]) for c in sub]
        for j, c in enumerate(sub):
            rel.append(pos[par[j]] if (par[j] >= 0 and par[j] in pos) else -1)
            pos[c] = j
        pl = plans[b] = (sub, np.asarray(sub, np.int32), np.asarray(rel, np.int32), np.asarray(par, np.int32))
    return pl


def _run_subtree(S, b, new_local, idx, mode, lim_max, lim_sum):
    """ek_subtree_eval -> (bones done, sub, rot [ns, n, 4], pos [ns, n, 3], err [ns, n], emax, esum, cnt) or None.
    mode 0 = full evaluation, 2 = stop at the first bone over its mean cap (lim_sum) or guard (lim_max)."""
    L = lib()
    if not _state_ok(S) or not (isinstance(new_local, np.ndarray) and new_local.dtype == np.float32 and new_local.shape == (S.F, 7)):
        return None
    new_local = np.ascontiguousarray(new_local)
    sub, sub_a, rel_a, par_a = _plan(S, b)
    ns = len(sub)
    if idx is None:
        n = S.F; ip = None
    else:
        idx = np.asarray(idx)
        if idx.ndim != 1 or idx.dtype.kind not in 'iu' or len(idx) == 0:
            return None
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        if int(idx.min()) < 0 or int(idx.max()) >= S.F:
            return None
        n = len(idx); ip = _p(idx)
    c = _cache(S)
    rot = np.empty((ns, n, 4)); pos = np.empty((ns, n, 3)); err = np.empty((ns, n))
    emax = np.empty(ns); esum = np.empty(ns); ecnt = np.empty(ns)
    lm = np.ascontiguousarray(lim_max, dtype=np.float64) if lim_max is not None else np.zeros(ns)
    ls = np.ascontiguousarray(lim_sum, dtype=np.float64) if lim_sum is not None else np.zeros(ns)
    nd = L.ek_subtree_eval(S.F, S.B, n, ip, ns, _p(sub_a), _p(rel_a), _p(par_a), 0, _p(new_local), _p(S.cur_local), _p(S.obj_rot),
                           _p(S.obj_pos), _p(c['rot_bm']), _p(c['pos_bm']), _p(c['cols_s']), _p(c['pos_s']), float(S.shell), int(mode),
                           _p(lm), _p(ls), float(S.p), _p(rot), _p(pos), _p(err), _p(emax), _p(esum), _p(ecnt), 0 if mode == 0 else 1,
                           int(c['fexit']))
    if nd < 0:
        return None
    return nd, sub, rot, pos, err, emax, esum, ecnt


def eval_subtree_mean(S, b, new_local, early_exit, mlim, glim):
    """Search.eval_subtree -> (ok, rot, pos, err, cnt) dicts, or None."""
    if lib() is None:
        return None
    sub_l = S.subtrees[b]
    F = float(S.F)
    r = _run_subtree(S, b, new_local, None, 2 if early_exit else 0, np.asarray(glim, np.float64)[sub_l], np.asarray(mlim, np.float64)[sub_l] * F)
    if r is None:
        return None
    nd, sub, rot, pos, err, emax, esum, ecnt = r
    t_rot, t_pos, t_err, cnt = {}, {}, {}, {}
    ok = True
    added = 0.0
    for j in range(nd):
        c = sub[j]
        s_c = float(esum[j])
        if s_c > mlim[c] * F or emax[j] > glim[c]:
            ok = False
            if early_exit:
                return ok, t_rot, t_pos, t_err, cnt
        added += s_c - S.esum[c]
        t_rot[c] = rot[j]; t_pos[c] = pos[j]; t_err[c] = err[j]; cnt[c] = float(ecnt[j])
    if nd < len(sub):
        raise RuntimeError('curvecodec.fast: subtree early exit mismatch')
    if np.isfinite(S.total) and added > S.total - float(S.esum.sum()) + 1e-9:
        ok = False
    return ok, t_rot, t_pos, t_err, cnt


def eval_subtree_frames(S, b, local_b, idx):
    """search.eval_subtree_frames -> (rot, pos, err) dicts, or None."""
    if lib() is None:
        return None
    r = _run_subtree(S, b, local_b, idx, 0, None, None)
    if r is None:
        return None
    nd, sub, rot, pos, err = r[:5]
    return {c: rot[j] for j, c in enumerate(sub)}, {c: pos[j] for j, c in enumerate(sub)}, {c: err[j] for j, c in enumerate(sub)}


def commit_rows(S):
    """Search.commit_pending's writes of the pending candidate in one call (True), or False (numpy writes them)."""
    L = lib()
    pend = S._pending
    F, B = S.F, S.B
    if F < 1 or B < 2 or not _state_ok(S):
        return False
    b, _, loc, t_rot, t_pos, t_err, cnt = pend
    if not (type(loc) is np.ndarray and loc.dtype == np.float32 and loc.shape == (F, 7) and loc.flags.c_contiguous):
        return False
    sub, sub_a, _, _ = _plan(S, b)
    ns = len(sub)
    ptr = np.empty(3 * ns, np.int64); cn = np.empty(ns)
    for j, c in enumerate(sub):
        r = t_rot.get(c); q = t_pos.get(c); e = t_err.get(c)
        if not (_ok_f64(r, (F, 4)) and _ok_f64(q, (F, 3)) and _ok_f64(e, (F,))) or c not in cnt:
            return False
        ptr[j] = r.__array_interface__['data'][0]; ptr[ns + j] = q.__array_interface__['data'][0]
        ptr[2 * ns + j] = e.__array_interface__['data'][0]
        cn[j] = cnt[c]
    p0 = _p(ptr)
    if L.ek_commit_rows(F, B, int(b), _p(loc), ns, _p(sub_a), p0, p0 + 8 * ns, p0 + 16 * ns, _p(cn), _p(S.cur_local), _p(S.obj_rot),
                        _p(S.obj_pos), _p(S.err), _p(S.emax), _p(S.cnt)) != 0:
        return False
    S._pending = None
    return True


class _EkState(ctypes.Structure):
    _fields_ = [('F', ctypes.c_int), ('B', ctypes.c_int), ('cur_local', _VP), ('obj_rot', _VP), ('obj_pos', _VP), ('err', _VP),
                ('emax', _VP), ('cnt', _VP), ('esum', _VP), ('limits', _VP), ('mlim', _VP), ('n_allow', _VP),
                ('raw_cols_s', _VP), ('raw_pos_s', _VP), ('shell', ctypes.c_double), ('p', ctypes.c_double), ('raw_a', _VP)]


class _EkTrack(ctypes.Structure):
    _fields_ = [('b', ctypes.c_int), ('kind', ctypes.c_int), ('nsub', ctypes.c_int), ('sub', _VP), ('par_rel', _VP), ('par_abs', _VP),
                ('v', _VP)]


class _EkComp(ctypes.Structure):
    _fields_ = [('K', _VP), ('V', _VP), ('D', _VP), ('SL', _VP), ('n', ctypes.c_int64)]


def _ek_state(S):
    if not _state_ok(S):
        return None
    c = _cache(S)
    return _EkState(S.F, S.B, _p(S.cur_local), _p(S.obj_rot), _p(S.obj_pos), _p(S.err), _p(S.emax), _p(S.cnt), _p(S.esum),
                    _p(S.limits), _p(S.mlim), _p(S.n_allow), _p(c['cols_s']), _p(c['pos_s']), float(S.shell), float(S.p), _p(c['raw_a']))


def _ek_track(S, tr, v):
    if not (isinstance(v, np.ndarray) and v.dtype == np.float64 and v.flags.c_contiguous and v.shape == (S.F, 3)):
        return None
    kind, b = tr
    sub, sub_a, rel_a, par_a = _plan(S, b)
    t = _EkTrack(int(b), 0 if kind == 'rot' else 1, len(sub), _p(sub_a), _p(rel_a), _p(par_a), _p(v))
    t._keep = (sub_a, rel_a, par_a, v)
    return t


# ================================================================================================================
# the key ladder in C
# ================================================================================================================
def remove_keys(S, tr, st, step, v, lam, reach, stats):
    """keys.remove_keys in one call: True when done (every state change made); False -> numpy (also to finish after a
    NaN, from the state reached)."""
    L = lib()
    K, dirty, slope = st['K'], st['dirty'], st['slope']
    if not (isinstance(K, np.ndarray) and K.dtype == np.int64 and K.ndim == 1 and dirty.dtype == bool and slope.dtype == np.float64
            and dirty.shape == K.shape and slope.shape == K.shape):
        return False
    ms = _ek_state(S); trk = _ek_track(S, tr, v); s3 = _step3(step)
    if ms is None or trk is None or s3 is None or not np.isfinite(S.total):
        return False
    sub = np.asarray(S.subtrees[tr[1]])
    cap_sub = np.ascontiguousarray(S.mlim[sub] * S.F, dtype=np.float64)
    Kc = K.copy(); dc = dirty.astype(np.uint8); sc = slope.copy(); oc = np.zeros((len(K), 3), np.int64)
    nK = np.array([len(K)], np.int64); acc = np.zeros(5); flag = np.zeros(1, np.int64)
    r = L.ek_rk_rd(ctypes.byref(ms), ctypes.byref(trk), _p(Kc), _p(oc), _p(dc), _p(sc), _p(nK), _p(s3), float(lam), int(reach),
                   int(2 * reach), 1, _p(cap_sub), float(S.total), _p(acc), _p(flag))
    if r < 0:
        raise MemoryError('curvecodec.fast: ek_rk_rd allocation failed')
    n = int(nK[0])
    if flag[0]:
        st['K'], st['dirty'], st['slope'] = Kc[:n].copy(), dc[:n].astype(bool), sc[:n].copy()
        st['gdirty'] = True
    else:
        dirty[...] = dc.astype(bool); slope[...] = sc
    stats['rounds'] += int(acc[0]); stats['evals'] += int(acc[1]); stats['n_acc'] += int(acc[4])
    return r != 1


class _Session:
    """Repeated evaluations of one track's candidate steps between two commits (enckern.c ek_sess_*)."""

    def __init__(self, S, tr, v, K):
        L = lib()
        self.L, self.S, self.b = L, S, tr[1]
        self.st = _ek_state(S); self.trk = _ek_track(S, tr, v)
        if self.st is None or self.trk is None:
            raise _Decline()
        self.K = np.ascontiguousarray(K, np.int64); self.off = np.zeros((len(self.K), 3), np.int64)
        self.ns = len(S.subtrees[self.b])
        self.stats = np.empty(3 * self.ns)
        if L.ek_sess_begin(ctypes.byref(self.st), ctypes.byref(self.trk)) < 0:
            raise MemoryError('curvecodec.fast: ek_sess_begin allocation failed')
        L.ek_sess_fexit(int(_cache(S)['fexit']))

    def close(self):
        self.L.ek_sess_end()

    def eval(self, step3, mode, lim_max, lim_sum):
        """-> (bones evaluated, stats [nd, 3] = max, sum, count over p)."""
        c = _cache(self.S)
        nd = self.L.ek_sess_eval(_p(self.K), len(self.K), _p(self.off), _p(step3), 0, _p(c['rot_bm']), _p(c['pos_bm']), int(mode),
                                 _p(lim_max), None if lim_sum is None else _p(lim_sum))
        if nd < -1:
            raise MemoryError('curvecodec.fast: ek_sess_eval allocation failed')
        if nd <= 0:
            return nd, None
        self.L.ek_sess_stats(_p(self.stats))
        return nd, self.stats[:3 * nd].reshape(-1, 3).copy()

    def commit(self):
        """Commit the kept candidate (Search.commit_pending's effect)."""
        if self.L.ek_sess_commit() != 0:
            raise RuntimeError('curvecodec.fast: no complete kept candidate to commit')
        S = self.S
        S._pending = None
        sub = S.subtrees[self.b]
        S.esum[sub] = S.err[:, sub].sum(axis=0)
        self.L.ek_sess_refresh_b()


class _Decline(Exception):
    pass


def grow_step(S, tr, st, v, grid, grow, lam, sizes, gbits, max_offers):
    """keys.grow_step with the offers evaluated in a C session (an offer's evaluation stops at the first bone over its
    guard or its cap): True when done, False -> numpy."""
    from ..quantize import quantize
    L = lib()
    K = st['K']
    if not (isinstance(K, np.ndarray) and K.dtype == np.int64 and K.ndim == 1):
        return False
    b = tr[1]
    sub = np.asarray(S.subtrees[b]); ns = len(sub)
    subl = sub.tolist()
    lim_max = np.ascontiguousarray(S.limits[sub], dtype=np.float64)
    idx, steps = grow['idx'], grow['steps']
    try:
        ss = None
        for _ in range(max_offers):
            if not (st['gdirty'] or st['gslope'] <= lam):
                return True
            best = None
            rem_sub = S.mlim[sub] * S.F - S.esum[sub]; rem_tot = S.total - float(S.esum.sum())
            pairs = np.ascontiguousarray(np.stack([S.esum[sub], rem_sub], axis=1).ravel(), dtype=np.float64)
            if ss is None:
                ss = _Session(S, tr, v, K)
            st['gdirty'] = False; st['gslope'] = np.inf
            for size in sizes:
                i2 = idx[tr] + size
                step2 = grid.step(i2)
                s3 = _step3(step2)
                nd, es = ss.eval(s3, 3, lim_max, pairs)
                if nd < ns:                       # over a guard or a cap (or a NaN, which numpy's tests also refuse)
                    continue
                emax, esum = es[:, 0], es[:, 1]
                if any(float(emax[j]) > S.limits[c] for j, c in enumerate(subl)):
                    continue
                D = np.array([float(esum[j]) - S.esum[c] for j, c in enumerate(subl)])
                bits = size * 3.0 * len(K) / grid.G * gbits
                sl = float(D.sum() / bits)
                if not (bool(np.all(D <= rem_sub + 1e-12)) and D.sum() <= rem_tot + 1e-12):
                    continue
                st['gslope'] = min(st['gslope'], sl)
                if sl <= lam:
                    L.ek_sess_keep()
                    best = (i2, step2)
                    break
            if best is None:
                return True
            i2, step2 = best
            ss.commit()
            idx[tr] = i2; steps[tr] = step2; grow['X'][tr] = quantize(v, step2)
            grow['recost']([tr])
            st['gdirty'] = True
        return True
    except _Decline:
        return False
    finally:
        if ss is not None:
            ss.close()


def ladder_c(S, kf_tracks, steps, vals, keys, nks, m_p, stats, lam0, ratio, max_levels, reach, gap_bits, flag_bits):
    """keys.ladder_c in one call -> {track: state}, or None (numpy)."""
    L = lib()
    if not np.isfinite(S.total):
        return None
    ms = _ek_state(S)
    if ms is None:
        return None
    arrs, comps, trks, st3 = [], (_EkComp * (3 * len(kf_tracks)))(), (_EkTrack * len(kf_tracks))(), np.zeros((len(kf_tracks), 3))
    for t, tr in enumerate(kf_tracks):
        trk = _ek_track(S, tr, vals[tr]); s3 = _step3(steps[tr])
        Kt = np.asarray(keys[tr]); nkt = np.asarray(nks[tr])
        if trk is None or s3 is None or Kt.dtype != np.int64 or nkt.dtype != np.int64 or nkt.shape != (len(Kt), 3):
            return None
        trks[t] = trk; st3[t] = s3; arrs.append(trk._keep)
        for c in range(3):
            Kc = np.ascontiguousarray(Kt.copy()); Vc = np.ascontiguousarray(nkt[:, c]); Dc = np.ones(len(Kc), np.uint8)
            Sc = np.full(len(Kc), np.inf)
            arrs.append((Kc, Vc, Dc, Sc))
            comps[3 * t + c] = _EkComp(_p(Kc), _p(Vc), _p(Dc), _p(Sc), len(Kc))
    pos = {tr: t for t, tr in enumerate(kf_tracks)}
    ord0 = np.array([pos[tr] for tr in sorted(kf_tracks, key=lambda t: -t[1])], np.int64)
    ord1 = np.array([pos[tr] for tr in sorted(kf_tracks, key=lambda t: t[1])], np.int64)
    acc = np.zeros(7); where = np.zeros(4)
    r = L.ek_ladder_c(ctypes.byref(ms), ctypes.addressof(trks), len(kf_tracks), _p(st3), ctypes.addressof(comps), _p(ord0), _p(ord1),
                      float(lam0), float(ratio), int(max_levels), float(m_p), int(reach), 1, float(gap_bits), float(gap_bits + 3 * flag_bits),
                      float(S.total), _p(acc), _p(where))
    if r < 0:
        raise MemoryError('curvecodec.fast: ek_ladder_c allocation failed')
    out = {}
    for t, tr in enumerate(kf_tracks):
        Ks, Ds, Ss = [], [], []
        for c in range(3):
            n = int(comps[3 * t + c].n)
            a = arrs[4 * t + 1 + c]
            Ks.append(a[0][:n].copy()); Ds.append(a[2][:n].astype(bool)); Ss.append(a[3][:n].copy())
        out[tr] = dict(Ks=Ks, dirty=Ds, slope=Ss)
    stats['rounds'] += int(acc[0]); stats['evals'] += int(acc[1]); stats['n_acc'] += int(acc[4])
    if r == 1:                                    # a NaN: the numpy ladder continues from where the C one stopped
        from ..keys import resume_ladder_c
        return resume_ladder_c(S, kf_tracks, out, steps, vals, keys, nks, m_p, stats, int(where[0]), int(where[1]), int(where[2]),
                               float(where[3]))
    return out
