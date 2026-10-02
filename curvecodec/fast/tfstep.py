"""The integer transformer step in C (tfstep.c): integer arithmetic only, so the build flags cannot change a bit (AVX-512
VNNI where available, a portable path elsewhere).

  state(model, C, pad)          -> a State whose step(x, act) returns model.IntTf's numpy step for the active rows
  c_model(model, C, pad)        -> (fn, ctx, release): the raw C step for the C decoder loop
        int tfs_step(void *ctx, int64_t t, int64_t n_act, const int64_t *act, const int64_t *x, int64_t *out)
  pack_forward(model, st, ...)  -> the encoder's whole model pass (features of every residual + the model) in C
"""
import ctypes
import os
import weakref

import numpy as np

from ._build import build_lib

_st = {'lib': None, 'tried': False, 'why': ''}
THREADS = int(os.environ.get('CURVECODEC_TF_THREADS', '1') or 1)
GPU = int(os.environ['CURVECODEC_TF_GPU']) if os.environ.get('CURVECODEC_TF_GPU', '') != '' else None
_VP = ctypes.c_void_p
_I64 = ctypes.c_int64


def lib():
    if _st['lib'] is None and not _st['tried']:
        _st['tried'] = True
        try:
            L = build_lib('tfstep', ['tfstep.c'], flags=['-pthread'], std='gnu11', link=['-pthread'])
            for name, res, args in (('tfs_model_create', _VP, [_VP, _VP]), ('tfs_model_free', None, [_VP]),
                                    ('tfs_state_create', _VP, [_VP, _I64, _I64]), ('tfs_state_free', None, [_VP]),
                                    ('tfs_step', ctypes.c_int, [_VP, _I64, _I64, _VP, _VP, _VP]), ('tfs_simd', ctypes.c_int, []),
                                    ('tfs_forward_lockstep', ctypes.c_int, [_VP, _I64, _I64, _VP, _VP, _VP, ctypes.c_int, _VP]),
                                    ('tfs_set_threads', ctypes.c_int, [_VP, ctypes.c_int, _VP])):
                f = getattr(L, name); f.restype = res; f.argtypes = args
            _st['lib'] = L
        except Exception as e:  # noqa: BLE001
            _st['why'] = f'{type(e).__name__}: {e}'
            import warnings
            warnings.warn(f'curvecodec.fast: C transformer step unavailable, using numpy ({_st["why"][:300]})')
    return _st['lib']


def status():
    L = lib()
    return dict(loaded=L is not None, simd=bool(L.tfs_simd()) if L is not None else None, threads=THREADS, gpu=GPU, why=_st['why'])


def pack_model(g):
    """IntTf -> (parameter vector, arrays, pointer array): the layout of tfs_model_create / tfg_model_create."""
    from ..features import EXP_LIM, EXP_TAB, LOGS_HI, LOGS_LO, NFEAT, RSQRT_TAB
    from ..model import AQ, H_LIM, HQ, LN_EPS_Q24, RSQ_BITS
    FF = g.layers[0]['W1'].shape[0]; FD = g.Wf.shape[0]
    P = [g.D, g.NL, g.NH, g.W, g.K, NFEAT, FF, FD, g.ws0, g.wsf, g.wsh, EXP_LIM, LOGS_LO, LOGS_HI, H_LIM, LN_EPS_Q24,
         RSQ_BITS, len(RSQRT_TAB), AQ, HQ]
    arrs = [g.W0, g.b0, g.Wf, g.bf, g.Wh, g.bh, EXP_TAB, RSQRT_TAB]
    for Ly in g.layers:
        if Ly['W1'].shape[0] != FF:
            raise ValueError('layers with different FFN widths')
        P += [Ly['wsq'], Ly['wso'], Ly['ws1'], Ly['ws2']]
        arrs += [Ly['Wqkv'], Ly['bqkv'], Ly['Wo'], Ly['bo'], Ly['W1'], Ly['b1'], Ly['W2'], Ly['b2'], Ly['rel']]
    P = np.ascontiguousarray(P, dtype=np.int64)
    arrs = [np.ascontiguousarray(np.asarray(a), dtype=np.int64) for a in arrs]
    ptrs = (ctypes.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
    return P, arrs, ptrs


class _Model:
    def __init__(self, g, L):
        self.L = L
        self._P, self._arrs, ptrs = pack_model(g)
        self.ptr = L.tfs_model_create(self._P.ctypes.data, ctypes.cast(ptrs, ctypes.c_void_p))
        if not self.ptr:
            raise RuntimeError('tfs_model_create failed')
        self.K = g.K
        self._fin = weakref.finalize(self, L.tfs_model_free, self.ptr)


def _model(g):
    L = lib()
    if L is None:
        return None
    m = g.__dict__.get('_c_model')
    if m is None:
        m = g._c_model = _Model(g, L)
    return m


class State:
    """The C caches of C curves after `pad` zero steps; step(x, act) -> (logw, mu, logs) [C, K] (the rows of act)."""

    def __init__(self, m, C, pad):
        self.m = m; self.C = int(C); self.t = 0
        self.ptr = m.L.tfs_state_create(m.ptr, self.C, int(pad))
        if not self.ptr:
            raise RuntimeError('tfs_state_create failed')
        self._fin = weakref.finalize(self, m.L.tfs_state_free, self.ptr)
        if THREADS > 1 and m.L.tfs_set_threads(self.ptr, THREADS, None):
            raise RuntimeError('tfs_set_threads failed')

    def free(self):
        self._fin()

    def step(self, x, act):
        C, K = self.C, self.m.K
        a = np.arange(C, dtype=np.int64) if act is None else np.ascontiguousarray(act, dtype=np.int64)
        xa = np.ascontiguousarray(np.asarray(x)[a], dtype=np.int64)
        out = np.empty((len(a), 3 * K), np.int64)
        rc = self.m.L.tfs_step(self.ptr, self.t, len(a), a.ctypes.data, xa.ctypes.data, out.ctypes.data)
        if rc:
            raise RuntimeError(f'tfs_step returned {rc} at t={self.t}')
        self.t += 1
        lw = np.zeros((C, K), np.int64); mu = np.zeros((C, K), np.int64); ls = np.zeros((C, K), np.int64)
        lw[a] = out[:, :K]; mu[a] = out[:, K:2 * K]; ls[a] = out[:, 2 * K:]
        return lw, mu, ls


def state(g, C, pad):
    m = _model(g)
    return None if m is None else State(m, C, pad)


def c_model(g, C, pad):
    """(fn address, ctx address, release) of a fresh step state for a C decode loop, or None."""
    if GPU is not None:
        from . import tfgpu
        return tfgpu.c_model(g, C, pad, GPU)
    m = _model(g)
    if m is None:
        return None
    st = State(m, C, pad)
    fn = ctypes.cast(m.L.tfs_step, ctypes.c_void_p).value

    def release(_st=st):
        _st.free()
    return fn, int(st.ptr), release


def pack_forward(g, st, n, n_max, G, R, Xl, pad):
    """The encoder's lockstep loop (codec._pack_residuals) in two C calls -> (logw, mu, logs [N, K], y [N]) in the loop's
    time-major order, or None."""
    from . import kernels
    m = _model(g)
    if m is None:
        return None
    fb = kernels.features_batch(st, n, n_max, G, R, Xl)
    if fb is None:
        return None
    X, y, _ = fb
    n = np.ascontiguousarray(n, np.int64); K = m.K
    out = np.empty((len(y), 3 * K), np.int64)
    rc = m.L.tfs_forward_lockstep(m.ptr, len(n), int(pad), n.ctypes.data, np.ascontiguousarray(X).ctypes.data, out.ctypes.data, THREADS, None)
    if rc:
        raise RuntimeError(f'tfs_forward_lockstep returned {rc}')
    return out[:, :K].copy(), out[:, K:2 * K].copy(), out[:, 2 * K:].copy(), y
