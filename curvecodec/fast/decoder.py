"""The decoder in C (decloop.c): the header after the type symbols, the coded columns and their static features, the
whole lockstep residual loop (features, the transformer step through a C hook, frequency tables, rANS, raw low bits,
predictors) and the reconstruction (dequantize, interpolate, exp_map with numpy's own sin / cos, float32 clip).
Python keeps the prefix, the grid steps and the constant tracks. Any C-level problem (a truncated stream, a value
outside the proven ranges) returns None and CurveCodec.decode decodes the blob from scratch in numpy.
"""
import ctypes

import numpy as np

from . import nploops
from ._build import build_lib

_st = {'lib': None, 'tried': False, 'why': '', 'expmap': None}
_vp = ctypes.c_void_p
_i64 = ctypes.c_int64


def _p(a):
    return a.ctypes.data


def _c64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def _declare(L):
    def d(name, res, args):
        f = getattr(L, name); f.restype = res; f.argtypes = args
    d('dl_open', _vp, [_vp, _i64, _vp, _i64, _i64, _i64, _i64, _i64, _i64, _vp, _i64])
    d('dl_close', None, [_vp])
    d('dl_syms', ctypes.c_int, [_vp, _i64, _i64, _vp])
    d('dl_mix_create', _vp, [_i64, _vp, _vp, _vp, _vp, _i64, _vp, _i64, _i64, _vp, _i64, _i64, _i64, _vp])
    d('dl_v2_new', _vp, [])
    d('dl_v2_free', None, [_vp])
    d('dl_v2_header', _i64, [_vp, _vp, _i64, _i64, _vp, _vp, _vp])
    d('dl_v2_sizes', None, [_vp, _vp])
    d('dl_v2_arrays', None, [_vp] + [_vp] * 8)
    d('dl_v2_columns', _i64, [_vp, _vp, _i64, _i64, _i64, _i64, _vp, _vp])
    d('dl_v2_lockstep', _i64, [_vp, _vp, _vp, _i64, _vp, _vp, _i64, _i64, _i64, _i64, _i64])
    d('dl_v2_interp', _i64, [_vp, _i64, _i64, _i64, _vp, _vp, _vp])
    d('dl_v2_write_rot', None, [_vp, _i64, _i64, _vp, _vp, _vp, _vp, _vp])
    d('dl_v2_write_trans', None, [_vp, _i64, _i64, _vp, _vp])
    d('dl_v2_write_rot_np', _i64, [_vp, _i64, _i64, _vp, ctypes.c_int, _vp, _vp, _vp, _vp, _vp])
    d('dl_norm3', None, [_vp, _i64, ctypes.c_int, _vp])


def _expmap_selftest(L):
    """exp_map in C: the norm's summation order that matches np.linalg.norm, and numpy's own sin / cos loops ->
    (norm variant, loop pointers) or None (-> numpy's norm / sin / cos, C for the rest)."""
    try:
        rng = np.random.default_rng(20260928)
        V = np.ascontiguousarray(np.concatenate([rng.normal(0, 1, (100000, 3)), rng.normal(0, 1e-3, (40000, 3)) * rng.uniform(0, 1e4, (40000, 1)),
                                                 np.exp(rng.uniform(-20, 20, (20000, 3)))]))
        ref = np.linalg.norm(V, axis=-1)
        for var in (0, 1):
            o = np.empty(len(V))
            L.dl_norm3(_p(V), len(V), var, _p(o))
            if np.array_equal(o, ref):
                lp = nploops.loops(('sin', 'cos'))
                if lp is None:
                    return None
                return var, (lp['sin'][0], lp['sin'][1] or None, lp['cos'][0], lp['cos'][1] or None)
        return None
    except Exception:  # noqa: BLE001
        return None


def lib():
    if _st['lib'] is None and not _st['tried']:
        _st['tried'] = True
        try:
            L = build_lib('decoder', ['decloop.c'])
            _declare(L)
            _st['expmap'] = _expmap_selftest(L)
            _st['lib'] = L
        except Exception as e:  # noqa: BLE001
            _st['why'] = f'{type(e).__name__}: {e}'
            import warnings
            warnings.warn(f'curvecodec.fast: C decoder unavailable, using numpy ({_st["why"][:300]})')
    return _st['lib']


def status():
    return dict(loaded=lib() is not None, expmap='numpy loops' if _st['expmap'] else 'numpy', why=_st['why'])


class _Fallback(Exception):
    """A condition the numpy decoder handles."""


_MIX = {}


def _mix(L):
    """The alphabet and mixture tables in C (kept alive here: the C side holds pointers)."""
    if 'm' not in _MIX:
        from .. import features as FT
        lo, hi = FT.ALPHABET
        arrs = (_c64(lo), _c64(hi), _c64(FT.WIDTHS)) + tuple(_c64(a) for a in (FT.EXP_TAB, FT.INVS_TAB, FT.SIG_TAB, FT.LOG_TAB))
        ptr = L.dl_mix_create(len(arrs[0]), _p(arrs[0]), _p(arrs[1]), _p(arrs[2]), _p(arrs[3]), FT.EXP_LIM, _p(arrs[4]), FT.LOGS_LO,
                              FT.LOGS_HI, _p(arrs[5]), FT.SIG_LIM, FT.MU_LIM, FT.FREQ_BITS, _p(arrs[6]))
        if not ptr:
            raise _Fallback('mix tables')
        _MIX['m'] = (ptr, arrs)
    return _MIX['m'][0]


_CB = ctypes.CFUNCTYPE(ctypes.c_int, _vp, _i64, _i64, _vp, _vp, _vp)


class _PyStep:
    """The numpy model step behind the C hook (when the C step is unavailable)."""

    def __init__(self, model, C, K):
        self.model = model; self.K = K
        self.X = np.zeros((C, 36), np.int64)
        self.exc = None
        self.cfn = _CB(self._cb)

    def _cb(self, ctx, t, n_act, act_p, x_p, out_p):
        try:
            K = self.K
            act = np.frombuffer((ctypes.c_int64 * n_act).from_address(act_p), np.int64).copy()
            self.X[act] = np.frombuffer((ctypes.c_int64 * (n_act * 36)).from_address(x_p), np.int64).reshape(n_act, 36)
            lw, mu, ls = self.model.step(self.X, act)
            out = np.frombuffer((ctypes.c_int64 * (n_act * 3 * K)).from_address(out_p), np.int64).reshape(n_act, 3, K)
            out[:, 0] = lw[act]; out[:, 1] = mu[act]; out[:, 2] = ls[act]
            return 0
        except BaseException as e:  # noqa: BLE001 -- never let an exception cross the C frame
            self.exc = e
            return -3


def decode(codec, blob, parents, offsets, fps):
    """CurveCodec.decode in C -> float32 [F, B, 7], or None."""
    L = lib()
    if L is None:
        return None
    from ..container import MAGIC, VERSION, read_leb, read_prec
    if blob[:2] != MAGIC or blob[2] != VERSION or blob[3] & ~1:
        return None
    flags = blob[3]; pos = 4
    F, pos = read_leb(blob, pos); B, pos = read_leb(blob, pos)
    p, pos = read_prec(blob, pos)
    n_rans, pos = read_leb(blob, pos); n_raw, pos = read_leb(blob, pos)
    rans = blob[pos:pos + n_rans]; pos += n_rans
    rawb = bytes(blob[pos:pos + n_raw])
    frees = []
    try:
        return _decode(L, codec, F, B, p, bool(flags & 1), rans, rawb, parents, offsets, fps, frees)
    except _Fallback:
        return None
    finally:
        for free, c in reversed(frees):
            free() if c is None else free(c)


def _decode(L, codec, F, B, p, use_ids, rans, rawb, parents, offsets, fps, frees):
    from .. import container as CT
    from ..features import LOG_TAB, STEP_COL, prec_index
    from ..metric import bone_depth, depth_class
    from ..model import PAD
    from ..predict import NLMS_CLAMP, NLMS_MU, NLMS_ORDER, NLMS_SH, NLMS_WMAX
    from ..quantize import SQRT3
    from ..rans import ADAPT_RATE, MODEL_INC, RANS_L, RANS_SCALE_BITS
    from ..rot import exp_map
    grid, cg = codec.grid, codec.cgrid
    init = _c64(codec.init)
    rb = np.frombuffer(bytes(rans), np.uint8); wb = np.frombuffer(bytes(rawb), np.uint8)
    h = L.dl_open(_p(rb) if len(rb) else None, len(rb), _p(init), init.shape[0], init.shape[1], RANS_SCALE_BITS, RANS_L, MODEL_INC,
                  ADAPT_RATE, _p(wb) if len(wb) else None, len(wb))
    if not h:
        raise _Fallback('dl_open')
    frees.append((L.dl_close, h))

    def syms(ctx, n):
        out = np.empty(n, np.int64)
        if n and L.dl_syms(h, int(ctx), n, _p(out)):
            raise _Fallback('header stream')
        return out
    rts_a = syms(CT.CT_TYPE_ROT, B); tts_a = syms(CT.CT_TYPE_TRANS, B)
    rts = rts_a.tolist(); tts = tts_a.tolist()
    if any(t not in (CT.T_ROT_DEF, CT.T_ROT_CONST, CT.T_ROT_ANIM) for t in rts) or \
            any(t not in (CT.T_TR_DEF, CT.T_TR_OFF, CT.T_TR_CONST, CT.T_TR_ANIM) for t in tts):
        raise _Fallback('unsupported track type')
    # header layout for the C side (the constant-rotation rig type and the per-component index context are unused: -1)
    cfg = np.array([CT.T_ROT_ANIM, CT.T_TR_ANIM, CT.T_ROT_CONST, -1, CT.T_TR_CONST, grid.first['rot'], grid.first['trans'], grid.eg_k,
                    0, -1, cg.first['crot'], cg.eg_k, CT.CT_X0_ROT, CT.CT_X0_TRANS, CT.CT_MODE, CT.CT_PID_ALL, CT.CT_PID_KF,
                    int(use_ids), CT.M_ALL, CT.M_KF, CT.M_CK, CT.M_STATIC, CT.CT_GAP, CT.NGAP, CT.NGAP - 2, CT.ESC, CT.CT_FLAG], np.int64)
    v = L.dl_v2_new()
    if not v:
        raise _Fallback('dl_v2_new')
    frees.append((L.dl_v2_free, v))
    if L.dl_v2_header(v, h, F, B, _p(rts_a), _p(tts_a), _p(cfg)):
        raise _Fallback('header')
    sz = np.zeros(5, np.int64); L.dl_v2_sizes(v, _p(sz))
    ntr, nrot = int(sz[0]), int(sz[1])
    bone = np.zeros(ntr, np.int64); I3 = np.zeros((ntr, 3), np.int64); x0 = np.zeros((ntr, 3), np.int64); mode = np.zeros(ntr, np.int64)
    pid = np.zeros((ntr, 3), np.int64); cidx = np.zeros(B, np.int64); crot = np.zeros((B, 3), np.int64); ctr = np.zeros((B, 3), np.int64)
    L.dl_v2_arrays(v, _p(bone), _p(I3), _p(x0), _p(mode), _p(pid), _p(cidx), _p(crot), _p(ctr))
    steps = grid.step(I3) if ntr else np.zeros((0, 3), np.float32)
    # the constant tracks (as CurveCodec.decode: quantize.const_rot / const_trans, batched)
    offsets = np.asarray(offsets, dtype=np.float32)
    T = np.zeros((B, 7), np.float32); T[:, 3] = 1.0
    rc_b = [b for b in range(B) if rts[b] == CT.T_ROT_CONST]
    if rc_b:
        st = cg.step(cidx[rc_b])
        vk = (crot[rc_b].astype(np.float64) * st.astype(np.float64)[:, None]).astype(np.float32)
        T[rc_b, :4] = exp_map(vk.astype(np.float64)).astype(np.float32)
    tc_b = [b for b in range(B) if tts[b] == CT.T_TR_CONST]
    if tc_b:
        T[tc_b, 4:7] = (ctr[tc_b].astype(np.float64) * np.float64(cg.step(cg.floor(p * codec.CFRAC / SQRT3)))).astype(np.float32)
    for b in range(B):
        if tts[b] == CT.T_TR_OFF:
            T[b, 4:7] = offsets[b]
    # the columns and the lockstep loop
    dcb = _c64(depth_class(bone_depth(np.asarray(parents))))
    lt = _c64(LOG_TAB)
    C = L.dl_v2_columns(v, _p(dcb), grid.G, int(round(fps)), prec_index(p), STEP_COL, _p(lt), _p(cfg))
    if C < 0:
        raise _Fallback('columns')
    if C > 0:
        from . import tfstep
        model = codec.model
        K = int(model.K)
        cm = tfstep.c_model(model, C, PAD)
        shim = None
        if cm is not None:
            fn, ctx, release = cm
            frees.append((release, None))
        else:
            model.start(C, PAD)
            shim = _PyStep(model, C, K)
            fn = ctypes.cast(shim.cfn, _vp).value; ctx = None
        rc = L.dl_v2_lockstep(v, h, _mix(L), K, fn, ctx, NLMS_ORDER, NLMS_SH, NLMS_MU, NLMS_CLAMP, NLMS_WMAX)
        if shim is not None and shim.exc is not None:
            raise shim.exc
        if rc:
            raise _Fallback(f'lockstep {rc}')
    # the reconstruction, per chunk of tracks: interpolation (C), exp_map and the float32 clip (C, numpy's sin / cos)
    loc = np.ascontiguousarray(np.broadcast_to(T, (F, B, 7)))
    stp = np.ascontiguousarray(steps, dtype=np.float32)
    em = _st['expmap']
    CH = max(1, min(64, (1 << 19) // max(F, 1)))
    for k0, k1, rot in ((0, nrot, True), (nrot, ntr, False)):
        for t0 in range(k0, k1, CH):
            t1 = min(t0 + CH, k1)
            V = np.empty((t1 - t0, F, 3))
            if L.dl_v2_interp(v, t0, t1, 1, _p(stp), _p(cfg), _p(V)):
                raise _Fallback('interp')
            if rot and em is not None:
                if L.dl_v2_write_rot_np(v, t0, t1, _p(V), int(em[0]), *em[1], _p(loc)):
                    raise _Fallback('exp_map')
            elif rot:
                theta = np.ascontiguousarray(np.linalg.norm(V, axis=-1))
                half = 0.5 * theta
                L.dl_v2_write_rot(v, t0, t1, _p(V), _p(theta), _p(np.sin(half)), _p(np.cos(half)), _p(loc))
            else:
                L.dl_v2_write_trans(v, t0, t1, _p(V), _p(loc))
    return loc
