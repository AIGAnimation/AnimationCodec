"""The blob layout, the adaptive-symbol contexts and the coded-cost proxies.

Blob: b'D6' + version (1) + flags + LEB128 F + LEB128 B + precision code + LEB128 len(rans) + LEB128 len(raw) + rans
stream + raw bit stream. flags bit 0 = per-curve predictor ids on the all-keys tracks (else nlms for every curve).

Symbol order (the decoder reads in the same order):
  types     per bone the rotation type, then per bone the translation type
  steps     per animated track (rotations first) the grid index, Exp-Golomb delta to the previous one (raw)
  consts    the grid indices of the constant rotations (raw), then the constant values (magnitude classes + raw mantissas)
  x0        the first integers of every animated track
  modes     per animated track: static / all-keys / keyframes / per-component keyframes
  ids       the predictor ids
  gaps      per keyframe track the key gaps (- 1; escape + Exp-Golomb above 81), contexts = previous gap's bit length;
            per-component tracks: the gaps of the union key set + three presence flags per interior union key
  residuals every coded curve in lockstep (all curves' residual t, then t + 1, ...): the learned model's symbol + the
            raw low bits
"""
import numpy as np

from .features import PRECISIONS
from .rans import bit_length, entropy_bits, zigzag, unzigzag

MAGIC = b'D6'
VERSION = 1
NSYM = 83                    # alphabet of the adaptive symbols
ESC = NSYM - 1               # gap escape
# rotation / translation track types
T_ROT_DEF, T_ROT_CONST, T_ROT_ANIM = 0, 1, 2
T_TR_DEF, T_TR_OFF, T_TR_CONST, T_TR_ANIM = 0, 1, 2, 3
# track modes
M_STATIC, M_ALL, M_KF, M_CK = 0, 1, 2, 3
# contexts of the adaptive symbols
CT_TYPE_ROT, CT_TYPE_TRANS, CT_X0_ROT, CT_X0_TRANS, CT_MODE, CT_PID_ALL, CT_PID_KF = range(7)
NGAP = 10                    # gap contexts per kind: the first gap, then 1 + bit length of the previous gap - 1 (capped)
CT_GAP = 7                   # rotation gaps CT_GAP .. +9, translation gaps CT_GAP + NGAP .. +9
CT_FLAG = CT_GAP + 2 * NGAP  # 20 presence-flag contexts (Sink.put_flags)
NCTX = CT_FLAG + 20
FLAG_BITS = 0.9              # cost proxy of one presence flag


def init_counts():
    """The prior counts [NCTX, NSYM] of the adaptive models."""
    init = np.ones((NCTX, NSYM), dtype=np.int64)
    init[CT_TYPE_ROT, [0, 1, 2, 3]] += (16, 16, 64, 16)
    init[CT_TYPE_TRANS, [0, 1, 2, 3, 4]] += (16, 64, 16, 32, 16)
    init[CT_MODE, [M_STATIC, M_ALL, M_KF, M_CK]] += (16, 48, 48, 48)
    init[CT_PID_ALL, :3] += 32
    init[CT_PID_KF, :4] += 32
    init[CT_X0_ROT, :26] += 8
    init[CT_X0_TRANS, :26] += 8
    init[CT_GAP:CT_FLAG] += np.rint(96.0 * 0.75 ** np.arange(NSYM)).astype(np.int64)
    init[CT_FLAG:, :2] += 8
    return init


# ----------------------------------------------------------------------------------------------------------------
# small codes
# ----------------------------------------------------------------------------------------------------------------
def leb(v):
    v = int(v); out = bytearray()
    while True:
        b = v & 0x7F; v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def read_leb(buf, pos):
    v = 0; sh = 0
    while True:
        b = buf[pos]; pos += 1
        v |= (b & 0x7F) << sh; sh += 7
        if not b & 0x80:
            return v, pos


def prec_code(p):
    for i, q in enumerate(PRECISIONS):
        if abs(p - q) < 1e-12:
            return bytes([i])
    return bytes([255]) + np.float32(p).tobytes()


def read_prec(buf, pos):
    c = buf[pos]; pos += 1
    if c < 255:
        return PRECISIONS[c], pos
    return float(np.frombuffer(buf[pos:pos + 4], dtype=np.float32)[0]), pos + 4


def mag_split(u):
    """Magnitudes u >= 0 -> (class = bit length, mantissa below the leading one, mantissa width)."""
    u = np.asarray(u, dtype=np.int64)
    c = bit_length(u)
    w = np.maximum(c - 1, 0)
    return c, u & ((np.int64(1) << w) - 1), w


def mag_join(c, m):
    c = int(c)
    return (1 << (c - 1)) | int(m) if c > 0 else 0


def gap_contexts(gm, kind):
    """Contexts of a track's gap symbols gm = gap - 1: the first gap, then 1 + bit length of the previous gm (capped)."""
    gm = np.asarray(gm, dtype=np.int64)
    out = np.zeros(len(gm), dtype=np.int64)
    if len(gm) > 1:
        bl = np.zeros(len(gm) - 1, dtype=np.int64)
        v = gm[:-1].copy()
        while np.any(v > 0):
            bl += v > 0; v >>= 1
        out[1:] = 1 + np.minimum(bl, NGAP - 2)
    return CT_GAP + (0 if kind == 'rot' else NGAP) + out


class Sink:
    """The adaptive symbols (decode order) and the raw bits of one clip."""

    def __init__(self):
        from .rans import BitWriter
        self.sym = []; self.ctx = []
        self.raw = BitWriter()

    def put(self, s, c):
        self.sym.append(int(s)); self.ctx.append(int(c))

    def put_many(self, syms, ctxs):
        self.sym.extend(int(s) for s in syms); self.ctx.extend(int(c) for c in ctxs)

    def put_mags(self, n, ctx):
        """Signed integers: zigzag magnitude class as a symbol, the mantissa raw."""
        c, m, w = mag_split(zigzag(np.asarray(n, dtype=np.int64).ravel()))
        for ci, mi, wi in zip(c.tolist(), m.tolist(), w.tolist()):
            self.put(ci, ctx); self.raw.write(mi, wi)

    def put_gaps(self, K, kind):
        gm = np.diff(K) - 1
        for v, c in zip(gm.tolist(), gap_contexts(gm, kind).tolist()):
            if v < ESC:
                self.put(v, c)
            else:
                self.put(ESC, c); self.raw.write_exp_golomb(v - ESC, 0)

    def put_flags(self, Ks, U):
        """Presence flags of the three components on the interior keys of the union set U. Contexts: component 0 its
        previous flag (2); component 1 previous flag x f0 (4); component 2 previous flag x f0 x f1 (8); the flag of
        component 2 is implied (1) when f0 = f1 = 0."""
        sets = [np.isin(U, K) for K in Ks]
        prev = [1, 1, 1]
        for j in range(1, len(U) - 1):
            f = [int(sets[c][j]) for c in range(3)]
            self.put(f[0], CT_FLAG + prev[0])
            self.put(f[1], CT_FLAG + 4 + prev[1] * 2 + f[0])
            if f[0] or f[1]:
                self.put(f[2], CT_FLAG + 12 + prev[2] * 4 + f[0] * 2 + f[1])
            prev = f


def get_mags(dec, rd, n, ctx):
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c = dec.decode(ctx)
        out[i] = mag_join(c, rd.read(max(c - 1, 0)))
    return unzigzag(out)


# ----------------------------------------------------------------------------------------------------------------
# coded-cost proxies (bits) of the encoder's mode and predictor decisions
# ----------------------------------------------------------------------------------------------------------------
def cls_cost(cls, c0):
    n = len(cls)
    return 0.0 if n == 0 else n * entropy_bits(cls) + c0 * n


def res_proxy(r, c0):
    """Residuals -> bits: zero-order entropy of the magnitude classes + mantissa bits + c0 per symbol."""
    u = zigzag(np.asarray(r, dtype=np.int64).ravel())
    if len(u) == 0:
        return 0.0
    cat = bit_length(u)
    return cls_cost(cat, c0) + float(np.maximum(cat - 1, 0).sum())


def gap_proxy(gaps, c0):
    g = np.asarray(gaps, dtype=np.int64) - 1
    if len(g) == 0:
        return 0.0
    s = np.minimum(g, ESC)
    extra = float((2 * bit_length(g[g >= ESC] - ESC + 1) - 1).sum())
    return cls_cost(s, c0) + extra
