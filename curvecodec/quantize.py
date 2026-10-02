"""Quantization: the log2 step grids, uniform quantizers of the track values, ACL's constant / default folding rule and
the constant tracks.

A rotation track is coded in its log-map representation (rotation vector of the hemisphere-continuous quaternion, rad),
a translation track as its raw vector (cm); both are quantized per component with a step from a log2 grid.
"""
import numpy as np

from .rot import exp_map, log_map, quat_cols, quat_normalize

SQRT3 = np.sqrt(3.0)


class Grid:
    """Log2 grid with G steps per octave: step(i) = float32(2^(i / G)), computed as ldexp(frac[k], q) with i = q G + k
    so that it is platform-independent. `first` = the delta-coding start of the indices, `eg_k` = the Exp-Golomb order
    of the index deltas, `idx_min` = the smallest animated step index."""

    def __init__(self, G):
        self.G = int(G)
        self.unit = max(1, self.G // 8)                         # one eighth of an octave in grid units
        self.frac = tuple(float(f'{2.0 ** (k / self.G):.15g}') for k in range(self.G))
        self.first = {'rot': -8 * self.G, 'trans': -6 * self.G, 'crot': -10 * self.G}
        self.eg_k = 2 + int(round(np.log2(self.G / 8)))
        self.idx_min = -28 * self.G

    def step1(self, i):
        q, k = divmod(int(i), self.G)
        return np.float32(np.ldexp(self.frac[k], q))

    def step(self, i):
        i = np.asarray(i)
        if i.ndim == 0:
            return self.step1(i)
        return np.array([self.step1(j) for j in i.ravel().tolist()], dtype=np.float32).reshape(i.shape)

    def floor(self, s):
        """The largest index whose step is <= s (as float32)."""
        s32 = np.float32(s)
        i = int(np.floor(self.G * np.log2(float(s))))
        while self.step1(i) > s32:
            i -= 1
        while self.step1(i + 1) <= s32:
            i += 1
        return i

    def caps_up(self, top=16):
        return tuple(int(2 ** k) for k in range(int(np.log2(top * self.unit)) + 1))

    def caps_down(self):
        base = (-1, -2, -3, -4, -6, -8, -12, -16, -24, -32, -48, -64)
        return tuple(sorted(set(base) | {e * self.unit for e in base}, reverse=True))


def idx3(i):
    """The three per-component grid indices of a track (all equal)."""
    return np.full(3, int(i), dtype=np.int64)


def quantize(vals, step):
    """vals float64 [n, 3], step float32 (scalar or [3]) -> int64 [n, 3]."""
    return np.rint(vals / np.asarray(step, dtype=np.float64)).astype(np.int64)


def dequantize(n, step):
    """int64 [n, 3] x float32 step -> float32 [n, 3] (the product in float64, rounded once)."""
    return (n.astype(np.float64) * np.asarray(step, dtype=np.float64)).astype(np.float32)


def fold_types(local, precision, R_dom):
    """ACL's constant / default sub-track rule: a rotation (translation) track is default when the identity (zero) keeps
    the local shell error at the dominant radius R_dom[b] (the translation error) within `precision` on every frame,
    constant when its first sample does. -> (rot types [B], trans types [B]): 0 default, 1 constant, 2 animated."""
    local = np.asarray(local)
    F, B, _ = local.shape
    rt = np.full(B, 2, dtype=np.int64); tt = np.full(B, 2, dtype=np.int64)
    ident_cols = np.eye(3)[None]
    for b in range(B):
        q = quat_normalize(local[:, b, :4].astype(np.float64)); t = local[:, b, 4:7].astype(np.float64)
        cols = quat_cols(q)

        def rot_err(ref_cols):
            return R_dom[b] * np.sqrt(np.max(np.sum((cols - ref_cols) ** 2, axis=-1), axis=-1))
        if np.all(rot_err(ident_cols) <= precision):
            rt[b] = 0
        elif np.all(rot_err(cols[:1]) <= precision):
            rt[b] = 1
        if np.all(np.linalg.norm(t, axis=-1) <= precision):
            tt[b] = 0
        elif np.all(np.linalg.norm(t - t[:1], axis=-1) <= precision):
            tt[b] = 1
    return rt, tt


def const_rot_ints(q0, step):
    """First-sample quaternion (float64 [4], unit) -> its log-map integers [3] at `step`."""
    return quantize(log_map(np.asarray(q0, dtype=np.float64)[None]), step)[0]


def const_rot(n, step):
    return exp_map(dequantize(np.asarray(n, dtype=np.int64)[None], step).astype(np.float64))[0].astype(np.float32)


def const_trans(n, step):
    return dequantize(np.asarray(n, dtype=np.int64)[None], step)[0]
