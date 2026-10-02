"""Integer features of the coded residuals and the model's probability tables (identical in the encoder, the decoder
and the training data).

Everything is integer arithmetic on int64 arrays (look-up tables shipped as data), so the results are bit-exact on
every platform. Log-domain quantities are Q8 fixed point (256 = one octave).

Per coded curve (a column: one component of a track, its residuals in time order) the state holds
  hist   the last 8 residuals                                  -> slog2q(r) = sign(r) L(|r|), L(v) = round(256 log2(1 + v))
  ewma   EWMA of |r| for tau = 2, 4, 8, 16, 32 (Q8)              -> log2(1 + m / 256) in Q8
  vel    the last 3 first differences and 2 second differences of the decoded integers (slog2q)
  gaps   keyframe curves: log2 of the gap before / after the key and their difference (0 for all-keys curves)
  static depth class one-hot (4), kind, keyframe flag, predictor id one-hot (4), log2 of the grid step (Q8), L(fps),
         precision index, L(t), L(curve length)
NFEAT = 36. The model maps a feature vector to a K-component logistic mixture (logw, mu, logs in Q8); its CDF at the
symbol boundaries gives the frequency table (sum 2^15, every symbol >= 1) of the sign-magnitude alphabet: symbol 0 for
r = 0, else (sign, bit length c of |r|, the bit below the leading one); the remaining c - 2 low bits are coded raw.
"""
import math
import os

import numpy as np

from . import fast as _fast
from .rans import bit_length

H = 8                               # residual history
TAUS_SH = (1, 2, 3, 4, 5)           # EWMA time constants 2 .. 32
NFEAT = H + len(TAUS_SH) + 5 + 3 + 15
STEP_COL = H + len(TAUS_SH) + 5 + 3 + 10
FEAT_NAMES = ([f'h{i}' for i in range(1, H + 1)] + [f'ewma{1 << s}' for s in TAUS_SH] + ['v1', 'v2', 'v3', 'a1', 'a2'] + ['gb', 'ga', 'gr'] +
              ['dc0', 'dc1', 'dc2', 'dc3', 'kind', 'kf', 'pid0', 'pid1', 'pid2', 'pid3', 'step', 'fps', 'p', 'pos', 'len'])
PRECISIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0)     # the precision ladder (its index is a feature and the blob's code)
SIG_LIM = 16 * 256                  # sigmoid argument clip (Q8)
LOGS_LO, LOGS_HI = -6 * 256, 12 * 256
EXP_LIM = 16 * 256
MU_LIM = 1 << 30                    # |256 r - mu| clip before the multiply by inv_s
FREQ_BITS = 15
MANTISSA_BITS = 1                   # modelled bits below the leading one
TABLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tables.npz')


def compute_tables():
    """The shipped look-up tables from libm (used to verify the shipped file; a libm rounding difference could change
    an entry between platforms, so the codec always reads the file): LOG_TAB[f] = round(256 log2(1 + f / 256)),
    SIG_TAB[z + SIG_LIM] = round(2^16 sigmoid(z / 256)), INVS_TAB[v - LOGS_LO] = round(2^16 exp(-v / 256)),
    EXP_TAB[v + EXP_LIM] = round(2^16 exp(v / 256)), RSQRT_TAB[i] = round(2^16 / sqrt((1024 + i) / 1024))."""
    return dict(LOG_TAB=np.array([int(round(256.0 * math.log2(1.0 + f / 256.0))) for f in range(256)], dtype=np.int64),
                SIG_TAB=np.array([int(round(65536.0 / (1.0 + math.exp(-z / 256.0)))) for z in range(-SIG_LIM, SIG_LIM + 1)], dtype=np.int64),
                INVS_TAB=np.array([int(round(65536.0 * math.exp(-v / 256.0))) for v in range(LOGS_LO, LOGS_HI + 1)], dtype=np.int64),
                EXP_TAB=np.array([int(round(65536.0 * math.exp(v / 256.0))) for v in range(-EXP_LIM, 1)], dtype=np.int64),
                RSQRT_TAB=np.array([int(round(65536.0 / math.sqrt((1024 + i) / 1024.0))) for i in range(3 * 1024)], dtype=np.int64))


_T = np.load(TABLES_FILE)
LOG_TAB, SIG_TAB, INVS_TAB, EXP_TAB, RSQRT_TAB = (_T[k].astype(np.int64) for k in ('LOG_TAB', 'SIG_TAB', 'INVS_TAB', 'EXP_TAB', 'RSQRT_TAB'))


def prec_index(p):
    for i, q in enumerate(PRECISIONS):
        if abs(p - q) < 1e-12:
            return i
    return len(PRECISIONS)


def ilog2_q8(v):
    """round(256 log2 v) for integers v >= 1 (bit length + fractional table)."""
    v = np.maximum(np.asarray(v, dtype=np.int64), 1)
    e = bit_length(v) - 1
    f = np.where(e >= 8, v >> np.maximum(e - 8, 0), v << np.maximum(8 - e, 0)) - 256
    return e * 256 + LOG_TAB[np.clip(f, 0, 255)]


def L(v):
    """round(256 log2(1 + v)) for integers v >= 0."""
    return ilog2_q8(np.asarray(v, dtype=np.int64) + 1)


def slog2q(x):
    x = np.asarray(x, dtype=np.int64)
    return np.sign(x) * L(np.abs(x))


class CurveState:
    """Feature state of C curves coded in lockstep (one time step for every curve at once). Arrays int64 [C] / [C, k].
    kind 0 = rotation / 1 = translation, dc = depth class, kf = keyframe curve, pid = predictor id, step_q8 = log2 of
    the grid step in Q8, n_len = residuals per curve, gaps [n_max, C] = key gap before residual t (keyframe curves)."""

    def __init__(self, kind, dc, kf, pid, step_q8, fps_i, p_idx, n_len, gaps):
        C = len(kind)
        self.C = C
        self.hist = np.zeros((C, H), dtype=np.int64)
        self.ew = np.full((C, len(TAUS_SH)), 256, dtype=np.int64)
        self.d = np.zeros((C, 3), dtype=np.int64)
        self.a = np.zeros((C, 2), dtype=np.int64)
        self.t = 0
        base = np.zeros((C, NFEAT), dtype=np.int64)
        col = H + len(TAUS_SH) + 5 + 3
        base[np.arange(C), col + np.asarray(dc)] = 256
        base[:, col + 4] = 256 * np.asarray(kind)
        base[:, col + 5] = 256 * np.asarray(kf)
        base[np.arange(C), col + 6 + np.clip(np.asarray(pid), 0, 3)] = 256
        base[:, col + 10] = np.asarray(step_q8)
        base[:, col + 11] = L(np.full(C, int(fps_i)) - 1)
        base[:, col + 12] = 256 * int(p_idx)
        base[:, col + 14] = L(np.asarray(n_len) - 1)
        self.base = base
        self.lg = L(np.maximum(np.asarray(gaps, dtype=np.int64), 1) - 1)          # log2(gap) in Q8, 0 where unused

    def start(self, x0):
        self._x_prev = np.asarray(x0, dtype=np.int64).copy()

    def features(self):
        """[C, NFEAT] features of the residual at the current step (from the past only)."""
        F = self.base.copy()
        F[:, :H] = slog2q(self.hist)
        F[:, H:H + len(TAUS_SH)] = ilog2_q8(256 + self.ew) - 8 * 256
        c = H + len(TAUS_SH)
        F[:, c:c + 3] = slog2q(self.d); F[:, c + 3:c + 5] = slog2q(self.a)
        c += 5
        t = self.t; n = self.lg.shape[0]
        gb = self.lg[min(t, n - 1)]; ga = self.lg[min(t + 1, n - 1)]
        gp = self.lg[max(t - 1, 0)] if t > 0 else gb
        F[:, c] = gb; F[:, c + 1] = ga; F[:, c + 2] = gb - gp
        F[:, c + 3 + 13] = L(np.full(self.C, self.t))
        return F

    def update(self, r, x_new):
        """After the residual r [C] and the decoded integer x_new [C] of every curve."""
        u = np.abs(r)
        self.hist[:, 1:] = self.hist[:, :-1]; self.hist[:, 0] = r
        for j, sh in enumerate(TAUS_SH):
            self.ew[:, j] += ((u << 8) - self.ew[:, j]) >> sh
        d_new = x_new - self._x_prev
        self.a[:, 1] = self.a[:, 0]; self.a[:, 0] = d_new - self.d[:, 0]
        self.d[:, 1:] = self.d[:, :-1]; self.d[:, 0] = d_new
        self._x_prev = x_new
        self.t += 1


# ----------------------------------------------------------------------------------------------------------------
# the mixture CDF and the frequency tables
# ----------------------------------------------------------------------------------------------------------------
def mixture_weights_q16(logw):
    """Softmax over the K components in Q16 (sums exactly to 65536, the remainder to the largest component)."""
    z = logw - logw.max(axis=1, keepdims=True)
    e = EXP_TAB[np.clip(z, -EXP_LIM, 0) + EXP_LIM]
    tot = e.sum(axis=1, keepdims=True)
    w = (e << 16) // tot
    rem = 65536 - w.sum(axis=1)
    w[np.arange(len(w)), np.argmax(e, axis=1)] += rem
    return w


def cdf_q16(logw, mu, logs, pts):
    """P(R <= r) in Q16 at the integer points pts [N, P] -> int64 [N, P]."""
    w = mixture_weights_q16(logw)
    inv_s = INVS_TAB[logs - LOGS_LO]
    d = np.clip((pts << 8) + 128, -MU_LIM, MU_LIM)
    z = np.clip(d[:, :, None] - mu[:, None, :], -MU_LIM, MU_LIM)
    z = np.clip((z * inv_s[:, None, :]) >> 16, -SIG_LIM, SIG_LIM)
    s = SIG_TAB[z + SIG_LIM]
    return (s * w[:, None, :]).sum(axis=2) >> 16


def symbol_freqs(logw, mu, logs, alphabet):
    """Frequency tables int64 [N, S] (sum 2^15, every symbol >= 1) of the alphabet's r-intervals [lo, hi]."""
    r = _fast.symbol_freqs(logw, mu, logs, alphabet)
    if r is not None:
        return r
    lo, hi = alphabet
    N = len(logw); S = len(lo)
    pts = np.concatenate([hi, lo - 1])[None, :].repeat(N, axis=0)
    F = cdf_q16(logw, mu, logs, pts)
    P = np.maximum(F[:, :S] - F[:, S:], 0)
    tot = P.sum(axis=1, keepdims=True)
    M = 1 << FREQ_BITS
    f = 1 + (P * (M - S)) // np.maximum(tot, 1)
    f[:, 0] += M - f.sum(axis=1)
    return f


# ----------------------------------------------------------------------------------------------------------------
# the sign-magnitude alphabet (159 symbols)
# ----------------------------------------------------------------------------------------------------------------
def make_alphabet(mb=MANTISSA_BITS):
    """((lo, hi) int64 r-intervals per symbol, raw low-bit widths): symbol 0 = r 0, then per sign the classes 1..40
    (bit length of |r|), classes <= mb enumerated, larger ones split by the mb bits below the leading one."""
    rows = [(0, 0, 0)]
    for sign in (1, -1):
        for c in range(1, 41):
            if c <= mb:
                for m in range(1 << (c - 1), 1 << c):
                    rows.append((sign * m, sign * m, 0))
                continue
            base = 1 << (c - 1); sub = 1 << (c - 1 - mb)
            for top in range(1 << mb):
                a = base + top * sub; b = a + sub - 1
                rows.append((a, b, c - 1 - mb) if sign > 0 else (-b, -a, c - 1 - mb))
    lo = np.array([r[0] for r in rows], np.int64); hi = np.array([r[1] for r in rows], np.int64)
    return (lo, hi), np.array([r[2] for r in rows], np.int64)


def symbol_index(r, mb=MANTISSA_BITS):
    """Residuals -> symbol index of make_alphabet(mb); the raw low bits are |r| & (2^width - 1)."""
    r = np.asarray(r, dtype=np.int64); m = np.abs(r)
    c = np.zeros_like(m); v = m.copy()
    while np.any(v > 0):
        c += v > 0; v >>= 1
    per_sign = sum((1 << (k - 1)) if k <= mb else (1 << mb) for k in range(1, 41))
    base = np.zeros(42, dtype=np.int64); cnt = 0
    for k in range(1, 41):
        base[k] = cnt; cnt += (1 << (k - 1)) if k <= mb else (1 << mb)
    top = np.where(c <= mb, m - (np.int64(1) << np.maximum(c - 1, 0)), (m >> np.maximum(c - 1 - mb, 0)) & ((1 << mb) - 1))
    idx = 1 + np.where(r < 0, per_sign, 0) + base[c] + top
    return np.where(r == 0, 0, idx)


def symbol_value(s, low, lo, hi):
    """(symbol index, raw low bits) -> residual r."""
    a = int(lo[s])
    return a + low if a >= 0 else int(hi[s]) - low


ALPHABET, WIDTHS = make_alphabet()
