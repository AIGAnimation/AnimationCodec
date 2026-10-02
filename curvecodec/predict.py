"""Integer predictors of the coded curves. The coder codes residuals r = x - P(past); the learned model is the probability
model of r. Predictor ids:

  all-keys curves (every frame coded)   0 d1 (previous value), 1 d2 (linear extrapolation), 2 nlms
  keyframe curves (the values at keys)  0 d1, 1 d2, 2 d2t (d2 scaled by the ratio of the key gaps), 3 nlms

nlms = normalised LMS on the first differences (4 taps, integer weights in Q16, starts as d2). r[0] is the curve's
first value x0 (coded separately). The encoder computes residual arrays in bulk; the decoder rebuilds the integers one
step at a time for every curve in lockstep (`IncPred`), because the model's features read the decoded values.
"""
import numpy as np

from . import fast as _fast

NLMS_ORDER = 4
NLMS_MU = 5                 # step size 2^-NLMS_MU
NLMS_SH = 16                # fraction bits of the weights
NLMS_WMAX = 4 << NLMS_SH
NLMS_CLAMP = 1 << 20        # |e|, |d| clamp inside the weight update


def fixed_residuals(X, order):
    """r_t = the min(t, order)-th backward difference of X [F, C] (r_0 = 0)."""
    F = X.shape[0]
    R = np.zeros_like(X)
    lev = X
    for k in range(1, order + 1):
        if k >= F:
            break
        nxt = np.zeros_like(X)
        nxt[k:] = lev[k:] - lev[k - 1:-1]
        if k < order:
            R[k] = nxt[k]
        else:
            R[k:] = nxt[k:]
        lev = nxt
    return R


def nlms_residuals(X, x0):
    """NLMS residuals of the curves X [F, C] (int64; r_0 = 0)."""
    r = _fast.nlms_run(X, x0, NLMS_ORDER, NLMS_MU, False, 'd2')
    if r is not None:
        return r
    F, C = X.shape
    out = np.zeros_like(X)
    hist = np.zeros((C, NLMS_ORDER), dtype=np.int64)
    w = np.zeros((C, NLMS_ORDER), dtype=np.int64)
    w[:, 0] = 1 << NLMS_SH
    xprev = np.asarray(x0, dtype=np.int64).copy()
    half = 1 << (NLMS_SH - 1)
    sh = NLMS_SH - NLMS_MU
    for t in range(1, F):
        P = xprev + (((w * hist).sum(axis=1) + half) >> NLMS_SH)
        x = X[t]
        out[t] = x - P
        e = np.clip(x - P, -NLMS_CLAMP, NLMS_CLAMP)
        d = x - xprev
        hc = np.clip(hist, -NLMS_CLAMP, NLMS_CLAMP)
        norm = (hc * hc).sum(axis=1) + NLMS_ORDER
        w += ((e[:, None] * hc) << sh) // norm[:, None]
        np.clip(w, -NLMS_WMAX, NLMS_WMAX, out=w)
        hist[:, 1:] = hist[:, :-1]
        hist[:, 0] = d
        xprev = x
    return out


def nlms_batch(arrs):
    """NLMS residuals of a list of [n_i, 3] key-value arrays at once (zero-padded; the curves are independent)."""
    if not arrs:
        return []
    n = max(a.shape[0] for a in arrs)
    A = np.zeros((n, 3 * len(arrs)), dtype=np.int64)
    X0 = np.zeros(3 * len(arrs), dtype=np.int64)
    for j, a in enumerate(arrs):
        A[:a.shape[0], 3 * j:3 * j + 3] = a; X0[3 * j:3 * j + 3] = a[0]
    out = nlms_residuals(A, X0)
    return [out[:a.shape[0], 3 * j:3 * j + 3].copy() for j, a in enumerate(arrs)]


def key_residuals(nk, pred, K):
    """Key values nk int64 [n] or [n, 3] at the keys K -> residuals (r_0 = nk_0) of d1 (0) / d2 (1) / d2t (2)."""
    nk = np.asarray(nk, dtype=np.int64)
    r = np.diff(nk, axis=0, prepend=np.zeros((1,) + nk.shape[1:], dtype=np.int64))
    if len(nk) >= 3:
        if pred == 1:
            r[2:] = r[2:] - r[1:-1]
        elif pred == 2:
            g = np.diff(np.asarray(K, dtype=np.int64)).astype(np.float64)
            ratio = g[1:] / g[:-1]
            if nk.ndim == 2:
                ratio = ratio[:, None]
            r[2:] = r[2:] - np.rint(r[1:-1].astype(np.float64) * ratio).astype(np.int64)
    return r


class IncPred:
    """The predictors of C curves stepped in lockstep (decoder side): predict() -> P_t, update(P, x_new)."""

    def __init__(self, pid, kf, x0, gaps):
        C = len(pid)
        pid = np.asarray(pid, dtype=np.int64); kf = np.asarray(kf, dtype=np.int64)
        self.x = np.asarray(x0, dtype=np.int64).copy()
        self.xp = self.x.copy()
        self.t = 0
        self.gaps = gaps
        self.is_nlms = ((kf == 0) & (pid == 2)) | ((kf == 1) & (pid == 3))
        self.is_d2t = (kf == 1) & (pid == 2)
        self.is_d2 = pid == 1
        self.hist = np.zeros((C, NLMS_ORDER), dtype=np.int64)
        self.w = np.zeros((C, NLMS_ORDER), dtype=np.int64); self.w[:, 0] = 1 << NLMS_SH
        self.half = 1 << (NLMS_SH - 1); self.sh = NLMS_SH - NLMS_MU

    def predict(self):
        t = self.t
        P = self.x.copy()
        if t >= 1:
            d = self.x - self.xp
            P = np.where(self.is_d2, self.x + d, P)
            if np.any(self.is_d2t):
                g = self.gaps
                ratio = g[t].astype(np.float64) / np.maximum(g[t - 1], 1).astype(np.float64)
                P = np.where(self.is_d2t, self.x + np.rint(d.astype(np.float64) * ratio).astype(np.int64), P)
        Pn = self.x + (((self.w * self.hist).sum(axis=1) + self.half) >> NLMS_SH)
        return np.where(self.is_nlms, Pn, P)

    def update(self, P, x_new):
        e = np.clip(x_new - P, -NLMS_CLAMP, NLMS_CLAMP)
        d = x_new - self.x
        hc = np.clip(self.hist, -NLMS_CLAMP, NLMS_CLAMP)
        norm = (hc * hc).sum(axis=1) + NLMS_ORDER
        self.w += ((e[:, None] * hc) << self.sh) // norm[:, None]
        np.clip(self.w, -NLMS_WMAX, NLMS_WMAX, out=self.w)
        self.hist[:, 1:] = self.hist[:, :-1]; self.hist[:, 0] = d
        self.xp = self.x; self.x = x_new
        self.t += 1
