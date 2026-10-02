"""The integer transformer: the probability model of every coded residual, bit-exact on every platform.

A causal transformer over the 36 integer features of each residual (features.py), one position per residual of a coded
curve, sliding window of W = 128 positions, relative-position attention bias, pre-LayerNorm blocks, a K = 3 logistic
mixture head (logw, mu, logs in Q8). The float model (train/net.py) is exported once to fixed point:

  x        Q8 integer features
  embed    int16 W0 with the input normalisation folded in, ReLU, Q8 -> the residual stream h in Q12 (h = e << 4),
           |h| clipped to 2^28 (every later product < 2^43, every partial sum < 2^53: exact in float64 BLAS)
  LN       mean = sum(h) >> log2 d; c = h - mean; var = sum(c^2) >> log2 d (Q24) + eps; 1 / sqrt(var) from a 1024-entry
           Q16 table over the mantissa (exact power-of-two shifts); a = (c x inv_std) >> 16 (Q12); gamma / beta are
           folded into the next linear layer
  attn     q, k, v = int16 Wqkv . a -> Q8 (1 / sqrt(dh) folded into Wq); logits = (q . k) >> 8 + rel[head, offset] (Q8);
           softmax over the <= W cached positions with the Q16 exp table, p = (e << 16) // sum e; o = (sum_w p v) >> 16;
           h += (Wo o + bo) >> (ws - 4)
  ffn      u = relu((W1 LN2(h) + b1) >> (ws + 4)) (Q8); h += (W2 u + b2) >> (ws - 4)
  head     f = relu((Wf h + bf) >> (ws + 4)) (Q8); o = (Wh f + bh) >> ws (Q8) -> (logw, mu, logs)
Every operation is an integer operation on int64 (or an exact integer product through float64 BLAS). The KV cache of a
fresh curve is the cache after `pad` zero-feature steps (the training convention).
"""
import math
import os

import numpy as np

from . import fast as _fast
from .features import EXP_LIM, EXP_TAB, LOGS_HI, LOGS_LO, NFEAT, RSQRT_TAB

AQ = 8                      # Q8 activations (q, k, v, FFN hidden, final)
HQ = 12                     # residual stream and LayerNorm output
H_LIM = 1 << 28             # residual stream clip
LN_EPS_Q24 = 168            # 1e-5 in Q24
RSQ_BITS = 10               # mantissa resolution of the reciprocal square-root table
PAD = 128                   # zero-feature steps before the first residual of every curve
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'model_int.npz')


def _bitlen(v):
    v = np.asarray(v, np.int64)
    out = np.zeros(v.shape, np.int64)
    x = v.copy()
    for s in (32, 16, 8, 4, 2, 1):
        m = x >= (np.int64(1) << s)
        out[m] += s; x[m] >>= s
    return out + (v > 0)


def inv_sqrt_q16_of_q24(v):
    """1 / sqrt(v / 2^24) in Q16 for int64 v >= 1: mantissa table x exact power-of-two shift."""
    v = np.maximum(np.asarray(v, np.int64), 1)
    e = _bitlen(v) - 1
    e2 = e & ~np.int64(1)
    sh = e2 - RSQ_BITS
    m = np.where(sh >= 0, v >> np.maximum(sh, 0), v << np.maximum(-sh, 0))
    r = RSQRT_TAB[m - (1 << RSQ_BITS)]
    k = 12 - (e2 >> 1)
    return np.where(k >= 0, r << np.maximum(k, 0), r >> np.maximum(-k, 0))


# ----------------------------------------------------------------------------------------------------------------
# export: float checkpoint -> fixed point
# ----------------------------------------------------------------------------------------------------------------
def _q(W, ws):
    return np.rint(np.asarray(W, np.float64) * 2.0 ** ws).astype(np.int64)


def _ws_for(W, bits=15, cap=24):
    wmax = float(np.abs(W).max()) or 1.0
    return min(cap, max(0, int(math.floor(math.log2((2 ** bits - 1) / wmax)))))


def export_int_model(state, args, out_path):
    """Quantise a float transformer (state dict of float64 numpy arrays, args with hidden / layers / heads / window /
    mix) to the shipped integer model npz."""
    sd = {k: np.asarray(v, np.float64) for k, v in state.items()}
    D = int(args['hidden']); NL = int(args['layers']); NH = int(args['heads']); W = int(args['window']); K = int(args['mix']); dh = D // NH
    mu_f, sd_f = sd['mu_f'], sd['sd_f']
    out = dict(D=np.int64(D), L=np.int64(NL), NH=np.int64(NH), W=np.int64(W), mix=np.int64(K))
    W0 = sd['embed.weight'] / sd_f[None, :]; b0 = sd['embed.bias'] - (W0 * mu_f[None, :]).sum(axis=1)
    ws0 = _ws_for(W0); assert ws0 >= AQ, 'embed weights too large for the Q8 rule'
    out.update(W0=_q(W0, ws0), b0=_q(b0, ws0), ws0=np.int64(ws0))
    for l in range(NL):
        p = f'layers.{l}.'
        gam1, be1 = sd[p + 'ln1.weight'], sd[p + 'ln1.bias']
        Wqkv = sd[p + 'qkv.weight'] * gam1[None, :]; bqkv = sd[p + 'qkv.bias'] + sd[p + 'qkv.weight'] @ be1
        Wqkv = Wqkv.copy(); Wqkv[:D] /= math.sqrt(dh); bqkv = bqkv.copy(); bqkv[:D] /= math.sqrt(dh)
        wsq = _ws_for(Wqkv); assert wsq + 4 >= 0
        Wo, bo = sd[p + 'proj.weight'], sd[p + 'proj.bias']; wso = _ws_for(Wo); assert wso >= 4, 'proj weights too large'
        gam2, be2 = sd[p + 'ln2.weight'], sd[p + 'ln2.bias']
        W1 = sd[p + 'ff1.weight'] * gam2[None, :]; b1 = sd[p + 'ff1.bias'] + sd[p + 'ff1.weight'] @ be2; ws1 = _ws_for(W1)
        W2, b2 = sd[p + 'ff2.weight'], sd[p + 'ff2.bias']; ws2 = _ws_for(W2); assert ws2 >= 4, 'ff2 weights too large'
        rel = np.rint(sd[p + 'rel'] * 256.0).astype(np.int64)
        out.update({f'Wqkv{l}': _q(Wqkv, wsq), f'bqkv{l}': _q(bqkv, wsq + HQ), f'wsq{l}': np.int64(wsq),
                    f'Wo{l}': _q(Wo, wso), f'bo{l}': _q(bo, wso + AQ), f'wso{l}': np.int64(wso),
                    f'W1{l}': _q(W1, ws1), f'b1{l}': _q(b1, ws1 + HQ), f'ws1{l}': np.int64(ws1),
                    f'W2{l}': _q(W2, ws2), f'b2{l}': _q(b2, ws2 + AQ), f'ws2{l}': np.int64(ws2), f'rel{l}': rel})
    Wf, bf = sd['final.weight'], sd['final.bias']; wsf = _ws_for(Wf)
    Wh, bh = sd['head.weight'], sd['head.bias']; wsh = _ws_for(Wh)
    out.update(Wf=_q(Wf, wsf), bf=_q(bf, wsf + HQ), wsf=np.int64(wsf), Wh=_q(Wh, wsh), bh=_q(bh, wsh + AQ), wsh=np.int64(wsh))
    np.savez(out_path, **out)
    return out_path


# ----------------------------------------------------------------------------------------------------------------
# the integer model
# ----------------------------------------------------------------------------------------------------------------
class IntTf:
    """start(C) -> fresh caches for C curves; step(x [C, NFEAT], act) -> (logw, mu, logs) int64 [C, K] of the curves'
    current residuals (the rows of `act` are valid), appending this position to every curve's cache."""

    def __init__(self, path=MODEL_FILE):
        d = np.load(path)
        self.path = path
        self.D = int(d['D']); self.NL = int(d['L']); self.NH = int(d['NH']); self.W = int(d['W']); self.K = int(d['mix']); self.dh = self.D // self.NH
        self.log2d = int(round(math.log2(self.D))); assert 1 << self.log2d == self.D
        self.W0 = d['W0'].astype(np.float64); self.b0 = d['b0'].astype(np.int64); self.ws0 = int(d['ws0'])
        self.layers = []
        for l in range(self.NL):
            self.layers.append(dict(Wqkv=d[f'Wqkv{l}'].astype(np.float64), bqkv=d[f'bqkv{l}'].astype(np.int64), wsq=int(d[f'wsq{l}']),
                                    Wo=d[f'Wo{l}'].astype(np.float64), bo=d[f'bo{l}'].astype(np.int64), wso=int(d[f'wso{l}']),
                                    W1=d[f'W1{l}'].astype(np.float64), b1=d[f'b1{l}'].astype(np.int64), ws1=int(d[f'ws1{l}']),
                                    W2=d[f'W2{l}'].astype(np.float64), b2=d[f'b2{l}'].astype(np.int64), ws2=int(d[f'ws2{l}']),
                                    rel=d[f'rel{l}'].astype(np.int64)))
        self.Wf = d['Wf'].astype(np.float64); self.bf = d['bf'].astype(np.int64); self.wsf = int(d['wsf'])
        self.Wh = d['Wh'].astype(np.float64); self.bh = d['bh'].astype(np.int64); self.wsh = int(d['wsh'])
        for W in [self.W0, self.Wf, self.Wh] + [L[k] for L in self.layers for k in ('Wqkv', 'Wo', 'W1', 'W2')]:
            assert np.abs(W).max() < 2 ** 15
        self.t = 0; self.Kc = None; self.Vc = None; self._pad_cache = {}
        self._c = None                                   # the C step's state for the current clip (fast/tfstep.py)

    def start(self, C, pad=PAD):
        """Caches for C curves after `pad` zero-feature steps (every pad position has the same K / V in every layer, so
        the cache is that state replicated; computed once per model)."""
        self._c = _fast.tf_state(self, C, pad)
        if self._c is not None:
            self.t = pad; self.Kc = self.Vc = None
            return
        if pad not in self._pad_cache:
            self.Kc = [np.zeros((self.NH, self.W, self.dh), np.float64) for _ in range(self.NL)]
            self.Vc = [np.zeros((self.NH, self.W, self.dh), np.float64) for _ in range(self.NL)]
            self.t = 0
            for _ in range(pad):
                self.step(np.zeros((1, NFEAT), np.int64), None)
            self._pad_cache[pad] = ([k.copy() for k in self.Kc], [v.copy() for v in self.Vc], self.t)
        Kc, Vc, t = self._pad_cache[pad]
        self.Kc = [np.tile(k, (C, 1, 1)) for k in Kc]; self.Vc = [np.tile(v, (C, 1, 1)) for v in Vc]; self.t = t

    @staticmethod
    def _mm(A_int, W):
        """Exact integer product A [C, n] x W^T through float64 BLAS (every partial sum < 2^53)."""
        return (A_int.astype(np.float64) @ W.T).astype(np.int64)

    def _ln(self, h):
        mean = h.sum(axis=1, keepdims=True) >> self.log2d
        c = h - mean
        var = ((c * c).sum(axis=1, keepdims=True) >> self.log2d) + LN_EPS_Q24
        return (c * inv_sqrt_q16_of_q24(var)) >> 16

    def step(self, x, act):
        if self._c is not None:
            self.t += 1
            return self._c.step(x, act)
        C = x.shape[0]; W = self.W; t = self.t; slot = t % W
        e = np.maximum((self._mm(x, self.W0) + self.b0[None, :]) >> (self.ws0 - AQ), 0)
        h = np.minimum(e, 1 << 24) << (HQ - AQ)
        off = (t - np.arange(W)) % W                                 # offset of the position held in each ring slot
        valid = off <= t
        for l, L in enumerate(self.layers):
            a = self._ln(h)
            qkv = (self._mm(a, L['Wqkv']) + L['bqkv'][None, :]) >> (L['wsq'] + HQ - AQ)
            q = qkv[:, :self.D].reshape(C, self.NH, self.dh); k = qkv[:, self.D:2 * self.D].reshape(C, self.NH, self.dh)
            v = qkv[:, 2 * self.D:].reshape(C, self.NH, self.dh)
            Kc = self.Kc[l]; Vc = self.Vc[l]
            Kc[:, slot, :] = k.reshape(C * self.NH, self.dh); Vc[:, slot, :] = v.reshape(C * self.NH, self.dh)
            lg = np.matmul(Kc, q.reshape(C * self.NH, self.dh, 1).astype(np.float64))[:, :, 0].astype(np.int64).reshape(C, self.NH, W) >> 8
            lg = lg + L['rel'][:, off][None]
            lg = np.where(valid[None, None, :], lg, -(1 << 40))
            z = lg - lg.max(axis=2, keepdims=True)
            ex = EXP_TAB[np.clip(z, -EXP_LIM, 0) + EXP_LIM]
            ex = np.where(valid[None, None, :], ex, 0)
            p = (ex << 16) // ex.sum(axis=2, keepdims=True)
            o = np.matmul(p.reshape(C * self.NH, 1, W).astype(np.float64), Vc)[:, 0, :].astype(np.int64) >> 16
            o = o.reshape(C, self.D)
            h = np.clip(h + ((self._mm(o, L['Wo']) + L['bo'][None, :]) >> (L['wso'] - (HQ - AQ))), -H_LIM, H_LIM)
            a = self._ln(h)
            u = np.minimum(np.maximum((self._mm(a, L['W1']) + L['b1'][None, :]) >> (L['ws1'] + HQ - AQ), 0), 1 << 24)
            h = np.clip(h + ((self._mm(u, L['W2']) + L['b2'][None, :]) >> (L['ws2'] - (HQ - AQ))), -H_LIM, H_LIM)
        self.t = t + 1
        f = np.minimum(np.maximum((self._mm(h, self.Wf) + self.bf[None, :]) >> (self.wsf + HQ - AQ), 0), 1 << 24)
        o = ((self._mm(f, self.Wh) + self.bh[None, :]) >> self.wsh).reshape(C, 3, self.K)
        return o[:, 0], o[:, 1], np.clip(o[:, 2], LOGS_LO, LOGS_HI)
