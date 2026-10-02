"""The float transformer (PyTorch) that model.py's integer model is exported from.

TfNet(hidden d, layers N, heads h, window W, ff, mix K):
  embed   Linear(36 -> d) + ReLU on the normalised features ((x - mu_f) / sd_f, buffers)
  trunk   N pre-LN blocks: h += Wo Attn(LN1 h); h += W2 ReLU(W1 LN2 h)   (FFN width ff x d)
          Attn = causal multi-head attention over the last W positions (the query included) with a learned bias per head
          and offset; no absolute positions (the 'pos' feature carries L(t))
  final   Linear(d -> d) + ReLU; head Linear(d -> 3 K) -> (log w, mu, log s) per position
Forward: x float [B, L, 36] -> (logw, mu, logs) [B, L, K]; attention in chunks of queries (O(L W) memory).
The loss is the discretised logistic mixture NLL of the integer residual, in bits.
"""
import math

import torch
import torch.nn.functional as Fn
from torch import nn

NFEAT = 36


class Block(nn.Module):
    def __init__(self, hidden, heads, window, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden); self.qkv = nn.Linear(hidden, 3 * hidden); self.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden); self.ff1 = nn.Linear(hidden, ff * hidden); self.ff2 = nn.Linear(ff * hidden, hidden)
        self.rel = nn.Parameter(torch.zeros(heads, window))


class TfNet(nn.Module):
    def __init__(self, hidden=64, layers=2, heads=4, window=128, ff=4, mix=3, chunk=1024):
        super().__init__()
        assert hidden % heads == 0
        self.hidden, self.heads, self.window, self.mix, self.chunk = hidden, heads, window, mix, max(chunk, window)
        self.register_buffer('mu_f', torch.zeros(NFEAT)); self.register_buffer('sd_f', torch.ones(NFEAT))
        self.embed = nn.Linear(NFEAT, hidden)
        self.layers = nn.ModuleList([Block(hidden, heads, window, ff) for _ in range(layers)])
        self.final = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 3 * mix)

    def _attn(self, blk, a):
        B, L, D = a.shape
        H = self.heads; dh = D // H; W = self.window
        q, k, v = blk.qkv(a).view(B, L, 3, H, dh).unbind(2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = torch.empty_like(q)
        for i0 in range(0, L, self.chunk):
            i1 = min(L, i0 + self.chunk); k0 = max(0, i0 - W + 1)
            off = torch.arange(i0, i1, device=a.device)[:, None] - torch.arange(k0, i1, device=a.device)[None, :]
            ok = (off >= 0) & (off < W)
            bias = blk.rel[:, off.clamp(0, W - 1)].masked_fill(~ok[None], float('-inf'))
            out[:, :, i0:i1] = Fn.scaled_dot_product_attention(q[:, :, i0:i1], k[:, :, k0:i1], v[:, :, k0:i1], attn_mask=bias)
        return out.transpose(1, 2).reshape(B, L, D)

    def forward(self, x):
        h = torch.relu(self.embed((x - self.mu_f) / self.sd_f))
        for blk in self.layers:
            h = h + blk.proj(self._attn(blk, blk.ln1(h)))
            h = h + blk.ff2(torch.relu(blk.ff1(blk.ln2(h))))
        o = self.head(torch.relu(self.final(h)))
        B, L, _ = o.shape
        o = o.view(B, L, 3, self.mix)
        return torch.log_softmax(o[..., 0, :], dim=-1), o[..., 1, :], o[..., 2, :].clamp(-6.0, 12.0)


def nll_bits(logw, mu, logs, y):
    """Discretised logistic mixture NLL in bits of the integers y."""
    y = y.float().unsqueeze(-1)
    s = torch.exp(logs)
    p = (torch.sigmoid((y + 0.5 - mu) / s) - torch.sigmoid((y - 0.5 - mu) / s)).clamp_min(1e-12)
    return -torch.logsumexp(logw + torch.log(p), dim=-1) / math.log(2.0)


def make_net(args):
    return TfNet(int(args['hidden']), int(args['layers']), int(args['heads']), int(args['window']), int(args['ff']), int(args['mix']))


def load_net(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    net = make_net(ck['args'])
    net.load_state_dict(ck['state'])
    return net.to(device).eval(), ck['args']
