"""Consistency of the training path with the codec, on BVH files:

  python -m curvecodec.train.check FILE.bvh [...] [--precisions 0.01,0.3] [--contract mean]

Per clip and precision:
  1. features  the encoder's own feature rows (the packer's lockstep loop, recorded at the model's input) == the rows
               the pool computes for its training windows from the dumped streams, for every residual
  2. bits      the integer model's code length of the residuals computed from the pool's rows (symbol + raw low bits,
               the codec's frequency tables) == the bits the dump recorded at encode time, and their sum vs the blob
               (blob = these bits + the header symbols / raw bits + the rANS flush)
  3. float     the float model's NLL (bits) on the same windows, the training loss (with --float CKPT)
"""
import argparse
import os
import sys
import tempfile

import numpy as np


def check_clip(clip, p, contract, float_ckpt=None):
    from .. import fast
    from ..codec import CurveCodec
    from ..features import ALPHABET, WIDTHS, symbol_freqs, symbol_index
    from .dump import save_streams
    from .pool import Pool, build
    codec = CurveCodec()
    fast.OFF.add('tfstep')                           # the numpy loop of the packer (the C pass has no Python hook)
    codec.model.start(1)                             # (the zero-step cache, before the recording starts)
    rec = []                                         # the packer's feature rows, time-major, with their columns
    step = codec.model.step

    def recording_step(x, act):
        if act is not None:                          # (the zero-feature start steps have no act)
            rec.append((np.asarray(x)[act].copy(), np.asarray(act).copy()))
        return step(x, act)
    codec.model.step = recording_step
    try:
        streams = {}
        blob = codec.encode(clip, p, contract, 1.0, streams=streams)
    finally:
        fast.OFF.discard('tfstep'); del codec.model.step
    if not streams:
        return dict(symbols=0)
    n = np.asarray(streams['n'])
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, 'clip.npz')
        save_streams(f, dict(group='check', clip=clip.name, p=p, blob_bytes=len(blob), acl_bytes=int(clip.ref(p)['compressed_size'])), streams)
        build([f], os.path.join(td, 'pool'))
        pool = Pool(os.path.join(td, 'pool'))
        # every residual's row from the pool: one window per OUT positions, context 0
        rows = [np.zeros((int(k), 36), np.int64) for k in n]
        cs, ws = pool.windows()
        x, y, m = pool.gather(cs, ws, ctx=0)
        for j in range(len(cs)):
            k = int(m[j].sum())
            rows[cs[j]][ws[j]:ws[j] + k] = x[j, :k]
    # 1. features: time-major from the pool vs the packer's record
    same = True
    for t, (xr, act) in enumerate(rec):
        if not np.array_equal(np.stack([rows[c][t] for c in act]), xr):
            same = False
            break
    same = same and len(rec) == int(n.max())
    # 2. bits: the integer model over the pool's rows, in lockstep as the codec runs it
    model = codec.model
    model.start(len(n))
    bits = np.zeros(len(n))
    r_curve = np.split(np.asarray(streams['r'], np.int64), np.cumsum(n)[:-1])
    for t in range(int(n.max())):
        act = np.flatnonzero(n > t)
        X = np.zeros((len(n), 36), np.int64)
        X[act] = np.stack([rows[c][t] for c in act])
        lw, mu, ls = model.step(X, act)
        f = symbol_freqs(lw[act], mu[act], ls[act], ALPHABET)
        yt = np.array([r_curve[c][t] for c in act], np.int64)
        s = symbol_index(yt)
        bits[act] += 15.0 - np.log2(f[np.arange(len(act)), s].astype(np.float64)) + WIDTHS[s]
    dumped = np.array([b.sum() for b in np.split(np.asarray(streams['bits']), np.cumsum(n)[:-1])])
    low = float(np.sum(WIDTHS[symbol_index(np.asarray(streams['r'], np.int64))]))
    budget = streams['prefix_bytes'] + (streams['header_bits'] + bits.sum() - low + streams['raw_bits']) / 8.0
    out = dict(symbols=int(n.sum()), features_equal=bool(same), bits_model=float(bits.sum()), bits_dumped=float(dumped.sum()),
               bits_equal=bool(np.allclose(bits, dumped, rtol=0, atol=1e-6)), residual_bytes=float(bits.sum()) / 8.0,
               blob_bytes=len(blob), blob_from_bits=float(budget), rans_overhead_bytes=float(len(blob) - budget))
    if float_ckpt:
        out['float_bits'] = float_nll(float_ckpt, x, y, m) / max(1, int(m.sum())) * int(n.sum())
    return out


def float_nll(ckpt, x, y, m):
    """Total float-model NLL (bits) of the scored positions of gathered windows (zero left context as in the codec)."""
    import torch
    from .net import load_net, nll_bits
    net, _ = load_net(ckpt, 'cpu')
    with torch.no_grad():
        pad = torch.zeros(len(x), 128, 36)
        xx = torch.cat([pad, torch.from_numpy(x)], dim=1)
        lw, mu, ls = net(xx)
        b = nll_bits(lw[:, 128:], mu[:, 128:], ls[:, 128:], torch.from_numpy(y))
    return float((b * torch.from_numpy(m)).sum())


def main(argv=None):
    from ..acl import acl_clip
    from ..contract import Contract
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('bvh', nargs='+'); ap.add_argument('--precisions', default='0.01,0.3'); ap.add_argument('--contract', default='mean')
    ap.add_argument('--float', default='', help='a float checkpoint: also its NLL on the same windows')
    a = ap.parse_args(argv)
    ps = [float(v) for v in a.precisions.split(',')]
    ok = True
    for f in a.bvh:
        clip = acl_clip(f, ps)
        for p in ps:
            r = check_clip(clip, p, Contract.parse(a.contract), a.float or None)
            ok &= r.get('features_equal', True) and r.get('bits_equal', True)
            print(f'{os.path.basename(f)} p {p:g}: ' + ', '.join(f'{k} {v:.1f}' if isinstance(v, float) else f'{k} {v}' for k, v in r.items()), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
