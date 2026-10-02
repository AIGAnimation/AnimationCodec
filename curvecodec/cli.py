"""Command line: one BVH file in, one compressed file out, and back to joint transforms.

  curvecodec encode clip.bvh -o clip.cc2 [--precision 0.01] [--contract mean | mean:C=1.5 | mean:F=0.05] [--scale auto|<float>]
  curvecodec decode clip.cc2 -o clip.npz
  curvecodec info   clip.cc2

encode runs ACL (cpp/build/acl_profile) on the BVH at precision p first: the contract is defined against ACL's own
result on the same clip (curvecodec/contract.py). A clip that misses the contract is encoded again with the fallback
margins 0.98 / 0.95 / 0.90 (every budget the encoder derives from ACL scaled down; the blob decodes the same way).
If every margin misses, the file is still written and encode exits with status 2.

File (.cc2): b'CC2\\x01', uint32 LE header length, UTF-8 JSON header (skeleton, frame rate, precision, contract, sizes),
codec blob. Like ACL's compressed tracks the blob does not carry the skeleton or the frame rate; the header does.
`blob_bytes` is the size compared with ACL's `acl_bytes`.
decode writes an .npz: local float32 [F, B, 7] (qx, qy, qz, qw, tx, ty, tz; cm), names, parents, offsets (cm), fps.
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

MAGIC = b'CC2\x01'


def encode_clip(codec, clip, p, contract):
    """Encode with the fallback margins -> (blob, margin, contract met, error stats, ratios, encode s, decode s)."""
    from .contract import MARGINS
    from .metric import error_stats, shell_error
    for margin in MARGINS:
        t0 = time.time()
        blob = codec.encode(clip, p, contract, margin)
        t1 = time.time()
        out = codec.decode(blob, clip.parents, clip.offsets, clip.fps)
        t2 = time.time()
        st = error_stats(shell_error(clip.local, out, clip.parents), p)
        ok, reasons, ratios = contract.check(st, clip.ref(p), p)
        if ok:
            break
    return blob, margin, ok, reasons, st, ratios, t1 - t0, t2 - t1


def write_file(path, header, blob):
    h = json.dumps(header, separators=(',', ':')).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(MAGIC + struct.pack('<I', len(h)) + h + blob)


def read_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != MAGIC:
        raise SystemExit(f'{path}: not a .cc2 file')
    n = struct.unpack('<I', data[4:8])[0]
    return json.loads(data[8:8 + n].decode('utf-8')), data[8 + n:]


def cmd_encode(a):
    from .acl import acl_clip
    from .codec import CurveCodec
    from .contract import Contract
    contract = Contract.parse(a.contract)
    t0 = time.time()
    clip = acl_clip(a.bvh, [a.precision], a.scale)
    t_acl = time.time() - t0
    blob, margin, ok, reasons, st, _, t_enc, t_dec = encode_clip(CurveCodec(), clip, a.precision, contract)
    ref = clip.ref(a.precision)
    header = dict(format='curvecodec', version=1, contract=str(contract), margin=margin, precision=float(a.precision), frames=clip.F,
                  bones=clip.B, fps=clip.fps, names=clip.names, parents=[int(x) for x in clip.parents],
                  offsets=[[float(v) for v in o] for o in clip.offsets], blob_bytes=len(blob), acl_bytes=int(ref['compressed_size']),
                  contract_met=bool(ok))
    out = a.output or os.path.splitext(a.bvh)[0] + '.cc2'
    write_file(out, header, blob)
    print(f'{os.path.basename(a.bvh)}: {clip.B} joints x {clip.F} frames @ {clip.fps:g} fps, p = {a.precision:g} cm, {contract}')
    print(f'  ACL     {ref["compressed_size"]:>10,} B  err mean {ref["mean"]:.4f}  max {ref["max"]:.4f} cm   ({t_acl:.1f} s)')
    print(f'  ours    {len(blob):>10,} B  err mean {st["err_mean"]:.4f}  max {st["err_max"]:.4f} cm   ({t_enc:.1f} s encode, {t_dec:.2f} s decode'
          + (f', margin {margin:g}' if margin != 1.0 else '') + ')')
    print(f'  ratio {len(blob) / ref["compressed_size"]:.3f} x ACL, contract {"met" if ok else "NOT met: " + "; ".join(reasons)}')
    print(f'  wrote {out}')
    return 0 if ok else 2


def cmd_decode(a):
    from .codec import CurveCodec
    h, blob = read_file(a.file)
    t0 = time.time()
    local = CurveCodec().decode(blob, np.asarray(h['parents'], np.int32), np.asarray(h['offsets'], np.float32), float(h['fps']))
    out = a.output or os.path.splitext(a.file)[0] + '.npz'
    np.savez(out, local=local, names=np.asarray(h['names']), parents=np.asarray(h['parents'], np.int32),
             offsets=np.asarray(h['offsets'], np.float32), fps=np.float32(h['fps']))
    print(f'decoded {local.shape[1]} joints x {local.shape[0]} frames in {time.time() - t0:.2f} s -> {out}')
    return 0


def cmd_info(a):
    h, _ = read_file(a.file)
    keep = {k: v for k, v in h.items() if k not in ('names', 'parents', 'offsets')}
    keep['ratio_vs_acl'] = round(h['blob_bytes'] / h['acl_bytes'], 4)
    print(json.dumps(keep, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='curvecodec', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('encode', help='BVH -> .cc2')
    e.add_argument('bvh')
    e.add_argument('-o', '--output')
    e.add_argument('--precision', type=float, default=0.01, help='ACL precision in cm (default 0.01)')
    e.add_argument('--contract', default='mean', help="mean[:G=..,C=..,T=..,F=..] (default 'mean')")
    e.add_argument('--scale', default='auto', help="unit scale of the BVH to cm ('auto' = acl_profile's rule)")
    d = sub.add_parser('decode', help='.cc2 -> .npz of local transforms')
    d.add_argument('file')
    d.add_argument('-o', '--output')
    i = sub.add_parser('info', help='print the header of a .cc2 file')
    i.add_argument('file')
    a = ap.parse_args(argv)
    return {'encode': cmd_encode, 'decode': cmd_decode, 'info': cmd_info}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
