"""Evaluate the codec against ACL on a folder of BVH files (or a list), in parallel.

  python -m curvecodec.evaluate DIR_OR_LIST [--precisions 0.01,0.05,0.1,0.3,1] [--contract mean ...] [--jobs 8]
                                            [--out report] [--scale auto]

For every clip and precision: ACL's reference (cpp/build/acl_profile), our encode with the fallback margins, the decode,
the shell error of the decoded clip, the contract check, bytes vs ACL, encode / decode times. Writes <out>.csv (one row
per clip x precision x contract) and <out>.md (per contract and precision: total bytes vs ACL, median clip ratio,
contract pass rate, mean / p99 / max error against ACL's, encode / decode seconds per million bone-frames).
A list file holds one BVH path per line (relative paths are relative to the list file); a folder is searched
recursively for *.bvh.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FIELDS = ('clip', 'precision', 'contract', 'frames', 'bones', 'acl_bytes', 'bytes', 'ratio', 'margin', 'contract_met', 'reasons',
          'err_mean', 'err_p99', 'err_max', 'acl_mean', 'acl_p99', 'acl_max', 'encode_s', 'decode_s', 'error')


def list_bvh(src):
    if os.path.isdir(src):
        out = []
        for d, _, fs in os.walk(src):
            out += [os.path.join(d, f) for f in fs if f.lower().endswith('.bvh')]
        return sorted(out)
    base = os.path.dirname(os.path.abspath(src))
    return [l if os.path.isabs(l) else os.path.join(base, l) for l in (x.strip() for x in open(src)) if l and not l.startswith('#')]


def _one(job):
    path, precisions, contracts, scale = job
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from .acl import acl_clip
    from .cli import encode_clip
    from .codec import CurveCodec
    from .contract import Contract
    rows = []
    try:
        clip = acl_clip(path, precisions, scale)
    except Exception as e:  # noqa: BLE001
        return [dict(clip=path, error=f'acl: {type(e).__name__}: {e}')]
    codec = CurveCodec()
    for spec in contracts:
        contract = Contract.parse(spec)
        for p in precisions:
            ref = clip.ref(p)
            row = dict(clip=path, precision=p, contract=str(contract), frames=clip.F, bones=clip.B, acl_bytes=int(ref['compressed_size']),
                       acl_mean=ref['mean'], acl_p99=ref['p99'], acl_max=ref['max'])
            try:
                blob, margin, ok, reasons, st, _, te, td = encode_clip(codec, clip, p, contract)
                row.update(bytes=len(blob), ratio=len(blob) / max(1, int(ref['compressed_size'])), margin=margin, contract_met=bool(ok),
                           reasons='; '.join(reasons), err_mean=st['err_mean'], err_p99=st['err_p99'], err_max=st['err_max'],
                           encode_s=te, decode_s=td)
            except Exception as e:  # noqa: BLE001
                row.update(error=f'{type(e).__name__}: {e}')
            rows.append(row)
    return rows


def report(rows, out):
    with open(out + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
    ok_rows = [r for r in rows if 'bytes' in r]
    lines = ['# CurveCodec vs ACL', '', f'{len({r["clip"] for r in rows})} clips, {len(ok_rows)} encodes, '
             f'{sum(1 for r in rows if r.get("error"))} errors.', '']
    for spec in sorted({r['contract'] for r in ok_rows}):
        lines += [f'## contract {spec}', '',
                  '| p (cm) | clips | bytes | ACL bytes | x ACL | x ACL (median clip) | contract met | fallback margin used | '
                  'err mean (ACL) | err p99 (ACL) | err max (ACL) | encode s / Mbf | decode s / Mbf |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for p in sorted({r['precision'] for r in ok_rows if r['contract'] == spec}):
            rs = [r for r in ok_rows if r['contract'] == spec and r['precision'] == p]
            bf = sum(r['frames'] * r['bones'] for r in rs) / 1e6
            ours = sum(r['bytes'] for r in rs); acl = sum(r['acl_bytes'] for r in rs)

            def mean(k):
                return float(np.mean([r[k] for r in rs]))
            lines.append(f'| {p:g} | {len(rs)} | {ours:,} | {acl:,} | {ours / acl:.3f} | {np.median([r["ratio"] for r in rs]):.3f} | '
                         f'{sum(r["contract_met"] for r in rs)}/{len(rs)} | {sum(r["margin"] != 1.0 for r in rs)} | '
                         f'{mean("err_mean"):.4f} ({mean("acl_mean"):.4f}) | {mean("err_p99"):.4f} ({mean("acl_p99"):.4f}) | '
                         f'{mean("err_max"):.4f} ({mean("acl_max"):.4f}) | {sum(r["encode_s"] for r in rs) / bf:.1f} | '
                         f'{sum(r["decode_s"] for r in rs) / bf:.2f} |')
        lines.append('')
    misses = [r for r in ok_rows if not r['contract_met']]
    if misses:
        lines += ['## contract misses', ''] + [f'- {r["clip"]} p {r["precision"]:g} {r["contract"]}: {r["reasons"]}' for r in misses] + ['']
    errs = [r for r in rows if r.get('error')]
    if errs:
        lines += ['## errors', ''] + [f'- {r["clip"]}: {r["error"]}' for r in errs] + ['']
    open(out + '.md', 'w').write('\n'.join(lines))
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src', help='a folder of BVH files or a list file')
    ap.add_argument('--precisions', default='0.01,0.05,0.1,0.3,1')
    ap.add_argument('--contract', action='append', help="mean[:G=..,C=..,T=..,F=..] (repeatable; default 'mean')")
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--scale', default='auto')
    ap.add_argument('--out', default='report')
    a = ap.parse_args(argv)
    files = list_bvh(a.src)
    precisions = [float(x) for x in a.precisions.split(',')]
    jobs = [(f, precisions, a.contract or ['mean'], a.scale) for f in files]
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, rs in enumerate(ex.map(_one, jobs), 1):
            rows += rs
            print(f'  [{k}/{len(jobs)}] {os.path.basename(rs[0]["clip"])} ({time.time() - t0:.0f} s)', flush=True)
    print(report(rows, a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
