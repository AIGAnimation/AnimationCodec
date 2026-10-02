"""Dump the codec's residual streams of BVH files (the entropy model's training data).

  python -m curvecodec.train.dump SRC --out DIR [--precisions 0.01,0.05,0.1,0.3,1] [--contract mean] [--group NAME] [--jobs 8]

SRC = a folder (searched recursively for *.bvh) or a list file. One file per clip and precision,
DIR/<group>__<clip>__p<p>.npz, holding the streams the packer codes (CurveCodec.pack): per coded curve its length n,
static fields (kind, depth class, keyframe flag, predictor id, log2 step) and first integer x0, and curve-major the
residuals r, the decoded increments dx = x_{t+1} - x_t, the key gaps g and the code length of every residual in the
blob (bits); meta = the clip, p, frame rate, blob and ACL sizes. The group (the budget unit of training: a corpus family)
is --group, else the first folder of the BVH's path below SRC.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def save_streams(path, meta, streams):
    """One dump file (atomic) -> the number of symbols."""
    arrs = {k: np.asarray(v) for k, v in streams.items()}
    tmp = path[:-4] + '.tmp.npz'
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrs)
    os.replace(tmp, path)
    return int(np.sum(arrs.get('n', 0)))


def dump_clip(job):
    try:
        return _dump_clip(job)
    except Exception as e:  # noqa: BLE001 -- one bad file does not stop the dump
        print(f'  {job[0]}: {type(e).__name__}: {e}', flush=True)
        return 0, 0


def _dump_clip(job):
    path, rel, group, precisions, contract_spec, out_dir, scale = job
    from ..acl import acl_clip
    from ..codec import CurveCodec
    from ..contract import Contract
    name = os.path.splitext(rel)[0].replace(os.sep, '__').replace(' ', '_')
    todo = [p for p in precisions if not os.path.exists(os.path.join(out_dir, f'{group}__{name}__p{p:g}.npz'))]
    if not todo:
        return 0, 0
    try:
        clip = acl_clip(path, todo, scale)
    except Exception as e:  # noqa: BLE001
        print(f'  skipped {path}: {type(e).__name__}: {e}', flush=True)
        return 0, 0
    codec = CurveCodec(); contract = Contract.parse(contract_spec)
    nsym = 0
    for p in todo:
        streams = {}
        blob = codec.encode(clip, p, contract, 1.0, streams=streams)
        meta = dict(group=group, clip=name, p=p, fps=clip.fps, frames=clip.F, bones=clip.B, blob_bytes=len(blob),
                    acl_bytes=int(clip.ref(p)['compressed_size']), contract=str(contract))
        nsym += save_streams(os.path.join(out_dir, f'{group}__{name}__p{p:g}.npz'), meta, streams)
    return len(todo), nsym


def main(argv=None):
    from ..evaluate import list_bvh
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src'); ap.add_argument('--out', required=True)
    ap.add_argument('--precisions', default='0.01,0.05,0.1,0.3,1'); ap.add_argument('--contract', default='mean')
    ap.add_argument('--group', default=''); ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--scale', default='auto')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    root = a.src if os.path.isdir(a.src) else os.path.dirname(os.path.abspath(a.src))
    precisions = [float(x) for x in a.precisions.split(',')]
    jobs = []
    for f in list_bvh(a.src):
        rel = os.path.relpath(f, root)
        group = a.group or (rel.split(os.sep)[0] if os.sep in rel else 'default')
        jobs.append((f, rel, group, precisions, a.contract, a.out, a.scale))
    t0 = time.time(); nf = ns = 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, (f, s) in enumerate(ex.map(dump_clip, jobs), 1):
            nf += f; ns += s
            if k % 10 == 0 or k == len(jobs):
                print(f'  [{k}/{len(jobs)}] {nf} dumps, {ns / 1e6:.2f} M symbols ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
