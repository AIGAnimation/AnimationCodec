"""ACL reference: run cpp/acl_profile (ACL compressing the BVH at a precision) and read its result.

acl_profile converts the BVH to local transforms (cm; translations scaled to cm by its rule unless --scale is given),
compresses them with ACL at the precision and reports the compressed size and ACL's own shell-error statistics, which
the mean contract is defined against.
"""
import json
import os
import re
import subprocess
import tempfile

import numpy as np

from .bvh import parse_header
from .clip import Clip, prec_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def binary():
    return os.environ.get('CURVECODEC_ACL_PROFILE', os.path.join(ROOT, 'cpp', 'build', 'acl_profile'))


# ----------------------------------------------------------------------------------------------------------------
# SJSON (ACL's simplified JSON): key = value pairs, bare keys, arrays with optional commas
# ----------------------------------------------------------------------------------------------------------------
_TOKEN = re.compile(r'\s+|//[^\n]*|"(?:\\.|[^"\\])*"|[{}\[\]=,]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[A-Za-z_][A-Za-z0-9_\-\.]*')


def _tokens(text):
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise ValueError(f'bad SJSON at {pos}: {text[pos:pos + 20]!r}')
        pos = m.end()
        tok = m.group(0)
        if not (tok.isspace() or tok.startswith('//')):
            yield tok


def load_sjson(path):
    toks = list(_tokens(open(path, encoding='utf-8', errors='replace').read()))
    i = 0

    def value():
        nonlocal i
        t = toks[i]; i += 1
        if t == '{':
            return obj('}')
        if t == '[':
            out = []
            while toks[i] != ']':
                if toks[i] == ',':
                    i += 1
                    continue
                out.append(value())
            i += 1
            return out
        if t[0] == '"':
            s = json.loads(t)
            return float(s) if s in ('nan', 'inf', '-inf') else s
        if t in ('true', 'false', 'null'):
            return {'true': True, 'false': False, 'null': None}[t]
        try:
            return int(t) if re.fullmatch(r'-?\d+', t) else float(t)
        except ValueError:
            return t

    def obj(closing):
        nonlocal i
        out = {}
        while i < len(toks) and toks[i] != closing:
            if toks[i] == ',':
                i += 1
                continue
            key = toks[i]; i += 1
            key = json.loads(key) if key[0] == '"' else key
            if toks[i] != '=':
                raise ValueError(f'expected = after {key}')
            i += 1
            out[key] = value()
        i += 1
        return out
    return obj(None)


def acl_clip(bvh_path, precisions, scale='auto'):
    """ACL at every precision (cm) on one BVH -> Clip with the raw local transforms and ACL's reference per precision."""
    exe = binary()
    if not os.path.exists(exe):
        raise FileNotFoundError(f'{exe} not found: build cpp/ first (see README) or set CURVECODEC_ACL_PROFILE')
    names, parents, offsets, _ = parse_header(bvh_path)
    acl = {}; local = None
    for p in (precisions if isinstance(precisions, (list, tuple)) else [precisions]):
        p = float(p)
        with tempfile.TemporaryDirectory() as td:
            sj, lo = os.path.join(td, 'p.sjson'), os.path.join(td, 'local.f32')
            cmd = [exe, bvh_path, sj, f'--export-local={lo}', '--no-strip-trivial', f'--precision={p:g}']
            if scale not in (None, '', 'auto'):
                cmd.append(f'--scale={float(scale):g}')
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f'acl_profile failed on {bvh_path}: {(r.stdout + r.stderr)[-800:]}')
            meta = load_sjson(sj)
            B = meta['num_bones']; F = meta['num_samples']
            if local is None:
                local = np.fromfile(lo, dtype=np.float32).reshape(F, B, 7)
        if list(names) != list(meta['bone_names']) or list(parents) != list(meta['bone_parents']):
            raise RuntimeError(f'{bvh_path}: the BVH header parse does not match acl_profile\'s skeleton')
        fe = meta['full_error']
        acl[prec_key(p)] = dict(compressed_size=int(meta['compressed_size']), mean=fe['mean'], max=fe['max'], p99=fe['p99'],
                                frac_above_precision=fe['frac_above_precision'], bone_max=fe['bone_max'], bone_mean=fe['bone_mean'])
        scale_cm = float(meta['scale_to_cm']); fps = float(meta['sample_rate'])
    return Clip(name=os.path.splitext(os.path.basename(bvh_path))[0], local=local, parents=np.asarray(parents, np.int32),
                offsets=(offsets * scale_cm).astype(np.float32), fps=fps, names=list(names), scale=scale_cm, acl=acl)
