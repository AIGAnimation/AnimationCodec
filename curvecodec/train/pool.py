"""The training pool: many stream dumps in one memory-mapped image; training windows with their features computed on
demand (the features of a curve's residual depend on the curve's whole past, so a window's rows are computed by running
the curve's feature recursion from its start, in C).

  python -m curvecodec.train.pool build DUMP_DIR [DUMP_DIR ...] --out POOL [--files-from LIST]

Image: curves.npy int64 [C, 13] (file, group, n, nmax, kind, dc, kf, pid, step_q8, fps_i, p_idx, x0, offset), r.npy /
dx.npy / g.npy (curve-major per symbol), files.json (dump file names, groups, curve ranges, symbols, codec bits).

A window (N9's unit) = OUT scored positions [w, w + OUT) of one curve with CTX positions of left context; rows before
the curve's start are zero features (the codec starts every curve after zero-feature steps), rows past its end are
masked: gather -> x float32 [B, CTX + OUT, 36], y int64 [B, CTX + OUT], mask bool.
"""
import argparse
import json
import os
import sys

import numpy as np

COLS = ('file', 'group', 'n', 'nmax', 'kind', 'dc', 'kf', 'pid', 'step_q8', 'fps_i', 'p_idx', 'x0', 'offset')
(C_FILE, C_GROUP, C_N, C_NMAX, C_KIND, C_DC, C_KF, C_PID, C_STEP, C_FPS, C_PIDX, C_X0, C_OFF) = range(len(COLS))
CTX = 128
OUT = 384


def build(dump_files, out_dir):
    """Dump .npz files -> the pool image in out_dir; returns the number of symbols."""
    os.makedirs(out_dir, exist_ok=True)
    rows, rs, dxs, gs, meta, groups = [], [], [], [], [], []
    off = 0
    for i, f in enumerate(dump_files):
        d = np.load(f)
        m = json.loads(str(d['meta']))
        if m['group'] not in groups:
            groups.append(m['group'])
        c0 = len(rows)
        if 'n' in d.files:
            n = d['n']
            for j in range(len(n)):
                rows.append([i, groups.index(m['group']), int(n[j]), int(d['nmax']), int(d['kind'][j]), int(d['dc'][j]), int(d['kf'][j]),
                             int(d['pid'][j]), int(d['step_q8'][j]), int(d['fps_i']), int(d['p_idx']), int(d['x0'][j]), off])
                off += int(n[j])
            rs.append(d['r']); dxs.append(d['dx']); gs.append(d['g'])
        meta.append(dict(file=os.path.basename(f), group=m['group'], curves=[c0, len(rows)], symbols=int(np.sum(d['n'])) if 'n' in d.files else 0,
                         bits=float(np.sum(d['bits'])) if 'bits' in d.files else 0.0, blob_bytes=m['blob_bytes'], acl_bytes=m['acl_bytes']))
    cat = (lambda a: np.concatenate(a) if a else np.zeros(0, np.int64))
    r, dx, g = cat(rs), cat(dxs), cat(gs)
    small = lambda a: a.astype(np.int32) if len(a) == 0 or (np.abs(a).max() < 2 ** 31 - 1) else a   # noqa: E731
    np.save(os.path.join(out_dir, 'curves.npy'), np.asarray(rows, np.int64).reshape(-1, len(COLS)))
    np.save(os.path.join(out_dir, 'r.npy'), small(r)); np.save(os.path.join(out_dir, 'dx.npy'), small(dx))
    np.save(os.path.join(out_dir, 'g.npy'), g.astype(np.int32))
    json.dump(dict(version=1, groups=groups, files=meta, symbols=int(off)), open(os.path.join(out_dir, 'files.json'), 'w'))
    return off


class Pool:
    """A pool image (memory-mapped): curves, windows, gather."""

    def __init__(self, pool_dir):
        from ..features import L
        self.curves = np.load(os.path.join(pool_dir, 'curves.npy'))
        self.r = np.load(os.path.join(pool_dir, 'r.npy'), mmap_mode='r')
        self.dx = np.load(os.path.join(pool_dir, 'dx.npy'), mmap_mode='r')
        self.g = np.load(os.path.join(pool_dir, 'g.npy'), mmap_mode='r')
        info = json.load(open(os.path.join(pool_dir, 'files.json')))
        self.groups = info['groups']; self.files = info['files']
        cv = self.curves
        self.n = cv[:, C_N].copy(); self.fid = cv[:, C_FILE].copy(); self.group = cv[:, C_GROUP].copy()
        self.N = int(self.n.sum())
        # the static features of every curve (features.CurveState's base row)
        C = len(cv)
        base = np.zeros((C, 36), np.int64)
        col = 21
        base[np.arange(C), col + cv[:, C_DC]] = 256
        base[:, col + 4] = 256 * cv[:, C_KIND]
        base[:, col + 5] = 256 * cv[:, C_KF]
        base[np.arange(C), col + 6 + np.clip(cv[:, C_PID], 0, 3)] = 256
        base[:, col + 10] = cv[:, C_STEP]
        base[:, col + 11] = L(cv[:, C_FPS] - 1)
        base[:, col + 12] = 256 * cv[:, C_PIDX]
        base[:, col + 14] = L(self.n - 1)
        self.base = base

    def windows(self, out=OUT):
        """Every OUT-aligned window of every curve -> (curve, window start)."""
        k = (self.n + out - 1) // out
        cs = np.repeat(np.arange(len(self.n), dtype=np.int64), k)
        first = np.repeat(np.cumsum(k) - k, k)
        return cs, (np.arange(len(cs), dtype=np.int64) - first) * out

    def curve(self, c, T):
        """(r [T], x [T + 1], lg [min(T + 1, n)]) of curve c's first T residuals."""
        from ..features import L
        cv = self.curves[c]
        o = int(cv[C_OFF]); n = int(cv[C_N])
        r = np.asarray(self.r[o:o + T], np.int64)
        x = np.empty(T + 1, np.int64); x[0] = cv[C_X0]; np.cumsum(np.asarray(self.dx[o:o + T], np.int64), out=x[1:]); x[1:] += cv[C_X0]
        G = min(T + 1, n)
        lg = L(np.maximum(np.asarray(self.g[o:o + G], np.int64), 1) - 1) if cv[C_KF] else np.zeros(G, np.int64)
        return r, x, lg

    def gather(self, cs, ws, ctx=CTX, out=OUT, numpy=False):
        from ..fast import kernels
        B = len(cs); Lw = ctx + out
        x = np.zeros((B, Lw, 36), np.float32); y = np.zeros((B, Lw), np.int64); m = np.zeros((B, Lw), bool)
        for b in range(B):
            c = int(cs[b]); w0 = int(ws[b]); n = int(self.n[c])
            T = min(w0 + out, n)
            if T <= 0:
                continue
            t_lo = max(w0 - ctx, 0); p0 = t_lo - (w0 - ctx)
            r, xx, lg = self.curve(c, T)
            if numpy or not kernels.ready():
                rows = self._features_numpy(c, r, xx, lg, T)[t_lo:]
                if np.abs(rows).max(initial=0) > 32000:
                    raise ValueError('a feature leaves int16')
                x[b, p0:p0 + T - t_lo] = rows
            elif not kernels.window_features(n, self.curves[c, C_NMAX], r, xx, lg, self.base[c], T, t_lo, x[b, p0:p0 + T - t_lo]):
                raise ValueError('a feature leaves int16')
            s = max(w0, t_lo)
            y[b, s - (w0 - ctx):T - (w0 - ctx)] = r[s:T]; m[b, s - (w0 - ctx):T - (w0 - ctx)] = True
        return x, y, m

    def _features_numpy(self, c, r, xx, lg, T):
        """The reference: features.CurveState stepped over the curve (one curve, its gap row padded as in the lockstep
        batch of its clip)."""
        from ..features import CurveState
        cv = self.curves[c]; n = int(cv[C_N]); nmax = int(cv[C_NMAX])
        gaps = np.zeros((nmax, 1), np.int64)
        o = int(cv[C_OFF])
        if cv[C_KF]:
            gaps[:n, 0] = np.asarray(self.g[o:o + n], np.int64)
        st = CurveState([cv[C_KIND]], [cv[C_DC]], [cv[C_KF]], [cv[C_PID]], [cv[C_STEP]], cv[C_FPS], cv[C_PIDX], [n], gaps)
        st.start(xx[:1])
        rows = np.zeros((T, 36), np.int64)
        for t in range(T):
            rows[t] = st.features()[0]
            st.update(r[t:t + 1], xx[t + 1:t + 2])
        return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('build'); b.add_argument('dumps', nargs='*'); b.add_argument('--out', required=True)
    b.add_argument('--files-from', default='', help='a list of dump files (one per line) instead of folders')
    a = ap.parse_args(argv)
    if a.files_from:
        files = [l.strip() for l in open(a.files_from) if l.strip()]
    else:
        files = sorted(os.path.join(d, f) for d in a.dumps for f in os.listdir(d) if f.endswith('.npz'))
    N = build(files, a.out)
    print(f'pool {a.out}: {len(files)} dumps, {N / 1e6:.2f} M symbols')
    return 0


if __name__ == '__main__':
    sys.exit(main())
