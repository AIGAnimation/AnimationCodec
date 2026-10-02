"""Train the float transformer on a pool, then export it to the fixed-point model.

  python -m curvecodec.train.train --pool POOL --out DIR [--dev-pool POOL] [--layers 2 --hidden 64 --heads 4 --window 128]
                                   [--budget sqrt[+group=share,..]] [--total-windows 40000000] [--batch 128] [--lr 2e-3]
                                   [--max-steps N] [--device cuda]

Recipe (the shipped model's): windows of 128 context + 384 scored positions, batch 128, AdamW (weight decay 1e-4) with a
one-cycle learning rate to 2e-3 (10 % warm-up, cosine), gradient clip 5, TF32 matmuls, the shipped feature
normalisation (data/feature_norm.npz), 3-logistic head. The run draws `--total-windows` windows in a fixed order: each
group (corpus family, the dump's --group) gets a share of the scored symbols -- the budget: 'natural' (every symbol
once), 'equal', 'sqrt' (share ~ sqrt(symbols of the group)), 'pow:A'; '+wild=0.04' raises a group's share to at least
0.04 -- converted to window shares; inside a group every window is equally likely, drawn without replacement.
Outputs: DIR/final.pt ({'state', 'args'}), DIR/hist.json (train / dev bits per symbol), DIR/model_int.npz (export).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from .pool import CTX, OUT, Pool

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


# ----------------------------------------------------------------------------------------------------------------
# the window schedule
# ----------------------------------------------------------------------------------------------------------------
def parse_budget(spec, sym):
    """'sqrt+wild=0.04' -> {group: symbol share} over the groups with symbols."""
    base, *floors = spec.split('+')
    gs = [g for g in sym if sym[g] > 0]
    if base == 'natural':
        w = {g: float(sym[g]) for g in gs}
    elif base == 'equal':
        w = {g: 1.0 for g in gs}
    elif base == 'sqrt':
        w = {g: float(sym[g]) ** 0.5 for g in gs}
    elif base.startswith('pow:'):
        w = {g: float(sym[g]) ** float(base[4:]) for g in gs}
    else:
        raise ValueError(f'unknown budget {spec}')
    tot = sum(w.values()); s = {g: w[g] / tot for g in gs}
    for fl in floors:
        g, v = fl.split('='); v = float(v)
        if g in s and s[g] < v:
            rest = 1.0 - s[g]
            for h in s:
                if h != g:
                    s[h] *= (1.0 - v) / rest
            s[g] = v
    return s


def schedule(pool, spec, total, seed):
    """-> (curve, window start) of `total` training windows in training order."""
    cs_all, ws_all = pool.windows(OUT)
    grp = pool.group[cs_all]
    names = pool.groups
    sym = {names[g]: int(pool.n[pool.group == g].sum()) for g in range(len(names))}
    W = {names[g]: int((grp == g).sum()) for g in range(len(names))}
    share = parse_budget(spec, sym)
    wsh = {g: share[g] * W[g] / max(sym[g], 1) for g in share}
    tot = sum(wsh.values())
    raw = {g: total * wsh[g] / tot for g in share}
    T = {g: int(np.floor(v)) for g, v in raw.items()}
    for g in sorted(raw, key=lambda g: raw[g] - T[g], reverse=True)[:total - sum(T.values())]:
        T[g] += 1
    cs, ws = [], []
    for gk, g in enumerate(names):
        idx = np.flatnonzero(grp == gk)
        need = T.get(g, 0); q = 0
        while need > 0 and len(idx):
            take = min(need, len(idx))
            pick = np.random.default_rng([seed, gk, q]).permutation(len(idx))[:take]
            cs.append(cs_all[idx[pick]]); ws.append(ws_all[idx[pick]])
            need -= take; q += 1
    cs = np.concatenate(cs); ws = np.concatenate(ws)
    order = np.random.default_rng([seed, 999]).permutation(len(cs))
    return cs[order], ws[order], share


def batches(pool, cs, ws, batch, device, workers=4, ahead=6):
    """Gathered batches in order, computed ahead by threads (the C gather releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    import torch
    starts = list(range(0, len(cs) - batch + 1, batch))

    def job(i):
        x, y, m = pool.gather(cs[i:i + batch], ws[i:i + batch], CTX, OUT)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, i) for i in starts[:ahead]]
        for k in range(len(starts)):
            x, y, m = futs[k].result()
            if k + ahead < len(starts):
                futs.append(ex.submit(job, starts[k + ahead]))
            futs[k] = None
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)


def dev_bits(net, pool, device, batch=128, max_windows=4096):
    import torch
    from .net import nll_bits
    cs, ws = pool.windows(OUT)
    if len(cs) > max_windows:
        sel = np.sort(np.random.default_rng(1).choice(len(cs), max_windows, replace=False)); cs, ws = cs[sel], ws[sel]
    net.eval(); tot = 0.0; cnt = 0.0
    with torch.no_grad():
        for i in range(0, len(cs), batch):
            x, y, m = pool.gather(cs[i:i + batch], ws[i:i + batch], CTX, OUT)
            x, y, m = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(m).to(device)
            b = nll_bits(*net(x), y)
            tot += float((b * m).sum()); cnt += float(m.sum())
    net.train()
    return tot / max(cnt, 1.0)


def main(argv=None):
    import torch
    from .net import make_net, nll_bits
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pool', required=True); ap.add_argument('--dev-pool', default=''); ap.add_argument('--out', required=True)
    ap.add_argument('--layers', type=int, default=2); ap.add_argument('--hidden', type=int, default=64); ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--window', type=int, default=128); ap.add_argument('--ff', type=int, default=4); ap.add_argument('--mix', type=int, default=3)
    ap.add_argument('--budget', default='sqrt'); ap.add_argument('--total-windows', type=int, default=40_000_000)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--amp', default='tf32', choices=['tf32', 'none'])
    ap.add_argument('--max-steps', type=int, default=0, help='stop after this many steps (smoke runs)')
    ap.add_argument('--dev-every', type=int, default=2000, help='steps between dev scores')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = a.amp == 'tf32'; torch.backends.cudnn.allow_tf32 = a.amp == 'tf32'
    pool = Pool(a.pool); devp = Pool(a.dev_pool) if a.dev_pool else None
    cs, ws, share = schedule(pool, a.budget, a.total_windows, a.seed)
    steps = len(cs) // a.batch
    print(f'pool {pool.N / 1e6:.2f} M symbols, {len(pool.groups)} groups; {len(cs):,} windows = {steps:,} steps; symbol shares '
          + ', '.join(f'{g} {v:.3f}' for g, v in share.items()), flush=True)
    torch.manual_seed(a.seed)
    net = make_net(dict(hidden=a.hidden, layers=a.layers, heads=a.heads, window=a.window, ff=a.ff, mix=a.mix))
    norm = np.load(os.path.join(DATA, 'feature_norm.npz'))
    net.mu_f.copy_(torch.from_numpy(norm['mu_f'])); net.sd_f.copy_(torch.from_numpy(norm['sd_f']))
    net = net.to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)
    args = dict(hidden=a.hidden, layers=a.layers, heads=a.heads, window=a.window, ff=a.ff, mix=a.mix, ctx=CTX, out=OUT, batch=a.batch,
                lr=a.lr, wd=a.wd, seed=a.seed, budget=a.budget, budget_unit='symbols', total_windows=a.total_windows, amp=a.amp)
    hist = dict(train=[], dev=[])
    if devp is not None:
        hist['dev'].append(dict(step=0, bits=dev_bits(net, devp, a.device)))
        print(f'  step 0: dev {hist["dev"][-1]["bits"]:.4f} bits / symbol', flush=True)
    net.train(); t0 = time.time(); acc_b = acc_n = 0.0; step = 0
    for x, y, m in batches(pool, cs, ws, a.batch, a.device, a.workers):
        b = nll_bits(*net(x), y)
        loss = (b * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step(); step += 1
        acc_b += float(loss); acc_n += 1
        if step % 100 == 0:
            hist['train'].append(dict(step=step, bits=acc_b / acc_n, lr=sched.get_last_lr()[0], s=time.time() - t0))
            print(f'  step {step:,}/{steps:,}  train {acc_b / acc_n:.4f} bits / symbol  lr {sched.get_last_lr()[0]:.2e}  '
                  f'{step * a.batch / (time.time() - t0):.0f} windows / s', flush=True)
            acc_b = acc_n = 0.0
        if devp is not None and step % a.dev_every == 0:
            hist['dev'].append(dict(step=step, bits=dev_bits(net, devp, a.device)))
            print(f'  step {step:,}: dev {hist["dev"][-1]["bits"]:.4f} bits / symbol', flush=True)
        if step >= steps or (a.max_steps and step >= a.max_steps):
            break
    if devp is not None and hist['dev'][-1]['step'] != step:
        hist['dev'].append(dict(step=step, bits=dev_bits(net, devp, a.device)))
        print(f'  step {step:,}: dev {hist["dev"][-1]["bits"]:.4f} bits / symbol', flush=True)
    ck = os.path.join(a.out, 'final.pt')
    torch.save(dict(state={k: v.detach().cpu() for k, v in net.state_dict().items()}, args=args), ck)
    json.dump(hist, open(os.path.join(a.out, 'hist.json'), 'w'), indent=1)
    from .export import export
    sha = export(ck, os.path.join(a.out, 'model_int.npz'))
    print(f'done: {step:,} steps in {time.time() - t0:.0f} s; {ck}; model_int.npz sha256 {sha}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
