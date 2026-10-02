"""The mean-error contract: a decoded clip must match ACL's own result at the same precision on the same clip.

With m_p = ACL's clip mean shell error at precision p (optionally floored at F x p), ACL's per-bone mean / max errors
a_mean[b] / a_max[b] and our decoded error field e [F, B]:

  per bone   mean_f e[:, b] <= C x max(m_p, a_mean[b])
  clip       mean e         <= T x m_p
  anti-pop   max_f e[:, b]  <= max(G x m_p, 1.05 x a_max[b])

Defaults G = 10, C = 1.05, T = 1.02, F = 0 ('mean'); any subset can be given, e.g. 'mean:C=1.5' (looser per-bone cap)
or 'mean:F=0.05' (for near-static game clips whose ACL mean is at the float32 floor, unreachable for any quantizer).

The encoder aims below the contract: its per-bone caps are (C / 1.05) x max(m_p, a_mean), its clip total is
(T / 1.02) x m_p (the default contract's margins), and the fallback margins 0.98 / 0.95 / 0.90 scale every budget it
derives from ACL's statistics when a clip misses the contract at margin 1.
"""
from dataclasses import dataclass

import numpy as np

TOL_BONE = 1.05          # the default per-bone tolerance; the encoder's cap is C / TOL_BONE
TOL_MEAN = 1.02          # the default clip tolerance; the encoder's clip factor is T / TOL_MEAN
TOL_MAX = 1.05           # anti-pop tolerance on ACL's per-bone max
MARGINS = (1.0, 0.98, 0.95, 0.90)


def _fmt(v):
    v = float(v)
    return 'inf' if not np.isfinite(v) else f'{v:g}'


@dataclass(frozen=True)
class Contract:
    G: float = 10.0          # anti-pop guard, in units of m_p
    C: float = 1.05          # per-bone mean cap
    T: float = 1.02          # clip mean tolerance
    F: float = 0.0           # budget floor: m_p >= F x p

    @classmethod
    def parse(cls, spec='mean'):
        """'mean' | 'mean:G=..,C=..,T=..,F=..' (any subset; 'inf' allowed for G and C) -> Contract."""
        mode, _, rest = str(spec).partition(':')
        if mode != 'mean':
            raise ValueError(f'unknown contract {spec!r} (mean[:G=..,C=..,T=..,F=..])')
        kw = {}
        for item in filter(None, rest.split(',')):
            k, _, v = item.partition('=')
            if k not in ('G', 'C', 'T', 'F') or not v:
                raise ValueError(f'bad contract parameter {item!r}')
            kw[k] = float(v)
        c = cls(**kw)
        if c.T < 1.0 or c.C < 1.0 or c.G <= 0 or c.F < 0:
            raise ValueError(f'contract parameters out of range: {c}')
        return c

    def __str__(self):
        s = f'mean:G={_fmt(self.G)},C={_fmt(self.C)},T={_fmt(self.T)}'
        return s + (f',F={_fmt(self.F)}' if self.F > 0 else '')

    def mean_budget(self, ref, p):
        """m_p: ACL's clip mean at p, floored at F x p."""
        m = float(ref['mean'])
        if self.F:
            m = max(m, float(self.F) * float(p))
        return m

    def budgets(self, ref, B, p, margin=1.0):
        """The encoder's targets: (m_p, per-bone mean caps L [B], per-bone quantization budgets b [B] (ACL's per-bone
        profile scaled to sum B m_p), anti-pop guard [B], clip factor t_enc), every ACL-derived budget x `margin`."""
        m_p = self.mean_budget(ref, p)
        abm = np.asarray(ref['bone_mean'], dtype=np.float64)
        abx = np.asarray(ref['bone_max'], dtype=np.float64)
        L0 = np.maximum(m_p, abm)
        b = L0 * (m_p / L0.mean()) if L0.mean() > 0 else np.full(B, m_p)
        c_enc = self.C / TOL_BONE
        L = np.full(B, np.inf) if not np.isfinite(c_enc) else c_enc * L0
        guard = np.maximum(self.G * m_p, abx)
        m = float(margin)
        return m * m_p, m * L, m * b, m * guard, self.T / TOL_MEAN

    def check(self, stats, ref, p):
        """(ok, reasons, ratios) of a decoded clip's error statistics (metric.error_stats) against ACL's at p."""
        reasons = []
        m_p = self.mean_budget(ref, p)
        acl_mean = np.asarray(ref['bone_mean'], dtype=np.float64)
        acl_max = np.asarray(ref['bone_max'], dtype=np.float64)
        bone_mean = np.asarray(stats['bone_mean'], dtype=np.float64)
        bone_max = np.asarray(stats['bone_max'], dtype=np.float64)
        lim = self.C * np.maximum(m_p, acl_mean)
        bad = np.flatnonzero(bone_mean > lim)
        if len(bad):
            w = bad[np.argmax(bone_mean[bad] / lim[bad])]
            reasons.append(f'bone_mean: {len(bad)} bones over the cap, worst bone {w}: {bone_mean[w]:.5f} > {lim[w]:.5f}')
        if stats['err_mean'] > self.T * m_p:
            reasons.append(f'mean: {stats["err_mean"]:.5f} > {self.T} x {m_p:.5f}')
        glim = np.maximum(self.G * m_p, TOL_MAX * acl_max)
        gbad = np.flatnonzero(bone_max > glim)
        if len(gbad):
            w = gbad[np.argmax(bone_max[gbad] / glim[gbad])]
            reasons.append(f'guard: {len(gbad)} bones over the anti-pop limit, worst bone {w}: {bone_max[w]:.5f} > {glim[w]:.5f}')
        ratios = dict(mean_budget=m_p, mean_ratio=float(stats['err_mean'] / m_p) if m_p > 0 else float('nan'),
                      bone_mean_ratio_worst=float(np.max(bone_mean / lim)) if len(bone_mean) else 0.0,
                      guard_ratio_worst=float(np.max(bone_max / glim)) if len(bone_max) else 0.0)
        return not reasons, reasons, ratios
