"""Posterior predictive check for a second capability factor.

The simulation study shows the one failure that the held-out checks cannot see:
benchmarks that load on a second model trait. If it is present, residuals from
the one-factor fit are correlated across benchmarks. The statistic is the mean
squared correlation of residuals over benchmark pairs sharing at least
``min_shared`` models. It is a posterior predictive check: for each posterior
draw the statistic is computed on the observed table and on a table simulated
from that same draw, both with residuals taken against that draw's expected
scores, and the p-value is the share of draws where the simulated statistic is
at least the observed one. Comparing within a draw keeps the two residual
definitions identical, which an earlier version (observed residuals against the
posterior-mean fit, replicates against the same mean) did not, and it was
anti-conservative. ``pollbench.calibrate`` runs the check on simulated tables with and
without a second factor, to put numbers on its type-I error and power.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .data import SCORE_EPS, Dataset
from .models import HierBetaFit


class _Pairs:
    """Every benchmark pair with enough shared models, as flat index arrays for a vectorised statistic."""

    def __init__(self, ds: Dataset, min_shared: int):
        by_b = [dict(zip(ds.model_idx[ds.bench_idx == b], np.flatnonzero(ds.bench_idx == b))) for b in range(ds.n_benchmarks)]
        ia, ib, lens = [], [], []
        for i in range(ds.n_benchmarks):
            for j in range(i + 1, ds.n_benchmarks):
                shared = sorted(by_b[i].keys() & by_b[j].keys())
                if len(shared) >= min_shared:
                    ia += [by_b[i][m] for m in shared]
                    ib += [by_b[j][m] for m in shared]
                    lens.append(len(shared))
        self.ia, self.ib = np.array(ia, dtype=int), np.array(ib, dtype=int)
        self.lens = np.array(lens, dtype=int)
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]]) if len(lens) else np.array([], dtype=int)

    @property
    def n_pairs(self) -> int:
        return len(self.lens)

    def statistic(self, resid: np.ndarray) -> float:
        """Mean squared correlation over the pairs; pairs with a constant side are skipped."""
        if not self.n_pairs:
            return float("nan")
        a, b = resid[self.ia], resid[self.ib]
        seg = lambda v: np.add.reduceat(v, self.starts)
        ma, mb = seg(a) / self.lens, seg(b) / self.lens
        a, b = a - np.repeat(ma, self.lens), b - np.repeat(mb, self.lens)
        saa, sbb, sab = seg(a * a), seg(b * b), seg(a * b)
        ok = (saa > 1e-12) & (sbb > 1e-12)
        return float(np.mean(sab[ok] ** 2 / (saa[ok] * sbb[ok]))) if ok.any() else float("nan")


def _expected(ds: Dataset, th, d, a, k, f=None):
    p = expit(a[ds.bench_idx] * (th[ds.model_idx] - d[ds.bench_idx]))
    return p if f is None else f[ds.bench_idx] + (1.0 - f[ds.bench_idx]) * p


def residual_pairs(ds: Dataset, fit: HierBetaFit, rng: np.random.Generator, *, min_shared: int = 12, top: int = 10) -> dict:
    """Exploratory: the most correlated residual pairs, to say where the leftover structure sits."""
    X = fit.draws(rng, 200)
    th, d, a, k = fit.split(X)
    fl = fit.floors(X)
    mean_p = np.mean([_expected(ds, th[i], d[i], a[i], k[i], None if fl is None else fl[i]) for i in range(len(th))], axis=0)
    resid = ds.score - mean_p
    by_b = [dict(zip(ds.model_idx[ds.bench_idx == b], resid[ds.bench_idx == b])) for b in range(ds.n_benchmarks)]
    out = []
    for i in range(ds.n_benchmarks):
        for j in range(i + 1, ds.n_benchmarks):
            shared = sorted(by_b[i].keys() & by_b[j].keys())
            if len(shared) >= min_shared:
                x = np.array([by_b[i][m] for m in shared])
                y = np.array([by_b[j][m] for m in shared])
                if x.std() > 1e-9 and y.std() > 1e-9:
                    out.append((float(np.corrcoef(x, y)[0, 1]), len(shared), ds.benchmarks[i], ds.benchmarks[j]))
    out.sort()
    fmt = lambda t: {"r": t[0], "n_models": t[1], "a": t[2], "b": t[3]}
    return {"n_pairs": len(out), "mean_r": float(np.mean([t[0] for t in out])) if out else None,
            "most_positive": [fmt(t) for t in out[::-1][:top]], "most_negative": [fmt(t) for t in out[:5]]}


def second_factor_check(ds: Dataset, fit: HierBetaFit, rng: np.random.Generator, *,
                        n_rep: int = 500, min_shared: int = 12) -> dict:
    pairs = _Pairs(ds, min_shared)
    X = fit.draws(rng, n_rep)
    th, d, a, k = fit.split(X)
    fl = fit.floors(X)
    obs, rep = [], []
    for i in range(n_rep):
        p = _expected(ds, th[i], d[i], a[i], k[i], None if fl is None else fl[i])
        kk = k[i][ds.bench_idx]
        y = np.clip(rng.beta(p * kk, (1 - p) * kk), SCORE_EPS, 1 - SCORE_EPS)
        obs.append(pairs.statistic(ds.score - p))
        rep.append(pairs.statistic(y - p))
    obs, rep = np.array(obs), np.array(rep)
    return {"statistic": float(np.nanmean(obs)), "n_pairs": pairs.n_pairs,
            "reference_mean": float(np.nanmean(rep)), "reference_p95": float(np.nanquantile(rep, 0.95)),
            "p_value": float((np.sum(rep >= obs) + 1) / (n_rep + 1))}
