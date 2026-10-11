"""Synthetic score tables with known truth.

These exist to test the methods, not to stand in for data: every table is
labelled synthetic in its provenance, and nothing here feeds a published
rating. The generator mirrors the structure that makes real tables hard:
models arrive over time and are scored on the benchmarks that exist, coverage
is sparse, benchmarks differ in difficulty, slope and noise, and optionally a
second capability dimension and selective reporting are switched on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .data import Dataset, SCORE_EPS


@dataclass
class Truth:
    theta: np.ndarray
    difficulty: np.ndarray
    slope: np.ndarray
    precision: np.ndarray
    loading2: np.ndarray | None = None
    floor: np.ndarray | None = None


def simulate(n_models: int = 120, n_benchmarks: int = 40, *, seed: int = 0,
             coverage: float = 0.18, second_factor: float = 0.0,
             selective_reporting: float = 0.0, floor_share: float = 0.0) -> tuple[Dataset, Truth]:
    """Draw a sparse table from the sigmoid-Beta model.

    ``second_factor`` is the sd of a benchmark-specific loading on an extra
    model trait (misspecification: capability is not one number).
    ``selective_reporting`` is the probability that a cell whose score is in
    the bottom quartile of its benchmark goes unreported.
    ``floor_share`` is the share of benchmarks with a chance floor (0.25 or 0.5):
    expected score is f + (1 - f) sigmoid, which the one-factor model does not have.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1, n_models))  # model release time
    theta = 3.0 * (t - 0.5) + 0.6 * rng.standard_normal(n_models)  # newer models are stronger
    bench_time = np.sort(rng.uniform(0, 0.9, n_benchmarks))
    difficulty = -1.5 + 4.0 * bench_time + 0.7 * rng.standard_normal(n_benchmarks)  # newer benchmarks are harder
    slope = np.exp(0.2 + 0.5 * rng.standard_normal(n_benchmarks))
    precision = np.exp(np.log(40) + 0.9 * rng.standard_normal(n_benchmarks))
    trait2 = rng.standard_normal(n_models)
    floor = None
    if floor_share:
        floor = np.where(rng.uniform(size=n_benchmarks) < floor_share, rng.choice([0.25, 0.5], size=n_benchmarks), 0.0)
    loading2 = second_factor * rng.standard_normal(n_benchmarks) if second_factor else None

    mi, bi, sc = [], [], []
    for m in range(n_models):
        avail = np.flatnonzero(bench_time <= t[m] + 0.15)
        if len(avail) == 0:
            avail = np.array([0])
        k = max(4, rng.binomial(len(avail), min(1.0, coverage * n_benchmarks / max(len(avail), 1))))
        k = min(k, len(avail))
        for b in rng.choice(avail, size=k, replace=False):
            z = slope[b] * (theta[m] - difficulty[b])
            if loading2 is not None:
                z = z + loading2[b] * trait2[m]
            p = expit(z)
            if floor is not None:
                p = floor[b] + (1 - floor[b]) * p
            p = np.clip(p, 1e-6, 1 - 1e-6)
            mi.append(m); bi.append(b)
            sc.append(rng.beta(p * precision[b], (1 - p) * precision[b]))
    mi, bi, sc = np.array(mi), np.array(bi), np.clip(np.array(sc), SCORE_EPS, 1 - SCORE_EPS)

    if selective_reporting > 0:
        keep = np.ones(len(sc), bool)
        for b in range(n_benchmarks):
            sel = np.flatnonzero(bi == b)
            if len(sel) < 6:
                continue
            q = np.quantile(sc[sel], 0.25)
            low = sel[sc[sel] <= q]
            keep[low] = rng.uniform(size=len(low)) > selective_reporting
        mi, bi, sc = mi[keep], bi[keep], sc[keep]

    # Mirror the loader's minimums so synthetic and real tables are filtered alike.
    from .data import build_dataset
    rows = [{"model": f"m{m:03d}", "benchmark": f"b{b:03d}", "score": s,
             "model_date": f"{2023 + int(t[m] * 3)}-{1 + int(t[m] * 36) % 12:02d}-15",
             "bench_date": f"{2023 + int(bench_time[b] * 3)}-{1 + int(bench_time[b] * 36) % 12:02d}-01",
             "source": "synthetic"} for m, b, s in zip(mi, bi, sc)]
    ds = build_dataset(rows)
    # Re-map truth to the filtered indexing.
    mkeep = [int(n[1:]) for n in ds.models]
    bkeep = [int(n[1:]) for n in ds.benchmarks]
    truth = Truth(theta[mkeep], difficulty[bkeep], slope[bkeep], precision[bkeep],
                  loading2[bkeep] if loading2 is not None else None,
                  floor[bkeep] if floor is not None else None)
    return ds, truth
