"""Benchmark ratings: model-based reliability, headroom, and held-out skill.

Reliability here is the share of between-model score variance that reflects
capability rather than noise, computed from the fitted model for the models
actually scored on each benchmark:

    R_b = Var_m(p_mb) / (Var_m(p_mb) + E_m[p_mb (1 - p_mb) / (kappa_b + 1)])

It is the model-based analogue of marginal reliability in test theory. Held-out
skill is its empirical check: R_b and skill should move together, and where
they diverge the model is wrong about that benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .data import Dataset
from .models import HierBetaFit
from .scoring import crps_samples  # noqa: F401  (re-exported for callers)

# Presentation bands, fixed before any real-data fit (see docs/protocol.md).
GRADE_BANDS = ((0.85, "A"), (0.70, "B"), (0.50, "C"))
GRADE_PROB = 0.90
MIN_MODELS_FOR_GRADE = 10
SATURATED_SCORE = 0.90
SATURATED_SHARE = 0.50
MAX_RHAT = 1.01  # rank-normalised split-R-hat (Vehtari et al. 2021), amendment 5


def reliability_draws(ds: Dataset, th: np.ndarray, d: np.ndarray, a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Reliability per draw and benchmark; arrays are (n_draws, n_models) or (n_draws, n_benchmarks)."""
    n = th.shape[0]
    out = np.full((n, ds.n_benchmarks), np.nan)
    for b in range(ds.n_benchmarks):
        ms = ds.model_idx[ds.bench_idx == b]
        if len(ms) < 3:
            continue
        p = expit(a[:, [b]] * (th[:, ms] - d[:, [b]]))
        signal = p.var(axis=1)
        noise = (p * (1 - p)).mean(axis=1) / (k[:, [b]][:, 0] + 1)
        out[:, b] = signal / (signal + noise)
    return out


def reliability_draws_all_models(ds: Dataset, th: np.ndarray, d: np.ndarray, a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Reliability each benchmark would have if every model in the table had been scored on it.

    The protocol's reliability is over the models that were actually scored, so a
    benchmark run only on a narrow slice of capability reads low however sharp it
    is. This holds the population fixed (all models in the table) and is reported
    next to it (protocol amendment 4).
    """
    n = th.shape[0]
    out = np.full((n, ds.n_benchmarks), np.nan)
    for b in range(ds.n_benchmarks):
        if (ds.bench_idx == b).sum() < 3:
            continue
        p = expit(a[:, [b]] * (th - d[:, [b]]))
        signal = p.var(axis=1)
        noise = (p * (1 - p)).mean(axis=1) / (k[:, b] + 1)
        out[:, b] = signal / (signal + noise)
    return out


def grade(draws: np.ndarray, n_models: int) -> str:
    """Letter from the posterior of R: the highest band that R clears with probability >= GRADE_PROB."""
    if n_models < MIN_MODELS_FOR_GRADE or np.isnan(draws).any():
        return "ungraded"
    for threshold, letter in GRADE_BANDS:
        if (draws >= threshold).mean() >= GRADE_PROB:
            return letter
    return "D"


@dataclass
class BenchmarkRow:
    benchmark: str
    n_models: int
    reliability_mean: float
    reliability_lo: float
    reliability_hi: float
    grade: str
    difficulty: float
    slope: float
    noise_points: float  # sd of a score at p = 0.5, in score points
    ceiling_share: float  # share of models with score >= SATURATED_SCORE
    recent_ceiling_share: float | None  # same, for the newest quarter of models
    saturated: bool
    all_models_reliability_mean: float | None = None  # same benchmark, population fixed at every model in the table
    all_models_reliability_lo: float | None = None
    all_models_reliability_hi: float | None = None
    capability_spread: float | None = None  # sd of capability among the models scored, over the sd among all models
    rhat: float | None = None  # worst split-R-hat of the benchmark's parameters; None without MCMC


def rate_benchmarks(ds: Dataset, fit: HierBetaFit, rng: np.random.Generator, n_draws: int = 1000) -> list[BenchmarkRow]:
    if not fit.has_mcmc:
        raise ValueError("rate_benchmarks needs an MCMC fit (HierBeta(mcmc=True)): grades are gated on split-R-hat")
    X = fit.draws(rng, n_draws)
    th, d, a, k = fit.split(X)
    rel = reliability_draws(ds, th, d, a, k)
    rel_all = reliability_draws_all_models(ds, th, d, a, k)
    sd_all = th.std(axis=1)
    dated = [i for i, x in enumerate(ds.model_date) if x]
    recent: set[int] = set()
    if len(dated) >= 8:
        order = sorted(dated, key=lambda i: ds.model_date[i])
        recent = set(order[int(len(order) * 0.75):])
    rhat = fit.benchmark_rhat() if fit.has_mcmc else None
    rows = []
    for b in range(ds.n_benchmarks):
        sel = ds.bench_idx == b
        sc = ds.score[sel]
        rec = np.isin(ds.model_idx[sel], list(recent)) if recent else np.zeros(sel.sum(), bool)
        rec_share = float((sc[rec] >= SATURATED_SCORE).mean()) if rec.sum() >= 3 else None
        r = rel[:, b]
        rows.append(BenchmarkRow(
            ds.benchmarks[b], int(sel.sum()),
            float(np.nanmean(r)), float(np.nanquantile(r, 0.05)), float(np.nanquantile(r, 0.95)),
            "ungraded" if rhat is not None and not rhat[b] <= MAX_RHAT else grade(r, int(sel.sum())),
            float(d[:, b].mean()), float(a[:, b].mean()),
            float(np.sqrt(0.25 / (k[:, b] + 1)).mean()),
            float((sc >= SATURATED_SCORE).mean()), rec_share,
            bool(rec_share is not None and rec_share >= SATURATED_SHARE),
            float(np.nanmean(rel_all[:, b])), float(np.nanquantile(rel_all[:, b], 0.05)),
            float(np.nanquantile(rel_all[:, b], 0.95)),
            float(np.mean(th[:, np.unique(ds.model_idx[sel])].std(axis=1) / sd_all)),
            None if rhat is None else float(rhat[b]),
        ))
    return rows


def true_reliability(ds: Dataset, theta, difficulty, slope, precision, floor=None) -> np.ndarray:
    """Reliability from known parameters (simulation only); ``floor`` is a per-benchmark chance level if given."""
    if floor is None:
        return reliability_draws(ds, theta[None, :], difficulty[None, :], slope[None, :], precision[None, :])[0]
    out = np.full(ds.n_benchmarks, np.nan)
    for b in range(ds.n_benchmarks):
        models = np.unique(ds.model_idx[ds.bench_idx == b])
        if len(models) < 3:
            continue
        p = floor[b] + (1 - floor[b]) * expit(slope[b] * (theta[models] - difficulty[b]))
        signal, noise = p.var(), (p * (1 - p)).mean() / (precision[b] + 1)
        out[b] = signal / (signal + noise)
    return out


def heldout_skill(ds: Dataset, results: dict[str, "object"], baseline: str = "benchmark_mean",
                  method: str = "hier_beta", n_boot: int = 500, seed: int = 0) -> dict[int, tuple[float, float, float, int]]:
    """Per-benchmark skill = 1 - CRPS(method) / CRPS(baseline) on held-out cells, bootstrap over cells.

    ``results`` maps method name to ``evaluate.Result`` for one split. Returns
    {benchmark index: (skill, lo, hi, n_cells)}.
    """
    rm, rb = results[method], results[baseline]
    order_m, order_b = np.argsort(rm.cell), np.argsort(rb.cell)
    cells = rm.cell[order_m]
    cm, cb = rm.crps[order_m], rb.crps[order_b]
    bench = ds.bench_idx[cells]
    rng = np.random.default_rng(seed)
    out = {}
    for b in np.unique(bench):
        sel = np.flatnonzero(bench == b)
        if len(sel) < 5:
            continue
        skill = 1 - cm[sel].mean() / cb[sel].mean()
        idx = rng.integers(0, len(sel), size=(n_boot, len(sel)))
        boots = 1 - cm[sel][idx].mean(axis=1) / cb[sel][idx].mean(axis=1)
        out[int(b)] = (float(skill), float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95)), len(sel))
    return out
