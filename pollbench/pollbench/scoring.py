"""Proper scores and interval checks for sample-based predictive distributions."""

from __future__ import annotations

import numpy as np


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of each predictive sample set against its observation.

    ``samples`` has shape (n_cells, n_draws). Uses the energy form
    E|X - y| - 0.5 E|X - X'|, with E|X - X'| from the sorted-sample identity.
    """
    s = np.sort(samples, axis=1)
    n = s.shape[1]
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    gini = 2.0 * (s * weights).sum(axis=1) / (n * (n - 1))  # E|X - X'| over distinct pairs (unbiased)
    return term1 - 0.5 * gini


def interval_hit(samples: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    """1.0 where y falls inside the central ``level`` predictive interval."""
    lo = np.quantile(samples, (1 - level) / 2, axis=1)
    hi = np.quantile(samples, 1 - (1 - level) / 2, axis=1)
    return ((y >= lo) & (y <= hi)).astype(float)


def paired_bootstrap(diff_by_model: dict[str, float], *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% interval of a per-model score difference, resampling models.

    Cells of one model share an estimated capability, so models, not cells,
    are the exchangeable unit.
    """
    vals = np.array(list(diff_by_model.values()), dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def crossed_bootstrap(model_idx: np.ndarray, bench_idx: np.ndarray, diff: np.ndarray, *,
                      n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% interval of a per-cell score difference, resampling models and benchmarks together.

    Each cell shares a model effect and a benchmark effect, so both are
    exchangeable units. A bootstrap draw gives every model and every benchmark a
    multinomial count and weights a cell by the product of its two counts.
    """
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    nm, nb = int(model_idx.max()) + 1, int(bench_idx.max()) + 1
    means = np.empty(n_boot)
    for i in range(n_boot):
        cm = np.bincount(rng.integers(0, nm, nm), minlength=nm)
        cb = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        w = cm[model_idx] * cb[bench_idx]
        means[i] = (w * diff).sum() / w.sum() if w.sum() > 0 else np.nan
    means = means[~np.isnan(means)]
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
