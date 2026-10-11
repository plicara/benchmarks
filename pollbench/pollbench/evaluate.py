"""Held-out evaluation: split cells, fit every method, grade predictive samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset
from .scoring import crps_samples, interval_hit

MIN_KEPT = 3  # every model and benchmark keeps at least this many training cells


@dataclass
class Split:
    name: str
    held: np.ndarray  # fold id per cell, -1 = always train


def _enforce_minimums(ds: Dataset, fold: np.ndarray, k: int) -> np.ndarray:
    """Return cells to training until each fit keeps MIN_KEPT cells per model and benchmark."""
    fold = fold.copy()
    for f in range(k):
        for _ in range(50):
            train = fold != f
            m_count = np.bincount(ds.model_idx[train], minlength=ds.n_models)
            b_count = np.bincount(ds.bench_idx[train], minlength=ds.n_benchmarks)
            bad = (fold == f) & ((m_count[ds.model_idx] < MIN_KEPT) | (b_count[ds.bench_idx] < MIN_KEPT))
            if not bad.any():
                break
            # A held cell on a thin model or benchmark goes back to training.
            thin_m = np.flatnonzero(m_count < MIN_KEPT)
            thin_b = np.flatnonzero(b_count < MIN_KEPT)
            back = (fold == f) & (np.isin(ds.model_idx, thin_m) | np.isin(ds.bench_idx, thin_b))
            fold[back] = -1
    return fold


def kfold_cells(ds: Dataset, k: int = 5, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, size=ds.n_cells)
    return Split(f"cells_{k}fold", _enforce_minimums(ds, fold, k))


def frontier_split(ds: Dataset, frac: float = 0.25, seed: int = 0) -> Split:
    """Hold out half the cells of the newest models; earlier models stay fully in training."""
    rng = np.random.default_rng(seed)
    dated = [i for i, d in enumerate(ds.model_date) if d]
    if len(dated) < 8:
        raise ValueError("frontier split needs at least 8 dated models")
    order = sorted(dated, key=lambda i: ds.model_date[i])
    newest = set(order[int(len(order) * (1 - frac)):])
    fold = np.full(ds.n_cells, -1)
    is_new = np.isin(ds.model_idx, list(newest))
    fold[is_new & (rng.uniform(size=ds.n_cells) < 0.5)] = 0
    return Split("frontier", _enforce_minimums(ds, fold, 1))


@dataclass
class Result:
    split: str
    method: str
    cell: np.ndarray  # index into the Dataset's cells
    crps: np.ndarray
    hit50: np.ndarray
    hit90: np.ndarray
    mean_pred: np.ndarray


def evaluate(ds: Dataset, methods, split: Split, *, n_draws: int = 300, seed: int = 0) -> list[Result]:
    rng = np.random.default_rng(seed)
    folds = sorted(set(split.held.tolist()) - {-1})
    out = []
    for method in methods:
        cells, crps, h50, h90, mp = [], [], [], [], []
        for f in folds:
            test = split.held == f
            fitted = method.fit(ds.subset(~test))
            samp = fitted.predict(ds.model_idx[test], ds.bench_idx[test], n_draws, rng)
            y = ds.score[test]
            cells.append(np.flatnonzero(test))
            crps.append(crps_samples(samp, y))
            h50.append(interval_hit(samp, y, 0.5))
            h90.append(interval_hit(samp, y, 0.9))
            mp.append(samp.mean(axis=1))
        out.append(Result(split.name, method.name, np.concatenate(cells), np.concatenate(crps),
                          np.concatenate(h50), np.concatenate(h90), np.concatenate(mp)))
    return out
