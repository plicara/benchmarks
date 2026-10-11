"""Score tables: one row per (model, benchmark) with a score in [0, 1].

The loader reads Epoch's ``benchmark_data.zip``-style CSV without any
network access. Nothing here fabricates or imputes scores: rows that cannot
be used are dropped and counted so the report can say how many.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Scores of exactly 0 or 1 break the Beta likelihood; clip symmetrically.
SCORE_EPS = 0.005


@dataclass
class Dataset:
    """Sparse model-by-benchmark score matrix in coordinate form."""

    model_idx: np.ndarray
    bench_idx: np.ndarray
    score: np.ndarray
    models: list[str]
    benchmarks: list[str]
    model_date: list[str]  # ISO date or "" when unknown
    bench_date: list[str]
    source: list[str] = field(default_factory=list)  # per-row provenance label
    dropped: dict[str, int] = field(default_factory=dict)
    raw: np.ndarray | None = None  # scores before clipping, so the clip level can be varied
    eps: float = SCORE_EPS
    file_sha256: str = ""  # hash of the input file bytes when loaded from disk

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_benchmarks(self) -> int:
        return len(self.benchmarks)

    @property
    def n_cells(self) -> int:
        return len(self.score)

    def subset(self, mask: np.ndarray) -> "Dataset":
        """Rows selected by ``mask``; model and benchmark indexing is kept."""
        return Dataset(
            self.model_idx[mask], self.bench_idx[mask], self.score[mask],
            self.models, self.benchmarks, self.model_date, self.bench_date,
            [s for s, keep in zip(self.source, mask) if keep] if self.source else [],
            dict(self.dropped), None if self.raw is None else self.raw[mask], self.eps, self.file_sha256,
        )

    def with_clip(self, eps: float) -> "Dataset":
        """The same cells clipped at ``eps`` instead of the default, for the clip-level sensitivity arm."""
        raw = self.raw if self.raw is not None else self.score
        out = self.subset(np.ones(self.n_cells, dtype=bool))
        out.raw, out.eps = raw, eps
        out.score = np.clip(raw, eps, 1 - eps)
        return out

    def rows(self) -> list[dict]:
        """The cells as the dicts ``build_dataset`` takes, so a subset can be re-filtered."""
        raw = self.raw if self.raw is not None else self.score
        return [{"model": self.models[m], "benchmark": self.benchmarks[b], "score": float(sc),
                 "model_date": self.model_date[m], "bench_date": self.bench_date[b],
                 "source": self.source[i] if self.source else ""}
                for i, (m, b, sc) in enumerate(zip(self.model_idx, self.bench_idx, raw))]

    def restrict(self, keep, **filters) -> "Dataset":
        """Rebuild from the rows for which ``keep(row)`` holds, re-applying the minimum-count filters."""
        filters.setdefault("eps", self.eps)
        out = build_dataset([r for r in self.rows() if keep(r)], **filters)
        out.file_sha256 = self.file_sha256
        return out

    def digest(self) -> str:
        """Stable hash of everything the analysis reads: cells, sources and the dates the splits use."""
        h = hashlib.sha256()
        order = np.lexsort((self.bench_idx, self.model_idx))
        for i in order:
            src = self.source[i] if self.source else ""
            h.update(f"{self.models[self.model_idx[i]]}|{self.benchmarks[self.bench_idx[i]]}|{self.score[i]:.9f}|{src}\n".encode())
        for m, d in zip(self.models, self.model_date):
            h.update(f"M|{m}|{d}\n".encode())
        for b, d in zip(self.benchmarks, self.bench_date):
            h.update(f"B|{b}|{d}\n".encode())
        return h.hexdigest()


def _modal(dates: dict[str, list[str]]) -> dict[str, str]:
    """The most frequent date per key; ties go to the earliest, so the choice does not depend on row order."""
    out = {}
    for k, vals in dates.items():
        counts: dict[str, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        out[k] = min(counts, key=lambda v: (-counts[v], v))
    return out


def build_dataset(rows: list[dict], *, min_models_per_benchmark: int = 5,
                  min_benchmarks_per_model: int = 4, eps: float = SCORE_EPS) -> Dataset:
    """Build a Dataset from raw rows, filtering until both minimums hold.

    Each row needs ``model``, ``benchmark`` and ``score``; ``model_date``,
    ``bench_date`` and ``source`` are optional. A model scored more than once
    on a benchmark keeps the maximum, which is the choice Epoch's ECI makes
    for models evaluated under several setups; the number of collapsed rows is
    reported in ``dropped["duplicate_cells_collapsed"]``.
    """
    dropped = {"non_numeric_score": 0, "out_of_range_score": 0, "duplicate_cells_collapsed": 0,
               "benchmarks_below_min_models": 0, "models_below_min_benchmarks": 0}
    cell: dict[tuple[str, str], tuple[float, str]] = {}
    mdates: dict[str, list[str]] = {}
    bdates: dict[str, list[str]] = {}
    for r in rows:
        try:
            s = float(r["score"])
        except (TypeError, ValueError):
            dropped["non_numeric_score"] += 1
            continue
        if not (0.0 <= s <= 1.0) or s != s:
            dropped["out_of_range_score"] += 1
            continue
        key = (r["model"], r["benchmark"])
        if key in cell:
            dropped["duplicate_cells_collapsed"] += 1
            if s <= cell[key][0]:
                continue
        cell[key] = (s, r.get("source", "") or "")
        if r.get("model_date"):
            mdates.setdefault(r["model"], []).append(r["model_date"])
        if r.get("bench_date"):
            bdates.setdefault(r["benchmark"], []).append(r["bench_date"])

    # Iteratively drop sparse benchmarks and models until both minimums hold.
    changed = True
    while changed:
        changed = False
        per_b: dict[str, int] = {}
        per_m: dict[str, int] = {}
        for (m, b) in cell:
            per_b[b] = per_b.get(b, 0) + 1
            per_m[m] = per_m.get(m, 0) + 1
        drop_b = {b for b, n in per_b.items() if n < min_models_per_benchmark}
        drop_m = {m for m, n in per_m.items() if n < min_benchmarks_per_model}
        if drop_b or drop_m:
            dropped["benchmarks_below_min_models"] += len(drop_b)
            dropped["models_below_min_benchmarks"] += len(drop_m)
            cell = {k: v for k, v in cell.items() if k[0] not in drop_m and k[1] not in drop_b}
            changed = True

    mdate, bdate = _modal(mdates), _modal(bdates)
    models = sorted({m for m, _ in cell})
    benches = sorted({b for _, b in cell})
    mi = {m: i for i, m in enumerate(models)}
    bi = {b: i for i, b in enumerate(benches)}
    keys = sorted(cell, key=lambda k: (mi[k[0]], bi[k[1]]))
    raw = np.array([cell[k][0] for k in keys])
    return Dataset(
        np.array([mi[m] for m, _ in keys], dtype=int),
        np.array([bi[b] for _, b in keys], dtype=int),
        np.clip(raw, eps, 1 - eps),
        models, benches,
        [mdate.get(m, "") for m in models], [bdate.get(b, "") for b in benches],
        [cell[k][1] for k in keys], dropped, raw, eps,
    )


def load_epoch_csv(path: str | Path, **kwargs) -> Dataset:
    """Read the long-format CSV: model, benchmark, performance, dates, source.

    Column names follow the file Epoch's loader consumes
    (``model_id, benchmark_id, performance, benchmark, benchmark_release_date,
    model, model_version, Model, date, source``). ``Model`` is the display
    name; ``model`` is the fallback.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"performance", "benchmark"} - set(reader.fieldnames or [])
        if missing or not ({"Model", "model"} & set(reader.fieldnames or [])):
            raise ValueError(f"unexpected columns {reader.fieldnames}; missing {sorted(missing)}")
        for r in reader:
            rows.append({
                "model": (r.get("Model") or r.get("model") or "").strip(),
                "benchmark": (r.get("benchmark") or "").strip(),
                "score": r.get("performance"),
                "model_date": (r.get("date") or "").strip(),
                "bench_date": (r.get("benchmark_release_date") or "").strip(),
                "source": (r.get("source") or "").strip(),
            })
    ds = build_dataset([r for r in rows if r["model"] and r["benchmark"]], **kwargs)
    ds.file_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return ds


_DEVELOPER_MARKERS = (
    "technical report", "system card", "model card", "announcement", "annoucement", "report", "herd of models",
    "claude 3 model family", "releasing claude instant",
    "open foundation", "open models", "mixtral", "llama", "gemma", "falcon", "baichuan", "yi:",
    "textbooks are all you need", "training compute-optimal",
)


def is_developer_reported(source: str) -> bool:
    """Protocol amendment 1: the cell's source is a developer's own report, card, paper or announcement."""
    low = (source or "").lower()
    return any(m in low for m in _DEVELOPER_MARKERS)
