"""Export the primary fit's curves and each model's capability, for the results page's figures.

The results JSON holds per-benchmark summaries only. The page's teaching figure
plots real scores against fitted capability, which needs the per-model
capability and the per-benchmark curve. This refits the primary model with the
same seed and sampler as ``analysis.run`` and checks that it reproduces the
committed reliabilities before writing anything.

    python -m pollbench.curves processed_data_for_eci.csv --results results/epoch-2026-10-10.json --out results/epoch-2026-10-10-curves.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import _final_fit
from .data import load_epoch_csv


def export(ds, fit, rows, seed: int) -> dict:
    rng = np.random.default_rng(seed + 5)
    X = fit.draws(rng, 1000)
    th, d, a, k = fit.split(X)
    th_mean, th_sd = th.mean(0), th.std(0)
    out = {"data_digest": ds.digest(), "seed": seed,
           "capability": {m: [round(float(th_mean[i]), 4), round(float(th_sd[i]), 4)] for i, m in enumerate(ds.models)},
           "benchmarks": {}}
    for b, name in enumerate(ds.benchmarks):
        sel = np.flatnonzero(ds.bench_idx == b)
        out["benchmarks"][name] = {
            "difficulty": round(float(d[:, b].mean()), 4), "slope": round(float(a[:, b].mean()), 4),
            "precision": round(float(k[:, b].mean()), 4),
            "scores": [[ds.models[ds.model_idx[c]], round(float(ds.score[c]), 4)] for c in sel],
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--results", required=True, help="committed results JSON the refit must reproduce")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    ds = load_epoch_csv(args.data)
    committed = json.loads(Path(args.results).read_text())
    if ds.digest() != committed["data_digest"]:
        raise SystemExit("data digest differs from the committed results")
    rows, _, fit = _final_fit(ds, args.seed)
    want = {r["benchmark"]: r["reliability_mean"] for r in committed["benchmarks"]}
    worst = max(abs(r.reliability_mean - want[r.benchmark]) for r in rows)
    if worst > 1e-6:
        raise SystemExit(f"refit does not reproduce the committed reliabilities (worst difference {worst:.2e})")
    Path(args.out).write_text(json.dumps(export(ds, fit, rows, args.seed), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}; refit matches committed reliabilities to {worst:.1e}")


if __name__ == "__main__":
    main()
