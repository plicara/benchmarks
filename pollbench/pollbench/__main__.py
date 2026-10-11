"""Command line: ``python -m pollbench {simulate,rate}``."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import analysis
from .data import load_epoch_csv
from .evaluate import evaluate, frontier_split, kfold_cells
from .models import HierBeta, default_methods
from .ratings import heldout_skill, rate_benchmarks
from .sim import simulate


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def cmd_simulate(args) -> None:
    ds, _ = simulate(n_models=args.models, n_benchmarks=args.benchmarks, seed=args.seed)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Model", "benchmark", "performance", "date", "benchmark_release_date", "source"])
        for m, b, s in zip(ds.model_idx, ds.bench_idx, ds.score):
            w.writerow([ds.models[m], ds.benchmarks[b], f"{s:.6f}", ds.model_date[m], ds.bench_date[b], "synthetic"])
    print(f"wrote {ds.n_cells} synthetic cells to {args.out}")


def cmd_rate(args) -> None:
    ds = load_epoch_csv(args.data)
    rng = np.random.default_rng(args.seed)
    fit = HierBeta(seed=args.seed, n_chains=4, n_warmup=args.warmup, n_samples=args.samples).fit(ds)
    rows = rate_benchmarks(ds, fit, rng)
    out = {"protocol": "docs/protocol.md v1", "data_digest": ds.digest(),
           "n_models": ds.n_models, "n_benchmarks": ds.n_benchmarks, "n_cells": ds.n_cells,
           "dropped": ds.dropped, "diagnostics": {k: v for k, v in fit.diagnostics.items() if k not in ("rhat", "ess")},
           "benchmarks": [asdict(r) for r in rows]}
    if not args.no_heldout:
        splits = {"kfold": kfold_cells(ds, 5, args.seed), "frontier": frontier_split(ds, 0.25, args.seed)}
        out["heldout"] = {}
        for name, sp in splits.items():
            res = {r.method: r for r in evaluate(ds, default_methods(quick=False), sp, seed=args.seed)}
            out["heldout"][name] = {
                "mean_crps": {m: float(r.crps.mean()) for m, r in res.items()},
                "coverage90": {m: float(r.hit90.mean()) for m, r in res.items()},
                "coverage50": {m: float(r.hit50.mean()) for m, r in res.items()},
                "skill": {ds.benchmarks[b]: v for b, v in heldout_skill(ds, res).items()},
            }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, default=_plain) + "\n")
    print(f"rated {len(rows)} benchmarks -> {args.out}")


def cmd_run(args) -> None:
    ds = load_epoch_csv(args.data)
    out = analysis.run(ds, seed=args.seed, workers=args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, default=_plain) + "\n")
    print(f"ran the protocol on {ds.n_cells} cells -> {args.out}")
    for k, v in out["conditions"].items():
        print(f"  {k}: {'holds' if v['holds'] else 'FAILS'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pollbench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="write a synthetic long-format CSV (known truth)")
    s.add_argument("--models", type=int, default=120)
    s.add_argument("--benchmarks", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_simulate)
    r = sub.add_parser("rate", help="fit, rate and score a long-format CSV per docs/protocol.md")
    r.add_argument("data")
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--samples", type=int, default=600)
    r.add_argument("--warmup", type=int, default=300)
    r.add_argument("--no-heldout", action="store_true")
    r.set_defaults(fn=cmd_rate)
    p = sub.add_parser("run", help="the full protocol analysis: ratings, held-out comparison, conditions, robustness")
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
