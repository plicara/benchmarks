"""Simulation study: do the ratings recover known truth, and how do they fail?

Each scenario draws tables from ``sim.simulate`` and checks, against the
generating parameters: (1) rank agreement of the estimated reliability with the
true reliability, (2) coverage of the 90% reliability interval, (3) held-out
CRPS of every method and 90% interval coverage, and (4) agreement between
model-based reliability and held-out skill. Scenarios include deliberate
misspecification so the failure modes are on the record.
"""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from .evaluate import evaluate, kfold_cells
from .models import HierBeta, default_methods
from .ratings import heldout_skill, rate_benchmarks, reliability_draws, true_reliability
from .sim import simulate

SCENARIOS = {
    "well_specified": dict(),
    "second_factor": dict(second_factor=0.8),
    "selective_reporting": dict(selective_reporting=0.6),
    "sparse": dict(coverage=0.09),
    "chance_floor": dict(floor_share=0.4),
}


@dataclass
class SeedResult:
    scenario: str
    seed: int
    n_cells: int
    rel_spearman: float
    rel_coverage90: float
    rel_bias: float
    skill_vs_reliability_spearman: float
    crps: dict[str, float]
    cover90: dict[str, float]
    cover50: dict[str, float]


def run_seed(scenario: str, seed: int) -> SeedResult:
    ds, truth = simulate(seed=seed, **SCENARIOS[scenario])
    rng = np.random.default_rng(seed + 1000)
    fit = HierBeta(seed=seed).fit(ds)
    X = fit.draws(rng, 600)
    th, d, a, k = fit.split(X)
    draws = reliability_draws(ds, th, d, a, k)
    true_r = true_reliability(ds, truth.theta, truth.difficulty, truth.slope, truth.precision, truth.floor)
    mean_r, lo, hi = np.nanmean(draws, 0), np.nanquantile(draws, 0.05, 0), np.nanquantile(draws, 0.95, 0)
    ok = ~np.isnan(true_r) & ~np.isnan(mean_r)
    rel_sp = float(spearmanr(mean_r[ok], true_r[ok]).statistic)
    cov = float(((true_r[ok] >= lo[ok]) & (true_r[ok] <= hi[ok])).mean())
    bias = float(np.mean(mean_r[ok] - true_r[ok]))

    methods = default_methods(quick=True)
    res = {r.method: r for r in evaluate(ds, methods, kfold_cells(ds, 5, seed), seed=seed)}
    skill = heldout_skill(ds, res)
    bs = sorted(skill)
    sp = float(spearmanr([skill[b][0] for b in bs], [mean_r[b] for b in bs]).statistic) if len(bs) > 3 else float("nan")
    return SeedResult(scenario, seed, ds.n_cells, rel_sp, cov, bias, sp,
                      {m: float(r.crps.mean()) for m, r in res.items()},
                      {m: float(r.hit90.mean()) for m, r in res.items()},
                      {m: float(r.hit50.mean()) for m, r in res.items()})


def summarise(results: list[SeedResult]) -> dict:
    out = {}
    for sc in sorted({r.scenario for r in results}):
        rs = [r for r in results if r.scenario == sc]
        methods = rs[0].crps.keys()
        mean = lambda xs: float(np.nanmean(xs))
        out[sc] = {
            "seeds": len(rs),
            "cells_mean": mean([r.n_cells for r in rs]),
            "reliability_rank_agreement": mean([r.rel_spearman for r in rs]),
            "reliability_interval_coverage90": mean([r.rel_coverage90 for r in rs]),
            "reliability_bias": mean([r.rel_bias for r in rs]),
            "skill_vs_reliability_rank_agreement": mean([r.skill_vs_reliability_spearman for r in rs]),
            "heldout_crps": {m: mean([r.crps[m] for r in rs]) for m in methods},
            "heldout_coverage90": {m: mean([r.cover90[m] for r in rs]) for m in methods},
            "heldout_coverage50": {m: mean([r.cover50[m] for r in rs]) for m in methods},
            "hier_beta_best_crps_in_seeds": sum(min(r.crps, key=r.crps.get) == "hier_beta" for r in rs),
        }
    return out


def _job(args):
    return run_seed(*args)


def run(seeds: list[int], scenarios: list[str] | None = None, workers: int = 4) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(sc, s) for sc in (scenarios or list(SCENARIOS)) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as ex:
        results = list(ex.map(_job, jobs))
    return {"seeds": seeds, "summary": summarise(results),
            "per_seed": [r.__dict__ for r in results]}


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/simulation-study.json")
    args = ap.parse_args(argv)
    from pathlib import Path
    out = run(list(range(args.seeds)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
