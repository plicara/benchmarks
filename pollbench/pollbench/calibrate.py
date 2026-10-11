"""Calibration of the second-factor check: type-I error and power on simulated tables.

``python -m pollbench.calibrate --reps 40 --out results/second-factor-calibration.json``
fits the model to tables drawn with no second factor (the null) and with
second-factor loadings of sd 0.3 and 0.8, runs ``second_factor_check`` on each
and records the rejection rate at 0.05. The check is a posterior predictive
check, which is known to be conservative, so the null rate is expected to sit
at or below 0.05 and the power is the number to read.
"""

from __future__ import annotations

import argparse
import multiprocessing
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .checks import second_factor_check
from .models import HierBeta
from .sim import simulate

LOADINGS = (0.0, 0.3, 0.8)


def _job(args):
    sec, seed = args
    ds, _ = simulate(n_models=200, n_benchmarks=50, seed=seed, coverage=0.12, second_factor=sec)
    fit = HierBeta(seed=seed, n_chains=2, n_warmup=200, n_samples=200).fit(ds)
    out = second_factor_check(ds, fit, np.random.default_rng(seed), n_rep=300)
    return {"second_factor_sd": sec, "seed": seed, "p_value": out["p_value"],
            "ratio": out["statistic"] / out["reference_mean"], "n_pairs": out["n_pairs"]}


def calibrate(reps: int = 40, workers: int = 4, alpha: float = 0.05, base_seed: int = 100) -> dict:
    jobs = [(s, base_seed + i) for s in LOADINGS for i in range(reps)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as ex:
        res = list(ex.map(_job, jobs))
    summary = {}
    for s in LOADINGS:
        r = [x for x in res if x["second_factor_sd"] == s]
        rej = sum(x["p_value"] < alpha for x in r)
        summary[str(s)] = {"reps": len(r), "rejections": rej, "rate": rej / len(r),
                           "median_ratio": float(np.median([x["ratio"] for x in r]))}
    return {"alpha": alpha, "design": "200 models x 50 benchmarks, 12% of cells observed, 300 posterior draws, NUTS 2 chains 200/200",
            "summary": summary, "runs": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/second-factor-calibration.json")
    a = ap.parse_args()
    out = calibrate(a.reps, a.workers)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out["summary"].items():
        print(f"second factor sd {k}: rejects {v['rejections']}/{v['reps']} at 0.05, median observed/reference {v['median_ratio']:.2f}")
