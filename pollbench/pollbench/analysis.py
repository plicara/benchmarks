"""The protocol's full analysis on one table: ratings, held-out comparison, conditions, robustness.

``run`` is what ``python -m pollbench run`` calls. Every fit is an independent
job, so the work is spread over processes; results are deterministic given the
seed. The output is one JSON document that the page renderer and the report read.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
from scipy.stats import spearmanr

from .checks import residual_pairs, second_factor_check
from .data import Dataset, is_developer_reported
from .evaluate import Result, evaluate, frontier_split, kfold_cells
from .models import HierBeta, default_methods
from .ratings import heldout_skill, rate_benchmarks
from .scoring import crossed_bootstrap, paired_bootstrap

METHOD_NAMES = ["benchmark_mean", "mean_rank", "low_rank_1", "eci_ls", "hier_beta"]
FINAL_SAMPLER = dict(n_chains=4, n_warmup=500, n_samples=1000)


def _final_fit(ds: Dataset, seed: int, **kw):
    fit = HierBeta(seed=seed, **FINAL_SAMPLER, **kw).fit(ds)
    rows = rate_benchmarks(ds, fit, np.random.default_rng(seed + 1))
    diag = {k: v for k, v in fit.diagnostics.items() if k not in ("rhat", "ess")}
    return rows, diag, fit


def _rating_job(args):
    name, ds, seed, kw = args
    rows, diag, fit = _final_fit(ds, seed, **kw)
    out = {"benchmarks": [asdict(r) for r in rows], "diagnostics": diag,
           "n_models": ds.n_models, "n_benchmarks": ds.n_benchmarks, "n_cells": ds.n_cells}
    if name == "floor_model":
        out["second_factor_check"] = second_factor_check(ds, fit, np.random.default_rng(seed + 2))
        out["residual_pairs"] = residual_pairs(ds, fit, np.random.default_rng(seed + 3))
        fl = fit.floors(fit.draws(np.random.default_rng(seed + 4), 400))
        out["floors"] = {b: [float(fl[:, i].mean()), float(np.quantile(fl[:, i], 0.05)), float(np.quantile(fl[:, i], 0.95))]
                         for i, b in enumerate(ds.benchmarks)}
    if name == "primary":
        out["second_factor_check"] = second_factor_check(ds, fit, np.random.default_rng(seed + 2))
        out["residual_pairs"] = residual_pairs(ds, fit, np.random.default_rng(seed + 3))
    return name, out


def _cv_job(args):
    split_name, ds, split, method_idx, seed = args
    method = default_methods(quick=True)[method_idx]
    res = evaluate(ds, [method], split, seed=seed)[0]
    return split_name, METHOD_NAMES[method_idx], res


def per_model_mean(ds: Dataset, res: Result) -> dict[str, float]:
    models = ds.model_idx[res.cell]
    return {ds.models[m]: float(res.crps[models == m].mean()) for m in np.unique(models)}


def compare(ds: Dataset, results: dict[str, Result], a: str, b: str, seed: int) -> dict:
    """Mean CRPS(a) - CRPS(b); negative means a is better.

    ``lo``/``hi`` come from the protocol's paired bootstrap over models (each
    model's mean difference is one observation). ``crossed_lo``/``crossed_hi``
    resample models and benchmarks together over the held-out cells, which also
    counts benchmark-to-benchmark variation and is the more cautious interval.
    """
    pa, pb = per_model_mean(ds, results[a]), per_model_mean(ds, results[b])
    common = sorted(set(pa) & set(pb))
    mean, lo, hi = paired_bootstrap({m: pa[m] - pb[m] for m in common}, n_boot=2000, seed=seed)
    ra, rb = results[a], results[b]
    shared, ia, ib = np.intersect1d(ra.cell, rb.cell, return_indices=True)
    cells_diff = ra.crps[ia] - rb.crps[ib]
    cmean, clo, chi = crossed_bootstrap(ds.model_idx[shared], ds.bench_idx[shared], cells_diff, n_boot=2000, seed=seed)
    return {"a": a, "b": b, "mean_diff": mean, "lo": lo, "hi": hi, "n_models": len(common),
            "cell_mean_diff": cmean, "crossed_lo": clo, "crossed_hi": chi, "n_cells": int(len(shared))}


def _letters(block: dict) -> dict[str, str]:
    return {r["benchmark"]: r["grade"] for r in block["benchmarks"]}


def _reliability(block: dict) -> dict[str, tuple[float, int]]:
    return {r["benchmark"]: (r["reliability_mean"], r["n_models"]) for r in block["benchmarks"]}


def stability_rho(full: dict, early: dict, min_models: int = 10) -> tuple[float | None, int]:
    f, e = _reliability(full), _reliability(early)
    names = [b for b in f if b in e and f[b][1] >= min_models and e[b][1] >= min_models
             and not np.isnan(f[b][0]) and not np.isnan(e[b][0])]
    if len(names) < 4:
        return None, len(names)
    return float(spearmanr([f[b][0] for b in names], [e[b][0] for b in names]).statistic), len(names)


def _cutoff(ds: Dataset, days: int = 365) -> str:
    newest = max(d for d in ds.model_date if d)
    return (date.fromisoformat(newest) - timedelta(days=days)).isoformat()


def run(ds: Dataset, *, seed: int = 0, workers: int = 4, quick: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    kf, fr = kfold_cells(ds, 5, seed), frontier_split(ds, 0.25, seed)
    cutoff = _cutoff(ds)
    early_ds = ds.restrict(lambda r: bool(r["model_date"]) and r["model_date"] <= cutoff)
    # Dropping developer-reported cells thins many benchmarks, so this arm uses looser minimums
    # and the comparison below is restricted to benchmarks graded in both fits.
    no_dev = ds.restrict(lambda r: not is_developer_reported(r["source"]),
                         min_models_per_benchmark=3, min_benchmarks_per_model=3)
    by_cells = np.bincount(ds.model_idx, minlength=ds.n_models)
    top3 = [ds.models[i] for i in np.argsort(-by_cells, kind="stable")[:3]]
    subsets = []
    for k in range(5):
        keep = set(rng.choice(ds.models, size=int(round(0.7 * ds.n_models)), replace=False).tolist())
        subsets.append((f"subset_{k}", ds.restrict(lambda r, keep=keep: r["model"] in keep), seed + 10 + k, {}))

    rating_jobs = [("primary", ds, seed, {}), ("stability_early", early_ds, seed, {}),
                   ("seed_repeat", ds, seed + 7, {}), ("floor_model", ds, seed, {"floor": True}), ("clip_0.02", ds.with_clip(0.02), seed, {}),
                   ("drop_developer_reported", no_dev, seed, {}),
                   ("precision_scale_0.5", ds, seed, {"sd_k": 0.5}), ("precision_scale_2", ds, seed, {"sd_k": 2.0})]
    rating_jobs += subsets
    rating_jobs += [(f"drop_model:{m}", ds.restrict(lambda r, m=m: r["model"] != m), seed, {}) for m in top3]
    cv_jobs = [(sn, ds, sp, mi, seed) for sn, sp in (("kfold", kf), ("frontier", fr)) for mi in range(len(METHOD_NAMES))]

    # heavy jobs first so the pool stays full
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as ex:
        cv_f = [ex.submit(_cv_job, j) for j in cv_jobs]
        rt_f = [ex.submit(_rating_job, j) for j in rating_jobs]
        cv = [f.result() for f in cv_f]
        ratings = dict(f.result() for f in rt_f)

    results = {"kfold": {}, "frontier": {}}
    for sn, name, res in cv:
        results[sn][name] = res
    primary = ratings["primary"]
    held = {}
    for sn in ("kfold", "frontier"):
        res = results[sn]
        held[sn] = {
            "mean_crps": {m: float(r.crps.mean()) for m, r in res.items()},
            "coverage90": {m: float(r.hit90.mean()) for m, r in res.items()},
            "coverage50": {m: float(r.hit50.mean()) for m, r in res.items()},
            "n_cells": int(len(res["hier_beta"].cell)),
            "hier_beta_vs": [compare(ds, res, "hier_beta", m, seed) for m in METHOD_NAMES if m != "hier_beta"],
            "skill": {ds.benchmarks[b]: list(v) for b, v in heldout_skill(ds, res).items()},
        }

    rel = {r["benchmark"]: r["reliability_mean"] for r in primary["benchmarks"]}
    sk = held["kfold"]["skill"]
    common = [b for b in sk if b in rel and not np.isnan(rel[b])]
    rho_skill = float(spearmanr([rel[b] for b in common], [sk[b][0] for b in common]).statistic) if len(common) > 3 else None
    rho_stab, n_stab = stability_rho(primary, ratings["stability_early"])
    c1 = next(c for c in held["kfold"]["hier_beta_vs"] if c["b"] == "mean_rank")
    c1b = next(c for c in held["kfold"]["hier_beta_vs"] if c["b"] == "eci_ls")
    cov = held["kfold"]["coverage90"]["hier_beta"]
    conditions = {
        "1_beats_mean_rank_split_A": {"kind": "predictive gain over the weakest serious baseline", "holds": bool(c1["hi"] < 0), "holds_crossed": bool(c1["crossed_hi"] < 0), **c1},
        "2_coverage90_in_0.80_0.95": {"kind": "calibration of the held-out intervals", "holds": bool(0.80 <= cov <= 0.95), "coverage90": cov},
        "3_temporal_stability_spearman_ge_0.70": {"kind": "consistency check: the early fit uses a subset of the full fit's data", "holds": bool(rho_stab is not None and rho_stab >= 0.70),
                                                  "rho": rho_stab, "n_benchmarks": n_stab, "cutoff_date": cutoff},
        "4_reliability_vs_skill_spearman_ge_0.50": {"kind": "consistency check: reliability and skill are both variance-explained quantities of the same fit", "holds": bool(rho_skill is not None and rho_skill >= 0.50),
                                                    "rho": rho_skill, "n_benchmarks": len(common)},
    }
    c3_and_c1_fail = not conditions["1_beats_mean_rank_split_A"]["holds"] and not conditions["3_temporal_stability_spearman_ge_0.70"]["holds"]
    base = _letters(primary)
    graded = lambda letters: {b for b, g in letters.items() if g != "ungraded"}

    def compare_letters(block):
        """Benchmarks graded in both fits whose letter differs, and how many could not be compared."""
        other = _letters(block)
        both = sorted(graded(base) & graded(other))
        return {"letters_changed": [b for b in both if other[b] != base[b]], "n_comparable": len(both),
                "n_not_assessable": len(graded(base)) - len(both)}

    def subset_changes():
        out = {}
        for b in graded(base):
            seen = [_letters(ratings[f"subset_{k}"]).get(b, "ungraded") for k in range(5)]
            comparable = [g for g in seen if g != "ungraded"]
            out[b] = {"changes": sum(g != base[b] for g in comparable), "comparable": len(comparable)}
        return out

    robustness = {
        "drop_developer_reported": {"n_flagged": int(sum(is_developer_reported(x) for x in ds.source)),
                                    "n_cells": ratings["drop_developer_reported"]["n_cells"],
                                    "n_models": ratings["drop_developer_reported"]["n_models"],
                                    **compare_letters(ratings["drop_developer_reported"])},
        "precision_scale_0.5": compare_letters(ratings["precision_scale_0.5"]),
        "precision_scale_2": compare_letters(ratings["precision_scale_2"]),
        "clip_0.02": compare_letters(ratings["clip_0.02"]),
        "seed_repeat": compare_letters(ratings["seed_repeat"]),
        "subsets_70pct": {"changes_per_benchmark": subset_changes()},
        "leave_one_model_out": {m: compare_letters(ratings[f"drop_model:{m}"]) for m in top3},
    }
    return {
        "protocol": "docs/protocol.md v1 + amendments 1-5", "seed": seed, "data_digest": ds.digest(),
        "file_sha256": ds.file_sha256,
        "score_sd": {ds.benchmarks[b]: float(ds.score[ds.bench_idx == b].std()) for b in range(ds.n_benchmarks)},
        "condition_1b_vs_eci_ls": c1b,
        "floor_model_check": {"second_factor_check": ratings["floor_model"]["second_factor_check"],
                              "residual_pairs": ratings["floor_model"]["residual_pairs"], "floors": ratings["floor_model"]["floors"],
                              "diagnostics": ratings["floor_model"]["diagnostics"]},
        "n_models": ds.n_models, "n_benchmarks": ds.n_benchmarks, "n_cells": ds.n_cells, "dropped": ds.dropped,
        "diagnostics": primary["diagnostics"], "benchmarks": primary["benchmarks"],
        "heldout": held, "conditions": conditions,
        "second_factor_check": primary["second_factor_check"], "residual_pairs": primary["residual_pairs"],
        "negative_result": bool(c3_and_c1_fail),
        "all_conditions_hold": bool(all(c["holds"] for c in conditions.values())),
        "robustness": robustness,
        "robustness_fit_diagnostics": {k: v["diagnostics"] for k, v in ratings.items()},
    }
