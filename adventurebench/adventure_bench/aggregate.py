"""Create deterministic, offline comparison artifacts from audited runs."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .collect import validate_run_id
from .rescore import EvidenceError, rescore_run


RELEASE_SCHEMA_VERSION = "1.0"
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_ALGORITHM = (
    "xorshift32 rejection-sampled cluster-by-case bootstrap with replacement; "
    "10000 replicates; nearest-rank two-sided 95% percentile intervals"
)


class ReleaseError(ValueError):
    """A proposed release set is not an auditable like-for-like comparison."""


def _inside(root: Path, child: Path) -> Path:
    resolved_root = root.resolve()
    resolved_child = child.resolve()
    if resolved_child != resolved_root and resolved_root not in resolved_child.parents:
        raise ReleaseError("refusing a release path outside the configured releases directory")
    return resolved_child


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def _score(passed: int, total: int) -> float | None:
    return round(passed / total, 12) if total else None


def _timestamp_key(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _benchmark_identity(result: dict[str, Any]) -> dict[str, Any]:
    benchmark = result["benchmark"]
    return {key: benchmark.get(key) for key in ("name", "version", "dataset_sha256", "prompt_sha256")}


def _coverage(result: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = result.get("case_results")
    if not isinstance(rows, list) or not rows:
        raise ReleaseError(f"run {result.get('run_id')!r} has no replayed case results")
    coverage: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ReleaseError(f"run {result.get('run_id')!r} has malformed case results")
        case_id, repetition = row.get("case_id"), row.get("repetition")
        tags, outcome, passed = row.get("tags"), row.get("outcome"), row.get("passed")
        if (not isinstance(case_id, str) or isinstance(repetition, bool) or not isinstance(repetition, int)
                or repetition < 1 or not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags)
                or not isinstance(outcome, dict) or set(outcome) != {"action", "target"}
                or not isinstance(outcome["action"], str)
                or outcome["target"] is not None and not isinstance(outcome["target"], str)
                or not isinstance(passed, bool)):
            raise ReleaseError(f"run {result.get('run_id')!r} has malformed case result data")
        key = case_id, repetition
        if key in coverage:
            raise ReleaseError(f"run {result.get('run_id')!r} duplicates coverage {case_id}/{repetition}")
        coverage[key] = row
    if len(coverage) != result["overall"]["total"]:
        raise ReleaseError(f"run {result.get('run_id')!r} case result coverage disagrees with its total")
    return coverage


def _next_xorshift32(state: int) -> int:
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def _sample_index(state: int, population: int) -> tuple[int, int]:
    """Return one unbiased index from a documented fixed PRNG."""
    limit = (1 << 32) - ((1 << 32) % population)
    while True:
        state = _next_xorshift32(state)
        if state < limit:
            return state, state % population


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    # Nearest-rank percentiles: ceil(n * q), expressed as a zero-based index.
    return [ordered[math.ceil(len(ordered) * 0.025) - 1], ordered[math.ceil(len(ordered) * 0.975) - 1]]


def _bootstrap(model_rows: list[tuple[str, dict[tuple[str, int], dict[str, Any]]]]) -> dict[str, Any]:
    case_ids = sorted({case_id for case_id, _ in model_rows[0][1]})
    rows_by_case: dict[str, dict[str, list[bool]]] = {}
    for slug, coverage in model_rows:
        values: dict[str, list[bool]] = defaultdict(list)
        for (case_id, repetition), row in sorted(coverage.items()):
            values[case_id].append(row["passed"])
        rows_by_case[slug] = dict(values)
    model_scores = {slug: [] for slug, _ in model_rows}
    pairs = [(left[0], right[0]) for index, left in enumerate(model_rows) for right in model_rows[index + 1:]]
    differences = {pair: [] for pair in pairs}
    state = BOOTSTRAP_SEED
    for _ in range(BOOTSTRAP_REPLICATES):
        counts = {slug: [0, 0] for slug, _ in model_rows}
        for _ in case_ids:
            state, index = _sample_index(state, len(case_ids))
            case_id = case_ids[index]
            for slug, _ in model_rows:
                outcomes = rows_by_case[slug][case_id]
                counts[slug][0] += sum(outcomes)
                counts[slug][1] += len(outcomes)
        scores = {slug: _score(*counts[slug]) for slug, _ in model_rows}
        for slug, _ in model_rows:
            model_scores[slug].append(scores[slug])
        for left, right in pairs:
            differences[left, right].append(round(scores[left] - scores[right], 12))
    estimates = {
        slug: _score(sum(row["passed"] for row in coverage.values()), len(coverage))
        for slug, coverage in model_rows
    }
    return {
        "confidence_level": 0.95,
        "cluster_unit": "case_id",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "algorithm": BOOTSTRAP_ALGORITHM,
        "score_intervals": [
            {"model_slug": slug, "estimate": estimates[slug], "interval": _interval(model_scores[slug])}
            for slug, _ in model_rows
        ],
        "paired_score_difference_intervals": [
            {
                "model_a": left, "model_b": right,
                "estimate": round(estimates[left] - estimates[right], 12),
                "interval": _interval(differences[left, right]),
            }
            for left, right in pairs
        ],
    }


def aggregate_release(*, runs_dir: Path, results_dir: Path, releases_dir: Path,
                      release_id: str, run_ids: Iterable[str], dataset_path: Path,
                      check: bool = False) -> dict[str, Any]:
    """Revalidate completed one-model runs and write or check a release set."""
    validate_run_id(release_id)
    requested_runs = list(run_ids)
    if len(requested_runs) < 2:
        raise ReleaseError("a release set requires at least two completed run IDs")
    if len(set(requested_runs)) != len(requested_runs):
        raise ReleaseError("release run IDs must be unique")
    derived: list[dict[str, Any]] = []
    for run_id in requested_runs:
        try:
            validate_run_id(run_id)
            replay = rescore_run(runs_dir=runs_dir, results_dir=results_dir, run_id=run_id,
                                 dataset_path=dataset_path, check=True)
        except (EvidenceError, ValueError) as err:
            raise ReleaseError(f"run {run_id!r} cannot be revalidated: {err}") from err
        if not replay["summary"]["validity"]["valid"]:
            raise ReleaseError(f"run {run_id!r} is invalid: {', '.join(replay['summary']['validity']['invalid_reasons'])}")
        if len(replay["models"]) != 1:
            raise ReleaseError(f"run {run_id!r} does not contain exactly one model")
        derived.append(next(iter(replay["models"].values())))
    baseline = _benchmark_identity(derived[0])
    if any(_benchmark_identity(result) != baseline for result in derived[1:]):
        raise ReleaseError("runs do not share the same benchmark, dataset hash, and prompt hash")
    slugs = [result["model_slug"] for result in derived]
    if len(set(slugs)) != len(slugs):
        raise ReleaseError("release model slugs must be unique")
    coverage = [_coverage(result) for result in derived]
    expected_keys = set(coverage[0])
    if any(set(item) != expected_keys for item in coverage[1:]):
        raise ReleaseError("runs do not have identical case/repetition coverage")
    for key in expected_keys:
        expected_tags = coverage[0][key]["tags"]
        if any(item[key]["tags"] != expected_tags for item in coverage[1:]):
            raise ReleaseError("runs disagree on replayed dataset tags")
    model_rows = list(zip(slugs, coverage))
    tags = sorted({tag for row in coverage[0].values() for tag in row["tags"]})
    tag_matrix = {
        tag: {
            result["model_slug"]: result["by_tag"].get(tag, {"passed": 0, "total": 0, "score": None})
            for result in derived
        }
        for tag in tags
    }
    artifact = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "kind": "adventure-bench-release-set",
        "release_id": release_id,
        "run_ids": requested_runs,
        "benchmark": baseline,
        "collection_window": {
            "started_at": min(
                (result["provenance"]["collection_window"]["started_at"] for result in derived),
                key=_timestamp_key,
            ),
            "completed_at": max(
                (result["provenance"]["collection_window"]["completed_at"] for result in derived),
                key=_timestamp_key,
            ),
        },
        "coverage": {
            "case_repetitions": len(expected_keys),
            "case_count": len({case_id for case_id, _ in expected_keys}),
            "repetitions": sorted({repetition for _, repetition in expected_keys}),
        },
        "models": [
            {
                "run_id": result["run_id"], "model_slug": result["model_slug"],
                "overall": result["overall"], "failures": result["failures"],
                "provenance": result["provenance"], "evidence": result["evidence"],
            }
            for result in derived
        ],
        "tag_matrix": tag_matrix,
        "bootstrap": _bootstrap(model_rows),
    }
    output_dir = _inside(releases_dir, releases_dir / release_id)
    output_path = output_dir / "release.json"
    if output_dir.exists():
        unexpected = {path.name for path in output_dir.glob("*.json")} - {"release.json"}
        if unexpected:
            raise ReleaseError(f"release contains unexpected file(s): {', '.join(sorted(unexpected))}")
    expected = _canonical_json(artifact)
    if check:
        if not output_path.exists() or output_path.read_text(encoding="utf-8") != expected:
            raise ReleaseError(f"release artifact drift: {output_path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(expected, encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create an offline, deterministic Adventure Bench release set.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--run-id", action="append", required=True,
                        help="completed one-model run ID; repeat in the intended comparison order")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--releases-dir", default="releases")
    parser.add_argument("--data", default="adventure_bench/data/cases.jsonl")
    parser.add_argument("--check", action="store_true", help="fail if the canonical release artifact drifts")
    args = parser.parse_args(argv)
    try:
        artifact = aggregate_release(runs_dir=Path(args.runs_dir), results_dir=Path(args.results_dir),
                                     releases_dir=Path(args.releases_dir), release_id=args.release_id,
                                     run_ids=args.run_id, dataset_path=Path(args.data), check=args.check)
    except (ReleaseError, ValueError) as err:
        parser.error(str(err))
    print(json.dumps({"release_id": artifact["release_id"], "models": [row["model_slug"] for row in artifact["models"]], "check": args.check}, sort_keys=True))


if __name__ == "__main__":
    main()
