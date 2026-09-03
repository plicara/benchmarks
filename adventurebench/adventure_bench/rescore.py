"""Deterministically derive Adventure Bench v1 results from raw evidence.

This module never calls a provider and deliberately replays the frozen v1
parser and outcome normalisation.  Recorded outcomes are checked as claims,
not used as inputs to scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .collect import SCHEMA_VERSION, safe_model_slug, selected_cases_hash, sha256_file, sha256_text, validate_run_id
from .runner import DATA_PATH, SYSTEM_PROMPT, extract_json, load_cases, reply_to_outcome


DERIVED_SCHEMA_VERSION = "1.0"


class EvidenceError(ValueError):
    """Evidence is incomplete, inconsistent, or does not match the benchmark."""


def _fail(message: str) -> None:
    raise EvidenceError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise EvidenceError(f"cannot read {label}: {path}") from err
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def _score(passed: int, total: int) -> float | None:
    return round(passed / total, 12) if total else None


def _required(mapping: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        _fail(f"{label} missing required field(s): {', '.join(missing)}")


def _as_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{label} must be a UTC date-time string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as err:
        raise EvidenceError(f"{label} must be a UTC date-time string") from err
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(f"{label} must be a UTC date-time string")
    return parsed


def _outcome(value: Any, label: str) -> tuple[str, str | None] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"action", "target"}:
        _fail(f"{label} must contain exactly action and target")
    action, target = value["action"], value["target"]
    if not isinstance(action, str) or (target is not None and not isinstance(target, str)):
        _fail(f"{label} has invalid action or target")
    return action, target


def _replay(case: dict[str, Any], record: dict[str, Any], label: str) -> tuple[tuple[str, str | None], bool, int]:
    """Replay raw attempts and verify all collection-side score claims."""
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
        _fail(f"{label}.attempts must contain one or two attempts")
    final: tuple[str, str | None] = ("unclear", None)
    transport = False
    parsing_failures = 0
    terminal = False
    for index, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict):
            _fail(f"{label}.attempts[{index}] must be an object")
        _required(attempt, ("number", "raw_completion", "parsed", "error"), f"{label}.attempts[{index}]")
        if _as_int(attempt["number"], f"{label}.attempts[{index}].number", minimum=1) != index:
            _fail(f"{label} has non-sequential attempt numbers")
        raw, recorded_parsed, error = attempt["raw_completion"], attempt["parsed"], attempt["error"]
        if raw is None:
            if not isinstance(error, dict) or error.get("kind") not in {"transport", "client"}:
                _fail(f"{label}.attempts[{index}] has no raw completion without a transport/client error")
            if recorded_parsed is not None:
                _fail(f"{label}.attempts[{index}] parses a failed transport attempt")
            if index != len(attempts):
                _fail(f"{label} retries after a transport/client error")
            transport = True
            terminal = True
            continue
        if not isinstance(raw, str):
            _fail(f"{label}.attempts[{index}].raw_completion must be string or null")
        replayed = reply_to_outcome(case, extract_json(raw))
        parsed = _outcome(recorded_parsed, f"{label}.attempts[{index}].parsed")
        expected_parsed = {"action": replayed[0], "target": replayed[1]} if replayed else None
        if recorded_parsed != expected_parsed:
            _fail(f"{label}.attempts[{index}].parsed does not match its raw completion")
        if replayed is not None:
            if error is not None:
                _fail(f"{label}.attempts[{index}] records an error for a valid completion")
            if index != len(attempts):
                _fail(f"{label} continues after a valid completion")
            final, terminal = replayed, True
            continue
        parsing_failures += 1
        if not isinstance(error, dict) or error.get("kind") != "parsing":
            _fail(f"{label}.attempts[{index}] omits its parsing error")
        if index == len(attempts):
            terminal = True
    if not terminal:
        _fail(f"{label} has unterminated attempts")
    if transport:
        final = ("unclear", None)
    if record.get("outcome") != {"action": final[0], "target": final[1]}:
        _fail(f"{label}.outcome does not match replayed raw completions")
    return final, transport, parsing_failures


def _read_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
    except OSError as err:
        raise EvidenceError(f"cannot read response evidence: {path}") from err
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as err:
        raise EvidenceError(f"response evidence is not valid UTF-8: {path}") from err
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            _fail(f"response evidence contains blank line at {path}:{number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as err:
            raise EvidenceError(f"malformed JSONL evidence at {path}:{number}") from err
        if not isinstance(record, dict):
            _fail(f"response evidence at {path}:{number} must be an object")
        records.append(record)
    return records, _sha256_bytes(raw)


def _recorded_cost(records: list[dict[str, Any]], label: str) -> float | None:
    """Sum provider-recorded per-response cost when every record exposes it."""
    total = 0.0
    for line, record in enumerate(records, 1):
        value = record.get("cost")
        if value is None:
            return None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            _fail(f"{label}:{line}.cost is not a non-negative finite number")
        total += float(value)
    return round(total, 12)


def _validate_manifest(manifest: dict[str, Any], run_id: str, dataset_path: Path) -> None:
    _required(manifest, ("schema_version", "benchmark", "run_id", "started_at", "completed_at", "case_count", "selected_case_ids_sha256", "repetitions", "configuration", "models", "validity"), "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        _fail(f"unsupported evidence schema version: {manifest['schema_version']!r}")
    if manifest["run_id"] != run_id:
        _fail("manifest run_id does not match requested run")
    _as_int(manifest["case_count"], "manifest.case_count", minimum=1)
    _as_int(manifest["repetitions"], "manifest.repetitions", minimum=1)
    if not isinstance(manifest["benchmark"], dict):
        _fail("manifest.benchmark must be an object")
    benchmark = manifest["benchmark"]
    _required(benchmark, ("name", "version", "dataset_sha256", "prompt_sha256"), "manifest.benchmark")
    if benchmark["name"] != "Adventure Bench" or benchmark["version"] != __version__:
        _fail("manifest identifies a different Adventure Bench version")
    if benchmark["dataset_sha256"] != sha256_file(dataset_path):
        _fail("manifest dataset hash does not match the supplied dataset")
    if benchmark["prompt_sha256"] != sha256_text(SYSTEM_PROMPT):
        _fail("manifest prompt hash does not match frozen v1 prompt")
    if not isinstance(manifest["configuration"], dict) or not isinstance(manifest["models"], dict):
        _fail("manifest configuration and models must be objects")
    if not isinstance(manifest["completed_at"], str) or not manifest["completed_at"]:
        _fail("manifest is incomplete: completed_at is missing")
    started_at = _utc_timestamp(manifest["started_at"], "manifest.started_at")
    completed_at = _utc_timestamp(manifest["completed_at"], "manifest.completed_at")
    if completed_at < started_at:
        _fail("manifest.completed_at precedes manifest.started_at")
    if len(manifest["models"]) != 1:
        _fail("a collection run must contain exactly one model")
    selected_case_ids = manifest.get("selected_case_ids")
    if selected_case_ids is not None:
        if (not isinstance(selected_case_ids, list) or len(selected_case_ids) != manifest["case_count"]
                or not all(isinstance(case_id, str) and case_id for case_id in selected_case_ids)
                or len(set(selected_case_ids)) != len(selected_case_ids)):
            _fail("manifest.selected_case_ids must be a unique ordered case-ID list")


def rescore_run(*, runs_dir: Path, results_dir: Path, run_id: str, dataset_path: Path = DATA_PATH, check: bool = False) -> dict[str, Any]:
    """Validate one run and write (or check) its deterministic derived results."""
    validate_run_id(run_id)
    run_dir = runs_dir / run_id
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path, "manifest")
    _validate_manifest(manifest, run_id, dataset_path)
    cases = load_cases(dataset_path)
    case_by_id = {case["id"]: case for case in cases}
    if len(case_by_id) != len(cases):
        _fail("dataset has duplicate case IDs")
    config = manifest["configuration"]
    models = manifest["models"]
    if not models:
        _fail("manifest has no models")
    results: dict[str, dict[str, Any]] = {}
    manifest_sha = sha256_file(manifest_path)
    expected_files = {"summary.json", "tag-matrix.json"}

    for slug in sorted(models):
        model_info = models[slug]
        if not isinstance(model_info, dict):
            _fail(f"manifest.models.{slug} must be an object")
        _required(model_info, ("requested", "responses_file", "records", "expected_records", "resolved_providers", "transport_failures", "valid", "invalid_reasons"), f"manifest.models.{slug}")
        requested = model_info["requested"]
        if not isinstance(requested, dict) or requested != config.get("requested"):
            _fail(f"manifest.models.{slug}.requested does not match run configuration")
        if safe_model_slug(str(requested.get("model", ""))) != slug:
            _fail(f"manifest model slug does not match requested model")
        evidence_name = model_info["responses_file"]
        if evidence_name != f"responses/{slug}.jsonl":
            _fail(f"manifest model evidence path is not canonical for {slug}")
        evidence_path = run_dir / evidence_name
        if evidence_path.parent.resolve() != (run_dir / "responses").resolve():
            _fail("manifest response path escapes responses directory")
        records, evidence_sha = _read_records(evidence_path)
        recorded_cost_usd = _recorded_cost(records, evidence_name)
        expected_files.add(f"{slug}.json")
        expected_count = manifest["case_count"] * manifest["repetitions"]
        if model_info["records"] != len(records) or model_info["expected_records"] != expected_count:
            _fail(f"manifest record counts do not match evidence for {slug}")
        seen: set[tuple[str, int]] = set()
        observed_ids: set[str] = set()
        resolved_models, resolved_providers, resolved_runtimes = set(), set(), set()
        overall = [0, 0]
        by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        by_repetition: dict[int, dict[str, Any]] = {}
        case_results: list[dict[str, Any]] = []
        transport_failures = parsing_failures = 0
        for line, record in enumerate(records, 1):
            label = f"{evidence_name}:{line}"
            _required(record, ("schema_version", "case_id", "repetition", "request", "attempts", "outcome", "expected_outcomes", "passed", "transport_error", "error", "resolved"), label)
            if record["schema_version"] != SCHEMA_VERSION:
                _fail(f"{label} has unsupported evidence schema version")
            case_id = record["case_id"]
            if not isinstance(case_id, str) or case_id not in case_by_id:
                _fail(f"{label} has an unknown case_id")
            repetition = _as_int(record["repetition"], f"{label}.repetition", minimum=1)
            if repetition > manifest["repetitions"]:
                _fail(f"{label} has repetition outside manifest range")
            key = case_id, repetition
            if key in seen:
                _fail(f"duplicate case/repetition evidence: {case_id}/{repetition}")
            seen.add(key)
            observed_ids.add(case_id)
            if record["request"] != config:
                _fail(f"{label}.request does not match manifest configuration")
            if record["request"].get("prompt_sha256") != sha256_text(SYSTEM_PROMPT):
                _fail(f"{label}.request has wrong prompt hash")
            expected = case_by_id[case_id]["expect"]
            if record["expected_outcomes"] != expected:
                _fail(f"{label}.expected_outcomes does not match dataset")
            outcome, transport, parsed_count = _replay(case_by_id[case_id], record, label)
            if record["transport_error"] is not transport:
                _fail(f"{label}.transport_error does not match replayed attempts")
            claimed_pass = not transport and [outcome[0], outcome[1]] in expected
            if record["passed"] is not claimed_pass:
                _fail(f"{label}.passed does not match replayed outcome")
            if transport:
                if not isinstance(record["error"], dict) or record["error"].get("kind") not in {"transport", "client"}:
                    _fail(f"{label}.error does not describe its transport failure")
                transport_failures += 1
            elif record["error"] is not None and record["error"].get("kind") != "parsing":
                _fail(f"{label}.error is inconsistent with a successful transport")
            parsing_failures += parsed_count
            resolved = record["resolved"]
            if not isinstance(resolved, dict) or set(resolved) != {"model", "provider", "runtime"}:
                _fail(f"{label}.resolved has invalid identity fields")
            for field, bucket in (("model", resolved_models), ("provider", resolved_providers)):
                value = resolved[field]
                if not isinstance(value, str) or not value:
                    _fail(f"{label}.resolved.{field} is missing")
                bucket.add(value)
            runtime = resolved["runtime"]
            if runtime is not None and (not isinstance(runtime, str) or not runtime):
                _fail(f"{label}.resolved.runtime has an invalid value")
            resolved_runtimes.add(runtime)
            passed = int(claimed_pass)
            overall[0] += passed
            overall[1] += 1
            # This is intentionally only replay-derived data.  Raw completions
            # remain in the evidence JSONL rather than being duplicated into a
            # publishable comparison artifact.
            case_results.append({
                "case_id": case_id,
                "repetition": repetition,
                "tags": sorted(case_by_id[case_id]["tags"]),
                "outcome": {"action": outcome[0], "target": outcome[1]},
                "passed": bool(claimed_pass),
            })
            rep = by_repetition.setdefault(repetition, {"passed": 0, "total": 0, "transport_failures": 0, "parsing_failures": 0, "by_tag": defaultdict(lambda: [0, 0])})
            rep["passed"] += passed
            rep["total"] += 1
            rep["transport_failures"] += int(transport)
            rep["parsing_failures"] += parsed_count
            for tag in case_by_id[case_id]["tags"]:
                by_tag[tag][0] += passed
                by_tag[tag][1] += 1
                rep["by_tag"][tag][0] += passed
                rep["by_tag"][tag][1] += 1
        if len(records) != expected_count or len(observed_ids) != manifest["case_count"]:
            _fail(f"evidence coverage does not match manifest for {slug}")
        selected_ids = manifest.get("selected_case_ids")
        if selected_ids is None:
            # Schema v1 evidence predating explicit selection IDs can still be
            # replayed when its first completed repetition preserves selection
            # order. New manifests make that order explicit.
            selected_ids = [record["case_id"] for record in records if record["repetition"] == 1]
        if set(selected_ids) != observed_ids or len(selected_ids) != len(observed_ids):
            _fail(f"selected case IDs do not match evidence coverage for {slug}")
        ordered_selected = [case_by_id[case_id] for case_id in selected_ids]
        if selected_cases_hash(ordered_selected) != manifest["selected_case_ids_sha256"]:
            _fail(f"selected-case hash does not match evidence and dataset for {slug}")
        if any((case["id"], repetition) not in seen for case in ordered_selected for repetition in range(1, manifest["repetitions"] + 1)):
            _fail(f"evidence is missing case/repetition coverage for {slug}")
        reasons: list[str] = []
        if len(resolved_providers) != 1:
            reasons.append("provider resolution is missing or inconsistent")
        if len(resolved_models) != 1 or len(resolved_runtimes) != 1:
            _fail(f"model or runtime resolution is inconsistent for {slug}")
        if transport_failures:
            reasons.append(f"{transport_failures} transport failure(s)")
        if model_info["resolved_providers"] != sorted(resolved_providers):
            _fail(f"manifest provider provenance does not match evidence for {slug}")
        if model_info["transport_failures"] != transport_failures:
            _fail(f"manifest transport failure count does not match evidence for {slug}")
        if bool(model_info["valid"]) != (not reasons) or model_info["invalid_reasons"] != reasons:
            _fail(f"manifest validity claim does not match evidence for {slug}")
        repetitions = []
        for number in sorted(by_repetition):
            rep = by_repetition[number]
            repetitions.append({
                "repetition": number, "passed": rep["passed"], "total": rep["total"], "score": _score(rep["passed"], rep["total"]),
                "transport_failures": rep["transport_failures"], "parsing_failures": rep["parsing_failures"],
                "by_tag": {tag: {"passed": counts[0], "total": counts[1], "score": _score(*counts)} for tag, counts in sorted(rep["by_tag"].items())},
            })
        result = {
            "schema_version": DERIVED_SCHEMA_VERSION,
            "kind": "adventure-bench-derived-result",
            "run_id": run_id,
            "model_slug": slug,
            "benchmark": {key: manifest["benchmark"].get(key) for key in ("name", "version", "source_commit", "repository_commit", "dataset_sha256", "prompt_sha256")},
            "evidence": {
                "manifest_sha256": manifest_sha,
                "responses_sha256": evidence_sha,
                "records": len(records),
                "recorded_cost_usd": recorded_cost_usd,
            },
            "collection_configuration": config,
            "provenance": {
                "requested": requested,
                "resolved": {"model": sorted(resolved_models)[0], "provider": sorted(resolved_providers)[0], "runtime": next(iter(resolved_runtimes))},
                "collection_window": {"started_at": manifest["started_at"], "completed_at": manifest["completed_at"]},
            },
            "validity": {"valid": not reasons, "invalid_reasons": reasons},
            "overall": {"passed": overall[0], "total": overall[1], "score": _score(*overall)},
            "repetitions": repetitions,
            "by_tag": {tag: {"passed": counts[0], "total": counts[1], "score": _score(*counts)} for tag, counts in sorted(by_tag.items())},
            "case_results": sorted(case_results, key=lambda item: (item["case_id"], item["repetition"])),
            "failures": {"transport": transport_failures, "parsing": parsing_failures},
        }
        results[slug] = result

    run_reasons = sorted({reason for result in results.values() for reason in result["validity"]["invalid_reasons"]})
    if manifest["validity"] != {"valid": not run_reasons, "invalid_reasons": run_reasons}:
        _fail("manifest run validity does not match model evidence")
    summary = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "kind": "adventure-bench-derived-summary",
        "run_id": run_id,
        "benchmark": {key: manifest["benchmark"].get(key) for key in ("name", "version", "source_commit", "repository_commit", "dataset_sha256", "prompt_sha256")},
        "evidence": {"manifest_sha256": manifest_sha},
        "collection_configuration": config,
        "validity": {"valid": not run_reasons, "invalid_reasons": run_reasons},
        "models": [{"model_slug": slug, "validity": result["validity"], "overall": result["overall"], "failures": result["failures"], "evidence": result["evidence"], "provenance": result["provenance"]} for slug, result in sorted(results.items())],
    }
    tags = sorted({tag for result in results.values() for tag in result["by_tag"]})
    matrix = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "kind": "adventure-bench-derived-tag-matrix",
        "run_id": run_id,
        "benchmark": {"version": manifest["benchmark"]["version"], "dataset_sha256": manifest["benchmark"]["dataset_sha256"], "prompt_sha256": manifest["benchmark"]["prompt_sha256"]},
        "evidence": {"manifest_sha256": manifest_sha},
        "tags": {tag: {slug: results[slug]["by_tag"].get(tag, {"passed": 0, "total": 0, "score": None}) for slug in sorted(results)} for tag in tags},
    }
    output_dir = results_dir / run_id
    files = {output_dir / f"{slug}.json": result for slug, result in results.items()}
    files[output_dir / "summary.json"] = summary
    files[output_dir / "tag-matrix.json"] = matrix
    if output_dir.exists():
        actual_files = {path.name for path in output_dir.glob("*.json")}
        unexpected = actual_files - expected_files
        if unexpected:
            _fail(f"derived results contain unexpected file(s): {', '.join(sorted(unexpected))}")
    mismatches = [path for path, value in files.items() if not path.exists() or path.read_text(encoding="utf-8") != _canonical_json(value)]
    if check:
        if mismatches:
            _fail("derived results drift: " + ", ".join(str(path) for path in mismatches))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, value in files.items():
            path.write_text(_canonical_json(value), encoding="utf-8")
    return {"models": results, "summary": summary, "tag_matrix": matrix}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline-rescore audited Adventure Bench evidence.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--check", action="store_true", help="fail if committed derived files differ from replay")
    args = parser.parse_args(argv)
    try:
        value = rescore_run(runs_dir=Path(args.runs_dir), results_dir=Path(args.results_dir), run_id=args.run_id, dataset_path=Path(args.data), check=args.check)
    except EvidenceError as err:
        parser.error(str(err))
    print(json.dumps({"run_id": args.run_id, "models": sorted(value["models"]), "check": args.check}, sort_keys=True))


if __name__ == "__main__":
    main()
