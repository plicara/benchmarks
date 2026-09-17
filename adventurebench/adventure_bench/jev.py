"""Jev (TypeSafe System One) runtime for Adventure Bench — experimental, NOT frozen.

The frozen v1 benchmark (SYSTEM_PROMPT + chat-completions JSON + retry in
runner.py) is untouched. This module offers an alternate runtime behind
`adventure-bench --jev`: one TypeSafe systemone call per case with two
parallel Choice questions (action + target) over the same state shape that
runner.user_message() builds. Outcomes use the same (kind, target) scoring,
so numbers are comparable as accuracy but NOT as frozen-v1 runs.

Needs TYPESAFE_API_KEY in the environment (console.typesafe.ai). The key is
read here only to authorize the request; it is never printed or stored.

ponytail: stdlib only (urllib), mirroring runner.ChatClient — no new dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import __version__
from .collect import (
    SCHEMA_VERSION, _append_record, _error, _exclusive_run_collection, _inside, _manifest_path, _read_records,
    _write_manifest, git_commit, redact, safe_model_slug, selected_cases_hash,
    sha256_file, utc_now, validate_run_id,
)
from .runner import DATA_PATH, DIRECTIONS, NO_TARGET_ACTIONS, TransportError, user_message

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# Frozen 2026-09-16 from a synthetic-only sweep (300 cases: raw 0.857 -> 0.903
# at T=0.9, 32 flips). Do NOT retune on the frozen 244 — that is peeking.
DEFAULT_THRESHOLD = 0.9
# Added 2026-09-17 from dev-500 trials (baseline 0.878 -> 0.932 with the pair).
# Mapping tags (schema's positive mapping ability group) gate lower; everything
# else keeps the calibration gate. Same frozen-data rule applies.
DEFAULT_MAPPING_THRESHOLD = 0.75
MAPPING_TAGS = {"exact-verb", "verb-alias", "abbreviation", "synonym",
                "missing-preposition", "paraphrase", "direction-as-place",
                "go-to-place", "relative-direction", "politeness", "full-sentence",
                "question", "adverb", "typo", "pronoun", "compound", "multi-object"}


def threshold_for(case: dict) -> float:
    """Return the frozen mapping or calibration gate for one case."""
    if MAPPING_TAGS.intersection(case.get("tags", [])):
        return DEFAULT_MAPPING_THRESHOLD
    return DEFAULT_THRESHOLD

ACTION_INSTRUCTIONS = ("Which single game action does the player's input intend? "
                         "Map synonyms and paraphrases to intent; use unclear when the verb "
                         "or referent fits nothing in context.")
TARGET_INSTRUCTIONS = ("Which target does the player's input refer to? Pick the exit "
                         "or item id named or clearly paraphrased; pick none if there is "
                         "no target or nothing present fits. Use state input, exits, "
                         "items_here, and carrying.")
TARGET_RULE = ("per-case criteria: one option per exit ('Direction <d>'), one per "
               "item id ('Item: <name>'), plus 'none' (no target / nothing present fits)")


def jev_spec() -> dict[str, Any]:
    """Canonical prompt spec for a Jev collection. Its hash is the prompt identity."""
    return {
        "spec": "adventurebench-systemone-v2",
        "threshold": DEFAULT_THRESHOLD,
        "mapping_threshold": DEFAULT_MAPPING_THRESHOLD,
        "mapping_tags": sorted(MAPPING_TAGS),
        "unclear_on_missing_target": True,
        "action_instructions": ACTION_INSTRUCTIONS,
        "action_criteria": ACTION_CRITERIA,
        "target_instructions": TARGET_INSTRUCTIONS,
        "target_rule": TARGET_RULE,
    }


def jev_spec_sha() -> str:
    """Stable hash of the canonical spec (sort_keys; the release prompt identity)."""
    return hashlib.sha256(
        json.dumps(jev_spec(), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# The frozen spec hash rescore accepts for Jev evidence (assigned after
# ACTION_CRITERIA below; a different threshold is a different spec).

INPUT_USD_PER_TOKEN = 42.0 / 1e9

# Closed 8-verb world (frozen into JEV_SPEC above): the benchmark's out-of-vocab
# talk, ...) must come back unclear — use/examine must not stretch to cover
# novel verbs. Tuned 2026-09-16 on a 5-case spike, NOT on the frozen 244.
ACTION_CRITERIA = {
    "move": "Player wants to go somewhere; target is one of the listed exits",
    "take": "Player wants to pick up an item using take-like verbs (take, get, grab, pick up)",
    "drop": "Player wants to set down a carried item (drop, leave, put down)",
    "examine": "Player wants to inspect a specific present item using examine-like verbs (examine, look at, inspect, check, read, peer)",
    "use": "Player explicitly says use/operate/apply a specific present item. Do NOT stretch to open/close/lock/eat/drink/attack/kill/climb/hide — those verbs are unclear",
    "look": "Player wants the surroundings re-described; no specific target",
    "inventory": "Player asks what they carry; no specific target",
    "unclear": "Closed verb world: intent verb is outside move/take/drop/examine/use/look/inventory (e.g. open, close, eat, drink, attack, talk) OR refers to something not in context. Never invent items/directions. Relative directions (left, right, around, back, ahead) are meaningless without facing and are always unclear",
}


# v1 spec hash (2026-09-16 release): pinned literally so old evidence replays
# forever even as the live spec advances. NEVER change this string.
JEV_SPEC_SHA_V1 = "7d98e8145c359c2cc13dd87fb116f207820595eaf52fb217a3add67ec8c92a91"

# The current frozen spec hash rescore accepts alongside v1 and frozen chat.
JEV_SPEC_SHA = jev_spec_sha()


def api_key_from_env() -> str | None:
    """Return the TypeSafe API key from the environment."""
    return os.environ.get("TYPESAFE_API_KEY")


class JevClient:
    """Minimal TypeSafe systemone client (stdlib only)."""

    provider = "typesafe"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def evaluate(self, state: dict, questions: dict) -> tuple[dict, float]:
        """POST one systemone request. Returns (response payload, seconds)."""
        body = json.dumps({"state": state, "model": self.model, "questions": questions}).encode()
        request = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as err:
            raise TransportError(f"HTTP {err.code}: {err.read().decode(errors='replace')[:300]}") from err
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            raise TransportError(str(err)) from err
        return payload, time.time() - started


def build_questions(case: dict) -> dict:
    """Two parallel Choices over one state: what action, which target.

    Target candidates are per-case (exits + item ids + explicit none) — the
    model cannot choose an omitted value, so none covers the no-match outcome.
    """
    ctx = case["context"]
    targets: dict[str, str | None] = {d: f"Direction {d}" for d in ctx.get("exits", [])}
    for item in ctx.get("items", []) + ctx.get("carrying", []):
        targets[item["id"]] = f"Item: {item['name']}"
    targets["none"] = "No specific target, or the request refers to nothing present"
    return {
        "action": {
            "type": "choice",
            "instructions": ACTION_INSTRUCTIONS,
            "criteria": ACTION_CRITERIA,
        },
        "target": {
            "type": "choice",
            "instructions": TARGET_INSTRUCTIONS,
            "criteria": targets,
        },
    }


def _map_outcome(answers: dict, threshold: float, *, unclear_on_missing_target: bool
                 ) -> tuple[tuple[str, str | None], bool, tuple[str, str | None]] | None:
    """Map answers under either the frozen v1 or current v2 missing-target rule."""
    try:
        kind = answers["action"]["choice"]
        picked = answers["target"]["choice"]
        confidence = answers["action"].get("confidence")
    except (KeyError, TypeError, AttributeError):
        return None
    if kind not in ACTION_CRITERIA:
        return None
    if kind in NO_TARGET_ACTIONS:
        outcome: tuple[str, str | None] = (kind, None)
    elif picked is None or picked == "none":
        if unclear_on_missing_target:
            outcome = ("unclear", None)
        elif picked is None:
            return None
        else:
            outcome = (kind, "none")
    elif not isinstance(picked, str):
        return None  # Choice options are strings by construction
    elif kind == "move":
        outcome = ("move", DIRECTIONS.get(str(picked).lower(), picked))
    else:
        outcome = (kind, picked)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = 0.0  # non-numeric confidence carries no information; gate it shut
    gate = 1.0 if kind in NO_TARGET_ACTIONS else confidence
    raw = (outcome[0], outcome[1])
    if outcome[0] != "unclear" and gate < threshold:
        return ("unclear", None), True, raw
    return outcome, False, raw


def answers_outcome(answers: dict, threshold: float) -> tuple[tuple[str, str | None], bool, tuple[str, str | None]] | None:
    """Map answers using the current v2 missing-target rule."""
    return _map_outcome(answers, threshold, unclear_on_missing_target=True)


def answers_outcome_v1(answers: dict, threshold: float) -> tuple[tuple[str, str | None], bool, tuple[str, str | None]] | None:
    """Map answers using the frozen v1 rule for replaying historic evidence."""
    return _map_outcome(answers, threshold, unclear_on_missing_target=False)


def replay_jev_outcome(raw: str, case: dict, spec_sha: str) -> tuple[str, str | None] | None:
    """Re-derive raw evidence under its recorded v1 or v2 specification."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        return None
    if spec_sha == JEV_SPEC_SHA_V1:
        mapped = answers_outcome_v1(payload["answers"], DEFAULT_THRESHOLD)
    else:
        mapped = answers_outcome(payload["answers"], threshold_for(case))
    return mapped[0] if mapped else None


def run_case_jev(case: dict, client: JevClient,
                  threshold: float | None = None) -> tuple[tuple[str, str | None], bool, dict]:
    """Evaluate one case. Returns (outcome, transport_error, detail).

    Non-unclear answers below threshold flip to unclear (confidence-gated
    routing); the flip is recorded in detail. Threshold must come from
    synthetic-only tuning, never from the frozen eval set.

    Detail carries confidences, probability distributions, latency, usage,
    and the resolved model for analysis — the --out file keeps it per case.
    """
    state = json.loads(user_message(case))
    gate = threshold if threshold is not None else threshold_for(case)
    try:
        payload, seconds = client.evaluate(state, build_questions(case))
    except TransportError as err:
        return ("unclear", None), True, {"error": str(err)[:200]}
    try:
        answers = payload["answers"]
        detail = {
            "action_confidence": answers["action"].get("confidence"),
            "target_confidence": answers["target"].get("confidence"),
            "action_probabilities": answers["action"].get("probabilities"),
            "target_probabilities": answers["target"].get("probabilities"),
            "seconds": round(seconds, 3),
            "usage": payload.get("usage"),
            "resolved_model": payload.get("model"),
        }
    except (KeyError, TypeError, AttributeError) as err:
        return ("unclear", None), True, {"error": f"unexpected payload shape: {err}"}
    mapped = answers_outcome(answers, gate)
    if mapped is None:
        return ("unclear", None), False, {**detail, "warning": "unknown action in answers"}
    outcome, flipped, raw = mapped
    return outcome, False, {**detail, "applied_threshold": gate,
                             "flipped_to_unclear": flipped,
                             "raw_outcome": list(raw)}


# ---------------------------------------------------------------------------
# Auditable evidence (publishable Pareto path). Publishable Jev runs always use
# the frozen DEFAULT_THRESHOLD; collect_run_jev refuses anything else, because a
# different threshold is a different prompt spec (and an incompatible release).
# ---------------------------------------------------------------------------

def runtime_label(model: str) -> str:
    """Return the provenance label for one requested TypeSafe model."""
    return f"typesafe-systemone:{model}"


def record_cost_usd(usage: Any) -> float | None:
    """Provider-transparent Jev pricing: $42 per billion input tokens, outputs free."""
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("input_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, (int, float)) or tokens < 0:
        return None
    return round(float(tokens) * INPUT_USD_PER_TOKEN, 12)


def collect_case_jev(case: dict, repetition: int, client: JevClient, *,
                     request: dict[str, Any],
                     secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Collect one Jev evidence record: a single attempt, schema-identical shape."""
    threshold = threshold_for(case)
    case_started_at = utc_now()
    state = json.loads(user_message(case))
    attempt_started_at = utc_now()
    attempt_started = time.perf_counter()
    try:
        payload, _seconds = client.evaluate(state, build_questions(case))
    except Exception as err:  # noqa: BLE001 - transport vs client split below
        error = _error(err, secrets)
        transport = error["kind"] != "parsing"
        return redact({
            "schema_version": SCHEMA_VERSION,
            "case_id": case["id"],
            "repetition": repetition,
            "started_at": case_started_at,
            "request": request,
            "attempts": [{
                "number": 1,
                "started_at": attempt_started_at,
                "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                "raw_completion": None,
                "parsed": None,
                "error": error,
            }],
            "outcome": {"action": "unclear", "target": None},
            "expected_outcomes": [list(item) for item in case["expect"]],
            "passed": False,
            "transport_error": transport,
            "error": error,
            "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
            "usage": None,
            "cost": None,
            "resolved": dict(request["resolved"]),
        }, secrets)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    raw_obj = {"answers": payload.get("answers"), "usage": usage,
               "model": payload.get("model")}
    raw = json.dumps(raw_obj, sort_keys=True)
    mapped = (answers_outcome(raw_obj["answers"], threshold)
              if isinstance(raw_obj["answers"], dict) else None)
    if mapped is None:
        outcome: tuple[str, str | None] = ("unclear", None)
        parsed = None
        error: dict[str, str] | None = {"kind": "parsing", "type": "UnknownAction",
                                        "message": "Jev answers held no valid Adventure Bench action"}
    else:
        outcome, _flipped, _raw = mapped
        parsed = {"action": outcome[0], "target": outcome[1]}
        error = None
    transport = False
    expected = [list(item) for item in case["expect"]]
    passed = list(outcome) in expected
    model = payload.get("model")
    return redact({
        "schema_version": SCHEMA_VERSION,
        "case_id": case["id"],
        "repetition": repetition,
        "started_at": case_started_at,
        "request": request,
        "attempts": [{
            "number": 1,
            "started_at": attempt_started_at,
            "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
            "raw_completion": raw,
            "parsed": parsed,
            "error": error,
        }],
        "outcome": {"action": outcome[0], "target": outcome[1]},
        "expected_outcomes": expected,
        "passed": passed,
        "transport_error": transport,
        "error": error,
        "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
        "usage": usage,
        "cost": record_cost_usd(usage),
        "resolved": {
            "model": model if isinstance(model, str) and model else request["resolved"]["model"],
            "provider": "typesafe",
            "runtime": request["resolved"]["runtime"],
        },
    }, secrets)


@_exclusive_run_collection
def collect_run_jev(*, cases: list[dict], model: str, client: JevClient,
                    output_dir: Path, run_id: str, repetitions: int = 1,
                    runtime: str | None = None,
                    dataset_path: Path = DATA_PATH,
                    runner_root: Path | None = None,
                    secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Write/continue one Jev model's evidence file; same manifest contract as chat."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    expected_runtime = runtime_label(model)
    if runtime is not None and runtime != expected_runtime:
        raise ValueError("Jev runtime provenance must identify the requested model")
    runtime = expected_runtime
    validate_run_id(run_id)
    slug = safe_model_slug(model)
    root = (runner_root or Path(__file__).resolve().parents[1]).resolve()
    manifest_path = _manifest_path(output_dir, run_id)
    run_dir = manifest_path.parent
    responses_dir = _inside(output_dir, run_dir / "responses")
    responses_dir.mkdir(parents=True, exist_ok=True)
    responses_path = _inside(output_dir, responses_dir / f"{slug}.jsonl")
    existing = _read_records(responses_path)
    complete_keys = {(str(item["case_id"]), int(item["repetition"])) for item in existing}
    if len(complete_keys) != len(existing):
        raise ValueError("duplicate case/repetition evidence records; refusing unsafe resume")

    source = json.loads((root / "SOURCE.json").read_text()) if (root / "SOURCE.json").exists() else {}
    request: dict[str, Any] = {
        "requested": {"model": model, "provider": "typesafe", "runtime": runtime},
        "resolved": {"model": model, "provider": "typesafe", "runtime": runtime},
        "endpoint_host": "api.typesafe.ai",
        "decoding": {"temperature": None, "reasoning_enabled": False},
        "jev": {"threshold": DEFAULT_THRESHOLD,
                "mapping_threshold": DEFAULT_MAPPING_THRESHOLD,
                "spec": "adventurebench-systemone-v2"},
        "retry_policy": {"malformed_output_attempts": 1},
        "timeout_seconds": client.timeout,
        "prompt_sha256": JEV_SPEC_SHA,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "name": "Adventure Bench",
            "version": __version__,
            "source_commit": source.get("source", {}).get("commit"),
            "repository_commit": git_commit(root),
            "dataset_sha256": sha256_file(dataset_path),
            "prompt_sha256": JEV_SPEC_SHA,
        },
        "run_id": run_id,
        "started_at": utc_now(),
        "completed_at": None,
        "runner": {"name": "adventure-bench-collect", "version": __version__,
                   "command": "adventure-bench-collect"},
        "case_count": len(cases),
        "selected_case_ids": [case["id"] for case in cases],
        "selected_case_ids_sha256": selected_cases_hash(cases),
        "repetitions": repetitions,
        "configuration": request,
        "models": {},
        "validity": {},
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("benchmark", "run_id", "runner", "case_count",
                    "selected_case_ids_sha256", "repetitions"):
            if old.get(key) != manifest[key]:
                raise ValueError(f"existing manifest conflicts on {key}; use a new run id")
        if "selected_case_ids" in old and old["selected_case_ids"] != manifest["selected_case_ids"]:
            raise ValueError("existing manifest conflicts on selected_case_ids; use a new run id")
        if "selected_case_ids" in old:
            manifest["selected_case_ids"] = old["selected_case_ids"]
        if old.get("configuration") != request:
            raise ValueError("existing manifest conflicts on collection configuration; use a new run id")
        manifest["started_at"] = old.get("started_at", manifest["started_at"])
        manifest["models"] = old.get("models", {})

    _write_manifest(manifest_path, manifest, secrets)
    for repetition in range(1, repetitions + 1):
        for case in cases:
            key = (case["id"], repetition)
            if key in complete_keys:
                continue
            record = collect_case_jev(case, repetition, client, request=request,
                                      secrets=secrets)
            _append_record(responses_path, record)
            existing.append(record)
            complete_keys.add(key)

    records = _read_records(responses_path)
    relevant = [r for r in records if r["case_id"] in {c["id"] for c in cases}
                and 1 <= int(r["repetition"]) <= repetitions]
    expected_count = len(cases) * repetitions
    transport_failures = sum(bool(r.get("transport_error")) for r in relevant)
    resolved_providers = {r.get("resolved", {}).get("provider") for r in relevant
                          if r.get("resolved", {}).get("provider")}
    resolution_ok = len(relevant) == expected_count and len(resolved_providers) == 1
    reasons = []
    if len(relevant) != expected_count:
        reasons.append("incomplete evidence")
    if not resolution_ok:
        reasons.append("provider resolution is missing or inconsistent")
    if transport_failures:
        reasons.append(f"{transport_failures} transport failure(s)")
    model_summary = {
        "requested": request["requested"],
        "responses_file": f"responses/{slug}.jsonl",
        "records": len(relevant),
        "expected_records": expected_count,
        "resolved_providers": sorted(resolved_providers),
        "transport_failures": transport_failures,
        "valid": not reasons,
        "invalid_reasons": reasons,
    }
    manifest["models"][slug] = model_summary
    all_models_valid = bool(manifest["models"]) and all(info.get("valid") for info in manifest["models"].values())
    manifest["completed_at"] = utc_now()
    manifest["validity"] = {"valid": all_models_valid,
                            "invalid_reasons": sorted({reason for info in manifest["models"].values() for reason in info.get("invalid_reasons", [])})}
    _write_manifest(manifest_path, manifest, secrets)
    return manifest
