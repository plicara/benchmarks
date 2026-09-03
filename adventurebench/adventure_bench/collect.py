"""Auditable collection for Adventure Bench v1.

This module deliberately reuses the frozen v1 prompt, parser, noun
normalisation, and malformed-output retry from :mod:`adventure_bench.runner`.
It records evidence; it does not change how an answer is scored.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from . import __version__
from .runner import DATA_PATH, SYSTEM_PROMPT, ChatClient, TransportError, api_key_from_env, extract_json, load_cases, reply_to_outcome, user_message


SCHEMA_VERSION = "1.0"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SECRET_KEY_RE = re.compile(
    r"(?i)(?:^|[-_])(authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|token|secret|password|user[-_]?id)(?:$|[-_])"
)
URL_WITH_PATH_RE = re.compile(r"https?://([^/\s?#]+)(?:[/\?#][^\s,;]*)?", re.IGNORECASE)


@dataclass(frozen=True)
class Completion:
    """A provider reply and the non-secret metadata returned with it.

    Custom integrations may return this object from their completion callable.
    The built-in OpenAI-compatible client only has requested provenance, so a
    provider must be explicitly pinned with ``--provider`` for a valid run.
    """

    text: str
    model: str | None = None
    provider: str | None = None
    runtime: str | None = None
    usage: dict[str, Any] | None = None
    cost: float | None = None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_cases_hash(cases: list[dict]) -> str:
    """Hash the ordered selection, so a smoke subset cannot resume as another."""
    return sha256_text("\n".join(str(case["id"]) for case in cases) + "\n")


def safe_model_slug(model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    if not slug:
        raise ValueError("model must contain at least one letter or digit")
    if len(slug) <= 80:
        return slug
    digest = sha256_text(model)[:10]
    return f"{slug[:69].rstrip('-')}-{digest}"


def make_run_id(model: str, when: datetime | None = None) -> str:
    """Make a path-safe ID deterministically from an instant and model ID."""
    instant = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    prefix = instant.strftime("%Y%m%dT%H%M%SZ").lower()
    model_slug = safe_model_slug(model)
    available = 64 - len(prefix) - 1
    if len(model_slug) > available:
        digest = sha256_text(model)[:10]
        model_slug = f"{model_slug[:available - 11].rstrip('-')}-{digest}"
    run_id = f"{prefix}-{model_slug}"
    if not RUN_ID_RE.fullmatch(run_id):  # defensive guard if the format changes
        raise ValueError(f"unsafe generated run id: {run_id!r}")
    return run_id


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run id must use lowercase letters, digits, and hyphens only (max 64 characters)")
    return run_id


def _inside(root: Path, child: Path) -> Path:
    resolved_root = root.resolve()
    resolved_child = child.resolve()
    if resolved_child != resolved_root and resolved_root not in resolved_child.parents:
        raise ValueError("refusing a path outside the configured output directory")
    return resolved_child


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def endpoint_host(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an absolute http(s) URL")
    return parsed.hostname.lower()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def redact(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    """Return JSON-safe evidence after removing credential-shaped fields/values."""
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if SECRET_KEY_RE.search(str(key)) else redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        clean = value
        for secret in secrets:
            if secret:
                clean = clean.replace(secret, "[redacted]")
        clean = re.sub(
            r"(?i)authorization\s*[:=]\s*(?:bearer\s+)?[^\s,;]+",
            "Authorization: [redacted]",
            clean,
        )
        clean = re.sub(
            r"(?i)\b(api[-_]?key|access[-_]?token|refresh[-_]?token|token|secret|password)\s*[:=]\s*[^\s,;]+",
            lambda match: f"{match.group(1)}=[redacted]",
            clean,
        )
        clean = re.sub(
            r'(?i)["\']?user[-_]?id["\']?\s*[=:]\s*(?:"[^"]*"|[^\s,;}\]]+)',
            "user_id=[redacted]",
            clean,
        )
        # Evidence records endpoint hosts, never endpoint paths or query strings.
        # Errors emitted by HTTP clients can include a request URL, so normalise
        # those before writing them as well.
        clean = URL_WITH_PATH_RE.sub(lambda match: f"https://{match.group(1)}", clean)
        return clean
    return value


def _error(err: BaseException, secrets: tuple[str, ...]) -> dict[str, str]:
    return {
        "kind": "transport" if isinstance(err, TransportError) else "client",
        "type": type(err).__name__,
        "message": redact(str(err), secrets),
    }


def _completion(value: Any) -> Completion:
    if isinstance(value, Completion):
        return value
    if isinstance(value, str):
        return Completion(text=value)
    raise TypeError("completion callable must return str or Completion")


def _observed_provider(data: dict[str, Any]) -> str | None:
    metadata = data.get("openrouter_metadata")
    if not isinstance(metadata, dict):
        return None
    endpoints = metadata.get("endpoints")
    available = endpoints.get("available") if isinstance(endpoints, dict) else None
    if not isinstance(available, list):
        return None
    selected = [item.get("provider") for item in available if isinstance(item, dict) and item.get("selected") is True]
    return selected[0] if len(selected) == 1 and isinstance(selected[0], str) and selected[0] else None


def evidence_completion(client: ChatClient) -> Callable[[list[dict]], Completion]:
    """Adapt a client response into completion text plus provider-supplied evidence."""
    def complete(messages: list[dict]) -> Completion:
        data = client.complete_response(messages, routing_metadata=True)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        cost_value = usage.get("cost") if usage else None
        cost = float(cost_value) if isinstance(cost_value, (int, float)) and not isinstance(cost_value, bool) else None
        model = data.get("model") if isinstance(data.get("model"), str) else None
        return Completion(
            text=client.response_text(data),
            model=model,
            provider=_observed_provider(data),
            usage=usage,
            cost=cost,
        )

    return complete


def _expected(case: dict) -> list[list[Any]]:
    return [list(item) for item in case["expect"]]


def collect_case(case: dict, repetition: int, complete: Callable[[list[dict]], Any], *, request: dict[str, Any], secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Collect one evidence record with the exact frozen malformed-output retry."""
    case_started_at = utc_now()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(case)},
    ]
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()
    final_outcome: tuple[str, str | None] = ("unclear", None)
    error: dict[str, str] | None = None
    resolved: Completion | None = None
    for number in range(1, 3):
        attempt_started_at = utc_now()
        attempt_started = time.perf_counter()
        try:
            completion = _completion(complete(messages))
        except Exception as err:  # client adapters may expose non-stdlib errors
            error = _error(err, secrets)
            attempts.append({
                "number": number,
                "started_at": attempt_started_at,
                "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                "raw_completion": None,
                "parsed": None,
                "error": error,
            })
            break
        resolved = completion
        obj = extract_json(completion.text)
        outcome = reply_to_outcome(case, obj)
        parse_error = None if outcome is not None else {
            "kind": "parsing", "type": "MalformedOutput",
            "message": "reply did not contain one valid Adventure Bench action object",
        }
        attempts.append({
            "number": number,
            "started_at": attempt_started_at,
            "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
            "raw_completion": redact(completion.text, secrets),
            "parsed": {"action": outcome[0], "target": outcome[1]} if outcome else None,
            "error": parse_error,
        })
        if outcome is not None:
            final_outcome = outcome
            break
        messages.append({"role": "assistant", "content": completion.text})
        messages.append({"role": "user", "content": "That was not a single valid JSON object matching the schema. Reply with only the JSON object."})

    if error is None and attempts and attempts[-1]["parsed"] is None:
        error = attempts[-1]["error"]
    transport = error is not None and error["kind"] != "parsing"
    expected = _expected(case)
    passed = not transport and list(final_outcome) in expected
    provider = (resolved.provider if resolved else None) or request["resolved"]["provider"]
    model = (resolved.model if resolved else None) or request["resolved"]["model"]
    runtime = (resolved.runtime if resolved else None) or request["resolved"]["runtime"]
    return redact({
        "schema_version": SCHEMA_VERSION,
        "case_id": case["id"],
        "repetition": repetition,
        "started_at": case_started_at,
        "request": request,
        "attempts": attempts,
        "outcome": {"action": final_outcome[0], "target": final_outcome[1]},
        "expected_outcomes": expected,
        "passed": passed,
        "transport_error": transport,
        "error": error,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "usage": resolved.usage if resolved else None,
        "cost": resolved.cost if resolved else None,
        "resolved": {"model": model, "provider": provider, "runtime": runtime},
    }, secrets)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"malformed JSONL evidence at {path}:{number}") from err
        if not isinstance(record, dict) or "case_id" not in record or "repetition" not in record:
            raise ValueError(f"invalid evidence record at {path}:{number}")
        records.append(record)
    return records


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _manifest_path(output_dir: Path, run_id: str) -> Path:
    return _inside(output_dir, output_dir / validate_run_id(run_id) / "manifest.json")


def _write_manifest(path: Path, manifest: dict[str, Any], secrets: tuple[str, ...]) -> None:
    path.write_text(json.dumps(redact(manifest, secrets), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def _collection_lock(run_dir: Path):
    """Prevent concurrent writers from corrupting one run's append-only evidence."""
    lock_path = run_dir / ".collection.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise ValueError(f"collection already active for {run_dir.name}; wait for it to finish or use a new run id") from err
        acquired = True
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        if acquired:
            lock_path.unlink(missing_ok=True)


def _exclusive_run_collection(function):
    """Apply a non-blocking advisory lock for the lifetime of a collection."""
    @wraps(function)
    def locked(*args, **kwargs):
        output_dir = kwargs["output_dir"]
        run_id = validate_run_id(kwargs["run_id"])
        run_dir = _inside(output_dir, output_dir / run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        with _collection_lock(run_dir):
            return function(*args, **kwargs)
    return locked


@_exclusive_run_collection
def collect_run(*, cases: list[dict], model: str, complete: Callable[[list[dict]], Any], output_dir: Path, run_id: str, repetitions: int = 1, requested_provider: str | None = None, resolved_model: str | None = None, resolved_provider: str | None = None, runtime: str | None = None, base_url: str | None = None, timeout_seconds: float = 60.0, max_output_tokens: int | None = None, dataset_path: Path = DATA_PATH, runner_root: Path | None = None, secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Write/continue one model's evidence file and return its final manifest."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if max_output_tokens is not None and (isinstance(max_output_tokens, bool) or max_output_tokens < 1):
        raise ValueError("max_output_tokens must be at least one")
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
    request = {
        "requested": {"model": model, "provider": requested_provider, "runtime": runtime},
        "resolved": {"model": resolved_model or model, "provider": resolved_provider, "runtime": runtime},
        "endpoint_host": endpoint_host(base_url),
        "decoding": {"temperature": 0, "reasoning_enabled": False, "max_output_tokens": max_output_tokens},
        "retry_policy": {"malformed_output_attempts": 2, "http_429_delays_seconds": [2, 4, 8]},
        "timeout_seconds": timeout_seconds,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "name": "Adventure Bench",
            "version": __version__,
            "source_commit": source.get("source", {}).get("commit"),
            "repository_commit": git_commit(root),
            "dataset_sha256": sha256_file(dataset_path),
            "prompt_sha256": sha256_text(SYSTEM_PROMPT),
        },
        "run_id": run_id,
        "started_at": utc_now(),
        "completed_at": None,
        "runner": {"name": "adventure-bench-collect", "version": __version__, "command": "adventure-bench-collect"},
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
        for key in ("benchmark", "run_id", "runner", "case_count", "selected_case_ids_sha256", "repetitions"):
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
            record = collect_case(case, repetition, complete, request=request, secrets=secrets)
            _append_record(responses_path, record)
            existing.append(record)
            complete_keys.add(key)

    records = _read_records(responses_path)
    relevant = [r for r in records if r["case_id"] in {c["id"] for c in cases} and 1 <= int(r["repetition"]) <= repetitions]
    expected_count = len(cases) * repetitions
    transport_failures = sum(bool(r.get("transport_error")) for r in relevant)
    resolved_providers = {r.get("resolved", {}).get("provider") for r in relevant if r.get("resolved", {}).get("provider")}
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
    manifest["validity"] = {"valid": all_models_valid, "invalid_reasons": sorted({reason for info in manifest["models"].values() for reason in info.get("invalid_reasons", [])})}
    _write_manifest(manifest_path, manifest, secrets)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect auditable Adventure Bench response evidence.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", help="Pinned requested provider; required for a valid default-client run")
    parser.add_argument("--resolved-model")
    parser.add_argument("--resolved-provider", help="Provider reported by a trusted runtime adapter")
    parser.add_argument("--runtime", help="Pinned runtime identity, such as a local server image digest")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, help="Bound each model completion and record the cap in evidence")
    parser.add_argument("--data")
    parser.add_argument("--tag")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-id")
    options = parser.parse_args(argv)
    api_key = api_key_from_env()
    if not api_key:
        parser.error("set ADVENTURE_BENCH_API_KEY, OPENROUTER_API_KEY, or OPENROUTER_KEY")
    data_path = Path(options.data) if options.data else DATA_PATH
    cases = load_cases(data_path)
    if options.tag:
        cases = [case for case in cases if options.tag in case["tags"]]
    if options.limit is not None:
        cases = cases[:options.limit]
    if not cases:
        parser.error("no cases matched")
    base_url = options.base_url or os.environ.get("ADVENTURE_BENCH_BASE_URL", "https://openrouter.ai/api/v1")
    client = ChatClient(api_key, model=options.model, base_url=base_url, provider=options.provider, timeout=options.timeout, max_output_tokens=options.max_output_tokens)
    run_id = options.run_id or make_run_id(options.model)
    try:
        manifest = collect_run(
            cases=cases, model=options.model, complete=evidence_completion(client), output_dir=Path(options.out_dir),
            run_id=run_id, repetitions=options.repetitions, requested_provider=options.provider,
            resolved_model=options.resolved_model or options.model,
            resolved_provider=options.resolved_provider or options.provider, runtime=options.runtime,
            base_url=base_url, timeout_seconds=options.timeout, max_output_tokens=options.max_output_tokens,
            dataset_path=data_path, secrets=(api_key,),
        )
    except ValueError as err:
        parser.error(str(err))
    print(json.dumps({"run_id": run_id, "valid": manifest["validity"]["valid"], "manifest": f"{options.out_dir}/{run_id}/manifest.json"}))


if __name__ == "__main__":
    main()
