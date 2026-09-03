"""Staged, opt-in collection from a public Adventure Bench release spec.

The spec freezes a release decision without containing credentials, endpoint
URLs, or local paths.  Planning is offline by default; collection requires an
explicit ``--execute`` flag and reuses :func:`collect_run` for every stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .collect import collect_run, evidence_completion, safe_model_slug, validate_run_id
from .runner import DATA_PATH, ChatClient, api_key_from_env, load_cases


SPEC_VERSION = "1.0"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}$")


class RunSpecError(ValueError):
    """A release plan is malformed or cannot be carried out safely."""


@dataclass(frozen=True)
class ModelSpec:
    model: str
    provider: str
    runtime: str

    @property
    def slug(self) -> str:
        return safe_model_slug(self.model)


@dataclass(frozen=True)
class RunSpec:
    release_id: str
    repetitions: int
    smoke_case_ids: tuple[str, ...]
    models: tuple[ModelSpec, ...]


def _identifier(value: Any, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise RunSpecError(f"{field} has unsupported characters")
    if "//" in value or any(part in {".", ".."} for part in value.split("/")):
        raise RunSpecError(f"{field} must not contain path segments")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunSpecError(f"{field} must be an integer of at least one")
    return value


def parse_run_spec(value: Any) -> RunSpec:
    """Validate an in-memory release spec without accepting secret-bearing fields."""
    if not isinstance(value, dict):
        raise RunSpecError("release spec must be a JSON object")
    required = {"schema_version", "release_id", "repetitions", "smoke_case_ids", "models"}
    if set(value) != required:
        raise RunSpecError("release spec must contain only schema_version, release_id, repetitions, smoke_case_ids, and models")
    if value["schema_version"] != SPEC_VERSION:
        raise RunSpecError(f"unsupported release spec schema_version: {value['schema_version']!r}")
    try:
        release_id = validate_run_id(value["release_id"])
    except ValueError as err:
        raise RunSpecError(str(err)) from err
    repetitions = _positive_integer(value["repetitions"], field="repetitions")

    smoke_value = value["smoke_case_ids"]
    if not isinstance(smoke_value, list) or not smoke_value:
        raise RunSpecError("smoke_case_ids must be a non-empty list")
    smoke_case_ids = tuple(_identifier(case_id, field="smoke case id", pattern=CASE_ID_RE) for case_id in smoke_value)
    if len(set(smoke_case_ids)) != len(smoke_case_ids):
        raise RunSpecError("smoke_case_ids must be unique")

    models_value = value["models"]
    if not isinstance(models_value, list) or not models_value:
        raise RunSpecError("models must be a non-empty list")
    models: list[ModelSpec] = []
    for index, entry in enumerate(models_value, 1):
        if not isinstance(entry, dict) or set(entry) != {"model", "provider", "runtime"}:
            raise RunSpecError(f"models[{index}] must contain only model, provider, and runtime")
        models.append(ModelSpec(
            model=_identifier(entry["model"], field=f"models[{index}].model", pattern=MODEL_RE),
            provider=_identifier(entry["provider"], field=f"models[{index}].provider", pattern=IDENTITY_RE),
            runtime=_identifier(entry["runtime"], field=f"models[{index}].runtime", pattern=IDENTITY_RE),
        ))
    slugs = [model.slug for model in models]
    if len(set(slugs)) != len(slugs):
        raise RunSpecError("models must have unique safe model IDs")
    return RunSpec(release_id, repetitions, smoke_case_ids, tuple(models))


def load_run_spec(path: Path) -> RunSpec:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise RunSpecError("could not read release spec") from err
    except json.JSONDecodeError as err:
        raise RunSpecError("release spec is not valid JSON") from err
    return parse_run_spec(value)


def stage_run_id(spec: RunSpec, model: ModelSpec, stage: str) -> str:
    """Derive a stable, path-safe run ID; unlike collection it has no clock."""
    if stage not in {"smoke", "full"}:
        raise RunSpecError("unknown collection stage")
    value = f"{spec.release_id}-{stage}-{model.slug}"
    if len(value) > 64:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
        room = 64 - len(stage) - len(digest) - 2
        value = f"{spec.release_id[:room].rstrip('-')}-{stage}-{digest}"
    return validate_run_id(value)


def selected_smoke_cases(spec: RunSpec, cases: list[dict]) -> list[dict]:
    by_id = {case.get("id"): case for case in cases}
    missing = [case_id for case_id in spec.smoke_case_ids if case_id not in by_id]
    if missing:
        raise RunSpecError("smoke_case_ids are not present in the selected dataset")
    return [by_id[case_id] for case_id in spec.smoke_case_ids]


def _validate_stage_ids(spec: RunSpec) -> None:
    run_ids = [stage_run_id(spec, model, stage) for model in spec.models for stage in ("smoke", "full")]
    if len(set(run_ids)) != len(run_ids):
        raise RunSpecError("release spec produces duplicate run IDs")


def maximum_completion_calls(spec: RunSpec, cases: list[dict]) -> int:
    """Return the hard request ceiling, including the frozen retry attempt."""
    smoke = selected_smoke_cases(spec, cases)
    attempts_per_case = 2
    return len(spec.models) * spec.repetitions * (len(smoke) + len(cases)) * attempts_per_case


def collection_plan(spec: RunSpec, cases: list[dict]) -> dict[str, Any]:
    """Return a stable, credential-free plan.  This function has no side effects."""
    smoke = selected_smoke_cases(spec, cases)
    _validate_stage_ids(spec)
    return {
        "schema_version": SPEC_VERSION,
        "release_id": spec.release_id,
        "repetitions": spec.repetitions,
        "maximum_completion_calls": maximum_completion_calls(spec, cases),
        "models": [
            {
                "model": model.model,
                "model_slug": model.slug,
                "provider": model.provider,
                "runtime": model.runtime,
                "smoke": {"run_id": stage_run_id(spec, model, "smoke"), "case_count": len(smoke)},
                "full": {"run_id": stage_run_id(spec, model, "full"), "case_count": len(cases)},
            }
            for model in spec.models
        ],
    }


def execute_plan(*, spec: RunSpec, cases: list[dict], output_dir: Path, complete_factory: Callable[[ModelSpec], Callable[[list[dict]], Any]], max_completions: int, dataset_path: Path = DATA_PATH, runner_root: Path | None = None, base_url: str | None = None, timeout_seconds: float = 60.0, max_output_tokens: int | None = None, secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Collect smoke evidence before full evidence for each model.

    A completed run is delegated to ``collect_run`` which makes reruns safe.
    Any invalid smoke result raises before its full run or a later model begins.
    """
    smoke_cases = selected_smoke_cases(spec, cases)
    _validate_stage_ids(spec)
    required_completions = maximum_completion_calls(spec, cases)
    if isinstance(max_completions, bool) or not isinstance(max_completions, int) or max_completions < required_completions:
        raise RunSpecError(f"max_completions must be at least the planned ceiling of {required_completions}")
    completed: list[dict[str, Any]] = []
    for model in spec.models:
        complete = complete_factory(model)
        smoke_id = stage_run_id(spec, model, "smoke")
        smoke_manifest = collect_run(
            cases=smoke_cases, model=model.model, complete=complete, output_dir=output_dir,
            run_id=smoke_id, repetitions=spec.repetitions, requested_provider=model.provider,
            resolved_model=model.model, resolved_provider=model.provider, runtime=model.runtime,
            base_url=base_url, timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens, dataset_path=dataset_path,
            runner_root=runner_root, secrets=secrets,
        )
        if not smoke_manifest["validity"]["valid"]:
            raise RunSpecError(f"smoke collection is invalid for {model.slug}; full collection was not started")
        full_id = stage_run_id(spec, model, "full")
        full_manifest = collect_run(
            cases=cases, model=model.model, complete=complete, output_dir=output_dir,
            run_id=full_id, repetitions=spec.repetitions, requested_provider=model.provider,
            resolved_model=model.model, resolved_provider=model.provider, runtime=model.runtime,
            base_url=base_url, timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens, dataset_path=dataset_path,
            runner_root=runner_root, secrets=secrets,
        )
        if not full_manifest["validity"]["valid"]:
            raise RunSpecError(f"full collection is invalid for {model.slug}; later models were not started")
        completed.append({
            "model_slug": model.slug,
            "smoke": {"run_id": smoke_id, "valid": smoke_manifest["validity"]["valid"]},
            "full": {"run_id": full_id, "valid": full_manifest["validity"]["valid"]},
        })
    return {"schema_version": SPEC_VERSION, "release_id": spec.release_id, "models": completed}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan or explicitly execute staged Adventure Bench collection.")
    parser.add_argument("--spec", required=True, help="Public JSON release specification")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--data")
    parser.add_argument("--base-url", help="Endpoint used only while executing; never copied into evidence")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, help="Bound each model completion and record the cap in evidence")
    parser.add_argument("--execute", action="store_true", help="Permit provider requests after the smoke stage")
    parser.add_argument("--max-completions", type=int, help="Required with --execute; must cover the dry-run ceiling")
    options = parser.parse_args(argv)
    if options.timeout <= 0:
        parser.error("timeout must be greater than zero")
    try:
        spec = load_run_spec(Path(options.spec))
        data_path = Path(options.data) if options.data else DATA_PATH
        cases = load_cases(data_path)
        plan = collection_plan(spec, cases)
    except (RunSpecError, ValueError) as err:
        parser.error(str(err))
    if not options.execute:
        print(json.dumps({"dry_run": True, "plan": plan}, sort_keys=True, separators=(",", ":")))
        return
    if options.max_completions is None:
        parser.error("--execute requires --max-completions at least equal to the dry-run maximum_completion_calls")
    api_key = api_key_from_env()
    if not api_key:
        parser.error("set ADVENTURE_BENCH_API_KEY, OPENROUTER_API_KEY, or OPENROUTER_KEY before --execute")
    base_url = options.base_url or os.environ.get("ADVENTURE_BENCH_BASE_URL", "https://openrouter.ai/api/v1")

    def complete_factory(model: ModelSpec) -> Callable[[list[dict]], Any]:
        client = ChatClient(api_key, model=model.model, base_url=base_url, provider=model.provider, timeout=options.timeout, max_output_tokens=options.max_output_tokens)
        return evidence_completion(client)

    try:
        completed = execute_plan(
            spec=spec, cases=cases, output_dir=Path(options.runs_dir), complete_factory=complete_factory,
            max_completions=options.max_completions,
            dataset_path=data_path, base_url=base_url, timeout_seconds=options.timeout, secrets=(api_key,),
            max_output_tokens=options.max_output_tokens,
        )
    except (RunSpecError, ValueError) as err:
        parser.error(str(err))
    print(json.dumps(completed, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
