"""Offline tests for the staged collection planner.  No provider is contacted."""
from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from adventure_bench.run_spec import (
    RunSpecError,
    collection_plan,
    execute_plan,
    parse_run_spec,
    selected_smoke_cases,
    stage_run_id,
    main,
    maximum_completion_calls,
)
from adventure_bench.runner import TransportError


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {
        "id": "case.1", "source": "original", "tags": ["mapping"], "input": "take lamp",
        "context": {"room": {"name": "Cell", "description": "A lamp."}, "exits": [],
                    "items": [{"id": "lamp", "name": "lamp"}], "carrying": []},
        "expect": [["take", "lamp"]],
    },
    {
        "id": "case.2", "source": "original", "tags": ["calibration"], "input": "sing",
        "context": {"room": {"name": "Cell", "description": "Empty."}, "exits": [], "items": [], "carrying": []},
        "expect": [["unclear", None]],
    },
]


def raw_spec(**changes):
    value = {
        "schema_version": "1.0", "release_id": "pilot-release", "repetitions": 2,
        "smoke_case_ids": ["case.2"],
        "models": [{"model": "Example/Model-One", "provider": "test-provider", "runtime": "offline:test"}],
    }
    value.update(changes)
    return value


class RunSpecTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "cases.jsonl"
        self.data.write_text("".join(json.dumps(case) + "\n" for case in CASES), encoding="utf-8")

    def test_plan_is_stable_and_contains_no_endpoint_or_local_path(self):
        spec = parse_run_spec(raw_spec())
        first = collection_plan(spec, CASES)
        self.assertEqual(first, collection_plan(spec, CASES))
        self.assertEqual(first["models"][0]["smoke"], {"run_id": "pilot-release-smoke-example-model-one", "case_count": 1})
        self.assertEqual(first["models"][0]["full"], {"run_id": "pilot-release-full-example-model-one", "case_count": 2})
        self.assertEqual(first["maximum_completion_calls"], 12)
        text = json.dumps(first)
        self.assertNotIn("http", text)
        self.assertNotIn(str(self.root), text)

    def test_cli_dry_run_needs_no_credentials_or_provider_client(self):
        spec_path = self.root / "release-spec.json"
        spec_path.write_text(json.dumps(raw_spec()), encoding="utf-8")
        stream = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), patch("adventure_bench.run_spec.ChatClient") as client, redirect_stdout(stream):
            main(["--spec", str(spec_path), "--data", str(self.data)])
        output = json.loads(stream.getvalue())
        self.assertTrue(output["dry_run"])
        self.assertEqual(output["plan"]["release_id"], "pilot-release")
        client.assert_not_called()

    def test_invalid_specs_reject_secret_paths_and_colliding_safe_ids(self):
        for changes in (
            {"models": [{"model": "Example/Model", "provider": "https://host/path?token=nope", "runtime": "r"}]},
            {"models": [{"model": "../model", "provider": "host", "runtime": "r"}]},
            {"smoke_case_ids": ["case.1", "case.1"]},
            {"models": [
                {"model": "Example/Model One", "provider": "host", "runtime": "r"},
                {"model": "example-model-one", "provider": "host", "runtime": "r"},
            ]},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(RunSpecError):
                    parse_run_spec(raw_spec(**changes))

        with self.assertRaisesRegex(RunSpecError, "not present"):
            selected_smoke_cases(parse_run_spec(raw_spec(smoke_case_ids=["case.404"])), CASES)

    def test_stage_ids_are_safe_unique_and_deterministic_when_long(self):
        spec = parse_run_spec(raw_spec(release_id="r" * 64))
        model = spec.models[0]
        smoke = stage_run_id(spec, model, "smoke")
        full = stage_run_id(spec, model, "full")
        self.assertLessEqual(len(smoke), 64)
        self.assertNotEqual(smoke, full)
        self.assertEqual(smoke, stage_run_id(spec, model, "smoke"))

    def test_invalid_smoke_halts_before_the_full_run(self):
        spec = parse_run_spec(raw_spec())
        calls = []

        def failing_factory(_model):
            def complete(_messages):
                calls.append(1)
                raise TransportError("https://api.example.test/private?api_key=secret")
            return complete

        with self.assertRaisesRegex(RunSpecError, "full collection was not started"):
            execute_plan(
                spec=spec, cases=CASES, output_dir=self.root / "runs", complete_factory=failing_factory,
                max_completions=maximum_completion_calls(spec, CASES),
                dataset_path=self.data, runner_root=ROOT, base_url="https://api.example.test/private?api_key=secret",
                secrets=("secret",),
            )
        # The smoke stage has one case and two requested repetitions. A
        # transport error is evidence for each repetition, but no full case
        # is collected after the smoke manifest is marked invalid.
        self.assertEqual(len(calls), 2)
        self.assertTrue((self.root / "runs" / "pilot-release-smoke-example-model-one" / "manifest.json").exists())
        self.assertFalse((self.root / "runs" / "pilot-release-full-example-model-one").exists())
        evidence = (self.root / "runs" / "pilot-release-smoke-example-model-one" / "responses" / "example-model-one.jsonl").read_text()
        self.assertNotIn("private", evidence)
        self.assertNotIn("api_key", evidence)
        self.assertNotIn("secret", evidence)

    def test_execute_is_resume_safe_after_a_complete_plan(self):
        spec = parse_run_spec(raw_spec())
        calls = []

        def factory(_model):
            def complete(messages):
                calls.append(1)
                command = json.loads(messages[1]["content"])["input"]
                return '{"action":"take","target":"lamp"}' if command == "take lamp" else '{"action":"unclear","target":null}'
            return complete

        first = execute_plan(
            spec=spec, cases=CASES, output_dir=self.root / "runs", complete_factory=factory,
            max_completions=maximum_completion_calls(spec, CASES),
            dataset_path=self.data, runner_root=ROOT, base_url="https://api.example.test/v1",
        )
        self.assertTrue(first["models"][0]["full"]["valid"])
        self.assertEqual(len(calls), 6)  # one smoke case + two full cases, twice each

        def must_not_call(_model):
            def complete(_messages):
                raise AssertionError("completed collection must resume without a provider call")
            return complete

        second = execute_plan(
            spec=spec, cases=CASES, output_dir=self.root / "runs", complete_factory=must_not_call,
            max_completions=maximum_completion_calls(spec, CASES),
            dataset_path=self.data, runner_root=ROOT, base_url="https://api.example.test/v1",
        )
        self.assertEqual(first, second)

    def test_completion_ceiling_and_invalid_full_halt_before_later_models(self):
        spec = parse_run_spec(raw_spec(models=[
            {"model": "Example/Model-One", "provider": "test-provider", "runtime": "offline:test"},
            {"model": "Example/Model-Two", "provider": "test-provider", "runtime": "offline:test"},
        ]))
        factories = []

        def factory(model):
            factories.append(model.slug)

            def complete(messages):
                command = json.loads(messages[1]["content"])["input"]
                if model.slug == "example-model-one" and command == "take lamp":
                    raise TransportError("full-stage failure")
                return '{"action":"unclear","target":null}'

            return complete

        required = maximum_completion_calls(spec, CASES)
        with self.assertRaisesRegex(RunSpecError, "planned ceiling"):
            execute_plan(
                spec=spec, cases=CASES, output_dir=self.root / "under-budget", complete_factory=factory,
                max_completions=required - 1, dataset_path=self.data, runner_root=ROOT,
            )
        self.assertEqual(factories, [])

        with self.assertRaisesRegex(RunSpecError, "later models were not started"):
            execute_plan(
                spec=spec, cases=CASES, output_dir=self.root / "runs", complete_factory=factory,
                max_completions=required, dataset_path=self.data, runner_root=ROOT,
                base_url="https://api.example.test/v1",
            )
        self.assertEqual(factories, ["example-model-one"])
        self.assertFalse((self.root / "runs" / "pilot-release-smoke-example-model-two").exists())


if __name__ == "__main__":
    unittest.main()
