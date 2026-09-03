"""Offline evidence-collection tests.  No network or credentials required."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from adventure_bench.collect import (
    Completion,
    _collection_lock,
    collect_run,
    evidence_completion,
    make_run_id,
    redact,
    safe_model_slug,
    validate_run_id,
)
from adventure_bench.runner import ChatClient, TransportError, api_key_from_env


CASE = {
    "id": "case.1", "source": "original", "tags": ["synonym"], "input": "grab the light",
    "context": {
        "room": {"name": "Cell", "description": "A cell with a lamp."},
        "exits": ["north"], "items": [{"id": "brass_lantern", "name": "brass lamp"}], "carrying": [],
    },
    "expect": [["take", "brass_lantern"]],
}
ROOT = Path(__file__).resolve().parents[1]


def records(root: Path, run_id: str, slug: str) -> list[dict]:
    path = root / run_id / "responses" / f"{slug}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


class CollectionTest(unittest.TestCase):
    def run_collection(self, complete, *, cases=None, **kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / "runs"
        settings = {
            "requested_provider": "pinned-host",
            "resolved_model": "example-model-1",
            "resolved_provider": "pinned-host",
            "runtime": "openai-compatible:test",
            "base_url": "https://api.example.test/v1?api_key=nope",
            "runner_root": ROOT,
        }
        settings.update(kwargs)
        manifest = collect_run(
            cases=deepcopy(cases or [CASE]), model="Example/Model 1", complete=complete,
            output_dir=root, run_id="pilot-example", **settings,
        )
        return root, manifest

    def test_client_records_a_configured_output_cap_in_its_request(self):
        client = ChatClient("secret", model="Example/Model", max_output_tokens=128)

        def inspect(request):
            return json.loads(request.data)

        with patch.object(client, "_post", side_effect=inspect):
            payload = client.complete_response([])

        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["reasoning"], {"enabled": False})

    def test_openrouter_key_environment_alias_is_supported(self):
        with patch.dict("os.environ", {"OPENROUTER_KEY": "key-from-dotenv"}, clear=True):
            self.assertEqual(api_key_from_env(), "key-from-dotenv")

    def test_records_raw_attempts_and_frozen_retry(self):
        replies = iter(["not JSON", Completion('{"action":"take", "target":"lamp"}', usage={"total_tokens": 17}, cost=0.002)])
        seen = []

        def complete(messages):
            seen.append(messages)
            return next(replies)

        root, manifest = self.run_collection(complete)
        self.assertTrue(manifest["validity"]["valid"])
        record = records(root, "pilot-example", "example-model-1")[0]
        self.assertEqual(record["outcome"], {"action": "take", "target": "brass_lantern"})
        self.assertTrue(record["passed"])
        self.assertEqual(len(record["attempts"]), 2)
        self.assertEqual(record["attempts"][0]["raw_completion"], "not JSON")
        self.assertIsNone(record["attempts"][0]["parsed"])
        self.assertEqual(record["attempts"][0]["error"]["kind"], "parsing")
        self.assertEqual(record["usage"], {"total_tokens": 17})
        self.assertEqual(record["cost"], 0.002)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1][-2]["role"], "assistant")
        self.assertEqual(record["request"]["endpoint_host"], "api.example.test")
        self.assertNotIn("api_key", json.dumps(record))
        self.assertEqual(manifest["runner"]["command"], "adventure-bench-collect")
        self.assertEqual(len(manifest["selected_case_ids_sha256"]), 64)

    def test_transport_failure_is_structured_invalid_and_secret_free(self):
        secret = "super-secret-token"

        def broken(_messages):
            raise TransportError(f"Authorization: Bearer {secret}; api_key=unknown-secret; {secret}")

        root, manifest = self.run_collection(broken, secrets=(secret,))
        self.assertFalse(manifest["validity"]["valid"])
        self.assertIn("1 transport failure(s)", manifest["validity"]["invalid_reasons"])
        record = records(root, "pilot-example", "example-model-1")[0]
        self.assertTrue(record["transport_error"])
        self.assertEqual(record["error"]["kind"], "transport")
        self.assertEqual(record["attempts"][0]["error"]["type"], "TransportError")
        text = (root / "pilot-example" / "manifest.json").read_text() + json.dumps(record)
        self.assertNotIn(secret, text)
        self.assertNotIn("Bearer " + secret, text)
        self.assertNotIn("unknown-secret", text)

    def test_resume_never_duplicates_completed_case_repetitions(self):
        calls = []

        def complete(_messages):
            calls.append(1)
            return '{"action":"take","target":"lamp"}'

        root, _ = self.run_collection(complete, repetitions=3)
        self.assertEqual(len(calls), 3)

        def must_not_run(_messages):
            raise AssertionError("completed evidence must not be recollected")

        second = collect_run(
            cases=[CASE], model="Example/Model 1", complete=must_not_run, output_dir=root,
            run_id="pilot-example", repetitions=3, requested_provider="pinned-host",
            resolved_model="example-model-1", resolved_provider="pinned-host",
            runtime="openai-compatible:test", base_url="https://api.example.test/v1", runner_root=ROOT,
        )
        self.assertTrue(second["validity"]["valid"])
        result = records(root, "pilot-example", "example-model-1")
        self.assertEqual([(item["case_id"], item["repetition"]) for item in result], [("case.1", 1), ("case.1", 2), ("case.1", 3)])

    def test_active_collection_lock_rejects_another_writer(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / "runs"
        run_dir = root / "pilot-example"
        run_dir.mkdir(parents=True)

        with _collection_lock(run_dir):
            with self.assertRaisesRegex(ValueError, "collection already active"):
                collect_run(
                    cases=[CASE], model="Example/Model 1", complete=lambda _messages: "unused",
                    output_dir=root, run_id="pilot-example", requested_provider="pinned-host",
                    resolved_model="example-model-1", resolved_provider="pinned-host",
                    runtime="openai-compatible:test", base_url="https://api.example.test/v1", runner_root=ROOT,
                )

    def test_repetitions_and_provider_provenance_are_per_record(self):
        other = deepcopy(CASE)
        other["id"] = "case.2"
        root, manifest = self.run_collection(
            lambda _messages: Completion('{"action":"take","target":"lamp"}', model="served-model", provider="observed-host", runtime="runtime@sha256:abc"),
            cases=[CASE, other], repetitions=2,
        )
        output = records(root, "pilot-example", "example-model-1")
        self.assertEqual(len(output), 4)
        self.assertEqual({(item["case_id"], item["repetition"]) for item in output}, {("case.1", 1), ("case.1", 2), ("case.2", 1), ("case.2", 2)})
        self.assertEqual({item["resolved"]["provider"] for item in output}, {"observed-host"})
        self.assertEqual({item["resolved"]["model"] for item in output}, {"served-model"})
        self.assertEqual({item["resolved"]["runtime"] for item in output}, {"runtime@sha256:abc"})
        self.assertEqual(manifest["models"]["example-model-1"]["resolved_providers"], ["observed-host"])

    def test_persistent_malformed_output_has_two_attempts_and_not_transport_error(self):
        root, manifest = self.run_collection(lambda _messages: "still not JSON")
        record = records(root, "pilot-example", "example-model-1")[0]
        self.assertFalse(record["transport_error"])
        self.assertEqual(record["outcome"], {"action": "unclear", "target": None})
        self.assertEqual(len(record["attempts"]), 2)
        self.assertEqual(record["error"]["kind"], "parsing")
        self.assertFalse(record["passed"])
        self.assertTrue(manifest["validity"]["valid"])

    def test_safe_identifiers_reject_path_traversal(self):
        self.assertEqual(safe_model_slug("ACME/Model_One"), "acme-model-one")
        self.assertEqual(make_run_id("ACME/Model", when=datetime(2026, 1, 2, tzinfo=timezone.utc)), "20260102t000000z-acme-model")
        long_a = "publisher/" + "a" * 100
        long_b = "publisher/" + "a" * 99 + "b"
        self.assertLessEqual(len(safe_model_slug(long_a)), 80)
        self.assertNotEqual(safe_model_slug(long_a), safe_model_slug(long_b))
        self.assertLessEqual(len(make_run_id(long_a)), 64)
        for unsafe in ("../run", "run/name", "Run", "", "a" * 65):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    validate_run_id(unsafe)

    def test_unresolved_provider_invalidates_evidence(self):
        root, manifest = self.run_collection(lambda _messages: '{"action":"take","target":"lamp"}', resolved_provider=None)
        self.assertFalse(manifest["validity"]["valid"])
        self.assertIn("provider resolution is missing or inconsistent", manifest["validity"]["invalid_reasons"])
        self.assertTrue((root / "pilot-example" / "manifest.json").exists())

    def test_authorization_redaction_consumes_bearer_credential(self):
        value = redact("request failed: Authorization: Bearer unknown-token; retrying")
        self.assertEqual(value, "request failed: Authorization: [redacted]; retrying")
        self.assertNotIn("unknown-token", value)

    def test_error_redaction_removes_provider_user_id(self):
        value = redact('request failed: user_id":"user_private-value"}')
        self.assertNotIn("user_private-value", value)
        self.assertIn("user_id=[redacted]", value)

    def test_process_interrupts_are_not_serialized_as_client_errors(self):
        def interrupt(_messages):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_collection(interrupt)

    def test_builtin_evidence_adapter_preserves_observed_usage_cost_and_identity(self):
        class FakeClient:
            def complete_response(self, _messages, *, routing_metadata=False):
                self.routing_metadata = routing_metadata
                return {
                    "model": "served/model-v2",
                    "choices": [{"message": {"content": '{"action":"take","target":"lamp"}'}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "cost": 0.0012},
                    "openrouter_metadata": {
                        "endpoints": {"available": [
                            {"model": "served/model-v2", "provider": "Observed Host", "selected": True},
                            {"model": "served/model-v2", "provider": "Other Host", "selected": False},
                        ]}
                    },
                }

            @staticmethod
            def response_text(data):
                return data["choices"][0]["message"]["content"]

        client = FakeClient()
        completion = evidence_completion(client)([])
        self.assertTrue(client.routing_metadata)
        self.assertEqual(completion.text, '{"action":"take","target":"lamp"}')
        self.assertEqual(completion.model, "served/model-v2")
        self.assertEqual(completion.provider, "Observed Host")
        self.assertEqual(completion.usage["total_tokens"], 25)
        self.assertEqual(completion.cost, 0.0012)


if __name__ == "__main__":
    unittest.main()
