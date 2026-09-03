"""Offline replay tests for auditable collection evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from adventure_bench.collect import collect_run, safe_model_slug
from adventure_bench.rescore import EvidenceError, rescore_run
from adventure_bench.runner import TransportError


ROOT = Path(__file__).resolve().parents[1]
CASE = {
    "id": "case.1", "source": "original", "tags": ["synonym"], "input": "grab the light",
    "context": {
        "room": {"name": "Cell", "description": "A cell with a lamp."},
        "exits": ["north"], "items": [{"id": "brass_lantern", "name": "brass lamp"}], "carrying": [],
    },
    "expect": [["take", "brass_lantern"]],
}


class RescoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runs = self.root / "runs"
        self.results = self.root / "results"
        self.data = self.root / "cases.jsonl"
        self.cases = [deepcopy(CASE)]
        self.data.write_text("".join(json.dumps(case) + "\n" for case in self.cases), encoding="utf-8")
        self.run_id = "pilot-example"
        self.slug = safe_model_slug("Example/Model 1")

    def collect(self, complete, *, repetitions=1):
        return collect_run(
            cases=deepcopy(self.cases), model="Example/Model 1", complete=complete,
            output_dir=self.runs, run_id=self.run_id, repetitions=repetitions,
            requested_provider="pinned-host", resolved_model="served-model",
            resolved_provider="observed-host", runtime="openai-compatible:test",
            base_url="https://api.example.test/v1", dataset_path=self.data, runner_root=ROOT,
        )

    @property
    def manifest_path(self):
        return self.runs / self.run_id / "manifest.json"

    @property
    def evidence_path(self):
        return self.runs / self.run_id / "responses" / f"{self.slug}.jsonl"

    def record(self):
        return json.loads(self.evidence_path.read_text().splitlines()[0])

    def replace_record(self, record):
        self.evidence_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    def rescore(self, *, check=False):
        return rescore_run(runs_dir=self.runs, results_dir=self.results, run_id=self.run_id, dataset_path=self.data, check=check)

    def test_clean_replay_writes_deterministic_result_and_check_passes(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}', repetitions=2)
        first = self.rescore()
        first_bytes = {path.name: path.read_bytes() for path in (self.results / self.run_id).glob("*.json")}
        self.rescore(check=True)
        second = self.rescore()
        second_bytes = {path.name: path.read_bytes() for path in (self.results / self.run_id).glob("*.json")}
        self.assertEqual(first_bytes, second_bytes)
        result = first["models"][self.slug]
        self.assertEqual(result["overall"], {"passed": 2, "total": 2, "score": 1.0})
        self.assertEqual(result["repetitions"][0]["by_tag"]["synonym"], {"passed": 1, "total": 1, "score": 1.0})
        self.assertEqual(result["case_results"], [{
            "case_id": "case.1", "repetition": 1, "tags": ["synonym"],
            "outcome": {"action": "take", "target": "brass_lantern"}, "passed": True,
        }, {
            "case_id": "case.1", "repetition": 2, "tags": ["synonym"],
            "outcome": {"action": "take", "target": "brass_lantern"}, "passed": True,
        }])
        self.assertEqual(len(result["evidence"]["responses_sha256"]), 64)
        self.assertIsNone(result["evidence"]["recorded_cost_usd"])
        self.assertEqual(result["provenance"]["collection_window"]["started_at"],
                         json.loads(self.manifest_path.read_text(encoding="utf-8"))["started_at"])
        self.assertEqual(second["summary"]["models"][0]["overall"], result["overall"])

    def test_persistent_malformed_output_replays_to_unclear_not_transport(self):
        self.collect(lambda _messages: "not json")
        result = self.rescore()["models"][self.slug]
        self.assertTrue(result["validity"]["valid"])
        self.assertEqual(result["overall"], {"passed": 0, "total": 1, "score": 0.0})
        self.assertEqual(result["failures"], {"transport": 0, "parsing": 2})

    def test_transport_failure_is_retained_and_invalidates_run(self):
        self.collect(lambda _messages: (_ for _ in ()).throw(TransportError("offline")))
        output = self.rescore()
        result = output["models"][self.slug]
        self.assertFalse(result["validity"]["valid"])
        self.assertEqual(result["failures"], {"transport": 1, "parsing": 0})
        self.assertFalse(output["summary"]["validity"]["valid"])

    def test_recorded_score_fields_cannot_be_tampered(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        record = self.record()
        record["passed"] = False
        self.replace_record(record)
        with self.assertRaisesRegex(EvidenceError, "passed does not match"):
            self.rescore()

    def test_raw_completion_tamper_is_detected_by_replay(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        record = self.record()
        record["attempts"][0]["raw_completion"] = '{"action":"move", "target":"north"}'
        self.replace_record(record)
        with self.assertRaisesRegex(EvidenceError, "parsed does not match"):
            self.rescore()

    def test_duplicate_and_missing_case_repetition_records_are_rejected(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        original = self.evidence_path.read_text(encoding="utf-8")
        self.evidence_path.write_text(original + original, encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "record counts|duplicate"):
            self.rescore()
        self.evidence_path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "record counts|coverage"):
            self.rescore()

    def test_non_utf8_evidence_is_rejected_as_evidence_error(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        self.evidence_path.write_bytes(b"\xff\xfe\n")
        with self.assertRaisesRegex(EvidenceError, "not valid UTF-8"):
            self.rescore()

    def test_incomplete_and_multi_model_manifests_are_rejected(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["completed_at"] = None
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "manifest is incomplete"):
            self.rescore()

        manifest = self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        manifest["models"]["another-model"] = deepcopy(manifest["models"][self.slug])
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "exactly one model"):
            self.rescore()

    def test_invalid_or_reversed_collection_window_is_rejected(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["started_at"] = "not-a-date"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "UTC date-time"):
            self.rescore()

        manifest["started_at"] = "2026-09-02T00:00:00Z"
        manifest["completed_at"] = "2026-09-01T00:00:00Z"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "precedes"):
            self.rescore()

    def test_wrong_dataset_prompt_and_selection_hashes_are_rejected(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        original = self.manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(original)
        manifest["benchmark"]["dataset_sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "dataset hash"):
            self.rescore()

        manifest = json.loads(original)
        manifest["benchmark"]["prompt_sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "prompt hash"):
            self.rescore()

        manifest = json.loads(original)
        manifest["selected_case_ids_sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "selected-case hash"):
            self.rescore()

    def test_replays_explicit_non_dataset_case_order(self):
        other = deepcopy(CASE)
        other["id"] = "case.2"
        self.cases = [deepcopy(CASE), other]
        self.data.write_text("".join(json.dumps(case) + "\n" for case in self.cases), encoding="utf-8")
        collect_run(
            cases=[other, self.cases[0]], model="Example/Model 1",
            complete=lambda _messages: '{"action":"take", "target":"lamp"}', output_dir=self.runs,
            run_id=self.run_id, repetitions=1, requested_provider="pinned-host",
            resolved_model="served-model", resolved_provider="observed-host",
            runtime="openai-compatible:test", base_url="https://api.example.test/v1",
            dataset_path=self.data, runner_root=ROOT,
        )
        result = self.rescore()
        self.assertEqual(result["models"][self.slug]["overall"]["total"], 2)

    def test_check_rejects_derived_drift_without_rewriting(self):
        self.collect(lambda _messages: '{"action":"take", "target":"lamp"}')
        self.rescore()
        derived = self.results / self.run_id / f"{self.slug}.json"
        derived.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "derived results drift"):
            self.rescore(check=True)
        self.assertEqual(derived.read_text(encoding="utf-8"), "{}\n")


if __name__ == "__main__":
    unittest.main()
