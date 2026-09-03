"""Offline integration tests for deterministic release-set aggregation."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from adventure_bench.aggregate import ReleaseError, aggregate_release
from adventure_bench.collect import collect_run
from adventure_bench.rescore import rescore_run
from adventure_bench.runner import TransportError


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {
        "id": "case.1", "source": "original", "tags": ["mapping"], "input": "grab the lamp",
        "context": {"room": {"name": "Cell", "description": "A lamp."}, "exits": [],
                    "items": [{"id": "lamp", "name": "lamp"}], "carrying": []},
        "expect": [["take", "lamp"]],
    },
    {
        "id": "case.2", "source": "original", "tags": ["calibration"], "input": "sing a song",
        "context": {"room": {"name": "Cell", "description": "Empty."}, "exits": [], "items": [], "carrying": []},
        "expect": [["unclear", None]],
    },
]


class AggregateReleaseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runs, self.results, self.releases = self.root / "runs", self.root / "results", self.root / "releases"
        self.data = self.root / "cases.jsonl"
        self.data.write_text("".join(json.dumps(case) + "\n" for case in CASES), encoding="utf-8")

    def collect(self, run_id, model, responder, *, cases=CASES, data=None):
        collect_run(
            cases=deepcopy(cases), model=model, complete=responder, output_dir=self.runs, run_id=run_id,
            repetitions=2, requested_provider="test-provider", resolved_model=model,
            resolved_provider="test-provider", runtime="offline:test", base_url="https://api.example.test/v1",
            dataset_path=data or self.data, runner_root=ROOT,
        )
        return rescore_run(runs_dir=self.runs, results_dir=self.results, run_id=run_id, dataset_path=data or self.data)

    def aggregate(self, run_ids, *, check=False):
        return aggregate_release(runs_dir=self.runs, results_dir=self.results, releases_dir=self.releases,
                                 release_id="pilot-release", run_ids=run_ids, dataset_path=self.data, check=check)

    def test_two_audited_models_produce_deterministic_paired_comparisons(self):
        def model_one(messages):
            item = json.loads(messages[1]["content"])["input"]
            return '{"action":"take","target":"lamp"}' if item == "grab the lamp" else '{"action":"unclear","target":null}'

        def model_two(messages):
            item = json.loads(messages[1]["content"])["input"]
            return '{"action":"move","target":"north"}' if item == "grab the lamp" else '{"action":"unclear","target":null}'

        self.collect("pilot-model-one", "Example/Model One", model_one)
        self.collect("pilot-model-two", "Example/Model Two", model_two)
        artifact = self.aggregate(["pilot-model-one", "pilot-model-two"])
        output = self.releases / "pilot-release" / "release.json"
        first = output.read_bytes()
        self.aggregate(["pilot-model-one", "pilot-model-two"])
        self.assertEqual(first, output.read_bytes())
        self.aggregate(["pilot-model-one", "pilot-model-two"], check=True)
        self.assertEqual([row["model_slug"] for row in artifact["models"]], ["example-model-one", "example-model-two"])
        self.assertEqual(artifact["tag_matrix"]["mapping"]["example-model-one"]["score"], 1.0)
        self.assertLessEqual(artifact["collection_window"]["started_at"],
                             artifact["collection_window"]["completed_at"])
        comparison = artifact["bootstrap"]["paired_score_difference_intervals"][0]
        self.assertEqual(comparison["estimate"], 0.5)
        self.assertEqual(comparison["model_a"], "example-model-one")
        self.assertGreater(comparison["interval"][1], comparison["interval"][0])

    def test_invalid_duplicate_and_incompatible_runs_are_rejected(self):
        valid = lambda _messages: '{"action":"unclear","target":null}'
        broken = lambda _messages: (_ for _ in ()).throw(TransportError("offline"))
        self.collect("invalid-model", "Example/Invalid", broken)
        self.collect("valid-model", "Example/Valid", valid)
        with self.assertRaisesRegex(ReleaseError, "is invalid"):
            self.aggregate(["invalid-model", "valid-model"])

        self.collect("same-model-a", "Example/Same", valid)
        self.collect("same-model-b", "Example/Same", valid)
        with self.assertRaisesRegex(ReleaseError, "slugs must be unique"):
            self.aggregate(["same-model-a", "same-model-b"])

        self.collect("coverage-a", "Example/Coverage A", valid)
        self.collect("coverage-b", "Example/Coverage B", valid, cases=[CASES[0]])
        with self.assertRaisesRegex(ReleaseError, "identical case/repetition coverage"):
            self.aggregate(["coverage-a", "coverage-b"])

    def test_incompatible_dataset_hash_cannot_enter_a_release_set(self):
        valid = lambda _messages: '{"action":"unclear","target":null}'
        other_data = self.root / "other-cases.jsonl"
        other_data.write_text(self.data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.collect("hash-a", "Example/Hash A", valid)
        self.collect("hash-b", "Example/Hash B", valid, data=other_data)
        with self.assertRaisesRegex(ReleaseError, "cannot be revalidated"):
            self.aggregate(["hash-a", "hash-b"])


if __name__ == "__main__":
    unittest.main()
