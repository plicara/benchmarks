"""Unit tests for the Jev runtime branches auditors flagged as untested.

Offline only: no provider calls. Covers the outcome mapping (all actions,
gating boundaries, malformed shapes incl. the non-numeric-confidence crash),
the replay helper, spec-hash stability, cost math, and the frozen-threshold
refusal in collect_run_jev.
"""
from __future__ import annotations

import unittest
import json
import tempfile
from copy import deepcopy
from pathlib import Path

from adventure_bench import jev
from adventure_bench.collect import safe_model_slug, sha256_text
from adventure_bench.jev import (
    JEV_SPEC_SHA,
    JEV_SPEC_SHA_V1,
    answers_outcome,
    answers_outcome_v1,
    jev_spec_sha,
    record_cost_usd,
    replay_jev_outcome,
    threshold_for,
)
from adventure_bench.rescore import EvidenceError, rescore_run
from adventure_bench.runner import SYSTEM_PROMPT


def answers(action, target="none", confidence=1.0):
    return {"action": {"choice": action, "confidence": confidence},
            "target": {"choice": target}}


CASE = {
    "id": "case.1", "source": "original", "tags": ["mapping"], "input": "take lamp",
    "context": {"room": {"name": "Cell", "description": "A lamp."}, "exits": [],
                "items": [{"id": "lamp", "name": "lamp"}], "carrying": []},
    "expect": [["take", "lamp"]],
}


class FakeJevClient:
    timeout = 1.0

    def evaluate(self, _state, _questions):
        return {"answers": answers("take", "lamp", 1.0), "usage": {"input_tokens": 1},
                "model": "jev-1.13.0"}, 0.001


class OutcomeMappingTest(unittest.TestCase):
    def test_all_actions_map(self):
        self.assertEqual(answers_outcome(answers("take", "lamp"), 0.9)[0], ("take", "lamp"))
        self.assertEqual(answers_outcome(answers("drop", "lamp"), 0.9)[0], ("drop", "lamp"))
        self.assertEqual(answers_outcome(answers("examine", "lamp"), 0.9)[0], ("examine", "lamp"))
        self.assertEqual(answers_outcome(answers("use", "lamp"), 0.9)[0], ("use", "lamp"))
        self.assertEqual(answers_outcome(answers("look"), 0.9)[0], ("look", None))
        self.assertEqual(answers_outcome(answers("inventory"), 0.9)[0], ("inventory", None))
        self.assertEqual(answers_outcome(answers("unclear"), 0.9)[0], ("unclear", None))

    def test_move_normalizes_directions(self):
        self.assertEqual(answers_outcome(answers("move", "N"), 0.9)[0], ("move", "north"))

    def test_none_target_is_unclear_under_v2(self):
        self.assertEqual(answers_outcome(answers("take", "none"), 0.9)[0], ("unclear", None))

    def test_flip_boundaries(self):
        self.assertEqual(answers_outcome(answers("take", "lamp", 0.8999), 0.9)[0], ("unclear", None))
        self.assertEqual(answers_outcome(answers("take", "lamp", 0.9), 0.9)[0], ("take", "lamp"))
        self.assertEqual(answers_outcome(answers("take", "lamp", 0.9001), 0.9)[0], ("take", "lamp"))
        # no-target answers are never gated, even at zero confidence
        self.assertEqual(answers_outcome(answers("look", None, 0.0), 0.9)[0], ("look", None))
        self.assertEqual(answers_outcome(answers("unclear", None, 0.0), 0.9)[0], ("unclear", None))

    def test_unknown_action_is_parsing_path(self):
        self.assertIsNone(answers_outcome(answers("open", "door"), 0.9))

    def test_malformed_shapes(self):
        self.assertIsNone(answers_outcome({}, 0.9))
        self.assertIsNone(answers_outcome({"action": "broken"}, 0.9))
        self.assertIsNone(answers_outcome({"action": {"choice": "take"}, "target": "broken"}, 0.9))
        self.assertIsNone(answers_outcome({"action": {"choice": "take", "confidence": 1.0},
                                           "target": {"choice": ["not", "a", "string"]}}, 0.9))

    def test_non_numeric_confidence_gates_shut(self):
        # B1 crash input: must not raise, must flip to unclear
        mapped = answers_outcome(answers("take", "lamp", "0.95"), 0.9)
        assert mapped is not None
        self.assertEqual(mapped[0], ("unclear", None))
        self.assertTrue(mapped[1])

    def test_flip_metadata(self):
        outcome, flipped, raw = answers_outcome(answers("take", "lamp", 0.5), 0.9) or (None, None, None)
        self.assertEqual(outcome, ("unclear", None))
        self.assertTrue(flipped)
        self.assertEqual(raw, ("take", "lamp"))


class ReplayTest(unittest.TestCase):
    def test_round_trip(self):
        raw = json.dumps({"answers": answers("take", "lamp", 0.95)})
        self.assertEqual(replay_jev_outcome(raw, CASE, JEV_SPEC_SHA), ("take", "lamp"))
        self.assertEqual(replay_jev_outcome(raw, CASE, JEV_SPEC_SHA_V1), ("take", "lamp"))

    def test_replay_dispatches_mapping_version(self):
        raw = json.dumps({"answers": answers("take", "none", 0.95)})
        self.assertEqual(replay_jev_outcome(raw, CASE, JEV_SPEC_SHA), ("unclear", None))
        self.assertEqual(replay_jev_outcome(raw, CASE, JEV_SPEC_SHA_V1), ("take", "none"))

    def test_malformed_raw(self):
        self.assertIsNone(replay_jev_outcome("not json", CASE, JEV_SPEC_SHA))
        self.assertIsNone(replay_jev_outcome("[1,2]", CASE, JEV_SPEC_SHA))
        self.assertIsNone(replay_jev_outcome("{}", CASE, JEV_SPEC_SHA))
        self.assertIsNone(replay_jev_outcome('{"answers": {}}', CASE, JEV_SPEC_SHA))


class SpecAndCostTest(unittest.TestCase):
    def test_spec_hash_stable(self):
        self.assertEqual(JEV_SPEC_SHA, jev_spec_sha())
        self.assertNotEqual(JEV_SPEC_SHA, JEV_SPEC_SHA_V1)

    def test_threshold_pair_and_v1_mapping_are_frozen(self):
        self.assertEqual(threshold_for({"tags": ["synonym"]}), 0.75)
        self.assertEqual(threshold_for({"tags": ["out-of-vocab"]}), 0.9)
        self.assertEqual(answers_outcome_v1(answers("take", "none", 0.95), 0.9)[0],
                         ("take", "none"))

    def test_cost_math(self):
        self.assertAlmostEqual(record_cost_usd({"input_tokens": 1000, "output_tokens": 50}), 0.000042)
        self.assertIsNone(record_cost_usd(None))
        self.assertIsNone(record_cost_usd({}))
        self.assertIsNone(record_cost_usd({"input_tokens": -1}))
        self.assertIsNone(record_cost_usd({"input_tokens": True}))


class EvidenceContractTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runs = self.root / "runs"
        self.results = self.root / "results"
        self.data = self.root / "cases.jsonl"
        self.data.write_text(json.dumps(CASE) + "\n", encoding="utf-8")
        self.run_id = "jev-pilot"
        self._collect()

    @property
    def manifest_path(self):
        return self.runs / self.run_id / "manifest.json"

    @property
    def evidence_path(self):
        return self.runs / self.run_id / "responses" / f"{safe_model_slug('jev-1.13.0')}.jsonl"

    def _collect(self):
        jev.collect_run_jev(
            cases=[deepcopy(CASE)], model="jev-1.13.0", client=FakeJevClient(),
            output_dir=self.runs, run_id=self.run_id, dataset_path=self.data,
            runner_root=Path(__file__).resolve().parents[1],
        )

    def _manifest(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _record(self):
        return json.loads(self.evidence_path.read_text(encoding="utf-8"))

    def _write(self, manifest, record):
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def _rescore(self):
        return rescore_run(runs_dir=self.runs, results_dir=self.results, run_id=self.run_id,
                           dataset_path=self.data)

    def test_runtime_tracks_the_requested_model(self):
        manifest = self._manifest()
        self.assertEqual(manifest["configuration"]["resolved"]["runtime"],
                         "typesafe-systemone:jev-1.13.0")

    def test_rejects_manifest_and_request_prompt_mismatch(self):
        manifest, record = self._manifest(), self._record()
        manifest["benchmark"]["prompt_sha256"] = sha256_text(SYSTEM_PROMPT)
        self._write(manifest, record)
        with self.assertRaisesRegex(EvidenceError, "prompt hash.*configuration"):
            self._rescore()

    def test_rejects_a_non_frozen_jev_threshold(self):
        manifest, record = self._manifest(), self._record()
        manifest["configuration"]["jev"]["threshold"] = 0.0
        record["request"] = manifest["configuration"]
        self._write(manifest, record)
        with self.assertRaisesRegex(EvidenceError, "frozen threshold"):
            self._rescore()

    def test_rejects_a_non_frozen_mapping_threshold(self):
        manifest, record = self._manifest(), self._record()
        manifest["configuration"]["jev"]["mapping_threshold"] = 0.5
        record["request"] = manifest["configuration"]
        self._write(manifest, record)
        with self.assertRaisesRegex(EvidenceError, "frozen v2 spec"):
            self._rescore()

    def test_rejects_runtime_provenance_for_another_model(self):
        manifest, record = self._manifest(), self._record()
        manifest["configuration"]["requested"]["runtime"] = "typesafe-systemone:jev-latest"
        manifest["configuration"]["resolved"]["runtime"] = "typesafe-systemone:jev-latest"
        manifest["models"]["jev-1-13-0"]["requested"] = manifest["configuration"]["requested"]
        record["request"] = manifest["configuration"]
        self._write(manifest, record)
        with self.assertRaisesRegex(EvidenceError, "runtime provenance"):
            self._rescore()

    def test_rejects_a_retrying_jev_record(self):
        manifest, record = self._manifest(), self._record()
        first = record["attempts"][0]
        malformed = {**first, "raw_completion": '{"answers": {}}', "parsed": None,
                     "error": {"kind": "parsing", "type": "UnknownAction", "message": "bad"}}
        first["number"] = 2
        record["attempts"] = [malformed, first]
        self._write(manifest, record)
        with self.assertRaisesRegex(EvidenceError, "exactly one attempt"):
            self._rescore()


if __name__ == "__main__":
    unittest.main()
