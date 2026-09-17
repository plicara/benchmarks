"""Unit tests for the Jev runtime branches auditors flagged as untested.

Offline only: no provider calls. Covers the outcome mapping (all actions,
gating boundaries, malformed shapes incl. the non-numeric-confidence crash),
the replay helper, spec-hash stability, cost math, and the frozen-threshold
refusal in collect_run_jev.
"""
from __future__ import annotations

import unittest

from adventure_bench import jev
from adventure_bench.jev import (
    JEV_SPEC_SHA,
    answers_outcome,
    jev_spec_sha,
    record_cost_usd,
    replay_jev_outcome,
)


def answers(action, target="none", confidence=1.0):
    return {"action": {"choice": action, "confidence": confidence},
            "target": {"choice": target}}


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

    def test_none_target_passes_through(self):
        self.assertEqual(answers_outcome(answers("take", "none"), 0.9)[0], ("take", "none"))

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
        import json
        raw = json.dumps({"answers": answers("take", "lamp", 0.95)})
        self.assertEqual(replay_jev_outcome(raw, 0.9), ("take", "lamp"))

    def test_malformed_raw(self):
        self.assertIsNone(replay_jev_outcome("not json", 0.9))
        self.assertIsNone(replay_jev_outcome("[1,2]", 0.9))
        self.assertIsNone(replay_jev_outcome("{}", 0.9))
        self.assertIsNone(replay_jev_outcome('{"answers": {}}', 0.9))


class SpecAndCostTest(unittest.TestCase):
    def test_spec_hash_stable(self):
        self.assertEqual(JEV_SPEC_SHA, jev_spec_sha())
        self.assertEqual(JEV_SPEC_SHA, jev_spec_sha(0.9))

    def test_frozen_threshold_refusal(self):
        from pathlib import Path
        with self.assertRaises(ValueError):
            jev.collect_run_jev(cases=[], model="jev-latest", client=None,  # type: ignore[arg-type]
                                output_dir=Path("/tmp"), run_id="x", threshold=0.5)

    def test_cost_math(self):
        self.assertAlmostEqual(record_cost_usd({"input_tokens": 1000, "output_tokens": 50}), 0.000042)
        self.assertIsNone(record_cost_usd(None))
        self.assertIsNone(record_cost_usd({}))
        self.assertIsNone(record_cost_usd({"input_tokens": -1}))
        self.assertIsNone(record_cost_usd({"input_tokens": True}))


if __name__ == "__main__":
    unittest.main()
