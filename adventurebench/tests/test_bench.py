"""Offline tests: runner mechanics with a fake model, plus dataset lints.
No network. These are what CI runs."""
import unittest

from adventure_bench.runner import (
    TransportError, extract_json, load_cases, reply_to_outcome, resolve_target, run_case, user_message,
)
from adventure_bench.validate import validate

CASE = {
    "id": "t.1", "source": "original", "tags": ["synonym"], "input": "grab the light",
    "context": {
        "room": {"name": "Cell", "description": "A bare cell. A lantern hangs on a nail."},
        "exits": ["north"],
        "items": [{"id": "brass_lantern", "name": "brass lamp"}],
        "carrying": [{"id": "rusty_key", "name": "rusty key"}],
    },
    "expect": [["take", "brass_lantern"]],
}


def scripted(*replies):
    replies = list(replies)
    calls = []

    def complete(messages):
        calls.append(messages)
        return replies.pop(0)

    complete.calls = calls
    return complete


class RunnerTest(unittest.TestCase):
    def test_dataset_is_valid(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 200)
        self.assertEqual(validate(cases), [])

    def test_user_message_exposes_context_only(self):
        msg = user_message(CASE)
        self.assertIn("brass_lantern", msg)
        self.assertIn("rusty_key", msg)
        self.assertIn("grab the light", msg)

    def test_clean_reply(self):
        got, transport = run_case(CASE, scripted('{"action": "take", "target": "lantern"}'))
        self.assertEqual(got, ("take", "brass_lantern"))
        self.assertFalse(transport)

    def test_fenced_reply_and_direction_alias(self):
        got, _ = run_case(CASE, scripted('sure! ```json\n{"action": "move", "target": "n"}\n```'))
        self.assertEqual(got, ("move", "north"))

    def test_underscore_and_name_resolution(self):
        self.assertEqual(resolve_target(CASE, "brass_lamp"), "brass_lantern")
        self.assertEqual(resolve_target(CASE, "brass lamp"), "brass_lantern")
        self.assertEqual(resolve_target(CASE, "lamp"), "brass_lantern")
        self.assertEqual(resolve_target(CASE, "rusty key"), "rusty_key")
        self.assertEqual(resolve_target(CASE, "sword"), "sword")  # unresolved passes through

    def test_malformed_retries_then_scores(self):
        complete = scripted("the player clearly wants the lamp", '{"action": "take", "target": "brass_lantern"}')
        got, _ = run_case(CASE, complete)
        self.assertEqual(got, ("take", "brass_lantern"))
        self.assertEqual(len(complete.calls), 2)
        self.assertEqual(complete.calls[1][-2]["role"], "assistant")

    def test_persistent_garbage_scores_unclear(self):
        got, transport = run_case(CASE, scripted("garbage", "more garbage"))
        self.assertEqual(got, ("unclear", None))
        self.assertFalse(transport)

    def test_transport_error_is_flagged_not_scored(self):
        def broken(messages):
            raise TransportError("connection refused")

        got, transport = run_case(CASE, broken)
        self.assertTrue(transport)

    def test_unknown_action_is_malformed(self):
        self.assertIsNone(reply_to_outcome(CASE, {"action": "dance"}))
        self.assertIsNone(reply_to_outcome(CASE, {"action": "take", "target": ""}))

    def test_no_target_actions(self):
        self.assertEqual(reply_to_outcome(CASE, {"action": "look", "target": "whatever"}), ("look", None))
        self.assertEqual(reply_to_outcome(CASE, {"action": "unclear", "reason": "?"}), ("unclear", None))

    def test_extract_json_edge_cases(self):
        self.assertIsNone(extract_json(None))
        self.assertIsNone(extract_json(""))
        self.assertIsNone(extract_json("no json here"))
        self.assertIsNone(extract_json("[1, 2]"))
        self.assertEqual(extract_json('x {"a": {"b": 1}} y'), {"a": {"b": 1}})


if __name__ == "__main__":
    unittest.main()
