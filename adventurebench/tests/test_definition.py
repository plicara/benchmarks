"""Checks that the published benchmark definition remains frozen."""
import hashlib
import unittest
from pathlib import Path

from adventure_bench.runner import SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]


class DefinitionTest(unittest.TestCase):
    def test_dataset_hash_is_frozen(self):
        dataset = ROOT / "adventure_bench" / "data" / "cases.jsonl"
        self.assertEqual(
            hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "f38e4650ecbe833b7f719744be2058f538fef0a27add1183e9d5ab9a96fea715",
        )

    def test_system_prompt_hash_is_frozen(self):
        self.assertEqual(
            hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "a5b12544f69942f57f4e8e81c21071b5bbb219436570ff63aa36e7b82994e786",
        )


if __name__ == "__main__":
    unittest.main()
