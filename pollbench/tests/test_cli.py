import json
import tempfile
import unittest
from pathlib import Path

from pollbench.__main__ import main


class CliTest(unittest.TestCase):
    def test_simulate_then_rate_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path, out = Path(d, "syn.csv"), Path(d, "ratings.json")
            main(["simulate", "--models", "40", "--benchmarks", "10", "--out", str(csv_path)])
            main(["rate", str(csv_path), "--out", str(out), "--samples", "60", "--warmup", "60", "--no-heldout"])
            got = json.loads(out.read_text())
        self.assertEqual(len(got["data_digest"]), 64)
        self.assertEqual(len(got["benchmarks"]), got["n_benchmarks"])
        self.assertIn("rhat_max", got["diagnostics"])


if __name__ == "__main__":
    unittest.main()
