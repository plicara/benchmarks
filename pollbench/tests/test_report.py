import json
import unittest
from pathlib import Path

from pollbench.report import render

RESULT = Path(__file__).resolve().parents[1] / "results" / "epoch-2026-10-10.json"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "results-epoch-2026-10-10.md"
CAL = Path(__file__).resolve().parents[1] / "results" / "second-factor-calibration.json"


class ReportTest(unittest.TestCase):
    def test_committed_report_matches_the_committed_results(self):
        d = json.loads(RESULT.read_text())
        if "Is capability one number" in REPORT.read_text() and "On simulated tables the check" in REPORT.read_text():
            d["second_factor_calibration"] = json.loads(CAL.read_text())
        self.assertEqual(render(d), REPORT.read_text())

    def test_consistency_checks_are_labelled_and_epoch_form_leads(self):
        text = render(json.loads(RESULT.read_text()))
        self.assertIn("consistency checks", text)
        self.assertLess(text.index("eci_ls"), text.index("mean_rank"))


if __name__ == "__main__":
    unittest.main()
