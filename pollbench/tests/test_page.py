import json
import unittest
from pathlib import Path

from pollbench.page import render

RESULT = Path(__file__).resolve().parents[1] / "results" / "epoch-2026-10-10.json"


class PageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(RESULT.read_text())

    def test_every_benchmark_appears_in_the_table_and_the_figure(self):
        html = render(self.data, placeholder="draft")
        for r in self.data["benchmarks"]:
            self.assertIn(f'<th scope="row" class="mono">{r["benchmark"]}</th>', html.replace("&amp;", "&"))
        self.assertGreaterEqual(html.count('class="rf-dot'), len(self.data["benchmarks"]))  # forest, plus the skill plot

    def test_title_and_terms_are_explained_before_the_ratings(self):
        html = render(self.data)
        self.assertIn("<h1>A pollster-style rating for AI benchmarks</h1>", html)
        for term in ("<strong>Reliability</strong>", "<strong>90% credible interval</strong>"):
            self.assertLess(html.index(term), html.index('id="ratings-heading"'))
        self.assertIn("<strong>held-out skill</strong>", html)
        self.assertIn("RELIABILITY: SHARE OF THE SPREAD BETWEEN MODELS THAT IS CAPABILITY", html)
        self.assertIn('<ul class="legend">', html)

    def test_every_reference_is_cited_and_every_citation_resolves(self):
        from pollbench.page import REFERENCES
        html = render(self.data)
        for i in range(1, len(REFERENCES) + 1):
            self.assertIn(f'id="fnref:{i}"', html)
            self.assertIn(f'<li id="fn:{i}">', html)
        self.assertIn("A Rosetta Stone for AI Benchmarks", html)

    def test_full_table_is_collapsed_and_every_figure_has_a_phone_layout(self):
        html = render(self.data)
        self.assertIn('<details class="full-table">', html)
        self.assertEqual(html.count('<svg class="wide"'), html.count('<svg class="compact"'))
        self.assertEqual(html.count('<svg class="wide"'), 4)

    def test_every_table_can_be_scrolled_from_the_keyboard(self):
        html = render(self.data)
        self.assertEqual(html.count('class="table-wrap"'), 2)
        self.assertEqual(html.count('class="table-wrap" tabindex="0" role="region" aria-label="'), 2)
        self.assertIn(".grade-table { max-width: var(--pb-measure); min-width: 0; }", html)

    def test_real_score_panels_need_the_curves_file(self):
        curves = json.loads((RESULT.parent / "epoch-2026-10-10-curves.json").read_text())
        self.assertEqual(curves["data_digest"], self.data["data_digest"])
        html = render(self.data, curves=curves)
        self.assertEqual(html.count('<div class="panel">'), 4)
        self.assertGreater(html.count('class="rf-pt"'), 300)
        self.assertNotIn('<div class="panel">', render(self.data))

    def test_placeholder_marks_the_page_noindex_and_published_does_not(self):
        self.assertIn('content="noindex"', render(self.data, placeholder="draft"))
        self.assertNotIn("noindex", render(self.data))

    def test_the_page_says_it_is_groups_and_credits_epoch(self):
        html = render(self.data)
        self.assertIn("groups, not a ranking", html)
        self.assertIn("Epoch AI", html)
        self.assertIn("CC BY 4.0", html)

    def test_exploratory_banner_follows_the_check_and_the_conditions(self):
        clean = json.loads(json.dumps(self.data))
        clean["conditions"] = {k: dict(v, holds=True) for k, v in clean["conditions"].items()}
        clean["second_factor_check"] = dict(clean["second_factor_check"], p_value=0.5)
        self.assertNotIn("Exploratory.", render(clean))
        rejects = json.loads(json.dumps(clean))
        rejects["second_factor_check"]["p_value"] = 0.01
        self.assertIn("Exploratory.", render(rejects))
        fails = json.loads(json.dumps(clean))
        next(iter(fails["conditions"].values()))["holds"] = False
        self.assertIn("Exploratory.", render(fails))

    def test_headroom_is_unknown_without_recent_models_and_d_is_not_called_failure(self):
        data = json.loads(json.dumps(self.data))
        for r in data["benchmarks"]:
            r["recent_ceiling_share"] = None
            r["saturated"] = False
        data["benchmarks"][0]["grade"] = "D"
        html = render(data)
        self.assertIn("unknown (no recent models)", html)
        self.assertNotIn("Does not clear", html)
        self.assertIn("Not shown to clear", html)


if __name__ == "__main__":
    unittest.main()
