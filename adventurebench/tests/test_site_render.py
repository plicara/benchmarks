"""End-to-end checks for the audited static Adventure Bench page."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

from adventure_bench.aggregate import aggregate_release
from adventure_bench.collect import Completion, collect_run
from adventure_bench.render_site import SiteRenderError, render_site
from adventure_bench.rescore import rescore_run


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


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.tags.append((tag, {key: value or "" for key, value in attrs}))

    def handle_data(self, data: str):
        self.text.append(data)


class SiteRenderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runs = self.root / "runs"
        self.results = self.root / "results"
        self.releases = self.root / "releases"
        self.site = self.root / "site"
        self.data = self.root / "cases.jsonl"
        self.data.write_text("".join(json.dumps(case) + "\n" for case in CASES), encoding="utf-8")

    def _collect(self, run_id: str, model: str, provider: str, runtime: str, responder):
        collect_run(
            cases=deepcopy(CASES), model=model, complete=responder, output_dir=self.runs, run_id=run_id,
            repetitions=2, requested_provider=provider, resolved_model=model,
            resolved_provider=provider, runtime=runtime, base_url="https://api.example.test/v1",
            dataset_path=self.data, runner_root=ROOT,
        )
        rescore_run(runs_dir=self.runs, results_dir=self.results, run_id=run_id, dataset_path=self.data)

    def _release(self):
        def zebra(messages):
            command = json.loads(messages[1]["content"])["input"]
            if command == "grab the lamp":
                return Completion('{"action":"take","target":"lamp"}', cost=0.001)
            return Completion('{"action":"unclear","target":null}', cost=0.001)

        def alpha(messages):
            command = json.loads(messages[1]["content"])["input"]
            return Completion("not JSON", cost=0.002) if command == "grab the lamp" else Completion('{"action":"move","target":"north"}', cost=0.002)

        self._collect(
            "zebra-model", "Zebra <unsafe>", "provider<zebra>&", "offline:<zebra>&",
            zebra,
        )
        self._collect(
            "alpha-model", "Alpha & <unsafe>", "provider<alpha>&", "offline:<alpha>&",
            alpha,
        )
        return aggregate_release(
            runs_dir=self.runs, results_dir=self.results, releases_dir=self.releases,
            release_id="render-pilot", run_ids=["zebra-model", "alpha-model"], dataset_path=self.data,
        )

    def _render(self, *, check: bool = False):
        return render_site(
            release_id="render-pilot", runs_dir=self.runs, results_dir=self.results,
            releases_dir=self.releases, dataset_path=self.data, site_dir=self.site, check=check,
        )

    def test_render_is_accessible_escaped_alphabetical_and_parity_checked(self):
        self._release()
        output = self._render()
        page = output.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(page)
        parser.close()
        self.assertIn(("body", {"data-scheme": "technical"}), parser.tags)
        self.assertIn(("main", {"class": "wrap results", "id": "main"}), parser.tags)
        self.assertIn("Alpha &amp; &lt;unsafe&gt;", page)
        self.assertNotIn("Alpha & <unsafe>", page)
        self.assertIn("provider&lt;alpha&gt;&amp;", page)
        self.assertLess(page.rindex("Alpha &amp; &lt;unsafe&gt;"), page.rindex("Zebra &lt;unsafe&gt;"))
        self.assertIn("4 / 0", page)
        self.assertIn("+1 / &minus;0", page)
        self.assertIn("These results are not a ranking.", page)
        self.assertIn("The score&ndash;cost frontier", page)
        self.assertIn("Recorded full-run cost", page)
        self.assertIn('<text class="pareto-label"', page)
        self.assertIn("Zebra &lt;unsafe&gt;</text>", page)
        self.assertNotIn("Alpha &amp; &lt;unsafe&gt;</text>", page)
        self.assertIn("exact values for every point appear in the table below", page)
        self.assertIn("unresolved comparisons are unresolved.", page)
        hrefs = {attrs["href"] for tag, attrs in parser.tags if tag == "a" and "href" in attrs}
        self.assertIn("https://github.com/plicara/benchmarks/tree/main/adventurebench", hrefs)
        self.assertIn("https://github.com/plicara/benchmarks/tree/main/adventurebench/runs", hrefs)
        self.assertIn("https://github.com/plicara/benchmarks/blob/main/adventurebench/DATASET_CARD.md", hrefs)
        self.assertIn("https://github.com/plicara/benchmarks/tree/main/adventurebench#auditable-result-collection", hrefs)
        self.assertNotIn('<p class="note mono">Source:', page)
        self.assertIn("make site-check RELEASE_ID=render-pilot", page)
        self.assertIn("collected on ", page)
        self.assertTrue(all("</p>" in line for line in page.splitlines() if "<p " in line))
        self._render(check=True)
        output.write_text(page + "drift", encoding="utf-8")
        with self.assertRaisesRegex(SiteRenderError, "site render drift"):
            self._render(check=True)

    def test_edited_or_malformed_release_artifacts_cannot_render(self):
        self._release()
        output = self._render()
        original_page = output.read_text(encoding="utf-8")
        release_path = self.releases / "render-pilot" / "release.json"
        original_release = release_path.read_text(encoding="utf-8")
        release_path.write_text(original_release.replace('"score": 1.0', '"score": 0.0', 1), encoding="utf-8")
        with self.assertRaisesRegex(SiteRenderError, "release artifact drift"):
            self._render()
        self.assertEqual(output.read_text(encoding="utf-8"), original_page)
        release_path.write_text(original_release.replace('"release_id": "render-pilot"', '"release_id": "edited-pilot"'), encoding="utf-8")
        with self.assertRaisesRegex(SiteRenderError, "release artifact ID"):
            self._render()
        self.assertEqual(output.read_text(encoding="utf-8"), original_page)
        release_path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(SiteRenderError, "cannot read release artifact"):
            self._render()
        self.assertEqual(output.read_text(encoding="utf-8"), original_page)


if __name__ == "__main__":
    unittest.main()
