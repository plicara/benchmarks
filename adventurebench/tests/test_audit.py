"""Repository-wide audit tests using only local synthetic evidence fixtures."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from adventure_bench.aggregate import aggregate_release
from adventure_bench.audit import AuditError, audit_repository
from adventure_bench.collect import collect_run
from adventure_bench.render_site import render_html
from adventure_bench.rescore import rescore_run


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {
        "id": "case.1", "source": "original", "tags": ["mapping"], "input": "grab lamp",
        "context": {"room": {"name": "Cell", "description": "A lamp."}, "exits": [],
                    "items": [{"id": "lamp", "name": "lamp"}], "carrying": []},
        "expect": [["take", "lamp"]],
    },
    {
        "id": "case.2", "source": "original", "tags": ["calibration"], "input": "sing",
        "context": {"room": {"name": "Cell", "description": "Empty."}, "exits": [], "items": [], "carrying": []},
        "expect": [["unclear", None]],
    },
]


class RepositoryAuditTest(unittest.TestCase):
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

    def _audit(self):
        return audit_repository(
            runs_dir=self.runs, results_dir=self.results, releases_dir=self.releases,
            dataset_path=self.data, site_dir=self.site,
        )

    def _seed(self, *, site: bool = True):
        def responder(messages):
            command = json.loads(messages[1]["content"])["input"]
            return '{"action":"take","target":"lamp"}' if command == "grab lamp" else '{"action":"unclear","target":null}'

        for run_id, model in (("model-a", "Model A"), ("model-b", "Model B")):
            collect_run(
                cases=deepcopy(CASES), model=model, complete=responder, output_dir=self.runs, run_id=run_id,
                repetitions=2, requested_provider="fixture", resolved_model=model,
                resolved_provider="fixture", runtime="test", base_url="https://api.example.test/v1",
                dataset_path=self.data, runner_root=ROOT,
            )
            rescore_run(runs_dir=self.runs, results_dir=self.results, run_id=run_id, dataset_path=self.data)
        artifact = aggregate_release(
            runs_dir=self.runs, results_dir=self.results, releases_dir=self.releases,
            release_id="audit-pilot", run_ids=["model-a", "model-b"], dataset_path=self.data,
        )
        if site:
            page = self.site / "benchmarks" / "adventurebench" / "index.html"
            page.parent.mkdir(parents=True)
            page.write_text(render_html(artifact), encoding="utf-8")

    def _snapshot(self) -> dict[str, bytes]:
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def test_empty_pre_release_repository_passes(self):
        report = self._audit()
        self.assertEqual(report.run_ids, ())
        self.assertEqual(report.release_ids, ())
        self.assertIsNone(report.site_release_id)

    def test_clean_repository_is_fully_checked_without_writing(self):
        self._seed()
        before = self._snapshot()
        report = self._audit()
        self.assertEqual(report.run_ids, ("model-a", "model-b"))
        self.assertEqual(report.release_ids, ("audit-pilot",))
        self.assertEqual(report.site_release_id, "audit-pilot")
        self.assertEqual(self._snapshot(), before)

    def test_tampered_raw_evidence_is_rejected(self):
        self._seed()
        evidence = self.runs / "model-a" / "responses" / "model-a.jsonl"
        evidence.write_text(evidence.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
        with self.assertRaisesRegex(AuditError, "offline rescore"):
            self._audit()

    def test_derived_result_drift_is_rejected(self):
        self._seed()
        summary = self.results / "model-a" / "summary.json"
        summary.write_text(summary.read_text(encoding="utf-8") + "drift", encoding="utf-8")
        with self.assertRaisesRegex(AuditError, "derived results drift"):
            self._audit()

    def test_release_drift_is_rejected(self):
        self._seed()
        release = self.releases / "audit-pilot" / "release.json"
        release.write_text(
            release.read_text(encoding="utf-8").replace('"score": 1.0', '"score": 0.0', 1),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(AuditError, "canonical aggregate"):
            self._audit()

    def test_site_drift_is_rejected(self):
        self._seed()
        page = self.site / "benchmarks" / "adventurebench" / "index.html"
        page.write_text(page.read_text(encoding="utf-8") + "drift", encoding="utf-8")
        with self.assertRaisesRegex(AuditError, "does not match exactly one"):
            self._audit()

    def test_missing_and_orphan_derived_artifacts_are_rejected(self):
        self._seed(site=False)
        shutil.rmtree(self.results / "model-a")
        with self.assertRaisesRegex(AuditError, "missing derived results"):
            self._audit()
        rescore_run(runs_dir=self.runs, results_dir=self.results, run_id="model-a", dataset_path=self.data)
        (self.results / "orphan").mkdir()
        with self.assertRaisesRegex(AuditError, "orphan derived results"):
            self._audit()

    def test_legacy_is_excluded_but_unexpected_run_artifacts_are_rejected(self):
        self._seed(site=False)
        legacy = self.results / "legacy"
        legacy.mkdir()
        (legacy / "historical.json").write_text("not an audited result\n", encoding="utf-8")
        self._audit()
        (self.runs / "model-a" / "notes.txt").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(AuditError, "run has unexpected artifacts"):
            self._audit()

    def test_orphan_site_and_unexpected_release_artifacts_are_rejected(self):
        self._seed()
        (self.releases / "audit-pilot" / "notes.txt").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(AuditError, "release has unexpected artifacts"):
            self._audit()
        (self.releases / "audit-pilot" / "notes.txt").unlink()
        shutil.rmtree(self.releases)
        with self.assertRaisesRegex(AuditError, "does not match exactly one"):
            self._audit()

    def test_site_page_cannot_escape_through_a_symlinked_parent(self):
        outside = self.root / "outside"
        page = outside / "adventurebench" / "index.html"
        page.parent.mkdir(parents=True)
        page.write_text("outside\n", encoding="utf-8")
        self.site.mkdir()
        (self.site / "benchmarks").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(AuditError, "outside the site directory"):
            self._audit()


if __name__ == "__main__":
    unittest.main()
