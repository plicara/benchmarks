import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pollbench.data import SCORE_EPS, build_dataset, load_epoch_csv


def rows(models, benches, f):
    return [{"model": m, "benchmark": b, "score": f(i, j)} for i, m in enumerate(models) for j, b in enumerate(benches)]


class BuildDatasetTest(unittest.TestCase):
    def test_minimums_hold_after_iterative_dropping(self):
        data = rows([f"m{i}" for i in range(8)], [f"b{j}" for j in range(6)], lambda i, j: 0.1 + 0.05 * (i + j) / 2)
        data.append({"model": "lonely", "benchmark": "b0", "score": 0.5})  # one benchmark only
        data.append({"model": "m0", "benchmark": "thin", "score": 0.5})  # a benchmark with one model
        ds = build_dataset(data, min_models_per_benchmark=5, min_benchmarks_per_model=4)
        self.assertNotIn("lonely", ds.models)
        self.assertNotIn("thin", ds.benchmarks)
        self.assertEqual(ds.dropped["models_below_min_benchmarks"], 1)
        self.assertEqual(ds.dropped["benchmarks_below_min_models"], 1)
        self.assertTrue((np.bincount(ds.bench_idx) >= 5).all())
        self.assertTrue((np.bincount(ds.model_idx) >= 4).all())

    def test_duplicate_cells_keep_the_maximum_and_are_counted(self):
        data = rows([f"m{i}" for i in range(6)], [f"b{j}" for j in range(5)], lambda i, j: 0.3)
        data += [{"model": "m0", "benchmark": "b0", "score": 0.7}, {"model": "m0", "benchmark": "b0", "score": 0.1}]
        ds = build_dataset(data)
        cell = (ds.model_idx == ds.models.index("m0")) & (ds.bench_idx == ds.benchmarks.index("b0"))
        self.assertAlmostEqual(float(ds.score[cell][0]), 0.7)
        self.assertEqual(ds.dropped["duplicate_cells_collapsed"], 2)

    def test_bad_scores_are_dropped_not_repaired(self):
        data = rows([f"m{i}" for i in range(6)], [f"b{j}" for j in range(5)], lambda i, j: 0.4)
        data += [{"model": "m1", "benchmark": "b1", "score": "n/a"}, {"model": "m2", "benchmark": "b2", "score": 1.5},
                 {"model": "m3", "benchmark": "b3", "score": float("nan")}]
        ds = build_dataset(data)
        self.assertEqual(ds.dropped["non_numeric_score"], 1)
        self.assertEqual(ds.dropped["out_of_range_score"], 2)
        self.assertEqual(ds.n_cells, 30)

    def test_extreme_scores_are_clipped_into_the_open_interval(self):
        data = rows([f"m{i}" for i in range(6)], [f"b{j}" for j in range(5)], lambda i, j: 0.0 if i == 0 else (1.0 if i == 1 else 0.5))
        ds = build_dataset(data)
        self.assertAlmostEqual(float(ds.score.min()), SCORE_EPS)
        self.assertAlmostEqual(float(ds.score.max()), 1 - SCORE_EPS)

    def test_digest_ignores_input_order_and_changes_with_content(self):
        data = rows([f"m{i}" for i in range(6)], [f"b{j}" for j in range(5)], lambda i, j: 0.1 * ((i + j) % 7) + 0.05)
        a = build_dataset(data).digest()
        b = build_dataset(list(reversed(data))).digest()
        self.assertEqual(a, b)
        data[0] = dict(data[0], score=0.99)
        self.assertNotEqual(a, build_dataset(data).digest())


class LoadCsvTest(unittest.TestCase):
    def test_reads_epoch_columns(self):
        header = ["model_id", "benchmark_id", "performance", "benchmark", "benchmark_release_date", "model", "model_version", "Model", "date", "source"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.csv"
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                for i in range(6):
                    for j in range(5):
                        w.writerow([f"m{i}", f"b{j}", 0.1 + 0.1 * ((i + j) % 5), f"Bench {j}", "2024-01-01", f"m{i}", "", f"Model {i}", "2024-06-01", "synthetic test file"])
            ds = load_epoch_csv(path)
        self.assertEqual(ds.n_cells, 30)
        self.assertIn("Model 0", ds.models)
        self.assertEqual(ds.model_date[0], "2024-06-01")

    def test_rejects_unexpected_columns(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.csv"
            path.write_text("a,b,c\n1,2,3\n")
            with self.assertRaises(ValueError):
                load_epoch_csv(path)


if __name__ == "__main__":
    unittest.main()


class DateAndDigestTest(unittest.TestCase):
    def _rows(self):
        return [{"model": f"m{i}", "benchmark": f"b{j}", "score": 0.1 + 0.01 * i + 0.02 * j,
                 "model_date": f"2025-01-{i + 1:02d}", "bench_date": "2024-01-01", "source": "x"}
                for i in range(8) for j in range(6)]

    def test_modal_date_does_not_depend_on_row_order(self):
        rows = self._rows()
        rows.append(dict(rows[0], benchmark="b1", model_date="2020-01-01"))  # a stray date for m0
        rows.append(dict(rows[0], benchmark="b2", model_date="2020-01-01"))
        a = build_dataset(rows).model_date[0]
        b = build_dataset(list(reversed(rows))).model_date[0]
        self.assertEqual(a, b)

    def test_digest_changes_with_dates_and_sources(self):
        rows = self._rows()
        base = build_dataset(rows).digest()
        moved = [dict(r, model_date="2026-01-01") if r["model"] == "m0" else r for r in rows]
        self.assertNotEqual(base, build_dataset(moved).digest())
        resourced = [dict(r, source="y") if r["model"] == "m0" else r for r in rows]
        self.assertNotEqual(base, build_dataset(resourced).digest())

    def test_with_clip_changes_only_the_clip(self):
        rows = [dict(r, score=0.0 if r["model"] == "m0" else r["score"]) for r in self._rows()]
        ds = build_dataset(rows)
        wide = ds.with_clip(0.02)
        self.assertAlmostEqual(float(wide.score.min()), 0.02)
        self.assertAlmostEqual(float(ds.score.min()), SCORE_EPS)
        self.assertEqual(wide.restrict(lambda r: r["model"] != "m1").eps, 0.02)


class DeveloperReportedTest(unittest.TestCase):
    def test_classification_follows_the_protocol_markers(self):
        from pollbench.data import is_developer_reported
        for src in ("Qwen Technical Report", "Opus 4.5 System Card", "Llama 2: Open Foundation and Fine-Tuned Chat Models",
                    "Mixtral of Experts", "Anthropic annoucement", "The Claude 3 Model Family: Opus, Sonnet, Haiku"):
            self.assertTrue(is_developer_reported(src), src)
        for src in ("", "Tweet from VPCT creator", "PapersWithCode", "https://arcprize.org/leaderboard", "Stanford CRFM Leaderboard", "lechmazur/writing Github repository"):
            self.assertFalse(is_developer_reported(src), src)

    def test_restrict_reapplies_minimums_and_keeps_dates(self):
        rows = [{"model": f"m{i}", "benchmark": f"b{j}", "score": 0.1 + 0.01 * i + 0.02 * j,
                 "model_date": f"2025-01-{i + 1:02d}", "bench_date": "2024-01-01", "source": "x"}
                for i in range(8) for j in range(6)]
        ds = build_dataset(rows)
        sub = ds.restrict(lambda r: r["model"] != "m0")
        self.assertEqual(sub.n_models, 7)
        self.assertEqual(sub.model_date[0], "2025-01-02")
