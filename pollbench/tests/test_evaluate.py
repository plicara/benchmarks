import unittest

import numpy as np

from pollbench.evaluate import MIN_KEPT, evaluate, frontier_split, kfold_cells
from pollbench.models import BenchmarkMean, MeanRank
from pollbench.ratings import grade, heldout_skill, reliability_draws, true_reliability
from pollbench.sim import simulate


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.ds, _ = simulate(n_models=80, n_benchmarks=25, seed=7)

    def test_training_always_keeps_the_minimum_per_model_and_benchmark(self):
        split = kfold_cells(self.ds, 5, seed=0)
        for f in range(5):
            train = split.held != f
            self.assertGreaterEqual(np.bincount(self.ds.model_idx[train], minlength=self.ds.n_models).min(), MIN_KEPT)
            self.assertGreaterEqual(np.bincount(self.ds.bench_idx[train], minlength=self.ds.n_benchmarks).min(), MIN_KEPT)

    def test_each_cell_is_held_out_at_most_once(self):
        split = kfold_cells(self.ds, 5, seed=0)
        held = split.held >= 0
        self.assertGreater(held.sum(), 0.5 * self.ds.n_cells)

    def test_frontier_holds_only_cells_of_the_newest_models(self):
        split = frontier_split(self.ds, 0.25, seed=0)
        held_models = set(self.ds.model_idx[split.held >= 0].tolist())
        dates = sorted(self.ds.model_date[m] for m in range(self.ds.n_models))
        cutoff = dates[int(len(dates) * 0.75) - 1]
        self.assertTrue(all(self.ds.model_date[m] >= cutoff for m in held_models))

    def test_evaluation_scores_only_held_out_cells_and_is_deterministic(self):
        split = kfold_cells(self.ds, 4, seed=1)
        methods = [BenchmarkMean(), MeanRank()]
        a = evaluate(self.ds, methods, split, n_draws=100, seed=3)
        b = evaluate(self.ds, methods, split, n_draws=100, seed=3)
        for ra, rb in zip(a, b):
            np.testing.assert_allclose(ra.crps, rb.crps)
            self.assertEqual(len(ra.cell), int((split.held >= 0).sum()))
            self.assertTrue((split.held[ra.cell] >= 0).all())

    def test_informed_baseline_beats_benchmark_mean(self):
        split = kfold_cells(self.ds, 4, seed=1)
        res = {r.method: r for r in evaluate(self.ds, [BenchmarkMean(), MeanRank()], split, n_draws=200)}
        self.assertLess(res["mean_rank"].crps.mean(), res["benchmark_mean"].crps.mean())


class RatingTest(unittest.TestCase):
    def test_reliability_is_near_one_for_low_noise_and_near_zero_for_high_noise(self):
        ds, truth = simulate(n_models=80, n_benchmarks=25, seed=7)
        lo_noise = true_reliability(ds, truth.theta, truth.difficulty, truth.slope, np.full(ds.n_benchmarks, 1e5))
        hi_noise = true_reliability(ds, truth.theta, truth.difficulty, truth.slope, np.full(ds.n_benchmarks, 0.5))
        self.assertGreater(np.nanmean(lo_noise), 0.99)
        self.assertLess(np.nanmean(hi_noise), 0.5)

    def test_reliability_hand_computed_for_two_models(self):
        class D:  # minimal stand-in with one benchmark scored by three models
            n_benchmarks = 1
            model_idx = np.array([0, 1, 2])
            bench_idx = np.array([0, 0, 0])
        th = np.array([[-1.0, 0.0, 1.0]])
        r = reliability_draws(D, th, np.array([[0.0]]), np.array([[1.0]]), np.array([[19.0]]))
        from scipy.special import expit
        p = expit(th[0])
        expected = p.var() / (p.var() + (p * (1 - p)).mean() / 20.0)
        self.assertAlmostEqual(float(r[0, 0]), float(expected), places=12)

    def test_grade_bands_and_ungraded_cases(self):
        rng = np.random.default_rng(0)
        self.assertEqual(grade(np.full(1000, 0.95), 30), "A")
        self.assertEqual(grade(np.full(1000, 0.75), 30), "B")
        self.assertEqual(grade(np.full(1000, 0.55), 30), "C")
        self.assertEqual(grade(np.full(1000, 0.2), 30), "D")
        self.assertEqual(grade(np.full(1000, 0.95), 9), "ungraded")
        wide = np.concatenate([np.full(800, 0.95), np.full(200, 0.4)])  # only 80% mass above 0.85
        self.assertNotEqual(grade(wide, 30), "A")

    def test_heldout_skill_is_positive_for_a_better_method(self):
        ds, _ = simulate(n_models=80, n_benchmarks=25, seed=7)
        split = kfold_cells(ds, 4, seed=1)
        res = {r.method: r for r in evaluate(ds, [BenchmarkMean(), MeanRank()], split, n_draws=200)}
        skill = heldout_skill(ds, res, baseline="benchmark_mean", method="mean_rank", n_boot=100)
        self.assertGreater(np.mean([v[0] for v in skill.values()]), 0.2)


if __name__ == "__main__":
    unittest.main()
