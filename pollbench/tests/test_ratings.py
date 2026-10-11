import unittest

import numpy as np

from pollbench.models import HierBeta
from pollbench.ratings import MAX_RHAT, grade, rate_benchmarks, reliability_draws, true_reliability
from pollbench.sim import simulate


class GradeTest(unittest.TestCase):
    def test_grade_uses_posterior_probability_not_the_mean(self):
        wide = np.concatenate([np.full(500, 0.95), np.full(500, 0.30)])  # mean 0.625
        self.assertEqual(grade(wide, 50), "D")
        tight = np.full(1000, 0.90)
        self.assertEqual(grade(tight, 50), "A")

    def test_thin_benchmarks_are_ungraded(self):
        self.assertEqual(grade(np.full(100, 0.95), 5), "ungraded")


class RateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds, cls.truth = simulate(n_models=60, n_benchmarks=12, seed=3)
        cls.fit = HierBeta(mcmc=True, n_warmup=100, n_samples=100, n_chains=2).fit(cls.ds)

    def test_rows_cover_every_benchmark_and_respect_the_rhat_rule(self):
        rows = rate_benchmarks(self.ds, self.fit, np.random.default_rng(0), n_draws=200)
        self.assertEqual(len(rows), self.ds.n_benchmarks)
        for r in rows:
            self.assertIsNotNone(r.rhat)
            if r.rhat > MAX_RHAT or np.isnan(r.rhat):
                self.assertEqual(r.grade, "ungraded")
            self.assertLessEqual(r.reliability_lo, r.reliability_hi)

    def test_all_models_reliability_and_spread_are_reported(self):
        rows = rate_benchmarks(self.ds, self.fit, np.random.default_rng(0), n_draws=200)
        for r in rows:
            self.assertTrue(0 <= r.all_models_reliability_mean <= 1)
            self.assertGreater(r.capability_spread, 0)

    def test_laplace_only_fits_are_refused(self):
        fit = HierBeta(mcmc=False).fit(self.ds)
        with self.assertRaises(ValueError):
            rate_benchmarks(self.ds, fit, np.random.default_rng(0))

    def test_true_reliability_is_a_share(self):
        t = self.truth
        r = true_reliability(self.ds, t.theta, t.difficulty, t.slope, t.precision)
        self.assertTrue(((r >= 0) & (r <= 1) | np.isnan(r)).all())


if __name__ == "__main__":
    unittest.main()
