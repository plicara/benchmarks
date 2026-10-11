import unittest

import numpy as np
from scipy.stats import norm

from pollbench.scoring import crossed_bootstrap, crps_samples, interval_hit, paired_bootstrap


class ScoringTest(unittest.TestCase):
    def test_point_mass_crps_is_absolute_error(self):
        samples = np.full((3, 50), 0.4)
        y = np.array([0.4, 0.9, 0.1])
        np.testing.assert_allclose(crps_samples(samples, y), np.abs(y - 0.4), atol=1e-12)

    def test_matches_the_normal_closed_form(self):
        rng = np.random.default_rng(0)
        mu, sd, y = 0.3, 0.1, 0.45
        samples = rng.normal(mu, sd, size=(1, 200_000))
        z = (y - mu) / sd
        exact = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        self.assertAlmostEqual(float(crps_samples(samples, np.array([y]))[0]), float(exact), places=3)

    def test_a_sharper_correct_forecast_scores_better(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 2000)
        sharp = y[:, None] + 0.1 * rng.standard_normal((2000, 100))
        diffuse = rng.normal(0, 3, size=(2000, 100))
        self.assertLess(crps_samples(sharp, y).mean(), crps_samples(diffuse, y).mean())

    def test_interval_coverage_of_a_calibrated_forecast(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=4000)
        samples = rng.normal(size=(4000, 400))
        self.assertAlmostEqual(interval_hit(samples, y, 0.9).mean(), 0.9, delta=0.02)
        self.assertAlmostEqual(interval_hit(samples, y, 0.5).mean(), 0.5, delta=0.03)

    def test_paired_bootstrap_interval_contains_the_mean_and_excludes_zero_for_a_clear_gap(self):
        diffs = {f"m{i}": 0.05 + 0.01 * np.sin(i) for i in range(40)}
        mean, lo, hi = paired_bootstrap(diffs)
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)
        self.assertGreater(lo, 0)

    def test_crps_is_unbiased_for_small_sample_sets(self):
        rng = np.random.default_rng(3)
        y = np.zeros(1)
        est = np.mean([crps_samples(rng.normal(size=(1, 5)), y)[0] for _ in range(20000)])
        exact = 1.0 * (0 + 2 * norm.pdf(0) - 1 / np.sqrt(np.pi))
        self.assertAlmostEqual(float(est), float(exact), places=2)

    def test_crossed_bootstrap_is_wider_than_resampling_models_alone_when_benchmarks_matter(self):
        rng = np.random.default_rng(4)
        nm, nb = 30, 12
        mi, bi = np.meshgrid(np.arange(nm), np.arange(nb), indexing="ij")
        mi, bi = mi.ravel(), bi.ravel()
        bench_effect = rng.normal(0, 0.05, nb)
        diff = 0.02 + bench_effect[bi] + rng.normal(0, 0.01, mi.size)
        mean, lo, hi = crossed_bootstrap(mi, bi, diff, n_boot=400)
        per_model = {f"m{m}": diff[mi == m].mean() for m in range(nm)}
        _, plo, phi = paired_bootstrap(per_model, n_boot=400)
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)
        self.assertGreater(hi - lo, 2 * (phi - plo))


if __name__ == "__main__":
    unittest.main()
