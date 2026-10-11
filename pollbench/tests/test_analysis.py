import unittest

import numpy as np

from pollbench.analysis import _cutoff, stability_rho
from pollbench.sim import simulate


class AnalysisPiecesTest(unittest.TestCase):
    def test_stability_needs_enough_shared_benchmarks(self):
        full = {"benchmarks": [{"benchmark": f"b{i}", "reliability_mean": i / 10, "n_models": 20} for i in range(8)]}
        early = {"benchmarks": [{"benchmark": f"b{i}", "reliability_mean": i / 10 + 0.01, "n_models": 20} for i in range(8)]}
        rho, n = stability_rho(full, early)
        self.assertEqual(n, 8)
        self.assertAlmostEqual(rho, 1.0)
        thin = {"benchmarks": [{"benchmark": f"b{i}", "reliability_mean": 0.5, "n_models": 5} for i in range(8)]}
        self.assertEqual(stability_rho(full, thin), (None, 0))

    def test_cutoff_is_a_year_before_the_newest_model(self):
        ds, _ = simulate(n_models=30, n_benchmarks=8, seed=1)
        newest = max(d for d in ds.model_date if d)
        self.assertEqual(int(newest[:4]) - int(_cutoff(ds)[:4]), 1)


if __name__ == "__main__":
    unittest.main()


class SecondFactorCheckTest(unittest.TestCase):
    def test_check_sees_a_second_factor_and_stays_quiet_without_one(self):
        from pollbench.checks import second_factor_check
        from pollbench.models import HierBeta
        quick = dict(n_chains=2, n_warmup=100, n_samples=100)
        out = {}
        for sec in (0.0, 0.8):
            ds, _ = simulate(seed=2, second_factor=sec)
            out[sec] = second_factor_check(ds, HierBeta(**quick).fit(ds), np.random.default_rng(1), n_rep=100)
        self.assertGreater(out[0.0]["p_value"], 0.05)
        self.assertLess(out[0.8]["p_value"], 0.05)
