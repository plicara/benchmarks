import unittest

import numpy as np

from pollbench.models import BenchmarkMean, EciLS, HierBeta, LowRank1, MeanRank
from pollbench.sim import simulate


class GradientTest(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        ds, _ = simulate(n_models=40, n_benchmarks=15, seed=3)
        h = HierBeta()
        M, B = ds.n_models, ds.n_benchmarks
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(size=M), rng.normal(size=B), 0.1 * rng.normal(size=B),
                            np.log(30) + 0.2 * rng.normal(size=B), [0.1, 0.1, 3.3]])
        args = (ds.model_idx, ds.bench_idx, ds.score, M, B)
        _, g = h._nlp(x, *args)
        for j in [0, M // 2, M, M + B // 2, M + B + 1, M + 2 * B + 2, M + 3 * B, M + 3 * B + 1, M + 3 * B + 2]:
            e = np.zeros_like(x)
            e[j] = 1e-6
            num = (h._nlp(x + e, *args)[0] - h._nlp(x - e, *args)[0]) / 2e-6
            self.assertAlmostEqual(g[j], num, delta=1e-4 * max(1.0, abs(num)), msg=f"parameter {j}")


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.ds, self.truth = simulate(n_models=80, n_benchmarks=25, seed=5)

    def test_hier_beta_map_recovers_capability_and_difficulty(self):
        fit = HierBeta(mcmc=False).fit(self.ds)
        rng = np.random.default_rng(0)
        th = fit.model_summary(rng)["capability_mean"]
        d = fit.benchmark_summary(rng)["difficulty_mean"]
        self.assertGreater(np.corrcoef(th, self.truth.theta)[0, 1], 0.95)
        self.assertGreater(np.corrcoef(d, self.truth.difficulty)[0, 1], 0.85)

    def test_eci_ls_recovers_capability(self):
        fit = EciLS().fit(self.ds)
        self.assertGreater(np.corrcoef(fit.theta, self.truth.theta)[0, 1], 0.95)

    def test_every_method_returns_valid_predictive_samples(self):
        rng = np.random.default_rng(0)
        for method in (BenchmarkMean(), MeanRank(), LowRank1(), EciLS(), HierBeta(mcmc=False)):
            fitted = method.fit(self.ds)
            samp = fitted.predict(self.ds.model_idx[:30], self.ds.bench_idx[:30], 50, rng)
            self.assertEqual(samp.shape, (30, 50), method.name)
            self.assertTrue(np.isfinite(samp).all(), method.name)
            self.assertTrue(((samp > 0) & (samp < 1)).all(), method.name)

    def test_mcmc_draws_are_close_to_the_mode_summary(self):
        fit = HierBeta(n_warmup=100, n_samples=100, n_chains=2).fit(self.ds)
        rng = np.random.default_rng(0)
        th = fit.model_summary(rng)["capability_mean"]
        self.assertGreater(np.corrcoef(th, self.truth.theta)[0, 1], 0.95)
        self.assertEqual(fit.diagnostics["rhat"].shape[0], fit.x.shape[0])


class NonCentredTest(unittest.TestCase):
    def test_round_trip_keeps_what_the_likelihood_sees_and_pins_location_and_scale(self):
        h = HierBeta()
        M, B = 20, 6
        x = np.random.default_rng(0).normal(0, 0.3, M + 3 * B + 3)
        x[M + 3 * B + 1] = 0.4
        y = h._from_nc(h._to_nc(x, M, B), M, B)
        lin = lambda v: np.exp(v[M + B:M + 2 * B])[:, None] * (v[:M][None, :] - v[M:M + B][:, None])
        np.testing.assert_allclose(lin(y), lin(x), atol=1e-12)
        self.assertAlmostEqual(float(y[:M].mean()), 0.0, places=12)
        self.assertAlmostEqual(float(y[M + B:M + 2 * B].mean()), 0.0, places=12)
        self.assertAlmostEqual(float(y[M:M + B].mean()), float(y[M + 3 * B]), places=12)


class ChainDiagnosticsTest(unittest.TestCase):
    def test_ess_matches_the_ar1_formula_and_rhat_flags_separated_chains(self):
        from pollbench.models import _chain_diagnostics
        rng = np.random.default_rng(0)
        phi, n, c = 0.8, 2000, 4
        x = np.zeros((c, n, 1))
        e = rng.standard_normal((c, n))
        for t in range(1, n):
            x[:, t, 0] = phi * x[:, t - 1, 0] + e[:, t]
        diag = _chain_diagnostics(x)
        expected = c * n * (1 - phi) / (1 + phi)
        self.assertAlmostEqual(diag["ess_min"] / expected, 1.0, delta=0.3)
        self.assertLess(diag["rhat_max"], 1.05)
        shifted = x.copy()
        shifted[0] += 5
        self.assertGreater(_chain_diagnostics(shifted)["rhat_max"], 1.1)


if __name__ == "__main__":
    unittest.main()
