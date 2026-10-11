"""Candidate predictors for held-out benchmark scores.

Every method is fitted on the training cells of a ``Dataset`` and returns
predictive samples for requested (model, benchmark) cells, so all methods are
graded by the same proper score. Methods:

* ``BenchmarkMean``: moment-matched Beta of the benchmark's training scores.
* ``MeanRank``: a model's average within-benchmark z-score, mapped to each
  benchmark by simple regression.
* ``LowRank1``: rank-one matrix completion (the PCA-style single factor).
* ``EciLS``: sigmoid(slope * (capability - difficulty)) fitted by ridge least
  squares, the form Epoch's Capabilities Index uses; the predictive spread is
  our wrapper (a per-benchmark Gaussian on training residuals), since ECI
  itself publishes none.
* ``HierBeta``: the same sigmoid mean with a Beta likelihood whose precision
  is estimated per benchmark, hierarchical priors on difficulty, slope and
  precision, and a Laplace approximation to the posterior around the mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, logit

from .data import SCORE_EPS, Dataset

P_EPS = 1e-6


class Fitted:
    def predict(self, model_idx: np.ndarray, bench_idx: np.ndarray, n_draws: int,
                rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


def _z_scores(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-benchmark z of each training cell plus the benchmark mean and sd."""
    mean = np.zeros(ds.n_benchmarks)
    sd = np.ones(ds.n_benchmarks)
    for b in range(ds.n_benchmarks):
        v = ds.score[ds.bench_idx == b]
        if len(v):
            mean[b] = v.mean()
            sd[b] = max(v.std(ddof=1) if len(v) > 1 else 0.0, 1e-3)
    return (ds.score - mean[ds.bench_idx]) / sd[ds.bench_idx], mean, sd


def _model_mean_z(ds: Dataset, z: np.ndarray) -> np.ndarray:
    total = np.bincount(ds.model_idx, weights=z, minlength=ds.n_models)
    count = np.bincount(ds.model_idx, minlength=ds.n_models)
    out = np.zeros(ds.n_models)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, SCORE_EPS, 1 - SCORE_EPS)


# ---------------------------------------------------------------- baselines

class _GaussianPerBenchmark(Fitted):
    """Prediction = f(model) with a per-benchmark Gaussian residual."""

    def __init__(self, mean_fn, resid_sd: np.ndarray):
        self.mean_fn = mean_fn
        self.resid_sd = resid_sd

    def predict(self, model_idx, bench_idx, n_draws, rng):
        mu = self.mean_fn(model_idx, bench_idx)
        sd = self.resid_sd[bench_idx]
        return _clip(mu[:, None] + sd[:, None] * rng.standard_normal((len(mu), n_draws)))


class BenchmarkMean:
    name = "benchmark_mean"

    def fit(self, ds: Dataset) -> Fitted:
        mean = np.full(ds.n_benchmarks, 0.5)
        var = np.full(ds.n_benchmarks, 0.05)
        for b in range(ds.n_benchmarks):
            v = ds.score[ds.bench_idx == b]
            if len(v) > 1:
                mean[b], var[b] = v.mean(), max(v.var(ddof=1), 1e-4)
        # Moment-matched Beta, with the variance capped by mean * (1 - mean).
        var = np.minimum(var, 0.95 * mean * (1 - mean))
        kappa = mean * (1 - mean) / var - 1

        class _F(Fitted):
            def predict(self, model_idx, bench_idx, n_draws, rng):
                m, k = mean[bench_idx], kappa[bench_idx]
                return _clip(rng.beta((m * k)[:, None], ((1 - m) * k)[:, None], size=(len(m), n_draws)))
        return _F()


class MeanRank:
    name = "mean_rank"

    def fit(self, ds: Dataset) -> Fitted:
        z, bmean, bsd = _z_scores(ds)
        mz = _model_mean_z(ds, z)
        slope = np.zeros(ds.n_benchmarks)
        icpt = bmean.copy()
        sd = np.full(ds.n_benchmarks, 0.1)
        for b in range(ds.n_benchmarks):
            sel = ds.bench_idx == b
            x, y = mz[ds.model_idx[sel]], ds.score[sel]
            if len(y) >= 4 and x.std() > 1e-9:
                A = np.vstack([np.ones_like(x), x]).T
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                icpt[b], slope[b] = coef
                resid = y - A @ coef
                sd[b] = max(np.sqrt((resid ** 2).sum() / max(len(y) - 2, 1)), 0.01)
            elif len(y) > 1:
                sd[b] = max(y.std(ddof=1), 0.01)
        return _GaussianPerBenchmark(lambda m, b: icpt[b] + slope[b] * mz[m], sd)


class LowRank1:
    name = "low_rank_1"

    def __init__(self, iters: int = 60, ridge: float = 0.05):
        self.iters, self.ridge = iters, ridge

    def fit(self, ds: Dataset) -> Fitted:
        z, bmean, bsd = _z_scores(ds)
        u = _model_mean_z(ds, z)
        u = u / max(u.std(), 1e-6)
        mu = bmean.copy()
        lam = np.zeros(ds.n_benchmarks)
        for _ in range(self.iters):
            for b in range(ds.n_benchmarks):
                sel = ds.bench_idx == b
                x, y = u[ds.model_idx[sel]], ds.score[sel]
                A = np.vstack([np.ones_like(x), x]).T
                reg = self.ridge * np.eye(2)
                coef = np.linalg.solve(A.T @ A + reg, A.T @ y)
                mu[b], lam[b] = coef
            for m in range(ds.n_models):
                sel = ds.model_idx == m
                l, r = lam[ds.bench_idx[sel]], ds.score[sel] - mu[ds.bench_idx[sel]]
                u[m] = (l @ r) / (l @ l + self.ridge)
            s = max(u.std(), 1e-6)
            u, lam = u / s, lam * s
        sd = np.full(ds.n_benchmarks, 0.1)
        for b in range(ds.n_benchmarks):
            sel = ds.bench_idx == b
            resid = ds.score[sel] - (mu[b] + lam[b] * u[ds.model_idx[sel]])
            sd[b] = max(np.sqrt((resid ** 2).sum() / max(sel.sum() - 2, 1)), 0.01)
        return _GaussianPerBenchmark(lambda m, b: mu[b] + lam[b] * u[m], sd)


# ------------------------------------------------------------ ECI-style LS

@dataclass
class EciLS:
    """Ridge least squares for score = sigmoid(a_b * (theta_m - d_b))."""

    ridge: float = 0.05
    name: str = "eci_ls"

    def _unpack(self, x, M, B):
        return x[:M], x[M:M + B], x[M + B:M + 2 * B]

    def fit(self, ds: Dataset) -> Fitted:
        M, B = ds.n_models, ds.n_benchmarks
        mi, bi, s = ds.model_idx, ds.bench_idx, ds.score
        z, bmean, _ = _z_scores(ds)
        theta0 = _model_mean_z(ds, z)
        x0 = np.concatenate([theta0, logit(_clip(bmean)), np.zeros(B)])

        def f(x):
            th, d, la = self._unpack(x, M, B)
            a = np.exp(la)
            zz = a[bi] * (th[mi] - d[bi])
            p = expit(zz)
            r = p - s
            loss = (r ** 2).sum() + self.ridge * (th @ th + d @ d + la @ la)
            g_z = 2 * r * p * (1 - p)
            g_th = np.bincount(mi, weights=g_z * a[bi], minlength=M) + 2 * self.ridge * th
            g_d = np.bincount(bi, weights=-g_z * a[bi], minlength=B) + 2 * self.ridge * d
            g_la = np.bincount(bi, weights=g_z * zz, minlength=B) + 2 * self.ridge * la
            return loss, np.concatenate([g_th, g_d, g_la])

        res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        th, d, la = self._unpack(res.x, M, B)
        a = np.exp(la)
        pred = expit(a[bi] * (th[mi] - d[bi]))
        sd = np.zeros(B)
        for b in range(B):
            sel = bi == b
            sd[b] = max(np.sqrt(((s[sel] - pred[sel]) ** 2).sum() / max(sel.sum() - 3, 1)), 0.01)
        fitted = _GaussianPerBenchmark(lambda m, b: expit(a[b] * (th[m] - d[b])), sd)
        fitted.theta, fitted.difficulty, fitted.slope = th, d, a  # type: ignore[attr-defined]
        return fitted


# --------------------------------------------------- hierarchical Beta model

@dataclass
class HierBeta:
    """Sigmoid mean, Beta noise with per-benchmark precision, MAP start + NUTS.

    theta_m ~ N(0, s^2); d_b ~ N(mu_d, (sd_d s)^2); log a_b ~ N(0, sd_a^2);
    log kappa_b ~ N(mu_k, sd_k). The scale of the capability axis is fixed by
    pinning the mean log slope at zero and letting the capability spread s be
    estimated (log s ~ N(log 1.5, 0.75^2)); difficulty is measured in units of s.
    Fixing theta's scale and freeing the slopes' mean instead leaves a ridge
    that point estimates collapse (theta sd 0.1, slopes e^3) and the sampler
    cannot cross. The other hyper-means get weak priors; the hyper scales are
    fixed and listed in the protocol. kappa_b is the Beta precision, read as an
    effective number of items: a benchmark whose kappa is far below its true
    item count carries excess variance beyond sampling.

    ``mcmc=False`` returns Laplace draws around the mode, which are for quick
    predictive use only: they carry no convergence diagnostic, and
    ``rate_benchmarks`` refuses them because grades are gated on split-R-hat.
    """

    sd_d: float = 2.0
    sd_a: float = 0.5
    sd_k: float = 1.0
    n_draws: int = 400
    mcmc: bool = True
    n_chains: int = 2
    n_warmup: int = 200
    n_samples: int = 300
    nc_d: bool = True  # difficulties non-centred around their exact mean mu_d (d = mu_d + sd_d s (w - mean w))
    dense_mass: bool = False
    floor: bool = False  # sensitivity model: a per-benchmark floor f_b (chance level) under the sigmoid
    start_scale: float = 1.0  # sd of the chain starts around the mode, in whitened units
    seed: int = 0
    name: str = "hier_beta"

    def _nlp(self, x, mi, bi, s, M, B):
        with np.errstate(over="ignore", invalid="ignore"):  # rejected trial steps can overflow; they are discarded
            return self._nlp_raw(x, mi, bi, s, M, B)

    def _nlp_raw(self, x, mi, bi, s, M, B):
        th, d, la, lk = x[:M], x[M:M + B], x[M + B:M + 2 * B], x[M + 2 * B:M + 3 * B]
        mu_d, lsd, mu_k = x[M + 3 * B:]
        a, k = np.exp(np.clip(la, -8, 8)), np.exp(np.clip(lk, -3, 12))  # clipping only bites on rejected trial steps
        zz = a[bi] * (th[mi] - d[bi])
        p = np.clip(expit(zz), P_EPS, 1 - P_EPS)
        kk = k[bi]
        al, be = p * kk, (1 - p) * kk
        ls, l1s = np.log(s), np.log1p(-s)
        ll = (al - 1) * ls + (be - 1) * l1s - (gammaln(al) + gammaln(be) - gammaln(kk))
        dpsi_a, dpsi_b = digamma(al), digamma(be)
        dll_dp = kk * (ls - l1s - dpsi_a + dpsi_b)
        dll_dk = p * ls + (1 - p) * l1s - (p * dpsi_a + (1 - p) * dpsi_b - digamma(kk))
        g_z = dll_dp * p * (1 - p)
        # priors
        lsd = float(np.clip(lsd, -4, 4))
        s2 = np.exp(2 * lsd)
        dd = ((d - mu_d) ** 2).sum()
        pri = 0.5 * (th @ th) / s2 + M * lsd + 0.5 * dd / (self.sd_d ** 2 * s2) + B * lsd \
            + 0.5 * (la ** 2).sum() / self.sd_a ** 2 + 0.5 * ((lk - mu_k) ** 2).sum() / self.sd_k ** 2 \
            + 0.5 * (mu_d / 3.0) ** 2 + 0.5 * ((lsd - np.log(1.5)) / 0.75) ** 2 + 0.5 * ((mu_k - np.log(30)) / 2.0) ** 2
        nlp = -ll.sum() + pri
        g_th = -np.bincount(mi, weights=g_z * a[bi], minlength=M) + th / s2
        g_d = np.bincount(bi, weights=g_z * a[bi], minlength=B) + (d - mu_d) / (self.sd_d ** 2 * s2)
        g_la = -np.bincount(bi, weights=g_z * zz, minlength=B) + la / self.sd_a ** 2
        g_lk = -np.bincount(bi, weights=dll_dk * kk, minlength=B) + (lk - mu_k) / self.sd_k ** 2
        g_mu_d = -((d - mu_d)).sum() / (self.sd_d ** 2 * s2) + mu_d / 9.0
        g_ls = -(th @ th) / s2 + M - dd / (self.sd_d ** 2 * s2) + B + (lsd - np.log(1.5)) / 0.75 ** 2
        g_mu_k = -((lk - mu_k)).sum() / self.sd_k ** 2 + (mu_k - np.log(30)) / 4.0
        return nlp, np.concatenate([g_th, g_d, g_la, g_lk, [g_mu_d, g_ls, g_mu_k]])

    def fit(self, ds: Dataset) -> "HierBetaFit":
        M, B = ds.n_models, ds.n_benchmarks
        mi, bi, s = ds.model_idx, ds.bench_idx, ds.score
        z, bmean, _ = _z_scores(ds)
        x0 = np.concatenate([_model_mean_z(ds, z), logit(_clip(bmean)), np.zeros(B),
                             np.full(B, np.log(30.0)), [0.0, np.log(1.5), np.log(30.0)]])
        # The joint mode is degenerate: the capability spread collapses (a mode has no
        # Jacobian term) and a sparse benchmark can be fitted perfectly with its
        # precision running away. So the starting point and whitening come from the
        # conditional mode with log s, the precisions and their hyper-mean held at
        # their prior centres; HMC then samples all of them.
        i_ls = M + 3 * B + 1
        free = np.ones(len(x0), bool)
        free[[i_ls, M + 3 * B + 2]] = False
        free[M + 2 * B:M + 3 * B] = False

        def fg(xr):
            x = x0.copy()
            x[free] = xr
            f, g = self._nlp(x, mi, bi, s, M, B)
            return f, g[free]

        res = minimize(fg, x0[free], jac=True, method="L-BFGS-B", options={"maxiter": 300})
        xr, converged = self._newton_polish(res.x, fg)
        cov_r = self._laplace_cov(xr, fg)
        x = x0.copy()
        x[free] = xr
        cov = np.zeros((len(x0), len(x0)))
        cov[np.ix_(free, free)] = cov_r
        cov[i_ls, i_ls] = 0.75 ** 2
        cov[M + 3 * B + 2, M + 3 * B + 2] = 2.0 ** 2
        idx = np.arange(M + 2 * B, M + 3 * B)
        cov[idx, idx] = self.sd_k ** 2
        obj = float(self._nlp(x, mi, bi, s, M, B)[0])
        fit = HierBetaFit(x, cov, M, B, self.n_draws, converged, obj)
        if self.mcmc:
            fit.set_mcmc(*self._hmc(x, cov, mi, bi, s, M, B))
        return fit

    # ---- NUTS in non-centred coordinates
    # Only a (theta - d) enters the likelihood, so location and scale are fixed
    # exactly: capability has mean zero (theta = s (z - mean z)) and the log
    # slopes have mean zero. Pinning them only through the priors left two ridges
    # (a common shift of thetas and difficulties, and s against the slopes) along
    # which chains drifted, with ESS near 4. theta = s z also removes the funnel
    # between the spread s and the thetas. The hand-written HMC used before got
    # chains trapped at a vanishing spread once they were started from dispersed
    # points, so the sampler is NumPyro's NUTS with its own warmup.

    def _from_nc(self, xp, M, B):
        x = np.array(xp, dtype=float, copy=True)
        s = np.exp(x[..., M + 3 * B + 1])[..., None]
        x[..., :M] = (xp[..., :M] - xp[..., :M].mean(-1, keepdims=True)) * s
        x[..., M + B:M + 2 * B] = xp[..., M + B:M + 2 * B] - xp[..., M + B:M + 2 * B].mean(-1, keepdims=True)
        if self.nc_d:
            w = xp[..., M:M + B] - xp[..., M:M + B].mean(-1, keepdims=True)
            x[..., M:M + B] = xp[..., M + 3 * B][..., None] + self.sd_d * s * w
        return x

    def _to_nc(self, x, M, B):
        x = np.array(x, dtype=float, copy=True)
        # the likelihood sees a (theta - d) only: move the mean log slope into the scale ...
        m = x[..., M + B:M + 2 * B].mean(-1, keepdims=True)
        x[..., M + B:M + 2 * B] -= m
        x[..., :M + B] *= np.exp(m)
        x[..., M + 3 * B] *= np.exp(m[..., 0])
        x[..., M + 3 * B + 1] += m[..., 0]
        xp = x.copy()
        s = np.exp(x[..., M + 3 * B + 1])[..., None]
        shift = x[..., :M].mean(-1, keepdims=True)  # ... and the mean capability into the difficulties
        xp[..., M:M + B] = x[..., M:M + B] - shift
        xp[..., M + 3 * B] = x[..., M + 3 * B] - shift[..., 0]
        x = xp.copy()
        xp[..., :M] = (x[..., :M] - shift) / s
        if self.nc_d:
            xp[..., M + 3 * B] = x[..., M:M + B].mean(-1)  # mu_d is the mean difficulty exactly
            xp[..., M:M + B] = (x[..., M:M + B] - xp[..., M + 3 * B][..., None]) / (self.sd_d * s)
        return xp

    def _hmc(self, x_hat, cov, mi, bi, s_obs, M, B):
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from jax.scipy.special import expit as jexpit
        from numpyro.infer import MCMC, NUTS
        jax.config.update("jax_enable_x64", True)
        numpyro.set_platform("cpu")
        sd_d, sd_a, sd_k = self.sd_d, self.sd_a, self.sd_k
        mi_j, bi_j, y = jnp.asarray(mi), jnp.asarray(bi), jnp.asarray(s_obs)

        def model():
            mu_d = numpyro.sample("mu_d", dist.Normal(0.0, 3.0))
            lsd = numpyro.sample("lsd", dist.Normal(np.log(1.5), 0.75))
            mu_k = numpyro.sample("mu_k", dist.Normal(np.log(30.0), 2.0))
            z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([M]))
            sc = jnp.exp(lsd)
            if self.nc_d:
                w = numpyro.sample("d", dist.Normal(0.0, 1.0).expand([B]))
                d = mu_d + sd_d * sc * (w - w.mean())
            else:
                d = numpyro.sample("d", dist.Normal(mu_d, sd_d * sc).expand([B]))
            la = numpyro.sample("la", dist.Normal(0.0, sd_a).expand([B]))
            la = la - la.mean()  # mean log slope pinned at 0 exactly (amendment 1 pinned it only through the prior)
            lk = numpyro.sample("lk", dist.Normal(mu_k, sd_k).expand([B]))
            th = sc * (z - z.mean())  # mean capability pinned at 0: only theta - d enters the likelihood
            p = jexpit(jnp.exp(la)[bi_j] * (th[mi_j] - d[bi_j]))
            if self.floor:
                lf = numpyro.sample("lf", dist.Normal(-3.0, 1.5).expand([B]))
                fl = jexpit(lf)[bi_j]
                p = fl + (1.0 - fl) * p
            p = jnp.clip(p, P_EPS, 1 - P_EPS)
            kk = jnp.exp(lk)[bi_j]
            numpyro.sample("y", dist.Beta(p * kk, (1 - p) * kk), obs=y)

        xp = self._to_nc(x_hat, M, B)
        sd = np.sqrt(np.clip(np.diag(self._nc_cov(x_hat, cov, M, B)), 1e-6, None))
        rng = np.random.default_rng(self.seed)
        # dispersed starts around the conditional mode, so R-hat can see a chain that stays away
        start = xp[None, :] + self.start_scale * sd[None, :] * rng.standard_normal((self.n_chains, len(xp)))
        init = {"z": start[:, :M], "d": start[:, M:M + B], "la": start[:, M + B:M + 2 * B],
                "lk": start[:, M + 2 * B:M + 3 * B], "mu_d": start[:, M + 3 * B],
                "lsd": start[:, M + 3 * B + 1], "mu_k": start[:, M + 3 * B + 2]}
        if self.floor:
            init["lf"] = np.full((self.n_chains, B), -3.0) + 0.3 * rng.standard_normal((self.n_chains, B))
        mcmc = MCMC(NUTS(model, target_accept_prob=0.9, dense_mass=self.dense_mass), num_warmup=self.n_warmup, num_samples=self.n_samples,
                    num_chains=self.n_chains, chain_method="sequential", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(self.seed), init_params=init, extra_fields=("diverging", "num_steps"))
        sm = mcmc.get_samples(group_by_chain=True)
        extra = mcmc.get_extra_fields(group_by_chain=True)
        xs = np.concatenate([np.asarray(sm["z"]), np.asarray(sm["d"]), np.asarray(sm["la"]), np.asarray(sm["lk"]),
                             np.asarray(sm["mu_d"])[..., None], np.asarray(sm["lsd"])[..., None],
                             np.asarray(sm["mu_k"])[..., None]], axis=-1)
        chains = self._from_nc(xs, M, B)
        if self.floor:  # floors ride after the standard coordinates; split() ignores them
            chains = np.concatenate([chains, np.asarray(sm["lf"])], axis=-1)
        div = np.asarray(extra["diverging"])
        return chains, {"divergences": int(div.sum()), "divergence_rate": float(div.mean()),
                        "mean_tree_steps": float(np.asarray(extra["num_steps"]).mean()), **_chain_diagnostics(chains)}

    def _nc_cov(self, x, cov, M, B):
        """Linearised covariance of the non-centred coordinates at x."""
        i_mu, i_ls = M + 3 * B, M + 3 * B + 1
        s = np.exp(x[i_ls])
        J = np.eye(len(x))
        J[:M, :M] = np.eye(M) / s
        J[:M, i_ls] = -x[:M] / s
        if self.nc_d:
            J[M:M + B, M:M + B] = np.eye(B) / (self.sd_d * s)
            J[M:M + B, i_mu] = -1.0 / (self.sd_d * s)
            J[M:M + B, i_ls] = -(x[M:M + B] - x[i_mu]) / (self.sd_d * s)
        return J @ cov @ J.T

    def _hessian(self, x, fg):
        n = len(x)
        H = np.empty((n, n))
        for j in range(n):
            h = 1e-4 * max(1.0, abs(x[j]))
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            H[:, j] = (fg(xp)[1] - fg(xm)[1]) / (2 * h)
        return 0.5 * (H + H.T)

    def _newton_polish(self, x, fg, iters: int = 40, tol: float = 1e-5):
        """Damped Newton steps from the L-BFGS point; ill-conditioning makes plain L-BFGS crawl."""
        f, g = fg(x)
        for _ in range(iters):
            if np.abs(g).max() < tol:
                return x, True
            w, V = np.linalg.eigh(self._hessian(x, fg))
            step = -(V * (1.0 / np.maximum(w, 1e-3))) @ (V.T @ g)
            t = 1.0
            while t > 1e-6:
                f2, g2 = fg(x + t * step)
                if f2 <= f + 1e-4 * t * (g @ step):
                    x, f, g = x + t * step, f2, g2
                    break
                t *= 0.5
            else:
                break
        return x, bool(np.abs(g).max() < 1e-3)

    def _laplace_cov(self, x, fg):
        w, V = np.linalg.eigh(self._hessian(x, fg))
        w = np.maximum(w, 1e-3)  # keep the approximation proper where the surface is flat
        return (V / w) @ V.T


def _ess(chains: np.ndarray) -> np.ndarray:
    """Effective sample size per parameter, Geyer's initial positive sequence on pooled autocorrelations (as in Stan)."""
    c, n, p = chains.shape
    x = chains - chains.mean(1, keepdims=True)
    f = np.fft.rfft(x, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n  # (c, n, p), biased autocovariance per chain
    mean_acov = acov.mean(0)  # (n, p)
    W = chains.var(1, ddof=1).mean(0)
    B_ = n * chains.mean(1).var(0, ddof=1) if c > 1 else np.zeros(p)
    var_plus = (n - 1) / n * W + B_ / n
    rho = 1.0 - (W[None, :] - mean_acov) / np.maximum(var_plus[None, :], 1e-12)
    rho[0] = 1.0
    ess = np.empty(p)
    for j in range(p):
        r = rho[:, j]
        total, prev = 0.0, np.inf
        for t in range(0, n - 1, 2):
            pair = r[t] + r[t + 1]
            if pair <= 0:
                break
            pair = min(pair, prev)  # monotone sequence
            total += pair
            prev = pair
        tau = max(-1.0 + 2.0 * total, 1.0 / np.log10(c * n))
        ess[j] = min(c * n / tau, c * n * np.log10(c * n))
    return ess


def _split_rhat(chains: np.ndarray) -> np.ndarray:
    c, n, p = chains.shape
    half = n // 2
    parts = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)  # (2c, half, p)
    means, var = parts.mean(1), parts.var(1, ddof=1)
    W, B_ = var.mean(0), half * means.var(0, ddof=1)
    return np.sqrt(((half - 1) / half * W + B_ / half) / np.maximum(W, 1e-12))


def _rank_normalise(chains: np.ndarray) -> np.ndarray:
    from scipy.stats import norm, rankdata
    c, n, p = chains.shape
    flat = chains.reshape(c * n, p)
    ranks = np.apply_along_axis(rankdata, 0, flat)
    return norm.ppf((ranks - 0.375) / (c * n + 0.25)).reshape(c, n, p)


def _chain_diagnostics(chains: np.ndarray) -> dict:
    """Rank-normalised split-R-hat (Vehtari et al. 2021: the larger of the bulk and folded versions) and ESS; chains is (c, n, p)."""
    c, n, p = chains.shape
    med = np.median(chains.reshape(c * n, p), axis=0)
    rhat = np.maximum(_split_rhat(_rank_normalise(chains)), _split_rhat(_rank_normalise(np.abs(chains - med))))
    ess = _ess(_rank_normalise(chains))  # bulk ESS
    return {"rhat_max": float(np.nanmax(rhat)), "ess_min": float(np.nanmin(ess)),
            "rhat": rhat, "ess": ess}


class HierBetaFit(Fitted):
    def __init__(self, x, cov, M, B, n_draws, converged, objective):
        self.x, self.cov, self.M, self.B = x, cov, M, B
        self.n_draws, self.converged, self.objective = n_draws, converged, objective
        self._L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(x)))
        self._mcmc: np.ndarray | None = None
        self.diagnostics: dict = {}

    def set_mcmc(self, chains: np.ndarray, diagnostics: dict) -> None:
        self._mcmc = chains.reshape(-1, chains.shape[-1])
        self.diagnostics = diagnostics

    @property
    def has_mcmc(self) -> bool:
        return self._mcmc is not None

    def benchmark_rhat(self) -> np.ndarray:
        """Worst split-R-hat among each benchmark's difficulty, slope and precision."""
        r = self.diagnostics["rhat"]
        M, B = self.M, self.B
        return np.max(np.vstack([r[M:M + B], r[M + B:M + 2 * B], r[M + 2 * B:M + 3 * B]]), axis=0)

    def draws(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        """Posterior draws: MCMC when available, else the Laplace approximation."""
        n = n or self.n_draws
        if self._mcmc is not None:
            return self._mcmc[rng.integers(0, len(self._mcmc), size=n)]
        return self.x[None, :] + rng.standard_normal((n, len(self.x))) @ self._L.T

    def split(self, X: np.ndarray):
        M, B = self.M, self.B
        return X[:, :M], X[:, M:M + B], np.exp(X[:, M + B:M + 2 * B]), np.exp(X[:, M + 2 * B:M + 3 * B])

    def floors(self, X: np.ndarray) -> np.ndarray | None:
        """Per-benchmark floors f_b for draws X, or None unless the fit used ``floor=True``."""
        n0 = self.M + 3 * self.B + 3
        return None if X.shape[1] <= n0 else expit(X[:, n0:])

    def predict(self, model_idx, bench_idx, n_draws, rng):
        X = self.draws(rng, n_draws)
        th, d, a, k = self.split(X)
        z = a[:, bench_idx] * (th[:, model_idx] - d[:, bench_idx])
        p = np.clip(expit(z), P_EPS, 1 - P_EPS)
        kk = k[:, bench_idx]
        return _clip(rng.beta(p * kk, (1 - p) * kk).T)

    def benchmark_summary(self, rng: np.random.Generator, n: int = 1000) -> dict[str, np.ndarray]:
        """Posterior mean and sd of difficulty, slope and precision per benchmark."""
        X = self.draws(rng, n)
        _, d, a, k = self.split(X)
        return {"difficulty_mean": d.mean(0), "difficulty_sd": d.std(0),
                "slope_mean": a.mean(0), "slope_sd": a.std(0),
                "precision_mean": k.mean(0), "precision_sd": k.std(0)}

    def model_summary(self, rng: np.random.Generator, n: int = 1000) -> dict[str, np.ndarray]:
        X = self.draws(rng, n)
        th = self.split(X)[0]
        return {"capability_mean": th.mean(0), "capability_sd": th.std(0)}


def default_methods(quick: bool = False):
    """The five graded methods; ``quick`` shortens the sampler for cross-validation folds."""
    hb = HierBeta(n_warmup=150, n_samples=150) if quick else HierBeta()
    return [BenchmarkMean(), MeanRank(), LowRank1(), EciLS(), hb]
