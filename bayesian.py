from __future__ import annotations

import time
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


@dataclass
class SamplerResult:
    name: str
    trace: az.InferenceData
    runtime_seconds: float


class BayesianGBMRegression:
    def __init__(self, returns: pd.Series, dt: float = 1 / 252, seed: int = 42):
        self.returns = returns.dropna().astype(float)
        self.dt = dt
        self.seed = seed
        self.model: pm.Model | None = None
        self.nuts: SamplerResult | None = None
        self.metropolis: SamplerResult | None = None

    def build_model(self) -> pm.Model:
        r = self.returns.values
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0.0, sigma=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            alpha = pm.Deterministic("alpha", mu - 0.5 * sigma**2)
            r_mean = alpha * self.dt
            r_sd = sigma * np.sqrt(self.dt)
            pm.Normal("r_obs", mu=r_mean, sigma=r_sd, observed=r)
        self.model = model
        return model

    def _ensure_model(self) -> pm.Model:
        return self.model if self.model is not None else self.build_model()

    def fit_nuts(self, draws: int, tune: int, chains: int, target_accept: float) -> SamplerResult:
        model = self._ensure_model()
        started = time.perf_counter()
        with model:
            try:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=self.seed,
                    progressbar=True,
                )
            except ZeroDivisionError:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=1,
                    target_accept=target_accept,
                    random_seed=self.seed,
                    progressbar=True,
                )
        self.nuts = SamplerResult("NUTS", trace, time.perf_counter() - started)
        return self.nuts

    def fit_metropolis(self, draws: int, tune: int, chains: int) -> SamplerResult:
        model = self._ensure_model()
        started = time.perf_counter()
        with model:
            step = pm.Metropolis()
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                step=step,
                random_seed=self.seed,
                progressbar=True,
            )
        self.metropolis = SamplerResult("Metropolis-Hastings", trace, time.perf_counter() - started)
        return self.metropolis

    def fit(self, draws: int, tune: int, chains: int, target_accept: float) -> tuple[SamplerResult, SamplerResult]:
        return (
            self.fit_nuts(draws, tune, chains, target_accept),
            self.fit_metropolis(draws, tune, chains),
        )

    @staticmethod
    def posterior_frame(result: SamplerResult) -> pd.DataFrame:
        post = result.trace.posterior
        return pd.DataFrame(
            {
                "sampler": result.name,
                "mu": post["mu"].values.reshape(-1),
                "sigma": post["sigma"].values.reshape(-1),
                "alpha": post["alpha"].values.reshape(-1),
            }
        )

    @staticmethod
    def summary(result: SamplerResult) -> pd.DataFrame:
        summary = az.summary(result.trace, var_names=["mu", "sigma", "alpha"], round_to=5)
        summary.insert(0, "sampler", result.name)
        summary["runtime_seconds"] = result.runtime_seconds
        if "diverging" in result.trace.sample_stats:
            summary["divergences"] = int(result.trace.sample_stats["diverging"].sum())
        return summary

    @staticmethod
    def mle_crosscheck(returns: pd.Series, dt: float = 1 / 252) -> dict[str, float]:
        r = returns.values
        sigma = r.std(ddof=1) / np.sqrt(dt)
        mu = r.mean() / dt + 0.5 * sigma**2
        return {"mu_mle": float(mu), "sigma_mle": float(sigma)}

    @staticmethod
    def posterior_predictive_returns(result: SamplerResult, n_draws: int = 1500, dt: float = 1 / 252, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        samples = BayesianGBMRegression.posterior_frame(result)
        sample = samples.sample(min(n_draws, len(samples)), random_state=seed)
        means = (sample["mu"].values - 0.5 * sample["sigma"].values**2) * dt
        sds = sample["sigma"].values * np.sqrt(dt)
        return rng.normal(means, sds)
