from __future__ import annotations

import numpy as np
import pandas as pd


class GBMSimulator:
    def __init__(self, posterior: pd.DataFrame, s0: float, dt: float = 1 / 252, seed: int = 42):
        self.posterior = posterior
        self.s0 = float(s0)
        self.dt = dt
        self.seed = seed

    def simulate_exact(self, horizon_days: int = 60, paths: int = 1500) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        draws = self.posterior.sample(paths, replace=len(self.posterior) < paths, random_state=self.seed)
        mu = draws["mu"].values
        sigma = draws["sigma"].values
        z = rng.standard_normal((paths, horizon_days))
        increments = (mu[:, None] - 0.5 * sigma[:, None] ** 2) * self.dt + sigma[:, None] * np.sqrt(self.dt) * z
        log_paths = np.cumsum(increments, axis=1)
        prices = np.empty((paths, horizon_days + 1))
        prices[:, 0] = self.s0
        prices[:, 1:] = self.s0 * np.exp(log_paths)
        columns = [f"day_{i}" for i in range(horizon_days + 1)]
        return pd.DataFrame(prices, columns=columns)

    @staticmethod
    def percentiles(paths: pd.DataFrame, levels=(1, 5, 25, 50, 75, 95, 99)) -> pd.DataFrame:
        arr = np.percentile(paths.values, levels, axis=0)
        out = pd.DataFrame(arr.T, columns=[f"p{p}" for p in levels])
        out["day"] = np.arange(len(out))
        return out

    @staticmethod
    def density_surface(paths: pd.DataFrame, bins: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = paths.values
        day_count = arr.shape[1]
        price_min, price_max = np.nanpercentile(arr, [1, 99])
        y_edges = np.linspace(price_min, price_max, bins + 1)
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z = np.zeros((bins, day_count))
        for day in range(day_count):
            hist, _ = np.histogram(arr[:, day], bins=y_edges, density=True)
            z[:, day] = hist
        x = np.arange(day_count)
        return x, y_centers, z
