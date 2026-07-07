from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from bayesian import SamplerResult


class BayesianDiagnostics:
    def __init__(self, result: SamplerResult):
        self.result = result
        self.trace = result.trace

    def summary(self) -> pd.DataFrame:
        out = az.summary(self.trace, var_names=["mu", "sigma", "alpha"], round_to=5)
        return out.reset_index(names="parameter")

    def ess_rhat(self) -> pd.DataFrame:
        summary = self.summary()
        cols = [c for c in ["parameter", "ess_bulk", "ess_tail", "r_hat"] if c in summary.columns]
        return summary[cols]

    def trace_frame(self, var_name: str) -> pd.DataFrame:
        arr = self.trace.posterior[var_name]
        frames = []
        for chain in arr.chain.values:
            values = arr.sel(chain=chain).values.reshape(-1)
            frames.append(pd.DataFrame({"draw": np.arange(len(values)), "value": values, "chain": str(chain)}))
        return pd.concat(frames, ignore_index=True)

    def autocorrelation_frame(self, var_name: str, nlags: int = 80) -> pd.DataFrame:
        values = self.trace.posterior[var_name].values.reshape(-1)
        vals = acf(values, nlags=nlags, fft=True)
        return pd.DataFrame({"lag": np.arange(len(vals)), "acf": vals, "parameter": var_name})

    def energy_frame(self) -> pd.DataFrame:
        if "energy" not in self.trace.sample_stats:
            return pd.DataFrame(columns=["draw", "energy", "chain"])
        arr = self.trace.sample_stats["energy"]
        frames = []
        for chain in arr.chain.values:
            values = arr.sel(chain=chain).values.reshape(-1)
            frames.append(pd.DataFrame({"draw": np.arange(len(values)), "energy": values, "chain": str(chain)}))
        return pd.concat(frames, ignore_index=True)

    def divergences(self) -> int:
        if "diverging" not in self.trace.sample_stats:
            return 0
        return int(self.trace.sample_stats["diverging"].sum())
