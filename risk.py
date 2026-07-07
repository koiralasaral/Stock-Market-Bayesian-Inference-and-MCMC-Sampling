from __future__ import annotations

import numpy as np
import pandas as pd


class RiskAnalyzer:
    def __init__(self, returns: pd.Series, prices: pd.Series, risk_free_rate: float = 0.04):
        self.returns = returns.dropna().astype(float)
        self.prices = prices.dropna().astype(float)
        self.risk_free_rate = risk_free_rate

    def var(self, confidence: float = 0.95) -> float:
        return float(np.quantile(self.returns, 1 - confidence))

    def cvar(self, confidence: float = 0.95) -> float:
        threshold = self.var(confidence)
        tail = self.returns[self.returns <= threshold]
        return float(tail.mean())

    def sharpe(self) -> float:
        excess = self.returns - self.risk_free_rate / 252
        return float(np.sqrt(252) * excess.mean() / excess.std(ddof=1))

    def sortino(self) -> float:
        excess = self.returns - self.risk_free_rate / 252
        downside = excess[excess < 0]
        return float(np.sqrt(252) * excess.mean() / downside.std(ddof=1))

    def max_drawdown(self) -> float:
        wealth = (1 + self.returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        return float(drawdown.min())

    def annual_return(self) -> float:
        return float((1 + self.returns.mean()) ** 252 - 1)

    def annual_volatility(self) -> float:
        return float(self.returns.std(ddof=1) * np.sqrt(252))

    def calmar(self) -> float:
        mdd = abs(self.max_drawdown())
        return float(self.annual_return() / mdd) if mdd else np.nan

    def omega(self, threshold: float = 0.0) -> float:
        gains = (self.returns - threshold).clip(lower=0).sum()
        losses = (threshold - self.returns).clip(lower=0).sum()
        return float(gains / losses) if losses else np.nan

    def table(self, confidence: float = 0.95) -> pd.DataFrame:
        metrics = {
            "Historical VaR": self.var(confidence),
            "Historical CVaR": self.cvar(confidence),
            "Sharpe Ratio": self.sharpe(),
            "Sortino Ratio": self.sortino(),
            "Maximum Drawdown": self.max_drawdown(),
            "Annual Return": self.annual_return(),
            "Annual Volatility": self.annual_volatility(),
            "Calmar Ratio": self.calmar(),
            "Omega Ratio": self.omega(),
        }
        return pd.DataFrame({"metric": metrics.keys(), "value": metrics.values()})

    @staticmethod
    def simulated_var(paths: pd.DataFrame, confidence: float = 0.95) -> tuple[float, float]:
        start = paths.iloc[:, 0].values
        end = paths.iloc[:, -1].values
        returns = end / start - 1
        var = float(np.quantile(returns, 1 - confidence))
        cvar = float(returns[returns <= var].mean())
        return var, cvar
