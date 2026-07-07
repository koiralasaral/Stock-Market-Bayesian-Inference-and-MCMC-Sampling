from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PlotlyVisualizer:
    template = "plotly_white"

    def candlestick_volume(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.72, 0.28],
        )
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#7f8c8d"), row=2, col=1)
        fig.update_layout(
            title=f"{ticker} Candlestick and Volume",
            template=self.template,
            height=680,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        return fig

    def regression_comparison(self, prices: pd.Series, posterior: pd.DataFrame, dt: float) -> go.Figure:
        s = prices.values.astype(float)
        t = np.arange(1, len(s)) * dt
        y = np.log(s[1:] / s[0])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Observed log(S_t/S0)", line=dict(color="#264653")))
        colors = {"NUTS": "#2a9d8f", "Metropolis-Hastings": "#e76f51"}
        for sampler, group in posterior.groupby("sampler"):
            alpha = group["alpha"].mean()
            sigma = group["sigma"].mean()
            band = sigma * np.sqrt(t)
            fig.add_trace(go.Scatter(x=t, y=alpha * t, mode="lines", name=f"{sampler} posterior mean", line=dict(color=colors.get(sampler))))
            fig.add_trace(
                go.Scatter(
                    x=np.r_[t, t[::-1]],
                    y=np.r_[alpha * t + 2 * band, (alpha * t - 2 * band)[::-1]],
                    fill="toself",
                    fillcolor="rgba(42,157,143,0.10)" if sampler == "NUTS" else "rgba(231,111,81,0.10)",
                    line=dict(width=0),
                    name=f"{sampler} +/-2 Ito SD",
                )
            )
        fig.update_layout(title="Bayesian GBM Regression: NUTS vs Metropolis-Hastings", xaxis_title="Years", yaxis_title="Log relative price", template=self.template, height=620)
        return fig

    def posterior_distributions(self, posterior: pd.DataFrame) -> go.Figure:
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Drift mu", "Volatility sigma", "Log-price slope alpha"])
        for i, var in enumerate(["mu", "sigma", "alpha"], start=1):
            for sampler in posterior["sampler"].unique():
                vals = posterior.loc[posterior["sampler"] == sampler, var]
                fig.add_trace(go.Histogram(x=vals, histnorm="probability density", opacity=0.55, name=f"{sampler} {var}"), row=1, col=i)
        fig.update_layout(barmode="overlay", template=self.template, height=520, title="Posterior Distributions")
        return fig

    def posterior_pair(self, posterior: pd.DataFrame) -> go.Figure:
        return px.scatter(
            posterior.sample(min(4000, len(posterior)), random_state=7),
            x="mu",
            y="sigma",
            color="sampler",
            opacity=0.35,
            template=self.template,
            title="Joint Posterior: Drift vs Volatility",
        )

    def trace_and_acf(self, trace_frames: dict[str, pd.DataFrame], acf_frames: dict[str, pd.DataFrame], parameter: str) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{parameter} Trace", f"{parameter} Autocorrelation"])
        for sampler, frame in trace_frames.items():
            for chain, chain_frame in frame.groupby("chain"):
                fig.add_trace(go.Scatter(x=chain_frame["draw"], y=chain_frame["value"], mode="lines", name=f"{sampler} chain {chain}", line=dict(width=1)), row=1, col=1)
        for sampler, frame in acf_frames.items():
            fig.add_trace(go.Bar(x=frame["lag"], y=frame["acf"], name=f"{sampler} ACF", opacity=0.65), row=1, col=2)
        fig.update_layout(template=self.template, height=560, title="Trace Plots and Autocorrelation")
        return fig

    def monte_carlo_fan(self, paths: pd.DataFrame, ticker: str) -> go.Figure:
        pct = self._path_percentiles(paths)
        fig = go.Figure()
        x = pct["day"]
        fig.add_trace(go.Scatter(x=x, y=pct["p95"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=pct["p5"], fill="tonexty", fillcolor="rgba(42,157,143,0.18)", line=dict(width=0), name="90% interval"))
        fig.add_trace(go.Scatter(x=x, y=pct["p75"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=pct["p25"], fill="tonexty", fillcolor="rgba(42,157,143,0.30)", line=dict(width=0), name="50% interval"))
        fig.add_trace(go.Scatter(x=x, y=pct["p50"], mode="lines", name="Median", line=dict(color="#1d3557", width=3)))
        fig.update_layout(template=self.template, height=620, title=f"{ticker} GBM Monte Carlo Simulation Using Exact Ito Solution", xaxis_title="Forecast day", yaxis_title="Price")
        return fig

    def price_density_surface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> go.Figure:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Viridis", colorbar=dict(title="Density"))])
        fig.update_layout(template=self.template, height=680, title="Interactive 3D Price Density Surface", scene=dict(xaxis_title="Forecast day", yaxis_title="Price", zaxis_title="Density"))
        return fig

    def risk_metrics(self, table: pd.DataFrame) -> go.Figure:
        fig = go.Figure(go.Bar(x=table["value"], y=table["metric"], orientation="h", marker_color="#457b9d"))
        fig.update_layout(template=self.template, height=540, title="Risk Metrics", xaxis_title="Value", yaxis_title="")
        return fig

    def posterior_predictive(self, observed: pd.Series, ppc: dict[str, np.ndarray]) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=observed, histnorm="probability density", opacity=0.5, name="Observed returns", marker_color="#264653"))
        for sampler, values in ppc.items():
            fig.add_trace(go.Histogram(x=values, histnorm="probability density", opacity=0.45, name=f"{sampler} posterior predictive"))
        fig.update_layout(barmode="overlay", template=self.template, height=560, title="Posterior Predictive Checks")
        return fig

    def energy(self, energy_frames: dict[str, pd.DataFrame]) -> go.Figure:
        fig = go.Figure()
        for sampler, frame in energy_frames.items():
            if not frame.empty:
                fig.add_trace(go.Histogram(x=frame["energy"], histnorm="probability density", opacity=0.55, name=sampler))
        fig.update_layout(barmode="overlay", template=self.template, height=420, title="Energy Diagnostics")
        return fig

    @staticmethod
    def _path_percentiles(paths: pd.DataFrame) -> pd.DataFrame:
        levels = [5, 25, 50, 75, 95]
        arr = np.percentile(paths.values, levels, axis=0).T
        out = pd.DataFrame(arr, columns=[f"p{x}" for x in levels])
        out["day"] = np.arange(len(out))
        return out
