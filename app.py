from __future__ import annotations

import arviz as az
import pandas as pd
import streamlit as st

from bayesian import BayesianGBMRegression
from config import DashboardConfig
from data import MarketDataLoader
from diagnostics import BayesianDiagnostics
from gbm import GBMSimulator
from plots import PlotlyVisualizer
from risk import RiskAnalyzer


class ShopifyBayesianDashboard:
    def __init__(self):
        self.config = DashboardConfig()
        self.visuals = PlotlyVisualizer()

    def sidebar(self) -> DashboardConfig:
        st.sidebar.header("Controls")
        self.config.ticker = st.sidebar.text_input("Ticker", self.config.ticker).upper()
        self.config.period = st.sidebar.selectbox("History", ["6mo", "1y", "2y", "5y", "10y"], index=2)
        self.config.forecast_days = st.sidebar.slider("Forecast days", 10, 252, self.config.forecast_days, 5)
        self.config.paths = st.sidebar.slider("Monte Carlo paths", 250, 10000, self.config.paths, 250)
        st.sidebar.divider()
        self.config.draws = st.sidebar.slider("MCMC draws", 250, 5000, self.config.draws, 250)
        self.config.tune = st.sidebar.slider("Tune steps", 250, 5000, self.config.tune, 250)
        self.config.chains = st.sidebar.select_slider("Chains", options=[1, 2, 3, 4], value=self.config.chains)
        self.config.target_accept = st.sidebar.slider("NUTS target accept", 0.80, 0.99, self.config.target_accept, 0.01)
        self.config.confidence = st.sidebar.slider("Risk confidence", 0.90, 0.99, self.config.confidence, 0.01)
        self.config.risk_free_rate = st.sidebar.number_input("Risk-free rate", value=self.config.risk_free_rate, step=0.005, format="%.3f")
        self.config.seed = st.sidebar.number_input("Random seed", value=self.config.seed, step=1)
        return self.config

    @st.cache_resource(show_spinner="Running Bayesian MCMC: NUTS and Metropolis-Hastings...")
    def fit_models(_self, returns_tuple, draws, tune, chains, target_accept, seed):
        returns = pd.Series(list(returns_tuple))
        model = BayesianGBMRegression(returns, seed=seed)
        return model.fit(draws=draws, tune=tune, chains=chains, target_accept=target_accept)

    def run(self) -> None:
        st.set_page_config(page_title="Shopify Bayesian Quant Dashboard", layout="wide")
        cfg = self.sidebar()
        st.title("Shopify Bayesian Quantitative Finance Dashboard")

        loader = MarketDataLoader(cfg.ticker, cfg.period, cfg.interval)
        try:
            prices_df = loader.fetch()
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        close = prices_df["Close"]
        log_returns = loader.log_returns(close)
        simple_returns = loader.simple_returns(close)

        nuts, mh = self.fit_models(
            tuple(log_returns.round(12).values),
            cfg.draws,
            cfg.tune,
            cfg.chains,
            cfg.target_accept,
            cfg.seed,
        )

        posterior = pd.concat(
            [BayesianGBMRegression.posterior_frame(nuts), BayesianGBMRegression.posterior_frame(mh)],
            ignore_index=True,
        )
        nuts_diag = BayesianDiagnostics(nuts)
        mh_diag = BayesianDiagnostics(mh)
        simulator = GBMSimulator(
            posterior=posterior.loc[posterior["sampler"] == "NUTS"],
            s0=float(close.iloc[-1]),
            dt=cfg.dt,
            seed=cfg.seed,
        )
        paths = simulator.simulate_exact(cfg.forecast_days, cfg.paths)
        risk = RiskAnalyzer(simple_returns, close, cfg.risk_free_rate)
        risk_table = risk.table(cfg.confidence)
        sim_var, sim_cvar = RiskAnalyzer.simulated_var(paths, cfg.confidence)
        risk_table = pd.concat(
            [
                risk_table,
                pd.DataFrame(
                    {
                        "metric": ["Bayesian Forecast VaR", "Bayesian Forecast CVaR"],
                        "value": [sim_var, sim_cvar],
                    }
                ),
            ],
            ignore_index=True,
        )

        self.header_metrics(close, log_returns, nuts, mh, risk_table)

        tabs = st.tabs(
            [
                "Candlestick + Volume",
                "Bayesian Regression",
                "Posterior Distributions",
                "Trace + Autocorrelation",
                "GBM Monte Carlo",
                "3D Density Surface",
                "Risk Metrics",
                "PPC + Diagnostics",
            ]
        )

        with tabs[0]:
            st.plotly_chart(self.visuals.candlestick_volume(prices_df, cfg.ticker), use_container_width=True)
            st.dataframe(prices_df.tail(20), use_container_width=True)

        with tabs[1]:
            st.plotly_chart(self.visuals.regression_comparison(close, posterior, cfg.dt), use_container_width=True)
            summaries = pd.concat([BayesianGBMRegression.summary(nuts), BayesianGBMRegression.summary(mh)])
            st.dataframe(summaries, use_container_width=True)
            st.write("Classical MLE cross-check", BayesianGBMRegression.mle_crosscheck(log_returns, cfg.dt))

        with tabs[2]:
            st.plotly_chart(self.visuals.posterior_distributions(posterior), use_container_width=True)
            st.plotly_chart(self.visuals.posterior_pair(posterior), use_container_width=True)
            st.download_button("Download posterior samples", posterior.to_csv(index=False), "posterior_samples.csv")

        with tabs[3]:
            parameter = st.selectbox("Parameter", ["mu", "sigma", "alpha"])
            st.plotly_chart(
                self.visuals.trace_and_acf(
                    {"NUTS": nuts_diag.trace_frame(parameter), "Metropolis-Hastings": mh_diag.trace_frame(parameter)},
                    {"NUTS": nuts_diag.autocorrelation_frame(parameter), "Metropolis-Hastings": mh_diag.autocorrelation_frame(parameter)},
                    parameter,
                ),
                use_container_width=True,
            )

        with tabs[4]:
            st.plotly_chart(self.visuals.monte_carlo_fan(paths, cfg.ticker), use_container_width=True)
            st.download_button("Download GBM simulations", paths.to_csv(index=False), "gbm_simulations.csv")

        with tabs[5]:
            x, y, z = GBMSimulator.density_surface(paths)
            st.plotly_chart(self.visuals.price_density_surface(x, y, z), use_container_width=True)

        with tabs[6]:
            st.plotly_chart(self.visuals.risk_metrics(risk_table), use_container_width=True)
            st.dataframe(risk_table, use_container_width=True)

        with tabs[7]:
            ppc = {
                "NUTS": BayesianGBMRegression.posterior_predictive_returns(nuts, dt=cfg.dt, seed=cfg.seed),
                "Metropolis-Hastings": BayesianGBMRegression.posterior_predictive_returns(mh, dt=cfg.dt, seed=cfg.seed + 1),
            }
            st.plotly_chart(self.visuals.posterior_predictive(log_returns, ppc), use_container_width=True)
            st.subheader("ESS and R-hat")
            st.dataframe(
                pd.concat(
                    [
                        nuts_diag.ess_rhat().assign(sampler="NUTS"),
                        mh_diag.ess_rhat().assign(sampler="Metropolis-Hastings"),
                    ]
                ),
                use_container_width=True,
            )
            st.plotly_chart(self.visuals.energy({"NUTS": nuts_diag.energy_frame(), "Metropolis-Hastings": mh_diag.energy_frame()}), use_container_width=True)
            st.write(
                {
                    "nuts_divergences": nuts_diag.divergences(),
                    "mh_divergences": mh_diag.divergences(),
                    "arviz_loo_available": hasattr(az, "loo"),
                }
            )

    @staticmethod
    def header_metrics(close, log_returns, nuts, mh, risk_table):
        cols = st.columns(5)
        cols[0].metric("Last price", f"${close.iloc[-1]:,.2f}")
        cols[1].metric("Daily log return", f"{log_returns.iloc[-1] * 100:.2f}%")
        cols[2].metric("NUTS runtime", f"{nuts.runtime_seconds:.1f}s")
        cols[3].metric("MH runtime", f"{mh.runtime_seconds:.1f}s")
        var = risk_table.loc[risk_table["metric"] == "Historical VaR", "value"].iloc[0]
        cols[4].metric("Historical VaR", f"{var * 100:.2f}%")


if __name__ == "__main__":
    ShopifyBayesianDashboard().run()
