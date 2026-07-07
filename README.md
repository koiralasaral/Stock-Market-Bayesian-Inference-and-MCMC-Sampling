# Shopify Bayesian Quantitative Finance Dashboard

An object-oriented Streamlit dashboard for Shopify (`SHOP`) using Yahoo Finance data, PyMC Bayesian MCMC, and Plotly.

## Features

- Candlestick and volume chart
- Bayesian GBM regression with two samplers: NUTS and Metropolis-Hastings
- Posterior distributions and joint posterior plots
- Trace plots and autocorrelation
- Exact Ito GBM Monte Carlo simulation
- Interactive 3D forecast price density surface
- Risk metrics: VaR, CVaR, Sharpe, Sortino, maximum drawdown, Calmar, Omega
- Posterior predictive checks and diagnostics: ESS, R-hat, divergences, energy

## Model

The dashboard fits the statistically correct GBM increment likelihood:

```text
r_i = log(S_i / S_{i-1}) ~ Normal((mu - 0.5 sigma^2) dt, sigma sqrt(dt))
```

This avoids treating cumulative log prices as independent observations. The cumulative Ito path equation is still used for interpretation and forward simulation:

```text
S_{t+1} = S_t exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z_t)
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

For a faster first run, set `MCMC draws` and `Tune steps` to `250` in the sidebar. Increase them for final analysis.

## Files

- `app.py`: Streamlit dashboard controller
- `data.py`: Yahoo Finance data loader
- `bayesian.py`: PyMC GBM regression with NUTS and Metropolis-Hastings
- `gbm.py`: exact Ito Monte Carlo engine
- `risk.py`: risk analytics
- `diagnostics.py`: ESS, R-hat, trace, ACF, energy helpers
- `plots.py`: Plotly visualization factory
- `config.py`: default dashboard settings
