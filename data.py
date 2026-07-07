from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf


class MarketDataLoader:
    def __init__(self, ticker: str, period: str, interval: str = "1d"):
        self.ticker = ticker.upper()
        self.period = period
        self.interval = interval

    @st.cache_data(show_spinner=False, ttl=60 * 30)
    def fetch(_self) -> pd.DataFrame:
        df = yf.download(
            _self.ticker,
            period=_self.period,
            interval=_self.interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if df.empty:
            raise RuntimeError(f"No Yahoo Finance data returned for {_self.ticker}.")
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(_self.ticker, axis=1, level=-1, drop_level=True)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out = df[keep].dropna().copy()
        out.index = pd.to_datetime(out.index)
        return out

    @staticmethod
    def log_returns(prices: pd.Series) -> pd.Series:
        s = prices.astype(float)
        returns = np.log(s / s.shift(1))
        returns.name = "log_return"
        return returns.dropna()

    @staticmethod
    def simple_returns(prices: pd.Series) -> pd.Series:
        returns = prices.astype(float).pct_change()
        returns.name = "simple_return"
        return returns.dropna()
