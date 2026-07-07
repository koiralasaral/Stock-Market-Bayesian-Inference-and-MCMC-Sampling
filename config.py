from dataclasses import dataclass


@dataclass
class DashboardConfig:
    ticker: str = "SHOP"
    period: str = "2y"
    interval: str = "1d"
    dt: float = 1 / 252
    forecast_days: int = 60
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    target_accept: float = 0.92
    paths: int = 1500
    seed: int = 42
    confidence: float = 0.95
    risk_free_rate: float = 0.04
