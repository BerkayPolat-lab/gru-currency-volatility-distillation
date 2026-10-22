"""Download and freeze the FRED DEXUSEU snapshot, drop unobserved dates, compute log returns."""

import numpy as np
import pandas as pd


def load_prices(cfg):
    """Observed DEXUSEU prices in the configured date range. Missing dates are dropped, never interpolated."""
    d = cfg["data"]
    prices = pd.read_csv(d["snapshot_path"], parse_dates=["observation_date"], index_col="observation_date")
    return prices["DEXUSEU"].loc[d["start"]:d["end"]].dropna()


def log_returns(prices):
    """r(t) = log(P(t) / P(t-1)) over consecutive observed prices."""
    return np.log(prices).diff().dropna().rename("log_return")


def load_examples(path="data/processed/examples.csv"):
    """The processed example table written by notebook 01: one row per forecast origin."""
    return pd.read_csv(path, parse_dates=["date", "target_end"], index_col="date")