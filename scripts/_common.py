"""Shared setup: fit the curve and load the sample portfolio."""
from pathlib import Path

import pandas as pd

from bondrisk.curve import fit_nelson_siegel
from bondrisk.portfolio import load_holdings

ROOT = Path(__file__).resolve().parents[1]
CURVE_CSV = ROOT / "data" / "mgs_curve_2026-11-18.csv"
PORTFOLIO_CSV = ROOT / "data" / "sample_portfolio.csv"
HISTORY_CSV = ROOT / "data" / "mgs_history.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def load():
    market = pd.read_csv(CURVE_CSV)
    curve, rmse = fit_nelson_siegel(market["maturity_years"], market["yield_pct"] / 100)
    holdings = load_holdings(PORTFOLIO_CSV)
    return market, curve, rmse, holdings
