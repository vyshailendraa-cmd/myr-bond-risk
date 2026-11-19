"""REST API over the risk engine.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bondrisk.curve import fit_nelson_siegel  # noqa: E402
from bondrisk.portfolio import load_holdings, portfolio_summary, value_portfolio  # noqa: E402
from bondrisk.stress import (SHAPE_SCENARIOS, key_rate_durations, opr_shift,  # noqa: E402
                             run_scenarios, scenario_pnl)

CURVE_CSV = ROOT / "data" / "mgs_curve_2026-11-18.csv"
PORTFOLIO_CSV = ROOT / "data" / "sample_portfolio.csv"
HISTORY_CSV = ROOT / "data" / "mgs_history.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

app = FastAPI(title="MYR Bond Risk API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache
def state():
    market = pd.read_csv(CURVE_CSV)
    curve, rmse = fit_nelson_siegel(market["maturity_years"], market["yield_pct"] / 100)
    holdings = load_holdings(PORTFOLIO_CSV)
    return market, curve, rmse, holdings


def records(df: pd.DataFrame):
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/curve")
def get_curve(opr_shock_bp: float = 0):
    market, curve, rmse, _ = state()
    tt = np.round(np.linspace(0.25, 10.5, 60), 3)
    fitted = curve(tt) * 100
    shocked = fitted + np.array([opr_shift(t, opr_shock_bp) for t in tt]) / 100
    return {
        "as_of": CURVE_CSV.stem.split("_")[-1],
        "params": {"level_pct": curve.b0 * 100, "slope_pct": curve.b1 * 100,
                   "curvature_pct": curve.b2 * 100, "tau_years": curve.tau, "rmse_bp": rmse},
        "market": records(market),
        "fitted": [{"t": float(t), "fit": float(f), "shocked": float(s)} for t, f, s in zip(tt, fitted, shocked)],
    }


@app.get("/portfolio")
def get_portfolio():
    _, curve, _, holdings = state()
    book = value_portfolio(holdings, curve)
    return {"summary": portfolio_summary(book), "holdings": records(book)}


@app.get("/stress/opr")
def get_opr_stress():
    _, curve, _, holdings = state()
    return records(run_scenarios(holdings, curve))


@app.get("/stress/shapes")
def get_shape_stress():
    _, curve, _, holdings = state()
    return records(scenario_pnl(holdings, curve, SHAPE_SCENARIOS))


@app.get("/stress/key-rates")
def get_key_rates():
    _, curve, _, holdings = state()
    return records(key_rate_durations(holdings, curve))


@app.get("/stress/opr-shock")
def opr_shock(bp: float = 25):
    """P&L for any OPR move, using maturity-decaying pass-through and a parallel shift."""
    _, curve, _, holdings = state()
    df = scenario_pnl(holdings, curve, {"OPR pass-through": lambda t: opr_shift(t, bp), "parallel": lambda t: bp})
    return {"bp": bp, "results": records(df)}


class CustomShock(BaseModel):
    short_bp: float = Field(0, description="Yield change at 2 years, bp")
    long_bp: float = Field(0, description="Yield change at 10 years, bp")


@app.post("/stress/custom")
def custom_shock(s: CustomShock):
    from bondrisk.stress import _pivot
    _, curve, _, holdings = state()
    return records(scenario_pnl(holdings, curve, {"custom": _pivot(s.short_bp, s.long_bp)}))[0]


@app.get("/forecast/backtest")
def get_backtest():
    path = OUT / "backtest_summary.csv"
    if not path.exists():
        raise HTTPException(404, "Run `python scripts/run_backtest.py` first.")
    df = pd.read_csv(path)
    return {"source": "real BNM history" if HISTORY_CSV.exists() else "SIMULATED history (not real data)",
            "rows": records(df)}


@app.get("/factsheet.pdf")
def get_factsheet():
    from bondrisk.factsheet import build_factsheet
    from bondrisk.mps import portfolio_impact
    market, curve, _, holdings = state()
    book = value_portfolio(holdings, curve)
    st = run_scenarios(holdings, curve)
    path = OUT / "factsheet.pdf"
    build_factsheet(path, curve, market, book, portfolio_summary(book), st,
                    key_rate_durations(holdings, curve), as_of=CURVE_CSV.stem.split("_")[-1],
                    commentary=portfolio_impact({"next_move_bias": "hold"}, st))
    return FileResponse(path, media_type="application/pdf", filename="factsheet.pdf")


class MPSRequest(BaseModel):
    text: str = Field(..., description="Full text of a BNM Monetary Policy Statement")


@app.post("/mps/summary")
def mps_summary(req: MPSRequest):
    from bondrisk.mps import portfolio_impact, read_statement
    _, curve, _, holdings = state()
    try:
        reading = read_statement(req.text)
    except Exception as e:
        raise HTTPException(502, f"AI summary failed: {e}")
    return {"reading": reading, "portfolio_impact": portfolio_impact(reading, run_scenarios(holdings, curve))}
