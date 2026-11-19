"""Fit the MGS curve, value the sample portfolio and run OPR stress tests.

Usage:  python scripts/run_report.py
Outputs: tables in the terminal, charts in output/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bondrisk.curve import fit_nelson_siegel
from bondrisk.portfolio import load_holdings, portfolio_summary, value_portfolio
from bondrisk.stress import SHAPE_SCENARIOS, key_rate_durations, opr_shift, run_scenarios, scenario_pnl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

# 1. Fit the curve -------------------------------------------------------
mkt = pd.read_csv(ROOT / "data" / "mgs_curve_2026-11-18.csv")
curve, rmse = fit_nelson_siegel(mkt["maturity_years"], mkt["yield_pct"] / 100)
print("=== Nelson-Siegel fit to the MGS curve ===")
print(f"level b0={curve.b0*100:.2f}%  slope b1={curve.b1*100:.2f}%  "
      f"curvature b2={curve.b2*100:.2f}%  tau={curve.tau:.2f}y  RMSE={rmse:.1f}bp\n")

# 2. Value the portfolio -------------------------------------------------
holdings = load_holdings(ROOT / "data" / "sample_portfolio.csv")
book = value_portfolio(holdings, curve)
summ = portfolio_summary(book)
print("=== Portfolio ===")
print(book[["name", "yield_%", "price", "market_value_rm", "mod_duration", "convexity", "dv01_rm", "weight_%"]]
      .to_string(index=False))
print(f"\nTotal MV RM{summ['market_value_rm']:,.0f} | yield {summ['yield_%']:.2f}% | "
      f"mod. duration {summ['mod_duration']:.2f} | DV01 RM{summ['dv01_rm']:,.0f}\n")

# 3. Stress tests --------------------------------------------------------
st = run_scenarios(holdings, curve)
print("=== OPR stress tests ===")
print(st.pivot(index="shock_bp", columns="scenario", values="pnl_rm").map(lambda v: f"RM{v:,.0f}").to_string())
print()
print("=== Curve-shape scenarios ===")
print(scenario_pnl(holdings, curve, SHAPE_SCENARIOS).to_string(index=False))
print()
print("=== Key-rate DV01 (RM per 1bp rise at each tenor) ===")
krd = key_rate_durations(holdings, curve)
print(krd.to_string(index=False), f"\nSum RM{krd['krd_dv01_rm'].sum():,.0f} vs parallel DV01 RM{summ['dv01_rm']:,.0f}\n")

# Charts -----------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
tt = np.linspace(0.1, 10.5, 300)
ax.plot(tt, curve(tt) * 100, lw=2.2, color="#1f3a5f", label="Nelson-Siegel fit")
ax.plot(tt, curve.shifted(50)(tt) * 100, lw=1.2, ls="--", color="#c0392b", label="+50bp parallel")
ax.plot(tt, (curve(tt) + np.array([opr_shift(t, 50) for t in tt]) / 1e4) * 100,
        lw=1.2, ls=":", color="#e67e22", label="+50bp OPR (pass-through)")
ax.scatter(mkt["maturity_years"], mkt["yield_pct"], s=45, color="#111", zorder=5, label="Market (BNM)")
ax.set_xlabel("Years to maturity")
ax.set_ylabel("Yield (%)")
ax.set_title("Malaysian government yield curve, 18 Nov 2026")
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(OUT / "yield_curve.png")

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
piv = st.pivot(index="shock_bp", columns="scenario", values="pnl_%")
x = np.arange(len(piv))
ax.bar(x - 0.2, piv["parallel"], 0.4, color="#1f3a5f", label="Parallel shift")
ax.bar(x + 0.2, piv["OPR pass-through"], 0.4, color="#e67e22", label="OPR pass-through")
ax.axhline(0, color="#111", lw=0.8)
ax.set_xticks(x, [f"{s:+d}bp" for s in piv.index])
ax.set_ylabel("Portfolio P&L (%)")
ax.set_title("Portfolio P&L under OPR shocks")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "stress_test.png")
print(f"Charts saved to {OUT}")
