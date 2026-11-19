"""Walk-forward forecast backtest: random walk vs Diebold-Li AR(1) vs gradient boosting.

Uses data/mgs_history.csv (real BNM data from fetch_history.py) if it exists.
Otherwise falls back to SIMULATED history, clearly labelled, for testing only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import HISTORY_CSV, OUT
from bondrisk.data import simulate_history
from bondrisk.forecast import summarise, walk_forward

if HISTORY_CSV.exists():
    daily, label = pd.read_csv(HISTORY_CSV), "real BNM history"
else:
    daily, label = simulate_history(), "SIMULATED history (not real data)"
    print("WARNING: no data/mgs_history.csv found, using simulated data. Run scripts/fetch_history.py first.\n")

res = walk_forward(daily)
summ = summarise(res)
res.to_csv(OUT / "backtest_by_tenor.csv", index=False)
summ.reset_index().to_csv(OUT / "backtest_summary.csv", index=False)
pd.set_option("display.width", 160)
print(f"Data: {label}\nAverage RMSE across tenors (bp):\n")
print(summ.round(2).to_string())

models = ["Random walk", "Diebold-Li AR(1)", "Gradient boosting"]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
w = 0.27
for i, m in enumerate(models):
    ax.bar([j + (i - 1) * w for j in range(len(summ))], summ[m], w, label=m,
           color=["#7f8c8d", "#1f3a5f", "#e67e22"][i])
ax.set_xticks(range(len(summ)), [f"{h} week{'s' if h > 1 else ''}" for h in summ.index])
ax.set_ylabel("Forecast error, RMSE (bp)")
ax.set_title(f"Yield forecast error by horizon ({label})")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "backtest.png")
print(f"\nSaved {OUT/'backtest.png'}")
