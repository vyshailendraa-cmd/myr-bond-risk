"""Generate the one-page portfolio factsheet PDF into output/factsheet.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CURVE_CSV, OUT, load
from bondrisk.factsheet import build_factsheet
from bondrisk.mps import portfolio_impact
from bondrisk.portfolio import portfolio_summary, value_portfolio
from bondrisk.stress import key_rate_durations, run_scenarios

market, curve, _, holdings = load()
book = value_portfolio(holdings, curve)
st = run_scenarios(holdings, curve)
path = build_factsheet(OUT / "factsheet.pdf", curve, market, book, portfolio_summary(book), st,
                       key_rate_durations(holdings, curve), as_of=CURVE_CSV.stem.split("_")[-1],
                       commentary=portfolio_impact({"next_move_bias": "hold"}, st))
print("Saved", path)
