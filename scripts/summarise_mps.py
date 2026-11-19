"""AI summary of a BNM Monetary Policy Statement, linked to the portfolio.

  set ANTHROPIC_API_KEY=...            (PowerShell: $env:ANTHROPIC_API_KEY="...")
  python scripts/summarise_mps.py path/to/statement.txt
  python scripts/summarise_mps.py https://www.bnm.gov.my/-/monetary-policy-statement-...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load
from bondrisk.mps import fetch_text, portfolio_impact, read_statement
from bondrisk.stress import run_scenarios

if len(sys.argv) < 2:
    sys.exit(__doc__)
_, curve, _, holdings = load()
reading = read_statement(fetch_text(sys.argv[1]))
print(json.dumps(reading, indent=2))
print("\n" + portfolio_impact(reading, run_scenarios(holdings, curve)))
