"""Download daily MGS benchmark yields from the Bank Negara Malaysia Open API.

  python scripts/fetch_history.py --probe            # check one day's raw response first
  python scripts/fetch_history.py --years 5          # backfill 5 years (resumable)
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import HISTORY_CSV
from bondrisk.data import fetch_history, probe

p = argparse.ArgumentParser()
p.add_argument("--probe", action="store_true")
p.add_argument("--years", type=float, default=5)
a = p.parse_args()

if a.probe:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    probe(d)
else:
    end = date.today()
    fetch_history(end - timedelta(days=int(a.years * 365)), end, HISTORY_CSV)
    print(f"Saved {HISTORY_CSV}")
