"""Bank Negara Malaysia Open API client.

Docs: https://apikijangportal.bnm.gov.my/openapi
- Base URL https://api.bnm.gov.my, no API key needed.
- Every request needs the header  Accept: application/vnd.BNM.API.v1+json
- Government securities yields: /public/gov-sec-yield (one trading day per call;
  non-trading days return no rows).
- Overnight Policy Rate: /public/opr

BNM's response layout isn't fully documented, so the parser below searches the
JSON for records that carry a tenure and a yield rather than relying on fixed
field names. Run `python scripts/fetch_history.py --probe` first: it prints one
raw response so you can confirm the fields before a long backfill.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://api.bnm.gov.my"
HEADERS = {"Accept": "application/vnd.BNM.API.v1+json"}
TENORS = [3, 5, 7, 10, 15, 20, 30]  # MGS benchmark tenors in years


def _get(path: str, timeout: float = 20.0):
    import httpx
    r = httpx.get(BASE + path, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _walk(obj):
    """Yield every dict nested anywhere in a JSON object."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _num(x):
    try:
        return float(str(x).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def parse_gov_sec_yield(payload) -> dict[int, float]:
    """Extract {tenor_years: yield_%} for conventional MGS benchmarks."""
    out: dict[int, float] = {}
    for rec in _walk(payload):
        keys = {k.lower(): k for k in rec}
        kind = " ".join(str(rec[keys[k]]) for k in keys if "type" in k or "security" in k).lower()
        if kind and ("gii" in kind or "islamic" in kind or "bill" in kind or "note" in kind):
            continue
        tenor_key = next((keys[k] for k in keys if "tenure" in k or "tenor" in k), None)
        yld_key = next((keys[k] for k in keys if k in ("closing", "close", "yield", "closing_yield")
                        or k.endswith("close") or k.endswith("yield")), None)
        if tenor_key is None or yld_key is None:
            continue
        m = re.search(r"(\d+)", str(rec[tenor_key]))
        y = _num(rec[yld_key])
        if m and y is not None:
            t = int(m.group(1))
            if t in TENORS:
                out.setdefault(t, y)
    return out


def fetch_curve(d: date) -> dict[int, float]:
    payload = _get(f"/public/gov-sec-yield/date/{d.isoformat()}")
    return parse_gov_sec_yield(payload)


def fetch_opr():
    return _get("/public/opr")


def fetch_history(start: date, end: date, out_csv: Path, pause: float = 0.3) -> pd.DataFrame:
    """Backfill daily MGS benchmark yields into a wide CSV (date x tenor).

    Resumes from an existing file, so it is safe to stop and restart.
    """
    out_csv = Path(out_csv)
    have = pd.read_csv(out_csv, parse_dates=["date"]) if out_csv.exists() else pd.DataFrame(columns=["date"])
    done = set(pd.to_datetime(have["date"]).dt.date) if len(have) else set()
    rows = have.to_dict("records")
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in done:
            try:
                curve = fetch_curve(d)
                if curve:
                    rows.append({"date": pd.Timestamp(d), **{f"y{t}": curve.get(t) for t in TENORS}})
                    print(d, curve)
            except Exception as e:  # network hiccup or non-trading day
                print(d, "skip:", e)
            time.sleep(pause)
            if len(rows) % 50 == 0:
                pd.DataFrame(rows).sort_values("date").to_csv(out_csv, index=False)
        d += timedelta(days=1)
    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(out_csv, index=False)
    return df


def probe(d: date) -> None:
    payload = _get(f"/public/gov-sec-yield/date/{d.isoformat()}")
    print(json.dumps(payload, indent=2)[:3000])
    print("\nParsed:", parse_gov_sec_yield(payload))


def simulate_history(n_days: int = 1500, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC daily MGS history for development and tests only.

    Simulates Nelson-Siegel factors as mean-reverting processes around the
    current MGS curve level, with occasional OPR-style jumps. It is not real
    data and must never be reported as a result.
    """
    import numpy as np
    from .curve import _loadings

    rng = np.random.default_rng(seed)
    tau = 1.5
    mu = np.array([0.0400, -0.0100, 0.0030])
    phi = np.array([0.997, 0.990, 0.985])
    sig = np.array([0.0004, 0.0005, 0.0008])
    f = mu.copy()
    dates = pd.bdate_range(end="2026-11-18", periods=n_days)
    X = _loadings(np.array(TENORS, float), tau)
    rows = []
    for d in dates:
        jump = rng.choice([0, 0.0025, -0.0025], p=[0.996, 0.002, 0.002])
        f = mu + phi * (f - mu) + sig * rng.standard_normal(3) + np.array([0.3, 0.7, 0]) * jump
        y = X @ f + 0.00015 * rng.standard_normal(len(TENORS))
        rows.append({"date": d, **{f"y{t}": round(v * 100, 4) for t, v in zip(TENORS, y)}})
    return pd.DataFrame(rows)
