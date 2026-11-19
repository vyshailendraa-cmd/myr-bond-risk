"""Value a bond portfolio off a fitted curve plus a credit spread per holding."""
from __future__ import annotations

import pandas as pd

from .bond import Bond, convexity, dv01, modified_duration, price
from .curve import NelsonSiegel


def load_holdings(path: str) -> list[tuple[Bond, float, float, str]]:
    """CSV columns: name, type, coupon_pct, maturity_years, spread_bp, face_rm."""
    df = pd.read_csv(path)
    out = []
    for r in df.itertuples():
        bond = Bond(r.name, r.coupon_pct / 100, float(r.maturity_years))
        out.append((bond, float(r.spread_bp), float(r.face_rm), r.type))
    return out


def value_portfolio(holdings, curve: NelsonSiegel) -> pd.DataFrame:
    rows = []
    for bond, spread_bp, face_rm, kind in holdings:
        y = float(curve(bond.maturity)[0]) + spread_bp / 1e4
        p = price(bond, y)
        mv = p / 100 * face_rm
        rows.append({
            "name": bond.name,
            "type": kind,
            "maturity_y": bond.maturity,
            "coupon_%": bond.coupon * 100,
            "yield_%": y * 100,
            "price": p,
            "market_value_rm": mv,
            "mod_duration": modified_duration(bond, y),
            "convexity": convexity(bond, y),
            "dv01_rm": dv01(bond, y) / 100 * face_rm,
        })
    df = pd.DataFrame(rows)
    df["weight_%"] = df["market_value_rm"] / df["market_value_rm"].sum() * 100
    return df


def portfolio_summary(df: pd.DataFrame) -> dict:
    w = df["market_value_rm"] / df["market_value_rm"].sum()
    return {
        "market_value_rm": df["market_value_rm"].sum(),
        "yield_%": float((w * df["yield_%"]).sum()),
        "mod_duration": float((w * df["mod_duration"]).sum()),
        "convexity": float((w * df["convexity"]).sum()),
        "dv01_rm": df["dv01_rm"].sum(),
    }
