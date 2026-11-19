"""Stress testing: OPR shocks, curve-shape scenarios and key-rate durations.

Every scenario is a function shift(t) -> basis points, applied on top of each
bond's current yield, after which the bond is fully repriced (no duration
approximation).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .bond import price
from .curve import NelsonSiegel

Shift = Callable[[float], float]


def opr_shift(t: float, opr_bp: float, long_end_passthrough: float = 0.4, decay: float = 3.0) -> float:
    """Yield change in bp at maturity t for an OPR move of opr_bp.

    Pass-through starts near 100% at the short end and decays towards
    long_end_passthrough. These parameters are illustrative assumptions.
    """
    w = long_end_passthrough + (1 - long_end_passthrough) * np.exp(-t / decay)
    return opr_bp * w


def _pivot(short_bp: float, long_bp: float, t_short: float = 2.0, t_long: float = 10.0) -> Shift:
    """Linear move from short_bp at t_short to long_bp at t_long, flat outside."""
    def f(t: float) -> float:
        w = min(max((t - t_short) / (t_long - t_short), 0.0), 1.0)
        return short_bp + w * (long_bp - short_bp)
    return f


# Named curve-shape scenarios (bp). Bear = yields up, bull = yields down.
SHAPE_SCENARIOS: dict[str, Shift] = {
    "Bear steepener (+10 short, +40 long)": _pivot(10, 40),
    "Bear flattener (+40 short, +10 long)": _pivot(40, 10),
    "Bull steepener (-40 short, -10 long)": _pivot(-40, -10),
    "Bull flattener (-10 short, -40 long)": _pivot(-10, -40),
    "Twist (short -20, long +20)": _pivot(-20, 20),
}


def base_positions(holdings, curve: NelsonSiegel):
    out = []
    for bond, spread_bp, face_rm, kind in holdings:
        y = float(curve(bond.maturity)[0]) + spread_bp / 1e4
        out.append((bond, y, face_rm, kind))
    return out


def revalue(positions, shift: Shift) -> float:
    return sum(price(b, y + shift(b.maturity) / 1e4) / 100 * f for b, y, f, _ in positions)


def scenario_pnl(holdings, curve: NelsonSiegel, scenarios: dict[str, Shift]) -> pd.DataFrame:
    pos = base_positions(holdings, curve)
    base = revalue(pos, lambda t: 0.0)
    rows = []
    for name, fn in scenarios.items():
        mv = revalue(pos, fn)
        rows.append({"scenario": name, "pnl_rm": mv - base, "pnl_%": (mv / base - 1) * 100})
    return pd.DataFrame(rows)


def run_scenarios(holdings, curve: NelsonSiegel, shocks_bp=(-100, -50, -25, 25, 50, 100)) -> pd.DataFrame:
    """OPR shocks, each run as a parallel shift and with maturity-decaying pass-through."""
    rows = []
    for s in shocks_bp:
        df = scenario_pnl(holdings, curve, {
            "parallel": lambda t, s=s: s,
            "OPR pass-through": lambda t, s=s: opr_shift(t, s),
        })
        df.insert(0, "shock_bp", s)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def key_rate_durations(holdings, curve: NelsonSiegel, key_tenors=(2, 5, 7, 10), bump_bp: float = 1.0) -> pd.DataFrame:
    """Ringgit P&L for a 1bp rise at each key tenor (triangular bumps).

    Shows WHERE on the curve the portfolio's rate risk sits. The key-rate
    DV01s add up to (almost exactly) the parallel DV01.
    """
    pos = base_positions(holdings, curve)
    base = revalue(pos, lambda t: 0.0)
    k = list(key_tenors)
    rows = []
    for i, kt in enumerate(k):
        left = k[i - 1] if i > 0 else None
        right = k[i + 1] if i < len(k) - 1 else None

        def bump(t, kt=kt, left=left, right=right):
            if t <= kt:
                if left is None:
                    return bump_bp
                return bump_bp * max(0.0, (t - left) / (kt - left))
            if right is None:
                return bump_bp
            return bump_bp * max(0.0, (right - t) / (right - kt))

        rows.append({"key_tenor_y": kt, "krd_dv01_rm": base - revalue(pos, bump)})
    return pd.DataFrame(rows)
