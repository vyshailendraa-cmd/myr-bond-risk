"""Fixed-rate bond maths: price, yield, duration, convexity, DV01.

Conventions (simplified, matching Malaysian government bonds):
- Coupons paid semi-annually (frequency = 2).
- Yields quoted as annual rates compounded at the coupon frequency.
- Face value 100. Settlement assumed on a coupon date, so there is no
  accrued interest and price = clean price = dirty price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Bond:
    name: str
    coupon: float          # annual coupon rate, e.g. 0.0392 for 3.92%
    maturity: float        # years to maturity
    freq: int = 2          # coupon payments per year
    face: float = 100.0

    def cashflows(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (times in years, cash amounts)."""
        n = max(int(round(self.maturity * self.freq)), 1)
        times = np.arange(1, n + 1) / self.freq
        # shift so the final payment lands exactly on maturity
        times = times - times[-1] + self.maturity
        cf = np.full(n, self.face * self.coupon / self.freq)
        cf[-1] += self.face
        return times, cf


def price(bond: Bond, ytm: float) -> float:
    """Price per 100 face given yield to maturity."""
    t, cf = bond.cashflows()
    df = (1 + ytm / bond.freq) ** (-bond.freq * t)
    return float(np.sum(cf * df))


def yield_to_maturity(bond: Bond, target_price: float) -> float:
    """Solve for the yield that reproduces target_price."""
    return brentq(lambda y: price(bond, y) - target_price, -0.05, 1.0)


def macaulay_duration(bond: Bond, ytm: float) -> float:
    """Present-value-weighted average time to cash flows (years)."""
    t, cf = bond.cashflows()
    pv = cf * (1 + ytm / bond.freq) ** (-bond.freq * t)
    return float(np.sum(t * pv) / np.sum(pv))


def modified_duration(bond: Bond, ytm: float) -> float:
    """% price change for a 1.00 (100%) change in yield, i.e. -dP/dy / P."""
    return macaulay_duration(bond, ytm) / (1 + ytm / bond.freq)


def convexity(bond: Bond, ytm: float) -> float:
    """Second-order price sensitivity, d2P/dy2 / P."""
    t, cf = bond.cashflows()
    f = bond.freq
    pv = cf * (1 + ytm / f) ** (-f * t)
    k = f * t
    return float(np.sum(pv * k * (k + 1)) / (np.sum(pv) * f**2 * (1 + ytm / f) ** 2))


def dv01(bond: Bond, ytm: float) -> float:
    """Price change (per 100 face) for a 1 basis point fall in yield."""
    return modified_duration(bond, ytm) * price(bond, ytm) * 1e-4


def approx_price_change(bond: Bond, ytm: float, dy: float) -> float:
    """Duration + convexity estimate of the % price change for a yield move dy."""
    d = modified_duration(bond, ytm)
    c = convexity(bond, ytm)
    return -d * dy + 0.5 * c * dy**2
