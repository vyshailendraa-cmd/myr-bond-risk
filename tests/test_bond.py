import numpy as np
import pytest

from bondrisk.bond import (Bond, approx_price_change, convexity, macaulay_duration,
                           modified_duration, price, yield_to_maturity)
from bondrisk.curve import NelsonSiegel, fit_nelson_siegel


def test_par_bond_prices_at_100():
    b = Bond("par", 0.04, 10)
    assert price(b, 0.04) == pytest.approx(100, abs=1e-9)


def test_price_falls_when_yield_rises():
    b = Bond("x", 0.04, 10)
    assert price(b, 0.05) < price(b, 0.04) < price(b, 0.03)


def test_discount_and_premium():
    b = Bond("x", 0.05, 5)
    assert price(b, 0.06) < 100 < price(b, 0.04)


def test_ytm_roundtrip():
    b = Bond("x", 0.0392, 9.91)
    assert yield_to_maturity(b, price(b, 0.0383)) == pytest.approx(0.0383, abs=1e-10)


def test_zero_coupon_duration_equals_maturity():
    b = Bond("zero", 0.0, 7)
    assert macaulay_duration(b, 0.04) == pytest.approx(7, abs=1e-9)


def test_duration_matches_finite_difference():
    b, y, h = Bond("x", 0.04, 10), 0.04, 1e-5
    fd = -(price(b, y + h) - price(b, y - h)) / (2 * h) / price(b, y)
    assert modified_duration(b, y) == pytest.approx(fd, rel=1e-6)


def test_convexity_matches_finite_difference():
    b, y, h = Bond("x", 0.04, 10), 0.04, 1e-4
    fd = (price(b, y + h) - 2 * price(b, y) + price(b, y - h)) / h**2 / price(b, y)
    assert convexity(b, y) == pytest.approx(fd, rel=1e-4)


def test_duration_convexity_approximation_is_close():
    b, y, dy = Bond("x", 0.04, 10), 0.04, 0.01
    exact = price(b, y + dy) / price(b, y) - 1
    assert approx_price_change(b, y, dy) == pytest.approx(exact, abs=5e-4)


def test_nelson_siegel_recovers_known_curve():
    true = NelsonSiegel(0.04, -0.01, 0.005, 2.0)
    t = np.array([0.25, 1, 2, 3, 5, 7, 10, 15])
    fit, rmse = fit_nelson_siegel(t, true(t), taus=np.linspace(0.5, 5, 451))
    assert rmse < 0.01
    assert fit.b0 == pytest.approx(0.04, abs=1e-6)
