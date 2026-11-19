"""Nelson-Siegel yield curve fitting.

y(t) = b0 + b1 * (1 - e^(-t/tau)) / (t/tau)
          + b2 * ((1 - e^(-t/tau)) / (t/tau) - e^(-t/tau))

b0 = long-run level, b1 = slope (short end minus long end),
b2 = curvature (the hump), tau = where the hump peaks.

For a fixed tau the model is linear in (b0, b1, b2), so we solve those by
least squares and search tau over a grid. That is fast and always converges,
unlike fitting all four parameters with a nonlinear optimiser.

With only a handful of benchmark points, tau is weakly identified: very
different taus fit almost equally well but extrapolate very differently past
10 years. We therefore restrict tau to 0.3-3 years, the range typically used
in practice (Diebold & Li fix it at about 1.4 years), which keeps the level
b0 a sensible long-run yield.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _loadings(t: np.ndarray, tau: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    x = t / tau
    slope = (1 - np.exp(-x)) / x
    curve = slope - np.exp(-x)
    return np.column_stack([np.ones_like(t), slope, curve])


@dataclass(frozen=True)
class NelsonSiegel:
    b0: float
    b1: float
    b2: float
    tau: float

    def __call__(self, t) -> np.ndarray:
        return _loadings(np.atleast_1d(t), self.tau) @ np.array([self.b0, self.b1, self.b2])

    def shifted(self, bp: float) -> "NelsonSiegel":
        """Parallel shift of the whole curve by bp basis points."""
        return NelsonSiegel(self.b0 + bp / 1e4, self.b1, self.b2, self.tau)


def fit_nelson_siegel(tenors, yields, taus=np.linspace(0.3, 3.0, 271)) -> tuple[NelsonSiegel, float]:
    """Fit to observed (tenor, yield) points. Returns the curve and RMSE in bps."""
    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)
    best = None
    for tau in taus:
        X = _loadings(t, tau)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((X @ beta - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, beta, tau)
    sse, beta, tau = best
    rmse_bp = np.sqrt(sse / len(t)) * 1e4
    return NelsonSiegel(*map(float, beta), float(tau)), float(rmse_bp)
