"""Yield curve forecasting, benchmarked honestly against a random walk.

Approach (Diebold & Li, 2006):
1. Each week, compress the whole MGS curve into three Nelson-Siegel factors
   (level, slope, curvature) with tau fixed, so the factors are comparable over time.
2. Forecast the factors h weeks ahead, then rebuild the full curve.

Models compared:
- Random walk: tomorrow's curve = today's. Very hard to beat for bond yields.
- Diebold-Li AR(1): each factor mean-reverts at its estimated speed.
- Gradient boosting (ML): predicts each factor's h-week change from current
  factors and their recent momentum.

Evaluation is strictly walk-forward: every forecast uses only data available
on its forecast date, and models are refitted as time moves on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .curve import _loadings

TAU = 1.37  # Diebold-Li's fixed decay (0.0609 per month), in years


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").resample("W-FRI").last().dropna(how="all")


def tenor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("y") and c[1:].isdigit()]


def extract_factors(weekly: pd.DataFrame, tau: float = TAU) -> pd.DataFrame:
    cols = tenor_columns(weekly)
    tenors = np.array([int(c[1:]) for c in cols], float)
    rows = []
    for d, r in weekly[cols].iterrows():
        ok = r.notna().values
        if ok.sum() < 3:
            continue
        X = _loadings(tenors[ok], tau)
        beta, *_ = np.linalg.lstsq(X, r.values[ok] / 100, rcond=None)
        rows.append({"date": d, "level": beta[0], "slope": beta[1], "curv": beta[2]})
    return pd.DataFrame(rows).set_index("date")


def _features(f: pd.DataFrame) -> pd.DataFrame:
    X = f.copy()
    for c in ["level", "slope", "curv"]:
        X[f"{c}_d4"] = f[c].diff(4)
        X[f"{c}_d13"] = f[c].diff(13)
    return X


def walk_forward(daily: pd.DataFrame, horizons=(1, 4, 13), min_train: int = 104,
                 refit_every: int = 13) -> pd.DataFrame:
    """Return RMSE (bp) per model, horizon and tenor."""
    weekly = to_weekly(daily)
    cols = tenor_columns(weekly)
    tenors = np.array([int(c[1:]) for c in cols], float)
    L = _loadings(tenors, TAU)
    F = extract_factors(weekly)
    weekly = weekly.loc[F.index]
    X_all = _features(F)
    names = ["level", "slope", "curv"]
    results = []

    for h in horizons:
        y_future = F[names].shift(-h) - F[names]  # target: factor change over h weeks
        errs = {m: [] for m in ("Random walk", "Diebold-Li AR(1)", "Gradient boosting")}
        models = None
        for i in range(min_train, len(F) - h):
            now, actual = F.index[i], weekly[cols].iloc[i + h].values / 100
            if np.isnan(actual).any():
                continue
            f_now = F[names].iloc[i].values

            # Random walk
            errs["Random walk"].append(L @ f_now - actual)

            # Diebold-Li AR(1), fitted on data up to now only
            hist = F[names].iloc[: i + 1].values
            ar_fc = []
            for j in range(3):
                x, y = hist[:-1, j], hist[1:, j]
                b, a = np.polyfit(x, y, 1)
                v = f_now[j]
                for _ in range(h):
                    v = a + b * v
                ar_fc.append(v)
            errs["Diebold-Li AR(1)"].append(L @ np.array(ar_fc) - actual)

            # Gradient boosting on factor changes; train only on targets already observed
            if models is None or (i - min_train) % refit_every == 0:
                train_end = i - h + 1
                Xt = X_all.iloc[13:train_end].dropna()
                yt = y_future.loc[Xt.index]
                models = [GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                    subsample=0.8, random_state=0).fit(Xt.values, yt[n].values)
                          for n in names]
            x_now = X_all.iloc[[i]].fillna(0).values
            gb_fc = f_now + np.array([m.predict(x_now)[0] for m in models])
            errs["Gradient boosting"].append(L @ gb_fc - actual)

        for model, e in errs.items():
            e = np.array(e) * 1e4  # to bp
            rmse = np.sqrt(np.mean(e**2, axis=0))
            for t, v in zip(tenors, rmse):
                results.append({"model": model, "horizon_w": h, "tenor_y": int(t), "rmse_bp": v, "n": len(e)})
    return pd.DataFrame(results)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Average RMSE across tenors, and % improvement over the random walk."""
    s = res.groupby(["horizon_w", "model"])["rmse_bp"].mean().unstack("model")
    for m in s.columns:
        if m != "Random walk":
            s[f"{m} vs RW %"] = (1 - s[m] / s["Random walk"]) * 100
    return s


def forecast_curve(daily: pd.DataFrame, h: int = 4) -> dict:
    """Latest Diebold-Li AR(1) forecast of the curve h weeks ahead."""
    F = extract_factors(to_weekly(daily))
    names = ["level", "slope", "curv"]
    hist = F[names].values
    fc = []
    for j in range(3):
        b, a = np.polyfit(hist[:-1, j], hist[1:, j], 1)
        v = hist[-1, j]
        for _ in range(h):
            v = a + b * v
        fc.append(v)
    return {"as_of": str(F.index[-1].date()), "horizon_weeks": h,
            "current": dict(zip(names, hist[-1].tolist())), "forecast": dict(zip(names, fc))}
