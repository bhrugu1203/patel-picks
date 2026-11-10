"""Odds maths and the walk-forward calibration that decides how far to trust the model."""
from __future__ import annotations

import numpy as np


def implied(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def payout(a):
    """Profit per 1 unit staked."""
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, a / 100, 100 / -a)


def devig(pa, pb):
    pa, pb = implied(pa), implied(pb)
    return pa / (pa + pb)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def fit_logistic(X, y, l2=3.0, iters=30):
    beta = np.zeros(X.shape[1])
    pen = np.eye(X.shape[1]) * l2
    for _ in range(iters):
        p = np.clip(expit(X @ beta), 1e-6, 1 - 1e-6)
        H = (X * (p * (1 - p))[:, None]).T @ X + pen
        step = np.linalg.solve(H, X.T @ (y - p) - pen @ beta)
        beta += step
        if np.abs(step).max() < 1e-6:
            break
    return beta


def calibration_design(p_model, p_market):
    """Intercept, plus the amount the model disagrees with the market.

    The market's own log-odds is an offset (its coefficient is pinned at 1), so with
    no evidence the blend equals the market and no bets are flagged. The model only
    moves the price to the extent its disagreements have been right in the past.
    """
    return np.column_stack([np.ones(len(p_model)), logit(p_model) - logit(p_market)])


def fit_calibration(p_model, p_market, y, l2=3.0):
    X = calibration_design(p_model, p_market)
    off = logit(p_market)
    beta = np.zeros(2)
    pen = np.eye(2) * l2
    for _ in range(30):
        p = np.clip(expit(X @ beta + off), 1e-6, 1 - 1e-6)
        H = (X * (p * (1 - p))[:, None]).T @ X + pen
        step = np.linalg.solve(H, X.T @ (y - p) - pen @ beta)
        beta += step
        if np.abs(step).max() < 1e-6:
            break
    return beta


def apply_calibration(beta, p_model, p_market):
    return expit(calibration_design(p_model, p_market) @ beta + logit(p_market))


def walk_forward(df, p_model="p_model", p_market="p_mkt", y="y", min_train=300, freq="14D", group=None):
    """Fair probability for each row using a calibration fit only on earlier rows."""
    df = df.sort_values("date").copy()
    df["p_fair"] = np.nan
    df["slope"] = np.nan
    bins = df["date"].dt.floor(freq)
    for b in sorted(bins.unique()):
        tr = df[(df["date"] < b) & df[y].notna()]
        target = bins == b
        if len(tr) < min_train:
            continue
        beta = fit_calibration(tr[p_model].values, tr[p_market].values, tr[y].values)
        df.loc[target, "p_fair"] = apply_calibration(beta, df.loc[target, p_model].values, df.loc[target, p_market].values)
        df.loc[target, "slope"] = beta[1]
    return df


def kelly(p, price, fraction=0.25, cap=0.02):
    b = payout(price)
    f = (p * (b + 1) - 1) / b
    return np.clip(f * fraction, 0, cap)
