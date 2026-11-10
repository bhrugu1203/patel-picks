"""Out-of-sample test of game lines and player props against real sportsbook prices.

Usage: python -m edge.backtest [--leagues nba,nfl,nhl,mlb]
Writes data/backtest.json, which the dashboard reads.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from . import data as D, players as P, pricing as X, teams as T
from .ingest import ROOT

warnings.filterwarnings("ignore")
FITS: dict = {}
THRESHOLDS = [0.0, 0.02, 0.04, 0.06, 0.10]
ASSUMED_PRICE = -110.0


def team_rows(league: str, games: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    pred = T.run(games, pg, league)
    g = games.merge(pred, on="game_id")
    rows = []
    for r in g.itertuples():
        pr = T.probabilities(league, r.lam_h, r.lam_a, r.sd_margin, r.sd_total, _v(r, "spread_home_close"), _v(r, "total_close"))
        margin, tot = r.home_score - r.away_score, r.home_score + r.away_score
        for mkt, a, b, won in (
            ("moneyline", "ml_home_close", "ml_away_close", (margin > 0) * 1.0),
            ("spread", "spread_home_price_close", "spread_away_price_close",
             _res(margin + _v(r, "spread_home_close"))),
            ("total", "over_price_close", "under_price_close", _res(tot - _v(r, "total_close"))),
        ):
            key = {"moneyline": "ml_home", "spread": "spread_home", "total": "over"}[mkt]
            pa, pb = _v(r, a), _v(r, b)
            if key not in pr or pa != pa or pb != pb or won != won:
                continue
            rows.append(dict(game_id=r.game_id, date=r.date, league=league, market=mkt, p_model=pr[key],
                             price_a=pa, price_b=pb, p_mkt=float(X.devig(pa, pb)), y=won,
                             open_a=_v(r, a.replace("close", "open")), open_b=_v(r, b.replace("close", "open"))))
    return pd.DataFrame(rows)


def _v(r, name):
    v = getattr(r, name, np.nan)
    return np.nan if v is None else v


def _res(x):
    if x != x or x == 0:
        return np.nan
    return float(x > 0)


def evaluate(df: pd.DataFrame) -> list[dict]:
    """Flat 1-unit bets on whichever side has the higher expected value at the listed price."""
    d = df.dropna(subset=["p_fair", "y"]).copy()
    if d.empty:
        return []
    ev_a = d["p_fair"] * (X.payout(d["price_a"]) + 1) - 1
    ev_b = (1 - d["p_fair"]) * (X.payout(d["price_b"]) + 1) - 1
    d["take_a"] = ev_a >= ev_b
    d["ev"] = np.where(d["take_a"], ev_a, ev_b)
    d["won"] = np.where(d["take_a"], d["y"], 1 - d["y"])
    pay = np.where(d["take_a"], X.payout(d["price_a"]), X.payout(d["price_b"]))
    d["pnl"] = np.where(d["won"] == 1, pay, -1.0)
    out = []
    games = d["game_id"].to_numpy()
    for t in THRESHOLDS:
        s = d[d["ev"] >= t]
        if len(s) < 20:
            out.append(dict(min_ev=t, bets=int(len(s))))
            continue
        lo, hi = _bootstrap(s["pnl"].to_numpy(), s["game_id"].to_numpy())
        out.append(dict(min_ev=t, bets=int(len(s)), hit=float(s["won"].mean()), roi=float(s["pnl"].mean()),
                        roi_lo=lo, roi_hi=hi, avg_ev=float(s["ev"].mean())))
    del games
    return out


def _bootstrap(pnl, groups, n=1000, seed=7):
    """95% interval for ROI, resampling whole games because bets within a game are correlated."""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(groups)
    k = codes.max() + 1
    sums, counts = np.bincount(codes, pnl, k), np.bincount(codes, minlength=k)
    draws = rng.integers(0, k, size=(n, k))
    roi = sums[draws].sum(1) / counts[draws].sum(1)
    return float(np.percentile(roi, 2.5)), float(np.percentile(roi, 97.5))


def likely_table(df: pd.DataFrame) -> list[dict]:
    """How often the highest-probability side actually won, by confidence bucket."""
    d = df.dropna(subset=["p_fair", "y"]).copy()
    if d.empty:
        return []
    take_a = d["p_fair"] >= 0.5
    d["p"] = np.where(take_a, d["p_fair"], 1 - d["p_fair"])
    d["won"] = np.where(take_a, d["y"], 1 - d["y"])
    pay = np.where(take_a, X.payout(d["price_a"]), X.payout(d["price_b"]))
    d["pnl"] = np.where(d["won"] == 1, pay, -1.0)
    d["bucket"] = pd.cut(d["p"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0], include_lowest=True)
    out = []
    for b, s in d.groupby("bucket", observed=True):
        if len(s) >= 20:
            out.append(dict(bucket=f"{b.left:.0%}-{b.right:.0%}", bets=int(len(s)), predicted=float(s["p"].mean()),
                            hit=float(s["won"].mean()), roi=float(s["pnl"].mean())))
    return out


def brier(p, y):
    return float(np.mean((p - y) ** 2))


CALIB: dict = {"games": {}, "props": {}}


def _final_beta(d: pd.DataFrame, min_rows: int = 200) -> list[float] | None:
    d = d.dropna(subset=["y", "p_model", "p_mkt"])
    if len(d) < min_rows:
        return None
    return X.fit_calibration(d["p_model"].values, d["p_mkt"].values, d["y"].values).tolist()


def game_report(all_rows: pd.DataFrame) -> dict:
    rep = {}
    for (lg, mkt), d in all_rows.groupby(["league", "market"]):
        w = X.walk_forward(d, min_train=200, freq="30D")
        oos = w.dropna(subset=["p_fair", "y"])
        CALIB["games"][f"{lg}|{mkt}"] = _final_beta(d)
        rep[f"{lg}|{mkt}"] = dict(
            n=int(len(oos)),
            brier_market=brier(oos["p_mkt"], oos["y"]) if len(oos) else None,
            brier_raw_model=brier(oos["p_model"], oos["y"]) if len(oos) else None,
            brier_blend=brier(oos["p_fair"], oos["y"]) if len(oos) else None,
            weight_on_model=float(w["slope"].dropna().iloc[-1]) if w["slope"].notna().any() else None,
            bets=evaluate(w), likely=likely_table(w))
    return rep


def prop_rows(league: str, frames: dict[str, pd.DataFrame], props: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stat, f in frames.items():
        p = props[props["stat"] == stat]
        if p.empty:
            continue
        m = p.merge(f[["game_id", "player_id", "date", "y", "mu", "mu_nomatch", "alpha", "base"]],
                    on=["game_id", "player_id"], how="inner").dropna(subset=["mu", "y"])
        if m.empty:
            continue
        m["p_model"] = P.p_over(m["mu"], m["alpha"], m["line"])
        m["assumed"] = m["over_price"].isna() | m["under_price"].isna()
        m["price_a"] = m["over_price"].fillna(ASSUMED_PRICE)
        m["price_b"] = m["under_price"].fillna(ASSUMED_PRICE)
        m["p_mkt"] = X.devig(m["price_a"], m["price_b"])
        m["outcome"] = m["y"]
        m["y"] = np.where(m["outcome"] > m["line"], 1.0, np.where(m["outcome"] < m["line"], 0.0, np.nan))
        m["league"] = league
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prop_report(d: pd.DataFrame) -> dict:
    rep = {}
    if d.empty:
        return rep
    for lg, s in d.groupby("league"):
        parts = []
        for stat, sub in s.groupby("stat"):
            parts.append(X.walk_forward(sub, min_train=250, freq="14D"))
            if sub["y"].notna().sum() >= 250:
                CALIB["props"][f"{lg}|{stat}"] = _final_beta(sub)
        CALIB["props"][f"{lg}|_pooled"] = _final_beta(s)
        w = pd.concat(parts, ignore_index=True)
        oos = w.dropna(subset=["p_fair", "y"])
        rep[lg] = dict(
            n=int(len(oos)), assumed_prices=bool(s["assumed"].mean() > 0.5),
            first_test=str(oos["date"].min())[:10] if len(oos) else None,
            brier_market=brier(oos["p_mkt"], oos["y"]) if len(oos) else None,
            brier_raw_model=brier(oos["p_model"], oos["y"]) if len(oos) else None,
            brier_blend=brier(oos["p_fair"], oos["y"]) if len(oos) else None,
            mae_line=float((oos["outcome"] - oos["line"]).abs().mean()) if len(oos) else None,
            mae_model=float((oos["outcome"] - oos["mu"]).abs().mean()) if len(oos) else None,
            bets=evaluate(w), bets_pregame_only=evaluate(w[w["pregame_close"]]), likely=likely_table(w),
            by_stat={k: evaluate(v)[:3] for k, v in w.groupby("stat")})
    return rep


def poisson_dev(y, mu):
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (t - (y - mu))))


def matchup_value(frames: dict[str, pd.DataFrame], last_fit: dict) -> list[dict]:
    """Did matchup, head-to-head and scheme features improve projections on unseen games?"""
    out = []
    for stat, f in frames.items():
        s = f.dropna(subset=["mu", "mu_nomatch", "y"])
        if len(s) < 200:
            continue
        d0, d1 = poisson_dev(s["y"], s["mu_nomatch"]), poisson_dev(s["y"], s["mu"])
        beta = last_fit.get(stat, {}).get("beta", {})
        out.append(dict(stat=stat, n=int(len(s)), dev_without=d0, dev_with=d1, improvement=1 - d1 / d0,
                        mae_base=float((s["y"] - s["base"]).abs().mean()),
                        mae_model=float((s["y"] - s["mu"]).abs().mean()),
                        coef={k: beta.get(k) for k in ("opp_factor", "h2h", "key_h2h")}))
    return out


def run_league(league: str):
    games = D.load_games(league)
    players = D.load_players(league)
    pg = P.add_keys(D.load_player_games(league, players), league)
    ctx = P.prepare(pg, games, league)
    frames, fits = {}, {}
    for stat in D.STATS[league]:
        f = P.stat_frame(ctx, stat)
        frames[stat], fits[stat] = P.fit_predict(f)
    return games, pg, frames, fits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default="nba,nfl,nhl,mlb")
    a = ap.parse_args(argv)
    team_all, prop_all, match = [], [], {}
    for lg in a.leagues.split(","):
        print("running", lg, flush=True)
        games, pg, frames, fits = run_league(lg)
        team_all.append(team_rows(lg, games, pg))
        prop_all.append(prop_rows(lg, frames, D.load_props(lg)))
        match[lg] = matchup_value(frames, fits)
        FITS[lg] = fits
    teams_df = pd.concat(team_all, ignore_index=True)
    props_df = pd.concat(prop_all, ignore_index=True)
    rep = dict(games=game_report(teams_df), props=prop_report(props_df), matchups=match)
    rep["matchups_fits"] = {lg: {k: v for k, v in f.items()} for lg, f in FITS.items()}
    (ROOT / "data" / "backtest.json").write_text(json.dumps(rep, indent=1, default=float))
    (ROOT / "data" / "calibration.json").write_text(json.dumps(CALIB, indent=1))
    summarize(rep)


def summarize(rep: dict):
    print("\n=== GAME LINES (out of sample, closing prices) ===")
    for k, v in rep["games"].items():
        b = v["bets"]
        print(f"{k:16s} n={v['n']:5d} brier mkt={v['brier_market']:.4f} raw={v['brier_raw_model']:.4f} blend={v['brier_blend']:.4f} weight={v['weight_on_model']}")
        for x in b:
            if "roi" in x:
                print(f"    EV>={x['min_ev']:.0%} bets={x['bets']:5d} hit={x['hit']:.3f} ROI={x['roi']:+.3f} [{x['roi_lo']:+.3f},{x['roi_hi']:+.3f}]")
    print("\n=== PLAYER PROPS (out of sample) ===")
    for k, v in rep["props"].items():
        print(f"{k} n={v['n']} from {v['first_test']} assumed_prices={v['assumed_prices']} brier mkt={v['brier_market']:.4f} raw={v['brier_raw_model']:.4f} blend={v['brier_blend']:.4f} | MAE line={v['mae_line']:.2f} model={v['mae_model']:.2f}")
        for name in ("bets", "bets_pregame_only"):
            print(" ", name)
            for x in v[name]:
                if "roi" in x:
                    print(f"    EV>={x['min_ev']:.0%} bets={x['bets']:5d} hit={x['hit']:.3f} ROI={x['roi']:+.3f} [{x['roi_lo']:+.3f},{x['roi_hi']:+.3f}]")
        print("  most likely to hit:")
        for x in v["likely"]:
            print(f"    {x['bucket']:10s} bets={x['bets']:5d} predicted={x['predicted']:.3f} hit={x['hit']:.3f} ROI={x['roi']:+.3f}")
    print("\n=== DO MATCHUPS HELP? (projection error reduction, unseen games) ===")
    for lg, rows in rep["matchups"].items():
        print(lg, " ".join(f"{r['stat']}:{r['improvement']:+.2%}" for r in rows))


if __name__ == "__main__":
    main()
