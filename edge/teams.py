"""Team scoring model: sequential offense/defense ratings with starter adjustments."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

# learning rate, between-season carryover, starting league mean score per team
CFG = {
    "nfl": dict(k=0.09, carry=0.55, mu=22.5),
    "nba": dict(k=0.045, carry=0.65, mu=114.0),
    "nhl": dict(k=0.02, carry=0.65, mu=3.05),
    "mlb": dict(k=0.012, carry=0.6, mu=4.4),
}
MLB_ALPHA = 0.18  # negative-binomial overdispersion for runs


def starters(pg: pd.DataFrame, league: str) -> pd.DataFrame:
    if league == "mlb":
        s = pg[pg["sp"].fillna(0) > 0][["game_id", "team_id", "player_id", "p_er", "p_outs"]]
        return s.rename(columns={"p_er": "a", "p_outs": "b"}).drop_duplicates(["game_id", "team_id"])
    if league == "nhl":
        g = pg[pg["group"] == "G"].sort_values("toi", ascending=False).drop_duplicates(["game_id", "team_id"])
        return g[["game_id", "team_id", "player_id", "ga", "sa"]].rename(columns={"ga": "a", "sa": "b"})
    return pd.DataFrame(columns=["game_id", "team_id", "player_id", "a", "b"])


def run(games: pd.DataFrame, pg: pd.DataFrame, league: str, probables: dict | None = None, **over) -> pd.DataFrame:
    """Walk through games in date order, predicting each before learning from it.

    `games` may include future games (scores NaN). `probables` maps
    (game_id, team_id) -> player_id for starters not yet in a box score.
    """
    cfg = {**CFG[league], **over}
    off, dfn = defaultdict(float), defaultdict(float)
    last_played: dict = {}
    mu, hfa, b2b = cfg["mu"], 0.03 * cfg["mu"], 0.0
    res2_m, res2_t, nres = (14.0 ** 2 if league == "nfl" else 13.0 ** 2), (14.0 ** 2 if league == "nfl" else 18.0 ** 2), 50
    pa, pb = defaultdict(float), defaultdict(float)  # decayed starter stats
    lg_rate = {"mlb": 0.155, "nhl": 0.1}.get(league, 0.0)  # earned runs per out / goals per shot
    st = starters(pg, league)
    st_map = {(r.game_id, r.team_id): r for r in st.itertuples()}
    probables = probables or {}
    season = None
    rows = []
    games = games.sort_values("date")
    for g in games.itertuples():
        if season is not None and g.season != season:
            for t in list(off):
                off[t] *= cfg["carry"]
                dfn[t] *= cfg["carry"]
        season = g.season
        h, a = g.home_id, g.away_id

        def starter_adj(team):
            rec = st_map.get((g.game_id, team))
            pid = rec.player_id if rec is not None else probables.get((g.game_id, team))
            if pid is None:
                return 0.0, None
            if league == "mlb":
                return (pa[pid] - lg_rate * pb[pid]) / (pb[pid] + 90) * 16.0, pid
            if league == "nhl":
                return (pa[pid] - lg_rate * pb[pid]) / (pb[pid] + 600) * 29.0, pid
            return 0.0, pid

        adj_h, _ = starter_adj(h)   # home starter affects the away team's scoring
        adj_a, _ = starter_adj(a)
        day = g.date.normalize()
        tired = lambda t: float(t in last_played and (day - last_played[t]).days == 1)
        bh, ba = tired(h), tired(a)
        home_edge = 0.0 if g.neutral else hfa / 2
        lam_h = max(0.2, mu + off[h] - dfn[a] + home_edge + adj_a + b2b * bh)
        lam_a = max(0.2, mu + off[a] - dfn[h] - home_edge + adj_h + b2b * ba)
        rows.append((g.game_id, lam_h, lam_a, np.sqrt(res2_m), np.sqrt(res2_t), adj_h, adj_a,
                     off[h] - dfn[h], off[a] - dfn[a]))

        if g.home_score != g.home_score:  # future game
            continue
        eh, ea = g.home_score - lam_h, g.away_score - lam_a
        k = cfg["k"]
        off[h] += k * eh; dfn[a] -= k * eh
        off[a] += k * ea; dfn[h] -= k * ea
        lr = 0.004
        mu += lr * (eh + ea) / 2
        if not g.neutral:
            hfa += lr * (eh - ea)
        if bh != ba:
            b2b += lr * (eh if bh else ea)
        nres += 1
        w = max(1 / nres, 0.004)
        res2_m += w * ((eh - ea) ** 2 - res2_m)
        res2_t += w * ((eh + ea) ** 2 - res2_t)
        last_played[h] = last_played[a] = day
        for t in (h, a):
            rec = st_map.get((g.game_id, t))
            if rec is not None and rec.b == rec.b and rec.a == rec.a:
                decay = 0.93 if league == "mlb" else 0.97
                pa[rec.player_id] = pa[rec.player_id] * decay + rec.a
                pb[rec.player_id] = pb[rec.player_id] * decay + rec.b
                if rec.b > 0:
                    lg_rate += 0.002 * (rec.a / rec.b - lg_rate)
    return pd.DataFrame(rows, columns=["game_id", "lam_h", "lam_a", "sd_margin", "sd_total", "sp_h", "sp_a", "net_h", "net_a"])


def _matrix(league: str, lh: float, la: float, size: int) -> np.ndarray:
    x = np.arange(size)
    if league == "nhl":
        ph, pa = poisson.pmf(x, lh), poisson.pmf(x, la)
    else:
        r = 1 / MLB_ALPHA
        ph, pa = nbinom.pmf(x, r, r / (r + lh)), nbinom.pmf(x, r, r / (r + la))
    return np.outer(ph, pa)


def probabilities(league: str, lam_h, lam_a, sd_m, sd_t, spread_home, total) -> dict:
    """Model probability for home ML, home covering the spread, and the over (pushes removed)."""
    out = {}
    margin = lam_h - lam_a
    if league in ("nba", "nfl"):
        out["ml_home"] = float(norm.cdf(margin / sd_m))
        if spread_home == spread_home:
            th = -spread_home
            win = 1 - norm.cdf(th + (0.5 if float(th).is_integer() else 0), margin, sd_m)
            lose = norm.cdf(th - (0.5 if float(th).is_integer() else 0), margin, sd_m)
            out["spread_home"] = float(win / (win + lose))
        if total == total:
            win = 1 - norm.cdf(total + (0.5 if float(total).is_integer() else 0), lam_h + lam_a, sd_t)
            lose = norm.cdf(total - (0.5 if float(total).is_integer() else 0), lam_h + lam_a, sd_t)
            out["over"] = float(win / (win + lose))
        return out
    size = 16 if league == "nhl" else 31
    m = _matrix(league, lam_h, lam_a, size)
    m = m / m.sum()
    grid_h, grid_a = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    diff, tot = grid_h - grid_a, grid_h + grid_a
    tie = m[diff == 0].sum()
    # extra time / shootout: nearly a coin flip, tilted slightly by the stronger side
    p_extra = 0.5 + np.clip(margin * 0.05, -0.08, 0.08)
    out["ml_home"] = float(m[diff > 0].sum() + tie * p_extra)
    if spread_home == spread_home:
        # a regulation tie ends +/-1 after extra time, which is always on the same side of 1.5
        win = m[diff + spread_home > 0].sum()
        lose = m[diff + spread_home < 0].sum()
        out["spread_home"] = float(win / (win + lose))
    if total == total:
        win, lose = m[tot > total].sum(), m[tot < total].sum()
        out["over"] = float(win / (win + lose))
    return out
