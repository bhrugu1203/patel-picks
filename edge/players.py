"""Walk-forward player projections with matchup, head-to-head and scheme features.

Every feature for a game uses only earlier games. Future games are appended as
rows with empty stats, so live projections go through exactly the same code as
the backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from .data import STATS

HALFLIFE = {"nba": (12, 6), "nfl": (6, 4), "nhl": (20, 8), "mlb": (40, 15)}  # (rate, exposure) in games
PRIOR_EXPOSURE = {"min": 150, "pass_att": 80, "rush_att": 25, "tgt": 25, "touch": 30, "gp": 4,
                  "toi": 150, "sa": 250, "pa": 120, "bf": 120}

# Opponent tendencies from box scores, the free-data stand-in for scheme charting.
# name -> (numerator, denominator or None), computed on what the defense allowed.
STYLE = {
    "nba": {"pace": ("poss", None), "three_rate": ("fg3a", "fga"), "ft_rate": ("fta", "fga"),
            "ast_rate": ("ast", "fgm"), "reb_allowed": ("reb", None)},
    "nfl": {"pass_volume": ("pass_att", None), "yds_per_att": ("pass_yds", "pass_att"),
            "yds_per_comp": ("pass_yds", "pass_cmp"), "cmp_rate": ("pass_cmp", "pass_att"),
            "yds_per_carry": ("rush_yds", "rush_att"), "sack_rate": ("sacked", "dropbacks"),
            "run_rate": ("rush_att", "plays")},
    "nhl": {"shots_allowed": ("sog", None), "goals_allowed": ("g", None)},
    "mlb": {"opp_k_rate": ("k", "pa"), "opp_bb_rate": ("bb", "pa"), "opp_hr_rate": ("hr", "pa"),
            "opp_hit_rate": ("h", "pa")},
}
OWN_STYLE = {"nba": {"blk_own": "blk", "stl_own": "stl"}, "nfl": {}, "nhl": {"hits_own": "hits", "blocks_own": "bs"},
             "mlb": {}}
SP_STYLE = {"sp_k": ("p_k", "bf"), "sp_bb": ("p_bb", "bf"), "sp_hr": ("p_hr", "bf"), "sp_h": ("p_h", "bf")}
ROTATION_Q = 0.4
MATCHUP_FEATURES = ["opp_factor", "h2h", "key_h2h"]
NON_MATCHUP = ["log_base", "is_home", "b2b", "env_total", "env_fav", "env_fav2", "vacated"]


SEASON_K: dict = {}  # league -> games of weight on the long-run level (blank = off)


def ew_prior(df: pd.DataFrame, by: str | list, cols: list[str], halflife: float) -> pd.DataFrame:
    """Exponentially weighted mean of each column using only rows before the current one."""
    grp = df.groupby(by, sort=False)
    ew = grp[cols].ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
    ew = ew.reset_index(level=list(range(ew.index.nlevels - 1)), drop=True).sort_index()
    return ew.groupby([df[c] for c in ([by] if isinstance(by, str) else by)], sort=False).shift(1)


def effective_n(n, halflife):
    return (1 - 0.5 ** (n / halflife)) / (1 - 0.5 ** (1 / halflife))


def team_totals(pg: pd.DataFrame, league: str) -> pd.DataFrame:
    cols = {"nba": ["fga", "fta", "tov", "oreb", "fg3a", "fgm", "ast", "reb", "blk", "stl", "pts"],
            "nfl": ["pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds", "sacked"],
            "nhl": ["sog", "g", "hits", "bs"],
            "mlb": ["pa", "k", "bb", "hr", "h"]}[league]
    t = pg[pg["_played"]].groupby(["game_id", "team_id"], sort=False)[cols].sum(min_count=1).reset_index()
    if league == "nba":
        t["poss"] = t["fga"] + 0.44 * t["fta"] + t["tov"] - t["oreb"]
    if league == "nfl":
        t["dropbacks"] = t["pass_att"] + t["sacked"]
        t["plays"] = t["pass_att"] + t["rush_att"]
    return t


def style_table(pg: pd.DataFrame, league: str, sched: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pre-game tendencies of each team's defense, z-scored. Key: (game_id, team_id = defense)."""
    t = team_totals(pg, league)
    # what the defense allowed = the opposing offense's line in that game
    s = sched.melt(id_vars=["game_id", "date"], value_vars=["home_id", "away_id"], value_name="def_id")
    s = s.merge(sched[["game_id", "home_id", "away_id"]], on="game_id")
    s["off_id"] = np.where(s["def_id"] == s["home_id"], s["away_id"], s["home_id"])
    if league == "mlb":  # pitchers are projected against the opposing lineup's own tendencies
        s["off_id"] = s["def_id"]
    allowed = s[["game_id", "date", "def_id", "off_id"]].merge(
        t, left_on=["game_id", "off_id"], right_on=["game_id", "team_id"], how="left").drop(columns="team_id")
    own_cols = OWN_STYLE[league]
    if own_cols:
        own = t[["game_id", "team_id", *own_cols.values()]].rename(
            columns={"team_id": "def_id", **{v: k for k, v in own_cols.items()}})
        allowed = allowed.merge(own, on=["game_id", "def_id"], how="left")
    allowed = allowed.sort_values(["date", "game_id"]).reset_index(drop=True)
    raw_cols = [c for c in allowed.columns if c not in ("game_id", "date", "def_id", "off_id")]
    ew = ew_prior(allowed, "def_id", raw_cols, 10)
    out = allowed[["game_id", "def_id"]].copy()
    names = []
    for name, (num, den) in STYLE[league].items():
        out[f"s_{name}"] = ew[num] / ew[den] if den else ew[num]
        names.append(f"s_{name}")
    for name in own_cols:
        out[f"s_{name}"] = ew[name]
        names.append(f"s_{name}")
    for c in names:
        out[c] = ((out[c] - out[c].mean()) / out[c].std()).clip(-3, 3).fillna(0)
    return out, names


def sp_table(pg: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Each starter's pre-game rates, z-scored. MLB only."""
    sp = pg[(pg["sp"].fillna(0) > 0) | pg["_future_sp"]].copy().reset_index(drop=True)
    cols = ["p_k", "p_bb", "p_hr", "p_h", "bf"]
    ew = ew_prior(sp, "player_id", cols, 12)
    out = sp[["game_id", "team_id", "player_id"]].rename(columns={"team_id": "sp_team", "player_id": "sp_id"})
    names = []
    for name, (n, d) in SP_STYLE.items():
        v = ew[n] / ew[d]
        out[f"s_{name}"] = ((v - v.mean()) / v.std()).clip(-3, 3).fillna(0)
        names.append(f"s_{name}")
    return out, names


def prepare(pg: pd.DataFrame, games: pd.DataFrame, league: str) -> dict:
    """Features that are shared across stats. `pg` may include future rows (stats empty)."""
    pg = pg.sort_values(["date", "game_id"]).reset_index(drop=True)
    for c in ("_future_sp", "_future_bat"):
        pg[c] = pg[c].fillna(False).astype(bool) if c in pg else False
    expo_col = {"nba": "min", "nhl": "toi", "mlb": "pa", "nfl": "touch"}[league]
    pg["_played"] = pg[expo_col].notna() | (league == "mlb" and pg.get("bf", pd.Series(index=pg.index)).notna())
    prev = pg.groupby("player_id", sort=False)["date"].shift(1)
    pg["b2b"] = ((pg["date"] - prev).dt.total_seconds() < 36 * 3600).astype(float)

    sched = games[["game_id", "date", "home_id", "away_id"]].copy()
    style, style_names = style_table(pg, league, sched)
    ctx = {"league": league, "pg": pg, "style": style, "style_names": style_names}
    if league == "mlb":
        ctx["sp"], ctx["sp_names"] = sp_table(pg)

    # market environment: game total and how favored the player's team is
    env = []
    for r in games.itertuples():
        def pick(*names):
            for n in names:
                v = getattr(r, n, None)
                if v is not None and v == v:
                    return v
            return np.nan
        total = pick("total_open", "total_close")
        mh, ma = pick("ml_home_open", "ml_home_close"), pick("ml_away_open", "ml_away_close")
        ph = np.nan
        if mh == mh and ma == ma:
            a, b = implied(mh), implied(ma)
            ph = a / (a + b)
        env.append((r.game_id, r.home_id, total, ph))
        env.append((r.game_id, r.away_id, total, 1 - ph if ph == ph else np.nan))
    ctx["env"] = pd.DataFrame(env, columns=["game_id", "team_id", "mkt_total", "mkt_win"]).drop_duplicates(
        ["game_id", "team_id"])
    return ctx


def implied(american):
    a = np.asarray(american, dtype=float)
    return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def stat_frame(ctx: dict, stat: str) -> pd.DataFrame:
    league = ctx["league"]
    expo, groups = STATS[league][stat]
    pg = ctx["pg"]
    d = pg
    if league == "mlb":
        batting = expo == "pa"
        d = d.assign(group="BAT" if batting else "SP")
        if not batting:
            d = d[(d["sp"].fillna(0) > 0) | d["_future_sp"]]
        else:
            d = d[d["pa"].notna() | d["_future_bat"]]
    elif groups:
        d = d[d["group"].isin(groups)]
    keep = ["game_id", "date", "player_id", "name", "team_id", "opp_id", "group", "is_home", "b2b", "season"]
    keep += [c for c in ("key_id", "key_name", "_future", "status", "_out") if c in d.columns]
    d = d.assign(expo=d[expo], y=d[stat])[keep + ["expo", "y"]].copy()
    d = d.sort_values(["date", "game_id"]).reset_index(drop=True)
    d.loc[d["expo"].fillna(0) <= 0, "expo"] = np.nan
    d.loc[d["expo"].isna(), "y"] = np.nan

    h_rate, h_expo = HALFLIFE[league]
    ew_r = ew_prior(d, "player_id", ["y", "expo"], h_rate)
    ew_e = ew_prior(d, "player_id", ["expo"], h_expo)
    n = d.groupby("player_id", sort=False)["y"].transform(lambda s: s.notna().cumsum().shift(1).fillna(0))
    d["n_prior"] = n
    w = effective_n(n, h_rate)
    gy = d.groupby("group", sort=False)
    py = gy["y"].transform(lambda s: s.fillna(0).cumsum().shift(1))
    pe = gy["expo"].transform(lambda s: s.fillna(0).cumsum().shift(1))
    prior_rate = (py / pe).ffill().bfill()
    m = PRIOR_EXPOSURE[expo]
    d["exp_hat"] = ew_e["expo"]
    # two-level shrinkage: recent form -> the player's own long-run rate -> position average
    h_long = h_rate * 5
    ew_l = ew_prior(d, "player_id", ["y", "expo"], h_long)
    wl = effective_n(n, h_long)
    long_rate = (ew_l["y"] * wl + prior_rate * m) / (ew_l["expo"] * wl + m)
    long_rate = long_rate.fillna(prior_rate)
    k = SEASON_K.get(league)
    if k:
        # role changes: once a player has a real sample this season, pull his long-run level toward it
        gs = d.groupby(["player_id", "season"], sort=False)
        sy = gs["y"].transform(lambda x: x.fillna(0).cumsum().shift(1).fillna(0))
        se = gs["expo"].transform(lambda x: x.fillna(0).cumsum().shift(1).fillna(0))
        ns = gs["y"].transform(lambda x: x.notna().cumsum().shift(1).fillna(0))
        srate = sy / se.replace(0, np.nan)
        long_rate = ((long_rate * k + srate.fillna(long_rate) * ns) / (k + ns)).fillna(long_rate)
    rate = (ew_r["y"] * w + long_rate * m) / (ew_r["expo"] * w + m)
    d["base"] = (d["exp_hat"] * rate.fillna(prior_rate)).clip(lower=0.01)
    d.loc[n < 1, "base"] = np.nan

    # recent form for display
    g = d.groupby("player_id", sort=False)["y"]
    d["l5"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    d["l10"] = g.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    d["season_avg"] = d.groupby(["player_id", "season"], sort=False)["y"].transform(
        lambda s: s.shift(1).expanding().mean())

    # defense vs position: what this opponent gives up to this position group relative to expectation
    has = d["y"].notna() & d["base"].notna()
    agg = d[has].groupby(["date", "game_id", "opp_id", "group"], sort=False).agg(a=("y", "sum"), b=("base", "sum")).reset_index()
    future = d[~d["y"].notna() & d["base"].notna()][["date", "game_id", "opp_id", "group"]].drop_duplicates()
    agg = pd.concat([agg, future.assign(a=np.nan, b=np.nan)]).drop_duplicates(["game_id", "opp_id", "group"])
    agg = agg.sort_values(["date", "game_id"]).reset_index(drop=True)
    agg["key"] = agg["opp_id"].astype(str) + "|" + agg["group"]
    ew = ew_prior(agg, "key", ["a", "b"], 12)
    cnt = agg.groupby("key", sort=False)["a"].transform(lambda s: s.notna().cumsum().shift(1).fillna(0))
    wt = effective_n(cnt, 12)
    k = 6.0
    agg["opp_ratio"] = (ew["a"] * wt + ew["b"].fillna(1) * k) / (ew["b"] * wt + ew["b"].fillna(1) * k)
    d = d.merge(agg[["game_id", "opp_id", "group", "opp_ratio"]], on=["game_id", "opp_id", "group"], how="left")
    d["opp_factor"] = np.log(d["opp_ratio"].clip(0.6, 1.6)).fillna(0)

    # player vs this team, and vs the key opposing player (starter, goalie, counterpart)
    _h2h(d, ["player_id", "opp_id"], "h2h", 8.0)
    if "key_id" in d.columns and d["key_id"].notna().any():
        _h2h(d, ["player_id", "key_id"], "key_h2h", 6.0)
    else:
        d["key_h2h"], d["key_h2h_n"], d["key_h2h_avg"] = 0.0, 0.0, np.nan

    # market environment
    d = d.merge(ctx["env"], on=["game_id", "team_id"], how="left")
    team_tot = d[d["y"].notna()].groupby(["date", "game_id"], sort=False)["mkt_total"].first()
    avg_total = team_tot.expanding().mean().shift(1).reset_index().drop_duplicates("game_id")
    d = d.merge(avg_total.rename(columns={"mkt_total": "avg_total"})[["game_id", "avg_total"]], on="game_id", how="left")
    d["avg_total"] = d["avg_total"].ffill().bfill()
    d["env_total"] = np.log(d["mkt_total"] / d["avg_total"]).fillna(0).clip(-0.4, 0.4)
    d["env_fav"] = (d["mkt_win"] - 0.5).fillna(0)
    d["env_fav2"] = d["env_fav"] ** 2

    # vacated usage: when teammates are missing, the remaining players absorb their share
    present = d.groupby(["game_id", "team_id"], sort=False)["base"].transform("sum")
    tt = d[d["y"].notna()].groupby(["date", "game_id", "team_id"], sort=False)["y"].sum().reset_index()
    fut = d[d["y"].isna()][["date", "game_id", "team_id"]].drop_duplicates().assign(y=np.nan)
    tt = pd.concat([tt, fut]).drop_duplicates(["game_id", "team_id"]).sort_values(["date", "game_id"]).reset_index(drop=True)
    tt["team_ew"] = ew_prior(tt, "team_id", ["y"], 8)["y"]
    d = d.merge(tt[["game_id", "team_id", "team_ew"]], on=["game_id", "team_id"], how="left")
    d["vacated"] = np.log(d["team_ew"] / present.replace(0, np.nan)).clip(-0.5, 0.7).fillna(0)
    if "_out" in d.columns:
        # upcoming games: the listed group includes fringe players who often sit, so the
        # historical formula reads low for everyone. Measure only what injured players take away.
        fut = d["_future"].fillna(False).astype(bool) & d["y"].isna()
        if fut.any():
            out = d["_out"].fillna(False).astype(bool)
            fb = d.loc[fut, ["game_id", "team_id"]].assign(all_=d.loc[fut, "base"].fillna(0),
                                                           act=d.loc[fut, "base"].fillna(0).where(~out[fut], 0))
            g = fb.groupby(["game_id", "team_id"], sort=False)
            d.loc[fut, "vacated"] = np.log(g["all_"].transform("sum") / g["act"].transform("sum").replace(0, np.nan)).clip(0, 0.7).fillna(0)

    # opponent scheme / style
    if league == "mlb" and expo == "pa":
        s = ctx["sp"].rename(columns={"sp_team": "opp_id"})
        d = d.merge(s.drop(columns="sp_id"), on=["game_id", "opp_id"], how="left")
        style_names = list(ctx["sp_names"])
    else:
        d = d.merge(ctx["style"].rename(columns={"def_id": "opp_id"}), on=["game_id", "opp_id"], how="left")
        style_names = list(ctx["style_names"])
    for c in style_names:
        d[c] = d[c].fillna(0)
    d["log_base"] = np.log(d["base"])
    d.attrs["style_names"] = style_names
    return d


def _h2h(d: pd.DataFrame, by: list[str], name: str, k: float):
    ok = d["y"].notna() & d["base"].notna()
    d["_a"] = d["y"].where(ok, 0.0)
    d["_b"] = d["base"].where(ok, 0.0)
    d["_n"] = ok.astype(float)
    g = d.groupby(by, sort=False)
    sa, sb, cn = (g[c].transform(lambda s: s.cumsum().shift(1).fillna(0)) for c in ("_a", "_b", "_n"))
    mean_b = d["base"].fillna(d["base"].median())
    d[name] = np.log(((sa + k * mean_b) / (sb + k * mean_b)).clip(0.5, 2)).fillna(0)
    d[f"{name}_n"] = cn
    d[f"{name}_avg"] = (sa / cn.replace(0, np.nan))
    d.drop(columns=["_a", "_b", "_n"], inplace=True)


def fit_poisson(X: np.ndarray, y: np.ndarray, offset: np.ndarray, l2: float = 5.0, iters: int = 25) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    pen = np.eye(X.shape[1]) * l2
    pen[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(X @ beta + offset, -8, 8)
        mu = np.exp(eta)
        H = (X * mu[:, None]).T @ X + pen
        g = X.T @ (y - mu) - pen @ beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-5:
            break
    return beta


def design(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(d)), *[d[c].fillna(0).to_numpy(float) for c in cols]])


def fit_predict(d: pd.DataFrame, min_train: int = 800) -> tuple[pd.DataFrame, dict]:
    """Refit monthly on the past only; predict the following month. Future rows get the latest fit."""
    styles = d.attrs.get("style_names", [])
    match_cols = NON_MATCHUP + MATCHUP_FEATURES + styles
    d = d.copy()
    d["mu"], d["mu_nomatch"], d["alpha"] = np.nan, np.nan, np.nan
    usable = d["base"].notna() & (d["n_prior"] >= 3)
    # train on rotation players (the ones books post lines for), not deep bench
    rotation = d["exp_hat"] >= d.groupby("group")["exp_hat"].transform(lambda s: s.quantile(ROTATION_Q))
    train_mask = usable & d["y"].notna() & rotation
    months = d["date"].dt.to_period("M")
    last = {}
    for mth in sorted(months[usable].unique()):
        start = mth.to_timestamp()
        tr = d[train_mask & (d["date"] < start)]
        target = usable & (months == mth)
        if len(tr) < min_train:
            continue
        off = tr["log_base"].to_numpy()
        y = tr["y"].to_numpy(float)
        b_full = fit_poisson(design(tr, match_cols), y, off)
        b_nom = fit_poisson(design(tr, NON_MATCHUP), y, off)
        in_mu = np.exp(design(tr, match_cols) @ b_full + off)
        alpha = max(0.002, float(np.sum((y - in_mu) ** 2 - in_mu) / np.sum(in_mu ** 2)))
        t = d[target]
        o = t["log_base"].to_numpy()
        d.loc[target, "mu"] = np.exp(design(t, match_cols) @ b_full + o)
        d.loc[target, "mu_nomatch"] = np.exp(design(t, NON_MATCHUP) @ b_nom + o)
        d.loc[target, "alpha"] = alpha
        last = {"beta": dict(zip(["const", *match_cols], b_full.round(4).tolist())), "alpha": alpha, "n_train": len(tr)}
    return d, last


def p_over(mu, alpha, line):
    """Probability of going over the line, with pushes removed (integer lines)."""
    mu, alpha, line = (np.asarray(x, dtype=float) for x in (mu, alpha, line))
    r = 1 / alpha
    p = r / (r + mu)
    over = nbinom.sf(np.floor(line), r, p)
    is_int = np.isclose(line, np.round(line))
    push = np.where(is_int, nbinom.pmf(np.round(line), r, p), 0.0)
    return np.clip(over / np.clip(1 - push, 1e-6, None), 1e-4, 1 - 1e-4)


def add_keys(pg: pd.DataFrame, league: str, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Key opposing player: starting pitcher, goalie, or NBA positional counterpart."""
    pg = pg.copy()
    if league == "mlb":
        sp = pg[pg["sp"].fillna(0) > 0][["game_id", "team_id", "player_id", "name"]].drop_duplicates(["game_id", "team_id"])
        sp.columns = ["game_id", "opp_id", "key_id", "key_name"]
        pg = pg.merge(sp, on=["game_id", "opp_id"], how="left")
    elif league == "nhl":
        gk = pg[pg["group"] == "G"].sort_values("toi", ascending=False).drop_duplicates(["game_id", "team_id"])
        gk = gk[["game_id", "team_id", "player_id", "name"]]
        gk.columns = ["game_id", "opp_id", "key_id", "key_name"]
        pg = pg.merge(gk, on=["game_id", "opp_id"], how="left")
    elif league == "nba":
        order = pg.sort_values(["date", "game_id"]).reset_index(drop=True)
        rec = order.groupby("player_id", sort=False)["min"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        order["_rec"] = rec.fillna(0)
        top = order.sort_values("_rec", ascending=False).drop_duplicates(["game_id", "team_id", "group"])
        top = top[["game_id", "team_id", "group", "player_id", "name"]]
        top.columns = ["game_id", "opp_id", "group", "key_id", "key_name"]
        pg = pg.merge(top, on=["game_id", "opp_id", "group"], how="left")
    else:
        pg["key_id"], pg["key_name"] = None, None
    return pg


def fit_live(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One fit on every finished game; projects only the upcoming (stat-less) rows."""
    styles = d.attrs.get("style_names", [])
    cols = NON_MATCHUP + MATCHUP_FEATURES + styles
    d = d.copy()
    usable = d["base"].notna() & (d["n_prior"] >= 1)
    rotation = d["exp_hat"] >= d.groupby("group")["exp_hat"].transform(lambda s: s.quantile(ROTATION_Q))
    tr = d[usable & d["y"].notna() & (d["n_prior"] >= 3) & rotation]
    off = tr["log_base"].to_numpy()
    y = tr["y"].to_numpy(float)
    beta = fit_poisson(design(tr, cols), y, off)
    mu_in = np.exp(design(tr, cols) @ beta + off)
    alpha = max(0.002, float(np.sum((y - mu_in) ** 2 - mu_in) / np.sum(mu_in ** 2)))
    fut = usable & d["y"].isna() & d["_future"].fillna(False).astype(bool)
    t = d[fut]
    d["mu"] = np.nan
    d.loc[fut, "mu"] = np.exp(design(t, cols) @ beta + t["log_base"].to_numpy())
    d["alpha"] = alpha
    return d, {"beta": dict(zip(["const", *cols], beta.round(4).tolist())), "alpha": alpha}
