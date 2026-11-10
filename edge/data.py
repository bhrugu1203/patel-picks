"""Turn raw stored box scores into one tidy player-game table per league."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .ingest import connect

# stat -> (exposure column, eligible position groups or None for all)
STATS = {
    "nba": {s: ("min", None) for s in
            ["pts", "reb", "ast", "fg3m", "stl", "blk", "pr", "pa", "ra", "pra", "sb"]},
    "nfl": {
        "pass_yds": ("pass_att", {"QB"}), "pass_att": ("pass_att", {"QB"}),
        "pass_cmp": ("pass_att", {"QB"}), "pass_td": ("pass_att", {"QB"}),
        "pass_int": ("pass_att", {"QB"}),
        "rush_yds": ("rush_att", {"QB", "RB", "WR"}), "rush_att": ("rush_att", {"QB", "RB", "WR"}),
        "rec": ("tgt", {"RB", "WR", "TE"}), "rec_yds": ("tgt", {"RB", "WR", "TE"}),
        "rush_rec_yds": ("touch", {"RB", "WR", "TE"}),
        "pass_rush_yds": ("pass_att", {"QB"}),
        "tot_td": ("touch", {"RB", "WR", "TE"}),
        "tackles": ("gp", {"DEF"}),
    },
    "nhl": {
        "sog": ("toi", {"F", "D"}), "g": ("toi", {"F", "D"}), "a": ("toi", {"F", "D"}),
        "p": ("toi", {"F", "D"}), "bs": ("toi", {"F", "D"}), "sv": ("sa", {"G"}),
    },
    "mlb": {
        "h": ("pa", {"BAT"}), "r": ("pa", {"BAT"}), "rbi": ("pa", {"BAT"}),
        "hrr": ("pa", {"BAT"}), "bb": ("pa", {"BAT"}), "k": ("pa", {"BAT"}), "hr": ("pa", {"BAT"}),
        "p_k": ("bf", {"SP"}), "p_outs": ("bf", {"SP"}), "p_h": ("bf", {"SP"}),
        "p_er": ("bf", {"SP"}), "p_bb": ("bf", {"SP"}),
    },
}
STAT_LABELS = {  # sportsbook-style market names
    "pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "Threes Made", "stl": "Steals",
    "blk": "Blocks", "pr": "Points + Rebounds", "pa": "Points + Assists", "ra": "Rebounds + Assists",
    "pra": "Points + Rebounds + Assists", "sb": "Steals + Blocks",
    "pass_yds": "Passing Yards", "pass_att": "Pass Attempts", "pass_cmp": "Pass Completions",
    "pass_td": "Passing Touchdowns", "pass_int": "Interceptions Thrown", "rush_yds": "Rushing Yards",
    "rush_att": "Rush Attempts", "rec": "Receptions", "rec_yds": "Receiving Yards",
    "rush_rec_yds": "Rushing + Receiving Yards", "pass_rush_yds": "Passing + Rushing Yards",
    "tot_td": "Anytime Touchdown Scorer", "tackles": "Tackles + Assists",
    "sog": "Shots on Goal", "g": "Goals", "a": "Assists", "p": "Points", "bs": "Blocked Shots",
    "sv": "Goalie Saves", "h": "Hits", "r": "Runs Scored", "rbi": "RBIs", "hrr": "Hits + Runs + RBIs",
    "bb": "Walks", "k": "Batter Strikeouts", "hr": "Home Runs", "p_k": "Pitcher Strikeouts",
    "p_outs": "Pitcher Outs Recorded", "p_h": "Hits Allowed", "p_er": "Earned Runs Allowed",
    "p_bb": "Walks Allowed",
}
# sportsbook market name -> stat
MARKETS = {
    "nba": {"Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast",
            "Total 3-Point Field Goals": "fg3m", "Total Steals": "stl", "Total Blocks": "blk",
            "Total Points and Rebounds": "pr", "Total Points and Assists": "pa",
            "Total Assists and Rebounds": "ra", "Total Points, Rebounds, and Assists": "pra",
            "Total Steals and Blocks": "sb"},
    "nfl": {"Total Passing Yards (incl. overtime)": "pass_yds",
            "Total Passing Attempts (incl. overtime)": "pass_att",
            "Total Pass Completions (incl. overtime)": "pass_cmp",
            "Total Passing Touchdowns (incl. overtime)": "pass_td",
            "Total Passing Interceptions (incl. overtime)": "pass_int",
            "Total Rushing Yards (incl. overtime)": "rush_yds",
            "Total Carries (incl. overtime)": "rush_att",
            "Total Receptions (incl. overtime)": "rec",
            "Total Receiving Yards (incl. overtime)": "rec_yds",
            "Total Rushing Plus Receiving Yards (incl. overtime)": "rush_rec_yds",
            "Total Passing Plus Rushing Yards (incl. overtime)": "pass_rush_yds",
            "Total Tackles Plus Assists (incl. overtime)": "tackles"},
    "nhl": {},  # filled from the data; see props_market_map
    "mlb": {"Total Hits": "h", "Total Runs Scored": "r", "Total RBIs": "rbi",
            "Total Hits + Runs + RBIs": "hrr", "Total Walks (Batter)": "bb",
            "Total Strikeouts": "p_k", "Total Outs Recorded": "p_outs",
            "Total Hits Allowed": "p_h", "Earned Runs Allowed": "p_er",
            "Total Walks Allowed": "p_bb"},
}
NHL_PATTERNS = [(r"shots on goal", "sog"), (r"^total goals", "g"), (r"^total assists", "a"),
                (r"^total points", "p"), (r"blocked", "bs"), (r"saves", "sv")]


def market_stat(league: str, market: str) -> str | None:
    if league == "nhl":
        m = market.lower()
        if any(x in m for x in ("milestone", "period", "team", "1st", "first")):
            return None
        for pat, stat in NHL_PATTERNS:
            if re.search(pat, m):
                return stat
        return None
    return MARKETS[league].get(market)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _pair(x, sep) -> tuple[float, float]:
    try:
        a, b = str(x).split(sep)
        return float(a), float(b)
    except ValueError:
        return 0.0, 0.0


def _clock(x) -> float:
    try:
        m, s = str(x).split(":")
        return float(m) + float(s) / 60
    except ValueError:
        return 0.0


def _nba(s: dict, _pos) -> dict:
    b = next((v for k, v in s.items() if isinstance(v, dict) and "PTS" in v), None)
    if not b:
        return {}
    fgm, fga = _pair(b.get("FG"), "-")
    fg3m, fg3a = _pair(b.get("3PT"), "-")
    ftm, fta = _pair(b.get("FT"), "-")
    r = {"min": _f(b.get("MIN")), "pts": _f(b.get("PTS")), "reb": _f(b.get("REB")),
         "ast": _f(b.get("AST")), "stl": _f(b.get("STL")), "blk": _f(b.get("BLK")),
         "tov": _f(b.get("TO")), "oreb": _f(b.get("OREB")), "fgm": fgm, "fga": fga,
         "fg3m": fg3m, "fg3a": fg3a, "fta": fta}
    r.update(pr=r["pts"] + r["reb"], pa=r["pts"] + r["ast"], ra=r["reb"] + r["ast"],
             pra=r["pts"] + r["reb"] + r["ast"], sb=r["stl"] + r["blk"])
    return r


def _nfl(s: dict, _pos) -> dict:
    p, ru, re_, de = (s.get(k) or {} for k in ("passing", "rushing", "receiving", "defensive"))
    cmp_, att = _pair(p.get("C/ATT"), "/")
    sk, _ = _pair(p.get("SACKS"), "-")
    r = {"pass_cmp": cmp_, "pass_att": att, "pass_yds": _f(p.get("YDS")), "pass_td": _f(p.get("TD")),
         "pass_int": _f(p.get("INT")), "sacked": sk, "rush_att": _f(ru.get("CAR")),
         "rush_yds": _f(ru.get("YDS")), "rush_td": _f(ru.get("TD")), "rec": _f(re_.get("REC")),
         "rec_yds": _f(re_.get("YDS")), "rec_td": _f(re_.get("TD")), "tgt": _f(re_.get("TGTS")),
         "tackles": _f(de.get("TOT")), "def_sacks": _f(de.get("SACKS")), "gp": 1.0,
         "has_off": float(bool(p or ru or re_)), "has_def": float(bool(de))}
    r["touch"] = r["rush_att"] + r["tgt"]
    r["rush_rec_yds"] = r["rush_yds"] + r["rec_yds"]
    r["pass_rush_yds"] = r["pass_yds"] + r["rush_yds"]
    r["tot_td"] = r["rush_td"] + r["rec_td"]
    return r


def _nhl(s: dict, _pos) -> dict:
    g = s.get("goalies")
    if g:
        return {"role": "G", "toi": _clock(g.get("TOI")), "sa": _f(g.get("SA")), "sv": _f(g.get("SV")),
                "ga": _f(g.get("GA"))}
    role, b = "F", s.get("forwards") or s.get("skaters")
    if s.get("defenses"):
        role, b = "D", s["defenses"]
    if not b:
        return {}
    r = {"role": role, "toi": _clock(b.get("TOI")), "pptoi": _clock(b.get("PPTOI")),
         "sog": _f(b.get("S")), "g": _f(b.get("G")), "a": _f(b.get("A")),
         "bs": _f(b.get("BS")), "hits": _f(b.get("HT"))}
    r["p"] = r["g"] + r["a"]
    return r


def _mlb(s: dict, _pos) -> dict:
    b, p = s.get("batting") or {}, s.get("pitching") or {}
    r: dict = {}
    if b:
        r.update(ab=_f(b.get("AB")), r=_f(b.get("R")), h=_f(b.get("H")), rbi=_f(b.get("RBI")),
                 hr=_f(b.get("HR")), bb=_f(b.get("BB")), k=_f(b.get("K")))
        r["pa"] = r["ab"] + r["bb"]
        r["hrr"] = r["h"] + r["r"] + r["rbi"]
    if p:
        whole, _, frac = str(p.get("IP", "0")).partition(".")
        outs = _f(whole) * 3 + _f(frac or 0)
        r.update(p_outs=outs, p_h=_f(p.get("H")), p_er=_f(p.get("ER")), p_r=_f(p.get("R")),
                 p_bb=_f(p.get("BB")), p_k=_f(p.get("K")), p_hr=_f(p.get("HR")),
                 sp=float(s.get("sp", 0)))
        r["bf"] = outs + r["p_h"] + r["p_bb"]
    return r


PARSERS = {"nba": _nba, "nfl": _nfl, "nhl": _nhl, "mlb": _mlb}
NBA_POS = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C",
           "FC": "F", "GF": "G", "CF": "C"}
NFL_DEF = {"DE", "DT", "NT", "LB", "OLB", "ILB", "MLB", "CB", "S", "FS", "SS", "DB", "DL"}


def load_games(league: str) -> pd.DataFrame:
    con = connect()
    g = pd.read_sql("SELECT * FROM games WHERE league=? AND final=1 ORDER BY date", con, params=(league,))
    con.close()
    g["date"] = pd.to_datetime(g["date"], utc=True).dt.tz_localize(None)
    lines = pd.DataFrame([json.loads(x) if x else {} for x in g.pop("lines")], index=g.index)
    return pd.concat([g, lines], axis=1)


def load_players(league: str) -> pd.DataFrame:
    con = connect()
    p = pd.read_sql("SELECT * FROM players WHERE league=?", con, params=(league,))
    con.close()
    return p


def load_props(league: str) -> pd.DataFrame:
    """Props with one row per player/market, split into over and under prices.

    A 'close' number is only trusted when its last update came before the game
    started. Otherwise it may be an in-game number, so the open is used instead.
    """
    con = connect()
    p = pd.read_sql("""SELECT p.*, g.date AS game_date FROM props p JOIN games g USING(game_id)
                       WHERE g.league=? AND g.final=1""", con, params=(league,))
    con.close()
    p["stat"] = [market_stat(league, m) for m in p["market"]]
    p = p[p["stat"].notna()].copy()
    return tidy_props(p)


def tidy_props(p: pd.DataFrame) -> pd.DataFrame:
    if p.empty:
        return p
    start = pd.to_datetime(p["game_date"], utc=True)
    upd = pd.to_datetime(p["updated"], utc=True)
    p["pregame_close"] = upd.notna() & (upd <= start)
    p["line"] = p["line_close"].where(p["pregame_close"], p["line_open"])
    p["price"] = p["price_close"].where(p["pregame_close"], p["price_open"])
    keys = ["game_id", "player_id", "market", "stat"]
    first = p[p["seq"] == 0].set_index(keys)
    second = p[p["seq"] == 1].set_index(keys)
    # ESPN lists the over first (verified against outcomes in tests)
    out = first[["line", "price", "line_open", "price_open", "pregame_close"]].rename(
        columns={"price": "over_price", "price_open": "over_price_open"})
    out["under_price"] = second["price"]
    out["under_price_open"] = second["price_open"]
    return out.reset_index().dropna(subset=["line"])


def load_player_games(league: str, players: pd.DataFrame | None = None) -> pd.DataFrame:
    con = connect()
    raw = pd.read_sql("""SELECT pg.*, g.date, g.home_id, g.away_id, g.season, g.season_type
                         FROM player_games pg JOIN games g USING(game_id)
                         WHERE g.league=? AND g.final=1 AND pg.dnp=0""", con, params=(league,))
    con.close()
    parse = PARSERS[league]
    rows = [parse(json.loads(s), None) for s in raw["stats"]]
    stats = pd.DataFrame(rows, index=raw.index)
    df = pd.concat([raw.drop(columns=["stats", "dnp"]), stats], axis=1)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    players = load_players(league) if players is None else players
    return assign_groups(df, league, players)


def assign_groups(df: pd.DataFrame, league: str, players: pd.DataFrame) -> pd.DataFrame:
    ppos = dict(zip(players["player_id"], players["pos"]))
    df["roster_pos"] = df["player_id"].map(ppos)
    pos = df["pos"].fillna(df["roster_pos"])
    if league == "nba":
        df["group"] = pos.map(NBA_POS).fillna("F")
        df = df[df["min"].fillna(0) > 0]
    elif league == "nhl":
        df["group"] = df["role"]
        df = df[df["toi"].fillna(0) > 0]
    elif league == "mlb":
        df["group"] = "BAT"
    else:  # NFL box scores omit position, so use the roster, then usage
        totals = df.groupby("player_id")[["pass_att", "rush_att", "tgt", "has_def"]].transform("sum")
        inferred = np.select(
            [totals["pass_att"] >= 15, totals["rush_att"] > totals["tgt"], totals["tgt"] > 0,
             totals["has_def"] > 0], ["QB", "RB", "WR", "DEF"], "OTH")
        g = pos.where(pos.isin(["QB", "RB", "WR", "TE"]), other=None)
        g = g.where(g.notna(), pos.map(lambda x: "DEF" if x in NFL_DEF else None))
        df["group"] = g.where(g.notna(), inferred)
    df["opp_id"] = np.where(df["team_id"] == df["home_id"], df["away_id"], df["home_id"])
    df["is_home"] = (df["team_id"] == df["home_id"]).astype(float)
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)
