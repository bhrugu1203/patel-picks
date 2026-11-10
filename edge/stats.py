"""Season stats for every team and every player in every league -> web/stats.json.

This is the reference layer behind the betting board: standings, scoring, against-the-spread
and over/under records for all teams, and per-game averages (season and last 5) for every
player who has appeared this season, whether or not a prop is posted for him today.

    python -m edge.stats
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import data as D
from .ingest import ROOT, connect, get

LEAGUES = ["nfl", "nba", "nhl", "mlb"]
SPORT = {"nfl": "football", "nba": "basketball", "nhl": "hockey", "mlb": "baseball"}
# columns shown per player category; the first is the usage column
COLS = {
    "nba": {"ALL": ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra"]},
    "nfl": {"QB": ["pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "rush_att", "rush_yds"],
            "RB": ["rush_att", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rush_rec_yds"],
            "WR": ["tgt", "rec", "rec_yds", "rec_td", "rush_yds"],
            "TE": ["tgt", "rec", "rec_yds", "rec_td"],
            "DEF": ["tackles", "def_sacks"]},
    "nhl": {"F": ["toi", "sog", "g", "a", "p", "bs", "hits"], "D": ["toi", "sog", "g", "a", "p", "bs", "hits"],
            "G": ["sa", "sv", "ga"]},
    "mlb": {"BAT": ["pa", "h", "r", "rbi", "hr", "bb", "k", "hrr"], "P": ["bf", "p_outs", "p_k", "p_h", "p_er", "p_bb"]},
}
LABELS = {**D.STAT_LABELS, "min": "Min", "tov": "TO", "toi": "TOI", "hits": "Hits", "sa": "Shots faced", "ga": "Goals allowed",
          "pa": "PA", "bf": "Batters faced", "tgt": "Targets", "rush_td": "Rush TD", "rec_td": "Rec TD", "def_sacks": "Sacks",
          "k": "Strikeouts"}


def team_meta(league: str) -> dict:
    """id -> abbr, name, short, logo, color for every team (cached daily)."""
    f = ROOT / "data" / f"teams_{league}.json"
    if f.exists() and dt.date.fromtimestamp(f.stat().st_mtime) == dt.date.today():
        return json.loads(f.read_text())
    lg = league
    js = get(f"https://site.api.espn.com/apis/site/v2/sports/{SPORT[league]}/{lg}/teams")
    out = {}
    for t in js["sports"][0]["leagues"][0]["teams"]:
        t = t["team"]
        logo = next((l["href"] for l in t.get("logos", []) if "default" in l.get("rel", [])), None) or \
            (t.get("logos") or [{}])[0].get("href")
        out[t["id"]] = {"abbr": t.get("abbreviation"), "name": t.get("displayName"), "short": t.get("shortDisplayName") or t.get("name"),
                        "logo": logo, "color": t.get("color")}
    f.write_text(json.dumps(out))
    return out


def _season(games: pd.DataFrame) -> int:
    reg = games[(games["season_type"] == 2) & (games["final"] == 1)]
    counts = reg.groupby("season").size()
    return int(counts.index.max())


def teams_table(league: str, meta: dict) -> tuple[list[dict], int]:
    con = connect()
    g = pd.read_sql("SELECT * FROM games WHERE league=?", con, params=(league,))
    con.close()
    season = _season(g)
    s = g[(g["season"] == season) & (g["season_type"] == 2) & (g["final"] == 1)].copy()
    s["date"] = pd.to_datetime(s["date"], utc=True)
    s = s.sort_values("date")
    lines = s["lines"].map(lambda x: json.loads(x) if x else {})
    s["spread"] = lines.map(lambda d: d.get("spread_home_close", d.get("spread_home_open")))
    s["total"] = lines.map(lambda d: d.get("total_close", d.get("total_open")))
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        r = pd.DataFrame({"team_id": s[f"{side}_id"], "opp_id": s[f"{opp}_id"], "date": s["date"], "home": side == "home",
                          "pf": s[f"{side}_score"], "pa": s[f"{opp}_score"], "total": s["total"],
                          "spread": s["spread"] if side == "home" else -s["spread"]})
        rows.append(r)
    t = pd.concat(rows).sort_values("date")
    t["win"] = t["pf"] > t["pa"]
    t["cover"] = np.where(t["spread"].isna(), np.nan, np.sign(t["pf"] - t["pa"] + t["spread"]))
    t["ou"] = np.where(t["total"].isna(), np.nan, np.sign(t["pf"] + t["pa"] - t["total"]))
    now = pd.Timestamp.now(tz="UTC")
    up = g[(g["final"] == 0)].copy()
    up["date"] = pd.to_datetime(up["date"], utc=True)
    up = up[up["date"] > now - pd.Timedelta(hours=4)].sort_values("date")
    out = []
    for tid, x in t.groupby("team_id"):
        if meta and tid not in meta:  # all-star / exhibition sides
            continue
        last10 = x.tail(10)
        streak, sgn = 0, None
        for w in x["win"].iloc[::-1]:
            if sgn is None:
                sgn = w
            if w != sgn:
                break
            streak += 1
        nxt = up[(up["home_id"] == tid) | (up["away_id"] == tid)].head(1)
        nx = None
        if len(nxt):
            n = nxt.iloc[0]
            home = n["home_id"] == tid
            nx = {"date": n["date"].isoformat(), "opp_id": n["away_id"] if home else n["home_id"],
                  "opp": n["away_name"] if home else n["home_name"], "home": bool(home), "game_id": n["game_id"]}
        m = meta.get(tid, {})
        name = m.get("name") or (s.loc[s["home_id"] == tid, "home_name"].head(1).tolist() or [tid])[0]
        cv, ou = x["cover"].dropna(), x["ou"].dropna()
        out.append({
            "team_id": tid, "name": name, "abbr": m.get("abbr"), "short": m.get("short"), "logo": m.get("logo"), "color": m.get("color"),
            "gp": int(len(x)), "w": int(x["win"].sum()), "l": int((~x["win"]).sum()),
            "home": f"{int(x[x.home]['win'].sum())}-{int((~x[x.home]['win']).sum())}",
            "away": f"{int(x[~x.home]['win'].sum())}-{int((~x[~x.home]['win']).sum())}",
            "l10": f"{int(last10['win'].sum())}-{int((~last10['win']).sum())}",
            "streak": (("W" if sgn else "L") + str(streak)) if sgn is not None else "",
            "pf": round(float(x["pf"].mean()), 1), "pa": round(float(x["pa"].mean()), 1),
            "diff": round(float((x["pf"] - x["pa"]).mean()), 1),
            "ats": f"{int((cv > 0).sum())}-{int((cv < 0).sum())}-{int((cv == 0).sum())}",
            "ou": f"{int((ou > 0).sum())}-{int((ou < 0).sum())}-{int((ou == 0).sum())}",
            "next": nx,
        })
    for key, asc in (("pf", False), ("pa", True), ("diff", False)):
        order = sorted(out, key=lambda r: r[key], reverse=not asc)
        for i, r in enumerate(order, 1):
            r[key + "_rank"] = i
    out.sort(key=lambda r: (-(r["w"] / max(1, r["gp"])), -r["diff"]))
    return out, season


def _cat(league: str, x: pd.DataFrame) -> str:
    if league == "nba":
        return "ALL"
    if league == "mlb":
        return "P" if x["bf"].fillna(0).sum() > x["pa"].fillna(0).sum() else "BAT"
    if league == "nhl":
        return "G" if x["sa"].fillna(0).sum() > 0 and x["toi"].fillna(0).mean() > 30 else (x["group"].mode().iloc[0] if x["group"].notna().any() else "F")
    g = x["group"].mode().iloc[0] if x["group"].notna().any() else "DEF"
    return g if g in COLS["nfl"] else "DEF"


def players_table(league: str, season: int, teams: dict) -> list[dict]:
    pg = D.load_player_games(league)
    pg = pg[(pg["season"] == season) & (pg["season_type"] == 2)].sort_values("date")
    roster = D.load_players(league)
    cur_team = dict(zip(roster["player_id"], roster["team_id"]))
    pos = dict(zip(roster["player_id"], roster["pos"]))
    out = []
    for pid, x in pg.groupby("player_id"):
        cat = _cat(league, x)
        cols = [c for c in COLS[league][cat] if c in x.columns]
        if league == "mlb":
            x = x[x["bf"].fillna(0) > 0] if cat == "P" else x[x["pa"].fillna(0) > 0]
        elif league == "nhl" and cat == "G":
            x = x[x["sa"].fillna(0) > 0]
        elif league == "nfl" and cat != "DEF":
            x = x[x["has_off"].fillna(0) > 0] if "has_off" in x else x
        if x.empty:
            continue
        tid = cur_team.get(pid) or x["team_id"].iloc[-1]
        last = x.tail(5)
        rec = {"player_id": pid, "name": x["name"].iloc[-1], "team_id": tid, "pos": pos.get(pid) or x["pos"].iloc[-1] or cat,
               "cat": cat, "gp": int(len(x)), "last": x["date"].iloc[-1].strftime("%Y-%m-%d"),
               "avg": {c: round(float(x[c].mean()), 1) for c in cols},
               "l5": {c: round(float(last[c].mean()), 1) for c in cols}}
        if league == "nhl" and cat == "G":
            sa, sv = x["sa"].sum(), x["sv"].sum()
            rec["avg"]["sv_pct"] = round(float(sv / sa), 3) if sa else None
        if league == "mlb" and cat == "P":
            outs = x["p_outs"].sum()
            rec["avg"]["era"] = round(float(x["p_er"].sum() * 27 / outs), 2) if outs else None
            rec["avg"]["ip"] = round(float(outs / 3), 1)
        if league == "mlb" and cat == "BAT" and x["ab"].sum():
            rec["avg"]["avg_"] = round(float(x["h"].sum() / x["ab"].sum()), 3)
        out.append(rec)
    return [r for r in out if not teams or r["team_id"] in teams]


def build() -> dict:
    res = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "labels": LABELS, "cols": COLS, "leagues": {}}
    for lg in LEAGUES:
        try:
            meta = team_meta(lg)
        except Exception:
            meta = {}
        teams, season = teams_table(lg, meta)
        res["leagues"][lg] = {"season": season, "teams": teams, "team_meta": meta,
                              "players": players_table(lg, season, meta)}
    return res


def write():
    obj = build()
    f = ROOT / "web" / "stats.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), default=lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)))
    tmp.replace(f)
    return obj


if __name__ == "__main__":
    o = write()
    for lg, v in o["leagues"].items():
        print(lg, "season", v["season"], "teams", len(v["teams"]), "players", len(v["players"]))
