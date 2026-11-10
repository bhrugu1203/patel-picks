"""Record every published pick and parlay, follow its line to kickoff, and grade it after the game.

The record is the honest scoreboard: wins, losses, units, ROI, and closing-line value
(whether the number we posted beat the last number before the game started).
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from . import data as D, pricing as X
from .ingest import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS picks(
  pick_id TEXT PRIMARY KEY, posted TEXT, league TEXT, game_id TEXT, player_id TEXT, player TEXT,
  team TEXT, opp TEXT, stat TEXT, label TEXT, side TEXT, book TEXT, line REAL, price REAL, p REAL,
  ev REAL, conf INT, close_line REAL, close_price REAL, actual REAL, result TEXT, profit REAL, graded TEXT);
CREATE TABLE IF NOT EXISTS parlays(
  parlay_id TEXT PRIMARY KEY, posted TEXT, name TEXT, legs TEXT, price REAL, p REAL, ev REAL,
  result TEXT, profit REAL, graded TEXT);
"""


def _con():
    con = connect()
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(picks)")}
    for c in ("grade", "src"):  # added later: model grade, and which list the play was shown on
        if c not in cols:
            con.execute(f"ALTER TABLE picks ADD COLUMN {c} TEXT")
            if c == "src":
                con.execute("UPDATE picks SET src='best' WHERE src IS NULL")
    for pid, ev, pr in con.execute("SELECT pick_id, ev, p FROM picks WHERE grade IS NULL").fetchall():
        con.execute("UPDATE picks SET grade=? WHERE pick_id=?", (grade_of(ev, pr or 0), pid))
    return con


def grade_of(ev, p) -> str:
    """Same grade the site shows: value against the price, plus hit chance."""
    if ev is not None and ev == ev:
        return "A+" if ev >= 0.10 and p >= 0.55 else "A" if ev >= 0.06 else "B" if ev >= 0.03 else "C"
    return "A" if p >= 0.64 else "B" if p >= 0.58 else "C"


def _pid(x: dict) -> str:
    return f"{x['game_id']}|{x['player_id']}|{x['stat']}|{x['side']}"


def record(boards: dict, priced: list[dict]):
    """Store new picks; refresh the latest line/price on open ones (becomes the closing number)."""
    con = _con()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    def ins(x, src):
        con.execute("""INSERT OR IGNORE INTO picks(pick_id, posted, league, game_id, player_id, player, team, opp,
                       stat, label, side, book, line, price, p, ev, conf, grade, src)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (_pid(x), now, x["league"], x["game_id"], x["player_id"], x["player"], x["team"], x["opp"],
                     x["stat"], x["label"], x["side"], x["book"], x["line"], x.get("price"), x["p"], x.get("ev"),
                     x.get("conf"), grade_of(x.get("ev"), x["p"]), src))
    # every play shown on the site is logged the first time it appears, so the record covers them all
    for x in boards.get("overs", []) + boards.get("unders", []):
        ins(x, "play")
    for x in boards.get("leans", []):
        ins(x, "lean")
    for x in boards.get("sheet", []):
        ins(x, "best")
        con.execute("UPDATE picks SET src='best' WHERE pick_id=?", (_pid(x),))
    for pl in boards.get("parlays", []):
        legs = [_pid(l) for l in pl["legs"]]
        for l in pl["legs"]:  # legs must be graded too, even ones that came from the most-likely list
            con.execute("""INSERT OR IGNORE INTO picks(pick_id, posted, league, game_id, player_id, player, team, opp,
                           stat, label, side, book, line, price, p, ev, conf) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (_pid(l), now, l["league"], l["game_id"], l["player_id"], l["player"], l["team"], l["opp"],
                         l["stat"], l["label"], l["side"], l["book"], l["line"], l["price"], l["p"], l.get("ev"), l.get("conf")))
        key = pl["name"] + "|" + ",".join(sorted(legs))
        con.execute("INSERT OR IGNORE INTO parlays(parlay_id, posted, name, legs, price, p, ev) VALUES(?,?,?,?,?,?,?)",
                    (key, now, pl["name"], json.dumps(legs), pl["price"], pl["p"], pl["ev"]))
    # closing number: keep overwriting while the game is still on the board
    live = {}
    for lg in priced:
        for p in lg["players"]:
            for stat, s in p["stats"].items():
                for o in s["offers"]:
                    live[(p["game_id"], p["player_id"], stat, o["book"])] = o
    for row in con.execute("SELECT pick_id, game_id, player_id, stat, side, book FROM picks WHERE result IS NULL").fetchall():
        o = live.get((row[1], row[2], row[3], row[5]))
        if o and o.get(row[4]) is not None:
            con.execute("UPDATE picks SET close_line=?, close_price=? WHERE pick_id=?", (o["line"], o[row[4]], row[0]))
    con.commit()
    con.close()


def grade():
    """Grade open picks whose games are final."""
    con = _con()
    open_ = pd.read_sql("SELECT * FROM picks WHERE result IS NULL", con)
    if open_.empty:
        con.close()
        return
    finals = pd.read_sql(f"""SELECT game_id, league FROM games WHERE final=1 AND game_id IN
                             ({','.join('?' * open_['game_id'].nunique())})""", con, params=list(open_["game_id"].unique()))
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for league, gids in finals.groupby("league")["game_id"]:
        box = pd.read_sql(f"""SELECT game_id, player_id, stats FROM player_games WHERE dnp=0 AND game_id IN
                              ({','.join('?' * len(gids))})""", con, params=list(gids))
        parse = D.PARSERS[league]
        stats = {(r.game_id, r.player_id): parse(json.loads(r.stats), None) for r in box.itertuples()}
        for r in open_[open_["game_id"].isin(set(gids))].itertuples():
            st = stats.get((r.game_id, r.player_id))
            val = None if st is None else st.get(r.stat)
            if val is None:
                result, profit = "void", 0.0
            elif val == r.line:
                result, profit = "push", 0.0
            else:
                won = (val > r.line) == (r.side == "over")
                result = "win" if won else "loss"
                if pd.isna(r.price):  # line posted without odds: graded for hit rate only
                    profit = None
                else:
                    profit = float(X.payout(r.price)) if won else -1.0
            con.execute("UPDATE picks SET actual=?, result=?, profit=?, graded=? WHERE pick_id=?",
                        (val, result, profit, now, r.pick_id))
    # parlays: settle once every leg is graded (a void or push leg drops out at even odds)
    res = dict(con.execute("SELECT pick_id, result FROM picks").fetchall())
    prices = dict(con.execute("SELECT pick_id, price FROM picks").fetchall())
    for pid, legs, price in con.execute("SELECT parlay_id, legs, price FROM parlays WHERE result IS NULL").fetchall():
        legs = json.loads(legs)
        rs = [res.get(l) for l in legs]
        if any(r is None for r in rs):
            continue
        if "loss" in rs:
            result, profit = "loss", -1.0
        else:
            live_legs = [l for l, r in zip(legs, rs) if r == "win"]
            if not live_legs:
                result, profit = "void", 0.0
            else:
                dec = 1.0
                for l in live_legs:
                    p = prices[l]
                    dec *= 1 + (p / 100 if p > 0 else 100 / -p)
                result, profit = "win", dec - 1
        con.execute("UPDATE parlays SET result=?, profit=?, graded=? WHERE parlay_id=?", (result, profit, now, pid))
    con.commit()
    con.close()


def summary() -> dict:
    con = _con()
    picks = pd.read_sql("SELECT * FROM picks ORDER BY posted DESC", con)
    parl = pd.read_sql("SELECT * FROM parlays ORDER BY posted DESC", con)
    con.close()
    if not picks.empty and "src" in picks:
        picks["src"] = picks["src"].fillna("best")
    best = picks[picks["src"] == "best"] if not picks.empty else picks
    plays = picks[picks["src"].isin(["best", "play"]) & picks["price"].notna()] if not picks.empty else picks
    leans = picks[picks["src"] == "lean"] if not picks.empty else picks
    out = {"overall": _rec(best), "plays": _rec(plays), "leans": _rec(leans), "by_league": {}, "by_day": [], "recent": [],
           "by_grade": {}, "by_side": {}, "by_stat": {}, "plays_by_league": {},
           "parlays": _rec(parl), "parlay_recent": []}
    if picks.empty:
        return out
    for lg, s in best.groupby("league"):
        out["by_league"][lg] = _rec(s)
    for key, col in (("by_grade", "grade"), ("by_side", "side"), ("by_stat", "label"), ("plays_by_league", "league")):
        for k, s in plays.groupby(col):
            out[key][k] = _rec(s)
    picks["day"] = pd.to_datetime(picks["posted"]).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    for day, s in picks[picks.index.isin(plays.index)].groupby("day"):
        out["by_day"].append({"day": day, **_rec(s)})
    out["by_day"].sort(key=lambda r: r["day"], reverse=True)
    picks = picks[picks["src"].isin(["best", "play", "lean"])]
    cols = ["posted", "league", "player", "team", "opp", "label", "side", "book", "line", "price", "conf", "ev",
            "close_line", "close_price", "actual", "result", "profit", "grade", "src"]
    out["recent"] = picks[cols].head(200).astype(object).where(picks[cols].head(200).notna(), None).to_dict("records")
    if not parl.empty:
        pc = ["posted", "name", "legs", "price", "p", "result", "profit"]
        out["parlay_recent"] = parl[pc].head(60).astype(object).where(parl[pc].head(60).notna(), None).to_dict("records")
    return out


def _rec(s: pd.DataFrame) -> dict:
    if s is None or s.empty:
        return {"n": 0, "pending": 0}
    g = s[s["result"].isin(["win", "loss", "push"])]
    w, l, p = (g["result"] == "win").sum(), (g["result"] == "loss").sum(), (g["result"] == "push").sum()
    priced = g[g["profit"].notna()]
    units = float(priced["profit"].sum()) if len(priced) else 0.0
    out = {"n": int(len(s)), "pending": int(s["result"].isna().sum()), "wins": int(w), "losses": int(l), "pushes": int(p),
           "units": round(units, 2), "roi": round(units / len(priced), 4) if len(priced) else None,
           "hit": round(w / (w + l), 4) if (w + l) else None}
    if "close_price" in s.columns:
        c = s.dropna(subset=["close_price", "price"])
        if len(c):
            # beat the close when our price implies a lower chance than the closing price on the same line
            same = c[c["close_line"] == c["line"]]
            better_line = ((c["side"] == "over") & (c["close_line"] > c["line"])) | ((c["side"] == "under") & (c["close_line"] < c["line"]))
            beat = ((X.implied(same["price"]) < X.implied(same["close_price"]))).sum() + better_line.sum()
            out["clv_beat"] = round(float(beat) / len(c), 4)
    return out
