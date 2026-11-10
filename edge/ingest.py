"""Download schedules, box scores, market lines, props, rosters and injuries from
ESPN's public JSON feeds into a local SQLite database. Safe to re-run: finished
games already stored are skipped.

Usage: python -m edge.ingest --since 2024-09-01 [--leagues nba,nfl]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
import requests  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "edge.db"
SITE = "https://site.api.espn.com/apis/site/v2/sports"
CORE = "https://sports.core.api.espn.com/v2/sports"
LEAGUES = {
    "nfl": ("football", 9),
    "nba": ("basketball", 10),
    "nhl": ("hockey", 10),
    "mlb": ("baseball", 3),
}
WORKERS = 6
_session = requests.Session()
_session.headers["User-Agent"] = "edge-model/1.0"


SCHEMA = """
CREATE TABLE IF NOT EXISTS games(
  game_id TEXT PRIMARY KEY, league TEXT, date TEXT, season INT, season_type INT,
  home_id TEXT, away_id TEXT, home_name TEXT, away_name TEXT,
  home_score REAL, away_score REAL, neutral INT, final INT, lines TEXT);
CREATE TABLE IF NOT EXISTS player_games(
  game_id TEXT, player_id TEXT, team_id TEXT, name TEXT, pos TEXT, starter INT,
  dnp INT, stats TEXT, PRIMARY KEY(game_id, player_id, team_id));
CREATE TABLE IF NOT EXISTS props(
  game_id TEXT, player_id TEXT, market TEXT, seq INT, line_open REAL, line_close REAL,
  price_open REAL, price_close REAL, updated TEXT, PRIMARY KEY(game_id, player_id, market, seq));
CREATE TABLE IF NOT EXISTS players(
  league TEXT, player_id TEXT, team_id TEXT, name TEXT, pos TEXT, bats TEXT, throws TEXT,
  PRIMARY KEY(league, player_id));
CREATE INDEX IF NOT EXISTS ix_pg_game ON player_games(game_id);
CREATE INDEX IF NOT EXISTS ix_games_league ON games(league, date);
"""


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB, timeout=60)
    con.executescript(SCHEMA)
    return con


def get(url: str, **params) -> dict | None:
    for attempt in range(4):
        try:
            r = _session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except (requests.RequestException, ValueError):
            time.sleep(1.5 * (attempt + 1))
    return None


def american(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100.0
    try:
        v = float(s.replace("+", ""))
    except ValueError:
        return None
    return v if abs(v) >= 100 else None


def num(x) -> float | None:
    if x is None:
        return None
    s = re.sub(r"^[ou]", "", str(x).strip().lower())
    if s in ("pk", "even"):
        return 0.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def parse_lines(pickcenter: list) -> dict:
    """Open and close prices for moneyline, spread and total from the top provider."""
    if not pickcenter:
        return {}
    pc = pickcenter[0]
    out: dict = {"provider": (pc.get("provider") or {}).get("name")}
    for when in ("open", "close"):
        ml = pc.get("moneyline") or {}
        ps = pc.get("pointSpread") or {}
        tot = pc.get("total") or {}
        side = lambda block, s: ((block.get(s) or {}).get(when) or {})
        out[f"ml_home_{when}"] = american(side(ml, "home").get("odds"))
        out[f"ml_away_{when}"] = american(side(ml, "away").get("odds"))
        out[f"spread_home_{when}"] = num(side(ps, "home").get("line"))
        out[f"spread_home_price_{when}"] = american(side(ps, "home").get("odds"))
        out[f"spread_away_price_{when}"] = american(side(ps, "away").get("odds"))
        out[f"total_{when}"] = num(side(tot, "over").get("line"))
        out[f"over_price_{when}"] = american(side(tot, "over").get("odds"))
        out[f"under_price_{when}"] = american(side(tot, "under").get("odds"))
    return out


def parse_box(summary: dict) -> list[dict]:
    rows = []
    for team in (summary.get("boxscore") or {}).get("players") or []:
        tid = str(team["team"]["id"])
        merged: dict[str, dict] = {}
        for group in team.get("statistics") or []:
            gname = group.get("name") or group.get("type") or "main"
            labels = group.get("labels") or []
            for a in group.get("athletes") or []:
                ath = a.get("athlete") or {}
                pid = str(ath.get("id"))
                row = merged.setdefault(pid, {
                    "player_id": pid, "team_id": tid, "name": ath.get("displayName"),
                    "pos": (ath.get("position") or {}).get("abbreviation"),
                    "starter": 0, "dnp": 0, "stats": {},
                })
                row["starter"] = max(row["starter"], int(bool(a.get("starter"))))
                row["dnp"] = max(row["dnp"], int(bool(a.get("didNotPlay"))))
                stats = a.get("stats") or []
                if stats:
                    row["stats"][gname] = dict(zip(labels, stats))
        rows.extend(merged.values())
    # first listed pitcher is the MLB starter
    for team in (summary.get("boxscore") or {}).get("players") or []:
        for group in team.get("statistics") or []:
            if (group.get("name") or group.get("type")) == "pitching" and group.get("athletes"):
                first = str(group["athletes"][0]["athlete"]["id"])
                for r in rows:
                    if r["player_id"] == first:
                        r["stats"]["sp"] = 1
    return rows


def parse_props(items: list) -> list[tuple]:
    seen: dict[tuple, int] = {}
    out = []
    for it in items:
        ath = ((it.get("athlete") or {}).get("$ref") or "")
        m = re.search(r"athletes/(\d+)", ath)
        if not m:
            continue
        pid = m.group(1)
        market = (it.get("type") or {}).get("name") or ""
        key = (pid, market)
        seen[key] = seen.get(key, -1) + 1
        odds = (it.get("odds") or {}).get("american") or {}
        close = ((it.get("current") or {}).get("target") or {}).get("value")
        opn = ((it.get("open") or {}).get("target") or {}).get("value")
        out.append((pid, market, seen[key], opn, close, american(odds.get("open")),
                    american(odds.get("value")), it.get("lastUpdated")))
    return out


def fetch_game(league: str, eid: str, want_props: bool = True):
    sport = LEAGUES[league][0]
    summary = get(f"{SITE}/{sport}/{league}/summary", event=eid)
    if not summary:
        return None
    items: list = []
    if want_props:
        url = f"{CORE}/{sport}/leagues/{league}/events/{eid}/competitions/{eid}/odds/100/propBets"
        page, pages = 1, 1
        while page <= pages:
            j = get(url, limit=1000, page=page)
            if not j:
                break
            items.extend(j.get("items") or [])
            pages = j.get("pageCount") or 1
            page += 1
    return summary, items


def upsert_game(con, league: str, ev: dict, summary: dict | None, items: list):
    comp = ev["competitions"][0]
    home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
    away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
    final = int(comp["status"]["type"]["completed"])
    lines = parse_lines((summary or {}).get("pickcenter") or [])
    score = lambda c: float(c["score"]) if final and c.get("score") not in (None, "") else None
    gid = ev["id"]
    con.execute(
        "INSERT OR REPLACE INTO games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (gid, league, ev["date"], ev["season"]["year"], ev["season"]["type"],
         str(home["team"]["id"]), str(away["team"]["id"]), home["team"]["displayName"],
         away["team"]["displayName"], score(home), score(away),
         int(bool(comp.get("neutralSite"))), final, json.dumps(lines)))
    if summary and final:
        for r in parse_box(summary):
            con.execute("INSERT OR REPLACE INTO player_games VALUES(?,?,?,?,?,?,?,?)",
                        (gid, r["player_id"], r["team_id"], r["name"], r["pos"], r["starter"],
                         r["dnp"], json.dumps(r["stats"])))
    if items:
        con.execute("DELETE FROM props WHERE game_id=?", (gid,))
        con.executemany("INSERT OR REPLACE INTO props VALUES(?,?,?,?,?,?,?,?,?)",
                        [(gid, *p) for p in parse_props(items)])


def scoreboard(league: str, day: dt.date) -> list[dict]:
    sport = LEAGUES[league][0]
    j = get(f"{SITE}/{sport}/{league}/scoreboard", dates=day.strftime("%Y%m%d"), limit=500)
    return (j or {}).get("events") or []


def backfill(leagues: list[str], since: dt.date, until: dt.date):
    con = connect()
    done = {r[0] for r in con.execute("SELECT game_id FROM games WHERE final=1")}
    days = [until - dt.timedelta(days=i) for i in range((until - since).days + 1)]  # newest first
    for league in leagues:
        t0 = time.time()
        months = LEAGUES[league]
        with ThreadPoolExecutor(WORKERS) as pool:
            boards = list(pool.map(lambda d: scoreboard(league, d), days))
            events = {e["id"]: e for b in boards for e in b
                      if e["season"]["type"] in (2, 3)}
            todo = [e for e in events.values()
                    if e["id"] not in done and e["competitions"][0]["status"]["type"]["completed"]]
            print(f"[{league}] {len(events)} games in window, {len(todo)} to fetch", flush=True)
            futs = {pool.submit(fetch_game, league, e["id"]): e for e in todo}
            n = 0
            for f in as_completed(futs):
                e = futs[f]
                try:
                    res = f.result()
                except Exception as exc:  # keep going; a re-run picks up gaps
                    print("  failed", e["id"], exc, flush=True)
                    continue
                if res:
                    upsert_game(con, league, e, *res)
                n += 1
                if n % 100 == 0:
                    con.commit()
                    print(f"[{league}] {n}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
        con.commit()
        print(f"[{league}] done in {time.time()-t0:.0f}s", flush=True)
        del months
    con.close()


def refresh_rosters(leagues: list[str]):
    con = connect()
    for league in leagues:
        sport = LEAGUES[league][0]
        teams = get(f"{SITE}/{sport}/{league}/teams", limit=100) or {}
        tl = [t["team"] for s in teams.get("sports", []) for l in s["leagues"] for t in l["teams"]]
        def roster(t):
            return t["id"], get(f"{SITE}/{sport}/{league}/teams/{t['id']}/roster") or {}
        current = []
        with ThreadPoolExecutor(WORKERS) as pool:
            for tid, j in pool.map(roster, tl):
                groups = j.get("athletes") or []
                athletes = []
                for g in groups:
                    athletes.extend(g["items"] if isinstance(g, dict) and "items" in g else [g])
                for a in athletes:
                    current.append({"id": str(a["id"]), "team_id": str(tid), "name": a.get("displayName"),
                                    "pos": (a.get("position") or {}).get("abbreviation")})
                    con.execute(
                        "INSERT OR REPLACE INTO players VALUES(?,?,?,?,?,?,?)",
                        (league, str(a["id"]), str(tid), a.get("displayName"),
                         (a.get("position") or {}).get("abbreviation"),
                         (a.get("bats") or {}).get("abbreviation"),
                         (a.get("throws") or {}).get("abbreviation")))
        con.commit()
        if current:  # today's rosters, so the site can list every rostered player
            (ROOT / "data" / f"rosters_{league}.json").write_text(json.dumps(current))
        print(f"[{league}] rosters refreshed: {len(current)} players", flush=True)
    con.close()


def fetch_injuries(league: str) -> dict[str, dict]:
    """player_id -> {status, note} for everyone on the league injury report."""
    sport = LEAGUES[league][0]
    j = get(f"{SITE}/{sport}/{league}/injuries") or {}
    out = {}
    for team in j.get("injuries") or []:
        for inj in team.get("injuries") or []:
            links = (inj.get("athlete") or {}).get("links") or []
            pid = None
            for l in links:
                m = re.search(r"/id/(\d+)", l.get("href", ""))
                if m:
                    pid = m.group(1)
                    break
            if pid:
                out[pid] = {"status": inj.get("status"), "note": inj.get("shortComment")}
    return out


def fetch_slate(league: str, day: dt.date) -> list[dict]:
    """Upcoming games on `day` with their current lines, props and probable starters."""
    out = []
    events = [e for e in scoreboard(league, day)
              if e["competitions"][0]["status"]["type"]["state"] == "pre"]
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda e: fetch_game(league, e["id"]), events))
    for ev, res in zip(events, results):
        summary, items = res if res else ({}, [])
        comp = ev["competitions"][0]
        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        probables = {}
        for c in (home, away):
            for p in c.get("probables") or []:
                ath = p.get("athlete") or {}
                probables[str(c["team"]["id"])] = {"id": str(ath.get("id")), "name": ath.get("displayName")}
        meta = {}
        for c in (home, away):
            t = c["team"]
            meta[str(t["id"])] = {"abbr": t.get("abbreviation"), "short": t.get("shortDisplayName") or t.get("name"),
                                  "logo": t.get("logo"), "color": t.get("color"),
                                  "record": ((c.get("records") or [{}])[0]).get("summary")}
        out.append({
            "team_meta": meta,
            "game_id": ev["id"], "league": league, "date": ev["date"],
            "home_id": str(home["team"]["id"]), "away_id": str(away["team"]["id"]),
            "home_name": home["team"]["displayName"], "away_name": away["team"]["displayName"],
            "neutral": int(bool(comp.get("neutralSite"))),
            "lines": parse_lines(summary.get("pickcenter") or []),
            "props": parse_props(items), "probables": probables,
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2024-09-01")
    ap.add_argument("--until", default=dt.date.today().isoformat())
    ap.add_argument("--leagues", default="nfl,nba,nhl,mlb")
    a = ap.parse_args(argv)
    leagues = a.leagues.split(",")
    refresh_rosters(leagues)
    backfill(leagues, dt.date.fromisoformat(a.since), dt.date.fromisoformat(a.until))


if __name__ == "__main__":
    sys.exit(main())
