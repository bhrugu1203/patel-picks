"""Keep the board steady between updates.

Two jobs:

guard(priced)      A bad fetch (ESPN down, an empty scoreboard, props missing) must not wipe or
                   scramble the board. Each league's last good board is saved in the database;
                   if a league fails to update, or comes back with most of its players missing,
                   the last good board is shown instead (minus games that have already started).

pick_of_day(...)   The pick of the day is chosen once per game day and then locked. It is only
                   replaced if it becomes invalid before its game starts: the player is ruled
                   out, or the game or the prop disappears from the board for two updates in a
                   row (postponed, moved or pulled). Once the game starts it stays put for the
                   rest of that day.

Game days are calendar dates in US Eastern time. On a day with no games for a sport, the
pick of the day comes from the next day that has games.
"""
from __future__ import annotations

import datetime as dt
import json
from zoneinfo import ZoneInfo

from .ingest import connect

ET = ZoneInfo("America/New_York")
SCOPES = ["all", "nfl", "nba", "nhl", "mlb"]
MAX_MISSES = 2              # updates in a row a locked pick may be missing before it is replaced
CACHE_MAX_AGE_H = 12        # never fall back to a saved board older than this
MIN_PLAYERS_TO_JUDGE = 20   # below this, a drop in players is not treated as a broken fetch
KEEP_SHARE = 0.4            # a fresh board with under 40% of the saved players (same games) is suspect
START_BUFFER = dt.timedelta(minutes=5)

SCHEMA = """
CREATE TABLE IF NOT EXISTS board_cache(league TEXT PRIMARY KEY, saved TEXT, body TEXT);
CREATE TABLE IF NOT EXISTS potd(
  scope TEXT, day TEXT, chosen TEXT, pick TEXT, misses INT DEFAULT 0, replaced TEXT,
  PRIMARY KEY(scope, day));
"""


def _con():
    con = connect()
    con.executescript(SCHEMA)
    return con


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _when(s) -> dt.datetime | None:
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def game_day(start) -> str | None:
    """Calendar date (YYYY-MM-DD) a game is played on, in US Eastern time."""
    t = _when(start)
    return t.astimezone(ET).date().isoformat() if t else None


def _json_default(x):
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------- bad-fetch guard

def _upcoming_only(block: dict | None, now: dt.datetime) -> dict | None:
    """A saved league board with games that have already started removed."""
    if not block:
        return None
    dates = block.get("game_dates") or {}
    keep = {gid for gid, d in dates.items() if (_when(d) or now) > now}
    if not keep:
        return None
    out = dict(block)
    out["game_dates"] = {gid: d for gid, d in dates.items() if gid in keep}
    out["games"] = [g for g in block.get("games", []) if g["game_id"] in keep]
    out["players"] = [p for p in block.get("players", []) if p["game_id"] in keep]
    return out


def _looks_broken(new: dict, old: dict | None) -> str | None:
    """Why a fresh league board looks like a failed fetch rather than real news, or None."""
    if not old:
        return None
    new_games = set((new.get("game_dates") or {}))
    old_games = set((old.get("game_dates") or {}))
    if not new_games and len(old_games) >= 2:
        return f"came back with no games while {len(old_games)} upcoming games were on the board"
    both = new_games & old_games
    old_n = sum(1 for p in old.get("players", []) if p["game_id"] in both)
    new_n = sum(1 for p in new.get("players", []) if p["game_id"] in both)
    if old_n >= MIN_PLAYERS_TO_JUDGE and new_n < KEEP_SHARE * old_n:
        return f"came back with {new_n} players for games that had {old_n}"
    return None


def guard(priced: list[dict], leagues: list[str], now: dt.datetime | None = None) -> tuple[list[dict], dict]:
    """Swap in the last good board for any league whose update failed or looks broken.

    Returns the leagues to publish and a note per league that fell back.
    """
    now = now or _utcnow()
    con = _con()
    saved = {}
    for lg, when, body in con.execute("SELECT league, saved, body FROM board_cache").fetchall():
        t = _when(when)
        if t and now - t <= dt.timedelta(hours=CACHE_MAX_AGE_H):
            try:
                saved[lg] = json.loads(body)
            except ValueError:
                pass
    by_lg = {p["league"]: p for p in priced}
    out, notes = [], {}
    for lg in leagues:
        new = by_lg.get(lg)
        old = _upcoming_only(saved.get(lg), now)
        if new is None:
            if old:
                out.append({**old, "stale": True})
                notes[lg] = "update failed: showing the last good board"
            continue
        why = _looks_broken(new, old)
        if why:
            out.append({**old, "stale": True})
            notes[lg] = f"update {why}: showing the last good board"
            continue
        out.append(new)
        if new.get("players") or new.get("games"):
            con.execute("INSERT OR REPLACE INTO board_cache VALUES(?,?,?)",
                        (lg, now.isoformat(timespec="seconds"), json.dumps(new, default=_json_default)))
    con.commit()
    con.close()
    return out, notes


# ---------------------------------------------------------------- pick of the day

def _g_score(x: dict) -> float:
    if x.get("ev") is not None:
        return x["ev"] + 0.5 * (x["p"] - 0.5)
    return 0.6 * (x["p"] - 0.5) - 0.02


def _role_flag(x: dict, players: dict) -> bool:
    """Same "streak fade" test the site shows: averages far from the projection, against the pick."""
    s = ((players.get((x["player_id"], x["game_id"])) or {}).get("stats") or {}).get(x["stat"])
    if not s or s.get("proj") is None or s.get("season") is None or s.get("l10") is None or s["proj"] <= 0:
        return False
    hi = s["season"] > s["proj"] * 1.25 and s["l10"] > s["proj"] * 1.25
    lo = s["season"] < s["proj"] * 0.75 and s["l10"] < s["proj"] * 0.75
    return (hi and x["side"] == "under") or (lo and x["side"] == "over")


def _ranked(boards: dict, players: dict, pool_filter) -> list[dict]:
    """Candidates in the order the site has always used for its play of the day:
    best bets first, then every other play by value, skipping streak fades when possible."""
    sheet = [x for x in boards.get("sheet", []) if pool_filter(x)]
    plays = boards.get("overs", []) + boards.get("unders", [])
    seen = {(x["player_id"], x["game_id"]) for x in plays}
    plays = plays + [x for x in boards.get("leans", []) if (x["player_id"], x["game_id"]) not in seen]
    plays = sorted([x for x in plays if pool_filter(x)], key=lambda x: -_g_score(x))
    clean = [x for x in sheet if not _role_flag(x, players)] + [x for x in plays if not _role_flag(x, players)]
    return clean + sheet + plays


KEEP_FIELDS = ("league", "game_id", "player_id", "player", "team", "team_id", "opp", "opp_id", "pos", "stat",
               "label", "side", "line", "book", "price", "p", "ev", "proj", "status", "why", "conf", "hit_l10",
               "hit_opp", "fair", "game_day", "start")


def _freeze(x: dict) -> dict:
    return {k: x.get(k) for k in KEEP_FIELDS}


def _choose(boards: dict, players: dict, starts: dict, ruled_out: dict, scope: str,
            now: dt.datetime) -> dict | None:
    def ok(x):
        if scope != "all" and x["league"] != scope:
            return False
        if x["player_id"] in ruled_out.get(x["league"], set()):
            return False
        s = ((players.get((x["player_id"], x["game_id"])) or {}).get("stats") or {}).get(x["stat"])
        if not s or not s.get("offers"):
            return False
        t = _when(starts.get(x["game_id"]) or x.get("start"))
        return t is not None and t > now + START_BUFFER
    ranked = _ranked(boards, players, ok)
    if not ranked:
        return None
    first_day = min(game_day(starts.get(x["game_id"]) or x.get("start")) for x in ranked)
    for x in ranked:
        if game_day(starts.get(x["game_id"]) or x.get("start")) == first_day:
            pick = _freeze(x)
            pick["start"] = starts.get(x["game_id"]) or x.get("start")
            pick["game_day"] = first_day
            return pick
    return None


def _check(pick: dict, players: dict, starts: dict, ruled_out: set, now: dt.datetime, misses: int):
    """(still valid?, reason if not, new miss count)."""
    start = _when(pick.get("start"))
    if start is not None and start <= now:
        return True, None, 0  # game under way or over: the pick is final for the day
    if pick["player_id"] in ruled_out.get(pick["league"], set()):
        return False, f"{pick['player']} was ruled out", misses
    gid = pick["game_id"]
    if gid in starts and game_day(starts[gid]) != pick.get("game_day"):
        return False, "the game moved to another day", misses
    p = players.get((pick["player_id"], gid))
    s = ((p or {}).get("stats") or {}).get(pick["stat"])
    present = gid in starts and s is not None and bool(s.get("offers"))
    if present:
        return True, None, 0
    misses += 1
    if misses >= MAX_MISSES:
        what = "the game" if gid not in starts else "the prop"
        return False, f"{what} came off the board (postponed, moved or pulled)", misses
    return True, None, misses


def pick_of_day(boards: dict, priced: list[dict], now: dt.datetime | None = None) -> dict:
    """The locked pick of the day for every sport and for all sports together."""
    now = now or _utcnow()
    today = now.astimezone(ET).date().isoformat()
    starts = {gid: d for lg in priced for gid, d in (lg.get("game_dates") or {}).items()}
    players = {(p["player_id"], p["game_id"]): p for lg in priced for p in lg.get("players", [])}
    ruled_out = {lg["league"]: set(lg.get("ruled_out") or []) for lg in priced}
    con = _con()
    out = {}
    for scope in SCOPES:
        row = con.execute("SELECT day, chosen, pick, misses FROM potd WHERE scope=? AND day>=? ORDER BY day LIMIT 1",
                          (scope, today)).fetchone()
        replaced = None
        if row:
            day, chosen, body, misses = row
            pick = json.loads(body)
            valid, reason, misses = _check(pick, players, starts, ruled_out, now, misses or 0)
            if valid:
                con.execute("UPDATE potd SET misses=? WHERE scope=? AND day=?", (misses, scope, day))
                out[scope] = {**pick, "locked_at": chosen, "replaced": None}
                continue
            con.execute("DELETE FROM potd WHERE scope=? AND day=?", (scope, day))
            replaced = reason
        pick = _choose(boards, players, starts, ruled_out, scope, now)
        if not pick:
            continue
        chosen = now.isoformat(timespec="seconds")
        con.execute("INSERT OR REPLACE INTO potd(scope, day, chosen, pick, misses, replaced) VALUES(?,?,?,?,0,?)",
                    (scope, pick["game_day"], chosen, json.dumps(pick, default=_json_default), replaced))
        out[scope] = {**pick, "locked_at": chosen, "replaced": replaced}
    con.execute("DELETE FROM potd WHERE day < ?", ((now - dt.timedelta(days=30)).astimezone(ET).date().isoformat(),))
    con.commit()
    con.close()
    for v in out.values():
        t = _when(v.get("start"))
        v["started"] = bool(t and t <= now)
    return out
