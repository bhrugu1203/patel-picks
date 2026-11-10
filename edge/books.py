"""Live player-prop prices from DraftKings, FanDuel, BetMGM and other books via The Odds API.

Put your key in a file named .env at the project root:
    ODDS_API_KEY=your_key_here
Free keys: https://the-odds-api.com (500 credits a month). Each event costs one
credit per market per region, so the refresh interval is kept separate from ESPN's.
Every pull is also saved to the database, building a priced history for future backtests.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import unicodedata

import pandas as pd

from .ingest import ROOT, _session, connect

API = "https://api.the-odds-api.com/v4"
SPORT = {"nfl": "americanfootball_nfl", "nba": "basketball_nba", "nhl": "icehockey_nhl", "mlb": "baseball_mlb"}
BOOKS = "draftkings,fanduel,betmgm,williamhill_us,espnbet,betrivers,fanatics"
# Odds API market key -> model stat
MARKETS = {
    "nfl": {"player_pass_yds": "pass_yds", "player_pass_attempts": "pass_att",
            "player_pass_completions": "pass_cmp", "player_pass_tds": "pass_td",
            "player_pass_interceptions": "pass_int", "player_rush_yds": "rush_yds",
            "player_rush_attempts": "rush_att", "player_receptions": "rec",
            "player_reception_yds": "rec_yds", "player_rush_reception_yds": "rush_rec_yds",
            "player_pass_rush_yds": "pass_rush_yds", "player_tackles_assists": "tackles"},
    "nba": {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
            "player_threes": "fg3m", "player_points_rebounds_assists": "pra"},
    "nhl": {"player_shots_on_goal": "sog", "player_points": "p", "player_goalie_saves": "sv"},
    "mlb": {"pitcher_strikeouts": "p_k", "batter_hits": "h", "batter_total_bases": None},
}
SCHEMA = """CREATE TABLE IF NOT EXISTS book_props(
  pulled TEXT, league TEXT, event TEXT, home TEXT, away TEXT, commence TEXT, book TEXT,
  stat TEXT, player TEXT, side TEXT, line REAL, price REAL)"""


def api_key() -> str | None:
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"].strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"') or None
    return None


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def parse_event(league: str, ev: dict, pulled: str) -> list[tuple]:
    """Flatten one Odds API event-odds response into rows."""
    rows = []
    mk = MARKETS[league]
    for bk in ev.get("bookmakers") or []:
        for m in bk.get("markets") or []:
            stat = mk.get(m.get("key"))
            if not stat:
                continue
            for o in m.get("outcomes") or []:
                side = (o.get("name") or "").lower()
                if side not in ("over", "under") or o.get("point") is None:
                    continue
                rows.append((pulled, league, ev["id"], ev.get("home_team"), ev.get("away_team"),
                             ev.get("commence_time"), bk["key"], stat, o.get("description"), side,
                             float(o["point"]), float(o["price"])))
    return rows


def fetch(league: str, hours_ahead: float = 36) -> tuple[pd.DataFrame, str]:
    """Pull current prices for upcoming games. Returns (rows, status message)."""
    key = api_key()
    cols = ["pulled", "league", "event", "home", "away", "commence", "book", "stat", "player", "side", "line", "price"]
    if not key:
        return pd.DataFrame(columns=cols), "no ODDS_API_KEY in .env"
    sport = SPORT[league]
    r = _session.get(f"{API}/sports/{sport}/events", params={"apiKey": key}, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame(columns=cols), f"events request failed ({r.status_code}): {r.text[:120]}"
    now = dt.datetime.now(dt.timezone.utc)
    events = [e for e in r.json()
              if 0 <= (dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) - now).total_seconds() <= hours_ahead * 3600]
    markets = ",".join(k for k, v in MARKETS[league].items() if v)
    pulled = now.isoformat(timespec="seconds")
    rows, remaining = [], None
    for e in events:
        r = _session.get(f"{API}/sports/{sport}/events/{e['id']}/odds", timeout=30, params={
            "apiKey": key, "regions": "us", "markets": markets, "oddsFormat": "american", "bookmakers": BOOKS})
        remaining = r.headers.get("x-requests-remaining", remaining)
        if r.status_code != 200:
            continue
        rows.extend(parse_event(league, r.json(), pulled))
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        con = connect()
        con.execute(SCHEMA)
        con.executemany(f"INSERT INTO book_props VALUES({','.join('?' * len(cols))})", rows)
        con.commit()
        con.close()
    return df, f"{len(events)} games, {len(df)} prices, {remaining} credits left"


def attach(book: pd.DataFrame, roster: pd.DataFrame, slate_games: list[dict]) -> pd.DataFrame:
    """Match book rows to ESPN player ids by normalized name within the game's two teams."""
    if book.empty:
        return book
    name_to_game = {}
    for g in slate_games:
        for side in ("home_name", "away_name"):
            name_to_game[norm_name(g[side])] = g["game_id"]
    book = book.copy()
    book["game_id"] = book["home"].map(lambda x: name_to_game.get(norm_name(x)))
    teams = {g["game_id"]: {g["home_id"], g["away_id"]} for g in slate_games}
    roster = roster.assign(key=roster["name"].map(norm_name))
    ids = []
    for r in book.itertuples():
        cand = roster[(roster["key"] == norm_name(r.player)) & roster["team_id"].isin(teams.get(r.game_id, set()))]
        ids.append(cand["player_id"].iloc[0] if len(cand) else None)
    book["player_id"] = ids
    return book.dropna(subset=["game_id", "player_id"])
