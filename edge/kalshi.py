"""Kalshi player-prop prices for the Parlays web app (web/parlay/).

    python -m edge.kalshi

Kalshi's public API refuses requests that come from a web page (it answers 403 to any request with
a browser Origin), so the phone app can't read it directly. This runs with every board update and
saves the open prop ladders, plus the model's calibration weights, to web/kalshi.json, which the
app reads from the same site. Read-only and keyless: it only lists public markets.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
API = "https://api.elections.kalshi.com/trade-api/v2/events"

# series ticker, league, the model's stat name (same list as the macOS parlay popup)
SERIES = [
    ("KXNBAPTS", "nba", "pts"), ("KXNBAPTSPROT", "nba", "pts"),
    ("KXNBAREB", "nba", "reb"), ("KXNBAREBPROT", "nba", "reb"),
    ("KXNBAAST", "nba", "ast"), ("KXNBAASTPROT", "nba", "ast"),
    ("KXNBAPRA", "nba", "pra"), ("KXNBA3PT", "nba", "fg3m"), ("KXNBA3PM", "nba", "fg3m"),
    ("KXNBABLK", "nba", "blk"), ("KXNBASTL", "nba", "stl"),
    ("KXNFLREC", "nfl", "rec"), ("KXNFLRECPROT", "nfl", "rec"),
    ("KXNFLRECYDS", "nfl", "rec_yds"), ("KXNFLRECYDSPROT", "nfl", "rec_yds"),
    ("KXNFLRSHYDS", "nfl", "rush_yds"), ("KXNFLRSHATT", "nfl", "rush_att"),
    ("KXNFLPASSYDS", "nfl", "pass_yds"), ("KXNFLRRYDS", "nfl", "rush_rec_yds"),
]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_events(payload: dict, series: str, league: str, stat: str) -> list[dict]:
    """Open rungs from one /events response: player, line, best bid and ask (dollars)."""
    out = []
    for e in payload.get("events") or []:
        for m in e.get("markets") or []:
            if (m.get("status") or "active") != "active":
                continue
            line, bid, ask = _num(m.get("floor_strike")), _num(m.get("yes_bid_dollars")), _num(m.get("yes_ask_dollars"))
            title, ticker = m.get("title") or "", m.get("ticker")
            if None in (line, bid, ask) or not ticker or not title:
                continue
            out.append({"ticker": ticker, "series": series, "league": league, "stat": stat,
                        "player": title.split(":")[0].strip(), "line": line, "bid": bid, "ask": ask})
    return out


def fetch(timeout: float = 12) -> list[dict] | None:
    markets, any_ok = [], False
    for series, league, stat in SERIES:
        q = urllib.parse.urlencode({"series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200})
        req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "patel-picks-board/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.load(r)
        except Exception as e:  # one series failing shouldn't drop the rest
            print(f"kalshi {series}: {e}", flush=True)
            continue
        any_ok = True
        markets += parse_events(payload, series, league, stat)
    return markets if any_ok else None


def write() -> int:
    markets = fetch()
    if markets is None:
        print("kalshi: no series answered; keeping the last kalshi.json", flush=True)
        return 0
    try:
        calib = json.loads((ROOT / "data" / "calibration.json").read_text()).get("props", {})
    except Exception:
        calib = {}
    doc = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "markets": markets, "calibration": calib}
    WEB.mkdir(exist_ok=True)
    tmp = WEB / "kalshi.json.tmp"
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    tmp.replace(WEB / "kalshi.json")
    print(f"kalshi: {len(markets)} rungs", flush=True)
    return len(markets)


if __name__ == "__main__":
    write()
