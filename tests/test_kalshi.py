"""Checks edge/kalshi.py's parsing on a saved-shape response (no network). Run: python tests/test_kalshi.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from edge import kalshi  # noqa: E402

payload = {"events": [{"markets": [
    {"status": "active", "ticker": "KXNBAPTS-1", "title": "Jalen Brunson: 25+ points", "floor_strike": 24.5,
     "yes_bid_dollars": "0.55", "yes_ask_dollars": "0.58"},
    {"status": "closed", "ticker": "KXNBAPTS-2", "title": "Old: 10+", "floor_strike": 9.5, "yes_bid_dollars": "0.1", "yes_ask_dollars": "0.2"},
    {"status": "active", "ticker": "KXNBAPTS-3", "title": "No price: 10+", "floor_strike": 9.5},
]}]}
rows = kalshi.parse_events(payload, "KXNBAPTS", "nba", "pts")
assert len(rows) == 1, rows
r = rows[0]
assert r["player"] == "Jalen Brunson" and r["line"] == 24.5 and r["bid"] == 0.55 and r["ask"] == 0.58, r
assert r["league"] == "nba" and r["stat"] == "pts" and r["ticker"] == "KXNBAPTS-1", r
assert kalshi.parse_events({}, "X", "nba", "pts") == []
print("test_kalshi: ok")
