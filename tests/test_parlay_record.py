"""Checks for the parlay record: book, closing price, and the by-type / by-book splits. Run: python tests/test_parlay_record.py"""
import sys, tempfile, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from edge import ingest, tracker as T
ingest.DB = pathlib.Path(tempfile.mkdtemp()) / "edge.db"

# closing price of a whole ticket
leg = lambda cp, line=19.5, cl=19.5, book="DK": {"book": book, "line": line, "close_line": cl, "close_price": cp}
assert T.parlay_close([leg(-110), leg(-110)], "DK") == 264, T.parlay_close([leg(-110), leg(-110)], "DK")
assert T.parlay_close([leg(-110), leg(None)], "DK") is None          # one leg has no close yet
assert T.parlay_close([leg(-110), leg(-110, cl=20.5)], "DK") is None  # line moved, prices aren't comparable
assert T.parlay_close([leg(-110), leg(-110, book="FD")], "DK") is None
assert T.parlay_close([leg(+150)], "DK") == 150 and T.parlay_close([leg(-200)], "DK") == -200

def mk(pid, gid, book="DK", price=-110):
    return {"league": "nba", "game_id": gid, "player_id": pid, "player": pid, "team": "A", "opp": "B", "stat": "pts",
            "label": "Points", "side": "over", "line": 19.5, "book": book, "price": price, "p": 0.6, "ev": 0.05, "conf": 3}

def priced(offers):  # offers: {(gid, pid): over price at DK}
    return [{"players": [{"game_id": g, "player_id": p, "stats": {"pts": {"offers": [{"book": "DK", "line": 19.5, "over": o}]}}}
                         for (g, p), o in offers.items()]}]

legs = [mk("a", "g1"), mk("b", "g2")]
boards = {"parlays": [{"name": "Value 2-leg", "legs": legs, "price": 264, "p": 0.36, "ev": 0.31, "book": "DK"}]}
T.record(boards, priced({("g1", "a"): -110, ("g2", "b"): -110}))
s = T.summary()
r = s["parlay_recent"][0]
assert r["book"] == "DK" and r["close_price"] == 264, r
# lines got shorter before kickoff: our +264 beat the close
T.record(boards, priced({("g1", "a"): -140, ("g2", "b"): -130}))
s = T.summary()
assert s["parlay_recent"][0]["close_price"] < 264, s["parlay_recent"][0]
assert s["parlays"]["clv_beat"] == 1.0 and s["parlays"]["clv_n"] == 1, s["parlays"]
assert s["parlays_by_name"]["Value 2-leg"]["n"] == 1 and s["parlays_by_book"]["DK"]["pending"] == 1, s
# an old table without the new columns still works
con = ingest.connect(); con.execute("DROP TABLE parlays")
con.execute("CREATE TABLE parlays(parlay_id TEXT PRIMARY KEY, posted TEXT, name TEXT, legs TEXT, price REAL, p REAL, ev REAL, result TEXT, profit REAL, graded TEXT)")
con.execute("INSERT INTO parlays VALUES('x','2026-11-16T00:00:00+00:00','Old',?,200,0.3,0.1,'win',2.0,'t')", (json.dumps(["g1|a|pts|over"]),))
con.commit(); con.close()
s = T.summary()
assert s["parlays"]["wins"] == 1 and s["parlays_by_book"] == {} and s["parlays_by_name"]["Old"]["wins"] == 1, s
print("parlay record: all checks pass")
