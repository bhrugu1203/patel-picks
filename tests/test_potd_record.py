"""Checks the play-of-the-day record. Run: python tests/test_potd_record.py"""
import json, sys, tempfile, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from edge import ingest, lock as LK
ingest.DB = pathlib.Path(tempfile.mkdtemp()) / "edge.db"

assert LK.potd_record() == {}  # empty db, no picks table yet

con = LK._con()
con.executescript("""CREATE TABLE IF NOT EXISTS picks(pick_id TEXT PRIMARY KEY, game_id TEXT, player_id TEXT,
  stat TEXT, actual REAL, result TEXT);""")
def lock(scope, day, gid, pid, side, line, price):
    pick = {"league": "nba", "game_id": gid, "player_id": pid, "player": pid.upper(), "stat": "pts", "label": "Points",
            "side": side, "line": line, "price": price}
    con.execute("INSERT INTO potd(scope, day, chosen, pick) VALUES(?,?,?,?)", (scope, day, day, json.dumps(pick)))
def box(gid, pid, actual, result="graded"):
    con.execute("INSERT INTO picks VALUES(?,?,?,?,?,?)", (gid + pid, gid, pid, "pts", actual, result))
lock("nba", "2026-11-10", "g1", "a", "over", 19.5, -110); box("g1", "a", 24)       # win at -110
lock("nba", "2026-11-11", "g2", "b", "under", 7.5, +120); box("g2", "b", 9)        # loss
lock("nba", "2026-11-12", "g3", "c", "over", 20.0, None); box("g3", "c", 20)       # push, no price
lock("nba", "2026-11-13", "g4", "d", "under", 5.5, None); box("g4", "d", 3)        # win, no price
lock("nba", "2026-11-14", "g5", "e", "over", 9.5, -110); box("g5", "e", None, "void")  # void
lock("nba", "2026-11-16", "g6", "f", "over", 9.5, -110)                             # not final yet
lock("all", "2026-11-10", "g1", "a", "over", 19.5, +150)                            # other scope, own price
con.commit(); con.close()

r = LK.potd_record()
n = r["nba"]
assert (n["wins"], n["losses"], n["pushes"]) == (2, 1, 1), n
assert n["priced"] == 2 and abs(n["units"] - (100 / 110 - 1)) < 0.01, n  # unpriced win counts in record only
assert [x["result"] for x in n["recent"]] == [None, "void", "win", "push", "loss", "win"], n["recent"]
assert r["all"]["wins"] == 1 and r["all"]["units"] == 1.5, r["all"]
assert len(LK.potd_record(recent=2)["nba"]["recent"]) == 2
print("potd record ok")
