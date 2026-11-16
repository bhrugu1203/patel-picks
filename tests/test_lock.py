"""Checks for the locked play of the day and the bad-fetch guard. Run: python tests/test_lock.py"""
import datetime as dt, sys, tempfile, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from edge import ingest, lock as LK
ingest.DB = pathlib.Path(tempfile.mkdtemp()) / "edge.db"
UTC = dt.timezone.utc

def mk_player(pid, gid, stat="pts", offers=True):
    return {"player_id": pid, "game_id": gid, "name": pid, "stats": {stat: {"proj": 20, "season": 20, "l10": 20,
            "offers": [{"book": "DraftKings", "line": 19.5}] if offers else []}}}

def mk_pick(pid, gid, start, lg="nba", ev=0.05, p=0.6, side="over"):
    return {"league": lg, "game_id": gid, "player_id": pid, "player": pid, "team": "A", "opp": "B", "team_id": "1",
            "opp_id": "2", "stat": "pts", "label": "Points", "side": side, "line": 19.5, "book": "DraftKings",
            "price": -110, "p": p, "ev": ev, "proj": 21, "start": start, "game_day": LK.game_day(start)}

def priced_for(lg, games, players, ruled_out=()):
    return {"league": lg, "games": [{"game_id": g} for g in games], "players": players,
            "game_dates": games, "ruled_out": list(ruled_out)}

# Monday Nov 16 2026, noon ET = 17:00 UTC
now = dt.datetime(2026, 11, 16, 17, 0, tzinfo=UTC)
sun = "2026-11-16T23:00Z"; sun_late = "2026-11-17T01:15Z"   # 8:15pm ET Monday -> still Monday
mon = "2026-11-18T01:15Z"  # Tuesday 8:15pm ET
assert LK.game_day(sun_late) == "2026-11-16", LK.game_day(sun_late)
assert LK.game_day(mon) == "2026-11-17"

games = {"g1": sun, "g2": sun_late, "g3": mon}
players = [mk_player("a", "g1"), mk_player("b", "g2"), mk_player("c", "g3")]
pr = [priced_for("nba", games, players)]
boards = {"sheet": [], "overs": [mk_pick("c", "g3", mon, ev=0.20), mk_pick("a", "g1", sun, ev=0.05), mk_pick("b", "g2", sun_late, ev=0.10)],
          "unders": [], "leans": []}
pd_ = LK.pick_of_day(boards, pr, now)
assert pd_["nba"]["player"] == "b" and pd_["nba"]["game_day"] == "2026-11-16", pd_["nba"]  # best on the nearest day, not Monday's
first_lock = pd_["nba"]["locked_at"]

# 30 min later the model likes someone else more: pick must stay
later = now + dt.timedelta(minutes=30)
boards2 = {**boards, "overs": [mk_pick("a", "g1", sun, ev=0.30)] + boards["overs"]}
pd2 = LK.pick_of_day(boards2, pr, later)
assert pd2["nba"]["player"] == "b" and pd2["nba"]["locked_at"] == first_lock, pd2["nba"]

# prop pulled once -> still kept; twice -> replaced
pr_missing = [priced_for("nba", games, [mk_player("a", "g1"), mk_player("b", "g2", offers=False), mk_player("c", "g3")])]
assert LK.pick_of_day(boards2, pr_missing, later + dt.timedelta(minutes=30))["nba"]["player"] == "b"
pd3 = LK.pick_of_day(boards2, pr_missing, later + dt.timedelta(minutes=60))
assert pd3["nba"]["player"] == "a" and pd3["nba"]["replaced"], pd3["nba"]

# ruled out -> replaced immediately
pr_out = [priced_for("nba", games, players, ruled_out=["a"])]
pd4 = LK.pick_of_day(boards2, pr_out, later + dt.timedelta(minutes=90))
assert pd4["nba"]["player"] == "b" and "ruled out" in pd4["nba"]["replaced"], pd4["nba"]

# after the game starts, the pick is final even if it vanishes from the board
after = dt.datetime(2026, 11, 17, 2, 0, tzinfo=UTC)  # 9pm ET Monday, g2 started
pr_after = [priced_for("nba", {"g3": mon}, [mk_player("c", "g3")])]
pd5 = LK.pick_of_day({"sheet": [], "overs": [mk_pick("c", "g3", mon)], "unders": [], "leans": []}, pr_after, after)
assert pd5["nba"]["player"] == "b" and pd5["nba"]["started"], pd5["nba"]

# Tuesday (next ET day): moves on to the next game day
mon_morning = dt.datetime(2026, 11, 17, 14, 0, tzinfo=UTC)
pd6 = LK.pick_of_day({"sheet": [], "overs": [mk_pick("c", "g3", mon)], "unders": [], "leans": []}, pr_after, mon_morning)
assert pd6["nba"]["player"] == "c" and pd6["nba"]["game_day"] == "2026-11-17"

# NFL: no game today, next one in two days -> that game is the pick, and it stays locked until then
tue = "2026-11-18T23:00Z"
nfl_pr = [priced_for("nfl", {"n1": tue}, [mk_player("q", "n1")])]
nb = {"sheet": [], "overs": [mk_pick("q", "n1", tue, lg="nfl")], "unders": [], "leans": []}
p7 = LK.pick_of_day(nb, nfl_pr, mon_morning)
assert p7["nfl"]["game_day"] == "2026-11-18" and p7["all"]
p8 = LK.pick_of_day(nb, nfl_pr, dt.datetime(2026, 11, 18, 14, 0, tzinfo=UTC))
assert p8["nfl"]["locked_at"] == p7["nfl"]["locked_at"]

# ---- guard
t0 = dt.datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
gd = {"x%d" % i: "2026-11-21T00:00Z" for i in range(3)}
good = priced_for("nhl", gd, [mk_player("p%d" % i, "x%d" % (i % 3)) for i in range(30)])
out, notes = LK.guard([good], ["nhl"], t0)
assert out[0] is good and not notes
# league failed entirely -> saved board comes back
out, notes = LK.guard([], ["nhl"], t0 + dt.timedelta(minutes=30))
assert len(out) == 1 and len(out[0]["players"]) == 30 and out[0]["stale"] and "nhl" in notes
# came back with no games
out, notes = LK.guard([priced_for("nhl", {}, [])], ["nhl"], t0 + dt.timedelta(minutes=30))
assert len(out[0]["players"]) == 30 and "no games" in notes["nhl"]
# came back with most players missing
thin = priced_for("nhl", gd, [mk_player("p0", "x0")])
out, notes = LK.guard([thin], ["nhl"], t0 + dt.timedelta(minutes=30))
assert len(out[0]["players"]) == 30 and "players" in notes["nhl"]
# normal shrink because games started is fine
after_start = dt.datetime(2026, 11, 21, 1, 0, tzinfo=UTC)
out, notes = LK.guard([priced_for("nhl", {}, [])], ["nhl"], after_start)
assert out[0]["players"] == [] and not notes
# saved board too old is never used
out, notes = LK.guard([], ["nhl"], t0 + dt.timedelta(hours=13))
assert out == [] and not notes
print("all lock tests passed")
