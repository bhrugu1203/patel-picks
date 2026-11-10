"""One full update cycle for the hosted site (GitHub Actions runs this on a schedule).

    python -m edge.ci

Downloads new final scores, grades finished picks, rebuilds season stats, then projects
and prices every upcoming game once and writes web/slate.json + web/stats.json.
"""
from __future__ import annotations

import datetime as dt
import os

from . import ingest, live, stats, tracker

LEAGUES = ["nfl", "nba", "nhl", "mlb"]


def main():
    today = dt.date.today()
    try:
        ingest.backfill(LEAGUES, today - dt.timedelta(days=3), today)
        tracker.grade()
    except Exception as e:  # a bad scoreboard response should not stop the board
        print("results update failed:", e, flush=True)
    stats.write()
    every = os.environ.get("REFRESH_MINUTES", "30")
    live.main(["--once", "--every", every, "--leagues", ",".join(LEAGUES)])


if __name__ == "__main__":
    main()
