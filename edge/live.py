"""Keep the board current: re-price every few minutes and serve the dashboard.

    python -m edge.live                      # all leagues, open http://localhost:8765
    python -m edge.live --every 3 --books-every 20 --leagues nba,nfl

Cadence (all adjustable):
  --every N           lines, prices and injuries from ESPN/DraftKings (free)      default 5 min
  --project-every N   full re-projection; also triggered by injury changes       default 30 min
  --books-every N     DraftKings/FanDuel/BetMGM/Caesars via The Odds API (quota)  default 30 min
  new final scores are downloaded hourly so the model keeps learning.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import http.server
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import books, ingest, slate as S, stats as ST, tracker
from .ingest import ROOT

WEB = ROOT / "web"


def serve(port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"dashboard: http://localhost:{port}", flush=True)


def write_json(obj: dict):
    WEB.mkdir(exist_ok=True)
    tmp = WEB / "slate.json.tmp"
    tmp.write_text(json.dumps(obj, default=_default))
    tmp.replace(WEB / "slate.json")


def _default(x):
    if isinstance(x, (pd.Timestamp, dt.datetime, dt.date)):
        return x.isoformat()
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def _out_set(inj: dict) -> set:
    return {pid for pid, v in (inj or {}).items() if (v.get("status") or "").lower() in S.SKIP_STATUS}


def _safe(fn):
    try:
        return fn()
    except Exception as e:
        return {"error": str(e)}


def backtest_summary() -> dict:
    f = ROOT / "data" / "backtest.json"
    if not f.exists():
        return {}
    rep = json.loads(f.read_text())
    return {"games": rep.get("games", {}), "props": rep.get("props", {}), "matchups": rep.get("matchups", {}),
            "updated": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default="nfl,nba,nhl,mlb")
    ap.add_argument("--every", type=float, default=5)
    ap.add_argument("--project-every", type=float, default=30)
    ap.add_argument("--books-every", type=float, default=30)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--once", action="store_true", help="one full cycle then exit")
    a = ap.parse_args(argv)
    leagues = a.leagues.split(",")
    if not a.once:
        serve(a.port)

    states: dict[str, dict] = {}
    projected_at: dict[str, float] = {}
    pending: dict = {}
    book_cache: dict[str, pd.DataFrame] = {}
    status: dict[str, str] = {}
    last_books = last_results = 0.0
    pool = ThreadPoolExecutor(1)  # projections run in the background so pricing never stalls

    def start_projection(lg):
        if lg in pending and not pending[lg].done():
            return
        status[lg] = "projecting..."
        pending[lg] = pool.submit(S.project, lg)

    for lg in leagues:
        start_projection(lg)
    first_done: set = set()
    try:
        ST.write()
    except Exception as e:
        print("stats build failed:", e, flush=True)

    while True:
        cycle = time.time()
        # collect finished projections
        for lg, fut in list(pending.items()):
            if fut.done():
                try:
                    states[lg] = fut.result()
                    projected_at[lg] = time.time()
                    status[lg] = f"projected {len(states[lg]['proj'])} player-stats for {len(states[lg]['slate'])} games"
                except Exception:
                    status[lg] = "projection failed: " + traceback.format_exc(limit=1).strip().splitlines()[-1]
                    projected_at[lg] = time.time()
                first_done.add(lg)
                del pending[lg]
        if (a.once or (WEB / "slate.json").exists()) and not first_done.issuperset(leagues):
            # wait for every league's first projection so no sport drops off the board at startup
            time.sleep(2)
            continue

        # sportsbook prices (quota-limited)
        if books.api_key() and time.time() - last_books >= a.books_every * 60:
            for lg in leagues:
                if lg in states and states[lg]["slate"]:
                    try:
                        hours = 24 * S.LOOKAHEAD[lg] + 12
                        book_cache[lg], msg = books.fetch(lg, hours_ahead=hours)
                    except Exception as e:  # keep the loop alive on network errors
                        msg = f"error {e}"
                    status[lg + "_books"] = msg
            last_books = time.time()
        elif not books.api_key():
            status["books"] = "Add ODDS_API_KEY to .env for FanDuel/DraftKings/BetMGM/Caesars prices"

        # re-price with fresh lines and injuries
        priced = []
        for lg in leagues:
            st = states.get(lg)
            if not st:
                continue
            try:
                before = _out_set(st.get("injuries"))
                out = S.price(st, refresh_lines=True, book_rows=book_cache.get(lg))
                if _out_set(st.get("injuries")) != before:
                    status[lg] = "injury change: re-projecting"
                    start_projection(lg)
                elif time.time() - projected_at.get(lg, 0) >= a.project_every * 60:
                    start_projection(lg)
                priced.append(out)
            except Exception:
                status[lg] = "pricing failed: " + traceback.format_exc(limit=1).strip().splitlines()[-1]

        now = dt.datetime.now(S.ET)
        if not priced and (WEB / "slate.json").exists():
            # still projecting at startup: keep showing the last board rather than an empty one
            time.sleep(15)
            continue
        boards = S.boards(priced)
        try:
            tracker.record(boards, priced)
        except Exception as e:
            print("tracker record failed:", e, flush=True)
        write_json({
            "generated": now.isoformat(timespec="seconds"),
            "next_refresh": (now + dt.timedelta(minutes=a.every)).isoformat(timespec="seconds"),
            "refresh_minutes": a.every, "status": status, "leagues": priced,
            "boards": boards, "backtest": backtest_summary(), "results": _safe(tracker.summary),
            "books_enabled": bool(books.api_key()),
        })
        print(f"{now:%H:%M:%S} priced {sum(len(p['players']) for p in priced)} players, "
              f"{sum(len(p['games']) for p in priced)} games | {status}", flush=True)
        if a.once:
            break

        # hourly: pull new final scores so tomorrow's projections include tonight's games
        if time.time() - last_results >= 3600:
            try:
                today = dt.date.today()
                ingest.backfill(leagues, today - dt.timedelta(days=3), today)
                tracker.grade()
                ST.write()
            except Exception as e:
                print("results update failed:", e, flush=True)
            last_results = time.time()

        time.sleep(max(5, a.every * 60 - (time.time() - cycle)))


if __name__ == "__main__":
    main()
