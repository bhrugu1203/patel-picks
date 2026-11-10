"""Live slate: project every active player in upcoming games and price every line.

Two layers so the refresh loop stays cheap:
  project(league)  - heavy: loads history, builds features, fits, projects (minutes).
  price(state)     - light: re-reads current lines and injuries, recomputes chances and EV.
"""
from __future__ import annotations

import datetime as dt
import json
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import books, data as D, ingest, insights as I, players as P, pricing as X, teams as T
from .ingest import ROOT

warnings.filterwarnings("ignore")
ET = ZoneInfo("America/New_York")
LOOKAHEAD = {"nfl": 7, "nba": 2, "nhl": 2, "mlb": 2}
SKIP_STATUS = {"out", "injured reserve", "suspension", "doubtful", "ir", "inactive", "not with team"}
RECENT_GAMES = {"nfl": 4, "nba": 10, "nhl": 10, "mlb": 10}
MIN_APPEAR = {"nfl": 2, "nba": 4, "nhl": 4, "mlb": 4}
STYLE_TAGS = {
    "nba": [("s_pace", 1, "Fast pace"), ("s_pace", -1, "Slow, grinding pace"),
            ("s_three_rate", 1, "Concedes threes, packs the paint"), ("s_three_rate", -1, "Runs shooters off the line"),
            ("s_ft_rate", 1, "Fouls a lot"), ("s_blk_own", 1, "Elite rim protection"),
            ("s_stl_own", 1, "Aggressive, forces turnovers"), ("s_reb_allowed", 1, "Weak on the glass"),
            ("s_ast_rate", 1, "Gives up ball movement")],
    "nfl": [("s_run_rate", 1, "Run funnel: teams run on them"), ("s_run_rate", -1, "Pass funnel: teams throw on them"),
            ("s_yds_per_comp", -1, "Bend-don't-break, keeps the ball in front"),
            ("s_yds_per_comp", 1, "Gives up explosive passes"), ("s_cmp_rate", 1, "Soft coverage, high completion % allowed"),
            ("s_sack_rate", 1, "Pressure-heavy pass rush"), ("s_yds_per_carry", 1, "Leaky run defense"),
            ("s_yds_per_carry", -1, "Stout run defense"), ("s_pass_volume", 1, "Faces heavy passing volume")],
    "nhl": [("s_shots_allowed", 1, "Allows lots of shots"), ("s_shots_allowed", -1, "Shot-suppressing structure"),
            ("s_hits_own", 1, "Physical, high-hit team"), ("s_blocks_own", 1, "Collapses and blocks shots"),
            ("s_goals_allowed", 1, "Leaky defense and goaltending")],
    "mlb": [("s_sp_k", 1, "Strikeout pitcher"), ("s_sp_k", -1, "Pitch-to-contact"), ("s_sp_bb", 1, "Control issues"),
            ("s_sp_hr", 1, "Homer-prone"), ("s_sp_h", 1, "Hittable"), ("s_opp_k_rate", 1, "Lineup strikes out a lot"),
            ("s_opp_k_rate", -1, "Contact-heavy lineup"), ("s_opp_hr_rate", 1, "Power lineup")],
}


def _calibration() -> dict:
    f = ROOT / "data" / "calibration.json"
    return json.loads(f.read_text()) if f.exists() else {"games": {}, "props": {}}


def _now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_convert(None) if pd.Timestamp(s).tzinfo else pd.Timestamp(s)


def fetch_upcoming(league: str) -> list[dict]:
    today = _now_et().date()
    out = []
    for i in range(LOOKAHEAD[league]):
        out += ingest.fetch_slate(league, today + dt.timedelta(days=i))
    seen, uniq = set(), []
    for g in out:
        if g["game_id"] not in seen:
            seen.add(g["game_id"])
            uniq.append(g)
    return uniq


def build_future(league: str, hist: pd.DataFrame, players: pd.DataFrame, slate: list[dict], injuries: dict,
                 season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    roster = players.set_index("player_id")
    g_rows, p_rows = [], []
    last = hist.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
    rec_min = hist.sort_values("date").groupby("player_id").tail(10).groupby("player_id")[
        {"nba": "min", "nhl": "toi", "mlb": "pa", "nfl": "touch"}[league]].mean()
    for g in slate:
        date = _ts(g["date"])
        lines = g["lines"]
        row = {"game_id": g["game_id"], "league": league, "date": date, "season": season, "season_type": 2,
               "home_id": g["home_id"], "away_id": g["away_id"], "home_name": g["home_name"],
               "away_name": g["away_name"], "home_score": np.nan, "away_score": np.nan,
               "neutral": g["neutral"], "final": 0}
        # current lines go in the *_close fields so features use the live number
        for k, v in lines.items():
            if k.endswith("_close"):
                row[k] = v
        g_rows.append(row)
        for team, opp in ((g["home_id"], g["away_id"]), (g["away_id"], g["home_id"])):
            tg = hist[hist["team_id"] == team].drop_duplicates("game_id").sort_values("date")["game_id"].tail(RECENT_GAMES[league])
            recent = hist[hist["game_id"].isin(tg) & (hist["team_id"] == team)]
            cnt = recent.groupby("player_id").size()
            on_team = set(players.loc[players["team_id"] == team, "player_id"]) if len(players) else set()
            cands = set(cnt[cnt >= MIN_APPEAR[league]].index)
            if on_team:
                # roster is the source of truth (catches trades); recent usage filters out deep bench
                traded = {p for p in on_team if p in last.index and last.loc[p, "team_id"] != team
                          and (date - last.loc[p, "date"]).days <= 21}
                cands = (cands & on_team) | traded
            prob = g["probables"].get(team)
            for pid in cands:
                if pid not in last.index:
                    continue
                st = (injuries.get(pid) or {}).get("status") or ""
                lr = last.loc[pid]
                grp = lr["group"]
                is_out = st.lower() in SKIP_STATUS
                if is_out and (grp == "G" or league == "mlb"):
                    continue
                if league == "nhl" and grp == "G" and prob and prob["id"] != pid:
                    continue
                if league == "mlb" and pid in roster.index and str(roster.loc[pid, "pos"]) in ("P", "SP", "RP"):
                    continue
                p_rows.append({"game_id": g["game_id"], "player_id": pid, "team_id": team, "opp_id": opp,
                               "name": lr["name"], "pos": lr.get("pos"), "group": grp, "date": date, "season": season,
                               "home_id": g["home_id"], "away_id": g["away_id"], "is_home": float(team == g["home_id"]),
                               "_future": True, "_future_bat": league == "mlb", "status": st or None,
                               "_rec": float(rec_min.get(pid, 0) or 0), "_out": is_out})
            if league == "nhl" and not prob:
                gk = [r for r in p_rows if r["game_id"] == g["game_id"] and r["team_id"] == team and r["group"] == "G"]
                if len(gk) > 1:  # no probable announced: keep the goalie with the most recent usage
                    gk.sort(key=lambda r: -last.loc[r["player_id"], "date"].value)
                    for r in gk[1:]:
                        p_rows.remove(r)
            if league == "mlb" and prob:
                p_rows.append({"game_id": g["game_id"], "player_id": prob["id"], "team_id": team, "opp_id": opp,
                               "name": prob["name"], "pos": "SP", "group": "BAT", "date": date, "season": season,
                               "home_id": g["home_id"], "away_id": g["away_id"], "is_home": float(team == g["home_id"]),
                               "_future": True, "_future_sp": True, "status": None, "_rec": 0.0, "_out": False})
    fp = pd.DataFrame(p_rows)
    if fp.empty:
        return pd.DataFrame(g_rows), fp
    # key opposing player for the head-to-head feature
    if league == "nba":
        top = fp[~fp["_out"]].sort_values("_rec", ascending=False).drop_duplicates(["game_id", "team_id", "group"])
        top = top[["game_id", "team_id", "group", "player_id", "name"]].rename(
            columns={"team_id": "opp_id", "player_id": "key_id", "name": "key_name"})
        fp = fp.merge(top, on=["game_id", "opp_id", "group"], how="left")
    elif league == "nhl":
        gk = fp[fp["group"] == "G"][["game_id", "team_id", "player_id", "name"]].rename(
            columns={"team_id": "opp_id", "player_id": "key_id", "name": "key_name"}).drop_duplicates(["game_id", "opp_id"])
        fp = fp.merge(gk, on=["game_id", "opp_id"], how="left")
    elif league == "mlb":
        sp = fp[fp.get("_future_sp", False) == True][["game_id", "team_id", "player_id", "name"]].rename(  # noqa: E712
            columns={"team_id": "opp_id", "player_id": "key_id", "name": "key_name"})
        fp = fp.merge(sp, on=["game_id", "opp_id"], how="left")
        fp.loc[fp.get("_future_sp", False) == True, "_future_bat"] = False  # noqa: E712
    else:
        fp["key_id"], fp["key_name"] = None, None
    return pd.DataFrame(g_rows), fp.drop(columns="_rec")


def project(league: str) -> dict | None:
    """Heavy step: features, fit and projections for the upcoming slate."""
    slate = fetch_upcoming(league)
    if not slate:
        return {"league": league, "slate": [], "proj": pd.DataFrame(), "history": {}, "teams": pd.DataFrame()}
    games = D.load_games(league)
    roster = D.load_players(league)
    hist = P.add_keys(D.load_player_games(league, roster), league)
    injuries = ingest.fetch_injuries(league)
    season = int(games["season"].max())
    fg, fp = build_future(league, hist, roster, slate, injuries, season)
    if fp.empty:
        return {"league": league, "slate": slate, "proj": pd.DataFrame(), "history": {}, "teams": pd.DataFrame()}
    all_games = pd.concat([games, fg], ignore_index=True)
    all_pg = pd.concat([hist, fp], ignore_index=True)
    all_pg["_future"] = all_pg["_future"].fillna(False).astype(bool)
    ctx = P.prepare(all_pg, all_games, league)
    proj, history = [], {}
    for stat in D.STATS[league]:
        f = P.stat_frame(ctx, stat)
        f, _ = P.fit_live(f)
        current_teams = set(games.loc[games["season"] == season, "home_id"]) | set(games.loc[games["season"] == season, "away_id"])
        latest = f[f["y"].notna() & f["opp_id"].isin(current_teams)].sort_values("date").drop_duplicates(["opp_id", "group"], keep="last")
        ranks = latest.groupby("group")["opp_ratio"].rank(method="min")  # 1 = toughest
        n_teams = latest.groupby("group")["opp_ratio"].transform("count")
        f = f.merge(latest.assign(def_rank=ranks, def_n=n_teams)[["opp_id", "group", "def_rank", "def_n"]],
                    on=["opp_id", "group"], how="left")
        out_rows = f["_out"].fillna(False).astype(bool) if "_out" in f.columns else False
        fut = f[f["_future"].fillna(False).astype(bool) & f["mu"].notna() & ~out_rows].copy()
        if fut.empty:
            continue
        fut["stat"] = stat
        names = f.attrs.get("style_names", [])
        fut["tags"] = [_tags(league, r, names) for _, r in fut.iterrows()]
        cur = f[f["y"].notna() & (f["season"] == season)]
        fut["season_exp"] = fut["player_id"].map(cur.groupby("player_id")["expo"].mean())
        proj.append(fut[["game_id", "player_id", "name", "team_id", "opp_id", "group", "stat", "mu", "alpha", "exp_hat",
                         "l5", "l10", "season_avg", "h2h_n", "h2h_avg", "key_name", "key_h2h_n", "key_h2h_avg",
                         "opp_ratio", "def_rank", "def_n", "status", "tags", "vacated", "env_total",
                         "season_exp"]])
        played = f[f["y"].notna() & f["player_id"].isin(fut["player_id"])].sort_values("date")
        for pid, s in played.groupby("player_id"):
            history[(pid, stat)] = list(zip(s["date"].dt.strftime("%Y-%m-%d"), s["opp_id"], s["y"].astype(float)))[-40:]
    teams = T.run(all_games, hist, league, probables={(g["game_id"], t): p["id"] for g in slate for t, p in g["probables"].items()})
    teams = teams[teams["game_id"].isin({g["game_id"] for g in slate})]
    return {"league": league, "slate": slate, "proj": pd.concat(proj, ignore_index=True) if proj else pd.DataFrame(),
            "history": history, "teams": teams, "games": games, "roster": roster,
            "projected_at": _now_et().isoformat(timespec="seconds"), "injuries": injuries}


def _tags(league: str, row: pd.Series, names: list[str]) -> list[str]:
    out = []
    for col, sign, label in STYLE_TAGS[league]:
        if col in names and row.get(col, 0) * sign >= 0.8:
            out.append(label)
    return out


def _team_name(state: dict) -> dict:
    return {g[k]: g[k.replace("_id", "_name")] for g in state["slate"] for k in ("home_id", "away_id")}


def _beta(calib: dict, key: str, pooled: str) -> np.ndarray:
    """Calibration for this market. No fit -> trust the market. A negative weight on the
    model (its disagreements were wrong) is clipped to zero rather than inverted."""
    b = np.array(calib.get(key) or calib.get(pooled) or [0.0, 0.0], dtype=float)
    b[1] = max(b[1], 0.0)
    return b


def proven_thresholds() -> dict:
    """Per-league minimum EV whose backtest ROI (pre-game lines only) was clearly above zero."""
    f = ROOT / "data" / "backtest.json"
    out = {}
    if not f.exists():
        return out
    rep = json.loads(f.read_text())
    for lg, v in (rep.get("props") or {}).items():
        rows = v.get("bets_pregame_only") or []
        ok = [r["min_ev"] for r in rows if r.get("roi_lo") is not None and r["roi_lo"] > 0]
        out[lg] = min(ok) if ok else None
    return out


def _offer(p_model: float, beta, over_price, under_price) -> dict:
    """Chance of each side and expected value at the posted price."""
    has_price = over_price == over_price and under_price == under_price and over_price is not None and under_price is not None
    p_mkt = float(X.devig(over_price, under_price)) if has_price else 0.5
    p = float(X.apply_calibration(beta, np.array([p_model]), np.array([p_mkt]))[0])
    out = {"p_over": p, "p_under": 1 - p, "p_model_raw": float(p_model), "p_market": p_mkt if has_price else None,
           "fair_over": _fair(p), "fair_under": _fair(1 - p)}
    if has_price:
        out["ev_over"] = float(p * (X.payout(over_price) + 1) - 1)
        out["ev_under"] = float((1 - p) * (X.payout(under_price) + 1) - 1)
    return out


def _fair(p: float) -> int:
    p = min(max(p, 0.01), 0.99)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def edge_score(ev: float | None, p: float, flags: int = 0) -> float:
    """0-10 score: 15% expected value scores 10; questionable status or thin history halves it."""
    if ev is None:
        return 0.0
    s = max(0.0, min(10.0, ev * 100 / 1.5))
    return round(s * (0.5 if flags else 1.0), 1)


def price(state: dict, refresh_lines: bool = True, book_rows: pd.DataFrame | None = None) -> dict:
    """Light step: current lines in, chances and EV out."""
    league = state["league"]
    if refresh_lines and state["slate"]:
        fresh = {g["game_id"]: g for g in fetch_upcoming(league)}
        state["slate"] = [fresh.get(g["game_id"], g) for g in state["slate"] if g["game_id"] in fresh]
        state["injuries"] = ingest.fetch_injuries(league)
    calib = _calibration()
    names = _team_name(state)
    proj = state["proj"]
    out = {"league": league, "projected_at": state.get("projected_at"), "games": [], "players": []}
    if not state["slate"]:
        return out
    started = {g["game_id"] for g in state["slate"]}
    proj = proj[proj["game_id"].isin(started)] if len(proj) else proj
    inj = state.get("injuries") or {}
    tname = {}
    if state.get("games") is not None and len(state["games"]):
        gm = state["games"]
        tname = {**dict(zip(gm["home_id"], gm["home_name"])), **dict(zip(gm["away_id"], gm["away_name"]))}
    tname.update(names)
    out["teams"] = tname
    out["team_meta"] = {k: v for g in state["slate"] for k, v in (g.get("team_meta") or {}).items()}

    # ---- offers: ESPN (DraftKings) lines + any sportsbook prices from The Odds API
    offers: dict[tuple, list] = {}
    for g in state["slate"]:
        rows = pd.DataFrame(g["props"], columns=["player_id", "market", "seq", "line_open", "line_close",
                                                 "price_open", "price_close", "updated"])
        if rows.empty:
            continue
        rows["stat"] = [D.market_stat(league, m) for m in rows["market"]]
        rows = rows.dropna(subset=["stat", "line_close"])
        for (pid, stat), s in rows.groupby(["player_id", "stat"]):
            o, u = s[s["seq"] == 0], s[s["seq"] == 1]
            if o.empty:
                continue
            offers.setdefault((g["game_id"], pid, stat), []).append({
                "book": "DraftKings", "line": float(o["line_close"].iloc[0]),
                "open_line": float(o["line_open"].iloc[0]) if o["line_open"].notna().iloc[0] else None,
                "over": o["price_close"].iloc[0] if o["price_close"].notna().iloc[0] else None,
                "under": u["price_close"].iloc[0] if len(u) and u["price_close"].notna().iloc[0] else None})
    if book_rows is not None and len(book_rows):
        roster = pd.concat([state["roster"][["player_id", "name", "team_id"]],
                            proj[["player_id", "name", "team_id"]].drop_duplicates()])
        b = books.attach(book_rows, roster, state["slate"])
        for (gid, pid, stat, book, line), s in b.groupby(["game_id", "player_id", "stat", "book", "line"]):
            ov, un = s[s["side"] == "over"]["price"], s[s["side"] == "under"]["price"]
            entry = {"book": book.replace("williamhill_us", "caesars"), "line": float(line),
                     "over": float(ov.iloc[0]) if len(ov) else None, "under": float(un.iloc[0]) if len(un) else None}
            lst = offers.setdefault((gid, pid, stat), [])
            if book == "draftkings":  # real DK price replaces the ESPN copy when it's missing a price
                lst[:] = [x for x in lst if not (x["book"] == "DraftKings" and x["over"] is None)]
            lst.append(entry)

    # ---- players
    hist = state["history"]
    for (gid, pid), rows in proj.groupby(["game_id", "player_id"], sort=False):
        first = rows.iloc[0]
        status = (inj.get(pid) or {}).get("status")
        if status and status.lower() in SKIP_STATUS:
            continue
        flags = int(bool(status))
        player = {"game_id": gid, "player_id": pid, "name": first["name"], "team": names.get(first["team_id"], ""),
                  "team_id": first["team_id"], "opp_id": first["opp_id"],
                  "opp": names.get(first["opp_id"], ""), "pos": first["group"], "status": status,
                  "note": (inj.get(pid) or {}).get("note"), "stats": {}}
        for r in rows.itertuples():
            vals = [v for _, _, v in hist.get((pid, r.stat), [])]
            line_list = offers.get((gid, pid, r.stat), [])
            beta = _beta(calib["props"], f"{league}|{r.stat}", f"{league}|_pooled")
            priced = []
            for off in line_list:
                pm = float(P.p_over(r.mu, r.alpha, off["line"]))
                res = _offer(pm, beta, off["over"] if off["over"] is not None else np.nan,
                             off["under"] if off["under"] is not None else np.nan)
                best_side = None
                if "ev_over" in res:
                    best_side = "over" if res["ev_over"] >= res["ev_under"] else "under"
                    res["score"] = edge_score(max(res["ev_over"], res["ev_under"]), res["p_" + best_side], flags)
                res.update(off, side=best_side)
                priced.append(res)
            main_line = priced[0]["line"] if priced else None
            stat_out = {
                "label": D.STAT_LABELS.get(r.stat, r.stat), "proj": round(float(r.mu), 2),
                "median": float(P.p_over(r.mu, r.alpha, np.arange(0, 400) + 0.5).__lt__(0.5).argmax()) if r.mu < 300 else round(float(r.mu)),
                "exp": None if r.exp_hat != r.exp_hat else round(float(r.exp_hat), 1),
                "l5": _r(r.l5), "l10": _r(r.l10), "season": _r(r.season_avg),
                "vs_opp": {"n": _i(r.h2h_n), "avg": _r(r.h2h_avg)},
                "vs_key": {"name": r.key_name if isinstance(r.key_name, str) else None, "n": _i(r.key_h2h_n),
                           "avg": _r(r.key_h2h_avg)},
                "def_rank": None if r.def_rank != r.def_rank else int(r.def_rank),
                "def_n": None if r.def_n != r.def_n else int(r.def_n),
                "def_pct": None if r.opp_ratio != r.opp_ratio else round((float(r.opp_ratio) - 1) * 100, 1),
                "tags": r.tags, "offers": priced,
                "log": [(d, tname.get(o, ""), v) for d, o, v in hist.get((pid, r.stat), [])[-20:]],
                "vacated": round(float(r.vacated), 3) if r.vacated == r.vacated else 0.0,
                "season_exp": _r(r.season_exp),
            }
            if main_line is not None and vals:
                stat_out["hit"] = {k: _hit(vals[-n:], main_line) for k, n in (("l5", 5), ("l10", 10), ("l20", 20))}
                stat_out["hit"]["all"] = _hit(vals, main_line)
                opp_vals = [v for _, o, v in hist.get((pid, r.stat), []) if o == first["opp_id"]]
                stat_out["hit"]["vs_opp"] = _hit(opp_vals, main_line)
            player["stats"][r.stat] = stat_out
        out["players"].append(player)

    # ---- games
    tm = state["teams"].set_index("game_id") if len(state["teams"]) else pd.DataFrame()
    for g in state["slate"]:
        if g["game_id"] not in tm.index:
            continue
        t = tm.loc[g["game_id"]]
        L = g["lines"]
        pr = T.probabilities(league, t.lam_h, t.lam_a, t.sd_margin, t.sd_total,
                             _n(L.get("spread_home_close")), _n(L.get("total_close")))
        markets = []
        for mkt, key, a, b, la, lb in (
            ("moneyline", "ml_home", "ml_home_close", "ml_away_close", g["home_name"], g["away_name"]),
            ("spread", "spread_home", "spread_home_price_close", "spread_away_price_close",
             f"{g['home_name']} {_fmt_line(L.get('spread_home_close'))}",
             f"{g['away_name']} {_fmt_line(-L['spread_home_close'] if L.get('spread_home_close') is not None else None)}"),
            ("total", "over", "over_price_close", "under_price_close", f"Over {L.get('total_close')}", f"Under {L.get('total_close')}"),
        ):
            if key not in pr or L.get(a) is None or L.get(b) is None:
                continue
            beta = _beta(calib["games"], f"{league}|{mkt}", "__none__")
            res = _offer(pr[key], beta, L[a], L[b])
            side_a = res["ev_over"] >= res["ev_under"]
            ev = res["ev_over"] if side_a else res["ev_under"]
            markets.append({"market": mkt, "a": la, "b": lb, "price_a": L[a], "price_b": L[b],
                            "p_a": res["p_over"], "p_b": res["p_under"], "p_model_a": pr[key],
                            "pick": la if side_a else lb, "ev": ev,
                            "score": edge_score(ev, res["p_over"] if side_a else res["p_under"])})
        out["games"].append({
            "game_id": g["game_id"], "date": g["date"], "home": g["home_name"], "away": g["away_name"],
            "home_id": g["home_id"], "away_id": g["away_id"],
            "proj_home": round(float(t.lam_h), 1), "proj_away": round(float(t.lam_a), 1),
            "sd_margin": float(t.sd_margin),
            "lines": {k: L.get(k) for k in ("spread_home_close", "total_close", "ml_home_close", "ml_away_close",
                                            "spread_home_open", "total_open")},
            "markets": markets, "probables": g["probables"], "h2h": _team_h2h(state.get("games"), g),
        })
    # ---- game scripts
    avg_total = None
    if state.get("games") is not None and len(state["games"]):
        recent = state["games"].tail(400)
        avg_total = float((recent["home_score"] + recent["away_score"]).mean())
    by_game = {}
    for p in out["players"]:
        by_game.setdefault(p["game_id"], []).append(p)
    for g in out["games"]:
        g["script"] = I.game_script(league, g, by_game.get(g["game_id"], []), avg_total)
    return out


def _team_h2h(games: pd.DataFrame | None, g: dict) -> list[dict]:
    if games is None or games.empty:
        return []
    m = games[((games["home_id"] == g["home_id"]) & (games["away_id"] == g["away_id"])) |
              ((games["home_id"] == g["away_id"]) & (games["away_id"] == g["home_id"]))].sort_values("date").tail(6)
    out = []
    for r in m.itertuples():
        sp, tot = getattr(r, "spread_home_close", None), getattr(r, "total_close", None)
        margin = r.home_score - r.away_score
        out.append({"date": str(r.date)[:10], "home": r.home_name, "away": r.away_name,
                    "score": f"{int(r.away_score)}-{int(r.home_score)}",
                    "ats": None if sp is None or sp != sp else ("home covered" if margin + sp > 0 else "push" if margin + sp == 0 else "away covered"),
                    "ou": None if tot is None or tot != tot else ("over" if r.home_score + r.away_score > tot else "push" if r.home_score + r.away_score == tot else "under")})
    return out[::-1]


def _hit(vals, line):
    vals = [v for v in vals if v == v]
    if not vals:
        return None
    return {"over": sum(v > line for v in vals), "n": len(vals)}


def _i(x):
    return 0 if x is None or x != x else int(x)


def _r(x):
    return None if x is None or x != x else round(float(x), 1)


def _n(x):
    return np.nan if x is None else x


def _fmt_line(x):
    if x is None:
        return ""
    return f"+{x:g}" if x > 0 else f"{x:g}"


def boards(priced: list[dict], top: int = 25) -> dict:
    """Bet sheet (value), most-likely list, and game edges across leagues.

    The bet sheet only includes bets at or above the EV level that made money in that
    league's backtest; everything else stays visible in the value list, marked unproven.
    """
    proven = proven_thresholds()
    value, likely, leans = [], [], []
    pmap = {(p["player_id"], p["game_id"]): (lg["league"], p) for lg in priced for p in lg["players"]}

    def explain(x):
        lgname, p = pmap[(x["player_id"], x["game_id"])]
        s = p["stats"][x["stat"]]
        off = next((o for o in s["offers"] if o["book"] == x["book"] and o["line"] == x["line"]), s["offers"][0])
        x["why"] = I.reasons(lgname, p, s, off, x["side"])
        x["conf"] = I.confidence(x["p"])
        h = (s.get("hit") or {})
        x["hit_l10"] = h.get("l10")
        x["hit_opp"] = h.get("vs_opp")
        return x
    for lg in priced:
        for p in lg["players"]:
            for stat, s in p["stats"].items():
                for o in s["offers"]:
                    base = {"league": lg["league"], "player": p["name"], "team": p["team"], "opp": p["opp"],
                            "game_id": p["game_id"], "player_id": p["player_id"], "stat": stat, "label": s["label"],
                            "proj": s["proj"], "line": o["line"], "book": o["book"], "status": p["status"],
                            "team_id": p["team_id"], "opp_id": p["opp_id"], "pos": p["pos"]}
                    if o.get("side"):
                        side = o["side"]
                        evv = o["ev_" + side]
                        need = proven.get(lg["league"])
                        ok = need is not None and evv >= need
                        value.append({**base, "side": side, "price": o[side], "p": o["p_" + side],
                                      "ev": evv, "fair": o["fair_" + side], "score": o["score"] if ok else 0.0,
                                      "proven": ok, "min_ev": need})
                    elif o.get("over") is None and o.get("under") is None and o.get("p_over") is not None:
                        side = "over" if o["p_over"] >= 0.5 else "under"
                        leans.append({**base, "side": side, "price": None, "p": o["p_" + side], "ev": None,
                                      "fair": o.get("fair_" + side), "edge": abs(s["proj"] - o["line"]) / max(1.0, o["line"])})
                    for side in ("over", "under"):
                        price = o.get(side)
                        likely.append({**base, "side": side, "p": o["p_" + side], "price": price,
                                       "ev": o.get("ev_" + side), "fair": o["fair_" + side]})
    value = [v for v in value if v["ev"] > 0]
    value.sort(key=lambda v: -v["ev"])
    sheet, per_player, per_game = [], set(), {}
    for v in value:  # one bet per player and at most two per game, to limit correlated exposure
        if not v["proven"]:
            continue
        if v["player_id"] in per_player or per_game.get(v["game_id"], 0) >= 2:
            continue
        per_player.add(v["player_id"])
        per_game[v["game_id"]] = per_game.get(v["game_id"], 0) + 1
        sheet.append(v)
    likely = [x for x in likely if x["ev"] is None or x["ev"] > -0.03]
    likely.sort(key=lambda x: -x["p"])
    seen, lk = set(), []
    for x in likely:
        k = (x["player_id"], x["stat"], x["side"])
        if k not in seen:
            seen.add(k)
            lk.append(x)
    games = [dict(league=lg["league"], home=g["home"], away=g["away"], **m)
             for lg in priced for g in lg["games"] for m in g["markets"] if m["ev"] > 0]
    games.sort(key=lambda m: -m["ev"])
    # best plays in each direction: priced, at least a coin flip, and not giving away much value
    def side_list(side):
        c = [x for x in likely if x["side"] == side and x["price"] is not None and x["p"] >= 0.5
             and x["ev"] is not None and x["ev"] > -0.02]
        c.sort(key=lambda x: -(x["ev"] + 0.5 * (x["p"] - 0.5)))
        seen_, out_ = set(), []
        for x in c:
            if x["player_id"] not in seen_:
                seen_.add(x["player_id"])
                out_.append(dict(x))
        return out_[:40]
    overs, unders = side_list("over"), side_list("under")
    leans.sort(key=lambda x: -x["p"])
    seen, ln = set(), []
    for x in leans:  # one lean per player
        if x["player_id"] not in seen:
            seen.add(x["player_id"])
            ln.append(x)
    sheet, value, lk, ln = sheet[:top], value[:150], lk[:150], ln[:60]
    for x in sheet + value + lk + ln + overs + unders:
        explain(x)
    return {"sheet": sheet, "value": value, "likely": lk, "leans": ln, "overs": overs, "unders": unders, "game_edges": games[:40],
            "proven": proven, "parlays": I.parlays(sheet, lk)}
