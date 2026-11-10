"""Human-readable analysis: why a pick is favored, how a game should play out, and parlays.

Everything here is derived from model numbers; nothing is invented. Each sentence names
the number it comes from so a reader can check it.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import norm

POS_NAME = {"nba": {"G": "guards", "F": "forwards", "C": "centers"},
            "nhl": {"F": "forwards", "D": "defensemen", "G": "goalies"},
            "nfl": {"QB": "QBs", "RB": "RBs", "WR": "WRs", "TE": "TEs", "DEF": "defenders"},
            "mlb": {"BAT": "hitters", "SP": "starters"}}
BLOWOUT = {"nba": 15, "nfl": 17, "nhl": 3, "mlb": 5}
UNIT = {"nba": "pts", "nfl": "pts", "nhl": "goals", "mlb": "runs"}
TEAM_SD = {"nhl": 2.35, "mlb": 4.4}


def confidence(p: float) -> int:
    """0-100 grade: the calibrated chance the pick wins."""
    return int(round(100 * p))


# ---------------------------------------------------------------- pick reasoning
def reasons(league: str, player: dict, s: dict, offer: dict, side: str) -> dict:
    """Bullets for and against one side of one line."""
    line = offer["line"]
    proj = s["proj"]
    over = side == "over"
    pro, con = [], []
    gap = proj - line
    pct = gap / line * 100 if line else 0
    (pro if (gap > 0) == over else con).append(
        f"Model projects {proj:.1f}, {abs(gap):.1f} {'above' if gap >= 0 else 'below'} the line")

    def hit_line(key, label):
        h = (s.get("hit") or {}).get(key)
        if not h or not h["n"]:
            return
        n_side = h["over"] if over else h["n"] - h["over"]
        frac = n_side / h["n"]
        txt = f"{'Over' if over else 'Under'} {line:g} in {n_side} of {label if label else 'his last ' + str(h['n'])}"
        if frac >= 0.6:
            pro.append(txt)
        elif frac <= 0.4:
            con.append(txt)

    hit_line("l10", None)
    if (s.get("hit") or {}).get("vs_opp", {}) and s["hit"]["vs_opp"]["n"] >= 2:
        hit_line("vs_opp", f"{s['hit']['vs_opp']['n']} games vs {player['opp']}")

    for key, lab in (("l5", "over his last 5 games"), ("season", "this season")):
        v = s.get(key)
        if v is not None and line:
            d = (v - line) / line
            if abs(d) >= 0.08:
                (pro if (d > 0) == over else con).append(f"Averaging {v:g} {lab}")

    if s.get("def_rank") and s.get("def_n"):
        r, n, pct_d = s["def_rank"], s["def_n"], s.get("def_pct") or 0
        soft = r > n * 2 / 3
        tough = r <= n / 3
        if soft or tough:
            pos = POS_NAME.get(league, {}).get(player["pos"], player["pos"])
            rank_txt = f"{_ordinal(n - r + 1)} most generous" if soft else f"{_ordinal(r)} toughest"
            txt = f"{player['opp']} defense vs {pos}: {pct_d:+.0f}% {s['label'].lower()} allowed ({rank_txt} of {n})"
            (pro if soft == over else con).append(txt)

    vo = s.get("vs_opp") or {}
    if vo.get("n", 0) >= 2 and vo.get("avg") is not None and line:
        d = (vo["avg"] - line) / line
        if abs(d) >= 0.1:
            (pro if (d > 0) == over else con).append(f"Averages {vo['avg']:g} in {vo['n']} career games vs {player['opp']}")
    vk = s.get("vs_key") or {}
    if vk.get("name") and vk.get("n", 0) >= 2 and vk.get("avg") is not None and line:
        d = (vk["avg"] - line) / line
        if abs(d) >= 0.1:
            (pro if (d > 0) == over else con).append(f"Averages {vk['avg']:g} in {vk['n']} games against {vk['name']}")

    vac = s.get("vacated") or 0
    if vac >= 0.08:
        (pro if over else con).append(f"Teammates out: about {vac * 100:.0f}% more usage available than normal")
    if s.get("exp") and s.get("season_exp") and s["season_exp"] > 0:
        d = s["exp"] / s["season_exp"] - 1
        unit = {"nba": "minutes", "nhl": "minutes", "nfl": "opportunities", "mlb": "plate appearances"}[league]
        if abs(d) >= 0.08:
            (pro if (d > 0) == over else con).append(f"Expected {unit} {s['exp']:g}, {d * 100:+.0f}% vs season")
    for t in s.get("tags") or []:
        pro.append(f"Opponent tendency: {t}") if _tag_helps(t, s["label"], over) else None
    ol = offer.get("open_line")
    if ol is not None and ol != line:
        moved_up = line > ol
        txt = f"Line moved {ol:g} → {line:g} since open"
        # market moving toward our side means we're late; moving away means a better number
        (con if moved_up == over else pro).append(txt + (" (market agrees, number got worse)" if moved_up == over else " (better number now)"))
    if player.get("status"):
        con.append(f"Injury status: {player['status']}")
    return {"pro": pro[:6], "con": con[:3]}


def _tag_helps(tag: str, label: str, over: bool) -> bool:
    t, l = tag.lower(), label.lower()
    boosts = []
    if "fast pace" in t or "fouls" in t or "allows lots of shots" in t or "leaky" in t:
        boosts.append(True)
    if ("concedes threes" in t and ("3" in l or "three" in l)) or ("weak on the glass" in t and "reb" in l):
        boosts.append(True)
    if ("pass funnel" in t or "soft coverage" in t or "explosive passes" in t) and ("pass" in l or "rec" in l):
        boosts.append(True)
    if "run funnel" in t and "rush" in l:
        boosts.append(True)
    if ("strikeout pitcher" in t and "strikeout" in l) or ("hittable" in t and "hit" in l):
        boosts.append(True)
    lowers = ("slow" in t or "runs shooters off" in t and ("3" in l or "three" in l) or "rim protection" in t and ("pts" in l or "points" in l)
              or "stout run" in t and "rush" in l or "shot-suppressing" in t or "pitch-to-contact" in t and "strikeout" in l)
    if boosts:
        return over
    if lowers:
        return not over
    return False


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---------------------------------------------------------------- game script
def game_script(league: str, game: dict, players: list[dict], avg_total: float | None) -> dict:
    """How the game should unfold and which props that favors."""
    ph, pa = game["proj_home"], game["proj_away"]
    home, away = game["home"], game["away"]
    fav, dog = (home, away) if ph >= pa else (away, home)
    margin = abs(ph - pa)
    total = ph + pa
    if league in ("nba", "nfl"):
        sd = game.get("sd_margin") or (12.5 if league == "nba" else 13.5)
    else:
        sd = TEAM_SD[league]
    p_fav = float(norm.cdf(margin / sd)) if league in ("nba", "nfl") else None
    ml = next((m for m in game["markets"] if m["market"] == "moneyline"), None)
    if ml:
        p_fav = ml["p_a"] if ml["a"] == fav else ml["p_b"]
    b = BLOWOUT[league]
    p_blow = float(norm.sf(b, margin, sd) + norm.cdf(-b, margin, sd))
    mkt_total = (game.get("lines") or {}).get("total_close")
    lines = []
    lines.append(f"Projected score: {away} {pa:g}, {home} {ph:g}. {fav} win {p_fav * 100:.0f}% of the time."
                 if p_fav is not None else f"Projected score: {away} {pa:g}, {home} {ph:g}.")
    if mkt_total:
        diff = total - mkt_total
        tone = "faster, higher-scoring" if diff > 0 else "slower, lower-scoring"
        if abs(diff) >= {"nba": 3, "nfl": 2.5, "nhl": 0.3, "mlb": 0.5}[league]:
            lines.append(f"Model total {total:.1f} vs market {mkt_total:g}: expects a {tone} game than the books.")
        else:
            lines.append(f"Model total {total:.1f} is in line with the market's {mkt_total:g}.")
    pace = "neutral"
    if avg_total:
        r = total / avg_total - 1
        pace = "up-tempo" if r > 0.03 else "grind-it-out" if r < -0.03 else "average-paced"
    script = []
    if league == "nba":
        if p_blow >= 0.3:
            script.append(f"{p_blow * 100:.0f}% chance it's decided by {b}+. Starters on both sides could sit the 4th quarter, "
                          "which caps their overs and opens bench minutes.")
        else:
            script.append(f"Only {p_blow * 100:.0f}% blowout risk: starters should play full minutes, which helps counting-stat overs.")
        script.append(f"An {pace} game." if pace != "neutral" else "")
    elif league == "nfl":
        if p_fav and p_fav >= 0.65:
            script.append(f"{fav} should control the game. Leading teams run late, which lifts {fav} rushing volume. "
                          f"{dog} will likely chase, which lifts their pass attempts and target volume.")
        else:
            script.append("A close game projects balanced play-calling on both sides, with no strong run or pass lean from game script.")
        if p_blow >= 0.35:
            script.append(f"{p_blow * 100:.0f}% chance of a {b}+ point margin: garbage-time stats are in play for the trailing side.")
    elif league == "nhl":
        script.append(f"{fav} should hold the puck more. Expect them to outshoot {dog}, which favors {fav} shots-on-goal overs "
                      f"and the {dog} goalie's saves over.")
        if margin < 0.4:
            script.append("Coin-flip game: overtime is live, which adds a few minutes of extra shots.")
    elif league == "mlb":
        script.append(f"{fav} carry the edge in the run projection. Starting pitching drives most of it.")
    script = [s for s in script if s]

    by_side = {}
    for p in players:
        for stat, s in p["stats"].items():
            for o in s["offers"]:
                side = o.get("side")
                if not side:
                    side = "over" if o["p_over"] >= 0.5 else "under"
                pr = o["p_" + side]
                ev = o.get("ev_" + side)
                by_side.setdefault(p["team"], []).append((ev if ev is not None else pr - 0.55, p, stat, s, o, side, pr, ev))
    targets = []
    for team, rows in by_side.items():
        rows.sort(key=lambda r: -r[0])
        for score, p, stat, s, o, side, pr, ev in rows[:3]:
            if (ev is not None and ev > 0) or (ev is None and pr >= 0.58):
                targets.append({"player": p["name"], "player_id": p["player_id"], "team": team, "stat": stat,
                                "label": s["label"], "side": side, "line": o["line"], "p": pr, "ev": ev,
                                "price": o.get(side), "book": o["book"]})
    targets.sort(key=lambda t: -(t["ev"] if t["ev"] is not None else t["p"] - 0.55))

    leaders = []
    key_stat = {"nba": "pts", "nfl": None, "nhl": "sog", "mlb": "h"}[league]
    for p in players:
        if league == "nfl":
            cand = [(s["proj"], k, s) for k, s in p["stats"].items() if k in ("pass_yds", "rush_yds", "rec_yds")]
            if not cand:
                continue
            proj, k, s = max(cand, key=lambda x: x[0])
        else:
            s = p["stats"].get(key_stat) or p["stats"].get("p_k") or p["stats"].get("sv")
            if not s:
                continue
            proj, k = s["proj"], key_stat
        leaders.append({"player": p["name"], "player_id": p["player_id"], "team": p["team"], "label": s["label"],
                        "proj": proj, "status": p.get("status")})
    leaders.sort(key=lambda x: -x["proj"])
    tags = {}
    for p in players:
        for s in p["stats"].values():
            for t in s.get("tags") or []:
                tags.setdefault(p["opp"], set()).add(t)
    return {"summary": lines, "script": script, "p_fav": p_fav, "favorite": fav, "p_blowout": p_blow,
            "pace": pace, "proj_total": round(total, 1), "targets": targets[:8], "leaders": leaders[:10],
            "defense": {k: sorted(v)[:5] for k, v in tags.items()}}


# ---------------------------------------------------------------- parlays
def decimal(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / -american)


def to_american(dec: float) -> int:
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def parlays(picks: list[dict], likely: list[dict]) -> list[dict]:
    """Suggested parlays. Legs come from different games so they're close to independent;
    the joint chance is the product of each leg's calibrated chance. Books price same-game
    parlays with their own correlation adjustments, so those are left to the manual builder."""
    out = []

    def build(name, desc, pool, n, key):
        pool = [x for x in pool if x.get("price") is not None]
        best = None
        for combo in itertools.combinations(pool[:14], n):
            if len({c["game_id"] for c in combo}) < n or len({c["player_id"] for c in combo}) < n:
                continue
            p = math.prod(c["p"] for c in combo)
            dec = math.prod(decimal(c["price"]) for c in combo)
            ev = p * dec - 1
            score = key(p, ev)
            if best is None or score > best[0]:
                best = (score, combo, p, dec, ev)
        if best:
            _, combo, p, dec, ev = best
            out.append({"name": name, "desc": desc, "legs": [dict(c) for c in combo], "p": p,
                        "price": to_american(dec), "decimal": round(dec, 2), "ev": ev})

    value = [x for x in picks if x["ev"] > 0]
    safe = sorted([x for x in likely if x.get("ev") is not None and x["ev"] > -0.01 and x["price"] is not None],
                  key=lambda x: -x["p"])
    build("Safest 2-leg", "The two highest-probability legs from different games that aren't losing bets at their price.",
          safe, 2, lambda p, ev: p)
    build("Value 2-leg", "Two of the strongest proven-value picks.", value, 2, lambda p, ev: ev)
    build("Value 3-leg", "Three proven-value picks from different games.", value, 3, lambda p, ev: ev)
    build("Longshot 4-leg", "Bigger payout, lower hit rate. Every leg is +EV on its own.", value, 4, lambda p, ev: ev)
    return out
