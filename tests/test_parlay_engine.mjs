// Checks for web/parlay/engine.js (the Parlays web app's math). Run: node tests/test_parlay_engine.mjs
import * as E from "../web/parlay/engine.js";
let pass = 0, fail = 0;
const check = (ok, what) => (ok ? pass++ : (fail++, console.log("FAIL:", what)));
const near = (a, b, tol, what) => check(Math.abs(a - b) <= tol, `${what}: got ${a}, want ${b} ± ${tol}`);

near(E.normCDF(0), 0.5, 1e-7, "normCDF(0)");
near(E.normCDF(1.959963985), 0.975, 1e-6, "normCDF(1.96)");
for (const p of [0.01, 0.2, 0.5, 0.64, 0.97]) near(E.normCDF(E.normInv(p)), p, 1e-6, `normInv round trip ${p}`);
for (const r of [-0.5, 0, 0.3, 0.6]) near(E.bivNormCDF(0, 0, r), 0.25 + Math.asin(r) / (2 * Math.PI), 1e-5, `bivNormCDF(0,0,${r})`);
near(E.bivNormCDF(E.normInv(0.64), E.normInv(0.58), 0.35), 0.4240, 0.002, "bivNormCDF vs Monte Carlo");

const leg = (playerId, gameId, side, p) => ({ league: "nba", gameId, teamId: "t1", playerId, stat: "pts", side, p });
const a = leg("a", "g1", "over", 0.64), b = leg("b", "g1", "over", 0.58), bU = leg("b", "g1", "under", 0.58);
near(E.jointHit([a, b], () => 0), 0.64 * 0.58, 1e-12, "independent legs multiply");
near(E.jointHit([a, b], () => 0.35), 0.4240, 0.002, "two-leg joint = bivariate normal");
check(E.jointHit([a, bU], () => 0.35) < 0.64 * 0.58, "over + under on linked stats clash");
near(E.parlayEV(0.42, 2.358), -0.00964, 1e-5, "parlay EV");

check(E.formatOdds(2.5, "american") === "+150" && E.formatOdds(1 + 100 / 120, "american") === "−120", "american");
check(E.formatOdds(2.5, "decimal") === "2.50" && E.formatOdds(23.58, "decimal") === "23.6", "decimal");
check(E.formatOdds(2.5, "fractional") === "3/2" && E.formatOdds(1 + 100 / 110, "fractional") === "10/11", "fractional");
check(E.formatOdds(6.5, "fractional") === "11/2" && E.formatOdds(41, "fractional") === "40/1", "fractional long");
check(E.formatOdds(2.5, "implied") === "40%" && E.formatOdds(2.5, "implied", true) === "40¢", "implied / cents");
check(E.formatOdds(1, "american") === "—", "no price");

near(E.nbOver(10, 0.1, 9.5), 1 - (() => { let r = 10, p = r / (r + 10), lp = r * Math.log(p), c = Math.exp(lp); for (let k = 1; k <= 9; k++) { lp += Math.log((k - 1 + r) / k) + Math.log(1 - p); c += Math.exp(lp); } return c; })(), 1e-9, "nbOver");
for (const [mu, line, a] of [[20, 24.5, 0.3], [20, 24.5, 0.05], [20, 15.5, 0.1], [6, 8.5, 0.2]]) {
  const t = E.nbOver(mu, a, line);
  near(E.nbOver(mu, E.fitAlpha(mu, line, t), line), t, 1e-4, `fitAlpha reproduces the chance (mu ${mu}, line ${line}, alpha ${a})`);
}

// tier building on a synthetic pool: picks distinct players and games, respects the leg count
const now = new Date("2026-11-18T12:00:00");
const g = (h) => new Date(`2026-11-18T${h}:00:00`);
const pool = ["p1", "p2", "p3", "p4", "p5", "p6"].map((pid, i) => ({ id: pid, league: "nba", playerId: pid, gameId: `g${i % 3}`,
  teamId: `t${i}`, stat: "pts", side: "over", p: 0.7 - i * 0.02, price: -120, ev: 0.7 * decimalOf(-120) - 1, gameDate: g(19 + (i % 3)) }));
function decimalOf(a) { return E.decimalOdds(a); }
const set = E.buildSet(pool, "today", "mix", now, E.TIERS, null);
check(set[0].length === 2 && new Set(set[0].map((l) => l.playerId)).size === 2, "SAFE has 2 different players");
check(set[1].length === 3 && new Set(set[1].map((l) => l.gameId)).size === 3, "SOLID spreads 3 games");
near(E.lockTime(pool, "mix", now).getHours(), 11, 0, "locks at 11 AM when games are in the evening");

// track record
const slate = { graded: {}, logs: { "x|pts": { "2026-11-17": 25 } } };
const saved = { legs: [{ player: "X", playerId: "x", stat: "pts", label: "Points", side: "over", line: 20.5, p: 0.6, gameDay: "2026-11-17", price: -110 }],
  predicted: 0.6, decimal: E.decimalOdds(-110) };
check(E.gradeSaved(saved, slate, now) && saved.outcome === "won", "graded from the game log");
const st = E.recordStats([saved, { outcome: "lost", predicted: 0.5, decimal: 2, payout: 0 }]);
check(st.won === 1 && st.decided === 2 && Math.abs(st.profit - (10 * E.decimalOdds(-110) - 20)) < 1e-9, "record stats");

console.log(`${pass} passed, ${fail} failed`);
if (typeof process !== "undefined") process.exit(fail ? 1 : 0);
export const result = { pass, fail };
