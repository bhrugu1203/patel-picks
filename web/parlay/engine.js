// Parlay engine for the Parlays web app. A straight port of the macOS parlay popup's math
// (bhrugu-collab/parlay-popup: main.swift, Parlay.swift, Record.swift, Odds.swift) so the phone
// app builds the same SAFE / SOLID / BOLD parlays. No DOM here: tests run it in Node.

// ---------------------------------------------------------------- tiers
export const MIN_EV = -0.04;
export const POOL = 18;
export const LOCK_HOUR = 11;

const tier = (name, legs, minP, valueWeight, blurb, o = {}) =>
  ({ name, legs, minP, valueWeight, blurb, floorP: 0.56, maxCost: 1, minEV: MIN_EV, minParlayEV: 0, ...o });

export const TIERS = [
  tier("SAFE", 2, 0.64, 0, "Two of the surest props on the board."),
  tier("SOLID", 3, 0.60, 0.5, "Three strong props, leaning toward good prices."),
  tier("BOLD", 4, 0.56, 1, "Bigger payout. Every leg still has the model on its side."),
];
export const KTIERS = [
  tier("SAFE", 2, 0.64, 0, "Two of the surest props on Kalshi.", { floorP: 0.56, maxCost: 0.86 }),
  tier("SOLID", 3, 0.55, 0.7, "Three props at 72¢ or less, priced at or below the model's fair value.",
    { floorP: 0.50, maxCost: 0.72, minEV: -0.01 }),
  tier("BOLD", 4, 0.45, 1, "Four props at 62¢ or less, each priced below what the model thinks it's worth.",
    { floorP: 0.40, maxCost: 0.62, minEV: 0 }),
];

// ---------------------------------------------------------------- small helpers
export const decimalOdds = (a) => (a > 0 ? 1 + a / 100 : 1 + 100 / -a);
export const fmtNum = (x) => (Math.abs(x - Math.round(x)) < 1e-9 ? String(Math.round(x)) : String(Math.round(x * 10) / 10));
const num = (v) => (v == null || v === "" || Number.isNaN(Number(v)) ? null : Number(v));
const pad = (n) => String(n).padStart(2, "0");
/** yyyy-mm-dd in the phone's own time zone */
export const dayKey = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
export const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x; };
export const normName = (s) => s.toLowerCase().split(/[^a-z]+/).filter((w) => w && !["jr", "sr", "ii", "iii", "iv", "v"].includes(w)).join("");

// ---------------------------------------------------------------- odds formats
export const ODDS_FORMATS = ["american", "decimal", "fractional", "implied"];
export const ODDS_LABEL = { american: "American", decimal: "Decimal", fractional: "Fractional", implied: "Implied %" };
const LADDER = [[1, 20], [1, 16], [1, 12], [1, 10], [1, 8], [1, 7], [1, 6], [1, 5], [2, 9], [1, 4], [2, 7], [3, 10], [1, 3],
  [4, 11], [2, 5], [4, 9], [1, 2], [8, 15], [4, 7], [8, 13], [2, 3], [8, 11], [4, 5], [5, 6], [10, 11], [1, 1],
  [21, 20], [11, 10], [6, 5], [5, 4], [11, 8], [3, 2], [13, 8], [7, 4], [15, 8], [2, 1], [9, 4], [5, 2], [11, 4],
  [3, 1], [10, 3], [7, 2], [4, 1], [9, 2], [5, 1], [11, 2], [6, 1], [13, 2], [7, 1], [15, 2], [8, 1], [9, 1],
  [10, 1], [11, 1], [12, 1], [14, 1], [16, 1], [18, 1], [20, 1], [22, 1], [25, 1], [28, 1], [33, 1], [40, 1],
  [50, 1], [66, 1], [80, 1], [100, 1]];

/** decimal odds in the chosen format; on Kalshi "implied" reads as cents, like Kalshi itself */
export function formatOdds(dec, f, kalshi = false) {
  if (!Number.isFinite(dec) || dec <= 1.0001) return "—";
  if (f === "decimal") return dec < 10 ? dec.toFixed(2) : dec < 100 ? dec.toFixed(1) : dec.toFixed(0);
  if (f === "fractional") {
    const x = dec - 1;
    if (x > 100) return `${Math.round(x)}/1`;
    const t = Math.log(1 + x);
    let best = LADDER[0];
    for (const fr of LADDER) if (Math.abs(Math.log(1 + fr[0] / fr[1]) - t) < Math.abs(Math.log(1 + best[0] / best[1]) - t)) best = fr;
    return `${best[0]}/${best[1]}`;
  }
  if (f === "implied") {
    const p = 100 / dec;
    return kalshi ? `${Math.round(p)}¢` : p < 1 ? "<1%" : `${Math.round(p)}%`;
  }
  const a = dec >= 2 ? Math.round((dec - 1) * 100) : Math.round(-100 / (dec - 1));
  return a > 0 ? `+${a}` : `−${-a}`;
}

// ---------------------------------------------------------------- normal distribution
const erfc = (x) => { // Numerical Recipes erfc, |error| < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
};
export const normCDF = (x) => 0.5 * erfc(-x / Math.SQRT2);
const horner = (c, x) => c.reduce((a, k) => a * x + k, 0);
const iA = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
const iB = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01, 1.0];
const iC = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
const iD = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00, 1.0];
export function normInv(p) {
  const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12), lo = 0.02425;
  if (q < lo) { const r = Math.sqrt(-2 * Math.log(q)); return horner(iC, r) / horner(iD, r); }
  if (q > 1 - lo) { const r = Math.sqrt(-2 * Math.log(1 - q)); return -horner(iC, r) / horner(iD, r); }
  const r = q - 0.5;
  return (horner(iA, r * r) * r) / horner(iB, r * r);
}
export function bivNormCDF(h, k, rho) {
  const r = Math.min(Math.max(rho, -0.95), 0.95), base = normCDF(h) * normCDF(k);
  if (Math.abs(r) < 1e-9) return base;
  const n = 64, step = r / n;
  let s = 0;
  for (let i = 0; i <= n; i++) {
    const t = i * step, w = i === 0 || i === n ? 1 : i % 2 ? 4 : 2, om = 1 - t * t;
    s += (w * Math.exp(-(h * h - 2 * t * h * k + k * k) / (2 * om))) / Math.sqrt(om);
  }
  return Math.min(Math.max(base + (s * step) / 3 / (2 * Math.PI), 0), Math.min(normCDF(h), normCDF(k)));
}

// ---------------------------------------------------------------- correlation + joint chance
export class CorrModel {
  constructor(logs, players) {
    this.logs = logs; this.pooled = new Map(); this.pairs = new Map(); this.byGame = new Map();
    const seen = new Set();
    for (const p of players) {
      const k = `${p.playerId}|${p.gameId}`;
      if (seen.has(k)) continue;
      seen.add(k);
      if (!this.byGame.has(p.gameId)) this.byGame.set(p.gameId, []);
      this.byGame.get(p.gameId).push(p);
    }
  }
  static pearson(a, b) {
    const xs = [], ys = [];
    for (const d in a) if (d in b) { xs.push(a[d]); ys.push(b[d]); }
    const n = xs.length;
    if (n < 4) return null;
    const mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
    if (sxx <= 1e-9 || syy <= 1e-9) return null;
    return { r: sxy / Math.sqrt(sxx * syy), n };
  }
  pooledRho(league, sameTeam, s1, s2) {
    const key = `${league}|${sameTeam ? "team" : "opp"}|${s1 < s2 ? s1 : s2}|${s1 < s2 ? s2 : s1}`;
    if (this.pooled.has(key)) return this.pooled.get(key);
    let sum = 0, n = 0;
    for (const ps of this.byGame.values()) for (const a of ps) {
      if (a.league !== league) continue;
      const la = this.logs[`${a.playerId}|${s1}`];
      if (!la) continue;
      for (const b of ps) {
        if (b.playerId === a.playerId || (b.teamId === a.teamId) !== sameTeam) continue;
        const lb = this.logs[`${b.playerId}|${s2}`];
        const r = lb && CorrModel.pearson(la, lb);
        if (r) { sum += r.r * r.n; n += r.n; }
      }
    }
    const v = sum / (n + 60);
    this.pooled.set(key, v);
    return v;
  }
  rho(a, b) {
    if (a.gameId !== b.gameId || a.playerId === b.playerId) return 0;
    const k1 = `${a.playerId}|${a.stat}`, k2 = `${b.playerId}|${b.stat}`, key = k1 < k2 ? `${k1}~${k2}` : `${k2}~${k1}`;
    if (this.pairs.has(key)) return this.pairs.get(key);
    const prior = this.pooledRho(a.league, a.teamId === b.teamId, a.stat, b.stat);
    let v = prior;
    const la = this.logs[k1], lb = this.logs[k2], r = la && lb && CorrModel.pearson(la, lb);
    if (r) v = (r.r * r.n + prior * 20) / (r.n + 20);
    v = Math.min(Math.max(v, -0.6), 0.6);
    this.pairs.set(key, v);
    return v;
  }
}

/** chance every leg hits; same-game pairs linked with a Gaussian copula */
export function jointHit(legs, rho) {
  let joint = legs.reduce((s, l) => s * l.p, 1);
  for (let i = 0; i < legs.length - 1; i++) for (let j = i + 1; j < legs.length; j++) {
    const a = legs[i], b = legs[j], r = rho(a, b) * ((a.side === "over") === (b.side === "over") ? 1 : -1);
    if (Math.abs(r) <= 1e-6) continue;
    joint *= bivNormCDF(normInv(a.p), normInv(b.p), r) / Math.max(a.p * b.p, 1e-9);
  }
  return Math.min(Math.max(joint, 0), Math.min(...legs.map((l) => l.p), 1));
}
export const jointP = (legs, corr) => (corr ? jointHit(legs, (a, b) => corr.rho(a, b)) : legs.reduce((s, l) => s * l.p, 1));
export const independentP = (legs) => legs.reduce((s, l) => s * l.p, 1);
export const parlayEV = (joint, decimal) => joint * decimal - 1;

/** what the combo pays per $1 (stake included) when every leg has a price */
export function comboDecimal(legs) {
  if (!legs.length) return null;
  if (legs.every((l) => l.cost != null)) return 1 / legs.reduce((s, l) => s * l.cost, 1);
  if (legs.every((l) => l.price != null)) return legs.reduce((s, l) => s * decimalOdds(l.price), 1);
  return null;
}
/** a book or Kalshi prices helpful same-game links in, so value counts a link only when it hurts */
export const valueJoint = (legs, corr) => Math.min(jointP(legs, corr), independentP(legs));
export function comboEV(legs, corr) { const d = comboDecimal(legs); return d == null ? null : parlayEV(valueJoint(legs, corr), d); }

// ---------------------------------------------------------------- reading the slate
const fadesTheAverages = (proj, season, recent, over) => {
  if (season == null || recent.length < 5 || proj <= 0) return false;
  const l10 = recent.reduce((s, v) => s + v, 0) / recent.length;
  return over ? season < proj * 0.75 && l10 < proj * 0.75 : season > proj * 1.25 && l10 > proj * 1.25;
};
export function nbOver(mu, alpha, line) {
  if (mu <= 0) return line < 0 ? 1 : 0;
  const r = 1 / Math.max(alpha, 1e-4), p = r / (r + mu), n = Math.floor(line);
  if (n < 0) return 1;
  let lp = r * Math.log(p), cdf = Math.exp(lp);
  for (let k = 1; k <= n; k++) { lp += Math.log((k - 1 + r) / k) + Math.log(1 - p); cdf += Math.exp(lp); }
  return Math.min(Math.max(1 - cdf, 1e-4), 1 - 1e-4);
}
/** Overdispersion that makes the distribution hit the model's own chance at a known line.
 *  For lines above the mean the chance rises then falls as alpha grows, so a plain bisection over the
 *  whole range can miss; scan for the first crossing (smallest alpha) and bisect inside it. */
export function fitAlpha(mu, line, pOver) {
  if (mu <= 0 || pOver <= 0.01 || pOver >= 0.99) return null;
  const f = (la) => nbOver(mu, Math.exp(la), line) - pOver, lo0 = Math.log(0.002), hi0 = Math.log(3), steps = 48;
  let prev = lo0, fp = f(prev), best = [Math.abs(fp), prev];
  for (let i = 1; i <= steps; i++) {
    const x = lo0 + ((hi0 - lo0) * i) / steps, fx = f(x);
    if (Math.abs(fx) < best[0]) best = [Math.abs(fx), x];
    if (fp === 0) return Math.exp(prev);
    if (fp * fx < 0) {
      let lo = prev, hi = x;
      for (let k = 0; k < 40; k++) { const mid = (lo + hi) / 2; if ((f(mid) > 0) === (fp > 0)) lo = mid; else hi = mid; }
      return Math.exp((lo + hi) / 2);
    }
    prev = x; fp = fx;
  }
  return Math.exp(best[1]);   // no crossing: the closest the distribution gets
}
const HURT = ["out", "doubtful", "questionable", "suspended", "injured"];

/** every sportsbook leg, player projections (for Kalshi), logs, team colours and graded results */
export function readSlate(d) {
  const why = {}, legs = [], teams = {}, logs = {}, pstats = {}, graded = {};
  const B = d.boards || {};
  for (const k of ["sheet", "value", "overs", "unders", "likely", "leans"]) for (const x of B[k] || []) {
    const pro = x.why?.pro;
    if (pro?.length) why[`${x.player_id}|${x.game_id}|${x.stat}|${x.side}`] = pro;
  }
  for (const L of d.leagues || []) {
    const lg = L.league, gdate = {};
    for (const g of L.games || []) gdate[g.game_id] = g.date ? new Date(g.date) : null;
    for (const [id, m] of Object.entries(L.team_meta || {})) teams[`${lg}|${id}`] = { abbr: m.abbr || "", color: m.color || null };
    for (const pl of L.players || []) {
      const pid = pl.player_id, gid = pl.game_id;
      if (!pid || !gid) continue;
      const status = (pl.status || "").toLowerCase(), hurt = HURT.some((h) => status.includes(h));
      const base = { league: lg, player: pl.name || "", playerId: pid, team: pl.team || "", teamId: pl.team_id || "",
        opp: pl.opp || "", oppId: pl.opp_id || "", gameId: gid, pos: pl.pos || "", gameDate: gdate[gid] || null };
      for (const [stat, sv] of Object.entries(pl.stats || {})) {
        const log = sv.log || [], lm = {};
        for (const e of log) if (e.length >= 3 && num(e[2]) != null) lm[e[0]] = num(e[2]);
        logs[`${pid}|${stat}`] = lm;
        const recent = log.slice(-10).map((e) => num(e[2])).filter((v) => v != null);
        const proj = num(sv.proj), label = sv.label || stat, season = num(sv.season);
        if (proj != null) {
          let fitted = null;
          for (const o of sv.offers || []) {
            const l = num(o.line), pr = num(o.p_model_raw);
            if (l != null && pr != null && Math.abs(pr - 0.5) >= 0.1) { const a = fitAlpha(proj, l, pr); if (a != null) { fitted = a; break; } }
          }
          pstats[`${normName(pl.name || "")}|${stat}`] = { ...base, stat, label, proj, alpha: fitted, hurt, recent, l5: num(sv.l5), season };
        }
        if (hurt) continue;
        for (const o of sv.offers || []) {
          const line = num(o.line);
          if (line == null) continue;
          for (const side of ["over", "under"]) {
            const p = num(o[`p_${side}`]);
            if (p == null) continue;
            const price = num(o[side]), ev = price == null ? null : p * decimalOdds(price) - 1, pj = proj ?? line;
            if (fadesTheAverages(pj, season, recent, side === "over")) continue;
            const id = `${pid}|${gid}|${stat}|${side}`;
            let pros = why[id] || [];
            if (!pros.length) {
              pros = [`Model projects ${fmtNum(pj)}, ${fmtNum(Math.abs(pj - line))} ${pj > line ? "above" : "below"} the line`];
              if (recent.length) {
                const hits = recent.filter((v) => (side === "over" ? v > line : v < line)).length;
                pros.push(`${side === "over" ? "Over" : "Under"} ${fmtNum(line)} in ${hits} of his last ${recent.length}`);
              }
              if (num(sv.l5) != null) pros.push(`Averaging ${fmtNum(num(sv.l5))} over his last 5 games`);
            }
            legs.push({ ...base, id, stat, label, side, book: o.book || "", line, proj: pj, p, price, ev, pros, venue: "books" });
          }
        }
      }
    }
  }
  // typical spread per stat, from how recent games scatter around the projection
  const nu = {}, de = {}, pooled = {};
  for (const v of Object.values(pstats)) {
    if (v.recent.length < 5 || v.proj <= 0.3) continue;
    const k = `${v.league}|${v.stat}`;
    for (const y of v.recent) { nu[k] = (nu[k] || 0) + (y - v.proj) ** 2 - v.proj; de[k] = (de[k] || 0) + v.proj * v.proj; }
  }
  for (const k in nu) if (de[k] > 0) pooled[k] = Math.min(Math.max(nu[k] / de[k], 0.01), 2);
  for (const v of Object.values(pstats)) {
    const b = pooled[`${v.league}|${v.stat}`];
    if (v.alpha != null && b != null) v.alpha = Math.min(Math.max(v.alpha, b * 0.5), b * 2);
    else if (v.alpha == null) v.alpha = b ?? null;
  }
  for (const r of d.results?.recent || []) {
    if (!r.result || !r.player || num(r.line) == null) continue;
    graded[`${r.player}|${r.label || ""}|${r.side || ""}|${num(r.line)}`] = r.result;
  }
  const corr = new CorrModel(logs, Object.values(pstats).map((v) => ({ league: v.league, playerId: v.playerId, teamId: v.teamId, gameId: v.gameId })));
  return { generated: d.generated ? new Date(d.generated) : new Date(), legs, teams, logs, pstats, graded, corr };
}

// ---------------------------------------------------------------- Kalshi
const kalshiFee = (px) => 0.07 * px * (1 - px);
const logit = (p) => { const q = Math.min(Math.max(p, 1e-4), 1 - 1e-4); return Math.log(q / (1 - q)); };
const expit = (x) => 1 / (1 + Math.exp(-x));

/** Kalshi rungs (from web/kalshi.json) turned into legs with the model's calibrated chance */
export function kalshiLegs(slate, markets, calib) {
  const best = new Map();
  for (const m of markets) {
    const ps = slate.pstats[`${normName(m.player)}|${m.stat}`];
    if (!ps || ps.hurt || ps.league !== m.league || ps.alpha == null || m.bid < 0.02 || m.ask > 0.98 || m.ask - m.bid > 0.10) continue;
    const mid = (m.bid + m.ask) / 2;
    if (mid < 0.15 || mid > 0.85) continue;
    const pModel = Math.min(Math.max(nbOver(ps.proj, ps.alpha, m.line), 0.05), 0.95);
    const beta = calib[`${m.league}|${m.stat}`] || calib[`${m.league}|_pooled`] || [0, 0.3];
    const pYes = expit(beta[0] + beta[1] * (logit(pModel) - logit(mid)) + logit(mid)), rung = Math.round(m.line + 0.5);
    for (const side of ["over", "under"]) {
      const p = side === "over" ? pYes : 1 - pYes, px = side === "over" ? m.ask : 1 - m.bid, cost = px + kalshiFee(px);
      if (px < 0.08 || px > 0.86 || fadesTheAverages(ps.proj, ps.season, ps.recent, side === "over")) continue;
      const pros = [`Model: ${Math.round(p * 100)}% vs Kalshi ${Math.round(px * 100)}¢ (${side === "over" ? "YES" : "NO"})`,
        `Projects ${fmtNum(Math.round(ps.proj * 10) / 10)} ${ps.label.toLowerCase()}`];
      if (ps.recent.length) {
        const hits = ps.recent.filter((v) => (side === "over" ? v > m.line : v < m.line)).length;
        pros.push(`${side === "over" ? `${rung}+` : `Under ${rung}`} in ${hits} of his last ${ps.recent.length}`);
      }
      const id = `k|${ps.playerId}|${ps.gameId}|${m.stat}|${side}|${rung}`;
      const leg = { ...ps, id, side, book: "Kalshi", line: m.line, p, price: null, ev: p / cost - 1, pros, venue: "kalshi", cost, ticker: m.ticker };
      const old = best.get(id);
      if (!old || cost < old.cost) best.set(id, leg);
    }
  }
  return [...best.values()];
}

// ---------------------------------------------------------------- building parlays
const boost = (l) => (l.ev == null ? 1 : Math.max(0.5, 1 + l.ev));

function buildTierAt(spec, minP, pool, avoid, corr, gate) {
  const ok = pool.filter((l) => l.p >= minP && (l.ev == null || l.ev >= spec.minEV) && (l.cost == null || l.cost <= spec.maxCost));
  const score1 = (l) => l.p * boost(l) ** spec.valueWeight;
  const seen = new Set();
  const cand = ok.sort((a, b) => score1(b) - score1(a))
    .filter((l) => { const k = `${l.playerId}|${l.stat}`; if (seen.has(k)) return false; seen.add(k); return true; })
    .slice(0, POOL);
  const games = Math.max(1, new Set(cand.map((l) => l.gameId)).size);
  let perGame = games >= spec.legs ? 1 : Math.ceil(spec.legs / games);
  const avoidIds = avoid.map((a) => new Set(a.map((l) => l.id)));

  const search = (strict) => {
    let best = null;
    const go = (start, chosen) => {
      if (chosen.length === spec.legs) {
        if (avoidIds.some((s) => s.size === chosen.length && chosen.every((c) => s.has(c.id)))) return;
        if (strict && avoid.some((a) => chosen.filter((c) => a.some((x) => x.playerId === c.playerId)).length > 1)) return;
        const joint = valueJoint(chosen, corr), dec = comboDecimal(chosen);
        if (gate && dec != null && parlayEV(joint, dec) < spec.minParlayEV) return;
        let score = joint;
        for (const c of chosen) score *= boost(c) ** spec.valueWeight;
        score *= 0.97 ** (chosen.length - new Set(chosen.map((c) => c.gameId)).size);
        if (!best || score > best[0]) best = [score, chosen];
        return;
      }
      for (let i = start; i < cand.length; i++) {
        const c = cand[i];
        if (chosen.some((x) => x.playerId === c.playerId)) continue;
        if (chosen.filter((x) => x.gameId === c.gameId).length >= perGame) continue;
        go(i + 1, [...chosen, c]);
      }
    };
    go(0, []);
    return best ? best[1] : [];
  };
  while (perGame <= spec.legs) {
    const s = search(true);
    if (s.length) return s;
    const loose = search(false);
    if (loose.length) return loose;
    perGame++;
  }
  return [];
}

export function buildTier(spec, pool, avoid, corr) {
  for (const gate of [true, false]) {
    const lowest = gate ? Math.max(spec.floorP, spec.minP - 0.04) : spec.floorP;
    for (let minP = spec.minP; minP >= lowest - 1e-9; minP -= 0.02) {
      const r = buildTierAt(spec, minP, pool, avoid, corr, gate);
      if (r.length) return r;
    }
  }
  return [];
}

/** legs for one day (games at least 10 minutes away), mixed or NBA only */
export function poolFor(legs, day, sport, now) {
  const target = dayKey(day === "today" ? now : addDays(now, 1)), soon = now.getTime() + 10 * 60000;
  return legs.filter((l) => l.gameDate && dayKey(l.gameDate) === target && l.gameDate.getTime() > soon && (sport === "mix" || l.league === "nba"));
}

export function buildSet(legs, day, sport, now, tiers, corr) {
  const p = poolFor(legs, day, sport, now), out = [];
  for (const t of tiers) out.push(buildTier(t, p, out.filter((s) => s.length), corr));
  return out;
}

/** today's parlays lock at 11 AM or two hours before the first game, whichever is first */
export function lockTime(allLegs, sport, now) {
  const first = poolFor(allLegs, "today", sport, now).reduce((m, l) => (!m || l.gameDate < m ? l.gameDate : m), null);
  const eleven = new Date(now); eleven.setHours(LOCK_HOUR, 0, 0, 0);
  return first && first.getTime() - 2 * 3600000 < eleven.getTime() ? new Date(first.getTime() - 2 * 3600000) : eleven;
}

export const placeBy = (legs) => {
  const t = legs.map((l) => l.gameDate).filter(Boolean).sort((a, b) => a - b)[0];
  return t ? new Date(t.getTime() - 10 * 60000) : null;
};

// ---------------------------------------------------------------- leg status + track record
/** Box scores in the slate are filed by the game's UTC date, so an 8 PM ET tip-off on Nov 17 is logged
 *  as Nov 18. Look a leg up by the UTC date of its own start time, never the local date, or tonight's
 *  leg picks up last night's game. */
export const logDay = (gameDate) => new Date(gameDate).toISOString().slice(0, 10);
const started = (l, now) => !l.gameDate || new Date(l.gameDate) <= now;

export function legState(l, slate, now) {
  const gd = l.gameDate && new Date(l.gameDate);
  if (gd && gd > now) return "pending"; // nothing can be settled before the game starts
  const g = slate.graded[`${l.player}|${l.label}|${l.side}|${l.line}`];
  if (g) return g === "win" ? "won" : g === "loss" ? "lost" : "push";
  const v = gd && slate.logs[`${l.playerId}|${l.stat}`]?.[logDay(gd)];
  if (v != null) return v === l.line ? "push" : (l.side === "over" ? v > l.line : v < l.line) ? "won" : "lost";
  if (gd) return now - gd < 4 * 3600000 ? "live" : "final";
  return "pending";
}
export function parlayState(legs, slate, now) {
  const st = legs.map((l) => legState(l, slate, now));
  if (st.includes("lost")) return "lost";
  if (st.length && st.every((s) => s === "won" || s === "push")) return "won";
  if (st.some((s) => s !== "pending")) return "live";
  return "pending";
}

export function legResult(l, slate, now, voidAfter = 36 * 3600000) {
  if (!started(l, now)) return null;
  const g = slate.graded[`${l.player}|${l.label}|${l.side}|${l.line}`];
  if (g) return g === "void" ? "push" : g;
  const day = l.gameDate ? logDay(l.gameDate) : l.gameDay, v = day && slate.logs[`${l.playerId}|${l.stat}`]?.[day];
  if (v != null) return v === l.line ? "push" : (l.side === "over" ? v > l.line : v < l.line) ? "win" : "loss";
  if (l.gameDate && now - new Date(l.gameDate) > voidAfter) return "push";
  return null;
}
export function paysPerDollar(legs) {
  if (!legs.length) return 1;
  if (legs.every((l) => l.cost != null)) return 1 / legs.reduce((s, l) => s * l.cost, 1);
  if (legs.every((l) => l.price != null)) return legs.reduce((s, l) => s * decimalOdds(l.price), 1);
  return null;
}
/** fills in leg results and, once decided, the outcome; returns true if anything changed */
export function gradeSaved(p, slate, now) {
  let changed = false;
  // undo results given before a leg's game had started (an older version graded tonight's legs from last night)
  for (const l of p.legs) if (l.result && !started(l, now)) { delete l.result; changed = true; }
  if (changed && p.outcome) { delete p.outcome; delete p.payout; }
  if (p.outcome) return false;
  for (const l of p.legs) if (!l.result) { const r = legResult(l, slate, now); if (r) { l.result = r; changed = true; } }
  if (p.legs.some((l) => l.result === "loss")) { p.outcome = "lost"; p.payout = 0; return true; }
  if (!p.legs.every((l) => l.result)) return changed;
  const live = p.legs.filter((l) => l.result === "win");
  if (!live.length) { p.outcome = "push"; p.payout = 1; } else { p.outcome = "won"; p.payout = paysPerDollar(live) ?? p.decimal; }
  return true;
}
export const savedLeg = (l) => ({ player: l.player, playerId: l.playerId, stat: l.stat, label: l.label, side: l.side,
  line: l.line, p: l.p, gameDay: l.gameDate ? logDay(l.gameDate) : null, gameDate: l.gameDate, price: l.price ?? null, cost: l.cost ?? null });

export function recordStats(list, stake = 10) {
  const s = { decided: 0, won: 0, pushed: 0, pending: 0, predictedSum: 0, staked: 0, returned: 0 };
  for (const p of list) {
    if (!p.outcome) { s.pending++; continue; }
    if (p.outcome === "push") s.pushed++;
    else { s.decided++; s.predictedSum += p.predicted; if (p.outcome === "won") s.won++; }
    if (p.decimal != null && p.payout != null) { s.staked += stake; s.returned += stake * p.payout; }
  }
  s.hitRate = s.decided ? s.won / s.decided : null;
  s.predictedRate = s.decided ? s.predictedSum / s.decided : null;
  s.profit = s.returned - s.staked;
  s.roi = s.staked ? s.profit / s.staked : null;
  s.enough = s.decided >= 20;
  return s;
}
