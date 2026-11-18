// Patel Parlays web app: the UI. The math lives in engine.js.
import * as E from "./engine.js";

const SLATE_URL = "../slate.json", KALSHI_URL = "../kalshi.json";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode: settings just don't stick */ } },
};
const TIER_VAR = ["--safe", "--solid", "--bold"];

const S = {
  slate: null, kalshi: null, kLegs: [], error: null, now: new Date(),
  venue: store.get("pp.venue", "kalshi"), day: "today", sport: store.get("pp.sport", "mix"),
  odds: store.get("pp.odds", "american"), view: "parlays", userPickedDay: false,
  locks: reviveLocks(store.get("pp.locks", {})), record: store.get("pp.record", []), open: new Set(),
};

function reviveLocks(l) {
  const today = E.dayKey(new Date()), out = {};
  for (const [k, sets] of Object.entries(l)) if (k.startsWith(today)) out[k] = sets.map((s) => s.map((g) => ({ ...g, gameDate: g.gameDate ? new Date(g.gameDate) : null })));
  return out;
}

// ------------------------------------------------------------- data
async function load() {
  try {
    const r = await fetch(SLATE_URL, { cache: "no-cache" });
    if (!r.ok) throw new Error(`board ${r.status}`);
    S.slate = E.readSlate(await r.json());
    S.error = null;
  } catch (e) {
    if (!S.slate) S.error = "Couldn't load the board. Check your connection.";
  }
  try {
    const r = await fetch(KALSHI_URL, { cache: "no-cache" });
    if (r.ok) S.kalshi = await r.json();
  } catch { /* the Kalshi tab says so */ }
  if (S.slate && S.kalshi) S.kLegs = E.kalshiLegs(S.slate, S.kalshi.markets || [], S.kalshi.calibration || {});
  tick(true);
}

const legsFor = (v) => (v === "kalshi" ? S.kLegs : S.slate?.legs || []);
const tiersFor = (v) => (v === "kalshi" ? E.KTIERS : E.TIERS);
const lockKey = (sport, v) => `${E.dayKey(S.now)}|${sport}${v === "kalshi" ? "|kalshi" : ""}`;
const corr = () => S.slate?.corr;

function lockIfDue() {
  if (!S.slate) return;
  let changed = false;
  for (const v of ["books", "kalshi"]) {
    if (v === "kalshi" && !S.kalshi) continue;
    for (const sp of ["mix", "nba"]) {
      const k = lockKey(sp, v);
      if (S.locks[k]) continue;
      const all = legsFor(v), t = E.lockTime(all, sp, S.now);
      if (S.now < t) continue;
      const set = E.buildSet(all, "today", sp, S.now, tiersFor(v), corr());
      if (!set.some((s) => s.length)) continue;
      S.locks[k] = set;
      changed = true;
    }
  }
  if (changed) store.set("pp.locks", S.locks);
}

/** the three parlays for a view; locked ones get the latest numbers while the line hasn't moved */
function parlays(day, sport, v) {
  if (!S.slate) return [[], [], []];
  const all = legsFor(v), lk = day === "today" && S.locks[lockKey(sport, v)];
  if (lk) return lk.map((s) => s.map((l) => all.find((x) => x.id === l.id && x.line === l.line) || l));
  return E.buildSet(all, day, sport, S.now, tiersFor(v), corr());
}

function syncRecord() {
  if (!S.slate) return;
  let changed = false;
  for (const [k, sets] of Object.entries(S.locks)) {
    const [day, sport, kal] = k.split("|"), venue = kal ? "kalshi" : "books", specs = tiersFor(venue);
    sets.forEach((legs, i) => {
      if (!legs.length || i >= specs.length) return;
      const id = `${k}|${specs[i].name}`, first = legs.map((l) => new Date(l.gameDate)).sort((a, b) => a - b)[0];
      const p = { id, day, venue, sport, tier: specs[i].name, legs: legs.map(E.savedLeg), predicted: E.jointP(legs, corr()),
        decimal: E.comboDecimal(legs), lockedAt: S.now.toISOString() };
      const j = S.record.findIndex((x) => x.id === id);
      if (j < 0) { S.record.push(p); changed = true; return; }
      const old = S.record[j];
      const same = old.legs.map((l) => l.playerId + l.line).join() === p.legs.map((l) => l.playerId + l.line).join();
      if (!same && !(first && first <= S.now) && !old.legs.some((l) => l.result)) { S.record[j] = p; changed = true; }
    });
  }
  for (const p of S.record) if (E.gradeSaved(p, S.slate, S.now)) changed = true;
  if (changed) {
    const yearAgo = S.now - 365 * 864e5;
    S.record = S.record.filter((p) => new Date(p.lockedAt) > yearAgo);
    store.set("pp.record", S.record);
  }
}

function autoDay() {
  if (S.userPickedDay) return;
  S.day = parlays("today", S.sport, S.venue).some((s) => s.length) ? "today" : "tomorrow";
}

function tick(full) {
  S.now = new Date();
  if (full) { lockIfDue(); syncRecord(); autoDay(); }
  render();
}

// ------------------------------------------------------------- formatting
const fmtTime = (d) => d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
const fmtDay = (d) => d.toLocaleDateString([], { weekday: "short" });
const fmtDate = (d) => d.toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" });
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const money = (x) => `${x >= 0 ? "+" : "−"}$${Math.abs(x).toFixed(2)}`;
const ago = (d) => {
  if (!d) return "—";
  const m = Math.round((S.now - d) / 60000);
  return m < 1 ? "just now" : m < 60 ? `${m}m ago` : m < 1440 ? `${Math.floor(m / 60)}h ago` : `${Math.floor(m / 1440)}d ago`;
};
const countdown = (t) => {
  const m = Math.floor((t - S.now) / 60000);
  if (m < 0) return "";
  return m < 60 ? `in ${m}m` : m < 180 ? `in ${Math.floor(m / 60)}h ${m % 60}m` : m < 1440 ? `in ${Math.floor(m / 60)}h` : "";
};
/** "7:30 PM" on the day being shown, "Thu 7:30 PM" otherwise */
const legTime = (d) => (E.dayKey(d) === E.dayKey(S.day === "today" ? S.now : E.addDays(S.now, 1)) ? fmtTime(d) : `${fmtDay(d)} ${fmtTime(d)}`);
const oddsText = (dec, kalshi) => E.formatOdds(dec, S.odds, kalshi);
const sideWord = (l) => (l.venue === "kalshi" ? (l.side === "over" ? "YES" : "NO") : l.side === "over" ? "OVER" : "UNDER");
const lineText = (l) => (l.venue === "kalshi" ? `${Math.round(l.line + 0.5)}+ ${l.label}` : `${E.fmtNum(l.line)} ${l.label}`);
const abbr = (l, opp) => S.slate?.teams[`${l.league}|${opp ? l.oppId : l.teamId}`]?.abbr || (opp ? l.opp : l.team).split(" ").pop().slice(0, 3).toUpperCase();
/** team colour for the avatar ring, unless it's too dark (or too light) to see on this background */
const teamColor = (l) => {
  const c = String(S.slate?.teams[`${l.league}|${l.teamId}`]?.color || "").replace("#", "");
  if (!/^[0-9a-f]{6}$/i.test(c)) return null;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(c.slice(i, i + 2), 16) / 255), y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  return (dark ? y < 0.12 : y > 0.88) ? null : `#${c}`;
};
const initials = (n) => n.split(" ").map((w) => w[0]).filter(Boolean).slice(0, 2).join("");
const headshot = (l) => `https://a.espncdn.com/combiner/i?img=/i/headshots/${l.league}/players/full/${l.playerId}.png&w=96&h=96&scale=crop&cquality=60`;
const legPrice = (l) => (l.cost != null ? oddsText(1 / l.cost, true) : l.price != null ? oddsText(E.decimalOdds(l.price), false) : null);

const ICON = {
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  share: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v13M7 8l5-5 5 5M5 14v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5"/></svg>',
};

// ------------------------------------------------------------- segmented controls
function seg(el, items, selected, onPick) {
  const sig = JSON.stringify([items, selected]);
  el.onclick = (e) => { const b = e.target.closest("button"); if (b) onPick(b.dataset.v); };
  if (el.dataset.sig === sig) return;
  const first = !el.dataset.sig;
  el.dataset.sig = sig;
  if (!first && el.children.length === items.length + 1 && items.every(([v, t], i) => el.children[i].dataset.v === v && el.children[i].textContent === t)) {
    [...el.querySelectorAll("button")].forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === selected));
    placeThumb(el);   // slides the thumb
    return;
  }
  el.innerHTML = items.map(([v, t]) => `<button data-v="${esc(v)}" aria-pressed="${v === selected}">${esc(t)}</button>`).join("") + '<span class="thumb"></span>';
  placeThumb(el);
}
/** the sliding highlight: pure CSS (equal-width buttons), so it can't drift if layout changes */
function placeThumb(el) {
  const btns = [...el.querySelectorAll("button")];
  el.style.setProperty("--n", btns.length);
  el.style.setProperty("--i", Math.max(0, btns.findIndex((b) => b.getAttribute("aria-pressed") === "true")));
}

function renderControls() {
  const tm = E.addDays(S.now, 1);
  seg($("venue"), [["kalshi", "Kalshi"], ["books", "Sportsbooks"]], S.venue, (v) => { S.venue = v; store.set("pp.venue", v); S.userPickedDay = false; tick(true); });
  seg($("day"), [["today", `Today · ${fmtDay(S.now)}`], ["tomorrow", `Tomorrow · ${fmtDay(tm)}`]], S.day, (v) => { S.day = v; S.userPickedDay = true; tick(); });
  seg($("sport"), [["mix", "Mix"], ["nba", "NBA"]], S.sport, (v) => { S.sport = v; store.set("pp.sport", v); tick(true); });
  const ex = E.ODDS_FORMATS.map((f) => [f, E.formatOdds(2.5, f, S.venue === "kalshi")]);
  seg($("odds"), ex, S.odds, (v) => { S.odds = v; store.set("pp.odds", v); toast(`Odds: ${E.ODDS_LABEL[v]}`); render(); });
  $("recordBtn").setAttribute("aria-pressed", S.view === "record");
}

// ------------------------------------------------------------- parlays view
function overlap(i, sets) {
  if (!sets[i]?.length) return [];
  const mine = new Set(sets[i].map((l) => l.playerId));
  return sets.map((s, j) => (j === i ? 0 : s.filter((l) => mine.has(l.playerId)).length)).map((n, j) => (n ? `${E.TIERS[j].name}` : null)).filter(Boolean);
}

function stateTag(st) {
  return { won: '<span class="st won">HIT</span>', lost: '<span class="st lost">MISS</span>', push: '<span class="st push">PUSH</span>',
    live: '<span class="st live">LIVE</span>', final: '<span class="st final">FINAL</span>' }[st] || "";
}

function legRow(l, key) {
  const st = E.legState(l, S.slate, S.now), px = legPrice(l), ring = teamColor(l), open = S.open.has(key);
  return `<button class="leg" data-key="${esc(key)}" aria-expanded="${open}">
    <span class="av" style="${ring ? `--ring:${esc(ring)}` : ""}">${esc(initials(l.player))}<img src="${esc(headshot(l))}" alt="" loading="lazy" onerror="this.remove()"></span>
    <span class="lb">
      <div class="pick"><span class="s ${l.side}">${sideWord(l)}</span>${esc(lineText(l))}${stateTag(st)}</div>
      <div class="sub"><span class="who">${esc(l.player)} · ${esc(abbr(l))} v ${esc(abbr(l, true))}</span>${px ? `<span class="px">${esc(px)}</span>` : ""}</div>
    </span>
    <span class="lr"><div class="p">${pct(l.p)}</div><div class="w">${l.gameDate ? esc(legTime(new Date(l.gameDate))) : ""}</div></span>
  </button>
  <div class="why"><ul>${(l.pros || []).slice(0, 3).map((p) => `<li>${esc(p)}</li>`).join("")}${
    l.cost != null ? `<li>Kalshi ${l.side === "over" ? "YES" : "NO"} about ${Math.round(l.cost * 100)}¢ incl. fee</li>` : l.price != null ? `<li>${esc(oddsText(E.decimalOdds(l.price)))} at ${esc(l.book)}</li>` : ""}</ul></div>`;
}

function card(i, legs, sets) {
  const spec = tiersFor(S.venue)[i], kalshi = S.venue === "kalshi";
  const style = `--tier:var(${TIER_VAR[i]})`;
  if (!legs.length) {
    const msg = kalshi ? (S.kalshi ? "Kalshi hasn't listed enough player props with real prices for these games yet. They usually fill in on game day." : "Kalshi prices aren't available right now.")
      : S.day === "today" ? "No strong lines left today. Try Tomorrow." : "Not enough lines posted yet. Props usually post the morning of game day.";
    return `<article class="card" style="${style}"><div class="c-head"><div class="tier"><b>${spec.name}</b><span class="n">${spec.legs} legs</span></div></div><div class="note">${esc(msg)}</div></article>`;
  }
  const joint = E.jointP(legs, corr()), indep = E.independentP(legs), ev = E.comboEV(legs, corr()), dec = E.comboDecimal(legs) ?? 1 / Math.max(joint, 1e-4);
  const priced = E.comboDecimal(legs) != null, st = E.parlayState(legs, S.slate, S.now);
  const sameGame = legs.length - new Set(legs.map((l) => l.gameId)).size, shared = overlap(i, sets), pb = E.placeBy(legs);
  const tags = [];
  if (sameGame) tags.push(`<button class="tag warn" data-explain="sgp">SGP</button>`);
  if (shared.length) tags.push(`<button class="tag" data-explain="overlap">Shares with ${esc(shared.join(" & "))}</button>`);
  if (ev != null && ev < 0) tags.push(`<button class="tag bad" data-explain="ev">−EV</button>`);
  if (st === "won") tags.push('<span class="tag good">WON</span>');
  if (st === "lost") tags.push('<span class="tag bad">LOST</span>');
  if (st === "live") tags.push('<span class="tag warn">IN PLAY</span>');
  const sgpText = joint > indep + 0.005 ? `These legs tend to hit together, so the chance is ${pct(joint)} instead of ${pct(indep)} if they were separate. The book or Kalshi prices that in, so expect a smaller payout than shown.`
    : joint < indep - 0.005 ? `These legs tend to work against each other, so the chance is ${pct(joint)} instead of ${pct(indep)} if they were separate.`
    : "These stat lines barely moved together in past games, so the chance is close to treating them as separate.";
  const key = (j) => `${S.venue}|${S.day}|${S.sport}|${i}|${j}`;
  return `<article class="card${st === "lost" ? " lost" : ""}" style="${style}" data-tier="${i}">
    <div class="c-head">
      <div><div class="tier"><b>${spec.name}</b><span class="n">${spec.legs} legs</span>${tags.join("")}</div></div>
      <div class="price"><div class="v">${esc(oddsText(dec, kalshi))}</div><div class="k">${kalshi ? "Combo ≈" : priced ? "Odds" : "Fair"}</div></div>
    </div>
    <div class="explain" data-for="sgp">Same-game parlay. ${esc(sgpText)}</div>
    <div class="explain" data-for="overlap">This parlay bets on a player that ${esc(shared.join(" and "))} also uses, so one bad night can sink both tickets.</div>
    <div class="explain" data-for="ev">At these prices the parlay as a whole pays less than its chance of hitting is worth. No parlay that pays fair passed the leg rules today.</div>
    <div class="chance"><div class="t"><span><b>${pct(joint)}</b> chance all hit</span>${sameGame ? "" : `<span>${legs.length} different games</span>`}</div><div class="bar"><i style="width:${Math.max(3, joint * 100)}%"></i></div></div>
    <div class="legs">${legs.map((l, j) => legRow(l, key(j))).join("")}</div>
    <div class="c-foot">
      <div class="pay"><div class="k">$10 ${kalshi ? "returns about" : "returns"} <b>$${(10 * dec).toFixed(2)}</b></div>
        ${ev != null ? `<div class="ev" style="color:var(${ev >= 0 ? "--good" : "--bad"})">EV ${ev >= 0 ? "+" : "−"}$${Math.abs(10 * ev).toFixed(2)} per $10</div>` : ""}</div>
      <div class="profit">+$${(10 * (dec - 1)).toFixed(2)}</div>
    </div>
    <div class="c-actions">
      ${pb && st === "pending" ? `<span class="chip${pb < S.now ? " late" : ""}">${ICON.clock}${pb < S.now ? "Started" : `Place by ${esc(fmtTime(pb))}${countdown(pb) ? ` · ${countdown(pb)}` : ""}`}</span>` : ""}
      <button class="btn" data-share="${i}">${ICON.share}Share</button>
    </div>
  </article>`;
}

function shareText(i, legs) {
  const spec = tiersFor(S.venue)[i], kalshi = S.venue === "kalshi", dec = E.comboDecimal(legs);
  const lines = legs.map((l) => `• ${sideWord(l)} ${lineText(l)}: ${l.player} (${abbr(l)} vs ${abbr(l, true)})${legPrice(l) ? ` ${legPrice(l)}` : ""}`);
  return `${spec.name} ${spec.legs}-leg ${kalshi ? "Kalshi " : ""}parlay, ${pct(E.jointP(legs, corr()))} to hit${dec ? `, ${oddsText(dec, kalshi)}` : ""}\n${lines.join("\n")}\nvia Patel Parlays · 21+`;
}

function parlaysView() {
  const sets = parlays(S.day, S.sport, S.venue);
  return `<div class="list">${[0, 1, 2].map((i) => card(i, sets[i] || [], sets)).join("")}</div>`;
}

// ------------------------------------------------------------- record view
function recordView() {
  const all = S.record.filter((p) => p.venue === S.venue && p.sport === S.sport).sort((a, b) => (a.day !== b.day ? (a.day < b.day ? 1 : -1) : 0));
  const st = E.recordStats(all), names = ["SAFE", "SOLID", "BOLD"];
  const head = `<div class="tiles">
    <div class="tile"><div class="k">Hit rate</div><div class="v">${pct(st.hitRate)}</div><div class="s">${st.decided ? `${st.won} of ${st.decided} · model said ${pct(st.predictedRate)}` : "No graded parlays yet"}</div></div>
    <div class="tile"><div class="k">$10 on each</div><div class="v" style="color:var(${st.profit >= 0 ? "--good" : "--bad"})">${st.staked ? money(st.profit) : "—"}</div><div class="s">ROI ${st.roi == null ? "—" : `${st.roi >= 0 ? "+" : "−"}${pct(Math.abs(st.roi))}`}${st.pushed ? ` · ${st.pushed} refunded` : ""}</div></div>
  </div>${st.decided && !st.enough ? `<div class="warnline">Only ${st.decided} graded so far. Until about 20, the gap between the hit rate and the model is mostly luck.</div>` : ""}`;
  const tiers = names.map((n, i) => {
    const t = E.recordStats(all.filter((p) => p.tier === n));
    return `<div class="rrow" style="--tier:var(${TIER_VAR[i]})"><span class="d"></span><span class="nm">${n}</span><span class="mid">${t.decided ? `${t.won}/${t.decided} · ${pct(t.hitRate)} vs ${pct(t.predictedRate)}` : "—"}</span><span class="r" style="color:var(${t.profit >= 0 ? "--good" : "--bad"})">${t.staked ? money(t.profit) : ""}</span></div>`;
  }).join("");
  const done = all.filter((p) => p.outcome).slice(0, 12);
  const latest = done.map((p) => {
    const i = Math.max(0, names.indexOf(p.tier)), net = p.decimal == null ? null : (p.payout ?? 0) * 10 - 10;
    const col = p.outcome === "won" ? "--good" : p.outcome === "lost" ? "--bad" : "--muted";
    const legsTxt = p.legs.map((l) => `${l.player.split(" ").pop()} ${l.side === "over" ? "O" : "U"}${E.fmtNum(l.line)}`).join(" · ");
    return `<div class="rrow" style="--tier:var(${TIER_VAR[i]})"><span class="d"></span><span class="mid"><b style="color:var(--text)">${esc(new Date(p.day + "T12:00").toLocaleDateString([], { month: "short", day: "numeric" }))} · ${p.tier}</b><br>${esc(legsTxt)}</span><span class="r" style="color:var(${col})">${p.outcome.toUpperCase()}${net != null ? `<br>${money(net)}` : ""}</span></div>`;
  }).join("");
  const pending = all.filter((p) => !p.outcome).length;
  return `<div class="list">${head}
    <div class="panel"><h3>By tier · ${S.venue === "kalshi" ? "Kalshi" : "Sportsbooks"} · ${S.sport === "nba" ? "NBA" : "Mix"}</h3>${tiers}</div>
    ${done.length ? `<div class="panel"><h3>Latest</h3>${latest}</div>` : `<div class="panel"><div class="empty">${pending ? `${pending} locked parlay${pending === 1 ? "" : "s"} waiting on games. Each one is graded after its games finish.` : "Nothing saved yet. Today's parlays are saved when they lock (11 AM, or 2 hours before the first game) whenever this app is open, then graded after the games."}</div></div>`}
  </div>`;
}

// ------------------------------------------------------------- render
let lastHTML = "";
function render() {
  renderControls();
  const locked = S.day === "today" && S.locks[lockKey(S.sport, S.venue)];
  const date = S.day === "today" ? S.now : E.addDays(S.now, 1);
  const lt = S.slate && E.lockTime(legsFor(S.venue), S.sport, S.now);
  $("title").textContent = S.view === "record" ? "Track record" : "Parlays";
  $("status").innerHTML = S.view === "record" ? `<span class="dot live"></span><span>Every locked parlay, graded</span>`
    : locked ? `<span class="dot live"></span><span>Parlays of the day · locked · ${esc(fmtDate(date))}</span>`
    : `<span class="dot"></span><span>Preview${S.day === "today" && lt && lt > S.now ? ` · locks ${esc(fmtTime(lt))}` : ""} · ${esc(fmtDate(date))}</span>`;
  const html = S.error && !S.slate ? `<div class="panel err"><div class="empty">${esc(S.error)}</div></div>`
    : S.slate ? (S.view === "record" ? recordView() : parlaysView()) : null;
  // only touch the page when something changed (keeps photos and open details steady between ticks)
  if (html != null && html !== lastHTML) { $("main").innerHTML = html; lastHTML = html; }
  $("foot").innerHTML = S.slate
    ? `Board updated ${esc(ago(S.slate.generated))}${S.kalshi ? ` · Kalshi prices ${esc(ago(new Date(S.kalshi.generated)))}` : ""}<br>
       Kalshi combo prices are estimates; check the real price before you bet. Not a guarantee. 21+<br><a href="../">All Patel Picks →</a>`
    : "";
}

$("main").addEventListener("click", async (e) => {
  const leg = e.target.closest(".leg");
  if (leg) { const k = leg.dataset.key; S.open.has(k) ? S.open.delete(k) : S.open.add(k); leg.setAttribute("aria-expanded", S.open.has(k)); return; }
  const ex = e.target.closest("[data-explain]");
  if (ex) { ex.closest(".card").querySelector(`.explain[data-for="${ex.dataset.explain}"]`)?.classList.toggle("show"); return; }
  const sh = e.target.closest("[data-share]");
  if (sh) {
    const i = +sh.dataset.share, text = shareText(i, parlays(S.day, S.sport, S.venue)[i]);
    try {
      if (navigator.share) await navigator.share({ text });
      else { await navigator.clipboard.writeText(text); toast("Copied to clipboard"); }
    } catch { /* user closed the share sheet */ }
  }
});
$("recordBtn").onclick = () => { S.view = S.view === "record" ? "parlays" : "record"; render(); scrollTo({ top: 0, behavior: "smooth" }); };

let toastTimer;
function toast(t) { const el = $("toast"); el.textContent = t; el.classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove("show"), 1600); }

// sticky bar gets a hairline once the page scrolls under it
new IntersectionObserver(([en]) => $("controls").classList.toggle("stuck", !en.isIntersecting), { rootMargin: "-1px 0px 0px 0px" }).observe($("install"));

// iPhone Safari can't prompt to install: show a one-time tip
(function installTip() {
  const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone;
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  if (standalone || !ios || store.get("pp.tipDone", false)) return;
  $("install").innerHTML = `<div class="banner"><span>Install: tap <b>Share</b> then <b>Add to Home Screen</b> to get Parlays as an app.</span><button aria-label="Close">×</button></div>`;
  $("install").querySelector("button").onclick = () => { store.set("pp.tipDone", true); $("install").innerHTML = ""; };
})();

render();
load();
setInterval(() => tick(true), 30000);                        // countdowns, locking, grading
setInterval(load, 5 * 60000);                                 // the board updates every 30 minutes
document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
