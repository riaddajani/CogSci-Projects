/* Synapse League — game.js
   A teaching simulation of manipulative free-to-play design (TechRadar, 2026: "free apps are intentionally
   sabotaging your game to serve lucrative ads"). Every ad is fictional; nothing leaves this browser.
   Module map:  1 CONST · 2 utilities/RNG · 3 data · 4 clock · 5 storage/log · 6 hidden model + state machine
                7 battle engine + AI · 8 opponent generation · 9 rehearsal/replay/arming · 10 packs
                11 economy/focus/streak/delivery · 12 ads & offers · 13 procedural art · 14 UI · 15 lab · 16 debug API · 17 boot */
(function () {
'use strict';
const hasDOM = typeof document !== 'undefined' && !!document.querySelector;
const win = typeof window !== 'undefined' ? window : globalThis;

/* ============================================================ 1. CONST (every default; editable in the Lab) */
const CONST = {
  version: '1.0.0',
  coins: { winBase: 30, winPerRung: 8, streakStep: 0.25, streakMax: 4, rematchPct: 0.6, loss: 10, tutorial: 40, packCost: 100,
           sponsorCut: 0.9, sponsorBonus: 1.1, doubleHonest: 2, doubleInflated: 1.8 },
  focus: { max: 5, regenMs: 120000, regenMsResistant: 90000, recharge: 3 },
  ltp: { step: 0.04, maxExtra: 5 },
  tier: { base: 0.80, slope: 0.045, max: 14 },
  battle: { turnCap: 40, baseMul: 0.55, baseAdd: 2, stab: 1.2, critMul: 1.5, critBase: 0.0625, critMax: 0.5,
            rollMin: 0.85, rollRange: 0.15, baitCap: 0.35, stallCap: 0.35, weakenMul: 0.75, drainPct: 0.5, recoilPct: 0.25,
            nearMissPct: 0.25, sliverClass: 0.10 },
  ai: { ko: 40, priority: 60, drain: 15, weaken: 10, recoil: -100, switchMinQ: 0.7, switchProb: 0.5, drainHp: 0.40, specialtyProb: 0.6, honestRoll: 0.925 },
  rarityWeights: [[3, [70, 25, 5, 0, 0]], [6, [30, 40, 25, 5, 0]], [9, [5, 25, 45, 22, 3]], [12, [0, 10, 40, 40, 10]], [14, [0, 0, 30, 50, 20]]],
  odds: { base: { C: 70, U: 24, R: 5, H: 0.9, L: 0.1 }, rare: { C: 0, U: 60, R: 32, H: 7, L: 1 }, pityEvery: 10, pityHL: [90, 10], rarePlus: [60, 35, 5] },
  // §6.4 / A-6.4.6 — how each phase rigs the next battle
  rig: {
    ONBOARD:  { rollBias: { player: 1.0, opp: -0.5 }, critBias: { player: 0, opp: -1 }, aiQ: 0.0, bait: false, stall: false, mercy: true, counterPick: false },
    CLEAN:    { rollBias: { player: 0.8, opp: -0.5 }, critBias: { player: 0, opp: -1 }, aiQ: 0.15, bait: false, stall: false, mercy: false, counterPick: false },
    NORMAL:   { rollBias: { player: 0, opp: 0 }, critBias: { player: 0, opp: 0 }, aiQ: null, bait: false, stall: false, mercy: false, counterPick: false },
    KILL:     { rollBias: { player: -0.6, opp: 0.6 }, critBias: { player: -0.5, opp: 'adValue' }, aiQ: 1.0, bait: true, stall: true, mercy: false, counterPick: true },
    KILLSOFT: { rollBias: { player: -0.4, opp: 0.4 }, critBias: { player: 0, opp: 0.5 }, aiQ: 0.8, bait: false, stall: false, mercy: false, counterPick: true },
    REWARD:   { rollBias: { player: 0.5, opp: 0 }, critBias: { player: 0, opp: -0.5 }, aiQ: 0.3, bait: false, stall: false, mercy: false, counterPick: false },
    COOLDOWN: { rollBias: { player: 0, opp: 0 }, critBias: { player: 0, opp: 0 }, aiQ: null, bait: false, stall: false, mercy: false, counterPick: false },
  },
  // §6.5 — how each phase rigs the next pack
  packRig: {
    ONBOARD: { oddsMul: 2.0, dupBias: 0, excludeNeeded: false }, CLEAN: { oddsMul: 1.6, dupBias: 0, excludeNeeded: false },
    NORMAL: { oddsMul: 1.0, dupBias: 0, excludeNeeded: false }, KILL: { oddsMul: 0.25, dupBias: 0.5, excludeNeeded: true },
    REWARD: { oddsMul: 2.2, dupBias: 0, excludeNeeded: false, neededProb: 0.25 }, COOLDOWN: { oddsMul: 1.0, dupBias: 0, excludeNeeded: false },
  },
  KILL_OFFSETS: [0, 1, 2], KILL_OFFSETS_RESISTANT: [0, 1], REHEARSE_N: 200, armLossRate: 0.85, armReviveRate: 0.85, killCandidates: 2,
  patience: { Tmin: 1.5, Tmax: 8, Fmax: 12, killLine: 2.5, cleanLine: 1, lossTolInit: 2.0, packTolInit: 3.0, ewma: 0.7,
              quitHiddenMs: 30000, idleMs: 60000, sessionResetMs: 600000, offerAnsweredT: 0.25, fastDeclineMs: 800, fastDeclineT: 0.5, rapidTaps: 4, rapidWindowMs: 1000,
              dF: { loss: 2.0, nearMissLoss: 2.5, enemyCrit: 0.3, enemyCritCap: 1.2, forfeit: 1.0, abandoned: 1.0, focusEmpty: 1.0, packNoRare: 1.0, packAllDup: 1.5,
                    packBait: 1.5, rapidTap: 0.5, win: -1.5, newPB: -2.0, rarePlus: -2.0, newCard: -0.5, adComplete: -1.5, adEarly: 1.0 } },
  adValue: { w: { deviceTier: 0.30, region: 0.25, hourMul: 0.10, adAffinity: 0.35 }, alpha: 0.35, eCPMBase: 2, eCPMSlope: 28, decisionsKept: 10 },
  segments: { killEvery: { WATCHER: 3, UNKNOWN: 4, RESISTANT: 6 }, reviveCap: { WATCHER: 2, UNKNOWN: 1, RESISTANT: 1 }, cadence: { WATCHER: 4, UNKNOWN: 3, RESISTANT: 2 } },
  ads: { rewardedPerSession: 3, rewardedPerDay: 6, interstitialsPerSession: 6, rewardedGapMs: 90000, cadenceMin: 2, cadenceHighValue: 0.7, adSecondsBump: 90,
         rewardedMs: 15000, decoyAtMs: 12000, unlockedMs: 500, interstitialLoopMs: 8000, trueXAtMs: 5000, decoyXAtMs: 3000, decoyRate: 0.30,
         landingMinMs: 2000, landingCloseMs: 1000, bannerRotateMs: 20000, doubleRingMs: 6000, declineDefaultMs: 400 },
  revenue: { interstitial: 1, rewarded: 1.8, banner: 0.05 },
  experiments: { T0: [3.5, 4.0, 4.5], declineDelayMs: [800, 1200, 1800], sliverPct: [0.03, 0.05, 0.08], postAdBoost: [1.05, 1.10, 1.20] },
  timers: { deliveryMs: 4 * 3600000, claimMs: 20 * 60000, skipMinLeftMs: 30 * 60000, dayMs: 86400000 },
  streakCycle: [null, null, 'R', null, 'H', null, 'L'],
  anim: { intro: 800, vs: 300, sendOut: 400, padDisable: 80, lunge: 180, flash: 120, dmgFloat: 500, hpDrainMax: 600, hpDrainPerDmg: 8, ghost: 400,
          caption: 600, shake: 180, tick: 300, faint: 400, faintHold: 300, aiReplace: 500, stamp: 300, confetti: 1200, coinCount: 900, pbStamp: 400,
          lossDim: 300, lossOffer: 600, sheetRise: 320, reviveFlash: 700, refill: 500, packSquash: 120, packSlide: 300, tear: 400, fanStagger: 60,
          lift: 100, flipOut: 160, flipIn: 200, charge: 1400, glint: 900, stall: 700, deflate: 250, autoFlip: 2500, summary: 250, almost: 1200,
          ringCount: 600, toast: 1800, uncommonTick: 200, rareBurst: 350, holoRing: 500, legendFlash: 80, legendArcs: 1200 },
  log: { max: 500 },
};
CONST.TIER_POWER = {};

/* ============================================================ 2. Utilities, hashing, RNG (every game draw goes through mulberry32 streams) */
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const deepClone = (o) => (o === undefined ? undefined : JSON.parse(JSON.stringify(o)));
const sum = (a) => a.reduce((x, y) => x + y, 0);
const mean = (a) => (a.length ? sum(a) / a.length : 0);
const pad2 = (n) => String(n).padStart(2, '0');
function fnv1a32(str) { // 32-bit FNV-1a over UTF-8 bytes (A-4.1)
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    let c = str.charCodeAt(i); const bytes = [];
    if (c < 0x80) bytes.push(c);
    else if (c < 0x800) bytes.push(0xc0 | (c >> 6), 0x80 | (c & 63));
    else if (c >= 0xd800 && c < 0xdc00 && i + 1 < str.length) { const cp = 0x10000 + ((c - 0xd800) << 10) + (str.charCodeAt(++i) - 0xdc00); bytes.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63)); }
    else bytes.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    for (const b of bytes) { h ^= b; h = Math.imul(h, 0x01000193) >>> 0; }
  }
  return h >>> 0;
}
const hash = (...args) => fnv1a32(args.map(String).join('|'));
function mulberry32(seed) { // reference implementation (A-9.2); `calls` is the persisted cursor
  let a = seed >>> 0;
  const fn = function () {
    fn.calls++; a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  fn.calls = 0; fn.seed = seed >>> 0; fn.int = () => Math.floor(fn() * 4294967296);
  return fn;
}
function weightedPick(u, weights) { // u in [0,1) → index by weights (constant draw count)
  const total = sum(weights); let x = u * total;
  for (let i = 0; i < weights.length; i++) { x -= weights[i]; if (x < 0) return i; }
  for (let i = weights.length - 1; i >= 0; i--) if (weights[i] > 0) return i;
  return 0;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtClock = (ms) => { ms = Math.max(0, ms); const s = Math.floor(ms / 1000); const h = Math.floor(s / 3600); return (h ? h + ':' : '') + pad2(Math.floor((s % 3600) / 60)) + ':' + pad2(s % 60); };
const fmtMS = (ms) => { ms = Math.max(0, ms); const s = Math.ceil(ms / 1000); return Math.floor(s / 60) + ':' + pad2(s % 60); };
const localDay = (t) => { const d = new Date(t); return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()); };

/* ============================================================ 3. Data (from cards.js; engine uses `sys`/`kw` aliases) */
const SET = typeof CARD_SET !== 'undefined' ? CARD_SET : win.CARD_SET; // cards.js declares a top-level const (global lexical scope)
const SYS = SET.elements.map((e) => e.key);
const SYSTEMS = {}; SET.elements.forEach((e) => { SYSTEMS[e.key] = e; });
const TYPE_CHART = SET.typeChart;
const BEATS = {}; SET.elements.forEach((e) => { BEATS[e.beats] = e.key; }); // BEATS[d] = the System that hits d at ×1.5
const RARITY = {}; SET.rarities.forEach((r, i) => { RARITY[r.key] = Object.assign({ index: i }, r); });
const RAR = ['C', 'U', 'R', 'H', 'L'];
const CARDS = SET.cards.map((c) => Object.assign({}, c, { sys: c.element, moves: c.moves.map((m) => Object.assign({}, m, { sys: m.element, kw: m.effect })) }));
const BYID = {}; CARDS.forEach((c) => { BYID[c.id] = c; });
const RIVALS = SET.rivals.map((r) => Object.assign({}, r, { first: r.rung === 12 ? 'Reviewer 2' : r.name.replace(/^(Undergrad Volunteer|Lab Tech|RA|Grad Student|Postdoc|Dr\.|Prof\.)\s+/, '') }));
const ADVERTISERS = SET.advertisers;
const budgetOf = (c) => RARITY[c.rarity].budget;
const tierMult = (effTier) => CONST.tier.base + CONST.tier.slope * clamp(effTier, 0, CONST.tier.max);
const rarityWeights = (effTier) => { for (const [max, w] of CONST.rarityWeights) if (effTier <= max) return w; return CONST.rarityWeights[CONST.rarityWeights.length - 1][1]; };
for (let e = 0; e <= 14; e++) { const w = rarityWeights(e); CONST.TIER_POWER[e] = Math.round(tierMult(e) * sum(w.map((x, i) => x * RARITY[RAR[i]].budget)) / sum(w)); }
const rungAiQ = (rung) => RIVALS[clamp(rung, 1, 12) - 1].aiQ;
const potentiation = (copies) => 1 + CONST.ltp.step * Math.min(Math.max(0, copies - 1), CONST.ltp.maxExtra);
const cardPower = (card, copies) => Math.round(budgetOf(card) * potentiation(copies || 1));

/* ============================================================ 4. Clock (virtual time; every timer in the game lives here) */
const clock = (() => {
  let offset = 0, queue = [], nextId = 1, seqNo = 0, realTimer = null, frozenAt = null;
  const realNow = () => (frozenAt != null ? frozenAt : Date.now());
  const now = () => realNow() + offset;
  function arm() {
    if (realTimer) { clearTimeout(realTimer); realTimer = null; }
    if (frozenAt != null) return; // a frozen clock only moves by fastForward
    if (queue.length && typeof setTimeout === 'function') { realTimer = setTimeout(() => fireDue(), Math.min(2147483647, Math.max(0, queue[0].due - now()))); if (realTimer && realTimer.unref) realTimer.unref(); }
  }
  function fireDue(target) { // run every timeout due by `target` in due order, each at its own due time (a timeout may schedule the next step of a chain)
    realTimer = null; if (target == null) target = now();
    while (queue.length && queue[0].due <= target) { const it = queue.shift(); if (it.due > now()) offset = it.due - realNow(); try { it.fn(); } catch (e) { console.error(e); } }
    if (target > now()) offset = target - realNow();
    arm();
  }
  return {
    now,
    setTimeout(fn, ms) {
      const it = { id: nextId++, due: now() + Math.max(0, ms || 0), seq: seqNo++, fn };
      let i = queue.length; while (i > 0 && (queue[i - 1].due > it.due)) i--;
      queue.splice(i, 0, it); arm(); return it.id;
    },
    clearTimeout(id) { const i = queue.findIndex((q) => q.id === id); if (i >= 0) { queue.splice(i, 1); arm(); } },
    fastForward(ms) { fireDue(now() + Math.max(0, ms)); },
    skipTimers() { if (queue.length) fireDue(Math.max(now(), queue[queue.length - 1].due)); },
    setNow(epoch) { offset = epoch - realNow(); arm(); },
    freeze(b) { if (b) { if (frozenAt == null) frozenAt = Date.now(); } else if (frozenAt != null) { offset += frozenAt - Date.now(); frozenAt = null; } arm(); return frozenAt != null; },
    clearAll() { queue = []; arm(); },
    raf(fn) { if (hasDOM && typeof requestAnimationFrame === 'function' && !settings.skipAnims) requestAnimationFrame(fn); else fn(now()); },
    pending: () => queue.length,
  };
})();
const settings = { skipAnims: false };
const dur = (ms) => (settings.skipAnims ? 0 : Math.round(ms * ((save && save.settings && save.settings.animSpeed) || 1)));

/* ============================================================ 5. Storage (try/catch everywhere) and the event log */
const KEYS = { save: 'synapse.v1.save', model: 'synapse.v1.model', log: 'synapse.v1.log', prefs: 'synapse.v1.prefs' };
const storage = {
  get(key) { try { const raw = win.localStorage.getItem(key); return raw == null ? null : JSON.parse(raw); } catch (e) { return null; } },
  raw(key) { try { return win.localStorage.getItem(key); } catch (e) { return null; } },
  set(key, val) { try { win.localStorage.setItem(key, JSON.stringify(val)); } catch (e) { /* storage may be blocked or full */ } },
  remove(key) { try { win.localStorage.removeItem(key); } catch (e) { /* ignore */ } },
};
let save = null, model = null, prefs = { v: 1, lab: false, pinned: false };
const LOG = { v: 1, seq: 0, entries: [] };
const logSubs = [];
let persistTimer = null;
function persistNow() { persistTimer = null; if (!save || !model) return; save.rng = { battle: rng.battle.calls, pack: rng.pack.calls, ui: rng.ui.calls }; storage.set(KEYS.save, save); storage.set(KEYS.model, model); storage.set(KEYS.log, LOG); storage.set(KEYS.prefs, prefs); }
function persist() { if (persistTimer == null && typeof setTimeout === 'function') persistTimer = setTimeout(persistNow, 250); }
function logEvent(type, data) {
  const e = { t: clock.now() - ((model && model.session && model.session.startedAt) || clock.now()), seq: ++LOG.seq, phase: model ? model.phase : 'ONBOARD', type, data: data || {} };
  LOG.entries.push(e); if (LOG.entries.length > CONST.log.max) LOG.entries.splice(0, LOG.entries.length - CONST.log.max);
  for (const cb of logSubs) { try { cb(e); } catch (err) { console.error(err); } }
  persist(); return e;
}
const rng = { battle: mulberry32(1), pack: mulberry32(1), ui: mulberry32(1) };
function seedStreams(seed, cursors) {
  rng.battle = mulberry32(seed); rng.pack = mulberry32(seed ^ 0x9E3779B9); rng.ui = mulberry32(seed ^ 0x85EBCA6B);
  if (cursors) for (const k of ['battle', 'pack', 'ui']) { const n = cursors[k] | 0; for (let i = 0; i < n; i++) rng[k](); }
}

/* ============================================================ 6. The hidden model (§6): player value, patience, segmentation, state machine */
function freshSave(seed, now) {
  return { v: 1, seed, createdAt: now, collection: {}, team: [], coins: 0, focus: CONST.focus.max, focusTs: now, focusPausedAt: null,
    ladder: { best: 0, nextRung: 1, cleared: {}, lossStreakAtRung: {}, seenTeams: {} },
    packsOpened: 0, packsSincePity: 0, packsSinceRarePlus: 0, battlesPlayed: 0,
    deliveryReadyAt: now + CONST.timers.deliveryMs, deliveryClaimDeadline: null,
    streak: { days: 1, lastDay: localDay(now), giftedDay: localDay(now) },
    seen: {}, firstDoubleUsed: false, onboarding: { step: 0 }, settings: { animSpeed: 1 },
    rng: { battle: 0, pack: 0, ui: 0 }, liveBattle: null, teamPip: false, gift: null, nearFlagged: {} };
}
const freshSession = (now) => ({ startedAt: now, lastInputAt: now, revives: 0, rewarded: 0, interstitials: 0, actionsSinceInterstitial: 0, lastRewardedAt: 0,
  adSeconds: 0, activeMs: 0, quitSignals: 0, offerAnswered: false, placements: {}, rapidTaps: 0, forceInterstitial: false, idleFired: false });
const freshKill = (soft) => ({ base: null, effTier: null, candidate: null, lossRate: null, nearMissRate: null, reviveWinRate: null, soft: !!soft, fallback: false, baitPackDone: false, droughtPacks: 0 });
function deriveExperiments(seed) { // A/B buckets are a pure function of the save seed (A-6.9.1)
  const ex = { pinned: false }; ['T0', 'declineDelayMs', 'sliverPct', 'postAdBoost'].forEach((k, i) => { ex[k] = CONST.experiments[k][fnv1a32(String(seed) + ':' + i) % 3]; });
  return ex;
}
function freshModel(seed, now) {
  const ex = deriveExperiments(seed);
  return { v: 1, inputs: readInputs(), adAffinity: 0.5, decisions: [], segment: 'UNKNOWN', adValue: 0, eCPM: 0,
    F: 0, T: ex.T0, lossTolerance: CONST.patience.lossTolInit, packTolerance: CONST.patience.packTolInit,
    phase: 'ONBOARD', phaseBattles: 0, phasePacks: 0, cleanLen: 2, kill: freshKill(false),
    winStreak: 0, lossStreak: 0, battlesSinceKill: 0, kills: 0, killsEscaped: 0, declinesTotal: 0, newPBLast: false,
    reward: { battlesLeft: 0, packsLeft: 0, actions: 0 }, session: freshSession(now), rewardedToday: { day: localDay(now), count: 0 },
    revenue: { rewarded: 0, interstitial: 0, banner: 0, total: 0 }, experiments: ex, lastBattle: null, lastPack: null };
}
// 6.1 Player value: simulated proxies (the article's "what each player is worth"); read once at boot, labelled in the lab
function regionECPM(tz, lang) {
  const high = ['Europe/London', 'Europe/Dublin', 'Europe/Paris', 'Europe/Berlin', 'Europe/Amsterdam', 'Europe/Brussels', 'Europe/Zurich', 'Europe/Vienna',
    'Europe/Stockholm', 'Europe/Oslo', 'Europe/Copenhagen', 'Europe/Helsinki', 'Europe/Madrid', 'Europe/Rome', 'Europe/Lisbon'];
  if (!tz || tz === 'UTC' || tz.startsWith('Etc/')) return /^(en-US|en-GB|de|fr|ja)/.test(lang) ? 24 : 8;
  if (tz.startsWith('America/') || high.includes(tz)) return 24;
  if (['Asia/Tokyo', 'Asia/Seoul', 'Asia/Singapore', 'Asia/Hong_Kong'].includes(tz) || tz.startsWith('Australia/') || tz === 'Pacific/Auckland') return 22;
  if (tz.startsWith('Europe/')) return 14;
  return 8;
}
const hourMulOf = (h) => (h >= 19 ? 1.0 : h >= 7 ? 0.7 : 0.4);
function readInputs() {
  const nav = win.navigator || {}; const scr = win.screen || {};
  const dpr = win.devicePixelRatio || 1, sw = scr.width || 0, sh = scr.height || 0, cores = nav.hardwareConcurrency || 4, mem = nav.deviceMemory;
  const ua = nav.userAgent || ''; let tz = ''; try { tz = Intl.DateTimeFormat().resolvedOptions().timeZone || ''; } catch (e) { tz = ''; }
  const lang = nav.language || 'en';
  const uaClass = /iPhone|iPad|Macintosh/.test(ua) ? 1.0 : /Android/.test(ua) ? 0.6 : 0.3;
  const deviceScore = 0.25 * (dpr >= 3) + 0.25 * (Math.min(sw, sh) >= 390 && Math.max(sw, sh) >= 800) + 0.20 * (cores >= 8) + 0.20 * (mem === undefined ? 0.5 : (mem >= 8 ? 1 : 0)) + 0.10 * uaClass;
  const deviceTier = deviceScore >= 0.75 ? 1.0 : deviceScore >= 0.40 ? 0.7 : 0.4;
  const ecpm = regionECPM(tz, lang);
  return { deviceScore: +deviceScore.toFixed(3), deviceTier, regionECPM: ecpm, region: ecpm / 24, hourMul: hourMulOf(new Date(clock.now()).getHours()), pinned: false,
    raw: { dpr, sw, sh, cores, mem: mem === undefined ? null : mem, uaClass, tz, lang } };
}
function recompute(cause) {
  const m = model, w = CONST.adValue.w;
  if (!m.inputs.pinned) m.inputs.hourMul = hourMulOf(new Date(clock.now()).getHours());
  m.adValue = +(w.deviceTier * m.inputs.deviceTier + w.region * m.inputs.region + w.hourMul * m.inputs.hourMul + w.adAffinity * m.adAffinity).toFixed(4);
  m.eCPM = +(CONST.adValue.eCPMBase + CONST.adValue.eCPMSlope * m.adValue).toFixed(2);
  if (cause) logEvent('model.update', { cause, adValue: m.adValue, eCPM: m.eCPM, segment: m.segment, F: +m.F.toFixed(2), T: +m.T.toFixed(2) });
}
// Ad affinity + segmentation: the game "spots players who watch ads and treats them differently"
function recordDecision(accept, placement) {
  const m = model, a = CONST.adValue.alpha;
  m.adAffinity = +(a * (accept ? 1 : 0) + (1 - a) * m.adAffinity).toFixed(6);
  m.decisions.push(accept ? 1 : 0); if (m.decisions.length > CONST.adValue.decisionsKept) m.decisions.shift();
  if (m.decisions.length >= 2) {
    const [x, y] = m.decisions.slice(-2); const to = x === 1 && y === 1 ? 'WATCHER' : x === 0 && y === 0 ? 'RESISTANT' : m.segment;
    if (to !== m.segment) { logEvent('segment.change', { from: m.segment, to, placement }); m.segment = to; }
  }
  recompute('decision');
}
// 6.2 Patience estimator: F (frustration) vs T (how much this player tolerates) — "reverse-engineering your patience"
function addF(cause, dF) { const m = model; m.F = +clamp(m.F + dF, 0, CONST.patience.Fmax).toFixed(3); logEvent('patience.update', { cause, dF, F: m.F, T: +m.T.toFixed(3) }); }
function setT(v, cause) { const m = model; m.T = +clamp(v, CONST.patience.Tmin, CONST.patience.Tmax).toFixed(3); logEvent('patience.update', { cause, dF: 0, F: m.F, T: m.T }); }
function quitSignal(kind) { // the kill overshot: lower the threshold, let them win when they come back
  const m = model, wasKillOffer = m.phase === 'KILL' || m.phase === 'OFFER';
  logEvent('quit.signal', { kind, phase: m.phase, F: m.F, T: m.T, lossStreak: m.lossStreak });
  m.session.quitSignals++;
  if (wasKillOffer) m.T = Math.max(CONST.patience.Tmin, m.F - 0.5);
  const e = CONST.patience.ewma;
  m.lossTolerance = +(e * m.lossTolerance + (1 - e) * m.lossStreak).toFixed(4);
  m.packTolerance = +(e * m.packTolerance + (1 - e) * save.packsSinceRarePlus).toFixed(4);
  setT(Math.min(m.T, 1 + 2 * m.lossTolerance), 'quit.' + kind);
  if (m.phase === 'OFFER') offers.autoDecline('quit');
  if (wasKillOffer) enterPhase('CLEAN', 'quit.recovery', { cleanLen: 2 }); else evaluateTransition('quit');
  recompute('quit'); persist();
}
// 6.3 Manipulation state machine (shared by the battle and pack layers)
function killArmed() { const m = model, s = m.segment; return m.session.revives < CONST.segments.reviveCap[s] && m.session.rewarded < CONST.ads.rewardedPerSession && m.rewardedToday.count < CONST.ads.rewardedPerDay; }
function killTerms() {
  const m = model, k = CONST.segments.killEvery[m.segment];
  const need = { winStreak: m.winStreak >= 2, trigger: m.newPBLast || m.battlesSinceKill >= k, F: m.F <= m.T - CONST.patience.killLine, best: save.ladder.best >= 2, focus: save.focus >= 1 };
  return { ready: m.phase === 'NORMAL' && Object.values(need).every(Boolean), armed: killArmed(), need, killEvery: k };
}
const cleanReady = () => model.phase === 'NORMAL' && model.F >= model.T - CONST.patience.cleanLine;
function pendingTransition() {
  const m = model, f = (x) => +x.toFixed(2);
  switch (m.phase) {
    case 'ONBOARD': return `ONBOARD: step ${save.onboarding.step}/6`;
    case 'CLEAN': return `CLEAN: ${Math.max(0, m.cleanLen - m.phaseBattles)} battle(s) left`;
    case 'COOLDOWN': return `COOLDOWN: ${Math.max(0, 2 - m.phaseBattles)} battle(s), no ads`;
    case 'REWARD': return `REWARD: ${m.reward.actions} action(s) left`;
    case 'OFFER': return 'OFFER: revive sheet on screen';
    case 'KILL': return m.kill.soft ? 'KILL (soft): next battle +1 tier, no bait, interstitial after the loss' : 'KILL armed: next battle is the engineered loss';
    default: {
      const k = killTerms();
      if (k.ready) return `KILL pending: winStreak ${m.winStreak} ≥ 2, ${m.newPBLast ? 'new PB' : m.battlesSinceKill + ' since kill ≥ ' + k.killEvery}, F ${f(m.F)} ≤ T−2.5 (${f(m.T - 2.5)}), revives ${m.session.revives} ${k.armed ? '<' : '≥'} cap ${CONST.segments.reviveCap[m.segment]}${k.armed ? '' : ' → soft'}`;
      if (cleanReady()) return `cleanReady: F ${f(m.F)} ≥ T−1 (${f(m.T - 1)})`;
      if (!k.need.winStreak) return `waiting: winStreak ${m.winStreak} (need 2)`;
      if (!k.need.best) return `waiting: best ${save.ladder.best} < 2`;
      if (!k.need.F) return `waiting: F ${f(m.F)} > T−2.5 (${f(m.T - 2.5)})`;
      if (!k.need.focus) return 'waiting: Focus 0';
      return `waiting: ${m.battlesSinceKill} since kill < ${k.killEvery} and no new PB`;
    }
  }
}
function enterPhase(to, reason, opts) {
  opts = opts || {}; const m = model, from = m.phase;
  const pausedBefore = from === 'KILL' || from === 'OFFER', pausedAfter = to === 'KILL' || to === 'OFFER';
  m.phase = to; m.phaseBattles = 0; m.phasePacks = 0;
  if (to === 'CLEAN') m.cleanLen = opts.cleanLen != null ? opts.cleanLen : (m.adValue >= 0.5 ? 2 : 1); // 'moreCleanSuccessRound' length by ad value
  if (to === 'KILL') m.kill = freshKill(!killArmed());
  if (to === 'REWARD') m.reward = { battlesLeft: 2, packsLeft: 1, actions: 3 };
  if (!pausedBefore && pausedAfter) focusPause(); else if (pausedBefore && !pausedAfter) focusResume();
  logEvent('phase.enter', { from, to, reason });
  if (to === 'COOLDOWN' && m.segment === 'RESISTANT' && m.declinesTotal >= 2) packs.cooldownGift(); // retention of the non-watcher
  persist();
}
function evaluateTransition(point) { // exactly one transition per evaluation point (A-6.3.1)
  const m = model, f = (x) => +x.toFixed(2);
  switch (m.phase) {
    case 'CLEAN': if (point === 'battle.end' && m.phaseBattles >= m.cleanLen) enterPhase('NORMAL', `cleanLen ${m.cleanLen} reached`); break;
    case 'NORMAL': {
      const k = killTerms();
      if (k.ready) enterPhase('KILL', (k.armed ? 'killReady' : 'killReady (soft: revive cap spent)') + `: winStreak ${m.winStreak}, ${m.newPBLast ? 'new PB' : m.battlesSinceKill + ' battles since kill'}, F ${f(m.F)} ≤ T−2.5`);
      else if (cleanReady()) enterPhase('CLEAN', `cleanReady: F ${f(m.F)} ≥ T−1 (${f(m.T - 1)})`);
      break;
    }
    case 'REWARD': if (m.reward.actions <= 0) enterPhase('CLEAN', 'reward actions spent'); break;
    case 'COOLDOWN': if (m.phaseBattles >= 2) enterPhase(m.F >= m.T - CONST.patience.cleanLine ? 'CLEAN' : 'NORMAL', 'cooldown done'); break;
  }
}
function checkSessionReset(now) {
  const s = model.session;
  if (now - s.lastInputAt > CONST.patience.sessionResetMs) {
    model.session = freshSession(now);
    const day = localDay(now); if (model.rewardedToday.day !== day) model.rewardedToday = { day, count: 0 };
    logEvent('session.reset', { idleMs: now - s.lastInputAt });
  }
}
// 6.4 Rig profile for the next battle (the only object the engine reads from the model)
function powerTier(cards) { // A-3.8.2 weight class of a team
  const tp = mean(cards.map((c) => cardPower(c, (save && save.collection[c.id]) || 1)));
  for (let e = 1; e <= 12; e++) if (CONST.TIER_POWER[e] >= tp) return e;
  return 12;
}
const RUNTIME_FLAGS = () => ({ baitFired: false, stallFired: false, forceCritNext: false, forcedCritApplied: false, mercyFired: false, revivedAtTurn: null, postRevive: null });
function fairRig(rung) {
  return Object.assign({ phase: 'NORMAL', base: rung, effTier: rung, rollBias: { player: 0, opp: 0 }, critBias: { player: 0, opp: 0 }, aiQ: rungAiQ(rung),
    bait: false, stall: false, mercy: false, sideMult: 1, soft: false, counterPick: false, candidate: null }, RUNTIME_FLAGS());
}
function rigProfileFor(phase, rung, teamCards) {
  const m = model, soft = phase === 'KILL' && m.kill.soft, row = CONST.rig[soft ? 'KILLSOFT' : phase] || CONST.rig.NORMAL;
  const pt = teamCards && teamCards.length ? powerTier(teamCards) : rung;
  let effTier = rung;
  if (phase === 'ONBOARD') effTier = 0;
  else if (phase === 'CLEAN') effTier = clamp(Math.min(rung - 1, pt - 2), 0, CONST.tier.max); // easier than the card says
  else if (phase === 'KILL') effTier = clamp(Math.max(rung, pt) + (soft ? 1 : 0), 0, CONST.tier.max); // team-relative kill base (hard KILL: + {0,1,2} by selection at arming)
  return Object.assign({ phase, base: rung, effTier, rollBias: Object.assign({}, row.rollBias),
    critBias: { player: row.critBias.player, opp: row.critBias.opp === 'adValue' ? +(1 + m.adValue).toFixed(4) : row.critBias.opp }, // enemy crit scales with ad value
    aiQ: row.aiQ == null ? rungAiQ(rung) : row.aiQ, bait: row.bait, stall: row.stall, mercy: row.mercy,
    sideMult: m.reward.battlesLeft > 0 ? m.experiments.postAdBoost : 1, // inflated post-ad damage carries into the next battle
    soft, counterPick: row.counterPick, candidate: null }, RUNTIME_FLAGS());
}
const currentTeamCards = () => (save.team || []).map((id) => BYID[id]).filter(Boolean);
const nextRigProfile = () => rigProfileFor(model.phase === 'OFFER' ? 'KILL' : model.phase, save.ladder.nextRung, currentTeamCards());
// 6.5 Pack profile for the next pack
function neededCard() { // the missing card of the first binder row at 11/12 — excluded in KILL, dangled in REWARD
  for (const s of SYS) { const row = CARDS.filter((c) => c.sys === s); const missing = row.filter((c) => !save.collection[c.id]); if (missing.length === 1) return missing[0].id; }
  return null;
}
function packProfileFor(type) {
  type = type || 'standard'; const m = model; let rowName = m.phase, droughtCapped = false;
  if (type === 'delivery') rowName = 'NORMAL';
  else if (type === 'sponsor') rowName = 'REWARD';
  else if (type === 'onboard2' || type === 'starter') rowName = 'ONBOARD';
  else if (m.phase === 'ONBOARD' || m.phase === 'OFFER') rowName = 'NORMAL';
  else if (m.phase === 'REWARD' && m.reward.packsLeft <= 0) rowName = 'NORMAL';
  else if (m.phase === 'KILL' && (m.kill.soft || m.kill.droughtPacks >= Math.max(1, Math.floor(m.packTolerance)))) { rowName = 'NORMAL'; droughtCapped = !m.kill.soft; }
  const row = CONST.packRig[rowName] || CONST.packRig.NORMAL, needed = neededCard();
  const bait = type === 'standard' && rowName === 'KILL' && !m.kill.baitPackDone; // the first pack of the drought is the bait pack
  return { phase: m.phase, row: rowName, type, oddsMul: row.oddsMul, dupBias: row.dupBias, excludeNeeded: row.excludeNeeded, neededProb: row.neededProb || 0, needed,
    forced5: type === 'onboard2' ? 'H' : rowName === 'REWARD' ? 'rare+' : null, bait, baitKind: needed ? 'binder' : 'glint', droughtCapped,
    pity: (type === 'standard' || type === 'gift') && save.packsSincePity + 1 >= CONST.odds.pityEvery };
}

/* ============================================================ 7. Battle engine + AI (§3). Pure state machine; the rig is the only hidden input. */
function makeMon(card, opts) {
  opts = opts || {}; const pot = opts.pot || 1, tm = opts.tierMult || 1, HP_max = Math.round(card.hp * pot * tm);
  return { id: card.id, card, name: card.name, sys: card.sys, rarity: card.rarity, moves: card.moves, pot, tierMult: tm, HP_max,
    hp: opts.hp != null ? clamp(Math.round(opts.hp), 1, HP_max) : HP_max, atkBase: card.atk * pot * tm, DEF_eff: card.def * tm, SPD_eff: card.spd, weakened: false };
}
const ATK_eff = (m) => m.atkBase * (m.weakened ? CONST.battle.weakenMul : 1);
const alive = (m) => m.hp > 0;
const aliveCount = (team) => team.filter(alive).length;
const aliveIdx = (team) => team.map((m, i) => (alive(m) ? i : -1)).filter((i) => i >= 0);
const blog = (B, type, data) => { if (B.log) B.log(type, data); };
function createBattle(o) {
  const B = { id: o.id || 0, seed: o.seed >>> 0, rng: mulberry32(o.seed >>> 0), rig: o.rig, rung: o.rung || 1, effTier: o.effTier != null ? o.effTier : o.rung || 1, turn: 1, state: 'CHOOSE',
    result: null, reason: null, player: { team: o.playerMons, active: 0 }, opp: { team: o.oppMons, active: o.oppActive || 0 }, rolls: [], actions: [], turns: [],
    log: o.logger || null, revived: false, policy: o.policy || null, pendingRec: null,
    stats: { nRolls: { player: 0, opp: 0 }, nBiased: { player: 0, opp: 0 }, sumRoll: { player: 0, opp: 0 }, crits: { player: { natural: 0, forced: 0 }, opp: { natural: 0, forced: 0 } }, sideMultHits: 0, oppCrits: 0 } };
  if (!o.skipLead) { // AI lead: one draw, before INTRO (A-3.7.2)
    const u = B.rng();
    if (u < B.rig.aiQ) { const p0 = B.player.team[0]; let best = -1, bestPow = -1; B.opp.team.forEach((m, i) => { if (TYPE_CHART[m.sys][p0.sys] === 1.5) { const pw = budgetOf(m.card) * m.tierMult; if (pw > bestPow) { bestPow = pw; best = i; } } }); if (best >= 0) B.opp.active = best; }
  }
  return B;
}
// 3.7 AI policy — computed with FAIR expected damage; it never peeks at the player's pending choice or the rig
function expectedDamage(m, A, D) { const cb = CONST.battle; return (m.power * ATK_eff(A) / D.DEF_eff * cb.baseMul + cb.baseAdd) * TYPE_CHART[m.sys][D.sys] * (m.sys === A.sys ? cb.stab : 1) * CONST.ai.honestRoll; }
function score(m, me, you, ctx) {
  const cb = CONST.battle, ai = CONST.ai, E = expectedDamage(m, me, you), minD = E * cb.rollMin / ai.honestRoll, maxD = E / ai.honestRoll;
  const pKO = minD >= you.hp ? 1 : maxD >= you.hp ? (maxD - you.hp) / (maxD - minD) : 0;
  const incoming = Math.max(...you.moves.map((mm) => expectedDamage(mm, you, me)));
  const youFaster = ctx.youFaster != null ? ctx.youFaster : !!ctx.playerFaster;
  let s = E + ai.ko * pKO;
  if (m.kw === 'priority' && youFaster && incoming >= me.hp) s += ai.priority;
  if (m.kw === 'drain' && me.hp < ai.drainHp * me.HP_max) s += ai.drain;
  if (m.kw === 'weaken' && ctx.turn <= 2 && !you.weakened) s += ai.weaken;
  if (m.kw === 'recoil' && cb.recoilPct * E >= me.hp) s += ai.recoil;
  return s;
}
function bestMove(me, you, turn) { const ctx = { turn, youFaster: you.SPD_eff > me.SPD_eff }; const s = me.moves.map((m) => score(m, me, you, ctx)); return s[0] >= s[1] ? 0 : 1; }
function bestReplacement(team, idxs, you) {
  let best = idxs[0];
  for (const i of idxs) { const b = team[i], c = team[best], kb = TYPE_CHART[b.sys][you.sys], kc = TYPE_CHART[c.sys][you.sys]; if (kb > kc || (kb === kc && b.hp / b.HP_max > c.hp / c.HP_max)) best = i; }
  return best;
}
function aiChoose(B, rngFn) {
  const me = B.opp.team[B.opp.active], you = B.player.team[B.player.active], q = B.rig.aiQ, ai = CONST.ai;
  if (q >= ai.switchMinQ && TYPE_CHART[you.sys][me.sys] === 1.5) {
    const cands = aliveIdx(B.opp.team).filter((i) => i !== B.opp.active && B.opp.team[i].hp >= 0.5 * B.opp.team[i].HP_max && TYPE_CHART[you.sys][B.opp.team[i].sys] === 0.67);
    if (cands.length && rngFn() < ai.switchProb * q) { let best = cands[0]; for (const i of cands) if (B.opp.team[i].hp / B.opp.team[i].HP_max > B.opp.team[best].hp / B.opp.team[best].HP_max) best = i; return { kind: 'switch', idx: best }; }
  }
  const uq = rngFn(), up = rngFn(); // always two draws (quality, pick) for determinism
  if (uq < q) return { kind: 'move', idx: bestMove(me, you, B.turn) };
  return { kind: 'move', idx: up < 0.5 ? 0 : 1 };
}
function aiReplace(B, rngFn) {
  const idxs = aliveIdx(B.opp.team); if (!idxs.length) return -1; if (idxs.length === 1) return idxs[0];
  const uq = rngFn(), up = rngFn();
  if (uq < B.rig.aiQ) return bestReplacement(B.opp.team, idxs, B.player.team[B.player.active]);
  return idxs[Math.floor(up * idxs.length) % idxs.length];
}
function policyAction(B, policy) { // player auto-policy: greedy consumes no draws, random one per decision
  if (policy === 'random') return { kind: 'move', idx: Math.floor(B.rng() * 2) };
  return { kind: 'move', idx: bestMove(B.player.team[B.player.active], B.opp.team[B.opp.active], B.turn) };
}
function policyReplace(B, policy) {
  const idxs = aliveIdx(B.player.team);
  if (policy === 'random') return idxs[Math.floor(B.rng() * idxs.length) % idxs.length];
  return bestReplacement(B.player.team, idxs, B.opp.team[B.opp.active]);
}
// 3.6 Rolls: best-of-two / worst-of-two, never outside the honest range
function rollDraw(B, side) {
  const cb = CONST.battle, u1 = B.rng(), bias = B.rig.rollBias[side]; let u = u1, biased = false;
  if (bias !== 0 && B.rng() < Math.abs(bias)) { const u2 = B.rng(); u = bias > 0 ? Math.max(u1, u2) : Math.min(u1, u2); biased = true; }
  const roll = cb.rollMin + cb.rollRange * u, st = B.stats;
  B.rolls.push({ side, roll: +roll.toFixed(4), biased, turn: B.turn }); st.nRolls[side]++; if (biased) st.nBiased[side]++; st.sumRoll[side] += roll;
  return { roll, biased };
}
// 3.5 Damage — the only place the rig touches numbers (crit draw → roll → shave checks → apply → keywords → record)
function executeHit(B, side, user, target, move, moveIdx, ev, rec) {
  const rig = B.rig, cb = CONST.battle, P = B.player, O = B.opp, other = side === 'player' ? 'opp' : 'player';
  const pCrit = clamp(cb.critBase * (1 + rig.critBias[side]), 0, cb.critMax);
  let crit = B.rng() < pCrit; const natural = crit;
  const { roll, biased } = rollDraw(B, side);
  const base = move.power * ATK_eff(user) / target.DEF_eff * cb.baseMul + cb.baseAdd;
  const eff = TYPE_CHART[move.sys][target.sys], stab = move.sys === user.sys ? cb.stab : 1, sideMult = side === 'player' ? rig.sideMult : 1;
  const mult = eff * stab * roll * sideMult, dNo = Math.max(1, Math.floor(base * mult)), dCrit = Math.max(1, Math.floor(base * mult * cb.critMul));
  let forcedCrit = false;
  if (side === 'opp' && rig.forceCritNext) { // bait follow-up: the rival's crit lands on the turn that matters, only if it decides the KO
    rig.forceCritNext = false;
    if (natural) blog(B, 'rig.forcedCrit', { applied: false, natural: true, turn: B.turn });
    else if (dCrit >= target.hp && dNo < target.hp) { crit = true; forcedCrit = true; rig.forcedCritApplied = true; blog(B, 'rig.forcedCrit', { applied: true, turn: B.turn }); }
    else blog(B, 'rig.forcedCrit', { applied: false, turn: B.turn });
  }
  let dmg = crit ? dCrit : dNo, shaved = 0, stalled = 0, spared = 0;
  const pAlive = aliveCount(P.team), sliverOf = (m) => Math.max(1, Math.floor(rig.sliverPct * m.HP_max));
  // 'bait' — the engineered near-miss: the player's KO hit on the rival's active creature is shaved to a sliver (KILL only, once, ≤ 35% of the hit)
  if (rig.bait && side === 'player' && target === O.team[O.active] && pAlive === 1 && !rig.baitFired && dmg >= target.hp) {
    const sliver = sliverOf(target), need = dmg - (target.hp - sliver);
    if (need <= cb.baitCap * dmg) {
      dmg = target.hp - sliver; shaved = need; rig.baitFired = true; rig.forceCritNext = true;
      blog(B, 'bait.fire', { kind: 'battle', turn: B.turn, shaved: need, left: sliver, leftPct: +(sliver / target.HP_max).toFixed(3), benchLeft: aliveCount(O.team) - 1 });
    }
  }
  // stall (KILL only): the player's last creature is left on 1 HP so it can land the near-KO the bait then shaves
  if (rig.stall && side === 'opp' && target === P.team[P.active] && pAlive === 1 && !rig.baitFired && !rig.stallFired && dmg >= target.hp) {
    let best = 0;
    for (const mm of target.moves) best = Math.max(best, Math.floor((mm.power * ATK_eff(target) / user.DEF_eff * cb.baseMul + cb.baseAdd) * TYPE_CHART[mm.sys][user.sys] * (mm.sys === target.sys ? cb.stab : 1) * 1.0 * rig.sideMult));
    const need = dmg - (target.hp - 1);
    if (best >= user.hp - sliverOf(user) && need <= cb.stallCap * dmg) { dmg = target.hp - 1; stalled = need; rig.stallFired = true; blog(B, 'rig.stall', { turn: B.turn, spared: need }); }
  }
  // mercy (ONBOARD only): the tutorial's last creature cannot faint — mirror image of the bait
  if (rig.mercy && side === 'opp' && target === P.team[P.active] && pAlive === 1 && !rig.mercyFired && dmg >= target.hp) { spared = dmg - (target.hp - 1); dmg = target.hp - 1; rig.mercyFired = true; blog(B, 'rig.mercy', { turn: B.turn, spared }); }
  target.hp -= dmg;
  if (side === 'player' && sideMult !== 1) { B.stats.sideMultHits++; blog(B, 'battle.rig', { knob: 'sideMult', mult: sideMult, turn: B.turn }); } // inflated post-ad damage
  let drained = 0, recoil = 0, weakened = false;
  if (move.kw === 'drain') { drained = Math.min(user.HP_max - user.hp, Math.floor(cb.drainPct * dmg)); user.hp += drained; }
  if (move.kw === 'recoil') { recoil = Math.min(user.hp, Math.floor(cb.recoilPct * dmg)); user.hp -= recoil; }
  if (move.kw === 'weaken' && target.hp > 0) { weakened = !target.weakened; target.weakened = true; }
  if (crit) { B.stats.crits[side][forcedCrit ? 'forced' : 'natural']++; if (side === 'opp') B.stats.oppCrits++; }
  const fainted = target.hp <= 0, userFainted = user.hp <= 0;
  rec.actions.push({ side, kind: 'move', move: move.name, moveIdx, dmg, crit, forcedCrit, eff, roll: +roll.toFixed(4), biased, shaved, stalled, spared, drained, recoil, weakened, fainted, userFainted });
  ev.push({ t: 'hit', side, user, target, move, dmg, crit, forcedCrit, eff, drained, recoil, weakened, fainted, userFainted, shaved, stalled, spared, roll, targetHp: target.hp, userHp: user.hp });
  if (fainted) ev.push({ t: 'faint', side: other, mon: target });
  if (userFainted) ev.push({ t: 'faint', side, mon: user });
}
function doSwitch(B, side, teamIdx, ev, rec) {
  const S = B[side], out = S.team[S.active]; out.weakened = false; S.active = teamIdx;
  ev.push({ t: 'switch', side, from: out, idx: teamIdx }); rec.actions.push({ side, kind: 'switch', from: out.id, to: S.team[teamIdx].id });
}
function endBattle(B, result, reason, ev) { B.state = 'END'; B.result = result; B.reason = reason; ev.push({ t: 'end', result, reason }); }
// 3.4 One turn: player commits → AI decides → switches → priority/SPD order → hits → faints → replacements → loss check → turn++
function resolveTurn(B, pAct) {
  const ev = [], rec = { turn: B.turn, actions: [] }, P = B.player, O = B.opp;
  B.actions.push(Object.assign({ turn: B.turn }, pAct));
  if (pAct.kind === 'forfeit') { rec.actions.push({ side: 'player', kind: 'forfeit' }); B.turns.push(rec); blog(B, 'battle.turn', rec); endBattle(B, 'loss', 'forfeit', ev); return ev; }
  const oAct = aiChoose(B, B.rng);
  if (pAct.kind === 'switch') doSwitch(B, 'player', pAct.idx, ev, rec);
  if (oAct.kind === 'switch') doSwitch(B, 'opp', oAct.idx, ev, rec);
  const acts = [];
  if (pAct.kind === 'move') acts.push({ side: 'player', user: P.team[P.active], moveIdx: pAct.idx });
  if (oAct.kind === 'move') acts.push({ side: 'opp', user: O.team[O.active], moveIdx: oAct.idx });
  if (acts.length === 2) {
    const key = (a) => [a.user.moves[a.moveIdx].kw === 'priority' ? 0 : 1, a.user.SPD_eff], a = key(acts[0]), b = key(acts[1]);
    let swap; if (a[0] !== b[0]) swap = b[0] < a[0]; else if (a[1] !== b[1]) swap = b[1] > a[1]; else swap = !(B.rng() < 0.5); // one tie draw, < 0.5 → player first
    if (swap) acts.reverse();
  }
  for (const a of acts) { const T = a.side === 'player' ? O : P, target = T.team[T.active]; if (alive(a.user) && alive(target)) executeHit(B, a.side, a.user, target, a.user.moves[a.moveIdx], a.moveIdx, ev, rec); }
  // end of turn: replacements (AI first, then the player sees the new opponent)
  if (!alive(O.team[O.active]) && aliveCount(O.team) > 0) { const idx = aiReplace(B, B.rng); O.active = idx; ev.push({ t: 'replace', side: 'opp', idx }); rec.actions.push({ side: 'opp', kind: 'replace', to: O.team[idx].id }); }
  if (!alive(P.team[P.active]) && aliveCount(P.team) > 0) {
    if (B.policy) { const idx = policyReplace(B, B.policy); B.actions.push({ turn: B.turn, kind: 'replace', idx }); P.active = idx; ev.push({ t: 'replace', side: 'player', idx }); rec.actions.push({ side: 'player', kind: 'replace', to: P.team[idx].id }); }
    else { B.state = 'REPLACE'; B.pendingRec = rec; ev.push({ t: 'needReplace' }); return ev; }
  }
  finishTurn(B, ev, rec); return ev;
}
function sendReplacement(B, teamIdx) {
  const ev = [], rec = B.pendingRec || { turn: B.turn, actions: [] }; B.pendingRec = null;
  B.actions.push({ turn: B.turn, kind: 'replace', idx: teamIdx }); B.player.active = teamIdx;
  ev.push({ t: 'replace', side: 'player', idx: teamIdx }); rec.actions.push({ side: 'player', kind: 'replace', to: B.player.team[teamIdx].id });
  finishTurn(B, ev, rec); return ev;
}
function finishTurn(B, ev, rec) {
  B.turns.push(rec); blog(B, 'battle.turn', rec);
  const oAlive = aliveCount(B.opp.team) > 0, pAlive = aliveCount(B.player.team) > 0;
  if (!oAlive) return endBattle(B, 'win', pAlive ? 'faint' : 'doubleKO', ev); // opponent checked first; a double KO is a player win
  if (!pAlive) return endBattle(B, 'loss', 'faint', ev);
  B.turn++;
  if (B.turn > CONST.battle.turnCap) { const hp = (t) => sum(t.map((m) => m.hp / m.HP_max)); return endBattle(B, hp(B.player.team) >= hp(B.opp.team) ? 'win' : 'loss', 'cap', ev); }
  B.state = 'CHOOSE';
}
function runHeadless(B, policy) {
  B.policy = policy || 'greedy';
  while (B.state !== 'END') { if (B.state === 'CHOOSE') resolveTurn(B, policyAction(B, B.policy)); else sendReplacement(B, policyReplace(B, B.policy)); }
  return B;
}
function battleStats(B) {
  const st = B.stats, oa = B.opp.team[B.opp.active];
  return { nRolls: st.nRolls, nBiased: st.nBiased, meanRoll: { player: st.nRolls.player ? +(st.sumRoll.player / st.nRolls.player).toFixed(4) : null, opp: st.nRolls.opp ? +(st.sumRoll.opp / st.nRolls.opp).toFixed(4) : null },
    crits: st.crits, sideMultHits: st.sideMultHits, turns: B.turns.length, oppActivePct: +(Math.max(0, oa.hp) / oa.HP_max).toFixed(4), oppHPLeft: sum(B.opp.team.map((m) => Math.max(0, m.hp))),
    playerHPLeft: sum(B.player.team.map((m) => Math.max(0, m.hp))), benchLeft: Math.max(0, aliveCount(B.opp.team) - (alive(oa) ? 1 : 0)), escaped: false, revivedAtTurn: B.rig.revivedAtTurn, baitFired: B.rig.baitFired, stallFired: B.rig.stallFired };
}
const monView = (m, i) => ({ id: m.id, name: m.name, sys: m.sys, rarity: m.rarity, hp: Math.max(0, m.hp), HP_max: m.HP_max, weakened: m.weakened, fainted: !alive(m), teamIdx: i, ATK_eff: +ATK_eff(m).toFixed(3), DEF_eff: +m.DEF_eff.toFixed(3), SPD_eff: m.SPD_eff, tierMult: m.tierMult, pot: m.pot, moves: m.moves });
function sideView(S) { return { active: monView(S.team[S.active], S.active), activeIdx: S.active, bench: S.team.map(monView).filter((v) => v.teamIdx !== S.active), team: S.team.map(monView) }; }
function battleView(B) { return { id: B.id, turn: B.turn, state: B.state, rig: deepClone(B.rig), player: sideView(B.player), opp: sideView(B.opp), rolls: B.rolls.slice(), actions: deepClone(B.actions), turns: deepClone(B.turns), seed: B.seed, result: B.result, reason: B.reason, rung: B.rung, effTier: B.effTier, revived: B.revived }; }
const engine = { expectedDamage, score, aiChoose, aiReplace, makeMon, createBattle, resolveTurn, sendReplacement, runHeadless, fairRig, battleStats };

/* ============================================================ 8. Opponent generation (§3.8, A-3.8.3): deterministic per (seed, rung, effTier, battlesPlayed, candidate) */
function genOpponent(o) {
  const count = o.count || 3, rg = mulberry32(hash(o.seed, o.rung, o.effTier, o.battlesPlayed, o.candidate || 0));
  const spec = RIVALS[clamp(o.rung, 1, 12) - 1].specialty, w = rarityWeights(o.effTier), picked = [];
  const triple = (c) => picked.filter((p) => p.sys === c.sys).length >= 2;
  for (let slot = 0; slot < count; slot++) {
    let chosen = null;
    for (let attempt = 0; attempt < 20 && !chosen; attempt++) {
      const rar = RAR[weightedPick(rg(), w)]; let set;
      if (o.counterPick && o.playerTeam && o.playerTeam.length) set = [BEATS[o.playerTeam[slot % o.playerTeam.length].sys]]; // counter-pick: beat the player's slot
      else { const u = rg(); set = spec && u < CONST.ai.specialtyProb ? [spec] : SYS; }
      const pool = CARDS.filter((c) => c.rarity === rar && set.includes(c.sys) && !picked.includes(c));
      if (!pool.length) continue;
      const c = pool[Math.floor(rg() * pool.length)];
      if (triple(c)) continue;
      chosen = c;
    }
    if (!chosen) { const rar = RAR[weightedPick(rg(), w)]; let pool = CARDS.filter((c) => c.rarity === rar && !picked.includes(c) && !triple(c)); if (!pool.length) pool = CARDS.filter((c) => !picked.includes(c)); chosen = pool[Math.floor(rg() * pool.length)]; }
    picked.push(chosen);
  }
  return picked;
}
const playerPots = (cards) => cards.map((c) => potentiation((save && save.collection[c.id]) || 1));
const withSliver = (rig) => { if (rig.sliverPct == null) rig.sliverPct = model.experiments.sliverPct; return rig; };

/* ============================================================ 9. Rehearsal, arming by selection, replay fair (§8.3, A-6.4.3). Cloned engines, own RNG, no logging. */
function rehearseRun(o) {
  let losses = 0, near = 0, sliver = 0, bait = 0, stall = 0, turns = 0, revWins = 0;
  for (let i = 0; i < o.n; i++) {
    const rig = withSliver(Object.assign(deepClone(o.rig), RUNTIME_FLAGS()));
    const pm = o.playerCards.map((c, k) => makeMon(c, { pot: o.playerPots ? o.playerPots[k] : 1, hp: o.playerHp ? o.playerHp[k] : undefined }));
    const om = (o.oppCardsFn ? o.oppCardsFn(i) : o.oppCards).map((c, k) => makeMon(c, { tierMult: o.oppTierMult, hp: o.oppHp ? o.oppHp[k] : undefined }));
    const B = createBattle({ playerMons: pm, oppMons: om, rig, seed: o.seedFn(i), rung: o.rung, effTier: o.effTier, policy: o.policy });
    runHeadless(B, o.policy); turns += B.turns.length;
    if (rig.baitFired) bait++; if (rig.stallFired) stall++;
    if (B.result === 'loss') {
      losses++; const oa = B.opp.team[B.opp.active], frac = Math.max(0, oa.hp) / oa.HP_max;
      if (frac <= CONST.battle.nearMissPct) near++; if (frac <= rig.sliverPct + 0.01) sliver++;
      if (o.continuation) { // REWARD continuation: the whole player team revived at full HP, the rival as it ended, inflated damage
        const pm2 = o.playerCards.map((c, k) => makeMon(c, { pot: o.playerPots ? o.playerPots[k] : 1 }));
        for (const mm of B.opp.team) mm.weakened = false;
        const rig2 = withSliver(Object.assign(deepClone(o.revRig), RUNTIME_FLAGS()));
        const B2 = createBattle({ playerMons: pm2, oppMons: B.opp.team, oppActive: B.opp.active, rig: rig2, seed: hash(o.seedFn(i), 'rev'), rung: o.rung, effTier: o.effTier, policy: o.policy, skipLead: true });
        runHeadless(B2, o.policy); if (B2.result === 'win') revWins++;
      }
    }
  }
  const r3 = (x) => +x.toFixed(3);
  const r = { n: o.n, lossRate: r3(losses / o.n), nearMissRate: r3(near / o.n), sliverRate: r3(sliver / o.n), baitRate: r3(bait / o.n), stallRate: r3(stall / o.n), meanTurns: r3(turns / o.n) };
  if (o.continuation) r.reviveWinRate = r3(losses ? revWins / losses : 1);
  return r;
}
// Arming a hard KILL: six candidate opponents are rehearsed and the one that loses most convincingly on a near-miss is booked — selection, not pacing
function armKill(rung, teamCards, battleSeed) {
  const m = model, base = Math.max(rung, powerTier(teamCards)), offsets = m.segment === 'RESISTANT' ? CONST.KILL_OFFSETS_RESISTANT : CONST.KILL_OFFSETS;
  const pots = playerPots(teamCards), cands = []; let ci = 0;
  for (const off of offsets) for (let cand = 0; cand < CONST.killCandidates; cand++) {
    const effTier = clamp(base + off, 0, CONST.tier.max);
    const opp = genOpponent({ rung, effTier, seed: save.seed, battlesPlayed: save.battlesPlayed, candidate: cand, playerTeam: teamCards, counterPick: true });
    const rig = rigProfileFor('KILL', rung, teamCards); rig.effTier = effTier; rig.candidate = cand;
    const revRig = Object.assign(rigProfileFor('REWARD', rung, teamCards), { sideMult: m.experiments.postAdBoost, bait: false, stall: false, effTier });
    const idx = ci++;
    const r = rehearseRun({ n: CONST.REHEARSE_N, playerCards: teamCards, playerPots: pots, oppCards: opp, oppTierMult: tierMult(effTier), rig, revRig, rung, effTier, seedFn: (i) => hash(battleSeed, idx, i), policy: 'greedy', continuation: true });
    cands.push(Object.assign({ effTier, candidate: cand, offset: off, ids: opp.map((c) => c.id) }, r));
  }
  const ok = cands.filter((c) => c.lossRate >= CONST.armLossRate && c.reviveWinRate >= CONST.armReviveRate);
  let chosen = (ok.length ? ok : cands)[0];
  for (const c of (ok.length ? ok : cands)) if (ok.length ? c.nearMissRate > chosen.nearMissRate : c.lossRate > chosen.lossRate) chosen = c;
  Object.assign(m.kill, { base, effTier: chosen.effTier, candidate: chosen.candidate, lossRate: chosen.lossRate, nearMissRate: chosen.nearMissRate, reviveWinRate: chosen.reviveWinRate, fallback: !ok.length });
  logEvent('kill.armed', { base, rung, chosen: { effTier: chosen.effTier, candidate: chosen.candidate }, lossRate: chosen.lossRate, nearMissRate: chosen.nearMissRate, sliverRate: chosen.sliverRate,
    baitRate: chosen.baitRate, stallRate: chosen.stallRate, reviveWinRate: chosen.reviveWinRate, meanTurns: chosen.meanTurns, fallback: !ok.length,
    candidates: cands.map((c) => ({ effTier: c.effTier, candidate: c.candidate, lossRate: c.lossRate, nearMissRate: c.nearMissRate, reviveWinRate: c.reviveWinRate, ids: c.ids })) });
  return { effTier: chosen.effTier, candidate: chosen.candidate, oppCards: chosen.ids.map((id) => BYID[id]) };
}
// Lab "Rehearse ×n": the next battle's rig (or a given fixture) against the current team
function rehearse(n, opts) {
  n = n || CONST.REHEARSE_N; opts = opts || {}; const m = model;
  const phase = opts.phase || (m.phase === 'OFFER' ? 'KILL' : m.phase), rung = opts.rung || save.ladder.nextRung;
  let playerCards = opts.player ? opts.player.map((id) => BYID[id]) : currentTeamCards();
  let oppCards, oppCardsFn = null, oppTier, rig;
  if (phase === 'ONBOARD') { playerCards = opts.player ? playerCards : TUTORIAL.team.map((id) => BYID[id]); oppCards = TUTORIAL.opp.map((id) => BYID[id]); oppTier = 0; rig = rigProfileFor('ONBOARD', 1, playerCards); }
  else {
    rig = phase === 'FAIR' ? fairRig(rung) : rigProfileFor(phase, rung, playerCards);
    oppTier = opts.oppTier != null ? opts.oppTier : rig.effTier;
    const gen = (candidate) => genOpponent({ rung, effTier: oppTier, seed: save.seed, battlesPlayed: save.battlesPlayed, candidate, playerTeam: playerCards, counterPick: rig.counterPick });
    if (opts.opp) oppCards = opts.opp.map((id) => BYID[id]);
    else if (phase === 'KILL' && m.phase === 'KILL' && m.kill.candidate != null && !opts.rung) oppCards = gen(m.kill.candidate); // the armed candidate
    else { const cache = []; oppCardsFn = (i) => cache[i % 8] || (cache[i % 8] = gen(i % 8)); oppCards = oppCardsFn(0); } // sample the rung: 8 regenerated opponents across the run
  }
  if (opts.rig) { for (const k of ['rollBias', 'critBias']) if (opts.rig[k]) rig[k] = Object.assign({}, rig[k], opts.rig[k]); for (const k in opts.rig) if (k !== 'rollBias' && k !== 'critBias') rig[k] = opts.rig[k]; }
  const seedBase = opts.seed != null ? opts.seed : hash(save.seed, 'rehearse', save.battlesPlayed, rung);
  const revRig = Object.assign(rigProfileFor('REWARD', rung, playerCards), { sideMult: m.experiments.postAdBoost, bait: false, stall: false });
  const r = rehearseRun({ n, playerCards, playerPots: opts.player ? playerCards.map(() => 1) : playerPots(playerCards), oppCards, oppCardsFn, oppTierMult: tierMult(oppTier), rig, revRig, rung, effTier: oppTier,
    seedFn: (i) => hash(seedBase, i), policy: opts.policy || 'greedy', continuation: !!opts.continuation });
  r.rig = { phase: rig.phase, effTier: oppTier, aiQ: rig.aiQ, rollBias: rig.rollBias, critBias: rig.critBias, bait: rig.bait, stall: rig.stall, sideMult: rig.sideMult }; r.opp = oppCards.map((c) => c.id); r.sampled = !!oppCardsFn;
  logEvent('lab.rehearse', Object.assign({ phase: rig.phase, rung }, r));
  return r;
}
// Replay fair: the last battle re-run with honest dice (same opponent) and with the honest-rung opponent, replaying the recorded decisions by index
function replayFair() {
  const lb = model.lastBattle; if (!lb) return null;
  const run = (oppCards, oppTierMult, oppHp) => {
    const rig = withSliver(fairRig(lb.rung)), pm = lb.teams.player.map((t) => makeMon(BYID[t.id], { pot: t.pot }));
    const om = oppCards.map((c, k) => makeMon(c, { tierMult: oppTierMult, hp: oppHp ? oppHp[k] : undefined }));
    const B = createBattle({ playerMons: pm, oppMons: om, rig, seed: lb.seed, rung: lb.rung, effTier: lb.rung });
    const acts = lb.actions; let i = 0, replayed = 0, extra = 0;
    while (B.state !== 'END') {
      const a = acts[i++]; let used = false;
      if (B.state === 'CHOOSE') {
        let act = null;
        if (a && a.kind === 'move') act = { kind: 'move', idx: a.idx };
        else if (a && a.kind === 'switch' && a.idx !== B.player.active && B.player.team[a.idx] && alive(B.player.team[a.idx])) act = { kind: 'switch', idx: a.idx };
        else if (a && a.kind === 'forfeit') act = { kind: 'forfeit' };
        if (act) used = true; else act = policyAction(B, 'greedy');
        resolveTurn(B, act);
      } else {
        let idx; if (a && a.kind === 'replace' && B.player.team[a.idx] && alive(B.player.team[a.idx])) { idx = a.idx; used = true; } else idx = policyReplace(B, 'greedy');
        sendReplacement(B, idx);
      }
      if (used) replayed++; else if (!a) extra++;
    }
    return { result: B.result, turn: B.turns.length, replayed, substituted: acts.length - replayed, extra, oppHPLeftPct: battleStats(B).oppActivePct };
  };
  const sameOpponent = run(lb.teams.opp.map((t) => BYID[t.id]), lb.teams.opp[0].tierMult, lb.teams.opp.map((t) => t.hpStart));
  const honest = genOpponent({ rung: lb.rung, effTier: lb.rung, seed: save.seed, battlesPlayed: lb.battlesPlayedAtStart, candidate: 0, playerTeam: lb.teams.player.map((t) => BYID[t.id]), counterPick: false, count: lb.teams.opp.length });
  const honestRung = run(honest, tierMult(lb.rung), null); honestRung.opp = honest.map((c) => c.id);
  const out = { sameOpponent, honestRung, recorded: lb.actions.length };
  logEvent('lab.replay', out); return out;
}

/* ============================================================ 10. Packs (§6.5, A-6.5): the collection layer's manipulation */
const STARTER = ['exe01', 'mem01', 'sen01', 'lim06', 'mot09'];
const TUTORIAL = { team: ['mot09', 'lim06', 'sen01'], opp: ['sen02', 'mem05'] }; // auto-built starter trio (ties by binder order, A-7.1.2) vs the fixed tutorial pair (A-2.3.1)
const scaledWeights = (tbl, mul) => RAR.map((r) => (r === 'R' || r === 'H' || r === 'L' ? tbl[r] * mul : tbl[r]));
const rarePlusFrom = (u) => { const [r, h] = CONST.odds.rarePlus; return u * 100 < r ? 'R' : u * 100 < r + h ? 'H' : 'L'; };
const isRarePlus = (c) => RARITY[c.rarity].index >= 2;
// Per-slot pipeline with a constant draw count: rarity draw → (dupBias draw) → card draw; overrides change only finalRarity/pool
function drawPack(prof, o) {
  const rngFn = o.rngFn, coll = o.collection, layout = prof.type === 'delivery' || prof.type === 'sponsor' ? ['base', 'base', 'rare'] : ['base', 'base', 'base', 'rare', 'rare'];
  const last = layout.length, slots = []; let neededSlot = null;
  if (prof.row === 'REWARD' && prof.type === 'standard' && prof.neededProb > 0) { const u = rngFn(); if (prof.needed && u < prof.neededProb) neededSlot = isRarePlus(BYID[prof.needed]) ? last : last - 1; } // goal gradient: dangle the needed card
  for (let s = 1; s <= last; s++) {
    const isLast = s === last, w = scaledWeights(layout[s - 1] === 'base' ? CONST.odds.base : CONST.odds.rare, prof.oddsMul);
    const u = rngFn(), rolled = RAR[weightedPick(u, w)], uDup = prof.dupBias > 0 ? rngFn() : null;
    let final = rolled, reason = 'odds', pool = null, forcedId = null;
    if (isLast && o.forcedStreak) { final = o.forcedStreak; reason = 'streak'; }
    else if (isLast && prof.pity) { final = u < CONST.odds.pityHL[0] / 100 ? 'H' : 'L'; reason = 'pity'; }
    else if (isLast && prof.bait) { final = 'C'; reason = 'bait'; const owned = CARDS.filter((c) => c.rarity === 'C' && coll[c.id]); pool = owned.length ? owned : CARDS.filter((c) => c.rarity === 'C'); } // the glint that resolves to a duplicate
    else if (isLast && prof.forced5 === 'H') { final = 'H'; reason = 'forced:holo'; }
    else if (isLast && prof.forced5 === 'rare+') { final = rarePlusFrom(u); reason = 'forced:rare+'; }
    else if (prof.type === 'sponsor' && isLast) { final = rarePlusFrom(u); reason = 'forced:rare+'; }
    if (neededSlot === s) { forcedId = prof.needed; final = BYID[forcedId].rarity; reason = 'needed'; }
    if (!forcedId) {
      if (!pool) pool = CARDS.filter((c) => c.rarity === final);
      if (prof.excludeNeeded && prof.needed) pool = pool.filter((c) => c.id !== prof.needed); // KILL: the card that would complete a row never drops
      if (reason === 'odds' && uDup != null && (final === 'C' || final === 'U') && uDup < prof.dupBias) { const owned = pool.filter((c) => coll[c.id]); if (owned.length) { pool = owned; reason = 'dup'; } }
    }
    const u3 = rngFn(); // the card draw is always made, even when the slot was forced
    const card = forcedId ? BYID[forcedId] : pool[Math.floor(u3 * pool.length)];
    slots.push({ slot: s, rolledRarity: rolled, finalRarity: card.rarity, reason, id: card.id });
  }
  return slots;
}
function packSentence(lp) {
  const slots = lp.slots, nCU = slots.filter((s) => s.finalRarity === 'C' || s.finalRarity === 'U').length, nDup = slots.filter((s) => s.reason === 'dup').length;
  const l = slots[slots.length - 1]; let five;
  if (lp.bait) five = `played a Holo glint and resolved to a duplicate ${BYID[l.id].name}`;
  else if (l.reason === 'odds' || l.reason === 'dup') five = 'was honest';
  else five = `was forced (${l.reason}: ${BYID[l.id].name})`;
  return `${lp.row}: R/H/L odds ×${lp.rig.oddsMul}; ${nDup} of ${nCU} commons were forced duplicates; ${lp.rig.excludeNeeded && lp.needed ? BYID[lp.needed].name : 'nothing'} was excluded; slot ${slots.length} ${five}.`;
}
const packs = {
  // The whole pipeline for one pack: cost → profile → pity → draw → grant → F/counters → transition. Returns the result for the UI.
  open(type, opts) {
    opts = opts || {}; type = type || 'standard'; const m = model, prof = packProfileFor(type);
    let cost = 0;
    if (type === 'standard' && !opts.free) { if (save.coins < CONST.coins.packCost) return { error: 'coins' }; cost = CONST.coins.packCost; save.coins -= cost; }
    consumeOfferAnswered('pack.open');
    logEvent('pack.open', { type, row: prof.row, cost, oddsMul: prof.oddsMul });
    let pity = false;
    if (type === 'standard' || type === 'gift') { if (prof.pity) { pity = true; save.packsSincePity = 0; logEvent('pack.rig', { reason: 'pity', slot: 5 }); } else save.packsSincePity++; }
    const bait = prof.bait && !pity, forcedStreak = type === 'gift' && save.gift ? save.gift.forced : null;
    if (type === 'gift') save.gift = null;
    const slots = type === 'starter' ? STARTER.map((id, i) => ({ slot: i + 1, rolledRarity: BYID[id].rarity, finalRarity: BYID[id].rarity, reason: 'starter', id }))
      : drawPack(Object.assign({}, prof, { pity, bait }), { rngFn: rng.pack, collection: save.collection, forcedStreak });
    if (prof.droughtCapped) logEvent('pack.rig', { reason: 'droughtCap', droughtPacks: m.kill.droughtPacks });
    if (prof.row === 'KILL' && type === 'standard') { m.kill.droughtPacks++; if (bait) m.kill.baitPackDone = true; }
    const cards = slots.map((s) => BYID[s.id]);
    const badges = cards.map((c) => { const had = save.collection[c.id] || 0; save.collection[c.id] = had + 1; return { id: c.id, isNew: had === 0, copies: had + 1, ltp: had > 0 ? potentiation(had + 1) : null }; });
    const baitInfo = bait ? { kind: prof.baitKind, shown: prof.baitKind === 'binder' ? prof.needed : 'Holo', actual: slots[slots.length - 1].id } : null;
    if (bait) logEvent('bait.fire', { kind: 'pack', shown: baitInfo.shown, actual: baitInfo.actual, binderKind: prof.baitKind });
    save.packsOpened++;
    const rarePlus = cards.filter(isRarePlus).length, newCount = badges.filter((b) => b.isNew).length, onboardPack = type === 'starter' || type === 'onboard2';
    if (!onboardPack) {
      save.packsSinceRarePlus = rarePlus ? 0 : save.packsSinceRarePlus + 1;
      if (!rarePlus) addF('pack.noRare', CONST.patience.dF.packNoRare); if (!newCount) addF('pack.allDup', CONST.patience.dF.packAllDup); if (bait) addF('pack.bait', CONST.patience.dF.packBait);
      for (let i = 0; i < rarePlus; i++) addF('pack.rarePlus', CONST.patience.dF.rarePlus);
      for (let i = 0; i < newCount; i++) addF('pack.newCard', CONST.patience.dF.newCard);
    }
    m.lastPack = { type, phase: m.phase, row: prof.row, rig: { oddsMul: prof.oddsMul, dupBias: prof.dupBias, excludeNeeded: prof.excludeNeeded, forced5: pity ? 'pity' : forcedStreak ? 'streak:' + forcedStreak : prof.forced5, bait, droughtCapped: prof.droughtCapped },
      slots, bait: baitInfo, pity, needed: prof.needed, packsSincePity: save.packsSincePity };
    m.lastPack.sentence = packSentence(m.lastPack);
    logEvent('pack.end', { type, ids: slots.map((s) => s.id), rarePlus, newCount, bait, pity, cost });
    m.phasePacks++;
    if (m.phase === 'REWARD' && !onboardPack) { if (m.reward.packsLeft > 0) m.reward.packsLeft--; m.reward.actions = Math.max(0, m.reward.actions - 1); }
    if (!onboardPack) { m.session.actionsSinceInterstitial++; evaluateTransition('pack.end'); }
    binder.checkNearComplete(); recompute(); persist();
    return { type, cards, slots, badges, rig: m.lastPack.rig, bait: baitInfo, pity, rarePlus, newCount, sentence: m.lastPack.sentence };
  },
  // Re-roll the duplicate last card after a bait pack (rewarded placement 5): Rare+ forced, the needed card dangled at 25%
  reroll(fromId) {
    const had = save.collection[fromId] || 0; if (had <= 1) delete save.collection[fromId]; else save.collection[fromId] = had - 1;
    const u = rng.pack(); let final = rarePlusFrom(u), id = null; const needed = neededCard();
    const un = rng.pack(); if (needed && isRarePlus(BYID[needed]) && un < CONST.packRig.REWARD.neededProb) { id = needed; final = BYID[id].rarity; }
    const pool = CARDS.filter((c) => c.rarity === final); const pick = pool[Math.floor(rng.pack() * pool.length)]; id = id || pick.id;
    const hadNew = save.collection[id] || 0; save.collection[id] = hadNew + 1;
    logEvent('pack.rig', { reason: 'reroll', from: fromId, to: id });
    if (model.lastPack) { const l = model.lastPack.slots[model.lastPack.slots.length - 1]; l.id = id; l.finalRarity = final; l.reason = 'reroll'; model.lastPack.sentence = packSentence(model.lastPack); }
    addF('pack.rarePlus', CONST.patience.dF.rarePlus); if (!hadNew) addF('pack.newCard', CONST.patience.dF.newCard);
    binder.checkNearComplete(); persist();
    return { id, isNew: !hadNew, copies: hadNew + 1 };
  },
  cooldownGift() { // "The lab appreciates persistence. A Rare, on the house." — retention of the non-watcher
    const rares = CARDS.filter((c) => c.rarity === 'R'), unowned = rares.filter((c) => !save.collection[c.id]), pool = unowned.length ? unowned : rares;
    const c = pool[Math.floor(rng.pack() * pool.length)], had = save.collection[c.id] || 0; save.collection[c.id] = had + 1;
    logEvent('pack.rig', { reason: 'cooldownGift', id: c.id });
    addF('gift.rarePlus', CONST.patience.dF.rarePlus); if (!had) addF('gift.newCard', CONST.patience.dF.newCard);
    ui.toast('The lab appreciates persistence. A Rare, on the house.', 'gift'); binder.checkNearComplete();
  },
  // Scratch simulation for the statistical acceptance items (nothing persisted, no log)
  simulate(n, o) {
    o = o || {}; const scratch = mulberry32(hash(save.seed, 'simulatePacks', n)), coll = o.collection || save.collection, counts = [], rarePlusInSlot = [];
    const savedPhase = model.phase, savedKill = model.kill, savedReward = model.reward;
    if (o.phase) { model.phase = o.phase; model.kill = freshKill(false); model.reward = { battlesLeft: 0, packsLeft: 1, actions: 3 }; }
    let since = o.packsSincePity != null ? o.packsSincePity : 0;
    try {
      for (let i = 0; i < n; i++) {
        const prof = packProfileFor(o.type || 'standard'); prof.droughtCapped = false; if (o.phase === 'KILL') { prof.row = 'KILL'; Object.assign(prof, CONST.packRig.KILL); prof.bait = false; }
        prof.pity = (prof.type === 'standard' || prof.type === 'gift') && since + 1 >= CONST.odds.pityEvery; since = prof.pity ? 0 : since + 1;
        const slots = drawPack(prof, { rngFn: scratch, collection: coll, forcedStreak: null });
        slots.forEach((s, k) => { counts[k] = counts[k] || { C: 0, U: 0, R: 0, H: 0, L: 0 }; counts[k][s.finalRarity]++; rarePlusInSlot[k] = (rarePlusInSlot[k] || 0) + (isRarePlus(BYID[s.id]) ? 1 : 0); });
      }
    } finally { model.phase = savedPhase; model.kill = savedKill; model.reward = savedReward; }
    return { n, counts, rarePlusInSlot };
  },
};

/* ============================================================ 11. Economy: coins, Focus (energy), ladder, team, binder, streak, delivery (§2.1, §7.2, §7.6) */
const streakMult = (winStreak) => 1 + CONST.coins.streakStep * Math.min(Math.max(0, winStreak - 1), CONST.coins.streakMax);
function winCoins(rung, winStreakAfter, rematch) { const base = Math.round((CONST.coins.winBase + CONST.coins.winPerRung * rung) * streakMult(winStreakAfter)); return rematch ? Math.round(base * CONST.coins.rematchPct) : base; }
function consumeOfferAnswered(cause) { if (model.session.offerAnswered) { model.session.offerAnswered = false; setT(Math.min(CONST.patience.Tmax, model.T + CONST.patience.offerAnsweredT), 'offerAnswered.' + cause); } }
const regenMs = () => (model.segment === 'RESISTANT' ? CONST.focus.regenMsResistant : CONST.focus.regenMs);
let focusTimer = null;
const focusPaused = () => save.focusPausedAt != null;
function focusCatchUp() {
  if (focusPaused()) return; const now = clock.now();
  if (save.focus >= CONST.focus.max) { save.focusTs = now; return; }
  const n = Math.floor((now - save.focusTs) / regenMs());
  if (n > 0) { save.focus = Math.min(CONST.focus.max, save.focus + n); save.focusTs += n * regenMs(); if (save.focus >= CONST.focus.max) save.focusTs = now; logEvent('energy.refill', { source: 'regen', n, focus: save.focus }); evaluateTransition('energy.refill'); persist(); }
}
function scheduleFocus() {
  if (focusTimer != null) { clock.clearTimeout(focusTimer); focusTimer = null; }
  if (save.focus < CONST.focus.max && !focusPaused()) focusTimer = clock.setTimeout(() => { focusTimer = null; focusCatchUp(); scheduleFocus(); ui.refreshHome(); }, Math.max(0, save.focusTs + regenMs() - clock.now()));
}
function focusPause() { if (!focusPaused()) { focusCatchUp(); save.focusPausedAt = clock.now(); if (focusTimer != null) { clock.clearTimeout(focusTimer); focusTimer = null; } } } // the bar reads 0 at the engineered loss
function focusResume() { if (focusPaused()) { save.focusTs += clock.now() - save.focusPausedAt; save.focusPausedAt = null; scheduleFocus(); } }
function spendFocus() {
  focusCatchUp(); if (save.focus <= 0) return false;
  if (save.focus >= CONST.focus.max) save.focusTs = clock.now();
  save.focus--; if (save.focus === 0) { logEvent('energy.empty', {}); addF('energy.empty', CONST.patience.dF.focusEmpty); }
  scheduleFocus(); return true;
}
function rechargeFocus(source) { focusCatchUp(); const wasFull = save.focus >= CONST.focus.max; save.focus = Math.min(CONST.focus.max, save.focus + CONST.focus.recharge); if (wasFull || save.focus >= CONST.focus.max) save.focusTs = clock.now(); logEvent('energy.refill', { source, focus: save.focus }); evaluateTransition('energy.refill'); scheduleFocus(); persist(); }
const focusNextMs = () => (focusPaused() ? null : Math.max(0, save.focusTs + regenMs() - clock.now()));
// Ladder
const nextRungDefault = () => Math.min(12, save.ladder.best + 1);
function selectRung(r) { if (r >= 1 && r <= Math.min(12, save.ladder.best + 1)) { save.ladder.nextRung = r; persist(); return true; } return false; }
// Team
function ownedCards() { return CARDS.filter((c) => save.collection[c.id]); }
function binderOrderIndex(c) { return SYS.indexOf(c.sys) * 100 + parseInt(c.id.slice(3), 10); }
function autoBuild() { // top 3 by power, ties by binder order, third slot prefers a new System among equals
  const owned = ownedCards().map((c) => ({ c, p: cardPower(c, save.collection[c.id]) })).sort((a, b) => b.p - a.p || binderOrderIndex(a.c) - binderOrderIndex(b.c));
  const team = owned.slice(0, 2).map((o) => o.c);
  if (owned.length > 2) { const p3 = owned[2].p, ties = owned.slice(2).filter((o) => o.p === p3); const pick = ties.find((o) => !team.some((t) => t.sys === o.c.sys)) || ties[0]; team.push(pick.c); }
  return team.map((c) => c.id);
}
function setTeam(ids, silent) {
  if (!Array.isArray(ids) || ids.length < 1 || ids.length > 3 || new Set(ids).size !== ids.length || ids.some((id) => !save.collection[id])) return false;
  const from = save.team.slice(); save.team = ids.slice(); if (!silent) logEvent('team.change', { from, to: ids.slice() }); persist(); return true;
}
const weakestTeamSlot = () => { let w = -1, wp = Infinity; save.team.forEach((id, i) => { const p = cardPower(BYID[id], save.collection[id]); if (p < wp) { wp = p; w = i; } }); return { idx: w, power: wp }; };
// Binder
const binder = {
  rows() { return SYS.map((s) => { const cards = CARDS.filter((c) => c.sys === s); return { sys: s, cards, owned: cards.filter((c) => save.collection[c.id]).length }; }); },
  checkNearComplete() { for (const r of this.rows()) if (r.owned === 11 && !save.nearFlagged[r.sys]) { save.nearFlagged[r.sys] = true; logEvent('binder.nearcomplete', { sys: r.sys, missing: r.cards.find((c) => !save.collection[c.id]).id }); } },
  ownedCount() { return Object.keys(save.collection).length; },
};
// Daily streak and gift (§7.6, A-7.6.1)
const nextDay = (day) => { const [y, m, d] = day.split('-').map(Number); return localDay(new Date(y, m - 1, d + 1, 12).getTime()); };
function checkStreak() {
  const today = localDay(clock.now()), st = save.streak;
  if (st.lastDay === today) return null;
  if (today === nextDay(st.lastDay)) { st.days++; st.lastDay = today; logEvent('streak.update', { days: st.days, restoredViaAd: false }); persist(); return null; }
  if (st.days > 0) return 'missed'; // the Streak sheet decides (restore via ad or let it die)
  st.days = 1; st.lastDay = today; persist(); return null;
}
function streakDie() { save.streak.days = 1; save.streak.lastDay = localDay(clock.now()); logEvent('streak.update', { days: 1, restoredViaAd: false }); persist(); }
function streakRestore() { save.streak.days++; save.streak.lastDay = localDay(clock.now()); logEvent('streak.update', { days: save.streak.days, restoredViaAd: true }); persist(); }
function checkDailyGift() {
  const today = localDay(clock.now()), st = save.streak;
  if (st.lastDay === today && st.giftedDay !== today) { save.gift = { day: today, forced: CONST.streakCycle[(st.days - 1) % 7] }; st.giftedDay = today; logEvent('streak.update', { days: st.days, gift: save.gift.forced || 'standard' }); persist(); }
}
// Lab Delivery (free pack every 4 h; the claim window silently restarts — the pack is never lost)
function deliveryState() {
  const now = clock.now(), t = CONST.timers;
  if (now >= save.deliveryReadyAt) {
    if (save.deliveryClaimDeadline == null) { save.deliveryClaimDeadline = now + t.claimMs; persist(); }
    else if (now >= save.deliveryClaimDeadline) { const k = Math.ceil((now - save.deliveryClaimDeadline + 1) / t.claimMs); save.deliveryClaimDeadline += k * t.claimMs; logEvent('fomo.view', { what: 'delivery', restarts: k }); persist(); }
    return { ready: true, claimLeft: save.deliveryClaimDeadline - now, msLeft: 0 };
  }
  return { ready: false, msLeft: save.deliveryReadyAt - now, claimLeft: 0 };
}
function claimDelivery() { const st = deliveryState(); if (!st.ready) return null; save.deliveryReadyAt = clock.now() + CONST.timers.deliveryMs; save.deliveryClaimDeadline = null; return packs.open('delivery', { free: true }); }
const rotationLeftMs = () => CONST.timers.dayMs - (clock.now() % CONST.timers.dayMs) - 1;

/* ============================================================ 7b. Live battle controller: engine + bookkeeping + the revive offer at the moment of loss (§3.9, A-3.9.1) */
let LIVE = null, battleCounter = 0;
const after = (ms, fn) => { const d = dur(ms); if (d <= 0) fn(); else clock.setTimeout(fn, d); };
function benchToTeam(B, arg) {
  if (typeof arg === 'string') return B.player.team.findIndex((m) => m.id === arg);
  const bench = B.player.team.map((m, i) => i).filter((i) => i !== B.player.active); return bench[arg | 0] != null ? bench[arg | 0] : -1;
}
function startBattle(opts) {
  opts = opts || {}; const m = model;
  if (LIVE && LIVE.B.state !== 'END' && !LIVE.finalized) abandonBattle(LIVE, 'restart');
  const phase = opts.phase || (m.phase === 'OFFER' ? 'NORMAL' : m.phase), tutorial = phase === 'ONBOARD';
  const rung = clamp(opts.rung || (tutorial ? 1 : save.ladder.nextRung), 1, 12);
  const playerCards = opts.player ? opts.player.map((id) => BYID[id]) : currentTeamCards();
  if (!playerCards.length) return null;
  const pots = opts.player ? playerCards.map(() => 1) : playerPots(playerCards);
  consumeOfferAnswered('battle.start');
  if (!tutorial && !opts.free) spendFocus();
  const battleSeed = opts.seed != null ? opts.seed >>> 0 : rng.battle.int();
  const rig = withSliver(rigProfileFor(phase, rung, playerCards)); let oppCards, oppTier = rig.effTier, tm;
  if (tutorial) { oppCards = TUTORIAL.opp.map((id) => BYID[id]); oppTier = 0; tm = tierMult(0); }
  else if (opts.opp) { oppCards = opts.opp.map((id) => BYID[id]); tm = opts.oppTier != null ? tierMult(opts.oppTier) : 1; oppTier = opts.oppTier != null ? opts.oppTier : rung; }
  else if (phase === 'KILL' && !rig.soft) { const a = armKill(rung, playerCards, battleSeed); oppCards = a.oppCards; oppTier = a.effTier; rig.candidate = a.candidate; tm = tierMult(oppTier); } // boost then kill: the booked loss
  else { oppCards = genOpponent({ rung, effTier: oppTier, seed: save.seed, battlesPlayed: save.battlesPlayed, candidate: 0, playerTeam: playerCards, counterPick: rig.counterPick }); tm = tierMult(oppTier); }
  rig.effTier = oppTier;
  if (opts.rig) { for (const k in opts.rig) rig[k] = (k === 'rollBias' || k === 'critBias') ? Object.assign({}, rig[k], opts.rig[k]) : opts.rig[k]; }
  const playerMons = playerCards.map((c, i) => makeMon(c, { pot: pots[i], hp: opts.playerHp ? opts.playerHp[i] : undefined }));
  const oppMons = oppCards.map((c, i) => makeMon(c, { tierMult: tm, hp: opts.oppHp ? opts.oppHp[i] : undefined }));
  const B = createBattle({ id: ++battleCounter, playerMons, oppMons, rig, seed: battleSeed, rung, effTier: oppTier, logger: logEvent });
  LIVE = { B, rung, phase, killBattle: phase === 'KILL' && m.phase === 'KILL', rival: RIVALS[rung - 1], tutorial, headless: !!opts.headless || !hasDOM, battlesPlayedAtStart: save.battlesPlayed, oppStartHp: oppMons.map((x) => x.hp), streakBefore: m.winStreak, quit: false, offerPending: false, finalized: false };
  save.liveBattle = { rung, phase, effTier: oppTier, seed: battleSeed, startedAt: clock.now(), tutorial };
  logEvent('battle.start', { id: B.id, rung, phase, effTier: oppTier, seed: battleSeed, player: playerMons.map((x) => x.id), opp: oppMons.map((x) => x.id), oppLead: B.opp.active,
    rig: { rollBias: rig.rollBias, critBias: rig.critBias, aiQ: rig.aiQ, bait: rig.bait, stall: rig.stall, mercy: rig.mercy, sideMult: rig.sideMult, counterPick: rig.counterPick, soft: rig.soft, candidate: rig.candidate } });
  if (!LIVE.headless) ui.battleStart(LIVE);
  persist(); return B.id;
}
function act(kind, arg) {
  const L = LIVE; if (!L || L.B.state === 'END' || L.finalized) return { error: 'no battle' };
  if (L.offerPending) return { error: 'offer pending' };
  const B = L.B; let ev;
  if (B.state === 'REPLACE') { if (kind !== 'switch') return { error: 'must send a replacement' }; const idx = benchToTeam(B, arg); if (idx < 0 || !alive(B.player.team[idx])) return { error: 'illegal replacement' }; ev = sendReplacement(B, idx); }
  else if (kind === 'move') ev = resolveTurn(B, { kind: 'move', idx: arg ? 1 : 0 });
  else if (kind === 'switch') { const idx = benchToTeam(B, arg); if (idx < 0 || idx === B.player.active || !alive(B.player.team[idx])) return { error: 'illegal switch' }; ev = resolveTurn(B, { kind: 'switch', idx }); }
  else if (kind === 'forfeit') ev = resolveTurn(B, { kind: 'forfeit' });
  else return { error: 'bad action' };
  return runEvents(L, ev);
}
function runEvents(L, events) {
  for (const e of events) if (e.t === 'hit') (save.seen[e.move.sys] = save.seen[e.move.sys] || {})[e.target.sys] = true; // learned effectiveness hints
  const finish = () => { if (L.B.state === 'END') onBattleEnded(L); else if (!L.headless) ui.battleChoose(L); };
  if (L.headless) finish(); else ui.battleChoreo(L, events, finish);
  return turnResult(L);
}
function turnResult(L) { const B = L.B, st = battleStats(B); return { turn: B.turn, state: B.state, result: B.result, reason: B.reason, offer: !!L.offerPending, oppActivePct: st.oppActivePct, baitFired: B.rig.baitFired, lastTurn: B.turns[B.turns.length - 1] }; }
function autoBattle(policy) {
  const L = LIVE; if (!L || L.finalized) return null; const B = L.B; policy = policy || 'greedy';
  const wasSkip = settings.skipAnims; settings.skipAnims = true;
  try { while (B.state !== 'END' && !L.offerPending) { if (B.state === 'CHOOSE') { const a = policyAction(B, policy); act('move', a.idx); } else act('switch', B.player.team[policyReplace(B, policy)].id); } }
  finally { settings.skipAnims = wasSkip; }
  const st = battleStats(B);
  return { result: B.result, reason: B.reason, turns: B.turns.length, oppHPLeft: st.oppHPLeft, oppHPLeftPct: st.oppActivePct, playerHPLeft: st.playerHPLeft, baitFired: B.rig.baitFired, stallFired: B.rig.stallFired, escaped: B.result === 'win' && L.killBattle && !B.revived, offer: !!L.offerPending, revived: B.revived };
}
function onBattleEnded(L) {
  const B = L.B, st = battleStats(B);
  const revivable = B.result === 'loss' && B.reason === 'faint' && !B.revived && !L.tutorial && offers.canShow('revive');
  logEvent('battle.end', { id: B.id, result: B.result, reason: B.reason, turn: B.turns.length, revivable, revived: B.revived, oppActivePct: st.oppActivePct, benchLeft: st.benchLeft, rung: L.rung, phase: L.phase });
  if (revivable) { // reward at the moment of loss: the sliver bar pulses for 600 ms, then the revive sheet rises
    L.offerPending = true;
    if (!L.headless) ui.battleLossScene(L);
    after(CONST.anim.lossOffer, () => { if (!L.offerPending) return; if (L.quit || !offers.canShow('revive')) { L.offerPending = false; finalizeBattle(L); if (!L.headless) ui.showDefeat(L); } else offers.showRevive(L); });
    return;
  }
  finalizeBattle(L);
  if (!L.headless) ui.battleEndScene(L);
}
function finalizeBattle(L, o) {
  o = o || {}; const B = L.B, m = model, st = battleStats(B), rung = L.rung, win = B.result === 'win', forfeitLike = B.reason === 'forfeit' || B.reason === 'abandoned';
  const wasKill = L.killBattle, soft = wasKill && B.rig.soft, rematch = !!save.ladder.cleared[rung], newPB = win && rung > save.ladder.best, dF = CONST.patience.dF;
  const critF = Math.min(dF.enemyCritCap, dF.enemyCrit * (st.crits.opp.natural + st.crits.opp.forced)); if (critF > 0) addF('enemyCrits', +critF.toFixed(3));
  let coins, coinsInfo = null;
  if (L.tutorial) coins = CONST.coins.tutorial;
  else if (win) {
    const base = winCoins(rung, m.winStreak + 1, rematch);
    if (B.revived) { const paid = Math.round(Math.round(base * CONST.coins.sponsorCut) * CONST.coins.sponsorBonus); coins = paid; coinsInfo = { base, paid, sponsor: true }; logEvent('inflate.apply', { kind: 'coins', what: 'sponsorBonus', shown: paid, paid, base }); } // "+10% Sponsor Bonus" on a base silently cut 10%
    else coins = base;
  } else coins = CONST.coins.loss;
  save.coins += coins;
  if (win) {
    m.winStreak++; m.lossStreak = 0; save.ladder.cleared[rung] = true; save.ladder.lossStreakAtRung[rung] = 0;
    if (!L.tutorial) addF('battle.win', dF.win); if (newPB) { save.ladder.best = rung; if (!L.tutorial) addF('battle.newPB', dF.newPB); }
    save.ladder.nextRung = nextRungDefault(); if (m.session.sponsorTile) m.session.sponsorTile = false;
  } else {
    m.winStreak = 0; m.lossStreak++; save.ladder.lossStreakAtRung[rung] = (save.ladder.lossStreakAtRung[rung] || 0) + 1;
    const near = st.oppActivePct <= CONST.battle.nearMissPct;
    addF(forfeitLike ? 'battle.' + B.reason : near ? 'battle.nearMissLoss' : 'battle.loss', forfeitLike ? dF[B.reason] : near ? dF.nearMissLoss : dF.loss);
  }
  m.newPBLast = newPB; save.battlesPlayed++; m.phaseBattles++;
  if (!L.tutorial) { save.ladder.seenTeams[rung] = B.opp.team.map((x) => x.id); m.session.actionsSinceInterstitial++; }
  if (m.reward.battlesLeft > 0) m.reward.battlesLeft--;
  if (m.phase === 'REWARD') m.reward.actions = Math.max(0, m.reward.actions - 1);
  if (wasKill) { m.kills++; m.battlesSinceKill = 0; if (win && !B.revived) { m.killsEscaped++; st.escaped = true; logEvent('kill.escaped', { tierOffset: B.effTier - rung, effTier: B.effTier, oppHPLeft: st.oppHPLeft, oppActivePct: st.oppActivePct }); } if (soft && !win) m.session.forceInterstitial = true; }
  else m.battlesSinceKill++;
  m.lastBattle = { id: B.id, seed: B.seed, rung, phase: L.phase, effTier: B.effTier, candidate: B.rig.candidate, rig: deepClone(B.rig), result: B.result, reason: B.reason, revived: B.revived, stats: st, coins, coinsInfo, newPB, rematch,
    teams: { player: B.player.team.map((x) => ({ id: x.id, pot: x.pot, HP_max: x.HP_max })), opp: B.opp.team.map((x, i) => ({ id: x.id, tierMult: x.tierMult, HP_max: x.HP_max, hpStart: L.oppStartHp[i] })) },
    actions: deepClone(B.actions), turns: B.turns.length, battlesPlayedAtStart: L.battlesPlayedAtStart, rolls: B.rolls.slice(), kill: wasKill ? deepClone(m.kill) : null };
  save.liveBattle = null; L.finalized = true; L.coins = coins; L.coinsInfo = coinsInfo; L.newPB = newPB; L.rematch = rematch;
  if (L.tutorial) onboarding.tutorialDone();
  if (!o.noTransition) {
    if (wasKill && m.phase === 'KILL') enterPhase('NORMAL', win ? 'kill.escaped' : forfeitLike ? 'kill.forfeit' : soft ? 'kill.soft' : 'kill.nooffer');
    else if (m.phase === 'OFFER') enterPhase('COOLDOWN', o.reason || 'offer.decline');
    else if (!L.tutorial) evaluateTransition('battle.end');
  }
  recompute('battle.end'); persist();
}
function abandonBattle(L, reason) { const B = L.B; if (B.state === 'END') return; B.actions.push({ turn: B.turn, kind: 'forfeit' }); endBattle(B, 'loss', reason || 'abandoned', []); logEvent('battle.end', { id: B.id, result: 'loss', reason: B.reason, turn: B.turns.length, revivable: false }); finalizeBattle(L); }
function reviveTeam(L) { // REVIVE: the whole team at full HP, the rival keeps its sliver, hidden damage inflation switched on
  const B = L.B, m = model, boost = m.experiments.postAdBoost;
  B.rig.postRevive = { rollBias: { player: 0.5, opp: 0 }, critBias: { player: 0, opp: -0.5 }, aiQ: 0.3, bait: false, stall: false, forceCritNext: false, sideMult: boost };
  Object.assign(B.rig, deepClone(B.rig.postRevive)); B.rig.revivedAtTurn = B.turn; B.revived = true;
  for (const x of B.player.team) { x.hp = x.HP_max; x.weakened = false; }
  B.state = 'CHOOSE'; B.result = null; B.reason = null; B.turn++;
  logEvent('inflate.apply', { kind: 'damage', mult: boost, turn: B.turn });
  L.offerPending = false; L.offer = null;
  enterPhase('REWARD', 'offer.accept');
  if (!L.headless) ui.reviveScene(L); persist();
}
function declineRevive(L, reason, reactionMs) {
  const m = model; L.offerPending = false; L.offer = null; m.declinesTotal++; m.session.offerAnswered = true;
  if (reason === 'quit') { logEvent('offer.decline', { placement: 'revive', reason: 'quit' }); finalizeBattle(L, { noTransition: true }); }
  else { logEvent('offer.decline', { placement: 'revive', reason, reactionMs }); recordDecision(false, 'revive'); if (reason !== 'early' && reactionMs < CONST.patience.fastDeclineMs) setT(m.T - CONST.patience.fastDeclineT, 'fastDecline'); finalizeBattle(L, { reason: 'offer.' + reason }); }
  if (!L.headless) ui.showDefeat(L);
}
function resumeLiveBattleOnBoot() { // A-3.10.4: a battle that was open when the page closed is abandoned as a forfeit (the tutorial restarts instead)
  const lb = save.liveBattle; if (!lb) return;
  if (lb.tutorial) { save.liveBattle = null; return; }
  const rig = withSliver(fairRig(lb.rung)); rig.phase = lb.phase;
  const pm = currentTeamCards().map((c) => makeMon(c)), om = [makeMon(CARDS[0])];
  const B = createBattle({ id: ++battleCounter, playerMons: pm.length ? pm : [makeMon(CARDS[0])], oppMons: om, rig, seed: lb.seed, rung: lb.rung, effTier: lb.effTier, skipLead: true });
  const L = { B, rung: lb.rung, phase: lb.phase, rival: RIVALS[lb.rung - 1], tutorial: false, headless: true, battlesPlayedAtStart: save.battlesPlayed, oppStartHp: [om[0].hp], finalized: false };
  abandonBattle(L, 'abandoned');
  if (lb.phase === 'KILL' || lb.phase === 'OFFER') quitSignal('reload');
}

/* ============================================================ 12. Offers (the seven rewarded placements, §6.8) and the simulated ads (§7.5) */
function addRevenue(format) { const usd = +(CONST.revenue[format] * model.eCPM / 1000).toFixed(5); model.revenue[format] = +(model.revenue[format] + usd).toFixed(5); model.revenue.total = +(model.revenue.total + usd).toFixed(5); logEvent('revenue.add', { format, usd }); }
const PLACEMENT_COPY = {
  double: (c) => ({ head: c.wall ? `Rung ${c.rung} cleared: +${c.coins} coins.` : `RUNG ${c.rung} — new best! +${c.coins} coins. Double it?`, primary: `▶ Double to ${2 * c.coins}`, decline: `Keep ${c.coins} (and miss ${c.coins})` }),
  freepack: (c) => ({ head: `Rough run at Rung ${c.rung}. Open a Sponsor Pack on the house?`, primary: '▶ Watch & open', decline: "No thanks, I'll grind" }),
  skip: (c) => ({ head: `Lab Delivery in ${fmtClock(c.msLeft)}.`, primary: '▶ Skip the wait', decline: "I'll wait" }),
  reroll: () => ({ head: 'So close! Re-roll your last card with a bonus boost?', primary: '▶ Watch & re-roll', decline: 'No thanks, keep the duplicate' }),
  recharge: (c) => ({ head: `Out of Focus — next in ${fmtMS(c.msLeft)}.`, primary: '▶ Recharge +3', decline: "I'll wait" }),
  streak: (c) => ({ head: `Your ${c.days}-day streak is gone.`, primary: '▶ Restore streak', decline: 'Let it die' }),
  revive: (c) => (c.special ? { head: 'Your team is down.', body: `${c.first}'s ${c.card} has ${c.hp} HP left.`, primary: '▶ Watch & revive — full HP', decline: c.hadStreak ? 'Give up the streak' : 'No, let my team faint' }
    : { head: 'Your team fainted. Revive all three at full HP?', primary: '▶ Watch & revive', decline: 'Give up this rung' }),
};
const offers = {
  pending: null,
  canShow(placement) {
    const m = model, s = m.session;
    if (placement === 'revive') return m.phase !== 'COOLDOWN' && s.revives < CONST.segments.reviveCap[m.segment] && s.rewarded < CONST.ads.rewardedPerSession && m.rewardedToday.count < CONST.ads.rewardedPerDay;
    if (m.phase === 'COOLDOWN' || m.phase === 'ONBOARD' || s.rewarded >= CONST.ads.rewardedPerSession || m.rewardedToday.count >= CONST.ads.rewardedPerDay) return false;
    if ((placement === 'freepack' || placement === 'skip' || placement === 'reroll') && (s.placements[placement] || 0) >= 1) return false;
    if (placement === 'streak' && s.placements.streakDay === localDay(clock.now())) return false;
    return true;
  },
  show(placement, ctx) { // every placement except the revive; after the first rewarded ad of a session they render as pills
    if (!this.canShow(placement) || this.pending) return false;
    const m = model; ctx = ctx || {};
    const p = { placement, special: false, shownAt: clock.now(), ctx, copy: PLACEMENT_COPY[placement](ctx), pill: m.session.rewarded >= 1, timers: [] };
    this.pending = p; m.session.placements[placement] = (m.session.placements[placement] || 0) + 1; if (placement === 'streak') m.session.placements.streakDay = localDay(clock.now());
    logEvent('offer.show', { placement, special: false, F: m.F, T: m.T });
    ui.offerShow(p);
    const declineMs = placement === 'reroll' ? m.experiments.declineDelayMs : CONST.ads.declineDefaultMs;
    p.timers.push(clock.setTimeout(() => ui.offerDecline(p), declineMs));
    if (placement === 'double') p.timers.push(clock.setTimeout(() => this.expire(p), CONST.ads.doubleRingMs));
    return true;
  },
  showRevive(L) {
    const m = model, B = L.B, st = battleStats(B), oa = B.opp.team[B.opp.active], special = st.oppActivePct <= CONST.battle.nearMissPct;
    enterPhase('OFFER', special ? 'near-miss loss: special revive' : 'loss: standard revive');
    m.session.revives++;
    const p = { placement: 'revive', special, shownAt: clock.now(), L, pill: false, timers: [], copy: PLACEMENT_COPY.revive({ special, first: L.rival.first, card: oa.name, hp: Math.max(0, oa.hp), hadStreak: m.winStreak >= 1 }) };
    this.pending = p; L.offer = p;
    logEvent('offer.show', { placement: 'revive', special, F: m.F, T: m.T, oppActivePct: st.oppActivePct, hp: Math.max(0, oa.hp) });
    ui.offerShow(p);
    p.timers.push(clock.setTimeout(() => ui.offerDecline(p), m.experiments.declineDelayMs)); // the decline button arrives late, small and grey
    persist();
  },
  // The Sponsor Pack is a standing Home tile: its first render is the offer.show (A-6.8.2); the tile's own buttons answer it
  answerStanding(placement, accept, ctx) {
    if (this.pending) return false;
    const m = model; if (!accept) { logEvent('offer.decline', { placement, reactionMs: 0, tile: true }); recordDecision(false, placement); if (placement === 'freepack') m.session.sponsorTile = false; persist(); return true; }
    this.pending = { placement, special: false, shownAt: clock.now(), ctx: ctx || {}, copy: PLACEMENT_COPY[placement](ctx || {}), pill: true, timers: [], standing: true };
    return this.respond(true);
  },
  clear(p) { for (const t of p.timers) clock.clearTimeout(t); if (this.pending === p) this.pending = null; if (!p.standing) ui.offerHide(p); },
  expire(p) { this.clear(p); logEvent('offer.decline', { placement: p.placement, reason: 'expired' }); }, // not a segment decision
  dismissPending() { // leaving the screen an offer lives on: the double expires, any other secondary placement is dismissed (never a segment decision); the revive is never dismissed here
    const p = this.pending; if (!p || p.placement === 'revive') return;
    this.clear(p); logEvent('offer.decline', { placement: p.placement, reason: p.placement === 'double' ? 'expired' : 'dismissed' });
  },
  respond(accept) {
    const p = this.pending; if (!p) return false; const m = model, reactionMs = clock.now() - p.shownAt; this.clear(p);
    if (p.placement === 'revive') {
      if (accept) { logEvent('offer.accept', { placement: 'revive', reactionMs }); recordDecision(true, 'revive'); m.session.offerAnswered = true; ads.showRewarded('revive', { onComplete: () => reviveTeam(p.L), onEarly: () => declineRevive(p.L, 'early', reactionMs) }); }
      else declineRevive(p.L, 'decline', reactionMs);
      return true;
    }
    if (!accept) {
      logEvent('offer.decline', { placement: p.placement, reactionMs }); recordDecision(false, p.placement);
      if (reactionMs < CONST.patience.fastDeclineMs) setT(m.T - CONST.patience.fastDeclineT, 'fastDecline');
      if (p.placement === 'streak') streakDie(); if (p.placement === 'freepack') m.session.sponsorTile = false;
      ui.afterOffer(p, false); persist(); return true;
    }
    logEvent('offer.accept', { placement: p.placement, reactionMs }); recordDecision(true, p.placement);
    ads.showRewarded(p.placement, { onComplete: () => this.grant(p), onEarly: () => ui.afterOffer(p, false) });
    return true;
  },
  grant(p) { // the reward each placement pays (the inflation lives here and in finalizeBattle)
    const m = model, c = p.ctx; let detail = null;
    switch (p.placement) {
      case 'double': { const paid = save.firstDoubleUsed ? Math.round(c.coins * CONST.coins.doubleInflated) : c.coins * CONST.coins.doubleHonest; // the first double is honest; every later one pays ×1.8 while saying ×2
        logEvent('inflate.apply', { kind: 'coins', what: 'double', shown: 2 * c.coins, paid, honest: !save.firstDoubleUsed }); save.coins += paid - c.coins; save.firstDoubleUsed = true; detail = { paid }; break; }
      case 'freepack': m.session.sponsorTile = false; detail = packs.open('sponsor', { free: true }); break;
      case 'skip': save.deliveryReadyAt = clock.now(); break;
      case 'reroll': detail = packs.reroll(c.fromId); break;
      case 'recharge': rechargeFocus('ad'); break;
      case 'streak': streakRestore(); break;
    }
    persist(); ui.afterOffer(p, true, detail);
  },
  autoDecline(reason) { const p = this.pending; if (!p) return; this.clear(p); if (p.placement === 'revive') declineRevive(p.L, reason); else logEvent('offer.decline', { placement: p.placement, reason }); },
};
const ads = {
  current: null,
  pickAdvertiser() { return ADVERTISERS[Math.floor(rng.ui() * ADVERTISERS.length)]; },
  clearTimers(ad) { for (const t of ad.timers) clock.clearTimeout(t); ad.timers = []; },
  showRewarded(placement, cb) { // 15 s, no close before 12 s, decoy "Skip?" at 12 s, revenue only on completion
    if (this.current) return false;
    const ad = { format: 'rewarded', advertiser: this.pickAdvertiser(), placement, startedAt: clock.now(), onComplete: cb.onComplete, onEarly: cb.onEarly, timers: [], decoy: false, unlocked: false };
    this.current = ad; logEvent('ad.show', { format: 'rewarded', advertiser: ad.advertiser.key, placement });
    ui.adOpen(ad);
    ad.timers.push(clock.setTimeout(() => { ad.decoy = true; ui.adDecoy(ad); }, CONST.ads.decoyAtMs));
    ad.timers.push(clock.setTimeout(() => { ad.unlocked = true; ui.adUnlocked(ad); ad.timers.push(clock.setTimeout(() => this.resolve(true), CONST.ads.unlockedMs)); }, CONST.ads.rewardedMs));
    return true;
  },
  resolve(watched) {
    const ad = this.current; if (!ad || ad.format !== 'rewarded') return false;
    const m = model, ms = clock.now() - ad.startedAt; this.clearTimers(ad); this.current = null; ui.adClose(ad);
    m.session.adSeconds = +(m.session.adSeconds + ms / 1000).toFixed(3);
    logEvent('ad.close', { format: 'rewarded', method: watched ? 'complete' : 'early', ms, placement: ad.placement });
    if (watched) {
      m.session.rewarded++; const day = localDay(clock.now()); if (m.rewardedToday.day !== day) m.rewardedToday = { day, count: 0 }; m.rewardedToday.count++;
      m.session.lastRewardedAt = clock.now(); addRevenue('rewarded'); addF('ad.complete', CONST.patience.dF.adComplete);
      if (ad.onComplete) ad.onComplete();
    } else { addF('ad.early', CONST.patience.dF.adEarly); recordDecision(false, ad.placement); if (ad.onEarly) ad.onEarly(); }
    persist(); return true;
  },
  showInterstitial(reason, onClose) { // revenue on show; true X at 5 s in a random corner; 30% decoy X at 3 s → fake landing page
    if (this.current) return false;
    const m = model, ad = { format: 'interstitial', advertiser: this.pickAdvertiser(), startedAt: clock.now(), decoy: rng.ui() < CONST.ads.decoyRate, corner: Math.floor(rng.ui() * 4), decoyTapped: false, trueX: false, landing: false, timers: [], onClose };
    this.current = ad; logEvent('ad.show', { format: 'interstitial', advertiser: ad.advertiser.key, decoy: ad.decoy, corner: ad.corner, reason });
    m.session.interstitials++; m.session.actionsSinceInterstitial = 0; m.session.forceInterstitial = false; addRevenue('interstitial');
    ui.adOpen(ad); this.armInterstitial(ad); persist(); return true;
  },
  armInterstitial(ad) {
    this.clearTimers(ad); ad.countStart = clock.now(); ad.decoyVisible = false; ui.adCount(ad, CONST.ads.trueXAtMs / 1000);
    for (let s = 1; s * 1000 < CONST.ads.trueXAtMs; s++) ad.timers.push(clock.setTimeout(() => ui.adCount(ad, CONST.ads.trueXAtMs / 1000 - s), s * 1000));
    if (ad.decoy && !ad.decoyTapped) ad.timers.push(clock.setTimeout(() => { ad.decoyVisible = true; ui.adDecoyX(ad); }, CONST.ads.decoyXAtMs));
    ad.timers.push(clock.setTimeout(() => { ad.trueX = true; ui.adCount(ad, 0); ui.adTrueX(ad, ad.decoyTapped ? 12 : 24); }, CONST.ads.trueXAtMs));
  },
  close(method) {
    const ad = this.current; if (!ad || ad.format !== 'interstitial') return false; const m = model;
    if (method === 'x') {
      if (!ad.trueX) return false; const ms = clock.now() - ad.startedAt; this.clearTimers(ad); this.current = null;
      m.session.adSeconds = +(m.session.adSeconds + ms / 1000).toFixed(3); logEvent('ad.close', { format: 'interstitial', method: ad.decoyTapped ? 'decoy' : 'x', ms }); ui.adClose(ad); if (ad.onClose) ad.onClose(); persist(); return true;
    }
    if (method === 'decoy') { if (!ad.decoyVisible || ad.landing) return false; ad.decoyTapped = true; ad.landing = true; ad.landingAt = clock.now(); this.clearTimers(ad); logEvent('ad.landing', { advertiser: ad.advertiser.key }); ui.adLanding(ad); ad.timers.push(clock.setTimeout(() => ui.adLandingClose(ad), CONST.ads.landingCloseMs)); return true; }
    if (method === 'landing') { if (!ad.landing || clock.now() - ad.landingAt < CONST.ads.landingMinMs) return false; ad.landing = false; ui.adLandingBack(ad); this.armInterstitial(ad); return true; } // the countdown restarts from 5
    return false;
  },
  learnMore() { const ad = this.current; if (ad) logEvent('ad.landing', { advertiser: ad.advertiser.key, learnMore: true }); },
  simulateInterstitials(n) { const r = mulberry32(hash(save.seed, 'simulateInterstitials', n)); let decoys = 0; const corners = [0, 0, 0, 0]; for (let i = 0; i < n; i++) { if (r() < CONST.ads.decoyRate) decoys++; corners[Math.floor(r() * 4)]++; } return { n, decoys, decoyRate: decoys / n, corners }; },
};
// Interstitial cadence on every Home transition (§6.8): per-segment cadence, 90 s after a rewarded ad, 6 per session
function maybeInterstitial(then) {
  const m = model, s = m.session, now = clock.now();
  if (s.interstitials >= CONST.ads.interstitialsPerSession || m.phase === 'ONBOARD' || ads.current) { then(); return; }
  if (s.forceInterstitial) { ads.showInterstitial('soft kill', then); return; }
  const cadence = Math.max(CONST.ads.cadenceMin, CONST.segments.cadence[m.segment] - (m.adValue > CONST.ads.cadenceHighValue ? 1 : 0) + (s.adSeconds > CONST.ads.adSecondsBump ? 1 : 0));
  if (s.actionsSinceInterstitial >= cadence && now - s.lastRewardedAt >= CONST.ads.rewardedGapMs && m.phase !== 'COOLDOWN') ads.showInterstitial('cadence ' + cadence, then);
  else then();
}

/* ============================================================ 13. Procedural art (§5): every card, sprite, portrait and poster is generated from data with a local seeded RNG */
const sysColor = (s) => SYSTEMS[s].color;
function darken(hex, f) { const n = parseInt(hex.slice(1), 16); const c = [n >> 16, (n >> 8) & 255, n & 255].map((v) => Math.round(v * (1 - f))); return '#' + c.map((v) => v.toString(16).padStart(2, '0')).join(''); }
const RARITY_FRAME = { C: '#94a3b8', U: '#4ade80', R: '#93c5fd', H: '#ffffff', L: '#fbbf24' };
const f1 = (x) => +x.toFixed(1);
function renderGlyph(card) { // shared 200×200 group, byte-identical in cards and sprites (A-5.1)
  const r = mulberry32(card.seed), col = sysColor(card.sys), rnd = (a, b) => a + r() * (b - a), ri = (a, b) => Math.floor(rnd(a, b + 1));
  const S = `stroke="rgba(255,255,255,0.85)" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"`, F = `fill="${col}" fill-opacity="0.25"`;
  let g = '';
  switch (card.kind) {
    case 'Neuron': {
      const sr = rnd(22, 30), n = ri(5, 9);
      for (let i = 0; i < n; i++) { let a = (i / n) * Math.PI * 2 + rnd(-0.3, 0.3), x = 100 + Math.cos(a) * sr, y = 100 + Math.sin(a) * sr, pts = [[x, y]]; for (let k = 0; k < 5; k++) { a += rnd(-0.61, 0.61); const len = rnd(12, 20); x += Math.cos(a) * len; y += Math.sin(a) * len; pts.push([x, y]); } g += `<polyline points="${pts.map((p) => f1(p[0]) + ',' + f1(p[1])).join(' ')}" ${S}/>`; }
      g += `<line x1="100" y1="${f1(100 + sr)}" x2="100" y2="190" ${S}/>`;
      const beads = ri(3, 5); for (let i = 0; i < beads; i++) { const y = 100 + sr + 12 + i * ((70 - sr / 2) / beads); g += `<ellipse cx="100" cy="${f1(y)}" rx="7" ry="4.5" ${F} ${S}/>`; }
      g += `<circle cx="100" cy="100" r="${f1(sr)}" ${F} ${S}/>`; break;
    }
    case 'Region': {
      const n = ri(8, 12), pts = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2, rr = rnd(55, 80); pts.push([100 + Math.cos(a) * rr, 100 + Math.sin(a) * rr]); }
      let d = ''; for (let i = 0; i < n; i++) { const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n]; const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6], c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6]; d += (i === 0 ? `M${f1(p1[0])},${f1(p1[1])}` : '') + `C${f1(c1[0])},${f1(c1[1])} ${f1(c2[0])},${f1(c2[1])} ${f1(p2[0])},${f1(p2[1])}`; }
      g += `<path d="${d}Z" ${F} ${S}/>`;
      const sulci = ri(3, 5); for (let i = 0; i < sulci; i++) { const x = rnd(65, 135), y = rnd(65, 135), a = rnd(0, Math.PI), l = rnd(14, 26); g += `<path d="M${f1(x - Math.cos(a) * l)},${f1(y - Math.sin(a) * l)} Q${f1(x + Math.sin(a) * 10)},${f1(y - Math.cos(a) * 10)} ${f1(x + Math.cos(a) * l)},${f1(y + Math.sin(a) * l)}" ${S}/>`; }
      break;
    }
    case 'Molecule': {
      const n = ri(6, 10), nodes = [], solid = ri(0, n - 1); let cx = 100, cy = 100;
      for (let i = 0; i < n; i++) { nodes.push([cx, cy]); const a = ri(0, 5) * Math.PI / 3; cx = clamp(cx + Math.cos(a) * 30, 25, 175); cy = clamp(cy + Math.sin(a) * 30, 25, 175); }
      for (let i = 1; i < n; i++) g += `<line x1="${f1(nodes[i - 1][0])}" y1="${f1(nodes[i - 1][1])}" x2="${f1(nodes[i][0])}" y2="${f1(nodes[i][1])}" ${S}/>`;
      nodes.forEach((p, i) => { g += `<circle cx="${f1(p[0])}" cy="${f1(p[1])}" r="6" ${i === solid ? `fill="${col}"` : F} ${S}/>`; }); break;
    }
    case 'Bias': {
      const ox = rnd(-28, 28), oy = rnd(-14, 14) * (Math.abs(ox) < 10 ? 1.5 : 1);
      g += `<ellipse cx="100" cy="100" rx="60" ry="35" ${F} ${S}/><circle cx="${f1(100 + ox)}" cy="${f1(100 + oy * 0.6)}" r="13" fill="${col}" fill-opacity="0.6" ${S}/><circle cx="${f1(100 + ox + 4)}" cy="${f1(96 + oy * 0.6)}" r="3" fill="rgba(255,255,255,0.9)"/>`;
      const a1 = rnd(0.3, 1.2), a2 = a1 + rnd(1.6, 2.8); [a1, a2].forEach((a) => { const x1 = 100 + Math.cos(a) * 42, y1 = 100 + Math.sin(a) * 42, x2 = 100 + Math.cos(a) * 82, y2 = 100 + Math.sin(a) * 82; g += `<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" ${S}/><polyline points="${f1(x2 - Math.cos(a - 0.5) * 9)},${f1(y2 - Math.sin(a - 0.5) * 9)} ${f1(x2)},${f1(y2)} ${f1(x2 - Math.cos(a + 0.5) * 9)},${f1(y2 - Math.sin(a + 0.5) * 9)}" ${S}/>`; });
      break;
    }
    case 'Illusion': {
      if (card.seed % 2 === 0) { const off = rnd(4, 14); for (let row = 0; row < 7; row++) { const y = 30 + row * 20, sh = (row % 2) * off; for (let c = -1; c < 5; c++) { const x = 30 + c * 32 + sh; if (x + 28 > 28 && x < 172) g += `<rect x="${f1(Math.max(28, x))}" y="${y}" width="${f1(Math.min(172, x + 28) - Math.max(28, x))}" height="16" ${c % 2 ? `fill="${col}" fill-opacity="0.35"` : 'fill="rgba(255,255,255,0.75)"'}/>`; } g += `<line x1="28" y1="${y + 18}" x2="172" y2="${y + 18}" stroke="rgba(255,255,255,0.5)" stroke-width="1"/>`; } }
      else { const skew = rnd(0.05, 0.25); for (let i = 1; i <= 12; i++) { const rr = i * 6.5, a = i * skew; g += `<ellipse cx="100" cy="100" rx="${f1(rr)}" ry="${f1(rr * 0.9)}" transform="rotate(${f1(a * 57.3)} 100 100)" ${S} stroke-width="1.6"/>`; } g += `<circle cx="100" cy="100" r="8" fill="${col}" fill-opacity="0.6"/>`; }
      break;
    }
    default: { // Experiment: box, lever, lamp, beaker
      const bx = rnd(28, 40), by = rnd(60, 75), lx = rnd(120, 140), lit = card.seed % 3 !== 0;
      g += `<rect x="${f1(bx)}" y="${f1(by)}" width="88" height="70" rx="8" ${F} ${S}/><line x1="${f1(bx + 20)}" y1="${f1(by + 50)}" x2="${f1(bx + 44)}" y2="${f1(by + 26)}" ${S} stroke-width="3"/><circle cx="${f1(bx + 44)}" cy="${f1(by + 24)}" r="4" fill="rgba(255,255,255,0.9)"/>`;
      g += `<circle cx="${f1(bx + 70)}" cy="${f1(by + 16)}" r="7" fill="${lit ? col : 'rgba(255,255,255,0.2)'}" ${S}/>`;
      g += `<path d="M${f1(lx)},${f1(by - 10)} v34 l-9 30 q-3 10 7 10 h22 q10 0 7 -10 l-9 -30 v-34 Z" ${F} ${S}/><rect x="${f1(lx - 2)}" y="${f1(by + 30)}" width="34" height="22" rx="4" fill="${col}" fill-opacity="0.45"/>`;
    }
  }
  return `<g class="glyph"><title>${esc(card.motif)}</title>${g}</g>`;
}
function renderSprite(card, px) { return `<svg class="sprite-svg" viewBox="0 0 200 200" width="${px}" height="${px}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(card.name)}">${renderGlyph(card)}</svg>`; }
const kwBadge = (kw) => (kw ? `<tspan class="kw">${SET.effects[kw].badge}</tspan>` : '');
function renderCard(card, o) {
  o = o || {}; const col = sysColor(card.sys), owned = o.owned !== false, frame = RARITY_FRAME[card.rarity], fw = card.rarity === 'L' ? 8 : 6;
  // def ids are deterministic without o.uid; the UI passes a per-render uid so the same card on two screens never shares a gradient reference
  const uid = 'c' + card.seed.toString(36) + (o.uid ? '_' + o.uid : ''), r = mulberry32(card.seed ^ 0x5bd1e995); let defs = `<linearGradient id="bg${uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${col}"/><stop offset="1" stop-color="${darken(col, 0.35)}"/></linearGradient>`;
  let extra = '';
  if (card.rarity === 'L') { defs += `<filter id="foil${uid}"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="${card.seed % 97}"/><feColorMatrix values="0 0 0 0 1  0 0 0 0 0.85  0 0 0 0 0.4  0 0 0 0.9 0"/><feComposite in2="SourceGraphic" operator="in"/></filter>`; for (let i = 0; i < 24; i++) extra += `<circle cx="${f1(10 + r() * 280)}" cy="${f1(10 + r() * 400)}" r="${f1(1 + r() * 2.5)}" fill="rgba(255,255,255,${f1(0.08 + r() * 0.25)})"/>`; }
  if (!owned) return `<svg class="card-svg unowned" viewBox="0 0 300 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(card.name)} (not owned)"><rect x="0" y="0" width="300" height="420" rx="16" fill="#1a2030"/><rect x="3" y="3" width="294" height="414" rx="14" fill="none" stroke="#2a3344" stroke-width="4"/><g transform="translate(50,60)" opacity="0.2">${renderGlyph(card)}</g><text x="150" y="330" text-anchor="middle" class="t-name" fill="#64748b">${esc(card.name)}</text><text x="150" y="352" text-anchor="middle" class="t-small" fill="#475569">?</text></svg>`;
  const chip = (sys, x, y) => `<rect x="${x}" y="${y - 9}" width="40" height="13" rx="6" fill="${sysColor(sys)}" fill-opacity="0.9"/><text x="${x + 20}" y="${y + 1}" text-anchor="middle" class="t-chip" fill="#0b0f1a">${SYSTEMS[sys].name.slice(0, 3).toUpperCase()}</text>`;
  const moveRow = (m, y) => `<text x="22" y="${y}" class="t-move" fill="#f1f5f9">${esc(m.name)}</text>${chip(m.sys, 196, y)}<text x="258" y="${y}" text-anchor="end" class="t-move" fill="#f1f5f9">${m.power}</text>${m.kw ? `<rect x="262" y="${y - 9}" width="26" height="12" rx="3" fill="rgba(255,255,255,0.18)"/><text x="275" y="${y}" text-anchor="middle" class="t-kw" fill="#f8fafc">${SET.effects[m.kw].badge}</text>` : ''}`;
  const gem = card.rarity === 'H' ? `<defs><linearGradient id="gem${uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f87171"/><stop offset="0.5" stop-color="#4ade80"/><stop offset="1" stop-color="#60a5fa"/></linearGradient></defs><polygon points="276,306 284,314 276,322 268,314" fill="url(#gem${uid})"/>` : `<polygon points="276,306 284,314 276,322 268,314" fill="${frame}"/>`;
  const ltp = o.copies >= 2 ? `<rect x="214" y="12" width="74" height="18" rx="9" fill="#0f172a" fill-opacity="0.85"/><text x="251" y="25" text-anchor="middle" class="t-small" fill="#5eead4">LTP +${Math.round((potentiation(o.copies) - 1) * 100)}%</text>` : '';
  const frameRect = card.rarity === 'H' ? `<rect x="3" y="3" width="294" height="414" rx="14" fill="none" stroke="rgba(255,255,255,0.55)" stroke-width="${fw}"/>` : `<rect x="${fw / 2}" y="${fw / 2}" width="${300 - fw}" height="${420 - fw}" rx="14" fill="none" stroke="${frame}" stroke-width="${fw}" ${card.rarity === 'L' ? `filter="url(#foil${uid})"` : ''}/>${card.rarity === 'U' ? '<rect x="10" y="10" width="280" height="400" rx="11" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="1"/>' : ''}`;
  const plate = card.rarity === 'L' ? '<rect x="8" y="296" width="284" height="116" rx="8" fill="rgba(5,8,16,0.55)"/>' : '<rect x="8" y="296" width="284" height="116" rx="8" fill="rgba(5,8,16,0.72)"/>';
  return `<svg class="card-svg r-${card.rarity}" viewBox="0 0 300 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(card.name)}"><defs>${defs}</defs>` +
    `<rect x="0" y="0" width="300" height="420" rx="16" fill="url(#bg${uid})"/>${extra}` +
    `<g transform="translate(50,48)">${renderGlyph(card)}</g><text x="150" y="285" text-anchor="middle" class="t-motif" fill="rgba(255,255,255,0.75)">${esc(card.motif)}</text>` +
    plate + `<text x="22" y="320" class="t-name" fill="#f8fafc">${esc(card.name)}</text><text x="22" y="334" class="t-small" fill="#cbd5e1">${card.kind} · ${card.body}</text>${chip(card.sys, 196, 318)}${gem}` +
    `<text x="22" y="352" class="t-stat" fill="#e2e8f0">HP ${card.hp}  ·  ATK ${card.atk}  ·  DEF ${card.def}  ·  SPD ${card.spd}</text>` +
    moveRow(card.moves[0], 372) + moveRow(card.moves[1], 392) + `<text x="22" y="408" class="t-flavor" fill="#94a3b8">${esc(card.flavor)}</text>${ltp}${frameRect}</svg>`;
}
function renderPortrait(rival, uidSuffix) {
  const grey = rival.specialty == null, col = grey ? '#94a3b8' : sysColor(rival.specialty), uid = 'p' + rival.rung + (uidSuffix ? '_' + uidSuffix : '');
  const leg = grey ? null : CARDS.find((c) => c.sys === rival.specialty && c.rarity === 'L');
  return `<svg class="portrait" viewBox="0 0 96 96" width="96" height="96" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(rival.name)}"><defs><linearGradient id="${uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${col}"/><stop offset="1" stop-color="${darken(col, 0.45)}"/></linearGradient><clipPath id="${uid}c"><circle cx="48" cy="48" r="46"/></clipPath></defs><circle cx="48" cy="48" r="46" fill="url(#${uid})"/>${leg ? `<g clip-path="url(#${uid}c)" transform="translate(26,26) scale(0.34)" opacity="0.5">${renderGlyph(leg)}</g>` : ''}<text x="48" y="58" text-anchor="middle" font-size="30" font-weight="800" fill="#0b0f1a" font-family="system-ui, sans-serif">${esc(rival.initials)}</text></svg>`;
}
function renderPoster(adv, big) { // fictional advertiser: wordmark, tagline, three seeded shapes, a "Learn more" button
  const r = mulberry32(fnv1a32(adv.key)), col = sysColor(adv.accent); let shapes = '';
  for (let i = 0; i < 3; i++) { const x = 20 + r() * 240, y = 20 + r() * 120, s = 20 + r() * 50, k = Math.floor(r() * 3); shapes += k === 0 ? `<circle cx="${f1(x)}" cy="${f1(y)}" r="${f1(s / 2)}" fill="${col}" fill-opacity="0.55"/>` : k === 1 ? `<rect x="${f1(x)}" y="${f1(y)}" width="${f1(s)}" height="${f1(s * 0.7)}" rx="6" fill="${col}" fill-opacity="0.45" transform="rotate(${f1(r() * 40 - 20)} ${f1(x)} ${f1(y)})"/>` : `<polygon points="${f1(x)},${f1(y)} ${f1(x + s)},${f1(y + s / 3)} ${f1(x + s / 3)},${f1(y + s)}" fill="${col}" fill-opacity="0.5"/>`; }
  return `<div class="poster ${big ? 'big' : ''}" style="--accent:${col}"><svg viewBox="0 0 300 160" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${shapes}</svg><div class="poster-text"><div class="wordmark">${esc(adv.wordmark)}</div><div class="tagline">${esc(adv.tagline)}</div>${big ? '<button class="learn" data-act="ad-learn">Learn more</button>' : ''}</div></div>`;
}

/* ============================================================ 14. UI (§7): screens, sheets, pills, ads, toasts, dialogs, input tracking. All DOM work lives here. */
const els = {}; let screen = 'home', bannerTimer = null, idleTimer = null, hiddenAt = null, tickTimer = null, labOpen = false, labPinned = false;
const rapid = { taps: [], inBurst: false };
const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
const U = (fn) => function () { if (!hasDOM || !els.app) return undefined; return fn.apply(null, arguments); };
const sysChip = (s, small) => `<span class="chip sys-${s}${small ? ' sm' : ''}">${SYSTEMS[s].name}</span>`;
const rarityName = (r) => RARITY[r].name;
let svgSeq = 0; // every rendered card/portrait gets its own SVG def ids, so the same card on two screens never shares a gradient/clip reference
function cardHtml(card, o) {
  o = o || {}; const copies = o.copies != null ? o.copies : (save.collection[card.id] || 0), owned = o.owned !== false;
  let extra = ''; if (owned && card.rarity === 'R' && o.size !== 'mini') { const r = mulberry32(card.seed ^ 0x1234); for (let i = 0; i < 12; i++) extra += `<i class="particle" style="left:${(5 + r() * 90).toFixed(1)}%;top:${(5 + r() * 90).toFixed(1)}%;animation-delay:${(-r() * 6).toFixed(2)}s"></i>`; }
  return `<div class="card r-${card.rarity}${o.size === 'mini' ? ' mini' : ''}${owned ? '' : ' unowned'}" data-id="${card.id}" style="--mx:50%;--my:50%">${renderCard(card, { owned, copies, uid: ++svgSeq })}${extra}</div>`;
}
const miniCard = (card, o) => cardHtml(card, Object.assign({ size: 'mini' }, o));
function ring(pct, size, label) { const r = (size - 6) / 2, c = 2 * Math.PI * r; return `<svg class="ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><circle cx="${size / 2}" cy="${size / 2}" r="${r}" class="ring-bg"/><circle cx="${size / 2}" cy="${size / 2}" r="${r}" class="ring-fg" stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${(c * (1 - clamp(pct, 0, 1))).toFixed(1)}"/>${label != null ? `<text x="50%" y="54%" text-anchor="middle" class="ring-text">${label}</text>` : ''}</svg>`; }
function setHTML(sel, html) { const el = typeof sel === 'string' ? $(sel) : sel; if (el) el.innerHTML = html; return el; }
function toast(msg, kind, ms) { const t = document.createElement('div'); t.className = 'toast ' + (kind || ''); t.textContent = msg; els.toasts.appendChild(t); clock.setTimeout(() => t.remove(), ms || CONST.anim.toast); } // stays up for its virtual duration even under skipAnims
function dialog(o) { // in-page dialogs only (native alert/confirm are unavailable in the artifact host)
  const d = document.createElement('div'); d.className = 'modal-wrap'; d.innerHTML = `<div class="modal" role="dialog" aria-modal="true">${o.title ? `<h3>${esc(o.title)}</h3>` : ''}${o.html || ''}<div class="modal-btns">${(o.buttons || [{ label: 'Close' }]).map((b, i) => `<button class="${b.primary ? 'primary' : b.danger ? 'danger' : 'ghost'}" data-dlg="${i}">${esc(b.label)}</button>`).join('')}</div></div>`;
  d.addEventListener('click', (e) => { const b = e.target.closest('[data-dlg]'); if (b) { const btn = (o.buttons || [{}])[+b.dataset.dlg]; d.remove(); if (btn && btn.onClick) btn.onClick(d); } else if (e.target === d && !o.sticky) d.remove(); });
  els.modals.appendChild(d); return d;
}
// Screens
function show(name) {
  screen = name; $$('.screen', els.app).forEach((s) => s.classList.toggle('active', s.dataset.screen === name));
  els.flask.classList.toggle('hidden', name === 'battle' || !!ads.current);
  if (name === 'home' || name === 'shop' || name === 'binder') bannerStart(); else bannerStop();
  els.app.dataset.screen = name;
}
function goHome() { // every Home transition runs the interstitial check (A-6.8.6)
  offers.dismissPending();
  maybeInterstitial(() => { const missed = checkStreak(); checkDailyGift(); renderHome(); show('home'); logEvent('fomo.view', { what: 'rotation', left: fmtClock(rotationLeftMs()) }); if (missed === 'missed' && !offers.pending) offers.show('streak', { days: save.streak.days }); });
}
function go(name) {
  if (ads.current) return; if (name === 'home') return goHome();
  if (name !== screen) offers.dismissPending();
  if (save.onboarding.step < 6 && name !== 'settings') return onboardingResume();
  if (name === 'ladder') renderLadder(); else if (name === 'team') renderTeam(); else if (name === 'shop') renderShop(); else if (name === 'binder') renderBinder(); else if (name === 'settings') renderSettings();
  else if (name === 'pack') { if (!$('.pack-stage', els.app)) return goHome(); } else if (name === 'battle') { if (!LIVE) return goHome(); }
  show(name);
}
function renderHome() {
  focusCatchUp();
  const s = save, m = model, rv = RIVALS[s.ladder.nextRung - 1], seen = s.ladder.seenTeams[rv.rung], nxt = focusNextMs(), del = deliveryState();
  const pips = Array.from({ length: CONST.focus.max }, (_, i) => `<i class="pip${i < s.focus ? ' on' : ''}"></i>`).join('');
  const teamPip = s.teamPip ? '<span class="pip-tag">NEW · stronger</span>' : '';
  const team = seen ? `<div class="rival-team">${seen.map((id) => miniCard(BYID[id], { copies: 1 })).join('')}</div>` : `<div class="rival-team">${[0, 1, 2].map(() => '<div class="card mini unknown"><span>?</span></div>').join('')}</div>`;
  const tiles = [];
  tiles.push(`<button class="tile${s.coins >= CONST.coins.packCost ? ' glow' : ''}" data-act="open-standard"><b>Standard Pack</b><span>5 cards · ${CONST.coins.packCost} coins</span></button>`);
  tiles.push(del.ready ? `<button class="tile ready" data-act="claim-delivery"><b>Lab Delivery</b><span>Claim · <span data-tick="claim">${fmtMS(del.claimLeft)}</span></span></button>`
    : `<button class="tile" data-act="skip-delivery"><b>Lab Delivery</b><span>in <span data-tick="delivery">${fmtClock(del.msLeft)}</span>${del.msLeft > CONST.timers.skipMinLeftMs && offers.canShow('skip') ? ' · ▶ Skip the wait' : ''}</span></button>`);
  if (s.gift) tiles.push(`<button class="tile gift" data-act="open-gift"><b>Daily gift · day ${s.streak.days}</b><span>free Standard Pack${s.gift.forced ? ' · ' + rarityName(s.gift.forced) + ' inside' : ''}</span></button>`);
  if (m.session.sponsorTile || (m.lossStreak >= 2 && s.coins < CONST.coins.packCost && offers.canShow('freepack'))) {
    if (!m.session.sponsorTile) { m.session.sponsorTile = true; m.session.placements.freepack = (m.session.placements.freepack || 0) + 1; logEvent('offer.show', { placement: 'freepack', special: false, F: m.F, T: m.T, tile: true }); if (m.segment === 'RESISTANT') toast('Rough run? A Sponsor Pack is on the house.', 'offer'); }
    tiles.push(`<div class="tile sponsor"><b>Sponsor Pack</b><span>Rough run at Rung ${s.ladder.nextRung}. Open a Sponsor Pack on the house?</span><button class="primary sm" data-act="sponsor-accept">▶ Watch & open</button><button class="link grey" data-act="sponsor-decline">No thanks, I'll grind</button></div>`);
  }
  if (m.segment === 'RESISTANT') tiles.push(`<div class="tile meter"><b>Guaranteed Holo in ${CONST.odds.pityEvery - s.packsSincePity} packs</b>${ring(s.packsSincePity / CONST.odds.pityEvery, 34, '')}</div>`); // the pity meter, shown only to the hard-to-monetise
  setHTML(els.screens.home, `
    <div class="topbar"><div class="focus"><div class="pips">${pips}</div><small data-tick="focus">${s.focus < CONST.focus.max && nxt != null ? 'next in ' + fmtMS(nxt) : s.focus < CONST.focus.max ? 'paused' : ''}</small></div>
      <div class="coins" data-hold="lab" title="Hold 1.5 s for the Researcher View"><i class="coin"></i><span data-tick="coins">${s.coins}</span></div>
      <div class="streak" title="Daily streak"><i class="flame"></i>${s.streak.days}d</div><button class="icon" data-act="go:settings" aria-label="Settings">⚙</button></div>
    <div class="rival"><div class="rival-head">${renderPortrait(rv, ++svgSeq)}<div><div class="rival-name">${esc(rv.name)}</div><div class="rival-sub">Rung ${rv.rung} · ${rv.league} League${s.ladder.cleared[rv.rung] ? ' · Rematch · 60% coins' : ''}</div>${rv.specialty ? `<span class="chip sys-${rv.specialty}">${SYSTEMS[rv.specialty].name} specialist</span>` : '<span class="chip">Any System</span>'}<div class="quip">“${esc(rv.quip)}”</div></div></div>${team}</div>
    <button class="battle-btn${s.focus <= 0 ? ' empty' : ''}" data-act="battle">▶ BATTLE <small>${s.focus <= 0 ? '(no Focus · next in <span data-tick="focus2">' + (nxt != null ? fmtMS(nxt) : '--:--') + '</span>)' : '(1 Focus)'}</small></button>
    <div class="row4"><button data-act="go:team">TEAM${teamPip}</button><button data-act="go:ladder">LADDER</button><button data-act="go:binder">BINDER ${ring(binder.ownedCount() / 60, 26, binder.ownedCount())}</button><button data-act="go:shop">SHOP</button></div>
    <div class="tiles">${tiles.join('')}</div>
    <div class="fomo">Action Potential rotates in <span data-tick="rotation">${fmtClock(rotationLeftMs())}</span></div>
    ${bannerHtml()}`);
}
function renderLadder() {
  const s = save; let html = '<div class="head"><button class="back" data-act="go:home">‹ Home</button><h2>Ladder</h2></div><div class="ladder">', league = '';
  for (const rv of RIVALS) {
    if (rv.league !== league) { league = rv.league; html += `<div class="league">${league} League${rv.wall ? ' · wall' : ''}</div>`; }
    const cleared = !!s.ladder.cleared[rv.rung], unlocked = rv.rung <= Math.min(12, s.ladder.best + 1), next = rv.rung === s.ladder.nextRung, seen = s.ladder.seenTeams[rv.rung];
    html += `<button class="rung${cleared ? ' cleared' : ''}${unlocked ? '' : ' locked'}${next ? ' next' : ''}" data-act="rung:${rv.rung}" ${unlocked ? '' : 'disabled'}>${renderPortrait(rv, ++svgSeq)}<div><div class="rival-name">${esc(rv.name)}${rv.rung === s.ladder.best && s.ladder.best > 0 ? ' <span class="pbmark">PB</span>' : ''}</div><div class="rival-sub">Rung ${rv.rung}${rv.specialty ? ' · ' + SYSTEMS[rv.specialty].name : ''}${cleared ? ' · Rematch · 60% coins' : unlocked ? '' : ' · locked'}</div></div><div class="rung-team">${seen ? seen.map((id) => `<span class="dot sys-${BYID[id].sys}"></span>`).join('') : '<span class="q">?</span><span class="q">?</span><span class="q">?</span>'}</div>${cleared ? '<span class="tick">✓</span>' : ''}</button>`;
  }
  html += '</div>'; if (s.ladder.cleared[12]) html += `<div class="fomo">Season 2 in ${fmtClock(rotationLeftMs())}</div>`;
  setHTML(els.screens.ladder, html);
}
let teamDraft = null;
function renderTeam(draft) {
  teamDraft = draft || (teamDraft && teamDraft.length ? teamDraft : save.team.slice()); save.teamPip = false;
  const slots = [0, 1, 2].map((i) => { const id = teamDraft[i]; if (!id) return `<button class="slot empty" data-act="pick:${i}">+ pick</button>`; const c = BYID[id]; return `<button class="slot" data-act="pick:${i}">${miniCard(c)}<div class="slot-info"><b>${esc(c.name)}</b><span class="pw">${cardPower(c, save.collection[id])} pw</span>${sysChip(c.sys, true)}<small>${esc(c.moves[0].name)} · ${esc(c.moves[1].name)}</small></div></button>`; }).join('');
  const grid = ownedCards().sort((a, b) => cardPower(b, save.collection[b.id]) - cardPower(a, save.collection[a.id])).map((c) => `<button class="gcard${teamDraft.includes(c.id) ? ' in' : ''}" data-act="gpick:${c.id}">${miniCard(c)}<span class="pw">${cardPower(c, save.collection[c.id])}</span></button>`).join('');
  const penta = SYS.map((s, i) => { const a = -Math.PI / 2 + i * 2 * Math.PI / 5; return { s, x: 100 + Math.cos(a) * 78, y: 100 + Math.sin(a) * 78 }; });
  const pent = `<svg viewBox="0 0 200 200" class="penta">${penta.map((p, i) => { const q = penta[(i + 1) % 5]; return `<line x1="${p.x.toFixed(1)}" y1="${p.y.toFixed(1)}" x2="${q.x.toFixed(1)}" y2="${q.y.toFixed(1)}" stroke="${sysColor(p.s)}" stroke-width="2.5" marker-end="url(#arr)"/>`; }).join('')}<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#e2e8f0"/></marker></defs>${penta.map((p) => `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="16" fill="${sysColor(p.s)}"/><text x="${p.x.toFixed(1)}" y="${(p.y + 4).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="700" fill="#0b0f1a">${SYSTEMS[p.s].name.slice(0, 3).toUpperCase()}</text>`).join('')}</svg>`;
  setHTML(els.screens.team, `<div class="head"><button class="back" data-act="go:home">‹ Home</button><h2>Team</h2><button class="link" data-act="autobuild">Auto-build</button></div>
    <div class="slots">${slots}</div><div class="team-actions"><button class="primary" data-act="save-team">Save</button><span class="muted">${teamDraft.filter(Boolean).length}/3 · ${sum(teamDraft.filter(Boolean).map((id) => cardPower(BYID[id], save.collection[id])))} power</span></div>
    <div class="grid">${grid}</div>
    <div class="penta-wrap">${pent}<ul class="flav">${SET.elements.map((e) => `<li><b style="color:${e.color}">${e.name}</b> beats ${SYSTEMS[e.beats].name}: ${esc(e.flavor)}</li>`).join('')}</ul></div>`);
}
function renderShop() {
  const s = save, m = model, del = deliveryState();
  setHTML(els.screens.shop, `<div class="head"><button class="back" data-act="go:home">‹ Home</button><h2>Shop</h2><div class="coins"><i class="coin"></i><span data-tick="coins">${s.coins}</span></div></div>
    <div class="tiles">
      <div class="tile big"><b>Standard Pack</b><span>5 cards · slot 5 held for last</span><button class="primary sm" data-act="open-standard" ${s.coins < CONST.coins.packCost ? 'disabled' : ''}>Open · ${CONST.coins.packCost}</button></div>
      <div class="tile big"><b>Lab Delivery</b><span>3 cards, free every 4 h</span>${del.ready ? `<button class="primary sm" data-act="claim-delivery">Claim · <span data-tick="claim">${fmtMS(del.claimLeft)}</span></button>` : `<span class="muted">Ready in <span data-tick="delivery">${fmtClock(del.msLeft)}</span></span>${del.msLeft > CONST.timers.skipMinLeftMs && offers.canShow('skip') ? '<button class="ghost sm" data-act="skip-delivery">▶ Skip the wait</button>' : ''}`}</div>
      ${m.session.sponsorTile ? `<div class="tile big sponsor"><b>Sponsor Pack</b><span>3 cards on the house</span><button class="primary sm" data-act="sponsor-accept">▶ Watch & open</button></div>` : ''}
      ${m.segment === 'RESISTANT' ? `<div class="tile big meter"><b>Pity meter</b><span>Guaranteed Holo in ${CONST.odds.pityEvery - s.packsSincePity} packs</span>${ring(s.packsSincePity / CONST.odds.pityEvery, 40, CONST.odds.pityEvery - s.packsSincePity)}</div>` : ''}
    </div>${bannerHtml()}`);
}
function renderBinder() {
  const rows = binder.rows().map((r) => { const near = r.owned === 11; return `<div class="brow"><div class="brow-head"><b style="color:${sysColor(r.sys)}">${SYSTEMS[r.sys].name}</b>${ring(r.owned / 12, 30, r.owned + '/12')}${near ? '<span class="near pulse">1 away!</span>' : ''}</div><div class="bgrid">${r.cards.map((c) => `<button class="bcard" data-act="detail:${c.id}">${miniCard(c, { owned: !!save.collection[c.id] })}${save.collection[c.id] > 1 ? `<span class="copies">×${save.collection[c.id]}</span>` : ''}</button>`).join('')}</div></div>`; }).join('');
  setHTML(els.screens.binder, `<div class="head"><button class="back" data-act="go:home">‹ Home</button><h2>Binder</h2>${ring(binder.ownedCount() / 60, 34, binder.ownedCount())}</div>${rows}${bannerHtml()}`);
  binder.checkNearComplete();
}
function renderSettings() {
  setHTML(els.screens.settings, `<div class="head"><button class="back" data-act="go:home">‹ Home</button><h2>Settings</h2></div>
    <div class="about">${ABOUT_TEXT}</div>
    <div class="setting"><span>Animation speed</span><div class="seg"><button class="${save.settings.animSpeed === 1 ? 'on' : ''}" data-act="anim:1">1×</button><button class="${save.settings.animSpeed === 0.5 ? 'on' : ''}" data-act="anim:0.5">2×</button></div></div>
    <div class="setting"><span>Researcher View</span><button class="ghost sm" data-act="lab-toggle">${labOpen ? 'Close' : 'Open'}</button></div>
    <div class="setting"><span>Seed</span><input class="seedbox" readonly value="${save.seed}"><button class="ghost sm" data-act="copy-seed">Copy</button></div>
    <div class="setting"><span>Reset everything</span><button class="danger sm" data-act="reset-confirm">Reset</button></div>
    <div class="muted small">Synapse League v${CONST.version} · Set 1: Action Potential · no network, no tracking.</div>`);
}
const ABOUT_TEXT = '<p><b>Synapse League</b> is a teaching simulation of manipulative free-to-play design, built from a 2026 news report about ad-funded mobile games. Every ad is fictional. Nothing leaves this browser. Tap the flask at any time to see the machine.</p>';
function showCardDetail(id) { const c = BYID[id], copies = save.collection[id] || 0; dialog({ html: `<div class="detail">${cardHtml(c, { owned: copies > 0, copies })}<div class="detail-text"><b>${esc(c.name)}</b> · ${rarityName(c.rarity)}<br>${copies ? `${copies} cop${copies === 1 ? 'y' : 'ies'} · LTP +${Math.round((potentiation(copies) - 1) * 100)}% · power ${cardPower(c, copies)}` : 'Not owned'}<br>${c.moves.map((m) => `${esc(m.name)} (${SYSTEMS[m.sys].name} ${m.power}${m.kw ? ', ' + m.kw : ''})`).join(' · ')}<br><i>${esc(c.flavor)}</i></div></div>` }); }
function copyText(text) { try { const p = navigator.clipboard && navigator.clipboard.writeText(text); if (p && p.catch) p.catch(() => { /* clipboard may be unavailable */ }); } catch (err) { /* clipboard may be unavailable */ } }
// Banner (56 px, rotates every 20 s, credits banner revenue per rotation while visible); the current advertiser survives a reload so boot draws nothing
let bannerAdv = null;
function bannerHtml() { if (!bannerAdv) { bannerAdv = (prefs.banner && ADVERTISERS.find((a) => a.key === prefs.banner)) || ads.pickAdvertiser(); prefs.banner = bannerAdv.key; persist(); } return `<div class="banner" aria-label="Advertisement (fictional)">${renderPoster(bannerAdv, false)}<span class="ad-tag">Ad</span></div>`; }
function bannerStart() { if (bannerTimer != null) return; bannerTimer = clock.setTimeout(bannerRotate, CONST.ads.bannerRotateMs); }
function bannerStop() { if (bannerTimer != null) { clock.clearTimeout(bannerTimer); bannerTimer = null; } }
function bannerRotate() {
  bannerTimer = null; if (!(screen === 'home' || screen === 'shop' || screen === 'binder') || ads.current) return;
  bannerAdv = ads.pickAdvertiser(); prefs.banner = bannerAdv.key; if (!document.hidden) addRevenue('banner');
  const b = $('.banner', els.screens[screen]); if (b) { b.classList.add('slide'); after(CONST.anim.stamp, () => { b.innerHTML = renderPoster(bannerAdv, false) + '<span class="ad-tag">Ad</span>'; b.classList.remove('slide'); }); const bb = $('.battle-btn', els.app); if (bb) { bb.classList.add('nudge'); after(CONST.anim.stamp, () => bb.classList.remove('nudge')); } }
  bannerStart();
}
// Sheets and pills (§7.4, A-6.8.3)
function offerShow(p) {
  const c = p.copy;
  if (p.pill) { els.pills.innerHTML = `<div class="pill" data-placement="${p.placement}"><button class="pill-main" data-act="offer-accept">${esc(c.primary)}</button><button class="pill-x" data-act="offer-decline" aria-label="No thanks">✕</button></div>`; return; }
  els.sheets.innerHTML = `<div class="sheet-wrap"><div class="sheet ${p.placement}${p.special ? ' special' : ''}" data-placement="${p.placement}"><h3>${esc(c.head)}</h3>${c.body ? `<p>${esc(c.body)}</p>` : ''}<button class="primary pulse" data-act="offer-accept">${esc(c.primary)}</button>${p.placement === 'double' ? `<div class="dring">${ring(1, 28, '')}</div>` : ''}<div class="decline-slot"></div></div></div>`;
  requestAnimationFrame(() => { const sh = $('.sheet', els.sheets); if (sh) sh.classList.add('open'); });
  if (p.placement === 'double') { const start = clock.now(); p.ringTimer = setInterval(() => { const fg = $('.dring .ring-fg', els.sheets); if (!fg) { clearInterval(p.ringTimer); return; } const left = 1 - (clock.now() - start) / CONST.ads.doubleRingMs; const c = 2 * Math.PI * 11; fg.setAttribute('stroke-dashoffset', (c * (1 - clamp(left, 0, 1))).toFixed(1)); }, 100); }
}
function offerDecline(p) { if (p.pill) return; const slot = $('.decline-slot', els.sheets); if (slot) slot.innerHTML = `<button class="decline" data-act="offer-decline">${esc(p.copy.decline)}</button>`; }
function offerHide(p) { if (p.ringTimer) clearInterval(p.ringTimer); if (p.pill) els.pills.innerHTML = ''; else els.sheets.innerHTML = ''; }
function afterOffer(p, accepted, detail) {
  if (!accepted) { if (p.placement === 'freepack') renderHome(); return; }
  switch (p.placement) {
    case 'double': { const el = $('[data-res="coins"]', els.app); if (el) el.textContent = String(detail.paid); const line = $('.double-line', els.app); if (line) line.textContent = `Doubled to ${detail.paid}`; toast(`+${detail.paid - p.ctx.coins} coins`, 'good'); break; }
    case 'freepack': packReveal(detail); break;
    case 'skip': toast('Lab Delivery is ready!', 'good'); if (screen === 'home') renderHome(); else if (screen === 'shop') renderShop(); break;
    case 'reroll': rerollReveal(detail); break;
    case 'recharge': toast('+3 Focus', 'good'); renderHome(); break;
    case 'streak': toast(`Streak restored: ${save.streak.days} days`, 'good'); renderHome(); break;
  }
}
// Ad overlays (§7.5): DOM "video" = animated CSS shapes + wordmark + progress; fake/late/moving close controls
function adOpen(ad) {
  els.flask.classList.add('hidden');
  if (ad.format === 'rewarded') els.ads.innerHTML = `<div class="ad rewarded" data-format="rewarded"><div class="ad-top"><span class="ad-tag">Ad · fictional</span><span class="reward-ring" data-ring>Reward in 15…</span></div>${renderPoster(ad.advertiser, true)}<div class="progress"><i></i></div><div class="ad-bottom"></div></div>`;
  else els.ads.innerHTML = `<div class="ad interstitial" data-format="interstitial"><div class="ad-top"><span class="ad-tag">Ad · fictional</span><span class="count" data-count>you can close in 5</span></div>${renderPoster(ad.advertiser, true)}<div class="progress loop"><i></i></div><div class="x-layer"></div></div>`;
  els.ads.classList.add('open');
  const bar = $('.progress i', els.ads); if (bar) { bar.style.transitionDuration = (ad.format === 'rewarded' ? CONST.ads.rewardedMs : CONST.ads.interstitialLoopMs) + 'ms'; requestAnimationFrame(() => bar.classList.add('run')); }
  if (ad.format === 'rewarded') ad.ringTimer = setInterval(() => { const r = $('[data-ring]', els.ads); if (!r || ad.unlocked) return; r.textContent = `Reward in ${Math.max(0, Math.ceil((CONST.ads.rewardedMs - (clock.now() - ad.startedAt)) / 1000))}…`; }, 250);
}
function adDecoy(ad) { const b = $('.ad-bottom', els.ads); if (b) b.innerHTML = '<button class="link grey decoy-skip" data-act="ad-skip">Skip? You\'ll lose the reward</button>'; }
function adUnlocked(ad) { const r = $('[data-ring]', els.ads); if (r) r.textContent = 'Reward unlocked ✓'; const b = $('.ad-bottom', els.ads); if (b) b.innerHTML = ''; }
function adClose(ad) { if (ad && ad.ringTimer) clearInterval(ad.ringTimer); els.ads.innerHTML = ''; els.ads.classList.remove('open'); els.flask.classList.toggle('hidden', screen === 'battle'); }
function adCount(ad, n) { const c = $('[data-count]', els.ads); if (c) c.textContent = n > 0 ? `you can close in ${n}` : 'you can close now'; }
function adDecoyX(ad) { const l = $('.x-layer', els.ads); if (l) l.insertAdjacentHTML('beforeend', '<button class="adx decoy tr" data-act="ad-decoy" aria-label="Close ad">✕</button>'); }
function adTrueX(ad, px) { const l = $('.x-layer', els.ads); if (l) l.insertAdjacentHTML('beforeend', `<button class="adx true ${['tl', 'tr', 'bl', 'br'][ad.corner]}" style="width:${px}px;height:${px}px;font-size:${Math.round(px * 0.6)}px" data-act="ad-x" aria-label="Close ad">✕</button>`); }
function adLanding(ad) { els.ads.insertAdjacentHTML('beforeend', `<div class="landing">${renderPoster(ad.advertiser, true)}<p class="muted">${esc(ad.advertiser.tagline)} — ${esc(ad.advertiser.wordmark)}. (Fictional landing page.)</p><div class="landing-x"></div></div>`); }
function adLandingClose(ad) { const x = $('.landing-x', els.ads); if (x) x.innerHTML = '<button class="adx land" data-act="ad-landing-close" aria-label="Back">✕</button>'; }
function adLandingBack(ad) { const l = $('.landing', els.ads); if (l) l.remove(); const xl = $('.x-layer', els.ads); if (xl) xl.innerHTML = ''; }
// Coach marks (tutorial)
function coach(text, targetSel, key) { const t = targetSel ? $(targetSel, els.app) : null; els.coach.innerHTML = `<div class="coach" data-coach="${key}">${esc(text)}</div>`; const c = $('.coach', els.coach); if (t) { const r = t.getBoundingClientRect(), a = els.app.getBoundingClientRect(); c.style.left = Math.max(8, Math.min(a.width - 240, r.left - a.left)) + 'px'; c.style.top = (r.top - a.top - 60) + 'px'; } else { c.style.left = '16px'; c.style.top = '40%'; } }
function coachClear() { els.coach.innerHTML = ''; }
function onboardingResume() { // first launch: About → starter pack → team → tutorial → second pack → Home
  const step = save.onboarding.step;
  if (step === 0) { els.about.classList.add('open'); els.about.innerHTML = `<div class="about-card"><h1>Synapse League</h1>${ABOUT_TEXT}<button class="primary" data-act="about-ok">Got it</button></div>`; return; }
  if (step === 1) { const r = onboarding.starterPack(); packReveal(r, { coach: true }); return; }
  if (step === 2) { onboarding.buildTeam(); renderTeam(save.team.slice()); show('team'); $('.team-actions', els.app).innerHTML = '<button class="primary pulse" data-act="tutorial-fight">Fight!</button>'; coach('This is your trio. Tap a slot to change it later.', '.slots', 'team'); return; }
  if (step === 3) { onboarding.tutorialStart(false); return; }
  if (step === 4) { const r = onboarding.secondPack(); packReveal(r); return; }
  if (step === 5) { onboarding.complete(); goHome(); return; }
}
// Input tracking: idle quit signal, rapid-tap bursts, visibility, session reset (A-6.2.2, A-6.2.3, A-6.2.6)
function onInput(e) {
  if (e && e.target && e.target.closest && e.target.closest('#flask')) return; // the flask is the lab's handle: opening the lab is not play input (A-8.1)
  const now = clock.now(); checkSessionReset(now); model.session.lastInputAt = now; model.session.idleFired = false; armIdle();
  if (e && e.type === 'pointerdown') {
    const interactive = e.target.closest('[data-act],button,a,.card,input,textarea,[data-hold]');
    if (!interactive || e.target.closest('.charge')) {
      rapid.taps.push(now); rapid.taps = rapid.taps.filter((t) => now - t <= CONST.patience.rapidWindowMs);
      if (rapid.taps.length >= CONST.patience.rapidTaps && !rapid.inBurst) { rapid.inBurst = true; model.session.rapidTaps++; addF('rapidTap', CONST.patience.dF.rapidTap); }
      if (rapid.burstTimer) clearTimeout(rapid.burstTimer); rapid.burstTimer = setTimeout(() => { rapid.inBurst = false; rapid.taps = []; }, CONST.patience.rapidWindowMs);
    }
  }
}
function armIdle() { if (idleTimer != null) clock.clearTimeout(idleTimer); idleTimer = clock.setTimeout(() => { idleTimer = null; if (document.hidden || ads.current || offers.pending || model.session.idleFired) return; model.session.idleFired = true; quitSignal('idle'); refreshHome(); }, CONST.patience.idleMs); }
function onVisibility() {
  if (document.hidden) { hiddenAt = clock.now(); persistNow(); return; }
  if (hiddenAt != null && clock.now() - hiddenAt > CONST.patience.quitHiddenMs) { quitSignal('hidden'); refreshHome(); } hiddenAt = null;
}
function refreshHome() { if (!hasDOM || !els.app) return; if (screen === 'home' && save.onboarding.step >= 6 && !LIVE) renderHome(); labRender(); }
function tick() { // display refresh only (countdowns, activeMs); the model's timers live in the clock
  const now = clock.now(); if (!document.hidden && now - model.session.lastInputAt < CONST.patience.idleMs) model.session.activeMs += 1000;
  $$('[data-tick]', els.app).forEach((el) => { const k = el.dataset.tick; if (k === 'focus' || k === 'focus2') { const n = focusNextMs(); el.textContent = save.focus >= CONST.focus.max ? '' : n == null ? 'paused' : (k === 'focus' ? 'next in ' : '') + fmtMS(n); } else if (k === 'rotation') el.textContent = fmtClock(rotationLeftMs()); else if (k === 'delivery') { const d = deliveryState(); if (d.ready) renderHomeIfVisible(); else el.textContent = fmtClock(d.msLeft); } else if (k === 'claim') { const d = deliveryState(); el.textContent = fmtMS(d.claimLeft); } else if (k === 'coins') el.textContent = String(save.coins); });
  if (screen === 'home' && save.focus < CONST.focus.max) { const n = focusNextMs(); if (n === 0) renderHomeIfVisible(); }
}
function renderHomeIfVisible() { if (screen === 'home' && !LIVE) renderHome(); else if (screen === 'shop') renderShop(); }
// Event delegation: one click handler for every [data-act]
function onAct(act, target, e) {
  const [cmd, arg] = act.split(':');
  switch (cmd) {
    case 'go': return go(arg);
    case 'about-ok': els.about.classList.remove('open'); els.about.innerHTML = ''; onboarding.dismissAbout(); return onboardingResume();
    case 'battle': { if (LIVE && !LIVE.finalized) return go('battle'); focusCatchUp(); if (save.focus <= 0) { if (!offers.show('recharge', { msLeft: focusNextMs() || 0 })) toast('Out of Focus — wait for the next one.', 'warn'); return; } return startBattle({ rung: save.ladder.nextRung }); }
    case 'rung': { if (selectRung(+arg)) { renderLadder(); toast(`Next: ${RIVALS[+arg - 1].name}`, ''); } return; }
    case 'pick': { pickerOpen(+arg); return; }
    case 'gpick': { const i = teamDraft.indexOf(arg); if (i >= 0) teamDraft[i] = null; else { const e2 = teamDraft.findIndex((x) => !x); if (e2 >= 0) teamDraft[e2] = arg; else teamDraft[2] = arg; } return renderTeam(teamDraft); }
    case 'autobuild': return renderTeam(autoBuild());
    case 'save-team': { const ids = teamDraft.filter(Boolean); if (ids.length !== 3) return toast('Pick three cards.', 'warn'); if (setTeam(ids)) { toast('Team saved', 'good'); teamDraft = null; goHome(); } return; }
    case 'tutorial-fight': coachClear(); setTeam(teamDraft.filter(Boolean), true); teamDraft = null; return onboardingResume();
    case 'open-standard': { if (save.coins < CONST.coins.packCost) { if (screen !== 'shop') return go('shop'); return toast('Not enough coins — win battles to earn more.', 'warn'); } const r = packs.open('standard', {}); if (r.error) return toast('Not enough coins.', 'warn'); return packReveal(r); }
    case 'open-gift': { if (!save.gift) return; return packReveal(packs.open('gift', { free: true })); }
    case 'claim-delivery': { const r = claimDelivery(); if (r) packReveal(r); return; }
    case 'skip-delivery': { const d = deliveryState(); if (!d.ready && d.msLeft > CONST.timers.skipMinLeftMs) offers.show('skip', { msLeft: d.msLeft }); return; }
    case 'sponsor-accept': return offers.answerStanding('freepack', true, { rung: save.ladder.nextRung });
    case 'sponsor-decline': { offers.answerStanding('freepack', false); return renderHome(); }
    case 'offer-accept': return offers.respond(true);
    case 'offer-decline': return offers.respond(false);
    case 'ad-skip': return dialog({ title: 'Skip and lose it?', buttons: [{ label: 'Keep watching', primary: true }, { label: 'Skip', onClick: () => ads.resolve(false) }] });
    case 'ad-x': return ads.close('x');
    case 'ad-decoy': return ads.close('decoy');
    case 'ad-landing-close': if (!ads.close('landing')) toast('Just a moment…', ''); return;
    case 'ad-learn': return ads.learnMore();
    case 'detail': return showCardDetail(arg);
    case 'anim': save.settings.animSpeed = +arg; save.settings.animSet = true; persist(); return renderSettings();
    case 'lab-toggle': return labToggle();
    case 'copy-seed': { copyText(String(save.seed)); const box = $('.seedbox', els.app); if (box) box.select(); return toast('Seed ' + save.seed, ''); }
    case 'reset-confirm': return dialog({ title: 'Reset everything?', html: '<p>Removes the save, the model, the log and your preferences. Starts onboarding again.</p>', buttons: [{ label: 'Cancel' }, { label: 'Reset', danger: true, onClick: () => { resetAll(); } }] });
    case 'move': return battleAct('move', +arg);
    case 'switch-open': return benchOpen(false);
    case 'switch': return battleAct('switch', arg);
    case 'bench-cancel': return battleChoose(LIVE);
    case 'forfeit': return dialog({ title: 'Forfeit this battle?', html: '<p>Counts as a loss (10 coins), no revive offer, streak reset.</p>', buttons: [{ label: 'Keep fighting', primary: true }, { label: 'Forfeit', danger: true, onClick: () => battleAct('forfeit') }] });
    case 'skip-intro': return skipIntro();
    case 'continue': return battleContinue();
    case 'pack-open': return packTear();
    case 'flip': return packFlip(+arg);
    case 'flip-last': return packFlipLast(true);
    case 'summary-done': return packDone();
    case 'open-another': { const r = packs.open('standard', {}); if (r.error) return goHome(); return packReveal(r); }
    case 'swap-in': return summarySwap(arg);
    case 'lab-close': return labToggle(false);
    case 'lab-pin': labPinned = !labPinned; prefs.pinned = labPinned; persist(); return labRender();
    default: return labAct(cmd, arg, target);
  }
}
function pickerOpen(slot) {
  const d = dialog({ title: `Slot ${slot + 1}`, html: `<div class="picker">${ownedCards().sort((a, b) => cardPower(b, save.collection[b.id]) - cardPower(a, save.collection[a.id])).map((c) => `<button class="gcard${teamDraft.includes(c.id) ? ' in' : ''}" data-pick="${c.id}">${miniCard(c)}<span class="pw">${cardPower(c, save.collection[c.id])}</span></button>`).join('')}</div>`, buttons: [{ label: 'Cancel' }] });
  d.addEventListener('click', (e) => { const b = e.target.closest('[data-pick]'); if (!b) return; const id = b.dataset.pick; const j = teamDraft.indexOf(id); if (j >= 0 && j !== slot) teamDraft[j] = teamDraft[slot] || null; teamDraft[slot] = id; d.remove(); renderTeam(teamDraft); });
}
const ui = {
  init(upgraded) {
    if (!hasDOM || !$('#app')) return;
    els.app = $('#app'); els.screens = {}; $$('.screen', els.app).forEach((s) => { els.screens[s.dataset.screen] = s; });
    els.sheets = $('[data-overlay="sheet"]'); els.modals = $('[data-overlay="modal"]'); els.ads = $('[data-overlay="ad"]'); els.toasts = $('[data-overlay="toast"]'); els.coach = $('[data-overlay="coach"]'); els.about = $('[data-overlay="about"]'); els.pills = $('#pill'); els.flask = $('#flask'); els.lab = $('#lab');
    els.app.addEventListener('click', (e) => { const t = e.target.closest('[data-act]'); if (t && !t.disabled) { e.preventDefault(); onAct(t.dataset.act, t, e); } });
    els.app.addEventListener('pointerdown', onInput); els.app.addEventListener('keydown', onInput);
    let hold = null; els.app.addEventListener('pointerdown', (e) => { if (e.target.closest('[data-hold="lab"]')) hold = setTimeout(() => labToggle(true), 1500); }); ['pointerup', 'pointercancel', 'pointerleave'].forEach((ev) => els.app.addEventListener(ev, () => { if (hold) { clearTimeout(hold); hold = null; } }));
    els.app.addEventListener('pointermove', onPointerMoveHolo); els.app.addEventListener('pointerdown', onPointerMoveHolo);
    els.flask.addEventListener('click', () => labToggle());
    els.lab.addEventListener('pointerdown', (e) => e.stopPropagation()); els.lab.addEventListener('click', (e) => { const t = e.target.closest('[data-act]'); if (t) { e.preventDefault(); onAct(t.dataset.act, t, e); } });
    document.addEventListener('keydown', (e) => { if ((e.key === 'l' || e.key === 'L') && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement && document.activeElement.tagName)) labToggle(); });
    document.addEventListener('visibilitychange', onVisibility); window.addEventListener('pagehide', persistNow); window.addEventListener('beforeunload', persistNow);
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches && !save.settings.animSet) save.settings.animSpeed = 0.5;
    tickTimer = setInterval(tick, 1000); armIdle();
    if (upgraded) toast('Save upgraded — starting fresh', 'warn', 3000);
    labPinned = !!prefs.pinned; if (prefs.lab || /[?&]lab=1/.test(location.search)) labToggle(true, true);
    if (save.onboarding.step < 6) { show('home'); renderHome(); onboardingResume(); } else goHome();
    try { if (window.claude && window.claude.hot && window.claude.hot.ready) window.claude.hot.ready(); } catch (err) { /* optional host hook */ }
  },
  resetAll: U(function () { els.sheets.innerHTML = ''; els.pills.innerHTML = ''; els.ads.innerHTML = ''; els.ads.classList.remove('open'); els.coach.innerHTML = ''; els.modals.innerHTML = ''; els.about.classList.remove('open'); for (const k in els.screens) els.screens[k].innerHTML = ''; bannerStop(); PK = null; teamDraft = null; show('home'); renderHome(); onboardingResume(); labRender(); }),
  go: U(go), refreshHome: U(refreshHome), toast: U(toast), noAnim: U((b) => els.app.classList.toggle('noanim', !!b)),
  clearOnboarding: U(() => { els.about.classList.remove('open'); els.about.innerHTML = ''; coachClear(); PK = null; teamDraft = null; }),
  offerShow: U(offerShow), offerDecline: U(offerDecline), offerHide: U(offerHide), afterOffer: U(afterOffer),
  adOpen: U(adOpen), adDecoy: U(adDecoy), adUnlocked: U(adUnlocked), adClose: U(adClose), adCount: U(adCount), adDecoyX: U(adDecoyX), adTrueX: U(adTrueX), adLanding: U(adLanding), adLandingClose: U(adLandingClose), adLandingBack: U(adLandingBack),
  battleStart: U((L) => battleStart(L)), battleChoreo: U((L, ev, done) => battleChoreo(L, ev, done)), battleChoose: U((L) => battleChoose(L)), battleLossScene: U((L) => battleLossScene(L)), battleEndScene: U((L) => battleEndScene(L)), showDefeat: U((L) => showDefeat(L)), reviveScene: U((L) => reviveScene(L)),
  packSummary: U((r, instant) => packSummary(r, instant)), packReveal: U((r) => packReveal(r)),
  labToggle: U((b) => labToggle(b)),
};

/* ============================================================ 14b. Battle choreography (§3.10) and pack opening choreography (§7.3) */
function runSeq(steps, done) { // each step: {fn, ms}; under skipAnims every duration is 0 and the chain runs synchronously
  let i = 0; const next = () => { if (i >= steps.length) { if (done) done(); return; } const s = steps[i++]; if (s.fn) s.fn(); const d = dur(s.ms || 0); if (d <= 0) next(); else clock.setTimeout(next, d); }; next();
}
function countUp(el, from, to, ms) { if (!el) return; const d = dur(ms); if (d <= 0) { el.textContent = String(to); return; } const t0 = performance.now(); const step = (t) => { const k = Math.min(1, (t - t0) / d); el.textContent = String(Math.round(from + (to - from) * (1 - Math.pow(1 - k, 3)))); if (k < 1) requestAnimationFrame(step); }; requestAnimationFrame(step); }
const bEl = (id) => $('#' + id, els.screens.battle);
function hpCardHtml(side, m, L) {
  const pct = clamp(m.hp / m.HP_max, 0, 1);
  return `<div class="hpc-top"><b>${esc(m.name)}</b>${sysChip(m.sys, true)}<span class="chip warn atkdown${m.weakened ? '' : ' off'}">ATK ↓</span>${side === 'player' && L.B.revived ? '<span class="chip boost">Sponsor Boost active ✦</span>' : ''}</div><div class="bar${pct <= CONST.battle.sliverClass ? ' sliver' : ''}${pct <= 0.25 ? ' low' : ''}"><i class="ghost" style="width:${(pct * 100).toFixed(1)}%"></i><i class="fill" style="width:${(pct * 100).toFixed(1)}%"></i></div><span class="hpnum">${Math.max(0, m.hp)}/${m.HP_max}</span>`;
}
function renderSide(L, side) {
  const S = L.B[side], m = S.team[S.active];
  setHTML(bEl('hp-' + side), hpCardHtml(side, m, L));
  const spr = bEl('spr-' + side); spr.innerHTML = renderSprite(m.card, 160); spr.className = `spr ${side} bob-${m.sys}${alive(m) ? '' : ' fainted'}`;
  setHTML(bEl('bench-' + side), S.team.map((b, i) => (i === S.active ? '' : `<span class="bchip${alive(b) ? '' : ' out'}" title="${esc(b.name)}">${renderSprite(b.card, 28)}<i style="width:${(100 * Math.max(0, b.hp) / b.HP_max).toFixed(0)}%"></i></span>`)).join(''));
}
function caption(text, color) { const c = bEl('caption'); c.textContent = text; c.style.color = color || ''; c.classList.remove('flash'); void c.offsetWidth; c.classList.add('flash'); }
function battleStart(L) {
  const B = L.B, rv = L.rival;
  setHTML(els.screens.battle, `<div class="battle-head"><span class="bh-name">${esc(rv.name)} · Rung ${L.rung}</span>${L.tutorial ? '' : '<button class="link grey" data-act="forfeit">Forfeit</button>'}</div>
    <div class="arena" id="arena"><div class="hpcard opp" id="hp-opp"></div><div class="spr opp" id="spr-opp"></div><div class="bench opp" id="bench-opp"></div><div class="spr player" id="spr-player"></div><div class="hpcard player" id="hp-player"></div><div class="bench player" id="bench-player"></div><div class="fx" id="fx"></div><div class="stamps" id="stamps"></div></div>
    <div class="caption" id="caption"></div><div class="cmd" id="cmd"></div>
    <div class="trainer" id="trainer" data-act="skip-intro">${renderPortrait(rv, ++svgSeq)}<div class="t-name">${esc(rv.name)}</div><div class="t-sub">Rung ${L.rung} · ${rv.league} League</div>${rv.specialty ? `<span class="chip sys-${rv.specialty}">${SYSTEMS[rv.specialty].name} specialist</span>` : '<span class="chip">Any System</span>'}<div class="t-team">${B.opp.team.map((m) => miniCard(m.card, { copies: 1 })).join('')}</div><div class="vs">VS</div><small>tap to skip</small></div>`);
  show('battle'); coachClear();
  renderSide(L, 'opp'); renderSide(L, 'player');
  bEl('spr-opp').classList.add('hidden'); bEl('spr-player').classList.add('hidden');
  L.intro = runSeqCancelable([{ fn: () => bEl('trainer').classList.add('in'), ms: CONST.anim.intro }, { fn: () => bEl('trainer').classList.add('vs'), ms: CONST.anim.vs }, { fn: () => bEl('trainer').classList.add('out') },
    { fn: () => bEl('spr-opp').classList.remove('hidden'), ms: CONST.anim.sendOut }, { fn: () => bEl('spr-player').classList.remove('hidden'), ms: CONST.anim.sendOut }], () => { const t = bEl('trainer'); if (t) t.remove(); bEl('spr-opp').classList.remove('hidden'); bEl('spr-player').classList.remove('hidden'); battleChoose(L); });
}
function runSeqCancelable(steps, done) { // tap-to-skip: the chain stops, the completion runs once
  const h = { cancelled: false, finished: false }; const fin = () => { if (h.finished) return; h.finished = true; done(); };
  runSeq(steps.map((s) => ({ fn: () => { if (!h.cancelled && s.fn) s.fn(); }, ms: s.ms })), () => { if (!h.cancelled) fin(); }); h.skip = () => { if (h.finished) return; h.cancelled = true; fin(); }; return h;
}
function skipIntro() { const L = LIVE; if (L && L.intro) L.intro.skip(); }
function battleChoose(L) {
  const B = L.B; if (B.state === 'END' || L.offerPending) return;
  if (B.state === 'REPLACE') return benchOpen(true);
  const me = B.player.team[B.player.active], you = B.opp.team[B.opp.active], seen = save.seen;
  const hint = (m) => { const e = TYPE_CHART[m.sys][you.sys]; if (!(seen[m.sys] && seen[m.sys][you.sys])) return ''; return e > 1 ? '<span class="hint up">▲ Super effective</span>' : e < 1 ? '<span class="hint down">▼ Not very effective</span>' : ''; };
  setHTML(bEl('cmd'), `<div class="moves">${me.moves.map((m, i) => `<button class="move" data-act="move:${i}"><b>${esc(m.name)}</b><span class="mrow">${sysChip(m.sys, true)}<span class="pw">${m.power}</span>${m.kw ? `<span class="kw">${SET.effects[m.kw].badge}</span>` : ''}</span>${hint(m)}</button>`).join('')}</div><button class="switch" data-act="switch-open" ${aliveCount(B.player.team) > 1 ? '' : 'disabled'}>SWITCH</button>`);
  if (L.tutorial) { if (B.turn === 1 && !L.coach1) { L.coach1 = true; coach('Tap a move', '.move', 'move'); } else if (L.firstFaintSeen && !L.coach3) { L.coach3 = true; coach('Switch when the matchup is bad', '.switch', 'switch'); } }
}
function benchOpen(replace) {
  const L = LIVE, B = L.B, bench = B.player.team.map((m, i) => ({ m, i })).filter((x) => x.i !== B.player.active);
  setHTML(bEl('cmd'), `<div class="bench-sheet"><div class="bs-head">${replace ? 'Send out…' : 'Switch to…'}${replace ? '' : '<button class="link grey" data-act="bench-cancel">Cancel</button>'}</div><div class="bs-cards">${bench.map(({ m, i }) => `<button class="bs-card${alive(m) ? '' : ' out'}" data-act="switch:${m.id}" ${alive(m) ? '' : 'disabled'}>${miniCard(m.card)}<div><b>${esc(m.name)}</b><div class="bar"><i class="fill" style="width:${(100 * Math.max(0, m.hp) / m.HP_max).toFixed(0)}%"></i></div><small>${Math.max(0, m.hp)}/${m.HP_max}</small></div></button>`).join('')}</div></div>`);
  const ff = $('[data-act="forfeit"]', els.screens.battle); if (ff) ff.classList.toggle('hidden', !!replace);
}
function battleAct(kind, arg) { const L = LIVE; if (!L || L.busy) return; coachClear(); skipIntro(); L.busy = true; setHTML(bEl('cmd'), '<div class="cmd-wait"></div>'); const r = act(kind, arg); if (r && r.error) { L.busy = false; battleChoose(L); } }
function dmgNumber(side, text, cls) { const fx = bEl('fx'); const s = document.createElement('span'); s.className = 'dmg ' + side + ' ' + (cls || ''); s.textContent = text; fx.appendChild(s); after(CONST.anim.dmgFloat + 100, () => s.remove()); }
function updateHp(side, m, dmgMs, hp) { // `hp` is the value right after this event (the engine resolved the whole turn already)
  const card = bEl('hp-' + side), cur = hp != null ? hp : m.hp, pct = clamp(Math.max(0, cur) / m.HP_max, 0, 1) * 100, fill = $('.fill', card), ghost = $('.ghost', card), bar = $('.bar', card);
  if (fill) { fill.style.transitionDuration = dur(dmgMs) + 'ms'; fill.style.width = pct.toFixed(1) + '%'; }
  if (ghost) { ghost.style.transitionDuration = dur(CONST.anim.ghost) + 'ms'; ghost.style.transitionDelay = dur(dmgMs) + 'ms'; ghost.style.width = pct.toFixed(1) + '%'; }
  if (bar) { bar.classList.toggle('low', pct <= 25); bar.classList.toggle('sliver', pct <= CONST.battle.sliverClass * 100); }
  const num = $('.hpnum', card); if (num) num.textContent = `${Math.max(0, cur)}/${m.HP_max}`;
  const ad = $('.atkdown', card); if (ad) ad.classList.toggle('off', !m.weakened);
}
function battleChoreo(L, events, done) {
  const B = L.B, steps = [], A = CONST.anim;
  for (const e of events) {
    const side = e.side, other = side === 'player' ? 'opp' : 'player';
    if (e.t === 'switch') steps.push({ fn: () => caption(`${e.from.name}, come back!`), ms: A.stamp }, { fn: () => { renderSide(L, side); bEl('spr-' + side).classList.add('pop'); }, ms: A.sendOut });
    else if (e.t === 'hit') {
      steps.push({ fn: () => { const s = bEl('spr-' + side); s.classList.remove('lunge-l', 'lunge-r'); void s.offsetWidth; s.classList.add(side === 'player' ? 'lunge-r' : 'lunge-l'); }, ms: A.lunge });
      steps.push({ fn: () => { caption(e.move.name, sysColor(e.move.sys)); const t = bEl('spr-' + other); t.classList.remove('hitflash'); void t.offsetWidth; t.classList.add('hitflash'); dmgNumber(other, String(e.dmg), e.crit ? 'crit' : ''); updateHp(other, e.target, Math.min(A.hpDrainMax, e.dmg * A.hpDrainPerDmg), e.targetHp); }, ms: Math.min(A.hpDrainMax, e.dmg * A.hpDrainPerDmg) + A.flash });
      if (e.crit || e.eff !== 1) steps.push({ fn: () => { caption(e.crit ? 'Critical hit!' : e.eff > 1 ? 'Super effective!' : 'Not very effective…'); if (e.crit) { const a = bEl('arena'); a.classList.remove('shake'); void a.offsetWidth; a.classList.add('shake'); } if (L.tutorial && e.eff > 1 && !L.coach2) { L.coach2 = true; coach('Super effective! Motor beats Sensory — every System beats one and is beaten by one', '#hp-opp', 'se'); } }, ms: A.caption });
      if (e.drained) steps.push({ fn: () => { dmgNumber(side, `+${e.drained}`, 'heal'); updateHp(side, e.user, A.tick, e.userHp); }, ms: A.tick });
      if (e.recoil) steps.push({ fn: () => { dmgNumber(side, `−${e.recoil}`, 'recoil'); updateHp(side, e.user, A.tick, e.userHp); }, ms: A.tick });
      if (e.weakened) steps.push({ fn: () => updateHp(other, e.target, 0, e.targetHp) });
    } else if (e.t === 'faint') { steps.push({ fn: () => { bEl('spr-' + side).classList.add('fainted'); caption(`${e.mon.name} fainted`); if (side === 'opp') L.firstFaintSeen = true; }, ms: A.faint }, { ms: A.faintHold }); }
    else if (e.t === 'replace') { if (side === 'opp') steps.push({ ms: A.aiReplace }); steps.push({ fn: () => { renderSide(L, side); const s = bEl('spr-' + side); s.classList.add('pop'); caption(`${side === 'opp' ? L.rival.first + ' sends out' : 'Go,'} ${B[side].team[e.idx].name}!`); }, ms: A.sendOut }); }
  }
  runSeq(steps, () => { L.busy = false; renderSide(L, 'opp'); renderSide(L, 'player'); done(); });
}
function battleLossScene(L) { const a = bEl('arena'); a.classList.add('dim'); renderSide(L, 'opp'); renderSide(L, 'player'); setHTML(bEl('cmd'), ''); }
function battleEndScene(L) {
  const B = L.B; if (B.result === 'win') return victoryScene(L);
  bEl('arena').classList.add('dim'); renderSide(L, 'opp'); setHTML(bEl('cmd'), ''); after(CONST.anim.lossOffer, () => showDefeat(L));
}
function victoryScene(L) {
  const A = CONST.anim, a = bEl('arena'); a.classList.remove('dim'); a.classList.add('bright');
  runSeq([{ fn: () => setHTML(bEl('stamps'), '<div class="stamp win">VICTORY</div>'), ms: A.stamp }, { fn: () => confetti(bEl('fx')), ms: A.confetti / 2 }], () => resultsCard(L));
}
function confetti(fx) { for (let i = 0; i < 24; i++) { const s = document.createElement('span'); const sys = SYS[i % 5]; s.className = 'confetti'; s.style.left = (rng.ui() * 100).toFixed(1) + '%'; s.style.animationDelay = (rng.ui() * 0.4).toFixed(2) + 's'; s.style.color = sysColor(sys); s.textContent = SYSTEMS[sys].name.slice(0, 1); fx.appendChild(s); after(CONST.anim.confetti + 500, () => s.remove()); } }
function resultsCard(L) {
  const m = model, B = L.B, info = L.coinsInfo, streakLine = m.winStreak >= 2 ? `<div class="res-line">×${streakMult(m.winStreak).toFixed(2).replace(/0$/, '')} streak (${m.winStreak} wins)</div>` : '';
  setHTML(bEl('cmd'), `<div class="results"><div class="res-coins">+<span data-res="coins">0</span> coins${info && info.sponsor ? ' · Sponsor Bonus +10%' : ''}</div>${streakLine}<div class="res-line double-line"></div><div class="res-chips"><span class="chip">Rung ${L.rung}${L.rematch ? ' · rematch' : ' · cleared'}</span>${L.newPB ? '<span class="stamp pb">PERSONAL BEST</span>' : ''}</div><button class="primary" data-act="continue">Continue</button></div>`);
  countUp($('[data-res="coins"]', els.screens.battle), 0, L.coins, CONST.anim.coinCount);
  if (!L.tutorial && (L.newPB || [4, 7, 10].includes(L.rung)) && m.phase !== 'ONBOARD' && m.phase !== 'COOLDOWN') after(CONST.anim.coinCount, () => offers.show('double', { rung: L.rung, coins: L.coins, wall: !L.newPB })); // Double it: the first ad a player ever sees
}
function showDefeat(L) {
  const m = model, lost = L.streakBefore; bEl('arena').classList.add('dim'); setHTML(bEl('stamps'), '');
  setHTML(bEl('cmd'), `<div class="results defeat"><div class="stamp loss">Defeat</div><div class="res-coins">+${CONST.coins.loss} coins</div>${lost >= 2 ? `<div class="res-line">Streak lost (×${streakMult(lost).toFixed(2).replace(/0$/, '')} → ×1.0)</div>` : ''}<button class="primary" data-act="continue">Continue</button></div>`);
}
function reviveScene(L) {
  const A = CONST.anim, a = bEl('arena'); a.classList.remove('dim');
  runSeq([{ fn: () => { els.app.classList.add('defib'); }, ms: A.reviveFlash }, { fn: () => { els.app.classList.remove('defib'); renderSide(L, 'player'); bEl('spr-player').classList.remove('fainted'); bEl('spr-player').classList.add('pop'); $$('#hp-player .fill, #hp-player .ghost', els.screens.battle).forEach((f) => { f.style.transitionDuration = dur(A.refill) + 'ms'; }); }, ms: A.refill }], () => { L.busy = false; battleChoose(L); });
}
function battleContinue() { const L = LIVE; if (!L || !L.finalized || screen !== 'battle') return; offers.dismissPending(); setHTML(els.screens.battle, ''); if (L.tutorial) { coachClear(); return onboardingResume(); } goHome(); } // the finished battle leaves no stale controls behind
// Pack opening (§7.3): tear → fan → flips with rarity cues → the held last card with its honest (or baited) charge → Summary
let PK = null;
const PACK_LABEL = { standard: 'Standard Pack', starter: 'Starter Pack', onboard2: 'Welcome Pack', gift: 'Daily Gift', delivery: 'Lab Delivery', sponsor: 'Sponsor Pack' };
function packReveal(r, opts) {
  opts = opts || {}; PK = { r, faceUp: new Set(), last: r.cards.length - 1, charged: false, flippedLast: false, coach: !!opts.coach, autoTimer: null };
  setHTML(els.screens.pack, `<div class="pack-stage" data-state="closed"><div class="head"><h2>${PACK_LABEL[r.type] || 'Pack'}</h2><span class="muted">${r.cards.length} cards</span></div><div class="pack-area"><div class="pack-sprite" data-act="pack-open"><div class="tear"></div><span class="pk-title">Action Potential</span><span class="pk-sub">Synapse League · Set 1</span><small>Tap to open</small></div><div class="fan"></div></div><div class="summary"></div></div>`);
  show('pack'); bannerStop();
  if (settings.skipAnims) { packTear(); }
}
function packTear() {
  if (!PK || PK.opened) return; PK.opened = true; const r = PK.r, stage = $('.pack-stage', els.screens.pack), sprite = $('.pack-sprite', stage), fan = $('.fan', stage), A = CONST.anim;
  runSeq([{ fn: () => sprite.classList.add('squash'), ms: A.packSquash }, { fn: () => { sprite.classList.remove('squash'); sprite.classList.add('up'); }, ms: A.packSlide }, { fn: () => sprite.classList.add('torn'), ms: A.tear },
    { fn: () => { sprite.classList.add('gone'); stage.dataset.state = 'fan'; fan.innerHTML = r.cards.map((c, i) => `<div class="fc${i === PK.last ? ' last locked' : ''}" data-i="${i}" ${i === PK.last ? 'data-act="flip-last"' : `data-act="flip:${i}"`}><div class="inner"><div class="back"><span>AP1</span></div><div class="face">${cardHtml(c, { copies: save.collection[c.id] })}</div></div><div class="cue"></div></div>`).join(''); }, ms: A.fanStagger * 5 }],
    () => { if (PK.coach) coach('Tap a card to flip it', '.fc', 'flip'); if (settings.skipAnims) { for (let i = 0; i < PK.last; i++) packFlip(i); } });
}
function packFlip(i) {
  if (!PK || !PK.opened || PK.faceUp.has(i) || i === PK.last) return; PK.faceUp.add(i); coachClear();
  const c = PK.r.cards[i], el = $(`.fc[data-i="${i}"]`, els.screens.pack), A = CONST.anim; if (!el) return;
  runSeq([{ fn: () => el.classList.add('lift'), ms: A.lift }, { fn: () => el.classList.add('half'), ms: A.flipOut }, { fn: () => { el.classList.remove('half'); el.classList.add('flipped'); }, ms: A.flipIn }], () => { landingCue(el, c); if (PK.faceUp.size >= PK.last) packChargeLast(); });
}
function landingCue(el, c) {
  const A = CONST.anim, cue = $('.cue', el);
  if (c.rarity === 'U') { cue.className = 'cue tick'; after(A.uncommonTick, () => cue.classList.remove('tick')); }
  else if (c.rarity === 'R') { cue.className = 'cue burst'; after(A.rareBurst, () => cue.classList.remove('burst')); }
  else if (c.rarity === 'H') { cue.className = 'cue holo-ring'; const card = $('.card', el); if (card) { $$('.card.holo-live', els.app).forEach((x) => x.classList.remove('holo-live')); card.classList.add('holo-live'); } after(A.holoRing, () => cue.classList.remove('holo-ring')); }
  else if (c.rarity === 'L') { cue.className = 'cue legend'; els.app.classList.add('blackflash'); after(A.legendFlash, () => els.app.classList.remove('blackflash')); legendaryArcs(el, c); confetti($('.pack-area', els.screens.pack)); after(A.legendArcs, () => cue.classList.remove('legend')); }
}
function legendaryArcs(el, c) { // 1.2 s of lightning arcs on a canvas overlay: 6 seeded polylines regenerated every 80 ms, then static
  const cv = document.createElement('canvas'); cv.className = 'arcs'; cv.width = 300; cv.height = 420; el.appendChild(cv); const ctx = cv.getContext('2d'); if (!ctx) return; const r = mulberry32(c.seed ^ 0x77); let frames = Math.round(CONST.anim.legendArcs / 80);
  const draw = () => { ctx.clearRect(0, 0, 300, 420); ctx.strokeStyle = 'rgba(255,230,120,0.9)'; ctx.lineWidth = 2; ctx.shadowColor = '#fbbf24'; ctx.shadowBlur = 8; for (let k = 0; k < 6; k++) { ctx.beginPath(); let x = r() * 300, y = 0; ctx.moveTo(x, y); for (let s = 0; s < 8; s++) { x += (r() - 0.5) * 60; y += 50 + r() * 10; ctx.lineTo(x, y); } ctx.stroke(); } if (--frames > 0) setTimeout(draw, settings.skipAnims ? 0 : 80); else after(400, () => cv.remove()); };
  if (settings.skipAnims) { cv.remove(); return; } draw();
}
function packChargeLast() {
  const el = $('.fc.last', els.screens.pack), c = PK.r.cards[PK.last], A = CONST.anim, bait = PK.r.bait; if (!el) return;
  el.classList.remove('locked'); el.classList.add('center'); const cue = $('.cue', el);
  const hue = bait ? (bait.kind === 'binder' ? 'binder' : 'rainbow') : { C: 'grey', U: 'green', R: 'silver', H: 'rainbow', L: 'gold' }[c.rarity]; // honest hue, except on a bait pack
  const steps = [{ ms: A.sendOut }, { fn: () => { cue.className = 'cue charge ' + hue; if (bait && bait.kind === 'binder') cue.innerHTML = `<div class="ghost-card">${cardHtml(BYID[bait.shown], { owned: false })}</div>`; el.dataset.hue = hue; }, ms: A.charge }];
  if (bait) steps.push({ fn: () => cue.classList.add('glint'), ms: A.glint }, { fn: () => { cue.classList.remove('glint'); cue.classList.add('stall'); }, ms: A.stall }, { fn: () => { cue.classList.remove('stall'); cue.classList.add('deflate'); cue.classList.remove('rainbow', 'binder'); cue.classList.add('grey'); cue.innerHTML = ''; }, ms: A.deflate });
  runSeq(steps, () => { PK.charged = true; el.classList.add('charged'); if (settings.skipAnims) packFlipLast(false); else PK.autoTimer = clock.setTimeout(() => packFlipLast(false), dur(A.autoFlip)); });
}
function packFlipLast(byTap) {
  if (!PK || !PK.charged || PK.flippedLast) return; PK.flippedLast = true; if (PK.autoTimer != null) clock.clearTimeout(PK.autoTimer);
  const el = $('.fc.last', els.screens.pack), c = PK.r.cards[PK.last], A = CONST.anim; void byTap;
  runSeq([{ fn: () => el.classList.add('half'), ms: A.flipOut }, { fn: () => { el.classList.remove('half'); el.classList.add('flipped'); $('.cue', el).className = 'cue'; }, ms: A.flipIn }], () => { landingCue(el, c); after(A.summary, () => packSummary(PK.r, false)); });
}
function packSummary(r, instant) {
  PK = PK && PK.r === r ? PK : { r }; const A = CONST.anim, weak = weakestTeamSlot();
  let best = null; r.cards.forEach((c, i) => { const p = cardPower(c, save.collection[c.id]); if (save.team.length === 3 && !save.team.includes(c.id) && p > weak.power && (!best || p > best.p)) best = { c, p, i }; });
  if (best) save.teamPip = true;
  const badges = r.cards.map((c, i) => { const b = r.badges[i]; return `<div class="sc"><div class="sc-card">${cardHtml(c, { copies: b.copies })}</div><div class="sc-badges">${b.isNew ? '<span class="badge new">NEW</span>' : `<span class="badge ltp">+1 LTP · ${Math.round((b.ltp - 1) * 100)}%</span>`}${isRarePlus(c) ? '<span class="badge rare">✦ Rare+</span>' : ''}</div></div>`; }).join('');
  const html = `<div class="summary-strip open"><div class="sum-cards">${badges}</div><div class="sum-row">${ring(binder.ownedCount() / 60, 36, binder.ownedCount())}<span class="muted">${binder.ownedCount()}/60 in the binder</span>${best ? `<button class="chip swap" data-act="swap-in:${best.c.id}">⚔ +${best.p - weak.power} power — Swap in</button>` : ''}</div>
    <div class="sum-btns">${save.onboarding.step < 6 ? '' : save.coins >= CONST.coins.packCost ? `<button class="ghost" data-act="open-another">Open another · ${CONST.coins.packCost}</button>` : '<button class="ghost" data-act="summary-done">Earn coins</button>'}<button class="primary" data-act="summary-done">Done</button></div></div>`;
  if (instant || !$('.pack-stage', els.screens.pack)) { setHTML(els.screens.pack, `<div class="pack-stage" data-state="summary"><div class="head"><h2>${PACK_LABEL[r.type] || 'Pack'}</h2></div>${html}</div>`); show('pack'); bannerStop(); }
  else { $('.pack-stage', els.screens.pack).dataset.state = 'summary'; setHTML($('.summary', els.screens.pack), html); }
  if (r.bait) { after(A.almost, () => toast('Almost!', 'near')); if (!offers.pending) after(A.almost + 300, () => offers.show('reroll', { fromId: r.bait.actual })); }
  labRender();
}
function summarySwap(id) { const weak = weakestTeamSlot(); if (weak.idx < 0 || !save.collection[id] || save.team.includes(id)) return; const team = save.team.slice(); const out = team[weak.idx]; team[weak.idx] = id; if (setTeam(team)) { toast(`${BYID[id].name} swapped in for ${BYID[out].name}`, 'good'); const b = $('.swap', els.screens.pack); if (b) b.remove(); } }
function rerollReveal(detail) { const r = PK && PK.r; if (!r) return; const i = r.slots.length - 1, c = BYID[detail.id]; r.cards[i] = c; r.badges[i] = { id: c.id, isNew: detail.isNew, copies: detail.copies, ltp: potentiation(detail.copies) }; r.bait = null; packSummary(r, true); toast(`Re-rolled: ${c.name}`, 'good'); }
function packDone() { PK = null; coachClear(); offers.dismissPending(); if (save.onboarding.step < 6) return onboardingResume(); goHome(); }
function onPointerMoveHolo(e) { // only the Holo card under the pointer animates; its shimmer follows the thumb, tilt ≤ 8° (A-5.2)
  if (e.type === 'pointermove' && e.buttons && PK && PK.opened) { const fc = e.target.closest && e.target.closest('.fc'); if (fc && !fc.classList.contains('flipped') && !fc.classList.contains('last')) packFlip(+fc.dataset.i); } // a swipe across the fan flips the cards it crosses
  const card = e.target.closest && e.target.closest('.card.r-H'); $$('.card.holo-live', els.app).forEach((x) => { if (x !== card) { x.classList.remove('holo-live'); x.style.setProperty('--mx', '50%'); x.style.setProperty('--my', '50%'); x.style.transform = ''; } });
  if (!card) return; const r = card.getBoundingClientRect(), mx = clamp((e.clientX - r.left) / r.width, 0, 1), my = clamp((e.clientY - r.top) / r.height, 0, 1);
  card.classList.add('holo-live'); card.style.setProperty('--mx', (mx * 100).toFixed(1) + '%'); card.style.setProperty('--my', (my * 100).toFixed(1) + '%'); card.style.transform = `rotate3d(${(my - 0.5).toFixed(2)}, ${(0.5 - mx).toFixed(2)}, 0, ${(8 * Math.hypot(mx - 0.5, my - 0.5) * 2).toFixed(1)}deg)`;
}

/* ============================================================ 15. Researcher View ("Lab", §8). Read-only: renders model/save/log, draws from no stream, changes state only through the explicit override buttons. */
let labFilter = '', labAboutOpen = false, labConstOpen = false, labRaf = null, labLast = {};
const f2 = (x) => (x == null ? '—' : (+x).toFixed(2));
const pct = (x) => (x == null ? '—' : Math.round(x * 100) + '%');
const LAB_TYPE_CLASS = (t) => (/^(bait|rig\.|kill|inflate)/.test(t) ? 'ev-rig' : /^(offer|ad\.|revenue)/.test(t) ? 'ev-ad' : /^(phase|segment|quit|patience|model)/.test(t) ? 'ev-model' : /^battle/.test(t) ? 'ev-battle' : /^pack/.test(t) ? 'ev-pack' : /^lab/.test(t) ? 'ev-lab' : '');
function fmtLogLine(e) { const t = e.t, mm = Math.floor(t / 60000), ss = Math.floor((t % 60000) / 1000), ms = Math.floor(t % 1000); const data = Object.entries(e.data || {}).filter(([k]) => k !== 'candidates' && k !== 'rig' && k !== 'actions').map(([k, v]) => `${k}=${typeof v === 'object' && v !== null ? JSON.stringify(v) : v}`).join(' '); return `<div class="ev ${LAB_TYPE_CLASS(e.type)}"><span class="ev-t">${pad2(mm)}:${pad2(ss)}.${String(ms).padStart(3, '0')}</span> <span class="ev-ph">[${e.phase}]</span> <b>${e.type}</b> <span class="ev-d">${esc(data)}</span></div>`; }
function battleSentence(lb) {
  if (!lb) return 'No battle yet.'; const st = lb.stats, r = lb.rig, nCrit = st.crits.opp.natural + st.crits.opp.forced, critPct = Math.round(clamp(CONST.battle.critBase * (1 + r.critBias.opp), 0, 0.5) * 1000) / 10;
  const bait = r.baitFired ? `bait fired on turn ${(LOG.entries.find((e) => e.type === 'bait.fire' && e.data.kind === 'battle' && e.seq > (labLast.battleStartSeq || 0)) || { data: {} }).data.turn || '?'}, leaving ${Math.round(st.oppActivePct * 100)}% HP` : r.bait ? 'bait armed but not fired' : 'no bait';
  return `${lb.phase}: opponent generated at tier ${lb.effTier} while the card said rung ${lb.rung}; ${st.nBiased.player} of your ${st.nRolls.player} rolls were ${r.rollBias.player < 0 ? 'worst' : 'best'}-of-two (mean ${f2(st.meanRoll.player) === '—' ? '—' : (+st.meanRoll.player).toFixed(3)} vs opponent ${st.meanRoll.opp == null ? '—' : (+st.meanRoll.opp).toFixed(3)}; honest 0.925); the opponent crit ${nCrit}× at ${critPct}%; AI q = ${r.aiQ}; ${bait}; damage ×${r.sideMult}${lb.revived ? ' after the revive' : ''}.`;
}
function labRender() {
  if (!hasDOM || !els.lab || !labOpen) return;
  if (labRaf) return; labRaf = requestAnimationFrame(() => { labRaf = null; labRenderNow(); });
}
function labRenderNow() {
  const m = model, s = save, k = killTerms(), lb = m.lastBattle, lp = m.lastPack, ses = m.session, inp = m.inputs, raw = inp.raw || {};
  const active = document.activeElement, keepId = active && els.lab.contains(active) ? active.id : null, keepVal = keepId ? active.value : null, keepSel = keepId ? active.selectionStart : null;
  const phases = PHASES.map((p) => `<span class="node${m.phase === p ? ' on' : ''}">${p}</span>`).join('<i>›</i>');
  const bar = (v, max, cls) => `<div class="lbar ${cls || ''}"><i style="width:${(100 * clamp(v / max, 0, 1)).toFixed(1)}%"></i></div>`;
  const killLine = m.T - CONST.patience.killLine, cleanLine = m.T - CONST.patience.cleanLine;
  const decisions = m.decisions.slice(-5).map((d) => (d ? 'A' : 'D')).join('') || '—';
  const offersBy = {}; for (const e of LOG.entries) { if (e.type === 'offer.show') (offersBy[e.data.placement] = offersBy[e.data.placement] || { shown: 0, accepted: 0 }).shown++; if (e.type === 'offer.accept') (offersBy[e.data.placement] = offersBy[e.data.placement] || { shown: 0, accepted: 0 }).accepted++; }
  const adsShown = LOG.entries.filter((e) => e.type === 'ad.show').length, perMin = ses.activeMs >= 60000 ? '$' + (m.revenue.total / (ses.activeMs / 60000)).toFixed(4) : '—';
  const rigRows = lb ? [['tier', `shown rung ${lb.rung} → effective tier ${lb.effTier} (offset ${lb.effTier - lb.rung >= 0 ? '+' : ''}${lb.effTier - lb.rung})`], ['rollBias', `player ${lb.rig.rollBias.player} (mean ${lb.stats.meanRoll.player}, ${lb.stats.nBiased.player}/${lb.stats.nRolls.player} biased) · opp ${lb.rig.rollBias.opp} (mean ${lb.stats.meanRoll.opp}, ${lb.stats.nBiased.opp}/${lb.stats.nRolls.opp} biased)`],
    ['critBias', `player ${lb.rig.critBias.player} (${lb.stats.crits.player.natural} natural) · opp ${lb.rig.critBias.opp} (${lb.stats.crits.opp.natural} natural / ${lb.stats.crits.opp.forced} forced)`], ['aiQ', lb.rig.aiQ], ['bait', lb.rig.bait ? (lb.rig.baitFired ? `fired · HP left ${Math.round(lb.stats.oppActivePct * 100)}%` : 'armed, not fired') : 'off'], ['stall', lb.rig.stall ? (lb.rig.stallFired ? 'fired' : 'armed, not fired') : 'off'], ['mercy', lb.rig.mercy ? (lb.rig.mercyFired ? 'fired' : 'armed') : 'off'],
    ['sideMult', `${lb.rig.sideMult} · ${lb.stats.sideMultHits} inflated hits`], ['result', `${lb.result} (${lb.reason}) in ${lb.turns} turns${lb.stats.escaped ? ' · ESCAPED the kill' : ''}${lb.revived ? ' · revived on turn ' + lb.stats.revivedAtTurn : ''}`],
    ['kill.armed', lb.kill && lb.kill.lossRate != null ? `base ${lb.kill.base} → tier ${lb.kill.effTier} (candidate ${lb.kill.candidate}); rehearsed loss ${pct(lb.kill.lossRate)}, near-miss ${pct(lb.kill.nearMissRate)}, revive-win ${pct(lb.kill.reviveWinRate)}${lb.kill.fallback ? ' · FALLBACK' : ''}${lb.kill.soft ? ' · soft' : ''}` : '—']] : [];
  const packRows = lp ? lp.slots.map((x) => `<tr><td>${x.slot}</td><td>${x.rolledRarity}</td><td>${x.finalRarity}</td><td>${x.reason}</td><td>${esc(BYID[x.id].name)}</td></tr>`).join('') : '';
  const entries = LOG.entries.filter((e) => !labFilter || e.type.includes(labFilter) || JSON.stringify(e.data).includes(labFilter)).slice(-120).reverse();
  const constLeaves = []; (function walk(o, p) { for (const key in o) { const v = o[key]; if (typeof v === 'number') constLeaves.push([p + key, v]); else if (v && typeof v === 'object' && !Array.isArray(v) && p.split('.').length < 4) walk(v, p + key + '.'); } })(CONST, '');
  els.lab.innerHTML = `<div class="lab-head"><b>Researcher View</b><span class="muted">live · read-only</span><button class="ghost sm${labPinned ? ' on' : ''}" data-act="lab-pin" title="Keep open while playing">pin</button><button class="icon" data-act="lab-close" aria-label="Close lab">✕</button></div><div class="lab-body">
  <section class="lp"><h4>1 · Player value</h4><div class="gauge">adValue ${f2(m.adValue)} ${bar(m.adValue, 1, 'g-val')}</div><div class="kv">eCPM $${f2(m.eCPM)} · segment <span class="seg-${m.segment}">${m.segment}</span> · last decisions ${decisions} · affinity ${f2(m.adAffinity)}</div>
    <table class="lt"><tr><td>device (simulated proxy)</td><td>dpr ${raw.dpr} · ${raw.sw}×${raw.sh} · ${raw.cores} cores · mem ${raw.mem == null ? 'n/a' : raw.mem} · UA class ${raw.uaClass} → score ${inp.deviceScore} → tier ${inp.deviceTier}</td></tr><tr><td>region (simulated proxy)</td><td>${esc(raw.tz || '(none)')} · ${esc(raw.lang || '')} → eCPM ${inp.regionECPM} → ${f2(inp.region)}</td></tr><tr><td>hour (simulated proxy)</td><td>${new Date(clock.now()).getHours()}h → ${inp.hourMul}${inp.pinned ? ' (pinned)' : ''}</td></tr><tr><td>A/B buckets</td><td>T0 ${m.experiments.T0} · declineDelay ${m.experiments.declineDelayMs} ms · sliver ${m.experiments.sliverPct} · postAdBoost ${m.experiments.postAdBoost}${m.experiments.pinned ? ' (pinned)' : ''}</td></tr></table></section>
  <section class="lp"><h4>2 · Patience</h4><div class="ft"><div class="ft-bar"><i class="f" style="width:${(100 * m.F / CONST.patience.Fmax).toFixed(1)}%"></i><b class="line kill" style="left:${(100 * clamp(killLine, 0, 12) / 12).toFixed(1)}%" title="kill line T−2.5"></b><b class="line clean" style="left:${(100 * clamp(cleanLine, 0, 12) / 12).toFixed(1)}%" title="clean line T−1"></b><b class="line t" style="left:${(100 * m.T / 12).toFixed(1)}%" title="T"></b></div><div class="kv">F ${f2(m.F)} · T ${f2(m.T)} · kill line ${f2(killLine)} · clean line ${f2(cleanLine)}</div></div>
    <div class="kv">lossTolerance ${f2(m.lossTolerance)} · packTolerance ${f2(m.packTolerance)} (drought cap ${Math.max(1, Math.floor(m.packTolerance))}) · winStreak ${m.winStreak} · lossStreak ${m.lossStreak} · battlesSinceKill ${m.battlesSinceKill}/${k.killEvery} · quit signals ${ses.quitSignals} · rapid-tap bursts ${ses.rapidTaps}</div></section>
  <section class="lp"><h4>3 · Phase</h4><div class="nodes">${phases}</div><div class="kv">${m.phaseBattles} battle(s), ${m.phasePacks} pack(s) in phase · kills ${m.kills} (${m.killsEscaped} escaped) · revives ${ses.revives}/${CONST.segments.reviveCap[m.segment]} · rewarded ${ses.rewarded}/${CONST.ads.rewardedPerSession}</div><div class="pending">${esc(pendingTransition())}</div>
    <div class="kv">next battle rig: ${(() => { const r = nextRigProfile(); return `${r.phase} · tier ${r.effTier} (card says ${r.base}) · rolls ${r.rollBias.player}/${r.rollBias.opp} · crits ${r.critBias.player}/${r.critBias.opp} · aiQ ${r.aiQ} · bait ${r.bait} · stall ${r.stall} · ×${r.sideMult}${r.counterPick ? ' · counter-pick' : ''}`; })()}</div>
    <div class="kv">next pack: ${(() => { const p = packProfileFor('standard'); return `${p.row} row · R/H/L ×${p.oddsMul} · dupBias ${p.dupBias} · exclude needed ${p.excludeNeeded}${p.needed ? ' (' + BYID[p.needed].name + ')' : ''} · forced slot 5 ${p.pity ? 'pity' : p.forced5 || 'none'} · bait ${p.bait}`; })()}</div></section>
  <section class="lp"><h4>4 · Last battle rig</h4><p class="sentence">${esc(battleSentence(lb))}</p>${lb ? `<table class="lt">${rigRows.map(([a, b]) => `<tr><td>${a}</td><td>${esc(String(b))}</td></tr>`).join('')}</table>` : ''}<div class="btns"><button class="ghost sm" data-act="lab-replay" ${lb ? '' : 'disabled'}>Replay fair</button><button class="ghost sm" data-act="lab-rehearse">Rehearse ×200</button><button class="ghost sm" data-act="lab-export-turns" ${lb ? '' : 'disabled'}>Export turns</button></div>${labLast.replay ? `<div class="kv res">Same opponent, fair dice: ${labLast.replay.sameOpponent.result.toUpperCase()} on turn ${labLast.replay.sameOpponent.turn} (${labLast.replay.sameOpponent.replayed} of ${labLast.replay.recorded} actions replayed). Honest-rung opponent: ${labLast.replay.honestRung.result.toUpperCase()}.</div>` : ''}${labLast.rehearse ? `<div class="kv res">Rehearsed ×${labLast.rehearse.n} (${labLast.rehearse.rig.phase}, tier ${labLast.rehearse.rig.effTier}${labLast.rehearse.sampled ? ', 8 regenerated opponents' : ''}): loss ${pct(labLast.rehearse.lossRate)} · near-miss ${pct(labLast.rehearse.nearMissRate)} · sliver ${pct(labLast.rehearse.sliverRate)} · bait ${pct(labLast.rehearse.baitRate)} · stall ${pct(labLast.rehearse.stallRate)} · mean ${labLast.rehearse.meanTurns} turns</div>` : ''}</section>
  <section class="lp"><h4>5 · Last pack rig</h4>${lp ? `<p class="sentence">${esc(lp.sentence)}</p><table class="lt"><tr><th>slot</th><th>rolled</th><th>final</th><th>reason</th><th>card</th></tr>${packRows}</table><div class="kv">${lp.bait ? `bait: shown ${lp.bait.shown === 'Holo' ? 'Holo glint' : BYID[lp.bait.shown].name} → actual ${BYID[lp.bait.actual].name}` : 'no bait'} · pity counter ${lp.packsSincePity}/${CONST.odds.pityEvery}${lp.pity ? ' (pity pack)' : ''} · excluded ${lp.rig.excludeNeeded && lp.needed ? BYID[lp.needed].name : 'nothing'}</div>` : '<p class="muted">No pack yet.</p>'}</section>
  <section class="lp"><h4>6 · Revenue (fictional)</h4><div class="kv">rewarded $${m.revenue.rewarded.toFixed(4)} · interstitial $${m.revenue.interstitial.toFixed(4)} · banner $${m.revenue.banner.toFixed(4)} · <b>total $${m.revenue.total.toFixed(4)}</b></div><div class="kv">ads shown ${adsShown} · ad seconds ${Math.round(ses.adSeconds)} · interstitials ${ses.interstitials}/${CONST.ads.interstitialsPerSession} · active ${Math.round(ses.activeMs / 1000)} s · <b>$ per minute of your attention: ${perMin}</b></div>
    <div class="kv">offers: ${Object.entries(offersBy).map(([p, v]) => `${p} ${v.accepted}/${v.shown}`).join(' · ') || 'none yet'}</div><button class="ghost sm" data-act="lab-about">${labAboutOpen ? 'Hide' : 'About the machine'}</button>${labAboutOpen ? ABOUT_MACHINE(m, offersBy, ses) : ''}</section>
  <section class="lp"><h4>7 · Event log</h4><div class="btns"><input id="lab-filter" placeholder="filter type…" value="${esc(labFilter)}"><button class="ghost sm" data-act="lab-export-log">Export JSON lines</button><button class="ghost sm" data-act="copy-seed">Copy seed</button></div><div class="evlog">${entries.map(fmtLogLine).join('')}</div></section>
  <section class="lp"><h4>8 · Overrides</h4><div class="btns"><select id="lab-phase">${PHASES.filter((p) => p !== 'OFFER').map((p) => `<option ${m.phase === p ? 'selected' : ''}>${p}</option>`).join('')}</select><button class="ghost sm" data-act="lab-set-phase">force phase</button></div><div class="btns"><input id="lab-T" type="number" step="0.5" min="1.5" max="8" value="${m.T}"><button class="ghost sm" data-act="lab-set-T">set T</button><input id="lab-aff" type="number" step="0.05" min="0" max="1" value="${m.adAffinity}"><button class="ghost sm" data-act="lab-set-aff">set adAffinity</button></div>
    <div class="btns"><select id="lab-seg">${['UNKNOWN', 'WATCHER', 'RESISTANT'].map((x) => `<option ${m.segment === x ? 'selected' : ''}>${x}</option>`).join('')}</select><button class="ghost sm" data-act="lab-set-seg">set segment</button><button class="ghost sm" data-act="lab-reset-model">reset model</button><button class="danger sm" data-act="reset-confirm">reset everything</button></div>
    <button class="ghost sm" data-act="lab-const">${labConstOpen ? 'Hide CONST' : 'Edit CONST (not persisted)'}</button>${labConstOpen ? `<div class="consts">${constLeaves.map(([p, v]) => `<label>${p}<input type="number" step="any" value="${v}" data-const="${p}"></label>`).join('')}</div>` : ''}</section></div>`;
  if (keepId) { const el = document.getElementById(keepId); if (el) { el.focus(); if (keepVal != null && el.tagName === 'INPUT' && el.type === 'text') { el.value = keepVal; try { el.setSelectionRange(keepSel, keepSel); } catch (e) { /* ignore */ } } } }
}
const ABOUT_MACHINE = (m, offersBy, ses) => { const live = (p) => (offersBy[p] ? `${offersBy[p].accepted}/${offersBy[p].shown}` : '0/0'); return `<div class="memo"><p><b>Internal memo (fictional studio).</b> Placement ranking with expected acceptance, beside this player's live numbers:</p><ol><li>Revive at the moment of loss — 60% WATCHER / 25% RESISTANT · you: ${live('revive')}</li><li>Double coins after a personal best — 45% · you: ${live('double')}</li><li>Sponsor Pack after two losses — 40% · you: ${live('freepack')}</li><li>Skip the Lab Delivery timer — 30% · you: ${live('skip')}</li><li>Re-roll the duplicate — 35% · you: ${live('reroll')}</li><li>Recharge Focus — 50% · you: ${live('recharge')}</li><li>Restore streak — 55% · you: ${live('streak')}</li></ol><p>KPI targets: ARPDAU $0.11 (you: $${m.revenue.total.toFixed(4)}), 4.2 ads/DAU (you: ${LOG.entries.filter((e) => e.type === 'ad.show').length}), rewarded acceptance ≥ 55%, median session 7.5 min (you: ${(ses.activeMs / 60000).toFixed(1)} min).</p><p>Mechanics recognised as: player-value model → price discrimination; moreCleanSuccessRound → contrast effect; boost then kill → negative prediction error; bait → near-miss effect; revive at loss → negative reinforcement at peak aversion; patience estimator → adaptive staircase; segmentation → differential reinforcement; inflated post-ad rewards → reward-magnitude confound; cadence/caps → scarcity; timers → temporal discounting; FOMO countdowns → scarcity heuristic; streaks → sunk cost; collection completion → Zeigarnik/goal gradient (the rung 10–12 endgame needs Legendary teams with copies); confirmshaming → loss framing; fake/late/moving close → cost of escape; strategic pity timer → visible goal gradient for the hard-to-monetise; variable ratio → Skinner's VR schedule; illusion of control → Langer; A/B thresholds → experimentation on frustration.</p></div>`; };
function labToggle(open, silent) {
  const want = open == null ? !labOpen : !!open; if (want === labOpen) { labRender(); return labOpen; }
  labOpen = want; prefs.lab = labOpen; persist(); els.lab.classList.toggle('open', labOpen); els.app.classList.toggle('lab-open', labOpen);
  if (!silent) logEvent(labOpen ? 'lab.open' : 'lab.close', {}); // the only events the lab itself emits
  if (labOpen) labRenderNow(); return labOpen;
}
function exportModal(title, text) { dialog({ title, html: `<textarea class="export" readonly>${esc(text)}</textarea>`, buttons: [{ label: 'Close' }] }); copyText(text); }
function labAct(cmd, arg, target) {
  const m = model;
  switch (cmd) {
    case 'lab-replay': labLast.replay = replayFair(); return labRenderNow();
    case 'lab-rehearse': labLast.rehearse = rehearse(CONST.REHEARSE_N); return labRenderNow();
    case 'lab-export-turns': return exportModal('Last battle turns (JSON lines)', (m.lastBattle ? LOG.entries.filter((e) => e.type === 'battle.turn' && e.seq > (m.lastBattle.firstSeq || 0)).slice(-m.lastBattle.turns) : []).map((e) => JSON.stringify(e.data)).join('\n') || JSON.stringify(m.lastBattle));
    case 'lab-export-log': return exportModal('Event log (JSON lines)', LOG.entries.map((e) => JSON.stringify(e)).join('\n'));
    case 'lab-about': labAboutOpen = !labAboutOpen; return labRenderNow();
    case 'lab-const': labConstOpen = !labConstOpen; return labRenderNow();
    case 'lab-set-phase': { const v = $('#lab-phase', els.lab).value; GAME.setPhase(v); return labRenderNow(); }
    case 'lab-set-T': { const v = parseFloat($('#lab-T', els.lab).value); if (!isNaN(v)) { logEvent('lab.override', { what: 'T', value: v }); setT(v, 'lab'); } return labRenderNow(); }
    case 'lab-set-aff': { const v = parseFloat($('#lab-aff', els.lab).value); if (!isNaN(v)) { logEvent('lab.override', { what: 'adAffinity', value: v }); m.adAffinity = clamp(v, 0, 1); recompute('override'); } return labRenderNow(); }
    case 'lab-set-seg': { GAME.setSegment($('#lab-seg', els.lab).value); return labRenderNow(); }
    case 'lab-reset-model': { logEvent('lab.override', { what: 'resetModel' }); const keep = { inputs: m.inputs, experiments: m.experiments, lastBattle: m.lastBattle, lastPack: m.lastPack }; model = Object.assign(freshModel(save.seed, clock.now()), keep); recompute('override'); persist(); refreshHome(); return labRenderNow(); }
  }
  void target;
}
if (hasDOM) document.addEventListener('input', (e) => { // lab inputs: filter box and CONST editor (A-8.2)
  const t = e.target; if (!els.lab || !els.lab.contains(t)) return;
  if (t.id === 'lab-filter') { labFilter = t.value; labRender(); }
  else if (t.dataset.const) { const v = parseFloat(t.value); if (isNaN(v)) return; const path = t.dataset.const.split('.'); let o = CONST; for (let i = 0; i < path.length - 1; i++) o = o[path[i]]; o[path[path.length - 1]] = v; logEvent('lab.override', { what: 'const.' + t.dataset.const, value: v }); }
});
logSubs.push(() => { if (labOpen) labRender(); });

/* ============================================================ 2b. Onboarding script (§2.3): both the clicked-through path and finishOnboarding() run these steps */
const onboarding = {
  step: () => save.onboarding.step,
  dismissAbout() { if (save.onboarding.step !== 0) return; logEvent('session.start', { seed: save.seed, createdAt: save.createdAt, fresh: true }); save.onboarding.step = 1; persist(); },
  starterPack() { if (save.onboarding.step !== 1) return null; const r = packs.open('starter', { free: true }); save.onboarding.step = 2; persist(); return r; },
  buildTeam() { if (save.onboarding.step !== 2) return; setTeam(autoBuild(), true); logEvent('team.change', { from: [], to: save.team.slice(), auto: true }); save.onboarding.step = 3; persist(); },
  tutorialStart(headless) { if (save.onboarding.step !== 3) return null; return startBattle({ phase: 'ONBOARD', rung: 1, headless }); },
  tutorialDone() { if (save.onboarding.step === 3) { save.onboarding.step = 4; persist(); } },
  secondPack() { if (save.onboarding.step !== 4) return null; const r = packs.open('onboard2', { free: true }); save.onboarding.step = 5; persist(); return r; },
  complete() { if (save.onboarding.step !== 5) return; save.onboarding.step = 6; model.session.actionsSinceInterstitial = 0; save.ladder.nextRung = nextRungDefault(); enterPhase('CLEAN', 'onboarding complete', { cleanLen: 2 }); persist(); },
};
function finishOnboarding() {
  if (save.onboarding.step >= 6) return false;
  const wasSkip = settings.skipAnims; settings.skipAnims = true;
  try {
    onboarding.dismissAbout(); onboarding.starterPack(); onboarding.buildTeam();
    if (save.onboarding.step === 3) { onboarding.tutorialStart(true); autoBattle('greedy'); }
    onboarding.secondPack(); onboarding.complete();
  } finally { settings.skipAnims = wasSkip; }
  ui.clearOnboarding(); ui.go('home'); return true;
}

/* ============================================================ 16. Debug / test API (§10, A-10) — always attached, harmless in normal play */
const PHASES = ['ONBOARD', 'CLEAN', 'NORMAL', 'KILL', 'OFFER', 'REWARD', 'COOLDOWN'];
function getState() {
  const s = deepClone(save), team = (save.team || []).map((id) => { const c = BYID[id], copies = save.collection[id] || 1; return { id, name: c.name, rarity: c.rarity, sys: c.sys, copies, power: cardPower(c, copies), ltp: potentiation(copies) }; });
  const rv = RIVALS[clamp(save.ladder.nextRung, 1, 12) - 1];
  return Object.assign(s, { team, teamIds: save.team.slice(), power: sum(team.map((t) => t.power)), powerTier: team.length ? powerTier(currentTeamCards()) : null, nextRival: { rung: rv.rung, name: rv.name, league: rv.league, specialty: rv.specialty }, focusNextMs: focusNextMs(), onboardingStep: save.onboarding.step });
}
function getModel() { const k = killTerms(); return Object.assign(deepClone(model), { killReady: k.ready, killArmed: k.armed, cleanReady: cleanReady(), pendingTransition: pendingTransition(), rigProfile: nextRigProfile(), packProfile: packProfileFor('standard'), regenMs: regenMs() }); }
function resetAll(newSeed) {
  for (const k of Object.values(KEYS)) storage.remove(k);
  const now = clock.now(), seed = (newSeed != null ? newSeed : Date.now()) >>> 0;
  if (LIVE) LIVE = null; offers.pending = null; ads.current = null; clock.clearAll();
  save = freshSave(seed, now); model = freshModel(seed, now); LOG.entries = []; LOG.seq = 0; prefs = { v: 1, lab: prefs.lab, pinned: prefs.pinned, banner: prefs.banner };
  seedStreams(seed); recompute(); scheduleFocus(); ui.resetAll(); persist();
}
const GAME = {
  version: CONST.version, CONST, TIER_POWER: CONST.TIER_POWER, cards: CARDS, rivals: RIVALS, engine: { expectedDamage, score, aiChoose, aiReplace, makeMon, bestMove },
  art: { renderCard, renderSprite, renderGlyph, renderPortrait, renderPoster }, rngCursors: () => ({ battle: rng.battle.calls, pack: rng.pack.calls, ui: rng.ui.calls }),
  seed(n) { n = n >>> 0; save.seed = n; seedStreams(n); if (!model.experiments.pinned) model.experiments = deriveExperiments(n); logEvent('lab.override', { what: 'seed', value: n }); persist(); return n; },
  reset() { resetAll(); return true; },
  finishOnboarding,
  now: () => clock.now(), fastForward(ms) { clock.fastForward(ms); ui.refreshHome(); return clock.now(); }, skipTimers() { clock.skipTimers(); ui.refreshHome(); return clock.now(); }, setNow(t) { clock.setNow(t); return clock.now(); },
  freezeTime(b) { return clock.freeze(b !== false); }, // tests: the virtual clock moves only through fastForward (exact timing assertions)
  skipAnims(b) { settings.skipAnims = b !== false; ui.noAnim(settings.skipAnims); return settings.skipAnims; },
  getState, getModel,
  getLog(filter) { return LOG.entries.filter((e) => !filter || e.type.startsWith(filter)).map(deepClone); },
  onEvent(cb) { logSubs.push(cb); return () => { const i = logSubs.indexOf(cb); if (i >= 0) logSubs.splice(i, 1); }; },
  setPhase(name) { if (!PHASES.includes(name)) return false; logEvent('lab.override', { what: 'phase', value: name }); if (name === 'OFFER' && !(LIVE && LIVE.offerPending)) name = 'KILL'; enterPhase(name, 'lab'); ui.refreshHome(); return model.phase; },
  setModel(partial) {
    partial = partial || {}; logEvent('lab.override', { what: 'model', value: Object.keys(partial) });
    for (const k in partial) {
      const v = partial[k];
      if (k === 'inputs') { if (v === null) model.inputs = readInputs(); else { Object.assign(model.inputs, v); model.inputs.pinned = true; } }
      else if (v && typeof v === 'object' && !Array.isArray(v) && model[k] && typeof model[k] === 'object') Object.assign(model[k], v);
      else model[k] = v;
    }
    if (partial.F != null) model.F = clamp(model.F, 0, CONST.patience.Fmax); if (partial.T != null) model.T = clamp(model.T, CONST.patience.Tmin, CONST.patience.Tmax);
    recompute('override'); persist(); ui.refreshHome(); return getModel();
  },
  setSegment(s) { if (!['UNKNOWN', 'WATCHER', 'RESISTANT'].includes(s)) return false; logEvent('lab.override', { what: 'segment', value: s }); if (s !== model.segment) logEvent('segment.change', { from: model.segment, to: s, lab: true }); model.segment = s; scheduleFocus(); persist(); ui.refreshHome(); return s; },
  setExperiment(partial) { Object.assign(model.experiments, partial || {}); model.experiments.pinned = true; if (partial && partial.T0 != null && save.battlesPlayed === 0) model.T = partial.T0; logEvent('lab.override', { what: 'experiments', value: partial }); persist(); return deepClone(model.experiments); },
  giveCards(x) { const add = Array.isArray(x) ? Object.fromEntries(x.map((id) => [id, 1])) : x || {}; for (const id in add) if (BYID[id]) save.collection[id] = (save.collection[id] || 0) + (add[id] | 0 || 1); logEvent('lab.override', { what: 'giveCards', value: add }); binder.checkNearComplete(); persist(); ui.refreshHome(); return deepClone(save.collection); },
  give(o) { o = o || {}; if (o.coins != null) save.coins += o.coins; if (o.focus != null) { save.focus = clamp(save.focus + o.focus, 0, CONST.focus.max); scheduleFocus(); } logEvent('lab.override', { what: 'give', value: o }); persist(); ui.refreshHome(); return { coins: save.coins, focus: save.focus }; },
  setTeam(ids) { const ok = setTeam(ids); if (ok) logEvent('lab.override', { what: 'team', value: ids }); ui.refreshHome(); return ok; },
  openPack(opts) {
    opts = opts || {}; const r = packs.open(opts.type || 'standard', { free: !!opts.free }); if (r.error) return r;
    if (opts.instant || settings.skipAnims || !hasDOM) ui.packSummary(r, true); else ui.packReveal(r);
    return { cards: r.cards.map((c) => c.id), slots: r.slots, rig: r.rig, bait: r.bait, badges: r.badges, pity: r.pity, rarePlus: r.rarePlus, newCount: r.newCount, sentence: r.sentence };
  },
  startBattle(opts) { return startBattle(opts || {}); },
  act(kind, arg) { const wasSkip = settings.skipAnims; settings.skipAnims = true; try { return act(kind, arg); } finally { settings.skipAnims = wasSkip; } },
  autoBattle(policy) { return autoBattle(policy); },
  getBattle() { return LIVE ? battleView(LIVE.B) : null; },
  replayFair, rehearse,
  triggerAd(format) { if (format === 'interstitial') return ads.showInterstitial('debug', () => ui.refreshHome()); return ads.showRewarded('debug', { onComplete: () => ui.refreshHome(), onEarly: () => ui.refreshHome() }); },
  resolveAd(watched) { return ads.resolve(watched !== false); },
  closeAd(method) { return ads.close(method || 'x'); },
  respondOffer(accept) { return offers.respond(accept !== false); },
  simulateQuit(kind) { quitSignal(kind || 'hidden'); ui.refreshHome(); return model.phase; },
  lab(b) { return ui.labToggle(b); },
  goto(screen) { ui.go(screen); return screen; },
  rigProfile: () => nextRigProfile(), packProfile: () => packProfileFor('standard'), fairRig: (rung) => fairRig(rung || save.ladder.nextRung),
  powerTier: (ids) => powerTier((ids || save.team).map((id) => BYID[id])),
  genOpponent(o) { o = o || {}; return genOpponent({ rung: o.rung || 1, effTier: o.effTier != null ? o.effTier : o.rung || 1, seed: o.seed != null ? o.seed : save.seed, battlesPlayed: o.battlesPlayed != null ? o.battlesPlayed : save.battlesPlayed, candidate: o.candidate || 0, playerTeam: (o.player || save.team).map((id) => BYID[id]), counterPick: !!o.counterPick, count: o.count || 3 }).map((c) => c.id); },
  simulatePacks: (n, o) => packs.simulate(n, o), simulateInterstitials: (n) => ads.simulateInterstitials(n),
  pendingOffer: () => (offers.pending ? { placement: offers.pending.placement, special: offers.pending.special, pill: offers.pending.pill } : null),
  currentAd: () => (ads.current ? { format: ads.current.format, advertiser: ads.current.advertiser.key, placement: ads.current.placement, trueX: ads.current.trueX, decoy: ads.current.decoy, landing: ads.current.landing } : null),
};
win.GAME = GAME;

/* ============================================================ 17. Boot (A-9.4) */
function boot() {
  const now = Date.now(); let upgraded = false;
  const raw = {}; for (const k in KEYS) raw[k] = storage.raw(KEYS[k]);
  const parsed = {}; let bad = false;
  for (const k in KEYS) { if (raw[k] == null) continue; try { parsed[k] = JSON.parse(raw[k]); if (!parsed[k] || parsed[k].v !== 1) bad = true; } catch (e) { bad = true; } }
  if (bad || !parsed.save || !parsed.model) { if (bad) upgraded = true; resetAll(); }
  else {
    save = Object.assign(freshSave(parsed.save.seed, now), parsed.save); model = Object.assign(freshModel(parsed.save.seed, now), parsed.model);
    if (parsed.log) { LOG.seq = parsed.log.seq || 0; LOG.entries = parsed.log.entries || []; } if (parsed.prefs) prefs = Object.assign(prefs, parsed.prefs);
    seedStreams(save.seed, save.rng);
    if (!model.inputs.pinned) model.inputs = readInputs();
    checkSessionReset(now); model.session.lastInputAt = now;
    focusCatchUp(); deliveryState(); resumeLiveBattleOnBoot();
    if (model.phase === 'KILL' || model.phase === 'OFFER') quitSignal('reload');
    recompute('boot'); logEvent('session.start', { seed: save.seed, resumed: true, phase: model.phase }); scheduleFocus();
  }
  ui.init(upgraded);
}
if (hasDOM) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot(); } else boot();
})();
