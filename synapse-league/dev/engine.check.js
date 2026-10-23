// Headless engine checks (node, no DOM): loads ../cards.js + ../game.js into a vm sandbox, runs the smoke sequence and the
// statistical acceptance items. Run: node dev/engine.check.js
const fs = require('fs'), vm = require('vm'), path = require('path');
const GAME_DIR = path.join(__dirname, '..');
const code = fs.readFileSync(path.join(GAME_DIR, 'cards.js'), 'utf8') + '\n' + fs.readFileSync(path.join(GAME_DIR, 'game.js'), 'utf8');
const mem = {}; const localStorage = { getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); }, removeItem: (k) => { delete mem[k]; } };
const ctx = { console, setTimeout, clearTimeout, Date, Math, JSON, Intl, localStorage, navigator: { userAgent: 'node', hardwareConcurrency: 4, language: 'en-US' }, screen: { width: 390, height: 844 }, devicePixelRatio: 2, performance };
ctx.window = ctx; ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(code + '\n;globalThis.CARD_SET = CARD_SET;', ctx, { filename: 'game.js' });
const G = ctx.GAME;
let fails = 0; const ok = (cond, msg) => { if (!cond) { fails++; console.log('  FAIL', msg); } else console.log('  ok  ', msg); };
const approx = (a, b, tol) => Math.abs(a - b) <= tol;

console.log('== smoke sequence (A-10.6)');
G.reset(); G.seed(1); G.setExperiment({ T0: 4.0, declineDelayMs: 1200, sliverPct: 0.05, postAdBoost: 1.10 }); G.setModel({ inputs: { deviceTier: 0.7, region: 1 / 3, hourMul: 0.7 } }); G.skipAnims(true);
G.finishOnboarding();
let st = G.getState(), md = G.getModel();
ok(md.phase === 'CLEAN' && md.cleanLen === 2, `after onboarding phase ${md.phase} cleanLen ${md.cleanLen}`);
ok(st.coins === 40 && st.ladder.best === 1 && st.focus === 5, `coins ${st.coins} best ${st.ladder.best} focus ${st.focus}`);
ok(st.packsOpened === 2 && st.packsSincePity === 0 && st.packsSinceRarePlus === 0, `packsOpened ${st.packsOpened} sincePity ${st.packsSincePity} sinceRare ${st.packsSinceRarePlus}`);
const packLog = G.getLog('pack.end');
ok(packLog[0].data.ids.join() === 'exe01,mem01,sen01,lim06,mot09', 'starter pack ids ' + packLog[0].data.ids.join());
ok(ctx.CARD_SET.cards.find((c) => c.id === packLog[1].data.ids[4]).rarity === 'H', 'second pack slot 5 is Holo: ' + packLog[1].data.ids[4]);
ok(md.winStreak === 1 && md.battlesSinceKill === 1 && md.newPBLast === true && st.battlesPlayed === 1, `tutorial bookkeeping winStreak ${md.winStreak} bsk ${md.battlesSinceKill} newPB ${md.newPBLast}`);
for (let i = 0; i < 4; i++) G.openPack({ instant: true, free: true });
st = G.getState();
const top3 = Object.keys(st.collection).map((id) => ({ id, p: G.CONST.TIER_POWER && 0 })).map((x) => { const c = ctx.CARD_SET.cards.find((c) => c.id === x.id); const r = ctx.CARD_SET.rarities.find((r) => r.key === c.rarity); return { id: x.id, p: Math.round(r.budget * (1 + 0.04 * Math.min(st.collection[x.id] - 1, 5))) }; }).sort((a, b) => b.p - a.p).slice(0, 3).map((x) => x.id);
ok(G.setTeam(top3), 'setTeam ' + top3.join());
console.log('  powerTier', G.powerTier(top3), 'TIER_POWER', JSON.stringify(G.TIER_POWER));
ok(G.TIER_POWER[5] === 254 && G.TIER_POWER[7] === 306 && G.TIER_POWER[12] === 393 && G.TIER_POWER[0] === 183 && G.TIER_POWER[14] === 440, 'TIER_POWER matches A-3.8.2');
ok(G.powerTier(['mot09', 'lim06', 'exe01']) === 5 && G.powerTier(['lim06', 'exe06', 'mot07']) === 5 && G.powerTier(['mot09', 'sen09', 'mem10']) === 7 && G.powerTier(['exe11', 'lim11', 'mem11']) === 8 && G.powerTier(['exe12', 'lim12', 'sen12']) === 10, 'powerTier examples');
G.setPhase('KILL');
const t0 = Date.now(); const bid = G.startBattle({ rung: st.ladder.best + 1 }); const armMs = Date.now() - t0;
const armed = G.getLog('kill.armed').pop();
console.log('  kill.armed', JSON.stringify({ lossRate: armed.data.lossRate, near: armed.data.nearMissRate, rev: armed.data.reviveWinRate, chosen: armed.data.chosen, base: armed.data.base, fallback: armed.data.fallback, ms: armMs }));
ok(armed.data.lossRate >= 0.85 && armed.data.reviveWinRate >= 0.85 && !armed.data.fallback, 'arming qualifies');
ok(armed.data.candidates.length === 6, 'six candidates');
ok(armMs < 1000, 'arming < 1 s: ' + armMs + ' ms');
let bt = G.getBattle();
ok(bt.rig.aiQ === 1 && bt.rig.bait && bt.rig.stall && bt.rig.rollBias.player === -0.6 && bt.rig.rollBias.opp === 0.6, 'KILL rig knobs');
let r = G.autoBattle();
console.log('  autoBattle', JSON.stringify(r));
ok(r.result === 'loss' && r.offer === true, 'KILL loss with offer');
ok(G.getLog('offer.show').some((e) => e.data.placement === 'revive'), 'offer.show revive logged');
ok(G.getModel().phase === 'OFFER', 'phase OFFER');
G.respondOffer(true); ok(G.currentAd() && G.currentAd().format === 'rewarded', 'rewarded ad open');
G.resolveAd(true);
bt = G.getBattle();
ok(bt.player.team.every((m) => m.hp === m.HP_max), 'team revived at full HP');
ok(approx(bt.rig.sideMult, 1.10, 1e-9), 'sideMult 1.10');
ok(G.getLog('inflate.apply').some((e) => e.data.kind === 'damage'), 'inflate.apply damage logged');
ok(G.getModel().phase === 'REWARD', 'phase REWARD after ad');
r = G.autoBattle(); console.log('  revived autoBattle', JSON.stringify(r));
ok(r.result === 'win', 'revived battle won');
md = G.getModel(); ok(md.phase === 'REWARD' && md.reward.actions === 2 && md.reward.battlesLeft === 1, `REWARD counters actions ${md.reward.actions} battlesLeft ${md.reward.battlesLeft}`);
ok(G.getLog('inflate.apply').some((e) => e.data.what === 'sponsorBonus'), 'sponsor bonus coins logged');
ok(md.kills === 1 && md.battlesSinceKill === 0, 'kills counted');
const pk = G.openPack({ instant: true, free: true });
ok('RHL'.includes(pk.slots[4].finalRarity), 'REWARD pack slot 5 Rare+: ' + pk.slots[4].finalRarity);
ok(approx(G.rigProfile().sideMult, 1.10, 1e-9), 'next rigProfile sideMult still 1.10');
G.startBattle({ rung: 2 }); r = G.autoBattle(); if (r.offer) { G.respondOffer(false); }
ok(G.rigProfile().sideMult === 1, 'rigProfile sideMult back to 1 after that');
console.log('  phase now', G.getModel().phase, G.getModel().pendingTransition);

console.log('== fixtures (A-11.5)');
G.reset(); G.seed(1); G.setExperiment({ T0: 4.0, declineDelayMs: 1200, sliverPct: 0.05, postAdBoost: 1.10 }); G.skipAnims(true); G.finishOnboarding();
G.startBattle({ player: ['mot09'], opp: ['sen02'], phase: 'KILL', rig: { critBias: { player: -1, opp: -1 } }, seed: 7 });
let tr = G.act('move', 0); bt = G.getBattle();
const bf = G.getLog('bait.fire').filter((e) => e.data.kind === 'battle').pop();
ok(bf && bf.data.turn === 1, 'bait.fire turn 1'); ok(bt.opp.active.hp === 4, 'opp left at 4: ' + bt.opp.active.hp);
ok(bt.rig.forceCritNext === true, 'forceCritNext armed after the shave');
const hit = tr.lastTurn.actions.find((a) => a.side === 'player'); ok(hit.shaved <= 0.35 * (hit.shaved + hit.dmg), 'shave ≤ 35%');
tr = G.act('move', 0); ok(tr.result === 'win', 'fixture win'); ok(!G.getLog('kill.escaped').length, 'no kill.escaped outside KILL phase');
ok(G.getBattle().rig.forceCritNext === false && G.getLog('rig.forcedCrit').some((e) => e.data.applied === false), 'forcedCrit consumed (applied:false)');
G.startBattle({ player: ['exe01'], opp: ['sen02'], phase: 'KILL', rig: { critBias: { player: -1, opp: -1 } }, seed: 7 }); G.act('move', 0);
ok(G.getLog('bait.fire').filter((e) => e.data.kind === 'battle').length === 1, 'no-bait control');
G.setPhase('KILL');
G.startBattle({ player: ['mem10'], opp: ['mot12'], playerHp: [70], oppHp: [30], phase: 'KILL', rig: { critBias: { player: -1, opp: -1 } }, seed: 3 });
tr = G.act('move', 0); console.log('  stall turn:', JSON.stringify(tr.lastTurn.actions.map((a) => [a.side, a.dmg, a.stalled, a.shaved])));
ok(G.getLog('rig.stall').length === 1, 'rig.stall logged'); ok(G.getBattle().player.active.hp === 1, 'player at 1 HP');
ok(G.getBattle().opp.active.hp === 6, 'opp at 6 after bait: ' + G.getBattle().opp.active.hp);
tr = G.act('move', 0); ok(tr.result === 'loss' && tr.offer === true, 'sliver loss with offer');
ok(G.getBattle().opp.active.hp / G.getBattle().opp.active.HP_max <= 0.06, 'opp ≤ 6%');
ok(G.getLog('offer.show').pop().data.special === true, 'special revive');
G.respondOffer(false); ok(G.getModel().phase === 'COOLDOWN', 'decline → COOLDOWN');

console.log('== damage formula (item 8)');
G.reset(); G.seed(5); G.skipAnims(true); G.finishOnboarding();
G.startBattle({ player: ['exe01', 'mem01', 'sen01'], opp: ['sen02', 'mem02', 'exe02'], rig: G.fairRig(2), seed: 7 });
const before = G.getBattle(); tr = G.act('move', 0);
const a0 = tr.lastTurn.actions.find((a) => a.kind === 'move');
{ const atk = a0.side === 'player' ? before.player.active : before.opp.active, def = a0.side === 'player' ? before.opp.active : before.player.active;
  const m = atk.moves[a0.moveIdx]; const base = m.power * atk.ATK_eff / def.DEF_eff * 0.55 + 2; const stab = m.sys === atk.sys ? 1.2 : 1;
  const want = Math.max(1, Math.floor(base * a0.eff * stab * (a0.crit ? 1.5 : 1) * a0.roll * 1)); ok(Math.abs(want - a0.dmg) <= 1, `first hit dmg ${a0.dmg} vs formula ${want} roll ${a0.roll}`); ok(a0.roll >= 0.85 && a0.roll <= 1.0, 'roll in range'); }

console.log('== rolls & crits (items 12/13)');
function rollStats(bias, crit, n) {
  const rolls = []; let crits = 0, hits = 0, k = 0;
  while (hits < n) {
    G.startBattle({ player: ['mem11'], opp: ['mem11'], rig: { rollBias: { player: bias, opp: 0 }, critBias: { player: crit, opp: -1 }, aiQ: 0 }, seed: 1000 + k++ }); G.autoBattle();
    const b = G.getBattle(); if (b.rig.sideMult !== 1) throw new Error('sideMult ' + b.rig.sideMult);
    for (const x of b.rolls) if (x.side === 'player') rolls.push(x.roll);
    for (const t of b.turns) for (const a of t.actions) if (a.kind === 'move' && a.side === 'player') { hits++; if (a.crit) crits++; }
    if (G.pendingOffer()) G.respondOffer(false);
  }
  return { mean: rolls.reduce((a, b) => a + b, 0) / rolls.length, n: rolls.length, critRate: crits / hits, hits, min: Math.min(...rolls), max: Math.max(...rolls) };
}
let rs = rollStats(0, 0, 2000); ok(approx(rs.mean, 0.925, 0.01) && rs.min >= 0.85 && rs.max <= 1.0, `fair roll mean ${rs.mean.toFixed(4)} over ${rs.n}`);
G.reset(); G.seed(5); G.skipAnims(true); G.finishOnboarding();
rs = rollStats(1, 0, 2000); ok(approx(rs.mean, 0.95, 0.01), `+1 roll mean ${rs.mean.toFixed(4)}`);
G.reset(); G.seed(5); G.skipAnims(true); G.finishOnboarding();
rs = rollStats(-1, 0, 2000); ok(approx(rs.mean, 0.90, 0.01), `-1 roll mean ${rs.mean.toFixed(4)}`);
G.reset(); G.seed(5); G.skipAnims(true); G.finishOnboarding();
rs = rollStats(0, -1, 4000); ok(rs.critRate === 0, `critBias -1 crit rate ${rs.critRate}`);
G.reset(); G.seed(5); G.skipAnims(true); G.finishOnboarding();
rs = rollStats(0, 1, 4000); ok(approx(rs.critRate, 0.125, 0.02), `critBias +1 crit rate ${rs.critRate.toFixed(4)} over ${rs.hits}`);

console.log('== packs (item 24 via simulatePacks)');
G.reset(); G.seed(3); G.skipAnims(true); G.finishOnboarding();
let sim = G.simulatePacks(4000, { phase: 'NORMAL' });
const pct = (slot, r) => (100 * sim.counts[slot][r] / sim.n);
ok(approx(pct(3, 'U'), 60, 3) && approx(pct(3, 'R'), 32, 3) && approx(pct(3, 'H'), 7, 3) && approx(pct(3, 'L'), 1, 3), `NORMAL rare slot U/R/H/L ${pct(3, 'U').toFixed(1)}/${pct(3, 'R').toFixed(1)}/${pct(3, 'H').toFixed(1)}/${pct(3, 'L').toFixed(1)}`);
ok(approx(pct(0, 'C'), 70, 3), `base slot Common ${pct(0, 'C').toFixed(1)}`);
let pity = 0; for (let k = 4; k < sim.n; k += 10) {} // pity: every 10th pack slot 5 is H/L
sim = G.simulatePacks(4000, { phase: 'KILL' }); ok(approx(100 * sim.rarePlusInSlot[3] / sim.n, 14.3, 2), `KILL slot 4 Rare+ ${(100 * sim.rarePlusInSlot[3] / sim.n).toFixed(1)}%`);
sim = G.simulatePacks(500, { phase: 'REWARD' }); ok(sim.rarePlusInSlot[4] === 500, `REWARD slot 5 Rare+ ${sim.rarePlusInSlot[4]}/500`);
sim = G.simulatePacks(30, { phase: 'NORMAL', packsSincePity: 9 }); ok(sim.counts[4].H + sim.counts[4].L >= 3, 'pity packs force Holo+');

console.log('== genOpponent (item 15)');
{ let bad = 0, holo = 0, common = 0; for (let i = 0; i < 1000; i++) { const ids = G.genOpponent({ rung: 3, effTier: 2, battlesPlayed: i }); const cs = ids.map((id) => ctx.CARD_SET.cards.find((c) => c.id === id)); if (new Set(ids).size !== 3) bad++; const sys = {}; cs.forEach((c) => { sys[c.element] = (sys[c.element] || 0) + 1; }); if (Object.values(sys).some((v) => v >= 3)) bad++; if (cs.some((c) => 'HL'.includes(c.rarity))) holo++; }
  for (let i = 0; i < 1000; i++) { const ids = G.genOpponent({ rung: 11, effTier: 11, battlesPlayed: i }); if (ids.map((id) => ctx.CARD_SET.cards.find((c) => c.id === id)).some((c) => c.rarity === 'C')) common++; }
  ok(bad === 0 && holo === 0 && common === 0, `gen: bad ${bad} holo@2 ${holo} common@11 ${common}`);
  let cpBad = 0; for (let i = 0; i < 200; i++) { const ids = G.genOpponent({ rung: 4, effTier: 4, battlesPlayed: i, player: ['mot09', 'lim06', 'exe01'], counterPick: true }); const want = ['lim', 'exe', 'mem']; ids.forEach((id, k) => { if (id.slice(0, 3) !== want[k]) cpBad++; }); } ok(cpBad === 0, 'counterPick beats each slot: bad ' + cpBad);
  const a = G.genOpponent({ rung: 4, effTier: 4, battlesPlayed: 3 }), b = G.genOpponent({ rung: 4, effTier: 4, battlesPlayed: 3 }), c = G.genOpponent({ rung: 4, effTier: 4, battlesPlayed: 4 }); ok(a.join() === b.join() && a.join() !== c.join(), 'gen deterministic & varies with battlesPlayed'); }

console.log('== rehearsals (items 6, 16, 17)');
G.reset(); G.seed(2); G.skipAnims(true); G.finishOnboarding();
let rh = G.rehearse(200, { phase: 'ONBOARD' }); ok(rh.lossRate === 0, 'tutorial lossRate 0: ' + JSON.stringify(rh));
rh = G.rehearse(200, { phase: 'CLEAN', rung: 2 }); ok(rh.lossRate <= 0.05, 'CLEAN rung 2 lossRate ' + rh.lossRate);
rh = G.rehearse(200, { phase: 'NORMAL', rung: 4, player: ['lim06', 'exe06', 'mot07'] }); ok(rh.lossRate >= 0.25 && rh.lossRate <= 0.75, 'NORMAL rung 4 3U lossRate ' + rh.lossRate);
const fr = G.rigProfile(); G.setPhase('NORMAL'); const np = G.rigProfile(); const fair = G.fairRig(2);
ok(JSON.stringify(np) === JSON.stringify(fair), 'NORMAL rig equals fair: ' + JSON.stringify(np) + ' vs ' + JSON.stringify(fair));

console.log('== AI reference (item 14/42)');
{ const E = ctx.GAME.engine; let mism = 0; const cards = G.cards; const rr = (() => { let s = 7; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; })();
  for (let i = 0; i < 100; i++) { const A = E.makeMon(cards[Math.floor(rr() * 60)], { hp: 1 + Math.floor(rr() * 150) }), D = E.makeMon(cards[Math.floor(rr() * 60)], { hp: 1 + Math.floor(rr() * 150) }); if (rr() < 0.3) D.weakened = true; const turn = 1 + Math.floor(rr() * 5);
    const ref = A.moves.map((m) => { const e = (m.power * (A.atkBase * (A.weakened ? 0.75 : 1)) / D.DEF_eff * 0.55 + 2) * ctx.CARD_SET.typeChart[m.element][D.sys] * (m.element === A.sys ? 1.2 : 1) * 0.925; const minD = e * 0.85 / 0.925, maxD = e / 0.925; const pko = minD >= D.hp ? 1 : maxD >= D.hp ? (maxD - D.hp) / (maxD - minD) : 0; const inc = Math.max(...D.moves.map((mm) => (mm.power * (D.atkBase * (D.weakened ? 0.75 : 1)) / A.DEF_eff * 0.55 + 2) * ctx.CARD_SET.typeChart[mm.element][A.sys] * (mm.element === D.sys ? 1.2 : 1) * 0.925)); let s = e + 40 * pko; if (m.effect === 'priority' && D.SPD_eff > A.SPD_eff && inc >= A.hp) s += 60; if (m.effect === 'drain' && A.hp < 0.4 * A.HP_max) s += 15; if (m.effect === 'weaken' && turn <= 2 && !D.weakened) s += 10; if (m.effect === 'recoil' && 0.25 * e >= A.hp) s -= 100; return s; });
    const got = A.moves.map((m) => E.score(m, A, D, { turn, playerFaster: D.SPD_eff > A.SPD_eff })); if (Math.abs(got[0] - ref[0]) > 1e-9 || Math.abs(got[1] - ref[1]) > 1e-9) mism++; }
  ok(mism === 0, 'score matches reference on 100 states, mismatches ' + mism); }

console.log('== segmentation & patience (items 26, 29)');
G.reset(); G.seed(9); G.setExperiment({ T0: 4.0, declineDelayMs: 1200, sliverPct: 0.05, postAdBoost: 1.10 }); G.skipAnims(true); G.finishOnboarding();
{ const aff = []; G.setPhase('NORMAL'); G.setModel({ session: { rewarded: 0 } });
  G.startBattle({ player: ['exe01'], opp: ['mot12'], seed: 11 }); r = G.autoBattle(); if (!r.offer) console.log('  (no offer on loss?)', JSON.stringify(r), G.getModel().phase);
  G.respondOffer(true); G.resolveAd(true); aff.push(G.getModel().adAffinity); G.autoBattle();
  G.setPhase('NORMAL'); G.setModel({ session: { revives: 0, rewarded: 0 } });
  G.startBattle({ player: ['exe01'], opp: ['mot12'], seed: 12 }); r = G.autoBattle(); G.respondOffer(false); aff.push(G.getModel().adAffinity);
  G.setPhase('NORMAL'); G.setModel({ session: { revives: 0, rewarded: 0 } });
  G.startBattle({ player: ['exe01'], opp: ['mot12'], seed: 13 }); r = G.autoBattle(); G.respondOffer(true); G.resolveAd(true); aff.push(G.getModel().adAffinity); G.autoBattle();
  ok(approx(aff[0], 0.675, 0.001) && approx(aff[1], 0.439, 0.001) && approx(aff[2], 0.635, 0.001), 'adAffinity sequence ' + aff.map((x) => x.toFixed(4)).join());
  G.setPhase('KILL'); const Fb = G.getModel().F; G.simulateQuit('hidden'); md = G.getModel(); ok(md.phase === 'CLEAN' && md.T <= Math.max(1.5, Fb - 0.5) + 1e-9 && G.getLog('quit.signal').length >= 1, `quit in KILL → CLEAN, T ${md.T} (F was ${Fb})`); }

console.log('== state machine (item 30)');
G.reset(); G.seed(4); G.setExperiment({ T0: 4.0, declineDelayMs: 1200, sliverPct: 0.05, postAdBoost: 1.10 }); G.skipAnims(true); G.finishOnboarding();
G.setPhase('NORMAL'); G.setModel({ F: 3.5, T: 4.0, session: { revives: 1 } }); G.startBattle({ player: ['exe01'], opp: ['mot12'], seed: 2 }); r = G.autoBattle(); if (r.offer) G.respondOffer(false);
{ const pe = G.getLog('phase.enter').pop(); ok(pe.data.to === 'CLEAN' && /cleanReady/.test(pe.data.reason), 'cleanReady → CLEAN: ' + pe.data.reason); }
G.setPhase('CLEAN'); G.startBattle({ rung: 2 }); r = G.autoBattle(); if (r.offer) G.respondOffer(false); ok(r.result === 'win' && G.getState().ladder.best === 2, 'rung 2 cleared (' + r.result + ')');
G.setPhase('NORMAL'); G.setModel({ F: 0, T: 4, winStreak: 2, newPBLast: true, session: { revives: 1 } }); G.setSegment('UNKNOWN');
md = G.getModel(); ok(md.killReady && !md.killArmed, 'killReady soft: ' + md.pendingTransition);
G.startBattle({ rung: 2 }); r = G.autoBattle(); if (r.offer) G.respondOffer(false);
md = G.getModel(); ok(md.phase === 'KILL' && md.kill.soft === true, 'entered soft KILL at battle end: ' + md.pendingTransition);
G.startBattle({ rung: 2 }); r = G.autoBattle();
ok(!r.offer && G.getLog('kill.armed').length === 0 && !G.getBattle().rig.bait, 'soft kill battle: no offer, no arming, no bait (' + r.result + ')');
ok(G.getModel().phase === 'NORMAL' && G.getModel().session.forceInterstitial === (r.result === 'loss'), 'soft kill → NORMAL, interstitial forced on loss');
md = G.getModel(); console.log('  phase after', md.phase, md.pendingTransition, 'kill', JSON.stringify(md.kill));

console.log('== fixes: HP clamp + turn-cap tally, early close = one decline, view-safe AI, engine exposures, pendingPack settles headless');
G.reset(); G.seed(1); G.setExperiment({ T0: 4.0, declineDelayMs: 1200, sliverPct: 0.05, postAdBoost: 1.10 }); G.skipAnims(true); G.finishOnboarding();
{ const old = G.CONST.battle.turnCap; G.CONST.battle.turnCap = 2; G.setPhase('NORMAL');
  G.startBattle({ player: ['sen02', 'sen03', 'sen04'], opp: ['mot09'], phase: 'NORMAL', rig: { rollBias: { player: 0, opp: 0 }, critBias: { player: -1, opp: -1 }, aiQ: 1 }, seed: 5 }); r = G.autoBattle(); if (r.offer) G.respondOffer(false); G.CONST.battle.turnCap = old;
  const b = G.getBattle(), tally = (t) => t.reduce((a, c) => a + Math.max(0, c.hp) / c.HP_max, 0), p = tally(b.player.team), o = tally(b.opp.team);
  ok(b.reason === 'cap' && b.result === (p >= o ? 'win' : 'loss') && b.result === 'win' && b.player.team.concat(b.opp.team).every((c) => c.hp >= 0), `turn cap (2) sen02/sen03/sen04 vs mot09 seed 5: ${b.result} by tally ${p.toFixed(3)} vs ${o.toFixed(3)}, hps ${b.player.team.map((c) => c.hp).join('/')}`); }
{ G.setPhase('NORMAL'); G.setModel({ session: { revives: 0, rewarded: 0 } }); G.startBattle({ player: ['exe01'], opp: ['mot12'], seed: 11 }); r = G.autoBattle();
  const m0 = G.getModel(), d0 = m0.decisions.slice(), a0 = m0.adAffinity, rev0 = m0.revenue.rewarded; G.respondOffer(true); G.fastForward(12500); G.resolveAd(false); md = G.getModel();
  const a2 = +(0.65 * +(0.35 + 0.65 * a0).toFixed(6)).toFixed(6);
  ok(r.offer && md.decisions.join() === d0.concat([1, 0]).join() && approx(md.adAffinity, a2, 1e-9) && md.phase === 'COOLDOWN' && md.revenue.rewarded === rev0 && G.getLog('ad.close').pop().data.method === 'early', `early-closed revive ad → decisions [${md.decisions.slice(-2).join(',')}], adAffinity ${md.adAffinity} (two-decision value ${a2}), phase ${md.phase}`); }
{ G.setPhase('NORMAL'); G.startBattle({ player: ['mot09', 'lim06', 'exe01'], opp: ['mem01', 'mot01', 'mot03'], phase: 'NORMAL', rig: { aiQ: 1, critBias: { player: -1, opp: -1 } }, seed: 77 });
  const E = G.engine, v = G.getBattle(), me = v.opp.active, you = v.player.active; let a, err = null; try { a = E.aiChoose(v, () => 0.99); } catch (e) { err = e.message; }
  const ref = me.moves.map((m) => E.score(m, me, you, { turn: v.turn, playerFaster: you.SPD_eff > me.SPD_eff })), want = ref[0] >= ref[1] ? 0 : 1;
  const w = Object.assign({}, me, { weakened: true }), m = me.moves[0], Ew = (m.power * w.ATK_eff / you.DEF_eff * 0.55 + 2) * ctx.CARD_SET.typeChart[m.sys][you.sys] * (m.sys === me.sys ? 1.2 : 1) * 0.925;
  ok(!err && a.kind === 'move' && a.idx === want, 'engine.aiChoose(getBattle() view, rng) → argmax move ' + (err || JSON.stringify(a)));
  ok(approx(E.expectedDamage(m, w, you), Ew, 1e-9), 'expectedDamage on a view creature applies weaken once (ATK_eff already effective)');
  let rep = 'none'; for (let i = 0; i < 8 && G.getBattle().state !== 'END'; i++) { G.act('move', 0); const v2 = G.getBattle(); if (v2.opp.team.some((c) => c.fainted) && v2.state !== 'END') { try { rep = E.aiReplace(v2, () => 0.1); } catch (e) { rep = 'throw: ' + e.message; } break; } }
  ok(rep === 'none' || (Number.isInteger(rep) && rep >= 0), 'engine.aiReplace(getBattle() view, rng) → ' + rep);
  ok(typeof E.genOpponent === 'function' && E.genOpponent({ rung: 3, effTier: 2, battlesPlayed: 4 }).join() === G.genOpponent({ rung: 3, effTier: 2, battlesPlayed: 4 }).join() && approx(E.tierMult(2), 0.89, 1e-9) && approx(E.potentiationOf(3), 1.08, 1e-9) && E.cardPower(ctx.CARD_SET.cards.find((c) => c.id === 'mot09'), 2) === 286, 'engine exposes genOpponent / tierMult / potentiationOf / cardPower'); }
{ G.autoBattle(); if (G.pendingOffer()) G.respondOffer(false); G.setModel({ session: { revives: 0, rewarded: 0 } }); G.setPhase('KILL'); G.startBattle({ rung: 2 }); G.autoBattle(); if (G.pendingOffer()) G.respondOffer(false); const ka = G.getLog('kill.armed').pop(); ok(ka && ka.data.candidates.length === 6 && ka.data.candidates.every((c) => typeof c.baitRate === 'number' && typeof c.stallRate === 'number'), 'kill.armed candidates carry baitRate/stallRate');
  const n = G.getLog('pack.end').length, po = G.getState().packsOpened; const p = G.openPack({ free: true }); st = G.getState();
  ok(p.cards.length === 5 && st.pendingPack === null && st.packsOpened === po + 1 && G.getLog('pack.end').length === n + 1, 'headless openPack credits and settles the pack at once (no pendingPack left)'); }
console.log('\n' + (fails ? fails + ' FAILURES' : 'ALL OK'));
process.exit(fails ? 1 : 0);
