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

console.log('== card data (acceptance 3 as replaced by Addendum 3 §6)');
{ const S = ctx.CARD_SET, SYSK = ['sen', 'mem', 'exe', 'lim', 'mot'], BUD = { C: 220, U: 245, R: 275, H: 310, L: 350, M: 360 }, KW = [null, 'priority', 'drain', 'recoil', 'weaken'];
  const KINDS = ['Neuron', 'Region', 'Molecule', 'Bias', 'Illusion', 'Experiment', 'Instrument', 'Creature', 'Mind'], reg = S.cards.filter((c) => c.id !== 'myt01'), uniq = (f) => new Set(S.cards.map(f)).size === S.cards.length;
  ok(S.rarities.map((r) => r.key).join() === 'C,U,R,H,L,M,IR,SIR' && S.rarities.filter((r) => r.variant).map((r) => r.key).join() === 'IR,SIR' && S.rarities.find((r) => r.key === 'H').name === 'Holo Rare' && S.rarities.find((r) => r.key === 'M').budget === BUD.M, 'rarity ladder C < U < R < H < L < M < IR < SIR; IR/SIR variant:true; "Holo Rare"; Mythic budget 360');
  ok(S.cards.length === 151 && reg.length === 150 && SYSK.every((s) => { const r = reg.filter((c) => c.element === s); return r.length === 30 && 'CURHL'.split('').map((k) => r.filter((c) => c.rarity === k).length).join() === '13,8,5,2,2'; }), '151 cards: 150 regular, 30 per System at 13 C / 8 U / 5 R / 2 H / 2 L');
  ok(S.cards.every((c) => c.hp + c.atk + c.def + c.spd === BUD[c.rarity] && c.moves.length === 2 && c.moves[0].element === c.element && c.moves.every((m) => KW.includes(m.effect) && (m.effect !== 'priority' || m.power <= 35))) && reg.every((c) => c.moves.every((m) => SYSK.includes(m.element))), 'every card on budget (Qualia 360), two moves, move 1 same-System, keywords in the closed list, priority ≤ 35');
  ok(S.kinds.join() === KINDS.join() && S.cards.every((c) => KINDS.includes(c.kind)) && KINDS.every((k) => S.cards.filter((c) => c.kind === k).length >= 6), 'nine kinds, each on ≥ 6 cards: ' + KINDS.map((k) => k + ' ' + S.cards.filter((c) => c.kind === k).length).join(', '));
  ok(uniq((c) => c.id) && uniq((c) => c.name) && uniq((c) => c.number) && uniq((c) => c.flavor) && uniq((c) => c.motif) && S.cards.every((c, i) => c.number === i + 1) && S.cards.every((c) => c.flavor.length <= 90), 'unique ids / names / numbers / flavours / motifs; numbers 1–151 in System order; flavour ≤ 90 chars');
  const q = S.cards.find((c) => c.id === 'myt01');
  ok(q === S.cards[150] && q.name === 'Qualia' && q.rarity === 'M' && q.element === 'all' && q.kind === 'Mind' && q.body === 'Mythic' && q.number === 151 && [q.hp, q.atk, q.def, q.spd].join() === '96,88,88,88' && JSON.stringify(q.moves) === JSON.stringify([{ name: 'Raw Feel', element: 'all', power: 55, effect: null }, { name: 'Explanatory Gap', element: 'all', power: 40, effect: 'weaken' }]) && q.flavor === 'Nobody has ever caught one. Everybody is sure they have.' && q.motif === 'a shimmer with no shape', 'myt01 "Qualia" exactly as Addendum 3 §3 specifies (#151, element all, 96/88/88/88)');
  const V = S.variants, IR = V.filter((v) => v.rarity === 'IR'), SIR = V.filter((v) => v.rarity === 'SIR'), baseOf = (v) => S.cards.find((c) => c.id === v.base);
  ok(V.length === 25 && IR.length === 15 && SIR.length === 10 && SYSK.every((s) => IR.filter((v) => baseOf(v).element === s).length === 3) && IR.every((v) => 'CUR'.includes(baseOf(v).rarity)) && new Set(IR.map((v) => v.base)).size === 15 && reg.filter((c) => c.rarity === 'L').every((c) => SIR.filter((v) => v.base === c.id).length === 1) && !V.some((v) => v.base === 'myt01'), '25 variants: 15 IR (3 per System, bases C/U/R, distinct), 10 SIR (one per Legendary), none of Qualia');
  ok(IR.every((v) => v.number >= 152 && v.number <= 166) && SIR.every((v) => v.number >= 167 && v.number <= 176) && new Set(V.map((v) => v.number)).size === 25 && V.every((v) => !S.cards.some((c) => c.id === v.id) && v.id === v.base + '-' + v.rarity.toLowerCase()), 'variant numbers IR 152–166, SIR 167–176; ids {base}-ir / {base}-sir, distinct from card ids');
  const fnv = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; }, P = G.printings(), F = ['name', 'sys', 'kind', 'body', 'hp', 'atk', 'def', 'spd', 'moves', 'motif'];
  ok(P.length === 176 && new Set(P.map((p) => p.id)).size === 176 && P.every((p) => { const c = G.cardOf(p.id); return c && p.baseId === c.id && p.seed === fnv(p.id) && F.every((f) => JSON.stringify(p[f]) === JSON.stringify(c[f])); }) && G.cardOf('exe12-sir').id === 'exe12' && G.cardOf('myt01').id === 'myt01' && G.isVariant('mem01-ir') && !G.isVariant('mem01') && G.cardOf('nope') === undefined, 'GAME.printings() lists 176; every printing resolves (GAME.cardOf) to its base stats, moves, body and element; seeds fnv1a32(id)');
  const ICONIC = "Phineas Gage · Milgram's Shock Box · Stanford Prison Study · Bobo Doll · Harlow's Wire Mother · Strange Situation · Sally–Anne Test · Invisible Gorilla · Rubin's Vase · Checker-Shadow Illusion · The Dress · Rorschach Inkblot · Mirror-Box Phantom Limb · Split-Brain · Blindsight · Grandmother Cell · Kandel's Sea Slug (Aplysia) · Cajal's Drawing · Golgi Stain · Action Potential · Myelin Sheath · Astrocyte · Default Mode Network · Grid Cell · Cerebellum · Broca's Area · Wernicke's Area · Corpus Callosum · Fusiform Face Area · Substantia Nigra · Dunning–Kruger · Placebo Effect · Cognitive Dissonance · Asch Conformity · Bystander Effect · Cocktail Party Effect · Ebbinghaus Forgetting Curve · Miller's Magic Seven · Method of Loci · Flashbulb Memory · Déjà Vu · Prospect Theory · Availability Heuristic · Framing Effect · Gambler's Fallacy · Hyperbolic Discounting · Peak–End Rule · Mere Exposure · Learned Helplessness · Flow State · Turing Test · Chinese Room · ELIZA · Backpropagation · Reward Prediction Error · Hebbian Synapse · Optogenetics · Patch Clamp · BOLD Signal (fMRI) · Alpha Wave (EEG) · Sleep Spindle · Suprachiasmatic Clock · Lucid Dream · Synesthesia · Tetris Effect · Mental Rotation · Weber's Law · Signal Detection · Predictive Coding · Global Workspace · Mary's Room · Philosophical Zombie · Capgras Delusion · Prosopagnosia · Hemineglect · Alien Hand · Hot Hand · Endowment Effect · Choice Overload · IKEA Effect · Nudge · Priming · Halo Effect · Barnum Effect · Fundamental Attribution Error · Yerkes–Dodson Curve · Working Memory Loop (Baddeley) · Waterfall Aftereffect · Troxler Fading · Phi Phenomenon · Binocular Rivalry · Saccade · Rods and Cones · Vestibular Canals · Proprioception · NMDA Receptor · GABA · Glutamate · Adrenaline · Endorphin · Melatonin · Norepinephrine · Thalamus · Hypothalamus · Insula · Brainstem · Prefrontal Cortex · Readiness Potential · Theory of Mind · Mirror Neuron · Pavlov's Bell · Skinner Box · Little Albert · Patient H.M. · Stroop Task · Marshmallow Test · Loss Aversion · Dopamine Spike · Variable Ratio Schedule · Hodgkin–Huxley Axon · Hopfield Attractor · Perceptron · Necker Cube · Müller-Lyer Arrows";
  const corpus = S.cards.flatMap((c) => [c.name].concat(c.moves.map((m) => m.name))).map((x) => x.toLowerCase()), missingIconic = ICONIC.split(' · ').filter((e) => { const m = /^(.*?)\s*\((.*)\)$/.exec(e), forms = m ? [m[1], m[2]] : [e]; return !corpus.some((n) => forms.some((f) => n.includes(f.toLowerCase()))); });
  ok(missingIconic.length === 0, 'every Addendum 3 §2 iconic entry appears as a card or move name (' + ICONIC.split(' · ').length + ' entries)' + (missingIconic.length ? ': missing ' + missingIconic.join(', ') : '')); }

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
const top3 = (() => { const seen = new Set(); return Object.keys(st.collection).map((id) => ({ id, p: G.powerOf(id), base: G.cardOf(id).id })).sort((a, b) => b.p - a.p || a.id.localeCompare(b.id)).filter((x) => !seen.has(x.base) && seen.add(x.base)).slice(0, 3).map((x) => x.id); })(); // top 3 creatures by power (LTP across printings), one printing each
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
ok(['R', 'H', 'L', 'M', 'IR', 'SIR'].includes(pk.slots[4].finalRarity), 'REWARD pack slot 5 Rare+: ' + pk.slots[4].finalRarity);
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

console.log('== packs (acceptance 24 amended, via simulatePacks; Addendum 2 §4, Addendum 3 §3)');
G.reset(); G.seed(3); G.skipAnims(true); G.finishOnboarding();
let sim = G.simulatePacks(2000, { phase: 'NORMAL' });
const pct = (slot, r) => (100 * sim.counts[slot][r] / sim.n), RARE = { U: 56, R: 30, H: 7, L: 1, IR: 5, SIR: 1 };
ok(Object.entries(RARE).every(([r, v]) => approx(pct(3, r), v, 3)) && sim.counts[3].C === 0 && sim.counts[3].M === 0, '2,000 NORMAL packs, rare slot within ±3 of U56/R30/H7/L1/IR5/SIR1: ' + Object.keys(RARE).map((r) => r + ' ' + pct(3, r).toFixed(1)).join(' / '));
ok(approx(pct(0, 'C'), 70, 3) && [0, 1, 2].every((k) => sim.counts[k].IR + sim.counts[k].SIR + sim.counts[k].M === 0), `base slot Common ${pct(0, 'C').toFixed(1)} (70 ± 3); no IR/SIR/M in base slots`);
ok(sim.mythic.total === 0 && sim.pity.packs === 200 && sim.pity.holoPlus === 200, `no myt01 in 2,000 NORMAL packs; every 10th pack's slot 5 is Holo+ (${sim.pity.holoPlus}/${sim.pity.packs})`);
sim = G.simulatePacks(4000, { phase: 'KILL' }); const k4 = 100 * sim.rarePlusInSlot[3] / sim.n;
ok(k4 <= 15 && approx(k4, 100 * 9.5 / 65.5, 2), `4,000 KILL packs (oddsMul 0.25, chase excluded): slot-4 Rare+ ${k4.toFixed(2)}% ≤ 15% (expected 9.5/65.5 = 14.50%)`);
ok(Object.keys(sim.chase).every((r) => r === 'pity') && [0, 1, 2, 3].every((k) => sim.counts[k].IR + sim.counts[k].SIR === 0) && sim.mythic.total === 0 && sim.pity.packs === 400 && sim.pity.holoPlus === 400, 'KILL packs: no IR/SIR except by pity ' + JSON.stringify(sim.chase) + '; pity still every 10th and Holo+');
sim = G.simulatePacks(500, { phase: 'REWARD' }); ok(sim.rarePlusInSlot[4] === 500, `REWARD slot 5 Rare+ ${sim.rarePlusInSlot[4]}/500`);
sim = G.simulatePacks(30, { phase: 'NORMAL', packsSincePity: 9 }); ok(sim.pity.packs === 3 && sim.holoPlusInSlot[4] >= 3, 'pity packs force Holo+ (H 85 / L 7 / IR 7 / SIR 1)');
sim = G.simulatePacks(5000, { phase: 'REWARD' });
{ const myt = sim.mythic.byReason['mythic-reward'] || 0, forced = (sim.reasons[4]['forced:rare+'] || 0) + myt;
  ok(approx(100 * myt / sim.n, 1, 0.5) && sim.mythic.total === myt && approx(100 * myt / forced, 1, 0.5), `5,000 REWARD packs: slot 5 is myt01 in ${(100 * myt / sim.n).toFixed(2)}% (${myt}; ${(100 * myt / forced).toFixed(2)}% of the ${forced} forced Rare+ slots, pity packs aside) — 1% ± 0.5, all reason 'mythic-reward'`); }
const all150 = Object.fromEntries(G.cards.filter((c) => c.id !== 'myt01').map((c) => [c.id, 1]));
for (const ph of ['NORMAL', 'KILL']) { sim = G.simulatePacks(50, { phase: ph, collection: all150 }); ok(sim.mythic.byReason.completion === 50 && sim.counts[4].M === 50, `completion route (${ph}, scratch 150/150 creatures): slot 5 is myt01 in ${sim.counts[4].M}/50`); }
G.setPhase('NORMAL'); G.giveCards(Object.keys(all150));
{ const prof = G.packProfile(); let p = G.openPack({ instant: true, free: true });
  ok(prof.completion && p.cards[4] === 'myt01' && p.slots[4].reason === 'completion' && G.getLog('pack.rig').some((e) => e.data.reason === 'completion') && G.getLog('pack.mythic').length === 1 && G.getState().collection.myt01 === 1, 'live completion route (NORMAL): 150/150 → the next pack\'s slot 5 is myt01, pack.rig{reason:completion}, pack.mythic logged');
  p = G.openPack({ instant: true, free: true }); ok(!G.packProfile().completion && p.slots[4].reason !== 'completion' && G.getLog('pack.mythic').length === 1, 'the route is spent once #151 is owned; pack.mythic logged once');
  G.reset(); G.seed(4); G.skipAnims(true); G.finishOnboarding(); G.giveCards(Object.keys(all150)); G.setPhase('KILL'); p = G.openPack({ instant: true, free: true });
  ok(p.cards[4] === 'myt01' && p.slots[4].reason === 'completion' && !p.bait && G.getLog('pack.mythic').length === 1, 'live completion route (KILL): slot 5 is myt01 and the bait pack is cancelled'); }

console.log('== needed printing (acceptance 40 statistics; Addendum 2 §5, Addendum 3 §5)');
G.reset(); G.seed(5); G.skipAnims(true); G.finishOnboarding();
{ const own = G.getState().collection, IRS = G.printings().filter((p) => p.rarity === 'IR').map((p) => p.id), miss = IRS[7], coll = Object.assign({}, own); IRS.forEach((id) => { if (id !== miss) coll[id] = 1; });
  ok(G.binderRows(coll).find((r) => r.row === 'IR').owned === 14, 'scratch collection: Illustration Rare row at 14/15, missing ' + miss);
  let s1 = G.simulatePacks(2000, { phase: 'KILL', collection: coll }); ok(s1.needed.id === miss && s1.needed.packs === 0, `the missing printing ${miss} never appears in 2,000 KILL packs (needed ${s1.needed.id})`);
  s1 = G.simulatePacks(2000, { phase: 'REWARD', collection: coll }); ok(s1.needed.id === miss && approx(s1.needed.last / s1.n, 0.25, 0.04), `…and lands in REWARD slot 5 at ${(s1.needed.last / s1.n).toFixed(3)} over 2,000 packs (0.25 ± 0.04)`);
  const coll2 = Object.assign({}, own); G.cards.filter((c) => c.sys === 'sen').forEach((c) => { coll2[c.id] = 1; }); delete coll2.sen01;
  s1 = G.simulatePacks(2000, { phase: 'KILL', collection: coll2 }); ok(s1.needed.id === 'sen01' && s1.needed.packs === 0, 'a System row at 29/30: the missing base printing (sen01) never appears in 2,000 KILL packs');
  const coll3 = Object.assign({}, all150, { 'mem01-ir': 1, 'mem10-ir': 1 }); delete coll3.mem01; delete coll3.mem10; delete coll3.mem05;
  const rows3 = G.binderRows(coll3); s1 = G.simulatePacks(2000, { phase: 'KILL', collection: coll3 });
  ok(rows3.find((r) => r.row === 'set').owned === 149 && rows3.find((r) => r.row === 'mem').owned === 27 && s1.needed.id === 'mem05' && s1.needed.packs === 0, "the 'set' row at 149/150 (mem01/mem10 held only as IR): needed mem05, never in 2,000 KILL packs"); }

console.log('== bait glint (Addendum 2 §6): rainbow "Holo" 0.7 / gold "Secret" 0.3 via rng.pack');
{ const shown = { Holo: 0, Secret: 0 }; let other = 0, mism = 0;
  for (let i = 0; i < 400; i++) { if (i % 20 === 0) { G.reset(); G.seed(200 + i); G.skipAnims(true); G.finishOnboarding(); } G.setPhase('KILL'); const p = G.openPack({ instant: true, free: true }); if (!p.bait) continue; if (p.bait.kind !== 'glint') { other++; continue; } shown[p.bait.shown]++; const e = G.getLog('bait.fire').pop().data; if (e.shown !== p.bait.shown || (e.glint === 'gold') !== (p.bait.shown === 'Secret')) mism++; }
  const n = shown.Holo + shown.Secret; ok(n >= 300 && other === 0 && mism === 0 && approx(shown.Secret / n, 0.3, 0.07), `glint baits: ${shown.Holo} Holo / ${shown.Secret} Secret of ${n} (${(100 * shown.Secret / n).toFixed(1)}% gold); bait.fire{shown, glint} agrees`); }

console.log('== genOpponent (item 15 amended: no myt01 from random draws, rung 12 fields it third)');
G.reset(); G.seed(3); G.skipAnims(true); G.finishOnboarding();
{ let bad = 0, holo = 0, common = 0; for (let i = 0; i < 1000; i++) { const ids = G.genOpponent({ rung: 3, effTier: 2, battlesPlayed: i }); const cs = ids.map((id) => ctx.CARD_SET.cards.find((c) => c.id === id)); if (new Set(ids).size !== 3) bad++; const sys = {}; cs.forEach((c) => { sys[c.element] = (sys[c.element] || 0) + 1; }); if (Object.values(sys).some((v) => v >= 3)) bad++; if (cs.some((c) => 'HL'.includes(c.rarity))) holo++; }
  for (let i = 0; i < 1000; i++) { const ids = G.genOpponent({ rung: 11, effTier: 11, battlesPlayed: i }); if (ids.map((id) => ctx.CARD_SET.cards.find((c) => c.id === id)).some((c) => c.rarity === 'C')) common++; }
  ok(bad === 0 && holo === 0 && common === 0, `gen: bad ${bad} holo@2 ${holo} common@11 ${common}`);
  let myt = 0, bad2 = 0; for (let i = 0; i < 2000; i++) { const ids = G.genOpponent({ rung: 1 + (i % 11), effTier: i % 15, battlesPlayed: i, candidate: i % 2, counterPick: i % 3 === 0, player: ['mot09', 'lim06', 'exe01'] }); if (ids.includes('myt01')) myt++; const sys = {}; ids.forEach((id) => { const e = G.cardOf(id).sys; sys[e] = (sys[e] || 0) + 1; }); if (new Set(ids).size !== 3 || Object.values(sys).some((v) => v >= 3)) bad2++; }
  ok(myt === 0 && bad2 === 0, `2,000 random generations (rungs 1–11, effTier 0–14, ± counter-pick): myt01 ${myt}, rule breaks ${bad2}`);
  let boss = 0; for (let i = 0; i < 2000; i++) { const ids = G.genOpponent({ rung: 12, effTier: 9 + (i % 6), battlesPlayed: i, candidate: i % 2, counterPick: i % 2 === 0, player: ['mot09', 'lim06', 'exe01'] }); if (ids.length === 3 && ids[2] === 'myt01' && ids[0] !== 'myt01' && ids[1] !== 'myt01' && ids[0] !== ids[1]) boss++; }
  ok(boss === 2000, `rung 12 (The Reviewer 2): slot 3 is myt01 in ${boss}/2,000 generations, slots 1–2 by the normal rules`);
  const cq = G.genOpponent({ rung: 5, effTier: 5, battlesPlayed: 1, counterPick: true, player: ['myt01', 'lim06', 'exe01'] }); ok(cq.length === 3 && !cq.includes('myt01') && cq.slice(1).map((id) => id.slice(0, 3)).join() === 'exe,mem', 'counter-pick against a Qualia slot falls back to any System (nothing beats "all"): ' + cq.join());
  let cpBad = 0; for (let i = 0; i < 200; i++) { const ids = G.genOpponent({ rung: 4, effTier: 4, battlesPlayed: i, player: ['mot09', 'lim06', 'exe01'], counterPick: true }); const want = ['lim', 'exe', 'mem']; ids.forEach((id, k) => { if (id.slice(0, 3) !== want[k]) cpBad++; }); } ok(cpBad === 0, 'counterPick beats each slot: bad ' + cpBad);
  const a = G.genOpponent({ rung: 4, effTier: 4, battlesPlayed: 3 }), b = G.genOpponent({ rung: 4, effTier: 4, battlesPlayed: 3 }), c = G.genOpponent({ rung: 4, effTier: 4, battlesPlayed: 4 }); ok(a.join() === b.join() && a.join() !== c.join(), 'gen deterministic & varies with battlesPlayed'); }

console.log('== printings, LTP and team (acceptance 38, 41; Addendum 2 §3)');
G.reset(); G.seed(1); G.skipAnims(true); G.finishOnboarding();
{ const L = (coll) => G.ltp('exe12', coll).pot;
  ok(L({ exe12: 1 }) === 1 && L({ exe12: 2 }) === 1.04 && G.ltp('mem01', { mem01: 1, 'mem01-ir': 1 }).pot === 1.08 && L({ 'exe12-sir': 1 }) === 1.08 && L({ exe12: 1, 'exe12-sir': 1 }) === 1.12 && L({ exe12: 9 }) === 1.2 && L({ exe12: 1, 'exe12-sir': 4 }) === 1.2 && L({}) === 1, 'potentiation: base×1 1.00, base×2 1.04, base+IR 1.08, SIR alone 1.08, base+SIR 1.12, cap 1.20 (exact)');
  ok(G.ltp('exe12-sir', { exe12: 1, 'exe12-sir': 1 }).units === 3 && G.powerOf('exe12', { exe12: 1 }) === 350 && G.powerOf('exe12-sir', { exe12: 1, 'exe12-sir': 1 }) === 392 && G.engine.potentiationOf(3) === 1.08, 'units = (copies − 1) + IR + 2·SIR across printings; power keyed on the creature (an SIR of an owned Legendary lifts it 350 → 392)');
  G.giveCards(['exe12', 'exe12-sir', 'mem01-ir', 'sen02']); const st0 = G.getState();
  ok(st0.collection['exe12-sir'] === 1 && st0.collection['mem01-ir'] === 1 && st0.collection.mem01 >= 1, 'giveCards takes printing ids; getState().collection is keyed by printing id');
  ok(G.setTeam(['exe12-sir', 'mem01-ir', 'sen02']) && G.getState().teamIds.join() === 'exe12-sir,mem01-ir,sen02' && G.getState().team.map((t) => t.base + ':' + t.printing).join() === 'exe12:SIR,mem01:IR,sen02:C', 'setTeam([exe12-sir, mem01-ir, sen02]) works when owned');
  const t0 = G.getState().teamIds.join(); ok(!G.setTeam(['exe12', 'exe12-sir', 'sen02']) && G.teamError(['exe12', 'exe12-sir', 'sen02']) === 'same creature twice' && G.getState().teamIds.join() === t0, 'setTeam([exe12, exe12-sir, sen02]) is rejected: "same creature twice"');
  G.startBattle({ rung: 2, free: true }); const bv = G.getBattle();
  ok(bv.player.team.map((m) => m.id).join() === 'exe12,mem01,sen02' && bv.player.team.map((m) => m.pid).join() === 'exe12-sir,mem01-ir,sen02' && bv.player.team.every((m) => { const c = G.cardOf(m.id), l = G.ltp(m.pid); return m.pot === l.pot && m.HP_max === Math.round(c.hp * l.pot) && Math.abs(m.ATK_eff - c.atk * l.pot) < 1e-3 && m.DEF_eff === c.def && m.SPD_eff === c.spd; }) && G.getLog('battle.start').pop().data.printings.join() === 'exe12-sir,mem01-ir,sen02', 'battle: stats = the base creature with its LTP; each fighter carries its printing (pid) for the art');
  G.act('forfeit');
  G.reset(); G.seed(1); G.skipAnims(true); G.finishOnboarding();
  const IRS = G.printings().filter((p) => p.rarity === 'IR').map((p) => p.id), want = 'mem01-ir'; G.giveCards(IRS.filter((id) => id !== want)); G.setTeam(['mem01', 'exe01', 'sen01']);
  ok(G.packProfile().needed === want && G.packProfile().neededRow === 'IR' && G.getLog('binder.nearcomplete').some((e) => e.data.row === 'IR' && e.data.missing === want), 'gallery row at 14/15 → binder.nearcomplete{row:IR}; the needed printing is ' + G.packProfile().needed);
  let eq = null; for (let i = 0; i < 40 && !eq; i++) { G.setPhase('REWARD'); const p = G.openPack({ instant: true, free: true }); if (p.cards.includes(want)) eq = p; }
  ok(eq && eq.badges[eq.cards.indexOf(want)].equipped === true && eq.badges[eq.cards.indexOf(want)].gain === 2 && G.getState().teamIds[0] === want && G.getLog('team.change').pop().data.auto === 'newArt', 'Summary bridge: mem01-ir pulled while mem01 is on the team → equipped at once (team.change{auto:newArt}), badge "+2 LTP"'); }

console.log('== Qualia, element "all" (acceptance 43)');
G.reset(); G.seed(7); G.skipAnims(true); G.finishOnboarding();
{ const E = G.engine, S5 = ['sen', 'mem', 'exe', 'lim', 'mot'];
  ok(S5.every((x) => E.typeMult('all', x) === 1 && E.typeMult(x, 'all') === 1) && E.typeMult('all', 'all') === 1 && E.typeMult('sen', 'mem') === 1.5 && E.typeMult('mem', 'sen') === 0.67, 'typeMult(all, x) == typeMult(x, all) == 1 for every System; the five-System cycle is unchanged');
  const Q = G.cardOf('myt01'), qm = E.makeMon(Q), dm = E.makeMon(G.cardOf('mem12')), want = (m) => (m.power * 88 / 80 * 0.55 + 2) * 1 * 1.2 * 0.925;
  ok(Q.moves.every((m) => Math.abs(E.expectedDamage(m, qm, dm) - want(m)) < 1e-9), "Qualia's moves get STAB (×1.2) at ×1.0 effectiveness (expectedDamage)");
  G.startBattle({ player: ['myt01'], opp: ['sen02'], rig: G.fairRig(2), seed: 9, free: true }); const tr = G.act('move', 0), hit = tr.lastTurn.actions.find((a) => a.side === 'player' && a.kind === 'move');
  const want2 = Math.max(1, Math.floor((55 * 88 / G.cardOf('sen02').def * 0.55 + 2) * 1 * 1.2 * (hit.crit ? 1.5 : 1) * hit.roll));
  ok(hit.eff === 1 && Math.abs(hit.dmg - want2) <= 1, `live hit: Raw Feel on Müller-Lyer eff ${hit.eff}, dmg ${hit.dmg} = formula with STAB ${want2}`);
  G.autoBattle(); if (G.pendingOffer()) G.respondOffer(false);
  const LEG = [['sen12', 'mem12', 'exe12'], ['lim12', 'mot12', 'sen30'], ['mem30', 'exe30', 'lim30'], ['mot30', 'sen12', 'lim12']]; let capped = 0, maxT = 0, n = 0, qWins = 0;
  for (const legs of LEG) for (let k = 0; k < 10; k++) for (const flip of [false, true]) {
    G.startBattle({ player: flip ? legs : ['myt01', 'myt01', 'myt01'], opp: flip ? ['myt01', 'myt01', 'myt01'] : legs, rig: G.fairRig(2), seed: 100 + k, free: true });
    const r = G.autoBattle(); if (r && r.offer) G.respondOffer(false); const b = G.getBattle(); n++; if (b.reason === 'cap') capped++; maxT = Math.max(maxT, b.turns.length); if ((b.result === 'win') !== flip) qWins++;
  }
  ok(capped === 0 && maxT <= 40 && n === 80, `three Qualia vs three Legendaries, fair rig, both sides (${n} battles): all end by KO inside the turn cap (max ${maxT} turns); the Qualia side won ${qWins}`);
  const rh = G.rehearse(100, { phase: 'FAIR', player: ['myt01', 'myt01', 'myt01'], opp: ['sen12', 'mem12', 'exe12'] }); ok(rh.capEnds === 0 && rh.maxTurns <= 40, 'rehearse ×100 (FAIR, Qualia ×3 vs sen12/mem12/exe12): ' + JSON.stringify({ lossRate: rh.lossRate, meanTurns: rh.meanTurns, maxTurns: rh.maxTurns, capEnds: rh.capEnds })); }

console.log('== rehearsals (items 6, 16, 17)');
G.reset(); G.seed(2); G.skipAnims(true); G.finishOnboarding();
let rh = G.rehearse(200, { phase: 'ONBOARD' }); ok(rh.lossRate === 0, 'tutorial lossRate 0: ' + JSON.stringify(rh));
rh = G.rehearse(200, { phase: 'CLEAN', rung: 2 }); ok(rh.lossRate <= 0.05, 'CLEAN rung 2 lossRate ' + rh.lossRate);
rh = G.rehearse(200, { phase: 'NORMAL', rung: 4, player: ['lim06', 'exe06', 'mot07'] }); ok(rh.lossRate >= 0.25 && rh.lossRate <= 0.75, 'NORMAL rung 4 3U lossRate ' + rh.lossRate);
const fr = G.rigProfile(); G.setPhase('NORMAL'); const np = G.rigProfile(); const fair = G.fairRig(2);
ok(JSON.stringify(np) === JSON.stringify(fair), 'NORMAL rig equals fair: ' + JSON.stringify(np) + ' vs ' + JSON.stringify(fair));

console.log('== AI reference (item 14/42)');
{ const E = ctx.GAME.engine; let mism = 0; const cards = G.cards; const rr = (() => { let s = 7; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; })();
  const TM = (a, d) => (a === 'all' || d === 'all' ? 1 : ctx.CARD_SET.typeChart[a][d]); // reference: 'all' (Qualia) is neutral both ways
  for (let i = 0; i < 100; i++) { const A = E.makeMon(cards[Math.floor(rr() * cards.length)], { hp: 1 + Math.floor(rr() * 150) }), D = E.makeMon(i % 10 === 0 ? cards[150] : cards[Math.floor(rr() * cards.length)], { hp: 1 + Math.floor(rr() * 150) }); if (rr() < 0.3) D.weakened = true; const turn = 1 + Math.floor(rr() * 5);
    const ref = A.moves.map((m) => { const e = (m.power * (A.atkBase * (A.weakened ? 0.75 : 1)) / D.DEF_eff * 0.55 + 2) * TM(m.element, D.sys) * (m.element === A.sys ? 1.2 : 1) * 0.925; const minD = e * 0.85 / 0.925, maxD = e / 0.925; const pko = minD >= D.hp ? 1 : maxD >= D.hp ? (maxD - D.hp) / (maxD - minD) : 0; const inc = Math.max(...D.moves.map((mm) => (mm.power * (D.atkBase * (D.weakened ? 0.75 : 1)) / A.DEF_eff * 0.55 + 2) * TM(mm.element, A.sys) * (mm.element === D.sys ? 1.2 : 1) * 0.925)); let s = e + 40 * pko; if (m.effect === 'priority' && D.SPD_eff > A.SPD_eff && inc >= A.hp) s += 60; if (m.effect === 'drain' && A.hp < 0.4 * A.HP_max) s += 15; if (m.effect === 'weaken' && turn <= 2 && !D.weakened) s += 10; if (m.effect === 'recoil' && 0.25 * e >= A.hp) s -= 100; return s; });
    const got = A.moves.map((m) => E.score(m, A, D, { turn, playerFaster: D.SPD_eff > A.SPD_eff })); if (Math.abs(got[0] - ref[0]) > 1e-9 || Math.abs(got[1] - ref[1]) > 1e-9) mism++; }
  ok(mism === 0, 'score matches reference on 100 states over all 151 creatures (Qualia as defender every 10th), mismatches ' + mism); }

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
console.log('== art pass: new-kind glyphs, Illustration Rares, Secrets, #151, sprite rings (acceptance 5 and 39, headless structure)');
{ // a minimal XML well-formedness check: balanced tags, every attribute quoted and unique, text escaped
  const xmlOk = (s) => { const stack = [], tag = /<(\/?)([a-zA-Z][\w:.-]*)([^<>]*?)(\/?)>/g; let last = 0, m;
    while ((m = tag.exec(s))) { const text = s.slice(last, m.index); if (/[<>]/.test(text) || /&(?!(amp|lt|gt|quot|#39);)/.test(text)) return 'text@' + m.index; last = tag.lastIndex;
      const [, close, name, attrs, self] = m; if (close) { if (stack.pop() !== name) return 'close ' + name; continue; }
      const names = [], rest = attrs.replace(/\s+([\w:-]+)="([^"<]*)"/g, (_, n) => { names.push(n); return ''; }); if (rest.trim()) return 'attr ' + name; if (new Set(names).size !== names.length) return 'dup ' + name; if (!self) stack.push(name); }
    return stack.length ? 'unclosed ' + stack.join(',') : null; };
  const P = G.printings(), A = G.art, by = Object.fromEntries(P.map((p) => [p.id, p])), bad = [], RING = /<g class="sprite-ring[^"]*">[\s\S]*?<\/g>/, num = (p) => String(p.number).padStart(3, '0') + '/151';
  for (const p of P) {
    const outs = [A.renderCard(p), A.renderCard(p, { size: 'mini' }), A.renderCard(p, { owned: false }), A.renderCard(p, { owned: false, size: 'mini' }), A.renderCard(p, { copies: 3, uid: 7 }), A.renderSprite(p, 48)];
    outs.forEach((s, i) => { const e = xmlOk(s); if (e) bad.push(p.id + ':' + i + ':' + e); });
    const card = outs[0], g = A.renderGlyph(G.cardOf(p.id)), sp = outs[5], ring = RING.exec(sp), want = { IR: 'ring-IR', SIR: 'ring-SIR', M: 'ring-M' }[p.rarity];
    if (card !== A.renderCard(p) || sp !== A.renderSprite(p, 48) || !card.includes(g) || !sp.includes(g)) bad.push(p.id + ':glyph/det');
    if (want ? !(ring && ring[0].includes(want) && sp.replace(ring[0], '') === A.renderSprite(by[p.baseId], 48).replace(RING, '')) : !!ring) bad.push(p.id + ':ring');
    if (!card.includes('>' + num(p) + '<')) bad.push(p.id + ':number');
    if (p.rarity === 'IR' && !(!card.includes('class="frame') && card.includes('<g class="scene"') && card.includes('ILLUSTRATION RARE') && !card.includes('SECRET') && card.includes('<title>' + p.scene.replace(/'/g, '&#39;') + '</title>') && card.includes('fill="#ffffff"/>'))) bad.push(p.id + ':IR');
    if (p.rarity === 'SIR' && !(/<rect class="frame gold"[^>]*stroke="url\(#gold[^"]*\)" stroke-width="8"/.test(card) && /<g class="sheen"[^>]*opacity="0\.35" style="mix-blend-mode:screen"/.test(card) && card.includes('SECRET ILLUSTRATION RARE') && p.number >= 167 && card.includes('fill="#fde68a">' + num(p) + '<'))) bad.push(p.id + ':SIR');
  }
  const q = A.renderCard(by.myt01), shimmer = /<g class="shimmer">([\s\S]*?)<\/g>/.exec(q);
  ok(bad.length === 0, 'all 176 printings: card (full, mini, unowned, unowned mini, LTP) and sprite are well-formed SVG, deterministic, carry the creature\'s glyph; IR = full-bleed scene, no frame; SIR = 8 px gold frame + 35% screen sheen + gold number ≥ 167; rings only on IR/SIR/#151 ' + bad.slice(0, 6).join(' '));
  ok(q.includes('>151/151<') && q.includes('>SECRET<') && q.includes('class="frame iris"') && shimmer && (shimmer[1].match(/<circle /g) || []).length === 40 && /<g transform="translate\(50,40\)" opacity="0\.15"><g class="glyph k-mind">/.test(q) && q.includes('class="sheen"') && q.includes('>ALL<'), 'Qualia: 151/151, SECRET, the iridescent conic frame and sheen, its Mind glyph at 15% over 40 shimmer particles, "All Systems" chips');
  const kinds = ['Instrument', 'Creature', 'Mind'].map((k) => { const cs = G.cards.filter((c) => c.kind === k), gs = cs.map((c) => A.renderGlyph(c)); return cs.length >= 6 && gs.every((s) => s.startsWith(`<g class="glyph k-${k.toLowerCase()}">`) && !s.includes('v34 l-9 30') && !xmlOk(s)) && new Set(gs).size === cs.length; });
  ok(kinds.every(Boolean), 'Instrument, Creature and Mind draw their own seeded glyphs (no Experiment fallback, all distinct, well-formed): ' + kinds.join());
  G.reset(); G.seed(1); G.skipAnims(true); G.finishOnboarding(); G.setPhase('NORMAL'); G.giveCards(['exe12']);
  const fp = G.openPack({ instant: true, free: true, force: [null, null, 'mem01-ir', 'exe12-sir', 'myt01'] }), sent = fp.sentence;
  ok(fp.cards.slice(2).join() === 'mem01-ir,exe12-sir,myt01' && fp.slots.slice(2).every((s) => s.reason === 'lab') && G.getLog('lab.override').some((e) => e.data.what === 'packSlots') && G.getLog('pack.chase').slice(-2).map((e) => e.data.printing).join() === 'mem01-ir,exe12-sir' && G.getLog('pack.mythic').length === 1 && fp.badges[3].gain === 3 && fp.badges[2].gain === 2, 'GAME.openPack({force}) deals the given printings (reason lab, lab.override{packSlots}); pack.chase ×2, pack.mythic once; LTP gains +2 (IR) / +3 (SIR)');
  ok(/slot 5 forced Mythic \(lab → myt01 · 151\/151 Qualia\)\.$/.test(sent), 'the Lab pack sentence names printings by id and number: ' + sent);
  G.reset(); G.seed(1); G.skipAnims(true); G.finishOnboarding(); G.giveCards(['exe12', 'lim12', 'sen12']); G.setTeam(['exe12', 'lim12', 'sen12']); G.setPhase('CLEAN');
  G.startBattle({ rung: 12 }); G.act('forfeit'); const s0 = G.getState();
  G.give({ focus: 5 }); G.startBattle({ rung: 12, oppHp: [1, 1] }); const rr = G.autoBattle(); if (rr.offer) G.respondOffer(false); const s1 = G.getState(), met = G.getBattle().turns.some((t) => t.actions.some((a) => a.side === 'opp' && a.to === 'myt01'));
  ok(s0.ladder.seenTeams[12][2] === 'myt01' && s0.seen151 === false && met && s1.seen151 === true, 'save.seen151: a rung-12 forfeit before #151 is sent out keeps the secret; once Qualia appears in a battle the flag is set (Home and Ladder may name it)'); }
console.log('\n' + (fails ? fails + ' FAILURES' : 'ALL OK'));
process.exit(fails ? 1 : 0);
