// Shared Playwright harness for playtesting the game (a copy lives in dev/ so the folder is self-contained).
// Usage: NODE_PATH=/opt/node-tools/node_modules node dev/harness.js   (or require() it)
//   const { withPage } = require('./harness');
//   withPage(async (page, ctx) => { ... }, { url, viewport, name })
const path = require('path');
const fs = require('fs');
const { chromium, devices } = require('playwright');

const SHOTS = path.join(__dirname, 'shots');
fs.mkdirSync(SHOTS, { recursive: true });

async function withPage(fn, opts = {}) {
  const url = opts.url || process.env.GAME_URL;
  if (!url) throw new Error('withPage: pass opts.url or set GAME_URL (file:///abs/path/index.html)');
  const browser = await chromium.launch();
  const context = await browser.newContext({
    ...(opts.desktop ? { viewport: { width: 1280, height: 800 } } : { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true }),
    ...(opts.contextOptions || {}),
  });
  const page = await context.newPage();
  const errors = [];
  const logs = [];
  page.on('console', m => { logs.push(`[${m.type()}] ${m.text()}`); if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));
  const ctx = {
    errors, logs,
    shot: async (name) => { const p = path.join(SHOTS, `${opts.name || 'shot'}-${name}.png`); await page.screenshot({ path: p, fullPage: false }); return p; },
  };
  try {
    await page.goto(url, { waitUntil: 'load' });
    const result = await fn(page, ctx);
    return { result, errors, logs };
  } finally {
    await browser.close();
  }
}

// Wrap an artifact-variant page (no doctype/html/head/body) in a faithful copy of the
// claude.ai artifact host skeleton so it can be tested from file:// like the real thing.
function wrapArtifact(artifactHtmlPath, outPath) {
  const inner = fs.readFileSync(artifactHtmlPath, 'utf8');
  const skeleton = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><style>
:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}
</style></head><body>${inner}</body></html>`;
  fs.writeFileSync(outPath, skeleton);
  return outPath;
}

module.exports = { withPage, wrapArtifact, SHOTS };

if (require.main === module) {
  withPage(async (page, ctx) => {
    await page.waitForTimeout(500);
    await ctx.shot('smoke');
    return await page.title();
  }, { name: 'smoke' }).then(r => { console.log(JSON.stringify(r, null, 2)); process.exit(r.errors.length ? 1 : 0); })
    .catch(e => { console.error(e); process.exit(2); });
}
