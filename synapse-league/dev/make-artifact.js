// make-artifact.js — last build step. Derives synapse-league/artifact.html (the Claude Artifact variant) from
// index.html + style.css. The artifact host wraps the file in its own document skeleton, so artifact.html has no
// doctype/html/head/body of its own: <title>, the claude:viewport-fit meta, ONE inline <style> with all of style.css,
// the body's inner markup, then the two relative script tags.
const fs = require('fs');
const path = require('path');
const dir = path.join(__dirname, 'synapse-league');
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(dir, 'style.css'), 'utf8');
const title = /<title>([^<]+)<\/title>/.exec(html)[1].trim();
let body = /<body[^>]*>([\s\S]*?)<\/body>/i.exec(html)[1];
body = body.replace(/<script\s+src="[^"]+"><\/script>\s*/g, '').replace(/<!--[\s\S]*?-->\s*/g, '').trim();
const out = `<title>${title}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n<style>\n${css}\n</style>\n${body}\n<script src="cards.js"></script><script src="game.js"></script>\n`;
if (/<!doctype|<html[\s>]|<head[\s>]|<body[\s>]/i.test(out)) throw new Error('artifact.html must not contain document skeleton tags');
if ((out.match(/<style[\s>]/g) || []).length !== 1) throw new Error('artifact.html must contain exactly one <style>');
fs.writeFileSync(path.join(dir, 'artifact.html'), out);
console.log(`artifact.html written (${out.length} bytes, title "${title}")`);
