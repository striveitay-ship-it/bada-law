// Render the reel with Playwright.
//   node tools/render.mjs stills <outDir> <t1,t2,...>          -> PNG per time
//   node tools/render.mjs beats  <outDir>                      -> one PNG per beat (+ a contact sheet is made by the caller)
//   node tools/render.mjs video  <out.mp4> [workers] [sub]     -> 60 fps, `sub` subframes blended per frame (motion blur)
//   node tools/render.mjs events <out.json>                    -> sound-event list from the timeline
import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';
import http from 'node:http';

const ROOT = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..');
const TYPES = { '.html': 'text/html; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg', '.woff2': 'font/woff2', '.js': 'text/javascript', '.json': 'application/json', '.svg': 'image/svg+xml' };
const server = http.createServer((req, res) => {
  const p = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!p.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
  fs.readFile(p, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(p)] || 'application/octet-stream', 'Cache-Control': 'max-age=3600' });
    res.end(data);
  });
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const PAGE = `http://127.0.0.1:${server.address().port}/index.html`;
const [, , mode, out, ...rest] = process.argv;

async function openPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  page.on('console', m => { if (m.type() === 'warning' || m.type() === 'error') console.error('[page]', m.text()); });
  page.on('pageerror', e => console.error('[pageerror]', e.message));
  await page.goto(PAGE);
  await page.evaluate(() => window.REEL.ready);
  const cdp = await page.context().newCDPSession(page);
  return { page, cdp };
}
async function grab({ page, cdp }, t, format = 'png') {
  await page.evaluate(tt => window.REEL.seek(tt), t);
  const r = await cdp.send('Page.captureScreenshot', { format, quality: format === 'jpeg' ? 96 : undefined, optimizeForSpeed: true, captureBeyondViewport: false, fromSurface: true });
  return Buffer.from(r.data, 'base64');
}

const browser = await chromium.launch({ args: ['--disable-gpu-vsync', '--disable-frame-rate-limit', '--font-render-hinting=none'] });
try {
  if (mode === 'events') {
    const { page } = await openPage(browser);
    const data = await page.evaluate(() => ({ T: window.REEL.T, BPM: window.REEL.BPM, events: window.REEL.EVENTS }));
    fs.writeFileSync(out, JSON.stringify(data, null, 1));
    console.log('events', data.events.length);
  } else if (mode === 'stills' || mode === 'beats') {
    fs.mkdirSync(out, { recursive: true });
    const P = await openPage(browser);
    const T = await P.page.evaluate(() => window.REEL.T);
    const times = mode === 'beats' ? Array.from({ length: 64 }, (_, i) => i * 0.5) : rest[0].split(',').map(Number);
    for (const t of times) {
      const buf = await grab(P, t);
      fs.writeFileSync(path.join(out, `t${t.toFixed(3).padStart(7, '0')}.png`), buf);
    }
    console.log('stills', times.length, 'T=', T);
  } else if (mode === 'video') {
    const workers = +(rest[0] || 3), SUB = +(rest[1] || 4), FPS = 60;
    const { page } = await openPage(browser);
    const T = await page.evaluate(() => window.REEL.T);
    await page.close();
    const TA = rest[2] != null ? +rest[2] : 0, TB = rest[3] != null ? +rest[3] : T;
    const NF0 = Math.round(TA * FPS), NF = Math.round(TB * FPS) - NF0;
    const tmp = out + '.parts'; fs.mkdirSync(tmp, { recursive: true });
    const per = Math.ceil(NF / workers);
    const t0 = Date.now();
    let done = 0;
    await Promise.all(Array.from({ length: workers }, async (_, w) => {
      const f0 = w * per, f1 = Math.min(NF, f0 + per);
      if (f0 >= f1) return;
      const P = await openPage(browser);
      const part = path.join(tmp, `part${w}.mkv`);
      const vf = SUB > 1 ? `tmix=frames=${SUB}:weights=${Array(SUB).fill(1).join(' ')},select='eq(mod(n\\,${SUB})\\,${SUB - 1})',setpts=N/${FPS}/TB` : `setpts=N/${FPS}/TB`;
      const ff = spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'image2pipe', '-framerate', String(FPS * SUB), '-c:v', 'png', '-i', '-',
        '-vf', vf, '-r', String(FPS), '-c:v', 'libx264', '-preset', 'veryfast', '-qp', '0', '-pix_fmt', 'yuv444p', part], { stdio: ['pipe', 'inherit', 'inherit'] });
      for (let f = f0; f < f1; f++) {
        for (let s = 0; s < SUB; s++) {
          const t = (NF0 + f) / FPS + s / (FPS * SUB);
          const buf = await grab(P, t);
          if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
        }
        done++;
        if (done % 60 === 0) { const el = (Date.now() - t0) / 1000; console.log(`frames ${done}/${NF}  ${el.toFixed(0)}s  eta ${(el / done * (NF - done)).toFixed(0)}s`); }
      }
      ff.stdin.end();
      await new Promise((res, rej) => ff.on('close', c => c === 0 ? res() : rej(new Error('ffmpeg ' + c))));
      await P.page.close();
    }));
    const list = path.join(tmp, 'list.txt');
    fs.writeFileSync(list, Array.from({ length: workers }, (_, w) => `file 'part${w}.mkv'`).filter((_, w) => w * per < NF).join('\n'));
    await new Promise((res, rej) => {
      const ff = spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', out], { stdio: 'inherit' });
      ff.on('close', c => c === 0 ? res() : rej(new Error('concat ' + c)));
    });
    console.log('video', out, NF, 'frames in', ((Date.now() - t0) / 1000).toFixed(0), 's');
  }
} finally {
  await browser.close();
  server.close();
}
