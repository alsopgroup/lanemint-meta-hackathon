'use strict';
/* Muse Data Check — frontend logic ported from the design prototype
   (Muse Data Check.dc.html). Samples, heuristics, and scoring reproduce
   the prototype exactly; live mode calls POST /api/check when the server
   reports a configured Muse API key, and falls back to simulated
   responses otherwise. */

const STEPS = [['read', 'Reading file', 0.16], ['dups', 'Checking duplicates', 0.18], ['gaps', 'Checking gaps', 0.2], ['prices', 'Checking prices', 0.24], ['splits', 'Checking splits', 0.22]];
const TYPES = { dups: ['Duplicate', 'var(--violet-500)'], gaps: ['Gap', 'var(--warning-500)'], prices: ['Bad price', 'var(--error-500)'], splits: ['Split', 'var(--lime-500)'] };

const state = { sid: 's1', step: 'data', stage: 'idle', elapsed: 0, view: 'raw', active: -1, drag: false, upload: null, live: null, err: null, runMode: null, liveAttempted: false };
let appConfig = { live: false, mode: 'simulated', model: 'muse-spark' };
let timer = null;
let pendingLive = false;
let currentRunId = 0;
let currentAbort = null;

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ---------------- prototype data helpers (ported verbatim) ---------------- */
function rng(s) { return () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function walk(seed, n, p0, vb, sig) { const r = rng(seed); let p = p0; const o = []; for (let i = 0; i < n; i++) { const op = p, c = op * (1 + (r() - 0.5) * sig), h = Math.max(op, c) * (1 + r() * sig * 0.35), l = Math.min(op, c) * (1 - r() * sig * 0.35); o.push({ o: op, h, l, c, v: Math.round(vb * (0.6 + r() * 0.8)) }); p = c; } return o; }
function tdays(start, n) { const o = []; let t = Date.parse(start + 'T00:00:00Z'); while (o.length < n) { const w = new Date(t).getUTCDay(); if (w > 0 && w < 6) o.push(new Date(t).toISOString().slice(0, 10)); t += 864e5; } return o; }
function hm(m) { return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); }
function tsf(ms) { const h = Math.floor(ms / 36e5), m = Math.floor(ms / 6e4) % 60, s = Math.floor(ms / 1e3) % 60, x = ms % 1000; return [h, m, s].map(v => String(v).padStart(2, '0')).join(':') + '.' + String(x).padStart(3, '0'); }
function mk(cells, oi) { return { cells, oi, tags: [], clean: null, cleanFor: null, remove: null, gap: null }; }
function fmtV(v, d) { if (v == null || v === '') return '—'; if (d === 'str') return String(v); if (d === 'int') return Math.round(v).toLocaleString('en-US').replace('-', '−'); return Number(v).toFixed(d).replace('-', '−'); }
function rowsTxt(rows) {
  if (!rows || !rows.length) return '';
  const r = [...new Set(rows)].sort((a, b) => a - b), parts = []; let a = r[0], p = r[0];
  for (let i = 1; i <= r.length; i++) { if (r[i] === p + 1) { p = r[i]; continue; } parts.push(a === p ? '' + a : a + '–' + p); a = p = r[i]; }
  const s = parts.length > 4 ? parts.slice(0, 4).join(', ') + ' +' + (parts.length - 4) : parts.join(', ');
  return (r.length > 1 ? 'Rows ' : 'Row ') + s;
}
function finish(s) {
  s.rows.forEach((r, i) => r.n = i + 1);
  const rn = id => s.rows.filter(r => r.tags.includes(id)).map(r => r.n);
  s.issues.forEach(is => { is.rows = rn(is.id); is.explain = is.ex(is.rows, s.rows); });
  (s.extra || []).forEach(e => { e.rows = rn(e.id); e.explain = e.ex(e.rows, s.rows); });
  s.findings = s.issues.filter(i => s.found.includes(i.id)).map(i => ({ step: i.step, title: i.title, rows: i.rows, explain: i.explain }))
    .concat((s.extra || []).map(e => ({ step: e.step, title: e.title, rows: e.rows, explain: e.explain, note: e.note })));
  const ord = STEPS.map(x => x[0]); s.findings.sort((a, b) => ord.indexOf(a.step) - ord.indexOf(b.step));
  return s;
}
function dupAfter(base, at, skip) {
  const rows = []; base.forEach(r => { if (skip.includes(r.oi)) return; rows.push(r); if (r.oi === at) { const c = mk({ ...r.cells }, at); c.tags.push('dup'); c.remove = 'dup'; r.tags.push('dup'); rows.push(c); } }); return rows;
}
function s1() {
  const d = tdays('2026-03-02', 16), b = walk(11, 16, 182.4, 4.2e6, 0.024), f2 = v => fmtV(v, 2);
  const base = b.map((x, i) => mk({ date: d[i], o: x.o, h: x.h, l: x.l, c: x.c, v: x.v }, i));
  const rows = dupAfter(base, 3, [6, 7]);
  rows.forEach(r => { const c = r.cells; if (r.oi >= 11) { c.o /= 2; c.h /= 2; c.l /= 2; c.c /= 2; c.v *= 2; } else { r.clean = { ...c, o: c.o / 2, h: c.h / 2, l: c.l / 2, c: c.c / 2, v: c.v * 2 }; r.cleanFor = 'split'; } });
  base[5].tags.push('gap'); base[8].tags.push('gap'); base[5].gap = { for: 'gap', text: '2 sessions missing · ' + d[6] + ', ' + d[7] + ' · flagged, not filled' };
  base[11].tags.push('split');
  return finish({
    id: 's1', name: 'SIMX daily bars', file: 'simx_daily_2026Q1.csv', desc: 'Equity · daily', ms: 2841, found: ['dup', 'gap', 'split'],
    cols: [['date', 'date', 'str'], ['o', 'open', 2], ['h', 'high', 2], ['l', 'low', 2], ['c', 'close', 2], ['v', 'volume', 'int']], rows,
    issues: [
      { id: 'dup', step: 'dups', title: 'Duplicate bar', ex: (rn, R) => `Rows ${rn[0]} and ${rn[1]} are identical in every column, volume included. The ${R[rn[0] - 1].cells.date} session is counted twice in any return or volume calculation. Keep one copy.` },
      { id: 'gap', step: 'gaps', title: 'Two missing sessions', ex: () => `Dates jump from ${d[5]} to ${d[8]}. ${d[6]} and ${d[7]} are weekdays with no rows, so a one-row return across this break actually covers three sessions.` },
      { id: 'split', step: 'splits', title: 'Unadjusted 2-for-1 split', ex: (rn, R) => { const r = R[rn[0] - 1], p = R[rn[0] - 2]; return `Close falls from ${f2(p.cells.c)} to ${f2(r.cells.c)} on ${r.cells.date} while volume roughly doubles, and prices stay at the new level. That is a 2-for-1 split that was not back-adjusted. Earlier prices should be divided by 2 and their volume doubled.`; } }]
  });
}
function s2() {
  const b = walk(23, 18, 64.1, 52000, 0.006), t = i => hm(570 + 5 * i), f2 = v => fmtV(v, 2);
  const base = b.map((x, i) => mk({ t: t(i), o: x.o, h: x.h, l: x.l, c: x.c, v: x.v }, i));
  base[7].cells.v *= 6; base[7].tags.push('fp1');
  { const c = base[14].cells; base[14].clean = { ...c }; const h = c.h; c.h = c.l; c.l = h; base[14].cleanFor = 'inv'; base[14].tags.push('inv'); }
  const rows = dupAfter(base, 4, [10, 11]);
  base[9].tags.push('gap'); base[12].tags.push('gap'); base[9].gap = { for: 'gap', text: '2 bars missing · ' + t(10) + ', ' + t(11) + ' · flagged, not filled' };
  return finish({
    id: 's2', name: 'SIMQ 5-minute bars', file: 'simq_5min_2026-03-04.csv', desc: 'Equity · intraday', ms: 3312, found: ['dup', 'gap', 'inv'],
    cols: [['t', 'time', 'str'], ['o', 'open', 2], ['h', 'high', 2], ['l', 'low', 2], ['c', 'close', 2], ['v', 'volume', 'int']], rows,
    issues: [
      { id: 'dup', step: 'dups', title: 'Duplicate 5-minute bar', ex: rn => `Rows ${rn[0]} and ${rn[1]} are both stamped ${t(4)} with identical values. One bar is a repeat and should be dropped.` },
      { id: 'gap', step: 'gaps', title: 'Two missing bars', ex: () => `The bar after ${t(9)} is stamped ${t(12)}. The ${t(10)} and ${t(11)} bars are missing during regular trading hours.` },
      { id: 'inv', step: 'prices', title: 'High below low', ex: (rn, R) => { const c = R[rn[0] - 1].cells; return `On the ${c.t} bar the high (${f2(c.h)}) is below the low (${f2(c.l)}). A bar can't trade outside its own range, so the two fields look swapped.`; } }],
    extra: [{ id: 'fp1', step: 'prices', title: 'Possible bad volume print', note: 'Not a planted error. Volume spikes are allowed in this seed.', ex: () => `Volume on the ${t(7)} bar is about 6× the typical bar. It may be a bad print and should be checked before it feeds a volume-weighted signal.` }]
  });
}
function s3() {
  const r = rng(37), f5 = v => fmtV(v, 5); let bid = 1.08421, ms = 14 * 36e5 + 2 * 6e4 + 1120; const base = [];
  for (let i = 0; i < 17; i++) { bid += Math.round((r() - 0.5) * 4) * 0.00001; const sp = 0.00008 + Math.round(r() * 3) * 0.00001; ms += 80 + Math.round(r() * 320); base.push(mk({ ts: tsf(ms), b: bid, a: bid + sp, s: (1 + Math.floor(r() * 5)) * 1e6 }, i)); }
  { const c = base[5].cells; base[5].clean = { ...c }; c.b = c.b * 10; base[5].cleanFor = 'fat'; base[5].tags.push('fat'); }
  for (let i = 12; i < 17; i++) { base[i].cells.b = base[11].cells.b; base[i].cells.a = base[11].cells.a; base[i].tags.push('stale'); base[i].remove = 'stale'; }
  const A = base[9], B = base[10]; A.tags.push('order'); B.tags.push('order'); A.cleanFor = B.cleanFor = 'order'; B.clean = { ...A.cells }; A.clean = { ...B.cells };
  const rows = base.slice(); rows[9] = B; rows[10] = A;
  return finish({
    id: 's3', name: 'EUR/SIM ticks', file: 'eursim_ticks_2026-03-05.csv', desc: 'FX · tick quotes', ms: 2604, found: ['fat', 'order'],
    cols: [['ts', 'timestamp', 'str'], ['b', 'bid', 5], ['a', 'ask', 5], ['s', 'size', 'int']], rows,
    issues: [
      { id: 'fat', step: 'prices', title: 'Fat-finger bid', ex: (rn, R) => { const c = R[rn[0] - 1].cells, p = R[rn[0] - 2].cells, nx = R[rn[0]].cells; return `Bid jumps from ${f5(p.b)} to ${f5(c.b)} and returns to ${f5(nx.b)} on the next tick, while the ask stays near ${f5(c.a)}. A 10× move for a single tick is a misplaced decimal, not a market move. Replace it with ${f5(c.b / 10)} or drop the tick.`; } },
      { id: 'order', step: 'gaps', title: 'Out-of-order timestamps', ex: (rn, R) => `Row ${rn[0]} is stamped ${R[rn[0] - 1].cells.ts}, later than row ${rn[1]} at ${R[rn[1] - 1].cells.ts} that follows it. Sort by timestamp before computing anything time-based.` },
      { id: 'stale', step: 'prices', title: 'Stale quotes', ex: rn => `Rows ${rn[0]}–${rn[rn.length - 1]} repeat the bid and ask from row ${rn[0] - 1} while size changes on every tick. The price feed stopped updating.` }]
  });
}
const S = [s1(), s2(), s3()];

function tparse(v) {
  v = String(v || '').trim(); const m = v.match(/^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?$/);
  if (m) return ((+m[1]) * 3600 + (+m[2]) * 60 + (+(m[3] || 0))) * 1000 + (+(String(m[4] || '0').padEnd(3, '0').slice(0, 3)));
  const t = Date.parse(v.includes(' ') && /^\d{4}-/.test(v) ? v.replace(' ', 'T') + 'Z' : v); return isNaN(t) ? null : t;
}
function heur(s) {
  const H = s.cols.map(c => c[1].toLowerCase()), R = s.rows, N = R.length;
  const col = re => { const i = H.findIndex(h => re.test(h)); return i < 0 ? null : 'c' + i; };
  const num = (r, k) => { const v = parseFloat(String(r.cells[k]).replace(/,/g, '')); return isFinite(v) ? v : null; };
  const list = rn => rn.length > 6 ? rn.slice(0, 6).join(', ') + ' and ' + (rn.length - 6) + ' more' : rn.join(', ');
  const add = (id, step, title, idxs, ex) => { if (!idxs.length) return; [...new Set(idxs)].forEach(i => R[i].tags.push(id)); s.issues.push({ id, step, title, ex: rn => ex(rn) }); s.found.push(id); };
  const seen = {}, di = []; R.forEach((r, i) => { const k = s.cols.map(c => r.cells[c[0]]).join('|'); if (k in seen) { di.push(seen[k], i); r.remove = 'u_dup'; } else seen[k] = i; });
  add('u_dup', 'dups', 'Duplicate rows', di, rn => `Rows ${list(rn)} repeat an earlier row exactly. Repeated rows double-count those bars in returns and volume. Keep the first copy of each.`);
  const tk = col(/date|time|^ts|stamp/);
  if (tk) {
    const tv = R.map(r => tparse(r.cells[tk]));
    if (tv.filter(v => v != null).length > N * 0.8) {
      const oo = []; for (let i = 1; i < N; i++) if (tv[i] != null && tv[i - 1] != null && tv[i] < tv[i - 1]) oo.push(i - 1, i);
      add('u_order', 'gaps', 'Out-of-order timestamps', oo, rn => `Rows ${list(rn)} go backwards in time. Sort by timestamp before computing anything time-based.`);
      const df = []; for (let i = 1; i < N; i++) if (tv[i] > tv[i - 1]) df.push(tv[i] - tv[i - 1]); df.sort((a, b) => a - b); const med = df[Math.floor(df.length / 2)] || 0;
      const gi = []; if (med) for (let i = 1; i < N; i++) { const d = tv[i] - tv[i - 1]; if (!(d > 0)) continue; let miss = false; if (med >= 20 * 36e5) { let wd = 0; for (let t = tv[i - 1] + 864e5; t < tv[i] - 36e5; t += 864e5) { const w = new Date(t).getUTCDay(); if (w > 0 && w < 6) wd++; } miss = wd >= 1; } else miss = d > 1.9 * med && d < 12 * 36e5; if (miss) gi.push(i - 1, i); }
      add('u_gap', 'gaps', 'Missing rows in the time series', gi, rn => `The spacing between rows ${list(rn)} is wider than the file's usual interval, so rows are missing. A return computed across each break covers more than one period.`);
    }
  }
  const pk = s.cols.filter((c, i) => /open|high|low|close|price|bid|ask|last|^o$|^h$|^l$|^c$/.test(H[i])).map(c => c[0]);
  const hk = col(/^high$|^h$/), lk = col(/^low$|^l$/), ck = col(/close|last|^price|^c$|bid/), vk = col(/vol|size|qty|quantity/);
  const neg = []; R.forEach((r, i) => { if (pk.some(k => { const v = num(r, k); return v != null && v <= 0; })) neg.push(i); });
  add('u_neg', 'prices', 'Zero or negative prices', neg, rn => `Rows ${list(rn)} contain a price at or below zero. That is usually a placeholder for a missing value. Set it to missing.`);
  if (hk && lk) { const inv = []; R.forEach((r, i) => { const h = num(r, hk), l = num(r, lk); if (h != null && l != null && h < l) inv.push(i); }); add('u_inv', 'prices', 'High below low', inv, rn => `On rows ${list(rn)} the high is below the low. The two fields are likely swapped.`); }
  if (ck) {
    const c = R.map(r => num(r, ck)), fat = [], spl = [], near = x => [2, 3, 4, 5, 10, 1.5].some(k => Math.abs(x - k) / k < 0.04 || Math.abs(x - 1 / k) * k < 0.04);
    for (let i = 1; i < N; i++) { if (!(c[i] > 0 && c[i - 1] > 0)) continue; const q = c[i] / c[i - 1]; if (Math.abs(Math.log(q)) < Math.log(1.3)) continue; const nx = c[i + 1]; if (nx > 0 && Math.abs(Math.log(nx / c[i - 1])) < Math.log(1.1)) fat.push(i); else if (near(q)) spl.push(i - 1, i); }
    add('u_fat', 'prices', 'One-row price spike', fat, rn => `Price on rows ${list(rn)} moves more than 30% and reverts on the next row. That is a bad print or a misplaced decimal, not a market move.`);
    add('u_split', 'splits', 'Possible unadjusted split', spl, rn => `Price changes by a clean ratio between rows ${list(rn)} and stays at the new level. That matches a split or reverse split that was not back-adjusted.`);
  }
  if (vk) { const z = []; R.forEach((r, i) => { if (num(r, vk) === 0) z.push(i); }); add('u_zero', 'prices', 'Zero-volume rows', z, rn => `Rows ${list(rn)} report zero volume. If prices are also flat these are filler rows from a feed outage.`); }
  s.ms = Math.min(4200, 1600 + N * 14);
}
function parseCSV(text, name) {
  const lines = text.replace(/\r/g, '').split('\n').filter(l => l.trim());
  if (lines.length < 2) throw new Error('The file needs a header row and at least one data row.');
  const split = l => { const o = []; let cur = '', q = false; for (const ch of l) { if (ch === '"') q = !q; else if (ch === ',' && !q) { o.push(cur); cur = ''; } else cur += ch; } o.push(cur); return o.map(x => x.trim()); };
  const head = split(lines[0]), body = lines.slice(1, 501).map(split);
  const cols = head.map((h, i) => ['c' + i, h || 'col' + (i + 1), 'str']);
  const rows = body.map((a, i) => mk(Object.fromEntries(head.map((_, j) => ['c' + j, a[j] ?? ''])), i));
  const s = { id: 'upload', name, file: name, desc: 'Your file', cols, rows, ms: 0, found: [], issues: [], upload: true, total: lines.length - 1 };
  heur(s); return finish(s);
}
function toCSV(s) { return [s.cols.map(c => c[1]).join(',')].concat(s.rows.map(r => s.cols.map(c => fmtV(r.cells[c[0]], c[2]).replace('−', '-').replace(/,/g, '')).join(','))).join('\n'); }
function cur() { return state.upload || S.find(s => s.id === state.sid); }
function serverLive() { return !!appConfig.live; }
/* Mode of the current/last run, snapshotted when the run starts so a
   late /api/config response or a mid-run failure cannot change what
   the timer is waiting for. Falls back to the server mode when idle. */
function simulatedRun() { return state.runMode ? state.runMode !== 'live' : !serverLive(); }
function cancelInFlight() {
  currentRunId++;
  if (currentAbort) { currentAbort.abort(); currentAbort = null; }
  pendingLive = false;
}
function score(s, findings) {
  if (s.upload) return null;
  const m = new Set();
  const verdict = findings.map(f => { const hit = s.issues.filter(is => is.step === f.step && is.rows.some(n => (f.rows || []).includes(n))); const t = hit.find(is => !m.has(is.id)) || hit[0]; if (t) { m.add(t.id); return 'ok'; } return 'fp'; });
  const caught = m.size, total = s.issues.length, ff = verdict.filter(v => v === 'fp').length, val = Math.max(0, caught - 0.5 * ff) / total;
  return { caught, total, ff, missed: s.issues.filter(i => !m.has(i.id)), val, pass: val >= 0.8, verdict, matched: m };
}
function scrollToApp() { const el = $('app'); if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 68, behavior: 'smooth' }); }

/* ---------------- run / file handling ---------------- */
function run() {
  if (state.stage === 'running') return;
  clearInterval(timer);
  cancelInFlight();
  const runId = currentRunId;
  const s = cur(), live = serverLive(), t0 = performance.now();
  Object.assign(state, { step: 'analyze', stage: 'running', elapsed: 0, live: null, err: null, limitHit: false, liveAttempted: false, active: -1, view: 'raw', runMode: live ? 'live' : 'simulated' });
  if (live) {
    // Only set this when a request actually goes out, so a late
    // /api/config "live" flip cannot claim a failed live call.
    state.liveAttempted = true;
    pendingLive = true;
    const controller = new AbortController();
    currentAbort = controller;
    fetch('/api/check', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal, body: JSON.stringify({ filename: s.file, csv: toCSV(s) }) })
      .then(async r => {
        if (!r.ok) {
          // Keep the server's status + message so a 429 can be shown
          // as a friendly limit notice instead of a generic failure.
          let serverError = '';
          try { serverError = (await r.json()).error || ''; } catch (e) { /* non-JSON error body */ }
          const err = new Error(serverError || ('Endpoint returned HTTP ' + r.status));
          err.status = r.status;
          err.serverError = serverError;
          throw err;
        }
        return r.json();
      })
      .then(j => {
        if (runId !== currentRunId) return; // stale run: a newer run owns the UI
        pendingLive = false;
        state.live = { findings: (j.findings || []).map(f => ({ step: f.type || f.step, title: f.title || 'Problem', rows: (f.rows || []).map(Number), explain: f.explain || f.explanation || '' })), ms: Math.round(j.latency_ms || performance.now() - t0), model: j.model };
      })
      .catch(fetchErr => {
        if (runId !== currentRunId) return; // aborted/superseded: stay silent
        pendingLive = false;
        // Fall back to simulated *within this run*: flipping runMode is
        // safe because the timer reads the per-run snapshot, and this
        // run now completes on simulated pacing instead of hanging.
        state.runMode = 'simulated';
        if (fetchErr && fetchErr.status === 429) {
          state.limitHit = true;
          state.err = 'Live checks are paused — ' + (fetchErr.serverError || 'usage limit reached') + '. Showing simulated demo results, not a live model run.';
        } else {
          state.err = 'Live call failed — showing simulated demo results, not a live model run.';
        }
        renderMode();
      });
  }
  render();
  timer = setInterval(() => {
    if (runId !== currentRunId) { clearInterval(timer); return; }
    const el = performance.now() - t0;
    const fin = ms => { clearInterval(timer); state.stage = 'done'; state.elapsed = ms; render(); };
    if (state.runMode === 'live') { if (!pendingLive && state.live) fin(state.live.ms); else { state.elapsed = el; renderAnalyze(); renderStepper(); } }
    else if (el >= s.ms) fin(s.ms); else { state.elapsed = el; renderAnalyze(); renderStepper(); }
  }, 40);
}
function trySample() { clearInterval(timer); cancelInFlight(); Object.assign(state, { sid: 's1', upload: null, stage: 'idle', step: 'data', live: null, err: null, limitHit: false, liveAttempted: false, runMode: null }); render(); scrollToApp(); setTimeout(run, 500); }
function pickFile() { const f = $('fileInput'); if (f) f.click(); }
function loadFile(file) {
  if (!file) return; const rd = new FileReader();
  rd.onload = () => { try { const up = parseCSV(String(rd.result), file.name); clearInterval(timer); cancelInFlight(); Object.assign(state, { upload: up, err: null, limitHit: false, liveAttempted: false, stage: 'idle', step: 'data', live: null, runMode: null }); render(); scrollToApp(); } catch (e) { state.err = e.message; state.step = 'data'; render(); scrollToApp(); } };
  rd.readAsText(file);
}

/* ---------------- rendering ---------------- */
function fmtCell(v, d, extra) { return { v: fmtV(v, d), al: d === 'str' ? 'left' : 'right', col: 'var(--text-2)', td: 'none', ...(extra || {}) }; }
function badge(tone, label, sm) {
  const tones = { pass: 'var(--status-pass)', fail: 'var(--status-fail)', warn: 'var(--status-warn)', info: 'var(--status-info)', neutral: null };
  const dot = tones[tone] ? `<span class="badge-dot" style="background:${tones[tone]}"></span>` : '';
  return `<span class="badge${sm ? ' badge-sm' : ''}">${dot}${esc(label)}</span>`;
}
function currentFindings() { return state.stage === 'done' ? (state.live ? state.live.findings : cur().findings) : []; }

function renderMode() {
  const fallback = state.runMode === 'simulated' && state.liveAttempted;
  const live = (state.runMode === 'live') || (!state.runMode && serverLive());
  $('modeDot').style.background = live ? 'var(--accent)' : 'var(--warning-500)';
  $('modeText').textContent = live
    ? 'live · /api/check · ' + (appConfig.model || 'muse-spark')
    : fallback
      ? state.limitHit
        ? 'live limit reached · showing simulated demo results'
        : 'live call failed · showing simulated demo results'
      : 'demo mode · no API key configured — simulated responses';
  const banner = $('demoBanner');
  if (banner) {
    banner.hidden = live;
    const bannerText = $('demoBannerText');
    if (bannerText && fallback) bannerText.textContent = state.limitHit
      ? 'Live checks are paused because a usage limit was reached, so the findings, timings, and scores below are simulated demo results — not a live model run.'
      : 'The live Muse call failed, so the findings, timings, and scores below are simulated demo results — not a live model run.';
  }
}

function renderStepper() {
  const st = state, s = cur();
  const running = st.stage === 'running', done = st.stage === 'done';
  const findings = currentFindings();
  const sc = done ? score(s, findings) : null;
  const model = (st.live && st.live.model) || appConfig.model || 'muse-spark';
  const order = ['data', 'analyze', 'findings', 'score'], ci = order.indexOf(st.step);
  const subs = { data: s.file, analyze: running ? 'running…' : done ? (simulatedRun() ? 'simulated' : fmtV(st.elapsed, 'int') + ' ms') : (serverLive() ? model : 'simulated'), findings: done ? findings.length + ' found' : '—', score: sc ? sc.val.toFixed(2) + (simulatedRun() ? ' · demo' : (sc.pass ? ' · pass' : ' · fail')) : (done && s.upload ? 'no answer key' : '—') };
  $('stepper').innerHTML = [['data', 'Data'], ['analyze', 'Analyze'], ['findings', 'Findings'], ['score', 'Score']].map(([k, label], i) => {
    const enabled = k === 'data' || k === 'analyze' ? (k === 'data' || st.stage !== 'idle') : done;
    const isCur = st.step === k, past = i < ci;
    const numBg = isCur ? 'var(--accent)' : 'transparent';
    const numCol = isCur ? 'var(--neutral-900)' : past ? 'var(--accent)' : 'var(--text-3)';
    const numRing = isCur ? 'var(--accent)' : past ? 'var(--accent)' : 'var(--line-2)';
    return `<div class="step" data-action="${enabled && !running ? 'go' : ''}" data-step="${k}" style="cursor:${enabled ? 'pointer' : 'default'};background:${isCur ? 'var(--surface-raised)' : 'transparent'};box-shadow:${isCur ? 'inset 0 -2px 0 var(--accent)' : 'none'};opacity:${enabled || isCur ? 1 : 0.45}">
      <span class="step-num" style="background:${numBg};color:${numCol};box-shadow:inset 0 0 0 1px ${numRing}">${i + 1}</span>
      <div class="step-labels"><span class="step-label">${label}</span><span class="step-sub">${esc(subs[k])}</span></div></div>`;
  }).join('');
}

function tableHead(s, withNote) {
  const heads = s.cols.map(([, l, d]) => `<span class="cell" style="text-align:${d === 'str' ? 'left' : 'right'}">${esc(l)}</span>`).join('');
  return `<div class="table-head-row"><span class="cell-num">#</span>${heads}${withNote ? '<span class="cell-note">note</span>' : ''}</div>`;
}

function renderData() {
  const st = state, s = cur();
  const sampleList = S.map(x => ({ ...x, ring: !st.upload && st.sid === x.id ? 'inset 0 0 0 1px var(--accent)' : 'inset 0 0 0 1px var(--line-1)' }));
  if (st.upload) sampleList.push({ ...st.upload, desc: 'Your file', file: st.upload.total + ' rows', ring: 'inset 0 0 0 1px var(--accent)', id: 'upload' });
  const previewRows = s.rows.slice(0, 200).map(r => `<div class="table-row"><span class="cell-num">${r.n}</span>${s.cols.map(([k, , d]) => { const c = fmtCell(r.cells[k], d); return `<span class="cell" style="text-align:${c.al}">${esc(c.v)}</span>`; }).join('')}</div>`).join('');
  $('stepData').innerHTML = `<div class="step-body">
    <div class="col-samples">
      <span class="card-label samples-label">Messy samples</span>
      ${sampleList.map(sm => `<div class="sample-card" data-action="sample" data-id="${sm.id}" style="box-shadow:${sm.ring}"><div class="sample-card-top"><span class="sample-name">${esc(sm.name)}</span><span class="sample-desc">${esc(sm.desc)}</span></div><span class="sample-file">${esc(sm.file)}</span></div>`).join('')}
      <div class="drop-zone${st.drag ? ' dragging' : ''}" id="dropZone" data-action="pick-file">
        <span class="drop-title">Drop your own CSV</span><span class="drop-sub">or click to browse · first 500 rows</span>
      </div>
      ${st.err ? `<span class="form-error">${esc(st.err)}</span>` : ''}
    </div>
    <div class="col-main table-card">
      <div class="table-card-head">
        <div class="table-file"><span class="table-file-name">${esc(s.file)}</span><span class="table-file-meta">${(s.total || s.rows.length)} rows × ${s.cols.length} cols${s.total > 500 ? ' · first 500 analyzed' : ''}</span></div>
        <button class="btn btn-primary btn-md" data-action="run" type="button">${serverLive() ? 'Run Muse Spark →' : 'Run simulated check →'}</button>
      </div>
      <div class="table-scroll"><div class="table-inner">${tableHead(s, false)}${previewRows}</div></div>
    </div></div>`;
}

function renderAnalyze() {
  const el = $('stepAnalyze'); if (!el || state.step !== 'analyze') return;
  const st = state, s = cur(), simulated = simulatedRun(), live = !simulated;
  const running = st.stage === 'running', done = st.stage === 'done';
  const model = simulated ? 'Simulated demo — no model called' : ((st.live && st.live.model) || appConfig.model || 'muse-spark');
  const total = live ? 5000 : s.ms, fr = done ? 1 : Math.min(st.elapsed / total, 0.97);
  const simF = live ? (st.live ? st.live.findings : []) : s.findings;
  let cum = 0;
  const stepsHtml = STEPS.map(([k, label, w]) => {
    const a = cum; cum += w; const isDone = done || fr >= cum, isRun = running && !isDone && fr >= a;
    const n = simF.filter(f => f.step === k).length, showN = isDone && (!live || done);
    const count = !isDone ? (isRun ? 'running' : '') : k === 'read' ? s.rows.length + ' rows · ' + s.cols.length + ' cols' : (showN ? (n ? n + ' found' : 'clear') : '');
    const dot = isDone ? (k !== 'read' && n ? TYPES[k][1] : 'var(--accent)') : isRun ? 'var(--lime-500)' : 'var(--line-2)';
    return `<div class="analyze-step"><span class="analyze-dot" style="background:${dot};animation:${isRun ? 'lmPulse 900ms ease-in-out infinite' : 'none'}"></span><span class="analyze-step-label" style="color:${isDone || isRun ? 'var(--text-1)' : 'var(--text-3)'}">${label}</span><span class="analyze-step-count" style="color:${isDone && k !== 'read' && n ? TYPES[k][1] : 'var(--text-3)'}">${esc(count)}</span><span class="analyze-step-ms"></span></div>`;
  }).join('');
  const logLines = [];
  if (st.stage !== 'idle') {
    if (simulated) {
      if (state.liveAttempted) {
        // A live request was actually sent (or blocked); don't claim none was.
        logLines.push({ t: `→ POST /api/check  model=${appConfig.model || 'muse-spark'}`, c: 'var(--text-2)' });
        logLines.push({ t: `  file=${s.file}  rows=${s.rows.length}  cols=${s.cols.length}`, c: 'var(--text-3)' });
        logLines.push({ t: st.limitHit
          ? '[simulated] Live request was blocked by a usage limit — showing demo data, not a model result'
          : '[simulated] Live request failed — showing demo data, not a model result', c: 'var(--warning-500)' });
      } else {
        logLines.push({ t: '[simulated] No API request was sent — demo data only, no model was called', c: 'var(--warning-500)' });
        logLines.push({ t: `  file=${s.file}  rows=${s.rows.length}  cols=${s.cols.length}`, c: 'var(--text-3)' });
      }
      let c2 = 0; STEPS.forEach(([k, , w]) => { c2 += w; if (done || fr >= c2) simF.filter(f => f.step === k).forEach(f => logLines.push({ t: `[simulated] {"type":"${k}","rows":[${(f.rows || []).join(',')}],"title":"${f.title}"}`, c: TYPES[k] ? TYPES[k][1] : 'var(--text-1)' })); });
      if (running) logLines.push({ t: '▍', c: 'var(--accent)' });
      if (done) logLines.push({ t: `Demo complete · ${currentFindings().length} simulated findings`, c: 'var(--warning-500)' });
    } else {
      logLines.push({ t: `→ POST /api/check  model=${model}`, c: 'var(--text-2)' });
      logLines.push({ t: `  file=${s.file}  rows=${s.rows.length}  cols=${s.cols.length}`, c: 'var(--text-3)' });
      let c2 = 0; STEPS.forEach(([k, , w]) => { c2 += w; if (done || fr >= c2) simF.filter(f => f.step === k).forEach(f => logLines.push({ t: `{"type":"${k}","rows":[${(f.rows || []).join(',')}],"title":"${f.title}"}`, c: TYPES[k] ? TYPES[k][1] : 'var(--text-1)' })); });
      if (running) logLines.push({ t: '▍', c: 'var(--accent)' });
      if (done) logLines.push({ t: `← 200 OK  ${fmtV(st.elapsed, 'int')} ms  ·  ${currentFindings().length} findings`, c: 'var(--accent)' });
    }
    if (st.err) logLines.push({ t: '← ' + st.err, c: 'var(--status-fail)' });
  }
  const statusLabel = simulated && st.stage === 'done' ? 'Demo complete' : { idle: 'Ready', running: 'Running', done: 'Complete', error: 'Error' }[st.stage];
  const statusTone = { idle: 'neutral', running: 'warn', done: 'pass', error: 'fail' }[st.stage];
  el.innerHTML = `<div class="step-body step-body-stretch">
    <div class="analyze-card">
      <div class="analyze-head">
        <div class="analyze-field"><span class="card-label">Model</span><span class="analyze-model" style="${simulated ? 'font-size:16px;color:var(--warning-500)' : ''}"><img src="/static/assets/meta-ai.png" alt="">${esc(model)}</span></div>
        <div class="analyze-field analyze-field-end"><span class="card-label">${simulated ? 'Simulated run' : 'Response time'}</span><span class="analyze-elapsed">${simulated ? '—' : fmtV(st.elapsed, 'int') + '<small>ms</small>'}</span></div>
      </div>
      <div class="progress-track"><div class="progress-bar" style="width:${Math.round(fr * 100)}%"></div></div>
      <div class="analyze-steps">${stepsHtml}</div>
      <div class="analyze-status">${badge(statusTone, statusLabel, true)}${st.err ? `<span class="form-error" style="font-size:13px">${esc(st.err)}</span>` : ''}</div>
    </div>
    <div class="stream-card">
      <div class="stream-head"><span>${simulated ? (state.liveAttempted ? 'simulated stream — live request failed' : 'simulated stream — no request sent') : 'response stream'}</span><span>${done ? currentFindings().length + ' findings' : running ? 'streaming' : ''}</span></div>
      <div class="stream-log">${logLines.map(ln => `<span style="color:${ln.c}">${esc(ln.t)}</span>`).join('')}</div>
      ${done ? '<div class="stream-foot"><button class="btn btn-primary btn-sm" data-action="go" data-step="findings" type="button">See findings →</button></div>' : ''}
    </div></div>`;
}

function renderFindings() {
  const st = state, s = cur();
  const findings = currentFindings();
  const sc = state.stage === 'done' ? score(s, findings) : null;
  const fixed = sc ? sc.matched : (state.stage === 'done' && s.upload && !state.live ? new Set(s.found) : new Set());
  const rowF = {}; findings.forEach((f, i) => (f.rows || []).forEach(n => { if (rowF[n] == null) rowF[n] = i; }));
  const act = st.active, actRows = act >= 0 && findings[act] ? new Set(findings[act].rows) : null;
  const cards = findings.map((f, i) => {
    const ty = TYPES[f.step] || [f.step, 'var(--text-2)'];
    return `<div class="finding-card" data-action="card" data-index="${i}" style="box-shadow:${act === i ? 'inset 0 0 0 1px ' + ty[1] : 'inset 0 0 0 1px var(--line-1)'}">
      <div class="finding-card-top"><span class="finding-type"><span class="finding-type-dot" style="background:${ty[1]}"></span>${esc(ty[0])}</span><span class="finding-rows">${esc(rowsTxt(f.rows))}</span></div>
      <span class="finding-title">${esc(f.title)}</span><p class="finding-explain">${esc(f.explain)}</p></div>`;
  }).join('');
  const tableRows = [];
  s.rows.slice(0, 200).forEach(r => {
    const fi = rowF[r.n], f = fi != null ? findings[fi] : null, ty = f ? (TYPES[f.step] || ['', 'var(--text-2)']) : null;
    const dim = actRows && !actRows.has(r.n) ? 0.35 : 1, on = actRows && actRows.has(r.n);
    if (st.view === 'raw') {
      tableRows.push({ isRow: true, html: `<div class="table-row" style="background:${f ? `color-mix(in oklch,${ty[1]} ${on ? 26 : 13}%,transparent)` : 'transparent'};box-shadow:${on ? 'inset 0 0 0 1px ' + ty[1] : 'none'};opacity:${dim}"><span class="cell-num">${r.n}</span>${s.cols.map(([k, , d]) => { const c = fmtCell(r.cells[k], d, f ? { col: 'var(--text-1)' } : {}); return `<span class="cell" style="text-align:${c.al};color:${c.col}">${esc(c.v)}</span>`; }).join('')}<span class="cell-note" style="color:${f ? ty[1] : 'var(--text-3)'}">${f ? esc(ty[0]) : ''}</span></div>` });
    } else {
      let status = '', cells, noteCol = 'var(--text-3)';
      if (r.remove && fixed.has(r.remove)) { status = 'removed'; cells = s.cols.map(([k, , d]) => fmtCell(r.cells[k], d, { col: 'var(--text-3)', td: 'line-through' })); }
      else if (r.clean && fixed.has(r.cleanFor)) { status = 'fixed'; noteCol = 'var(--accent)'; cells = s.cols.map(([k, , d]) => { const ch = fmtV(r.clean[k], d) !== fmtV(r.cells[k], d); return fmtCell(r.clean[k], d, ch ? { col: 'var(--accent)' } : {}); }); }
      else { cells = s.cols.map(([k, , d]) => fmtCell(r.cells[k], d)); if (f) { status = 'flagged'; noteCol = 'var(--warning-500)'; } }
      tableRows.push({ isRow: true, note: status, html: `<div class="table-row" style="background:${status === 'fixed' ? 'color-mix(in oklch,var(--accent) 10%,transparent)' : 'transparent'};box-shadow:${on ? 'inset 0 0 0 1px var(--line-2)' : 'none'};opacity:${dim}"><span class="cell-num">${r.n}</span>${cells.map(c => `<span class="cell" style="text-align:${c.al};color:${c.col};text-decoration:${c.td}">${esc(c.v)}</span>`).join('')}<span class="cell-note" style="color:${noteCol}">${status}</span></div>` });
    }
    if (st.view === 'clean' && r.gap && fixed.has(r.gap.for)) tableRows.push({ isRow: false, html: `<div class="gap-row">${esc(r.gap.text)}</div>` });
  });
  const kept = st.view === 'clean' ? tableRows.filter(x => x.isRow && x.note !== 'removed').length : s.rows.length;
  $('stepFindings').innerHTML = `<div class="step-body">
    <div class="findings-col">
      ${state.stage === 'done' && !findings.length ? '<div class="empty-card">No problems reported.</div>' : cards}
      <button class="btn btn-primary btn-md" data-action="go" data-step="score" type="button" style="align-self:flex-start">See the score →</button>
    </div>
    <div class="col-main table-card">
      <div class="table-card-head">
        <span class="table-caption">${st.view === 'raw' ? 'Before · ' + Object.keys(rowF).length + ' rows flagged' : 'After · ' + kept + ' rows kept' + (s.upload || st.live ? '' : ' · fixes applied for caught problems only')}</span>
        <div class="view-toggle">
          <button class="btn btn-sm ${st.view === 'raw' ? 'btn-accent' : 'btn-ghost'}" data-action="view" data-view="raw" type="button">Before</button>
          <button class="btn btn-sm ${st.view === 'clean' ? 'btn-accent' : 'btn-ghost'}" data-action="view" data-view="clean" type="button">After</button>
        </div>
      </div>
      <div class="table-scroll table-scroll-tall"><div class="table-inner-wide">${tableHead(s, true)}${tableRows.map(r => r.html).join('')}</div></div>
    </div></div>`;
}

function renderScore() {
  const st = state, s = cur();
  const findings = currentFindings();
  const sc = state.stage === 'done' ? score(s, findings) : null;
  let html = '<div class="score-body">';
  if (state.stage === 'done' && !sc) {
    html += '<div class="score-empty">Uploaded files have no answer key, so they can\'t be scored. Run one of the seeded samples to see a score.</div>';
  }
  if (sc) {
    const simulated = simulatedRun();
    const verdicts = findings.map((f, i) => ({ tag: sc.verdict[i] === 'ok' ? 'Caught' : 'False flag', tone: sc.verdict[i] === 'ok' ? 'pass' : 'warn', title: f.title, rowsTxt: rowsTxt(f.rows), note: sc.verdict[i] === 'ok' ? '' : (f.note || 'Does not match any planted problem.'), hasNote: sc.verdict[i] !== 'ok' }))
      .concat(sc.missed.map(m => ({ tag: 'Missed', tone: 'fail', title: m.title, rowsTxt: rowsTxt(m.rows), note: m.explain, hasNote: true })));
    html += `<div class="score-flex">
      <div class="score-main" style="box-shadow:inset 0 0 0 1px ${sc.pass ? 'var(--accent)' : 'var(--status-fail)'}">
        <div class="score-main-top"><span class="card-label">Deterministic scorer${simulated ? ' (demo)' : ''}</span>${badge(sc.pass ? 'pass' : 'fail', simulated ? (sc.pass ? 'Demo pass' : 'Demo fail') : (sc.pass ? 'Pass' : 'Fail'), false)}</div>
        <div class="score-value-row"><span class="score-value" style="color:${sc.pass ? 'var(--accent)' : 'var(--status-fail)'}">${sc.val.toFixed(2)}</span><span class="score-scale">/ 1.00 · pass at 0.80</span></div>
        <span class="score-formula">score = (caught − 0.5 × false flags) ÷ planted${simulated ? '<br>Demo score against the seeded answer key — not a live model result.' : ''}</span>
      </div>
      <div class="score-side">
        <div class="score-tiles">
          <div class="metric-tile" style="background:var(--surface-sunken)"><span class="metric-label">Caught</span><span class="metric-value">${sc.caught} / ${sc.total}</span></div>
          <div class="metric-tile" style="background:var(--surface-sunken)"><span class="metric-label">Missed</span><span class="metric-value">${sc.missed.length}</span></div>
          <div class="metric-tile" style="background:var(--surface-sunken)"><span class="metric-label">Falsely flagged</span><span class="metric-value">${sc.ff}</span></div>
        </div>
        ${verdicts.map(v => `<div class="verdict-card"><div class="verdict-top"><span class="verdict-title-wrap">${badge(v.tone, v.tag, true)}<span class="verdict-title">${esc(v.title)}</span></span><span class="verdict-rows">${esc(v.rowsTxt)}</span></div>${v.hasNote ? `<p class="verdict-note">${esc(v.note)}</p>` : ''}</div>`).join('')}
      </div></div>`;
  }
  html += '<div class="score-actions"><button class="btn btn-secondary btn-md" data-action="restart" type="button">Try another file</button></div></div>';
  $('stepScore').innerHTML = html;
}

/* ---------------- Leaderboard (held-out results, both tiers) ----------------
   All numbers come from GET /api/leaderboard, which serves the single
   data file app/data/leaderboard.json (totals and Wilson intervals are
   recomputed server-side). Nothing is hard-coded here. The file holds
   two boards: the frontier tools-off/tools-on comparison and the
   tools_frontier per-family board (tools on, 12 turns). Approximate
   rates (~85%) and partial runs are rendered exactly as labelled in
   the data — no exact count is invented for them. */
function renderLeaderboard(data) {
  const boards = data.boards && data.boards.length ? data.boards
    : [{ id: 'legacy', title: data.title || 'Leaderboard', families: (data.benchmark && data.benchmark.families) || [], note: data.benchmark && data.benchmark.note, notes: (data.benchmark && data.benchmark.notes) || [], entries: data.entries || [] }];
  const famPct = f => Math.round(100 * f.passed / f.total);
  const pctTxt = t => (Number.isInteger(t.pct) ? t.pct : t.pct.toFixed(1)) + '%';
  const resultCell = (v, totalCell) => {
    if (!v) return '<div class="lb-cell lb-dim">—</div>';
    if (v.display && v.passed == null) return `<div class="lb-cell"><span class="lb-frac">${esc(v.display)}</span></div>`;
    if (v.approx) return `<div class="lb-cell lb-rate"><span class="lb-pct">${esc(v.display || pctTxt(v))}</span><div class="lb-mini-track"><div class="lb-mini-fill" style="width:${Math.max(Math.round(v.pct), 2)}%"></div></div><span class="lb-ci">approximate, as reported</span></div>`;
    const ci = totalCell && v.ci95 ? `<span class="lb-ci">95% CI ${v.ci95[0]}–${v.ci95[1]}%</span>` : '';
    return `<div class="lb-cell${totalCell ? ' lb-rate' : ''}"><span class="lb-frac${totalCell ? ' lb-pct' : ''}">${totalCell ? pctTxt(v) : `${v.passed}/${v.total}`}</span><div class="lb-mini-track"><div class="lb-mini-fill" style="width:${Math.max(famPct(v), v.passed ? 2 : 0)}%"></div></div>${ci}</div>`;
  };
  const notes = [];
  $('leaderboardBody').innerHTML = boards.map(board => {
    const fams = board.families || [];
    const isCompare = board.entries.some(e => e.tools_off || e.tools_on);
    if (board.note) notes.push(board.note);
    (board.notes || []).forEach(n => notes.push(n));
    board.entries.forEach(e => (e.notes || []).forEach(n => {
      const line = e.name + ': ' + n;
      if (!notes.includes(line)) notes.push(line);
    }));
    const head = board.headline ? `<div class="lb-board-head"><h3>${esc(board.title)}</h3><p>${esc(board.headline)}</p></div>` : `<div class="lb-board-head"><h3>${esc(board.title)}</h3></div>`;
    if (isCompare) {
      const rows = board.entries.map(e => {
        const nameCell = `<div class="lb-model"><span class="lb-model-name">${esc(e.name)}</span><span class="lb-model-sub">${esc(e.provider || '')}${e.scope ? ` · <span class="lb-chip">${esc(e.scope)}</span>` : ''}</span></div>`;
        if (e.status === 'coming_soon') return `<div class="lb-row lb-row-compare lb-pending">${nameCell}<div class="lb-cell lb-dim">—</div><div class="lb-cell"><span class="lb-chip lb-chip-soon">Coming soon</span></div></div>`;
        const off = e.tools_off ? (e.tools_off.passed == null ? resultCell(e.tools_off, true) : `<div class="lb-cell"><span class="lb-frac">${e.tools_off.passed}/${e.tools_off.total} (${pctTxt(e.tools_off)})</span><div class="lb-mini-track"><div class="lb-mini-fill lb-mini-off" style="width:${Math.max(famPct(e.tools_off), e.tools_off.passed ? 2 : 0)}%"></div></div></div>`) : '<div class="lb-cell lb-dim">—</div>';
        const on = e.tools_on ? (e.tools_on.passed == null ? resultCell(e.tools_on, true) : `<div class="lb-cell lb-rate"><span class="lb-frac lb-pct">${e.tools_on.passed}/${e.tools_on.total} (${pctTxt(e.tools_on)})</span><div class="lb-mini-track"><div class="lb-mini-fill" style="width:${Math.max(famPct(e.tools_on), e.tools_on.passed ? 2 : 0)}%"></div></div></div>`) : '<div class="lb-cell lb-dim">—</div>';
        return `<div class="lb-row lb-row-compare">${nameCell}${off}${on}</div>`;
      }).join('');
      return `<div class="lb-board">${head}<div class="lb-row lb-row-compare lb-head"><div class="lb-model">Model</div><div class="lb-cell">Tools off</div><div class="lb-cell">Tools on</div></div>${rows}</div>`;
    }
    const rows = board.entries.map(e => {
      const nameCell = `<div class="lb-model"><span class="lb-model-name">${esc(e.name)}</span><span class="lb-model-sub">${esc(e.provider || '')}${e.scope ? ` · <span class="lb-chip">${esc(e.scope)}</span>` : ''}</span></div>`;
      if (e.status === 'coming_soon') {
        return `<div class="lb-row lb-row-tf lb-pending">${nameCell}` +
          fams.map(() => '<div class="lb-cell lb-dim">—</div>').join('') +
          `<div class="lb-cell lb-dim">—</div><div class="lb-cell"><span class="lb-chip lb-chip-soon">Coming soon</span></div></div>`;
      }
      const famCells = fams.map(f => {
        const v = e.families && e.families[f.id];
        if (!v) return '<div class="lb-cell lb-dim">—</div>';
        if (v.passed == null) return `<div class="lb-cell"><span class="lb-frac">${esc(v.display || '—')}</span></div>`;
        return `<div class="lb-cell"><span class="lb-frac">${v.passed}/${v.total}</span><div class="lb-mini-track"><div class="lb-mini-fill" style="width:${Math.max(famPct(v), v.passed ? 2 : 0)}%"></div></div></div>`;
      }).join('');
      const t = e.total;
      const totalCell = t ? `<div class="lb-cell"><span class="lb-frac lb-total">${t.passed}/${t.total}</span></div>${resultCell(t, true)}` : '<div class="lb-cell lb-dim">—</div><div class="lb-cell lb-dim">—</div>';
      return `<div class="lb-row lb-row-tf">${nameCell}${famCells}${totalCell}</div>`;
    }).join('');
    return `<div class="lb-board">${head}<div class="lb-row lb-row-tf lb-head"><div class="lb-model">Model</div>` +
      fams.map(f => `<div class="lb-cell"><span class="lb-fam">${esc(f.id)} · ${esc(f.label)}</span>${f.type ? `<span class="lb-fam-type">${esc(f.type)}</span>` : ''}</div>`).join('') +
      `<div class="lb-cell">Total</div><div class="lb-cell">Pass rate</div></div>${rows}</div>`;
  }).join('');
  $('lbEyebrow').textContent = 'Leaderboard · frontier + tools_frontier · held-out tasks';
  if (data.updated) $('lbUpdated').textContent = 'held-out run · ' + data.updated;
  $('leaderboardNotes').innerHTML = notes.map(n => `<p>${esc(n)}</p>`).join('');
}

function loadLeaderboard() {
  fetch('/api/leaderboard').then(r => r.ok ? r.json() : null).then(data => {
    if (data && (data.boards || data.entries)) renderLeaderboard(data);
    else $('leaderboardBody').innerHTML = '<p class="lb-loading">Leaderboard unavailable right now.</p>';
  }).catch(() => { $('leaderboardBody').innerHTML = '<p class="lb-loading">Leaderboard unavailable right now.</p>'; });
}

function render() {
  renderMode(); renderStepper(); renderData(); renderAnalyze(); renderFindings(); renderScore();
  $('stepData').hidden = state.step !== 'data';
  $('stepAnalyze').hidden = state.step !== 'analyze';
  $('stepFindings').hidden = state.step !== 'findings';
  $('stepScore').hidden = state.step !== 'score';
}

/* ---------------- WaveLines (canvas, ported from design system) ---------------- */
function initWaves() {
  const c = $('waveCanvas'); if (!c) return;
  const ctx = c.getContext('2d');
  const lines = 56, amplitude = 0.16, spread = 0.2, tilt = 0.25, speed = 1, opacity = 0.5, lineWidth = 1;
  const colors = ['#009D5E', '#DDFF00'];
  const hex = s => { s = s.replace('#', ''); return [0, 2, 4].map(i => parseInt(s.slice(i, i + 2), 16)); };
  const ca = hex(colors[0]), cb = hex(colors[1]);
  let raf = 0, w = 0, h = 0; const t0 = performance.now();
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function draw(now) {
    const time = reduce ? 2 : (now - t0) / 1000 * speed;
    ctx.clearRect(0, 0, w, h); ctx.lineWidth = lineWidth;
    const steps = Math.max(80, Math.round(w / 5));
    for (let i = 0; i < lines; i++) {
      const k = lines > 1 ? i / (lines - 1) : 0;
      const col = ca.map((v, j) => Math.round(v + (cb[j] - v) * k)).join(',');
      ctx.strokeStyle = 'rgba(' + col + ',' + opacity * (0.2 + 0.8 * Math.sin(Math.PI * k)) + ')';
      ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const u = s / steps, x = u * w;
        const env = 0.3 + 0.7 * Math.abs(Math.sin(u * Math.PI * 1.3 + time * 0.35));
        const base = Math.sin(u * Math.PI * 2.1 + time * 0.6) * amplitude * h + Math.sin(u * Math.PI * 0.9 - time * 0.27) * amplitude * 0.6 * h;
        const y = h * 0.5 + (u - 0.5) * tilt * h + base + (k - 0.5) * spread * h * env * 2 + Math.sin(u * 7 + k * 4 + time * 1.3) * 3 * k;
        s ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
    }
    if (!reduce) raf = requestAnimationFrame(draw);
  }
  function fit() {
    const r = c.getBoundingClientRect(); const d = window.devicePixelRatio || 1;
    w = r.width; h = r.height; c.width = Math.max(1, w * d); c.height = Math.max(1, h * d);
    ctx.setTransform(d, 0, 0, d, 0, 0);
    if (reduce) draw(performance.now());
  }
  const ro = new ResizeObserver(fit); ro.observe(c); fit();
  if (!reduce) raf = requestAnimationFrame(draw);
}

/* ---------------- events ---------------- */
document.addEventListener('click', (e) => {
  const t = e.target.closest('[data-action]'); if (!t) return;
  const a = t.dataset.action; if (!a) return;
  if (a === 'go') { state.step = t.dataset.step; render(); }
  else if (a === 'sample') { const id = t.dataset.id; if (id === 'upload') return; clearInterval(timer); cancelInFlight(); Object.assign(state, { sid: id, upload: null, stage: 'idle', step: 'data', live: null, err: null, limitHit: false, liveAttempted: false, active: -1, runMode: null }); render(); }
  else if (a === 'pick-file') pickFile();
  else if (a === 'run') run();
  else if (a === 'card') { const i = Number(t.dataset.index); state.active = state.active === i ? -1 : i; render(); }
  else if (a === 'view') { state.view = t.dataset.view; render(); }
  else if (a === 'restart') { clearInterval(timer); cancelInFlight(); Object.assign(state, { step: 'data', stage: 'idle', active: -1, live: null, err: null, limitHit: false, liveAttempted: false, runMode: null }); render(); scrollToApp(); }
});
$('fileInput').addEventListener('change', (e) => { loadFile(e.target.files && e.target.files[0]); e.target.value = ''; });
$('heroTrySample').addEventListener('click', trySample);
$('heroUpload').addEventListener('click', pickFile);
/* Drag styling toggles the class directly on the node: re-rendering
   here would destroy the drop zone mid-drag and flicker/miss drops. */
document.addEventListener('dragover', (e) => { const z = e.target.closest && e.target.closest('#dropZone'); if (z) { e.preventDefault(); z.classList.add('dragging'); } });
document.addEventListener('dragleave', (e) => { const z = e.target.closest && e.target.closest('#dropZone'); if (z && !(e.relatedTarget instanceof Node && z.contains(e.relatedTarget))) z.classList.remove('dragging'); });
document.addEventListener('drop', (e) => { const z = e.target.closest && e.target.closest('#dropZone'); if (z) { e.preventDefault(); z.classList.remove('dragging'); loadFile(e.dataTransfer.files && e.dataTransfer.files[0]); } });

/* ---------------- boot ---------------- */
render();
initWaves();
fetch('/api/config').then(r => r.ok ? r.json() : null).then(cfg => { if (cfg) { appConfig = cfg; render(); } }).catch(() => {});
loadLeaderboard();
