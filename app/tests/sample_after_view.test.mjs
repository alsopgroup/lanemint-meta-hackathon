/* After-view fix alignment tests for app/static/js/app.js.
 *
 * Run by app/tests/test_sample_answers.py under Node (skipped when no
 * Node runtime is available). Loads app.js with minimal DOM stubs,
 * then drives the real parseCSV / buildSampleScoreIssues /
 * applySampleFixTags / score / renderFindings code:
 *  - a recognised sample upload must remove, in the After table, the
 *    same duplicate (and bad-price) rows the Score tab counts caught;
 *  - if a finding is missed, no fix is applied for it;
 *  - built-in seeded samples and non-matched uploads keep working.
 */
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const appJsPath = path.join(here, '..', 'static', 'js', 'app.js');
const samplesDir = path.join(here, '..', '..', 'samples');

/* ---------------- minimal DOM stubs (only what app.js touches) ---------------- */
const elements = new Map();
function makeEl(id) {
  return {
    id, style: {}, hidden: false, textContent: '', innerHTML: '', dataset: {},
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {}, getContext: () => null,
    getBoundingClientRect: () => ({ top: 0, width: 100, height: 100 }),
  };
}
globalThis.window = globalThis;
globalThis.scrollTo = () => {};
globalThis.document = {
  getElementById(id) {
    if (id === 'waveCanvas') return null;
    if (!elements.has(id)) elements.set(id, makeEl(id));
    return elements.get(id);
  },
  addEventListener() {},
};
globalThis.fetch = () => Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
globalThis.ResizeObserver = class { observe() {} };

const code = fs.readFileSync(appJsPath, 'utf8');
vm.runInThisContext(
  code + '\n;globalThis.__app = { state, S, parseCSV, buildSampleScoreIssues, applySampleFixTags, score, renderFindings };',
  { filename: appJsPath },
);
const { state, parseCSV, buildSampleScoreIssues, applySampleFixTags, score, renderFindings } = globalThis.__app;

function rowSegment(html, n) {
  const marker = `<span class="cell-num">${n}</span>`;
  const at = html.indexOf(marker);
  assert.notEqual(at, -1, `row ${n} not rendered`);
  const start = html.lastIndexOf('<div class="table-row"', at);
  const end = html.indexOf('<div class="table-row"', at);
  return html.slice(start, end === -1 ? undefined : end);
}
function showFindings(upload, extra = {}) {
  Object.assign(state, {
    upload, stage: 'done', step: 'findings', view: 'clean',
    live: null, runMode: 'simulated', active: -1, ...extra,
  });
  renderFindings();
  return elements.get('stepFindings').innerHTML;
}
function recognised(tid) {
  const csv = fs.readFileSync(path.join(samplesDir, `${tid}.csv`), 'utf8');
  const answer = JSON.parse(fs.readFileSync(path.join(samplesDir, 'answers', `${tid}_answer.json`), 'utf8'));
  const up = parseCSV(csv, `${tid}.csv`);
  up.matchedSample = { task_id: tid, file_name: `${tid}.csv` };
  up.matchedAnswer = answer;
  up.scoreIssues = buildSampleScoreIssues(up, answer);
  applySampleFixTags(up, up.scoreIssues);
  return { up, answer };
}

/* ---------------- recognised task_001: duplicates + bad price ---------------- */
{
  const { up } = recognised('task_001');
  const dupIssue = up.scoreIssues.find(i => i.kind === 'duplicate');
  const badIssue = up.scoreIssues.find(i => i.kind === 'bad_price');
  assert.ok(dupIssue && badIssue);
  const dupCopy = up.rows.find(r => r.n === dupIssue.rows[1]);
  const badRow = up.rows.find(r => r.n === badIssue.rows[0]);
  assert.equal(dupCopy.remove, dupIssue.id, 'later duplicate copy tagged with the score-issue id');
  assert.equal(badRow.remove, badIssue.id, 'bad-price row tagged with the score-issue id');

  const sc = score(up, up.findings);
  assert.ok(sc.matched.has(dupIssue.id), 'score counts the duplicate as caught');

  const html = showFindings(up);
  assert.ok(rowSegment(html, dupCopy.n).includes('>removed<'), 'After view removes the caught duplicate row');
  assert.ok(!rowSegment(html, dupIssue.rows[0]).includes('>removed<'), 'first copy is kept');
  assert.ok(rowSegment(html, badRow.n).includes('>removed<'), 'After view removes the caught bad-price row');
  assert.ok(html.includes('1 session missing · 2024-03-22'), 'After view shows the missing-day marker');
  const kept = Number(html.match(/After · (\d+) rows kept/)[1]);
  assert.equal(kept, up.rows.length - 2, 'kept count excludes both removed rows');

  /* A missed finding must apply no fix for that issue. */
  const missed = showFindings(up, { live: { findings: [] }, runMode: 'live' });
  assert.ok(!rowSegment(missed, dupCopy.n).includes('>removed<'), 'missed duplicate stays in the After view');
}

/* ---------------- recognised task_002: duplicate ---------------- */
{
  const { up } = recognised('task_002');
  const dupIssue = up.scoreIssues.find(i => i.kind === 'duplicate');
  const html = showFindings(up);
  assert.ok(rowSegment(html, dupIssue.rows[1]).includes('>removed<'), 'task_002 duplicate removed in After view');
}

/* ---------------- recognised task_003: split back-adjustment ---------------- */
{
  const { up } = recognised('task_003');
  const splitIssue = up.scoreIssues.find(i => i.kind === 'split');
  const pre = up.rows.find(r => r.n === splitIssue.rows[0] - 1);
  assert.equal(pre.cleanFor, splitIssue.id, 'pre-split row tagged for back-adjustment');
  assert.ok(Math.abs(pre.clean.c5 * 2 - parseFloat(pre.cells.c5)) < 0.02, 'pre-split close is back-adjusted to half');
  const html = showFindings(up);
  assert.ok(rowSegment(html, pre.n).includes('>fixed<'), 'After view shows the pre-split row as fixed');
}

/* ---------------- built-in seeded sample s1 is unchanged ---------------- */
{
  Object.assign(state, { sid: 's1' });
  const s = globalThis.__app.S[0];
  const html = showFindings(null);
  const dupCopy = s.rows.find(r => r.remove === 'dup');
  assert.ok(rowSegment(html, dupCopy.n).includes('>removed<'), 'built-in duplicate still removed');
  const splitFixed = s.rows.find(r => r.cleanFor === 'split');
  assert.ok(rowSegment(html, splitFixed.n).includes('>fixed<'), 'built-in split adjustment still shown');
}

/* ---------------- non-matched upload keeps heuristic After fixes ---------------- */
{
  const csv = 'date,ticker,open,high,low,close,volume\n'
    + '2024-01-02,TEST,1,1,1,1,10\n2024-01-03,TEST,1,1,1,1,10\n2024-01-03,TEST,1,1,1,1,10\n';
  const up = parseCSV(csv, 'my_own.csv');
  const html = showFindings(up);
  assert.ok(rowSegment(html, 3).includes('>removed<'), 'non-matched duplicate still removed');
  assert.equal(score(up, up.findings), null, 'non-matched upload still has no score');
}

console.log('[sample_after_view.test] all assertions passed');
