import { applyStroke } from './brushes.js';

const canvas = document.querySelector('#painting');
const ctx = canvas.getContext('2d', { alpha: false });
const $ = (selector) => document.querySelector(selector);
const playButton = $('#play');
const pauseButton = $('#pause');
const stepButton = $('#step');
const resetButton = $('#reset');
const exportButton = $('#export');
const speedInput = $('#speed');
const speedValue = $('#speedValue');
const strokeStatus = $('#strokeStatus');
const phaseStatus = $('#phaseStatus');
const runStatus = $('#runStatus');

let documentData;
let strokes = [];
let cursor = 0;
let running = false;
let raf = null;

function paintBackground() {
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = documentData?.canvas?.background ?? '#111318';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function updateStatus() {
  strokeStatus.textContent = `${cursor.toLocaleString()} / ${strokes.length.toLocaleString()}`;
  const phaseId = cursor > 0 ? strokes[Math.min(cursor - 1, strokes.length - 1)]?.phase : null;
  const phase = documentData?.phases?.find((candidate) => candidate.id === phaseId);
  phaseStatus.textContent = phase?.label ?? phaseId ?? '—';
  runStatus.textContent = running ? '描画中' : cursor >= strokes.length ? '完了' : '停止中';
}

function renderStroke(index) {
  const stroke = strokes[index];
  if (stroke) applyStroke(ctx, stroke);
}

function renderFrame() {
  if (!running) return;
  const perFrame = Number(speedInput.value);
  const target = Math.min(cursor + perFrame, strokes.length);
  while (cursor < target) renderStroke(cursor++);
  updateStatus();
  if (cursor >= strokes.length) {
    running = false;
    raf = null;
    window.__paintingReady = true;
    updateStatus();
    return;
  }
  raf = requestAnimationFrame(renderFrame);
}

function play() {
  if (running || cursor >= strokes.length) return;
  running = true;
  updateStatus();
  raf = requestAnimationFrame(renderFrame);
}

function pause() {
  running = false;
  if (raf) cancelAnimationFrame(raf);
  raf = null;
  updateStatus();
}

function reset() {
  pause();
  cursor = 0;
  window.__paintingReady = false;
  paintBackground();
  updateStatus();
}

function step() {
  pause();
  if (cursor >= strokes.length) return;
  renderStroke(cursor++);
  updateStatus();
}

function renderAll() {
  pause();
  while (cursor < strokes.length) renderStroke(cursor++);
  window.__paintingReady = true;
  updateStatus();
}

function exportPng() {
  const link = document.createElement('a');
  link.download = `${documentData?.metadata?.slug ?? 'painting'}-${cursor}-strokes.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}

async function loadDocument() {
  const params = new URLSearchParams(location.search);
  const filename = params.get('mode') === 'generated' ? 'strokes.generated.json' : 'strokes.json';
  const response = await fetch(`./${filename}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${filename} の読み込みに失敗しました: ${response.status}`);
  return response.json();
}

async function init() {
  documentData = await loadDocument();
  strokes = documentData.strokes ?? [];
  canvas.width = documentData.canvas?.width ?? 768;
  canvas.height = documentData.canvas?.height ?? 1024;
  paintBackground();
  updateStatus();
  window.__paintingDocument = documentData;
  window.__paintingReady = false;
  if (new URLSearchParams(location.search).has('instant')) renderAll();
}

playButton.addEventListener('click', play);
pauseButton.addEventListener('click', pause);
stepButton.addEventListener('click', step);
resetButton.addEventListener('click', reset);
exportButton.addEventListener('click', exportPng);
speedInput.addEventListener('input', () => { speedValue.textContent = speedInput.value; });

init().catch((error) => {
  console.error(error);
  runStatus.textContent = 'エラー';
  phaseStatus.textContent = error.message;
});
