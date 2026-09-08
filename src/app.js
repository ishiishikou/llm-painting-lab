import { applyStroke } from "./brushes.js";

const canvas = document.querySelector("#painting");
const ctx = canvas.getContext("2d", { alpha: false });
const playButton = document.querySelector("#play");
const pauseButton = document.querySelector("#pause");
const stepButton = document.querySelector("#step");
const resetButton = document.querySelector("#reset");
const exportButton = document.querySelector("#export");
const speedInput = document.querySelector("#speed");
const speedValue = document.querySelector("#speedValue");
const strokeStatus = document.querySelector("#strokeStatus");
const phaseStatus = document.querySelector("#phaseStatus");
const runStatus = document.querySelector("#runStatus");

let documentData;
let strokes = [];
let cursor = 0;
let running = false;
let raf = null;

function paintBackground() {
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = documentData?.canvas?.background ?? "#111318";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function updateStatus() {
  strokeStatus.textContent = `${cursor.toLocaleString()} / ${strokes.length.toLocaleString()}`;
  const phaseId = cursor > 0 ? strokes[Math.min(cursor - 1, strokes.length - 1)]?.phase : null;
  const phase = documentData?.phases?.find((candidate) => candidate.id === phaseId);
  phaseStatus.textContent = phase?.label ?? phaseId ?? "—";
  runStatus.textContent = running ? "painting" : cursor >= strokes.length ? "complete" : "paused";
}

function renderStroke(index) {
  const stroke = strokes[index];
  if (!stroke) return;
  applyStroke(ctx, stroke);
}

function renderFrame() {
  if (!running) return;
  const perFrame = Number(speedInput.value);
  const target = Math.min(cursor + perFrame, strokes.length);

  while (cursor < target) {
    renderStroke(cursor);
    cursor += 1;
  }

  updateStatus();
  if (cursor >= strokes.length) {
    running = false;
    raf = null;
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
  paintBackground();
  updateStatus();
}

function step() {
  pause();
  if (cursor >= strokes.length) return;
  renderStroke(cursor);
  cursor += 1;
  updateStatus();
}

function exportPng() {
  const link = document.createElement("a");
  link.download = `${documentData?.metadata?.slug ?? "painting"}-${cursor}-strokes.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

async function init() {
  const response = await fetch("./strokes.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load strokes.json: ${response.status}`);
  documentData = await response.json();
  strokes = documentData.strokes ?? [];

  canvas.width = documentData.canvas?.width ?? 768;
  canvas.height = documentData.canvas?.height ?? 1024;
  paintBackground();
  updateStatus();
}

playButton.addEventListener("click", play);
pauseButton.addEventListener("click", pause);
stepButton.addEventListener("click", step);
resetButton.addEventListener("click", reset);
exportButton.addEventListener("click", exportPng);
speedInput.addEventListener("input", () => { speedValue.textContent = speedInput.value; });

init().catch((error) => {
  console.error(error);
  runStatus.textContent = "error";
  phaseStatus.textContent = error.message;
});
