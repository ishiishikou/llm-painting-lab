import fs from "node:fs";

const file = new URL("../strokes.json", import.meta.url);
const data = JSON.parse(fs.readFileSync(file, "utf8"));
const errors = [];
const allowedBrushes = new Set([
  "line", "curve", "dab", "softDab", "dryBrush",
  "variableBrush", "mixerBrush", "smudgeBrush", "glaze"
]);
const bannedKeyPattern = /^(src|href|image|imageUrl|bitmap|texture|textureUrl|dataUrl)$/i;

function walk(value, path = "root") {
  if (Array.isArray(value)) {
    value.forEach((item, index) => walk(item, `${path}[${index}]`));
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    if (bannedKeyPattern.test(key)) errors.push(`${path}.${key}: raster/image references are not allowed`);
    walk(child, `${path}.${key}`);
  }
}

if (data.version !== 1) errors.push("version must be 1");
if (!Number.isFinite(data.canvas?.width) || !Number.isFinite(data.canvas?.height)) {
  errors.push("canvas.width and canvas.height must be numbers");
}
if (!Array.isArray(data.phases) || data.phases.length === 0) errors.push("phases must be a non-empty array");
if (!Array.isArray(data.strokes)) errors.push("strokes must be an array");

const phaseIds = new Set((data.phases ?? []).map((phase) => phase.id));
const ids = new Set();
for (const [index, stroke] of (data.strokes ?? []).entries()) {
  const prefix = `strokes[${index}]`;
  if (!Number.isFinite(stroke.id)) errors.push(`${prefix}.id must be numeric`);
  if (ids.has(stroke.id)) errors.push(`${prefix}.id duplicates ${stroke.id}`);
  ids.add(stroke.id);
  if (!phaseIds.has(stroke.phase)) errors.push(`${prefix}.phase is unknown: ${stroke.phase}`);
  if (!allowedBrushes.has(stroke.brush)) errors.push(`${prefix}.brush is unsupported: ${stroke.brush}`);
  if (typeof stroke.color !== "string") errors.push(`${prefix}.color must be a string`);
  if (stroke.opacity != null && (stroke.opacity < 0 || stroke.opacity > 1)) errors.push(`${prefix}.opacity must be between 0 and 1`);
}

walk(data);

if (errors.length) {
  console.error(`Stroke document validation failed with ${errors.length} error(s):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`Validated ${data.strokes.length} strokes across ${data.phases.length} phases.`);
console.log("Raster/image reference guard: passed.");
