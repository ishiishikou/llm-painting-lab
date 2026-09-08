function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function setCommon(ctx, stroke) {
  ctx.globalAlpha = stroke.opacity ?? 1;
  ctx.globalCompositeOperation = stroke.blendMode ?? "source-over";
  ctx.strokeStyle = stroke.color ?? "#ffffff";
  ctx.fillStyle = stroke.color ?? "#ffffff";
  ctx.lineWidth = stroke.width ?? 1;
  ctx.lineCap = stroke.lineCap ?? "round";
  ctx.lineJoin = stroke.lineJoin ?? "round";
}

function drawLine(ctx, s) {
  ctx.beginPath();
  ctx.moveTo(s.x1, s.y1);
  ctx.lineTo(s.x2, s.y2);
  ctx.stroke();
}

function drawCurve(ctx, s) {
  ctx.beginPath();
  ctx.moveTo(s.x1, s.y1);
  if (s.cx2 != null && s.cy2 != null) {
    ctx.bezierCurveTo(s.cx1, s.cy1, s.cx2, s.cy2, s.x2, s.y2);
  } else {
    ctx.quadraticCurveTo(s.cx1, s.cy1, s.x2, s.y2);
  }
  ctx.stroke();
}

function drawDab(ctx, s) {
  ctx.beginPath();
  ctx.arc(s.x, s.y, s.radius ?? Math.max(1, (s.width ?? 1) / 2), 0, Math.PI * 2);
  ctx.fill();
}

function drawSoftDab(ctx, s) {
  const radius = s.radius ?? 20;
  const gradient = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, radius);
  gradient.addColorStop(0, s.color ?? "#ffffff");
  gradient.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawDryBrush(ctx, s) {
  const rand = mulberry32(s.seed ?? s.id ?? 1);
  const strands = Math.max(4, s.strands ?? 18);
  const dx = s.x2 - s.x1;
  const dy = s.y2 - s.y1;
  const length = Math.hypot(dx, dy) || 1;
  const nx = -dy / length;
  const ny = dx / length;
  const spread = s.spread ?? (s.width ?? 12);

  ctx.lineWidth = Math.max(0.5, (s.width ?? 8) / strands * 0.7);
  for (let i = 0; i < strands; i++) {
    const offset = (rand() - 0.5) * spread;
    const jitter1 = (rand() - 0.5) * 2;
    const jitter2 = (rand() - 0.5) * 2;
    ctx.globalAlpha = (s.opacity ?? 1) * (0.25 + rand() * 0.7);
    ctx.beginPath();
    ctx.moveTo(s.x1 + nx * offset + jitter1, s.y1 + ny * offset + jitter1);
    ctx.lineTo(s.x2 + nx * offset + jitter2, s.y2 + ny * offset + jitter2);
    ctx.stroke();
  }
}

export function applyStroke(ctx, stroke) {
  ctx.save();
  setCommon(ctx, stroke);

  switch (stroke.brush ?? stroke.type ?? "line") {
    case "line": drawLine(ctx, stroke); break;
    case "curve": drawCurve(ctx, stroke); break;
    case "dab": drawDab(ctx, stroke); break;
    case "softDab": drawSoftDab(ctx, stroke); break;
    case "dryBrush": drawDryBrush(ctx, stroke); break;
    default: throw new Error(`Unknown brush: ${stroke.brush ?? stroke.type}`);
  }

  ctx.restore();
}
