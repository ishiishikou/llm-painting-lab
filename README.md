# llm-painting-lab

A small experimental environment for testing how far an LLM can improve visual output through better tools, persistent state, and iterative feedback.

The goal is **not** a fair benchmark between models. The goal is to see how much a weaker or cheaper model can close the gap to a stronger model when the environment does more of the scaffolding.

## Current MVP

The browser app replays an ordered list of logical brush strokes from `strokes.json` onto an HTML Canvas.

Supported logical brushes:

- `line`
- `curve`
- `dab`
- `softDab`
- `dryBrush`

The dry brush is deterministic when given a seed, so the same stroke document reproduces the same output.

## Guardrails

This first experiment deliberately forbids raster shortcuts:

- no source PNG/JPEG is loaded into the canvas
- no image URL is allowed in the stroke document
- no texture bitmap is allowed in the stroke document
- output is reconstructed only from ordered stroke data

`npm run validate` checks the document structure and rejects common raster/image reference keys.

## Run locally

Requires Node.js 20+.

```bash
npm run validate
npm run serve
```

Open:

```text
http://127.0.0.1:8000
```

The UI supports play, pause, one-stroke stepping, reset, speed control, and PNG export.

## Repository structure

```text
.
├── index.html
├── styles.css
├── strokes.json
├── src/
│   ├── app.js
│   └── brushes.js
├── tools/
│   ├── serve.mjs
│   └── validate-strokes.mjs
└── .github/workflows/validate.yml
```

## Intended iteration loop

The next stage is not simply to increase the stroke count. It is to make the model operate in an explicit feedback loop:

```text
Plan
  -> generate/update strokes
  -> render
  -> inspect the rendered result
  -> critique the largest visible defect
  -> add corrective strokes
  -> render again
```

Later stages can add checkpoints, region-level scoring, separate painter/critic roles, persistent evaluation state, and automated repair passes.

## Experimental principle

The environment should carry as much mechanical burden as possible so the model can spend inference on high-level visual decisions rather than raw pixel manipulation.
