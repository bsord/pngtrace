# pngtrace

Lightweight pipeline to turn a raster image into centerline SVG paths, optionally with an AI pre-processing step to produce clean black/white line art before vectorization.

## Features
- Classical centerline tracer (`pngtrace.py`) using skeletonization + smoothing + corner preservation.
- Advanced experimental version (`pngtrace_v2.py`) with adaptive thresholding, sub‑pixel refinement, spline smoothing, curvature‑aware simplification.
- AI pipeline (`pipeline.py`) that:
  1. Loads an input raster from `inputs/` (or any path).
  2. Calls a Gemini model to produce a high‑contrast stroke image.
  3. Falls back gracefully to the original if AI output is absent.
  4. Runs the classical tracer to produce an SVG.
  5. In `--debug` mode writes a composite triptych PNG (original | AI result | final SVG raster).
- Per‑run timestamped output directories under `outputs/` capturing all artifacts.
- Environment variable loading via `.env` (place your `GOOGLE_API_KEY` there).

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Set up API key
cp .env.example .env  # create and edit with your key (you create .env.example)
```
Add to `.env`:
```
GOOGLE_API_KEY=your_key_here
```

### Basic centerline trace (no AI)
```bash
python pngtrace.py inputs/your_image.png output.svg --stroke-width 0.2mm --rdp 0.8
```

### Run AI-assisted pipeline
```bash
python pipeline.py inputs/your_image.png output.svg --debug
```
The `--debug` flag adds verbose model info and a `debug_triptych.png` composite.

## Outputs
Each pipeline run creates: `outputs/YYYYMMDD-HHMMSS-RANDOM/` containing:
- `original.<ext>` – original input copy
- `ai_result.png` – AI produced raster (if available)
- `<your_output>.svg` – final centerline SVG
- `debug_triptych.png` – (debug only) side-by-side visualization

## Configuration Notes
- Stroke width is stored as provided (supports units like `px`, `mm`, `in`).
- If you need a white background rectangle in the SVG (for viewing only), use `--background #ffffff`; by default no filled rectangle is added to avoid CAM engraving fills.
- `pngtrace_v2.py` is optional and not yet wired into the AI pipeline—use directly while experimenting.

## Troubleshooting
- Black third panel in triptych: install Cairo system libs (`sudo apt-get install -y libcairo2 libcairo2-dev`) so CairoSVG can render accurately.
- Empty AI panel: ensure `GOOGLE_API_KEY` is set and network access works; pipeline falls back to original if AI image not returned.
- Too many small segments: decrease `--rdp` (e.g. 0.5). Too much simplification: increase it (e.g. 1.2).

## Roadmap / Ideas
- Integrate v2 pipeline into AI stage.
- Post-processing to auto-detect and hollow large filled shapes.
- CLI argument to override the AI prompt.
- Export metadata JSON (model, path count, timings).

## License
MIT (add a `LICENSE` file if desired).
