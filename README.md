# pngtrace

Streamlit‑first tool to turn a raster image into clean black/white vector output (centerline strokes or filled regions) with an optional Gemini AI pre‑processing step that normalizes contrast before tracing.

## Key Features
* Streamlit UI (primary workflow) – upload or pick an image, run AI enhancement, then vectorize.
* Two vector modes:
  * Centerline: skeleton + smoothing + simplification → stroke paths.
  * Fill: extracts solid regions + holes → single even‑odd filled path (great for engrave / cut fills).
* Gemini AI step (required in the UI flow) produces a high‑contrast black/white version; falls back to original if no image is returned.
* Pure‑Python tracing core (no system binaries). Optional extras: OpenCV (better contour hierarchy), CairoSVG (nicer SVG preview) – both degrade gracefully if missing.
* Per‑run artifact folder under `outputs/` (original, AI raster, SVG, optional debug composite).

## Get an API Key
Obtain a Google Gemini API key here: https://aistudio.google.com/apikey

Set it in a `.env` file as:
```
GEMINI_API_KEY=your_key_here
```

## Quick Start (Streamlit)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=YOUR_KEY" > .env
streamlit run streamlit_app.py
```

### Using the App
1. Select or upload an image (placed in `inputs/`).
2. (Auto‑required) Run the AI step – you can tweak the prompt.
3. Choose Mode: centerline or fill.
4. Adjust simplification / stroke width (stroke width disabled in fill mode).
5. Run vectorization → preview + download SVG.
6. Re‑run AI or vector steps independently; previous results persist in session state.

## Optional CLI Usage
Plain centerline trace (bypasses AI):
```bash
python pngtrace.py inputs/your_image.png output.svg --stroke-width 0.2mm --rdp 0.8
```

Full AI → trace pipeline (creates timestamped run dir under `outputs/`):
```bash
python pipeline.py inputs/your_image.png output.svg --debug
```
`--debug` also emits `debug_triptych.png` (original | AI | final SVG raster).

## Outputs
`outputs/<timestamp>/` contains:
* `original.<ext>` – input copy
* `ai_result.png` – Gemini output (if any)
* `<name>.svg` – final vector (centerline strokes or even‑odd filled path)
* `debug_triptych.png` – only when `--debug` (CLI) is used

## Tips & Troubleshooting
* Missing AI image: ensure `GEMINI_API_KEY` is set and valid.
* Too many short segments: lower `--rdp` or UI simplification slider (value closer to 0).
* Over‑simplified / jagged: raise `--rdp`.
* Preview missing (SVG rasterization): install `cairosvg` or rely on fallback sampler.
* Fill mode uses even‑odd rule; small speckles can be pruned by pre‑cleaning your raster (e.g. blur + threshold) before upload.

## Experimental
`pngtrace_v2.py` contains a more aggressive adaptive pipeline; not yet integrated into the main UI.

## License
MIT
