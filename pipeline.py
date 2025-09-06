#!/usr/bin/env python3
"""
AI-assisted centerline vectorization pipeline.

Workflow (simplified):
 1. Input raster image.
 2. Send to Google Gemini image model with a prompt requesting a HIGH CONTRAST black-on-white
     thick centerline style version.
 3. Use the returned PNG (first image part) as input to the classical v1 centerline pipeline.

No SVG handling is performed (Gemini web currently indicates SVG output isn't available). If
future models emit SVG, you can extend this again.

NOTE: This script depends on an environment variable:
  GEMINI_API_KEY  (Google Generative AI key)

Usage:
  pipeline.py input.png out.svg

If the model already returns an <svg> with acceptable centerlines, you could bypass the
traditional skeletonization and just emit it. For now we feed the model's result into v1
for consistency / further smoothing.
"""
import os, sys, io, argparse, tempfile, re, base64, json
from pathlib import Path
from typing import Optional
from datetime import datetime
import shutil

import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import re
import math

# Import v1 main logic by executing module functions directly.
import pngtrace as v1

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

MODEL_CANDIDATES = [
    "gemini-2.5-flash-image-preview"
]
PROMPT = (
    "I need a centerline trace of this, so it can later be easily be converted to an svg. Only black and white may be used with an even-odd loop so svg tracing is easier. fills may be used if rerpesenting a black or white area from input image."  # default AI prompt
)


def call_gemini(image_path: str, debug: bool = False, custom_prompt: Optional[str] = None):
    """Call Gemini; return list of image bytes (PNG/JPEG).
    Args:
        image_path: Input raster path.
        debug: Verbose logging flag.
        custom_prompt: Optional override prompt (falls back to PROMPT constant).
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY environment variable not set (or .env missing)")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # idempotent
    out_bytes = []
    last_error = None
    # Load and encode once
    with Image.open(image_path) as im:
        im = im.convert("RGBA")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    for model_name in MODEL_CANDIDATES:
        try:
            if debug:
                print(f"[DEBUG] Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            prompt_text = custom_prompt if custom_prompt else PROMPT
            response = model.generate_content([
                prompt_text,
                {"mime_type": "image/png", "data": image_bytes}
            ])
            if debug:
                dump = getattr(response, 'to_dict', lambda: None)()
                if dump:
                    print(f"[DEBUG] response keys ({model_name}):", list(dump.keys()))
            if not response or not getattr(response, 'candidates', None):
                continue
            for ci, cand in enumerate(response.candidates):
                content = getattr(cand, 'content', None)
                parts = getattr(content, 'parts', []) if content else []
                if debug:
                    print(f"[DEBUG] {model_name} candidate {ci} parts={len(parts)}")
                for pi, part in enumerate(parts):
                    inline = getattr(part, 'inline_data', None)
                    if inline and getattr(inline, 'data', None):
                        data = inline.data
                        out_bytes.append(data)
                        if debug:
                            print(f"[DEBUG] {model_name} image part ci={ci} pi={pi} size={len(data)} bytes")
                    elif getattr(part, 'text', None) and debug:
                        print(f"[DEBUG] {model_name} text part ci={ci} pi={pi}: {part.text[:160]}{'...' if len(part.text)>160 else ''}")
            if out_bytes:
                break
        except Exception as e:
            last_error = e
            if debug:
                print(f"[DEBUG] model {model_name} error: {e}")
    if not out_bytes:
        if debug and last_error:
            print(f"[DEBUG] Last model error: {last_error}")
    return out_bytes


def save_image_bytes(image_bytes: bytes) -> str:
    tmp_png = tempfile.mktemp(suffix='_ai.png')
    with open(tmp_png, 'wb') as f:
        f.write(image_bytes)
    return tmp_png


def run_pipeline(input_image: str, output_svg: str, debug: bool=False, outputs_dir: str = "outputs", outline_padding: float = 0.0):
    # Prepare run directory
    ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
    run_dir = Path(outputs_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    # Copy original input
    input_path = Path(input_image)
    input_copy = run_dir / f"original{input_path.suffix.lower()}"
    try:
        shutil.copy2(input_path, input_copy)
    except Exception as e:
        print(f"[WARN] Failed to copy original input: {e}")

    print("[AI] Requesting high-contrast stroke image from Gemini...")
    images = call_gemini(input_image, debug=debug)
    if images:
        ai_img_path = run_dir / "ai_result.png"
        with open(ai_img_path, 'wb') as f:
            f.write(images[0])
        print(f"[AI] Received AI-enhanced raster: {ai_img_path}")
        tmp_png = str(ai_img_path)
    else:
        print("[AI] No AI image returned. Falling back to original input image.")
        tmp_png = str(input_copy)

    # Final SVG path inside run directory
    final_svg_path = run_dir / Path(output_svg).name

    prev_input = v1.INPUT_FILE
    prev_output = v1.OUTPUT_FILE
    prev_outline = getattr(v1, 'OUTLINE_PADDING', 0.0)
    try:
        v1.INPUT_FILE = tmp_png
        v1.OUTPUT_FILE = str(final_svg_path)
        try:
            v1.OUTLINE_PADDING = float(outline_padding or 0.0)
        except Exception:
            pass
        old_argv = sys.argv
        sys.argv = [old_argv[0]]
        try:
            v1.main()
        finally:
            sys.argv = old_argv
    finally:
        v1.INPUT_FILE = prev_input
        v1.OUTPUT_FILE = prev_output
        try:
            v1.OUTLINE_PADDING = prev_outline
        except Exception:
            pass
    print(f"[PIPELINE] Completed. Run dir: {run_dir}\n[PIPELINE] Final SVG: {final_svg_path}")
    # Debug composite (original + AI + final vector rasterization)
    if debug:
        try:
            trip_path = run_dir / "debug_triptych.png"
            panels = []
            labels = []
            # Original
            try:
                panels.append(Image.open(input_copy).convert("RGB"))
                labels.append("original")
            except Exception:
                pass
            # AI result (may be same as original fallback)
            ai_img_file = run_dir / "ai_result.png"
            if ai_img_file.exists():
                try:
                    panels.append(Image.open(ai_img_file).convert("RGB"))
                    labels.append("ai_result")
                except Exception:
                    pass
            # Rasterize SVG if cairosvg available else simple fallback
            svg_raster = None
            try:
                import cairosvg  # type: ignore
                svg_png_bytes = cairosvg.svg2png(url=str(final_svg_path))
                _tmp = Image.open(io.BytesIO(svg_png_bytes))
                if _tmp.mode in ("RGBA","LA"):
                    bg = Image.new("RGBA", _tmp.size, (255,255,255,255))
                    bg.paste(_tmp, mask=_tmp.split()[-1])
                    _tmp = bg
                svg_raster = _tmp.convert("RGB")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] CairoSVG unavailable or failed ({e}); using simple fallback rasterizer.")
                # Fallback minimalist rasterizer (handles M/m L/l H/h V/v Q/q Z)
                try:
                    svg_text = final_svg_path.read_text()
                    # Extract width/height or viewBox
                    viewbox_match = re.search(r'viewBox="([0-9.+\-eE ]+)"', svg_text)
                    if viewbox_match:
                        nums = [float(x) for x in viewbox_match.group(1).split()]
                        _, _, vb_w, vb_h = nums if len(nums)==4 else (0,0,1024,1024)
                    else:
                        w_match = re.search(r'width="([0-9.]+)"', svg_text)
                        h_match = re.search(r'height="([0-9.]+)"', svg_text)
                        vb_w = float(w_match.group(1)) if w_match else 1024
                        vb_h = float(h_match.group(1)) if h_match else 1024
                    paths = re.findall(r'<path[^>]* d="([^"]+)"', svg_text)
                    polylines = []
                    for d in paths:
                        tokens = re.findall(r'[A-Za-z]|[-+]?[0-9]*\.?[0-9]+', d)
                        i=0
                        cx=cy=0
                        start=None
                        current=[]
                        cmd=None
                        while i < len(tokens):
                            t = tokens[i]
                            if re.match(r'[A-Za-z]', t):
                                cmd=t
                                i+=1
                                continue
                            if cmd in ('M','L'):
                                x=float(t); y=float(tokens[i+1]); i+=2
                                cx,cy=x,y
                                if cmd=='M':
                                    if current:
                                        polylines.append(current); current=[]
                                    current.append((cx,cy))
                                    start=(cx,cy)
                                else:
                                    current.append((cx,cy))
                            elif cmd in ('m','l'):
                                x=float(t); y=float(tokens[i+1]); i+=2
                                if cmd=='m' and not current:
                                    cx+=x; cy+=y
                                    current.append((cx,cy)); start=(cx,cy)
                                else:
                                    cx+=x; cy+=y
                                    current.append((cx,cy))
                            elif cmd in ('H'):
                                x=float(t); i+=1; cx=x; current.append((cx,cy))
                            elif cmd in ('h'):
                                x=float(t); i+=1; cx+=x; current.append((cx,cy))
                            elif cmd in ('V'):
                                y=float(t); i+=1; cy=y; current.append((cx,cy))
                            elif cmd in ('v'):
                                y=float(t); i+=1; cy+=y; current.append((cx,cy))
                            elif cmd in ('Q','q'):
                                # Quadratic bezier: control (x1,y1) endpoint (x2,y2)
                                x1=float(t); y1=float(tokens[i+1]); x2=float(tokens[i+2]); y2=float(tokens[i+3]); i+=4
                                if cmd=='q':
                                    x1+=cx; y1+=cy; x2+=cx; y2+=cy
                                # Sample curve
                                x0,y0 = cx,cy
                                seg_len = math.hypot(x2-x0, y2-y0)
                                steps = max(6, min(64, int(seg_len/2)+1))
                                for s in range(1,steps+1):
                                    tpar = s/steps
                                    # Quadratic Bezier formula
                                    xa = (1-tpar)*x0 + tpar*x1
                                    ya = (1-tpar)*y0 + tpar*y1
                                    xb = (1-tpar)*x1 + tpar*x2
                                    yb = (1-tpar)*y1 + tpar*y2
                                    xq = (1-tpar)*xa + tpar*xb
                                    yq = (1-tpar)*ya + tpar*yb
                                    current.append((xq,yq))
                                cx,cy=x2,y2
                            elif cmd in ('Z','z'):
                                if start and current and current[-1]!=start:
                                    current.append(start)
                                i+=1
                            else:
                                # Skip unsupported command numbers (curves) by consuming appropriate count heuristically
                                i+=1
                        if current:
                            polylines.append(current)
                    # Render
                    scale = 1024 / max(vb_w, vb_h)
                    out_w = int(vb_w*scale)
                    out_h = int(vb_h*scale)
                    img = Image.new('RGB',(out_w,out_h),'white')
                    dr = ImageDraw.Draw(img)
                    for pl in polylines:
                        if len(pl)>=2:
                            pts=[(x*scale,y*scale) for x,y in pl]
                            dr.line(pts, fill='black', width=1)
                    svg_raster = img
                except Exception as fe:
                    if debug:
                        print(f"[DEBUG] Fallback rasterizer failed: {fe}")
            if svg_raster is not None:
                panels.append(svg_raster)
                labels.append("final_svg")
            if len(panels) >= 2:
                # Normalize heights
                max_h = max(im.height for im in panels)
                norm_panels = []
                for im in panels:
                    if im.height != max_h:
                        new_w = int(im.width * (max_h / im.height))
                        im = im.resize((new_w, max_h), Image.Resampling.LANCZOS)
                    norm_panels.append(im)
                total_w = sum(im.width for im in norm_panels) + 10 * (len(norm_panels)-1)
                composite = Image.new("RGB", (total_w, max_h + 24), color=(255,255,255))
                draw = ImageDraw.Draw(composite)
                x = 0
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                for im,label in zip(norm_panels, labels):
                    composite.paste(im, (x,0))
                    # Label bar area
                    if font:
                        draw.rectangle([x, max_h, x+im.width, max_h+24], fill=(255,255,255))
                        draw.text((x+4, max_h+4), label, fill=(0,0,0), font=font)
                    x += im.width + 10
                composite.save(trip_path)
                print(f"[DEBUG] Wrote composite debug image: {trip_path}")
            else:
                print("[DEBUG] Not enough panels to build triptych")
        except Exception as e:
            print(f"[DEBUG] Failed to build debug triptych: {e}")


def parse_args():
    ap = argparse.ArgumentParser(description="AI + classical centerline vectorization pipeline")
    ap.add_argument("input", help="Input raster image")
    ap.add_argument("output", nargs='?', default="pipeline_output.svg", help="Output SVG file")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    ap.add_argument("--outputs-dir", default="outputs", help="Base directory for per-run output folders")
    ap.add_argument("--outline-padding", type=float, default=0.0, help="External outline cut padding in output pixels (0 disables)")
    return ap.parse_args()


def main():
    a = parse_args()
    run_pipeline(a.input, a.output, debug=a.debug, outputs_dir=a.outputs_dir, outline_padding=a.outline_padding)

if __name__ == "__main__":
    main()
