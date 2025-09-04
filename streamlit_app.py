#!/usr/bin/env python3
"""
Streamlit UI for pngtrace.

Features:
 - Select existing image from inputs/ OR upload a new one.
 - Optional AI enhancement via Gemini (centerline-friendly high contrast image).
 - Run classical tracer (v1) to produce SVG.
 - Preview rasterized SVG (pure Python path-only renderer; no system libs).
 - Download SVG.

Environment:
 - Requires GEMINI_API_KEY in .env for AI enhancement; if absent, AI step can be skipped.
"""
import os, re, math, tempfile, io, time
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw
import pngtrace as v1

# Helper functions first (avoid forward reference issues)

def _rasterize_paths(svg_text: str, size: int = 1024):
    """Minimal pure-Python path sampler for preview (M L H V Q Z)."""
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
        i=0; cx=cy=0; start=None; current=[]; cmd=None
        while i < len(tokens):
            t = tokens[i]
            if re.match(r'[A-Za-z]', t):
                cmd=t; i+=1; continue
            if cmd in ('M','L'):
                x=float(t); y=float(tokens[i+1]); i+=2; cx,cy=x,y
                if cmd=='M':
                    if current: polylines.append(current); current=[]
                    current.append((cx,cy)); start=(cx,cy)
                else:
                    current.append((cx,cy))
            elif cmd in ('m','l'):
                x=float(t); y=float(tokens[i+1]); i+=2; cx+=x; cy+=y
                if cmd=='m' and not current:
                    current.append((cx,cy)); start=(cx,cy)
                else:
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
                x1=float(t); y1=float(tokens[i+1]); x2=float(tokens[i+2]); y2=float(tokens[i+3]); i+=4
                if cmd=='q': x1+=cx; y1+=cy; x2+=cx; y2+=cy
                x0,y0=cx,cy; seg_len=math.hypot(x2-x0,y2-y0); steps=max(6,min(64,int(seg_len/2)+1))
                for s in range(1,steps+1):
                    tpar=s/steps
                    xa=(1-tpar)*x0 + tpar*x1; ya=(1-tpar)*y0 + tpar*y1
                    xb=(1-tpar)*x1 + tpar*x2; yb=(1-tpar)*y1 + tpar*y2
                    xq=(1-tpar)*xa + tpar*xb; yq=(1-tpar)*ya + tpar*yb
                    current.append((xq,yq))
                cx,cy=x2,y2
            elif cmd in ('Z','z'):
                if start and current and current[-1]!=start: current.append(start)
                i+=1
            else:
                i+=1
        if current: polylines.append(current)
    scale = size / max(vb_w, vb_h)
    out_w = int(vb_w*scale); out_h=int(vb_h*scale)
    img = Image.new('RGB',(out_w,out_h),'white')
    dr = ImageDraw.Draw(img)
    for pl in polylines:
        if len(pl)>=2:
            pts=[(x*scale,y*scale) for x,y in pl]
            dr.line(pts, fill='black', width=1)
    return img

def _svg_to_png_bytes(svg_text: str):
    try:
        import cairosvg  # type: ignore
        return cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
    except Exception:
        raise

# Optional AI pipeline import (only used if user enables AI)
try:
    import pipeline  # for call_gemini
except Exception:
    pipeline = None

INPUTS_DIR = Path('inputs')
INPUTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="pngtrace", layout="wide")
st.title("pngtrace – Centerline Vectorizer")
st.caption("Left-to-right stages: 1) Source  2) AI  3) Vectorize  4) Result")

def init_state():
    if 'workspace_dir' not in st.session_state:
        st.session_state.workspace_dir = tempfile.TemporaryDirectory()
    for k,v in [
        ('original_path', None),
        ('ai_path', None),
        ('svg_text', None),
        ('vector_time', None),
        ('ai_time', None),
        ('path_count', None),
    ]:
        st.session_state.setdefault(k, v)

init_state()

top_left, top_mid, top_right = st.columns([1,2,1])
with top_left:
    if st.button("Reset Session"):
        for k in ['original_path','ai_path','svg_text','vector_time','ai_time','path_count','ai_prompt']:
            st.session_state[k] = None
        st.experimental_rerun()
with top_mid:
    st.markdown("**Workflow:** Source → AI → Vectorize → Result")
with top_right:
    if os.environ.get('GEMINI_API_KEY'):
        st.success("GEMINI_API_KEY ✔", icon="🔑")
    else:
        st.warning("No GEMINI_API_KEY", icon="⚠️")

col1, col2, col3, col4 = st.columns(4, gap="large")

# ---- Column 1: Source Image ----
with col1:
    st.markdown("### 1. Source")
    existing_files = sorted([p.name for p in INPUTS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'}])
    choice = st.selectbox("Select from inputs/", options=["(none)"] + existing_files, key='file_choice')
    uploaded = st.file_uploader("Or drag/drop", type=["png","jpg","jpeg"], key='uploader')
    src_image = None; src_name=None
    if uploaded is not None:
        src_image = Image.open(uploaded)
        src_name = uploaded.name
    elif choice != '(none)':
        try:
            src_image = Image.open(INPUTS_DIR/choice)
            src_name = choice
        except Exception as e:
            st.error(f"Load failed: {e}")
    if src_image is not None:
        if st.session_state.original_path is None or (src_name and st.session_state.original_path and not st.session_state.original_path.endswith(src_name)):
            for k in ['ai_path','svg_text','vector_time','ai_time','path_count']:
                st.session_state[k] = None
        work_dir = Path(st.session_state.workspace_dir.name)
        input_path = work_dir / (src_name or 'input.png')
        try:
            if input_path.suffix.lower() in {'.jpg','.jpeg'} and src_image.mode == 'RGBA':
                src_image.convert('RGB').save(input_path, quality=95)
            else:
                if input_path.suffix.lower()=='.png' and src_image.mode not in ('RGB','RGBA'):
                    src_image = src_image.convert('RGBA')
                src_image.save(input_path)
        except OSError:
            src_image.convert('RGB').save(input_path)
        st.session_state.original_path = str(input_path)
        # Preload cached AI image if exists: <basename>_ai.(png|jpg|jpeg) in inputs/
        if st.session_state.ai_path is None and src_name:
            base = Path(src_name).stem
            candidates = [INPUTS_DIR / f"{base}_ai{ext}" for ext in ('.png', '.jpg', '.jpeg')]
            for c in candidates:
                if c.exists():
                    st.session_state.ai_path = str(c)
                    st.session_state.ai_time = 0.0
                    st.info(f"Preloaded cached AI image: {c.name}")
                    break
        st.image(src_image, caption="Original", width='stretch')
    else:
        st.info("Pick or upload an image.")

# Guard: need a source image for later columns
if st.session_state.original_path is None:
    st.stop()

# ---- Column 2: AI Enhancement ----
with col2:
    st.markdown("### 2. AI Enhance")
    default_prompt = ("I need a centerline trace of this, so it can later be easily be converted to an svg. Only black and white may be used with an even-odd loop so svg tracing is easier")
    if 'ai_prompt' not in st.session_state or st.session_state.ai_prompt is None:
        st.session_state.ai_prompt = default_prompt
    st.session_state.ai_prompt = st.text_area("Prompt", value=st.session_state.ai_prompt, height=140, help="Edit prompt then re-run AI.")
    if pipeline is None:
        st.warning("pipeline.py not importable; cannot proceed.")
    elif os.environ.get('GEMINI_API_KEY') is None:
        st.warning("Set GEMINI_API_KEY to run AI (flow blocked until set).")
    else:
        run_ai = st.button("Run / Re-run AI", key="run_ai_btn")
        clear_ai = st.button("Clear AI Result", key="clear_ai_btn")
        if clear_ai:
            st.session_state.ai_path = None
            st.session_state.ai_time = None
        if run_ai:
            t0=time.time()
            try:
                imgs = pipeline.call_gemini(st.session_state.original_path, debug=False, custom_prompt=st.session_state.ai_prompt) if hasattr(pipeline, 'call_gemini') else []
                if imgs:
                    work_dir = Path(st.session_state.workspace_dir.name)
                    ai_path = work_dir/'ai_result.png'
                    ai_path.write_bytes(imgs[0])
                    st.session_state.ai_path = str(ai_path)
                    st.session_state.ai_time = time.time()-t0
                    st.success(f"AI completed in {st.session_state.ai_time:.2f}s (re-run OK)")
                else:
                    # Treat AI step as executed but fallback to original
                    st.session_state.ai_path = st.session_state.original_path
                    st.info("Model returned no image; using original as AI fallback.")
            except Exception as e:
                st.error(f"AI failed: {e}")
    if st.session_state.ai_path:
        ai_path_obj = Path(st.session_state.ai_path)
        if ai_path_obj.name == 'ai_result.png':
            label = "AI Enhanced"
        elif ai_path_obj.name.endswith('_ai.png') or ai_path_obj.name.endswith('_ai.jpg') or ai_path_obj.name.endswith('_ai.jpeg'):
            label = "Preloaded AI"
        elif st.session_state.ai_path == st.session_state.original_path:
            label = "AI Fallback (Original)"
        else:
            label = "AI Image"
        try:
            st.image(Image.open(st.session_state.ai_path), caption=label, width='stretch')
        except Exception:
            st.warning("Failed to display AI image.")

# ---- Column 3: Vectorization ----
with col3:
    st.markdown("### 3. Vectorize")
    stroke_width = st.text_input("Stroke width", value="1mm")
    rdp = st.number_input("RDP epsilon", value=0.8, step=0.1)
    thresh_bias = st.number_input("Threshold bias", value=0.2, step=0.05)
    upscale = st.number_input("Upscale", value=4, step=1, min_value=1)
    corner_angle = st.number_input("Corner angle", value=65, step=5)
    smooth_win = st.number_input("Smooth window", value=21, step=2)
    smooth_order = st.number_input("Smooth order", value=3, step=1)
    if not st.session_state.ai_path:
        st.info("Run AI step first.")
        run_vec = st.button("Run / Re-run Vectorization", key='run_vec_btn', disabled=True)
    else:
        run_vec = st.button("Run / Re-run Vectorization", key='run_vec_btn')
    clear_svg = st.button("Clear SVG Result", key='clear_svg_btn')
    if clear_svg:
        st.session_state.svg_text = None
        st.session_state.vector_time = None
        st.session_state.path_count = None
    if run_vec and st.session_state.ai_path:
        src_for_trace = st.session_state.ai_path  # enforced AI path
        prev = (v1.INPUT_FILE, v1.OUTPUT_FILE, v1.STROKE_WIDTH, v1.RDP_EPS, v1.THRESH_BIAS, v1.UPSCALE, v1.CORNER_ANGLE, v1.SMOOTH_WIN, v1.SMOOTH_ORDER)
        work_dir = Path(st.session_state.workspace_dir.name)
        out_svg_path = work_dir/'output.svg'
        t0=time.time()
        try:
            v1.INPUT_FILE = src_for_trace
            v1.OUTPUT_FILE = str(out_svg_path)
            v1.STROKE_WIDTH = stroke_width
            v1.RDP_EPS = rdp
            v1.THRESH_BIAS = thresh_bias
            v1.UPSCALE = upscale
            v1.CORNER_ANGLE = corner_angle
            v1.SMOOTH_WIN = smooth_win
            v1.SMOOTH_ORDER = smooth_order
            v1.main()
        except Exception as e:
            st.error(f"Vectorization failed: {e}")
        finally:
            (v1.INPUT_FILE, v1.OUTPUT_FILE, v1.STROKE_WIDTH, v1.RDP_EPS, v1.THRESH_BIAS,
             v1.UPSCALE, v1.CORNER_ANGLE, v1.SMOOTH_WIN, v1.SMOOTH_ORDER) = prev
        if out_svg_path.exists():
            st.session_state.svg_text = out_svg_path.read_text()
            st.session_state.vector_time = time.time()-t0
            st.session_state.path_count = st.session_state.svg_text.count('<path')
            st.success(f"Vectorization completed in {st.session_state.vector_time:.2f}s – {st.session_state.path_count} paths (re-run OK)")
    if st.session_state.vector_time:
        st.caption(f"Last run: {st.session_state.vector_time:.2f}s, paths={st.session_state.path_count}")

# ---- Column 4: Result ----
with col4:
    st.markdown("### 4. Result")
    if st.session_state.svg_text:
        svg_text = st.session_state.svg_text
        try:
            preview = Image.open(io.BytesIO(_svg_to_png_bytes(svg_text)))
            if preview.mode in ("RGBA","LA"):
                bg = Image.new("RGB", preview.size, (255,255,255))
                alpha = preview.split()[-1]
                bg.paste(preview, mask=alpha)
                preview = bg
            else:
                preview = preview.convert("RGB")
        except Exception:
            preview = _rasterize_paths(svg_text)
        st.image(preview, caption="SVG Preview", width='stretch')
        st.download_button("Download SVG", data=svg_text, file_name='trace.svg', mime='image/svg+xml')
        with st.expander("SVG Source"):
            st.code(svg_text, language='xml')
    else:
        st.info("Run vectorization to view result.")

## End main layout

