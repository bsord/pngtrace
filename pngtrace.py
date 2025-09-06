#!/usr/bin/env python3
"""
Vectorizer for CNC/Laser
------------------------

Modes:
    - centerline (existing): skeleton-based centerline stroke paths.
    - fill: extract filled region polygons (black shapes) with even-odd rule for holes.

In fill mode the binary mask is contour traced (outer + hole boundaries) and output as
filled paths (no stroke) enabling direct engraving / cutting fill workflows.

Dependencies:
        pip install pillow numpy scikit-image scipy opencv-python
"""

import math
import sys
import argparse
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage import measure
from scipy import ndimage as ndi
try:
    import cv2  # Optional (preferred for contour hierarchy)
except Exception:  # pragma: no cover
    cv2 = None
from scipy.signal import savgol_filter

# ---------------------------
# Parameters (tweak these)
# ---------------------------
INPUT_FILE   = "input.png"      # default; can be overridden by CLI
OUTPUT_FILE  = "output_centerline.svg"

UPSCALE      = 4          # scale factor to reduce pixel jaggies
THRESH_BIAS  = 0.8        # threshold bias; increase if faint lines disappear
SMOOTH_WIN   = 21         # Savitzky-Golay window (odd, adaptive if too short)
SMOOTH_ORDER = 3          # polynomial order for Savitzky-Golay
RDP_EPS      = 0.8        # simplification tolerance (px); lower = more detail
STROKE_WIDTH = "1mm"    # e.g. "0.1mm", "0.001in", "2px"
CORNER_ANGLE = 65         # preserve corners sharper than this (deg)
# Optional background color for visualization (set to '#ffffff' for white). None = no rect.
BACKGROUND_COLOR = None   # Adding a filled rect may create a 'fill' object some CAM tools try to engrave.
                          # Leave None for machining exports; set to color only for on-screen viewing.
MODE         = "fill"  # or "centerline"
# Optional outline cut padding (in output pixels). 0 disables.
OUTLINE_PADDING = 0.0
# ---------------------------

# Utility functions
def rdp(points, epsilon):
    """Ramer–Douglas–Peucker simplification."""
    if len(points) < 3: return points

    def perp_dist(p, a, b):
        (x0,y0),(x1,y1) = a,b
        (x,y) = p
        if (x0==x1 and y0==y1): return math.hypot(x-x0,y-y0)
        num = abs((y1-y0)*x - (x1-x0)*y + x1*y0 - y1*x0)
        den = math.hypot(y1-y0, x1-x0)
        return num/den if den!=0 else 0.0

    dmax, idx = 0.0, 0
    for i in range(1, len(points)-1):
        d = perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            idx, dmax = i, d
    if dmax > epsilon:
        rec1 = rdp(points[:idx+1], epsilon)
        rec2 = rdp(points[idx:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [points[0], points[-1]]

def angle(a,b,c):
    """Angle at point b (deg)."""
    bax, bay = a[0]-b[0], a[1]-b[1]
    bcx, bcy = c[0]-b[0], c[1]-b[1]
    na, nc = math.hypot(bax,bay), math.hypot(bcx,bcy)
    if na==0 or nc==0: return 180.0
    cosang = (bax*bcx + bay*bcy)/(na*nc)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def split_at_corners(pts, thresh=65):
    """Split polyline at sharp corners."""
    if len(pts)<3: return [pts]
    segs, cur = [], [pts[0]]
    for i in range(1,len(pts)-1):
        cur.append(pts[i])
        if angle(pts[i-1],pts[i],pts[i+1]) < thresh:
            segs.append(cur[:])
            cur=[pts[i]]
    cur.append(pts[-1])
    segs.append(cur)
    return segs

def smooth_poly(poly, scale):
    """Savitzky-Golay smoothing on polyline."""
    n=len(poly)
    xs=np.array([p[1] for p in poly],dtype=float)
    ys=np.array([p[0] for p in poly],dtype=float)
    if n>=7:
        win=min(SMOOTH_WIN, n if n%2==1 else n-1)
        order=min(SMOOTH_ORDER, win-2)
        xs=savgol_filter(xs, win, order, mode="interp")
        ys=savgol_filter(ys, win, order, mode="interp")
    return list(zip(xs/scale, ys/scale)) # (x,y)

# ---------------------------
# Main pipeline
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Raster vectorizer: PNG -> SVG (centerline or filled)",
        epilog="Usage: pngtrace.py input.png [output.svg]"
    )
    parser.add_argument("input", nargs='?', default=INPUT_FILE, help="Input raster (PNG, JPG, etc.)")
    parser.add_argument("output", nargs='?', default=OUTPUT_FILE, help="Output SVG file")
    parser.add_argument("--stroke-width", "-w", dest="stroke_width", default=STROKE_WIDTH,
                        help="Stroke width for SVG (e.g. 0.1mm, 1px)")
    parser.add_argument("--background", dest="background", default=BACKGROUND_COLOR,
                        help="Background color (e.g. #ffffff) or 'none'")
    parser.add_argument("--upscale", type=int, default=UPSCALE, help="Upscale factor before skeletonization")
    parser.add_argument("--rdp", type=float, default=RDP_EPS, help="RDP simplification epsilon")
    parser.add_argument("--corner-angle", type=float, default=CORNER_ANGLE, help="Corner split threshold degrees")
    parser.add_argument("--smooth-win", type=int, default=SMOOTH_WIN, help="Savitzky-Golay window (odd)")
    parser.add_argument("--smooth-order", type=int, default=SMOOTH_ORDER, help="Savitzky-Golay polynomial order")
    parser.add_argument("--thresh-bias", type=float, default=THRESH_BIAS, help="Threshold bias")
    parser.add_argument("--mode", choices=["centerline","fill"], default=MODE, help="Vectorization mode")
    parser.add_argument("--outline-padding", type=float, default=OUTLINE_PADDING,
                        help="External outline cut padding in output pixels (0 disables)")
    return parser.parse_args()

def polygon_area(loop):
    """Signed area (x,y)."""
    a=0.0
    for i in range(len(loop)):
        x1,y1=loop[i]
        x2,y2=loop[(i+1)%len(loop)]
        a+=x1*y2 - x2*y1
    return a/2.0

def simplify_loop(loop, epsilon):
    """RDP for closed loop; keeps closure."""
    if len(loop)<3:
        return loop
    # duplicate first at end for continuity; RDP expects list -> open path; we treat open then re-close
    open_path = loop + [loop[0]]
    simp = rdp(open_path, epsilon)
    if simp[0]!=simp[-1]:
        simp[-1]=simp[0]
    return simp[:-1]

def extract_fill_loops(binary_mask, scale):
    """Return list of loops (each list of (x,y)) scaled down by 'scale'.
    Uses OpenCV if available to get hierarchy (outer vs holes) else skimage.measure.find_contours.
    """
    loops=[]
    if cv2 is not None:
        m = (binary_mask>0).astype(np.uint8)*255
        contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            pts = [(p[0][0]/scale, p[0][1]/scale) for p in cnt]  # (x,y)
            if len(pts)>=3:
                loops.append(pts)
    else:
        # skimage returns (row,col) contours at level 0.5
        for c in measure.find_contours((binary_mask>0).astype(float), 0.5):
            pts=[(p[1]/scale, p[0]/scale) for p in c]
            if len(pts)>=3:
                loops.append(pts)
    return loops

def compute_outline_loops(binary_mask, scale, padding_px):
    """Compute external outline loop(s) around the foreground with padding in output pixels.
    - binary_mask: upscaled binary (uint8/boolean) where True/1 = foreground
    - scale: UPSCALE factor to downscale coordinates to output units
    - padding_px: padding at output scale; converted to upscaled pixel radius
    Returns list of loops (each list of (x,y)) scaled down by 'scale'.
    """
    if padding_px is None or padding_px <= 0:
        return []
    # Convert padding from output pixels to upscaled mask pixels
    rad_up = max(1, int(round(padding_px * scale)))
    mask_bool = (binary_mask > 0)
    try:
        selem = disk(rad_up)
        dil = binary_dilation(mask_bool, footprint=selem)
    except Exception:
        dil = binary_dilation(mask_bool, iterations=rad_up)
    # Fill holes so outline hugs only the outer perimeter(s)
    dil_filled = ndi.binary_fill_holes(dil)
    loops = []
    if cv2 is not None:
        m = (dil_filled.astype(np.uint8)) * 255
        contours, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            pts = [(p[0][0]/scale, p[0][1]/scale) for p in cnt]
            if len(pts) >= 3:
                loops.append(pts)
    else:
        for c in measure.find_contours(dil_filled.astype(float), 0.5):
            pts=[(p[1]/scale, p[0]/scale) for p in c]
            if len(pts)>=3:
                loops.append(pts)
    # Light simplification to keep outline smooth but faithful
    out = []
    for loop in loops:
        lp = simplify_loop(loop, max(0.5, RDP_EPS/2))
        if len(lp) >= 3:
            out.append(lp)
    return out

def main():
    global INPUT_FILE, OUTPUT_FILE, STROKE_WIDTH, BACKGROUND_COLOR
    global UPSCALE, RDP_EPS, CORNER_ANGLE, SMOOTH_WIN, SMOOTH_ORDER, THRESH_BIAS, MODE
    if len(sys.argv)>1:
        # Use argparse only if user supplies args (keeps zero-arg backward compatibility speed)
        args = parse_args()
        INPUT_FILE = args.input
        OUTPUT_FILE = args.output
        STROKE_WIDTH = args.stroke_width
        BACKGROUND_COLOR = None if (args.background in (None, 'none', 'None')) else args.background
        UPSCALE = args.upscale
        RDP_EPS = args.rdp
        CORNER_ANGLE = args.corner_angle
        SMOOTH_WIN = args.smooth_win
        SMOOTH_ORDER = args.smooth_order
        THRESH_BIAS = args.thresh_bias
        MODE = args.mode
        try:
            globals()['OUTLINE_PADDING'] = float(args.outline_padding)
        except Exception:
            pass
    # Load and upscale
    img=Image.open(INPUT_FILE).convert("L")
    img=ImageOps.autocontrast(img)
    img_up=img.resize((img.width*UPSCALE,img.height*UPSCALE),resample=Image.Resampling.BICUBIC)
    arr=np.array(img_up)

    # Threshold
    thr=np.mean(arr)-THRESH_BIAS*np.std(arr)
    binary=(arr<thr).astype(np.uint8)

    if MODE == "fill":
        # Extract filled region contours
        loops = extract_fill_loops(binary, UPSCALE)
        # Simplify & optionally smooth (reuse smoothing by mapping into synthetic poly order)
        out_loops=[]
        for loop in loops:
            # Optional smoothing: treat x,y arrays
            if len(loop)>=7:
                xs=np.array([p[0] for p in loop])
                ys=np.array([p[1] for p in loop])
                win=min(SMOOTH_WIN, len(loop) if len(loop)%2==1 else len(loop)-1)
                win=max(5, win)
                order=min(SMOOTH_ORDER, win-2)
                try:
                    xs=savgol_filter(xs, win, order, mode="wrap")
                    ys=savgol_filter(ys, win, order, mode="wrap")
                except Exception:
                    pass
                loop=[(xs[i], ys[i]) for i in range(len(xs))]
            loop=simplify_loop(loop, RDP_EPS)
            if len(loop)>=3:
                out_loops.append(loop)
        # Build a single path with subpaths for even-odd fill
        path_d=""
        for lp in out_loops:
            path_d += f"M {lp[0][0]:.2f},{lp[0][1]:.2f} "
            for (x,y) in lp[1:]:
                path_d += f"L {x:.2f},{y:.2f} "
            path_d += "Z "
        # Compute outline and bounds expansion if needed
        try:
            outline_loops = compute_outline_loops(binary, UPSCALE, OUTLINE_PADDING)
        except Exception:
            outline_loops = []
        left_exp = top_exp = right_exp = bottom_exp = 0.0
        if outline_loops:
            xs = [x for lp in outline_loops for (x,_) in lp]
            ys = [y for lp in outline_loops for (_,y) in lp]
            if xs and ys:
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                left_exp = max(0.0, 0.0 - minx)
                top_exp = max(0.0, 0.0 - miny)
                right_exp = max(0.0, maxx - img.width)
                bottom_exp = max(0.0, maxy - img.height)
        dx, dy = left_exp, top_exp
        new_w = img.width + left_exp + right_exp
        new_h = img.height + top_exp + bottom_exp
        # Build SVG with optional group translation
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(math.ceil(new_w))}" height="{int(math.ceil(new_h))}" viewBox="0 0 {new_w:.2f} {new_h:.2f}">\n'
        if BACKGROUND_COLOR:
            svg+=f'  <rect x="0" y="0" width="100%" height="100%" fill="{BACKGROUND_COLOR}" stroke="none"/>\n'
        if dx!=0 or dy!=0:
            svg+=f'  <g transform="translate({dx:.2f},{dy:.2f})">\n'
            indent = "    "
        else:
            indent = "  "
        svg+=indent + f'<path d="{path_d.strip()}" fill="black" fill-rule="evenodd" stroke="none"/>\n'
        for lp in outline_loops:
            d_o = f"M {lp[0][0]:.2f},{lp[0][1]:.2f} " + " ".join(f"L {x:.2f},{y:.2f}" for (x,y) in lp[1:]) + " Z"
            svg+=indent + f'<path d="{d_o}" fill="none" stroke="black" stroke-width="{STROKE_WIDTH}" stroke-linejoin="round" stroke-linecap="round" data-cut="outline"/>\n'
        if dx!=0 or dy!=0:
            svg+="  </g>\n"
        svg+="</svg>\n"
        with open(OUTPUT_FILE,"w") as f:
            f.write(svg)
        print(f"Saved: {OUTPUT_FILE}  (filled loops={len(out_loops)}; outline={len(outline_loops)})")
        return

    # Skeletonize (centerline mode)
    skel=skeletonize(binary>0)
    H,W=skel.shape
    coords=set(zip(*np.where(skel)))

    # Graph build
    nbrs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    def neighbors(p):
        r,c=p
        for dr,dc in nbrs:
            rr,cc=r+dr,c+dc
            if (rr,cc) in coords: yield (rr,cc)
    deg={p:sum(1 for _ in neighbors(p)) for p in coords}

    # Trace
    visited=set(); paths=[]
    def edge(a,b): return tuple(sorted((a,b)))
    for p in coords:
        if deg[p]==1: # endpoint
            for n in neighbors(p):
                ek=edge(p,n)
                if ek in visited: continue
                poly=[p,n]; visited.add(ek)
                prev,cur=p,n
                while True:
                    nbrs2=[x for x in neighbors(cur) if x!=prev]
                    if deg.get(cur,0)!=2 or not nbrs2: break
                    nxt=[x for x in nbrs2 if edge(cur,x) not in visited]
                    if not nxt: break
                    nxt=nxt[0]; visited.add(edge(cur,nxt))
                    poly.append(nxt); prev,cur=cur,nxt
                if len(poly)>2: paths.append(poly)
    # cycles
    for p in coords:
        for n in neighbors(p):
            ek=edge(p,n)
            if ek not in visited:
                poly=[p,n]; visited.add(ek)
                prev,cur=p,n
                while True:
                    nbrs2=[x for x in neighbors(cur) if x!=prev]
                    nxt=None
                    for x in nbrs2:
                        if edge(cur,x) not in visited:
                            nxt=x; break
                    if not nxt: break
                    visited.add(edge(cur,nxt))
                    poly.append(nxt); prev,cur=cur,nxt
                    if cur==p: break
                if len(poly)>2: paths.append(poly)

    # Smooth & simplify
    segments=[]
    for poly in paths:
        sm=smooth_poly(poly,UPSCALE)
        sm=rdp(sm,RDP_EPS)
        for seg in split_at_corners(sm,CORNER_ANGLE):
            if len(seg)<2: continue
            d=f"M {seg[0][0]:.2f},{seg[0][1]:.2f} "
            for i in range(1,len(seg)):
                cx=(seg[i-1][0]*2+seg[i][0])/3
                cy=(seg[i-1][1]*2+seg[i][1])/3
                d+=f"Q {cx:.2f},{cy:.2f} {seg[i][0]:.2f},{seg[i][1]:.2f} "
            segments.append(d)

    # Write SVG
    # Optional outline and bounds expansion
    try:
        outline_loops = compute_outline_loops(binary, UPSCALE, OUTLINE_PADDING)
    except Exception:
        outline_loops = []
    left_exp = top_exp = right_exp = bottom_exp = 0.0
    if outline_loops:
        xs = [x for lp in outline_loops for (x,_) in lp]
        ys = [y for lp in outline_loops for (_,y) in lp]
        if xs and ys:
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            left_exp = max(0.0, 0.0 - minx)
            top_exp = max(0.0, 0.0 - miny)
            right_exp = max(0.0, maxx - img.width)
            bottom_exp = max(0.0, maxy - img.height)
    dx, dy = left_exp, top_exp
    new_w = img.width + left_exp + right_exp
    new_h = img.height + top_exp + bottom_exp
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(math.ceil(new_w))}" height="{int(math.ceil(new_h))}" viewBox="0 0 {new_w:.2f} {new_h:.2f}">\n'
    if BACKGROUND_COLOR:
        svg+=f'  <rect x="0" y="0" width="100%" height="100%" fill="{BACKGROUND_COLOR}" stroke="none" data-non-machining="true"/>\n'
    if dx!=0 or dy!=0:
        svg+=f'  <g transform="translate({dx:.2f},{dy:.2f})">\n'
        indent = "    "
    else:
        indent = "  "
    for d in segments:
        svg+=indent + f'<path d="{d}" fill="none" stroke="black" stroke-width="{STROKE_WIDTH}" stroke-linecap="round" stroke-linejoin="round"/>\n'
    for lp in outline_loops:
        d_o = f"M {lp[0][0]:.2f},{lp[0][1]:.2f} " + " ".join(f"L {x:.2f},{y:.2f}" for (x,y) in lp[1:]) + " Z"
        svg+=indent + f'<path d="{d_o}" fill="none" stroke="black" stroke-width="{STROKE_WIDTH}" stroke-linejoin="round" stroke-linecap="round" data-cut="outline"/>\n'
    if dx!=0 or dy!=0:
        svg+="  </g>\n"
    svg+="</svg>\n"
    with open(OUTPUT_FILE,"w") as f: f.write(svg)
    print(f"Saved: {OUTPUT_FILE}  (paths={len(segments)}; outline={len(outline_loops)})")

if __name__=="__main__":
    main()
