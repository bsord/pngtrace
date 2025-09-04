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
from skimage.morphology import skeletonize
from skimage import measure
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
THRESH_BIAS  = 0.2        # threshold bias; increase if faint lines disappear
SMOOTH_WIN   = 21         # Savitzky-Golay window (odd, adaptive if too short)
SMOOTH_ORDER = 3          # polynomial order for Savitzky-Golay
RDP_EPS      = 0.8        # simplification tolerance (px); lower = more detail
STROKE_WIDTH = "1mm"    # e.g. "0.1mm", "0.001in", "2px"
CORNER_ANGLE = 65         # preserve corners sharper than this (deg)
# Optional background color for visualization (set to '#ffffff' for white). None = no rect.
BACKGROUND_COLOR = None   # Adding a filled rect may create a 'fill' object some CAM tools try to engrave.
                          # Leave None for machining exports; set to color only for on-screen viewing.
MODE         = "centerline"  # or "fill"
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
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" viewBox="0 0 {img.width} {img.height}">\n'
        if BACKGROUND_COLOR:
            svg+=f'  <rect x="0" y="0" width="100%" height="100%" fill="{BACKGROUND_COLOR}" stroke="none"/>\n'
        svg+=f'  <path d="{path_d.strip()}" fill="black" fill-rule="evenodd" stroke="none"/>\n'
        svg+="</svg>\n"
        with open(OUTPUT_FILE,"w") as f:
            f.write(svg)
        print(f"Saved: {OUTPUT_FILE}  (filled loops={len(out_loops)})")
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
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" viewBox="0 0 {img.width} {img.height}">\n'
    if BACKGROUND_COLOR:
        # Non-stroking background rectangle for visibility; data- attr hints to downstream tools to ignore.
        svg+=f'  <rect x="0" y="0" width="100%" height="100%" fill="{BACKGROUND_COLOR}" stroke="none" data-non-machining="true"/>\n'
    for d in segments:
        svg+=f'  <path d="{d}" fill="none" stroke="black" stroke-width="{STROKE_WIDTH}" stroke-linecap="round" stroke-linejoin="round"/>\n'
    svg+="</svg>\n"
    with open(OUTPUT_FILE,"w") as f: f.write(svg)
    print(f"Saved: {OUTPUT_FILE}  (paths={len(segments)})")

if __name__=="__main__":
    main()
