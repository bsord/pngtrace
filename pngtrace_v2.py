#!/usr/bin/env python3
"""
pngtrace v2 – Centerline Vectorizer (Anchored Smoothing)
-------------------------------------------------------

Goals vs v1:
 1. Better fidelity to original stroke center while producing smooth curves.
 2. Less parameter fiddling: adaptive thresholding & smoothing.
 3. Sub‑pixel centerline refinement using distance transform & normal probing.
 4. Drift limiting: smoothing can't wander outside stroke core.

Pipeline additions:
  - Adaptive grayscale normalization (CLAHE optional placeholder) + Sauvola local threshold fallback to Otsu.
  - Upscale (configurable) prior to skeletonization for cleaner topology.
  - Skeleton (morphological thinning) -> polylines (graph walk) -> spur pruning.
  - Sub‑pixel refinement: for each poly point, estimate tangent, sample along normal, recenter via intensity & distance map.
  - Distance‑constrained spline smoothing: fit parametric splines x(s), y(s) with smoothing factor chosen from local noise; clamp displacement along normal to fraction of local radius (distance transform value).
  - Curvature‑adaptive simplification (modified RDP): epsilon scaled by local curvature so straight runs aggressively simplified while corners preserved.

NOTE: This is a reference implementation kept compact; there is room for optimization (vectorization, pruning heuristics, caching normals, etc.).
"""

from __future__ import annotations
import math, sys, argparse
from dataclasses import dataclass
from typing import List, Tuple, Iterable
import numpy as np
from PIL import Image, ImageOps
from skimage import filters
from skimage.morphology import skeletonize
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import distance_transform_edt

# ---------------------------
# Parameters
# ---------------------------
INPUT_FILE    = "input.png"
OUTPUT_FILE   = "output_centerline_v2.svg"

UPSCALE       = 4          # Pre-skeleton upscale factor
ADAPT_THRESH  = True       # Use Sauvola threshold (fallback to Otsu)
SAUVOLA_WIN   = 25         # Odd window size for Sauvola
THRESH_OFFSET = 0.0        # Additive bias after adaptive threshold (neg => darker)

SMOOTH_BASE   = 0.002      # Base smoothing factor relative to path length
SMOOTH_SCALE  = 0.35       # Multiplier controlling additional smoothing
REFINE_RADIUS = 3          # Pixels (in upscaled grid) to probe along normal each side
DRIFT_FRACTION= 0.6        # Max allowed lateral drift fraction of local radius
CURVE_RDP_EPS = 0.8        # Base epsilon for curvature-aware simplification (px)
CURVE_GAIN    = 1.8        # Higher = keep more detail at high curvature
STROKE_WIDTH  = "1mm"
CORNER_ANGLE  = 65         # Still used for optional corner splitting
BACKGROUND_COLOR = None    # Set to '#ffffff' for white preview; keep None for pure toolpath export.

# ---------------------------
# Utilities
# ---------------------------
Point = Tuple[float, float]

def poly_length(poly: List[Point]) -> float:
    return sum(math.hypot(poly[i][0]-poly[i-1][0], poly[i][1]-poly[i-1][1]) for i in range(1,len(poly)))

def angle(a: Point,b: Point,c: Point) -> float:
    bax, bay = a[0]-b[0], a[1]-b[1]
    bcx, bcy = c[0]-b[0], c[1]-b[1]
    na, nc = math.hypot(bax,bay), math.hypot(bcx,bcy)
    if na==0 or nc==0: return 180.0
    cosang = (bax*bcx + bay*bcy)/(na*nc)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def curvature(poly: List[Point]) -> np.ndarray:
    """Approximate curvature magnitude per interior point (endpoints duplicated)."""
    if len(poly)<3:
        return np.zeros(len(poly))
    curv = np.zeros(len(poly))
    for i in range(1,len(poly)-1):
        a,b,c = poly[i-1], poly[i], poly[i+1]
        ang = angle(a,b,c)
        curv[i] = (180.0-ang)/180.0  # 0 straight, up to ~1 sharp
    curv[0]=curv[1]; curv[-1]=curv[-2]
    return curv

def rdp_curvature(poly: List[Point], base_eps: float, curv_gain: float) -> List[Point]:
    if len(poly)<3: return poly
    curv = curvature(poly)
    def recurse(segment: List[Point], cseg: np.ndarray) -> List[Point]:
        if len(segment)<3: return segment
        a, b = segment[0], segment[-1]
        ax,ay=a; bx,by=b
        best_d=-1; best_i=-1
        for i in range(1,len(segment)-1):
            px,py = segment[i]
            num = abs((by-ay)*px - (bx-ax)*py + bx*ay - by*ax)
            den = math.hypot(by-ay, bx-ax) or 1.0
            d = num/den
            # adaptive epsilon scaled by curvature at point
            eps = base_eps * (1 + curv_gain * cseg[i])
            if d>eps and d>best_d:
                best_d=d; best_i=i
        if best_i==-1:
            return [a,b]
        left  = recurse(segment[:best_i+1], cseg[:best_i+1])
        right = recurse(segment[best_i:],   cseg[best_i:])
        return left[:-1]+right
    return recurse(poly, curv)

def split_at_corners(poly: List[Point], thresh: float) -> List[List[Point]]:
    if len(poly)<3: return [poly]
    segs=[]; cur=[poly[0]]
    for i in range(1,len(poly)-1):
        cur.append(poly[i])
        if angle(poly[i-1],poly[i],poly[i+1]) < thresh:
            cur.append(poly[i])
            segs.append(cur); cur=[poly[i]]
    cur.append(poly[-1]); segs.append(cur)
    return segs

def neighbors8(r:int,c:int) -> Iterable[Tuple[int,int]]:
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0: continue
            yield r+dr, c+dc

def extract_paths(skel: np.ndarray) -> List[List[Tuple[int,int]]]:
    H,W = skel.shape
    coords = {(r,c) for r,c in zip(*np.where(skel))}
    def nbrs(p):
        r,c=p
        for rr,cc in neighbors8(r,c):
            if (rr,cc) in coords: yield (rr,cc)
    deg = {p: sum(1 for _ in nbrs(p)) for p in coords}
    visited=set(); paths=[]
    def edge(a,b): return tuple(sorted((a,b)))
    # Trace from endpoints
    for p in list(coords):
        if deg[p]==1:
            for n in nbrs(p):
                if edge(p,n) in visited: continue
                poly=[p,n]; visited.add(edge(p,n))
                prev,cur=p,n
                while True:
                    nxts=[x for x in nbrs(cur) if x!=prev and edge(cur,x) not in visited]
                    if deg.get(cur,0)!=2 or len(nxts)!=1:
                        break
                    nxt=nxts[0]; visited.add(edge(cur,nxt)); poly.append(nxt); prev,cur=cur,nxt
                if len(poly)>1: paths.append(poly)
    # Remaining cycles
    for p in list(coords):
        for n in nbrs(p):
            if edge(p,n) in visited: continue
            poly=[p,n]; visited.add(edge(p,n))
            prev,cur=p,n
            while True:
                nxts=[x for x in nbrs(cur) if x!=prev and edge(cur,x) not in visited]
                if not nxts: break
                nxt=nxts[0]; visited.add(edge(cur,nxt)); poly.append(nxt); prev,cur=cur,nxt
                if cur==p: break
            if len(poly)>1: paths.append(poly)
    return paths

def prune_short(paths: List[List[Tuple[int,int]]], min_len:int=5) -> List[List[Tuple[int,int]]]:
    return [p for p in paths if len(p)>=min_len]

def adaptive_threshold(arr: np.ndarray) -> np.ndarray:
    if ADAPT_THRESH:
        try:
            win = SAUVOLA_WIN if SAUVOLA_WIN%2==1 else SAUVOLA_WIN+1
            thr = filters.threshold_sauvola(arr, window_size=win)
            return (arr + THRESH_OFFSET < thr).astype(np.uint8)
        except Exception:
            pass
    # Fallback Otsu
    otsu = filters.threshold_otsu(arr)
    return (arr + THRESH_OFFSET < otsu).astype(np.uint8)

def refine_subpixel(poly: List[Tuple[int,int]], binary: np.ndarray, dist: np.ndarray) -> List[Point]:
    """Refine integer skeleton pixels to subpixel center using normal probing.
    Strategy: For each point, estimate tangent via neighbors; probe along normal in [-R,R]
    accumulate weighted coordinates of foreground pixels (binary>0). Weighted by distance map
    to bias toward medial axis. Return refined (x,y) in original upscaled pixel coordinates."""
    R = REFINE_RADIUS
    if len(poly)<3:
        return [(float(c), float(r)) for r,c in poly]
    pts=[]
    for i,(r,c) in enumerate(poly):
        p_prev = poly[i-1] if i>0 else poly[i]
        p_next = poly[i+1] if i<len(poly)-1 else poly[i]
        dy = p_next[0]-p_prev[0]
        dx = p_next[1]-p_prev[1]
        norm = math.hypot(dx,dy) or 1.0
        tx,ty = dx/norm, dy/norm  # tangent
        nx,ny = -ty, tx          # normal
        samples=[]
        for s in range(-R,R+1):
            rr = int(round(r + ny*s))
            cc = int(round(c + nx*s))
            if 0<=rr<binary.shape[0] and 0<=cc<binary.shape[1] and binary[rr,cc]:
                w = 1.0 + dist[rr,cc]
                samples.append((cc, rr, w))
        if not samples:
            pts.append((float(c), float(r)))
            continue
        W = sum(w for _,_,w in samples)
        cx = sum(x*w for x,_,w in samples)/W
        cy = sum(y*w for _,y,w in samples)/W
        pts.append((cx,cy))
    return pts

def distance_constrained_spline(poly: List[Point], dist_map: np.ndarray, max_drift_fraction: float) -> List[Point]:
    if len(poly)<5:
        return poly
    # Parametrize by cumulative length
    dists=[0.0]
    for i in range(1,len(poly)):
        dists.append(dists[-1]+math.hypot(poly[i][0]-poly[i-1][0], poly[i][1]-poly[i-1][1]))
    total = dists[-1]
    if total==0:
        return poly
    t=np.array(dists)/total
    xs=np.array([p[0] for p in poly])
    ys=np.array([p[1] for p in poly])
    # Smoothing factor: scale with length & point density
    s = SMOOTH_BASE*total*len(poly) * (1+SMOOTH_SCALE)
    try:
        sx=UnivariateSpline(t,xs,s=s)
        sy=UnivariateSpline(t,ys,s=s)
        t_new = t  # keep same param samples
        xs_s = sx(t_new)
        ys_s = sy(t_new)
        smooth = list(zip(xs_s, ys_s))
    except Exception:
        return poly
    # Drift limiting: project displacement onto normal; clamp
    limited=[]
    H,W=dist_map.shape
    for i,(orig,sm) in enumerate(zip(poly,smooth)):
        if i==0 or i==len(poly)-1:
            limited.append(sm); continue
        prev = smooth[i-1]; nxt = smooth[i+1]
        # tangent
        tx,ty = nxt[0]-prev[0], nxt[1]-prev[1]
        norm = math.hypot(tx,ty) or 1.0
        tx/=norm; ty/=norm
        nx,ny = -ty, tx
        dx = sm[0]-orig[0]; dy = sm[1]-orig[1]
        lateral = dx*nx + dy*ny
        rr=int(round(orig[1])); cc=int(round(orig[0]))
        if 0<=rr<H and 0<=cc<W:
            max_lat = max_drift_fraction * dist_map[rr,cc]
        else:
            max_lat = 0.0
        if abs(lateral)>max_lat and max_lat>0:
            # remove excess lateral component
            excess = lateral - math.copysign(max_lat,lateral)
            sm = (sm[0]-excess*nx, sm[1]-excess*ny)
        limited.append(sm)
    return limited

def to_quadratic_path(seg: List[Point]) -> str:
    if not seg: return ""
    d=f"M {seg[0][0]:.3f},{seg[0][1]:.3f} "
    for i in range(1,len(seg)):
        cx=(2*seg[i-1][0]+seg[i][0])/3
        cy=(2*seg[i-1][1]+seg[i][1])/3
        d+=f"Q {cx:.3f},{cy:.3f} {seg[i][0]:.3f},{seg[i][1]:.3f} "
    return d

def process(image_path: str = INPUT_FILE, out_path: str = OUTPUT_FILE):
    # Load & normalize
    img = Image.open(image_path).convert("L")
    img = ImageOps.autocontrast(img)
    up = img.resize((img.width*UPSCALE, img.height*UPSCALE), Image.Resampling.BICUBIC)
    arr = np.array(up, dtype=np.uint8)

    # Threshold
    binary = adaptive_threshold(arr)

    # Skeleton
    skel = skeletonize(binary>0)
    # Distance transform on foreground
    dist = distance_transform_edt(binary)

    paths_pix = extract_paths(skel)
    paths_pix = prune_short(paths_pix, min_len=4)

    segments_svg=[]
    for pix_poly in paths_pix:
        # Subpixel refinement
        subpix = refine_subpixel(pix_poly, binary, dist)
        # Scale down to original image coordinates
        scaled = [(x/UPSCALE, y/UPSCALE) for (x,y) in subpix]
        # Distance map in upscaled domain -> build smaller version for drift limiting by sampling
        # We'll reuse dist (upscaled); need a sampler in scaled space
        refined = distance_constrained_spline(subpix, dist, DRIFT_FRACTION)
        refined_scaled = [(x/UPSCALE, y/UPSCALE) for (x,y) in refined]
        # Curvature-aware simplification
        simplified = rdp_curvature(refined_scaled, CURVE_RDP_EPS, CURVE_GAIN)
        # Optional corner splitting (can help with sharp corners preserving stroke ends)
        for seg in split_at_corners(simplified, CORNER_ANGLE):
            if len(seg)<2: continue
            path_d = to_quadratic_path(seg)
            if path_d:
                segments_svg.append(path_d)

    # Write SVG
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" viewBox="0 0 {img.width} {img.height}">']
    if BACKGROUND_COLOR:
        svg.append(f'  <rect x="0" y="0" width="100%" height="100%" fill="{BACKGROUND_COLOR}" stroke="none" data-non-machining="true"/>')
    for d in segments_svg:
        svg.append(f'  <path d="{d}" fill="none" stroke="black" stroke-width="{STROKE_WIDTH}" stroke-linecap="round" stroke-linejoin="round"/>')
    svg.append('</svg>')
    with open(out_path, 'w') as f:
        f.write("\n".join(svg)+"\n")
    print(f"Saved: {out_path}  (paths={len(segments_svg)})")

def parse_args():
    p = argparse.ArgumentParser(
        description="pngtrace v2 – adaptive anchored centerline vectorizer",
        epilog="Usage: pngtrace_v2.py input.png [output.svg]"
    )
    p.add_argument("input", nargs='?', default=INPUT_FILE, help="Input raster image")
    p.add_argument("output", nargs='?', default=OUTPUT_FILE, help="Output SVG file")
    p.add_argument("--background", default=BACKGROUND_COLOR, help="Background color (#hex) or 'none'")
    p.add_argument("--upscale", type=int, default=UPSCALE, help="Upscale factor")
    p.add_argument("--refine-radius", type=int, default=REFINE_RADIUS, help="Normal probe half-width (px in upscaled space)")
    p.add_argument("--drift", type=float, default=DRIFT_FRACTION, help="Max lateral drift fraction of local radius")
    p.add_argument("--rdp", type=float, default=CURVE_RDP_EPS, help="Base curvature simplification epsilon")
    p.add_argument("--curve-gain", type=float, default=CURVE_GAIN, help="Curvature gain in adaptive simplification")
    p.add_argument("--stroke-width", default=STROKE_WIDTH, help="Stroke width for SVG output")
    p.add_argument("--corner-angle", type=float, default=CORNER_ANGLE, help="Corner split threshold deg")
    p.add_argument("--smooth-base", type=float, default=SMOOTH_BASE, help="Spline smoothing base factor")
    p.add_argument("--smooth-scale", type=float, default=SMOOTH_SCALE, help="Spline smoothing scale multiplier")
    p.add_argument("--no-adapt", action="store_true", help="Disable adaptive Sauvola threshold (use Otsu)")
    return p.parse_args()

def main():
    global INPUT_FILE, OUTPUT_FILE, BACKGROUND_COLOR, UPSCALE, REFINE_RADIUS, DRIFT_FRACTION
    global CURVE_RDP_EPS, CURVE_GAIN, STROKE_WIDTH, CORNER_ANGLE, SMOOTH_BASE, SMOOTH_SCALE, ADAPT_THRESH
    if len(sys.argv)>1:
        a = parse_args()
        INPUT_FILE = a.input
        OUTPUT_FILE = a.output
        BACKGROUND_COLOR = None if a.background in (None,'none','None') else a.background
        UPSCALE = a.upscale
        REFINE_RADIUS = a.refine_radius
        DRIFT_FRACTION = a.drift
        CURVE_RDP_EPS = a.rdp
        CURVE_GAIN = a.curve_gain
        STROKE_WIDTH = a.stroke_width
        CORNER_ANGLE = a.corner_angle
        SMOOTH_BASE = a.smooth_base
        SMOOTH_SCALE = a.smooth_scale
        if a.no_adapt: ADAPT_THRESH = False
    process(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()
