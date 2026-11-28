#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fact Bites motion-graphics engine.
Draws every scene by code (maps, globes, size comparisons, charts, isometric cities, water tanks...) from a small library of components.
The planner (LLM, with a rule-based backup) picks components from the narration, so no two videos use the same fixed template.
The channel identity (logo, captions, voices, orange brand colour) lives in main.py and never changes; only the visuals change here.
"""
import math, random, json, os, re, hashlib, time, ast, signal
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H, SS = 720, 1280, 2
BOX = (36, 135, 684, 835)             # safe area for content (logo above it, captions below it)
BRAND = (255, 140, 30)
INK, DIM = (236, 242, 255), (125, 140, 178)
_font, _log = None, print

def init(w, h, font_fn, brand, log_fn):
    global W, H, BOX, _font, BRAND, _log
    W, H, _font, BRAND, _log = w, h, font_fn, tuple(brand), log_fn
    BOX = (int(w * .05), int(h * .135), int(w * .95), int(h * .655))
    load_custom()

# ───────────── small helpers ─────────────
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lerp(a, b, k): return a + (b - a) * k
def eo3(k): k = clamp(k); return 1 - (1 - k) ** 3
def smooth(k): k = clamp(k); return k * k * (3 - 2 * k)
def eob(k):
    k = clamp(k); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (k - 1) ** 3 + c1 * (k - 1) ** 2
def seg(k, a, b): return clamp((k - a) / max(b - a, 1e-6))
def mixc(a, b, k): return tuple(int(lerp(a[i], b[i], clamp(k))) for i in range(3))
def shade(c, f): return tuple(int(clamp(v * f, 0, 255)) for v in c)
def num(x, default=0.0):
    if isinstance(x, (int, float)): return float(x)
    s_ = str(x).lower().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s_)
    if not m: return default
    v = float(m.group()); tail = s_[m.end():]
    mm = re.match(r"\s*(trillion|billion|million|thousand|bn)\b", tail)
    if mm: return v * {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "million": 1e6, "thousand": 1e3}[mm.group(1)]
    if re.match(r"k\b", tail): return v * 1e3
    return v
def fmt(v, dec=None):
    a = abs(v)
    if dec is None: dec = 0 if a >= 100 or abs(v - round(v)) < 1e-9 else (1 if a >= 10 else 2)
    if a >= 1e12: return f"{v / 1e12:.{1 if a < 1e14 else 0}f} trillion"
    if a >= 1e9: return f"{v / 1e9:.{1 if a < 1e11 else 0}f} billion"
    if a >= 1e7: return f"{v / 1e6:.0f} million"
    if a >= 1e6: return f"{v / 1e6:.1f} million"
    return f"{v:,.{dec}f}"
def shorten(s, n): s = str(s).strip(); return s if len(s) <= n else s[:n - 1].rstrip() + "…"

# ───────────── canvas: draws at 2x then box-filters down (smooth edges) ─────────────
class G:
    """drawing canvas: starts from the opaque backdrop (so translucent shapes really blend), draws at 2x, then box-filters down"""
    def __init__(s, bg=None):
        s.im = (bg.convert("RGB").resize((W * SS, H * SS), Image.Resampling.BILINEAR) if bg is not None else Image.new("RGB", (W * SS, H * SS), (8, 12, 30)))
        s.d = ImageDraw.Draw(s.im, "RGBA")
    def P(s, pts): return [(x * SS, y * SS) for x, y in pts]
    def poly(s, pts, fill=None, outline=None, w=2):
        if len(pts) < 3: return
        if fill is not None: s.d.polygon(s.P(pts), fill=fill)
        if outline is not None: s.d.line(s.P(list(pts) + [pts[0]]), fill=outline, width=int(w * SS), joint="curve")
    def line(s, pts, fill, w=3):
        if len(pts) >= 2: s.d.line(s.P(pts), fill=fill, width=max(1, int(w * SS)), joint="curve")
    def rect(s, x0, y0, x1, y1, fill=None, outline=None, w=2, r=0):
        b = [x0 * SS, y0 * SS, x1 * SS, y1 * SS]
        if x1 <= x0 or y1 <= y0: return
        if r > 0: s.d.rounded_rectangle(b, radius=min(r, (x1 - x0) / 2, (y1 - y0) / 2) * SS, fill=fill, outline=outline, width=int(w * SS))
        else: s.d.rectangle(b, fill=fill, outline=outline, width=int(w * SS))
    def circle(s, x, y, r, fill=None, outline=None, w=2):
        if r <= 0: return
        s.d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], fill=fill, outline=outline, width=int(w * SS))
    def ellipse(s, x, y, rx, ry, fill=None, outline=None, w=2):
        if rx <= 0 or ry <= 0: return
        s.d.ellipse([(x - rx) * SS, (y - ry) * SS, (x + rx) * SS, (y + ry) * SS], fill=fill, outline=outline, width=int(w * SS))
    def arc(s, x, y, r, a0, a1, fill, w=4):
        s.d.arc([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], a0, a1, fill=fill, width=max(1, int(w * SS)))
    def tw(s, t, size): return _font(int(size * SS)).getlength(str(t)) / SS
    def text(s, x, y, t, size, fill=INK, anchor="mm", stroke=0, sfill=(8, 10, 24)):
        t = str(t); fill = tuple(fill) if len(fill) == 4 else tuple(fill) + (255,)
        s.d.text((x * SS, y * SS), t, font=_font(int(size * SS)), fill=fill, anchor=anchor, stroke_width=int(stroke * SS), stroke_fill=tuple(sfill) + (fill[3],))
    def fit(s, x, y, t, size, maxw, fill=INK, anchor="mm", stroke=0):
        while size > 14 and s.tw(t, size) > maxw: size -= 2
        s.text(x, y, t, size, fill, anchor, stroke)
    def sprite(s, spr, x, y, scale=1.0):
        """paste an RGBA sprite (made at SS scale) centred at logical x,y"""
        if scale <= 0.02: return
        if scale != 1.0: spr = spr.resize((max(1, int(spr.width * scale)), max(1, int(spr.height * scale))), Image.Resampling.BICUBIC)
        px, py = int(x * SS - spr.width / 2), int(y * SS - spr.height / 2)
        if _inside(spr, px, py, s.im.size): s.im.paste(spr.convert("RGB"), (px, py), spr.getchannel("A"))
    def glow(s, x, y, r, col, a=.5):
        if r > 3: s.sprite(_radial(int(r * SS), col, a), x, y)
    def finish(s): return s.im.reduce(SS)

def _inside(spr, x, y, size): return x > -spr.width and y > -spr.height and x < size[0] and y < size[1]

_RAD, _SPR = {}, {}
def _radial(r, col, a):
    key = (max(4, r // 4 * 4), tuple(col), round(a, 1))
    if key not in _RAD:
        rr = key[0]; yy, xx = np.mgrid[-rr:rr, -rr:rr]; d = np.hypot(xx, yy) / rr
        al = (np.clip(1 - d, 0, 1) ** 2 * 255 * a).astype(np.uint8)
        arr = np.zeros((2 * rr, 2 * rr, 4), np.uint8); arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3] = col[0], col[1], col[2], al
        _RAD[key] = Image.fromarray(arr, "RGBA")
    return _RAD[key]

def sphere_sprite(r, col, light=(-.5, -.6), bands=0):
    """shaded ball (at SS scale); cached"""
    R = max(4, int(r * SS) // 4 * 4); key = (R, tuple(col), light, bands)
    if key not in _SPR:
        yy, xx = np.mgrid[-R:R, -R:R].astype(np.float32); nx, ny = xx / R, yy / R; d2 = nx * nx + ny * ny
        nz = np.sqrt(np.clip(1 - d2, 0, 1)); lx, ly = light; lz = math.sqrt(max(.05, 1 - lx * lx - ly * ly))
        lam = np.clip(nx * lx + ny * ly + nz * lz, 0, 1); spec = np.clip(lam, 0, 1) ** 24 * .45
        base = np.array(col, np.float32)[None, None, :] * (.30 + .85 * lam[..., None])
        if bands:
            b = (np.sin(ny * bands * 3.1) * .5 + .5)[..., None] * .22 + .89; base = base * b
        base = base + spec[..., None] * 255; edge = np.clip((1 - np.sqrt(d2)) * R, 0, 1)
        arr = np.zeros((2 * R, 2 * R, 4), np.uint8); arr[..., :3] = np.clip(base, 0, 255).astype(np.uint8); arr[..., 3] = (edge * 255).astype(np.uint8)
        _SPR[key] = Image.fromarray(arr, "RGBA")
    return _SPR[key]

# ───────────── per-video identity + backdrop ─────────────
BG_STYLES = ["grid", "dots", "rays", "waves", "stars", "blueprint"]

class Identity:
    """what changes from video to video INSIDE the fixed channel look: background pattern, accent order, camera rhythm"""
    def __init__(s, pal, seed, avoid_bg=None):
        r = random.Random(seed); s.r = r
        choices = [b for b in BG_STYLES if b != avoid_bg] or BG_STYLES
        s.bg = r.choice(choices); s.pal = pal; s.seed = seed
        s.acc, s.acc2 = (pal["acc"], pal["acc2"]) if r.random() < .5 else (pal["acc2"], pal["acc"])
        s.cols = [s.acc, s.acc2, BRAND, (90, 205, 150), (170, 125, 255), (255, 95, 120), (80, 170, 255)]
        r.shuffle(s.cols[3:]); s.bg1, s.bg2 = pal["bg1"], pal["bg2"]
        s._base = None
    def base(s):
        if s._base is None:
            yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
            c1, c2 = np.array(s.bg1, np.float32), np.array(s.bg2, np.float32)
            top = c1 * .55; bot = c1 * .35 + c2 * .45
            arr = (top * (1 - yy) + bot * yy) * np.ones((1, W, 1), np.float32)
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
            rad = _radial(int(W * .85), s.acc, .22); im.alpha_composite(rad, (int(W / 2 - rad.width / 2), int(H * .92 - rad.height / 2)), (0, 0, rad.width, min(rad.height, H - int(H * .92 - rad.height / 2))) if H - int(H * .92 - rad.height / 2) > 0 else (0, 0, 1, 1))
            s._base = im.convert("RGB")
        return s._base
    def frame(s, t):
        im = s.base().copy().convert("RGBA"); lay = Image.new("RGBA", (W + 160, H + 160), (0, 0, 0, 0)); d = ImageDraw.Draw(lay, "RGBA")
        ac = s.acc; r = random.Random(s.seed + 7); ox, oy = (t * 14) % 80, (t * 9) % 80
        if s.bg in ("grid", "blueprint"):
            step = 80 if s.bg == "grid" else 40
            for x in range(0, W + 160, step): d.line([(x, 0), (x, H + 160)], fill=ac + (22 if s.bg == "grid" else 30,), width=1)
            for y in range(0, H + 160, step): d.line([(0, y), (W + 160, y)], fill=ac + (22 if s.bg == "grid" else 30,), width=1)
            if s.bg == "blueprint":
                for x in range(0, W + 160, 200): d.line([(x, 0), (x, H + 160)], fill=ac + (55,), width=2)
                for y in range(0, H + 160, 200): d.line([(0, y), (W + 160, y)], fill=ac + (55,), width=2)
        elif s.bg == "dots":
            for x in range(0, W + 160, 48):
                for y in range(0, H + 160, 48): d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=ac + (46,))
        elif s.bg == "rays":
            cx, cy = (W + 160) / 2, H * .38
            for i in range(14):
                a0 = i * math.pi / 7 + t * .05; a1 = a0 + .13
                d.polygon([(cx, cy), (cx + math.cos(a0) * 1800, cy + math.sin(a0) * 1800), (cx + math.cos(a1) * 1800, cy + math.sin(a1) * 1800)], fill=ac + (16,))
        elif s.bg == "waves":
            for j in range(7):
                pts = [(x, H * (.12 + j * .13) + 30 * math.sin(x / 120 + j + t * .6)) for x in range(0, W + 161, 20)]
                d.line(pts, fill=ac + (30,), width=3)
        else:   # stars
            for i in range(70):
                x, y = r.random() * (W + 160), r.random() * (H + 160); a = 40 + 90 * abs(math.sin(t * 1.2 + i)); rr = 1 + (i % 3) * .6
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(255, 255, 255, int(a)))
        im.alpha_composite(lay, (0, 0), (int(ox), int(oy), int(ox) + W, int(oy) + H))
        return im

CAMS = ["push", "pull", "pan_l", "pan_r", "rise", "fall", "tilt", "still"]
def apply_camera(img, cam, k, t=0.0, seed=0):
    """virtual camera on the finished scene + a very small handheld shake"""
    k = smooth(k); z, dx, dy, rot = 1.0, 0.0, 0.0, 0.0
    if cam == "push": z = lerp(1.0, 1.10, k)
    elif cam == "pull": z = lerp(1.12, 1.0, k)
    elif cam == "pan_l": z, dx = 1.08, lerp(18, -18, k)
    elif cam == "pan_r": z, dx = 1.08, lerp(-18, 18, k)
    elif cam == "rise": z, dy = 1.08, lerp(16, -16, k)
    elif cam == "fall": z, dy = 1.08, lerp(-16, 16, k)
    elif cam == "tilt": z, rot = 1.06, lerp(-2.2, 2.2, k)
    else: z = 1.0 + .012 * math.sin(k * 3.14)
    z = max(z, 1.02); dx += 2.0 * math.sin(t * 2.3 + seed); dy += 1.6 * math.sin(t * 1.7 + seed * 2); rot += .12 * math.sin(t * 1.3 + seed)
    cx, cy = W / 2, H / 2; a = math.radians(rot); ca, sa = math.cos(a) / z, math.sin(a) / z
    return img.transform((W, H), Image.Transform.AFFINE, (ca, -sa, cx - ca * cx + sa * cy - dx, sa, ca, cy - sa * cx - ca * cy - dy), resample=Image.Resampling.BICUBIC)

_VIG = {}
def cinematic(fr, t):
    """bloom on bright areas, vignette, light film grain, gentle contrast/saturation"""
    a = np.asarray(fr, np.float32)
    small = fr.resize((W // 4, H // 4), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.Resampling.BILINEAR)
    b = np.asarray(small, np.float32); a = a + np.clip((b - 135) / 120, 0, 1) * 85
    if (W, H) not in _VIG:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)); _VIG[(W, H)] = (1 - .34 * np.clip(r - .35, 0, 1.2) ** 1.6)[..., None]
    a = a * _VIG[(W, H)]
    lum = a.mean(axis=2, keepdims=True); a = lum + (a - lum) * 1.10; a = (a - 128) * 1.05 + 128
    a = a + np.random.default_rng(int(t * 30)).normal(0, 1.1, (H, W, 1)).astype(np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")

# ───────────── scene context handed to every component ─────────────
class Ctx:
    def __init__(s, ident, dur, seed, geo=None):
        s.id, s.dur, s.seed, s.geo = ident, dur, seed, geo
        s.box = BOX; s.cx = (BOX[0] + BOX[2]) / 2; s.cy = (BOX[1] + BOX[3]) / 2; s.bw = BOX[2] - BOX[0]; s.bh = BOX[3] - BOX[1]
        s.acc, s.acc2, s.cols = ident.acc, ident.acc2, ident.cols; s.rng = random.Random(seed)
        s.panel = (10, 18, 44, 218)


REG = {}
def comp(name, desc, example, defaults=None):
    def deco(fn): REG[name] = dict(fn=fn, desc=desc, example=example, custom=False); return fn
    return deco


# ───────────── component: big_number ─────────────
@comp("big_number", "one huge animated number with a label (the key number of the scene); pulsing rings behind it",
      '{"title":"POWER NEEDED","value":"50 billion","label":"solar panels","sub":"to power the whole planet"}')
def c_big_number(g, p, t, k, S):
    v = str(p.get("value", "")); label = str(p.get("label", "")); sub = str(p.get("sub", "")); n = num(v, None) if re.search(r"\d", v) else None
    a = eo3(seg(k, .02, .5)); cx, cy = S.cx, S.cy - 30
    for i in range(4):
        rr = 110 + i * 70 + 18 * math.sin(t * 2 + i); g.circle(cx, cy, rr * (.6 + .4 * eo3(seg(k, 0, .4))), outline=S.acc + (int(70 - i * 14),), w=3)
    g.glow(cx, cy, 300, S.acc, .30)
    if n is not None:
        cur = n * eo3(seg(k, .05, .6))
        txt = fmt(cur, 0 if n >= 100 or float(n).is_integer() else (len(v.split(".")[1].split()[0]) if "." in v and v.split(".")[1][:1].isdigit() else 1))
        mt = re.search(r"[a-zA-Z%×x]+\s*$", v.strip()); 
        if n >= 1e6 and re.search(r"(million|billion|trillion)", v.lower()): txt = f"{cur / (1e12 if 'trillion' in v.lower() else 1e9 if 'billion' in v.lower() else 1e6):.{1 if '.' in v else 0}f}"; txt += " " + re.search(r"(million|billion|trillion)", v.lower()).group(1)
        elif mt and not re.search(r"(million|billion|trillion)", v.lower()): txt += (" " if mt.group().isalpha() and len(mt.group()) > 2 else "") + mt.group()
    else: txt = v
    pre = str(p.get("prefix", "")); txt = pre + txt; size = 150
    while size > 40 and g.tw(txt, size) > S.bw * .96: size -= 6
    sc = .85 + .15 * eob(seg(k, 0, .35))
    g.text(cx, cy, txt, int(size * sc), BRAND + (int(255 * a),), stroke=7)
    g.text(cx, cy, txt, int(size * sc), (255, 255, 255, 0), stroke=0)
    if label: g.fit(cx, cy + size * .62 + 12, label.upper(), 44, S.bw * .94, (255, 255, 255, int(255 * eo3(seg(k, .25, .55)))), stroke=4)
    if sub: g.fit(cx, cy + size * .62 + 66, sub, 32, S.bw * .92, S.acc + (int(255 * eo3(seg(k, .4, .7))),), stroke=3)
    head(g, S, p, k)

# ───────────── component: size_compare ─────────────
@comp("size_compare", "objects drawn to true relative size side by side (planets, balls, buildings, animals as shaded shapes) so the viewer SEES how much bigger one is",
      '{"title":"SIZE COMPARISON","items":[{"label":"Earth","value":12742,"shape":"sphere","display":"12,742 km"},{"label":"Jupiter","value":139820,"shape":"sphere","display":"139,820 km"}]}')
def c_size_compare(g, p, t, k, S):
    items = [dict(i) for i in p["items"]][:4]; head(g, S, p, k)
    vals = [max(num(i.get("value"), 1), 1e-9) for i in items]; vmax = max(vals); vmin = min(vals)
    gap = 26; avail = S.bw - 20; maxd = min(S.bw * .62, S.bh * .5) if len(items) <= 2 else S.bw * .5
    order = sorted(range(len(items)), key=lambda i: vals[i])
    # sizes: diameter proportional to value; the largest is as big as fits
    ds = [max(10, maxd * vals[i] / vmax) for i in range(len(items))]
    ext = lambda i: ds[i] * (1.9 if items[i].get('rings') else 1.0)
    while sum(ext(i) for i in range(len(ds))) + gap * (len(ds) - 1) > avail: ds = [d * .93 for d in ds]
    if len(items) >= 2 and sum(ds) + gap * (len(ds) - 1) < avail * .6 and max(ds) < S.bh * .5: pass
    total = sum(ext(i) for i in range(len(ds))) + gap * (len(ds) - 1); x = S.cx - total / 2; cy = S.cy + 10
    for n, idx in enumerate(range(len(items))):
        it = items[idx]; d = ds[idx]; r = d / 2; ap = eob(seg(k, .05 + n * .13, .38 + n * .13)); col = S.cols[n % len(S.cols)]; xc = x + ext(idx) / 2
        shape = str(it.get("shape", "sphere")); y = cy
        if shape == "sphere":
            g.glow(xc, y, r * 1.5, col, .25)
            if it.get("rings"):
                rx, ry, rw = r * 1.9 * ap, r * .42 * ap, max(3, r * .09)
                g.ellipse(xc, y, rx, ry, outline=(235, 215, 170, 210), w=rw); g.sprite(sphere_sprite(r, col, bands=3), xc, y, ap)
                g.line([(xc + rx * math.cos(a / 20 * math.pi), y + ry * math.sin(a / 20 * math.pi)) for a in range(0, 21)], (235, 215, 170, 255), rw)
            else: g.sprite(sphere_sprite(r, col, bands=3 if r > 90 and n % 2 else 0), xc, y, ap)
        elif shape == "box":
            g.rect(xc - r * ap, y + r - 2 * r * ap, xc + r * ap, y + r, fill=col + (255,), outline=(255, 255, 255, 200), w=3, r=8)
        else:
            hh = d * 1.3 * ap; g.rect(xc - r * .5, y + r - hh, xc + r * .5, y + r, fill=col + (255,), outline=(255, 255, 255, 200), w=3, r=6)
        ly = cy + max(ds) / 2 + 40
        la = int(255 * eo3(seg(k, .25 + n * .13, .5 + n * .13)))
        lx = clamp(xc, S.box[0] + 70, S.box[2] - 70)
        g.fit(lx, ly, str(it.get("label", "")).upper(), 30, max(d, 150), (255, 255, 255, la), stroke=3)
        if it.get("display"): g.fit(lx, ly + 36, str(it["display"]), 28, max(d, 150), col + (la,), stroke=3)
        x += ext(idx) + gap
    if len(items) >= 2 and vmax / vmin >= 1.5:
        ra = eo3(seg(k, .55, .85)); big = vmax / vmin
        g.rect(S.cx - 130, S.box[1] + 52, S.cx + 130, S.box[1] + 106, fill=BRAND + (int(255 * ra),), r=26)
        g.text(S.cx, S.box[1] + 79, f"×{big:.0f}" if big >= 10 else f"×{big:.1f}", 40, (20, 14, 30, int(255 * ra)))

# ───────────── component: bars ─────────────
@comp("bars", "animated horizontal bar chart ranking 2-6 things with counting numbers; the leader is highlighted in the brand orange",
      '{"title":"LARGEST COUNTRIES","items":[{"label":"Russia","value":17.1},{"label":"Canada","value":10.0}],"unit":"million km²"}')
def c_bars(g, p, t, k, S):
    items = [dict(i) for i in p["items"]][:6]; head(g, S, p, k); unit = str(p.get("unit", ""))
    vals = [max(num(i.get("value"), 0), 0) for i in items]; vmax = max(vals) or 1
    top = sorted(range(len(items)), key=lambda i: -vals[i]); n = len(items); rowh = min(110, (S.bh - 90) / n); y0 = S.box[1] + 70; lab_w = 0
    for r_i, idx in enumerate(top):
        it = items[idx]; y = y0 + r_i * rowh + rowh / 2; ap = eo3(seg(k, .05 + r_i * .09, .55 + r_i * .09)); col = BRAND if r_i == 0 else S.cols[(r_i % 2)]
        x0 = S.box[0] + 6; maxw = S.bw - 12; bw = max(8, maxw * vals[idx] / vmax * ap)
        g.text(x0, y - rowh * .30, shorten(str(it.get("label", "")).upper(), 26), 26, (255, 255, 255, 255), anchor="lm", stroke=3)
        g.rect(x0, y - rowh * .12, x0 + bw, y + rowh * .30, fill=col + (255,), r=10)
        g.rect(x0, y - rowh * .12, x0 + bw, y + rowh * .02, fill=(255, 255, 255, 50), r=8)
        txt = fmt(vals[idx] * ap, 1 if vmax < 100 else 0) + (" " + unit if unit else "")
        tx = x0 + bw - 10
        if g.tw(txt, 26) + 24 > bw: g.text(x0 + bw + 10, y + rowh * .09, txt, 26, col + (255,), anchor="lm", stroke=3)
        else: g.text(tx, y + rowh * .09, txt, 26, (20, 14, 30, 255), anchor="rm")

# ───────────── component: line_chart ─────────────
@comp("line_chart", "a line that draws itself over time with a glowing end point; for growth, decline, history of a number",
      '{"title":"WORLD POPULATION","points":[[1900,1.6],[1950,2.5],[2000,6.1],[2024,8.1]],"unit":"billion","xlabel":"year"}')
def c_line_chart(g, p, t, k, S):
    pts = p["points"]; head(g, S, p, k); X = [num(a[0]) for a in pts]; Y = [num(a[1]) for a in pts]
    x0, x1, y0, y1 = S.box[0] + 60, S.box[2] - 20, S.box[1] + 120, S.box[3] - 90
    lo, hi = min(Y + [0]) if min(Y) >= 0 else min(Y), max(Y) * 1.08 or 1; xs = (min(X), max(X) if max(X) > min(X) else min(X) + 1)
    px = lambda v: x0 + (v - xs[0]) / (xs[1] - xs[0]) * (x1 - x0); py = lambda v: y1 - (v - lo) / (hi - lo or 1) * (y1 - y0)
    for i in range(5):
        yy = lerp(y0, y1, i / 4); g.line([(x0, yy), (x1, yy)], (255, 255, 255, 28), 1)
        g.text(x0 - 8, yy, fmt(lerp(hi, lo, i / 4), 1 if hi < 50 else 0), 18, DIM + (255,), anchor="rm")
    prog = eo3(seg(k, .08, .8)); path = [(px(a), py(b)) for a, b in zip(X, Y)]
    tot = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)) or 1; run = prog * tot; cur = []; acc = 0
    for i in range(len(path) - 1):
        d = math.dist(path[i], path[i + 1])
        if i == 0: cur.append(path[0])
        if acc + d <= run: cur.append(path[i + 1]); acc += d
        else: f = (run - acc) / d if d else 0; cur.append((lerp(path[i][0], path[i + 1][0], f), lerp(path[i][1], path[i + 1][1], f))); break
    if len(cur) >= 2:
        g.poly(cur + [(cur[-1][0], y1), (cur[0][0], y1)], fill=S.acc + (55,)); g.line(cur, S.acc + (255,), 7)
        ex, ey = cur[-1]; g.glow(ex, ey, 70, BRAND, .6); g.circle(ex, ey, 11, fill=BRAND + (255,), outline=(255, 255, 255, 255), w=3)
        val = lerp(Y[0], Y[-1], prog) if len(Y) > 1 else Y[0]
        lab = fmt(val) + (" " + str(p.get("unit", "")) if p.get("unit") else ""); g.text(min(ex, S.box[2] - g.tw(lab, 30) / 2 - 6), ey - 34, lab, 30, (255, 255, 255, 255), stroke=4)
    for i, (a, b) in enumerate(zip(X, Y)):
        if i in (0, len(X) - 1) or len(X) <= 5: g.text(px(a), y1 + 24, fmt(a, 0).replace(",", "") if abs(a) >= 1000 and a < 3000 else fmt(a), 20, DIM + (255,))

# ───────────── component: donut ─────────────
@comp("donut", "a ring that fills to a percentage with the % in the middle; for shares, proportions, 'x out of y'",
      '{"title":"EARTH IS WATER","value":71,"label":"of the surface","rest":"land 29%"}')
def c_donut(g, p, t, k, S):
    head(g, S, p, k); v = clamp(num(p.get("value"), 50), 0, 100); cur = v * eo3(seg(k, .05, .7)); R = min(S.bw * .36, S.bh * .34); cx, cy = S.cx, S.cy - 10
    g.glow(cx, cy, R * 1.7, S.acc, .22); g.circle(cx, cy, R, outline=(255, 255, 255, 40), w=34)
    g.arc(cx, cy, R, -90, -90 + 3.6 * cur, BRAND + (255,), 34)
    g.arc(cx, cy, R, -90, -90 + 3.6 * cur, (255, 255, 255, 70), 10)
    g.text(cx, cy - 6, f"{cur:.0f}%" if abs(v - round(v)) < .01 else f"{cur:.1f}%", int(R * .72), (255, 255, 255, 255), stroke=6)
    if p.get("label"): g.fit(cx, cy + R + 56, str(p["label"]).upper(), 38, S.bw * .92, (255, 255, 255, int(255 * eo3(seg(k, .3, .6)))), stroke=4)
    if p.get("rest"): g.fit(cx, cy + R + 104, str(p["rest"]), 28, S.bw * .9, S.acc + (int(255 * eo3(seg(k, .45, .75))),), stroke=3)

# ───────────── component: grid_fill ─────────────
@comp("grid_fill", "a grid of small icons (squares, people, houses) that fills up to show a part of a whole, 'X of every Y', or a huge count as units",
      '{"title":"1 IN 10","total":100,"filled":10,"icon":"person","label":"people affected","unit_note":"each icon = 1 million"}')
def c_grid_fill(g, p, t, k, S):
    head(g, S, p, k); total = int(clamp(num(p.get("total"), 100), 4, 400)); filled = int(clamp(num(p.get("filled"), total / 4), 0, total)); icon = str(p.get("icon", "square"))
    cols = int(math.ceil(math.sqrt(total * S.bw / (S.bh - 140)))); rows = int(math.ceil(total / cols)); cell = min((S.bw - 20) / cols, (S.bh - 150) / rows)
    gw, gh = cols * cell, rows * cell; x0, y0 = S.cx - gw / 2, S.box[1] + 70 + (S.bh - 150 - gh) / 2
    prog = eo3(seg(k, .08, .75)) * total
    for i in range(total):
        c, r = i % cols, i // cols; x, y = x0 + c * cell + cell / 2, y0 + r * cell + cell / 2; on = i < filled; shown = i < prog
        col = BRAND if on else (70, 90, 140); a = 255 if (shown and on) else (120 if shown else 40); sz = cell * .36 * (eob(seg(prog - i, 0, 3)) if shown else .6)
        if icon == "person": g.circle(x, y - sz * .55, sz * .42, fill=col + (a,)); g.rect(x - sz * .45, y - sz * .05, x + sz * .45, y + sz * .9, fill=col + (a,), r=sz * .3)
        elif icon == "house": g.poly([(x - sz, y), (x, y - sz * .9), (x + sz, y)], fill=shade(col, 1.15) + (a,)); g.rect(x - sz * .75, y, x + sz * .75, y + sz * .85, fill=col + (a,))
        else: g.rect(x - sz, y - sz, x + sz, y + sz, fill=col + (a,), r=max(2, sz * .25))
    if p.get("label"): g.fit(S.cx, S.box[3] - 36, str(p["label"]).upper() + (f" — {p['unit_note']}" if p.get("unit_note") else ""), 30, S.bw * .96, (255, 255, 255, int(255 * eo3(seg(k, .3, .6)))), stroke=4)

# ───────────── component: water_tank ─────────────
def _submerge(f):
    """depth (in radii*2 units 0..1) of a floating ball with volume fraction f below the surface"""
    f = clamp(f, 0, 1); lo, hi = 0.0, 2.0
    for _ in range(40):
        h = (lo + hi) / 2
        if h * h * (3 - h) / 4 < f: lo = h
        else: hi = h
    return (lo + hi) / 2 / 2

@comp("water_tank", "a tank of water where objects are dropped and FLOAT or SINK according to their density (physics is computed); perfect for density, buoyancy, 'would it float' facts",
      '{"title":"WOULD IT FLOAT?","liquid":"water","liquid_density":1.0,"objects":[{"label":"Saturn","density":0.69,"color":"gold"},{"label":"Earth","density":5.51}]}')
def c_water_tank(g, p, t, k, S):
    head(g, S, p, k); objs = [dict(o) for o in p["objects"]][:3]; ld = max(num(p.get("liquid_density"), 1.0), .01)
    tx0, tx1, ty0, ty1 = S.box[0] + 30, S.box[2] - 30, S.box[1] + 90, S.box[3] - 30; wl = ty0 + (ty1 - ty0) * .46
    g.rect(tx0, ty0, tx1, ty1, fill=(10, 18, 44, 120), outline=(255, 255, 255, 120), w=4, r=22)
    n = len(objs); slot = (tx1 - tx0) / n; r = min(slot * .38, 110)
    water = [(tx0 + 4 + i * 8, wl + 5 * math.sin(i * .6 + t * 3)) for i in range(int((tx1 - tx0 - 8) / 8) + 1)]
    for i, o in enumerate(objs):
        d = num(o.get("density"), 1.0); f = d / ld; xc = tx0 + slot * (i + .5); col = S.cols[i % len(S.cols)]
        if isinstance(o.get("color"), (list, tuple)) and len(o["color"]) == 3: col = tuple(int(c) for c in o["color"])
        drop = eo3(seg(k, .05 + i * .1, .45 + i * .1)); sink = eo3(seg(k, .35 + i * .1, .75 + i * .1)); bob = math.sin(t * 2.4 + i) * 4 * sink
        if f < 1: y_rest = wl - r + 2 * r * _submerge(f) + bob; kind = "float"
        else: y_rest = ty1 - r - 6; kind = "sink"
        y_top = ty0 - r - 40; y = lerp(y_top, wl - r * .4, drop) if sink <= 0 else lerp(wl - r * .4, y_rest, sink)
        g.glow(xc, y, r * 1.4, col, .2); g.sprite(sphere_sprite(r, col, bands=3 if i == 0 else 0), xc, y)
    g.poly(water + [(tx1 - 4, ty1 - 4), (tx0 + 4, ty1 - 4)], fill=(40, 130, 230, 120))
    g.line(water, (170, 225, 255, 230), 4)
    for i, o in enumerate(objs):
        d = num(o.get("density"), 1.0); xc = tx0 + slot * (i + .5); col = S.cols[i % len(S.cols)]; la = int(255 * eo3(seg(k, .5 + i * .1, .8 + i * .1)))
        g.fit(xc, ty0 + 30, str(o.get("label", "")).upper(), 30, slot - 10, (255, 255, 255, la), stroke=3)
        g.text(xc, ty0 + 66, "density " + f"{d:g}", 24, col + (la,), stroke=2)
        fl = d / ld < 1; g.text(xc, ty1 - 20, "FLOATS" if fl else "SINKS", 32, ((120, 255, 170) if fl else (255, 110, 110)) + (la,), stroke=4)
    g.text(tx1 - 12, wl - 16, f"{str(p.get('liquid', 'water')).upper()} = {ld:g}", 22, (170, 225, 255, 255), anchor="rm", stroke=2)


# ───────────── geography data (free, downloaded once and cached) ─────────────
GEO_URLS = {"countries": "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json",
            "states": "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"}
ALIASES = {"usa": "united states of america", "us": "united states of america", "united states": "united states of america", "america": "united states of america",
           "uk": "united kingdom", "britain": "united kingdom", "great britain": "united kingdom", "russia": "russia", "uae": "united arab emirates",
           "czechia": "czech republic", "drc": "democratic republic of the congo", "congo": "republic of the congo", "ivory coast": "ivory coast",
           "south korea": "south korea", "north korea": "north korea", "korea": "south korea", "holland": "netherlands", "burma": "myanmar", "eswatini": "swaziland"}

class Geo:
    def __init__(s, cache_dir):
        s.cache, s.items, s.ok = cache_dir, {}, False
        os.makedirs(cache_dir, exist_ok=True)
        for kind, url in GEO_URLS.items():
            path = os.path.join(cache_dir, kind + ".json"); data = None
            try:
                if os.path.exists(path) and os.path.getsize(path) > 1000: data = json.load(open(path, encoding="utf-8"))
                else:
                    import requests
                    r = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
                    if r.ok and len(r.content) > 1000: data = r.json(); open(path, "w", encoding="utf-8").write(r.text)
            except Exception as e: _log("geo", kind, "unavailable:", str(e)[:100])
            if data: s._load(data, kind)
        s.ok = bool(s.items)
    def _load(s, data, kind):
        for f in data.get("features", []):
            nm = str((f.get("properties") or {}).get("name", "")).strip().lower(); geom = f.get("geometry") or {}
            if not nm: continue
            polys = geom.get("coordinates", []) if geom.get("type") == "MultiPolygon" else ([geom.get("coordinates")] if geom.get("type") == "Polygon" else [])
            rings = [[(float(x), float(y)) for x, y in poly[0]] for poly in polys if poly and poly[0]]
            if rings: s.items.setdefault(nm, dict(rings=rings, kind=kind))
    def find(s, name):
        n = str(name).strip().lower(); n = ALIASES.get(n, n)
        if n in s.items: return s.items[n]
        for k2, v in s.items.items():
            if n and (n == k2 or k2.startswith(n)): return v
        return None
    def names(s, kind=None, n=60): return [k for k, v in s.items.items() if kind in (None, v["kind"])][:n]





# ───────────── component: map_compare ─────────────


# ───────────── component: globe ─────────────

# ───────────── component: orbit ─────────────

# ───────────── component: iso_scene ─────────────



# ───────────── component: timeline ─────────────

# ───────────── component: flow ─────────────

# ───────────── component: layers ─────────────

# ───────────── component: person_scale ─────────────

# ───────────── component: versus ─────────────

# ───────────── component: kinetic_text ─────────────

# ───────────── geography data (free, downloaded once and cached) ─────────────
GEO_URLS = {"countries": "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json",
            "states": "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"}
ALIASES = {"usa": "united states of america", "us": "united states of america", "united states": "united states of america", "america": "united states of america",
           "uk": "united kingdom", "britain": "united kingdom", "great britain": "united kingdom", "russia": "russia", "uae": "united arab emirates",
           "czechia": "czech republic", "drc": "democratic republic of the congo", "congo": "republic of the congo", "ivory coast": "ivory coast",
           "south korea": "south korea", "north korea": "north korea", "korea": "south korea", "holland": "netherlands", "burma": "myanmar", "eswatini": "swaziland"}

class Geo:
    def __init__(s, cache_dir):
        s.cache, s.items, s.ok = cache_dir, {}, False
        os.makedirs(cache_dir, exist_ok=True)
        for kind, url in GEO_URLS.items():
            path = os.path.join(cache_dir, kind + ".json"); data = None
            try:
                if os.path.exists(path) and os.path.getsize(path) > 1000: data = json.load(open(path, encoding="utf-8"))
                else:
                    import requests
                    r = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
                    if r.ok and len(r.content) > 1000: data = r.json(); open(path, "w", encoding="utf-8").write(r.text)
            except Exception as e: _log("geo", kind, "unavailable:", str(e)[:100])
            if data: s._load(data, kind)
        s.ok = bool(s.items)
    def _load(s, data, kind):
        for f in data.get("features", []):
            nm = str((f.get("properties") or {}).get("name", "")).strip().lower(); geom = f.get("geometry") or {}
            if not nm: continue
            polys = geom.get("coordinates", []) if geom.get("type") == "MultiPolygon" else ([geom.get("coordinates")] if geom.get("type") == "Polygon" else [])
            rings = [[(float(x), float(y)) for x, y in poly[0]] for poly in polys if poly and poly[0]]
            if rings: s.items.setdefault(nm, dict(rings=rings, kind=kind))
    def find(s, name):
        n = str(name).strip().lower(); n = ALIASES.get(n, n)
        if n in s.items: return s.items[n]
        for k2, v in s.items.items():
            if n and (n == k2 or k2.startswith(n)): return v
        return None
    def names(s, kind=None, n=60): return [k for k, v in s.items.items() if kind in (None, v["kind"])][:n]

def ring_area_km2(ring):
    if len(ring) < 3: return 0
    lat0 = sum(p[1] for p in ring) / len(ring); kx = 111.32 * math.cos(math.radians(lat0)); ky = 110.57
    a = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i][0] * kx, ring[i][1] * ky; x2, y2 = ring[(i + 1) % len(ring)][0] * kx, ring[(i + 1) % len(ring)][1] * ky; a += x1 * y2 - x2 * y1
    return abs(a) / 2

def area_of(item): return sum(ring_area_km2(r) for r in item["rings"])

def centroid(item):
    best = max(item["rings"], key=ring_area_km2); return sum(p[0] for p in best) / len(best), sum(p[1] for p in best) / len(best)

def project_item(item, lon0, lat0, scale, dx=0.0, dy=0.0, only_main=False):
    kx = math.cos(math.radians(lat0)); out = []
    rings = [max(item["rings"], key=ring_area_km2)] if only_main else item["rings"]
    for r in rings: out.append([((x - lon0) * kx * scale + dx, -(y - lat0) * scale + dy) for x, y in r])
    return out

# ───────────── component: map_compare ─────────────

def fix_map_compare(p, S):
    if S.geo is None or not S.geo.ok: return None
    try:
        a, b = p.get("a") or {}, p.get("b") or {}
        if isinstance(a, str): a = {"name": a}
        if isinstance(b, str): b = {"name": b}
        if not (S.geo.find(a.get("name", "")) and S.geo.find(b.get("name", ""))): return None
        return dict(p, a=a, b=b)
    except Exception: return None

# ───────────── component: globe ─────────────
@comp("globe", "a rotating Earth globe with real continents and pulsing markers on places (lat/lon); for 'where in the world', distances, global scale",
      '{"title":"WHERE IT HAPPENS","markers":[{"label":"Sahara","lat":23,"lon":13},{"label":"Tokyo","lat":35.7,"lon":139.7}],"focus_lon":20,"focus_lat":15}')
def c_globe(g, p, t, k, S):
    space_bg(g, S, t, 0); R = S.bw * .5; cx, cy = S.cx, S.cy + 30
    flon, flat = num(p.get("focus_lon"), 10), num(p.get("focus_lat"), 15); lon0 = flon - 60 + 60 * eo3(seg(k, 0, .7)) + t * 4; lat0 = math.radians(clamp(flat, -50, 50) * .6)
    g.glow(cx, cy, R * 1.45, S.acc, .35); g.sprite(sphere_sprite(R, (24, 60, 130), (-.45, -.5)), cx, cy)
    def pj(lon, lat):
        lo, la = math.radians(lon - lon0), math.radians(lat); x = math.cos(la) * math.sin(lo); y = math.cos(lat0) * math.sin(la) - math.sin(lat0) * math.cos(la) * math.cos(lo)
        z = math.sin(lat0) * math.sin(la) + math.cos(lat0) * math.cos(la) * math.cos(lo); return x, y, z
    if S.geo is not None and S.geo.ok:
        for nm, it in S.geo.items.items():
            if it["kind"] != "countries": continue
            for ring in it["rings"]:
                pts, vis = [], 0
                for lon, lat in ring[::2 if len(ring) > 60 else 1]:
                    x, y, z = pj(lon, lat)
                    if z < 0: n = math.hypot(x, y) or 1; x, y = x / n, y / n
                    else: vis += 1
                    pts.append((cx + x * R, cy - y * R))
                if vis > 0 and len(pts) >= 3: g.poly(pts, fill=(60, 150, 105, 235), outline=(150, 230, 180, 150), w=1)
    for la in range(-60, 61, 30):
        pts = [(cx + pj(lo, la)[0] * R, cy - pj(lo, la)[1] * R) for lo in range(int(lon0) - 90, int(lon0) + 91, 6) if pj(lo, la)[2] > 0]
        g.line(pts, (255, 255, 255, 28), 1)
    g.circle(cx, cy, R, outline=S.acc + (200,), w=4)
    for i, m in enumerate(p.get("markers", [])[:4]):
        x, y, z = pj(num(m.get("lon")), num(m.get("lat")))
        if z > .05:
            mx, my = cx + x * R, cy - y * R; pu = (t * 1.6 + i * .4) % 1; ap = eob(seg(k, .35 + i * .1, .6 + i * .1))
            g.circle(mx, my, 10 + 34 * pu, outline=BRAND + (int(220 * (1 - pu) * ap),), w=4); g.circle(mx, my, 10 * ap, fill=BRAND + (255,), outline=(255, 255, 255, 255), w=3)
            lab = str(m.get("label", "")).upper()
            if lab: tag(g, mx, my - 46, lab, 30, (255, 255, 255), 255 * ap)

# ───────────── component: orbit ─────────────
@comp("orbit", "planets/moons circling a glowing centre on tilted orbits with labels; for solar system, years, distances, 'how long to go around'",
      '{"title":"INNER PLANETS","center":"Sun","bodies":[{"label":"Mercury","orbit":0.25,"size":0.3,"period":0.24},{"label":"Earth","orbit":0.6,"size":0.5,"period":1}]}')
def c_orbit(g, p, t, k, S):
    head(g, S, p, k); bodies = [dict(b) for b in p["bodies"]][:5]; cx, cy = S.cx, S.cy + 20; rx0 = S.bw * .46; tilt = .42
    g.glow(cx, cy, 190, (255, 190, 80), .75); g.sprite(sphere_sprite(46 * eob(seg(k, 0, .3)), (255, 200, 90), (-.1, -.1)), cx, cy)
    if p.get("center"): g.text(cx, cy + 66, str(p["center"]).upper(), 22, (255, 230, 170, 255), stroke=3)
    mo = max(num(b.get("orbit"), 1) for b in bodies) or 1
    for i, b in enumerate(sorted(bodies, key=lambda b: num(b.get("orbit"), 1))):
        rr = rx0 * (.22 + .78 * num(b.get("orbit"), 1) / mo) * eo3(seg(k, .05 + i * .06, .4 + i * .06)); per = max(num(b.get("period"), 1), .05)
        ang = (t * 1.1 / (per ** .55) + i * 1.7); col = S.cols[(i + 1) % len(S.cols)]
        g.ellipse(cx, cy, rr, rr * tilt, outline=(255, 255, 255, 46), w=2); x, y = cx + rr * math.cos(ang), cy + rr * tilt * math.sin(ang); sz = 9 + 22 * clamp(num(b.get("size"), .5) / 1.2, .15, 1)
        g.glow(x, y, sz * 2.4, col, .3); g.sprite(sphere_sprite(sz, col), x, y)
        g.text(x, y - sz - 16, str(b.get("label", "")).upper(), 22, (255, 255, 255, int(255 * eo3(seg(k, .4, .7)))), stroke=3)

# ───────────── component: iso_scene ─────────────
def _iso(x, y, z, ang, sc, ox, oy):
    ca, sa = math.cos(ang), math.sin(ang); rx, ry = x * ca - y * sa, x * sa + y * ca
    return ox + (rx - ry) * sc, oy + (rx + ry) * sc * .5 - z * sc, rx + ry

def _cuboid(g, x, y, z0, w, d, h, ang, sc, ox, oy, col, alpha=255):
    pts = lambda *xyz: [_iso(a, b, c, ang, sc, ox, oy)[:2] for a, b, c in xyz]
    top = pts((x, y, z0 + h), (x + w, y, z0 + h), (x + w, y + d, z0 + h), (x, y + d, z0 + h))
    ca = math.cos(ang); sa = math.sin(ang)
    # faces facing the viewer: choose the two lower-front sides in screen space
    f1 = pts((x, y + d, z0), (x + w, y + d, z0), (x + w, y + d, z0 + h), (x, y + d, z0 + h))
    f2 = pts((x + w, y, z0), (x + w, y + d, z0), (x + w, y + d, z0 + h), (x + w, y, z0 + h))
    f3 = pts((x, y, z0), (x + w, y, z0), (x + w, y, z0 + h), (x, y, z0 + h)); f4 = pts((x, y, z0), (x, y + d, z0), (x, y + d, z0 + h), (x, y, z0 + h))
    faces = [(f1, shade(col, .72)), (f2, shade(col, .55)), (f3, shade(col, .72)), (f4, shade(col, .55))]
    cen = lambda f: sum(q[1] for q in f) / 4
    for f, c in sorted(faces, key=lambda fc: cen(fc[0]))[2:]: g.poly(f, fill=c + (alpha,))
    g.poly(top, fill=shade(col, 1.12) + (alpha,), outline=shade(col, 1.3) + (alpha,), w=1)

@comp("iso_scene", "a 3D-looking isometric miniature that slowly rotates: a district of houses, a skyline of towers, a field of solar panels or farm plots; a highlighted group shows 'this many'",
      '{"title":"1 MILLION HOMES","kind":"houses","count":36,"highlight":12,"label":"homes powered by 20 panels each"}')
def c_iso_scene(g, p, t, k, S):
    head(g, S, p, k); kind = str(p.get("kind", "houses")); n = int(clamp(num(p.get("count"), 25), 4, 100)); hl = int(clamp(num(p.get("highlight"), n // 3), 0, n))
    cols = int(math.ceil(math.sqrt(n))); rows = int(math.ceil(n / cols)); ang = math.radians(32) + t * .10 + (S.seed % 5) * .03
    sc = min(S.bw / ((cols + rows) * 1.25), S.bh / ((cols + rows) * .9)) * 1.0; ox, oy = S.cx, S.cy + S.bh * .22; pitch = 1.5; items = []
    r = random.Random(S.seed)
    for i in range(n):
        gx, gy = (i % cols) - cols / 2, (i // cols) - rows / 2; items.append((i, gx * pitch, gy * pitch, _iso(gx * pitch, gy * pitch, 0, ang, sc, ox, oy)[2]))
    g.poly([_iso(-cols / 2 * pitch - .6, -rows / 2 * pitch - .6, 0, ang, sc, ox, oy)[:2], _iso(cols / 2 * pitch, -rows / 2 * pitch - .6, 0, ang, sc, ox, oy)[:2], _iso(cols / 2 * pitch, rows / 2 * pitch, 0, ang, sc, ox, oy)[:2], _iso(-cols / 2 * pitch - .6, rows / 2 * pitch, 0, ang, sc, ox, oy)[:2]], fill=(20, 60, 50, 200) if kind in ("farm", "solar") else (30, 40, 70, 210), outline=(255, 255, 255, 40), w=2)
    for i, x, y, dep in sorted(items, key=lambda it: it[3]):
        grow = eob(seg(k, .05 + (i / max(n, 1)) * .45, .30 + (i / max(n, 1)) * .45)); on = i < hl; col = BRAND if on else ((70, 130, 200) if kind != "farm" else (90, 170, 90))
        if kind == "towers": h = (1.6 + (i * 7 % 5) * .55) * grow
        elif kind == "solar": h = .16 * grow; col = (255, 190, 60) if on else (40, 70, 140)
        elif kind == "farm": h = .1 * grow; col = (BRAND if on else (80 + (i * 37) % 60, 150, 70))
        else: h = .85 * grow
        wd = .62 if kind == "towers" else (1.2 if kind in ("solar", "farm") else .95)
        _cuboid(g, x, y, 0, wd, wd, max(h, .01), ang, sc, ox, oy, col)
        if kind == "houses" and grow > .3:
            tp = [_iso(x + wd / 2, y + wd / 2, (.85 + .5) * grow, ang, sc, ox, oy)[:2]]; base = [_iso(x - .05, y - .05, .85 * grow, ang, sc, ox, oy)[:2], _iso(x + wd + .05, y - .05, .85 * grow, ang, sc, ox, oy)[:2], _iso(x + wd + .05, y + wd + .05, .85 * grow, ang, sc, ox, oy)[:2], _iso(x - .05, y + wd + .05, .85 * grow, ang, sc, ox, oy)[:2]]
            for j in range(4): g.poly([base[j], base[(j + 1) % 4], tp[0]], fill=shade((230, 100, 70) if not on else (255, 220, 120), .75 + .12 * j), outline=None)
    if p.get("label"): g.fit(S.cx, S.box[3] - 20, str(p["label"]).upper(), 30, S.bw * .96, (255, 255, 255, int(255 * eo3(seg(k, .35, .65)))), stroke=4)

# ───────────── component: timeline ─────────────

# ───────────── component: flow ─────────────

# ───────────── component: layers ─────────────
@comp("layers", "a cut-through of stacked layers with a probe descending (Earth interior, atmosphere, ocean depth, skyscraper floors); each layer labelled with size",
      '{"title":"INSIDE EARTH","layers":[{"label":"Crust","size":"30 km","weight":1},{"label":"Mantle","size":"2,900 km","weight":4},{"label":"Core","size":"3,400 km","weight":5}]}')
def c_layers(g, p, t, k, S):
    head(g, S, p, k); ls = [dict(l) for l in p["layers"]][:5]; tot = sum(max(num(l.get("weight"), 1), .4) for l in ls); x0, x1 = S.box[0] + 150, S.box[2] - 20; y0, y1 = S.box[1] + 84, S.box[3] - 20; y = y0
    prog = eo3(seg(k, .05, .8)) * (y1 - y0); pal = [(120, 200, 255), (255, 190, 90), (255, 120, 80), (190, 90, 255), (90, 220, 170)]
    for i, l in enumerate(ls):
        h = (y1 - y0) * max(num(l.get("weight"), 1), .4) / tot; vis = clamp(prog - (y - y0), 0, h); col = pal[i % 5]
        if vis > 1:
            g.rect(x0, y, x1, y + vis, fill=col + (235,), outline=(255, 255, 255, 150), w=2)
            if vis > h * .6:
                g.fit((x0 + x1) / 2, y + h / 2 - 12, str(l.get("label", "")).upper(), 34, x1 - x0 - 20, (20, 14, 30, 255)); g.fit((x0 + x1) / 2, y + h / 2 + 22, str(l.get("size", "")), 26, x1 - x0 - 20, (20, 14, 30, 230))
        y += h
    py = y0 + prog; g.poly([(x0 - 14, py), (x0 - 70, py - 20), (x0 - 70, py + 20)], fill=BRAND + (255,)); g.glow(x0 - 40, py, 60, BRAND, .5)

# ───────────── component: person_scale ─────────────
@comp("person_scale", "a human figure stacked again and again next to a tall object to show how many people tall it is (towers, mountains, whales, trees)",
      '{"title":"HOW TALL?","label":"Burj Khalifa","height_m":828,"person_m":1.75}')
def c_person_scale(g, p, t, k, S):
    head(g, S, p, k); hm = max(num(p.get("height_m"), 100), 1); pm = max(num(p.get("person_m"), 1.75), .3); n = hm / pm; base = S.box[3] - 70; top = S.box[1] + 100; H_ = base - top
    ox, ow = S.cx + 40, 150; rise = eo3(seg(k, .05, .55)); hh = H_ * rise
    g.poly([(ox - ow / 2, base), (ox + ow / 2, base), (ox + ow * .18, base - hh), (ox - ow * .18, base - hh)], fill=(70, 120, 200, 235), outline=(190, 220, 255, 255), w=3)
    for j in range(int(hh / 38)): g.line([(ox - lerp(ow / 2, ow * .18, j * 38 / max(H_, 1)) + 8, base - 20 - j * 38), (ox + lerp(ow / 2, ow * .18, j * 38 / max(H_, 1)) - 8, base - 20 - j * 38)], (255, 255, 255, 40), 2)
    g.line([(ox, base - hh), (ox, base - hh - 36 * rise)], (255, 255, 255, 230), 4)
    ph = max(H_ / n, 6.0); stack = int(min(n, 400) * eo3(seg(k, .35, .85))); per_col = max(1, int(H_ / max(ph, 6)))
    if n <= 40:
        for i in range(stack): xx = S.cx - 190; yy = base - (i + 1) * ph; sz = ph * .5; g.circle(xx, yy + ph * .22, sz * .55, fill=BRAND + (255,)); g.rect(xx - sz * .5, yy + ph * .45, xx + sz * .5, yy + ph, fill=BRAND + (255,), r=sz * .2)
    else:
        cols_ = int(math.ceil(n / 60)); 
        for i in range(stack):
            c, r_ = i // 60, i % 60; xx = S.cx - 240 + c * 14; yy = base - (r_ + 1) * (H_ / 60); g.circle(xx, yy + 3, 4, fill=BRAND + (255,)); g.rect(xx - 3, yy + 6, xx + 3, yy + H_ / 60, fill=BRAND + (255,))
    g.text(S.cx - 190 if n <= 40 else S.cx - 190, top - 30, f"{int(n):,} PEOPLE", 36, BRAND + (int(255 * eo3(seg(k, .6, .9))),), stroke=4)
    g.fit(ox, base + 30, f"{str(p.get('label', '')).upper()} · {fmt(hm)} m", 28, S.bw * .6, (255, 255, 255, 255), stroke=3)

# ───────────── component: versus ─────────────

# ───────────── component: kinetic_text ─────────────
@comp("kinetic_text", "big kinetic typography: 1-3 short lines whose words slam in, one word highlighted; for hooks, twists and the closing punchline when no data visual fits",
      '{"lines":["WHAT IF","SATURN","HIT THE OCEAN?"],"accent":1}')
def c_kinetic_text(g, p, t, k, S):
    lines = [str(l).upper() for l in p["lines"]][:3]; ac = int(num(p.get("accent"), len(lines) - 1)); n = len(lines)
    for i in range(6):
        a = (t * .35 + i / 6) % 1; g.circle(S.cx, S.cy, 60 + a * 420, outline=S.acc + (int(60 * (1 - a)),), w=3)
    g.glow(S.cx, S.cy, 320, BRAND, .25)
    for i, l in enumerate(lines):
        ap = eob(seg(k, .03 + i * .12, .28 + i * .12)); size = 120
        while size > 36 and g.tw(l, size) > S.bw * .82: size -= 4
        y = S.cy + (i - (n - 1) / 2) * 140 + (1 - ap) * 80; col = BRAND if i == ac else (255, 255, 255)
        g.text(S.cx, y, l, int(size * (.8 + .2 * ap)), col + (int(255 * clamp(ap)),), stroke=8)


# ───────────── custom components: the LLM may write a NEW one when nothing in the library fits ─────────────
CUSTOM_DIR = os.environ.get("GFX_CUSTOM_DIR", "components")
_SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in
                  ("range", "len", "min", "max", "abs", "int", "float", "str", "sum", "enumerate", "zip", "round", "sorted", "list", "tuple", "dict", "bool", "isinstance", "reversed", "any", "all", "map", "pow")}
_BANNED = {"eval", "exec", "open", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input", "exit", "quit", "breakpoint", "type", "object", "super", "help", "dir", "os", "sys", "subprocess"}

def check_code(code):
    tree = ast.parse(code); fn = False
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.While, ast.AsyncFunctionDef, ast.ClassDef, ast.Await)): raise ValueError("forbidden syntax: " + type(n).__name__)
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"): raise ValueError("forbidden attribute " + n.attr)
        if isinstance(n, ast.Name) and (n.id in _BANNED or n.id.startswith("__")): raise ValueError("forbidden name " + n.id)
        if isinstance(n, ast.FunctionDef) and n.name == "draw": fn = True
    if not fn: raise ValueError("no draw(g, p, t, k, S) function")
    return True

def _timed(fn, secs=1.5):
    def alarm(*a): raise TimeoutError("component too slow")
    try:
        signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, secs)
        try: return fn()
        finally: signal.setitimer(signal.ITIMER_REAL, 0)
    except ValueError: return fn()      # not the main thread: no timer available

def _compile_custom(name, code):
    check_code(code)
    env = {"__builtins__": _SAFE_BUILTINS, "math": math, "clamp": clamp, "lerp": lerp, "eo3": eo3, "smooth": smooth, "eob": eob, "seg": seg, "mixc": mixc, "shade": shade, "BRAND": BRAND, "INK": INK, "DIM": DIM, "num": num, "fmt": fmt, "head": head, "label_card": label_card}
    exec(compile(code, f"<custom {name}>", "exec"), env)
    return env["draw"]

def _register_custom(name, desc, example, code):
    draw = _compile_custom(name, code)
    REG[name] = dict(fn=lambda g, p, t, k, S, d=draw: _timed(lambda: d(g, p, t, k, S)), desc=desc, example=example, custom=True, code=code)

def load_custom():
    try:
        if not os.path.isdir(CUSTOM_DIR): return
        for fn in sorted(os.listdir(CUSTOM_DIR)):
            if fn.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(CUSTOM_DIR, fn), encoding="utf-8")); _register_custom(d["name"], d["desc"], d.get("example", "{}"), d["code"])
                except Exception as e: _log("custom component skipped:", fn, str(e)[:100])
    except Exception as e: _log("custom components:", e)

CUSTOM_PROMPT = """You write ONE new drawing component for a Python motion-graphics engine (vertical 720x1280 video, dark navy background, orange brand accent).
Need: {need}
Write exactly one function: def draw(g, p, t, k, S):
- g draws (logical px, content must stay inside S.box = (x0,y0,x1,y1)): g.rect(x0,y0,x1,y1,fill=None,outline=None,w=2,r=0) g.circle(x,y,r,fill,outline,w) g.ellipse(x,y,rx,ry,fill,outline,w) g.poly(points,fill,outline,w) g.line(points,fill,w) g.arc(x,y,r,a0,a1,fill,w) g.text(x,y,str,size,fill,anchor='mm',stroke=0) g.fit(x,y,str,size,maxw,fill) g.glow(x,y,r,color,alpha). Colours are RGB or RGBA tuples.
- t = seconds into the scene, k = progress 0..1 (finish the animation by k=0.7, then hold). p = dict of parameters (use p.get with defaults). S.cx,S.cy,S.bw,S.bh = centre and size of the box; S.acc,S.acc2 = accent colours; S.cols = colour list.
- Available helpers (no imports allowed): math, clamp, lerp, eo3, smooth, eob, seg(k,a,b), mixc, shade, BRAND, INK, DIM, num, fmt, head(g,S,p,k) (title chip), label_card.
- No while loops, no imports, no classes, no names starting with underscore. Use for-loops with range(). Keep it fast (< 300 shapes).
- Always call head(g, S, p, k) first so the title chip shows, and animate with k so the drawing builds up.
Return JSON only: {{"name":"snake_case_name","desc":"what it shows and when to use it","example":"{{\\"title\\":\\"...\\"}}","code":"def draw(g, p, t, k, S):\\n    ..."}}"""

def make_custom(need, llm, ident):
    """ask the LLM for a new component, validate it by really rendering it, save it to the library"""
    if llm is None: return None
    try:
        d = llm(CUSTOM_PROMPT.format(need=need)); name = re.sub(r"[^a-z0-9_]", "", str(d["name"]).lower())[:30] or "custom_x"
        if name in REG: name += "_" + hashlib.md5(need.encode()).hexdigest()[:4]
        _register_custom(name, str(d["desc"])[:200], str(d.get("example", "{}")), d["code"])
        ex = json.loads(d.get("example") or "{}") if isinstance(d.get("example"), str) else (d.get("example") or {})
        S = Ctx(ident, 3.0, 1, None)
        for kk in (.2, .6, 1.0):
            gg = G(ident.frame(1.0)); REG[name]["fn"](gg, ex, 1.0, kk, S)
        os.makedirs(CUSTOM_DIR, exist_ok=True)
        json.dump(dict(name=name, desc=REG[name]["desc"], example=REG[name]["example"], code=d["code"]), open(os.path.join(CUSTOM_DIR, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        _log("new component created and saved:", name); return name
    except Exception as e:
        _log("custom component failed:", str(e)[:160]); REG.pop(locals().get("name", ""), None); return None

# ───────────── formats = what kind of visual video a given day is (the channel look stays the same) ─────────────
FORMATS = {
    "scale":     dict(desc="SCALE: things shown at true relative size in full-screen 3D worlds: planets side by side, people stacked beside a tower, bars rising from the floor, a district seen from above",
                      prefer=["planets3d", "tower3d", "bars3d", "city3d", "big_number", "kinetic_text"]),
    "physics":   dict(desc="PHYSICS & SPACE: a 3D simulation that shows WHY (a planet floating or sinking in an ocean, orbits, spinning worlds, scale of space)",
                      prefer=["sea3d", "planets3d", "orbit", "globe", "big_number", "kinetic_text"]),
    "numbers":   dict(desc="NUMBERS: how many panels/homes/people are needed, shown as a 3D city or solar farm with a flying camera, bars rising, a counter",
                      prefer=["city3d", "bars3d", "big_number", "globe", "map_compare", "kinetic_text"]),
    "mechanism": dict(desc="MECHANISM: how something works or scales, shown in 3D worlds, orbits and simulations with camera moves",
                      prefer=["city3d", "sea3d", "orbit", "tower3d", "planets3d", "kinetic_text"]),
    "maps":      dict(desc="MAPS: geography comparisons: true-size country/state shapes laid over each other on relief terrain, globe with markers, rankings as 3D bars",
                      prefer=["map_compare", "globe", "bars3d", "planets3d", "big_number", "kinetic_text"], topic=True),
}
def format_for(weekday, theme):
    return {0: "scale", 1: "physics", 2: "numbers", 3: "mechanism", 4: "numbers", 5: "maps", 6: "physics"}.get(weekday, "scale")

# ───────────── validation: a component is only used if it really renders with the given parameters ─────────────
def validate(name, params, S, ident):
    try:
        if name not in REG or not isinstance(params, dict): return False
        for kk in (.35, .95):
            gg = G(ident.frame(1.0)); REG[name]["fn"](gg, params, 1.2, kk, S)
        return True
    except Exception as e:
        _log(f"component {name} rejected: {type(e).__name__} {str(e)[:80]}"); return False

# ───────────── planning ─────────────
HIDDEN = {"size_compare", "bars", "line_chart", "donut", "grid_fill", "water_tank", "timeline", "flow", "layers", "person_scale", "versus"}   # flat UI cards: emergency fallbacks only

def catalogue(geo_ok):
    out = []
    for n, d in REG.items():
        if n in HIDDEN or (n == "map_compare" and not geo_ok): continue
        out.append(f'- {n}{" (custom)" if d.get("custom") else ""}: {d["desc"]}. params example: {d["example"]}')
    return "\n".join(out)

PLAN_PROMPT = """You are the visual director of a YouTube Shorts science channel. The scenes are drawn by code from a component library (no photos, no stock footage).
For EACH narration line choose the component that shows its meaning best and fill its params with REAL facts.
Today's video format: {fmt}
Narration, in order (with approximate seconds):
{scenes}

Component library:
{cat}
{custom_line}
Rules:
- Every visual is a FULL-SCREEN 3D world or cinematic scene with a moving camera; never ask for cards, panels, tables or boxes. Make it look like a premium explainer, not a slide.
- Decide from the MEANING of each line, never from a fixed order. The hook scene must be the most striking visual.
- Never use the same component in two consecutive scenes; use each component at most twice; the text-only components (kinetic_text, big_number) at most ONCE each (hook or closing line) - every other scene must be a real picture: 3D world, map, planets, ocean, bars, tower, globe. Use at least {minvar} different components.
- Fill params ONLY with numbers and names stated in the narration or well-known verified facts; never invent data. Keep labels short (max 3 words). Numbers in params may be plain numbers.
- map_compare / globe need real English country or US-state names / real lat-lon. water_tank densities are in g/cm3 (water = 1).
- "camera" is one of: push, pull, pan_l, pan_r, rise, fall, tilt, still. Never the same camera twice in a row.
- "idea" is one short sentence describing what the viewer sees.
{recent}
Return JSON only: {{"scenes":[{{"i":1,"component":"name","params":{{...}},"camera":"push","idea":"..."}}, ...]}}  (one entry per narration line, i starting at 1)"""

def _lines(text, n=3):
    w = text.upper().replace(".", "").replace(",", "").split(); k = max(1, math.ceil(len(w) / n)); return [" ".join(w[i:i + k]) for i in range(0, len(w), k)][:n]

def heuristic_all(text, i, n, S, geo, fmt_name):
    """rule-based candidates (all valid ones) when the LLM plan is missing or rejected"""
    low = text.lower(); nums = re.findall(r"\d[\d,.]*(?:\s*(?:million|billion|trillion|percent|thousand))?", low)
    cand = []
    if geo is not None and getattr(geo, "ok", False):
        names = sorted({n_ for n_ in geo.items if len(n_) > 3 and re.search(r"\b" + re.escape(n_) + r"\b", low)}, key=low.index)
        if len(names) >= 2: cand.append(("map_compare", dict(title=f"{names[0]} vs {names[1]}".upper(), a=dict(name=names[0]), b=dict(name=names[1]))))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)", low)
    if m: cand.append(("donut", dict(title="THE SHARE", value=float(m.group(1)), label=" ".join(text.split()[:3]))))
    yrs = re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", low)
    if len(yrs) >= 2: cand.append(("timeline", dict(title="TIMELINE", events=[dict(year=y, label="") for y in yrs[:3]])))
    if nums and i != 0: cand.append(("big_number", dict(title="THE NUMBER", value=nums[0].strip(), label=" ".join(text.split()[-3:]).strip(".,"))))
    parts = [x.strip(" .,") for x in re.split(r",| and | so | then | which | because | but ", text) if len(x.strip(" .,")) > 3]
    if len(parts) >= 2 and i != 0: cand.append(("flow", dict(title="STEP BY STEP", steps=[x[:1].upper() + x[1:] for x in parts[:3]])))
    cand.append(("kinetic_text", dict(lines=_lines(text), accent=1)))
    return [(nm, p) for nm, p in cand if validate(nm, p, S, S.id)] or [("kinetic_text", dict(lines=_lines(text), accent=1))]

def heuristic(text, i, n, S, geo, fmt_name): return heuristic_all(text, i, n, S, geo, fmt_name)[0]

def plan_scenes(script, durs, fmt_name, recent, llm, geo, ident, seed):
    """returns a list of per-scene plans: component, params, camera, idea. Always returns something valid."""
    n = len(script["scenes"]); S = Ctx(ident, 3.0, seed, geo); F = FORMATS.get(fmt_name, FORMATS["scale"]); plans = None
    lines = "\n".join(f'{i + 1}. ({durs[i]:.1f}s) {s["text"]}' for i, s in enumerate(script["scenes"]))
    if llm is not None:
        try:
            rec = ("Recently used visuals in earlier videos (do not repeat the same component+subject combination): " + "; ".join(recent[-24:])) if recent else ""
            cl = '- If NOTHING above can show an idea, you may ask for ONE new component for the whole video: {"component":"custom","need":"one sentence describing exactly what to draw"}.'
            d = llm(PLAN_PROMPT.format(fmt=F["desc"] + ". Prefer: " + ", ".join(F["prefer"]), scenes=lines, cat=catalogue(geo is not None and geo.ok), custom_line=cl, recent=rec, minvar=min(6, max(4, n // 2))))
            plans = d.get("scenes") if isinstance(d, dict) else d
        except Exception as e: _log("visual planner failed, using rules:", str(e)[:150])
    out = []; made_custom = False
    for i in range(n):
        raw = (plans[i] if plans and i < len(plans) and isinstance(plans[i], dict) else {}); name = str(raw.get("component", "")).strip(); p = raw.get("params") or {}
        if name == "custom" and not made_custom:
            made_custom = True; nm = make_custom(str(raw.get("need") or raw.get("idea") or script["scenes"][i]["text"]), llm, ident)
            if nm: name = nm; p = raw.get("params") or json.loads(REG[nm]["example"] or "{}")
        if not validate(name, p, S, ident): name, p = heuristic(script["scenes"][i]["text"], i, n, S, geo, fmt_name); raw = dict(raw, idea=raw.get("idea") or "")
        out.append(dict(component=name, params=p, camera=str(raw.get("camera", "")), idea=str(raw.get("idea") or ""), transition=""))
    return diversify(out, script, S, geo, fmt_name, ident, seed)

def diversify(plans, script, S, geo, fmt_name, ident, seed):
    """no two neighbours alike, no component more than twice, camera rhythm, strong first scene"""
    r = random.Random(seed); n = len(plans); count = {}
    for i, pl in enumerate(plans):
        prev = plans[i - 1]["component"] if i else None; lim = lambda c: 1 if c in ("kinetic_text", "big_number") else 2
        if pl["component"] == prev or count.get(pl["component"], 0) >= lim(pl["component"]):
            alts = [(nm, p) for nm, p in heuristic_all(script["scenes"][i]["text"], i, n, S, geo, fmt_name) if nm != prev and count.get(nm, 0) < lim(nm)]
            if alts: pl["component"], pl["params"] = alts[0]
        count[pl["component"]] = count.get(pl["component"], 0) + 1
    last = None
    for i, pl in enumerate(plans):
        cam = pl["camera"] if pl["camera"] in CAMS else None
        if cam is None or cam == last: cam = r.choice([c for c in CAMS if c != last and c != "still"])
        if i == 0: cam = "push"
        pl["camera"] = last = cam
    return plans

def signature(plans):
    """short strings describing what visuals a video used (kept in state to avoid repeating them in later videos)"""
    out = []
    for pl in plans:
        p = pl["params"]; subj = []
        for key in ("a", "b", "left", "right"): 
            if isinstance(p.get(key), dict): subj.append(str(p[key].get("name") or p[key].get("label")))
        for key in ("items", "objects", "bodies", "layers", "events"):
            for it in (p.get(key) or [])[:3]:
                if isinstance(it, dict): subj.append(str(it.get("label") or it.get("year")))
        for key in ("label", "kind", "title"):
            if isinstance(p.get(key), str): subj.append(p[key]); break
        out.append(pl["component"] + ":" + "/".join(x for x in subj if x)[:50])
    return out

def plan_markdown(script, plans, durs, trans):
    rows = ["| # | Narration | Visual idea | Elements | Motion | Camera | Sec | Transition |", "|---|---|---|---|---|---|---|---|"]
    for i, (s, pl) in enumerate(zip(script["scenes"], plans)):
        d = REG[pl["component"]]["desc"].split(";")[0].split(":")[0][:60]
        rows.append(f"| {i + 1} | {s['text']} | {pl['idea'] or d} | {pl['component']} | {d} | {pl['camera']} | {durs[i]:.1f} | {trans[i]} |")
    return "\n".join(rows)

# ───────────── rendering one frame of a planned scene ─────────────
def render(sc, t, ident, geo):
    """sc needs: gfx (plan dict), dur, t0, seed. Returns an RGB image W x H."""
    S = getattr(sc, "_gctx", None)
    if S is None: S = sc._gctx = Ctx(ident, sc.dur, sc.seed, geo)
    t = clamp(t, 0, sc.dur); k = t / max(sc.dur, .01); pl = sc.gfx
    g = G(ident.frame(sc.t0 + t))
    try: REG[pl["component"]]["fn"](g, pl["params"], t, k, S)
    except Exception as e:
        _log("render error in", pl["component"], str(e)[:100]); REG["kinetic_text"]["fn"](g, dict(lines=_lines(sc.text), accent=1), t, k, S)
    fr = g.finish()
    return cinematic(apply_camera(fr, pl["camera"], k, sc.t0 + t, sc.seed % 9), sc.t0 + t)

# ═════════════════════════ v3: full-bleed 3D worlds (real perspective), cinematic finishing ═════════════════════════
REG.pop("iso_scene", None)
CARD_STYLE = set()      # components that look like UI cards: kept only as emergency fallbacks, hidden from the planner

def tag(g, x, y, text, size=34, col=(255, 255, 255), a=255, anchor="mm", maxw=None):
    """label drawn straight on the picture: bold text with a thick dark outline and soft shadow (no boxes)"""
    text = str(text)
    if maxw:
        while size > 16 and g.tw(text, size) > maxw: size -= 2
    y = max(y, BOX[1] + size * .6)          # never under the channel logo at the top
    g.text(x + 2, y + 3, text, size, (0, 0, 0, int(a * .45)), anchor, stroke=size * .16, sfill=(0, 0, 0))
    g.text(x, y, text, size, tuple(col)[:3] + (int(a),), anchor, stroke=size * .13, sfill=(12, 14, 30))

def _norm(v): n = np.linalg.norm(v); return v / n if n else v

class Mesh:
    """low-poly world: faces are quads/tris with a colour, an object id (growth animation) and a layer (0 ground, 1 shadow, 2 objects)"""
    def __init__(s): s.f = []; s.nobj = 0; s.beacons = []
    def obj(s): s.nobj += 1; return s.nobj - 1
    def quad(s, p, col, o, layer=2, emis=False, alpha=255):
        p = np.array(p, np.float32)
        if len(p) == 3: p = np.vstack([p, p[2:3]])
        n = np.cross(p[1] - p[0], p[2] - p[0]); n = _norm(n) if np.linalg.norm(n) > 1e-9 else np.array([0, 1, 0], np.float32)
        s.f.append((p, col, o, layer, emis, n, alpha))
    def box(s, x, z, w, d, h, col, o, y0=0.0, windows=None, top=None):
        x1, z1, y1 = x + w, z + d, y0 + h; c = col
        s.quad([(x, y0, z1), (x1, y0, z1), (x1, y1, z1), (x, y1, z1)], c, o); s.quad([(x1, y0, z), (x1, y0, z1), (x1, y1, z1), (x1, y1, z)], shade(c, .8), o)
        s.quad([(x1, y0, z), (x, y0, z), (x, y1, z), (x1, y1, z)], shade(c, .9), o); s.quad([(x, y0, z1), (x, y0, z), (x, y1, z), (x, y1, z1)], shade(c, .72), o)
        s.quad([(x, y1, z1), (x1, y1, z1), (x1, y1, z), (x, y1, z)], top or shade(c, 1.15), o)
        if windows:
            wc, lit = windows; cols_ = max(1, int(w // 2.4)); rows_ = max(1, int(h // 3.0)); rr = random.Random(int(x * 7 + z * 13))
            for fc in range(cols_):
                for fr_ in range(rows_):
                    wx = x + (fc + .5) * w / cols_; wy = y0 + (fr_ + .55) * h / rows_; on = lit and rr.random() < .6
                    for (px_, fz) in ((wx, z1 + .03),):
                        s.quad([(px_ - .55, wy - .6, fz), (px_ + .55, wy - .6, fz), (px_ + .55, wy + .6, fz), (px_ - .55, wy + .6, fz)], (255, 226, 140) if on else wc, o, 2, on)
    def gable(s, x, z, w, d, y, hh, col, o):
        a, b, c_, d_ = (x - .4, y, z - .4), (x + w + .4, y, z - .4), (x + w + .4, y, z + d + .4), (x - .4, y, z + d + .4); a1, a2 = (x - .4, y + hh, z + d / 2), (x + w + .4, y + hh, z + d / 2)
        s.quad([d_, c_, a2, a1], shade(col, 1.08), o); s.quad([b, a, a1, a2], shade(col, .82), o); s.quad([c_, b, a2], shade(col, .72), o); s.quad([a, d_, a1], shade(col, .62), o)
    def tree(s, x, z, o, r=1.8):
        s.box(x - .25, z - .25, .5, .5, 1.6, (96, 64, 40), o)
        for j, (rr, yy, hh) in enumerate(((r, 1.4, 2.6), (r * .72, 3.0, 2.2))):
            pts = [(x - rr, yy, z - rr), (x + rr, yy, z - rr), (x + rr, yy, z + rr), (x - rr, yy, z + rr)]; ap = (x, yy + hh, z); col = (44 + j * 12, 138 + j * 14, 68)
            for q in range(4): s.quad([pts[q], pts[(q + 1) % 4], ap], shade(col, .7 + .14 * q), o)
    def panel(s, x, z, w, d, y, col, o, tilt=1.0):
        s.quad([(x, y, z + d), (x + w, y, z + d), (x + w, y + tilt, z), (x, y + tilt, z)], col, o)
        for j in range(1, 3): xx = x + w * j / 3; s.quad([(xx - .04, y + .01, z + d), (xx + .04, y + .01, z + d), (xx + .04, y + tilt + .01, z), (xx - .04, y + tilt + .01, z)], shade(col, 1.6), o)
    def shadow(s, x, z, w, d, h, o, L=(.55, .0, .35)):
        dx, dz = L[0] * h * .8, L[2] * h * .8; a = 110
        s.quad([(x, .03, z + d), (x + w, .03, z + d), (x + w + dx, .03, z + d + dz), (x + dx, .03, z + d + dz)], (0, 0, 0), o, 1, False, a)
        s.quad([(x + w, .03, z), (x + w, .03, z + d), (x + w + dx, .03, z + d + dz), (x + w + dx, .03, z + dz)], (0, 0, 0), o, 1, False, a)
    def person(s, x, y, z, o, col, sc=1.0):
        s.box(x - .17 * sc, z - .1 * sc, .34 * sc, .2 * sc, .8 * sc, col, o, y0=y + .45 * sc); s.box(x - .13 * sc, z - .1 * sc, .26 * sc, .2 * sc, .26 * sc, (240, 200, 160), o, y0=y + 1.27 * sc)
        s.box(x - .15 * sc, z - .08 * sc, .12 * sc, .16 * sc, .5 * sc, (40, 50, 90), o, y0=y); s.box(x + .03 * sc, z - .08 * sc, .12 * sc, .16 * sc, .5 * sc, (40, 50, 90), o, y0=y)
    def pack(s):
        s.V = np.stack([f[0] for f in s.f]); s.C = np.array([f[1] for f in s.f], np.float32); s.O = np.array([f[2] for f in s.f]); s.L = np.array([f[3] for f in s.f])
        s.E = np.array([f[4] for f in s.f]); s.N = np.stack([f[5] for f in s.f]); s.A = np.array([f[6] for f in s.f], np.float32); s.y0 = s.V[:, :, 1].copy(); return s

SKIES = {"day": ((70, 140, 230), (205, 228, 250), (118, 176, 92)), "dusk": ((70, 62, 140), (255, 172, 112), (84, 112, 76)), "night": ((6, 12, 38), (36, 56, 112), (22, 40, 56))}

def build_city(kind, n, hl, seed):
    r = random.Random(seed); M = Mesh(); ground = M.obj(); n = int(clamp(n, 4, 80)); hl = int(clamp(hl, 0, n))
    cols = int(math.ceil(math.sqrt(n * (1.5 if kind in ("houses", "towers") else 1.0)))); rows = int(math.ceil(n / cols)); pitch = {"houses": 16, "towers": 13, "solar": 9, "farm": 14}.get(kind, 14)
    ext = max(cols, rows) * pitch; gcol = {"houses": (96, 172, 86), "towers": (74, 80, 98), "solar": (112, 156, 82), "farm": (126, 160, 80)}.get(kind, (96, 172, 86)); T = 30
    for gx in range(-9, 10):
        for gz in range(-9, 10): M.quad([(gx * T, 0, gz * T), ((gx + 1) * T, 0, gz * T), ((gx + 1) * T, 0, (gz + 1) * T), (gx * T, 0, (gz + 1) * T)], shade(gcol, 1 + .035 * ((gx * 3 + gz * 5) % 3 - 1)), ground, 0)
    if kind in ("houses", "towers"):
        for rr in range(rows + 1):
            z = (rr - rows / 2) * pitch - pitch / 2 - 3; M.quad([(-cols * pitch / 2 - 14, .02, z), (cols * pitch / 2 + 14, .02, z), (cols * pitch / 2 + 14, .02, z + 6), (-cols * pitch / 2 - 14, .02, z + 6)], (58, 62, 72), ground, 0)
            for dsh in range(int(cols * 2 + 4)): xx = -cols * pitch / 2 - 12 + dsh * 8; M.quad([(xx, .04, z + 2.8), (xx + 3, .04, z + 2.8), (xx + 3, .04, z + 3.2), (xx, .04, z + 3.2)], (235, 235, 220), ground, 0)
        sel = set(r.sample(range(n), hl)) if hl else set()
        for i in range(n):
            c_, r_ = i % cols, i // cols; x = (c_ - cols / 2) * pitch; z = (r_ - rows / 2) * pitch; o = M.obj(); on = i in sel
            if kind == "houses":
                wall = r.choice([(236, 210, 176), (218, 156, 116), (228, 228, 216), (204, 174, 146)]); M.shadow(x, z, 8, 8, 5, o); M.box(x, z, 8, 8, 5, wall, o, windows=((120, 165, 205), False)); M.gable(x, z, 8, 8, 5, 2.8, (70, 80, 112) if on else (156, 70, 54), o)
                M.quad([(x + 3.6, .05, z + 8.03), (x + 4.4, .05, z + 8.03), (x + 4.4, 2.3, z + 8.03), (x + 3.6, 2.3, z + 8.03)], (112, 72, 52), o)
                if on:
                    po = M.obj(); M.panel(x + .8, z + 3.6, 3.0, 2.2, 5.5, (26, 52, 120), po, 1.5); M.panel(x + 4.2, z + 3.6, 3.0, 2.2, 5.5, (26, 52, 120), po, 1.5); M.beacons.append((x + 4, 14, z + 4, po))
                if r.random() < .55: M.tree(x + r.choice([-3.2, 11.2]), z + r.uniform(0, 7), M.obj())
            else:
                h = r.uniform(16, 52) * (1.35 if on else 1); wcol = (255, 150, 50) if on else r.choice([(112, 134, 184), (92, 114, 156), (132, 144, 176)]); M.shadow(x, z, 8, 8, h, o); M.box(x, z, 8, 8, h, wcol, o, windows=((150, 192, 238), r.random() < .5))
                if on: M.beacons.append((x + 4, h + 10, z + 4, o))
    else:
        rowsN = max(3, int(math.sqrt(n))); per = int(math.ceil(n / rowsN))
        for i in range(n):
            c_, r_ = i % per, i // per; x = (c_ - per / 2) * pitch; z = (r_ - rowsN / 2) * pitch * 1.4; o = M.obj(); on = i < hl
            if kind == "solar":
                M.shadow(x, z, 6, 3, 1.6, o); M.box(x + 2.6, z + 1, .4, .4, 1.1, (130, 130, 140), o); M.panel(x, z, 6, 3.4, 1.1, (255, 196, 64) if on else (26, 58, 140), o, 1.5)
                if on and i % 3 == 0: M.beacons.append((x + 3, 6, z + 1.5, o))
            else: M.box(x, z, pitch - 2, pitch * 1.2, .35, (255, 150, 40) if on else r.choice([(120, 170, 70), (205, 184, 84), (90, 150, 80)]), o)
        for tt in range(16): M.tree(r.uniform(-ext, ext), r.uniform(ext * .7, ext * 1.2), M.obj(), 2.4)
    M.pack(); M.ext = ext; M.kind = kind; return M

def cam_pose(move, k, ext, seed):
    a0 = (seed % 7) * .8; k = smooth(k)
    if move == "fly":    pos = np.array([-ext * .12, ext * .30, ext * .95 - k * ext * .80]); tgt = np.array([ext * .04, 2, -ext * .35 - k * ext * .20])
    elif move == "rise": pos = np.array([ext * .55, lerp(ext * .10, ext * .72, k), ext * .78]); tgt = np.array([0, 3, 0])
    elif move == "push": rr = lerp(ext * 1.25, ext * .62, k); pos = np.array([rr * .45, ext * .46, rr * .8]); tgt = np.array([0, 3, 0])
    else:                a = a0 + k * .95; rr = ext * .88; pos = np.array([math.sin(a) * rr, ext * .42, math.cos(a) * rr]); tgt = np.array([0, 3, 0])
    return pos, tgt

def _clip_near(P, near):
    out = []; n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]; ia, ib = a[2] > near, b[2] > near
        if ia: out.append(a)
        if ia != ib: t = (near - a[2]) / (b[2] - a[2]); out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near))
    return out

def sky_image(top, hor, ground, yh, key_cache):
    key = (top, hor, ground, int(yh // 12))
    if key not in key_cache:
        key_cache.clear(); yy = np.arange(H, dtype=np.float32)[:, None, None]; yhc = max(yh, 8.0)
        kt = np.clip(yy / yhc, 0, 1) ** 1.2; sky = np.array(top, np.float32) * (1 - kt) + np.array(hor, np.float32) * kt
        kg = np.clip((yy - yhc) / max(H - yhc, 1), 0, 1) ** .6; gr = np.array(mixc(ground, hor, .5), np.float32) * (1 - kg) + np.array(ground, np.float32) * .92 * kg
        arr = np.where(yy > yhc, gr, sky) * np.ones((1, W, 1), np.float32)
        key_cache[key] = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").resize((W * SS, H * SS), Image.Resampling.BILINEAR)
    return key_cache[key]

def draw_mesh(g, M, pos, tgt, S, k, t, tod="day", fogd=None):
    top, hor, gcol = SKIES.get(tod, SKIES["day"]); fwd = _norm(tgt - pos); right = _norm(np.cross(fwd, np.array([0, 1, 0]))); up = np.cross(right, fwd)
    f = W * .62; cx, cy = W / 2, H * .40; yh = cy + f * math.tan(math.asin(clamp(fwd[1], -1, 1)))
    g.im.paste(sky_image(top, hor, gcol, yh, S.__dict__.setdefault("_skyc", {})))
    if tod == "night":
        rr = random.Random(5)
        for i in range(70): g.circle(rr.random() * W, rr.random() * max(yh, 60), rr.random() * 1.3 + .4, fill=(255, 255, 255, int(110 + 100 * abs(math.sin(t * 1.3 + i)))))
        g.glow(W * .75, max(yh * .35, 130), 120, (210, 225, 255), .55)
    else: g.glow(W * .78, max(yh * .30, 130), 200, (255, 240, 205), .8)
    hp = [(0, yh + 4)] + [(x, yh - 30 - 22 * math.sin(x / 85 + 1) - 10 * math.sin(x / 33)) for x in range(0, W + 1, 20)] + [(W, yh + 4)]; g.poly(hp, fill=mixc(gcol, hor, .5) + (255,))
    nobj = M.nobj; grow = np.ones(nobj, np.float32)
    for o in range(1, nobj): grow[o] = eob(seg(k, .02 + (o / nobj) * .40, .24 + (o / nobj) * .40))
    gy = grow[M.O][:, None]; V = M.V.copy(); V[:, :, 1] = M.y0 * np.where(M.L[:, None] > 0, np.maximum(gy, .001), 1)
    rel = V - pos; xc = rel @ right; yc = rel @ up; zc = rel @ fwd; near = .8
    vis = np.einsum("ij,ij->i", M.N, pos - V.mean(axis=1)) > 0; inview = zc.max(axis=1) > near
    keep = inview & (vis | (M.L < 2)); dep = np.maximum(zc.mean(axis=1), 0); fogk = 1 - np.exp(-dep / (fogd or M.ext * 7))
    objdep = np.zeros(nobj, np.float32); cnt = np.zeros(nobj, np.float32); np.add.at(objdep, M.O, dep); np.add.at(cnt, M.O, 1); objdep /= np.maximum(cnt, 1)
    Ld = _norm(np.array([-.45, .8, -.4])); lam = np.clip(M.N @ Ld, 0, 1) * .55 + .5; order = np.lexsort((-dep, -objdep[M.O], M.L))
    zs = np.maximum(zc, near); sx = cx + f * xc / zs; sy = cy - f * yc / zs; hv = np.array(hor, np.float32)
    for i in order:
        if not keep[i]: continue
        if zc[i].min() <= near: 
            P = _clip_near([(xc[i, j], yc[i, j], zc[i, j]) for j in range(4)], near)
            if len(P) < 3: continue
            pts = [(cx + f * a / b, cy - f * c / b) for a, c, b in P]
        else: pts = [(float(sx[i, j]), float(sy[i, j])) for j in range(4)]
        if max(q[0] for q in pts) < -40 or min(q[0] for q in pts) > W + 40 or max(q[1] for q in pts) < -40 or min(q[1] for q in pts) > H + 40: continue
        if M.L[i] == 1: g.poly(pts, fill=(8, 16, 26, int(M.A[i] * min(1, gy[i, 0])))); continue
        c2 = M.C[i] if M.E[i] else M.C[i] * lam[i]; c2 = c2 * (1 - fogk[i] * .85) + hv * fogk[i] * .85 if not M.E[i] else c2 * (1 - fogk[i] * .5) + hv * fogk[i] * .5
        g.poly(pts, fill=tuple(int(clamp(v, 0, 255)) for v in c2) + (255,))
    # beacons over the highlighted objects
    for (bx, by, bz, o) in M.beacons:
        v = np.array([bx, by + 2.4 * math.sin(t * 3 + bx), bz]) - pos; zz = v @ fwd
        if zz > near and grow[o] > .2:
            px, py = cx + f * (v @ right) / zz, cy - f * (v @ up) / zz; a = int(255 * clamp(grow[o])); sz = clamp(f * 3.2 / zz, 10, 52)
            g.glow(px, py, sz * 2.2, BRAND, .55); g.poly([(px - sz * .5, py - sz), (px + sz * .5, py - sz), (px, py)], fill=BRAND + (a,), outline=(255, 255, 255, a), w=2)
    return fwd, right, up, f, cx, cy, yh

@comp("city3d", "a REAL 3D miniature world seen by a moving camera (orbit, fly-over, crane rise, dolly): a suburb of houses with roofs/trees/roads, a downtown skyline with lit windows, a solar farm or farm plots; a highlighted group with orange beacons shows 'this many' (houses getting solar panels, taller towers)",
      '{"kind":"houses","count":40,"highlight":12,"label":"20 panels per home","move":"fly","time":"day"}')
def c_city3d(g, p, t, k, S):
    kind = str(p.get("kind", "houses")); key = ("city", kind, int(num(p.get("count"), 30)), int(num(p.get("highlight"), 10)))
    if getattr(S, "_ck", None) != key: S._city = build_city(kind, num(p.get("count"), 30), num(p.get("highlight"), 10), S.seed); S._ck = key
    M = S._city; move = str(p.get("move") or random.Random(S.seed).choice(["orbit", "fly", "rise", "push"])); tod = str(p.get("time") or ("night" if kind == "towers" and S.seed % 3 == 0 else ("dusk" if S.seed % 4 == 1 else "day")))
    pos, tgt = cam_pose(move, k, M.ext, S.seed); draw_mesh(g, M, pos, tgt, S, k, t, tod)
    if p.get("label"): tag(g, S.cx, S.box[3] - 62, str(p["label"]).upper(), 38, BRAND, 255 * eo3(seg(k, .35, .65)), maxw=S.bw)

# ── bars3d: extruded bars rising from a ground grid, camera orbits ──
def build_bars(items, seed):
    M = Mesh(); g0 = M.obj(); vals = [max(num(i.get("value"), 0), 0) for i in items]; vm = max(vals) or 1; n = len(items); wdt = 13; gap = 9; ext = n * (wdt + gap)
    for gx in range(-8, 9):
        for gz in range(-8, 9): M.quad([(gx * 20, 0, gz * 20), ((gx + 1) * 20, 0, gz * 20), ((gx + 1) * 20, 0, (gz + 1) * 20), (gx * 20, 0, (gz + 1) * 20)], (30, 38, 70) if (gx + gz) % 2 else (36, 46, 82), g0, 0)
    order = sorted(range(n), key=lambda i: -vals[i]); pal = [BRAND, (80, 190, 255), (120, 220, 160), (190, 140, 255), (255, 110, 130), (255, 220, 90)]
    for r_, i in enumerate(order):
        o = M.obj(); h = 4 + 46 * vals[i] / vm; x = (r_ - (n - 1) / 2) * (wdt + gap) - wdt / 2; M.shadow(x, -wdt / 2, wdt, wdt, h, o); M.box(x, -wdt / 2, wdt, wdt, h, pal[r_ % 6] if r_ else BRAND, o); M.beacons.append((x + wdt / 2, h + 7, 0, o))
    M.pack(); M.ext = max(ext, 40) * .95; M.kind = "bars"; return M

@comp("bars3d", "a 3D bar chart: glowing extruded bars of different heights standing on a grid floor with the camera sweeping around them; for rankings and comparisons of 2-6 things (values labelled in the picture)",
      '{"items":[{"label":"Russia","value":17.1},{"label":"Canada","value":10.0},{"label":"USA","value":9.8}],"unit":"million km²"}')
def c_bars3d(g, p, t, k, S):
    items = [dict(i) for i in p["items"]][:6]; key = ("bars", json.dumps(items, sort_keys=True, default=str))
    if getattr(S, "_ck", None) != key: S._city = build_bars(items, S.seed); S._ck = key
    M = S._city; a = -.40 + smooth(k) * .7; rr = M.ext * .92; pos = np.array([math.sin(a) * rr, M.ext * .46, math.cos(a) * rr]); tgt = np.array([0, 11, 0])
    fwd, right, up, f, cx, cy, yh = draw_mesh(g, M, pos, tgt, S, k, t, "night", fogd=M.ext * 5)
    vals = [max(num(i.get("value"), 0), 0) for i in items]; vm = max(vals) or 1; n = len(items); order = sorted(range(n), key=lambda i: -vals[i]); unit = str(p.get("unit", ""))
    for r_, i in enumerate(order):
        h = 4 + 30 * vals[i] / vm * eob(seg(k, .02 + r_ * .06, .30 + r_ * .06)); x = (r_ - (n - 1) / 2) * 22; v = np.array([x, h + 5, 0]) - pos; zz = v @ fwd
        if zz > 1:
            px, py = cx + f * (v @ right) / zz, cy - f * (v @ up) / zz; la = 255 * eo3(seg(k, .3 + r_ * .05, .55 + r_ * .05)); sc = clamp(f * 9 / zz / 60, .5, 1.3)
            tag(g, px, py - 40 * sc - (r_ % 2) * 26, shorten(str(items[i].get("label", "")).upper(), 12), int(22 * sc) + 6, (255, 255, 255), la); tag(g, px, py - (r_ % 2) * 26, fmt(vals[i], 1 if vm < 100 else 0) + (" " + unit if unit and r_ == 0 else ""), int(26 * sc) + 8, BRAND if r_ == 0 else (160, 215, 255), la)

# ── tower3d: a tall tower with people stacked next to it to show its height ──
def build_tower(hm, pm, seed):
    M = Mesh(); g0 = M.obj(); H_ = 70.0; sc = H_ / hm
    for gx in range(-8, 9):
        for gz in range(-8, 9): M.quad([(gx * 20, 0, gz * 20), ((gx + 1) * 20, 0, gz * 20), ((gx + 1) * 20, 0, (gz + 1) * 20), (gx * 20, 0, (gz + 1) * 20)], (60, 68, 90) if (gx + gz) % 2 else (66, 76, 100), g0, 0)
    o = M.obj(); M.shadow(-5, -5, 10, 10, H_, o); M.box(-5, -5, 10, 10, H_ * .62, (96, 140, 215), o, windows=((190, 220, 255), True)); M.box(-3.4, -3.4, 6.8, 6.8, H_ * .3, (110, 154, 228), o, y0=H_ * .62, windows=((190, 220, 255), True)); M.box(-.8, -.8, 1.6, 1.6, H_ * .1, (200, 210, 230), o, y0=H_ * .92)
    n = hm / pm; cnt = int(min(n, 90)); sh = H_ / cnt
    for i in range(cnt): po = M.obj(); M.box(-12.5, -1.2, 2.4, 2.4, sh * .94, (255, 140, 30) if i % 2 == 0 else (255, 205, 110), po, y0=i * sh)
    fo = M.obj(); M.person(-19, 0, 0, fo, (255, 255, 255), 5.0); M.cnt = cnt
    M.pack(); M.ext = 120; M.kind = "tower"; M.H = H_; M.n = n; return M

@comp("tower3d", "a tall landmark standing in a 3D plaza with tiny people stacked one on top of another beside it, counting how many people tall it is (towers, buildings, mountains, whales)",
      '{"label":"Burj Khalifa","height_m":828,"person_m":1.75}')
def c_tower3d(g, p, t, k, S):
    hm = max(num(p.get("height_m"), 100), 1); pm = max(num(p.get("person_m"), 1.75), .3); key = ("tw", hm, pm)
    if getattr(S, "_ck", None) != key: S._city = build_tower(hm, pm, S.seed); S._ck = key
    M = S._city; kk = smooth(k); pos = np.array([lerp(30, 14, kk), lerp(8, 30, kk), lerp(52, 40, kk)]); tgt = np.array([-6, lerp(16, 36, kk), 0])
    fwd, right, up, f, cx, cy, yh = draw_mesh(g, M, pos, tgt, S, k, t, "dusk", fogd=420)
    n_show = int(M.n * eo3(seg(k, .30, .85))); la = 255 * eo3(seg(k, .55, .85))
    v = np.array([-11.3, M.H + 8, 0]) - pos; zz = v @ fwd
    if zz > 1: tag(g, cx + f * (v @ right) / zz, cy - f * (v @ up) / zz, f"{n_show:,} PEOPLE TALL", 46, BRAND, 255 * eo3(seg(k, .3, .6)), maxw=S.bw - 30)
    tag(g, S.cx, S.box[3] - 62, f"{str(p.get('label', '')).upper()} · {fmt(hm)} m", 36, (255, 255, 255), la, maxw=S.bw)

# ── no more UI chips / cards: titles are dropped, labels are outlined text ──
def head(g, S, p, k): return None
def label_card(g, x, y, t, size=26, col=INK, a=255, sub=None, subcol=DIM):
    tag(g, x, y, t, size, col, a)
    if sub: tag(g, x, y + size + 4, sub, size - 6, subcol, a)

# ── textured planets (spinning) ──
_TEX, _PSPR = {}, {}
def _noise_tex(seed, w=256, h=128, octs=5):
    if seed in _TEX: return _TEX[seed]
    r = np.random.default_rng(seed); out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0.0
    for o in range(octs):
        n = r.random((3 * 2 ** o, 6 * 2 ** o)).astype(np.float32); im = Image.fromarray((n * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
        out += np.asarray(im, np.float32) / 255 * amp; tot += amp; amp *= .55
    _TEX[seed] = out / tot; return _TEX[seed]

def planet_sprite(r, kind, rot, col=(200, 160, 110), seed=3):
    R = max(8, int(r * SS) // 4 * 4); rq = int(rot * 20) % 20; key = (R, kind, rq, tuple(col), seed)
    if key in _PSPR: return _PSPR[key]
    yy, xx = np.mgrid[-R:R, -R:R].astype(np.float32); nx, ny = xx / R, yy / R; d2 = nx * nx + ny * ny; nz = np.sqrt(np.clip(1 - d2, 0, 1))
    lon = np.arctan2(nx, np.maximum(nz, 1e-4)) + rq / 20 * 2 * math.pi; lat = np.arcsin(np.clip(ny, -1, 1)); tex = _noise_tex(seed); h, w = tex.shape
    u = (((lon / (2 * math.pi)) % 1) * (w - 1)).astype(int); v = (((lat / math.pi) + .5) * (h - 1)).astype(int); n = tex[v, u]
    if kind == "earth":
        land = n > .52; c = np.where(land[..., None], np.array([70, 130, 70], np.float32) * (.7 + n[..., None] * .8), np.array([18, 70, 160], np.float32) * (.8 + n[..., None] * .5))
        cl = _noise_tex(seed + 9)[v, u]; c = c * (1 - np.clip((cl - .55) * 3, 0, .8)[..., None]) + 255 * np.clip((cl - .55) * 3, 0, .8)[..., None]
        ice = (np.abs(ny) > .88)[..., None]; c = np.where(ice, 240, c)
    elif kind == "sun": c = np.array([255, 170, 50], np.float32) * (.8 + n[..., None] * .5)
    elif kind == "rock": c = np.array(col, np.float32) * (.55 + n[..., None] * .9) * (.9 + .1 * np.sin(n[..., None] * 40))
    else:
        band = np.sin(lat * 15 + n * 4) * .5 + .5; c = np.array(col, np.float32) * (.72 + .5 * band[..., None] + .15 * n[..., None])
    L = _norm(np.array([-.55, -.45, .7])); lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    amb = .10 if kind != "sun" else 1.0; lit = (amb + (1 - amb) * lam)[..., None] if kind != "sun" else 1.0
    rim = (np.clip(1 - nz, 0, 1) ** 3 * (.9 if kind == "earth" else .35))[..., None] * np.array([90, 150, 255], np.float32) * (lam[..., None] * .8 + .2)
    out = np.clip(c * lit + rim, 0, 255); edge = np.clip((1 - np.sqrt(d2)) * R, 0, 1)
    arr = np.zeros((2 * R, 2 * R, 4), np.uint8); arr[..., :3] = out.astype(np.uint8); arr[..., 3] = (edge * 255).astype(np.uint8)
    if len(_PSPR) > 160: _PSPR.clear()
    _PSPR[key] = Image.fromarray(arr, "RGBA"); return _PSPR[key]

def space_bg(g, S, t, push=0.0):
    """full-frame deep space: tinted nebula, parallax stars"""
    rr = random.Random(S.seed + 1); acc = S.acc2
    g.glow(W * (.25 + .1 * math.sin(S.seed)), H * .30, 520, mixc((40, 40, 120), acc, .35), .45); g.glow(W * .8, H * .55, 460, mixc((20, 60, 140), S.acc, .3), .35)
    for layer, (cnt, spd, sz) in enumerate(((90, 6, 1.0), (50, 14, 1.6), (22, 30, 2.4))):
        for i in range(cnt):
            x = (rr.random() * W - t * spd * .3) % W; y = (rr.random() * H + push * spd * .5) % H; a = int(90 + 120 * abs(math.sin(t * (.8 + layer * .4) + i)))
            g.circle(x, y, sz * rr.uniform(.6, 1.2), fill=(255, 255, 255, a))

@comp("planets3d", "full-screen outer space with big lit, spinning planets drawn at TRUE relative size (Earth with continents and clouds, ringed Saturn, gas giants, the Sun, rocky moons); the camera drifts and a big ×N ratio appears; for 'how much bigger', size and scale of space objects",
      '{"items":[{"label":"Earth","value":12742,"kind":"earth","display":"12,742 km"},{"label":"Saturn","value":116460,"kind":"gas","rings":true,"color":[214,180,120],"display":"116,460 km"}]}')
def c_planets3d(g, p, t, k, S):
    items = [dict(i) for i in p["items"]][:3]; push = smooth(k); space_bg(g, S, t, push * 40)
    vals = [max(num(i.get("value"), 1), 1e-9) for i in items]; vmax = max(vals); n = len(items)
    Rbig = S.bw * (.78 if n == 2 else .52) * (1 + .08 * push); order = sorted(range(n), key=lambda i: -vals[i]); pos = {}
    if n == 1: pos[0] = (S.cx, S.cy - 30, Rbig * .8)
    else:
        big = order[0]; rmax = S.bw * (.21 if any(i.get('rings') for i in items) else .36); Rbig = min(Rbig, rmax / .56); pos[big] = (S.cx + S.bw * (.02 if rmax < S.bw * .3 else .1), S.box[1] + S.bh * .40 - 10 * push, Rbig * .56)
        slots = [(S.box[0] + 70, S.box[3] - 150), (S.box[2] - 70, S.box[3] - 80)]
        for j, i in enumerate(order[1:]): r = max(10, Rbig * .56 * vals[i] / vmax); pos[i] = (slots[j][0] + (r if j == 0 else -r) * .0, slots[j][1] - r * .2, r)
    for idx in sorted(range(n), key=lambda i: pos[i][2]):
        it = items[idx]; x, y, r = pos[idx]; ap = eob(seg(k, .03 + idx * .1, .32 + idx * .1)); kind = str(it.get("kind", "gas")); col = tuple(int(c) for c in it.get("color", (200, 160, 110))) if isinstance(it.get("color"), (list, tuple)) and len(it.get("color")) == 3 else (200, 160, 110)
        rot = t * (.04 + .02 * idx); rr_ = r * ap
        if rr_ < 3: continue
        if kind in ("earth", "gas", "sun", "rock"): g.glow(x, y, rr_ * 1.5, (90, 150, 255) if kind == "earth" else mixc(col, (255, 255, 255), .2), .35 if kind != "sun" else .8)
        if it.get("rings"):
            rx, ry = rr_ * 2.0, rr_ * .46; g.ellipse(x, y, rx, ry, outline=(226, 206, 160, 200), w=max(3, rr_ * .1)); g.ellipse(x, y, rx * .86, ry * .86, outline=(190, 170, 130, 170), w=max(2, rr_ * .05))
        g.sprite(planet_sprite(rr_, kind, rot, col, seed=3 + idx * 11), x, y)
        if it.get("rings"): g.line([(x + rr_ * 2.0 * math.cos(a_ / 20 * math.pi), y + rr_ * .46 * math.sin(a_ / 20 * math.pi)) for a_ in range(0, 21)], (226, 206, 160, 255), max(3, rr_ * .1)); g.line([(x + rr_ * 1.72 * math.cos(a_ / 20 * math.pi), y + rr_ * .4 * math.sin(a_ / 20 * math.pi)) for a_ in range(0, 21)], (190, 170, 130, 230), max(2, rr_ * .05))
        la = 255 * eo3(seg(k, .35 + idx * .08, .6 + idx * .08)); ly = min(y + rr_ + 46, S.box[3] - 8) if r > 60 else y + rr_ + 34
        tag(g, clamp(x, S.box[0] + 90, S.box[2] - 90), ly, str(it.get("label", "")).upper(), 36 if r > 60 else 30, (255, 255, 255), la); 
        if it.get("display"): tag(g, clamp(x, S.box[0] + 90, S.box[2] - 90), ly + 36, str(it["display"]), 26, S.acc2, la)
    if n >= 2 and vmax / min(vals) >= 1.5:
        ra = eo3(seg(k, .5, .85)); big_ = vmax / min(vals); tag(g, S.cx - S.bw * .18, S.box[1] + 120, f"×{big_:.0f}" if big_ >= 10 else f"×{big_:.1f}", int(150 * (.8 + .2 * eob(seg(k, .5, .75)))), BRAND, 255 * ra)

# ── sea3d: a real horizon, an ocean and things that float or sink ──
@comp("sea3d", "a wide ocean scene with sky, sun and waves where big spheres (planets, balls, rocks) drop in and FLOAT or SINK according to their density (physics computed: water = 1); for density, buoyancy, 'would it float' facts",
      '{"liquid_density":1.0,"objects":[{"label":"Saturn","density":0.69,"kind":"gas","color":[214,180,120],"rings":true},{"label":"Earth","density":5.51,"kind":"earth"}]}')
def c_sea3d(g, p, t, k, S):
    objs = [dict(o) for o in p["objects"]][:3]; ld = max(num(p.get("liquid_density"), 1.0), .01); top, hor = (70, 140, 230), (210, 232, 250); yh = H * .36
    yy = np.arange(H, dtype=np.float32)[:, None, None]; kt = np.clip(yy / yh, 0, 1) ** 1.3; arr = (np.array(top, np.float32) * (1 - kt) + np.array(hor, np.float32) * kt) * np.ones((1, W, 1), np.float32)
    water = np.clip((yy - yh) / (H - yh), 0, 1); wc = np.array((12, 60, 120), np.float32) * (.55 + .45 * (1 - water)) + np.array((70, 190, 210), np.float32) * (1 - water) ** 2 * .55
    arr = np.where(yy > yh, wc * np.ones((1, W, 1), np.float32), arr); g.im.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").resize((W * SS, H * SS), Image.Resampling.BILINEAR)); g.glow(W * .75, yh * .55, 210, (255, 240, 205), .8)
    sea_y = lambda x, depth: yh + 110 + depth + 6 * math.sin(x / 38 + t * 2.1) + 3 * math.sin(x / 17 - t * 3)
    for i in range(40):
        yb = yh + 14 + i * i * .62; g.line([(x, yb + 3 * math.sin(x / (30 + i * 2) + t * 1.5 + i)) for x in range(0, W + 1, 24)], (255, 255, 255, int(46 - i)), 1 + i * .06)
    n = len(objs); slot = S.bw / n; r = min(slot * .42, 150)
    for i, o in enumerate(objs):
        d = num(o.get("density"), 1.0); f = d / ld; kind = str(o.get("kind", "gas")); col = tuple(o.get("color", (200, 160, 110))) if isinstance(o.get("color"), (list, tuple)) else (200, 160, 110)
        xc = S.box[0] + slot * (i + .5); wl = sea_y(xc, 0); drop = eo3(seg(k, .06 + i * .1, .42 + i * .1)); settle = eo3(seg(k, .38 + i * .1, .8 + i * .1)); bob = math.sin(t * 2.2 + i * 1.7) * 5 * settle
        if f < 1: y_rest = wl - r + 2 * r * _submerge(f) + bob
        else: y_rest = wl + 150 + r * .6 + math.sin(t * 1.4) * 4
        y0_ = S.box[1] - r - 20; y = lerp(y0_, wl - r * .5, drop) if settle <= 0 else lerp(wl - r * .5, y_rest, settle)
        if f >= 1 and settle > 0: g.glow(xc, y, r * 1.2, (120, 210, 255), .0)
        if o.get("rings"): g.ellipse(xc, y, r * 2.0, r * .4, outline=(226, 206, 160, 200), w=max(3, r * .09))
        spr = planet_sprite(r, kind if kind in ("earth", "gas", "sun", "rock") else "gas", t * .05, col, seed=3 + i * 11); g.sprite(spr, xc, y)
        if o.get("rings"): g.line([(xc + r * 2.0 * math.cos(a_ / 20 * math.pi), y + r * .4 * math.sin(a_ / 20 * math.pi)) for a_ in range(0, 21)], (226, 206, 160, 255), max(3, r * .09))
        if f >= 1 and y + r > wl:    # sinking: water closes over it
            pass
    edge = [(x, sea_y(x, 0)) for x in range(0, W + 1, 8)]
    g.poly(edge + [(W, H), (0, H)], fill=(18, 84, 166, 150)); g.line(edge, (210, 240, 255, 235), 4)
    for i, o in enumerate(objs):
        d = num(o.get("density"), 1.0); xc = S.box[0] + slot * (i + .5); la = 255 * eo3(seg(k, .5 + i * .1, .8 + i * .1)); ok = d / ld < 1
        g.circle(xc, sea_y(xc, 0) + 4, r * (.5 + .12 * math.sin(t * 3)), outline=(255, 255, 255, 60), w=2)
        tag(g, xc, S.box[1] + 20, str(o.get("label", "")).upper(), 38, (255, 255, 255), la); tag(g, xc, S.box[1] + 62, f"density {d:g}", 28, S.acc2, la)
        tag(g, xc, S.box[3] - 80, "FLOATS" if ok else "SINKS", 46, (130, 255, 175) if ok else (255, 120, 120), la)

# ── terrain map: relief-textured land, true-scale country shape extruded and laid over another place ──
def _fbm(seed, w=W, h=H, octs=5):
    r = np.random.default_rng(seed); out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0.0
    for o in range(octs):
        n = r.random((3 * 2 ** o, 2 * 2 ** o)).astype(np.float32); im = Image.fromarray((n * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC); out += np.asarray(im, np.float32) / 255 * amp; tot += amp; amp *= .55
    return out / tot

def terrain_bg(S, geo, lon0, lat0, scale, cx, cy):
    key = ("terrain", round(lon0, 2), round(lat0, 2), round(scale, 2)); cache = S.__dict__.setdefault("_tb", {})
    if key in cache: return cache[key]
    kx = math.cos(math.radians(lat0)); proj = lambda ring: [((x - lon0) * kx * scale + cx, -(y - lat0) * scale + cy) for x, y in ring]
    land = Image.new("L", (W, H), 0); ld = ImageDraw.Draw(land); borders = []
    for nm, it in geo.items.items():
        for ring in it["rings"]:
            xs = [q[0] for q in ring]; ys = [q[1] for q in ring]
            if max(xs) < lon0 - 90 / max(kx, .3) or min(xs) > lon0 + 90 / max(kx, .3) or max(ys) < lat0 - 60 or min(ys) > lat0 + 60: continue
            pr = proj(ring)
            if max(q[0] for q in pr) < -40 or min(q[0] for q in pr) > W + 40 or max(q[1] for q in pr) < -40 or min(q[1] for q in pr) > H + 40: continue
            if len(pr) >= 3: ld.polygon(pr, fill=255); borders.append(pr)
    m = np.asarray(land, np.float32) / 255; shallow = np.asarray(land.filter(ImageFilter.GaussianBlur(26)), np.float32) / 255
    n1 = _fbm(S.seed + 3); n2 = _fbm(S.seed + 4, octs=6); ocean = np.array((8, 38, 88), np.float32) * (.8 + n1[..., None] * .5); ocean = ocean * (1 - shallow[..., None] * .75) + np.array((38, 150, 170), np.float32) * shallow[..., None] * .75
    ramp = [(.0, (58, 108, 58)), (.35, (96, 138, 66)), (.55, (160, 150, 92)), (.75, (150, 118, 82)), (1.0, (214, 206, 190))]
    v = np.clip(n2 * 1.3 - .15, 0, 1); land_c = np.zeros((H, W, 3), np.float32)
    for (a0, c0), (a1, c1) in zip(ramp[:-1], ramp[1:]): mk = ((v >= a0) & (v <= a1))[..., None]; land_c = np.where(mk, np.array(c0, np.float32) + (np.array(c1, np.float32) - np.array(c0, np.float32)) * ((v[..., None] - a0) / (a1 - a0)), land_c)
    gx, gy_ = np.gradient(n2); hs = 1 + (-gx - gy_) * 14; land_c = land_c * np.clip(hs, .6, 1.5)[..., None]
    img = ocean * (1 - m[..., None]) + land_c * m[..., None]; im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB"); d = ImageDraw.Draw(im, "RGBA")
    for pr in borders: d.line(pr + [pr[0]], fill=(255, 255, 255, 70), width=1)
    cache.clear(); cache[key] = im; return im

@comp("map_compare", "geography: a real country/US-state shape is laid at TRUE scale over another country/state on relief terrain (the camera pushes in), like 'Sweden is bigger than Italy'; needs real English place names",
      '{"a":{"name":"Sweden"},"b":{"name":"Italy"}}')
def c_map_compare(g, p, t, k, S):
    geo = S.geo; A, B = geo.find(p["a"]["name"]), geo.find(p["b"]["name"]); ca, cb = centroid(A), centroid(B); aa, ab = area_of(A), area_of(B)
    lon0, lat0 = cb; probe = project_item(B, lon0, lat0, 1, only_main=True)[0] + project_item(A, ca[0], ca[1], 1, only_main=True)[0]
    xs = [q[0] for q in probe]; ys = [q[1] for q in probe]; scale = min((S.bw * .8) / max(max(xs) - min(xs), 1e-6), (S.bh * .72) / max(max(ys) - min(ys), 1e-6)); cx, cy = S.cx, S.cy
    bg = terrain_bg(S, geo, lon0, lat0, scale, cx, cy); g.im.paste(bg.resize((W * SS, H * SS), Image.Resampling.BILINEAR))
    B_r = project_item(B, lon0, lat0, scale, cx, cy, only_main=True)[0]; land = eo3(seg(k, .12, .6)); ox, oy = (1 - land) * S.bw * .7, (1 - land) * -S.bh * .25
    g.poly(B_r, fill=(40, 90, 200, 110), outline=(180, 220, 255, 255), w=4)
    A_r = project_item(A, ca[0], ca[1], scale, cx + ox, cy + oy, only_main=True)[0]; dep = 16 * (1 - .4 * land)
    g.poly([(x + 10, y + dep + 6) for x, y in A_r], fill=(0, 0, 0, 110))
    for dd in range(int(dep), 0, -2): g.poly([(x, y + dd) for x, y in A_r], fill=(150, 70, 10, 255))
    g.poly(A_r, fill=BRAND + (235,), outline=(255, 255, 255, 255), w=4)
    la = 255 * eo3(seg(k, .5, .8)); bx = [q[0] for q in B_r]; by = [q[1] for q in B_r]
    tag(g, cx, min(max(by) + 48, S.box[3] - 110), f"{str(p['b']['name']).upper()}  {fmt(ab)} km²", 34, (190, 225, 255), la, maxw=S.bw)
    tag(g, cx, min(max(by) + 94, S.box[3] - 64), f"{str(p['a']['name']).upper()}  {fmt(aa)} km²", 34, BRAND, la, maxw=S.bw)


# ═════════════ v4: box-free versions of the remaining 2D components (everything is text/shapes drawn on the picture itself) ═════════════
@comp("flow", "a glowing path with 2-4 numbered orbs; each step lights up in turn with the step text written on the picture; for 'how it works', cause and effect, processes",
      '{"title":"HOW IT WORKS","steps":["Sunlight hits the panel","Electrons start moving","Electric current flows"]}')
def c_flow(g, p, t, k, S):
    st = [str(x) for x in p["steps"]][:4]; n = len(st); x = S.box[0] + 70; y0, y1 = S.box[1] + 70, S.box[3] - 40
    if p.get("title"): tag(g, S.cx, S.box[1] + 10, str(p["title"]).upper(), 36, S.acc, 255 * eo3(seg(k, 0, .15)), maxw=S.bw)
    ys = [y0 + (y1 - y0) * (i / max(n - 1, 1)) for i in range(n)]; prog = eo3(seg(k, .05, .85)) * (y1 - y0)
    g.line([(x, y0), (x, y1)], (255, 255, 255, 40), 6); g.line([(x, y0), (x, y0 + prog)], BRAND + (255,), 8); g.glow(x, y0 + prog, 60, BRAND, .6)
    for i, (yy, s_) in enumerate(zip(ys, st)):
        ap = eob(seg(k, .08 + i * .17, .3 + i * .17)); g.glow(x, yy, 70 * ap, BRAND, .5); g.circle(x, yy, 30 * ap, fill=BRAND + (255,), outline=(255, 255, 255, 255), w=4)
        g.text(x, yy, str(i + 1), int(34 * ap) + 1, (20, 14, 30, 255))
        tag(g, x + 56, yy, shorten(s_, 44), 36, (255, 255, 255), 255 * clamp(ap), anchor="lm", maxw=S.bw - 130)

@comp("timeline", "a vertical timeline: big years pop in one after another along a glowing line with the event written beside each (history, 'in 1969... in 1972...')",
      '{"title":"HOW IT HAPPENED","events":[{"year":"1957","label":"First satellite"},{"year":"1969","label":"Moon landing"}]}')
def c_timeline(g, p, t, k, S):
    ev = [dict(e) for e in p["events"]][:5]; n = len(ev); y0, y1 = S.box[1] + 90, S.box[3] - 50; x = S.cx - 150
    if p.get("title"): tag(g, S.cx, S.box[1] + 10, str(p["title"]).upper(), 36, S.acc, 255 * eo3(seg(k, 0, .15)), maxw=S.bw)
    g.line([(x, y0 - 20), (x, lerp(y0 - 20, y1 + 10, eo3(seg(k, 0, .6))))], S.acc + (220,), 6)
    for i, e in enumerate(ev):
        y = y0 + (y1 - y0) * (i / max(n - 1, 1)) if n > 1 else (y0 + y1) / 2; ap = eob(seg(k, .08 + i * .13, .3 + i * .13))
        g.glow(x, y, 46 * ap, BRAND, .55); g.circle(x, y, 16 * ap, fill=BRAND + (255,), outline=(255, 255, 255, 255), w=4)
        tag(g, x - 34, y, str(e.get("year", "")), 56, (255, 255, 255), 255 * clamp(ap), anchor="rm")
        tag(g, x + 36, y, shorten(str(e.get("label", "")), 34), 32, S.acc2, 255 * clamp(ap), anchor="lm", maxw=S.box[2] - x - 40)

@comp("versus", "split-screen face-off: the picture is cut diagonally into two coloured halves, each with a huge number and its name, a VS in the middle and the ratio at the bottom; for 'A vs B' comparisons of one number",
      '{"title":"HEAD TO HEAD","left":{"label":"Cheetah","value":"120 km/h"},"right":{"label":"Car in town","value":"50 km/h"},"verdict":"2.4x faster"}')
def c_versus(g, p, t, k, S):
    L, R = p["left"], p["right"]; a = eo3(seg(k, 0, .35)); sk = 90; mid = W / 2; y0, y1 = S.box[1] - 20, S.box[3] + 40
    g.poly([(0, y0), (mid + sk * (1 - a) * 0 + sk / 2 * a, y0), (mid - sk / 2 * a, y1), (0, y1)], fill=BRAND + (int(70 * a),))
    g.poly([(mid + sk / 2 * a, y0), (W, y0), (W, y1), (mid - sk / 2 * a, y1)], fill=S.acc + (int(70 * a),))
    g.line([(mid + sk / 2 * a, y0), (mid - sk / 2 * a, y1)], (255, 255, 255, int(220 * a)), 5)
    for i, (it, col) in enumerate(((L, BRAND), (R, S.acc))):
        ap = eob(seg(k, .1 + i * .12, .45 + i * .12)); cx = W * (.25 if i == 0 else .75); v = str(it.get("value", "")); n_ = num(v, None) if re.search(r"\d", v) else None
        if n_ is not None: v = re.sub(r"[\d,]+(\.\d+)?", lambda m: fmt(n_ * eo3(seg(k, .3, .75)), 1 if "." in m.group() else 0), v, count=1)
        tag(g, cx, S.cy - 90, str(it.get("label", "")).upper(), 40, (255, 255, 255), 255 * clamp(ap), maxw=W / 2 - 40)
        tag(g, cx, S.cy + 10, v, 70, col, 255 * clamp(ap), maxw=W / 2 - 40)
    ra = eob(seg(k, .4, .6)); g.circle(mid, S.cy - 20, 46 * ra, fill=(255, 255, 255, 255), outline=BRAND + (255,), w=6); g.text(mid, S.cy - 20, "VS", int(40 * ra) + 1, (20, 14, 30, 255))
    if p.get("verdict"): tag(g, S.cx, S.box[3] - 20, str(p["verdict"]).upper(), 54, BRAND, 255 * eo3(seg(k, .6, .9)), maxw=S.bw)

for _n in ("water_tank", "size_compare", "bars", "person_scale", "iso_scene"): REG.pop(_n, None)     # replaced by the full-bleed 3D versions
