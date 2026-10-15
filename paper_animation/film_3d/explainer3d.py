"""
The explainer again, in 3D: the same script, numbers and images as
scenes.py, staged on a lit set and rendered by engine3d.py.

Text, prompts, colours and every figure come from film_2d/scenes.py and the
asset JSONs prepare/extract_cutouts.py writes, so the two films cannot drift
apart. What
changes is the staging: panels are glass, tiles are slabs that lift and
shuffle in depth, the recall wall becomes a skyline cut by a descending
cutoff plane, and the funnel starts from all 20,000 images as a carpet.

    python film_3d/explainer3d.py                   # every beat, then the film
    python film_3d/explainer3d.py Recall Results    # just these beats
    python film_3d/explainer3d.py Recall --still 4 12.5
"""

import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the script, prompts and colours live with the 2D film
sys.path.insert(0, os.path.join(ROOT, "film_2d"))

from engine3d import (CAP_PER_PT, FPS, FFMPEG, Beat, Camera, Env, Group, Hud,
                      HudRule, Instances, Label, Picture, Renderer, Rod, Slab,
                      Solid, Track, back, cap_ratio, encoder, font, inn, inout, lin,
                      linear, measure, out, outq, still)
from scenes import (ALT, ANOM, ASSETS, CAND, DIM, DROP, INK, LENS,
                    LIKERT_PROMPT, LOGOS, PANEL_W, PICK, STAGE_COLORS,
                    TOURNAMENT_PROMPT, WARM, wrap_prompt)

OUT = os.path.join(ROOT, "media", "3d")
BG = "#04060d"
Y0 = 3.6                 # manim's y = 0 sits this far above the floor


def image_list():
    """Every cutout the film shows, one texture layer each."""
    items = [(os.path.basename(p)[:-4], p)
             for p in sorted(glob.glob(os.path.join(ASSETS, "*.png")))]
    items += [("a:" + os.path.basename(p)[:-4], p)
              for p in sorted(glob.glob(os.path.join(ANOM, "*.png")))]
    items += [("c:" + os.path.basename(p)[:-4], p)
              for p in sorted(glob.glob(os.path.join(CAND, "*.png")))]
    return items


IMAGES = image_list()
LAYER = {k: i for i, (k, _) in enumerate(IMAGES)}


class Data:
    """The same ranked pools scenes.Beats.setup builds."""

    def __init__(self):
        with open(os.path.join(ASSETS, "cutouts.json")) as f:
            self.meta = json.load(f)
        ranked = sorted(self.meta, key=lambda k: -(self.meta[k]["likert"] or 0))
        self.high = [n for n in ranked if (self.meta[n]["likert"] or 0) >= 45]
        self.mid = [n for n in ranked if 18 <= (self.meta[n]["likert"] or 0) <= 30]
        self.low = [n for n in ranked if (self.meta[n]["likert"] or 0) <= 2]

    def score(self, name):
        return self.meta[name]["likert"] or 0


D = Data()


def edge_fill(key):
    """True if more than 4% of a cutout is saturated white fill."""
    from PIL import Image
    arr = np.asarray(Image.open(dict(IMAGES)[key]).convert("L"))
    return (arr >= 250).mean() > 0.04


# ── staging helpers ────────────────────────────────────────────────

def P(x, y, z=0.0):
    """A point in the 2D film's frame, lifted onto the set."""
    return (x, y + Y0, z)


def _from(track, v, t, dur, ease):
    """Start `track` at v and tween it to the value it was built with."""
    target = track.v0.copy()
    track.v0 = np.broadcast_to(np.asarray(v, float), target.shape).copy()
    track.to(t, dur, target, ease)


def show(node, t, dur=0.6, ease=out, d=None, s=None, r=None, fade=None):
    """
    Bring a node in at t, from an offset d, a scale factor s and an extra
    rotation r. Call before scheduling anything else on the node.
    """
    if d is not None:
        _from(node.pos, node.pos.v0 + np.asarray(d, float), t, dur, ease)
    if s is not None:
        _from(node.scale, node.scale.v0 * s, t, dur, ease)
    if r is not None:
        _from(node.rot, node.rot.v0 + np.asarray(r, float), t, dur, ease)
    _from(node.alpha, 0.0, t, fade or dur * 0.8, out)
    return node


def hide(node, t, dur=0.5, ease=inout, d=None, s=None, r=None):
    if d is not None:
        node.pos.to(t, dur, node.pos(t) + np.asarray(d, float), ease)
    if s is not None:
        node.scale.to(t, dur, node.scale(t) * s, ease)
    if r is not None:
        node.rot.to(t, dur, node.rot(t) + np.asarray(r, float), ease)
    node.alpha.to(t, dur, 0.0, ease)
    return node


def hud_in(h, t, dur=0.6, dy=-0.1, wipe=False):
    x, y = h.pos.v0
    _from(h.pos, (x, y + dy), t, dur, out)
    h.alpha.to(t, dur * 0.8, 1.0, out)
    if wipe:
        _from(h.reveal, 0.0, t, dur * 1.25, inout)
    return h


def hud_out(h, t, dur=0.5, dy=0.0):
    if dy:
        x, y = h.pos(t)
        h.pos.to(t, dur, (x, y + dy), inout)
    h.alpha.to(t, dur, 0.0, inout)
    return h


def title(beat, text, t=0.1, dur=0.9):
    """The heading each beat opens under, where scenes.scene_title puts it."""
    return hud_in(beat.overlay(Hud(text, 46, INK, 0, 3.5, anchor="t")), t, dur,
                  dy=-0.08, wipe=True)


def glass(w, h, color, pos=(0, 0, 0), d=0.08, r=0.14, sw=0.02, **kw):
    return Slab(w, h, d, r, color=color, glass=True, stroke=color, stroke_w=sw,
                stroke_i=1.35, pos=pos, **kw)


def tile(key, size, pos, stroke=DIM, sw=0.014, d=0.08, si=1.0, **kw):
    return Slab(size, size, d, 0.035, color="#0a0e17", image=LAYER[key],
                stroke=stroke, stroke_w=sw, stroke_i=si, pos=pos, **kw)


def chip(value, color, size, pos, font_size=24):
    """A score badge: a solid rounded block with the number on its face."""
    g = Group(pos=pos)
    depth = size * 0.34
    g.add(Slab(size, size, depth, size * 0.23, color=color, glow=0.35))
    g.add(Label(str(value), font_size, BG, face="bold",
                pos=(0, 0, depth / 2 + 0.006)))
    return g


def width(text, size, face="regular"):
    return measure(text, size, face)[0]


def cap(size, face="regular"):
    return CAP_PER_PT * size


def lerp(a, b, u):
    return a + (b - a) * u


def ramp(t, t0, dur):
    return float(np.clip((t - t0) / dur, 0.0, 1.0))


# ── 1. the problem ──────────────────────────────────────────

def opening():
    """
    A field of cutouts running out to the horizon, mostly unremarkable, as
    the real set is (8,098 of 20,000 score zero). The camera rises off it;
    the field dims under the question and a handful of striking ones lift.
    """
    b = Beat("Opening", 7.0, fade_in=0.25, fade_out=0.5)
    b.env = Env(refl=0.3, tilt=0.8, tilt_focus=0.44, tilt_band=0.16, fog=0.05,
                fog0=9.0, stars=0.7, light_box=((0, 0, -3), 24.0),
                pool=(0.0, -2.0, 14.0), light_dir=(0.45, -1.0, -0.35))
    rng = np.random.default_rng(7)
    cols = rows = 76
    pitch, size = 1.12, 1.0
    gx, gz = np.meshgrid(np.arange(cols) - (cols - 1) / 2,
                         np.arange(rows) - (rows - 1) / 2)
    x, z = gx.ravel() * pitch, gz.ravel() * pitch - 6.0
    n = len(x)

    # Cutouts on the edge of the HST footprint carry a white wedge where
    # there is no data; true to the set, but in a decorative field of
    # thousands they read as rendering faults, so the field leaves them out.
    cands = [k for k in LAYER if k.startswith("c:")]
    groups = [(D.low, 3.0), (D.mid, 1.6), (D.high, 0.35), (cands, 0.3)]
    pool, weight = [], []
    for names_, w in groups:
        for k in names_:
            if not edge_fill(k):
                pool.append(k)
                weight.append(w)
    weight = np.array(weight) / sum(weight)
    layers = np.array([LAYER[k] for k in pool])[rng.choice(len(pool), n, p=weight)]
    bright = rng.uniform(0.72, 1.05, n)

    rows_ = np.zeros((n, 32), "f4")
    rows_[:, 0], rows_[:, 5], rows_[:, 10], rows_[:, 15] = size, 0.05, size, 1
    rows_[:, 12], rows_[:, 13], rows_[:, 14] = x, 0.025, z
    rows_[:, 16:19] = lin("#0a0e17")
    rows_[:, 19] = 1
    rows_[:, 20:23] = lin(DIM)
    rows_[:, 23] = 0.012
    rows_[:, 24] = layers
    rows_[:, 28] = 1
    rows_[:, 30] = rng.integers(0, 8, n)
    delay = np.hypot(x, z - 2.0) / 15.0 + rng.uniform(0, 0.25, n)

    # The striking few, where the finished shot leaves room around the words.
    spots = [(-8.9, -2.0), (-5.6, 3.6), (6.7, 3.6), (9.0, -3.1), (-4.5, -9.8),
             (3.4, -8.7), (0.0, 5.8)]
    special = []
    for i, (sx, sz) in enumerate(spots):
        k = int(np.argmin((x - sx) ** 2 + (z - sz) ** 2))
        special.append(k)
        s = b.add(tile(D.high[i], size, (x[k], 0.025, z[k]), d=0.05,
                       rot=(-90, 0, 0), sw=0.012))
        s.orient = int(rows_[k, 30])
        show(s, float(delay[k]), 0.55, outq, d=(0, -0.35, 0))
        t_up = 2.75 + 0.09 * i
        s.anim(t_up, 0.9, outq, pos=(x[k], 0.62, z[k]))
        s.anim(t_up, 0.6, out, stroke=PICK, stroke_w=0.03, stroke_i=2.2)

    def field(t):
        u = np.clip((t - delay) / 0.55, 0, 1)
        e = 1 - (1 - u) ** 3
        r = rows_.copy()
        r[:, 13] = 0.025 - 0.35 * (1 - e)
        r[:, 28] = e
        veil = 1 - 0.7 * inout(ramp(t, 2.0, 0.9))
        r[:, 26] = bright * veil
        r[:, 27] = 0.9 * veil
        r[special, 28] = 0
        return r

    b.add(Instances(("box", "field"), field, size=(size, size, 0.03),
                    transparent=False))

    b.cam = Camera(look=(0, 0, 2.2), az=14, el=15, dist=5.2, fov=40, drift=0.03)
    b.cam.anim(0.0, 6.7, inout, look=(0, 0, -2.5), az=-3, el=46, dist=16.5, fov=36)

    ys = [1.29, 0.49, -0.15, -0.75, -1.35]
    hud_in(b.overlay(Hud("Millions of images", 54, INK, 0, ys[0])), 1.9, 1.0,
           wipe=True)
    hud_in(b.overlay(Hud("Which ones deserve astronomers' attention?", 30, DIM,
                         0, ys[1])), 2.8, 0.6, dy=-0.2)
    rule = b.overlay(HudRule(4.8, 0, ys[2], DIM))
    rule.set(0, reveal=0.0).anim(3.6, 0.5, out, alpha=0.5).anim(3.6, 0.6, inout,
                                                                reveal=1.0)
    hud_in(b.overlay(Hud("We can leverage", 26, DIM, 0, ys[3])), 3.6, 0.5, dy=0)
    hud_in(b.overlay(Hud("Vision-Language Models (VLMs)", 42, PICK, 0, ys[4])),
           4.1, 0.8, dy=-0.2)
    return b


# ── 2. what a vision-language model does with one ──────────────────

def vlm():
    """
    One call, left to right, laid out exactly as scenes.vlm lays it out:
    inputs, encoders, fusion, representation, answer. The camera starts at
    the inputs and pulls back to take in the whole pipeline.
    """
    b = Beat("VLM", 10.4)
    b.env = Env(light_box=((0, 3.3, 0), 10.0), pool=(0.0, 0.0, 9.0))
    title(b, "VLM Mechanism", dur=0.8)
    rng = random.Random(11)
    xs = [-5.64, -2.82, 0.0, 2.82, 5.64]
    top_y, bot_y, mid_y = 1.15, -1.35, -0.1
    K = 0.96
    # the 2D diagram spans y -2.33..3.51 and is scaled and hung 0.5 under
    # the title; one group carries that transform for every piece
    dia = b.add(Group(pos=(0, Y0 - 0.256 - 0.59 * K, 0), scale=K))

    def box(h, col, x, y):
        return dia.add(glass(PANEL_W, h, col, (x, y, 0)))

    def titled(x, y, h, text, col, size=18):
        return dia.add(Label(text, size, col, pos=(x, y + h / 2 - 0.26, 0.06)))

    # stage heads: coins that flip in, numbered, with the stage name below
    heads = []
    for i, name in enumerate(["Inputs", "Encode", "Align & Fuse", "Understand",
                              "Answer"]):
        col = STAGE_COLORS[i]
        coin = dia.add(Group(pos=(xs[i], 3.28, 0)))
        coin.add(Solid("cyl", col, glow=0.3, rot=(90, 0, 0), pos=(0, 0, -0.04),
                       scale=(0.23, 0.08, 0.23)))
        coin.add(Label(str(i + 1), 22, BG, face="bold", pos=(0, 0, 0.046)))
        lab = dia.add(Label(name, 24, col, pos=(xs[i], 2.77, 0)))
        heads.append((coin, lab))

    # 1 - inputs
    img_box = box(1.95, STAGE_COLORS[0], xs[0], top_y)
    img_lab = titled(xs[0], top_y, 1.95, "Image", STAGE_COLORS[0])
    pic = dia.add(tile(LENS, 1.2, (xs[0], top_y - 0.14, 0.09), d=0.06))
    txt_box = box(1.95, STAGE_COLORS[0], xs[0], bot_y)
    txt_lab = titled(xs[0], bot_y, 1.95, "Text prompt", STAGE_COLORS[0])
    lines = [dia.add(Label(s, 17, INK, pos=(xs[0], bot_y - 0.18 + (1.5 - k) * 0.33,
                                            0.06)))
             for k, s in enumerate(["Which of these are", "scientifically",
                                    "interesting?", "Score each 1–5."])]

    # 2 - encoders: the vision patches stand out of the panel by how much
    # each one carries; the text tokens are bars
    vis_box = box(1.95, STAGE_COLORS[1], xs[1], top_y)
    vis_lab = titled(xs[1], top_y, 1.95, "Vision encoder", STAGE_COLORS[1])
    patches = []
    for r_ in range(4):
        for c in range(4):
            v = rng.uniform(0.25, 0.95)
            dz = 0.04 + 0.3 * v
            p = dia.add(Solid("box", STAGE_COLORS[1], glow=0.15 + 0.9 * v,
                              pos=(xs[1] + (c - 1.5) * 0.29,
                                   top_y - 0.16 - (r_ - 1.5) * 0.29, 0.04 + dz / 2),
                              scale=(0.24, 0.24, dz)))
            patches.append(p)
    enc_box = box(1.95, STAGE_COLORS[1], xs[1], bot_y)
    enc_lab = titled(xs[1], bot_y, 1.95, "Text encoder", STAGE_COLORS[1])
    toks = []
    for k, widths in enumerate(([0.50, 0.34, 0.42], [0.38, 0.50, 0.30],
                                [0.44, 0.36])):
        total = sum(widths) + 0.09 * (len(widths) - 1)
        cx = xs[1] - total / 2
        for w in widths:
            toks.append(dia.add(Slab(w, 0.17, 0.08, 0.07, color=STAGE_COLORS[1],
                                     glow=0.5, pos=(cx + w / 2, bot_y - 0.16
                                                    + (1 - k) * 0.32, 0.09))))
            cx += w + 0.09

    # 3 - fusion, spanning both input rows
    fuse_box = box(4.45, STAGE_COLORS[2], xs[2], mid_y)
    fuse_lab = titled(xs[2], mid_y, 4.45, "Multimodal model", STAGE_COLORS[2])
    columns = []
    for dx, n in ((-0.62, 3), (0.0, 4), (0.62, 3)):
        columns.append([np.array([xs[2] + dx, mid_y - 0.3 + (k - (n - 1) / 2) * 0.78,
                                  0.1]) for k in range(n)])
    edges = [(p, q) for a, b_ in zip(columns, columns[1:]) for p in a for q in b_]
    rods = [dia.add(Rod(p, q, radius=0.008, color=STAGE_COLORS[2], glow=0.35,
                        alpha=0.7)) for p, q in edges]
    nodes = [dia.add(Solid("sphere", STAGE_COLORS[2], glow=0.9, pos=tuple(p),
                           scale=0.1)) for col in columns for p in col]

    # 4 - what it now holds
    rep_box = box(2.2, STAGE_COLORS[3], xs[3], mid_y)
    rep_lab = [dia.add(Label(s, 18, STAGE_COLORS[3], pos=(xs[3], mid_y + 0.7 + dy,
                                                         0.06)))
               for s, dy in (("Learned", 0.12), ("representation", -0.12))]
    dots_w = width("…", 22)
    row_w = 5 * 0.26 + 4 * 0.08 + 0.12 + dots_w
    kf = min(1.0, (PANEL_W - 0.3) / row_w)
    x0 = xs[3] - row_w * kf / 2
    cells = []
    for k in range(5):
        v = rng.uniform(0.35, 0.95)
        dz = 0.05 + 0.3 * v
        cells.append(dia.add(Solid("box", STAGE_COLORS[3], glow=0.2 + 0.8 * v,
                                   pos=(x0 + (0.13 + k * 0.34) * kf, mid_y - 0.4,
                                        0.04 + dz / 2),
                                   scale=(0.26 * kf, 0.26 * kf, dz))))
    dots = dia.add(Label("…", 22 * kf, STAGE_COLORS[3], anchor="l",
                         pos=(x0 + (5 * 0.26 + 4 * 0.08 + 0.12) * kf,
                              mid_y - 0.4, 0.06)))

    # 5 - the number that comes back
    ans_box = box(2.2, STAGE_COLORS[4], xs[4], mid_y)
    ans_lab = titled(xs[4], mid_y, 2.2, "Score", STAGE_COLORS[4])
    w5, wof = width("5", 62), width("of 5", 20)
    rx0 = xs[4] - (w5 + 0.18 + wof) / 2
    big = dia.add(Label("5", 62, PICK, anchor="bl", pos=(rx0, -0.36, 0.07)))
    of = dia.add(Label("of 5", 20, DIM, anchor="bl", pos=(rx0 + w5 + 0.18, -0.36,
                                                          0.07)))
    verdict = [dia.add(Label(s, 15, INK, pos=(xs[4], y, 0.07)))
               for s, y in (("exceptionally", -0.6), ("interesting", -0.8))]

    def flow(a, b_, col):
        a, b_ = np.array([*a, 0.0]), np.array([*b_, 0.0])
        u = (b_ - a) / np.linalg.norm(b_ - a)
        return dia.add(Rod(a + u * 0.07, b_ - u * 0.07, radius=0.021, color=col,
                           glow=0.9, tip=True, tip_len=0.16, tip_r=0.065))

    arrows = [
        flow((xs[0] + 1.1, top_y), (xs[1] - 1.1, top_y), STAGE_COLORS[0]),
        flow((xs[0] + 1.1, bot_y), (xs[1] - 1.1, bot_y), STAGE_COLORS[0]),
        flow((xs[1] + 1.1, top_y), (xs[2] - 1.1, mid_y + 1.25), STAGE_COLORS[1]),
        flow((xs[1] + 1.1, bot_y), (xs[2] - 1.1, mid_y - 1.25), STAGE_COLORS[1]),
        flow((xs[2] + 1.1, mid_y), (xs[3] - 1.1, mid_y), STAGE_COLORS[2]),
        flow((xs[3] + 1.1, mid_y), (xs[4] - 1.1, mid_y), STAGE_COLORS[3]),
    ]

    def grow(rod, t, dur=0.6):
        rod.grow.to(t, dur, 1.0, outq)
        pulse = dia.add(Solid("sphere", "#ffffff", glow=3.0, scale=0.055,
                              pos=tuple(rod.a), casts=False))
        show(pulse, t, 0.1)
        pulse.anim(t, dur + 0.1, inout, pos=tuple(rod.b))
        hide(pulse, t + dur, 0.25)

    def rise(node, t, dur=0.7):
        show(node, t, dur, outq, d=(0, 0, -0.6))

    # the beats, on the 2D film's clock
    for i, (coin, lab) in enumerate(heads):
        t = 0.55 + 0.12 * i
        show(coin, t, 0.7, back, r=(0, -100, 0))
        show(lab, t + 0.1, 0.5, d=(0, -0.12, 0))

    t = 1.45
    for n_ in (img_box, txt_box):
        rise(n_, t)
    for n_ in (img_lab, txt_lab):
        show(n_, t + 0.15, 0.5)
    show(pic, t + 0.2, 0.7, back, s=0.8, r=(0, 30, 0))
    for k, l_ in enumerate(lines):
        show(l_, t + 0.25 + 0.06 * k, 0.5, d=(0, -0.08, 0))

    t = 2.45
    grow(arrows[0], t)
    grow(arrows[1], t)
    for n_ in (vis_box, enc_box):
        rise(n_, t + 0.1)
    for n_ in (vis_lab, enc_lab):
        show(n_, t + 0.25, 0.5)
    for k, p in enumerate(patches):
        show(p, t + 0.35 + 0.035 * k, 0.5, back, s=(1, 1, 0.05))
    for k, tk in enumerate(toks):
        show(tk, t + 0.45 + 0.07 * k, 0.45, back, s=(0.2, 1, 1))

    t = 3.75
    grow(arrows[2], t)
    grow(arrows[3], t)
    rise(fuse_box, t + 0.1)
    show(fuse_lab, t + 0.25, 0.5)
    for k, rd in enumerate(rods):
        show(rd, t + 0.3 + 0.02 * k, 0.3)
        rd.grow.to(t + 0.3 + 0.02 * k, 0.6, 1.0, outq)
    for k, nd in enumerate(nodes):
        show(nd, t + 0.35 + 0.05 * k, 0.5, back, s=0.2)

    # signals keep running through the network once it exists
    prng = random.Random(5)
    for k in range(22):
        p, q = edges[prng.randrange(len(edges))]
        ts = 4.9 + k * 0.24
        sp = dia.add(Solid("sphere", "#b8ccff", glow=3.0, scale=0.035, pos=tuple(p),
                           casts=False))
        show(sp, ts, 0.08)
        sp.anim(ts, 0.5, inout, pos=tuple(q))
        hide(sp, ts + 0.42, 0.12)

    t = 5.05
    grow(arrows[4], t)
    rise(rep_box, t + 0.1)
    for n_ in rep_lab:
        show(n_, t + 0.25, 0.5)
    for k, c in enumerate(cells):
        show(c, t + 0.35 + 0.08 * k, 0.5, back, s=(1, 1, 0.05))
    show(dots, t + 0.75, 0.4)

    t = 6.05
    grow(arrows[5], t)
    rise(ans_box, t + 0.1)
    show(ans_lab, t + 0.25, 0.5)
    for n_ in (big, of, *verdict):
        show(n_, t + 0.4, 0.6, back, s=0.8)

    # Indicate: the score swells and glows
    big.anim(7.15, 0.3, out, scale=1.15, emit_k=2.4)
    big.anim(7.45, 0.3, inout, scale=1.0, emit_k=1.25)

    # aimed high enough that the stage heads stay clear of the title
    b.cam = Camera(look=(-3.0, Y0 + 0.3, 0), az=-24, el=7, dist=11.8)
    b.cam.anim(0.3, 7.2, inout, look=(0, Y0 - 0.12, 0), az=0, el=5, dist=12.7)
    b.cam.anim(7.5, 2.9, linear, dist=12.35, az=2.5)
    return b


# ── 2b. which two models ────────────────────────────────────────

def models():
    """Two cards, as scenes.models: how each model runs and its list price."""
    b = Beat("Models", 8.5)
    b.env = Env(light_box=((0, 3.0, 0), 9.0), pool=(0.0, 0.0, 9.0))
    title(b, "VLM Choice", dur=1.1)
    specs = [
        ("gemini", "Gemini 3.1 Flash-Lite", STAGE_COLORS[2], -3.45,
         ["closed-weight, called over an API",
          "$0.25 in / $1.50 out per million tokens"]),
        ("qwen", "Qwen3.5-397B-A17B", ALT, 3.45,
         ["open-weight, served via OpenRouter",
          "$0.39 in / $2.34 out per million tokens"]),
    ]
    for i, (slug, name, col, x, body) in enumerate(specs):
        side = -1 if x < 0 else 1
        card = b.add(Group(pos=P(x, -0.55, 0)))
        slab = card.add(glass(6.2, 4.3, col, d=0.12, r=0.18, sw=0.026))
        logo = card.add(Picture(os.path.join(LOGOS, f"{slug}.png"), 0.95,
                                pos=(0, 1.2, 0.32)))
        logo.motion = (lambda ph: lambda t: (0, 0.035 * math.sin(1.3 * t + ph), 0))(i * 1.7)
        head = card.add(Label(name, 30, col, pos=(0, 0.18, 0.08)))
        k = min(1.0, 5.4 / head.width)
        head.scale.v0 = head.scale.v0 * k
        lines = [card.add(Label(s, 22, INK, pos=(0, y, 0.08)))
                 for s, y in zip(body, (-0.58, -1.06))]
        kb = min(1.0, 5.5 / max(l_.width for l_ in lines))
        for l_ in lines:
            l_.scale.v0 = l_.scale.v0 * kb

        t = 1.2 + 0.14 * i
        show(card, t, 1.3, outq, r=(0, side * 62, 0), d=(side * 1.2, 0, -1.6))
        show(logo, t + 0.35, 0.9, back, s=0.55, r=(0, side * -40, 0))
        head.set(0, reveal=0.0).anim(2.5, 0.9, inout, reveal=1.0)
        show(head, 2.5, 0.4)
        for k_, l_ in enumerate(lines):
            show(l_, 3.4 + 0.1 * k_, 0.9, out, d=(0, -0.15, 0))

    b.cam = Camera(look=P(0, -0.35), az=10, el=9, dist=13.6)
    b.cam.anim(0.0, 8.5, inout, az=-4, el=4, dist=12.5)
    return b


# ── 2c. what the models are actually asked ────────────────────────

def prompts():
    """
    Appendix A, word for word, set as two cards angled like an open book.
    Highlights come up in the same three steps as scenes.prompts.
    """
    b = Beat("Prompts", 8.0)
    b.env = Env(light_box=((0, 3.0, 0), 9.0), pool=(0.0, 0.0, 9.0), bloom=0.4)
    title(b, "VLM Prompts", dur=0.7)

    COLS, CARD_W = 56, 6.75
    adv = (CARD_W - 0.5) / COLS
    em = adv / (font("mono", 400).getlength("M") / 400)
    size = em * cap_ratio("mono") / CAP_PER_PT
    SPACING = 1.25
    shared = ["You are an expert astronomer", "scientifically interesting"]
    specs = [("Likert prompt", PICK, -3.5, LIKERT_PROMPT,
              ["assign a Score from 1 to 5"], 1),
             ("Tournament prompt", ALT, 3.5, TOURNAMENT_PROMPT,
              ["keep for human inspection", "Do not force a winner."], -1)]

    cards = []
    for name, colour, x, items, own, side in specs:
        lines = wrap_prompt(items, COLS, shared + own)
        block = Label("\n".join(lines), size, INK, face="mono", anchor="tl",
                      spacing=SPACING)
        cards.append(dict(name=name, colour=colour, x=x, lines=lines,
                          block=block, own=own, side=side))

    lh = em * SPACING
    capw = cap(size)
    body_h = max(capw + (len(c["lines"]) - 1) * lh + 0.28 * em for c in cards)
    CARD_H = body_h + 1.35
    top = 2.45
    # the real per-character advance after the text size is rounded to pixels
    u = cards[0]["block"].u
    adv_real = font("mono", int(round(cards[0]["block"].px))).getlength("M") * u

    bands = {"role": [], "criterion": [], "own": []}
    for c in cards:
        card = b.add(Group(pos=P(c["x"], top - CARD_H / 2, 0),
                           rot=(0, c["side"] * 7, 0)))
        card.add(glass(CARD_W, CARD_H, c["colour"], d=0.1, r=0.16, sw=0.024))
        head = card.add(Label(c["name"], 26, c["colour"],
                              pos=(0, CARD_H / 2 - 0.45, 0.07)))
        bx, by = -CARD_W / 2 + 0.25, CARD_H / 2 - 0.99
        c["block"].pos.v0 = np.array([bx, by, 0.075])
        card.add(c["block"])
        c["block"].wipe_down = True

        text = "\n".join(c["lines"])

        def band(phrase, colour):
            at = text.find(phrase)
            line = text[:at].count("\n")
            col = at - (text.rfind("\n", 0, at) + 1)
            w = len(phrase) * adv_real + 0.14
            cx = bx + col * adv_real + len(phrase) * adv_real / 2
            cy = by - line * lh - capw * 0.5 - em * 0.06
            return card.add(Slab(w, em * 1.28, 0.01, 0.03, color=colour, glow=0.25,
                                 alpha=0.0, casts=False, pos=(cx, cy, 0.062)))

        bands["role"].append(band(shared[0], WARM))
        bands["criterion"].append(band(shared[1], WARM))
        bands["own"] += [band(p, c["colour"]) for p in c["own"]]

        t = 0.75 + (0 if c["side"] > 0 else 0.1)
        show(card, t, 0.8, outq, d=(0, 0, -1.0), r=(0, c["side"] * 14, 0))
        show(head, t + 0.2, 0.4)
        c["block"].set(0, reveal=0.0)
        c["block"].anim(1.35 + (0 if c["side"] > 0 else 0.2), 1.1, inout, reveal=1.0)

    for key, t, a in (("role", 2.95, 0.42), ("criterion", 3.85, 0.42),
                      ("own", 4.75, 0.38)):
        for bd in bands[key]:
            bd.anim(t, 0.5, out, alpha=a)

    b.cam = Camera(look=P(0, 0.05), az=3, el=4, dist=12.9)
    b.cam.anim(0.0, 8.0, inout, az=-2, dist=12.25)
    return b


# ── 3. the two formats ──────────────────────────────────────────

def format_card(b, name, color, points, t, x=-3.07, width_=5.8):
    """scenes.format_card in glass: the format's name and four bullets."""
    card = b.add(Group(pos=P(x, -0.35, 0)))
    card.add(glass(width_, 4.9, color, d=0.1, r=0.18, sw=0.024))
    head = card.add(Label(name, 36, color, pos=(0, 2.45 - 0.52, 0.07)))
    items = []
    y = 1.93 - cap(36) / 2 - 0.45
    left = -width_ / 2 + 0.52
    pitch = cap(21) + 0.16
    for lines in points:
        k = min(1.0, (width_ - 1.0) / max(width(s, 21) for s in lines))
        grp = card.add(Group(pos=(0, 0, 0)))
        first = y - cap(21) / 2
        for j, s in enumerate(lines):
            lab = grp.add(Label(s, 21, INK, anchor="l",
                                pos=(left, first - j * pitch, 0.07)))
            lab.scale.v0 = lab.scale.v0 * k
        grp.add(Solid("sphere", color, glow=0.8, scale=0.055,
                      pos=(left - 0.28, first, 0.07), casts=False))
        items.append(grp)
        y -= cap(21) + (len(lines) - 1) * pitch + 0.06 + 0.32
    show(card, t, 1.0, outq, d=(0, 0, -1.2), r=(0, -30, 0))
    head.set(0, reveal=0.0).anim(t + 0.15, 0.85, inout, reveal=1.0)
    for g in items:
        show(g, t + 1.0, 0.7, out, d=(-0.25, 0, 0))
    return card


def readout_row(parts, cx, y, gap):
    """x centres for labels laid left to right with gaps, centred on cx."""
    widths = [w for w, _ in parts]
    total = sum(widths) + sum(gap[:len(widths) - 1])
    x = cx - total / 2
    out_ = []
    for i, w in enumerate(widths):
        out_.append((x, x + w))
        x += w + (gap[i] if i < len(gap) else 0)
    return out_


def formats():
    """
    scenes.formats: Likert on the left with its grid scored three rounds on
    the right, then the Tournament the same way. In depth, picked tiles lift
    toward the camera and a reshuffle deals the grid like cards.
    """
    b = Beat("Formats", 25.7, fade_out=0.55)
    b.env = Env(light_box=((0, 3.0, 0), 10.0), pool=(0.0, 0.0, 9.0))
    title(b, "Scoring Formats", dur=0.9)

    # Likert ------------------------------------------------------------
    t0 = 0.9
    lcard = format_card(b, "Likert", PICK, [
        ["4×4 grid — sixteen images per call"],
        ["the model picks the interesting", "ones, and scores each 1–5"],
        ["unselected images score 0;", "every image is seen ten times"],
        ["total 0–50  ·  51 possible scores"],
    ], t0)

    rng = random.Random(3)
    GX, GY, CELL = 3.4, -0.15, 1.18
    names = [LENS] + D.high[1:3] + D.mid[:4] + D.low[:9]

    def cell(i, z=0.0):
        return P(GX + (i % 4 - 1.5) * (CELL + 0.14), GY + (1.5 - i // 4) * (CELL + 0.14), z)

    def rest(nm):
        return (WARM, 0.026, 1.6) if nm == LENS else (DIM, 0.014, 1.0)

    rng.shuffle(names)
    tiles = {}
    for i, nm in enumerate(names):
        col, sw, si = rest(nm)
        tl = b.add(tile(nm, CELL, cell(i), stroke=col, sw=sw, si=si))
        show(tl, 2.6 + 0.035 * i, 0.7, outq, d=(0, -0.3, -1.2), r=(45, 0, 0))
        tiles[nm] = tl

    rnd = b.add(Group())
    rtrack = Track(1.0)
    ttrack = Track(0.0)
    # readout under the grid: "round N of 10"   "this one X / 50"
    parts = [(width("round", 24), None), (width("10", 24), None),
             (width("of 10", 24), None), (width("this one", 24), None),
             (width("50", 28), None), (width("/ 50", 28), None)]
    spans = readout_row(parts, GX, GY - 3.0, [0.32, 0.32, 0.8, 0.5, 0.16])
    ry = Y0 + GY - 3.0
    labs = [
        Label("round", 24, DIM, anchor="l", pos=(spans[0][0], ry, 0.05)),
        Label(lambda t: str(int(rtrack(t))), 24, DIM, pos=((spans[1][0] + spans[1][1]) / 2, ry, 0.05)),
        Label("of 10", 24, DIM, anchor="l", pos=(spans[2][0], ry, 0.05)),
        Label("this one", 24, WARM, anchor="l", pos=(spans[3][0], ry, 0.05)),
        Label(lambda t: str(int(ttrack(t))), 28, WARM, anchor="r", pos=(spans[4][1], ry, 0.05)),
        Label("/ 50", 28, WARM, anchor="l", pos=(spans[5][0], ry, 0.05)),
    ]
    for l_ in labs:
        rnd.add(l_)
    show(rnd, 3.4, 0.5, d=(0, -0.1, 0))

    t = 3.9
    ROUNDS = 3
    for r in range(1, ROUNDS + 1):
        picked = [nm for nm in names if round(D.score(nm) / 10) > 0]
        chips = []
        dur = 0.8 if r == 1 else 0.55
        for k, nm in enumerate(picked):
            i = names.index(nm)
            tl = tiles[nm]
            tk = t + k * 0.06
            tl.anim(tk, 0.45, outq, pos=cell(i, 0.24))
            if nm != LENS:
                tl.anim(tk, 0.3, out, stroke=PICK, stroke_w=0.026, stroke_i=1.5)
            c = b.add(chip(round(D.score(nm) / 10), PICK, 0.42,
                           np.array(cell(i, 0.24 + 0.14)) + (0.33, 0.33, 0), 24))
            show(c, tk, 0.5, back, s=0.25, r=(0, 120, 0))
            chips.append(c)
        t += dur
        ttrack.to(t, 0.35, r * 5, linear)
        t += 0.35 + (0.3 if r == 1 else 0.12)
        for c in chips:
            hide(c, t, 0.3, inn, s=0.4, d=(0, 0, 0.3))
        t += 0.3
        if r < ROUNDS:
            rng.shuffle(names)
            lift = random.Random(r)
            for i, nm in enumerate(names):
                tl = tiles[nm]
                col, sw, si = rest(nm)
                tl.anim(t, 0.8, inout, lift=(0, 0, 0.5 + 0.9 * lift.random()),
                        pos=cell(i))
                wob = lift.uniform(-28, 28)
                tl.anim(t, 0.4, out, rot=(0, wob, 0))
                tl.anim(t + 0.4, 0.4, inout, rot=(0, 0, 0))
                tl.anim(t, 0.5, inout, stroke=col, stroke_w=sw, stroke_i=si)
            rtrack.to(t, 0.8, r + 1, linear)
            t += 0.8
    t += 1.1
    t_out = t                                     # 10.99 on the 2D clock
    for n_ in [lcard, rnd, *tiles.values()]:
        hide(n_, t_out, 0.6, inn, d=(0, -0.2, -0.6))

    # Tournament ---------------------------------------------------------
    t = t_out + 0.6
    tcard = format_card(b, "Tournament", ALT, [
        ["2×2 grid — four images per call"],
        ["keep or reject, nothing else —", "no score, no reasoning"],
        ["rejects leave the pool;", "survivors reshuffle and go again"],
        ["score = rounds survived", "0–10  ·  11 possible scores"],
    ], t)
    t += 1.7
    GXT, CT = 3.4, 1.9
    colsx, rowsy = [GXT - 1.07, GXT + 1.07], [1.15, -1.45]

    def seat(i, z=0.0):
        return P(colsx[i % 2], rowsy[i // 2], z)

    rtr = b.add(Group())
    rtrack2 = Track(1.0)
    parts = [(width("round", 24), None), (width("10", 24), None),
             (width("of 10", 24), None)]
    # a little lower than the 2D film: seen in depth, the lifted tiles'
    # keep / reject marks otherwise crowd it
    spans = readout_row(parts, GXT, -3.3, [0.32, 0.32])
    ry = Y0 - 3.3
    rtr.add(Label("round", 24, DIM, anchor="l", pos=(spans[0][0], ry, 0.05)))
    rtr.add(Label(lambda t_: str(int(rtrack2(t_))), 24, DIM,
                  pos=((spans[1][0] + spans[1][1]) / 2, ry, 0.05)))
    rtr.add(Label("of 10", 24, DIM, anchor="l", pos=(spans[2][0], ry, 0.05)))

    seats = []
    for i, nm in enumerate([LENS, D.high[2], D.low[0], D.low[1]]):
        tl = b.add(tile(nm, CT, seat(i), sw=0.016))
        show(tl, t + 0.05 * i, 0.9, outq, d=(0, -0.3, -1.2), r=(45, 0, 0))
        seats.append(tl)
    show(rtr, t, 0.9, d=(0, -0.1, 0))
    t += 0.9

    KEEPS = [[True, True, False, False],
             [True, False, True, False],
             [True, True, False, False]]
    REFILL = [[D.high[4], D.mid[1]], [D.mid[2], D.low[2]]]
    for r, keeps in enumerate(KEEPS, start=1):
        marks = []
        dur = 1.2 if r == 1 else 0.85
        for i, keep in enumerate(keeps):
            tk = t + i * 0.2 * dur / 1.2
            col = PICK if keep else DROP
            tl = seats[i]
            tl.anim(tk, 0.35, out, stroke=col, stroke_w=0.03, stroke_i=1.6)
            if keep:
                tl.anim(tk, 0.45, outq, pos=seat(i, 0.22))
            m = b.add(Label("keep" if keep else "reject", 20, col,
                            pos=seat(i, 0.3 if keep else 0.08)))
            m.pos.v0 = m.pos.v0 + np.array([0, -CT / 2 - 0.1 - cap(20) / 2 - 0.04, 0])
            show(m, tk, 0.5, back, s=1.2)
            marks.append(m)
        t += dur + (0.5 if r == 1 else 0.3)
        out_ = [i for i, keep in enumerate(keeps) if not keep]
        # rejects tip over backwards and drop out of the pool
        for i in out_:
            hide(seats[i], t, 0.8, inn, r=(-80, 0, 0), d=(0, -0.7, -0.9))
            hide(marks[i], t, 0.5, inn)
        t += 0.8
        for i, keep in enumerate(keeps):
            if keep:
                hide(marks[i], t, 0.35)
        t += 0.35
        if r == len(KEEPS):
            break
        for i, keep in enumerate(keeps):
            if keep:
                seats[i].anim(t, 0.6, inout, stroke=DIM, stroke_w=0.016,
                              stroke_i=1.0, pos=seat(i))
        for slot, nm in zip(out_, REFILL[r - 1]):
            tl = b.add(tile(nm, CT, seat(slot), sw=0.016))
            show(tl, t + 0.08 * slot, 1.0, outq, d=(4.5, 0.4, -2.5), r=(0, -50, 0))
            seats[slot] = tl
        rtrack2.to(t, 1.0, r + 1, linear)
        t += 1.0
    t += 0.9
    for i, keep in enumerate(KEEPS[-1]):
        if keep:
            hide(seats[i], t, 0.6, inn, d=(0, -0.2, -0.6))
    hide(tcard, t, 0.6, inn, d=(0, -0.2, -0.6))
    hide(rtr, t, 0.6)

    b.cam = Camera(look=P(0.2, -0.3), az=12, el=8, dist=13.3)
    b.cam.anim(0.0, t_out, inout, az=5, el=6, dist=12.9)
    b.cam.anim(t_out, 1.6, inout, az=-9)
    b.cam.anim(t_out + 1.6, b.duration - t_out - 1.6, inout, az=-3, dist=12.6)
    return b


# ── 4. the recall skyline ────────────────────────────────────────

def recall():
    """
    All 167 reference anomalies as a skyline: each stands as tall as its
    Likert score, highest at the back. A glass plane at the cutoff comes
    down from 50 to 33; each top it passes lights, on the 2D film's clock.
    Counts are the real shortlist size and catch (assets/anomalies.json).
    """
    with open(os.path.join(ASSETS, "anomalies.json")) as f:
        data = json.load(f)
    meta, curve = data["tiles"], {int(k): v for k, v in data["curve"].items()}
    order = sorted(meta, key=lambda k: -(meta[k]["likert"] or 0))
    score = np.array([meta[k]["likert"] or 0 for k in order], float)
    n = len(order)

    b = Beat("Recall", 22.5, fade_out=0.75)
    b.env = Env(light_box=((0, 1.5, 0), 11.0), pool=(0.0, 0.0, 10.0), refl=0.3,
                light_dir=(0.4, -1.0, -0.5))
    title(b, "Anomaly Recovery", dur=0.9)
    hud_in(b.overlay(Hud("Recovering 167 previously identified anomalies", 26,
                         DIM, 0, 2.5)), 1.0, 0.7, dy=0)

    COLS, PITCH, TOP = 21, 0.66, 0.58
    col = np.arange(n) % COLS
    row = np.arange(n) // COLS
    x = (col - (COLS - 1) / 2) * PITCH
    zc = (row - (row.max()) / 2) * PITCH

    def height(s):
        return 0.25 + (np.asarray(s, float) - 20) * 0.085

    h = height(score)

    # the cutoff's clock: 50 first, then down a point at a time to 33
    FIRST = 50
    steps = [(FIRST, 3.9, 0.9)]
    t = 3.9 + 0.9 + 0.8
    for c in range(FIRST - 1, 32, -1):
        dur = 0.9 if c == 33 else 0.75 if c > 47 else 0.5
        steps.append((c, t, dur))
        t += dur + (0.4 if c == FIRST - 1 else 0)
    t_end = t
    lit = np.full(n, np.inf)
    lit_dur = np.full(n, 0.5)
    for c, ts, dur in steps:
        hit = (score >= c) & ~np.isfinite(lit)
        lit[hit], lit_dur[hit] = ts, dur
    missed = ~np.isfinite(lit)
    t_missed = t_end + 1.3
    t_punch = t_missed + 0.8 + 0.9

    grow_at = 1.7 + col / (COLS - 1) * 1.3 + row * 0.06

    pillar = np.zeros((n, 32), "f4")
    pillar[:, 0], pillar[:, 10], pillar[:, 15] = 0.6, 0.6, 1
    pillar[:, 12], pillar[:, 14] = x, zc
    pillar[:, 16:19] = lin("#0c1220")
    pillar[:, 19] = 1
    pillar[:, 24] = -1
    pillar[:, 26] = 1
    pillar[:, 28] = 1

    def pillars(t):
        g = outq_v(np.clip((t - grow_at) / 0.8, 0, 1))
        r = pillar.copy()
        hh = np.maximum(h * g, 0.002)
        r[:, 5] = hh
        r[:, 13] = hh / 2
        r[:, 28] = np.clip(g * 3, 0, 1)
        return r

    tile_rows = np.zeros((n, 32), "f4")
    # lying flat: the front face turned up (Rx(-90))
    tile_rows[:, 0] = 1
    tile_rows[:, 6] = -1
    tile_rows[:, 9] = 1
    tile_rows[:, 15] = 1
    tile_rows[:, 12], tile_rows[:, 14] = x, zc
    tile_rows[:, 16:19] = lin("#0a0e17")
    tile_rows[:, 19] = 1
    tile_rows[:, 24] = [LAYER["a:" + k] for k in order]
    tile_rows[:, 28] = 1
    dim_c, pick_c, drop_c = lin(DIM), lin(PICK), lin(DROP)

    def tops(t):
        g = outq_v(np.clip((t - grow_at) / 0.8, 0, 1))
        e = np.clip((t - lit) / np.maximum(lit_dur, 0.4), 0, 1)
        e = np.where(np.isfinite(lit), e, 0.0)
        flash = np.sin(np.pi * e)
        r = tile_rows.copy()
        r[:, 13] = h * g + 0.022 + 0.07 * outq_v(e)
        r[:, 28] = np.clip(g * 3 - 1.2, 0, 1)
        r[:, 26] = 0.2 + 0.8 * e + 0.5 * flash
        sc = dim_c[None] * (1 - e[:, None]) + pick_c[None] * e[:, None]
        m = np.clip((t - t_missed) / 0.8, 0, 1) * missed
        sc = sc * (1 - m[:, None]) + drop_c[None] * m[:, None]
        r[:, 20:23] = sc
        r[:, 23] = 0.012 + 0.012 * e + 0.022 * m
        r[:, 27] = 1.0 + 1.8 * flash + 0.4 * e + 1.0 * m * (0.8 + 0.2 * math.sin(t * 5))
        return r

    b.add(Instances(("box", "pillar"), pillars, size=(0.6, 0.6, 0.02),
                    transparent=False, casts=True))
    b.add(Instances(("rbox", TOP, TOP, 0.04, 0.03), tops, size=(TOP, TOP, 0.03),
                    transparent=False, casts=False))

    # the cutoff plane
    plane = b.add(Slab(COLS * PITCH + 1.2, (row.max() + 1) * PITCH + 1.0, 0.03,
                       0.25, color=PICK, glass=True, stroke=PICK, stroke_w=0.035,
                       stroke_i=1.3, fill=0.42, rot=(-90, 0, 0),
                       pos=(0, float(height(FIRST)) - 0.05, 0)))
    show(plane, 3.9, 0.9, outq, d=(0, 0.8, 0))
    for c, ts, dur in steps[1:]:
        plane.anim(ts, dur, inout, pos=(0, float(height(c)) - 0.05, 0))

    # read-outs, where scenes.recall puts them
    cut_t = Track(FIRST)
    sel_t = Track(curve[FIRST][0])
    found_t = Track(curve[FIRST][1])
    for c, ts, dur in steps[1:]:
        cut_t.to(ts, dur, c, linear)
        sel_t.to(ts, dur, curve[c][0], linear)
        found_t.to(ts, dur, curve[c][1], linear)

    def comma(v):
        return f"{int(v):,}"

    wl, wn = width("Cutoff Score", 26), width("50", 52)
    x0 = -(wl + 0.4 + wn) / 2
    rows_hud = [
        Hud("Cutoff Score", 26, DIM, x0, 1.8, anchor="l"),
        Hud(lambda t: str(int(round(float(cut_t(t))))), 52, PICK, x0 + wl + 0.4, 1.8,
            anchor="l"),
        Hud("Images to review", 24, DIM, -4.6, -3.55),
        Hud(lambda t: comma(sel_t(t)), 30, INK,
            -4.6 + width("Images to review", 24) / 2 + 0.32, -3.55, anchor="l"),
        Hud("Anomalies found", 24, DIM, 1.5, -3.55),
        Hud(lambda t: comma(found_t(t)), 30, PICK,
            1.5 + width("Anomalies found", 24) / 2 + 0.32, -3.55, anchor="l"),
        Hud("of 167", 24, DIM, 4.9, -3.55),
    ]
    for hh_ in rows_hud:
        hud_in(b.overlay(hh_), 3.9, 0.9, dy=0)
    hud_in(b.overlay(Hud("Top ~10% of the dataset contains ~98% of anomalies.", 30,
                         INK, 0, -2.9)), t_punch, 0.9, dy=-0.12)

    b.cam = Camera(look=(0, 1.95, 0.3), az=-17, el=31, dist=20.5)
    b.cam.anim(2.0, t_end - 2.0, inout, az=12)
    b.cam.anim(t_end, b.duration - t_end, inout, az=6, el=33, dist=20.0)
    return b


def outq_v(u):
    return 1 - (1 - u) ** 5


# ── 5. table 1 ──────────────────────────────────────────────

def results():
    """
    Table 1 as scenes.results sets it, each row a glass strip that swings
    up into place, then the two call-outs.
    """
    b = Beat("Results", 12.9, fade_out=0.6)
    b.env = Env(light_box=((0, 3.0, 0), 10.0), pool=(0.0, 0.0, 9.0))
    title(b, "Results", dur=1.0)
    COLS = [(-7.4, "l"), (-3.4, "l"), (-0.9, "l"), (2.6, "r"), (4.5, "r"), (6.6, "r")]
    GUTTER = 7.0
    HEADERS = ["Model", "Method", "Threshold", "Selected", "Recall", "$ / 10,000"]
    ROWS = [("Gemini 3.1 Flash-Lite", STAGE_COLORS[2], "Likert", "33/50",
             "2,095", "98.2%", "$5.05"),
            ("Qwen3.5-397B-A17B", ALT, "Likert", "36/50",
             "1,984", "95.8%", "$6.69"),
            ("Gemini 3.1 Flash-Lite", STAGE_COLORS[2], "Tournament",
             "Round 10", "2,019", "95.8%", "$3.66"),
            ("Qwen3.5-397B-A17B", ALT, "Tournament", "Round 10",
             "2,099", "93.4%", "$1.80")]
    Y = [1.35, 0.0, -1.35, -2.70]

    # the whole table, gutter included, scaled to 13.2 wide under the title
    right = GUTTER + max(width("best recall", 22), width("cheapest", 22))
    left = -7.5
    k = 12.8 / (right - left)
    xc = (left + right) / 2
    top = 2.8 + cap(23) / 2
    tab = b.add(Group(pos=(-xc * k, Y0 + (3.5 - cap(46)) - 0.75 - top * k, 0),
                      scale=k))

    header = []
    for text, (x, e) in zip(HEADERS, COLS):
        header.append(tab.add(Label(text, 23, DIM, anchor=e, pos=(x, 2.8, 0.03))))
    rule = tab.add(Rod((-7.5, 2.25, 0), (6.7, 2.25, 0), radius=0.012, color=DIM,
                       glow=0.25, alpha=0.8))

    rows, cellmap = [], {}
    for i, ((name, colr, method, thr, n_, rec, cost), y) in enumerate(zip(ROWS, Y)):
        g = tab.add(Group(pos=(0, y, 0)))
        g.add(Slab(14.6, 1.02, 0.07, 0.12, color=colr, glass=True, stroke=colr,
                   stroke_w=0.012, stroke_i=0.9, fill=0.7, pos=(-0.45, 0, -0.05)))
        vals = [(name, colr), (method, INK), (thr, INK), (n_, INK), (rec, INK),
                (cost, INK)]
        for j, ((text, c), (x, e)) in enumerate(zip(vals, COLS)):
            cellmap[(i, j)] = g.add(Label(text, 26, c, anchor=e, pos=(x, 0, 0.03)))
        rows.append(g)

    def flag(i, j, text, t):
        lab = cellmap[(i, j)]
        x0, y0, x1, y1 = lab.local_box()
        ox, oy, _ = lab.pos.v0
        w, hgt = x1 - x0 + 0.28, (y1 - y0) + 0.28 + 0.06
        frame = tab.add(Slab(w, hgt, 0.04, 0.1, color=PICK, fill=0.0, stroke=PICK,
                             stroke_w=0.035, stroke_i=1.6, casts=False,
                             pos=(ox + (x0 + x1) / 2, Y[i] + oy + (y0 + y1) / 2 - 0.03,
                                  0.08)))
        show(frame, t, 0.9, outq, s=1.25)
        note = tab.add(Label(text, 22, PICK, anchor="l", pos=(GUTTER, Y[i], 0.03)))
        show(note, t + 0.1, 0.8, out, d=(0.4, 0, 0))
        lab.anim(t, 0.5, out, emit_k=1.5)

    for hd in header:
        show(hd, 1.1, 1.0, d=(0, 0.1, 0))
    show(rule, 1.1, 0.4)
    rule.grow.to(1.1, 1.0, 1.0, inout)
    for i, g in enumerate(rows):
        show(g, 2.1 + 0.8 * i, 0.85, outq, r=(-70, 0, 0), d=(0, -0.2, -0.8))
    flag(0, 4, "best recall", 7.1)
    flag(3, 5, "cheapest", 8.0)

    b.cam = Camera(look=P(0, -0.25), az=22, el=-5, dist=13.6)
    b.cam.anim(0.0, 5.3, inout, az=0, el=3, dist=12.5)
    b.cam.anim(5.3, 7.6, inout, az=-3, el=4, dist=12.15)
    return b


# ── 6. what the top of the list contains ───────────────────────────

def funnel():
    """
    All 20,000 scored images as a carpet of chips, narrowed as in
    scenes.funnel: 204 at 45+/50 rise, the 102 not previously identified
    turn green, and those fly up to become the wall of Appendix B's 30.
    """
    with open(os.path.join(ASSETS, "candidates.json")) as f:
        data = json.load(f)
    meta, f_ = data["tiles"], data["funnel"]
    order = sorted(meta, key=lambda k: -(meta[k]["score"] or 0))

    b = Beat("Funnel", 15.0, fade_out=0.7)
    b.env = Env(light_box=((1.0, 1.5, 0.5), 10.0), pool=(1.5, 0.5, 9.0), refl=0.45,
                light_dir=(0.35, -1.0, -0.5))
    title(b, "New Anomaly Candidates", dur=1.0)

    rng = np.random.default_rng(12)
    NX, NZ, PITCH = 200, 100, 0.062
    total = NX * NZ
    assert total == f_["scored"], (total, f_["scored"])
    cx, cz = 3.9, 0.7
    ix, iz = np.meshgrid(np.arange(NX), np.arange(NZ))
    x = cx + (ix.ravel() - (NX - 1) / 2) * PITCH
    z = cz + (iz.ravel() - (NZ - 1) / 2) * PITCH
    sel = rng.choice(total, f_["above"], replace=False)
    new = sel[: f_["released"]]
    known = sel[f_["released"]:]
    is_sel = np.zeros(total, bool)
    is_sel[sel] = True
    is_new = np.zeros(total, bool)
    is_new[new] = True
    is_known = is_sel & ~is_new

    grey = 0.004 + 0.045 * rng.random(total) ** 2.5
    base_c = np.stack([grey * 0.92, grey * 0.97, grey * 1.12], 1)
    ink_c, pick_c, dim_c = lin(INK), lin(PICK), lin(DIM) * 0.35
    appear = 1.1 + (x - x.min()) / (x.max() - x.min()) * 1.0 + rng.uniform(0, 0.15, total)
    T2, T3 = 2.3, 3.4
    stag = rng.uniform(0, 0.3, total)
    T_FLY = 8.3

    carpet = np.zeros((total, 32), "f4")
    carpet[:, 0], carpet[:, 5], carpet[:, 10], carpet[:, 15] = 0.048, 0.015, 0.048, 1
    carpet[:, 12], carpet[:, 14] = x, z
    carpet[:, 19] = 1
    carpet[:, 24] = -1
    carpet[:, 26] = 1
    carpet[:, 28] = 1

    def chips_state(t):
        a = outq_v(np.clip((t - appear) / 0.5, 0, 1))
        e2 = inout_v(np.clip((t - T2 - stag) / 0.8, 0, 1))
        e3 = inout_v(np.clip((t - T3 - stag) / 0.9, 0, 1))
        y = 0.0075 + 0.5 * (1 - a)
        col = base_c.copy()
        glow = np.full(total, 1.1)
        # step 2: the 204 rise and brighten, the rest dim
        dimk = 1 - 0.6 * e2
        col = col * np.where(is_sel, 1.0, dimk)[:, None]
        s2 = is_sel * e2
        col = col * (1 - s2[:, None]) + ink_c[None] * 0.5 * s2[:, None]
        y = y + 0.35 * s2
        glow = glow + 1.2 * s2
        # step 3: the 102 new ones turn green; the known ones settle back
        s3n, s3k = is_new * e3, is_known * e3
        col = col * (1 - s3n[:, None]) + pick_c[None] * 0.6 * s3n[:, None]
        col = col * (1 - s3k[:, None]) + dim_c[None] * s3k[:, None]
        y = y + 0.28 * s3n - 0.33 * s3k
        glow = glow + 0.8 * s3n - 1.9 * s3k
        return a, y, col, glow

    def carpet_rows(t):
        a, y, col, glow = chips_state(t)
        r = carpet.copy()
        r[:, 13] = y
        r[:, 16:19] = col
        r[:, 25] = glow
        r[:, 28] = a * ~(is_new & (t >= fly_start))
        dimmer = 1 - 0.55 * inout(ramp(t, T_FLY, 1.5))
        r[:, 25] *= dimmer
        r[:, 16:19] *= dimmer
        return r

    b.add(Instances(("box", "carpet"), carpet_rows, size=(0.048, 0.048, 0.01),
                    transparent=False))

    # the flight: 30 to Appendix B's wall, the rest up and out like sparks
    WCOLS, WPITCH, WCELL = 10, 1.30, 1.20
    wx0 = -(WCOLS - 1) / 2 * WPITCH
    slots = [np.array(P(wx0 + (i % WCOLS) * WPITCH, 0.55 - (i // WCOLS) * WPITCH, 0))
             for i in range(len(order))]
    fly_start = np.full(total, np.inf)
    dest = np.zeros((total, 3))
    arrive = {}
    frng = random.Random(4)
    new_list = list(new)
    frng.shuffle(new_list)
    to_wall = new_list[: len(order)]
    sparks = new_list[len(order):]
    for k, idx in enumerate(to_wall):
        fly_start[idx] = T_FLY + 0.2 + k * 0.05
        dest[idx] = slots[k]
        arrive[k] = fly_start[idx] + 1.1
    for idx in sparks:
        fly_start[idx] = T_FLY + frng.uniform(0.0, 0.9)
        dest[idx] = (frng.uniform(-8, 8), frng.uniform(5.5, 9.0), frng.uniform(-4, 0))
    flying = np.array(new_list)
    is_wall = np.zeros(total, bool)
    is_wall[to_wall] = True

    def flight_rows(t):
        s = fly_start[flying]
        u = np.clip((t - s) / 1.1, 0, 1)
        live = (t >= s) & (u < 1)
        if not live.any():
            return None
        idx = flying[live]
        e = inout_v(u[live])
        p0 = np.stack([x[idx], np.full(len(idx), 0.65), z[idx]], 1)
        p2 = dest[idx]
        p1 = (p0 + p2) / 2 + np.array([0, 2.2, 1.2])
        p = ((1 - e) ** 2)[:, None] * p0 + (2 * (1 - e) * e)[:, None] * p1 + (e ** 2)[:, None] * p2
        wall = is_wall[idx]
        sz = np.where(wall, 0.05 + 0.25 * e ** 3, 0.05)
        r = np.zeros((len(idx), 32), "f4")
        r[:, 0], r[:, 5], r[:, 10], r[:, 15] = sz, sz, sz, 1
        r[:, 12:15] = p
        r[:, 16:19] = pick_c * 0.7
        r[:, 19] = 1
        r[:, 24] = -1
        r[:, 25] = 2.2
        r[:, 26] = 1
        r[:, 28] = np.where(wall, 1 - np.clip((e - 0.85) / 0.15, 0, 1),
                            1 - np.clip((e - 0.3) / 0.7, 0, 1))
        return r

    b.add(Instances(("box", "flight"), flight_rows, size=(1, 1, 0.1)))

    wall = []
    for k, name in enumerate(order):
        tl = b.add(tile("c:" + name, WCELL, tuple(slots[k]), sw=0.014))
        show(tl, arrive[k] - 0.1, 0.6, back, s=0.08, r=(0, 90, 0))
        wall.append(tl)

    # the three counts, down the left
    steps = [(f_["scored"], "images scored by VLMs"),
             (f_["above"], "images scored 45+/50"),
             (f_["released"], "non-previously identified anomalies")]
    XL = -6.4
    huds = []
    times = [(1.1, 1.2), (2.3, 1.1), (3.4, 1.3)]
    for i, ((num, label), (ts, dur)) in enumerate(zip(steps, times)):
        yn = 1.45 - 1.45 * i
        colr = PICK if i == len(steps) - 1 else INK
        hn = b.overlay(Hud(f"{num:,}", 38, colr, XL, yn, anchor="l"))
        hl = b.overlay(Hud(label, 25, DIM, XL, yn - 0.55, anchor="l"))
        hud_in(hn, ts, dur, dy=0.2)
        hud_in(hl, ts + 0.1, dur, dy=0.2)
        huds.append((hn, hl))
    links = []
    for i in range(len(steps) - 1):
        y_top = 1.45 - 1.45 * i - 0.55 - cap(25) / 2 - 0.12
        y_bot = 1.45 - 1.45 * (i + 1) + cap(38) / 2 + 0.12
        ln = b.overlay(HudRule(y_top - y_bot, XL - 0.35, (y_top + y_bot) / 2, DIM,
                               vertical=True))
        ln.set(0, reveal=1.0)
        ln.anim(times[i + 1][0], 0.6, out, alpha=0.5)
        links.append(ln)

    # the last count becomes the heading
    t = 6.5
    for hn, hl in huds[:-1]:
        hud_out(hn, t, 0.5)
        hud_out(hl, t, 0.5)
    for ln in links:
        hud_out(ln, t, 0.5)
    t += 0.5
    hn, hl = huds[-1]
    wnum = width(f"{f_['released']:,}", 36)
    wlab = width("candidates for further investigation", 28)
    hx = -(wnum + 0.35 + wlab) / 2
    hn.anim(t, 0.9, inout, pos=(hx, 2.3), scale=36 / 38)
    hud_out(hl, t, 0.6, dy=0.3)
    hud_in(b.overlay(Hud("candidates for further investigation", 28, DIM,
                         hx + wnum + 0.35, 2.3, anchor="l")), t, 0.9, dy=-0.3)
    t += 0.9
    hud_in(b.overlay(Hud("Below are the 30 highest-scoring candidates", 26, DIM,
                         0, 1.65)), t, 0.6, dy=0)

    b.cam = Camera(look=(2.3, 0.0, 0.75), az=0, el=52, dist=11.8)
    b.cam.anim(0.0, 6.6, inout, az=-5, el=47, dist=11.1)
    b.cam.anim(6.8, 2.6, inout, look=P(0, 0.05), az=0, el=3, dist=12.6)
    b.cam.anim(9.4, b.duration - 9.4, inout, az=2, dist=12.3)
    return b


def inout_v(u):
    return np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2)


# ── driver ────────────────────────────────────────────────────

BEATS = {"Opening": opening, "VLM": vlm, "Models": models, "Prompts": prompts,
         "Formats": formats, "Recall": recall, "Results": results,
         "Funnel": funnel}


def render_one(name):
    beat = BEATS[name]()
    r = Renderer(IMAGES)
    frames = int(round(beat.duration * FPS))
    path = os.path.join(OUT, f"{name}.mp4")
    enc = encoder(path)
    t0 = time.time()
    for f in range(frames):
        enc.stdin.write(r.frame(beat, f / FPS))
        if f % 120 == 0:
            print(f"{name}: {f}/{frames}  {time.time() - t0:.0f}s", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"{name}: done, {frames} frames in {time.time() - t0:.0f}s -> {path}",
          flush=True)


def stills(name, times, where):
    beat = BEATS[name]()
    r = Renderer(IMAGES)
    os.makedirs(where, exist_ok=True)
    for t in times:
        p = os.path.join(where, f"{name}_{t:05.2f}.png")
        still(r, beat, t, p)
        print(p, flush=True)


def film(names):
    listing = os.path.join(OUT, "concat.txt")
    with open(listing, "w") as f:
        for n in names:
            f.write(f"file '{os.path.join(OUT, n + '.mp4')}'\n")
    path = os.path.join(OUT, "Explainer3D.mp4")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-c", "copy", "-movflags", "+faststart", path],
                   check=True)
    os.remove(listing)
    print("film ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("beats", nargs="*", help="beat names (default: all)")
    ap.add_argument("--still", nargs="+", type=float, help="render PNGs at times")
    ap.add_argument("--where", default=os.path.join(OUT, "stills"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=10,
                    help="llvmpipe threads per render")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        render_one(a.one)
        return
    names = a.beats or list(BEATS)
    if a.still:
        for n in names:
            stills(n, a.still, a.where)
        return
    env = dict(os.environ, LP_NUM_THREADS=str(a.threads))
    running, queue = [], list(names)
    while queue or running:
        while queue and len(running) < a.jobs:
            n = queue.pop(0)
            running.append((n, subprocess.Popen([sys.executable, __file__, "--one", n],
                                                env=env)))
        time.sleep(1)
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                if item[1].returncode:
                    sys.exit(f"{item[0]} failed")
    if not a.beats:
        film(list(BEATS))


if __name__ == "__main__":
    main()
