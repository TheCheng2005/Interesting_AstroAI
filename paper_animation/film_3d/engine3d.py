"""
A small offscreen OpenGL renderer for explainer3d.py.

It draws rounded slabs (glass panels, image tiles, chips), spheres, rods,
text sprites, a glossy floor with planar reflections, soft shadows from one
key light, a starfield sky, bloom and fog, and pipes frames to ffmpeg.

By default it runs on Mesa's llvmpipe, a software rasteriser, so it needs no
GPU: EGL is pointed at the Mesa vendor library before moderngl loads it. Set
LIBGL_ALWAYS_SOFTWARE and __EGL_VENDOR_LIBRARY_FILENAMES yourself to override.

World units match manim's: a camera 12.3 units back with a 36 degree field
of view frames the same 8-unit-high picture as the 2D film, so layouts from
scenes.py carry over with y shifted up off the floor.
"""

import functools
import math
import os
import shutil
import subprocess

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H, FPS = 1920, 1080, 60
PX_PER_UNIT = H / 8.0            # manim's frame is 8 units high
CAP_PER_PT = 0.009913            # manim Text cap height per font_size unit
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"

FONTS = {
    "regular": "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
    "medium": "/usr/share/fonts/truetype/lato/Lato-Medium.ttf",
    "semibold": "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf",
    "bold": "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "light": "/usr/share/fonts/truetype/lato/Lato-Light.ttf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}


# ── colour and easing ────────────────────────────────────────────────

def srgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def lin(c):
    """Hex or sRGB triple to linear light, which is what the shaders mix in."""
    c = srgb(c) if isinstance(c, str) else np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear(u):
    return u


def inout(u):
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def out(u):
    return 1 - (1 - u) ** 3


def outq(u):
    return 1 - (1 - u) ** 5


def inn(u):
    return u ** 3


def back(u, s=1.6):
    u -= 1
    return 1 + (s + 1) * u ** 3 + s * u ** 2


def smooth(u):
    return u * u * (3 - 2 * u)


# ── animated values ────────────────────────────────────────────────

class Track:
    """
    A value over time: a start value and tween segments, scheduled in
    chronological order. Each segment starts from wherever the track is when
    it begins, so `to` never needs the previous value spelled out.
    """
    __slots__ = ("v0", "segs")

    def __init__(self, v):
        self.v0 = np.asarray(v, float)
        self.segs = []

    def __call__(self, t):
        v = self.v0
        for a, b, f, g, ease, lift in self.segs:
            if t < a:
                break
            if t >= b:
                v = g
                continue
            u = ease((t - a) / (b - a))
            r = f + (g - f) * u
            if lift is not None:
                r = r + lift * math.sin(math.pi * min(max(u, 0), 1))
            return r
        return v

    def to(self, t, dur, v, ease=inout, lift=None):
        f = self(t)
        v = np.broadcast_to(np.asarray(v, float), f.shape).copy()
        self.segs.append((t, t + max(dur, 1e-6), f, v, ease,
                          None if lift is None else np.asarray(lift, float)))
        self.segs.sort(key=lambda s: s[0])
        return self


def T(p):
    m = np.eye(4)
    m[:3, 3] = p
    return m


def S(s):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def R(rot):
    """Degrees (pitch about x, yaw about y, roll about z), applied Ry Rx Rz."""
    x, y, z = np.radians(rot)
    cx, sx, cy, sy, cz, sz = (math.cos(x), math.sin(x), math.cos(y),
                              math.sin(y), math.cos(z), math.sin(z))
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx @ rz
    return m


COLOR_PROPS = {"color", "stroke", "tint"}


class Node:
    """Anything with a transform and an opacity; children inherit both."""

    def __init__(self, pos=(0, 0, 0), rot=(0, 0, 0), scale=1.0, alpha=1.0):
        self.pos = Track(pos)
        self.rot = Track(rot)
        self.scale = Track(np.broadcast_to(np.asarray(scale, float), (3,)))
        self.alpha = Track(alpha)
        self.parent = None
        self.children = []
        self.motion = None      # optional t -> offset, for idle floating
        self._t = None

    def add(self, *kids):
        for k in kids:
            k.parent = self
            self.children.append(k)
        return kids[0] if len(kids) == 1 else kids

    def anim(self, t, dur, ease=inout, lift=None, **props):
        for k, v in props.items():
            if k in COLOR_PROPS and isinstance(v, str):
                v = lin(v)
            getattr(self, k).to(t, dur, v, ease, lift if k == "pos" else None)
        return self

    def set(self, t, **props):
        return self.anim(t, 0, **props)

    def _eval(self, t):
        if self._t == t:
            return
        p = self.pos(t)
        if self.motion is not None:
            p = p + np.asarray(self.motion(t), float)
        m = T(p) @ R(self.rot(t)) @ S(self.scale(t))
        a = float(self.alpha(t))
        if self.parent is not None:
            self.parent._eval(t)
            m = self.parent._m @ m
            a *= self.parent._a
        self._m, self._a, self._t = m, a, t

    def world(self, t):
        self._eval(t)
        return self._m

    def opacity(self, t):
        self._eval(t)
        return self._a

    def emit(self, t, out, cam):
        pass


class Group(Node):
    pass


# ── meshes ─────────────────────────────────────────────────────────

def rbox_mesh(w, h, d, r, seg=6):
    """
    A rounded-rectangle prism, front face at +z. Faces carry uv over the
    whole rectangle and a face flag (+1 front, -1 back, 0 side), which the
    shader uses for images and the stroke band.
    """
    r = max(min(r, w / 2 - 1e-4, h / 2 - 1e-4), 1e-3)
    ring = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                       (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for k in range(seg + 1):
            a = math.radians(a0 + 90 * k / seg)
            ring.append((cx + r * math.cos(a), cy + r * math.sin(a),
                         math.cos(a), math.sin(a)))
    n = len(ring)
    v = []

    def put(x, y, z, nx, ny, nz, face):
        v.append((x, y, z, nx, ny, nz, x / w + 0.5, y / h + 0.5, face))

    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        put(0, 0, d / 2, 0, 0, 1, 1)
        put(a[0], a[1], d / 2, 0, 0, 1, 1)
        put(b[0], b[1], d / 2, 0, 0, 1, 1)
        put(0, 0, -d / 2, 0, 0, -1, -1)
        put(b[0], b[1], -d / 2, 0, 0, -1, -1)
        put(a[0], a[1], -d / 2, 0, 0, -1, -1)
        put(a[0], a[1], d / 2, a[2], a[3], 0, 0)
        put(a[0], a[1], -d / 2, a[2], a[3], 0, 0)
        put(b[0], b[1], d / 2, b[2], b[3], 0, 0)
        put(b[0], b[1], d / 2, b[2], b[3], 0, 0)
        put(a[0], a[1], -d / 2, a[2], a[3], 0, 0)
        put(b[0], b[1], -d / 2, b[2], b[3], 0, 0)
    return np.array(v, "f4")


def box_mesh():
    """Unit cube centred on the origin, flat-shaded, no faces flagged."""
    v = []
    for axis in range(3):
        for sgn in (-1, 1):
            nrm = np.zeros(3)
            nrm[axis] = sgn
            u_ax, v_ax = [a for a in range(3) if a != axis]
            quad = []
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis] = sgn * 0.5
                p[u_ax], p[v_ax] = du * 0.5, dv * 0.5
                quad.append(p)
            tri = [0, 1, 2, 0, 2, 3]
            # keep the winding outward-facing
            if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), nrm) < 0:
                tri = [0, 2, 1, 0, 3, 2]
            face = 1.0 if (axis == 1 and sgn == 1) else 0.0
            for k in tri:
                p = quad[k]
                v.append((*p, *nrm, p[0] + 0.5, 0.5 - p[2], face))
    return np.array(v, "f4")


def sphere_mesh(nu=20, nv=14):
    v = []

    def pt(i, j):
        th, ph = 2 * math.pi * i / nu, math.pi * j / nv
        x, y, z = math.sin(ph) * math.cos(th), math.cos(ph), math.sin(ph) * math.sin(th)
        return (x, y, z, x, y, z, i / nu, j / nv, 0.0)

    for i in range(nu):
        for j in range(nv):
            a, b, c, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
            v += [a, c, b, a, d, c]
    return np.array(v, "f4")


def cyl_mesh(seg=18, cone=False):
    """Unit radius, y from 0 to 1; a cone when `cone`, tip at y=1."""
    v = []
    top_r = 0.0 if cone else 1.0
    slope = 1.0 if cone else 0.0
    for i in range(seg):
        a0, a1 = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        n0 = np.array([c0, slope, s0]) / math.hypot(1, slope)
        n1 = np.array([c1, slope, s1]) / math.hypot(1, slope)
        p = [(c0, 0, s0), (c1, 0, s1), (c1 * top_r, 1, s1 * top_r),
             (c0 * top_r, 1, s0 * top_r)]
        for k, nn in ((0, n0), (2, n1), (1, n1), (0, n0), (3, n0), (2, n1)):
            v.append((*p[k], *nn, 0, 0, 0.0))
        v += [(0, 0, 0, 0, -1, 0, 0, 0, 0.0), (c0, 0, s0, 0, -1, 0, 0, 0, 0.0),
              (c1, 0, s1, 0, -1, 0, 0, 0, 0.0)]
        if not cone:
            v += [(0, 1, 0, 0, 1, 0, 0, 0, 0.0), (c1, 1, s1, 0, 1, 0, 0, 0, 0.0),
                  (c0, 1, s0, 0, 1, 0, 0, 0, 0.0)]
    return np.array(v, "f4")


# ── renderable nodes ─────────────────────────────────────────────────

def inst_row(m, color, alpha, stroke=(0, 0, 0), stroke_w=0.0, layer=-1.0,
             glow=0.0, bright=1.0, stroke_i=1.0, glass=0.0, orient=0.0):
    """One instance: model matrix columns, then four vec4s of material."""
    return np.concatenate([m.T.reshape(16), [color[0], color[1], color[2], 1.0],
                           [stroke[0], stroke[1], stroke[2], stroke_w],
                           [layer, glow, bright, stroke_i],
                           [alpha, glass, orient, 0.0]]).astype("f4")


class Slab(Node):
    """
    A rounded panel with depth. Glass panels are tinted and see-through with
    a bright rim; solid ones are lit like painted card. An `image` shows a
    cutout on the front face, lit like a screen rather than like paint.
    """

    def __init__(self, w, h, d=0.06, r=0.12, color="#1a2233", glass=False,
                 stroke=None, stroke_w=0.0, image=None, bright=1.0, glow=0.0,
                 stroke_i=1.25, casts=None, orient=0, fill=1.0, **kw):
        super().__init__(**kw)
        self.key = ("rbox", round(w, 4), round(h, 4), round(d, 4), round(r, 4))
        self.size = (w, h, r)
        self.color = Track(lin(color))
        self.stroke = Track(lin(stroke) if stroke else lin(color))
        self.stroke_w = Track(stroke_w)
        self.glow = Track(glow)
        self.bright = Track(bright)
        self.stroke_i = Track(stroke_i)
        self.glass = glass
        self.image = image
        self.orient = orient
        self.fill = Track(fill)
        self.casts = (not glass and fill > 0.5) if casts is None else casts

    def emit(self, t, out, cam):
        a = self.opacity(t)
        if a <= 0.003:
            return
        m = self.world(t)
        fill = float(self.fill(t))
        row = inst_row(m, self.color(t), a, self.stroke(t), float(self.stroke_w(t)),
                       -1.0 if self.image is None else float(self.image),
                       float(self.glow(t)), float(self.bright(t)),
                       float(self.stroke_i(t)), 1.0 if self.glass else 0.0,
                       float(self.orient))
        row[19] = fill
        out.mesh(self.key, row, m[:3, 3],
                 transparent=self.glass or a < 0.999 or fill < 0.999,
                 casts=self.casts and a > 0.5, size=self.size)


class Solid(Node):
    """A sphere, a cube or a cone, sized by `scale`."""

    def __init__(self, kind, color="#ffffff", glow=0.0, casts=True, **kw):
        super().__init__(**kw)
        self.key = (kind,)
        self.color = Track(lin(color))
        self.glow = Track(glow)
        self.casts = casts

    def emit(self, t, out, cam):
        a = self.opacity(t)
        if a <= 0.003:
            return
        m = self.world(t)
        row = inst_row(m, self.color(t), a, glow=float(self.glow(t)))
        out.mesh(self.key, row, m[:3, 3], transparent=a < 0.999,
                 casts=self.casts and a > 0.5)


def frame_between(a, b):
    """A matrix taking the unit y-axis onto the segment a -> b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d)
    y = d / max(L, 1e-9)
    ref = np.array([0, 0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(y, ref)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, a
    return m, L


class Rod(Node):
    """
    A glowing tube from a to b in its parent's space, grown by `grow` (0-1).
    With `tip`, it ends in a cone and reads as an arrow.
    """

    def __init__(self, a, b, radius=0.025, color="#ffffff", glow=0.6, tip=False,
                 tip_len=0.2, tip_r=0.075, grow=0.0, **kw):
        super().__init__(**kw)
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.radius, self.tip, self.tip_len, self.tip_r = radius, tip, tip_len, tip_r
        self.color = Track(lin(color))
        self.glow = Track(glow)
        self.grow = Track(grow)

    def emit(self, t, out, cam):
        a = self.opacity(t)
        g = float(self.grow(t))
        if a <= 0.003 or g <= 0.001:
            return
        base = self.world(t)
        f, L = frame_between(self.a, self.b)
        col, glow = self.color(t), float(self.glow(t))
        shaft = L * g - (self.tip_len if self.tip else 0.0)
        tr = a < 0.999
        if shaft > 1e-3:
            m = base @ f @ S((self.radius, shaft, self.radius))
            out.mesh(("cyl",), inst_row(m, col, a, glow=glow), m[:3, 3], tr, False)
        if self.tip:
            k = min(1.0, L * g / self.tip_len)
            m = base @ f @ T((0, max(shaft, 0), 0)) @ S(
                (self.tip_r * k, self.tip_len * k, self.tip_r * k))
            out.mesh(("cone",), inst_row(m, col, a, glow=glow), m[:3, 3], tr, False)


class Instances(Node):
    """
    Many copies of one mesh driven by arrays, for fields of tiles and
    particle-like crowds. `fn(t)` returns an (N, 32) float32 row array.
    """

    def __init__(self, key, fn, size=None, transparent=True, casts=False, **kw):
        super().__init__(**kw)
        self.key, self.fn, self.size = key, fn, size
        self.transparent, self.casts = transparent, casts

    def emit(self, t, out, cam):
        a = self.opacity(t)
        if a <= 0.003:
            return
        rows = self.fn(t)
        if rows is None or len(rows) == 0:
            return
        rows = rows.copy()
        base = self.world(t)
        if not np.allclose(base, np.eye(4)):
            ms = rows[:, :16].reshape(-1, 4, 4).transpose(0, 2, 1)
            ms = base[None] @ ms
            rows[:, :16] = ms.transpose(0, 2, 1).reshape(-1, 16)
        rows[:, 28] *= a
        centre = rows[:, 12:15].mean(0)
        see_through = self.transparent or bool((rows[:, 28] < 0.999).any())
        out.mesh(self.key, rows, centre, see_through, self.casts, size=self.size,
                 many=True)


# ── text and pictures ─────────────────────────────────────────────

@functools.lru_cache(None)
def font(face, px):
    return ImageFont.truetype(FONTS[face], px)


@functools.lru_cache(None)
def cap_ratio(face):
    f = font(face, 200)
    x0, y0, x1, y1 = f.getbbox("H")
    return (y1 - y0) / 200


def text_px(size, face="regular"):
    """The font pixel size whose cap height matches manim's at 1080p."""
    return CAP_PER_PT * size * PX_PER_UNIT / cap_ratio(face)


def raster_text(text, px, face="regular", spacing=1.3):
    """
    White text on transparent, returned with its optical box in pixels: left
    edge, cap top of the first line, right edge, baseline of the last line.
    Everything positions by that box, as manim positions by ink.
    """
    px = max(4, int(round(px)))
    f = font(face, px)
    lines = text.split("\n")
    asc, desc = f.getmetrics()
    lh = int(round(px * spacing))
    widths = [f.getlength(s) for s in lines]
    pad = max(6, px // 4)
    w = int(math.ceil(max(widths))) + 2 * pad
    h = asc + desc + lh * (len(lines) - 1) + 2 * pad
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        d.text((pad, pad + i * lh), s, font=f, fill=255)
    cap = cap_ratio(face) * px
    box = (pad, pad + asc - cap, pad + max(widths), pad + (len(lines) - 1) * lh + asc)
    return img, box, lh


def measure(text, size, face="regular", spacing=1.3):
    """World width and cap height of a label, without rasterising it."""
    px = 400
    f = font(face, px)
    lines = text.split("\n")
    u = CAP_PER_PT * size / (cap_ratio(face) * px)
    lh = px * spacing
    return (max(f.getlength(s) for s in lines) * u,
            (cap_ratio(face) * px + lh * (len(lines) - 1)) * u)


WORLD_PPU = 300    # texture pixels per world unit for text drawn in the scene


class Sprite(Node):
    """
    A flat textured quad: text (tinted white) or a picture. `anchor` is where
    the node's origin sits on the optical box: "c", "l", "r", "t", "b" or a
    combination like "tl".
    """

    def __init__(self, image, box, units_per_px, anchor="c", tint="#ffffff",
                 rgb=False, emit=1.0, **kw):
        super().__init__(**kw)
        self.tint = Track(lin(tint))
        self.emit_k = Track(emit)
        self.reveal = Track(1.0)
        self.wipe_down = False    # reveal top to bottom instead of left to right
        self.rgb = rgb
        self.anchor = anchor
        self._set_image(image, box, units_per_px)

    def _set_image(self, image, box, u):
        self.image, self.box, self.u = image, box, u
        self.tex_key = id(image)
        iw, ih = image.size
        bx0, by0, bx1, by1 = box
        ax = {"l": bx0, "r": bx1}.get(next((c for c in self.anchor if c in "lr"), "c"),
                                      (bx0 + bx1) / 2)
        ay = {"t": by0, "b": by1}.get(next((c for c in self.anchor if c in "tb"), "c"),
                                      (by0 + by1) / 2)
        # quad centre relative to the anchor point, in world units (y up)
        self.offset = ((iw / 2 - ax) * u, (ay - ih / 2) * u)
        self.dims = (iw * u, ih * u)

    @property
    def width(self):
        return (self.box[2] - self.box[0]) * self.u

    @property
    def height(self):
        return (self.box[3] - self.box[1]) * self.u

    def local_box(self):
        """The optical box in the node's own coordinates: x0, y0, x1, y1."""
        iw, ih = self.image.size
        ox, oy = self.offset
        x0 = ox - self.dims[0] / 2 + self.box[0] * self.u
        x1 = ox - self.dims[0] / 2 + self.box[2] * self.u
        y1 = oy + self.dims[1] / 2 - self.box[1] * self.u
        y0 = oy + self.dims[1] / 2 - self.box[3] * self.u
        return x0, y0, x1, y1

    def current(self, t):
        return self.image

    def emit(self, t, out, cam):
        a = self.opacity(t)
        if a <= 0.003:
            return
        img = self.current(t)
        m = self.world(t) @ T((self.offset[0], self.offset[1], 0)) @ S(
            (self.dims[0], self.dims[1], 1))
        r = float(self.reveal(t))
        out.sprite(img, m, self.tint(t), a, float(self.emit_k(t)), self.rgb,
                   -1.0 - r if self.wipe_down else r)


class Label(Sprite):
    """
    Text in the scene, sized in manim font units so it matches the 2D film.
    `text` may be a function of time, for counters; each distinct string is
    rasterised once.
    """

    def __init__(self, text, size=26, color="#e8ecf5", face="regular",
                 anchor="c", spacing=1.3, **kw):
        self.fn = text if callable(text) else None
        self.size, self.face, self.spacing = size, face, spacing
        first = self.fn(0.0) if self.fn else text
        px = CAP_PER_PT * size * WORLD_PPU / cap_ratio(face)
        self.px = px
        img, box, _ = raster_text(first, px, face, spacing)
        u = CAP_PER_PT * size / (cap_ratio(face) * round(px))
        self._cache = {first: (img, box)}
        super().__init__(img, box, u, anchor=anchor, tint=color, **kw)

    def current(self, t):
        if self.fn is None:
            return self.image
        s = self.fn(t)
        if s not in self._cache:
            img, box, _ = raster_text(s, self.px, self.face, self.spacing)
            self._cache[s] = (img, box)
        img, box = self._cache[s]
        if img is not self.image:
            self._set_image(img, box, self.u)
        return img


class Picture(Sprite):
    """An RGBA image (a logo) of a given world height."""

    def __init__(self, path, height, **kw):
        img = Image.open(path).convert("RGBA")
        # premultiply so mipmaps and blending do not fringe
        arr = np.asarray(img).astype("f4") / 255
        arr[..., :3] *= arr[..., 3:4]
        img = Image.fromarray((arr * 255 + 0.5).astype("u1"), "RGBA")
        u = height / img.size[1]
        super().__init__(img, (0, 0, img.size[0], img.size[1]), u, rgb=True, **kw)


class Hud:
    """
    Text laid over the finished frame: titles and read-outs. Positions are in
    manim's frame coordinates, so they read against the 2D film's layout.
    """

    def __init__(self, text, size=26, color="#e8ecf5", x=0.0, y=0.0, anchor="c",
                 face="regular", alpha=0.0, spacing=1.3):
        self.fn = text if callable(text) else None
        self.size, self.face, self.anchor, self.spacing = size, face, anchor, spacing
        self.color = srgb(color)
        self.pos = Track((x, y))
        self.alpha = Track(alpha)
        self.reveal = Track(1.0)
        self.scale = Track(1.0)
        self.px = text_px(size, face)
        self._cache = {}
        self.text0 = self.fn(0.0) if self.fn else text

    def raster(self, s):
        if s not in self._cache:
            self._cache[s] = raster_text(s, self.px, self.face, self.spacing)[:2]
        return self._cache[s]

    def anim(self, t, dur, ease=inout, **props):
        for k, v in props.items():
            getattr(self, k).to(t, dur, v, ease)
        return self

    def set(self, t, **props):
        return self.anim(t, 0, **props)

    def width(self):
        img, box = self.raster(self.text0)
        return (box[2] - box[0]) / PX_PER_UNIT


class HudRule(Hud):
    """A thin line over the frame, `length` units long, horizontal or not."""

    def __init__(self, length, x=0.0, y=0.0, color="#7d879c", vertical=False,
                 thickness=2, alpha=0.0, anchor="c"):
        super().__init__("", 10, color, x, y, anchor, alpha=alpha)
        n = int(round(length * PX_PER_UNIT))
        pad = 3
        if vertical:
            img = Image.new("L", (thickness + 2 * pad, n), 0)
            ImageDraw.Draw(img).rectangle([pad, 0, pad + thickness - 1, n], fill=255)
        else:
            img = Image.new("L", (n, thickness + 2 * pad), 0)
            ImageDraw.Draw(img).rectangle([0, pad, n, pad + thickness - 1], fill=255)
        self.img = img
        self.box = (0, 0, img.size[0], img.size[1])

    def raster(self, s):
        return self.img, self.box


# ── draw lists ────────────────────────────────────────────────────

class DrawList:
    def __init__(self, view):
        self.view = view
        self.opaque = {}
        self.sizes = {}
        self.shadow = {}
        self.trans = []

    def depth(self, p):
        return -(self.view @ np.append(p, 1.0))[2]

    def mesh(self, key, rows, centre, transparent, casts, size=None, many=False):
        rows = rows.reshape(-1, 32)
        if size is not None:
            self.sizes[key] = size
        if casts:
            self.shadow.setdefault(key, []).append(rows)
        if transparent:
            self.trans.append((self.depth(centre), "mesh", key, rows))
        else:
            self.opaque.setdefault(key, []).append(rows)

    def sprite(self, img, m, tint, alpha, emit, rgb, reveal=1.0):
        self.trans.append((self.depth(m[:3, 3]), "sprite", img,
                           (m, tint, alpha, emit, rgb, reveal)))

    def sorted_trans(self):
        """Back to front, with neighbouring runs of one mesh merged."""
        items = sorted(self.trans, key=lambda it: -it[0])
        merged = []
        for d, kind, key, payload in items:
            if (kind == "mesh" and merged and merged[-1][0] == "mesh"
                    and merged[-1][1] == key):
                merged[-1][2].append(payload)
            elif kind == "mesh":
                merged.append(["mesh", key, [payload]])
            else:
                merged.append(["sprite", key, payload])
        return merged


# ── camera and environment ──────────────────────────────────────────

class Camera:
    """
    Orbits a target: azimuth (degrees, 0 = looking down -z), elevation
    (degrees above the target) and distance, plus field of view. A slow drift
    keeps a held shot from looking frozen.
    """

    def __init__(self, look=(0, 4, 0), az=0.0, el=0.0, dist=12.31, fov=36.0,
                 drift=0.02):
        self.look = Track(look)
        self.az = Track(az)
        self.el = Track(el)
        self.dist = Track(dist)
        self.fov = Track(fov)
        self.roll = Track(0.0)
        self.drift = drift

    def anim(self, t, dur, ease=inout, **props):
        for k, v in props.items():
            getattr(self, k).to(t, dur, v, ease)
        return self

    def eye(self, t):
        az, el, d = math.radians(float(self.az(t))), math.radians(float(self.el(t))), float(self.dist(t))
        look = self.look(t)
        e = look + d * np.array([math.sin(az) * math.cos(el), math.sin(el),
                                 math.cos(az) * math.cos(el)])
        k = self.drift
        e = e + k * np.array([math.sin(t * 0.61), 0.6 * math.sin(t * 0.83 + 1.3),
                              math.sin(t * 0.47 + 2.1)])
        return e, look

    def matrices(self, t):
        eye, look = self.eye(t)
        f = look - eye
        f /= np.linalg.norm(f)
        up = np.array([0, 1.0, 0])
        rl = math.radians(float(self.roll(t)))
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        s, u = s * math.cos(rl) + u * math.sin(rl), u * math.cos(rl) - s * math.sin(rl)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = s, u, -f
        view[:3, 3] = -view[:3, :3] @ eye
        fov = math.radians(float(self.fov(t)))
        n, fa = 0.1, 400.0
        k = 1 / math.tan(fov / 2)
        proj = np.zeros((4, 4))
        proj[0, 0], proj[1, 1] = k / (W / H), k
        proj[2, 2], proj[2, 3] = (fa + n) / (n - fa), 2 * fa * n / (n - fa)
        proj[3, 2] = -1
        return view, proj, eye


class Env:
    """Look of the set: sky, floor, light, fog and post-processing."""

    def __init__(self, **kw):
        self.zenith = "#010208"
        self.horizon = "#070c19"
        self.stars = 0.9
        self.floor = True
        self.floor_col = "#030409"
        self.refl = 0.6
        self.grid = 0.0
        self.grid_size = 1.0
        self.pool = (0.0, 0.0, 8.0)
        self.pool_col = "#0d1528"
        self.light_dir = (0.32, -1.0, -0.55)
        self.light_col = (1.05, 1.0, 0.94)
        self.sky_amb = "#161d2c"
        self.gnd_amb = "#040509"
        self.light_box = ((0, 3, 0), 12.0)
        self.fog = 0.03
        self.fog0 = 18.0
        self.bloom = 0.5
        self.bloom_thresh = 0.9
        self.vignette = 0.35
        self.tilt = 0.0
        self.tilt_focus = 0.45
        self.tilt_band = 0.12
        self.__dict__.update(kw)


class Beat:
    """One scene: what is in it, how the camera moves, and how long it runs."""

    def __init__(self, name, duration, fade_in=0.35, fade_out=0.45):
        self.name, self.duration = name, duration
        self.fade_in, self.fade_out = fade_in, fade_out
        self.nodes, self.hud = [], []
        self.cam = Camera()
        self.env = Env()
        self.exposure = None   # optional Track overriding the fades

    def add(self, *nodes):
        self.nodes.extend(nodes)
        return nodes[0] if len(nodes) == 1 else nodes

    def overlay(self, *items):
        self.hud.extend(items)
        return items[0] if len(items) == 1 else items

    def expose(self, t):
        if self.exposure is not None:
            return float(self.exposure(t))
        a = min(1.0, t / self.fade_in) if self.fade_in > 0 else 1.0
        b = min(1.0, (self.duration - t) / self.fade_out) if self.fade_out > 0 else 1.0
        return smooth(max(0.0, min(a, b)))


# ── shaders ─────────────────────────────────────────────────────────

MESH_VS = """
#version 330
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_face;
in vec4 i_m0; in vec4 i_m1; in vec4 i_m2; in vec4 i_m3;
in vec4 i_color; in vec4 i_stroke; in vec4 i_extra; in vec4 i_extra2;
uniform mat4 u_vp; uniform float u_mirror;
out vec3 v_world; out vec3 v_nrm; out vec2 v_uv; out float v_face;
flat out vec4 v_color; flat out vec4 v_stroke; flat out vec4 v_extra; flat out vec4 v_extra2;
void main(){
    mat4 M = mat4(i_m0, i_m1, i_m2, i_m3);
    vec4 w = M * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_nrm = mat3(M) * in_nrm;
    w.y *= u_mirror;
    gl_Position = u_vp * w;
    v_uv = in_uv; v_face = in_face;
    v_color = i_color; v_stroke = i_stroke; v_extra = i_extra; v_extra2 = i_extra2;
}
"""

LIGHT_GLSL = """
uniform vec3 u_cam; uniform vec3 u_L; uniform vec3 u_Lcol;
uniform vec3 u_sky; uniform vec3 u_gnd;
uniform sampler2DShadow u_shadow; uniform mat4 u_lvp; uniform float u_texel;
uniform vec3 u_fogc; uniform float u_fogd; uniform float u_fog0;
uniform int u_mirror_pass;
float shadowf(vec3 w){
    vec4 p = u_lvp * vec4(w, 1.0);
    vec3 q = p.xyz / p.w * 0.5 + 0.5;
    if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0 || q.z > 1.0) return 1.0;
    float s = 0.0;
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            s += texture(u_shadow, vec3(q.xy + vec2(i, j) * u_texel * 1.25, q.z - 0.0012));
    return s / 25.0;
}
float fogf(vec3 w){
    float d = length(u_cam - w);
    return 1.0 - exp(-max(d - u_fog0, 0.0) * u_fogd);
}
"""

MESH_FS = """
#version 330
in vec3 v_world; in vec3 v_nrm; in vec2 v_uv; in float v_face;
flat in vec4 v_color; flat in vec4 v_stroke; flat in vec4 v_extra; flat in vec4 v_extra2;
uniform sampler2DArray u_img;
uniform vec3 u_size;
out vec4 f_color;
""" + LIGHT_GLSL + """
float sd_rrect(vec2 p, vec2 b, float r){
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
vec2 orient(vec2 uv, float k){
    int o = int(k + 0.5);
    if ((o & 4) != 0) uv.x = 1.0 - uv.x;
    for (int i = 0; i < (o & 3); i++) uv = vec2(uv.y, 1.0 - uv.x);
    return uv;
}
void main(){
    if (u_mirror_pass == 1 && v_world.y < -0.002) discard;
    vec3 N = normalize(v_nrm);
    vec3 V = normalize(u_cam - v_world);
    if (dot(N, V) < 0.0) N = -N;
    vec3 L = normalize(-u_L);
    float sh = shadowf(v_world);
    float nl = max(dot(N, L), 0.0);
    vec3 Hh = normalize(L + V);
    float nh = max(dot(N, Hh), 0.0);
    float nv = max(dot(N, V), 0.0);
    float fres = pow(1.0 - nv, 4.0);
    vec3 amb = mix(u_gnd, u_sky, N.y * 0.5 + 0.5);
    float glass = v_extra2.y;
    float layer = v_extra.x;
    bool face = abs(v_face) > 0.5;
    vec3 base = v_color.rgb;
    float a = v_color.a;
    vec3 col;
    if (face && layer >= 0.0) {
        vec2 uv = orient(vec2(v_uv.x, 1.0 - v_uv.y), v_extra2.z);
        float g = texture(u_img, vec3(uv, layer)).r;
        vec3 im = vec3(pow(g, 2.2)) * v_extra.z;
        col = im * (0.86 + 0.22 * nl * sh) + u_Lcol * pow(nh, 90.0) * 0.18 * sh;
    } else if (glass > 0.5) {
        col = base * (0.05 + 0.07 * nl * sh) + amb * 0.12
            + u_Lcol * pow(nh, 160.0) * 1.2 * sh + u_sky * fres * 1.6;
        col += base * v_extra.y;
        a = clamp(v_color.a * (0.34 + fres * 0.5), 0.0, 1.0);
    } else {
        col = base * (amb + u_Lcol * nl * sh * 0.95)
            + u_Lcol * pow(nh, 36.0) * 0.22 * sh + u_sky * fres * 0.35;
        col += base * v_extra.y;
    }
    if (face && v_stroke.a > 0.0) {
        vec2 p = (v_uv - 0.5) * u_size.xy;
        float d = -sd_rrect(p, u_size.xy * 0.5, u_size.z);
        float aa = fwidth(d) * 0.8;
        float band = 1.0 - smoothstep(v_stroke.a - aa, v_stroke.a + aa, d);
        col = mix(col, v_stroke.rgb * v_extra.w, band);
        a = mix(a, 1.0, band);
    }
    col = mix(col, u_fogc, fogf(v_world));
    float A = a * v_extra2.x;
    f_color = vec4(col * A, A);
}
"""

SHADOW_VS = """
#version 330
in vec3 in_pos;
in vec4 i_m0; in vec4 i_m1; in vec4 i_m2; in vec4 i_m3;
uniform mat4 u_lvp;
void main(){ gl_Position = u_lvp * mat4(i_m0, i_m1, i_m2, i_m3) * vec4(in_pos, 1.0); }
"""
SHADOW_FS = """
#version 330
void main(){}
"""

SPRITE_VS = """
#version 330
in vec2 in_xy;
uniform mat4 u_vp; uniform mat4 u_m; uniform float u_mirror;
out vec2 v_uv; out vec3 v_world;
void main(){
    vec4 w = u_m * vec4(in_xy, 0.0, 1.0);
    v_world = w.xyz;
    w.y *= u_mirror;
    gl_Position = u_vp * w;
    v_uv = vec2(in_xy.x + 0.5, 0.5 - in_xy.y);
}
"""
SPRITE_FS = """
#version 330
in vec2 v_uv; in vec3 v_world;
uniform sampler2D u_tex; uniform vec3 u_tint; uniform float u_alpha;
uniform float u_emit; uniform int u_rgb; uniform float u_reveal;
out vec4 f_color;
""" + LIGHT_GLSL + """
void main(){
    if (u_mirror_pass == 1 && v_world.y < -0.002) discard;
    vec4 t = texture(u_tex, v_uv);
    vec3 c = u_rgb == 1 ? pow(t.rgb / max(t.a, 1e-4), vec3(2.2)) * t.a : vec3(t.a);
    c *= u_tint * u_emit;
    c = mix(c, u_fogc * t.a, fogf(v_world));
    // a negative reveal wipes top to bottom; its magnitude less one is progress
    float r = u_reveal < 0.0 ? -u_reveal - 1.0 : u_reveal;
    float along = u_reveal < 0.0 ? v_uv.y : v_uv.x;
    float wipe = clamp((r * 1.12 - along) / 0.12, 0.0, 1.0);
    f_color = vec4(c, t.a) * u_alpha * wipe;
}
"""

QUAD_VS = """
#version 330
in vec2 in_xy;
out vec2 v_uv;
void main(){ v_uv = in_xy * 0.5 + 0.5; gl_Position = vec4(in_xy, 0.0, 1.0); }
"""

SKY_FS = """
#version 330
in vec2 v_uv;
uniform mat4 u_inv_vp; uniform vec3 u_cam;
uniform vec3 u_zen; uniform vec3 u_hor; uniform float u_stars; uniform float u_time;
out vec4 f_color;
float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
void main(){
    vec4 p = u_inv_vp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = normalize(p.xyz / p.w - u_cam);
    float h = dir.y;
    vec3 col = mix(u_hor, u_zen, smoothstep(-0.02, 0.55, h));
    float s = 0.0;
    for (int k = 0; k < 2; k++) {
        float sc = k == 0 ? 180.0 : 330.0;
        vec3 q = dir * sc;
        vec3 c = floor(q);
        float r = hash(c + float(k) * 13.1);
        if (r > 0.93) {
            vec3 o = vec3(hash(c + 1.7), hash(c + 3.1), hash(c + 5.3)) * 0.6 + 0.2;
            float d = length(fract(q) - o);
            float tw = 0.75 + 0.25 * sin(u_time * (1.0 + 3.0 * r) + r * 40.0);
            s += smoothstep(0.16, 0.0, d) * (r - 0.93) / 0.07 * tw * (k == 0 ? 1.0 : 0.55);
        }
    }
    col += vec3(0.85, 0.9, 1.0) * s * u_stars * 0.9 * smoothstep(0.0, 0.2, h);
    f_color = vec4(col, 1.0);
}
"""

FLOOR_VS = """
#version 330
in vec2 in_xy;
uniform mat4 u_vp;
out vec3 v_world;
void main(){ v_world = vec3(in_xy.x * 300.0, 0.0, in_xy.y * 300.0);
    gl_Position = u_vp * vec4(v_world, 1.0); }
"""
FLOOR_FS = """
#version 330
in vec3 v_world;
uniform sampler2D u_refl; uniform vec2 u_res;
uniform vec3 u_floor; uniform float u_reflk; uniform float u_grid; uniform float u_gsize;
uniform vec3 u_pool; uniform vec3 u_poolc; uniform vec3 u_hor;
out vec4 f_color;
""" + LIGHT_GLSL + """
void main(){
    vec2 suv = gl_FragCoord.xy / u_res;
    vec3 V = normalize(u_cam - v_world);
    float dist = length(u_cam - v_world);
    float fres = 0.05 + 0.95 * pow(1.0 - max(V.y, 0.0), 5.0);
    float sh = shadowf(v_world);
    vec2 dp = v_world.xz - u_pool.xy;
    float pool = exp(-dot(dp, dp) / (u_pool.z * u_pool.z));
    vec3 col = u_floor * (0.45 + 0.55 * sh) + u_poolc * pool * (0.25 + 0.75 * sh);
    vec2 g = v_world.xz / u_gsize;
    vec2 gw = abs(fract(g - 0.5) - 0.5) / max(fwidth(g), 1e-4);
    float line = 1.0 - min(min(gw.x, gw.y), 1.0);
    col += u_poolc * 1.6 * line * u_grid * exp(-dist * 0.05);
    float lod = 1.2 + clamp(dist * 0.05, 0.0, 2.5);
    vec3 refl = textureLod(u_refl, suv, lod).rgb;
    col += refl * u_reflk * (0.35 + 0.65 * fres);
    col += u_hor * fres * 0.6;
    col = mix(col, u_fogc, fogf(v_world));
    f_color = vec4(col, 1.0);
}
"""

DOWN_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_src; uniform vec2 u_texel; uniform float u_thresh;
out vec4 f_color;
void main(){
    vec3 c = texture(u_src, v_uv).rgb * 4.0;
    c += texture(u_src, v_uv + u_texel * vec2(-1.0, -1.0)).rgb;
    c += texture(u_src, v_uv + u_texel * vec2( 1.0, -1.0)).rgb;
    c += texture(u_src, v_uv + u_texel * vec2(-1.0,  1.0)).rgb;
    c += texture(u_src, v_uv + u_texel * vec2( 1.0,  1.0)).rgb;
    c /= 8.0;
    if (u_thresh >= 0.0) {
        float b = max(c.r, max(c.g, c.b));
        c *= smoothstep(u_thresh, u_thresh + 0.35, b);
    }
    f_color = vec4(c, 1.0);
}
"""
UP_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_src; uniform vec2 u_texel;
out vec4 f_color;
void main(){
    vec3 c = vec3(0.0);
    c += texture(u_src, v_uv + u_texel * vec2(-2.0, 0.0)).rgb;
    c += texture(u_src, v_uv + u_texel * vec2( 2.0, 0.0)).rgb;
    c += texture(u_src, v_uv + u_texel * vec2(0.0, -2.0)).rgb;
    c += texture(u_src, v_uv + u_texel * vec2(0.0,  2.0)).rgb;
    c += texture(u_src, v_uv + u_texel * vec2(-1.0, -1.0)).rgb * 2.0;
    c += texture(u_src, v_uv + u_texel * vec2( 1.0, -1.0)).rgb * 2.0;
    c += texture(u_src, v_uv + u_texel * vec2(-1.0,  1.0)).rgb * 2.0;
    c += texture(u_src, v_uv + u_texel * vec2( 1.0,  1.0)).rgb * 2.0;
    f_color = vec4(c / 12.0, 1.0);
}
"""
COMPOSITE_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_scene; uniform sampler2D u_bloom; uniform sampler2D u_blur;
uniform float u_bloomk; uniform float u_exposure; uniform float u_vig; uniform float u_time;
uniform float u_tilt; uniform float u_focus; uniform float u_band;
out vec4 f_color;
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec3 knee(vec3 x){
    const float k = 0.78;
    vec3 over = max(x - k, 0.0);
    return min(x, vec3(k)) + (1.0 - k) * (1.0 - exp(-over / (1.0 - k)));
}
void main(){
    vec3 c = texture(u_scene, v_uv).rgb;
    if (u_tilt > 0.0) {
        float k = u_tilt * smoothstep(u_band, u_band + 0.28, abs(v_uv.y - u_focus));
        c = mix(c, texture(u_blur, v_uv).rgb, k);
    }
    c += texture(u_bloom, v_uv).rgb * u_bloomk;
    c = knee(c);
    vec2 q = (v_uv - 0.5) * vec2(1.0, 0.72);
    c *= 1.0 - u_vig * smoothstep(0.18, 0.62, length(q));
    c = mix(c * 12.92, 1.055 * pow(max(c, 0.0), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
    c *= u_exposure;
    c += (hash(v_uv * vec2(1920.0, 1080.0) + fract(u_time) * 91.7) - 0.5) * (2.2 / 255.0);
    f_color = vec4(c, 1.0);
}
"""
HUD_VS = """
#version 330
in vec2 in_xy;
uniform vec4 u_rect;
out vec2 v_uv;
void main(){
    vec2 p = u_rect.xy + (in_xy + 0.5) * u_rect.zw;
    gl_Position = vec4(p.x / 960.0 - 1.0, 1.0 - p.y / 540.0, 0.0, 1.0);
    v_uv = in_xy + 0.5;
}
"""
HUD_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex; uniform vec3 u_tint; uniform float u_alpha; uniform float u_reveal;
out vec4 f_color;
void main(){
    float wipe = clamp((u_reveal * 1.12 - v_uv.x) / 0.12, 0.0, 1.0);
    float a = texture(u_tex, v_uv).a * u_alpha * wipe;
    f_color = vec4(u_tint * a, a);
}
"""


# ── renderer ────────────────────────────────────────────────────────

def gl_context():
    last = None
    for i in range(4):
        try:
            ctx = moderngl.create_standalone_context(backend="egl", device_index=i)
            return ctx
        except Exception as e:           # the NVIDIA device fails to init
            last = e
    raise last


def as_mat(m):
    return np.asarray(m, "f4").T.tobytes()


class Renderer:
    SHADOW = 2048

    def __init__(self, image_paths, samples=4):
        self.ctx = ctx = gl_context()
        self.samples = min(samples, ctx.max_samples)
        prog = ctx.program
        self.p_mesh = prog(vertex_shader=MESH_VS, fragment_shader=MESH_FS)
        self.p_shadow = prog(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
        self.p_sprite = prog(vertex_shader=SPRITE_VS, fragment_shader=SPRITE_FS)
        self.p_sky = prog(vertex_shader=QUAD_VS, fragment_shader=SKY_FS)
        self.p_floor = prog(vertex_shader=FLOOR_VS, fragment_shader=FLOOR_FS)
        self.p_down = prog(vertex_shader=QUAD_VS, fragment_shader=DOWN_FS)
        self.p_up = prog(vertex_shader=QUAD_VS, fragment_shader=UP_FS)
        self.p_comp = prog(vertex_shader=QUAD_VS, fragment_shader=COMPOSITE_FS)
        self.p_hud = prog(vertex_shader=HUD_VS, fragment_shader=HUD_FS)

        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4")
        self.fs_vbo = ctx.buffer(quad)
        half = np.array([-.5, -.5, .5, -.5, .5, .5, -.5, -.5, .5, .5, -.5, .5], "f4")
        self.half_vbo = ctx.buffer(half)
        self.vao = {}
        for name, p, vbo in (("sky", self.p_sky, self.fs_vbo), ("floor", self.p_floor, self.fs_vbo),
                             ("down", self.p_down, self.fs_vbo), ("up", self.p_up, self.fs_vbo),
                             ("comp", self.p_comp, self.fs_vbo), ("sprite", self.p_sprite, self.half_vbo),
                             ("hud", self.p_hud, self.half_vbo)):
            self.vao[name] = ctx.vertex_array(p, [(vbo, "2f", "in_xy")], skip_errors=True)

        self.meshes = {}
        self.inst = {}
        self._mesh_src = {"box": box_mesh(), "sphere": sphere_mesh(),
                          "cyl": cyl_mesh(), "cone": cyl_mesh(cone=True)}

        # images: one layer each, single channel
        self.layers = {}
        if image_paths:
            n = len(image_paths)
            data = np.zeros((n, 256, 256), "u1")
            for i, (name, path) in enumerate(image_paths):
                im = Image.open(path).convert("L").resize((256, 256), Image.LANCZOS)
                data[i] = np.asarray(im)
                self.layers[name] = i
            self.img = ctx.texture_array((256, 256, n), 1, data.tobytes())
        else:
            self.img = ctx.texture_array((4, 4, 1), 1, bytes(16))
        self.img.build_mipmaps()
        self.img.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.img.anisotropy = 8.0

        self.shadow_tex = ctx.depth_texture((self.SHADOW, self.SHADOW))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)

        s = self.samples
        self.msaa = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((W, H), 4, samples=s, dtype="f2")],
            depth_attachment=ctx.depth_renderbuffer((W, H), samples=s))
        self.scene_tex = ctx.texture((W, H), 4, dtype="f2")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_tex.repeat_x = self.scene_tex.repeat_y = False
        self.scene_fbo = ctx.framebuffer(color_attachments=[self.scene_tex])

        rw, rh = W // 2, H // 2
        self.refl_tex = ctx.texture((rw, rh), 4, dtype="f2")
        self.refl_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.refl_tex.repeat_x = self.refl_tex.repeat_y = False
        self.refl_fbo = ctx.framebuffer(color_attachments=[self.refl_tex],
                                        depth_attachment=ctx.depth_renderbuffer((rw, rh)))

        self.bloom = []
        w, h = W, H
        for _ in range(6):
            w, h = max(1, w // 2), max(1, h // 2)
            t = ctx.texture((w, h), 4, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self.bloom.append((t, ctx.framebuffer(color_attachments=[t])))
        self.blur = []
        w, h = W, H
        for _ in range(3):
            w, h = w // 2, h // 2
            t = ctx.texture((w, h), 4, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self.blur.append((t, ctx.framebuffer(color_attachments=[t])))

        self.out_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4)])
        self.textures = {}
        self.profile = None     # a dict to collect per-pass seconds into

    def _mark(self, name):
        if self.profile is None:
            return
        import time
        self.ctx.finish()
        now = time.perf_counter()
        last = self.profile.get("_t")
        if last is not None:
            self.profile[name] = self.profile.get(name, 0.0) + now - last
        self.profile["_t"] = now

    # resources

    def mesh(self, key, size=None):
        if key not in self.meshes:
            if key[0] == "rbox":
                data = rbox_mesh(*key[1:])
            else:
                data = self._mesh_src[key[0]]
            vbo = self.ctx.buffer(data)
            self.meshes[key] = (vbo, len(data))
        return self.meshes[key]

    def vao_for(self, prog_name, key, n):
        """A VAO for mesh `key` with an instance buffer of at least n rows."""
        vbo, count = self.mesh(key)
        k = (prog_name, key)
        buf, cap, vao = self.inst.get(k, (None, 0, None))
        if cap < n:
            if vao is not None:
                vao.release()
                buf.release()
            cap = max(n, 16, cap * 2)
            buf = self.ctx.buffer(reserve=cap * 128, dynamic=True)
            p = self.p_mesh if prog_name == "mesh" else self.p_shadow
            vao = self.ctx.vertex_array(p, [
                (vbo, "3f 3f 2f 1f", "in_pos", "in_nrm", "in_uv", "in_face"),
                (buf, "4f 4f 4f 4f 4f 4f 4f 4f/i", "i_m0", "i_m1", "i_m2", "i_m3",
                 "i_color", "i_stroke", "i_extra", "i_extra2")], skip_errors=True)
            self.inst[k] = (buf, cap, vao)
        return buf, vao, count

    def texture(self, img):
        k = id(img)
        if k not in self.textures:
            if img.mode == "L":
                a = np.asarray(img)
                data = np.repeat(a[..., None], 4, axis=2)
            else:
                data = np.asarray(img.convert("RGBA"))
            tex = self.ctx.texture(img.size, 4, np.ascontiguousarray(data).tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 8.0
            tex.repeat_x = tex.repeat_y = False
            self.textures[k] = (tex, img)      # hold img so its id stays unique
        return self.textures[k][0]

    def forget_textures(self):
        for tex, _ in self.textures.values():
            tex.release()
        self.textures = {}

    # passes

    def _set(self, prog, **u):
        for k, v in u.items():
            if k not in prog:
                continue
            if isinstance(v, np.ndarray) and v.shape == (4, 4):
                prog[k].write(as_mat(v))
            elif isinstance(v, (tuple, list, np.ndarray)):
                prog[k].value = tuple(float(x) for x in v)
            else:
                prog[k].value = v

    def _draw_meshes(self, rows_by_key, sizes, mirror):
        for key, chunks in rows_by_key:
            rows = np.concatenate(chunks) if isinstance(chunks, list) else chunks
            if not len(rows):
                continue
            buf, vao, count = self.vao_for("mesh", key, len(rows))
            buf.write(rows.astype("f4").tobytes())
            if "u_size" in self.p_mesh:
                size = sizes.get(key, (1, 1, 0.1))
                self.p_mesh["u_size"].value = tuple(float(x) for x in size)
            vao.render(instances=len(rows))

    def _draw_sprite(self, img, payload, vp, mirror_pass):
        m, tint, alpha, emit, rgb, reveal = payload
        tex = self.texture(img)
        tex.use(3)
        p = self.p_sprite
        self._set(p, u_m=m, u_tint=tint, u_alpha=float(alpha), u_emit=float(emit),
                  u_rgb=1 if rgb else 0, u_reveal=float(reveal))
        self.vao["sprite"].render()

    def _scene_pass(self, dl, vp, mirror_pass):
        ctx = self.ctx
        mir = -1.0 if mirror_pass else 1.0
        for p in (self.p_mesh, self.p_sprite):
            self._set(p, u_vp=vp, u_mirror=mir, u_mirror_pass=1 if mirror_pass else 0)
        # opaque
        ctx.disable(moderngl.BLEND)
        fbo = self.refl_fbo if mirror_pass else self.msaa
        fbo.depth_mask = True
        self._draw_meshes(list(dl.opaque.items()), dl.sizes, mirror_pass)
        # transparent, back to front
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        fbo.depth_mask = False
        for kind, key, payload in dl.sorted_trans():
            if kind == "mesh":
                self._draw_meshes([(key, payload)], dl.sizes, mirror_pass)
            else:
                self._draw_sprite(key, payload, vp, mirror_pass)
        fbo.depth_mask = True
        ctx.disable(moderngl.BLEND)

    def _light(self, env, eye, mirror_pass):
        L = np.asarray(env.light_dir, float)
        L = L / np.linalg.norm(L)
        centre, half = env.light_box
        centre = np.asarray(centre, float)
        eye_l = centre - L * 60
        f = L
        up = np.array([0, 0, 1.0]) if abs(f[1]) > 0.99 else np.array([0, 1.0, 0])
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = s, u, -f
        view[:3, 3] = -view[:3, :3] @ eye_l
        n, fa = 1.0, 140.0
        proj = np.eye(4)
        proj[0, 0] = proj[1, 1] = 1 / half
        proj[2, 2], proj[2, 3] = -2 / (fa - n), -(fa + n) / (fa - n)
        lvp = proj @ view
        cam = eye.copy()
        if mirror_pass:
            cam[1] = -cam[1]
        u = dict(u_cam=cam, u_L=L, u_Lcol=env.light_col, u_sky=lin(env.sky_amb),
                 u_gnd=lin(env.gnd_amb), u_lvp=lvp, u_texel=1.0 / self.SHADOW,
                 u_fogc=lin(env.horizon), u_fogd=env.fog, u_fog0=env.fog0,
                 u_mirror_pass=1 if mirror_pass else 0)
        return lvp, u

    def frame(self, beat, t):
        ctx = self.ctx
        env = beat.env
        view, proj, eye = beat.cam.matrices(t)
        vp = proj @ view
        self._mark('start')
        dl = DrawList(view)
        for n in beat.nodes:
            stack = [n]
            while stack:
                x = stack.pop()
                x.emit(t, dl, beat.cam)
                stack.extend(x.children)

        self._mark('collect')
        lvp, lu = self._light(env, eye, False)
        # shadow map
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        self._set(self.p_shadow, u_lvp=lvp)
        for key, chunks in dl.shadow.items():
            rows = np.concatenate(chunks)
            buf, vao, _ = self.vao_for("shadow", key, len(rows))
            buf.write(rows.tobytes())
            vao.render(instances=len(rows))

        self._mark('shadow')
        self.shadow_tex.use(1)
        self.img.use(0)
        for p in (self.p_mesh, self.p_sprite, self.p_floor):
            self._set(p, u_img=0, u_shadow=1, u_tex=3, u_refl=2)

        # reflection
        if env.floor and env.refl > 0:
            _, mu = self._light(env, eye, True)
            for p in (self.p_mesh, self.p_sprite):
                self._set(p, **mu)
            self.refl_fbo.use()
            self.refl_fbo.clear(0, 0, 0, 0, depth=1.0)
            self._scene_pass(dl, vp, True)
            self.refl_tex.build_mipmaps()

        self._mark('reflection')
        # main
        for p in (self.p_mesh, self.p_sprite, self.p_floor):
            self._set(p, **lu)
        self.msaa.use()
        self.msaa.clear(0, 0, 0, 1, depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        self._set(self.p_sky, u_inv_vp=np.linalg.inv(vp), u_cam=eye, u_zen=lin(env.zenith),
                  u_hor=lin(env.horizon), u_stars=float(env.stars), u_time=float(t))
        self.vao["sky"].render()
        ctx.enable(moderngl.DEPTH_TEST)
        if env.floor:
            self.refl_tex.use(2)
            fvp = vp
            self._set(self.p_floor, u_vp=fvp, u_res=(W, H), u_floor=lin(env.floor_col),
                      u_reflk=float(env.refl), u_grid=float(env.grid),
                      u_gsize=float(env.grid_size), u_pool=env.pool,
                      u_poolc=lin(env.pool_col), u_hor=lin(env.horizon))
            self.vao["floor"].render()
        self._mark('sky+floor')
        self._scene_pass(dl, vp, False)
        self._mark('scene')

        ctx.copy_framebuffer(self.scene_fbo, self.msaa)
        ctx.disable(moderngl.DEPTH_TEST)
        self._mark('resolve')

        # bloom: threshold on the way down, tent on the way up
        src = self.scene_tex
        for i, (tex, fbo) in enumerate(self.bloom):
            fbo.use()
            src.use(0)
            self._set(self.p_down, u_src=0, u_texel=(1 / src.width, 1 / src.height),
                      u_thresh=float(env.bloom_thresh) if i == 0 else -1.0)
            self.vao["down"].render()
            src = tex
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self.bloom) - 1, 0, -1):
            small, _ = self.bloom[i]
            _, big = self.bloom[i - 1]
            big.use()
            small.use(0)
            self._set(self.p_up, u_src=0, u_texel=(0.5 / small.width, 0.5 / small.height))
            self.vao["up"].render()
        ctx.disable(moderngl.BLEND)

        if env.tilt > 0:
            src = self.scene_tex
            for tex, fbo in self.blur:
                fbo.use()
                src.use(0)
                self._set(self.p_down, u_src=0, u_texel=(1 / src.width, 1 / src.height),
                          u_thresh=-1.0)
                self.vao["down"].render()
                src = tex

        self.out_fbo.use()
        self.scene_tex.use(0)
        self.bloom[0][0].use(1)
        self.blur[-1][0].use(2)
        self._set(self.p_comp, u_scene=0, u_bloom=1, u_blur=2, u_bloomk=float(env.bloom),
                  u_exposure=beat.expose(t), u_vig=float(env.vignette), u_time=float(t),
                  u_tilt=float(env.tilt), u_focus=float(env.tilt_focus),
                  u_band=float(env.tilt_band))
        self.vao["comp"].render()

        self._mark('post')
        # overlay
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        k = beat.expose(t)
        for h in beat.hud:
            a = float(h.alpha(t)) * k
            if a <= 0.003:
                continue
            s = h.fn(t) if h.fn else h.text0
            img, box = h.raster(s)
            x, y = h.pos(t)
            k_s = float(h.scale(t))
            px, py = W / 2 + x * PX_PER_UNIT, H / 2 - y * PX_PER_UNIT
            bx0, by0, bx1, by1 = [v * k_s for v in box]
            ax = {"l": bx0, "r": bx1}.get(next((c for c in h.anchor if c in "lr"), "c"),
                                          (bx0 + bx1) / 2)
            ay = {"t": by0, "b": by1}.get(next((c for c in h.anchor if c in "tb"), "c"),
                                          (by0 + by1) / 2)
            x0, y0 = px - ax, py - ay
            if h.fn is None and abs(k_s - 1) < 1e-6:
                x0, y0 = round(x0), round(y0)
            self.texture(img).use(3)
            self._set(self.p_hud, u_tex=3,
                      u_rect=(x0, y0, img.size[0] * k_s, img.size[1] * k_s),
                      u_tint=h.color, u_alpha=a, u_reveal=float(h.reveal(t)))
            self.vao["hud"].render()
        ctx.disable(moderngl.BLEND)
        self._mark('hud')
        data = self.out_fbo.read(components=3, alignment=1)
        self._mark('read')
        return data


def encoder(path, fps=FPS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-vf", "vflip",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", path], stdin=subprocess.PIPE)


def still(renderer, beat, t, path):
    data = renderer.frame(beat, t)
    img = Image.frombytes("RGB", (W, H), data).transpose(Image.FLIP_TOP_BOTTOM)
    img.save(path)
    return img
