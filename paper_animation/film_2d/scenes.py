"""
The 2D explainer, about 108 seconds: what a VLM does with an image, the two
models and their prompts, the Likert and Tournament formats, how much of the
known anomaly set they recover, Table 1, and the new candidates.

Every number on screen is real. Cutouts come from the HST archive via
prepare/extract_cutouts.py, which also writes the recall beat's cutoff curve
straight out of the run CSV. Likert scores are the Sept-10 Gemini run; costs
and Table 1 are the paper's.

The one reconstruction: a Likert run logs only each image's final 0-50 total,
not the 1-5 it was given in each of the 10 rounds. Per-round values shown here
are final/10, which is the correct average by construction.

Render the whole thing, or one beat while iterating, from paper_animation/:

    manim -qh film_2d/scenes.py Explainer
    manim -ql film_2d/scenes.py Recall
"""

import json
import os
import random

from manim import *

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
ANOM = os.path.join(ASSETS, "anomalies")
CAND = os.path.join(ASSETS, "candidates")
LOGOS = os.path.join(ASSETS, "logos")

config.background_color = "#04060d"

INK = "#e8ecf5"
DIM = "#7d879c"
PICK = "#00e5a0"     # selected / survived
DROP = "#ff5470"     # rejected
WARM = "#ffc857"     # the tracked object

# The five stages of a VLM call, tinted the way the reference infographic is.
STAGE_COLORS = ("#4ade80", "#2dd4bf", "#4c7fff", "#a78bfa", "#c084fc")
PANEL_W = 2.2
ALT = "#a78bfa"      # the second format, set against PICK

MONO = "DejaVu Sans Mono"

# Both prompts verbatim from the paper's Appendix A, as (prefix, text) items.
# A prefix hangs over the item's wrapped continuation lines, as in the paper.
LIKERT_PROMPT = [
    ("", "You are an expert astronomer evaluating a 4x4 grid of astronomical "
         "images. The images are indexed 1-4 in the top row, 5-8 in the second "
         "row, 9-12 in the third row, and 13-16 in the bottom row."),
    ("", "Your task:"),
    ("1. ", "Select any images that are scientifically interesting. Skip "
            "blank, empty, or uninformative frames."),
    ("2. ", "For each selected image, assign a Score from 1 to 5 (no "
            "explanation needed):"),
    ("   ", "1 = mildly interesting, 2 = somewhat interesting, 3 = "
            "interesting, 4 = very interesting, 5 = exceptionally interesting."),
    ("", "Return a list of selected images. If none are interesting, return "
         "an empty list."),
]
TOURNAMENT_PROMPT = [
    ("", "You are an expert astronomer evaluating a 2x2 grid of four "
         "astronomical images, labeled Image 1 (top-left), Image 2 "
         "(top-right), Image 3 (bottom-left), and Image 4 (bottom-right). "
         "Evaluate each image independently. Do not force a winner. For each "
         "image, decide whether it is scientifically interesting enough to "
         "keep for human inspection. It is acceptable to keep any number of "
         "the four images (none, some, or all). Interesting images may show "
         "unusual morphology, asymmetry, interactions, merger-like features, "
         "arcs, rings, tails, shells, clumps, distortions, rare-looking "
         "objects, or other features worth human inspection. Reject "
         "artifacts, blank images, noisy frames, and obvious non-astronomical "
         "defects. Return only whether to keep each of Image 1, Image 2, "
         "Image 3, and Image 4. No explanation needed."),
]

LENS = "4001415925883"        # SDSS J1205+4910: 50/50 Likert, 10/10 Tournament


def cutout(name, size=1.0):
    img = ImageMobject(os.path.join(ASSETS, f"{name}.png"))
    img.set_height(size)
    return img


def framed(img, color=DIM, width=1.5):
    return SurroundingRectangle(img, color=color, stroke_width=width, buff=0)


def badge(value, anchor, color=PICK, size=0.44, font=26):
    """A filled chip holding a score. Bare text on a cutout is unreadable."""
    chip = RoundedRectangle(width=size, height=size, corner_radius=size * 0.23,
                            fill_color=color, fill_opacity=1.0, stroke_width=0)
    num = Text(str(value), font_size=font, color="#04060d", weight=BOLD)
    group = VGroup(chip, num)
    num.move_to(chip.get_center())
    group.move_to(anchor)
    return group


def caption(text, size=26, color=INK):
    return Text(text, font_size=size, color=color)


def scene_title(text):
    """The heading a beat opens under: one size, one place, every beat."""
    return caption(text, 46, INK).to_edge(UP, buff=0.5)


def wrap_prompt(items, width, keep=()):
    """
    Wrap (prefix, text) items to `width` columns, never splitting a phrase in
    `keep` across lines, so each one can be highlighted as a single run.
    """
    import textwrap
    glue = "\x1f"                      # stands in for a space that must hold
    lines = []
    for prefix, text in items:
        for phrase in keep:
            text = text.replace(phrase, phrase.replace(" ", glue))
        for i, line in enumerate(textwrap.wrap(
                text, width=width - len(prefix),
                break_long_words=False, break_on_hyphens=False)):
            lead = prefix if i == 0 else " " * len(prefix)
            lines.append((lead + line).replace(glue, " "))
    return lines


def glyphs(block, lines, start, length):
    """
    The glyphs of characters [start, start+length) of a multi-line Text.
    manim keeps one submobject per visible character and none for spaces or
    line breaks, so a character offset maps to a glyph index by counting.
    """
    text = "\n".join(lines)
    g0 = sum(1 for ch in text[:start] if not ch.isspace())
    n = sum(1 for ch in text[start:start + length] if not ch.isspace())
    return VGroup(*block.submobjects[g0:g0 + n])


def fit(mob, width):
    """Shrink a label that would otherwise overflow its column."""
    if mob.width > width:
        mob.scale(width / mob.width)
    return mob


def panel(height, color, center, width=PANEL_W):
    box = RoundedRectangle(width=width, height=height, corner_radius=0.14,
                           stroke_color=color, stroke_width=1.6,
                           fill_color=color, fill_opacity=0.06)
    return box.move_to(center)


def flow(a, b, color):
    """A short arrow from one panel's edge to the next."""
    return Arrow(a, b, buff=0.07, stroke_width=3, color=color,
                 max_tip_length_to_length_ratio=0.42, tip_length=0.16)


# A DecimalNumber draws ~1.43x smaller than a Text at the same font_size, so a
# counter set beside a label looks shrunken unless it is scaled up to match.
NUMBER_FACTOR = 1.43


def number(value, size=26, color=INK):
    """A counter that visually matches caption() at the same size."""
    return Integer(value, font_size=size * NUMBER_FACTOR, color=color)


def counter_on(tracker, size=26, color=INK):
    """
    An Integer driven by a ValueTracker.

    Never animate a DecimalNumber with .animate.set_value(): that builds a
    Transform between two states whose glyph counts differ (20,000 has five
    digits, 1,824 has four), and manim burns the whole run_time trying to
    match points between them - the tournament beat hung there. Driving the
    value from a tracker redraws the digits each frame instead, which is cheap.
    """
    mob = number(int(tracker.get_value()), size, color)
    mob.add_updater(lambda m: m.set_value(int(tracker.get_value())))
    return mob


def pin_edge(mob, edge=RIGHT):
    """
    Hold one edge of a counter still as its digit count changes.

    A DecimalNumber regrows from its centre, so a total running 0 -> 50 spreads
    both ways and collides with whatever sits beside it. Call this after the
    layout is fixed; the updater recomputes the offset each frame, so it does
    not accumulate.
    """
    target = mob.get_edge_center(edge).copy()
    mob.add_updater(lambda m: m.shift(target - m.get_edge_center(edge)))
    return mob


class Beats(Scene):
    """Shared setup; each beat is a method so it can be rendered alone."""

    def setup(self):
        with open(os.path.join(ASSETS, "cutouts.json")) as f:
            self.meta = json.load(f)
        ranked = sorted(self.meta, key=lambda k: -(self.meta[k]["likert"] or 0))
        self.high = [n for n in ranked if (self.meta[n]["likert"] or 0) >= 45]
        self.mid = [n for n in ranked if 18 <= (self.meta[n]["likert"] or 0) <= 30]
        self.low = [n for n in ranked if (self.meta[n]["likert"] or 0) <= 2]

    def score_of(self, name):
        return self.meta[name]["likert"] or 0

    # ── 1. the problem ──────────────────────────────────────
    def opening(self):
        # Mostly unremarkable with a few striking ones scattered through, which
        # is what the real distribution looks like: 8,098 of 20,000 score zero.
        rng = random.Random(7)
        pool = self.high[:6] + self.mid[:12] + (self.low * 2)[:30]
        rng.shuffle(pool)

        cols, rows = 8, 5
        tiles = Group()
        for i, name in enumerate(pool[:cols * rows]):
            t = cutout(name, size=1.35)
            t.move_to([(i % cols - (cols - 1) / 2) * 1.45,
                       ((rows - 1) / 2 - i // cols) * 1.45, 0])
            tiles.add(t)

        self.play(LaggedStart(*[FadeIn(t, scale=0.7) for t in tiles],
                              lag_ratio=0.05, run_time=1.9))
        self.wait(0.2)

        veil = Rectangle(width=16, height=9, fill_color="#04060d",
                         fill_opacity=0.84, stroke_width=0)
        title = caption("Millions of images", 54)
        sub = caption("Which ones deserve astronomers' attention?", 30, DIM)
        rule = Line(LEFT * 2.4, RIGHT * 2.4, stroke_color=DIM,
                    stroke_width=1.5, stroke_opacity=0.5)
        lead = caption("We can leverage", 26, DIM)
        punch = caption("Vision-Language Models (VLMs)", 42, PICK)
        sub.next_to(title, DOWN, buff=0.38)
        rule.next_to(sub, DOWN, buff=0.55)
        lead.next_to(rule, DOWN, buff=0.5)
        punch.next_to(lead, DOWN, buff=0.26)
        block = VGroup(title, sub, rule, lead, punch).move_to(ORIGIN)

        self.play(FadeIn(veil), Write(title), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.6)
        self.wait(0.5)
        self.play(Create(rule), FadeIn(lead), run_time=0.5)
        self.play(FadeIn(punch, shift=UP * 0.2), run_time=0.8)
        self.wait(0.8)
        # The veil is the background colour, so it can stay put while what is
        # under and over it fades, then go unseen. Fading it with the rest
        # uncovered the tiles faster than they faded, so the wall brightened
        # for a moment on its way out.
        #
        # Each piece is faded on its own because they are already on screen.
        # Handing FadeOut a new Group would make manim add that group on top
        # of everything, lifting the tiles above the veil for the whole fade.
        self.play(*[FadeOut(t) for t in tiles], *[FadeOut(m) for m in block],
                  run_time=0.55)
        self.remove(veil)

    # ── 2. what a vision-language model does with one ─────────────────
    def vlm(self):
        """
        One call, left to right: inputs, encoders, fusion, representation,
        answer. The prompt and the 1-5 scale are the ones the scoring scripts
        actually send; the answer shown is this image's real per-round score.
        """
        rng = random.Random(11)
        xs = [-5.64, -2.82, 0.0, 2.82, 5.64]
        top_y, bot_y, mid_y = 1.15, -1.35, -0.1

        heads = VGroup()
        for i, name in enumerate(["Inputs", "Encode", "Align & Fuse",
                                  "Understand", "Answer"]):
            col = STAGE_COLORS[i]
            disc = Circle(radius=0.23, fill_color=col, fill_opacity=1.0,
                          stroke_width=0)
            num = Text(str(i + 1), font_size=22, color="#04060d", weight=BOLD)
            num.move_to(disc.get_center())
            lab = fit(caption(name, 24, col), PANEL_W)
            lab.next_to(disc, DOWN, buff=0.16)
            heads.add(VGroup(disc, num, lab).move_to([xs[i], 3.05, 0]))

        def titled(box, text, color, size=18):
            lab = fit(caption(text, size, color), PANEL_W - 0.25)
            lab.move_to(box.get_top() + DOWN * 0.26)
            return lab

        # 1 - inputs
        img_box = panel(1.95, STAGE_COLORS[0], [xs[0], top_y, 0])
        img_lab = titled(img_box, "Image", STAGE_COLORS[0])
        pic = cutout(LENS, size=1.2).move_to(img_box.get_center() + DOWN * 0.14)
        txt_box = panel(1.95, STAGE_COLORS[0], [xs[0], bot_y, 0])
        txt_lab = titled(txt_box, "Text prompt", STAGE_COLORS[0])
        lines = VGroup(*[caption(s, 17, INK) for s in (
            "Which of these are",
            "scientifically",
            "interesting?",
            "Score each 1–5.")]).arrange(DOWN, buff=0.12)
        fit(lines, PANEL_W - 0.3)
        lines.move_to(txt_box.get_center() + DOWN * 0.18)

        # 2 - encoders
        vis_box = panel(1.95, STAGE_COLORS[1], [xs[1], top_y, 0])
        vis_lab = titled(vis_box, "Vision encoder", STAGE_COLORS[1])
        patches = VGroup()
        for r in range(4):
            for c in range(4):
                patches.add(Square(side_length=0.24, stroke_width=0,
                                   fill_color=STAGE_COLORS[1],
                                   fill_opacity=rng.uniform(0.25, 0.95))
                            .move_to(vis_box.get_center()
                                     + DOWN * 0.16
                                     + RIGHT * (c - 1.5) * 0.29
                                     + DOWN * (r - 1.5) * 0.29))
        enc_box = panel(1.95, STAGE_COLORS[1], [xs[1], bot_y, 0])
        enc_lab = titled(enc_box, "Text encoder", STAGE_COLORS[1])
        toks = VGroup()
        for widths in ([0.50, 0.34, 0.42], [0.38, 0.50, 0.30], [0.44, 0.36]):
            toks.add(VGroup(*[
                RoundedRectangle(width=w, height=0.17, corner_radius=0.07,
                                 stroke_width=0, fill_color=STAGE_COLORS[1],
                                 fill_opacity=0.8) for w in widths
            ]).arrange(RIGHT, buff=0.09))
        toks.arrange(DOWN, buff=0.15).move_to(enc_box.get_center() + DOWN * 0.16)

        # 3 - fusion, spanning both input rows
        fuse_box = panel(4.45, STAGE_COLORS[2], [xs[2], mid_y, 0])
        fuse_lab = titled(fuse_box, "Multimodal model", STAGE_COLORS[2])
        columns = []
        for dx, n in ((-0.62, 3), (0.0, 4), (0.62, 3)):
            columns.append([fuse_box.get_center() + RIGHT * dx + DOWN * 0.3
                            + UP * (k - (n - 1) / 2) * 0.78 for k in range(n)])
        edges = VGroup(*[Line(p, q, stroke_color=STAGE_COLORS[2],
                              stroke_width=1.2, stroke_opacity=0.4)
                         for a, b in zip(columns, columns[1:])
                         for p in a for q in b])
        nodes = VGroup(*[Dot(p, radius=0.1, color=STAGE_COLORS[2])
                         for col in columns for p in col])

        # 4 - what it now holds
        rep_box = panel(2.2, STAGE_COLORS[3], [xs[3], mid_y, 0])
        rep_lab = VGroup(caption("Learned", 18, STAGE_COLORS[3]),
                         caption("representation", 18, STAGE_COLORS[3]))
        rep_lab.arrange(DOWN, buff=0.06)
        rep_lab.move_to(rep_box.get_top() + DOWN * 0.4)
        cells = VGroup(*[Square(side_length=0.26, stroke_width=0,
                                fill_color=STAGE_COLORS[3],
                                fill_opacity=rng.uniform(0.35, 0.95))
                         for _ in range(5)]).arrange(RIGHT, buff=0.08)
        dots = caption("…", 22, STAGE_COLORS[3])
        rep_row = VGroup(cells, dots).arrange(RIGHT, buff=0.12)
        fit(rep_row, PANEL_W - 0.3)
        rep_row.move_to(rep_box.get_center() + DOWN * 0.4)

        # 5 - the number that comes back
        ans_box = panel(2.2, STAGE_COLORS[4], [xs[4], mid_y, 0])
        ans_lab = titled(ans_box, "Score", STAGE_COLORS[4])
        big = caption("5", 62, PICK)
        of = caption("of 5", 20, DIM)
        verdict = VGroup(caption("exceptionally", 15, INK),
                         caption("interesting", 15, INK)).arrange(DOWN, buff=0.04)
        answer = VGroup(VGroup(big, of).arrange(RIGHT, buff=0.18,
                                                aligned_edge=DOWN), verdict)
        answer.arrange(DOWN, buff=0.16)
        answer.move_to(ans_box.get_center() + DOWN * 0.22)

        arrows = [
            flow(img_box.get_right(), vis_box.get_left(), STAGE_COLORS[0]),
            flow(txt_box.get_right(), enc_box.get_left(), STAGE_COLORS[0]),
            flow(vis_box.get_right(), fuse_box.get_left() + UP * 1.25,
                 STAGE_COLORS[1]),
            flow(enc_box.get_right(), fuse_box.get_left() + DOWN * 1.25,
                 STAGE_COLORS[1]),
            flow(fuse_box.get_right(), rep_box.get_left(), STAGE_COLORS[2]),
            flow(rep_box.get_right(), ans_box.get_left(), STAGE_COLORS[3]),
        ]

        # The diagram is laid out against the full frame; scale it as one piece
        # to make room for the title rather than re-deriving every position.
        title = scene_title("VLM Mechanism")
        diagram = Group(heads, img_box, img_lab, pic, txt_box, txt_lab, lines,
                        vis_box, vis_lab, patches, enc_box, enc_lab, toks,
                        fuse_box, fuse_lab, edges, nodes, rep_box, rep_lab,
                        rep_row, ans_box, ans_lab, answer, *arrows)
        diagram.scale(0.96).next_to(title, DOWN, buff=0.5)

        self.play(Write(title), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(h, shift=DOWN * 0.15) for h in heads],
                              lag_ratio=0.12), run_time=0.9)
        self.play(FadeIn(img_box), FadeIn(img_lab), FadeIn(pic),
                  FadeIn(txt_box), FadeIn(txt_lab), FadeIn(lines), run_time=1.0)
        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]),
                  FadeIn(vis_box), FadeIn(vis_lab),
                  FadeIn(enc_box), FadeIn(enc_lab),
                  LaggedStart(*[FadeIn(p, scale=0.6) for p in patches],
                              lag_ratio=0.02),
                  LaggedStart(*[FadeIn(t) for t in toks], lag_ratio=0.12),
                  run_time=1.3)
        self.play(GrowArrow(arrows[2]), GrowArrow(arrows[3]),
                  FadeIn(fuse_box), FadeIn(fuse_lab),
                  Create(edges), FadeIn(nodes), run_time=1.3)
        self.play(GrowArrow(arrows[4]), FadeIn(rep_box), FadeIn(rep_lab),
                  LaggedStart(*[FadeIn(s, scale=0.6) for s in cells],
                              lag_ratio=0.08),
                  FadeIn(dots), run_time=1.0)
        self.play(GrowArrow(arrows[5]), FadeIn(ans_box), FadeIn(ans_lab),
                  FadeIn(answer, scale=0.8), run_time=1.1)
        self.play(Indicate(big, scale_factor=1.15, color=PICK), run_time=0.6)
        self.wait(1.4)
        self.play(FadeOut(Group(title, diagram)), run_time=0.6)

    # ── 2b. which two models, and why ──────────────────────────
    def models(self):
        """
        Ten seconds on the two models: how each is run, and its list price
        per million tokens (the figures metrics/make_table1.py prices with).
        """
        head = scene_title("VLM Choice")
        self.play(Write(head), run_time=1.1)

        specs = [
            ("gemini", "Gemini 3.1 Flash-Lite", STAGE_COLORS[2], -3.45,
             ["closed-weight, called over an API",
              "$0.25 in / $1.50 out per million tokens"]),
            ("qwen", "Qwen3.5-397B-A17B", ALT, 3.45,
             ["open-weight, served via OpenRouter",
              "$0.39 in / $2.34 out per million tokens"]),
        ]

        boxes, logos, titles, bodies = VGroup(), Group(), VGroup(), VGroup()
        for slug, name, col, x, lines in specs:
            box = RoundedRectangle(width=6.2, height=4.3, corner_radius=0.18,
                                   stroke_color=col, stroke_width=2,
                                   fill_color=col, fill_opacity=0.05)
            box.move_to([x, -0.55, 0])
            logo = ImageMobject(os.path.join(LOGOS, f"{slug}.png"))
            logo.set_height(0.95)
            logo.move_to(box.get_top() + DOWN * 0.95)
            title = fit(caption(name, 30, col), 5.4)
            title.next_to(logo, DOWN, buff=0.4)
            body = VGroup(*[caption(t, 22, INK) for t in lines])
            body.arrange(DOWN, buff=0.22)
            fit(body, 5.5)
            body.next_to(title, DOWN, buff=0.5)
            boxes.add(box)
            logos.add(logo)
            titles.add(title)
            bodies.add(body)

        self.play(FadeIn(boxes), FadeIn(logos, scale=0.85), run_time=1.3)
        self.play(Write(titles[0]), Write(titles[1]), run_time=0.9)
        self.play(FadeIn(bodies, shift=UP * 0.15), run_time=1.1)

        self.wait(3.2)
        self.play(FadeOut(Group(head, boxes, logos, titles, bodies)),
                  run_time=0.7)

    # ── 2c. what the models are actually asked ────────────────────
    def prompts(self):
        """
        The two prompts from the paper's Appendix A, side by side and word for
        word. Highlights land in three steps: the role both prompts give the
        model, the criterion both use, then what each format actually asks.
        """
        title = scene_title("VLM Prompts")
        self.play(Write(title), run_time=0.7)

        COLS, CARD_W = 56, 6.75
        # Size the type so COLS monospace columns exactly fill a card.
        a20 = (Text("M" * (COLS + 1), font=MONO, font_size=20).width
               - Text("M", font=MONO, font_size=20).width) / COLS
        size = 20 * (CARD_W - 0.5) / (COLS * a20)

        shared = ["You are an expert astronomer", "scientifically interesting"]
        specs = [("Likert prompt", PICK, -3.5, LIKERT_PROMPT,
                  ["assign a Score from 1 to 5"]),
                 ("Tournament prompt", ALT, 3.5, TOURNAMENT_PROMPT,
                  ["keep for human inspection", "Do not force a winner."])]

        cards = []
        for name, colour, x, items, own in specs:
            lines = wrap_prompt(items, COLS, shared + own)
            block = Text("\n".join(lines), font=MONO, font_size=size,
                         color=INK, line_spacing=0.9)
            rows, g = [], 0
            for line in lines:
                n = sum(1 for ch in line if not ch.isspace())
                rows.append(VGroup(*block.submobjects[g:g + n]))
                g += n
            cards.append(dict(name=name, colour=colour, x=x, lines=lines,
                              block=block, rows=rows, own=own))

        # Both cards share one height, set by the longer prompt.
        body_h = max(c["block"].height for c in cards)
        CARD_H = body_h + 1.35
        top = 2.45
        ref_h = Text("Mg", font=MONO, font_size=size).height

        def highlight(card, phrase, colour):
            """A highlighter band behind one phrase, invisible until lit."""
            text = "\n".join(card["lines"])
            span = glyphs(card["block"], card["lines"], text.find(phrase),
                          len(phrase))
            band = Rectangle(width=span.width + 0.14, height=ref_h * 1.35,
                             stroke_width=0, fill_color=colour, fill_opacity=0)
            band.move_to(span).align_to(span, UP).shift(UP * ref_h * 0.18)
            return band

        boxes, heads, bands = VGroup(), VGroup(), {"shared": [], "own": []}
        for c in cards:
            box = RoundedRectangle(width=CARD_W, height=CARD_H,
                                   corner_radius=0.16, stroke_color=c["colour"],
                                   stroke_width=2, fill_color=c["colour"],
                                   fill_opacity=0.05)
            box.move_to([c["x"], top - CARD_H / 2, 0])
            head = caption(c["name"], 26, c["colour"])
            head.move_to(box.get_top() + DOWN * 0.45)
            c["block"].next_to(head, DOWN, buff=0.35)
            c["block"].align_to(box.get_left() + RIGHT * 0.25, LEFT)
            boxes.add(box)
            heads.add(head)
            bands["shared"] += [highlight(c, p, WARM) for p in shared]
            bands["own"] += [highlight(c, p, c["colour"]) for p in c["own"]]

        # The bands go in first, so the text is drawn over them; they stay
        # invisible until lit. Lighting them only changes their opacity, which
        # does not move them in front of the text.
        every_band = bands["shared"] + bands["own"]
        self.add(*every_band)
        self.play(FadeIn(boxes), FadeIn(heads), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r) for c in cards for r in c["rows"]],
                              lag_ratio=0.05), run_time=1.2)
        self.wait(0.4)

        role, criterion = bands["shared"][0::2], bands["shared"][1::2]
        for step in (role, criterion):
            self.play(*[b.animate.set_fill(opacity=0.42) for b in step],
                      run_time=0.5)
            self.wait(0.4)
        self.play(*[b.animate.set_fill(opacity=0.38) for b in bands["own"]],
                  run_time=0.6)
        self.wait(1.8)
        self.play(FadeOut(Group(*every_band, boxes, heads,
                                *[c["block"] for c in cards], title)),
                  run_time=0.7)

    # ── 3. the two formats: said on the left, shown on the right ──────
    def formats_heading(self):
        head = scene_title("Scoring Formats")
        self.play(Write(head), run_time=0.9)
        return head

    def format_card(self, name, color, points, width=5.8, x=-3.07):
        """The left panel: what one format asks for, in words."""
        box = RoundedRectangle(width=width, height=4.9, corner_radius=0.18,
                               stroke_color=color, stroke_width=2,
                               fill_color=color, fill_opacity=0.05)
        box.move_to([x, -0.35, 0])
        title = caption(name, 36, color)
        title.move_to(box.get_top() + DOWN * 0.52)
        items = VGroup()
        for lines in points:
            body = VGroup(*[caption(t, 21, INK) for t in lines])
            body.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            fit(body, width - 1.0)
            dot = Dot(radius=0.055, color=color)
            dot.move_to([body.get_left()[0] - 0.28, body[0].get_center()[1], 0])
            items.add(VGroup(dot, body))
        items.arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        items.next_to(title, DOWN, buff=0.45)
        items.align_to(box.get_left() + RIGHT * 0.52, LEFT)
        return box, title, items

    def formats_likert(self, head=None):
        """
        Bullets, then the thing itself: a 4x4 grid scored round after round.
        Three of the ten rounds are drawn; the counter says "of 10" so the
        rest are implied rather than sat through.
        """
        own = head is None
        if own:
            head = self.formats_heading()

        box, title, items = self.format_card("Likert", PICK, [
            ["4×4 grid — sixteen images per call"],
            ["the model picks the interesting",
             "ones, and scores each 1–5"],
            ["unselected images score 0;",
             "every image is seen ten times"],
            ["total 0–50  ·  51 possible scores"],
        ])
        self.play(FadeIn(box), Write(title), run_time=1.0)
        self.play(FadeIn(items, shift=RIGHT * 0.25), run_time=0.7)

        rng = random.Random(3)
        GX, GY, CELL = 3.4, -0.15, 1.18
        # Weighted like the real distribution rather than evenly: most of a
        # grid scores nothing (8,098 of 20,000 images score zero).
        names = [LENS] + self.high[1:3] + self.mid[:4] + self.low[:9]

        def cell(i):
            return [GX + (i % 4 - 1.5) * (CELL + 0.14),
                    GY + (1.5 - i // 4) * (CELL + 0.14), 0]

        def rest_stroke(nm):
            return (WARM, 3) if nm == LENS else (DIM, 1.5)

        # One tile per image for the whole beat: a reshuffle moves them to new
        # cells rather than swapping in a fresh grid, which blinked the grid
        # off and drew empty frames before the new images arrived.
        rng.shuffle(names)
        tiles = {}
        for i, nm in enumerate(names):
            im = cutout(nm, size=CELL).move_to(cell(i))
            tiles[nm] = (im, framed(im, *rest_stroke(nm)))
        grid = Group(*[tiles[nm][0] for nm in names])
        boxes = VGroup(*[tiles[nm][1] for nm in names])
        self.play(FadeIn(grid), Create(boxes), run_time=0.8)

        round_t, total_t = ValueTracker(1), ValueTracker(0)
        # A DecimalNumber regrows from its centre, so leave slack either side
        # or "1" becoming "10" collides with the words around it.
        rlbl = VGroup(caption("round", 24, DIM), counter_on(round_t, 24, DIM),
                      caption("of 10", 24, DIM)).arrange(RIGHT, buff=0.32)
        tnum = counter_on(total_t, 28, WARM)
        # The number is pinned by its right edge, so it grows leftwards as
        # digits arrive; the label needs a wide enough gap to absorb that.
        score = VGroup(tnum, caption("/ 50", 28, WARM)).arrange(RIGHT, buff=0.16)
        tlbl = VGroup(caption("this one", 24, WARM), score).arrange(RIGHT,
                                                                   buff=0.5)
        readout = VGroup(rlbl, tlbl).arrange(RIGHT, buff=0.8)
        readout.move_to([GX, GY - 3.0, 0])
        pin_edge(tnum, RIGHT)   # only after the final move, or it snaps back
        self.play(FadeIn(readout), run_time=0.5)

        ROUNDS = 3
        for r in range(1, ROUNDS + 1):
            marks = VGroup()
            for nm in names:
                per_round = round(self.score_of(nm) / 10)
                if per_round <= 0:
                    continue
                im, frame = tiles[nm]
                if nm != LENS:
                    frame.set_stroke(PICK, 2.5)
                marks.add(badge(per_round,
                                im.get_corner(UR) + LEFT * 0.26
                                + DOWN * 0.26, size=0.42, font=24))
            self.play(LaggedStart(*[FadeIn(m, scale=1.4) for m in marks],
                                  lag_ratio=0.06),
                      run_time=0.8 if r == 1 else 0.55)
            self.play(total_t.animate.set_value(r * 5), run_time=0.35)
            self.wait(0.3 if r == 1 else 0.12)

            if r < ROUNDS:
                self.play(FadeOut(marks), run_time=0.3)
                rng.shuffle(names)
                # Arcs, so tiles trading places swing past each other instead
                # of all running through the middle of the grid at once.
                moves = []
                for i, nm in enumerate(names):
                    im, frame = tiles[nm]
                    moves.append(im.animate(path_arc=PI / 2).move_to(cell(i)))
                    moves.append(frame.animate(path_arc=PI / 2)
                                 .move_to(cell(i)).set_stroke(*rest_stroke(nm)))
                self.play(*moves, round_t.animate.set_value(r + 1),
                          run_time=0.8)
            else:
                self.play(FadeOut(marks), run_time=0.3)

        self.wait(1.1)

        shown = Group(box, title, items, grid, boxes, readout)
        if own:
            self.play(FadeOut(Group(shown, head)), run_time=0.7)
        else:
            self.play(FadeOut(shown), run_time=0.6)

    def formats_tournament(self, head=None):
        """
        Same shape, other format: two rounds of keep-or-reject on a 2x2 grid,
        with the rejects dropping out of the pool between them.
        """
        own = head is None
        if own:
            head = self.formats_heading()

        box, title, items = self.format_card("Tournament", ALT, [
            ["2×2 grid — four images per call"],
            ["keep or reject, nothing else —",
             "no score, no reasoning"],
            ["rejects leave the pool;",
             "survivors reshuffle and go again"],
            ["score = rounds survived",
             "0–10  ·  11 possible scores"],
        ])
        self.play(FadeIn(box), Write(title), run_time=1.0)
        self.play(FadeIn(items, shift=RIGHT * 0.25), run_time=0.7)

        GX, CELL = 3.4, 1.9
        cols, rows = [GX - 1.07, GX + 1.07], [1.15, -1.45]

        round_t = ValueTracker(1)
        # A DecimalNumber regrows from its centre, so leave slack either side
        # or "1" becoming "10" collides with the words around it.
        rlbl = VGroup(caption("round", 24, DIM), counter_on(round_t, 24, DIM),
                      caption("of 10", 24, DIM)).arrange(RIGHT, buff=0.32)
        rlbl.move_to([GX, -3.05, 0])

        imgs, boxes = [], []
        for i, nm in enumerate([LENS, self.high[2], self.low[0], self.low[1]]):
            im = cutout(nm, size=CELL).move_to([cols[i % 2], rows[i // 2], 0])
            imgs.append(im)
            boxes.append(framed(im))
        self.play(FadeIn(Group(*imgs)), Create(VGroup(*boxes)), FadeIn(rlbl),
                  run_time=0.9)

        # Two of four go each round, and two fresh images take the empty seats
        # - the pool really is refilled from the images still active.
        KEEPS = [[True, True, False, False],
                 [True, False, True, False],
                 [True, True, False, False]]
        REFILL = [[self.high[4], self.mid[1]], [self.mid[2], self.low[2]]]

        for r, keeps in enumerate(KEEPS, start=1):
            marks = VGroup()
            for i, keep in enumerate(keeps):
                boxes[i].set_stroke(PICK if keep else DROP, 3.5)
                m = caption("keep" if keep else "reject", 20,
                            PICK if keep else DROP)
                m.next_to(imgs[i], DOWN, buff=0.1)
                marks.add(m)
            self.play(LaggedStart(*[FadeIn(m, scale=1.2) for m in marks],
                                  lag_ratio=0.2),
                      run_time=1.2 if r == 1 else 0.85)
            self.wait(0.5 if r == 1 else 0.3)

            out = [i for i, keep in enumerate(keeps) if not keep]
            # Rejects shrink away where they sit. Sliding them down, as this
            # used to, ran every bottom-row reject over the round counter.
            self.play(*[FadeOut(Group(imgs[i], boxes[i]), scale=0.8,
                                shift=DOWN * 0.25) for i in out],
                      *[FadeOut(marks[i]) for i in out], run_time=0.8)
            self.play(*[FadeOut(marks[i]) for i, keep in enumerate(keeps)
                        if keep], run_time=0.35)
            if r == len(KEEPS):
                break

            fresh, fboxes = Group(), VGroup()
            reset = [boxes[i].animate.set_stroke(DIM, 1.5)
                     for i, keep in enumerate(keeps) if keep]
            for slot, nm in zip(out, REFILL[r - 1]):
                im = cutout(nm, size=CELL)
                im.move_to([cols[slot % 2], rows[slot // 2], 0])
                imgs[slot], boxes[slot] = im, framed(im)
                fresh.add(im)
                fboxes.add(boxes[slot])
            self.play(FadeIn(Group(fresh, fboxes), shift=LEFT * 1.2), *reset,
                      round_t.animate.set_value(r + 1), run_time=1.0)

        self.wait(0.9)

        # Only what survived the last round is still on screen. Fading out an
        # image that has already been removed makes manim add it back at full
        # opacity for the length of the fade, which is what the rejects did.
        alive = [i for i, keep in enumerate(KEEPS[-1]) if keep]
        shown = Group(box, title, items, rlbl, *[imgs[i] for i in alive],
                      *[boxes[i] for i in alive])
        if own:
            self.play(FadeOut(Group(shown, head)), run_time=0.7)
        else:
            self.play(FadeOut(shown), run_time=0.6)

    def formats(self):
        head = self.formats_heading()
        self.formats_likert(head)
        self.formats_tournament(head)
        self.play(FadeOut(head), run_time=0.5)

    # ── 5. the whole of table 1 ──────────────────────────────
    def results(self):
        """
        Table 1, row by row, and nothing else on screen.

        Recall is the paper's threshold recall, the same 98.2% the recall beat
        derives, not the tie-corrected Expected Recall@2000 column. Precision
        is left out: it runs 7.4-8.1% across all four rows and separates
        nothing. The table is laid out at whatever width it wants and then
        scaled to the frame, so adding a column cannot push it off the edge.
        """
        # Text columns hang off their left edge, numbers off their right.
        COLS = [(-7.4, LEFT), (-3.4, LEFT), (-0.9, LEFT), (2.6, RIGHT),
                (4.5, RIGHT), (6.6, RIGHT)]
        GUTTER = 7.0
        HEADERS = ["Model", "Method", "Threshold", "Selected", "Recall",
                   "$ / 10,000"]
        ROWS = [("Gemini 3.1 Flash-Lite", STAGE_COLORS[2], "Likert", "33/50",
                 "2,095", "98.2%", "$5.05"),
                ("Qwen3.5-397B-A17B", ALT, "Likert", "36/50",
                 "1,984", "95.8%", "$6.69"),
                ("Gemini 3.1 Flash-Lite", STAGE_COLORS[2], "Tournament",
                 "Round 10", "2,019", "95.8%", "$3.66"),
                ("Qwen3.5-397B-A17B", ALT, "Tournament", "Round 10",
                 "2,099", "93.4%", "$1.80")]
        Y = [1.35, 0.0, -1.35, -2.70]

        def place(text, size, colour, x, edge, y):
            mob = caption(text, size, colour)
            mob.move_to([x, y, 0])
            mob.align_to([x, y, 0], edge)
            return mob

        header = VGroup(*[place(t, 23, DIM, x, e, 2.8)
                          for t, (x, e) in zip(HEADERS, COLS)])
        rule = Line([-7.5, 2.25, 0], [6.7, 2.25, 0], stroke_color=DIM,
                    stroke_width=1.5, stroke_opacity=0.55)

        cells = []
        for (name, col, method, thr, n, recall, cost), y in zip(ROWS, Y):
            values = [(name, col), (method, INK), (thr, INK), (n, INK),
                      (recall, INK), (cost, INK)]
            cells.append(VGroup(*[place(t, 26, c, x, e, y)
                                  for (t, c), (x, e) in zip(values, COLS)]))

        def flag(cell, y, text, colour):
            box = SurroundingRectangle(cell, color=colour, stroke_width=2.5,
                                       buff=0.14, corner_radius=0.08)
            lab = caption(text, 22, colour)
            lab.move_to([GUTTER, y, 0]).align_to([GUTTER, y, 0], LEFT)
            return box, lab

        best_box, best_lab = flag(cells[0][4], Y[0], "best recall", PICK)
        cheap_box, cheap_lab = flag(cells[3][5], Y[3], "cheapest", PICK)

        # One scale for everything, so the callout gutter is inside the
        # frame. The title is added after, so it keeps its own size.
        table = VGroup(header, rule, *cells, best_box, best_lab,
                       cheap_box, cheap_lab)
        fit(table, 13.2)
        head = scene_title("Results")
        table.next_to(head, DOWN, buff=0.75)

        self.play(Write(head), run_time=1.0)
        self.play(FadeIn(header), Create(rule), run_time=1.0)
        for row in cells:
            self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.8)
        self.wait(1.8)
        self.play(Create(best_box), FadeIn(best_lab), run_time=0.9)
        self.play(Create(cheap_box), FadeIn(cheap_lab), run_time=0.9)
        self.wait(3.2)
        self.play(FadeOut(VGroup(head, table)), run_time=0.8)

    # ── 4. the recall wall ───────────────────────────────────
    def recall(self):
        """
        All 167 reference anomalies at once, ordered by Likert score, with the
        score cutoff walking down from 50. Every tile lights the moment the
        cutoff reaches its score, so the wall fills in the order the method
        ranks them, and the two counters are the real shortlist size and the
        real number caught (assets/anomalies.json, written from the run CSV).
        """
        with open(os.path.join(ASSETS, "anomalies.json")) as f:
            data = json.load(f)
        meta, curve = data["tiles"], {int(k): v for k, v in data["curve"].items()}
        order = sorted(meta, key=lambda k: -(meta[k]["likert"] or 0))

        title = scene_title("Anomaly Recovery")
        head = caption("Recovering 167 previously identified anomalies",
                       26, DIM).move_to([0, 2.5, 0])
        self.play(Write(title), run_time=0.9)
        self.play(FadeIn(head), run_time=0.7)

        # Tiles are a little smaller than before to fit the title in above.
        COLS, PITCH, CELL = 21, 0.47, 0.43
        x0, y0 = -(COLS - 1) / 2 * PITCH, 1.085
        wall, blocks = Group(), {}
        for i, name in enumerate(order):
            im = ImageMobject(os.path.join(ANOM, f"{name}.png"))
            im.set_height(CELL)
            im.move_to([x0 + (i % COLS) * PITCH, y0 - (i // COLS) * PITCH, 0])
            # A dark lid over each tile, lifted when the cutoff reaches it.
            # Fading a rectangle is cheap; fading an ImageMobject is not.
            lid = Rectangle(width=CELL, height=CELL, fill_color="#04060d",
                            fill_opacity=0.9, stroke_width=0)
            lid.move_to(im.get_center())
            wall.add(Group(im, lid))
            blocks.setdefault(int(meta[name]["likert"] or 0), VGroup()).add(lid)
        self.play(LaggedStart(*[FadeIn(c) for c in wall], lag_ratio=0.006),
                  run_time=2.2)

        FIRST = 50
        cut_t = ValueTracker(FIRST)
        sel_t = ValueTracker(curve[FIRST][0])
        found_t = ValueTracker(curve[FIRST][1])
        cut_lbl = caption("Cutoff Score", 26, DIM)
        cut_num = counter_on(cut_t, 52, PICK)
        cut_num.next_to(cut_lbl, RIGHT, buff=0.4)
        cut_row = VGroup(cut_lbl, cut_num).move_to([0, 1.8, 0])

        RY = -3.55                  # the readout line along the bottom

        def readout(label, tracker, x, color, suffix=None, sx=None):
            lab = caption(label, 24, DIM).move_to([x, RY, 0])
            num = counter_on(tracker, 30, color)
            num.next_to(lab, RIGHT, buff=0.32)
            pin_edge(num, LEFT)     # digits grow rightwards into clear space
            row = VGroup(lab, num)
            if suffix:
                row.add(caption(suffix, 24, DIM).move_to([sx, RY, 0]))
            return row

        left = readout("Images to review", sel_t, -4.6, INK)
        right = readout("Anomalies found", found_t, 1.5, PICK, "of 167", 4.9)
        self.play(FadeIn(cut_row), FadeIn(left), FadeIn(right),
                  *([FadeOut(blocks[FIRST])] if FIRST in blocks else []),
                  run_time=0.9)
        self.wait(0.8)

        for c in range(FIRST - 1, 32, -1):
            lids = blocks.get(c)
            self.play(*([FadeOut(lids)] if lids else []),
                      cut_t.animate.set_value(c),
                      sel_t.animate.set_value(curve[c][0]),
                      found_t.animate.set_value(curve[c][1]),
                      run_time=0.9 if c == 33 else 0.75 if c > 47 else 0.5)
            if c == FIRST - 1:
                self.wait(0.4)      # let the early tiles register
        self.wait(1.3)

        # the three it never reaches: they score 31, 30 and 23, just under
        missed = VGroup(*[SurroundingRectangle(wall[i][0], color=DROP,
                                               stroke_width=2.5, buff=0.02)
                          for i, nm in enumerate(order)
                          if (meta[nm]["likert"] or 0) < 33])
        self.play(Create(missed), run_time=0.8)
        self.wait(0.9)

        punch = caption("Top ~10% of the dataset contains ~98% of anomalies.",
                        30, INK)
        punch.move_to([0, -2.9, 0])
        self.play(FadeIn(punch), run_time=0.9)
        self.wait(2.4)
        # Fade the images, not the wall: the wall still holds every lid the
        # sweep lifted, and fading a removed mobject brings it back for the
        # length of the fade. Only the three missed tiles still have theirs.
        lidded = [lid for score, lids in blocks.items() if score < 33
                  for lid in lids]
        self.play(FadeOut(Group(title, head, *[c[0] for c in wall], *lidded,
                                missed, cut_row, left, right, punch)),
                  run_time=0.8)

    # ── 6. what the top of the list actually contains ─────────────
    def funnel(self):
        """
        A standalone clip: 20,000 scored images narrowed to the released
        candidate catalogue, then the catalogue's highest-scoring members.

        Three steps, simplified for an explainer. 20,000 scored and 204 at
        score >= 45 are recomputed from the run CSV. 102 is the paper's, and
        it hides two screens it states: of the 204, 66 are already AnomalyMatch
        reference anomalies, and of the remaining 138, 33 match the O'Ryan
        interacting-galaxy catalogue, 1 the galaxy-mentions dataset and 2 a
        manual literature review. The wall is Appendix B, the 30
        highest-scoring of the 102.
        """
        with open(os.path.join(ASSETS, "candidates.json")) as f:
            data = json.load(f)
        meta, f_ = data["tiles"], data["funnel"]
        order = sorted(meta, key=lambda k: -(meta[k]["score"] or 0))

        steps = [
            (f_["scored"], ["images scored by VLMs"]),
            (f_["above"], ["images scored 45+/50"]),
            (f_["released"], ["non-previously identified anomalies"]),
        ]

        rows = VGroup()
        for i, (n, label) in enumerate(steps):
            col = PICK if i == len(steps) - 1 else INK
            num = number(n, 38, col)
            body = VGroup(*[caption(t, 25, DIM) for t in label])
            body.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            body.next_to(num, RIGHT, buff=0.5)
            body.align_to(num, UP)
            rows.add(VGroup(num, body))
        rows.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        # arrange() aligns on each row's bounding box, so re-hang the numbers
        # off one right edge and the labels off one left edge.
        for row in rows:
            row[0].align_to(rows[0][0], RIGHT)
            row[1].align_to(rows[0][1], LEFT)
        rows.move_to([0, -0.1, 0])

        spine = min(r[0].get_left()[0] for r in rows) - 0.45
        links = VGroup()
        for a, b in zip(rows, rows[1:]):
            links.add(Line([spine, a.get_bottom()[1] - 0.1, 0],
                           [spine, b.get_top()[1] + 0.1, 0],
                           stroke_color=DIM, stroke_width=1.5,
                           stroke_opacity=0.5))

        title = scene_title("New Anomaly Candidates")
        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(rows[0], shift=DOWN * 0.2), run_time=1.2)
        for i in range(1, len(steps)):
            self.play(Create(links[i - 1]), FadeIn(rows[i], shift=DOWN * 0.2),
                      run_time=1.1 if i < len(steps) - 1 else 1.3)
        self.wait(1.8)

        # the last row becomes the heading and the objects themselves come up
        head = VGroup(number(f_["released"], 36, PICK),
                      caption("candidates for further investigation", 28, DIM))
        head.arrange(RIGHT, buff=0.35).move_to([0, 2.3, 0])
        # Clear the rows above first: moving the last row up through them,
        # while its label morphed into the new one, read as a glitch. The
        # number keeps its glyphs, so it can glide; the words cross-fade.
        last = rows[len(steps) - 1]
        self.play(FadeOut(VGroup(*rows[:-1], links)), run_time=0.5)
        self.play(ReplacementTransform(last[0], head[0]),
                  FadeOut(last[1], shift=UP * 0.3),
                  FadeIn(head[1], shift=UP * 0.3), run_time=0.9)

        sub = caption("Below are the 30 highest-scoring candidates", 26, DIM)
        sub.move_to([0, 1.65, 0])
        self.play(FadeIn(sub), run_time=0.6)

        COLS, PITCH, CELL = 10, 1.30, 1.20
        x0 = -(COLS - 1) / 2 * PITCH
        wall = Group()
        for i, name in enumerate(order):
            im = ImageMobject(os.path.join(CAND, f"{name}.png"))
            im.set_height(CELL)
            im.move_to([x0 + (i % COLS) * PITCH, 0.55 - (i // COLS) * PITCH, 0])
            wall.add(im)
        self.play(LaggedStart(*[FadeIn(t, scale=0.85) for t in wall],
                              lag_ratio=0.045), run_time=2.4)
        self.wait(3.0)
        self.play(FadeOut(Group(title, head, sub, wall)), run_time=0.8)


class Opening(Beats):
    def construct(self):
        self.opening()


class VLM(Beats):
    def construct(self):
        self.vlm()


class FormatsLikert(Beats):
    def construct(self):
        self.formats_likert()


class FormatsTournament(Beats):
    def construct(self):
        self.formats_tournament()


class Models(Beats):
    def construct(self):
        self.models()


class Prompts(Beats):
    def construct(self):
        self.prompts()


class Formats(Beats):
    def construct(self):
        self.formats()


class Recall(Beats):
    def construct(self):
        self.recall()


class Results(Beats):
    def construct(self):
        self.results()


class Funnel(Beats):
    def construct(self):
        self.funnel()


class Explainer(Beats):
    def construct(self):
        for beat in (self.opening, self.vlm, self.models, self.prompts,
                     self.formats, self.recall, self.results, self.funnel):
            beat()
            self.clear()
