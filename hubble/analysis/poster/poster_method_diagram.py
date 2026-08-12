"""
poster_method_diagram.py
--------------------------
Poster diagrams explaining the two classification methods (Tournament and
Likert), based directly on the actual logic in gemini_tournament.py /
qwen_tournament.py and gemini_likert.py / qwen_likert.py:

Tournament (elimination bracket):
    - Images grouped into 2x2 grids (batches of 4).
    - Model independently keeps/rejects each image in the grid.
    - Kept images advance to the next round; rejected images are dropped.
    - Repeated for up to 10 rounds.
    - ImageScore = number of rounds survived (0-10).

Likert (repeated rating pool):
    - Images grouped into 4x4 grids (batches of 16).
    - Model selects any subset of the 16 and rates each 1-5.
    - Every image is re-shown across ~10 random groupings.
    - ImageScore = sum of ratings received across all rounds.

Two takes on how to draw this for a poster, at 16x8in / 200dpi, white bg:
    poster_method_diagram_icons.png      - pictogram/metaphor style
                                            (bracket narrowing vs. star pool
                                            growing), minimal text
    poster_method_diagram_flowchart.png  - literal flowchart style
                                            (boxes, decision diamond, loop
                                            arrows), more technical detail
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
from matplotlib.path import Path
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TOURNAMENT_COLOR = "#d9770a"   # amber
LIKERT_COLOR = "#6d28d9"       # violet
INTERESTING_COLOR = "#146c2e"  # green (kept / high rating)
REJECT_COLOR = "#b91c1c"       # red (rejected)
GRAY = "#9ca3af"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def new_ax(figsize, dpi=200):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def arrow(ax, xy0, xy1, color="#333333", lw=2.5, mutation_scale=20, style="-|>", connectionstyle=None):
    a = FancyArrowPatch(xy0, xy1, arrowstyle=style, color=color, lw=lw,
                         mutation_scale=mutation_scale, zorder=4,
                         connectionstyle=connectionstyle, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


# ═════════════════════════════════════════════════════════════════════════
# OPTION A — ICON / PICTOGRAM STYLE (portrait, vertical flow, no title)
# ═════════════════════════════════════════════════════════════════════════
def plot_icons():
    fig, ax = plt.subplots(figsize=(11, 12), dpi=200)
    ax.set_xlim(0, 92)  # 92:100 matches the 11:12 figure aspect -> square units
    ax.set_ylim(0, 100)
    ax.axis("off")

    icon_dim = 17.0
    arrow_gap = 1.8
    box_w = 37

    col_x = {"tournament": 23, "likert": 69}

    for method, cx, color, title, subtitle in [
        ("tournament", col_x["tournament"], TOURNAMENT_COLOR, "TOURNAMENT",
         "elimination — keep or reject, round after round"),
        ("likert", col_x["likert"], LIKERT_COLOR, "LIKERT",
         "accumulation — rate every image, round after round"),
    ]:
        y = 99  # each column gets the full height, independent of the other

        ax.text(cx, y, title, fontsize=25, fontweight="bold", color=color, ha="center", va="top")
        y -= 5.2
        ax.text(cx, y, subtitle, fontsize=12.5, color="#444444", ha="center", va="top", style="italic")
        y -= 4.6

        # ── Step 1: batch grid (2x2 for tournament, 4x4 for likert) ──────
        batch_center = y - icon_dim / 2
        n = 2 if method == "tournament" else 4
        bc = icon_dim / n
        for r in range(n):
            for c in range(n):
                ax.add_patch(Rectangle(
                    (cx - icon_dim / 2 + c * bc + 0.25, batch_center - icon_dim / 2 + r * bc + 0.25),
                    bc - 0.5, bc - 0.5,
                    facecolor="white", edgecolor=color, linewidth=3.0, zorder=2,
                ))
        y = batch_center - icon_dim / 2 - 1.2
        label_batch = "Batch of 4\n(2×2 grid)" if method == "tournament" else "Batch of 16\n(4×4 grid)"
        ax.text(cx, y, label_batch, fontsize=15.5, fontweight="bold", ha="center", va="top")
        y -= 5.4

        arrow(ax, (cx, y), (cx, y - arrow_gap), color=color, lw=3.5, mutation_scale=30)
        y -= arrow_gap

        # ── Step 2: model verdict ─────────────────────────────────────────
        verdict_center = y - icon_dim / 2
        if method == "tournament":
            verdicts = [True, False, False, True]
            vn = 2
            vc = icon_dim / vn
            for i, keep in enumerate(verdicts):
                r, c = divmod(i, vn)
                vx = cx - icon_dim / 2 + c * vc + vc / 2
                vy = verdict_center + icon_dim / 2 - r * vc - vc / 2
                mark_color = INTERESTING_COLOR if keep else REJECT_COLOR
                ax.add_patch(Rectangle((vx - vc / 2 + 0.3, vy - vc / 2 + 0.3), vc - 0.6, vc - 0.6,
                                        facecolor="white", edgecolor=mark_color, linewidth=3.2, zorder=2))
                mark = "✓" if keep else "✗"
                ax.text(vx, vy, mark, fontsize=46, fontweight="bold", color=mark_color,
                        ha="center", va="center", zorder=3)
            verdict_label = "Model keeps or\nrejects each image"
        else:
            starred = {2, 5, 9, 13}
            ratings = {2: 3, 5: 5, 9: 2, 13: 4}
            vn = 4
            vc = icon_dim / vn
            for i in range(16):
                r, c = divmod(i, vn)
                vx = cx - icon_dim / 2 + c * vc + vc / 2
                vy = verdict_center + icon_dim / 2 - r * vc - vc / 2
                if i in starred:
                    ax.add_patch(Rectangle((vx - vc / 2 + 0.25, vy - vc / 2 + 0.25), vc - 0.5, vc - 0.5,
                                            facecolor="#efe6fb", edgecolor=LIKERT_COLOR, linewidth=2.6, zorder=2))
                    ax.text(vx, vy, f"★{ratings[i]}", fontsize=17, fontweight="bold",
                            color=LIKERT_COLOR, ha="center", va="center", zorder=3)
                else:
                    ax.add_patch(Rectangle((vx - vc / 2 + 0.25, vy - vc / 2 + 0.25), vc - 0.5, vc - 0.5,
                                            facecolor="white", edgecolor="#cccccc", linewidth=1.8, zorder=2))
            verdict_label = "Model selects a subset\n& rates each 1–5"
        y = verdict_center - icon_dim / 2 - 1.2
        ax.text(cx, y, verdict_label, fontsize=15.5, fontweight="bold", ha="center", va="top")
        y -= 5.4

        arrow(ax, (cx, y), (cx, y - arrow_gap), color=color, lw=3.5, mutation_scale=30)
        y -= arrow_gap

        # ── Step 3: round-loop metaphor (shrinking bars / growing bars) ──
        loop_top = y
        bar_w, gap, max_h = 5.6, 2.0, 11.0
        counts = [4, 3, 2, 1] if method == "tournament" else [1, 2, 3, 4]
        total_w = len(counts) * (bar_w + gap) - gap + 8  # + room for "..."
        start_x = cx - total_w / 2
        for i, c in enumerate(counts):
            h = max_h * (c / 4)
            bx = start_x + i * (bar_w + gap)
            ax.add_patch(Rectangle((bx, loop_top - h), bar_w, h, facecolor=color,
                                    alpha=0.35 + 0.65 * (i / (len(counts) - 1)), zorder=2))
        ax.text(start_x + len(counts) * (bar_w + gap) + 1.2, loop_top - max_h / 2, "...",
                fontsize=32, fontweight="bold", color=color, va="center")
        y = loop_top - max_h - 1.2
        loop_caption = "pool shrinks each round\n(× up to 10 rounds)" if method == "tournament" \
            else "ratings accumulate each round\n(× 10 random groupings)"
        ax.text(cx, y, loop_caption, fontsize=14, fontweight="bold", ha="center", va="top", color="#333333")
        y -= 5.0

        arrow(ax, (cx, y), (cx, y - arrow_gap), color=color, lw=3.5, mutation_scale=30)
        y -= arrow_gap

        # ── Step 4: final score ───────────────────────────────────────────
        score_h = 11.5
        score_center = y - score_h / 2
        ax.add_patch(FancyBboxPatch((cx - box_w / 2, score_center - score_h / 2), box_w, score_h,
                                     boxstyle="round,pad=0.6,rounding_size=1.6",
                                     facecolor=color, edgecolor="none", alpha=0.15, zorder=1))
        ax.add_patch(FancyBboxPatch((cx - box_w / 2, score_center - score_h / 2), box_w, score_h,
                                     boxstyle="round,pad=0.6,rounding_size=1.6",
                                     facecolor="none", edgecolor=color, linewidth=2.6, zorder=2))
        score_text = "ImageScore =\nrounds survived (0–10)" if method == "tournament" else \
            "ImageScore = Σ ratings\nacross rounds (max 50)"
        ax.text(cx, score_center, score_text, fontsize=15.5, fontweight="bold", color=color,
                ha="center", va="center", zorder=3)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
    out = os.path.join(SCRIPT_DIR, "poster_method_diagram_icons.png")
    fig.savefig(out)
    print(f"Saved {out}")


# ═════════════════════════════════════════════════════════════════════════
# OPTION B — LITERAL FLOWCHART STYLE
# ═════════════════════════════════════════════════════════════════════════
def box(ax, cx, cy, w, h, text, color, fontsize=13, fontweight="bold", text_color=None, fill_alpha=0.12):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                 boxstyle="round,pad=0.3,rounding_size=1.2",
                                 facecolor=color, alpha=fill_alpha, edgecolor=color, linewidth=2.2, zorder=2))
    ax.text(cx, cy, text, fontsize=fontsize, fontweight=fontweight, color=text_color or "#1a1a1a",
            ha="center", va="center", zorder=3)


def diamond(ax, cx, cy, w, h, text, color, fontsize=12.5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=0.12,
                          edgecolor=color, linewidth=2.2, zorder=2))
    ax.text(cx, cy, text, fontsize=fontsize, fontweight="bold", ha="center", va="center", zorder=3)


def plot_flowchart():
    fig, ax = new_ax((16, 7.6))
    ax.set_ylim(28, 100)  # crop the empty lower band instead of leaving it blank

    col_x = {"tournament": 24, "likert": 76}
    box_w = 30

    for method, cx, color, title in [
        ("tournament", col_x["tournament"], TOURNAMENT_COLOR, "TOURNAMENT"),
        ("likert", col_x["likert"], LIKERT_COLOR, "LIKERT"),
    ]:
        ax.text(cx, 97, title, fontsize=24, fontweight="bold", color=color, ha="center")

        y = 90
        box(ax, cx, y, box_w, 6.5, "Active image pool", color)

        y2 = y - 11.5
        arrow(ax, (cx, y - 3.5), (cx, y2 + 4), color=color)
        grid_txt = "Shuffle & group into\n2×2 grids (batches of 4)" if method == "tournament" \
            else "Randomly group into\n4×4 grids (batches of 16)"
        box(ax, cx, y2, box_w, 7.5, grid_txt, color)

        y3 = y2 - 12.5
        arrow(ax, (cx, y2 - 4), (cx, y3 + 4.5), color=color)
        model_txt = "Model evaluates each image\nin the grid independently" if method == "tournament" \
            else "Model selects any subset of\nthe 16 & rates each 1–5"
        box(ax, cx, y3, box_w, 8.5, model_txt, color)

        # Loop-back x sits in the gap between the two columns (30 <= x <= 70),
        # on this column's own side, so labels never cross into the other lane.
        loop_x = cx + 16 if method == "tournament" else cx - 16
        loop_label_ha = "left" if method == "tournament" else "right"
        loop_label_x = loop_x + 1.5 if method == "tournament" else loop_x - 4.5

        if method == "tournament":
            y4 = y3 - 14
            arrow(ax, (cx, y3 - 4.5), (cx, y4 + 5.5), color=color)
            diamond(ax, cx, y4, 24, 10.5, "Keep this\nimage?", color)

            # YES -> advance, loop back up to the grid step
            yes_x = cx + 16
            arrow(ax, (cx + 12, y4), (yes_x, y4), color=INTERESTING_COLOR)
            ax.text((cx + 12 + yes_x) / 2, y4 + 2.0, "YES", fontsize=11, fontweight="bold",
                    color=INTERESTING_COLOR, ha="center")
            box(ax, yes_x, y4, 13, 6, "Advance to\nnext round", INTERESTING_COLOR, fontsize=10.5)
            arrow(ax, (yes_x, y4 + 3), (loop_x, y2), color=INTERESTING_COLOR,
                  connectionstyle="arc3,rad=-0.3")
            ax.text(loop_label_x, (y4 + y2) / 2, "repeat ×\nup to 10\nrounds", fontsize=9.5,
                    color="#333333", ha=loop_label_ha, va="center")

            # NO -> eliminated
            no_x = cx - 16
            arrow(ax, (cx - 12, y4), (no_x, y4), color=REJECT_COLOR)
            ax.text((cx - 12 + no_x) / 2, y4 + 2.0, "NO", fontsize=11, fontweight="bold",
                    color=REJECT_COLOR, ha="center")
            box(ax, no_x, y4, 13, 6, "Eliminated\n(dropped)", REJECT_COLOR, fontsize=10.5)

            y5 = y4 - 12.5
            arrow(ax, (cx, y4 - 5.5), (cx, y5 + 4.5), color=color)
            box(ax, cx, y5, box_w, 8, "ImageScore =\n# rounds survived (0–10)",
                color, fontsize=13, fill_alpha=0.22)
        else:
            y4 = y3 - 13
            arrow(ax, (cx, y3 - 4.5), (cx, y4 + 4), color=color)
            box(ax, cx, y4, box_w, 8, "Accumulate rating into\nthat image's running total", color)

            arrow(ax, (cx - 15, y4), (loop_x, y4), color=color)
            arrow(ax, (loop_x, y4 + 0.2), (loop_x, y2), color=color, connectionstyle="arc3,rad=-0.3")
            ax.text(loop_label_x, (y4 + y2) / 2, "repeat ×\n10 random\ngroupings", fontsize=9.5,
                    color="#333333", ha=loop_label_ha, va="center")

            y5 = y4 - 12.5
            arrow(ax, (cx, y4 - 4), (cx, y5 + 4.5), color=color)
            box(ax, cx, y5, box_w, 8, "ImageScore =\nΣ ratings across all rounds",
                color, fontsize=13, fill_alpha=0.22)

    fig.suptitle("Classification Method Logic", fontsize=28, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    out = os.path.join(SCRIPT_DIR, "poster_method_diagram_flowchart.png")
    fig.savefig(out)
    print(f"Saved {out}")


plot_icons()
plot_flowchart()
print("\nDone.")
