"""
poster_score_distribution.py
------------------------------
Poster plot: distribution of Gemini-tournament ImageScore (0-10 = rounds
survived, per gemini_tournament.py) split into ground-truth interesting vs.
boring images, log-scale y-axis. Same source CSVs (subset_test/*.csv) and
ground-truth column as poster_recall_plot.py, but uses a single replicate
(gemini_tournament_1.csv) as a representative example rather than combining
all three subset runs.

A log y-axis is used because score 0 (eliminated in round 1) dominates by
orders of magnitude over the higher scores, and the interesting/boring split
is shown as grouped (not stacked) bars since log-scale bar stacking doesn't
add visually the way linear stacking does.

Usage:
    python poster_score_distribution.py [subset_test_dir]
Output:
    poster_score_distribution_gemini_tournament.png
"""

import os
import csv
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "subset_test")

INTERESTING_COLOR = "#146c2e"
BORING_COLOR = "#94a3b8"
TOURNAMENT_COLOR = "#d9770a"

EXAMPLE_FILE = "gemini_tournament_1.csv"


def parse_binary(value):
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "yes", "y", "true", "interesting", "selected"} else 0


# ── Load a single representative replicate, count per score ─────────────────
path = next(
    (os.path.join(root, f) for root, _dirs, fs in os.walk(AI_CSV_DIR) for f in fs if f == EXAMPLE_FILE),
    None,
)
if path is None:
    print(f"Error: {EXAMPLE_FILE} not found under {AI_CSV_DIR}")
    sys.exit(1)

interesting_counts = Counter()
boring_counts = Counter()

with open(path, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    header = next(reader)
    lower_header = [h.lower().strip() for h in header]
    score_idx = next((i for i, h in enumerate(lower_header) if h in ["imagescore", "image_score", "ai", "ai_selected"]), None)
    gt_idx = next((i for i, h in enumerate(lower_header) if h in ["interesting", "is_interesting", "label"]), None)
    if score_idx is None or gt_idx is None:
        print(f"Error: {EXAMPLE_FILE} missing required columns")
        sys.exit(1)
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        try:
            score = int(round(float(row[score_idx])))
            gt = parse_binary(row[gt_idx])
        except (ValueError, IndexError):
            continue
        if gt:
            interesting_counts[score] += 1
        else:
            boring_counts[score] += 1
print(f"Loaded {EXAMPLE_FILE}")

scores = sorted(set(interesting_counts) | set(boring_counts))
for s in scores:
    print(f"  score={s:>2}  boring={boring_counts.get(s, 0):>6}  interesting={interesting_counts.get(s, 0):>4}")

# ── Poster plot ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.linewidth": 1.4,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

x = list(range(len(scores)))
width = 0.38
boring_vals = [boring_counts.get(s, 0) for s in scores]
interesting_vals = [interesting_counts.get(s, 0) for s in scores]

ax.bar([xi - width / 2 for xi in x], boring_vals, width, color=BORING_COLOR,
       edgecolor="white", linewidth=0.8, label="Boring", zorder=3)
ax.bar([xi + width / 2 for xi in x], interesting_vals, width, color=INTERESTING_COLOR,
       edgecolor="white", linewidth=0.8, label="Interesting", zorder=3)

# Value labels above each bar (skip zeros to avoid clutter on the log axis).
for xi, v in zip(x, boring_vals):
    if v > 0:
        ax.text(xi - width / 2, v * 1.15, f"{v:,}", ha="center", va="bottom", fontsize=14, color=BORING_COLOR, fontweight="bold")
for xi, v in zip(x, interesting_vals):
    if v > 0:
        ax.text(xi + width / 2, v * 1.15, f"{v:,}", ha="center", va="bottom", fontsize=14, color=INTERESTING_COLOR, fontweight="bold")

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in scores], fontsize=17, fontweight="bold")
ax.tick_params(axis="y", labelsize=15)
ax.set_xlabel("ImageScore (rounds survived, 0-10)", fontsize=21, fontweight="bold", labelpad=12, color=TOURNAMENT_COLOR)
ax.set_ylabel("Image Count", fontsize=20, fontweight="bold", labelpad=12)
ax.grid(axis="y", which="major", color="#dddddd", linewidth=1, zorder=0)
ax.grid(axis="y", which="minor", color="#eeeeee", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.legend(loc="upper center", fontsize=16, frameon=True, facecolor="white", edgecolor="#cccccc", ncol=2)

fig.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.14)
out = os.path.join(SCRIPT_DIR, "poster_score_distribution_gemini_tournament.png")
fig.savefig(out)
print(f"\nSaved {out}")
