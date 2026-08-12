"""
poster_recall_plot.py
----------------------
Poster-ready static plot: Recall vs. $ Cost for the tournament and likert
methods (Gemini + Qwen), on a 12x7 white-background figure.

Data loading / Recall@N math / cost model are lifted straight from
plots_only_html.py:
    - ground truth + AI ImageScore per image, per replicate CSV
    - exact tie-aware recall@N curve (recall_curve_from)
    - per-provider token pricing -> $ cost per 1,000 images evaluated
    - replicates grouped by (provider, format), everything averaged across
      replicates (no error bars plotted here, per request)

Cost sits on the x-axis (each method occupies one x position, since cost
doesn't depend on N). Each method is drawn as a vertical dumbbell: a hollow
point at Recall@100, a filled point at Recall@1000, connected by an arrow
(since recall always rises with N). Method identity is color, and the N=100
vs. N=1000 cutoff is marker fill — both explained in side legends rather than
axis labels, so nothing is on the plot but the cost/recall data itself.

Usage:
    python poster_recall_plot.py [subset_test_dir]
Output (in this directory):
    poster_recall_vs_cost.png
"""

import os
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "subset_test")

N_SMALL = 100
N_LARGE = 1000

# Only these two test formats, per request.
FORMATS_KEEP = {"tournament", "likert"}

# USD per 1,000,000 tokens, keyed by provider (same as plots_only_html.py).
PRICING = {
    "gemini": {"input": 0.25, "output": 1.50},
    "qwen":   {"input": 0.45, "output": 3.00},
}

# Provider = hue family (Gemini -> blue, Qwen -> green); format = shade
# within that family (tournament = light, likert = dark) so the four
# methods read as two consistent, related pairs rather than four unrelated
# colors.
COLOR_MAP = {
    ("gemini", "tournament"): "#7fb8f5",
    ("gemini", "likert"):     "#0b3d91",
    ("qwen", "tournament"):   "#8fd9a8",
    ("qwen", "likert"):       "#146c2e",
}


def parse_binary(value):
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "yes", "y", "true", "interesting", "selected"} else 0


def clean_filename(value):
    return os.path.basename((value or "").strip())


def describe_method(stem, parent=""):
    low = f"{stem} {parent}".lower()
    if "gemini" in low:
        provider = "gemini"
    elif "qwen" in low:
        provider = "qwen"
    else:
        provider = "other"

    if "single_elim" in low or "single-elim" in low or "singleelim" in low:
        fmt = "single-elim"
    elif "tournament" in low:
        fmt = "tournament"
    elif "likert" in low:
        fmt = "likert"
    elif "hybrid" in low:
        fmt = "hybrid"
    else:
        fmt = None
    return provider, fmt


def recall_curve_from(scores, ground_truth, total_positives):
    if total_positives == 0 or not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    curve = [0.0] * n
    cum_images = 0
    cum_tp = 0.0
    i = 0
    while i < n:
        band_score = ranked[i][1]
        j = i
        band_tp = 0
        while j < n and ranked[j][1] == band_score:
            if ground_truth.get(ranked[j][0], False):
                band_tp += 1
            j += 1
        b = j - i
        for k in range(1, b + 1):
            expected_tp = cum_tp + band_tp * (k / b)
            curve[cum_images + k - 1] = expected_tp / total_positives
        cum_images += b
        cum_tp += band_tp
        i = j
    return curve


def compute_cost(provider, input_tokens, output_tokens):
    rates = PRICING.get(provider)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + \
           (output_tokens / 1_000_000) * rates["output"]


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ── 2. LOAD + COMPUTE recall@100 / recall@1000 / cost PER REPLICATE ─────────
print(f"Loading AI CSVs from {AI_CSV_DIR}...")
ai_files = sorted(
    os.path.join(root, f)
    for root, _dirs, files in os.walk(AI_CSV_DIR)
    for f in files
    if f.lower().endswith(".csv")
)
if not ai_files:
    print(f"Error: no CSVs found in {AI_CSV_DIR}")
    sys.exit(1)

grouped = defaultdict(lambda: {"r100": [], "r1000": [], "cost_per_1k": []})

for path in ai_files:
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    provider, fmt = describe_method(stem, parent)
    if fmt is None or fmt not in FORMATS_KEEP:
        continue

    scores, ground_truth = {}, {}
    footer_input = None
    footer_output = None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        lower_header = [h.lower().strip() for h in header]
        fname_idx = next((i for i, h in enumerate(lower_header) if h in ["filename", "name"]), None)
        score_idx = next((i for i, h in enumerate(lower_header) if h in ["imagescore", "image_score", "ai", "ai_selected"]), None)
        gt_idx = next((i for i, h in enumerate(lower_header) if h in ["interesting", "is_interesting", "label"]), None)
        if fname_idx is None or score_idx is None or gt_idx is None:
            print(f"Skipping {os.path.basename(path)}: missing columns")
            continue
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                key = row[0].lstrip("# ").strip().lower()
                if key == "totalinputtokens" and len(row) > 1:
                    footer_input = int(str(row[1]).replace(",", "").strip())
                elif key == "totaloutputtokens" and len(row) > 1:
                    footer_output = int(str(row[1]).replace(",", "").strip())
                continue
            try:
                score = float(row[score_idx])
                fname = clean_filename(row[fname_idx])
                gt = parse_binary(row[gt_idx])
                if not fname:
                    continue
                scores[fname] = score
                ground_truth[fname] = bool(gt)
            except (ValueError, IndexError):
                continue

    if not scores:
        continue

    total_positives = sum(1 for v in ground_truth.values() if v)
    curve = recall_curve_from(scores, ground_truth, total_positives)
    if not curve:
        continue

    r100 = curve[min(N_SMALL, len(curve)) - 1]
    r1000 = curve[min(N_LARGE, len(curve)) - 1]

    images_evaluated = len(scores)
    input_tokens = footer_input or 0
    output_tokens = footer_output or 0
    cost = compute_cost(provider, input_tokens, output_tokens)
    cost_per_1k = (cost / images_evaluated) * 1000 if images_evaluated else 0.0

    grouped[(provider, fmt)]["r100"].append(r100)
    grouped[(provider, fmt)]["r1000"].append(r1000)
    grouped[(provider, fmt)]["cost_per_1k"].append(cost_per_1k)
    print(f"  {os.path.basename(path)}: recall@{N_SMALL}={r100:.3f}  "
          f"recall@{N_LARGE}={r1000:.3f}  cost/1k img=${cost_per_1k:.3f}")

if not grouped:
    print("No usable data.")
    sys.exit(1)

methods = []
for (provider, fmt), vals in grouped.items():
    methods.append({
        "label": f"{provider.upper()} {fmt}",
        "provider": provider,
        "format": fmt,
        "color": COLOR_MAP.get((provider, fmt), "#64748b"),
        "r100": mean(vals["r100"]),
        "r1000": mean(vals["r1000"]),
        "cost_per_1k": mean(vals["cost_per_1k"]),
    })

# Sort by recall@1000 descending so the plot reads top-to-bottom / left-to-right
methods.sort(key=lambda m: m["r1000"], reverse=True)

for m in methods:
    print(f"{m['label']:>20s}  R@100={m['r100']:.3f}  R@1000={m['r1000']:.3f}  "
          f"cost/1k img=${m['cost_per_1k']:.3f}")

# ── 3. POSTER PLOTTING SETUP ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.linewidth": 1.4,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

FIGSIZE = (12, 6)
DPI = 200


# ── Recall vs. Cost scatter — cost on the x-axis, no legend: every point is
#    labeled directly (method name at the top point, value at each point) ───
def plot_recall_vs_cost():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # The two "tournament" methods sit almost on top of each other in cost
    # ($0.216 vs $0.238) and near-identically low in recall@100 (~4-5%), so
    # their labels are hand-nudged apart (dx in axes-fraction-ish data units,
    # ha flipped) to keep them legible without a legend to fall back on.
    label_offset = {
        ("qwen", "tournament"): {"top_dx": -0.012, "top_ha": "right", "bot_dx": -0.014, "bot_ha": "right"},
        ("gemini", "tournament"): {"top_dx": 0.012, "top_ha": "left", "bot_dx": 0.014, "bot_ha": "left"},
        ("gemini", "likert"): {"top_dx": 0, "top_ha": "center", "bot_dx": 0, "bot_ha": "center"},
        ("qwen", "likert"): {"top_dx": 0, "top_ha": "center", "bot_dx": 0, "bot_ha": "center"},
    }

    for m in methods:
        x = m["cost_per_1k"]
        off = label_offset.get((m["provider"], m["format"]), {"top_dx": 0, "top_ha": "center", "bot_dx": 0, "bot_ha": "center"})

        ax.plot([x, x], [m["r100"], m["r1000"]], color=m["color"], lw=2.2, alpha=0.5, zorder=1)
        ax.scatter([x], [m["r100"]], s=240, color="white", edgecolor=m["color"], linewidth=3, zorder=3)
        ax.scatter([x], [m["r1000"]], s=300, color=m["color"], edgecolor="white", linewidth=1.5, zorder=3)

        # Top point: method name + its value + which cutoff this is (compact
        # "@1000" suffix rather than the full phrase, so 8 labels don't
        # collide across the plot).
        ax.annotate(
            f"{m['label']}\n{m['r1000']:.0%} Recall@{N_LARGE}",
            xy=(x, m["r1000"]), xytext=(x + off["top_dx"], m["r1000"] + 0.035),
            fontsize=16, fontweight="bold", color=m["color"], ha=off["top_ha"], va="bottom",
        )
        # Bottom point: its value + which cutoff (method identity already given above).
        ax.annotate(
            f"{m['r100']:.0%} Recall@{N_SMALL}",
            xy=(x, m["r100"]), xytext=(x + off["bot_dx"], m["r100"] - 0.035),
            fontsize=15, fontweight="bold", color=m["color"], ha=off["bot_ha"], va="top",
        )

    ax.set_xlabel("Cost ($ / 1,000 images)", fontsize=21, fontweight="bold", labelpad=12)
    ax.set_ylabel("Recall rate", fontsize=21, fontweight="bold", labelpad=18)
    xmax = max(m["cost_per_1k"] for m in methods)
    ax.set_xlim(0, xmax * 1.3)
    ax.set_ylim(-0.05, 1.12)
    ax.xaxis.set_major_formatter(lambda x, _: f"${x:.2f}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(color="#dddddd", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.13)
    out = os.path.join(SCRIPT_DIR, "poster_recall_vs_cost.png")
    fig.savefig(out)
    print(f"Saved {out}")


plot_recall_vs_cost()
print("\nDone.")
