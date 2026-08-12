"""
plots_only_html.py
------------------
Reads the Hubble AI classification CSVs (Gemini vs. Qwen, across the test
formats: tournament, single-elimination, likert/grid, and hybrid) and
generates a comprehensive HTML dashboard. The headline comparison is GEMINI vs
QWEN (the underlying model provider).

Replicates / error bars
    Each (provider, format) is run on several random subsets, one CSV per
    subset. The six test scripts (gemini/qwen × likert/single_elim/tournament)
    write their runs into ./subset_test as:

        subset_test/gemini_likert_1.csv
        subset_test/gemini_likert_2.csv
        subset_test/qwen_single_elim_1.csv
        subset_test/gemini_tournament_3.csv
        ...  i.e. {provider}_{format}_{run}.csv

    Files are grouped by (provider, format); the trailing run number is only a
    replicate id, so the number of replicates per group may vary freely (e.g.
    4 in one group, 5 in another). Provider and format are detected from the
    filename, or, if the format token is absent, from the parent folder name
    (so a per-format-subfolder layout also works). Every scalar metric is
    reported as mean ± sample standard deviation (ddof=1) across a group's
    replicates.

Performance (Precision, Recall, F1, PR AUC) is plotted against ACTUAL DOLLAR
COST, computed from each run's input/output token usage and the provider's
pricing.

Recall@N
    Recall@N is no longer a fixed table column. Instead there is a dedicated
    interactive scatter chart of Recall@N vs. cost: type any N and each method
    is plotted at (mean $ cost, mean recall among its top-N scored images),
    with x/y std error bars over the subsets.

    Ties at the cutoff are handled by the EXACT expected recall under a random
    ordering of the tied band (closed form), not by Monte-Carlo shuffling:

        recall@N = ( TP_above + TP_tie * (N - n_above) / n_tie ) / P

    where n_above / TP_above are the images (and true positives) scoring
    strictly above the boundary score, n_tie / TP_tie are the size and true
    positives of the tied band straddling rank N, and P = total positives.
    This yields fractional values like 12.4/20 but is deterministic and exact.

Pricing (USD per 1M tokens):
    Gemini (Google): $0.25 input   /  $1.50 output
    Qwen           : $0.45 input   /  $3.00 output

Token usage:
    - Every CSV carries a footer block (# TotalInputTokens /
      # TotalOutputTokens) with the exact input/output split. These are
      averaged across a group's replicates and reported as mean ± std.

Ground truth is taken directly from the `interesting` column embedded in each
CSV (1 = interesting, 0 = boring). For every method we vary the AI ImageScore
threshold against that fixed ground truth.

Usage:
    python plots_only_html.py ./subset_test
"""

import os
import io
import csv
import sys
import json
import base64
from collections import Counter, defaultdict

# Optional: image gallery needs h5py + Pillow (available in the gemini_env conda
# env). If they're missing the dashboard still builds; the gallery just shows
# placeholders. Run with that env to embed the thumbnails.
try:
    import h5py
    from PIL import Image
    HDF5_LIBS = True
except ImportError:
    HDF5_LIBS = False

# ── 1. CONFIGURATION ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "subset_test")
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "plots_only.html")

AI_FILE_PATTERN = "*.csv"
TOP_N = 100  # Default N for the interactive Recall@N chart and the provider cards.

# Bottom-of-page image gallery: top-N images per run, embedded from the HDF5.
HDF5_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    SCRIPT_DIR, "..", "hubble_data",
    "10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5",
)
GALLERY_TOP_N = 20   # how many top-scored images to show per run
GALLERY_THUMB_PX = 180  # thumbnail max edge (keeps the embedded HTML small)
GALLERY_JPEG_QUALITY = 80

# Cost model: USD per 1,000,000 tokens, keyed by provider.
PRICING = {
    "gemini": {"input": 0.25, "output": 1.50},
    "qwen":   {"input": 0.45, "output": 3.00},
}

# One brand color per provider for the head-to-head summary.
PROVIDER_COLORS = {
    "gemini": "#00e5a0",
    "qwen":   "#ff4d6d",
    "other":  "#94a3b8",
}


def parse_binary(value):
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "yes", "y", "true", "interesting", "selected"} else 0


def clean_filename(value):
    return os.path.basename((value or "").strip())


def parse_header_tokens(header):
    """Combined token total is stored as the last cell of the header row."""
    if not header:
        return 0
    try:
        return int(str(header[-1]).replace(",", "").strip())
    except (ValueError, IndexError):
        return 0


def describe_method(stem, parent=""):
    """
    Identify a run's provider and test format from its filename stem and, as a
    fallback, its parent folder name. Returns a display label, provider,
    format, and a color grouped by provider (cool tones = Gemini, warm tones =
    Qwen). Grouping into replicates is done later on (provider, format), so the
    seed/replicate portion of the filename does not need to be parsed out.
    """
    # Search filename first, then fall back to the folder name for the format.
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
        fmt = None  # undetermined — caller warns and skips

    color_map = {
        ("gemini", "tournament"): "#00e5a0",
        ("gemini", "single-elim"): "#22d3ee",
        ("gemini", "likert"): "#3b82f6",
        ("gemini", "hybrid"): "#a3e635",
        ("qwen", "tournament"): "#ff4d6d",
        ("qwen", "single-elim"): "#f59e0b",
        ("qwen", "likert"): "#a855f7",
        ("qwen", "hybrid"): "#fb7185",
    }
    color = color_map.get((provider, fmt), "#7b78ff")

    label = f"{provider.upper()} {fmt}" if fmt else None
    return label, provider, fmt, color


def decode_hdf5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_filename_keys(filename):
    """Robust filename matching: 'x', 'x.jpg', '/p/x.jpg' share lookup keys."""
    filename = str(filename).strip()
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    keys = {filename, base, stem, filename.lower(), base.lower(), stem.lower()}
    return {k for k in keys if k}


def build_hdf5_filename_lookup(h5):
    """Map every normalized filename key -> HDF5 row index."""
    lookup = {}
    filenames = h5["filenames"]
    for i in range(len(filenames)):
        for key in normalize_filename_keys(decode_hdf5_string(filenames[i])):
            lookup.setdefault(key, i)
    return lookup


def find_in_lookup(filename, lookup):
    for key in normalize_filename_keys(filename):
        if key in lookup:
            return lookup[key]
    return None


def hdf5_bytes_to_thumb_data_url(img_bytes, max_px=GALLERY_THUMB_PX, quality=GALLERY_JPEG_QUALITY):
    """Decode HDF5 image bytes, downscale to a thumbnail, return a JPEG data URL."""
    with Image.open(io.BytesIO(img_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def compute_cost(provider, input_tokens, output_tokens):
    rates = PRICING.get(provider)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + \
           (output_tokens / 1_000_000) * rates["output"]


def mean_std(vals):
    """Return (mean, sample-std). Sample std uses ddof=1; 0 when fewer than 2 values."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


def recall_curve_from(scores, ground_truth, total_positives):
    """
    Exact expected recall@N for every N = 1..len(scores), returned as a list
    indexed by N-1. Ties are handled analytically: within a band of equal
    scores, each true positive contributes its fractional share of the slots
    that N leaves open in that band.
    """
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
        b = j - i  # band size
        for k in range(1, b + 1):  # N = cum_images + k
            expected_tp = cum_tp + band_tp * (k / b)
            curve[cum_images + k - 1] = expected_tp / total_positives
        cum_images += b
        cum_tp += band_tp
        i = j

    return curve


# ── 2. LOAD AI DATA & CALCULATE METRICS (per replicate) ─────────────────────
print(f"Loading AI CSVs from {AI_CSV_DIR}...")
# Recurse so both flat layouts and per-format subfolders are picked up.
ai_files = sorted(
    os.path.join(root, f)
    for root, _dirs, files in os.walk(AI_CSV_DIR)
    for f in files
    if f.lower().endswith(".csv")
)

if not ai_files:
    print(f"Error: No AI files found in {AI_CSV_DIR}")
    sys.exit(1)

all_reps = []  # one entry per CSV (per replicate)

for path in ai_files:
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    method_name, provider, fmt, color = describe_method(stem, parent)

    if fmt is None:
        print(f"Skipping {os.path.basename(path)}: could not determine test format "
              f"from filename or parent folder '{parent}'.")
        continue

    scores = {}
    ground_truth = {}
    header_total_tokens = 0
    footer_input = None
    footer_output = None

    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)

            header_total_tokens = parse_header_tokens(header)

            lower_header = [h.lower().strip() for h in header]

            fname_idx = next((i for i, h in enumerate(lower_header) if h in ["filename", "name"]), None)
            score_idx = next((i for i, h in enumerate(lower_header) if h in ["imagescore", "image_score", "ai", "ai_selected"]), None)
            gt_idx = next((i for i, h in enumerate(lower_header) if h in ["interesting", "is_interesting", "label"]), None)

            if fname_idx is None or score_idx is None or gt_idx is None:
                print(f"Skipping {os.path.basename(path)}: Missing required columns.")
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
    except Exception as e:
        print(f"Error processing {path}: {e}")
        continue

    images_evaluated = len(scores)
    if images_evaluated == 0:
        print(f"Skipping {os.path.basename(path)}: No usable rows.")
        continue

    # Token split comes straight from the footer block on every CSV.
    if footer_input is not None and footer_output is not None:
        input_tokens = footer_input
        output_tokens = footer_output
    else:
        input_tokens = 0
        output_tokens = 0
        print(f"Warning: {os.path.basename(path)} has no TotalInputTokens/"
              f"TotalOutputTokens footer; tokens and cost recorded as 0.")

    total_tokens = input_tokens + output_tokens
    cost = compute_cost(provider, input_tokens, output_tokens)
    cost_per_1k = (cost / images_evaluated) * 1000 if images_evaluated else 0.0

    total_positives = sum(1 for v in ground_truth.values() if v)

    unique_scores = sorted(list(set(scores.values())), reverse=True)
    floor_score = min(unique_scores) - 1.0
    thresholds = unique_scores + [floor_score]
    max_valid_score = max(unique_scores) if unique_scores else 0.0

    pr_curve = []
    max_f1 = 0.0
    best_metrics = {}

    for t in thresholds:
        tp = sum(1 for f, s in scores.items() if s >= t and ground_truth[f])
        fp = sum(1 for f, s in scores.items() if s >= t and not ground_truth[f])
        fn = sum(1 for f, s in scores.items() if s < t and ground_truth[f])
        tn = sum(1 for f, s in scores.items() if s < t and not ground_truth[f])

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if t > max_valid_score else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        pt = {
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }

        if not pr_curve or pr_curve[-1]["recall"] != recall:
            pr_curve.append(pt)

        if f1 >= max_f1:
            max_f1 = f1
            best_metrics = pt

    # Step-function AUC
    pr_curve = sorted(pr_curve, key=lambda p: p["recall"])
    pr_auc = 0.0
    for i in range(1, len(pr_curve)):
        prev = pr_curve[i - 1]
        curr = pr_curve[i]
        pr_auc += (curr["recall"] - prev["recall"]) * curr["precision"]

    if not best_metrics:
        best_metrics = {"threshold": 0, "precision": 0, "recall": 0, "f1": 0}

    # Exact tie-aware recall@N for every N, plus recall@TOP_N for the cards.
    recall_curve = recall_curve_from(scores, ground_truth, total_positives)
    if recall_curve:
        idx = min(TOP_N, len(recall_curve)) - 1
        recall_at_top_n = recall_curve[idx]
    else:
        recall_at_top_n = 0.0

    # Build histogram: count interesting vs boring images at each unique score
    hist_interesting = Counter()
    hist_boring = Counter()
    for fname, s in scores.items():
        key = round(s, 6)
        if ground_truth.get(fname, False):
            hist_interesting[key] += 1
        else:
            hist_boring[key] += 1
    hist_bins = sorted(set(hist_interesting.keys()) | set(hist_boring.keys()))
    hist = {
        "labels": hist_bins,
        "interesting": [hist_interesting.get(b, 0) for b in hist_bins],
        "boring": [hist_boring.get(b, 0) for b in hist_bins],
    }

    # Top-N images for the bottom gallery (by this run's own ImageScore).
    top_ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:GALLERY_TOP_N]
    top_images = [
        {"filename": f, "score": s, "interesting": bool(ground_truth.get(f, False))}
        for f, s in top_ranked
    ]

    all_reps.append({
        "stem": stem,
        "method": method_name,
        "provider": provider,
        "format": fmt,
        "color": color,
        "top_images": top_images,
        "images_evaluated": images_evaluated,
        "total_positives": total_positives,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "cost_per_1k": cost_per_1k,
        "best_threshold": best_metrics["threshold"],
        "max_f1": max_f1,
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "recall_at_top_n": recall_at_top_n,
        "pr_auc": pr_auc,
        "pr_curve": pr_curve,
        "recall_curve": recall_curve,
        "hist": hist,
    })

    print(
        f"Processed {os.path.basename(path)} | Max F1: {max_f1:.3f} | "
        f"Recall@{TOP_N}: {recall_at_top_n:.3f} | Cost: ${cost:.4f}"
    )

if not all_reps:
    print("No valid data processed. Exiting.")
    sys.exit(1)


# ── 2b. AGGREGATE REPLICATES INTO METHOD GROUPS ─────────────────────────────
def interp_precision(pr_curve, r):
    """Interpolated precision at recall r: max precision over points with recall >= r."""
    best = 0.0
    for pt in pr_curve:
        if pt["recall"] >= r and pt["precision"] > best:
            best = pt["precision"]
    return best


PR_GRID = [i / 50 for i in range(51)]  # 0.00 .. 1.00

grouped = defaultdict(list)
for rep in all_reps:
    grouped[(rep["provider"], rep["format"])].append(rep)

groups = []
for (provider, fmt), reps in grouped.items():
    color = reps[0]["color"]
    label = reps[0]["method"]

    f1_m, f1_s = mean_std([r["max_f1"] for r in reps])
    prec_m, prec_s = mean_std([r["best_precision"] for r in reps])
    rec_m, rec_s = mean_std([r["best_recall"] for r in reps])
    auc_m, auc_s = mean_std([r["pr_auc"] for r in reps])
    cost_m, cost_s = mean_std([r["cost"] for r in reps])
    cpk_m, cpk_s = mean_std([r["cost_per_1k"] for r in reps])
    in_m, in_s = mean_std([r["input_tokens"] for r in reps])
    out_m, out_s = mean_std([r["output_tokens"] for r in reps])
    thr_m, thr_s = mean_std([r["best_threshold"] for r in reps])

    # Mean PR curve across replicates on a common recall grid.
    pr_curve_mean = [
        {"x": r, "y": sum(interp_precision(rep["pr_curve"], r) for rep in reps) / len(reps)}
        for r in PR_GRID
    ]

    # Per-score-bin histogram: mean ± std of the counts across replicates.
    # A bin missing from a replicate counts as 0 there, so the std reflects
    # subset-to-subset variability at that score.
    bins = sorted(
        {b for rep in reps for b in rep["hist"]["labels"]}
    )
    rep_int = [dict(zip(rep["hist"]["labels"], rep["hist"]["interesting"])) for rep in reps]
    rep_bor = [dict(zip(rep["hist"]["labels"], rep["hist"]["boring"])) for rep in reps]

    int_mean, int_std, bor_mean, bor_std = [], [], [], []
    total_mean, total_std = [], []
    for b in bins:
        iv = [m.get(b, 0) for m in rep_int]
        bv = [m.get(b, 0) for m in rep_bor]
        tv = [i + j for i, j in zip(iv, bv)]
        im, isd = mean_std(iv)
        bm, bsd = mean_std(bv)
        tm, tsd = mean_std(tv)
        int_mean.append(im); int_std.append(isd)
        bor_mean.append(bm); bor_std.append(bsd)
        total_mean.append(tm); total_std.append(tsd)

    hist = {
        "labels": bins,
        "interesting": int_mean, "interesting_std": int_std,
        "boring": bor_mean, "boring_std": bor_std,
        "total": total_mean, "total_std": total_std,
    }

    groups.append({
        "method": label,
        "provider": provider,
        "format": fmt,
        "color": color,
        "n_reps": len(reps),
        "max_f1_mean": f1_m, "max_f1_std": f1_s,
        "precision_mean": prec_m, "precision_std": prec_s,
        "recall_mean": rec_m, "recall_std": rec_s,
        "pr_auc_mean": auc_m, "pr_auc_std": auc_s,
        "cost_mean": cost_m, "cost_std": cost_s,
        "cpk_mean": cpk_m, "cpk_std": cpk_s,
        "input_mean": in_m, "input_std": in_s,
        "output_mean": out_m, "output_std": out_s,
        "threshold_mean": thr_m, "threshold_std": thr_s,
        "pr_curve_mean": pr_curve_mean,
        "hist": hist,
        "recall_reps": [
            {"curve": rep["recall_curve"], "pos": rep["total_positives"], "len": rep["images_evaluated"]}
            for rep in reps
        ],
    })

# Sort leaderboard by mean Max F1 descending
groups.sort(key=lambda g: g["max_f1_mean"], reverse=True)


# ── 2c. PROVIDER-LEVEL SUMMARY (Gemini vs Qwen), over all replicate runs ─────
provider_summary = defaultdict(list)
for rep in all_reps:
    provider_summary[rep["provider"]].append(rep)

provider_cards = []
for p in ("gemini", "qwen"):
    runs = provider_summary.get(p, [])
    if not runs:
        continue
    best = max(runs, key=lambda x: x["max_f1"])
    best_recall_run = max(runs, key=lambda x: x["recall_at_top_n"])
    avg_f1 = sum(x["max_f1"] for x in runs) / len(runs)
    avg_auc = sum(x["pr_auc"] for x in runs) / len(runs)
    avg_recall_at_n = sum(x["recall_at_top_n"] for x in runs) / len(runs)
    total_cost = sum(x["cost"] for x in runs)
    cheapest = min(runs, key=lambda x: x["cost_per_1k"])
    provider_cards.append({
        "provider": p,
        "color": PROVIDER_COLORS[p],
        "runs": len(runs),
        "best_f1": best["max_f1"],
        "best_config": best["format"],
        "best_recall_at_n": best_recall_run["recall_at_top_n"],
        "best_recall_config": best_recall_run["format"],
        "avg_f1": avg_f1,
        "avg_auc": avg_auc,
        "avg_recall_at_n": avg_recall_at_n,
        "total_cost": total_cost,
        "cheapest_cost_per_1k": cheapest["cost_per_1k"],
        "cheapest_config": cheapest["format"],
    })


# ── 2d. IMAGE GALLERY (top-N per run, embedded from HDF5) ───────────────────
# One selectable entry per individual run; images encoded once and shared by
# filename so repeats across runs don't bloat the file.
gallery_runs = []
for rep in all_reps:
    gallery_runs.append({
        "label": f"{rep['provider'].upper()} · {rep['format']} · {rep['stem']}",
        "color": rep["color"],
        "images": rep["top_images"],
    })

needed_filenames = {img["filename"] for rep in all_reps for img in rep["top_images"]}
image_urls = {}

if not needed_filenames:
    pass
elif not HDF5_LIBS:
    print("Gallery: h5py/Pillow not available — run with the gemini_env conda "
          "env to embed thumbnails. Skipping image embedding.")
elif not os.path.exists(HDF5_PATH):
    print(f"Gallery: HDF5 not found at {HDF5_PATH} — skipping image embedding.")
else:
    print(f"Gallery: embedding up to {len(needed_filenames)} thumbnails from HDF5...")
    try:
        with h5py.File(HDF5_PATH, "r") as h5:
            if "filenames" not in h5 or "images" not in h5:
                raise KeyError("HDF5 missing 'filenames'/'images' datasets.")
            lookup = build_hdf5_filename_lookup(h5)
            images_ds = h5["images"]
            missing = 0
            for fname in needed_filenames:
                idx = find_in_lookup(fname, lookup)
                if idx is None:
                    missing += 1
                    continue
                try:
                    image_urls[fname] = hdf5_bytes_to_thumb_data_url(bytes(images_ds[idx]))
                except Exception as e:
                    missing += 1
                    print(f"  Warning: failed to encode {fname}: {e}")
        print(f"Gallery: embedded {len(image_urls)} thumbnails ({missing} missing).")
    except Exception as e:
        print(f"Gallery: failed to read HDF5 ({e}) — skipping image embedding.")


# ── 3. GENERATE HTML DASHBOARD ──────────────────────────────────────────────
print("\nGenerating HTML Dashboard...")


def pm(mean, std, prec=3):
    """Format a 'mean ± std' string."""
    return f"{mean:.{prec}f} <span class=\"pm-err\">± {std:.{prec}f}</span>"


cards_html = ""
for c in provider_cards:
    cards_html += f"""
    <div class="prov-card" style="--prov:{c['color']}">
        <div class="prov-head">
            <span class="color-dot" style="background:{c['color']}"></span>
            <h3>{c['provider'].upper()}</h3>
            <span class="prov-runs">{c['runs']} runs</span>
        </div>
        <div class="prov-metrics">
            <div><span class="pm-val">{c['best_f1']:.3f}</span><span class="pm-lbl">Best F1 ({c['best_config']})</span></div>
            <div><span class="pm-val">{c['best_recall_at_n']:.3f}</span><span class="pm-lbl">Best Recall@{TOP_N} ({c['best_recall_config']})</span></div>
            <div><span class="pm-val">{c['avg_f1']:.3f}</span><span class="pm-lbl">Avg F1</span></div>
            <div><span class="pm-val">{c['avg_recall_at_n']:.3f}</span><span class="pm-lbl">Avg Recall@{TOP_N}</span></div>
            <div><span class="pm-val">{c['avg_auc']:.3f}</span><span class="pm-lbl">Avg PR AUC</span></div>
            <div><span class="pm-val">${c['total_cost']:.2f}</span><span class="pm-lbl">Total Cost (all runs)</span></div>
            <div><span class="pm-val">${c['cheapest_cost_per_1k']:.3f}</span><span class="pm-lbl">Cheapest $/1k img ({c['cheapest_config']})</span></div>
        </div>
    </div>
    """

leaderboard_rows = ""
for g in groups:
    cutoff_str = f"&ge; {g['threshold_mean']:g}"
    leaderboard_rows += f"""
    <tr>
        <td>
            <span class="color-dot" style="background:{g['color']}"></span>
            <strong>{g['provider'].upper()}</strong>
        </td>
        <td>{g['format']} <span class="frac">×{g['n_reps']}</span></td>
        <td>{cutoff_str}</td>
        <td class="highlight">{pm(g['max_f1_mean'], g['max_f1_std'])}</td>
        <td>{pm(g['precision_mean'], g['precision_std'])}</td>
        <td>{pm(g['recall_mean'], g['recall_std'])}</td>
        <td>{pm(g['pr_auc_mean'], g['pr_auc_std'])}</td>
        <td>{g['input_mean']:,.0f} <span class="frac">± {g['input_std']:,.0f}</span></td>
        <td>{g['output_mean']:,.0f} <span class="frac">± {g['output_std']:,.0f}</span></td>
        <td class="cost">${g['cost_mean']:.4f} <span class="frac">± {g['cost_std']:.4f}</span></td>
        <td>${g['cpk_mean']:.3f} <span class="frac">± {g['cpk_std']:.3f}</span></td>
    </tr>
    """

chart_data = json.dumps([{
    "method": g["method"],
    "provider": g["provider"],
    "format": g["format"],
    "color": g["color"],
    "n_reps": g["n_reps"],
    "cost_mean": g["cost_mean"],
    "cost_std": g["cost_std"],
    "max_f1_mean": g["max_f1_mean"],
    "max_f1_std": g["max_f1_std"],
    "pr_curve": g["pr_curve_mean"],
    "hist": g["hist"],
    "recall_reps": g["recall_reps"],
} for g in groups])

provider_colors_json = json.dumps(PROVIDER_COLORS)
gallery_runs_json = json.dumps(gallery_runs)
image_urls_json = json.dumps(image_urls)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemini vs Qwen — Hubble Classification Cost/Performance Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    *,*::before,*::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
        --bg: #040812;
        --surface: #0b1120;
        --border: rgba(255,255,255,0.08);
        --text: #b8c8e0;
        --text-dim: #4a5878;
        --accent: #00e5a0;
        --mono: 'IBM Plex Mono', monospace;
        --display: 'Syne', sans-serif;
    }}
    body {{
        background: var(--bg); color: var(--text); font-family: var(--mono);
        font-size: 13px; line-height: 1.6; padding: 40px 20px 100px;
    }}
    .container {{ max-width: 1300px; margin: 0 auto; }}
    header {{ border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }}
    h1 {{ font-family: var(--display); font-size: 42px; color: #fff; line-height: 1.1; margin-bottom: 8px; }}
    .subtitle {{ color: var(--text-dim); font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em; }}

    .prov-cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }}
    .prov-card {{
        background: var(--surface); border: 1px solid var(--border);
        border-top: 3px solid var(--prov); border-radius: 12px; padding: 20px 24px;
    }}
    .prov-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }}
    .prov-head h3 {{ font-family: var(--display); font-size: 22px; color: #fff; }}
    .prov-runs {{ margin-left: auto; color: var(--text-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .prov-metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px 20px; }}
    .prov-metrics > div {{ display: flex; flex-direction: column; }}
    .pm-val {{ color: #fff; font-size: 22px; font-weight: 700; }}
    .pm-lbl {{ color: var(--text-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }}

    .table-container {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; overflow-x: auto; margin-bottom: 40px;
    }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; }}
    th, td {{ padding: 14px 20px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    th {{ background: rgba(255,255,255,0.02); color: var(--text-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    td {{ color: #fff; font-size: 13px; }}
    .color-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
    .highlight {{ color: var(--accent); font-weight: 700; font-size: 14px; }}
    .cost {{ color: #f59e0b; font-weight: 600; }}
    .frac {{ color: var(--text-dim); font-weight: 400; font-size: 11px; }}
    .pm-err {{ color: var(--text-dim); font-weight: 400; font-size: 11px; }}

    .dashboard-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
    .chart-box {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 20px; height: 450px;
    }}
    .chart-box.full-width {{ grid-column: 1 / -1; height: 650px; }}
    .chart-box.with-controls {{ display: flex; flex-direction: column; }}
    .chart-box.with-controls canvas {{ flex: 1 1 auto; min-height: 0; }}

    .hist-controls {{
        display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
        margin-bottom: 12px;
    }}
    .hist-controls label {{ color: var(--text-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .hist-controls select, .hist-controls input {{
        background: #0f1a2e; border: 1px solid var(--border); color: #fff;
        font-family: var(--mono); font-size: 12px; padding: 6px 10px;
        border-radius: 6px; outline: none;
    }}
    .hist-controls select:focus, .hist-controls input:focus {{
        border-color: var(--accent);
    }}
    .hist-controls input {{ width: 90px; }}
    .hist-controls .hint {{ color: var(--text-dim); font-size: 10px; text-transform: none; letter-spacing: 0; }}

    .gallery-box {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 20px; margin-bottom: 24px;
    }}
    .gallery-grid {{
        display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 14px; margin-top: 12px;
    }}
    .gthumb {{
        border: 3px solid var(--border); border-radius: 8px; overflow: hidden;
        background: #0f1a2e; display: flex; flex-direction: column;
    }}
    .gthumb.interesting {{ border-color: #22c55e; }}
    .gthumb.boring {{ border-color: #ef4444; }}
    .gthumb .imgwrap {{
        aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center;
        background: #06101f; color: var(--text-dim); font-size: 10px; text-align: center;
    }}
    .gthumb .imgwrap img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .gthumb .gcap {{
        display: flex; justify-content: space-between; gap: 6px; padding: 6px 8px;
        font-size: 11px; color: #fff;
    }}
    .gthumb .gcap .rank {{ color: var(--text-dim); }}
    .gthumb .gcap .gt {{ font-weight: 700; }}

    @media(max-width: 900px) {{
        .dashboard-grid {{ grid-template-columns: 1fr; }}
        .prov-cards {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>

<div class="container">
    <header>
        <h1>Gemini <em>vs</em> Qwen</h1>
        <p class="subtitle">Hubble classification performance vs. actual $ cost &middot; embedded ground truth &middot; mean ± std over random subsets</p>
    </header>

    <div class="prov-cards">
        {cards_html}
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Provider</th>
                    <th>Test Format</th>
                    <th>Opt AI Cutoff</th>
                    <th>Max F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>PR AUC</th>
                    <th>Input Tok</th>
                    <th>Output Tok</th>
                    <th>Total Cost</th>
                    <th>Cost / 1k img</th>
                </tr>
            </thead>
            <tbody>
                {leaderboard_rows}
            </tbody>
        </table>
    </div>

    <div class="dashboard-grid">

        <div class="chart-box full-width">
            <canvas id="prChart"></canvas>
        </div>

        <div class="chart-box">
            <canvas id="efficiencyChart"></canvas>
        </div>

        <div class="chart-box">
            <canvas id="barChart"></canvas>
        </div>

        <div class="chart-box full-width with-controls" style="height:560px">
            <div class="hist-controls">
                <label for="recallN">Recall @ top</label>
                <input id="recallN" type="number" min="1" step="1" value="{TOP_N}">
                <span class="hint">images, plotted vs. $ cost &middot; mean ± std over subsets, ties resolved by exact expectation</span>
            </div>
            <canvas id="recallChart"></canvas>
        </div>

        <div class="chart-box full-width with-controls" style="height:560px">
            <div class="hist-controls">
                <label for="histSelect">Method</label>
                <select id="histSelect"></select>
                <label for="histYMax">Y-axis max</label>
                <input id="histYMax" type="number" min="1" step="1" placeholder="auto">
                <span class="hint">mean count per subset · error bar = ± std of the stack total</span>
            </div>
            <canvas id="histChart"></canvas>
        </div>
    </div>

    <div class="gallery-box">
        <div class="hist-controls">
            <label for="gallerySelect">Run</label>
            <select id="gallerySelect"></select>
            <span class="hint">top {GALLERY_TOP_N} images by ImageScore &middot; <span style="color:#22c55e">green = interesting</span>, <span style="color:#ef4444">red = boring</span> (ground truth)</span>
        </div>
        <div id="galleryGrid" class="gallery-grid"></div>
    </div>
</div>

<script>
    Chart.defaults.color = '#b8c8e0';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
    Chart.defaults.font.family = 'IBM Plex Mono, monospace';

    const rawData = {chart_data};
    const providerColors = {provider_colors_json};

    const usd = v => '$' + v.toLocaleString(undefined, {{ minimumFractionDigits: 4, maximumFractionDigits: 4 }});
    const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
    const std = a => {{
        if (a.length < 2) return 0;
        const m = mean(a);
        return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
    }};

    // Error-bar plugin: reads chart.options.plugins.errorBars.
    //   mode 'vbar' -> vertical bars from yVals ± yErr (uses bar element x centers)
    //   mode 'xy'   -> scatter points at (xVals,yVals) with xErr / yErr
    const errorBarPlugin = {{
        id: 'errorBars',
        afterDatasetsDraw(chart) {{
            const opt = chart.options.plugins.errorBars;
            if (!opt) return;
            const ctx = chart.ctx;
            const meta = chart.getDatasetMeta(0);
            const yScale = chart.scales.y, xScale = chart.scales.x;
            ctx.save();
            ctx.strokeStyle = opt.color || 'rgba(255,255,255,0.75)';
            ctx.lineWidth = 1.5;
            const cap = 5;
            if (opt.mode === 'vbar') {{
                meta.data.forEach((el, i) => {{
                    const v = opt.yVals[i], e = opt.yErr[i];
                    if (!e) return;
                    const cx = el.x;
                    const top = yScale.getPixelForValue(v + e);
                    const bot = yScale.getPixelForValue(Math.max(0, v - e));
                    ctx.beginPath();
                    ctx.moveTo(cx, top); ctx.lineTo(cx, bot);
                    ctx.moveTo(cx - cap, top); ctx.lineTo(cx + cap, top);
                    ctx.moveTo(cx - cap, bot); ctx.lineTo(cx + cap, bot);
                    ctx.stroke();
                }});
            }} else if (opt.mode === 'xy') {{
                opt.xVals.forEach((xv, i) => {{
                    const yv = opt.yVals[i];
                    const xe = opt.xErr ? opt.xErr[i] : 0;
                    const ye = opt.yErr ? opt.yErr[i] : 0;
                    const cx = xScale.getPixelForValue(xv);
                    const cy = yScale.getPixelForValue(yv);
                    if (ye) {{
                        const t = yScale.getPixelForValue(yv + ye);
                        const b = yScale.getPixelForValue(yv - ye);
                        ctx.beginPath();
                        ctx.moveTo(cx, t); ctx.lineTo(cx, b);
                        ctx.moveTo(cx - cap, t); ctx.lineTo(cx + cap, t);
                        ctx.moveTo(cx - cap, b); ctx.lineTo(cx + cap, b);
                        ctx.stroke();
                    }}
                    if (xe) {{
                        const l = xScale.getPixelForValue(xv - xe);
                        const r = xScale.getPixelForValue(xv + xe);
                        ctx.beginPath();
                        ctx.moveTo(l, cy); ctx.lineTo(r, cy);
                        ctx.moveTo(l, cy - cap); ctx.lineTo(l, cy + cap);
                        ctx.moveTo(r, cy - cap); ctx.lineTo(r, cy + cap);
                        ctx.stroke();
                    }}
                }});
            }}
            ctx.restore();
        }}
    }};
    Chart.register(errorBarPlugin);

    // 1. PR Curve: mean curve per method
    new Chart(document.getElementById('prChart'), {{
        type: 'line',
        data: {{
            datasets: rawData.map(d => ({{
                label: d.provider.toUpperCase() + ' · ' + d.format,
                data: d.pr_curve,
                borderColor: d.color,
                backgroundColor: d.color,
                borderWidth: 2,
                fill: false,
                tension: 0,
                pointRadius: 0,
                pointHoverRadius: 5
            }}))
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{
                title: {{ display: true, text: 'PR Curve: Gemini vs Qwen (mean over subsets)', color: '#fff', font: {{ size: 16 }} }},
                tooltip: {{
                    callbacks: {{
                        label: function(ctx) {{
                            const pt = ctx.raw;
                            return `${{ctx.dataset.label}} — P: ${{pt.y.toFixed(3)}} | R: ${{pt.x.toFixed(3)}}`;
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{ type: 'linear', min: 0, max: 1.05, title: {{ display: true, text: 'Recall' }} }},
                y: {{ type: 'linear', min: 0, max: 1.05, title: {{ display: true, text: 'Precision' }} }}
            }}
        }}
    }});

    // 2. Efficiency Scatter (mean $ Cost vs mean Max F1) with x/y error bars
    const effXVals = rawData.map(d => d.cost_mean);
    const effYVals = rawData.map(d => d.max_f1_mean);
    new Chart(document.getElementById('efficiencyChart'), {{
        type: 'scatter',
        data: {{
            datasets: [{{
                label: 'Method (mean ± std)',
                data: rawData.map(d => ({{ x: d.cost_mean, y: d.max_f1_mean, method: d.provider.toUpperCase() + ' · ' + d.format }})),
                backgroundColor: rawData.map(d => d.color),
                borderColor: rawData.map(d => d.color),
                pointRadius: 7,
                pointHoverRadius: 10
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }},
                title: {{ display: true, text: 'Actual $ Cost vs. Max F1 (mean ± std)', color: '#fff', font: {{ size: 16 }} }},
                tooltip: {{
                    callbacks: {{
                        label: function(ctx) {{
                            const pt = ctx.raw;
                            return `${{pt.method}}: ${{usd(pt.x)}} | F1: ${{pt.y.toFixed(3)}}`;
                        }}
                    }}
                }},
                errorBars: {{
                    mode: 'xy',
                    xVals: effXVals, yVals: effYVals,
                    xErr: rawData.map(d => d.cost_std),
                    yErr: rawData.map(d => d.max_f1_std)
                }}
            }},
            scales: {{
                x: {{ type: 'linear', title: {{ display: true, text: 'Total Cost (USD)' }}, ticks: {{ callback: v => '$' + v.toFixed(2) }} }},
                y: {{ type: 'linear', title: {{ display: true, text: 'Max F1 Score' }} }}
            }}
        }}
    }});

    // 3. Max F1 bar chart with error bars
    new Chart(document.getElementById('barChart'), {{
        type: 'bar',
        data: {{
            labels: rawData.map(d => d.provider.toUpperCase() + ' · ' + d.format),
            datasets: [{{
                label: 'Max F1',
                data: rawData.map(d => d.max_f1_mean),
                backgroundColor: rawData.map(d => d.color)
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }},
                title: {{ display: true, text: 'Max F1 by Method (mean ± std)', color: '#fff', font: {{ size: 16 }} }},
                errorBars: {{
                    mode: 'vbar',
                    yVals: rawData.map(d => d.max_f1_mean),
                    yErr: rawData.map(d => d.max_f1_std)
                }}
            }},
            scales: {{
                x: {{ ticks: {{ maxRotation: 60, minRotation: 60, font: {{ size: 9 }} }} }},
                y: {{ min: 0, max: 1 }}
            }}
        }}
    }});

    // 4. Interactive Recall@N bar chart with error bars
    const recallNInput = document.getElementById('recallN');
    let recallChart = null;

    function recallStatsAtN(group, N) {{
        const vals = [], counts = [], poss = [];
        group.recall_reps.forEach(rep => {{
            if (!rep.len) return;
            const idx = Math.min(Math.max(1, N), rep.len) - 1;
            const v = rep.curve[idx] || 0;
            vals.push(v);
            counts.push(v * rep.pos);
            poss.push(rep.pos);
        }});
        return {{ mean: mean(vals), std: std(vals), meanCount: mean(counts), meanPos: mean(poss) }};
    }}

    function buildRecall() {{
        let N = parseInt(recallNInput.value);
        if (!Number.isFinite(N) || N < 1) N = 1;

        const stats = rawData.map(d => recallStatsAtN(d, N));
        const points = rawData.map((d, i) => ({{
            x: d.cost_mean, y: stats[i].mean,
            method: d.provider.toUpperCase() + ' · ' + d.format
        }}));
        const xVals = rawData.map(d => d.cost_mean);
        const yVals = stats.map(s => s.mean);
        const xErr = rawData.map(d => d.cost_std);
        const yErr = stats.map(s => s.std);
        const title = `Recall @ top ${{N}} vs. Cost (mean ± std over subsets)`;

        if (recallChart) {{
            recallChart.data.datasets[0].data = points;
            recallChart.options.plugins.title.text = title;
            Object.assign(recallChart.options.plugins.errorBars, {{ xVals, yVals, xErr, yErr }});
            recallChart.$stats = stats;
            recallChart.update();
            return;
        }}

        recallChart = new Chart(document.getElementById('recallChart'), {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Method (mean ± std)',
                    data: points,
                    backgroundColor: rawData.map(d => d.color),
                    borderColor: rawData.map(d => d.color),
                    pointRadius: 7,
                    pointHoverRadius: 10
                }}]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: true, text: title, color: '#fff', font: {{ size: 16 }} }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                const pt = ctx.raw;
                                const s = ctx.chart.$stats[ctx.dataIndex];
                                return [
                                    pt.method,
                                    `Cost: ${{usd(pt.x)}} | Recall: ${{s.mean.toFixed(3)}} ± ${{s.std.toFixed(3)}}`,
                                    `≈ ${{s.meanCount.toFixed(1)}} / ${{s.meanPos.toFixed(1)}} interesting imgs`
                                ];
                            }}
                        }}
                    }},
                    errorBars: {{ mode: 'xy', xVals, yVals, xErr, yErr }}
                }},
                scales: {{
                    x: {{ type: 'linear', title: {{ display: true, text: 'Total Cost (USD)' }}, ticks: {{ callback: v => '$' + v.toFixed(2) }} }},
                    y: {{ min: 0, max: 1.05, title: {{ display: true, text: 'Recall rate' }} }}
                }}
            }}
        }});
        recallChart.$stats = stats;
    }}

    recallNInput.addEventListener('input', buildRecall);

    // 5. Score Histogram (summed across subsets)
    const histSelect = document.getElementById('histSelect');
    const histYMaxInput = document.getElementById('histYMax');

    rawData.forEach((d, i) => {{
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = d.provider.toUpperCase() + ' · ' + d.format;
        histSelect.appendChild(opt);
    }});

    let histChart = null;

    function buildHistogram() {{
        const d = rawData[parseInt(histSelect.value)];
        const yMaxRaw = histYMaxInput.value.trim();
        const yMax = yMaxRaw === '' ? undefined : parseInt(yMaxRaw);

        const labels = d.hist.labels.map(v => v);
        const title = `Score Distribution — ${{d.provider.toUpperCase()}} · ${{d.format}} (mean per subset ± std)`;

        if (histChart) {{
            histChart.data.labels = labels;
            histChart.data.datasets[0].data = d.hist.boring;
            histChart.data.datasets[1].data = d.hist.interesting;
            histChart.options.scales.y.max = yMax;
            histChart.options.plugins.title.text = title;
            histChart.options.plugins.errorBars.yVals = d.hist.total;
            histChart.options.plugins.errorBars.yErr = d.hist.total_std;
            histChart.update();
            return;
        }}

        histChart = new Chart(document.getElementById('histChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Boring (interesting = 0)',
                        data: d.hist.boring,
                        backgroundColor: '#3b82f6',
                        stack: 'stack'
                    }},
                    {{
                        label: 'Interesting (interesting = 1)',
                        data: d.hist.interesting,
                        backgroundColor: '#ef4444',
                        stack: 'stack'
                    }}
                ]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{
                    title: {{ display: true, text: title, color: '#fff', font: {{ size: 16 }} }},
                    tooltip: {{
                        callbacks: {{
                            title: ctx => `Score: ${{ctx[0].label}}`,
                            label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}`
                        }}
                    }},
                    errorBars: {{ mode: 'vbar', yVals: d.hist.total, yErr: d.hist.total_std }}
                }},
                scales: {{
                    x: {{ stacked: true, title: {{ display: true, text: 'AI ImageScore' }} }},
                    y: {{ stacked: true, min: 0, max: yMax, title: {{ display: true, text: 'Mean images per subset' }} }}
                }}
            }}
        }});
    }}

    histSelect.addEventListener('change', buildHistogram);
    histYMaxInput.addEventListener('input', buildHistogram);

    buildRecall();
    if (rawData.length > 0) buildHistogram();

    // 6. Bottom image gallery — top images per run (visual sanity check)
    const galleryRuns = {gallery_runs_json};
    const imageUrls = {image_urls_json};
    const gallerySelect = document.getElementById('gallerySelect');
    const galleryGrid = document.getElementById('galleryGrid');

    galleryRuns.forEach((run, i) => {{
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = run.label;
        gallerySelect.appendChild(opt);
    }});

    function renderGallery() {{
        const run = galleryRuns[parseInt(gallerySelect.value)];
        galleryGrid.innerHTML = '';
        if (!run) return;
        run.images.forEach((im, idx) => {{
            const url = imageUrls[im.filename];
            const cls = im.interesting ? 'interesting' : 'boring';
            const cell = document.createElement('div');
            cell.className = 'gthumb ' + cls;
            const imgHtml = url
                ? `<img src="${{url}}" alt="${{im.filename}}" loading="lazy">`
                : `<span>no image<br>${{im.filename}}</span>`;
            cell.innerHTML = `
                <div class="imgwrap">${{imgHtml}}</div>
                <div class="gcap">
                    <span class="rank">#${{idx + 1}}</span>
                    <span>${{(+im.score).toFixed(2)}}</span>
                    <span class="gt" style="color:${{im.interesting ? '#22c55e' : '#ef4444'}}">${{im.interesting ? 'INT' : 'BOR'}}</span>
                </div>`;
            galleryGrid.appendChild(cell);
        }});
    }}

    gallerySelect.addEventListener('change', renderGallery);
    if (galleryRuns.length > 0) renderGallery();
</script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\nSuccess! Dashboard written to {OUTPUT_HTML}")
