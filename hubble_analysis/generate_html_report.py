"""
generate_html.py

Generate a self-contained HTML report for mixed Gemini image scoring results.

Expected CSV columns:
    index, filename, imagescore, interesting, classification, SourceRA, SourceDec

Important:
- All images are loaded from the HDF5 file by filename lookup.
- Images are converted to JPEG and embedded directly in the HTML as base64
  data URLs (the top MAX_JPEG_IMAGES by score, default 2000, unioned with every
  row marked "interesting", so no true-positive/false-negative image is left
  out even if its score placed it outside the top N) so the report is fully
  self-contained and viewable anywhere.
- True non-interesting images with ImageScore = 0 are NOT shown in the image grid,
  but they ARE still included in all data plot calculations.

Usage:
    python generate_html.py
    python generate_html.py results.csv hdf5_file.hdf5
    python generate_html.py results.csv hdf5_file.hdf5 output.html
    python generate_html.py results.csv hdf5_file.hdf5 output.html none
    python generate_html.py results.csv hdf5_file.hdf5 output.html 1000
    python generate_html.py results.csv hdf5_file.hdf5 output.html none 2000
"""

import base64
import csv
import html
import io
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
from PIL import Image


# ── 1. CONFIGURATION ───────────────────────────────────────────────────────

CLASSIFICATION_CSV = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "gemini_likert_1.csv"
)

HDF5_PATH = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "../hubble_data/10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5"
)

OUTPUT_HTML = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "hsc_report_mixed.html"
)

MAX_CARDS = (
    int(sys.argv[4])
    if len(sys.argv) > 4 and sys.argv[4].lower() != "none"
    else None
)

MAX_JPEG_IMAGES = (
    int(sys.argv[5])
    if len(sys.argv) > 5
    else 2000
)
SORT_BY_SCORE = True


# ── 2. HELPERS ─────────────────────────────────────────────────────────────

def get_field(row, *names, default=""):
    lowered = {str(k).lower(): v for k, v in row.items()}

    for name in names:
        val = lowered.get(name.lower())
        if val is not None:
            return val

    return default


def safe_int(value, default=0):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(str(value).strip())
    except Exception:
        return default


def decode_hdf5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def normalize_filename_keys(filename):
    """
    Make filename matching robust.

    Example:
        4001219269105
        4001219269105.jpg
        /path/to/4001219269105.jpg

    will all share useful lookup keys.
    """
    filename = str(filename).strip()
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]

    keys = {
        filename,
        base,
        stem,
        filename.lower(),
        base.lower(),
        stem.lower(),
    }

    return {k for k in keys if k}


def infer_mime_from_path(path):
    suffix = Path(path).suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix in [".tif", ".tiff"]:
        return "image/tiff"

    return "image/png"


def image_file_to_data_url(path):
    mime = infer_mime_from_path(path)

    with open(path, "rb") as f:
        img_bytes = f.read()

    b64 = base64.b64encode(img_bytes).decode("ascii")

    return f"data:{mime};base64,{b64}"


def image_bytes_to_data_url(img_bytes):
    """
    Convert encoded image bytes into a browser-displayable data URL.
    """
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            fmt = (img.format or "PNG").lower()

        if fmt == "jpg":
            fmt = "jpeg"

        mime = f"image/{fmt}"

    except Exception:
        mime = "image/png"

    b64 = base64.b64encode(img_bytes).decode("ascii")

    return f"data:{mime};base64,{b64}"


def hdf5_bytes_to_jpeg_data_url(img_bytes, quality=90):
    """
    Convert HDF5 image bytes to a JPEG-encoded base64 data URL.
    Returns the data URL on success, empty string on failure.
    """
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality)
            jpeg_bytes = buf.getvalue()

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"Warning: failed to convert image to JPEG: {e}")
        return ""


def score_color(score, max_score):
    if max_score <= 0:
        return "#6a7897"

    ratio = score / max_score

    if ratio >= 0.70:
        return "#00e5a0"
    if ratio >= 0.40:
        return "#f0b429"

    return "#ff4d6d"


# ── 3. IMAGE LOOKUPS ───────────────────────────────────────────────────────

def build_hdf5_filename_lookup(hdf5_path):
    """
    Build filename -> HDF5 row index lookup.

    The result CSV index is the mixed dataset index, so it should not be used
    to index directly into HDF5.
    """
    lookup = {}

    with h5py.File(hdf5_path, "r") as h5:
        if "filenames" not in h5:
            raise KeyError("HDF5 file does not contain dataset 'filenames'.")

        filenames = h5["filenames"]

        for i in range(len(filenames)):
            fname = decode_hdf5_string(filenames[i])

            for key in normalize_filename_keys(fname):
                if key not in lookup:
                    lookup[key] = i

    return lookup


def build_interesting_image_lookup(image_dir):
    """
    Build filename -> image path lookup for JPEG/PNG interesting images.
    """
    allowed_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff",
    }

    lookup = {}

    if not os.path.isdir(image_dir):
        print(f"Warning: interesting image directory does not exist: {image_dir}")
        return lookup

    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)

        if not os.path.isfile(path):
            continue

        if Path(fname).suffix.lower() not in allowed_exts:
            continue

        for key in normalize_filename_keys(fname):
            if key not in lookup:
                lookup[key] = path

    return lookup


def find_in_lookup(filename, lookup):
    for key in normalize_filename_keys(filename):
        if key in lookup:
            return lookup[key]

    return None


# ── 4. READ CSV ────────────────────────────────────────────────────────────

if not os.path.exists(CLASSIFICATION_CSV):
    raise FileNotFoundError(f"Could not find classification CSV: {CLASSIFICATION_CSV}")

rows = []

with open(CLASSIFICATION_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        index_raw = get_field(row, "index", "Index")

        if str(index_raw).strip() == "":
            continue

        rows.append({
            "index": safe_int(index_raw),
            "filename": str(get_field(row, "filename", "Filename")).strip(),
            "imagescore": safe_float(
                get_field(row, "imagescore", "ImageScore", "ai_score")
            ),
            "interesting": safe_int(
                get_field(row, "interesting", "Interesting"),
                default=0,
            ),
            "classification": str(
                get_field(row, "classification", "Classification")
            ).strip(),
            "SourceRA": safe_float(
                get_field(row, "SourceRA", "source_ra"),
                default=0.0,
            ),
            "SourceDec": safe_float(
                get_field(row, "SourceDec", "source_dec"),
                default=0.0,
            ),
        })

if not rows:
    raise RuntimeError(f"No valid rows found in {CLASSIFICATION_CSV}")

if SORT_BY_SCORE:
    rows.sort(key=lambda r: r["imagescore"], reverse=True)


# ── 5. LOAD IMAGES ─────────────────────────────────────────────────────────

if not os.path.exists(HDF5_PATH):
    raise FileNotFoundError(f"Could not find HDF5 file: {HDF5_PATH}")

print("Building HDF5 filename lookup...")
hdf5_filename_lookup = build_hdf5_filename_lookup(HDF5_PATH)

missing_images = 0
saved_jpeg_images = 0

with h5py.File(HDF5_PATH, "r") as h5:
    if "images" not in h5:
        raise KeyError("HDF5 file does not contain dataset 'images'.")

    images = h5["images"]

    for rank, r in enumerate(rows):
        filename = str(r["filename"]).strip()
        hdf5_idx = find_in_lookup(filename, hdf5_filename_lookup)

        if hdf5_idx is None:
            r["data_url"] = ""
            missing_images += 1
            print(f"Warning: HDF5 image filename not found: {filename}")
            continue

        should_embed = rank < MAX_JPEG_IMAGES or int(r["interesting"]) == 1

        try:
            img_bytes = bytes(images[hdf5_idx])

            if should_embed:
                data_url = hdf5_bytes_to_jpeg_data_url(img_bytes)

                if data_url:
                    r["data_url"] = data_url
                    saved_jpeg_images += 1
                else:
                    r["data_url"] = ""
                    missing_images += 1
            else:
                r["data_url"] = ""

        except Exception as e:
            r["data_url"] = ""
            missing_images += 1
            print(f"Warning: failed to load HDF5 image {filename}: {e}")


# ── 6. SUMMARY STATS AND METRICS ───────────────────────────────────────────

total = len(rows)
scores = [float(r["imagescore"]) for r in rows]

avg_score = sum(scores) / total if total else 0.0
max_score = max(scores) if scores else 0.0

interesting_counts = Counter(int(r["interesting"]) for r in rows)
num_interesting = interesting_counts.get(1, 0)
num_not_interesting = interesting_counts.get(0, 0)


def compute_metrics_at_threshold(threshold):
    tp = fp = tn = fn = 0

    for r in rows:
        y_true = 1 if int(r["interesting"]) == 1 else 0
        y_pred = 1 if float(r["imagescore"]) >= threshold else 0

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "selected": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


unique_thresholds = sorted({float(r["imagescore"]) for r in rows}, reverse=True)

if unique_thresholds:
    threshold_metrics = [
        compute_metrics_at_threshold(unique_thresholds[0] + 1e-9)
    ]
    threshold_metrics += [
        compute_metrics_at_threshold(t)
        for t in unique_thresholds
    ]
else:
    threshold_metrics = [compute_metrics_at_threshold(0.0)]

real_threshold_metrics = (
    threshold_metrics[1:]
    if len(threshold_metrics) > 1
    else threshold_metrics
)

best_metric = max(
    real_threshold_metrics,
    key=lambda m: (m["f1"], m["recall"], m["precision"]),
)

best_threshold = float(best_metric["threshold"])
best_precision = float(best_metric["precision"])
best_recall = float(best_metric["recall"])
best_f1 = float(best_metric["f1"])


pr_auc = 0.0
pr_points_for_auc = sorted(
    [(m["recall"], m["precision"]) for m in threshold_metrics],
    key=lambda x: x[0],
)

for (r0, p0), (r1, p1) in zip(pr_points_for_auc[:-1], pr_points_for_auc[1:]):
    pr_auc += (r1 - r0) * (p0 + p1) / 2


# ── 7. HISTOGRAM SVG ───────────────────────────────────────────────────────

hist_by_score = {}

for r in rows:
    score_bin = int(float(r["imagescore"]))
    interesting = int(r["interesting"])

    if score_bin not in hist_by_score:
        hist_by_score[score_bin] = {
            "interesting": 0,
            "not_interesting": 0,
        }

    if interesting == 1:
        hist_by_score[score_bin]["interesting"] += 1
    else:
        hist_by_score[score_bin]["not_interesting"] += 1

hist_items = sorted(hist_by_score.items())

max_hist_count = max(
    [
        counts["interesting"] + counts["not_interesting"]
        for _, counts in hist_items
    ],
    default=1,
)

HIST_W = 980
HIST_H = 390
HIST_L = 58
HIST_R = 22
HIST_T = 24
HIST_B = 54
HIST_PLOT_W = HIST_W - HIST_L - HIST_R
HIST_PLOT_H = HIST_H - HIST_T - HIST_B
HIST_BASELINE = HIST_T + HIST_PLOT_H

hist_svg_parts = [
    (
        f'<svg id="histogramSvg" class="plot-svg" '
        f'viewBox="0 0 {HIST_W} {HIST_H}" '
        f'role="img" '
        f'aria-label="Vertical ImageScore histogram split by ground truth" '
        f'data-default-ymax="{max_hist_count}" '
        f'data-plot-top="{HIST_T}" '
        f'data-plot-height="{HIST_PLOT_H}" '
        f'data-baseline="{HIST_BASELINE}">'
    ),
    (
        f'<line x1="{HIST_L}" y1="{HIST_BASELINE}" '
        f'x2="{HIST_L + HIST_PLOT_W}" y2="{HIST_BASELINE}" '
        f'class="axis-line"/>'
    ),
    (
        f'<line x1="{HIST_L}" y1="{HIST_T}" '
        f'x2="{HIST_L}" y2="{HIST_BASELINE}" '
        f'class="axis-line"/>'
    ),
]

for frac in [0, 0.25, 0.5, 0.75, 1.0]:
    y = HIST_T + HIST_PLOT_H * (1 - frac)
    val = int(round(max_hist_count * frac))

    hist_svg_parts.append(
        f'<line x1="{HIST_L}" y1="{y:.2f}" '
        f'x2="{HIST_L + HIST_PLOT_W}" y2="{y:.2f}" '
        f'class="grid-line"/>'
    )
    hist_svg_parts.append(
        f'<text data-hist-ylabel="{frac}" '
        f'x="{HIST_L - 12}" y="{y + 4:.2f}" '
        f'text-anchor="end" class="axis-label">{val}</text>'
    )

bar_gap = 8
n_hist = max(len(hist_items), 1)
slot_w = HIST_PLOT_W / n_hist
bar_w = max(10, min(42, slot_w - bar_gap))

for i, (score, counts) in enumerate(hist_items):
    interesting_count = counts["interesting"]
    not_interesting_count = counts["not_interesting"]
    total_count = interesting_count + not_interesting_count

    x_center = HIST_L + slot_w * (i + 0.5)
    x = x_center - bar_w / 2

    blue_h = (
        HIST_PLOT_H * not_interesting_count / max_hist_count
        if max_hist_count
        else 0
    )
    red_h = (
        HIST_PLOT_H * interesting_count / max_hist_count
        if max_hist_count
        else 0
    )

    blue_y = HIST_BASELINE - blue_h
    red_y = blue_y - red_h

    title = (
        f"ImageScore {score:g}: "
        f"blue/not interesting = {not_interesting_count}; "
        f"red/interesting = {interesting_count}; "
        f"total = {total_count}"
    )

    hist_svg_parts.append(
        f'<rect data-hist-bar="blue" '
        f'data-count="{not_interesting_count}" '
        f'data-score="{score:g}" '
        f'x="{x:.2f}" y="{blue_y:.2f}" '
        f'width="{bar_w:.2f}" height="{blue_h:.2f}" '
        f'class="hist-blue-svg">'
        f'<title>{html.escape(title)}</title>'
        f'</rect>'
    )

    hist_svg_parts.append(
        f'<rect data-hist-bar="red" '
        f'data-count="{interesting_count}" '
        f'data-score="{score:g}" '
        f'x="{x:.2f}" y="{red_y:.2f}" '
        f'width="{bar_w:.2f}" height="{red_h:.2f}" '
        f'class="hist-red-svg">'
        f'<title>{html.escape(title)}</title>'
        f'</rect>'
    )

    hist_svg_parts.append(
        f'<text x="{x_center:.2f}" y="{HIST_BASELINE + 22}" '
        f'text-anchor="middle" class="axis-label">{score:g}</text>'
    )

hist_svg_parts.append(
    f'<text x="{HIST_L + HIST_PLOT_W / 2}" y="{HIST_H - 10}" '
    f'text-anchor="middle" class="axis-title">ImageScore bin</text>'
)
hist_svg_parts.append(
    f'<text x="16" y="{HIST_T + HIST_PLOT_H / 2}" '
    f'text-anchor="middle" class="axis-title" '
    f'transform="rotate(-90 16 {HIST_T + HIST_PLOT_H / 2})">Count</text>'
)
hist_svg_parts.append("</svg>")

hist_svg = "\n".join(hist_svg_parts)


# ── 8. PRECISION-RECALL SVG ────────────────────────────────────────────────

PR_W = 980
PR_H = 430
PR_L = 64
PR_R = 24
PR_T = 28
PR_B = 62
PR_PLOT_W = PR_W - PR_L - PR_R
PR_PLOT_H = PR_H - PR_T - PR_B


def pr_x(recall):
    return PR_L + recall * PR_PLOT_W


def pr_y(precision):
    return PR_T + (1 - precision) * PR_PLOT_H


pr_plot_points = sorted(
    threshold_metrics,
    key=lambda m: (m["recall"], m["precision"]),
)

polyline_points = " ".join(
    f'{pr_x(m["recall"]):.2f},{pr_y(m["precision"]):.2f}'
    for m in pr_plot_points
)

pr_svg_parts = [
    (
        f'<svg class="plot-svg" viewBox="0 0 {PR_W} {PR_H}" '
        f'role="img" aria-label="Precision recall curve">'
    ),
    (
        f'<line x1="{PR_L}" y1="{PR_T + PR_PLOT_H}" '
        f'x2="{PR_L + PR_PLOT_W}" y2="{PR_T + PR_PLOT_H}" '
        f'class="axis-line"/>'
    ),
    (
        f'<line x1="{PR_L}" y1="{PR_T}" '
        f'x2="{PR_L}" y2="{PR_T + PR_PLOT_H}" '
        f'class="axis-line"/>'
    ),
]

for frac in [0, 0.25, 0.5, 0.75, 1.0]:
    x = PR_L + frac * PR_PLOT_W
    y = PR_T + (1 - frac) * PR_PLOT_H

    pr_svg_parts.append(
        f'<line x1="{x:.2f}" y1="{PR_T}" '
        f'x2="{x:.2f}" y2="{PR_T + PR_PLOT_H}" '
        f'class="grid-line"/>'
    )
    pr_svg_parts.append(
        f'<line x1="{PR_L}" y1="{y:.2f}" '
        f'x2="{PR_L + PR_PLOT_W}" y2="{y:.2f}" '
        f'class="grid-line"/>'
    )
    pr_svg_parts.append(
        f'<text x="{x:.2f}" y="{PR_T + PR_PLOT_H + 24}" '
        f'text-anchor="middle" class="axis-label">{frac:.2f}</text>'
    )
    pr_svg_parts.append(
        f'<text x="{PR_L - 12}" y="{y + 4:.2f}" '
        f'text-anchor="end" class="axis-label">{frac:.2f}</text>'
    )

if polyline_points:
    pr_svg_parts.append(
        f'<polyline points="{polyline_points}" class="pr-line"/>'
    )

label_stride = max(1, len(pr_plot_points) // 14)

for i, m in enumerate(pr_plot_points):
    x = pr_x(m["recall"])
    y = pr_y(m["precision"])

    is_best = abs(float(m["threshold"]) - best_threshold) < 1e-12
    cls = "pr-point best" if is_best else "pr-point"
    r_point = 5 if is_best else 3.5

    title = (
        f'threshold={m["threshold"]:.3g}, '
        f'precision={m["precision"]:.3f}, '
        f'recall={m["recall"]:.3f}, '
        f'F1={m["f1"]:.3f}'
    )

    pr_svg_parts.append(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r_point}" '
        f'class="{cls}"><title>{html.escape(title)}</title></circle>'
    )

    if is_best or (i % label_stride == 0 and m["threshold"] <= max_score):
        pr_svg_parts.append(
            f'<text x="{x:.2f}" y="{y - 10:.2f}" '
            f'text-anchor="middle" class="threshold-label">'
            f'{m["threshold"]:.2g}</text>'
        )

pr_svg_parts.append(
    f'<text x="{PR_L + PR_PLOT_W / 2}" y="{PR_H - 14}" '
    f'text-anchor="middle" class="axis-title">Recall</text>'
)
pr_svg_parts.append(
    f'<text x="18" y="{PR_T + PR_PLOT_H / 2}" '
    f'text-anchor="middle" class="axis-title" '
    f'transform="rotate(-90 18 {PR_T + PR_PLOT_H / 2})">Precision</text>'
)
pr_svg_parts.append("</svg>")

pr_svg = "\n".join(pr_svg_parts)


# ── 8b. AVG AI SCORE BY CLASSIFICATION BAR CHART ──────────────────────────

# Bucket every row by its classification label.
# Rows with no classification get the key "" (shown as "No classification").
cls_score_buckets: dict[str, list[float]] = {}

for r in rows:
    cls_key = str(r.get("classification", "")).strip()
    cls_score_buckets.setdefault(cls_key, []).append(float(r["imagescore"]))

# Build (label, avg) pairs; sort ascending by avg; append overall total last.
cls_avg_pairs = []
for cls_key, scores_list in cls_score_buckets.items():
    label = cls_key if cls_key else "No classification"
    cls_avg_pairs.append((label, sum(scores_list) / len(scores_list)))

cls_avg_pairs.sort(key=lambda x: x[1])  # ascending → lowest bar on left

overall_avg = sum(scores) / total if total else 0.0
cls_avg_pairs.append(("Total Average", overall_avg))  # always rightmost

# SVG geometry
CAVG_W = 980
CAVG_H = 420
CAVG_L = 68   # left margin (y-axis labels)
CAVG_R = 22
CAVG_T = 28
CAVG_B = 90   # extra bottom for rotated x-axis labels
CAVG_PLOT_W = CAVG_W - CAVG_L - CAVG_R
CAVG_PLOT_H = CAVG_H - CAVG_T - CAVG_B
CAVG_BASELINE = CAVG_T + CAVG_PLOT_H

cavg_max_val = max(v for _, v in cls_avg_pairs) if cls_avg_pairs else 1.0
cavg_y_max = cavg_max_val * 1.12  # 12 % headroom

n_cavg = len(cls_avg_pairs)
cavg_slot_w = CAVG_PLOT_W / n_cavg
cavg_bar_w = max(8, min(38, cavg_slot_w - 6))

cavg_svg_parts = [
    (
        f'<svg class="plot-svg" '
        f'viewBox="0 0 {CAVG_W} {CAVG_H}" '
        f'role="img" '
        f'aria-label="Average AI score by classification">'
    ),
    (
        f'<line x1="{CAVG_L}" y1="{CAVG_BASELINE}" '
        f'x2="{CAVG_L + CAVG_PLOT_W}" y2="{CAVG_BASELINE}" '
        f'class="axis-line"/>'
    ),
    (
        f'<line x1="{CAVG_L}" y1="{CAVG_T}" '
        f'x2="{CAVG_L}" y2="{CAVG_BASELINE}" '
        f'class="axis-line"/>'
    ),
]

# Y-axis grid lines and labels
for frac in [0, 0.25, 0.5, 0.75, 1.0]:
    y = CAVG_T + CAVG_PLOT_H * (1 - frac)
    val = cavg_y_max * frac
    cavg_svg_parts.append(
        f'<line x1="{CAVG_L}" y1="{y:.2f}" '
        f'x2="{CAVG_L + CAVG_PLOT_W}" y2="{y:.2f}" '
        f'class="grid-line"/>'
    )
    cavg_svg_parts.append(
        f'<text x="{CAVG_L - 8}" y="{y + 4:.2f}" '
        f'text-anchor="end" class="axis-label">{val:.1f}</text>'
    )

for i, (label, avg_val) in enumerate(cls_avg_pairs):
    x_center = CAVG_L + cavg_slot_w * (i + 0.5)
    x = x_center - cavg_bar_w / 2
    bar_h = CAVG_PLOT_H * avg_val / cavg_y_max if cavg_y_max else 0
    bar_y = CAVG_BASELINE - bar_h

    is_total = label == "Total Average"
    bar_color = "var(--accent)" if is_total else "var(--accent2)"
    bar_opacity = "1" if is_total else "0.85"

    title_text = f"{html.escape(label)}: avg score = {avg_val:.2f}"
    cavg_svg_parts.append(
        f'<rect x="{x:.2f}" y="{bar_y:.2f}" '
        f'width="{cavg_bar_w:.2f}" height="{bar_h:.2f}" '
        f'fill="{bar_color}" opacity="{bar_opacity}">'
        f'<title>{title_text}</title>'
        f'</rect>'
    )

    # Value label above bar
    cavg_svg_parts.append(
        f'<text x="{x_center:.2f}" y="{bar_y - 5:.2f}" '
        f'text-anchor="middle" class="axis-label" '
        f'style="font-size:9px">{avg_val:.1f}</text>'
    )

    # Rotated x-axis label
    label_y = CAVG_BASELINE + 8
    safe_label = html.escape(label)
    cavg_svg_parts.append(
        f'<text '
        f'x="{x_center:.2f}" y="{label_y}" '
        f'text-anchor="end" class="axis-label" '
        f'style="font-size:10px" '
        f'transform="rotate(-40 {x_center:.2f} {label_y})">'
        f'{safe_label}</text>'
    )

cavg_svg_parts.append(
    f'<text x="{CAVG_L + CAVG_PLOT_W / 2}" y="{CAVG_H - 4}" '
    f'text-anchor="middle" class="axis-title">Classification</text>'
)
cavg_svg_parts.append(
    f'<text x="18" y="{CAVG_T + CAVG_PLOT_H / 2}" '
    f'text-anchor="middle" class="axis-title" '
    f'transform="rotate(-90 18 {CAVG_T + CAVG_PLOT_H / 2})">Avg AI Score</text>'
)
cavg_svg_parts.append("</svg>")
cavg_svg = "\n".join(cavg_svg_parts)


# ── 8c. DETAILED PER-CLASSIFICATION HISTOGRAM DATA ────────────────────────

# Build a dict: subset_key -> {score_bin -> count}
# subset_key "" means "not interesting / no classification"
# We also add keys for True Positives (tp), False Positives (fp), etc.
detail_hist_data: dict[str, dict[int, int]] = {}

for r in rows:
    cls_key = str(r.get("classification", "")).strip()
    score = float(r["imagescore"])
    score_bin = int(score)
    
    # Standard classification buckets
    detail_hist_data.setdefault(cls_key, {})
    detail_hist_data[cls_key][score_bin] = detail_hist_data[cls_key].get(score_bin, 0) + 1

    # Prediction type buckets (TP/FP/FN/TN)
    truth = int(r["interesting"])
    pred = 1 if score >= best_threshold else 0
    
    if truth == 1 and pred == 1:
        err_key = "tp"
    elif truth == 0 and pred == 1:
        err_key = "fp"
    elif truth == 1 and pred == 0:
        err_key = "fn"
    else:
        err_key = "tn"

    detail_hist_data.setdefault(err_key, {})
    detail_hist_data[err_key][score_bin] = detail_hist_data[err_key].get(score_bin, 0) + 1

# Also add an "all" key that covers every row
all_bins: dict[int, int] = {}
for r in rows:
    b = int(float(r["imagescore"]))
    all_bins[b] = all_bins.get(b, 0) + 1
detail_hist_data["__all__"] = all_bins

# Serialise for embedding in HTML (keys are strings for JSON)
import json as _json

detail_hist_json = _json.dumps(
    {k: {str(bin_k): v for bin_k, v in bins.items()}
     for k, bins in detail_hist_data.items()},
    separators=(",", ":"),
)

# ── 9. BUILD IMAGE GRID CARDS ──────────────────────────────────────────────

# Only show images that were converted to JPEG (top MAX_JPEG_IMAGES by score,
# plus every "interesting" row regardless of score).
# All rows are kept for plots and statistical analysis.
grid_rows = [
    r for r in rows
    if r.get("data_url") and r.get("data_url").strip()
]

if MAX_CARDS is not None:
    grid_rows = grid_rows[:MAX_CARDS]

classification_values = sorted({
    str(r.get("classification", "")).strip()
    for r in rows
    if int(r.get("interesting", 0)) == 1
    and str(r.get("classification", "")).strip()
})

# Build the dropdown options for the detailed histogram
# Order: All → Prediction Types (TP/FP/TN/FN) → Classifications → No classification
detail_cls_options = '<option value="__all__">All images</option>'

detail_cls_options += '<optgroup label="Prediction Types">'
detail_cls_options += '<option value="tp">True Positives</option>'
detail_cls_options += '<option value="fp">False Positives</option>'
detail_cls_options += '<option value="fn">False Negatives</option>'
detail_cls_options += '<option value="tn">True Negatives</option>'
detail_cls_options += '</optgroup>'

detail_cls_options += '<optgroup label="Classifications">'
for cls_name in classification_values:            # already sorted alphabetically
    safe_v = html.escape(cls_name, quote=True)
    safe_l = html.escape(cls_name)
    detail_cls_options += f'<option value="{safe_v}">{safe_l}</option>'
detail_cls_options += '<option value="">No classification / not interesting</option>'
detail_cls_options += '</optgroup>'

# Geometry constants re-used by the JS renderer (match existing hist sizing)
DHIST_W = HIST_W
DHIST_H = HIST_H
DHIST_L = HIST_L
DHIST_R = HIST_R
DHIST_T = HIST_T
DHIST_B = HIST_B
DHIST_PLOT_W = HIST_PLOT_W
DHIST_PLOT_H = HIST_PLOT_H
DHIST_BASELINE = HIST_BASELINE

classification_options = '<option value="all">All classifications</option>'
classification_options += '<option value="none">No classification / not interesting</option>'

for cls_name in classification_values:
    safe_value = html.escape(cls_name.lower(), quote=True)
    safe_label = html.escape(cls_name)
    classification_options += f'<option value="{safe_value}">{safe_label}</option>'

cards = ""

for rank, r in enumerate(grid_rows, start=1):
    filename = html.escape(str(r["filename"]))
    image_score = float(r["imagescore"])
    interesting = int(r["interesting"])

    classification = str(r.get("classification", "")).strip()
    classification_safe = html.escape(classification)
    classification_data = html.escape(classification.lower(), quote=True)

    score_col = score_color(image_score, max_score)
    score_pct = (image_score / max_score * 100) if max_score > 0 else 0

    interesting_label = "Interesting" if interesting == 1 else "Not interesting"
    interesting_class = "positive" if interesting == 1 else "negative"

    search_text = html.escape(
        f"{filename} {image_score:g} {interesting_label} {classification}".lower(),
        quote=True,
    )

    source_ra = float(r.get("SourceRA", 0.0))
    source_dec = float(r.get("SourceDec", 0.0))
    has_coords = bool(source_ra or source_dec)

    legacy_survey_url = (
        f"https://www.legacysurvey.org/viewer?ra={source_ra}&dec={source_dec}"
        f"&layer=ls-dr9&zoom=12"
        if has_coords
        else ""
    )

    if r.get("data_url"):
        safe_caption = html.escape(
            (
                f"{filename} | ImageScore={image_score:g} | "
                f"Ground truth: {interesting_label}"
                + (f" | Classification: {classification}" if classification else "")
            ),
            quote=True,
        )

        img_block = (
            f'<img src="{r["data_url"]}" '
            f'alt="{filename}" '
            f'loading="lazy" '
            f'onclick="openModal(this.src, this.dataset.caption)" '
            f'data-caption="{safe_caption}">'
        )
    else:
        img_block = '<div class="no-img">Image unavailable</div>'

    meta_blocks = ""

    if classification:
        meta_blocks += f"""
          <div class="coord">
            <span class="clabel">Classification</span>
            <span class="cval">{classification_safe}</span>
          </div>"""

    if has_coords:
        meta_blocks += f"""
          <div class="coord">
            <span class="clabel">RA</span>
            <span class="cval">{source_ra:.6f}°</span>
          </div>
          <div class="coord">
            <span class="clabel">Dec</span>
            <span class="cval">{source_dec:.6f}°</span>
          </div>"""

    cards += f"""
    <article
      class="card"
      data-score="{image_score}"
      data-interesting="{interesting}"
      data-classification="{classification_data}"
      data-search="{search_text}"
    >
      <div class="img-wrap">{img_block}</div>

      <div class="truth-under-image {interesting_class}">
        Ground truth: {interesting_label}
      </div>

      <div class="card-body">
        <div class="rank-row">
          <span class="rank">#{rank}</span>
          <span class="tag {interesting_class}">{interesting_label}</span>
        </div>

        <div class="prediction-badge" data-prediction-badge>
          Prediction pending
        </div>

        <h2 class="fname" title="{filename}">{
          f'<a href="{legacy_survey_url}" target="_blank" rel="noopener noreferrer">{filename}</a>'
          if legacy_survey_url else filename
        }</h2>

        <div class="meta-grid">
          {meta_blocks}
        </div>

        <div class="score-item">
          <div class="score-header">
            <span class="slabel">AI ImageScore</span>
            <span class="snum" style="color:{score_col}">{image_score:g}</span>
          </div>
          <div class="bar-track">
            <div
              class="bar-fill"
              style="width:{score_pct:.1f}%;background:{score_col}"
            ></div>
          </div>
        </div>
      </div>
    </article>"""


# ── 10. HTML DOCUMENT ──────────────────────────────────────────────────────

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mixed Gemini ImageScore Report</title>

<style>
*,*::before,*::after {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

:root {{
  --bg: #040812;
  --surface: #0b1120;
  --surface2: #111a2e;
  --border: rgba(255,255,255,.08);
  --text: #b8c8e0;
  --text-dim: #6a7897;
  --accent: #00e5a0;
  --accent2: #7b78ff;
  --danger: #ff4d6d;
  --blue: #4f8cff;
  --warn: #f0b429;
  --good: #00e5a0;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
          "Liberation Mono", "Courier New", monospace;
  --display: Inter, system-ui, -apple-system, BlinkMacSystemFont,
             "Segoe UI", sans-serif;
}}

body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1.7;
}}

body::before {{
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(1px 1px at 12% 18%, rgba(255,255,255,.7) 0%, transparent 100%),
    radial-gradient(1px 1px at 67% 55%, rgba(255,255,255,.4) 0%, transparent 100%),
    radial-gradient(1.5px 1.5px at 91% 80%, rgba(123,120,255,.6) 0%, transparent 100%);
}}

.page {{
  position: relative;
  z-index: 1;
  max-width: 1240px;
  margin: 0 auto;
  padding: 42px 24px 90px;
}}

header {{
  margin-bottom: 26px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 24px;
}}

.eyebrow {{
  font-size: 10px;
  letter-spacing: .25em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 10px;
}}

header h1 {{
  font-family: var(--display);
  font-size: clamp(32px, 5vw, 64px);
  font-weight: 900;
  color: #fff;
  line-height: 1.02;
  letter-spacing: -.045em;
}}

header h1 em {{
  font-style: normal;
  color: var(--accent2);
}}

.stats {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 24px;
}}

.stat {{
  background: var(--surface);
  padding: 18px 20px;
}}

.stat-label {{
  font-size: 9px;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 4px;
}}

.stat-val {{
  font-family: var(--display);
  font-size: 27px;
  font-weight: 900;
  color: #fff;
}}

.stat-val.g {{ color: var(--accent); }}
.stat-val.p {{ color: var(--accent2); }}
.stat-val.r {{ color: var(--danger); }}
.stat-val.b {{ color: var(--blue); }}

.tab-controls {{
  display: flex;
  gap: 16px;
  margin-bottom: 22px;
  border-bottom: 1px solid var(--border);
}}

.tab-btn {{
  background: transparent;
  color: var(--text-dim);
  border: none;
  padding: 12px 20px;
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: .1em;
  position: relative;
}}

.tab-btn.active {{
  color: var(--accent);
}}

.tab-btn.active::after {{
  content: "";
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 100%;
  height: 2px;
  background: var(--accent);
}}

.tab-btn:hover:not(.active) {{
  color: #fff;
}}

.tab-panel {{
  display: none;
}}

.tab-panel.active {{
  display: block;
}}

.controls {{
  display: grid;
  grid-template-columns: minmax(220px, 1fr) auto auto auto;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
  padding: 16px;
  background: rgba(255,255,255,.03);
  border: 1px solid var(--border);
  border-radius: 12px;
  position: sticky;
  top: 0;
  z-index: 20;
  backdrop-filter: blur(10px);
}}

.search-box,
select,
.plot-controls input {{
  background: #060d1a;
  color: #fff;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  font-family: var(--mono);
}}

.result-count {{
  margin: 0 0 18px;
  color: var(--text-dim);
  font-size: 11px;
  letter-spacing: .08em;
  text-transform: uppercase;
}}

.grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 18px;
}}

.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: transform .2s, box-shadow .2s, border-color .2s;
}}

.card:hover {{
  transform: translateY(-3px);
  box-shadow: 0 18px 45px rgba(0,0,0,.65);
  border-color: rgba(123,120,255,.35);
}}

.card.hidden {{
  display: none;
}}

.card.tp {{ border-color: rgba(0,229,160,.45); }}
.card.fp {{ border-color: rgba(79,140,255,.50); }}
.card.fn {{ border-color: rgba(255,77,109,.60); }}
.card.tn {{ border-color: rgba(255,255,255,.08); }}

.img-wrap {{
  width: 100%;
  aspect-ratio: 1 / 1;
  background: #060d1a;
  overflow: hidden;
  position: relative;
}}

.img-wrap img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  cursor: zoom-in;
  transition: transform .3s ease;
}}

.card:hover .img-wrap img {{
  transform: scale(1.03);
}}

.no-img {{
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-dim);
  font-size: 11px;
  letter-spacing: .15em;
  text-transform: uppercase;
}}

.truth-under-image {{
  padding: 8px 12px;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  font-size: 10px;
  letter-spacing: .08em;
  text-transform: uppercase;
  background: rgba(255,255,255,.025);
}}

.truth-under-image.positive {{ color: var(--danger); }}
.truth-under-image.negative {{ color: var(--blue); }}

.card-body {{
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
}}

.rank-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}}

.rank {{
  color: var(--accent2);
  font-weight: 700;
}}

.tag,
.prediction-badge {{
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 9px;
  letter-spacing: .08em;
  text-transform: uppercase;
}}

.tag.positive {{
  color: var(--danger);
  background: rgba(255,77,109,.08);
  border-color: rgba(255,77,109,.28);
}}

.tag.negative {{
  color: var(--blue);
  background: rgba(79,140,255,.08);
  border-color: rgba(79,140,255,.28);
}}

.prediction-badge {{
  display: inline-flex;
  width: max-content;
}}

.prediction-badge.tp {{
  color: var(--good);
  background: rgba(0,229,160,.08);
  border-color: rgba(0,229,160,.35);
}}

.prediction-badge.fp {{
  color: var(--blue);
  background: rgba(79,140,255,.08);
  border-color: rgba(79,140,255,.35);
}}

.prediction-badge.fn {{
  color: var(--danger);
  background: rgba(255,77,109,.08);
  border-color: rgba(255,77,109,.35);
}}

.prediction-badge.tn {{
  color: var(--text-dim);
}}

.fname {{
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  letter-spacing: .04em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.fname a {{
  color: inherit;
  text-decoration: none;
}}

.fname a:hover {{
  color: var(--accent2);
  text-decoration: underline;
}}

.meta-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}}

.coord {{
  min-width: 0;
  border-top: 1px solid var(--border);
  padding-top: 8px;
}}

.clabel {{
  display: block;
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: .15em;
  color: var(--text-dim);
}}

.cval {{
  display: block;
  color: #fff;
  font-size: 11px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}}

.score-item {{
  margin-top: auto;
}}

.score-header {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 7px;
}}

.slabel {{
  font-size: 9px;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--text-dim);
}}

.snum {{
  font-size: 18px;
  font-weight: 700;
}}

.bar-track {{
  height: 6px;
  background: rgba(255,255,255,.07);
  border-radius: 999px;
  overflow: hidden;
}}

.bar-fill {{
  height: 100%;
  border-radius: 999px;
}}

.plot-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 22px;
}}

.plot-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px;
  overflow: hidden;
}}

.plot-title {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 6px;
}}

.plot-title h2 {{
  font-family: var(--display);
  font-size: 19px;
  color: #fff;
}}

.plot-title span {{
  font-size: 11px;
  color: var(--text-dim);
}}

.plot-note {{
  color: var(--text-dim);
  font-size: 11px;
  margin-bottom: 14px;
}}

.plot-controls {{
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  color: var(--text-dim);
  font-size: 10px;
  letter-spacing: .12em;
  text-transform: uppercase;
}}

.plot-controls input {{
  width: 120px;
}}

.plot-svg {{
  width: 100%;
  height: auto;
  display: block;
  background: rgba(255,255,255,.015);
  border-radius: 10px;
}}

.axis-line {{
  stroke: rgba(255,255,255,.35);
  stroke-width: 1;
}}

.grid-line {{
  stroke: rgba(255,255,255,.06);
  stroke-width: 1;
}}

.axis-label {{
  fill: var(--text-dim);
  font-size: 11px;
  font-family: var(--mono);
}}

.axis-title {{
  fill: var(--text);
  font-size: 12px;
  font-family: var(--mono);
  font-weight: 700;
}}

.hist-blue-svg {{
  fill: var(--blue);
  opacity: .9;
}}

.hist-red-svg {{
  fill: var(--danger);
  opacity: .95;
}}

.pr-line {{
  fill: none;
  stroke: var(--accent);
  stroke-width: 2.5;
}}

.pr-point {{
  fill: var(--accent);
  stroke: #04100d;
  stroke-width: 1.5;
}}

.pr-point.best {{
  fill: var(--danger);
  stroke: #fff;
  stroke-width: 1.5;
}}

.threshold-label {{
  fill: var(--text);
  font-size: 10px;
  font-family: var(--mono);
}}

.legend {{
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  color: var(--text-dim);
  font-size: 11px;
  margin: 10px 0 0;
}}

.legend span {{
  display: flex;
  align-items: center;
  gap: 7px;
}}

.swatch {{
  width: 12px;
  height: 12px;
  border-radius: 3px;
  display: inline-block;
}}

.swatch.blue {{ background: var(--blue); }}
.swatch.red {{ background: var(--danger); }}
.swatch.green {{ background: var(--accent); }}

.detail-hist-controls {{
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}}

.detail-hist-controls label {{
  font-size: 10px;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--text-dim);
}}

.detail-hist-controls select {{
  background: #060d1a;
  color: #fff;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  font-family: var(--mono);
  font-size: 12px;
  min-width: 200px;
}}

.modal {{
  display: none;
  position: fixed;
  z-index: 999;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(4,8,18,.92);
  backdrop-filter: blur(4px);
  cursor: pointer;
}}

.modal-content {{
  margin: auto;
  display: block;
  max-width: 92%;
  max-height: 82vh;
  margin-top: 5vh;
  border-radius: 8px;
  box-shadow: 0 24px 60px rgba(0,0,0,.7);
  border: 1px solid var(--border);
}}

.modal-caption {{
  margin: auto;
  display: block;
  width: 80%;
  text-align: center;
  color: #fff;
  padding: 15px 0;
  font-family: var(--mono);
  font-size: 14px;
}}

.close-modal {{
  position: absolute;
  top: 20px;
  right: 35px;
  color: var(--text-dim);
  font-size: 40px;
  font-weight: bold;
  transition: color .2s;
}}

.close-modal:hover {{
  color: var(--accent);
}}

footer {{
  margin-top: 72px;
  border-top: 1px solid var(--border);
  padding-top: 20px;
  color: var(--text-dim);
  font-size: 10px;
  letter-spacing: .12em;
  text-transform: uppercase;
}}

@media(max-width: 1050px) {{
  .grid {{
    grid-template-columns: repeat(3, 1fr);
  }}

  .stats {{
    grid-template-columns: repeat(3, 1fr);
  }}

  .controls {{
    grid-template-columns: 1fr 1fr;
  }}
}}

@media(max-width: 760px) {{
  .grid {{
    grid-template-columns: repeat(2, 1fr);
  }}

  .stats {{
    grid-template-columns: 1fr;
  }}

  .controls {{
    position: static;
    grid-template-columns: 1fr;
  }}
}}

@media(max-width: 520px) {{
  .grid {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>

<body>
<div class="page">
  <header>
    <p class="eyebrow">Gemini Vision Pipeline · Mixed HDF5/JPEG Report</p>
    <h1>Hubble Image <em>Scoring</em></h1>
  </header>

  <div class="stats">
    <div class="stat">
      <div class="stat-label">Images in CSV</div>
      <div class="stat-val">{total}</div>
    </div>
    <div class="stat">
      <div class="stat-label">True interesting</div>
      <div class="stat-val r">{num_interesting}</div>
    </div>
    <div class="stat">
      <div class="stat-label">True not interesting</div>
      <div class="stat-val b">{num_not_interesting}</div>
    </div>
    <div class="stat">
      <div class="stat-label">PR-AUC</div>
      <div class="stat-val g">{pr_auc:.3f}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Best F1 threshold</div>
      <div class="stat-val p">{best_threshold:.3g}</div>
    </div>
  </div>

  <div class="tab-controls">
    <button class="tab-btn active" onclick="switchTab(event, 'grid-view')">
      Image Grid
    </button>
    <button class="tab-btn" onclick="switchTab(event, 'plots-view')">
      Data Plots
    </button>
  </div>

  <section id="grid-view" class="tab-panel active">
    <div class="controls">
      <input
        id="searchBox"
        class="search-box"
        type="text"
        placeholder="Search filename, score, or classification..."
        oninput="applyFilters()"
      >

      <select id="truthFilter" onchange="applyFilters()">
        <option value="all">All truth labels</option>
        <option value="1">True interesting</option>
        <option value="0">True not interesting</option>
      </select>

      <select id="classificationFilter" onchange="applyFilters()">
        {classification_options}
      </select>

      <select id="errorFilter" onchange="applyFilters()">
        <option value="all">All prediction types</option>
        <option value="tp">True positives</option>
        <option value="fp">False positives</option>
        <option value="fn">False negatives</option>
        <option value="tn">True negatives</option>
      </select>
    </div>

    <div class="result-count" id="resultCount">
      Showing {len(grid_rows)} / {len(grid_rows)} grid images · {total} images kept for plot analysis
    </div>

    <div class="grid" id="imageGrid">
      {cards}
    </div>
  </section>

  <section id="plots-view" class="tab-panel">
    <div class="plot-grid">
      <article class="plot-card">
        <div class="plot-title">
          <h2>ImageScore histogram by ground truth</h2>
          <span>Vertical stacked histogram</span>
        </div>

        <p class="plot-note">
          Blue bars are non-interesting images. Red bars are truly interesting images.
          Hover over each bar for both counts. The y-axis maximum is adjustable below.
        </p>

        <div class="plot-controls">
          <label for="histYMax">Histogram y-axis max</label>
          <input
            id="histYMax"
            type="number"
            min="1"
            step="1"
            value="{max_hist_count}"
            oninput="updateHistogramYMax()"
          >
        </div>

        {hist_svg}

        <div class="legend">
          <span><i class="swatch blue"></i>Not interesting</span>
          <span><i class="swatch red"></i>Interesting</span>
        </div>
      </article>

      <article class="plot-card">
        <div class="plot-title">
          <h2>Precision–recall curve</h2>
          <span>PR-AUC = {pr_auc:.3f}</span>
        </div>

        <p class="plot-note">
          Prediction rule: selected if ImageScore ≥ threshold.
          Best-F1 threshold = {best_threshold:.3g},
          precision = {best_precision:.3f},
          recall = {best_recall:.3f},
          F1 = {best_f1:.3f}.
        </p>

        {pr_svg}

        <div class="legend">
          <span><i class="swatch green"></i>PR curve</span>
          <span><i class="swatch red"></i>Best-F1 point</span>
        </div>
      </article>
      <article class="plot-card">
        <div class="plot-title">
          <h2>Average AI score by classification</h2>
          <span>Sorted ascending · total average shown separately</span>
        </div>

        <p class="plot-note">
          Each bar shows the mean AI ImageScore for all images with that classification label.
          Images with no classification are grouped together.
          The rightmost bar (<strong>Total Average</strong>) covers all images in the dataset.
          Hover over a bar for the exact value.
        </p>

        {cavg_svg}

        <div class="legend">
          <span><i class="swatch" style="background:var(--accent2)"></i>Classification average</span>
          <span><i class="swatch green"></i>Total average</span>
        </div>
      </article>

      <article class="plot-card">
        <div class="plot-title">
          <h2>AI score distribution by subset</h2>
          <span>Detailed histogram</span>
        </div>

        <p class="plot-note">
          Select a classification or prediction type (TP/FP/TN/FN) from the dropdown to see the full AI score histogram
          for that group. All images in the selected category are included.
        </p>

        <div class="detail-hist-controls">
          <label for="detailHistSelect">Subset</label>
          <select id="detailHistSelect" onchange="renderDetailHist(this.value)">
            {detail_cls_options}
          </select>
        </div>

        <div id="detailHistContainer"></div>

        <div class="legend">
          <span><i class="swatch blue"></i>Image count per AI score bin</span>
        </div>
      </article>
    </div>
  </section>
  <footer>
    Generated from {html.escape(os.path.basename(CLASSIFICATION_CSV))}
    · HDF5 source: {html.escape(os.path.basename(HDF5_PATH))}
    · JPEG conversion: {saved_jpeg_images} images embedded
    (top {MAX_JPEG_IMAGES} by score + all {num_interesting} interesting, minus overlap)
  </footer>
</div>

<div id="imageModal" class="modal" onclick="closeModal()">
  <span class="close-modal">&times;</span>
  <img class="modal-content" id="modalImg" src="">
  <div class="modal-caption" id="modalCaption"></div>
</div>

<script>
const BEST_THRESHOLD = {best_threshold:.12g};

function openModal(src, caption) {{
  document.getElementById("imageModal").style.display = "block";
  document.getElementById("modalImg").src = src;
  document.getElementById("modalCaption").textContent = caption || "";
}}

function closeModal() {{
  document.getElementById("imageModal").style.display = "none";
}}

document.addEventListener("keydown", event => {{
  if (event.key === "Escape") closeModal();
}});

function switchTab(evt, tabName) {{
  document.querySelectorAll(".tab-panel").forEach(panel => {{
    panel.classList.remove("active");
  }});

  document.querySelectorAll(".tab-btn").forEach(btn => {{
    btn.classList.remove("active");
  }});

  document.getElementById(tabName).classList.add("active");
  evt.currentTarget.classList.add("active");
}}

function classifyCard(card) {{
  const score = Number(card.dataset.score || 0);
  const truth = Number(card.dataset.interesting || 0);
  const pred = score >= BEST_THRESHOLD ? 1 : 0;

  if (truth === 1 && pred === 1) return "tp";
  if (truth === 0 && pred === 1) return "fp";
  if (truth === 1 && pred === 0) return "fn";
  return "tn";
}}

function labelForType(type) {{
  if (type === "tp") return "True positive";
  if (type === "fp") return "False positive";
  if (type === "fn") return "False negative";
  return "True negative";
}}

function applyFilters() {{
  const query = document.getElementById("searchBox").value.toLowerCase().trim();
  const truthFilter = document.getElementById("truthFilter").value;
  const classificationFilter = document.getElementById("classificationFilter").value.toLowerCase();
  const errorFilter = document.getElementById("errorFilter").value;

  const cards = Array.from(document.querySelectorAll(".card"));
  let shown = 0;

  cards.forEach(card => {{
    const haystack = card.dataset.search || card.textContent.toLowerCase();
    const cardInteresting = card.dataset.interesting;
    const cardClassification = (card.dataset.classification || "").toLowerCase();
    const type = classifyCard(card);

    card.classList.remove("tp", "fp", "fn", "tn");
    card.classList.add(type);

    const badge = card.querySelector("[data-prediction-badge]");

    if (badge) {{
      badge.classList.remove("tp", "fp", "fn", "tn");
      badge.classList.add(type);
      badge.textContent = labelForType(type);
    }}

    const matchesSearch = query === "" || haystack.includes(query);
    const matchesTruth = truthFilter === "all" || cardInteresting === truthFilter;

    const matchesClassification =
      classificationFilter === "all" ||
      (classificationFilter === "none" && cardClassification === "") ||
      cardClassification === classificationFilter;

    const matchesErrorType =
      errorFilter === "all" || type === errorFilter;

    const visible =
      matchesSearch &&
      matchesTruth &&
      matchesClassification &&
      matchesErrorType;

    card.classList.toggle("hidden", !visible);

    if (visible) shown += 1;
  }});

  document.getElementById("resultCount").textContent =
    `Showing ${{shown}} / ${{cards.length}} grid images · {total} images kept for plot analysis`;
}}

function updateHistogramYMax() {{
  const svg = document.getElementById("histogramSvg");
  const input = document.getElementById("histYMax");

  if (!svg || !input) return;

  const defaultMax = Number(svg.dataset.defaultYmax || 1);
  const yMax = Math.max(1, Number(input.value || defaultMax));
  const plotH = Number(svg.dataset.plotHeight || 1);
  const baseline = Number(svg.dataset.baseline || plotH);

  svg.querySelectorAll("[data-hist-ylabel]").forEach(label => {{
    const frac = Number(label.dataset.histYlabel || 0);
    label.textContent = String(Math.round(yMax * frac));
  }});

  const bars = Array.from(svg.querySelectorAll("[data-hist-bar]"));
  const byScore = new Map();

  bars.forEach(bar => {{
    const score = bar.dataset.score;

    if (!byScore.has(score)) {{
      byScore.set(score, {{}});
    }}

    byScore.get(score)[bar.dataset.histBar] = bar;
  }});

  byScore.forEach(pair => {{
    const blue = pair.blue;
    const red = pair.red;

    const blueCount = blue ? Number(blue.dataset.count || 0) : 0;
    const redCount = red ? Number(red.dataset.count || 0) : 0;

    const blueH = Math.min(plotH, plotH * blueCount / yMax);
    const redH = Math.min(plotH, plotH * redCount / yMax);

    if (blue) {{
      blue.setAttribute("height", blueH.toFixed(2));
      blue.setAttribute("y", (baseline - blueH).toFixed(2));
    }}

    if (red) {{
      red.setAttribute("height", redH.toFixed(2));
      red.setAttribute("y", (baseline - blueH - redH).toFixed(2));
    }}
  }});
}}


// ── Detail histogram data & renderer ──────────────────────────────────────
const DETAIL_HIST_DATA = {detail_hist_json};

const DH = {{
  W: {DHIST_W}, H: {DHIST_H},
  L: {DHIST_L}, R: {DHIST_R}, T: {DHIST_T}, B: {DHIST_B},
  PLOT_W: {DHIST_PLOT_W}, PLOT_H: {DHIST_PLOT_H},
  BASELINE: {DHIST_BASELINE}
}};

function renderDetailHist(clsKey) {{
  const container = document.getElementById("detailHistContainer");
  if (!container) return;

  const bins = DETAIL_HIST_DATA[clsKey] || {{}};
  const scoreKeys = Object.keys(bins).map(Number).sort((a, b) => a - b);

  if (scoreKeys.length === 0) {{
    container.innerHTML = '<p style="color:var(--text-dim);font-size:12px;padding:12px 0">No data for this selection.</p>';
    return;
  }}

  const maxCount = Math.max(...scoreKeys.map(k => bins[String(k)]));
  const n = scoreKeys.length;
  const slotW = DH.PLOT_W / n;
  const barW = Math.max(10, Math.min(42, slotW - 8));

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${{DH.W}} ${{DH.H}}`);
  svg.setAttribute("class", "plot-svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `AI score histogram for ${{clsKey || "no classification"}}`);

  function el(tag, attrs, title) {{
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (title) {{
      const t = document.createElementNS(ns, "title");
      t.textContent = title;
      e.appendChild(t);
    }}
    return e;
  }}

  // Axes
  svg.appendChild(el("line", {{ x1: DH.L, y1: DH.BASELINE, x2: DH.L + DH.PLOT_W, y2: DH.BASELINE, class: "axis-line" }}));
  svg.appendChild(el("line", {{ x1: DH.L, y1: DH.T, x2: DH.L, y2: DH.BASELINE, class: "axis-line" }}));

  // Y grid
  for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {{
    const y = DH.T + DH.PLOT_H * (1 - frac);
    const val = Math.round(maxCount * frac);
    svg.appendChild(el("line", {{ x1: DH.L, y1: y.toFixed(2), x2: DH.L + DH.PLOT_W, y2: y.toFixed(2), class: "grid-line" }}));
    const t = el("text", {{ x: DH.L - 12, y: (y + 4).toFixed(2), "text-anchor": "end", class: "axis-label" }});
    t.textContent = val;
    svg.appendChild(t);
  }}

  // Bars
  scoreKeys.forEach((score, i) => {{
    const count = bins[String(score)] || 0;
    const xCenter = DH.L + slotW * (i + 0.5);
    const x = xCenter - barW / 2;
    const bh = maxCount > 0 ? DH.PLOT_H * count / maxCount : 0;
    const by = DH.BASELINE - bh;

    svg.appendChild(el("rect", {{
      x: x.toFixed(2), y: by.toFixed(2),
      width: barW.toFixed(2), height: bh.toFixed(2),
      class: "hist-blue-svg"
    }}, `Score ${{score}}: ${{count}} image${{count !== 1 ? "s" : ""}}`));

    const xt = el("text", {{ x: xCenter.toFixed(2), y: DH.BASELINE + 22, "text-anchor": "middle", class: "axis-label" }});
    xt.textContent = score;
    svg.appendChild(xt);
  }});

  // Axis titles
  const xTitle = el("text", {{ x: (DH.L + DH.PLOT_W / 2).toFixed(2), y: DH.H - 10, "text-anchor": "middle", class: "axis-title" }});
  xTitle.textContent = "AI Score bin";
  svg.appendChild(xTitle);

  const yTitle = el("text", {{
    x: "16", y: (DH.T + DH.PLOT_H / 2).toFixed(2),
    "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 16 ${{(DH.T + DH.PLOT_H / 2).toFixed(2)}})`
  }});
  yTitle.textContent = "Count";
  svg.appendChild(yTitle);

  container.innerHTML = "";
  container.appendChild(svg);
}}

// Page-load initialisation
applyFilters();
updateHistogramYMax();
renderDetailHist(document.getElementById("detailHistSelect").value);
</script>
</body>
</html>
"""


# ── 11. WRITE OUTPUT ───────────────────────────────────────────────────────

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"Done! Report saved to: {OUTPUT_HTML}")
print(f"Rows in CSV: {total}")
print(f"Rows shown in image grid: {len(grid_rows)}")
print(f"Rows kept for plot analysis: {total}")
print(
    f"Embedded JPEG images: {saved_jpeg_images} "
    f"(top {MAX_JPEG_IMAGES} by score + all {num_interesting} interesting, minus overlap)"
)

if missing_images:
    print(f"Warning: {missing_images} images could not be loaded.")