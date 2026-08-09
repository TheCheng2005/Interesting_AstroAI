"""
generate_classification_grid_results_html.py
--------------------------------------------
Reads grid_results.csv plus any number of expert CSV files and writes a
self-contained HTML report comparing continuous AI ImageScore values against
expert selection rates.

Expected grid_results.csv columns:
    Filename, ImageScore, Reasoning, RA, Dec, AnomalyScore, URL, volunteer_rating

Expected expert CSV columns:
    name, url, interesting

Usage:
    python generate_classification_grid_results_html.py
    python generate_classification_grid_results_html.py grid_results.csv
    python generate_classification_grid_results_html.py grid_results.csv ./expert_csvs

Definitions:
    expert_score = (# experts who selected image) / (# experts who rated image)

Scatter plot:
    x-axis = AI Score / ImageScore
    y-axis = Expert Score
    includes linear fit and R^2

Precision-recall plot:
    ground truth positive = expert_score >= 0.4
    predicted positive = ImageScore >= AI threshold
"""

import csv
import sys
import os
import json
import glob
import html
from collections import defaultdict


AI_CSV = sys.argv[1] if len(sys.argv) > 1 else "tournament_no_shot_990.csv"
EXPERT_DIR = sys.argv[2] if len(sys.argv) > 2 else "."
OUTPUT_HTML = "expert_ai_agreement_report.html"

EXPERT_FILE_PATTERN = "expert_csvs/expert_scores_*.csv"
EXPERT_THRESHOLD = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_filename(value):
    return os.path.basename((value or "").strip())


def parse_float(value, default=None):
    try:
        if value is None:
            return default

        text = str(value).strip()

        if text == "" or text.lower() in {"nan", "none", "null", "n/a", "na"}:
            return default

        return float(text)

    except (TypeError, ValueError):
        return default


def parse_binary(value):
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "yes", "y", "true", "interesting", "selected"} else 0


def get_first(row, *names, default=""):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]

    lowered = {str(k).lower(): v for k, v in row.items()}

    for name in names:
        key = str(name).lower()

        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]

    return default


def expert_name_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]

    if stem.startswith("expert_scores_"):
        stem = stem[len("expert_scores_"):]

    return stem or "Unknown"


def pct_text(value):
    return f"{100 * value:.1f}%"


def esc(value):
    return html.escape(str(value if value is not None else ""), quote=True)


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return precision, recall, f1


# ─────────────────────────────────────────────────────────────────────────────
# Load grid_results.csv
# ─────────────────────────────────────────────────────────────────────────────

ai_rows = {}

try:
    with open(AI_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = clean_filename(
                get_first(row, "Filename", "filename", "name")
            )

            if not filename:
                continue

            image_score = parse_float(
                get_first(row, "ImageScore", "image_score", default="0"),
                default=0.0,
            )

            ai_rows[filename] = {
                "filename": filename,
                "image_score": image_score,
                "reasoning": get_first(row, "Reasoning", "reasoning", default=""),
                "ra": get_first(row, "RA", "ra", default="N/A"),
                "dec": get_first(row, "Dec", "DEC", "dec", default="N/A"),
                "anomaly_score": get_first(
                    row,
                    "AnomalyScore",
                    "anomaly_score",
                    "Anomaly",
                    default="N/A",
                ),
                "url": get_first(row, "URL", "url", default=""),
                "volunteer_rating": get_first(
                    row,
                    "volunteer_rating",
                    "VolunteerRating",
                    "volunteer",
                    default="N/A",
                ),
            }

except FileNotFoundError:
    print(f"Error: Could not find AI CSV: {AI_CSV}")
    sys.exit(1)


if not ai_rows:
    print(f"Error: No valid rows found in AI CSV: {AI_CSV}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Load expert CSVs
# ─────────────────────────────────────────────────────────────────────────────

expert_pattern = os.path.join(EXPERT_DIR, EXPERT_FILE_PATTERN)
expert_files = sorted(glob.glob(expert_pattern))

if not expert_files:
    print(f"Error: No expert files found matching: {expert_pattern}")
    print("Expected files named like: expert_scores_Name.csv")
    sys.exit(1)


expert_names = []
expert_votes = defaultdict(dict)
expert_urls = {}

for path in expert_files:
    expert_name = expert_name_from_path(path)
    expert_names.append(expert_name)

    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                filename = clean_filename(
                    get_first(row, "name", "Filename", "filename")
                )

                if not filename:
                    continue

                vote = parse_binary(
                    get_first(row, "interesting", "selected", "vote", default="0")
                )

                expert_votes[filename][expert_name] = vote

                url = get_first(row, "url", "URL", default="")

                if url and filename not in expert_urls:
                    expert_urls[filename] = url

    except FileNotFoundError:
        print(f"Warning: could not read expert file: {path}")


expert_names = sorted(set(expert_names))
expert_count = len(expert_names)


# ─────────────────────────────────────────────────────────────────────────────
# Merge AI rows and expert votes
# ─────────────────────────────────────────────────────────────────────────────

all_filenames = sorted(set(ai_rows.keys()) | set(expert_votes.keys()))

rows = []
missing_from_ai = []
missing_from_experts = []

for filename in all_filenames:
    ai = ai_rows.get(filename)
    votes_by_name = expert_votes.get(filename, {})

    if ai is None:
        missing_from_ai.append(filename)

        ai = {
            "filename": filename,
            "image_score": 0.0,
            "reasoning": "Not present in grid_results.csv.",
            "ra": "N/A",
            "dec": "N/A",
            "anomaly_score": "N/A",
            "url": expert_urls.get(filename, ""),
            "volunteer_rating": "N/A",
        }

    if not votes_by_name:
        missing_from_experts.append(filename)

    observed_votes = [
        votes_by_name[name]
        for name in expert_names
        if name in votes_by_name
    ]

    expert_n = len(observed_votes)
    expert_selected_n = sum(observed_votes)
    expert_score = expert_selected_n / expert_n if expert_n else 0.0

    rows.append({
        **ai,
        "url": ai.get("url") or expert_urls.get(filename, ""),
        "expert_score": expert_score,
        "expert_score_text": pct_text(expert_score),
        "expert_n": expert_n,
        "expert_selected_n": expert_selected_n,
        "expert_positive": 1 if expert_score >= EXPERT_THRESHOLD else 0,
        "expert_votes": votes_by_name,
        "selected_by": [
            name for name in expert_names
            if votes_by_name.get(name, 0) == 1
        ],
    })


rows.sort(
    key=lambda r: (
        r["expert_score"],
        r["image_score"],
        r["expert_selected_n"],
    ),
    reverse=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────────────────────────────────────

total_images = len(rows)

valid_expert_rows = [r for r in rows if r["expert_n"] > 0]
valid_expert_total = len(valid_expert_rows)

avg_image_score = (
    sum(r["image_score"] for r in rows) / total_images
    if total_images
    else 0.0
)

avg_expert_score = (
    sum(r["expert_score"] for r in valid_expert_rows) / valid_expert_total
    if valid_expert_total
    else 0.0
)

expert_positive_total = sum(r["expert_positive"] for r in rows)

anomaly_values = [
    parse_float(r.get("anomaly_score"))
    for r in rows
]

anomaly_values = [
    v for v in anomaly_values
    if v is not None
]

avg_anomaly = (
    sum(anomaly_values) / len(anomaly_values)
    if anomaly_values
    else 0.0
)


# ─────────────────────────────────────────────────────────────────────────────
# Scatter plot data: AI Score vs Expert Score
# ─────────────────────────────────────────────────────────────────────────────

density_bins = defaultdict(int)
scatter_base = []

for r in rows:
    if r["expert_n"] <= 0:
        continue

    x = r["image_score"]
    y = r["expert_score"]

    x_bin = round(x / 1.0)
    y_bin = round(y / 0.02)

    density_bins[(x_bin, y_bin)] += 1
    scatter_base.append((r, x, y, x_bin, y_bin))


scatter_points = []

for r, x, y, x_bin, y_bin in scatter_base:
    density = density_bins[(x_bin, y_bin)]
    alpha = min(0.92, 0.16 + 0.10 * (density - 1))

    scatter_points.append({
        "x": x,
        "y": y,
        "url": r.get("url", ""),
        "filename": r["filename"],
        "imageScore": r["image_score"],
        "expertScore": r["expert_score_text"],
        "expertSelected": r["expert_selected_n"],
        "expertTotal": r["expert_n"],
        "density": density,
        "pointColor": f"rgba(0, 229, 160, {alpha:.3f})",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Linear fit and R^2
# ─────────────────────────────────────────────────────────────────────────────

fit_rows = [r for r in rows if r["expert_n"] > 0]
n_fit = len(fit_rows)

if n_fit >= 2:
    xs = [r["image_score"] for r in fit_rows]
    ys = [r["expert_score"] for r in fit_rows]

    x_mean = sum(xs) / n_fit
    y_mean = sum(ys) / n_fit

    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_yy = sum((y - y_mean) ** 2 for y in ys)

    if ss_xx > 0:
        fit_slope = (
            sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            / ss_xx
        )

        fit_intercept = y_mean - fit_slope * x_mean

        ss_res = sum(
            (y - (fit_slope * x + fit_intercept)) ** 2
            for x, y in zip(xs, ys)
        )

        r_squared = 1 - ss_res / ss_yy if ss_yy > 0 else 0.0

        x_min_fit = min(xs)
        x_max_fit = max(xs)

        fit_line_points = [
            {
                "x": x_min_fit,
                "y": fit_slope * x_min_fit + fit_intercept,
            },
            {
                "x": x_max_fit,
                "y": fit_slope * x_max_fit + fit_intercept,
            },
        ]

    else:
        fit_slope = 0.0
        fit_intercept = y_mean
        r_squared = 0.0
        fit_line_points = []

else:
    fit_slope = 0.0
    fit_intercept = 0.0
    r_squared = 0.0
    fit_line_points = []


# ─────────────────────────────────────────────────────────────────────────────
# Precision / recall / F1 data
# ─────────────────────────────────────────────────────────────────────────────
#
# Ground truth:
#     expert_positive = expert_score >= 0.4
#
# Prediction:
#     AI positive = ImageScore >= threshold
#
# We sweep over all observed ImageScore thresholds.

valid_pr_rows = [r for r in rows if r["expert_n"] > 0]

possible_thresholds = sorted({
    r["image_score"]
    for r in valid_pr_rows
}, reverse=True)

if possible_thresholds:
    possible_thresholds = [max(possible_thresholds) + 1] + possible_thresholds

pr_points = []

for threshold in possible_thresholds:
    tp = sum(
        1 for r in valid_pr_rows
        if r["image_score"] >= threshold
        and r["expert_score"] >= EXPERT_THRESHOLD
    )

    fp = sum(
        1 for r in valid_pr_rows
        if r["image_score"] >= threshold
        and r["expert_score"] < EXPERT_THRESHOLD
    )

    fn = sum(
        1 for r in valid_pr_rows
        if r["image_score"] < threshold
        and r["expert_score"] >= EXPERT_THRESHOLD
    )

    tn = sum(
        1 for r in valid_pr_rows
        if r["image_score"] < threshold
        and r["expert_score"] < EXPERT_THRESHOLD
    )

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    pr_points.append({
        "x": recall,
        "y": precision,
        "threshold": threshold,
        "thresholdLabel": f"{threshold:.2f}",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": tp + fp,
    })


best_metric = max(
    pr_points,
    key=lambda m: (
        m["f1"],
        m["precision"],
        m["recall"],
        -m["threshold"],
    ),
    default={
        "x": 0.0,
        "y": 0.0,
        "threshold": 0.0,
        "thresholdLabel": "0.00",
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "predicted_positive": 0,
    },
)


pr_points = sorted(pr_points, key=lambda p: (p["x"], p["y"]))

pr_auc = 0.0

for a, b in zip(pr_points, pr_points[1:]):
    pr_auc += abs(b["x"] - a["x"]) * (a["y"] + b["y"]) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Image cards
# ─────────────────────────────────────────────────────────────────────────────

def expert_color(score):
    if score >= EXPERT_THRESHOLD:
        return "#00e5a0"

    if score > 0:
        return "#f0b429"

    return "#ff4d6d"


def image_score_color(score):
    if score >= 35:
        return "#00e5a0"

    if score >= 20:
        return "#f0b429"

    return "#ff4d6d"


cards_html = ""

for r in rows:
    ai_col = image_score_color(r["image_score"])
    exp_col = expert_color(r["expert_score"])
    exp_pct = min(max(r["expert_score"] * 100, 0), 100)

    selected_by_text = ", ".join(r["selected_by"]) if r["selected_by"] else "None"
    selected_by_data = "|".join(r["selected_by"])

    img_block = (
        f'<img src="{esc(r["url"])}" alt="{esc(r["filename"])}" loading="lazy">'
        if r.get("url")
        else '<div class="no-img">Image URL not found</div>'
    )

    url_link = (
        f'<a class="src-link" href="{esc(r["url"])}" target="_blank" rel="noopener">Open source image ↗</a>'
        if r.get("url")
        else ""
    )

    anomaly_html = ""

    if r.get("anomaly_score", "N/A") != "N/A":
        anomaly_html = (
            '<div class="score-item">'
            '<div class="score-header">'
            '<span class="slabel">Anomaly Score</span>'
            f'<span class="snum" style="color:#7b78ff">{esc(r.get("anomaly_score", "N/A"))}</span>'
            '</div></div>'
        )

    volunteer_html = ""

    if r.get("volunteer_rating", "N/A") != "N/A":
        volunteer_html = (
            '<div class="score-item">'
            '<div class="score-header">'
            '<span class="slabel">Volunteer Rating</span>'
            f'<span class="snum" style="color:#b8c8e0">{esc(r.get("volunteer_rating", "N/A"))}</span>'
            '</div></div>'
        )

    cards_html += f"""
    <article class="card"
      data-filename="{esc(r['filename'])}"
      data-image-score="{r['image_score']:.6f}"
      data-expert-score="{r['expert_score']:.6f}"
      data-selected-by="{esc(selected_by_data)}">

      <div class="img-wrap">{img_block}</div>

      <div class="card-body">
        <h2 class="fname" title="{esc(r['filename'])}">{esc(r['filename'])}</h2>

        <div class="coords-row">
          <div class="coord">
            <span class="clabel">RA</span>
            <span class="cval">{esc(r.get('ra', 'N/A'))}</span>
          </div>

          <div class="coord">
            <span class="clabel">Dec</span>
            <span class="cval">{esc(r.get('dec', 'N/A'))}</span>
          </div>
        </div>

        <div class="scores">
          <div class="score-item">
            <div class="score-header">
              <span class="slabel">Expert Score</span>
              <span class="snum" style="color:{exp_col}">
                {r['expert_score_text']}
                <span class="denom"> · {r['expert_selected_n']}/{r['expert_n']}</span>
              </span>
            </div>

            <div class="bar-track">
              <div class="bar-fill" style="width:{exp_pct:.1f}%;background:{exp_col}"></div>
            </div>
          </div>

          <div class="score-item">
            <div class="score-header">
              <span class="slabel">AI Score</span>
              <span class="snum" style="color:{ai_col}">{r['image_score']:.0f}</span>
            </div>
          </div>

          {anomaly_html}
          {volunteer_html}
        </div>

        <div class="expert-list">
          <span>Selected by</span>
          <strong>{esc(selected_by_text)}</strong>
        </div>

        <p class="reasoning">{esc(r.get('reasoning', ''))}</p>
        {url_link}
      </div>
    </article>
    """


expert_buttons = ""

for name in expert_names:
    count = sum(1 for r in rows if name in r["selected_by"])

    expert_buttons += (
        f'<button class="expert-btn" data-expert="{esc(name)}" '
        f"onclick='filterByExpert({json_dumps(name)})'>"
        f'{esc(name)} <span>{count}</span></button>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON payloads
# ─────────────────────────────────────────────────────────────────────────────

scatter_points_json = json_dumps(scatter_points)
fit_line_points_json = json_dumps(fit_line_points)
pr_points_json = json_dumps(pr_points)
best_metric_json = json_dumps(best_metric)
expert_names_json = json_dumps(expert_names)


# ─────────────────────────────────────────────────────────────────────────────
# HTML document
# ─────────────────────────────────────────────────────────────────────────────

html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">

<title>Astronomical Classification Report</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
*,*::before,*::after{
  box-sizing:border-box;
  margin:0;
  padding:0;
}

:root{
  --bg:#040812;
  --surface:#0b1120;
  --border:rgba(255,255,255,0.08);
  --text:#b8c8e0;
  --text-dim:#4a5878;
  --accent:#00e5a0;
  --accent2:#7b78ff;
  --danger:#ff4d6d;
  --warn:#f0b429;
  --mono:'IBM Plex Mono',monospace;
  --display:'Syne',sans-serif;
}

body{
  background:var(--bg);
  color:var(--text);
  font-family:var(--mono);
  font-size:13px;
  line-height:1.7;
}

body::before{
  content:'';
  position:fixed;
  inset:0;
  z-index:0;
  pointer-events:none;
  background:
    radial-gradient(1px 1px at 12% 18%,rgba(255,255,255,.7) 0%,transparent 100%),
    radial-gradient(1px 1px at 28% 72%,rgba(255,255,255,.5) 0%,transparent 100%),
    radial-gradient(1.5px 1.5px at 44% 8%,rgba(255,255,255,.6) 0%,transparent 100%),
    radial-gradient(1px 1px at 67% 55%,rgba(255,255,255,.4) 0%,transparent 100%),
    radial-gradient(1px 1px at 83% 22%,rgba(255,255,255,.65) 0%,transparent 100%),
    radial-gradient(1.5px 1.5px at 91% 80%,rgba(123,120,255,.6) 0%,transparent 100%),
    radial-gradient(1px 1px at 55% 91%,rgba(0,229,160,.5) 0%,transparent 100%);
}

.page{
  position:relative;
  z-index:1;
  max-width:1280px;
  margin:0 auto;
  padding:60px 24px 100px;
}

header{
  margin-bottom:32px;
  border-bottom:1px solid var(--border);
  padding-bottom:36px;
}

.eyebrow{
  font-size:10px;
  letter-spacing:.25em;
  text-transform:uppercase;
  color:var(--accent);
  margin-bottom:10px;
}

header h1{
  font-family:var(--display);
  font-size:clamp(30px,5vw,54px);
  font-weight:800;
  color:#fff;
  line-height:1.05;
  letter-spacing:-.02em;
}

header h1 em{
  font-style:normal;
  color:var(--accent2);
}

.subtitle{
  margin-top:12px;
  color:var(--text-dim);
  font-size:12px;
}

.stats{
  display:flex;
  gap:1px;
  background:var(--border);
  border:1px solid var(--border);
  border-radius:12px;
  overflow:hidden;
  margin-bottom:32px;
}

.stat{
  flex:1;
  background:var(--surface);
  padding:18px 22px;
}

.stat-label{
  font-size:9px;
  letter-spacing:.2em;
  text-transform:uppercase;
  color:var(--text-dim);
  margin-bottom:4px;
}

.stat-val{
  font-family:var(--display);
  font-size:28px;
  font-weight:800;
  color:#fff;
}

.stat-val.g{
  color:var(--accent);
}

.stat-val.p{
  color:var(--accent2);
}

.expert-panel{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  padding:20px;
  margin-bottom:32px;
}

.panel-title{
  font-size:10px;
  letter-spacing:.2em;
  text-transform:uppercase;
  color:var(--text-dim);
  margin-bottom:14px;
}

.expert-controls{
  display:flex;
  flex-wrap:wrap;
  gap:10px;
}

.expert-btn,
.small-filter-btn{
  background:rgba(255,255,255,.04);
  color:var(--text);
  border:1px solid var(--border);
  border-radius:999px;
  padding:8px 13px;
  font-family:var(--mono);
  font-size:11px;
  line-height:1.2;
  cursor:pointer;
  transition:background .2s,border-color .2s,color .2s;
}

.expert-btn span,
.small-filter-btn span{
  color:var(--accent);
  margin-left:6px;
}

.expert-btn:hover,
.expert-btn.active,
.small-filter-btn:hover,
.small-filter-btn.active{
  background:rgba(0,229,160,.12);
  border-color:rgba(0,229,160,.45);
  color:#fff;
}

.tab-controls{
  display:flex;
  gap:16px;
  margin-bottom:28px;
  border-bottom:1px solid var(--border);
}

.tab-btn{
  background:transparent;
  color:var(--text-dim);
  border:none;
  padding:12px 20px;
  font-family:var(--mono);
  font-size:14px;
  font-weight:700;
  cursor:pointer;
  text-transform:uppercase;
  letter-spacing:.1em;
  transition:color .2s;
  position:relative;
}

.tab-btn.active{
  color:var(--accent);
}

.tab-btn.active::after{
  content:'';
  position:absolute;
  bottom:-1px;
  left:0;
  width:100%;
  height:2px;
  background:var(--accent);
}

.tab-btn:hover:not(.active){
  color:#fff;
}

.tab-panel{
  display:none;
}

.tab-panel.active{
  display:block;
}

.status-line{
  margin-bottom:18px;
  color:var(--text-dim);
  font-size:11px;
  letter-spacing:.08em;
  text-transform:uppercase;
}

.grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:20px;
}

.card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  overflow:hidden;
  display:flex;
  flex-direction:column;
  transition:transform .25s,box-shadow .25s,border-color .25s;
}

.card.hidden{
  display:none;
}

.card:hover{
  transform:translateY(-4px);
  box-shadow:0 24px 60px rgba(0,0,0,.7),0 0 0 1px rgba(123,120,255,.3);
  border-color:rgba(123,120,255,.3);
}

.img-wrap{
  width:100%;
  aspect-ratio:1/1;
  background:#060d1a;
  overflow:hidden;
  position:relative;
  cursor:pointer;
}

.img-wrap img{
  width:100%;
  height:100%;
  object-fit:contain;
  display:block;
  transition:transform .4s ease;
}

.card:hover .img-wrap img{
  transform:scale(1.03);
}

.no-img{
  width:100%;
  height:100%;
  display:flex;
  align-items:center;
  justify-content:center;
  color:var(--text-dim);
  font-size:11px;
  letter-spacing:.15em;
  text-transform:uppercase;
}

.card-body{
  padding:22px;
  display:flex;
  flex-direction:column;
  gap:14px;
  flex:1;
}

.fname{
  font-family:var(--mono);
  font-size:12px;
  font-weight:700;
  color:#fff;
  letter-spacing:.04em;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}

.coords-row{
  display:flex;
  gap:20px;
  padding:10px 14px;
  background:rgba(0,0,0,.3);
  border-radius:8px;
  border:1px solid var(--border);
}

.coord{
  display:flex;
  flex-direction:column;
  gap:2px;
}

.clabel{
  font-size:9px;
  letter-spacing:.18em;
  text-transform:uppercase;
  color:var(--text-dim);
}

.cval{
  font-size:13px;
  font-weight:700;
  color:#fff;
}

.scores{
  display:flex;
  flex-direction:column;
  gap:10px;
}

.score-item{
  display:flex;
  flex-direction:column;
  gap:5px;
}

.score-header{
  display:flex;
  justify-content:space-between;
  align-items:baseline;
  gap:12px;
}

.slabel{
  font-size:10px;
  letter-spacing:.15em;
  text-transform:uppercase;
  color:var(--text-dim);
}

.snum{
  font-family:var(--display);
  font-size:20px;
  font-weight:800;
}

.denom{
  font-size:11px;
  color:var(--text-dim);
  font-weight:400;
}

.bar-track{
  height:3px;
  background:rgba(255,255,255,.07);
  border-radius:99px;
  overflow:hidden;
}

.bar-fill{
  height:100%;
  border-radius:99px;
}

.expert-list{
  background:rgba(255,255,255,.035);
  border:1px solid var(--border);
  border-radius:8px;
  padding:9px 11px;
}

.expert-list span{
  display:block;
  font-size:9px;
  letter-spacing:.16em;
  text-transform:uppercase;
  color:var(--text-dim);
}

.expert-list strong{
  display:block;
  font-size:11px;
  color:#fff;
  font-weight:700;
  margin-top:2px;
}

.reasoning{
  font-size:12px;
  color:var(--text);
  line-height:1.7;
  flex:1;
}

.src-link{
  display:inline-block;
  font-size:10px;
  letter-spacing:.1em;
  text-transform:uppercase;
  color:var(--accent2);
  text-decoration:none;
  border:1px solid rgba(123,120,255,.3);
  border-radius:6px;
  padding:5px 12px;
  align-self:flex-start;
  transition:background .2s,color .2s;
}

.src-link:hover{
  background:rgba(123,120,255,.15);
  color:#fff;
}

.chart-container{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  padding:24px;
  margin-bottom:24px;
  position:relative;
  height:470px;
  width:100%;
}

.metric-summary{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  padding:16px 18px;
  margin-bottom:24px;
  color:var(--text);
  font-size:12px;
}

.metric-summary strong{
  color:#fff;
}

.metric-summary span{
  color:var(--accent);
  font-weight:700;
}

.modal{
  display:none;
  position:fixed;
  z-index:999;
  left:0;
  top:0;
  width:100%;
  height:100%;
  background-color:rgba(4,8,18,.9);
  backdrop-filter:blur(4px);
  cursor:pointer;
}

.modal-content{
  margin:auto;
  display:block;
  max-width:90%;
  max-height:80vh;
  margin-top:5vh;
  border-radius:8px;
  box-shadow:0 24px 60px rgba(0,0,0,.7);
  border:1px solid var(--border);
}

.modal-caption{
  margin:auto;
  display:block;
  width:80%;
  text-align:center;
  color:#fff;
  padding:15px 0;
  font-family:var(--mono);
  font-size:14px;
}

.close-modal{
  position:absolute;
  top:20px;
  right:35px;
  color:var(--text-dim);
  font-size:40px;
  font-weight:bold;
  transition:color .2s;
}

.close-modal:hover{
  color:var(--accent);
}

footer{
  margin-top:72px;
  border-top:1px solid var(--border);
  padding-top:20px;
  color:var(--text-dim);
  font-size:10px;
  letter-spacing:.12em;
  text-transform:uppercase;
}

@media(max-width:900px){
  .grid{
    grid-template-columns:repeat(2,1fr);
  }
}

@media(max-width:620px){
  .grid{
    grid-template-columns:1fr;
  }

  .stats{
    flex-direction:column;
    gap:1px;
  }

  .tab-controls{
    overflow-x:auto;
  }
}
</style>
</head>

<body>
<div class="page">

  <header>
    <p class="eyebrow">Gemini Vision Pipeline · Classification Report</p>
    <h1>Astronomical Image <em>Analysis</em></h1>
    <p class="subtitle">Continuous ImageScore · expert agreement diagnostics</p>
  </header>

  <div class="stats">
    <div class="stat">
      <div class="stat-label">Images</div>
      <div class="stat-val">__TOTAL_IMAGES__</div>
    </div>

    <div class="stat">
      <div class="stat-label">Avg AI Score</div>
      <div class="stat-val g">__AVG_IMAGE_SCORE__</div>
    </div>

    <div class="stat">
      <div class="stat-label">Avg Expert Score</div>
      <div class="stat-val p">__AVG_EXPERT_SCORE__</div>
    </div>

    <div class="stat">
      <div class="stat-label">Scatter R²</div>
      <div class="stat-val">__R_SQUARED__</div>
    </div>
  </div>

  <div class="expert-panel">
    <div class="panel-title">Filter image grid by expert selection</div>

    <div class="expert-controls">
      <button class="small-filter-btn active" id="allBtn" onclick="clearExpertFilter()">
        All images <span>__TOTAL_IMAGES__</span>
      </button>

      <button class="small-filter-btn" id="expertPositiveBtn" onclick="filterByExpertPositive()">
        Expert score ≥ __EXPERT_THRESHOLD__ <span>__EXPERT_POSITIVE_TOTAL__</span>
      </button>

      __EXPERT_BUTTONS__
    </div>
  </div>

  <div class="tab-controls">
    <button class="tab-btn active" onclick="switchTab(event,'grid-view')">Image Grid</button>
    <button class="tab-btn" onclick="switchTab(event,'plots-view')">Data Plots</button>
  </div>

  <div id="grid-view" class="tab-panel active">
    <div class="status-line" id="statusLine">
      Showing all __TOTAL_IMAGES__ images. Sorted by expert score, highest first.
    </div>

    <div class="grid" id="imageGrid">
      __CARDS_HTML__
    </div>
  </div>

  <div id="plots-view" class="tab-panel">
    <div class="metric-summary">
      <strong>Linear fit:</strong>
      expert score =
      <span>__FIT_SLOPE__</span> × AI score +
      <span>__FIT_INTERCEPT__</span>;
      R² = <span>__R_SQUARED_LONG__</span>
    </div>

    <div class="chart-container">
      <canvas id="chartScatter"></canvas>
    </div>

    <div class="metric-summary" id="bestMetricBox"></div>

    <div class="chart-container">
      <canvas id="chartPrecisionRecall"></canvas>
    </div>
  </div>

  <footer>
    Expert files: __EXPERT_FILES__<br>
    Missing from grid_results.csv: __MISSING_FROM_AI__ · Missing from expert CSVs: __MISSING_FROM_EXPERTS__
  </footer>

</div>

<div id="imgModal" class="modal" onclick="closeModal()">
  <span class="close-modal">&times;</span>
  <img class="modal-content" id="modalImg">
  <div class="modal-caption" id="modalCaption"></div>
</div>

<script>
const expertNames = __EXPERT_NAMES_JSON__;
const scatterPoints = __SCATTER_POINTS_JSON__;
const fitLinePoints = __FIT_LINE_POINTS_JSON__;
const prData = __PR_POINTS_JSON__;
const bestMetric = __BEST_METRIC_JSON__;
const prAUC = __PR_AUC__;
const expertThreshold = __EXPERT_THRESHOLD__;

Chart.defaults.color = '#b8c8e0';
Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';
Chart.defaults.font.family = 'IBM Plex Mono, monospace';


function switchTab(event, id) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

  event.currentTarget.classList.add('active');
  document.getElementById(id).classList.add('active');
}


function setActiveFilterButton(activeBtn) {
  document.querySelectorAll('.expert-controls button').forEach(btn => btn.classList.remove('active'));

  if (activeBtn) {
    activeBtn.classList.add('active');
  }
}


function updateVisibleCards(predicate, message) {
  let shown = 0;

  document.querySelectorAll('.card').forEach(card => {
    const show = predicate(card);
    card.classList.toggle('hidden', !show);

    if (show) {
      shown += 1;
    }
  });

  const status = document.getElementById('statusLine');

  if (status) {
    status.textContent = `${message} Showing ${shown} image(s).`;
  }
}


function clearExpertFilter() {
  setActiveFilterButton(document.getElementById('allBtn'));
  updateVisibleCards(() => true, 'Showing all images.');
}


function filterByExpertPositive() {
  setActiveFilterButton(document.getElementById('expertPositiveBtn'));
  updateVisibleCards(
    card => Number(card.dataset.expertScore) >= expertThreshold,
    'Expert-positive images.'
  );
}


function filterByExpert(name) {
  const btn = Array
    .from(document.querySelectorAll('.expert-btn'))
    .find(b => b.dataset.expert === name);

  setActiveFilterButton(btn);

  updateVisibleCards(
    card => card.dataset.selectedBy.split('|').includes(name),
    `Images selected by ${name}.`
  );
}


function openModal(url, caption) {
  if (!url) {
    return;
  }

  document.getElementById('imgModal').style.display = 'block';
  document.getElementById('modalImg').src = url;
  document.getElementById('modalCaption').textContent = caption || '';
}


function closeModal() {
  document.getElementById('imgModal').style.display = 'none';
}


document.querySelectorAll('.img-wrap img').forEach(img => {
  img.addEventListener('click', () => openModal(img.src, img.alt));
});


const maxAiScore = scatterPoints.length
  ? Math.max(50, Math.ceil(Math.max(...scatterPoints.map(p => Number(p.x) || 0)) / 5) * 5)
  : 50;


new Chart(
  document.getElementById('chartScatter').getContext('2d'),
  {
    type: 'scatter',

    data: {
      datasets: [
        {
          label: 'All images',
          data: scatterPoints,
          backgroundColor: scatterPoints.map(p => p.pointColor),
          borderColor: scatterPoints.map(p => p.pointColor),
          pointRadius: 3,
          pointHoverRadius: 6
        },
        {
          label: 'Linear fit, R² = __R_SQUARED__',
          data: fitLinePoints,
          type: 'line',
          borderColor: '#ff4d6d',
          backgroundColor: '#ff4d6d',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 0,
          fill: false,
          tension: 0
        }
      ]
    },

    options: {
      responsive: true,
      maintainAspectRatio: false,

      onClick: (e, elements, chart) => {
        if (elements.length > 0) {
          const datasetIndex = elements[0].datasetIndex;
          const dataIndex = elements[0].index;
          const dataPoint = chart.data.datasets[datasetIndex].data[dataIndex];

          if (dataPoint.url) {
            openModal(dataPoint.url, dataPoint.filename);
          }
        }
      },

      plugins: {
        legend: {
          display: true,
          labels: {
            color: '#b8c8e0'
          }
        },

        title: {
          display: true,
          text: 'Expert Score vs. AI Score',
          color: '#fff',
          font: {
            size: 16,
            weight: 'bold'
          }
        },

        subtitle: {
          display: true,
          text: 'AI Score is the x-axis. Expert score is the y-axis. Linear fit R² = __R_SQUARED__.',
          color: '#4a5878',
          font: {
            size: 11
          }
        },

        tooltip: {
          callbacks: {
            label: function(context) {
              const p = context.raw;

              if (!p.filename) {
                return [
                  `Linear fit`,
                  `AI Score: ${Number(p.x).toFixed(2)}`,
                  `Predicted expert score: ${Number(p.y).toFixed(3)}`
                ];
              }

              return [
                p.filename,
                `AI Score: ${Number(p.x).toFixed(0)}`,
                `Expert score: ${p.expertScore} (${p.expertSelected}/${p.expertTotal})`,
                `Local density bin: ${p.density} point(s)`
              ];
            }
          }
        }
      },

      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: maxAiScore,
          grid: {
            color: 'rgba(255,255,255,0.05)'
          },
          title: {
            display: true,
            text: 'AI Score',
            color: '#b8c8e0',
            font: {
              weight: 'bold'
            }
          }
        },

        y: {
          type: 'linear',
          min: -0.04,
          max: 1.04,
          grid: {
            color: 'rgba(255,255,255,0.05)'
          },
          title: {
            display: true,
            text: 'Expert Score',
            color: '#b8c8e0',
            font: {
              weight: 'bold'
            }
          }
        }
      }
    }
  }
);


document.getElementById('bestMetricBox').innerHTML = `
  <strong>Best AI Score threshold:</strong>
  <span>&ge; ${bestMetric.threshold.toFixed(2)}</span>
  &nbsp;·&nbsp; Precision <span>${(100 * bestMetric.precision).toFixed(1)}%</span>
  &nbsp;·&nbsp; Recall <span>${(100 * bestMetric.recall).toFixed(1)}%</span>
  &nbsp;·&nbsp; F1 <span>${bestMetric.f1.toFixed(3)}</span>
  &nbsp;·&nbsp; PR AUC <span>${prAUC.toFixed(4)}</span>
  &nbsp;·&nbsp; TP/FP/FN/TN =
  ${bestMetric.tp}/${bestMetric.fp}/${bestMetric.fn}/${bestMetric.tn}
`;


const thresholdLabelPlugin = {
  id: 'thresholdLabelPlugin',

  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;

    chart.data.datasets.forEach((dataset, i) => {
      if (dataset.skipThresholdLabels) {
        return;
      }

      const meta = chart.getDatasetMeta(i);

      meta.data.forEach((element, index) => {
        const dataPoint = dataset.data[index];

        if (
          dataPoint &&
          dataPoint.threshold !== null &&
          Number.isFinite(dataPoint.threshold)
        ) {
          ctx.save();

          ctx.fillStyle = 'rgba(184, 200, 224, 0.90)';
          ctx.font = '10px "IBM Plex Mono"';
          ctx.textAlign = 'center';

          ctx.fillText(
            dataPoint.threshold.toFixed(1),
            element.x,
            element.y - 12
          );

          ctx.restore();
        }
      });
    });
  }
};


new Chart(
  document.getElementById('chartPrecisionRecall').getContext('2d'),
  {
    type: 'scatter',

    data: {
      datasets: [
        {
          label: 'Precision-Recall Curve',
          data: prData,
          borderColor: '#00e5a0',
          backgroundColor: 'rgba(0, 229, 160, 0.10)',
          fill: true,
          showLine: true,
          tension: 0,
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: '#00e5a0',
          pointBorderColor: '#00e5a0'
        },
        {
          label: 'Best F1 threshold',
          data: [bestMetric],
          borderColor: '#ff4d6d',
          backgroundColor: '#ff4d6d',
          pointRadius: 8,
          pointHoverRadius: 10,
          showLine: false,
          skipThresholdLabels: false
        }
      ]
    },

    options: {
      responsive: true,
      maintainAspectRatio: false,

      plugins: {
        legend: {
          display: true,
          labels: {
            color: '#b8c8e0'
          }
        },

        title: {
          display: true,
          text: `Precision-Recall Curve (AUC: ${prAUC.toFixed(4)} | Optimal AI Score Threshold: ${bestMetric.threshold.toFixed(2)})`,
          color: '#fff',
          font: {
            size: 16,
            weight: 'bold'
          }
        },

        subtitle: {
          display: true,
          text: `Ground truth positive = expert score ≥ ${expertThreshold}. Prediction positive = AI Score ≥ threshold.`,
          color: '#4a5878',
          font: {
            size: 11
          }
        },

        tooltip: {
          callbacks: {
            label: function(context) {
              const pt = context.raw;

              return [
                `Precision: ${pt.y.toFixed(3)}`,
                `Recall: ${pt.x.toFixed(3)}`,
                `F1: ${pt.f1.toFixed(3)}`,
                `AI Score threshold: ${pt.threshold.toFixed(2)}`,
                `TP/FP/FN/TN: ${pt.tp}/${pt.fp}/${pt.fn}/${pt.tn}`,
                `Predicted positive: ${pt.predicted_positive}`
              ];
            }
          }
        }
      },

      scales: {
        x: {
          type: 'linear',
          min: -0.01,
          max: 1.05,
          grid: {
            color: 'rgba(255,255,255,0.05)'
          },
          title: {
            display: true,
            text: 'Recall',
            color: '#b8c8e0',
            font: {
              weight: 'bold'
            }
          }
        },

        y: {
          type: 'linear',
          min: 0,
          max: 1.05,
          grid: {
            color: 'rgba(255,255,255,0.05)'
          },
          title: {
            display: true,
            text: 'Precision',
            color: '#b8c8e0',
            font: {
              weight: 'bold'
            }
          }
        }
      }
    },

    plugins: [thresholdLabelPlugin]
  }
);
</script>
</body>
</html>
"""


html_doc = html_template
html_doc = html_doc.replace("__TOTAL_IMAGES__", str(total_images))
html_doc = html_doc.replace("__AVG_IMAGE_SCORE__", f"{avg_image_score:.1f}")
html_doc = html_doc.replace("__AVG_EXPERT_SCORE__", f"{100 * avg_expert_score:.1f}%")
html_doc = html_doc.replace("__R_SQUARED__", f"{r_squared:.3f}")
html_doc = html_doc.replace("__R_SQUARED_LONG__", f"{r_squared:.6f}")
html_doc = html_doc.replace("__FIT_SLOPE__", f"{fit_slope:.6f}")
html_doc = html_doc.replace("__FIT_INTERCEPT__", f"{fit_intercept:.6f}")
html_doc = html_doc.replace("__EXPERT_THRESHOLD__", f"{EXPERT_THRESHOLD:.1f}")
html_doc = html_doc.replace("__EXPERT_POSITIVE_TOTAL__", str(expert_positive_total))
html_doc = html_doc.replace("__EXPERT_BUTTONS__", expert_buttons)
html_doc = html_doc.replace("__CARDS_HTML__", cards_html)
html_doc = html_doc.replace(
    "__EXPERT_FILES__",
    esc(", ".join(os.path.basename(p) for p in expert_files))
)
html_doc = html_doc.replace("__MISSING_FROM_AI__", str(len(missing_from_ai)))
html_doc = html_doc.replace("__MISSING_FROM_EXPERTS__", str(len(missing_from_experts)))
html_doc = html_doc.replace("__EXPERT_NAMES_JSON__", expert_names_json)
html_doc = html_doc.replace("__SCATTER_POINTS_JSON__", scatter_points_json)
html_doc = html_doc.replace("__FIT_LINE_POINTS_JSON__", fit_line_points_json)
html_doc = html_doc.replace("__PR_POINTS_JSON__", pr_points_json)
html_doc = html_doc.replace("__BEST_METRIC_JSON__", best_metric_json)
html_doc = html_doc.replace("__PR_AUC__", f"{pr_auc:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_doc)


print(f"Wrote {OUTPUT_HTML}")
print(f"Images: {total_images}")
print(f"Experts: {expert_count} ({', '.join(expert_names)})")
print(f"Expert-positive threshold: expert_score >= {EXPERT_THRESHOLD}")
print(f"Average AI Score: {avg_image_score:.3f}")
print(f"Average expert score: {avg_expert_score:.3f}")
print(f"Linear fit: expert_score = {fit_slope:.6f} * ImageScore + {fit_intercept:.6f}")
print(f"R^2: {r_squared:.6f}")
print(
    "Best AI threshold: "
    f"{best_metric['threshold']:.6f}, "
    f"precision={best_metric['precision']:.6f}, "
    f"recall={best_metric['recall']:.6f}, "
    f"F1={best_metric['f1']:.6f}, "
    f"PR AUC={pr_auc:.6f}"
)