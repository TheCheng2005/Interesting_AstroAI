"""
generate_report.py
------------------
Reads a comprehensive classification results CSV and writes a self-contained HTML report.
"""

import csv
import sys
import os
import json

# ── Paths & Variables ──────────────────────────────────────────────────────
CLASSIFICATION_CSV = sys.argv[1] if len(sys.argv) > 1 else "grid_results_score_3.1_low_4x4_fewshot_05-27_14.csv"
OUTPUT_HTML        = "report.html"

# ** IMPORTANT ** # Change this threshold to define what constitutes a "True" / "Positive"
# ground truth for your Precision-Recall calculation.
VOLUNTEER_RATING_THRESHOLD = 0.5

# ── 1. Load classification.csv ─────────────────────────────────────────────
rows = []
try:
    with open(CLASSIFICATION_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            filename = row.get("Filename", "").strip()
            if not filename:
                continue

            rows.append({
                "filename":      filename,
                "ai_score":      row.get("ImageScore", ""),
                "reasoning":     row.get("Reasoning", ""),
                "ra":            row.get("RA", "N/A"),
                "dec":           row.get("Dec", "N/A"),
                "url":           row.get("URL", ""),
                "anomaly_score": row.get("AnomalyScore", "N/A"),
                "volunteer_rating": row.get("volunteer_rating", "N/A")
            })
except FileNotFoundError:
    print(f"Error: Could not find '{CLASSIFICATION_CSV}'. Please provide a valid CSV file.")
    sys.exit(1)

not_found = [r["filename"] for r in rows if r["url"] == ""]
if not_found:
    print(f"WARNING: No URL found in CSV for {len(not_found)} file(s).")

# ── 2. Summary stats & Chart Data ──────────────────────────────────────────
total = len(rows)

def safe_avg(vals):
    nums = []
    for v in vals:
        try:
            nums.append(float(v))
        except (ValueError, TypeError):
            pass
    return f"{sum(nums)/len(nums):.2f}" if nums else "—"

avg_ai  = safe_avg([r["ai_score"]      for r in rows])
avg_ano = safe_avg([r["anomaly_score"] for r in rows])

# Prepare data for Chart.js scatter plots
ai_vs_vol = []
ai_vs_ano = []
y_true = []
y_scores = []

for r in rows:
    try:
        ai_v = float(r["ai_score"])
    except:
        ai_v = None

    try:
        vol_v = float(r["volunteer_rating"])
    except:
        vol_v = None

    try:
        ano_v = float(r["anomaly_score"])
    except:
        ano_v = None

    if ai_v is not None and vol_v is not None:
        ai_vs_vol.append({
            "x": ai_v,
            "y": vol_v,
            "url": r["url"],
            "filename": r["filename"]
        })

        # Extract binary ground truth and predictions for PR Curve
        y_true.append(1 if vol_v > VOLUNTEER_RATING_THRESHOLD else 0)
        y_scores.append(ai_v)

    if ai_v is not None and ano_v is not None:
        ai_vs_ano.append({
            "x": ai_v,
            "y": ano_v,
            "url": r["url"],
            "filename": r["filename"]
        })


# ── 3. Calculate Precision-Recall Data ─────────────────────────────────────
pr_data = []
pr_auc = 0.0
best_f1 = 0.0
optimal_threshold = 0.0

try:
    from sklearn.metrics import precision_recall_curve, auc
    import numpy as np

    if len(y_true) > 0 and sum(y_true) > 0:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        # F1 optimization
        with np.errstate(divide='ignore', invalid='ignore'):
            p_arr = np.array(precision[:-1])
            r_arr = np.array(recall[:-1])
            f1_scores = 2 * (p_arr * r_arr) / (p_arr + r_arr)
            f1_scores = np.nan_to_num(f1_scores)

            if len(f1_scores) > 0:
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
                best_f1 = f1_scores[optimal_idx]

        print("--- PR Curve Metrics ---")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"Optimal ImageScore Threshold (Best F1 = {best_f1:.2f}): {optimal_threshold:.2f}\n")

        # thresholds is 1 item shorter than precision and recall; pad with None
        thresholds_padded = list(thresholds) + [None]

        # Keep the natural threshold sequence returned by sklearn.
        for r, p, t in zip(recall, precision, thresholds_padded):
            pr_data.append({
                "x": float(r),
                "y": float(p),
                "threshold": float(t) if t is not None else None
            })

except ImportError:
    print("\n[Notice] 'scikit-learn' and 'numpy' are required to generate the Precision-Recall curve.")
    print("Run 'pip install scikit-learn numpy' to enable this feature in the HTML report.\n")


# Convert Python lists to JSON strings for HTML injection
ai_vs_vol_json = json.dumps(ai_vs_vol)
ai_vs_ano_json = json.dumps(ai_vs_ano)
pr_curve_json  = json.dumps(pr_data)


# ── 4. Score to colour helper ──────────────────────────────────────────────
def score_color(val, max_val=10):
    try:
        ratio = float(val) / float(max_val)
    except (ValueError, TypeError, ZeroDivisionError):
        return "#555"

    if ratio >= 0.7:
        return "#00e5a0"
    if ratio >= 0.4:
        return "#f0b429"
    return "#ff4d6d"


# ── 5. Build cards ─────────────────────────────────────────────────────────
cards = ""

for r in rows:
    ai_col  = score_color(r["ai_score"], max_val=50)
    ano_col = score_color(r["anomaly_score"])

    try:
        ai_pct = min(float(r["ai_score"]) / 50 * 100, 100)
    except:
        ai_pct = 0

    try:
        ano_pct = min(float(r["anomaly_score"]) / 10 * 100, 100)
    except:
        ano_pct = 0

    img_block = (
        f'<img src="{r["url"]}" alt="{r["filename"]}" loading="lazy">'
        if r["url"] else
        '<div class="no-img">Image URL not found</div>'
    )

    url_link = (
        f'<a class="src-link" href="{r["url"]}" target="_blank" rel="noopener">Open source image ↗</a>'
        if r["url"] else ""
    )

    cards += f"""
    <article class="card">
      <div class="img-wrap">{img_block}</div>
      <div class="card-body">
        <h2 class="fname" title="{r['filename']}">{r['filename']}</h2>
        <div class="coords-row">
          <div class="coord"><span class="clabel">RA</span><span class="cval">{r['ra']}</span></div>
          <div class="coord"><span class="clabel">Dec</span><span class="cval">{r['dec']}</span></div>
        </div>
        <div class="scores">
          <div class="score-item">
            <div class="score-header">
              <span class="slabel">AI Interest</span>
              <span class="snum" style="color:{ai_col}">{r['ai_score']}<span class="denom">/50</span></span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{ai_pct:.1f}%;background:{ai_col}"></div></div>
          </div>
          <div class="score-item">
            <div class="score-header">
              <span class="slabel">Anomaly Score</span>
              <span class="snum" style="color:{ano_col}">{r['anomaly_score']}</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{ano_pct:.1f}%;background:{ano_col}"></div></div>
          </div>
        </div>
        <p class="reasoning">{r['reasoning']}</p>
        {url_link}
      </div>
    </article>"""


# ── 6. Full HTML ───────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Astronomical Classification Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}

:root{{
  --bg:#040812;
  --surface:#0b1120;
  --border:rgba(255,255,255,0.08);
  --text:#b8c8e0;
  --text-dim:#4a5878;
  --accent:#00e5a0;
  --accent2:#7b78ff;
  --mono:'IBM Plex Mono',monospace;
  --display:'Syne',sans-serif;
}}

body{{
  background:var(--bg);
  color:var(--text);
  font-family:var(--mono);
  font-size:13px;
  line-height:1.7;
}}

body::before{{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(1px 1px at 12% 18%,rgba(255,255,255,.7) 0%,transparent 100%),
    radial-gradient(1px 1px at 28% 72%,rgba(255,255,255,.5) 0%,transparent 100%),
    radial-gradient(1.5px 1.5px at 44% 8%,rgba(255,255,255,.6) 0%,transparent 100%),
    radial-gradient(1px 1px at 67% 55%,rgba(255,255,255,.4) 0%,transparent 100%),
    radial-gradient(1px 1px at 83% 22%,rgba(255,255,255,.65) 0%,transparent 100%),
    radial-gradient(1.5px 1.5px at 91% 80%,rgba(123,120,255,.6) 0%,transparent 100%),
    radial-gradient(1px 1px at 55% 91%,rgba(0,229,160,.5) 0%,transparent 100%),
    radial-gradient(1px 1px at 6% 60%,rgba(255,255,255,.4) 0%,transparent 100%),
    radial-gradient(1px 1px at 75% 40%,rgba(255,255,255,.3) 0%,transparent 100%);
}}

.page{{position:relative;z-index:1;max-width:1140px;margin:0 auto;padding:60px 24px 100px}}

header{{margin-bottom:56px;border-bottom:1px solid var(--border);padding-bottom:36px}}
.eyebrow{{font-size:10px;letter-spacing:.25em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}}
header h1{{font-family:var(--display);font-size:clamp(30px,5vw,54px);font-weight:800;color:#fff;line-height:1.05;letter-spacing:-.02em}}
header h1 em{{font-style:normal;color:var(--accent2)}}
.subtitle{{margin-top:12px;color:var(--text-dim);font-size:12px}}

.stats{{display:flex;gap:1px;background:var(--border);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:32px}}
.stat{{flex:1;background:var(--surface);padding:18px 22px}}
.stat-label{{font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--text-dim);margin-bottom:4px}}
.stat-val{{font-family:var(--display);font-size:28px;font-weight:800;color:#fff}}
.stat-val.g{{color:var(--accent)}}
.stat-val.p{{color:var(--accent2)}}

.tab-controls {{ display:flex; gap:16px; margin-bottom:32px; border-bottom:1px solid var(--border); }}
.tab-btn {{ background:transparent; color:var(--text-dim); border:none; padding:12px 20px; font-family:var(--mono); font-size:14px; font-weight:700; cursor:pointer; text-transform:uppercase; letter-spacing:0.1em; transition:color 0.2s; position:relative; }}
.tab-btn.active {{ color:var(--accent); }}
.tab-btn.active::after {{ content:''; position:absolute; bottom:-1px; left:0; width:100%; height:2px; background:var(--accent); }}
.tab-btn:hover:not(.active) {{ color:#fff; }}
.tab-panel {{ display:none; }}
.tab-panel.active {{ display:block; }}

.grid{{display:grid;grid-template-columns:repeat(3, 1fr);gap:20px}}

.card{{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;overflow:hidden;
  display:flex;flex-direction:column;
  transition:transform .25s,box-shadow .25s,border-color .25s;
}}
.card:hover{{
  transform:translateY(-4px);
  box-shadow:0 24px 60px rgba(0,0,0,.7),0 0 0 1px rgba(123,120,255,.3);
  border-color:rgba(123,120,255,.3);
}}

.img-wrap{{width:100%;aspect-ratio:1 / 1;background:#060d1a;overflow:hidden;position:relative}}
.img-wrap img{{width:100%;height:100%;object-fit:contain;display:block;transition:transform .4s ease}}
.card:hover .img-wrap img{{transform:scale(1.03)}}

.no-img{{width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:var(--text-dim);font-size:11px;letter-spacing:.15em;text-transform:uppercase}}

.card-body{{padding:22px;display:flex;flex-direction:column;gap:14px;flex:1}}

.fname{{font-family:var(--mono);font-size:12px;font-weight:700;color:#fff;letter-spacing:.04em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}

.coords-row{{display:flex;gap:20px;padding:10px 14px;background:rgba(0,0,0,.3);border-radius:8px;border:1px solid var(--border)}}
.coord{{display:flex;flex-direction:column;gap:2px}}
.clabel{{font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:var(--text-dim)}}
.cval{{font-size:13px;font-weight:700;color:#fff}}

.scores{{display:flex;flex-direction:column;gap:10px}}
.score-item{{display:flex;flex-direction:column;gap:5px}}
.score-header{{display:flex;justify-content:space-between;align-items:baseline}}
.slabel{{font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--text-dim)}}
.snum{{font-family:var(--display);font-size:20px;font-weight:800}}
.denom{{font-size:11px;color:var(--text-dim);font-weight:400}}
.bar-track{{height:3px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:99px}}

.reasoning{{font-size:12px;color:var(--text);line-height:1.7;flex:1}}

.src-link{{
  display:inline-block;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent2);text-decoration:none;
  border:1px solid rgba(123,120,255,.3);border-radius:6px;padding:5px 12px;
  align-self:flex-start;transition:background .2s,color .2s;
}}
.src-link:hover{{background:rgba(123,120,255,.15);color:#fff}}

.chart-container {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 24px;
    position: relative;
    height: 450px;
    width: 100%;
}}

.axis-controls {{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:14px;
    padding:16px 18px;
    margin:0 0 12px 0;
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:16px;
}}

.axis-control-group {{
    display:flex;
    flex-direction:column;
    gap:8px;
}}

.axis-control-title {{
    display:flex;
    justify-content:space-between;
    gap:12px;
    font-size:10px;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:var(--text-dim);
}}

.axis-range-readout {{ color:#fff; letter-spacing:0; text-transform:none; }}

.slider-row {{
    display:grid;
    grid-template-columns:44px 1fr 70px;
    align-items:center;
    gap:8px;
    font-size:10px;
    color:var(--text);
}}

.slider-row input[type=range] {{
    width:100%;
    accent-color:var(--accent);
}}

.slider-row input[type=number] {{
    width:70px;
    background:rgba(255,255,255,0.04);
    color:#fff;
    border:1px solid var(--border);
    border-radius:6px;
    padding:4px 6px;
    font-family:var(--mono);
    font-size:10px;
}}

.reset-range-btn {{
    align-self:flex-start;
    background:rgba(255,255,255,0.04);
    color:var(--text);
    border:1px solid var(--border);
    border-radius:6px;
    padding:6px 10px;
    font-family:var(--mono);
    font-size:10px;
    letter-spacing:.08em;
    text-transform:uppercase;
    cursor:pointer;
}}

.reset-range-btn:hover {{
    background:rgba(0,229,160,0.12);
    color:#fff;
    border-color:rgba(0,229,160,0.4);
}}

@media(max-width:760px){{ .axis-controls{{grid-template-columns:1fr}} }}

.modal {{
  display:none; position:fixed; z-index:999; left:0; top:0;
  width:100%; height:100%; background-color:rgba(4,8,18,0.9);
  backdrop-filter:blur(4px); cursor:pointer;
}}
.modal-content {{
  margin:auto; display:block; max-width:90%; max-height:80vh;
  margin-top:5vh; border-radius:8px;
  box-shadow:0 24px 60px rgba(0,0,0,.7); border:1px solid var(--border);
}}
.modal-caption {{
  margin:auto; display:block; width:80%; text-align:center;
  color:#fff; padding:15px 0; font-family:var(--mono); font-size:14px;
}}
.close-modal {{
  position:absolute; top:20px; right:35px; color:var(--text-dim);
  font-size:40px; font-weight:bold; transition:color 0.2s;
}}
.close-modal:hover {{ color:var(--accent); }}

footer{{margin-top:72px;border-bottom:1px solid var(--border);padding-top:20px;color:var(--text-dim);font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:30px}}

@media(max-width:900px){{ .grid{{grid-template-columns:repeat(2, 1fr)}} }}
@media(max-width:620px){{
  .grid{{grid-template-columns:1fr}}
  .stats{{flex-direction:column;gap:1px}}
}}
</style>
</head>
<body>
<div class="page">

  <header>
    <p class="eyebrow">Gemini Vision Pipeline · Classification Report</p>
    <h1>Astronomical Image <em>Analysis</em></h1>
    <p class="subtitle">AI interest scoring · anomaly detection · source coordinates</p>
  </header>

  <div class="stats">
    <div class="stat"><div class="stat-label">Images</div><div class="stat-val">{total}</div></div>
    <div class="stat"><div class="stat-label">Avg AI Score</div><div class="stat-val g">{avg_ai}</div></div>
    <div class="stat"><div class="stat-label">Avg Anomaly Score</div><div class="stat-val p">{avg_ano}</div></div>
  </div>

  <div class="tab-controls">
    <button class="tab-btn active" onclick="switchTab(event, 'grid-view')">Image Grid</button>
    <button class="tab-btn" onclick="switchTab(event, 'plots-view')">Data Plots</button>
  </div>

  <div id="grid-view" class="tab-panel active">
      <div class="grid">
        {cards}
      </div>
  </div>

  <div id="plots-view" class="tab-panel">
      <div class="chart-container" id="prChartContainer" style="display:none;">
          <canvas id="chartPR"></canvas>
      </div>

      <div id="chartVolControls" class="axis-controls"></div>
      <div class="chart-container">
          <canvas id="chartVol"></canvas>
      </div>

      <div id="chartAnoControls" class="axis-controls"></div>
      <div class="chart-container">
          <canvas id="chartAno"></canvas>
      </div>
  </div>

  <footer>Generated from {os.path.basename(CLASSIFICATION_CSV)} · Astronomical Vision Pipeline</footer>
</div>

<div id="imageModal" class="modal" onclick="this.style.display='none'">
  <span class="close-modal">&times;</span>
  <img class="modal-content" id="modalImg" src="">
  <div class="modal-caption" id="modalCaption"></div>
</div>

<script>
function switchTab(evt, tabName) {{
    var i, tabpanels, tabbtns;

    tabpanels = document.getElementsByClassName("tab-panel");
    for (i = 0; i < tabpanels.length; i++) {{
        tabpanels[i].style.display = "none";
        tabpanels[i].classList.remove("active");
    }}

    tabbtns = document.getElementsByClassName("tab-btn");
    for (i = 0; i < tabbtns.length; i++) {{
        tabbtns[i].classList.remove("active");
    }}

    document.getElementById(tabName).style.display = "block";
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}}

function linearRegression(pts) {{
    const n = pts.length;
    if (n < 2) return null;

    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    let minX = Infinity, maxX = -Infinity;

    for (const p of pts) {{
        sumX  += p.x;
        sumY  += p.y;
        sumXY += p.x * p.y;
        sumX2 += p.x * p.x;

        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
    }}

    const denom = n * sumX2 - sumX * sumX;
    if (denom === 0) return null;

    const slope = (n * sumXY - sumX * sumY) / denom;
    const intercept = (sumY - slope * sumX) / n;

    const yMean = sumY / n;
    let ssTot = 0;
    let ssRes = 0;

    for (const p of pts) {{
        ssTot += (p.y - yMean) ** 2;
        ssRes += (p.y - (slope * p.x + intercept)) ** 2;
    }}

    const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

    return {{ slope, intercept, r2, minX, maxX }};
}}

function fitLineDataset(reg, color) {{
    if (!reg) return null;

    return {{
        type: 'line',
        label: 'Best Fit',
        data: [
            {{ x: reg.minX, y: reg.slope * reg.minX + reg.intercept }},
            {{ x: reg.maxX, y: reg.slope * reg.maxX + reg.intercept }}
        ],
        borderColor: color,
        borderWidth: 2,
        borderDash: [6, 4],
        pointRadius: 0,
        fill: false,
        tension: 0
    }};
}}

const r2LabelPlugin = {{
    id: 'r2Label',
    afterDraw(chart) {{
        const r2 = chart.config.options.r2Value;
        if (r2 == null) return;

        const {{ ctx, chartArea: {{ right, top }} }} = chart;

        ctx.save();
        ctx.font = "bold 13px 'IBM Plex Mono', monospace";
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.textAlign = 'right';
        ctx.fillText(`R² = ${{r2.toFixed(4)}}`, right - 8, top + 20);
        ctx.restore();
    }}
}};

Chart.register(r2LabelPlugin);

Chart.defaults.color = '#4a5878';
Chart.defaults.font.family = "'IBM Plex Mono', monospace";

function makeChart(canvasId, rawData, dotColor, fitColor, yLabel, title) {{
    const reg = linearRegression(rawData);
    const fitLine = fitLineDataset(reg, fitColor);

    const datasets = [
        {{
            type: 'scatter',
            label: 'Images',
            data: rawData,
            backgroundColor: dotColor,
            borderColor: dotColor,
            pointRadius: 4,
            pointHoverRadius: 6,
            order: 2
        }}
    ];

    if (fitLine) {{
        fitLine.order = 1;
        datasets.push(fitLine);
    }}

    return new Chart(document.getElementById(canvasId).getContext('2d'), {{
        type: 'scatter',
        data: {{ datasets }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            r2Value: reg ? reg.r2 : null,
            onClick: (e, elements, chart) => {{
                if (elements.length > 0) {{
                    const datasetIndex = elements[0].datasetIndex;
                    const dataIndex = elements[0].index;
                    const dataPoint = chart.data.datasets[datasetIndex].data[dataIndex];

                    if (dataPoint && dataPoint.url) {{
                        const modal = document.getElementById("imageModal");
                        const modalImg = document.getElementById("modalImg");
                        const captionText = document.getElementById("modalCaption");

                        modal.style.display = "block";
                        modalImg.src = dataPoint.url;
                        captionText.innerHTML = dataPoint.filename;
                    }}
                }}
            }},
            plugins: {{
                legend: {{ display: false }},
                title: {{
                    display: true,
                    text: title,
                    color: '#fff',
                    font: {{ size: 16 }}
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            let label = context.raw.filename || '';
                            if (label) {{
                                label += ': ';
                            }}
                            label += `(${{context.parsed.x}}, ${{context.parsed.y}})`;
                            return label;
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{
                    type: 'linear',
                    grid: {{ color: 'rgba(255,255,255,0.05)' }},
                    title: {{
                        display: true,
                        text: 'AI Interest Score',
                        color: '#00e5a0',
                        font: {{ weight: 'bold' }}
                    }}
                }},
                y: {{
                    type: 'linear',
                    grid: {{ color: 'rgba(255,255,255,0.05)' }},
                    title: {{
                        display: true,
                        text: yLabel,
                        color: '#b8c8e0',
                        font: {{ weight: 'bold' }}
                    }}
                }}
            }}
        }}
    }});
}}

function getAxisLimits(data, axis, fallbackMin, fallbackMax) {{
    const vals = data
        .map(p => Number(p[axis]))
        .filter(v => Number.isFinite(v));

    if (vals.length === 0) {{
        return {{ min: fallbackMin, max: fallbackMax }};
    }}

    const dataMin = Math.min(...vals);
    const dataMax = Math.max(...vals);
    const span = dataMax - dataMin || 1;
    const pad = span * 0.04;

    return {{
        min: Math.floor((dataMin - pad) * 100) / 100,
        max: Math.ceil((dataMax + pad) * 100) / 100
    }};
}}

function niceStep(min, max) {{
    const span = Math.abs(max - min) || 1;

    if (span <= 1.5) return 0.01;
    if (span <= 15) return 0.1;
    return 1;
}}

function formatAxisValue(value, step) {{
    return Number(value).toFixed(step < 1 ? 2 : 0);
}}

function createAxisControl(containerId, chart, data, title, yLabel) {{
    const container = document.getElementById(containerId);

    const xLimits = getAxisLimits(data, 'x', 0, 50);
    const yLimits = getAxisLimits(data, 'y', 0, 10);

    const axes = [
        {{
            key: 'x',
            label: 'AI Interest Score',
            min: xLimits.min,
            max: xLimits.max,
            step: niceStep(xLimits.min, xLimits.max)
        }},
        {{
            key: 'y',
            label: yLabel,
            min: yLimits.min,
            max: yLimits.max,
            step: niceStep(yLimits.min, yLimits.max)
        }}
    ];

    container.innerHTML = axes.map(axis => `
        <div class="axis-control-group" data-axis="${{axis.key}}">
            <div class="axis-control-title">
                <span>${{title}} · ${{axis.label}}</span>
                <span class="axis-range-readout" id="${{containerId}}-${{axis.key}}-readout"></span>
            </div>

            <div class="slider-row">
                <span>Lower</span>
                <input
                    id="${{containerId}}-${{axis.key}}-lower"
                    type="range"
                    min="${{axis.min}}"
                    max="${{axis.max}}"
                    step="${{axis.step}}"
                    value="${{axis.min}}"
                >
                <input
                    id="${{containerId}}-${{axis.key}}-lower-num"
                    type="number"
                    min="${{axis.min}}"
                    max="${{axis.max}}"
                    step="${{axis.step}}"
                    value="${{axis.min}}"
                >
            </div>

            <div class="slider-row">
                <span>Upper</span>
                <input
                    id="${{containerId}}-${{axis.key}}-upper"
                    type="range"
                    min="${{axis.min}}"
                    max="${{axis.max}}"
                    step="${{axis.step}}"
                    value="${{axis.max}}"
                >
                <input
                    id="${{containerId}}-${{axis.key}}-upper-num"
                    type="number"
                    min="${{axis.min}}"
                    max="${{axis.max}}"
                    step="${{axis.step}}"
                    value="${{axis.max}}"
                >
            </div>

            <button class="reset-range-btn" id="${{containerId}}-${{axis.key}}-reset">
                Reset ${{axis.key.toUpperCase()}}
            </button>
        </div>
    `).join('');

    function syncAxis(axis) {{
        const lower = document.getElementById(`${{containerId}}-${{axis.key}}-lower`);
        const upper = document.getElementById(`${{containerId}}-${{axis.key}}-upper`);
        const lowerNum = document.getElementById(`${{containerId}}-${{axis.key}}-lower-num`);
        const upperNum = document.getElementById(`${{containerId}}-${{axis.key}}-upper-num`);
        const readout = document.getElementById(`${{containerId}}-${{axis.key}}-readout`);

        let lo = Number(lower.value);
        let hi = Number(upper.value);

        if (!Number.isFinite(lo)) lo = axis.min;
        if (!Number.isFinite(hi)) hi = axis.max;

        if (lo >= hi) {{
            if (document.activeElement === lower || document.activeElement === lowerNum) {{
                lo = hi - axis.step;
            }} else {{
                hi = lo + axis.step;
            }}
        }}

        lo = Math.max(axis.min, Math.min(lo, axis.max - axis.step));
        hi = Math.min(axis.max, Math.max(hi, axis.min + axis.step));

        lower.value = lo;
        upper.value = hi;
        lowerNum.value = formatAxisValue(lo, axis.step);
        upperNum.value = formatAxisValue(hi, axis.step);

        readout.textContent = `${{formatAxisValue(lo, axis.step)}} → ${{formatAxisValue(hi, axis.step)}}`;

        chart.options.scales[axis.key].min = lo;
        chart.options.scales[axis.key].max = hi;
        chart.update('none');
    }}

    axes.forEach(axis => {{
        const lower = document.getElementById(`${{containerId}}-${{axis.key}}-lower`);
        const upper = document.getElementById(`${{containerId}}-${{axis.key}}-upper`);
        const lowerNum = document.getElementById(`${{containerId}}-${{axis.key}}-lower-num`);
        const upperNum = document.getElementById(`${{containerId}}-${{axis.key}}-upper-num`);
        const reset = document.getElementById(`${{containerId}}-${{axis.key}}-reset`);

        lower.addEventListener('input', () => {{
            lowerNum.value = lower.value;
            syncAxis(axis);
        }});

        upper.addEventListener('input', () => {{
            upperNum.value = upper.value;
            syncAxis(axis);
        }});

        lowerNum.addEventListener('input', () => {{
            lower.value = lowerNum.value;
            syncAxis(axis);
        }});

        upperNum.addEventListener('input', () => {{
            upper.value = upperNum.value;
            syncAxis(axis);
        }});

        reset.addEventListener('click', () => {{
            lower.value = axis.min;
            upper.value = axis.max;
            lowerNum.value = axis.min;
            upperNum.value = axis.max;
            syncAxis(axis);
        }});

        syncAxis(axis);
    }});
}}

const volData = {ai_vs_vol_json};
const anoData = {ai_vs_ano_json};
const prData = {pr_curve_json};

// More opaque than before.
// Change 0.55 higher if you want even more solid points, e.g. 0.70.
const chartVolObj = makeChart(
    'chartVol',
    volData,
    'rgba(123, 120, 255, 0.55)',
    '#c0bcff',
    'Volunteer Rating',
    'AI Interest Score vs. Volunteer Rating'
);

const chartAnoObj = makeChart(
    'chartAno',
    anoData,
    'rgba(0, 229, 160, 0.55)',
    '#80ffda',
    'Anomaly Score',
    'AI Interest Score vs. Anomaly Score'
);

createAxisControl('chartVolControls', chartVolObj, volData, 'Volunteer', 'Volunteer Rating');
createAxisControl('chartAnoControls', chartAnoObj, anoData, 'Anomaly', 'Anomaly Score');

// ── Precision Recall Curve Execution ─────────────────────────────────────────
if (prData && prData.length > 0) {{
    document.getElementById('prChartContainer').style.display = 'block';

    const thresholdLabelPlugin = {{
        id: 'thresholdLabelPlugin',
        afterDatasetsDraw(chart) {{
            const ctx = chart.ctx;

            chart.data.datasets.forEach((dataset, i) => {{
                const meta = chart.getDatasetMeta(i);

                meta.data.forEach((element, index) => {{
                    const dataPoint = dataset.data[index];

                    if (dataPoint.threshold !== null) {{
                        ctx.fillStyle = 'rgba(184, 200, 224, 0.9)';
                        ctx.font = '10px "IBM Plex Mono"';
                        ctx.textAlign = 'center';
                        ctx.fillText(dataPoint.threshold.toFixed(2), element.x, element.y - 12);
                    }}
                }});
            }});
        }}
    }};

    new Chart(document.getElementById('chartPR').getContext('2d'), {{
        type: 'scatter',
        data: {{
            datasets: [{{
                label: 'PR Curve',
                data: prData,
                borderColor: '#00e5a0',
                backgroundColor: 'rgba(0, 229, 160, 0.1)',
                fill: true,
                showLine: true,
                tension: 0,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: '#00e5a0'
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }},
                title: {{
                    display: true,
                    text: `Precision-Recall Curve (AUC: {pr_auc:.4f} | Optimal Threshold: {optimal_threshold:.2f})`,
                    color: '#fff',
                    font: {{ size: 16 }}
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            const pt = context.raw;

                            let lines = [
                                `Precision: ${{pt.y.toFixed(3)}}`,
                                `Recall: ${{pt.x.toFixed(3)}}`
                            ];

                            if (pt.threshold !== null) {{
                                lines.push(`Threshold: ${{pt.threshold.toFixed(2)}}`);
                            }}

                            return lines;
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{
                    type: 'linear',
                    grid: {{ color: 'rgba(255,255,255,0.05)' }},
                    title: {{
                        display: true,
                        text: 'Recall',
                        color: '#b8c8e0',
                        font: {{ weight: 'bold' }}
                    }},
                    min: -0.01,
                    max: 1.05
                }},
                y: {{
                    type: 'linear',
                    grid: {{ color: 'rgba(255,255,255,0.05)' }},
                    title: {{
                        display: true,
                        text: 'Precision',
                        color: '#b8c8e0',
                        font: {{ weight: 'bold' }}
                    }},
                    min: 0,
                    max: 1.05
                }}
            }}
        }},
        plugins: [thresholdLabelPlugin]
    }});
}}
</script>
</body>
</html>"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Done! Report saved to: {OUTPUT_HTML}  ({total} cards)")