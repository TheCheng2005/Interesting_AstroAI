"""
generate_unidentified_html_report.py

Generate a self-contained HTML report combining all three outputs of
find_unidentified_objects.py:
    - unidentified_objects.csv: images matched by none of HF/SIMBAD/NED
    - matched_objects.csv: images matched by HF, SIMBAD, or NED, enriched
      with object_name/object_type/ref_count (paper count)/redshift/
      arxiv_url/summary
    - simbad_bibliography.csv: full per-paper bibliography (bibcode/year/
      title) for every SIMBAD-matched image
    - ned_bibliography.csv: same, for every NED-matched image
    - deep_dive_summaries.json: AI-synthesized (Gemini reading ADS abstracts)
      literature summaries for the ~100 most-cited matched objects

Every scored image (matched or not) is shown in one browsable grid, so you
can compare "high Gemini score + no match" against "high Gemini score +
heavily-studied object" side by side.

Important:
- All images are loaded from the HDF5 file by filename lookup.
- Images are converted to JPEG and embedded directly in the HTML as base64
  data URLs (the top MAX_JPEG_IMAGES by avg_score across the COMBINED
  matched+unidentified set, default 2000) so the report is fully
  self-contained and viewable anywhere.
- SIMBAD-matched cards with a bibliography get a "View N papers" button that
  opens a modal listing every paper (year, title, ADS link) - built from an
  embedded per-card JSON blob, not a live lookup.
- There is no ground-truth "interesting" label here (that's the whole point:
  we're trying to find literature-unidentified images), so there is no PR
  curve or TP/FP/FN/TN breakdown - just the score distribution and a
  browsable, sortable/filterable image grid.

Usage:
    python generate_unidentified_html_report.py
    python generate_unidentified_html_report.py unidentified_objects.csv hdf5_file.hdf5
    python generate_unidentified_html_report.py unidentified_objects.csv hdf5_file.hdf5 output.html
    python generate_unidentified_html_report.py unidentified_objects.csv hdf5_file.hdf5 output.html 1000
    python generate_unidentified_html_report.py unidentified_objects.csv hdf5_file.hdf5 output.html 1000 2000 matched_objects.csv simbad_bibliography.csv
"""

import base64
import csv
import html
import io
import json as _json
import os
import sys
from pathlib import Path

import h5py
from PIL import Image


# ── 1. CONFIGURATION ───────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

UNIDENTIFIED_CSV = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(SCRIPT_DIR, "unidentified_objects.csv")
)

HDF5_PATH = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "../hubble_data/10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5"
)

OUTPUT_HTML = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "unidentified_objects_report.html"
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

MATCHED_CSV = (
    sys.argv[6]
    if len(sys.argv) > 6
    else os.path.join(SCRIPT_DIR, "matched_objects.csv")
)

SIMBAD_BIBLIOGRAPHY_CSV = (
    sys.argv[7]
    if len(sys.argv) > 7
    else os.path.join(SCRIPT_DIR, "simbad_bibliography.csv")
)

NED_BIBLIOGRAPHY_CSV = (
    sys.argv[8]
    if len(sys.argv) > 8
    else os.path.join(SCRIPT_DIR, "ned_bibliography.csv")
)

DEEP_DIVE_JSON = (
    sys.argv[9]
    if len(sys.argv) > 9
    else os.path.join(SCRIPT_DIR, "deep_dive_summaries.json")
)

DISCUSSION_CLASSIFICATION_CSV = (
    sys.argv[10]
    if len(sys.argv) > 10
    else os.path.join(SCRIPT_DIR, "discussion_classification.csv")
)


# ── 2. HELPERS ─────────────────────────────────────────────────────────────

def get_field(row, *names, default=""):
    lowered = {str(k).lower(): v for k, v in row.items()}

    for name in names:
        val = lowered.get(name.lower())
        if val is not None:
            return val

    return default


def safe_float(value, default=0.0):
    try:
        return float(str(value).strip())
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        return int(float(str(value).strip()))
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


# ── 3. IMAGE LOOKUP ─────────────────────────────────────────────────────────

def build_hdf5_filename_lookup(hdf5_path):
    """
    Build filename -> HDF5 row index lookup.
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


def find_in_lookup(filename, lookup):
    for key in normalize_filename_keys(filename):
        if key in lookup:
            return lookup[key]

    return None


# ── 4. READ CSVs ────────────────────────────────────────────────────────────

if not os.path.exists(UNIDENTIFIED_CSV):
    raise FileNotFoundError(f"Could not find unidentified CSV: {UNIDENTIFIED_CSV}")

rows = []

with open(UNIDENTIFIED_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        filename_raw = get_field(row, "filename", "Filename")

        if str(filename_raw).strip() == "":
            continue

        rows.append({
            "filename": str(filename_raw).strip(),
            "avg_score": safe_float(
                get_field(row, "avg_score", "AvgScore", "imagescore")
            ),
            "SourceRA": safe_float(
                get_field(row, "SourceRA", "source_ra"),
                default=0.0,
            ),
            "SourceDec": safe_float(
                get_field(row, "SourceDec", "source_dec"),
                default=0.0,
            ),
            "status": "unidentified",
            "checked_ned": int(safe_float(get_field(row, "checked_ned"), default=0.0)),
            "matched_source": "",
            "object_name": "",
            "object_type": "",
            "ref_count": None,
            "redshift": "",
            "arxiv_url": "",
            "summary": "",
        })

n_unidentified = len(rows)

if os.path.exists(MATCHED_CSV):
    with open(MATCHED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename_raw = get_field(row, "filename", "Filename")

            if str(filename_raw).strip() == "":
                continue

            rows.append({
                "filename": str(filename_raw).strip(),
                "avg_score": safe_float(
                    get_field(row, "avg_score", "AvgScore", "imagescore")
                ),
                "SourceRA": safe_float(
                    get_field(row, "SourceRA", "source_ra"),
                    default=0.0,
                ),
                "SourceDec": safe_float(
                    get_field(row, "SourceDec", "source_dec"),
                    default=0.0,
                ),
                "status": "matched",
                "checked_ned": 0,
                "matched_source": str(get_field(row, "matched_source")).strip(),
                "object_name": str(get_field(row, "object_name")).strip(),
                "object_type": str(get_field(row, "object_type")).strip(),
                "ref_count": safe_int(get_field(row, "ref_count"), default=None),
                "redshift": str(get_field(row, "redshift")).strip(),
                "arxiv_url": str(get_field(row, "arxiv_url")).strip(),
                "summary": str(get_field(row, "summary")).strip(),
            })
else:
    print(f"Warning: matched objects CSV not found ({MATCHED_CSV}); showing unidentified images only.")

if not rows:
    raise RuntimeError(f"No valid rows found in {UNIDENTIFIED_CSV} / {MATCHED_CSV}")

# A positional catalog match (SIMBAD/NED) alone isn't "identified" - if no
# paper actually discusses the object (see classify_genuine_discussion.py),
# it's reclassified as unidentified here, even though a catalog knows about
# it. HF matches are exempt: they're sourced from an actual paper-mention
# coordinate in the first place, so they're "discussed" by construction.
discussion_by_object = {}
if os.path.exists(DISCUSSION_CLASSIFICATION_CSV):
    with open(DISCUSSION_CLASSIFICATION_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            discussion_by_object[row["object_name"]] = {
                "genuinely_discussed": str(row["genuinely_discussed"]).strip().lower() == "true",
                "reasoning": row["reasoning"],
            }
    print(f"Loaded genuine-discussion classification for {len(discussion_by_object)} objects.")
else:
    print(f"Warning: discussion classification CSV not found ({DISCUSSION_CLASSIFICATION_CSV}); "
          f"all catalog matches will be counted as identified.")

n_reclassified = 0
for r in rows:
    if r["status"] != "matched" or r["matched_source"] == "HF":
        continue

    verdict = discussion_by_object.get(r["object_name"])
    if verdict is not None and not verdict["genuinely_discussed"]:
        r["status"] = "unidentified"
        r["not_discussed_reasoning"] = verdict["reasoning"]
        n_reclassified += 1

if n_reclassified:
    print(f"Reclassified {n_reclassified} catalog matches as unidentified (no paper genuinely discusses them).")

n_unidentified = sum(1 for r in rows if r["status"] == "unidentified")
n_matched = sum(1 for r in rows if r["status"] == "matched")

rows.sort(key=lambda r: r["avg_score"], reverse=True)

# Full per-paper bibliography, keyed by filename. SIMBAD and NED each expose
# a bibliography join (HF does not; HF matches get arxiv_url/summary
# instead). A given filename is only ever matched by one source, so the two
# dicts can be merged without collision.
bibliography_by_filename = {}


def _load_bibliography_csv(path, source_label):
    if not os.path.exists(path):
        print(f"Warning: {source_label} bibliography CSV not found ({path}); no paper links will be shown for {source_label} matches.")
        return 0

    n_before = len(bibliography_by_filename)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = str(get_field(row, "filename")).strip()

            if not filename:
                continue

            bibliography_by_filename.setdefault(filename, []).append({
                "bibcode": str(get_field(row, "bibcode")).strip(),
                "year": safe_int(get_field(row, "year"), default=None),
                "title": str(get_field(row, "title")).strip(),
            })

    return len(bibliography_by_filename) - n_before


n_simbad_biblio = _load_bibliography_csv(SIMBAD_BIBLIOGRAPHY_CSV, "SIMBAD")
n_ned_biblio = _load_bibliography_csv(NED_BIBLIOGRAPHY_CSV, "NED")

for filename, papers in bibliography_by_filename.items():
    papers.sort(key=lambda p: p["year"] or 0, reverse=True)

print(
    f"Loaded bibliography for {len(bibliography_by_filename)} matched images "
    f"({n_simbad_biblio} SIMBAD, {n_ned_biblio} NED)."
)

# Deep-dive literature summaries (top ~100 most-cited objects only, see
# deep_dive_summaries.py), keyed by object_name.
deep_dive_by_object = {}

if os.path.exists(DEEP_DIVE_JSON):
    with open(DEEP_DIVE_JSON, encoding="utf-8") as f:
        for entry in _json.load(f):
            object_name = str(entry.get("object_name", "")).strip()
            if object_name:
                deep_dive_by_object[object_name] = entry
    print(f"Loaded deep-dive literature summaries for {len(deep_dive_by_object)} objects.")
else:
    print(f"Warning: deep-dive summaries JSON not found ({DEEP_DIVE_JSON}); no literature summaries will be shown.")


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
            continue

        should_embed = rank < MAX_JPEG_IMAGES

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


# ── 6. SUMMARY STATS ────────────────────────────────────────────────────────

total = len(rows)
scores = [float(r["avg_score"]) for r in rows]

avg_score_overall = sum(scores) / total if total else 0.0
max_score = max(scores) if scores else 0.0
num_checked_ned = sum(1 for r in rows if r["status"] == "unidentified" and r["checked_ned"] == 1)
num_unchecked_ned = n_unidentified - num_checked_ned
num_with_bibliography = sum(1 for r in rows if r["filename"] in bibliography_by_filename)


# ── 7. SCORE HISTOGRAM SVG ──────────────────────────────────────────────────
# Bucket by rounded avg_score (avg_score can be fractional since it is
# averaged over 1-3 runs of a 0-50 likert scale). Split by matched/unidentified.

hist_by_score = {}

for r in rows:
    score_bin = round(float(r["avg_score"]))
    bucket = hist_by_score.setdefault(score_bin, {"matched": 0, "unidentified": 0})
    bucket[r["status"]] += 1

hist_items = sorted(hist_by_score.items())
max_hist_count = max(
    (counts["matched"] + counts["unidentified"] for _, counts in hist_items),
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
        f'aria-label="Average score histogram split by matched vs unidentified" '
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
    matched_count = counts["matched"]
    unidentified_count = counts["unidentified"]

    x_center = HIST_L + slot_w * (i + 0.5)
    x = x_center - bar_w / 2

    blue_h = HIST_PLOT_H * matched_count / max_hist_count if max_hist_count else 0
    warn_h = HIST_PLOT_H * unidentified_count / max_hist_count if max_hist_count else 0

    blue_y = HIST_BASELINE - blue_h
    warn_y = blue_y - warn_h

    title = (
        f"Avg score {score}: "
        f"matched={matched_count}, unidentified={unidentified_count}"
    )

    hist_svg_parts.append(
        f'<rect data-hist-bar="blue" '
        f'data-count="{matched_count}" '
        f'data-score="{score:g}" '
        f'x="{x:.2f}" y="{blue_y:.2f}" '
        f'width="{bar_w:.2f}" height="{blue_h:.2f}" '
        f'class="hist-blue-svg">'
        f'<title>{html.escape(title)}</title>'
        f'</rect>'
    )

    hist_svg_parts.append(
        f'<rect data-hist-bar="warn" '
        f'data-count="{unidentified_count}" '
        f'data-score="{score:g}" '
        f'x="{x:.2f}" y="{warn_y:.2f}" '
        f'width="{bar_w:.2f}" height="{warn_h:.2f}" '
        f'class="hist-warn-svg">'
        f'<title>{html.escape(title)}</title>'
        f'</rect>'
    )

    hist_svg_parts.append(
        f'<text x="{x_center:.2f}" y="{HIST_BASELINE + 22}" '
        f'text-anchor="middle" class="axis-label">{score:g}</text>'
    )

hist_svg_parts.append(
    f'<text x="{HIST_L + HIST_PLOT_W / 2}" y="{HIST_H - 10}" '
    f'text-anchor="middle" class="axis-title">Average score bin</text>'
)
hist_svg_parts.append(
    f'<text x="16" y="{HIST_T + HIST_PLOT_H / 2}" '
    f'text-anchor="middle" class="axis-title" '
    f'transform="rotate(-90 16 {HIST_T + HIST_PLOT_H / 2})">Count</text>'
)
hist_svg_parts.append("</svg>")

hist_svg = "\n".join(hist_svg_parts)


# ── 8. BUILD IMAGE GRID CARDS ────────────────────────────────────────────────
# Only show images that were converted to JPEG (top MAX_JPEG_IMAGES by score,
# across the combined matched+unidentified set). All rows are kept for the
# histogram/summary stats.

grid_rows = [
    r for r in rows
    if r.get("data_url") and r.get("data_url").strip()
]

if MAX_CARDS is not None:
    grid_rows = grid_rows[:MAX_CARDS]

SOURCE_TAG_CLASS = {
    "HF": "src-hf",
    "SIMBAD": "src-simbad",
    "NED": "src-ned",
}

cards = ""
card_bibliography = {}  # filename -> papers, restricted to rendered cards only

for rank, r in enumerate(grid_rows, start=1):
    filename = html.escape(str(r["filename"]))
    avg_score = float(r["avg_score"])
    status = r["status"]

    score_col = score_color(avg_score, max_score)
    score_pct = (avg_score / max_score * 100) if max_score > 0 else 0

    object_name = str(r.get("object_name", ""))
    search_text = html.escape(
        f"{filename} {avg_score:g} {object_name} {r.get('matched_source', '')}".lower(),
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
        caption_bits = [f"{filename}", f"Avg score={avg_score:g}"]
        if status == "matched":
            caption_bits.append(f"{r['matched_source']} match: {object_name}")
        safe_caption = html.escape(" | ".join(caption_bits), quote=True)

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

    ref_count = r.get("ref_count")
    matched_source = r.get("matched_source", "")
    papers = bibliography_by_filename.get(r["filename"], [])
    has_papers = bool(papers)

    if has_papers:
        card_bibliography[r["filename"]] = papers

    # The catalogs' own paper-count fields (SIMBAD nbref / NED n_crosref,
    # both landing in the CSV's ref_count) can disagree with the actual
    # bibliography join/fetch - sometimes because the join covers every
    # object within the match radius rather than just the representative
    # match, sometimes because the catalog's own count is simply stale
    # (NED's classic references endpoint returns "No Reference found" for
    # a large fraction of objects whose n_crosref is nonzero). Since we
    # attempt the bibliography fetch for every SIMBAD/NED match, its result
    # - including zero - is ground truth; only fall back to the catalog's
    # own count for sources we never attempt a bibliography fetch for (HF).
    if matched_source in ("SIMBAD", "NED"):
        effective_ref_count = len(papers)
    else:
        effective_ref_count = ref_count
    ref_count_val = effective_ref_count if isinstance(effective_ref_count, int) else -1

    if status == "unidentified":
        checked_ned = int(r.get("checked_ned", 0))
        status_badge = '<span class="tag unidentified">Unidentified</span>'

        if r.get("object_name"):
            # Reclassified: SIMBAD/NED knows about this object positionally,
            # but no paper genuinely discusses it (see
            # classify_genuine_discussion.py), so it doesn't count as
            # identified under our definition.
            source_class = SOURCE_TAG_CLASS.get(matched_source, "src-hf")
            source_tag = (
                f'<span class="tag {source_class}">{html.escape(matched_source)} catalog match</span>'
                f'<span class="tag not-discussed">not discussed by a paper</span>'
            )
            object_type_str = f" ({html.escape(r['object_type'])})" if r.get("object_type") else ""
            reasoning = r.get("not_discussed_reasoning", "")
            status_note = f"{html.escape(r['object_name'])}{object_type_str} &middot; in {html.escape(matched_source)} but no paper discusses it individually"

            reasoning_block = (
                f'<p class="summary-text">{html.escape(reasoning)}</p>'
                if reasoning
                else ""
            )
            papers_button = (
                f'<button class="papers-btn" onclick="openPapersModal(\'{filename}\')">'
                f'View {len(papers)} paper{"s" if len(papers) != 1 else ""} (survey-only mentions)</button>'
                if has_papers
                else ""
            )
            object_block = f"""
        <div class="object-block">
          {reasoning_block}
          {papers_button}
        </div>"""
        else:
            source_tag = (
                '<span class="tag ned-checked">HF+SIMBAD+NED checked</span>'
                if checked_ned == 1
                else '<span class="tag ned-unchecked">HF+SIMBAD only</span>'
            )
            status_note = (
                "Not found in HF, SIMBAD, or NED"
                if checked_ned == 1
                else "Not found in HF or SIMBAD (not checked against NED)"
            )
            object_block = ""
    else:
        checked_ned = 0
        matched_source = r["matched_source"]
        source_class = SOURCE_TAG_CLASS.get(matched_source, "src-hf")
        source_tag = f'<span class="tag {source_class}">{html.escape(matched_source)} match</span>'
        status_badge = '<span class="tag matched">Matched</span>'

        object_type_str = f" ({html.escape(r['object_type'])})" if r.get("object_type") else ""
        ref_count_str = (
            f"{effective_ref_count} paper{'s' if effective_ref_count != 1 else ''}"
            if isinstance(effective_ref_count, int)
            else "paper count unknown"
        )
        redshift_str = (
            f'<div class="coord"><span class="clabel">Redshift</span>'
            f'<span class="cval">{html.escape(str(r["redshift"]))}</span></div>'
            if r.get("redshift")
            else ""
        )

        papers_button = (
            f'<button class="papers-btn" onclick="openPapersModal(\'{filename}\')">'
            f'View {len(papers)} paper{"s" if len(papers) != 1 else ""}</button>'
            if has_papers
            else ""
        )

        arxiv_link = (
            f'<a class="arxiv-link" href="{html.escape(r["arxiv_url"], quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">View source paper &rarr;</a>'
            if r.get("arxiv_url")
            else ""
        )

        summary_block = (
            f'<p class="summary-text">{html.escape(r["summary"])}</p>'
            if r.get("summary")
            else ""
        )

        deep_dive = deep_dive_by_object.get(object_name)
        deep_dive_block = ""
        if deep_dive:
            deep_dive_block = f"""
          <details class="deep-dive">
            <summary class="deep-dive-label">Literature deep dive (AI-synthesized)</summary>
            <p class="deep-dive-text"><strong>What's known:</strong> {html.escape(deep_dive.get('literature_summary', ''))}</p>
            <p class="deep-dive-text"><strong>Why notable:</strong> {html.escape(deep_dive.get('why_notable', ''))}</p>
            <p class="deep-dive-text"><strong>Likely visual reason:</strong> {html.escape(deep_dive.get('likely_visual_reason', ''))}</p>
          </details>"""

        status_note = f"{html.escape(object_name)}{object_type_str} &middot; {ref_count_str}"

        object_block = f"""
        <div class="object-block">
          {redshift_str}
          {papers_button}
          {arxiv_link}
          {summary_block}
          {deep_dive_block}
        </div>"""

    cards += f"""
    <article
      class="card"
      data-score="{avg_score}"
      data-ref-count="{ref_count_val}"
      data-status="{status}"
      data-source="{r.get('matched_source', '')}"
      data-checked-ned="{checked_ned}"
      data-reclassified="{1 if status == 'unidentified' and r.get('object_name') else 0}"
      data-search="{search_text}"
    >
      <div class="img-wrap">{img_block}</div>

      <div class="truth-under-image {'matched' if status == 'matched' else 'unidentified'}">
        {status_note}
      </div>

      <div class="card-body">
        <div class="rank-row">
          <span class="rank">#{rank}</span>
          {status_badge}
        </div>

        {source_tag}

        <h2 class="fname" title="{filename}">{
          f'<a href="{legacy_survey_url}" target="_blank" rel="noopener noreferrer">{filename}</a>'
          if legacy_survey_url else filename
        }</h2>

        <div class="meta-grid">
          {meta_blocks}
        </div>

        {object_block}

        <div class="score-item">
          <div class="score-header">
            <span class="slabel">Avg Gemini Score</span>
            <span class="snum" style="color:{score_col}">{avg_score:g}</span>
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

bibliography_json = _json.dumps(card_bibliography, separators=(",", ":"))


# ── 9. HTML DOCUMENT ─────────────────────────────────────────────────────────

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Object Explorer Report</title>

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
.stat-val.w {{ color: var(--warn); }}

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
  grid-template-columns: minmax(200px, 1fr) auto auto auto;
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
  letter-spacing: .04em;
  text-transform: none;
  background: rgba(255,255,255,.025);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.truth-under-image.unidentified {{ color: var(--warn); text-transform: uppercase; letter-spacing: .08em; }}
.truth-under-image.matched {{ color: var(--text); }}

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

.tag {{
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 9px;
  letter-spacing: .08em;
  text-transform: uppercase;
  width: max-content;
}}

.tag.unidentified {{
  color: var(--warn);
  background: rgba(240,180,41,.08);
  border-color: rgba(240,180,41,.30);
}}

.tag.matched {{
  color: var(--good);
  background: rgba(0,229,160,.08);
  border-color: rgba(0,229,160,.30);
}}

.tag.ned-checked {{
  color: var(--good);
  background: rgba(0,229,160,.08);
  border-color: rgba(0,229,160,.30);
}}

.tag.ned-unchecked {{
  color: var(--text-dim);
  background: rgba(255,255,255,.03);
}}

.tag.not-discussed {{
  color: var(--warn);
  background: rgba(240,180,41,.08);
  border-color: rgba(240,180,41,.30);
}}

.tag.src-hf {{
  color: var(--accent2);
  background: rgba(123,120,255,.08);
  border-color: rgba(123,120,255,.30);
}}

.tag.src-simbad {{
  color: var(--accent);
  background: rgba(0,229,160,.08);
  border-color: rgba(0,229,160,.30);
}}

.tag.src-ned {{
  color: var(--warn);
  background: rgba(240,180,41,.08);
  border-color: rgba(240,180,41,.30);
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

.object-block {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}

.summary-text {{
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}}

.papers-btn {{
  background: rgba(0,229,160,.10);
  color: var(--good);
  border: 1px solid rgba(0,229,160,.35);
  border-radius: 8px;
  padding: 8px 10px;
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .04em;
  cursor: pointer;
  text-align: left;
}}

.papers-btn:hover {{
  background: rgba(0,229,160,.18);
}}

.arxiv-link {{
  color: var(--accent2);
  font-size: 11px;
  text-decoration: none;
  font-weight: 700;
}}

.arxiv-link:hover {{
  text-decoration: underline;
}}

.deep-dive {{
  border: 1px solid rgba(0,229,160,.25);
  background: rgba(0,229,160,.05);
  border-radius: 8px;
  padding: 8px 10px;
}}

.deep-dive-label {{
  color: var(--good);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .04em;
  cursor: pointer;
  text-transform: uppercase;
}}

.deep-dive[open] .deep-dive-label {{
  margin-bottom: 6px;
}}

.deep-dive-text {{
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.5;
  margin-top: 6px;
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

.hist-warn-svg {{
  fill: var(--warn);
  opacity: .95;
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
.swatch.warn {{ background: var(--warn); }}

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
  cursor: pointer;
  z-index: 1;
}}

.close-modal:hover {{
  color: var(--accent);
}}

.papers-modal-content {{
  cursor: default;
  margin: 6vh auto;
  width: min(720px, 92%);
  max-height: 82vh;
  overflow-y: auto;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px;
}}

.papers-modal-title {{
  font-family: var(--display);
  font-size: 18px;
  color: #fff;
  margin-bottom: 16px;
  padding-right: 30px;
}}

.paper-row {{
  border-top: 1px solid var(--border);
  padding: 12px 0;
}}

.paper-row:first-child {{
  border-top: none;
}}

.paper-year {{
  color: var(--accent2);
  font-weight: 700;
  font-size: 11px;
  margin-right: 8px;
}}

.paper-title {{
  color: var(--text);
  font-size: 12px;
  line-height: 1.5;
}}

.paper-link {{
  color: var(--accent);
  font-size: 10px;
  text-decoration: none;
  display: inline-block;
  margin-top: 4px;
}}

.paper-link:hover {{
  text-decoration: underline;
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
    <p class="eyebrow">Gemini Vision Pipeline · Cross-matched vs. HF galaxy-mentions, SIMBAD, and NED</p>
    <h1>Object <em>Explorer</em></h1>
  </header>

  <div class="stats">
    <div class="stat">
      <div class="stat-label">Total scored images</div>
      <div class="stat-val">{total}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Unidentified</div>
      <div class="stat-val w">{n_unidentified}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Matched (paper-discussed)</div>
      <div class="stat-val g">{n_matched}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Catalog match, not discussed</div>
      <div class="stat-val w">{n_reclassified}</div>
    </div>
    <div class="stat">
      <div class="stat-label">With bibliography</div>
      <div class="stat-val p">{num_with_bibliography}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Mean avg score</div>
      <div class="stat-val b">{avg_score_overall:.2f}</div>
    </div>
  </div>

  <div class="tab-controls">
    <button class="tab-btn active" onclick="switchTab(event, 'grid-view')">
      Image Grid
    </button>
    <button class="tab-btn" onclick="switchTab(event, 'plots-view')">
      Score Distribution
    </button>
  </div>

  <section id="grid-view" class="tab-panel active">
    <div class="controls">
      <input
        id="searchBox"
        class="search-box"
        type="text"
        placeholder="Search filename, score, or object name..."
        oninput="applyFilters()"
      >

      <select id="statusFilter" onchange="applyFilters()">
        <option value="all">All images</option>
        <option value="unidentified-checked">Unidentified &middot; HF+SIMBAD+NED checked</option>
        <option value="unidentified-unchecked">Unidentified &middot; HF+SIMBAD only</option>
        <option value="unidentified-reclassified">Unidentified &middot; catalog match, not discussed</option>
        <option value="matched-HF">Matched &middot; HF</option>
        <option value="matched-SIMBAD">Matched &middot; SIMBAD</option>
        <option value="matched-NED">Matched &middot; NED</option>
      </select>

      <select id="sortBy" onchange="applySort()">
        <option value="score">Sort by avg score</option>
        <option value="refcount">Sort by ref count (papers)</option>
      </select>

      <select id="scoreFilter" onchange="applyFilters()">
        <option value="all">All scores</option>
        <option value="nonzero">Score &gt; 0</option>
        <option value="zero">Score = 0</option>
      </select>
    </div>

    <div class="result-count" id="resultCount">
      Showing {len(grid_rows)} / {len(grid_rows)} grid images · {total} images total ({n_unidentified} unidentified, {n_matched} matched)
    </div>

    <div class="grid" id="imageGrid">
      {cards}
    </div>
  </section>

  <section id="plots-view" class="tab-panel">
    <div class="plot-grid">
      <article class="plot-card">
        <div class="plot-title">
          <h2>Average score histogram</h2>
          <span>Matched vs unidentified</span>
        </div>

        <p class="plot-note">
          Each image's score is averaged across however many of the 3
          gemini_likert runs it appeared in (boring images are randomly
          subsampled per seed, so most appear in only 1 run). Bars are
          binned by rounded average score and split by whether the image
          matched an object in HF/SIMBAD/NED. The y-axis maximum is
          adjustable below.
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
          <span><i class="swatch blue"></i>Matched (HF/SIMBAD/NED)</span>
          <span><i class="swatch warn"></i>Unidentified</span>
        </div>
      </article>
    </div>
  </section>
  <footer>
    Generated from {html.escape(os.path.basename(UNIDENTIFIED_CSV))}
    + {html.escape(os.path.basename(MATCHED_CSV))}
    + {html.escape(os.path.basename(SIMBAD_BIBLIOGRAPHY_CSV))}
    + {html.escape(os.path.basename(NED_BIBLIOGRAPHY_CSV))}
    + {html.escape(os.path.basename(DEEP_DIVE_JSON))}
    + {html.escape(os.path.basename(DISCUSSION_CLASSIFICATION_CSV))}
    · HDF5 source: {html.escape(os.path.basename(HDF5_PATH))}
    · JPEG conversion: {saved_jpeg_images} images embedded (top {MAX_JPEG_IMAGES} by avg score, combined matched+unidentified)
    · Cross-matched against astronolan/galaxy-mentions (coordinate_resolution config), SIMBAD (bulk TAP upload,
    all images, full bibliography via basic+has_ref+ref join), and NED (per-object TAP query, all images,
    full bibliography via astroquery classic references endpoint)
    · A SIMBAD/NED catalog match only counts as "identified" if Gemini judges that at least one paper genuinely,
    individually discusses the object (not just lists it as a survey/sample member) - see
    classify_genuine_discussion.py. {n_reclassified} catalog matches were reclassified as unidentified on this
    basis. HF matches are exempt (sourced from an actual paper-mention coordinate already).
    · Top {len(deep_dive_by_object)} most-cited matched objects have an AI-synthesized literature deep dive
    (Gemini reading ADS abstracts) - expand "Literature deep dive" on their card
  </footer>
</div>

<div id="imageModal" class="modal" onclick="closeModal()">
  <span class="close-modal">&times;</span>
  <img class="modal-content" id="modalImg" src="">
  <div class="modal-caption" id="modalCaption"></div>
</div>

<div id="papersModal" class="modal" onclick="closePapersModal(event)">
  <span class="close-modal">&times;</span>
  <div class="papers-modal-content" onclick="event.stopPropagation()">
    <div class="papers-modal-title" id="papersModalTitle"></div>
    <div id="papersModalList"></div>
  </div>
</div>

<script>
const BIBLIOGRAPHY = {bibliography_json};

function openModal(src, caption) {{
  document.getElementById("imageModal").style.display = "block";
  document.getElementById("modalImg").src = src;
  document.getElementById("modalCaption").textContent = caption || "";
}}

function closeModal() {{
  document.getElementById("imageModal").style.display = "none";
}}

function openPapersModal(filename) {{
  const papers = BIBLIOGRAPHY[filename] || [];
  const title = document.getElementById("papersModalTitle");
  const list = document.getElementById("papersModalList");

  title.textContent = `${{filename}} — ${{papers.length}} paper${{papers.length !== 1 ? "s" : ""}}`;

  list.innerHTML = papers.map(p => {{
    const adsUrl = p.bibcode
      ? `https://ui.adsabs.harvard.edu/abs/${{encodeURIComponent(p.bibcode)}}/abstract`
      : "";
    const link = adsUrl
      ? `<a class="paper-link" href="${{adsUrl}}" target="_blank" rel="noopener noreferrer">View on ADS &rarr;</a>`
      : "";

    return `
      <div class="paper-row">
        <span class="paper-year">${{p.year || "?"}}</span>
        <span class="paper-title">${{p.title || "(untitled)"}}</span><br>
        ${{link}}
      </div>`;
  }}).join("") || '<p style="color:var(--text-dim);font-size:12px">No papers found.</p>';

  document.getElementById("papersModal").style.display = "block";
}}

function closePapersModal(evt) {{
  document.getElementById("papersModal").style.display = "none";
}}

document.addEventListener("keydown", event => {{
  if (event.key === "Escape") {{
    closeModal();
    closePapersModal();
  }}
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

function applyFilters() {{
  const query = document.getElementById("searchBox").value.toLowerCase().trim();
  const scoreFilter = document.getElementById("scoreFilter").value;
  const statusFilter = document.getElementById("statusFilter").value;

  const cards = Array.from(document.querySelectorAll(".card"));
  let shown = 0;

  cards.forEach(card => {{
    const haystack = card.dataset.search || card.textContent.toLowerCase();
    const score = Number(card.dataset.score || 0);
    const status = card.dataset.status;
    const source = card.dataset.source;
    const checkedNed = card.dataset.checkedNed === "1";
    const reclassified = card.dataset.reclassified === "1";

    const matchesSearch = query === "" || haystack.includes(query);
    const matchesScore =
      scoreFilter === "all" ||
      (scoreFilter === "nonzero" && score > 0) ||
      (scoreFilter === "zero" && score === 0);

    let matchesStatus = true;
    if (statusFilter === "unidentified-checked") {{
      matchesStatus = status === "unidentified" && checkedNed && !reclassified;
    }} else if (statusFilter === "unidentified-unchecked") {{
      matchesStatus = status === "unidentified" && !checkedNed && !reclassified;
    }} else if (statusFilter === "unidentified-reclassified") {{
      matchesStatus = reclassified;
    }} else if (statusFilter.startsWith("matched-")) {{
      const wantSource = statusFilter.split("-")[1];
      matchesStatus = status === "matched" && source === wantSource;
    }}

    const visible = matchesSearch && matchesScore && matchesStatus;

    card.classList.toggle("hidden", !visible);

    if (visible) shown += 1;
  }});

  document.getElementById("resultCount").textContent =
    `Showing ${{shown}} / ${{cards.length}} grid images · {total} images total ({n_unidentified} unidentified, {n_matched} matched)`;
}}

function applySort() {{
  const sortBy = document.getElementById("sortBy").value;
  const grid = document.getElementById("imageGrid");
  const cards = Array.from(grid.querySelectorAll(".card"));

  cards.sort((a, b) => {{
    if (sortBy === "refcount") {{
      const diff = Number(b.dataset.refCount) - Number(a.dataset.refCount);
      if (diff !== 0) return diff;
      return Number(b.dataset.score) - Number(a.dataset.score);
    }}
    return Number(b.dataset.score) - Number(a.dataset.score);
  }});

  cards.forEach(card => grid.appendChild(card));
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
    const warnBar = pair.warn;

    const blueCount = blue ? Number(blue.dataset.count || 0) : 0;
    const warnCount = warnBar ? Number(warnBar.dataset.count || 0) : 0;

    const blueH = Math.min(plotH, plotH * blueCount / yMax);
    const warnH = Math.min(plotH, plotH * warnCount / yMax);

    if (blue) {{
      blue.setAttribute("height", blueH.toFixed(2));
      blue.setAttribute("y", (baseline - blueH).toFixed(2));
    }}

    if (warnBar) {{
      warnBar.setAttribute("height", warnH.toFixed(2));
      warnBar.setAttribute("y", (baseline - blueH - warnH).toFixed(2));
    }}
  }});
}}

// Page-load initialisation
applyFilters();
updateHistogramYMax();
</script>
</body>
</html>
"""


# ── 10. WRITE OUTPUT ─────────────────────────────────────────────────────────

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"Done! Report saved to: {OUTPUT_HTML}")
print(f"Rows total: {total} (unidentified={n_unidentified}, matched={n_matched})")
print(f"Rows shown in image grid: {len(grid_rows)}")
print(f"Embedded JPEG images: {saved_jpeg_images} (top {MAX_JPEG_IMAGES} by avg score)")
print(f"Cards with a bibliography link: {len(card_bibliography)}")

if missing_images:
    print(f"Warning: {missing_images} images could not be loaded.")
