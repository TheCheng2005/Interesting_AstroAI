"""
update_interesting_radius.py

Re-label the "interesting" / "classification" columns of an existing results
CSV using a tighter match radius against Interesting.csv, WITHOUT re-running
the scoring pipeline.

Background:
    The scoring scripts label an image "interesting" if it is the nearest
    catalog image to some Interesting.csv entry AND that entry is within
    MATCH_RADIUS_ARCSEC of it. They now use 3" directly, so this script is a
    no-op on the subset runs; it exists for the older full-catalog CSVs in
    results/full_catalog/, which were labelled at 10". It recomputes the same
    nearest-match logic at NEW_RADIUS_ARCSEC using the SourceRA/SourceDec
    already present in the results CSV as the catalog (no parquet reload
    needed, since the results CSV already covers every HDF5 image).

    Both header spellings are accepted, and the output is written in the same
    spelling as the input with its token footer intact - a re-labelled CSV is
    read by exactly the same consumers as the file it came from.

Usage:
    python update_interesting_radius.py
    python update_interesting_radius.py results.csv interesting.csv output.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# The stage folders are siblings, so put the analysis root on the path to
# reach common/paths.py (see its docstring).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import scores_csv
from common.paths import FULL_RUN_DIR, INTERESTING_CSV_PATH


RESULTS_CSV = (
    sys.argv[1] if len(sys.argv) > 1
    else os.path.join(FULL_RUN_DIR, "10M_10arc_dedup_centered.csv")
)
INTERESTING_CSV = (
    sys.argv[2] if len(sys.argv) > 2
    else INTERESTING_CSV_PATH
)
OUTPUT_CSV = (
    sys.argv[3] if len(sys.argv) > 3
    else RESULTS_CSV.replace(".csv", "_3arcsec.csv")
)

OLD_RADIUS_ARCSEC = 10.0
NEW_RADIUS_ARCSEC = 3.0


# ── 1. LOAD RESULTS CSV (acts as the full image catalog) ───────────────────

source = scores_csv.read(RESULTS_CSV)

# Full-catalogue CSVs can carry rows with no coordinates; they cannot be
# matched against anything, so drop them rather than crash on float("").
rows = [
    row for row in source.rows
    if (row["SourceRA"] or "").strip() and (row["SourceDec"] or "").strip()
]
n_dropped = len(source.rows) - len(rows)
if n_dropped:
    print(f"Dropped {n_dropped} rows with no coordinates")

print(f"Loaded {len(rows)} rows from {RESULTS_CSV}")

source_ids = [str(row["filename"]).strip() for row in rows]
catalog_ra = np.array([float(row["SourceRA"]) for row in rows])
catalog_dec = np.array([float(row["SourceDec"]) for row in rows])
catalog_coords = SkyCoord(ra=catalog_ra * u.deg, dec=catalog_dec * u.deg)


# ── 2. LOAD Interesting.csv AND MATCH EACH ENTRY TO ITS NEAREST IMAGE ──────

df = pd.read_csv(INTERESTING_CSV)

required_columns = {"SourceID", "Classification", "RA", "Dec"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Interesting.csv is missing columns: {sorted(missing_columns)}")

interesting_coords = SkyCoord(ra=df["RA"].values * u.deg, dec=df["Dec"].values * u.deg)
nearest_idx, sep2d, _ = interesting_coords.match_to_catalog_sky(catalog_coords)

# best (minimum) separation and classification seen per matched row index,
# restricted to entries that fall within the ORIGINAL 10" radius (this is
# exactly the set of matches the scoring scripts used to build "interesting").
best_sep_by_row = {}
best_cls_by_row = {}

n_matched_within_old_radius = 0

for entry_idx, (row_idx, sep) in enumerate(zip(nearest_idx, sep2d.arcsec)):
    if sep > OLD_RADIUS_ARCSEC:
        continue

    n_matched_within_old_radius += 1
    classification = str(df.iloc[entry_idx]["Classification"]).strip()

    if row_idx not in best_sep_by_row or sep < best_sep_by_row[row_idx]:
        best_sep_by_row[row_idx] = sep
        best_cls_by_row[row_idx] = classification

print(
    f"{n_matched_within_old_radius} / {len(df)} Interesting.csv entries matched "
    f"within {OLD_RADIUS_ARCSEC}\" ({len(best_sep_by_row)} unique images)."
)


# ── 3. SANITY CHECK: every currently-interesting row must be in that set ──

old_interesting_rows = [
    i for i, row in enumerate(rows) if str(row["interesting"]).strip() == "1"
]

print(f"\nRows currently marked interesting=1: {len(old_interesting_rows)}")

unmatched = [i for i in old_interesting_rows if i not in best_sep_by_row]
over_old_radius = [
    i for i in old_interesting_rows
    if i in best_sep_by_row and best_sep_by_row[i] > OLD_RADIUS_ARCSEC
]

if unmatched:
    print(
        f"WARNING: {len(unmatched)} currently-interesting rows were NOT "
        f"recovered by the {OLD_RADIUS_ARCSEC}\" nearest-match recomputation "
        f"(catalog/coordinate mismatch?). First few indices: {unmatched[:10]}"
    )
else:
    print(f"OK: all {len(old_interesting_rows)} currently-interesting rows were recovered.")

if over_old_radius:
    print(f"WARNING: {len(over_old_radius)} matched rows exceed {OLD_RADIUS_ARCSEC}\" (should be impossible).")
else:
    print(f"OK: all matched separations are within {OLD_RADIUS_ARCSEC}\".")


# ── 4. RE-LABEL AT THE NEW (TIGHTER) RADIUS ────────────────────────────────

n_still_interesting = 0
n_demoted = 0

for i, row in enumerate(rows):
    sep = best_sep_by_row.get(i)

    if sep is not None and sep <= NEW_RADIUS_ARCSEC:
        new_interesting = 1
        new_classification = best_cls_by_row[i]
        if str(row["interesting"]).strip() != "1":
            # Shouldn't happen (new radius is strictly tighter than old), but
            # guard against it anyway.
            pass
        n_still_interesting += 1
    else:
        new_interesting = 0
        new_classification = ""
        if str(row["interesting"]).strip() == "1":
            n_demoted += 1

    row["interesting"] = new_interesting
    row["classification"] = new_classification

print(
    f"\nAt {NEW_RADIUS_ARCSEC}\" radius: {n_still_interesting} interesting "
    f"({n_demoted} rows demoted from interesting=1 to interesting=0)."
)


# ── 5. WRITE OUTPUT ─────────────────────────────────────────────────────────

scores_csv.write(OUTPUT_CSV, rows, source.header, source.tokens)

print(f"\nDone! Wrote {len(rows)} rows to {OUTPUT_CSV}")
