"""
Cross-match our Gemini-scored Hubble images against three literature/object
catalogs, to surface high-scoring images that do not correspond to any
already-identified object:
    1. astronolan/galaxy-mentions (Hugging Face) - literature-mention derived
       coordinates. Matches are enriched with arxiv_url/summary from the
       galaxy_mentions config (joined via mention_id).
    2. SIMBAD (CDS Strasbourg) - bulk TAP-upload cross-match, checked for
       every image. Matches are enriched with object type and nbref (paper
       count); the full bibliography (bibcode/year/title per paper) for
       every matched object is written to a separate CSV.
    3. NED (NASA/IPAC Extragalactic Database) - checked one object at a time
       (NED's TAP service does not support bulk table uploads), so only run
       against the top NED_TOP_N highest-scoring images that survive the HF
       + SIMBAD pass. Matches are enriched with object type, n_crosref
       (paper count), and redshift. Images below the NED_TOP_N cutoff are
       left NED-unchecked and flagged as such in the output (checked_ned=0).

Pipeline:
1. Load the three gemini_likert_{1,2,3}.csv runs from subset_test/. Each run
   scores a different (mostly non-overlapping) random subset of boring images
   plus every labeled-interesting image, so the union of filenames across the
   three runs is much bigger than any single run.
2. Average each image's imagescore over however many of the 3 runs it appears
   in (1, 2, or 3 - whatever data exists for that filename).
3. Download (and cache locally) the "coordinate_resolution" and
   "galaxy_mentions" configs of the astronolan/galaxy-mentions HF dataset.
   Cross-match against coordinate_resolution; enrich matches via
   galaxy_mentions.
4. Bulk cross-match every image still unidentified after step 3 against
   SIMBAD via a single TAP table-upload query per chunk. Fetch nbref/otype
   for matches and the full bibliography for each matched object.
5. Take the top NED_TOP_N (by avg_score) images still unidentified after
   HF + SIMBAD, and cross-match each individually against NED, fetching
   n_crosref/otype/redshift for matches.
6. Images matched by none of the three are "unidentified." Write them to a
   CSV, highest average score first. Images matched by any of the three are
   written to a separate "matched" CSV with the enrichment fields above.

All three cross-matches use MATCH_RADIUS_ARCSEC (3").

Output files:
- unidentified_objects.csv columns:
    filename, avg_score, SourceRA, SourceDec, checked_ned

  checked_ned:
  - 1 if this image was one of the NED_TOP_N checked against NED (and came
    back with no match there either).
  - 0 if it was never checked against NED (below the NED_TOP_N cutoff) - it
    is only confirmed unidentified in HF + SIMBAD, not NED.

- matched_objects.csv columns:
    filename, avg_score, SourceRA, SourceDec, matched_source, object_name,
    object_type, ref_count, redshift, arxiv_url, summary

  matched_source: "HF", "SIMBAD", or "NED" (whichever caught it first).
  ref_count: nbref (SIMBAD) or n_crosref (NED); blank for HF matches (HF has
  no per-object reference count, but arxiv_url/summary are filled instead).

- simbad_bibliography.csv columns:
    filename, main_id, bibcode, year, title

  One row per (SIMBAD-matched image, paper that mentions its SIMBAD object).
  NED does not expose a bibliography join table over TAP, so this list is
  SIMBAD-only.
"""

import os
import csv
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyvo
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.ned import Ned


# ── CONFIGURATION ───────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBSET_DIR = os.path.join(SCRIPT_DIR, "subset_test")

LIKERT_RUN_FILES = [
    os.path.join(SUBSET_DIR, "gemini_likert_1.csv"),
    os.path.join(SUBSET_DIR, "gemini_likert_2.csv"),
    os.path.join(SUBSET_DIR, "gemini_likert_3.csv"),
]

HF_CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
HF_RESOLVE_BASE = (
    "https://huggingface.co/datasets/astronolan/galaxy-mentions/resolve/"
    "refs%2Fconvert%2Fparquet"
)
COORD_RESOLUTION_PARQUET_URL = f"{HF_RESOLVE_BASE}/coordinate_resolution/train/0000.parquet"
COORD_RESOLUTION_PARQUET_PATH = os.path.join(HF_CACHE_DIR, "coordinate_resolution.parquet")
GALAXY_MENTIONS_PARQUET_URL = f"{HF_RESOLVE_BASE}/galaxy_mentions/train/0000.parquet"
GALAXY_MENTIONS_PARQUET_PATH = os.path.join(HF_CACHE_DIR, "galaxy_mentions.parquet")

MATCH_RADIUS_ARCSEC = 3.0  # cross-match radius against all three catalogs
MATCH_RADIUS_DEG = MATCH_RADIUS_ARCSEC / 3600.0

SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap"
SIMBAD_CHUNK_SIZE = 5000  # rows per bulk TAP-upload query

NED_TAP_URL = "https://ned.ipac.caltech.edu/tap"
NED_TOP_N = None  # None = check every HF+SIMBAD survivor (not just a top-scoring subset)
NED_WORKERS = 8  # polite concurrency for NED's public per-object TAP service
NED_BIBLIO_WORKERS = 5  # concurrency for NED's classic (non-TAP) references endpoint

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "unidentified_objects.csv")
MATCHED_CSV = os.path.join(SCRIPT_DIR, "matched_objects.csv")
SIMBAD_BIBLIOGRAPHY_CSV = os.path.join(SCRIPT_DIR, "simbad_bibliography.csv")
NED_BIBLIOGRAPHY_CSV = os.path.join(SCRIPT_DIR, "ned_bibliography.csv")

# Checkpoint files so a multi-hour NED run can be interrupted and resumed
# without re-querying objects already checked.
NED_CHECKPOINT_CSV = os.path.join(SCRIPT_DIR, "ned_checkpoint.csv")
NED_BIBLIO_CHECKPOINT_CSV = os.path.join(SCRIPT_DIR, "ned_biblio_checkpoint.csv")


# ── 1. LOAD AND AVERAGE OUR GEMINI LIKERT SCORES ────────────────────────────

def load_likert_run(path):
    """
    Load one gemini_likert_*.csv run. Trailing '# TOKEN USAGE SUMMARY' rows
    are skipped (they don't have a valid filename in column 1).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = row.get("index", "") or ""
            filename = row.get("filename", "")
            if not filename or index.startswith("#"):
                continue
            rows.append({
                "filename": filename,
                "imagescore": int(row["imagescore"]),
                "SourceRA": float(row["SourceRA"]) if row["SourceRA"] else None,
                "SourceDec": float(row["SourceDec"]) if row["SourceDec"] else None,
            })
    return rows


def average_scores_across_runs(run_paths):
    """
    Average imagescore per filename over however many runs it appears in.
    SourceRA/SourceDec are taken from the first run in which the filename
    appears (they are the same across runs for a given filename).

    Returns a list of dicts: filename, avg_score, n_runs, SourceRA, SourceDec.
    """
    totals = {}
    counts = {}
    coords = {}

    for path in run_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected likert run CSV not found: {path}")

        for row in load_likert_run(path):
            filename = row["filename"]
            totals[filename] = totals.get(filename, 0) + row["imagescore"]
            counts[filename] = counts.get(filename, 0) + 1

            if filename not in coords:
                coords[filename] = (row["SourceRA"], row["SourceDec"])

    averaged = []
    for filename, total in totals.items():
        n_runs = counts[filename]
        ra, dec = coords[filename]

        averaged.append({
            "filename": filename,
            "avg_score": total / n_runs,
            "n_runs": n_runs,
            "SourceRA": ra,
            "SourceDec": dec,
        })

    print(
        f"Loaded {len(run_paths)} likert runs. "
        f"Union of scored images: {len(averaged)}."
    )
    n_all_three = sum(1 for r in averaged if r["n_runs"] == len(run_paths))
    print(f"Images present in all {len(run_paths)} runs: {n_all_three}")

    return averaged


# ── 2. HF COORDINATE-RESOLUTION + GALAXY-MENTIONS CATALOGS ──────────────────

def download_parquet(url, dest_path):
    """Download a parquet file if not already cached."""
    if os.path.exists(dest_path):
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {os.path.basename(dest_path)} from Hugging Face...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")


def load_resolved_catalog(parquet_path):
    """
    Load the coordinate_resolution config and keep only rows with a valid
    resolved RA/Dec (has_resolved_coordinates == True).
    """
    df = pd.read_parquet(parquet_path)

    resolved = df[df["has_resolved_coordinates"] == True].copy()
    resolved = resolved.dropna(subset=["resolved_ra_deg", "resolved_dec_deg"])

    print(
        f"Loaded coordinate_resolution catalog: {len(df)} total rows, "
        f"{len(resolved)} with resolved RA/Dec."
    )

    return resolved


def load_mentions_lookup(parquet_path):
    """Load galaxy_mentions and index by mention_id for arxiv_url/summary lookup."""
    df = pd.read_parquet(parquet_path, columns=["mention_id", "arxiv_url", "summary"])
    return df.set_index("mention_id")[["arxiv_url", "summary"]].to_dict("index")


def _run_sync_with_retry(service, query, uploads=None, delays=(1, 2, 4, 8, 16)):
    """
    Run a TAP sync query with retries: both SIMBAD's and NED's public TAP
    services occasionally drop connections under load, especially on
    larger/heavier queries like the bibliography join.
    """
    for i, delay in enumerate(delays):
        try:
            if uploads is not None:
                return service.run_sync(query, uploads=uploads)
            return service.run_sync(query)
        except Exception:
            if i == len(delays) - 1:
                raise
            time.sleep(delay)


def cross_match_hf(records, resolved_catalog, mentions_lookup, radius_arcsec=MATCH_RADIUS_ARCSEC):
    """
    Split records into (unmatched, matched) against the HF coordinate_resolution
    catalog. Matched records are enriched with object_name (resolved_name or
    ned_object_name) and arxiv_url/summary (joined via mention_id).
    """
    catalog_coords = SkyCoord(
        ra=resolved_catalog["resolved_ra_deg"].values * u.deg,
        dec=resolved_catalog["resolved_dec_deg"].values * u.deg,
    )

    image_coords = SkyCoord(
        ra=[r["SourceRA"] for r in records] * u.deg,
        dec=[r["SourceDec"] for r in records] * u.deg,
    )

    nearest_idx, sep2d, _ = image_coords.match_to_catalog_sky(catalog_coords)

    unmatched = []
    matched = []

    resolved_reset = resolved_catalog.reset_index(drop=True)

    for record, idx, sep in zip(records, nearest_idx, sep2d.arcsec):
        if sep > radius_arcsec:
            unmatched.append(record)
            continue

        cat_row = resolved_reset.iloc[idx]
        mention = mentions_lookup.get(cat_row.get("mention_id"), {})

        object_name = cat_row.get("resolved_name") or cat_row.get("ned_object_name") or ""

        matched.append({
            **record,
            "matched_source": "HF",
            "object_name": object_name,
            "object_type": "",
            "ref_count": "",
            "redshift": "",
            "arxiv_url": mention.get("arxiv_url", ""),
            "summary": mention.get("summary", ""),
        })

    print(
        f"HF cross-match: {len(records)} images vs "
        f"{len(resolved_catalog)} resolved catalog entries within {radius_arcsec}\". "
        f"Matched: {len(matched)}  Unmatched: {len(unmatched)}"
    )

    return unmatched, matched


# ── 3. SIMBAD BULK CROSS-MATCH ──────────────────────────────────────────────

def cross_match_simbad(records, radius_deg=MATCH_RADIUS_DEG, chunk_size=SIMBAD_CHUNK_SIZE):
    """
    Bulk cross-match records against SIMBAD via TAP table upload, in chunks.
    Fetches main_id/otype/nbref for every match. If a record falls within
    radius_deg of more than one SIMBAD object, the one with the highest nbref
    (most-studied) is kept as the representative match.

    Returns (unmatched, matched) records; matched records are enriched with
    object_name, object_type, ref_count (nbref).
    """
    simbad = pyvo.dal.TAPService(SIMBAD_TAP_URL)
    best_match_by_filename = {}

    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]

        upload_table = Table({
            "filename": [r["filename"] for r in chunk],
            "ra": [r["SourceRA"] for r in chunk],
            "dec": [r["SourceDec"] for r in chunk],
        })

        query = f"""
        SELECT mine.filename, basic.main_id, basic.otype, basic.nbref
        FROM TAP_UPLOAD.mine AS mine
        JOIN basic
        ON 1=CONTAINS(POINT('ICRS', basic.ra, basic.dec),
                       CIRCLE('ICRS', mine.ra, mine.dec, {radius_deg}))
        """

        t0 = time.time()
        result = _run_sync_with_retry(simbad, query, uploads={"mine": upload_table})

        for row in result:
            filename = row["filename"]
            nbref = int(row["nbref"]) if row["nbref"] is not None else 0
            existing = best_match_by_filename.get(filename)

            if existing is None or nbref > existing["ref_count"]:
                best_match_by_filename[filename] = {
                    "object_name": str(row["main_id"]),
                    "object_type": str(row["otype"]),
                    "ref_count": nbref,
                }

        print(
            f"SIMBAD chunk {i}-{i + len(chunk)}: {time.time() - t0:.1f}s, "
            f"running total matched={len(best_match_by_filename)}"
        )

    unmatched = [r for r in records if r["filename"] not in best_match_by_filename]
    matched = [
        {
            **r,
            "matched_source": "SIMBAD",
            "object_name": best_match_by_filename[r["filename"]]["object_name"],
            "object_type": best_match_by_filename[r["filename"]]["object_type"],
            "ref_count": best_match_by_filename[r["filename"]]["ref_count"],
            "redshift": "",
            "arxiv_url": "",
            "summary": "",
        }
        for r in records
        if r["filename"] in best_match_by_filename
    ]

    print(
        f"SIMBAD cross-match: {len(records)} images. "
        f"Matched: {len(matched)}  Unmatched: {len(unmatched)}"
    )

    return unmatched, matched


def fetch_simbad_bibliography(matched_simbad_records, radius_deg=MATCH_RADIUS_DEG, chunk_size=1000):
    """
    For every SIMBAD-matched record, fetch the full list of papers
    (bibcode, year, title) that mention its SIMBAD object via the
    basic -> has_ref -> ref join. Returns a list of dicts, one row per
    (filename, paper).
    """
    if not matched_simbad_records:
        return []

    simbad = pyvo.dal.TAPService(SIMBAD_TAP_URL)
    biblio_rows = []

    for i in range(0, len(matched_simbad_records), chunk_size):
        chunk = matched_simbad_records[i:i + chunk_size]

        upload_table = Table({
            "filename": [r["filename"] for r in chunk],
            "ra": [r["SourceRA"] for r in chunk],
            "dec": [r["SourceDec"] for r in chunk],
        })

        query = f"""
        SELECT mine.filename, basic.main_id, ref.bibcode, ref."year" AS pub_year, ref.title
        FROM TAP_UPLOAD.mine AS mine
        JOIN basic ON 1=CONTAINS(POINT('ICRS', basic.ra, basic.dec),
                                  CIRCLE('ICRS', mine.ra, mine.dec, {radius_deg}))
        JOIN has_ref ON has_ref.oidref = basic.oid
        JOIN ref ON ref.oidbib = has_ref.oidbibref
        """

        t0 = time.time()
        result = _run_sync_with_retry(simbad, query, uploads={"mine": upload_table})

        for row in result:
            biblio_rows.append({
                "filename": row["filename"],
                "main_id": str(row["main_id"]),
                "bibcode": str(row["bibcode"]),
                "year": int(row["pub_year"]) if row["pub_year"] is not None else "",
                "title": str(row["title"]),
            })

        print(f"SIMBAD bibliography chunk {i}-{i + len(chunk)}: {time.time() - t0:.1f}s, {len(result)} paper rows")

    print(f"SIMBAD bibliography: {len(biblio_rows)} (image, paper) rows total")

    return biblio_rows


# ── 4. NED PER-OBJECT CROSS-MATCH (TOP-N ONLY) ──────────────────────────────

def _ned_query_one(record, radius_deg):
    """
    Run a single NED cone-search query for one record's coordinates, with
    retries: NED's public TAP service occasionally drops connections under
    concurrent load. Fetches prefname/prefphytype/n_crosref/z for the nearest
    match (TOP 1).
    """
    ned = pyvo.dal.TAPService(NED_TAP_URL)
    ra, dec = record["SourceRA"], record["SourceDec"]

    query = f"""
    SELECT TOP 1 prefname, prefphytype, n_crosref, z
    FROM NEDTAP.objdir
    WHERE 1=CONTAINS(POINT('J2000', ra, dec),
                      CIRCLE('J2000', {ra}, {dec}, {radius_deg}))
    """

    result = _run_sync_with_retry(ned, query)

    if len(result) == 0:
        return record["filename"], None

    row = result[0]
    z_value = row["z"]
    redshift = "" if z_value is None or np.ma.is_masked(z_value) or pd.isna(z_value) else float(z_value)

    return record["filename"], {
        "object_name": str(row["prefname"]),
        "object_type": str(row["prefphytype"]),
        "ref_count": int(row["n_crosref"]) if row["n_crosref"] is not None else 0,
        "redshift": redshift,
    }


def _load_ned_checkpoint(path):
    """
    Load a previously-written NED checkpoint CSV, if any. Returns a dict
    filename -> match_info (or None if that filename was checked and found
    to have no NED match).
    """
    checkpoint = {}
    if not os.path.exists(path):
        return checkpoint

    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["matched"] == "1":
                checkpoint[row["filename"]] = {
                    "object_name": row["object_name"],
                    "object_type": row["object_type"],
                    "ref_count": int(row["ref_count"]) if row["ref_count"] else 0,
                    "redshift": float(row["redshift"]) if row["redshift"] else "",
                }
            else:
                checkpoint[row["filename"]] = None

    print(f"Resuming NED cross-match from checkpoint: {len(checkpoint)} images already checked.")
    return checkpoint


def cross_match_ned(records, radius_deg=MATCH_RADIUS_DEG, top_n=NED_TOP_N, workers=NED_WORKERS,
                     checkpoint_path=NED_CHECKPOINT_CSV):
    """
    Cross-match records against NED (one query per object, run with modest
    thread-pool concurrency since NED's TAP service does not support bulk
    table uploads). If top_n is None, every record is checked; otherwise
    only the top_n highest-scoring records are checked and the rest are
    passed through untouched (NED-unchecked).

    Results are checkpointed to checkpoint_path as they arrive, so an
    interrupted run can be resumed without re-querying already-checked
    objects.

    Returns (unmatched, matched, checked_filenames).
    """
    records_sorted = sorted(records, key=lambda r: r["avg_score"], reverse=True)
    to_check = records_sorted if top_n is None else records_sorted[:top_n]
    passthrough = [] if top_n is None else records_sorted[top_n:]

    checkpoint = _load_ned_checkpoint(checkpoint_path)
    still_to_query = [r for r in to_check if r["filename"] not in checkpoint]

    print(
        f"NED cross-match: {len(to_check)} images to check "
        f"({len(to_check) - len(still_to_query)} already in checkpoint, "
        f"{len(still_to_query)} remaining)..."
    )

    checkpoint_is_new = not os.path.exists(checkpoint_path)
    checkpoint_file = open(checkpoint_path, "a", newline="", encoding="utf-8")
    checkpoint_writer = csv.writer(checkpoint_file)
    if checkpoint_is_new:
        checkpoint_writer.writerow(["filename", "matched", "object_name", "object_type", "ref_count", "redshift"])

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_ned_query_one, record, radius_deg)
                for record in still_to_query
            ]

            for n_done, future in enumerate(as_completed(futures), start=1):
                filename, match_info = future.result()
                checkpoint[filename] = match_info

                if match_info is not None:
                    checkpoint_writer.writerow([
                        filename, 1, match_info["object_name"], match_info["object_type"],
                        match_info["ref_count"], match_info["redshift"],
                    ])
                else:
                    checkpoint_writer.writerow([filename, 0, "", "", "", ""])

                if n_done % 100 == 0:
                    checkpoint_file.flush()

                if n_done % 500 == 0:
                    print(f"  NED progress: {n_done}/{len(still_to_query)} newly checked, {time.time() - t0:.0f}s elapsed")
    finally:
        checkpoint_file.close()

    match_info_by_filename = {
        filename: info for filename, info in checkpoint.items() if info is not None
    }
    checked_filenames = {r["filename"] for r in to_check}

    unmatched_checked = [r for r in to_check if r["filename"] not in match_info_by_filename]
    matched = [
        {
            **r,
            "matched_source": "NED",
            "object_name": match_info_by_filename[r["filename"]]["object_name"],
            "object_type": match_info_by_filename[r["filename"]]["object_type"],
            "ref_count": match_info_by_filename[r["filename"]]["ref_count"],
            "redshift": match_info_by_filename[r["filename"]]["redshift"],
            "arxiv_url": "",
            "summary": "",
        }
        for r in to_check
        if r["filename"] in match_info_by_filename
    ]
    unmatched = unmatched_checked + passthrough

    print(
        f"NED cross-match done in {time.time() - t0:.0f}s: "
        f"{len(to_check)} checked ({len(still_to_query)} newly queried), matched={len(matched)}, "
        f"unmatched-and-checked={len(unmatched_checked)}, "
        f"passed-through-unchecked={len(passthrough)}"
    )

    return unmatched, matched, checked_filenames


# ── 4b. NED BIBLIOGRAPHY (classic non-TAP references endpoint) ──────────────

def _ned_biblio_query_one(object_name, delays=(1, 2, 4, 8, 16)):
    """
    Fetch the full reference list (bibcode, title) for one NED object via
    astroquery's classic (non-TAP) interface, with retries. Objects with no
    references raise an astroquery exception, which we treat as an empty
    list rather than an error.
    """
    for i, delay in enumerate(delays):
        try:
            table = Ned.get_table(object_name, table="references")
            papers = []
            for row in table:
                bibcode = str(row["Refcode"]).strip()
                title = str(row["Article Title"]).strip() if row["Article Title"] else ""
                year = bibcode[:4] if bibcode[:4].isdigit() else ""
                papers.append({"bibcode": bibcode, "year": year, "title": title})
            return object_name, papers
        except Exception as e:
            msg = str(e).lower()
            if "no ref" in msg or "no references" in msg or "no match" in msg:
                return object_name, []
            if i == len(delays) - 1:
                print(f"  NED biblio failed for {object_name!r} after retries: {e}")
                return object_name, []
            time.sleep(delay)


def fetch_ned_bibliography(matched_ned_records, workers=NED_BIBLIO_WORKERS,
                            checkpoint_path=NED_BIBLIO_CHECKPOINT_CSV):
    """
    For every unique NED object matched, fetch its full reference list
    (bibcode/year/title) via the classic references endpoint. Results are
    checkpointed per-object (not per-image) since several images can
    resolve to the same NED object. Returns a list of dicts, one row per
    (filename, paper).
    """
    if not matched_ned_records:
        return []

    filenames_by_object = {}
    for r in matched_ned_records:
        filenames_by_object.setdefault(r["object_name"], []).append(r["filename"])

    unique_objects = sorted(filenames_by_object.keys())

    papers_by_object = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                papers_by_object.setdefault(row["object_name"], []).append({
                    "bibcode": row["bibcode"], "year": row["year"], "title": row["title"],
                })
        # NED_NO_PAPERS marker rows record objects we already checked that
        # have zero references, so we don't requery them.
        checked_objects = set(papers_by_object.keys())
        if os.path.exists(checkpoint_path + ".done"):
            with open(checkpoint_path + ".done", "r", encoding="utf-8") as f:
                checked_objects |= {line.strip() for line in f if line.strip()}
    else:
        checked_objects = set()

    still_to_query = [o for o in unique_objects if o not in checked_objects]
    print(
        f"NED bibliography: {len(unique_objects)} unique matched objects "
        f"({len(unique_objects) - len(still_to_query)} already in checkpoint, "
        f"{len(still_to_query)} remaining)..."
    )

    checkpoint_is_new = not os.path.exists(checkpoint_path)
    checkpoint_file = open(checkpoint_path, "a", newline="", encoding="utf-8")
    checkpoint_writer = csv.writer(checkpoint_file)
    if checkpoint_is_new:
        checkpoint_writer.writerow(["object_name", "bibcode", "year", "title"])
    done_file = open(checkpoint_path + ".done", "a", encoding="utf-8")

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_ned_biblio_query_one, obj) for obj in still_to_query]

            for n_done, future in enumerate(as_completed(futures), start=1):
                object_name, papers = future.result()
                papers_by_object[object_name] = papers

                for p in papers:
                    checkpoint_writer.writerow([object_name, p["bibcode"], p["year"], p["title"]])
                done_file.write(object_name + "\n")

                if n_done % 100 == 0:
                    checkpoint_file.flush()
                    done_file.flush()

                if n_done % 500 == 0:
                    print(f"  NED biblio progress: {n_done}/{len(still_to_query)} newly checked, {time.time() - t0:.0f}s elapsed")
    finally:
        checkpoint_file.close()
        done_file.close()

    biblio_rows = []
    for object_name, filenames in filenames_by_object.items():
        for p in papers_by_object.get(object_name, []):
            for filename in filenames:
                biblio_rows.append({
                    "filename": filename,
                    "object_name": object_name,
                    "bibcode": p["bibcode"],
                    "year": p["year"],
                    "title": p["title"],
                })

    print(f"NED bibliography done in {time.time() - t0:.0f}s: {len(biblio_rows)} (image, paper) rows total")

    return biblio_rows


# ── 5. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading and averaging our Gemini likert scores across runs...")
    averaged_images = average_scores_across_runs(LIKERT_RUN_FILES)

    print("\nLoading Hugging Face catalogs...")
    download_parquet(COORD_RESOLUTION_PARQUET_URL, COORD_RESOLUTION_PARQUET_PATH)
    download_parquet(GALAXY_MENTIONS_PARQUET_URL, GALAXY_MENTIONS_PARQUET_PATH)
    resolved_catalog = load_resolved_catalog(COORD_RESOLUTION_PARQUET_PATH)
    mentions_lookup = load_mentions_lookup(GALAXY_MENTIONS_PARQUET_PATH)

    print(f"\n--- Stage 1: HF cross-match ({MATCH_RADIUS_ARCSEC}\") ---")
    after_hf, matched_hf = cross_match_hf(averaged_images, resolved_catalog, mentions_lookup, MATCH_RADIUS_ARCSEC)

    print(f"\n--- Stage 2: SIMBAD cross-match ({MATCH_RADIUS_ARCSEC}\") ---")
    after_simbad, matched_simbad = cross_match_simbad(after_hf, MATCH_RADIUS_DEG)

    print("\n--- Stage 2b: SIMBAD bibliography fetch ---")
    simbad_bibliography = fetch_simbad_bibliography(matched_simbad, MATCH_RADIUS_DEG)

    ned_scope = "all" if NED_TOP_N is None else f"top {NED_TOP_N} by score"
    print(f"\n--- Stage 3: NED cross-match ({MATCH_RADIUS_ARCSEC}\", {ned_scope}) ---")
    after_ned, matched_ned, ned_checked_filenames = cross_match_ned(after_simbad, MATCH_RADIUS_DEG, NED_TOP_N, NED_WORKERS)

    print("\n--- Stage 3b: NED bibliography fetch ---")
    ned_bibliography = fetch_ned_bibliography(matched_ned, NED_BIBLIO_WORKERS)

    after_ned.sort(key=lambda r: r["avg_score"], reverse=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "avg_score", "SourceRA", "SourceDec", "checked_ned"])

        for record in after_ned:
            checked_ned = 1 if record["filename"] in ned_checked_filenames else 0

            writer.writerow([
                record["filename"],
                record["avg_score"],
                record["SourceRA"],
                record["SourceDec"],
                checked_ned,
            ])

    print(f"\nDone! Wrote {len(after_ned)} unidentified images to {OUTPUT_CSV}")
    print(
        f"Of these, {sum(1 for r in after_ned if r['filename'] in ned_checked_filenames)} "
        f"were confirmed unidentified in HF + SIMBAD + NED; the rest are only "
        f"confirmed unidentified in HF + SIMBAD (not checked against NED)."
    )

    all_matched = matched_hf + matched_simbad + matched_ned
    all_matched.sort(key=lambda r: r["avg_score"], reverse=True)

    with open(MATCHED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "avg_score", "SourceRA", "SourceDec", "matched_source",
            "object_name", "object_type", "ref_count", "redshift", "arxiv_url", "summary",
        ])

        for record in all_matched:
            writer.writerow([
                record["filename"],
                record["avg_score"],
                record["SourceRA"],
                record["SourceDec"],
                record["matched_source"],
                record["object_name"],
                record["object_type"],
                record["ref_count"],
                record["redshift"],
                record["arxiv_url"],
                record["summary"],
            ])

    print(
        f"Wrote {len(all_matched)} matched images to {MATCHED_CSV} "
        f"(HF={len(matched_hf)}, SIMBAD={len(matched_simbad)}, NED={len(matched_ned)})"
    )

    with open(SIMBAD_BIBLIOGRAPHY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "main_id", "bibcode", "year", "title"])

        for row in sorted(simbad_bibliography, key=lambda r: (r["filename"], r["year"] or 0)):
            writer.writerow([row["filename"], row["main_id"], row["bibcode"], row["year"], row["title"]])

    print(f"Wrote {len(simbad_bibliography)} (image, paper) rows to {SIMBAD_BIBLIOGRAPHY_CSV}")

    with open(NED_BIBLIOGRAPHY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "object_name", "bibcode", "year", "title"])

        for row in sorted(ned_bibliography, key=lambda r: (r["filename"], r["year"] or "0")):
            writer.writerow([row["filename"], row["object_name"], row["bibcode"], row["year"], row["title"]])

    print(f"Wrote {len(ned_bibliography)} (image, paper) rows to {NED_BIBLIOGRAPHY_CSV}")
