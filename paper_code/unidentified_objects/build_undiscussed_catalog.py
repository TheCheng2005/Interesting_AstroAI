"""
Build the released candidate catalogue: high-scoring cutouts that are not
reference anomalies, that no catalogue of already-recognised objects claims,
and that no paper discusses individually.

Four conditions, applied in order:

  1. imagescore >= MIN_SCORE (45, the released cut).
  2. interesting == 0 - not one of the AnomalyMatch reference anomalies.
  3. No counterpart within MATCH_RADIUS_ARCSEC in either
       a. the HF galaxy-mentions coordinate catalogue - a hit means a paper
          already resolved a name to this position; or
       b. the interacting-galaxy catalogues of O'Ryan et al. (2023),
          Zenodo 7684876 - a hit means the morphology was already recognised.
  4. No SIMBAD or NED object matched to the image is *genuinely discussed* in
     the literature.

Condition 3 runs before condition 4 on purpose: an image dropped there costs
no ADS quota and no LLM call.

Condition 4 is the point of the exercise, and it is deliberately weaker than
"uncatalogued". An object can sit in SIMBAD purely as row 400 of a survey
table with no paper ever saying a word about it. Such an image stays in this
catalogue: being catalogued is fine, being discussed is not. So SIMBAD and
NED are queried here *only to collect papers*, never to disqualify an image.

Two consequences for how that cross-match runs:

  - Both SIMBAD and NED are queried for every surviving image, rather than
    stopping at the first catalogue that answers. Stopping early would leave
    SIMBAD-matched images with no NED bibliography, understating the
    literature on exactly the objects most likely to have some.
  - An image can resolve to more than one catalogue object within the match
    radius (an optical and a radio designation for one source, say). The
    image is dropped if *any* of its objects is genuinely discussed, so what
    survives has no discussed counterpart at all.

Images that match no catalogue object have no papers, hence no discussion,
and are kept without an LLM call.

Evidence for the discussion verdict is every paper's abstract from ADS, plus
an ADS full-text search that returns verbatim in-body snippets - the thing
that separates "listed in Table 3" from "we model its tidal tail in
Section 4".

The script is self-contained: it does the catalogue cross-match itself
(sections 2A-i to 2A-iv below) rather than importing it.

Checkpoints under cache/undiscussed_catalog/ let an interrupted run resume
without re-spending API quota:
    crossmatch.json     images, their matched objects, and their papers
    verdicts.csv        per-object discussion verdicts
    ned_*.csv           NED per-object and bibliography checkpoints
The shared fulltext_hits.csv / fulltext_search_checked.txt are reused and
extended, so objects an earlier sweep already searched cost nothing here.

Outputs:
    undiscussed_catalog.csv               the released catalogue
    all_nonreference_above_threshold.csv  every image entering condition 3,
                                          with `dropped_by` naming what
                                          removed it (blank = released)
    candidate_counts.json                 the counts quoted in the paper

Usage:
    python build_undiscussed_catalog.py [scores.csv] [output.csv]

Requires GEMINI_API_KEY, ~/.ads_api_key, and ORYAN_CATALOGUE_DIR.
"""

import os
import csv
import glob
import json
import sys
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

# The stage folders are siblings; reach common/, this stage's own modules and
# the literature helpers. Running as a script would put this file's directory
# on sys.path implicitly, but naming it here means the module also imports
# cleanly (python -m, a test harness, another script), rather than failing on
# find_unidentified_objects depending on how it was invoked.
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _HERE)
_sys.path.insert(0, os.path.join(_ROOT, "literature_crossmatch"))

from common import scores_csv as scores_csv_reader
from common.paths import (
    RESULTS_DIR,
    HF_CACHE_DIR,
    NED_CHECKPOINT_CSV,
    NED_BIBLIO_CHECKPOINT_CSV,
    UNDISCUSSED_CATALOG_CSV,
    CANDIDATES_ALL_CSV,
    CANDIDATES_COUNTS_JSON,
    ORYAN_CATALOGUE_DIR,
    CACHE_DIR,
    PARQUET_PATH,
    FULLTEXT_HITS_CSV,
    FULLTEXT_CHECKPOINT_DONE,
)
from ads_abstracts import (
    load_ads_api_key,
    load_abstract_cache,
    fetch_abstracts,
)
from classify_genuine_discussion import (
    find_name_hits,
    select_papers_for_classification,
    classify_one,
    GEMINI_MODEL,
    GEMINI_WORKERS,
)
from fulltext_search_classification import (
    build_fulltext_query,
    build_bibcode_filter,
    search_fulltext,
    extract_snippets,
    FIELDNAMES as FULLTEXT_FIELDNAMES,
)


# ── CONFIGURATION ──────────────────────────────────────────────────────────

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


MIN_SCORE = int(os.environ.get("CATALOG_MIN_SCORE", "45"))

SCORES_CSV = (
    sys.argv[1] if len(sys.argv) > 1
    else os.path.join(RESULTS_DIR, "gemini_likert_scores.csv")
)
OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else UNDISCUSSED_CATALOG_CSV

CHECKPOINT_DIR = os.path.join(CACHE_DIR, "undiscussed_catalog")
CROSSMATCH_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "crossmatch.json")
VERDICTS_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "verdicts.csv")
# Dedicated NED checkpoints: the stage-2 ones hold results from the chained
# run, which never reached the SIMBAD-matched images this build needs.
NED_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ned_checkpoint.csv")
NED_BIBLIO_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ned_biblio_checkpoint.csv")
ALIASES_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "aliases.json")

# Alias resolution (see fetch_object_aliases).
ALIAS_CHUNK_SIZE = 200
MAX_ALIASES_PER_OBJECT = 25   # bounds the OR-clause count in one ADS query
MIN_ALIAS_LENGTH = 5          # shorter strings are catalogue stubs, not designations

VERDICT_FIELDNAMES = ["object_name", "n_papers_checked", "n_name_hits",
                      "genuinely_discussed", "votes_discussed", "n_votes", "reasoning"]

# Each object is judged CLASSIFY_VOTES times and the majority wins. This is
# insurance, not a fix for a known problem: classification_config pins
# temperature=0.0, and three passes over identical evidence agreed exactly
# here (8 discussed out of 445 every time, no split votes). It is kept because
# a positive is what removes an image from the catalogue, positives are rare
# enough that one flaky call would be visible in the result, and a repeat pass
# over cached evidence costs about 30 seconds and no ADS quota. The split-vote
# count printed at the end is the diagnostic worth watching.
CLASSIFY_VOTES = int(os.environ.get("CLASSIFY_VOTES", "3"))

OUTPUT_FIELDNAMES = [
    "filename", "imagescore", "SourceRA", "SourceDec",
    "catalogued", "simbad_name", "simbad_type", "ned_name", "ned_type",
    "redshift", "n_objects", "n_papers", "n_fulltext_hits", "discussion_checked",
]


# ── 1. CONDITIONS 1 AND 2: SCORE, AND NOT AN ANOMALYMATCH ANOMALY ──────────

def load_candidates(scores_csv, min_score):
    """
    Rows scoring at least min_score that are not labelled anomalies, joined
    to their sky position.

    Returns (records, selection_counts); the counts describe the cut before
    any cross-match and are what the appendix quotes.

    The scoring CSV carries no coordinates, so RA/Dec come from the image
    catalogue parquet, keyed by SourceID. Reading goes through
    common/scores_csv.py, so either header spelling works and an
    unrecognised one raises instead of yielding an empty candidate list.
    """
    import pandas as pd

    rows = []
    for row in scores_csv_reader.read(scores_csv).rows:
        try:
            score = int(float(row["imagescore"]))
        except (TypeError, ValueError):
            continue
        rows.append((row["filename"], score, str(row["interesting"] or "0").strip()))

    total = len(rows)
    by_score = [r for r in rows if r[1] >= min_score]
    kept = [r for r in by_score if r[2] != "1"]
    print(f"{total} scored images")
    print(f"  condition 1  imagescore >= {min_score}: {len(by_score)}")
    print(f"  condition 2  not an AnomalyMatch anomaly: {len(kept)} "
          f"({len(by_score) - len(kept)} dropped)")

    # Only the surviving handful of positions are needed, so filter the
    # 223,195-row catalogue down to them before materialising anything -
    # building a dict of the whole thing to look up ~140 keys is pure waste.
    wanted = {str(f) for f, _, _ in kept}
    coords = pd.read_parquet(PARQUET_PATH, columns=["SourceID", "SourceRA", "SourceDec"])
    coords["SourceID"] = coords["SourceID"].astype(str)
    coords = coords[coords["SourceID"].isin(wanted)].set_index("SourceID")

    records, missing = [], 0
    for filename, score, _ in kept:
        if str(filename) not in coords.index:
            missing += 1
            continue
        row = coords.loc[str(filename)]
        records.append({
            "filename": str(filename),
            "imagescore": score,
            # cross_match_ned orders its queue by avg_score.
            "avg_score": float(score),
            "SourceRA": float(row["SourceRA"]),
            "SourceDec": float(row["SourceDec"]),
        })
    if missing:
        print(f"  WARNING: {missing} images have no parquet coordinates and were dropped")
    print(f"  {len(records)} candidates carry coordinates")
    return records, dict(scored=total, above_threshold=len(by_score),
                         nonreference=len(kept))


# ── 2A-i. THE HF GALAXY-MENTIONS CATALOGUES ────────────────────────────────

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


# ── 2B. CONDITION 3 + PAPER COLLECTION ─────────────────────────────────────

def _dropped_row(record, why):
    """What an image condition 3 removed still knows about itself."""
    return {
        "filename": record["filename"],
        "imagescore": record["imagescore"],
        "SourceRA": record["SourceRA"],
        "SourceDec": record["SourceDec"],
        "dropped_by": why,
    }


def run_crossmatch(records):
    """
    Apply condition 3 - drop images already claimed by the galaxy-mentions or
    O'Ryan catalogues - then query SIMBAD and NED for every survivor purely to
    collect papers.

    Condition 3 runs before the expensive part on purpose: an image dropped
    here costs no ADS quota and no LLM call.

    Returns (images, papers_by_object, dropped) where each image carries its
    matched objects, papers_by_object maps an object name to its bibliography,
    and dropped maps a filename to the catalogue that removed it.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dropped = {}

    # -- condition 3a: HF galaxy-mentions. A hit means a paper already
    #    resolved a name to this position.
    download_parquet(COORD_RESOLUTION_PARQUET_URL, COORD_RESOLUTION_PARQUET_PATH)
    download_parquet(GALAXY_MENTIONS_PARQUET_URL, GALAXY_MENTIONS_PARQUET_PATH)
    resolved = load_resolved_catalog(COORD_RESOLUTION_PARQUET_PATH)
    mentions = load_mentions_lookup(GALAXY_MENTIONS_PARQUET_PATH)

    survivors, hf_matched = cross_match_hf(records, resolved, mentions)
    for r in hf_matched:
        dropped[r["filename"]] = _dropped_row(r, "galaxy-mentions")
    print(f"  condition 3a no galaxy-mentions counterpart: {len(survivors)} "
          f"({len(hf_matched)} dropped)")

    # -- condition 3b: the O'Ryan interacting-galaxy catalogues. A hit means
    #    the morphology was already recognised, so the image is not a new
    #    candidate.
    survivors, oryan_matched = screen_oryan(survivors)
    for r in oryan_matched:
        dropped[r["filename"]] = _dropped_row(r, "oryan")
    print(f"  condition 3b no O'Ryan counterpart: {len(survivors)} "
          f"({len(oryan_matched)} dropped)")

    # -- SIMBAD and NED, for papers only. Every survivor goes to both.
    simbad_unmatched, simbad_matched = cross_match_simbad(survivors)
    print(f"  SIMBAD: {len(simbad_matched)} of {len(survivors)} images matched")

    ned_unmatched, ned_matched, _ = cross_match_ned(
        survivors, top_n=None, checkpoint_path=NED_CHECKPOINT
    )
    print(f"  NED:    {len(ned_matched)} of {len(survivors)} images matched")

    simbad_biblio = fetch_simbad_bibliography(simbad_matched)
    ned_biblio = fetch_ned_bibliography(ned_matched, checkpoint_path=NED_BIBLIO_CHECKPOINT)

    # -- fold the two catalogues into one object list per image
    simbad_by_file = {r["filename"]: r for r in simbad_matched}
    ned_by_file = {r["filename"]: r for r in ned_matched}

    papers_by_object = {}
    for row in simbad_biblio:
        papers_by_object.setdefault(row["main_id"], {})[row["bibcode"]] = {
            "bibcode": row["bibcode"], "year": row.get("year", ""),
            "title": row.get("title", ""), "alias": row["main_id"],
        }
    for row in ned_biblio:
        papers_by_object.setdefault(row["object_name"], {})[row["bibcode"]] = {
            "bibcode": row["bibcode"], "year": row.get("year", ""),
            "title": row.get("title", ""), "alias": row["object_name"],
        }
    papers_by_object = {k: list(v.values()) for k, v in papers_by_object.items()}

    images = []
    for r in survivors:
        fn = r["filename"]
        s, n = simbad_by_file.get(fn), ned_by_file.get(fn)
        objects = []
        if s:
            objects.append(s["object_name"])
        if n and (not s or n["object_name"] != s["object_name"]):
            objects.append(n["object_name"])
        images.append({
            "filename": fn,
            "imagescore": r["imagescore"],
            "SourceRA": r["SourceRA"],
            "SourceDec": r["SourceDec"],
            "simbad_name": s["object_name"] if s else "",
            "simbad_type": s["object_type"] if s else "",
            "ned_name": n["object_name"] if n else "",
            "ned_type": n.get("object_type", "") if n else "",
            "redshift": n.get("redshift", "") if n else "",
            "objects": objects,
        })

    n_cat = sum(1 for i in images if i["objects"])
    n_pap = sum(1 for i in images if any(papers_by_object.get(o) for o in i["objects"]))
    print(f"  {n_cat}/{len(images)} images have a catalogue counterpart; "
          f"{n_pap} have at least one paper")
    return images, papers_by_object, dropped


# ── 3. ALIASES: EVERY NAME THE LITERATURE MIGHT USE ────────────────────────

def fetch_object_aliases(images):
    """
    object_name -> every designation SIMBAD records at that sky position.

    Searching a paper's body for only SIMBAD's main_id or NED's preferred name
    misses the object whenever the authors used another catalogue's
    designation, and that is the normal case rather than the exception. The
    miss is not harmless: an object discussed under a name we never searched
    looks undiscussed, and its image wrongly stays in the catalogue. The alias
    set is therefore the recall floor of condition 4.

    Resolution is positional rather than by name, because a name lookup
    inherits whatever sparseness that one SIMBAD entry has. The strong lens
    4001415925883 is the worked example: its matched entry, "SLACS SDSS
    J1205+4910", carries exactly one identifier, while the same 3" circle also
    holds "FIRST J120540.4+491029" with SDSS, 2MASX and Gaia designations -
    the names the literature actually prints. All of them describe the source
    in this cutout, so all of them are worth searching for.
    """
    import pyvo
    from astropy.table import Table

    object_names = sorted({o for i in images for o in i["objects"]})
    if os.path.exists(ALIASES_CHECKPOINT):
        with open(ALIASES_CHECKPOINT, encoding="utf-8") as f:
            cached = json.load(f)
        if all(n in cached for n in object_names):
            print(f"  reusing {len(cached)} cached alias sets")
            return cached

    print(f"  resolving designations at {len(images)} positions via SIMBAD...")
    simbad = pyvo.dal.TAPService(SIMBAD_TAP_URL)
    by_filename = {}

    for i in range(0, len(images), ALIAS_CHUNK_SIZE):
        chunk = images[i:i + ALIAS_CHUNK_SIZE]
        query = f"""
        SELECT mine.filename, ident.id AS alias
        FROM TAP_UPLOAD.mine AS mine
        JOIN basic AS b
          ON 1=CONTAINS(POINT('ICRS', b.ra, b.dec),
                        CIRCLE('ICRS', mine.ra, mine.dec, {MATCH_RADIUS_DEG}))
        JOIN ident ON ident.oidref = b.oid
        """
        upload = Table({
            "filename": [r["filename"] for r in chunk],
            "ra": [float(r["SourceRA"]) for r in chunk],
            "dec": [float(r["SourceDec"]) for r in chunk],
        })
        try:
            result = simbad.run_sync(query, uploads={"mine": upload}).to_table()
        except Exception as e:
            print(f"    alias chunk {i}-{i + len(chunk)} failed: {e}")
            continue
        for row in result:
            alias = str(row["alias"]).strip()
            if alias:
                by_filename.setdefault(str(row["filename"]).strip(), set()).add(alias)
        print(f"    {min(i + len(chunk), len(images))}/{len(images)} positions resolved")

    # Every designation found at an image's position applies to each catalogue
    # object matched there - they are names for the one source in the cutout.
    aliases = {}
    for img in images:
        found = by_filename.get(img["filename"], set())
        for name in img["objects"]:
            aliases.setdefault(name, set()).update(found)

    cached = {}
    for name in object_names:
        # Own name first, then distinct designations. Very short strings are
        # catalogue stubs that would match half the literature as a phrase.
        found = aliases.get(name, set())
        keep = sorted(a for a in found if len(a) >= MIN_ALIAS_LENGTH and a != name)
        cached[name] = [name] + keep[:MAX_ALIASES_PER_OBJECT]

    with open(ALIASES_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(cached, f)
    gained = sum(1 for v in cached.values() if len(v) > 1)
    total = sum(len(v) for v in cached.values())
    print(f"  {gained}/{len(cached)} objects gained extra designations "
          f"({total} names total, was {len(cached)})")
    return cached


# ── 4. CONDITION 4: IS ANY MATCHED OBJECT GENUINELY DISCUSSED? ─────────────

def load_fulltext_state():
    """Existing full-text hits and the set of objects already searched."""
    hits = {}
    if os.path.exists(FULLTEXT_HITS_CSV):
        with open(FULLTEXT_HITS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hits.setdefault(row["object_name"], []).append(row)
    checked = set()
    if os.path.exists(FULLTEXT_CHECKPOINT_DONE):
        with open(FULLTEXT_CHECKPOINT_DONE, encoding="utf-8") as f:
            checked = {line.strip() for line in f if line.strip()}
    return hits, checked


def search_missing_fulltext(objects, papers_by_object, hits, checked, ads_api_key, alias_map):
    """
    Ask ADS which of an object's *own* papers mention it in their body text,
    and with what surrounding sentence.

    The query is restricted to the bibcodes SIMBAD and NED already attribute
    to the object, so this can only ever return papers the catalogues linked
    to it. Searching the object's designation across all of ADS instead would
    also turn up papers from other fields that happen to reuse the string -
    full:"ASV 25" matches microbiology papers about amplicon sequence
    variants - and no amount of evidence from such a paper says anything about
    a galaxy.

    Hits are appended to the shared hits CSV and checkpoint.
    """
    todo = [o for o in objects if o not in checked and papers_by_object.get(o)]
    if not todo:
        print("  every candidate object has already been full-text searched")
        return hits
    print(f"  full-text searching {len(todo)} objects not covered by the earlier sweep...")

    is_new = not os.path.exists(FULLTEXT_HITS_CSV)
    hits_file = open(FULLTEXT_HITS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(hits_file)
    if is_new:
        writer.writerow(FULLTEXT_FIELDNAMES)
    checked_file = open(FULLTEXT_CHECKPOINT_DONE, "a", encoding="utf-8")

    n_req = n_hit = 0
    t0 = time.time()
    try:
        for name in todo:
            own_papers = papers_by_object[name]
            aliases = ({name}
                       | {p["alias"] for p in own_papers if p.get("alias")}
                       | set(alias_map.get(name, [])))
            bibcodes = [p["bibcode"] for p in own_papers if p.get("bibcode")]
            query = build_fulltext_query(aliases)
            bibcode_filter = build_bibcode_filter(bibcodes)
            if query is None or bibcode_filter is None:
                checked_file.write(name + "\n")
                continue
            # Ask for every one of its papers back, not the default top 10 -
            # the point is to know which of them carry an in-body mention.
            docs, highlighting, remaining = search_fulltext(
                query, ads_api_key, rows=max(len(bibcodes), 10), fq=bibcode_filter
            )
            n_req += 1
            rows = []
            for d in docs:
                row = {
                    "object_name": name, "bibcode": d["bibcode"],
                    "title": d.get("title", [""])[0], "year": d.get("year", ""),
                    "matched_alias": next(iter(aliases), name),
                    "snippets": extract_snippets(d, highlighting),
                }
                rows.append(row)
                writer.writerow([row[k] for k in FULLTEXT_FIELDNAMES])
            if rows:
                hits.setdefault(name, []).extend(rows)
                n_hit += 1
            checked_file.write(name + "\n")
            if n_req % 50 == 0:
                hits_file.flush(); checked_file.flush()
                print(f"    {n_req}/{len(todo)} searched, {n_hit} with hits, "
                      f"{time.time() - t0:.0f}s, ADS remaining: {remaining}")
    finally:
        hits_file.close(); checked_file.close()

    print(f"  full-text search done: {n_req} ADS requests, {n_hit} objects with hits")
    return hits


def load_verdict_checkpoint():
    if not os.path.exists(VERDICTS_CHECKPOINT):
        return {}
    with open(VERDICTS_CHECKPOINT, newline="", encoding="utf-8") as f:
        return {r["object_name"]: r for r in csv.DictReader(f)}


def save_verdicts(verdicts):
    tmp = VERDICTS_CHECKPOINT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VERDICT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(verdicts.values())
    os.replace(tmp, VERDICTS_CHECKPOINT)


def classify_objects(objects, papers_by_object, hits, gemini_api_key, ads_api_key, alias_map):
    """Verdict per object, using abstracts plus in-body snippets."""
    verdicts = load_verdict_checkpoint()
    todo = [o for o in objects if o not in verdicts and papers_by_object.get(o)]
    if verdicts:
        print(f"  {len(verdicts)} verdicts already cached from an earlier run")
    if not todo:
        return verdicts

    # Papers the full-text search turned up that the catalogue join missed.
    for name in todo:
        known = {p["bibcode"] for p in papers_by_object[name]}
        for row in hits.get(name, []):
            if row["bibcode"] not in known:
                papers_by_object[name].append({
                    "bibcode": row["bibcode"], "year": row.get("year", ""),
                    "title": row.get("title", ""), "alias": row.get("matched_alias", ""),
                })
                known.add(row["bibcode"])

    all_bibcodes = [p["bibcode"] for o in todo for p in papers_by_object[o]]
    print(f"  fetching abstracts for {len(set(all_bibcodes))} unique bibcodes...")
    cache = fetch_abstracts(all_bibcodes, ads_api_key, load_abstract_cache())

    for name in todo:
        snippets = {r["bibcode"]: r["snippets"] for r in hits.get(name, [])
                    if (r.get("snippets") or "").strip()}
        for p in papers_by_object[name]:
            entry = cache.get(p["bibcode"], {})
            p["_title"] = entry.get("title") or p.get("title", "")
            p["_abstract"] = entry.get("abstract", "")
            p["_snippets"] = snippets.get(p["bibcode"], "")

    print(f"  classifying {len(todo)} objects with Gemini ({GEMINI_MODEL}, "
          f"{GEMINI_WORKERS} workers, best of {CLASSIFY_VOTES})...")

    # Evidence is fixed per object; only the model call is repeated.
    evidence = {}
    for name in todo:
        papers = papers_by_object[name]
        aliases = ({name} | {p["alias"] for p in papers if p.get("alias")}
                   | set(alias_map.get(name, [])))
        hit_bibcodes = (find_name_hits(papers, aliases)
                        | {r["bibcode"] for r in hits.get(name, [])})
        evidence[name] = (select_papers_for_classification(papers, hit_bibcodes), hit_bibcodes)

    tally = {name: [] for name in todo}
    reasons = {}
    t0 = time.time()
    for round_i in range(1, CLASSIFY_VOTES + 1):
        n_done = 0
        with ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as ex:
            futures = {
                ex.submit(classify_one, name, evidence[name][0], evidence[name][1],
                          gemini_api_key): name
                for name in todo
            }
            for future in as_completed(futures):
                name = futures[future]
                object_name, data, error = future.result()
                n_done += 1
                if error:
                    print(f"    FAILED {object_name}: {error}")
                    continue
                tally[object_name].append(bool(data.genuinely_discussed))
                # Keep a reason that matches the majority side where possible.
                if data.genuinely_discussed or object_name not in reasons:
                    reasons[object_name] = data.reasoning
        n_pos = sum(1 for v in tally.values() if v and sum(v) > len(v) / 2)
        print(f"    vote {round_i}/{CLASSIFY_VOTES}: {n_done} judged, "
              f"{sum(1 for v in tally.values() if v and v[-1])} discussed this pass, "
              f"running majority {n_pos}, {time.time() - t0:.0f}s")

    for name, votes in tally.items():
        if not votes:
            continue
        selected, hit_bibcodes = evidence[name]
        verdicts[name] = {
            "object_name": name,
            "n_papers_checked": len(selected),
            "n_name_hits": len(hit_bibcodes),
            "genuinely_discussed": sum(votes) > len(votes) / 2,
            "votes_discussed": sum(votes),
            "n_votes": len(votes),
            "reasoning": reasons.get(name, ""),
        }
    save_verdicts(verdicts)

    split = sum(1 for v in tally.values() if v and 0 < sum(v) < len(v))
    print(f"  classification done in {time.time() - t0:.0f}s; "
          f"{split} objects had a split vote (decided by majority)")
    return verdicts


def is_discussed(verdict_row):
    return str(verdict_row.get("genuinely_discussed", "")).strip().lower() == "true"


# ── 5. CONDITION 3b: THE O'RYAN INTERACTING-GALAXY CATALOGUES ──────────────

def load_oryan_catalogues(directory):
    """Every per-catalogue CSV in `directory`, concatenated. (None, []) if empty."""
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not paths:
        return None, []
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = {"SourceID", "RA", "Dec"} - set(frame.columns)
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        frames.append(frame.assign(catalogue=os.path.basename(path)))
    return pd.concat(frames, ignore_index=True), [os.path.basename(p) for p in paths]


def separation_arcsec(images, catalogue):
    """
    Great-circle separation in arcseconds between every image and every
    catalogue row, as an (n_images, n_catalogue) matrix.

    Positions are matched rather than identifiers, so a source is found under
    whatever designation each catalogue happens to use for it.
    """
    ra1 = np.deg2rad(np.array([float(i["SourceRA"]) for i in images]))[:, None]
    dec1 = np.deg2rad(np.array([float(i["SourceDec"]) for i in images]))[:, None]
    ra2 = np.deg2rad(catalogue.RA.to_numpy())[None, :]
    dec2 = np.deg2rad(catalogue.Dec.to_numpy())[None, :]
    haversine = (np.sin((dec2 - dec1) / 2) ** 2
                 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2)
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))) * 3600


def screen_oryan(images, radius_arcsec=MATCH_RADIUS_ARCSEC):
    """
    Split `images` into (survivors, matched) against the O'Ryan catalogues.

    A match means the interacting morphology was already recognised, so the
    image is not a new candidate and is dropped.

    A missing catalogue directory stops the run. It is a missing input, not an
    empty result, and silently releasing images the screen should have removed
    would be worse than failing.
    """
    catalogue, names = load_oryan_catalogues(ORYAN_CATALOGUE_DIR)
    if catalogue is None:
        raise SystemExit(
            f"No catalogue CSVs in {ORYAN_CATALOGUE_DIR}\n"
            "Condition 3b needs the O'Ryan et al. (2023) catalogues "
            "(Zenodo 7684876).\n"
            "Set ORYAN_CATALOGUE_DIR to the directory holding them."
        )
    if not images:
        return [], []
    print(f"    {len(names)} O'Ryan catalogues, {len(catalogue):,} rows")
    hit_any = (separation_arcsec(images, catalogue) <= radius_arcsec).any(axis=1)
    return ([img for img, hit in zip(images, hit_any) if not hit],
            [img for img, hit in zip(images, hit_any) if hit])


# ── 6. MAIN ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not set in the environment.")
    ads_api_key = load_ads_api_key()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Scores: {SCORES_CSV}")
    print(f"Match radius: {MATCH_RADIUS_ARCSEC}\"\n")

    records, selection = load_candidates(SCORES_CSV, MIN_SCORE)
    n_scored = selection["scored"]
    n_above_threshold = selection["above_threshold"]
    n_nonreference = selection["nonreference"]

    # The checkpoint records the cut it was built at. Without that, lowering
    # or raising CATALOG_MIN_SCORE and re-running would silently reuse the old
    # selection and report counts for a threshold nobody asked for.
    state = None
    if os.path.exists(CROSSMATCH_CHECKPOINT):
        with open(CROSSMATCH_CHECKPOINT, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("min_score") != MIN_SCORE:
            print(f"\nIgnoring cross-match checkpoint: built at score >= "
                  f"{state.get('min_score', 'unknown')}, this run is >= {MIN_SCORE}")
            state = None

    if state is not None:
        print(f"\nReusing cross-match checkpoint {CROSSMATCH_CHECKPOINT}")
        images = state["images"]
        papers_by_object = state["papers_by_object"]
        dropped_early = state.get("dropped_early", {})
    else:
        print("\nCross-matching...")
        images, papers_by_object, dropped_early = run_crossmatch(records)
        with open(CROSSMATCH_CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump({"min_score": MIN_SCORE, "images": images,
                       "papers_by_object": papers_by_object,
                       "dropped_early": dropped_early}, f)
        print(f"  checkpoint written to {CROSSMATCH_CHECKPOINT}")

    candidate_objects = sorted({o for i in images for o in i["objects"]})
    with_papers = [o for o in candidate_objects if papers_by_object.get(o)]
    print(f"\n{len(candidate_objects)} distinct catalogue objects, "
          f"{len(with_papers)} with at least one paper")

    print("\nCondition 4: genuine-discussion check")
    alias_map = fetch_object_aliases(images)
    hits, checked = load_fulltext_state()
    hits = search_missing_fulltext(with_papers, papers_by_object, hits, checked,
                                   ads_api_key, alias_map)
    verdicts = classify_objects(with_papers, papers_by_object, hits,
                                gemini_api_key, ads_api_key, alias_map)

    discussed = {o for o in candidate_objects
                 if o in verdicts and is_discussed(verdicts[o])}
    print(f"  {len(discussed)} of {len(with_papers)} objects are genuinely discussed")

    # -- assemble the catalogue
    rows, kept = [], []
    for img in images:
        n_papers = sum(len(papers_by_object.get(o, [])) for o in img["objects"])
        row = {
            "filename": img["filename"],
            "imagescore": img["imagescore"],
            "SourceRA": img["SourceRA"],
            "SourceDec": img["SourceDec"],
            "catalogued": 1 if img["objects"] else 0,
            "simbad_name": img["simbad_name"],
            "simbad_type": img["simbad_type"],
            "ned_name": img["ned_name"],
            "ned_type": img["ned_type"],
            "redshift": img["redshift"],
            "n_objects": len(img["objects"]),
            "n_papers": n_papers,
            "n_fulltext_hits": sum(len(hits.get(o, [])) for o in img["objects"]),
            "discussion_checked": 1 if n_papers else 0,
        }
        row["dropped_by"] = "discussed" if any(o in discussed for o in img["objects"]) else ""
        rows.append(row)
        if not row["dropped_by"]:
            kept.append({k: row[k] for k in OUTPUT_FIELDNAMES})

    order = lambda r: (-r["imagescore"], r["filename"])
    rows.sort(key=order)
    kept.sort(key=order)

    def write(path, records, fieldnames):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)

    write(OUTPUT_CSV, kept, OUTPUT_FIELDNAMES)
    # Images condition 3 removed never reached the cross-match, so they carry
    # only what the score CSV and the parquet knew about them.
    rows.extend(dropped_early.values())
    rows.sort(key=lambda r: (-int(r.get("imagescore") or 0), r["filename"]))
    write(CANDIDATES_ALL_CSV, rows, OUTPUT_FIELDNAMES + ["dropped_by"])

    n_cat = sum(1 for r in kept if r["catalogued"])
    # Count only what condition 4 removed; `rows` now also holds the images
    # condition 3 dropped before the cross-match.
    n_dropped = sum(1 for r in rows if r.get("dropped_by") == "discussed")
    counts = {
        "score_threshold": MIN_SCORE,
        "matching_radius_arcsec": MATCH_RADIUS_ARCSEC,
        "scored_images": n_scored,
        "all_images_above_threshold": n_above_threshold,
        "reference_images_above_threshold": n_above_threshold - n_nonreference,
        "nonreference_above_threshold": n_nonreference,
        "entered_discussion_screen": len(images),
        "distinct_catalogue_objects": len(candidate_objects),
        "objects_with_papers": len(with_papers),
        "objects_genuinely_discussed": len(discussed),
        "dropped_by_discussion_screen": n_dropped,
        "released_candidates": len(kept),
        "released_with_papers_assessed": sum(1 for r in kept if r["discussion_checked"]),
        "released_without_papers": sum(1 for r in kept if not r["discussion_checked"]),
        "released_uncatalogued": len(kept) - n_cat,
        "released_catalogued_not_discussed": n_cat,
        "dropped_by_galaxy_mentions": sum(1 for r in dropped_early.values()
                                          if r["dropped_by"] == "galaxy-mentions"),
        "dropped_by_oryan": sum(1 for r in dropped_early.values()
                                if r["dropped_by"] == "oryan"),
    }
    with open(CANDIDATES_COUNTS_JSON, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
        f.write("\n")

    print(f"\n{'='*66}")
    print(f"Wrote {len(kept)} released candidates to {OUTPUT_CSV}")
    print(f"      {len(rows)} screened images to {CANDIDATES_ALL_CSV}")
    print(f"      counts to {CANDIDATES_COUNTS_JSON}")
    print()
    print(json.dumps(counts, indent=2))
