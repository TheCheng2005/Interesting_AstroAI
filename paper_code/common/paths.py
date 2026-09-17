"""
Every filesystem location the paper's scripts read or write.

Paths are anchored to this file, not to the working directory, so any script
runs correctly from anywhere. Layout (this file sits in common/):

    paper_code/
        common/                 this module, shared by every stage
        dataset/                stage 0 - build the cutout catalogue
        scoring/                stage 1 - the VLM scoring protocols
        unidentified_objects/   stage 2A - find the papers a catalogue
                                attributes to each position, and stage 2B -
                                judge whether any of them discuss the object
        literature_crossmatch/  the ADS and classifier helpers 2B uses
        metrics/                stage 3 - the numbers printed in the paper

Because the stage folders are siblings, a script imports this module with:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.paths import ...

This package is code only. Both the inputs and the results live outside it
and are located through two environment variables:

    HUBBLE_DATA_DIR    the cutout catalogue and the AnomalyMatch labels
                       (default: hubble_data/ next to this tree)
    PAPER_RESULTS_DIR  the root under which result CSVs are read and written
                       (default: this tree, so a fresh run is self-contained)

Point PAPER_RESULTS_DIR at the analysis tree of the results repository to
recompute the published numbers from the published CSVs; leave it unset to
have stage 2 write a fresh set here.
"""

import os

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_ROOT = os.path.dirname(COMMON_DIR)


# ── INPUTS (not redistributed with this code) ──────────────────────────────
#
# The 223,195-cutout catalogue is ~3 GB. Only the scoring stage needs it;
# stages 2 and 3 run from CSVs alone. Build it with
#
#     python dataset/pipeline.py --output-dir $HUBBLE_DATA_DIR
#
# which writes both files below under the stem it hardcodes.

DATA_DIR = os.environ.get(
    "HUBBLE_DATA_DIR",
    os.path.join(os.path.dirname(ANALYSIS_ROOT), "hubble_data"),
)
HDF5_PATH = os.path.join(DATA_DIR, "10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5")
PARQUET_PATH = os.path.join(DATA_DIR, "10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.parquet")

# The interacting-galaxy catalogues of O'Ryan et al. (2023), Zenodo 7684876.
# A directory of per-catalogue CSVs, each with SourceID / RA / Dec columns;
# "interacting-catalogue.csv" is the interacting sample itself, the rest are
# the comparison catalogues distributed with it.
ORYAN_CATALOGUE_DIR = os.environ.get(
    "ORYAN_CATALOGUE_DIR",
    os.path.join(DATA_DIR, "zenodo_7684876", "catalogues"),
)

# The AnomalyMatch anomaly positions (Gomez et al. 2025). The `interesting`
# ground-truth column in every results CSV comes from matching against this
# at 3"; see scoring/update_interesting_radius.py.
INTERESTING_CSV_PATH = os.path.join(DATA_DIR, "Interesting.csv")


# ── RESULTS (read and written outside this package) ────────────────────────

RESULTS_ROOT = os.environ.get("PAPER_RESULTS_DIR", ANALYSIS_ROOT)


# ── STAGE 1: MODEL SCORING ─────────────────────────────────────────────────

SCORING_DIR = os.path.join(ANALYSIS_ROOT, "scoring")

# The role/taxonomy prompt prepended to every scoring request (Appendix A).
GEMINI_MD_PATH = os.path.join(SCORING_DIR, "GEMINI.md")

RESULTS_DIR = os.path.join(RESULTS_ROOT, "results")

# Named {provider}_{protocol}_{run}.csv so replicates group automatically.
SUBSET_DIR = os.path.join(RESULTS_DIR, "subset_test")

# Whole-catalogue runs (TEST_MODE = False). Not used by the paper.
FULL_RUN_DIR = os.path.join(RESULTS_DIR, "full_catalog")


# ── STAGE 2: CATALOGUE CROSS-MATCH ─────────────────────────────────────────

UNIDENTIFIED_DIR = os.path.join(RESULTS_ROOT, "unidentified_objects")
UNIDENTIFIED_CSV = os.path.join(UNIDENTIFIED_DIR, "unidentified_objects.csv")

# The released candidate catalogue, the full screened selection behind it
# (including the images the discussion screen removed), and the counts the
# appendix quotes. All three are written by build_undiscussed_catalog.py.
UNDISCUSSED_CATALOG_CSV = os.path.join(UNIDENTIFIED_DIR, "undiscussed_catalog.csv")
CANDIDATES_ALL_CSV = os.path.join(UNIDENTIFIED_DIR, "all_nonreference_above_threshold.csv")
CANDIDATES_COUNTS_JSON = os.path.join(UNIDENTIFIED_DIR, "candidate_counts.json")

# Written by a standalone stage-2A sweep, and read by its own corpus-wide
# tools, so they live together.
LITERATURE_DIR = os.path.join(RESULTS_ROOT, "literature_crossmatch")
MATCHED_CSV = os.path.join(LITERATURE_DIR, "matched_objects.csv")
SIMBAD_BIBLIOGRAPHY_CSV = os.path.join(LITERATURE_DIR, "simbad_bibliography.csv")
NED_BIBLIOGRAPHY_CSV = os.path.join(LITERATURE_DIR, "ned_bibliography.csv")


# ── STAGE 2b: LITERATURE DISCUSSION ────────────────────────────────────────
#
# Outputs of the genuine-discussion pass over the papers SIMBAD and NED
# attribute to each matched object.

CLASSIFICATION_CSV = os.path.join(LITERATURE_DIR, "discussion_classification.csv")
FULLTEXT_HITS_CSV = os.path.join(LITERATURE_DIR, "fulltext_hits.csv")
DEEP_DIVE_CSV = os.path.join(LITERATURE_DIR, "deep_dive_summaries.csv")
DEEP_DIVE_JSON = os.path.join(LITERATURE_DIR, "deep_dive_summaries.json")

# Tracked alongside the results, unlike the other checkpoints: it records
# which objects the ADS full-text quota has already been spent on, which is
# worth days of quota at 5,000 requests/day.
FULLTEXT_CHECKPOINT_DONE = os.path.join(LITERATURE_DIR, "fulltext_search_checked.txt")


# ── CACHE AND RESUMABLE STATE ──────────────────────────────────────────────
#
# Everything here can be rebuilt: re-downloaded from its source, or re-earned
# by re-querying a public service. None of it is a result.

CACHE_DIR = os.path.join(RESULTS_ROOT, "cache")
HF_CACHE_DIR = os.path.join(CACHE_DIR, "hf_cache")
NED_CHECKPOINT_CSV = os.path.join(CACHE_DIR, "ned_checkpoint.csv")
NED_BIBLIO_CHECKPOINT_CSV = os.path.join(CACHE_DIR, "ned_biblio_checkpoint.csv")
ADS_ABSTRACT_CACHE_PATH = os.path.join(CACHE_DIR, "ads_abstract_cache.json")

# The ADS API token. Kept outside the repository on purpose - never commit it.
ADS_API_KEY_PATH = os.path.join(os.path.expanduser("~"), ".ads_api_key")


def ensure_dirs():
    """
    Create every output directory. Safe to call repeatedly.

    Only directories that are written to are created; nothing here creates
    DATA_DIR, whose absence should surface as a missing-input error rather
    than as an empty folder.
    """
    for d in (
        SUBSET_DIR, FULL_RUN_DIR, UNIDENTIFIED_DIR, LITERATURE_DIR,
        CACHE_DIR, HF_CACHE_DIR,
    ):
        os.makedirs(d, exist_ok=True)


ensure_dirs()
