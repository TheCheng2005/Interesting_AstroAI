"""
Every filesystem location the pipeline reads or writes, in one place.

Stages used to resolve data with a mix of `os.path.join(SCRIPT_DIR, ...)`
and bare "../hubble_data/..." strings. The latter resolved against the
working directory, so a stage only worked when launched from one particular
folder and silently wrote its outputs wherever it happened to be started -
which broke outright once the scripts moved one level down into stage
folders. Everything here is anchored to this file's location instead, so any
stage can be run from anywhere.

Layout (this file sits in common/):

    analysis/
        common/                 this module, shared by every stage
        scoring/                stage 1 - the AI-judging protocols
        results/                stage 1 output
            subset_test/            pilot runs, one CSV per model x protocol
            full_catalog/           full-scale production runs
        unidentified_objects/   stage 2 - catalogue cross-match
        literature_crossmatch/  stage 3 - is it genuinely discussed?
        reports/                stage 4 - HTML dashboards and figures
        poster/                 poster figures
        sample_data/            small samples so schemas are inspectable
        cache/                  rebuildable or resumable state, NOT tracked

Because the stage folders are siblings, a script imports this module with:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.paths import ...
"""

import os

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_ROOT = os.path.dirname(COMMON_DIR)


# ── INPUTS (not in this repository - see README) ───────────────────────────
#
# The image catalogue is multiple GB and is not redistributed here. By default
# it is looked for in a hubble_data/ folder next to the analysis tree; set
# HUBBLE_DATA_DIR to point somewhere else.

DATA_DIR = os.environ.get(
    "HUBBLE_DATA_DIR",
    os.path.join(os.path.dirname(ANALYSIS_ROOT), "hubble_data"),
)
HDF5_PATH = os.path.join(DATA_DIR, "10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5")
PARQUET_PATH = os.path.join(DATA_DIR, "10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.parquet")
INTERESTING_CSV_PATH = os.path.join(DATA_DIR, "Interesting.csv")


# ── STAGE 1: MODEL SCORING ─────────────────────────────────────────────────

SCORING_DIR = os.path.join(ANALYSIS_ROOT, "scoring")

# The role/taxonomy prompt prepended to every scoring request.
GEMINI_MD_PATH = os.path.join(SCORING_DIR, "GEMINI.md")

RESULTS_DIR = os.path.join(ANALYSIS_ROOT, "results")

# Pilot runs, named {provider}_{protocol}_{run}.csv so the dashboard groups
# replicates by (provider, protocol) automatically.
SUBSET_DIR = os.path.join(RESULTS_DIR, "subset_test")

# Whole-catalogue runs (TEST_MODE = False), named per seed and timestamp.
FULL_RUN_DIR = os.path.join(RESULTS_DIR, "full_catalog")


# ── STAGE 2: CATALOGUE CROSS-MATCH ─────────────────────────────────────────

UNIDENTIFIED_DIR = os.path.join(ANALYSIS_ROOT, "unidentified_objects")
UNIDENTIFIED_CSV = os.path.join(UNIDENTIFIED_DIR, "unidentified_objects.csv")
UNIDENTIFIED_REPORT_HTML = os.path.join(UNIDENTIFIED_DIR, "unidentified_objects_report.html")


# ── STAGE 3: LITERATURE CLASSIFICATION ─────────────────────────────────────
#
# The matched objects and their bibliographies are written by stage 2 but read
# almost exclusively by stage 3, so they live with the stage that consumes them.

LITERATURE_DIR = os.path.join(ANALYSIS_ROOT, "literature_crossmatch")
MATCHED_CSV = os.path.join(LITERATURE_DIR, "matched_objects.csv")
SIMBAD_BIBLIOGRAPHY_CSV = os.path.join(LITERATURE_DIR, "simbad_bibliography.csv")
NED_BIBLIOGRAPHY_CSV = os.path.join(LITERATURE_DIR, "ned_bibliography.csv")
CLASSIFICATION_CSV = os.path.join(LITERATURE_DIR, "discussion_classification.csv")
FULLTEXT_HITS_CSV = os.path.join(LITERATURE_DIR, "fulltext_hits.csv")
DEEP_DIVE_CSV = os.path.join(LITERATURE_DIR, "deep_dive_summaries.csv")
DEEP_DIVE_JSON = os.path.join(LITERATURE_DIR, "deep_dive_summaries.json")

# Tracked, unlike the other checkpoints: it records which objects the ADS
# full-text quota has already been spent on, which is worth days of quota.
FULLTEXT_CHECKPOINT_DONE = os.path.join(LITERATURE_DIR, "fulltext_search_checked.txt")


# ── STAGE 4: REPORTS AND FIGURES ───────────────────────────────────────────

REPORTS_DIR = os.path.join(ANALYSIS_ROOT, "reports")
SCORING_REPORT_HTML = os.path.join(REPORTS_DIR, "hsc_report_mixed.html")
PLOTS_ONLY_HTML = os.path.join(REPORTS_DIR, "plots_only.html")
TIER_REPORT_HTML = os.path.join(REPORTS_DIR, "score40_tiers.html")

POSTER_DIR = os.path.join(ANALYSIS_ROOT, "poster")


# ── CACHE AND RESUMABLE STATE (gitignored) ─────────────────────────────────
#
# Everything here can be rebuilt: re-downloaded from its source, or re-earned
# by spending API quota again. None of it is a pipeline result.

CACHE_DIR = os.path.join(ANALYSIS_ROOT, "cache")
ADS_ABSTRACT_CACHE_PATH = os.path.join(CACHE_DIR, "ads_abstract_cache.json")
HF_CACHE_DIR = os.path.join(CACHE_DIR, "hf_cache")
NED_CHECKPOINT_CSV = os.path.join(CACHE_DIR, "ned_checkpoint.csv")
NED_BIBLIO_CHECKPOINT_CSV = os.path.join(CACHE_DIR, "ned_biblio_checkpoint.csv")

# The ADS API token. Kept outside the repository on purpose - never commit it.
ADS_API_KEY_PATH = os.path.join(os.path.expanduser("~"), ".ads_api_key")


def ensure_dirs():
    """Create every output directory. Safe to call repeatedly."""
    for d in (
        SUBSET_DIR, FULL_RUN_DIR, UNIDENTIFIED_DIR, LITERATURE_DIR,
        REPORTS_DIR, CACHE_DIR, HF_CACHE_DIR,
    ):
        os.makedirs(d, exist_ok=True)


# Run on import: cache/ is gitignored, so it is absent in a fresh clone and
# the first stage to write there would otherwise fail.
ensure_dirs()
