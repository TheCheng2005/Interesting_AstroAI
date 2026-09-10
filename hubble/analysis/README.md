# Hubble / analysis

Can an AI judge triage ~10 million Hubble/ACS cutouts down to a tractable set of
scientifically interesting candidates — and are the survivors actually *unknown*?

The pipeline runs in four stages: **score** every image with a vision LLM →
**cross-match** the top scorers against catalogues → ask whether a catalogue hit
means the object is genuinely **discussed** in the literature → turn it all into
**reports**.

The headline output is a shortlist of high-scoring cutouts that no catalogue
records and no paper discusses, plus the evidence trail behind every verdict.

```
   ../hubble_data/*.hdf5                      (multiple GB, not in this repo)
              │
   ┌──────────▼───────────┐
   │ 1  scoring/          │  gemini_likert.py, gemini_tournament.py, …
   └──────────┬───────────┘  → results/subset_test/, results/full_catalog/
              │  three Likert replicates, averaged per image
   ┌──────────▼───────────┐
   │ 2  unidentified_     │  find_unidentified_objects.py
   │    objects/          │  HF galaxy-mentions → SIMBAD → NED
   └──────────┬───────────┘  → unidentified_objects.csv, matched_objects.csv
              │  matched objects + their per-paper bibliographies
   ┌──────────▼───────────┐
   │ 3  literature_       │  classify_genuine_discussion.py
   │    crossmatch/       │  fulltext_search_classification.py
   │                      │  reclassify_with_fulltext.py
   └──────────┬───────────┘  → discussion_classification.csv
              │
   ┌──────────▼───────────┐
   │ 4  reports/, poster/ │  self-contained HTML + paper figures
   └──────────────────────┘
```

---

## `common/`

`paths.py` — every file location the pipeline reads or writes, in one place.

Paths are anchored to that file, **not** to the working directory, so any stage
runs correctly from anywhere. (This replaces the old `../hubble_data/...`
strings, which broke as soon as scripts moved into stage folders.) Because the
stage folders are siblings, each script reaches it with a two-line bootstrap:

```python
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HDF5_PATH, SUBSET_DIR
```

The image catalogue is expected in `hubble_data/` next to this tree; set
`HUBBLE_DATA_DIR` to point elsewhere.

## `scoring/` → `results/`

Eight protocol scripts, `{gemini,qwen}_{likert,tournament,single_elim,hybrid}.py`,
plus the shared prompt. Each loads cutouts from the HDF5, labels an image
*interesting* when it is the nearest image (within 3″) to an `Interesting.csv`
entry, and asks the model to score it — one CSV per random seed.

- [`GEMINI.md`](scoring/GEMINI.md) — the role, taxonomy and output schema given
  to the model: flag rare/anomalous/energetic phenomena (lenses, mergers,
  jellyfish galaxies, AGN…) and aggressively reject imaging artifacts (cosmic
  rays, diffraction spikes, satellite trails).

The protocols are **not** interchangeable:

| Protocol | What the model sees | Score meaning |
|---|---|---|
| **Likert** | 4×4 grid, 10 rounds | Summed 1–5 ratings, **0–50** |
| **Tournament** | 2×2 grid, iterative retention | Rounds survived |
| **Single-elim** | 2×2 grid, one pass | Rounds survived |
| **Hybrid** | one cheap filter round, then Likert on survivors | Combined — used for the full-catalog runs, because full Likert over 10M images was too expensive |

`TEST_MODE = True` (the default) scores a 20,000-image subset — every labelled
interesting image plus a random fill of boring ones — into
`results/subset_test/{provider}_{protocol}_{run}.csv`. With `TEST_MODE = False`
the whole catalogue is scored into `results/full_catalog/`.

> **Only the Likert runs feed stage 2.** `find_unidentified_objects.py` reads
> `gemini_likert_{1,2,3}.csv` specifically. The other protocols exist for the
> method comparison in `reports/plots_only_html.py` and `reports/build_fig1.py`.

`update_interesting_radius.py` re-labels an existing results CSV under a tighter
match radius without re-running (and re-paying for) the scoring.
`reports/examples/` shows exactly what one API call is given: a 4×4 Likert grid
and a 2×2 tournament quad.

## `unidentified_objects/`

`find_unidentified_objects.py` averages each image's score across the three
Likert replicates, then cross-matches at 3″ against three sources **in order**:

1. **HF `astronolan/galaxy-mentions`** — literature-mention derived coordinates
2. **SIMBAD** — one bulk TAP table-upload per chunk
3. **NED** — one object at a time (its TAP service has no bulk upload), checkpointed

Images matched by nothing land in `unidentified_objects.csv`. Everything else
goes to `matched_objects.csv` with object type, redshift and paper count — plus
`simbad_bibliography.csv` / `ned_bibliography.csv`, the per-paper bibliographies
that make stage 3 possible. Those three live in `literature_crossmatch/`, with
the stage that consumes them.

`generate_unidentified_html_report.py` builds the browsable report, applying the
stage-3 verdicts.

## `literature_crossmatch/`

A positional catalogue match is not a discovery. An object can sit in SIMBAD
purely as row 400 of some survey table, with no paper ever saying a word about
it. This stage redefines *identified* as **at least one paper discusses this
object individually**, in three passes that must run in order:

| | Script | What it does |
|---|---|---|
| **3a** | `classify_genuine_discussion.py` | Fetches **all** of an object's abstracts from ADS, name/alias-matches them, asks Gemini for a verdict. 0 papers → `False` with no LLM call. |
| **3b** | `fulltext_search_classification.py` | Searches ADS **full text** (`full:`) and keeps verbatim in-body snippets. This is what separates "one row in Table 3" from "we model its tidal tail in §4" — a distinction abstracts almost never support. |
| **3c** | `reclassify_with_fulltext.py` | Re-judges with the body evidence folded in, working **down the score ranking**. |

`deep_dive_summaries.py` is the shared library all three import (loaders, ADS
fetching, abstract cache); run on its own it also synthesises literature
summaries for the report cards. All of them cache per object, so re-running only
pays for genuinely new work.

**3c reads stage 2 directly.** It takes candidates from `matched_objects.csv`
ordered by `avg_score` (highest first), skips objects with 0 papers and objects
already ruled `True`, and for any candidate the 3b sweep hasn't reached it runs
the ADS full-text query inline — appending to `fulltext_hits.csv` and to
`fulltext_search_checked.txt` so 3b never re-spends that quota.

```bash
RECLASSIFY_MAX_OBJECTS=1000 \
RECLASSIFY_FULLTEXT_BUDGET=3000 \
RECLASSIFY_MIN_SCORE=40 \
python literature_crossmatch/reclassify_with_fulltext.py
```

`fulltext_search_checked.txt` is tracked deliberately: it records which objects
the ADS quota has already been spent on, which is worth days of runtime.

## `reports/` and `poster/`

| Script | Output |
|---|---|
| `plots_only_html.py` | `plots_only.html` — provider/protocol comparison dashboard |
| `generate_html_report.py` | `hsc_report_mixed.html` — scored-image browser |
| `build_tier_report.py` | `score40_tiers.html` — every image ≥40, tiered by how much is known about it |
| `build_fig1.py` | Figure 1 of the workshop paper |
| `poster/poster_*.py` | poster figures |

---

## Setup

```bash
pip install -r ../../requirements.txt   # or: conda activate gemini_env
export JB_API_KEY=...                   # Gemini
echo "<token>" > ~/.ads_api_key         # https://ui.adsabs.harvard.edu → Account Settings → API Token
```

Stages 1, 2 and 4 need the image catalogue, which is **not** redistributed here
(see `sample_data/` for the schema, and the root README for the source papers):

| File | Size | Purpose |
|---|---|---|
| `10m_dedup_..._minsep10p0arcsec.hdf5` | 2.8 GB | the image cutouts |
| `10m_dedup_..._minsep10p0arcsec.parquet` | 13 MB | per-image RA/Dec |
| `Interesting.csv` | 60 KB | the labelled anomalies |

Stage 3 needs none of it — it works entirely from stage-2 CSVs and the ADS API.

### API quota is what actually paces this pipeline

**ADS enforces a hard 5,000 requests/day**, and abstract fetching and full-text
search draw on the same pool. A full 3b sweep over all 18,714 objects with
papers is therefore a multi-day job; both scripts checkpoint and resume.
`FULLTEXT_REQUEST_BUDGET` defaults to the entire daily quota, so either lower it
to ~4,500 or run 3b and 3c on different days.

`cache/` (untracked) holds everything rebuildable: the ADS abstract cache, the
HF parquet downloads, NED resume checkpoints and run logs. Deleting it costs
quota and time, never results.

## Known gaps

- **`build_tier_report.py` is not reproducible from this repo.** It reads
  `tier_rows.json` and `imgs_png_b64.json` from `$TIER_SCRATCH_DIR`, and nothing
  here generates them — nor the `AnomalyMatch` flag its tiers depend on. The
  committed `score40_tiers.html` is the artefact; the recipe is missing.
- `plots_only_html.py`, `build_fig1.py` and `update_interesting_radius.py` do
  their work at module level rather than under `if __name__ == "__main__"`, so
  importing them runs them.
- `build_fig1.py` hardcodes `N_ANOM = 167` and writes outside this tree, into
  `../Workshop Paper/`.
