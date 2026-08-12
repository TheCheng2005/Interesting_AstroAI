# Hubble / analysis

Pipeline, in order: **score** the full image catalog with an AI judge →
take the highest-scoring candidates and **cross-match** them against known
literature (NED/SIMBAD/ADS) → whatever has no match is a genuine
**unidentified** candidate worth a closer look → **reports**/**poster**
turn all of the above into human-readable output.

Scripts reference the source data with relative paths like
`../hubble_data/...` (one level above this folder). Since scripts now live
one level deeper under `scoring/`, those paths need updating (e.g. to
`../../hubble_data/...`) before re-running.

## `scoring/`

The AI-judging protocol scripts and the shared prompt they all use.

- [`GEMINI.md`](scoring/GEMINI.md) — the classification prompt/taxonomy
  given to the model: flag rare/anomalous/high-energy phenomena
  (lenses, mergers, jellyfish galaxies, AGN, ...) and aggressively reject
  imaging artifacts (cosmic rays, diffraction spikes, satellite trails).
- `gemini_*.py` / `qwen_*.py` — one protocol per file, run against the same
  underlying HDF5/parquet image catalog:
  - `*_tournament.py` — pairwise elimination tournament.
  - `*_single_elim.py` — single-elimination variant.
  - `*_likert.py` — repeated 4×4 grid selection with a 1–5 interestingness
    score, summed across appearances.
  - `*_hybrid.py` — one cheap tournament-filter round to eliminate most
    boring images, then Likert-scores only the survivors. This is the
    protocol used for the full-catalog production runs in
    `results/full_catalog/` (see below) — it exists specifically because
    scoring every image with the full Likert protocol was too expensive
    at ~10M images.
- `update_interesting_radius.py` — utility to re-label an existing results
  CSV's `interesting`/`classification` columns with a tighter match radius
  against `Interesting.csv`, without re-running the scoring pipeline.

Ground truth for "interesting" is nearest-image match (within a configurable
match radius, in arcsec) to entries in an `Interesting.csv` list of known
objects of interest.

## `results/`

Output of the scoring scripts — columns are
`index/filename, imagescore, interesting, classification, SourceRA, SourceDec`.

- `full_catalog/` — the full-scale production results:
  - `10M_10arc_dedup_centered.csv` / `10M_10arc_dedup_uncentered.csv` —
    Gemini hybrid-protocol scores over the ~10M-image, 10″-dedup catalog
    (centered vs. uncentered cutout crop).
  - `qwen_10M_10arc_dedup_centered.csv` / `qwen_10M_10arc_dedup_uncentered.csv`
    — same, scored with Qwen instead of Gemini.
  - `1M_1arc_dedup.csv` — Gemini run on a smaller 1M-image, 1″-dedup catalog.
- `subset_test/` — small pilot runs (one CSV per model × protocol) used to
  sanity-check each scoring script before committing to a full-catalog run.

## `literature_crossmatch/`

Checks whether the AI-flagged candidates are already known objects in the
literature, using NED, SIMBAD, and ADS full-text search.

- `ned_bibliography.csv` / `simbad_bibliography.csv` — NED/SIMBAD lookups
  for candidate coordinates.
- `matched_objects.csv` — candidates matched to a named object
  (`matched_source`, `object_name`, `object_type`, redshift, arXiv ref).
- `classify_genuine_discussion.py` / `discussion_classification.csv` — use
  an LLM to classify whether NED/SIMBAD bibliography entries genuinely
  discuss the object (vs. incidental mentions).
- `fulltext_search_classification.py` / `fulltext_hits.csv` /
  `fulltext_search_checked.txt` — ADS full-text search for candidates with
  no direct catalog match.
- `reclassify_with_fulltext.py` — folds full-text search results back into
  the classification.

## `unidentified_objects/`

Candidates that survived scoring and literature cross-matching with **no**
match found — i.e. not currently a known/discussed object.

- `find_unidentified_objects.py` / `unidentified_objects.csv` — the
  filtered candidate list (`avg_score`, coordinates, whether NED was
  checked).
- `deep_dive_summaries.py` / `deep_dive_summaries.csv` /
  `deep_dive_summaries.json` — LLM-generated written summaries for each
  unidentified candidate.
- `generate_unidentified_html_report.py` / `unidentified_objects_report.html`
  — rendered browsable report of the candidates.

## `reports/`

General result visualization, separate from the unidentified-objects report
above.

- `generate_html_report.py` / `hsc_report_mixed.html` — full scoring-run
  report (score distributions, example images).
- `plots_only_html.py` / `plots_only.html` — plots-only version of the same.

## `poster/`

Figures generated for the project poster.

- `poster_method_diagram.py` (+ `poster_method_diagram_flowchart.png`,
  `poster_method_diagram_icons.png`) — pipeline diagram.
- `poster_recall_plot.py` (+ `poster_recall_vs_cost.png`) — recall vs. API
  cost tradeoff across protocols.
- `poster_score_distribution.py` (+
  `poster_score_distribution_gemini_tournament.png`) — score distribution
  plot.

## `sample_data/`

- `catalog_sample.csv` — first 20 rows of the source HDF5/parquet catalog
  (columns: match/source IDs, coordinates, instrument/filter, image path).
- `sample_image_*.jpg` — 10 example 150×150 JPEG thumbnails decoded directly
  from the source HDF5 image store.
