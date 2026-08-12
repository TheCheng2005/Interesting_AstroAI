# Hubble

Tests whether AI vision-language models (Gemini, Qwen) can triage a large
catalog of Hubble Legacy Archive cutouts down to a tractable set of
scientifically interesting candidates — gravitational lenses, mergers,
jellyfish galaxies, AGN, and other rare morphologies — while rejecting
imaging artifacts (cosmic rays, diffraction spikes, satellite trails).

- [`analysis/`](analysis/) — AI-judging scripts, bibliography/NED
  cross-matching, and result reports.

Full raw dataset (the source HDF5 image cutout catalog, multiple GB) is not
included here; see O'Ryan & Gómez (2025) in the top-level
[README](../README.md) for the source. A small sample of catalog rows and
example thumbnail images is included at
[`analysis/sample_data/`](analysis/sample_data/).

## Methods (`analysis/`)

The classification prompt/taxonomy used across all AI-judging scripts is in
[`GEMINI.md`](analysis/GEMINI.md) — models are instructed to flag rare/
anomalous/high-energy phenomena and aggressively reject imaging artifacts.

- `gemini_*` / `qwen_*` scripts — one AI-judging protocol per file:
  - `*_tournament.py` — pairwise elimination tournament.
  - `*_single_elim.py` — single-elimination variant.
  - `*_likert.py` — 4×4 grid selection with a 1–5 interestingness score.
  - `*_hybrid.py` — combined protocol.
- `classify_genuine_discussion.py` / `fulltext_search_classification.py` /
  `reclassify_with_fulltext.py` — cross-check candidate objects against
  NED/ADS literature to see if they're already known/discussed.
- `find_unidentified_objects.py` / `deep_dive_summaries.py` /
  `generate_unidentified_html_report.py` — flag and summarize candidates
  with no literature match.
- `update_interesting_radius.py` — nearest-match labeling against
  `Interesting.csv` within a configurable match radius (used by the
  AI-judging scripts to derive ground-truth labels).
- `poster_*.py` — figures for the project poster.

Ground truth for "interesting" is nearest-image match (within a fixed
match radius, in arcsec) to entries in an `Interesting.csv` list of known
objects of interest.

## Sample data (`analysis/sample_data/`)

- `catalog_sample.csv` — first 20 rows of the source HDF5/parquet catalog
  (columns: match/source IDs, coordinates, instrument/filter, image path).
- `sample_image_*.jpg` — 10 example 150×150 JPEG thumbnails decoded directly
  from the source HDF5 image store.
