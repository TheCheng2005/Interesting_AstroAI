# Paper code

The code behind *Identifying Scientifically Interesting Galaxies with
Vision-Language Models*.

**This package is code only — no data.** Every number in Table 1, Table 2 and
the candidates section is produced by a script here, but the image catalogue
and the result CSVs live outside it.

## Models

| Model | Access | Used for |
|---|---|---|
| **Gemini 3.1 Flash-Lite** (`gemini-3.1-flash-lite`) | Google Gen AI SDK, `GEMINI_API_KEY` | `scoring/gemini_likert.py`, `scoring/gemini_tournament.py` |
| **Qwen3.5-397B-A17B** (`qwen/qwen3.5-397b-a17b`) | OpenRouter, `OPENROUTER_API_KEY` | `scoring/qwen_likert.py`, `scoring/qwen_tournament.py` |

Both are prompted zero-shot with the same text (`scoring/GEMINI.md`,
reproduced in Appendix A). Gemini runs at thinking level `low`; Qwen has
reasoning disabled. Two methods × two models is the four runs of Table 1.

## Pipeline

```
  ┌────────────────────┐
  │ 0  dataset/        │  pipeline.py
  └─────────┬──────────┘  → 223,195 cutouts as .hdf5 + .parquet (~3 GB)
            │                MAST + HSC v3, CasJobs credentials, many hours
  ┌─────────▼──────────┐
  │ 1  scoring/        │  gemini_likert.py, gemini_tournament.py, …
  └─────────┬──────────┘  → results/subset_test/*.csv
            │                needs stage 0 + an API key; bills real tokens
  ┌─────────▼──────────┐
  │ 2  unidentified_   │  build_undiscussed_catalog.py  <- the only command
  │    objects/        │
  │   ├─ 2A ───────────┤  FIND THE PAPERS (find_unidentified_objects.py,
  │   │                │  imported as a library): 3" match against HF
  │   │                │  galaxy-mentions, SIMBAD and NED, and pull the
  │   │                │  bibliography each attributes to the object
  │   ├─ 2B ───────────┤  JUDGE THE PAPERS: do any actually discuss it?
  │                    │  abstracts + in-body snippets -> Gemini verdict,
  │                    │  then the O'Ryan et al. (2023) screening flags
  └─────────┬──────────┘  → undiscussed_catalog.csv, candidate_counts.json
            │                needs an ADS token + GEMINI_API_KEY
  ┌─────────▼──────────┐
  │ 3  metrics/        │  make_table1.py
  └────────────────────┘  → the published numbers, stdlib only, seconds
```

2A finds the papers, 2B reads them — but **2A is not a command**.
`build_undiscussed_catalog.py` imports 2A's cross-match code and calls it, so
one run does both. Stage 3 is independent of either.

Stage 2 and stage 3 never touch the image catalogue. **If you only want to check
the published numbers, skip to step 3 below** — it takes seconds and needs no
key, no download and no install.

## Configuration

Two environment variables tell the code where data lives:

| Variable | Points at | Needed by |
|---|---|---|
| `HUBBLE_DATA_DIR` | the cutout catalogue (`*.hdf5`, `*.parquet`) and `Interesting.csv` | stage 1 |
| `PAPER_RESULTS_DIR` | the tree holding `results/`, `unidentified_objects/`, `literature_crossmatch/` | stages 2, 3 |

Leave `PAPER_RESULTS_DIR` unset and a stage writes a fresh set of results into
this folder instead; those paths are gitignored so the package stays code only.

---

# Steps

## Step 0 — build the dataset (hours, ~3 GB, optional)

Only needed if you are re-running the scoring. Requires a MAST CasJobs
account (free, register at <https://mastweb.stsci.edu/ps1casjobs/>).

```bash
pip install -r requirements.txt

cat > .env <<'EOF'
CASJOBS_WSID=<your numeric CasJobs WebServices ID>
CASJOBS_PW=<your CasJobs password>
EOF

python dataset/pipeline.py --work-dir work --output-dir ~/hubble_data
```

What it does, in order: inventories every HAP-SVM ACS/WFC F814W `drc.fits`
product on MAST; uploads that image list to CasJobs; selects the first 10
million extended-source detections (`Det='Y'`, odd `Flags`) from HSC v3
`DetailedCatalog`; greedily suppresses any detection within 10″ of a kept one,
preferring unsaturated sources with low flags; then downloads each FITS image,
cuts a 150×150 px stamp at every surviving position, stretches it with
`ZScaleInterval(contrast=0.05)`, and stores it JPEG-encoded in the HDF5. The
survivors are the **223,195** cutouts of §3.

Outputs into `--output-dir`:

```
10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5      images + filenames
10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.parquet   positions + metadata
```

Set `--output-dir` to whatever you will use as `HUBBLE_DATA_DIR`. The run
resumes: the MAST inventory, HSC counts, CasJobs chunks and the deduplicated
Parquet are each reused if already present, and `--stop-after catalog|parquet`
stops early.

HSC v3 is a frozen catalogue release, so the same query returns the same 10
million detections and the same 223,195 cutouts on every run.

One thing the pipeline does **not** produce: `Interesting.csv`, the
AnomalyMatch anomaly positions. It comes from Gomez et al. (2025) and must be
placed in `HUBBLE_DATA_DIR` alongside the two files above, or every
ground-truth label is missing.

## Step 1 — score the images (optional, costs money)

```bash
export HUBBLE_DATA_DIR=~/hubble_data
export GEMINI_API_KEY=<key>          # for scoring/gemini_*.py
export OPENROUTER_API_KEY=<key>      # for scoring/qwen_*.py

python scoring/gemini_likert.py       # Likert Select, 4x4 grid, 10 rounds
python scoring/gemini_tournament.py   # Tournament, 2x2, iterative retention
python scoring/qwen_likert.py
python scoring/qwen_tournament.py
```

Each script reads `TEST_MODE = True` and scores a 20,000-image working set —
all 167 anomalies plus a random fill — once per seed in `RANDOM_SEEDS`, writing
`results/subset_test/{provider}_{protocol}_{run}.csv`. The seeds match the
paper: `[44, 45, 46]` in the Gemini scripts, `[44]` in the Qwen scripts, which
is why Table 1 reports Gemini as a mean over three seeds and Qwen over one.
Set `TEST_MODE = False` to score the whole catalogue.

The `interesting` column is the ground truth: 1 when an `Interesting.csv`
position falls within `MATCH_RADIUS_ARCSEC` of the cutout centre. All four
scoring scripts use 3″, which is the paper's "cutouts receive an anomaly label
when an AnomalyMatch detection lies within 3″", so the labels are final as
written — nothing re-labels them afterwards.

Re-running bills real tokens and, because the models are not deterministic,
will not reproduce the published CSVs image-for-image. The aggregate metrics
are stable: the Discussion quotes recall 0.980 ± 0.015 at score ≥33 across
three independently seeded Gemini *Likert* runs.

**Log the cost record.** The cost column of Table 2 is read back out of each
CSV's `# TOKEN USAGE SUMMARY` footer, so a run has to write it:
`# TotalInputTokens`, `# TotalOutputTokens`, and — for Gemini, which bills
thinking at the output rate — `# TotalThinkingTokens`. A Qwen run should
record `# TotalCostUSD`, the charge OpenRouter actually applied, which is
what the paper quotes rather than a token-price estimate.

## Step 2 — from scores to a screened catalogue

This stage answers two different questions, and it helps to keep them apart:

- **2A — which papers even mention this position?** Cross-match the image
  against sky catalogues, and collect the bibliography each catalogue
  attributes to whatever object it finds there.
- **2B — do any of those papers actually *say something* about the object?**
  Read them and decide, because a catalogue entry is not a discussion.

**One command runs both.** `build_undiscussed_catalog.py` calls 2A's
cross-match functions itself; you do not run 2A first. It is split out below
only because the two halves answer different questions, and because 2A is
also runnable on its own for a different purpose.

### Step 2A — find the papers (runs inside 2B)

There is no 2A command. `unidentified_objects/find_unidentified_objects.py`
is the cross-match **library** that 2B imports and calls; the file is
required — delete it and 2B dies on `ModuleNotFoundError` — but you never
invoke it yourself. This section describes what happens inside the run.

For each candidate image, 2A searches a 3″ circle around the position in
three catalogues:

| Catalogue | What a hit means | What it yields |
|---|---|---|
| HF `astronolan/galaxy-mentions` | a paper already resolved a name to this position | the mention itself |
| SIMBAD (bulk TAP upload) | SIMBAD records an object here | `main_id`, object type, and **every paper SIMBAD links to it** |
| NED (one object at a time) | NED records an object here | name, type, redshift, and **its reference list** |

The SIMBAD and NED bibliographies are the point: they are how an image gets
from "there is an object at these coordinates" to "here are the 40 papers
that cite it". That list is the evidence 2B then judges. This is the paper's
"we search for positional counterparts within 3″ in SIMBAD, NED, a dataset of
galaxy mentions in astronomical papers, and the interacting-galaxy catalogue
of O'Ryan et al. (2023)" — applied to the 138 candidates, not to the whole
corpus.

Two choices worth knowing. Both SIMBAD and NED are queried for **every**
image rather than stopping at the first catalogue that answers — stopping
early would leave SIMBAD-matched images with no NED bibliography, understating
the literature on exactly the objects most likely to have some. And one image
can resolve to several catalogue objects within 3″ (an optical and a radio
designation for one source), so all of their papers are pooled.

The file also carries a `__main__` block that sweeps the **whole** scored
corpus and writes `unidentified_objects.csv`, `matched_objects.csv` and the
two bibliographies. No number in the paper comes from it. It is left in place
because it is the same code path 2B calls, but it is not a step here.

**SIMBAD and NED are live**, so a later run sees positions that have since
been catalogued. Quote the query date alongside any count taken from them.

### Step 2B — judge whether those papers discuss the object

2A hands over a pile of papers per object. The question here is whether any of
them says something about *that* object, or whether it only appears as row 400
of a survey table. Being catalogued is fine; being discussed is not. This is
the step you actually run:

```bash
export PAPER_RESULTS_DIR=/path/to/analysis
export HUBBLE_DATA_DIR=~/hubble_data                       # parquet coordinates
export ORYAN_CATALOGUE_DIR=~/hubble_data/zenodo_7684876/catalogues
export GEMINI_API_KEY=<key>
echo "<your ADS token>" > ~/.ads_api_key

python unidentified_objects/build_undiscussed_catalog.py [scores.csv] [out.csv]
```

Four conditions and a screening pass, at the released cut of 45
(`CATALOG_MIN_SCORE`):

| # | Step | Survivors |
|---|---|---|
| — | scored images | 20,000 |
| 1 | `imagescore >= 45` | 204 |
| 2 | not a reference anomaly (`interesting == 0`) | **138** |
| 3 | no HF galaxy-mentions counterpart within 3″ | 136 |
| 4 | no matched object is *genuinely discussed* | **131** |
| 5 | O'Ryan screening flags — recorded, never excluded | 131 |

Condition 4 is the point, and it is deliberately weaker than "uncatalogued".
Unlike `find_unidentified_objects.py`, SIMBAD and NED are queried here **only
to collect papers**, never to disqualify an image — being catalogued is fine,
being discussed is not. Six objects across five images are judged
individually discussed; the highest-scoring is the strong lens
`SDSS J1205+4910` at 50/50.

Step 5 matches every image at 3″ against the interacting-galaxy catalogues of
O'Ryan et al. (2023) (Zenodo 7684876 — a directory of CSVs with `SourceID`,
`RA`, `Dec` columns) and **flags rather than excludes**. The released list is
a screening result, not a claim of novelty: prior recognition of interacting
morphology is context a reader needs, so it lands in
`oryan_interacting_match`, `oryan_any_catalogue_match`, `oryan_catalogues`,
`oryan_source_ids` and `nearest_oryan_arcsec`. Without
`ORYAN_CATALOGUE_DIR` the run still completes, with those columns empty and a
message saying so.

Outputs:

- `undiscussed_catalog.csv` — the released catalogue, one row per candidate
- `all_nonreference_above_threshold.csv` — every image that entered the
  discussion screen, including the ones it removed, with
  `passed_discussion_screen`
- `candidate_counts.json` — the counts quoted in the appendix

Verified against the published counts: **138** non-reference images at ≥45,
**131** released, of which **89** have retrieved papers assessed by the
classifier and **42** have none — all four exact.

Three details decide the verdict's quality:

- **Papers come from the catalogues, not from ADS free-text.** The full-text
  query is restricted by `fq=bibcode:(…)` to the bibcodes SIMBAD and NED
  already attribute to the object. Searching the designation across all of ADS
  instead pulls in papers from other fields that happen to reuse the string —
  `full:"ASV 25"` matches microbiology papers about amplicon sequence variants.
- **Aliases are resolved positionally, not by name.** Every designation SIMBAD
  records inside the 3″ circle is searched, because an object discussed under
  a name that was never searched looks undiscussed and wrongly stays in the
  catalogue. The lens above carries one identifier under its own entry but ten
  at its position.
- **Evidence is abstracts plus verbatim in-body snippets**, which is what
  separates "listed in Table 3" from "we model its tidal tail in Section 4".
  Each object is judged `CLASSIFY_VOTES = 3` times at temperature 0 and the
  majority wins; the split-vote count is printed at the end.

Everything checkpoints under `cache/undiscussed_catalog/`, and the ADS
full-text checkpoint is shared with the rest of the stage, so an interrupted
run re-spends no quota (ADS allows 5,000 requests/day). The cross-match
checkpoint records the score cut it was built at and is ignored if you change
`CATALOG_MIN_SCORE`, so a re-run at a new threshold cannot silently reuse the
old selection.

## Step 3 — recompute the published numbers (seconds, no install)

```bash
export PAPER_RESULTS_DIR=/path/to/analysis
python metrics/make_table1.py            # Tables 1 and 2
```

Standard library only. It prints the recomputed values beside the published
ones, flags any disagreement, says where each cost figure came from, and
names any run it could not find rather than quietly omitting the row.

---

## What produces what

| Paper element | Script |
|---|---|
| §3 dataset construction — the 223,195 cutouts | `dataset/pipeline.py` |
| §3 *Likert Select* scores | `scoring/{gemini,qwen}_likert.py` |
| §3 *Tournament* scores | `scoring/{gemini,qwen}_tournament.py` |
| §3 the `interesting` ground-truth column (3″ to AnomalyMatch) | the scoring scripts themselves, `MATCH_RADIUS_ARCSEC = 3.0` |
| Table 1 — N selected, recall, precision, Expected Recall@2000, $/10k | `metrics/make_table1.py` |
| Table 2 — tokens/image and $/10,000 images | `metrics/make_table1.py` |
| Discussion — seed-to-seed recall spread | `metrics/make_table1.py` |
| 2A — finding the papers a catalogue attributes to each position | `unidentified_objects/find_unidentified_objects.py` (library, not a command) |
| 2B — the released catalogue, 138 → 131, with O'Ryan flags | `unidentified_objects/build_undiscussed_catalog.py` |
| 2B — the ADS retrieval and the discussion classifier it calls | `literature_crossmatch/{classify_genuine_discussion,fulltext_search_classification,deep_dive_summaries}.py` |
| Appendix A — the exact prompt | `scoring/GEMINI.md` |

`common/paths.py` holds every file location, anchored to itself rather than to
the working directory, so any script runs from anywhere.

## CSV schemas

The protocols emit two header spellings for the same information:

| Protocols | Header |
|---|---|
| Likert Select, Hybrid | `index,filename,imagescore,interesting,classification,SourceRA,SourceDec` |
| Tournament, Single-elim | `Filename,ImageScore,interesting,classification,SourceRA,SourceDec` |

Every CSV also ends with a `# TOKEN USAGE SUMMARY` footer holding the run's
input/output token totals, which is what Table 2's cost is computed from.

`common/scores_csv.py` is the single reader for both spellings, used by every
script that consumes a scoring CSV, so either one works anywhere with no
adjustment. It raises on a header it does not recognise rather than returning
an empty result, and its writer round-trips the original spelling and the
footer intact.

## Not included

The 223,195-cutout HDF5 (~3 GB), the result CSVs, and `Interesting.csv` — the
AnomalyMatch anomaly positions from Gomez et al. (2025), on which every
ground-truth label depends.
