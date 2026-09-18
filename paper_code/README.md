# Paper code

Code for *Identifying Scientifically Interesting Galaxies with Vision-Language
Models*.

**Code only — no data.** Paths to the data come from environment variables, so
nothing is baked in.

## Models

Both are prompted zero-shot with the same text (`scoring/GEMINI.md`, Appendix A).

| Model | Access | Scripts |
|---|---|---|
| Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) | Google Gen AI, `GEMINI_API_KEY` | `scoring/gemini_*.py` |
| Qwen3.5-397B-A17B (`qwen/qwen3.5-397b-a17b`) | OpenRouter, `OPENROUTER_API_KEY` | `scoring/qwen_*.py` |

Gemini runs at thinking level `low`; Qwen has reasoning disabled.

## The four stages

```
0  dataset/pipeline.py                 HSC v3 + MAST  →  223,195 cutouts (~3 GB)
1  scoring/{gemini,qwen}_{likert,tournament}.py   →  a score per image
2  unidentified_objects/build_undiscussed_catalog.py  →  the released catalogue
3  metrics/make_table1.py              →  Tables 1 and 2
```

Stages 0 and 1 are the expensive ones. Everything downstream needs only their
end product, the scoring CSVs. **To check the published numbers, run stage 3
alone** — seconds, no key, no install.

## Configuration

| Variable or file | Points at | Needed by |
|---|---|---|
| `PAPER_RESULTS_DIR` | the tree holding `results/` and `unidentified_objects/` | 2, 3 |
| `HUBBLE_DATA_DIR` | the cutout catalogue and `Interesting.csv` | 1, and 2 (the `.parquet` only) |
| `ORYAN_CATALOGUE_DIR` | the O'Ryan et al. (2023) catalogue CSVs | 2 |
| `GEMINI_API_KEY` | Google Gen AI token | 1, 2 |
| `OPENROUTER_API_KEY` | OpenRouter token | 1 (Qwen) |
| `~/.ads_api_key` | a file holding the ADS token | 2 |
| `CASJOBS_WSID`, `CASJOBS_PW` | MAST CasJobs credentials, in a `.env` file | 0 |

`Interesting.csv` is needed **only by stage 1**, where it creates the
`interesting` column. After that the label travels inside the scoring CSVs, so
stages 2 and 3 run without it.

Leave `PAPER_RESULTS_DIR` unset and a stage writes results into this folder;
those paths are gitignored.

---

## Stage 0 — build the dataset

Needs a free MAST CasJobs account. Many hours, ~3 GB.

```bash
pip install -r requirements.txt
printf 'CASJOBS_WSID=%s\nCASJOBS_PW=%s\n' <id> <password> > .env
python dataset/pipeline.py --output-dir ~/hubble_data
```

Samples 10 million extended-source detections from Hubble Source Catalog v3,
keeps those more than 10″ from any other, and cuts a 150×150 px `ZScaleInterval`
stamp at each — the 223,195 cutouts of §3. HSC v3 is frozen, so the same query
returns the same set every run. It resumes from whatever it has already done.

`Interesting.csv` is **not** produced here. It holds the AnomalyMatch anomaly
positions from Gomez et al. (2025) and must be placed in `HUBBLE_DATA_DIR`.

## Stage 1 — score the images

Costs real money.

```bash
export HUBBLE_DATA_DIR=~/hubble_data GEMINI_API_KEY=<key> OPENROUTER_API_KEY=<key>
python scoring/gemini_likert.py       # Likert: 4×4 grid, 10 rounds, score 0–50
python scoring/gemini_tournament.py   # Tournament: 2×2, survive 0–10 rounds
python scoring/qwen_likert.py
python scoring/qwen_tournament.py
```

Each scores a 20,000-image set — all 167 anomalies plus a random fill — once per
seed in `RANDOM_SEEDS`. All four Table 1 rows come from the common seed-44 set,
so they are directly comparable; the Gemini scripts also run seeds 45 and 46,
which is what the Discussion's seed spread is measured over.

An image is labelled `interesting` when an `Interesting.csv` position lies
within 3″ of its centre.

The models are not deterministic, so a re-run will not match the published CSVs
image for image. The aggregate metrics are stable.

**Log the cost record.** Table 2 is read back out of each CSV's
`# TOKEN USAGE SUMMARY` footer, so a run must write it: `# TotalInputTokens`,
`# TotalOutputTokens`, and for Gemini `# TotalThinkingTokens` (billed at the
output rate). A Qwen run should record `# TotalCostUSD`, the charge OpenRouter
actually applied, which is what the paper quotes.

## Stage 2 — build the released catalogue

```bash
export PAPER_RESULTS_DIR=/path/to/analysis HUBBLE_DATA_DIR=~/hubble_data
export ORYAN_CATALOGUE_DIR=~/hubble_data/zenodo_7684876/catalogues
export GEMINI_API_KEY=<key>
echo "<ADS token>" > ~/.ads_api_key

python unidentified_objects/build_undiscussed_catalog.py [scores.csv] [out.csv]
```

Four conditions, at the released cut of 45 (`CATALOG_MIN_SCORE`):

| # | Condition |
|---|---|
| 1 | `imagescore >= 45` |
| 2 | not a reference anomaly |
| 3 | no counterpart within 3″ in **galaxy-mentions** or the **O'Ryan** catalogues |
| 4 | no matched SIMBAD or NED object is **genuinely discussed** |

Condition 3 is applied before condition 4 on purpose: an image dropped there
costs no ADS quota and no LLM call.

Condition 4 is the interesting one. SIMBAD and NED are queried **only to collect
papers**, never to disqualify an image: an object can sit in a catalogue as row
400 of a survey table with nobody ever having said a word about it. Being
catalogued is fine; being discussed is not. The verdict comes from Gemini
reading each paper's abstract plus the verbatim in-body snippets ADS returns
around every mention of the object — which is what separates "listed in Table 3"
from "we model its tidal tail in Section 4".

Writes three files: the catalogue, every image that entered condition 3 with a
`dropped_by` column saying what removed it, and `candidate_counts.json`.

SIMBAD and NED are live, so a later run sees positions that have been
catalogued since — quote the query date with any count. Everything checkpoints,
so an interrupted run re-spends no ADS quota (the limit is 5,000 requests/day).

## Stage 3 — recompute the tables

```bash
export PAPER_RESULTS_DIR=/path/to/analysis
python metrics/make_table1.py
```

Standard library only. Prints the recomputed values beside the published ones,
flags disagreements, says where each cost figure came from, and names any run it
could not find.

---

## What produces each paper number

| Paper | Script |
|---|---|
| §3 the 223,195 cutouts | `dataset/pipeline.py` |
| §3 *Likert* and *Tournament* scores | `scoring/{gemini,qwen}_{likert,tournament}.py` |
| Tables 1 and 2, and the Discussion's seed spread | `metrics/make_table1.py` |
| the released candidate catalogue | `unidentified_objects/build_undiscussed_catalog.py` |
| ↳ ADS retrieval and the discussion classifier it calls | `literature_crossmatch/*.py` |
| Appendix A, the exact prompt | `scoring/GEMINI.md` |

`common/paths.py` holds every file location. `common/scores_csv.py` reads the
scoring CSVs, which come in two header spellings — *Likert* has a leading
`index` column, *Tournament* does not — plus the token footer.

## Not included

Bring these from their own sources: the cutout `.hdf5`/`.parquet` (stage 0
rebuilds them), `Interesting.csv` (Gomez et al. 2025), the O'Ryan et al. (2023)
catalogues (Zenodo 7684876), and the scoring CSVs if you are not re-running
stage 1.
