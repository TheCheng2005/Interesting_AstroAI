# Weird and Wonderful

Tests whether AI vision-language models (Gemini, Qwen) can reproduce the
judgments of Galaxy Zoo volunteers on the *Weird and Wonderful* project —
identifying unusual or scientifically interesting galaxy images out of a
large citizen-science subject set.

- [`data/`](data/) — subject metadata, volunteer tags, comments, and
  consensus aggregation from the Zooniverse project. Ships as small samples
  only (see below); full dataset described in Mantha et al. (2024), see the
  top-level [README](../README.md) for the citation.
- [`analysis/`](analysis/) — AI-judging scripts and their result CSVs.

## Methods (`analysis/`)

Each script runs one AI-judging protocol against the W&W subject images and
writes a result CSV:

- `grid_score_*.py` — n×n grid selection with a 1–5 interestingness score.
  Variants differ in grid size, prompt (`_zoo_prompt` frames the model as a
  Galaxy Zoo volunteer rather than an expert astronomer), and whether
  few-shot artifact examples are included.
- `single_elimination.py` — pairwise elimination tournament.
- `tournament_filtering.py` — post-hoc filtering/aggregation of tournament
  results.
- `download_png.py` / `generate_html.py` — subject image download and
  HTML report generation for eyeballing results (`report.html`).

Result CSVs are timestamped (e.g. `grid_results_score_3.1_low_05-20_15.csv`)
— filename encodes the scoring version and run date/hour.

## Data (`data/`)

| File | Contents |
|---|---|
| `all_ww_data.csv` | Per-subject anomaly/image/feature scores and coordinates |
| `galaxy-zoo-weird-and-wonderful-subjects.csv` | Zooniverse subject metadata |
| `latest_consensus_aggregation_table.dat` | Volunteer consensus classification counts |
| `project-14993-comments_2023-08-02.json` | Volunteer discussion comments |
| `project-14993-tags_2023-08-02.json` / `project-5733-tags_2023-11-28.json` | Volunteer-applied tags |

Each file here is truncated to its first ~20 rows/records as a schema
example. For the full dataset, see Mantha et al. (2024) (citation in the
top-level [README](../README.md)).

**Note:** some analysis scripts reference these files by their original
relative paths (`ww_data/...`) or hardcoded absolute paths (e.g. `IMAGE_DIR`
in `grid_score_zoo_prompt.py`) — update those paths before re-running against
the full dataset.
