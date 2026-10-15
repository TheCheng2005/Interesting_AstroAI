# Paper animation

Two explainer videos for *Identifying Scientifically Interesting Galaxies with
Vision-Language Models*: what a VLM does with an image, the two models and
their prompts, the Likert and Tournament scoring formats, how much of the
known anomaly set they recover, Table 1, and the new candidates. The 2D and
3D films tell the same story with the same numbers.

**Watch:** [2D (108 s)](https://github.com/TheCheng2005/Interesting_AstroAI/releases/download/paper-animation-v1/explainer_2d.mp4)
· [3D (110 s)](https://github.com/TheCheng2005/Interesting_AstroAI/releases/download/paper-animation-v1/explainer_3d.mp4)
— both 1080p60, attached to the
[`paper-animation-v1` release](https://github.com/TheCheng2005/Interesting_AstroAI/releases/tag/paper-animation-v1)
rather than committed.

## Layout

```
paper_animation/
├── prepare/
│   ├── extract_cutouts.py   real cutouts and run statistics → assets/
│   └── fetch_logos.py       the two model logos → assets/logos/
├── film_2d/
│   └── scenes.py            the 2D film (manim), and the script both films share
├── film_3d/
│   ├── engine3d.py          a small offscreen OpenGL renderer
│   └── explainer3d.py       the 3D film, staged with it
├── assets/                  generated, gitignored
└── media/                   rendered videos, gitignored
```

Code only. The cutouts and scores are read from the paper's data and written
to `assets/`, which is never committed.

## Build

Run everything from this folder.

**1. Assets.** In the [`paper_code`](../paper_code) environment, with its data
variables set:

```bash
export HUBBLE_DATA_DIR=/path/to/hubble_data        # the cutout HDF5
export PAPER_RESULTS_DIR=/path/to/analysis          # holds results/…/*.csv
python prepare/extract_cutouts.py
python prepare/fetch_logos.py
```

`extract_cutouts.py` needs `gemini_likert_scores_sept10.csv` (the run behind
Table 1) and `gemini_tournament_1.csv`. It looks for them under
`results/` and then `results/subset_test/`, the same way
`paper_code/metrics/make_table1.py` does.

**2. Films.** In an environment built from `requirements.txt` (manim comes
from conda-forge):

```bash
manim -qh film_2d/scenes.py Explainer        # → media/videos/scenes/1080p60/
manim -ql film_2d/scenes.py Recall           # one beat, low quality, while iterating

python film_3d/explainer3d.py                # every beat, then media/3d/Explainer3D.mp4
python film_3d/explainer3d.py Recall         # one beat
python film_3d/explainer3d.py Recall --still 12   # a PNG of one frame
```

The 3D renderer runs on Mesa's software rasteriser (llvmpipe), so it needs no
GPU, only Linux with Mesa's EGL. It renders beats in parallel (`--jobs`,
`--threads` each); the whole film takes about 12 minutes on 24 cores. Text is
set in Lato and DejaVu Sans Mono (`fonts-lato`, `fonts-dejavu`).

## Sequence

| Beat | 2D scene | 3D | What it shows |
|---|---|---|---|
| The pile | `Opening` 6.8 s | 7.0 s | Millions of images — which deserve astronomers' attention? Leverage VLMs |
| VLM Mechanism | `VLM` 10.0 s | 10.4 s | Inputs → encode → align & fuse → understand → answer |
| VLM Choice | `Models` 8.3 s | 8.5 s | Gemini and Qwen: how each is run, and its list price per million tokens |
| VLM Prompts | `Prompts` 7.8 s | 8.0 s | Both prompts from Appendix A, key phrases highlighted in three steps |
| Scoring Formats | `Formats` 25.6 s | 25.7 s | Likert: a 4×4 grid scored over three rounds. Tournament: three rounds of 2×2 keep/reject |
| Anomaly Recovery | `Recall` 22.4 s | 22.5 s | All 167 reference anomalies; the cutoff score walks down from 50 to 33 |
| Results | `Results` 12.8 s | 12.9 s | Table 1, row by row, with best recall and lowest cost flagged |
| New Anomaly Candidates | `Funnel` 14.6 s | 15.0 s | 20,000 scored → 204 at 45+/50 → 102 candidates; Appendix B's 30 |

In 2D, each beat is its own manim Scene, so one can be re-rendered alone;
`Explainer` plays all eight. `FormatsLikert` and `FormatsTournament` render
the two halves of `Formats` separately.

### Beat notes

- **VLM Mechanism.** The prompt and the 1–5 scale on screen are the ones
  `paper_code/scoring/gemini_likert.py` sends. It shows one image for
  clarity; the real call carries a 4×4 grid, which Scoring Formats shows.
- **VLM Choice.** List prices are the figures `metrics/make_table1.py` prices
  the runs with ($0.25 / $1.50 for Gemini, $0.39 / $2.34 for Qwen). The logos
  come from Wikimedia Commons: the Gemini icon is public domain, the Qwen
  logo CC0.
- **VLM Prompts.** Both prompts are set word for word as Appendix A states
  them, rewrapped to 56 columns without splitting a highlighted phrase.
  Highlights land in three steps: the role both prompts give the model, the
  criterion both use, then what each format asks.
- **Scoring Formats.** Every bullet is the behaviour of the scoring scripts:
  Likert sends a 4×4 grid, scores each picked image 1–5 and unpicked ones 0,
  over ten rounds; Tournament sends a 2×2 grid and asks only keep or reject,
  and rejects leave the pool. Three rounds of each are drawn and the counter
  reads "of 10". The Likert grid is stocked like the real distribution, where
  8,098 of 20,000 images score zero.
- **Anomaly Recovery.** Tiles light in the order the method ranks them as the
  cutoff falls. The two counters are the real shortlist size and catch at
  every cutoff. It ends on cutoff 33: 2,095 shortlisted, 164 of 167 found
  (98.2%). The three it misses score 31, 30 and 23.
- **Results.** Likert thresholds are written out of 50; Tournament rows say
  Round 10. Recall is threshold recall, so 98.2% matches the Recall beat, not
  Expected Recall@2000. Precision is left out: it runs 7.4–8.1% across all
  rows and separates nothing.
- **New Anomaly Candidates.** 20,000 and 204 are recomputed from the run CSV;
  102 is the paper's. Between them the paper screens out 66 AnomalyMatch
  reference anomalies, 33 O'Ryan et al. (2023) interacting galaxies, one
  galaxy-mentions match and two from a literature review;
  `assets/candidates.json` keeps those counts. The wall is Appendix B, the 30
  highest-scoring of the 102, each checked against the run CSV.

### The 3D version

`explainer3d.py` imports its text, prompts, colours and numbers from
`film_2d/scenes.py` and the asset JSONs, so the two films cannot drift apart.
Only the staging changes:

- **Opening:** a field of thousands of cutouts to the horizon; the camera
  rises off it and a few striking ones lift out. About fourteen low-scoring
  cutouts sit on the edge of the HST footprint and carry a white wedge where
  there is no data. The decorative field leaves those out; the Formats grids
  keep them, as the 2D film does.
- **VLM Mechanism:** stage numbers flip in as coins; signals run along the
  arrows and through the network.
- **Scoring Formats:** picked Likert tiles lift toward the camera under their
  badges, and a reshuffle deals the grid through depth. Tournament rejects
  tip over and drop; refills fly in.
- **Anomaly Recovery:** a skyline, each anomaly as tall as its Likert score,
  with a glass cutoff plane descending from 50 to 33.
- **New Anomaly Candidates:** all 20,000 images as a carpet of chips; the 204
  rise, the 102 turn green and fly up, and 30 land as Appendix B's wall.

`engine3d.py` draws glass panels, image slabs, text, soft shadows from one key
light, a reflective floor, bloom and fog, and pipes frames to ffmpeg.

## Every number on screen is real

- **Cutouts:** the HST archive, via `prepare/extract_cutouts.py`. The hero
  object is `4001415925883`, the strong lens SDSS J1205+4910: 50/50 under
  Likert and 10/10 under Tournament.
- **Scores and recall:** the Sept-10 Gemini Likert run. At cutoff 33 the
  shortlist holds 164 of the 167 reference anomalies, 98.2%, Table 1's Gemini
  Likert row exactly.
- **Table 1 and costs:** as published.

**One reconstruction.** A Likert run logs only each image's final 0–50 total,
not the 1–5 it got in each round. The per-round badges show final ÷ 10, which
is the correct average by construction.

## Notes for editing the 2D film

Three manim traps this film hit, all commented in the code:

- Never animate a counter with `.animate.set_value()` when its digit count
  changes: manim burns the whole run time matching glyphs between the two
  states. Drive it from a `ValueTracker` instead (`counter_on`).
- Never rebuild a `Text` inside `always_redraw`; each rebuild calls pango.
- Never hand `FadeOut` or `.animate` a new `Group` of things already on screen
  unless you mean to restack them. manim adds a group it hasn't seen on top
  of everything first, which brings back objects already faded out and lifts
  things over what was covering them. Fade only what is on screen, piece by
  piece when stacking order matters.

A `DecimalNumber` also draws about 1.43× smaller than a `Text` at the same
`font_size` and regrows from its centre, hence `number()` and `pin_edge()`.
