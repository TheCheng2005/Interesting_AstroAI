# Interesting AstroAI

Can AI vision-language models (Gemini, Qwen) stand in for human volunteers and
domain experts when the task is spotting scientifically "interesting" —
rare, anomalous, high-value — objects in large astronomical image sets?

This repository holds the code and results for two parallel studies that ask
that question of different datasets, plus a shared validation study
comparing the AI judges against real experts and citizen-science volunteers.

## Contents

| Folder | Question | Data source |
|---|---|---|
| [`weird_and_wonderful/`](weird_and_wonderful/) | Do AI judges agree with Galaxy Zoo volunteers on which subjects are "weird and wonderful"? | Galaxy Zoo: Weird and Wonderful citizen-science project |
| [`hubble/`](hubble/) | Can AI judges triage millions of Hubble cutouts down to a tractable set of scientifically interesting candidates? | Hubble Legacy Archive cutouts |
| [`expert_validation/`](expert_validation/) | How well do AI judgments agree with trained astronomers, and can that agreement be modeled? | Expert ratings collected on both datasets above |

Each folder has its own README with method details and how to reproduce the
headline results.

## Method family

Across both datasets, the same family of AI-judging protocols is compared:

- **Grid selection** — the model is shown an n×n grid of images and picks
  any subset it finds interesting (optionally with a 1–5 Likert score).
- **Tournament / single-elimination** — images are shown pairwise and the
  model picks the more interesting one; winners advance.
- **Few-shot vs. no-shot** — with or without example artifact images in the
  prompt, to see how much that steers the model away from false positives
  (cosmic rays, diffraction spikes, satellite trails, etc.).

Both Gemini and Qwen are evaluated under each protocol.
<img width="2200" height="2400" alt="poster_method_diagram_icons" src="https://github.com/user-attachments/assets/d73399eb-8573-42e6-9725-851d4302952f" />

## Data

Both datasets are large (the Hubble cutout catalog alone is millions of
images / multiple GB) and are not redistributed in full here. Each data
folder ships a small sample (first ~20 rows/records, or a handful of example
images) so the schema is inspectable without cloning gigabytes. For the full
datasets, see the source papers:

- Mantha, K. B., Roberts, H., Fortson, L., Lintott, C., Dickinson, H., Keel,
  W., Sankar, R., Krawczyk, C., Simmons, B., Walmsley, M., Garland, I.,
  Makechemu, J. S., Trouille, L., & Johnson, C. (2024). Through the citizen
  scientists' eyes: Insights into using citizen science with machine
  learning for effective identification of unknown-unknowns in big data.
  *Citizen Science: Theory and Practice*, 9(1), Article 40, 1–15.
  https://doi.org/10.5334/cstp.740
- O'Ryan, D., & Gómez, P. (2025). Identifying astrophysical anomalies in
  99.6 million cutouts from the Hubble Legacy Archive using AnomalyMatch.
  *Astronomy & Astrophysics*, 704, A227.
  https://doi.org/10.1051/0004-6361/202555512

## Setup

```
pip install -r requirements.txt
```

Scripts that call Gemini expect a `JB_API_KEY` environment variable; scripts
that call Qwen via OpenRouter expect `OPENROUTER_API_KEY`.

## License

MIT — see [LICENSE](LICENSE).
