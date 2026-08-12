# Expert Validation

Checks how well the AI-judging protocols (see [`weird_and_wonderful/`](../weird_and_wonderful/)
and [`hubble/`](../hubble/)) agree with trained astronomers, and whether
that agreement can be modeled to predict expert opinion from AI + volunteer
signals alone.

- [`expert_comparison/`](expert_comparison/) — raw per-expert ratings
  against AI (Gemini/Qwen) ratings, and the agreement report.
- [`regression/`](regression/) — a regression model predicting expert
  agreement from AI/volunteer features, and its evaluation.

## `expert_comparison/`

- `expert_csvs/expert_scores_<Name>.csv` — one CSV per human expert rater,
  each row an image with a binary `interesting` label.
- `ai_csvs/<model>_<protocol>_<shot>_990.csv` — matching AI ratings for the
  same 990-image set, one file per model × protocol × few-shot/no-shot
  combination (mirrors the protocols described in the `weird_and_wonderful`
  and `hubble` READMEs).
- `*_few_shot.py` / `*_no_shot.py` — the scripts that generated the AI CSVs
  above.
- `html_grid.py` / `html_tournament.py` / `plots_only_html.py` — report/plot
  generation.
- `expert_ai_agreement_report.html` — the rendered agreement report.

## `regression/`

- `expert_vs_volunteer.csv` — training data: expert selection percentage,
  volunteer rating, and AI anomaly score per image.
- `expert_regression.py` — fits a linear regression (scikit-learn) predicting
  expert agreement from volunteer + AI signals, applied to
  `tournament_filter_no_shot.csv` / `tournament_filter_few_shot.csv`.
- `predicted_vs_actual.png` / `model_comparison.png` — regression fit plots.
- `tournament_*.py` — tournament runs feeding the regression's target files.
