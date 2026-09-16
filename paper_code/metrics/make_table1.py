"""
Recompute Table 1 (performance) and Table 2 (token usage and cost) from the
scoring CSVs.

Reads nothing but those CSVs: no image catalogue, no API key, no network.
This package ships code only, so set PAPER_RESULTS_DIR to the tree that holds
them (see common/paths.py).

Each CSV carries its own ground truth in the `interesting` column (1 when an
AnomalyMatch anomaly lies within 3" of the cutout centre, applied by the
scoring script itself) and its own cost record in a trailing
`# TOKEN USAGE SUMMARY` footer, so every number in both tables comes from the
run that produced it.

Ties at the cutoff are handled by the exact expected recall under a random
ordering of the tied band, not by shuffling:

    recall@N = ( TP_above + TP_tie * (N - n_above) / n_tie ) / P

where n_above/TP_above count images strictly above the boundary score,
n_tie/TP_tie are the size and true positives of the band straddling rank N,
and P is the total number of positives. This is what the paper reports as
"Expected Recall@2000 (accounting for ties)", and it matters because Likert
scores are coarse (51 distinct values over 20,000 images), so the band at a
threshold is often large.

Cost per 10,000 images comes from whichever record the run actually has:

    - a recorded provider charge (`# TotalCostUSD`) when present, which is
      what the paper quotes for Qwen - it is what OpenRouter billed, not an
      estimate; or
    - token counts priced at PRICING, with Gemini's thinking tokens billed at
      the output rate, which is how the paper's Gemini figures are built.

Usage:  PAPER_RESULTS_DIR=/path/to/analysis python metrics/make_table1.py
"""

import os
import glob
import statistics

# The stage folders are siblings, so put the package root on the path to
# reach common/paths.py (see its docstring).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import RESULTS_DIR, SUBSET_DIR
from common import scores_csv


# ── CONFIGURATION ───────────────────────────────────────────────────────────

# USD per million tokens, as quoted in the caption of the cost table. Gemini
# thinking tokens are billed at the output rate.
PRICING = {
    "gemini": {"input": 0.25, "output": 1.50},
    "qwen":   {"input": 0.39, "output": 2.34},
}

# The equal-budget review set the paper compares methods at.
REVIEW_SET_SIZE = 2000

# One entry per row of Table 1: the run to read, the score cut, and the
# published values. Filenames are tried in order against RESULTS_DIR and then
# SUBSET_DIR, so a run named for its seed or for its protocol both resolve.
# Published values are carried only so the script can report a diff; they are
# never used in the computation.
ROWS = [
    dict(model="Gemini 3.1 Flash-Lite", method="Likert", provider="gemini",
         threshold=33,
         files=["gemini_likert_scores.csv", "gemini_likert_scores_sept10.csv",
                "gemini_likert_1.csv"],
         published=dict(n_selected=2095, recall=0.982, precision=0.078,
                        expected_recall=0.976, cost_per_10k=5.05)),
    dict(model="Qwen3.5-397B-A17B", method="Likert", provider="qwen",
         threshold=36,
         files=["qwen_likert_scores.csv", "qwen_likert_1.csv"],
         published=dict(n_selected=1984, recall=0.958, precision=0.081,
                        expected_recall=0.959, cost_per_10k=6.69)),
    dict(model="Gemini 3.1 Flash-Lite", method="Tournament", provider="gemini",
         threshold=10,
         files=["gemini_tournament_scores.csv", "gemini_tournament_1.csv"],
         published=dict(n_selected=2019, recall=0.958, precision=0.079,
                        expected_recall=0.949, cost_per_10k=3.66)),
    dict(model="Qwen3.5-397B-A17B", method="Tournament", provider="qwen",
         threshold=10,
         files=["qwen_tournament_scores.csv", "qwen_tournament_1.csv"],
         published=dict(n_selected=2099, recall=0.934, precision=0.074,
                        expected_recall=0.890, cost_per_10k=1.80)),
]

# The seed-to-seed spread quoted in the Discussion, measured on the Gemini
# Likert replicates at that section's threshold. The glob deliberately matches
# only the "_scores" naming of the current runs: mixing replicates from
# different sampling setups would report a spread that is really a setup
# difference, which at threshold 33 is the larger effect by far.
SEED_SPREAD_GLOB = "gemini_likert_scores*.csv"
SEED_SPREAD_THRESHOLD = 33


# ── 1. LOADING ──────────────────────────────────────────────────────────────

def find_run(names):
    """First of `names` that exists in RESULTS_DIR or SUBSET_DIR, else None."""
    for name in names:
        for directory in (RESULTS_DIR, SUBSET_DIR):
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


# ── 2. METRICS ──────────────────────────────────────────────────────────────

def recall_curve(scores, truth, total_positives):
    """
    Exact expected recall@N for every N = 1..len(scores), indexed by N-1.

    Within a band of equal scores each true positive contributes its
    fractional share of the slots N leaves open in that band, so the curve is
    exact rather than dependent on how ties happen to be ordered.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    curve = [0.0] * n

    seen, seen_tp, i = 0, 0.0, 0
    while i < n:
        band_score = ranked[i][1]
        j, band_tp = i, 0
        while j < n and ranked[j][1] == band_score:
            if truth.get(ranked[j][0]):
                band_tp += 1
            j += 1
        band = j - i
        for k in range(1, band + 1):  # N = seen + k
            curve[seen + k - 1] = (seen_tp + band_tp * (k / band)) / total_positives
        seen += band
        seen_tp += band_tp
        i = j

    return curve


def cost_per_10k(tokens, n_images, provider):
    """
    USD per 10,000 images, and how it was derived.

    A recorded provider charge wins over a token-price estimate: it is what
    was actually billed. Returns (cost, source) with cost None when the run
    logged neither.
    """
    if tokens.get("cost_usd") is not None:
        return tokens["cost_usd"] / n_images * 10_000, "recorded charge"

    if not tokens["input"] and not tokens["output"]:
        return None, "no cost record in the CSV"

    rates = PRICING[provider]
    billed_output = tokens["output"] + tokens["thinking"]
    dollars = (tokens["input"] / 1e6) * rates["input"] + (billed_output / 1e6) * rates["output"]
    source = "token counts" + ("" if tokens["thinking"] else ", no thinking tokens logged")
    return dollars / n_images * 10_000, source


def evaluate(path, threshold, provider):
    """Every Table 1 and Table 2 quantity for one run."""
    data = scores_csv.read(path)
    scores, truth = data.scores(), data.ground_truth()
    n_images = len(scores)
    positives = sum(truth.values())
    if not positives:
        raise SystemExit(f"{path}: no rows have interesting=1")

    shortlist = [f for f in scores if scores[f] >= threshold]
    true_positives = sum(1 for f in shortlist if truth[f])
    cost, cost_source = cost_per_10k(data.tokens, n_images, provider)

    return dict(
        path=path, n_images=n_images, positives=positives,
        n_selected=len(shortlist),
        recall=true_positives / positives,
        precision=true_positives / len(shortlist) if shortlist else 0.0,
        expected_recall=recall_curve(scores, truth, positives)[REVIEW_SET_SIZE - 1],
        cost_per_10k=cost, cost_source=cost_source,
        in_per_img=data.tokens["input"] / n_images,
        out_per_img=data.tokens["output"] / n_images,
        think_per_img=data.tokens["thinking"] / n_images,
    )


# ── 3. REPORT ───────────────────────────────────────────────────────────────

def fmt(value, spec):
    return "--" if value is None else format(value, spec)


def main():
    results, missing = [], []
    for row in ROWS:
        path = find_run(row["files"])
        if path is None:
            missing.append(row)
            continue
        results.append((row, evaluate(path, row["threshold"], row["provider"])))

    if not results:
        raise SystemExit(
            f"None of the expected run CSVs were found under {RESULTS_DIR} or "
            f"{SUBSET_DIR}.\nSet PAPER_RESULTS_DIR to the tree holding them.")

    print(f"Table 1 - performance against the 167 reference anomalies "
          f"(Expected Recall@{REVIEW_SET_SIZE})")
    print(f"  {'Model':23}{'Method':12}{'Thr':>5}{'N Sel':>7}{'Recall':>8}"
          f"{'Prec':>7}{'ExpR@%d' % REVIEW_SET_SIZE:>10}{'$/10k':>8}")
    for row, m in results:
        threshold = f"R{row['threshold']}" if row["method"] == "Tournament" else row["threshold"]
        print(f"  {row['model']:23}{row['method']:12}{threshold:>5}{m['n_selected']:>7}"
              f"{m['recall']:>8.3f}{m['precision']:>7.3f}{m['expected_recall']:>10.3f}"
              f"{fmt(m['cost_per_10k'], '.2f'):>8}")

    print(f"\nTable 2 - token usage per image and cost per 10,000 images")
    print(f"  {'Model':23}{'Method':12}{'Input':>9}{'Output':>9}{'Thinking':>10}"
          f"{'$/10k':>8}   cost from")
    for row, m in results:
        thinking = f"{m['think_per_img']:.1f}" if m["think_per_img"] else "--"
        print(f"  {row['model']:23}{row['method']:12}{m['in_per_img']:>9.1f}"
              f"{m['out_per_img']:>9.1f}{thinking:>10}"
              f"{fmt(m['cost_per_10k'], '.2f'):>8}   {m['cost_source']}")

    print("\nAgreement with the published tables (to the precision printed there)")
    for row, m in results:
        diffs = []
        for key, want in row["published"].items():
            got = m[key]
            if got is None:
                diffs.append(f"{key}: not computable")
            elif key == "n_selected":
                if got != want:
                    diffs.append(f"{key}: got {got}, paper {want}")
            elif abs(got - want) >= (5e-3 if key == "cost_per_10k" else 5e-4):
                diffs.append(f"{key}: got {got:.3f}, paper {want}")
        status = "exact" if not diffs else "; ".join(diffs)
        print(f"  {row['model']} {row['method']}: {status}")
        print(f"      from {os.path.basename(m['path'])}")

    for row in missing:
        print(f"  {row['model']} {row['method']}: NO RUN FOUND "
              f"(looked for {', '.join(row['files'])})")

    # Seed-to-seed spread, quoted in the Discussion.
    seeds = sorted(set(glob.glob(os.path.join(RESULTS_DIR, SEED_SPREAD_GLOB))
                       + glob.glob(os.path.join(SUBSET_DIR, SEED_SPREAD_GLOB))))
    if len(seeds) == 1:
        print(f"\nSeed spread - only one Gemini Likert run found "
              f"({os.path.basename(seeds[0])}); the Discussion quotes three.")
    if len(seeds) >= 2:
        reps = [evaluate(p, SEED_SPREAD_THRESHOLD, "gemini") for p in seeds]
        recalls = [r["recall"] for r in reps]
        sizes = [r["n_selected"] for r in reps]
        print(f"\nSeed spread - Gemini Likert at score >= {SEED_SPREAD_THRESHOLD} "
              f"({len(reps)} runs)")
        print(f"  recall        {statistics.mean(recalls):.3f} "
              f"+/- {statistics.stdev(recalls):.3f}")
        print(f"  images selected {min(sizes)}-{max(sizes)}")
        for p, r in zip(seeds, reps):
            print(f"    {os.path.basename(p):34} N={r['n_selected']:>5}  "
                  f"recall={r['recall']:.3f}")


if __name__ == "__main__":
    main()
