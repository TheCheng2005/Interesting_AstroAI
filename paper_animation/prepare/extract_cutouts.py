"""
Pull the cutouts the animation shows out of the HDF5, as PNGs.

The archive stores each cutout as JPEG bytes in a variable-length array,
keyed by SourceID in a parallel `filenames` dataset. Images are written to
assets/<SourceID>.png, upscaled with nearest-neighbour so the 150x150 pixels
stay crisp rather than blurring when manim scales them up.

Data is located the way ../paper_code locates it: HUBBLE_DATA_DIR for the
cutout HDF5, PAPER_RESULTS_DIR for the tree holding the scoring CSVs.

Usage:  HUBBLE_DATA_DIR=... PAPER_RESULTS_DIR=... python prepare/extract_cutouts.py
"""

import csv
import io
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "paper_code"))
from common import scores_csv  # noqa: E402
from common.paths import HDF5_PATH, RESULTS_DIR, SUBSET_DIR  # noqa: E402

ASSETS = os.path.join(ROOT, "assets")
# The recall wall draws all 167 reference anomalies at once, a few dozen pixels
# each, so they are kept separately and at native resolution.
ANOM_DIR = os.path.join(ASSETS, "anomalies")
CAND_DIR = os.path.join(ASSETS, "candidates")
UPSCALE = 3                      # 150 -> 450 px, nearest-neighbour
LENS = "4001415925883"           # SDSS J1205+4910, 50/50 Likert and 10/10 Tournament

# The Sept-10 Gemini Likert run behind Table 1, and a Gemini Tournament run.
LIKERT_CSV = "gemini_likert_scores_sept10.csv"
TOURNAMENT_CSV = "gemini_tournament_1.csv"
CANDIDATE_MIN_SCORE = 45

# The 30 highest-scoring of the 102 released candidates, in the paper's
# Appendix B order. The 102 themselves cannot be recomputed here: reaching
# them needs the O'Ryan et al. (2023) interacting-galaxy catalogue, which is
# not on this machine. Every id below is checked against the run CSV.
APPENDIX_B = [
    "4000736938609", "4000787734924", "4000813117448", "4000987073198",
    "4000739672025", "4000767317950", "4000994445010", "4000995502977",
    "4001058626020", "4001244667344", "4001302576886", "4001417239471",
    "4000721971379", "4000722902315", "4000755600966", "4000756620603",
    "4000830667463", "4000866128822", "4000880785798", "4000900784961",
    "4000953983040", "4000960858908", "4001037563111", "4001058624829",
    "4001070552086", "4001156420724", "4001231647667", "4001243324096",
    "4001243325091", "4001421277365",
]

# Counts the paper states that no local file can reproduce: of the 138, 33
# match the interacting-galaxy catalogue and 1 the galaxy-mentions dataset,
# leaving 104; a manual literature review removes 2 more, leaving 102.
PAPER_AFTER_CATALOGUES = 104
PAPER_RELEASED = 102

N_HIGH, N_MID, N_LOW = 12, 12, 40   # how many of each band to pull


def find_csv(name):
    """A run CSV under RESULTS_DIR or SUBSET_DIR, as metrics/make_table1.py looks."""
    for directory in (RESULTS_DIR, SUBSET_DIR):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    sys.exit(f"{name} not found under {RESULTS_DIR} or {SUBSET_DIR}; "
             "set PAPER_RESULTS_DIR to the tree holding it.")


def pick(scores, lo, hi, n, exclude):
    """n filenames whose score is in [lo, hi], highest first, excluding some."""
    band = sorted(((v, f) for f, v in scores.items() if lo <= v <= hi and f not in exclude),
                  reverse=True)
    return [f for _, f in band[:n]]


def main():
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(ANOM_DIR, exist_ok=True)
    os.makedirs(CAND_DIR, exist_ok=True)
    likert = scores_csv.read(find_csv(LIKERT_CSV))
    tourn = scores_csv.read(find_csv(TOURNAMENT_CSV))
    ls, ts = likert.scores(), tourn.scores()
    truth = likert.ground_truth()

    wanted = [LENS]
    wanted += pick(ls, 45, 50, N_HIGH, set(wanted))
    wanted += pick(ls, 18, 30, N_MID, set(wanted))
    wanted += pick(ls, 0, 2, N_LOW, set(wanted))
    wanted = list(dict.fromkeys(wanted))
    anomalies = sorted((f for f, v in truth.items() if v),
                       key=lambda f: -(ls.get(f) or 0))
    candidates = APPENDIX_B
    print(f"{len(wanted)} cutouts wanted, {len(anomalies)} reference anomalies")

    with h5py.File(HDF5_PATH, "r") as h5:
        names = h5["filenames"]
        need = set(wanted) | set(anomalies) | set(candidates)
        index = {}
        for i, raw in enumerate(names[:]):
            name = raw.decode() if isinstance(raw, bytes) else str(raw)
            if name in need:
                index[name] = i
        missing = [w for w in wanted if w not in index]
        if missing:
            print(f"  WARNING: {len(missing)} not in the HDF5: {missing[:3]}")

        meta = {}
        for name in wanted:
            if name not in index:
                continue
            img = Image.open(io.BytesIO(bytes(h5["images"][index[name]])))
            img = img.convert("L").resize(
                (img.width * UPSCALE, img.height * UPSCALE), Image.NEAREST)
            img.save(os.path.join(ASSETS, f"{name}.png"))
            meta[name] = {
                "likert": ls.get(name),
                "tournament": ts.get(name),
                "anomaly": bool(truth.get(name)),
            }

        wall = {}
        for name in anomalies:
            if name not in index:
                continue
            img = Image.open(io.BytesIO(bytes(h5["images"][index[name]])))
            img.convert("L").save(os.path.join(ANOM_DIR, f"{name}.png"))
            wall[name] = {"likert": ls.get(name), "tournament": ts.get(name)}

        cand = {}
        for name in candidates:
            if name not in index:
                continue
            img = Image.open(io.BytesIO(bytes(h5["images"][index[name]])))
            img.convert("L").save(os.path.join(CAND_DIR, f"{name}.png"))
            cand[name] = {"score": ls.get(name), "anomaly": bool(truth.get(name))}

    stray = [n for n, v in cand.items()
             if v["anomaly"] or (v["score"] or 0) < CANDIDATE_MIN_SCORE]
    if stray:
        print(f"  WARNING: {len(stray)} Appendix B ids fail the paper's own "
              f"criteria: {stray[:3]}")

    above = [f for f, v in ls.items() if v >= CANDIDATE_MIN_SCORE]
    funnel = {
        "scored": len(ls),
        "above": len(above),
        "nonreference": sum(1 for f in above if not truth.get(f)),
        "after_catalogues": PAPER_AFTER_CATALOGUES,
        "released": PAPER_RELEASED,
        "shown": len(cand),
    }
    with open(os.path.join(ASSETS, "candidates.json"), "w") as f:
        json.dump({"tiles": cand, "funnel": funnel}, f, indent=2)
    print(f"wrote {len(cand)} candidate tiles to {CAND_DIR}")
    print("  funnel:", funnel)

    # The recall sweep walks the score cutoff down; at each stop it needs the
    # size of the shortlist and how many of the 167 it has caught.
    curve = {}
    for c in range(50, 32, -1):
        shortlist = [f for f, v in ls.items() if v >= c]
        curve[c] = [len(shortlist), sum(1 for f in shortlist if truth.get(f))]

    with open(os.path.join(ASSETS, "anomalies.json"), "w") as f:
        json.dump({"tiles": wall, "curve": curve}, f, indent=2)
    print(f"wrote {len(wall)} anomaly tiles to {ANOM_DIR}")
    print(f"  cutoff 33 -> {curve[33][0]} shortlisted, {curve[33][1]}/{len(wall)} found")

    with open(os.path.join(ASSETS, "cutouts.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {len(meta)} PNGs to {ASSETS}")
    hi = sum(1 for m in meta.values() if (m['likert'] or 0) >= 45)
    print(f"  {hi} score >=45, {sum(1 for m in meta.values() if m['anomaly'])} are reference anomalies")


if __name__ == "__main__":
    main()
