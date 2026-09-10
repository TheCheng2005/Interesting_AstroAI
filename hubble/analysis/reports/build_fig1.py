"""
Rebuild Figure 1 of the workshop paper.

Changes requested after the meeting with Jo:
  (a) Tournament series removed - its ranking carries no order inside the
      round-10 tie, so a curve through it implies precision it does not have.
      Likert Select is now drawn as discrete points, one per integer score
      threshold (@50, @49, @48, ...), against the number of images a reviewer
      would have to look at to reach that threshold. That makes the tie
      structure visible: where a score band is large, consecutive points jump
      a long way along x.
  (b) unchanged in content - Gemini Tournament score distribution, seed 44.
  (c) unchanged - three high-scoring Likert Select cutouts.

Palette (validated with the dataviz validator, light surface, all-pairs):
  panel a  Gemini #2a78d6 (circles) / Qwen #eb6834 (diamonds)
           -> all checks PASS
  panel b  bulk #6b7280 (deliberate neutral) / anomaly #008300
           -> CVD dE 12.5, normal-vision dE 19.5, both >=3:1 contrast; the
              validator's chroma-floor note on the gray is intended, since it
              encodes "the rest of the population", not an identity.
"""

import os, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# The stage folders are siblings, so put the analysis root on the path to
# reach common/paths.py (see its docstring).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import SUBSET_DIR as SUBSET, HDF5_PATH as HDF5, PARQUET_PATH as PARQUET, ANALYSIS_ROOT

# The paper lives alongside this project, not inside it.
OUT = os.path.join(os.path.dirname(ANALYSIS_ROOT), "Workshop Paper", "poster_figure_composite.png")

N_ANOM = 167
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY, GREEN = "#6b7280", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

PANELS = [  # (label, RA, Dec, score, kind)
    ("i",   181.418362,  49.174771, "50/50", "gravitational ring"),
    ("ii",  202.009029, -31.515268, "49/50", "relativistic jet"),
    ("iii", 334.403219,   0.835206, "47/50", "jellyfish galaxy"),
]


def load(p):
    d = pd.read_csv(p, dtype=str)
    d.columns = [c.lower() for c in d.columns]
    d = d[d["interesting"].notna()].copy()
    d["interesting"] = d.interesting.astype(int)
    d["imagescore"] = d.imagescore.astype(float)
    return d


def threshold_points(runs, lo=20, hi=50):
    """For each integer threshold, mean/sd of (images reviewed, recall)."""
    ts = list(range(hi, lo - 1, -1))
    N, R = [], []
    for t in ts:
        ns = [len(d[d.imagescore >= t]) for d in runs]
        rs = [d[d.imagescore >= t].interesting.sum() / N_ANOM for d in runs]
        N.append((np.mean(ns), np.std(ns, ddof=1) if len(ns) > 1 else 0.0))
        R.append((np.mean(rs), np.std(rs, ddof=1) if len(rs) > 1 else 0.0))
    return ts, np.array(N), np.array(R)


def cutouts():
    import h5py
    pq = pd.read_parquet(PARQUET)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cat = SkyCoord(ra=pq.SourceRA.values * u.deg, dec=pq.SourceDec.values * u.deg)
    ids = []
    for _, ra, dec, _, _ in PANELS:
        t = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        i = int(t.separation(cat).arcsec.argmin())
        ids.append(str(int(pq.SourceID.values[i])))
    with h5py.File(HDF5, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["filenames"][:]]
        loc = {os.path.splitext(n)[0]: k for k, n in enumerate(names)}
        return [Image.open(io.BytesIO(bytes(f["images"][loc[s]]))).convert("L") for s in ids]


# ---------------------------------------------------------------- data
GL = [load(os.path.join(SUBSET, f"gemini_likert_{i}.csv")) for i in (1, 2, 3)]
QL = [load(os.path.join(SUBSET, "qwen_likert_1.csv"))]
GT1 = load(os.path.join(SUBSET, "gemini_tournament_1.csv"))

gts, gN, gR = threshold_points(GL)
qts, qN, qR = threshold_points(QL)
imgs = cutouts()

# ---------------------------------------------------------------- layout
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.1,
})
fig = plt.figure(figsize=(16, 9), dpi=200, facecolor="white")
gs = GridSpec(2, 2, figure=fig, width_ratios=[2.45, 1], height_ratios=[1, 1],
              left=.055, right=.985, top=.94, bottom=.075, wspace=.13, hspace=.30)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[1, 0])
gsc = gs[:, 1].subgridspec(3, 1, hspace=.08)
axc = [fig.add_subplot(gsc[i]) for i in range(3)]

# ---------------------------------------------------------------- (a)
for (N, R, col, mk, lab, ann) in [
    (gN, gR, BLUE, "o", "Gemini 3.1 Flash-Lite  Likert Select", True),
    (qN, qR, ORANGE, "D", "Qwen3.5-397B-A17B  Likert Select", False),
]:
    axa.errorbar(N[:, 0], R[:, 0] * 100, xerr=N[:, 1], yerr=R[:, 1] * 100,
                 fmt=mk, ms=9, mfc=col, mec="white", mew=1.4,
                 ecolor=col, elinewidth=1.4, capsize=0, alpha=.95,
                 linestyle="none", label=lab, zorder=3)

LABEL_AT = (50, 45, 40, 35, 30)   # the dense upper tail needs no labels
for t, n, r in zip(gts, gN[:, 0], gR[:, 0]):
    if t in LABEL_AT:                      # below the point, unless it would
        dy = -19 if r * 100 > 8 else 13    # fall off the bottom of the axes
        axa.annotate(f"@{t}", (n, r * 100), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=12.5, color=BLUE)
for t, n, r in zip(qts, qN[:, 0], qR[:, 0]):
    if t in LABEL_AT:
        dy = 12                            # Qwen labels always sit above
        axa.annotate(f"@{t}", (n, r * 100), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=12.5, color=ORANGE)

axa.axvline(100, color=INK2, lw=1.1, ls=(0, (5, 4)), zorder=1)
axa.annotate("100 images reviewed", (100, 3), xytext=(6, 0),
             textcoords="offset points", rotation=90, va="bottom",
             fontsize=12.5, color=INK2)
axa.set_xscale("log")
axa.set_xlim(2, 4200)
axa.set_ylim(0, 100)
axa.set_xlabel("Images a reviewer must inspect (log scale)")
axa.set_ylabel(f"Recall of the {N_ANOM} anomalies (%)")
axa.set_title("(a)  Recall vs review budget, one point per Likert score threshold",
              loc="left", fontsize=16, fontweight="bold", pad=10)
axa.grid(True, which="major", color=GRID, lw=.9)
axa.grid(True, which="minor", color=GRID, lw=.5, alpha=.6)
axa.set_axisbelow(True)
for sp in ("top", "right"):
    axa.spines[sp].set_visible(False)
axa.legend(loc="upper left", frameon=False, fontsize=13.5, handletextpad=.4)

# ---------------------------------------------------------------- (b)
h = GT1.groupby("imagescore").agg(n=("interesting", "size"),
                                  a=("interesting", "sum")).reindex(range(11), fill_value=0)
bulk = (h.n - h.a).values
anom = h.a.values
x = np.arange(11)
w = .40
axb.bar(x - w / 2, bulk, w, color=GRAY, edgecolor="white", linewidth=1.2,
        label="Non-interesting", zorder=3)
axb.bar(x + w / 2, np.where(anom > 0, anom, np.nan), w, color=GREEN,
        edgecolor="white", linewidth=1.2, label="Anomaly", zorder=3)
for xi, v in zip(x, bulk):
    if v:
        axb.annotate(f"{v:,}", (xi - w / 2, v), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=11.5, color=GRAY)
for xi, v in zip(x, anom):
    if v:
        axb.annotate(f"{v:,}", (xi + w / 2, v), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=11.5,
                     color=GREEN, fontweight="bold")
axb.set_yscale("log")
axb.set_ylim(.6, 5e4)
axb.set_xticks(x)
axb.set_xlabel("Rounds survived (Tournament score)")
axb.set_ylabel("Images")
axb.set_title("(b)  Gemini Tournament score distribution, seed 44",
              loc="left", fontsize=16, fontweight="bold", pad=10)
axb.grid(True, axis="y", color=GRID, lw=.9)
axb.set_axisbelow(True)
for sp in ("top", "right"):
    axb.spines[sp].set_visible(False)
axb.legend(loc="upper center", frameon=False, fontsize=13.5, ncol=2)

# ---------------------------------------------------------------- (c)
for ax, im, (lab, ra, dec, sc, kind) in zip(axc, imgs, PANELS):
    ax.imshow(np.asarray(im), cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK2); sp.set_linewidth(1.1)
    ax.text(.03, .95, lab, transform=ax.transAxes, va="top", ha="left",
            fontsize=17, fontweight="bold", color="white")
    ax.text(.97, .05, sc, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=14, color="white")
axc[0].set_title("(c)  Highest-scoring Likert Select cutouts",
                 loc="left", fontsize=16, fontweight="bold", pad=10)

fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=.12)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")
im = Image.open(OUT); print("size:", im.size, " aspect H/W:", round(im.size[1]/im.size[0], 3))
