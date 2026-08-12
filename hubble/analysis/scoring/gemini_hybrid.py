"""
Hybrid Hubble image scoring: one tournament filtering round, then Likert scoring.

This combines the two existing formats to save tokens:

Phase 1 (tournament filter, gemini_tournament.py format):
- All active images are shuffled and grouped into 2x2 grids of four.
- Gemini sees a 2x2 grid labeled Image 1 (top-left), Image 2 (top-right),
  Image 3 (bottom-left), and Image 4 (bottom-right), and decides for each image
  independently whether it is scientifically interesting enough to keep
  (no reasoning requested, to save tokens).
- Exactly one round is run. Kept images advance to Phase 2. Rejected images are
  eliminated and receive a final ImageScore of 0.
- This first round eliminates most of the boring images cheaply, so the more
  expensive Likert phase only scores the survivors.

Phase 2 (Likert scoring, gemini_likert.py format):
- Only the images that survived Phase 1 are scored.
- Each surviving image is shown ROUNDS_PER_IMAGE (10) times in randomized
  4x4 grids. For each grid, Gemini selects the interesting images and scores
  each selected image on a 1-5 scale. Scores are summed across all appearances.
- A surviving image that is never selected still ends up with a score of 0,
  same as an image eliminated in Phase 1.

Dataset construction (shared with the other formats):
- Build the labeled dataset from the HDF5 file.
- Each image is labeled interesting if it is the nearest-image match
  (within MATCH_RADIUS_ARCSEC) to at least one Interesting.csv entry.
- In TEST_MODE, every interesting image is injected manually and the rest of
  the TEST_NUM_IMAGES budget is filled with a random subset of boring images.
  Outside TEST_MODE, every image is used.
- The whole experiment is run once per seed in RANDOM_SEEDS, producing one CSV
  file per seed.

No few-shot examples are used in the tournament phase: this dataset is inherently
different from the Galaxy Zoo Weird & Wonderful dataset, so no artifact/
interesting/boring example images are uploaded.

Final output CSV columns (same as the Likert format):
    index, filename, imagescore, interesting, classification, SourceRA, SourceDec

imagescore:
- 0 for any image eliminated in the tournament round, or for a survivor never
  selected in the Likert phase.
- Otherwise the sum of the 1-5 Likert scores across its appearances.
"""

import os
import csv
import time
import random
import math
import queue
import io
from multiprocessing import Pool, Manager

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont


# ── 1. CONFIGURATION ───────────────────────────────────────────────────────

HDF5_PATH = "../hubble_data/10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5"
INTERESTING_CSV_PATH = "../hubble_data/Interesting.csv"
PARQUET_PATH = "../hubble_data/10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.parquet"

NUM_CORES = 25

# Phase 1 (tournament filter): 2x2 grid of images evaluated per API call.
GROUP_SIZE = 4
QUAD_THUMB_W = 240
QUAD_THUMB_H = 240
TOURNAMENT_ROUNDS = 1  # a single elimination round precedes the Likert phase

# Phase 2 (Likert scoring): 4x4 grid, each surviving image shown 10 times.
ROUNDS_PER_IMAGE = 10
GRID_DIM = 4
BATCH_SIZE = GRID_DIM * GRID_DIM

MATCH_RADIUS_ARCSEC = 3.0  # radius search: match each Interesting.csv item to its nearest image

# The experiment is run once per seed, producing one CSV per seed.
RANDOM_SEEDS = [44, 45, 46]

TEST_MODE = True  # Set to True for test run with limited images
TEST_NUM_IMAGES = 20000  # Number of images in the test subset


# All test-subset result CSVs land here, named {provider}_{format}_{run}.csv so
# the dashboard groups replicates by (provider, format) automatically.
SUBSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subset_test")


def results_filename(run_idx, seed):
    """Output CSV path for a given run. Test subsets go to subset_test/ as
    gemini_hybrid_<run_idx>.csv; run_idx is the 1-based replicate number."""
    if TEST_MODE:
        os.makedirs(SUBSET_DIR, exist_ok=True)
        return os.path.join(SUBSET_DIR, f"gemini_hybrid_{run_idx}.csv")
    return "hybrid_seed{}_{}.csv".format(seed, time.strftime("%m-%d_%H"))


# ── Phase 1 (tournament filter) schema and config ───────────────────────────

class InterestingSelection(BaseModel):
    KeepImage1: bool = Field(
        description=(
            "True if Image 1 (top-left) is scientifically interesting enough "
            "to keep for human inspection. False otherwise."
        )
    )
    KeepImage2: bool = Field(
        description=(
            "True if Image 2 (top-right) is scientifically interesting enough "
            "to keep for human inspection. False otherwise."
        )
    )
    KeepImage3: bool = Field(
        description=(
            "True if Image 3 (bottom-left) is scientifically interesting enough "
            "to keep for human inspection. False otherwise."
        )
    )
    KeepImage4: bool = Field(
        description=(
            "True if Image 4 (bottom-right) is scientifically interesting enough "
            "to keep for human inspection. False otherwise."
        )
    )


tournament_config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer evaluating a 2x2 grid of four astronomical "
        "images, labeled Image 1 (top-left), Image 2 (top-right), Image 3 "
        "(bottom-left), and Image 4 (bottom-right). "
        "Evaluate each image independently. Do not force a winner. "
        "For each image, decide whether it is scientifically interesting enough "
        "to keep for human inspection. "
        "It is acceptable to keep any number of the four images (none, some, or all). "
        "Interesting images may show unusual morphology, asymmetry, interactions, "
        "merger-like features, arcs, rings, tails, shells, clumps, distortions, "
        "rare-looking objects, or other features worth human inspection. "
        "Reject artifacts, blank images, noisy frames, and obvious non-astronomical defects. "
        "Return only whether to keep each of Image 1, Image 2, Image 3, and Image 4. "
        "No explanation needed."
    ),
    response_mime_type="application/json",
    response_schema=InterestingSelection,
    temperature=0.1,
)


# ── Phase 2 (Likert scoring) schema and config ──────────────────────────────

class SelectedImage(BaseModel):
    GridIndex: int = Field(
        ge=1,
        le=16,
        description=(
            "Index 1-16 of the interesting image in the 4x4 grid, "
            "reading top-to-bottom, left-to-right."
        ),
    )
    Score: int = Field(
        ge=1,
        le=5,
        description=(
            "Scientific interest score from 1 mildly interesting "
            "to 5 exceptionally interesting."
        ),
    )


few_shot_context = ""
if os.path.exists("GEMINI.md"):
    with open("GEMINI.md", "r", encoding="utf-8") as f:
        few_shot_context = f"\n\n{f.read()}"


likert_config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer evaluating a 4x4 grid of astronomical images. "
        "The images are indexed 1-4 in the top row, 5-8 in the second row, "
        "9-12 in the third row, and 13-16 in the bottom row. "
        "Your task:\n"
        "1. Select any images that are scientifically interesting. "
        "Skip blank, empty, or uninformative frames.\n"
        "2. For each selected image, assign a Score from 1 to 5 "
        "(no explanation needed):\n"
        "   1 = mildly interesting, 2 = somewhat interesting, 3 = interesting, "
        "   4 = very interesting, 5 = exceptionally interesting.\n"
        "Return a list of selected images. If none are interesting, return an empty list.\n"
        f"{few_shot_context}"
    ),
    response_mime_type="application/json",
    response_schema=list[SelectedImage],
    temperature=0.1,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
)


# ── 2. DATA LOADING ────────────────────────────────────────────────────────

def decode_hdf5_filename(name):
    """
    Decode an HDF5 filename entry into a normal Python string.
    """
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def load_interesting_metadata(csv_path, coordinates, radius_arcsec=MATCH_RADIUS_ARCSEC):
    """
    Match each Interesting.csv item to its nearest image (by SourceRA/SourceDec
    in `coordinates`) within radius_arcsec, and return matched image SourceID ->
    Classification.

    Every Interesting.csv item is matched independently to its own single
    nearest image (a radius search, keeping the closest candidate), so an
    image can only end up "interesting" if it is the closest image to at least
    one Interesting.csv object and that object is within radius_arcsec of it.

    Expected header:
        SourceID,Classification,RA,Dec,Referenced
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Interesting.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"SourceID", "Classification", "RA", "Dec"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Interesting.csv is missing required columns: {sorted(missing_columns)}"
        )

    catalog_ids = list(coordinates.keys())
    catalog_ra = np.array([coordinates[sid]["SourceRA"] for sid in catalog_ids])
    catalog_dec = np.array([coordinates[sid]["SourceDec"] for sid in catalog_ids])
    catalog_coords = SkyCoord(ra=catalog_ra * u.deg, dec=catalog_dec * u.deg)

    interesting_coords = SkyCoord(ra=df["RA"].values * u.deg, dec=df["Dec"].values * u.deg)

    nearest_idx, sep2d, _ = interesting_coords.match_to_catalog_sky(catalog_coords)

    metadata = {}
    n_items_matched = 0

    for row_idx, (match_idx, sep) in enumerate(zip(nearest_idx, sep2d.arcsec)):
        if sep <= radius_arcsec:
            n_items_matched += 1
            matched_source_id = catalog_ids[match_idx]
            classification = str(df.iloc[row_idx]["Classification"]).strip()
            metadata[matched_source_id] = classification

    print(
        f"Matched {n_items_matched} / {len(df)} Interesting.csv entries to an "
        f"image within {radius_arcsec}\" ({len(metadata)} unique images, "
        f"{n_items_matched - len(metadata)} shared their nearest image with "
        f"another Interesting.csv entry)."
    )

    return metadata


def load_parquet_coordinates(parquet_path):
    """
    Load SourceID -> (SourceRA, SourceDec) mapping from parquet file.
    Uses pandas to read the parquet file.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to load parquet coordinates. "
            "Please install it in your environment."
        )

    df = pd.read_parquet(parquet_path)
    coordinates = {}

    for _, row in df.iterrows():
        source_id = str(int(row["SourceID"]))
        coordinates[source_id] = {
            "SourceRA": row["SourceRA"],
            "SourceDec": row["SourceDec"],
        }

    print(f"Loaded coordinates for {len(coordinates)} sources from parquet.")
    return coordinates


def build_labeled_records(hdf5_path, interesting_csv_path, coordinates):
    """
    Build the labeled dataset from HDF5 with images labeled from Interesting.csv.

    Loads every image from the HDF5 file and labels each one interesting if its
    SourceID matches an Interesting.csv entry. Records keep their HDF5 order and
    are NOT shuffled or subset here; call prepare_run_records() per seed for that.
    """
    print("Loading all images from HDF5...")

    interesting_metadata = load_interesting_metadata(interesting_csv_path, coordinates)

    with h5py.File(hdf5_path, "r") as f:
        filenames = f["filenames"]
        images = f["images"]

        n_total = len(images)

        all_records = []
        for hdf5_idx in range(n_total):
            filename = decode_hdf5_filename(filenames[hdf5_idx])
            img_bytes = bytes(images[hdf5_idx])

            source_id = os.path.splitext(filename)[0]
            classification = interesting_metadata.get(source_id, "")
            is_interesting = 1 if classification else 0

            coord_data = coordinates.get(source_id, {"SourceRA": None, "SourceDec": None})

            all_records.append({
                "index": hdf5_idx,
                "filename": filename,
                "image_bytes": img_bytes,
                "interesting": is_interesting,
                "classification": classification,
                "SourceRA": coord_data["SourceRA"],
                "SourceDec": coord_data["SourceDec"],
            })

    n_interesting_found = sum(r["interesting"] for r in all_records)

    print(f"HDF5 contains {n_total} images.")
    print(f"Found {n_interesting_found} interesting images in CSV.")

    return all_records


def prepare_run_records(all_records, random_seed):
    """
    Produce the record list for a single run at the given seed.

    In TEST_MODE, every interesting image is injected manually and the remainder
    of the TEST_NUM_IMAGES budget is filled with a random subset of boring
    images (so all interesting images always appear in the subset). Outside
    TEST_MODE, every image is used. Records are copied, shuffled, and re-indexed
    0..N-1 so runs at different seeds do not share mutable index state.
    """
    rng = random.Random(random_seed)
    records = [dict(r) for r in all_records]

    if TEST_MODE:
        interesting = [r for r in records if r["interesting"]]
        boring = [r for r in records if not r["interesting"]]

        n_boring_needed = max(0, TEST_NUM_IMAGES - len(interesting))
        n_boring_needed = min(n_boring_needed, len(boring))
        boring_subset = rng.sample(boring, n_boring_needed)

        records = interesting + boring_subset
        print(
            f"\n*** TEST MODE (seed {random_seed}): injected all "
            f"{len(interesting)} interesting + {len(boring_subset)} boring "
            f"= {len(records)} images ***"
        )

    rng.shuffle(records)

    for i, record in enumerate(records):
        record["index"] = i

    n_true_interesting = sum(record["interesting"] for record in records)
    n_true_boring = len(records) - n_true_interesting

    print("\nDataset prepared.")
    print(f"  Boring images:      {n_true_boring}")
    print(f"  Interesting images: {n_true_interesting}")
    print(f"  Total images:       {len(records)}")
    if len(records) > 0:
        print(f"  Interesting rate:   {n_true_interesting / len(records):.5f}\n")

    return records


# ── 3. IMAGE UTILITIES ─────────────────────────────────────────────────────

def create_quad_image(quad_records):
    """
    Stitches up to 4 image records into a 2x2 grid with labels (Phase 1).

    Cells are laid out as:
        Image 1 (top-left)     Image 2 (top-right)
        Image 3 (bottom-left)  Image 4 (bottom-right)

    quad_records is a list of up to 4 records, each with an "image_bytes" key.
    Entries may be None for padding.
    """
    margin = 8
    label_h = 32
    cell_w = QUAD_THUMB_W
    cell_h = QUAD_THUMB_H + label_h

    canvas_w = cell_w * 2 + margin * 3
    canvas_h = cell_h * 2 + margin * 3

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            22,
        )
    except Exception:
        font = ImageFont.load_default()

    for i, record in enumerate(quad_records):
        if record is None:
            continue

        row = i // 2
        col = i % 2

        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)

        label = f"Image {i + 1}"
        draw.text(
            (x + 6, y),
            label,
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )

        try:
            img_bytes = record["image_bytes"]

            with Image.open(io.BytesIO(img_bytes)) as img:
                img = img.convert("RGB").resize((QUAD_THUMB_W, QUAD_THUMB_H))
                canvas.paste(img, (x, y + label_h))

        except Exception as e:
            print(
                f"Failed to place image index={record.get('index')} "
                f"filename={record.get('filename')}: {e}"
            )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf


def split_into_groups(active_records, group_size=GROUP_SIZE):
    """
    Shuffle active_records before calling this.

    Returns:
    - groups: list of (group_number, [record, ...]) each of length group_size
    - bye_records: leftover records (fewer than group_size) carried forward
      automatically (kept) to the next phase
    """
    active_records = active_records[:]
    bye_records = []

    n_bye = len(active_records) % group_size
    for _ in range(n_bye):
        bye_records.append(active_records.pop())

    groups = []
    for i in range(0, len(active_records), group_size):
        group_number = len(groups) + 1
        groups.append((group_number, active_records[i:i + group_size]))

    return groups, bye_records


def create_grid(image_records):
    """
    Stitches 16 image records into a 4x4 grid with borders and labels (Phase 2).

    image_records is a list of dictionaries with an "image_bytes" key. Some
    entries may be None for padding.
    """
    thumb_w, thumb_h = 200, 200
    margin = 4

    grid_w = (thumb_w * GRID_DIM) + (margin * (GRID_DIM + 1))
    grid_h = (thumb_h * GRID_DIM) + (margin * (GRID_DIM + 1))

    canvas = Image.new("RGB", (grid_w, grid_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            20,
        )
    except Exception:
        font = ImageFont.load_default()

    for i, record in enumerate(image_records):
        if record is None:
            continue

        row = i // GRID_DIM
        col = i % GRID_DIM

        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + margin)

        try:
            img_bytes = record["image_bytes"]

            with Image.open(io.BytesIO(img_bytes)) as img:
                img = img.convert("RGB")
                img = img.resize((thumb_w, thumb_h))
                canvas.paste(img, (x, y))

            draw.text(
                (x + 5, y + 5),
                str(i + 1),
                fill="white",
                font=font,
                stroke_width=1,
                stroke_fill="black",
            )

        except Exception as e:
            print(
                f"Failed to place image index={record.get('index')} "
                f"filename={record.get('filename')}: {e}"
            )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)

    return buf


# ── 4. WORKER FUNCTIONS ────────────────────────────────────────────────────

def process_matches(args):
    """Phase 1 worker: evaluate 2x2 grids and record keep/reject decisions."""
    worker_id, round_number, groups, api_key, progress_queue = args
    client = genai.Client(api_key=api_key)
    checkpoint_file = f"checkpoint_hybrid_tourn_worker_{worker_id}.csv"

    total_input_tokens = 0
    total_output_tokens = 0

    def retry_api(func, *args, **kwargs):
        delays = [1, 2, 4, 8, 16]
        for i, delay in enumerate(delays):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == len(delays) - 1:
                    raise e
                time.sleep(delay)

    with open(checkpoint_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if os.stat(checkpoint_file).st_size == 0:
            writer.writerow([
                "Round",
                "Group",
                "Index",
                "Filename",
                "Kept",
                "InputTokens",
                "OutputTokens",
                "TotalTokens",
            ])

        for group_number, group_records in groups:
            quad_buf = create_quad_image(group_records)
            quad_bytes = quad_buf.getvalue()

            try:
                quad_part = types.Part.from_bytes(
                    data=quad_bytes,
                    mime_type="image/png",
                )

                contents = [
                    (
                        "Evaluate this 2x2 grid of four astronomical images. "
                        "Image 1 is top-left, Image 2 is top-right, Image 3 is "
                        "bottom-left, and Image 4 is bottom-right. "
                        "Evaluate each image independently. Do not force a winner. "
                        "For each image, decide whether it is scientifically interesting "
                        "enough to keep for human inspection."
                    ),
                    quad_part,
                ]

                response = retry_api(
                    client.models.generate_content,
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                    config=tournament_config,
                )

                data = response.parsed

                usage = response.usage_metadata
                input_tok = getattr(usage, "prompt_token_count", 0) or 0
                output_tok = getattr(usage, "candidates_token_count", 0) or 0
                total_tok = getattr(usage, "total_token_count", 0) or (input_tok + output_tok)

                total_input_tokens += input_tok
                total_output_tokens += output_tok

                keep_flags = [
                    bool(data.KeepImage1),
                    bool(data.KeepImage2),
                    bool(data.KeepImage3),
                    bool(data.KeepImage4),
                ]

                for idx, record in enumerate(group_records):
                    kept = 1 if keep_flags[idx] else 0

                    writer.writerow([
                        round_number,
                        group_number,
                        record["index"],
                        record["filename"],
                        kept,
                        input_tok,
                        output_tok,
                        total_tok,
                    ])

                kept_names = [
                    group_records[i]["filename"]
                    for i in range(len(group_records))
                    if keep_flags[i]
                ]

                if len(kept_names) == 0:
                    kept_str = "NONE"
                else:
                    kept_str = ", ".join(kept_names)

                print(
                    f"Worker {worker_id} | Round {round_number}, Group {group_number}: "
                    f"kept: {kept_str} [tokens in={input_tok} out={output_tok}]"
                )

                f.flush()
                os.fsync(f.fileno())

            except Exception as e:
                print(
                    f"Worker {worker_id} error in Round {round_number}, "
                    f"Group {group_number}: {e}"
                )

                # Conservative fallback:
                # If Gemini/API fails, keep every image in the group so we do not
                # accidentally discard potentially interesting images due to a
                # technical failure.
                for record in group_records:
                    writer.writerow([
                        round_number,
                        group_number,
                        record["index"],
                        record["filename"],
                        1,
                        0,
                        0,
                        0,
                    ])

                f.flush()
                os.fsync(f.fileno())

            progress_queue.put(1)

    return worker_id, total_input_tokens, total_output_tokens


def process_batches(args):
    """Phase 2 worker: evaluate 4x4 grids and record Likert scores."""
    worker_id, batches, api_key, progress_queue = args

    client = genai.Client(api_key=api_key)
    checkpoint_file = f"checkpoint_hybrid_grid_worker_{worker_id}.csv"

    def retry_api(func, *args, **kwargs):
        delays = [1, 2, 4, 8, 16]

        for i, delay in enumerate(delays):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == len(delays) - 1:
                    raise e

                print(
                    f"Worker {worker_id}: API error, retrying in {delay}s. "
                    f"Error: {e}"
                )
                time.sleep(delay)

    with open(checkpoint_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if os.stat(checkpoint_file).st_size == 0:
            writer.writerow(["Index", "Filename", "Score"])

        for batch_records in batches:
            grid_buf = create_grid(batch_records)
            grid_bytes = grid_buf.getvalue()

            p_tok, c_tok = 0, 0

            try:
                grid_part = types.Part.from_bytes(
                    data=grid_bytes,
                    mime_type="image/png",
                )

                contents = [
                    (
                        "Analyze this 4x4 grid of astronomical images. "
                        "Select any scientifically interesting images and "
                        "rate each selected image from 1 to 5."
                    ),
                    grid_part,
                ]

                response = retry_api(
                    client.models.generate_content,
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                    config=likert_config,
                )

                # Extract token usage from the response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    p_tok = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    c_tok = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                data = response.parsed

                if data is None:
                    data = []

                print(
                    f"Worker {worker_id} processed a batch. "
                    f"Found {len(data)} selected images."
                )

                for item in data:
                    idx = item.GridIndex - 1

                    if (
                        0 <= idx < len(batch_records)
                        and batch_records[idx] is not None
                    ):
                        record = batch_records[idx]

                        writer.writerow([
                            record["index"],
                            record["filename"],
                            item.Score,
                        ])

                f.flush()
                os.fsync(f.fileno())

            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

            # Send back the completed flag along with the token usage for this batch
            progress_queue.put((1, p_tok, c_tok))

    return worker_id


# ── 5. PHASE ORCHESTRATORS ─────────────────────────────────────────────────

def run_tournament_filter(active_records, api_key, random_seed):
    """
    Phase 1: a single 2x2-grid retention round.

    Returns (survivors, eliminated, input_tokens, output_tokens), where
    survivors advance to the Likert phase and eliminated images receive a
    final score of 0.
    """
    # Clean up any stale checkpoints from a crashed prior run.
    for i in range(NUM_CORES):
        cp = f"checkpoint_hybrid_tourn_worker_{i + 1}.csv"
        if os.path.exists(cp):
            os.remove(cp)

    random.seed(random_seed)
    random.shuffle(active_records)

    round_number = 1
    round_start_count = len(active_records)
    all_groups, bye_records = split_into_groups(active_records)

    print("\n" + "=" * 70)
    print(f"PHASE 1: Tournament filter (round {round_number})")
    print(f"Images entering round: {round_start_count}")
    print(f"2x2 grids this round:  {len(all_groups)}")

    if bye_records:
        print(
            f"{len(bye_records)} leftover image(s) carried forward "
            f"automatically: {[r['filename'] for r in bye_records]}"
        )

    print("=" * 70)

    manager = Manager()
    progress_queue = manager.Queue()

    chunk_size = math.ceil(len(all_groups) / NUM_CORES)
    worker_args = []

    for i in range(NUM_CORES):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(all_groups))

        if start < len(all_groups):
            worker_args.append((
                i + 1,
                round_number,
                all_groups[start:end],
                api_key,
                progress_queue,
            ))

    with Pool(processes=NUM_CORES) as pool:
        async_result = pool.map_async(process_matches, worker_args)

        with tqdm(
            total=len(all_groups),
            desc=f"Processing Round {round_number}",
            unit="grid",
        ) as pbar:
            completed = 0

            while completed < len(all_groups):
                try:
                    _ = progress_queue.get(timeout=1.0)
                    pbar.update(1)
                    completed += 1
                except queue.Empty:
                    if async_result.ready():
                        break

        worker_results = async_result.get()

    round_input_tokens = sum(r[1] for r in worker_results)
    round_output_tokens = sum(r[2] for r in worker_results)

    print(
        f"\nRound {round_number} token usage: "
        f"input={round_input_tokens:,}  output={round_output_tokens:,}  "
        f"total={round_input_tokens + round_output_tokens:,}"
    )

    # ── Round aggregation ──────────────────────────────────────────────────
    print(f"\nAggregating Round {round_number} results...")

    round_decisions = {}
    seen_this_round = set()

    for i in range(NUM_CORES):
        cp = f"checkpoint_hybrid_tourn_worker_{i + 1}.csv"

        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)

                for row in reader:
                    # row:
                    # Round, Group, Index, Filename, Kept,
                    # InputTokens, OutputTokens, TotalTokens
                    if not row:
                        continue

                    try:
                        row_round = int(row[0])
                    except (ValueError, IndexError):
                        continue

                    if row_round != round_number:
                        continue

                    try:
                        image_index = int(row[2])
                        kept = int(row[4])
                    except (ValueError, IndexError):
                        continue

                    seen_this_round.add(image_index)
                    round_decisions[image_index] = kept

            os.remove(cp)

    # Add leftover images carried forward by logistics.
    for bye_record in bye_records:
        bye_index = bye_record["index"]
        seen_this_round.add(bye_index)
        round_decisions[bye_index] = 1

    expected_seen = {record["index"] for record in active_records}
    actual_seen = set(seen_this_round)

    if expected_seen != actual_seen:
        missing = expected_seen - actual_seen
        extra = actual_seen - expected_seen

        raise RuntimeError(
            f"Round {round_number} aggregation mismatch.\n"
            f"Missing indices: {list(missing)[:10]}\n"
            f"Extra indices: {list(extra)[:10]}\n"
            f"Stopping to avoid corrupting results."
        )

    survivors = [
        record
        for record in active_records
        if round_decisions[record["index"]] == 1
    ]
    eliminated = [
        record
        for record in active_records
        if round_decisions[record["index"]] == 0
    ]

    print(f"Phase 1 complete.")
    print(f"Images entering:     {round_start_count}")
    print(f"Model rejections:    {len(eliminated)}")
    print(f"Images advancing:    {len(survivors)}")

    return survivors, eliminated, round_input_tokens, round_output_tokens


def run_likert_scoring(survivors, api_key, random_seed):
    """
    Phase 2: Likert scoring of the surviving images.

    Each survivor is shown ROUNDS_PER_IMAGE times in randomized 4x4 grids and
    scored 1-5 per selection. Returns (scores, input_tokens, output_tokens),
    where scores maps image index -> summed Likert score (0 if never selected).
    """
    # Clean up any stale checkpoints from a crashed prior run.
    for i in range(NUM_CORES):
        cp = f"checkpoint_hybrid_grid_worker_{i + 1}.csv"
        if os.path.exists(cp):
            os.remove(cp)

    scores = {record["index"]: 0 for record in survivors}

    if not survivors:
        print("\nPHASE 2: no survivors to score. Skipping Likert phase.")
        return scores, 0, 0

    # Every surviving image appears in ROUNDS_PER_IMAGE different randomized batches.
    master_pool = survivors * ROUNDS_PER_IMAGE

    random.seed(random_seed)
    random.shuffle(master_pool)

    all_batches = []

    for i in range(0, len(master_pool), BATCH_SIZE):
        batch = master_pool[i: i + BATCH_SIZE]

        while len(batch) < BATCH_SIZE:
            batch.append(None)

        all_batches.append(batch)

    print("\n" + "=" * 70)
    print("PHASE 2: Likert scoring")
    print(f"Surviving Images:  {len(survivors)}")
    print(f"Total Evaluations: {len(master_pool)}")
    print(f"Total 4x4 Grids:   {len(all_batches)}")
    print("=" * 70)

    manager = Manager()
    progress_queue = manager.Queue()

    chunk_size = math.ceil(len(all_batches) / NUM_CORES)
    worker_args = []

    for i in range(NUM_CORES):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(all_batches))

        if start < len(all_batches):
            worker_args.append((
                i + 1,
                all_batches[start:end],
                api_key,
                progress_queue,
            ))

    total_prompt_tokens = 0
    total_candidate_tokens = 0

    with Pool(processes=NUM_CORES) as pool:
        async_result = pool.map_async(process_batches, worker_args)

        with tqdm(
            total=len(all_batches),
            desc="Processing Grids",
            unit="grid",
        ) as pbar:
            completed = 0

            while completed < len(all_batches):
                try:
                    # Get tuple containing progress and token counts
                    item = progress_queue.get(timeout=1.0)
                    if isinstance(item, tuple) and len(item) == 3:
                        _, p_tok, c_tok = item
                        total_prompt_tokens += p_tok
                        total_candidate_tokens += c_tok

                    pbar.update(1)
                    completed += 1
                except queue.Empty:
                    if async_result.ready():
                        break

        async_result.get()

    # ── Aggregate Likert scores ────────────────────────────────────────────
    print("\nAggregating Likert scores...")

    for i in range(NUM_CORES):
        cp = f"checkpoint_hybrid_grid_worker_{i + 1}.csv"

        if os.path.exists(cp):
            with open(cp, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)

                for row in reader:
                    if not row:
                        continue

                    try:
                        image_index = int(row[0])
                        base_score = int(row[2])
                    except (ValueError, IndexError):
                        continue

                    if image_index not in scores:
                        continue

                    scores[image_index] += base_score

            os.remove(cp)

    return scores, total_prompt_tokens, total_candidate_tokens


# ── 6. SINGLE-RUN ORCHESTRATOR ─────────────────────────────────────────────

def run_experiment(all_images, api_key, random_seed, results_file):
    """
    Run the hybrid experiment (tournament filter then Likert scoring) for a
    single prepared record set and write the results to results_file.
    """
    print(f"Total Images: {len(all_images)}")

    records_by_index = {record["index"]: record for record in all_images}

    # ── Phase 1: tournament filter ─────────────────────────────────────────
    survivors, eliminated, tourn_in_tok, tourn_out_tok = run_tournament_filter(
        all_images[:], api_key, random_seed
    )

    # ── Phase 2: Likert scoring of survivors ───────────────────────────────
    survivor_scores, likert_in_tok, likert_out_tok = run_likert_scoring(
        survivors, api_key, random_seed
    )

    # ── Final scores: eliminated images score 0; survivors get their Likert sum ─
    scores = {record["index"]: 0 for record in all_images}
    for image_index, score in survivor_scores.items():
        scores[image_index] = score

    total_input_tokens = tourn_in_tok + likert_in_tok
    total_output_tokens = tourn_out_tok + likert_out_tok
    total_tokens = total_input_tokens + total_output_tokens

    print("\n" + "=" * 70)
    print("Hybrid scoring complete.")
    print(f"Images eliminated (Phase 1): {len(eliminated)}")
    print(f"Images scored (Phase 2):     {len(survivors)}")
    print(
        f"Phase 1 tokens: input={tourn_in_tok:,}  output={tourn_out_tok:,}"
    )
    print(
        f"Phase 2 tokens: input={likert_in_tok:,}  output={likert_out_tok:,}"
    )
    print(f"Total input tokens:  {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Total tokens:        {total_tokens:,}")
    print("=" * 70)

    # ── FINAL AGGREGATE ─────────────────────────────────────────────────────
    print("\nAggregating final results...")

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "index",
            "filename",
            "imagescore",
            "interesting",
            "classification",
            "SourceRA",
            "SourceDec",
        ])

        # Highest-scoring images first.
        for image_index in sorted(scores, key=scores.get, reverse=True):
            record = records_by_index[image_index]

            writer.writerow([
                image_index,
                record["filename"],
                scores[image_index],
                record["interesting"],
                record["classification"],
                record["SourceRA"],
                record["SourceDec"],
            ])

        # Token usage summary appended as trailing metadata rows.
        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", total_input_tokens])
        writer.writerow(["# TotalOutputTokens", total_output_tokens])
        writer.writerow(["# TotalTokens", total_tokens])

    n_interesting = sum(record["interesting"] for record in all_images)
    n_boring = len(all_images) - n_interesting

    print(f"Done! Results saved to {results_file}")
    print(
        f"Final dataset contained {n_boring} boring images "
        f"and {n_interesting} interesting images."
    )
    print("\n" + "═" * 40)
    print(" TOKEN USAGE SUMMARY")
    print("═" * 40)
    print(f"Total Input Tokens:   {total_input_tokens:,}")
    print(f"Total Output Tokens:  {total_output_tokens:,}")
    print(f"Total Tokens:         {total_tokens:,}")
    print("═" * 40 + "\n")


# ── 7. MAIN: RUN ONCE PER SEED ─────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("JB_API_KEY")
    if not api_key:
        raise ValueError("JB_API_KEY not found.")

    print("Loading parquet coordinates...")
    coordinates = load_parquet_coordinates(PARQUET_PATH)

    print("Loading HDF5 dataset with Interesting.csv labels...")
    all_labeled = build_labeled_records(HDF5_PATH, INTERESTING_CSV_PATH, coordinates)

    for run_idx, seed in enumerate(RANDOM_SEEDS, start=1):
        print("\n" + "#" * 70)
        print(f"# RUN {run_idx}/{len(RANDOM_SEEDS)}  (random seed {seed})")
        print("#" * 70)

        run_images = prepare_run_records(all_labeled, seed)
        run_experiment(run_images, api_key, seed, results_filename(run_idx, seed))
