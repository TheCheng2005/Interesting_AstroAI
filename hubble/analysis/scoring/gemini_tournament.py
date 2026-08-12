"""
Iterative retention filtering for Hubble image scoring (2x2 grid format).

Logic:
- Build the labeled dataset from the HDF5 file.
- Each image is labeled interesting if it is the nearest-image match
  (within MATCH_RADIUS_ARCSEC) to at least one Interesting.csv entry.
- In TEST_MODE, every interesting image is injected manually and the rest of
  the TEST_NUM_IMAGES budget is filled with a random subset of boring images.
  Outside TEST_MODE, every image is used.
- In each round, active images are randomly reshuffled.
- Images are grouped into 2x2 grids of four.
- Gemini sees a 2x2 grid labeled Image 1 (top-left), Image 2 (top-right),
  Image 3 (bottom-left), and Image 4 (bottom-right).
- Gemini evaluates each image independently and decides whether each image
  is scientifically interesting enough to keep for human inspection
  (no reasoning requested, to save tokens).
- Gemini may keep any number of the four images (zero through four).
- Kept images advance to the next round.
- Rejected images are removed from the active pool.
- Exactly MAX_ROUNDS rounds are run (or fewer if all images are eliminated).
- The whole experiment is run once per seed in RANDOM_SEEDS, producing one CSV
  file per seed.

No few-shot examples are used: this dataset is inherently different from the
Galaxy Zoo Weird & Wonderful dataset, so no artifact/interesting/boring
example images are uploaded.

Final output CSV columns:
    Filename, ImageScore, interesting, classification, SourceRA, SourceDec

ImageScore:
- 0-MAX_ROUNDS: the number of rounds the image survived (was kept).
  An image rejected in round 1 scores 0; one that survives all rounds
  scores MAX_ROUNDS.
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

GROUP_SIZE = 4  # 2x2 grid of images evaluated per API call
QUAD_THUMB_W = 240
QUAD_THUMB_H = 240

MATCH_RADIUS_ARCSEC = 3.0  # radius search: match each Interesting.csv item to its nearest image

MAX_ROUNDS = 10

# The experiment is run once per seed, producing one CSV per seed.
RANDOM_SEEDS = [44, 45, 46]

TEST_MODE = True  # Set to True for test run with limited images
TEST_NUM_IMAGES = 20000  # Number of images in the test subset


# All test-subset result CSVs land here, named {provider}_{format}_{run}.csv so
# the dashboard groups replicates by (provider, format) automatically.
SUBSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subset_test")


def results_filename(run_idx, seed):
    """Output CSV path for a given run. Test subsets go to subset_test/ as
    gemini_tournament_<run_idx>.csv; run_idx is the 1-based replicate number."""
    if TEST_MODE:
        os.makedirs(SUBSET_DIR, exist_ok=True)
        return os.path.join(SUBSET_DIR, f"gemini_tournament_{run_idx}.csv")
    return "tournament_seed{}_{}.csv".format(seed, time.strftime("%m-%d_%H"))


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


config = types.GenerateContentConfig(
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
    Stitches up to 4 image records into a 2x2 grid with labels.

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
      automatically to the next round
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


# ── 4. WORKER FUNCTION ─────────────────────────────────────────────────────

def process_matches(args):
    worker_id, round_number, groups, api_key, progress_queue = args
    client = genai.Client(api_key=api_key)
    checkpoint_file = f"checkpoint_tournament_worker_{worker_id}.csv"

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
                    config=config,
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


# ── 5. SINGLE-RUN ORCHESTRATOR ─────────────────────────────────────────────

def run_experiment(all_images, api_key, random_seed, results_file):
    """
    Run the full 2x2-grid retention filtering for a single prepared record set
    and write the results to results_file.
    """
    # Clean up any stale checkpoints from a crashed prior run.
    for i in range(NUM_CORES):
        cp = f"checkpoint_tournament_worker_{i + 1}.csv"
        if os.path.exists(cp):
            os.remove(cp)

    print(f"Total Images: {len(all_images)}")
    print(f"Running exactly {MAX_ROUNDS} rounds of filtering.")

    # Scores accumulate: +1 each round an image is kept. Range: 0-MAX_ROUNDS.
    scores = {record["index"]: 0 for record in all_images}

    records_by_index = {record["index"]: record for record in all_images}

    active_records = all_images[:]
    random.seed(random_seed)

    round_number = 1
    grand_total_input_tokens = 0
    grand_total_output_tokens = 0

    while round_number <= MAX_ROUNDS:
        random.shuffle(active_records)

        round_start_count = len(active_records)
        all_groups, bye_records = split_into_groups(active_records)

        print("\n" + "=" * 70)
        print(f"Round {round_number}")
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
        grand_total_input_tokens += round_input_tokens
        grand_total_output_tokens += round_output_tokens

        print(
            f"\nRound {round_number} token usage: "
            f"input={round_input_tokens:,}  output={round_output_tokens:,}  "
            f"total={round_input_tokens + round_output_tokens:,}"
        )

        # ── Round aggregation ──────────────────────────────────────────────
        print(f"\nAggregating Round {round_number} results...")

        round_decisions = {}
        seen_this_round = set()

        for i in range(NUM_CORES):
            cp = f"checkpoint_tournament_worker_{i + 1}.csv"

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

        rejected_by_model = [
            record
            for record in active_records
            if round_decisions[record["index"]] == 0
        ]

        removed_indices = {record["index"] for record in rejected_by_model}

        next_pool = [
            record
            for record in active_records
            if record["index"] not in removed_indices
        ]

        # Increment score for every image that survived this round.
        for record in next_pool:
            scores[record["index"]] += 1

        removed_this_round = len(removed_indices)

        print(f"Round {round_number} complete.")
        print(f"Images entering:     {round_start_count}")
        print(f"Model rejections:    {removed_this_round}")
        print(f"Images advancing:    {len(next_pool)}")
        print(
            f"Cumulative tokens:   input={grand_total_input_tokens:,}  "
            f"output={grand_total_output_tokens:,}  "
            f"total={grand_total_input_tokens + grand_total_output_tokens:,}"
        )

        active_records = next_pool
        round_number += 1

        if not active_records:
            print("All images eliminated. Stopping early.")
            break

    grand_total_tokens = grand_total_input_tokens + grand_total_output_tokens

    print("\n" + "=" * 70)
    print("Pairwise retention filtering complete.")
    print(f"Rounds completed:            {round_number - 1}")
    print(f"Images surviving all rounds: {len(active_records)}")
    print(f"Total input tokens:          {grand_total_input_tokens:,}")
    print(f"Total output tokens:         {grand_total_output_tokens:,}")
    print(f"Total tokens:                {grand_total_tokens:,}")
    print("=" * 70)

    # ── FINAL AGGREGATE ─────────────────────────────────────────────────────
    print("\nAggregating final results...")

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Filename",
            "ImageScore",
            "interesting",
            "classification",
            "SourceRA",
            "SourceDec",
        ])

        # Highest-scoring images first.
        for image_index in sorted(scores, key=scores.get, reverse=True):
            score = scores[image_index]
            record = records_by_index[image_index]

            writer.writerow([
                record["filename"],
                score,
                record["interesting"],
                record["classification"],
                record["SourceRA"],
                record["SourceDec"],
            ])

        # Token usage summary appended as trailing metadata rows.
        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", grand_total_input_tokens])
        writer.writerow(["# TotalOutputTokens", grand_total_output_tokens])
        writer.writerow(["# TotalTokens", grand_total_tokens])

    print(f"Done! Results saved to {results_file}")
    print(
        f"Token usage summary: input={grand_total_input_tokens:,}  "
        f"output={grand_total_output_tokens:,}  "
        f"total={grand_total_tokens:,}"
    )


# ── 6. MAIN: RUN ONCE PER SEED ─────────────────────────────────────────────

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
