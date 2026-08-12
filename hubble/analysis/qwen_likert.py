"""
I ask Qwen (via OpenRouter) to pick any number of interesting images from a 4x4 grid.
Qwen scores each selected image on a scale of 1-5.

HDF5-only dataset version:
- All images are loaded from the HDF5 file.
- Each image is labeled interesting if it is the nearest-image match
  (within a 10" radius) to at least one Interesting.csv entry.

- Final test set:
      ALL images from the HDF5 file (no sampling/capping).

- Output CSV columns:
      index, filename, imagescore, interesting, classification

  For boring/background images, classification is left blank.
"""

import os
import csv
import time
import random
import math
import queue
import io
import base64
from multiprocessing import Pool, Manager

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont


# ── 1. CONFIGURATION ───────────────────────────────────────────────────────

HDF5_PATH = "../hubble_data/10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5"
INTERESTING_CSV_PATH = "../hubble_data/Interesting.csv"
PARQUET_PATH = "../hubble_data/10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.parquet"

OPENROUTER_MODEL = "qwen/qwen3.5-397b-a17b"

NUM_CORES = 25
ROUNDS_PER_IMAGE = 10
GRID_DIM = 4
BATCH_SIZE = GRID_DIM * GRID_DIM

MATCH_RADIUS_ARCSEC = 3.0  # radius search: match each Interesting.csv item to its nearest image

# The experiment is run once per seed, producing one CSV per seed.
RANDOM_SEEDS = [44]

TEST_MODE = True  # Set to True for test run with limited images
TEST_NUM_IMAGES = 20000  # Number of images in the test subset


# All test-subset result CSVs land here, named {provider}_{format}_{run}.csv so
# the dashboard groups replicates by (provider, format) automatically.
SUBSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subset_test")


def results_filename(run_idx, seed):
    """Output CSV path for a given run. Test subsets go to subset_test/ as
    qwen_likert_<run_idx>.csv; run_idx is the 1-based replicate number."""
    if TEST_MODE:
        os.makedirs(SUBSET_DIR, exist_ok=True)
        return os.path.join(SUBSET_DIR, f"qwen_likert_{run_idx}.csv")
    return "qwen_10M_10arcsec_dedup_seed{}_{}.csv".format(seed, time.strftime("%m-%d_%H"))


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


class SelectedImages(BaseModel):
    selections: list[SelectedImage] = Field(
        description="List of selected images. Empty list if none are interesting."
    )


few_shot_context = ""
if os.path.exists("GEMINI.md"):
    with open("GEMINI.md", "r", encoding="utf-8") as f:
        few_shot_context = f"\n\n{f.read()}"


SYSTEM_INSTRUCTION = (
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
    nearest image (a 10" radius search, keeping the closest candidate), so an
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

        # Load all image records and label by CSV
        all_records = []
        for hdf5_idx in range(n_total):
            filename = decode_hdf5_filename(filenames[hdf5_idx])
            img_bytes = bytes(images[hdf5_idx])

            source_id = os.path.splitext(filename)[0]
            classification = interesting_metadata.get(source_id, "")
            is_interesting = 1 if classification else 0

            # Get coordinates from parquet data
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


# ── 3. GRID UTILITIES ──────────────────────────────────────────────────────

def create_grid(image_records):
    """
    Stitches 16 image records into a 4x4 grid with borders and labels.

    image_records is a list of dictionaries:
        {
            "index": int,
            "filename": str,
            "image_bytes": bytes,
            "interesting": int,
            "classification": str,
        }

    Some entries may be None for padding.
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


# ── 4. WORKER FUNCTION ─────────────────────────────────────────────────────

def process_batches(args):
    worker_id, batches, api_key, progress_queue = args

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    checkpoint_file = f"checkpoint_grid_worker_{worker_id}.csv"

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

            p_tok, c_tok, cost = 0, 0, 0.0

            try:
                grid_b64 = base64.b64encode(grid_bytes).decode()

                response = retry_api(
                    client.beta.chat.completions.parse,
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this 4x4 grid of astronomical images. "
                                    "Select any scientifically interesting images and "
                                    "rate each selected image from 1 to 5."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{grid_b64}"
                                },
                            },
                        ]},
                    ],
                    response_format=SelectedImages,
                    temperature=0.1,
                    extra_body={
                        "usage": {"include": True},
                        "reasoning": {"enabled": False},
                        # DigitalOcean is the only OpenRouter backend for this
                        # model that ignores reasoning.enabled=False, silently
                        # running full chain-of-thought (~40x more tokens/cost).
                        "provider": {"ignore": ["DigitalOcean"]},
                    },
                )

                # Extract token usage and actual dollar cost from the response
                if hasattr(response, "usage") and response.usage:
                    p_tok = getattr(response.usage, "prompt_tokens", 0) or 0
                    c_tok = getattr(response.usage, "completion_tokens", 0) or 0
                    cost = getattr(response.usage, "cost", 0.0) or 0.0

                parsed = response.choices[0].message.parsed
                data = parsed.selections if parsed is not None else []

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

            # Send back the completed flag along with the token usage and cost for this batch
            progress_queue.put((1, p_tok, c_tok, cost))

    return worker_id


# ── 5. SINGLE-RUN ORCHESTRATOR ─────────────────────────────────────────────

def run_experiment(all_images, api_key, random_seed, results_file):
    """
    Run the full likert scoring experiment for a single prepared record set
    and write the results to results_file.
    """
    # Clean up any stale checkpoints from a crashed prior run.
    for i in range(NUM_CORES):
        cp = f"checkpoint_grid_worker_{i + 1}.csv"
        if os.path.exists(cp):
            os.remove(cp)

    # Every image appears in ROUNDS_PER_IMAGE different randomized batches.
    master_pool = all_images * ROUNDS_PER_IMAGE

    random.seed(random_seed)
    random.shuffle(master_pool)

    all_batches = []

    for i in range(0, len(master_pool), BATCH_SIZE):
        batch = master_pool[i: i + BATCH_SIZE]

        while len(batch) < BATCH_SIZE:
            batch.append(None)

        all_batches.append(batch)

    print(f"Total Images:      {len(all_images)}")
    print(f"Total Evaluations: {len(master_pool)}")
    print(f"Total 4x4 Grids:   {len(all_batches)}\n")

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
    total_cost = 0.0

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
                    # Get tuple containing progress, token counts, and cost
                    item = progress_queue.get(timeout=1.0)
                    if isinstance(item, tuple) and len(item) == 4:
                        _, p_tok, c_tok, cost = item
                        total_prompt_tokens += p_tok
                        total_candidate_tokens += c_tok
                        total_cost += cost

                    pbar.update(1)
                    completed += 1
                except queue.Empty:
                    if async_result.ready():
                        break

        async_result.get()

    # ── AGGREGATE RESULTS ──────────────────────────────────────────────────

    print("\nAggregating results...")

    scores = {
        record["index"]: 0
        for record in all_images
    }

    filenames = {
        record["index"]: record["filename"]
        for record in all_images
    }

    interesting_labels = {
        record["index"]: record["interesting"]
        for record in all_images
    }

    classifications = {
        record["index"]: record.get("classification", "")
        for record in all_images
    }

    source_ras = {
        record["index"]: record.get("SourceRA", "")
        for record in all_images
    }

    source_decs = {
        record["index"]: record.get("SourceDec", "")
        for record in all_images
    }

    for i in range(NUM_CORES):
        cp = f"checkpoint_grid_worker_{i + 1}.csv"

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

        for image_index in sorted(scores, key=scores.get, reverse=True):
            writer.writerow([
                image_index,
                filenames[image_index],
                scores[image_index],
                interesting_labels[image_index],
                classifications[image_index],
                source_ras[image_index],
                source_decs[image_index],
            ])

        # Token usage summary appended as trailing metadata rows, matching the
        # single-elim / tournament formats (input and output tracked separately).
        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", total_prompt_tokens])
        writer.writerow(["# TotalOutputTokens", total_candidate_tokens])
        writer.writerow(["# TotalTokens", total_prompt_tokens + total_candidate_tokens])
        writer.writerow(["# TotalCostUSD", f"{total_cost:.4f}"])

    n_interesting = sum(interesting_labels.values())
    n_boring = len(interesting_labels) - n_interesting

    print(f"Done! Results saved to {results_file}")
    print(
        f"Final dataset contained {n_boring} boring images "
        f"and {n_interesting} interesting images."
    )
    print("\n" + "═" * 40)
    print(" TOKEN USAGE SUMMARY")
    print("═" * 40)
    print(f"Total Prompt Tokens:     {total_prompt_tokens:,}")
    print(f"Total Candidate Tokens:  {total_candidate_tokens:,}")
    print(f"Total Tokens:            {total_prompt_tokens + total_candidate_tokens:,}")
    print(f"Total Cost (USD):        ${total_cost:.4f}")
    print("═" * 40 + "\n")


# ── 6. MAIN: RUN ONCE PER SEED ─────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found.")

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
