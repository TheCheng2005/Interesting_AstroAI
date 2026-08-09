"""
Pairwise elimination tournament for Hubble image scoring, using Qwen
(via OpenRouter) instead of Gemini.

Logic:
- Load all images from the HDF5 file (no sampling/capping unless TEST_MODE).
- Each image is labeled interesting if it is the nearest-image match
  (within MATCH_RADIUS_ARCSEC) to at least one Interesting.csv entry.
- In each round, surviving images are randomly reshuffled.
- If there is an odd number of images, one receives a "bye" and advances automatically.
- Images are paired 1v1.
- Qwen sees a side-by-side image labeled Image 1 and Image 2.
- Qwen selects the more interesting image (no reasoning requested, to save tokens).
- Winner gets +1 ImageScore and advances to the next round.
- Loser gets +0 and is eliminated.
- The tournament runs until a single champion remains.

No few-shot examples are used: this dataset is inherently different from the
Galaxy Zoo Weird & Wonderful dataset, so no artifact/interesting/boring
example images are uploaded.

Final output CSV columns:
    Filename, ImageScore, interesting, classification, SourceRA, SourceDec

ImageScore:
- The number of rounds the image won (survived). An image eliminated in
  round 1 scores 0; the champion scores one point per round it won.
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

PAIR_THUMB_W = 240
PAIR_THUMB_H = 240

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
    qwen_single_elim_<run_idx>.csv; run_idx is the 1-based replicate number."""
    if TEST_MODE:
        os.makedirs(SUBSET_DIR, exist_ok=True)
        return os.path.join(SUBSET_DIR, f"qwen_single_elim_{run_idx}.csv")
    return "qwen_single_elim_seed{}_{}.csv".format(seed, time.strftime("%m-%d_%H"))


class SelectedWinner(BaseModel):
    Winner: int = Field(
        ge=1,
        le=2,
        description="The more scientifically interesting image. Must be 1 or 2."
    )


SYSTEM_INSTRUCTION = (
    "You are an expert astronomer comparing two astronomical images. "
    "The left image is Image 1 and the right image is Image 2. "
    "Your task is to select the image that is more scientifically interesting. "
    "Interesting images may show unusual morphology, asymmetry, interactions, "
    "merger-like features, arcs, rings, tails, shells, clumps, distortions, "
    "rare-looking objects, or other features worth human inspection. "
    "Skip artifacts, blank images, noisy frames, and obvious non-astronomical defects. "
    "Return only the winning image index. No explanation needed."
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

def create_pair_image(pair_records):
    """
    Stitches 2 image records side-by-side with labels.
    pair_records should be [record1, record2], each with an "image_bytes" key.
    """
    margin = 8
    label_h = 32

    canvas_w = PAIR_THUMB_W * 2 + margin * 3
    canvas_h = PAIR_THUMB_H + label_h + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            22,
        )
    except Exception:
        font = ImageFont.load_default()

    for i, record in enumerate(pair_records):
        x = margin + i * (PAIR_THUMB_W + margin)
        y = label_h + margin

        try:
            img_bytes = record["image_bytes"]

            with Image.open(io.BytesIO(img_bytes)) as img:
                img = img.convert("RGB").resize((PAIR_THUMB_W, PAIR_THUMB_H))
                canvas.paste(img, (x, y))

            label = f"Image {i + 1}"
            draw.text(
                (x + 6, 5),
                label,
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


def split_into_pairs(active_records):
    """
    Shuffle active_records before calling this.

    Returns:
    - matches: list of (match_number, [record1, record2])
    - bye_record: record carried forward automatically if odd number of records
    """
    active_records = active_records[:]
    bye_record = None

    if len(active_records) % 2 == 1:
        bye_record = active_records.pop()

    matches = []
    for i in range(0, len(active_records), 2):
        match_number = len(matches) + 1
        pair = [active_records[i], active_records[i + 1]]
        matches.append((match_number, pair))

    return matches, bye_record


# ── 4. WORKER FUNCTION ─────────────────────────────────────────────────────

def process_matches(args):
    worker_id, round_number, matches, api_key, progress_queue = args

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    checkpoint_file = f"checkpoint_elimination_worker_{worker_id}.csv"

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

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
                "Match",
                "Index",
                "Filename",
                "Score",
                "LoserIndex",
                "LoserFilename",
                "InputTokens",
                "OutputTokens",
                "TotalTokens",
            ])

        for match_number, pair_records in matches:
            pair_buf = create_pair_image(pair_records)
            pair_bytes = pair_buf.getvalue()

            input_tok, output_tok, total_tok, cost = 0, 0, 0, 0.0

            try:
                pair_b64 = base64.b64encode(pair_bytes).decode()

                response = retry_api(
                    client.beta.chat.completions.parse,
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Compare this side-by-side pair of astronomical images. "
                                    "The left image is Image 1. The right image is Image 2. "
                                    "Choose the image that is more scientifically interesting."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{pair_b64}"
                                },
                            },
                        ]},
                    ],
                    response_format=SelectedWinner,
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

                if hasattr(response, "usage") and response.usage:
                    input_tok = getattr(response.usage, "prompt_tokens", 0) or 0
                    output_tok = getattr(response.usage, "completion_tokens", 0) or 0
                    cost = getattr(response.usage, "cost", 0.0) or 0.0

                total_tok = input_tok + output_tok

                total_input_tokens += input_tok
                total_output_tokens += output_tok
                total_cost += cost

                data = response.choices[0].message.parsed

                winner_index = data.Winner - 1
                if winner_index not in (0, 1):
                    raise ValueError(f"Invalid winner index: {data.Winner}")

                winner_record = pair_records[winner_index]
                loser_record = pair_records[1 - winner_index]

                # Winner gets 1 point for surviving this round.
                writer.writerow([
                    round_number,
                    match_number,
                    winner_record["index"],
                    winner_record["filename"],
                    1,
                    loser_record["index"],
                    loser_record["filename"],
                    input_tok,
                    output_tok,
                    total_tok,
                ])

                print(
                    f"Worker {worker_id} | Round {round_number}, Match {match_number}: "
                    f"{pair_records[0]['filename']} vs {pair_records[1]['filename']} -> "
                    f"{winner_record['filename']} [tokens in={input_tok} out={output_tok}]"
                )

                f.flush()
                os.fsync(f.fileno())

            except Exception as e:
                print(
                    f"Worker {worker_id} error in Round {round_number}, "
                    f"Match {match_number}: {e}"
                )

            progress_queue.put(1)

    return worker_id, total_input_tokens, total_output_tokens, total_cost


# ── 5. SINGLE-RUN ORCHESTRATOR ─────────────────────────────────────────────

def run_experiment(all_images, api_key, random_seed, results_file):
    """
    Run the full single-elimination tournament for a single prepared record set
    and write the results to results_file.
    """
    # Clean up any stale checkpoints from a crashed prior run.
    for i in range(NUM_CORES):
        cp = f"checkpoint_elimination_worker_{i + 1}.csv"
        if os.path.exists(cp):
            os.remove(cp)

    print(f"Total Images: {len(all_images)}")

    scores = {record["index"]: 0 for record in all_images}
    records_by_index = {record["index"]: record for record in all_images}

    active_records = all_images[:]
    random.seed(random_seed)

    round_number = 1
    grand_total_input_tokens = 0
    grand_total_output_tokens = 0
    grand_total_cost = 0.0

    while len(active_records) > 1:
        random.shuffle(active_records)

        all_matches, bye_record = split_into_pairs(active_records)

        print("\n" + "=" * 70)
        print(f"Round {round_number}")
        print(f"Images entering round: {len(active_records)}")
        print(f"Matches this round:    {len(all_matches)}")

        if bye_record is not None:
            print(f"Odd image carried forward automatically: {bye_record['filename']}")

        print("=" * 70)

        manager = Manager()
        progress_queue = manager.Queue()

        chunk_size = math.ceil(len(all_matches) / NUM_CORES)
        worker_args = []

        for i in range(NUM_CORES):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(all_matches))

            if start < len(all_matches):
                worker_args.append((
                    i + 1,
                    round_number,
                    all_matches[start:end],
                    api_key,
                    progress_queue,
                ))

        with Pool(processes=NUM_CORES) as pool:
            async_result = pool.map_async(process_matches, worker_args)

            with tqdm(
                total=len(all_matches),
                desc=f"Processing Round {round_number}",
                unit="match",
            ) as pbar:
                completed = 0

                while completed < len(all_matches):
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
        round_cost = sum(r[3] for r in worker_results)
        grand_total_input_tokens += round_input_tokens
        grand_total_output_tokens += round_output_tokens
        grand_total_cost += round_cost

        # ── Round aggregation ──────────────────────────────────────────────
        print(f"\nAggregating Round {round_number} results...")

        winners = []

        for i in range(NUM_CORES):
            cp = f"checkpoint_elimination_worker_{i + 1}.csv"

            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)

                    for row in reader:
                        # row:
                        # Round, Match, Index, Filename, Score,
                        # LoserIndex, LoserFilename, InputTokens, OutputTokens, TotalTokens
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
                            base_score = int(row[4])
                        except (ValueError, IndexError):
                            continue

                        if image_index in scores:
                            scores[image_index] += base_score
                            winners.append(image_index)

                os.remove(cp)

        if bye_record is not None:
            winners.append(bye_record["index"])
            scores[bye_record["index"]] += 1

        if len(winners) != len(all_matches) + (1 if bye_record else 0):
            raise RuntimeError(f"Round {round_number} winner count mismatch. Stopping.")

        active_records = [records_by_index[idx] for idx in winners]

        print(f"Round {round_number} complete.")
        print(f"Images advancing: {len(active_records)}")
        print(
            f"Cumulative tokens: input={grand_total_input_tokens:,} "
            f"output={grand_total_output_tokens:,} "
            f"cost=${grand_total_cost:.4f}"
        )

        round_number += 1

    champion = active_records[0]
    grand_total_tokens = grand_total_input_tokens + grand_total_output_tokens

    print("\n" + "=" * 70)
    print("Tournament complete.")
    print(f"Champion: {champion['filename']}")
    print(f"Champion ImageScore: {scores[champion['index']]}")
    print(f"Total input tokens:  {grand_total_input_tokens:,}")
    print(f"Total output tokens: {grand_total_output_tokens:,}")
    print(f"Total tokens:        {grand_total_tokens:,}")
    print(f"Total cost (USD):    ${grand_total_cost:.4f}")
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

        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", grand_total_input_tokens])
        writer.writerow(["# TotalOutputTokens", grand_total_output_tokens])
        writer.writerow(["# TotalTokens", grand_total_tokens])
        writer.writerow(["# TotalCostUSD", f"{grand_total_cost:.4f}"])

    print(f"Done! Results saved to {results_file}")
    print(f"Total cost (USD): ${grand_total_cost:.4f}")


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
