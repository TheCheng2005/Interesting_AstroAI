"""
Iterative pairwise retention filtering for astronomical image scoring.

Logic:
- Select a subset of images.
- Images do NOT need to be a power of 2.
- In each round, active images are randomly reshuffled.
- Images are paired 1v1.
- Gemini sees a side-by-side image labeled Image 1 and Image 2.
- Gemini evaluates each image independently and decides whether each image
  is scientifically interesting enough to keep for human inspection.
- Gemini may keep neither image, one image, or both images.
- Kept images advance to the next round.
- Rejected images are removed from the active pool.
- Once a round removes fewer than STOP_REMOVED_THRESHOLD images, the filtering
  has become weak, so we stop and call the remaining pool interesting.

Final output format is preserved:
Filename, ImageScore, Reasoning, RA, Dec, AnomalyScore, URL, volunteer_rating

ImageScore:
- 1 = image survived into the final interesting pool
- 0 = image was filtered out
"""

import os
import csv
import time
import random
import math
import queue
import io
from multiprocessing import Pool, Manager
from tqdm import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont


# ── 1. CONFIGURATION ───────────────────────────────────────────────────────
IMAGE_DIR = "/geir_data/scr/yxcheng/png_images/"
WW_DATA_PATH = "../ww_data/all_ww_data.csv"
RESULTS_FILE = "tournament_filter_{}.csv".format(time.strftime("%m-%d_%H"))
NUM_CORES = 10

PAIR_THUMB_W = 240
PAIR_THUMB_H = 240

# Stop once the round removes fewer than this many images.
STOP_REMOVED_THRESHOLD = 20

# Avoid stopping after only one permissive round.
MIN_ROUNDS = 0

# Safety cap to avoid unexpected infinite loops.
MAX_ROUNDS = 2

RANDOM_SEED = 42

ARTIFACT_EXAMPLES = [
    "69555401925880559.png",
    "70386478097659329.png",
    "69569278965207176.png",
    "40128764209818324.png",
    "70342656546335458.png",
    "70342772510452129.png",
]


class InterestingSelection(BaseModel):
    KeepImage1: bool = Field(
        description=(
            "True if Image 1 is scientifically interesting enough to keep "
            "for human inspection. False otherwise."
        )
    )
    KeepImage2: bool = Field(
        description=(
            "True if Image 2 is scientifically interesting enough to keep "
            "for human inspection. False otherwise."
        )
    )
    Reasoning: str = Field(
        description=(
            "One-sentence technical explanation. If one or both images are kept, "
            "explain what is interesting about the kept image(s). If neither is kept, "
            "briefly explain why neither is worth human inspection."
        )
    )


few_shot_context = ""
if os.path.exists("GEMINI.md"):
    with open("GEMINI.md", "r", encoding="utf-8") as f:
        few_shot_context = f"\n\n{f.read()}"


config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer evaluating astronomical images. "
        "The left image is Image 1 and the right image is Image 2. "
        "Evaluate each image independently. Do not force a winner. "
        "For each image, decide whether it is scientifically interesting enough "
        "to keep for human inspection. "
        "It is acceptable to keep neither image, one image, or both images. "
        "Interesting images may show unusual morphology, asymmetry, interactions, "
        "merger-like features, arcs, rings, tails, shells, clumps, distortions, "
        "rare-looking objects, or other features worth human inspection. "
        "Reject artifacts, blank images, noisy frames, and obvious non-astronomical defects. "
        "Return only whether to keep Image 1, whether to keep Image 2, and a brief reasoning. "
        f"{few_shot_context}"
    ),
    response_mime_type="application/json",
    response_schema=InterestingSelection,
    temperature=0.1
)


# ── 2. IMAGE UTILITIES ─────────────────────────────────────────────────────
def create_pair_image(image_filenames):
    """
    Stitches 2 images side-by-side with labels.
    image_filenames should be [fname1, fname2].
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
            22
        )
    except Exception:
        font = ImageFont.load_default()

    for i, fname in enumerate(image_filenames):
        img_path = os.path.join(IMAGE_DIR, fname)

        x = margin + i * (PAIR_THUMB_W + margin)
        y = label_h + margin

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize((PAIR_THUMB_W, PAIR_THUMB_H))
                canvas.paste(img, (x, y))

                label = f"Image {i + 1}"
                draw.text(
                    (x + 6, 5),
                    label,
                    fill="white",
                    font=font,
                    stroke_width=1,
                    stroke_fill="black"
                )
        except Exception as e:
            print(f"Could not load image {fname}: {e}")

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf


def split_into_pairs(active_images):
    """
    Shuffle active_images before calling this.

    Returns:
    - matches: list of (match_number, [img1, img2])
    - bye_image: image carried forward automatically if odd number of images
    """
    active_images = active_images[:]
    bye_image = None

    if len(active_images) % 2 == 1:
        bye_image = active_images.pop()

    matches = []
    for i in range(0, len(active_images), 2):
        match_number = len(matches) + 1
        pair = [active_images[i], active_images[i + 1]]
        matches.append((match_number, pair))

    return matches, bye_image


# ── 3. WORKER FUNCTION ─────────────────────────────────────────────────────
def process_matches(args):
    worker_id, round_number, matches, api_key, artifact_uris, progress_queue = args
    client = genai.Client(api_key=api_key)
    checkpoint_file = f"checkpoint_retention_worker_{worker_id}.csv"

    # Re-hydrate artifact file references from URIs for use in contents.
    artifact_files = [
        types.File(uri=uri, mime_type="image/png")
        for uri in artifact_uris
    ]

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
                "Filename",
                "Kept",
                "Reasoning"
            ])

        for match_number, pair_images in matches:
            pair_buf = create_pair_image(pair_images)

            try:
                uploaded_pair = retry_api(
                    client.files.upload,
                    file=pair_buf,
                    config={"mime_type": "image/png"}
                )

                contents = [
                    "The following images are examples of ARTIFACTS that must NOT be kept:",
                    *artifact_files,
                    (
                        "Now evaluate this side-by-side pair of astronomical images. "
                        "The left image is Image 1. The right image is Image 2. "
                        "Evaluate each image independently. Do not force a winner. "
                        "For each image, decide whether it is scientifically interesting "
                        "enough to keep for human inspection. "
                        "It is acceptable to keep neither image, one image, or both images."
                    ),
                    uploaded_pair
                ]

                response = retry_api(
                    client.models.generate_content,
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                    config=config
                )

                data = response.parsed

                keep_flags = [
                    bool(data.KeepImage1),
                    bool(data.KeepImage2)
                ]

                reasoning = data.Reasoning.strip()

                for idx, fname in enumerate(pair_images):
                    kept = 1 if keep_flags[idx] else 0

                    writer.writerow([
                        round_number,
                        match_number,
                        fname,
                        kept,
                        reasoning if kept else ""
                    ])

                kept_names = [
                    pair_images[i]
                    for i in range(2)
                    if keep_flags[i]
                ]

                if len(kept_names) == 0:
                    kept_str = "NONE"
                else:
                    kept_str = ", ".join(kept_names)

                print(
                    f"Worker {worker_id} | Round {round_number}, Match {match_number}: "
                    f"{pair_images[0]} vs {pair_images[1]} -> kept: {kept_str}"
                )

                f.flush()
                os.fsync(f.fileno())

                retry_api(client.files.delete, name=uploaded_pair.name)

            except Exception as e:
                print(
                    f"Worker {worker_id} error in Round {round_number}, "
                    f"Match {match_number}: {e}"
                )

                # Conservative fallback:
                # If Gemini/API fails, keep both images so we do not accidentally
                # discard potentially interesting images due to technical failure.
                fallback_reason = "Kept due to API or processing failure; requires human inspection."

                for fname in pair_images:
                    writer.writerow([
                        round_number,
                        match_number,
                        fname,
                        1,
                        fallback_reason
                    ])

                f.flush()
                os.fsync(f.fileno())

            progress_queue.put(1)

    return worker_id


# ── 4. MAIN ORCHESTRATOR ───────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("JB_API_KEY")
    if not api_key:
        raise ValueError("JB_API_KEY not found.")

    # Upload artifact examples once at startup.
    # Workers receive URIs so each worker can reference them without re-uploading.
    print("Uploading artifact examples...")
    client_main = genai.Client(api_key=api_key)
    artifact_uris = []

    for fname in ARTIFACT_EXAMPLES:
        path = os.path.join(IMAGE_DIR, fname)

        if not os.path.exists(path):
            print(f"  Warning: artifact example not found, skipping: {fname}")
            continue

        uploaded = client_main.files.upload(file=path)
        artifact_uris.append(uploaded.uri)
        print(f"  Uploaded artifact: {fname}")

    print(f"Done. {len(artifact_uris)} artifact examples ready.\n")

    # Load images.
    # Do not hardcode the number of images here.
    # You can subset for testing using normal Python slicing below.
    all_images = [
        img for img in os.listdir(IMAGE_DIR)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Optional testing subset.
    # Change or comment this out as needed.
    all_images = all_images[3000:4000]

    if len(all_images) < 2:
        raise ValueError(f"Need at least 2 images. Current number: {len(all_images)}")

    print(f"Total Images: {len(all_images)}")
    print(f"Stopping after at least {MIN_ROUNDS} rounds once fewer than "
          f"{STOP_REMOVED_THRESHOLD} images are removed in a round.")
    print(f"Maximum rounds: {MAX_ROUNDS}")

    # Final binary scores.
    # These stay 0 until the final interesting pool is determined.
    scores = {img: 0 for img in all_images}

    # Reasoning is overwritten whenever the image is kept in a later round.
    # Final output uses the most recent keep reasoning.
    reasons = {img: "" for img in all_images}

    active_images = all_images[:]
    random.seed(RANDOM_SEED)

    round_number = 1

    while len(active_images) > 1 and round_number <= MAX_ROUNDS:
        random.shuffle(active_images)

        round_start_count = len(active_images)
        all_matches, bye_image = split_into_pairs(active_images)

        print("\n" + "=" * 70)
        print(f"Round {round_number}")
        print(f"Images entering round: {round_start_count}")
        print(f"Matches this round:    {len(all_matches)}")

        if bye_image is not None:
            print(f"Odd image carried forward automatically: {bye_image}")

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
                    artifact_uris,
                    progress_queue
                ))

        with Pool(processes=NUM_CORES) as pool:
            async_result = pool.map_async(process_matches, worker_args)

            with tqdm(
                total=len(all_matches),
                desc=f"Processing Round {round_number}",
                unit="match"
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

            async_result.get()

        # ── Round aggregation ──────────────────────────────────────────────
        print(f"\nAggregating Round {round_number} results...")

        next_pool = []
        seen_this_round = set()

        for i in range(NUM_CORES):
            cp = f"checkpoint_retention_worker_{i + 1}.csv"

            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)

                    for row in reader:
                        # row:
                        # Round, Match, Filename, Kept, Reasoning
                        if not row:
                            continue

                        try:
                            row_round = int(row[0])
                        except (ValueError, IndexError):
                            continue

                        if row_round != round_number:
                            continue

                        try:
                            fname = row[2]
                            kept = int(row[3])
                            reasoning = row[4] if len(row) > 4 else ""
                        except IndexError:
                            continue

                        seen_this_round.add(fname)

                        if kept == 1:
                            next_pool.append(fname)

                            if fname in reasons and reasoning:
                                # Overwrite previous reasoning with the latest keep reasoning.
                                reasons[fname] = reasoning

                os.remove(cp)

        # Add odd image carried forward by logistics.
        # Reasoning is not changed because Gemini did not evaluate it this round.
        if bye_image is not None:
            next_pool.append(bye_image)
            seen_this_round.add(bye_image)

        expected_seen = set(active_images)
        actual_seen = set(seen_this_round)

        if expected_seen != actual_seen:
            missing = expected_seen - actual_seen
            extra = actual_seen - expected_seen

            raise RuntimeError(
                f"Round {round_number} aggregation mismatch.\n"
                f"Missing images: {list(missing)[:10]}\n"
                f"Extra images: {list(extra)[:10]}\n"
                f"Stopping to avoid corrupting results."
            )

        # Deduplicate while preserving order, just in case.
        deduped_next_pool = []
        already_added = set()

        for img in next_pool:
            if img not in already_added:
                deduped_next_pool.append(img)
                already_added.add(img)

        next_pool = deduped_next_pool

        removed_this_round = round_start_count - len(next_pool)

        print(f"Round {round_number} complete.")
        print(f"Images entering:  {round_start_count}")
        print(f"Images advancing: {len(next_pool)}")
        print(f"Images removed:   {removed_this_round}")

        active_images = next_pool

        # If Gemini rejects everything, stop immediately.
        if len(active_images) == 0:
            print("No images survived this round. Final interesting pool is empty.")
            break

        # Stop once filtering becomes weak.
        if (
            round_number >= MIN_ROUNDS
            and removed_this_round < STOP_REMOVED_THRESHOLD
        ):
            print(
                f"Stopping because only {removed_this_round} images were removed, "
                f"which is fewer than STOP_REMOVED_THRESHOLD={STOP_REMOVED_THRESHOLD}."
            )
            break

        round_number += 1

    print("\n" + "=" * 70)
    print("Pairwise retention filtering complete.")
    print(f"Final interesting pool size: {len(active_images)}")
    print("=" * 70)

    # Assign final binary scores.
    final_interesting_pool = set(active_images)

    for img in scores:
        if img in final_interesting_pool:
            scores[img] = 1
        else:
            scores[img] = 0

    # ── 5. FINAL AGGREGATE / METADATA ENRICHMENT ──────────────────────────
    print("\nAggregating final results...")

    ww_lookup = {}

    with open(WW_DATA_PATH, newline="", encoding="utf-8") as ww_f:
        for row in csv.DictReader(ww_f):
            fname = row.get("filename", "").strip()

            if fname:
                nwc_str = row.get("normalized_weighted_count", "").strip()
                nwc_val = ""

                if nwc_str:
                    try:
                        nwc_val = float(nwc_str) * 0.01
                    except ValueError:
                        nwc_val = ""

                # Key by filename without extension, matching your later lookup format.
                ww_lookup[os.path.splitext(fname)[0]] = {
                    "RA": row.get("RA", ""),
                    "Dec": row.get("Dec", ""),
                    "anomaly_score": row.get("anomaly_score", ""),
                    "URL": row.get("URL", ""),
                    "volunteer_rating": nwc_val
                }

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Filename",
            "ImageScore",
            "Reasoning",
            "RA",
            "Dec",
            "AnomalyScore",
            "URL",
            "volunteer_rating"
        ])

        # Interesting images first, then non-interesting images.
        for img in sorted(scores, key=scores.get, reverse=True):
            score = scores[img]

            if score == 1:
                reasoning = reasons.get(img, "")
            else:
                reasoning = ""

            base_name = os.path.splitext(img)[0]
            meta = ww_lookup.get(base_name, {})

            writer.writerow([
                img,
                score,
                reasoning,
                meta.get("RA", ""),
                meta.get("Dec", ""),
                meta.get("anomaly_score", ""),
                meta.get("URL", ""),
                meta.get("volunteer_rating", "")
            ])

    print(f"Done! Results saved to {RESULTS_FILE}")