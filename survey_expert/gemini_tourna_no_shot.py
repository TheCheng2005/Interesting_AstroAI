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
- Exactly 10 rounds are run (or fewer if all images are eliminated).

Final output format is preserved:
Filename, ImageScore, Reasoning, RA, Dec, AnomalyScore, URL, volunteer_rating

ImageScore:
- 0–10: the number of rounds the image survived (was kept).
  An image rejected in round 1 scores 0; one that survives all 10 rounds scores 10.
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
EXPERT_CSV_DIR = "expert_csvs"
# The same dataset is run 3 times; each run writes its own CSV.
RESULT_TEMPLATE = "ai_csvs/gemini_tournament_no_shot_990_run{}.csv"
NUM_CORES = 20

PAIR_THUMB_W = 240
PAIR_THUMB_H = 240

MAX_ROUNDS = 10
RANDOM_SEED = 42

ARTIFACT_EXAMPLES = [
    "69555401925880559.png",
    "70386478097659329.png",
    "69569278965207176.png",
    "40128764209818324.png",
    "70342656546335458.png",
    "70342772510452129.png",
]

# NOTE: These are the 10 images that were used as few-shot examples in the
# few-shot version. They are still excluded from the testing dataset here
# so that the no-shot and few-shot results are comparable.
FEW_SHOT_EXCLUDED = [
    "70347028823040378.png",
    "70365200829662492.png",
    "41214781050350168.png",
    "70342656546334671.png",
    "41192936846682523.png",
    "70405045241278791.png",
    "70381951202130938.png",
    "70391567633903843.png",
    "41218771074969925.png",
    "69563914551059502.png",
]

INTERESTING_EXAMPLES = []

BORING_EXAMPLES = []


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
        "Return only whether to keep Image 1 and whether to keep Image 2. "
        "IMPORTANT: Any provided examples of 'interesting' or 'boring' images are strictly "
        "a small, non-exhaustive subset. There are many other types of interesting and "
        "boring images. Do not limit your evaluations only to those specific morphologies."
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

    # Re-hydrate file references from URIs for use in contents.
    artifact_files = [types.File(uri=uri, mime_type="image/png") for uri in artifact_uris]

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
                "Match",
                "Filename",
                "Kept",
                "Reasoning",
                "InputTokens",
                "OutputTokens",
                "TotalTokens"
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

                usage = response.usage_metadata
                input_tok = getattr(usage, "prompt_token_count", 0) or 0
                output_tok = getattr(usage, "candidates_token_count", 0) or 0
                total_tok = getattr(usage, "total_token_count", 0) or (input_tok + output_tok)

                total_input_tokens += input_tok
                total_output_tokens += output_tok

                keep_flags = [
                    bool(data.KeepImage1),
                    bool(data.KeepImage2)
                ]

                reasoning = ""

                for idx, fname in enumerate(pair_images):
                    kept = 1 if keep_flags[idx] else 0

                    writer.writerow([
                        round_number,
                        match_number,
                        fname,
                        kept,
                        reasoning if kept else "",
                        input_tok,
                        output_tok,
                        total_tok
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
                    f"{pair_images[0]} vs {pair_images[1]} -> kept: {kept_str} "
                    f"[tokens in={input_tok} out={output_tok}]"
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
                fallback_reason = ""

                for fname in pair_images:
                    writer.writerow([
                        round_number,
                        match_number,
                        fname,
                        1,
                        fallback_reason,
                        0,
                        0,
                        0
                    ])

                f.flush()
                os.fsync(f.fileno())

            progress_queue.put(1)

    return worker_id, total_input_tokens, total_output_tokens


# ── 4. DATASET FROM EXPERT CSVS ─────────────────────────────────────────────
def load_expert_image_names(expert_dir):
    """Return the sorted list of image filenames the experts scored, taken from
    the `name` column across every CSV in expert_dir (all files list the same
    set). This is the test set the AI is evaluated on."""
    names = set()
    for fn in sorted(os.listdir(expert_dir)):
        if not fn.lower().endswith(".csv"):
            continue
        with open(os.path.join(expert_dir, fn), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip().strip('"')
                if name:
                    names.add(name)
    return sorted(names)


# ── 5. SINGLE-RUN ORCHESTRATOR ──────────────────────────────────────────────
def run_experiment(run_idx, api_key, all_images, artifact_uris):
    results_file = RESULT_TEMPLATE.format(run_idx)
    print("\n" + "#" * 70)
    print(f"# RUN {run_idx} of 3  ->  {results_file}")
    print("#" * 70)

    # Remove any stale checkpoints from a previous run.
    for _i in range(NUM_CORES):
        _cp = f"checkpoint_retention_worker_{_i + 1}.csv"
        if os.path.exists(_cp):
            os.remove(_cp)

    print(f"Total Images: {len(all_images)}")
    print(f"Running exactly {MAX_ROUNDS} rounds of filtering.")

    # Scores accumulate: +1 each round an image is kept. Range: 0–MAX_ROUNDS.
    scores = {img: 0 for img in all_images}

    # Reasoning is overwritten whenever the image is kept in a later round.
    # Final output uses the most recent keep reasoning.
    reasons = {img: "" for img in all_images}

    active_images = all_images[:]
    random.seed(RANDOM_SEED)

    round_number = 1
    grand_total_input_tokens = 0
    grand_total_output_tokens = 0

    while round_number <= MAX_ROUNDS:
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
            cp = f"checkpoint_retention_worker_{i + 1}.csv"

            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)

                    for row in reader:
                        # row:
                        # Round, Match, Filename, Kept, Reasoning,
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
                            fname = row[2]
                            kept = int(row[3])
                            reasoning = row[4] if len(row) > 4 else ""
                        except IndexError:
                            continue

                        seen_this_round.add(fname)
                        round_decisions[fname] = (kept, reasoning)

                        if kept == 1 and fname in reasons and reasoning:
                            # Overwrite previous reasoning with the latest keep reasoning.
                            reasons[fname] = reasoning

                os.remove(cp)

        # Add odd image carried forward by logistics.
        # Reasoning is not changed because Gemini did not evaluate it this round.
        if bye_image is not None:
            seen_this_round.add(bye_image)
            round_decisions[bye_image] = (1, reasons.get(bye_image, ""))

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

        rejected_by_model = [
            fname
            for fname in active_images
            if round_decisions[fname][0] == 0
        ]

        removed_images = set(rejected_by_model)

        next_pool = [
            fname
            for fname in active_images
            if fname not in removed_images
        ]

        # Increment score for every image that survived this round.
        for fname in next_pool:
            scores[fname] += 1

        removed_this_round = len(removed_images)

        print(f"Round {round_number} complete.")
        print(f"Images entering:     {round_start_count}")
        print(f"Model rejections:    {removed_this_round}")
        print(f"Images advancing:    {len(next_pool)}")
        print(
            f"Cumulative tokens:   input={grand_total_input_tokens:,}  "
            f"output={grand_total_output_tokens:,}  "
            f"total={grand_total_input_tokens + grand_total_output_tokens:,}"
        )

        active_images = next_pool
        round_number += 1

        if not active_images:
            print("All images eliminated. Stopping early.")
            break

    grand_total_tokens = grand_total_input_tokens + grand_total_output_tokens

    print("\n" + "=" * 70)
    print("Pairwise retention filtering complete.")
    print(f"Rounds completed:            {round_number - 1}")
    print(f"Images surviving all rounds: {len(active_images)}")
    print(f"Total input tokens:          {grand_total_input_tokens:,}")
    print(f"Total output tokens:         {grand_total_output_tokens:,}")
    print(f"Total tokens:                {grand_total_tokens:,}")
    print("=" * 70)

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

    with open(results_file, "w", newline="", encoding="utf-8") as f:
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

        # Highest-scoring images first.
        for img in sorted(scores, key=scores.get, reverse=True):
            score = scores[img]

            if score > 0:
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


# ── 6. MAIN: RUN THE SAME DATASET THREE TIMES ───────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("JB_API_KEY")
    if not api_key:
        raise ValueError("JB_API_KEY not found.")

    client_main = genai.Client(api_key=api_key)

    def upload_example_set(file_list, category_name):
        print(f"Uploading {category_name} examples...")
        uris = []
        for fname in file_list:
            path = os.path.join(IMAGE_DIR, fname)
            if not os.path.exists(path):
                print(f"  Warning: {category_name} example not found, skipping: {fname}")
                continue
            uploaded = client_main.files.upload(file=path)
            uris.append(uploaded.uri)
            print(f"  Uploaded {category_name}: {fname}")
        return uris

    # Upload artifact examples once; reused across all 3 runs.
    artifact_uris = upload_example_set(ARTIFACT_EXAMPLES, "artifact")
    print(f"Done. Ready with {len(artifact_uris)} artifact examples (no-shot mode: no interesting/boring examples).\n")

    # Test set = every image the experts scored (expert_csvs/).
    all_images = load_expert_image_names(EXPERT_CSV_DIR)
    # Exclude the 10 few-shot example images for parity with the few-shot runs.
    all_images = [img for img in all_images if img not in set(FEW_SHOT_EXCLUDED)]
    print(f"Loaded {len(all_images)} images from expert CSVs.\n")

    # Same 990-image dataset, run three times.
    for run_idx in range(1, 4):
        run_experiment(run_idx, api_key, all_images, artifact_uris)