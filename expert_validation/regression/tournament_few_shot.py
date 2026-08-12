"""
Iterative few-shot 2x2 retention filtering for astronomical image scoring.

Logic:
- Select a subset of images.
- Images do NOT need to be a power of 2.
- In each round, active images are randomly reshuffled.
- Images are grouped into batches of up to four.
- Gemini sees a 2x2 grid labeled Image 1 through Image 4.
- Gemini evaluates each image independently and decides whether each image
  is scientifically interesting enough to keep for human inspection.
- Gemini may keep any number of images, including none or all four.
- Kept images advance to the next round.
- Rejected images are removed from the active pool.
- Rounds continue until exactly 100 active images remain.
- During the final round, only enough rejected images are removed to leave
  exactly 100 images. Any additional rejected images remain in the pool.

Final output format:
Filename, ImageScore, Reasoning, RA, Dec, AnomalyScore, URL, volunteer_rating

ImageScore:
- 1 = image survived into the final pool of exactly 100 images
- 0 = image was filtered out
"""

import csv
import io
import math
import os
import queue
import random
import time
from multiprocessing import Manager, Pool

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from tqdm import tqdm


# ── 1. CONFIGURATION ────────────────────────────────────────────────────────

IMAGE_DIR = "/geir_data/scr/yxcheng/png_images/"
WW_DATA_PATH = "../ww_data/all_ww_data.csv"

RESULTS_FILE = "tournament_filter_{}.csv".format(
    time.strftime("%m-%d_%H")
)

NUM_CORES = 10

GRID_THUMB_W = 240
GRID_THUMB_H = 240

TARGET_REMAINING_IMAGES = 100

RANDOM_SEED = 42


# These artifact examples are still included.
ARTIFACT_EXAMPLES = [
    "69555401925880559.png",
    "70386478097659329.png",
    "69569278965207176.png",
    "40128764209818324.png",
    "70342656546335458.png",
    "70342772510452129.png",
]


# Keep the interesting examples.
INTERESTING_EXAMPLES = [
    "70347028823040378.png",
    "70365200829662492.png",
    "41214781050350168.png",
    "70342656546334671.png",
    "41192936846682523.png",
]


# Keep the boring examples.
BORING_EXAMPLES = [
    "70405045241278791.png",
    "70381951202130938.png",
    "70391567633903843.png",
    "41218771074969925.png",
    "69563914551059502.png",
]


# ── 2. STRUCTURED GEMINI RESPONSE ──────────────────────────────────────────

class InterestingSelection(BaseModel):
    KeepImage1: bool = Field(
        description="Keep Image 1 for human inspection."
    )

    KeepImage2: bool = Field(
        description="Keep Image 2 for human inspection."
    )

    KeepImage3: bool = Field(
        description="Keep Image 3 for human inspection."
    )

    KeepImage4: bool = Field(
        description="Keep Image 4 for human inspection."
    )

    Reasoning: str = Field(
        description=(
            "One-sentence technical explanation identifying which images "
            "are worth human inspection and why, or why none should be kept."
        )
    )


# Gemini.md is intentionally not loaded or included anywhere.
config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer evaluating astronomical images. "

        "The targets are arranged in a 2x2 grid. "
        "Image 1 is at the top-left. "
        "Image 2 is at the top-right. "
        "Image 3 is at the bottom-left. "
        "Image 4 is at the bottom-right. "

        "Evaluate each image independently. Do not force a winner. "

        "For each image, decide whether it is scientifically interesting "
        "enough to keep for human inspection. "

        "It is acceptable to keep any number of images, including none or "
        "all four. "

        "Interesting images may show unusual morphology, asymmetry, "
        "interactions, merger-like features, arcs, rings, tails, shells, "
        "clumps, distortions, rare-looking objects, or other features worth "
        "human inspection. "

        "Reject artifacts, blank images, noisy frames, and obvious "
        "non-astronomical defects. "

        "Return only whether to keep Images 1 through 4 and a brief "
        "technical explanation. "

        "IMPORTANT: Any provided examples of interesting or boring images "
        "are strictly a small, non-exhaustive subset. There are many other "
        "types of interesting and boring images. Do not limit evaluations "
        "only to the specific example morphologies."
    ),
    response_mime_type="application/json",
    response_schema=InterestingSelection,
    temperature=0.1,
)


# ── 3. IMAGE UTILITIES ──────────────────────────────────────────────────────

def create_grid_image(image_filenames):
    """
    Create a labeled 2x2 grid from one to four image filenames.

    Grid arrangement:

        Image 1 | Image 2
        --------+--------
        Image 3 | Image 4
    """
    if not 1 <= len(image_filenames) <= 4:
        raise ValueError(
            "create_grid_image requires between one and four filenames."
        )

    margin = 8
    label_h = 32

    canvas_w = GRID_THUMB_W * 2 + margin * 3
    canvas_h = (GRID_THUMB_H + label_h) * 2 + margin * 3

    canvas = Image.new(
        "RGB",
        (canvas_w, canvas_h),
        color=(50, 50, 50),
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            22,
        )
    except Exception:
        font = ImageFont.load_default()

    for i, filename in enumerate(image_filenames):
        image_path = os.path.join(IMAGE_DIR, filename)

        row, col = divmod(i, 2)

        x = margin + col * (GRID_THUMB_W + margin)

        label_y = (
            margin
            + row * (GRID_THUMB_H + label_h + margin)
        )

        image_y = label_y + label_h

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    (GRID_THUMB_W, GRID_THUMB_H)
                )

                canvas.paste(image, (x, image_y))

            draw.text(
                (x + 6, label_y + 3),
                f"Image {i + 1}",
                fill="white",
                font=font,
                stroke_width=1,
                stroke_fill="black",
            )

        except Exception as error:
            print(
                f"Could not load image {filename}: {error}"
            )

            draw.rectangle(
                [
                    (x, image_y),
                    (
                        x + GRID_THUMB_W,
                        image_y + GRID_THUMB_H,
                    ),
                ],
                fill=(20, 20, 20),
            )

            draw.text(
                (x + 10, image_y + 10),
                "Image load error",
                fill="white",
                font=font,
            )

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def split_into_groups(active_images, group_size=4):
    """Divide the active image pool into numbered groups of up to four."""
    return [
        (
            index // group_size + 1,
            active_images[index:index + group_size],
        )
        for index in range(0, len(active_images), group_size)
    ]


# ── 4. WORKER FUNCTION ──────────────────────────────────────────────────────

def process_matches(args):
    (
        worker_id,
        round_number,
        matches,
        api_key,
        artifact_uris,
        interesting_uris,
        boring_uris,
        progress_queue,
    ) = args

    client = genai.Client(api_key=api_key)

    checkpoint_file = (
        f"checkpoint_retention_worker_{worker_id}.csv"
    )

    # Recreate Gemini file references from the uploaded URIs.
    artifact_files = [
        types.File(
            uri=uri,
            mime_type="image/png",
        )
        for uri in artifact_uris
    ]

    interesting_files = [
        types.File(
            uri=uri,
            mime_type="image/png",
        )
        for uri in interesting_uris
    ]

    boring_files = [
        types.File(
            uri=uri,
            mime_type="image/png",
        )
        for uri in boring_uris
    ]

    def retry_api(function, *function_args, **function_kwargs):
        delays = [1, 2, 4, 8, 16]

        for attempt_index, delay in enumerate(delays):
            try:
                return function(
                    *function_args,
                    **function_kwargs,
                )

            except Exception:
                is_last_attempt = (
                    attempt_index == len(delays) - 1
                )

                if is_last_attempt:
                    raise

                time.sleep(delay)

    with open(
        checkpoint_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as checkpoint_handle:
        writer = csv.writer(checkpoint_handle)

        if os.stat(checkpoint_file).st_size == 0:
            writer.writerow(
                [
                    "Round",
                    "Match",
                    "Filename",
                    "Kept",
                    "Reasoning",
                ]
            )

        for match_number, grid_images in matches:
            grid_buffer = create_grid_image(grid_images)
            uploaded_grid = None

            try:
                uploaded_grid = retry_api(
                    client.files.upload,
                    file=grid_buffer,
                    config={
                        "mime_type": "image/png",
                    },
                )

                contents = [
                    (
                        "The following images are examples of ARTIFACTS "
                        "that must NOT be kept:"
                    ),
                    *artifact_files,

                    (
                        "The following images are examples of INTERESTING "
                        "targets worth keeping:"
                    ),
                    *interesting_files,

                    (
                        "The following images are examples of BORING "
                        "targets that should be filtered out:"
                    ),
                    *boring_files,

                    (
                        "The interesting and boring examples above are "
                        "strictly a small, non-exhaustive subset. Do not "
                        "limit the decision solely to those morphologies."
                    ),

                    (
                        "Now evaluate this 2x2 grid of astronomical images. "
                        "Image 1 is top-left, Image 2 is top-right, "
                        "Image 3 is bottom-left, and Image 4 is "
                        "bottom-right. Evaluate each image independently. "
                        "Do not force a winner. For each image, decide "
                        "whether it is scientifically interesting enough "
                        "to keep for human inspection. It is acceptable "
                        "to keep any number of images, including none or "
                        "all four."
                    ),

                    uploaded_grid,
                ]

                response = retry_api(
                    client.models.generate_content,
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                    config=config,
                )

                data = response.parsed

                if data is None:
                    raise ValueError(
                        "Gemini returned no parsed response."
                    )

                keep_flags = [
                    bool(data.KeepImage1),
                    bool(data.KeepImage2),
                    bool(data.KeepImage3),
                    bool(data.KeepImage4),
                ]

                reasoning = data.Reasoning.strip()

                for image_index, filename in enumerate(
                    grid_images
                ):
                    kept = int(keep_flags[image_index])

                    writer.writerow(
                        [
                            round_number,
                            match_number,
                            filename,
                            kept,
                            reasoning if kept else "",
                        ]
                    )

                kept_names = [
                    grid_images[index]
                    for index in range(len(grid_images))
                    if keep_flags[index]
                ]

                if kept_names:
                    kept_string = ", ".join(kept_names)
                else:
                    kept_string = "NONE"

                print(
                    f"Worker {worker_id} | "
                    f"Round {round_number}, "
                    f"Match {match_number}: "
                    f"{', '.join(grid_images)} "
                    f"-> kept: {kept_string}"
                )

                checkpoint_handle.flush()
                os.fsync(checkpoint_handle.fileno())

            except Exception as error:
                print(
                    f"Worker {worker_id} error in "
                    f"Round {round_number}, "
                    f"Match {match_number}: {error}"
                )

                # Conservative fallback:
                # keep all four images when an API or processing failure
                # occurs, avoiding false rejection caused by technical errors.
                fallback_reason = (
                    "Kept due to API or processing failure; "
                    "requires human inspection."
                )

                for filename in grid_images:
                    writer.writerow(
                        [
                            round_number,
                            match_number,
                            filename,
                            1,
                            fallback_reason,
                        ]
                    )

                checkpoint_handle.flush()
                os.fsync(checkpoint_handle.fileno())

            finally:
                if uploaded_grid is not None:
                    try:
                        retry_api(
                            client.files.delete,
                            name=uploaded_grid.name,
                        )
                    except Exception as delete_error:
                        print(
                            "Could not delete temporary uploaded "
                            f"grid {uploaded_grid.name}: "
                            f"{delete_error}"
                        )

                progress_queue.put(1)

    return worker_id


# ── 5. EXAMPLE UPLOAD UTILITY ───────────────────────────────────────────────

def upload_example_set(
    client,
    file_list,
    category_name,
):
    """
    Upload one category of few-shot example images and return its file URIs.
    """
    print(f"Uploading {category_name} examples...")

    uris = []

    for filename in file_list:
        image_path = os.path.join(
            IMAGE_DIR,
            filename,
        )

        if not os.path.exists(image_path):
            print(
                f"  Warning: {category_name} example "
                f"not found, skipping: {filename}"
            )
            continue

        uploaded = client.files.upload(
            file=image_path
        )

        uris.append(uploaded.uri)

        print(
            f"  Uploaded {category_name}: {filename}"
        )

    return uris


# ── 6. MAIN ORCHESTRATOR ────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("JB_API_KEY")

    if not api_key:
        raise ValueError(
            "JB_API_KEY environment variable not found."
        )

    client_main = genai.Client(api_key=api_key)

    # Gemini.md is not opened, read, or added to the system prompt.

    artifact_uris = upload_example_set(
        client=client_main,
        file_list=ARTIFACT_EXAMPLES,
        category_name="artifact",
    )

    interesting_uris = upload_example_set(
        client=client_main,
        file_list=INTERESTING_EXAMPLES,
        category_name="interesting",
    )

    boring_uris = upload_example_set(
        client=client_main,
        file_list=BORING_EXAMPLES,
        category_name="boring",
    )

    print(
        "\nDone. Ready with "
        f"{len(artifact_uris)} artifact examples, "
        f"{len(interesting_uris)} interesting examples, and "
        f"{len(boring_uris)} boring examples.\n"
    )

    all_images = [
        filename
        for filename in os.listdir(IMAGE_DIR)
        if filename.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
            )
        )
    ]

    # Optional testing subset.
    # Change or remove this line when running the full dataset.
    all_images = all_images[4000:5000]

    if len(all_images) < TARGET_REMAINING_IMAGES:
        raise ValueError(
            f"Need at least {TARGET_REMAINING_IMAGES} images. "
            f"Current number: {len(all_images)}"
        )

    print(f"Total Images: {len(all_images)}")

    print(
        f"Rounds will continue until exactly "
        f"{TARGET_REMAINING_IMAGES} images remain."
    )

    # Final binary scores.
    scores = {
        image: 0
        for image in all_images
    }

    # The reasoning for a survivor is overwritten whenever it is
    # evaluated and retained in a later round.
    reasons = {
        image: ""
        for image in all_images
    }

    active_images = all_images[:]

    random.seed(RANDOM_SEED)

    round_number = 1

    # Continue until the active pool contains exactly 100 images.
    while len(active_images) > TARGET_REMAINING_IMAGES:
        random.shuffle(active_images)

        round_start_count = len(active_images)
        all_matches = split_into_groups(active_images)

        print("\n" + "=" * 70)
        print(f"Round {round_number}")
        print(f"Images entering round: {round_start_count}")
        print(f"2x2 grids this round:  {len(all_matches)}")
        print("=" * 70)

        manager = Manager()
        progress_queue = manager.Queue()

        actual_worker_count = min(NUM_CORES, len(all_matches))
        chunk_size = math.ceil(len(all_matches) / actual_worker_count)
        worker_args = []

        for worker_index in range(actual_worker_count):
            start = worker_index * chunk_size
            end = min((worker_index + 1) * chunk_size, len(all_matches))

            if start >= len(all_matches):
                continue

            worker_args.append(
                (
                    worker_index + 1,
                    round_number,
                    all_matches[start:end],
                    api_key,
                    artifact_uris,
                    interesting_uris,
                    boring_uris,
                    progress_queue,
                )
            )

        with Pool(processes=actual_worker_count) as pool:
            async_result = pool.map_async(process_matches, worker_args)

            with tqdm(
                total=len(all_matches),
                desc=f"Processing Round {round_number}",
                unit="grid",
            ) as progress_bar:
                completed = 0

                while completed < len(all_matches):
                    try:
                        progress_queue.get(timeout=1.0)
                        progress_bar.update(1)
                        completed += 1
                    except queue.Empty:
                        if async_result.ready():
                            break

            async_result.get()

        manager.shutdown()

        print(f"\nAggregating Round {round_number} results...")

        round_decisions = {}
        seen_this_round = set()

        for worker_index in range(actual_worker_count):
            checkpoint_path = (
                "checkpoint_retention_worker_"
                f"{worker_index + 1}.csv"
            )

            if not os.path.exists(checkpoint_path):
                continue

            with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_handle:
                reader = csv.reader(checkpoint_handle)
                next(reader, None)

                for row in reader:
                    if not row:
                        continue

                    try:
                        row_round = int(row[0])
                    except (ValueError, IndexError):
                        continue

                    if row_round != round_number:
                        continue

                    try:
                        filename = row[2]
                        kept = int(row[3])
                        reasoning = row[4] if len(row) > 4 else ""
                    except (ValueError, IndexError):
                        continue

                    seen_this_round.add(filename)
                    round_decisions[filename] = (kept, reasoning)

                    if kept == 1 and filename in reasons and reasoning:
                        reasons[filename] = reasoning

            os.remove(checkpoint_path)

        expected_images = set(active_images)
        actual_images = set(seen_this_round)

        if expected_images != actual_images:
            missing = expected_images - actual_images
            extra = actual_images - expected_images
            raise RuntimeError(
                f"Round {round_number} aggregation mismatch.\n"
                f"Missing images: {list(missing)[:10]}\n"
                f"Extra images: {list(extra)[:10]}\n"
                "Stopping to avoid corrupting results."
            )

        rejected_by_model = [
            filename
            for filename in active_images
            if round_decisions[filename][0] == 0
        ]

        removals_needed = round_start_count - TARGET_REMAINING_IMAGES
        removals_to_apply = min(removals_needed, len(rejected_by_model))
        removed_images = set(rejected_by_model[:removals_to_apply])

        next_pool = [
            filename
            for filename in active_images
            if filename not in removed_images
        ]

        removed_this_round = len(removed_images)

        print(f"Round {round_number} complete.")
        print(f"Images entering:     {round_start_count}")
        print(f"Model rejections:    {len(rejected_by_model)}")
        print(f"Rejections applied:  {removed_this_round}")
        print(f"Images advancing:    {len(next_pool)}")

        active_images = next_pool

        if len(active_images) == TARGET_REMAINING_IMAGES:
            print(
                f"Stopping because exactly "
                f"{TARGET_REMAINING_IMAGES} images remain."
            )
            break

        if removed_this_round == 0:
            raise RuntimeError(
                f"Round {round_number} removed no images. "
                f"There are still {len(active_images)} images remaining, "
                f"so the exact target of {TARGET_REMAINING_IMAGES} "
                "cannot be reached."
            )

        if len(active_images) < TARGET_REMAINING_IMAGES:
            raise RuntimeError(
                f"Round {round_number} produced {len(active_images)} images. "
                f"The pool must never fall below "
                f"{TARGET_REMAINING_IMAGES}."
            )

        round_number += 1

    if len(active_images) != TARGET_REMAINING_IMAGES:
        raise RuntimeError(
            f"Filtering ended with {len(active_images)} images "
            f"instead of exactly {TARGET_REMAINING_IMAGES}."
        )

    print("\n" + "=" * 70)
    print("2x2-grid retention filtering complete.")
    print(
        f"Final interesting pool size: "
        f"{len(active_images)}"
    )
    print("=" * 70)

    final_interesting_pool = set(active_images)

    for image in scores:
        scores[image] = int(
            image in final_interesting_pool
        )

    # ── 7. METADATA ENRICHMENT ────────────────────────────────────────────

    print("\nAggregating final results...")

    ww_lookup = {}

    with open(
        WW_DATA_PATH,
        newline="",
        encoding="utf-8",
    ) as ww_handle:
        for row in csv.DictReader(ww_handle):
            filename = row.get(
                "filename",
                "",
            ).strip()

            if not filename:
                continue

            normalized_weighted_count = row.get(
                "normalized_weighted_count",
                "",
            ).strip()

            volunteer_rating = ""

            if normalized_weighted_count:
                try:
                    volunteer_rating = (
                        float(normalized_weighted_count)
                        * 0.01
                    )
                except ValueError:
                    volunteer_rating = ""

            filename_without_extension = (
                os.path.splitext(filename)[0]
            )

            ww_lookup[filename_without_extension] = {
                "RA": row.get("RA", ""),
                "Dec": row.get("Dec", ""),
                "anomaly_score": row.get(
                    "anomaly_score",
                    "",
                ),
                "URL": row.get("URL", ""),
                "volunteer_rating": volunteer_rating,
            }

    # ── 8. WRITE FINAL RESULTS ─────────────────────────────────────────────

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as results_handle:
        writer = csv.writer(results_handle)

        writer.writerow(
            [
                "Filename",
                "ImageScore",
                "Reasoning",
                "RA",
                "Dec",
                "AnomalyScore",
                "URL",
                "volunteer_rating",
            ]
        )

        # Write interesting images first.
        sorted_images = sorted(
            scores,
            key=lambda image: scores[image],
            reverse=True,
        )

        for image in sorted_images:
            score = scores[image]

            reasoning = (
                reasons.get(image, "")
                if score == 1
                else ""
            )

            base_name = os.path.splitext(image)[0]

            metadata = ww_lookup.get(
                base_name,
                {},
            )

            writer.writerow(
                [
                    image,
                    score,
                    reasoning,
                    metadata.get("RA", ""),
                    metadata.get("Dec", ""),
                    metadata.get(
                        "anomaly_score",
                        "",
                    ),
                    metadata.get("URL", ""),
                    metadata.get(
                        "volunteer_rating",
                        "",
                    ),
                ]
            )

    print(
        f"Done! Results saved to {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()