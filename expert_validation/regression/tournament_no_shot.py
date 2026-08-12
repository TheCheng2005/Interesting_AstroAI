"""
Iterative 2x2 retention filtering for astronomical image scoring.

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


# ── 1. CONFIGURATION ───────────────────────────────────────────────────────

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

ARTIFACT_EXAMPLES = [
    "69555401925880559.png",
    "70386478097659329.png",
    "69569278965207176.png",
    "40128764209818324.png",
    "70342656546335458.png",
    "70342772510452129.png",
]


# ── 2. GEMINI RESPONSE MODEL ───────────────────────────────────────────────

class InterestingSelection(BaseModel):
    KeepImage1: bool = Field(
        description="Whether to keep Image 1."
    )

    KeepImage2: bool = Field(
        description="Whether to keep Image 2."
    )

    KeepImage3: bool = Field(
        description="Whether to keep Image 3, if present."
    )

    KeepImage4: bool = Field(
        description="Whether to keep Image 4, if present."
    )

    Reasoning: str = Field(
        description=(
            "One-sentence technical explanation identifying the noteworthy "
            "features of any kept images, or why none are worth human "
            "inspection."
        )
    )


config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer evaluating astronomical images. "
        "You will receive a 2x2 grid labeled Image 1 through Image 4. "
        "Some grids may contain fewer than four images. "
        "Evaluate each present image independently. "
        "Do not force a winner. "
        "For each image, decide whether it is scientifically interesting "
        "enough to keep for human inspection. "
        "It is acceptable to keep any number of the present images, "
        "including none or all. "
        "Interesting images may show unusual morphology, asymmetry, "
        "interactions, merger-like features, arcs, rings, tails, shells, "
        "clumps, distortions, rare-looking objects, or other features worth "
        "human inspection. "
        "Reject artifacts, blank images, noisy frames, and obvious "
        "non-astronomical defects. "
        "Return only whether to keep Images 1 through 4 and a brief reasoning."
    ),
    response_mime_type="application/json",
    response_schema=InterestingSelection,
    temperature=0.1,
)


# ── 3. IMAGE UTILITIES ─────────────────────────────────────────────────────

def create_grid_image(image_filenames):
    """
    Stitch up to four images into a labeled 2x2 grid.

    Parameters
    ----------
    image_filenames : list[str]
        A list containing one to four image filenames.

    Returns
    -------
    io.BytesIO
        PNG image data for the generated grid.
    """
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

    for index, filename in enumerate(image_filenames):
        row, col = divmod(index, 2)

        x = margin + col * (GRID_THUMB_W + margin)
        y = margin + row * (
            GRID_THUMB_H + label_h + margin
        )

        image_path = os.path.join(
            IMAGE_DIR,
            filename,
        )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                image = image.resize(
                    (GRID_THUMB_W, GRID_THUMB_H)
                )

                canvas.paste(
                    image,
                    (x, y + label_h),
                )

                draw.text(
                    (x + 6, y + 5),
                    f"Image {index + 1}",
                    fill="white",
                    font=font,
                    stroke_width=1,
                    stroke_fill="black",
                )

        except Exception as error:
            print(
                f"Could not load image {filename}: {error}"
            )

    buffer = io.BytesIO()

    canvas.save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    return buffer


def split_into_groups(active_images, group_size=4):
    """
    Divide the active image pool into numbered groups.

    Parameters
    ----------
    active_images : list[str]
        Current active image filenames.

    group_size : int
        Maximum number of images per group.

    Returns
    -------
    list[tuple[int, list[str]]]
        Numbered image groups.
    """
    return [
        (
            index // group_size + 1,
            active_images[index:index + group_size],
        )
        for index in range(
            0,
            len(active_images),
            group_size,
        )
    ]


# ── 4. WORKER FUNCTION ─────────────────────────────────────────────────────

def process_matches(args):
    """
    Process one worker's assigned image groups.

    Each worker:
    - Creates a 2x2 grid for each group.
    - Uploads the grid to Gemini.
    - Requests independent keep/reject decisions.
    - Writes results to a worker-specific checkpoint CSV.
    """
    (
        worker_id,
        round_number,
        matches,
        api_key,
        artifact_uris,
        progress_queue,
    ) = args

    client = genai.Client(
        api_key=api_key
    )

    checkpoint_file = (
        f"checkpoint_retention_worker_{worker_id}.csv"
    )

    artifact_files = [
        types.File(
            uri=uri,
            mime_type="image/png",
        )
        for uri in artifact_uris
    ]

    def retry_api(function, *function_args, **function_kwargs):
        """
        Retry an API operation using exponential backoff.
        """
        delays = [1, 2, 4, 8, 16]

        for attempt_index, delay in enumerate(delays):
            try:
                return function(
                    *function_args,
                    **function_kwargs,
                )

            except Exception:
                is_final_attempt = (
                    attempt_index == len(delays) - 1
                )

                if is_final_attempt:
                    raise

                time.sleep(delay)

    with open(
        checkpoint_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as checkpoint_handle:
        writer = csv.writer(
            checkpoint_handle
        )

        if os.stat(checkpoint_file).st_size == 0:
            writer.writerow([
                "Round",
                "Match",
                "Filename",
                "Kept",
                "Reasoning",
            ])

        for match_number, group_images in matches:
            grid_buffer = create_grid_image(
                group_images
            )

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
                        f"Now evaluate this 2x2 grid containing "
                        f"{len(group_images)} astronomical image(s), "
                        "labeled in reading order as Image 1 through Image 4. "
                        "Evaluate each present image independently and keep "
                        "any number that are scientifically interesting enough "
                        "for human inspection. "
                        "Ignore unused grid positions when fewer than four "
                        "images are present."
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
                    raise RuntimeError(
                        "Gemini returned no parsed structured response."
                    )

                keep_flags = [
                    bool(data.KeepImage1),
                    bool(data.KeepImage2),
                    bool(data.KeepImage3),
                    bool(data.KeepImage4),
                ][:len(group_images)]

                reasoning = data.Reasoning.strip()

                for image_index, filename in enumerate(group_images):
                    kept = (
                        1
                        if keep_flags[image_index]
                        else 0
                    )

                    writer.writerow([
                        round_number,
                        match_number,
                        filename,
                        kept,
                        reasoning if kept else "",
                    ])

                kept_names = [
                    group_images[image_index]
                    for image_index in range(
                        len(group_images)
                    )
                    if keep_flags[image_index]
                ]

                kept_string = (
                    ", ".join(kept_names)
                    if kept_names
                    else "NONE"
                )

                print(
                    f"Worker {worker_id} | "
                    f"Round {round_number}, "
                    f"Match {match_number}: "
                    f"{', '.join(group_images)} "
                    f"-> kept: {kept_string}"
                )

                checkpoint_handle.flush()
                os.fsync(
                    checkpoint_handle.fileno()
                )

            except Exception as error:
                print(
                    f"Worker {worker_id} error in "
                    f"Round {round_number}, "
                    f"Match {match_number}: "
                    f"{error}"
                )

                fallback_reason = (
                    "Kept due to API or processing failure; "
                    "requires human inspection."
                )

                for filename in group_images:
                    writer.writerow([
                        round_number,
                        match_number,
                        filename,
                        1,
                        fallback_reason,
                    ])

                checkpoint_handle.flush()
                os.fsync(
                    checkpoint_handle.fileno()
                )

            finally:
                if uploaded_grid is not None:
                    try:
                        retry_api(
                            client.files.delete,
                            name=uploaded_grid.name,
                        )

                    except Exception as delete_error:
                        print(
                            f"Worker {worker_id} could not delete "
                            f"temporary grid file: {delete_error}"
                        )

                progress_queue.put(1)

    return worker_id


# ── 5. MAIN ORCHESTRATOR ───────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get(
        "JB_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "JB_API_KEY not found."
        )

    client_main = genai.Client(
        api_key=api_key
    )

    def upload_example_set(file_list, category_name):
        """
        Upload a set of reference images and return their Gemini URIs.
        """
        print(
            f"Uploading {category_name} examples..."
        )

        uploaded_uris = []

        for filename in file_list:
            path = os.path.join(
                IMAGE_DIR,
                filename,
            )

            if not os.path.exists(path):
                print(
                    f"  Warning: {category_name} example not found, "
                    f"skipping: {filename}"
                )

                continue

            uploaded_file = client_main.files.upload(
                file=path
            )

            uploaded_uris.append(
                uploaded_file.uri
            )

            print(
                f"  Uploaded {category_name}: {filename}"
            )

        return uploaded_uris

    artifact_uris = upload_example_set(
        ARTIFACT_EXAMPLES,
        "artifact",
    )

    print(
        f"Done. Ready with {len(artifact_uris)} "
        "artifact examples.\n"
    )

    # Load all supported image files.
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
    # Remove or modify this line to process the full dataset.
    all_images = all_images[4000:5000]

    if len(all_images) < TARGET_REMAINING_IMAGES:
        raise ValueError(
            f"Need at least {TARGET_REMAINING_IMAGES} images. "
            f"Current number: {len(all_images)}"
        )

    print(
        f"Total Images: {len(all_images)}"
    )

    print(
        f"Rounds will continue until exactly "
        f"{TARGET_REMAINING_IMAGES} images remain."
    )

    scores = {
        filename: 0
        for filename in all_images
    }

    reasons = {
        filename: ""
        for filename in all_images
    }

    active_images = all_images[:]

    random.seed(
        RANDOM_SEED
    )

    round_number = 1

    # Continue until the active pool contains exactly 100 images.
    while len(active_images) > TARGET_REMAINING_IMAGES:
        random.shuffle(
            active_images
        )

        round_start_count = len(
            active_images
        )

        all_matches = split_into_groups(
            active_images
        )

        print(
            "\n" + "=" * 70
        )

        print(
            f"Round {round_number}"
        )

        print(
            f"Images entering round: {round_start_count}"
        )

        print(
            f"Matches this round:    {len(all_matches)}"
        )

        print(
            "=" * 70
        )

        manager = Manager()

        progress_queue = manager.Queue()

        chunk_size = math.ceil(
            len(all_matches) / NUM_CORES
        )

        worker_args = []

        for worker_index in range(NUM_CORES):
            start_index = (
                worker_index * chunk_size
            )

            end_index = min(
                (worker_index + 1) * chunk_size,
                len(all_matches),
            )

            if start_index >= len(all_matches):
                continue

            worker_args.append((
                worker_index + 1,
                round_number,
                all_matches[start_index:end_index],
                api_key,
                artifact_uris,
                progress_queue,
            ))

        with Pool(
            processes=NUM_CORES
        ) as pool:
            async_result = pool.map_async(
                process_matches,
                worker_args,
            )

            with tqdm(
                total=len(all_matches),
                desc=f"Processing Round {round_number}",
                unit="match",
            ) as progress_bar:
                completed_matches = 0

                while completed_matches < len(all_matches):
                    try:
                        progress_queue.get(
                            timeout=1.0
                        )

                        progress_bar.update(1)

                        completed_matches += 1

                    except queue.Empty:
                        if async_result.ready():
                            break

            # Re-raise any worker exception.
            async_result.get()

        manager.shutdown()

        # ── ROUND AGGREGATION ──────────────────────────────────────────────

        print(
            f"\nAggregating Round {round_number} results..."
        )

        round_decisions = {}

        seen_this_round = set()

        for worker_index in range(NUM_CORES):
            checkpoint_path = (
                f"checkpoint_retention_worker_"
                f"{worker_index + 1}.csv"
            )

            if not os.path.exists(checkpoint_path):
                continue

            with open(
                checkpoint_path,
                "r",
                encoding="utf-8",
            ) as checkpoint_handle:
                reader = csv.reader(
                    checkpoint_handle
                )

                next(
                    reader,
                    None,
                )

                for row in reader:
                    if not row:
                        continue

                    try:
                        row_round = int(
                            row[0]
                        )

                    except (ValueError, IndexError):
                        continue

                    if row_round != round_number:
                        continue

                    try:
                        filename = row[2]

                        kept = int(
                            row[3]
                        )

                        reasoning = (
                            row[4]
                            if len(row) > 4
                            else ""
                        )

                    except (ValueError, IndexError):
                        continue

                    seen_this_round.add(
                        filename
                    )

                    round_decisions[filename] = (
                        kept,
                        reasoning,
                    )

                    if (
                        kept == 1
                        and filename in reasons
                        and reasoning
                    ):
                        reasons[filename] = reasoning

            os.remove(
                checkpoint_path
            )

        expected_images = set(
            active_images
        )

        actual_images = set(
            seen_this_round
        )

        if expected_images != actual_images:
            missing_images = (
                expected_images - actual_images
            )

            extra_images = (
                actual_images - expected_images
            )

            raise RuntimeError(
                f"Round {round_number} aggregation mismatch.\n"
                f"Missing images: {list(missing_images)[:10]}\n"
                f"Extra images: {list(extra_images)[:10]}\n"
                "Stopping to avoid corrupting results."
            )

        rejected_by_model = [
            filename
            for filename in active_images
            if round_decisions[filename][0] == 0
        ]

        # This is the exact number of removals required to reach 100.
        removals_needed = (
            round_start_count
            - TARGET_REMAINING_IMAGES
        )

        # During ordinary rounds, every rejected image is removed.
        #
        # During the final round, if Gemini rejects more images than needed,
        # only enough rejected images are removed to produce exactly 100.
        removals_to_apply = min(
            removals_needed,
            len(rejected_by_model),
        )

        removed_images = set(
            rejected_by_model[:removals_to_apply]
        )

        next_pool = [
            filename
            for filename in active_images
            if filename not in removed_images
        ]

        removed_this_round = len(
            removed_images
        )

        print(
            f"Round {round_number} complete."
        )

        print(
            f"Images entering:     {round_start_count}"
        )

        print(
            f"Model rejections:    {len(rejected_by_model)}"
        )

        print(
            f"Rejections applied:  {removed_this_round}"
        )

        print(
            f"Images advancing:    {len(next_pool)}"
        )

        active_images = next_pool

        # The script must end only at exactly 100 images.
        if len(active_images) == TARGET_REMAINING_IMAGES:
            print(
                f"Stopping because exactly "
                f"{TARGET_REMAINING_IMAGES} images remain."
            )

            break

        # This prevents an infinite loop if Gemini rejects nothing.
        if removed_this_round == 0:
            raise RuntimeError(
                f"Round {round_number} removed no images. "
                f"There are still {len(active_images)} images remaining, "
                f"so the exact target of "
                f"{TARGET_REMAINING_IMAGES} cannot be reached."
            )

        # This should be impossible because removals are capped.
        if len(active_images) < TARGET_REMAINING_IMAGES:
            raise RuntimeError(
                f"Round {round_number} produced "
                f"{len(active_images)} images. "
                f"The pool must never fall below "
                f"{TARGET_REMAINING_IMAGES}."
            )

        round_number += 1

    # Validate the final pool size.
    if len(active_images) != TARGET_REMAINING_IMAGES:
        raise RuntimeError(
            f"Filtering ended with {len(active_images)} images "
            f"instead of exactly {TARGET_REMAINING_IMAGES}."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "2x2 retention filtering complete."
    )

    print(
        f"Final interesting pool size: "
        f"{len(active_images)}"
    )

    print(
        "=" * 70
    )

    # Assign final binary scores.
    final_interesting_pool = set(
        active_images
    )

    for filename in scores:
        scores[filename] = (
            1
            if filename in final_interesting_pool
            else 0
        )

    # ── 6. METADATA ENRICHMENT ─────────────────────────────────────────────

    print(
        "\nAggregating final results..."
    )

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

            normalized_weighted_count_string = row.get(
                "normalized_weighted_count",
                "",
            ).strip()

            volunteer_rating = ""

            if normalized_weighted_count_string:
                try:
                    volunteer_rating = (
                        float(
                            normalized_weighted_count_string
                        )
                        * 0.01
                    )

                except ValueError:
                    volunteer_rating = ""

            base_name = os.path.splitext(
                filename
            )[0]

            ww_lookup[base_name] = {
                "RA": row.get(
                    "RA",
                    "",
                ),
                "Dec": row.get(
                    "Dec",
                    "",
                ),
                "anomaly_score": row.get(
                    "anomaly_score",
                    "",
                ),
                "URL": row.get(
                    "URL",
                    "",
                ),
                "volunteer_rating": volunteer_rating,
            }

    # ── 7. FINAL OUTPUT ────────────────────────────────────────────────────

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as results_handle:
        writer = csv.writer(
            results_handle
        )

        writer.writerow([
            "Filename",
            "ImageScore",
            "Reasoning",
            "RA",
            "Dec",
            "AnomalyScore",
            "URL",
            "volunteer_rating",
        ])

        # Sort final images first, followed by rejected images.
        sorted_images = sorted(
            scores,
            key=lambda filename: scores[filename],
            reverse=True,
        )

        for filename in sorted_images:
            score = scores[filename]

            reasoning = (
                reasons.get(filename, "")
                if score == 1
                else ""
            )

            base_name = os.path.splitext(
                filename
            )[0]

            metadata = ww_lookup.get(
                base_name,
                {},
            )

            writer.writerow([
                filename,
                score,
                reasoning,
                metadata.get(
                    "RA",
                    "",
                ),
                metadata.get(
                    "Dec",
                    "",
                ),
                metadata.get(
                    "anomaly_score",
                    "",
                ),
                metadata.get(
                    "URL",
                    "",
                ),
                metadata.get(
                    "volunteer_rating",
                    "",
                ),
            ])

    print(
        f"Done! Results saved to {RESULTS_FILE}"
    )