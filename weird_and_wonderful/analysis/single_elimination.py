'''
Pairwise elimination tournament for astronomical image scoring.

Logic:
- Select a subset of images.
- The number of images must be a power of 2.
- In each round, surviving images are randomly reshuffled.
- Images are paired 1v1.
- Gemini sees a side-by-side image labeled Image 1 and Image 2.
- Gemini selects the more interesting image and gives a short reasoning.
- Winner gets +1 ImageScore.
- Loser gets +0 and is eliminated.
- Winner advances to the next round.
- Reasoning is overwritten each time the image wins, so the final reasoning
  reflects the most recent comparison it survived.

Final output format is preserved:
Filename, ImageScore, Reasoning, RA, Dec, AnomalyScore, URL, volunteer_rating
'''

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
WW_DATA_PATH = "ww_data/all_ww_data.csv"
RESULTS_FILE = "single_elimination_{}.csv".format(time.strftime("%m-%d_%H"))
NUM_CORES = 8

PAIR_THUMB_W = 240
PAIR_THUMB_H = 240

ARTIFACT_EXAMPLES = [
    "69555401925880559.png",
    "70386478097659329.png",
    "69569278965207176.png",
    "40128764209818324.png",
    "70342656546335458.png",
    "70342772510452129.png",
]


class SelectedWinner(BaseModel):
    Winner: int = Field(
        ge=1,
        le=2,
        description="The more scientifically interesting image. Must be 1 or 2."
    )
    Reasoning: str = Field(
        description="1-sentence technical explanation for why the winning image is scientifically interesting."
    )


few_shot_context = ""
if os.path.exists("GEMINI.md"):
    with open("GEMINI.md", "r", encoding="utf-8") as f:
        few_shot_context = f"\n\n{f.read()}"


config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer comparing two astronomical images. "
        "The left image is Image 1 and the right image is Image 2. "
        "Your task is to select the image that is more scientifically interesting. "
        "Interesting images may show unusual morphology, asymmetry, interactions, "
        "merger-like features, arcs, rings, tails, shells, clumps, distortions, "
        "rare-looking objects, or other features worth human inspection. "
        "Skip artifacts, blank images, noisy frames, and obvious non-astronomical defects. "
        "Return only the winning image index and a brief reasoning for why it is interesting. "
        f"{few_shot_context}"
    ),
    response_mime_type="application/json",
    response_schema=SelectedWinner,
    temperature=0.1
)


# ── 2. IMAGE UTILITIES ─────────────────────────────────────────────────────
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


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
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
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
        except Exception:
            pass

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── 3. WORKER FUNCTION ─────────────────────────────────────────────────────
def process_matches(args):
    worker_id, round_number, matches, api_key, artifact_uris, progress_queue = args
    client = genai.Client(api_key=api_key)
    checkpoint_file = f"checkpoint_grid_worker_{worker_id}.csv"

    # Re-hydrate artifact file references from URIs for use in contents
    artifact_files = [types.File(uri=uri, mime_type="image/png") for uri in artifact_uris]

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
            writer.writerow(["Round", "Match", "Filename", "Score", "Reasoning", "Loser"])

        for match_number, pair_images in matches:
            pair_buf = create_pair_image(pair_images)

            try:
                uploaded_pair = retry_api(
                    client.files.upload,
                    file=pair_buf,
                    config={"mime_type": "image/png"}
                )

                contents = [
                    "The following images are examples of ARTIFACTS that must NOT be selected:",
                    *artifact_files,
                    (
                        "Now compare this side-by-side pair of astronomical images. "
                        "The left image is Image 1. The right image is Image 2. "
                        "Choose the image that is more scientifically interesting. "
                        "Return the winner and a one-sentence technical explanation."
                    ),
                    uploaded_pair
                ]

                response = retry_api(
                    client.models.generate_content,
                    model="gemini-2.5-flash-lite",
                    contents=contents,
                    config=config
                )

                data = response.parsed

                winner_index = data.Winner - 1
                if winner_index not in [0, 1]:
                    raise ValueError(f"Invalid winner index: {data.Winner}")

                winner = pair_images[winner_index]
                loser = pair_images[1 - winner_index]
                reasoning = data.Reasoning

                # Winner gets 1 point for surviving this round.
                writer.writerow([
                    round_number,
                    match_number,
                    winner,
                    1,
                    reasoning,
                    loser
                ])

                print(
                    f"Worker {worker_id} | Round {round_number}, Match {match_number}: "
                    f"{pair_images[0]} vs {pair_images[1]} -> {winner}"
                )

                f.flush()
                os.fsync(f.fileno())

                retry_api(client.files.delete, name=uploaded_pair.name)

            except Exception as e:
                print(f"Worker {worker_id} error in Round {round_number}, Match {match_number}: {e}")
                pass

            progress_queue.put(1)

    return worker_id


# ── 4. MAIN ORCHESTRATOR ───────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("JB_API_KEY")
    if not api_key:
        raise ValueError("JB_API_KEY not found.")

    # Upload artifact examples once at startup; pass URIs to workers so each
    # worker can reference them without re-uploading
    print("Uploading artifact examples...")
    client_main = genai.Client(api_key=api_key)
    artifact_uris = []

    for fname in ARTIFACT_EXAMPLES:
        path = os.path.join(IMAGE_DIR, fname)
        uploaded = client_main.files.upload(file=path)
        artifact_uris.append(uploaded.uri)
        print(f"  Uploaded artifact: {fname}")

    print(f"Done. {len(artifact_uris)} artifact examples ready.\n")

    all_images = [img for img in os.listdir(IMAGE_DIR) if img.endswith((".png", ".jpg"))]
    all_images = all_images[3000:4024]  # Subsetting for testing; this gives 1024 images

    if not is_power_of_two(len(all_images)):
        raise ValueError(
            f"Number of images must be a power of 2 for elimination tournament. "
            f"Current number: {len(all_images)}"
        )

    print(f"Total Images: {len(all_images)}")
    print(f"Total Rounds: {int(math.log2(len(all_images)))}")

    scores = {img: 0 for img in all_images}
    reasons = {img: "" for img in all_images}

    active_images = all_images[:]
    random.seed(42)

    round_number = 1

    while len(active_images) > 1:
        random.shuffle(active_images)

        all_matches = []
        for i in range(0, len(active_images), 2):
            match_number = len(all_matches) + 1
            pair = [active_images[i], active_images[i + 1]]
            all_matches.append((match_number, pair))

        print("\n" + "=" * 70)
        print(f"Round {round_number}")
        print(f"Images entering round: {len(active_images)}")
        print(f"Matches this round:    {len(all_matches)}")
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

            with tqdm(total=len(all_matches), desc=f"Processing Round {round_number}", unit="match") as pbar:
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

        winners = []

        for i in range(NUM_CORES):
            cp = f"checkpoint_grid_worker_{i + 1}.csv"

            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)

                    for row in reader:
                        # row: Round, Match, Filename, Score, Reasoning, Loser
                        if not row:
                            continue

                        try:
                            row_round = int(row[0])
                        except (ValueError, IndexError):
                            continue

                        if row_round != round_number:
                            continue

                        fname = row[2]

                        try:
                            base_score = int(row[3])
                        except (ValueError, IndexError):
                            base_score = 0

                        reasoning = row[4] if len(row) > 4 else ""

                        if fname in scores:
                            scores[fname] += base_score

                            # Overwrite previous reasoning with latest winning reasoning.
                            reasons[fname] = reasoning

                            winners.append(fname)

                os.remove(cp)

        if len(winners) != len(all_matches):
            raise RuntimeError(
                f"Round {round_number} produced {len(winners)} winners, "
                f"but expected {len(all_matches)}. "
                f"Stopping to avoid corrupting the tournament."
            )

        active_images = winners

        print(f"Round {round_number} complete.")
        print(f"Images advancing: {len(active_images)}")

        round_number += 1

    champion = active_images[0]
    print("\n" + "=" * 70)
    print("Tournament complete.")
    print(f"Champion: {champion}")
    print(f"Champion ImageScore: {scores[champion]}")
    print(f"Champion Reasoning: {reasons[champion]}")
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
                    nwc_val = float(nwc_str) * 0.01

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

        for img in sorted(scores, key=scores.get, reverse=True):
            score = scores[img]
            reasoning = reasons[img] if score > 0 else ""

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