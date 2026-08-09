"""
Iterative pairwise retention filtering for astronomical image scoring (Qwen Few-Shot).

Logic:
- Select a subset of images.
- In each round, active images are randomly reshuffled.
- Images are paired 1v1.
- Qwen sees a side-by-side image labeled Image 1 and Image 2.
- Qwen evaluates each image independently and decides whether each image
  is scientifically interesting enough to keep for human inspection (without explanations).
- Qwen may keep neither image, one image, or both images.
- Kept images advance to the next round. Rejected images are removed from the active pool.
- Exactly 10 rounds are run (or fewer if all images are eliminated).

Final output format is preserved:
Filename, ImageScore, Reasoning, RA, Dec, AnomalyScore, URL, volunteer_rating
(Reasoning is written as empty to preserve schema)
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
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

# ── 1. CONFIGURATION ───────────────────────────────────────────────────────
IMAGE_DIR = "/geir_data/scr/yxcheng/png_images/"
WW_DATA_PATH = "../ww_data/all_ww_data.csv"
RESULTS_DIR = "ai_csvs"
EXPERT_CSV_DIR = "expert_csvs"
# The same dataset is run 3 times; each run writes its own CSV.
RESULT_TEMPLATE = os.path.join(RESULTS_DIR, "qwen_tournament_few_shot_990_run{}.csv")

OPENROUTER_MODEL = "qwen/qwen3.5-397b-a17b"
PAIR_THUMB_W = 240
PAIR_THUMB_H = 240
MAX_ROUNDS = 10
NUM_CORES = 20
RANDOM_SEED = 42

ARTIFACT_EXAMPLES = [
    "69555401925880559.png", "70386478097659329.png", "69569278965207176.png",
    "40128764209818324.png", "70342656546335458.png", "70342772510452129.png",
]
INTERESTING_EXAMPLES = [
    "70347028823040378.png", "70365200829662492.png", "41214781050350168.png",
    "70342656546334671.png", "41192936846682523.png",
]
BORING_EXAMPLES = [
    "70405045241278791.png", "70381951202130938.png", "70391567633903843.png",
    "41218771074969925.png", "69563914551059502.png",
]

class InterestingSelection(BaseModel):
    KeepImage1: bool = Field(description="True if Image 1 is scientifically interesting enough to keep.")
    KeepImage2: bool = Field(description="True if Image 2 is scientifically interesting enough to keep.")

SYSTEM_INSTRUCTION = (
    "You are an expert astronomer evaluating astronomical images. "
    "The left image is Image 1 and the right image is Image 2. "
    "Evaluate each image independently. Do not force a winner. "
    "For each image, decide whether it is scientifically interesting enough "
    "to keep for human inspection. It is acceptable to keep neither, one, or both. "
    "Interesting images may show unusual morphology, asymmetry, interactions, "
    "merger-like features, arcs, rings, tails, shells, clumps, distortions, "
    "rare-looking objects, or other features worth human inspection. "
    "Reject artifacts, blank images, noisy frames, and obvious non-astronomical defects. "
    "Return only whether to keep Image 1 and whether to keep Image 2."
)

# ── 2. IMAGE UTILITIES ─────────────────────────────────────────────────────
def create_pair_image(image_filenames):
    margin = 8
    label_h = 32
    canvas_w = PAIR_THUMB_W * 2 + margin * 3
    canvas_h = PAIR_THUMB_H + label_h + margin * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
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
                draw.text((x + 6, 5), label, fill="white", font=font, stroke_width=1, stroke_fill="black")
        except Exception:
            pass

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf

def split_into_pairs(active_images):
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

def encode_file_to_b64(fname):
    path = os.path.join(IMAGE_DIR, fname)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        return None

# ── 3. WORKER FUNCTION ─────────────────────────────────────────────────────
def process_matches(args):
    worker_id, round_number, matches, api_key, artifact_b64s, interesting_b64s, boring_b64s, progress_queue = args
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    checkpoint_file = f"checkpoint_retention_worker_{worker_id}.csv"
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
            writer.writerow(["Round", "Match", "Filename", "Kept", "Reasoning", "InputTokens", "OutputTokens", "TotalTokens"])

        for match_number, pair_images in matches:
            pair_buf = create_pair_image(pair_images)
            pair_b64 = base64.b64encode(pair_buf.getvalue()).decode("utf-8")

            try:
                contents = []
                contents.append({"type": "text", "text": "The following images are examples of ARTIFACTS that must NOT be kept:"})
                for b64 in artifact_b64s:
                    if b64: contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

                contents.append({"type": "text", "text": "The following images are examples of INTERESTING targets worth keeping:"})
                for b64 in interesting_b64s:
                    if b64: contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

                contents.append({"type": "text", "text": "The following images are examples of BORING targets that should be filtered out:"})
                for b64 in boring_b64s:
                    if b64: contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

                contents.append({
                    "type": "text",
                    "text": (
                        "Now evaluate this side-by-side pair of astronomical images. "
                        "The left image is Image 1. The right image is Image 2. "
                        "Evaluate each image independently. Do not force a winner. "
                        "For each image, decide whether it is scientifically interesting "
                        "enough to keep for human inspection."
                    )
                })
                contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pair_b64}"}})

                response = retry_api(
                    client.beta.chat.completions.parse,
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": contents}
                    ],
                    response_format=InterestingSelection,
                    temperature=0.1,
                    extra_body={
                        "usage": {"include": True},
                        "reasoning": {"enabled": False},
                        "provider": {"ignore": ["DigitalOcean"]},
                    }
                )

                data = response.choices[0].message.parsed

                if hasattr(response, "usage") and response.usage:
                    input_tok = getattr(response.usage, "prompt_tokens", 0) or 0
                    output_tok = getattr(response.usage, "completion_tokens", 0) or 0
                else:
                    input_tok, output_tok = 0, 0
                total_tok = input_tok + output_tok

                total_input_tokens += input_tok
                total_output_tokens += output_tok

                keep_flags = [bool(data.KeepImage1), bool(data.KeepImage2)]
                for idx, fname in enumerate(pair_images):
                    kept = 1 if keep_flags[idx] else 0
                    writer.writerow([round_number, match_number, fname, kept, "", input_tok, output_tok, total_tok])

                f.flush()
                os.fsync(f.fileno())

            except Exception as e:
                print(f"Worker {worker_id} error in Round {round_number}, Match {match_number}: {e}")
                for fname in pair_images:
                    writer.writerow([round_number, match_number, fname, 1, "", 0, 0, 0])
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
def run_experiment(run_idx, api_key, all_images, artifact_b64s, interesting_b64s, boring_b64s):
    results_file = RESULT_TEMPLATE.format(run_idx)
    print("\n" + "#" * 70)
    print(f"# RUN {run_idx} of 3  ->  {results_file}")
    print("#" * 70)

    # Remove any stale checkpoints from a previous run.
    for _i in range(NUM_CORES):
        _cp = f"checkpoint_retention_worker_{_i + 1}.csv"
        if os.path.exists(_cp):
            os.remove(_cp)

    scores = {img: 0 for img in all_images}
    active_images = all_images[:]
    random.seed(RANDOM_SEED)

    round_number = 1
    grand_total_input_tokens = 0
    grand_total_output_tokens = 0

    while round_number <= MAX_ROUNDS:
        random.shuffle(active_images)
        round_start_count = len(active_images)
        all_matches, bye_image = split_into_pairs(active_images)

        print(f"\n=== Round {round_number} | Entering: {round_start_count} | Matches: {len(all_matches)} ===")

        manager = Manager()
        progress_queue = manager.Queue()
        chunk_size = math.ceil(len(all_matches) / NUM_CORES)
        worker_args = []

        for i in range(NUM_CORES):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(all_matches))
            if start < len(all_matches):
                worker_args.append((
                    i + 1, round_number, all_matches[start:end], api_key,
                    artifact_b64s, interesting_b64s, boring_b64s, progress_queue
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
                        if async_result.ready(): break
            worker_results = async_result.get()

        grand_total_input_tokens += sum(r[1] for r in worker_results)
        grand_total_output_tokens += sum(r[2] for r in worker_results)

        round_decisions = {}
        seen_this_round = set()

        for i in range(NUM_CORES):
            cp = f"checkpoint_retention_worker_{i + 1}.csv"
            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if not row or int(row[0]) != round_number: continue
                        fname, kept = row[2], int(row[3])
                        seen_this_round.add(fname)
                        round_decisions[fname] = kept
                os.remove(cp)

        if bye_image is not None:
            seen_this_round.add(bye_image)
            round_decisions[bye_image] = 1

        rejected_by_model = [fname for fname in active_images if round_decisions[fname] == 0]
        removed_images = set(rejected_by_model)
        next_pool = [fname for fname in active_images if fname not in removed_images]

        for fname in next_pool: scores[fname] += 1
        active_images = next_pool
        round_number += 1

        if not active_images: break

    # ── 5. FINAL AGGREGATE / METADATA ENRICHMENT ──────────────────────────
    ww_lookup = {}
    with open(WW_DATA_PATH, newline="", encoding="utf-8") as ww_f:
        for row in csv.DictReader(ww_f):
            fname = row.get("filename", "").strip()
            if fname:
                nwc_str = row.get("normalized_weighted_count", "").strip()
                nwc_val = float(nwc_str) * 0.01 if nwc_str else ""
                ww_lookup[os.path.splitext(fname)[0]] = {
                    "RA": row.get("RA", ""), "Dec": row.get("Dec", ""),
                    "anomaly_score": row.get("anomaly_score", ""), "URL": row.get("URL", ""),
                    "volunteer_rating": nwc_val
                }

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "ImageScore", "Reasoning", "RA", "Dec", "AnomalyScore", "URL", "volunteer_rating"])

        for img in sorted(scores, key=scores.get, reverse=True):
            meta = ww_lookup.get(os.path.splitext(img)[0], {})
            writer.writerow([
                img, scores[img], "", meta.get("RA", ""), meta.get("Dec", ""),
                meta.get("anomaly_score", ""), meta.get("URL", ""), meta.get("volunteer_rating", "")
            ])

        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", grand_total_input_tokens])
        writer.writerow(["# TotalOutputTokens", grand_total_output_tokens])

    print(f"Done! Results saved to {results_file}")


# ── 6. MAIN: RUN THE SAME DATASET THREE TIMES ───────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JB_API_KEY")
    if not api_key:
        raise ValueError("API Key not found.")

    # Encode examples once; reused across all 3 runs.
    print("Encoding examples once to Base64...")
    artifact_b64s = [encode_file_to_b64(x) for x in ARTIFACT_EXAMPLES if encode_file_to_b64(x) is not None]
    interesting_b64s = [encode_file_to_b64(x) for x in INTERESTING_EXAMPLES if encode_file_to_b64(x) is not None]
    boring_b64s = [encode_file_to_b64(x) for x in BORING_EXAMPLES if encode_file_to_b64(x) is not None]
    print(f"Done. Prepared {len(artifact_b64s)} artifacts, {len(interesting_b64s)} interesting, and {len(boring_b64s)} boring examples.\n")

    # Test set = every image the experts scored (expert_csvs/).
    all_images = load_expert_image_names(EXPERT_CSV_DIR)
    # Exclude the 10 few-shot example images so all runs evaluate the same 990 images.
    all_images = [img for img in all_images if img not in set(INTERESTING_EXAMPLES + BORING_EXAMPLES)]
    print(f"Loaded {len(all_images)} images from expert CSVs.\n")

    # Same 990-image dataset, run three times.
    for run_idx in range(1, 4):
        run_experiment(run_idx, api_key, all_images, artifact_b64s, interesting_b64s, boring_b64s)