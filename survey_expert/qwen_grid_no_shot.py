"""
Evaluating astronomical images in a 4x4 Grid using Qwen (No-Shot).

Logic:
- Group images into 4x4 grids.
- Each image appears in exactly 10 different random batches.
- Qwen selects scientifically interesting images and rates them 1-5 (no explanations).

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
RESULT_TEMPLATE = os.path.join(RESULTS_DIR, "qwen_grid_likert_no_shot_990_run{}.csv")
NUM_CORES = 20
ROUNDS_PER_IMAGE = 10  # How many different random batches each image appears in
GRID_DIM = 4           # 4x4 grid
BATCH_SIZE = GRID_DIM * GRID_DIM

OPENROUTER_MODEL = "qwen/qwen3.5-397b-a17b"

ARTIFACT_EXAMPLES = [
    "69555401925880559.png",
    "70386478097659329.png",
    "69569278965207176.png",
    "40128764209818324.png",
    "70342656546335458.png",
    "70342772510452129.png",
]

# Excluded to ensure parity with the 990-image test set from few-shot runs
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


class SelectedImage(BaseModel):
    GridIndex: int = Field(
        ge=1,
        le=16,
        description="Index (1-16) of the interesting image in the 4x4 grid (reading top-to-bottom, left-to-right)."
    )
    Score: int = Field(
        ge=1,
        le=5,
        description="Scientific interest score from 1 (mildly interesting) to 5 (exceptionally interesting)."
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
    "The images are indexed 1-4 (top row), 5-8 (second row), 9-12 (third row), and 13-16 (bottom row). "
    "Your task:\n"
    "1. Select any images that are scientifically interesting (skip artifacts and blank/noisy frames).\n"
    "2. For each selected image, assign a Score from 1 to 5:\n"
    "   1 = mildly interesting, 2 = somewhat interesting, 3 = interesting, "
    "   4 = very interesting, 5 = exceptionally interesting.\n"
    "Return a list of selected images. If none are interesting, return an empty list.\n"
    f"{few_shot_context}"
)


# ── 2. GRID UTILITIES ──────────────────────────────────────────────────────
def create_grid(image_filenames):
    thumb_w, thumb_h = 200, 200
    margin = 4
    grid_w = (thumb_w * GRID_DIM) + (margin * (GRID_DIM + 1))
    grid_h = (thumb_h * GRID_DIM) + (margin * (GRID_DIM + 1))

    canvas = Image.new('RGB', (grid_w, grid_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, fname in enumerate(image_filenames):
        if not fname: continue

        row = i // GRID_DIM
        col = i % GRID_DIM

        img_path = os.path.join(IMAGE_DIR, fname)
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize((thumb_w, thumb_h))
                x = margin + col * (thumb_w + margin)
                y = margin + row * (thumb_h + margin)
                canvas.paste(img, (x, y))
                draw.text((x + 5, y + 5), str(i + 1), fill="white", font=font, stroke_width=1, stroke_fill="black")
        except Exception:
            pass

    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    buf.seek(0)
    return buf


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
def process_batches(args):
    worker_id, batches, api_key, artifact_b64s, progress_queue = args
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    checkpoint_file = f"checkpoint_grid_worker_{worker_id}.csv"

    total_input_tokens = 0
    total_output_tokens = 0

    def retry_api(func, *args, **kwargs):
        delays = [1, 2, 4, 8, 16]
        for i, delay in enumerate(delays):
            try: return func(*args, **kwargs)
            except Exception as e:
                if i == len(delays) - 1: raise e
                time.sleep(delay)

    with open(checkpoint_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if os.stat(checkpoint_file).st_size == 0:
            writer.writerow(["Filename", "Score", "Reasoning", "InputTokens", "OutputTokens", "TotalTokens"])

        for batch_images in batches:
            grid_buf = create_grid(batch_images)
            grid_b64 = base64.b64encode(grid_buf.getvalue()).decode("utf-8")

            try:
                contents = []
                contents.append({"type": "text", "text": "The following images are examples of ARTIFACTS that must NOT be selected:"})
                for b64 in artifact_b64s:
                    if b64:
                        contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

                contents.append({
                    "type": "text",
                    "text": (
                        "Now analyze this 4x4 grid of astronomical images. "
                        "Select any scientifically interesting images and rate each 1-5."
                    )
                })

                contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{grid_b64}"}})

                response = retry_api(
                    client.beta.chat.completions.parse,
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": contents}
                    ],
                    response_format=SelectedImages,
                    temperature=0.1,
                    extra_body={
                        "usage": {"include": True},
                        "reasoning": {"enabled": False},
                        "provider": {"ignore": ["DigitalOcean"]},
                    }
                )
                
                # Token Tracking
                if hasattr(response, "usage") and response.usage:
                    input_tok = getattr(response.usage, "prompt_tokens", 0) or 0
                    output_tok = getattr(response.usage, "completion_tokens", 0) or 0
                else:
                    input_tok, output_tok = 0, 0
                total_tok = input_tok + output_tok

                total_input_tokens += input_tok
                total_output_tokens += output_tok

                data = response.choices[0].message.parsed
                parsed_selections = data.selections if data else []
                print(f"Worker {worker_id} processed a batch. Found {len(parsed_selections)} interesting images. [tokens in={input_tok} out={output_tok}]")

                if parsed_selections:
                    for item in parsed_selections:
                        idx = item.GridIndex - 1 
                        if 0 <= idx < len(batch_images) and batch_images[idx] is not None:
                            filename = batch_images[idx]
                            writer.writerow([filename, item.Score, "", input_tok, output_tok, total_tok])

                f.flush()
                os.fsync(f.fileno())

            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                pass

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
def run_experiment(run_idx, api_key, all_images, artifact_b64s):
    results_file = RESULT_TEMPLATE.format(run_idx)
    print("\n" + "#" * 70)
    print(f"# RUN {run_idx} of 3  ->  {results_file}")
    print("#" * 70)

    # Remove any stale checkpoints from a previous run.
    for _i in range(NUM_CORES):
        _cp = f"checkpoint_grid_worker_{_i + 1}.csv"
        if os.path.exists(_cp):
            os.remove(_cp)

    # Create the master randomized pool; every image appears in ROUNDS_PER_IMAGE batches
    master_pool = all_images * ROUNDS_PER_IMAGE
    random.seed(42)
    random.shuffle(master_pool)

    # Group into batches of 16 (GRID_DIM * GRID_DIM)
    all_batches = []
    for i in range(0, len(master_pool), BATCH_SIZE):
        batch = master_pool[i : i + BATCH_SIZE]
        while len(batch) < BATCH_SIZE:
            batch.append(None)
        all_batches.append(batch)

    print(f"Total Unique Images: {len(all_images)}")
    print(f"Total Evaluations:   {len(master_pool)}")
    print(f"Total 4x4 Grids:     {len(all_batches)}")

    manager = Manager()
    progress_queue = manager.Queue()

    chunk_size = math.ceil(len(all_batches) / NUM_CORES)
    worker_args = []
    for i in range(NUM_CORES):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(all_batches))
        if start < len(all_batches):
            worker_args.append((
                i+1, 
                all_batches[start:end], 
                api_key, 
                artifact_b64s, 
                progress_queue
            ))

    with Pool(processes=NUM_CORES) as pool:
        async_result = pool.map_async(process_batches, worker_args)
        with tqdm(total=len(all_batches), desc="Processing Grids", unit="grid") as pbar:
            completed = 0
            while completed < len(all_batches):
                try:
                    _ = progress_queue.get(timeout=1.0)
                    pbar.update(1)
                    completed += 1
                except queue.Empty:
                    if async_result.ready(): break
                    
        worker_results = async_result.get()

    grand_total_input_tokens = sum(r[1] for r in worker_results)
    grand_total_output_tokens = sum(r[2] for r in worker_results)
    grand_total_tokens = grand_total_input_tokens + grand_total_output_tokens

    # ── 5. AGGREGATE ──────────────────────────────────────────────────────
    print("\nAggregating results...")
    scores = {img: 0 for img in all_images}

    for i in range(NUM_CORES):
        cp = f"checkpoint_grid_worker_{i+1}.csv"
        if os.path.exists(cp):
            with open(cp, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if not row or row[0] not in scores:
                        continue
                    fname = row[0]
                    try:
                        base_score = int(row[1])
                    except (ValueError, IndexError):
                        base_score = 0
                        
                    scores[fname] += base_score
            os.remove(cp)

    ww_lookup = {}
    with open(WW_DATA_PATH, newline='', encoding='utf-8') as ww_f:
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
                ww_lookup[os.path.splitext(fname)[0]] = {
                    "RA": row.get("RA", ""),
                    "Dec": row.get("Dec", ""),
                    "anomaly_score": row.get("anomaly_score", ""),
                    "URL": row.get("URL", ""),
                    "volunteer_rating": nwc_val
                }

    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "ImageScore", "Reasoning", "RA", "Dec", "AnomalyScore", "URL", "volunteer_rating"])
        for img in sorted(scores, key=scores.get, reverse=True):
            score = scores[img]
            base_name = os.path.splitext(img)[0]
            meta = ww_lookup.get(base_name, {})
            writer.writerow([
                img, score, "",
                meta.get("RA", ""), meta.get("Dec", ""),
                meta.get("anomaly_score", ""), meta.get("URL", ""),
                meta.get("volunteer_rating", "")
            ])

        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", grand_total_input_tokens])
        writer.writerow(["# TotalOutputTokens", grand_total_output_tokens])
        writer.writerow(["# TotalTokens", grand_total_tokens])

    print(f"Done! Results saved to {results_file}")
    print(f"Token usage summary: input={grand_total_input_tokens:,}  output={grand_total_output_tokens:,}  total={grand_total_tokens:,}")


# ── 6. MAIN: RUN THE SAME DATASET THREE TIMES ───────────────────────────────
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JB_API_KEY")
    if not api_key:
        raise ValueError("API Key not found.")

    # Encode artifact examples once; reused across all 3 runs.
    print("Encoding artifact examples to Base64...")
    artifact_b64s = [encode_file_to_b64(x) for x in ARTIFACT_EXAMPLES if encode_file_to_b64(x) is not None]
    print(f"Done. Prepared {len(artifact_b64s)} artifact examples.\n")

    # Test set = every image the experts scored (expert_csvs/).
    all_images = load_expert_image_names(EXPERT_CSV_DIR)
    # Exclude the 10 few-shot example images for parity with the few-shot runs.
    all_images = [img for img in all_images if img not in set(FEW_SHOT_EXCLUDED)]
    print(f"Loaded {len(all_images)} images from expert CSVs.\n")

    # Same 990-image dataset, run three times.
    for run_idx in range(1, 4):
        run_experiment(run_idx, api_key, all_images, artifact_b64s)