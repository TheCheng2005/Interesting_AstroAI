"""
I ask the Gemini 3.1 flash-lite (low thinking level) to pick any number of interesting images from a grid. 
Gemini also scores each selected image on a scale of 1-5.
The grid is nxn (currently 4x4) and the images are randomly sampled from the pool. 
Each image is shown exactly n times (currently 10)
The model is prompted with a few shot of artifact examples to help it understand what NOT to select.
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
RESULT_TEMPLATE = "ai_csvs/gemini_grid_likert_no_shot_990_run{}.csv"
NUM_CORES = 20
ROUNDS_PER_IMAGE = 10  # How many different random batches each image appears in
GRID_DIM = 4           # 4x4 grid
BATCH_SIZE = GRID_DIM * GRID_DIM

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
    GridIndex: int = Field(ge=1, le=16, description="Index (1-16) of the interesting image in the 4x4 grid (reading top-to-bottom, left-to-right).")
    Score: int = Field(ge=1, le=5, description="Scientific interest score from 1 (mildly interesting) to 5 (exceptionally interesting).")

few_shot_context = ""
if os.path.exists("GEMINI.md"):
    with open("GEMINI.md", "r", encoding="utf-8") as f:
        few_shot_context = f"\n\n{f.read()}"

config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer evaluating a 4x4 grid of astronomical images. "
        "The images are indexed 1-4 (top row), 5-8 (second row), 9-12 (third row), and 13-16 (bottom row). "
        "Your task:\n"
        "1. Select any images that are scientifically interesting (skip artifacts and blank/noisy frames).\n"
        "2. For each selected image, assign a Score from 1 to 5:\n"
        "   1 = mildly interesting, 2 = somewhat interesting, 3 = interesting, "
        "   4 = very interesting, 5 = exceptionally interesting.\n"
        "Return a list of selected images. If none are interesting, return an empty list.\n"
        f"{few_shot_context}"
    ),
    response_mime_type="application/json",
    response_schema=list[SelectedImage],
    temperature=0.1,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
)

# ── 2. GRID UTILITIES ──────────────────────────────────────────────────────
def create_grid(image_filenames):
    """Stitches 16 images into a 4x4 grid with borders and labels."""
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

# ── 3. WORKER FUNCTION ─────────────────────────────────────────────────────
def process_batches(args):
    worker_id, batches, api_key, artifact_uris, progress_queue = args
    client = genai.Client(api_key=api_key)
    checkpoint_file = f"checkpoint_grid_worker_{worker_id}.csv"

    # Re-hydrate artifact file references from URIs for use in contents
    artifact_files = [types.File(uri=uri, mime_type="image/png") for uri in artifact_uris]

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

            try:
                uploaded_grid = retry_api(client.files.upload, file=grid_buf, config={'mime_type': 'image/png'})
                # Prepend artifact examples to every prompt so the model has
                # a visual reference for what to reject before seeing the grid
                contents = [
                    "The following images are examples of ARTIFACTS that must NOT be selected:",
                    *artifact_files,
                    (
                        "Now analyze this 4x4 grid of astronomical images. "
                        "Select any scientifically interesting images, rate each 1-5. "
                    ),
                    uploaded_grid
                ]

                response = retry_api(
                    client.models.generate_content,
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                    config=config
                )
                
                # Token Tracking
                usage = response.usage_metadata
                input_tok = getattr(usage, "prompt_token_count", 0) or 0
                output_tok = getattr(usage, "candidates_token_count", 0) or 0
                total_tok = getattr(usage, "total_token_count", 0) or (input_tok + output_tok)

                total_input_tokens += input_tok
                total_output_tokens += output_tok

                # data is now directly a list of SelectedImage objects
                data = response.parsed
                print(f"Worker {worker_id} processed a batch. Found {len(data) if data else 0} interesting images. [tokens in={input_tok} out={output_tok}]")

                # Write the valid selected images to the worker's CSV
                if data:
                    for item in data:
                        idx = item.GridIndex - 1 
                        if 0 <= idx < len(batch_images) and batch_images[idx] is not None:
                            filename = batch_images[idx]
                            writer.writerow([filename, item.Score, "", input_tok, output_tok, total_tok])

                f.flush()
                os.fsync(f.fileno())

                retry_api(client.files.delete, name=uploaded_grid.name)

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
def run_experiment(run_idx, api_key, all_images, artifact_uris):
    results_file = RESULT_TEMPLATE.format(run_idx)
    print("\n" + "#" * 70)
    print(f"# RUN {run_idx} of 3  ->  {results_file}")
    print("#" * 70)

    # Remove any stale checkpoints from a previous run.
    for _i in range(NUM_CORES):
        _cp = f"checkpoint_grid_worker_{_i + 1}.csv"
        if os.path.exists(_cp):
            os.remove(_cp)

    # Create the master randomized pool; every image appears in 10 random batches
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
            worker_args.append((i+1, all_batches[start:end], api_key, artifact_uris, progress_queue))

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
    reasons = {img: "" for img in all_images}

    for i in range(NUM_CORES):
        cp = f"checkpoint_grid_worker_{i+1}.csv"
        if os.path.exists(cp):
            with open(cp, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    # row: Filename, Score, Reasoning, InputTokens, OutputTokens, TotalTokens
                    if not row or row[0] not in scores:
                        continue
                    fname = row[0]
                    try:
                        base_score = int(row[1])
                    except (ValueError, IndexError):
                        base_score = 0
                        
                    # Accumulate score across all times this image was selected
                    scores[fname] += base_score
                    
                    # Keep the first reasoning provided for this image
                    if not reasons[fname] and len(row) > 2:
                        reasons[fname] = row[2]
            os.remove(cp)

    # Load ww_data lookup for metadata enrichment
    ww_lookup = {}
    with open(WW_DATA_PATH, newline='', encoding='utf-8') as ww_f:
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

    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "ImageScore", "Reasoning", "RA", "Dec", "AnomalyScore", "URL", "volunteer_rating"])
        for img in sorted(scores, key=scores.get, reverse=True):
            score = scores[img]
            reasoning = reasons[img] if score > 0 else ""
            base_name = os.path.splitext(img)[0]
            meta = ww_lookup.get(base_name, {})
            writer.writerow([
                img, score, reasoning,
                meta.get("RA", ""), meta.get("Dec", ""),
                meta.get("anomaly_score", ""), meta.get("URL", ""),
                meta.get("volunteer_rating", "")
            ])

        # Token usage summary appended as trailing metadata rows.
        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", grand_total_input_tokens])
        writer.writerow(["# TotalOutputTokens", grand_total_output_tokens])
        writer.writerow(["# TotalTokens", grand_total_tokens])

    print(f"Done! Results saved to {results_file}")
    print(f"Token usage summary: input={grand_total_input_tokens:,}  output={grand_total_output_tokens:,}  total={grand_total_tokens:,}")


# ── 6. MAIN: RUN THE SAME DATASET THREE TIMES ───────────────────────────────
if __name__ == '__main__':
    api_key = os.environ.get("JB_API_KEY")
    if not api_key:
        raise ValueError("JB_API_KEY not found.")

    # Upload artifact examples once; reused across all 3 runs.
    print("Uploading artifact examples...")
    client_main = genai.Client(api_key=api_key)
    artifact_uris = []
    for fname in ARTIFACT_EXAMPLES:
        path = os.path.join(IMAGE_DIR, fname)
        if os.path.exists(path):
            uploaded = client_main.files.upload(file=path)
            artifact_uris.append(uploaded.uri)
            print(f"  Uploaded artifact: {fname}")
        else:
            print(f"  Warning: Artifact example not found, skipping: {fname}")
    print(f"Done. {len(artifact_uris)} artifact examples ready.\n")

    # Test set = every image the experts scored (expert_csvs/).
    all_images = load_expert_image_names(EXPERT_CSV_DIR)
    # Exclude the 10 few-shot example images for parity with the few-shot runs.
    all_images = [img for img in all_images if img not in set(FEW_SHOT_EXCLUDED)]
    print(f"Loaded {len(all_images)} images from expert CSVs.\n")

    # Same 990-image dataset, run three times.
    for run_idx in range(1, 4):
        run_experiment(run_idx, api_key, all_images, artifact_uris)