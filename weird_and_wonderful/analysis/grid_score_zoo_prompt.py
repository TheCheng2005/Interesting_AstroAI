'''
I ask the Gemini to pick any number of interesting images from a grid. 
Gemini also scores each selected image on a scale of 1-5.
The grid is nxn (currently 3x3) and the images are randomly sampled from the pool. 
Each image is shown exactly n times (currently 10)
The model is prompted with a few shot of artifact examples to help it understand what NOT to select.
I tell the model that it is a Galaxy Zoo Volunteer and not an expert Astronomer!
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
RESULTS_FILE = "grid_results_score_{}.csv".format(time.strftime("%m-%d_%H"))
NUM_CORES = 8
ROUNDS_PER_IMAGE = 10  # How many different random batches each image appears in
GRID_DIM = 3           # 3x3 grid
BATCH_SIZE = GRID_DIM * GRID_DIM

ARTIFACT_EXAMPLES = [
    "69555401925880559.png",
    "70386478097659329.png",
    "69569278965207176.png",
    "40128764209818324.png",
    "70342656546335458.png",
    "70342772510452129.png",
]

class SelectedImage(BaseModel):
    GridIndex: int = Field(ge=1, le=9, description="Index (1-9) of the interesting image in the 3x3 grid (reading top-to-bottom, left-to-right).")
    Reasoning: str = Field(description="1-sentence technical explanation for why this specific image is scientifically interesting.")
    Score: int = Field(ge=1, le=5, description="Scientific interest score from 1 (mildly interesting) to 5 (exceptionally interesting).")

few_shot_context = ""
if os.path.exists("GEMINI_zoo.md"):
    with open("GEMINI_zoo.md", "r", encoding="utf-8") as f:
        few_shot_context = f"\n\n{f.read()}"

config = types.GenerateContentConfig(
    system_instruction=(
        "You are a GalaxyZoo Volunteer evaluating a 3x3 grid of astronomical images. "
        "The images are indexed 1-3 (top row), 4-6 (middle row), and 7-9 (bottom row). "
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
    temperature=0.1
)

# ── 2. GRID UTILITIES ──────────────────────────────────────────────────────
def create_grid(image_filenames):
    """Stitches 9 images into a 3x3 grid with borders and labels."""
    thumb_w, thumb_h = 180, 180
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
            writer.writerow(["Filename", "Score", "Reasoning"])

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
                        "Now analyze this 3x3 grid of astronomical images. "
                        "Select any scientifically interesting images, rate each 1-5. "
                    ),
                    uploaded_grid
                ]

                response = retry_api(
                    client.models.generate_content,
                    model="gemini-2.5-flash-lite",
                    contents=contents,
                    config=config
                )
                
                # data is now directly a list of SelectedImage objects
                data = response.parsed
                print(f"Worker {worker_id} processed a batch. Found {len(data)} interesting images.")

                # Write the valid selected images to the worker's CSV
                if data:
                    for item in data:
                        idx = item.GridIndex - 1 
                        if 0 <= idx < len(batch_images) and batch_images[idx] is not None:
                            filename = batch_images[idx]
                            writer.writerow([filename, item.Score, item.Reasoning])

                f.flush()
                os.fsync(f.fileno())

                retry_api(client.files.delete, name=uploaded_grid.name)

            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                pass

            progress_queue.put(1)

    return worker_id

# ── 4. MAIN ORCHESTRATOR ───────────────────────────────────────────────────
if __name__ == '__main__':
    api_key = os.environ.get("JB_API_KEY")
    if not api_key: raise ValueError("JB_API_KEY not found.")

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

    all_images = [img for img in os.listdir(IMAGE_DIR) if img.endswith(('.png', '.jpg'))]
    all_images = all_images[3000:4000]  # Subsetting for testing

    # Create the master randomized pool; every image appears in 10 random batches
    master_pool = all_images * ROUNDS_PER_IMAGE
    random.seed(42)
    random.shuffle(master_pool)

    # Group into batches of 9
    all_batches = []
    for i in range(0, len(master_pool), BATCH_SIZE):
        batch = master_pool[i : i + BATCH_SIZE]
        while len(batch) < BATCH_SIZE:
            batch.append(None)
        all_batches.append(batch)

    print(f"Total Images:      {len(all_images)}")
    print(f"Total Evaluations: {len(master_pool)}")
    print(f"Total 3x3 Grids:   {len(all_batches)}")

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
        async_result.get()

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
                    # row: Filename, Score, Reasoning
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

    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
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

    print(f"Done! Results saved to {RESULTS_FILE}")