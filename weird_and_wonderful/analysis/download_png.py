import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_FILE     = 'ww_data/all_ww_data.csv'
OUT_DIR      = 'png_images'
WORKERS      = 32    # bump up — these are just static file downloads
TIMEOUT      = 30    # seconds per request

def download_one(row, out_dir):
    obj_id = int(row['filename'])
    url    = row['URL']
    out    = os.path.join(out_dir, f"{obj_id}.png")

    if os.path.exists(out):
        return 'skipped'

    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    with open(out, 'wb') as f:
        f.write(r.content)
    return 'downloaded'


def main():
    df = pd.read_csv(CSV_FILE)
    os.makedirs(OUT_DIR, exist_ok=True)

    total      = len(df)
    done       = 0
    errors     = 0
    start_time = time.time()

    print(f"Loaded {total:,} objects from {CSV_FILE}")
    print(f"Downloading with {WORKERS} workers → {OUT_DIR}/\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(download_one, row, OUT_DIR): int(row['filename'])
            for _, row in df.iterrows()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                errors += 1
                print(f"  ERROR {futures[future]}: {e}")
            done += 1

            if done % 500 == 0 or done == total:
                elapsed = time.time() - start_time
                rate    = done / elapsed if elapsed > 0 else 0
                eta_min = (total - done) / rate / 60 if rate > 0 else 0
                print(f"  {done:,}/{total:,} | {rate:.1f} img/s | ETA {eta_min:.1f}min | errors: {errors}")

    print(f"\nDone. Errors: {errors:,}")


if __name__ == "__main__":
    main()
