"""
For a set of literature-matched objects, fetch the abstracts of their most
relevant papers from ADS and ask Gemini to synthesize what the literature
says about the object and why it's scientifically notable. This lets us
show, for objects our pipeline already matched to a known SIMBAD/NED source,
a concrete "here's what's known about this and here's why it's interesting"
writeup instead of just a bare name.

Two object sets are combined (deduped by object_name, so an object picked by
both counts once):
  - top TOP_N_BY_REFCOUNT objects by paper count (most heavily studied)
  - top TOP_N_BY_SCORE objects by our own avg_score (whatever our model
    itself found most interesting), restricted to objects that actually have
    at least one paper - otherwise there's no literature to synthesize.

Results are cached per-object_name across runs (deep_dive_summaries.json is
loaded first and any object already present is kept as-is, not
re-summarized or re-billed) - so re-running this script after the object set
changes only pays for genuinely new objects, and old summaries are always
preserved.

Pipeline:
1. Load matched_objects.csv + simbad_bibliography.csv + ned_bibliography.csv.
2. Group matches by object_name (several images can resolve to the same
   object); pick the highest-scoring image as each object's representative.
3. Effective ref_count = number of papers actually found in the bibliography
   join (falls back to the CSV's ref_count if no bibliography exists, e.g.
   objects matched before this script's bibliography stage existed).
4. Rank objects by effective ref_count and separately by avg_score; combine
   the two top-N lists, deduped by object_name.
5. Skip any object_name already present in the existing deep_dive_summaries
   output (reuse its cached summary instead of recomputing).
6. For each new object, keep its PAPERS_PER_OBJECT most recent papers
   (bounds token cost - some objects have 100+ papers).
7. Batch-fetch abstracts for every needed bibcode from the ADS API
   (api.adsabs.harvard.edu), caching to ads_abstract_cache.json so repeat
   runs don't re-fetch.
8. One Gemini call per new object: feed title/year/abstract of its capped
   paper set + our metadata (object type, redshift, our avg score), ask for
   a literature summary, why it's notable, and what likely made it visually
   interesting.
9. Append the new results to the existing deep_dive_summaries.csv/json
   (old entries untouched).

Requires an ADS API token at ~/.ads_api_key (free, from
https://ui.adsabs.harvard.edu -> Account Settings -> API Token) and
JB_API_KEY in the environment (Gemini).
"""

import os
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ── CONFIGURATION ───────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MATCHED_CSV = os.path.join(SCRIPT_DIR, "matched_objects.csv")
SIMBAD_BIBLIOGRAPHY_CSV = os.path.join(SCRIPT_DIR, "simbad_bibliography.csv")
NED_BIBLIOGRAPHY_CSV = os.path.join(SCRIPT_DIR, "ned_bibliography.csv")

ADS_API_KEY_PATH = os.path.expanduser("~/.ads_api_key")
ADS_ABSTRACT_CACHE_PATH = os.path.join(SCRIPT_DIR, "ads_abstract_cache.json")
ADS_BATCH_SIZE = 50  # bibcodes per ADS query

TOP_N_BY_REFCOUNT = 100  # most heavily-studied matched objects
TOP_N_BY_SCORE = 100  # objects behind our own top-scoring images (that have >=1 paper)
PAPERS_PER_OBJECT = 8  # most recent papers per object fed to the LLM

DEEP_DIVE_CSV = os.path.join(SCRIPT_DIR, "deep_dive_summaries.csv")
DEEP_DIVE_JSON = os.path.join(SCRIPT_DIR, "deep_dive_summaries.json")

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_WORKERS = 5


# ── 1. LOAD + RANK OBJECTS ───────────────────────────────────────────────────

def load_bibliography(path):
    """
    Load a bibliography CSV into filename -> list of {bibcode, year, title,
    alias}. `alias` is the specific catalog name the paper join matched on
    (SIMBAD's main_id, or NED's object_name) - for a given filename this can
    include more than one distinct alias, since the 3" match radius can pick
    up multiple nearby SIMBAD/NED entries (e.g. an optical name and a radio
    counterpart name for the same physical source).
    """
    by_filename = {}
    if not os.path.exists(path):
        return by_filename

    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_filename.setdefault(row["filename"], []).append({
                "bibcode": row["bibcode"],
                "year": row["year"],
                "title": row["title"],
                "alias": row.get("main_id") or row.get("object_name") or "",
            })
    return by_filename


def load_matched_objects(matched_csv, simbad_biblio_by_filename, ned_biblio_by_filename):
    """
    Group matched_objects.csv rows by object_name, pick the highest-scoring
    representative image per object, and compute effective ref_count from
    the bibliography join. Returns a dict object_name -> object info.
    """
    best_by_object = {}

    with open(matched_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            object_name = row["object_name"]
            if not object_name:
                continue

            filename = row["filename"]
            papers = simbad_biblio_by_filename.get(filename) or ned_biblio_by_filename.get(filename) or []

            csv_ref_count = int(row["ref_count"]) if row["ref_count"] not in ("", None) else 0
            effective_ref_count = len(papers) if papers else csv_ref_count

            candidate = {
                "filename": filename,
                "object_name": object_name,
                "object_type": row["object_type"],
                "matched_source": row["matched_source"],
                "redshift": row["redshift"],
                "avg_score": float(row["avg_score"]),
                "ref_count": effective_ref_count,
                "papers": papers,
            }

            existing = best_by_object.get(object_name)
            if existing is None or candidate["avg_score"] > existing["avg_score"]:
                best_by_object[object_name] = candidate

    return best_by_object


def select_top(best_by_object, sort_key, top_n, require_papers=True):
    """
    Sort best_by_object.values() by sort_key descending and take the top_n.
    If require_papers, objects with no papers found are excluded first
    (there'd be nothing to synthesize a deep dive from).
    """
    candidates = best_by_object.values()
    if require_papers:
        candidates = [r for r in candidates if r["papers"]]

    ranked = sorted(candidates, key=lambda r: r[sort_key], reverse=True)
    return ranked[:top_n]


def attach_papers_for_llm(objects, papers_per_object=PAPERS_PER_OBJECT):
    for obj in objects:
        papers_sorted = sorted(
            obj["papers"],
            key=lambda p: p["year"] if str(p["year"]).isdigit() else "0",
            reverse=True,
        )
        obj["papers_for_llm"] = papers_sorted[:papers_per_object]


# ── 2. ADS ABSTRACT FETCH ────────────────────────────────────────────────────

def load_ads_api_key():
    if not os.path.exists(ADS_API_KEY_PATH):
        raise FileNotFoundError(
            f"ADS API key not found at {ADS_API_KEY_PATH}. "
            "Get a free token at https://ui.adsabs.harvard.edu (Account Settings -> API Token)."
        )
    return open(ADS_API_KEY_PATH, "r", encoding="utf-8").read().strip()


def load_abstract_cache():
    if os.path.exists(ADS_ABSTRACT_CACHE_PATH):
        with open(ADS_ABSTRACT_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_abstract_cache(cache):
    with open(ADS_ABSTRACT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def fetch_abstracts(bibcodes, api_key, cache, delays=(1, 2, 4, 8, 16)):
    """
    Fetch {bibcode: {title, abstract, year}} for every bibcode not already
    in cache, batching ADS_BATCH_SIZE bibcodes per query. Updates cache
    in-place and returns it.
    """
    needed = [bc for bc in set(bibcodes) if bc and bc not in cache]
    if not needed:
        return cache

    print(f"Fetching {len(needed)} new abstracts from ADS (batches of {ADS_BATCH_SIZE})...")

    for i in range(0, len(needed), ADS_BATCH_SIZE):
        batch = needed[i:i + ADS_BATCH_SIZE]
        query = "bibcode:(" + " OR ".join(batch) + ")"

        for attempt, delay in enumerate(delays):
            try:
                resp = requests.get(
                    "https://api.adsabs.harvard.edu/v1/search/query",
                    params={"q": query, "fl": "bibcode,title,abstract,year", "rows": len(batch)},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                resp.raise_for_status()
                docs = resp.json()["response"]["docs"]
                break
            except Exception as e:
                if attempt == len(delays) - 1:
                    print(f"  ADS batch {i}-{i + len(batch)} failed after retries: {e}")
                    docs = []
                else:
                    time.sleep(delay)

        for doc in docs:
            cache[doc["bibcode"]] = {
                "title": (doc.get("title") or [""])[0],
                "abstract": doc.get("abstract", "") or "",
                "year": doc.get("year", ""),
            }
        # bibcodes ADS didn't return (no abstract on record) - cache as empty
        # so we don't keep re-requesting them.
        found = {doc["bibcode"] for doc in docs}
        for bc in batch:
            if bc not in found:
                cache[bc] = {"title": "", "abstract": "", "year": ""}

        print(f"  ADS batch {i}-{i + len(batch)}: {len(docs)}/{len(batch)} abstracts found")

    save_abstract_cache(cache)
    return cache


# ── 3. LLM SYNTHESIS ─────────────────────────────────────────────────────────

class DeepDiveSummary(BaseModel):
    literature_summary: str = Field(
        description=(
            "3-5 sentence summary of what the literature says about this object: "
            "what it is, notable properties, and what it's been studied for."
        )
    )
    why_notable: str = Field(
        description="1-3 sentences on specifically what makes this object scientifically notable or unusual."
    )
    likely_visual_reason: str = Field(
        description=(
            "1-2 sentences speculating on what about this object's *appearance* in a Hubble "
            "image likely caught an image-scoring model's attention, based on the literature "
            "(e.g. a visible jet, tidal tails, an unusual morphology, a bright arc/lens). "
            "If the literature gives no clue to visual appearance, say so plainly."
        )
    )


deep_dive_config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer. You will be given an astronomical object's name, "
        "catalog type, redshift (if known), and a set of paper titles/abstracts that "
        "mention it. Synthesize what is known about the object from these excerpts. "
        "Do not invent facts not supported by the abstracts. If the abstracts are sparse "
        "or uninformative, say so rather than speculating."
    ),
    response_mime_type="application/json",
    response_schema=DeepDiveSummary,
    temperature=0.2,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
)


def build_prompt(obj):
    lines = [
        f"Object: {obj['object_name']}",
        f"Catalog type: {obj['object_type'] or 'unknown'}",
        f"Redshift: {obj['redshift'] or 'unknown'}",
        f"Matched via: {obj['matched_source']}",
        f"Our model's average interest score for this image (1-5 scale): {obj['avg_score']:.2f}",
        "",
        f"Papers ({len(obj['papers_for_llm'])} of {obj['ref_count']} total found):",
    ]
    for p in obj["papers_for_llm"]:
        abstract = ""
        if p.get("_abstract"):
            abstract = p["_abstract"][:1500]
        lines.append(f"\n[{p['year']}] {p.get('_title') or p['title']}\n{abstract}")

    return "\n".join(lines)


def summarize_one(obj, api_key, delays=(1, 2, 4, 8, 16)):
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(obj)

    for attempt, delay in enumerate(delays):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=deep_dive_config,
            )
            data = response.parsed
            if data is None:
                raise ValueError("empty response.parsed")
            return obj["object_name"], data, None
        except Exception as e:
            if attempt == len(delays) - 1:
                return obj["object_name"], None, str(e)
            time.sleep(delay)


# ── 4. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gemini_api_key = os.environ.get("JB_API_KEY")
    if not gemini_api_key:
        raise SystemExit("JB_API_KEY not set in environment.")
    ads_api_key = load_ads_api_key()

    print("Loading bibliographies...")
    simbad_biblio = load_bibliography(SIMBAD_BIBLIOGRAPHY_CSV)
    ned_biblio = load_bibliography(NED_BIBLIOGRAPHY_CSV)
    print(f"SIMBAD bibliography covers {len(simbad_biblio)} images. NED bibliography covers {len(ned_biblio)} images.")

    print("\nLoading and ranking matched objects...")
    best_by_object = load_matched_objects(MATCHED_CSV, simbad_biblio, ned_biblio)

    top_by_refcount = select_top(best_by_object, "ref_count", TOP_N_BY_REFCOUNT)
    top_by_score = select_top(best_by_object, "avg_score", TOP_N_BY_SCORE)

    combined_by_object = {}
    for obj in top_by_refcount + top_by_score:
        combined_by_object.setdefault(obj["object_name"], obj)

    print(
        f"Top {len(top_by_refcount)} by ref count + top {len(top_by_score)} by avg score "
        f"= {len(combined_by_object)} unique objects "
        f"({len(top_by_refcount) + len(top_by_score) - len(combined_by_object)} overlap)."
    )

    # Existing output is our cache: any object_name already summarized is
    # kept as-is (not re-billed, not overwritten).
    existing_rows = []
    if os.path.exists(DEEP_DIVE_JSON):
        with open(DEEP_DIVE_JSON, encoding="utf-8") as f:
            existing_rows = json.load(f)
    already_done = {r["object_name"] for r in existing_rows}

    new_objects = [obj for name, obj in combined_by_object.items() if name not in already_done]
    print(f"{len(already_done)} objects already have a cached summary; {len(new_objects)} are new.")

    if not new_objects:
        print("Nothing new to summarize. Existing deep_dive_summaries.csv/json are already up to date.")
        raise SystemExit(0)

    attach_papers_for_llm(new_objects)

    all_bibcodes = [p["bibcode"] for obj in new_objects for p in obj["papers_for_llm"]]
    print(f"\nNeed abstracts for {len(all_bibcodes)} (object, paper) pairs ({len(set(all_bibcodes))} unique bibcodes)...")

    cache = load_abstract_cache()
    cache = fetch_abstracts(all_bibcodes, ads_api_key, cache)

    for obj in new_objects:
        for p in obj["papers_for_llm"]:
            entry = cache.get(p["bibcode"], {})
            p["_title"] = entry.get("title") or p["title"]
            p["_abstract"] = entry.get("abstract", "")

    print(f"\nSynthesizing literature summaries for {len(new_objects)} new objects with Gemini ({GEMINI_MODEL})...")

    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as executor:
        futures = [executor.submit(summarize_one, obj, gemini_api_key) for obj in new_objects]

        for n_done, future in enumerate(as_completed(futures), start=1):
            object_name, data, error = future.result()
            if error:
                print(f"  FAILED {object_name}: {error}")
            else:
                results[object_name] = data
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(new_objects)} done, {time.time() - t0:.0f}s elapsed")

    print(f"Done in {time.time() - t0:.0f}s. {len(results)}/{len(new_objects)} succeeded.")

    new_rows = []
    for obj in new_objects:
        summary = results.get(obj["object_name"])
        if summary is None:
            continue  # failed after retries - leave it out rather than caching an empty summary
        new_rows.append({
            "object_name": obj["object_name"],
            "filename": obj["filename"],
            "object_type": obj["object_type"],
            "redshift": obj["redshift"],
            "matched_source": obj["matched_source"],
            "avg_score": obj["avg_score"],
            "ref_count": obj["ref_count"],
            "papers_used": ";".join(p["bibcode"] for p in obj["papers_for_llm"]),
            "literature_summary": summary.literature_summary,
            "why_notable": summary.why_notable,
            "likely_visual_reason": summary.likely_visual_reason,
        })

    all_rows = existing_rows + new_rows

    with open(DEEP_DIVE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(DEEP_DIVE_JSON, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print(
        f"Wrote {len(all_rows)} total deep-dive summaries "
        f"({len(existing_rows)} pre-existing + {len(new_rows)} new) to {DEEP_DIVE_CSV} and {DEEP_DIVE_JSON}"
    )
