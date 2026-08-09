"""
Take the output of fulltext_search_classification.py (fulltext_hits.csv -
objects whose full paper text, not just abstract, contains their name/
designation) and re-run the Gemini genuine-discussion classification for
just those objects, with the full-text hit(s) folded in as strong evidence
and any newly-discovered papers added to the set Gemini sees.

Only objects currently classified genuinely_discussed=False are
reclassified (a full-text hit is only interesting if it might flip a
verdict; already-True objects are left alone to avoid unnecessary spend).
discussion_classification.csv is updated in place for just the affected
objects - everything else is untouched.

Run this after fulltext_search_classification.py. Rerun
generate_unidentified_html_report.py afterward to apply any changed
verdicts to the report.
"""

import os
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from classify_genuine_discussion import (
    CLASSIFICATION_CSV,
    CLASSIFICATION_FIELDNAMES,
    classify_one,
    GEMINI_MODEL,
    GEMINI_WORKERS,
)
from fulltext_search_classification import FULLTEXT_HITS_CSV
from deep_dive_summaries import (
    SCRIPT_DIR,
    MATCHED_CSV,
    SIMBAD_BIBLIOGRAPHY_CSV,
    NED_BIBLIOGRAPHY_CSV,
    load_bibliography,
    load_matched_objects,
    load_ads_api_key,
    load_abstract_cache,
    fetch_abstracts,
)


if __name__ == "__main__":
    gemini_api_key = os.environ.get("JB_API_KEY")
    if not gemini_api_key:
        raise SystemExit("JB_API_KEY not set in environment.")
    ads_api_key = load_ads_api_key()

    if not os.path.exists(FULLTEXT_HITS_CSV):
        raise SystemExit(f"{FULLTEXT_HITS_CSV} not found - run fulltext_search_classification.py first.")

    print("Loading full-text hits...")
    hits_by_object = {}
    with open(FULLTEXT_HITS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hits_by_object.setdefault(row["object_name"], []).append(row)
    print(f"{len(hits_by_object)} objects have at least one full-text search result.")

    print("Loading current classifications...")
    existing_rows = []
    if os.path.exists(CLASSIFICATION_CSV):
        with open(CLASSIFICATION_CSV, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
    row_by_object = {r["object_name"]: r for r in existing_rows}

    candidates = [
        name for name in hits_by_object
        if name in row_by_object and str(row_by_object[name]["genuinely_discussed"]).strip().lower() != "true"
    ]
    print(f"{len(candidates)} currently-'not discussed' objects have a full-text hit and will be reclassified.")

    if not candidates:
        print("Nothing to reclassify.")
        raise SystemExit(0)

    print("\nLoading bibliographies and matched objects...")
    simbad_biblio = load_bibliography(SIMBAD_BIBLIOGRAPHY_CSV)
    ned_biblio = load_bibliography(NED_BIBLIOGRAPHY_CSV)
    best_by_object = load_matched_objects(MATCHED_CSV, simbad_biblio, ned_biblio)

    # Merge each candidate's known bibliography papers with any newly
    # full-text-discovered papers (dedup by bibcode), and mark every bibcode
    # that showed up in the full-text search as a "hit" for the prompt.
    for name in candidates:
        o = best_by_object[name]
        known_bibcodes = {p["bibcode"] for p in o["papers"]}
        hit_rows = hits_by_object[name]
        hit_bibcodes = {r["bibcode"] for r in hit_rows}

        merged_papers = list(o["papers"])
        for r in hit_rows:
            if r["bibcode"] not in known_bibcodes:
                merged_papers.append({
                    "bibcode": r["bibcode"],
                    "year": r["year"],
                    "title": r["title"],
                    "alias": r.get("matched_alias", ""),
                })
                known_bibcodes.add(r["bibcode"])

        o["merged_papers"] = merged_papers
        o["hit_bibcodes"] = hit_bibcodes

    all_bibcodes = [p["bibcode"] for name in candidates for p in best_by_object[name]["merged_papers"]]
    print(f"\nFetching abstracts for {len(all_bibcodes)} (object, paper) pairs ({len(set(all_bibcodes))} unique bibcodes)...")
    cache = load_abstract_cache()
    cache = fetch_abstracts(all_bibcodes, ads_api_key, cache)

    for name in candidates:
        for p in best_by_object[name]["merged_papers"]:
            entry = cache.get(p["bibcode"], {})
            p["_title"] = entry.get("title") or p["title"]
            p["_abstract"] = entry.get("abstract", "")

    print(f"\nReclassifying {len(candidates)} objects with Gemini ({GEMINI_MODEL}, {GEMINI_WORKERS} workers)...")

    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as executor:
        futures = [
            executor.submit(
                classify_one, name,
                best_by_object[name]["merged_papers"],
                best_by_object[name]["hit_bibcodes"],
                gemini_api_key,
            )
            for name in candidates
        ]
        for n_done, future in enumerate(as_completed(futures), start=1):
            object_name, data, error = future.result()
            if error:
                print(f"  FAILED {object_name}: {error}")
            else:
                results[object_name] = data
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(candidates)} done, {time.time() - t0:.0f}s elapsed")

    print(f"Done in {time.time() - t0:.0f}s. {len(results)}/{len(candidates)} succeeded.")

    n_flipped = 0
    for name, data in results.items():
        old_verdict = str(row_by_object[name]["genuinely_discussed"]).strip().lower() == "true"
        new_verdict = data.genuinely_discussed
        if new_verdict and not old_verdict:
            n_flipped += 1
        row_by_object[name] = {
            "object_name": name,
            "n_papers_checked": len(best_by_object[name]["merged_papers"]),
            "n_name_hits": len(best_by_object[name]["hit_bibcodes"]),
            "genuinely_discussed": new_verdict,
            "reasoning": data.reasoning,
        }

    all_rows = list(row_by_object.values())
    with open(CLASSIFICATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLASSIFICATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} total classifications to {CLASSIFICATION_CSV}.")
    print(f"{n_flipped} objects flipped from 'not discussed' to 'genuinely discussed' based on full-text evidence.")
