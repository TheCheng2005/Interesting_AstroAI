"""
Re-run the Gemini genuine-discussion classification over the stage-2
cross-match output, working down the score ranking.

This script used to be driven by fulltext_hits.csv: it could only revisit
objects the (quota-bound, multi-day) full-text sweep had already reached,
and that sweep walks the catalog in descending paper-count order. The
objects we actually care about - the highest-scoring images, the ones a
follow-up programme would target - sit anywhere in that order, so the most
interesting verdicts were the last to be revised.

It now reads matched_objects.csv (find_unidentified_objects.py's output)
directly and works down avg_score, highest first:

  - Objects with 0 papers in the bibliography join are skipped entirely.
    There is no literature to read, so classify_genuine_discussion.py's
    automatic genuinely_discussed=False verdict already stands and no LLM
    call can change it.
  - Objects already classified genuinely_discussed=True are skipped. New
    evidence can only flip False -> True, so re-billing them buys nothing.
  - Everything else with at least one paper is a candidate, whether or not
    the standalone sweep has reached it, and whether or not it has ever
    been classified before.
  - For any candidate the sweep has not searched, the ADS full-text query
    is run inline here. Results are appended to fulltext_hits.csv and the
    object is added to fulltext_search_checked.txt, so the standalone sweep
    will not re-spend quota on it later.

Evidence fed to Gemini per object: every paper from the bibliography join,
merged with any newly-discovered full-text papers, with verbatim in-body
snippets attached where ADS returned them. Papers are flagged as hits when
the object's name/alias appears in the abstract (find_name_hits) or in the
body (the full-text search), then capped by the same
select_papers_for_classification budget classify_genuine_discussion.py
uses, so both scripts show the model a comparable evidence set.

Budgets (both env-overridable, both bounded by ADS's hard 5000 requests/day
across this account - inline searches and abstract fetches draw on the same
quota):
    RECLASSIFY_MAX_OBJECTS      objects to classify this run (default 1000)
    RECLASSIFY_FULLTEXT_BUDGET  inline ADS full-text searches (default 3000)
    RECLASSIFY_MIN_SCORE        skip objects scoring below this (default 0)

discussion_classification.csv is updated in place for just the objects
processed; every other row is preserved byte-for-byte. Verdicts are flushed
to disk every FLUSH_EVERY completions, so an interrupted run keeps the work
it paid for and the next run resumes further down the ranking.

Rerun generate_unidentified_html_report.py afterward to apply any changed
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
    find_name_hits,
    select_papers_for_classification,
    GEMINI_MODEL,
    GEMINI_WORKERS,
)
from fulltext_search_classification import (
    FULLTEXT_HITS_CSV,
    FULLTEXT_CHECKPOINT_DONE,
    FIELDNAMES as FULLTEXT_FIELDNAMES,
    build_fulltext_query,
    search_fulltext,
    extract_snippets,
)
from deep_dive_summaries import (
    MATCHED_CSV,
    SIMBAD_BIBLIOGRAPHY_CSV,
    NED_BIBLIOGRAPHY_CSV,
    load_bibliography,
    load_matched_objects,
    load_ads_api_key,
    load_abstract_cache,
    fetch_abstracts,
)


# ── CONFIGURATION ──────────────────────────────────────────────────────────

# Objects classified per run, taken from the top of the score ranking. The
# next run picks up below them (already-True objects drop out; objects still
# False are retried only if new evidence arrives, since a False verdict is
# cheap to leave standing).
MAX_OBJECTS = int(os.environ.get("RECLASSIFY_MAX_OBJECTS", "1000"))

# Inline ADS full-text searches for candidates the standalone sweep has not
# reached. One request per object. Kept below the 5000/day account quota so
# the abstract fetching further down still has room.
FULLTEXT_BUDGET = int(os.environ.get("RECLASSIFY_FULLTEXT_BUDGET", "3000"))

# Optional score floor - useful for "only ever look at things above 40".
MIN_AVG_SCORE = float(os.environ.get("RECLASSIFY_MIN_SCORE", "0"))

# Verdicts are written to disk this often (whole-file rewrite, ~28k rows).
FLUSH_EVERY = 200


# ── 1. STATE ON DISK ───────────────────────────────────────────────────────

def load_classifications():
    """object_name -> current discussion_classification.csv row."""
    if not os.path.exists(CLASSIFICATION_CSV):
        return {}
    with open(CLASSIFICATION_CSV, newline="", encoding="utf-8") as f:
        return {row["object_name"]: row for row in csv.DictReader(f)}


def is_discussed(row):
    return str(row.get("genuinely_discussed", "")).strip().lower() == "true"


def write_classifications(row_by_object):
    """
    Rewrite discussion_classification.csv from the in-memory table. Written
    to a temporary file and moved into place, so an interruption mid-flush
    cannot truncate a classification set that cost real API spend.
    """
    tmp_path = CLASSIFICATION_CSV + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLASSIFICATION_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row_by_object.values())
    os.replace(tmp_path, CLASSIFICATION_CSV)


def load_fulltext_hits():
    """object_name -> list of fulltext_hits.csv rows."""
    hits_by_object = {}
    if not os.path.exists(FULLTEXT_HITS_CSV):
        return hits_by_object
    with open(FULLTEXT_HITS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hits_by_object.setdefault(row["object_name"], []).append(row)
    return hits_by_object


def load_fulltext_checked():
    """Object names the full-text sweep has already spent a request on."""
    if not os.path.exists(FULLTEXT_CHECKPOINT_DONE):
        return set()
    with open(FULLTEXT_CHECKPOINT_DONE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


# ── 2. CANDIDATE SELECTION ─────────────────────────────────────────────────

def select_candidates(with_papers, row_by_object):
    """
    Objects worth spending an LLM call on this run, best-scoring first.

    Skips objects already judged genuinely discussed; keeps objects that are
    currently False and objects that have never been classified at all (the
    latter happens when new matches land in matched_objects.csv after the
    last classify_genuine_discussion.py pass).
    """
    names = [
        name for name in with_papers
        if not (name in row_by_object and is_discussed(row_by_object[name]))
    ]

    if MIN_AVG_SCORE > 0:
        names = [n for n in names if with_papers[n]["avg_score"] >= MIN_AVG_SCORE]

    # Highest score first; ties broken by paper count, then name, so a run is
    # reproducible and consecutive runs walk the ranking in a stable order.
    names.sort(key=lambda n: (-with_papers[n]["avg_score"], -len(with_papers[n]["papers"]), n))

    if MAX_OBJECTS > 0:
        names = names[:MAX_OBJECTS]
    return names


# ── 3. INLINE FULL-TEXT SEARCH ─────────────────────────────────────────────

def search_missing_fulltext(names, with_papers, already_checked, ads_api_key, budget):
    """
    Run the ADS full-text query for candidates the standalone sweep has not
    covered, in the order given (i.e. best-scoring first).

    Hits are appended to fulltext_hits.csv and every searched object is
    appended to fulltext_search_checked.txt as it completes, so this work is
    shared with fulltext_search_classification.py rather than duplicated by
    it - and so an interrupted run does not re-spend quota on the objects it
    already searched.

    Returns object_name -> list of newly-found hit rows.
    """
    to_search = [n for n in names if n not in already_checked]
    if not to_search:
        print("Every candidate has already been full-text searched.")
        return {}

    n_planned = min(len(to_search), budget)
    print(
        f"{len(to_search)} candidates have never been full-text searched; "
        f"searching {n_planned} of them now (budget {budget})."
    )
    if n_planned < len(to_search):
        print(
            f"  {len(to_search) - n_planned} will be classified on abstracts alone "
            f"this run and searched by a later run."
        )

    new_hits = {}
    hits_file_is_new = not os.path.exists(FULLTEXT_HITS_CSV)
    hits_file = open(FULLTEXT_HITS_CSV, "a", newline="", encoding="utf-8")
    hits_writer = csv.writer(hits_file)
    if hits_file_is_new:
        hits_writer.writerow(FULLTEXT_FIELDNAMES)
    checked_file = open(FULLTEXT_CHECKPOINT_DONE, "a", encoding="utf-8")

    n_requests = 0
    n_with_new_papers = 0
    remaining = None
    t0 = time.time()
    try:
        for name in to_search:
            if n_requests >= budget:
                break

            o = with_papers[name]
            known_bibcodes = {p["bibcode"] for p in o["papers"]}
            aliases = {name} | {p["alias"] for p in o["papers"] if p.get("alias")}

            query = build_fulltext_query(aliases)
            if query is None:
                # No usable phrase for this designation - record it as checked
                # so no later run pays to rediscover that.
                checked_file.write(name + "\n")
                continue

            docs, highlighting, remaining = search_fulltext(query, ads_api_key)
            n_requests += 1

            rows = []
            for d in docs:
                matched_alias = next((a for a in aliases), name)  # ADS doesn't say which phrase hit
                row = {
                    "object_name": name,
                    "bibcode": d["bibcode"],
                    "title": d.get("title", [""])[0],
                    "year": d.get("year", ""),
                    "matched_alias": matched_alias,
                    "snippets": extract_snippets(d, highlighting),
                }
                rows.append(row)
                hits_writer.writerow([row[k] for k in FULLTEXT_FIELDNAMES])

            if rows:
                new_hits[name] = rows
                if any(r["bibcode"] not in known_bibcodes for r in rows):
                    n_with_new_papers += 1

            checked_file.write(name + "\n")

            if n_requests % 100 == 0:
                hits_file.flush()
                checked_file.flush()
                print(
                    f"  {n_requests}/{n_planned} searched, {len(new_hits)} with hits, "
                    f"{time.time() - t0:.0f}s elapsed, ADS remaining today: {remaining}"
                )
    finally:
        hits_file.close()
        checked_file.close()

    print(
        f"Inline full-text search: {n_requests} ADS requests, "
        f"{len(new_hits)} objects returned at least one paper "
        f"({n_with_new_papers} of them a paper the catalog join did not know about)."
    )
    return new_hits


# ── 4. EVIDENCE ASSEMBLY ───────────────────────────────────────────────────

def merge_fulltext_evidence(o, hit_rows):
    """
    Fold an object's full-text hit rows into its bibliography papers.

    Papers ADS found in the body but the catalog join never knew about are
    appended; snippets are attached per bibcode. Sets, on the object:
      merged_papers          bibliography + newly-discovered papers
      fulltext_bibcodes      bibcodes that hit in the body
      snippets_by_bibcode    verbatim in-body context, where ADS gave any
    """
    known_bibcodes = {p["bibcode"] for p in o["papers"]}
    merged_papers = list(o["papers"])

    snippets_by_bibcode = {}
    fulltext_bibcodes = set()

    for r in hit_rows:
        bibcode = r["bibcode"]
        fulltext_bibcodes.add(bibcode)
        if (r.get("snippets") or "").strip():
            snippets_by_bibcode[bibcode] = r["snippets"]
        if bibcode not in known_bibcodes:
            merged_papers.append({
                "bibcode": bibcode,
                "year": r.get("year", ""),
                "title": r.get("title", ""),
                "alias": r.get("matched_alias", ""),
            })
            known_bibcodes.add(bibcode)

    o["merged_papers"] = merged_papers
    o["fulltext_bibcodes"] = fulltext_bibcodes
    o["snippets_by_bibcode"] = snippets_by_bibcode


# ── 5. MAIN ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gemini_api_key = os.environ.get("JB_API_KEY")
    if not gemini_api_key:
        raise SystemExit("JB_API_KEY not set in environment.")
    ads_api_key = load_ads_api_key()

    print("Loading bibliographies and matched objects...")
    simbad_biblio = load_bibliography(SIMBAD_BIBLIOGRAPHY_CSV)
    ned_biblio = load_bibliography(NED_BIBLIOGRAPHY_CSV)
    best_by_object = load_matched_objects(MATCHED_CSV, simbad_biblio, ned_biblio)
    print(f"{len(best_by_object)} unique matched objects in {os.path.basename(MATCHED_CSV)}.")

    # Objects with no literature at all can never be flipped by more reading.
    with_papers = {name: o for name, o in best_by_object.items() if o["papers"]}
    print(
        f"{len(with_papers)} have at least one paper; "
        f"{len(best_by_object) - len(with_papers)} zero-paper objects skipped."
    )

    row_by_object = load_classifications()
    n_true = sum(1 for r in row_by_object.values() if is_discussed(r))
    print(f"{len(row_by_object)} existing classifications ({n_true} genuinely discussed, left alone).")

    candidates = select_candidates(with_papers, row_by_object)
    if not candidates:
        print("Nothing left to reclassify.")
        raise SystemExit(0)

    scores = [with_papers[n]["avg_score"] for n in candidates]
    print(
        f"\n{len(candidates)} objects selected this run, best-scoring first "
        f"(avg_score {scores[0]:.2f} down to {scores[-1]:.2f})."
    )

    # -- full-text evidence: what the sweep already has, plus inline searches
    print("\nLoading existing full-text hits...")
    hits_by_object = load_fulltext_hits()
    already_checked = load_fulltext_checked()
    print(
        f"{len(hits_by_object)} objects have stored hits; "
        f"{len(already_checked)} have been searched."
    )

    new_hits = search_missing_fulltext(
        candidates, with_papers, already_checked, ads_api_key, FULLTEXT_BUDGET
    )
    for name, rows in new_hits.items():
        hits_by_object.setdefault(name, []).extend(rows)

    for name in candidates:
        merge_fulltext_evidence(with_papers[name], hits_by_object.get(name, []))

    # -- abstracts for every paper we are about to show the model
    all_bibcodes = [p["bibcode"] for name in candidates for p in with_papers[name]["merged_papers"]]
    print(
        f"\nNeed abstracts for {len(all_bibcodes)} (object, paper) pairs "
        f"({len(set(all_bibcodes))} unique bibcodes)..."
    )
    cache = load_abstract_cache()
    cache = fetch_abstracts(all_bibcodes, ads_api_key, cache)

    n_with_snippets = 0
    for name in candidates:
        o = with_papers[name]
        for p in o["merged_papers"]:
            entry = cache.get(p["bibcode"], {})
            p["_title"] = entry.get("title") or p.get("title", "")
            p["_abstract"] = entry.get("abstract", "")
            p["_snippets"] = o["snippets_by_bibcode"].get(p["bibcode"], "")
            if p["_snippets"]:
                n_with_snippets += 1
    print(f"{n_with_snippets} (object, paper) pairs carry verbatim in-body snippets.")

    # A paper counts as a hit if the object's name shows up in its abstract or
    # in its body. Both go to the model as "strong evidence" flags, and both
    # guarantee the paper survives the PAPERS_SOFT_CAP cut.
    n_with_hits = 0
    for name in candidates:
        o = with_papers[name]
        aliases = {name} | {p["alias"] for p in o["merged_papers"] if p.get("alias")}
        hit_bibcodes = find_name_hits(o["merged_papers"], aliases) | o["fulltext_bibcodes"]
        o["hit_bibcodes"] = hit_bibcodes
        o["papers_for_classification"] = select_papers_for_classification(
            o["merged_papers"], hit_bibcodes
        )
        if hit_bibcodes:
            n_with_hits += 1
    print(f"{n_with_hits}/{len(candidates)} objects have a name hit in an abstract or paper body.")

    # -- classify
    print(f"\nReclassifying {len(candidates)} objects with Gemini ({GEMINI_MODEL}, {GEMINI_WORKERS} workers)...")

    n_flipped = 0
    n_failed = 0
    n_done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as executor:
        # Submitted in score order, so an interrupted run has still spent its
        # budget on the highest-ranking objects.
        futures = {
            executor.submit(
                classify_one, name,
                with_papers[name]["papers_for_classification"],
                with_papers[name]["hit_bibcodes"],
                gemini_api_key,
            ): name
            for name in candidates
        }

        for future in as_completed(futures):
            object_name, data, error = future.result()
            n_done += 1

            if error:
                n_failed += 1
                print(f"  FAILED {object_name}: {error}")
            else:
                o = with_papers[object_name]
                previous = row_by_object.get(object_name)
                was_discussed = is_discussed(previous) if previous else False
                if data.genuinely_discussed and not was_discussed:
                    n_flipped += 1
                row_by_object[object_name] = {
                    "object_name": object_name,
                    "n_papers_checked": len(o["papers_for_classification"]),
                    "n_name_hits": len(o["hit_bibcodes"]),
                    "genuinely_discussed": data.genuinely_discussed,
                    "reasoning": data.reasoning,
                }

            if n_done % FLUSH_EVERY == 0:
                write_classifications(row_by_object)
                print(
                    f"  {n_done}/{len(candidates)} done, {n_flipped} flipped so far, "
                    f"{time.time() - t0:.0f}s elapsed (verdicts saved)"
                )

    write_classifications(row_by_object)

    print(f"\nDone in {time.time() - t0:.0f}s. {n_done - n_failed}/{len(candidates)} classified, {n_failed} failed.")
    print(f"Wrote {len(row_by_object)} total classifications to {os.path.basename(CLASSIFICATION_CSV)}.")
    print(f"{n_flipped} objects flipped to 'genuinely discussed' on abstract + full-text evidence.")

    n_remaining = len([
        n for n in with_papers
        if not (n in row_by_object and is_discussed(row_by_object[n]))
    ]) - len(candidates)
    if n_remaining > 0:
        print(f"{n_remaining} lower-ranked objects still unreviewed - re-run to continue down the ranking.")
