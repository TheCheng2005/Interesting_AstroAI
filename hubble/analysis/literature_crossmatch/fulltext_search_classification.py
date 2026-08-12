"""
For every matched object with at least one paper, search ADS's full-text
index (not just abstracts) for a literal mention of the object's name or any
of its catalog aliases. This catches the common case where a paper's
abstract never spells out an individual object's designation but the body
text (methods, tables, figure captions) does - which abstract-only search
(classify_genuine_discussion.py) systematically misses.

This is server-side keyword/phrase search against ADS's full-text index
(https://ui.adsabs.harvard.edu, 'full:' field), not embedding-based semantic
search - we never fetch or store the underlying paper text ourselves (most
of it is paywalled), we only get back hit/no-hit + bibliographic metadata.

ADS enforces a hard 5000 requests/day quota on this account, and 18,714
objects need up to one query each, so this cannot complete in one run.
Objects are processed in descending order of paper count (the ones most
likely to have a hidden individual discussion, and most valuable to get
right), with a request budget cap per run and a checkpoint
(fulltext_hits.csv) so subsequent runs pick up where the previous one left
off - across multiple days if needed.

A hit here is strong new evidence, not an automatic verdict: after this
script runs, re-run classify_reclassify_with_fulltext.py to feed
newly-discovered hit papers back through Gemini and update
discussion_classification.csv only for objects whose verdict could plausibly
change.
"""

import os
import re
import csv
import time

import requests

from deep_dive_summaries import (
    SCRIPT_DIR,
    MATCHED_CSV,
    SIMBAD_BIBLIOGRAPHY_CSV,
    NED_BIBLIOGRAPHY_CSV,
    load_bibliography,
    load_matched_objects,
    load_ads_api_key,
)


# ── CONFIGURATION ───────────────────────────────────────────────────────────

REQUEST_BUDGET_PER_RUN = 4500  # stay under the 5000/day ADS quota with a safety margin
ROWS_PER_QUERY = 10  # max full-text hits to keep per object

FULLTEXT_HITS_CSV = os.path.join(SCRIPT_DIR, "fulltext_hits.csv")
FULLTEXT_CHECKPOINT_DONE = os.path.join(SCRIPT_DIR, "fulltext_search_checked.txt")

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
FIELDNAMES = ["object_name", "bibcode", "title", "year", "matched_alias"]

_UNSAFE_CHARS_RE = re.compile(r'["\\]')
_COORD_DESIGNATION_RE = re.compile(r"J(\d{4,10}(?:\.\d+)?)([+\-])(\d{2,9}(?:\.\d+)?)")
_CATALOG_PREFIX_RE = re.compile(
    r"^(NAME|SLACS|SDSS|FIRST|WISEA|2MASS|2MASX|COSMOS2015|COSMOS2020|COSMOS-DASH|"
    r"COSMOS|CANDELS|MCPS|OMS2023|LFC2025|NVSS|VLASS|GALEX|PSO|PS1|Gaia DR[23])\s+",
    re.IGNORECASE,
)
_TRAILING_NOISE_RE = re.compile(r"\s+(source|lens|system|component|counterpart|galaxy)$", re.IGNORECASE)


def sanitize_phrase(alias):
    """Strip characters that would break a Solr quoted-phrase query."""
    return _UNSAFE_CHARS_RE.sub("", alias).strip()


def expand_alias_to_phrases(alias):
    """
    A raw SIMBAD/NED alias like 'SLACS SDSS J1205+4910 source' rarely
    appears in paper text verbatim - the literature usually drops the
    survey prefix and any trailing descriptor ('SDSS J1205+4910' or just
    'J1205+4910'), and coordinate designations get truncated to varying
    precision ('J120540.4+491029' vs 'J1205+4910'). Generate a small set of
    plausible phrase variants to OR together in one query (this doesn't cost
    extra requests - only extra OR terms in a single query).
    """
    phrases = set()

    m = _COORD_DESIGNATION_RE.search(alias)
    if m:
        ra, sign, dec = m.groups()
        ra_int = ra.split(".")[0]
        dec_int = dec.split(".")[0]
        phrases.add(f"J{ra}{sign}{dec}")  # full precision as given
        if len(ra_int) >= 4 and len(dec_int) >= 4:
            phrases.add(f"J{ra_int[:4]}{sign}{dec_int[:4]}")  # common Jhhmm+ddmm shorthand
        if len(ra_int) >= 6 and len(dec_int) >= 6:
            phrases.add(f"J{ra_int[:6]}{sign}{dec_int[:6]}")  # Jhhmmss+ddmmss, no decimals

    stripped = _CATALOG_PREFIX_RE.sub("", alias)
    stripped = _TRAILING_NOISE_RE.sub("", stripped).strip()
    if stripped and len(stripped) >= 4:
        phrases.add(stripped)
        # also without a leading catalog-name-only prefix stripped further
        # (e.g. 'SDSS J1205+4910' -> already handled by coord extraction above)

    return phrases


def build_fulltext_query(aliases):
    phrases = set()
    for alias in aliases:
        phrases |= expand_alias_to_phrases(alias)

    clauses = [f'full:"{sanitize_phrase(p)}"' for p in phrases if sanitize_phrase(p)]
    if not clauses:
        return None
    return " OR ".join(clauses)


def search_fulltext(query, api_key, delays=(1, 2, 4, 8)):
    for attempt, delay in enumerate(delays):
        try:
            resp = requests.get(
                ADS_SEARCH_URL,
                params={"q": query, "fl": "bibcode,title,year", "rows": ROWS_PER_QUERY},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if resp.status_code == 429:
                raise RuntimeError("ADS rate limit hit (429)")
            resp.raise_for_status()
            docs = resp.json()["response"]["docs"]
            return docs, remaining
        except Exception as e:
            if attempt == len(delays) - 1:
                print(f"  full-text query failed after retries: {e}")
                return [], None
            time.sleep(delay)


if __name__ == "__main__":
    ads_api_key = load_ads_api_key()

    print("Loading bibliographies and matched objects...")
    simbad_biblio = load_bibliography(SIMBAD_BIBLIOGRAPHY_CSV)
    ned_biblio = load_bibliography(NED_BIBLIOGRAPHY_CSV)
    best_by_object = load_matched_objects(MATCHED_CSV, simbad_biblio, ned_biblio)

    with_papers = {name: o for name, o in best_by_object.items() if o["papers"]}
    print(f"{len(with_papers)} unique matched objects have at least one paper.")

    already_checked = set()
    if os.path.exists(FULLTEXT_CHECKPOINT_DONE):
        with open(FULLTEXT_CHECKPOINT_DONE, encoding="utf-8") as f:
            already_checked = {line.strip() for line in f if line.strip()}
    print(f"{len(already_checked)} objects already full-text searched in a previous run.")

    to_search = sorted(
        (name for name in with_papers if name not in already_checked),
        key=lambda n: len(with_papers[n]["papers"]),
        reverse=True,
    )
    print(
        f"{len(to_search)} objects remain to search, processing highest paper-count first "
        f"(budget this run: {REQUEST_BUDGET_PER_RUN} requests)."
    )

    hits_file_is_new = not os.path.exists(FULLTEXT_HITS_CSV)
    hits_file = open(FULLTEXT_HITS_CSV, "a", newline="", encoding="utf-8")
    hits_writer = csv.writer(hits_file)
    if hits_file_is_new:
        hits_writer.writerow(FIELDNAMES)

    checked_file = open(FULLTEXT_CHECKPOINT_DONE, "a", encoding="utf-8")

    n_requests = 0
    n_with_new_hits = 0
    t0 = time.time()

    for name in to_search:
        if n_requests >= REQUEST_BUDGET_PER_RUN:
            print(f"\nHit this run's request budget ({REQUEST_BUDGET_PER_RUN}). Stopping - re-run this script later to continue.")
            break

        o = with_papers[name]
        known_bibcodes = {p["bibcode"] for p in o["papers"]}
        aliases = {name} | {p["alias"] for p in o["papers"] if p.get("alias")}

        query = build_fulltext_query(aliases)
        if query is None:
            checked_file.write(name + "\n")
            continue

        docs, remaining = search_fulltext(query, ads_api_key)
        n_requests += 1

        new_hits = [d for d in docs if d["bibcode"] not in known_bibcodes]
        if new_hits:
            n_with_new_hits += 1
        for d in docs:
            matched_alias = next((a for a in aliases), name)  # ADS doesn't tell us which phrase hit
            hits_writer.writerow([name, d["bibcode"], d.get("title", [""])[0], d.get("year", ""), matched_alias])

        checked_file.write(name + "\n")

        if n_requests % 100 == 0:
            hits_file.flush()
            checked_file.flush()
            elapsed = time.time() - t0
            print(
                f"  {n_requests}/{min(len(to_search), REQUEST_BUDGET_PER_RUN)} searched, "
                f"{n_with_new_hits} objects found a new (previously-unknown) paper, "
                f"{elapsed:.0f}s elapsed, ADS remaining today: {remaining}"
            )

    hits_file.close()
    checked_file.close()

    n_left = len(to_search) - n_requests
    print(f"\nDone this run: {n_requests} objects searched, {n_with_new_hits} found a new paper.")
    if n_left > 0:
        print(f"{n_left} objects still unsearched - re-run this script (tomorrow, once the ADS quota resets) to continue.")
    else:
        print("All objects with papers have now been full-text searched.")
