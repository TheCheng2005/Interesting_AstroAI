"""
For every matched object with at least one paper, search ADS's full-text
index (not just abstracts) for a literal mention of the object's name or any
of its catalog aliases. This catches the common case where a paper's
abstract never spells out an individual object's designation but the body
text (methods, tables, figure captions) does - which abstract-only search
(classify_genuine_discussion.py) systematically misses.

This is server-side keyword/phrase search against ADS's full-text index
(https://ui.adsabs.harvard.edu, 'full:' field), not embedding-based semantic
search - we never fetch or store whole papers (most are paywalled). We do ask
ADS for highlighted snippets: short verbatim excerpts of the body text around
each mention, which ADS caps at 4 per paper x 100 characters to prevent bulk
scraping of its holdings. That is up to ~4kB of real in-body context per
object, and it is what lets the classifier tell "one of 400 rows in Table 3"
apart from "we model its tidal tail in Section 4" - a distinction abstracts
alone almost never support.

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
    MATCHED_CSV,
    SIMBAD_BIBLIOGRAPHY_CSV,
    NED_BIBLIOGRAPHY_CSV,
    load_bibliography,
    load_matched_objects,
    load_ads_api_key,
)


# The stage folders are siblings, so put the analysis root on the path to
# reach common/paths.py (see its docstring).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import FULLTEXT_HITS_CSV, FULLTEXT_CHECKPOINT_DONE

# -- CONFIGURATION ----------------------------------------------------------

REQUEST_BUDGET_PER_RUN = int(os.environ.get("FULLTEXT_REQUEST_BUDGET", "5000"))
                               # NOTE: the ADS quota is a hard 5000/day, so the default
                               # budget consumes all of it and leaves nothing for the
                               # abstract fetching in reclassify_with_fulltext.py - run
                               # that on a different day, or lower this back to ~4500.
ROWS_PER_QUERY = 10  # max full-text hits to keep per object

# Optional scope restriction. The default sweep covers every matched object with
# a paper (18,714 of them, i.e. ~4 days of ADS quota). Setting FULLTEXT_MIN_SCORE
# restricts the run to objects whose best image scored at least that much, and
# searches them highest-score-first, so a partial run still covers the objects a
# follow-up programme would actually target.
MIN_AVG_SCORE = float(os.environ.get("FULLTEXT_MIN_SCORE", "0"))

# Full-text snippets, via ADS's Solr highlighting. Both caps below are ADS
# maximums; asking for more is silently clamped.
HL_SNIPPETS = 4
HL_FRAGSIZE = 100  # characters
# ADS only highlights within the first 51,200 characters of the body by default.
# An object named solely in a table or appendix late in a long paper would then
# return a full-text hit with zero snippets - precisely the catalog-filler case
# this script exists to detect - so widen the analysed window.
HL_MAX_ANALYZED_CHARS = 500000

# Objects searched before snippets existed have context-free hit rows. Re-queue
# them so every hit carries context (see purge_snippetless_hits).
RESEARCH_MISSING_SNIPPETS = True

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
FIELDNAMES = ["object_name", "bibcode", "title", "year", "matched_alias", "snippets"]

_UNSAFE_CHARS_RE = re.compile(r'["\\]')
_EM_TAG_RE = re.compile(r"</?em>")
SNIPPET_SEPARATOR = " ... "
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
                params={
                    "q": query,
                    # 'id' is Solr's uniqueKey, and the highlighting map is keyed
                    # by it rather than by bibcode - without it snippets cannot be
                    # attributed back to a paper.
                    "fl": "id,bibcode,title,year",
                    "rows": ROWS_PER_QUERY,
                    "hl": "true",
                    "hl.fl": "body",
                    "hl.snippets": HL_SNIPPETS,
                    "hl.fragsize": HL_FRAGSIZE,
                    "hl.maxAnalyzedChars": HL_MAX_ANALYZED_CHARS,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if resp.status_code == 429:
                raise RuntimeError("ADS rate limit hit (429)")
            resp.raise_for_status()
            payload = resp.json()
            docs = payload["response"]["docs"]
            return docs, payload.get("highlighting", {}), remaining
        except Exception as e:
            if attempt == len(delays) - 1:
                print(f"  full-text query failed after retries: {e}")
                return [], {}, None
            time.sleep(delay)


def extract_snippets(doc, highlighting):
    """Body snippets for one document, as a single readable string.

    ADS wraps the matched phrase in <em> tags; strip them so the stored text
    stays legible when hand-checking verdicts. Returns "" when the paper was a
    full-text hit but no snippet came back - see HL_MAX_ANALYZED_CHARS.
    """
    entry = highlighting.get(doc.get("id")) or {}
    fragments = entry.get("body", []) if isinstance(entry, dict) else []
    cleaned = [_EM_TAG_RE.sub("", f).strip() for f in fragments]
    return SNIPPET_SEPARATOR.join(c for c in cleaned if c)


def purge_snippetless_hits(already_checked):
    """Re-queue objects searched before snippets were collected.

    Their rows carry no context, so drop those rows and remove the objects from
    the checkpoint; the main loop then re-searches them (highest paper-count
    first) and stores snippets this time. Objects that returned no hits at all
    never appear in the CSV and are deliberately left checked - re-searching
    them would spend quota to learn the same nothing.
    """
    if not os.path.exists(FULLTEXT_HITS_CSV):
        return already_checked

    with open(FULLTEXT_HITS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    seen, with_snippets = set(), set()
    for r in rows:
        name = r.get("object_name")
        if not name:
            continue
        seen.add(name)
        if (r.get("snippets") or "").strip():
            with_snippets.add(name)

    stale = seen - with_snippets
    if not stale:
        return already_checked

    kept = [r for r in rows if r.get("object_name") not in stale]
    with open(FULLTEXT_HITS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    remaining = already_checked - stale
    with open(FULLTEXT_CHECKPOINT_DONE, "w", encoding="utf-8") as f:
        for name in sorted(remaining):
            f.write(name + "\n")

    print(
        f"Re-queued {len(stale)} objects searched before the snippet upgrade "
        f"({len(rows) - len(kept)} context-free hit rows dropped)."
    )
    return remaining


if __name__ == "__main__":
    ads_api_key = load_ads_api_key()

    print("Loading bibliographies and matched objects...")
    simbad_biblio = load_bibliography(SIMBAD_BIBLIOGRAPHY_CSV)
    ned_biblio = load_bibliography(NED_BIBLIOGRAPHY_CSV)
    best_by_object = load_matched_objects(MATCHED_CSV, simbad_biblio, ned_biblio)

    with_papers = {name: o for name, o in best_by_object.items() if o["papers"]}
    print(f"{len(with_papers)} unique matched objects have at least one paper.")

    if MIN_AVG_SCORE > 0:
        with_papers = {
            name: o for name, o in with_papers.items()
            if o["avg_score"] >= MIN_AVG_SCORE
        }
        print(f"{len(with_papers)} of them score >= {MIN_AVG_SCORE:g} (FULLTEXT_MIN_SCORE).")

    already_checked = set()
    if os.path.exists(FULLTEXT_CHECKPOINT_DONE):
        with open(FULLTEXT_CHECKPOINT_DONE, encoding="utf-8") as f:
            already_checked = {line.strip() for line in f if line.strip()}
    print(f"{len(already_checked)} objects already full-text searched in a previous run.")

    if RESEARCH_MISSING_SNIPPETS:
        already_checked = purge_snippetless_hits(already_checked)

    order_key = (
        (lambda n: with_papers[n]["avg_score"]) if MIN_AVG_SCORE > 0
        else (lambda n: len(with_papers[n]["papers"]))
    )
    to_search = sorted(
        (name for name in with_papers if name not in already_checked),
        key=order_key,
        reverse=True,
    )
    print(
        f"{len(to_search)} objects remain to search, processing "
        f"{'highest-score' if MIN_AVG_SCORE > 0 else 'highest paper-count'} first "
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

        docs, highlighting, remaining = search_fulltext(query, ads_api_key)
        n_requests += 1

        new_hits = [d for d in docs if d["bibcode"] not in known_bibcodes]
        if new_hits:
            n_with_new_hits += 1
        for d in docs:
            matched_alias = next((a for a in aliases), name)  # ADS doesn't tell us which phrase hit
            hits_writer.writerow([
                name, d["bibcode"], d.get("title", [""])[0], d.get("year", ""),
                matched_alias, extract_snippets(d, highlighting),
            ])

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
