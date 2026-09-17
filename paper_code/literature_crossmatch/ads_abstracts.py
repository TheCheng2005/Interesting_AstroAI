"""
Fetch paper abstracts from ADS, with an on-disk cache.

The genuine-discussion classifier judges an object from what the literature
says about it, and an abstract is the cheapest form of that evidence: one
ADS query returns ADS_BATCH_SIZE of them at a time, where fetching the
papers themselves would be impossible (most are paywalled) and pointless
(the full-text search in fulltext_search_classification.py already returns
the in-body sentences around each mention).

Everything fetched is cached to ADS_ABSTRACT_CACHE_PATH and reloaded on the
next run, so re-running the pipeline after new objects appear only pays for
the new bibcodes. The cache is keyed by bibcode and is safe to delete.

Used by unidentified_objects/build_undiscussed_catalog.py. The ADS token is
read from ~/.ads_api_key (ADS_API_KEY_PATH); ADS allows 5,000 requests/day
and the same quota covers the full-text search, so the cache is what keeps a
re-run affordable.
"""

import os
import json
import time

import requests


# The stage folders are siblings, so put the analysis root on the path to
# reach common/paths.py (see its docstring).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import (
    ADS_API_KEY_PATH,
    ADS_ABSTRACT_CACHE_PATH,
)


# ── CONFIGURATION ───────────────────────────────────────────────────────────

ADS_BATCH_SIZE = 50  # bibcodes per ADS query


# ── 1. LOAD + RANK OBJECTS ───────────────────────────────────────────────────


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


# ── 4. MAIN ──────────────────────────────────────────────────────────────────
