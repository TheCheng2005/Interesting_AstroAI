"""
Classify whether each literature-matched object (HF/SIMBAD/NED) is
genuinely, individually discussed by at least one paper, or only appears as
an uncommented member of a larger survey/sample/catalog list. This
redefines "identified" for the unidentified-objects pipeline: a positional
catalog match alone is not enough. If no paper substantively discusses the
object, it now counts as unidentified, even though HF/SIMBAD/NED "know"
about it.

For every unique matched object (grouped by object_name, same grouping as
deep_dive_summaries.py):
  - 0 papers found in the bibliography join -> genuinely_discussed=False
    automatically, no LLM call needed.
  - >=1 paper found -> fetch ALL of its papers' abstracts from ADS (the
    whole pipeline only touches ~3,900 unique bibcodes total, so this is
    cheap - no need to subsample), search every abstract/title for a direct
    mention of the object's name or any of its catalog aliases (SIMBAD
    main_id / NED object_name recorded per bibliography row - a match
    radius can pick up more than one nearby catalog entry for the same
    physical source, e.g. an optical name and a radio counterpart name), and
    feed Gemini every paper that hit plus a spanning oldest+newest sample of
    the rest (capped at PAPERS_SOFT_CAP total) so a genuinely relevant paper
    is never left out just because it wasn't among the most recent. A name
    hit is flagged explicitly in the prompt as strong (not conclusive -
    could still just be a table entry) evidence. Gemini makes the final
    call on whether any paper individually discusses the object (dedicated
    analysis, specific measurements, notable feature) as opposed to just
    listing it as one of many sample members.

Results are cached per object_name across runs (discussion_classification.csv
is loaded first; already-classified objects are kept, not re-billed) -
re-running after new matches appear only classifies the new ones.

This script only produces discussion_classification.csv. Rerun
generate_unidentified_html_report.py afterward to apply the reclassification
to the report (matched objects with genuinely_discussed=False are shown and
counted as unidentified there).

Requires an ADS API token at ~/.ads_api_key and JB_API_KEY in the environment
(Gemini) - see deep_dive_summaries.py for details.
"""

import os
import re
import csv
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

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


# ── CONFIGURATION ───────────────────────────────────────────────────────────

PAPERS_SOFT_CAP = 40  # cap on papers fed to the LLM per object (name-hits always included)
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_WORKERS = 20

CLASSIFICATION_CSV = os.path.join(SCRIPT_DIR, "discussion_classification.csv")
CLASSIFICATION_FIELDNAMES = ["object_name", "n_papers_checked", "n_name_hits", "genuinely_discussed", "reasoning"]


# ── 1. NAME/ALIAS MATCHING ───────────────────────────────────────────────────

_COORD_DESIGNATION_RE = re.compile(r"J\d{4,10}(?:\.\d+)?[+\-]\d{2,9}(?:\.\d+)?")
_CATALOG_PREFIX_RE = re.compile(
    r"^(NAME|SLACS|SDSS|FIRST|WISEA|2MASS|2MASX|COSMOS2015|COSMOS2020|COSMOS-DASH|"
    r"COSMOS|CANDELS|MCPS|OMS2023|LFC2025|NVSS|VLASS|GALEX|PSO|PS1|Gaia DR[23])\s+",
    re.IGNORECASE,
)


def _normalize_text(s):
    """
    Strip whitespace and periods (decimal points in coordinate designations
    vary in position/precision between sources) and lowercase, so a
    substring search is robust to 'J120540.43+491029.3' vs
    'J120540.4+491029' vs 'J1205+4910' style formatting differences. '+'/'-'
    are kept since they're meaningful (RA/Dec sign).
    """
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"[\s.]+", "", s).lower()


def _coord_variants(designation):
    """
    A coordinate designation like 'J120540.43+491029.3' can appear in the
    literature at varying precision ('J120540.4+491029', 'J1205+4910',
    etc.). Generate progressively-truncated RA/Dec variants (operating on
    the full digit stream, decimal point included, matching _normalize_text)
    so a substring search still matches across formatting differences,
    while keeping enough digits (4 RA + 2 Dec minimum, i.e.
    ~arcmin/degree-level) to stay specific.
    """
    m = re.match(r"J(\d{4,10}(?:\.\d+)?)([+\-])(\d{2,9}(?:\.\d+)?)", designation)
    if not m:
        return set()

    ra, sign, dec = m.groups()
    ra_digits = ra.replace(".", "")
    dec_digits = dec.replace(".", "")
    ra_min = max(4, len(ra.split(".")[0]))
    dec_min = max(2, len(dec.split(".")[0]))

    variants = set()
    for ra_len in range(len(ra_digits), ra_min - 1, -1):
        for dec_len in range(len(dec_digits), dec_min - 1, -1):
            variants.add(f"j{ra_digits[:ra_len]}{sign}{dec_digits[:dec_len]}")
    return variants


def build_search_keys(alias):
    """
    Build a set of normalized substrings worth searching for in paper
    text for a given catalog alias/object name: the coordinate designation
    (in several precisions) if present, plus the full name with common
    survey/catalog prefixes stripped (for non-coordinate proper names like
    'NAME Knot in M87 Jet' -> 'Knot in M87 Jet').
    """
    if not alias:
        return set()

    keys = set()

    for m in _COORD_DESIGNATION_RE.finditer(alias):
        for variant in _coord_variants(m.group(0)):
            keys.add(_normalize_text(variant))

    stripped = _CATALOG_PREFIX_RE.sub("", alias).strip()
    if stripped and len(stripped) >= 4 and not _COORD_DESIGNATION_RE.fullmatch(stripped):
        keys.add(_normalize_text(stripped))

    return {k for k in keys if len(k) >= 6}


def find_name_hits(papers, aliases):
    """
    Search every paper's title+abstract for a mention of any alias. Returns
    the set of bibcodes that hit.
    """
    search_keys = set()
    for alias in aliases:
        search_keys |= build_search_keys(alias)

    if not search_keys:
        return set()

    hits = set()
    for p in papers:
        text = _normalize_text((p.get("_title") or p.get("title") or "") + " " + (p.get("_abstract") or ""))
        if any(key in text for key in search_keys):
            hits.add(p["bibcode"])

    return hits


# ── 2. PAPER SELECTION ───────────────────────────────────────────────────────

def select_papers_for_classification(papers, hit_bibcodes, n=PAPERS_SOFT_CAP):
    """
    Always include every name-hit paper. Fill remaining slots (up to n
    total) spanning oldest + newest of the rest - the dedicated/discovery
    paper is often the oldest, and individual follow-up studies are often
    the newest, while papers in between are disproportionately likely to be
    yet another survey re-listing the object.
    """
    hit_papers = [p for p in papers if p["bibcode"] in hit_bibcodes]
    rest = [p for p in papers if p["bibcode"] not in hit_bibcodes]

    if len(hit_papers) >= n:
        return hit_papers

    remaining_slots = n - len(hit_papers)
    if len(rest) <= remaining_slots:
        return hit_papers + rest

    sorted_rest = sorted(rest, key=lambda p: p["year"] if str(p["year"]).isdigit() else "0")
    half = remaining_slots // 2
    sampled_rest = sorted_rest[:half] + sorted_rest[-(remaining_slots - half):]
    return hit_papers + sampled_rest


# ── 2. LLM CLASSIFICATION ────────────────────────────────────────────────────

class DiscussionClassification(BaseModel):
    genuinely_discussed: bool = Field(
        description=(
            "True if at least one of the given papers specifically discusses this "
            "object individually - a dedicated study, specific measurements "
            "attributed to it, or a description of what makes it notable. False if "
            "the object only appears as an uncommented member of a larger sample, "
            "survey, or catalog table, with no individual discussion."
        )
    )
    reasoning: str = Field(description="One sentence justifying the verdict.")


classification_config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer. You will be given an astronomical object's "
        "name and a set of paper titles/abstracts that mention it (found via a "
        "positional catalog cross-match, not necessarily because the paper is "
        "specifically about this object). Decide whether ANY of the papers "
        "genuinely, individually discusses this object - e.g. a dedicated study, "
        "specific measurements attributed to it, or a description of a notable "
        "feature - versus the object merely being one of many members in a sample, "
        "survey, or catalog table with no individual discussion. Judge "
        "conservatively: an abstract that just says the paper studies 'a sample of "
        "N objects including this one' does NOT count as genuine discussion unless "
        "the object itself is specifically called out with its own findings."
    ),
    response_mime_type="application/json",
    response_schema=DiscussionClassification,
    temperature=0.0,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
)


def build_prompt(object_name, papers, hit_bibcodes):
    lines = [f"Object: {object_name}", "", f"Papers ({len(papers)}):"]
    for p in papers:
        abstract = (p.get("_abstract") or "")[:1200]
        hit_note = (
            " [NOTE: this object's name/designation was found as a direct text match in this "
            "paper's title/abstract - likely but not certainly a sign of individual discussion, "
            "confirm from context]"
            if p["bibcode"] in hit_bibcodes
            else ""
        )
        lines.append(f"\n[{p['year']}]{hit_note} {p.get('_title') or p['title']}\n{abstract}")
    return "\n".join(lines)


def classify_one(object_name, papers, hit_bibcodes, api_key, delays=(1, 2, 4, 8, 16)):
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(object_name, papers, hit_bibcodes)

    for attempt, delay in enumerate(delays):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=classification_config,
            )
            data = response.parsed
            if data is None:
                raise ValueError("empty response.parsed")
            return object_name, data, None
        except Exception as e:
            if attempt == len(delays) - 1:
                return object_name, None, str(e)
            time.sleep(delay)


# ── 3. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gemini_api_key = os.environ.get("JB_API_KEY")
    if not gemini_api_key:
        raise SystemExit("JB_API_KEY not set in environment.")
    ads_api_key = load_ads_api_key()

    print("Loading bibliographies and matched objects...")
    simbad_biblio = load_bibliography(SIMBAD_BIBLIOGRAPHY_CSV)
    ned_biblio = load_bibliography(NED_BIBLIOGRAPHY_CSV)
    best_by_object = load_matched_objects(MATCHED_CSV, simbad_biblio, ned_biblio)
    print(f"{len(best_by_object)} unique matched objects.")

    existing_rows = []
    if os.path.exists(CLASSIFICATION_CSV):
        with open(CLASSIFICATION_CSV, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
    already_done = {r["object_name"] for r in existing_rows}
    print(f"{len(already_done)} objects already classified (cached, will not be re-billed).")

    zero_paper_objects = [
        name for name, o in best_by_object.items()
        if not o["papers"] and name not in already_done
    ]
    with_paper_objects = [
        (name, o) for name, o in best_by_object.items()
        if o["papers"] and name not in already_done
    ]
    print(
        f"{len(zero_paper_objects)} new zero-paper objects (auto not-discussed), "
        f"{len(with_paper_objects)} new objects need LLM classification."
    )

    new_rows = [
        {
            "object_name": name,
            "n_papers_checked": 0,
            "n_name_hits": 0,
            "genuinely_discussed": False,
            "reasoning": "No papers found in bibliography.",
        }
        for name in zero_paper_objects
    ]

    if with_paper_objects:
        # Fetch every paper's abstract up front (not just a capped sample) -
        # the whole dataset only has ~3,900 unique bibcodes, so this is cheap
        # and lets us search ALL abstracts for a name/alias hit before
        # deciding which papers to show the LLM.
        all_bibcodes = [p["bibcode"] for _, o in with_paper_objects for p in o["papers"]]
        print(
            f"\nNeed abstracts for {len(all_bibcodes)} (object, paper) pairs "
            f"({len(set(all_bibcodes))} unique bibcodes)..."
        )
        cache = load_abstract_cache()
        cache = fetch_abstracts(all_bibcodes, ads_api_key, cache)

        for name, o in with_paper_objects:
            for p in o["papers"]:
                entry = cache.get(p["bibcode"], {})
                p["_title"] = entry.get("title") or p["title"]
                p["_abstract"] = entry.get("abstract", "")

        n_with_hits = 0
        for name, o in with_paper_objects:
            aliases = {name} | {p["alias"] for p in o["papers"] if p.get("alias")}
            hit_bibcodes = find_name_hits(o["papers"], aliases)
            o["hit_bibcodes"] = hit_bibcodes
            o["papers_for_classification"] = select_papers_for_classification(o["papers"], hit_bibcodes)
            if hit_bibcodes:
                n_with_hits += 1
        print(f"{n_with_hits}/{len(with_paper_objects)} objects have a direct name/alias hit in at least one paper.")

        print(f"\nClassifying {len(with_paper_objects)} objects with Gemini ({GEMINI_MODEL}, {GEMINI_WORKERS} workers)...")

        results = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as executor:
            futures = [
                executor.submit(classify_one, name, o["papers_for_classification"], o["hit_bibcodes"], gemini_api_key)
                for name, o in with_paper_objects
            ]

            for n_done, future in enumerate(as_completed(futures), start=1):
                object_name, data, error = future.result()
                if error:
                    print(f"  FAILED {object_name}: {error}")
                else:
                    results[object_name] = data
                if n_done % 200 == 0:
                    print(f"  {n_done}/{len(with_paper_objects)} done, {time.time() - t0:.0f}s elapsed")

        print(f"Done in {time.time() - t0:.0f}s. {len(results)}/{len(with_paper_objects)} succeeded.")

        for name, o in with_paper_objects:
            data = results.get(name)
            if data is None:
                continue  # failed after retries - leave unclassified, retry next run
            new_rows.append({
                "object_name": name,
                "n_papers_checked": len(o["papers_for_classification"]),
                "n_name_hits": len(o["hit_bibcodes"]),
                "genuinely_discussed": data.genuinely_discussed,
                "reasoning": data.reasoning,
            })

    all_rows = existing_rows + new_rows

    with open(CLASSIFICATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLASSIFICATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    n_true = sum(1 for r in all_rows if str(r["genuinely_discussed"]).strip().lower() == "true")
    print(
        f"\nWrote {len(all_rows)} total classifications ({len(new_rows)} new) to {CLASSIFICATION_CSV}: "
        f"{n_true} genuinely discussed, {len(all_rows) - n_true} not (now counted as unidentified)."
    )
