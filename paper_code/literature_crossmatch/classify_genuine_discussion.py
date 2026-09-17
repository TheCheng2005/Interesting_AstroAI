"""
Decide whether a paper genuinely discusses an object, or only lists it.

A positional catalogue match is not evidence that anyone has studied
something: an object can sit in SIMBAD as row 400 of a survey table with no
paper ever saying a word about it. This module draws that line, and it is
what lets build_undiscussed_catalog.py keep an image that is catalogued but
undiscussed.

The evidence for one object is its papers - title, abstract, and the verbatim
in-body snippets ADS returned around each mention. Two steps prepare it:

  find_name_hits                  which papers mention the object's name or
                                  any of its catalogue aliases at all. Name
                                  matching is normalised and expanded to
                                  coordinate-designation variants, because
                                  "SDSS J1205+4910" and "J120540.4+491029"
                                  are the same source written two ways.
  select_papers_for_classification every name-hit, plus a spanning
                                  oldest-to-newest sample of the rest up to
                                  PAPERS_SOFT_CAP, so a relevant paper is
                                  never dropped merely for being old.

classify_one then asks Gemini for the verdict. The bar is deliberately low
and stated as such in the prompt: one sentence saying something about this
specific object is enough - a measured property in a table row counts, a
dedicated study is not required. False only when every mention is a bare
listing. temperature is 0, and build_undiscussed_catalog.py takes a majority
of CLASSIFY_VOTES runs over identical evidence.

A name hit is flagged to the model as strong but not conclusive evidence: it
can still be a table entry, which is exactly the distinction being drawn.

Used by unidentified_objects/build_undiscussed_catalog.py. Requires
GEMINI_API_KEY; abstracts come from ads_abstracts.py.
"""

import os
import re
import time
import unicodedata

from pydantic import BaseModel, Field
from google import genai
from google.genai import types



# The stage folders are siblings, so put the analysis root on the path to
# reach common/paths.py (see its docstring).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── CONFIGURATION ───────────────────────────────────────────────────────────

PAPERS_SOFT_CAP = 40  # cap on papers fed to the LLM per object (name-hits always included)
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_WORKERS = 20


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
            "True if at least one of the given papers contains at least one "
            "sentence that says something about this specific object - its "
            "morphology, a measurement, a classification, a behaviour, or why it "
            "was singled out. One sentence is enough; a dedicated study or a whole "
            "paragraph is NOT required. False only if every mention is a bare "
            "listing with nothing said about the object: a name in a "
            "comma-separated list, a table row, a coordinate column, or an "
            "enumeration of sample members."
        )
    )
    reasoning: str = Field(description="One sentence justifying the verdict.")


classification_config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer. You will be given an astronomical object's "
        "name and a set of paper titles/abstracts that mention it (found via a "
        "positional catalog cross-match, not necessarily because the paper is "
        "specifically about this object). Decide whether ANY of the papers says "
        "anything about this object, as opposed to merely listing it.\n\n"
        "THE BAR IS ONE SENTENCE. If a single sentence anywhere in any paper "
        "states something about this specific object - its morphology, a measured "
        "property, a classification, a redshift, a behaviour, or the reason it was "
        "selected - that is genuine discussion and the answer is True. The object "
        "does NOT need a dedicated paper, a paragraph, a section, or its own "
        "headline findings. A survey paper that tabulates this object and says one "
        "thing about it counts. Being one of many sample members does not "
        "disqualify it, so long as something is actually said about it.\n\n"
        "Answer False only when every mention is a bare listing with nothing said: "
        "a name inside a comma-separated list, a row of a table, a coordinate "
        "column, or an enumeration of sample members - the object's name appears "
        "and nothing about it follows.\n\n"
        "Some papers carry IN-BODY CONTEXT: short verbatim excerpts of the paper's "
        "own body text around each mention. Weigh these above the abstract, since "
        "they show how the object is actually used. An object can easily be absent "
        "from an abstract and still be discussed in a sentence in the body."
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

        snippets = (p.get("_snippets") or "").strip()
        if snippets:
            lines.append(f"IN-BODY CONTEXT: {snippets}")
        elif p["bibcode"] in hit_bibcodes:
            # A hit with no snippet usually means the mention sits beyond the
            # highlighted window, not that there is no mention. Say so, rather
            # than letting its absence read as evidence against discussion.
            lines.append(
                "IN-BODY CONTEXT: unavailable - the name matches this paper's full text "
                "but no surrounding excerpt could be retrieved; judge it on the abstract."
            )
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
