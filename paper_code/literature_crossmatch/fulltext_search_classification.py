"""
Ask ADS which papers mention an object in their body text, and with what
surrounding sentence.

An abstract almost never spells out an individual object's designation; the
methods, tables and figure captions do. Searching abstracts alone therefore
misses most real discussion, and - worse - cannot tell "one of 400 rows in
Table 3" from "we model its tidal tail in Section 4". The snippets this
module returns are what make that distinction possible.

This is server-side phrase search against ADS's full-text index (the `full:`
field), not semantic search, and no paper is ever fetched or stored: ADS
returns at most HL_SNIPPETS fragments of HL_FRAGSIZE characters per paper.

Two query details decide whether it works at all, both learned the hard way:

  - The search is restricted to the object's own papers with a `bibcode:`
    FILTER QUERY (fq), never a clause inside q. Both a `bibcode:` clause and
    `database:astronomy` inside q silently disable Solr highlighting, so the
    request still succeeds and simply returns no snippets.
  - Restricting to the bibcodes SIMBAD and NED already attribute to the
    object is also what keeps the evidence honest. Searching a designation
    across all of ADS pulls in papers from other fields that happen to reuse
    the string - full:"ASV 25" matches microbiology papers about amplicon
    sequence variants - and no amount of evidence from such a paper says
    anything about a galaxy.

Aliases are expanded to the coordinate-designation spellings the literature
actually prints, since an object discussed under a name that was never
searched looks undiscussed.

Used by unidentified_objects/build_undiscussed_catalog.py. The ADS token is
read from ~/.ads_api_key; the 5,000 requests/day quota is shared with the
abstract fetch in ads_abstracts.py.
"""

import os
import re
import time

import requests



# The stage folders are siblings, so put the analysis root on the path to
# reach common/paths.py (see its docstring).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -- CONFIGURATION ----------------------------------------------------------
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
    """
    One ADS query matching any of an object's name variants in paper bodies.

    Two things must stay OUT of this query, because either one silently
    switches off Solr highlighting - ADS still reports which papers mention
    the object, but returns no excerpt showing what they say about it, which
    is the evidence the classifier actually needs:

      - a bibcode: clause. To confine the search to papers a catalogue
        already attributes to the object, pass build_bibcode_filter() as `fq`
        to search_fulltext instead. As a filter query the same restriction
        leaves highlighting intact.
      - database:astronomy. Verified directly: full:"SDSS J1205+4910" returns
        4 body fragments, and database:astronomy AND full:"SDSS J1205+4910"
        returns 0 on the same documents.

    That matters because ADS indexes all of the sciences, and short catalogue
    designations collide across fields - an unrestricted full:"ASV 25" matches
    microbiology papers using ASV for "amplicon sequence variant". The
    bibliography filter, not a database clause, is what rules those out: every
    paper it admits was already linked to this object by SIMBAD or NED. A
    caller that searches without `fq` gets ADS-wide reach and inherits that
    collision risk.
    """
    phrases = set()
    for alias in aliases:
        phrases |= expand_alias_to_phrases(alias)

    clauses = [f'full:"{sanitize_phrase(p)}"' for p in phrases if sanitize_phrase(p)]
    if not clauses:
        return None
    return "(" + " OR ".join(clauses) + ")"


def build_bibcode_filter(bibcodes):
    """Solr filter query restricting results to a specific set of papers."""
    cleaned = [b for b in {sanitize_phrase(b) for b in bibcodes or []} if b]
    if not cleaned:
        return None
    return "bibcode:(" + " OR ".join(f'"{b}"' for b in cleaned) + ")"


def search_fulltext(query, api_key, delays=(1, 2, 4, 8), rows=None, fq=None):
    for attempt, delay in enumerate(delays):
        try:
            resp = requests.get(
                ADS_SEARCH_URL,
                params={
                    "q": query,
                    # Restricting by filter query rather than inside q keeps
                    # Solr highlighting alive (see build_fulltext_query).
                    **({"fq": fq} if fq else {}),
                    # 'id' is Solr's uniqueKey, and the highlighting map is keyed
                    # by it rather than by bibcode - without it snippets cannot be
                    # attributed back to a paper.
                    "fl": "id,bibcode,title,year",
                    "rows": rows or ROWS_PER_QUERY,
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
