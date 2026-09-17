"""
One reader for the stage-1 scoring CSVs.

The eight scoring protocols write two different header spellings:

    Likert / Hybrid        index,filename,imagescore,interesting,classification,SourceRA,SourceDec
    Tournament / Single-elim     Filename,ImageScore,interesting,classification,SourceRA,SourceDec

Both carry the same information; Tournament simply has no per-image index.
Consumers used to hardcode one spelling, which made a Tournament CSV either
crash a reader or - worse - read as zero rows without any error, because
`row.get("filename")` returns "" for a file whose column is called
"Filename". Everything that reads a scoring CSV goes through this module
instead, so either spelling works everywhere with no adjustment.

Every CSV also ends with a token-accounting footer:

    # TOKEN USAGE SUMMARY
    # TotalInputTokens,<n>
    # TotalOutputTokens,<n>
    # TotalTokens,<n>
    # TotalThinkingTokens,<n>   (Gemini runs that log them separately)
    # TotalCostUSD,<x>          (Qwen runs: the charge OpenRouter recorded)

Those rows are the run's own cost record. They are parsed into `tokens` here
and must be written back by anything that rewrites a CSV, or the cost column
of the paper's cost table silently becomes zero.

`TotalCostUSD` is preferred over a token-price estimate wherever present,
because it is what the provider actually billed - which is what the paper
quotes for Qwen.
"""

import csv


# Canonical column name -> the spellings that mean it, lowercased for lookup.
_ALIASES = {
    "index":          ("index",),
    "filename":       ("filename",),
    "imagescore":     ("imagescore",),
    "interesting":    ("interesting",),
    "classification": ("classification",),
    "SourceRA":       ("sourcera",),
    "SourceDec":      ("sourcedec",),
}

# Without these a file is not a scoring CSV at all.
_REQUIRED = ("filename", "imagescore", "interesting")


class ScoresCSV:
    """
    A parsed scoring CSV.

    rows       list of dicts keyed by the canonical names in _ALIASES.
               `index` is None for the Tournament/single-elim spelling.
    header     the original header row, so a rewrite can round-trip it.
    tokens     {"input", "output", "thinking": int, "cost_usd": float or None}
               from the footer. Counts are zero and cost None when absent.
    """

    def __init__(self, rows, header, tokens, path):
        self.rows = rows
        self.header = header
        self.tokens = tokens
        self.path = path

    def scores(self):
        """filename -> imagescore, as float."""
        return {r["filename"]: float(r["imagescore"]) for r in self.rows}

    def ground_truth(self):
        """filename -> True when an AnomalyMatch anomaly is within 3"."""
        return {r["filename"]: str(r["interesting"]).strip() == "1" for r in self.rows}

    def __len__(self):
        return len(self.rows)


def read(path):
    """
    Parse one scoring CSV, accepting either header spelling.

    Raises ValueError rather than returning an empty result, so a schema the
    reader does not understand can never be mistaken for a file with no rows.
    """
    rows, header = [], None
    tokens = {"input": 0, "output": 0, "thinking": 0, "cost_usd": None}
    column_of = None

    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.reader(f):
            if not raw:
                continue

            if raw[0].startswith("#"):
                if len(raw) > 1:
                    key = raw[0].lstrip("# ").strip()
                    value = raw[1].strip()
                    if key == "TotalInputTokens":
                        tokens["input"] = int(value)
                    elif key == "TotalOutputTokens":
                        tokens["output"] = int(value)
                    elif key == "TotalThinkingTokens":
                        tokens["thinking"] = int(value)
                    elif key == "TotalCostUSD":
                        tokens["cost_usd"] = float(value)
                continue

            if column_of is None:
                header = list(raw)
                lowered = [c.strip().lower() for c in raw]
                column_of = {
                    canonical: next(
                        (lowered.index(a) for a in aliases if a in lowered), None
                    )
                    for canonical, aliases in _ALIASES.items()
                }
                missing = [c for c in _REQUIRED if column_of[c] is None]
                if missing:
                    raise ValueError(
                        f"{path}: not a scoring CSV - header {raw} is missing "
                        f"{missing}"
                    )
                continue

            rows.append({
                canonical: (raw[i] if i is not None and i < len(raw) else None)
                for canonical, i in column_of.items()
            })

    if column_of is None:
        raise ValueError(f"{path}: no header row found")
    if not rows:
        raise ValueError(f"{path}: header parsed but no data rows")

    return ScoresCSV(rows, header, tokens, path)


def write(path, rows, header, tokens):
    """
    Write rows back in the spelling `header` uses, footer included.

    Kept so that anything rewriting a scoring CSV leaves it readable by
    exactly the same consumers as the file it came from, with the token
    accounting that the cost table is computed from still intact.
    """
    lowered = [c.strip().lower() for c in header]
    order = []
    for column in header:
        canonical = next(
            (c for c, aliases in _ALIASES.items()
             if column.strip().lower() in aliases),
            None,
        )
        if canonical is None:
            raise ValueError(f"{path}: cannot map output column {column!r}")
        order.append(canonical)
    del lowered

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[c] for c in order])

        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", tokens["input"]])
        writer.writerow(["# TotalOutputTokens", tokens["output"]])
        writer.writerow(["# TotalTokens", tokens["input"] + tokens["output"]])
        if tokens.get("thinking"):
            writer.writerow(["# TotalThinkingTokens", tokens["thinking"]])
        if tokens.get("cost_usd") is not None:
            writer.writerow(["# TotalCostUSD", tokens["cost_usd"]])
