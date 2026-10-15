"""
Download the two model logos the `models` beat shows, as PNGs.

Both come from Wikimedia Commons and are freely licensed: the Gemini icon is
public domain, the Qwen logo is CC0. They are used to identify the models the
paper evaluates, nothing more.

Usage:  python prepare/fetch_logos.py
"""

import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGOS = os.path.join(ROOT, "assets", "logos")

SOURCES = {
    "gemini": ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/"
               "Google_Gemini_icon_2025.svg/960px-Google_Gemini_icon_2025.svg.png"),
    "qwen": ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/"
             "Qwen_logo.svg/960px-Qwen_logo.svg.png"),
}


def main():
    os.makedirs(LOGOS, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(LOGOS, f"{name}.png")
        req = urllib.request.Request(url, headers={"User-Agent": "paper-animation/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
