# Hubble

Tests whether AI vision-language models (Gemini, Qwen) can triage a large
catalog of Hubble Legacy Archive cutouts down to a tractable set of
scientifically interesting candidates — gravitational lenses, mergers,
jellyfish galaxies, AGN, and other rare morphologies — while rejecting
imaging artifacts (cosmic rays, diffraction spikes, satellite trails).

- [`analysis/`](analysis/) — AI-judging scripts, results, literature
  cross-matching, reports, and poster figures. See
  [`analysis/README.md`](analysis/README.md) for the full breakdown of the
  pipeline and what each subfolder contains.

Full raw dataset (the source HDF5 image cutout catalog, multiple GB) is not
included here; see O'Ryan & Gómez (2025) in the top-level
[README](../README.md) for the source. A small sample of catalog rows and
example thumbnail images is included at
[`analysis/sample_data/`](analysis/sample_data/).
