# waveform-qc

Per-event triage for muon-veto scintillator panel data: **what each recorded event actually
is.** Every event is labelled CLEAN / SATURATED / PILEUP / NOISE, and everything downstream
depends on that split. Also here: pulse-window diagnostics and the hodoscope efficiency
measurement.

Analysis and code by **Remy Strichartz** (Yale).

## The three repos

```
   waveform-io          layout, ingestion, shared primitives
     ^        ^
     |        |
 waveform-qc  waveform-analysis
  <- HERE       optimal filter + boxcar, spectra,
                the muon line, run stability
```

This repo installs [waveform-io](https://github.com/remy-strichartz/waveform-io) and is
installed by nothing. It does not import `waveform-analysis` and `waveform-analysis` does not
import it — the two are independent siblings that share primitives through the base repo.

See [`preprocessing/README.md`](preprocessing/README.md) for the drivers and the cuts.

## Install

`waveform-io` supplies `hodoscope_common` (the file layout and the classification
primitives). It is not on PyPI, so install it first:

```bash
# development — tracks your local edits
pip install -e ../waveform-io

# or reproducible — pinned tag.  waveform-io is public, so this needs no credentials.
pip install "waveform-io @ git+https://github.com/remy-strichartz/waveform-io.git@v0.1.0"
```

Then:

```bash
pip install -r requirements.txt
```

Point the stack at your data tree — required, because `hodoscope_common` is now installed
rather than sitting beside this repo:

```bash
export WAVEFORM_FILES=/path/to/waveform_files      # bash
$env:WAVEFORM_FILES = "C:\path\to\waveform_files"  # PowerShell
```

Then run the drivers as scripts from the repo root, exactly as before:

```bash
python preprocessing/waveform_triage.py --input run00270_ch0.h5 --detector pmt
```

## Environment

Python 3.11+. CI runs the suite on Linux against a plain pip install.

| package | floor | validated |
|---|---|---|
| numpy | | 2.4.1 |
| scipy | | 1.16.3 |
| h5py | | 3.16.0 |
| matplotlib | | 3.10.8 |

scipy is used for `optimize` and `signal` only. The **`scipy >= 1.16` floor that
`waveform-analysis` carries does not apply here** — that floor exists for
`scipy.stats.landau`, the muon-line fit, which this repo never calls.

## The figures are in this repo; the data is not

`preprocessing_results/` is tracked: the triage galleries, overviews and efficiency plots
read straight from a clone, with no data and no re-run. Every `.h5` / `.h5.gz` / `.mid` file
is gitignored wherever it lands.

## Tests

A synthetic end-to-end regression — it builds its own waveforms and touches no real data:

```bash
python preprocessing/tests/test_preprocessing.py
```

10 tests. Also collected by `pytest` from the repo root, and run on every push
([`.github/workflows/tests.yml`](.github/workflows/tests.yml), which checks out
`waveform-io` alongside this repo and installs it).
