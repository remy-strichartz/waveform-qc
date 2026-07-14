#!/usr/bin/env python3
"""
waveform_triage.py
==================
Standalone triage tool for single-channel hodoscope waveform windows.

Reads an HDF5 file of waveform windows and classifies each
event into one of four classes:

  CLEAN      -- a single real pulse in the pulse window (the coincidence pulse
                the AND trigger fired on).
  SATURATED  -- the pulse clips the ADC rail (flat-topped); charge is corrupted.
  PILEUP     -- a real pulse in the pulse window PLUS a second real pulse
                elsewhere, from a particle that did not pass both panels.
  NOISE      -- no real pulse in the pulse window (trigger fired on noise, or
                the whole record is oscillating).

By default it only classifies and prints diagnostics (and draws plots if asked);
it does NOT write any files.  Pass --export to write one .h5 per class (raw
waveforms, untouched) into the per-run results folder (see below).  The
downstream energy_reconstruction tools re-run this classification in-memory (they
import classify_events) and do not read these subset files, so --export is only
needed when you want the subsets on disk.  Each exported class file also carries
the input's per-event time axis (event_time_unix, headers_*, source_event_index),
row-filtered to the class, so subsets stay aligned with the run's recovered times
(timing_stability) -- re-join via /source_event_index.  With --save-plots /
interactive display it also draws an overview panel and a browsable example
gallery per class.

Conventions
-----------
* The pulse sits in a fixed PULSE WINDOW (default samples 366-726), not exactly
  at the window center.  Tune with --pulse-lo / --pulse-hi.
* Noise sigma is GLOBAL: one robust value pooled across all waveforms.
* Waveforms are analyzed RAW (no per-event baseline subtraction); a single
  global baseline level is estimated once and pulse heights are measured
  relative to it.
* POLARITY: positive-going (hodoscope/SiPM) pulses are the default.  Negative-going
  PMT pulses are supported via --polarity negative (or --polarity auto): the record
  is reflected about its baseline (oriented = 2*baseline - raw) so the spike points
  UP, after which every cut is identical to the positive case.  Reflection preserves
  both the baseline level and the noise sigma, so thresholds keep their meaning.  The
  files written to disk are always the ORIGINAL raw waveforms, untouched; only the
  internal analysis copy is oriented.  PMT pulses are typically a little faster, so
  for them you may want a shorter --min-separation / leading-edge window.

Usage
-----
    python waveform_triage.py --input run00270_ch0.h5                       # classify + report only
    python waveform_triage.py --input run00270_ch0.h5 --export              # also write per-class .h5
    python waveform_triage.py --input run00270_ch0.h5 --save-plots
    python waveform_triage.py --input run00270_pmt.h5 --detector pmt        # PMT preset (auto-window on)
    python waveform_triage.py --input run00270_ch0.h5 --no-auto-window --pulse-lo 366 --pulse-hi 726
    python waveform_triage.py --input run00270_ch0.h5 --gallery noise       # browse one cut type
    python waveform_triage.py --input run00270_ch0.h5 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from peakfind import find_peaks_manual

logger = logging.getLogger("waveform_triage")

# Project root is two levels above preprocessing/waveform_triage.py.
_PROJECT_ROOT        = Path(__file__).parent.parent
_DEFAULT_WAVEFORM_DIR = _PROJECT_ROOT / "waveform_files"

# Shared waveform-file and per-run results-folder conventions
# (see file_manipulation/output_paths.py).
sys.path.insert(0, str(_PROJECT_ROOT / "file_manipulation"))
from output_paths import resolve_input, resolve_results_dir  # noqa: E402

# Recommended cut presets per detector readout, derived from the run00270 data.
# SiPM pulses are tall, slow, positive and can clip the ADC rail; PMT pulses are
# small, FAST, negative and rarely saturate.  So PMT mode uses:
#   * a HIGHER relative spike threshold (noise_prominence) -- the sharp, fast PMT
#     pulse needs more separation from noise to be called real;
#   * FEWER consecutive at-rail samples (consec) for the flat-top saturation test --
#     a fast pulse that did clip would sit on the rail for only a sample or two;
#   * a SMALLER min-separation -- the pulse is narrower, so a genuine second pulse
#     can arrive closer.
# Any value given explicitly on the command line overrides the preset.
DETECTOR_PRESETS = {
    "sipm": {"polarity": "positive", "noise_prominence": 5.0, "consec": 4, "min_separation": 40},
    "pmt":  {"polarity": "negative", "noise_prominence": 6.0, "consec": 2, "min_separation": 25},
}
GALLERY_CLASSES = ("clean", "saturated", "pileup", "noise")

# The preset entries that are CUT parameters (accepted by classify_events); "polarity"
# describes the readout, not a cut, so it is not one of them.
_CUT_KEYS = ("noise_prominence", "consec", "min_separation")

# This project's two readouts are 1:1 with pulse polarity (SiPM positive, PMT negative),
# so a caller that knows only the SIGN -- e.g. mv_pipeline, whose Config carries polarity
# but no detector -- can still resolve the right preset.
_DETECTOR_FOR_POLARITY = {"positive": "sipm", "negative": "pmt"}


def detector_for_polarity(polarity: str) -> str | None:
    """The readout implied by a pulse polarity, or None when it is not determinable
    ("auto": the caller has not resolved the sign yet)."""
    return _DETECTOR_FOR_POLARITY.get(polarity)


def triage_cuts(detector: str | None = None, *, polarity: str | None = None) -> dict:
    """The classify_events cut parameters for a detector readout -- the ONE place the
    per-readout values are resolved, so every caller of the shared classifier cuts on
    identical grounds instead of by convention.

    classify_events' own defaults ARE the 'sipm' preset, so a caller that just took the
    defaults was silently applying SiPM cuts to a PMT channel.  The one that bites is
    min_separation=40 on a fast PMT pulse: a genuine second pulse arriving 25-39 samples
    after the main one is skipped, and the event survives as CLEAN.

    Pass `detector`, or `polarity` when only the sign is known (see detector_for_polarity).
    An unresolvable polarity ("auto") yields {} -- i.e. the classify_events defaults.
    """
    if detector is None and polarity is not None:
        detector = detector_for_polarity(polarity)
    if detector is None:
        return {}
    if detector not in DETECTOR_PRESETS:
        raise ValueError(f"unknown detector {detector!r}; "
                         f"expected one of {sorted(DETECTOR_PRESETS)}")
    preset = DETECTOR_PRESETS[detector]
    return {k: preset[k] for k in _CUT_KEYS if k in preset}


# ===========================================================================
# Loading
# ===========================================================================

def _resolve_waveform_dataset(f) -> tuple[str, Any]:
    """Locate the waveform dataset in an OPEN HDF5 file without reading any
    samples: the dataset named 'waveforms', else the largest 2-D dataset (falling
    back to the largest dataset overall).  Returning the dataset handle (not its
    data) lets the whole-file loader and the block-streaming path share the exact
    same discovery rule."""
    import h5py
    datasets: list = []
    f.visititems(lambda n, o: datasets.append((n, o))
                 if isinstance(o, h5py.Dataset) else None)
    logger.info("HDF5 datasets: %s", [f"{n}{d.shape}" for n, d in datasets])
    # Prefer a dataset named "waveforms"; fall back to largest 2-D dataset.
    if "waveforms" in f and isinstance(f["waveforms"], h5py.Dataset):
        return "waveforms", f["waveforms"]
    two_d = [(n, d) for n, d in datasets if d.ndim == 2]
    return max(two_d or datasets, key=lambda nd: int(np.prod(nd[1].shape)))


def load_waveforms(path: Path, channel: int | None = None) -> tuple[np.ndarray, dict]:
    """Load waveforms as a 2-D float32 array (N, L).  Auto-discovers the layout
    of an HDF5 file (largest 2-D dataset).

    For a 3-D (N, n_ch, L) file, `channel` selects the channel (required to be
    explicit for multi-channel files; None squeezes a size-1 channel axis and
    otherwise falls back to channel 0 with a warning)."""
    p = Path(path)
    suffix = p.suffix.lower()
    info: dict[str, Any] = {"source": str(p)}

    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError:
            raise SystemExit("h5py is required to read HDF5 files "
                             "(pip install h5py / conda install h5py).")
        with h5py.File(p, "r") as f:
            name, ds = _resolve_waveform_dataset(f)
            # Load as float32, not float64: the raw ADC samples are small integers
            # (exactly representable in float32), so float32 gives identical cuts
            # while halving peak memory versus float64.
            arr = np.asarray(ds[()], dtype=np.float32)
            logger.info("Using dataset '%s' %s.", name, arr.shape)
            info["dataset"] = name
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if arr.ndim == 3:                                   # (N, n_ch, L) -> one channel
        n_ch = arr.shape[1]
        if channel is None:
            if n_ch > 1:
                logger.warning("3-D array %s; taking channel 0 (pass channel= to select).", arr.shape)
            arr = arr[:, 0, :]
        elif 0 <= channel < n_ch:
            arr = arr[:, channel, :]
            info["channel"] = int(channel)
        else:
            raise ValueError(f"channel {channel} out of range 0..{n_ch - 1} for {p} {arr.shape}.")
    elif channel not in (None, 0):
        raise ValueError(f"{p} is 2-D single-channel but channel {channel} was requested.")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D (N, L) waveforms; got shape {arr.shape}.")

    logger.info("Loaded %d waveforms of length %d.", *arr.shape)
    info["n_events"], info["length"] = int(arr.shape[0]), int(arr.shape[1])
    return arr, info


# ===========================================================================
# Global baseline + noise (computed ONCE for the whole file)
# ===========================================================================

def global_baseline_noise(waveforms: np.ndarray, pulse_lo: int) -> tuple[float, float]:
    """Single global baseline level and noise sigma, pooled across all events.

    Uses the pre-pulse region (samples before pulse_lo), which is pulse-free,
    for every waveform at once.  Baseline = global median; sigma = global MAD.
    """
    pre = waveforms[:, :max(pulse_lo, 5)]
    baseline = float(np.median(pre))
    sigma = 1.4826 * float(np.median(np.abs(pre - baseline)))
    sigma = max(sigma, 1e-6)
    logger.info("Global baseline = %.4g ADC, noise sigma = %.4g ADC "
                "(pooled over first %d samples).", baseline, sigma, pre.shape[1])
    return baseline, sigma


# ===========================================================================
# Polarity (negative-going PMT pulses) -> orient so the spike points UP
# ===========================================================================

def polarity_vote(waveforms: np.ndarray) -> tuple[str, float, float, bool]:
    """The project's ONE polarity vote (mv_pipeline.resolve_polarity delegates
    here): center each event on its own median, then compare the 95th percentiles
    of the up- and down-excursions.  The high quantile keeps the vote with the
    pulse-bearing events even when only a modest fraction of events fire; a median
    vote there compares noise against noise and the sign is a coin flip.
    Returns (polarity, up, down, ambiguous)."""
    centered = waveforms - np.median(waveforms, axis=1, keepdims=True)
    up = float(np.percentile(centered.max(axis=1), 95))
    down = float(np.percentile(-centered.min(axis=1), 95))
    polarity = "negative" if down > up else "positive"
    # BIAS GUARD: if up and down excursions are comparable the sign is not safely
    # determinable from the data (dead channel, or symmetric oscillation).
    bigger, smaller = max(up, down), min(up, down)
    ambiguous = bigger <= 0 or smaller / bigger > 0.7
    return polarity, up, down, ambiguous


def detect_polarity(waveforms: np.ndarray) -> str:
    """Auto-detect pulse polarity from the data (95th-percentile excursion vote,
    see polarity_vote).  WARNS when the two excursions are too close to call."""
    polarity, up, down, ambiguous = polarity_vote(waveforms)
    logger.info("Detected polarity = %s (95th-pct up-excursion %.3g vs down-excursion %.3g ADC).",
                polarity, up, down)
    if ambiguous:
        logger.warning("Polarity is AMBIGUOUS (up %.3g vs down %.3g ADC are comparable); "
                       "set --polarity / --detector explicitly for this channel.", up, down)
    return polarity


def resolve_polarity(polarity: str, waveforms: np.ndarray) -> str:
    """Resolve "auto" to a concrete "positive"/"negative"; pass others through.

    When an explicit polarity is given it is still cross-checked against the data,
    and a mismatch is WARNED (e.g. a channel mislabeled via --detector sipm/pmt) --
    a silent wrong sign would invert every downstream height/charge.  The mismatch
    warning is suppressed when the vote itself is ambiguous: a dead/disconnected
    channel has no meaningful detected sign to disagree with."""
    if polarity == "auto":
        return detect_polarity(waveforms)
    if polarity not in ("positive", "negative"):
        raise ValueError(f"polarity must be positive/negative/auto, got {polarity!r}")
    detected, up, down, ambiguous = polarity_vote(waveforms)
    if polarity != detected and not ambiguous:
        logger.warning("Declared polarity '%s' disagrees with the data (looks '%s'; up %.3g "
                       "vs down %.3g ADC); check --polarity / --detector for this channel.",
                       polarity, detected, up, down)
    return polarity


def orient_waveforms(waveforms: np.ndarray, polarity: str,
                     baseline: float) -> np.ndarray:
    """Return an analysis copy in which the pulse points UP.

    Positive polarity is returned unchanged.  Negative (PMT) polarity is REFLECTED
    about the baseline: oriented = 2*baseline - raw.  Reflection turns a downward
    spike into an identical upward one while leaving the baseline level and the
    (MAD) noise sigma unchanged, so all downstream thresholds keep their meaning.
    The ADC floor a negative pulse may clip against (e.g. 0) maps to an upper rail
    at 2*baseline, where the existing saturation flat-top logic detects it."""
    if polarity == "negative":
        # Preserve the input dtype: a float32 array stays float32 (the subtraction
        # would otherwise promote to float64 and undo the loader's memory saving).
        return (2.0 * baseline - waveforms).astype(waveforms.dtype, copy=False)
    return waveforms


def leading_edge_pos(waveforms: np.ndarray, baseline: float, peak_pos: np.ndarray,
                     frac: float = 0.5, subsample: bool = False) -> np.ndarray:
    """Vectorized constant-fraction timing pick: for each event, the LAST sample
    at/below `frac` of the peak height (above baseline) on the rising edge of the
    peak at `peak_pos`.  Unlike the raw argmax time it neither jitters with
    coherent pickup riding on the crest nor lands on an arbitrary sample of a
    saturated flat top, so inter-channel Delta-t built from it is sharper.
    `waveforms` must be ORIENTED (pulses up).

    With `subsample`, the integer edge is refined by linear interpolation to the
    exact threshold crossing (float samples).  Degenerate events (no sample below
    threshold left of the peak) fall back to the peak position itself."""
    N, L = waveforms.shape
    rows = np.arange(N)
    thr = baseline + frac * (waveforms[rows, peak_pos] - baseline)
    idxgrid = np.arange(L)[None, :]
    below = (waveforms <= thr[:, None]) & (idxgrid <= peak_pos[:, None])
    edge = np.where(below, idxgrid, -1).max(axis=1)
    edge = np.where(edge >= 0, edge, peak_pos)
    if not subsample:
        return edge.astype(np.int64)
    e = np.clip(edge, 0, L - 2)
    v0 = waveforms[rows, e]
    v1 = waveforms[rows, e + 1]
    denom = np.where(v1 != v0, v1 - v0, 1.0)
    step = np.clip((thr - v0) / denom, 0.0, 1.0)
    return e + step


# ===========================================================================
# Auto pulse window (derive [pulse_lo, pulse_hi] from the data)
# ===========================================================================

def median_pulse_extent(median_pulse: np.ndarray, peak: int, frac: float = 0.1,
                        fall_frac: float | None = None) -> tuple[int, int]:
    """Samples the median pulse stays above a fraction of its peak, left (`frac`)
    and right (`fall_frac`, default `frac`) of the peak -- its rise and fall
    extent.  The ONE shared extent rule: mv_pipeline sizes its template with a
    LOWER fall fraction (2%) so the decay tail is captured."""
    h = median_pulse[peak]
    if h <= 0:
        return 0, 0
    thr = frac * h
    lo = peak
    while lo > 0 and median_pulse[lo] > thr:
        lo -= 1
    thr = (frac if fall_frac is None else fall_frac) * h
    hi = peak
    while hi < median_pulse.size - 1 and median_pulse[hi] > thr:
        hi += 1
    return peak - lo, hi - peak


def recommend_window(waveforms: np.ndarray, baseline: float, sigma: float,
                     min_sigma: float = 8.0, coverage: float = 0.99,
                     pad: int = 5, max_lead: int = 120) -> tuple[int, int] | None:
    """Recommend (pulse_lo, pulse_hi) from the data so the window actually CONTAINS
    the real pulses -- the cure for events whose pulse peaks just outside a hand-set
    window and is then misread as NOISE.

    `waveforms` must be ORIENTED (pulses up).  Real pulses (peak >= min_sigma*sigma)
    have their peak positions collected; the window spans `coverage` of those (e.g.
    99%), widened by the median pulse's own rise/fall extent plus `pad`.

    `max_lead` caps how far the window may extend from the bulk pulse (the median
    trace's peak).  Without it, a small tail of early stray/after-pulses unrelated to
    the coincidence (common on a noisy PMT) would drag the window open by hundreds of
    samples, which both wastes baseline and risks counting those strays as the pulse.

    Returns None (keep the caller's window) if there are too few real pulses to be
    reliable.
    """
    L = waveforms.shape[1]
    peak_pos = waveforms.argmax(axis=1)
    real = (waveforms.max(axis=1) - baseline) >= min_sigma * sigma
    positions = peak_pos[real]
    if positions.size < 20:
        logger.warning("Auto-window: only %d pulses clear %g sigma; keeping the given window.",
                       positions.size, min_sigma)
        return None
    med = np.median(waveforms[real] - baseline, axis=0)
    mpk = int(med.argmax())                         # bulk pulse position (robust anchor)
    rise, fall = median_pulse_extent(med, mpk)
    tail = (1.0 - coverage) / 2.0 * 100.0
    q_lo, q_hi = (int(np.floor(v)) for v in np.percentile(positions, [tail, 100.0 - tail]))
    # Reject peak-position outliers far from the bulk pulse before padding.
    q_lo = max(q_lo, mpk - max_lead)
    q_hi = min(q_hi, mpk + max_lead)
    rec_lo = max(q_lo - rise - pad, 5)
    rec_hi = min(q_hi + fall + pad, L)
    if not (5 <= rec_lo < rec_hi <= L):
        logger.warning("Auto-window: derived window [%d, %d) invalid; keeping the given window.",
                       rec_lo, rec_hi)
        return None
    return rec_lo, rec_hi


# ===========================================================================
# Shared channel preparation (rough -> refined baseline/window/orientation)
# ===========================================================================

class PreparedChannel(NamedTuple):
    """One channel prepared for classification: the oriented analysis copy plus
    the refined baseline/noise, the final pulse window, and the resolved polarity."""
    oriented: np.ndarray
    baseline: float
    sigma: float
    pulse_lo: int
    pulse_hi: int
    polarity: str


def prepare_channel(raw: np.ndarray, polarity: str, pulse_lo: int, pulse_hi: int, *,
                    auto_window: bool = True, coverage: float = 0.99,
                    min_sigma: float = 8.0, pad: int = 5) -> PreparedChannel:
    """The shared rough->refine channel preparation recipe -- the ONE place it is
    written down.  triage(), hodoscope_efficiency.classify_panel and
    channel_diagnostics.channel_stats all call this, so their windows, baselines
    and noise sigmas are identical by construction rather than by convention.

      1. Validate the (provisional) window against the record length.
      2. Rough baseline/noise: from the first 20% of the record with auto_window
         (the hand-set window is only provisional), else from the fixed window's
         own pre-pulse region.
      3. Resolve polarity and orient so the pulse points UP.
      4. With auto_window, derive [pulse_lo, pulse_hi) from the data
         (recommend_window; the given window is kept if there are too few real
         pulses), then recompute baseline/noise from the FINAL window's pre-pulse
         region and re-orient.
    """
    L = raw.shape[1]
    pulse_hi = min(pulse_hi, L)
    if not (5 <= pulse_lo < pulse_hi):
        raise ValueError(
            f"Invalid pulse window [{pulse_lo}, {pulse_hi}) for records of length {L}: "
            "need 5 <= pulse_lo < pulse_hi <= length (pulse_lo leaves the pre-pulse "
            "baseline region). Set --pulse-lo / --pulse-hi.")

    prov_stop = max(int(0.2 * L), 5) if auto_window else pulse_lo
    baseline, sigma = global_baseline_noise(raw, prov_stop)
    polarity = resolve_polarity(polarity, raw)
    oriented = orient_waveforms(raw, polarity, baseline)

    if auto_window:
        rec = recommend_window(oriented, baseline, sigma, min_sigma=min_sigma,
                               coverage=coverage, pad=pad)
        if rec is not None and rec != (pulse_lo, pulse_hi):
            pulse_lo, pulse_hi = rec
            logger.info("Auto-window -> --pulse-lo %d --pulse-hi %d (coverage %.3f).",
                        pulse_lo, pulse_hi, coverage)
        baseline, sigma = global_baseline_noise(raw, pulse_lo)
        oriented = orient_waveforms(raw, polarity, baseline)

    return PreparedChannel(oriented, float(baseline), float(sigma),
                           int(pulse_lo), int(pulse_hi), polarity)


# ===========================================================================
# Classification (all on ORIENTED waveforms; heights measured above global baseline)
# ===========================================================================

def detect_saturation(waveforms: np.ndarray, saturation_adc: float | None,
                      consec: int = 4, rail_tol: float = 0.01
                      ) -> tuple[np.ndarray, float, bool]:
    """Flag events that clip the ADC rail: peak reaches the rail, or there is a
    flat top of `consec` samples within rail_tol of the rail.

    Returns (mask, rail_used, rail_found).  rail_found is True when the rail is
    trusted -- user-supplied or auto-detected from a genuine pile-up of peaks at
    the top of the distribution.  When no true rail exists the threshold falls
    back to the 99.99th percentile of peaks and rail_found is False; callers
    should then treat the mask as advisory (tallest-event candidates) rather
    than a cut."""
    row_max = waveforms.max(axis=1)
    rail_found = True

    if saturation_adc is None:
        vals, counts = np.unique(np.round(row_max).astype(np.int64), return_counts=True)
        top_val, top_count = int(vals[np.argmax(counts)]), int(counts.max())
        # A genuine rail is where the TALLEST pulses pile up -- at the high end of
        # the peak distribution.  Require the clustered value both to be common AND
        # to sit at/above the median peak; otherwise the "cluster" is the noise
        # floor (which dominates when many events are misses, e.g. an efficiency
        # run's middle panel), and treating it as a rail would collapse the rail to
        # the baseline and flag everything as saturated.
        if top_count / len(waveforms) > 0.005 and top_val >= np.median(row_max):
            saturation_adc = float(top_val)
            logger.info("Auto saturation rail = %d ADC.", top_val)
        else:
            saturation_adc = float(np.percentile(row_max, 99.99))
            rail_found = False
            logger.info("No rail; using 99.99th pct of peaks = %.1f ADC (advisory only).",
                        saturation_adc)

    # Only flag events with a genuine flat top AT the rail (consec consecutive
    # samples within rail_tol of saturation_adc).  A single peak-hit without a
    # flat top is a tall clean pulse that happened to approach the rail; it is
    # not truncated and its energy estimate is unbiased.
    rail_band = saturation_adc * (1.0 - rail_tol)

    near_rail = (waveforms >= rail_band).astype(np.int32)
    cum = np.zeros((near_rail.shape[0], near_rail.shape[1] + 1), dtype=np.int32)
    cum[:, 1:] = np.cumsum(near_rail, axis=1)
    # consec consecutive samples within rail_band already implies row_max >= rail_band,
    # so no separate row_max test is needed.
    mask = (cum[:, consec:] - cum[:, :-consec]).max(axis=1) >= consec
    logger.info("Saturation: %d events (%.1f%%).", int(mask.sum()), 100 * mask.mean())
    return mask, float(saturation_adc), rail_found


def detect_saturation_cut(waveforms: np.ndarray, saturation_adc: float | None,
                          consec: int = 4, rail_tol: float = 0.01
                          ) -> tuple[np.ndarray, float, bool]:
    """detect_saturation resolved into an ACTUAL cut: when no genuine rail exists
    the threshold is only a percentile fallback and nothing is truly clipped, so
    the advisory mask is dropped (empty cut).  The ONE place that rule lives."""
    sat, sat_thr, rail_found = detect_saturation(waveforms, saturation_adc,
                                                 consec=consec, rail_tol=rail_tol)
    if not rail_found:
        sat = np.zeros(len(waveforms), dtype=bool)
    return sat, sat_thr, rail_found


def _is_real_pulse(wf: np.ndarray, peak: int, baseline: float, min_height: float,
                   rise: int = 40) -> bool:
    """A genuine pulse clears min_height above baseline and has a sharp leading
    edge (climbs most of its height from a low foot within `rise` samples)."""
    h = wf[peak] - baseline
    if h < min_height:
        return False
    foot = float(np.min(wf[max(peak - rise, 0):peak + 1])) - baseline
    return (h - foot) >= 0.5 * h            # most of the height is on the rise


def _pulse_window_peak(wf: np.ndarray, pulse_lo: int, pulse_hi: int) -> int:
    """Index of the highest sample inside the pulse window [pulse_lo, pulse_hi]."""
    return pulse_lo + int(np.argmax(wf[pulse_lo:pulse_hi]))


def _count_big_peaks(wf: np.ndarray, baseline: float, height: float,
                     distance: int) -> int:
    """Number of distinct peaks in the whole record that rise above `height`
    over baseline.  An isolated pulse has 1, a genuine pileup has 2, oscillating
    noise has many."""
    peaks = find_peaks_manual(wf - baseline, height=height, distance=distance)
    return int(peaks.size)


def _pulse_dominates(wf: np.ndarray, peak: int, baseline: float,
                     distance: int, sigma: float, max_peaks: int = 4,
                     dom_floor_sigma: float = 4.0) -> bool:
    """True if the record contains only a FEW big peaks (an isolated pulse, or a
    pulse plus a second pulse), rather than the MANY comparable crests of an
    oscillating-noise window.

    The discriminator is the COUNT of peaks reaching osc_height (a large fraction
    of the main pulse).  A real pulse -> 1 peak.  A genuine pileup -> 2 peaks.
    Oscillating noise -> many peaks of comparable height.  So we accept up to
    max_peaks (default 3, allowing pulse + second pulse + a little slack) and
    reject anything busier as noise.  This keeps a real pulse on a noisy baseline
    (its oscillation crests are far below osc_height, so they aren't counted) and
    rejects a true oscillation (whose crests ARE near the window peak height).

    osc_height is half the main-pulse height for a TALL pulse.  For a SMALL pulse
    (a low-amplitude PMT pulse near the detection floor) half its height sinks into
    the NOISE band, where ordinary noise crests would be miscounted and the real
    pulse wrongly tagged as an oscillating record.  To prevent that, osc_height is
    floored at dom_floor_sigma * sigma -- above typical noise crests.  This is a
    no-op for pulses taller than 2*dom_floor_sigma*sigma (~8 sigma at the default),
    so tall-pulse behavior is unchanged; only small pulses are rescued.  A true
    oscillation still has many crests above the floor (-> NOISE)."""
    h = wf[peak] - baseline
    if h <= 0:
        return False
    osc_height = max(0.5 * h, dom_floor_sigma * sigma)
    # Fast path: if no sample OUTSIDE the main pulse's +/-distance neighborhood
    # reaches osc_height, every countable crest sits inside a 2*distance span,
    # where the peak-finder's distance suppression allows at most 3 peaks -- so
    # dominance holds whenever max_peaks >= 3, without running the peak search.
    if max_peaks >= 3:
        lo_n, hi_n = max(peak - distance, 0), min(peak + distance + 1, wf.size)
        out_max = max(wf[:lo_n].max(initial=-np.inf), wf[hi_n:].max(initial=-np.inf))
        if out_max - baseline < osc_height:
            return True
    n_big = _count_big_peaks(wf, baseline, osc_height, distance)
    return n_big <= max_peaks


def classify(waveforms: np.ndarray, baseline: float, sigma: float,
             pulse_lo: int, pulse_hi: int, pileup_prominence: float,
             noise_prominence: float, extra_frac: float, extra_min_sigma: float,
             min_separation: int, max_extra_pulses: int = 2,
             post_pulse_veto: int = 300, undershoot_sigma: float = 6.0,
             undershoot_window: int = 80, max_peaks: int = 4,
             dom_floor_sigma: float = 4.0
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Return (pileup_mask, noise_mask, pulse_peaks, info) -- all on raw data.

    For each event: find the highest peak in the pulse window; decide whether it
    is a real pulse; look for a second real pulse elsewhere (pileup); flag the
    event as noise if there is no real pulse in the window or the record is a
    pre-trigger-busy oscillation.

    A GENUINE pileup is the main pulse plus one (rarely two) isolated extra
    pulses.  Post-pulse ringing or a record oscillating throughout produces MANY
    crests that each pass the single-pulse shape test; such events have more than
    `max_extra_pulses` extras and are routed to NOISE (an oscillating record),
    not PILEUP -- they are not multiple particles and must not pollute the clean
    or pileup samples.
    """
    N, L = waveforms.shape
    extra_abs = extra_min_sigma * sigma
    undershoot_abs = undershoot_sigma * sigma
    # find_peaks needs only a lower-bound height; the exact per-event gate
    # (wf[p] - baseline >= extra_thresh, never below extra_abs) is applied below.
    # We floor the search at extra_abs rather than a separate, weaker
    # pileup_prominence*sigma height that the gate would always dominate.
    # pileup_prominence still governs the (topographic) prominence requirement,
    # which a height floor cannot replace.
    find_height = baseline + extra_abs

    # Vectorized pre-screen: an extra (pileup) candidate must lie OUTSIDE the pulse
    # window (in-window peaks are never extras) and clear find_height, so an event
    # whose outside-window maximum stays below find_height cannot yield any extra --
    # its peak search is skipped entirely (the common case on clean data).
    outside_max = np.maximum(waveforms[:, :pulse_lo].max(axis=1, initial=-np.inf),
                             waveforms[:, pulse_hi:].max(axis=1, initial=-np.inf))
    may_have_extra = outside_max >= find_height

    pileup = np.zeros(N, dtype=bool)
    noise = np.zeros(N, dtype=bool)
    pulse_peaks = np.zeros(N, dtype=np.int64)
    info: list[dict] = []

    for i in range(N):
        wf = waveforms[i]
        cand = _pulse_window_peak(wf, pulse_lo, pulse_hi)
        pulse_peaks[i] = cand

        # An event has a real coincidence pulse only if the pulse-window peak is
        # (a) a genuine pulse shape, and (b) DOMINATES the record -- towering over
        # any excursion elsewhere.  Oscillating-noise windows fail (b) because
        # their crests are comparable everywhere; a real pulse on a noisy baseline
        # still passes (b) because the pulse dwarfs the oscillation.  Either
        # failure -> noise.
        if (not _is_real_pulse(wf, cand, baseline, noise_prominence * sigma) or
                not _pulse_dominates(wf, cand, baseline, min_separation, sigma,
                                     max_peaks, dom_floor_sigma)):
            noise[i] = True
            info.append({"center_peak": None, "extra_peaks": []})
            continue

        # A real dominant pulse is present.  Look for a SECOND real pulse outside
        # the pulse window that is itself a sizable fraction of the main pulse --
        # a genuine second particle, not a tail ripple or a noise wiggle.
        peaks = (find_peaks_manual(wf, height=find_height, prominence=pileup_prominence * sigma,
                                   distance=min_separation)
                 if may_have_extra[i] else ())
        center_h = wf[cand] - baseline
        extra_thresh = max(extra_abs, extra_frac * center_h)
        extra = []
        for p in peaks:
            p = int(p)
            if (pulse_lo <= p < pulse_hi) or abs(p - cand) < min_separation:
                continue
            if (wf[p] - baseline) < extra_thresh:
                continue
            if not _is_real_pulse(wf, p, baseline, extra_thresh):
                continue
            # Veto extras in the post-pulse dead-time just AFTER the main pulse.
            # Afterpulsing, reflections and PMT ringing produce small crests within
            # a few hundred samples of the main peak; even though the baseline
            # recovers between them, they are not a second particle.  A genuine
            # second pulse arrives well clear of this window (or BEFORE the main
            # pulse, which is unaffected).
            if 0 < (p - cand) <= post_pulse_veto:
                continue
            # Require the baseline to have RECOVERED around the extra.  A genuine
            # second particle lands on a flat baseline (local min ~0); a crest of a
            # ringing/oscillating tail rides on a deep negative undershoot.  This
            # catches the long ringing of tall pulses that extends past the fixed
            # post_pulse_veto, and pre-pulse oscillation, without a per-amplitude
            # veto length.
            lo_w, hi_w = max(p - undershoot_window, 0), min(p + undershoot_window + 1, L)
            if (float(wf[lo_w:hi_w].min()) - baseline) < -undershoot_abs:
                continue
            extra.append(p)
        if len(extra) > max_extra_pulses:
            noise[i] = True            # a forest of crests = ringing / oscillation
        elif extra:
            pileup[i] = True           # one or two isolated extras = real pileup
        info.append({"center_peak": int(cand), "extra_peaks": extra})

    logger.info("Pileup: %d (%.1f%%).  Noise: %d (%.1f%%).",
                int(pileup.sum()), 100 * pileup.mean(),
                int(noise.sum()), 100 * noise.mean())
    return pileup, noise, pulse_peaks, info


# Exclusive class labels, in triage's priority order.
CLASS_LABELS = ("CLEAN", "SATURATED", "PILEUP", "NOISE")


def classify_events(waveforms: np.ndarray, baseline: float, sigma: float,
                    pulse_lo: int, pulse_hi: int, *,
                    saturation_adc: float | None = None,
                    pileup_prominence: float = 6.0, noise_prominence: float = 5.0,
                    extra_frac: float = 0.08, extra_min_sigma: float = 12.0,
                    rail_tol: float = 0.01, consec: int = 4, min_separation: int = 40,
                    max_extra_pulses: int = 2, post_pulse_veto: int = 300,
                    undershoot_sigma: float = 6.0, undershoot_window: int = 80,
                    max_peaks: int = 4, dom_floor_sigma: float = 4.0
                    ) -> tuple[np.ndarray, list[dict], float, np.ndarray]:
    """Per-event class label for an ORIENTED waveform array (pulses pointing UP).

    This is the single reusable entry point other tools should call to get the
    triage decision per event WITHOUT re-running file I/O.  It applies the real
    cuts (`detect_saturation` + `classify`) and resolves them to ONE label each
    using triage's exact priority PILEUP > SATURATED > NOISE > CLEAN.

    `waveforms` must already be oriented (see `orient_waveforms`) and `baseline`
    /`sigma` must come from `global_baseline_noise` on that same oriented array.

    Returns
    -------
    labels : (N,) array of str, each one of CLASS_LABELS.
    info   : per-event peak info (center_peak / extra_peaks), as `classify`.
    sat_thr: the saturation rail (ADC) actually used.
    peaks  : (N,) int, the pulse-window peak sample of EVERY event (argmax in the
             window, valid regardless of the event's label).
    """
    sat, sat_thr, _ = detect_saturation_cut(waveforms, saturation_adc,
                                            consec=consec, rail_tol=rail_tol)
    pileup, noise, peaks, info = classify(
        waveforms, baseline, sigma, pulse_lo, pulse_hi, pileup_prominence,
        noise_prominence, extra_frac, extra_min_sigma, min_separation,
        max_extra_pulses, post_pulse_veto, undershoot_sigma, undershoot_window,
        max_peaks, dom_floor_sigma)

    # Priority PILEUP > SATURATED > NOISE > CLEAN, identical to triage().
    sat_only = sat & ~pileup
    noise_only = noise & ~pileup & ~sat_only
    clean = ~pileup & ~sat_only & ~noise_only

    labels = np.empty(len(waveforms), dtype="<U9")
    labels[clean]      = "CLEAN"
    labels[sat_only]   = "SATURATED"
    labels[pileup]     = "PILEUP"
    labels[noise_only] = "NOISE"
    return labels, info, float(sat_thr), peaks


# ===========================================================================
# Diagnostics + plots (raw waveforms; subtract the global baseline only for display)
# ===========================================================================

def print_diagnostics(waveforms, baseline, sigma, sat, pileup, noise, clean, info,
                      polarity="positive") -> None:
    N, L = waveforms.shape
    row_max = waveforms.max(axis=1)
    print("\n" + "=" * 64)
    print("Waveform triage diagnostics")
    print("=" * 64)
    print(f"Source file:        {info.get('source')}")
    if "dataset" in info:
        print(f"Dataset:            {info['dataset']}")
    print(f"Events:             {N:,}")
    print(f"Window length:      {L} samples")
    print(f"Polarity:           {polarity}"
          f"{'  (reflected about baseline for analysis)' if polarity == 'negative' else ''}")
    print(f"Global baseline:    {baseline:.4g} ADC")
    print(f"Global noise sigma: {sigma:.4g} ADC")
    print(f"Peak ADC  min/med/max:  {row_max.min():.0f} / {np.median(row_max):.0f} / {row_max.max():.0f}")
    print("-" * 64)
    print(f"  SATURATED:  {int(sat.sum()):>7,}  ({100*sat.mean():5.1f}%)")
    print(f"  PILEUP:     {int(pileup.sum()):>7,}  ({100*pileup.mean():5.1f}%)")
    print(f"  NOISE:      {int(noise.sum()):>7,}  ({100*noise.mean():5.1f}%)")
    print(f"  CLEAN:      {int(clean.sum()):>7,}  ({100*clean.mean():5.1f}%)")
    print("=" * 64 + "\n")


def plot_overview(waveforms, baseline, sat, pileup, noise, clean,
                  saturation_adc, pulse_lo, pulse_hi, out_dir, show, save,
                  rail_found: bool = True) -> None:
    import matplotlib.pyplot as plt
    N, L = waveforms.shape
    row_max = waveforms.max(axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.hist(row_max, bins=200, color="C0", alpha=0.8)
    rail_label = (f"saturation = {saturation_adc:.0f}" if rail_found
                  else f"99.99th pct = {saturation_adc:.0f} (no rail; not cut)")
    ax.axvline(saturation_adc, ls="--", color="r", label=rail_label)
    ax.set_yscale("log")
    ax.set(xlabel="Peak ADC", ylabel="Events", title="Peak amplitude distribution")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    names = ["Clean", "Saturated", "Pileup", "Noise"]
    counts = [int(clean.sum()), int(sat.sum()), int(pileup.sum()), int(noise.sum())]
    ax.bar(names, counts, color=["C2", "C3", "C1", "C7"], alpha=0.85)
    for j, c in enumerate(counts):
        ax.text(j, c, f"{c:,}\n{100*c/N:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Events", title="Event classification")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(clean)
    if idx.size:
        for k in rng.choice(idx, min(200, idx.size), replace=False):
            ax.plot(waveforms[k] - baseline, color="C2", alpha=0.05, lw=0.6)
    ax.axvspan(pulse_lo, pulse_hi, color="gray", alpha=0.12, label="pulse window")
    ax.set(xlabel="Sample", ylabel="ADC (baseline-sub.)", title="Clean waveforms (sample of 200)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for mask, name, col in ((clean, "Clean", "C2"), (sat, "Saturated", "C3"),
                            (pileup, "Pileup", "C1"), (noise, "Noise", "C7")):
        idx = np.flatnonzero(mask)
        if idx.size:
            ax.plot(np.median(waveforms[idx] - baseline, axis=0), color=col, lw=1.5,
                    label=f"{name} (median)")
    ax.axvspan(pulse_lo, pulse_hi, color="gray", alpha=0.12)
    ax.set(xlabel="Sample", ylabel="ADC (baseline-sub.)", title="Median waveform per class")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Waveform triage overview", fontsize=14)
    fig.tight_layout()
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "triage_overview.png", dpi=200, bbox_inches="tight")
        logger.info("Saved %s", out_dir / "triage_overview.png")
    if show:
        plt.show()
    plt.close(fig)


def plot_gallery(waveforms, baseline, mask, info, title, fname_stem,
                 pulse_lo, pulse_hi, out_dir, show, save,
                 per_page: int = 12, n_pages: int = 1,
                 raw_waveforms=None, raw_adc: bool = False) -> None:
    """Browsable gallery of example waveforms from a class.  Interactive: tap
    LEFT / RIGHT (or n / p) to page through ALL events; Esc / Q to close.

    By default each trace is the ORIENTED analysis copy (pulses up) with the
    global baseline subtracted, matching what the cuts saw.  A RAW-ADC mode shows
    the ORIGINAL waveforms exactly as stored -- neither sign-flipped for negative
    polarity nor baseline-subtracted -- for inspecting the true recorded levels.
    Pass `raw_waveforms` (the original un-oriented array) to enable it; toggle it
    live with the `r` key, or start in it via `raw_adc=True`.  Only the DISPLAY
    changes; classification is untouched."""
    import matplotlib.pyplot as plt
    from plotting import paged_figure
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        logger.info("No events in class '%s'; skipping gallery.", title)
        return

    shuffled = np.random.default_rng(1).permutation(idx)
    n_all = int(np.ceil(len(shuffled) / per_page))
    n_save = min(n_pages, n_all)
    can_raw = raw_waveforms is not None
    view = {"raw": bool(raw_adc) and can_raw}

    def draw(fig, page):
        raw_on = view["raw"] and can_raw
        page_idx = shuffled[page * per_page:(page + 1) * per_page]
        rows = int(np.ceil(len(page_idx) / 3))
        axes = fig.subplots(rows, 3, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for k, ev in enumerate(page_idx):
            ax = axes[k // 3][k % 3]; ax.axis("on")
            trace = raw_waveforms[ev] if raw_on else waveforms[ev] - baseline
            ax.plot(trace, color="C0", lw=0.8)
            ax.axvspan(pulse_lo, pulse_hi, color="gray", alpha=0.1)
            if info is not None:
                cp = info[ev].get("center_peak")
                if cp is not None:
                    ax.axvline(cp, ls="-", color="C2", lw=0.8, alpha=0.6)
                for p in info[ev].get("extra_peaks", []):
                    ax.axvline(p, ls="--", color="C3", lw=1.0)
            ax.set_title(f"event {ev}", fontsize=8); ax.grid(True, alpha=0.3)
        mode = "raw ADC" if raw_on else "oriented, baseline-sub."
        hint = "← → or n/p to browse, " + ("r=display" if raw_on else "r=raw ADC, ") \
            if can_raw else "← → or n/p to browse, "
        fig.suptitle(f"{title}  (page {page + 1} / {n_all}  |  {mode}  |  "
                     f"{hint}Esc/Q to close)", fontsize=11)
        fig.tight_layout()

    if show:
        def _toggle_raw():
            view["raw"] = not view["raw"]

        paged_figure(n_all, draw, figsize=(14, 12),
                     extra_keys={"r": _toggle_raw} if can_raw else None)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        for page in range(n_save):
            fig = plt.figure(figsize=(14, 12)); draw(fig, page)
            suffix = f"_p{page + 1}" if n_pages > 1 else ""
            fig.savefig(out_dir / f"{fname_stem}{suffix}.png", dpi=200, bbox_inches="tight")
            logger.info("Saved %s", out_dir / f"{fname_stem}{suffix}.png")
            plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def read_row_aligned_aux(input_path, n_events: int) -> dict:
    """Read the per-event (row-aligned) metadata datasets from the triage input
    so they can be carried through, row-filtered, into each class subset file.

    Returns {name: (array, attrs_dict)} for every dataset whose first axis
    matches the event count -- the time axis (``event_time_unix``), the fine
    trigger-time-tag source (``headers_*``) and the parent-file row map
    (``source_event_index``).  Datasets not aligned to the event axis (e.g.
    ``selected_source_channels``) are skipped so a subset never carries a
    mis-length array.  A file with none of these (e.g. the CAEN path) yields {}.
    """
    import h5py
    with h5py.File(input_path, "r") as f:
        return _read_row_aligned_aux_from_file(f, n_events)


def _read_row_aligned_aux_from_file(f, n_events: int) -> dict:
    """read_row_aligned_aux for an already-open HDF5 file (so the streaming path,
    which holds the input file open for the whole run, does not have to reopen it
    -- reopening the same file read-only can trip HDF5 file locking)."""
    import h5py
    keep = ("source_event_index", "event_time_unix")
    aux: dict[str, tuple[np.ndarray, dict]] = {}
    for name, ds in f.items():
        if not isinstance(ds, h5py.Dataset):
            continue
        if name == "waveforms" or ds.shape[:1] != (n_events,):
            continue
        if name in keep or name.startswith("headers_"):
            aux[name] = (ds[()], dict(ds.attrs))
    if aux:
        logger.info("Carrying time axis into subsets: %s", ", ".join(sorted(aux)))
    return aux


def save_class(waveforms: np.ndarray, mask: np.ndarray, path: Path,
               polarity: str = "positive", aux: dict | None = None) -> None:
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    data = waveforms[mask].astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("waveforms", data=data,
                         compression="gzip", compression_opts=4, shuffle=True)
        # Carry the per-event time axis through, row-filtered by the same mask,
        # so the subset stays aligned with the parent run's recovered times
        # (re-join via /source_event_index) and remains time-aware on its own.
        for name, (arr, attrs) in (aux or {}).items():
            d = f.create_dataset(name, data=np.asarray(arr)[mask],
                                 compression="gzip", compression_opts=4, shuffle=True)
            for k, v in attrs.items():
                d.attrs[k] = v
        f.attrs["n_events"] = int(mask.sum())
        f.attrs["layout"]   = "waveforms[event, sample]"
        f.attrs["polarity"] = polarity
    logger.info("Wrote %d waveforms to %s", int(mask.sum()), path)


def triage(input_path, output_dir, saturation_adc, pulse_lo, pulse_hi,
           pileup_prominence, noise_prominence, extra_frac, extra_min_sigma,
           rail_tol, min_separation, max_extra_pulses, post_pulse_veto,
           undershoot_sigma, undershoot_window, max_peaks,
           gallery_pages, save_plots, show_plots, dry_run,
           polarity="positive", dom_floor_sigma=4.0, consec=4,
           gallery_classes=None, auto_window=False, window_coverage=0.99,
           run_diagnostics=True, full_diagnostics=False,
           gallery_raw_adc=False, export=False) -> dict:
    raw, info = load_waveforms(input_path)

    # The shared rough->refine recipe (validate window, provisional baseline,
    # resolve polarity + orient, auto-window, refined baseline + re-orient) --
    # see prepare_channel.  Every cut below runs on the ORIENTED `wf`; the
    # ORIGINAL `raw` is what gets written to disk.
    prep = prepare_channel(raw, polarity, pulse_lo, pulse_hi,
                           auto_window=auto_window, coverage=window_coverage)
    wf, baseline, sigma = prep.oriented, prep.baseline, prep.sigma
    pulse_lo, pulse_hi, polarity = prep.pulse_lo, prep.pulse_hi, prep.polarity

    # A user-supplied rail is given in raw ADC (e.g. the negative-pulse floor); map
    # it into the oriented frame so the flat-top detector sees it as an upper rail.
    if saturation_adc is not None and polarity == "negative":
        saturation_adc = 2.0 * baseline - saturation_adc

    sat, sat_thresh, rail_found = detect_saturation_cut(wf, saturation_adc,
                                                        consec=consec, rail_tol=rail_tol)
    if rail_found:
        top_pct = sat            # gallery shows exactly what was cut
    else:
        # No true ADC rail: the cut is already empty (no events removed, no
        # _saturated.h5 written).  The tallest events (peaks at/above the
        # 99.99th-pct fallback) are still collected so the gallery lets you
        # eyeball them for missed saturation.
        top_pct = wf.max(axis=1) >= sat_thresh
        logger.info("No true rail: saturation cut disabled; %d top-percentile "
                    "events kept for the gallery.", int(top_pct.sum()))
    pileup, noise_raw, _, peak_info = classify(
        wf, baseline, sigma, pulse_lo, pulse_hi, pileup_prominence,
        noise_prominence, extra_frac, extra_min_sigma, min_separation, max_extra_pulses,
        post_pulse_veto, undershoot_sigma, undershoot_window, max_peaks, dom_floor_sigma)

    # Priority: PILEUP > SATURATED > NOISE > CLEAN.  A clipped pulse with a real
    # second pulse is tagged pileup (the second pulse is recoverable physics).
    pileup_only = pileup
    sat_only = sat & ~pileup_only
    noise_only = noise_raw & ~pileup_only & ~sat_only
    clean = ~pileup_only & ~sat_only & ~noise_only

    print_diagnostics(wf, baseline, sigma, sat_only, pileup_only, noise_only, clean,
                      info, polarity)

    # `output_dir` is the already-resolved per-run results folder (see main); every output of
    # this run -- exported class files, plots and diagnostics -- lands in it, rather than
    # being dumped flat into preprocessing_results/.
    stem = Path(input_path).stem
    run_dir = Path(output_dir)
    if export and not dry_run:
        aux = read_row_aligned_aux(input_path, raw.shape[0])
        save_class(raw, clean,       run_dir / f"{stem}_clean.h5", polarity, aux)
        if rail_found:
            save_class(raw, sat_only, run_dir / f"{stem}_saturated.h5", polarity, aux)
        save_class(raw, pileup_only, run_dir / f"{stem}_pileup.h5", polarity, aux)
        save_class(raw, noise_only,  run_dir / f"{stem}_noise.h5", polarity, aux)
        # Everything that is NOT noise (clean + saturated + pileup) in one file.
        save_class(raw, ~noise_only, run_dir / f"{stem}_not_noise.h5", polarity, aux)

    # Plots/diagnostics use the ORIENTED copy so the pulse-window shading and peak
    # markers line up with what was actually classified (pulses shown pointing up).
    # `gallery_classes` selects which cut type(s) to draw a gallery for (default all).
    want = set(gallery_classes) if gallery_classes else set(GALLERY_CLASSES)
    if rail_found:
        sat_gallery = (sat_only, "Saturated examples")
    else:
        sat_gallery = (top_pct, "Tallest events (>= 99.99th pct; no rail -- NOT cut)")
    gallery_specs = [
        ("pileup",    pileup_only, peak_info, "Pileup examples (green=pulse, red=extra)", "gallery_pileup"),
        ("saturated", sat_gallery[0], None,   sat_gallery[1],                             "gallery_saturated"),
        ("noise",     noise_only,  peak_info, "Noise examples",                           "gallery_noise"),
        ("clean",     clean,       peak_info, "Clean examples",                           "gallery_clean"),
    ]
    if save_plots or show_plots:
        plot_overview(wf, baseline, sat_only, pileup_only, noise_only, clean,
                      sat_thresh, pulse_lo, pulse_hi, run_dir, show_plots, save_plots,
                      rail_found=rail_found)
        for cls, mask, pinfo, title, fstem in gallery_specs:
            if cls in want:
                plot_gallery(wf, baseline, mask, pinfo, title, fstem,
                             pulse_lo, pulse_hi, run_dir, show_plots, save_plots,
                             n_pages=gallery_pages, raw_waveforms=raw,
                             raw_adc=gallery_raw_adc)

    # Cut diagnostics (pulse_window.py).  On by default; --no-diagnostics skips.
    # Imported lazily to avoid a circular import (pulse_window imports this module).
    # The oriented copy is passed so the diagnostics match the classification above
    # (it is already pointing up, so pulse_window treats it as positive polarity),
    # and the ACTUAL cut values used here are passed through so the diagnostics
    # reproduce this run's classification (not the SiPM defaults).
    if run_diagnostics and not dry_run:
        try:
            import pulse_window
            pulse_window.run_all(wf, run_dir, show=show_plots,
                                 pulse_lo=pulse_lo, pulse_hi=pulse_hi,
                                 full=full_diagnostics,
                                 cut_overrides={
                                     "saturation_adc": saturation_adc,
                                     "rail_tol": rail_tol, "consec": consec,
                                     "noise_prominence": noise_prominence,
                                     "pileup_prominence": pileup_prominence,
                                     "extra_frac": extra_frac,
                                     "extra_min_sigma": extra_min_sigma,
                                     "min_separation": min_separation,
                                     "max_extra_pulses": max_extra_pulses,
                                     "post_pulse_veto": post_pulse_veto,
                                     "undershoot_sigma": undershoot_sigma,
                                     "undershoot_window": undershoot_window,
                                     "max_peaks": max_peaks,
                                     "dom_floor_sigma": dom_floor_sigma})
        except Exception as exc:                       # diagnostics must never break triage
            logger.warning("Diagnostics step failed: %s", exc)

    return {"n_total": len(raw), "n_clean": int(clean.sum()),
            "n_saturated": int(sat_only.sum()), "n_pileup": int(pileup_only.sum()),
            "n_noise": int(noise_only.sum()), "baseline": baseline, "noise_sigma": sigma,
            "polarity": polarity}


# ===========================================================================
# Streaming triage (memory-bounded path for large files)
# ===========================================================================
# The in-memory triage() above loads the WHOLE run as one array and holds an
# oriented copy -- fine for interactive/plotting use, but it caps the dataset at
# what fits in RAM.  The functions below do the identical classification block by
# block, so peak memory is O(block) instead of O(run): the baseline/noise/window/
# polarity/rail are estimated ONCE from a head sample, then each block is read,
# oriented, classified with the SAME classify()/detect_saturation(), and appended
# to the per-class output files.  Plots/diagnostics are not produced here (they
# need the whole array); use triage() for those.

STREAM_OUT_CHUNK = 1000          # output rows per HDF5 chunk (append granularity)


def _iter_waveform_blocks(ds, block_events: int, channel: int | None = None):
    """Yield (start, stop, block) with `block` a float32 (n, L) array read from an
    on-disk waveform dataset WITHOUT ever loading it whole.  A 3-D (N, n_ch, L)
    dataset is sliced to one channel per block (channel None -> channel 0), matching
    load_waveforms' single-channel behavior."""
    if ds.ndim == 2:
        n_events = ds.shape[0]
        for start in range(0, n_events, block_events):
            stop = min(start + block_events, n_events)
            yield start, stop, np.asarray(ds[start:stop], dtype=np.float32)
    elif ds.ndim == 3:
        n_events, n_ch = ds.shape[0], ds.shape[1]
        ch = 0 if channel is None else int(channel)
        if not (0 <= ch < n_ch):
            raise ValueError(f"channel {ch} out of range 0..{n_ch - 1} for {ds.shape}.")
        for start in range(0, n_events, block_events):
            stop = min(start + block_events, n_events)
            yield start, stop, np.asarray(ds[start:stop, ch, :], dtype=np.float32)
    else:
        raise ValueError(f"Expected 2-D or 3-D waveforms; got shape {ds.shape}.")


class _AppendWriter:
    """Growable, compressed per-class HDF5 writer for the streaming path: it creates
    /waveforms (plus any carried per-event aux datasets) lazily and appends each
    block's rows, so no class subset is ever held whole in memory.  Mirrors the
    format save_class writes (float32 waveforms, same compression, same attrs)."""

    def __init__(self, path: Path, length: int, polarity: str, aux: dict):
        import h5py
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "w")
        self.n = 0
        self.wf = self.f.create_dataset(
            "waveforms", shape=(0, length), maxshape=(None, length),
            chunks=(STREAM_OUT_CHUNK, length), dtype=np.float32,
            compression="gzip", compression_opts=4, shuffle=True)
        self.aux: dict = {}
        for name, (arr, attrs) in aux.items():
            tail = tuple(arr.shape[1:])
            d = self.f.create_dataset(
                name, shape=(0,) + tail, maxshape=(None,) + tail,
                chunks=(STREAM_OUT_CHUNK,) + tail, dtype=arr.dtype,
                compression="gzip", compression_opts=4, shuffle=True)
            for k, v in attrs.items():
                d.attrs[k] = v
            self.aux[name] = d
        self.f.attrs["layout"]   = "waveforms[event, sample]"
        self.f.attrs["polarity"] = polarity

    def append(self, rows: np.ndarray, aux_rows: dict) -> None:
        k = int(rows.shape[0])
        if k == 0:
            return
        self.wf.resize(self.n + k, axis=0)
        self.wf[self.n:self.n + k] = rows.astype(np.float32, copy=False)
        for name, d in self.aux.items():
            d.resize(self.n + k, axis=0)
            d[self.n:self.n + k] = aux_rows[name]
        self.n += k

    def close(self) -> None:
        self.f.attrs["n_events"] = self.n
        self.f.close()


def _print_stream_diagnostics(source, dataset, n, length, baseline, sigma, polarity,
                              counts, peak_min, peak_max, rail_thresh, rail_found,
                              pulse_lo, pulse_hi, n_sample) -> None:
    print("\n" + "=" * 64)
    print("Waveform triage diagnostics (streaming)")
    print("=" * 64)
    print(f"Source file:        {source}")
    print(f"Dataset:            {dataset}")
    print(f"Events:             {n:,}")
    print(f"Window length:      {length} samples")
    print(f"Pulse window:       [{pulse_lo}, {pulse_hi})")
    print(f"Prep sample:        first {n_sample:,} events"
          f"{'  (= whole file)' if n_sample >= n else ''}")
    print(f"Polarity:           {polarity}"
          f"{'  (reflected about baseline for analysis)' if polarity == 'negative' else ''}")
    print(f"Global baseline:    {baseline:.4g} ADC")
    print(f"Global noise sigma: {sigma:.4g} ADC")
    print(f"Peak ADC  min/max:  {peak_min:.0f} / {peak_max:.0f}")
    print(f"Saturation rail:    {rail_thresh:.0f}"
          f"{'' if rail_found else ' (no rail; not cut)'}")
    print("-" * 64)
    for label in ("SATURATED", "PILEUP", "NOISE", "CLEAN"):
        c = counts[label]
        print(f"  {label + ':':<11}{c:>10,}  ({(100 * c / n if n else 0.0):5.1f}%)")
    print("=" * 64 + "\n")


def stream_triage(input_path, output_dir, saturation_adc, pulse_lo, pulse_hi,
                  pileup_prominence, noise_prominence, extra_frac, extra_min_sigma,
                  rail_tol, min_separation, max_extra_pulses, post_pulse_veto,
                  undershoot_sigma, undershoot_window, max_peaks,
                  polarity="positive", dom_floor_sigma=4.0, consec=4,
                  auto_window=False, window_coverage=0.99, dry_run=False,
                  block_events=50000, sample_events=100000, channel=None,
                  export=False) -> dict:
    """Memory-bounded triage for large files -- see the section header above.

    Baseline/noise, polarity, pulse window and the saturation rail are estimated
    ONCE from the first `sample_events` events; when the file fits, that sample IS
    the whole file, so the result matches the in-memory triage() exactly.  Each
    block is then oriented and classified with the SAME classify()/
    detect_saturation() and its rows appended to the per-class output files.
    Returns the same summary dict as triage()."""
    import h5py
    stem = Path(input_path).stem
    run_dir = Path(output_dir)          # already-resolved per-run results folder (see main)

    with h5py.File(input_path, "r") as f:
        name, ds = _resolve_waveform_dataset(f)
        if ds.ndim == 2:
            n_events, length = int(ds.shape[0]), int(ds.shape[1])
        elif ds.ndim == 3:
            n_events, length = int(ds.shape[0]), int(ds.shape[2])
        else:
            raise ValueError(f"Expected 2-D or 3-D waveforms; got shape {ds.shape}.")
        logger.info("Streaming triage: dataset '%s' %s, %d events in blocks of %d.",
                    name, ds.shape, n_events, block_events)

        # --- Estimate the channel prep ONCE from a head sample ----------------
        n_sample = min(int(sample_events), n_events)
        if ds.ndim == 2:
            sample = np.asarray(ds[:n_sample], dtype=np.float32)
        else:
            ch = 0 if channel is None else int(channel)
            sample = np.asarray(ds[:n_sample, ch, :], dtype=np.float32)
        if n_sample < n_events:
            logger.info("Prep (baseline/noise/window/polarity/rail) estimated from the "
                        "first %d of %d events.", n_sample, n_events)

        prep = prepare_channel(sample, polarity, pulse_lo, pulse_hi,
                               auto_window=auto_window, coverage=window_coverage)
        baseline, sigma = prep.baseline, prep.sigma
        pulse_lo, pulse_hi, polarity = prep.pulse_lo, prep.pulse_hi, prep.polarity

        # Resolve the saturation rail (in the ORIENTED frame) once, from the sample,
        # then reuse that fixed rail on every block so the cut is global (never
        # re-auto-detected per block).  A user-supplied negative-pulse rail is mapped
        # into the oriented frame exactly as triage() does.
        if saturation_adc is not None and polarity == "negative":
            saturation_adc = 2.0 * baseline - saturation_adc
        _, rail_thresh, rail_found = detect_saturation_cut(
            prep.oriented, saturation_adc, consec=consec, rail_tol=rail_tol)
        del sample, prep

        aux = _read_row_aligned_aux_from_file(f, n_events)

        counts = {"CLEAN": 0, "SATURATED": 0, "PILEUP": 0, "NOISE": 0}
        peak_min, peak_max = np.inf, -np.inf

        writers: dict[str, _AppendWriter] = {}
        if export and not dry_run:
            writers["clean"] = _AppendWriter(run_dir / f"{stem}_clean.h5", length, polarity, aux)
            if rail_found:
                writers["saturated"] = _AppendWriter(run_dir / f"{stem}_saturated.h5", length, polarity, aux)
            writers["pileup"]    = _AppendWriter(run_dir / f"{stem}_pileup.h5", length, polarity, aux)
            writers["noise"]     = _AppendWriter(run_dir / f"{stem}_noise.h5", length, polarity, aux)
            writers["not_noise"] = _AppendWriter(run_dir / f"{stem}_not_noise.h5", length, polarity, aux)

        try:
            for start, stop, raw_block in _iter_waveform_blocks(ds, block_events, channel):
                wf = orient_waveforms(raw_block, polarity, baseline)
                row_max = wf.max(axis=1)
                peak_min = min(peak_min, float(row_max.min()))
                peak_max = max(peak_max, float(row_max.max()))

                if rail_found:
                    sat, _, _ = detect_saturation(wf, rail_thresh, consec=consec, rail_tol=rail_tol)
                else:
                    sat = np.zeros(len(wf), dtype=bool)
                pileup, noise_raw, _, _ = classify(
                    wf, baseline, sigma, pulse_lo, pulse_hi, pileup_prominence,
                    noise_prominence, extra_frac, extra_min_sigma, min_separation,
                    max_extra_pulses, post_pulse_veto, undershoot_sigma,
                    undershoot_window, max_peaks, dom_floor_sigma)

                # Priority PILEUP > SATURATED > NOISE > CLEAN, identical to triage().
                pileup_only = pileup
                sat_only = sat & ~pileup_only
                noise_only = noise_raw & ~pileup_only & ~sat_only
                clean = ~pileup_only & ~sat_only & ~noise_only

                counts["CLEAN"]     += int(clean.sum())
                counts["SATURATED"] += int(sat_only.sum())
                counts["PILEUP"]    += int(pileup_only.sum())
                counts["NOISE"]     += int(noise_only.sum())

                if export and not dry_run:
                    aux_all = {nm: arr[start:stop] for nm, (arr, _) in aux.items()}
                    sub = lambda m: {nm: a[m] for nm, a in aux_all.items()}   # noqa: E731
                    writers["clean"].append(raw_block[clean], sub(clean))
                    if rail_found:
                        writers["saturated"].append(raw_block[sat_only], sub(sat_only))
                    writers["pileup"].append(raw_block[pileup_only], sub(pileup_only))
                    writers["noise"].append(raw_block[noise_only], sub(noise_only))
                    writers["not_noise"].append(raw_block[~noise_only], sub(~noise_only))

                logger.info("  events %d-%d classified.", start, stop - 1)
        finally:
            for w in writers.values():
                w.close()

    if peak_min == np.inf:                              # empty file guard
        peak_min = peak_max = 0.0
    _print_stream_diagnostics(input_path, name, n_events, length, baseline, sigma,
                              polarity, counts, peak_min, peak_max, rail_thresh,
                              rail_found, pulse_lo, pulse_hi, n_sample)

    return {"n_total": n_events, "n_clean": counts["CLEAN"],
            "n_saturated": counts["SATURATED"], "n_pileup": counts["PILEUP"],
            "n_noise": counts["NOISE"], "baseline": baseline, "noise_sigma": sigma,
            "polarity": polarity}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Triage hodoscope waveforms into clean / saturated / pileup / noise.")
    p.add_argument("--input", type=Path, required=True,
                   help="Input .h5 / .hdf5 waveform file.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Base output directory; outputs go into "
                        "<output-dir>/<input-stem>_triage_results[_N]/. Default base: "
                        "preprocessing/preprocessing_results/ (a re-run gets a fresh _N folder).")
    p.add_argument("--saturation-adc", type=float, default=None,
                   help="ADC rail for saturation. Default: auto-detect.")
    p.add_argument("--detector", choices=["sipm", "pmt"], default="sipm",
                   help="Detector readout preset. 'sipm' (default): tall, slow, positive pulses "
                        "that can clip the rail. 'pmt': small, FAST, negative pulses that rarely "
                        "saturate -> sets polarity=negative and the recommended PMT cuts (higher "
                        "--noise-prominence, lower --consec, smaller --min-separation). Any of "
                        "these can still be overridden explicitly on the command line.")
    p.add_argument("--polarity", choices=["positive", "negative", "auto"], default=None,
                   help="Pulse polarity. Default follows --detector (sipm->positive, pmt->negative). "
                        "'negative' reflects the record about its baseline for analysis; 'auto' "
                        "detects it from the data (95th-percentile excursion vote).")
    p.add_argument("--pulse-lo", type=int, default=366,
                   help="Start of the pulse window (samples). Default: 366. Used as the provisional "
                        "window for auto-window; the fixed window only with --no-auto-window.")
    p.add_argument("--pulse-hi", type=int, default=726,
                   help="End of the pulse window (samples). Default: 726. Used as the provisional "
                        "window for auto-window; the fixed window only with --no-auto-window.")
    p.add_argument("--auto-window", dest="auto_window", action="store_true", default=True,
                   help="Derive --pulse-lo/--pulse-hi from the data so the window CONTAINS the real "
                        "pulses (fixes pulses peaking just outside a hand-set window being misread as "
                        "NOISE). ON by default; disable with --no-auto-window.")
    p.add_argument("--no-auto-window", dest="auto_window", action="store_false",
                   help="Use the fixed --pulse-lo/--pulse-hi instead of the data-derived window.")
    p.add_argument("--window-coverage", type=float, default=0.99,
                   help="Fraction of real-pulse peak positions the auto-window must span. Higher "
                        "captures more of the timing tails. Default: 0.99.")
    p.add_argument("--pileup-prominence", type=float, default=6.0,
                   help="Topographic prominence (in noise sigma) a second pulse must stand out "
                        "by; its height floor is set by --extra-min-sigma. Default: 6.")
    p.add_argument("--noise-prominence", type=float, default=None,
                   help="Min pulse-window height in noise sigma; below this -> noise. "
                        "Default: 5 (sipm) / 6 (pmt).")
    p.add_argument("--extra-frac", type=float, default=0.08,
                   help="A pileup extra pulse must be >= this fraction of the main pulse. Default: 0.08.")
    p.add_argument("--extra-min-sigma", type=float, default=12.0,
                   help="A pileup extra pulse must also exceed this many noise sigma. Default: 12.")
    p.add_argument("--rail-tol", type=float, default=0.01,
                   help="Flat-top saturation requires samples within this fraction of the rail. Default: 0.01.")
    p.add_argument("--consec", type=int, default=None,
                   help="Saturation flat-top requires this many CONSECUTIVE samples at the rail. "
                        "Default: 4 (sipm) / 2 (pmt -- a fast pulse clips for fewer samples).")
    p.add_argument("--min-separation", type=int, default=None,
                   help="Minimum sample separation between the main and extra pulse (>= pulse rise "
                        "time so one pulse is not split into multiple peaks). Default: 40 (sipm) / "
                        "25 (pmt -- narrower pulse).")
    p.add_argument("--max-extra-pulses", type=int, default=2,
                   help="More than this many extra pulses -> noise (oscillation), not pileup. Default: 2.")
    p.add_argument("--max-peaks", type=int, default=4,
                   help="Record dominance: more than this many peaks above the comparable-crest "
                        "height -> noise (oscillating record). Default: 4.")
    p.add_argument("--dom-floor-sigma", type=float, default=4.0,
                   help="Dominance comparable-crest height is floored at this many noise sigma, so "
                        "small (e.g. low-amplitude PMT) pulses are not mistaken for oscillating "
                        "noise. No effect on pulses taller than ~2x this (in sigma). Default: 4.")
    p.add_argument("--post-pulse-veto", type=int, default=300,
                   help="Samples after the main peak in which an extra pulse is treated as "
                        "afterpulsing/ringing, not pileup. Extras BEFORE the main pulse and "
                        "beyond this window are unaffected. Default: 300.")
    p.add_argument("--undershoot-sigma", type=float, default=6.0,
                   help="An extra (pileup) pulse is rejected if the baseline near it dips below "
                        "this many noise sigma -- a sign it rides on a ringing/oscillating tail "
                        "rather than a recovered baseline. Default: 6.")
    p.add_argument("--undershoot-window", type=int, default=80,
                   help="Half-width (samples) of the window around an extra pulse checked for "
                        "the undershoot above. Default: 80.")
    p.add_argument("--gallery", nargs="+", default=["all"],
                   choices=["all", "clean", "saturated", "pileup", "noise"],
                   help="Which cut type(s) to show/save a gallery for. Default: all. "
                        "e.g. --gallery noise   or   --gallery pileup saturated.")
    p.add_argument("--gallery-pages", type=int, default=1,
                   help="Gallery pages to SAVE per class (interactive browses all). Default: 1.")
    p.add_argument("--gallery-raw-adc", action="store_true",
                   help="Start the gallery in RAW-ADC mode: show the original waveforms as "
                        "stored (NOT sign-flipped for negative polarity, NOT baseline-subtracted) "
                        "instead of the oriented, baseline-subtracted analysis view. Interactive "
                        "galleries can also toggle this live with the 'r' key. Display only -- the "
                        "cuts are unchanged.")
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--diagnostics", dest="run_diagnostics", action="store_true",
                   help="Run the pulse_window.py cut diagnostics after triage, saving "
                        "diag_*.png to the output dir. Off by default.")
    p.add_argument("--no-diagnostics", dest="run_diagnostics", action="store_false",
                   help="Explicitly skip the cut diagnostics (this is the default).")
    p.set_defaults(run_diagnostics=False)
    p.add_argument("--full-diagnostics", action="store_true",
                   help="Also run the opt-in (exploratory) sculpted-spectrum diagnostic. The core "
                        "QA diagnostics (window / cutflow) always run. The parameter scan and the "
                        "local-vs-global baseline study are ad-hoc pulse_window.py sub-commands.")
    p.add_argument("--stream", choices=["auto", "on", "off"], default="auto",
                   help="Memory-bounded block-streaming triage for large files: the run is "
                        "classified and written block by block instead of loaded whole. "
                        "'auto' (default): stream when no plots/diagnostics are requested, "
                        "else load in memory. 'on': force streaming (plots/diagnostics are "
                        "skipped -- they need the whole array). 'off': always load in memory.")
    p.add_argument("--block-events", type=int, default=50000,
                   help="Events per block in streaming mode (peak memory ~ block x length). "
                        "Default: 50000.")
    p.add_argument("--sample-events", type=int, default=100000,
                   help="Head events used to estimate baseline/noise/window/polarity/rail in "
                        "streaming mode. When >= the event count the sample is the whole file "
                        "and the result matches the in-memory path. Default: 100000.")
    p.add_argument("--export", action="store_true",
                   help="Write the per-class .h5 subset files (<stem>_clean/_saturated/_pileup/"
                        "_noise/_not_noise.h5) to the output dir. OFF by default: triage just "
                        "classifies and reports (and draws plots if asked) without writing any "
                        "subset files. Downstream tools (energy_reconstruction) re-run the "
                        "classification in-memory and do not read these files, so exporting is "
                        "only needed when you want the subsets on disk.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # --full-diagnostics is meaningless without the diagnostics running, so it
    # implies --diagnostics (which is otherwise off by default).
    if args.full_diagnostics:
        args.run_diagnostics = True
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")

    # Bare filename -> waveform_files/, including its per-run folders.
    input_path = resolve_input(args.input)

    # Resolve the per-run results folder: preprocessing_results/<stem>_triage_results[_N]
    # (or <output-dir>/<stem>_triage_results[_N] if --output-dir is given).  All of
    # this run's outputs -- exported class files, plots, diagnostics -- go here.
    output_dir = resolve_results_dir(__file__, input_path.stem,
                                     base=args.output_dir, program="triage")

    # Apply the detector preset for any preset-controlled value left unset on the CLI.
    preset = DETECTOR_PRESETS[args.detector]
    polarity         = args.polarity         if args.polarity         is not None else preset["polarity"]
    noise_prominence = args.noise_prominence if args.noise_prominence is not None else preset["noise_prominence"]
    min_separation   = args.min_separation   if args.min_separation   is not None else preset["min_separation"]
    consec           = args.consec           if args.consec           is not None else preset["consec"]
    if args.detector == "pmt":
        logger.info("PMT preset: polarity=%s noise_prominence=%g consec=%d min_separation=%d "
                    "(override any explicitly on the CLI).",
                    polarity, noise_prominence, consec, min_separation)

    gallery_classes = (set(GALLERY_CLASSES) if "all" in args.gallery else set(args.gallery))

    # Streaming (memory-bounded) vs in-memory triage.  Plots/diagnostics need the
    # whole array, so 'auto' streams exactly when none of them are requested; 'on'
    # forces streaming (and skips any requested plots with a warning); 'off' always
    # loads in memory.
    show_plots = not args.no_show
    want_plots = show_plots or args.save_plots or args.run_diagnostics
    if args.stream == "on":
        use_stream = True
    elif args.stream == "off":
        use_stream = False
    else:
        use_stream = not want_plots

    if use_stream:
        if want_plots:
            logger.warning("Streaming mode does not produce plots/diagnostics; skipping them "
                           "(use --stream off for the in-memory plotting path).")
        stream_triage(
            input_path=input_path, output_dir=output_dir,
            saturation_adc=args.saturation_adc, pulse_lo=args.pulse_lo, pulse_hi=args.pulse_hi,
            pileup_prominence=args.pileup_prominence, noise_prominence=noise_prominence,
            extra_frac=args.extra_frac, extra_min_sigma=args.extra_min_sigma,
            rail_tol=args.rail_tol, min_separation=min_separation,
            max_extra_pulses=args.max_extra_pulses, post_pulse_veto=args.post_pulse_veto,
            undershoot_sigma=args.undershoot_sigma, undershoot_window=args.undershoot_window,
            max_peaks=args.max_peaks, polarity=polarity, dom_floor_sigma=args.dom_floor_sigma,
            consec=consec, auto_window=args.auto_window, window_coverage=args.window_coverage,
            dry_run=args.dry_run, block_events=args.block_events,
            sample_events=args.sample_events, export=args.export)
        return

    triage(input_path=input_path, output_dir=output_dir,
           saturation_adc=args.saturation_adc, pulse_lo=args.pulse_lo, pulse_hi=args.pulse_hi,
           pileup_prominence=args.pileup_prominence, noise_prominence=noise_prominence,
           extra_frac=args.extra_frac, extra_min_sigma=args.extra_min_sigma,
           rail_tol=args.rail_tol, min_separation=min_separation,
           max_extra_pulses=args.max_extra_pulses, post_pulse_veto=args.post_pulse_veto,
           undershoot_sigma=args.undershoot_sigma, undershoot_window=args.undershoot_window,
           max_peaks=args.max_peaks,
           gallery_pages=args.gallery_pages, save_plots=args.save_plots,
           show_plots=not args.no_show, dry_run=args.dry_run,
           polarity=polarity, dom_floor_sigma=args.dom_floor_sigma, consec=consec,
           gallery_classes=gallery_classes, auto_window=args.auto_window,
           window_coverage=args.window_coverage, run_diagnostics=args.run_diagnostics,
           full_diagnostics=args.full_diagnostics, gallery_raw_adc=args.gallery_raw_adc,
           export=args.export)


if __name__ == "__main__":
    main()