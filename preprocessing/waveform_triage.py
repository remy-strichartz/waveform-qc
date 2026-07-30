#!/usr/bin/env python3
"""Standalone triage tool for single-channel waveform windows.

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

import numpy as np

# Direct execution (`python preprocessing/waveform_triage.py`) needs the project root on
# the path; under an editable install (pip install -e .) this line is a no-op.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.output_paths import resolve_input, resolve_results_dir      # noqa: E402
from common.waveform_ops import (DETECTOR_PRESETS, _resolve_waveform_dataset,  # noqa: E402
                                 classify, detect_saturation, detect_saturation_cut,
                                 load_waveforms, orient_waveforms, prepare_channel)

logger = logging.getLogger("waveform_triage")

# The four per-class galleries --export / --gallery can draw.  A presentation concept
# (which subsets get a figure), not a cut -- so it stays with the driver.
GALLERY_CLASSES = ("clean", "saturated", "pileup", "noise")


# ===========================================================================
# Diagnostics + plots (raw waveforms; subtract the global baseline only for display)
# ===========================================================================

def print_diagnostics(source, dataset, n_events, length, baseline, sigma, polarity,
                      counts, peak_min, peak_max, rail_thresh, rail_found,
                      pulse_lo, pulse_hi, peak_med=None, n_sample=None) -> None:
    """The ONE triage summary block -- both the in-memory triage() and the streaming
    stream_triage() print through here, so the two paths report the same fields.

    The PULSE WINDOW and the SATURATION RAIL are part of the summary, not just the
    INFO log: auto-window is on by default and the rail is auto-detected, so the
    window and rail actually used are results of the run, and a summary that omits
    them cannot be read on its own.

    `peak_med` is optional (the streaming path tracks only a running min/max, never
    holding the run whole); `n_sample` is the streaming path's prep-sample size and
    marks the block as streaming when given.
    """
    print("\nWaveform triage" + ("  (streaming)" if n_sample is not None else ""))
    print(f"Source file:        {source}")
    if dataset:
        print(f"Dataset:            {dataset}")
    print(f"Events:             {n_events:,}")
    print(f"Window length:      {length} samples")
    print(f"Pulse window:       [{pulse_lo}, {pulse_hi})")
    if n_sample is not None:
        print(f"Prep sample:        first {n_sample:,} events"
              f"{'  (= whole file)' if n_sample >= n_events else ''}")
    print(f"Polarity:           {polarity}"
          f"{'  (reflected about baseline for analysis)' if polarity == 'negative' else ''}")
    print(f"Global baseline:    {baseline:.4g} ADC")
    print(f"Global noise sigma: {sigma:.4g} ADC")
    if peak_med is None:
        print(f"Peak ADC  min/max:      {peak_min:.0f} / {peak_max:.0f}")
    else:
        print(f"Peak ADC  min/med/max:  {peak_min:.0f} / {peak_med:.0f} / {peak_max:.0f}")
    print(f"Saturation rail:    {rail_thresh:.0f}"
          f"{'' if rail_found else '  (no rail; not cut)'}")
    print("-" * 64)
    for label in ("SATURATED", "PILEUP", "NOISE", "CLEAN"):
        c = int(counts[label])
        print(f"  {label + ':':<11}{c:>10,}  ({(100 * c / n_events if n_events else 0.0):5.1f}%)")
    print()


def plot_overview(waveforms, baseline, sat, pileup, noise, clean,
                  saturation_adc, pulse_lo, pulse_hi, out_dir, show, save,
                  rail_found: bool = True) -> None:
    import matplotlib.pyplot as plt
    N, L = waveforms.shape
    row_max = waveforms.max(axis=1)
    # Three panels, each showing something the printed summary cannot: where the rail
    # sits in the peak distribution, the event-level spread of the clean pulses, and
    # how the four classes separate in shape.  (There is deliberately no class-count
    # bar chart: print_diagnostics already tabulates those four numbers, and a bar
    # chart annotated with them adds nothing -- a rare class like PILEUP at 0.0%
    # renders as an invisible sliver, so the table is strictly the better view.)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.hist(row_max, bins=200, color="C0", alpha=0.8)
    rail_label = (f"saturation = {saturation_adc:.0f}" if rail_found
                  else f"99.99th pct = {saturation_adc:.0f} (no rail; not cut)")
    ax.axvline(saturation_adc, ls="--", color="r", label=rail_label)
    ax.set_yscale("log")
    ax.set(xlabel="Peak ADC", ylabel="Events", title="Peak amplitude distribution")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(clean)
    if idx.size:
        for k in rng.choice(idx, min(200, idx.size), replace=False):
            ax.plot(waveforms[k] - baseline, color="C2", alpha=0.05, lw=0.6)
    ax.axvspan(pulse_lo, pulse_hi, color="gray", alpha=0.12, label="pulse window")
    ax.set(xlabel="Sample", ylabel="ADC (baseline-sub.)", title="Clean waveforms (sample of 200)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
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
    from common.plotting import paged_figure
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

    row_max = wf.max(axis=1)
    print_diagnostics(
        info.get("source"), info.get("dataset"), len(raw), wf.shape[1], baseline, sigma,
        polarity,
        {"SATURATED": int(sat_only.sum()), "PILEUP": int(pileup_only.sum()),
         "NOISE": int(noise_only.sum()), "CLEAN": int(clean.sum())},
        float(row_max.min()), float(row_max.max()), sat_thresh, rail_found,
        pulse_lo, pulse_hi, peak_med=float(np.median(row_max)))

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

    # Cut diagnostics (pulse_window.py).  The CLI leaves them OFF (--diagnostics
    # opts in); the True default here serves programmatic callers only.
    # Imported lazily because the diagnostics are optional and pulse_window pulls in
    # matplotlib; a --no-diagnostics run should not pay for it.  (Until the common/
    # extraction this was lazy for a harder reason -- pulse_window imported THIS module,
    # so the two were circular.  Both now take the primitives from common.waveform_ops.)
    # The oriented copy is passed so the diagnostics match the classification above
    # (it is already pointing up, so pulse_window treats it as positive polarity),
    # and the ACTUAL cut values used here are passed through so the diagnostics
    # reproduce this run's classification (not the SiPM defaults).
    if run_diagnostics and not dry_run:
        try:
            from preprocessing import pulse_window
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
    print_diagnostics(input_path, name, n_events, length, baseline, sigma, polarity,
                      counts, peak_min, peak_max, rail_thresh, rail_found,
                      pulse_lo, pulse_hi, n_sample=n_sample)

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
                        "preprocessing/preprocessing_results/triage/ (a re-run gets a fresh "
                        "_N folder unless --overwrite).")
    p.add_argument("--overwrite", action="store_true",
                   help="Write into the canonical (un-suffixed) results folder, replacing "
                        "that run's files in place, instead of creating a fresh _N folder. "
                        "Use when re-running a set of channels and expecting exactly one "
                        "folder per channel (same convention as energy_reconstruction).")
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

    # Bare filename -> waveform_files/, wherever its dataset folder keeps it.
    input_path = resolve_input(args.input)

    # Resolve the per-run results folder: preprocessing_results/triage/<stem>_triage_results[_N]
    # (or <output-dir>/<stem>_triage_results[_N] if --output-dir is given).  All of
    # this run's outputs -- exported class files, plots, diagnostics -- go here.
    output_dir = resolve_results_dir(__file__, input_path.stem,
                                     base=args.output_dir, program="triage", group="triage",
                                     overwrite=args.overwrite)

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
