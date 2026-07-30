#!/usr/bin/env python3
"""Measure the muon detection efficiency of the MIDDLE panel of a 3-panel
scintillator telescope (top / middle / bottom) by the standard telescope method.

Method
------
A muon that fires BOTH the top and the bottom panel (in time coincidence) must
have passed through the middle panel as well.  So:

    denominator ("muon crossed") = events where TOP and BOTTOM both registered a
                                    real pulse
    numerator   ("middle hit")   = those events where the MIDDLE panel ALSO
                                    registered a real pulse
    efficiency  = numerator / denominator        (Wilson score CI, see below)

"Registered a real pulse" is decided per event by the shared triage classifier
(common.waveform_ops.classify_events): a panel SAW a particle if its class is CLEAN,
SATURATED, or PILEUP -- i.e. anything that is NOT pure NOISE.  Saturated and
pile-up events are still real crossings (the panel did detect a particle, the
charge is merely clipped or accompanied by a second pulse), so they count; only
NOISE (no real pulse in the window) is a miss.  The per-panel class breakdown is
printed and plotted so the saturated / pile-up contributions stay auditable.

Inputs
------
Three raw HDF5 waveform-window files (top, middle, bottom), ROW-ALIGNED: row i of
each file is the same AND-triggered event.  The files may be a fixed number of
samples time-shifted relative to one another (cable / time-of-flight offset, a few
samples with event-to-event variance).  That offset is far smaller than the
pulse window, so it does not bias the classification; the top-bottom Delta-t
histogram measures it from the data.

Detector types (two supported stand configurations):
  (1) PMT top/bottom + SiPM middle   ->  --tb-readout pmt   --middle-readout sipm
  (2) SiPM top/bottom/middle         ->  (defaults: all sipm)
SiPM pulses are positive-going, PMT pulses negative-going; each panel is oriented
accordingly (the shared polarity handling) before classification.

Diagnostics (saved as eff_*.png, alongside the project's diag_*/maker_* figures)
  * eff_dt_coincidence.png  -- top-bottom Delta-t histogram (validates the offset),
                               over events where BOTH panels saw a pulse, width quoted
                               as a robust MAD sigma
  * eff_noise_residuals.png -- per-panel pre-pulse noise-residual overlay (integer-aligned
                               bins, MAD sigma vs RMS, non-Gaussian panels flagged)
  * eff_sensitivity.png     -- efficiency vs a swept cut parameter (with CI band); the
                               sweep is a SYSTEMATIC and runs even without plots
The per-panel class breakdown is PRINTED (print_report), not plotted -- a stacked bar of
that same table hid exactly the small classes (SATURATED / PILEUP) it needed to show.

Usage
-----
    # three separate single-channel files
    python hodoscope_efficiency.py --top run_top.h5 --middle run_mid.h5 --bottom run_bot.h5

    # one multi-channel file extracted as [top, middle, bottom] (e.g. channels 9,0,10),
    # PMT top/bottom + SiPM middle, pulse near sample 440:
    python hodoscope_efficiency.py --input run00270_ch-9-0-10.h5 \
        --top-channel 0 --middle-channel 1 --bottom-channel 2 \
        --tb-readout pmt --middle-readout sipm --save-plots
    # Each panel's window is derived from its own data by default (auto-window); this
    # fixes real pulses peaking just outside a hand-set window being misread as NOISE.
    # Pass --no-auto-window with --pulse-lo/--pulse-hi to force one fixed window.

    # sweep a cut to check the efficiency is not threshold-sensitive
    python hodoscope_efficiency.py --top t.h5 --middle m.h5 --bottom b.h5 \
        --sensitivity-param noise_prominence --sens-lo 3 --sens-hi 8 --sens-n 6
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (see README)

# The real triage classifier (so the cut logic is shared, not duplicated), the shared
# waveform-file and per-run results-folder conventions, and the DAQ dead-time bound.
from common.output_paths import (dataset_of, find_related,          # noqa: E402
                                 resolve_input, resolve_results_dir)
from common.plotting import finish_figure, setup_mpl as _setup_mpl  # noqa: E402
from common.timing_ops import dead_time_bound                       # noqa: E402
from common import waveform_ops as ops                               # noqa: E402

logger = logging.getLogger("hodoscope_efficiency")

# A panel "saw a particle" if its triage class is anything but NOISE.
HIT_CLASSES = ("CLEAN", "SATURATED", "PILEUP")


def dead_time_systematic(input_path: Path, times_path: Path | None) -> dict | None:
    """The DAQ dead-time contribution to veto inefficiency, or None if this run has no
    recoverable time axis.

    This is a genuine systematic on the number the script reports, and it is not
    measurable from the waveforms: a muon that arrives while the DAQ is dead is simply
    never recorded, so it is missing from BOTH the numerator and the denominator and the
    offline efficiency cannot see it.  Only the arrival times can bound it -- a dead time
    tau removes every inter-arrival interval below tau, so the smallest interval actually
    observed is a hard upper bound on it (common/timing_ops.dead_time_bound).

    The bound is quoted alongside the statistical CI so the reader can see which one
    dominates -- "no correction warranted" is worth stating explicitly rather than
    leaving unexamined."""
    import h5py

    def _times(path: Path) -> np.ndarray | None:
        if path is None or not path.exists():
            return None
        with h5py.File(path, "r") as f:
            if "event_time_rel_s" not in f:
                return None
            return np.asarray(f["event_time_rel_s"][()], dtype=np.float64)

    t = _times(input_path)                                  # written at conversion
    if t is None:
        t = _times(times_path)                              # explicit --times
    if t is None:                                           # event_times.py's output
        t = _times(find_related(input_path, f"{input_path.stem}_times.h5"))
    if t is None:
        # The run's PARENT multi-channel file: arrival times are a RUN property
        # (identical for every channel and every extraction of the same rows), and
        # only interval statistics are used here -- row alignment with THIS file is
        # not required.  This is what serves the canonical run00270_ch9-0-10 input,
        # whose extraction predates the converter-written axis: without this
        # fallback the documented "<0.03%" dead-time block silently vanished.
        t = _times(find_related(input_path, f"{dataset_of(input_path.name)}.h5"))
    if t is None or t.size < 3:
        return None

    return dead_time_bound(t)


def _readout_polarity(readout: str) -> str:
    """Map a detector readout type to the pulse polarity triage expects."""
    r = readout.lower()
    if r == "sipm":
        return "positive"
    if r == "pmt":
        return "negative"
    if r == "auto":
        return "auto"
    raise ValueError(f"readout must be sipm/pmt/auto, got {readout!r}")


# ===========================================================================
# Wilson score confidence interval (implemented from scratch)
# ===========================================================================

@dataclass
class Efficiency:
    """A binomial efficiency k/n with a Wilson score confidence interval."""
    k: int                 # hits (numerator)
    n: int                 # trials (denominator)
    z: float               # standard-normal quantile (1.96 -> 95%)
    point: float           # k / n
    lo: float              # Wilson lower bound
    hi: float              # Wilson upper bound

    @property
    def err_lo(self) -> float:
        return abs(self.point - self.lo)      # abs() avoids a -0.00 at the bounds

    @property
    def err_hi(self) -> float:
        return abs(self.hi - self.point)

    def __str__(self) -> str:
        cl = 100.0 * (1.0 - 2.0 * (1.0 - _phi(self.z)))   # confidence level from z
        return (f"{100*self.point:.2f}%  (+{100*self.err_hi:.2f} / -{100*self.err_lo:.2f}, "
                f"{cl:.0f}% Wilson CI)   [{self.k:,}/{self.n:,}]")


def _phi(z: float) -> float:
    """Standard-normal CDF (for reporting the confidence level a z implies)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilson_interval(k: int, n: int, z: float = 1.96) -> Efficiency:
    """Wilson score interval for a binomial proportion k successes in n trials.

    Closed form (Wilson 1927), which -- unlike the naive normal interval -- stays
    inside [0, 1] and behaves well for efficiencies near 0/1 and small n:

        center = (p + z^2/2n) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/4n^2 )

    The reported point estimate is the plain p = k/n; the band is [center-half,
    center+half] clipped to [0, 1].  n == 0 yields an undefined point and the
    whole [0, 1] interval.
    """
    if n <= 0:
        return Efficiency(k=0, n=0, z=z, point=float("nan"), lo=0.0, hi=1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))
    return Efficiency(k=int(k), n=int(n), z=z,
                      point=p, lo=max(0.0, center - half), hi=min(1.0, center + half))


# ===========================================================================
# Per-panel loading + classification
# ===========================================================================

@dataclass
class PanelResult:
    name: str
    labels: np.ndarray     # (N,) str, one of waveform_ops.CLASS_LABELS
    peak_pos: np.ndarray   # (N,) int, absolute sample index of the window peak
    edge_pos: np.ndarray   # (N,) float, sub-sample leading-edge half-max crossing
    polarity: str
    baseline: float
    sigma: float
    sat_thr: float
    pulse_lo: int
    pulse_hi: int

    @property
    def saw_pulse(self) -> np.ndarray:
        """Boolean mask: the panel registered a real pulse (class != NOISE)."""
        return self.labels != "NOISE"


def classify_panel(raw: np.ndarray, name: str, readout: str,
                   pulse_lo: int, pulse_hi: int, offset: int,
                   cut_kwargs: dict, auto_window: bool = False,
                   window_coverage: float = 0.99) -> PanelResult:
    """Orient a panel for its detector type and run the shared triage classifier.

    `offset` shifts this panel's pulse window by a fixed number of samples to
    track a known inter-channel timing offset (it is small compared to the window,
    so 0 is fine for the classification itself; it mainly keeps the pre-pulse
    baseline region clean if a panel is strongly shifted).

    With `auto_window`, the window is derived from THIS panel's own data (so a
    narrow PMT panel and a wide SiPM panel each get the right window, and pulses
    near the edge are not misread as NOISE) -- the per-panel windows are matched to
    each panel's real pulses, which reduces bias rather than adding it."""
    L = raw.shape[1]
    plo = max(pulse_lo + offset, 5)
    phi = min(pulse_hi + offset, L)
    if not plo < phi:
        raise ValueError(f"{name}: invalid window [{plo}, {phi}) after offset {offset} "
                         f"for length {L}.")

    # Shared rough->refine preparation (provisional baseline, polarity + orient,
    # per-panel auto-window, refined baseline) -- the SAME routine triage runs, so
    # this panel's numbers match a standalone triage of the same channel.
    prep = ops.prepare_channel(raw, _readout_polarity(readout), plo, phi,
                              auto_window=auto_window, coverage=window_coverage)
    wf, baseline, sigma = prep.oriented, prep.baseline, prep.sigma
    plo, phi, polarity = prep.pulse_lo, prep.pulse_hi, prep.polarity

    labels, _info, sat_thr, _peaks = ops.classify_events(wf, baseline, sigma, plo, phi,
                                                        **cut_kwargs)
    # Peak position straight from the oriented window (independent of the label, so
    # it is valid even for SATURATED events whose classify() peak may be unset).
    peak_pos = plo + np.argmax(wf[:, plo:phi], axis=1)
    # Sub-sample edges: on integer edges the Delta-t MAD is pinned to the ADC-sample
    # lattice (it can only read 0, 0.5, 1, ... -> quoted sigma on the 1.4826-rung
    # ladder; run00270 read exactly 1.48 where the interpolated width is 1.24).
    edge_pos = ops.leading_edge_pos(wf, baseline, peak_pos, subsample=True)

    logger.info("%-6s: polarity=%s baseline=%.4g sigma=%.4g window[%d,%d) -> %s",
                name, polarity, baseline, sigma, plo, phi, _label_counts(labels))
    return PanelResult(name=name, labels=labels, peak_pos=peak_pos.astype(np.int64),
                       edge_pos=edge_pos, polarity=polarity, baseline=baseline, sigma=sigma,
                       sat_thr=sat_thr, pulse_lo=plo, pulse_hi=phi)


def _label_counts(labels: np.ndarray) -> dict:
    return {c: int(np.sum(labels == c)) for c in ops.CLASS_LABELS}


# ===========================================================================
# Efficiency
# ===========================================================================

@dataclass
class EfficiencyReport:
    coincidence: np.ndarray    # denominator mask (top & bottom saw a pulse)
    hit: np.ndarray            # numerator mask (coincidence & middle saw a pulse)
    primary: Efficiency        # not-NOISE definition (CLEAN/SAT/PILEUP all count)
    clean_only: Efficiency     # stricter comparison: middle must be CLEAN
    middle_breakdown: dict     # class -> count, among coincidence events


def compute_efficiency(top: PanelResult, mid: PanelResult, bot: PanelResult,
                       z: float, denominator: str = "coincidence") -> EfficiencyReport:
    """Telescope efficiency of the middle panel with a Wilson CI.

    `denominator` selects the "muon crossed" sample:
      * "coincidence" -- events where top AND bottom both pass the offline triage
        (class != NOISE).  Use when the trigger did NOT already guarantee them, or
        to additionally clean accidental triggers.  NOTE: if top/bottom are small
        PMT pulses, the triage shape/dominance cut can reject real pulses, removing
        genuine crossings and biasing the efficiency -- check the per-panel NOISE.
      * "all" -- every recorded event.  Correct when the DAQ hardware-trigger
        already required a top&bottom coincidence (so every event is a crossing),
        which avoids re-deriving them with cuts that may misfire on small pulses.
    """
    if denominator == "all":
        coincidence = np.ones(len(mid.labels), dtype=bool)
    elif denominator == "coincidence":
        coincidence = top.saw_pulse & bot.saw_pulse
    else:
        raise ValueError(f"denominator must be 'coincidence' or 'all', got {denominator!r}")
    n = int(coincidence.sum())

    hit = coincidence & mid.saw_pulse
    primary = wilson_interval(int(hit.sum()), n, z)

    # Stricter cross-check: count the middle only when it is fully CLEAN.
    clean_hit = coincidence & (mid.labels == "CLEAN")
    clean_only = wilson_interval(int(clean_hit.sum()), n, z)

    middle_breakdown = {c: int(np.sum(coincidence & (mid.labels == c)))
                        for c in ops.CLASS_LABELS}
    return EfficiencyReport(coincidence=coincidence, hit=hit, primary=primary,
                            clean_only=clean_only, middle_breakdown=middle_breakdown)


# ===========================================================================
# Reporting
# ===========================================================================

def print_report(panels: dict[str, PanelResult], rep: EfficiencyReport,
                 denominator: str = "coincidence",
                 dead_time: dict | None = None) -> None:
    N = len(panels["top"].labels)
    n = int(rep.coincidence.sum())
    print("\nHodoscope middle-panel efficiency")
    print(f"Events per file:            {N:,}")
    print("-" * 70)
    print(f"{'Panel':<8}{'Readout/pol':<14}{'CLEAN':>8}{'SATUR.':>8}{'PILEUP':>8}{'NOISE':>8}")
    for name in ("top", "middle", "bottom"):
        p = panels[name]
        c = _label_counts(p.labels)
        print(f"{name:<8}{p.polarity:<14}"
              f"{c['CLEAN']:>8,}{c['SATURATED']:>8,}{c['PILEUP']:>8,}{c['NOISE']:>8,}")
    print("-" * 70)
    if denominator == "all":
        print(f"Denominator = ALL triggered events:       {n:,}  "
              f"(hardware top&bottom coincidence assumed)")
    else:
        print(f"Coincidence (top & bottom saw a pulse):   {n:,}  "
              f"({100*n/N:.1f}% of events)")
    print(f"Middle, among those {'events' if denominator=='all' else 'coincidence events'}:")
    for c in ops.CLASS_LABELS:
        k = rep.middle_breakdown[c]
        tag = "  (counts as HIT)" if c in HIT_CLASSES else "  (counts as MISS)"
        print(f"    {c:<10} {k:>8,}  ({100*k/n:5.1f}% of coincidences){tag}" if n else
              f"    {c:<10} {k:>8,}")
    print("-" * 70)
    print(f"  MIDDLE EFFICIENCY (real pulse): {rep.primary}")
    print(f"  (clean-only cross-check):       {rep.clean_only}")
    if dead_time is not None:
        # DAQ dead time is veto inefficiency this measurement is BLIND to: a muon
        # arriving while the DAQ is dead never enters the file at all, so it is absent
        # from numerator and denominator alike.  Quote the bound next to the statistical
        # error so the reader can see which one dominates.
        stat_pct = 100 * max(rep.primary.err_lo, rep.primary.err_hi)
        dead_pct = 100 * (1.0 - dead_time["livetime_frac_min"])
        print("-" * 70)
        print(f"  DAQ dead-time systematic:       < {dead_pct:.4f} %  "
              f"(livetime > {100 * dead_time['livetime_frac_min']:.4f} %, "
              f"dead time < {dead_time['dt_min_s'] * 1e3:.1f} ms)")
        print(f"  Statistical (binomial) error:     {stat_pct:.4f} %")
        verdict = ("negligible beside the statistical error -- no correction warranted"
                   if dead_pct < 0.2 * stat_pct else
                   "COMPARABLE to the statistical error -- correct for it")
        print(f"  => dead time is {verdict}.")
    print()


# ===========================================================================
# Diagnostics / plots
# ===========================================================================

def _save(fig, out_dir: Path, fname: str, show: bool, save: bool) -> None:
    finish_figure(fig, out_dir / fname if save else None, show)


def plot_dt_coincidence(top: PanelResult, bot: PanelResult,
                        out_dir: Path, show: bool, save: bool) -> dict:
    """Top-bottom Delta-t (bottom edge - top edge), over events where BOTH panels
    actually registered a pulse.

    This validates the assumed inter-channel offset: a single sharp peak confirms a
    genuine, fixed time-of-flight coincidence; a flat or multi-modal distribution
    would mean the files are mis-aligned or the coincidence is dominated by
    accidentals.  The offset itself is measured from the data, not checked against an
    assumed value.

    Timing uses each panel's leading-edge (half-max) crossing rather than the argmax
    peak sample: the edge is immune to the coherent pickup and flat-top saturation that
    jitter the peak time, so a genuine coincidence shows up as a tighter peak here.

    Two deliberate choices: the selection is top.saw_pulse & bot.saw_pulse (never the
    efficiency denominator -- Delta-t is only DEFINED when both panels fired; a no-pulse
    panel's "edge" is a noise pick landing anywhere in the record), and the quoted width
    is the robust MAD sigma (Delta-t keeps hard outliers, which inflate a plain std
    severalfold).
    """
    plt = _setup_mpl(show)
    both = top.saw_pulse & bot.saw_pulse
    dt = (bot.edge_pos - top.edge_pos)[both].astype(float)

    if dt.size:
        med = float(np.median(dt))
        mad_sigma = 1.4826 * float(np.median(np.abs(dt - med)))
    else:
        med = mad_sigma = float("nan")
    stats = {"n": int(dt.size), "n_both_saw": int(both.sum()), "median": med,
             "mad_sigma": mad_sigma,
             "mean": float(np.mean(dt)) if dt.size else float("nan"),
             "std": float(np.std(dt)) if dt.size else float("nan")}

    fig, ax = plt.subplots(figsize=(9, 5))
    if dt.size:
        # Frame the ROBUST core, not the outlier span: a 0.5-99.5 percentile range put
        # the 3-sample-wide coincidence peak in 1.2% of the canvas.  Events outside the
        # frame are counted in the legend rather than silently dropped.
        half = max(10.0, 8.0 * mad_sigma)
        lo, hi = med - half, med + half
        bins = np.arange(math.floor(lo), math.ceil(hi) + 1) - 0.5
        n_out = int(np.sum((dt < lo) | (dt > hi)))
        ax.hist(dt, bins=bins, color="C0", alpha=0.85)
        ax.axvline(med, color="C3", ls="-", lw=1.5, label=f"median = {med:.2f} samp")
        ax.axvspan(med - mad_sigma, med + mad_sigma, color="C3", alpha=0.12,
                   label=f"+/- 1 MAD sigma ({mad_sigma:.2f})")
        if n_out:
            ax.plot([], [], " ", label=f"{n_out:,} events outside frame "
                                       f"({100 * n_out / dt.size:.2f}%)")
    ax.set(xlabel="Delta-t  (bottom edge - top edge)  [samples]",
           ylabel="events (both panels saw a pulse)",
           title="Top-bottom time coincidence")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "eff_dt_coincidence.png", show, save)
    return stats


# NOTE: there is deliberately no per-panel class-breakdown plot.  print_report already
# tabulates all 12 cells (3 panels x 4 classes) with exact counts; a stacked bar of the
# same table added nothing and was strictly WORSE -- it labelled only the cells above
# 3%, which hid SATURATED and PILEUP, the two the hit definition actually turns on.


def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def plot_noise_residuals(raws: dict[str, np.ndarray], panels: dict[str, PanelResult],
                         out_dir: Path, show: bool, save: bool) -> dict:
    """Per-panel pre-pulse noise-residual distributions, one subplot each, with a
    Gaussian fit.

    The residual is the pre-pulse region (samples before each panel's own window
    start) minus that panel's baseline -- exactly the samples that set the noise
    floor in ops.global_baseline_noise.  Showing the three side by side (each with a
    least-squares Gaussian fit, its MAD, and the panel's global median baseline)
    reveals whether the panels share a consistent noise level, or whether one channel
    is noisier / has an offset baseline -- which would shift its noise_prominence cut
    and bias its NOISE fraction.  The MAD is the robust width the noise sigma is built
    from (sigma = 1.4826 * MAD); the Gaussian sigma is the least-squares comparison.

    BINNING: the residual is a difference of INTEGER ADC samples and a median baseline,
    so it lives on an integer lattice, and the bins must be integer-ALIGNED (unit width,
    edges on half-integers).  Fractional-width bins alternately catch a lattice point or
    nothing, and a least-squares fit gets dragged down to the mean of that spike/zero comb.

    NON-GAUSSIANITY: these baselines are not always Gaussian (spike noise gives a peaked,
    heavy-tailed residual).  A single fitted sigma would hide that, so RMS/MAD-sigma is
    reported per panel and flagged when the tails inflate it -- a Gaussian has RMS = MAD
    sigma, so a ratio well above 1 means the fitted sigma is not the whole story."""
    from scipy.optimize import curve_fit

    plt = _setup_mpl(show)
    names = ["top", "middle", "bottom"]
    colors = {"top": "C0", "middle": "C1", "bottom": "C2"}
    # Above this RMS/MAD-sigma ratio the residual is too heavy-tailed for a single
    # Gaussian sigma to describe, and the panel is flagged on the plot.
    NON_GAUSS_RATIO = 1.2

    resids, stats = {}, {}
    for nm in names:
        p = panels[nm]
        pre = raws[nm][:, :max(p.pulse_lo, 5)].astype(float) - p.baseline
        r = pre.ravel()
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        sigma = 1.4826 * mad
        rms = float(np.std(r))
        resids[nm] = r
        stats[nm] = {"baseline": float(p.baseline), "mad": mad, "sigma": sigma,
                     "rms": rms, "rms_over_sigma": rms / sigma if sigma > 0 else float("nan"),
                     "n": int(r.size)}

    # Common symmetric, INTEGER-ALIGNED bin range so the three panels are directly
    # comparable AND each bin holds exactly one lattice point (see BINNING above).
    allr = np.concatenate([resids[nm] for nm in names]) if resids else np.array([0.0])
    span = float(np.percentile(np.abs(allr), 99.5)) if allr.size else 1.0
    span = math.ceil(max(span, 1.0))
    bins = np.arange(-span, span + 2) - 0.5
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, nm in zip(axes, names):
        s = stats[nm]
        counts, _ = np.histogram(resids[nm], bins=bins, density=True)
        ax.bar(centers, counts, width=np.diff(bins), color=colors[nm], alpha=0.5,
               align="center")

        # Least-squares Gaussian fit, seeded from the robust moments.
        g_sigma = float("nan")
        try:
            p0 = (counts.max(), 0.0, max(s["sigma"], span / 20))
            popt, _ = curve_fit(_gaussian, centers, counts, p0=p0, maxfev=10000)
            g_sigma = abs(float(popt[2]))
            xf = np.linspace(-span, span, 400)
            ax.plot(xf, _gaussian(xf, *popt), color="k", lw=1.6,
                    label=f"Gaussian fit\nsigma = {g_sigma:.3g}")
        except (RuntimeError, ValueError):
            logger.warning("%s: noise-residual Gaussian fit did not converge.", nm)
        s["gauss_sigma"] = g_sigma

        ratio = s["rms_over_sigma"]
        heavy = np.isfinite(ratio) and ratio > NON_GAUSS_RATIO
        if heavy:
            # Say so ON the plot: a lone Gaussian sigma on a heavy-tailed baseline
            # understates the real excursions the noise_prominence cut has to survive.
            ax.plot([], [], " ", label=f"NON-GAUSSIAN\nRMS/sigma = {ratio:.2f}")
        ax.set_title(f"{nm}\nbaseline = {s['baseline']:.4g} ADC   "
                     f"MAD sigma = {s['sigma']:.3g}   RMS = {s['rms']:.3g}"
                     f"{'  (!)' if heavy else ''}", fontsize=9)
        ax.set_xlabel("pre-pulse residual  (ADC - baseline)")
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("probability density")
    fig.suptitle("Per-panel noise-residual comparison", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "eff_noise_residuals.png", show, save)
    return stats


def plot_sensitivity(raws: dict[str, np.ndarray], readouts: dict[str, str],
                     pulse_lo: int, pulse_hi: int, offsets: dict[str, int],
                     panel_kw: dict, param: str, values: np.ndarray, z: float,
                     out_dir: Path, show: bool, save: bool,
                     denominator: str = "coincidence",
                     auto_window: bool = False, window_coverage: float = 0.99) -> None:
    """Re-run the whole efficiency measurement while sweeping ONE cut parameter,
    to show the result is not an artifact of a particular threshold.  The swept
    value replaces that parameter in EVERY panel's (preset-derived) cut set."""
    plt = _setup_mpl(show)
    pts, los, his, ns = [], [], [], []
    print(f"\nSensitivity scan: {param} over {list(np.round(values, 3))}")
    for v in values:
        typed = int(round(v)) if param in _INT_CUTS else float(v)
        panels = {}
        for nm in ("top", "middle", "bottom"):
            kw = dict(panel_kw[nm]); kw[param] = typed
            panels[nm] = classify_panel(raws[nm], nm, readouts[nm], pulse_lo, pulse_hi,
                                        offsets[nm], kw, auto_window=auto_window,
                                        window_coverage=window_coverage)
        rep = compute_efficiency(panels["top"], panels["middle"], panels["bottom"], z, denominator)
        e = rep.primary
        pts.append(e.point); los.append(e.lo); his.append(e.hi); ns.append(e.n)
        print(f"  {param}={v:<8.3g}  eff={100*e.point:6.2f}%  "
              f"[{100*e.lo:.2f}, {100*e.hi:.2f}]  n={e.n:,}")

    pts, los, his = np.array(pts), np.array(los), np.array(his)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(values, 100 * los, 100 * his, color="C0", alpha=0.2, label="Wilson CI")
    ax.plot(values, 100 * pts, "o-", color="C0", label="efficiency")
    ax.set(xlabel=f"cut parameter: {param}", ylabel="middle efficiency (%)",
           title=f"Efficiency vs {param}  (with Wilson CI band)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "eff_sensitivity.png", show, save)


_INT_CUTS = {"min_separation", "max_extra_pulses", "post_pulse_veto",
             "undershoot_window", "max_peaks", "consec"}


# ===========================================================================
# Main
# ===========================================================================

def _resolve_sources(args) -> dict[str, tuple[Path, int]]:
    """Map each panel to (file, channel).  Two input styles are supported:
      * one multi-channel file via --input + --*-channel (default channels 0/1/2);
      * three separate files via --top/--middle/--bottom (+ optional --*-channel)."""
    def ch(val, default):
        return default if val is None else val
    # Bare filenames -> waveform_files/, wherever their dataset folders keep them.
    if args.input is not None:
        c = {"top": ch(args.top_channel, 0), "middle": ch(args.middle_channel, 1),
             "bottom": ch(args.bottom_channel, 2)}
        src = resolve_input(args.input)
        return {nm: (src, c[nm]) for nm in ("top", "middle", "bottom")}
    missing = [f"--{nm}" for nm in ("top", "middle", "bottom") if getattr(args, nm) is None]
    if missing:
        raise SystemExit(f"Provide either --input (one multi-channel file) or all of "
                         f"--top/--middle/--bottom; missing {', '.join(missing)}.")
    return {"top": (resolve_input(args.top), ch(args.top_channel, 0)),
            "middle": (resolve_input(args.middle), ch(args.middle_channel, 0)),
            "bottom": (resolve_input(args.bottom), ch(args.bottom_channel, 0))}


def panel_cut_kwargs(readout: str, args) -> dict:
    """Cut parameters for ONE panel: start from the detector preset for its readout
    (PMT vs SiPM, shared with waveform_triage) and let any value set explicitly on
    the CLI override it.  So a PMT top/bottom panel automatically gets the higher
    spike threshold, shorter flat-top and smaller separation, while the SiPM middle
    keeps the standard values -- in a single run."""
    preset = ops.DETECTOR_PRESETS["pmt" if readout == "pmt" else "sipm"]
    pick = lambda cli, key: preset[key] if cli is None else cli
    return dict(
        saturation_adc=args.saturation_adc,
        noise_prominence=pick(args.noise_prominence, "noise_prominence"),
        pileup_prominence=args.pileup_prominence, extra_frac=args.extra_frac,
        extra_min_sigma=args.extra_min_sigma, rail_tol=args.rail_tol,
        consec=pick(args.consec, "consec"),
        min_separation=pick(args.min_separation, "min_separation"),
        max_extra_pulses=args.max_extra_pulses, post_pulse_veto=args.post_pulse_veto,
        undershoot_sigma=args.undershoot_sigma, undershoot_window=args.undershoot_window,
        max_peaks=args.max_peaks, dom_floor_sigma=args.dom_floor_sigma)


def run(args) -> EfficiencyReport:
    readouts = {"top": args.tb_readout, "middle": args.middle_readout, "bottom": args.tb_readout}
    offsets = {"top": args.offset_top, "middle": args.offset_middle, "bottom": args.offset_bottom}
    sources = _resolve_sources(args)
    for nm, (path, c) in sources.items():
        logger.info("%-6s panel <- %s [channel %d, %s]", nm, path, c, readouts[nm])
    raws = {nm: ops.load_waveforms(path, channel=c)[0] for nm, (path, c) in sources.items()}

    # Row-aligned coincidence requires identical event counts.
    lengths = {nm: r.shape[0] for nm, r in raws.items()}
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"Files have different event counts {lengths}; the three panels "
                         "must be row-aligned (same AND-triggered events). Re-extract them "
                         "from the same run.")

    # Per-panel cut parameters (PMT preset for PMT panels, SiPM for SiPM panels).
    # The preset needs a CONCRETE readout: 'auto' used to fall through to the SiPM
    # preset even on a panel whose polarity votes PMT -- the exact trap
    # triage_cuts(polarity=...) documents (min_separation=40 on a fast PMT pulse).
    # Resolve it here with the same polarity vote classify_panel's preparation will
    # run on the same data, so the cuts and the orientation cannot disagree.
    cut_readouts = dict(readouts)
    for nm in ("top", "middle", "bottom"):
        if readouts[nm] == "auto":
            pol = ops.polarity_vote(raws[nm])[0]
            cut_readouts[nm] = ops.detector_for_polarity(pol)
            logger.info("%-6s readout 'auto' -> %s cut preset (polarity vote: %s).",
                        nm, cut_readouts[nm], pol)
    panel_kw = {nm: panel_cut_kwargs(cut_readouts[nm], args) for nm in ("top", "middle", "bottom")}
    for nm in ("top", "middle", "bottom"):
        logger.info("%-6s cuts: noise_prom=%g consec=%d min_sep=%d", nm,
                    panel_kw[nm]["noise_prominence"], panel_kw[nm]["consec"],
                    panel_kw[nm]["min_separation"])

    panels = {nm: classify_panel(raws[nm], nm, readouts[nm], args.pulse_lo, args.pulse_hi,
                                 offsets[nm], panel_kw[nm],
                                 auto_window=args.auto_window, window_coverage=args.window_coverage)
              for nm in ("top", "middle", "bottom")}

    # BIAS GUARD: the default 'coincidence' denominator re-derives top&bottom offline.
    # If the DAQ hardware trigger already REQUIRED a top&bottom coincidence, every
    # recorded event is a crossing, and re-deriving it with offline cuts can drop
    # marginal (small PMT) pulses and bias the efficiency -- '--denominator all' avoids that.
    if args.denominator == "coincidence":
        logger.warning("Denominator='coincidence' re-derives top&bottom offline; if the DAQ "
                       "already hardware-triggered on a top&bottom coincidence this can bias the "
                       "efficiency (drops marginal pulses). Consider --denominator all.")

    rep = compute_efficiency(panels["top"], panels["middle"], panels["bottom"],
                             args.z, args.denominator)
    # Dead time is an inefficiency this measurement cannot see (see dead_time_systematic);
    # it needs the arrival times, which the middle-panel file carries when it came through
    # the current converter.  Absent a time axis the report is simply as it always was.
    dead_time = dead_time_systematic(sources["middle"][0],
                                     resolve_input(args.times) if args.times else None)
    if dead_time is None:
        logger.info("No per-event time axis found; the DAQ dead-time systematic is not "
                    "reported. Re-extract the channel or pass --times to include it.")
    print_report(panels, rep, args.denominator, dead_time)

    # Group this run's eff_*.png in its own per-run results folder,
    # preprocessing_results/hodoscope/<input-stem>_efficiency_results[_N]/, keyed by the
    # multi-channel input file (or the top-panel file in three-file mode); a re-run
    # gets a fresh _N folder instead of overwriting the previous one.
    dataset = (args.input if args.input is not None else args.top).stem
    out_dir = resolve_results_dir(__file__, dataset, base=args.output_dir,
                                  program="efficiency", group="hodoscope",
                                  overwrite=args.overwrite)
    show, save = (not args.no_show), args.save_plots
    if show or save:
        dt_stats = plot_dt_coincidence(panels["top"], panels["bottom"], out_dir, show, save)
        logger.info("Top-bottom Delta-t (both panels saw a pulse, n=%d): median=%.2f "
                    "MAD sigma=%.2f samples  [non-robust mean=%.2f std=%.2f].",
                    dt_stats["n"], dt_stats["median"], dt_stats["mad_sigma"],
                    dt_stats["mean"], dt_stats["std"])
        noise_stats = plot_noise_residuals(raws, panels, out_dir, show, save)
        for nm in ("top", "middle", "bottom"):
            s = noise_stats[nm]
            logger.info("%-6s noise residual: baseline=%.4g MAD sigma=%.4g RMS=%.4g "
                        "(RMS/sigma=%.2f) gauss_sigma=%.4g ADC (n=%d).",
                        nm, s["baseline"], s["sigma"], s["rms"], s["rms_over_sigma"],
                        s["gauss_sigma"], s["n"])
    # The cut-sensitivity sweep is a SYSTEMATIC on the reported efficiency, not a
    # picture: it runs whether or not plots were asked for (only the figure is gated),
    # because "the number moves by 5% if you nudge the threshold" is something you must
    # be told even on a --no-show --no-save run.  It prints its table either way.
    if args.sens_n > 1:
        values = np.linspace(args.sens_lo, args.sens_hi, args.sens_n)
        plot_sensitivity(raws, readouts, args.pulse_lo, args.pulse_hi, offsets,
                         panel_kw, args.sensitivity_param, values, args.z,
                         out_dir, show, save, args.denominator,
                         auto_window=args.auto_window, window_coverage=args.window_coverage)
    return rep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Middle-panel muon detection efficiency for a 3-panel telescope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input: EITHER one multi-channel file (--input + --*-channel) OR three files.
    p.add_argument("--input", type=Path, default=None,
                   help="One multi-channel HDF5 file holding all three panels (use with "
                        "--top-channel/--middle-channel/--bottom-channel). Alternative to "
                        "--top/--middle/--bottom.")
    p.add_argument("--top", type=Path, default=None, help="Top-panel HDF5 file (if not using --input).")
    p.add_argument("--middle", type=Path, default=None, help="Middle-panel HDF5 file (if not using --input).")
    p.add_argument("--bottom", type=Path, default=None, help="Bottom-panel HDF5 file (if not using --input).")
    p.add_argument("--top-channel", type=int, default=None,
                   help="Channel index for the top panel (default: 0 with --input, else 0).")
    p.add_argument("--middle-channel", type=int, default=None,
                   help="Channel index for the middle panel (default: 1 with --input, else 0).")
    p.add_argument("--bottom-channel", type=int, default=None,
                   help="Channel index for the bottom panel (default: 2 with --input, else 0).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Base output directory; eff_*.png go into "
                        "<output-dir>/<input-stem>_efficiency_results[_N]/. Default base: "
                        "preprocessing/preprocessing_results/hodoscope/ (a re-run gets a "
                        "fresh _N folder unless --overwrite).")
    p.add_argument("--overwrite", action="store_true",
                   help="Write into the canonical (un-suffixed) results folder, replacing that "
                        "run's files in place, instead of creating a fresh _N folder.")

    # Detector types -> polarity (config 1: pmt/sipm; config 2: all sipm).
    p.add_argument("--tb-readout", choices=["sipm", "pmt", "auto"], default="pmt",
                   help="Readout of the TOP and BOTTOM panels.")
    p.add_argument("--middle-readout", choices=["sipm", "pmt", "auto"], default="sipm",
                   help="Readout of the MIDDLE panel.")

    # Pulse window + per-panel timing offsets.
    p.add_argument("--pulse-lo", type=int, default=366,
                   help="Provisional pulse-window start (samples); the fixed window only with --no-auto-window.")
    p.add_argument("--pulse-hi", type=int, default=726,
                   help="Provisional pulse-window end (samples); the fixed window only with --no-auto-window.")
    p.add_argument("--auto-window", dest="auto_window", action="store_true", default=True,
                   help="Derive each panel's pulse window from ITS OWN data, so a narrow PMT panel "
                        "and a wide SiPM panel each get the right window and edge pulses are not "
                        "misread as NOISE. ON by default; disable with --no-auto-window.")
    p.add_argument("--no-auto-window", dest="auto_window", action="store_false",
                   help="Use the fixed --pulse-lo/--pulse-hi for all panels instead of per-panel auto windows.")
    p.add_argument("--window-coverage", type=float, default=0.99,
                   help="Fraction of real-pulse peaks the auto-window must span. Default: 0.99.")
    p.add_argument("--offset-top", type=int, default=0, help="Sample shift of the top window.")
    p.add_argument("--offset-middle", type=int, default=0, help="Sample shift of the middle window.")
    p.add_argument("--offset-bottom", type=int, default=0, help="Sample shift of the bottom window.")

    # Denominator definition for the efficiency.
    p.add_argument("--denominator", choices=["coincidence", "all"], default="all",
                   help="'all' (default): every recorded event -- correct when the DAQ hardware "
                        "trigger ALREADY required a top&bottom coincidence (avoids re-deriving "
                        "them with cuts that can reject small PMT pulses). 'coincidence': require "
                        "top AND bottom to pass offline triage -- use when the trigger did NOT "
                        "guarantee them, or to additionally clean accidental triggers.")

    # Confidence level.
    p.add_argument("--z", type=float, default=1.96,
                   help="Standard-normal quantile for the Wilson CI (1.96=95%%, 2.576=99%%).")

    # Triage cut parameters (shared with waveform_triage; defaults match it).
    # noise_prominence / consec / min_separation default to the per-panel detector
    # preset (PMT vs SiPM) when left unset; an explicit value here applies to all panels.
    p.add_argument("--saturation-adc", type=float, default=None)
    p.add_argument("--noise-prominence", type=float, default=None,
                   help="Min pulse height in noise sigma. Default per panel: 5 (sipm) / 6 (pmt).")
    p.add_argument("--pileup-prominence", type=float, default=6.0)
    p.add_argument("--extra-frac", type=float, default=0.08)
    p.add_argument("--extra-min-sigma", type=float, default=12.0)
    p.add_argument("--rail-tol", type=float, default=0.01)
    p.add_argument("--consec", type=int, default=None,
                   help="Saturation flat-top consecutive samples. Default per panel: 4 (sipm) / 2 (pmt).")
    p.add_argument("--min-separation", type=int, default=None,
                   help="Min separation main/extra pulse. Default per panel: 40 (sipm) / 25 (pmt).")
    p.add_argument("--max-extra-pulses", type=int, default=2)
    p.add_argument("--post-pulse-veto", type=int, default=300)
    p.add_argument("--undershoot-sigma", type=float, default=6.0)
    p.add_argument("--undershoot-window", type=int, default=80)
    p.add_argument("--max-peaks", type=int, default=4)
    p.add_argument("--dom-floor-sigma", type=float, default=4.0,
                   help="Dominance comparable-crest height floor (noise sigma) so small PMT "
                        "pulses are not mistaken for oscillating noise. Default: 4.")

    # Sensitivity sweep.
    p.add_argument("--sensitivity-param", default="noise_prominence",
                   help="Cut parameter to sweep for the efficiency-vs-cut check.")
    p.add_argument("--sens-lo", type=float, default=3.0, help="Sweep start.")
    p.add_argument("--sens-hi", type=float, default=8.0, help="Sweep end.")
    p.add_argument("--sens-n", type=int, default=6, help="Sweep points (1 disables the sweep).")

    p.add_argument("--times", type=Path, default=None,
                   help="Times .h5 carrying /event_time_rel_s, used to bound the DAQ "
                        "dead-time systematic on the efficiency. Only needed when the panel "
                        "files themselves have no time axis (an older conversion); default: "
                        "the run's own waveform_files/<run>/times/<middle-stem>_times.h5 "
                        "if it exists.")
    p.add_argument("--save-plots", action="store_true", help="Save eff_*.png figures.")
    p.add_argument("--no-show", action="store_true", help="Do not open plot windows.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
