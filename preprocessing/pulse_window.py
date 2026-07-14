#!/usr/bin/env python3
"""
pulse_window.py
===============
General-purpose diagnostic tool for the waveform_triage cuts.

The triage classifier applies a chain of thresholds (pulse-window height, leading-
edge shape, record "dominance", saturation flat-top, pile-up extra height /
fraction / separation / undershoot, ...).  This tool makes each of those cuts and
its parameters VISIBLE: where the threshold sits relative to the population it
acts on, how many events it moves, and how the energy spectrum changes.

It calls the REAL triage functions (waveform_triage.classify / detect_saturation /
global_baseline_noise) so what you see is exactly what the classifier does.

Sub-commands
------------
  window    Recommend / validate the pulse window (the original tool).
  features  Per-cut distributions with the threshold drawn on top, plus a cutflow
            table (how many events each cut removes, in priority order).
  scan      Sweep ONE parameter over a range and plot the four class counts.
  baseline  Compare estimating baseline + MAD-noise GLOBALLY (one pooled value)
            vs LOCALLY (per event): stability, and effect on the spectrum.
  spectrum  Energy / pulse-height spectrum, sculpted cut-by-cut, with the clean
            spectrum's peaks marked.

Examples
--------
  python pulse_window.py window   --input run00270_ch0.h5
  python pulse_window.py features --input run00270_ch0.h5 --save-plot feat.png
  python pulse_window.py scan     --input run00270_ch0.h5 --param noise_prominence --lo 2 --hi 10 --n 17
  python pulse_window.py baseline --input run00270_ch0.h5
  python pulse_window.py spectrum --input run00270_ch0.h5 --metric charge
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import waveform_triage as wt
from peakfind import find_peaks_manual
from plotting import setup_mpl as _setup_mpl, finish_figure as _finish, paged_figure

# Shared waveform-file lookup (bare filename -> waveform_files/, including its
# per-run folders), the same convention waveform_triage / hodoscope_efficiency use.
sys.path.insert(0, str(Path(__file__).parent.parent / "file_manipulation"))
from output_paths import resolve_input  # noqa: E402

logger = logging.getLogger("pulse_window")


# ===========================================================================
# Parameter bundle (mirrors the waveform_triage CLI defaults)
# ===========================================================================

@dataclass
class Params:
    pulse_lo: int = 366
    pulse_hi: int = 726
    saturation_adc: float | None = None
    rail_tol: float = 0.01
    consec: int = 4
    noise_prominence: float = 5.0
    rise: int = 40
    pileup_prominence: float = 6.0
    extra_frac: float = 0.08
    extra_min_sigma: float = 12.0
    min_separation: int = 40
    max_extra_pulses: int = 2
    post_pulse_veto: int = 300
    undershoot_sigma: float = 6.0
    undershoot_window: int = 80
    max_peaks: int = 4
    dom_floor_sigma: float = 4.0


# Scannable parameters: name -> (is_int)
SCAN_PARAMS = {
    "pulse_lo": True, "pulse_hi": True, "noise_prominence": False,
    "pileup_prominence": False, "extra_frac": False, "extra_min_sigma": False,
    "min_separation": True, "max_extra_pulses": True, "post_pulse_veto": True,
    "undershoot_sigma": False, "undershoot_window": True, "rail_tol": False,
    "max_peaks": True, "dom_floor_sigma": False,
}


# ===========================================================================
# Core: baseline, classification, per-event features
# ===========================================================================

def global_baseline(waveforms: np.ndarray, pulse_lo: int) -> tuple[float, float]:
    return wt.global_baseline_noise(waveforms, pulse_lo)


def local_baseline(waveforms: np.ndarray, pulse_lo: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-event baseline (median) and noise sigma (MAD) from each event's own
    pre-pulse region."""
    pre = waveforms[:, :max(pulse_lo, 5)]
    base = np.median(pre, axis=1)
    sigma = 1.4826 * np.median(np.abs(pre - base[:, None]), axis=1)
    sigma = np.maximum(sigma, 1e-6)
    return base, sigma


def classify_all(waveforms, base, sigma, P: Params) -> dict:
    """Run the real triage via wt.classify_events (the shared entry point, so the
    class priority pileup > saturated > noise > clean is triage's own, not a local
    copy) and return the four exclusive class masks plus the per-event info list."""
    labels, info, sat_thr, peaks = wt.classify_events(
        waveforms, base, sigma, P.pulse_lo, P.pulse_hi,
        saturation_adc=P.saturation_adc, pileup_prominence=P.pileup_prominence,
        noise_prominence=P.noise_prominence, extra_frac=P.extra_frac,
        extra_min_sigma=P.extra_min_sigma, rail_tol=P.rail_tol, consec=P.consec,
        min_separation=P.min_separation, max_extra_pulses=P.max_extra_pulses,
        post_pulse_veto=P.post_pulse_veto, undershoot_sigma=P.undershoot_sigma,
        undershoot_window=P.undershoot_window, max_peaks=P.max_peaks,
        dom_floor_sigma=P.dom_floor_sigma)
    return {"sat": labels == "SATURATED", "pileup": labels == "PILEUP",
            "noise": labels == "NOISE", "clean": labels == "CLEAN",
            "info": info, "peaks": peaks, "sat_thr": float(sat_thr)}


def longest_near_rail_run(waveforms: np.ndarray, rail_band: float) -> np.ndarray:
    """Longest run of consecutive samples at/above rail_band, per event.
    Fully vectorized: pad each row with False, diff to find run starts (+1) and
    ends (-1); flattened row-major, the k-th start pairs with the k-th end (runs
    are closed within their row), so end-start is every run length at once."""
    near = np.zeros((waveforms.shape[0], waveforms.shape[1] + 2), dtype=np.int8)
    near[:, 1:-1] = waveforms >= rail_band
    d = np.diff(near, axis=1)
    starts = np.flatnonzero(d.ravel() == 1)
    ends = np.flatnonzero(d.ravel() == -1)
    best = np.zeros(len(waveforms))
    np.maximum.at(best, starts // d.shape[1], (ends - starts).astype(float))
    return best


def compute_features(waveforms, base, sigma, P: Params) -> dict:
    """Per-event scalars that the cuts act on, plus flat per-extra-candidate arrays.
    Mirrors the quantities inside waveform_triage so each threshold can be drawn
    against the distribution it filters."""
    N, L = waveforms.shape
    cls = classify_all(waveforms, base, sigma, P)
    cand = cls["peaks"]
    center_h = waveforms[np.arange(N), cand] - base

    rise_frac = np.zeros(N)
    dom_count = np.zeros(N, dtype=int)
    charge = np.zeros(N)
    for i in range(N):
        wf = waveforms[i]
        c = cand[i]
        h = center_h[i]
        foot = float(np.min(wf[max(c - P.rise, 0):c + 1])) - base
        rise_frac[i] = (h - foot) / h if h > 0 else 0.0
        # Mirror _pulse_dominates' SNR-aware comparable-crest height (floored at
        # dom_floor_sigma*sigma) so this diagnostic matches what the cut actually does.
        osc_height = max(0.5 * h, P.dom_floor_sigma * sigma)
        dom_count[i] = wt._count_big_peaks(wf, base, osc_height, P.min_separation) if h > 0 else 0
        charge[i] = float(np.sum(wf[P.pulse_lo:P.pulse_hi] - base))

    # Per-extra-candidate diagnostics (mirror classify's inner loop, recording the
    # quantity each gate tests rather than only the final decision).
    find_height = base + P.extra_min_sigma * sigma
    # Same vectorized pre-screen as wt.classify: a candidate must lie OUTSIDE the
    # pulse window and clear find_height, so events whose outside-window maximum
    # stays below it cannot yield any candidate -- skip their peak search.
    outside_max = np.maximum(waveforms[:, :P.pulse_lo].max(axis=1, initial=-np.inf),
                             waveforms[:, P.pulse_hi:].max(axis=1, initial=-np.inf))
    ex_h, ex_frac, ex_under, ex_passed = [], [], [], []
    ex_fail_height, ex_fail_shape, ex_fail_veto, ex_fail_undershoot = [], [], [], []
    for i in range(N):
        wf = waveforms[i]
        c = cand[i]
        h = center_h[i]
        if h <= 0 or outside_max[i] < find_height:
            continue
        peaks = find_peaks_manual(wf, height=find_height,
                                  prominence=P.pileup_prominence * sigma,
                                  distance=P.min_separation)
        extra_thresh = max(P.extra_min_sigma * sigma, P.extra_frac * h)
        for p in peaks:
            p = int(p)
            if (P.pulse_lo <= p < P.pulse_hi) or abs(p - c) < P.min_separation:
                continue
            ph = wf[p] - base
            lo_w, hi_w = max(p - P.undershoot_window, 0), min(p + P.undershoot_window + 1, L)
            under = (float(wf[lo_w:hi_w].min()) - base) / sigma
            shape_ok = wt._is_real_pulse(wf, p, base, extra_thresh)
            vetoed = 0 < (p - c) <= P.post_pulse_veto
            # Per-gate rejection flags (a candidate may fail several at once).
            fail_height = ph < extra_thresh
            fail_shape = not shape_ok
            fail_veto = vetoed
            fail_undershoot = under < -P.undershoot_sigma
            passed = not (fail_height or fail_shape or fail_veto or fail_undershoot)
            ex_h.append(ph / sigma)
            ex_frac.append(ph / h)
            ex_under.append(under)
            ex_passed.append(passed)
            ex_fail_height.append(fail_height)
            ex_fail_shape.append(fail_shape)
            ex_fail_veto.append(fail_veto)
            ex_fail_undershoot.append(fail_undershoot)

    return {
        "cls": cls, "center_sigma": center_h / sigma, "center_h": center_h,
        "rise_frac": rise_frac, "dom_count": dom_count, "charge": charge,
        "row_max": waveforms.max(axis=1),
        "ex_h": np.array(ex_h), "ex_frac": np.array(ex_frac),
        "ex_under": np.array(ex_under), "ex_passed": np.array(ex_passed, dtype=bool),
        "ex_fail_height": np.array(ex_fail_height, dtype=bool),
        "ex_fail_shape": np.array(ex_fail_shape, dtype=bool),
        "ex_fail_veto": np.array(ex_fail_veto, dtype=bool),
        "ex_fail_undershoot": np.array(ex_fail_undershoot, dtype=bool),
    }


# ===========================================================================
# Plot helpers
# ===========================================================================

def counts_line(cls: dict) -> str:
    n = len(cls["clean"])
    return (f"clean={int(cls['clean'].sum()):,} ({100*cls['clean'].mean():.1f}%)  "
            f"sat={int(cls['sat'].sum()):,}  pileup={int(cls['pileup'].sum()):,}  "
            f"noise={int(cls['noise'].sum()):,}   [N={n:,}]")


# ===========================================================================
# Sub-command: window  (recommend / validate the pulse window)
# ===========================================================================

def cmd_window(waveforms, args):
    P = Params()
    N, L = waveforms.shape
    # Rough first run: baseline/noise from the first 20% of the record; the refined
    # determination is the recommended window this command reports.
    prov_lo = min(max(int(0.2 * L), 5), L - 1)
    base, sigma = global_baseline(waveforms, prov_lo)

    peak_pos = waveforms.argmax(axis=1)
    peak_h = waveforms.max(axis=1) - base
    real = peak_h >= args.min_sigma * sigma
    positions = peak_pos[real]
    heights = peak_h[real]
    if positions.size < 20:
        logger.warning("Only %d events clear %.0f sigma; recommendation unreliable.",
                       positions.size, args.min_sigma)
    tail = (1 - args.coverage) / 2 * 100
    q_lo, q_hi = (int(np.floor(v)) for v in np.percentile(positions, [tail, 100 - tail]))
    med = np.median(waveforms[real] - base, axis=0)
    mpk = int(med.argmax())
    rise, fall = wt.median_pulse_extent(med, mpk)
    # Recommend via the SAME routine the auto-window applies (incl. its max_lead
    # robustness cap), so the 'recommended' shown here is exactly the window that
    # triage / hodoscope_efficiency would actually use -- one source of truth.
    rec = wt.recommend_window(waveforms, base, sigma, min_sigma=args.min_sigma,
                              coverage=args.coverage, pad=args.pad)
    rec_lo, rec_hi = rec if rec is not None else (max(q_lo - rise - args.pad, 5),
                                                  min(q_hi + fall + args.pad, L))

    print("\n" + "=" * 64)
    print("Pulse-window analysis")
    print("=" * 64)
    print(f"Events: {N:,} (length {L})   baseline {base:.4g}   sigma {sigma:.4g}")
    print(f"Real pulses: {positions.size:,}   peak pos min/med/max: "
          f"{positions.min()}/{int(np.median(positions))}/{positions.max()}")
    print(f"{int(args.coverage*100)}% of peaks within [{q_lo}, {q_hi}]   "
          f"median pulse rise/fall {rise}/{fall} (peak @ {mpk})")
    print(f"RECOMMENDED:  --pulse-lo {rec_lo} --pulse-hi {rec_hi}  "
          f"(leaves {rec_lo} baseline samples)")
    if args.pulse_lo is not None and args.pulse_hi is not None:
        lo, hi = args.pulse_lo, args.pulse_hi
        inside = np.mean((positions >= lo) & (positions < hi))
        before = np.mean(positions < lo)
        after = np.mean(positions >= hi)
        print("-" * 64)
        print(f"CURRENT [{lo}, {hi}):  inside {100*inside:.1f}%   "
              f"before-lo {100*before:.1f}%   after-hi {100*after:.1f}%")
    print("=" * 64 + "\n")

    if args.save_plot or not args.no_show:
        plt = _setup_mpl(not args.no_show)
        # Plot the constant-fraction (50%) leading edge, NOT the argmax: the argmax
        # of a rail-clipped flat top lands on its FIRST rail sample, so saturated
        # pulses smear systematically early and fake a second "early hump" in the
        # position distribution (seen on run00270 ch0, where that shoulder was 85%
        # ADC-clipped events).  The CF edge is saturation-immune, so what remains
        # early+tall here is genuine trigger time-walk.  The window recommendation
        # itself still uses argmax positions -- the window must contain the peaks.
        edges_cf = wt.leading_edge_pos(waveforms[real], base, peak_pos[real],
                                       frac=0.5, subsample=True)
        fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
        ax[0].hist(edges_cf, bins=min(L, 400), color="C0", alpha=0.8)
        ax[0].set_yscale("log")
        ax[0].set(ylabel="Real pulses",
                  title="Arrival-time distribution (50% leading edge; saturation-immune)")
        ax[1].plot(med, color="C0", lw=1.5); ax[1].axhline(0, color="gray", lw=0.6)
        ax[1].set(ylabel="ADC (base-sub)", title="Median pulse")
        # Full-trace peak HEIGHT vs ARRIVAL: if the early-arriving pulses are
        # systematically the TALL ones, that is trigger time-walk (a steep leading
        # edge crosses the fixed trigger threshold sooner, so big pulses land earlier
        # in the record).  A window edge slicing through this cloud drops those tall
        # early pulses -- so this panel shows both the cause and what the window loses.
        ax[2].scatter(edges_cf, heights, s=3, alpha=0.15, color="C0", edgecolors="none")
        ax[2].axhline(args.min_sigma * sigma, color="gray", ls=":", lw=1.0,
                      label=f"real-pulse floor ({args.min_sigma:.0f} sigma)")
        ax[2].set_yscale("log")
        ax[2].set(xlabel="Sample (50% leading edge)", ylabel="peak height above baseline (ADC)",
                  title="Peak height vs arrival  (early + tall => trigger time-walk)")
        for a in ax:
            a.axvspan(rec_lo, rec_hi, color="C2", alpha=0.18, label=f"recommended [{rec_lo},{rec_hi}]")
            if args.pulse_lo is not None:
                a.axvspan(args.pulse_lo, args.pulse_hi, color="C3", alpha=0.15,
                          label=f"current [{args.pulse_lo},{args.pulse_hi}]")
            a.legend(); a.grid(True, alpha=0.3)
        fig.suptitle("Pulse window"); fig.tight_layout()
        _finish(fig, args.save_plot, not args.no_show)


# ===========================================================================
# Sub-command: features  (per-cut distributions + cutflow)
# ===========================================================================

def cmd_features(waveforms, args):
    P = params_from_args(args)
    base, sigma = global_baseline(waveforms, P.pulse_lo)
    F = compute_features(waveforms, base, sigma, P)
    cls = F["cls"]
    N = len(waveforms)

    # ---- cutflow: additive (priority PILEUP > SATURATED > NOISE > CLEAN) ----
    real_fail = np.array([info["center_peak"] is None for info in cls["info"]])
    print("\n" + "=" * 70)
    print("CUTFLOW  (baseline %.4g, sigma %.4g, window [%d,%d))"
          % (base, sigma, P.pulse_lo, P.pulse_hi))
    print("  priority: PILEUP > SATURATED > NOISE > CLEAN  (rows sum to total)")
    print("=" * 70)
    print(f"  events total ........................... {N:>8,}")
    print(f"  PILEUP (real 2nd pulse) ................ {int(cls['pileup'].sum()):>8,}")
    print(f"  SATURATED (of the rest, rail {cls['sat_thr']:.0f}) .. {int(cls['sat'].sum()):>8,}")
    print(f"  NOISE (of the rest) ................... {int(cls['noise'].sum()):>8,}")
    print(f"  CLEAN ................................. {int(cls['clean'].sum()):>8,}   "
          f"({100*cls['clean'].mean():.1f}%)")
    print("  " + "-" * 60)
    print("  diagnostic sub-reasons (may overlap with the above):")
    print(f"    no real pulse in window ............. {int(real_fail.sum()):>8,}   "
          f"(height<{P.noise_prominence}s / weak edge / not dominant)")
    print(f"    near the saturation rail ............ {int((F['row_max'] >= cls['sat_thr']*(1-P.rail_tol)).sum()):>8,}")
    print("=" * 70 + "\n")

    # ---- per-gate breakdown of rejected extra-pulse candidates ----
    n_ex = int(F["ex_h"].size)
    pct = (lambda k: f"{100*k/n_ex:4.1f}%" if n_ex else "  n/a")
    print("Extra-pulse candidate gate breakdown:")
    print(f"  total candidates examined:     {n_ex:>6,}")
    print(f"  failed height (< extra_thresh): {int(F['ex_fail_height'].sum()):>6,}  ({pct(int(F['ex_fail_height'].sum()))})")
    print(f"  failed shape (_is_real_pulse):  {int(F['ex_fail_shape'].sum()):>6,}  ({pct(int(F['ex_fail_shape'].sum()))})")
    print(f"  failed post-pulse veto:         {int(F['ex_fail_veto'].sum()):>6,}  ({pct(int(F['ex_fail_veto'].sum()))})")
    print(f"  failed undershoot:              {int(F['ex_fail_undershoot'].sum()):>6,}  ({pct(int(F['ex_fail_undershoot'].sum()))})")
    print(f"  passed all gates (kept):        {int(F['ex_passed'].sum()):>6,}  ({pct(int(F['ex_passed'].sum()))})")
    print("  (candidates can fail multiple gates)\n")

    if not (args.save_plot or not args.no_show):
        return
    plt = _setup_mpl(not args.no_show)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # 1: pulse-window height (noise_prominence)
    a = ax[0, 0]
    cs = np.clip(F["center_sigma"], 0, np.percentile(F["center_sigma"], 99.5))
    a.hist(cs, bins=120, color="C0", alpha=0.8)
    a.axvline(P.noise_prominence, color="r", ls="--", label=f"noise_prom={P.noise_prominence}")
    a.set_yscale("log"); a.set(xlabel="pulse-window height / sigma", ylabel="events",
                               title="CUT: real-pulse height\n(below line -> NOISE)")
    a.legend(); a.grid(True, alpha=0.3)

    # 2: leading-edge rise fraction (_is_real_pulse)
    a = ax[0, 1]
    a.hist(np.clip(F["rise_frac"], 0, 1.2), bins=120, color="C0", alpha=0.8)
    a.axvline(0.5, color="r", ls="--", label="rise_frac=0.5")
    a.set_yscale("log"); a.set(xlabel="(h - foot)/h within rise window", ylabel="events",
                               title="CUT: leading-edge shape\n(left of line -> NOISE)")
    a.legend(); a.grid(True, alpha=0.3)

    # 3: dominance peak count (_pulse_dominates / max_extra)
    a = ax[0, 2]
    mx = int(min(F["dom_count"].max(), 25))
    a.hist(np.clip(F["dom_count"], 0, mx), bins=np.arange(0, mx + 2) - 0.5, color="C0", alpha=0.8)
    a.axvline(P.max_peaks + 0.5, color="r", ls="--", label=f"dominance keeps <={P.max_peaks}")
    a.set_yscale("log"); a.set(xlabel="# peaks above max(0.5 h, dom_floor*sigma)", ylabel="events",
                               title="CUT: record dominance\n(right of line -> NOISE)")
    a.legend(); a.grid(True, alpha=0.3)

    # 4: saturation flat-top run length
    a = ax[1, 0]
    rail_band = cls["sat_thr"] * (1 - P.rail_tol)
    near = F["row_max"] >= rail_band
    runs = longest_near_rail_run(waveforms[near], rail_band) if near.any() else np.array([0])
    a.hist(runs, bins=np.arange(0, max(runs.max(), P.consec + 2) + 1) - 0.5, color="C0", alpha=0.8)
    a.axvline(P.consec - 0.5, color="r", ls="--", label=f"consec={P.consec}")
    a.set_yscale("log"); a.set(xlabel="longest run at rail (samples)", ylabel="near-rail events",
                               title="CUT: saturation flat-top\n(right of line -> SATURATED)")
    a.legend(); a.grid(True, alpha=0.3)

    # 5: extra-pulse height & fraction, colored by the FIRST failing gate
    #    (priority height -> shape -> veto -> undershoot; green = passed all)
    a = ax[1, 1]
    if F["ex_h"].size:
        from matplotlib.lines import Line2D
        col = np.full(F["ex_h"].size, "green", dtype=object)
        col[F["ex_fail_undershoot"]] = "purple"   # assigned low-to-high priority so
        col[F["ex_fail_veto"]] = "orange"          # the highest-priority failing gate
        col[F["ex_fail_shape"]] = "blue"           # ends up as the final color
        col[F["ex_fail_height"]] = "red"
        a.scatter(F["ex_h"], F["ex_frac"], s=8, alpha=0.5, c=list(col))
        vline = a.axvline(P.extra_min_sigma, color="k", ls="--", label=f"extra_min={P.extra_min_sigma}s")
        hline = a.axhline(P.extra_frac, color="k", ls=":", label=f"extra_frac={P.extra_frac}")
        gate_handles = [
            Line2D([0], [0], marker="o", ls="", color="green", label="passed (kept)"),
            Line2D([0], [0], marker="o", ls="", color="red", label="fail height"),
            Line2D([0], [0], marker="o", ls="", color="blue", label="fail shape"),
            Line2D([0], [0], marker="o", ls="", color="orange", label="fail veto"),
            Line2D([0], [0], marker="o", ls="", color="purple", label="fail undershoot"),
        ]
        a.legend(handles=gate_handles + [vline, hline], fontsize=7, loc="best")
    a.set(xlabel="extra height / sigma", ylabel="extra / main height",
          title="CUT: pile-up extra size\n(colored by first failing gate)")
    a.grid(True, alpha=0.3)

    # 6: extra undershoot (baseline recovery)
    a = ax[1, 2]
    if F["ex_under"].size:
        a.hist(np.clip(F["ex_under"], -40, 5), bins=80, color="C0", alpha=0.8)
        a.axvline(-P.undershoot_sigma, color="r", ls="--", label=f"-undershoot={-P.undershoot_sigma}s")
    a.set_yscale("log"); a.set(xlabel="min baseline near extra / sigma", ylabel="extra candidates",
                               title="CUT: extra baseline recovery\n(left of line -> not pile-up)")
    a.legend(); a.grid(True, alpha=0.3)

    fig.suptitle(f"Triage cut diagnostics   |   {counts_line(cls)}", fontsize=13)
    fig.tight_layout()
    _finish(fig, args.save_plot, not args.no_show)


# ===========================================================================
# Sub-command: scan  (sweep one parameter)
# ===========================================================================

def cmd_scan(waveforms, args):
    if args.param not in SCAN_PARAMS:
        raise SystemExit(f"--param must be one of: {', '.join(SCAN_PARAMS)}")
    is_int = SCAN_PARAMS[args.param]
    base_P = params_from_args(args)          # detector preset baseline for non-swept params
    base, sigma = global_baseline(waveforms, base_P.pulse_lo)
    vals = np.linspace(args.lo, args.hi, args.n)
    if is_int:
        vals = np.unique(np.round(vals).astype(int))

    rows = []
    print(f"\nScanning {args.param} over {list(vals)} (recomputing classification each step)\n")
    for v in vals:
        P = replace(base_P, **{args.param: (int(v) if is_int else float(v))})
        b, s = (base, sigma)
        if args.param in ("pulse_lo", "pulse_hi"):     # window change moves baseline region
            b, s = global_baseline(waveforms, P.pulse_lo)
        cls = classify_all(waveforms, b, s, P)
        rows.append((v, int(cls["clean"].sum()), int(cls["sat"].sum()),
                     int(cls["pileup"].sum()), int(cls["noise"].sum())))
        print(f"  {args.param}={v:<8}  clean={rows[-1][1]:>6,}  sat={rows[-1][2]:>5,}  "
              f"pileup={rows[-1][3]:>4,}  noise={rows[-1][4]:>6,}")
    rows = np.array(rows, dtype=float)

    if args.save_plot or not args.no_show:
        plt = _setup_mpl(not args.no_show)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for k, (name, col) in enumerate([("clean", "C2"), ("sat", "C3"),
                                         ("pileup", "C1"), ("noise", "C7")], start=1):
            ax[0].plot(rows[:, 0], rows[:, k], "o-", color=col, label=name)
        ax[0].set(xlabel=args.param, ylabel="events", title="Class counts vs parameter")
        ax[0].legend(); ax[0].grid(True, alpha=0.3)
        # zoom on the rare classes
        for k, (name, col) in enumerate([("sat", "C3"), ("pileup", "C1")], start=2):
            ax[1].plot(rows[:, 0], rows[:, k], "o-", color=col, label=name)
        ax[1].set(xlabel=args.param, ylabel="events", title="Rare classes (zoom)")
        ax[1].legend(); ax[1].grid(True, alpha=0.3)
        fig.suptitle(f"Parameter scan: {args.param}"); fig.tight_layout()
        _finish(fig, args.save_plot, not args.no_show)


# ===========================================================================
# Sub-command: baseline  (local vs global)
# ===========================================================================

def cmd_baseline(waveforms, args):
    P = params_from_args(args)
    g_base, g_sigma = global_baseline(waveforms, P.pulse_lo)
    l_base, l_sigma = local_baseline(waveforms, P.pulse_lo)
    N = len(waveforms)
    win = P.pulse_hi - P.pulse_lo

    # Charge under each baseline convention (per-event vs single global level).
    raw_sum = waveforms[:, P.pulse_lo:P.pulse_hi].sum(axis=1)
    charge_global = raw_sum - g_base * win
    charge_local = raw_sum - l_base * win

    # How many events would cross the first (most baseline-sensitive) cut differently?
    peak_pos = waveforms[:, P.pulse_lo:P.pulse_hi].argmax(axis=1) + P.pulse_lo
    ph = waveforms[np.arange(N), peak_pos]
    cross_global = (ph - g_base) >= P.noise_prominence * g_sigma
    cross_local = (ph - l_base) >= P.noise_prominence * l_sigma
    flipped = int((cross_global != cross_local).sum())

    # Drift: median baseline within consecutive blocks of events.  Per-event
    # baseline scatters with noise, so a robust per-block median exposes any slow
    # trend across the run that a single global value would miss.
    n_blocks = min(50, max(N // 100, 1))
    edges = np.linspace(0, N, n_blocks + 1).astype(int)
    blk_center = np.array([(edges[k] + edges[k + 1]) / 2 for k in range(n_blocks) if edges[k + 1] > edges[k]])
    blk_median = np.array([np.median(l_base[edges[k]:edges[k + 1]])
                           for k in range(n_blocks) if edges[k + 1] > edges[k]])
    drift_span = float(blk_median.max() - blk_median.min())
    drift_max_dev = float(np.max(np.abs(blk_median - g_base)))

    print("\n" + "=" * 66)
    print("Baseline / noise: GLOBAL (pooled) vs LOCAL (per event)")
    print("=" * 66)
    print(f"  GLOBAL baseline = {g_base:.4g} ADC      sigma = {g_sigma:.4g} ADC")
    print(f"  LOCAL  baseline: median {np.median(l_base):.4g}  "
          f"spread (IQR) {np.subtract(*np.percentile(l_base, [75, 25])):.3g}  "
          f"min/max {l_base.min():.4g}/{l_base.max():.4g}")
    print(f"  LOCAL  sigma   : median {np.median(l_sigma):.4g}  "
          f"IQR {np.subtract(*np.percentile(l_sigma, [75, 25])):.3g}  "
          f"min/max {l_sigma.min():.4g}/{l_sigma.max():.4g}")
    print(f"  events crossing the {P.noise_prominence}-sigma pulse cut differently "
          f"(global vs local): {flipped:,}  ({100*flipped/N:.2f}%)")
    print(f"  charge spread (robust sigma) GLOBAL={1.4826*np.median(np.abs(charge_global-np.median(charge_global))):.4g}  "
          f"LOCAL={1.4826*np.median(np.abs(charge_local-np.median(charge_local))):.4g}")
    print(f"  drift over run ({n_blocks} blocks): block-median span = {drift_span:.3g} ADC, "
          f"max |block - global| = {drift_max_dev:.3g} ADC ({100*drift_max_dev/g_sigma:.1f}% of sigma)")
    print("=" * 66 + "\n")

    if args.save_plot or not args.no_show:
        plt = _setup_mpl(not args.no_show)
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        a = ax[0, 0]
        a.hist(l_base, bins=120, color="C0", alpha=0.8)
        a.axvline(g_base, color="r", ls="--", label=f"global={g_base:.0f}")
        a.set(xlabel="per-event baseline (ADC)", ylabel="events", title="Baseline level"); a.legend(); a.grid(True, alpha=0.3)
        a = ax[0, 1]
        a.hist(l_sigma, bins=120, color="C0", alpha=0.8)
        a.axvline(g_sigma, color="r", ls="--", label=f"global={g_sigma:.2f}")
        a.set(xlabel="per-event noise sigma (ADC)", ylabel="events", title="Noise sigma"); a.legend(); a.grid(True, alpha=0.3)
        a = ax[1, 0]
        a.plot(l_base, ",", alpha=0.3, color="C0")
        a.plot(blk_center, blk_median, "-", color="C1", lw=2, label="block median")
        a.axhline(g_base, color="r", ls="--", label=f"global={g_base:.0f}")
        # Zoom y to the block-median trend so a small drift is actually visible.
        pad = max(2.0 * (blk_median.max() - blk_median.min()), 3.0)
        a.set_ylim(min(blk_median.min(), g_base) - pad, max(blk_median.max(), g_base) + pad)
        a.set(xlabel="event index", ylabel="baseline (ADC)",
              title=f"Baseline drift (span {drift_span:.2g} ADC)"); a.legend(); a.grid(True, alpha=0.3)
        a = ax[1, 1]
        lo, hi = np.percentile(charge_global, [0.5, 99.5])
        bins = np.linspace(lo, hi, 200)
        a.hist(charge_global, bins=bins, histtype="step", color="r", label="global baseline")
        a.hist(charge_local, bins=bins, histtype="step", color="C0", label="local baseline")
        a.set_yscale("log"); a.set(xlabel="window charge (ADC.sample)", ylabel="events",
                                   title="Spectrum: global vs local baseline"); a.legend(); a.grid(True, alpha=0.3)
        fig.suptitle("Baseline estimation: local vs global"); fig.tight_layout()
        _finish(fig, args.save_plot, not args.no_show)


# ===========================================================================
# Sub-command: spectrum  (energy spectrum sculpted by cuts)
# ===========================================================================

def cmd_spectrum(waveforms, args):
    P = params_from_args(args)
    base, sigma = global_baseline(waveforms, P.pulse_lo)
    cls = classify_all(waveforms, base, sigma, P)
    N, L = waveforms.shape

    if args.metric == "charge":
        metric = waveforms[:, P.pulse_lo:P.pulse_hi].sum(axis=1) - base * (P.pulse_hi - P.pulse_lo)
        xlabel = "window charge (ADC.sample)"
    else:
        peak_pos = waveforms[:, P.pulse_lo:P.pulse_hi].argmax(axis=1) + P.pulse_lo
        metric = waveforms[np.arange(N), peak_pos] - base
        xlabel = "pulse height (ADC)"

    pos = metric[metric > 0]
    lo, hi = np.percentile(pos, [0.5, 99.5])
    bins = np.linspace(lo, hi, args.bins)
    centers = 0.5 * (bins[1:] + bins[:-1])

    # Peaks in the CLEAN spectrum (manual finder on the counts).
    clean_counts, _ = np.histogram(metric[cls["clean"]], bins=bins)
    smooth = np.convolve(clean_counts, np.ones(5) / 5, mode="same")
    pk = find_peaks_manual(smooth, height=max(smooth.max() * 0.02, 3),
                           prominence=max(smooth.max() * 0.02, 3),
                           distance=max(args.bins // 40, 2))

    print("\n" + "=" * 60)
    print(f"Spectrum ({args.metric})   {counts_line(cls)}")
    print("=" * 60)
    print(f"  range used: [{lo:.3g}, {hi:.3g}]   bins: {args.bins}")
    if pk.size:
        print("  peaks in CLEAN spectrum at:")
        for p in pk:
            print(f"    {centers[p]:.4g}   (~{int(smooth[p])} counts/bin)")
    else:
        print("  no clear peaks found in the clean spectrum.")
    print("=" * 60 + "\n")

    if args.save_plot or not args.no_show:
        plt = _setup_mpl(not args.no_show)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        # Left: contribution by class (stacked-ish overlay)
        for name, col in (("clean", "C2"), ("sat", "C3"), ("pileup", "C1"), ("noise", "C7")):
            ax[0].hist(metric[cls[name]], bins=bins, histtype="step", color=col, label=name)
        for p in pk:
            ax[0].axvline(centers[p], color="k", ls=":", alpha=0.5)
        ax[0].set_yscale("log"); ax[0].set(xlabel=xlabel, ylabel="events",
                                           title="Spectrum by class (peaks marked)")
        ax[0].legend(); ax[0].grid(True, alpha=0.3)
        # Right: sculpting (cumulative removal)
        keep = np.ones(N, bool)
        ax[1].hist(metric[keep], bins=bins, histtype="step", color="C0", label="all events")
        keep &= ~cls["sat"]
        ax[1].hist(metric[keep], bins=bins, histtype="step", color="C3", label="- saturated")
        keep &= ~cls["noise"]
        ax[1].hist(metric[keep], bins=bins, histtype="step", color="C7", label="- noise")
        keep &= ~cls["pileup"]
        ax[1].hist(metric[keep], bins=bins, histtype="step", color="C2", label="- pileup = CLEAN")
        ax[1].set_yscale("log"); ax[1].set(xlabel=xlabel, ylabel="events",
                                           title="Spectrum sculpted cut-by-cut")
        ax[1].legend(); ax[1].grid(True, alpha=0.3)
        fig.suptitle(f"Energy spectrum ({args.metric})"); fig.tight_layout()
        _finish(fig, args.save_plot, not args.no_show)


# ===========================================================================
# Sub-command: gallery  (browse the waveforms in a triage output file)
# ===========================================================================

def cmd_gallery(waveforms, args):
    """Browse the waveforms in a triage output file (e.g. *_pileup.h5).  Loads the
    file as-is and displays them like waveform_triage's plot_gallery -- 12 per page,
    arrow keys to browse, pulse window shaded -- without re-running triage.

    By default each trace has the display baseline subtracted.  A RAW-ADC mode
    (--raw-adc, or the `r` key) instead shows the values exactly as stored -- not
    baseline-subtracted (and, since the file is left as loaded, never sign-flipped)."""
    plt = _setup_mpl(not args.no_show)
    N, L = waveforms.shape
    name = Path(args.input).name
    pulse_lo = args.pulse_lo if args.pulse_lo is not None else Params().pulse_lo
    pulse_hi = args.pulse_hi if args.pulse_hi is not None else Params().pulse_hi
    if N == 0:
        logger.info("No events in %s; nothing to show.", name)
        return
    base = float(np.median(waveforms[:, :max(pulse_lo, 5)]))   # display baseline only
    per_page = 12
    n_all = max(int(np.ceil(N / per_page)), 1)
    view = {"raw": bool(getattr(args, "raw_adc", False))}

    def draw(fig, page):
        raw_on = view["raw"]
        page_idx = np.arange(page * per_page, min((page + 1) * per_page, N))
        rows = max(int(np.ceil(len(page_idx) / 3)), 1)
        axes = fig.subplots(rows, 3, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for k, ev in enumerate(page_idx):
            ax = axes[k // 3][k % 3]; ax.axis("on")
            trace = waveforms[ev] if raw_on else waveforms[ev] - base
            ax.plot(trace, color="C0", lw=0.8)
            ax.axvspan(pulse_lo, pulse_hi, color="gray", alpha=0.1)
            ax.set_title(f"event {ev}", fontsize=8); ax.grid(True, alpha=0.3)
        mode = "raw ADC" if raw_on else "baseline-sub."
        hint = "r=display" if raw_on else "r=raw ADC"
        fig.suptitle(f"{name}  ({N:,} events)   page {page + 1}/{n_all}   {mode}   "
                     f"|  <- -> or n/p to browse, {hint}, Esc/Q to close", fontsize=11)
        fig.tight_layout()

    if not args.no_show:
        def _toggle_raw():
            view["raw"] = not view["raw"]

        paged_figure(n_all, draw, figsize=(14, 12), extra_keys={"r": _toggle_raw})

    if args.save_plot is not None:
        fig = plt.figure(figsize=(14, 12)); draw(fig, 0)
        Path(args.save_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_plot, dpi=200, bbox_inches="tight")
        logger.info("Saved %s", args.save_plot)
        plt.close(fig)


# ===========================================================================
# Sub-command: all  (run every diagnostic in one go)
# ===========================================================================

def run_all(waveforms, out_dir, show=False, pulse_lo=None, pulse_hi=None,
            full=False, cut_overrides: dict | None = None):
    """Run the diagnostics and save diag_*.png into out_dir.  Importable so
    waveform_triage can call it after a run.

    `cut_overrides` carries the ACTUAL cut values the caller classified with
    (e.g. the PMT preset's noise_prominence / consec / min_separation), so the
    diagnostics reproduce that classification instead of silently reverting to
    the SiPM defaults.

    Core QA diagnostics (always): window placement and the per-cut cutflow --
    these tell you whether the cuts are correctly placed.  The sculpted-spectrum
    plot is exploratory and runs only with `full=True` (--full-diagnostics).
    The one-parameter sensitivity scan and the local-vs-global baseline study
    were dropped from the routine set (flat / long-settled on real data); they
    remain available as the `scan` and `baseline` sub-commands for ad-hoc use."""
    from types import SimpleNamespace
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ns = not show
    P = Params()
    plo = pulse_lo if pulse_lo is not None else P.pulse_lo
    phi = pulse_hi if pulse_hi is not None else P.pulse_hi
    overrides = dict(cut_overrides or {})

    print("\n" + "#" * 70)
    print("# Triage diagnostics  ->  " + str(out) + ("   (full)" if full else ""))
    print("#" * 70)

    def cut_ns(**kw):
        """Namespace with every Params field (None = preset default), the caller's
        actual cut values applied, then any explicit kw on top."""
        ns_obj = SimpleNamespace()
        for f in P.__dataclass_fields__:
            setattr(ns_obj, f, None)
        for k, v in overrides.items():
            setattr(ns_obj, k, v)
        ns_obj.pulse_lo, ns_obj.pulse_hi = plo, phi
        for k, v in kw.items():
            setattr(ns_obj, k, v)
        return ns_obj

    # ---- Core QA (always) ----
    cmd_window(waveforms, SimpleNamespace(
        save_plot=out / "diag_window.png", no_show=ns, pulse_lo=plo, pulse_hi=phi,
        min_sigma=8.0, coverage=0.99, pad=5))

    cmd_features(waveforms, cut_ns(save_plot=out / "diag_features.png", no_show=ns))

    # ---- Exploratory (opt-in) ----
    if full:
        cmd_spectrum(waveforms, cut_ns(save_plot=out / "diag_spectrum.png", no_show=ns,
                                       metric="charge", bins=200))


def cmd_all(waveforms, args):
    # The explicit 'all' sub-command runs the routine diagnostics (window, features,
    # spectrum); the scan / baseline studies are ad-hoc sub-commands only.
    out = args.save_dir if args.save_dir is not None else Path(args.input).parent
    run_all(waveforms, out, show=not args.no_show, full=True)


# ===========================================================================
# CLI
# ===========================================================================

def params_from_args(args) -> Params:
    # Start from the detector preset (shared with waveform_triage), then let any
    # value explicitly set on the CLI override it -- so --detector pmt gives the
    # PMT cut defaults and the diagnostics match a triage PMT-mode run.
    P = Params()
    preset = wt.DETECTOR_PRESETS[getattr(args, "detector", "sipm")]
    for key in ("noise_prominence", "consec", "min_separation"):
        setattr(P, key, preset[key])
    for f in P.__dataclass_fields__:
        if hasattr(args, f) and getattr(args, f) is not None:
            setattr(P, f, getattr(args, f))
    return P


def _add_common(sp):
    sp.add_argument("--input", type=Path, required=True, help="Input .h5 / .hdf5 file.")
    sp.add_argument("--save-plot", type=Path, default=None, help="Save the figure to this path.")
    sp.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    sp.add_argument("--detector", choices=["sipm", "pmt"], default="sipm",
                    help="Detector readout preset (mirrors waveform_triage). 'pmt' sets "
                         "polarity=negative and the PMT cut defaults (higher noise_prominence, "
                         "lower consec, smaller min_separation); explicit flags still override.")
    sp.add_argument("--polarity", choices=["positive", "negative", "auto"], default=None,
                    help="Pulse polarity. Default follows --detector (sipm->positive, pmt->negative). "
                         "'negative' (PMT) records are reflected about the baseline so the "
                         "diagnostics match waveform_triage; 'auto' detects it.")
    sp.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_cut_overrides(sp):
    """Let any triage parameter be overridden so its effect can be inspected."""
    sp.add_argument("--pulse-lo", dest="pulse_lo", type=int, default=None)
    sp.add_argument("--pulse-hi", dest="pulse_hi", type=int, default=None)
    sp.add_argument("--noise-prominence", dest="noise_prominence", type=float, default=None)
    sp.add_argument("--pileup-prominence", dest="pileup_prominence", type=float, default=None)
    sp.add_argument("--extra-frac", dest="extra_frac", type=float, default=None)
    sp.add_argument("--extra-min-sigma", dest="extra_min_sigma", type=float, default=None)
    sp.add_argument("--min-separation", dest="min_separation", type=int, default=None)
    sp.add_argument("--max-extra-pulses", dest="max_extra_pulses", type=int, default=None)
    sp.add_argument("--post-pulse-veto", dest="post_pulse_veto", type=int, default=None)
    sp.add_argument("--undershoot-sigma", dest="undershoot_sigma", type=float, default=None)
    sp.add_argument("--undershoot-window", dest="undershoot_window", type=int, default=None)
    sp.add_argument("--rail-tol", dest="rail_tol", type=float, default=None)
    sp.add_argument("--max-peaks", dest="max_peaks", type=int, default=None)
    sp.add_argument("--dom-floor-sigma", dest="dom_floor_sigma", type=float, default=None)
    sp.add_argument("--consec", dest="consec", type=int, default=None)


def parse_args():
    p = argparse.ArgumentParser(description="Diagnostics for the waveform_triage cuts.")
    sub = p.add_subparsers(dest="mode", required=True)

    w = sub.add_parser("window", help="Recommend / validate the pulse window.")
    _add_common(w)
    w.add_argument("--pulse-lo", type=int, default=None)
    w.add_argument("--pulse-hi", type=int, default=None)
    w.add_argument("--min-sigma", type=float, default=8.0)
    w.add_argument("--coverage", type=float, default=0.99)
    w.add_argument("--pad", type=int, default=5)

    f = sub.add_parser("features", help="Per-cut distributions + cutflow.")
    _add_common(f); _add_cut_overrides(f)

    s = sub.add_parser("scan", help="Sweep one parameter.")
    _add_common(s)
    s.add_argument("--param", required=True, help=f"One of: {', '.join(SCAN_PARAMS)}")
    s.add_argument("--lo", type=float, required=True)
    s.add_argument("--hi", type=float, required=True)
    s.add_argument("--n", type=int, default=13)

    b = sub.add_parser("baseline", help="Local vs global baseline / MAD.")
    _add_common(b); _add_cut_overrides(b)

    sp = sub.add_parser("spectrum", help="Energy spectrum sculpted by cuts.")
    _add_common(sp); _add_cut_overrides(sp)
    sp.add_argument("--metric", choices=["charge", "height"], default="charge")
    sp.add_argument("--bins", type=int, default=200)

    a = sub.add_parser("all", help="Run the routine diagnostics (window, features, "
                                   "spectrum) and save the figures.")
    _add_common(a)
    a.add_argument("--save-dir", type=Path, default=None,
                   help="Directory for diag_*.png (default: the input file's folder).")

    g = sub.add_parser("gallery", help="Browse waveforms in a triage output file.")
    _add_common(g)
    g.add_argument("--pulse-lo", type=int, default=None, help="Pulse window start to shade.")
    g.add_argument("--pulse-hi", type=int, default=None, help="Pulse window end to shade.")
    g.add_argument("--raw-adc", dest="raw_adc", action="store_true",
                   help="Start in RAW-ADC mode: show the stored values without subtracting "
                        "the display baseline (never sign-flipped). Toggle live with the 'r' key.")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    # Bare filename -> waveform_files/, including its per-run folders (the same
    # lookup waveform_triage does); an explicit path is used as given.
    args.input = resolve_input(args.input)
    waveforms, _ = wt.load_waveforms(args.input)

    # Orient negative (PMT) records up-front so every sub-command sees up-going
    # pulses, exactly as waveform_triage does internally.  The 'gallery' mode just
    # displays a triage OUTPUT file (already raw of a known polarity); leave it as
    # loaded so what is shown matches the file on disk.
    # Polarity defaults to the --detector preset (sipm->positive, pmt->negative).
    polarity = getattr(args, "polarity", None)
    if polarity is None:
        polarity = wt.DETECTOR_PRESETS[getattr(args, "detector", "sipm")]["polarity"]
    if args.mode != "gallery" and polarity != "positive":
        plo = getattr(args, "pulse_lo", None) or Params().pulse_lo
        base, _ = global_baseline(waveforms, plo)
        polarity = wt.resolve_polarity(polarity, waveforms)
        waveforms = wt.orient_waveforms(waveforms, polarity, base)
        logger.info("Polarity %s: analysing the baseline-reflected (up-going) record.", polarity)

    {"window": cmd_window, "features": cmd_features, "scan": cmd_scan,
     "baseline": cmd_baseline, "spectrum": cmd_spectrum, "all": cmd_all,
     "gallery": cmd_gallery}[args.mode](waveforms, args)


if __name__ == "__main__":
    main()
