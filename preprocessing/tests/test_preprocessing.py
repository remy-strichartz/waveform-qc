#!/usr/bin/env python3
"""
Automated regression tests for the preprocessing (triage) layer.

Everything here runs on SYNTHETIC waveforms with known truth -- no real dataset is
read.  This layer decides WHAT EACH EVENT IS, and every failure mode it has is
quiet: a misclassified event does not crash anything, it just moves a number
downstream.  So the tests pin the contracts that would otherwise rot in silence:

  * PEAKFIND       peakfind.find_peaks_manual claims to be a drop-in subset of
                   scipy.signal.find_peaks, applying height -> distance ->
                   prominence in scipy's own order.  That is a falsifiable claim
                   about an external library, so it is checked AGAINST scipy on
                   randomised signals rather than against a hand-picked case.

  * FOUR CLASSES   classify_events must label a known-clean pulse CLEAN, a
                   flat-topped one SATURATED, a double pulse PILEUP, and a
                   pulse-free record NOISE.  This is the primitive the whole
                   project stands on (energy_reconstruction imports it).

  * SAT != HEIGHT  saturation is a FLAT TOP, not "tall".  A very tall but pointed
                   pulse must NOT be called saturated -- the 2026-07-14 audit found
                   a height-only test doing exactly that, which quietly deleted the
                   brightest real muons from a spectrum.

  * NOISE IS THE   hodoscope efficiency counts a panel as having SEEN a particle
    ONLY MISS      when its class is anything but NOISE.  SATURATED and PILEUP are
                   real crossings with spoiled charge; counting them as misses
                   silently understates the efficiency.  Pinned here because it is
                   an easy, invisible thing to "clean up" wrongly.

  * WILSON         the interval must stay inside [0, 1] at k == 0 and k == n, where
                   the naive normal interval does not, and must bracket p.

Run:  python preprocessing/tests/test_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from common import peakfind as pf                            # noqa: E402
from common import waveform_ops as wt                        # noqa: E402
from preprocessing import hodoscope_efficiency as hodo       # noqa: E402


# ---------------------------------------------------------------------------------
# Synthetic waveform builders -- one knob per triage class
# ---------------------------------------------------------------------------------

N_SAMP   = 1024        # the real record length; see PILEUP below -- 400 is too short to
PULSE_LO = 150         # even HOLD a second particle, because of the post-pulse veto
PULSE_HI = 250
BASELINE = 100.0
SIGMA    = 2.0
RAIL     = 4095.0

# classify_events(post_pulse_veto=300) deliberately IGNORES extra pulses within 300
# samples after the main one: those are afterpulses and cable reflections, not a
# second particle.  So a synthetic PILEUP event has to place its second pulse beyond
# that veto or the classifier will -- correctly -- call it CLEAN.
SECOND_PULSE_AT = 800


def _pulse(t0: float, amp: float, rise: float = 3.0, decay: float = 25.0) -> np.ndarray:
    """A one-sided exponential pulse, pointing UP (triage works on oriented arrays)."""
    t = np.arange(N_SAMP, dtype=float)
    s = np.zeros(N_SAMP)
    m = t >= t0
    s[m] = amp * (1.0 - np.exp(-(t[m] - t0) / rise)) * np.exp(-(t[m] - t0) / decay)
    return s


def _events(rng, kind: str, n: int = 40) -> np.ndarray:
    """n synthetic events of one triage class, as float waveforms."""
    w = BASELINE + rng.normal(0.0, SIGMA, size=(n, N_SAMP))
    for i in range(n):
        if kind == "noise":
            continue                                    # baseline only: no pulse at all
        amp = 300.0 + 40.0 * rng.standard_normal()
        w[i] += _pulse(PULSE_LO + 20 + rng.uniform(-3, 3), amp)
        if kind == "pileup":
            # A second real pulse from a particle that missed the AND -- placed BEYOND
            # post_pulse_veto, or it reads as an afterpulse and the event is CLEAN.
            w[i] += _pulse(SECOND_PULSE_AT + rng.uniform(-5, 5), 0.8 * amp)
        elif kind == "saturated":
            w[i] += _pulse(PULSE_LO + 20, 6000.0)       # drive it hard into the rail
            np.clip(w[i], None, RAIL, out=w[i])         # ... and FLATTEN it there
    return w


def _classify(w: np.ndarray, **kw):
    base, sig = wt.global_baseline_noise(w, PULSE_LO)
    labels, _info, _thr, _peaks = wt.classify_events(
        w, base, sig, PULSE_LO, PULSE_HI, saturation_adc=RAIL, **kw)
    return labels


# ---------------------------------------------------------------------------------
# peakfind: the claim is "drop-in subset of scipy" -- so check it against scipy
# ---------------------------------------------------------------------------------

def test_peakfind_matches_scipy():
    """peakfind reimplements find_peaks so the cuts are inspectable.  That is only
    worth anything if it AGREES with scipy -- including the filter ORDER (height ->
    distance -> prominence), which changes which peaks survive."""
    from scipy.signal import find_peaks

    rng = np.random.default_rng(20260714)
    checked = 0
    for _ in range(60):
        x = rng.normal(0, 1, size=500)
        x += 6.0 * np.exp(-0.5 * ((np.arange(500) - rng.integers(50, 450)) / 4.0) ** 2)
        for kwargs in ({}, {"height": 1.0}, {"distance": 12},
                       {"prominence": 1.5}, {"height": 0.5, "distance": 8},
                       {"height": 0.5, "distance": 8, "prominence": 1.0}):
            want, _ = find_peaks(x, **kwargs)
            got = pf.find_peaks_manual(x, **kwargs)
            assert np.array_equal(np.asarray(got), want), (
                f"peakfind disagrees with scipy for {kwargs}: "
                f"{np.asarray(got)[:8]} vs {want[:8]}")
            checked += 1

    # A flat top must report the plateau MIDPOINT, as scipy does -- this is how a
    # saturated pulse's peak position is found, so an off-by-one here walks timing.
    flat = np.array([0., 1., 5., 5., 5., 5., 1., 0.])
    assert np.array_equal(np.asarray(pf.find_peaks_manual(flat)),
                          find_peaks(flat)[0]), "plateau midpoint differs from scipy"
    assert checked >= 300


# ---------------------------------------------------------------------------------
# The four classes
# ---------------------------------------------------------------------------------

def test_classify_events_labels_each_class():
    """The primitive the rest of the project is built on: a clean pulse, a clipped
    pulse, a double pulse and an empty record must land in their own buckets."""
    rng = np.random.default_rng(7)
    for kind, want in (("clean", "CLEAN"), ("saturated", "SATURATED"),
                       ("pileup", "PILEUP"), ("noise", "NOISE")):
        labels = _classify(_events(rng, kind))
        frac = float(np.mean(labels == want))
        assert frac >= 0.9, (
            f"{kind!r} events were labelled {want} only {frac:.0%} of the time; "
            f"got {dict(zip(*np.unique(labels, return_counts=True)))}")
        assert set(np.unique(labels)) <= set(wt.CLASS_LABELS)


def test_saturation_is_a_flat_top_not_a_height():
    """A tall pulse is not a clipped pulse.  The rail test must key on the FLAT TOP:
    an audit found a height-only test calling the brightest real muons SATURATED,
    which silently deletes the top of every spectrum."""
    rng = np.random.default_rng(11)

    # Very tall, but POINTED and comfortably under the rail.
    tall = BASELINE + rng.normal(0.0, SIGMA, size=(40, N_SAMP))
    for i in range(40):
        tall[i] += _pulse(PULSE_LO + 20, 0.85 * (RAIL - BASELINE))
    assert tall.max() < RAIL
    labels = _classify(tall)
    assert not np.any(labels == "SATURATED"), (
        "a tall but pointed pulse was called SATURATED -- the cut is keying on "
        "height, not on a flat top")
    assert np.mean(labels == "CLEAN") >= 0.9

    # Same peak height, but genuinely clipped: must flip to SATURATED.
    clipped = _events(rng, "saturated")
    assert np.mean(_classify(clipped) == "SATURATED") >= 0.9


def test_pileup_needs_a_second_real_pulse():
    """PILEUP must mean a second REAL pulse -- not a noise excursion.  If the
    prominence floor slips, ordinary clean events start being thrown away."""
    rng = np.random.default_rng(3)
    assert np.mean(_classify(_events(rng, "clean")) == "PILEUP") <= 0.02


def test_afterpulse_inside_the_veto_is_not_pileup():
    """An extra pulse just AFTER the main one is an afterpulse or a cable reflection
    -- the same detector responding to the same particle -- not a second particle.
    post_pulse_veto exists to say so, and this run00270 channel really does ring.
    Drop the veto and a large slice of perfectly good events reclassifies as PILEUP
    and leaves the spectrum."""
    rng = np.random.default_rng(17)
    w = BASELINE + rng.normal(0.0, SIGMA, size=(40, N_SAMP))
    for i in range(40):
        main = PULSE_LO + 20
        w[i] += _pulse(main, 300.0)
        w[i] += _pulse(main + 120, 150.0)      # a big reflection, WELL inside the veto

    assert np.mean(_classify(w) == "CLEAN") >= 0.9, (
        "an afterpulse inside post_pulse_veto was counted as a second particle")

    # ... and with the veto turned off it IS seen -- proving the veto is what did it,
    # not that the pulse was too small to notice.
    assert np.mean(_classify(w, post_pulse_veto=0) == "PILEUP") >= 0.9


def test_global_baseline_noise_recovers_truth():
    """Baseline = median, sigma = 1.4826*MAD over the PRE-PULSE region only.  If the
    pulse leaks into the window the sigma inflates and every sigma-scaled cut moves."""
    rng = np.random.default_rng(5)
    w = _events(rng, "clean", n=200)
    base, sig = wt.global_baseline_noise(w, PULSE_LO)
    assert abs(base - BASELINE) < 0.5, f"baseline {base} != {BASELINE}"
    assert abs(sig - SIGMA) < 0.3, f"sigma {sig} != {SIGMA} (pulse leaking in?)"


# ---------------------------------------------------------------------------------
# Hodoscope efficiency: the invariant that is easy to "tidy up" wrongly
# ---------------------------------------------------------------------------------

def _panel(name: str, labels) -> hodo.PanelResult:
    labels = np.asarray(labels, dtype=object).astype(str)
    n = len(labels)
    return hodo.PanelResult(name=name, labels=labels,
                            peak_pos=np.full(n, PULSE_LO + 20),
                            edge_pos=np.full(n, PULSE_LO + 10),
                            polarity="positive", baseline=BASELINE, sigma=SIGMA,
                            sat_thr=RAIL, pulse_lo=PULSE_LO, pulse_hi=PULSE_HI)


def test_saturated_and_pileup_count_as_detections():
    """A panel SAW a particle unless its class is NOISE.  SATURATED and PILEUP are
    real crossings whose charge is merely spoiled -- counting either as a miss
    understates the efficiency, and nothing would crash if someone did."""
    mid = _panel("middle", ["CLEAN", "SATURATED", "PILEUP", "NOISE"])
    assert list(mid.saw_pulse) == [True, True, True, False]

    top = _panel("top",    ["CLEAN"] * 4)
    bot = _panel("bottom", ["CLEAN"] * 4)
    rep = hodo.compute_efficiency(top, mid, bot, z=1.96, denominator="all")

    # 3 of 4 crossings hit the middle -- only the NOISE event is a miss.
    assert rep.primary.k == 3 and rep.primary.n == 4
    assert abs(rep.primary.point - 0.75) < 1e-12, (
        f"efficiency {rep.primary.point} != 3/4 -- SATURATED/PILEUP are being "
        "treated as misses")

    # The CLEAN-only cross-check is stricter BY DESIGN and must not be the headline.
    assert rep.clean_only.k == 1
    assert rep.clean_only.point < rep.primary.point


def test_denominator_all_vs_coincidence():
    """'all' trusts the hardware AND-trigger (every event is a crossing); the
    'coincidence' re-derives it offline.  A NOISE top panel must shrink only the
    latter's denominator -- confusing the two silently rescales the efficiency."""
    top = _panel("top",    ["CLEAN", "CLEAN", "NOISE"])
    mid = _panel("middle", ["CLEAN", "NOISE", "CLEAN"])
    bot = _panel("bottom", ["CLEAN", "CLEAN", "CLEAN"])

    all_ = hodo.compute_efficiency(top, mid, bot, z=1.96, denominator="all")
    assert (all_.primary.k, all_.primary.n) == (2, 3)

    coin = hodo.compute_efficiency(top, mid, bot, z=1.96, denominator="coincidence")
    assert (coin.primary.k, coin.primary.n) == (1, 2), (
        "the event whose TOP panel saw nothing is not a crossing and must leave "
        "the coincidence denominator")

    try:
        hodo.compute_efficiency(top, mid, bot, z=1.96, denominator="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown denominator must raise, not pick a default")


def test_wilson_interval_stays_in_bounds():
    """The reason Wilson is used instead of the normal interval: it must not run
    outside [0, 1] at the edges, which is exactly where a good detector sits."""
    perfect = hodo.wilson_interval(50, 50)
    assert perfect.point == 1.0
    assert perfect.hi <= 1.0 and perfect.lo < 1.0, "interval escaped [0, 1] at k == n"

    empty = hodo.wilson_interval(0, 50)
    assert empty.point == 0.0
    assert empty.lo >= 0.0 and empty.hi > 0.0, "interval escaped [0, 1] at k == 0"

    mid = hodo.wilson_interval(40, 50)
    assert mid.lo < mid.point < mid.hi, "the band must bracket the point estimate"
    assert 0.0 <= mid.lo and mid.hi <= 1.0

    undef = hodo.wilson_interval(0, 0)
    assert np.isnan(undef.point) and (undef.lo, undef.hi) == (0.0, 1.0), (
        "n == 0 must be undefined with a full [0, 1] band, not a divide-by-zero")

    # Tighter statistics must give a tighter band.
    assert (hodo.wilson_interval(400, 500).hi - hodo.wilson_interval(400, 500).lo) \
        < (mid.hi - mid.lo)


def test_classification_is_deterministic():
    """Same waveforms in, same labels out -- no RNG, no ordering dependence.  The
    whole audit trail assumes a rerun reproduces the classification exactly."""
    rng = np.random.default_rng(99)
    w = np.vstack([_events(rng, k, n=10) for k in ("clean", "saturated", "pileup", "noise")])
    assert np.array_equal(_classify(w), _classify(w.copy()))


# ---------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------

def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)} / {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
