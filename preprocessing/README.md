# preprocessing — deciding what each event actually is

## Why this stage exists

`file_manipulation` hands you one channel's waveform windows. Nothing about that file says
which of its events are *physics*. The DAQ's AND trigger fires on a coincidence, but it also
fires on noise; a real muon can arrive with a second particle riding along; a bright one can
drive the ADC into its rail. Downstream, every one of those is a different kind of wrong:

* a **noise** event has no pulse to reconstruct, so it drags a template and a spectrum toward
  nothing;
* a **saturated** event has a flat top, so its charge is corrupt and its amplitude is a lie;
* a **pile-up** event has a *second* pulse the fit does not know about.

So before anything is fitted, each event gets a label. That is this package.

## The one idea: four classes

`waveform_triage.classify_events` puts every event in exactly one bucket:

| class | meaning | is it a real particle? |
|---|---|---|
| `CLEAN` | one real pulse in the pulse window — the coincidence the trigger fired on | yes |
| `SATURATED` | the pulse clips the ADC rail (flat-topped) | yes, but the charge is corrupt |
| `PILEUP` | the real pulse **plus** a second pulse elsewhere in the record | yes, plus an extra |
| `NOISE` | no real pulse in the window — the trigger fired on nothing | **no** |

That distinction is the whole package. `SATURATED` and `PILEUP` are *detections* — the panel
genuinely saw a particle, the charge is merely spoiled — and only `NOISE` is a true miss. Get
that wrong and the efficiency measurement below is wrong with it.

This classification is the shared primitive: `hodoscope_efficiency` and `pulse_window` both
call the real triage functions rather than reimplementing them, and `energy_reconstruction`
imports `classify_events` and re-runs it in memory. **No downstream stage depends
automatically on the exported class files** — `--export` exists for subsets you choose to
analyze as inputs in their own right (the canonical energy-reconstruction batch feeds
`run00270_ch9_clean.h5` and `_ch10_clean.h5` to muon-mode compares that way).

| module | role |
|---|---|
| `waveform_triage.py` | **the triage driver** — reports and exports the classes. |
| `pulse_window.py` | makes the classifier's cuts *visible* — where each threshold sits, what it moves |
| `hodoscope_efficiency.py` | the physics result: middle-panel muon efficiency by the telescope method |

The cuts themselves are not in this repo. They live in `hodoscope_common/waveform_ops.py`,
in the [waveform-io](https://github.com/remy-strichartz/waveform-io) repo — baseline/noise,
the polarity vote, the auto pulse window, the flat-top saturation test and
`classify_events` — because `energy_reconstruction` and
`file_manipulation/channel_diagnostics.py` apply the same ones and must not drift from
these. That shared ownership is exactly why they sit in a repo both sides install rather
than in either one. Also in `hodoscope_common/`: `peakfind.py` (a dependency-free
`scipy.signal.find_peaks`, so no cut hides in an opaque C routine) and `plotting.py`
(shared matplotlib setup, forces `Agg` when only saving).

## waveform_triage.py

```bash
python waveform_triage.py --input run00270_ch9.h5 --detector pmt --save-plots --no-show
python waveform_triage.py --input run00270_ch9.h5 --detector pmt --export   # subsets to disk
```

Classifies and prints; it writes **no files** unless you ask. `--detector {sipm,pmt}` picks a
preset for every threshold you do not set yourself — the PMT panels are fast and the SiPM
panels are not, and a single set of cuts does not serve both.

`--export` writes one raw-waveform `.h5` per class (`<stem>_clean.h5`, `_pileup.h5`,
`_noise.h5`, plus `_not_noise.h5` = everything that saw a particle; `_saturated.h5` only
when a true rail was found) into the results folder. Each carries the input's per-event
time axis, row-filtered to the class, so a subset stays joinable back to the run's
recovered times via `/source_event_index`.

Results land in `preprocessing_results/triage/<stem>_triage_results[_N]/`.

## pulse_window.py

The classifier is a chain of thresholds — pulse-window height, leading-edge shape, record
dominance, saturation flat-top, pile-up extra height / fraction / separation / undershoot. This
tool exists so none of them is a magic number you have to take on faith. It calls the *real*
triage functions — but note the **window**: standalone runs use the fixed default (or
`--pulse-lo/--pulse-hi`), not triage's auto-derived one, so class counts can differ from a
triage run of the same file (ch9: 61 events, 0.4%). Run `window` first to get the derived
window, or invoke via `waveform_triage --diagnostics`, which passes the resolved window and
cuts through — there the match is exact.

| sub-command | what it answers |
|---|---|
| `window` | where should the pulse window be, and does the current one hold? |
| `features` | for each cut: where does the threshold sit in the population it acts on, and how many events does it move? (plus a cutflow table) |
| `scan` | sweep one parameter — how do the four class counts respond? |
| `baseline` | one pooled baseline + MAD noise, or per-event? does it change the spectrum? |
| `spectrum` | the pulse-height spectrum, sculpted cut by cut |
| `all` | the routine set |
| `gallery` | browse the waveforms in a triage output file |

```bash
python pulse_window.py features --input run00270_ch9.h5 --detector pmt --save-plots --no-show
```

## hodoscope_efficiency.py

Measures the **middle** panel's muon detection efficiency in a three-panel telescope, by the
standard method: a muon that fired the top *and* the bottom panel must have crossed the middle
one too.

```
denominator  =  events where TOP and BOTTOM both saw a real pulse
numerator    =  those where the MIDDLE panel also saw one
efficiency   =  numerator / denominator          (Wilson score interval)
```

"Saw a real pulse" is `class != NOISE` — which is why the four-class split above has to be
right. The per-panel class breakdown is printed, so the saturated and pile-up contributions
to the numerator stay auditable rather than buried in a single ratio. (It is deliberately
*not* plotted — the stacked bar hid exactly the small classes it needed to show.)

`--denominator {all,coincidence}` chooses whether every recorded event counts as a coincidence
(correct when the DAQ trigger *is* the AND of top and bottom, which is the usual case here) or
whether the coincidence is re-derived from the waveforms. `--middle-readout` applies the right
triage preset to the middle panel, which need not match the outer two.

Results land in `preprocessing_results/hodoscope/<dataset>_efficiency_results[_N]/`.

Note that DAQ dead time is veto inefficiency this measurement is **blind** to — a muon arriving
while the DAQ is busy is never recorded at all, so it appears in neither the numerator nor the
denominator. The dead-time systematic is quoted separately; see `timing_stability`.

What "middle panel" means on run00270 (geometry established 2026-07-15, arXiv:2505.06129):
the middle panel is one 100×50 cm² prototype with **eight** fiber-swirl mini-modules
(ch0–ch7), and the trigger footprint sits over the **ch0 corner** — so the canonical
`--middle-channel` ch0 measures the response of the mini-module under the footprint, not an
OR over the whole panel. That choice was cross-checked against the other seven mini-modules
(re-measured 2026-07-29 with the shared classifier): ch0-miss events carry ~3–5× less light
in the rest of the bank than ch0 hits (best-sibling pulse-height median 127 vs 611 ADC),
and an OR-of-8 hit definition rescues 161 of the 2,770 ch0 misses — **81.53% → 82.60%**,
a +1.1 pp shift (~1.7× the statistical error) that leaves every conclusion drawn from the
headline intact. The number is a property of the ch0 cell, not of the whole panel. Part
of the remaining inefficiency is plausibly geometric acceptance (the footprint is ~3 cm from
the panel's edge, so angled tracks can clip out the side) rather than detection failure —
the paper quotes 98±1% intrinsic panel efficiency; settling that split would take a
position/angle study, which no tool here performs.

## hodoscope_common/peakfind.py

A small, vectorized stand-in for `scipy.signal.find_peaks`, reproducing the `height`,
`distance` and `prominence` filters *in scipy's order* so results match. It exists so the
triage cuts are fully inspectable and so the diagnostic tools can reuse the exact peak logic
the classifier used, rather than an approximation of it.

## One thing that surprises everyone: the post-pulse veto

A second pulse *just after* the main one is **not** pile-up. It is the same detector still
ringing at the same particle — an afterpulse, or a cable reflection (`run00270_ch9`'s template
has two of them, at roughly +21 and +39 samples). `post_pulse_veto` (300 samples by default)
exists to say so, and extras inside it are ignored.

The consequence is easy to trip over: a record has to be long enough to *hold* a second
particle beyond that veto before `PILEUP` is even reachable. Turn the veto off and a large
slice of perfectly good events reclassifies as `PILEUP` and walks out of the spectrum.

## Tests

```bash
python tests/test_preprocessing.py
```

Synthetic waveforms with known truth; no real data is read. They pin the contracts whose
failure is *silent* rather than loud:

* **`peakfind` is checked against scipy itself** on randomised signals — its whole claim is
  to be a drop-in subset, including the filter order, and that is a falsifiable claim about an
  external library rather than an opinion.
* **Saturation is a flat top, not a height.** A tall-but-pointed pulse must not be called
  `SATURATED`; a height-only test would quietly delete the brightest real muons from every
  spectrum.
* **`SATURATED` and `PILEUP` count as detections** in the efficiency, and only `NOISE` is a
  miss. Nothing would crash if someone "tidied" that up — the efficiency would just be wrong.
* **The Wilson interval stays inside [0, 1]** at `k == 0` and `k == n`, which is exactly where
  a good detector sits and exactly where the naive normal interval escapes.
* **The post-pulse veto**, both ways: an afterpulse inside it is `CLEAN`, and the same event
  with the veto off is `PILEUP`.
