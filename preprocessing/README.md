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
imports `classify_events` and re-runs it in memory. **Nothing downstream reads the exported
class files** — so `--export` is only for when you want the subsets on disk.

| module | role |
|---|---|
| `waveform_triage.py` | **the classifier.** Everything else here is downstream of it. |
| `pulse_window.py` | makes the classifier's cuts *visible* — where each threshold sits, what it moves |
| `hodoscope_efficiency.py` | the physics result: middle-panel muon efficiency by the telescope method |
| `peakfind.py` | a dependency-free `scipy.signal.find_peaks`, so no cut hides in an opaque C routine |
| `plotting.py` | shared matplotlib setup (forces `Agg` when only saving) |

## waveform_triage.py

```bash
python waveform_triage.py --input run00270_ch9.h5 --detector pmt --save-plots --no-show
python waveform_triage.py --input run00270_ch9.h5 --detector pmt --export   # subsets to disk
```

Classifies and prints; it writes **no files** unless you ask. `--detector {sipm,pmt}` picks a
preset for every threshold you do not set yourself — the PMT panels are fast and the SiPM
panels are not, and a single set of cuts does not serve both.

`--export` writes one raw-waveform `.h5` per class (`<stem>_clean.h5`, `_saturated.h5`,
`_pileup.h5`, `_noise.h5`) into the results folder. Each carries the input's per-event time
axis, row-filtered to the class, so a subset stays joinable back to the run's recovered times
via `/source_event_index`.

Results land in `preprocessing_results/triage/<stem>_triage_results[_N]/`.

## pulse_window.py

The classifier is a chain of thresholds — pulse-window height, leading-edge shape, record
dominance, saturation flat-top, pile-up extra height / fraction / separation / undershoot. This
tool exists so none of them is a magic number you have to take on faith. It calls the *real*
triage functions, so what it draws is exactly what the classifier does.

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
right. The per-panel class breakdown is printed and plotted, so the saturated and pile-up
contributions to the numerator stay auditable rather than buried in a single ratio.

`--denominator {all,coincidence}` chooses whether every recorded event counts as a coincidence
(correct when the DAQ trigger *is* the AND of top and bottom, which is the usual case here) or
whether the coincidence is re-derived from the waveforms. `--middle-readout` applies the right
triage preset to the middle panel, which need not match the outer two.

Results land in `preprocessing_results/hodoscope/<dataset>_efficiency_results[_N]/`.

Note that DAQ dead time is veto inefficiency this measurement is **blind** to — a muon arriving
while the DAQ is busy is never recorded at all, so it appears in neither the numerator nor the
denominator. The dead-time systematic is quoted separately; see `timing_stability`.

## peakfind.py

A small, vectorized stand-in for `scipy.signal.find_peaks`, reproducing the `height`,
`distance` and `prominence` filters *in scipy's order* so results match. It exists so the
triage cuts are fully inspectable and so the diagnostic tools can reuse the exact peak logic
the classifier used, rather than an approximation of it.
