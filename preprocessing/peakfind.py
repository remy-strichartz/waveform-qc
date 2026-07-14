#!/usr/bin/env python3
"""
peakfind.py
===========
A small, dependency-free replacement for scipy.signal.find_peaks, written so the
triage cuts are fully transparent (no opaque C routine) and so the diagnostic
tools can reuse the exact same peak logic.

It reproduces the parts of find_peaks the pipeline relies on -- local maxima with
plateau handling, and the `height`, `distance` and `prominence` filters -- and
applies them in the SAME order scipy does (height -> distance -> prominence) so
results match.

Design / efficiency
-------------------
* Local maxima are found by run-length encoding the signal (one np.diff +
  np.flatnonzero), which is fully vectorized and handles flat tops by reporting
  the plateau midpoint, exactly like scipy.
* `height` is a cheap boolean mask, applied first so the more expensive steps see
  only a handful of candidates.
* `distance` is scipy's greedy highest-first suppression.
* `prominence` is computed per surviving peak with vectorized left/right base
  searches (each bounded by the nearest higher sample), so the common case --
  a few tall candidates -- is fast.
"""

from __future__ import annotations

import numpy as np


def local_maxima(x: np.ndarray) -> np.ndarray:
    """Indices of local maxima in 1-D `x`.  Flat tops report the plateau midpoint;
    the first and last samples are never peaks (matching scipy)."""
    n = x.size
    if n < 3:
        return np.empty(0, dtype=np.intp)
    # Run-length encode: each maximal run of equal values is one "level".
    ends = np.flatnonzero(np.diff(x) != 0)          # last index of each run but the last
    if ends.size < 2:                               # 0 or 1 internal edges -> no interior run
        return np.empty(0, dtype=np.intp)
    run_start = np.concatenate(([0], ends + 1))
    run_end = np.concatenate((ends, [n - 1]))
    val = x[run_start]
    m = val.size
    if m < 3:
        return np.empty(0, dtype=np.intp)
    j = np.arange(1, m - 1)                          # interior runs only
    is_pk = (val[j] > val[j - 1]) & (val[j] > val[j + 1])
    jj = j[is_pk]
    return ((run_start[jj] + run_end[jj]) // 2).astype(np.intp)


def _select_by_distance(peaks: np.ndarray, priority: np.ndarray, distance: int) -> np.ndarray:
    """Greedy suppression: keep the highest-priority peak, remove any peak within
    `distance` samples of a kept one, repeat.  `peaks` must be sorted ascending."""
    n = peaks.size
    keep = np.ones(n, dtype=bool)
    for i in np.argsort(priority, kind="stable")[::-1]:   # highest priority first
        if not keep[i]:
            continue
        k = i - 1
        while k >= 0 and peaks[i] - peaks[k] < distance:
            keep[k] = False
            k -= 1
        k = i + 1
        while k < n and peaks[k] - peaks[i] < distance:
            keep[k] = False
            k += 1
    return keep


def prominences(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Topographic prominence of each peak: peak height minus the higher of the two
    'key cols' (the lowest point reached going left/right before a taller sample
    or the signal border)."""
    prom = np.empty(peaks.size, dtype=np.float64)
    for idx in range(peaks.size):
        pk = int(peaks[idx])
        h = x[pk]
        left_higher = np.flatnonzero(x[:pk] > h)
        l0 = int(left_higher[-1]) + 1 if left_higher.size else 0
        left_min = x[l0:pk].min() if pk > l0 else h
        right_higher = np.flatnonzero(x[pk + 1:] > h)
        r1 = pk + 1 + int(right_higher[0]) if right_higher.size else x.size
        right_min = x[pk + 1:r1].min() if r1 > pk + 1 else h
        prom[idx] = h - max(left_min, right_min)
    return prom


def find_peaks_manual(x: np.ndarray, height: float | None = None,
                      prominence: float | None = None,
                      distance: int | None = None,
                      return_prominences: bool = False):
    """Drop-in subset of scipy.signal.find_peaks.

    Returns the peak indices (and, if `return_prominences`, the prominence of each
    surviving peak).  Filters are applied in scipy's order: height, distance,
    prominence.
    """
    peaks = local_maxima(x)

    if height is not None and peaks.size:
        peaks = peaks[x[peaks] >= height]

    if distance is not None and distance > 1 and peaks.size:
        peaks = peaks[_select_by_distance(peaks, x[peaks], int(distance))]

    prom = None
    if prominence is not None and peaks.size:
        prom = prominences(x, peaks)
        mask = prom >= prominence
        peaks, prom = peaks[mask], prom[mask]

    if return_prominences:
        if prom is None:
            prom = prominences(x, peaks) if peaks.size else np.empty(0)
        return peaks, prom
    return peaks
