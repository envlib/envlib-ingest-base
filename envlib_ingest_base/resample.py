"""Source-agnostic resampling of irregular / gappy station telemetry to a fixed cadence.

Two aggregation kinds, matching the two physical measurement types:

- ``'mean'`` (instantaneous signals: river stage, flow) -> an **exact trapezoidal
  time-weighted mean**. The signal is treated as piecewise-linear between consecutive
  observations; each output interval's value is the time-integral of that signal over the
  interval, divided by the covered time. This is exact for *any* spacing (regular or
  irregular) and reduces to the plain hourly mean for regularly-sampled data.

- ``'sum'`` (accumulations: rainfall) -> a right-closed interval sum with ``min_count=1``,
  so an interval with **no** reading is missing (``NaN``), never a fabricated ``0``, while a
  genuinely-reported ``0.0`` is kept.

Gaps/outages are first-class: consecutive observations farther apart than ``max_gap`` are a
hole. The signal is never interpolated across a hole, and output intervals whose covered
fraction is below ``min_coverage`` are dropped (``NaN``) rather than ramped across the gap.

``max_gap`` may be given absolutely, or (default) adapted per series as
``gap_multiplier * median(native interval)`` so a 5-min site and a 60-min site each get a
threshold matched to their own cadence.

All timestamps are treated as an ordered numeric axis; timezone handling is the caller's job
(pass UTC). Output interval labels are **interval starts** (envlib convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_MEAN_POINTS = 2  # need >= 2 observations to form one linear segment


def _freq_ns(freq) -> int:
    return int(pd.Timedelta(freq).value)


def _prep(times, values) -> tuple[np.ndarray, np.ndarray]:
    """Sort, drop NaT/NaN, and collapse duplicate timestamps (by mean). Returns int64-ns t, float v."""
    ts = pd.to_datetime(times)
    v = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype='float64')
    t = np.asarray(ts, dtype='datetime64[ns]')
    ok = ~pd.isna(pd.Series(t)).to_numpy() & np.isfinite(v)
    t = t[ok].astype('int64')
    v = v[ok]
    if t.size == 0:
        return t, v
    order = np.argsort(t, kind='stable')
    t, v = t[order], v[order]
    if t.size > 1 and np.any(np.diff(t) == 0):
        ut, inv = np.unique(t, return_inverse=True)
        v = np.bincount(inv, weights=v) / np.bincount(inv)
        t = ut
    return t, v


def _resolve_max_gap(t: np.ndarray, max_gap, gap_multiplier: float, freq_ns: int) -> float:
    """Absolute max_gap in ns if given, else adaptive gap_multiplier * median native interval."""
    if max_gap is not None:
        return float(pd.Timedelta(max_gap).value)
    if t.size < _MIN_MEAN_POINTS:
        return float(freq_ns)
    med = float(np.median(np.diff(t)))
    return max(med * gap_multiplier, float(freq_ns))  # never below one output interval


def resample_mean(times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0, min_coverage=0.5) -> pd.Series:
    """Exact trapezoidal time-weighted mean of an instantaneous signal onto a fixed cadence.

    Returns a Series indexed by interval-start (datetime64), gap/under-covered intervals absent.
    """
    freq_ns = _freq_ns(freq)
    t, v = _prep(times, values)
    empty = pd.Series(dtype='float64')
    empty.index = pd.DatetimeIndex([], name='time')
    if t.size < _MIN_MEAN_POINTS:
        return empty
    max_gap_ns = _resolve_max_gap(t, max_gap, gap_multiplier, freq_ns)

    seg_dur = np.diff(t)  # (n-1,)
    seg_gap = seg_dur > max_gap_ns

    # interval boundaries strictly inside (t0, t_last); these become cut points so no
    # sub-segment ever straddles a boundary (=> each sub-segment lies in exactly one interval).
    first_b = (t[0] // freq_ns + 1) * freq_ns
    last_b = (t[-1] // freq_ns) * freq_ns
    hb = np.arange(first_b, last_b + 1, freq_ns, dtype='int64') if last_b >= first_b else np.empty(0, 'int64')
    if hb.size:
        si = np.clip(np.searchsorted(t, hb, side='right') - 1, 0, t.size - 2)
        frac = (hb - t[si]) / seg_dur[si]
        v_hb = v[si] + (v[si + 1] - v[si]) * frac
    else:
        v_hb = np.empty(0, 'float64')

    cut_t = np.concatenate([t, hb])
    cut_v = np.concatenate([v, v_hb])
    order = np.argsort(cut_t, kind='stable')
    cut_t, cut_v = cut_t[order], cut_v[order]

    a_t, b_t = cut_t[:-1], cut_t[1:]
    a_v, b_v = cut_v[:-1], cut_v[1:]
    dur = (b_t - a_t).astype('float64')
    # which original segment each sub-segment sits in (via its midpoint) -> gap status
    mid = a_t + (b_t - a_t) // 2
    sub_seg = np.clip(np.searchsorted(t, mid, side='right') - 1, 0, t.size - 2)
    covered = (~seg_gap[sub_seg]) & (dur > 0)
    if not covered.any():
        return empty

    area = (a_v + b_v) * 0.5 * dur  # trapezoidal integral of the linear sub-segment
    hour = a_t // freq_ns  # sub-segment lies in one interval -> floor(left)

    hh = hour[covered]
    uh, inv = np.unique(hh, return_inverse=True)
    integ = np.bincount(inv, weights=area[covered])
    cov = np.bincount(inv, weights=dur[covered])
    val = integ / cov
    keep = (cov / freq_ns) >= min_coverage
    idx = pd.DatetimeIndex((uh[keep] * freq_ns).astype('datetime64[ns]'), name='time')
    return pd.Series(val[keep], index=idx)


def resample_sum(times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0) -> pd.Series:  # noqa: ARG001
    """Right-closed interval sum of an accumulation signal (min_count=1: empty interval -> NaN).

    A hole longer than ``max_gap`` never yields a value: its intervals simply have no reading,
    so ``min_count=1`` leaves them NaN. Output is dropna'd (missing intervals absent); a
    reported ``0.0`` is kept. ``max_gap``/``gap_multiplier`` are accepted for API symmetry
    (a hole simply produces empty intervals, which ``min_count=1`` already leaves NaN).
    """
    t, v = _prep(times, values)
    empty = pd.Series(dtype='float64')
    empty.index = pd.DatetimeIndex([], name='time')
    if t.size == 0:
        return empty
    s = pd.Series(v, index=pd.DatetimeIndex(t.astype('datetime64[ns]')))
    out = s.resample(pd.Timedelta(freq), closed='right', label='left').sum(min_count=1)
    out = out.dropna()
    out.index.name = 'time'
    return out


def resample_station(times, values, kind, **kwargs) -> pd.Series:
    """Dispatch on aggregation kind. ``kind`` in {'mean', 'sum'}."""
    if kind == 'mean':
        return resample_mean(times, values, **kwargs)
    if kind == 'sum':
        return resample_sum(times, values, **{k: v for k, v in kwargs.items() if k != 'min_coverage'})
    msg = f"kind must be 'mean' or 'sum', got {kind!r}"
    raise ValueError(msg)
