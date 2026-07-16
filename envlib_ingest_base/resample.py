"""Source-agnostic resampling of irregular / gappy station telemetry to a fixed cadence.

Two aggregation kinds, matching the two physical measurement types:

- ``'mean'`` (instantaneous signals: river stage, flow) -> an **exact trapezoidal
  time-weighted mean**. The signal is treated as piecewise-linear between consecutive
  observations; each output interval's value is the time-integral of that signal over the
  interval, divided by the covered time. This is exact for *any* spacing (regular or
  irregular) and reduces to the plain hourly mean for regularly-sampled data.

- ``'sum'`` (accumulations: rainfall) -> a right-closed interval sum. An interval with
  **no** reading is missing (absent from the output), never a fabricated ``0``, while a
  genuinely-reported ``0.0`` is kept.

Gaps/outages are first-class: consecutive observations farther apart than ``max_gap`` are a
hole. The signal is never interpolated across a hole, and output intervals whose covered
fraction is below ``min_coverage`` are dropped rather than ramped across the gap.

``max_gap`` may be given absolutely, or (default) adapted per series as
``gap_multiplier * median(native interval)`` so a 5-min site and a 60-min site each get a
threshold matched to their own cadence.

``round_to`` declares the source's TRUE timestamp precision (e.g. ``'1min'`` for a station
whose feed stamps 10:59:59 / 11:00:01 for what is really an 11:00:00 reading). Timestamps
are rounded to the nearest multiple *before* binning — jittered boundary readings bin into
the correct interval, and two jittery renderings of the same reading collide and merge via
the duplicate collapse instead of double-counting. ``None`` (default) = no rounding.

Input/output contract:

- Time is handled in **microseconds** (``datetime64[us]``) — python ``datetime``'s native
  resolution, with a ±290,000-year range so realistic corrupt dates convert visibly instead
  of overflowing. Sub-microsecond input precision is truncated.
- ``times``: anything convertible to ``datetime64`` — ISO-format strings, ``datetime64``
  arrays of any unit, python ``datetime`` objects, or pandas DatetimeIndex/Series (tz-aware
  input is converted to UTC-naive). ``None``/``NaT`` entries are dropped. Timezone handling
  is otherwise the caller's job (pass UTC).
- ``values``: numeric array-like; ``None``/``NaN`` entries are dropped, non-numeric raises.
- Returns are plain ``(times, values)`` tuples: ascending ``datetime64[us]`` interval-start
  labels (envlib convention) + ``float64`` values. **Empty means ``times.size == 0``** —
  never test the pair itself with ``len()``/truthiness; unpack immediately
  (``t, v = resample_mean(...)``).
- Both resamplers are **epoch-anchored** (labels are multiples of ``freq`` since
  1970-01-01), which differs from pandas' start-of-day anchoring only for frequencies that
  do not divide 24 h.
"""

from __future__ import annotations

import datetime
import re

import numpy as np

_MIN_MEAN_POINTS = 2  # need >= 2 observations to form one linear segment
_UNIT_US = {
    'h': 3_600_000_000,
    'min': 60_000_000,
    's': 1_000_000,
    'D': 86_400_000_000,
}
_FREQ_RE = re.compile(r'^(\d+)?(h|min|s|D)$')


def _freq_us(freq) -> int:
    """A fixed interval -> integer microseconds.

    Accepts ``np.timedelta64``, ``datetime.timedelta``, or a ``'<n><unit>'`` string with
    unit in {'h', 'min', 's', 'D'} (n defaults to 1, e.g. '1h', '15min', 'D'). Anything
    else raises ValueError — never silently mis-parses.
    """
    if isinstance(freq, np.timedelta64):
        return int(freq.astype('timedelta64[us]').astype('int64'))
    if isinstance(freq, datetime.timedelta):
        return freq // datetime.timedelta(microseconds=1)
    m = _FREQ_RE.match(freq) if isinstance(freq, str) else None
    if m is None:
        msg = f"unrecognized freq {freq!r}: expected '<n><unit>', unit in ('h', 'min', 's', 'D'), e.g. '1h', '15min'"
        raise ValueError(msg)
    return int(m.group(1) or 1) * _UNIT_US[m.group(2)]


def _to_dt64us(times) -> np.ndarray:
    """Convert to ``datetime64[us]`` (python datetime's native resolution).

    The ``[us]`` range is ±290k years, so out-of-range overflow is a non-issue for any
    realistic input (and beyond that numpy raises on the array cast). Sub-microsecond
    precision in ``datetime64`` input is truncated.
    """
    a = np.asarray(times)
    if a.dtype.kind == 'M':
        return a.astype('datetime64[us]')
    return np.asarray(times, dtype='datetime64[us]')


def _empty() -> tuple[np.ndarray, np.ndarray]:
    return np.empty(0, dtype='datetime64[us]'), np.empty(0, dtype='float64')


def _prep(times, values, round_to=None) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaT/NaN, round to the declared precision, sort, and collapse duplicate timestamps
    (by mean — rounding runs first, so jittery duplicates collide and merge). Returns int64-us t, float v."""
    dt = _to_dt64us(times)
    v = np.asarray(values, dtype='float64')
    ok = ~np.isnat(dt) & np.isfinite(v)
    t = dt[ok].astype('int64')
    v = v[ok]
    if t.size == 0:
        return t, v
    if round_to is not None:
        r = _freq_us(round_to)
        t = ((t + r // 2) // r) * r  # round half up; floor-division-correct pre-1970 too
    order = np.argsort(t, kind='stable')
    t, v = t[order], v[order]
    if t.size > 1 and np.any(np.diff(t) == 0):
        ut, inv = np.unique(t, return_inverse=True)
        v = np.bincount(inv, weights=v) / np.bincount(inv)
        t = ut
    return t, v


def _resolve_max_gap(t: np.ndarray, max_gap, gap_multiplier: float, freq_us: int) -> float:
    """Absolute max_gap in us if given, else adaptive gap_multiplier * median native interval."""
    if max_gap is not None:
        return float(_freq_us(max_gap))
    if t.size < _MIN_MEAN_POINTS:
        return float(freq_us)
    med = float(np.median(np.diff(t)))
    return max(med * gap_multiplier, float(freq_us))  # never below one output interval


def resample_mean(
    times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0, min_coverage=0.5, round_to=None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact trapezoidal time-weighted mean of an instantaneous signal onto a fixed cadence.

    Returns ``(times, values)``: interval-start ``datetime64[us]`` labels + ``float64``
    values, gap/under-covered intervals absent. Empty means ``times.size == 0``.
    ``round_to``: the source's true timestamp precision (see the module docstring).
    """
    freq_us = _freq_us(freq)
    t, v = _prep(times, values, round_to)
    if t.size < _MIN_MEAN_POINTS:
        return _empty()
    max_gap_us = _resolve_max_gap(t, max_gap, gap_multiplier, freq_us)

    seg_dur = np.diff(t)  # (n-1,)
    seg_gap = seg_dur > max_gap_us

    # interval boundaries strictly inside (t0, t_last); these become cut points so no
    # sub-segment ever straddles a boundary (=> each sub-segment lies in exactly one interval).
    first_b = (t[0] // freq_us + 1) * freq_us
    last_b = (t[-1] // freq_us) * freq_us
    hb = np.arange(first_b, last_b + 1, freq_us, dtype='int64') if last_b >= first_b else np.empty(0, 'int64')
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
        return _empty()

    area = (a_v + b_v) * 0.5 * dur  # trapezoidal integral of the linear sub-segment
    hour = a_t // freq_us  # sub-segment lies in one interval -> floor(left)

    hh = hour[covered]
    uh, inv = np.unique(hh, return_inverse=True)
    integ = np.bincount(inv, weights=area[covered])
    cov = np.bincount(inv, weights=dur[covered])
    val = integ / cov
    keep = (cov / freq_us) >= min_coverage
    return (uh[keep] * freq_us).astype('datetime64[us]'), val[keep]


def resample_sum(
    times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0, round_to=None  # noqa: ARG001
) -> tuple[np.ndarray, np.ndarray]:
    """Right-closed interval sum of an accumulation signal (interval-start labels).

    A reading is attributed to the interval it *closes* (a reading exactly on a boundary
    belongs to the interval ending there) — which is why ``round_to`` matters here: a
    boundary reading jittered to 11:00:01 would otherwise sum into the wrong hour. An
    interval with no reading is absent from the output — a hole longer than ``max_gap``
    simply produces empty intervals — while a reported ``0.0`` is kept.
    ``max_gap``/``gap_multiplier`` are accepted for API symmetry. Duplicate timestamps
    collapse **by mean** before summing (the web-service-retry failure mode: the same
    reading served twice must not double-count); with ``round_to``, jittery near-duplicates
    collide onto the same timestamp first and merge the same way. Bins are epoch-anchored
    (see the module docstring).
    """
    freq_us = _freq_us(freq)
    t, v = _prep(times, values, round_to)
    if t.size == 0:
        return _empty()
    lab = ((t - 1) // freq_us) * freq_us  # right-closed, left-labeled
    ul, inv = np.unique(lab, return_inverse=True)
    return ul.astype('datetime64[us]'), np.bincount(inv, weights=v)


def resample_station(times, values, kind, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch on aggregation kind. ``kind`` in {'mean', 'sum'}."""
    if kind == 'mean':
        return resample_mean(times, values, **kwargs)
    if kind == 'sum':
        return resample_sum(times, values, **{k: v for k, v in kwargs.items() if k != 'min_coverage'})
    msg = f"kind must be 'mean' or 'sum', got {kind!r}"
    raise ValueError(msg)
