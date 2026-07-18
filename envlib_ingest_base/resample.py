"""Source-agnostic resampling of irregular / gappy station telemetry to a fixed cadence.

Two aggregation kinds, matching the two physical measurement types:

- ``'mean'`` (instantaneous signals: river stage, flow) -> an **exact trapezoidal
  time-weighted mean**. The signal is treated as piecewise-linear between consecutive
  observations; each output interval's value is the time-integral of that signal over the
  covered portion of the interval, divided by the covered time. Exact for *any* spacing;
  for regularly-sampled grid-aligned data it equals the trapezoid-rule mean (interval
  endpoints carry half-weight — this coincides with the plain arithmetic mean for linear
  ramps, but is NOT the same as pandas' left-closed ``.resample().mean()``).

- ``'sum'`` (accumulations: rainfall) -> a right-closed interval sum. An interval with
  **no** reading is missing (absent from the output), never a fabricated ``0``, while a
  genuinely-reported ``0.0`` is kept. **Values must be per-slot totals** (each reading is
  the accumulation over its own reporting slot) — NOT since-last-report accumulations from
  a totalizing gauge; such feeds must be differenced upstream. Resampling accumulations to
  a cadence FINER than the native reporting interval is not meaningful.

Missing data is handled by a two-rule pipeline — the two rules are anchored to two
different clocks and are NOT derivable from each other:

1. **The gap rule (``max_gap``) — the data's clock.** The estimator for the mean spans
   beyond the aggregation interval (the piecewise-linear model reaches from reading to
   reading regardless of interval boundaries) and must be told where to stop: consecutive
   observations farther apart than ``max_gap`` are a *hole* the signal is never
   interpolated across. By default the threshold adapts per segment as
   ``gap_multiplier x`` the **local median native interval** (a rolling window of
   ``_GAP_WINDOW`` segments), so a station that logged 30-min data twenty years ago and
   5-min data today gets the right threshold in each era (a global median would misjudge
   one era wholesale). ``max_gap`` may instead be given absolutely (same forms as
   ``freq``). Interpolation honesty is a property of the source's sampling design — never
   of the output frequency, which is why there is deliberately no coupling to ``freq``.

2. **The coverage rule (``min_coverage``) — the output's clock.** Once holes are excluded,
   each output interval is judged by how much of it is genuinely covered: below
   ``min_coverage`` (default **0.75**, in line with common climatological completeness
   practice) the interval is absent rather than summarized from too little data. For the
   mean, coverage is the non-hole time fraction; for the sum, each reading covers its
   accumulation slot (the span back to the previous reading, capped at **one local native
   interval** — a reading after skipped slots covers only its own slot, so losses are never
   credited as measured), and an under-covered interval is absent rather than published as
   a silent undercount. A single-reading series has an unknowable slot and contributes no
   coverage (kept only when ``min_coverage=0``). Missingness in telemetry is not random —
   outages correlate with extreme events — so the default is deliberately conservative. Set
   ``min_coverage=0`` to disable. The threshold comparison is float-based; prefer
   binary-friendly fractions (0.75, 0.5) for exact behavior at the boundary.

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
  (``t, v = resample(...)``).
- Both resamplers are **epoch-anchored** (labels are multiples of ``freq`` since
  1970-01-01), which differs from pandas' start-of-day anchoring only for frequencies that
  do not divide 24 h.
"""

from __future__ import annotations

import datetime
import re

import numpy as np

_MIN_MEAN_POINTS = 2  # need >= 2 observations to form one linear segment
_GAP_WINDOW = 15  # segments in the rolling local-median window of the adaptive gap rule
_UNIT_US = {
    'h': 3_600_000_000,
    'min': 60_000_000,
    's': 1_000_000,
    'D': 86_400_000_000,
}
_FREQ_RE = re.compile(r'^([1-9]\d*)?(h|min|s|D)$')


def _freq_us(freq) -> int:
    """A fixed positive interval -> integer microseconds.

    Accepts ``np.timedelta64`` (fixed units only — calendar M/Y raise), ``datetime.timedelta``,
    or a ``'<n><unit>'`` string with unit in {'h', 'min', 's', 'D'} (n defaults to 1, e.g.
    '1h', '15min', 'D'), plus the envlib CV code ``'day'`` (= '1D'). Anything else — including
    zero, negative, and NaT durations — raises ValueError; never silently mis-parses.
    """
    if isinstance(freq, np.timedelta64):
        if np.datetime_data(freq.dtype)[0] in ('M', 'Y'):
            msg = f'unrecognized freq {freq!r}: calendar units are not fixed durations'
            raise ValueError(msg)
        if np.isnat(freq):
            msg = f'unrecognized freq {freq!r}: NaT is not a duration'
            raise ValueError(msg)
        us = int(freq.astype('timedelta64[us]').astype('int64'))
    elif isinstance(freq, datetime.timedelta):
        us = freq // datetime.timedelta(microseconds=1)
    elif freq == 'day':  # the envlib frequency_interval CV code for daily
        us = _UNIT_US['D']
    else:
        m = _FREQ_RE.match(freq) if isinstance(freq, str) else None
        if m is None:
            msg = f"unrecognized freq {freq!r}: expected '<n><unit>', unit in ('h', 'min', 's', 'D'), e.g. '15min'"
            raise ValueError(msg)
        us = int(m.group(1) or 1) * _UNIT_US[m.group(2)]
    if us <= 0:
        msg = f'freq must be a positive duration, got {freq!r}'
        raise ValueError(msg)
    return us


def _to_dt64us(times) -> np.ndarray:
    """Convert to ``datetime64[us]`` (python datetime's native resolution).

    The ``[us]`` range is ±290k years, so overflow is a non-issue for realistic input;
    ``datetime64`` array casts raise OverflowError beyond it (ISO strings with absurd
    5+-digit years beyond that range are numpy-parsed and may still wrap — practical
    inputs are unaffected). Sub-microsecond precision in ``datetime64`` input is truncated.
    """
    a = np.asarray(times)
    if a.dtype.kind == 'M':
        return a.astype('datetime64[us]')
    return np.asarray(times, dtype='datetime64[us]')


def _empty() -> tuple[np.ndarray, np.ndarray]:
    return np.empty(0, dtype='datetime64[us]'), np.empty(0, dtype='float64')


def _prep(times, values, round_to=None) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaT/NaN/±inf, round to the declared precision, sort, and collapse duplicate timestamps
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


def _local_medians(diffs: np.ndarray) -> np.ndarray:
    """Per-segment LOCAL median native interval (rolling window, edge-padded).

    The local (not global) median makes the adaptive gap rule survive within-station
    regime changes — a station that logged 30-min data for years and 5-min data since
    is judged by each era's own cadence. Falls back to the global median for series
    shorter than the window.
    """
    if diffs.size < _GAP_WINDOW:
        return np.full(diffs.shape, float(np.median(diffs))) if diffs.size else diffs.astype('float64')
    # edges use CENTERED SHRINKING windows (clipped to the real data). The two tempting
    # alternatives both fail a real case: edge-value padding replicates a leading/trailing
    # outage into its own window; a fixed nearest-full-window median swamps a short edge era
    # (< window/2 segments) with the neighboring regime. A centered clipped window keeps the
    # edge segment's own side in the majority in both cases.
    pad = _GAP_WINDOW // 2
    med = np.empty(diffs.shape, dtype='float64')
    interior = np.median(np.lib.stride_tricks.sliding_window_view(diffs, _GAP_WINDOW), axis=1)
    med[pad : pad + interior.size] = interior
    n = diffs.size
    for i in range(pad):
        med[i] = np.median(diffs[: i + pad + 1])
        med[n - 1 - i] = np.median(diffs[n - 1 - i - pad :])
    return med


def _gap_thresholds(diffs: np.ndarray, max_gap, gap_multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment (local median native interval, gap threshold) in us — no freq coupling."""
    med = _local_medians(diffs)
    if max_gap is not None:
        return med, np.full(diffs.shape, float(_freq_us(max_gap)))
    return med, med * gap_multiplier


def _resample_mean(
    times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0, min_coverage=0.75, round_to=None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact trapezoidal time-weighted mean of an instantaneous signal onto a fixed cadence.

    Returns ``(times, values)``: interval-start ``datetime64[us]`` labels + ``float64``
    values; holes are never interpolated across, and intervals covered below
    ``min_coverage`` are absent (see the module docstring for the two-rule pipeline).
    Empty means ``times.size == 0``. ``round_to``: the source's true timestamp precision.
    """
    freq_us = _freq_us(freq)
    t, v = _prep(times, values, round_to)
    if t.size < _MIN_MEAN_POINTS:
        return _empty()

    seg_dur = np.diff(t)  # (n-1,)
    _med, thr = _gap_thresholds(seg_dur, max_gap, gap_multiplier)
    seg_gap = seg_dur > thr

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
    keep = cov >= min_coverage * freq_us
    return (uh[keep] * freq_us).astype('datetime64[us]'), val[keep]


def _resample_sum(
    times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0, min_coverage=0.75, round_to=None
) -> tuple[np.ndarray, np.ndarray]:
    """Right-closed interval sum of an accumulation signal (interval-start labels).

    A reading is attributed to the interval it *closes* (a reading exactly on a boundary
    belongs to the interval ending there) — which is why ``round_to`` matters here: a
    boundary reading jittered to 11:00:01 would otherwise sum into the wrong hour.

    Coverage: each reading covers its accumulation slot — the span back to the previous
    reading, capped at ONE local native interval (values are per-slot totals: a reading
    after skipped slots covers only its own slot, so losses are never credited as measured;
    see the module docstring). An interval covered below ``min_coverage`` is absent rather
    than published as a silent undercount (an hour holding 8 of its 12 five-minute slots is
    not an hourly total). A single-reading series has an unknowable slot and contributes no
    coverage (kept only when ``min_coverage=0``). An interval with no reading at all is
    absent — never a fabricated ``0`` — while a reported ``0.0`` counts as full coverage
    of its slot and is kept.

    Duplicate timestamps collapse **by mean** before summing (the web-service-retry
    failure mode: the same reading served twice must not double-count); with ``round_to``,
    jittery near-duplicates collide onto the same timestamp first and merge the same way.
    Bins are epoch-anchored (see the module docstring).
    """
    freq_us = _freq_us(freq)
    t, v = _prep(times, values, round_to)
    if t.size == 0:
        return _empty()
    lab = ((t - 1) // freq_us) * freq_us  # right-closed, left-labeled

    if t.size == 1:
        # a lone reading's slot is unknowable: it contributes NO coverage (kept only when
        # min_coverage == 0) — assuming a full interval would fabricate completeness while
        # a two-reading series gets honestly judged
        spans = np.array([0.0])
    else:
        diffs = np.diff(t)
        med, _thr = _gap_thresholds(diffs, max_gap, gap_multiplier)
        d = diffs.astype('float64')
        # each reading covers AT MOST one local native slot: crediting the full gap back
        # would count skipped slots as measured (an hour missing every third rain slot
        # would publish as complete). min(d, med) also caps post-hole readings for free.
        spans = np.concatenate([[float(med[0])], np.minimum(d, med)])

    ul, inv = np.unique(lab, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cov = np.minimum(np.bincount(inv, weights=spans), float(freq_us))
    keep = cov >= min_coverage * freq_us
    return ul[keep].astype('datetime64[us]'), sums[keep]


_STATISTICS = {'mean': _resample_mean, 'sum': _resample_sum}


def resample(
    times, values, *, statistic='mean', freq='1h', max_gap=None, gap_multiplier=3.0, min_coverage=0.75, round_to=None
) -> tuple[np.ndarray, np.ndarray]:
    """Resample one station's raw telemetry to a fixed cadence.

    ``statistic`` selects both the aggregation and the signal model behind it — pass the
    dataset's envlib ``aggregation_statistic`` metadata value, so the declared identity and
    the computation can never drift:

    - ``'mean'`` — instantaneous signals (river stage, flow): the exact trapezoidal
      time-weighted mean of the piecewise-linear signal.
    - ``'sum'`` — accumulations (rainfall): the right-closed per-slot-total sum.

    Future statistics of the instantaneous signal (median/min/max share its whole segment
    machinery, differing only in the final reduction) will slot in here without an API
    change. The two-rule missing-data pipeline (module docstring) applies to every
    statistic. Returns the standard ``(times, values)`` tuple.
    """
    try:
        fn = _STATISTICS[statistic]
    except (KeyError, TypeError) as e:
        msg = f'unsupported statistic {statistic!r}: supported {tuple(_STATISTICS)}'
        raise ValueError(msg) from e
    return fn(
        times,
        values,
        freq=freq,
        max_gap=max_gap,
        gap_multiplier=gap_multiplier,
        min_coverage=min_coverage,
        round_to=round_to,
    )
