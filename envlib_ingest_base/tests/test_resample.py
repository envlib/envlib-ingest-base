"""Golden tests for the resampler: frozen hand-computed expected values.

These assert exactness (not a live cross-call to tethys, which can't run in this env).
The key correctness properties: exact trapezoidal time-weighting for irregular spacing, no
interpolation/contamination across gaps, and the numpy input-conversion guardrails (no
silent wrapping, documented None/nan asymmetries, loud freq parsing).
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from envlib_ingest_base.resample import _freq_us, resample


def dts(*strs):
    return np.array(strs, dtype='datetime64[us]')


def _rs(*args, **kwargs):  # sum-statistic shorthand for the accumulation fixtures
    return resample(*args, statistic='sum', **kwargs)


def check(result, times, values):
    t, v = result
    assert np.array_equal(t, dts(*times))
    np.testing.assert_allclose(v, values, rtol=0, atol=1e-9)


# --- golden values (identical to the pandas-based implementation) ---


def test_mean_regular_15min_is_exact_hourly_mean():
    # linear ramp 0,1,2,...,8 at 15-min spacing over [00:00, 02:00]
    times = dts(
        '2020-01-01 00:00',
        '2020-01-01 00:15',
        '2020-01-01 00:30',
        '2020-01-01 00:45',
        '2020-01-01 01:00',
        '2020-01-01 01:15',
        '2020-01-01 01:30',
        '2020-01-01 01:45',
        '2020-01-01 02:00',
    )
    check(resample(times, np.arange(9.0)), ('2020-01-01 00:00', '2020-01-01 01:00'), [2.0, 6.0])


def test_mean_irregular_is_time_weighted_not_equal_weight():
    # 10@:00, 20@:15, 100@:00(next). True time-weighted = 48.75; equal-weight (tethys) = 37.5.
    t, v = resample(dts('2020-01-01 00:00', '2020-01-01 00:15', '2020-01-01 01:00'), [10.0, 20.0, 100.0])
    check((t, v), ('2020-01-01 00:00',), [48.75])
    # explicitly NOT the equal-weight-of-segment-midpoints answer
    assert v[0] != 37.5


def test_mean_gap_drops_hours_without_contamination():
    # hourly 0s, then a 10h outage, resume at 120. Hour 02:00 must be absent (not ramped to 6.0).
    times = dts('2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00', '2020-01-01 12:00', '2020-01-01 13:00')
    t, v = resample(times, [0.0, 0.0, 0.0, 120.0, 120.0])
    check((t, v), ('2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 12:00'), [0.0, 0.0, 120.0])
    # the contaminated hour tethys would emit is gone
    assert dts('2020-01-01 02:00')[0] not in t


def test_mean_60min_cadence():
    t, v = resample(dts('2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00'), [10.0, 20.0, 30.0])
    check((t, v), ('2020-01-01 00:00', '2020-01-01 01:00'), [15.0, 25.0])


def test_mean_duplicate_timestamps_collapse_by_mean():
    times = dts('2020-01-01 00:00', '2020-01-01 00:00', '2020-01-01 01:00')
    vals = [10.0, 20.0, 40.0]  # dup at 00:00 -> mean 15; then [15@:00, 40@1:00] -> (15+40)/2
    check(resample(times, vals), ('2020-01-01 00:00',), [27.5])


def test_mean_single_reading_is_empty():
    t, v = resample(dts('2020-01-01 00:30'), [5.0])
    assert t.size == 0
    assert v.size == 0
    assert t.dtype == np.dtype('datetime64[us]')


def test_mean_partial_first_hour_dropped_by_min_coverage():
    # first hour covered only 00:40->01:00 (20 min < 50%): dropped; next hour full.
    times = dts(
        '2020-01-01 00:40',
        '2020-01-01 00:50',
        '2020-01-01 01:00',
        '2020-01-01 01:20',
        '2020-01-01 01:40',
        '2020-01-01 02:00',
    )
    t, _v = resample(times, [1.0] * 6)
    assert dts('2020-01-01 00:00')[0] not in t
    assert dts('2020-01-01 01:00')[0] in t


def test_sum_outage_is_nan_reported_zero_kept():
    # hourly accumulations timestamped at hour-end; 00-01=0, 01-02=0.5, 02-03=0, gap 03-05, 06-07=1.0
    times = dts('2020-01-01 01:00', '2020-01-01 02:00', '2020-01-01 03:00', '2020-01-01 07:00')
    t, v = _rs(times, [0.0, 0.5, 0.0, 1.0])
    check(
        (t, v),
        ('2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00', '2020-01-01 06:00'),
        [0.0, 0.5, 0.0, 1.0],
    )
    # outage hours never fabricated as 0
    for h in ('03:00', '04:00', '05:00'):
        assert dts(f'2020-01-01 {h}')[0] not in t


def test_sum_messy_timestamps_bin_to_interval_start():
    # a HH:59:59 reading and a HH:00:00 reading both land on the correct interval-start
    times = dts('2020-01-01 15:59:59', '2020-01-01 18:00:00')
    check(_rs(times, [0.5, 0.3]), ('2020-01-01 15:00', '2020-01-01 17:00'), [0.5, 0.3])


# --- numpy-rewrite guardrails (review fixtures) ---


def test_sum_right_closed_at_us_resolution():
    # a reading exactly ON a boundary closes the earlier interval; +1us opens the next
    times = dts('2020-01-01 01:00:00.000000', '2020-01-01 01:00:00.000001')
    t, v = resample(times, [1.0, 2.0], statistic='sum', min_coverage=0)
    check((t, v), ('2020-01-01 00:00', '2020-01-01 01:00'), [1.0, 2.0])


def test_mean_pre_1970():
    # negative-us floor division must floor, not truncate toward zero
    times = dts('1969-12-31 22:00', '1969-12-31 23:00', '1970-01-01 00:00')
    check(resample(times, [0.0, 10.0, 20.0]), ('1969-12-31 22:00', '1969-12-31 23:00'), [5.0, 15.0])


def test_sum_pre_1970():
    times = dts('1969-12-31 23:00', '1969-12-31 23:30')
    check(_rs(times, [1.0, 2.0], min_coverage=0), ('1969-12-31 22:00', '1969-12-31 23:00'), [1.0, 2.0])


def test_tz_aware_pandas_input_converts_to_utc():
    # the duck-typing promise: a tz-aware DatetimeIndex converts to UTC-naive (12:00 NZST -> 00:00 UTC)
    idx = pd.to_datetime(['2020-01-01 12:00', '2020-01-01 12:15', '2020-01-01 13:00']).tz_localize('Etc/GMT-12')
    check(resample(idx, [10.0, 20.0, 100.0]), ('2020-01-01 00:00',), [48.75])


def test_coarser_datetime64_units_upconvert():
    times_s = np.array(['2020-01-01T00:00', '2020-01-01T01:00', '2020-01-01T02:00'], dtype='datetime64[s]')
    check(resample(times_s, [10.0, 20.0, 30.0]), ('2020-01-01 00:00', '2020-01-01 01:00'), [15.0, 25.0])
    times_us = times_s.astype('datetime64[us]')
    check(resample(times_us, [10.0, 20.0, 30.0]), ('2020-01-01 00:00', '2020-01-01 01:00'), [15.0, 25.0])


def test_far_future_dates_convert_not_wrap():
    # at [ns] a year-3000 string silently wrapped to 1830; at [us] it converts correctly,
    # so a corrupt far-future date stays VISIBLY absurd instead of plausibly historical
    t, v = _rs(['3000-01-01', '3000-01-02'], [1.0, 2.0])
    assert np.array_equal(t, dts('2999-12-31 23:00', '3000-01-01 23:00'))
    np.testing.assert_allclose(v, [1.0, 2.0])


def test_times_none_dropped_but_nan_raises():
    # documented asymmetry: None -> NaT -> dropped; a float nan in a times list raises
    check(_rs(['2020-01-01 00:30', None], [1.0, 2.0], min_coverage=0), ('2020-01-01 00:00',), [1.0])
    with pytest.raises((ValueError, TypeError)):
        _rs(['2020-01-01 00:30', np.nan], [1.0, 2.0])


def test_values_none_dropped_but_string_raises():
    t, v = _rs(dts('2020-01-01 00:30', '2020-01-01 01:30'), [1.0, None], min_coverage=0)
    check((t, v), ('2020-01-01 00:00',), [1.0])
    with pytest.raises(ValueError):
        _rs(dts('2020-01-01 00:30'), ['abc'])


def test_round_to_corrects_boundary_jitter():
    # feed stamps 02:00:01 for the true 02:00 reading: unrounded it sums into the WRONG hour
    times = dts('2020-01-01 00:59:59', '2020-01-01 02:00:01')
    t, _v = _rs(times, [1.0, 2.0])
    assert dts('2020-01-01 02:00')[0] in t  # the mis-binned hour, without rounding
    check(
        _rs(times, [1.0, 2.0], round_to='1min'),
        ('2020-01-01 00:00', '2020-01-01 01:00'),
        [1.0, 2.0],
    )


def test_round_to_merges_jittery_duplicates():
    # the same reading served twice as 00:59:59 and 01:00:01: rounding makes them collide,
    # then the duplicate collapse merges them (mean) instead of double-counting into two bins
    times = dts('2020-01-01 00:59:59', '2020-01-01 01:00:01')
    check(_rs(times, [1.0, 3.0], round_to='1min', min_coverage=0), ('2020-01-01 00:00',), [2.0])
    t, _v = _rs(times, [1.0, 3.0], min_coverage=0)  # unrounded: two bins, double-counted
    assert t.size == 2


def test_round_to_restores_exact_mean_on_jittered_grid():
    times = dts('2020-01-01 00:00:01', '2020-01-01 00:59:59', '2020-01-01 02:00:01')
    check(
        resample(times, [10.0, 20.0, 30.0], round_to='1min'),
        ('2020-01-01 00:00', '2020-01-01 01:00'),
        [15.0, 25.0],
    )


@pytest.mark.parametrize(
    ('freq', 'us'),
    [
        ('1h', 3_600_000_000),
        ('15min', 900_000_000),
        ('h', 3_600_000_000),
        ('90s', 90_000_000),
        ('2D', 172_800_000_000),
        ('day', 86_400_000_000),
        (np.timedelta64(1, 'h'), 3_600_000_000),
        (datetime.timedelta(hours=1), 3_600_000_000),
    ],
)
def test_freq_parser_accepts(freq, us):
    assert _freq_us(freq) == us


@pytest.mark.parametrize('freq', ['1.5h', '1H', 'T', 'W', '2 days', '2day', '1', '', None])
def test_freq_parser_rejects_loudly(freq):
    with pytest.raises(ValueError, match='unrecognized freq'):
        _freq_us(freq)


def test_sum_5h_bins_are_epoch_anchored():
    # epoch-anchored 5h bins (differs from pandas' start-of-day origin — deliberate, matches mean):
    # 2020-03-04 20:00 is a multiple of 5h since epoch; 01:00 exactly closes that bin.
    times = dts('2020-03-05 01:00', '2020-03-05 02:00')
    t, v = _rs(times, [1.0, 2.0], freq='5h', min_coverage=0)
    check((t, v), ('2020-03-04 20:00', '2020-03-05 01:00'), [1.0, 2.0])


def test_empty_and_singleton_shapes():
    for t, v in (resample([], []), _rs([], [])):
        assert t.size == 0
        assert v.size == 0
        assert t.dtype == np.dtype('datetime64[us]')
        assert v.dtype == np.dtype('float64')
    # a lone sum reading has an unknowable slot: zero coverage -> absent at the default
    # threshold, kept only when the coverage rule is disabled (mean needs >= 2 points always)
    t, v = _rs(dts('2020-01-01 00:30'), [2.5])
    assert t.size == 0
    check(_rs(dts('2020-01-01 00:30'), [2.5], min_coverage=0), ('2020-01-01 00:00',), [2.5])


# --- Phase-7 fixtures: the local gap rule + the coverage rule (rulings 2026-07-17/18) ---


def test_mean_local_median_survives_regime_change():
    # 30-min logging era followed by a 5-min era (a real station history). A GLOBAL median
    # would classify every normal 30-min step of the early era as a gap and erase the era;
    # the LOCAL median judges each era by its own cadence.
    era_a = np.datetime64('2020-01-01T00:00', 'us') + np.timedelta64(30, 'm') * np.arange(13)  # 00:00..06:00
    era_b = np.datetime64('2020-01-01T06:05', 'us') + np.timedelta64(5, 'm') * np.arange(24)  # 06:05..08:00
    times = np.concatenate([era_a, era_b])
    t, v = resample(times, np.full(times.size, 10.0))
    assert np.array_equal(t, dts(*(f'2020-01-01 0{h}:00' for h in range(8))))
    np.testing.assert_allclose(v, 10.0)


def test_mean_short_outage_not_bridged_below_freq():
    # 5-min site with a 30-min outage inside one hour (the review's fabricated-ramp case).
    # The gap rule runs on the DATA's clock (no freq floor): the outage is a hole, never
    # interpolated — the hour is judged on its genuinely covered halves, not a bridged ramp.
    times = dts(*(f'2020-01-01 00:{m:02d}' for m in (0, 5, 10, 15, 45, 50, 55)), '2020-01-01 01:00')
    vals = [0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0]
    t, _v = resample(times, vals)  # default 0.75: the half-covered hour is absent
    assert t.size == 0
    t, v = resample(times, vals, min_coverage=0.4)
    check((t, v), ('2020-01-01 00:00',), [50.0])  # covered halves only: (0*15 + 100*15)/30


def test_sum_partial_coverage_is_absent_not_undercount():
    # 5-min rain: an hour holding only 6 of its 12 slots must NOT publish a half-hour
    # total as an hourly total; a fully-slotted hour is kept; a lone post-outage slot
    # covers one native tick and is nowhere near an hourly total.
    full = [f'2020-01-01 01:{m:02d}' for m in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)]
    half = [f'2020-01-01 02:{m:02d}' for m in (5, 10, 15, 20, 25, 30)]
    times = dts(*full, '2020-01-01 02:00', *half, '2020-01-01 06:00')
    t, v = _rs(times, [0.1] * 19)
    assert dts('2020-01-01 01:00')[0] in t  # 12/12 slots -> kept
    np.testing.assert_allclose(v[t == dts('2020-01-01 01:00')[0]], [1.2])
    assert dts('2020-01-01 02:00')[0] not in t  # 6/12 slots -> absent, not 0.6
    assert dts('2020-01-01 05:00')[0] not in t  # lone post-outage slot -> absent


@pytest.mark.parametrize(
    'freq',
    [
        '0min',
        '0h',
        '0s',
        np.timedelta64(0, 'h'),
        np.timedelta64(-1, 'h'),
        np.timedelta64('NaT'),
        np.timedelta64(1, 'M'),
        datetime.timedelta(0),
    ],
)
def test_freq_parser_rejects_degenerate_durations(freq):
    # a '0min' config typo must never silently collapse a series onto the epoch
    with pytest.raises(ValueError, match='freq'):
        _freq_us(freq)


def test_sum_skipped_slots_never_credited_as_covered():
    # per-slot totals: a reading after skipped slots covers only its own slot. An hour with
    # 8 of 12 five-minute slots (gaps below the hole threshold!) is 40/60 covered -> absent;
    # 9 of 12 is exactly 0.75 -> kept.
    hour0 = [f'2020-01-01 00:{m:02d}' for m in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)]
    eight = [f'2020-01-01 01:{m:02d}' for m in (5, 10, 20, 25, 35, 40, 50, 55)]
    times = dts(*hour0, '2020-01-01 01:00', *eight)
    t, _v = _rs(times, [0.1] * 20)
    assert dts('2020-01-01 00:00')[0] in t  # full hour kept
    assert dts('2020-01-01 01:00')[0] not in t  # 8/12 slots -> absent, not published as a total
    nine = [f'2020-01-01 01:{m:02d}' for m in (5, 10, 20, 25, 35, 40, 45, 50, 55)]
    times9 = dts(*hour0, '2020-01-01 01:00', *nine)
    t9, v9 = _rs(times9, [0.1] * 21)
    assert dts('2020-01-01 01:00')[0] in t9  # 9/12 = 0.75 exactly -> kept (inclusive)
    np.testing.assert_allclose(v9[t9 == dts('2020-01-01 01:00')[0]], [0.9])


def test_mean_local_median_short_edge_era():
    # a SHORT era (< window/2 segments) at the series START must be judged by its own
    # cadence: 7 x 30-min segments then a long 5-min era. A fixed nearest-full-window edge
    # median would be swamped by the 5-min majority and erase the early era wholesale.
    era_a = np.datetime64('2020-01-01T00:00', 'us') + np.timedelta64(30, 'm') * np.arange(8)  # 00:00..03:30
    era_b = np.datetime64('2020-01-01T03:35', 'us') + np.timedelta64(5, 'm') * np.arange(31)  # 03:35..06:05
    times = np.concatenate([era_a, era_b])
    t, v = resample(times, np.full(times.size, 10.0))
    for h in range(6):
        assert dts(f'2020-01-01 0{h}:00')[0] in t  # early 30-min hours survive
    np.testing.assert_allclose(v, 10.0)


def test_statistic_dispatch():
    with pytest.raises(ValueError, match='unsupported statistic'):
        resample(dts('2020-01-01 00:30'), [1.0], statistic='median')
    with pytest.raises(ValueError, match='unsupported statistic'):
        resample(dts('2020-01-01 00:30'), [1.0], statistic=None)
