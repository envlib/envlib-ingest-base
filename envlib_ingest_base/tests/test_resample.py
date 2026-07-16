"""Golden tests for the resampler: frozen hand-computed expected values.

These assert exactness (not a live cross-call to tethys, which can't run under pandas 3.x).
The key correctness properties: exact trapezoidal time-weighting for irregular spacing, and
no interpolation/contamination across gaps.
"""

import numpy as np
import pandas as pd

from envlib_ingest_base.resample import resample_mean, resample_sum


def dts(*strs):
    return pd.to_datetime(list(strs))


def as_dict(s):
    return {pd.Timestamp(k): round(float(v), 6) for k, v in s.items()}


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
    vals = np.arange(9.0)
    out = as_dict(resample_mean(times, vals))
    assert out == {pd.Timestamp('2020-01-01 00:00'): 2.0, pd.Timestamp('2020-01-01 01:00'): 6.0}


def test_mean_irregular_is_time_weighted_not_equal_weight():
    # 10@:00, 20@:15, 100@:00(next). True time-weighted = 48.75; equal-weight (tethys) = 37.5.
    times = dts('2020-01-01 00:00', '2020-01-01 00:15', '2020-01-01 01:00')
    vals = [10.0, 20.0, 100.0]
    out = as_dict(resample_mean(times, vals))
    assert out == {pd.Timestamp('2020-01-01 00:00'): 48.75}
    # explicitly NOT the equal-weight-of-segment-midpoints answer
    assert out[pd.Timestamp('2020-01-01 00:00')] != 37.5


def test_mean_gap_drops_hours_without_contamination():
    # hourly 0s, then a 10h outage, resume at 120. Hour 02:00 must be absent (not ramped to 6.0).
    times = dts('2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00', '2020-01-01 12:00', '2020-01-01 13:00')
    vals = [0.0, 0.0, 0.0, 120.0, 120.0]
    out = as_dict(resample_mean(times, vals))
    assert out == {
        pd.Timestamp('2020-01-01 00:00'): 0.0,
        pd.Timestamp('2020-01-01 01:00'): 0.0,
        pd.Timestamp('2020-01-01 12:00'): 120.0,
    }
    # the contaminated hour tethys would emit is gone
    assert pd.Timestamp('2020-01-01 02:00') not in out


def test_mean_60min_cadence():
    times = dts('2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00')
    vals = [10.0, 20.0, 30.0]
    out = as_dict(resample_mean(times, vals))
    assert out == {pd.Timestamp('2020-01-01 00:00'): 15.0, pd.Timestamp('2020-01-01 01:00'): 25.0}


def test_mean_duplicate_timestamps_collapse_by_mean():
    times = dts('2020-01-01 00:00', '2020-01-01 00:00', '2020-01-01 01:00')
    vals = [10.0, 20.0, 40.0]  # dup at 00:00 -> mean 15; then [15@:00, 40@1:00] -> (15+40)/2
    out = as_dict(resample_mean(times, vals))
    assert out == {pd.Timestamp('2020-01-01 00:00'): 27.5}


def test_mean_single_reading_is_empty():
    out = resample_mean(dts('2020-01-01 00:30'), [5.0])
    assert len(out) == 0


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
    vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    out = resample_mean(times, vals)
    assert pd.Timestamp('2020-01-01 00:00') not in out.index
    assert pd.Timestamp('2020-01-01 01:00') in out.index


def test_sum_outage_is_nan_reported_zero_kept():
    # hourly accumulations timestamped at hour-end; 00-01=0, 01-02=0.5, 02-03=0, gap 03-05, 06-07=1.0
    times = dts('2020-01-01 01:00', '2020-01-01 02:00', '2020-01-01 03:00', '2020-01-01 07:00')
    vals = [0.0, 0.5, 0.0, 1.0]
    out = as_dict(resample_sum(times, vals))
    assert out == {
        pd.Timestamp('2020-01-01 00:00'): 0.0,
        pd.Timestamp('2020-01-01 01:00'): 0.5,
        pd.Timestamp('2020-01-01 02:00'): 0.0,
        pd.Timestamp('2020-01-01 06:00'): 1.0,
    }
    # outage hours never fabricated as 0
    for h in ('03:00', '04:00', '05:00'):
        assert pd.Timestamp(f'2020-01-01 {h}') not in out


def test_sum_messy_timestamps_bin_to_interval_start():
    # a HH:59:59 reading and a HH:00:00 reading both land on the correct interval-start
    times = dts('2020-01-01 15:59:59', '2020-01-01 18:00:00')
    vals = [0.5, 0.3]
    out = as_dict(resample_sum(times, vals))
    assert out == {pd.Timestamp('2020-01-01 15:00'): 0.5, pd.Timestamp('2020-01-01 17:00'): 0.3}
