"""cfdb-level tests for ts_ortho build + idempotent merge (no remote needed).

The axis cadence comes from ``meta.frequency_interval`` (the envlib CV) — the matrix here pins
the natural-unit coord dtypes, the explicit step, the two phase-contract guards, the merge
cross-checks, and legacy-[us]-axis compatibility.
"""

import hashlib
import logging
import tracemalloc

import booklet
import cfdb as cfdb_module
import envlib
import numpy as np
import pytest
import shapely
from cfdb import dtypes, open_dataset
from envlib import Catalogue

from envlib_ingest_base.tsortho import (
    STATION_ALTITUDE_VAR,
    _altitudes,
    _points_ids_names,
    build_local,
    merge_dataset,
)

BASE = np.datetime64('2020-01-01T00:00', 'us')
HOUR = np.timedelta64(1, 'h')
DAY = np.timedelta64(1, 'D')
Q15 = np.timedelta64(15, 'm')
H12 = np.timedelta64(12, 'h')

ENC = {'variable': 'streamflow', 'units': 'm^3/s', 'precision': 4, 'min_value': 0, 'max_value': 100000}


def make_meta(frequency_interval='1h', utc_offset='+00:00'):
    return envlib.Metadata(
        feature='waterway',
        variable='streamflow',
        method='sensor_recording',
        product_code=None,
        processing_level='raw',
        owner='ecan',
        aggregation_statistic='mean',
        frequency_interval=frequency_interval,
        utc_offset=utc_offset,
        spatial_resolution='point',
        version='1',
        license='CC-BY-4.0',
        attribution='Environment Canterbury',
    )


def stations_dict(refs_lonlat):
    return {ref: {'lon': x[0], 'lat': x[1], 'name': x[2]} for ref, x in refs_lonlat.items()}


STNS_AB = {'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo')}


def _series(ref_vals, start=BASE, n=4, step=HOUR):
    times = start + step * np.arange(n)
    return {ref: (times, np.asarray(vals, dtype='float64')) for ref, vals in ref_vals.items()}


def _time_axis(ds):
    return np.asarray(ds['time'].data)


# --- hourly baseline (the original suite, on the generalized code) ---


def test_build_and_validate(tmp_path):
    stns = stations_dict(STNS_AB)
    h = _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, h, **ENC)

    with open_dataset(str(p)) as ds:
        assert ds.dataset_type == 'ts_ortho'
        assert ds['streamflow'].shape == (2, 4)
        assert _time_axis(ds).dtype == np.dtype('datetime64[h]')  # natural unit for 1h
        assert ds['time'].step == 1  # explicit step, native ticks
        expected_ids = [
            envlib.compute_station_id(shapely.Point(172.5, -43.5)),
            envlib.compute_station_id(shapely.Point(171.9, -43.1)),
        ]
        assert list(ds['station_id'].data) == expected_ids
        # the source's native identifier (the stations-dict key) is persisted per station
        assert list(ds['station_ref'].data) == ['A', 'B']
        np.testing.assert_allclose(np.asarray(ds['streamflow'][:].data)[0], [1, 2, 3, 4], rtol=1e-3)

    res = Catalogue(remotes=[], cache=str(tmp_path / 'cache')).validate(str(p))
    assert res['metadata'].variable == 'streamflow'
    assert res['state']['dataset_type'] == 'ts_ortho'
    assert res['standard_name']['value'] == 'water_volume_transport_in_river_channel'


def test_merge_idempotent(tmp_path):
    stns = stations_dict(STNS_AB)
    h = _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, h, **ENC)

    # a window that overlaps the last 2 hours and adds 2 new hours (revised + new)
    win = _series({'A': [30.0, 40.0, 50.0, 60.0], 'B': [12.0, 13.0, 14.0, 15.0]}, start=BASE + 2 * HOUR, n=4)

    def run_merge():
        with open_dataset(str(p), flag='w') as ds:
            return merge_dataset(ds, stns, win, variable='streamflow')

    r1 = run_merge()
    with open_dataset(str(p)) as ds:
        a1 = ds['streamflow'][:].data.copy()
        t1 = _time_axis(ds)
    r2 = run_merge()  # second identical run
    with open_dataset(str(p)) as ds:
        a2 = ds['streamflow'][:].data.copy()
        t2 = _time_axis(ds)

    assert r1['new_steps'] == 2 and r2['new_steps'] == 0  # second run adds nothing
    assert np.array_equal(t1, t2)
    np.testing.assert_array_equal(np.nan_to_num(a1, nan=-1), np.nan_to_num(a2, nan=-1))
    np.testing.assert_allclose(a2[0], [1, 2, 30, 40, 50, 60], rtol=1e-3)


def test_merge_offline_station_not_clobbered(tmp_path):
    stns = stations_dict(STNS_AB)
    h = _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, h, **ENC)

    # only A reports this window (B offline, empty tuple) -> B's existing hours must survive
    win = _series({'A': [300.0, 400.0]}, start=BASE + 2 * HOUR, n=2)
    win['B'] = (np.empty(0, dtype='datetime64[us]'), np.empty(0, dtype='float64'))
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, win, variable='streamflow')
    with open_dataset(str(p)) as ds:
        a = ds['streamflow'][:].data
    np.testing.assert_allclose(a[0], [1, 2, 300, 400], rtol=1e-3)  # A overwritten
    np.testing.assert_allclose(a[1], [10, 11, 12, 13], rtol=1e-3)  # B untouched


def test_merge_new_station(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)

    stns2 = stations_dict({'A': (172.5, -43.5, 'Alpha'), 'C': (170.0, -44.0, 'Charlie')})
    win = _series({'A': [3.0, 4.0], 'C': [99.0, 98.0]}, start=BASE + 2 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns2, win, variable='streamflow')
    assert r['new_stations'] == 1
    with open_dataset(str(p)) as ds:
        assert ds['streamflow'].shape[0] == 2
        assert 'Charlie' in list(ds['station_name'].data)
        # the appended station carries its ref, row-aligned with its name
        names_ = list(ds['station_name'].data)
        refs_ = list(ds['station_ref'].data)
        assert refs_[names_.index('Charlie')] == 'C'
        assert refs_[names_.index('Alpha')] == 'A'
        a = ds['streamflow'][:].data
    np.testing.assert_allclose(a[1, 2:4], [99, 98], rtol=1e-3)

    res = Catalogue(remotes=[], cache=str(tmp_path / 'c')).validate(str(p))  # still valid after merge
    assert res['state']['dataset_type'] == 'ts_ortho'


def test_merge_ignores_stations_without_data(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    full = stations_dict({'A': (172.5, -43.5, 'Alpha'), 'D': (170.5, -43.8, 'Delta')})  # D has no data
    win = _series({'A': [3.0, 4.0]}, start=BASE + 2 * HOUR, n=2)  # only A reports
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, full, win, variable='streamflow')
    assert r['new_stations'] == 0
    with open_dataset(str(p)) as ds:
        assert ds['streamflow'].shape[0] == 1
        assert list(ds['station_name'].data) == ['Alpha']


# --- the frequency matrix ---


def test_daily_build_merge_idempotent(tmp_path):
    stns = stations_dict(STNS_AB)
    h = _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]}, step=DAY)
    p = tmp_path / 'daily.cfdb'
    build_local(p, make_meta('day'), stns, h, **ENC)
    with open_dataset(str(p)) as ds:
        assert _time_axis(ds).dtype == np.dtype('datetime64[D]')
        assert ds['time'].step == 1
        np.testing.assert_allclose(np.asarray(ds['streamflow'][:].data)[0], [1, 2, 3, 4], rtol=1e-3)
    res = Catalogue(remotes=[], cache=str(tmp_path / 'cache')).validate(str(p))
    assert res['state']['dataset_type'] == 'ts_ortho'

    win = _series({'A': [30.0, 40.0, 50.0, 60.0]}, start=BASE + 2 * DAY, n=4, step=DAY)

    def run():
        with open_dataset(str(p), flag='w') as ds:
            return merge_dataset(ds, stns, win, variable='streamflow')

    r1, r2 = run(), run()
    assert r1['new_steps'] == 2 and r2['new_steps'] == 0
    with open_dataset(str(p)) as ds:
        assert _time_axis(ds).dtype == np.dtype('datetime64[D]')
        np.testing.assert_allclose(ds['streamflow'][:].data[0], [1, 2, 30, 40, 50, 60], rtol=1e-3)


def test_15min_build_merge(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'q.cfdb'
    build_local(p, make_meta('15min'), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}, step=Q15), **ENC)
    with open_dataset(str(p)) as ds:
        assert _time_axis(ds).dtype == np.dtype('datetime64[m]')
        assert ds['time'].step == 15
    win = _series({'A': [9.0, 8.0]}, start=BASE + 4 * Q15, n=2, step=Q15)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')
    assert r['new_steps'] == 2
    res = Catalogue(remotes=[], cache=str(tmp_path / 'cache')).validate(str(p))
    assert res['state']['dataset_type'] == 'ts_ortho'


def test_12h_build(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'h12.cfdb'
    build_local(p, make_meta('12h'), stns, _series({'A': [1.0, 2.0, 3.0]}, n=3, step=H12), **ENC)
    with open_dataset(str(p)) as ds:
        t = _time_axis(ds)
        assert t.dtype == np.dtype('datetime64[h]')
        assert ds['time'].step == 12
        assert t[1] == np.datetime64('2020-01-01T12', 'h')  # labels at 00:00Z / 12:00Z
        np.testing.assert_allclose(np.asarray(ds['streamflow'][:].data)[0], [1, 2, 3], rtol=1e-3)


def test_single_timestep_build_then_merge_gap_fills(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'one.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [5.0]}, n=1), **ENC)
    with open_dataset(str(p)) as ds:
        assert _time_axis(ds).size == 1
        assert ds['time'].step == 1  # explicit step even with one value (auto-detect would be None)
    win = _series({'A': [7.0, 8.0]}, start=BASE + 3 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')
    assert r['new_steps'] == 4
    with open_dataset(str(p)) as ds:
        t = _time_axis(ds)
        assert t.size == 5  # dense: 00,01,02,03,04
        a = np.asarray(ds['streamflow'][:].data)[0]
    np.testing.assert_allclose(a[[0, 3, 4]], [5, 7, 8], rtol=1e-3)
    assert np.isnan(a[[1, 2]]).all()  # gap-filled hours empty, not fabricated


def test_alignment_guard(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    # hourly-resampled series into a daily-declared dataset
    with pytest.raises(ValueError, match=r"station 'A'.*not aligned.*'day'"):
        build_local(tmp_path / 'x.cfdb', make_meta('day'), stns, _series({'A': [1.0, 2.0]}, n=2), **ENC)
    # a non-multiple timestamp under 1h
    bad = {'A': (np.array(['2020-01-01T00:30'], dtype='datetime64[us]'), np.array([1.0]))}
    with pytest.raises(ValueError, match='not aligned'):
        build_local(tmp_path / 'y.cfdb', make_meta('1h'), stns, bad, **ENC)


def test_phase_metadata_guard(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    h = _series({'A': [1.0, 2.0]}, n=2, step=DAY)
    # day + retained +12:00 declares local-midnight binning -> rejected
    with pytest.raises(ValueError, match='phase-anchored'):
        build_local(tmp_path / 'x.cfdb', make_meta('day', utc_offset='+12:00'), stns, h, **ENC)
    # positive control: 1h + +12:00 REDUCES to +00:00 (12h % 1h == 0) -> allowed
    hourly = _series({'A': [1.0, 2.0]}, n=2)
    build_local(tmp_path / 'ok.cfdb', make_meta('1h', utc_offset='+12:00'), stns, hourly, **ENC)


def test_calendar_and_none_rejected(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    h = _series({'A': [1.0, 2.0]}, n=2, step=DAY)
    with pytest.raises(ValueError, match='calendar-based'):
        build_local(tmp_path / 'm.cfdb', make_meta('month'), stns, h, **ENC)
    with pytest.raises(ValueError, match='irregular'):
        build_local(tmp_path / 'n.cfdb', make_meta(None), stns, h, **ENC)


def test_merge_crosscheck_declared_vs_actual(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    with open_dataset(str(p), flag='w') as ds:
        ds.attrs['envlib_frequency_interval'] = 'day'  # doctor the declaration
    win = _series({'A': [9.0]}, start=BASE + 4 * DAY, n=1, step=DAY)
    with open_dataset(str(p), flag='w') as ds, pytest.raises(ValueError, match='not a dense'):
        merge_dataset(ds, stns, win, variable='streamflow')


def _build_legacy_us(path, meta, stations, series):
    """Replicate the pre-generalization builder: datetime64[us] hourly axis, auto-detected step."""
    hour_us = 3_600_000_000
    non = {r: (np.asarray(t, dtype='datetime64[us]'), np.asarray(v, 'float64')) for r, (t, v) in series.items()}
    t0 = min(int(t.astype('int64').min()) for t, _ in non.values())
    t1 = max(int(t.astype('int64').max()) for t, _ in non.values())
    times = np.arange(t0, t1 + 1, hour_us).astype('datetime64[us]')
    points, ids, names, _refs = _points_ids_names(stations)  # legacy shape: no station_ref var
    data = np.full((len(stations), times.size), np.nan)
    for i, ref in enumerate(stations):
        t, v = non[ref]
        data[i, (t.astype('int64') - t0) // hour_us] = v
    dt = dtypes.dtype('float32', precision=4, min_value=0, max_value=100000)
    with open_dataset(str(path), flag='n', dataset_type='ts_ortho') as ds:
        ds.create.coord.point()
        ds['point'].append(points)
        ds.create.coord.time(data=times)
        ds.create.crs.from_user_input(4326, xy_coord='point')
        dv = ds.create.data_var.generic('streamflow', ('point', 'time'), dtype=dt)
        dv.attrs['units'] = 'm^3/s'
        dv[:] = data
        sid = ds.create.data_var.generic('station_id', ('point',), dtype=dtypes.dtype('str'))
        sid[:] = ids
        snm = ds.create.data_var.generic('station_name', ('point',), dtype=dtypes.dtype('str'))
        snm[:] = names
        ds.attrs.update(meta.to_dict())


def test_merge_onto_legacy_us_axis(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'legacy.cfdb'
    _build_legacy_us(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}))
    win = _series({'A': [30.0, 40.0]}, start=BASE + 2 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')
    assert r['new_steps'] == 0
    with open_dataset(str(p)) as ds:
        assert _time_axis(ds).dtype == np.dtype('datetime64[us]')  # stored dtype preserved
        np.testing.assert_allclose(ds['streamflow'][:].data[0], [1, 2, 30, 40], rtol=1e-3)


def test_merge_revise_only_and_window_before_axis(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    # revise-only: window fully inside the axis -> no new steps, values overwritten
    win = _series({'A': [20.0, 30.0]}, start=BASE + 1 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')
    assert r['new_steps'] == 0
    with open_dataset(str(p)) as ds:
        np.testing.assert_allclose(ds['streamflow'][:].data[0], [1, 20, 30, 4], rtol=1e-3)
    # a window entirely before the axis start must raise, not write a degenerate slice
    before = _series({'A': [9.0]}, start=BASE - 3 * HOUR, n=1)
    with open_dataset(str(p), flag='w') as ds, pytest.raises(ValueError, match='before the axis start'):
        merge_dataset(ds, stns, before, variable='streamflow')


def test_pre_1970_daily(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    start = np.datetime64('1969-12-28T00:00', 'us')
    p = tmp_path / 'old.cfdb'
    build_local(p, make_meta('day'), stns, _series({'A': [1.0, 2.0, 3.0]}, start=start, n=3, step=DAY), **ENC)
    with open_dataset(str(p)) as ds:
        t = _time_axis(ds)
        assert t.dtype == np.dtype('datetime64[D]')
        assert str(t[0]) == '1969-12-28'
    win = _series({'A': [40.0, 50.0]}, start=start + 2 * DAY, n=2, step=DAY)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')
    assert r['new_steps'] == 1
    with open_dataset(str(p)) as ds:
        np.testing.assert_allclose(ds['streamflow'][:].data[0], [1, 2, 40, 50], rtol=1e-3)


# --- Phase-7 fixtures: min/max QC, zombie refs, gap reporting (rulings 2026-07-17/18) ---


def test_qc_out_of_range_becomes_nan_and_never_clobbers(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    # incoming: hour 1 = valid revision (wins), hour 2 = out-of-range garbage -> QC NaN,
    # so the STORED valid 3.0 must survive (rejected values lose to existing data)
    win = _series({'A': [20.0, 5e9]}, start=BASE + 1 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')
    assert r['qc_rejected'] == 1
    with open_dataset(str(p)) as ds:
        np.testing.assert_allclose(ds['streamflow'][:].data[0], [1, 20, 3, 4], rtol=1e-3)


def test_qc_at_build_stores_nan(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 5e9, 3.0, 4.0]}), **ENC)
    with open_dataset(str(p)) as ds:
        a = np.asarray(ds['streamflow'][:].data)[0]
    np.testing.assert_allclose(a[[0, 2, 3]], [1, 3, 4], rtol=1e-3)
    assert np.isnan(a[1])  # rejected -> missing; never a fabricated in-range value


def test_zombie_series_refs_raise_pre_mutation(tmp_path):
    # a series ref absent from stations is a broken premise (station-list-driven
    # extraction): raise loudly, never silently drop (build) or KeyError mid-write (merge)
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    ghost = _series({'A': [1.0, 2.0], 'GHOST': [9.0, 9.0]}, n=2)
    with pytest.raises(ValueError, match=r"missing from stations: \['GHOST'\]"):
        build_local(tmp_path / 'x.cfdb', make_meta(), stns, ghost, **ENC)
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    with open_dataset(str(p), flag='w') as ds, pytest.raises(ValueError, match='GHOST'):
        merge_dataset(ds, stns, ghost, variable='streamflow')
    with open_dataset(str(p)) as ds:  # the merge raised before any mutation
        assert ds['streamflow'].shape == (1, 4)


def test_merge_reports_gap_steps(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    contig = _series({'A': [5.0, 6.0]}, start=BASE + 4 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        assert merge_dataset(ds, stns, contig, variable='streamflow')['gap_steps'] == 0
    hole = _series({'A': [9.0]}, start=BASE + 10 * HOUR, n=1)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, hole, variable='streamflow')
    assert r['gap_steps'] == 4  # hours 6..9 left unfilled behind the window (healable: NaN)


# --- fix-round fixtures (dual review 2026-07-18): tripwire, declared-bounds QC, straddle, floor ---


def test_stored_holes_survive_merge_at_production_chunk_width(tmp_path):
    # TRIPWIRE for the encode-fabrication class: cfdb's partial-chunk read-modify-write
    # re-encodes STORED holes on every merge. At production chunk width (SIMD cast path),
    # broken cfdb (< 0.9.4) fabricates 214747.36 into every hole; this must never recur.
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    t1 = BASE + HOUR * np.arange(10)
    t2 = BASE + HOUR * np.arange(20, 48)  # 10-hour outage hole at hours 10..19
    times = np.concatenate([t1, t2])
    p = tmp_path / 'hole.cfdb'
    build_local(p, make_meta(), stns, {'A': (times, np.full(times.size, 50.0))}, chunk_shape=(1, 6000), **ENC)
    win = _series({'A': [60.0] * 6}, start=BASE + 44 * HOUR, n=6)
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, win, variable='streamflow')
    with open_dataset(str(p)) as ds:
        a = np.asarray(ds['streamflow'][:].data)[0]
    assert np.isnan(a[10:20]).all()  # the hole survives the chunk re-encode


def test_qc_rejects_inf_and_declared_bounds(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    # 150000 is ENCODABLE (uint32 headroom) but outside the DECLARED max of 100000 -> reject;
    # +inf must be rejected and counted, and neither may clobber stored valid values
    win = _series({'A': [np.inf, 150000.0]}, start=BASE + 1 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')
    assert r['qc_rejected'] == 2
    with open_dataset(str(p)) as ds:
        np.testing.assert_allclose(ds['streamflow'][:].data[0], [1, 2, 3, 4], rtol=1e-3)
        assert float(ds['streamflow'].attrs['valid_max']) == 100000.0  # declared bounds persisted


def test_merge_straddling_window_counts_dropped_values(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    # window straddles the axis start: pre-axis values are dropped but REPORTED, in-axis land
    win = _series({'A': [7.0, 8.0, 9.0, 20.0]}, start=BASE - 3 * HOUR, n=4)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')
    assert r['dropped_before_axis'] == 3
    with open_dataset(str(p)) as ds:
        np.testing.assert_allclose(ds['streamflow'][:].data[0], [20, 2, 3, 4], rtol=1e-3)


def test_version_floor_refuses_broken_cfdb(tmp_path, monkeypatch):
    monkeypatch.setattr(cfdb_module, '__version__', '0.9.3')
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    with pytest.raises(RuntimeError, match=r'cfdb >= 0\.9\.4 required'):
        build_local(tmp_path / 'x.cfdb', make_meta(), stns, _series({'A': [1.0, 2.0]}, n=2), **ENC)


# --- station CF attrs + optional station_altitude ---


def _alt_stations(spec):
    """{ref: (lon, lat, name, altitude_or_None)} -> stations dict; the 'altitude' key is omitted
    entirely when the altitude is None (mirrors a source that supplies the column for some sites)."""
    out = {}
    for ref, (lon, lat, name, alt) in spec.items():
        d = {'lon': lon, 'lat': lat, 'name': name}
        if alt is not None:
            d['altitude'] = alt
        out[ref] = d
    return out


def test_build_with_altitude(tmp_path):
    stns = _alt_stations({'A': (172.5, -43.5, 'Alpha', 217.4), 'B': (171.9, -43.1, 'Bravo', None)})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [5.0, 6.0, 7.0, 8.0]}), **ENC)
    with open_dataset(str(p)) as ds:
        assert STATION_ALTITUDE_VAR in ds
        alt = np.asarray(ds[STATION_ALTITUDE_VAR][:].data, dtype='float64')  # row order = dict order A,B
        np.testing.assert_allclose(alt[0], 217.4, rtol=1e-4)
        assert np.isnan(alt[1])  # B supplied no altitude
        a = ds[STATION_ALTITUDE_VAR].attrs
        assert a['standard_name'] == 'altitude'
        assert a['units'] == 'm'
        assert a['long_name'] == 'station altitude'
        assert float(a['valid_min']) == -500.0 and float(a['valid_max']) == 9000.0
        assert 'not characterised' in a['comment']  # honest "no precision declared" note
        # the string station vars carry their CF attrs; station_id marks the timeseries instance
        assert ds['station_id'].attrs['cf_role'] == 'timeseries_id'
        assert ds['station_id'].attrs['long_name'] == 'envlib station identifier'
        assert ds['station_name'].attrs['long_name'] == 'station name'
        assert ds['station_ref'].attrs['long_name'] == 'source station reference identifier'
        assert ds.attrs.data['featureType'] == 'timeSeries'  # cfdb writes this for ts_ortho
    res = Catalogue(remotes=[], cache=str(tmp_path / 'c')).validate(str(p))
    assert res['state']['dataset_type'] == 'ts_ortho'


def test_build_without_altitude_no_var(tmp_path):
    # ECan shape: no 'altitude' key anywhere -> no var created
    p1 = tmp_path / 'ecan.cfdb'
    build_local(p1, make_meta(), stations_dict(STNS_AB), _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [5, 6, 7, 8]}), **ENC)
    with open_dataset(str(p1)) as ds:
        assert STATION_ALTITUDE_VAR not in ds
    # all-None altitudes: still no var (the trigger is any NON-null altitude, not the key's presence)
    p2 = tmp_path / 'allnone.cfdb'
    stns = _alt_stations({'A': (172.5, -43.5, 'Alpha', None), 'B': (171.9, -43.1, 'Bravo', None)})
    build_local(p2, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [5, 6, 7, 8]}), **ENC)
    with open_dataset(str(p2)) as ds:
        assert STATION_ALTITUDE_VAR not in ds


def test_build_altitude_zero_is_real(tmp_path):
    # altitude 0.0 is a real datum-relative value, not "missing" -> var created, stores 0.0
    stns = _alt_stations({'A': (172.5, -43.5, 'Alpha', 0.0)})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    with open_dataset(str(p)) as ds:
        assert STATION_ALTITUDE_VAR in ds
        assert float(np.asarray(ds[STATION_ALTITUDE_VAR][:].data)[0]) == 0.0


def test_merge_heals_attrs(tmp_path):
    # _build_legacy_us writes station_id + station_name with NO attrs (and no station_ref/altitude)
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'legacy.cfdb'
    _build_legacy_us(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}))
    with open_dataset(str(p)) as ds:
        assert dict(ds['station_id'].attrs.data) == {}  # attr-less to start with
    win = _series({'A': [3.0, 4.0]}, start=BASE + 2 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, win, variable='streamflow')
    with open_dataset(str(p)) as ds:
        assert ds['station_id'].attrs['cf_role'] == 'timeseries_id'
        assert ds['station_id'].attrs['long_name'] == 'envlib station identifier'
        assert ds['station_name'].attrs['long_name'] == 'station name'
        assert 'station_ref' not in ds  # absent vars are simply skipped, no error
        first = dict(ds['station_id'].attrs.data)
    # a second merge leaves the attrs identical (idempotent heal)
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, win, variable='streamflow')
    with open_dataset(str(p)) as ds:
        assert dict(ds['station_id'].attrs.data) == first


def test_merge_new_station_altitude(tmp_path):
    stns = _alt_stations({'A': (172.5, -43.5, 'Alpha', 100.0)})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    # merge adds C (with altitude) and D (without); existing A must be untouched
    stns2 = _alt_stations(
        {
            'A': (172.5, -43.5, 'Alpha', 100.0),
            'C': (170.0, -44.0, 'Charlie', 555.5),
            'D': (169.0, -43.0, 'Delta', None),
        }
    )
    win = _series({'A': [3.0, 4.0], 'C': [9.0, 8.0], 'D': [1.0, 2.0]}, start=BASE + 2 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns2, win, variable='streamflow')
    assert r['new_stations'] == 2
    with open_dataset(str(p)) as ds:
        names_ = list(ds['station_name'].data)
        alt = np.asarray(ds[STATION_ALTITUDE_VAR][:].data, dtype='float64')  # point-aligned with names
        assert alt[names_.index('Alpha')] == 100.0  # existing row untouched
        np.testing.assert_allclose(alt[names_.index('Charlie')], 555.5, rtol=1e-4)
        assert np.isnan(alt[names_.index('Delta')])  # new station lacking altitude -> NaN


def test_merge_altitude_warning_when_var_missing(tmp_path, caplog):
    # dataset built WITHOUT altitude; an EXISTING station later reports one (n_new == 0 path)
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    stns2 = _alt_stations({'A': (172.5, -43.5, 'Alpha', 42.0)})
    win = _series({'A': [3.0, 4.0]}, start=BASE + 2 * HOUR, n=2)
    with caplog.at_level(logging.WARNING):
        with open_dataset(str(p), flag='w') as ds:
            r = merge_dataset(ds, stns2, win, variable='streamflow')
    assert r['new_stations'] == 0
    assert any('has no station_altitude var' in m for m in caplog.messages)
    with open_dataset(str(p)) as ds:  # data merged despite the dropped altitude
        assert STATION_ALTITUDE_VAR not in ds
        np.testing.assert_allclose(ds['streamflow'][:].data[0], [1, 2, 3, 4], rtol=1e-3)


def test_altitudes_conversion(caplog):
    stns = {
        'A': {'lon': 0, 'lat': 0, 'name': 'a', 'altitude': 12.5},
        'B': {'lon': 0, 'lat': 0, 'name': 'b'},  # absent -> NaN
        'C': {'lon': 0, 'lat': 0, 'name': 'c', 'altitude': None},  # None -> NaN
        'D': {'lon': 0, 'lat': 0, 'name': 'd', 'altitude': np.nan},  # NaN -> NaN
        'E': {'lon': 0, 'lat': 0, 'name': 'e', 'altitude': '123.4'},  # str coerces
        'F': {'lon': 0, 'lat': 0, 'name': 'f', 'altitude': 217},  # int coerces
    }
    alt = _altitudes(stns)
    assert alt.dtype == np.float32
    np.testing.assert_allclose(alt[0], 12.5, rtol=1e-4)
    assert np.isnan(alt[1]) and np.isnan(alt[2]) and np.isnan(alt[3])
    np.testing.assert_allclose(alt[4], 123.4, rtol=1e-4)
    np.testing.assert_allclose(alt[5], 217.0, rtol=1e-4)

    # numeric-but-out-of-band (sentinel / ±inf / float32 overflow) -> MISSING (NaN) + named warning,
    # NOT a raise: the declared range doubles as QC, exactly like the data variable.
    with caplog.at_level(logging.WARNING):
        bad = _altitudes(
            {
                'G': {'lon': 0, 'lat': 0, 'name': 'g', 'altitude': -9999.0},  # sentinel
                'H': {'lon': 0, 'lat': 0, 'name': 'h', 'altitude': np.inf},
                'I': {'lon': 0, 'lat': 0, 'name': 'i', 'altitude': 1e39},  # overflows float32
                'J': {'lon': 0, 'lat': 0, 'name': 'j', 'altitude': 250.0},  # in band
            }
        )
    assert np.isnan(bad[0]) and np.isnan(bad[1]) and np.isnan(bad[2])
    np.testing.assert_allclose(bad[3], 250.0, rtol=1e-4)
    assert any('out-of-range altitude' in m for m in caplog.messages)

    # a non-coercible / wrong-type value is an adapter bug -> raises, naming the station
    with pytest.raises(ValueError, match=r"station 'X': invalid altitude"):
        _altitudes({'X': {'lon': 0, 'lat': 0, 'name': 'x', 'altitude': 'n/a'}})


def test_merge_heals_on_empty_window(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'legacy.cfdb'
    _build_legacy_us(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}))
    with open_dataset(str(p), flag='w') as ds:  # empty series -> zero report, but attrs still heal
        r = merge_dataset(ds, stns, {}, variable='streamflow')
    assert r == {
        'new_stations': 0,
        'new_steps': 0,
        'written_block': 0,
        'gap_steps': 0,
        'qc_rejected': 0,
        'anc_qc_rejected': 0,
        'dropped_before_axis': 0,
    }
    with open_dataset(str(p)) as ds:
        assert ds['station_id'].attrs['cf_role'] == 'timeseries_id'


def test_merge_bad_altitude_raises_before_mutating(tmp_path):
    # a NON-COERCIBLE altitude (adapter bug) must raise BEFORE any row is appended — never leave a
    # half-written station whose altitude is then NaN forever (write-once). Numeric out-of-band
    # values are handled as missing instead (see test_altitude_out_of_range_becomes_nan).
    stns = _alt_stations({'A': (172.5, -43.5, 'Alpha', 100.0)})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)
    with open_dataset(str(p)) as ds:
        before = ds['point'].shape[0]

    # new station E carries a non-coercible altitude -> must raise, and E must NOT be appended
    stns2 = _alt_stations({'A': (172.5, -43.5, 'Alpha', 100.0), 'E': (168.0, -45.0, 'Echo', None)})
    stns2['E']['altitude'] = 'n/a'
    win = _series({'A': [3.0, 4.0], 'E': [1.0, 2.0]}, start=BASE + 2 * HOUR, n=2)
    with pytest.raises(ValueError, match=r"station 'E': invalid altitude"):
        with open_dataset(str(p), flag='w') as ds:
            merge_dataset(ds, stns2, win, variable='streamflow')
    with open_dataset(str(p)) as ds:
        assert ds['point'].shape[0] == before  # no zombie row left behind
        assert 'Echo' not in list(ds['station_name'].data)

    # same loudness on a non-coercible value when the dataset has NO altitude var
    p2 = tmp_path / 'noalt.cfdb'
    build_local(p2, make_meta(), stations_dict({'A': (172.5, -43.5, 'Alpha')}), _series({'A': [1.0, 2.0]}, n=2), **ENC)
    bad = {'A': {'lon': 172.5, 'lat': -43.5, 'name': 'Alpha', 'altitude': 'xyz'}}
    with pytest.raises(ValueError, match=r"station 'A': invalid altitude"):
        with open_dataset(str(p2), flag='w') as ds:
            merge_dataset(ds, bad, _series({'A': [3.0, 4.0]}, start=BASE + 1 * HOUR, n=2), variable='streamflow')


def test_heal_rerun_is_upload_clean(tmp_path):
    # the "re-running the heal re-uploads nothing" claim: after the first heal stamps the attrs,
    # a second empty-window merge must leave the file BYTE-identical (cfdb's attrs finalizer only
    # writes on real change). A future cfdb regression that re-writes unchanged attrs breaks this.
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'legacy.cfdb'
    _build_legacy_us(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}))  # attr-less to start

    def _sha():
        return hashlib.sha256(p.read_bytes()).hexdigest()

    with open_dataset(str(p), flag='w') as ds:  # first heal writes the attrs
        merge_dataset(ds, stns, {}, variable='streamflow')
    h1 = _sha()
    with open_dataset(str(p), flag='w') as ds:  # re-run must not touch a byte
        merge_dataset(ds, stns, {}, variable='streamflow')
    assert _sha() == h1


def test_altitude_out_of_range_becomes_nan(tmp_path, caplog):
    # a sentinel altitude (-9999) is treated as MISSING (NaN) at build, the in-band station is kept,
    # a warning is logged, and the band is persisted as valid_min/valid_max attrs. Then a merge with
    # an out-of-band new-station altitude lands NaN too (missing, not a raise).
    stns = _alt_stations({'A': (172.5, -43.5, 'Alpha', 250.0), 'B': (171.9, -43.1, 'Bravo', -9999.0)})
    p = tmp_path / 'sf.cfdb'
    with caplog.at_level(logging.WARNING):
        build_local(p, make_meta(), stns, _series({'A': [1.0, 2, 3, 4], 'B': [5.0, 6, 7, 8]}), **ENC)
    assert any('out-of-range altitude' in m for m in caplog.messages)
    with open_dataset(str(p)) as ds:
        names_ = list(ds['station_name'].data)
        alt = np.asarray(ds[STATION_ALTITUDE_VAR][:].data, dtype='float64')
        np.testing.assert_allclose(alt[names_.index('Alpha')], 250.0, rtol=1e-4)
        assert np.isnan(alt[names_.index('Bravo')])  # sentinel -> missing
        a = ds[STATION_ALTITUDE_VAR].attrs
        assert float(a['valid_min']) == -500.0 and float(a['valid_max']) == 9000.0

    # merge a NEW station whose altitude is out of band -> stored NaN (missing), no raise
    stns2 = _alt_stations(
        {
            'A': (172.5, -43.5, 'Alpha', 250.0),
            'B': (171.9, -43.1, 'Bravo', -9999.0),
            'C': (170.0, -44.0, 'Charlie', 1e39),  # overflows float32 -> missing
        }
    )
    win = _series({'A': [3.0, 4.0], 'B': [5.0, 6.0], 'C': [7.0, 8.0]}, start=BASE + 2 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns2, win, variable='streamflow')
    assert r['new_stations'] == 1
    with open_dataset(str(p)) as ds:
        names_ = list(ds['station_name'].data)
        alt = np.asarray(ds[STATION_ALTITUDE_VAR][:].data, dtype='float64')
        assert np.isnan(alt[names_.index('Charlie')])  # out-of-band new station -> missing


# --- ancillary variables (per-timestep companion planes, e.g. a NEMS quality grade) ---

QC_SPEC = {
    'quality_code': {
        'units': '1',
        'precision': 0,
        'min_value': 0,
        'max_value': 1000,
        'attrs': {'standard_name': 'quality_flag', 'long_name': 'NEMS quality code'},
    }
}


def _series_qc(ref_vals, ref_codes, start=BASE, n=4, step=HOUR):
    """3-tuple series: values with an extras dict positionally paired to them."""
    times = start + step * np.arange(n)
    return {
        ref: (
            times,
            np.asarray(vals, dtype='float64'),
            {'quality_code': np.asarray(ref_codes[ref], dtype='float64')},
        )
        for ref, vals in ref_vals.items()
    }


def _read(p, var):
    with open_dataset(str(p)) as ds:
        return np.asarray(ds[var][:].data, dtype='float64')


def test_ancillary_build_roundtrip(tmp_path):
    stns = stations_dict(STNS_AB)
    s = _series_qc(
        {'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]},
        {'A': [600, 600, 400, 500], 'B': [600, 520, 520, 600]},
    )
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, s, **ENC, ancillary=QC_SPEC)

    with open_dataset(str(p)) as ds:
        # CF: the primary names its companion; the companion carries its own QC bounds
        assert ds['streamflow'].attrs['ancillary_variables'] == 'quality_code'
        qa = ds['quality_code'].attrs
        assert qa['standard_name'] == 'quality_flag'
        assert float(qa['valid_min']) == 0.0 and float(qa['valid_max']) == 1000.0
        assert ds['quality_code'].shape == ds['streamflow'].shape
        assert ds['quality_code'].chunk_shape == ds['streamflow'].chunk_shape
        codes = np.asarray(ds['quality_code'][:].data, dtype='float64')
    np.testing.assert_array_equal(codes[0], [600, 600, 400, 500])
    np.testing.assert_array_equal(codes[1], [600, 520, 520, 600])

    Catalogue(remotes=[], cache=str(tmp_path / 'cache')).validate(str(p))


def test_ancillary_code_without_value_survives_merge(tmp_path):
    # THE union-mask case: a grade whose whole job is to explain an ABSENT value (NEMS 100,
    # "missing record") must land, even though the value plane is NaN at that slot.
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series_qc({'A': [1.0, 2.0, 3.0, 4.0]}, {'A': [600, 600, 600, 600]}),
                **ENC, ancillary=QC_SPEC)

    win = {'A': (BASE + 2 * HOUR + HOUR * np.arange(2),
                 np.asarray([np.nan, np.nan]),
                 {'quality_code': np.asarray([100.0, 100.0])})}
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, win, variable='streamflow')

    vals, codes = _read(p, 'streamflow')[0], _read(p, 'quality_code')[0]
    assert np.isnan(vals[2]) and np.isnan(vals[3])  # value retracted to missing by the union mask
    np.testing.assert_array_equal(codes[2:], [100, 100])  # ...and the reason survived
    np.testing.assert_allclose(vals[:2], [1.0, 2.0], rtol=1e-3)  # untouched slots keep their pair
    np.testing.assert_array_equal(codes[:2], [600, 600])


def test_ancillary_value_without_code_does_not_leave_stale_code(tmp_path):
    # a run that supplies a value but no grade must not leave the PREVIOUS run's grade beside it
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series_qc({'A': [1.0, 2.0, 3.0, 4.0]}, {'A': [600, 600, 600, 600]}),
                **ENC, ancillary=QC_SPEC)

    win = {'A': (BASE + 2 * HOUR + HOUR * np.arange(2), np.asarray([30.0, 40.0]), {})}
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, win, variable='streamflow')

    vals, codes = _read(p, 'streamflow')[0], _read(p, 'quality_code')[0]
    np.testing.assert_allclose(vals[2:], [30.0, 40.0], rtol=1e-3)
    assert np.isnan(codes[2]) and np.isnan(codes[3])  # stale 600 must NOT survive under a new value
    np.testing.assert_array_equal(codes[:2], [600, 600])


def test_ancillary_all_nan_incoming_leaves_stored_pair(tmp_path):
    # the offline-station protection still holds with two planes: an all-NaN incoming slot
    # changes nothing (this is also why the merge cannot express a deletion)
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series_qc({'A': [1.0, 2.0, 3.0, 4.0]}, {'A': [600, 600, 600, 600]}),
                **ENC, ancillary=QC_SPEC)

    win = {'A': (BASE + 2 * HOUR + HOUR * np.arange(2),
                 np.asarray([np.nan, np.nan]),
                 {'quality_code': np.asarray([np.nan, np.nan])})}
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, win, variable='streamflow')

    np.testing.assert_allclose(_read(p, 'streamflow')[0], [1, 2, 3, 4], rtol=1e-3)
    np.testing.assert_array_equal(_read(p, 'quality_code')[0], [600, 600, 600, 600])


def test_ancillary_merge_idempotent(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series_qc({'A': [1.0, 2.0, 3.0, 4.0]}, {'A': [600, 600, 600, 600]}),
                **ENC, ancillary=QC_SPEC)
    win = _series_qc({'A': [30.0, 40.0, 50.0, 60.0]}, {'A': [400, 400, 520, 520]}, start=BASE + 2 * HOUR, n=4)

    def run():
        with open_dataset(str(p), flag='w') as ds:
            return merge_dataset(ds, stns, win, variable='streamflow')

    r1 = run()
    v1, c1 = _read(p, 'streamflow'), _read(p, 'quality_code')
    r2 = run()
    v2, c2 = _read(p, 'streamflow'), _read(p, 'quality_code')

    assert r1['new_steps'] == 2 and r2['new_steps'] == 0
    np.testing.assert_array_equal(np.nan_to_num(v1, nan=-1), np.nan_to_num(v2, nan=-1))
    np.testing.assert_array_equal(np.nan_to_num(c1, nan=-1), np.nan_to_num(c2, nan=-1))
    np.testing.assert_array_equal(c2[0], [600, 600, 400, 400, 520, 520])


def test_ancillary_qc_value_reject_drops_its_code(tmp_path):
    # a value refused by the primary QC bounds takes its grade with it: a plausibility grade
    # for a measurement we declined to store describes nothing
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    s = _series_qc({'A': [1.0, 1e9, 3.0, 4.0]}, {'A': [600, 600, 600, 600]})  # 1e9 > max_value
    build_local(p, make_meta(), stns, s, **ENC, ancillary=QC_SPEC)

    vals, codes = _read(p, 'streamflow')[0], _read(p, 'quality_code')[0]
    assert np.isnan(vals[1]) and np.isnan(codes[1])
    np.testing.assert_array_equal(codes[[0, 2, 3]], [600, 600, 600])


def test_ancillary_qc_code_reject_keeps_value(tmp_path):
    # an out-of-band CODE is dropped on its own; the measurement it annotates is still good.
    # NB the encoder passes above-max through, so valid_min/valid_max are what enforce this.
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    s = _series_qc({'A': [1.0, 2.0, 3.0, 4.0]}, {'A': [600, 32767, 600, 600]})  # int16 fill leaked in
    build_local(p, make_meta(), stns, s, **ENC, ancillary=QC_SPEC)

    vals, codes = _read(p, 'streamflow')[0], _read(p, 'quality_code')[0]
    np.testing.assert_allclose(vals, [1, 2, 3, 4], rtol=1e-3)  # value untouched
    assert np.isnan(codes[1])


def test_ancillary_ragged_extras_raise_pre_mutation(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    bad = {'A': (BASE + HOUR * np.arange(4), np.asarray([1.0, 2, 3, 4]),
                 {'quality_code': np.asarray([600.0, 600.0])})}  # 2 codes for 4 values
    with pytest.raises(ValueError, match='positionally paired'):
        build_local(tmp_path / 'sf.cfdb', make_meta(), stns, bad, **ENC, ancillary=QC_SPEC)
    assert not (tmp_path / 'sf.cfdb').exists()


def test_ancillary_undeclared_name_raises(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    s = {'A': (BASE + HOUR * np.arange(4), np.asarray([1.0, 2, 3, 4]),
               {'not_declared': np.asarray([1.0, 2, 3, 4])})}
    with pytest.raises(ValueError, match='not declared'):
        build_local(tmp_path / 'sf.cfdb', make_meta(), stns, s, **ENC, ancillary=QC_SPEC)


def test_extras_into_legacy_dataset_raise(tmp_path):
    # a dataset built WITHOUT an ancillary roster must refuse incoming codes rather than
    # silently discard them (no rebuild puts per-timestep data back)
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0]}), **ENC)  # no ancillary
    win = _series_qc({'A': [30.0, 40.0]}, {'A': [400, 400]}, start=BASE + 2 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds, pytest.raises(ValueError, match='not declared'):
        merge_dataset(ds, stns, win, variable='streamflow')


def test_no_ancillary_collapses_to_old_behaviour(tmp_path):
    # backward compatibility: with no ancillary plane the union mask must reduce EXACTLY to the
    # old "non-NaN incoming wins" rule, and no roster attr may appear.
    # (Not asserted byte-wise: two independent builds differ by cfdb's per-file uuid — the
    # existing sha test compares one file before/after, which is a different claim.)
    stns = stations_dict(STNS_AB)
    h = _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, h, **ENC)
    with open_dataset(str(p)) as ds:
        assert 'ancillary_variables' not in ds['streamflow'].attrs.data

    # B's first slot is NaN incoming -> must NOT clobber the stored 12.0
    win = _series({'A': [30.0, 40.0], 'B': [np.nan, 15.0]}, start=BASE + 2 * HOUR, n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns, win, variable='streamflow')

    a = _read(p, 'streamflow')
    np.testing.assert_allclose(a[0], [1, 2, 30, 40], rtol=1e-3)
    np.testing.assert_allclose(a[1], [10, 11, 12, 15], rtol=1e-3)  # 12.0 survived the NaN
    assert r['anc_qc_rejected'] == 0  # key present and inert on the no-ancillary path


def test_ancillary_repeat_merge_is_value_stable_not_byte_stable(tmp_path):
    # Pins the real contract, and guards against over-claiming it. A repeated non-empty merge is
    # idempotent in VALUES on both planes, but NOT byte-stable: booklet appends, so the file grows
    # a little each run (dead space, prunable). This is pre-existing and independent of ancillary
    # support — test_heal_rerun_is_upload_clean's byte claim holds only for an EMPTY window, which
    # returns before any write.
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'sf.cfdb'
    build_local(p, make_meta(), stns, _series_qc({'A': [1.0, 2.0, 3.0, 4.0]}, {'A': [600, 600, 600, 600]}),
                **ENC, ancillary=QC_SPEC)
    win = _series_qc({'A': [30.0, 40.0]}, {'A': [400, 400]}, start=BASE + 2 * HOUR, n=2)

    def _run():
        with open_dataset(str(p), flag='w') as ds:
            merge_dataset(ds, stns, win, variable='streamflow')
        return _read(p, 'streamflow'), _read(p, 'quality_code'), hashlib.sha256(p.read_bytes()).hexdigest()

    v1, c1, h1 = _run()
    v2, c2, h2 = _run()
    np.testing.assert_array_equal(np.nan_to_num(v1, nan=-1), np.nan_to_num(v2, nan=-1))
    np.testing.assert_array_equal(np.nan_to_num(c1, nan=-1), np.nan_to_num(c2, nan=-1))
    assert h1 != h2  # documents the append-growth; flip this if cfdb ever becomes write-eliding


# --- chunked writes: build_local must never materialise a whole plane --------------------------
# These lock in the 2026-08-06 refactor (dual-blind reviewed, round `chunked-1`). Before it,
# build_local allocated np.full((n_stations, n_times)) per plane and assigned it in one statement,
# which is what a chunked array database exists to avoid: memory scaled with the DIMENSIONS rather
# than with how much data there actually was. Measured on the real ECan streamflow build: 4.5 GB
# peak for a 51 MB, 20%-dense file.


def test_build_local_peak_allocation_is_bounded_by_one_row(tmp_path):
    """The PROPERTY, not the implementation: peak allocation must track a ROW, not a PLANE.

    Fails hard on the pre-refactor code — the dense plane alone is 240 MB here against ~4 MB of
    real data. `tracemalloc` is used rather than RSS because it captures numpy allocations
    deterministically and does not vary with the machine or the allocator's high-water mark.
    """
    n_stations, n_steps = 200, 150_000
    plane_bytes = n_stations * n_steps * 8          # 240 MB
    row_bytes = n_steps * 8                         # 1.2 MB

    rng = np.random.default_rng(0)
    stns = stations_dict({f'S{i:03d}': (171.0 + i * 0.01, -43.0 - i * 0.01, f's{i}') for i in range(n_stations)})
    series = {}
    for i, ref in enumerate(stns):
        if i % 5 == 0:
            continue                                # no data: still gets an all-NaN row
        idx = np.sort(rng.choice(n_steps, size=400, replace=False))
        series[ref] = (BASE + HOUR * idx, np.abs(rng.normal(10, 3, idx.size)))

    p = tmp_path / 'bounded.cfdb'
    tracemalloc.start()
    build_local(p, make_meta(), stns, series, **ENC)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # generous: 20 rows of headroom for cfdb's per-call chunk buffer and encode copies, but still
    # an order of magnitude below a single plane. Pre-refactor this measured >6x the plane.
    assert peak < 20 * row_bytes, (
        f'peak {peak / 1e6:.1f} MB exceeds 20 rows ({20 * row_bytes / 1e6:.1f} MB); '
        f'a whole plane would be {plane_bytes / 1e6:.1f} MB — the build is materialising again'
    )


def test_rows_align_to_stations_not_to_series(tmp_path):
    """Stations WITHOUT data must not shift the rows of the stations after them.

    The point coord and the station_id/ref/name vars are built in `stations` order, so row i must
    be the i-th entry of `stations`. Driving the row loop from `series` instead maps the i-th
    station-that-has-data to point i and silently mislabels every station after the first
    data-less one — producing a file that validates cleanly and is wrong.

    A fixture where stations == series cannot catch this, which is exactly why this one has gaps
    at the START, MIDDLE and END of the station order.
    """
    order = ['A', 'B', 'C', 'D', 'E']
    stns = stations_dict({r: (171.0 + i, -43.0 - i, f'name-{r}') for i, r in enumerate(order)})
    # A (first), C (middle) and E (last) have no data
    series = _series({'B': [1.0, 2.0, 3.0, 4.0], 'D': [10.0, 20.0, 30.0, 40.0]})

    p = tmp_path / 'align.cfdb'
    build_local(p, make_meta(), stns, series, **ENC)

    with open_dataset(str(p)) as ds:
        refs = [str(x) for x in np.asarray(ds['station_ref'].data)]
        names = [str(x) for x in np.asarray(ds['station_name'].data)]
        vals = np.asarray(ds['streamflow'][:])

    assert refs == order, f'station_ref order changed: {refs}'
    assert names == [f'name-{r}' for r in order]
    # the data must sit on B (row 1) and D (row 3) — NOT rows 0 and 1
    assert np.isnan(vals[0]).all(), 'row 0 (A, no data) should be all-NaN'
    assert np.isnan(vals[2]).all(), 'row 2 (C, no data) should be all-NaN'
    assert np.isnan(vals[4]).all(), 'row 4 (E, no data) should be all-NaN'
    np.testing.assert_allclose(vals[1, :4], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(vals[3, :4], [10.0, 20.0, 30.0, 40.0])


def test_each_chunk_written_exactly_once(tmp_path):
    """Revisiting a chunk is invisible in the output but costs real money.

    A rewritten chunk means a full decompress+recompress, orphans the previous block (the store is
    log-structured, so the file GROWS until pruned), and can force a synchronous buffer flush. The
    resulting file is still correct — so only a write counter catches a regression here.
    """
    n_steps = 60_000                                  # spans 3 chunks at the 25k default
    stns = stations_dict(STNS_AB)
    rng = np.random.default_rng(1)
    series = {r: (BASE + HOUR * np.sort(rng.choice(n_steps, 300, replace=False)),
                  np.abs(rng.normal(5, 1, 300)))
              for r in stns}

    p = tmp_path / 'once.cfdb'
    writes = {'n': 0}
    # cfdb writes chunks through booklet's VariableLengthValue.set (there is no `Booklet` class)
    real_set = booklet.VariableLengthValue.set

    def counting_set(self, key, value, *a, **kw):
        writes['n'] += 1
        return real_set(self, key, value, *a, **kw)

    booklet.VariableLengthValue.set = counting_set
    try:
        build_local(p, make_meta(), stns, series, **ENC)
    finally:
        booklet.VariableLengthValue.set = real_set

    with booklet.open(str(p)) as f:
        n_keys = len(list(f.keys()))
    # every stored key written once; a small allowance for metadata keys rewritten at close
    assert writes['n'] <= n_keys + 5, (
        f'{writes["n"]} writes for {n_keys} stored keys — chunks are being revisited'
    )


def test_merge_peak_does_not_scale_with_a_backdated_station(tmp_path):
    """One stale timestamp must not widen the read for EVERY station.

    `merge_dataset` used to read `dv[:, w_lo:w_hi+1]` across the full point dimension, with the
    bounds taken from the GLOBAL min/max incoming timestamp — so a single station carrying one
    backdated value stretched the block across all of them. Measured before the per-station
    rewrite: a routine 48-step merge went from 6 MB to 552 MB on a 150 x 60,000 dataset, and at
    production scale that is gigabytes inside an hourly cron.

    Per-station, only the offending station reads a wide row, so the peak is unchanged.
    """
    # enough stations that a whole-point-dim block is an order of magnitude above one row —
    # otherwise the assertion cannot discriminate between the two implementations
    n_stations, n_steps = 200, 20_000
    stns = stations_dict({f'S{i:03d}': (171.0 + i * 0.01, -43.0 - i * 0.01, f's{i}') for i in range(n_stations)})
    rng = np.random.default_rng(0)
    initial = {r: (BASE + HOUR * np.arange(n_steps), np.abs(rng.normal(10, 2, n_steps)))
               for r in stns}
    p = tmp_path / 'wide.cfdb'
    build_local(p, make_meta(), stns, initial, **ENC)

    # a routine window at the END of the axis, except ONE station also reports near the START
    tail = BASE + HOUR * np.arange(n_steps - 48, n_steps)
    window = {}
    for i, r in enumerate(stns):
        t, v = tail, np.abs(rng.normal(10, 2, 48))
        if i == 0:
            t = np.concatenate([[BASE + HOUR * 5], t])
            v = np.concatenate([[1.0], v])
        window[r] = (t, v)

    tracemalloc.start()
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, window, variable='streamflow')
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The old code held ~6-7 arrays of (n_point x width); the new one holds a handful of ROWS.
    # Assert against a quarter of ONE whole-point-dim block: comfortably above the per-station
    # cost (which includes cfdb's per-call chunk buffer) and far below even a single block.
    whole_block = n_stations * n_steps * 8
    assert peak < whole_block / 4, (
        f'merge peak {peak / 1e6:.1f} MB exceeds a quarter of one whole-point-dim block '
        f'({whole_block / 4e6:.1f} MB; the full block is {whole_block / 1e6:.1f} MB, and the old '
        f'code held ~6-7 of them) — a single backdated station is widening the read for every '
        f'station again'
    )


def test_merge_does_not_rewrite_stations_that_reported_nothing(tmp_path):
    """A station absent from the window must not have its chunks rewritten.

    The old whole-block merge wrote `existing` back over every station in the block, including
    ones that supplied nothing — identical bytes, but a rewrite all the same. In a log-structured
    store that orphans the previous block (the file grows), and because the remote pushes changed
    keys it also uploads those chunks on every run.
    """
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo')})
    p = tmp_path / 'quiet.cfdb'
    build_local(p, make_meta(), stns, _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]}), **ENC)

    writes = []
    real_set = booklet.VariableLengthValue.set

    def counting_set(self, key, value, *a, **kw):
        writes.append(key)
        return real_set(self, key, value, *a, **kw)

    booklet.VariableLengthValue.set = counting_set
    try:
        with open_dataset(str(p), flag='w') as ds:
            # n=2 is REQUIRED: _series defaults to n=4 and would build 4 timestamps for 2 values
            merge_dataset(ds, stns, _series({'A': [5.0, 6.0]}, start=BASE + 2 * HOUR, n=2),
                          variable='streamflow')
    finally:
        booklet.VariableLengthValue.set = real_set

    # booklet keys are `str`, formatted '{var}!{start0}.{start1}' — so B (row 1) is 'streamflow!1.*'.
    # NOTE the type: an earlier version of this test filtered on `bytes` and therefore matched
    # NOTHING, passing identically on the old and new code. A filter that never fires is not a test.
    assert all(isinstance(k, str) for k in writes), 'key type changed; this filter is now vacuous'
    a_row_writes = [k for k in writes if k.startswith('streamflow!0.')]
    b_row_writes = [k for k in writes if k.startswith('streamflow!1.')]
    assert a_row_writes, (
        'station A reported data but no chunk of its row was written — the filter is wrong, '
        'not the code'
    )
    assert not b_row_writes, f'station B reported nothing but its chunks were rewritten: {b_row_writes}'
