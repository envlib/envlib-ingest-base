"""cfdb-level tests for ts_ortho build + idempotent merge (no remote needed).

The axis cadence comes from ``meta.frequency_interval`` (the envlib CV) — the matrix here pins
the natural-unit coord dtypes, the explicit step, the two phase-contract guards, the merge
cross-checks, and legacy-[us]-axis compatibility.
"""

import cfdb as cfdb_module
import envlib
import numpy as np
import pytest
import shapely
from cfdb import dtypes, open_dataset
from envlib import Catalogue

from envlib_ingest_base.tsortho import _points_ids_names, build_local, merge_dataset

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
    points, ids, names = _points_ids_names(stations)
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
