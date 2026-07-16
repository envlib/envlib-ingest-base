"""cfdb-level tests for ts_ortho build + idempotent merge (no remote needed)."""

import envlib
import numpy as np
import pandas as pd
import shapely
from cfdb import open_dataset
from envlib import Catalogue

from envlib_ingest_base.tsortho import build_local, merge_dataset


def meta_streamflow():
    return envlib.Metadata(
        feature='waterway',
        variable='streamflow',
        method='sensor_recording',
        product_code=None,
        processing_level='raw',
        owner='ecan',
        aggregation_statistic='mean',
        frequency_interval='1h',
        utc_offset='+00:00',
        spatial_resolution='point',
        version='1',
        license='CC-BY-4.0',
        attribution='Environment Canterbury',
    )


def stations_df(refs_lonlat):
    return pd.DataFrame(
        {
            'lon': [x[0] for x in refs_lonlat.values()],
            'lat': [x[1] for x in refs_lonlat.values()],
            'name': [x[2] for x in refs_lonlat.values()],
        },
        index=list(refs_lonlat.keys()),
    )


def hourly(ref_to_pairs):
    out = {}
    for ref, pairs in ref_to_pairs.items():
        idx = pd.DatetimeIndex([p[0] for p in pairs], name='time')
        out[ref] = pd.Series([p[1] for p in pairs], index=idx)
    return out


BASE = pd.Timestamp('2020-01-01 00:00')


def _series(ref_vals, start=BASE, n=4):
    idx = pd.date_range(start, periods=n, freq='1h')
    return {ref: pd.Series(vals, index=idx) for ref, vals in ref_vals.items()}


def test_build_and_validate(tmp_path):
    stns = stations_df({'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo')})
    h = _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(
        p, meta_streamflow(), stns, h, variable='streamflow', units='m^3/s', precision=4, min_value=0, max_value=100000
    )

    with open_dataset(str(p)) as ds:
        assert ds.dataset_type == 'ts_ortho'
        assert ds['streamflow'].shape == (2, 4)
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
    stns = stations_df({'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo')})
    h = _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(
        p, meta_streamflow(), stns, h, variable='streamflow', units='m^3/s', precision=4, min_value=0, max_value=100000
    )

    # a window that overlaps the last 2 hours and adds 2 new hours (revised + new)
    win = _series({'A': [30.0, 40.0, 50.0, 60.0], 'B': [12.0, 13.0, 14.0, 15.0]}, start=BASE + pd.Timedelta('2h'), n=4)

    def run_merge():
        with open_dataset(str(p), flag='w') as ds:
            return merge_dataset(ds, stns, win, variable='streamflow')

    r1 = run_merge()
    with open_dataset(str(p)) as ds:
        a1 = ds['streamflow'][:].data.copy()
        t1 = pd.DatetimeIndex(ds['time'].data)
    r2 = run_merge()  # second identical run
    with open_dataset(str(p)) as ds:
        a2 = ds['streamflow'][:].data.copy()
        t2 = pd.DatetimeIndex(ds['time'].data)

    assert r1['new_hours'] == 2 and r2['new_hours'] == 0  # second run adds nothing
    assert t1.equals(t2)
    np.testing.assert_array_equal(np.nan_to_num(a1, nan=-1), np.nan_to_num(a2, nan=-1))
    # revised hour 2 overwritten; new hours appended
    row_a = a2[0]
    np.testing.assert_allclose(row_a, [1, 2, 30, 40, 50, 60], rtol=1e-3)


def test_merge_offline_station_not_clobbered(tmp_path):
    stns = stations_df({'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo')})
    h = _series({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 11.0, 12.0, 13.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(
        p, meta_streamflow(), stns, h, variable='streamflow', units='m^3/s', precision=4, min_value=0, max_value=100000
    )

    # only A reports this window (B offline) -> B's existing hours must survive
    win = {
        'A': pd.Series([300.0, 400.0], index=pd.date_range(BASE + pd.Timedelta('2h'), periods=2, freq='1h')),
        'B': pd.Series(dtype='float64', index=pd.DatetimeIndex([], name='time')),
    }
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stns, win, variable='streamflow')
    with open_dataset(str(p)) as ds:
        a = ds['streamflow'][:].data
    np.testing.assert_allclose(a[0], [1, 2, 300, 400], rtol=1e-3)  # A overwritten
    np.testing.assert_allclose(a[1], [10, 11, 12, 13], rtol=1e-3)  # B untouched


def test_merge_new_station(tmp_path):
    stns = stations_df({'A': (172.5, -43.5, 'Alpha')})
    h = _series({'A': [1.0, 2.0, 3.0, 4.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(
        p, meta_streamflow(), stns, h, variable='streamflow', units='m^3/s', precision=4, min_value=0, max_value=100000
    )

    stns2 = stations_df({'A': (172.5, -43.5, 'Alpha'), 'C': (170.0, -44.0, 'Charlie')})
    win = _series({'A': [3.0, 4.0], 'C': [99.0, 98.0]}, start=BASE + pd.Timedelta('2h'), n=2)
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, stns2, win, variable='streamflow')
    assert r['new_stations'] == 1
    with open_dataset(str(p)) as ds:
        assert ds['streamflow'].shape[0] == 2
        assert 'Charlie' in list(ds['station_name'].data)
        a = ds['streamflow'][:].data
    # Charlie's pre-existence hours are fill; its window hours written
    np.testing.assert_allclose(a[1, 2:4], [99, 98], rtol=1e-3)

    res = Catalogue(remotes=[], cache=str(tmp_path / 'c')).validate(str(p))  # still valid after merge
    assert res['state']['dataset_type'] == 'ts_ortho'


def test_merge_ignores_stations_without_data(tmp_path):
    # `stations` may be the full discovery frame; a station with no data this window must NOT be added
    stns = stations_df({'A': (172.5, -43.5, 'Alpha')})
    h = _series({'A': [1.0, 2.0, 3.0, 4.0]})
    p = tmp_path / 'sf.cfdb'
    build_local(
        p, meta_streamflow(), stns, h, variable='streamflow', units='m^3/s', precision=4, min_value=0, max_value=100000
    )
    full = stations_df({'A': (172.5, -43.5, 'Alpha'), 'D': (170.5, -43.8, 'Delta')})  # D has no data
    win = _series({'A': [3.0, 4.0]}, start=BASE + pd.Timedelta('2h'), n=2)  # only A reports
    with open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, full, win, variable='streamflow')
    assert r['new_stations'] == 0
    with open_dataset(str(p)) as ds:
        assert ds['streamflow'].shape[0] == 1  # only A; D must not have been added
        assert list(ds['station_name'].data) == ['Alpha']
