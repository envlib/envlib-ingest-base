"""
Tests for the ts_forecast builder.

Every failure mode this builder guards against is SILENT -- each produces a file that validates,
reads back cleanly, and answers wrongly -- so most of these are mutation targets rather than
happy-path coverage. The load-bearing one is the off-grid init refusal: cfdb does not enforce a
datetime coordinate's declared step (fixed in 0.9.7, but this toolkit must not assume the fix is
installed), and envlib does not check axis uniformity, so nothing else in the stack would catch a
mislabelled init.
"""

import logging

import envlib
import numpy as np
import pytest
import shapely
from cfdb import dtypes, open_dataset

from envlib_ingest_base.tsforecast import FRT_COORD, PERIOD_COORD, build_local, merge_run

# The real Ashburton gauges, at the 5-dp coordinates the archive stores. The expected station_ids
# are recorded facts, verified by recomputing envlib.compute_station_id from stored geometry.
TURTONS = {'lon': 171.36288, 'lat': -43.35631, 'name': 'Ashburton River at Turtons Saddle'}
PUDDING = {'lon': 171.53509, 'lat': -43.53661, 'name': 'Pudding Hill Stream at Mt Hutt'}
SOMERS = {'lon': 171.30985, 'lat': -43.61217, 'name': 'South Ashburton River at Mt Somers'}
EXPECTED_IDS = {
    '314412': 'a5c6b23443a9a0300a9030dc',
    '315510': '37f298d61fe9778f2d150022',
    '316310': 'bb91273603d77e9b811cc27b',
}

STATIONS = {'315510': PUDDING, '316310': SOMERS}
LEADS = np.arange(1, 76, dtype='int32')
INIT = np.datetime64('2026-08-24T22:00', 'm')

ENC = {
    'variable': 'precipitation',
    'units': 'mm',
    'precision': 1,
    'min_value': 0,
    'max_value': 1000,
}


def make_meta(**over):
    kwargs = {
        'feature': 'atmosphere',
        'variable': 'precipitation',
        'method': 'forecast',
        'processing_level': 'raw',
        'owner': 'MetService',
        'aggregation_statistic': 'sum',
        'frequency_interval': '1H',
        'utc_offset': '+00:00',
        'spatial_resolution': 'point',
        'product_code': 'MetService_WRF_ECMWF_4km_forecast',
        'version': '1',
        'license': 'CC-BY-4.0',
        'attribution': 'Data from MetService',
    }
    kwargs.update(over)
    return envlib.Metadata(**kwargs)


def make_run(init=INIT, leads=LEADS, refs=('315510', '316310'), fill=None):
    leads = np.asarray(leads)
    values = {}
    for i, r in enumerate(refs):
        values[r] = np.full(leads.size, float(i + 1) if fill is None else fill)
    return {'reference_time': init, 'leads': leads, 'values': values}


def build(path, stations=None, run=None, **over):
    kwargs = {**ENC, 'frt_step_hours': 6, 'lead_units': 'h'}
    kwargs.update(over)
    return build_local(str(path), make_meta(), stations or STATIONS, run or make_run(), **kwargs)


# --------------------------------------------------------------------------------------- build


def test_build_roundtrips_as_ts_forecast(tmp_path):
    p = tmp_path / 'f.cfdb'
    build(p)
    with open_dataset(str(p)) as ds:
        assert ds.dataset_type == 'ts_forecast'
        assert type(ds).__name__ == 'TimeSeriesForecast'
        assert ds[PERIOD_COORD].attrs['units'] == 'h'
        assert ds[FRT_COORD].step == 360


def test_build_passes_envlib_validation(tmp_path):
    """The guard that replaces the catalogue: this archive is never published, so build_local's
    validate call is the ONLY place envlib's checks ever run."""
    p = tmp_path / 'f.cfdb'
    build(p)
    result = envlib.validate_dataset(str(p))
    assert result['state']['dataset_type'] == 'ts_forecast'
    assert result['metadata'].method == 'forecast'


def test_station_ids_match_the_recorded_values(tmp_path):
    """The forecast<->measured join is these hashes colliding at 5 dp with the gauge's."""
    p = tmp_path / 'f.cfdb'
    build(p)
    with open_dataset(str(p)) as ds:
        stored = dict(
            zip(
                [str(r) for r in ds['station_ref'].data],
                [str(v) for v in ds['station_id'].data],
                strict=True,
            )
        )
    assert stored == {r: EXPECTED_IDS[r] for r in STATIONS}


def test_valid_time_range_uses_hours_not_minutes(tmp_path):
    """The unit trap, at the dtype the format actually uses. Written at MINUTE resolution: at
    datetime64[h] the naive `frt + int(lead)` bug passes."""
    p = tmp_path / 'f.cfdb'
    build(p)
    state = envlib.validate_dataset(str(p))['state']
    assert state['time_start'].startswith('2026-08-24T23')  # init + 1 h
    assert state['time_end'].startswith('2026-08-28T01')  # init + 75 h


def test_chunk_point_dim_has_headroom(tmp_path):
    """The point dim of a chunk is locked at creation, so sizing it from the station count means
    every later station lands in a partial chunk forever."""
    p = tmp_path / 'f.cfdb'
    build(p)
    with open_dataset(str(p)) as ds:
        assert ds['precipitation'].chunk_shape[0] > len(STATIONS)


def test_build_refuses_float_leads(tmp_path):
    with pytest.raises(ValueError, match='integer dtype'):
        build(tmp_path / 'f.cfdb', run=make_run(leads=np.array([1.5, 2.5], dtype='float64')))


def test_build_refuses_missing_lead_units(tmp_path):
    with pytest.raises(ValueError, match='lead_units'):
        build(tmp_path / 'f.cfdb', lead_units='')


def test_build_refuses_fractional_cadence(tmp_path):
    with pytest.raises(ValueError, match='whole number of minutes'):
        build(tmp_path / 'f.cfdb', frt_step_hours=1.001)


def test_build_refuses_values_not_paired_with_leads(tmp_path):
    run = make_run()
    run['values']['315510'] = np.ones(3)
    with pytest.raises(ValueError, match='positionally paired'):
        build(tmp_path / 'f.cfdb', run=run)


def test_build_refuses_a_ref_absent_from_stations(tmp_path):
    with pytest.raises(ValueError, match='missing from stations'):
        build(tmp_path / 'f.cfdb', run=make_run(refs=('315510', '999999')))


def test_qc_bounds_reject_out_of_range(tmp_path):
    p = tmp_path / 'f.cfdb'
    run = make_run()
    run['values']['315510'] = np.concatenate([[5000.0], np.full(LEADS.size - 1, 1.0)])
    build(p, run=run)
    with open_dataset(str(p)) as ds:
        row = np.squeeze(np.asarray(ds['precipitation'][:, 0, :].data))
    assert np.isnan(row[0, 0])


# --------------------------------------------------------------------------------------- merge


def merge(path, run, stations=None, **kw):
    with open_dataset(str(path), flag='w') as ds:
        return merge_run(ds, stations or STATIONS, run, variable='precipitation', **kw)


def read_run(path, idx):
    with open_dataset(str(path)) as ds:
        return np.squeeze(np.asarray(ds['precipitation'][:, idx, :].data))


def test_merge_appends_the_next_run(tmp_path):
    p = tmp_path / 'f.cfdb'
    build(p)
    report = merge(p, make_run(init=INIT + np.timedelta64(6, 'h'), fill=4.0))
    assert report['init_index'] == 1
    assert report['autofilled_slots'] == 0
    assert np.allclose(read_run(p, 1), 4.0, atol=0.05)
    assert np.allclose(read_run(p, 0)[0], 1.0, atol=0.05)  # the first run is untouched


def test_merge_refuses_an_offgrid_init(tmp_path):
    """THE guard. cfdb <0.9.7 accepts this silently onto an axis that still declares step=360,
    and envlib never checks axis uniformity -- so nothing else in the stack would catch it."""
    p = tmp_path / 'f.cfdb'
    build(p)
    with pytest.raises(ValueError, match='not on the declared grid'):
        merge(p, make_run(init=INIT + np.timedelta64(7, 'h')))


def test_offgrid_refusal_leaves_the_axis_uniform(tmp_path):
    """The observable damage the refusal prevents, asserted on the axis itself."""
    p = tmp_path / 'f.cfdb'
    build(p)
    merge(p, make_run(init=INIT + np.timedelta64(6, 'h')))
    with pytest.raises(ValueError):
        merge(p, make_run(init=INIT + np.timedelta64(13, 'h')))
    with open_dataset(str(p)) as ds:
        gaps = np.diff(np.asarray(ds[FRT_COORD].data)).astype('timedelta64[m]').astype(int)
    assert set(gaps.tolist()) == {360}, gaps


def test_merge_refuses_a_duplicate_init(tmp_path):
    p = tmp_path / 'f.cfdb'
    build(p)
    with pytest.raises(ValueError, match='immutable'):
        merge(p, make_run(fill=9.0))


def test_overwrite_replaces_the_run_and_warns(tmp_path, caplog):
    p = tmp_path / 'f.cfdb'
    build(p)
    with caplog.at_level(logging.WARNING):
        report = merge(p, make_run(fill=9.0), overwrite=True)
    assert report['overwritten'] is True
    assert np.allclose(read_run(p, 0), 9.0, atol=0.05)
    assert 'OVERWRITING' in caplog.text


def test_missed_run_autofills_then_backfills(tmp_path):
    """The reason forecast_reference_time declares a step at all: skip a run, recover it later."""
    p = tmp_path / 'f.cfdb'
    build(p)
    report = merge(p, make_run(init=INIT + np.timedelta64(12, 'h'), fill=3.0))
    assert report['autofilled_slots'] == 1
    assert report['init_index'] == 2
    assert np.isnan(read_run(p, 1)).all()  # the skipped slot exists and is missing
    back = merge(p, make_run(init=INIT + np.timedelta64(6, 'h'), fill=7.0))
    assert back['init_index'] == 1
    assert np.allclose(read_run(p, 1), 7.0, atol=0.05)  # recovered in place
    assert np.allclose(read_run(p, 2), 3.0, atol=0.05)  # and the later run is undisturbed


def test_merge_refuses_an_init_before_the_axis(tmp_path):
    p = tmp_path / 'f.cfdb'
    build(p)
    with pytest.raises(ValueError, match='predates the axis'):
        merge(p, make_run(init=INIT - np.timedelta64(6, 'h')))


def test_a_station_appearing_later_lands_in_the_right_row(tmp_path):
    """The Lake Stream -> Turtons Saddle switch, which is live with the provider now."""
    p = tmp_path / 'f.cfdb'
    build(p)
    grown = {'315510': PUDDING, '316310': SOMERS, '314412': TURTONS}
    run = make_run(init=INIT + np.timedelta64(6, 'h'), refs=('315510', '316310', '314412'))
    run['values'] = {'315510': np.full(75, 1.0), '316310': np.full(75, 2.0), '314412': np.full(75, 3.0)}
    report = merge(p, run, stations=grown)
    assert report['new_stations'] == 1
    with open_dataset(str(p)) as ds:
        refs = [str(r) for r in ds['station_ref'].data]
        ids = [str(v) for v in ds['station_id'].data]
    assert dict(zip(refs, ids, strict=True)) == EXPECTED_IDS
    row = {r: i for i, r in enumerate(refs)}
    new = read_run(p, 1)
    assert np.allclose(new[row['315510']], 1.0, atol=0.05)
    assert np.allclose(new[row['316310']], 2.0, atol=0.05)
    assert np.allclose(new[row['314412']], 3.0, atol=0.05)
    old = read_run(p, 0)
    assert np.allclose(old[row['315510']], 1.0, atol=0.05)  # prior run intact
    assert np.isnan(old[row['314412']]).all()  # new station has no history


def test_values_follow_the_station_not_the_dict_order(tmp_path):
    """A station retired from the config keeps its row; deriving the row from station_id is what
    stops the next run's values landing on the wrong gauge."""
    p = tmp_path / 'f.cfdb'
    build(p)
    only_somers = {'316310': SOMERS}
    run = {'reference_time': INIT + np.timedelta64(6, 'h'), 'leads': LEADS, 'values': {'316310': np.full(75, 8.0)}}
    merge(p, run, stations=only_somers)
    with open_dataset(str(p)) as ds:
        refs = [str(r) for r in ds['station_ref'].data]
    row = {r: i for i, r in enumerate(refs)}
    new = read_run(p, 1)
    assert np.allclose(new[row['316310']], 8.0, atol=0.05)
    assert np.isnan(new[row['315510']]).all()


def test_short_run_writes_a_prefix_and_leaves_the_tail_missing(tmp_path):
    p = tmp_path / 'f.cfdb'
    build(p)
    short = make_run(init=INIT + np.timedelta64(6, 'h'), leads=LEADS[:70], fill=5.0)
    merge(p, short)
    row = read_run(p, 1)
    assert np.allclose(row[:, :70], 5.0, atol=0.05)
    assert np.isnan(row[:, 70:]).all()


def test_longer_run_extends_the_lead_axis(tmp_path):
    p = tmp_path / 'f.cfdb'
    build(p)
    longer = np.arange(1, 81, dtype='int32')
    report = merge(p, make_run(init=INIT + np.timedelta64(6, 'h'), leads=longer, fill=6.0))
    assert report['new_leads'] == 5
    with open_dataset(str(p)) as ds:
        assert np.asarray(ds[PERIOD_COORD].data)[-1] == 80
    assert np.allclose(read_run(p, 1), 6.0, atol=0.05)
    assert np.isnan(read_run(p, 0)[:, 75:]).all()  # the first run gains a missing tail


def test_incompatible_lead_vector_is_refused(tmp_path):
    p = tmp_path / 'f.cfdb'
    build(p)
    shifted = np.arange(2, 77, dtype='int32')  # same length, different first lead
    with pytest.raises(ValueError, match='not compatible'):
        merge(p, make_run(init=INIT + np.timedelta64(6, 'h'), leads=shifted))


def test_merge_refuses_a_stepless_axis(tmp_path):
    """Without a declared step a missed run is unrecoverable (cfdb cannot insert into the middle
    of a coordinate), so refuse loudly before writing anything rather than building an archive
    that silently cannot be repaired."""
    p = str(tmp_path / 'nostep.cfdb')
    with open_dataset(p, flag='n', dataset_type='ts_forecast') as ds:
        ds.create.coord.point()
        ds['point'].append([shapely.Point(PUDDING['lon'], PUDDING['lat'])])
        ds.create.coord.forecast_reference_time(data=np.array([INIT]), step=None)  # <- no cadence
        ds.create.coord.forecast_period(data=LEADS, step=1)
        ds[PERIOD_COORD].attrs['units'] = 'h'
        ds.create.crs.from_user_input(4326, xy_coord='point')
        ds.create.data_var.generic(
            'precipitation',
            ('point', FRT_COORD, PERIOD_COORD),
            dtype=dtypes.dtype('float32', precision=1, min_value=0, max_value=1000),
            chunk_shape=(8, 1, 75),
        )
        ds.create.data_var.generic('station_id', ('point',), dtype=dtypes.dtype('str'))[:] = np.array(
            [EXPECTED_IDS['315510']], dtype=object
        )
    with open_dataset(p, flag='w') as ds:
        with pytest.raises(ValueError, match='no declared step'):
            merge_run(
                ds,
                {'315510': PUDDING},
                make_run(init=INIT + np.timedelta64(6, 'h'), refs=('315510',)),
                variable='precipitation',
            )


def test_merged_dataset_still_validates(tmp_path):
    p = tmp_path / 'f.cfdb'
    build(p)
    merge(p, make_run(init=INIT + np.timedelta64(6, 'h'), fill=2.0))
    state = envlib.validate_dataset(str(p))['state']
    assert state['time_end'].startswith('2026-08-28T07')  # last init (+6 h) + 75 h
