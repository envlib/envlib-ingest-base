"""The telemetry layout (2026-10-08): the default time chunk, the block-major build, rechunk_copy,
the equality gate compare_datasets, and group_bytes.

compare_datasets is the only check before a public dataset is deleted and republished, so every
kind of difference it must catch has a mutant here that it has to report.
"""

import json
from unittest import mock

import booklet
import envlib.vocabularies as vocab_pkg
import numpy as np
import pytest
from cfdb import open_dataset
from ebooklet import DEFAULT_GROUP_BYTES
from envlib.vocabularies import frequency_entry
from rechunkit.main import composite_numbers

from envlib_ingest_base.tests.test_tsortho import BASE, ENC, HOUR, make_meta, stations_dict
from envlib_ingest_base.tsortho import (
    _block_bytes,
    _default_time_chunk,
    _pow2_group_bytes,
    build_and_publish,
    build_local,
    compare_datasets,
    group_bytes_for,
    merge_dataset,
    rechunk_copy,
)

QC = {'quality_code': {'units': '1', 'precision': 0, 'min_value': 0, 'max_value': 1000}}
STNS3 = {'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo'), 'C': (170.0, -44.0, 'Charlie')}

# derived independently of _default_time_chunk (planning run, 2026-10-08): the first highly
# composite number >= 2520 whose duration is a whole number of days
EXPECTED_CHUNK = {
    '1min': 10080, '5min': 10080, '10min': 5040, '15min': 10080, '30min': 5040,
    '1h': 2520, '3h': 2520, '6h': 2520, '12h': 2520, 'day': 2520,
}


def _fixed_codes():
    with open(vocab_pkg.__path__[0] + '/frequency_interval.json') as f:
        doc = json.load(f)
    entries = doc if isinstance(doc, list) else doc.get('entries', doc)
    items = entries if isinstance(entries, list) else list(entries.values())
    return sorted(e['name'] for e in items if e['kind'] == 'fixed')


def _series3(n, start=BASE, *, with_qc=True, nan_every=7):
    """Three stations, n hourly steps, a NaN every nan_every steps in A, C starting late."""
    t = start + HOUR * np.arange(n)
    out = {}
    for k, (ref, off) in enumerate((('A', 0), ('B', 0), ('C', n // 3))):
        tt = t[off:]
        v = np.round(np.sin(np.arange(tt.size) / 5.0) * 10 + 20 + k, 3)
        if ref == 'A':
            v[::nan_every] = np.nan
        ex = {'quality_code': np.full(tt.size, 500.0 + k)} if with_qc else None
        out[ref] = (tt, v, ex) if with_qc else (tt, v)
    return out


def _build(path, n=60, chunk=(1, 12), start=BASE, precision=None, *, with_qc=True):
    enc = dict(ENC)
    if precision is not None:
        enc['precision'] = precision
    build_local(path, make_meta(), stations_dict(STNS3), _series3(n, start, with_qc=with_qc), chunk_shape=chunk,
                ancillary=QC if with_qc else None, **enc)
    return path


def _rechunk(src_path, dst_path, time_chunk):
    with open_dataset(str(src_path)) as src, open_dataset(str(dst_path), flag='n', dataset_type='ts_ortho') as dst:
        rechunk_copy(src, dst, time_chunk=time_chunk)
    return dst_path


def _physical_data_keys(path, names):
    """Data-variable chunk keys in physical (write) order: ebooklet groups follow this order."""
    with booklet.open(str(path), 'r') as b:
        locs = sorted((off, key) for key, _ts, off, _len in b.locations())
    return [key for _off, key in locs if key.split('!')[0] in names]


def _diff(a_path, b_path):
    with open_dataset(str(a_path)) as a, open_dataset(str(b_path)) as b:
        return compare_datasets(a, b, block=12)


# --- the default time chunk ---


def test_default_time_chunk_over_the_whole_cv():
    codes = _fixed_codes()
    assert codes == sorted(EXPECTED_CHUNK), 'a new fixed cadence needs its chunk decided here'
    for code in codes:
        step_us = int(frequency_entry(code)['seconds']) * 1_000_000
        n = _default_time_chunk(step_us)
        assert n == EXPECTED_CHUNK[code], code
        assert n in composite_numbers and n >= 2520 and (n * step_us) % 86_400_000_000 == 0


def test_short_first_build_is_not_clipped(tmp_path):
    p = tmp_path / 'short.cfdb'
    build_local(p, make_meta(), stations_dict(STNS3), _series3(720, with_qc=False), **ENC)
    with open_dataset(str(p)) as ds:
        assert ds['streamflow'].chunk_shape == (1, 2520)
    # the axis grows across the first chunk boundary through the ordinary merge
    win = {'A': (BASE + HOUR * np.arange(2500, 2600), np.arange(100.0))}
    with open_dataset(str(p), flag='w') as ds:
        merge_dataset(ds, stations_dict(STNS3), win, variable='streamflow')
    with open_dataset(str(p)) as ds:
        row = np.asarray(ds['streamflow'][0, 2500:2600].data).ravel()
    np.testing.assert_allclose(row, np.arange(100.0), atol=1e-3)


# --- block-major build ---


def test_build_writes_block_major_with_ancillary_interleaved(tmp_path):
    p = _build(tmp_path / 'b.cfdb', n=36, chunk=(1, 12))
    keys = _physical_data_keys(p, {'streamflow', 'quality_code'})
    expected = [f'{v}!{i}.{b0}' for b0 in (0, 12, 24) for i in range(3) for v in ('streamflow', 'quality_code')]
    assert keys == expected


def test_block_major_build_keeps_values(tmp_path):
    p = _build(tmp_path / 'b.cfdb', n=40, chunk=(1, 12))
    s = _series3(40)
    with open_dataset(str(p)) as ds:
        sf = np.asarray(ds['streamflow'][:].data)
        qc = np.asarray(ds['quality_code'][:].data)
    for i, ref in enumerate(('A', 'B', 'C')):
        t, v, ex = s[ref]
        cols = ((t - BASE) // HOUR).astype(int)
        np.testing.assert_allclose(sf[i, cols], v, atol=1e-3)
        np.testing.assert_array_equal(qc[i, cols], ex['quality_code'])
        assert np.isnan(sf[i, np.setdiff1d(np.arange(40), cols)]).all()


# --- rechunk_copy ---


def test_rechunk_copy_round_trip_with_ancillary_and_a_nonzero_origin(tmp_path):
    src = _build(tmp_path / 'src.cfdb', n=60, chunk=(1, 25))
    with open_dataset(str(src), flag='w') as ds:
        ds['time'].truncate(start=ds['time'].data[7])
    with open_dataset(str(src)) as ds:
        assert ds['time'].origin == 7
    dst = _rechunk(src, tmp_path / 'dst.cfdb', 12)

    assert _diff(src, dst) == []
    with open_dataset(str(src)) as a, open_dataset(str(dst)) as b:
        assert b['streamflow'].chunk_shape == (1, 12) and b['quality_code'].chunk_shape == (1, 12)
        assert b['time'].origin == 0
        np.testing.assert_array_equal(np.asarray(a['streamflow'][:].data), np.asarray(b['streamflow'][:].data))
        assert b.crs == a.crs
    assert _physical_data_keys(dst, {'streamflow', 'quality_code'})[:4] == [
        'streamflow!0.0', 'quality_code!0.0', 'streamflow!1.0', 'quality_code!1.0']


def test_rechunk_copy_refuses_a_non_empty_destination(tmp_path):
    src = _build(tmp_path / 'src.cfdb')
    other = _build(tmp_path / 'other.cfdb')
    with open_dataset(str(src)) as a, open_dataset(str(other), flag='w') as b:
        with pytest.raises(ValueError, match='must be empty'):
            rechunk_copy(a, b, time_chunk=12)


# --- the equality gate ---


@pytest.fixture
def pair(tmp_path):
    """A source with (1, 25) chunks and its (1, 12) rechunked copy: equal by construction."""
    src = _build(tmp_path / 'src.cfdb', n=60, chunk=(1, 25))
    ref = _rechunk(src, tmp_path / 'ref.cfdb', 12)
    return tmp_path, src, ref


def test_gate_reports_an_unmutated_rechunk_as_equal(pair):
    _tmp, src, ref = pair
    assert _diff(src, ref) == []


def _mutant(tmp, src, edit, name='m.cfdb'):
    m = _rechunk(src, tmp / name, 12)
    with open_dataset(str(m), flag='w') as ds:
        edit(ds)
    return m


def _set_value(ds):
    ds['streamflow'][1, 30] = 999.0


def _swap_stations(ds):
    a = np.asarray(ds['streamflow'][0, :].data).copy()
    b = np.asarray(ds['streamflow'][1, :].data).copy()
    ds['streamflow'][0, :] = np.where(np.isnan(b), -1, b)
    ds['streamflow'][1, :] = np.where(np.isnan(a), -1, a)


def _nan_to_zero(ds):
    row = np.asarray(ds['streamflow'][0, :].data)
    j = int(np.flatnonzero(np.isnan(row))[1])
    ds['streamflow'][0, j] = 0.0


def _extra_step(ds):
    ds['time'].append(np.array([ds['time'].data[-1] + np.timedelta64(1, 'h')]))


def _drop_last_block(ds):
    ds['time'].truncate(stop=ds['time'].data[47])


@pytest.mark.parametrize('edit, expect', [
    (_set_value, 'streamflow: values differ'),
    (_swap_stations, 'streamflow: values differ'),
    (_nan_to_zero, 'streamflow: values differ'),
    (lambda ds: ds.attrs.update({'comment': 'changed'}), 'dataset attrs'),
    (lambda ds: ds['streamflow'].attrs.pop('valid_max'), "attrs differ: ['valid_max']"),
    (lambda ds: ds['streamflow'].attrs.update({'units': 'L/s'}), "attrs differ: ['units']"),
    (lambda ds: ds['station_name'].__setitem__(slice(None), np.array(['Alpha', 'Bravo', 'X'], dtype='T')),
     'station_name: values differ'),
    (_extra_step, 'coord time: values differ'),
    (_drop_last_block, 'coord time: values differ'),
    (lambda ds: ds.__delitem__('quality_code'), 'variables:'),
], ids=['value', 'swapped_stations', 'nan_to_zero', 'dataset_attr', 'valid_max', 'units', 'station_name',
        'extra_step', 'dropped_last_block', 'dropped_ancillary'])
def test_gate_catches_each_mutant(pair, edit, expect):
    tmp, src, ref = pair
    m = _mutant(tmp, src, edit)
    diffs = _diff(ref, m)
    assert any(expect in d for d in diffs), diffs


def test_gate_catches_a_time_shift(tmp_path):
    a = _build(tmp_path / 'a.cfdb', n=60, chunk=(1, 12))
    b = _build(tmp_path / 'b.cfdb', n=60, chunk=(1, 12), start=BASE + HOUR)
    diffs = _diff(a, b)
    assert any('coord time: values differ' in d for d in diffs), diffs


def test_gate_catches_a_precision_change(tmp_path):
    a = _build(tmp_path / 'a.cfdb', n=60, chunk=(1, 12))
    b = _build(tmp_path / 'b.cfdb', n=60, chunk=(1, 12), precision=3)
    diffs = _diff(a, b)
    assert any('streamflow: dtype/encoding' in d for d in diffs), diffs


def test_gate_catches_a_missing_crs(pair):
    tmp, src, ref = pair
    m = tmp / 'nocrs.cfdb'
    with open_dataset(str(src)) as a, open_dataset(str(m), flag='n', dataset_type='ts_ortho') as b:
        a.crs = None  # this handle only: the copy then carries no CRS
        rechunk_copy(a, b, time_chunk=12)
    diffs = _diff(ref, m)
    assert any(d.startswith('crs:') for d in diffs), diffs


# --- group_bytes ---


@pytest.mark.parametrize('block, expect', [
    (480_384, 131_072),             # ECan streamflow-like block (RESULTS.md) -> 128 KiB
    (125_000, 65_536),              # ECan precipitation-like -> the 64 KiB floor
    (10, 65_536),
    (2 * 131_072, 131_072),         # exactly two groups
    (2 * 131_072 - 1, 65_536),
    (10**12, DEFAULT_GROUP_BYTES),  # capped at ebooklet's default
])
def test_pow2_group_bytes(block, expect):
    assert _pow2_group_bytes(block) == expect


def test_group_bytes_for_measures_the_stored_chunk_bytes(tmp_path):
    p = _build(tmp_path / 'g.cfdb', n=60, chunk=(1, 12))
    with booklet.open(str(p), 'r') as b:
        stored = sum(len(b[k]) for k in b.keys() if k.startswith('streamflow!'))
    assert group_bytes_for(p, 'streamflow') == _pow2_group_bytes(stored * 12 / 60)


@pytest.mark.parametrize('given', ['auto', None, 1 << 20])
def test_build_and_publish_group_bytes(tmp_path, given):
    calls = []

    class FakeCat:
        def publish(self, _path, _member_conn, _rcg_conn, group_bytes=None):
            calls.append(group_bytes)
            return 'ok'

    p = tmp_path / 'p.cfdb'
    build_and_publish(FakeCat(), p, None, None, make_meta(), stations_dict(STNS3), _series3(60, with_qc=False),
                      group_bytes=given, **ENC)
    want = group_bytes_for(p, 'streamflow') if given == 'auto' else given
    assert calls == [want]


# --- survivors of the code review's mutation pass (round ecan-telemetry-code-1) ---


def test_rechunk_copy_keeps_coordinate_attrs_and_the_gate_compares_them(tmp_path):
    src = _build(tmp_path / 'src.cfdb', n=60, chunk=(1, 25))
    with open_dataset(str(src), flag='w') as ds:
        ds['time'].attrs['comment'] = 'interval-start UTC'
    dst = _rechunk(src, tmp_path / 'dst.cfdb', 12)
    with open_dataset(str(dst)) as b:
        assert dict(b['time'].attrs).get('comment') == 'interval-start UTC'
    assert _diff(src, dst) == []
    with open_dataset(str(dst), flag='w') as ds:
        ds['time'].attrs['comment'] = 'changed'
    assert any('coord time: attrs differ' in d for d in _diff(src, dst))


def test_build_sorts_a_station_whose_times_arrive_unsorted(tmp_path):
    s = _series3(40)
    t, v, ex = s['B']
    order = np.random.default_rng(0).permutation(t.size)
    shuffled = dict(s, B=(t[order], v[order], {k: a[order] for k, a in ex.items()}))
    a = _build(tmp_path / 'a.cfdb', n=40)
    b = tmp_path / 'b.cfdb'
    build_local(b, make_meta(), stations_dict(STNS3), shuffled, chunk_shape=(1, 12), ancillary=QC, **ENC)
    assert _diff(a, b) == []


def test_group_bytes_for_counts_only_the_variable_named(tmp_path):
    """'streamflow' must not also count a variable that merely starts with that name."""
    p = _build(tmp_path / 'g.cfdb', n=60, chunk=(1, 12))
    with open_dataset(str(p), flag='w') as ds:
        x = ds.create.data_var.like('streamflow_extra', ds['streamflow'])
        x[:] = np.asarray(ds['streamflow'][:].data) * 7.0 + 1000.0
    with booklet.open(str(p), 'r') as b:
        stored = sum(len(b[k]) for k in b.keys() if k.startswith('streamflow!'))
    assert _block_bytes(p, 'streamflow') == pytest.approx(stored * 12 / 60)
    assert group_bytes_for(p, 'streamflow') == _pow2_group_bytes(stored * 12 / 60)


def test_build_and_publish_defaults_to_grouped(tmp_path):
    """ECan's first-run path relies on the default, so it is tested without passing group_bytes."""
    calls = []

    class FakeCat:
        def publish(self, _path, _member_conn, _rcg_conn, group_bytes=None):
            calls.append(group_bytes)

    p = tmp_path / 'p.cfdb'
    build_and_publish(FakeCat(), p, None, None, make_meta(), stations_dict(STNS3), _series3(60, with_qc=False), **ENC)
    assert calls == [group_bytes_for(p, 'streamflow')] and calls[0] >= 65536


def test_gate_compares_every_coordinate_dtype_field(pair):
    """Coordinates go through the same _dtype_fields comparison as variables, not str(dtype): make
    every dtype object differ in a hidden field and the time coordinate must be reported."""
    _tmp, src, ref = pair
    with mock.patch('envlib_ingest_base.tsortho._dtype_fields',
                    side_effect=lambda dt: {'name': str(getattr(dt, 'name', None)), 'hidden': str(id(dt))}):
        diffs = _diff(ref, src)
    assert any('coord time: dtype/step' in d for d in diffs), diffs
