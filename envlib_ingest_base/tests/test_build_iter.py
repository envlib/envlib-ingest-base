"""The iterator form of build_local, the decoded dtype, the time-label attrs and the moved-station
warning (toolkit 0.7.0).

The iterator form exists so a build never holds the whole dataset in memory. It must produce the
same dataset as the dict form, minus the chunks that hold nothing, and it must refuse every input
the dict form would have caught by looking at all the data at once.
"""

import logging
import os

import booklet
import envlib
import numpy as np
import pytest
from cfdb import open_dataset

from envlib_ingest_base.tests.test_tsortho import BASE, ENC, HOUR, Q15, make_meta, stations_dict
from envlib_ingest_base.tsortho import build_and_publish, build_local, compare_datasets, merge_dataset

QC = {'quality_code': {'units': '1', 'precision': 0, 'min_value': 0, 'max_value': 1000}}
STNS = {'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo'), 'C': (170.0, -44.0, 'Charlie')}
CHUNK = (1, 12)  # small chunks so a short axis spans several blocks


def _sparse_series():
    """A long axis (300 h): A dense, B only two short bursts far apart, C one code-only slot."""
    ta = BASE + HOUR * np.arange(300)
    va = np.linspace(1.0, 300.0, 300)
    tb = np.concatenate([BASE + HOUR * np.arange(5, 9), BASE + HOUR * np.arange(250, 253)])
    vb = np.arange(tb.size, dtype='float64') + 10.0
    tc = BASE + HOUR * np.array([100, 101])
    vc = np.array([np.nan, 7.0])  # 100 h is a code-only slot: no value, a grade
    return {
        'A': (ta, va, {'quality_code': np.full(ta.size, 600.0)}),
        'B': (tb, vb, {'quality_code': np.full(tb.size, 500.0)}),
        'C': (tc, vc, {'quality_code': np.array([100.0, 600.0])}),
    }


def _gen(series, order=None):
    for ref in order or list(series):
        t, v, ex = series[ref]
        yield ref, t, v, ex


def _range(series):
    lo = min(t.min() for t, _v, _e in series.values())
    hi = max(t.max() for t, _v, _e in series.values())
    return lo, hi


def _data_keys(path, names=('streamflow', 'quality_code')):
    with booklet.open(str(path), 'r') as b:
        return {k for k in b.keys() if k.split('!')[0] in names}


def _build_both(tmp_path, series, *, order=None):
    stns = stations_dict(STNS)
    pd_, pi = tmp_path / 'dict.cfdb', tmp_path / 'iter.cfdb'
    build_local(pd_, make_meta(), stns, series, **ENC, ancillary=QC, chunk_shape=CHUNK)
    build_local(pi, make_meta(), stns, _gen(series, order), **ENC, ancillary=QC, chunk_shape=CHUNK,
                time_range=_range(series))
    return pd_, pi


# --- equivalence with the dict form ---


def test_iterable_equals_dict_form(tmp_path):
    pd_, pi = _build_both(tmp_path, _sparse_series())
    with open_dataset(str(pd_)) as a, open_dataset(str(pi)) as b:
        assert compare_datasets(a, b, block=12) == []


def test_iterable_rows_follow_stations_not_yield_order(tmp_path):
    pd_, pi = _build_both(tmp_path, _sparse_series(), order=['C', 'A', 'B'])
    with open_dataset(str(pd_)) as a, open_dataset(str(pi)) as b:
        assert compare_datasets(a, b, block=12) == []
        assert list(b['station_ref'].data) == ['A', 'B', 'C']


def test_iterable_skips_exactly_the_empty_chunks(tmp_path):
    pd_, pi = _build_both(tmp_path, _sparse_series())
    kd, ki = _data_keys(pd_), _data_keys(pi)
    assert ki < kd  # strictly fewer, never a key the dict form lacks
    # every key the iterator skipped is a chunk that holds only NaN in the dict-form file
    with open_dataset(str(pd_)) as ds:
        for key in kd - ki:
            name, idx = key.split('!')
            row, col = (int(x) for x in idx.split('.'))
            seg = np.asarray(ds[name][row, col : col + CHUNK[1]].data, dtype='float64')
            assert np.isnan(seg).all(), key
    # and every chunk the iterator kept holds something
    with open_dataset(str(pi)) as ds:
        for key in ki:
            name, idx = key.split('!')
            row, col = (int(x) for x in idx.split('.'))
            other = 'quality_code' if name == 'streamflow' else 'streamflow'
            seg = np.asarray(ds[name][row, col : col + CHUNK[1]].data, dtype='float64')
            seg2 = np.asarray(ds[other][row, col : col + CHUNK[1]].data, dtype='float64')
            assert not (np.isnan(seg).all() and np.isnan(seg2).all()), key


def test_iterable_keeps_code_only_slot(tmp_path):
    _pd, pi = _build_both(tmp_path, _sparse_series())
    with open_dataset(str(pi)) as ds:
        v = np.asarray(ds['streamflow'][2, 100:102].data, dtype='float64').ravel()
        c = np.asarray(ds['quality_code'][2, 100:102].data, dtype='float64').ravel()
    assert np.isnan(v[0]) and c[0] == 100.0
    assert v[1] == pytest.approx(7.0) and c[1] == 600.0


def test_iterable_keeps_a_code_only_slot_alone_in_its_chunk(tmp_path):
    # a grade with no value anywhere in its chunk must still be written (the skip rule looks at
    # every plane, not just the values)
    s = _sparse_series()
    s['C'] = (BASE + HOUR * np.array([30, 101]), np.array([np.nan, 7.0]), {'quality_code': np.array([100.0, 600.0])})
    pd_, pi = _build_both(tmp_path, s)
    with open_dataset(str(pi)) as ds:
        c = np.asarray(ds['quality_code'][2, 30:31].data, dtype='float64').ravel()
        v = np.asarray(ds['streamflow'][2, 24:36].data, dtype='float64').ravel()
    assert c[0] == 100.0 and np.isnan(v).all()  # chunk 2 (steps 24-35) holds only that grade
    with open_dataset(str(pd_)) as a, open_dataset(str(pi)) as b:
        assert compare_datasets(a, b, block=12) == []


def test_iterable_refuses_duplicate_stamps(tmp_path):
    s = _sparse_series()
    t, v, ex = s['B']
    codes = np.concatenate([ex['quality_code'], ex['quality_code'][:1]])
    s['B'] = (np.concatenate([t, t[:1]]), np.concatenate([v, v[:1]]), {'quality_code': codes})
    with pytest.raises(ValueError, match='duplicate timestamps'):
        _iter_build(tmp_path, s)


def test_iterable_qc_rejected_value_takes_its_code(tmp_path):
    s = _sparse_series()
    t, v, ex = s['A']
    v = v.copy()
    v[3] = -5.0  # below min_value: rejected, and its grade with it (as in the dict form)
    s['A'] = (t, v, ex)
    pd_, pi = _build_both(tmp_path, s)
    with open_dataset(str(pd_)) as a, open_dataset(str(pi)) as b:
        assert compare_datasets(a, b, block=12) == []
        assert np.isnan(np.asarray(b['quality_code'][0, 3].data, dtype='float64').ravel()[0])


# --- refusals ---


def _iter_build(tmp_path, series, **kw):
    stns = kw.pop('stations', stations_dict(STNS))
    if 'time_range' not in kw:
        kw['time_range'] = _range(series)
    p = tmp_path / 'x.cfdb'
    build_local(p, make_meta(), stns, kw.pop('gen', _gen(series)), **ENC, ancillary=QC, chunk_shape=CHUNK, **kw)
    return p


def test_iterable_needs_time_range(tmp_path):
    with pytest.raises(ValueError, match='needs time_range'):
        _iter_build(tmp_path, _sparse_series(), time_range=None)


def test_dict_series_refuses_time_range(tmp_path):
    s = _sparse_series()
    with pytest.raises(ValueError, match='time_range is for an iterable'):
        build_local(tmp_path / 'd.cfdb', make_meta(), stations_dict(STNS), s, **ENC, ancillary=QC,
                    time_range=_range(s))


def test_iterable_refuses_unknown_ref(tmp_path):
    s = _sparse_series()
    s['Z'] = s.pop('C')
    with pytest.raises(ValueError, match="'Z', which is not in stations"):
        _iter_build(tmp_path, s)


def test_iterable_refuses_duplicate_ref(tmp_path):
    s = _sparse_series()
    with pytest.raises(ValueError, match="'A' twice"):
        _iter_build(tmp_path, s, gen=_gen(s, ['A', 'B', 'C', 'A']))


def test_iterable_refuses_station_never_yielded(tmp_path):
    s = _sparse_series()
    with pytest.raises(ValueError, match='never yielded'):
        _iter_build(tmp_path, s, gen=_gen(s, ['A', 'B']))


def test_iterable_refuses_empty_station(tmp_path):
    s = _sparse_series()
    s['C'] = (np.array([], 'datetime64[us]'), np.array([]), {'quality_code': np.array([])})
    with pytest.raises(ValueError, match="'C': yielded no data"):
        _iter_build(tmp_path, s, time_range=(BASE, BASE + 299 * HOUR))


def test_iterable_refuses_station_that_qc_empties(tmp_path):
    s = _sparse_series()
    s['C'] = (s['C'][0], np.array([-1.0, -2.0]), {'quality_code': np.array([600.0, 600.0])})
    with pytest.raises(ValueError, match="'C': nothing survives QC"):
        _iter_build(tmp_path, s)


def test_iterable_refuses_stamp_outside_time_range(tmp_path):
    s = _sparse_series()
    with pytest.raises(ValueError, match='falls outside time_range'):
        _iter_build(tmp_path, s, time_range=(BASE, BASE + 200 * HOUR))


@pytest.mark.parametrize('end', ['start', 'end'])
def test_iterable_refuses_time_range_wider_than_data(tmp_path, end):
    s = _sparse_series()
    lo, hi = _range(s)
    tr = (lo - HOUR, hi) if end == 'start' else (lo, hi + HOUR)
    with pytest.raises(ValueError, match='does not match the data'):
        _iter_build(tmp_path, s, time_range=tr)


def test_iterable_refuses_misaligned_time_range(tmp_path):
    s = _sparse_series()
    lo, hi = _range(s)
    with pytest.raises(ValueError, match='not aligned'):
        _iter_build(tmp_path, s, time_range=(lo + np.timedelta64(1, 'm'), hi))


def test_iterable_refuses_point_chunk_above_one(tmp_path):
    s = _sparse_series()
    with pytest.raises(ValueError, match='point chunk of 1'):
        build_local(tmp_path / 'x.cfdb', make_meta(), stations_dict(STNS), _gen(s), **ENC, ancillary=QC,
                    chunk_shape=(2, 12), time_range=_range(s))


def test_iterable_refuses_three_tuples(tmp_path):
    s = _sparse_series()
    with pytest.raises(ValueError, match=r'yields \(ref, times, values, extras\)'):
        _iter_build(tmp_path, s, gen=((r, t, v) for r, (t, v, _e) in s.items()))


def test_iterable_refuses_misaligned_station_stamp(tmp_path):
    s = _sparse_series()
    t, v, ex = s['B']
    s['B'] = (t + np.timedelta64(1, 'm'), v, ex)  # every B stamp one minute off the hourly grid
    lo, hi = _range(_sparse_series())
    with pytest.raises(ValueError, match='not aligned'):
        _iter_build(tmp_path, s, time_range=(lo, hi))


def test_iterable_refuses_undeclared_ancillary(tmp_path):
    s = _sparse_series()
    t, v, ex = s['B']
    s['B'] = (t, v, {**ex, 'unit_flag': np.zeros(t.size)})
    with pytest.raises(ValueError, match='not declared by this dataset'):
        _iter_build(tmp_path, s)


def test_iterable_failure_leaves_no_file(tmp_path):
    s = _sparse_series()
    with pytest.raises(ValueError):
        _iter_build(tmp_path, s, gen=_gen(s, ['A', 'B']))
    assert not os.listdir(tmp_path)  # neither the target nor the .partial staging file


def test_iterable_failed_rebuild_keeps_the_previous_file(tmp_path):
    """Staging exists for this: a rebuild that fails must not destroy the good file at path."""
    s = _sparse_series()
    p = _iter_build(tmp_path, s)
    before = p.read_bytes()
    with pytest.raises(ValueError, match='never yielded'):
        _iter_build(tmp_path, s, gen=_gen(s, ['A', 'B']))
    assert p.read_bytes() == before
    assert sorted(os.listdir(tmp_path)) == ['x.cfdb']


def test_iterable_is_consumed_lazily(tmp_path):
    """The point of the form: the producer's series are not all alive at once."""
    s = _sparse_series()
    live, peak = [], [0]

    class Tracked(np.ndarray):
        def __del__(self):
            live.remove(id(self)) if id(self) in live else None

    def gen():
        for ref in s:
            t, v, ex = s[ref]
            vv = np.array(v).view(Tracked)
            live.append(id(vv))
            peak[0] = max(peak[0], len(live))
            yield ref, t, vv, ex
            del vv

    _iter_build(tmp_path, s, gen=gen())
    assert peak[0] == 1


# --- decoded dtype ---


def test_float64_decode_is_exact_where_float32_is_not(tmp_path):
    vals = np.array([2345.6789, 16837.1234, 999.9999, 0.0001])
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    series = {'A': (BASE + HOUR * np.arange(4), vals)}
    out = {}
    for dtn in ('float32', 'float64'):
        p = tmp_path / f'{dtn}.cfdb'
        build_local(p, make_meta(), stns, series, **ENC, decoded_dtype=dtn)
        with open_dataset(str(p)) as ds:
            out[dtn] = np.asarray(ds['streamflow'][0, :].data, dtype='float64').ravel()
    assert np.array_equal(np.round(out['float64'] * 1e4).astype(np.int64), np.round(vals * 1e4).astype(np.int64))
    assert np.abs(out['float64'] - vals).max() < 1e-9
    # the reason the option exists: float32 cannot hold the 4th decimal above ~1000
    assert np.abs(out['float32'] - vals).max() > 1e-5


def test_decoded_dtype_refuses_other_types(tmp_path):
    with pytest.raises(ValueError, match='decoded_dtype'):
        build_local(tmp_path / 'x.cfdb', make_meta(), stations_dict({'A': (172.5, -43.5, 'A')}),
                    {'A': (BASE + HOUR * np.arange(2), np.array([1.0, 2.0]))}, **ENC, decoded_dtype='int32')


# --- time-label attrs ---


def _point_meta():
    m = make_meta('15min').to_dict()
    kw = {k.removeprefix('envlib_'): v for k, v in m.items() if k.startswith('envlib_') and 'dataset' not in k}
    kw.update(aggregation_statistic='point', license='CC-BY-4.0', attribution='Environment Canterbury')
    return envlib.Metadata(**kw)


def test_time_label_attrs_mean_and_point(tmp_path):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p1 = tmp_path / 'h.cfdb'
    build_local(p1, make_meta(), stns, {'A': (BASE + HOUR * np.arange(3), np.array([1.0, 2.0, 3.0]))}, **ENC)
    p2 = tmp_path / 'q.cfdb'
    t15 = BASE + Q15 * np.arange(3)
    build_local(p2, _point_meta(), stns, ((r, t15, np.array([1.0, 2.0, 3.0]), {}) for r in ['A']), **ENC,
                time_range=(t15[0], t15[-1]))
    with open_dataset(str(p1)) as a, open_dataset(str(p2)) as b:
        ta, tb = dict(a['time'].attrs.data), dict(b['time'].attrs.data)
    assert (ta['time_label'], ta['interval_minutes']) == ('interval_start', 60)
    assert tb['time_label'] == 'instant' and 'interval_minutes' not in tb


# --- build_and_publish with an iterable ---


def _physical_order(path):
    with booklet.open(str(path), 'r') as b:
        locs = sorted((off, key) for key, _ts, off, _len in b.locations())
    return [k for _o, k in locs if k.split('!')[0] == 'streamflow']


@pytest.mark.parametrize('group_bytes', [None, 1 << 20])
def test_build_and_publish_iterable(tmp_path, group_bytes):
    calls = []

    class FakeCat:
        def publish(self, path, _m, _r, group_bytes=None):
            calls.append((path, group_bytes))
            return 'ok'

    s = _sparse_series()
    ref = tmp_path / 'ref.cfdb'
    build_local(ref, make_meta(), stations_dict(STNS), s, **ENC, ancillary=QC, chunk_shape=CHUNK)
    p = tmp_path / 'pub.cfdb'
    build_and_publish(FakeCat(), p, None, None, make_meta(), stations_dict(STNS), _gen(s), group_bytes=group_bytes,
                      **ENC, ancillary=QC, chunk_shape=CHUNK, time_range=_range(s))
    assert calls == [(str(p), group_bytes)]
    assert sorted(os.listdir(tmp_path)) == ['pub.cfdb', 'ref.cfdb']  # no staging file left behind
    with open_dataset(str(ref)) as a, open_dataset(str(p)) as b:
        assert compare_datasets(a, b, block=12) == []
    order = _physical_order(p)
    cols = [int(k.split('!')[1].split('.')[1]) for k in order]
    if group_bytes is not None:  # block-major: the column index never decreases in write order
        assert cols == sorted(cols)
    else:  # per-key: published as built, station-major
        assert cols != sorted(cols)


# --- merge: a station whose coordinates changed ---


def test_merge_warns_when_a_known_ref_gets_a_new_row(tmp_path, caplog):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo')})
    p = tmp_path / 'm.cfdb'
    build_local(p, make_meta(), stns, {r: (BASE + HOUR * np.arange(3), np.ones(3)) for r in stns}, **ENC)
    moved = stations_dict({'A': (172.6, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo')})
    win = {r: (BASE + HOUR * np.arange(3, 5), np.ones(2)) for r in moved}
    with caplog.at_level(logging.WARNING), open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, moved, win, variable='streamflow')
    assert r['new_stations'] == 1
    assert r['new_rows_for_existing_refs'] == 1
    assert any("station_ref 'A' gains a new row" in m for m in caplog.messages)


def test_merge_new_station_is_not_a_moved_one(tmp_path, caplog):
    stns = stations_dict({'A': (172.5, -43.5, 'Alpha')})
    p = tmp_path / 'm.cfdb'
    build_local(p, make_meta(), stns, {'A': (BASE + HOUR * np.arange(3), np.ones(3))}, **ENC)
    more = stations_dict({'A': (172.5, -43.5, 'Alpha'), 'B': (171.9, -43.1, 'Bravo')})
    with caplog.at_level(logging.WARNING), open_dataset(str(p), flag='w') as ds:
        r = merge_dataset(ds, more, {'B': (BASE + HOUR * np.arange(2), np.ones(2))}, variable='streamflow')
    assert r['new_stations'] == 1
    assert r['new_rows_for_existing_refs'] == 0
    assert not any('gains a new row' in m for m in caplog.messages)
