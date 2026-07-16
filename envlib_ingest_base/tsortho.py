"""Build and idempotently update envlib ``ts_ortho`` (station) datasets.

A ts_ortho dataset stores an orthogonal ``(point, time)`` layout: a geometry coordinate of
shapely Points (one per station, order-free), a shared dense hourly ``time`` axis, one primary
data variable named after ``meta.variable``, and a ``station_id`` (envlib's deterministic hash)
+ ``station_name`` attribute variable per station.

Two inputs recur:

- ``stations``: a dict ``{ref: {'lon': float, 'lat': float, 'name': str}}`` (from a pandas
  frame: ``df.to_dict('index')``).
- ``hourly``: a dict ``{ref -> (times, values)}`` of resampled hourly tuples (the output of
  ``resample_*``): ascending interval-start ``datetime64[us]`` times + ``float64`` values.
  An entry that is ``None`` or has ``times.size == 0`` is simply skipped — never test the
  tuple itself with ``len()``/truthiness.

The **merge** is the operational core: each run resamples a recent window and folds it in via a
single contiguous read-modify-write of the affected time block, writing incoming values only
where they are non-NaN (so a station that is briefly offline, or an hour that resamples to NaN,
never clobbers good stored data). New stations append to the point axis; new hours extend the
time axis. This makes a re-run over the same window a no-op — the property the updater relies on.

The cfdb-level functions (``build_local``, ``merge_dataset``) take an open dataset / path and are
unit-tested without any remote. ``build_and_publish`` / ``update_and_publish`` wrap them with the
ebooklet edataset + ``envlib.Catalogue`` publish (exercised against a live remote in Phase 3+).
"""

from __future__ import annotations

import envlib
import numpy as np
import shapely
from cfdb import dtypes, open_dataset, open_edataset

STATION_ID_VAR = 'station_id'
STATION_NAME_VAR = 'station_name'
_HOUR_US = 3_600_000_000


def _hour_index(hours: np.ndarray, t0_us: int) -> np.ndarray:
    """Integer hour offset of datetime64 values from t0 (a dense hourly axis origin)."""
    return ((np.asarray(hours, dtype='datetime64[us]').astype('int64') - t0_us) // _HOUR_US).astype('int64')


def _floor_hour_us(us: int) -> int:
    return (us // _HOUR_US) * _HOUR_US


def _non_empty(hourly: dict) -> dict:
    """Entries that actually carry data, normalized to (datetime64[us] times, float64 values)."""
    out = {}
    for ref, s in hourly.items():
        if s is None:
            continue
        t, v = s
        t = np.asarray(t, dtype='datetime64[us]')
        if t.size:
            out[ref] = (t, np.asarray(v, dtype='float64'))
    return out


def _assemble(stations: dict, hourly: dict, times: np.ndarray) -> np.ndarray:
    """Dense (n_point, n_time) float array; NaN where a station has no value at an hour."""
    t0 = int(times.astype('int64')[0])
    data = np.full((len(stations), times.size), np.nan, dtype='float64')
    for i, ref in enumerate(stations):
        s = hourly.get(ref)
        if s is None:
            continue
        t, v = s
        col = _hour_index(t, t0)
        ok = (col >= 0) & (col < times.size)
        data[i, col[ok]] = v[ok]
    return data


def _points_ids_names(stations: dict):
    points = [shapely.Point(float(d['lon']), float(d['lat'])) for d in stations.values()]
    ids = np.array([envlib.compute_station_id(p) for p in points], dtype=object)
    names = np.array([str(d['name']) for d in stations.values()], dtype=object)
    return points, ids, names


def build_local(
    path,
    meta,
    stations: dict,
    hourly: dict,
    *,
    variable: str,
    units: str,
    precision,
    min_value,
    max_value,
    chunk_shape=None,
    standard_name=None,
    extra_var_attrs=None,
):
    """Create a fresh local ts_ortho cfdb from all stations + their hourly series.

    ``stations``: dict ``{ref: {'lon', 'lat', 'name'}}``.
    ``hourly``: ``{ref -> (times, values)}`` of hourly tuples (interval-start times).
    """
    non_empty = _non_empty(hourly)
    if not non_empty:
        msg = 'no hourly data to build from'
        raise ValueError(msg)
    tmin = _floor_hour_us(min(int(t.astype('int64').min()) for t, _ in non_empty.values()))
    tmax = _floor_hour_us(max(int(t.astype('int64').max()) for t, _ in non_empty.values()))
    times = np.arange(tmin, tmax + 1, _HOUR_US).astype('datetime64[us]')
    data = _assemble(stations, non_empty, times)
    points, ids, names = _points_ids_names(stations)

    if chunk_shape is None:
        tchunk = int(min(times.size, max(1, 2_000_000 // max(1, 2 * len(stations)))))
        chunk_shape = (len(stations), tchunk)

    dt = dtypes.dtype('float32', precision=precision, min_value=min_value, max_value=max_value)
    with open_dataset(str(path), flag='n', dataset_type='ts_ortho') as ds:
        ds.create.coord.point()
        ds['point'].append(points)
        ds.create.coord.time(data=times, dtype=times.dtype)
        ds.create.crs.from_user_input(4326, xy_coord='point')

        dv = ds.create.data_var.generic(variable, ('point', 'time'), dtype=dt, chunk_shape=chunk_shape)
        dv.attrs['units'] = units
        if standard_name is not None:
            dv.attrs['standard_name'] = standard_name
        if extra_var_attrs:
            dv.attrs.update(extra_var_attrs)
        dv[:] = data

        sid = ds.create.data_var.generic(STATION_ID_VAR, ('point',), dtype=dtypes.dtype('str'))
        sid[:] = ids
        snm = ds.create.data_var.generic(STATION_NAME_VAR, ('point',), dtype=dtypes.dtype('str'))
        snm[:] = names

        ds.attrs.update(meta.to_dict())
    return str(path)


def merge_dataset(ds, stations: dict, hourly: dict, *, variable: str):
    """Fold a recent window into an OPEN ts_ortho dataset (cfdb Dataset or EDataset, flag='w').

    Appends new stations (point + station_id + station_name) and new hours, then does a single
    contiguous read-modify-write over the affected time block, keeping existing values where the
    incoming value is NaN. Idempotent for a fixed input window.
    """
    non_empty = _non_empty(hourly)
    if not non_empty:
        return {'new_stations': 0, 'new_hours': 0, 'written_block': 0}

    cur_ids = list(ds[STATION_ID_VAR].data)
    cur_times = np.asarray(ds['time'].data, dtype='datetime64[us]').astype('int64')
    t0 = int(cur_times[0])

    # --- new stations (only those that actually have data this window; `stations` may be the full
    #     discovery dict, but a station with no incoming data must never be added) ---
    active = list(non_empty)
    points, ids, names = _points_ids_names({r: stations[r] for r in active})
    id_to_row = {sid: i for i, sid in enumerate(cur_ids)}
    new_mask = np.array([sid not in id_to_row for sid in ids])
    n_new = int(new_mask.sum())
    if n_new:
        ds['point'].append([p for p, m in zip(points, new_mask, strict=True) if m])
        base = len(cur_ids)
        ds[STATION_ID_VAR][base : base + n_new] = ids[new_mask]
        ds[STATION_NAME_VAR][base : base + n_new] = names[new_mask]
        for k, sid in enumerate(ids[new_mask]):
            id_to_row[sid] = base + k

    # --- new hours (extend the dense hourly axis) ---
    inc_min = _floor_hour_us(min(int(t.astype('int64').min()) for t, _ in non_empty.values()))
    inc_max = _floor_hour_us(max(int(t.astype('int64').max()) for t, _ in non_empty.values()))
    cur_max = int(cur_times[-1])
    n_newhours = 0
    if inc_max > cur_max:
        new_hours = np.arange(cur_max + _HOUR_US, inc_max + 1, _HOUR_US)
        ds['time'].append(new_hours.astype('datetime64[us]'))
        n_newhours = new_hours.size

    # --- write block [w_lo, w_hi] (inclusive) via read-modify-write, non-NaN incoming wins ---
    w_lo = max((inc_min - t0) // _HOUR_US, 0)
    w_hi = (inc_max - t0) // _HOUR_US
    width = w_hi - w_lo + 1
    dv = ds[variable]
    existing = np.asarray(dv[:, w_lo : w_hi + 1].data, dtype='float64')  # (n_point_total, width)
    incoming = np.full_like(existing, np.nan)
    for ref, (t, v) in non_empty.items():
        d = stations[ref]
        sid = envlib.compute_station_id(shapely.Point(float(d['lon']), float(d['lat'])))
        row = id_to_row[sid]
        col = _hour_index(t, t0) - w_lo
        ok = (col >= 0) & (col < width)
        incoming[row, col[ok]] = v[ok]
    merged = np.where(np.isnan(incoming), existing, incoming)
    dv[:, w_lo : w_hi + 1] = merged
    return {'new_stations': n_new, 'new_hours': int(n_newhours), 'written_block': int(width)}


def build_and_publish(cat, path, member_conn, rcg_conn, meta, stations, hourly, *, num_groups, **build_kwargs):
    """First publish: build a local ts_ortho cfdb and publish it to the commons (data then RCG entry)."""
    build_local(path, meta, stations, hourly, **build_kwargs)
    return cat.publish(str(path), member_conn, rcg_conn, num_groups=num_groups)


def update_and_publish(cat, path, member_conn, rcg_conn, stations, hourly, *, variable, num_groups):
    """Incremental update: pull the remote, merge the recent window, then publish (diff + entry refresh).

    ``path`` is a local working cache linked to ``member_conn``; the merge reads only the coords +
    the affected time block, so no full-remote materialization is required.
    """
    with open_edataset(member_conn, str(path), flag='w', num_groups=num_groups) as ds:
        report = merge_dataset(ds, stations, hourly, variable=variable)
    result = cat.publish(str(path), member_conn, rcg_conn, num_groups=num_groups)
    return {'merge': report, 'publish': result}
