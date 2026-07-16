"""Build and idempotently update envlib ``ts_ortho`` (station) datasets.

A ts_ortho dataset stores an orthogonal ``(point, time)`` layout: a geometry coordinate of
shapely Points (one per station, order-free), a shared dense hourly ``time`` axis, one primary
data variable named after ``meta.variable``, and a ``station_id`` (envlib's deterministic hash)
+ ``station_name`` attribute variable per station.

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
import pandas as pd
import shapely
from cfdb import dtypes, open_dataset, open_edataset

STATION_ID_VAR = 'station_id'
STATION_NAME_VAR = 'station_name'
_HOUR_NS = 3_600_000_000_000


def _hour_index(hours: np.ndarray, t0_ns: int) -> np.ndarray:
    """Integer hour offset of datetime64 values from t0 (a dense hourly axis origin)."""
    return ((hours.astype('datetime64[ns]').astype('int64') - t0_ns) // _HOUR_NS).astype('int64')


def _assemble(stations: pd.DataFrame, hourly: dict, times: pd.DatetimeIndex) -> np.ndarray:
    """Dense (n_point, n_time) float array; NaN where a station has no value at an hour."""
    t0 = int(times[0].value)
    data = np.full((len(stations), len(times)), np.nan, dtype='float64')
    for i, ref in enumerate(stations.index):
        s = hourly.get(ref)
        if s is None or len(s) == 0:
            continue
        col = _hour_index(s.index.values, t0)
        ok = (col >= 0) & (col < len(times))
        data[i, col[ok]] = np.asarray(s.values, dtype='float64')[ok]
    return data


def _points_ids_names(stations: pd.DataFrame):
    points = [shapely.Point(float(lon), float(lat)) for lon, lat in zip(stations['lon'], stations['lat'], strict=True)]
    ids = np.array([envlib.compute_station_id(p) for p in points], dtype=object)
    names = stations['name'].astype(str).to_numpy(dtype=object)
    return points, ids, names


def build_local(
    path,
    meta,
    stations: pd.DataFrame,
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

    ``stations``: DataFrame indexed by station ref, columns ``lon, lat, name``.
    ``hourly``: ``{ref -> pd.Series}`` of hourly values (interval-start index).
    """
    hours = pd.DatetimeIndex(sorted({t for s in hourly.values() if len(s) for t in [s.index.min(), s.index.max()]}))
    if len(hours) == 0:
        msg = 'no hourly data to build from'
        raise ValueError(msg)
    tmin = min(s.index.min() for s in hourly.values() if len(s)).floor('h')
    tmax = max(s.index.max() for s in hourly.values() if len(s)).floor('h')
    times = pd.date_range(tmin, tmax, freq='1h')
    data = _assemble(stations, hourly, times)
    points, ids, names = _points_ids_names(stations)

    tvals = times.values.astype('datetime64[ns]')
    if chunk_shape is None:
        tchunk = int(min(len(times), max(1, 2_000_000 // max(1, 2 * len(stations)))))
        chunk_shape = (len(stations), tchunk)

    dt = dtypes.dtype('float32', precision=precision, min_value=min_value, max_value=max_value)
    with open_dataset(str(path), flag='n', dataset_type='ts_ortho') as ds:
        ds.create.coord.point()
        ds['point'].append(points)
        ds.create.coord.time(data=tvals, dtype=tvals.dtype)
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


def merge_dataset(ds, stations: pd.DataFrame, hourly: dict, *, variable: str):
    """Fold a recent window into an OPEN ts_ortho dataset (cfdb Dataset or EDataset, flag='w').

    Appends new stations (point + station_id + station_name) and new hours, then does a single
    contiguous read-modify-write over the affected time block, keeping existing values where the
    incoming value is NaN. Idempotent for a fixed input window.
    """
    non_empty = {r: s for r, s in hourly.items() if s is not None and len(s)}
    if not non_empty:
        return {'new_stations': 0, 'new_hours': 0, 'written_block': 0}

    cur_ids = list(ds[STATION_ID_VAR].data)
    cur_times = pd.DatetimeIndex(ds['time'].data)
    t0 = int(cur_times[0].value)

    # --- new stations (only those that actually have data this window; `stations` may be the full
    #     discovery frame, but a station with no incoming data must never be added) ---
    active = list(non_empty)
    points, ids, names = _points_ids_names(stations.loc[active])
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
    inc_min = min(s.index.min() for s in non_empty.values()).floor('h')
    inc_max = max(s.index.max() for s in non_empty.values()).floor('h')
    cur_max = cur_times[-1]
    n_newhours = 0
    if inc_max > cur_max:
        new_hours = pd.date_range(cur_max + pd.Timedelta('1h'), inc_max, freq='1h')
        ds['time'].append(new_hours.values.astype('datetime64[ns]'))
        n_newhours = len(new_hours)

    # --- write block [w_lo, w_hi] (inclusive) via read-modify-write, non-NaN incoming wins ---
    w_lo = int(_hour_index(np.array([inc_min.to_datetime64()]), t0)[0])
    w_hi = int(_hour_index(np.array([inc_max.to_datetime64()]), t0)[0])
    w_lo = max(w_lo, 0)
    width = w_hi - w_lo + 1
    dv = ds[variable]
    existing = np.asarray(dv[:, w_lo : w_hi + 1].data, dtype='float64')  # (n_point_total, width)
    incoming = np.full_like(existing, np.nan)
    for ref, s in non_empty.items():
        sid = envlib.compute_station_id(shapely.Point(float(stations.loc[ref, 'lon']), float(stations.loc[ref, 'lat'])))
        row = id_to_row[sid]
        col = _hour_index(s.index.values, t0) - w_lo
        ok = (col >= 0) & (col < width)
        incoming[row, col[ok]] = np.asarray(s.values, dtype='float64')[ok]
    merged = np.where(np.isnan(incoming), existing, incoming)
    dv[:, w_lo : w_hi + 1] = merged
    return {'new_stations': n_new, 'new_hours': n_newhours, 'written_block': width}


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
