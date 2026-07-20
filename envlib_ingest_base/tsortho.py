"""Build and idempotently update envlib ``ts_ortho`` (station) datasets.

A ts_ortho dataset stores an orthogonal ``(point, time)`` layout: a geometry coordinate of
shapely Points (one per station, order-free), a shared **dense fixed-step time axis at the
cadence declared by ``meta.frequency_interval``**, one primary data variable named after
``meta.variable``, and per-station metadata as auxiliary ``(point,)`` string variables —
the established nomenclature for station data in envlib/cfdb ts_ortho datasets:
``station_id`` (envlib's deterministic geometry hash), ``station_name``, and
``station_ref`` (the SOURCE's native identifier = the stations-dict key, the stable join
key back to the provider's records). Future well-known station fields (e.g. ``altitude``
when a source provides it) join the same pattern as ``(point,)`` variables with CF attrs.

**The envlib metadata is the single source of truth for the cadence** — there is no separate
freq parameter. ``build_local`` reads ``meta.frequency_interval`` (a closed envlib controlled
vocabulary; only *fixed* codes are usable — ``month``/``year``/``None`` raise, a dense
fixed-step axis needs a fixed duration), and ``merge_dataset`` reads it back from the stored
dataset's own attrs, cross-checked against the actual axis. The time coordinate is stored in
the cadence's **natural datetime64 unit** (``day`` -> ``datetime64[D]``, the hourly family ->
``[h]``, the minute family -> ``[m]``) with an explicit step, so the axis itself tells the
reader the data's precision. Reader note: string ``.loc``/``truncate`` queries are truncated
by numpy to the axis unit (e.g. ``'...T06:00'`` on a daily axis widens back to midnight).

Two inputs recur:

- ``stations``: a dict ``{ref: {'lon': float, 'lat': float, 'name': str}}`` (from a pandas
  frame: ``df.to_dict('index')``).
- ``series``: a dict ``{ref -> (times, values)}`` of resampled tuples (the output of
  ``resample_*`` at the SAME freq the metadata declares): ascending interval-start
  ``datetime64`` times + ``float64`` values. An entry that is ``None`` or has
  ``times.size == 0`` is simply skipped — never test the tuple itself with ``len()``/truthiness.

Two guards enforce a strict **epoch-anchored phase contract** (phase-anchored binning — e.g. a
local-midnight or 9am-rain-day daily product — is deliberately unsupported until the resampler
grows an ``origin`` feature; see the OPEN_WORK follow-up):

- *alignment*: every incoming timestamp must be an exact multiple of the declared step
  (the resampler's labels are, by construction) — this also turns cfdb's silent truncation of
  misaligned appends into a loud producer-side error;
- *metadata*: a fixed cadence whose (reduced) ``utc_offset`` is not ``+00:00`` *declares*
  phase-anchored binning that epoch-anchored data does not have, and is rejected so the
  identity-hashed metadata can never lie about phase.

Known one-directional limit: data resampled *coarser* than declared (daily labels are valid
hour multiples) builds a sparse-but-aligned axis the guards cannot detect.

The **merge** is the operational core: each run resamples a recent window and folds it in via a
single contiguous read-modify-write of the affected time block, writing incoming values only
where they are non-NaN (so a station that is briefly offline, or an interval that resamples to
NaN, never clobbers good stored data). New stations append to the point axis; new steps extend
the time axis (in the coord's own stored dtype). A re-run over the same window is a no-op.

The cfdb-level functions (``build_local``, ``merge_dataset``) take an open dataset / path and
are unit-tested without any remote. ``build_and_publish`` / ``update_and_publish`` wrap them
with the ebooklet edataset + ``envlib.Catalogue`` publish.
"""

from __future__ import annotations

import logging

import cfdb
import envlib
import numpy as np
import shapely
from cfdb import dtypes, open_dataset, open_edataset
from envlib.vocabularies import frequency_entry

logger = logging.getLogger(__name__)

_MIN_CFDB = (0, 9, 4)


def _require_fixed_cfdb() -> None:
    """cfdb < 0.9.4 fabricates values when its packed encoder meets NaN/out-of-range data —
    and cfdb's own partial-chunk read-modify-write re-encodes STORED holes on every merge,
    so no toolkit-side substitution can make older versions safe. Refuse to write."""
    ver = tuple(int(x) for x in cfdb.__version__.split('.')[:3])
    if ver < _MIN_CFDB:
        msg = (
            f'cfdb >= 0.9.4 required (installed: {cfdb.__version__}): older versions fabricate '
            f'values when encoding NaN/out-of-range data through packed dtypes (cfdb changelog 0.9.4)'
        )
        raise RuntimeError(msg)


STATION_ID_VAR = 'station_id'
STATION_NAME_VAR = 'station_name'
STATION_REF_VAR = 'station_ref'
_DEFAULT_TIME_CHUNK = 25_000  # steps; see the chunk_shape default rationale in build_local
# natural-unit ladder: the largest unit that divides the step becomes the stored coord unit
_UNIT_US = (('D', 86_400_000_000), ('h', 3_600_000_000), ('m', 60_000_000), ('s', 1_000_000))


def _freq_step(code) -> tuple[int, str]:
    """An envlib ``frequency_interval`` code -> ``(step in us, natural datetime64 unit)``.

    Raises ValueError for ``None`` (irregular), calendar codes (``month``/``year``), or
    unknown codes.
    """
    if code is None:
        msg = 'frequency_interval is None (irregular cadence) — a ts_ortho dense axis needs a fixed step'
        raise ValueError(msg)
    try:
        entry = frequency_entry(code)
    except (TypeError, ValueError) as e:
        msg = f'unknown frequency_interval {code!r}: {e}'
        raise ValueError(msg) from e
    if entry['kind'] != 'fixed':
        msg = f'frequency_interval {code!r} is calendar-based — a dense fixed-step axis needs a fixed duration'
        raise ValueError(msg)
    step_us = int(entry['seconds']) * 1_000_000
    for unit, unit_us in _UNIT_US:
        if step_us % unit_us == 0:
            return step_us, unit
    return step_us, 'us'  # unreachable with the current CV (whole-second codes only)


def _floor_step(us: int, step_us: int) -> int:
    return (us // step_us) * step_us


def _step_index(times, t0_us: int, step_us: int) -> np.ndarray:
    """Integer step offset of datetime64 values from t0 (a dense fixed-step axis origin)."""
    return ((np.asarray(times, dtype='datetime64[us]').astype('int64') - t0_us) // step_us).astype('int64')


def _check_aligned(non_empty: dict, step_us: int, code) -> None:
    """Every incoming timestamp must sit on the epoch-anchored step grid (module docstring)."""
    for ref, (t, _v) in non_empty.items():
        res = t.astype('int64') % step_us
        bad = np.nonzero(res)[0]
        if bad.size:
            msg = (
                f'station {ref!r}: {bad.size} timestamp(s) not aligned to frequency_interval {code!r} '
                f'(first: {t[bad[0]]}, residue {int(res[bad[0]])} us) — the resample freq must match '
                f'the declared frequency_interval'
            )
            raise ValueError(msg)


def _check_phase(utc_offset, code) -> None:
    """Reject metadata that declares phase-anchored binning (module docstring)."""
    if utc_offset not in (None, '+00:00'):
        msg = (
            f'utc_offset {utc_offset!r} with fixed frequency_interval {code!r} declares phase-anchored '
            f'binning (e.g. local-midnight days) that the epoch-anchored resampler cannot produce — '
            f'not yet supported (needs a resampler origin feature)'
        )
        raise ValueError(msg)


def _check_refs(non_empty: dict, stations: dict) -> None:
    """A series ref absent from stations is a broken premise, not an operational hiccup:
    extraction is station-list-driven, and without metadata there is no geometry, no
    station_id, and no row to put the data in. Raise — never silently drop or skip."""
    missing = sorted(set(non_empty) - set(stations))
    if missing:
        msg = f'series refs missing from stations: {missing} — station metadata is required'
        raise ValueError(msg)


def _encodable_range(dt) -> tuple:
    """The decodable value interval of a packed dtype; (None, None) when unpacked."""
    enc = getattr(dt, 'dtype_encoded', None)
    if enc is None or dt.precision is None or dt.offset is None:
        return None, None
    factor = 10**dt.precision
    info = np.iinfo(enc)
    return (1 / factor) + dt.offset, (info.max / factor) + dt.offset


def _qc_filter(non_empty: dict, lo, hi) -> tuple[dict, int]:
    """min/max QC (ruling 2026-07-17): the DECLARED min/max double as plausibility bounds,
    so values outside them — including ±inf — become NaN (missing). Runs BEFORE the merge
    combine, so a rejected incoming value can never displace a stored valid one."""
    if lo is None:
        return non_empty, 0
    out, n = {}, 0
    for ref, (t, v) in non_empty.items():
        bad = ~np.isnan(v) & ((v < lo) | (v > hi))
        nbad = int(bad.sum())
        vv = np.where(bad, np.nan, v) if nbad else v
        n += nbad
        out[ref] = (t, vv)
    return out, n


def _qc_bounds(dv) -> tuple:
    """The QC bounds for an existing data var: the DECLARED ``valid_min``/``valid_max`` attrs
    (written at build), falling back to the encodable range for datasets built before the
    attrs existed (wider — encodability only)."""
    attrs = dv.attrs.data
    if 'valid_min' in attrs and 'valid_max' in attrs:
        return float(attrs['valid_min']), float(attrs['valid_max'])
    return _encodable_range(dv.dtype)


def _nan_safe(data: np.ndarray, dt) -> np.ndarray:
    """Substitute NaN with the dtype's ``offset`` value, which encodes to the reserved
    fillvalue exactly (decoding back to NaN). NOTE: this shields only the toolkit's own
    writes — cfdb's partial-chunk read-modify-write re-encodes STORED holes internally, so
    this is belt-and-braces on top of ``_require_fixed_cfdb``, NOT a substitute for it.
    A no-op for unpacked dtypes."""
    off = getattr(dt, 'offset', None)
    if off is None:
        return data
    return np.where(np.isnan(data), float(off), data)


def _non_empty(series: dict) -> dict:
    """Entries that actually carry data, normalized to (datetime64[us] times, float64 values)."""
    out = {}
    for ref, s in series.items():
        if s is None:
            continue
        t, v = s
        t = np.asarray(t, dtype='datetime64[us]')
        if t.size:
            out[ref] = (t, np.asarray(v, dtype='float64'))
    return out


def _assemble(stations: dict, series: dict, t0_us: int, n_times: int, step_us: int) -> np.ndarray:
    """Dense (n_point, n_time) float array; NaN where a station has no value at a step."""
    data = np.full((len(stations), n_times), np.nan, dtype='float64')
    for i, ref in enumerate(stations):
        s = series.get(ref)
        if s is None:
            continue
        t, v = s
        col = _step_index(t, t0_us, step_us)
        ok = (col >= 0) & (col < n_times)
        data[i, col[ok]] = v[ok]
    return data


def _points_ids_names(stations: dict):
    points = [shapely.Point(float(d['lon']), float(d['lat'])) for d in stations.values()]
    ids = np.array([envlib.compute_station_id(p) for p in points], dtype=object)
    names = np.array([str(d['name']) for d in stations.values()], dtype=object)
    refs = np.array([str(r) for r in stations], dtype=object)
    return points, ids, names, refs


def build_local(
    path,
    meta,
    stations: dict,
    series: dict,
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
    """Create a fresh local ts_ortho cfdb from all stations + their resampled series.

    The axis cadence comes from ``meta.frequency_interval`` (fixed CV codes only); the time
    coordinate is stored in the cadence's natural unit with an explicit step.
    ``stations``: dict ``{ref: {'lon', 'lat', 'name'}}``; ``series``: ``{ref -> (times, values)}``.
    """
    _require_fixed_cfdb()
    step_us, unit = _freq_step(meta.frequency_interval)
    attrs = meta.to_dict()
    _check_phase(attrs.get('envlib_utc_offset'), meta.frequency_interval)
    non_empty = _non_empty(series)
    if not non_empty:
        msg = 'no series data to build from'
        raise ValueError(msg)
    _check_refs(non_empty, stations)
    _check_aligned(non_empty, step_us, meta.frequency_interval)

    dt = dtypes.dtype('float32', precision=precision, min_value=min_value, max_value=max_value)
    non_empty, n_qc = _qc_filter(non_empty, float(min_value), float(max_value))
    if n_qc:
        logger.warning('build %s: %d value(s) outside [%s, %s] set to NaN (QC)', variable, n_qc, min_value, max_value)

    tmin = _floor_step(min(int(t.astype('int64').min()) for t, _ in non_empty.values()), step_us)
    tmax = _floor_step(max(int(t.astype('int64').max()) for t, _ in non_empty.values()), step_us)
    times_us = np.arange(tmin, tmax + 1, step_us)
    data = _assemble(stations, non_empty, tmin, times_us.size, step_us)
    points, ids, names, refs = _points_ids_names(stations)

    if chunk_shape is None:
        # ts_ortho default: point dim = 1 (ruling 2026-07-20). The dominant consumer read is
        # ONE station's history — an all-stations point chunk forces downloading the whole
        # dataset to read one station (~250x amplification); one-station chunk columns also
        # keep new-station appends and per-station updates independent. The time chunk trades
        # object count vs per-chunk compression vs the perpetual per-update upload (each run
        # rewrites every station's TAIL chunk): ~25k steps ≈ a handful of objects per station
        # at ~50-200 KB each, a few MB of upload per update run.
        chunk_shape = (1, int(min(times_us.size, _DEFAULT_TIME_CHUNK)))

    unit_us = dict(_UNIT_US).get(unit, 1)
    tvals = times_us.astype('datetime64[us]').astype(f'datetime64[{unit}]')
    with open_dataset(str(path), flag='n', dataset_type='ts_ortho') as ds:
        ds.create.coord.point()
        ds['point'].append(points)
        ds.create.coord.time(data=tvals, step=int(step_us // unit_us))
        ds.create.crs.from_user_input(4326, xy_coord='point')

        dv = ds.create.data_var.generic(variable, ('point', 'time'), dtype=dt, chunk_shape=chunk_shape)
        dv.attrs['units'] = units
        # the declared QC bounds, persisted so merges apply the SAME filter (CF-style attrs)
        dv.attrs['valid_min'] = float(min_value)
        dv.attrs['valid_max'] = float(max_value)
        if standard_name is not None:
            dv.attrs['standard_name'] = standard_name
        if extra_var_attrs:
            dv.attrs.update(extra_var_attrs)
        dv[:] = _nan_safe(data, dt)

        sid = ds.create.data_var.generic(STATION_ID_VAR, ('point',), dtype=dtypes.dtype('str'))
        sid[:] = ids
        snm = ds.create.data_var.generic(STATION_NAME_VAR, ('point',), dtype=dtypes.dtype('str'))
        snm[:] = names
        # the SOURCE's native station identifier (the stations-dict key, e.g. an ECan site
        # number) — the stable join key back to the provider's own records; station_id is
        # geometry-derived and changes if the provider corrects coordinates
        srf = ds.create.data_var.generic(STATION_REF_VAR, ('point',), dtype=dtypes.dtype('str'))
        srf[:] = refs

        ds.attrs.update(attrs)
    return str(path)


def merge_dataset(ds, stations: dict, series: dict, *, variable: str):
    """Fold a recent window into an OPEN ts_ortho dataset (cfdb Dataset or EDataset, flag='w').

    The cadence comes from the dataset's own ``envlib_frequency_interval`` attr (written at
    build), cross-checked against the actual axis (origin on the epoch grid, dense constant
    step). Appends new stations and new steps (in the coord's stored dtype), then does a single
    contiguous read-modify-write over the affected time block, keeping existing values where
    the incoming value is NaN. Idempotent for a fixed input window.
    """
    _require_fixed_cfdb()
    dsattrs = ds.attrs.data
    if 'envlib_frequency_interval' not in dsattrs:
        msg = 'dataset has no envlib_frequency_interval attr — not built by this toolkit'
        raise ValueError(msg)
    code = dsattrs['envlib_frequency_interval']
    step_us, _unit = _freq_step(code)
    _check_phase(dsattrs.get('envlib_utc_offset'), code)

    non_empty = _non_empty(series)
    if not non_empty:
        return {
            'new_stations': 0,
            'new_steps': 0,
            'written_block': 0,
            'gap_steps': 0,
            'qc_rejected': 0,
            'dropped_before_axis': 0,
        }
    _check_refs(non_empty, stations)
    _check_aligned(non_empty, step_us, code)

    dv = ds[variable]
    lo, hi = _qc_bounds(dv)
    non_empty, n_qc = _qc_filter(non_empty, lo, hi)
    if n_qc:
        logger.warning('merge %s: %d value(s) outside [%s, %s] set to NaN (min/max QC)', variable, n_qc, lo, hi)

    cur_ids = list(ds[STATION_ID_VAR].data)
    tdata = np.asarray(ds['time'].data)
    if tdata.size == 0:
        msg = 'dataset has an empty time axis — nothing to merge into'
        raise ValueError(msg)
    cur_times = tdata.astype('datetime64[us]').astype('int64')
    t0 = int(cur_times[0])
    if t0 % step_us:
        msg = f'stored time axis origin {tdata[0]} is not aligned to frequency_interval {code!r}'
        raise ValueError(msg)
    if cur_times.size > 1 and not np.all(np.diff(cur_times) == step_us):
        msg = f'stored time axis is not a dense {code!r} grid (gap, or step mismatch vs the declared frequency)'
        raise ValueError(msg)

    # --- new stations (only those that actually have data this window; `stations` may be the full
    #     discovery dict, but a station with no incoming data must never be added) ---
    active = list(non_empty)
    points, ids, names, refs = _points_ids_names({r: stations[r] for r in active})
    id_to_row = {sid: i for i, sid in enumerate(cur_ids)}
    new_mask = np.array([sid not in id_to_row for sid in ids])
    n_new = int(new_mask.sum())
    if n_new:
        ds['point'].append([p for p, m in zip(points, new_mask, strict=True) if m])
        base = len(cur_ids)
        ds[STATION_ID_VAR][base : base + n_new] = ids[new_mask]
        ds[STATION_NAME_VAR][base : base + n_new] = names[new_mask]
        ds[STATION_REF_VAR][base : base + n_new] = refs[new_mask]
        for k, sid in enumerate(ids[new_mask]):
            id_to_row[sid] = base + k

    # --- new steps (extend the dense axis, in the coord's own stored dtype) ---
    inc_min = _floor_step(min(int(t.astype('int64').min()) for t, _ in non_empty.values()), step_us)
    inc_max = _floor_step(max(int(t.astype('int64').max()) for t, _ in non_empty.values()), step_us)
    cur_max = int(cur_times[-1])
    # steps between the stored axis end and the incoming window start = a hole this window
    # cannot fill (pipeline downtime). The extension below NaN-fills it; report it so the
    # caller can refetch a wider window to heal (holes are NaN, so a later merge fills them).
    gap_steps = max((inc_min - cur_max) // step_us - 1, 0)
    n_new_steps = 0
    if inc_max > cur_max:
        new_steps = np.arange(cur_max + step_us, inc_max + 1, step_us)
        ds['time'].append(new_steps.astype('datetime64[us]').astype(tdata.dtype))
        n_new_steps = new_steps.size

    # --- write block [w_lo, w_hi] (inclusive) via read-modify-write, non-NaN incoming wins ---
    w_hi = (inc_max - t0) // step_us
    if w_hi < 0:
        msg = f'incoming window ends before the axis start ({tdata[0]}) — refusing to prepend history'
        raise ValueError(msg)
    w_lo = max((inc_min - t0) // step_us, 0)
    width = w_hi - w_lo + 1
    existing = np.asarray(dv[:, w_lo : w_hi + 1].data, dtype='float64')  # (n_point_total, width)
    incoming = np.full_like(existing, np.nan)
    n_before = 0
    for ref, (t, v) in non_empty.items():
        d = stations[ref]
        sid = envlib.compute_station_id(shapely.Point(float(d['lon']), float(d['lat'])))
        row = id_to_row[sid]
        col = _step_index(t, t0, step_us) - w_lo
        ok = (col >= 0) & (col < width)
        n_before += int((col < 0).sum())  # window straddles the axis start: pre-axis values
        incoming[row, col[ok]] = v[ok]
    if n_before:
        logger.warning('merge %s: %d incoming value(s) predate the axis start and were dropped', variable, n_before)
    merged = np.where(np.isnan(incoming), existing, incoming)
    dv[:, w_lo : w_hi + 1] = _nan_safe(merged, dv.dtype)
    return {
        'new_stations': n_new,
        'new_steps': int(n_new_steps),
        'written_block': int(width),
        'gap_steps': int(gap_steps),
        'qc_rejected': int(n_qc),
        'dropped_before_axis': int(n_before),
    }


def build_and_publish(cat, path, member_conn, rcg_conn, meta, stations, series, *, num_groups=None, **build_kwargs):
    """First publish: build a local ts_ortho cfdb and publish it to the commons (data then RCG entry).

    ``num_groups=None`` (default) stores each chunk as its own S3 object — the right choice
    for continuously-updated ts_ortho datasets (small key counts, frequent single-chunk
    updates: every push/pull moves exactly the changed data). Grouping is a request-batching
    optimization for LARGE, rarely-updated archives (thousands of keys — see ebooklet's
    guidance of 10-100MB per group); pass a prime ``num_groups`` only for that shape.
    The choice is fixed at first publish.
    """
    build_local(path, meta, stations, series, **build_kwargs)
    return cat.publish(str(path), member_conn, rcg_conn, num_groups=num_groups)


def update_and_publish(cat, path, member_conn, rcg_conn, stations, series, *, variable, num_groups=None):
    """Incremental update: pull the remote, merge the recent window, then publish (diff + entry refresh).

    ``path`` is a local working cache linked to ``member_conn``; the merge reads only the coords +
    the affected time block, so no full-remote materialization is required. ``num_groups`` is
    read from the remote for existing datasets (the first-publish choice wins) — leave it None.
    """
    with open_edataset(member_conn, str(path), flag='w', num_groups=num_groups) as ds:
        report = merge_dataset(ds, stations, series, variable=variable)
    result = cat.publish(str(path), member_conn, rcg_conn, num_groups=num_groups)
    return {'merge': report, 'publish': result}
