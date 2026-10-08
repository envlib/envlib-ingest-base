"""Build and idempotently update envlib ``ts_ortho`` (station) datasets.

A ts_ortho dataset stores an orthogonal ``(point, time)`` layout: a geometry coordinate of
shapely Points (one per station, order-free), a shared **dense fixed-step time axis at the
cadence declared by ``meta.frequency_interval``**, one primary data variable named after
``meta.variable``, and per-station metadata as auxiliary ``(point,)`` variables — the
established nomenclature for station data in envlib/cfdb ts_ortho datasets:
``station_id`` (envlib's deterministic geometry hash; carries the CF ``cf_role='timeseries_id'``
that marks it as the timeseries instance identifier), ``station_name``, and ``station_ref`` (the
SOURCE's native identifier = the stations-dict key, the stable join key back to the provider's
records) — each carrying CF ``long_name``/``comment`` attrs. ``station_altitude`` (metres, CF
``standard_name='altitude'``, unpacked float32) is added when — and only when — a source supplies
a non-null ``'altitude'`` value in the stations dict; it is stored as-reported (no declared
precision — survey accuracy varies station-to-station and is not characterised) and QC'd against a
universal plausibility band (out-of-range values, incl -9999-type sentinels, become missing).
Further well-known fields join the same ``(point,)`` pattern. Station metadata is written once,
when a station first appears (not revised on later merges).

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

- ``stations``: a dict ``{ref: {'lon': float, 'lat': float, 'name': str[, 'altitude': float]}}``
  (from a pandas frame: ``df.to_dict('index')``). ``altitude`` is optional (metres).
- ``series``: a dict ``{ref -> (times, values)}`` of resampled tuples (the output of
  ``resample_*`` at the SAME freq the metadata declares): ascending interval-start
  ``datetime64`` times + ``float64`` values. An entry that is ``None`` or has
  ``times.size == 0`` is simply skipped — never test the tuple itself with ``len()``/truthiness.
  An entry may instead be the 3-tuple ``(times, values, extras)``, where ``extras`` is
  ``{ancillary_name: array}`` positionally paired with ``values`` — see below.

Optionally a dataset carries **ancillary variables**: companion ``(point, time)`` planes holding
per-timestep metadata about each value, declared via ``build_local(..., ancillary=...)`` and named
on the primary by the CF ``ancillary_variables`` attr. The motivating case is a per-observation
quality grade (e.g. NEMS) on a quality-controlled product. They are stored as packed floats rather
than integers because the whole merge/QC machinery is NaN-based, and they are governed on update by
a single union mask so a stored (value, grade) pair always comes from one run (see
``merge_dataset``). There is deliberately NO deletion path: the merge cannot express "this reading
is now missing", which is what protects an offline station's history from being erased.

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

The **merge** is the operational core: each run resamples a recent window and folds it in with a
read-modify-write **per station, each over its own window**, writing incoming values only where
they are non-NaN (so a station that is briefly offline, or an interval that resamples to NaN,
never clobbers good stored data). New stations append to the point axis; new steps extend the
time axis (in the coord's own stored dtype). A re-run over the same window is a no-op.

The chunk shape is ``(1, time_chunk)``: one station, and by default a time chunk that is a highly
composite number of steps spanning whole days (2520 for hourly data; see ``_default_time_chunk``).
The build writes **block by block** — every station's chunk for one time block, then the next
block — because ebooklet's write-order groups pack chunks in the order they were written: a
block-major file keeps each time block in its own few groups, so the hourly update re-uploads the
current block's groups, not the whole dataset. The merge iterates by station over its window.
Peak memory tracks one block of one station, never the point dimension.
Note the store is log-structured, so each merge orphans the chunks it rewrites: a continuously
merged dataset grows every run and wants a periodic `ds.prune()` (local-only, timestamp-preserving,
so it cannot inflate a push).

The cfdb-level functions (``build_local``, ``merge_dataset``) take an open dataset / path and
are unit-tested without any remote. ``build_and_publish`` / ``update_and_publish`` wrap them
with the ebooklet edataset + ``envlib.Catalogue`` publish.
"""

from __future__ import annotations

import logging
import math

import booklet
import numpy as np
from cfdb import dtypes, open_dataset, open_edataset
from ebooklet import DEFAULT_GROUP_BYTES
from envlib.vocabularies import frequency_entry
from rechunkit.main import composite_numbers

from envlib_ingest_base.stations import (
    STATION_ALTITUDE_VAR,
    STATION_ID_VAR,
    STATION_NAME_VAR,
    STATION_REF_VAR,
    _altitudes,
    _apply_station_attrs,
    _check_refs,
    _nan_safe,
    _points_ids_names,
    _qc_bounds,
    _require_fixed_cfdb,
)

logger = logging.getLogger(__name__)


# the default time chunk is the first highly composite number of steps at or above this that spans
# whole days (2520 hourly = 105 days); see _default_time_chunk and the layout decision of 2026-10-08
_MIN_TIME_CHUNK = 2520
_DAY_US = 86_400_000_000
# group_bytes for a grouped remote: a power of two at most half a block (all stations x one time
# chunk), so one block spans at least two groups and a single-station read stays exact
_MIN_GROUP_BYTES = 64 * 1024
# a series entry is (times, values) or (times, values, extras) — the 3rd slot carries ancillary data
_SERIES_WITH_EXTRAS = 3
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


def _default_time_chunk(step_us: int) -> int:
    """The default ts_ortho time chunk for a fixed step: the first highly composite number (from
    rechunkit's list, so later rechunks stay cheap) that is at least ``_MIN_TIME_CHUNK`` steps and
    spans a whole number of days. Hourly -> 2520 (105 days), 15-minute -> 10,080 (105 days).

    The size trades the hourly push (the current chunks plus their groups) against the remote
    index every reader re-downloads after each change (one entry per chunk) and against
    compression (~2 % worse than 25,000 steps). Measurements: ebooklet
    ``planning/write-order-groups-evidence/telemetry-layout/RESULTS.md``.
    """
    for n in composite_numbers:
        if n >= _MIN_TIME_CHUNK and (n * step_us) % _DAY_US == 0:
            return int(n)
    msg = f'no highly composite time chunk spans whole days at a step of {step_us} us'
    raise ValueError(msg)


def _pow2_group_bytes(block_bytes: float) -> int:
    """The largest power of two at most half of ``block_bytes``, floored at 64 KiB and capped at
    ebooklet's default group size (above it, full-band reads get slower)."""
    half = block_bytes / 2
    if half < _MIN_GROUP_BYTES:
        return _MIN_GROUP_BYTES
    return int(min(2 ** math.floor(math.log2(half)), DEFAULT_GROUP_BYTES))


def group_bytes_for(path, variable: str) -> int:
    """The ``group_bytes`` for publishing the local ts_ortho cfdb at ``path`` grouped.

    One block — all stations x one time chunk of ``variable`` — must span at least two groups, or
    a group holds several blocks of every station and a single-station read over-reads by about a
    group. The block size is measured from the stored (compressed) chunk bytes of ``variable``,
    scaled from the file's time length to one chunk. A short or sparse first build measures
    small, which errs toward smaller groups (the safe direction), and a remote can be re-packed to
    another value later by passing it on a push.
    """
    return _pow2_group_bytes(_block_bytes(path, variable))


def _block_bytes(path, variable: str) -> float:
    """Stored bytes of one block of ``variable``: its live chunk bytes scaled from the file's time
    length to one time chunk."""
    with open_dataset(str(path)) as ds:
        n_times = ds[variable].shape[1]
        time_chunk = ds[variable].chunk_shape[1]
    prefix = f'{variable}!'
    with booklet.open(str(path), 'r') as blt:
        stored = sum(vlen for key, _ts, _off, vlen in blt.locations() if key.startswith(prefix))
    return stored * time_chunk / n_times


def _floor_step(us: int, step_us: int) -> int:
    return (us // step_us) * step_us


def _step_index(times, t0_us: int, step_us: int) -> np.ndarray:
    """Integer step offset of datetime64 values from t0 (a dense fixed-step axis origin)."""
    return ((np.asarray(times, dtype='datetime64[us]').astype('int64') - t0_us) // step_us).astype('int64')


def _check_aligned(non_empty: dict, step_us: int, code) -> None:
    """Every incoming timestamp must sit on the epoch-anchored step grid (module docstring)."""
    for ref, (t, *_rest) in non_empty.items():
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


def _qc_filter(non_empty: dict, lo, hi, anc_bounds: dict | None = None) -> tuple[dict, int, int]:
    """min/max QC (ruling 2026-07-17): the DECLARED min/max double as plausibility bounds,
    so values outside them — including ±inf — become NaN (missing). Runs BEFORE the merge
    combine, so a rejected incoming value can never displace a stored valid one.

    Ancillary planes are filtered against their OWN declared bounds, and a value rejected by
    the PRIMARY filter takes its ancillary entries with it: a quality grade attached to a
    measurement we refused to store describes nothing.
    """
    anc_bounds = anc_bounds or {}
    out, n, n_anc = {}, 0, 0
    for ref, (t, v, ex) in non_empty.items():
        if lo is None:
            bad = np.zeros(v.shape, dtype=bool)
        else:
            bad = ~np.isnan(v) & ((v < lo) | (v > hi))
        nbad = int(bad.sum())
        vv = np.where(bad, np.nan, v) if nbad else v
        n += nbad
        ee = {}
        for k, a in ex.items():
            b = anc_bounds.get(k)
            if b is None or b[0] is None:
                abad = np.zeros(a.shape, dtype=bool)
            else:
                abad = ~np.isnan(a) & ((a < b[0]) | (a > b[1]))
            n_anc += int(abad.sum())
            drop = abad | bad
            ee[k] = np.where(drop, np.nan, a) if drop.any() else a
        out[ref] = (t, vv, ee)
    return out, n, n_anc


def _non_empty(series: dict) -> dict:
    """Entries that actually carry data, normalized to ``(times, values, extras)``.

    Accepts both the 2-tuple ``(times, values)`` and the 3-tuple ``(times, values, extras)``
    where ``extras`` is ``{ancillary_name: array}`` **positionally paired with values** — same
    length, same timestamps. The pairing is structural precisely so a value and its ancillary
    entry can never desynchronize; a ragged extras array is a producer bug and raises here,
    before anything is written.
    """
    out = {}
    for ref, s in series.items():
        if s is None:
            continue
        if len(s) == _SERIES_WITH_EXTRAS:
            t, v, extras = s
        else:
            t, v = s
            extras = None
        t = np.asarray(t, dtype='datetime64[us]')
        v = np.asarray(v, dtype='float64')
        ex = {}
        for k, arr in (extras or {}).items():
            a = np.asarray(arr, dtype='float64')
            if a.shape != v.shape:
                msg = (
                    f'station {ref!r}: ancillary {k!r} has shape {a.shape} but values have '
                    f'{v.shape} — extras must be positionally paired with values'
                )
                raise ValueError(msg)
            ex[k] = a
        if t.size:
            out[ref] = (t, v, ex)
    return out


def _check_extras(non_empty: dict, roster) -> None:
    """Incoming ancillary data naming a variable the dataset does not declare RAISES.

    Deliberately louder than the altitude warn-and-continue: silently dropping per-timestep
    quality information from a QC'd product is data loss with no later recovery path (there is
    no rebuild that puts it back without re-running the whole extraction).
    """
    known = set(roster)
    for ref, (_t, _v, ex) in non_empty.items():
        unknown = sorted(set(ex) - known)
        if unknown:
            msg = (
                f'station {ref!r}: ancillary {unknown} not declared by this dataset '
                f'(declared: {sorted(known)})'
            )
            raise ValueError(msg)


def _write_blocks(planes, stations: dict, series: dict, t0_us: int, n_times: int, step_us: int, block: int) -> None:
    """Write the dense ``(n_point, n_time)`` planes BLOCK BY BLOCK: for each time block of
    ``block`` steps (the time chunk), every station's chunk of every plane, then the next block.

    ``planes`` is a list of ``(var, dt, plane)``: ``plane=None`` writes the primary values, a
    string writes that named ancillary (stations lacking it stay NaN). Within a block each station's
    planes are written together, so a value and its ancillary grade sit next to each other.

    **Why block-major.** ebooklet's write-order groups pack chunks in the order they were written
    to the local file. Written station by station, each group holds whole station histories, so the
    hourly update (which rewrites every station's current chunk) touches nearly every group and
    re-uploads about the whole dataset until the next rollover. Written block by block, each group
    holds one time block, and an update touches only the current block's groups.

    It replaced a whole-plane ``_assemble`` (3.01 GB per plane on a 606-station x 620,771-step
    dataset) and then a row-at-a-time writer. Peak memory is one block of one station.

    ⚠️ **Iterate ``stations``, never ``series``.** The point coordinate and the
    ``station_id``/``station_ref``/``station_name`` arrays are built in ``stations`` order, so row
    ``i`` MUST be the ``i``-th entry of ``stations``. Driving the loop from ``series`` instead
    would map the *i*-th station-that-has-data to point *i* and silently mislabel every station
    after the first data-less one — a file that validates cleanly and is wrong.

    A station with no data still gets explicitly written all-NaN chunks: an unwritten chunk also
    reads as NaN, but it would change which chunks exist and the file size.

    With a point chunk of 1 (the ts_ortho default) each chunk is written exactly once (blocks are
    chunk-aligned on a fresh axis); a caller's larger point chunk is revisited once per station, as
    the row writer before it did. Revisiting a chunk costs a full decompress+recompress, orphans the
    previous block (the store is log-structured, so the file grows until pruned), and can force a
    synchronous buffer flush.
    """
    # per station, its integer timestamps in ascending order (the series contract; sorted once if
    # a producer broke it), so each block's slice is two binary searches, not a scan
    ordered = {}
    for ref in stations:
        s = series.get(ref)
        if s is None:
            continue
        t, v, ex = s
        ti = t.view('int64')
        if ti.size > 1 and not bool((ti[1:] >= ti[:-1]).all()):
            order = np.argsort(ti, kind='stable')
            ti, v, ex = ti[order], v[order], {k: a[order] for k, a in ex.items()}
        ordered[ref] = (ti, v, ex)

    for b0 in range(0, n_times, block):
        b1 = min(b0 + block, n_times)
        lo_us, hi_us = t0_us + b0 * step_us, t0_us + b1 * step_us
        for i, ref in enumerate(stations):
            s = ordered.get(ref)
            if s is not None:
                ti, v, ex = s
                lo, hi = (int(x) for x in np.searchsorted(ti, [lo_us, hi_us], side='left'))
            for var, dt, plane in planes:
                # a fresh segment per write: nothing is ever aliased into cfdb's chunk buffer
                seg = np.full(b1 - b0, np.nan, dtype='float64')
                if s is not None and hi > lo:
                    arr = v if plane is None else ex.get(plane)
                    if arr is not None:
                        seg[(ti[lo:hi] - lo_us) // step_us] = arr[lo:hi]
                var[i, b0:b1] = _nan_safe(seg, dt)


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
    ancillary=None,
):
    """Create a fresh local ts_ortho cfdb from all stations + their resampled series.

    The axis cadence comes from ``meta.frequency_interval`` (fixed CV codes only); the time
    coordinate is stored in the cadence's natural unit with an explicit step.
    ``stations``: dict ``{ref: {'lon', 'lat', 'name'[, 'altitude']}}`` (``altitude`` optional, metres —
    the ``station_altitude`` var is created only if at least one station has it); ``series``:
    ``{ref -> (times, values)}`` or ``{ref -> (times, values, extras)}``.

    ``ancillary`` declares optional companion ``(point, time)`` variables carrying per-timestep
    metadata about each value — a NEMS/quality grade being the motivating case::

        ancillary={'quality_code': {'units': '1', 'precision': 0,
                                    'min_value': 0, 'max_value': 1000,
                                    'attrs': {'standard_name': 'quality_flag', ...}}}

    Their data arrives in each series entry's ``extras`` dict. The primary variable gets a CF
    ``ancillary_variables`` attr naming them, which is also the roster ``merge_dataset`` reads
    back — the stored dataset, not a caller argument, is the single source of truth on update.

    Each ancillary is stored as a PACKED FLOAT (not an integer dtype): the whole merge/QC
    machinery is NaN-based, and an integer plane cannot express "no incoming code" at all.
    ``valid_min``/``valid_max`` are written from the declared bounds and are load-bearing, not
    decorative — the encoder rejects below-min and non-finite but passes ABOVE-max values
    straight through, so without those attrs a merge falls back to the far wider encodable
    range and a junk high code would persist silently.
    """
    _require_fixed_cfdb()
    step_us, unit = _freq_step(meta.frequency_interval)
    attrs = meta.to_dict()
    _check_phase(attrs.get('envlib_utc_offset'), meta.frequency_interval)
    ancillary = ancillary or {}
    non_empty = _non_empty(series)
    if not non_empty:
        msg = 'no series data to build from'
        raise ValueError(msg)
    _check_refs(non_empty, stations)
    _check_extras(non_empty, ancillary)
    _check_aligned(non_empty, step_us, meta.frequency_interval)

    dt = dtypes.dtype('float32', precision=precision, min_value=min_value, max_value=max_value)
    anc_bounds = {k: (float(s['min_value']), float(s['max_value'])) for k, s in ancillary.items()}
    non_empty, n_qc, n_anc_qc = _qc_filter(non_empty, float(min_value), float(max_value), anc_bounds)
    if n_qc:
        logger.warning('build %s: %d value(s) outside [%s, %s] set to NaN (QC)', variable, n_qc, min_value, max_value)
    if n_anc_qc:
        logger.warning(
            'build %s: %d ancillary value(s) outside their declared bounds set to NaN (QC)', variable, n_anc_qc
        )

    tmin = _floor_step(min(int(t.astype('int64').min()) for t, *_ in non_empty.values()), step_us)
    tmax = _floor_step(max(int(t.astype('int64').max()) for t, *_ in non_empty.values()), step_us)
    times_us = np.arange(tmin, tmax + 1, step_us)
    # NOTE: the dense plane is NOT assembled here. Blocks are written individually inside the
    # dataset context below (see _write_blocks) so peak memory is one block, not one plane.
    points, ids, names, refs = _points_ids_names(stations)

    if chunk_shape is None:
        # ts_ortho default: point dim = 1 (ruling 2026-07-20). The dominant consumer read is
        # ONE station's history — an all-stations point chunk forces downloading the whole
        # dataset to read one station (~250x amplification); one-station chunk columns also
        # keep new-station appends and per-station updates independent. The time chunk is the
        # telemetry layout of 2026-10-08 (see _default_time_chunk). It is NOT clipped to this
        # build's length: the chunk shape is fixed for the dataset's life, so a short first
        # build (e.g. a one-month backfill) must not fix a short chunk forever; cfdb stores the
        # partly filled chunk and the axis grows into it.
        chunk_shape = (1, _default_time_chunk(step_us))

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
        # CF: name the companion variables on the primary. This attr is also the roster
        # merge_dataset reads back, so the stored dataset stays the source of truth on update.
        if ancillary:
            dv.attrs['ancillary_variables'] = ' '.join(ancillary)
        planes = [(dv, dt, None)]

        for aname, spec in ancillary.items():
            adt = dtypes.dtype(
                'float32',
                precision=spec['precision'],
                min_value=spec['min_value'],
                max_value=spec['max_value'],
            )
            av = ds.create.data_var.generic(aname, ('point', 'time'), dtype=adt, chunk_shape=chunk_shape)
            av.attrs['units'] = spec.get('units', '1')
            if spec.get('attrs'):
                av.attrs.update(spec['attrs'])
            # written LAST so a caller-supplied attrs dict can never weaken the QC bounds
            av.attrs['valid_min'] = float(spec['min_value'])
            av.attrs['valid_max'] = float(spec['max_value'])
            planes.append((av, adt, aname))

        # every plane in one block-major pass, so a value and its ancillary grade land together
        _write_blocks(planes, stations, non_empty, tmin, times_us.size, step_us, chunk_shape[1])

        sid = ds.create.data_var.generic(STATION_ID_VAR, ('point',), dtype=dtypes.dtype('str'))
        sid[:] = ids
        snm = ds.create.data_var.generic(STATION_NAME_VAR, ('point',), dtype=dtypes.dtype('str'))
        snm[:] = names
        # the SOURCE's native station identifier (the stations-dict key, e.g. an ECan site
        # number) — the stable join key back to the provider's own records; station_id is
        # geometry-derived and changes if the provider corrects coordinates
        srf = ds.create.data_var.generic(STATION_REF_VAR, ('point',), dtype=dtypes.dtype('str'))
        srf[:] = refs

        # optional station altitude: create the var only when the source actually supplies it (a
        # source with none, e.g. ECan, gets no empty var); NaN for any station that lacks a value.
        # Created after ds['point'].append so the auto chunk_shape sees a non-empty coord.
        alts = _altitudes(stations)
        if bool((~np.isnan(alts)).any()):
            salt = ds.create.data_var.generic(STATION_ALTITUDE_VAR, ('point',), dtype=dtypes.dtype('float32'))
            salt[:] = alts

        _apply_station_attrs(ds)
        ds.attrs.update(attrs)
    return str(path)


def merge_dataset(ds, stations: dict, series: dict, *, variable: str):
    """Fold a recent window into an OPEN ts_ortho dataset (cfdb Dataset or EDataset, flag='w').

    The cadence comes from the dataset's own ``envlib_frequency_interval`` attr (written at
    build), cross-checked against the actual axis (origin on the epoch grid, dense constant
    step). Appends new stations and new steps (in the coord's stored dtype), then read-modify-writes
    ONE STATION AT A TIME, each over its own window, keeping existing values where the incoming
    value is NaN. Idempotent for a fixed input window. Stations that supply nothing this run are
    not read or written at all.

    Requires a writable dataset: the station-attr heal runs on every call (even an empty window,
    before the early return), so calling this on a read-only dataset raises rather than no-opping.

    **Ancillary variables.** The roster is read from the primary's ``ancillary_variables`` attr
    (never a caller argument), and the write is governed by a single UNION mask: a slot counts as
    "supplied by this run" when the primary OR any ancillary is non-NaN there, and every plane is
    then taken from that run. Two properties follow. A stored (value, code) pair always originates
    from one run — a later run can never leave its value sitting beside an earlier run's grade. And
    a grade can exist without a value, so a code whose entire job is to explain an absence (NEMS
    100, "missing record") is representable rather than silently dropped.

    Note the deliberate asymmetry with deletion: an all-NaN incoming slot leaves the stored pair
    untouched, which is what stops a briefly-offline station from erasing good history — and is
    also why this merge cannot express a genuine retraction. A source that revises a reading *to*
    missing has no way to say so; that needs an explicit deletion path this toolkit does not have.

    **Crash window.** The primary and each ancillary are separate block writes, so an interruption
    between them can leave new values beside old grades for one block. Re-running heals it — but
    only if a subsequent window actually covers the crashed block; the horizon guard forbids
    reaching arbitrarily far back, so a crash outside every later window needs a supervised
    re-merge over that range.
    """
    _require_fixed_cfdb()
    dsattrs = ds.attrs.data
    if 'envlib_frequency_interval' not in dsattrs:
        msg = 'dataset has no envlib_frequency_interval attr — not built by this toolkit'
        raise ValueError(msg)
    code = dsattrs['envlib_frequency_interval']
    step_us, _unit = _freq_step(code)
    _check_phase(dsattrs.get('envlib_utc_offset'), code)

    # back-fill the canonical station-var CF attrs onto pre-existing datasets (idempotent, attrs
    # only). Done here — before the empty-window early return — so even quiet update runs heal.
    _apply_station_attrs(ds)

    non_empty = _non_empty(series)
    if not non_empty:
        return {
            'new_stations': 0,
            'new_steps': 0,
            'written_block': 0,
            'gap_steps': 0,
            'qc_rejected': 0,
            'anc_qc_rejected': 0,
            'dropped_before_axis': 0,
        }
    _check_refs(non_empty, stations)
    _check_aligned(non_empty, step_us, code)

    dv = ds[variable]
    # the ancillary roster comes from the STORED dataset, never from a caller argument
    anc_names = [n for n in str(dv.attrs.data.get('ancillary_variables', '')).split() if n in ds]
    anc_vars = {n: ds[n] for n in anc_names}
    _check_extras(non_empty, anc_names)

    lo, hi = _qc_bounds(dv)
    anc_bounds = {n: _qc_bounds(av) for n, av in anc_vars.items()}
    non_empty, n_qc, n_anc_qc = _qc_filter(non_empty, lo, hi, anc_bounds)
    if n_qc:
        logger.warning('merge %s: %d value(s) outside [%s, %s] set to NaN (min/max QC)', variable, n_qc, lo, hi)
    if n_anc_qc:
        logger.warning('merge %s: %d ancillary value(s) outside declared bounds set to NaN (QC)', variable, n_anc_qc)

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
    sub = {r: stations[r] for r in active}
    points, ids, names, refs = _points_ids_names(sub)
    # ONE derivation per station per merge, reused by the write loop below. That loop used to
    # recompute the id from stations[ref] independently, which meant the store path and the
    # row-lookup path each had their own spelling of "the id for this station" and nothing
    # forced them to agree.
    ref_to_sid = dict(zip(refs, ids, strict=True))
    # validated up front, BEFORE any row mutation: a junk/non-finite altitude then raises
    # pre-append (never leaving a half-written station row), and the check is uniform — merge
    # is as loud on bad altitude as build, regardless of new-vs-existing or var presence.
    alts = _altitudes(sub)
    id_to_row = {sid: i for i, sid in enumerate(cur_ids)}
    new_mask = np.array([sid not in id_to_row for sid in ids])
    n_new = int(new_mask.sum())
    if n_new:
        ds['point'].append([p for p, m in zip(points, new_mask, strict=True) if m])
        base = len(cur_ids)
        ds[STATION_ID_VAR][base : base + n_new] = ids[new_mask]
        ds[STATION_NAME_VAR][base : base + n_new] = names[new_mask]
        ds[STATION_REF_VAR][base : base + n_new] = refs[new_mask]
        if STATION_ALTITUDE_VAR in ds:
            ds[STATION_ALTITUDE_VAR][base : base + n_new] = alts[new_mask]
        for k, sid in enumerate(ids[new_mask]):
            id_to_row[sid] = base + k
    # incoming altitude with no var to hold it = silent omission until a rebuild. Warn once per
    # merge, independent of n_new (a source may start supplying altitude for EXISTING stations).
    # Station metadata is write-once: existing rows' altitude is not revised here (as with name/ref).
    if STATION_ALTITUDE_VAR not in ds and bool((~np.isnan(alts)).any()):
        logger.warning(
            'merge %s: incoming stations carry altitude but the dataset has no %s var; rebuild to add it',
            variable,
            STATION_ALTITUDE_VAR,
        )

    # --- new steps (extend the dense axis, in the coord's own stored dtype) ---
    inc_min = _floor_step(min(int(t.astype('int64').min()) for t, *_ in non_empty.values()), step_us)
    inc_max = _floor_step(max(int(t.astype('int64').max()) for t, *_ in non_empty.values()), step_us)
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
    # ONE STATION AT A TIME, each over ITS OWN window — never one block spanning every point.
    #
    # This previously read `dv[:, w_lo:w_hi+1]` across the FULL point dimension, where w_lo/w_hi
    # came from the GLOBAL min/max incoming timestamp. Two consequences, both bad:
    #
    #   * MEMORY: ~6-7 float64 blocks of (n_point x global_width) live at once. One station
    #     reporting a backdated timestamp widened the block for EVERY station — measured, a
    #     single stale value took a 48-step merge from 9 MB to 228 MB, and at production scale
    #     that is gigabytes inside an hourly cron.
    #   * WRITE AMPLIFICATION: every station's chunks in the block were rewritten even when that
    #     station supplied nothing this run (its slots merged to `existing`, i.e. back to
    #     themselves). Since the store is log-structured and the remote pushes changed keys,
    #     that is an upload of the whole block's chunks on every single run.
    #
    # Per-station is simply correct here: the union mask is elementwise and `incoming[row]` only
    # ever carries data for the station at `row`, so there is NO cross-station dependency to
    # preserve. Each station's window is its own, so a backdated value widens only its own read,
    # and stations that reported nothing are not touched at all.
    #
    # chunk_shape is (1, time_chunk), so a row IS the chunk row: this reads and writes exactly
    # the chunks it must, one station's worth at a time.
    n_before = 0
    widest = 0
    for ref, (t, v, ex) in non_empty.items():
        row = id_to_row[ref_to_sid[ref]]
        col_abs = _step_index(t, t0, step_us)
        n_before += int((col_abs < 0).sum())  # window straddles the axis start: pre-axis values
        keep = col_abs >= 0
        if not keep.any():
            continue
        lo, hi = int(col_abs[keep].min()), int(col_abs[keep].max())
        width = hi - lo + 1
        widest = max(widest, width)
        col = col_abs[keep] - lo

        existing = np.asarray(dv[row, lo : hi + 1].data, dtype='float64').ravel()
        incoming = np.full(width, np.nan)
        incoming[col] = v[keep]
        existing_a, incoming_a = {}, {}
        for n in anc_names:
            existing_a[n] = np.asarray(anc_vars[n][row, lo : hi + 1].data, dtype='float64').ravel()
            inc = np.full(width, np.nan)
            a = ex.get(n)
            if a is not None:
                inc[col] = a[keep]
            incoming_a[n] = inc

        # UNION mask: a slot is "supplied by this run" if the primary OR any ancillary has data
        # there. One mask drives every plane, so a stored (value, code) pair can never be
        # assembled from two different runs — and a code that explains an absent value survives.
        mask = ~np.isnan(incoming)
        for n in anc_names:
            mask |= ~np.isnan(incoming_a[n])
        dv[row, lo : hi + 1] = _nan_safe(np.where(mask, incoming, existing), dv.dtype)
        for n in anc_names:
            merged_a = np.where(mask, incoming_a[n], existing_a[n])
            anc_vars[n][row, lo : hi + 1] = _nan_safe(merged_a, anc_vars[n].dtype)

    if n_before:
        logger.warning('merge %s: %d incoming value(s) predate the axis start and were dropped', variable, n_before)
    return {
        'new_stations': n_new,
        'new_steps': int(n_new_steps),
        # WIDEST single station's window, not one global block — the merge no longer has a
        # single block width. Nothing consumes this field (ingest.py reads only gap_steps); it
        # is a diagnostic, and per-station is the more useful reading of it anyway: a large
        # value now means one station genuinely carried a wide window, not that one stale
        # timestamp stretched the block across every station.
        'written_block': int(widest),
        'gap_steps': int(gap_steps),
        'qc_rejected': int(n_qc),
        'anc_qc_rejected': int(n_anc_qc),
        'dropped_before_axis': int(n_before),
    }


def rechunk_copy(src, dst, *, time_chunk: int) -> None:
    """Copy the ts_ortho dataset ``src`` into the EMPTY ts_ortho dataset ``dst`` with
    ``(1, time_chunk)`` data chunks, written block by block (see ``_write_blocks`` for why).

    ``src`` is any open ts_ortho dataset (a local file, or an EDataset whose chunks it pulls).
    ``dst`` is open for write and empty; the caller decides its header (for example passing an
    existing remote's ``init_bytes`` through ``open_dataset`` to keep that remote's uuid).

    Copied: the dataset attrs, the CRS, the point and time coordinates (the time axis from its
    decoded values, so a source origin left by a truncation does not carry over) with their attrs,
    every ``(point, time)`` variable with the source's dtype/encoding and attrs, and every
    ``(point,)`` station variable with its values and attrs. Anything else (another coordinate, a
    variable on other dims) raises rather than being dropped silently. Memory is one block of every
    ``(point, time)`` variable across all stations, plus what cfdb holds to serve a slice (up to one
    source chunk per station; measured 19.7 MB peak for a 6 MB block). Compression is not copied:
    ``dst`` keeps whatever it was opened with, so pass ``compression=`` there to choose it.
    """
    for name, ds in (('src', src), ('dst', dst)):
        if ds.dataset_type != 'ts_ortho':
            msg = f'rechunk_copy: {name} is a {ds.dataset_type!r} dataset, not ts_ortho'
            raise ValueError(msg)
    if dst.coord_names or dst.data_var_names:
        msg = f'rechunk_copy: dst must be empty, but holds {dst.coord_names + dst.data_var_names}'
        raise ValueError(msg)
    extra = set(src.coord_names) - {'point', 'time'}
    if extra:
        msg = f'rechunk_copy: coordinates {sorted(extra)} are not supported'
        raise NotImplementedError(msg)
    planes, statics = [], []
    for name in src.data_var_names:
        dims = tuple(src[name].coord_names)
        if dims == ('point', 'time'):
            planes.append(name)
        elif dims == ('point',):
            statics.append(name)
        else:
            msg = f'rechunk_copy: variable {name!r} on dims {dims} is not supported'
            raise NotImplementedError(msg)

    dst.create.coord.point()
    dst['point'].append(src['point'].data)
    dst.create.coord.time(data=src['time'].data, step=src['time'].step)
    for name in ('point', 'time'):
        dst[name].attrs.update(dict(src[name].attrs))
    if src.crs is not None:
        dst.create.crs.from_user_input(src.crs, xy_coord='point')

    out = {}
    for name in planes:
        out[name] = dst.create.data_var.generic(
            name, ('point', 'time'), dtype=src[name].dtype, chunk_shape=(1, int(time_chunk))
        )
        out[name].attrs.update(dict(src[name].attrs))
    if planes:
        n_points, n_times = src[planes[0]].shape
        for b0 in range(0, n_times, time_chunk):
            b1 = min(b0 + time_chunk, n_times)
            block = {name: np.asarray(src[name][:, b0:b1].data, dtype='float64') for name in planes}
            for i in range(n_points):
                for name in planes:
                    out[name][i, b0:b1] = _nan_safe(block[name][i].copy(), out[name].dtype)

    for name in statics:
        sv = dst.create.data_var.generic(name, ('point',), dtype=src[name].dtype)
        sv[:] = src[name].data
        sv.attrs.update(dict(src[name].attrs))
    dst.attrs.update(dict(src.attrs))


_DTYPE_FIELDS = ('name', 'dtype_encoded', 'dtype_decoded', 'precision', 'fillvalue', 'offset')


def _dtype_fields(dt) -> dict:
    return {f: str(getattr(dt, f, None)) for f in _DTYPE_FIELDS}


def _values_differ(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return True
    if x.dtype.kind in 'fc' or y.dtype.kind in 'fc':
        return not np.array_equal(x.astype('float64'), y.astype('float64'), equal_nan=True)
    return list(x.ravel().tolist()) != list(y.ravel().tolist())


def compare_datasets(a, b, *, block: int = _MIN_TIME_CHUNK) -> list:
    """Every difference between two ts_ortho datasets, ignoring only chunk shapes and the time
    coordinate's origin. An empty list means equal. Built as the gate before an irreversible
    republish, so it checks everything a rechunk must preserve:

    - the dataset type, its attrs and the CRS;
    - the coordinates: decoded values (points by their WKB), every dtype field, step and attrs;
    - every variable: dims, shape, attrs, and every dtype/encoding field (``_DTYPE_FIELDS``);
    - the values: ``(point, time)`` variables block by block (``block`` steps across all stations,
      NaN-aware), station variables whole.

    Read through an EDataset, a chunk the local file lacks is pulled, not read as fill; read
    through a plain local file, a chunk that was never materialised reads as fill — so compare
    against a remote read when the question is "did the copy keep everything".
    """
    diffs = []
    if a.dataset_type != b.dataset_type:
        diffs.append(f'dataset_type: {a.dataset_type!r} != {b.dataset_type!r}')
    # dict() first: cfdb's Attributes.__iter__ returns dict_keys, not an iterator, so set()/iter() on it raise
    da, db = dict(a.attrs), dict(b.attrs)
    if da != db:
        diffs.append(f'dataset attrs differ: {sorted(k for k in da.keys() | db.keys() if da.get(k) != db.get(k))}')
    if (a.crs is None) != (b.crs is None) or (a.crs is not None and a.crs != b.crs):
        diffs.append(f'crs: {a.crs!r} != {b.crs!r}')

    if set(a.coord_names) != set(b.coord_names):
        diffs.append(f'coordinates: {sorted(a.coord_names)} != {sorted(b.coord_names)}')
    for name in sorted(set(a.coord_names) & set(b.coord_names)):
        ca, cb = a[name], b[name]
        fa, fb = _dtype_fields(ca.dtype), _dtype_fields(cb.dtype)
        if fa != fb or ca.step != cb.step:
            diffs.append(f'coord {name}: dtype/step {fa}/{ca.step} != {fb}/{cb.step}')
        if dict(ca.attrs) != dict(cb.attrs):
            diffs.append(f'coord {name}: attrs differ')
        va, vb = np.asarray(ca.data), np.asarray(cb.data)
        if name == 'point':
            same = va.shape == vb.shape and [g.wkb for g in va] == [g.wkb for g in vb]
        else:
            same = not _values_differ(va, vb)
        if not same:
            diffs.append(f'coord {name}: values differ')

    if set(a.data_var_names) != set(b.data_var_names):
        diffs.append(f'variables: {sorted(a.data_var_names)} != {sorted(b.data_var_names)}')
    for name in sorted(set(a.data_var_names) & set(b.data_var_names)):
        va, vb = a[name], b[name]
        if tuple(va.coord_names) != tuple(vb.coord_names) or va.shape != vb.shape:
            diffs.append(f'{name}: dims/shape {va.coord_names}{va.shape} != {vb.coord_names}{vb.shape}')
            continue
        aa, ab = dict(va.attrs), dict(vb.attrs)
        if aa != ab:
            diffs.append(f'{name}: attrs differ: {sorted(k for k in aa.keys() | ab.keys() if aa.get(k) != ab.get(k))}')
        if _dtype_fields(va.dtype) != _dtype_fields(vb.dtype):
            diffs.append(f'{name}: dtype/encoding {_dtype_fields(va.dtype)} != {_dtype_fields(vb.dtype)}')
        if tuple(va.coord_names) == ('point', 'time'):
            n_times = va.shape[1]
            for b0 in range(0, n_times, block):
                b1 = min(b0 + block, n_times)
                if _values_differ(va[:, b0:b1].data, vb[:, b0:b1].data):
                    diffs.append(f'{name}: values differ in steps [{b0}, {b1})')
                    break
        elif _values_differ(va.data, vb.data):
            diffs.append(f'{name}: values differ')
    return diffs


def build_and_publish(cat, path, member_conn, rcg_conn, meta, stations, series, *, group_bytes='auto', **build_kwargs):
    """First publish: build a local ts_ortho cfdb and publish it to the commons (data then RCG entry).

    ``group_bytes='auto'`` (default) publishes grouped with ``group_bytes_for(path, variable)``:
    the telemetry layout of 2026-10-08, where the block-major build keeps each time block in its
    own few groups, so the hourly update uploads the current block's groups rather than one object
    per station. Pass an int to choose the group size, or ``None`` for per-key storage (one S3
    object per chunk; e.g. a frozen dataset that is never updated). Later pushes inherit the mode
    and the recorded group size.

    ``**build_kwargs`` go straight to ``build_local`` — including ``ancillary=`` to declare
    companion ``(point, time)`` planes (e.g. a per-timestep quality grade), whose data rides in
    each ``series`` entry's third tuple slot. ``update_and_publish`` needs no equivalent argument:
    the roster is read back from the stored dataset's ``ancillary_variables`` attr.
    """
    build_local(path, meta, stations, series, **build_kwargs)
    if group_bytes == 'auto':
        group_bytes = group_bytes_for(path, build_kwargs['variable'])
        logger.info('publish %s: grouped, group_bytes=%d', build_kwargs['variable'], group_bytes)
    return cat.publish(str(path), member_conn, rcg_conn, group_bytes=group_bytes)


def update_and_publish(cat, path, member_conn, rcg_conn, stations, series, *, variable):
    """Incremental update: pull the remote, merge the recent window, then publish (diff + entry refresh).

    ``path`` is a local working cache linked to ``member_conn``; the merge reads only the coords +
    the affected time block, so no full-remote materialization is required. The storage mode is
    the existing remote's (fixed at first publish), so none is passed here.
    """
    with open_edataset(member_conn, str(path), flag='w') as ds:
        report = merge_dataset(ds, stations, series, variable=variable)
    result = cat.publish(str(path), member_conn, rcg_conn)
    return {'merge': report, 'publish': result}
