"""Build and incrementally extend envlib ``ts_forecast`` (point forecast) datasets.

A forecast archive is a sequence of **runs**. Each run is issued at an *init* time
(CF ``forecast_reference_time``) and predicts a series of future steps at fixed *leads*
(CF ``forecast_period``). Valid time is ``init + lead``. The layout is
``(point, forecast_reference_time, forecast_period)`` -- a LEAD axis, not a valid-time axis,
because ``(point, init, valid_time)`` is ~97 % empty while ``(point, init, lead)`` is dense.

**The write unit is one RUN, not one station** -- and that is the whole difference from
``tsortho``. Each run is a dense ``(all points, 1 init, all leads)`` slab, which is exactly one
chunk, so a run is appended in a single write with no read-modify-write::

    ds['forecast_reference_time'].append([init])   # explicit step -> a missed run auto-fills
    dv[:, frt_index, :] = slab                     # (n_points, n_leads); one chunk, one write

Everything else in this module is a guard, and they exist because every failure mode here is
SILENT -- each produces a file that validates, reads back cleanly, and answers wrongly.

Three inputs recur:

- ``stations``: ``{ref: {'lon': float, 'lat': float, 'name': str[, 'altitude': float]}}``,
  identical to ``tsortho``. Station identity comes from :mod:`envlib_ingest_base.stations`, so the
  two builders cannot fork on the 5-decimal-place geometry canonicalisation.
- ``run``: ``{'reference_time': datetime64, 'leads': int array, 'values': {ref: float array}}``.
  ``leads`` and each ``values`` array are POSITIONALLY PAIRED -- same length, same order. The
  pairing is structural precisely so a value and its lead cannot desynchronise.
- ``variable``: the primary data variable's name, which must equal ``meta.variable``.

Four things this module refuses, each of which the storage layer will otherwise accept quietly:

*An OFF-GRID init.* ``forecast_reference_time`` declares an explicit step, and a step is the only
thing that makes a *missed* run recoverable: it auto-fills the skipped slot, and a later merge
writes into that slot rather than attempting a middle insertion. But **cfdb does not enforce its
own step for datetime coordinates** -- ``utils._generate_step_fill``'s datetime branch truncates
the gap to an int and then tests ``isclose`` against that int's own rounding, which is true for
every input (fixed in cfdb 0.9.7; this guard does NOT assume that fix is present). An off-grid
init is therefore appended silently onto an axis that goes on reporting its declared step, and
envlib's validation does not check axis uniformity either. So this module is the only thing
between a mislabelled init and the archive -- the same role ``tsortho._check_aligned`` plays for
a dense time axis. It refuses rather than snapping: flooring an init onto the declared cadence
would silently label a run with an init it does not have, and a loud refusal is recoverable while
a mislabelled init is not.

*A duplicate init.* A forecast run is immutable. Re-ingesting an init raises unless the caller
passes ``overwrite=True``, which is logged loudly. Callers should not rely on the raise for
idempotence -- check ``init in ds['forecast_reference_time']`` and skip, so a routine re-read of
an unchanged source file is a quiet no-op rather than an alert.

*A non-integer lead.* cfdb has no timedelta dtype, so a lead is a bare integer carrying a CF
``units`` attribute, and a float lead would be silently truncated by the encoder.

*A lead vector that is neither the stored one, a prefix of it, nor an extension of it.* Anything
else -- a different step, a different first lead, a gap -- is a source change that needs a human.

Chunking: the point dimension of a chunk is **locked at creation**. Sizing it from the station
count means every later station lands in a second, mostly-padding partial chunk forever, so
``point_headroom`` deliberately over-sizes it. This matters in practice, not in theory: a forecast
provider's station roster changes whenever someone asks them to move a site.
"""

from __future__ import annotations

import logging

import envlib
import numpy as np
from cfdb import dtypes, open_dataset, open_edataset

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

FRT_COORD = 'forecast_reference_time'
PERIOD_COORD = 'forecast_period'

# The forecast_reference_time coordinate is stored as datetime64[m] (cfdb-vars' definition), so
# its declared step is an integer number of MINUTES.
_MINUTES_PER_HOUR = 60

# Over-size the chunk's point dimension so a station added later still lands in the same chunk.
# 8 covers every point-forecast roster this toolkit has seen; the cost is padding in a ~1 KB chunk.
DEFAULT_POINT_HEADROOM = 8


def _normalise_run(run: dict, stations: dict) -> tuple:
    """``run`` -> ``(init, leads, values)`` with every structural guard applied.

    Returns ``init`` as ``datetime64[m]`` (the storage unit of the init axis), ``leads`` as an
    ascending int64 array, and ``values`` as ``{ref: float64 array}`` aligned to ``leads``.
    """
    for key in ('reference_time', 'leads', 'values'):
        if key not in run:
            msg = f'run is missing {key!r}; expected keys reference_time, leads, values'
            raise ValueError(msg)

    init = np.asarray(run['reference_time'], dtype='datetime64[m]').reshape(())
    raw_leads = np.asarray(run['leads'])
    if raw_leads.ndim != 1 or raw_leads.size == 0:
        msg = f'run leads must be a non-empty 1-D array; got shape {raw_leads.shape}'
        raise ValueError(msg)
    # A float lead is REFUSED rather than cast: forecast_period is a bare integer coordinate
    # (cfdb has no timedelta dtype), so 1.5 would encode as 1 and shift the valid time of every
    # value beneath it by half a step, with no error anywhere.
    if raw_leads.dtype.kind not in ('i', 'u'):
        msg = (
            f'run leads must be an integer dtype, not {raw_leads.dtype!r} -- {PERIOD_COORD} is a '
            f'bare integer coordinate and a fractional lead would be silently truncated'
        )
        raise ValueError(msg)
    leads = raw_leads.astype('int64')
    if leads.size > 1:
        gaps = np.unique(np.diff(leads))
        if gaps.size != 1 or gaps[0] <= 0:
            msg = f'run leads must be strictly ascending with one uniform step; got diffs {gaps.tolist()}'
            raise ValueError(msg)

    values = {}
    for ref, arr in run['values'].items():
        a = np.asarray(arr, dtype='float64')
        if a.shape != leads.shape:
            msg = (
                f'station {ref!r}: values have shape {a.shape} but leads have {leads.shape} -- '
                f'values must be positionally paired with leads'
            )
            raise ValueError(msg)
        values[ref] = a
    if not values:
        msg = 'run carries no station values'
        raise ValueError(msg)
    _check_refs(values, stations)
    return init, leads, values


def _lead_step(leads: np.ndarray) -> int:
    return int(leads[1] - leads[0]) if leads.size > 1 else 1


def _check_init_on_grid(init, origin, step_minutes: int) -> None:
    """THE guard this module exists for -- see the module docstring.

    The incoming init must sit exactly on the axis's own grid. cfdb will not check this for a
    datetime coordinate, and nothing downstream re-derives an axis from its declared step, so a
    mislabelled init is invisible at every layer once written.
    """
    delta = (init.astype('datetime64[m]') - origin.astype('datetime64[m]')).astype('int64')
    if delta % step_minutes:
        msg = (
            f'{FRT_COORD} {init} is not on the declared grid: it is {delta} min from the axis '
            f'origin {origin}, which is not a multiple of the {step_minutes} min step. Refusing '
            f'rather than snapping -- a snapped init would label this run with a time it does not '
            f'have. Check the source file, or the declared run cadence.'
        )
        raise ValueError(msg)


def _resolve_lead_block(stored: np.ndarray, leads: np.ndarray) -> tuple:
    """Reconcile a run's leads against the stored ``forecast_period`` axis.

    Returns ``(n_write, extension)``: how many leads of the slab to write, and any new lead
    values to append to the coordinate first. Accepts exactly three shapes -- identical, a
    prefix of the stored axis (a short run; the tail stays missing), or an extension of it (a
    longer product). Anything else is a source change a human should look at.
    """
    stored = np.asarray(stored, dtype='int64')
    n = min(stored.size, leads.size)
    if not np.array_equal(stored[:n], leads[:n]):
        msg = (
            f'run leads are not compatible with the stored {PERIOD_COORD} axis: stored starts '
            f'{stored[:5].tolist()}, run starts {leads[:5].tolist()}. A run must match the stored '
            f'leads, be a prefix of them (a short run), or extend them (a longer product).'
        )
        raise ValueError(msg)
    if leads.size <= stored.size:
        return int(leads.size), None
    return int(leads.size), leads[stored.size :]


def _slab(stations: dict, values: dict, n_leads: int, dt) -> np.ndarray:
    """The run's dense ``(n_point, n_leads)`` plane, in ``stations`` order.

    ⚠️ Iterate ``stations``, never ``values``. The point coordinate and the station_id/ref/name
    arrays are built in ``stations`` order, so row *i* MUST be the *i*-th entry of ``stations``.
    Driving the loop from ``values`` would map the *i*-th station-that-reported to point *i* and
    silently mislabel every station after the first absent one -- a file that validates and is
    wrong. (``tsortho._write_rows`` carries the same warning for the same reason.)

    A station absent from this run keeps an all-NaN row: a run is written once, into a slot that
    was empty, so there is nothing to clobber.
    """
    plane = np.full((len(stations), n_leads), np.nan, dtype='float64')
    for i, ref in enumerate(stations):
        arr = values.get(ref)
        if arr is not None:
            plane[i, :] = arr[:n_leads]
    return _nan_safe(plane, dt)


def _qc(values: dict, lo, hi, variable: str) -> dict:
    """The declared min/max double as plausibility bounds (the toolkit-wide ruling): values
    outside them -- including infinities -- become missing rather than being stored."""
    if lo is None:
        return values
    out, n = {}, 0
    for ref, v in values.items():
        bad = ~np.isnan(v) & ((v < lo) | (v > hi))
        n += int(bad.sum())
        out[ref] = np.where(bad, np.nan, v) if bad.any() else v
    if n:
        logger.warning('%s: %d value(s) outside [%s, %s] set to NaN (QC)', variable, n, lo, hi)
    return out


def build_local(
    path,
    meta,
    stations: dict,
    run: dict,
    *,
    variable: str,
    units: str,
    precision,
    min_value,
    max_value,
    frt_step_hours,
    lead_units: str = 'h',
    point_headroom: int = DEFAULT_POINT_HEADROOM,
    chunk_shape=None,
    standard_name=None,
    extra_var_attrs=None,
    validate: bool = True,
):
    """Create a fresh local ``ts_forecast`` cfdb from all stations and the FIRST run.

    ``frt_step_hours`` is the provider's run cadence and becomes the init axis's explicit step.
    Declaring it is what makes a *missed* run recoverable later, and ``step=True`` is not a
    substitute: auto-detect infers nothing from the single-value axis a first run creates.

    ``lead_units`` is written to ``forecast_period``'s CF ``units`` attribute and is load-bearing,
    not decoration -- cfdb-vars deliberately supplies no default, because a consumer adding a bare
    integer lead to the ``datetime64[m]`` init axis would otherwise add MINUTES.

    ``validate=True`` runs envlib's full dataset validation on the finished file. Leave it on: this
    archive is not published to a catalogue, so this is the only place the guards that catalogue
    publication would apply -- above all the ``station_id`` recomputation from the stored geometry
    -- ever run.
    """
    _require_fixed_cfdb()
    init, leads, values = _normalise_run(run, stations)
    # The init axis is datetime64[m], so the cadence must land on a whole minute. A fractional
    # one is REFUSED rather than rounded: rounding would declare a step the provider does not
    # keep, and every later on-grid check would then be measured against the wrong grid.
    step_exact = float(frt_step_hours) * _MINUTES_PER_HOUR
    step_minutes = round(step_exact)
    if step_minutes <= 0 or step_minutes != step_exact:
        msg = (
            f'frt_step_hours must be positive and a whole number of minutes; got '
            f'{frt_step_hours!r} ({step_exact} min)'
        )
        raise ValueError(msg)
    if not lead_units:
        msg = f'lead_units is required -- {PERIOD_COORD} carries no default units by design'
        raise ValueError(msg)

    dt = dtypes.dtype('float32', precision=precision, min_value=min_value, max_value=max_value)
    values = _qc(values, float(min_value), float(max_value), variable)
    points, ids, names, refs = _points_ids_names(stations)
    alts = _altitudes(stations)
    if chunk_shape is None:
        chunk_shape = (max(int(point_headroom), len(stations)), 1, int(leads.size))

    with open_dataset(str(path), flag='n', dataset_type='ts_forecast') as ds:
        ds.create.coord.point()
        ds['point'].append(points)
        ds.create.coord.forecast_reference_time(data=np.array([init]), step=step_minutes)
        ds.create.coord.forecast_period(data=leads.astype('int32'), step=_lead_step(leads))
        ds[PERIOD_COORD].attrs['units'] = lead_units
        ds.create.crs.from_user_input(4326, xy_coord='point')

        dv = ds.create.data_var.generic(variable, ('point', FRT_COORD, PERIOD_COORD), dtype=dt, chunk_shape=chunk_shape)
        dv.attrs['units'] = units
        # persisted so every later merge applies the SAME plausibility filter
        dv.attrs['valid_min'] = float(min_value)
        dv.attrs['valid_max'] = float(max_value)
        if standard_name is not None:
            dv.attrs['standard_name'] = standard_name
        if extra_var_attrs:
            dv.attrs.update(extra_var_attrs)
        dv[:, 0, :] = _slab(stations, values, int(leads.size), dt)

        sid = ds.create.data_var.generic(STATION_ID_VAR, ('point',), dtype=dtypes.dtype('str'))
        sid[:] = ids
        snm = ds.create.data_var.generic(STATION_NAME_VAR, ('point',), dtype=dtypes.dtype('str'))
        snm[:] = names
        srf = ds.create.data_var.generic(STATION_REF_VAR, ('point',), dtype=dtypes.dtype('str'))
        srf[:] = refs
        if bool((~np.isnan(alts)).any()):
            salt = ds.create.data_var.generic(STATION_ALTITUDE_VAR, ('point',), dtype=dtypes.dtype('float32'))
            salt[:] = alts

        _apply_station_attrs(ds)
        ds.attrs.update(meta.to_dict())
        if validate:
            _validate(ds)
    return str(path)


def _validate(ds) -> None:
    """Run envlib's dataset validation against the OPEN dataset.

    Deliberately against the open object rather than the path: reopening a file that is linked to
    a remote starts a second session against it, and an open ``EDataset`` pulls transparently, so
    extents come from the whole dataset rather than whichever chunks happen to be local.
    """
    envlib.validate_dataset(ds)


def merge_run(ds, stations: dict, run: dict, *, variable: str, overwrite: bool = False) -> dict:
    """Fold ONE run into an open ``ts_forecast`` dataset (cfdb Dataset or EDataset, ``flag='w'``).

    Appends any station that is new, reconciles the lead axis, places the init on the declared
    grid, and writes the run's slab in a single call. Returns a report dict.

    ``overwrite=True`` replaces an init that is already present -- the whole slab, since a run is
    atomic. It is logged at WARNING because overwriting a stored forecast is a recovery action,
    not routine.
    """
    _require_fixed_cfdb()
    _apply_station_attrs(ds)  # heal the canonical CF attrs on pre-existing datasets, idempotently
    init, leads, values = _normalise_run(run, stations)

    frt = ds[FRT_COORD]
    step_minutes = frt.step
    if not step_minutes:
        msg = (
            f'{FRT_COORD} has no declared step, so a missed run could never be back-filled '
            f'(cfdb cannot insert into the middle of a coordinate). Rebuild with frt_step_hours.'
        )
        raise ValueError(msg)
    stored_frt = np.asarray(frt.data).astype('datetime64[m]')
    _check_init_on_grid(init, stored_frt[0], int(step_minutes))

    dv = ds[variable]
    lo, hi = _qc_bounds(dv)
    values = _qc(values, lo, hi, variable)

    # --- lead axis -------------------------------------------------------------------------
    stored_leads = np.asarray(ds[PERIOD_COORD].data, dtype='int64')
    n_write, extension = _resolve_lead_block(stored_leads, leads)
    n_new_leads = 0
    if extension is not None:
        ds[PERIOD_COORD].append(extension.astype(np.asarray(ds[PERIOD_COORD].data).dtype))
        n_new_leads = int(extension.size)

    # --- new stations ----------------------------------------------------------------------
    points, ids, names, refs = _points_ids_names(stations)
    alts = _altitudes(stations)
    stored_ids = [str(v) for v in ds[STATION_ID_VAR].data]
    id_to_row = {sid: i for i, sid in enumerate(stored_ids)}
    new_mask = np.array([sid not in id_to_row for sid in ids])
    n_new = int(new_mask.sum())
    if n_new:
        base = len(stored_ids)
        ds['point'].append([p for p, m in zip(points, new_mask, strict=True) if m])
        ds[STATION_ID_VAR][base : base + n_new] = ids[new_mask]
        ds[STATION_NAME_VAR][base : base + n_new] = names[new_mask]
        ds[STATION_REF_VAR][base : base + n_new] = refs[new_mask]
        if STATION_ALTITUDE_VAR in ds:
            ds[STATION_ALTITUDE_VAR][base : base + n_new] = alts[new_mask]
        for k, sid in enumerate(ids[new_mask]):
            id_to_row[sid] = base + k
    if STATION_ALTITUDE_VAR not in ds and bool((~np.isnan(alts)).any()):
        logger.warning(
            'merge %s: incoming stations carry altitude but the dataset has no %s var; rebuild to add it',
            variable,
            STATION_ALTITUDE_VAR,
        )

    # --- place the init --------------------------------------------------------------------
    existing = np.flatnonzero(stored_frt == init)
    replaced = False
    if existing.size:
        # An init being ON THE AXIS does not mean a run was ever written there. A declared step
        # AUTO-FILLS the slots of missed runs, and those placeholders are exactly what a later
        # back-fill is supposed to write into -- so testing membership alone would make the
        # immutability rule forbid the recovery the step exists to enable. Distinguish the two by
        # the DATA: an all-missing slab is a placeholder, anything else is a stored run.
        # (A real run in which every station was absent is indistinguishable from a placeholder,
        # and is correctly treated as one -- it carries nothing to protect.)
        idx = int(existing[0])
        stored_slab = np.asarray(ds[variable][:, idx, :].data, dtype='float64')
        replaced = bool(np.isfinite(stored_slab).any())
        if replaced and not overwrite:
            msg = (
                f'{FRT_COORD} {init} already holds a stored run -- a forecast run is immutable. '
                f'Pass overwrite=True to replace it deliberately, or skip this run (checking '
                f'whether the init is already on the axis is the right idempotence test for a '
                f're-read of an unchanged source file).'
            )
            raise ValueError(msg)
        if replaced:
            logger.warning('merge %s: OVERWRITING stored run %s', variable, init)
        else:
            logger.info('merge %s: back-filling the auto-filled slot at %s', variable, init)
    elif init < stored_frt[0]:
        msg = f'{FRT_COORD} {init} predates the axis start {stored_frt[0]}; refusing to prepend history'
        raise ValueError(msg)
    else:
        # the declared step auto-fills any skipped inits; those slots read missing until a later
        # merge writes into them, which is exactly what makes a missed run recoverable
        frt.append(np.array([init]))
        idx = int(np.flatnonzero(np.asarray(frt.data).astype('datetime64[m]') == init)[0])
    filled = 0 if existing.size else max(idx - (stored_frt.size - 1) - 1, 0)

    # --- one run, one slab, one write ------------------------------------------------------
    # The plane is sized to the DATASET's point count and addressed BY ROW INDEX, not built in
    # `stations` order. Those two coincide on a fresh build and stop coinciding the moment the
    # roster changes -- a station retired from the config still occupies its row, and a station
    # added later sits past the end of the config's ordering. Deriving each row from the stored
    # station_id is the only thing that keeps a value attached to the station it belongs to.
    dv = ds[variable]
    ref_to_sid = dict(zip(refs, ids, strict=True))
    n_points = int(np.asarray(ds['point'].data).size)
    plane = np.full((n_points, n_write), np.nan, dtype='float64')
    for ref, arr in values.items():
        plane[id_to_row[ref_to_sid[str(ref)]], :] = arr[:n_write]
    dv[:, idx, :n_write] = _nan_safe(plane, dv.dtype)

    if filled:
        logger.warning(
            'merge %s: %d missed run slot(s) auto-filled before %s; they read missing until back-filled',
            variable,
            filled,
            init,
        )
    return {
        'init': str(init),
        'init_index': idx,
        'new_stations': n_new,
        'new_leads': n_new_leads,
        'autofilled_slots': int(filled),
        'overwritten': replaced,
        'backfilled': bool(existing.size) and not replaced,
        'stations_reporting': len(values),
    }


def build_and_push(member_conn, path, meta, stations: dict, run: dict, *, num_groups=None, **build_kwargs):
    """First run: build the local ``ts_forecast`` and push it to its S3 remote.

    There is deliberately **no catalogue here**. A forecast archive of a commercial provider's
    product is not published to a public commons, so ``envlib.Catalogue`` contributes nothing --
    but the metadata is still generated by ``envlib.Metadata`` and the file still passes the full
    ``envlib.validate_dataset`` in ``build_local``, so the dataset carries CV-validated identity
    fields and is trivially promotable later if that ever changes.

    ``num_groups=None`` (per-key objects) is right for a continuously-appended archive: each run
    is one small chunk, and every push moves exactly the run that changed.
    """
    build_local(path, meta, stations, run, **build_kwargs)
    with open_edataset(member_conn, str(path), flag='w', num_groups=num_groups) as eds:
        result = eds.push()
    return {'path': str(path), 'push': result}


def update_and_push(
    member_conn, path, stations: dict, run: dict, *, variable: str, overwrite: bool = False, num_groups=None
):
    """Incremental run: pull the remote, merge one run, push.

    ``path`` is a local working cache linked to ``member_conn``. Note the store is log-structured,
    so every run rewrites the init coordinate and orphans the previous copy -- the local file grows
    on every run and wants a periodic ``ds.prune()``, AFTER the push (pruning is local-only and
    preserves key timestamps, so it cannot inflate the next push).
    """
    with open_edataset(member_conn, str(path), flag='w', num_groups=num_groups) as ds:
        report = merge_run(ds, stations, run, variable=variable, overwrite=overwrite)
        result = ds.push()
    return {'merge': report, 'push': result}
