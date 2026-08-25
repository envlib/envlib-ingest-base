"""Station identity, altitude and encoding helpers shared by every dataset builder.

Extracted from ``tsortho`` when the forecast builder arrived, because the two must not fork.
``_points_ids_names`` in particular is the reason this module exists: it canonicalises each point
with ``envlib.canonical_station_point`` BEFORE the geometry is written, and a second copy of that
logic is a second chance to get the 5-decimal-place rounding wrong. A live ECan publish already
failed exactly that way on 2026-08-24 -- the stored geometry no longer derived the ``station_id``
stored beside it, and the dataset could never be published.

Nothing here knows about time, cadence, leads or runs. That is the dividing line: a helper belongs
in this module when it concerns WHICH STATION a row is, and in the builder when it concerns WHEN.
"""

from __future__ import annotations

import logging

import cfdb
import envlib
import numpy as np
import shapely

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

STATION_ALTITUDE_VAR = 'station_altitude'

# station_altitude plausibility band (metres): a UNIVERSAL physical range for a station's
# elevation — above every surface station on Earth (highest AWS ~8,810 m; Everest summit 8,849)
# down through the deepest dry-land depressions (Dead Sea shore ~-430 m). Numeric values outside
# it (incl ±inf, float32 overflow, and the classic -9999/-999 sentinels) are treated as MISSING,
# not real. Precision is deliberately NOT declared: a station's survey accuracy varies
# station-to-station, is rarely documented, and would be dishonest to assert — altitude is stored
# as-reported (unpacked float32), and the comment attr says so.
_ALTITUDE_MIN = -500.0

_ALTITUDE_MAX = 9000.0

_MAX_WARN_REFS = 10  # cap the station list shown in the out-of-range altitude warning

# CF attributes for the auxiliary station-metadata variables. cfdb already writes the global
# featureType='timeSeries' for ts_ortho; per CF discrete-sampling-geometry conventions (§9.5)
# exactly one variable carries cf_role to mark the timeseries instance identifier — station_id
# (envlib's canonical geometry-derived station identity). station_altitude is optional (created
# only when a source supplies it; see build_local), so its attrs apply only where the var exists.
_STATION_VAR_ATTRS = {
    STATION_ID_VAR: {
        'long_name': 'envlib station identifier',
        'cf_role': 'timeseries_id',
        'comment': (
            'Deterministic hash of the station geometry (envlib.compute_station_id); envlib '
            'canonical station identity. Changes if the source corrects the coordinates.'
        ),
    },
    STATION_NAME_VAR: {'long_name': 'station name'},
    STATION_REF_VAR: {
        'long_name': 'source station reference identifier',
        'comment': (
            "The data provider's own native station id (the source's site key); the stable "
            'join key back to the provider records.'
        ),
    },
    STATION_ALTITUDE_VAR: {
        'long_name': 'station altitude',
        'standard_name': 'altitude',
        'units': 'm',
        'valid_min': _ALTITUDE_MIN,
        'valid_max': _ALTITUDE_MAX,
        'comment': (
            'Station elevation as reported by the source; survey method and precision vary between '
            'stations and are not characterised. Values outside the valid range are treated as missing.'
        ),
    },
}


def _encodable_range(dt) -> tuple:
    """The decodable value interval of a packed dtype; (None, None) when unpacked."""
    enc = getattr(dt, 'dtype_encoded', None)
    if enc is None or dt.precision is None or dt.offset is None:
        return None, None
    factor = 10**dt.precision
    info = np.iinfo(enc)
    return (1 / factor) + dt.offset, (info.max / factor) + dt.offset


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


def _points_ids_names(stations: dict):
    """Station geometry + labels, in ``stations`` order.

    The points are CANONICALISED (``envlib.canonical_station_point``) before they leave this
    function, and that is not cosmetic: cfdb stores a point coordinate by encoding it with
    ``shapely.to_wkt(..., rounding_precision=5)``, whose ``trim=True`` default rounds the
    shortest decimal string half-to-even, while ``compute_station_id`` rounds the binary value
    via ``wkt.dumps`` (``trim=False``). They disagree on ~9% of coordinates supplied to 6
    decimal places, so writing the RAW point stores a geometry that no longer derives the
    ``station_id`` written beside it and the dataset can never be published — which is exactly
    how a live ECan publish failed on 2026-08-24. Canonicalising first makes the store's
    round-trip a fixed point. Ids are unaffected (the canonical point is what they already
    hashed), so this changes nothing for any station already stored.
    """
    points = [
        envlib.canonical_station_point(shapely.Point(float(d['lon']), float(d['lat']))) for d in stations.values()
    ]
    ids = np.array([envlib.compute_station_id(p) for p in points], dtype=object)
    names = np.array([str(d['name']) for d in stations.values()], dtype=object)
    refs = np.array([str(r) for r in stations], dtype=object)

    # Two stations inside the same ~1 m 5-dp cell collapse to ONE station_id. That is the
    # id contract, not a defect here — but it must be said out loud. Without this, the build
    # dies in cfdb's coord-uniqueness check ('The data for coords must be unique.'), which
    # names neither station, and before canonicalisation it did something worse: wrote two
    # rows carrying the same id and failed much later inside catalogue validation.
    if len(set(ids)) != len(ids):
        seen: dict = {}
        clashes = []
        for ref, sid, p in zip(refs, ids, points, strict=True):
            if sid in seen:
                first_ref, first_p = seen[sid]
                clashes.append(f'{first_ref!r} ({first_p.x}, {first_p.y}) and {ref!r} ({p.x}, {p.y}) -> {sid}')
            else:
                seen[sid] = (ref, p)
        msg = (
            'stations collapse to the same station_id (identical to 5 decimal places, ~1 m): '
            + '; '.join(clashes)
            + '. A station_id IS its rounded location, so these are one station to envlib — '
            'drop one, or correct the coordinates if they are wrong.'
        )
        raise ValueError(msg)
    return points, ids, names, refs


def _altitudes(stations: dict) -> np.ndarray:
    """Per-station altitude in metres (unpacked float32), NaN where a station lacks one; order
    matches the dict. A missing/None/NaN value -> NaN. A numeric value outside the plausibility
    band ``[_ALTITUDE_MIN, _ALTITUDE_MAX]`` (incl ±inf, float32 overflow, and sentinels like
    -9999) is treated as MISSING -> NaN with a station-naming warning — the same "the declared
    range doubles as QC" rule the data var uses. A NON-coercible / wrong-type value RAISES naming
    the station: that is an adapter bug (clean floats/None are expected), not a sensor sentinel."""
    out, rejected = [], []
    for ref, d in stations.items():
        v = d.get('altitude')
        if v is None:
            out.append(np.nan)
            continue
        try:
            f = float(v)
        except (TypeError, ValueError) as e:
            msg = f'station {ref!r}: invalid altitude {v!r}'
            raise ValueError(msg) from e
        if np.isnan(f):
            out.append(np.nan)
        elif _ALTITUDE_MIN <= f <= _ALTITUDE_MAX:
            out.append(f)
        else:  # sentinel / ±inf / float32 overflow -> missing (not real)
            rejected.append(ref)
            out.append(np.nan)
    if rejected:
        shown = rejected if len(rejected) <= _MAX_WARN_REFS else [*rejected[:_MAX_WARN_REFS], '...']
        logger.warning(
            'altitude: %d station(s) with out-of-range altitude set to NaN (outside [%g, %g] m): %s',
            len(rejected),
            _ALTITUDE_MIN,
            _ALTITUDE_MAX,
            shown,
        )
    return np.array(out, dtype='float32')


def _apply_station_attrs(ds) -> None:
    """Idempotently stamp the canonical CF attrs onto whatever station vars exist in ``ds``.
    Unconditional ``update`` is safe: cfdb's attrs finalizer writes to the store only when the
    attrs actually change, so re-running on every merge dirties nothing and re-uploads nothing."""
    for var, attrs in _STATION_VAR_ATTRS.items():
        if var in ds:
            ds[var].attrs.update(attrs)


def _check_refs(non_empty: dict, stations: dict) -> None:
    """A series ref absent from stations is a broken premise, not an operational hiccup:
    extraction is station-list-driven, and without metadata there is no geometry, no
    station_id, and no row to put the data in. Raise — never silently drop or skip."""
    missing = sorted(set(non_empty) - set(stations))
    if missing:
        msg = f'series refs missing from stations: {missing} — station metadata is required'
        raise ValueError(msg)
