# envlib-ingest-base

Shared, tested toolkit for the envlib data-ingest family — and the Docker base image that carries
the envlib stack pre-installed. Ingest repos depend on this package and build `FROM` its image, so
the tricky, reusable machinery is written and tested **once** instead of copied per repo.

Two modules:

- **`resample`** — source-agnostic resampling of irregular / gappy station telemetry to a fixed cadence.
- **`tsortho`** — build + idempotent commons-update of station (`ts_ortho`) datasets.

The package is deliberately **pandas-free** (numpy + shapely + the envlib stack only). Data flows
through it as plain numpy arrays and dicts — the same shapes cfdb ingests — while pandas objects
still work as *inputs* by duck-typing (see below).

## Install / use from an ingest repo

```toml
# pyproject.toml
dependencies = ["envlib-ingest-base", ...]
# for local dev against a sibling checkout (drop once it's on PyPI):
[tool.uv.sources]
envlib-ingest-base = { path = "../envlib-ingest-base", editable = true }
```

```python
from envlib_ingest_base import (
    resample_mean, resample_sum, resample_station,   # resampling
    build_local, merge_dataset,                       # ts_ortho (cfdb-level)
    build_and_publish, update_and_publish,            # ts_ortho + commons publish
)
```

---

## Resampling (`resample`)

Turn a single station's raw `(times, values)` into a fixed-cadence series, returned as a plain
**`(times, values)` tuple**: ascending `datetime64[us]` **interval-start** labels (envlib
convention; pass UTC in, get UTC out) + `float64` values. Unpack immediately —
`t, v = resample_mean(...)` — and test emptiness as `t.size == 0` (never `len()`/truthiness on
the pair itself, which is always 2/truthy).

**Inputs** are anything convertible to `datetime64`: ISO-format strings, `datetime64` arrays of
any unit, python `datetime` objects, or pandas DatetimeIndex/Series (tz-aware converts to
UTC-naive) — no pandas required by the toolkit itself. Time is handled in **microseconds**
(python `datetime`'s native resolution; ±290k-year range, so overflow is a non-issue in
practice). `None`/`NaT`/`NaN` entries are dropped; non-numeric values raise rather than
coercing silently.

Two kinds, matching the two physical measurement types. Gaps are first-class: consecutive
observations farther apart than `max_gap` are a *hole* the signal is never interpolated across.
Both kinds are epoch-anchored (labels are multiples of `freq` since 1970-01-01) — identical to
pandas for any `freq` that divides 24 h.

Common parameters:

| param | default | meaning |
|---|---|---|
| `freq` | `'1h'` | output cadence: `'<n><unit>'` with unit in `h`, `min`, `s`, `D` (e.g. `'1h'`, `'15min'`), or a `np.timedelta64` / `datetime.timedelta`. Anything else raises. |
| `max_gap` | `None` | absolute hole threshold (same forms as `freq`, e.g. `'2h'`); `None` ⇒ adaptive (below). |
| `gap_multiplier` | `3.0` | adaptive threshold = `gap_multiplier × median(native interval)`, so a 5-min site → ~15 min and a 60-min site → ~3 h. |
| `min_coverage` | `0.5` | (`resample_mean` only) drop an interval covered less than this fraction. |
| `round_to` | `None` | the source's **true timestamp precision** (same forms as `freq`, e.g. `'1min'`). Timestamps are rounded to the nearest multiple before binning — a boundary reading jittered to `11:00:01` bins into the correct hour, and two jittery renderings of one reading collide and merge via the duplicate collapse instead of double-counting. |

### `resample_mean(times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0, min_coverage=0.5)`

Exact **trapezoidal time-weighted mean** for an instantaneous signal (river stage, flow). The signal
is treated as piecewise-linear between observations and integrated over each interval — exact for any
spacing, and equal to the plain hourly mean for regularly-sampled data.

```python
from envlib_ingest_base import resample_mean

# irregular spacing: 10 @ 00:00, 20 @ 00:15, 100 @ 01:00
times = ['2024-06-01 00:00', '2024-06-01 00:15', '2024-06-01 01:00']
t, v = resample_mean(times, [10.0, 20.0, 100.0])
# t -> ['2024-06-01T00:00']    (datetime64[us])
# v -> [48.75]                 # time-weighted (the 45-min stretch near 100 dominates),
#                              # NOT the equal-weight 37.5 a naive .resample().mean() gives
```

Gaps are dropped, never ramped across:

```python
times = ['2024-06-01 00:00', '2024-06-01 01:00', '2024-06-01 02:00',
         '2024-06-01 12:00', '2024-06-01 13:00']
t, v = resample_mean(times, [0.0, 0.0, 0.0, 120.0, 120.0])
# t -> ['2024-06-01T00:00', '2024-06-01T01:00', '2024-06-01T12:00']
# v -> [0.0, 0.0, 120.0]    # hours 02:00–11:00 are ABSENT (outage), not contaminated by 120
```

### `resample_sum(times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0)`

Right-closed interval **sum** for an accumulation signal (rainfall): an interval with **no**
reading is missing (absent from the output), never a fabricated `0`, while a genuinely reported
`0.0` is kept. A reading is attributed to the interval it *closes* (interval-start label).
Duplicate timestamps collapse by mean before summing (the same reading served twice must not
double-count).

```python
from envlib_ingest_base import resample_sum

# hourly rain totals; note the 03:00–06:00 outage
times = ['2024-06-01 01:00', '2024-06-01 02:00', '2024-06-01 07:00']
t, v = resample_sum(times, [0.0, 0.5, 1.0])
# t -> ['2024-06-01T00:00', '2024-06-01T01:00', '2024-06-01T06:00']
# v -> [0.0, 0.5, 1.0]    # reported 0.0 kept (a confirmed dry hour);
#                         # hours 02:00–05:00 ABSENT (no reading) — never fabricated as 0
```

### `resample_station(times, values, kind, **kwargs)`

Dispatch helper — `kind='mean'` → `resample_mean`, `kind='sum'` → `resample_sum`. Convenient when the
aggregation is config-driven:

```python
from envlib_ingest_base import resample_station
t, v = resample_station(times, values, cfg['kind'], gap_multiplier=3.0, min_coverage=0.5)
```

---

## Station datasets (`tsortho`)

Build and update envlib `ts_ortho` datasets — an orthogonal `(point, time)` layout: a geometry
coordinate of shapely Points (one per station), a shared dense hourly axis, one data variable named
after `meta.variable`, and `station_id` (envlib's deterministic hash) + `station_name` per station.

Two inputs recur:

- **`stations`** — a dict `{ref: {'lon': float, 'lat': float, 'name': str}}`. From a pandas
  frame indexed by ref with those columns: `df.to_dict('index')`.
- **`hourly`** — a dict `{ref -> (times, values)}` of resampled hourly tuples (the output of
  `resample_*`). Entries that are `None` or have `times.size == 0` are simply skipped.

```python
import envlib
from envlib_ingest_base import resample_mean

stations = {
    'A': {'lon': 172.5, 'lat': -43.5, 'name': 'River A at X'},
    'B': {'lon': 171.9, 'lat': -43.1, 'name': 'River B at Y'},
}
hourly = {'A': resample_mean(times_a, values_a), 'B': resample_mean(times_b, values_b)}
meta = envlib.Metadata(
    feature='waterway', variable='streamflow', method='sensor_recording', product_code=None,
    processing_level='raw', owner='ecan', aggregation_statistic='mean', frequency_interval='1h',
    utc_offset='+00:00', spatial_resolution='point', version='1',
    license='CC-BY-4.0', attribution='Data licenced by Environment Canterbury (CC-BY-4.0).',
)
```

### `build_local(path, meta, stations, hourly, *, variable, units, precision, min_value, max_value, chunk_shape=None, standard_name=None, extra_var_attrs=None)`

Create a fresh local ts_ortho cfdb. `precision` is **decimal places** (cfdb picks the int packing width
from `precision` + `min_value`/`max_value`); pass `standard_name` to override envlib's auto-derived CF
name. Returns the path.

```python
from envlib_ingest_base import build_local

build_local('flow.cfdb', meta, stations, hourly,
            variable='streamflow', units='m^3/s',
            precision=4, min_value=0, max_value=100_000)   # -> int32 packing, ~0.1 L/s resolution
```

The file validates against envlib directly:

```python
from envlib import Catalogue
Catalogue(remotes=[]).validate('flow.cfdb')   # {'metadata', 'dataset_version_id', 'state', ...}
```

### `merge_dataset(ds, stations, hourly, *, variable)`

Fold a recent window into an **open** ts_ortho dataset (cfdb `Dataset` or `EDataset`, opened `flag='w'`).
Appends genuinely-new stations and new hours, then does one contiguous read-modify-write over the
affected block, writing incoming values **only where they are non-NaN** — so a station that is offline
this window (or an hour that resamples to NaN) is never clobbered. Idempotent for a fixed input window.
Returns a small report dict.

```python
from cfdb import open_dataset
from envlib_ingest_base import merge_dataset

with open_dataset('flow.cfdb', flag='w') as ds:
    merge_dataset(ds, stations, hourly_recent, variable='streamflow')
    # -> {'new_stations': 0, 'new_hours': 24, 'written_block': 72}
```

### `build_and_publish(...)` / `update_and_publish(...)` — publish to the commons

Wrap the above with `envlib.Catalogue` + the ebooklet member/RCG connections.

```python
import ebooklet
from envlib import Catalogue
from envlib_ingest_base import build_and_publish, update_and_publish

member = ebooklet.S3Connection(access_key_id=..., access_key=..., bucket=..., endpoint_url=...,
                               db_key='datasets/ecan-streamflow',
                               db_url='https://b2.envlib.xyz/file/envlib/datasets/ecan-streamflow')
rcg = ebooklet.S3Connection(access_key_id=..., access_key=..., bucket=..., endpoint_url=...,
                            db_key='envlib-commons/catalogue')
cat = Catalogue(remotes=[rcg])

# FIRST run: build the full backfill locally, push data, then write the catalogue entry
build_and_publish(cat, 'flow.cfdb', member, rcg, meta, stations, hourly,
                  num_groups=17, variable='streamflow', units='m^3/s',
                  precision=4, min_value=0, max_value=100_000)

# LATER runs: pull the remote, merge a recent window, push the diff, refresh the entry.
# The remote is pulled each call, so `path` can be an ephemeral cache (stateless containers OK).
update_and_publish(cat, 'flow.cfdb', member, rcg, stations, hourly_recent,
                   variable='streamflow', num_groups=17)
```

`build_and_publish` takes the same keyword build args as `build_local`
(`variable`/`units`/`precision`/`min_value`/`max_value`/`standard_name`/…); `update_and_publish` only
needs `variable` (the dataset already exists). Both push the cfdb data **before** the catalogue entry,
and a crash between the two leaves the entry's advertised time-range safely *behind* the data (the next
run heals it).

---

## Develop

```bash
uv sync
uv run pytest                       # golden resample fixtures + numpy guardrails + ts_ortho idempotency
uv run ruff check . && uv run black --check .
```

Tests live in `envlib_ingest_base/tests/` and are excluded from the built wheel. pandas appears
only in the dev group — fixtures pin the promise that pandas objects keep working as inputs.

## Docker base image

```bash
docker build -t <namespace>/envlib-ingest-base:<tag> .   # tag recorded in docker-compose.yml
```

`requirements_base.txt` (scientific base) and `requirements_envlib.txt` (the envlib stack) are split
so a stack bump doesn't invalidate the heavy base layer. The image keeps pandas for downstream fetch
layers (CSV parsing) even though the toolkit itself doesn't need it. Downstream repos build `FROM`
this image.
