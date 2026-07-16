# envlib-ingest-base

Shared, tested toolkit for the envlib data-ingest family — and the Docker base image that carries
the envlib stack pre-installed. Ingest repos depend on this package and build `FROM` its image, so
the tricky, reusable machinery is written and tested **once** instead of copied per repo.

Two modules:

- **`resample`** — source-agnostic resampling of irregular / gappy station telemetry to a fixed cadence.
- **`tsortho`** — build + idempotent commons-update of station (`ts_ortho`) datasets.

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

Turn a single station's raw `(times, values)` into a fixed-cadence `pandas.Series` indexed by
**interval start** (envlib convention; pass UTC in, get UTC out). Two kinds, matching the two
physical measurement types. Gaps are first-class: consecutive observations farther apart than
`max_gap` are a *hole* the signal is never interpolated across.

Common parameters:

| param | default | meaning |
|---|---|---|
| `freq` | `'1h'` | output cadence (any pandas offset). |
| `max_gap` | `None` | absolute hole threshold (e.g. `'2h'`); `None` ⇒ adaptive (below). |
| `gap_multiplier` | `3.0` | adaptive threshold = `gap_multiplier × median(native interval)`, so a 5-min site → ~15 min and a 60-min site → ~3 h. |
| `min_coverage` | `0.5` | (`resample_mean` only) drop an interval covered less than this fraction. |

### `resample_mean(times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0, min_coverage=0.5)`

Exact **trapezoidal time-weighted mean** for an instantaneous signal (river stage, flow). The signal
is treated as piecewise-linear between observations and integrated over each interval — exact for any
spacing, and equal to the plain hourly mean for regularly-sampled data.

```python
import pandas as pd
from envlib_ingest_base import resample_mean

# irregular spacing: 10 @ 00:00, 20 @ 00:15, 100 @ 01:00
times = pd.to_datetime(['2024-06-01 00:00', '2024-06-01 00:15', '2024-06-01 01:00'])
resample_mean(times, [10.0, 20.0, 100.0])
# time
# 2024-06-01 00:00:00    48.75     # time-weighted (the 45-min stretch near 100 dominates),
#                                  # NOT the equal-weight 37.5 a naive .resample().mean() gives
```

Gaps are dropped, never ramped across:

```python
times = pd.to_datetime(['2024-06-01 00:00', '2024-06-01 01:00', '2024-06-01 02:00',
                        '2024-06-01 12:00', '2024-06-01 13:00'])
resample_mean(times, [0.0, 0.0, 0.0, 120.0, 120.0])
# 2024-06-01 00:00:00      0.0
# 2024-06-01 01:00:00      0.0
# 2024-06-01 12:00:00    120.0     # hours 02:00–11:00 are ABSENT (outage), not contaminated by 120
```

### `resample_sum(times, values, *, freq='1h', max_gap=None, gap_multiplier=3.0)`

Right-closed interval **sum** for an accumulation signal (rainfall), with `min_count=1`: an interval
with **no** reading is missing (dropped), never a fabricated `0`, while a genuinely reported `0.0` is
kept. A reading is attributed to the interval it *closes* (interval-start label).

```python
from envlib_ingest_base import resample_sum

# hourly rain totals; note the 03:00–06:00 outage
times = pd.to_datetime(['2024-06-01 01:00', '2024-06-01 02:00', '2024-06-01 07:00'])
resample_sum(times, [0.0, 0.5, 1.0])
# 2024-06-01 00:00:00    0.0     # reported 0.0 kept (a confirmed dry hour)
# 2024-06-01 01:00:00    0.5
# 2024-06-01 06:00:00    1.0     # hours 02:00–05:00 ABSENT (no reading) — never fabricated as 0
```

### `resample_station(times, values, kind, **kwargs)`

Dispatch helper — `kind='mean'` → `resample_mean`, `kind='sum'` → `resample_sum`. Convenient when the
aggregation is config-driven:

```python
from envlib_ingest_base import resample_station
hourly = resample_station(times, values, cfg['kind'], gap_multiplier=3.0, min_coverage=0.5)
```

---

## Station datasets (`tsortho`)

Build and update envlib `ts_ortho` datasets — an orthogonal `(point, time)` layout: a geometry
coordinate of shapely Points (one per station), a shared dense hourly axis, one data variable named
after `meta.variable`, and `station_id` (envlib's deterministic hash) + `station_name` per station.

Two inputs recur:

- **`stations`** — a `pandas.DataFrame` indexed by station ref, columns `lon`, `lat`, `name`.
- **`hourly`** — a dict `{ref -> pd.Series}` of resampled hourly series (the output of `resample_*`).
  Stations absent from `hourly` (or with an empty series) are simply skipped.

```python
import pandas as pd
import envlib
from envlib_ingest_base import resample_mean

stations = pd.DataFrame(
    {'lon': [172.5, 171.9], 'lat': [-43.5, -43.1], 'name': ['River A at X', 'River B at Y']},
    index=['A', 'B'],
)
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
uv run pytest                       # golden resample fixtures + ts_ortho idempotency
uv run ruff check . && uv run black --check .
```

Tests live in `envlib_ingest_base/tests/` and are excluded from the built wheel.

## Docker base image

```bash
docker build -t <namespace>/envlib-ingest-base:<tag> .   # tag recorded in docker-compose.yml
```

`requirements_base.txt` (scientific base) and `requirements_envlib.txt` (the envlib stack) are split
so a stack bump doesn't invalidate the heavy base layer. Downstream repos build `FROM` this image.
