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
    resample,                               # resampling (statistic keyed to envlib metadata)
    build_local, merge_dataset,             # ts_ortho (cfdb-level)
    build_and_publish, update_and_publish,  # ts_ortho + commons publish
)
```

---

## Resampling (`resample`)

Turn a single station's raw `(times, values)` into a fixed-cadence series, returned as a plain
**`(times, values)` tuple**: ascending `datetime64[us]` **interval-start** labels (envlib
convention; pass UTC in, get UTC out) + `float64` values. Unpack immediately —
`t, v = resample(...)` — and test emptiness as `t.size == 0` (never `len()`/truthiness on
the pair itself, which is always 2/truthy).

**Inputs** are anything convertible to `datetime64`: ISO-format strings, `datetime64` arrays of
any unit, python `datetime` objects, or pandas DatetimeIndex/Series (tz-aware converts to
UTC-naive) — no pandas required by the toolkit itself. Time is handled in **microseconds**
(python `datetime`'s native resolution; ±290k-year range, so overflow is a non-issue in
practice). `None`/`NaT`/`NaN` entries are dropped; non-numeric values raise rather than
coercing silently.

Two kinds, matching the two physical measurement types. Both are epoch-anchored (labels are
multiples of `freq` since 1970-01-01) — identical to pandas for any `freq` that divides 24 h.

### Missing data and aggregation rules

These rules are the core of the toolkit and deliberate (design rulings 2026-07-17/18); read
them before producing a dataset.

**The estimator spans beyond the aggregation interval and must be told where to stop
(`max_gap`); once stopped, each interval is judged by how much of it remains covered
(`min_coverage`).** The two rules are anchored to two different clocks and are NOT derivable
from each other:

1. **The gap rule — the data's clock.** The mean treats the signal as piecewise-linear
   between consecutive readings, regardless of interval boundaries — so it needs a rule for
   when interpolation stops being honest. That rule is a property of the *source's sampling
   design*, never of the output frequency: by default, a segment longer than
   `gap_multiplier ×` the **local median native interval** is a *hole* that is never
   interpolated across. The median is local (a rolling window of segments), so a station
   that logged 30-min data twenty years ago and 5-min data today is judged by each era's own
   cadence — a global median would misclassify one era wholesale. A 5-min site therefore
   gets a ~15-min threshold and a 60-min site ~3 h, at any output freq.
2. **The coverage rule — the output's clock.** After holes are excluded, an interval covered
   below `min_coverage` is **absent** rather than summarized from too little data. The
   default `0.75` follows common climatological completeness practice and is deliberately
   conservative: telemetry missingness is not random — outages correlate with the extreme
   events you most want recorded. For the mean, coverage is the non-hole time fraction; for
   the **sum**, each reading covers its accumulation slot, capped at **one local native
   interval** — a reading after skipped slots covers only its own slot, so losses are never
   credited as measured. An hour holding 8 of its 12 five-minute rain slots is *missing*,
   never published as a silent undercount, and a single-reading series has an unknowable
   slot and contributes no coverage (kept only when `min_coverage=0`).

Sums assume **per-slot totals** (each reading is the accumulation over its own reporting
slot). Feeds from totalizing gauges (since-last-report accumulations) must be differenced
upstream, and resampling accumulations *finer* than their native interval is not meaningful.

Common parameters:

| param | default | meaning |
|---|---|---|
| `freq` | `'1h'` | output cadence: `'<n><unit>'` with unit in `h`, `min`, `s`, `D` (e.g. `'1h'`, `'15min'`), the envlib CV code `'day'`, or a `np.timedelta64` / `datetime.timedelta`. Anything else — including `'0min'`, negative/NaT/calendar timedeltas — raises. |
| `max_gap` | `None` | absolute hole threshold (same forms as `freq`, e.g. `'2h'`); `None` ⇒ adaptive (the local-median rule above). |
| `gap_multiplier` | `3.0` | adaptive threshold = `gap_multiplier × local median(native interval)`. |
| `min_coverage` | `0.75` | both kinds: an interval covered below this fraction is absent. `0` disables. Float comparison at the boundary — prefer binary-friendly fractions (0.75, 0.5). |
| `round_to` | `None` | the source's **true timestamp precision** (same forms as `freq`, e.g. `'1min'`). Timestamps are rounded to the nearest multiple before binning — a boundary reading jittered to `11:00:01` bins into the correct hour, and two jittery renderings of one reading collide and merge via the duplicate collapse instead of double-counting. |

### `resample(times, values, *, statistic='mean', freq='1h', max_gap=None, gap_multiplier=3.0, min_coverage=0.75, round_to=None)`

One entry point; `statistic` selects both the aggregation and the signal model behind it — pass
the dataset's envlib **`aggregation_statistic`** metadata value, so the declared identity and the
computation can never drift (the same single-source-of-truth pattern tsortho uses for
`frequency_interval`). Future statistics of the instantaneous signal (median/min/max share its
segment machinery, differing only in the final reduction) will slot in without an API change;
unknown values raise.

#### `statistic='mean'` — instantaneous signals

Exact **trapezoidal time-weighted mean** for an instantaneous signal (river stage, flow). The signal
is treated as piecewise-linear between observations and integrated over each interval — exact for any
spacing. For grid-aligned regular data this is the trapezoid-rule mean (interval endpoints carry
half-weight): it coincides with the plain arithmetic mean for linear ramps but is NOT pandas'
left-closed `.resample().mean()`.

```python
from envlib_ingest_base import resample

# irregular spacing: 10 @ 00:00, 20 @ 00:15, 100 @ 01:00
times = ['2024-06-01 00:00', '2024-06-01 00:15', '2024-06-01 01:00']
t, v = resample(times, [10.0, 20.0, 100.0])
# t -> ['2024-06-01T00:00']    (datetime64[us])
# v -> [48.75]                 # time-weighted (the 45-min stretch near 100 dominates),
#                              # NOT the equal-weight 37.5 a naive .resample().mean() gives
```

Gaps are dropped, never ramped across:

```python
times = ['2024-06-01 00:00', '2024-06-01 01:00', '2024-06-01 02:00',
         '2024-06-01 12:00', '2024-06-01 13:00']
t, v = resample(times, [0.0, 0.0, 0.0, 120.0, 120.0])
# t -> ['2024-06-01T00:00', '2024-06-01T01:00', '2024-06-01T12:00']
# v -> [0.0, 0.0, 120.0]    # hours 02:00–11:00 are ABSENT (outage), not contaminated by 120
```

#### `statistic='sum'` — accumulations

Right-closed interval **sum** for an accumulation signal (rainfall): an interval with **no**
reading is missing (absent from the output), never a fabricated `0`, while a genuinely reported
`0.0` is kept. A reading is attributed to the interval it *closes* (interval-start label).
Duplicate timestamps collapse by mean before summing (the same reading served twice must not
double-count).

```python
# hourly rain totals; note the 03:00–06:00 outage
times = ['2024-06-01 01:00', '2024-06-01 02:00', '2024-06-01 07:00']
t, v = resample(times, [0.0, 0.5, 1.0], statistic='sum')
# t -> ['2024-06-01T00:00', '2024-06-01T01:00', '2024-06-01T06:00']
# v -> [0.0, 0.5, 1.0]    # reported 0.0 kept (a confirmed dry hour);
#                         # hours 02:00–05:00 ABSENT (no reading) — never fabricated as 0
```

Config-driven pipelines key both `statistic` and `freq` straight off the dataset's metadata:

```python
t, v = resample(times, values,
                statistic=cfg['metadata']['aggregation_statistic'],
                freq=cfg['metadata']['frequency_interval'])
```

---

## Station datasets (`tsortho`)

Build and update envlib `ts_ortho` datasets — an orthogonal `(point, time)` layout: a geometry
coordinate of shapely Points (one per station), a shared **dense fixed-step time axis at the
cadence declared by `meta.frequency_interval`**, one data variable named after `meta.variable`,
and per-station metadata as auxiliary `(point,)` variables — the established nomenclature for
station data in envlib ts_ortho datasets:

- `station_id` — envlib's deterministic *geometry* hash (`compute_station_id`); changes if the
  provider corrects a site's coordinates. Carries the CF `cf_role = "timeseries_id"` marking it as
  the timeseries instance identifier (cfdb writes the global `featureType = "timeSeries"`).
- `station_name` — the human-readable name.
- `station_ref` — the **source's native identifier** (the stations-dict key, e.g. an ECan site
  number): the stable join key back to the provider's own records.
- `station_altitude` — station elevation in metres (CF `standard_name = "altitude"`). **Optional**:
  created only when a source supplies a non-null `'altitude'` value in the stations dict (a source
  with none, e.g. ECan — or one whose altitude column is all-null — gets no such variable); NaN for
  any individual station lacking a value. Stored **as-reported** (unpacked float32, **no declared
  precision** — survey accuracy varies station-to-station and isn't characterised), and QC'd against
  a universal plausibility band (`valid_min = -500`, `valid_max = 9000` m, persisted as attrs);
  out-of-range values (incl `-9999`-type sentinels, ±inf) become missing with a warning, while a
  non-coercible value raises (that's an adapter bug, not a sentinel).

Each station var carries a CF `long_name` (and `station_id`/`station_ref` a `comment`); these
back-fill onto pre-existing datasets on the next update run. Station metadata is **written once**,
when a station first appears — it is not revised on later merges (correcting it needs a rebuild).
Version notes: `station_ref` exists in datasets built by toolkit >= 0.1.2 (merging *new stations*
into an older dataset raises — rebuild it — while revise-only merges keep working); the CF attrs
and `station_altitude` land from toolkit >= 0.1.4.

**The envlib metadata is the single source of truth for the cadence** — there is no `freq`
parameter. `frequency_interval` is a closed envlib vocabulary; only *fixed* codes work here
(`1min`…`30min`, `1h`…`12h`, `day` — `month`/`year`/`None` raise, a dense axis needs a fixed
duration). The time coordinate is stored in the cadence's **natural datetime64 unit** — a daily
dataset reads as `datetime64[D]`, hourly as `[h]`, 15-min as `[m]` — with an explicit step, so
the axis itself tells the reader the data's precision.

Two guards keep declared and actual cadence honest (both fail loud):

- *alignment* — every incoming timestamp must be an exact multiple of the declared step (the
  `resample_*` labels are, by construction, when resampled at the same freq the metadata declares);
- *phase* — a fixed cadence with a (reduced) `utc_offset` other than `+00:00` declares
  phase-anchored binning (e.g. local-midnight days) that the epoch-anchored resampler can't
  produce, and is rejected. Phase-anchored datasets are a planned follow-up (resampler `origin`).

Two documented limits: data resampled *coarser* than declared (daily labels are valid hour
multiples) builds a sparse-but-aligned axis the guards cannot detect; and string `.loc`/`truncate`
queries on a published dataset are truncated by numpy to the axis unit (`'…T06:00'` on a `[D]`
axis widens back to midnight — normal numpy semantics, worth knowing as a reader).

Two inputs recur:

- **`stations`** — a dict `{ref: {'lon': float, 'lat': float, 'name': str[, 'altitude': float]}}`.
  From a pandas frame indexed by ref with those columns: `df.to_dict('index')`. `altitude` is
  optional (metres); include it and the `station_altitude` variable is created automatically.
- **`series`** — a dict `{ref -> (times, values)}` of resampled tuples (the output of
  `resample_*` at the metadata's freq). Entries that are `None` or have `times.size == 0` are
  simply skipped.

```python
import envlib
from envlib_ingest_base import resample

stations = {
    'A': {'lon': 172.5, 'lat': -43.5, 'name': 'River A at X'},
    'B': {'lon': 171.9, 'lat': -43.1, 'name': 'River B at Y'},
}
series = {'A': resample(times_a, values_a), 'B': resample(times_b, values_b)}
meta = envlib.Metadata(
    feature='waterway', variable='streamflow', method='sensor_recording', product_code=None,
    processing_level='raw', owner='ecan', aggregation_statistic='mean', frequency_interval='1h',
    utc_offset='+00:00', spatial_resolution='point', version='1',
    license='CC-BY-4.0', attribution='Data licenced by Environment Canterbury (CC-BY-4.0).',
)
```

### `build_local(path, meta, stations, series, *, variable, units, precision, min_value, max_value, chunk_shape=None, standard_name=None, extra_var_attrs=None)`

Create a fresh local ts_ortho cfdb at the cadence of `meta.frequency_interval`. **Requires
cfdb >= 0.9.4** (older versions fabricate values when the packed encoder meets NaN or
out-of-range data — enforced at runtime). `min_value`/`max_value` are also the dataset's
**QC bounds** (values outside them, including ±inf, are stored as missing — pick physical
plausibility limits that cannot realistically be exceeded; persisted as `valid_min`/`valid_max`
attrs so merges apply the same filter). `precision` is
**decimal places** (cfdb picks the int packing width from `precision` + `min_value`/`max_value`);
pass `standard_name` to override envlib's auto-derived CF name. Returns the path.

```python
from envlib_ingest_base import build_local

build_local('flow.cfdb', meta, stations, series,
            variable='streamflow', units='m^3/s',
            precision=4, min_value=0, max_value=100_000)   # -> uint32 packing, ~0.1 L/s resolution
```

The file validates against envlib directly:

```python
from envlib import Catalogue
Catalogue(remotes=[]).validate('flow.cfdb')   # {'metadata', 'dataset_version_id', 'state', ...}
```

### `merge_dataset(ds, stations, series, *, variable)`

Fold a recent window into an **open** ts_ortho dataset (cfdb `Dataset` or `EDataset`, opened `flag='w'`).
The cadence is read back from the dataset's own envlib attrs and cross-checked against the stored
axis (epoch-aligned origin, dense constant step — mismatches raise). Appends genuinely-new stations
and new steps, then does one contiguous read-modify-write over the affected block, writing incoming
values **only where they are non-NaN** — so a station that is offline this window (or an interval
that resamples to NaN) is never clobbered. Idempotent for a fixed input window; a window entirely
before the axis start raises (no history prepending). Returns a small report dict.

```python
from cfdb import open_dataset
from envlib_ingest_base import merge_dataset

with open_dataset('flow.cfdb', flag='w') as ds:
    merge_dataset(ds, stations, series_recent, variable='streamflow')
    # -> {'new_stations': 0, 'new_steps': 24, 'written_block': 72, 'gap_steps': 0,
    #     'qc_rejected': 0, 'dropped_before_axis': 0}
    # gap_steps: steps left unfilled behind this window (pipeline downtime) — holes are NaN,
    #            so refetching a wider window and re-merging heals them
    # qc_rejected: values outside the encodable range set to NaN by the min/max QC
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
build_and_publish(cat, 'flow.cfdb', member, rcg, meta, stations, series,
                  variable='streamflow', units='m^3/s',
                  precision=4, min_value=0, max_value=100_000)

# LATER runs: pull the remote, merge a recent window, push the diff, refresh the entry.
# The remote is pulled each call, so `path` can be an ephemeral cache (stateless containers OK).
update_and_publish(cat, 'flow.cfdb', member, rcg, stations, series_recent,
                   variable='streamflow')
```

**Remote storage layout**: by default (`num_groups=None`) each chunk is its own S3 object —
the right choice for continuously-updated ts_ortho datasets (tens of keys; every hourly
push/pull moves exactly the changed chunk and nothing else). Grouped storage is a
request-batching optimization for **large, rarely-updated archives** (thousands of keys —
ebooklet's guidance is 10–100 MB per group); pass a prime `num_groups` at first publish only
for that shape. The choice is fixed once the dataset is first pushed.

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
# run AFTER this version is on PyPI (export so compose sees it)
export TOOLKIT_VERSION=$(sed -n "s/^__version__ = '\(.*\)'/\1/p" envlib_ingest_base/__init__.py)
docker compose build
docker compose push
```

The image installs the **released package from PyPI** (`TOOLKIT_VERSION` build arg) — the
package's own pyproject is the single source of truth for dependencies, so there are no
separately-maintained requirements files to drift out of sync (the cfdb >= 0.9.4 floor rides in
via `requires_dist`). The image is pandas-free, like the toolkit: downstream repos build `FROM`
this image and install their own extra runtime deps (e.g. the ECan repo adds pandas for its CSV
fetch layer).
