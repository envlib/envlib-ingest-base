"""envlib-ingest-base: shared toolkit for envlib data-ingest repos.

- ``resample``: source-agnostic time-series resampling (exact trapezoidal mean; accumulation sum).
- ``tsortho``: build + idempotent commons-update of station (ts_ortho) datasets.
- ``tsforecast``: build + append of point-forecast (ts_forecast) datasets, one run at a time.
- ``stations``: station identity/altitude helpers shared by both builders.
"""

from envlib_ingest_base.resample import resample
from envlib_ingest_base.tsforecast import build_and_push, merge_run, update_and_push
from envlib_ingest_base.tsforecast import build_local as build_forecast_local
from envlib_ingest_base.tsortho import build_and_publish, build_local, merge_dataset, update_and_publish

__version__ = '0.4.1'
__all__ = [
    'build_and_publish',
    'build_and_push',
    'build_forecast_local',
    'build_local',
    'merge_dataset',
    'merge_run',
    'resample',
    'update_and_publish',
    'update_and_push',
]
