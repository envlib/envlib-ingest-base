"""envlib-ingest-base: shared toolkit for envlib data-ingest repos.

- ``resample``: source-agnostic time-series resampling (exact trapezoidal mean; accumulation sum).
- ``tsortho``: build + idempotent commons-update of station (ts_ortho) datasets.
"""

from envlib_ingest_base.resample import resample
from envlib_ingest_base.tsortho import build_and_publish, build_local, merge_dataset, update_and_publish

__version__ = '0.1.4'
__all__ = [
    'build_and_publish',
    'build_local',
    'merge_dataset',
    'resample',
    'update_and_publish',
]
