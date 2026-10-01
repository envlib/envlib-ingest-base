"""cfdb 0.10 rollout close-check: report the recorded compression of every dataset the rollout touches.

Read-only. Run it inside the READER image so the probe also proves that image opens a
byte-shuffled dataset (the WRF 3 km precipitation is `zstd_shuffle`; everything an existing
writer appends to should stay `zstd`):

    docker run --rm -v $PWD/planning/cfdb010_compression_probe.py:/probe.py:ro \
        --entrypoint python mullenkamp/flow-forecast-app-envlib:0.6 /probe.py

Baseline 2026-10-02 (before any writer ran on cfdb 0.10): WRF 3 km `zstd_shuffle`; the three ECan
raw datasets, ESA SST, the MetService point forecasts and the flow forecasts all `zstd`.
"""
import tempfile, pathlib, cfdb, ebooklet, envlib, numpy as np
tmp = pathlib.Path(tempfile.mkdtemp())
def report(label, ds):
    dv = [v for v in ds.data_var_names][0]
    var = ds[dv]
    sel = tuple(slice(0, 1) for _ in var.coord_names)
    n = np.asarray(var[sel].data).size
    print(f'{label:45s} compression={ds.compression:13s} read {dv}{list(var.coord_names)} ok ({n} value)')
for ref in envlib.Catalogue().datasets:
    m = ref.metadata
    with ref.open(file_path=tmp / f"{m['dataset_id']}.cfdb") as ds:
        report(f"{m['owner']}/{m['processing_level']}/{m['variable']}", ds)
for label, url in [('metservice point-forecasts (rain_db_url)', 'https://b2.tethys-ts.xyz/file/metservice/envlib/point-forecasts'),
                   ('flow-forecasts streamflow (forecast_db_url)', 'https://b2.tethys-ts.xyz/file/point-forecasts/envlib/flow-forecasts/streamflow')]:
    with cfdb.open_edataset(ebooklet.S3Connection(db_url=url), tmp / (label.split()[0] + '.cfdb'), flag='r') as ds:
        report(label, ds)
