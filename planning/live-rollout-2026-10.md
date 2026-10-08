# Live rollout of ebooklet 0.11 and the ECan telemetry layout (October 2026)

> **Status:** live reference for the open checklist in `../OPEN_WORK.md` (the first Backlog item, "Roll ebooklet 0.11
> (storage format 3) and the telemetry layout out to the live datasets"). That item holds the ordered steps; this
> file holds their detail, moved here verbatim on 2026-10-09 so the item stays under the 6,000-character budget
> (global CLAUDE.md §2). Step numbers below are the item's.

## Step 3 — republish the commons catalogue per-key

       - Republish the commons catalogue **per-key** (`group_bytes=None`, format 2) FROM ITS HYDRATED FILE, so it keeps its uuid: hydrate with `planning/write-order-groups-migration/hydrate_and_delete.py` under 0.10.5, delete, republish under 0.11. A per-key format-2 remote is readable by both 0.10 and 0.11 (`docs/ops.md`, and Fable ran both versions), so there is no cutover window. It must happen BEFORE esa-sst is republished: otherwise that publish recreates the catalogue with a new uuid (and grouped), and `Catalogue.refresh` shows every cached reader an empty catalogue (envlib `OPEN_WORK.md`, the `[envlib]` `Catalogue.refresh` item). The live catalogue is hash-grouped (`format_version 2`, `num_groups 13`, read from its public headers on 2026-10-08), which neither 0.10 nor 0.11 can bridge.
       - **Step 4 progress (2026-10-08):** `hydrate_and_delete.py` dry runs PASSED, deleting nothing, for esa-sst (127,611 keys, same uuid and commit stamp as the remote, nothing downloaded) and for the commons catalogue (16 entries, hydrated into a fresh scratch copy). That copy is in a session scratchpad and will not last. The real run hydrates again, into a durable path Mike has not yet chosen (perhaps beside `~/.envlib/commons/`, under a dated name). Never use the stale `~/.envlib/commons/catalogue.rcg`. Both dry runs are re-checked by the real run anyway, which compares against a fresh download of the db object immediately before deleting.

## Step 4 — deploy the readers

       - **Image status (checked on Docker Hub 2026-10-09):**
         - base `0.5.0` and `0.6.0` are pushed;
         - ECan raw `:0.9` (tag bumped in `raw/docker-compose.yml` and `raw/docker-swarm.yml`) is built locally (Mike's workstation, on base 0.5.0, so ebooklet 0.11.0 without the index prune) but NOT pushed: the tag returns 404 and Hub's newest tag is `0.8`;
         - `:0.10` and flow-forecast-app `:0.7` are not built or pushed;
         - the ECan service still runs `:0.8` (Mike, 2026-10-09: not redeployed yet).
       - `:0.9` is also the writer of the ECan datasets, so until step 5 its hourly commits still upload unpruned indexes, as `:0.8`'s do. Only `:0.10` (base 0.6.0, ebooklet 0.11.1) prunes, and it is deployed per dataset AFTER that dataset's rechunk (step 5), not before.

## Step 5 — rechunk each ECan dataset under supervision (precipitation, then gage height, then streamflow)

       - stop the ECan service well before the run and keep it stopped through the post-check (run condition P1);
       - `PYTHONPATH=.:raw uv run python raw/rechunk_republish.py --dataset <d> --dry-run` and read the gate report;
       - the same with `--yes` (after any failure, `--resume`);
       - deploy `:0.10` and restart the service;
       - check the first push against the measured expectation: ~2–5 group PUTs plus the index, and up to 7 for ~2 days after a rollover;
       - check that flow-forecast-app shows data newer than the republish without a restart.

## Step 7 — the 12 WRF-3k datasets

    7. **The 12 WRF-3k datasets, from the machine holding their archives.** The write-order-groups plan's step 4 adds (2026-10-07): re-run the wrf-3k gates, because hydration changes file size and mtime and the gate is bound to them; and precipitation, soil moisture and altitude may lack band manifests (unverified there), so check that first — see the `[ingest/wrf-3k]` item below, "Manifests not on GitHub".

## Scope, close-checks, gaps and design facts

  - **Out of the migration:** the MEGA `era5_cfdb` remote is format 1, which 0.10.x cannot read either, and the local `era5_d01.cfdb` is a different, partial database (91 keys; it also carries an unclean-close flag). Mike: something else is used for ERA5. Nothing exists at MEGA `sst_cfdb`'s configured key; esa-sst moved to the `ecmwf-data` bucket on 2026-07-28.
  - **Close-checks per ECan dataset:** the dry-run gate report; the script's post-check (orphan sweep, fsck clean, a full compare of a fresh read against the backup, the warm-reader read and the public URL); the first hourly push; a two-week read of one station costing ~1–3 GETs.
  - **Not verified yet:**
    - the ebooklet and envlib live suites on the released 0.11.1 stack (they need Mike's go);
    - anything on B2 or behind the CDN;
    - real compressibility and irregular station reporting.
  - **Design facts to keep:**
    - The rebuilt file keeps the old uuid (it is created from the remote's header bytes). Warm reader caches then follow the change by themselves, and ebooklet's uuid check keeps its full protection. The earlier idea of a reader self-heal was dropped because it deleted only-copy files.
    - Because the uuid is kept, the script checks that the remote is ABSENT before it publishes. Without that check the rebuilt file is pushed into the old remote, which silently stays per-key with stale old-layout objects (fake S3: 33 index keys against 21).
    - A publish that fails part-way is live but partial (readers see fill values with no error). `--resume` completes it and sweeps the orphans.
    - Warm readers follow only if the new keys' timestamps are newer than the old ones, so a writer with a skewed clock breaks it. Run condition P1 covers this rather than code (Mike, D-d).
    - The rebuilt datasets use `zstd_shuffle` on purpose (the live ones are `zstd`; D-a), so readers need cfdb ≥ 0.10.
