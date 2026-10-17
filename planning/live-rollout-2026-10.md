# Live rollout of ebooklet 0.11 and the ECan telemetry layout (October 2026)

> **Status: COMPLETE (2026-10-18), kept as the record.** The checklist this file details, "Roll ebooklet 0.11
> (storage format 3) and the telemetry layout out to the live datasets", is closed in `../OPEN_WORK_DONE.md`; its
> step 8 continues as the open item "Delete the 2026-10 rollout backups after a soak" in `../OPEN_WORK.md`. This
> file holds the steps' detail, moved here verbatim on 2026-10-09 so the item stayed under the 6,000-character
> budget (global CLAUDE.md §2). Step numbers below are the item's.

## Step 3 — republish the commons catalogue per-key

       - Republish the commons catalogue **per-key** (`group_bytes=None`, format 2) FROM ITS HYDRATED FILE, so it keeps its uuid: hydrate with `planning/write-order-groups-migration/hydrate_and_delete.py` under 0.10.5, delete, republish under 0.11. A per-key format-2 remote is readable by both 0.10 and 0.11 (`docs/ops.md`, and Fable ran both versions), so there is no cutover window. It must happen BEFORE esa-sst is republished: otherwise that publish recreates the catalogue with a new uuid (and grouped), and `Catalogue.refresh` shows every cached reader an empty catalogue (envlib `OPEN_WORK.md`, the `[envlib]` `Catalogue.refresh` item). The live catalogue is hash-grouped (`format_version 2`, `num_groups 13`, read from its public headers on 2026-10-08), which neither 0.10 nor 0.11 can bridge.
       - **Step 4 progress (2026-10-08):** `hydrate_and_delete.py` dry runs PASSED, deleting nothing, for esa-sst (127,611 keys, same uuid and commit stamp as the remote, nothing downloaded) and for the commons catalogue (16 entries, hydrated into a fresh scratch copy). That copy is in a session scratchpad and will not last. The real run hydrates again, into a durable path Mike has not yet chosen (perhaps beside `~/.envlib/commons/`, under a dated name). Never use the stale `~/.envlib/commons/catalogue.rcg`. Both dry runs are re-checked by the real run anyway, which compares against a fresh download of the db object immediately before deleting.

       - **Done 2026-10-14 (moved from `OPEN_WORK.md` on 2026-10-15):** Re-hydrated into `~/.envlib/commons/catalogue-hydrated-20261014.rcg` (16 of 16 entries), backed up server-side to `envlib-commons/catalogue-backup-20261014` (hash-grouped, 0.10-readable; delete after the soak with the other backups), deleted, and republished per-key by `ebooklet/planning/write-order-groups-migration/republish_per_key.py` (new; `group_bytes=None` explicit): format 2, no groups, uuid `019f5a67-…` kept, all 16 entries identical to a baseline taken with the service stopped. Rehearsed first end to end on a scratch copy (including a mutant without `group_bytes=None`, which republished format 3 and FAILED verification). Read back by envlib 0.1.8/ebooklet 0.11.1 (`Catalogue()`: 16) and by ebooklet 0.10.5 (16, identical).

## Step 4 — deploy the readers

> **2026-10-15:** `:0.9` is skipped and `:0.10` moves to base 0.7.0 (Mike). The image status below is the
> 2026-10-09 snapshot. The current status is in the item's steps 2 and 4.

       - **Image status (checked on Docker Hub 2026-10-09):**
         - base `0.5.0` and `0.6.0` are pushed;
         - ECan raw `:0.9` (tag bumped in `raw/docker-compose.yml` and `raw/docker-swarm.yml`) is built locally (Mike's workstation, on base 0.5.0, so ebooklet 0.11.0 without the index prune) but NOT pushed: the tag returns 404 and Hub's newest tag is `0.8`;
         - `:0.10` and flow-forecast-app `:0.7` are not built or pushed;
         - the ECan service still runs `:0.8` (Mike, 2026-10-09: not redeployed yet).
       - `:0.9` is also the writer of the ECan datasets, so until step 5 its hourly commits still upload unpruned indexes, as `:0.8`'s do. Only `:0.10` (base 0.6.0, ebooklet 0.11.1) prunes, and it is deployed per dataset AFTER that dataset's rechunk (step 5), not before.

       - **flow-forecast-app-envlib `:0.7` (moved from `OPEN_WORK.md` on 2026-10-15):** The app ran `:0.5` (ebooklet 0.10.4). Its *warm* cached catalogue (`/root/.envlib/cache/*.rcg`) still described the hash-grouped incarnation, so every refresh from 00:58:19Z raised `RemoteIntegrityError` (`12.<unmanifested>` missing) and the page served the 00:43Z snapshot for ~17 h while `/healthz` stayed `ok`. A cold `:0.5` read the republished catalogue fine; step 3's 0.10.5 read-back was also cold, which is why it could not see this. `:0.7` built its first snapshot in 26 s. **Before step 5 or 6 republishes anything, check that every LONG-RUNNING reader is on ebooklet ≥ 0.11.** The cron jobs (model, ECan, metservice) start cold every run and were unaffected.

       - **Moved from `OPEN_WORK.md` on 2026-10-18 (steps 2 and 4):**
         - ECan `:0.10`: Hub digest `sha256:ca9ec234a8a1…` matches the local build, which holds ebooklet 0.11.1 and toolkit 0.7.0.
         - `:0.10` merging into a per-key remote WORKED in its first runs (2026-10-14 ~22:11 UTC: streamflow and gage height updated, gage height pruned 164 dead items).
         - Model `:0.5`: the old `:0.4` (ebooklet 0.10.4) cannot read a format-3 grouped remote, so its first run after a rechunk would fail. `model/Dockerfile` now floors `envlib-ingest-base>=0.7.0`. The image holds ebooklet 0.11.1, cfdb 0.11.0, envlib 0.1.8, toolkit 0.7.0 and the sidecar's model pins; Hub's `:0.5` digest `sha256:9ac059de6531…` matches the local build.

## Step 5 — rechunk each ECan dataset under supervision (precipitation, then gage height, then streamflow)

       - stop the ECan service well before the run and keep it stopped through the post-check (run condition P1);
       - **the dry run needs the pause too** (found 2026-10-15, read from the code): its `hydrate` opens the live remote with `flag='w'` (`raw/rechunk_republish.py`, `hydrate`), so it holds the dataset's write lock while it downloads every object. It changes nothing remote, but a concurrent cron run would time out, and a crash mid-dry-run would leave an orphan lock ticket (the 2026-10-13 failure). Pause once and run the dry run and `--yes` in the same window;
       - `PYTHONPATH=.:raw uv run python raw/rechunk_republish.py --dataset <d> --dry-run` and read the gate report;
       - the same with `--yes` (after any failure, `--resume`);
       - deploy `:0.10` and restart the service;
       - check the first push against the measured expectation: ~2–5 group PUTs plus the index, and up to 7 for ~2 days after a rollover;
       - check that flow-forecast-app shows data newer than the republish without a restart.

       - **Precipitation, moved 2026-10-14 20:58–21:00 UTC (moved from `OPEN_WORK.md` on 2026-10-15):** Independent read-back: format 3, 1 group object, uuid `019f7ebc-…e18e` kept; fsck shows 0 missing, 0 orphans and no unmanifested groups. A public read equals the pre-move hydrated copy (`compare_datasets`), and the catalogue entry runs to 2026-10-14T19. The backup `ecan-precipitation-backup-20261014` is format 2, with 110 objects and fsck clean. The run started ~5.5 min after the last cron commit (20:53:16), not P1's hour; the script's warm-reader post-check passed, and this machine's clock is NTP-synced.

       - (moved from `OPEN_WORK.md` on 2026-10-15)
       - **Precipitation MOVED 2026-10-14 20:58–21:00 UTC (Mike, `--yes`).** Read-back and timing: `planning/live-rollout-2026-10.md`, step 5. `:0.10` is deployed and the cron re-enabled (the `--label-add …enable=false` pause survived the stack deploy; re-enable it explicitly). **Open: the first precipitation push on the grouped layout has not happened.** From ~22:11 UTC ECan's rainfall summary endpoint (`/data/51/…`) returned HTTP 500 after ~30 s to every caller (reproduced with plain `curl`; the water endpoint was fine). Precipitation fails at `discover_stations` before touching the remote, so the gap heals on the first run after ECan recovers, within the 1-week window. Mike tracks the failure emails. Close-check: `time_end` advances, fsck stays clean, and the app shows the new data without a restart. **Gage height and streamflow wait for that push:** pause once, dry-run both, then `--yes` each a full hour after the last cron commit.

       - **All three moved (moved from `OPEN_WORK.md` on 2026-10-18):** Independent read-back after the 01:49 UTC cron: all three are format 3 with their uuids kept and fsck clean; gage height 1,552 → 235 objects, streamflow 1,090 → 190; 5 sample stations per dataset equal their backup up to 7 days before the end.

## Step 7 — the 12 WRF-3k datasets

    7. **The 12 WRF-3k datasets, from the machine holding their archives.** The write-order-groups plan's step 4 adds (2026-10-07): re-run the wrf-3k gates, because hydration changes file size and mtime and the gate is bound to them; and precipitation, soil moisture and altitude may lack band manifests (unverified there), so check that first — see the `[ingest/wrf-3k]` item below, "Manifests not on GitHub".

       - **Done 2026-10-18 (steps 6 and 7):** record in `OPEN_WORK_DONE.md`, "Rollout steps 6 and 7"; per-dataset sizes and groups in wrf-3k `PROVENANCE.md`.
       - **Superseded 2026-10-18:** the hydrate-and-re-gate procedure above is replaced by the approved runbook `~/.claude/plans/flickering-mixing-balloon.md` (delete, `publish.py`, read-back; no hydration, no backups: nobody uses these datasets and a rebuild is the fallback, Mike).
       - **Moved from `OPEN_WORK.md` on 2026-10-18 (steps 1 and 6):**
         - The uncommitted ports: wrf-3k (`config.py`, `scripts/build.py`, `scripts/publish.py`, `scripts/verify_build.py`, `dev/plants.py`, `dev/scratch_cycle.py`, `pyproject.toml`, `uv.lock`, README and PROVENANCE) and esa-sst (`config.py`, `scripts/publish.py`, `pyproject.toml` and `uv.lock`).
         - The move itself is tested at the ebooklet level (`test_republish_in_place_after_delete`: same key, uuid kept, an old reader cache heals).

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
