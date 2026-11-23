# Lease-style lock expiry: incident record (moved from OPEN_WORK.md, 2026-11-24)

Moved verbatim from the OPEN_WORK.md item "Lease-style lock expiry — IN PROGRESS" to keep that item under its size limit. The item keeps the status, the decisions and the close-check.

  - **The second orphan (2026-11-22).** `uswest9`, a Contabo VPS and processor node, degraded for about 23 h and then froze. The ECan commit-object history shows healthy runs take 180–200 s, while runs on the degrading node took 30–40 min. Three successive slow runs each stalled one dataset earlier and left a complete `-0`/`-1` ticket: precipitation 13:24, gage-height 15:11 and streamflow 16:07 UTC. Every later run timed out. All three were broken by hand at 19:33 with the age-gated `break_other_locks()`, and fsck was clean on all three. The holders were stalled, not dead, so releasing the lock on shutdown would not have prevented it.
  - **History (the first orphan and the 2026-10-15 ruling).** On 2026-10-13 at 09:53 UTC a writer of `ecan-precipitation` died after its writability probe and left an exclusive lock ticket (and a harmless `_writable_probe_*` orphan); every later hourly run failed after the 300 s `lock_timeout`, and precipitation stopped updating. `fsck` found the remote consistent (110 objects, nothing missing). The ticket was broken by hand at 14 h 41 m old with the age-gated `S3Lock.break_other_locks()` (Mike's go, 2026-10-14); what killed the writer is unknown (the swarm logs were not read). ebooklet 0.10.5 and 0.11 behave alike: they time out but never break an orphan; only `force_lock=True` does, age-gated to tickets older than 2 hours, which the ebooklet changelog calls safe to use routinely. **Rejected for now (Mike, 2026-10-15): toolkit `force_lock=True`.** It relies on s3func's fixed 2 h `default_break_age`, which is arbitrary, and long holds can be legitimate. One incident with an unknown cause does not justify a reactive protocol change. **Direction when picked up:** the holder declares a maximum hold time in its ticket, and later acquirers may break a ticket past it. It belongs in ebooklet/s3func so every client inherits it (cfdb, envlib including the commons catalogue's own RCG lock, the toolkit, metservice and scripts). **First action:** read the swarm logs for the ECan task around 2026-10-13 09:53 UTC to find what killed the writer; if it was a kill signal, releasing the lock on shutdown may prevent most orphans without a lease. **Revisit trigger:** a second orphaned lock, or a cause known to recur. Meanwhile the failure stays loud: ebooklet's `TimeoutError` names the blocking lock's age and the manual remedy (`ebooklet/main.py:420-426`). The probe orphan is gone: the precipitation rechunk (rollout step 5) deleted that remote, and fsck of the republished one shows 0 orphans (2026-10-14).

## Releases (moved from OPEN_WORK.md, 2026-11-24)

    - **s3func 0.10.0** (leases, presigned `put_object(expires_at=)`) was code-reviewed in round `s3func-lease-code-1`, with every finding fixed. It is released and committed, and its wheel matches the commit. The provider probe (`benchmarks/lease_probe.py`) showed B2 and MEGA enforce presigned-PUT expiry and a signed `Content-MD5`, and that the fence holds with the client clock skewed.
    - **ebooklet 0.12.0** (the commit guard for grouped writers, plus `lock_lease=`) was code-reviewed in round `ebooklet-lease-code-1`, with every finding fixed, and released.
      - **ebooklet 0.13.0** (2026-11-24, a different session: the MongoDB read path) came out on top of it. The lease and guard code and its tests are unchanged in 0.13.0. That is checked by grep, not by running 0.13.0's suite in this session.
    - **envlib-ingest-base 0.8.0** (`lock_lease=` on `update_and_publish` / `build_and_publish`) is released and committed (`697b5f1`), and its wheel matches the commit.

## Step 5 build record (moved from OPEN_WORK.md, 2026-11-24, before review round `ecan-lease-code-1`)

    - **Step 5, ECan** (`~/git/envlib-repos/ingest/envlib-ingest-ecan-env`), is uncommitted, and the source is left for Mike's review:
      - `LOCK_LEASE` is in `raw/config.py`, with its derivation. `raw/ingest.py` passes it to both `update_and_publish` calls (the update and the heal).
      - **The first-run `build_and_publish` is deliberately unleased.** It runs only for a genuinely absent remote, and its supervised backfill can hold the lock longer than the lease, so a lease would abort it mid-push.
      - The floors are `envlib-ingest-base>=0.8.0` and `ebooklet>=0.12.0`, and the lock is refreshed.
      - `raw/tests/test_lease_call_sites.py` passes (4 tests, 0.5 s). It caught all its mutants: the lease dropped from each update site, a lease added to the backfill, a lease above the clearance bound (4000), and one below the floor (391; 392 passes).
      - **Image.** The base image `envlib-ingest-base:0.8.0` was NOT rebuilt. Mike built and pushed it on 2026-11-23 with ebooklet 0.12.0, the reviewed lease release, and overwriting a published tag with different contents was avoided. ECan's raw ingest does not use 0.13.0's MongoDB read path.
      - ECan `:0.11` is built locally `FROM` that image, with the tag bumped in `raw/docker-swarm.yml` and `raw/docker-compose.yml`. It is NOT pushed. Inside it, pip reports ebooklet 0.12.0, s3func 0.10.0 and toolkit 0.8.0. The modules import, `LOCK_LEASE` is 1500, and the call-site test passes.
      - **Planted orphan (close-check item 3), passed 2026-11-23 21:28 UTC** inside the `:0.11` image, on a scratch prefix in the B2 test bucket.
        - A process took a lease-392 ticket and was SIGKILLed.
        - The next writer (lease 1500) waited 455 s and broke exactly that ticket at an age of 454 s (threshold 452 s). Its own id carried the `l1500` suffix.
        - It then released, and 0 versions were left under the prefix.
        - Script: the session scratchpad's `planted_orphan.py`. It is not kept in the repo.
      - **Remaining:** push `:0.11`, then `docker stack deploy -c raw/docker-swarm.yml ecan-env-raw` (both on Mike's go), then the production checks.

## Review round `ecan-lease-code-1` (2026-11-24)

Code review of Step 5 (the ECan lease) by Fable 5.2 (high) and Sonnet 5.5 (xhigh), blind and in
parallel. Both arms drove the real `run_dataset` → toolkit → envlib → cfdb → ebooklet → s3func path
over moto. Reports: `~/git/ai-review-harness/results/ecan-lease-code-1/`.

| # | Finding | Verdict | Likelihood | Remedy (Mike, 2026-11-24) |
|---|---|---|---|---|
| 1 | The merge session in `update_and_publish` never pushes, so nothing checked its lock. A merge that stalled past its lease could resume after another run's commit and publish its stale result over it, returning normally. The unleased version failed loudly instead. | Verified: the source read, both arms executed it, and a rig of my own reproduced it (axis rolled back from 21:00 to 18:00). | Rare | Toolkit 0.8.1: check the lock before the merge session closes. On a lost lock, raise `LockLostError`, delete the working cache and publish nothing. |
| 2 | `skip-running=true` skips every fire while a task is `running` on a live node, or has a desired state other than `shutdown`. No run deadline, no missed-run alarm. | Verified against swarm-cronjob's source (`internal/docker/service.go`, `internal/worker/worker.go`, master). The deployed controller version was not checked. | Uncommon, pre-existing | Tracked: "ECan cron: a run deadline and a missed-run alarm". |
| 3 | A deploy that kills a `:0.10` run mid-session leaves an unleased ticket that no leased run breaks. | Verified (s3func never breaks unleased tickets; Sonnet executed it). | Per deploy, inside a run window | The deploy procedure in ECan's README. |
| 4 | The test passed a float lease and a toolkit that drops the lease. | Verified. | n/a | Real-class offline check, an int check, a toolkit version check, and the end-to-end test `raw/tests/test_merge_lock_lost.py`. |
| 5 | The next run clears an orphan only up to `o` ≈ 39 min. "~15 min" is a commit-to-commit gap, not a per-session hold. | Verified by arithmetic (`o < 2340 + a` s). | Uncommon | `config.py` wording. |
| 6 | A lease that expires mid-push leaves an orphan group object, which nothing in the cron sweeps. | Accepted: ebooklet's documented design, and both arms executed it. | Rare | None (fsck reports orphans). |
| 7 | The README overclaimed: the commons catalogue lock is still unleased. | Verified. | Rare, pre-existing | README and `config.py` wording. It is tracked as "Lease the commons catalogue". |
| 8 | A per-key dataset remote would make every leased run fail loudly. | Verified (both arms executed it). | Rare | README note: the datasets must stay grouped. |
| 9 | The Dockerfile comment named the wrong ebooklet. `rechunk_republish.py` named `:0.10`. | Verified. | Certain | Fixed. |

**Verification of the fixes:**
- **ECan:** `raw/tests/test_merge_lock_lost.py` (2 tests) and `test_lease_call_sites.py` (8 tests) pass in 0.8 s. They fail on each mutant:
  - no check at merge close;
  - verify ignored;
  - the cache not discarded;
  - `LOCK_LEASE` set to 1500.0, 391 or 4000;
  - the toolkit's `_lease_kw` returning `{}`.
- **Toolkit:** five mutants were all caught by `test_lock_lease_forwarding.py`:
  - no check;
  - no discard;
  - verify ignored;
  - the sidecar glob too broad;
  - the main file kept.

  The toolkit suite: 216 passed in 9.2 s.
