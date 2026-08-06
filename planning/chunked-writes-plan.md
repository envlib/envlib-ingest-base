# Plan χ — chunked writes in `envlib-ingest-base`

> **Archived 2026-08-07. Status: IMPLEMENTED — shipped as `envlib-ingest-base` 0.3.0** (on PyPI
> 2026-08-06; base image `mullenkamp/envlib-ingest-base:0.3.0` on Docker Hub the same day).
>
> This is the plan of record for the chunked-write refactor, preserved here after implementation.
> It was **dual-blind reviewed** (round `chunked-1`, Fable 5 high + Gemini 3.1 Pro high) and §7a
> holds that record — including the finding both arms reached independently, which reversed my
> claim that the heal path was unreachable.
>
> It lived only in `~/.claude/plans/`, which is **not in git**, and that file has since been reused
> for the follow-on ECan precipitation-repair plan (α2). Without this copy the review record would
> have been lost. Convention: a reviewed plan is archived into `planning/` once implemented.
>
> **What actually shipped**, measured rather than predicted:
> - `build_local` writes one station row at a time — traced peak **1,506 MB → 10.9 MB** on a
>   400 × 150 k fixture, and *faster* (0.8 s vs 1.2 s). Output byte-identical at the stored-key level.
> - The real ECan streamflow build went **4.5 GB → 899 MB**, byte-identical (12,023 keys, 0
>   differing bytes).
> - `merge_dataset` became per-station too (**552.5 MB → 6.0 MB** with one backdated station), which
>   is beyond what §3 scoped — see the toolkit's `OPEN_WORK.md` Done entry.
> - Phase 0's heal-path horizon guard landed in `envlib-ingest-ecan-env/raw/ingest.py`, not here.
>
> **Phase 2 (the streaming `series` contract) was NOT implemented** and remains open in
> `OPEN_WORK.md`. §4 below is a design sketch, not a record of shipped work.
>
> ⚠️ **Two things below are now known to be wrong**, left in place because they are what the plan
> reasoned from. (1) `_prune_local`'s motivation cites a persistent `ecan_data` volume that **does
> not exist in production** — `raw/docker-swarm.yml` declares no volumes, so the swarm service is
> stateless and its working file lives in a discarded container tempdir; the prune is still correct,
> only its stated rationale is not, and it does apply to the local `docker-compose` dev path.
> (2) cfdb line citations are against **0.9.5-dev** while production resolves **0.9.4** — §3 already
> carries that caveat. Both were confirmed later, in round `ecan-alpha2-1`.

## Core

Building the ECan streamflow dataset peaked at **4.5 GB of RAM** for a file that is 51 MB on disk
and only 20 % populated. The cause is not incidental: `envlib-ingest-base` materialises the entire
dense `(station, time)` plane in memory — twice, once per variable — and then assigns it in a
single statement. cfdb reads and writes in **chunks**, and the default chunk shape here is already
`(1, 25_000)` — *one station row* — so the code allocates 1.18 GB to write something it could
write 4.5 MB at a time.

*(Post-review. Dual blind, both arms in, synthesised once — record in §7a. The review **withdrew one
exclusion**, **upgraded the verification gate**, and **corrected four factual claims**.)*

**Phase 0, added by review:** the live hourly cron's heal path (`ingest.py:154-157`) reaches
`merge_dataset` with **no recency guard** — its own comment says so. One backdated timestamp
widens the merge block across every station; an arm measured a 25× memory jump from a single
value. I had called this "verified not reachable"; **both arms refuted that independently.** It is
~5 lines to fix and goes first.

This plan then fixes the build path in two phases, ordered so the risky half comes second:

- **Phase 1 — internals only.** `build_local` writes row by row instead of assembling planes.
  **No API change, no version bump, all 78 existing tests pass unmodified, and the output dataset
  must be content-identical.** This alone removes ~88 % of the addressable peak (6.02 GB of
  groundwater's 6.80 GB).
- **Phase 2 — the streaming contract.** `series` becomes an iterator so callers stop materialising
  every station first. Breaking, version bump — but there is exactly **one consumer repo**.

**Nothing about the stored file changes.** Same bytes-on-disk semantics, same chunk shape, same
identity. This is a pure memory refactor of the write path, which is what makes the verification
strong: any content difference is a bug, full stop.

Ordering decided (Mike, 2026-08-06): **fix the toolkit before finishing α's publishes**, so the
remaining datasets are built by the fixed code — groundwater in particular, whose 10.2 GB build is
the worst case.

---

## 1. Context

Plan α's implementation surfaced this. `/usr/bin/time -v` on the streamflow build reported
**4.5 GB peak RSS**; a review arm measured **10.2 GB** for groundwater. Both were treated at the
time as numbers to check rather than as a design to question — including by both dual-blind review
arms, who carefully computed the memory of an allocation neither asked should exist.

The root cause was then found in three places at once and all three are now fixed, which is why
this plan is about code rather than documentation:

- the **`cfdb` skill** led its "Writing Data" section with `temp[:] = data_array` labelled
  *"Write all data at once"* — actively teaching the anti-pattern;
- `cfdb_summary_usage.md` (loaded by the Gemini review arm) had the identical defect;
- neither `CLAUDE.md` nor `GEMINI.md` carried the principle at all.

A `PostToolUse` hook now flags whole-array patterns in files touching cfdb/netCDF/xarray. **This
plan is the code half of the same correction.**

**Outcome:** memory scales with a chunk, not with the dataset. A source ten times larger than ECan
becomes a matter of runtime rather than of whether the machine has enough RAM.

---

## 2. What was measured (not assumed)

`_assemble` (`tsortho.py:353`) allocates `np.full((len(stations), n_times), np.nan, 'float64')` —
note `len(stations)`, the full discovery dict, so stations with no data still get an all-NaN row.
It is called **once per plane**, and the primary is still live when the ancillary is allocated.
`_nan_safe` (`:290`) then adds a full-plane bool mask **plus another full plane** at write time, so
`build_local:524` has ~17 bytes/cell live before cfdb's own float32 and packed-uint encode copies.

| dataset | stations × steps | one plane (f64) | ×2 planes | sparse dict | measured peak |
|---|---|---|---|---|---|
| streamflow | 261 × 565,824 | 1.18 GB | 2.36 GB | 0.72 GB | **4.5 GB** |
| groundwater_depth | 606 × 620,771 | 3.01 GB | 6.02 GB | 0.78 GB | **10.2 GB** |

**The split is what drives the phasing.** The dense planes are purely internal (Phase 1, no API
change); the sparse dict is the caller's `fetch_all` result and needs the contract to change
(Phase 2). Phase 1 alone addresses 6.02 of groundwater's 6.80 GB.

**There is already a correct reference implementation in-house.** `tethys-archiver`'s
`export.py:438-482` allocates a **1-D row** (`np.full(n_time, np.nan)`, ~15 MB) and writes
`self._v[i, :]` into a variable chunked `(1, 25000)`, releasing each row per iteration. Same shape
of problem, solved, and running in production — it produced the very archive this project reads.
*(It writes netCDF4 via h5netcdf rather than cfdb, so it is a pattern reference; cfdb's own write
path is verified separately in §3.)*

**Expected result.** Peak becomes one row plus cfdb's per-call chunk buffer, not one plane:
groundwater's 3.01 GB plane → **~5 MB per row** (620,771 × 8 B), with `_nan_safe`'s mask and copy
shrinking by the same 606× factor. The remaining floor is `fetch_all`'s 784 MB dict — Phase 2.

---

## 2a. Phase 0 — guard the heal path (added by review; do this FIRST, it is ~5 lines)

**I was wrong, and both arms refuted me independently.** The plan claimed the wide-merge hazard was
"verified not reachable through the only consumer". That holds for the *guarded* branch
(`ingest.py:126-134`) — but **not** for the automated heal at `ingest.py:154-157`, whose own comment
says *"the recency guard is deliberately not applied here."* My error was checking the main path and
generalising, without reading the one path whose comment says it skips the check.

What is verified, and what is not:

- **Verified (first-party):** the heal path passes its fetch result straight to `update_and_publish`
  with no recency check; `merge_dataset` derives `w_lo` from the global `inc_min` with no clamp of
  its own; the toolkit has **no self-protection at all** and relies on a consumer guard that one
  path deliberately bypasses.
- **Verified by experiment (Fable):** on a 100 × 61,000 dataset, a normal 48-step merge peaked at
  9 MB; the same window plus **one backdated observation** widened the block to 60,900 steps and
  peaked at 228 MB — a 25× jump from a single value. At streamflow scale that extrapolates to
  ~8 GB inside the hourly cron container.
- **NOT verifiable from here (both arms agree):** whether ECan's endpoint can actually return rows
  older than the requested `Period`. The comment's claim that "the reach is bounded by the deep
  fetch period itself" is **an assumption about an external API, enforced nowhere in code**.
  `_assert_cadence` checks spacing, not recency, and only for precipitation.

**The fix, in `ecan-env/raw/ingest.py`:** apply the same horizon check to the heal result, at
deep-window scale — or drop pre-horizon rows before merging. It converts an unverifiable
external-API assumption into an enforced invariant, and it is the cheapest item in this plan.

**Also record in the toolkit's `OPEN_WORK.md`** that `merge_dataset` should clamp its own block
width rather than trusting callers. Not fixed here (it is a behaviour change to a live path
deserving its own round), but a consumer-side guard protecting a toolkit invariant is backwards.

---

## 3. Phase 1 — `build_local` writes row by row

**File:** `envlib-ingest-base/envlib_ingest_base/tsortho.py`. No other file changes.

### The shape of the change

Replace "assemble a plane, then `dv[:] = _nan_safe(plane)`" with "for each station, build its row,
`_nan_safe` the row, write `dv[i, :] = row`". `_assemble` either gains a per-row form or is
replaced by a row builder; `_nan_safe` is applied per row rather than per plane, so its mask and
copy shrink by the station count.

> ⚠️ **The iteration trap — the single most likely way to get this wrong (review finding).**
> The loop **must** iterate `stations` (equivalently `refs`), **not** `series`/`non_empty`.
> The point coordinate and the `station_id`/`station_ref`/`station_name` arrays are built in
> `stations` order, and today `_assemble` iterates `stations` and does `series.get(ref)`. Iterating
> `series` instead maps the *i*-th station-with-data to point *i*, **silently mislabelling every
> station after the first gap** whenever `len(stations) > len(series)`. The file still validates.
> A fixture where the two sets are equal will not catch it — hence the explicit test in §6.

**Phase 1 is confirmed self-contained:** `_assemble` and `_nan_safe` are used nowhere outside
`tsortho.py`, the tests import neither, and `merge_dataset`'s use of `_nan_safe` is per-block and
untouched (verified by grep, review arm).

Do the same for each ancillary plane (`build_local:540-542`), where the current code nests an
`_assemble` call *inside* `_nan_safe` while the primary plane is still resident — the worst single
moment in the build.

### The structural constraint to respect

`build_local:487-488` reduces over the whole `series` dict twice to find `tmin`/`tmax` and size the
time axis. **The axis extent must be known before any row is written**, so this pre-pass stays in
Phase 1 (it is cheap — one int64 copy per station, transient). It is the thing that makes a
*single-pass* iterator hard, and it is why the streaming contract is Phase 2, not Phase 1.

### cfdb mechanics — verified, with a version caveat the review caught

> ⚠️ **VERSION SKEW — read this before citing any line number below.** The `cfdb` *repo* is
> **0.9.5-dev (unreleased)**; both venvs resolve **PyPI 0.9.4** (verified in both `uv.lock` files).
> Every line reference I originally gave came from the unreleased tree. Two concrete differences:
> the empty-coordinate raise I cited as a dependency property (`utils.py:703-710`) **does not exist
> in 0.9.4** (verified — absent from the installed package), and 0.9.4's `set()` has no
> encoded-space overlay branch, so packed writes take the decoded-space path.
>
> **A review arm re-verified every load-bearing conclusion on 0.9.4 and they all hold.** But state
> the target version per claim: an implementer diffing against the installed 0.9.4 will find the
> quoted code does not match, and merge semantics genuinely differ if 0.9.5 ships.

`var[i, :] = row` is **the optimal write path**, not merely a supported one:

- Chunk keys are per-row: `{var}!{start0}.{start1}` — **dot**-joined (`utils.py:720`,
  `dims = '.'.join(...)`; my original "comma" was wrong). With `chunk_shape=(1, N)`, row `i` maps
  only to `var!i.*` — **no other row's chunks are touched**.
- On a fresh build the chunk does not exist, so `set()` takes the `b1 is None` branch — **no
  decompress, no read-modify-write**. Fill, compress, write.
- `var[i, :]` is accepted directly: the int is converted to `slice(i, i+1)` and the 1-D row is
  reshaped to `(1, N)`.
- Booklet coalesces small writes behind a 4 MiB buffer, so this is not I/O-per-row. A `get` on a
  key *absent* from the buffer does not trip a sync — so fresh-build get-misses are free.

**Measured, not predicted** (review arm, on 0.9.4): a 400 × 150,000 build via the current
whole-plane path peaked at **1208 MB traced / 1307 MB RSS in 1.2 s**; the row prototype peaked at
**11 MB traced / 162 MB RSS in 0.8 s** — *faster*, with all 2,412 stored keys byte-identical.

**Three traps that constrain the implementation:**

1. **Write each chunk exactly once.** Rewriting a chunk costs a full decompress+recompress, orphans
   the old block (log-structured store — the file *grows* until `prune()`), and can force a
   synchronous flush. So build each station's **whole row in one call** — never write a row in
   column-slices, and never revisit a chunk.
2. **A `get` on an un-flushed key forces a full sync** (`booklet/main.py:479-482`), and `set()`
   always `get`s before writing. Writing *distinct* chunks sequentially never trips this; writing
   the same chunk twice trips it every time. Reinforces trap 1.
3. **`Coordinate.append()` is O(entire coordinate) per call** — it rewrites every chunk of the
   coordinate (`support_classes.py:1123-1148`) and does a full `np.append` copy. Per-timestep
   appending would be O(N²). **The current bulk `create.coord.time(data=tvals)` must stay bulk** —
   this refactor changes the *data-variable* write, never the coordinate write.

### Order of operations

Unchanged and load-bearing: create coords → populate coord data → create data var (with explicit
`chunk_shape`) → write rows. A data variable created against an *empty* coordinate raises when
`chunk_shape=None` (`cfdb/utils.py:703-710`) and its `shape` is derived live from the coords
(`support_classes.py:1827`), so an empty coord makes every write fail.

### One deliberate behaviour change to decide, not to slip through

`_assemble` iterates `len(stations)` — the full discovery dict — so **a station with no data
currently gets an explicitly-written all-NaN row**. Writing rows only for stations that *have* data
would leave those chunks absent, and cfdb reads absent cells as NaN, so **values read identically**
— but the set of stored chunks, and the file size, would differ.

Recommendation: **keep writing every station's row**, matching current behaviour exactly, so
Phase 1 stays a pure memory refactor with nothing else moving. Skipping empty stations is a
legitimate optimisation but belongs in its own change with its own verification, not smuggled into
this one. (Note `build_local`'s callers already filter `stations` to those with data — `backfill.py`
does `stations = {ref: stations[ref] for ref in series}` — so in practice the two are usually the
same set anyway.)

### What must NOT change

The stored file: same values, same NaN placement, same packed encoding, same attrs, same
`chunk_shape`, same `ancillary_variables` roster, same station ordering (rows are positional — a
reordering would silently mislabel every station). Same QC counts reported.

---

## 4. Phase 2 — the streaming contract

**Files:** `tsortho.py` (API), `envlib-ingest-ecan-env/qa/source.py` + `qa/backfill.py` +
`raw/source.py` + `raw/ingest.py` (the only consumer).

`series: dict` becomes an iterable of `(ref, times, values, extras)` — or a callable returning one
— so `fetch_all` can `yield` per station instead of building 784 MB of dict first.

**Blast radius is one repo.** `envlib-ingest-esa-sst` does not use this toolkit (it has its own
grid publish path), and nothing else imports it. Breaking change ⇒ **0.2.x → 0.3.0**, manual PyPI
publish, then `uv lock --upgrade-package envlib-ingest-base` downstream.

**The two-pass problem is the real design work.** The axis extent must be known before writing, so
either the caller declares the span up front, or the iterator is consumed twice (needs a re-iterable
source, not a generator), or a cheap metadata pre-pass collects only per-station min/max. **Decide
this explicitly in Phase 2's own design step** — it is the part most likely to be got wrong.

Per-station validation (`_check_aligned`, `_check_extras`, `_qc_filter`, `_non_empty`) is already
single-pass with no cross-station state, so it folds into the streaming loop. `_check_refs` needs
only the key set.

---

## 5. Explicitly NOT in scope — recorded, not fixed

Each was found during exploration and is genuinely lower priority. Recording them prevents
re-discovery.

*(The `merge_dataset` exclusion that was here has been **withdrawn** — both review arms refuted it
independently. It is now in scope as Phase 0; see §2a.)*
- **`merge_dataset` reads the whole time axis** (4–5 copies, `:640-649`) on every run with data —
  5 MB today, grows monotonically with history.
- **`resample._resample_mean`** allocates ~15–20 arrays of `(n_obs + n_intervals)`; a sparse station
  with a long record pays for the intervals. `_local_medians`' `np.median` materialises ~15 × n_obs.
- **`qa/scan.py`'s `all_vals`/`np.concatenate`** ≈ 780 MB peak on groundwater. **Deliberate** — it
  replaced the subsampling that produced wrong published percentiles. Streaming percentiles would
  fix it properly; not now.
- **`envlib/catalogue.py:265`** reads the entire time axis to use only `[0]` and `[-1]`.
- **`screen_precip_stations.py`** never closes its two xarray datasets. *(An arm reported this file
  as non-existent — that was **my scoping error**, not a defect: I did not mount
  `ecan-flow-forecasts` this round. The file is real and the finding stands.)*
- **`tethys-archiver`** is already correct per-station; its one dataset-scale structure is a
  timestamp→index dict (`export.py:238`, ~60–90 MB) — a considered trade, documented in place.

---

## 6. Verification

**⚠️ Do NOT verify with a file checksum.** `test_tsortho.py:941-942` records that two independent
builds differ by cfdb's **per-file uuid**; the existing sha256 test compares one file *before and
after* an operation, which is a different claim. A naive "byte-identical" check fails for a reason
that has nothing to do with correctness.

**The gate: booklet key→bytes map equality** — upgraded on review evidence. My original gate was
decoded-value comparison, and an arm showed it **would wave through the exact change §3 says must
not slip through**: a no-data station's row skipped rather than written decodes identically (NaN
either way) and differs only in the key set.

Two builds from identical input are **byte-identical per stored key** (measured: all 2,412 keys),
because the per-file uuid and timestamps live *outside* the key values. So:

1. Build with current `HEAD`; keep the artifact.
2. Build again with the refactor, same input, same `--work-dir`.
3. **Pass/fail criterion:** equal key sets **and** equal value-bytes per key **and** equal file
   size. This subsumes decoded comparison and additionally catches a skipped station row, any
   encoded-representation drift that decodes identically, and dead space from double writes.
4. Keep the decoded comparison (values, NaN masks, coords, attrs incl. `valid_min`/`valid_max` and
   `ancillary_variables`, dtypes, `chunk_shape`, QC counts) — **for diagnosability**, since a
   key-map mismatch alone will not tell you *what* moved.
5. **Measure peak RSS both ways** with `/usr/bin/time -v`.

**⚠️ The gate dataset MUST carry an ancillary plane.** Raw streamflow has none, so a
streamflow-only gate exercises **none** of the `build_local:540-542` ancillary change — which this
plan itself calls "the worst single moment in the build". Gate on the **qa** streamflow build
(which has `quality_code`) or a synthetic with `ancillary=`, or the riskiest half ships unverified.

**Also test `len(stations) > len(series)`.** See §3's iteration trap: a fixture where those sets are
equal passes even if the loop wrongly iterates `series`.

**RSS expectation, calibrated:** post-Phase-1 streamflow lands at **~0.9–1.0 GB**, dominated by the
caller's 0.72 GB `fetch_all` dict (Phase 2's floor) plus interpreter and libraries. My original
"well under 1 GB" was optimistic as a hard gate and could fail for reasons Phase 1 does not own.

**Existing suite must pass unmodified** — **100 tests** (47 `test_tsortho` + 53 `test_resample`,
verified by running them; my earlier "78 (47+31)" was wrong).
Modifying a test to accommodate the refactor is a signal the refactor changed behaviour; if a test
genuinely encodes the old memory shape rather than behaviour, say so explicitly rather than editing
it quietly.

**Add** a test that fails on the old code: assert peak allocation stays bounded while building a
dataset whose dense plane would be far larger than its data (e.g. many stations × long axis, ~1 %
populated) — the property, not the implementation. `tracemalloc.get_traced_memory()[1]` bounds
Python-level allocation deterministically, without depending on RSS or on the machine.

**Also assert each chunk is written once.** The dead-space/forced-sync traps in §3 are invisible in
the output — the file is *correct*, just larger and slower. A counter around `Booklet.set`, or a
post-build file-size comparison against the old code, catches a regression that content-equality
cannot see.

**End-to-end:** rebuild ECan streamflow, re-run α's round-trip check on station `68801` (Ashburton
at SH1) — 213,808 values, max |Δ| ≤ 1e-4, codes identical, zero NaN introduced.

---

## 7. Review

**Floor applies: at least one independent arm before implementation.**

**Nominated for dual blind.** Two triggers fire: it is **hard to verify by eye** (a row-indexing or
off-by-one slip silently mislabels stations or shifts values, and the file still validates), and it
is **costly if subtly wrong** — this is the write path for permanent published datasets. The
correctness-gate trigger does *not* fire here, since the verification is a differential comparison
against the old code rather than a gate I author from scratch.

Scope to propose: `envlib-ingest-base`, `cfdb`, `envlib-ingest-ecan-env`, and `tethys-archiver` as
the reference implementation.

---

## 7a. Review record — round `chunked-1`, synthesised 2026-08-06

Dual blind: **Fable 5 high** + **Gemini 3.1 Pro high**, same brief, same code version,
`--authored-by claude-opus-5`. Both arms ran experiments. Every finding was given a first-party
verdict before adoption.

| # | my claim | after review | verdict |
|---|---|---|---|
| 1 | heal path "verified not reachable" | **WRONG — withdrawn.** `ingest.py:154-157` skips the guard by design; now Phase 0 | **refuted by BOTH arms independently**, mechanism verified first-party |
| 2 | cfdb claims cite the dependency | cite **0.9.5-dev**; production resolves **0.9.4**, where the empty-coord raise is absent | **verified** first-party (both `uv.lock`s + installed package) |
| 3 | gate = decoded-value equality | **key→bytes map equality** — decoded comparison would wave through a skipped station row | **verified** — arm measured 2,412 keys byte-identical |
| 4 | gate on streamflow | streamflow has **no ancillary**, so it would test none of the riskiest change | **verified** |
| 5 | 78 tests (47+31) | **100** (47+53) | **verified** by running them |
| 6 | chunk key `{var}!a,b` | dot-joined `{var}!a.b` (`utils.py:720`) | **verified** |
| 7 | iterate rows | must iterate `stations`, never `series`, or every station after a gap is mislabelled | **verified** — accepted as a new constraint |
| 8 | "well under 1 GB" post-fix | ~0.9–1.0 GB; the caller's dict is Phase 2's floor | **verified** |
| 9 | ~88 % of addressable peak | holds exactly (6.02/6.80 = 88.5 %), and is **conservative** vs the measured 10.2 GB | **verified** by both arms |
| 10 | row-by-row never worse | confirmed: byte-identical output, and *faster* (0.8 s vs 1.2 s on 400×150k) | **verified** by experiment |

**Convergence worth noting:** both arms independently reached the heal-path finding — the one thing
I flagged as my weakest claim. Agreement across decorrelated arms on a point the author doubted is
the strongest signal this method produces.

**No cross-arm disagreements** this round; the arms were complementary rather than conflicting
(Gemini: the `stations`-vs-`series` iteration trap and the arithmetic; Fable: version skew, the
key-map gate, the ancillary gap, the test count).

**One reported finding refuted:** `screen_precip_stations.py` "does not exist" — my scoping error
(I did not mount `ecan-flow-forecasts`), not a defect in the code.

**Accepted unverified:** that ECan's endpoint honours `Period` recency (unknowable without the live
service — which is precisely why Phase 0 replaces the assumption with an enforced check), and that
`envlib-ingest-esa-sst` is a non-consumer (repo not mounted this round; I verified it separately by
grep before the review).

---

## 8. Step 0 — preserve Plan α's record first

Before anything else: copy the α plan (rounds `ecan-alpha-2` and `ecan-code-1`, the before/after
table, the cross-arm disagreements, the verdicts) into
`ecan-flow-forecasts/planning/phase1-alpha-plan.md`, alongside `phase1-data-foundation.md`, so the
convention holds that a reviewed plan survives in git rather than only in `~/.claude/plans/`.

## 9. Sequencing

```
Step 0  preserve α's review record  ──────────────────────────┐
Phase 0 heal-path horizon guard (~5 lines, ingest.py)  ───────┤ both cheap, both independent
  └→ Phase 1 build_local writes rows ──→ implement ──→ key-map + RSS gate
        └→ then α's remaining publishes, built by the fixed code
              └→ Phase 2 streaming contract (0.3.0) — own design step for the two-pass problem
```

**Review is done** (§7a) — Phase 0 and Phase 1 are ready to implement. Phase 2 gets its own design
step and its own review, because the two-pass axis-extent problem is where it can go wrong.
