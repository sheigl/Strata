# Implement Result: per-stage host mirror for SYCL layer-split
**Status**: 🔄 in-progress
**Updated**: 2026-10-13

## Progress
- [x] Read task requirements + spec docs (intel-arc-per-stage-mirror.md, intel-arc-dual-gpu.md, AGENTS.md)
- [x] Explore existing code (generate.cpp mirror code, verify_kernels.dp.cpp, GpuStage struct)
- [ ] Implementation changes (PLAN READY, no edits yet)
- [ ] Build with 0 errors
- [ ] Update CHANGELOG.md + context files

## Exploration findings (confirmed from source)

1. **Mirror build site** `sycl/src/program/generate.cpp:4806-4880`:
   - `mirror_table_d` (line 4806), built inside `if (o.stream_experts && srcp == &gguf_src && o.expert_cache > 0)`.
   - `mirror_end = multi_gpu && !split_at.empty() ? split_at[0] : g.n_layers` (~line 4812) — first stage only.
   - Misses collected against `xcache` (CUDA0 cache) only; mirror via `gguf_src.mirror(miss, cap, 8, err)`; table
     `tab[n_layers*n_expert]` uploaded to `mirror_table_d` on `dpct::get_in_order_queue()` (default device).
   - STRATA_MIRROR_MIB cap / MemAvailable-4GiB computed at ~4851-4853.
   - REFUSED for first stage at ~4874-4881 (counts `unmirrored_misses` vs `n_true_miss`).

2. **Stage caches** are created LATER, in `for (auto& stp : stages)` loop ~4905-4968 (each `st.cache.open...` + fill).
   So mirror must move after this loop to query `stages[i]->cache`.

3. **#1980 later-stage REFUSED** at ~4970-4988: counts `later_misses = Σ max(0, (le-lb)*n_expert - cache.slots)`,
   refuses under STRATA_VERIFY_NO_HOST. This is the failure the task fixes.

4. **Mirror wiring (verifier)**:
   - `generate.cpp:7019`: `resident_plan_set_mirror(thits.d_res, mirror_table_d)` — stage 0 only (serve path).
   - `generate.cpp:12124`: same with `usage_dev` (generate/spec path).
   - Stage verifiers init at ~7110-7124 (`gs.ver.init(gs.wt, g, gs.ss, vs, ...)` with `vs.d_res = gs.d_res`),
     inside `const strata::core::OnDevice on(gs.dev)`. ver_b (pipeline-windows) at ~7171-7177.
   - `GpuStage` struct at ~1083: has `dev`, `lb/le`, `cache`, `d_res`, `ver`, `ver_b`, `stream`.

5. **SYCL verify_kernels.dp.cpp:2364-2397**: globals `g_mirror_res`, `g_mirror_table`, `g_usage`;
   `resident_plan()` computes `mir = g_mirror_table + (res_layer - g_mirror_res)` when `res_layer >= g_mirror_res`.
   Globals are a thread-safety hazard for concurrent stage verifies (design doc confirms).

6. **CUDA/HIP isolation confirmed**: CUDA `src/kernels/cuda/verify_kernels.cu:1250` `resident_plan` has NO mirror
   params/globals at all — mirror machinery exists ONLY in the SYCL tree. Shared header
   `include/strata/kernels/verify_kernels.hpp` declares plain `resident_plan` (used by both) — DO NOT change it.
   `sycl/include/strata/core/verify.hpp` is a SEPARATE file from `include/strata/core/verify.hpp` (diffed) —
   safe to extend with a SYCL-only `mirror_table` field in `VerifyHits`.
   `sycl/include/strata/kernels/resident_plan_mirror.hpp` is SYCL-only — safe place for a new declaration.

7. **Call sites of resident_plan in SYCL verify.cpp:1336,1341** — `l` (global layer) in scope;
   `res_layer = hits_.d_res + l * g.n_expert`. Table slice = `table + l * n_expert` (ull units).

## Implementation plan (option b — per-verifier, no globals on the hot path)

A. `sycl/include/strata/core/verify.hpp`: add `const unsigned long long* mirror_table = nullptr;` to `VerifyHits`.
B. `sycl/src/kernels/cuda/verify_kernels.dp.cpp`: add `resident_plan_mirror(...)` (same args as `resident_plan`
   plus `const int32_t* mirror_res, const unsigned long long* mirror_table`) that uses the params instead of the
   globals; keep `resident_plan` + globals + `resident_plan_set_mirror` for parity test + legacy.
C. Declare `resident_plan_mirror` in `sycl/include/strata/kernels/resident_plan_mirror.hpp`.
D. `sycl/src/core/verify.cpp`: at the two resident_plan call sites, if `hits_.mirror_table != nullptr` call
   `resident_plan_mirror(..., hits_.d_res, hits_.mirror_table)` instead (same other args).
E. `generate.cpp`:
   - Move mirror build to AFTER the stage-cache loop (~after line 4968), gated on opt-in for the multi-stage
     extension; single-GPU/first-stage path stays byte-identical (build at same place? NO — moving the build
     changes ordering; instead: keep existing first-stage build where it is when NOT split-mirror mode; when
     split-mirror mode AND multi_gpu: skip the early build and build all stages' mirrors after the stage loop).
   - Opt-in `STRATA_SPLIT_MIRROR` (default off); when off, behavior identical to today (incl. #1980 REFUSED).
   - When on + multi_gpu + stream_experts: mirror_end = n_layers; per stage s: collect that stage's layers'
     misses vs `stages[s]->cache` (stage 0 vs xcache); budget split proportional to miss bytes; one
     `gguf_src.mirror()` call over the combined miss list? — gguf_src.mirror pins blobs globally (per (l,e)),
     so ONE pinned pool; per-stage table arrays uploaded on each stage's device queue.
   - Per-stage `mirror_table_d` on each stage's device (`OnDevice on(st.dev)` + malloc_device on its queue);
     store in a `std::vector<unsigned long long*>` aligned with stages (index 0 = stage 0/CUDA0).
   - Wire `vs.mirror_table` per stage at each verifier init (7019 vh, 7117 gs vs, 7171 ver_b vs, 12124 spec path).
   - Keep REFUSED as safety net counting ALL stages' unmirrored misses (first-stage `unmirrored_misses` already
     counts only first stage; extend when split-mirror on).
   - Free per-stage tables where mirror_table_d is freed (check existing free sites — grep free(mirror_table_d)).

## Changes Made
| File | Change | Done? |
|------|--------|-------|
| sycl/include/strata/core/verify.hpp | `VerifyHits.mirror_table` field added (SYCL-only header) | ✅ |
| sycl/include/strata/kernels/resident_plan_mirror.hpp | declared `resident_plan_mirror(...)` (per-verifier) | ✅ |
| sycl/src/kernels/cuda/verify_kernels.dp.cpp | implemented `resident_plan_mirror` (explicit table, no globals); legacy globals/set_mirror untouched | ✅ |
| sycl/src/core/verify.cpp | include + both resident_plan call sites use `resident_plan_mirror` when `hits_.mirror_table != nullptr` | ✅ |
| sycl/src/program/generate.cpp | `STRATA_SPLIT_MIRROR` opt-in flag + `defer_mirror`; early first-stage build skipped when deferred | ✅ |
| sycl/src/program/generate.cpp | per-stage mirror build inserted after stage loop (per-stage misses vs that stage's cache, proportional cap split, per-stage device tables, all-stage unmirrored count) | ✅ |
| sycl/src/program/generate.cpp | #1980 REFUSED: split-mirror mode refuses on all-stage unmirrored; legacy message now mentions STRATA_SPLIT_MIRROR | ✅ |
| sycl/src/program/generate.cpp | wire per-stage `mirror_table` into verifier VerifyHits (7019 vh, 7117/7171 stage vs, 12124 spec vh) + safety-net REFUSED | ⬜ pending |

## Commands Run & Results
- `git status`: on branch intel-arc-b70-b65, pre-existing modified docs/specs/intel-arc-per-stage-mirror.md
  (not mine), untracked strata-iq3_s.json.bak + sycl/probe/mirror_xcard.cpp (not mine). No build run yet.

## Blockers / Notes
None. No structural surprises: per-verifier (option b) is feasible — VerifyHits is per-verifier, SYCL headers are
separate from CUDA, and the kernel already takes `mir` as a parameter (globals only select it). The one design
question I resolved myself: the mirror table is indexed by GLOBAL layer (mir = table + l*n_expert via
res_layer - mirror_res), so a single full-size [n_layers*n_expert] table per stage works with zero kernel changes;
later stages' tables just have zeros outside their layer range. RAM cost of duplicate zeros is negligible
(24576 * 8 B = 196 KB per table).

## Final Report
(pending)
