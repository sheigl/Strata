# Mixed-Vendor GPU RPC Tasks

Technical contract: [docs/specs/mixed-gpu-rpc.md](../docs/specs/mixed-gpu-rpc.md). Intent:
[docs/intent/mixed-gpu-rpc.md](../docs/intent/mixed-gpu-rpc.md).

## Task 1: Versioned RPC wire protocol

**Description:** Add the backend-neutral frame/message definitions, strict encoder/decoder, opcode and state rules,
sequence validation, and bounded control payloads described in the spec. No GPU or process behavior yet.

**Acceptance criteria:**
- [x] Every initial opcode round-trips with fixed-width little-endian fields and explicit version/capability values.
- [x] Truncated, oversized, unknown, stale, duplicate, and out-of-state messages return typed errors without reading
  past input.
- [x] No CUDA/HIP/SYCL/ggml dependency enters the common RPC library.

**Verification:**
- [x] Build and run the focused protocol test with `ctest --test-dir build -R '^rpc_protocol' --output-on-failure`.
- [ ] Configure the SYCL build to show the common sources remain backend-neutral.

**Dependencies:** None.

**Files likely touched:**
- `include/strata/rpc/protocol.hpp`
- `src/rpc/protocol.cpp`
- `tests/rpc/protocol_test.cpp`
- `cmake/strata_rpc.cmake`
- `CMakeLists.txt`

**Estimated scope:** Medium.

## Task 2: Linux local transport and shared bulk buffer

**Description:** Implement ordered Unix-domain-socket transport, same-UID/token checks, deadline-aware I/O,
`SCM_RIGHTS` fd passing, and sealed memfd/mmap bulk regions. Provide an unsupported-platform stub.

**Acceptance criteria:**
- [x] Controller and worker negotiate one bulk region and exchange framed messages without GPU code.
- [x] Wrong peer UID/token, dead peer, short read/write, timeout, and bad fd are rejected and cleaned up.
- [x] No TCP listener, token in argv/logs, or GPU-visible shared flag is created.

**Verification:**
- [x] `ctest --test-dir build -R '^rpc_(protocol|local_transport)' --output-on-failure`.
- [x] Run abnormal-exit and fd-leak checks in the transport test.

**Dependencies:** Task 1.

**Files likely touched:**
- `include/strata/rpc/transport.hpp`
- `src/rpc/local_transport_posix.cpp`
- `src/rpc/local_transport_stub.cpp`
- `tests/rpc/local_transport_test.cpp`
- `cmake/strata_rpc.cmake`

**Estimated scope:** Medium.

## Task 3: `rpc_split` configuration validation

**Description:** Add the validated run-config schema for one prefix worker, explicit boundary, backend/device,
optional development executable, and restricted backend environment. Reject unsupported combinations explicitly.

**Acceptance criteria:**
- [x] Valid CUDA-controller/SYCL-prefix config produces the expected internal worker/controller arguments.
- [x] Invalid ranges, unknown backends, multiple workers, unsafe env keys, and combinations with `--layer-split`,
  peer mode, batch, images, or restore fail with plain errors.
- [x] Existing GPU-role and mixed-vendor behavior is byte-for-byte unchanged when `rpc_split` is absent.

**Verification:**
- [x] `python -m pytest tests/terminal/test_rpc_split.py tests/terminal/test_mixed_vendors.py`.
- [x] `python -m unittest serve.test_server` (or the repository's focused equivalent for modified cases).

**Dependencies:** None; can run beside Tasks 1-2.

**Files likely touched:**
- `strata_control/rpc_config.py`
- `serve/server.py`
- `tests/terminal/test_rpc_split.py`
- `serve/test_server.py`

**Estimated scope:** Medium.

## Task 4: Worker process supervision

**Description:** Add a supervisor that resolves the worker backend, creates protected socket/token state, starts the
worker before the controller, propagates readiness/failure, and reaps both processes.

**Acceptance criteria:**
- [x] Fake workers prove startup order, ready timeout, crash propagation, shutdown, and no orphaned child.
- [x] Authentication material never appears in argv or logs; private socket/runtime files are removed.
- [x] A worker is never restarted into an existing conversation.

**Verification:**
- [x] `python -m unittest serve.test_rpc_workers`.
- [x] Existing server lifecycle tests pass.

**Dependencies:** Tasks 2 and 3.

**Files likely touched:**
- `serve/rpc_workers.py`
- `serve/server.py`
- `serve/test_rpc_workers.py`
- `strata_control/engine_component.py`

**Estimated scope:** Medium.

## Checkpoint: Foundation

- [x] Tasks 1-4 pass with no GPU.
- [x] No configured worker means no new socket, process, allocation, or engine argument.

## Task 5: Extract CUDA stage runtime without behavior changes

**Description:** Move the existing CUDA stage construction/execution needed by split mode into a backend-local
runtime unit while keeping the normal engine and in-process split on the same code paths.

**Acceptance criteria:**
- [x] Stage runtime exposes layer-range weight/state/cache/prefill/verifier ownership without changing handoff layout.
- [x] Single-GPU and same-vendor split behavior is unchanged; no RPC option is required to use the refactor.
- [x] No unrelated formatting or cleanup is included.

**Verification:**
- [x] CUDA `strata` build and affected test targets pass on RTX 3060. The HIP variant was not run: this host has no ROCm/hipcc.
- [x] Run existing split, prefill, verifier, and conversation tests: 24 focused CUDA tests passed.
- [x] Same-day RTX 3060 greedy smoke: single-GPU and same-device in-process split (`--layer-split 24 --split-device 0`) generated identical token IDs with the baseline build.

**Dependencies:** None, but its interface must follow the spec.

**Files likely touched:**
- `include/strata/program/stage_runtime.hpp`
- `src/program/stage_runtime.cpp`
- `src/program/generate.cpp`
- `CMakeLists.txt`

**Estimated scope:** Medium; stop and split further if more files are needed.

## Task 6: Extract the decode-stage boundary without behavior changes

**Description:** Put only the recursive decode run/commit boundary behind a narrow interface. Keep local Verifier
semantics and all sampling/MTP behavior in the existing path.

**Acceptance criteria:**
- [x] Local chained stages behave exactly as before, including partial commits and diagnostics.
- [x] Unsupported batch calls have an explicit capability path rather than accidental remote behavior.
- [x] No wire or worker code is added in this task.

**Verification:**
- [x] `ctest --test-dir build-cuda-smoke13 -R '(verify|split|stage_chain)' --output-on-failure`: 9 tests passed.
- [x] Same-device in-process split greedy smoke (`--layer-split 24 --split-device 0`) generated identical token IDs with the pre-refactor baseline.

**Dependencies:** Task 5.

**Files likely touched:**
- `include/strata/core/decode_stage.hpp`
- `include/strata/core/verify.hpp`
- `src/core/verify.cpp`
- `src/program/generate.cpp`
- `tests/core/stage_chain_test.cpp`

**Estimated scope:** Medium.

## Task 7: CUDA worker and RPC prefix decode vertical slice

**Description:** Add worker mode and the controller-side RPC prefix proxy so two CUDA processes execute one decode
window chain through shared-memory handoff. The controller owns suffix/head/sampling/MTP.

**Acceptance criteria:**
- [x] Worker loads only `[0,K)`, completes handshake/READY, executes WINDOW/COMMIT, and shuts down cleanly.
- [x] Controller executes `[K,n_layers)` and final sampling without sending logits or sampling state to the worker.
- [x] Same-backend RPC fixed windows and generated tokens match the equivalent in-process split bit-for-bit,
  including rejected drafts/partial commits. Evidence and the one-card cache-placement limitation are in
  `tasks/evidence/task7.md`.

**Verification:**
- [x] Focused RPC decode tests plus existing split tests: 13/13 CUDA CTests and 4/4 `rpc_*` CTests passed;
  `serve.test_server` 355/355 and the RPC/terminal Python set 27/27 passed.
- [x] Two-process same-backend hardware comparison with exact commands/logs recorded: 12 generated tokens equal
  with `--spec-corrupt 2` partial commits (`keep 2 of 4`, `keep 1 of 4`) on the RTX 3060, using the diagnostic
  `STRATA_RPC_CACHE_SLOTS_CAP=200` and a real second in-process `GpuStage` baseline.
- [x] CUDA build succeeds.
- [ ] Affected HIP build: not run; this host has no `hipcc`/ROCm runtime.

**Dependencies:** Tasks 2, 4, and 6.

**Files likely touched:**
- `src/program/rpc_worker.cpp`
- `src/program/rpc_prefix_stage.cpp`
- `src/program/stage_runtime.cpp`
- `src/program/generate.cpp`
- `tests/rpc/decode_split_test.cpp`

**Estimated scope:** Medium.

## Checkpoint: Same-Backend Decode

- [ ] Same-backend RPC decode is bit-exact against in-process split.
- [ ] Disconnect, bad sequence, and commit failure never return a successful partial response.

## Task 8: Extract the prefill-stage boundary without behavior changes

**Description:** Narrowly abstract the existing prefill next-stage call while preserving chunks, positions, callbacks,
MTP draft behavior, and local threading.

**Acceptance criteria:**
- [x] Existing one-stage and in-process split prompt reads retain behavior and output.
- [x] Stop/cancel and chunk callback ordering are covered before RPC is introduced. Their callbacks stay in the
  existing `Prefill` path; `serve.test_server` (355 tests) and `batch_read_test` pass, and `tasks/evidence/task8.md`
  records the unchanged prompt/output comparison.

**Verification:**
- [x] `ctest --test-dir build-cuda-smoke13 -R '(prefill|batch_read)' --output-on-failure`: 7/7 passed (one
  `prefill_fused_iq_test` OOM while a prompt smoke held the GPU; rerun alone and the full set passed).
- [x] Same-backend prompt/output comparison against the pre-refactor baseline: both prompts generated identical
  tokens and text.

**Dependencies:** Task 7.

**Files likely touched:**
- `include/strata/prefill/prefill_stage.hpp`
- `include/strata/prefill/prefill.hpp`
- `src/prefill/prefill.cpp`
- `src/program/generate.cpp`
- `tests/prefill/prefill_stage_test.cpp`

**Estimated scope:** Medium.

## Task 9: RPC prefill vertical slice

**Description:** Transfer prompt-chunk residual rows through the negotiated bulk region and execute the suffix in the
controller. Start synchronously; preserve stop and MTP semantics.

**Acceptance criteria:**
- [ ] Same-backend RPC prompts match in-process split output bit-for-bit across multiple chunk sizes and boundaries.
- [ ] A prompt larger than one chunk works, and cancellation at a chunk boundary leaves no live worker operation.
- [ ] RPC prefill counters report worker/controller/copy time without changing non-RPC telemetry.

**Verification:**
- [ ] Focused RPC prefill tests and same-backend hardware long-prompt comparison.
- [ ] Existing prefill and batch-read tests pass.

**Dependencies:** Tasks 2 and 8.

**Files likely touched:**
- `src/program/rpc_worker.cpp`
- `src/program/rpc_prefix_stage.cpp`
- `src/program/stage_runtime.cpp`
- `src/program/generate.cpp`
- `tests/rpc/prefill_split_test.cpp`

**Estimated scope:** Medium.

## Task 10: RPC memory ownership and budgets

**Description:** Make range trimming mandatory, scope each stage's KV/expert/mirror state, and print/refuse using a
pre-load VRAM+RAM budget. Audit and fix any full-model CUDA pinned-arena duplication for worker mode.

**Acceptance criteria:**
- [ ] Each process reports only its owned layers' dense weights, KV/GDN state, cache/mirror, and reserved overhead.
- [ ] A configuration that exceeds either process's VRAM or the host RAM cap refuses before allocating the model.
- [ ] No full-model expert arena or mirror is duplicated merely because RPC mode is active.

**Verification:**
- [ ] Focused memory arithmetic/unit tests.
- [ ] Compare reported budgets with process RSS and per-device allocations in a same-backend load.
- [ ] Existing split auto-placement and memory tests remain unchanged outside RPC mode.

**Dependencies:** Tasks 7 and 9.

**Files likely touched:**
- `include/strata/core/expert_source.hpp`
- `src/core/expert_source.cpp`
- `src/program/stage_runtime.cpp`
- `src/program/generate.cpp`
- `tests/core/rpc_stage_memory_test.cpp`

**Estimated scope:** Medium.

## Task 11: Cancellation, failure, and diagnostics hardening

**Description:** Complete timeout/cancel/error semantics, safe teardown, and per-stage RPC timing/counters before
mixed hardware runs.

**Acceptance criteria:**
- [ ] Startup mismatch, malformed frame, disconnect, timeout, and local/remote commit failure all fail closed with a
  stage/operation-specific error.
- [ ] Active prefill/decode cancellation and worker crash leave no orphan process or reusable corrupt conversation.
- [ ] Diagnostics expose sequence/stage/operation, sizes, timings, and OS/GPU errors without tokens or credentials.

**Verification:**
- [ ] Protocol, supervisor, and engine fault-injection tests pass.
- [ ] Manual same-backend kill/timeout/cancel matrix passes with logs retained.

**Dependencies:** Tasks 4, 7, and 9.

**Files likely touched:**
- `src/program/rpc_worker.cpp`
- `src/program/rpc_prefix_stage.cpp`
- `serve/rpc_workers.py`
- `serve/test_rpc_workers.py`
- `tests/rpc/failure_test.cpp`

**Estimated scope:** Medium.

## Checkpoint: Same-Backend Feature

- [ ] Tasks 1-11 pass; default paths remain unchanged.
- [ ] Same-backend RPC handles long prompts, speculative commits, memory budgets, cancellation, and crashes.

## Task 12: Extract SYCL stage runtime without behavior changes

**Description:** Mirror the backend-local stage extraction in the SYCL tree, retaining its own expert source, mirror,
prefill, verifier, and xe-specific synchronization behavior.

**Acceptance criteria:**
- [ ] Existing native SYCL single-GPU and same-vendor split paths behave as before.
- [ ] No CUDA header, runtime handle, or host-flag assumption enters the SYCL runtime.
- [ ] The common RPC library builds in the SYCL target without changing backend exclusivity.

**Verification:**
- [ ] Build `build-sycl` and run relevant SYCL parity/split tests on Arc.
- [ ] Native same-vendor Arc split smoke matches the pre-refactor baseline.

**Dependencies:** Tasks 1 and 5 (interface contract only; no CUDA runtime dependency).

**Files likely touched:**
- `sycl/include/strata/program/stage_runtime.hpp`
- `sycl/src/program/stage_runtime.cpp`
- `sycl/src/program/generate.cpp`
- `sycl/CMakeLists.txt`

**Estimated scope:** Medium.

## Task 13: SYCL RPC prefix worker

**Description:** Implement the worker loop using SYCL stage runtime and the shared protocol/transport. Use stream sync
plus CPU staging; do not port CUDA doorbells or mapped flags.

**Acceptance criteria:**
- [ ] SYCL worker completes model/geometry handshake and serves WINDOW, COMMIT, and PREFILL_CHUNK for its layer range.
- [ ] Prefix expert cache/mirror and KV settings are honored; no later-stage global mirror table is used.
- [ ] Two-process same-SYCL RPC output matches the equivalent native in-process split bit-for-bit.

**Verification:**
- [ ] Build and run SYCL RPC tests on Arc.
- [ ] Run same-SYCL two-process decode/prefill comparison and crash/cancel checks.
- [ ] Build CUDA after any shared RPC change.

**Dependencies:** Tasks 2, 7, 9, and 12.

**Files likely touched:**
- `sycl/src/program/rpc_worker.cpp`
- `sycl/src/program/stage_runtime.cpp`
- `sycl/src/program/generate.cpp`
- `sycl/CMakeLists.txt`
- `sycl/serve/strata-sycl-host.sh`

**Estimated scope:** Medium.

## Task 14: Cross-backend stage validation harness

**Description:** Add a development harness that sends identical stage inputs through CUDA and SYCL worker/controller
paths, checks finite outputs and state progression, and records reference error without claiming bit parity.

**Acceptance criteria:**
- [ ] GDN, QSA, MoE hit/miss/mirror, several valid boundaries, and partial commits are exercised.
- [ ] Handshake rejects wrong artifact, geometry, range, context, and protocol version before inference.
- [ ] Repeated mixed runs with the same seed/config are deterministic; tolerances come from the established reference
  procedure, not from a failing result.

**Verification:**
- [ ] Harness runs on the actual CUDA + Arc host with raw outputs retained.
- [ ] Fixed served-output correctness suite passes and divergences are documented.

**Dependencies:** Tasks 11 and 13.

**Files likely touched:**
- `tools/rpc_stage_parity.cpp`
- `tests/rpc/mixed_stage_test.cpp`
- `tests/rpc/README.md`
- `cmake/strata_rpc.cmake`
- `CMakeLists.txt`

**Estimated scope:** Medium.

## Task 15: Mixed long-context hardware gate and measured optimization

**Description:** Run the confirmed Qwen workload on CUDA controller + SYCL prefix worker. Measure first, then apply
only the spec's ordered optimizations if needed to reach the target.

**Acceptance criteria:**
- [ ] Qwen3.8-Flash-Next Q4_K_M or better, configured context >=131,072, runs with both expected processes/devices
  and layer ranges active.
- [ ] A real roughly 104K+ token prompt is answered coherently and the fixed correctness suite passes.
- [ ] Sustained decode is >=20 generated tokens/s after the long prompt; prompt/decode/stage/RPC timings, draft
  acceptance, memory budgets, variability, and exact hardware/config are reported.
- [ ] Any optimization passes same-backend parity and default-path tests, with same-day before/after measurements.

**Verification:**
- [ ] Preserve raw logs and individual runs; separate passed, failed, and not-run checks.
- [ ] Repeat active-request cancel, worker crash, and full shutdown at long context.

**Dependencies:** Tasks 10, 11, and 14.

**Files likely touched:**
- `docs/specs/mixed-gpu-rpc.md` (validation record only)
- `docs/COMMUNITY_BENCHMARKS.md`-compliant report location chosen by the maintainer
- Optimization source files only when a measured bottleneck justifies them

**Estimated scope:** Hardware validation plus separately scoped optimizations.

## Checkpoint: Mixed Hardware

- [ ] All workload, correctness, performance, isolation, and lifecycle gates pass on the named machine.
- [ ] No unmeasured compatibility or performance claim is added to user documentation.

## Task 16: Opt-in launch surface, documentation, and final validation record

**Description:** Expose the validated explicit config through the Terminal without changing default recommendations;
update mixed-vendor and multi-GPU docs with actual limits and measurements.

**Acceptance criteria:**
- [ ] The new option is explicit and existing mixed-vendor behavior remains unchanged without it.
- [ ] Docs describe process topology, config, supported limits, failure behavior, and the exact measured result; no
  network/AMD/batch/image/session-restore support is implied.
- [ ] The spec/tasks record final passed, failed, and not-run checks and remaining work for network/auto placement.

**Verification:**
- [ ] `python -m pytest tests/terminal`.
- [ ] Documentation links/examples and run-config validation tests pass.
- [ ] Final CUDA, HIP (when shared CUDA-tree files changed), and SYCL build/test evidence is attached.

**Dependencies:** Task 15.

**Files likely touched:**
- `strata_cli/launch/params.py`
- `strata_cli/launch/runner.py`
- `tests/terminal/test_mixed_vendors.py`
- `docs/MULTI_GPU.md`
- `docs/TERMINAL_LAUNCH.md`

**Estimated scope:** Medium.
