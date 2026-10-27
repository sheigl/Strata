# Implementation Plan: Mixed-Vendor GPU RPC

Status: **implementation in progress**. Tasks 1-7 are implemented; Task 7's fixed-window parity evidence and limits are
in [evidence/task7.md](evidence/task7.md). Mixed-vendor hardware validation is still unmeasured.

## Overview

Add an opt-in, Linux-first RPC layer split that runs Qwen3.8-Flash-Next across separate CUDA and SYCL engine
processes. A prefix worker owns initial layers and the controller owns final layers, head, sampling, and MTP. Control
uses versioned Unix-socket frames; activations use an fd-passed memfd with synchronous CPU staging. The full design,
protocol, limits, and acceptance gates are in [docs/specs/mixed-gpu-rpc.md](../docs/specs/mixed-gpu-rpc.md).

## Architecture Decisions

- Use whole-layer process boundaries, not tensor/expert-compute RPC.
- Keep Python responsible for backend selection and process supervision; keep decode/prefill orchestration in C++.
- Preserve the existing `Verifier` and `Prefill` handoff layouts and local stage behavior.
- Prove same-backend RPC bit parity before enabling mixed CUDA + SYCL execution.
- Require explicit layer placement and weight/state trimming first; defer auto placement, networking, GUI, batch,
  images, and session restore.
- Fail closed on worker/protocol errors. Never silently fall back to one GPU or restart into a live conversation.

## Task List

Detailed acceptance criteria and verification are in [tasks/todo.md](todo.md).

### Phase 1: Foundation
- [x] Task 1: Versioned RPC wire protocol
- [x] Task 2: Linux local transport and shared bulk buffer
- [x] Task 3: `rpc_split` configuration validation
- [x] Task 4: Worker process supervision

### Checkpoint: Foundation
- [x] Protocol, transport, config, and lifecycle tests pass without a GPU.
- [x] No engine behavior changes when `rpc_split` is absent.

### Phase 2: Same-Backend Decode
- [x] Task 5: Extract CUDA stage runtime without behavior changes
- [x] Task 6: Extract the decode-stage boundary without behavior changes
- [x] Task 7: CUDA worker and RPC prefix decode vertical slice

### Checkpoint: Decode
- [x] Two-process same-backend decode matches the equivalent in-process split bit-for-bit (one-card diagnostic
  cache placement; see [evidence/task7.md](evidence/task7.md)).
- [x] Partial speculative commits and worker failures behave correctly.

### Phase 3: Prompt and Memory
- [x] Task 8: Extract the prefill-stage boundary without behavior changes
- [ ] Task 9: RPC prefill vertical slice
- [ ] Task 10: RPC memory ownership and budgets
- [ ] Task 11: Cancellation, failure, and diagnostics hardening

### Checkpoint: Same-Backend Feature
- [ ] Long prompts, commit/cancel, memory budgets, and cleanup pass on the same-backend RPC split.
- [ ] Default single-GPU and same-vendor split tests still pass.

### Phase 4: Mixed CUDA + SYCL
- [ ] Task 12: Extract SYCL stage runtime without behavior changes
- [ ] Task 13: SYCL RPC prefix worker
- [ ] Task 14: Cross-backend stage validation harness
- [ ] Task 15: Mixed long-context hardware gate and measured optimization

### Checkpoint: Mixed Hardware
- [ ] Qwen3.8-Flash-Next Q4_K_M or better runs at >=131,072 context on CUDA + SYCL.
- [ ] Correctness suite passes and sustained decode is >=20 tok/s after a real long prompt.
- [ ] Measurements name the exact hardware and include stage/RPC timings and all limitations.

### Phase 5: Handoff and Documentation
- [ ] Task 16: Opt-in launch surface, documentation, and final validation record

## Dependency Summary

```text
T1 -> T2 -> T4
T3 -> T4
T5 -> T6 -> T7 -> T8 -> T9 -> T10 -> T11
T2 + T4 + T7 -> same-backend decode checkpoint
T12 -> T13 -> T14 -> T15
T2 + T4 + T11 + T13 -> T15
T15 -> T16
```

T5/T6 and T12 can be developed independently once the protocol contract is stable, but mixed validation remains
blocked on the full same-backend feature and failure hardening.

## Risks and Mitigations

See the risk table in the spec. The highest-risk items are stage-extraction regressions, duplicated weight/RAM
ownership, and decode-boundary latency. The plan isolates each behind a parity or measurement checkpoint.

## Open Inputs for the Implementer

- Record the exact NVIDIA and Intel Arc models, VRAM, drivers, backend binaries, and artifact manifest before the
  hardware run.
- Choose the initial controller/device assignment from measured memory and speed. The first slice assumes CUDA
  controller + SYCL prefix worker; reversing it requires the corresponding SYCL controller proxy and CUDA worker.
- If same-backend RPC cannot preserve bit parity, stop and fix transport/state semantics before mixed-vendor work.
- If the mixed run misses 20 tok/s, use the optimization order in the spec; do not weaken the gate or claim success.

## Handoff Rule

A separate model may implement tasks only after the human reviews this plan. Each task should leave the tree working,
include its stated verification, and keep passed/failed/not-run evidence separate.
