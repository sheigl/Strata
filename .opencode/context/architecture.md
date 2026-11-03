# Strata engine — architecture notes for the rig workstream

Short, code-anchored map of the parts the mixed-vendor rig touches. `docs/DETAILS.md` and `docs/MULTI_GPU.md`
are the long form; this is the shape an implementer needs to place code correctly.

## 1. Three parts, two engine trees

- **App**: `apps/desktop` (Tauri) + `apps/host` (Node host, web UI).
- **Terminal**: `strata.py`, `strata_cli/` (`strata launch`, `strata connect`), `strata_control/` (components,
  tuning, gpu roles, feed).
- **Engine + server**: `src/`, `include/` (CUDA/HIP), `sycl/` (the Intel port), `metal/`, `serve/server.py`
  (OpenAI- and Anthropic-compatible API, one engine process per model over a stdin/stdout line protocol:
  `READY`, `SERR`, `SESSION`, `SWAIT`, request lines; `ENGINE_STALL_S`/`ENGINE_READY_S` watchdogs).

`src/` and `sycl/src/` are **compile-time forks**, not a runtime abstraction. `sycl/CMakeLists.txt`'s
`strata_resolve()` picks the port's migrated copy (`*.dp.cpp`) when it exists and the root source otherwise, so
anything vendor-neutral belongs in the root tree and is shared automatically. The port stubs what it does not
have: `src/core/vmm.cpp` (no driver VMM on Level Zero), `src/core/remote_expert_opt_stub.cpp`, and
`peer_experts` as a second-GPU tier.

## 2. The layer split (what a "stage" is)

`--layer-split K[,K2..]|auto` with `--split-device`: stage 0 runs layers `[0, K)`, stage i runs
`[K_{i-1}, K_i)`, the last stage runs to the end **and holds the LM head, the MTP drafter and the sampler**
(the "Main" card in `docs/MULTI_GPU.md`).

- `struct GpuStage` (`src/program/generate.cpp:1215`, port `:1093`): `dev`, `lb`, `le`, its own `WeightTable`,
  `NativeDense`, `NativeHead`, `SessionState` (the layer-range K/V carve), `ExpertCache`, `d_res` residency
  table, `Verifier ver` + `ver_b` (the `--pipeline-windows` parity), and its own `Prefill sp`.
- Wiring is in one block (`src/program/generate.cpp:9040-9131`): the hand-off buffers are
  `cudaHostAlloc(Mapped|Portable)` + `cudaHostGetDevicePointer`, sized
  `rows * Verifier::handoff_floats(g) * sizeof(float)`; `set_stage(lb, le, in, out)` and
  `set_next(next, &SplitDrive)` chain the stages; `chain_overlap()` (`src/core/verify.cpp:2834`) is the
  overlapped variant and needs GPU-raised **host-visible** flags, unavailable under the port's
  `STRATA_VERIFY_NO_HOST=1`.
- The hand-off payload per token is `hc*n_embd + n_embd + hc` floats (residual R, the pending FFN write `bo`,
  `inject`): 12,804 floats ≈ 51 KB at hc=4, n_embd=2560 (the Flash-Next family, 48 layers).
- `SplitDrive` (`:1189`) picks the stage by **layer index** and swaps that stage's `GpuPlanSink`, cache base and
  PCIe share into the shared `Drive` before `expert_pool_dispatch_multi`. So each stage's expert misses are
  served by the host pool of the process that owns the stage.
- Capacity rules are pure functions in `include/strata/program/split_rules.hpp`: `kMinStageLayers = 3`,
  `share_floors`, `Placement`, `pick_with_headroom` (#1440), `link_miss_scale` (#2007), `min_spare`.
- Memory relief for a stage whose cache cannot hold its layers' experts: the **per-stage host mirror**
  (`STRATA_SPLIT_MIRROR`, `STRATA_MIRROR_MIB`, `docs/specs/intel-arc-per-stage-mirror.md`); stage weight trim is
  `--trim-stage-weights` / `STRATA_STAGE_TRIM=1`.
- Split K/V and checkpoints: session `SAVE`/`RESTORE` are refused with a split (`:12132`); in-RAM split parking
  exists (`conversation_checkpoints_split`/`merge`, `include/strata/core/conversation_cache.hpp:80`,
  `SavedConversation::stage_images`).

## 3. The expert side (vendor-neutral contracts, per-vendor tiers)

`ExpertSource` (`include/strata/core/expert_source.hpp:178`) is the vendor-neutral contract for "where one
routed expert's bytes come from": `blob`, `blob_stable`, `blob_parts`, `copy_blob(s)`, `pinned`,
`device_alias`, `prefetch`/`warm`, `pcie_layer`. Implementations: `FileExpertSource` (mmap `experts.bin`, or a
GGUF read in place), `ArenaExpertSource`, and the opt-in wrapper `RemotePrefillSource` that takes blobs from
another GPU's cache instead of the files.

Per-vendor expert tiers: `RemoteExperts` (`include/strata/core/remote_experts.hpp`, a static profile-filled
tier on another CUDA device, `cuda_runtime.h` in the header), `PeerExperts` (an adaptive second-GPU tier),
`RemoteExpertOpt`. The port has no equivalent (stubs). This is the seam a later out-of-process expert/mirror
worker (design B) plugs into.

## 4. The rig (this workstream)

```
serve/server.py ── engine CONTROLLER (vendor of the last stage)
                       └── StageWorker (the other vendor's engine binary, `--rig-worker`)
```

- One process per (vendor, device). The last stage stays in the controller (head, drafter, sampler).
- Control: `socketpair(AF_UNIX, SOCK_SEQPACKET)`, an inherited fd, `SCM_RIGHTS` for `memfd` hand-off rings.
  No listening socket in the first setup.
- Hand-off across the boundary is **copy-through-pinned**: the vendor-native pinned buffers stay as they are;
  a plain shm ring between them is bracketed by two host memcpys and guarded by `seq`/`epoch` (+ optional
  payload hash). No cross-process, cross-vendor device-visible memory in the MVP.
- New code: `include/strata/rig/{wire,link,shm,next_stage,rig,remote_stage,worker,handoff,planner}.hpp` and
  `src/rig/*.cpp`, vendor-neutral, reached from each tree through `RigStageHooks`.
- `Verifier::set_next` takes a `NextStage` callback struct instead of a `Verifier*`, so a `RemoteStage` proxy can
  answer `run`/`commit`/`wait_commit`/`copy_logits`/`run_slot_rows`/`capture` without a class hierarchy.

## 5. What the port cannot do (capability refusals)

Elastic K/V via VMM, `--peer-device`, `--remote-expert-opt`, GPU vision on the Arc card (the CPU reader
`--vision cpu` exists), and the overlapped split path under `STRATA_VERIFY_NO_HOST`. SYCL is
`linux-x86_64` only (`strata_control/targets.py::BACKEND_MATRIX`), and `strata_control/engine_component.py`
resolves exactly one engine backend per PC — a mixed rig needs a second engine component
(`strata_control/feed.py` already publishes packages per backend).
