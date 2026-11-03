# Design: Pooled mixed-vendor rig (whole-layer split across worker processes)

**Status**: 🔄 in-progress
**Story**: confirmed intent — opt-in llama.cpp-inspired RPC/worker capability to pool mixed-vendor GPU
memory for one inference workload; prove one host first (NVIDIA CUDA + Intel Arc SYCL), then networked
workers. Success gate: Qwen3.8-Flash-Next Q4_K_M or better, ≥131,072 context, correct/stable,
≥20 decode tok/s, record prefill and decode.
**Updated**: 2026-10-26
**Written for**: the Implement agent. No code is written by this document.

## Overview

Pool two graphics cards of different makes into one inference workload by running **one process per vendor
engine binary** and moving a whole-layer range of the model (its dense weights, its share of the K/V, its
expert cache and its prompt path) into the second process. The existing in-process layer split already owns
everything a stage needs (`GpuStage`, `Verifier::set_stage`/`set_next`, the per-stage session carve, the
per-stage expert cache and mirror, the per-stage `Prefill`); what is missing is only the **process boundary**:
a control channel, a shared-memory hand-off, a capacity handshake and a lifecycle. That is Design A, and it is
the recommended path.

## User story reference

The confirmed intent in the caller's prompt (no story file was provided). Restated as acceptance criteria in
[Gate](#9-gate-and-validation-strategy):

1. Opt-in; the default path stays byte-identical to the last release.
2. First setup: CUDA + SYCL on one Linux host. No mixed cluster, no AMD, no multi-host, no llama.cpp wire
   compatibility.
3. Whole-layer / whole-expert / coarse-block partitioning only — **no tensor splitting, no tensor parallelism**.
4. Q4_K_M (or better) of the Flash-Next family at ≥131,072 context runs correct and stable at ≥20 decode
   tok/s; prefill and decode recorded.

## Ground truth read from the repository (what the design stands on)

| Fact | Where |
|---|---|
| CUDA and SYCL are separate compile-time trees; the SYCL tree is a mirror resolved by `strata_resolve()` (root source when the port has no migrated copy) | `sycl/CMakeLists.txt:107-127`, `sycl/CMakeLists.txt:272` |
| A stage is already self-contained: `GpuStage{dev, lb, le, wt, dense, head, ss, cache, d_res, ver, ver_b, sp}` | `src/program/generate.cpp:1215`, `sycl/src/program/generate.cpp:1093` |
| Stage chain = `set_stage(lb, le, handoff_in, handoff_out)` + `set_next(next, user)`; the hand-off is per token `hc*n_embd + n_embd + hc` floats (R, pending bo, inject), pinned `cudaHostAlloc(Mapped|Portable)` + device alias | `include/strata/core/verify.hpp:167-194`, `src/program/generate.cpp:9040-9131` |
| `chain_overlap()` needs `next_->device_ != device_` and host-visible flag stores; the xe port runs `STRATA_VERIFY_NO_HOST=1`, so the GPU-raised host flag is not available there | `src/core/verify.cpp:2834`, `sycl/serve/strata-sycl-host.sh:29` |
| A stage's expert misses are served by that stage's own host pool over `SplitDrive` (per-stage plan/cache/PCIe share) | `src/program/generate.cpp:1189-1210` |
| Vendor-neutral expert source contracts already exist (`ExpertSource`, `FileExpertSource`, `ArenaExpertSource`, `PeerExperts`, `RemoteExperts`, `RemotePrefillSource`); `PeerExperts`/`RemoteExperts` are per-vendor (`cuda_runtime.h`) and stubbed in the SYCL port | `include/strata/core/expert_source.hpp:178`, `include/strata/core/remote_experts.hpp:20`, `sycl/CMakeLists.txt:259-263` |
| Placement rules are pure-CPU and testable: `kMinStageLayers`, `share_floors`, `Placement`, `pick_with_headroom`, `link_miss_scale`, `min_spare` | `include/strata/program/split_rules.hpp` |
| Session files (`SAVE`/`RESTORE`) are refused with `--layer-split`; in-RAM split parking exists (`conversation_checkpoints_split`, `SavedConversation::stage_images`) | `src/program/generate.cpp:12132`, `include/strata/core/conversation_cache.hpp:80-92` |
| Per-stage host mirror exists and is what let Q4_K_M start on the Arc pair; the head-card squeeze is "dense + head + MTP + KV", not the KV alone | `docs/specs/intel-arc-per-stage-mirror.md`, AGENTS.md WS5 |
| Engine ↔ server is one process per model over a line protocol (`READY`, `SERR`, `SWAIT`, `SESSION`, watchdog `ENGINE_STALL_S`); feature gating is done by grepping the engine binary for a flag name | `src/program/generate.cpp:12110-12132`, `serve/server.py:2960-2990` |
| Engine binaries ship as feed components, one entry per platform/backend/variant; the machine resolves exactly one `component:strata-engine` | `strata_control/feed.py:7`, `strata_control/engine_component.py:964` |
| Mixed makes are refused today by documentation ("two or more graphics cards of the same make") and by the single-backend component rule | `docs/MULTI_GPU.md:1-3`, `strata_control/engine_component.py:487` |
| The current pair numbers this design must beat/match: Q4_K_M, 131,072, `--layer-split 24`, 26.6 tok/s decode, 412 tok/s read of a 109,557-token prompt | AGENTS.md ("B580 removed 2026-10-17") |

## 1. Recommended architecture, and the alternatives that were rejected

### 1.1 The two candidates, compared on the things that decide it

| Axis | **A — whole-layer split across worker processes** (a stage lives in another process) | **B — out-of-process expert / mirror provider** (an `ExpertSource` + expert-row compute served to the controller) |
|---|---|---|
| What it pools | Dense weights of a layer range, that range's **K/V**, its expert cache, its pinned mirror, its prompt buffers, its PLE/PLE-shard share | Only expert bytes: a VRAM expert-cache tier and/or a host mirror. Dense weights and the whole K/V stay on one card |
| Can it pass the capacity gate | **Yes.** Q4_K_M at 131,072 needs dense + head + MTP + K/V + expert cache spread over the cards. `docs/specs/intel-arc-per-stage-mirror.md` measured that on the Arc pair the binding term is "dense weights + head + MTP + KV" and that dropping to 64K context barely moved the overflow — so expert pooling alone cannot open the gate | **No, not for this gate.** It is exactly WS5 (the mirror), which was necessary *beside* a layer split, not instead of one |
| Decode critical path | **One hand-off per token per boundary**: 12,804 floats ≈ 51 KB/token for hc=4, n_embd=2560, plus 2 control messages. At 20 tok/s that is ~1 MB/s and 40 msg/s | **48 round trips per token** (one per layer), each carrying up to `k` rows × `n_embd` floats in and out (≈100 KB/layer at k=10). Two orders of magnitude more synchronisation on the decode path |
| Where a failure lands | A layer **boundary**, before any `moe_combine` consumed partial rows | Mid-layer, into a half-filled `parts` — the reason `ExpertDispatch::failed` latches and never resumes (`include/strata/core/expert_source.hpp:652-656`) |
| Numerics across vendors | Each whole layer is computed by one vendor's kernels, self-consistent over its range. Reproducible against *itself*; not byte-identical to a single-vendor run (as the intent already says) | A layer's router rows are summed across vendors inside one `moe_combine`, so vendor differences appear *inside* one layer; no parity test is meaningful and a change of expert placement changes bits |
| Reuse of what exists | `set_stage`, `set_next`, `handoff_floats`, per-stage session carve, per-stage `Prefill`, `--prefill-pipe`, `--pipeline-windows`, `--trim-stage-weights`/`STRATA_STAGE_TRIM`, per-stage mirror, `split_rules.hpp`, `conversation_checkpoints_split`, `--layer-split`/`--split-device` config plumbing, and the WS2/WS5/WS6 measurements on this exact hardware | `ExpertSource` (vendor-neutral), `RemoteExperts::copy_cached_blobs`, `RemotePrefillSource`, the mirror machinery. But `RemoteExperts`/`PeerExperts` are per-vendor and stubbed in the SYCL port (`sycl/CMakeLists.txt:259-263`) |
| Effort to first proof | Medium: process + transport + hand-off plumbing; no kernel work | Low for a data-only mirror; high for a compute tier that is fast enough, because the per-layer host sync must beat the CPU pool it replaces |
| Room for the other path | **Excellent.** A's link/handshake/capacity/lifecycle is exactly what a later B worker needs: "give me these (layer, expert) blobs / these rows" is a new message type on the same channel, and the planner gains a second resource term | Poor. B builds no stage lifecycle, no hand-off, no per-stage capacity negotiation, so A would still be unpaved afterwards |

**Decision: build A now, and shape its protocol so B is a message-type extension later** (milestone M5).
A is chosen for the two facts that decide it — capacity (the gate is a dense+KV problem, not only an expert
problem) and synchronisation frequency (51 KB once per token per boundary against 48 round trips per token).

### 1.2 Process topology (A)

```
serve/server.py                       unchanged: exactly one engine per model, same line protocol
  └── engine CONTROLLER process       the vendor binary of the Main card (CUDA in the first setup)
        │  owns: the line protocol, sampler, the last stage (head + MTP + drafter), stage 0 or later stages
        │  owns: the rig — spawn, handshake, capacity plan, hand-off rings, teardown, watchdog relay
        └── StageWorker process       the other vendor's binary, run as `strata --rig-worker`
              owns: one layer range [lb,le): its dense weights, its K/V carve, its expert cache + mirror,
              its host expert pool, its Prefill, its own pinned buffers
```

Rules, and why:

1. **One process per vendor binary.** The engine has no runtime multi-vendor backend; a CUDA process cannot
   enumerate an Arc and a `level_zero` process cannot enumerate a GeForce. The worker is literally the *other*
   engine file (`build-sycl/strata`, or the feed component for that backend), so mixed makes need no new
   binary form and no universal backend.
2. **One process per (vendor, device) pair.** A process may hold several stages of the same card. The first
   setup is exactly two processes.
3. **The last stage stays in the controller** (head, MTP drafter, sampler, `copy_logits`, `--batch-mtp` slot
   drafters all live on the last stage; keeping them in-process keeps the protocol and the sampler single-owner).
   `--worker` on the last stage is REFUSED with that reason. This makes the Arc the **first** stage
   (layers `0..K`), which also matches "the first card runs the first layers and, by default, the picture
   reader" (`docs/MULTI_GPU.md`) — and for a mixed rig we recommend `--vision cpu` (WS1's CPU reader) so the
   Arc card is not also an image encoder.
4. **The controller is the parent.** It spawns the worker with an inherited socketpair fd; the worker exits on
   `PR_SET_PDEATHSIG` or on socket HUP, so no orphan keeps 20 GiB of VRAM.
5. **Card visibility is separated by environment**, not by code: the child gets `CUDA_VISIBLE_DEVICES` or
   `ONEAPI_DEVICE_SELECTOR` for its own card only (the pair config already uses
   `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`, `strata-swift-q4_k_m-split.json`). Stage → card numbering stays the
   `--split-device` notion the server already writes.
6. **Server, Control, app and the Terminal keep seeing one engine.** Nothing in the line protocol changes; the
   worker never talks to the server.

### 1.3 Transport (the decision that removes the hard part)

**Control channel:** `socketpair(AF_UNIX, SOCK_SEQPACKET)`, one end inherited by the child
(`STRATA_RIG_SOCK=<fd>`); no listening path on the filesystem, so nothing else can attach. SEQPACKET keeps
message boundaries, which the codebase's "a wrong token beats a stalled engine" style prefers over a hand-rolled
stream framer. Windows is refused in the first setup (named pipes later).

**Data plane:** `memfd_create` + `SCM_RIGHTS` fd passing, **one segment per stage boundary** (max 7 boundaries,
`SplitDrive::kMax = 8`).

**Hand-off crossing: copy-through-pinned, not shared device-visible memory.** Each side allocates its hand-off
buffers exactly as today (`cudaHostAlloc(Mapped|Portable)` + `cudaHostGetDevicePointer`; `sycl::malloc_host` in
the port), and the process boundary is crossed by host `memcpy` between the vendor-native pinned buffer and the
plain shm ring:

```
stage N's GPU  --writes-->  stage N's pinned hand-off (vendor-native, unchanged)
       writer link thread: memcpy pinned -> shm slot, stamp seq, (optional payload hash), raise slot flag
       reader link thread: poll slot seq, memcpy shm -> its own pinned hand-off, raise ITS OWN device flag
stage N+1's GPU  --reads-->  its pinned hand-off (unchanged code path)
```

Why, in order:

- It needs **no cross-process, cross-vendor memory registration**. Vendor host-USM (`sycl::malloc_host`) has no
  fd to share, and Level Zero has no host-register equivalent of `cudaHostRegister`; inventing one is the
  riskiest part of the feature and it buys nothing measurable (below).
- The payload is tiny: `Verifier::handoff_floats` = `hc*n_embd + n_embd + hc` = 12,804 floats ≈ **51 KB per
  token** at hc=4, n_embd=2560. Two 51 KB memcpys are ~10 µs; a token budget at 20 tok/s is 50 ms. The copies
  cost **~0.02 %**, and 0.1 ms per token even if the copy is 10× worse than that estimate (unconfirmed
  estimate — the first mixed run measures it with `--stage-timing` and the rig counters).
- Bit behaviour is preserved by construction: the floats are copied verbatim, and each stage's kernels are
  untouched. Parity against the in-process split is then testable, which it would not be if a foreign runtime
  mapped the same pages.
- A zero-copy variant (`cudaHostRegister` on a memfd; a Level Zero import if the probe finds one) is kept as a
  later optimisation behind the probe in T1; the ring layout does not change if it lands.

**Overlap across processes is deliberately out of the first build.** Today's default split path is synchronous
per boundary and that is what measures 26.6 tok/s on the Arc pair; `STRATA_SPLIT_OVERLAP`/`chain_overlap` needs
GPU-raised **host-visible** flags, and the xe port runs `STRATA_VERIFY_NO_HOST=1` because the i915/xe driver
does not make those stores visible (`sycl/serve/strata-sycl-host.sh:29`). So `--worker` + overlap is REFUSED in
M2–M4 and becomes an optimisation milestone (M5, T16) with its own probe.

**Later (networked) workers:** the same frames over a stream transport. The first setup is one host, so the
first setup ships no listening socket at all.

### 1.4 Protocol, versioning, and refusals

Frame header (24 B, fixed, all scalars little-endian):

```c
struct RigHeader {           // 24 bytes
    uint32_t magic;          // 'SWRK'
    uint8_t  proto;          // RIG_PROTO, 1 for this build
    uint8_t  type;           // RigMsg below
    uint16_t flags;
    uint32_t seq;            // per-direction, wraps; a gap is a hard error
    uint32_t len;            // payload bytes after the header (0..64 KiB; bigger payloads live in shm)
    uint64_t epoch;          // the controller's run id: kills a stale worker / reused shm
};
```

Messages (one `type` per behaviour the in-process chain performs today, because the chain is what has to be
reproduced across the boundary): `HELLO`/`HELLO_ACK` (proto, `build_id`, vendor, device, capability bits,
geometry hash), `CAPACITY` (free MiB, probed link GB/s, per-stage dense/session/cache byte model, lend-region
bytes), `CONFIGURE`/`CONFIG_ACK`/`REFUSE`, `SHM_FD`, `READY`, `RUN_WINDOW`, `WINDOW_DONE`, `COMMIT`,
`WAIT_COMMIT`, `ADAPT`, `LOGITS`, `LOGPROBS`, `PARK`/`RESTAGE`, `PROGRESS`, `STATS`, `PING`/`PONG`,
`TEARDOWN`/`QUIT`, `ERROR`.

Rules that mirror existing project habits:

- **Version skew is a start-up refusal with both numbers**, never a silent degrade (`proto 1 != 2`, like
  `FileExpertSource`'s size refusal).
- **Geometry/pack identity** is carried in `CONFIGURE` (a 64-bit hash of geometry + the pack's expert blob size
  + the native GGUF tensor names the profile was made for) so a worker that opened a different pack refuses
  instead of computing a plausible wrong token.
- **Capability bits decide refusals at `CONFIGURE`**, not at run time: elastic VMM (`vmm.cpp` is a stub in the
  port), `--peer-device`, `--batch`/`--batch-mtp`/`--batch-groups`, GPU vision, pipeline overlap, host-visible
  flags. The first mixed setup refuses them with a reason naming the flag to drop.
- The server gates the feature by grepping the engine binary for `--rig-worker` — the pattern already used for
  `--tail-role-token` (`serve/server.py:2960`).
- **No llama.cpp wire compatibility**: frames are Strata's own, `SWRK` is not interoperable with anything.

### 1.5 Failure semantics

The governing rule is the one already written for `ExpertDispatch::failed`: a failure **latches**; resuming into
a plausible-looking token is the outcome every guard exists to prevent.

| Event | Detection | Behaviour |
|---|---|---|
| Worker died | `recv` 0 / `HUP`/`ERR` on the SEQPACKET, or `PONG` missed twice | Latch the rig; abort the in-flight request with `SERR worker-lost <stage> <why>`; emit no further token of that window; flush the evidence line; exit non-zero so the server's existing restart ladder (`ENGINE_QUIT/TERM/KILL_WAIT_S`) restarts the rig |
| Hand-off never arrives | bounded wait (`--worker-timeout-s`, default 30 s; the in-process split already has a bounded hand-off wait and `herr_h_`) | same latch; the watchdog names the stage and the phase, like `stage_text()` does today |
| Worker alive but slow | `PROGRESS` relayed into `progress_at()` (`where`/`detail`) | the controller keeps the heartbeat so `ENGINE_STALL_S` does not kill a rig that is loading 60 GiB; the stall line names the stage and card |
| Controller died | worker sees `HUP`, and `PR_SET_PDEATHSIG(SIGTERM)` | worker frees its device and exits within `--worker-exit-timeout-s` (default 10 s); no VRAM orphan |
| Capacity will not fit | `CAPACITY` + planner | one `REFUSED:` line with both numbers and the flags that would fix it (`--layer-split`, `--vram-reserve-mib`, `--worker-resident-budget-gib`, `--kv-resident`); nothing partial starts |
| Window/token mismatch | `seq` gap, epoch mismatch, ring slot stamp mismatch | latch as above; never silently re-request |

No automatic fallback: a rig never quietly drops a stage, quietly changes the cuts, or quietly becomes
single-card. Refusing loudly is the default path's contract, and the default path never runs any of this code.

### 1.6 Startup and weight loading (ordered, with prints the server can watch)

1. Controller parses `--worker EXE` + `--layer-split` cuts + `--split-device` and **refuses the unsupported set
   first** (Windows, remote last stage, `--batch*`, `--peer-device`, session files, overlap).
2. Controller creates the socketpair, `fork`+`exec`s the other vendor's binary with `--rig-worker
   <fd>` plus the same artifact arguments (`--pack`, `--native`, `--ple-gguf`, `--mtp`, tokenizer), the stage's
   device-selector environment, and its own `--rig-stage` id. `execv` in a small helper so the engine's own
   GPU state is not inherited beyond fds.
3. `HELLO`/`HELLO_ACK`: proto, `build_id` (the same `+cuda12.6` / `+sycl` tag `bench_facts.py` parses),
   vendor, device, free MiB from the vendor's own query, capability bits.
4. `CAPACITY`: worker reports free MiB, probed H2D GB/s (the `link_miss_scale` input), the byte model it would
   spend per candidate layer range (dense bytes for the range from the pack index, session bytes from the
   existing session math, expert bytes from `need_prefix`). The controller never guesses a foreign device's room.
5. The planner (T11b) picks cuts/slots/mirror/`kv_resident`/pool cores → `CONFIGURE`.
6. Worker maps the pack the same way the controller does (`experts.bin` or the GGUF shards read in place): the
   **page cache is shared**, so the 60 GiB arena is read from disk once, not twice. Only its own layers' dense
   weights are loaded (`--trim-stage-weights`, `STRATA_STAGE_TRIM=1` precedent).
7. Worker allocates its K/V carve, its expert cache from `--expert-profile`, and its mirror **within the byte
   budget the controller gave it** (see the RAM risk below), then answers `READY` with its numbers
   (slots, resident pairs, mirror GiB, K/V slots, pinned GiB).
8. Controller wires `set_stage`/`set_next` with the remote proxy, prints its `layer split: layers …` line
   extended with the process/vendor of each stage, then `READY` to the server. `engine_ready_s` (default
   `ENGINE_READY_S` 900 s) covers both loads; a mixed config should raise it, and `apply_ready_timeout` exists.

### 1.7 Placement, capacity and the CPU budget

- **Whole layer ranges only.** Cuts are layer boundaries; the planner never splits a tensor. `kMinStageLayers=3`
  and `share_floors` stay, and the pair's measured lesson (a stage starved of layers starves the prompt pipeline,
  #1616/#2644) keeps its weight.
- **A new vendor term.** Today's search scores placements by predicted window time with a fitted per-card cost.
  A mixed rig adds the asymmetry term: a slow card with many layers is the serial bottleneck (or, with overlap
  later, the pipeline bottleneck). New pure function `vendor_stage_cost(layers, vendor_ms_per_layer, misses,
  link_scale)`; `vendor_ms_per_layer` starts as a small measured table (CUDA and xe entries seeded from the
  repository's own measurements) or from `--calibrate`, and `--layer-split` explicit cuts bypass the search
  entirely (the supported first path). New guard: `vendor_max_layers` — never give a slower vendor more layers
  than its measured share of the token budget allows, and `REFUSED` when even the minimum 3 layers cannot fit
  its VRAM.
- **Per-stage memory terms stay the ones the split already uses**: `min_spare` (room − session − expert need),
  per-stage mirror (`STRATA_SPLIT_MIRROR`), `--vram-reserve-mib`, `--kv-resident`. The worker's `CAPACITY`
  replaces the local probe for its card.
- **CPU expert pool partition (new, and the quiet way this feature can look slow).** Today the pool takes every
  physical core but one (`--pool-workers` default 0). Two processes would each take all cores and oversubscribe
  the decode path. The controller therefore partitions cores and passes the worker an explicit
  `pool_workers` + affinity set in `CONFIGURE` (`--worker-pool-cores` for the operator override), and the
  evidence line reports both pools' counts.
- **Host RAM/pinned budget partition (new).** `host_available_memory`/`clamp_resident_budget` are per-process
  views of the same `MemAvailable`: two processes that each "fit" can together exceed it (the #1250 OOM-kill
  failure mode). The controller computes the split (its own reservation first, the remainder as the worker's
  `resident_budget_bytes`/`mirror_bytes`), and the worker's `READY` reports what it actually pinned so the
  controller's later adaptive steps account for it.

### 1.8 KV, prompt cache and checkpoints

- **K/V is per stage and stays per stage** — the worker's K/V lives on its card, which is where the 131,072
  context capacity comes from. `--kv`, `--kv-resident` are split per stage by the planner; the pair's measured
  `--kv-resident 65536` arm is the reference for the xe stage.
- **Elastic K/V (`--vram-elastic`, CUDA VMM) is refused for a SYCL worker** (its `vmm.cpp` is a stub).
- **Session `SAVE`/`RESTORE` stay refused** with a worker rig, as with `--layer-split` today
  (`generate.cpp:12132`); the refusal text gains "and a stage worker".
- **In-RAM prompt cache / parking is milestone M4.** The state form already exists (`conversation_checkpoints_split`,
  `SavedConversation::stage_images`) — across a process boundary it becomes `PARK`/`RESTAGE` carrying those
  bytes. Until M4, a worker rig **refuses to park and runs with `--prompt-cache 0`** and says so at start, so a
  benchmark on such a rig is honest about being cold-read every request. This is the one deliberate functional
  gap in the gate milestone, called out in §9.
- `--prefill-pipe R` and the #448 chunk clamp must clamp on the **worker's** lend region, which the worker
  reports in `CAPACITY` (the in-process rule is `append_stage_lend_regions`).

### 1.9 Alternatives rejected (each with the reason, not just a name)

1. **B as the first build** (out-of-process expert/mirror provider): cannot pass the capacity gate (§1.1), and
   puts 48 host round trips per token on the decode path. Kept as M5.
2. **One fat multi-vendor binary with dlopen'd runtimes**: no runtime multi-vendor backend exists today, the
   trees are compile-time forks, and a CUDA process cannot enumerate an Arc. It would also put both runtimes in
   one address space, where a driver fault takes the whole rig.
3. **Cross-process device-visible memory as the MVP transport**: the vendor-specific registration problem
   (§1.3) for ~0.02 % of a token budget. Probe exists (T1); the optimisation is M5.
4. **A separate coordinator process with N symmetric engine peers**: the server already models one engine per
   model with a line protocol and a watchdog; a third process would have to re-implement READY/stall/restart
   semantics and would change the default path's process shape for a feature that is opt-in.
5. **Two independent servers with a router at the HTTP layer** (llama.cpp's `rpc-server`-style per-layer
   microservice): puts the MoE/KV state across a request boundary, breaks the conversation cache, and cannot
   keep one workload's K/V consistent.
6. **Tensor parallelism / row-column splitting of a layer across cards**: excluded by the intent, and it would
   add per-layer traffic an order of magnitude above the whole-layer hand-off for a link-bound mixed rig.
   The planner has no code path for it.
7. **AMD/HIP in the first setup**: the intent excludes it; `hip` is a third tree and `PeerExperts` is a stub in
   the port. The design carries a `vendor` field so it is a config/plumbing step, not a redesign.
8. **Multi-host (networked) first**: intent says one host first; latency and authentication are separate
   problems, and the frame format is already transport-neutral so this is a later transport swap.

## 2. Interfaces and data structures

New headers live in `include/strata/rig/` and sources in `src/rig/`, in the **root** tree only: `strata_resolve()`
picks the root source when the SYCL port has no migrated copy (`sycl/CMakeLists.txt:107-127`), so the rig layer
is written once and is vendor-neutral by construction. Vendor-specific parts reach it through small
function-pointer structs, the pattern the codebase already uses for `PoolFn`, `GpuPlanSink::publish/fetch` and
`ExpertDispatch`.

### 2.1 Wire (`include/strata/rig/wire.hpp`, header-only, no GPU, no vendor include)

```cpp
namespace strata::rig {

inline constexpr uint32_t kMagic = 0x4b525753;      // 'SWRK'
inline constexpr uint8_t  kProto = 1;               // bump on any incompatible frame or message change
inline constexpr size_t   kMaxStages = 8;           // matches SplitDrive::kMax
inline constexpr uint32_t kMaxFrame = 64u << 10;    // bigger payloads travel through shm, not the socket

enum class Msg : uint8_t {
    Hello = 1, HelloAck = 2, Capacity = 3, Configure = 4, ConfigAck = 5, Refuse = 6,
    ShmFd = 7, Ready = 8, RunWindow = 10, WindowDone = 11, Commit = 12, WaitCommit = 13,
    Adapt = 14, Logits = 15, Logprobs = 16, Park = 20, Restage = 21, Teardown = 22, Quit = 23,
    Ping = 24, Pong = 25, Progress = 26, Stats = 27, Error = 28,
};

struct Header { uint32_t magic; uint8_t proto; uint8_t type; uint16_t flags;
                uint32_t seq; uint32_t len; uint64_t epoch; };   // 24 B, little-endian, static_assert

/// What a worker may and may not be asked to do. Decided at CONFIGURE, never discovered at run time.
struct Capabilities {
    uint64_t bits = 0;       // bit per feature; see kCapVmm, kCapHostFlags, kCapOverlap, kCapBatch,
                             // kCapBatchMtp, kCapBatchGroups, kCapPeerTier, kCapGpuVision, kCapPark, kCapMirror
    bool has(uint64_t b) const { return (bits & b) == b; }
};

struct HelloAck {
    uint8_t  proto;                  // the worker's kProto
    char     build_id[32];           // the same tag bench_facts.py parses ("+cuda12.6", "+sycl")
    char     vendor[8];              // "cuda" | "hip" | "sycl" | "metal" | "cpu"
    int32_t  device;                 // the card this process owns, in the engine's numbering
    uint64_t epoch;
    Capabilities cap;
    uint64_t geometry_hash;          // see GeometryIdentity
    int64_t  free_mib;               // this card's free device memory, from the vendor's own query
    double   link_gbps;              // probed host->device, the link_miss_scale input
};

/// Carried in CONFIGURE: the pack this cut was planned for. A mismatch REFUSES the start; it is the same class
/// of guard as FileExpertSource refusing a short experts.bin.
struct GeometryIdentity {
    int64_t n_layers, n_embd, n_expert, hc;
    uint64_t expert_bytes_total;     // experts.bin size, or the GGUF tensor-name hash when read in place
    uint64_t profile_hash;           // the --expert-profile this stage's cache is filled from
    uint64_t hash() const;
};

/// What the worker offers and what the controller then commands. Every number the in-process split computes
/// locally for its own card, computed by the worker for its card and accepted by the planner.
struct StagePlan {
    int64_t lb, le;                    // whole layers; the only partition this feature can express
    int64_t expert_slots;              // 0 = --no-expert-cache on this stage
    uint64_t mirror_bytes;             // the stage's share of the host mirror (0 = none)
    uint64_t resident_budget_bytes;    // the stage's share of page-locked RAM (the RAM-guard split)
    int64_t kv_cells, kv_resident;     // its K/V carve
    int32_t pool_workers;              // the CPU expert pool's core count on this side
    int64_t lend_from_slot;            // the prompt path's lend region (#448 chunk clamp)
    int32_t reserve_mib;               // device memory left free on its card
};

struct Refuse { int32_t code; char why[256]; };   // one line, with both numbers and the flag to change
```

Refusal codes: `kRefuseProto`, `kRefuseGeometry`, `kRefuseCapacity`, `kRefuseCapability`, `kRefuseRam`,
`kRefuseLayers` (a stage cannot hold `kMinStageLayers`), `kRefuseRuntime` (the vendor cannot run this build on
that card — the WS0 failure class).

### 2.2 Control link (`include/strata/rig/link.hpp`, `src/rig/link.cpp`)

```cpp
class Link {                        // one SEQPACKET unix socket, one worker
public:
    static bool make_pair(Link& controller_side, int& child_fd, std::string& err);   // socketpair(AF_UNIX, SOCK_SEQPACKET)
    bool send(Msg type, const void* payload, size_t len, std::string& err);
    /// Waits up to `timeout_ms`; 0 = a closed peer (the worker died). `seq` gaps are reported as an error.
    bool recv(Msg& type, std::vector<uint8_t>& payload, int timeout_ms, std::string& err);
    bool send_fd(int fd, std::string& err);                       // SCM_RIGHTS (memfd for a hand-off ring)
    bool recv_fd(int& fd, std::string& err);
    bool peer_uid(uid_t expected, std::string& err);              // SO_PEERCRED, defensive
    void close();
    int fd() const;                                              // for poll() in the controller's loop
};
```

No threads inside `Link`; the controller's rig thread owns it (see 2.5).

### 2.3 Hand-off ring (`include/strata/rig/shm.hpp`, `src/rig/shm.cpp`)

```cpp
/// One producer, one consumer, one segment per stage boundary, mapped in both processes.
/// Sized from Verifier::handoff_floats(g) * rows * sizeof(float), rows = kVerifyMaxT (or --batch slots,
/// exactly as the in-process split sizes today's cudaHostAlloc hand-off at generate.cpp:9045).
class HandoffRing {
public:
    static int  create(size_t bytes);                            // memfd_create + ftruncate; returns the fd
    static std::unique_ptr<HandoffRing> open(int fd, size_t bytes, std::string& err);
    struct SlotHdr { uint64_t seq; uint32_t rows; uint32_t bytes; uint64_t payload_hash; };  // 24 B, one per row
    uint8_t* slot(size_t row);                                   // the row's payload
    void publish(size_t row, uint64_t seq, bool hash);           // release-store of seq after the copy
    /// Bounded: false on timeout (the caller latches). Fills `out` with exactly the bytes written.
    bool acquire(size_t row, uint64_t expect_seq, void* out, size_t bytes, int timeout_ms, bool check_hash);
    uint64_t epoch;                                              // stamped at create; a mismatch is fatal
};
```

`SlotHdr::seq` is the guard against the failure this project fears most — a stale or half-written hand-off
producing a plausible token: `acquire` refuses on a seq mismatch, and when `check_hash` on, on a payload hash
mismatch (FNV-1a over the row; opt-in per run, measured before it is made the default).

### 2.4 The chain surface, factored out of `Verifier` (`include/strata/rig/next_stage.hpp`)

Today `Verifier::set_next(Verifier*, void*)` and `next_->run(...)` are a direct call inside the host loop
(`src/core/verify.cpp:3107, 3163, 3438, 3452, 3811, 3875, 4144`). The change is to make that member a small
callback struct — the same shape as `PoolMultiFn`/`GpuPlanSink` — so a `RemoteStage` can answer it without a
class hierarchy and without a virtual call in the token path:

```cpp
struct NextStage {
    void* self;
    bool (*run)          (void* self, int T, const int32_t* tokens, int64_t pos0,
                          PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    bool (*commit)       (void* self, int n_keep, std::string& err);
    bool (*wait_commit)  (void* self, std::string& err);
    bool (*run_slot_rows)(void* self, const int* rows, int S, const int32_t* tokens, const int64_t* pos,
                          PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    bool (*commit_slot_prefixes)(void* self, const int* keep, std::string& err);
    bool (*copy_logits)  (void* self, int t, float* host);
    bool (*window_logprobs)(void* self, const int32_t* targets, int T, int64_t pos0, int32_t extra_id,
                            std::FILE* out, std::string& err);
    bool (*set_logit_bias)(void* self, const std::vector<float>& bias, std::string& err);
    bool (*capture)      (void* self, int T, std::string& err);
    bool (*capture_commit)(void* self, std::string& err);
};
```

`Verifier` keeps its methods unchanged and gains `static const NextStage kSelfStage;` (its own functions are
plain static wrappers taking `self = Verifier*`); `set_next(const NextStage&, void*)` keeps the old call sites
working with the same compiled code path. **Default-path parity is a test, not an assertion** (T8: compare
`--window-hashes` of an in-process split before and after the refactor; the flags `--window-hashes` and
`--spec-follow` exist precisely for this).

### 2.5 Controller side (`include/strata/rig/rig.hpp`, `src/rig/controller.cpp`, `src/rig/remote_stage.cpp`)

```cpp
struct WorkerSpec {                  // one --worker option
    std::string exe;                 // the other vendor's engine binary (path or resolved component)
    int32_t device;                  // the card it owns (the --split-device number)
    int32_t stage;                   // which split stage lives there (the LAST stage is refused)
    std::vector<std::string> extra_args;
    int pool_workers = -1;           // -1 = the controller decides
    int64_t resident_budget_bytes = -1, mirror_bytes = -1;   // -1 = the planner decides
};

struct Rig {                         // the controller's owner of everything cross-process
    uint64_t epoch;
    std::vector<Worker> workers;     // pid, Link, HandoffRing in/out, thread, latched failure
    std::atomic<bool> failed{false}; std::string fail;   // latches, exactly like ExpertDispatch::failed
    void abort_current(const char* why);                 // -> SERR worker-lost ... , exit non-zero
};

/// The proxy that answers NextStage for an out-of-process stage. Owns: the shm<->pinned copies (2.3), the
/// message pair for a window (RunWindow/WindowDone), commit forwarding, and the bounded wait.
class RemoteStage {
public:
    void set_handoff(void* pinned_in, void* pinned_out, int rows, int64_t floats_per_row);
    /// the pinned buffers are the ones generate.cpp already allocates for that stage: the copy brackets them
    static const NextStage kStage;      // wired into Verifier::set_next for a worker stage
    bool healthy(std::string& err) const;
    RigStats stats;                     // copied over: ms_link, ms_copy, bytes, windows, timeouts
};
```

### 2.6 Worker side (`src/rig/worker.cpp` + a per-tree hook struct)

The protocol and the message loop are shared; the vendor-specific act of building a stage stays in each tree's
`generate.cpp`, reached through hooks so the shared file includes no vendor header:

```cpp
struct RigStageHooks {               // filled by src/program/generate.cpp and by sycl/src/program/generate.cpp
    void* user;
    bool (*preflight)(void*, int device, double& free_gib, std::string& err);   // the vendor's own probe
    bool (*link_probe)(void*, double& gbps, std::string& err);
    bool (*build_stage)(void*, const StagePlan&, std::string& err);             // the stage code at generate.cpp:9094+
    void (*handoff_pointers)(void*, void*& in, void*& out, int& rows);          // the pinned buffers to bracket
    bool (*run_window)(void*, int T, const int32_t*, int64_t pos0, std::string& err);
    bool (*commit)(void*, int n_keep, std::string& err);
    bool (*park)(void*, std::vector<uint8_t>& stage_image, std::string& err);   // M4
    bool (*restage)(void*, const uint8_t* stage_image, size_t n, std::string& err);
    void (*progress)(void*, const char* where, int64_t detail);
};
bool rig_worker_main(int argc, char** argv, const RigStageHooks& hooks);   // `--rig-worker <fd>`
```

### 2.7 Placement (`include/strata/rig/planner.hpp`, header-only like `split_rules.hpp`)

```cpp
struct StageRoom { int64_t free_bytes; double link_gbps; int64_t vendor_ms_per_layer_x1000; };
struct MixedPlan { std::vector<int64_t> cuts; std::vector<StagePlan> stages; std::string refuse; };
/// Reuses split_rules::share_floors / min_spare / pick_with_headroom / link_miss_scale, and adds the vendor
/// term plus the RAM and CPU-core partition. Testable with no GPU, like tests/program/split_rules_test.cpp.
MixedPlan plan_mixed_rig(const ModelGeometry&, const std::vector<StageRoom>&, int64_t max_context,
                         const std::vector<int64_t>& need_prefix, const std::vector<WorkerSpec>&,
                         uint64_t host_ram_available, int host_cores, const std::string& explicit_cuts);
```

### 2.8 Names an operator sees

| Surface | Name | Meaning |
|---|---|---|
| engine flag | `--worker EXE[:STAGE]` (repeatable) | run that split stage in another process, from that engine binary |
| engine flag | `--rig-worker FD` (internal, added by the controller) | this process is a stage worker |
| engine flag | `--rig-probe` | handshake, report both cards' capacity, exit 0 before loading weights (bring-up and `--plan`) |
| engine flag | `--worker-reserve-mib N`, `--worker-resident-budget-gib N`, `--worker-mirror-gib N` | the worker card's memory budgets |
| engine flag | `--worker-pool-cores N` | the worker's CPU expert pool size (the oversubscription guard) |
| engine flag | `--worker-timeout-s N` (30), `--worker-exit-timeout-s N` (10) | the bounded waits |
| env (diagnostics) | `STRATA_RIG_HASH=1`, `STRATA_RIG_STATS=1` | payload hash verify; per-boundary copy/link timing |
| unchanged | `--layer-split`, `--split-device`, `--vram-reserve-mib`, `--kv-resident`, `--expert-profile` | keep their exact meaning; `--worker` only moves a stage into a process |
| run config | `"stage_workers": [{"stage": 1, "backend": "sycl", "device": 1}]`, `"worker_budgets": {...}` | written by `strata launch`, read by `serve/server.py` into engine args |
| Terminal | `strata launch --worker sycl` (a card role line: `--gpu-roles 1=helper+worker`), `strata launch --plan` shows the rig | the Cards step gains a "separate process (this make)" line |

The server resolves a worker engine by backend through the component machinery (`component:strata-engine-<backend>`),
because `engine_component.py` today activates exactly one backend per PC (line 487) while the feed already lists
engine packages per backend (`strata_control/feed.py:7`).

## 3. Files to create and modify

### 3.1 New files

| File | Purpose | Key responsibilities |
|---|---|---|
| `include/strata/rig/wire.hpp` | the frame, the messages, the capability bits, the refusal codes | shared by both trees; no vendor include; header-only |
| `include/strata/rig/link.hpp`, `src/rig/link.cpp` | SEQPACKET control channel | `socketpair`, framing + `seq`/`epoch` checks, `SCM_RIGHTS`, `SO_PEERCRED`, poll-with-deadline, HUP |
| `include/strata/rig/shm.hpp`, `src/rig/shm.cpp` | the hand-off ring in `memfd` shm | create/open/share, per-row slots, seq stamps, bounded `acquire`, optional payload hash |
| `include/strata/rig/next_stage.hpp` | the chain surface as a callback struct | lets `RemoteStage` replace `Verifier*` in `set_next` with no virtual call |
| `include/strata/rig/rig.hpp`, `src/rig/controller.cpp` | the controller's rig owner | spawn/execv the other binary, env per card, `HELLO`/`CAPACITY`, `CONFIGURE`, teardown order, the latch |
| `include/strata/rig/remote_stage.hpp`, `src/rig/remote_stage.cpp` | the proxy stage | `NextStage` implementation, pinned↔ring copies, `RunWindow`/`WindowDone`, `Commit`, timing counters |
| `include/strata/rig/worker.hpp`, `src/rig/worker.cpp` | the `--rig-worker` message loop | handshake, version/geometry refusal, capacity report, drive the vendor hooks, HUP/`PDEATHSIG` exit |
| `include/strata/rig/handoff.hpp`, `src/rig/handoff.cpp` | the copy brackets | pinned→ring and ring→pinned, row layout identical to `Verifier::handoff_floats`, counters |
| `include/strata/rig/planner.hpp` | mixed-rig placement + budgets | cuts, per-stage slots/mirror/KV/pool-cores, RAM split, refusal text |
| `tests/rig/wire_test.cpp` | codec, version skew, truncated/oversize frame, epoch | no GPU |
| `tests/rig/link_test.cpp` | socketpair round trip, fd passing, peer death | fork arm |
| `tests/rig/shm_test.cpp` | ring wrap, seq/hash guards, bounded wait | fork arm |
| `tests/rig/handoff_test.cpp` | byte-exact pinned→ring→pinned over the real row layout | no GPU |
| `tests/rig/planner_test.cpp` | cuts, `kMinStageLayers`, vendor caps, RAM/core split, refusal strings | no GPU |
| `tests/rig/handshake_test.cpp` | a mock worker: hello, refuse-on-proto, refuse-on-geometry, capacity | no GPU |
| `probe/rig_shm_register.cu` | does `cudaHostRegister` take a memfd region | decides the M5 zero-copy optimisation |
| `sycl/probe/rig_shm_register.dp.cpp` | can Level Zero make plain shm device-visible | same question on the port (precedent: `sycl/probe/mirror_xcard.cpp`) |
| `docs/MIXED_VENDOR.md` | the feature for a non-technical reader | what it is, the two cards, what is refused, the measured numbers with their hardware |

### 3.2 Modified files

| File | Change | Reason |
|---|---|---|
| `include/strata/core/verify.hpp`, `src/core/verify.cpp` | `set_next(const NextStage&, void*)`; `chain_overlap`/`hflag_*` untouched | the boundary replaces the direct `next_->run` call |
| `src/program/generate.cpp` | parse `--worker*`; spawn; build `RemoteStage` in the `n_stages > 1` block (9040-9131); refusal gates; the split-plan print; the evidence line; rig counters in `--stats` | the controller side |
| `sycl/src/program/generate.cpp` | the worker branch: `--rig-worker` + `RigStageHooks` for the port's device/cache/mirror | the worker side of the first setup |
| `include/strata/program/split_rules.hpp` | accept a per-stage vendor cost vector (no rule changes) | the mixed asymmetry term |
| `include/strata/core/conversation_cache.hpp`, `src/core/conversation_file.cpp` | park/resume across a boundary (M4), reusing `conversation_checkpoints_split`/`stage_images` | multi-turn serving |
| `CMakeLists.txt` | `strata_rig` static library; the test targets; the probe | both trees need the library |
| `sycl/CMakeLists.txt` | link `src/rig/*.cpp` through `strata_resolve`; the SYCL probe | root sources resolve automatically |
| `serve/server.py`, `serve/test_server.py` | `"stage_workers"` → engine args; resolve the second engine package; `worker-lost` in the restart reasons; a rig-aware READY wait | the server must not see a rig as one confused process |
| `strata_control/engine_component.py` | keep and resolve a second engine component for the worker's backend | it activates exactly one backend per PC today (line 487) |
| `strata_control/gpu_roles.py` (+ `apps/host/src/lib/gpu-roles.ts`, `tests/data/gpu_roles_vectors.json`) | a card list spanning two makes; a role meaning "owns a worker process" | Main/Helper/Off are single-make today |
| `strata_cli/launch/*` (`params.py`, `screens.py`, `runner.py`, `settings.py`) | the Cards step's worker option, `--plan` output, `--status`/`--stop` covering the child | the Terminal is the agent-facing path |
| `setup.py`, `sycl/setup_intel.py` | admit the option in `SETUP_CHOICE_FLAGS`; never enable it by default | a default change needs a measurement, and there is none yet |
| `docs/MULTI_GPU.md`, `docs/INTEL.md`, `docs/TERMINAL_LAUNCH.md`, `AGENTS.md` | replace "of the same make" with a pointer to `MIXED_VENDOR.md`; record the gate numbers with their hardware | the docs currently refuse mixed makes |

## 4. Build, packaging, CLI and authentication detail

### 4.1 CMake

```cmake
# CMakeLists.txt, next to strata_plan / strata_engine
add_library(strata_rig STATIC src/rig/link.cpp src/rig/shm.cpp src/rig/handoff.cpp
                              src/rig/controller.cpp src/rig/remote_stage.cpp src/rig/worker.cpp)
target_link_libraries(strata_rig PUBLIC strata_includes)     # no CUDA target: vendor-neutral on purpose
target_link_libraries(strata PRIVATE strata_rig)
if(STRATA_BUILD_TESTS)
  foreach(t wire link shm handoff planner handshake)
    add_executable(rig_${t}_test tests/rig/${t}_test.cpp)
    target_link_libraries(rig_${t}_test PRIVATE strata_rig)
    add_test(NAME rig_${t} COMMAND rig_${t}_test)
  endforeach()
endif()
```

In `sycl/CMakeLists.txt` the same sources come in through `strata_resolve(_rig src/rig/link.cpp ...)` — no
migrated copy is needed, because the rig layer touches no device API — and `strata` links it. The two probes
build only where their toolchain exists.

Two consequences the implementer must hold: the rig library must **not** include `cuda_runtime.h` or any SYCL
header (`wire`/`link`/`shm`/`handoff`/`planner` are pure POSIX + `<cstring>`), and `RigStageHooks` is what keeps
the vendor code out of it.

### 4.2 Packaging

The worker binary is an engine package of another backend. `strata_control/feed.py` already publishes engine
packages per platform/backend/variant, and `strata_control/targets.py::backend_supported` already refuses an
unsupported backend × platform (`sycl` is `linux-x86_64` only) — which is exactly the guard the mixed rig needs:
a CUDA + SYCL rig is refused on Windows by the same rule that refuses SYCL there. `engine_component.py` must
learn to keep and resolve a second component for the worker's backend under the same chain of trust (signed
feed, `sha256`, `manifest_sha256`): a worker binary is as untrusted as any other downloaded file.

### 4.3 Authentication and trust

- **M2–M4 (one host, the only shipped configuration):** the worker is a **child** reached through an inherited
  `socketpair` fd. There is no listening socket and no filesystem path, so nothing else can attach; `SO_PEERCRED`
  still checks that the peer uid equals our own (cheap and defensive). The hand-off segments are `memfd`s passed
  by fd, never files in a shared directory, and are closed on teardown. No secret is passed, because there is no
  channel that could carry one safely.
- **Later (networked workers, M6, outside the gate):** the frames do not change; the transport does. Require the
  outer tunnel the product already supports (`--access tailscale`/cloudflare, with the tunnel's device identity)
  rather than adding a TLS stack inside the engine, **plus** a per-rig token HMAC'd over the `HELLO` body and
  re-checked at `CONFIGURE`, **plus** strict validation of every remote frame's size and shape — a remote worker
  is untrusted input in a way a forked child is not. The MVP puts no socket anywhere, so it cannot be reached
  remotely.
- A worker that fails authorisation gets `REFUSE{kRefuseCapability}` and no further detail.

## 5. Task graph (ordered by dependency)

Each task is sized for one focused session and names at most 5 files. `[Mx]` is the milestone; **CP** is the
checkpoint that closes the task.

### M0 — groundwork, no behaviour change

**T1 `[M0]` Probe: can a plain shm region become device-visible on each vendor?**
- Files: `probe/rig_shm_register.cu`, `sycl/probe/rig_shm_register.dp.cpp`, `CMakeLists.txt`, `sycl/CMakeLists.txt`
- Do: allocate a `memfd` region, map it, try `cudaHostRegister(Mapped|Portable)` + `cudaHostGetDevicePointer`
  on the CUDA side; on the port try what Level Zero offers for an externally mapped host range, and print the
  verdict with driver and oneAPI versions. No engine code.
- CP: the two verdicts are recorded in this plan (a `+`/`-` line each, with the version it was seen on). The
  MVP does **not** depend on either result — it decides only whether M5's zero-copy variant is possible. Mark it
  unconfirmed if the machine is not available.

**T2 `[M0]` Wire format and its tests.**
- Files: `include/strata/rig/wire.hpp`, `tests/rig/wire_test.cpp`, `CMakeLists.txt`
- Do: header/codec, `kProto`, capability bits, `GeometryIdentity`, refusal codes; tests for a bad magic, a
  version mismatch, a truncated frame, an oversize frame, an epoch mismatch, a `seq` gap.
- CP: `ctest -R rig_wire` green with no GPU. Byte layout `static_assert`ed at 24 B.

**T3 `[M0]` Control link.**
- Files: `include/strata/rig/link.hpp`, `src/rig/link.cpp`, `tests/rig/link_test.cpp`, `CMakeLists.txt`
- Do: `socketpair(AF_UNIX, SOCK_SEQPACKET)`, send/recv with the header, `SCM_RIGHTS` fd passing, `SO_PEERCRED`,
  `recv` returning "peer closed", deadline polling.
- CP: unit tests plus a fork arm where the child exits abruptly and the parent reports peer death, not a hang.

**T4 `[M0]` Hand-off ring.**
- Files: `include/strata/rig/shm.hpp`, `src/rig/shm.cpp`, `tests/rig/shm_test.cpp`, `CMakeLists.txt`
- Do: `memfd_create`, mapping in both processes, per-row slots with seq stamps, bounded `acquire`, the payload
  hash path, epoch stamping.
- CP: tests for a wrap, a stale seq, a hash mismatch and a timeout; a fork arm that writes 100k rows and verifies
  every byte round-trips.

### M1 — a worker exists, does nothing

**T5 `[M1]` Spawn, handshake, capacity, teardown.**
- Files: `include/strata/rig/rig.hpp`, `src/rig/controller.cpp`, `include/strata/rig/worker.hpp`,
  `src/rig/worker.cpp`, `tests/rig/handshake_test.cpp`
- Do: `fork`/`execv` of the other binary with `--rig-worker <fd>` and the artifact arguments; the per-card
  device-selector environment; `PR_SET_PDEATHSIG`; `HELLO`/`HELLO_ACK`/`CAPACITY`/`CONFIGURE`/`READY`/`TEARDOWN`;
  the geometry and proto refusals; a mock worker for the test.
- CP: `strata … --worker <other-vendor-binary> --layer-split 24 --rig-probe` prints both cards' free MiB, the
  vendor/build id of each and exits 0 **before** loading weights; killing the controller leaves no worker alive
  within `--worker-exit-timeout-s`; killing the worker makes the controller exit non-zero.

**T6 `[M1]` Option parsing and the refusal gates (no compute).**
- Files: `src/program/generate.cpp`, `sycl/src/program/generate.cpp`,
  `tests/program/worker_args_test.cpp`, `CMakeLists.txt`
- Do: `--worker`, `--worker-*`, `--rig-worker`; refuse on Windows, on the last stage being remote, beside
  `--batch*`, `--peer-device`, `--vram-elastic` on a SYCL worker, session files, and overlap; print the
  gate list in `--help`.
- CP: each refusal is one line naming the flag to drop; **without `--worker` the binary's behaviour is
  unchanged** (the existing split tests still pass, and `--window-hashes` is unchanged on a single-GPU run).

### M2 — the same split, in two processes, on one vendor (the equivalence gate)

**T7 `[M2]` Hand-off across the boundary.**
- Files: `include/strata/rig/handoff.hpp`, `src/rig/handoff.cpp`, `tests/rig/handoff_test.cpp`, `CMakeLists.txt`
- Do: bracket the vendor-native pinned buffers with ring copies, using the row count and
  `Verifier::handoff_floats` layout exactly as `generate.cpp:9045` sizes them today; counters for bytes and ms.
- CP: a test with two processes and fake pinned buffers reproduces the real row layout byte for byte, including
  the `--batch` rows case (`set_handoff_rows`).

**T8 `[M2]` Factored chain surface (default-path change, so it needs a parity test).**
- Files: `include/strata/rig/next_stage.hpp`, `include/strata/core/verify.hpp`, `src/core/verify.cpp`,
  `src/program/generate.cpp`, `sycl/src/program/generate.cpp`
- Do: `set_next(const NextStage&, void*)`; `Verifier`'s own `kSelfStage`; the `chain_overlap` path keeps
  requiring two `Verifier*`s (it stays off for a worker rig).
- CP: an in-process two-GPU split's `--window-hashes` file is **identical** before and after this task, and
  `docs/MULTI_GPU.md`'s measured arm reproduces on the pair (or on whatever two same-make cards are here).

**T9 `[M2]` The proxy stage, decode path.**
- Files: `include/strata/rig/remote_stage.hpp`, `src/rig/remote_stage.cpp`, `src/rig/controller.cpp`,
  `src/program/generate.cpp`, `src/rig/worker.cpp`
- Do: `RunWindow`/`WindowDone`/`Commit`/`WaitCommit`/`Logits`; the bounded waits; the latch; the split-plan print
  naming process and vendor per stage; rig counters in `--stats`.
- CP (the milestone gate): a CUDA+CUDA two-process split generates **the same tokens and the same
  `--window-hashes`** as the in-process split of the same model at the same cut, on the same prompt; the decode
  cost of the boundary is reported (ms/token attributable to copy+message) and is within the budget planned here
  (target ≤0.2 ms/token at 20 tok/s, i.e. ≤4 %).

**T10 `[M2]` Prompt path across the boundary.**
- Files: `src/rig/remote_stage.cpp`, `src/rig/worker.cpp`, `include/strata/rig/planner.hpp`,
  `src/program/generate.cpp`, `tests/program/prefill_plan_test.cpp`
- Do: drive each side's `Prefill` for its own range; clamp the chunk by the **worker's** lend region reported in
  `CAPACITY` (the in-process rule is `append_stage_lend_regions`, #448); keep `--prefill-pipe R` meaningful.
- CP: a long prompt (≥100k tokens) reads to identical tokens on the two-process rig as on the in-process rig;
  read tok/s recorded both ways.

### M3 — mixed vendor: CUDA controller + SYCL worker (the intent gate)

**T11 `[M3]` The worker brings up its own stage on the other vendor.**
- Files: `sycl/src/program/generate.cpp`, `include/strata/rig/worker.hpp`, `src/rig/worker.cpp`,
  `sycl/include/strata/core/verify.hpp` (only if the port's copy needs the `set_next` change), `sycl/CMakeLists.txt`
- Do: fill `RigStageHooks` in the port: preflight, link probe, stage build (its own dense trim, session/KV carve,
  expert cache from the profile, per-stage mirror within the given byte budget, `STRATA_VERIFY_NO_HOST` aware),
  hand-off pointers, run/commit.
- CP: `--rig-probe` and then a real start on the CUDA+Arc host prints the two-stage plan
  (`layers 0-K (sycl/1, process …) , K-47 (cuda/0)`) and reaches `READY` with both stages' resident pairs and
  mirror GiB.

**T12 `[M3]` Capacity, RAM and CPU-core partition.**
- Files: `include/strata/rig/planner.hpp`, `tests/rig/planner_test.cpp`,
  `include/strata/program/split_rules.hpp`, `src/rig/controller.cpp`, `src/program/generate.cpp`
- Do: the vendor cost term and `vendor_max_layers`; the RAM split (controller reserves first, worker gets the
  remainder); `--worker-pool-cores` defaulting to a disjoint core set; explicit `--layer-split` bypassing the
  search.
- CP: planner tests green (including "the slower vendor never gets more layers than its share", "a stage that
  cannot hold 3 layers refuses with numbers", "the two processes' pinned budgets never both claim the same
  MemAvailable"); on the machine, total pinned RAM recorded against `MemAvailable` before and after start.

**T13 `[M3]` The gate run and its record.**
- Files: `docs/MIXED_VENDOR.md`, `docs/MULTI_GPU.md`, `docs/INTEL.md`, `AGENTS.md`, plus the run config
  (e.g. `strata-*-mixed.json`)
- Do: the measured run — Q4_K_M (or better), ≥131,072 context, correct/stable; prefill and decode recorded with
  the flags and hardware; the same-day single-vendor arm for comparison; the failure arms recorded apart
  (passed / failed / not run) as `CONTRIBUTING.md` requires.
- CP (**the intent's success gate**): ≥20 decode tok/s sustained, no wrong-token or stall event across the
  recorded runs, and the write-up states plainly that the mixed path is **not** byte-identical to a single-vendor
  run and what was compared instead (identical tokens across repeats of the mixed path, and identical tokens
  against the same-vendor cross-process rig at the same cut where the layers are the same vendor's).

### M4 — it serves conversations

**T14 `[M4]` Park and restage across the boundary.**
- Files: `src/rig/remote_stage.cpp`, `src/rig/worker.cpp`, `src/core/conversation_file.cpp`,
  `include/strata/core/conversation_cache.hpp`, `tests/core/conversation_split_failure_test.cpp`
- Do: `PARK`/`RESTAGE` carrying the worker's `stage_images` bytes, reusing `conversation_checkpoints_split`/
  `merge`; keep the session-file refusal with its updated text.
- CP: a multi-turn conversation on a worker rig produces the same tokens and the same checkpoint reuse as the
  in-process rig; `SAVE` still refuses with the right reason; the #752 two-phase failure tests still pass.

**T15 `[M4]` Server, Terminal and Control integration.**
- Files: `serve/server.py`, `serve/test_server.py`, `strata_control/engine_component.py`,
  `strata_control/gpu_roles.py`, `tests/terminal/test_launch_workers.py`
- Do: `"stage_workers"` in the run config → engine args; resolve the second engine component; add `worker-lost`
  to the restart reasons; `--plan`/`--status`/`--stop` cover the child; the app's role model gains the worker
  role (a follow-up task in `apps/host` if the app UI is in scope).
- CP: `strata launch --plan` shows the rig and refuses an unsupported make × platform; `pkill -f build-sycl/strata`
  mid-request restarts the rig through the server's existing ladder; `--stop` leaves no process holding VRAM
  (checked with `level-zero`/`nvidia-smi` memory back to idle).

### M5 — the room this design leaves (post-gate, each needs its own measurement)

**T16 Overlap across processes.** Probe first (why: `chain_overlap` needs GPU-raised host-visible flags and the
xe port runs `STRATA_VERIFY_NO_HOST=1`); then one ring per parity, as `--pipeline-windows` uses `hand_b[]`
today. Files: `src/core/verify.cpp`, `src/rig/remote_stage.cpp`, `src/rig/handoff.cpp`, `sycl/src/program/generate.cpp`.
CP: decode measured better than the synchronous boundary by a recorded margin, on both vendors, or it does not land.

**T17 Design B on the same link: an expert/mirror worker.** A `RigExpertSource : public ExpertSource` whose
`blob`/`copy_blob`/`device_alias` answers come from a worker's VRAM cache or mirror — the shape of
`RemotePrefillSource` with the helper replaced by a process. Files: `include/strata/rig/expert_wire.hpp`,
`src/rig/expert_source_link.cpp`, `src/rig/worker.cpp`, `src/program/generate.cpp`,
`tests/core/expert_source_link_test.cpp`. CP: on a rig where dense + KV already fit one card, the expert tier
raises the cache hit rate and prefill/decode speed, measured against the same-day file tier; the port's
`remote_expert_opt_stub.cpp`/`peer_experts` gap is recorded, not hidden.

**T18 Zero-copy hand-off, if T1 said yes.** Register the memfd with each vendor and drop the two memcpys.
Files: `src/rig/shm.cpp`, `src/rig/handoff.cpp`, `probe/…`. CP: measured ms/token saved is worth the added
driver coupling; otherwise it stays off.

### M6 — beyond one host (outside the gate)

**T19 Networked worker transport + authorisation** (token HMAC over `HELLO`, tunnel required, frame validation),
**T20 AMD/HIP as a third vendor** (its own tree, `docs/AMD_HIP.md`, and the same capability bits). Neither is
started before M3 and M4 are green.

## 6. Sequencing, in one picture

```
T1 (probe) ─┐
T2 wire ─────┼─ T5 spawn+handshake ─ T6 flags+refusals ─┐
T3 link ─────┤                                          ├─ T9 proxy (decode) ─ T10 prompt ─ T11 sycl stage ─ T12 budgets ─ T13 GATE
T4 ring ─────┴─ T7 hand-off copies ─ T8 NextStage ──────┘                                                                    │
                                                                                                                             T14 park ─ T15 UX
post-gate: T16 overlap · T17 design B · T18 zero-copy · T19 network · T20 AMD
```

**Minimum viable first milestone: T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9** — a same-vendor, two-process,
whole-layer split that is byte-identical to the in-process split on the same cut (T9's checkpoint). It is the
smallest slice that proves every hard part — process topology, transport, frame protocol, hand-off fidelity,
lifecycle and failure semantics — while it can still be compared against ground truth, which a mixed-vendor run
never can. The intent's gate (T11–T13) is then a vendor swap plus the capacity plan, not a new architecture.

## 7. Risk register

| # | Risk | Why it is real here | Mitigation / what to do first |
|---|---|---|---|
| 1 | **The slow vendor becomes the serial bottleneck** and the rig decodes below 20 tok/s even though it fits | The stages run in sequence; an Arc stage's per-layer time is several times a GeForce's (the port's own pair numbers are 20-27 tok/s for the whole model) | `vendor_max_layers` in the planner (T12); explicit cuts first, search later; measure per-stage ms with `--stage-timing`/`--stats`; the gate records prefill and decode per stage, not only end to end |
| 2 | **Two CPU expert pools oversubscribe the cores** and decode collapses | `--pool-workers` defaults to every physical core but one; two processes would each take them all | the controller partitions cores and passes `pool_workers` + affinity in `CONFIGURE` (T12); the evidence line prints both counts |
| 3 | **Host RAM / pinned-budget double counting** → the #1250 OOM-kill or a failed pin mid-run | `host_available_memory` and `clamp_resident_budget` are per-process views of one `MemAvailable`; both processes can each "fit" | the controller computes the split and hands the worker an explicit byte budget; the worker reports what it pinned; a run records pinned GiB of both processes against `MemAvailable` |
| 4 | **The hand-off is where a silent wrong token would be born** (stale, torn or reused row) | a hand-off row is the residual stream; a stale one is plausible, not an error | seq stamps + epoch + optional payload hash (T4); `acquire` refuses rather than accepting; T7's byte-exact test; latch-and-abort, never resume |
| 5 | **A per-vendor `ExpertDispatch`/`SplitDrive` mistake**: the worker's layers served by the controller's cache or pool, or vice versa | `SplitDrive` selects a stage's plan/cache/PCIe share by layer index; two processes each hold their own | the worker serves only `[lb,le)`; an out-of-range layer request is a `REFUSE`/`ERROR`, never a fallback; a test asserts each side's layer range in a window |
| 6 | **The port cannot do what the CUDA tree does** (VMM, `--peer-device`, remote expert opt are stubs in `sycl/`) | `sycl/CMakeLists.txt:259-263`, `vmm.cpp` stub | capability bits turn each into a start-up refusal naming the flag; the docs list what a mixed rig cannot do |
| 7 | **xe driver quirks around host-visible flags / doorbells** break any overlap idea | `STRATA_VERIFY_NO_HOST=1`, `STRATA_VERIFY_DEVICE_PLAN=1` in `sycl/serve/strata-sycl-host.sh` | the MVP is synchronous per boundary (the measured 26.6 tok/s shape); overlap is T16 behind a probe |
| 8 | **Weight load in two processes doubles startup time or disk read** | 60 GB arena + dense per stage; `ENGINE_READY_S` is 900 s | mmap the same files (page cache shared, read once); `--trim-stage-weights` for the worker's own layers; relay `PROGRESS` into `progress_at` so the stall watchdog sees life; document `engine_ready_s` |
| 9 | **`--worker` changes the default path by accident** (the refactor in T8 touches every split) | `set_next`/`Verifier` is on the hot path of the existing split | T8 is guarded by an in-process `--window-hashes` diff before/after; the worker code never runs without the flag; `CONTRIBUTING.md`'s byte-identical default rule is a test, not a promise |
| 10 | **Orphaned worker holding VRAM** after a crash or Ctrl-C | 20 GiB pinned on an Arc card blocks the next start and any desktop use | `PR_SET_PDEATHSIG` + HUP + an idle timeout; the server's `ENGINE_*_WAIT_S` ladder; T15's check that `--stop` returns both cards to idle |
| 11 | **Version or pack skew between the two binaries** (two engine packages, updated independently) | the worker is a separate package with its own update path | proto + `build_id` + `GeometryIdentity` in the handshake, refused with both numbers; the component resolver keeps both at one feed sequence |
| 12 | **Prompt cache semantics differ across processes** (a checkpoint restored on one stage but not the other) | the in-process split keeps both stages' parts together; a rig does not | M4 refuses parking until `PARK`/`RESTAGE` lands (one honest gap, stated at start); the pair parks/restores atomically under the existing #752 two-phase code |
| 13 | **The gate is met but the feature is useless interactively** (cache off, session save off) | the gate is a benchmark, serving is the product | M4 before the feature is recommended; the docs say plainly what a worker rig cannot do until then |
| 14 | **Mixed-vendor numerics are blamed for a real bug** | vendor kernels differ; people will read any token difference as corruption | the comparison ladder is fixed in the plan: default vs default, in-process split vs cross-process split (must be identical), mixed vs same-vendor-cross-process (expected to differ); `docs/MIXED_VENDOR.md` states it before the numbers |

## 8. Standards and conventions for this workstream

Written to `.opencode/context/standards.md` (new) and `.opencode/context/architecture.md` (new) so the
implementer and the test agent share them.

- Follow the house style of the files being touched: dense comments that state **why** a guard exists and what
  going wrong would look like ("a wrong token", "REFUSED rather than mis-indexed"), measured numbers with the
  machine they were measured on.
- Refuse loudly at start-up: one line with both numbers and the flag that would change it. Never degrade,
  never retry silently, never resume after a latched failure.
- Vendor-neutral code goes in the root tree so `strata_resolve` picks it up in the port; a port-only file is a
  last resort.
- No new dependency. POSIX sockets/shm only; no TLS, no RPC framework, no llama.cpp interop.
- New engine flags are `--long-form` with a `--help` line; run-config keys are snake_case; env vars are
  diagnostics only (`STRATA_RIG_*`).
- Tests: pure-CPU for anything that is a rule (wire codec, planner, ring) in `tests/rig/`, mirroring
  `tests/program/split_rules_test.cpp`; parity by `--window-hashes` + `--spec-follow`; the Terminal tests under
  `tests/terminal` (`python -m pytest tests/terminal`), the server's under `serve/test_server.py`. Sample test
  shapes are in `.opencode/context/testing.md`.

## 9. Gate and validation strategy

1. **Unit, no GPU**: `rig_wire_test`, `rig_link_test`, `rig_shm_test`, `rig_handoff_test`, `rig_planner_test`,
   `rig_handshake_test`.
2. **Same-vendor cross-process parity (T9/T10)** — the strongest test in the plan: same model, same cut, same
   prompt; require identical `--window-hashes` lines and identical argmaxes against the in-process split, and
   identical output across repeats. This is where the boundary is proved.
3. **Mixed vendor (T13)** — the gate: Q4_K_M or better, ≥131,072 context, `--max-context 135168` style config,
   ≥20 decode tok/s, correct answers on the existing golden chat fixtures (`serve/chat_golden.json`) and a tool
   call round trip, stable across the recorded runs (no stall, no watchdog stop). Record: prefill tok/s (cold and
   with whatever cache state the build has — stated, because M3 runs with `--prompt-cache 0`), decode tok/s,
   drafts accepted, per-stage ms, mirror GiB and pinned GiB of **both** processes, and the boundary cost
   (`STRATA_RIG_STATS=1`). Compare against a same-day run of the same build on the single-vendor pair.
4. **Failure arms, recorded apart from the passes**: kill the worker mid-window; kill the controller; run a
   capacity-too-small config; run a version-skewed worker; run an unsupported combination (Windows, `--batch`).
5. **Not run / not claimed**: AMD, multi-host, llama.cpp interop, overlap speed, and any byte-identity claim
   between mixed and single-vendor runs. Anything derived only from reading code is labelled unconfirmed static
   analysis, per `CONTRIBUTING.md`.

## 10. Handoff to Implementer

**Design document**: `.opencode/discovery/plans/mixed-vendor-pooled-rig-plan.md` (this file).
**User story**: the confirmed intent in §"User story reference" (no story file exists).
**Estimated complexity**: **High** (two trees, one new subsystem, a default-path refactor in T8, and a measured
gate on hardware).
**Key files**: `src/program/generate.cpp` (the split block at 9040-9131 and the stage build at 9094+),
`include/strata/core/verify.hpp` + `src/core/verify.cpp` (`set_stage`/`set_next`/`handoff_floats`),
`include/strata/program/split_rules.hpp`, `sycl/src/program/generate.cpp`, and the new `include/strata/rig/`
headers that everything else hangs off.
**Start with**: T2 → T3 → T4 (all no-GPU, all pure), then T5. Do **not** start with T9: the parity harness in
T8 and the copies in T7 are what make T9 debuggable.
**Acceptance criteria**: the intent's gate (§9 item 3), plus the T9 same-vendor byte-identity checkpoint and the
rule that the default path runs none of this code.
**First refusal to expect on this machine**: the pair's live config is SYCL-only (no CUDA card in the SYCL
config), so the CUDA controller half needs the CUDA engine package and a card in the box; if the CUDA card is
not present, T9 must be run with two same-make cards (the `--expert-cache-device1..3` class of rig) and T11-T13
wait for hardware — record that as "not run", never as a pass.
