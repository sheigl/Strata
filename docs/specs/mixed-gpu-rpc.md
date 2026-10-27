# Spec: mixed-vendor GPU pooling through RPC workers

Status: **draft technical plan with a same-backend decode implementation, 2026-10-26**. The protocol, local
transport, worker supervision, trimmed CUDA worker, controller-side decode proxy, and the prefill-stage boundary exist
(Tasks 1-8); fixed-window
same-backend parity and its one-card cache-placement limit are recorded in
[tasks/evidence/task7.md](../../tasks/evidence/task7.md). Mixed-vendor execution and long-context measurements are
still unmeasured. Read the confirmed [intent](../intent/mixed-gpu-rpc.md) first.

House rules: the feature is opt-in; the path with no `rpc_split` configuration must retain its current behavior;
build every backend a change touches; never claim correctness or speed without running the stated check.

## 1. Objective and gates

Run one Qwen3.8-Flash-Next inference workload across an NVIDIA CUDA device and an Intel Arc SYCL device on one
Linux host. Pool the devices' weight, expert-cache, and layer-state capacity. Use whole layers as the first
partitioning unit.

The first complete milestone passes all of these gates on the maintainer's machine, with the exact NVIDIA and Arc
models, VRAM, driver versions, model artifact, and run configuration recorded:

1. **Workload gate:** literal Q4_K_M or better, configured context at least 131,072 tokens, text generation.
2. **Correctness gate:** a real prompt of roughly 104K tokens or more is read and answered coherently; no NaN,
   watchdog, host livelock, orphaned worker, or silent fallback to one GPU. The fixed prompt suite in §11 passes.
3. **Performance gate:** sustained decode is at least **20 generated tokens/s** after that long prompt. Report
   prompt tok/s, decode tok/s, draft acceptance, per-stage time, RPC wait/copy time, and variability across runs.
4. **Isolation gate:** with `rpc_split` absent, existing single-GPU and same-vendor `--layer-split` behavior and
   tests are unchanged. Existing mixed-vendor launch refusal remains unless the new option is explicit.
5. **Lifecycle gate:** startup mismatch, disconnect, cancel, timeout, and worker crash fail the request/model load
   cleanly. A worker is never restarted into an existing conversation.

A same-backend RPC prototype, a short-context demo, or an expert-only memory server is useful progress but does not
satisfy the milestone.

## 2. Initial limits and non-goals

The first implementation deliberately has these limits:

- Linux only; one local host; exactly one prefix worker plus the controller's suffix stage.
- Qwen4Exp/Qwen3.8-Flash-Next only. Other families follow the same interfaces later.
- Explicit layer boundary. No automatic placement in the engine's first vertical slice.
- Single sequence, text only, greedy and existing controller-side sampling. No batch slots or images in RPC mode.
- The controller owns the final layer range, native head, sampler, and MTP drafter. The worker owns an initial
  contiguous layer range and the token embedding needed by that range.
- Conversation save/restore remains refused in RPC-split mode, as it is for the existing layer split. Parking or
  resuming a live RPC conversation is not part of the first milestone.
- No llama.cpp protocol compatibility, tensor parallelism, TCP listener, Internet exposure, AMD validation, or
  automatic desktop/setup recommendation in the first milestone.

These are scope limits, not permissions to weaken the final workload gate. MTP must remain available because the
20 tok/s target may depend on the existing speculative verify path.

## 3. Source-backed constraints

These findings come from the current tree and should be rechecked before implementation:

- CUDA/HIP and SYCL are separate binaries and source trees. `CMakeLists.txt:14-21` rejects a combined CUDA/SYCL
  build. A mixed-vendor stage therefore needs a separate process, not another `cudaSetDevice` ordinal.
- The existing `GpuStage` construction in `src/program/generate.cpp` and `sycl/src/program/generate.cpp` already
  gives each layer-split stage its own dense weights, session carve, expert cache, verifier, prefill, and streams.
- `Verifier::set_stage` and `Verifier::handoff_floats` in `include/strata/core/verify.hpp:166-194` define the decode
  boundary: per token, residual `R` (`hc * n_embd` floats), pending `bo` (`n_embd`), and `inject` (`hc`). Preserve
  this layout; do not invent a tensor-graph RPC format.
- `Prefill::set_stage` in `include/strata/prefill/prefill.hpp:223-229` is the analogous prompt-chunk boundary.
- `SessionState` already carves KV and recurrent state by `[layer_lo, layer_hi)`. Each process must allocate only
  its own range's state, at the full configured context.
- CUDA and SYCL kernels are not bit-identical. Same-backend RPC must be bit-exact against the equivalent in-process
  split; mixed-backend RPC is a new numerical configuration and must not be advertised as bit-identical to either
  single-vendor path.
- GPU-captured graphs and ready flags assume backend-visible host memory. In particular, the xe path cannot use the
  CUDA split's host-flag scheme. The first transport therefore uses stream synchronization plus CPU copies, not
  cross-process GPU doorbells.
- The Python server owns engine startup and supervision. `strata_control/engine_component.py` and the SYCL wrapper
  already know how to select backend binaries and environments. Keep process lifecycle there rather than teaching
  C++ how to locate oneAPI or installed components.

## 4. Architecture decision

### 4.1 Topology

```text
serve/server.py
  ├── RpcWorkerSupervisor
  │     └── SYCL engine process: layers [0, K), embedding, prefix KV/experts/prefill/verifier
  │            ▲
  │            │ versioned control frames over Unix domain socket
  │            │ activation bytes through a memfd shared buffer; fd passed locally
  │            ▼
  └── CUDA engine process (controller): layers [K, 48), suffix KV/experts/prefill/verifier,
         native head, sampling, MTP, tokenizer-facing engine protocol, conversation ownership
```

The controller may be SYCL and the worker CUDA after both worker implementations exist; the first validation uses
whichever assignment the measured placement favors. The architecture does not require NVIDIA to be the controller.

**Controller-last is deliberate.** Tokens and positions cross the wire, but logits, sampling histories, logit bias,
speculative acceptance, and final tokens stay in the controller. This avoids a remote sampling API in the first
milestone and preserves the existing server/engine line protocol.

### 4.2 Why whole layers first

- It pools dense weights, expert capacity, and per-layer KV/GDN state, not merely expert bytes.
- The existing split already defines the exact activation and state boundaries.
- One boundary crossing per verify window or prefill chunk is much less latency-sensitive than RPC per tensor or
  per expert miss.
- It uses both GPUs for compute and has a plausible path to the required 20 tok/s.

### 4.3 Alternatives not chosen

- **One process loading CUDA and SYCL:** incompatible with the current backend build model and captured runtime
  assumptions.
- **Python orchestration per token/window:** would put server IPC and Python scheduling on the decode critical path
  and duplicate speculative state machinery. Python supervises processes; C++ executes windows.
- **Fine-grained tensor or per-expert compute RPC:** too many synchronous crossings and no cross-vendor bit parity.
- **Expert/mirror-only worker as the primary design:** useful as a later `ExpertSource` decorator, but it leaves
  dense weights and layer state on the controller and does not prove whole-model mixed-GPU execution.
- **llama.cpp wire protocol:** explicitly out of scope. Borrow the worker idea, not its protocol.

## 5. Components

### 5.1 Backend-neutral RPC library

Add a small C++ library with no CUDA, HIP, SYCL, or ggml includes, shared by both engine builds:

- `include/strata/rpc/protocol.hpp` and `src/rpc/protocol.cpp`: frame encode/decode, message types, limits,
  version/capability flags, and malformed-input errors.
- `include/strata/rpc/transport.hpp`: abstract ordered byte transport and bulk-buffer interface.
- `src/rpc/local_transport_posix.cpp`: Unix domain sockets, `SCM_RIGHTS` fd passing, `memfd_create`, `mmap`, seals,
  peer-credential check, and deadline-aware I/O.
- `src/rpc/local_transport_stub.cpp`: clean `unsupported on this platform` error on non-Linux builds.
- `cmake/strata_rpc.cmake`: one source list/function included by the top-level and SYCL builds without moving the
  existing backend early-return logic.

Use fixed-width little-endian fields, explicit enum values, and bounds checks. Do not add a JSON or RPC dependency
to the engine. Suggested frame header:

```text
magic:u32 | protocol_version:u16 | opcode:u16 | request_id:u64
stage_id:u32 | flags:u32 | payload_bytes:u64 | bulk_bytes:u64
```

Control payloads are capped (initially 1 MiB); tokens and small commit metadata may be inline. Activations always
use the negotiated bulk buffer. Every frame belongs to one request/window sequence and one stage.

### 5.2 Backend-local stage runtime

Extract the existing stage construction and execution from each backend's large `generate.cpp` into backend-local
runtime units. CUDA/HIP and SYCL keep their own kernels and stage types; only the RPC contract is shared.

The runtime must expose enough for both normal in-process split and worker mode:

- load/trim dense weights and embedding for `[layer_begin, layer_end)`;
- allocate the matching `SessionState` carve, expert source/cache/mirror, prefill, and verifier;
- return geometry, artifact identity, memory use, and readiness;
- run a decode window, commit its accepted prefix, run a prefill chunk, cancel at a safe boundary, and unload.

This extraction is a separate parity task. No RPC behavior should be hidden inside it.

### 5.3 Controller proxies

Introduce narrow `DecodeStage` and `PrefillStage` interfaces only at the current recursive stage boundaries. The
existing `Verifier`/`Prefill` chain remains the local implementation. A controller-side `RpcPrefixStage` implements
the same boundary by calling the worker and then the local suffix.

Do not make all engine operations virtual and do not replace the engine with a general graph executor. Batch and
other unsupported calls return an explicit unsupported error in RPC mode rather than silently taking a different
path.

### 5.4 Python configuration and supervision

Add a validated `rpc_split` run-config section and an `RpcWorkerSupervisor` beside `StrataEngine`:

- resolve the requested worker backend through the installed engine-component mechanism;
- preserve the SYCL wrapper/environment behavior, including an explicit `ONEAPI_DEVICE_SELECTOR` for the worker;
- create a private runtime directory (mode 0700), socket path, random 256-bit session token, and memfd negotiation;
- start the worker, wait for its socket and handshake, then start the controller;
- supervise both processes, reap them, remove sockets/manifests, and connect worker failures to the engine's error
  and watchdog paths;
- pass the token through the child environment or an inherited fd, never in argv or logs.

The existing `StrataEngine` stdin/stdout protocol remains unchanged.

## 6. Protocol and state machine

### 6.1 Messages

Initial opcodes:

- `HELLO`, `HELLO_ACK`, `CAPABILITIES`
- `INIT_STAGE`, `INIT_PROGRESS`, `READY`
- `WINDOW`, `WINDOW_DONE`
- `COMMIT`, `COMMIT_DONE`
- `PREFILL_CHUNK`, `PREFILL_CHUNK_DONE`
- `CANCEL`, `CANCEL_DONE`
- `STATS`, `STATS_RESPONSE`, `PING`, `PONG`, `SHUTDOWN`, `ERROR`

`INIT_STAGE` carries the layer range, context and KV settings, max verify window, prefill chunk limits, quant/cache
settings, and the bulk-buffer layout. Model paths and backend process arguments come from the supervisor's launch
configuration, not from an untrusted peer.

The handshake compares at least: protocol version, engine version, backend and device, model family, compiled model
geometry, layer count/range, `n_embd`, `hc`, tensor-type/layout digest, artifact manifest identity when available,
maximum context, and handoff float count. Any mismatch is a startup error. Do not read a 100+ GB artifact merely to
hash it at every launch; use the existing artifact manifest identity, with a metadata/shard-layout digest fallback.

### 6.2 State and ordering

```text
STARTING -> LOADING -> READY -> WINDOW_RUNNING -> READY
                         |          |
                         |          -> COMMITTING -> READY
                         -> PREFILL_RUNNING -> READY
any state -> FAILED -> CLOSED
```

The first version allows one in-flight operation per worker and uses single-buffered synchronous transfers. Sequence
IDs must increase monotonically; stale, duplicate, out-of-state, oversized, or truncated messages fail the session.
No GPU spin flag in shared memory is part of this protocol.

### 6.3 Decode window

1. Controller sends `WINDOW(request_id, T, tokens[T], pos0, mode_flags)`.
2. Worker embeds tokens, runs layers `[0,K)`, synchronizes its output stream, and writes the existing handoff layout
   into its local pinned staging buffer.
3. Worker copies staged floats to the agreed memfd region and sends `WINDOW_DONE` with sequence, float count, and
   actual byte count. The controller never reads the region before this message.
4. Controller copies the bytes into its own pinned `handoff_in`, runs the local suffix and head, and proceeds with
   normal speculative acceptance/sampling.
5. For the accepted prefix, controller issues the worker `COMMIT(request_id, n_keep)` and performs the local suffix
   commit in the same relative order as the existing stage chain. Both must succeed; otherwise the conversation is
   invalid and the engine fails closed.

The worker uses its own CPU expert pool and mirror. No function pointer, device pointer, CUDA handle, USM pointer,
or C++ object crosses the protocol boundary.

### 6.4 Prefill

Use the existing prompt chunk size and handoff-row layout. For each chunk, the worker embeds and processes its prefix
range, returns residual rows through the bulk region, and the controller processes its suffix. Preserve position,
chunk, checkpoint-callback, stop, and MTP-draft semantics.

Start synchronously. Add chunk pipelining only behind a separate measured option; do not reuse
`STRATA_SPLIT_OVERLAP` or its GPU-visible flags across processes.

### 6.5 Errors, cancellation, and timeout

- Protocol violation, artifact mismatch, worker exit, timeout, or local/remote commit failure produces one clear
  engine error naming the stage and operation. There is no silent CPU or one-GPU fallback.
- `STOP`/cancel is observed at window and prefill-chunk boundaries. If the worker cannot drain safely, supervision
  terminates the whole RPC model instance rather than reusing uncertain state.
- A worker can be restarted only with a fresh model load and fresh conversation. Existing layer-split checkpoint
  refusal is extended to RPC split.
- Logs redact tokens, model content, and authentication material. Diagnostics retain opcodes, sequence IDs, sizes,
  timings, backend/stage IDs, and OS/GPU error codes.

## 7. Memory, weights, KV, and placement

### 7.1 Explicit placement

The first config supplies one boundary `K`, producing worker `[0,K)` and controller `[K,n_layers)`. Reject empty,
overlapping, out-of-order, or multiple non-contiguous ranges. Use the existing minimum-stage-layer and geometry
rules where applicable.

Before loading, print a per-process budget containing dense weights, embedding/head/MTP ownership, expert cache,
mirror/pinned RAM, KV/GDN state at the requested context, runtime reserve, and handoff buffers. Reuse
`session_bytes`, `stage_room`, `WeightTable::pool_bytes`, and split placement arithmetic instead of adding a second
memory model.

### 7.2 Weight and expert ownership

- Make `--trim-stage-weights`/`STRATA_STAGE_TRIM` behavior explicit and mandatory for RPC stages; do not load full
  dense weights into both processes.
- The prefix worker owns embedding only when its range starts at layer 0. The controller owns head/MTP because its
  range ends at the final layer.
- Each stage keeps only its layer range's expert residency table, cache slots, mirror table, and prefill/verify
  plan. Do not share device pointers or global mirror tables between processes.
- Audit the CUDA full pinned expert arena before allowing a CUDA prefix worker. If it still duplicates all layers,
  add a range-limited `FileExpertSource`/arena mode first; otherwise mixed mode can fail from host RAM even when VRAM
  fits.
- Preserve per-stage `--kv-resident`, VRAM reserve, mirror RAM cap, and link-cost settings in the worker manifest.
  KV is never transmitted during normal inference.

### 7.3 Automatic placement later

Only after the explicit path meets the gates may an auto planner enumerate boundaries. It should price per-stage
free VRAM, host RAM, KV at context, head/MTP ownership, measured device speed, and measured RPC cost. It must not
reuse same-vendor SM-weighted `auto` unchanged.

## 8. Configuration surface

Public run config (names are part of this plan and should be used consistently):

```json
"rpc_split": {
  "boundary": 24,
  "workers": [
    {
      "id": "arc-prefix",
      "backend": "sycl",
      "device": 0,
      "engine": "/optional/development/path/strata-sycl-host.sh",
      "env": {"ONEAPI_DEVICE_SELECTOR": "level_zero:0"}
    }
  ]
}
```

The main configured engine is the controller and owns the suffix. Production installs should omit `engine` and use
the installed component. Arbitrary environment keys are not accepted; validate a small allowlist needed by backend
selection.

Internal engine arguments, generated by `serve/server.py`:

- worker: `--rpc-listen <socket> --rpc-layer-range 0:K`;
- controller: `--rpc-connect <socket> --rpc-layer-range K:<layers>`;
- both receive the same model/cache/context arguments and the per-run token through a protected channel.

`--layer-split` and `rpc_split` are mutually exclusive in the first version. `--peer-device`, batch, images, and
session restore are rejected explicitly in RPC mode until implemented. Existing `gpu_roles` and mixed-vendor refusal
stay unchanged when `rpc_split` is absent. Do not add desktop recommendations until the hardware gate and failure
matrix pass.

## 9. Numerical correctness policy

There are two different correctness claims:

1. **Same-backend RPC:** with identical binaries, devices, layer boundary, and settings, output must match the
   in-process split bit-for-bit for fixed greedy windows and prompts. This validates transport, ordering, state, and
   handoff sizes without a numerical excuse.
2. **Mixed-backend RPC:** no bit-identity claim. Add a cross-backend stage harness that sends identical inputs to
   CUDA and SYCL stages, records finite outputs and error against the available CPU/reference path, and runs the
   existing fixed served-output suite. Establish tolerances from the reference and same-backend variability before
   making them CI gates; do not choose a tolerance merely to make a failing run pass.

At minimum, exercise every layer type (GDN and QSA), MoE hit/miss/mirror paths, several valid split boundaries,
rejected drafts and commit prefixes, and a long-context read. Repeated mixed runs with the same seed and config must
be deterministic unless an existing engine option explicitly says otherwise.

## 10. Performance and observability

The first transport is intentionally simple: synchronize, CPU-copy through shared memory, then run the next stage.
It is not called zero-copy. The handoff is small relative to model weights, but its latency is on the decode path.

Add per-window/chunk counters for worker GPU time, controller GPU time, worker wait, controller wait, copy time and
bytes, commit time, prefill chunk time, and reconnect/retry count (which must remain zero). Expose them through the
existing diagnostic/INFO path only when RPC is active.

If the measured mixed run is below 20 tok/s, optimize in this order:

1. verify layer placement and head/MTP ownership; a bad boundary can leave the final card without expert/KV room;
2. remove extra synchronizations and copies, without changing the default path;
3. enlarge/double-buffer prefill transfers and pipeline prompt chunks;
4. evaluate backend-specific registered shared-memory staging separately for CUDA and SYCL;
5. only then consider overlapping decode windows. Preserve commit ordering and do not import the in-process GPU flag
   protocol.

Every optimization needs same-day before/after runs on the same mixed hardware and the same long-context workload.

## 11. Validation matrix

### CPU/build tests

- Frame round-trip, every malformed/truncated/oversized case, unknown version/opcode, sequence/state violations.
- UDS fd passing, memfd sizing/seals, nonblocking/deadline I/O, peer UID/token rejection, cleanup after abnormal exit.
- Config schema, backend/device resolution, forbidden option combinations, no token in argv/logs.
- Supervisor startup order, startup timeout, crash propagation, cancel, shutdown, and orphan-process checks.
- Default engine startup and same-vendor split tests with no `rpc_split`.
- Build CUDA and SYCL for every shared RPC change; build HIP too when a shared CUDA-tree file is touched.

### Same-backend hardware gate

Run two engine processes on two same-vendor devices. Compare against the equivalent in-process layer split:

- fixed decode windows, including partial commits, match bit-for-bit;
- fixed prompts and generated tokens match bit-for-bit;
- prompt/decode telemetry identifies RPC overhead;
- disconnect and timeout never produce partial output presented as successful.

### Mixed CUDA + SYCL hardware gate

- Handshake rejects wrong model geometry, artifact, context, layer range, and protocol version.
- Small-context smoke first, then Q4_K_M at 131,072 context.
- Run the fixed correctness suite and repeated-seed determinism check.
- Read a real prompt of roughly 104K tokens or more, answer coherently, then decode enough tokens for a stable rate.
- Meet **>=20 generated tokens/s**, record prefill/decode and RPC/stage timings, and show both processes/devices were
  active with the expected layer ranges and memory budgets.
- Repeat startup/shutdown/cancel/crash checks while the model is loaded and while a request is active.
- Mark anything not executed as **not run**, per CONTRIBUTING.md; compilation is not hardware validation.

## 12. Security and later network workers

Local mode uses a same-UID private socket directory, peer credentials, a per-run random token, and an fd-passed
memfd. It creates no TCP listener. A local worker still treats all lengths, states, and paths as untrusted input.

Network workers are a later phase, not a flag on the local transport. That phase needs an explicit bind/access
choice, TLS or an equivalent authenticated channel, certificate/key provisioning, model-artifact distribution or a
verified shared artifact, reconnect policy, and a measured WAN/LAN latency budget. The frame/version/bulk interface
should permit a TCP transport, but the first milestone must not expose one.

## 13. Implementation order and checkpoints

The executable task breakdown is in [tasks/todo.md](../../tasks/todo.md), with the dependency overview in
[tasks/plan.md](../../tasks/plan.md). The required order is:

1. protocol/transport/config/supervision foundation;
2. backend-local stage extraction with default-path parity;
3. same-backend decode RPC, then prefill RPC, as vertical slices;
4. SYCL worker support and mixed CUDA+SYCL execution;
5. failure hardening, measured performance, docs, and only then public launch/setup polish.

Do not begin with automatic placement, a GUI, network exposure, or an expert-only substitute. Do not declare the
feature complete until the mixed long-context 20 tok/s gate has actually run.

## 14. Main risks

| Risk | Impact | Mitigation |
|---|---|---|
| RPC boundary latency prevents 20 tok/s | High | Whole-layer windows, controller-last head, bulk memfd transfers, detailed timings, measured placement before zero-copy complexity |
| Stage extraction changes existing decode/prefill | High | Land extraction separately; same-backend and default-path parity checkpoint before RPC |
| Full weights/arena duplicated in each process | High | Mandatory range trim; audit CUDA pinned arena; explicit RAM/VRAM budget before load |
| Cross-vendor numerical drift changes tokens | Medium | No bit-exact claim; same-backend exact gate, mixed reference/output suite, deterministic repeated runs |
| Worker crash corrupts or strands conversation state | High | Fail closed, no mid-session restart, retain split-checkpoint refusal, process cleanup tests |
| SYCL host-memory/graph semantics differ | High | CPU-copy after stream sync; no shared GPU flags; backend-specific optimization only after measurement |
| Large `generate.cpp` refactor becomes unreviewable | Medium | Small backend-local extraction tasks; no formatting or unrelated cleanup; one behavior per PR |
| Feature becomes an experimental but advertised default | Medium | Explicit run config only; retain current mixed-vendor policy until all gates and docs are complete |
