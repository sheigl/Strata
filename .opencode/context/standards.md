# Strata — coding standards for the pooled mixed-vendor rig workstream

Version 0.2.5 / engine 0.1.50 conventions, read from the code, narrowed to what the rig feature must follow.
See `AGENTS.md` and `CONTRIBUTING.md` for the project-wide rules; this file adds the specifics for
`src/rig/`, the two engine trees and the layer-split path.

## 1. Where code goes

- **Vendor-neutral code lives in the root tree.** `sycl/CMakeLists.txt`'s `strata_resolve()` takes the root
  source when the port has no migrated copy (`sycl/CMakeLists.txt:107-127`). So `src/rig/*.cpp` and
  `include/strata/rig/*.hpp` compile in both trees with no duplication. A port-only copy of a shared file is a
  last resort and must say why in its header comment.
- **The rig layer may not include a vendor header.** No `cuda_runtime.h`, no `sycl`/`dpct` headers in
  `wire.hpp`, `link.*`, `shm.*`, `handoff.*`, `planner.hpp`, `worker.cpp`, `remote_stage.cpp`. Vendor behaviour
  reaches the rig layer through function-pointer hooks (`RigStageHooks`), the same shape as `PoolFn`,
  `PoolMultiFn` and `GpuPlanSink::publish/fetch`.
- Engine entry points stay in each tree's `src/program/generate.cpp` (the port keeps its own migrated copy).
  Do not move stage construction out of that file; hook into it.

## 2. Error handling: refuse at start, latch at run time

The house pattern, in the code's own words:

> "`failed` LATCHES and `fail` IS ONLY THE MESSAGE … resuming into a plausible-looking token is the exact
> outcome this whole mechanism exists to prevent." (`include/strata/core/expert_source.hpp:652`)

- Start-up problems are **one line with both numbers and the flag that would fix it**, e.g.
  `REFUSED: stage 2 needs 18.4 GiB of expert cache, sycl/1 has 15.2 free; --layer-split 16, --vram-reserve-mib,
  or --worker-resident-budget-gib N` — the shape of the existing split refusals
  (`src/program/generate.cpp:5077`).
- Runtime failures latch a flag and abort the request; they never fall back to another stage, another cut or the
  single-card path. Never emit a token after a boundary failure.
- Every bounded wait prints what it waited for and how long (`Verifier`'s bounded hand-off wait and `herr_h_` are
  the precedent). Timeouts: `--worker-timeout-s` (default 30), `--worker-exit-timeout-s` (default 10).
- Capability mismatches are decided at `CONFIGURE` (a bitmap), not discovered mid-request.

## 3. Numerics and parity

- The default path (no `--worker`) must be byte-identical to the last release. Any change to shared code —
  T8's `set_next` factoring is the one — is proved with `--window-hashes` + `--spec-follow` on an in-process
  split, diffed before and after, and that diff is part of the change, not a promise.
- A mixed-vendor run makes **no** byte-identity claim against a single-vendor run. The comparison ladder is:
  default vs default (identical) → in-process split vs cross-process split at the same cut (identical) →
  mixed vs same-vendor-cross-process (expected to differ; what differs is recorded).
- No tensor splitting anywhere. A cut list is a list of whole layer boundaries; the planner has no other shape.

## 4. Style of the files being touched

- Comments state **why** a guard exists and what going wrong would look like, usually with the issue number and
  the measured figure (`#1616: 576 -> 347 tok/s`). A comment that only restates the code is noise here.
- Flags: long `--form`, parsed in the big `else if (a == "--flag")` chain, echoed in `--help` with the reason
  and the default. Run-config keys: `snake_case`. Environment variables are diagnostics (`STRATA_RIG_HASH`,
  `STRATA_RIG_STATS`), never the only way to turn something on.
- No new third-party dependency. No RPC framework, no TLS library, no llama.cpp interop. POSIX
  `socketpair`/`memfd` only.
- Pure rules go in header-only files under `include/strata/program/` or `include/strata/rig/` so they are
  unit-testable with no GPU, like `include/strata/program/split_rules.hpp`.

## 5. Sample: a rig rule and its test

```cpp
// include/strata/rig/planner.hpp - a mixed rig never gives a slower card more layers than its share of the
// token budget: the stages run in sequence, so the slow stage is the whole rig (the pair measured 26.6 tok/s
// for 48 layers across two xe cards; one GeForce is several times that per layer).  Header-only, no GPU, like
// split_rules.hpp.
inline int64_t vendor_max_layers(int64_t n_layers, int64_t stage_ms, int64_t total_ms_budget,
                                 int64_t min_stage_layers) {
    if (stage_ms <= 0) return n_layers;
    const int64_t cap = (total_ms_budget * n_layers) / stage_ms;
    return cap < min_stage_layers ? -1 : cap;      // -1: this card cannot hold a stage at all -> REFUSED
}
```

```cpp
// tests/rig/planner_test.cpp - the mixed rig's placement rules: no GPU, no model.
#include <cstdio>
#include "strata/rig/planner.hpp"
static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
int main() {
    CHECK(vendor_max_layers(48, 30 /*ms per stage*/, 40 /*ms/token budget*/, 3) == 64);  // no cap
    CHECK(vendor_max_layers(48, 300, 40, 3) == 6);        // the slow card: 6 layers, not a balanced 24
    CHECK(vendor_max_layers(48, 3000, 40, 3) == -1);      // cannot even hold 3 layers inside the budget
    return failed ? 1 : 0;
}
```

## 6. Cross-process invariants the rig code must keep

- A worker owns exactly its `[lb, le)` layer range. A request for a layer outside it is an error message, never
  a fallback (`SplitDrive` picks a stage by layer index; two processes must not be able to disagree).
- Hand-off rows are validated by `seq` + `epoch` (+ payload hash when enabled) before any kernel reads them.
- Each process's pinned/mirror allocations are bounded by a budget the **controller** computed from the host's
  free RAM minus its own reservation. `host_available_memory()` is a per-process view of one machine: two
  processes that each fit can together not fit.
- Each process's CPU expert pool is given a core count; the default "every core but one" is wrong for two
  processes.
