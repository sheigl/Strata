# Testing the pooled mixed-vendor rig

What the Test agent runs, and the shapes the tests take. Four layers: pure-C++ unit tests, two-process
transport tests, engine parity (byte-identity) runs, and the Terminal/server Python tests. GPU work is only in
the last two, and the gate run needs both cards.

## 1. Commands

```sh
cmake -B build -DSTRATA_BUILD_TESTS=ON && cmake --build build -j && ctest --test-dir build -R rig_   # rig unit tests, no GPU
python -m pytest tests/terminal                                                                  # the Terminal
cd apps/host && npm test                                                                         # the app UI (only if the app role UI changes)
python tools/test_setup_choices.py                                                               # setup's own choice/admission tests
```

Build every backend a touched file belongs to before asking for review (`CONTRIBUTING.md`): the CUDA build and
`sycl/` must both configure and compile when `src/rig/**` or `include/strata/rig/**` changes.

## 2. Pure rule tests (`tests/rig/`, no GPU, no model)

The house style, copied from `tests/program/split_rules_test.cpp`:

```cpp
// The hand-off frame's guards: no GPU, no process.
#include <cstdio>
#include <vector>
#include "strata/rig/wire.hpp"

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    using namespace strata::rig;
    static_assert(sizeof(Header) == 24, "the frame header is on the wire");
    Header h{kMagic, kProto, (uint8_t) Msg::RunWindow, 0, 7, 40, 0x1234};
    // a worker built on another protocol is refused with both numbers, never half-started
    Header other = h; other.proto = kProto + 1;
    std::string err;
    CHECK(!check_header(other, err));
    CHECK(err.find("proto") != std::string::npos && err.find(std::to_string(kProto)) != std::string::npos);
    // a truncated frame and an oversize one are errors, not short reads
    Header big = h; big.len = kMaxFrame + 1;
    CHECK(!check_header(big, err));
    Header bad = h; bad.magic = 0;                       // never trust a stream we did not write
    CHECK(!check_header(bad, err));
    return failed ? 1 : 0;
}
```

Planner rules are tested the same way: cuts honour `kMinStageLayers`, a slower vendor never gets more layers
than its cost cap, an impossible rig produces one refusal string with both numbers, and the two processes'
pinned budgets never both claim the same free RAM.

## 3. Two-process transport tests

`link_test.cpp` and `shm_test.cpp` fork: the child either moves bytes or dies abruptly.

```cpp
// the ring refuses a stale row instead of handing the residual stream to a kernel
CHECK(ring.publish(0, 41, /*hash=*/true));
void* out = buf.data();
std::string err;
CHECK(!reader.acquire(0, /*expect=*/42, out, bytes, /*timeout_ms=*/50, true, err));   // stale: refuse
CHECK(ring.publish(0, 42, true));
CHECK(reader.acquire(0, 42, out, bytes, 50, true, err));                              // exact: accept
// the writer died: acquire times out rather than spinning forever
CHECK(!reader.acquire(0, 43, out, bytes, /*timeout_ms=*/100, true, err));
```

`handoff_test.cpp` must use the real row layout (`Verifier::handoff_floats(g)` with the pack's `hc` and
`n_embd`, rows = `kVerifyMaxT` and the `--batch` rows case) and assert the bytes round-trip exactly.

## 4. Engine parity — the tests that matter

```sh
# ground truth: the in-process split
strata --pack PACK --native GGUF … --serve --layer-split 24 --split-device 1 \
       --window-hashes /tmp/inproc.wh --spec-follow /tmp/tokens.txt
# the rig: two processes, same cut, same prompt
strata --pack PACK --native GGUF … --serve --layer-split 24 --split-device 1 \
       --worker build-cuda/strata --window-hashes /tmp/rig.wh --spec-follow /tmp/tokens2.txt
cmp /tmp/inproc.wh /tmp/rig.wh && cmp /tmp/tokens.txt /tmp/tokens2.txt
```

`--window-hashes` (index, position, size, 64-bit hash of the window's final residual rows and its argmaxes) and
`--spec-follow` exist exactly for this comparison; run with `--adapt-every 0` and `--seed` fixed so the two runs
are on the same text. Expected results per comparison:

| Comparison | Expected |
|---|---|
| default (no split) before vs after the T8 refactor | byte-identical |
| in-process split vs two-process split, same vendor, same cut | **byte-identical** (the boundary is proved here) |
| mixed (CUDA+SYCL) vs any single-vendor run | differs; only self-consistency across repeats is claimed |

## 5. Serve behaviour on a rig

Extend `serve/test_server.py`: `"stage_workers"` reaches the engine args; a config naming a backend the platform
does not support is refused; `worker-lost` is a restart reason and the engine's `ENGINE_QUIT/TERM/KILL` ladder is
kept; the feature is gated by grepping the engine binary for `--rig-worker` (the `--tail-role-token` pattern).

Terminal (`tests/terminal/`): `strata launch --plan` prints the rig (two processes, one card each) and refuses an
unsupported make × platform; `strata launch --stop` leaves no process holding device memory. These run against a
mock engine, as the existing launch tests do.

## 6. The gate run (recorded, not just passed)

On the CUDA + Arc host, Q4_K_M at ≥131,072 context: record prefill tok/s and decode tok/s, drafts accepted,
per-stage ms (`--stats`, `--stage-timing`), pinned and mirror GiB of **both** processes, and the boundary cost
with `STRATA_RIG_STATS=1`; compare against a same-day run of the same build on the single-vendor pair. Kill the
worker mid-window, kill the controller, start a too-small capacity config and a version-skewed worker, and record
those arms separately from the passes. Anything inferred from reading code is written as unconfirmed static
analysis.
