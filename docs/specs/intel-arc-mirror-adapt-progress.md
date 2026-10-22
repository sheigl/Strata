# STRATA_MIRROR_ADAPT: implementation + validation progress (2026-10-22, session 2)

This file reports the progress of the session that picked up the handover in
[`docs/specs/intel-arc-mirror-adapt.md`](intel-arc-mirror-adapt.md). It is written as the work proceeds;
the state below is the last thing verified, with what passed, what failed and what was not run kept apart.

## 0. Headline

- The change **builds, serves, is byte-identical and never crashed** in ~6 sessions on the pair (build fix in §1).
- The deferred-path guard (§5.2.1) is verified: with the switch unset, the split configs now turn the adaptive
  tier off under `STRATA_VERIFY_NO_HOST`, with the message naming `STRATA_MIRROR_ADAPT=1`.
- **The feature has no observable effect in the tested configuration** (single-stream serve, `STRATA_VERIFY_NO_HOST`):
  the adaptive tier never *selects* a swap there, so there is nothing to adopt — no cap spend, no adopt, no
  speed change. Root cause found in code and confirmed empirically (§3): the tier's routing counts
  (`drive.d.usage`) are written only by the host CPU pool, which the NO_HOST device plan bypasses; the device
  plan can count only when given a counter buffer, which the live configs do not do.
- Byte identity holds (§5). No decode gain in this workload; the read-speed difference seen in the A/B is
  environmental (run ordering / page cache), not the feature (§4).

## 1. Build (§5.1) — done

The handover's build command was re-run after fixing the one open compile error:

- The previous session's forward declaration `namespace sycl { class queue; }` in
  `sycl/include/strata/core/gguf_expert_source.hpp` declared a **new class** in namespace `sycl`, but the real
  queue is `sycl::_V1::queue` (the SYCL headers put it in `inline namespace _V1`). Every TU that included both
  the header and SYCL produced `reference to 'queue' is ambiguous` / `member access into incomplete type`
  (25 errors, in `gguf_expert_source.cpp`, `prefill.cpp`, `generate.cpp`).
- Fix: `namespace sycl { inline namespace _V1 { class queue; } }` — the same shape the Intel SYCL headers use
  for their own forward declaration (`sycl/ext/oneapi/bindless_images_memory.hpp`). One line, no redesign.
- Rebuild: `BUILD EXIT 0`, `errors: 0`. No new warnings in the three touched files.

Files in the change: `gguf_expert_source.hpp` (+12), `gguf_expert_source.cpp` (+52), `generate.cpp` (+111) —
the handover's §4 state plus the one-line header fix.

## 2. Validation gates (§5.2)

### Gate 1 — default path (done, with one correction to the doc's premise)

- The **IQ3_S pair is not 100% resident**: its split build mirrors **2.84 GiB** of start misses, so
  `miss.empty()` is false and the deferred-path guard fires. The handover's "(no misses) prints nothing new"
  expectation only holds for a config with zero misses, which the IQ3_S split config is not.
- With the switch unset, the IQ3_S pair now prints (once, at start):
  `strata generate: adaptive swaps off: with STRATA_VERIFY_NO_HOST an expert they move out of VRAM would be
  neither there nor mirrored (STRATA_MIRROR_ADAPT=1 adopts swap-outs into the mirror and keeps the tier on)`
  and sets `adapt_every = 0`. This is the **intended bug fix**: the deferred `STRATA_SPLIT_MIRROR` path now
  gets the same guard the single-table path always had.
- Q4_K_M split, switch unset: same line (verified in the A/B off arm).
- Not changed: no other new log lines; the engine starts, serves and stops normally.

### Gate 2 — feature on (done; the feature is wired but inert in this configuration)

- Q4_K_M pair + `STRATA_MIRROR_ADAPT=1`: the engine prints
  `strata generate: STRATA_MIRROR_ADAPT=1: the adaptive tier stays on under STRATA_VERIFY_NO_HOST; each
  swap-out is adopted into the pinned mirror first (cap STRATA_MIRROR_MIB)` and serves normally.
- The serve path has no `STRATA_TRACE_ADAPT` output at all (the trace lines exist only in the CLI path,
  `generate.cpp` ~12696/12737; the serve path's `adapt` lambda ~8260 has none), so §5.2.2's "watch the swap
  round" cannot be done on the serve path as-is.
- Direct probe (existing test aid, no code change): `STRATA_CHECK_SWAPS=1` with the switch on, warmup + two
  docs prompts (≈460 tokens of decode) → **zero** check lines. `resident_stage_swaps` early-returns on
  `swaps.empty()` (line 325) before the check print, so this means **no swap was ever selected** in the whole
  session.
- The cap gate likewise produced no `the mirror cap is spent` note: nothing was ever adopted.

### Key finding — why the tier never swaps under NO_HOST

The adaptive tier's candidate rule is `u[e] >= 2.0f` on `drive.d.usage` (the decayed routing counts), where
`u[e]` is non-resident experts. Where can those counts come from in a serve session?

- Host side: `expert_source.cpp:2578` (inside `expert_pool_dispatch_multi`) — written **only when the host CPU
  pool computes routed experts**. Under `STRATA_VERIFY_NO_HOST` the device plan serves the verify window from
  VRAM and the mirror; the host pool does not run, so the counts stay at their initial zeros (assigned at
  `generate.cpp:8147` when the tier is on).
- Device side: the verify kernel's `usage` counter (`verify_kernels.dp.cpp:2116`) is only wired when
  `resident_plan_set_mirror` gets a non-null counter buffer, which happens only with `--expert-profile-save`
  (`generate.cpp:7364`, `usage_dev_srv`) or `STRATA_USAGE_DUMP` (CLI path, 12626). The live pair configs set
  neither.

So in single-stream NO_HOST serving the tier scans 24,576 zero counts every `--adapt-every` rounds and returns
with an empty swap list — exactly what the probe saw. This also explains why the handover's observed hole
"hasn't blown up in single-stream use": swaps need counts, and single-stream NO_HOST produces none. (The
handover's note that "it needs a batch window counting misses to trip" is consistent: a batch path that runs
the host pool would produce counts.)

**Consequence:** `STRATA_MIRROR_ADAPT=1` is correct but has nothing to act on in the tested configuration.
The §3 motivation ("the cache never adapts to the session … ~27 tok/s ceiling") is real, but the missing
piece is the counts, not the guard: to make the tier adapt under NO_HOST, the device plan's counts would have
to be turned on and read back when the tier is on (e.g., allocate `usage_dev_srv` whenever
`mirror_adapt && mirror_table_d`, and refresh `drive.d.usage` from it before each adapt round). That is a
change beyond this handover and was not made.

### Gate 3 — speed A/B (done; no attributable difference)

Harness: `/tmp/opencode/mirror-adapt-ab2.sh` — Q4_K_M pair, same deterministic prompt set (3 sequential
~9K-char docs prompts + 200 tokens each, then the 109K-token docs prompt + 400 tokens), temperature 0,
token capture from the server's OpenAI `logprobs`. Arms: `off2` (switch unset) then `on2` (switch set).

Big-prompt (110,154 tokens) numbers, engine's own timing:

| arm | read tok/s | decode tok/s (gen) | drafts accepted |
|---|---|---|---|
| `off2` (switch unset, first run of its pair) | 388.4 | 30.5 (400) | 226 of 328 |
| `on2` (switch set, second run of its pair) | 507.1 | 30.6 (400) | 226 of 325 |
| `off3` (switch unset, warm cache, run later) | 476.2 | 27.0 (200) | 100 of 134 |

The third arm settles it: **the switch-off arm reads at 476.2 tok/s once the page cache is warm** — the
388.4 vs 507.1 difference was run ordering / cache state, not the feature. With the tier inert in both arms
(the key finding above), no speed difference is expected, and none attributable is measured. Decode is
unchanged within noise (30.5 vs 30.6 with 400 generated; 27.0–27.1 in the 200-token arms).

### Gate 4 — byte identity (done, pass)

The exact generated token **bytes** (from the server's OpenAI `logprobs`, `top_logprobs=0`) are identical
between `off2` and `on2` for all four prompts: 732 / 818 / 692 / 1603 token bytes, prompt by prompt
(`cmp` clean). Greedy output is token-for-token identical with the switch off and on.

### Gate 5 — cap behavior (done; not exercised)

`STRATA_MIRROR_MIB=24000` (≈600 MiB above the 22.85 GiB start mirror) with the switch on: the engine starts,
serves the full prompt set and stops normally — **no crash, no cap note**, because the tier never selected a
swap to adopt (the key finding above). The cap path in `adopt()` itself was not exercised end-to-end; that
needs a configuration where swaps happen (a batch window) and is listed under "what is left".

### Gate 6 — setup suites (done)

- `python3 -m pytest tests/terminal -q --continue-on-collection-errors`: **1097 passed, 23 failed,
  69 skipped, 8 collection errors**. The 8 collection errors are `ModuleNotFoundError: No module named
  'textual'` (the TUI dependency is not installed in this environment). The 23 failures are pre-existing:
  re-running the same files with the change stashed reproduces the same 23 (filesystem mtime granularity in
  `test_shared_models_folder` and friends). None of it touches the change (C++ only, `sycl/`).
- Setup suites: `test_setup_sycl`, `test_setup_choices`, `test_setup_amd`, `test_sycl_build_guard` — all
  **OK**.

## 3. Harness notes and mistakes fixed mid-session

- The first A/B passed `"0"` to `if on:` in Python (a non-empty string is truthy), so *both* arms ran with the
  switch on; its replies were also empty (the 200-token budget went to `reasoning_content`). Discarded and
  re-run with token capture.
- Readiness is the lowercase `ready:` line; the first harnesses grepped `READY` and wasted 20-minute polls.
- The engine's RSS is not a mirror proxy (both arms grow ~900 MiB; the 22.85 GiB mirror never shows in VmRSS,
  `VmPin` reads 0 on this kernel).

## 4. State of the working tree

On branch `intel-arc-0.2.2`:

- `sycl/include/strata/core/gguf_expert_source.hpp` — `sycl::_V1::queue` forward decl, `adopt()` decl, chunk members
- `sycl/src/core/gguf_expert_source.cpp` — `GgufExpertSource::adopt()`
- `sycl/src/program/generate.cpp` — `mirror_cap_bytes`, `mirror_adapt`, the deferred-path guard, the adopt
  pre-pass + `apply_pending` publish in both the serve and CLI paths
- `docs/specs/intel-arc-mirror-adapt.md` (the handover) and this file

Committed on the branch as a single commit (guard + feature, per the handover's §5.3).

## 5. What is left / recommended next

1. **Give the tier counts under NO_HOST** if the §3 payoff is wanted: allocate the device counter buffer when
   the tier is on (not only with `--expert-profile-save`) and refresh `drive.d.usage` from it before each
   adapt round. Small, measurable, and it is the piece that would make the cache actually follow the session.
2. A batch-window run (`--batch 2 --batch-groups 1 --batch-mtp`, with `STRATA_CHECK_SWAPS=1`) would exercise
   the adopt path end-to-end (counts exist there) and is the natural place to validate the cap behavior and
   the adopt bytes. Not run this session.
3. PR split per the handover §5.3: (a) the deferred-path guard, (b) the feature — the PR text must say the
   feature is opt-in and, in the tested single-stream NO_HOST config, inert pending item 1; byte-identity and
   the guard are the verified parts.
4. Known limits (from the handover, still true): a stage whose table is null drops adopts silently; the mirror
   only grows; `STRATA_CHECK_SWAPS=1` validates the exchange path, not the adopt bytes; the non-serve CLI path
   refills CUDA0's cache only (pre-existing).

Not run / not planned this session (state in the PR): CUDA/HIP builds, single-card non-defer runtime, batch
windows, three-card runs (the B580 is retired).
