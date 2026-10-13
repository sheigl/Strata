# Design: per-stage host mirror for layer-split (WS5 capacity fix)

Status: **design, 2026-10-13.** The fix that lets a later split stage's expert overflow be served from pinned
host memory instead of REFUSED. Unblocks Q4_K_M (62 GB arena) on the B70+B65 pair. Read with
[intel-arc-dual-gpu.md](intel-arc-dual-gpu.md) WS5 and [intel-arc-ws0-rootcause.md](intel-arc-ws0-rootcause.md).

## The problem (measured)

On a 2-card split under `STRATA_VERIFY_NO_HOST` (required on xe), the engine REFUSES to start when a **later**
stage's expert cache can't hold all its layers' experts:

```
REFUSED: 3686 experts of the later layer-split stages are neither in VRAM nor mirrored
(the host mirror covers the first stage only); ... the card is lost at the first verify window
```

Q4_K_M on the pair at `--layer-split 24`, 131K context: B65 (layers 0-23) holds its 12,288; B70 (layers 24-47 +
head + MTP + the whole session KV) holds 8,602 of 12,288 → 3,686 overflow, nowhere to live → REFUSED. Lowering
context to 64K barely moved it (3,665) — the squeeze is the head card's dense weights + head + MTP + KV, not the
KV size alone.

## Why it's first-stage-only today (the code)

- `generate.cpp:4812`: `mirror_end = multi_gpu ? split_at[0] : n_layers` — the mirror covers only the first
  stage's layers.
- The mirror is one `FileExpertSource`/`GgufExpertSource` host-pinned allocation + one device table
  `mirror_table_d` (`generate.cpp:4868`), allocated on the default in-order queue (the first card's context).
- The verify plan reads a **global** `g_mirror_table` via `resident_plan_set_mirror` (verify_kernels.dp.cpp:2368),
  and it's wired only to the first stage's residency table `thits.d_res` (`generate.cpp:7019, 12124`). Later
  stages' verifiers get no mirror.
- Comment `#1054` (generate.cpp:4809-4811) gives the original reason: the pinned memory belongs to the first
  GPU's context, and when it's built the later stages' caches don't exist yet — so a naive all-layers mirror
  would pin RAM for experts the other card holds.

## Design

All stages live in **one process**; each `GpuStage` (generate.cpp:1083) has its own `dev`, `cache`, `d_res`, and
verifier `ver`. The fix is to give **each later stage its own host mirror**, populated with *its own* layers'
non-resident experts, and point that stage's verifier at it.

Key facts that make it tractable:

- The SYCL host-mirror is **unified/shared pinned host memory** read by the GPU over PCIe (`device_alias` returns a
  host address the device dereferences). It is not a device-local buffer, so a later stage's GPU can read a mirror
  allocated on the host — no cross-device P2P needed. (Confirm on the path: `device_alias` /
  `resident_plan_kernel` only needs a host-visible address.)
- The residency check `res_layer >= g_mirror_res` and the table slice `g_mirror_table + (res_layer - g_mirror_res)`
  already index by global layer; a per-stage table just needs the right base pointer and `d_res` for that stage.

### Changes (ordered)

1. **Build the mirror after all stages' caches exist.** Today it's built where only the first cache exists
   (that's why `mirror_end = split_at[0]`). Move mirror construction to after the stage loop, and extend the
   layer range past `split_at[0]` to `n_layers`, collecting each layer's misses against *that stage's* cache
   (`stages[i]->cache`), not just the first stage's `xcache`.
2. **Per-stage mirror table + residency wiring.** Instead of one global `g_mirror_table` on the default device,
   give each stage a `mirror_table_d` allocated on **its** device context and call `resident_plan_set_mirror` for
   **each stage's** verifier with that stage's `d_res` and table. Because `g_mirror_table` is a global, this means
   either (a) making it per-stage (an array indexed by stage, set before that stage's verify), or (b) passing the
   table through the verifier rather than a global. **(b) is cleaner and the upstreamable shape** but touches the
   `resident_plan` call signature; (a) is a smaller diff but keeps a global that must be swapped between stages —
   a thread-safety hazard if stages run concurrently. Decide at build time; default to (b).
3. **RAM budget across stages.** The single `STRATA_MIRROR_MIB` / MemAvailable cap must be split across stages so
   the first stage doesn't consume it all. Allocate the cap proportionally to each stage's miss bytes, or give
   each stage `cap / n_stages` with the hottest-first order already used.
4. **Keep the REFUSED as a safety net**, now counting *all* stages' unmirrored misses (a stage whose overflow
   exceeds its mirror share still refuses cleanly rather than losing the card at verify).

### Explicitly not in scope

- Splitting the session KV across cards (the other big head-card load) — separate, larger work.
- The MMQ K-quant prompt path (speed) — separate WS5 fast-path item.
- Changing decode numerics. The mirror is a *source* for experts a slot doesn't hold; output is identical to a
  full-VRAM run by the same argument the existing first-stage mirror already relies on.

### Risks / what to verify on-card

- **Whether a later stage's GPU can dereference a host pointer allocated under the first stage's pinned
  allocation.** ~~The design assumes host-pinned memory is readable by any card (true for SYCL host USM on xe).
  Verify with a 2-card smoke test before the full build.~~ **CONFIRMED 2026-10-13:** `sycl/probe/mirror_xcard.cpp`
  (host USM allocated on the B65's queue, read by a kernel on the B70, 4 MB) → `0/1048576 wrong`,
  "cross-card host-mirror read WORKS". The design's central assumption holds on this machine (bmg-g31, oneAPI
  2026.1.1, xe).
- **Thread-safety** if `resident_plan_set_mirror` stays global and stages verify concurrently — the strongest
  argument for the per-verifier (b) shape.
- **PCIe bandwidth** when both cards stream mirrored misses at once (each on its own link; the docs' probes were
  per-card). Decode slowdown on mirrored experts is expected and acceptable for a first correct result.
- House rules: opt-in (a `STRATA_SPLIT_MIRROR=1` or automatic-when-needed), default path byte-identical, measured
  numbers on the pair, and it must not change single-card or first-stage-only behavior.

## Acceptance

Q4_K_M `--layer-split 24` (or the right skew) on the B70+B65 pair starts instead of REFUSING, serves a real prompt
correctly at a stated decode tok/s, at a stated context, with no card loss at the verify window. Then re-measure
against the 131K gate.
