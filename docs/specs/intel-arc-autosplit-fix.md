# Design: fix the auto-split K=2 pathology (Intel Arc pair)

Status: **design, 2026-10-13.** Companion to [intel-arc-per-stage-mirror.md](intel-arc-per-stage-mirror.md).
Read with [intel-arc-dual-gpu.md](intel-arc-dual-gpu.md) WS2.

## The bug (measured)

`--layer-split auto` on the B65+B70 pair chose **K=2** — 2 layers on the B65, 46 on the B70 — and the engine then
REFUSED to start because the B70's expert cache couldn't hold its 46 layers. An explicit balanced
`--layer-split 24` works and runs at 67 tok/s.

## Why (the code)

`sycl/src/program/generate.cpp`, the `auto` placement search (lines ~3556-3745):

- Per-card speed estimate (3556-3558): `speed = SMs × GHz`, so `layer_ms` for the B65 = 1.51 ms (20 SMs @
  2.40 GHz) and the B70 = 0.81 ms (32 SMs @ 2.80 GHz).
- The objective (3721): `ms = miss_ms*(1 − held_mass) + Σ (layers_i × layer_ms[i])` — it **minimizes predicted
  decode-window latency**. Because the B70 is ~1.9× faster per layer, the latency term dominates and the search
  piles 46 layers onto the B70 to minimize `Σ layers×layer_ms`.
- The capacity side (`held_mass`, via `range_hold`) is priced, but the latency term wins. **The search optimizes
  speed, not feasibility.** It never checks the `later_misses > 0` infeasibility that `generate.cpp:4979` computes
  *afterwards* and REFUSES on. So `auto` can pick a split the engine itself then refuses to run.

## The defect, one line

The auto-split's objective has **no feasibility term**: a placement that would leave a later stage's experts
unhomed (the REFUSED condition) is scored only on latency and can win, then fails at start.

## Fix options

1. **Feasibility gate in the search (the real fix).** While enumerating placements, compute each candidate's
   `later_misses` the same way line 4979 does (`(le−lb)*n_expert − cache_slots_for_that_stage`) and **reject
   infeasible placements** (skip them) before comparing latency. `range_hold` already knows each stage's held
   count (`held_at[key]`), so a stage's misses are `(le−lb)*n_expert − held_at[key]`. Under
   `STRATA_VERIFY_NO_HOST`, drop any placement with `later_misses > 0`. This makes `auto` never choose a split
   the engine would refuse. Cost: negligible (the data is already computed). **This is the upstreamable fix.**
2. **Capacity-first weighting (softer).** Weight `held_mass` higher (lower `miss_ms`, or add a term rewarding
   balanced residency) so equal-VRAM cards trend to a balanced split. Helps but doesn't *guarantee* feasibility.
3. **Setup-side default (out of engine).** When setup detects two equal-VRAM Arcs, write an explicit balanced
   `--layer-split` instead of `auto`. A workaround, not a fix; the engine bug remains for hand-run configs.

## Recommendation

**Option 1** — add the feasibility gate to the `auto` search. It's small, uses data the search already computes,
and makes `auto` safe on any pair (not just this one). Keep it byte-identical for placements that were already
feasible (it only removes infeasible winners, never changes a feasible choice). Option 2 is a possible follow-up
tuning; Option 3 is a stopgap if the engine change is deferred.

## Acceptance

`--layer-split auto` on the B65+B65/B70 pair (and any unequal-speed, equal-VRAM pair) no longer selects a split
that triggers the `later_misses` REFUSED — it picks a feasible placement (balanced or skewed within capacity) and
the engine starts. Measured on the pair; the previously-working NVIDIA/AMD feasible placements unchanged.
