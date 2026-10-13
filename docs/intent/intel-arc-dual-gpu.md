# Intent: CUDA-level Intel Arc support, dual-GPU pooled (B70 + B65)

Status: **confirmed by user (explicit yes), 2026-10-12.** This is the handoff doc for the project. Future
sessions working on Intel Arc support start here. The downstream spec lives at
[docs/specs/intel-arc-dual-gpu.md](../specs/intel-arc-dual-gpu.md).

## Confirmed statement of intent

- **Outcome:** Bring Strata's Intel GPU support to CUDA-level quality — full model/quant parity, real speed,
  one-click setup — with the two Arc cards' VRAM pooled so they act as one ~32 GB device (B70 32 GB + B65, per
  the user; see open question below on actual card VRAM).
- **User:** The requester, on their own dual-Arc Linux machine; their 12 GB RTX 3060 is the standing reference
  point.
- **Why now:** Today's Intel path is an artificial straitjacket — setup presents only the q1-class variant as
  supported on Arc, and memory-management features (expert pinning) don't engage — so a single B70 can't run
  the iq3_xxs quant that the *lesser* 3060 runs fine. The hardware is capable; the software path is the
  bottleneck.
- **Success (hard gate):** The pooled B70+B65 pair runs at least a **q4_k_m** quant of the model — a *higher*
  quality tier than the 3060's iq3_xxs — correctly, at usable speed, **with a context of at least 131,072
  tokens**, installed via the same one-click Linux flow as CUDA.
- **Constraints:** Backend stack is oneAPI + Intel XPU libraries (already the port's direction); Linux only;
  the two cards pool VRAM (one logical device), not two independent workers; every model/quant variant that
  works on CUDA must work on Intel.
- **Out of scope:** Windows support; model fine-tuning/training ("fine tune" meant engine tuning);
  throughput-oriented multi-instance batching across the two cards; changes to non-Intel backends beyond what
  parity requires. The fork is only the vehicle — land upstream-shaped if the maintainers take it.

## Key corrections to the original premise (read before scoping work)

Found while writing this doc, from [docs/INTEL.md](../INTEL.md) and [docs/INTEL_ARC.md](../INTEL_ARC.md):

1. **A serious SYCL/oneAPI port already exists** (`sycl/`), by maxfridbe in PR #423, merged as experimental in
   0.1.39. It is not a stub: 157 build targets, kernel parity tests, measured 78 tok/s decode on a single B70
   (Coder IQ1_M), 256K context measured on one card. "Intel support is limited" is true at the
   setup/policy/polish level, not the engine-core level.
2. **The "only q1 works on Intel" claim is a setup/conservatism artifact**, not an engine limit: the port has
   run Coder IQ1_M, IQ2_XS, Flash-Next IQ2_XS and IQ3_S, and Swift 1.5 IQ2_XS on real Arc cards.
3. **Dual-GPU pooling already exists and has been run on Intel:** `--layer-split` is generic (not CUDA-only), and
   a community 2x Arc Pro B70 report in #423 measured Flash-Next IQ3_XXS at **66 tok/s decode, 394 tok/s
   prompt**. 2x B60 results exist too. So "make two Arcs pool" is hardening, wiring and validation, not greenfield.
4. **User-reported failure mode (this machine, 2026-10):** non-q1 quants (e.g. iq3_xxs) **never loaded at all**
   on the B70 — not slow, not wrong-output, but failed to load. The same iq3_xxs runs fine on their 12 GB RTX
   3060. Root cause unknown as of this writing: could be setup's model gating, the VRAM planner's fit check on
   `xe`, or a quant-kernel load path. The spec's parity workstream must reproduce and root-cause this first;
   it is the user's actual pain.
5. The maintainers' stated contributing rules ([AGENTS.md](../../AGENTS.md)) require: one change per PR, opt-in
   features with byte-identical default paths, measured numbers with hardware named, and building every backend a
   file touches. This project's PRs must follow that or they won't land.

## What "as good as CUDA" means here (yardstick, confirmed)

All three axes are binding, per the interview:

- **Parity:** every model variant, quant, and feature that runs on CUDA runs on Intel (including MTP speculative
  decoding, images if CUDA has them, `--kv-resident`, expert mirroring, the works).
- **Speed:** at least in the band of what the same money buys on NVIDIA; the user's concrete bar is dual 32 GB
  Arcs on q4_k_m ≥ their 3060 on iq3_xxs, plus the pooled pair not embarrassing a single card.
- **Setup:** `./setup.sh --backend sycl` (or plain `./setup.sh` auto-detecting Arcs) must just work: detect both
  cards, offer pooling, pick a valid model/context, no manual flags.

## The 131,072-token context requirement (added after first confirmation)

Per INTEL.md's own measured table, a single 32 GB B70 tops out at 131K context (1.5 GB headroom) and 262K
*livelocked the host* on the xe driver. With pooling, 131K must go from "practical ceiling" to "validated
default." Non-negotiable acceptance criterion:

- q4_k_m (or better) model, **≥131,072-token context**, a real prompt of ~104K+ tokens read and answered
  coherently, decode afterwards at a stated tok/s, no host livelock, on the B70+B65 pair.

## Current known gaps to close (starting list for the spec)

- Multi-GPU on Intel is community-reported, not maintainer-validated; setup does not offer Arc multi-GPU the way
  it does for NVIDIA (`--gpus`, auto recommendations).
- INTEL.md explicitly lists "Intel GPUs" under multi-GPU "Not supported" in [docs/MULTI_GPU.md](../MULTI_GPU.md)
  — that line must die, with numbers.
- Quant coverage: setup/whitelist limits which models Arc is offered; the engine runs more than setup admits.
- AOT targets exist for bmg-g31 (B70) and bmg-g21 (B580/B570/B60); confirm what the B65 needs (likely bmg-g31 or
  a sibling id; must be probed with `ocloc ids`).
- Images are "not yet" on Intel per the parity table.
- No ready-made Intel engine in release zips; source build only.

## Machine probe (2026-10-12, the user's box)

Ran at WS0 start. **The machine has three Arc GPUs, not two:**

| slot | PCI id | card | Level Zero index | AOT arch |
|---|---|---|---|---|
| 03:00.0 | `8086:e222` | Arc Pro **B65** | `level_zero:0` | `bmg-g31` (20.2.0) |
| 07:00.0 | `8086:e223` | Arc Pro **B70** | `level_zero:1` | `bmg-g31` (20.2.0) |
| 0c:00.0 | `8086:e20b` | Arc **B580** (12 GB) | `level_zero:2` | `bmg-g21` (20.1.0) |

Stack: `xe` driver on all three; oneAPI **2026.1.1** installed natively (`/opt/intel/oneapi`, `icpx` 2026.1.1-325,
`ocloc`, oneMKL — matches the version INTEL.md's numbers were made with; an earlier probe note saying 2025.3 was a
misread of the version dirs and is corrected here); Level Zero V2 runtime, NEO 26.31.39395.14; CPU i7-11700K,
**62 GB RAM**. No `strata-sycl-dev` Docker image present; native toolchain available and current.

Implications:

- The PCI ids `e222`/`e223`/`e20b` are **not in Strata's known list** (INTEL_ARC.md names B60 = `e211`/`e221`).
  `sycl/setup_intel.py` may not detect these cards at all — a likely contributor to the user's "quants never
  loaded" report, and the first thing WS0 checks in code.
- Mixed AOT targets across the pair the user wants (both G31, good) but the third card (B580, G21) complicates
  "use all cards" and must be excludable.
- The user's stated "32 GB pooled" assumed two 32 GB-class cards; actual pooled capacity of the B65+B70 pair
  needs VRAM confirmation (both are 32 GB-class; a third 12 GB card exists but the intent's gate is the pair).
- oneAPI is **2026.1.1 natively and current** (the earlier "2025.3 vs 2026.1" concern was a misread, corrected in
  the intent doc): the native build path is usable, no toolchain upgrade needed.

## Open questions the spec must resolve
- Whether "pooling" is layer-split (existing, works) or something deeper; the intent accepts layer-split as the
  architecture if it meets the gates, since that's what MULTI_GPU.md already ships.
- Which model variant is the q4_k_m gate target (Qwen3.8-Flash-Next base vs Coder vs Swift 1.5 vs Unsloth).
- Maintainer appetite for upstreaming; check TEST_REQUESTS.md and open issues before the first PR.

## How to resume this project in a new session

1. Read this file.
2. Read the spec: [docs/specs/intel-arc-dual-gpu.md](../specs/intel-arc-dual-gpu.md).
3. Read the WS0 root-cause: [docs/specs/intel-arc-ws0-rootcause.md](../specs/intel-arc-ws0-rootcause.md) —
   static analysis done 2026-10-12; primary suspect is the B65's PCI id (`e222`) missing from `INTEL_ARC` in
   `sycl/setup_intel.py`. Runtime confirmation is the pending next step.
4. Check git log/branch for where implementation left off.
5. House rules: [AGENTS.md](../../AGENTS.md), plus one change per PR, opt-in, byte-identical defaults,
   measured numbers with named hardware.
