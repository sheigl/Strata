# Intel Arc support in Strata: the B70+B65 dual-GPU effort — complete technical record

**Date:** 2026-10-13. **Branch:** `intel-arc-b70-b65` on the fork `github.com/sheigl/Strata`.
**Machine:** Arc Pro B65 (32 GB, `bmg-g31`, PCI `e222`) + Arc Pro B70 (32 GB, `bmg-g31`, PCI `e223`) + an idle
Arc B580 (12 GB, `bmg-g21`, PCI `e20b`); i7-11700K; 62 GB RAM; Ubuntu, `xe` driver, Level Zero V2
(1.17.39395), oneAPI 2026.1.1, no Docker (native host path).

This document is the full account: what we set out to do, what we found, every lever we pulled, every lever left,
and the Intel-specific optimization space still open. It consolidates [../intent/intel-arc-dual-gpu.md](../intent/intel-arc-dual-gpu.md),
[intel-arc-dual-gpu.md](intel-arc-dual-gpu.md), [intel-arc-ws0-rootcause.md](intel-arc-ws0-rootcause.md),
[intel-arc-per-stage-mirror.md](intel-arc-per-stage-mirror.md), and
[intel-arc-autosplit-fix.md](intel-arc-autosplit-fix.md).

---

## 1. The goal

Bring Strata's Intel Arc support to CUDA-level quality on the user's dual-Arc machine, where "as good as CUDA"
was confirmed (via the interview) to mean all three of: **parity** (every model/quant/feature that runs on CUDA
runs on Intel), **speed** (a stated, usable tok/s), and **setup** (one-click, like the NVIDIA flow). Hard
requirements added along the way: the two Arcs **pool VRAM** (act as one device for a bigger model), the result
runs at **≥ 131,072 context**, and the quality bar is **literal Q4_K_M** (the user's own Swift-1.5 Q4_K_M GGUF).

## 2. What the premise got wrong (and right)

The user's starting belief was "Intel support is limited; only the q1 variant works; bigger quants never load."
The investigation corrected almost every part of it:

- **A serious SYCL/oneAPI port already exists** (`sycl/`, merged experimental in 0.1.39, by maxfridbe in PR #423).
  Not a stub: full engine build, kernel parity tests, measured single-card numbers.
- **There is no quant whitelist.** Setup offers every quant on Intel (it fakes RAM high, so all "fit"), and even
  recommends IQ3_XXS. The "only q1" impression was wrong.
- **The real failure (WS0 root cause):** the user's installed engine was the **known-broken 0.1.39 binary** — per
  INTEL_ARC.md, that port "did not compile against its own engine sources (#784) and its ring waits never saw a
  slow layer as still running (#866, #867), fixed in 0.1.40," and *no one ran an unpatched 0.1.39 on real Arc
  hardware*. The user's logs confirmed it: IQ3_S **loaded fully** then crashed post-load with
  `verify: layer 0 never rang` + `level_zero error 37 (UR_RESULT_ERROR_UNINITIALIZED)`. Quants load; the crash was
  the wall.
- **A second, separate failure:** the user's dual-GPU attempt died on a malformed Level Zero selector
  (`level_zero:1,level_zero:0` → "Too many colons").
- **The one part that was right:** the B65 was genuinely unsupported — its PCI id `e222` was missing from
  `INTEL_ARC`. But that was secondary (the configs targeted the B70 anyway).

## 3. What we changed (the levers we pulled), with measured results

### Code changes (all committed to the branch)

1. **B65 detection** — `sycl/setup_intel.py`: added `e222 → ("Arc Pro B65", 32.0)`. Verified: all three cards now
   detect by name with correct VRAM.
2. **Native host wrapper** — `sycl/serve/strata-sycl-host.sh` (new): runs the SYCL engine natively (no Docker) as
   a drop-in `exe` for the server, mirroring `strata-sycl.sh`'s env handling but exec'ing the binary with host
   paths. Needed because the `strata-sycl-dev` Docker image was never built on this machine and the old native
   wrapper had been lost.
3. **Q4_K_M admission** — `setup.py`: added `MODELS["Q4_K_M"]` (swift family, 3 shards, gguf-dir-only,
   experimental, `pack_args: ["--compat-bf16"]`), named it in `SUPPORTED_GGUFS`, and changed the pack call to read
   **model-level** `pack_args` with a family fallback (it was family-only, so the K-quant's `--compat-bf16` never
   reached `iq_pack.py`).
4. **Per-stage host mirror (the big one)** — `STRATA_SPLIT_MIRROR=1`, opt-in:
   `sycl/src/program/generate.cpp`, `sycl/src/core/gguf_expert_source.cpp`, `sycl/src/core/verify.cpp`,
   `sycl/include/strata/core/verify.hpp`, `sycl/include/strata/kernels/resident_plan_mirror.hpp`,
   `sycl/src/kernels/cuda/verify_kernels.dp.cpp`. On a split, the host mirror used to cover only the *first*
   stage; a later stage whose cache couldn't hold its experts was REFUSED under `STRATA_VERIFY_NO_HOST`. Now the
   mirror is built after all stages' caches exist, covers all layers, and each stage gets its own device-side
   mirror table wired per-verifier (`resident_plan_mirror`) instead of the single global. Two bugs found by
   testing and fixed: `mirror()` **reset** the mirror on each call (only the last stage survived — now appends),
   and the RAM cap was split proportionally (stranded the big later-stage overflow — now each stage offers its
   misses against the *remaining* budget, hottest-first).

### Measured results (all on the pair, this machine)

| Milestone | Config | Result |
|---|---|---|
| WS0: bigger quant serves | IQ3_S, single B70 | 32 tok/s decode, 131K, no crash |
| WS2: dual-GPU pooled | IQ3_S, `--layer-split 24` | **67 tok/s** decode, 93 tok/s cached prompt |
| WS2: 131K under load | IQ3_S split | **115,959-token prompt read at 1,595 tok/s**, answered, no livelock |
| WS5: Q4_K_M fits | `--layer-split 32` (skew) | serves, 18.8 tok/s |
| WS5: Q4_K_M + per-stage mirror | `--layer-split 24` + `STRATA_SPLIT_MIRROR=1` | **29.5 tok/s** (was REFUSED) |
| WS5: + lower VRAM reserve | + `--vram-reserve-mib 1024` | **32.2 tok/s** (mirror 21.3→19.3 GiB) |

### Split sweep (Q4_K_M, mirror on, reserve 1024) — balanced 24 is the optimum

| K | decode | note |
|---|---|---|
| 26 | 30.2 tok/s | drags the weaker B65 |
| **24** | **32.2 tok/s** | balanced — the winner |
| 28 | 26.7 tok/s | overloads the B70's cache into the mirror |

**Canonical configs:** `strata-iq3_s-split.json` (IQ3_S, fast tier) and `strata-swift-q4_k_m-split.json`
(Q4_K_M: K24, mirror on, reserve 1024), both `--port 8081`, selector `level_zero:0,1`.

## 4. The bugs and diagnoses that fell out

- **The 0.1.39 SYCL binary was the whole "q1-only" problem** — a post-load crash, fixed upstream (0.1.40+). The
  fix for the user was a rebuild to 0.1.43.1, not a quant change.
- **Auto-split pathology:** `--layer-split auto` minimizes *decode latency* (SM×GHz-weighted), so on this pair it
  picked a pathological **K=2** (2 layers on the B65, 46 on the B70) that overflowed the B70's cache → REFUSED.
  The search has **no feasibility term** — it never checks the `later_misses > 0` condition it later refuses on.
  Design in [intel-arc-autosplit-fix.md](intel-arc-autosplit-fix.md): add a feasibility gate to the search.
- **The Level Zero selector** for multi-GPU is `level_zero:0,1` (comma-separated ordinals, single prefix) — not
  the doubled `level_zero:1,level_zero:0` that crashed earlier.
- **Mirror cap / append bugs** (above) — found only by running; the per-stage loop kept just the last stage's
  mirror until fixed.
- **Recurring RAM-tight warning** at Q4_K_M size: 75 GB of experts on a 63 GB PC; the 19-21 GiB pinned mirror plus
  the model footprint is tight and bounds how aggressive the mirror can be.

## 5. Where the Q4_K_M speed gap actually is (profiled)

Q4_K_M at 32.2 tok/s vs IQ3_S at 67: the gap is **decode**, and decode is dominated by the **PCIe mirror**, not
the prompt path. Q4_K_M's 62 GB arena leaves ~6,900 experts (19-21 GiB) in the pinned host mirror, read over PCIe
at 12-14 GB/s each verify round; IQ3_S (50 GB) fits VRAM and reads ~0. Porting the MMQ K-quant **prompt** path
would speed prompt *ingest* but not the decode number — so it is not the first lever for the headline speed.

## 6. Levers left to pull (ranked by expected payoff on this hardware)

### Capacity / decode speed (the real Q4_K_M bottleneck)

1. **Fit more experts in VRAM / shrink the mirror.** The head card (B70) carries 24 layers' dense weights + the
   output head + the MTP draft + the whole session KV. Splitting or shedding that load frees its cache. Sub-levers:
   lower `--vram-reserve-mib` further (1024 → 768/512, small risk of the xe over-allocation livelock — measure
   carefully), and a capacity-aware split placement that accounts for the head card's extra load (vs the balanced
   24 that's optimal for equal-residency i-quants).
2. **Faster PCIe mirror reads.** The stager reads mirrored experts one blob at a time; a grouped/coalesced DMA
   read would cut the per-round cost. Lifts any mirror-heavy config, not just Q4_K_M.
3. **The idle B580 as a third card** (12 GB, `bmg-g21`). More total VRAM → smaller mirror → faster decode, and it
   relieves the RAM-tight mirror budget. **But it's a separate project:** the current build is AOT `bmg-g31`-only,
   so this needs a multi-arch build (bmg-g31 + bmg-g21), a 3-way split, and an auto-splitter that won't let the
   much weaker card become the bottleneck (MULTI_GPU.md's own warning). Not a quick win.

### Prompt speed (matters for long-context ingest, not decode)

4. **Port the K-quant MMQ prompt path to SYCL.** The MMQ prompt library is entirely compiled out of the SYCL
   build today (`sycl/CMakeLists.txt` never lists it; `STRATA_PREFILL_MMQ` undefined), so *all* quants prompt via
   the dequant→oneMKL fallback. The pinned llama.cpp MMQ has a **DP4A (non-tensor-core) route** that Arc can run
   (`use_mma_data_layout()` is false on Intel), so no `mma.sync` port is needed. The work: add a `strata_mmq`
   target + instantiation files + two compile definitions, then the K-quant cases in `moe_mmq.dp.cpp`'s
   `supported()`/`run()`. Watch: `ggml_cuda_info().cc` on Arc is a SYCL version, so the MMQ config selector must
   land on a valid DP4A tile (else `J_best=0` aborts). **Opt-in** per house rules — it would speed up *all*
   quants, so the default must stay byte-identical.

### Setup / parity (making it "just work" like CUDA)

5. **Auto-split feasibility gate** (the K=2 fix, option 1 in the design) — reject infeasible placements in the
   search so `auto` never picks a split the engine then refuses. Small, upstreamable, benefits every pair.
6. **Native-path default in setup** — when the `strata-sycl-dev` Docker image isn't built, `setup_intel.py` should
   write the native host wrapper (`strata-sycl-host.sh`) instead of the Docker `exe`, instead of failing with
   `pull access denied`.
7. **Intel multi-GPU offering in setup** — today a two-Arc machine is never *offered* pooling the way NVIDIA is;
   the chooser (`choose_gpus`/`together_ok`) is NVIDIA-shaped (`int(g["arch"])` raises on `arch="xe"`). Wire the
   offer so `--gpus 0,1` / auto is presented.
8. **Images on Intel** (the parity table's "not yet"), and the rest of the parity matrix audit.

## 7. Intel-specific optimizations worth investigating (deeper, from the docs' own roadmap)

Beyond our work, [../INTEL.md](../INTEL.md) and the port's notes map these Arc/Xe2-specific levers, unported or
parked, that could matter on this hardware:

- **XMX (the Arc matrix engines) for the prompt path.** oneMKL FP16 GEMMs already use XMX; the prompt path is
  bound by the *dequant* feeding them, not the products. The parked `xmx_gemm_iq` (fused dequant+GEMM straight
  from quantized rows) and `qsa_prompt_attn_xmx` exist but lose to the current paths on the B70 — worth revisiting
  with the dequant-focused framing.
- **INT8 expert dots on XMX (the parked plan-1).** A decode window fits one INT8 DPAS; today it's scalar `dp4a`,
  ALU-bound (77% XVE active). Three `joint_matrix` versions were 1.4-2-3x slower and one hung the GPU — parked,
  but the ALU-bound profile says the headroom is real if a working XMX integer dot lands.
- **QSA block selection on XMX** (plan-4): every query against every pooled block, a dense product that grows with
  context — the long-context cost we saw (prompt read at scale).
- **Misaligned-load kernel variants.** The 0.1.43 Q6_K alignment fix (two aligned 16-byte loads + a shift) bought
  a large decode win; the docs note IQ4_XS/IQ4_NL/Q8_0 are *still* misaligned (~100-280 GB/s of a 608 GB/s card).
  Applying the same trick to those formats is a measured, known-shape optimization.
- **SIMD16 vs SIMD32 retune.** Xe2's native sub-group is 16-wide; the port runs 32. A prior sweep found SIMD16
  wins alone but not in-engine — worth re-checking after the other kernels land.
- **Fewer graph nodes per decode round** (~2,500 at ~5 µs launch gap): norm+rope, scores+top-k, gate+quantize
  fusion. A structural latency win independent of quant.
- **Wider speculation** (two draft branches per verify window): the kernels are latency-bound, so it's nearly
  free.

## 8. What is *not* the bottleneck (so it isn't chased again)

- K-quant **kernel support**: the Q4_K/Q5_K decode dots and dequant are already in the SYCL port and correct.
  Q4_K_M was never blocked by kernels.
- The **prompt MMQ path**, for *decode* speed (see §5).
- A **quant whitelist** — never existed.

## 9. Reproduce / resume

- Branch `intel-arc-b70-b65` on `github.com/sheigl/Strata`; builds with
  `source /opt/intel/oneapi/setvars.sh && cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx
  -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_AOT=bmg-g31 && cmake --build build-sycl --target strata`.
- Fast tier (IQ3_S): `strata-iq3_s-split.json`. Quality tier (Q4_K_M): `strata-swift-q4_k_m-split.json`
  (needs `STRATA_SPLIT_MIRROR=1`, set in the config's env). Both on port 8081.
- **Hot-swap (llama-swap):** both are registered in `~/.config/llama-swap/config.yaml` as `strata-iq3s` and
  `strata-q4km` (ids `strata:iq3_s` / `strata:q4_k_m`), launched by the local scripts `run-iq3_s-split.sh` /
  `run-swift-q4_k_m-split.sh` (gitignored — machine-local). Both use both Arcs, so they sit in no shared set:
  each runs alone and evicts everything else. llama-swap polls `/health` (503 while the cache fills; the config's
  `healthCheckTimeout: 900` covers the slow first load). Both have `ttl: 0` (never auto-unload; the active one
  stays resident until you swap). Verified working 2026-10-13: a request to `strata-iq3s` on `:8080` loaded the
  engine and answered.
- **Sampling (set 2026-10-13):** both configs carry `"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`
  — Swift-1.5's own `generation_config.json` recommendation (and Strata's documented default sampler). Without the
  block the engine runs *greedy* (temp 0), which on a reasoning model can loop on long thinking chains; the block
  makes plain clients sampled by default. Applied via the server's `sampling_defaults_from_config` (the request's
  own fields still win; an explicit temperature=0 still means greedy).
- **Remote access (Tailscale):** the configs' `allowed_hosts` include `sheigl-ms-7a38.tailc63ae8.ts.net`, so the
  models answer over Tailscale (Strata's DNS-rebinding check otherwise blocks a non-localhost Host header). Add any
  new hostname the same way.
- House rules for any PR off this work: one change per PR, opt-in, default path byte-identical, measured numbers
  with the hardware named, build every backend a file touches.
