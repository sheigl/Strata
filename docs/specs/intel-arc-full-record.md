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
8. **Images on Intel** — done 2026-10-14 on the CPU encoder (§10): `--vision cpu` builds and wires
   `strata-vision` beside the SYCL engine. What is left of that item is a SYCL (GPU) build of the encoder, and the
   rest of the parity matrix audit.

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

## 10. Images on Intel: the CPU vision tower (2026-10-14)

**The premise that needed correcting:** the parity table's "images: not yet" read as if the vision path had to be
ported. It does not. The SYCL engine has the whole read side — `--vision` is parsed in
`sycl/src/program/generate.cpp:1860`, the image M-RoPE table is allocated at start (`:2897`, `(max_context+64) x 3`
int32 per card = 1.6 MB at 131,072), and the `GENI <max_new> <embeddings file> <ids>` request is read and its `SVE1`
embeddings validated (`:8652`, width check `:9606`). What was missing was only the encoder and the setup/config
wiring: `sycl/setup_intel.py` stubbed `hip_vision` to `"none"`, `to_sycl` deleted any `vision` block, and the forced
`--vision none` was appended to setup's argv. So this was plumbing, not a kernel port.

**What changed** (3 files, all opt-in; the default path writes the same config as before):

- `sycl/setup_intel.py`: `sycl_vision_answer()` maps setup's `--vision` to `cpu`/`none` (`yes`/`gpu` answer `none`
  with a warning — there is no SYCL build of the encoder, the same answer the AMD port gives for its missing HIP
  one). `sycl_vision()` calls upstream's own CPU step (`setup.build_vision_cpu`, #304), which configures
  `tools/vision` with `-DSTRATA_VISION_CUDA=OFF` into `build-vision/` and installs the binary beside the SYCL engine
  as `build-sycl/strata-vision`; a symlink of the same name is left in setup's stand-in engine dir so setup's own
  "is the encoder there" checks hold. `to_sycl` now keeps the `vision` block and rewrites only its `exe`. The forced
  `--vision none` is gone, so the default reaches `hip_vision(None) -> "none"`.
- `setup.py`: `MODELS["Q4_K_M"]` gets `"vision": True` plus model-level `mmproj` / `mmproj_hf` keys, read through a
  new `model_mmproj(fam, model)` helper (the house pattern of `model_file`/`model_shards`). It has to be per-model:
  the plain (non-GSQ-RCO) Swift 1.5 release names its encoder `mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf` and
  ships it in its own repo, which is now pinned in `HF_REVISIONS` too. Every other model still resolves its family's
  file (guarded by a test).
- `tools/test_setup_sycl.py`: 5 new tests — the `--vision` answer map, the encoder's location, the Q4_K_M mmproj and
  repository against every other model's family default, the `vision` block keeping
  `mmproj`/`model`/`gpu`/`max_tokens`/`threads` while its `exe` moves, and no block at all when nothing was asked for.
  `tools/test_setup_choices.py`'s two Q4_K_M expectations were stale from the WS5 admission (Q4_K_M is a setup size
  now) and are updated.

**Measured** (this machine: B65 `e222` + B70 `e223`, i7-11700K, 62 GB, engine 0.1.43.1-sycl with IQ3_S loaded and
answering — the CPU was not idle; encoder `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 907,543,008 bytes, size matched
against the pinned HF revision):

| picture | becomes | encode | encoder RSS |
|---|---|---|---|
| 512x512, 300-token cap | 256 tok (16x16) | 2.4 s | 1.07 GiB |
| 1024x1024, 300-token cap | 289 (17x17) | 3.0 s at 8 threads, 2.7 s at 16 | 1.10 GiB |
| 1024x1024, 1,024-token cap | 1,024 (32x32) | 22.3 s at 8 threads, 18.2 s at 16 | 2.19 GiB |

`READY 2560` prints 0.7 s after start with IQ3_S's shard 1 as `--model` (3.2 s with the Q4_K_M shard) — metadata only,
no 54 GB load. The embeddings file is the `SVE1` record (`n x 2560` float32 + 20-byte header): 2.6 MB measured for the
256-token picture. CPU encode costs **no VRAM**, so the tuned `--vram-reserve-mib` (1,024 MiB on the
Q4_K_M config) is unchanged; the engine's only image cost is the 1.6 MB M-RoPE table. The 300-token
default is setup's (`VISION["cpu"]`); mtmd wants >= 1,024 for Qwen-VL grounding, which is the 22 s row.

**Not tested:** one image end to end through the API. The encoder side is measured; the engine's `GENI` read needs a
model start with `--vision`, and the model serving this session is the one under test, so activation (a reload) is
left to the operator. Two known gaps left open deliberately: the embeddings land in the system temp dir, which the
`strata-sycl-dev` container does not share with the host (the native host path works; wiring that folder under
`/work` would mean touching shared `serve/server.py`), and there is no SYCL build of the encoder — the idle B580
(`e20b`, bmg-g21) as an encode card stays the deferred option in §6.

**On this machine** (both gitignored split configs now carry the block, verified before writing):

```json
"vision": {"exe": "/mnt/1TB/AI/Strata/build-sycl/strata-vision",
           "mmproj": "<the model's mmproj>", "model": "<the model's shard 1>",
           "gpu": false, "max_tokens": 300, "threads": 8}
```

`strata-iq3_s-split.json` points at `mmproj-Qwen3.8-Flash-Next-BF16.gguf` + IQ3_S shard 1 (that mmproj was downloaded
from the pinned GSQ-RCO revision: 907,543,008 bytes, exactly as Hugging Face lists it);
`strata-swift-q4_k_m-split.json` at `mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf` + Q4_K_M shard 1. Both commands as
the server builds them (`serve.engine_args`, `VisionEncoder`) were run by hand from a `/tmp/strata-vision-*` work dir
with a 512x512 PNG: each printed `READY 2560` and then `OK 256 16 16` with a 2,621,460-byte embeddings file (Q4_K_M:
3.2 s end to end, 1.04 GiB peak RSS). `engine_args` adds `--vision` to the engine itself, and does **not** touch
`--vram-reserve-mib` for `"gpu": false`, so the tuned reserve survives. A wrong path here would stop the server at
boot (`Vision._start` raises and nothing catches it), which is why the command was run standalone first.

## 11. The third card: the B580 as expert cache (2026-10-15)

### The ask, and what it turned out to be

The B580 (12 GB) was idle. The stated reason for wanting it in the pool was **system RAM**: the Q4_K_M pair at 131K
keeps a 19.44 GiB **pinned** host mirror. `free` puts it in `Buffers`, not in the engine's RSS (`VmRSS 2.2 GiB`,
`VmLck 0`, `Buffers 16.9 GB` while it is built), so it is invisible to `ps`/`htop` but unswappable all the same:
`MemAvailable` drops by the mirror's size. That is the memory the user is losing.

Profiling the pair's decode made the second half of the case: in the pair the **last** stage (the B70 - head, MTP
drafter and 8,873 of its 12,288 pairs) makes a host round trip inside every verify window, while a stage whose own
layers are fully resident runs a `100% VRAM resident: zero-doorbell graph`. So a stage that holds only what it covers
is worth more than its layer count suggests, and a card with a slow link should never hold overflow.

### The measurements (same build, same day, same prompt)

All three rows: Q4_K_M (`swift-q4_k_m` pack), `--max-context 131072`, `--vram-reserve-mib 1024`, `--prefill 4096`,
`--kv int8 --kv-resident 32768`, `--spec 4`, `STRATA_SPLIT_MIRROR=1`, `STRATA_MIRROR_MIB 51200`, engine
`0.1.43.1-sycl`, one 500-token reasoning answer at temperature 0 after a 68-token prompt.

| layout | layers B65/B70/B580 | held pairs | mirror (build time) | start | decode | drafts accepted |
|---|---|---|---|---|---|---|
| pair | 24 / 24 / - | 18,232 | 19.44 GiB (170.8 s) | 550 s | 18.1 tok/s | 309/382 |
| three | 22 / 24 / 2 | 19,822 | 14.51 GiB (99.7 s) | 280 s | 27.0 tok/s | 308/390 |
| three | **18 / 25 / 5** | 21,301 | **9.78 GiB** (85.7 s) | 480 s | 26.3 tok/s | 316/416 |

`held pairs` is read off the per-stage lines (`N of its M profiled pairs`); the mirror is the engine's own
`STRATA_SPLIT_MIRROR: ... (X GiB, Y s)` line. The three-card rows are 45% faster on tokens per second and the last one
halves the pinned RAM. The pair's 18.1 tok/s is *this* prompt: an earlier record on the same config (section 3) says
29.5-32.2 tok/s, which was a different, shorter answer whose tokens route more inside `data/expert-profile-sycl.bin`
(the profile is built from chat, code, maths and summaries). The two three-card rows exist to be compared with the
pair row, not with that one - they are one build, one day, one prompt.

### What was wrong in the engine

1. **The measured coverage curve was gated to `ns >= 4`.** A 3-stage split fell back to the 0.1.39 power law
   `(r+1)^-1.2`, which on this profile claimed 10,914 of 24,576 pairs held 97.5% of the routed mass. `auto` therefore
   picked `K=2,47`: B65 2 layers (27.22 GiB free, 3.42 GiB used), B70 45 layers, B580 1 layer + the head. 14,286
   experts unheld, mirror **43.36 GiB**, built in **561.6 s** (77 MiB/s), and the server declared `EngineDied: the
   engine did not report READY within 900 s`. With `STRATA_SPLIT_COVER_B=3` the same search picked `K=15,41`
   (predicted 53.x ms), still giving the B580 7 layers it cannot cover. Gate is now `ns >= 3`; `ns == 2` is untouched.
2. **The SYCL port never got the CUDA engine's per-stage link pricing.** `src/program/generate.cpp` (lines ~3716,
   3814) probes each stage's link inside the search, scales that stage's miss cost by `clamp(20.0 / bw, 1, 64)`, and
   accumulates the missing routed mass **per stage**; the port had one global `held_mass` and one `miss_ms`. Ported
   in shape. Logged on this machine: `CUDA0 link 13.8 GB/s -> miss cost x1.45`, `CUDA1 14.0 -> x1.43`,
   `CUDA2 1.7 -> x11.67`. The port's probes read a *pinned* source, which is its own stage-probe convention since
   0.1.41, where the twin reads a pageable one - only the absolute numbers differ, not the ratios. On equal links the
   outcome is byte-identical, so a 2-GPU placement does not move. The pick it changes is measurable: with the coverage
   curve on and this missing, `auto` chose `K=15,41` (the B580's 7 layers, 3,584 pairs, against its 2,560-slot
   cache); with it, `K=18,43` - the B580's 5 layers, 2,560 of 2,560.
3. **`sycl/setup_intel.py` never wrote `STRATA_SPLIT_MIRROR`.** Q4_K_M on any split is REFUSED without it
   (`STRATA_VERIFY_NO_HOST=1` on xe), and the live configs only worked because the key had been added by hand - a
   setup rerun would have dropped it. `split_env()` writes it (and `STRATA_STAGE_TRIM`) for every split;
   `STRATA_MIRROR_MIB` is left to the engine.

With 1 and 2 in place and **no environment variable**, `auto` now picks `K=18,43` - the measured-best shape, found by
the engine instead of by hand.

### Why B580 = 5 layers, and the shape rule

B580 VRAM 11.93 GiB; after the dense weights, its session `[43,48)` and the head, 8.54 GiB is left for experts, which
is exactly 2,560 pairs = 5 layers × 512. It holds `2560 of its 2560` = 100%, so **no miss ever touches its 1.71 GB/s
link**. Its 160 CU at 2.85 GHz make its five layers cheap (1.27 ms each, measured by the search). Give it a sixth
layer and the overflow moves onto the worst link in the machine; the mirror shrinks by nothing, because a pair is
mirrored whether its own stage holds it or not.

The B65 covers 18 of its own layers completely (9,216 slots / 9,216 pairs); the B70 takes the remaining 25 and covers
9,518 of 12,800 = 74%, and its 3,282 missing pairs are the 9.78 GiB mirror. Total held 21,301 of 24,576, the ceiling
for these three cards at this context.

### Open

- `auto` leaves the first stage **2.16 GiB less** free VRAM than an explicit `--layer-split` with the same K
  (`expert cache auto: 27.94 GiB free -> 8070 slots` vs `30.10 -> 8718`; the search ran and the explicit path did
  not), so its mirror is 11.27 GiB against 9.78. Reproduced twice, identical numbers. Something between the stage
  setup and the search's `cap[]` loop holds VRAM on the first device; not yet found.
- The search's absolute prediction is still far off the measurement (55.9 ms predicted, ~154 ms measured), although
  its *ranking* is now right. The obvious next model: the last stage's misses cost more than the middle stages', since
  nothing is behind them to hide the latency.
- 135K / 200K context on three cards, and IQ3_S on three cards (its per-stage mirror should not even be needed), are
  unmeasured. The B580 as a *vision encode card* (section 10) is still unbuilt.
