# Spec: Intel Arc support at CUDA-level quality, dual-GPU pooled (B70 + B65)

Status: **draft, 2026-10-12**, written against the confirmed intent at
[docs/intent/intel-arc-dual-gpu.md](../intent/intel-arc-dual-gpu.md). Read that first; it corrects the premise
(this is elevation of an existing SYCL port, not a greenfield port).

House rules for every change this spec produces ([AGENTS.md](../../AGENTS.md)): one change per PR; new behavior
opt-in with the default path byte-identical to the last release; every number measured, with the hardware named;
build every backend a file touches.

## Objective and completion criteria

Make the Intel Arc path meet the same bar as the CUDA path on the user's machine, where "the bar" is all three
of parity, speed and setup. The work is done when **all** of the gates below hold, measured on the user's dual-Arc
Linux box (B70 + B65) and reported per [docs/COMMUNITY_BENCHMARKS.md](../COMMUNITY_BENCHMARKS.md):

1. **Parity gate:** every model variant and quant that setup offers on CUDA is offered, loads, and produces
   correct output on Intel. Correct = the kernel parity tests pass on-card and a 10-prompt served-output check
   matches the reference the way [INTEL_ARC.md](../INTEL_ARC.md) describes for the maintainer gate.
2. **The quant-load fix:** the user-reported failure — non-q1 quants (iq3_xxs class) never loading on a single
   B70 while a 12 GB RTX 3060 runs them — is root-caused and fixed. "Fixed" means those quants load and run on
   one B70, before any dual-GPU work is claimed.
3. **Headline gate (the user's words, confirmed literal):** the pooled B70+B65 pair runs **literal Q4_K_M**
   (the user's Swift-1.5 Q4_K_M GGUF, 113 GB) — not merely an i-quant tier — correctly, at a stated decode tok/s,
   with **a context of at least 131,072 tokens**. Validation: a real prompt of ~104K tokens (79% of window, the
   pattern INTEL.md used) read and answered coherently, decode afterwards measured and stated, no host livelock.
   **This requires WS5 (K-quant support on the SYCL port): the K-quant prompt kernels are CUDA-only today.**
4. **Setup gate:** `./setup.sh --backend sycl` (or plain `./setup.sh` detecting the Arcs) on a fresh Ubuntu
   24.04 install detects both cards, offers the pooled configuration the way MULTI_GPU.md's menu does for
   NVIDIA, picks a valid model and context, and reaches a serving state with no manual flags.
5. **Docs gate:** MULTI_GPU.md no longer lists Intel under "Not supported"; INTEL_ARC.md's experimental warning
   is revisited; every claim in the changed docs carries a measurement and the machine it was measured on.

A spec that meets gates 1-2 but not 3-5 is a useful interim state, not completion.

## Verified baseline (what exists today, from the repo)

Read alongside [INTEL.md](../INTEL.md) and [INTEL_ARC.md](../INTEL_ARC.md); do not re-derive these.

- The SYCL/oneAPI port (`sycl/`) builds the full engine: 157 targets, kernel parity tests, AOT for `bmg-g31`
  (B70) and `bmg-g21` (B580/B570/B60).
- Single B70 measured: Coder IQ1_M 70-78 tok/s decode, ~790-1,000 tok/s prompt; 131K context is the measured
  single-card ceiling (1.5 GB headroom); 262K livelocked the host on the `xe` driver (allocation past VRAM does
  not fail, it evicts until the kernel dies — this failure mode is load-bearing for gate 3's "no livelock").
- Dual Arc is community-reported, not maintainer-validated: 2x B70 `--layer-split` ran Flash-Next IQ3_XXS at 66
  tok/s decode (#423); 2x B60 ran Coder IQ1_M at 56.6 tok/s. The `stage_room` fix those reports needed is in
  0.1.39+.
- MULTI_GPU.md currently says Intel multi-GPU is "Not supported" — stale, but it tells you setup never offers
  it and nobody owns it.
- User's report (2026-10): on their B70, quants beyond q1-class never loaded. Reproduce first; the gap list
  below assumes this is real until proven otherwise.
- Open: the B65's VRAM size to confirm (32 GB-class; the machine probe at WS0 identified it as PCI `e222`,
  `bmg-g31`, Level Zero index 0) — see the intent doc's machine-probe section. A third card (B580 12 GB, G21)
  is present and must be excludable from the pooled pair.

## Workstreams

Ordered; each is separately PR-able per house rules.

### WS0 — Reproduce and root-cause the quant-load failure — **DONE (2026-10-13)**

**Resolved: [intel-arc-ws0-rootcause.md](intel-arc-ws0-rootcause.md).** No quant whitelist exists. The lived
failure was a **post-load startup crash** (`verify: layer 0 never rang` + UR error 37) on the stale **0.1.39**
binary plus a malformed multi-GPU selector. Rebuilt the engine to **0.1.43.1** and **IQ3_S now loads and serves
on the B70** (decode 35.6 tok/s at 131K context). Also fixed: `e222`/B65 detection (verified); recreated the
missing native `strata-sycl-host.sh`; resolved the split selector (`level_zero:0,1` = B65+B70). Remaining WS1/WS2
follow-ups are listed in that doc.

### WS1 — Parity: one Intel model/quant/feature matrix

- Enumerate what setup offers on CUDA (variants, quants, vision, MTP, KV modes) and make the Intel path offer
  the same set, minus only entries with a *measured, documented* reason.
- Wire the missing pieces the parity table already flags (images on Intel are "not yet"; audit the rest).
- Extend `tools/test_setup_sycl.py`-style setup tests so the matrix stays honest without a GPU.
- Acceptance: gate 1.

### WS2 — Dual-GPU pooling — **WORKING (2026-10-13, first split up)**

B65+B70 layer-split validated on the user's machine. Config `strata-iq3_s-split.json`: `--layer-split 24
--split-device 1`, `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`, `STRATA_STAGE_TRIM=1`, IQ3_S, 131K context.

**Measured (warmed, 3x 256-token runs):** decode **67.0 tok/s** (vs 32.0 single B70 — **2.1x**), prompt read
**93 tok/s** cached (vs 58), draft acceptance ~80% (vs 71%). Matches the community 2xB70 report (~66 tok/s on
the smaller IQ3_XXS).

**Two findings for the workstream:**

1. **The `auto` split is broken for this pair.** `auto` saw the B65 as 20 SMs @ 2.40 GHz vs the B70's 32 SMs @
   2.80 GHz, and chose a pathological **K=2** (2 layers on the B65, 46 on the B70). With that skew, 10,790 of
   the later stage's experts fit neither in VRAM nor the host mirror (the mirror covers only the first stage
   under `STRATA_VERIFY_NO_HOST`), and the engine **refused** to start. Forcing `--layer-split 24` (balanced)
   worked. **Action item:** for the two-32-GB-card case, `auto`'s SM-weighted placement must not skew past the
   point where the bigger card can't hold its experts; or setup should default two-equal-VRAM cards to a balanced
   split.
2. **`STRATA_STAGE_TRIM=1` works** (CUDA1 trimmed to its own layers' dense weights) but alone doesn't fix the
   `auto` skew — the placement, not just the trim, decides whether the later stage's experts fit.

**Card asymmetry noted:** B65 = 20 SMs @ 2.40 GHz, B70 = 32 SMs @ 2.80 GHz (the B70 is the stronger card, so it
correctly took the head + MTP + more layers under any sane split). Device order under `level_zero:0,1`:
device 0 = B65 (layers 0-23), device 1 = B70 (24-47 + head).

**131K context validated under real load (2026-10-13, on the pooled pair):**

| prompt tokens | prompt read | decode after | result |
|---|---|---|---|
| 58,867 | **1,725 tok/s** (34 s) | 56.5 tok/s | coherent, no issue |
| **115,959** (88% of window) | **1,595 tok/s** (73.5 s) | 51.2 tok/s | **correct answer, no livelock/watchdog** |

This is the gate that livelocked a single B70 at 262K (INTEL.md's measured table). The pooled pair holds 131K
comfortably — expert residency doubled, prompt reading *faster* at scale (batching), decode stays >50 tok/s.
**Gate 3's context criterion is met.** (A ~104K-token request overshooting the window returns a clean HTTP 400,
not a crash — the limit is enforced safely.)

- `--layer-split` works on Arc today but is unowned: validate it on the B70+B65 pair, fix what breaks
  (candidate knowns from the docs: `STRATA_MIRROR_MIB=0` semantics on a split, the "N experts neither in VRAM
  nor mirrored" line counting the other card's layers, per-card AOT when the two cards differ).
- Setup detects multiple Arcs and offers the pooled configuration exactly like the NVIDIA flow (`--gpus`,
  `auto` split-point placement, card-order rules).
- KV streaming (`--kv-resident`) and expert mirroring validated on the split; 131K context is the floor, and
  the run must not livelock (respect the measured VRAM tables; leave ≥1.5 GB free per card).
- Acceptance: gates 2-3.

### WS3 — Speed to the parity bar

No blanket "make it faster": measure the pooled pair on the headline configuration, compare against both the
single-B70 numbers and the 3060 reference, and close the gap where it's embarrassing. Candidate levers already
mapped in INTEL.md: QSA block selection on XMX, INT8 prompt GEMMs, fewer graph nodes per decode round, wider
speculation. Each is its own measured PR. Acceptance: headline-gate decode tok/s stated and defensible.

### WS4 — Setup and docs

- One-click setup flow for dual Arc (gate 4), including non-Docker and Docker paths both documented.
- MULTI_GPU.md "Not supported" line removed with the validating numbers attached; INTEL_ARC.md experimental
  status revisited; INTEL.md updated with dual-card measurements.
- A community benchmark report per [docs/COMMUNITY_BENCHMARKS.md](../COMMUNITY_BENCHMARKS.md) from the user's
  machine — this is hardware the maintainers do not all have, and the report format exists for exactly this.

### WS5 — Literal Q4_K_M (K-quant) support on the SYCL port

Added 2026-10-12 after the user confirmed the headline gate is *literal* Q4_K_M, not an i-quant tier. Today
`SUPPORTED_GGUFS` (setup.py:1579) refuses plain K-quants, and the K-quant prompt kernels (`STRATA_MMQ_KQUANTS`)
are CUDA-only (`UD-Q4_K_XL` is `nvidia_only`). Scope: bring K-quant expert/prompt kernels to the SYCL port and
admit Swift-1.5 Q4_K_M in setup. This is real engine work, sequenced **after** WS0-WS2 prove the base port on the
pair; sizing it honestly is part of this workstream, not a promise made here. Acceptance: gate 3.

**Scoped 2026-10-13 (static, against the source).** ~~Smaller than feared~~ **Revised after a deep code read
(subagent, evidence file:line): the premise "add the three MMQ cases" was stale.** Findings:

- **The SYCL build does not compile `moe_mmq.dp.cpp` AT ALL** — for *any* quant. `sycl/CMakeLists.txt` builds the
  prompt path from `gemm.cu/kernels.cu/prefill.cpp` + `moe_fused_stub.cpp` + `xmx_moe.dp.cpp`; it never lists the
  MMQ TUs, and `STRATA_PREFILL_MMQ` is never defined, so `prefill.cpp` compiles the **stub** (`mmq::built()`→false).
  The on-disk `sycl/src/prefill/moe_mmq.dp.cpp` is dead code (`migrate.sh` skips it). No MMQ of any kind runs on
  SYCL today; the i-quants work via a *separate* XMX grouped-GEMM + dequant→oneMKL path.
- **Q4_K_M runs CORRECTLY today** (if admitted past setup) via the fallback: prompt path `mmq_plan` sees
  `built()==false` → `fallback=true` → dequant-to-FP16 + oneMKL GEMM; decode uses the existing K-quant VMMQ dots
  (`vec_dot_q4_K_q8_1`), which pass the `is_iq()` gate (Q4_K/Q5_K included unconditionally). No hard error for
  Q4_K_M. The **only** refusal is at setup (`SUPPORTED_GGUFS` / `gguf_unsupported`), before the engine.

**So WS5 splits into "make it work" (small) vs "make it fast" (large):**

1. **Make Q4_K_M work (small, correctness):** admit Q4_K_M in `setup.py` (`MODELS` + `SUPPORTED_GGUFS`, keyed to
   the sycl backend, not `nvidia_only`). The engine already runs it correctly via the dequant+decode-dots
   fallback. Verify with a real load + a served prompt on the pair. **This alone may satisfy gate 3** — the gate
   says "runs correctly at a stated decode tok/s," not "fast."
2. **Make Q4_K_M fast (large, opt-in):** wire the whole MMQ library into the SYCL build (`strata_mmq` target,
   `-DSTRATA_PREFILL_MMQ` + `-DSTRATA_MMQ_KQUANTS`, the per-type instantiation files, the quantizer, host-symbol
   shim), then add the K-quant cases to `moe_mmq.dp.cpp`'s `supported()`/`run()`. The pinned llama.cpp MMQ has a
   **DP4A (non-tensor-core) route** that Arc can run (`use_mma_data_layout()` is false on Intel), so no mma.sync
   port is needed. Risk to watch: `ggml_cuda_info().cc` on Arc is a SYCL version, so the MMQ config selector must
   land on a valid DP4A tile config (else `J_best=0` abort). **Opt-in per house rules** — it would also speed up
   Q2_0/i-quants, so it must not change the default path.

**Decision point for the user:** try "make it work" (step 1) first and measure the actual Q4_K_M speed on the
pair before committing to the MMQ build-wiring (step 2). The decode dots being already present means Q4_K_M decode
might be acceptable without the MMQ prompt path.

**Tried 2026-10-13 (make-it-work), and it surfaces the REAL blocker.** Q4_K_M admitted to setup (MODELS +
SUPPORTED_GGUFS + model-level `pack_args: ["--compat-bf16"]`, and setup.py's pack call changed to
`MODELS[model].get("pack_args", fam.get(...))` since pack_args was family-only). The pack builds fine. **The
engine accepts the K-quant and runs the load** (`token embedding Q4_K in mapped host memory`, `native Q5_K head`,
expert cache fills) — the decode-dot premise holds. **But it REFUSES to start for a memory reason, on both one
card and the pair:**

- Single B70 (131K): 15,681 of 16,721 non-resident experts mirrored (47.8 GiB, the RAM pinning cap), 1,040 left
  with nowhere to live → REFUSED.
- Pair split (131K): 9,032 + 8,602 resident; 3,256 mirrored; **3,686 of the LATER stage (B70, layers 24-47) fit
  neither in its VRAM (26.2 GiB cache holds 8,602 of 12,288) nor the mirror** → REFUSED.
- Pair split at **65,536** context: still **3,665** unhomed (barely moved) — **the shortfall is NOT KV/context.**
  The cause is structural: the B70 stage holds 46 layers' worth of dense weights + head + MTP AND the whole
  session KV, so its expert cache is squeezed, and the host mirror only covers the FIRST stage (B65), so the
  squeezed stage's overflow has no fallback.

**The actual WS5 work to make Q4_K_M serve on this pair** (none of it is the K-quant kernels — those are fine):

1. **Split the KV / session state across the cards** (or `--kv-resident` harder) so the B70 stage isn't carrying
   the whole context's KV on top of its layers — freeing its expert cache.
2. **Extend the host mirror to cover later split stages** (today it covers only the first stage under
   `STRATA_VERIFY_NO_HOST`), so a later stage's overflow can be mirrored. This is the engine's own suggested fix
   direction.
3. **A VRAM-weighted or capacity-aware split** for unequal-VRAM-role pairs (the head/MTP card needs more room),
   rather than the balanced `--layer-split 24` that's right for equal-residency i-quants but starves the
   head-card on a 62 GB model.

Until one of those lands, **IQ3_S is the quality ceiling that runs on the pair today** (50 GB arena fits; 67
tok/s, 131K validated); Q4_K_M (62 GB) is just over the line. This is the honest WS5 status: K-quant *support*
works; K-quant *capacity planning on a split* is the open work.

**Capacity-fix sizing (2026-10-13, from `sycl/src/program/generate.cpp`):**

- The limitation is explicit in code (`#1980`, lines 4970-4987): the host mirror and the REFUSED count cover the
  **first stage only**. Under `STRATA_VERIFY_NO_HOST` (required on xe), a later split stage whose cache can't
  hold all its layers' experts has **no host fallback** → REFUSED with the exact message we hit.
- `later_misses` (line 4979) = Σ over later stages of `(le-lb)*n_expert − cache.slots`. For Q4_K_M at
  `--layer-split 24`: the B70 stage is layers 24-47 (24 layers × 512 = 12,288 pairs) minus its 8,602-slot cache =
  **3,686 misses**, all unhomed. Its cache is squeezed because the head card carries the dense weights of 24
  layers + the output head + the MTP draft + the **entire session KV** on top of its experts.
- **Fix ranking:**
  - **(1) Extend the host mirror to later stages — the real fix.** Per-stage mirrors so a later stage's overflow
    is served (slowly) instead of refused. Medium engine change (allocate + verify-path per stage); directly
    removes the REFUSED. This is the general solution and matches the engine's own comment.
  - **(2) Capacity-aware split placement — cheap to try, uncertain.** Skew layers onto the B65 (higher K) so the
    B70's 12,288 pairs fit its cache. Empirically slow to test (each fill is ~5 min) and the gap is large; a
    balanced 24 is wrong for a 62 GB model but the needed skew may be extreme. Worth one or two more measured
    attempts (K=32/36) before dismissing, but not the structural fix.
  - **(3) Split the session KV across cards / lean harder on `--kv-resident` — largest.** Moves the head card's
    biggest non-expert load off it. Architectural; highest risk.
- **Recommendation:** Fix 1 is the durable answer and the upstreamable one; Fix 2 is a same-day experiment to
  confirm the pair *can* hold Q4_K_M at all. Effort on Fix 1 is days of careful verify-path work, not weeks.

## Explicit non-goals

- Windows / WSL2 support (INTEL_ARC.md's Windows section stays accurate).
- Model fine-tuning or training of any kind.
- Multi-instance throughput batching across the two cards (`--batch` on a split is its own upstream topic).
- Tensor parallelism or any pooling deeper than the existing layer-split architecture, unless WS2 measures say
  layer-split cannot meet gate 3 (then renegotiate before building).
- Changes to CUDA/HIP behavior beyond what the parity matrix strictly requires.

## Risks and open questions

- **The quant-load failure may be deep.** If WS0 finds an engine-level limitation rather than a setup/planner
  bug, scope and timeline change. WS0 reports before anything else proceeds.
- **The xe driver's allocation-past-VRAM livelock** is the way gate 3 fails catastrophically. Every dual-card
  context configuration must be sized from measured VRAM tables, not computed optimism.
- **B65/B70/B580 identity:** resolved by the WS0 machine probe (see the intent doc): B65 = `e222`, B70 = `e223`,
  both `bmg-g31`; B580 = `e20b`, `bmg-g21`. **None of these PCI ids are in Strata's known list** (INTEL_ARC.md
  names only B60 `e211`/`e221`) — `sycl/setup_intel.py` detection is the first thing WS0 checks in code, since
  failed detection is a plausible root cause of the user's "quants never loaded" report.
- **Upstream appetite:** before the first PR, search open issues/PRs and TEST_REQUESTS.md; the maintainers
  compile and kernel-test the port but (per INTEL_ARC.md) have limited Intel hardware, so community reports
  carry extra weight. The fork remains the fallback vehicle, not the plan.
- **Which model variant** is the q4_k_m gate target (base / Coder / Swift / Unsloth) — pick with the user at
  WS0 time; the intent names the family, not the variant.

## How to resume in a new session

1. [docs/intent/intel-arc-dual-gpu.md](../intent/intel-arc-dual-gpu.md) — the confirmed intent.
2. This file — scope, gates, workstreams.
3. Git state — which workstream is in flight.
