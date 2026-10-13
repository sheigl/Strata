# WS0: root-cause — why non-q1 quants never loaded on the user's Arc machine

Status: **static analysis complete, 2026-10-12.** Read with [../intent/intel-arc-dual-gpu.md](../intent/intel-arc-dual-gpu.md)
and [intel-arc-dual-gpu.md](intel-arc-dual-gpu.md) (the spec). This is the WS0 deliverable: what is actually wrong,
found by reading the setup/engine path against the user's machine. **No runtime reproduction yet** (the user chose
static-first); the runtime confirms the primary suspect below.

## The user's report

On this machine (B65 + B70 + B580), Strata on Intel never loaded anything beyond the q1-class quant; the same
iq3_xxs runs fine on their 12 GB RTX 3060. The working assumption going in was "a whitelist says only q1 works on
Intel." That assumption is **wrong**, and correcting it is the first result.

## What the code actually does (REVISED 2026-10-12, after the logs surfaced)

There is **no quant whitelist for Intel**, and — correcting the earlier framing — **the quant load is not the
failure.** The user's old configs and logs (`strata-iq3_s.json`, `strata-iq3_s.log`, engine 0.1.39-sycl) show
**IQ3_S fully loaded on the B70**: 12,513 experts cached to VRAM, "session is up", MTP draft loaded, 131,072
context configured. The run then died **after** load, at the first verify:

```
strata serve: verify: layer 0 never rang (graph finished)
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  level_zero backend failed with error: 37 (UR_RESULT_ERROR_UNINITIALIZED)
```

And a separate dual-GPU attempt shows the Level Zero selector being built wrong:

```
Error parsing selector string "level_zero:1,level_zero:0"  Too many colons (:)
```

So the user's lived experience "bigger quants never worked" is real, but the mechanism is **a startup crash after
load + a broken multi-GPU selector**, not "quants refused at load." The `e222` detection bug (below) is real and
worth fixing, but it is **not** the primary cause — the configs already targeted `gpu: 1` (the recognized B70).

## Defect 0 (actual primary): the 0.1.39-sycl binary is stale; the crash fix is likely upstream

The installed engine reports **0.1.39-sycl**; the source tree is **~50+ PRs ahead** (main is at #2038/#2055/#2088/
#2090). Per INTEL_ARC.md, the 0.1.39 SYCL port **"did not compile against 0.1.39's own engine sources (#784) and
its ring waits never saw a slow layer as still running (#866, #867); both fixed in 0.1.40"**, and *"no one has run
an unpatched 0.1.39 port on an Arc."* The `verify: layer 0 never rang` + `UR_RESULT_ERROR_UNINITIALIZED` signature
is exactly the class of bug those fixes address. **The first runtime step is therefore: rebuild the SYCL engine
from current source and rerun IQ3_S** — the crash may already be fixed upstream. This is a *hypothesis to confirm
by rebuild+rerun*, not yet proven.

## Defect 1: the B65's PCI id is unknown to Strata (real, but not the primary cause)

`setup_intel.py:44`, the `INTEL_ARC` table, **lacked `e222` (the B65)**. **Fixed 2026-10-12** and verified:
`intel_gpus()` now returns all three cards by name (B65 32 GB, B70 32 GB, B580 12 GB). Impact before the fix:
the B65 (card0) read as `Intel GPU e222 (xe)` with a VRAM fallback that can return 0.0, so a config that defaulted
to card0 would be mis-sized. The user's saved configs used `gpu: 1` (the B70), so this was not their crash — but
it breaks any default/B65 path and must stay fixed.

## Defect 2: the multi-GPU Level Zero selector is malformed (blocks pooling) — RESOLVED (syntax)

`setup_intel.py:292` correctly uses `level_zero:*` for a split, but the user's log shows a selector of
`level_zero:1,level_zero:0` — which the UR rejects (`Too many colons`). **Probed on this machine (Level Zero V2,
1.17.39395, 2026-10-12):** `level_zero:0,1` selects exactly the B65+B70 pair (excludes the B580); `level_zero:*`
selects all three. The old failure was a doubled `level_zero:` prefix (hand edit or older code), not a current
limitation. **For WS2:** to pool *only* the two 32 GB cards and leave the 12 GB B580 out, setup should emit
`level_zero:0,1` (comma-separated device ordinals, single prefix), and the engine's layer-split must address the
cards within that filtered set. Confirm the engine's `--split-device` indexing maps to the *filtered* Level Zero
order, not raw sysfs order.

## Defect 3: stale/again-mismatched docs and configs pointing at moved paths

- MULTI_GPU.md "Not supported: Intel GPUs" — stale given the split code in setup_intel.py; suppresses attempts.
- The saved configs reference `/mnt/1TB/AI/Strata-data/...`, **which no longer exists** (models moved to
  `chat_models`). Any rerun must repoint paths (or re-run setup), or it will fail on missing files regardless of
  the engine.
- INTEL.md's single-B70 context table (131K ceiling, 262K livelock) is the binding constraint for the headline
  131K-pooled gate and is not yet wired into any guard.

## Root cause (REVISED, one line)

The user's "Intel only runs q1" is **two layered things**: (a) an **outdated 0.1.39-sycl engine that crashes after
load** at the first verify — a bug class fixed upstream in 0.1.40+ — and (b) a **malformed multi-GPU Level Zero
selector** that blocks the two-card pooled run. The `e222` detection gap is real but secondary. There is no quant
whitelist and quants do load.

## What WS0 hands to WS1/WS2 (the fix list, updated)

1. **Rebuild the SYCL engine from current source** (AOT `bmg-g31`) and rerun IQ3_S on the B70 — confirm whether
   the post-load crash is already fixed upstream. *(first runtime step)*
2. Fix the multi-GPU `ONEAPI_DEVICE_SELECTOR` for this Level Zero version (WS2's pooling blocker).
3. Keep the `e222 → ("Arc Pro B65", 32.0)` `INTEL_ARC` fix (done); make the VRAM BAR fallback warn on 0.
4. Repoint/re-run setup so configs reference existing model paths.
5. Correct MULTI_GPU.md's "Not supported" line once 2 lands, with numbers.

## Verification status

- [x] Machine probed (three Arcs identified, AOT targets, oneAPI 2025.3).
- [x] Setup/offer path read end to end; no quant whitelist found.
- [x] Primary root cause identified in code (`e222` missing from `INTEL_ARC`).
- [x] **Local model checked (2026-10-12):** the user's `/mnt/1TB/AI/chat_models/ukisai/Swift-1.5-...-GGUF/Q4_K_M`
  (3 shards, 113 GB) **cannot be used by Strata** — `SUPPORTED_GGUFS` (setup.py:1579) accepts only GSQ-RCO
  i-quants and Unsloth's UD-Q4_K_XL / UD-IQ4_XS; plain K-quants like Q4_K_M are explicitly refused. So this file
  is no shortcut to a runtime test, and it re-frames the headline gate (below).
- [x] **Detection fix verified (2026-10-12):** adding `e222 → ("Arc Pro B65", 32.0)` to `INTEL_ARC` makes
  `intel_gpus()` return all three cards by name with correct VRAM (B65 32 GB, B70 32 GB, B580 12 GB). Before the
  fix the B65 read as `Intel GPU e222 (xe)` with vram from a sysfs BAR fallback that can return 0.0 — and being
  card0 it was the default selection, so configs were sized against a mis-identified card. Root cause confirmed
  at the detection layer.
- [x] **Runtime load+serve CONFIRMED (2026-10-13):** IQ3_S on the B70 (`level_zero:1`, engine 0.1.43.1, native
  host path) loaded — 12,673 experts cached in 24 GiB VRAM, 13,103 mirrored in pinned RAM — and served a real
  prompt: coherent `fibonacci` answer, 64 prompt + 120 generated tokens, **decode 35.6 tok/s (75% draft
  acceptance)**, at 131,072 context. No UR crash, no `layer 0 never rang`. **The post-load crash that produced
  the user's "bigger quants don't work" is fixed by rebuilding from current source (0.1.39 → 0.1.43.1).**
- [x] **Warmed benchmark (2026-10-13, same B70, IQ3_S, 131K, 3x 256-token runs):** decode **31.9–32.2 tok/s**
  (sustained), draft acceptance ~71%, prompt read ~53 tok/s cold / ~58 tok/s with prefix-cache reuse. First
  short request was 35.6 tok/s. Reference points from the docs: single B70 Coder IQ1_M (23 GB model) ~78 tok/s;
  community 2x B70 split Flash-Next IQ3_XXS ~66 tok/s. So ~32 tok/s for the 50 GB IQ3_S on one card is in the
  expected band; the B65+B70 pool (WS2) is the lever that should raise it.

### Runtime test logistics that mattered (for WS1/WS2 and reproduction)

- **Docker vs native:** the config's default `exe` is `sycl/serve/strata-sycl.sh`, which needs the
  `strata-sycl-dev` Docker image — **not built on this machine**, so the first start failed with
  `pull access denied`. The native host wrapper `sycl/serve/strata-sycl-host.sh` was **missing** (the old configs
  referenced it); recreated it (setvars.sh + `STRATA_VERIFY_DEVICE_PLAN=1`/`NO_HOST=1`/`STAGER_THREADS=12` + the
  binary, host paths not `/work`). Config `exe` repointed to it. **WS1/WS2 should make setup emit the host path
  when no Docker image is present.**
- **Port:** 8080 is held by the user's `llama-swap`; tests run on **8081**.
- **Config path note:** a start re-uses the config's stored port; `--port` to setup only applies at install. To
  move a port, pass `--port` to `server_intel.py` directly.
- The MTP draft layer downloads ~5 GB from the original Qwen checkpoint even for a local `--gguf-dir` model.

## Engine rebuild + runtime state (2026-10-12/13)

- [x] **Old configs/logs recovered (2026-10-12):** `strata-iq3_s.json/.log` show IQ3_S *fully loaded* on the B70
  (12,513 experts cached, "session is up", MTP loaded, 131K context) then crashed post-load:
  `verify: layer 0 never rang` + `level_zero error 37 (UR_RESULT_ERROR_UNINITIALIZED)`. A dual-GPU attempt died
  on a malformed `level_zero:1,level_zero:0` selector ("Too many colons"). So quants load; the lived failure is
  a post-load startup crash + a broken split selector. This **revises the earlier root-cause framing** (detection
  was secondary; the crash is primary).
- [x] **Engine rebuilt 2026-10-12 → 0.1.43.1** (AOT bmg-g31, native icpx 2026.1.1), replacing the known-broken
  0.1.39 binary; the #784/#866/#867 ring-wait/compile fixes are in. *Note: prior probe text said oneAPI 2025.3;
  that was a misread — it is 2026.1.1.*
- [x] **Selector syntax resolved (2026-10-12, probed):** `level_zero:0,1` selects exactly B65+B70 (excludes the
  B580); `level_zero:*` selects all three. The old crash was a doubled `level_zero:` prefix.
- [x] **IQ3_XXS was copied to disk (2026-10-13)** then **superseded the same day**: the user chose to test IQ3_S
  instead (for a quality comparison against Q4_K_M) and is **deleting the IQ3_XXS copy and downloading IQ3_S**.
  Do not assume the IQ3_XXS shards are still present.
- [x] **Prior config paths stale:** the saved configs point at `/mnt/1TB/AI/Strata-data/...`, which no longer
  exists; any rerun needs a fresh setup (pack + MTP rebuild) or repointed paths.

## Open user decision (2026-10-13): which quant to test, and the quality gate

The user is reconsidering tiers. Direction stated: **test IQ3_S** (not IQ3_XXS) **and get it running, to compare
quality against their local Q4_K_M.** That makes the runtime test double as a quality comparison, not just a
crash check. Open points:

- **IQ3_S is not yet on disk** (only IQ3_XXS was copied). Its GGUF must be fetched/copied first, **and the old
  IQ3_S config points at a deleted `Strata-data/`**, so it needs a fresh setup (pack + MTP rebuild). Disk is
  tight (~26 GB free) for a pack beside the GGUF.
- Q4_K_M still can't run on Strata (K-quants refused) — so "compare IQ3_S to Q4_K_M" means running Q4_K_M under
  a *different* engine (e.g. llama.cpp) for the quality baseline, then comparing Strata's IQ3_S output to it.
  That comparison harness is itself a small task.
- Whether this supersedes the literal-Q4_K_M headline gate (WS5) is still the user's call; for now the plan is:
  get IQ3_S running on the B70 → produce its output on a fixed prompt set → compare against Q4_K_M output →
  decide the final quality gate from measured quality, not Reddit.

## The q4_k_m gate needs renegotiating (flagged now, decided with the user)

The intent's success bar is "at least a q4_k_m-class quant." Two honest readings now that the constraint is known:

- **Literal Q4_K_M (this GGUF):** requires Strata to *add support* for plain K-quant experts — the docs note the
  K-quant prompt kernels (`STRATA_MMQ_KQUANTS`) are CUDA-only today and UD-Q4_K_XL is flagged `nvidia_only`.
  That's real engine work on top of WS0-WS2, and out of the original "elevation, not new feature" framing.
- **The intent behind it (recommended):** the user named q4_k_m as "a tier *above* the 3060's iq3_xxs." Strata's
  actual ladder already has that tier: **IQ3_S** ("matches the full model," 3.5-bit) and **UD-IQ4_XS** (~4-bit,
  *between IQ3_S and UD-Q4_K_XL in quality* — i.e. above the iq3_xxs the 3060 ran). Both are loadable by setup
  today. Meeting the gate as "pooled pair runs IQ3_S or UD-IQ4_XS at ≥131K" honors the intent without a K-quant
  engine port. Literal Q4_K_M support becomes a *separate, later* opt-in workstream if still wanted.

This is the user's call — recorded here so the spec's gate 3 is written against the chosen reading.
