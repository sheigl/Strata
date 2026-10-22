# STRATA_MIRROR_ADAPT: adaptive expert swaps backed by the pinned mirror (handover, 2026-10-22)

This file is the design doc **and** the work-in-progress handover for the session that implemented it.
Read §1-§3 for context, §4 for exactly what is in the working tree, §5 for the remaining steps.

## 1. Where the branch is

Branch `intel-arc-0.2.2` in `/mnt/1TB/AI/Strata` (fork `sheigl/Strata`, upstream `Niko1221/Strata` as `origin`):

- `3c73ea4c` — merge of v0.2.2 (origin/main, engine 0.1.47) into the branch. Clean.
- `89a3dbf3` — docs: v0.2.2's images-on-Intel story (AGENTS.md, docs/INTEL.md, docs/specs/intel-arc-dual-gpu.md).
- `a85d2209` — cherry-pick of upstream PRs #2433 + #2436 (SYCL `--pipeline-windows` fix: poll a window still on
  the GPU instead of `verify: error`). Applied clean, built, **measured**: pair decode +8-12% (see §6).
- **UNCOMMITTED**: the STRATA_MIRROR_ADAPT implementation (§4). It compiled past everything except one header
  error that was just fixed (`sycl::queue` forward declaration); the rebuild after that fix was interrupted.
  **Next step is `bash` the build (§5.1) and fix whatever falls out.**

Gates already passed on this branch (all on the rig below): setup suites (test_setup_sycl 30/30,
test_setup_choices 50/50, test_setup_amd 51/51, test_sycl_build_guard 3/3); merge gate: 0.1.47's FP16 prompt
path **bit-identical** to 0.1.45 (first-window logits md5 `104c4bc8…`, both binaries), int8 prompt path +38%
and different bits by design.

## 2. The rig

Arc Pro B65 32 GB (`level_zero:0`) + Arc Pro B70 32 GB (`level_zero:1`), both Battlemage (AOT `bmg-g31`),
63 GB RAM, SSD with slow random 4K reads (`--ple-io mmap` is the fix; `direct` can stall seconds).
oneAPI 2026.2 at `/opt/intel/oneapi/setvars.sh`; `SYCL_CACHE_PERSISTENT=0` always.
The B580 was retired; three-card runs are impossible now.

Key measured numbers (this rig, this branch): pair IQ3_S 100% resident: 55.9 tok/s decode / 583 tok/s read
(0.1.43.1, agent turns); pair Q4_K_M 69% resident: ~27 tok/s decode / ~408 tok/s read (0.1.47) — the gap is
PCIe mirror drag; single B70 IQ3_S: int8 prompt 905 tok/s vs FP16 655 (bit-identical to 0.1.45).

## 3. The problem the change solves

- With `--stream-experts` + `STRATA_VERIFY_NO_HOST` (the strata-sycl host wrapper's default), the verify
  device plan can only run experts that are in VRAM or in the pinned mirror.
- The mirror holds **the start's misses only**. An expert the adaptive tier evicts later was resident at
  start, so it is in no mirror → under NO_HOST the device plan would have nothing to read for it.
- The non-deferred mirror path (`generate.cpp`) therefore prints "adaptive swaps off" and sets
  `adapt_every = 0`. **The deferred `STRATA_SPLIT_MIRROR` path never had that guard** — our Q4_K_M pair
  configs run the tier *on* under NO_HOST (latent #1980-class hole; it needs a batch window counting misses
  to trip, which is why it hasn't blown up in single-stream use).
- So today the pair's Q4_K_M cache never adapts to the session: 69% coverage forever, every window pays the
  mirror for 31% of routed experts → ~27 tok/s ceiling.

**The change:** `STRATA_MIRROR_ADAPT=1` (opt-in). When the adaptive tier evicts an expert whose bytes are
not mirrored, copy them D2H from its cache slot into the pinned mirror *before* the slot is overwritten
(same in-order refill stream), publish the alias in the owning stage's device mirror table when the round's
copies have landed (`apply_pending`), and keep the tier on under NO_HOST. Victims the cap has no room for
keep their slots (their swaps drop out of the round). Byte-identity: the mirror holds the slot's own bytes,
identical to the GGUF's; only residency changes. Default (switch unset): byte- and behavior-identical to
before — and the deferred path now gets the same "adaptive swaps off" guard the single-table path always had
(that's the bug fix riding along; message mentions the new switch).

## 4. What is in the working tree right now (uncommitted)

1. `sycl/include/strata/core/gguf_expert_source.hpp`
   - `namespace sycl { class queue; }` forward declaration (the header must not include SYCL).
   - `adopt()` declaration (doc comment explains the contract).
   - Members `adopt_cur_ / adopt_off_ / adopt_size_` (rolling adopt chunk).
2. `sycl/src/core/gguf_expert_source.cpp`
   - `GgufExpertSource::adopt(layer, expert, dev_src, bytes, cap_bytes, q, err)`: no-op + alias if already
     `pinned()`; refuses when `mirror_bytes_ + blob > cap_bytes`; allocates 256 MiB USM-host chunks
     (`sycl::malloc_host` on the caller's queue); queues `q.memcpy(dst, dev_src, b)` (in-order with the
     overwrite); registers `mirror_ptr_`/`layer_first_`; bumps `mirror_bytes_`; returns the alias.
3. `sycl/src/program/generate.cpp`
   - `mirror_cap_bytes` (function scope, set in **both** mirror paths where `cap` is computed).
   - `mirror_adapt` env flag next to `split_mirror`.
   - Non-deferred disable site: gated `!mirror_adapt`; added the opt-in announcement line.
   - **Deferred path: added the missing guard** in the `if (multi_gpu)` block after the REFUSED checks
     (`defer_mirror && later_misses > 0 && no_host_split` → disable, or announce with the switch on).
   - Serve adapt round (after `resident_stage_swaps`, before `pin_blobs`): pre-pass adopting each
     non-pinned victim via `gguf_src.adopt(...)` from the owning card's slot on its adapt stream; drops
     non-adoptable swaps; accumulates `adopt_pending {tab, idx, alias, q}` (struct declared beside
     `pending`). `stage_mirror.empty() ? mirror_table_d : stage_mirror[stn]` picks the table.
   - Serve `apply_pending`: after the stage events complete, publishes the table entries with
     `ap.q->memcpy(ap.tab + ap.idx, &ap.alias, 8).wait()` (streams idle then → synchronous).
   - Non-serve (CLI) adapt round + its `apply_pending`: the same pre-pass/publish, single table
     (`mirror_table_d`), own `AdoptPub` declaration.

Known design limits (fine for v1, say so in the PR): a stage whose table is null (no start misses) drops
adopts silently; the mirror only grows (no un-adopt); `STRATA_CHECK_SWAPS=1` still validates the exchange
path, not the adopt bytes.

## 5. Remaining steps, in order

### 5.1 Build (was interrupted right after the header fix)

```sh
cd /mnt/1TB/AI/Strata
bash -c 'set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u; \
  export SYCL_CACHE_PERSISTENT=0; \
  AOT=bmg-g31 BUILD_DIR=/mnt/1TB/AI/Strata/build-sycl JOBS=8 bash sycl/tools/build.sh 2>&1 | tail -4'
```

Expect either success (`BUILD EXIT 0`) or a handful of type errors in `generate.cpp` (the pre-pass touches
`stage_mirror`, `stage_of`, `adapt_stream` — all in scope where used; the AdoptPub struct is declared twice,
once per path, intentionally). Fix what the compiler says; do not redesign.

### 5.2 Validation gates

1. **Default path unchanged:** run the pair IQ3_S config with NO new switch: log must NOT contain
   "STRATA_MIRROR_ADAPT"; Q4_K_M split config must now print the "adaptive swaps off … (STRATA_MIRROR_ADAPT=1
   adopts swap-outs …)" line (deferred-path guard). IQ3_S pair (no misses) prints nothing new.
2. **Feature on:** Q4_K_M pair config + `STRATA_MIRROR_ADAPT=1` in `env`: start prints the "stays on" line;
   watch `STRATA_TRACE_ADAPT=1` for swap rounds; mirror grows (`mirrored_bytes` in logs) as rounds adopt.
3. **Speed A/B (the payoff):** same Q4_K_M pair config, fixed prompt set (e.g. 3 sequential ~9K-char docs
   prompts + 200 tokens each, temperature 0): decode tok/s with switch off vs on over a long session.
   Baseline today: ~27 tok/s decode / ~408 tok/s read at 109-110K context.
4. **Byte-identity:** greedy outputs with the switch off vs on over the same prompts must match token for
   token (residency changes, bytes don't). Use the same docs-prompt harness as the earlier gates.
5. **Cap behavior:** set `STRATA_MIRROR_MIB` small (e.g. just above the start mirror), run, expect the
   one-time "the mirror cap is spent" note and swaps limited to already-mirrored victims — no crash.
6. Setup suites again (`python -m pytest tests/terminal -q`, and `python tools/test_setup_sycl.py`) — nothing
   they cover changed, but run them.

### 5.3 Commit and PR shape

- One commit for the bug guard + feature together is fine for the branch; for upstream it splits into two PRs
  (one problem per PR): **(a)** "[Engine] SYCL: the deferred split-mirror path also turns adaptive swaps off
  under NO_HOST" — tiny, safe; **(b)** "[Engine] SYCL: STRATA_MIRROR_ADAPT=1 …" with the §5.2 numbers.
  Defaults byte-identical; opt-in; sycl/ only; CUDA/HIP untouched; say what was not run (CUDA/HIP builds,
  single-card non-defer runtime, batch windows).
- Title rules: area prefix `[Engine]`; search first; the related issues are #1980 (the class of hole),
  #1440/#1054 (split regressions), #533 (split reserves).

### 5.4 After that (the queue, in priority order)

1. **Post the #2433 validation** on upstream PR #2433 (our A/B table: serial 52.5/60.0/61.0 vs
   `--pipeline-windows 2` 58.6/65.4/65.7 tok/s, zero `verify: error`, B65+B70, streamed experts, MTP
   `--spec 4`). Community validation nudges the merge. Already committed locally as `a85d2209`.
2. **Q4_K/Q6_K int8 prompt experts** (the big ownable item): today's pair gate proved Q4_K_M gets 0% from
   the int8 prompt path (coverage: IQ3_S/IQ3_XXS/IQ2_S gate/up + IQ4_NL/Q2_0 down only). Kernels + parity
   tests; scope issue first per CONTRIBUTING. Nobody has an open PR for Flash-Next's SYCL path (#3679/#3699
   are the q36/qwen35 engine).
3. Watch, don't duplicate (all open + MERGEABLE upstream): #3248 (B70/B65 +10-35% prompt — **conflicts with
   our fork in `sycl/setup_intel.py` and `docs/INTEL.md`**; let it land upstream, take it in the next merge),
  #3309 (`STRATA_WARM_PROMPT`), #3306/#3310 (per-GPU chunk, prompt-loan slots), #2804 (Q4_K decode kernels),
   #3095 (PW_WITH_BATCH), #3185 (arena RAM hand-back).
4. **Docs + vault pass** (after the mirror-adapt numbers land): docs/INTEL.md gets today's measured findings —
   `--pipeline-windows 2` +8-12% pair decode with the fix; `--prefill 8192` *slower* on the split (borrow
   eviction); `parallel: 2` default a loss on the pair; int8 = +3% decode on pair IQ3_S, no prompt gain at
   2,280 tokens. Re-index the vault speed record (the draft is `/tmp/opencode/speed-record.md`; extend it with
   §6, then `memory-vault-write.index_document` again, same name).
5. **One untested arm, low priority:** `--batch 2 --batch-groups 1 --batch-mtp` (serial windows + per-slot
   drafters) — the shape upstream measured best for two streams (89.2 aggregate on 2×3080). Only worth
   measuring if multi-client serving matters; single-agent use stays on one stream.

## 6. Experiment results this session (measured on the rig; don't re-run)

| experiment | result |
|---|---|
| pair IQ3_S int8 auto vs `STRATA_PREFILL_MMQ=0` | 53.6 vs 52.0 tok/s decode (+3%); prompt 2,280 tok ~8 s both — the pair's read is not expert-GEMM-bound |
| `--ple-io mmap` on the pair | no change (~9 s vs ~8 s); keep for watchdog safety |
| `--prefill 8192` on the pair | **35% slower** (76 s vs 56 s on 15,258 tok); the docs' +10-16% was single-card |
| `--pipeline-windows 2` unpatched | `verify: error` → HTTP 400 every request |
| `--pipeline-windows 2` + #2433+#2436 | 0 errors; decode +8-12% (table in §5.4.1) |
| `parallel: 2` (default pipelined batch) | 6.4 + 9.4 tok/s per stream vs 53.6 solo — aggregate below one stream; drops MTP on a split; slot cost 0.24 GiB/GPU at 8K ctx |
| merge gate, single B70 IQ3_S, 2,760-token prompt | int8 905.1 tok/s; FP16 654.8 vs 0.1.45's 656.7; FP16 **bit-identical** (md5 `104c4bc8…`), int8 differs by design; output ids identical in all arms |

Cold-start caveat for all short-prompt numbers: first request after READY reads low (documented 16.1 vs 28.3
tok/s); A/B arms in this session used the same cold/warm pattern on both sides.

## 7. Harness bits

- Pair configs: `strata-iq3_s-split-047.json` (port 8092), `strata-swift-q4_k_m-split-047.json` (8091);
  both use `build-sycl/strata`, `--layer-split 24 --split-device 1`, `STRATA_STAGE_TRIM/SPLIT_MIRROR=1`.
  Variant configs for experiments are generated by `/tmp/opencode/pair-exp.sh` into `/tmp/opencode/pair-exp-*.json`.
- Gate scripts in `/tmp/opencode/`: `pair-gate.sh` (Q4_K_M pair), `pair-iq3s-gate2.sh` (IQ3_S pair int8/FP16),
  `pair-pw-ab.sh` (serial vs pipeline A/B), `pair-exp.sh` (the 5-arm matrix), `gate-i8-3.sh` (single-card).
- Server start: `.venv/bin/python sycl/serve/server_intel.py --engine strata --config <cfg> --port <p>`;
  readiness = `ready:` line in the log; engine stderr is in the config's `"log"` file, the `[strata]`-prefixed
  lines in the server stdout.
- Speed record draft: `/tmp/opencode/speed-record.md`; already indexed in the memory vault (document 3541,
  "Strata Intel Arc speed tests - one index") — searchable.
- CONTRIBUTING.md rules that apply to the PR: area-prefixed title, one problem per PR, opt-in + byte-identical
  default, measured numbers with the rig described, passed/failed/not-run kept apart, mark static analysis.
