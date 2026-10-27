# Task 8 evidence: prefill-stage boundary

Date: 2026-10-26. Same host and model as Task 7 (RTX 3060 12 GiB, driver 595.99.02, Swift-Qwen3.8-Flash-Next
IQ3_XXS, 131,072 context).

## Extraction

`PrefillStage` now names the per-chunk boundary: `run_chunk`, `set_handoff`, `drain_pipeline`, `reserve_uploads`,
`stage_count`, `chain_experts_streamed`, plus the `STRATA_MMQ_DIRECT_LOG` diagnostic hook. `Prefill` implements it,
and the existing recursive `run_impl`/pipeline-thread call sites use the interface. Chunk tokens/positions, the
two-slot pinned hand-off, hand-off events, `single_chunk`/`pipe_chunked`, pipeline drain and upload reservation keep
their existing arguments and order. `should_stop`, `stop_at_chunk`, `on_chunk`, MTP `draft_kv`, PLE and local
threading remain in the existing `Prefill` path.

## Verification

- CUDA engine build: `cmake --build build-cuda-smoke13 --target strata -j8` - passed.
- `ctest --test-dir build-cuda-smoke13 -R '(prefill|batch_read)' --output-on-failure` - 7/7 passed:
  `batch_read_test`, `prefill_plan_test`, `prefill_stage_test`, `prefill_fused_moe_test`, `prefill_fused_iq_test`,
  `prefill_mmq_scatter_test`, `prefill_mmq_kquant_test`.
  - The first `prefill_fused_iq_test` run failed with `cudaMalloc: out of memory` while a 42 GiB prompt smoke held
    the GPU. After that process ended, the test alone passed in 97 s and the full set above passed.
- Prompt/output comparison against the pre-refactor baseline:
  - Old baseline: `/tmp/opencode/stage-smoke-split-baseline.json` (recorded before Tasks 6-8).
  - Current: `/tmp/opencode/stage-smoke-split-task8.json` (same config, same `--new 32`).
  - Both prompts generated identical token IDs and decoded text. This exercises the in-process split prompt path and
    the following decode.
- Existing cancellation/stop callback code is unchanged; `serve.test_server` (355 tests, including cancellation and
  stopping) passed after the extraction.

Limits: no new long-prompt/cancel-focused prefill test was added. The prompt comparison is a short prompt on one
GPU; it is a behavior regression check, not a long-context measurement. HIP was not run (no ROCm/hipcc on this host).
