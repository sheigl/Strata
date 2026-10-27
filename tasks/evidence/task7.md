# Task 7 evidence: CUDA RPC prefix decode

Date: 2026-10-26. Host: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 595.99.02. One visible CUDA card; no
ROCm/hipcc on this host. Model: Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS, pack
`/mnt/1TB/AI/Strata-data/packs/swift-iq3_xxs`, 131,072 context.

## Build and focused tests

- CUDA engine: `cmake --build build-cuda-smoke13 --target strata -j8` - passed.
- `ctest --test-dir build-cuda-smoke13 -R '(verify|split|stage_chain|rpc_)' --output-on-failure` - 13/13 passed:
  `rpc_protocol_test`, `rpc_local_transport_test`, `rpc_control_test`, `rpc_decode_split_test`,
  `verify_parity`, `spec_verify_parity`, `verify_batch_parity`, split/GGUF/stage tests.
- `ctest --test-dir build-rpc -R '^rpc_'` - 4/4 passed.
- `python -m unittest tests.terminal.test_rpc_split tests.terminal.test_mixed_vendors serve.test_rpc_workers` -
  27/27 passed.
- `python -m unittest serve.test_server` - 355/355 passed.
- HIP build: **not run**; this host has no `hipcc` and no ROCm runtime.

## Same-backend fixed-window parity

Command:

```sh
/mnt/1TB/AI/Strata/.venv/bin/python /tmp/opencode/rpc_parity_smoke.py \
  --config /tmp/opencode/strata-smoke-rpc.json \
  --exe /mnt/1TB/AI/Strata/build-cuda-smoke13/strata \
  --out /tmp/opencode/rpc-parity-corrupt-serve.json --new 12
```

Result: `equal: true`. Both arms generated:

```text
248068 198 760 1156 16561 264 4145 6673 3296 25 220 17
```

The run used `--spec-corrupt 2` in the serve loop so both arms received the same deliberately wrong drafts, and it
produced partial commits (`keep 2 of 4`, `keep 1 of 4` in the commit trace). `DONE` reported 3 drafts accepted in both
arms (3 of 25 in the baseline, 3 of 27 over RPC). This is the rejected-draft/partial-commit parity case.

This comparison needs matching per-stage expert-cache placement. On one 12 GiB card the two RPC processes cannot
allocate the same VRAM-free cache as two in-process stages, so the run used the diagnostic
`STRATA_RPC_CACHE_SLOTS_CAP=200` and `--adapt-every 0` on both arms. The baseline also used
`STRATA_SPLIT_SAME_STAGES=1 --trim-stage-weights --vram-reserve-later-mib 2192` to allocate a real second
`GpuStage` (own weights, session and cache) on CUDA0. Profile and cache-slot hashes matched exactly:

```text
stage 0 / worker:   profile bc44a7a42544c7e7, cache slots 2748d3e1de3c4b35
stage 1 / controller: profile d7083e488c1070eb, cache slots 77419eb951c8bced
```

Without the cache-slot cap, the two processes had different cache capacities (observed 890 vs 889 and larger
differences in other runs), and their GPU/CPU expert split changed a token. That is a cache-placement difference, not
a hand-off or commit-order difference: a control run with `STRATA_RPC_FORCE_CPU_EXPERTS=1` (all experts on the CPU,
cache retained for the verifier) was also bit-equal with rejected drafts.

RPC mode forces `STRATA_ROUTE_TAIL_SKIP=0` in both processes. That option changes the answer with the verify-window
size, and MTP draft windows can differ when the controller owns a suffix-only session.

## Live supervisor/API run

Started through `serve.server` with the config's `rpc_split` section (worker CUDA, boundary 24), port 8085. The
worker owned layers 0-23 and the controller 24-47; the log said `RPC prefix READY`. A greedy OpenAI request returned
8 tokens at 3.4 tok/s after a 58-token prompt (RPC prompt read through windows). The server stopped cleanly and left
no `--rpc-listen`/`--rpc-connect` process behind.

Limits: one GPU, so worker and controller shared the 3060; this is not mixed-vendor validation. The run is a short
prompt and a small cache (300 requested per side, 2,048 MiB reserve), not the 131K long-context gate. It uses a
controller-side suffix-only session, so its MTP draft stream is not identical to the in-process split; the fixed-window
parity above controls that by corrupting the drafts identically on both sides. Mixed CUDA + Intel Arc remains
unmeasured.
