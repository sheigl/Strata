# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5, Unsloth, Gyro and NVFP4
variants, and Qwen3.6-35B-A3B) on a normal PC: one NVIDIA or AMD graphics card plus system RAM on Windows or Linux,
or an Apple Silicon Mac. Version 0.2.2, engine 0.1.47. It has three parts:

- **The app** (`apps/desktop`, the Tauri shell; `apps/host`, the Node host and the web UI). It is for people who do
  not want a terminal. The installer has the engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is the older one-click installer from source.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For an agent that has a shell, the short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh                  # Linux, Mac
   irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). The old route (clone
the repository and run `START-HERE.bat --yes` or `./setup.sh --yes`) still works and is the fallback when the
Terminal cannot run. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](packaging/README.md) and [packaging/UPDATES.md](packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `tests/terminal` (`python -m pytest tests/terminal`). The app's UI
  tests run with `npm test` in `apps/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Active workstreams

- **Intel Arc, CUDA-level parity, dual-GPU pooled (B70 + B65):** confirmed intent and handoff doc at
  [docs/intent/intel-arc-dual-gpu.md](docs/intent/intel-arc-dual-gpu.md); spec at
  [docs/specs/intel-arc-dual-gpu.md](docs/specs/intel-arc-dual-gpu.md). **The complete technical record (what we
  did, every lever pulled, every lever left, the open Intel optimization space) is
  [docs/specs/intel-arc-full-record.md](docs/specs/intel-arc-full-record.md) — start there.** Success gate: pooled
  pair runs ≥ q4_k_m at ≥ 131,072 context via the one-click Linux setup. Read the intent doc first; it corrects
  the "Intel support is a stub" premise (the `sycl/` port and community dual-B70 results already exist). **Progress 2026-10-13:
  WS0 done** (root cause = stale 0.1.39 binary's post-load crash; rebuilt to 0.1.43.1; B65 `e222` detection fixed;
  native `sycl/serve/strata-sycl-host.sh` recreated) and **WS2 done** (B65+B70 layer-split, 131K context,
  validated at 115,959 tokens read). **WS5 done:** Q4_K_M admitted to setup and now **serves on the pair**
  via the new per-stage host mirror (`STRATA_SPLIT_MIRROR=1`, `sycl/src/program/generate.cpp` +
  `gguf_expert_source.cpp`): balanced `--layer-split 24` with `--vram-reserve-mib 1024` runs at 32.2 tok/s (was
  REFUSED). **WS1 (images) done 2026-10-14:** the SYCL port reads pictures through the CPU image encoder
  (`--vision cpu`; 2.4-3.0 s for a 300-token picture, no VRAM used) — see "Images on the SYCL port" in
  [docs/INTEL.md](docs/INTEL.md); a SYCL build of the encoder is not done. **WS6 (third card) done 2026-10-15:**
  the idle B580 joins the pool as expert cache — `--layer-split 18,43`, three stages, measured 26.3 tok/s and a
  **9.78 GiB** pinned mirror against the pair's 18.1 tok/s and 19.44 GiB on the same prompt (Q4_K_M, 131K; the pair's
  32.2 tok/s in WS5 is a different, shorter prompt). The B580 takes only the 5 layers whose 2,560 pairs its 8.54 GiB
  covers completely, because its link is 1.71 GB/s against 13.8/14.0 for the Pros — its verify window then runs a
  `100% VRAM resident: zero-doorbell graph`. Two engine defects fixed: the port was missing the CUDA engine's
  per-stage link pricing (`link_scale`), and the measured coverage curve was gated to `ns >= 4`, so 3-GPU `auto`
  picked `K=2,47` (43.4 GiB mirror, never started inside the 900 s budget) — it now picks `K=18,43` with no
  environment variable. `setup_intel.py` writes `STRATA_SPLIT_MIRROR=1`/`STRATA_STAGE_TRIM=1` for every split. **Long
  context, measured 2026-10-15 on the previous install:** at 200,000 context `--kv-resident 0` (the whole K/V in
  VRAM, ~1.3 GiB) decodes at **35.2 tok/s** with 89K tokens of context behind it against 15.5 tok/s for the
  `--kv-resident 32768` the live configs and the AMD docs use, and reads an 88,949-token prompt at 1,026 against 764
  tok/s; the mirror grows 9.85 → 11.14 GiB for +0.5 GiB net pinned RAM. **Re-measured 2026-10-17 on the reinstalled
  box, engine 0.1.45 (oneAPI 2026.2):** the direction holds — `--kv-resident 0` **26.1 tok/s** against the `32768`
  arm's 21.7 and 530 against 374 tok/s read, at 109,557 tokens of context; three cards **26.1** against the pair's
  20.1 for two thirds of the pinned RAM (16.15 against 24.92 GiB), the pair at 200,000 with everything resident being
  the run section 11 had listed as unmeasured. At 131,072 with `--kv-resident 32768` the mirror still halves (13.84
  against 22.62 GiB) but the decode lead is 1.06x, not the 1.45x the 2026-10-15 short-prompt table records: that
  prompt leaves nothing past 32,768 cells to stream, this one streams 76,789 in both arms. Its short-prompt case is
  half-measured — three cards **16.1 tok/s** at a 100-token prompt against 28.3 at 109,557 tokens, same flags and
  cache (a cold-start effect: only the long prompt's 203 s prefill warms the cards) — and the pair's arm still needs
  one run. What moved the absolute figures is the install, not the merge: it
  charges each card ~1.7 GiB more before the cache exists, so the caches are 8,627/1,930 pairs against 9,162/2,477
  and the mirror 16.15 against 11.14 GiB — the previous install's own `0.1.43.1` binary reproduces the new figures
  digit for digit, while `0.1.45` decodes **1.34x** that binary's rate on byte-identical output. **B580 removed
  2026-10-17:** the pair (B65+B70) at 135,168 cannot hold `--kv-resident 131072` — REFUSED, 0.84 GiB free on the head
  stage against the 851 MiB the smallest cache needs — so the live `strata-swift-q4_k_m-split.json` runs
  `--kv-resident 65536` (measured: READY 578 s, 412 tok/s read of a 109,557-token prompt, 26.6 tok/s decode, 22.85
  GiB mirror + 0.80 GiB K/V pinned, `level_zero:0,1`). See "Long context",
  "Three cards" and "Re-measured on 0.1.45, on the reinstalled box" in [docs/INTEL.md](docs/INTEL.md) and section 11
  of [docs/specs/intel-arc-full-record.md](docs/specs/intel-arc-full-record.md). Remaining gap to IQ3_S's 67 tok/s is
  PCIe mirror drag; open: the ~2.16 GiB VRAM difference between `auto` and an explicit split (not reproduced on the
  new box — an explicit `18,43` reads `27.80 GiB free -> 8029 slots`, the `auto` range), and the B580 as a vision
  encode card. Designs: [docs/specs/intel-arc-per-stage-mirror.md](docs/specs/intel-arc-per-stage-mirror.md),
  [docs/specs/intel-arc-autosplit-fix.md](docs/specs/intel-arc-autosplit-fix.md).

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
