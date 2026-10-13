# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Active workstreams

- **Intel Arc, CUDA-level parity, dual-GPU pooled (B70 + B65):** confirmed intent and handoff doc at
  [docs/intent/intel-arc-dual-gpu.md](docs/intent/intel-arc-dual-gpu.md); spec at
  [docs/specs/intel-arc-dual-gpu.md](docs/specs/intel-arc-dual-gpu.md). Success gate: pooled pair runs ≥ q4_k_m
  at ≥ 131,072 context via the one-click Linux setup. Read the intent doc first; it corrects the "Intel support
  is a stub" premise (the `sycl/` port and community dual-B70 results already exist). **Progress 2026-10-13:
  WS0 done** (root cause = stale 0.1.39 binary's post-load crash; rebuilt to 0.1.43.1; B65 `e222` detection fixed;
  native `sycl/serve/strata-sycl-host.sh` recreated) and **WS2 done** (B65+B70 layer-split, 131K context,
  validated at 115,959 tokens read). **WS5 in progress:** Q4_K_M admitted to setup and now **serves on the pair**
  via the new per-stage host mirror (`STRATA_SPLIT_MIRROR=1`, `sycl/src/program/generate.cpp` +
  `gguf_expert_source.cpp`): balanced `--layer-split 24` runs at 29.5 tok/s (was REFUSED). Remaining gap to
  IQ3_S's 67 tok/s is PCIe mirror drag (21 GiB of experts off-card); the idle B580 (bmg-g21) as a 3rd card is a
  separate, unscoped project. Designs: [docs/specs/intel-arc-per-stage-mirror.md](docs/specs/intel-arc-per-stage-mirror.md),
  [docs/specs/intel-arc-autosplit-fix.md](docs/specs/intel-arc-autosplit-fix.md).

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
