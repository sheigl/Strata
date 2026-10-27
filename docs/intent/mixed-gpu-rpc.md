# Mixed-GPU RPC Intent

Confirmed 2026-10-26.

- **Outcome:** Add an opt-in, llama.cpp-inspired RPC/worker capability to Strata that pools mixed-vendor GPU memory for one inference workload. Prove it on one host first, then extend it to networked workers.
- **User:** The maintainer's NVIDIA + Intel Arc setup first, using a reusable design for other Strata users rather than a hardware-specific private patch.
- **Why now:** A concrete Qwen workload needs more GPU memory than one device provides. RPC is the means, not the goal.
- **Success:** Qwen3.8-Flash-Next at Q4_K_M or better and at least 131,072 context runs correctly and stably across NVIDIA and Intel backends, sustaining at least 20 generated tokens/s. Record prompt-processing and decode performance.
- **Constraint:** Whole-layer, expert, or similarly coarse block partitioning is acceptable. Planning and technical planning are required before implementation handoff.
- **Out of scope:** llama.cpp wire compatibility, mixed Strata/llama.cpp clusters, mandatory fine-grained tensor splitting, and AMD or multi-host validation in the first milestone.
