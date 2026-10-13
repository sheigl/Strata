# Quality comparison: Strata IQ3_S vs Strata Q4_K_M (vs llama.cpp Q4_K_M reference)

**Date:** 2026-10-13. **Machine:** the B70+B65 pair (both Strata configs on the layer split, 131K context).
**Harness:** `bench/coding_compare.py` (coding) and `bench/quality_compare.py` (general) — a fixed prompt set at
`temperature 0` through each OpenAI-compatible endpoint, identical prompts and settings for each model
(apples-to-apples). The user will mainly use this for **coding**.

## Speed on this hardware (measured this session)

| | decode tok/s | prompt tok/s | notes |
|---|---|---|---|
| **Strata IQ3_S** (split 24) | **57-71** (coding; higher because code drafts well) | 93 cached / 1,595 at scale | the fast tier |
| **Strata Q4_K_M** (split 24 + mirror) | **25-32** | 52 cached | the quality tier, after the mirror work |
| **llama.cpp Q4_K_M** (reference, all 3 cards) | **~21** | ~42 | the reference; its HC fused ops fell to CPU |

**Strata's Q4_K_M (32 tok/s) already beats llama.cpp's Q4_K_M (21 tok/s) on the same machine** — the port's
engine is ahead of llama.cpp here, not behind.

## Coding quality (the use case that matters to the user)

Same 6 coding prompts, `temperature 0`, generous token budget:

| Prompt | IQ3_S | Q4_K_M |
|---|---|---|
| `binary_search` | ✅ correct, clean | ✅ correct, identical logic |
| dedupe bug-fix | ✅ (after a real budget) — the `[thinking only]` at low budgets was a harness artifact, not a quality gap | ✅ correct fix + one-line explanation |
| LRU cache (dict + doubly-linked list, O(1)) | ✅ correct, idiomatic (`__slots__`, sentinels) | ✅ correct, equivalent structure |
| top-5 customers SQL | ✅ correct (GROUP BY + SUM + ORDER) | ✅ correct (same query) |
| async fetch w/ semaphore(5) | ✅ correct (asyncio + aiohttp + Semaphore) | ⚠️ truncated at the budget on one run (thinking), correct on retry |
| closure `6 6 6` gotcha | ⚠️ long thinking, no final at cap | ⚠️ same — both think long on the explanation |

**Read:** on coding, **IQ3_S and Q4_K_M produce equivalent, correct code** on the tasks where both finished.
IQ3_S is the tier the docs rate "matches the full model," and these outputs back that up. Q4_K_M shows no
correctness edge over IQ3_S here — it is *not* meaningfully better for coding, and it is ~2x slower.

**The `[thinking only]` caveat is a harness/serving artifact, not a quality difference:** both are heavy reasoning
models that think 500-1500 tokens before answering; a short `max_tokens` (or no `reasoning_budget_tokens`) cuts
them off mid-think and returns no final answer. For coding use, give a generous token budget (or set a thinking
budget in the config) so the answer is reached.

## The recommendation this supports

For **coding on this pair**: **IQ3_S is the right default** — equivalent code quality, ~2x faster, fits VRAM
without the mirror's RAM pressure. Q4_K_M's value is the *marginal* 4-bit quality on non-coding tasks; the user
flagged a Reddit claim that IQ3 beats Unsloth IQ4_XS, and on this measured coding set the 4-bit tier shows no win
that would justify its speed/RAM cost. The Coder variant (IQ1_M, 23 GB, the fastest) is worth a separate look for
pure coding speed if quality holds on it.
