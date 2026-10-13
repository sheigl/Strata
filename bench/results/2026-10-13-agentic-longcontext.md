# Agentic + long-context comparison: Strata IQ3_S vs Q4_K_M

**Date:** 2026-10-13. **Machine:** B70+B65 pair, layer split, 131K context. **Harness:**
`bench/agentic2_compare.py` — identical prompts, `temperature 0`, per model. Extends
`2026-10-13-quality-comparison.md` (coding) and `2026-10-13-broad-comparison.md` (general) with the two hardest
axes: **agentic / tool-calling** and **long-context recall**. The user asked specifically for these.

## Agentic / tool-calling (8 tasks, real OpenAI tool schemas, 6 tools incl. distractors)

| Task | what's correct | IQ3_S | Q4_K_M |
|---|---|---|---|
| weather in Tokyo | 1 call, city=Tokyo | ✅ | ✅ |
| `124*456+789` | call `calculate`, don't guess | ✅ | ✅ |
| find tests then run them | search_files → run_tests | ✅ (deps) | ✅ (deps) |
| weather + stock price | 2 independent calls, parallel | ✅ | ✅ |
| read config, weather there | read_file | ✅ | ✅ |
| explain photosynthesis | **no tool call** (none fits) | ✅ | ✅ |
| find markdown files | search_files `*.md` | ✅ | ✅ |
| weather where the Eiffel Tower is | reason → Paris → get_weather | ✅ | ✅ |

**8/8 both.** Both emit valid-JSON tool calls, orchestrate multi-intent and dependent calls, make parallel
independent calls, and correctly decline to call a tool when none fits. The engine's tool-call path works on both
quants (the Strata server logs show "writing a tool call: ..." then "tool call complete" — the structured output
is generated, not regex-recovered).

## Long-context recall (needle-in-a-haystack, ~75K-token document ≈ 57% of the 131K window)

A single secret (`XJ-9942-QUAIL`) planted at increasing depth; the question asks for it.

| depth | IQ3_S | Q4_K_M |
|---|---|---|
| 10% | ✅ FOUND | ✅ FOUND |
| 50% | ✅ FOUND | ✅ FOUND |
| 90% | ✅ FOUND | ✅ FOUND |
| 98% | ✅ FOUND | ✅ FOUND |

**4/4 both.** Notably, Q4_K_M's recall is **not** hurt by the PCIe mirror — the mirrored experts being read over
PCIe costs *speed* but not *accuracy*; the model still finds the needle at the window's edge. This directly
answers "does quality hold near the context limit": yes, on both, for retrieval. (This tests retrieval, not
long-range *reasoning*; the latter is the harder thing to measure and is unmeasured here.)

## Speed on these runs (the cost of the quality)

- **IQ3_S:** 48-76 tok/s on agentic (short structured outputs draft very well); the 75K-token needle prompts read
  at ~1,600-1,700 tok/s (the prompt-ingest path is fast), and recall decodes at ~57-66 tok/s.
- **Q4_K_M:** 14-34 tok/s decode; on the *long* needle prompts the mirror drag compounds (the longer the read +
  decode, the more PCIe reads), which is where its slowdown is most visible.

## Verdict across all three suites (coding + broad + agentic/long-context)

**IQ3_S and Q4_K_M are quality-equivalent on every axis measured** — coding correctness, general knowledge, logic
traps, web design, creative writing, instruction-following, agentic tool-calling, and long-context retrieval.
**No task showed a Q4_K_M win.** IQ3_S is ~2x faster and fits VRAM. For this user's use (coding + agentic), IQ3_S
is the clear default; Q4_K_M (32 tok/s, 131K) is proven working for any case where 4-bit would matter, but none
surfaced. The quant difference these tests were designed to expose did not appear — both quants sit above the
capability threshold of every task tried, so the honest conclusion is that *at this model's size, IQ3_S is not
the limiting factor* on these workloads.
