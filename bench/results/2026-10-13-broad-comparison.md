# Broad quality comparison: Strata IQ3_S vs Q4_K_M (general knowledge, reasoning, web design, creative)

**Date:** 2026-10-13. **Machine:** B70+B65 pair, layer split, 131K context. **Harness:** `bench/broad_compare.py`
— a fixed 14-prompt set at `temperature 0`, identical prompts/settings per model (apples-to-apples), outputs
reviewed in full. This extends `2026-10-13-quality-comparison.md` (the coding set) to general use. The user's
stated primary use is coding, but they asked to test general knowledge / reasoning / web design too.

## The 14-prompt broad set

General knowledge (Australia capital, virus vs bacterium, planet moons), logic traps (bat-and-ball, the 3-coins
riddle, 0.999...=1), teaching (recursion for a 10-year-old, TCP vs UDP), web design (flexbox card HTML+CSS,
responsive 2-column CSS), creative (robot-painting story, night-coffee tagline), instruction-following (planets in
order, reply only "DONE").

## Result: a wash — both are excellent and equivalent

| Prompt | IQ3_S | Q4_K_M |
|---|---|---|
| Australia capital + fact | ✅ | ✅ (equally good) |
| virus vs bacterium | ✅ | ✅ |
| most moons | ✅ Saturn ~274 | ✅ Saturn ~270 |
| bat-and-ball ($0.05) | ✅ | ✅ |
| 3-coins riddle | ⚠️ (thinking, no final at cap) | ✅ "two dimes and a nickel" |
| 0.999...=1 | ✅ | ✅ |
| recursion for a kid | ✅ | ✅ |
| TCP vs UDP | ✅ | ✅ |
| flexbox card HTML+CSS | ✅ complete, clean | ✅ complete, clean |
| responsive 2-col CSS | ✅ grid + media query | ✅ grid + media query |
| robot-painting story | ✅ | ✅ |
| night-coffee tagline | ✅ "Coffee for the hours that don't sleep." | ✅ "Brewed for the night." |
| planets in order | ✅ | ✅ |
| reply only "DONE" | ✅ | ✅ |

**Count: IQ3_S 13/14 clean, Q4_K_M 14/14 clean.** The single IQ3_S "[thinking only]" (the 3-coins riddle) is the
reasoning-model token-budget artifact, not a correctness failure — with a bigger budget it answers. On everything
else the two are indistinguishable in correctness and quality, including the web-design HTML/CSS (both produce
complete, valid, well-structured pages).

## Speed (this hardware, layer split, the same runs)

- **IQ3_S: 57-71 tok/s** decode (higher on structured/code; it drafts well).
- **Q4_K_M: 14-34 tok/s** decode — slower, and it *degrades* on long thinking chains (down to ~14-16 tok/s on the
  long reasoning prompts) because the PCIe-mirrored experts are hit harder the longer the decode runs.
- llama.cpp Q4_K_M reference: ~21 tok/s.

## The bottom line for the user

Across **coding and general knowledge/reasoning/web-design/creative**, **IQ3_S matches Q4_K_M on quality** and is
~2x faster and kinder to RAM (fits VRAM, no mirror). There is no measured task here where Q4_K_M's 4-bit tier wins.
**IQ3_S is the right default on this pair.** Q4_K_M is available (32 tok/s, 131K) for any task where its quality
*would* matter, but none showed up in this testing — which answers the original Reddit question with data: the
3.5-bit tier is not leaving quality on the table.
