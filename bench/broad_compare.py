#!/usr/bin/env python3
"""Broad quality comparison: general knowledge, reasoning, explanation, creative/web, instruction-following.
Run per endpoint:  python3 bench/broad_compare.py <port> "<tag>"
Identical prompts + temperature 0 across models (apples-to-apples). Outputs printed in full for review."""
import json, sys, urllib.request

PROMPTS = [
    # general knowledge / factual
    "What is the capital of Australia, and name one thing it is known for. Two sentences.",
    "Explain the difference between a virus and a bacterium in one sentence each.",
    "Which planet has the most moons, and roughly how many? One sentence.",
    # reasoning / logic (the classic traps)
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? One line.",
    "If you have 3 coins and one is not a dime, but all are US coins totaling 30 cents, what are they? One line.",
    "Is 0.9 repeating (0.999...) equal to 1? Yes or no, with a one-line justification.",
    # explanation / teaching
    "Explain recursion to a 10-year-old in two sentences.",
    "What is the difference between TCP and UDP, in one sentence each?",
    # web design / front-end (the 'web design' ask)
    "Write the HTML+CSS for a centered card with a title, a paragraph, and a rounded blue button, using flexbox. Code only.",
    "Write CSS to make a responsive two-column layout that stacks on screens under 600px. Code only.",
    # creative / writing
    "Write a two-sentence story about a robot who learns to paint.",
    "Give me a catchy tagline for a coffee shop that opens at night. One line.",
    # instruction following / format
    "List the planets in order from the sun, comma-separated, nothing else.",
    "Reply with only the word 'DONE' and nothing else.",
]

def run(port, tag):
    print(f"\n########## {tag} (port {port}) ##########")
    for p in PROMPTS:
        body = {"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 1200, "temperature": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=240))
            msg = r["choices"][0]["message"]
            out = msg.get("content") or ""
            if not out and msg.get("reasoning_content"):
                out = "[thinking only] " + msg["reasoning_content"][:150]
        except Exception as e:
            out = f"<error: {e}>"
        print(f"\n=== Q ===\n{p}\n=== A ===\n{out}\n")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "8081", sys.argv[2] if len(sys.argv) > 2 else "?")
