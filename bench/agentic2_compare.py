#!/usr/bin/env python3
"""Agentic + long-context quality comparison.

Agentic: many tools, distractors, parallel calls, 'no tool needed', schema constraints.
Long context: needle-in-a-haystack (a fact buried at a target depth of the context window).

Run per endpoint:  python3 bench/agentic2_compare.py <port> "<tag>" [needle_depths]
Identical prompts/settings per model. temperature 0."""
import json, sys, urllib.request, random, string

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Find files matching a glob.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file's contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the test suite for a path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate a math expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_stock_price", "description": "Current stock price for a ticker.",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
]

AGENTIC = [
    # (prompt, note on what's correct)
    ("What's the weather in Tokyo?", "one get_weather, city=Tokyo"),
    ("What is 124 * 456 + 789?", "should call calculate, NOT answer directly"),
    ("Find the test files and then run them.", "search_files then run_tests (dependency)"),
    ("What's the weather in Paris in fahrenheit and the stock price of AAPL?", "TWO independent calls, parallel"),
    ("Read config.yaml and tell me the weather there.", "read_file (the city is ambiguous -> may still call weather)"),
    ("Explain how photosynthesis works.", "NO tool fits: should answer in text, no tool call"),
    ("Search for all markdown files.", "one search_files, pattern like **/*.md"),
    ("What is the weather in the city where the Eiffel Tower is?", "needs reasoning: Paris -> get_weather(Paris)"),
]

def needle_text(depth_frac, total_tokens_est, secret, question):
    """Build filler with the secret planted at depth_frac of the way through."""
    words = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
             "incididunt ut labore et dolore magna aliqua enim ad minim veniam").split()
    n = total_tokens_est
    insert_at = int(n * depth_frac)
    out, i = [], 0
    while len(out) < n:
        if len(out) == insert_at:
            out.append(secret)
        out.append(words[i % len(words)]); i += 1
    return " ".join(out)

def call(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]

def tool_summary(m):
    tcs = m.get("tool_calls") or []
    if not tcs:
        return "[NO TOOL CALL] " + ((m.get("content") or "")[:100])
    parts = []
    for tc in tcs:
        fn = tc.get("function", {}); args = fn.get("arguments", "")
        try: parsed = json.loads(args) if isinstance(args, str) else args; valid = "ok"
        except Exception: valid, parsed = "BAD-JSON", args
        parts.append(f"{fn.get('name')}({parsed})[{valid}]")
    return "; ".join(parts)

def run_agentic(port, tag):
    print(f"\n########## {tag}: AGENTIC (port {port}) ##########")
    for p, note in AGENTIC:
        body = {"model": "m", "messages": [{"role": "user", "content": p}],
                "max_tokens": 800, "temperature": 0, "tools": TOOLS, "tool_choice": "auto"}
        try: out = tool_summary(call(port, body)["message"])
        except Exception as e: out = f"<error: {e}>"
        print(f"\nQ: {p}\n   (correct: {note})\nA: {out}\n")

def run_needle(port, tag, depths):
    print(f"\n########## {tag}: NEEDLE-IN-HAYSTACK (port {port}) ##########")
    secret = "The secret code is XJ-9942-QUAIL."
    q = "What is the secret code mentioned in this document? Answer with the code only."
    # ~60k tokens of filler (fits the window with room); ~0.8 words/token
    total_words = 75000
    for d in depths:
        secret_tok = f"IMPORTANT: {secret}"
        text = needle_text(d, total_words, secret_tok, q)
        prompt = text + "\n\n" + q
        body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200, "temperature": 0}
        try:
            m = call(port, body)["message"]
            out = m.get("content") or ("[thinking only] " + (m.get("reasoning_content") or "")[:100])
            found = "XJ-9942-QUAIL" in out
            out = f"{'FOUND' if found else 'NOT FOUND'}: {out[:120]}"
        except Exception as e:
            out = f"<error: {e}>"
        print(f"\n  needle at {int(d*100)}% depth -> {out}")

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "8081"
    tag = sys.argv[2] if len(sys.argv) > 2 else "?"
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"
    if mode in ("all", "agentic"): run_agentic(port, tag)
    if mode in ("all", "needle"): run_needle(port, tag, [0.1, 0.5, 0.9, 0.98])
