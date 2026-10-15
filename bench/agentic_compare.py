#!/usr/bin/env python3
"""Hard + agentic quality comparison: tool-calling / agentic tasks and edge-of-capability reasoning.
Run per endpoint:  python3 bench/agentic_compare.py <port> "<tag>"
Identical prompts + temperature 0 per model. Tool-calling uses a real OpenAI tools schema.
Outputs printed in full; structured outputs are checked for validity where applicable."""
import json, sys, urllib.request, re

# A real tool schema for the agentic tests
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "The city name"},
                                      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city"]},
    },
}, {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search the repo for files matching a glob.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]},
    },
}]

# (prompt, kind) — kind 'tool' expects a tool call, 'text' a text answer
PROMPTS = [
    # --- agentic / tool-calling ---
    ("What's the weather in Tokyo right now?", "tool"),                       # single correct tool call
    ("Find all Python test files, then tell me the weather in Paris.", "tool"),  # multi-intent: search_files + get_weather
    ("Is it going to rain in London tomorrow? Use whatever tools you need.", "tool"),  # must NOT call get_weather(city missing? no - city=London)
    # --- hard reasoning ---
    ("Three people check into a hotel room that costs $30. They each pay $10. The clerk realizes the room was $25 and gives $5 to the bellhop to return. The bellhop keeps $2 and gives $1 back to each guest. Now each guest paid $9 (total $27), plus the bellhop's $2 = $29. Where did the missing dollar go? Answer in two sentences.", "text"),
    ("A lily pad doubles in size every day and covers the whole pond on day 48. On what day did it cover half the pond? One line, then one sentence of why.", "text"),
    ("You have 8 balls; one is slightly heavier. Using a balance scale, what is the minimum number of weighings to guarantee you find the heavy one, and how? Two sentences.", "text"),
    # --- hard code ---
    ("Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] intervals. Handle unsorted input and touching intervals ([1,3],[3,5] -> [1,5]). Code only.", "text"),
    ("This is supposed to be a thread-safe singleton but has a race. Fix it, one line of explanation:\nclass S:\n    _i = None\n    def instance():\n        if S._i is None:\n            S._i = S()\n        return S._i", "text"),
    ("Write a Python generator `primes()` that yields primes forever using the Sieve of Eratosthenes incrementally (no fixed upper bound). Code only.", "text"),
]

def call(port, p, use_tools):
    body = {"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 1500, "temperature": 0}
    if use_tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]

def summarize(choice, kind):
    m = choice["message"]
    if kind == "tool":
        tcs = m.get("tool_calls") or []
        if not tcs:
            return "[NO TOOL CALL] " + ((m.get("content") or "")[:120])
        out = []
        for tc in tcs:
            fn = tc.get("function", {})
            args = fn.get("arguments", "")
            # is the arguments JSON valid?
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
                valid = "valid-json"
            except Exception:
                valid, parsed = "INVALID-json", args
            out.append(f"{fn.get('name')}({json.dumps(parsed) if isinstance(parsed,(dict,)) else parsed}) [{valid}]")
        return "TOOL CALLS: " + "; ".join(out)
    out = m.get("content") or ("[thinking only] " + (m.get("reasoning_content") or "")[:150])
    return out

def run(port, tag):
    print(f"\n########## {tag} (port {port}) ##########")
    for p, kind in PROMPTS:
        try:
            choice = call(port, p, use_tools=(kind == "tool"))
            out = summarize(choice, kind)
        except Exception as e:
            out = f"<error: {e}>"
        print(f"\n=== Q ({kind}) ===\n{p}\n=== A ===\n{out}\n")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "8081", sys.argv[2] if len(sys.argv) > 2 else "?")
