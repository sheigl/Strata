#!/usr/bin/env python3
"""Coding-focused quality comparison: real programming tasks through an OpenAI-compatible endpoint.
Run per endpoint:  python3 bench/coding_compare.py <port> "<tag>"
The point is correctness + usefulness of the code, so outputs are printed in full for review."""
import json, sys, urllib.request

PROMPTS = [
    # algorithm + correctness
    "Write a Python function `binary_search(arr, target)` on a sorted list returning the index or -1. Code only.",
    # bug fixing (give it a bug)
    "This Python is supposed to dedupe a list preserving order but mutates wrong. Fix it and explain in one line:\n"
    "def dedupe(xs):\n    for x in xs:\n        if xs.count(x) > 1: xs.remove(x)\n    return xs",
    # a real implementation task
    "Implement a Python LRU cache class with get(key) and put(key, value) in O(1), using a dict + doubly linked list. Code only.",
    # SQL
    "Given tables orders(id, customer_id, total) and customers(id, name), write SQL for the top 5 customers by total spend. SQL only.",
    # concurrency / harder reasoning
    "Write a Python async function that fetches a list of URLs concurrently with a limit of 5 at a time using asyncio and aiohttp. Code only.",
    # debugging reasoning
    "Why does this print '6 6 6' instead of '0 1 2', and how do you fix it? One short paragraph:\n"
    "fns = [lambda: i for i in range(3)]\nprint(*[f() for f in fns])",
]

def run(port, tag):
    print(f"\n########## {tag} (port {port}) ##########")
    for p in PROMPTS:
        body = {"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 1500, "temperature": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=240))
            msg = r["choices"][0]["message"]
            out = msg.get("content") or ""
            if not out and msg.get("reasoning_content"):
                out = "[thinking only, no final answer] " + msg["reasoning_content"][:200]
        except Exception as e:
            out = f"<error: {e}>"
        print(f"\n=== Q ===\n{p}\n=== A ===\n{out}\n")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "8081", sys.argv[2] if len(sys.argv) > 2 else "?")
