#!/usr/bin/env python3
"""Quality comparison: run a fixed prompt set through several endpoints and print the outputs side by side.

Endpoints (positional none; edit PORTS): each is an OpenAI-compatible /v1/chat/completions.
  llama.cpp Q4_K_M (reference)  :8082
  Strata Q4_K_M                 :8081 (when its config is the q4_k_m split)
  Strata IQ3_S                  :8081 (when its config is the iq3_s split)
Run one endpoint's set at a time (Strata serves one model at a time), tag the run with --tag.
"""
import json, sys, urllib.request

PROMPTS = [
    # reasoning / correctness
    "A farmer has 17 sheep. All but 9 die. How many are left? Answer with the number and one short sentence.",
    "If a train travels 60 km in 45 minutes, what is its average speed in km/h? One line.",
    # code
    "Write a Python function `is_prime(n)` that returns True if n is prime. Code only.",
    "Write a SQL query to find the second-highest salary from an `employees` table with columns (id, salary). SQL only.",
    # knowledge / factual
    "What is the capital of Australia? One word.",
    "In one sentence, what does the CAP theorem state?",
    # instruction following / format
    "List exactly three colors, comma-separated, nothing else.",
    # summarization / language
    "Summarize in one sentence: The mitochondria is the powerhouse of the cell, generating most of the cell's supply of adenosine triphosphate (ATP) through oxidative phosphorylation, and it has its own DNA.",
]

def run(port, tag):
    print(f"\n########## {tag} (port {port}) ##########")
    for p in PROMPTS:
        body = {"model": "m", "messages": [{"role": "user", "content": p}],
                "max_tokens": 220, "temperature": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            msg = r["choices"][0]["message"]
            out = msg.get("content") or msg.get("reasoning_content") or ""
            out = " ".join(out.split())  # one line
        except Exception as e:
            out = f"<error: {e}>"
        print(f"\nQ: {p}\nA: {out}\n")

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "8082"
    tag = sys.argv[2] if len(sys.argv) > 2 else f"port {port}"
    run(port, tag)
