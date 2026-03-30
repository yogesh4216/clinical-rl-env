"""
inference.py — OpenEnv Competition Inference Script

Evaluates an LLM-based agent on the Clinical Workflow Optimization
Environment across Easy, Medium, and Hard difficulty levels.

Required Environment Variables:
    API_BASE_URL  — The API endpoint for the LLM (OpenAI-compatible)
    MODEL_NAME    — The model identifier to use for inference
    HF_TOKEN      — Hugging Face / API key for authentication

The environment is assumed to be running locally in the same container
on port 7860 (Hugging Face Spaces default).
"""

import os
import json
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Environment URL — runs in the same container
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# LLM Client — OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
client = None
if API_BASE_URL and HF_TOKEN:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

SYSTEM_PROMPT = """You are a clinical AI agent controlling a surgical procedure simulation.

At each step you receive the current patient state (vitals, progress, complications, time).
You must respond with EXACTLY one of two JSON actions:
  {"action": "perform_step"}       — advance the procedure
  {"action": "handle_complication"} — address an active complication

Decision rules:
- If there is an active complication (complication is not null), you MUST handle it first.
- If vitals are critical (O2 < 70 or HR > 120), prioritise stabilisation.
- Otherwise, advance the procedure.

Respond with ONLY the JSON object, no extra text."""


def ask_llm(state: dict) -> str:
    """Query the LLM for the next action given the current state."""
    if client is None:
        # Fallback to rule-based
        return _rule_based(state)

    user_message = f"Current patient state:\n{json.dumps(state, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=50,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Parse the action from the LLM response
        if "handle_complication" in raw:
            return "handle_complication"
        return "perform_step"

    except Exception as e:
        print(f"  [LLM fallback] {e}")
        return _rule_based(state)


def _rule_based(state: dict) -> str:
    """Deterministic rule-based fallback policy."""
    if state.get("complication"):
        return "handle_complication"
    return "perform_step"


def run_episode(difficulty: str) -> float:
    """Run a single episode and return the grader score (0.0–1.0)."""

    # 1. Reset — POST /reset with difficulty
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"difficulty": difficulty},
    )
    resp.raise_for_status()
    result = resp.json()

    # The OpenEnv ResetResponse has: observation, reward, done
    obs = result.get("observation", result)
    done = result.get("done", False)
    step_count = 0
    max_steps = 50

    # 2. Step loop
    while not done and step_count < max_steps:
        action_str = ask_llm(obs)

        resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": {"action": action_str}},
        )
        resp.raise_for_status()
        result = resp.json()

        obs = result.get("observation", result)
        done = result.get("done", False)
        step_count += 1

    # 3. Grade — GET /grader?task=difficulty
    resp = requests.get(f"{ENV_URL}/grader", params={"task": difficulty})
    resp.raise_for_status()
    score = resp.json()["score"]

    return score


def main():
    """Evaluate the agent across all difficulty levels and print final scores."""
    print("=" * 60)
    print("  Clinical Workflow RL — OpenEnv Inference")
    print("=" * 60)
    print(f"  API_BASE_URL : {API_BASE_URL or '(not set — rule-based fallback)'}")
    print(f"  MODEL_NAME   : {MODEL_NAME or '(not set)'}")
    print(f"  HF_TOKEN     : {'***' if HF_TOKEN else '(not set)'}")
    print(f"  ENV_URL      : {ENV_URL}")
    print("=" * 60)

    levels = ["easy", "medium", "hard"]
    num_episodes = 5
    results = {}

    for level in levels:
        scores = []
        for ep in range(num_episodes):
            score = run_episode(level)
            scores.append(score)
            print(f"  [{level:6s}] Episode {ep+1}/{num_episodes} → Score: {score:.3f}")

        avg = sum(scores) / len(scores)
        results[level] = round(avg, 4)
        print(f"  [{level:6s}] Average Score: {avg:.4f}\n")

    # Final output
    print("=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for level in levels:
        print(f"  {level:6s}: {results[level]}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
