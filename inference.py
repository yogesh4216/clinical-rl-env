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
import sys
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "<your-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-model-name>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment URL — runs in the same container
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# LLM Client — OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
client = None
if HF_TOKEN and API_BASE_URL and not API_BASE_URL.startswith("<"):
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


def _log(msg: str) -> None:
    """Log informational messages to stderr so they don't pollute stdout."""
    print(msg, file=sys.stderr, flush=True)


def ask_llm(state: dict) -> str:
    """Query the LLM for the next action given the current state."""
    if client is None:
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

        if "handle_complication" in raw:
            return "handle_complication"
        return "perform_step"

    except Exception as e:
        _log(f"  [LLM fallback] {e}")
        return _rule_based(state)


def _rule_based(state: dict) -> str:
    """Deterministic rule-based fallback policy."""
    if state.get("complication"):
        return "handle_complication"
    return "perform_step"


def _wait_for_server(url: str, timeout: int = 60) -> bool:
    """Wait for the environment server to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/tasks", timeout=5)
            if resp.status_code == 200:
                _log(f"  Server ready at {url}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass
        time.sleep(2)
    return False


def run_episode(difficulty: str, ep_num: int) -> float:
    """Run a single episode and return the grader score (0.0–1.0)."""
    task_name = f"{difficulty}_ep{ep_num}"

    # Always print [START] — this is required by the validator
    print(f"[START] task={task_name}", flush=True)

    score_val = 0.0
    step_count = 0

    try:
        # 1. Reset — POST /reset with difficulty
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": difficulty},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        # The OpenEnv ResetResponse has: observation, reward, done
        obs = result.get("observation", result)
        done = result.get("done", False)
        max_steps = 50

        # 2. Step loop
        while not done and step_count < max_steps:
            action_str = ask_llm(obs)

            resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": {"action": action_str}},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            obs = result.get("observation", result)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            step_count += 1
            print(f"[STEP] step={step_count} reward={reward}", flush=True)

        # 3. Grade — GET /grader?task=difficulty
        resp = requests.get(
            f"{ENV_URL}/grader",
            params={"task": difficulty},
            timeout=30,
        )
        resp.raise_for_status()
        score = resp.json()
        score_val = score.get("score", 0.0)

    except Exception as e:
        _log(f"  [ERROR] Episode {task_name} failed: {e}")

    # Always print [END] — this is required by the validator
    print(f"[END] task={task_name} score={score_val} steps={step_count}", flush=True)

    return score_val


def main():
    """Evaluate the agent across all difficulty levels and print final scores."""
    _log("=" * 60)
    _log("  Clinical Workflow RL — OpenEnv Inference")
    _log("=" * 60)
    _log(f"  API_BASE_URL : {API_BASE_URL}")
    _log(f"  MODEL_NAME   : {MODEL_NAME}")
    _log(f"  HF_TOKEN     : {'***' if HF_TOKEN else '(not set)'}")
    _log(f"  ENV_URL      : {ENV_URL}")
    _log("=" * 60)

    # Wait for the environment server to be ready
    if not _wait_for_server(ENV_URL, timeout=60):
        _log(f"  [WARN] Server at {ENV_URL} not reachable, proceeding anyway...")

    levels = ["easy", "medium", "hard"]
    num_episodes = 5
    results = {}

    for level in levels:
        scores = []
        for ep in range(num_episodes):
            score = run_episode(level, ep + 1)
            scores.append(score)
            _log(f"  [{level:6s}] Episode {ep+1}/{num_episodes} -> Score: {score:.3f}")

        avg = sum(scores) / len(scores)
        results[level] = round(avg, 4)
        _log(f"  [{level:6s}] Average Score: {avg:.4f}\n")

    # Final summary to stderr (not stdout)
    _log("=" * 60)
    _log("  FINAL SCORES")
    _log("=" * 60)
    for level in levels:
        _log(f"  {level:6s}: {results[level]}")
    _log("=" * 60)

    return results


if __name__ == "__main__":
    main()
