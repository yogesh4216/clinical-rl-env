"""
inference.py — OpenEnv Competition Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
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
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment URL — runs in the same container
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Environment name (matches openenv.yaml)
ENV_NAME = "clinical_workflow_env"

# ---------------------------------------------------------------------------
# LLM Client — OpenAI-compatible endpoint (using OpenAI Client as required)
# ---------------------------------------------------------------------------
client = None
if HF_TOKEN:
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


# ---------------------------------------------------------------------------
# Helpers — all informational output to stderr only
# ---------------------------------------------------------------------------
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


def _compute_score(obs: dict) -> float:
    """Compute a grader score (0.0–1.0) from the final observation."""
    completion = obs.get("progress", 0.0)
    vitals = obs.get("vitals", {})
    o2 = vitals.get("O2", 98)
    time_val = obs.get("time", 0)

    vitals_score = o2 / 100.0
    time_penalty = min(time_val / 20.0, 1.0)

    score = 0.5 * completion + 0.3 * vitals_score + 0.2 * (1.0 - time_penalty)
    return max(0.0, min(1.0, round(score, 2)))


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


# ---------------------------------------------------------------------------
# Core episode runner — emits [START] / [STEP] / [END] to stdout
# ---------------------------------------------------------------------------
def run_episode(task_name: str) -> float:
    """Run a single episode for the given task.

    Emits exactly the structured stdout format required by the validator:
      [START] task=<task_name> env=<env_name> model=<model_name>
      [STEP]  step=<n> action=<action> reward=<0.00> done=<bool> error=<msg|null>
      [END]   success=<bool> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
    """
    model_display = MODEL_NAME
    rewards_list: list[float] = []
    step_count = 0
    score_val = 0.0
    success = False
    last_error = None

    # === [START] ===
    print(
        f"[START] task={task_name} env={ENV_NAME} model={model_display}",
        flush=True,
    )

    try:
        # 1. Reset environment
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": task_name},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

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
            reward = float(result.get("reward", 0.0))
            done = result.get("done", False)
            step_count += 1
            rewards_list.append(reward)

            # === [STEP] ===
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_count} action={action_str} "
                f"reward={reward:.2f} done={done_str} error=null",
                flush=True,
            )

        # 3. Compute score from final observation
        score_val = _compute_score(obs)
        success = True

    except Exception as e:
        last_error = str(e).replace("\n", " ")
        _log(f"  [ERROR] {task_name}: {last_error}")

    # === [END] — always emitted, even on exception ===
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list) if rewards_list else "0.00"
    print(
        f"[END] success={success_str} steps={step_count} "
        f"score={score_val:.2f} rewards={rewards_str}",
        flush=True,
    )

    return score_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Evaluate the agent across all difficulty levels."""
    _log("=" * 60)
    _log("  Clinical Workflow RL — OpenEnv Inference")
    _log("=" * 60)
    _log(f"  API_BASE_URL : {API_BASE_URL}")
    _log(f"  MODEL_NAME   : {MODEL_NAME}")
    _log(f"  HF_TOKEN     : {'***' if HF_TOKEN else '(not set — rule-based)'}")
    _log(f"  ENV_URL      : {ENV_URL}")
    _log("=" * 60)

    # Wait for the environment server to be ready
    if not _wait_for_server(ENV_URL, timeout=60):
        _log(f"  [WARN] Server at {ENV_URL} not reachable, proceeding anyway...")

    # Enumerate tasks from the /tasks endpoint
    task_names = ["easy", "medium", "hard"]  # fallback
    try:
        resp = requests.get(f"{ENV_URL}/tasks", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            task_names = [t["name"] for t in data.get("tasks", [])]
    except Exception:
        pass

    _log(f"  Tasks: {task_names}")

    # Run one episode per task
    for task in task_names:
        score = run_episode(task)
        _log(f"  [{task}] Score: {score:.2f}")

    _log("=" * 60)
    _log("  Inference complete.")
    _log("=" * 60)


if __name__ == "__main__":
    main()
