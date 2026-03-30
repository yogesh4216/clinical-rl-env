"""
FastAPI application for the Clinical Workflow Optimization Environment.

Exposes the ClinicalEnvironment over HTTP and WebSocket endpoints
using the OpenEnv framework's create_app() factory.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os

# Ensure the project root is on the Python path so that `models` and
# `server` are importable regardless of the working directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openenv.core.env_server.http_server import create_app

from models import ClinicalAction, ClinicalObservation
from server.clinical_environment import ClinicalEnvironment

# ------------------------------------------------------------------
# Create the app using OpenEnv's factory
# ------------------------------------------------------------------
app = create_app(
    ClinicalEnvironment,
    ClinicalAction,
    ClinicalObservation,
    env_name="clinical_workflow_env",
)


# ------------------------------------------------------------------
# Additional custom endpoints (tasks, grader, baseline)
# These extend the standard OpenEnv endpoints.
# ------------------------------------------------------------------
import random
from fastapi import Query


@app.get("/tasks")
def get_tasks():
    """Return the list of available tasks (difficulty levels)."""
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Low complication risk (10%); vitals remain stable.",
            },
            {
                "name": "medium",
                "description": "Moderate complication risk (40%); requires vigilant monitoring.",
            },
            {
                "name": "hard",
                "description": "High complication risk (70%); frequent procedural hurdles.",
            },
        ]
    }


def _compute_score(obs_dict: dict) -> float:
    """Compute a grader score (0.0–1.0) from an observation dict."""
    completion = obs_dict.get("progress", 0.0)
    vitals = obs_dict.get("vitals", {})
    o2 = vitals.get("O2", 98)
    time_val = obs_dict.get("time", 0)

    vitals_score = o2 / 100.0
    time_penalty = min(time_val / 20.0, 1.0)

    score = 0.5 * completion + 0.3 * vitals_score + 0.2 * (1.0 - time_penalty)
    return max(0.0, min(1.0, round(score, 4)))


@app.get("/grader")
def grade(task: str = Query(default="easy", description="Task name (easy/medium/hard)")):
    """
    Run a complete episode with the rule-based agent and return the grader score.

    Accepts a ``task`` query parameter to select difficulty.
    Returns a score between 0.0 and 1.0.
    """
    env = ClinicalEnvironment(difficulty=task)
    obs = env.reset()
    done = obs.done

    while not done:
        action_str = (
            "handle_complication" if obs.complication else "perform_step"
        )
        obs = env.step(ClinicalAction(action=action_str))
        done = obs.done

    obs_dict = obs.model_dump()
    score = _compute_score(obs_dict)
    return {"task": task, "score": score}


@app.get("/baseline")
def run_baseline():
    """
    Run rule-based and random agents across all difficulty levels.

    Returns averaged scores (0.0–1.0) per difficulty.
    """

    def _run_episode(policy: str, level: str) -> float:
        env = ClinicalEnvironment(difficulty=level)
        obs = env.reset()
        done = obs.done

        while not done:
            if policy == "rule":
                action_str = (
                    "handle_complication" if obs.complication else "perform_step"
                )
            else:
                action_str = random.choice(["perform_step", "handle_complication"])
            obs = env.step(ClinicalAction(action=action_str))
            done = obs.done

        return _compute_score(obs.model_dump())

    results = {}
    for level in ["easy", "medium", "hard"]:
        rule_scores = [_run_episode("rule", level) for _ in range(5)]
        random_scores = [_run_episode("random", level) for _ in range(5)]
        results[level] = {
            "rule_based": round(sum(rule_scores) / 5, 3),
            "random": round(sum(random_scores) / 5, 3),
        }

    return results


# ------------------------------------------------------------------
# Direct execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
