from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import ClinicalEnv
from env.grader import compute_score
import random

app = FastAPI()

# Global env
env = ClinicalEnv()
current_state = None

# -------------------------
# Models
# -------------------------
class ActionInput(BaseModel):
    action: str


# -------------------------
# Reset Endpoint
# -------------------------
@app.post("/reset")
def reset_env(difficulty: str = "easy"):
    global env, current_state
    env = ClinicalEnv(difficulty=difficulty)
    current_state = env.reset()
    return current_state


# -------------------------
# Step Endpoint
# -------------------------
@app.post("/step")
def step_env(action_input: ActionInput):
    global current_state

    state, reward, done, _ = env.step(action_input.action)
    current_state = state

    return {
        "state": state,
        "reward": reward,
        "done": done
    }


# -------------------------
# State Endpoint
# -------------------------
@app.get("/state")
def get_state():
    if current_state is None:
        return {"error": "Environment not initialized. Call /reset first."}
    return current_state


# -------------------------
# Grader Endpoint
# -------------------------
@app.get("/grader")
def grade():
    score = compute_score(current_state)
    return {"score": score}


# -------------------------
# Tasks Endpoint
# -------------------------
@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {"name": "easy", "description": "No complications"},
            {"name": "medium", "description": "Moderate complications"},
            {"name": "hard", "description": "Frequent complications"}
        ]
    }


# -------------------------
# Baseline (FINAL VERSION)
# -------------------------
@app.get("/baseline")
def run_baseline():

    def run_episode(policy, level):
        env_local = ClinicalEnv(difficulty=level)
        state = env_local.reset()
        done = False

        while not done:
            if policy == "rule":
                if state["complication"]:
                    action = "handle_complication"
                else:
                    action = "perform_step"
            else:
                action = random.choice(["perform_step", "handle_complication"])

            state, reward, done, _ = env_local.step(action)

        return compute_score(state)

    results = {}

    for level in ["easy", "medium", "hard"]:
        rule_scores = []
        random_scores = []

        for _ in range(5):  # averaging
            rule_scores.append(run_episode("rule", level))
            random_scores.append(run_episode("random", level))

        results[level] = {
            "rule_based": round(sum(rule_scores)/5, 3),
            "random": round(sum(random_scores)/5, 3)
        }

    return results