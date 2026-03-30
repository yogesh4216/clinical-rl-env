import json

class EpisodeLogger:
    def __init__(self):
        self.steps = []

    def log(self, step, state, action, reward):
        self.steps.append({
            "step": step,
            "action": action,
            "state": state,
            "reward": reward
        })

    def save(self, filename="episode.json"):
        with open(filename, "w") as f:
            json.dump(self.steps, f, indent=2)