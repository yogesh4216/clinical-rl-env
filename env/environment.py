import random

class ClinicalEnv:
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty
        self.complications = ["bleeding", "infection", "tool_failure"]
        self.reset()

    def reset(self):
        self.state = {
            "phase": "incision",
            "progress": 0.0,
            "vitals": {"HR": 80, "BP": 120, "O2": 98},
            "complication": None,
            "time": 0
        }
        self.done = False
        return self.state

    def step(self, action):
        reward = 0

        # -------------------------
        # 1. Perform step
        # -------------------------
        if action == "perform_step":

            # ❌ Do NOT allow progress if complication exists
            if self.state["complication"]:
                reward -= 2  # strong penalty
            else:
                self.state["progress"] += 0.1
                reward += 1
                self.state["progress"] = min(self.state["progress"], 1.0)

        # -------------------------
        # 2. Time penalty
        # -------------------------
        self.state["time"] += 1
        reward -= 0.1

        # fatigue effect (cumulative damage)
        if self.state["time"] > 10:
            self.state["vitals"]["O2"] -= 1

        # -------------------------
        # 3. Difficulty control
        # -------------------------
        if self.difficulty == "easy":
            prob = 0.1
        elif self.difficulty == "medium":
            prob = 0.4
        else:
            prob = 0.7

        # -------------------------
        # 4. Generate complication
        # -------------------------
        if random.random() < prob:
            comp = random.choice(self.complications)
            self.state["complication"] = comp
            reward -= 1  
            if comp == "bleeding":
                self.state["vitals"]["O2"] -= 5

            elif comp == "infection":
                self.state["vitals"]["HR"] += 15

            elif comp == "tool_failure":
                reward -= 2

        # -------------------------
        # 5. Handle complication
        # -------------------------
        if action == "handle_complication" and self.state["complication"]:
            comp = self.state["complication"]

            if comp == "bleeding":
                reward += 2

            elif comp == "infection":
                self.state["vitals"]["HR"] -= 3  # partial recovery
                reward += 1.5

            elif comp == "tool_failure":
                reward += 1

            self.state["complication"] = None

            # ❌ Penalize useless handling (no complication present)
        if action == "handle_complication" and not self.state["complication"]:
            reward -= 1.5

        # -------------------------
        # 6. Penalty if ignored
        # -------------------------
        if self.state["complication"] and action != "handle_complication":
            reward -= 4

            if self.state["complication"] == "bleeding":
                self.state["vitals"]["O2"] -= 2

            elif self.state["complication"] == "infection":
                self.state["vitals"]["HR"] += 3

        self.state["vitals"]["HR"] = min(self.state["vitals"]["HR"], 160)

        # -------------------------
        # 7. Failure condition
        # -------------------------
        if self.state["vitals"]["O2"] < 50 or self.state["vitals"]["HR"] > 140:
            self.done = True

            # 🔥 Strong failure penalty
            reward -= 10

            # Reduce progress to reflect failed procedure
            self.state["progress"] *= 0.7

        # -------------------------
        # 8. Success condition
        # -------------------------
        if self.state["progress"] >= 1.0:
            self.done = True
            reward += 5

        return self.state, reward, self.done, {}

    def state_fn(self):
        return self.state