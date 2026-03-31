"""
Clinical Workflow Optimization Environment — OpenEnv Environment subclass.

Implements the full OpenEnv Environment interface with typed Action/Observation/State.
Simulates a surgical procedure where the agent must balance procedural progress
with patient safety under stochastic complications.
"""

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Observation, State

from models import ClinicalAction, ClinicalObservation, ClinicalState, Vitals


class ClinicalEnvironment(Environment):
    """
    OpenEnv-compliant clinical workflow RL environment.

    The agent controls a surgical procedure simulation, choosing between:
    - ``perform_step``: advance the procedure (risk of complications)
    - ``handle_complication``: address an active complication

    Difficulty levels control complication probability:
    - easy:   10%
    - medium: 40%
    - hard:   70%
    """

    def __init__(self, difficulty: str = "easy"):
        super().__init__()
        self._difficulty = difficulty
        self._complications = ["bleeding", "infection", "tool_failure"]
        self._env_done = False
        self._last_reward = 0.0

        # Internal procedure state
        self._phase = "incision"
        self._progress = 0.0
        self._vitals = {"HR": 80, "BP": 120, "O2": 98}
        self._complication = None
        self._time = 0

        # Episode metadata
        self._episode_id = str(uuid4())
        self._step_count = 0

    # ------------------------------------------------------------------
    # OpenEnv interface: reset()
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ClinicalObservation:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            random.seed(seed)

        self._difficulty = kwargs.get("difficulty", self._difficulty)
        self._env_done = False
        self._last_reward = 0.0
        self._phase = "incision"
        self._progress = 0.0
        self._vitals = {"HR": 80, "BP": 120, "O2": 98}
        self._complication = None
        self._time = 0
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0

        return self._make_observation(reward=0.0)

    # ------------------------------------------------------------------
    # OpenEnv interface: step()
    # ------------------------------------------------------------------
    def step(
        self,
        action: ClinicalAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ClinicalObservation:
        """Execute one step in the environment."""
        self._step_count += 1
        action_str = action.action if isinstance(action, ClinicalAction) else str(action)
        
        # Robust parsing: if user accidentally typed JSON into the UI (e.g. '{"action": "perform_step"}')
        if "{" in action_str and "}" in action_str:
            try:
                import json
                parsed = json.loads(action_str)
                if isinstance(parsed, dict) and "action" in parsed:
                    action_str = parsed["action"]
            except Exception:
                pass
        
        # Clean quotes just in case
        action_str = action_str.strip('\'"')

        reward = 0.0

        # Track whether a complication existed BEFORE this step
        had_complication = self._complication is not None

        # 1. Handle complication (must come first — treat existing issues)
        just_handled = False
        if action_str == "handle_complication":
            if self._complication:
                comp = self._complication
                if comp == "bleeding":
                    reward += 2
                elif comp == "infection":
                    self._vitals["HR"] -= 3
                    reward += 1.5
                elif comp == "tool_failure":
                    reward += 1
                self._complication = None
                just_handled = True
            else:
                # Penalize useless handling when no complication exists
                reward -= 1.5

        # 2. Perform step (advance procedure)
        if action_str == "perform_step":
            if had_complication:
                # Penalty for ignoring an existing complication
                reward -= 2
            else:
                self._progress += 0.1
                reward += 1
                self._progress = min(self._progress, 1.0)

        # 3. Penalty for ignoring a PRE-EXISTING complication
        #    (only if complication was there before this step and not handled)
        if had_complication and self._complication is not None:
            reward -= 4
            if self._complication == "bleeding":
                self._vitals["O2"] -= 2
            elif self._complication == "infection":
                self._vitals["HR"] += 3

        # 4. Time penalty
        self._time += 1
        reward -= 0.1

        # Fatigue effect
        if self._time > 10:
            self._vitals["O2"] -= 1

        # 5. Stochastic complication generation (skip if just handled — grace turn)
        prob = {"easy": 0.1, "medium": 0.4, "hard": 0.7}.get(self._difficulty, 0.1)
        if not self._complication and not just_handled and random.random() < prob:
            comp = random.choice(self._complications)
            self._complication = comp
            reward -= 1
            if comp == "bleeding":
                self._vitals["O2"] -= 5
            elif comp == "infection":
                self._vitals["HR"] += 15
            elif comp == "tool_failure":
                reward -= 2

        # Clamp HR
        self._vitals["HR"] = min(self._vitals["HR"], 160)

        # 6. Failure condition
        if self._vitals["O2"] < 50 or self._vitals["HR"] > 140:
            self._env_done = True
            reward -= 10
            self._progress *= 0.7

        # 7. Success condition
        if self._progress >= 1.0:
            self._env_done = True
            reward += 5

        self._last_reward = reward
        return self._make_observation(reward=reward)

    # ------------------------------------------------------------------
    # OpenEnv interface: state (property)
    # ------------------------------------------------------------------
    @property
    def state(self) -> ClinicalState:
        """Return episode metadata (not the observation)."""
        return ClinicalState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            difficulty=self._difficulty,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_observation(self, reward: float = 0.0) -> ClinicalObservation:
        """Build a typed observation from internal state."""
        return ClinicalObservation(
            phase=self._phase,
            progress=round(self._progress, 4),
            vitals=Vitals(
                HR=self._vitals["HR"],
                BP=self._vitals["BP"],
                O2=self._vitals["O2"],
            ),
            complication=self._complication,
            time=self._time,
            done=self._env_done,
            reward=round(reward, 4),
        )

    def close(self) -> None:
        """Cleanup (no-op for this lightweight env)."""
        pass
