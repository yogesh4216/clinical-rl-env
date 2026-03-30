"""
Typed Pydantic models for the Clinical Workflow Optimization Environment.

Defines the Action, Observation, and State models used by the OpenEnv framework
for type-safe communication between client and server.
"""

from typing import Optional
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Sub-models
# --------------------------------------------------------------------------
class Vitals(BaseModel):
    """Patient vital signs."""
    HR: int = Field(default=80, ge=40, le=160, description="Heart Rate in bpm")
    BP: int = Field(default=120, ge=60, le=200, description="Blood Pressure (systolic) in mmHg")
    O2: int = Field(default=98, ge=0, le=100, description="Oxygen Saturation in %")


# --------------------------------------------------------------------------
# Action — what the agent sends
# --------------------------------------------------------------------------
class ClinicalAction(BaseModel):
    """
    Agent action for the clinical workflow environment.

    The agent must choose one of two discrete actions each step:
    - ``perform_step``: advance the surgical procedure
    - ``handle_complication``: mitigate an active complication
    """
    action: str = Field(
        description="One of: 'perform_step' or 'handle_complication'"
    )
    metadata: dict = Field(default_factory=dict, description="Optional metadata")


# --------------------------------------------------------------------------
# Observation — what the environment returns
# --------------------------------------------------------------------------
class ClinicalObservation(BaseModel):
    """
    Observation returned by the environment after each step or reset.

    Contains the full state of the simulated clinical procedure including
    patient vitals, procedural progress, active complications, and timing.
    """
    phase: str = Field(default="incision", description="Current procedural phase")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Procedure completion (0.0–1.0)")
    vitals: Vitals = Field(default_factory=Vitals, description="Patient vital signs")
    complication: Optional[str] = Field(default=None, description="Active complication or null")
    time: int = Field(default=0, ge=0, description="Elapsed procedural steps")
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: float = Field(default=0.0, description="Reward signal from the last action")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


# --------------------------------------------------------------------------
# State — episode metadata
# --------------------------------------------------------------------------
class ClinicalState(BaseModel):
    """
    Internal environment state / episode metadata.

    Separate from the Observation: this tracks episode-level bookkeeping
    (episode ID, step count, difficulty) rather than the patient/procedure state.
    """
    episode_id: Optional[str] = Field(default=None, description="Unique episode identifier")
    step_count: int = Field(default=0, ge=0, description="Steps taken this episode")
    difficulty: str = Field(default="easy", description="Current difficulty level")
