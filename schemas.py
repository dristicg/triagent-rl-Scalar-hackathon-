from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional


class Vitals(BaseModel):
    bp_systolic: int
    bp_diastolic: int
    heart_rate: int
    respiratory_rate: int
    spo2: float
    temperature: float
    gcs: int
    pain_scale: int


class BedStatus(BaseModel):
    total_beds: int
    available_beds: int
    resuscitation_bays: int
    resuscitation_available: int
    acute_care_beds: int
    acute_care_available: int


class PatientInfo(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    chief_complaint: str
    vitals: Vitals
    medical_history: List[str] = []
    current_medications: List[str] = []
    arrival_time_minutes: int = 0
    wait_time_minutes: int = 0
    deteriorated: bool = False

class InternalPatient(PatientInfo):
    """Internal model including ground truth (for environment state)."""
    true_esi: int
    true_routing: str
    assigned_esi: Optional[int] = None
    assigned_zone: Optional[str] = None
    resources: List[str] = []
    discharged: bool = False


class Observation(BaseModel):
    task_id: int
    step: int
    max_steps: int
    current_patient: Optional[Dict[str, Any]] = None
    pending_patients: List[str] = []
    triaged_patients: List[str] = []
    bed_status: Dict[str, int]
    time_elapsed_minutes: int
    alerts: List[str] = []
    task_description: str
    score_so_far: float


class TriageAction(BaseModel):
    reasoning: str = Field(default="", description="Briefly explain your step-by-step clinical logic, time management, and bed availability reasoning before deciding on the action.")
    action_type: str = Field(..., description="Type of triage action")
    patient_id: str = Field(..., description="ID of the patient being triaged")
    esi_level: Optional[int] = Field(None, ge=1, le=5)
    routing_zone: Optional[str] = None
    resources: Optional[List[str]] = None
    notes: Optional[str] = None

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        valid_types = {"assign_esi", "request_resources", "escalate", "reassess", "discharge", "wait"}
        if v not in valid_types:
            raise ValueError(f"Action type must be one of {valid_types}")
        return v


class Reward(BaseModel):
    score: float = Field(..., description="Numerical reward score")
    feedback: str = Field(..., description="Textual feedback for the action")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Detailed reward breakdown")

    @property
    def step_reward(self) -> float:
        """Compatibility property for OpenEnv validator."""
        return self.score

class StepResult(BaseModel):
    """Compatibility model for the StepResult expected by the validator."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

class EpisodeResult(BaseModel):
    """Compatibility model for grading results."""
    task_id: int
    final_score: float
    total_reward: float
    steps_taken: int
    correct_esi_count: int
    total_patients: int
    critical_misses: int
    deteriorations: int
    grade_breakdown: Dict[str, float]
    passed: bool
