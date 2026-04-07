"""
Typed models for the Medical Triage OpenEnv environment.
Uses Python dataclasses (stdlib only — no pydantic required).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VALID_ROUTING_ZONES = {
    "resuscitation_bay", "acute_care", "fast_track", "waiting_room", "discharge"
}
VALID_ACTION_TYPES = {
    "assign_esi", "request_resources", "escalate", "reassess", "discharge", "wait"
}
VALID_RESOURCES = {"ecg", "iv_access", "blood_panel", "chest_xray", "ct_head"}


@dataclass
class Vitals:
    bp_systolic: int
    bp_diastolic: int
    heart_rate: int
    respiratory_rate: int
    spo2: float
    temperature: float
    gcs: int
    pain_scale: int

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ["bp_systolic","bp_diastolic","heart_rate","respiratory_rate",
                 "spo2","temperature","gcs","pain_scale"]}


@dataclass
class Patient:
    patient_id: str
    name: str
    age: int
    gender: str
    chief_complaint: str
    vitals: Vitals
    true_esi: int
    true_routing: str
    medical_history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    arrival_time_minutes: int = 0
    assigned_esi: Optional[int] = None
    assigned_routing: Optional[str] = None
    resources_requested: List[str] = field(default_factory=list)
    wait_time_minutes: int = 0
    deteriorated: bool = False
    discharged: bool = False

    def to_agent_dict(self):
        """Returns patient info visible to the agent (hides ground truth)."""
        return {
            "patient_id": self.patient_id, "name": self.name, "age": self.age,
            "gender": self.gender, "chief_complaint": self.chief_complaint,
            "vitals": self.vitals.to_dict(),
            "medical_history": self.medical_history,
            "current_medications": self.current_medications,
            "arrival_time_minutes": self.arrival_time_minutes,
            "wait_time_minutes": self.wait_time_minutes,
            "deteriorated": self.deteriorated,
        }


@dataclass
class BedStatus:
    total_beds: int
    available_beds: int
    resuscitation_bays: int
    resuscitation_available: int
    acute_care_beds: int
    acute_care_available: int

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ["total_beds","available_beds","resuscitation_bays",
                 "resuscitation_available","acute_care_beds","acute_care_available"]}


@dataclass
class Observation:
    task_id: int
    step: int
    max_steps: int
    current_patient: Optional[Dict[str, Any]]
    pending_patients: List[str]
    triaged_patients: List[str]
    bed_status: Dict[str, Any]
    time_elapsed_minutes: int
    alerts: List[str]
    task_description: str
    score_so_far: float

    def model_dump(self):
        return {
            "task_id": self.task_id, "step": self.step, "max_steps": self.max_steps,
            "current_patient": self.current_patient,
            "pending_patients": self.pending_patients,
            "triaged_patients": self.triaged_patients,
            "bed_status": self.bed_status,
            "time_elapsed_minutes": self.time_elapsed_minutes,
            "alerts": self.alerts, "task_description": self.task_description,
            "score_so_far": self.score_so_far,
        }


@dataclass
class Action:
    action_type: str
    patient_id: str
    esi_level: Optional[int] = None
    routing_zone: Optional[str] = None
    resources: Optional[List[str]] = None
    notes: Optional[str] = None
    escalate_reason: Optional[str] = None

    def __post_init__(self):
        if self.action_type not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action_type: {self.action_type}")
        if self.esi_level is not None and not (1 <= self.esi_level <= 5):
            raise ValueError(f"esi_level must be 1-5, got {self.esi_level}")
        if self.routing_zone is not None and self.routing_zone not in VALID_ROUTING_ZONES:
            raise ValueError(f"Invalid routing_zone: {self.routing_zone}")


@dataclass
class Reward:
    step_reward: float
    cumulative_reward: float
    breakdown: Dict[str, float]
    feedback: str


@dataclass
class StepResult:
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


@dataclass
class EpisodeResult:
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
