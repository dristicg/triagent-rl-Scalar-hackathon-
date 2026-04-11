"""
Compatibility bridge for TriageNet models.
Redirects imports to schemas.py (Pydantic-based).
"""
from schemas import (
    Vitals, 
    TriageAction as Action, 
    Observation, 
    Reward, 
    StepResult, 
    EpisodeResult,
    BedStatus,
    InternalPatient as Patient
)

# Compatibility Constants
VALID_ROUTING_ZONES = {
    "resuscitation_bay", "acute_care", "fast_track", "waiting_room", "discharge"
}
VALID_ACTION_TYPES = {
    "assign_esi", "request_resources", "escalate", "reassess", "discharge", "wait"
}
