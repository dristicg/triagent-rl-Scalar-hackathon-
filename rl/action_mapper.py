"""
Action mapper for converting discrete actions to structured Action objects.
Handles 11 discrete actions with action masking for valid moves.
"""

import os
import sys
import numpy as np
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action


class ActionMapper:
    """Maps discrete actions to structured Action objects with masking."""
    
    def __init__(self):
        self.action_descriptions = {
            0: "assign_esi=1 → resuscitation_bay",
            1: "assign_esi=2 → acute_care",
            2: "assign_esi=3 → fast_track",
            3: "assign_esi=4 → waiting_room",
            4: "assign_esi=5 → waiting_room",
            5: "request_resources=[ecg, iv_access, blood_panel]",
            6: "request_resources=[chest_xray, blood_panel]",
            7: "escalate",
            8: "reassess",
            9: "discharge",
            10: "wait"
        }
    
    def int_to_action(self, action_int: int, patient_id: str) -> Action:
        """Convert integer action to Action object."""
        if action_int == 0:
            return Action(
                action_type="assign_esi",
                patient_id=patient_id,
                esi_level=1,
                routing_zone="resuscitation_bay"
            )
        elif action_int == 1:
            return Action(
                action_type="assign_esi",
                patient_id=patient_id,
                esi_level=2,
                routing_zone="acute_care"
            )
        elif action_int == 2:
            return Action(
                action_type="assign_esi",
                patient_id=patient_id,
                esi_level=3,
                routing_zone="fast_track"
            )
        elif action_int == 3:
            return Action(
                action_type="assign_esi",
                patient_id=patient_id,
                esi_level=4,
                routing_zone="waiting_room"
            )
        elif action_int == 4:
            return Action(
                action_type="assign_esi",
                patient_id=patient_id,
                esi_level=5,
                routing_zone="waiting_room"
            )
        elif action_int == 5:
            return Action(
                action_type="request_resources",
                patient_id=patient_id,
                resources=["ecg", "iv_access", "blood_panel"]
            )
        elif action_int == 6:
            return Action(
                action_type="request_resources",
                patient_id=patient_id,
                resources=["chest_xray", "blood_panel"]
            )
        elif action_int == 7:
            return Action(
                action_type="escalate",
                patient_id=patient_id,
                escalate_reason="Critical case escalation"
            )
        elif action_int == 8:
            return Action(
                action_type="reassess",
                patient_id=patient_id
            )
        elif action_int == 9:
            return Action(
                action_type="discharge",
                patient_id=patient_id
            )
        elif action_int == 10:
            return Action(
                action_type="wait",
                patient_id=patient_id
            )
        else:
            raise ValueError(f"Invalid action_int: {action_int}. Must be 0-10.")
    
    def get_action_mask(self, obs_dict: dict) -> np.ndarray:
        """Get boolean mask of valid actions for current observation."""
        mask = np.ones(11, dtype=bool)
        
        current_patient = obs_dict.get("current_patient")
        pending_patients = obs_dict.get("pending_patients", [])
        triaged_patients = obs_dict.get("triaged_patients", [])
        
        # No patient available - only wait is valid
        if not current_patient or not pending_patients:
            mask[:10] = False
            mask[10] = True  # wait
            return mask
        
        patient_id = current_patient.get("patient_id", "")
        
        # ESI assignment actions (0-4) - only if patient not triaged
        if patient_id in triaged_patients:
            mask[0:5] = False
        
        # Resource request actions (5-6) - always valid for existing patient
        # No restrictions
        
        # Escalate action (7) - always valid for existing patient
        # No restrictions
        
        # Reassess action (8) - valid if wait time >= 10 or deteriorated
        wait_time = current_patient.get("wait_time_minutes", 0)
        deteriorated = current_patient.get("deteriorated", False)
        if wait_time < 10 and not deteriorated:
            mask[8] = False
        
        # Discharge action (9) - always valid for existing patient
        # No restrictions
        
        # Wait action (10) - always valid
        # No restrictions
        
        return mask
    
    def sample_valid(self, obs_dict: dict, rng: Optional[np.random.Generator] = None) -> int:
        """Sample a valid action from the masked action space."""
        if rng is None:
            rng = np.random.default_rng()
        
        mask = self.get_action_mask(obs_dict)
        valid_actions = np.where(mask)[0]
        
        if len(valid_actions) == 0:
            # Fallback to wait if no valid actions
            return 10
        
        return rng.choice(valid_actions)
    
    def action_space_size(self) -> int:
        """Return the size of the discrete action space."""
        return 11
    
    def describe(self, action_int: int) -> str:
        """Get human-readable description of action."""
        # Convert numpy array to int if needed
        if isinstance(action_int, np.ndarray):
            action_int = int(action_int.item()) if action_int.size == 1 else int(action_int[0])
        return self.action_descriptions.get(action_int, f"Unknown action {action_int}")


if __name__ == "__main__":
    # Test the action mapper
    mapper = ActionMapper()
    
    print("Action Mapper Test:")
    print(f"Action space size: {mapper.action_space_size()}")
    
    # Test action descriptions
    print("\nAction descriptions:")
    for i in range(11):
        print(f"{i:2d}: {mapper.describe(i)}")
    
    # Test action conversion
    print("\nTesting action conversion:")
    for i in range(11):
        action = mapper.int_to_action(i, "TEST001")
        print(f"Action {i}: {action.action_type} (esi={action.esi_level}, routing={action.routing_zone})")
    
    # Test action masking
    print("\nTesting action masking:")
    
    # Test with untriaged patient
    obs_untriaged = {
        "current_patient": {
            "patient_id": "P001",
            "wait_time_minutes": 5,
            "deteriorated": False
        },
        "pending_patients": ["P001", "P002"],
        "triaged_patients": []
    }
    
    mask_untriaged = mapper.get_action_mask(obs_untriaged)
    print(f"Untriaged patient mask: {mask_untriaged}")
    print(f"Valid actions: {np.where(mask_untriaged)[0].tolist()}")
    print(f"Can assign ESI (action 0): {mask_untriaged[0]}")
    
    # Test with triaged patient
    obs_triaged = {
        "current_patient": {
            "patient_id": "P001",
            "wait_time_minutes": 15,
            "deteriorated": False
        },
        "pending_patients": ["P002"],
        "triaged_patients": ["P001"]
    }
    
    mask_triaged = mapper.get_action_mask(obs_triaged)
    print(f"\nTriaged patient mask: {mask_triaged}")
    print(f"Valid actions: {np.where(mask_triaged)[0].tolist()}")
    print(f"Can assign ESI (action 0): {mask_triaged[0]}")
    
    # Test with deteriorated patient
    obs_deteriorated = {
        "current_patient": {
            "patient_id": "P001",
            "wait_time_minutes": 25,
            "deteriorated": True
        },
        "pending_patients": ["P001"],
        "triaged_patients": []
    }
    
    mask_deteriorated = mapper.get_action_mask(obs_deteriorated)
    print(f"\nDeteriorated patient mask: {mask_deteriorated}")
    print(f"Can reassess (action 8): {mask_deteriorated[8]}")
    
    # Test valid action sampling
    print("\nTesting valid action sampling:")
    for _ in range(5):
        action = mapper.sample_valid(obs_untriaged)
        print(f"Sampled action: {action} ({mapper.describe(action)})")
    
    # Test assertions
    assert mask_untriaged[0] == True, "Untriaged patient should allow ESI assignment"
    assert mask_triaged[0] == False, "Triaged patient should not allow ESI assignment"
    assert mask_deteriorated[8] == True, "Deteriorated patient should allow reassessment"
    
    print(f"\n✅ Action mapper test passed!")
