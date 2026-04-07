"""
Medical Triage Environment - Emergency Department Simulation
Implements the MedicalTriageEnv class with ESI triage, patient deterioration,
and realistic clinical decision-making under time pressure.
"""

from __future__ import annotations
import copy
import random
from typing import Any, Dict, List, Optional

from models import (
    Action, BedStatus, EpisodeResult, Observation, Patient, Reward, StepResult
)
from tasks import ALL_PATIENTS, TASK_CONFIGS, grade_task1, grade_task2, grade_task3


# ── Reward constants (exact values as specified) ─────────────────────────────
R_ESI_EXACT          =  0.30
R_ESI_OFF1           =  0.10
R_ESI_WRONG          = -0.10
R_CRITICAL_MISS      = -0.50
R_ROUTING_CORRECT    =  0.20
R_ROUTING_PARTIAL    =  0.05
R_ROUTING_WRONG      = -0.15
R_RESOURCE_GOOD      =  0.15
R_RESOURCE_POOR      = -0.05
R_ESCALATE_CORRECT   =  0.25
R_ESCALATE_UNNEEDED  = -0.10
R_REASSESS_USEFUL    =  0.10
R_WAIT_PENALTY       = -0.08
R_DUPLICATE_ACTION   = -0.20
R_DETERIORATION      = -0.35
R_DISCHARGE_CORRECT  =  0.10
R_DISCHARGE_CRITICAL = -0.40


class MedicalTriageEnv:
    """Emergency Department Triage Environment with ESI classification."""
    
    ENV_NAME = "medical-triage-v1"
    VERSION  = "1.0.0"

    def __init__(self, task_id: int = 1, seed: Optional[int] = 42):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")
        
        self.task_id = task_id
        self._config = TASK_CONFIGS[task_id]
        self.seed = seed or 42
        random.seed(self.seed)
        
        # State variables
        self._patients: Dict[str, Patient] = {}
        self._bed_status: BedStatus
        self._step: int = 0
        self._cumulative_reward: float = 0.0
        self._deteriorations: int = 0
        self._last_actions: List[str] = []
        self._done: bool = False
        
        # Initialize environment
        self._reset_state()

    def _reset_state(self):
        """Reset all environment state for new episode."""
        # Load patients with unique IDs (handle duplicates in Task 3)
        patient_ids = self._config["patient_ids"]
        seen_ids = set()
        self._patients = {}
        
        for pid in patient_ids:
            if pid in seen_ids:
                # Create unique suffix for duplicates
                suffix = "_B"
                unique_pid = f"{pid}{suffix}"
                while unique_pid in seen_ids:
                    suffix = chr(ord(suffix[-1]) + 1)
                    unique_pid = f"{pid}_{suffix}"
                pid = unique_pid
            seen_ids.add(pid)
            
            # Deep copy patient to avoid mutations across episodes
            patient = copy.deepcopy(ALL_PATIENTS[pid.split("_")[0]])  # Remove suffix for lookup
            patient.patient_id = pid  # Assign the unique ID
            self._patients[pid] = patient
        
        # Reset bed status
        beds_config = self._config["available_beds"]
        total_resus = beds_config["resuscitation_bays"]
        total_acute = beds_config["acute_care_beds"]
        self._bed_status = BedStatus(
            total_beds=total_resus + total_acute,
            available_beds=total_resus + total_acute,
            resuscitation_bays=total_resus,
            resuscitation_available=total_resus,
            acute_care_beds=total_acute,
            acute_care_available=total_acute,
        )
        
        # Reset counters
        self._step = 0
        self._cumulative_reward = 0.0
        self._deteriorations = 0
        self._last_actions = []
        self._done = False

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self._reset_state()
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Execute one step in the environment."""
        if self._done:
            raise RuntimeError("Cannot call step() after episode is done")
        
        self._step += 1
        step_reward = 0.0
        feedback_parts = []
        breakdown = {}
        
        # Validate patient exists
        if action.patient_id not in self._patients:
            step_reward += -0.30
            feedback_parts.append("Invalid patient ID")
            breakdown["invalid_patient"] = -0.30
        else:
            patient = self._patients[action.patient_id]
            
            # Loop detection
            action_sig = f"{action.action_type}:{action.patient_id}:{action.esi_level}:{action.routing_zone}"
            if action_sig in self._last_actions[-3:]:
                step_reward += R_DUPLICATE_ACTION
                feedback_parts.append("Duplicate action detected")
                breakdown["duplicate_action"] = R_DUPLICATE_ACTION
            self._last_actions.append(action_sig)
            
            # Simulate deterioration
            self._simulate_deterioration()
            
            # Dispatch to action handlers
            if action.action_type == "assign_esi":
                reward_part, feedback = self._handle_assign_esi(action, patient)
                step_reward += reward_part
                feedback_parts.append(feedback)
                breakdown["esi_assignment"] = reward_part
            elif action.action_type == "request_resources":
                reward_part, feedback = self._handle_resources(action, patient)
                step_reward += reward_part
                feedback_parts.append(feedback)
                breakdown["resources"] = reward_part
            elif action.action_type == "escalate":
                reward_part, feedback = self._handle_escalate(action, patient)
                step_reward += reward_part
                feedback_parts.append(feedback)
                breakdown["escalation"] = reward_part
            elif action.action_type == "reassess":
                reward_part, feedback = self._handle_reassess(action, patient)
                step_reward += reward_part
                feedback_parts.append(feedback)
                breakdown["reassessment"] = reward_part
            elif action.action_type == "discharge":
                reward_part, feedback = self._handle_discharge(action, patient)
                step_reward += reward_part
                feedback_parts.append(feedback)
                breakdown["discharge"] = reward_part
            elif action.action_type == "wait":
                step_reward += R_WAIT_PENALTY
                feedback_parts.append("Waiting")
                breakdown["wait"] = R_WAIT_PENALTY
        
        # Clamp reward
        step_reward = max(-1.0, min(1.0, step_reward))
        self._cumulative_reward += step_reward
        
        # Check termination
        self._check_done()
        
        # Build result
        obs = self._build_observation()
        reward = Reward(
            step_reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts)
        )
        
        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={"deteriorations": self._deteriorations}
        )

    def _handle_assign_esi(self, action: Action, patient: Patient) -> tuple[float, str]:
        """Handle ESI assignment action."""
        if action.esi_level is None:
            return -0.15, "No ESI level provided"
        
        # Assign ESI and routing
        patient.assigned_esi = action.esi_level
        if action.routing_zone:
            patient.assigned_routing = action.routing_zone
        
        # Score ESI accuracy
        from tasks import grade_esi_accuracy, grade_routing, is_critical_miss
        esi_score, esi_feedback = grade_esi_accuracy(action.esi_level, patient.true_esi)
        
        if action.esi_level == patient.true_esi:
            reward = R_ESI_EXACT
        elif abs(action.esi_level - patient.true_esi) == 1:
            reward = R_ESI_OFF1
        else:
            reward = R_ESI_WRONG
        
        # Critical miss penalty
        if is_critical_miss(action.esi_level, patient.true_esi):
            reward += R_CRITICAL_MISS
            esi_feedback += " | CRITICAL MISS"
        
        # Score routing
        routing_reward = 0.0
        routing_feedback = ""
        if action.routing_zone:
            routing_score, routing_feedback = grade_routing(
                action.routing_zone, patient.true_routing, patient.true_esi
            )
            if routing_score == 1.0:
                routing_reward = R_ROUTING_CORRECT
            elif routing_score >= 0.5:
                routing_reward = R_ROUTING_PARTIAL
            else:
                routing_reward = R_ROUTING_WRONG
            
            # Update bed availability
            self._update_beds(action.routing_zone)
        
        total_reward = reward + routing_reward
        feedback = f"{esi_feedback} | {routing_feedback}"
        
        return total_reward, feedback

    def _handle_resources(self, action: Action, patient: Patient) -> tuple[float, str]:
        """Handle resource request action."""
        if action.resources:
            patient.resources_requested.extend(action.resources)
        
        from tasks import grade_resources
        resource_score, resource_feedback = grade_resources(
            patient.resources_requested, patient.true_esi
        )
        
        if resource_score >= 0.8:
            reward = R_RESOURCE_GOOD
        else:
            reward = R_RESOURCE_POOR
        
        return reward, resource_feedback

    def _handle_escalate(self, action: Action, patient: Patient) -> tuple[float, str]:
        """Handle escalation action."""
        if patient.true_esi <= 2:
            reward = R_ESCALATE_CORRECT
            feedback = "Correct escalation for critical patient"
        else:
            reward = R_ESCALATE_UNNEEDED
            feedback = "Unnecessary escalation"
        
        return reward, feedback

    def _handle_reassess(self, action: Action, patient: Patient) -> tuple[float, str]:
        """Handle reassessment action."""
        if patient.deteriorated or patient.wait_time_minutes > 20:
            reward = R_REASSESS_USEFUL
            feedback = "Useful reassessment"
        else:
            reward = -0.05
            feedback = "Unnecessary reassessment"
        
        return reward, feedback

    def _handle_discharge(self, action: Action, patient: Patient) -> tuple[float, str]:
        """Handle discharge action."""
        patient.discharged = True
        
        if patient.true_esi >= 4:
            reward = R_DISCHARGE_CORRECT
            feedback = "Appropriate discharge"
        else:
            reward = R_DISCHARGE_CRITICAL
            feedback = "Inappropriate discharge of critical patient"
        
        return reward, feedback

    def _simulate_deterioration(self):
        """Simulate patient deterioration based on wait times."""
        for patient in self._patients.values():
            if patient.assigned_esi is None and not patient.discharged:
                patient.wait_time_minutes += 5  # Each step = 5 minutes
                
                # Deterioration rules
                if patient.true_esi == 1 and patient.wait_time_minutes >= 10:
                    if not patient.deteriorated:
                        patient.deteriorated = True
                        self._deteriorations += 1
                elif patient.true_esi == 2 and patient.wait_time_minutes >= 20:
                    if not patient.deteriorated:
                        patient.deteriorated = True
                        self._deteriorations += 1

    def _update_beds(self, routing_zone: str):
        """Update bed availability based on routing."""
        if routing_zone == "resuscitation_bay" and self._bed_status.resuscitation_available > 0:
            self._bed_status.resuscitation_available -= 1
            self._bed_status.available_beds -= 1
        elif routing_zone == "acute_care" and self._bed_status.acute_care_available > 0:
            self._bed_status.acute_care_available -= 1
            self._bed_status.available_beds -= 1

    def _check_done(self):
        """Check if episode should terminate."""
        # Max steps reached
        if self._step >= self._config["max_steps"]:
            self._done = True
            return
        
        # All patients triaged or discharged
        all_processed = all(
            p.assigned_esi is not None or p.discharged 
            for p in self._patients.values()
        )
        if all_processed:
            self._done = True

    def _build_observation(self) -> Observation:
        """Build current observation for the agent."""
        pending = [
            pid for pid, p in self._patients.items()
            if p.assigned_esi is None and not p.discharged
        ]
        triaged = [
            pid for pid, p in self._patients.items()
            if p.assigned_esi is not None or p.discharged
        ]
        
        # Current patient (first pending, or None)
        current_patient = None
        if pending:
            current_patient = self._patients[pending[0]].to_agent_dict()
        
        # Alerts for deteriorated patients
        alerts = []
        for pid, p in self._patients.items():
            if p.deteriorated:
                alerts.append(f"Patient {pid} has deteriorated!")
        
        return Observation(
            task_id=self.task_id,
            step=self._step,
            max_steps=self._config["max_steps"],
            current_patient=current_patient,
            pending_patients=pending,
            triaged_patients=triaged,
            bed_status=self._bed_status.to_dict(),
            time_elapsed_minutes=self._step * 5,
            alerts=alerts,
            task_description=self._config["description"],
            score_so_far=self._cumulative_reward,
        )

    def state(self) -> Dict[str, Any]:
        """Return full environment state (including ground truth)."""
        return {
            "task_id": self.task_id,
            "step": self._step,
            "max_steps": self._config["max_steps"],
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
            "deteriorations": self._deteriorations,
            "patients": {
                pid: {
                    "patient_id": p.patient_id,
                    "name": p.name,
                    "true_esi": p.true_esi,
                    "true_routing": p.true_routing,
                    "assigned_esi": p.assigned_esi,
                    "assigned_routing": p.assigned_routing,
                    "wait_time_minutes": p.wait_time_minutes,
                    "deteriorated": p.deteriorated,
                    "discharged": p.discharged,
                }
                for pid, p in self._patients.items()
            },
            "bed_status": self._bed_status.to_dict(),
        }

    def grade(self) -> EpisodeResult:
        """Grade the completed episode and return results."""
        if not self._done:
            raise RuntimeError("Cannot grade incomplete episode")
        
        # Count correct ESI assignments
        correct_esi = sum(
            1 for p in self._patients.values()
            if p.assigned_esi == p.true_esi
        )
        
        # Get task-specific grade
        if self.task_id == 1:
            grade_result = grade_task1(self._patients)
        elif self.task_id == 2:
            grade_result = grade_task2(self._patients)
        else:  # task_id == 3
            grade_result = grade_task3(self._patients, self._deteriorations, self._step)
        
        return EpisodeResult(
            task_id=self.task_id,
            final_score=grade_result["final_score"],
            total_reward=self._cumulative_reward,
            steps_taken=self._step,
            correct_esi_count=correct_esi,
            total_patients=len(self._patients),
            critical_misses=grade_result.get("critical_misses", 0),
            deteriorations=self._deteriorations,
            grade_breakdown=grade_result["breakdown"],
            passed=grade_result["passed"],
        )
