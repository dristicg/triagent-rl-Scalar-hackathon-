from __future__ import annotations
import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from schemas import Observation, TriageAction as Action, Reward, PatientInfo, BedStatus, Vitals, EpisodeResult
from tasks import ALL_PATIENTS

# Reward Constants (Aligned with tasks.py weights)
R_ESI_CORRECT = 0.10
R_RESOURCE_WASTE = -0.02
R_WAIT_PENALTY = -0.01
R_CRITICAL_MISS = -0.20
R_DETERIORATION = -0.08
R_DUPLICATE_ACTION = -0.10

class MedicalTriageEnv:
    """
    LLM-native Medical Triage Environment complying with OpenEnv specification.
    """
    
    TASK_CONFIGS = {
        1: {
            "name": "Easy: Single Patient ESI",
            "difficulty": "easy",
            "description": "Assess a single high-risk patient and assign correct ESI level.",
            "patient_ids": ["P005"],
            "max_steps": 5,
            "beds": {"resuscitation_bays": 2, "acute_care_beds": 8}
        },
        2: {
            "name": "Medium: Multi-Patient Queue",
            "difficulty": "medium",
            "description": "Prioritize and triage 8 patients with varying acuity.",
            "patient_ids": ["P001", "P004", "P006", "P008", "P010", "P011", "P013", "P007"],
            "max_steps": 20,
            "beds": {"resuscitation_bays": 2, "acute_care_beds": 6}
        },
        3: {
            "name": "Hard: Mass Casualty Surge",
            "difficulty": "hard",
            "description": "Mass casualty event with 15 patients and severe resource constraints.",
            "patient_ids": [
                "P001", "P002", "P003", "P004", "P005",
                "P006", "P007", "P008", "P009", "P010",
                "P011", "P012", "P013", "P014", "P001"
            ],
            "max_steps": 40,
            "beds": {"resuscitation_bays": 1, "acute_care_beds": 3}
        }
    }

    def __init__(self, task_id: int = 1, seed: int = 42):
        if task_id not in self.TASK_CONFIGS:
            raise ValueError(f"Invalid task_id: {task_id}")
        
        self.task_id = task_id
        self.config = self.TASK_CONFIGS[task_id]
        random.seed(seed)
        
        self._patients: Dict[str, Any] = {}
        self._bed_status: Dict[str, int] = {}
        self._step = 0
        self._score = 0.0
        self._done = False
        self._last_actions: List[str] = []
        self._deteriorations = 0

    def reset(self) -> Observation:
        """Returns the initial state as an Observation model."""
        self._step = 0
        self._score = 0.0
        self._done = False
        self._last_actions = []
        self._deteriorations = 0
        
        # Load patients
        self._patients = {}
        for pid in self.config["patient_ids"]:
            # Handle duplicates by appending suffix if needed
            unique_pid = pid
            count = 1
            while unique_pid in self._patients:
                unique_pid = f"{pid}_{count}"
                count += 1
            
            p_data = copy.deepcopy(ALL_PATIENTS[pid])
            p_data.patient_id = unique_pid
            # Convert internal patient to a dict for easy tracking
            self._patients[unique_pid] = {
                "info": p_data,
                "assigned_esi": None,
                "assigned_zone": None,
                "resources": [],
                "wait_time": 0,
                "deteriorated": False,
                "discharged": False
            }
            
        # Initialize beds
        beds = self.config["beds"]
        self._bed_status = {
            "total_beds": beds["resuscitation_bays"] + beds["acute_care_beds"],
            "available_beds": beds["resuscitation_bays"] + beds["acute_care_beds"],
            "resuscitation_bays": beds["resuscitation_bays"],
            "resuscitation_available": beds["resuscitation_bays"],
            "acute_care_beds": beds["acute_care_beds"],
            "acute_care_available": beds["acute_care_beds"],
        }
        
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """Processes the action and returns (Observation, Reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is already finished.")
            
        self._step += 1
        step_reward = 0.0
        feedback_parts = []
        breakdown = {}
        
        # 1. Validation
        if action.patient_id not in self._patients:
            step_reward = -0.3
            return self._build_observation(), Reward(score=step_reward, feedback="Invalid patient ID", breakdown={"error": -0.3}), self._done, {}

        patient = self._patients[action.patient_id]
        
        # 2. Duplicate Action Detection
        action_sig = f"{action.action_type}:{action.patient_id}:{action.esi_level}"
        if action_sig in self._last_actions[-3:]:
            step_reward += R_DUPLICATE_ACTION
            feedback_parts.append("Repeated action detected.")
            breakdown["duplicate"] = R_DUPLICATE_ACTION
        self._last_actions.append(action_sig)
        
        # 3. Simulate World (Deterioration)
        self._simulate_deterioration()
        
        # 4. Handle Action
        if action.action_type == "assign_esi":
            if action.esi_level == patient["info"].true_esi:
                step_reward += R_ESI_CORRECT
                feedback_parts.append(f"Correctly assigned ESI {action.esi_level}.")
                breakdown["esi_correct"] = R_ESI_CORRECT
                patient["assigned_esi"] = action.esi_level
            else:
                penalty = -0.1
                if abs(action.esi_level - patient["info"].true_esi) >= 2 and patient["info"].true_esi <= 2:
                    penalty += R_CRITICAL_MISS
                    feedback_parts.append("CRITICAL MISS: Low-acuity ESI assigned to life-threatening case!")
                else:
                    feedback_parts.append(f"Incorrect ESI level {action.esi_level}.")
                step_reward += penalty
                breakdown["esi_incorrect"] = penalty
                patient["assigned_esi"] = action.esi_level
            
            # Simple routing logic
            if action.routing_zone:
                patient["assigned_zone"] = action.routing_zone
                # Consume bed if appropriate
                if action.routing_zone == "resuscitation_bay" and self._bed_status["resuscitation_available"] > 0:
                    self._bed_status["resuscitation_available"] -= 1
                    self._bed_status["available_beds"] -= 1
                elif action.routing_zone == "acute_care" and self._bed_status["acute_care_available"] > 0:
                    self._bed_status["acute_care_available"] -= 1
                    self._bed_status["available_beds"] -= 1

        elif action.action_type == "request_resources":
            # Hard task resource constraint simulation
            unnecessary = ["ct_head", "blood_panel"] if patient["info"].true_esi >= 4 else []
            has_unnecessary = any(r in (action.resources or []) for r in unnecessary)
            if has_unnecessary:
                step_reward += R_RESOURCE_WASTE
                feedback_parts.append("Requested unnecessary resources for low-acuity patient.")
                breakdown["resource_waste"] = R_RESOURCE_WASTE
            else:
                step_reward += 0.05
                feedback_parts.append("Appropriate resources requested.")
                breakdown["resource_valid"] = 0.05
            if action.resources:
                patient["resources"].extend(action.resources)

        elif action.action_type == "wait":
            step_reward += R_WAIT_PENALTY
            feedback_parts.append("Idle step; patients are waiting.")
            breakdown["wait_penalty"] = R_WAIT_PENALTY

        # 5. Check Termination
        self._check_done()
        
        # 6. Finalize Step
        self._score += step_reward
        obs = self._build_observation()
        reward = Reward(score=step_reward, feedback=" | ".join(feedback_parts), breakdown=breakdown)
        
        return obs, reward, self._done, {"deteriorations": self._deteriorations}

    def grade(self) -> EpisodeResult:
        """Returns the official grading report using tasks.py logic."""
        import tasks
        
        # Prepare internal patient objects for tasks.py graders
        graded_patients = {pid: p["info"] for pid, p in self._patients.items()}
        for pid, p in self._patients.items():
            graded_patients[pid].assigned_esi = p["assigned_esi"]
            graded_patients[pid].assigned_routing = p["assigned_zone"]
            graded_patients[pid].resources_requested = p["resources"]

        if self.task_id == 1:
            report = tasks.grade_task1(graded_patients)
        elif self.task_id == 2:
            report = tasks.grade_task2(graded_patients)
        else:
            report = tasks.grade_task3(graded_patients, self._deteriorations, self._step)
            
        return EpisodeResult(
            task_id=self.task_id,
            final_score=report["final_score"],
            total_reward=self._score,
            steps_taken=self._step,
            correct_esi_count=sum(1 for p in self._patients.values() if p["assigned_esi"] == p["info"].true_esi),
            total_patients=len(self._patients),
            critical_misses=report.get("critical_misses", 0),
            deteriorations=self._deteriorations,
            grade_breakdown=report.get("breakdown", {}),
            passed=report["passed"]
        )

    def state(self) -> dict:
        """Returns the full internal state for debugging."""
        return {
            "task_id": self.task_id,
            "step": self._step,
            "score": self._score,
            "done": self._done,
            "patients": self._patients,
            "bed_status": self._bed_status,
            "deteriorations": self._deteriorations
        }

    def _simulate_deterioration(self):
        for pid, p in self._patients.items():
            if p["assigned_esi"] is None and not p["discharged"]:
                p["wait_time"] += 5
                # Rules: ESI 1 deteriorates after 10 mins, ESI 2 after 20 mins
                if p["info"].true_esi == 1 and p["wait_time"] >= 10 and not p["deteriorated"]:
                    p["deteriorated"] = True
                    self._deteriorations += 1
                    self._score += R_DETERIORATION
                elif p["info"].true_esi == 2 and p["wait_time"] >= 20 and not p["deteriorated"]:
                    p["deteriorated"] = True
                    self._deteriorations += 1
                    self._score += R_DETERIORATION

    def _check_done(self):
        # All patients triaged or max steps reached
        all_triaged = all(p["assigned_esi"] is not None for p in self._patients.values())
        if all_triaged or self._step >= self.config["max_steps"]:
            self._done = True

    def _build_observation(self) -> Observation:
        pending = [pid for pid, p in self._patients.items() if p["assigned_esi"] is None]
        triaged = [pid for pid, p in self._patients.items() if p["assigned_esi"] is not None]
        
        current_p_dict = None
        if pending:
            p = self._patients[pending[0]]
            current_p_dict = {
                "patient_id": p["info"].patient_id,
                "name": p["info"].name,
                "age": p["info"].age,
                "gender": p["info"].gender,
                "chief_complaint": p["info"].chief_complaint,
                "vitals": p["info"].vitals.__dict__ if hasattr(p["info"].vitals, "__dict__") else p["info"].vitals,
                "medical_history": p["info"].medical_history,
                "arrival_time": p["info"].arrival_time_minutes,
                "deteriorated": p["deteriorated"]
            }
            
        alerts = [f"ALERT: Patient {pid} has deteriorated!" for pid, p in self._patients.items() if p["deteriorated"]]
        
        return Observation(
            task_id=self.task_id,
            step=self._step,
            max_steps=self.config["max_steps"],
            current_patient=current_p_dict,
            pending_patients=pending,
            triaged_patients=triaged,
            bed_status=self._bed_status,
            time_elapsed_minutes=self._step * 5,
            alerts=alerts,
            task_description=self.config["description"],
            score_so_far=self._score
        )
