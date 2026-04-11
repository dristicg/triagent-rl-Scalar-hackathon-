"""
Rule-based baseline agent for Medical Triage environment.
Implements clinical heuristics for ESI classification and routing.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

from env import MedicalTriageEnv
from models import Action

def compute_patient_priority(patient: dict) -> int:
    """
    Returns priority score for a patient.
    Lower number = more urgent = should be triaged first.
    Used to sort pending patients before acting.
    """
    if not patient:
        return 999
    
    vitals = patient.get("vitals", {})
    complaint = (patient.get("chief_complaint") or "").lower()
    wait_time = patient.get("wait_time_minutes", 0)
    
    # Start with ESI classification from vitals
    gcs = vitals.get("gcs", 15)
    spo2 = vitals.get("spo2", 100)
    bp_sys = vitals.get("bp_systolic", 120)
    hr = vitals.get("heart_rate", 70)
    rr = vitals.get("respiratory_rate", 16)
    pain = vitals.get("pain_scale", 0)
    temp = vitals.get("temperature", 37.0)
    
    # Determine ESI from vitals
    if (gcs <= 8 or spo2 < 85 or bp_sys < 80 or hr == 0):
        esi = 1
    elif (spo2 < 92 or hr > 120 or hr < 50 or bp_sys < 90 or 
          bp_sys > 180 or rr > 25 or gcs < 13 or pain >= 9):
        esi = 2
    elif (pain >= 5 or hr > 100 or bp_sys > 150 or temp > 38.5):
        esi = 3
    elif (pain >= 2 or temp > 37.5):
        esi = 4
    else:
        esi = 5
    
    # Base priority from ESI (1 = most urgent)
    priority = esi * 10
    
    # Boost priority if already waiting long
    if wait_time >= 15:
        priority -= 5
    if wait_time >= 25:
        priority -= 10
    
    # Boost priority for deteriorated patients
    if patient.get("deteriorated", False):
        priority -= 20
    
    return priority

class RuleBasedAgent:
    """Rule-based triage agent using clinical heuristics."""
    
    def __init__(self):
        # Keyword patterns for ESI classification
        self.critical_keywords = {
            "cardiac arrest", "unresponsive", "cpr", "seizure not stopping",
            "active seizure", "apnea", "no pulse",
            "not breathing", "stopped breathing",
            "unconscious", "found unresponsive", "arrest"
        }
        self.high_risk_keywords = {
            "worst headache", "thunderclap", "worst of my life",
            "chest pressure", "chest pain", "chest tightness",
            "radiating to arm", "radiating to left",
            "stroke", "weakness", "facial droop",
            "slurred speech", "sudden onset",
            "severe shortness", "can't breathe", "unable to breathe",
            "severe bleeding", "not stopping", "won't stop",
            "sudden severe", "severe sudden"
        }
        self.urgent_keywords = {
            "abdominal pain", "nausea", "vomiting", "blood in urine", "bleeding"
        }
        self.minor_keywords = {
            "sore throat", "mild fever", "prescription refill", "minor cut",
            "ankle sprain", "sprain"
        }
        
        # State tracking for Task 3
        self._triaged = set()
        self._resource_requested = set()
        self._escalated = set()
        
        # Resource mapping by ESI
        self.resource_map = {
            1: ["iv_access", "blood_panel", "ecg"],
            2: ["ecg", "blood_panel", "iv_access", "chest_xray"],
            3: ["blood_panel", "chest_xray"],
            4: [],
            5: []
        }
        
        # Routing mapping by ESI
        self.routing_map = {
            1: "resuscitation_bay",
            2: "acute_care", 
            3: "fast_track",
            4: "waiting_room",
            5: "waiting_room"
        }

    def classify_esi_by_vitals(self, vitals: Dict) -> int:
        """Classify ESI based on vital signs - tuned for accuracy."""
        gcs = vitals.get("gcs", 15)
        spo2 = vitals.get("spo2", 100)
        bp_sys = vitals.get("bp_systolic", 120)
        hr = vitals.get("heart_rate", 70)
        rr = vitals.get("respiratory_rate", 16)
        pain = vitals.get("pain_scale", 0)
        temp = vitals.get("temperature", 37.0)
        
        # ESI 1: Life-threatening vitals
        if (gcs <= 8 or spo2 < 85 or bp_sys < 80 or hr == 0):
            return 1
        
        # ESI 2: High-risk vitals
        if (spo2 < 92 or hr > 120 or hr < 50 or bp_sys < 90 or 
            bp_sys > 180 or rr > 25 or gcs < 13 or pain >= 9):
            return 2
        
        # ESI 3: Urgent vitals
        if (pain >= 5 or hr > 100 or bp_sys > 150 or temp > 38.5):
            return 3
        
        # ESI 4 vs 5: More strict boundary to avoid misclassification
        if (pain >= 2 or temp > 37.5 or hr > 90 or bp_sys > 140):
            return 4
        
        # ESI 5: Truly normal vitals  
        return 5

    def classify_esi_by_keywords(self, complaint: str) -> int:
        """Classify ESI based on chief complaint keywords."""
        complaint_lower = complaint.lower()
        
        # Check for critical patterns
        for keyword in self.critical_keywords:
            if keyword in complaint_lower:
                return 1
        
        # Check for high-risk patterns
        for keyword in self.high_risk_keywords:
            if keyword in complaint_lower:
                return 2
        
        # Check for urgent patterns
        for keyword in self.urgent_keywords:
            if keyword in complaint_lower:
                return 3
        
        # Check for minor patterns
        for keyword in self.minor_keywords:
            if keyword in complaint_lower:
                return 4
        
        # Default to moderate urgency
        return 3

    def get_esi_level(self, patient: Dict) -> int:
        """Get ESI level using combined vitals and keyword analysis."""
        # Get ESI from both methods
        vitals_esi = self.classify_esi_by_vitals(patient.get("vitals", {}))
        keyword_esi = self.classify_esi_by_keywords(patient.get("chief_complaint", ""))
        
        # Take more urgent (lower) ESI
        return min(vitals_esi, keyword_esi)
    
    def reset(self):
        """Reset tracking sets for new episode."""
        self._triaged = set()
        self._resource_requested = set()
        self._escalated = set()

    def select_action(self, obs: Dict) -> Action:
        """Select action based on current observation."""
        current_patient = obs.get("current_patient")
        if not current_patient:
            return Action(action_type="wait", patient_id="none")
        
        patient_id = current_patient["patient_id"]
        
        # Check if patient already triaged
        if patient_id in obs.get("triaged_patients", []):
            return Action(action_type="wait", patient_id=patient_id)
        
        # Get ESI level
        esi_level = self.get_esi_level(current_patient)
        routing_zone = self.routing_map[esi_level]
        
        # Create action
        return Action(
            action_type="assign_esi",
            patient_id=patient_id,
            esi_level=esi_level,
            routing_zone=routing_zone,
            notes=f"Rule-based: ESI {esi_level} based on vitals and complaint"
        )
    
    def act(self, obs: Dict, task_id: int = 1) -> Optional[Action]:
        """
        Simple and efficient action selection - focus on accurate ESI assignment.
        For Task 3: uses improved keyword matching to reduce off-by-one errors.
        """
        current_patient = obs.get("current_patient")
        if not current_patient:
            return Action(action_type="wait", patient_id="none")
        
        pid = current_patient.get("patient_id")
        if not pid:
            return Action(action_type="wait", patient_id="none")
        
        # Skip already-triaged patients
        if pid in obs.get("triaged_patients", []):
            return None
        
        vitals = current_patient.get("vitals", {})
        complaint = (current_patient.get("chief_complaint") or "").lower()
        esi_v = self.classify_esi_by_vitals(vitals)
        esi_c = self.classify_esi_by_keywords(complaint)
        esi = min(esi_v, esi_c)
        
        # Always assign ESI - this is the primary action
        if pid not in self._triaged:
            self._triaged.add(pid)
            routing_zone = self.routing_map.get(esi, "fast_track")
            return Action(
                action_type="assign_esi",
                patient_id=pid,
                esi_level=esi,
                routing_zone=routing_zone,
                notes=f"ESI {esi}",
            )
        
        return None


def run_task(task_id: int, seed: int = 42, verbose: bool = True) -> Dict:
    """Run baseline agent on a single task."""
    env = MedicalTriageEnv(task_id=task_id, seed=seed)
    agent = RuleBasedAgent()
    agent.reset()
    obs = env.reset()
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Task {task_id} | {env.config['name']}")
        print(f"Difficulty: {env.config['difficulty'].upper()}")
        print(f"Patients: {len(env.config['patient_ids'])}")
        print(f"{'='*50}")
    
    done = False
    steps = 0
    
    while not done and steps < 100:  # Safety limit
        # Use improved act() method for all tasks
        action = agent.act(obs.model_dump(), task_id=task_id)
        
        # Fall back to assign_esi if act returns None
        if action is None:
            action = agent.select_action(obs.model_dump())
        
        obs, reward, done, info = env.step(action)
        steps += 1
        
        if verbose and steps <= 20:  # Show first 20 steps for visibility
            action_type = action.action_type
            esi_str = f"ESI={action.esi_level}" if action.esi_level else ""
            print(f"Step {steps:02d}: {action_type:15s} {esi_str:10s} -> reward {reward.step_reward:+.3f}")
    
    grade = env.grade()
    
    if verbose:
        print(f"\nFinal Score: {grade.final_score:.3f}")
        print(f"Correct ESI: {grade.correct_esi_count}/{grade.total_patients}")
        print(f"Passed: {'YES' if grade.passed else 'NO'}")
    
    return {
        "task_id": task_id,
        "final_score": grade.final_score,
        "steps_taken": steps,
        "correct_esi": grade.correct_esi_count,
        "total_patients": grade.total_patients,
        "critical_misses": grade.critical_misses,
        "deteriorations": grade.deteriorations,
        "passed": grade.passed,
    }


def run_all(seed: int = 42, verbose: bool = True, json_output: bool = False):
    """Run baseline agent on all tasks."""
    results = {}
    
    for task_id in [1, 2, 3]:
        results[task_id] = run_task(task_id, seed, verbose)
    
    # Print summary table
    if verbose:
        print(f"\n{'='*60}")
        print("BASELINE AGENT RESULTS")
        print(f"{'='*60}")
        print(f"{'Task':<6} {'Difficulty':<10} {'Score':>8} {'Passed':>8}")
        print(f"{'-'*38}")
        
        total_score = 0.0
        for tid, result in results.items():
            difficulty = ["Easy", "Medium", "Hard"][tid-1]
            passed = "YES" if result["passed"] else "NO"
            print(f"Task {tid}  {difficulty:<10} {result['final_score']:>7.3f}  {passed}")
            total_score += result["final_score"]
        
        print(f"{'-'*38}")
        print(f"{'AVERAGE':<16} {total_score/3:>7.3f}")
        print(f"{'='*60}")
    
    if json_output:
        with open("results/baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: results/baseline_results.json")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run rule-based baseline agent")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], 
                       help="Run single task (default: all)")
    parser.add_argument("--all", action="store_true", 
                       help="Run all tasks (default)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--quiet", action="store_true", 
                       help="Minimal output")
    parser.add_argument("--json", action="store_true", 
                       help="Save results to JSON")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.task:
        result = run_task(args.task, args.seed, verbose)
        if args.json:
            with open("results/baseline_results.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved: results/baseline_results.json")
    else:
        run_all(args.seed, verbose, args.json)


if __name__ == "__main__":
    main()
