"""
OpenEnv Validation Script for TriageNet-RL
Performs 70 compliance checks to ensure environment meets OpenEnv standards.
Exit code 0 if all pass, 1 if any fail.
"""

import sys
import traceback
import yaml
from typing import Any, Dict

# Test imports
try:
    from models import Action, Observation, Patient, Vitals, BedStatus, Reward, StepResult, EpisodeResult
    from env import MedicalTriageEnv
    from tasks import ALL_PATIENTS, TASK_CONFIGS
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    MODELS_AVAILABLE = False


class OpenEnvValidator:
    """Validates TriageNet-RL environment against OpenEnv specifications."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def check(self, condition: bool, description: str) -> bool:
        """Check a condition and track results."""
        if condition:
            print(f"✅ {description}")
            self.passed += 1
            return True
        else:
            print(f"❌ {description}")
            self.failed += 1
            self.errors.append(description)
            return False
    
    def validate_yaml(self) -> bool:
        """Validate openenv.yaml structure and content."""
        print("\n🔍 YAML Validation")
        print("-" * 50)
        
        try:
            with open("openenv.yaml", "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.check(False, f"Failed to load openenv.yaml: {e}")
            return False
        
        # Required top-level fields
        required_fields = ["name", "version", "description", "author", "license", "tags", 
                          "environment", "tasks", "observation_space", "action_space", 
                          "reward_structure", "baseline_scores", "dependencies"]
        
        for field in required_fields:
            self.check(field in config, f"YAML has required field: {field}")
        
        # Environment section
        env = config.get("environment", {})
        env_fields = ["type", "observation_type", "action_type", "reward_type", "episode_termination"]
        for field in env_fields:
            self.check(field in env, f"Environment has field: {field}")
        
        # Tasks section
        tasks = config.get("tasks", [])
        self.check(len(tasks) >= 3, f"Has at least 3 tasks (found {len(tasks)})")
        
        for i, task in enumerate(tasks):
            task_fields = ["id", "name", "difficulty", "max_steps", "pass_threshold"]
            for field in task_fields:
                self.check(field in task, f"Task {i+1} has field: {field}")
        
        # Observation space
        obs_space = config.get("observation_space", {})
        self.check(obs_space.get("type") == "object", "Observation space is object type")
        
        obs_fields = obs_space.get("fields", {})
        required_obs = ["task_id", "step", "max_steps", "current_patient", "pending_patients",
                       "triaged_patients", "bed_status", "time_elapsed_minutes", "alerts",
                       "task_description", "score_so_far"]
        for field in required_obs:
            self.check(field in obs_fields, f"Observation space has field: {field}")
        
        # Action space
        act_space = config.get("action_space", {})
        self.check(act_space.get("type") == "object", "Action space is object type")
        
        act_fields = act_space.get("fields", {})
        self.check("action_type" in act_fields, "Action space has action_type")
        self.check("patient_id" in act_fields, "Action space has patient_id")
        
        # Reward structure
        reward = config.get("reward_structure", {})
        self.check(reward.get("dense") == True, "Reward structure is dense")
        self.check("range" in reward, "Reward structure has range")
        self.check("components" in reward, "Reward structure has components")
        
        return self.failed == 0
    
    def validate_models(self) -> bool:
        """Validate model classes and data structures."""
        print("\n🔍 Model Validation")
        print("-" * 50)
        
        if not MODELS_AVAILABLE:
            self.check(False, "Models cannot be imported")
            return False
        
        # Test dataclass instantiation
        try:
            vitals = Vitals(
                bp_systolic=120, bp_diastolic=80, heart_rate=70, respiratory_rate=16,
                spo2=98.0, temperature=37.0, gcs=15, pain_scale=2
            )
            self.check(True, "Vitals dataclass instantiates")
            
            patient = Patient(
                patient_id="TEST001", name="Test Patient", age=30, gender="M",
                chief_complaint="test complaint", vitals=vitals, true_esi=3, true_routing="fast_track"
            )
            self.check(True, "Patient dataclass instantiates")
            
            # Test to_dict methods
            vitals_dict = vitals.to_dict()
            self.check(isinstance(vitals_dict, dict), "Vitals.to_dict() returns dict")
            
            patient_dict = patient.to_agent_dict()
            self.check(isinstance(patient_dict, dict), "Patient.to_agent_dict() returns dict")
            self.check("true_esi" not in patient_dict, "Patient.to_agent_dict() hides ground truth")
            
            # Test Action with validation
            action = Action(action_type="assign_esi", patient_id="TEST001", esi_level=3, routing_zone="fast_track")
            self.check(True, "Action dataclass instantiates with valid parameters")
            self.check(action.esi_level == 3, "Action stores esi_level correctly")
            
            # Test invalid action (should raise)
            try:
                invalid_action = Action(action_type="invalid_type", patient_id="TEST001")
                self.check(False, "Action validation should reject invalid action_type")
            except ValueError:
                self.check(True, "Action validation rejects invalid action_type")
            
        except Exception as e:
            self.check(False, f"Model instantiation failed: {e}")
            traceback.print_exc()
        
        return self.failed == 0
    
    def validate_core_api(self) -> bool:
        """Validate core environment API for all tasks."""
        print("\n🔍 Core API Validation")
        print("-" * 50)
        
        if not MODELS_AVAILABLE:
            return False
        
        for task_id in [1, 2, 3]:
            print(f"\n  Testing Task {task_id}:")
            
            try:
                # Test environment creation
                env = MedicalTriageEnv(task_id=task_id, seed=42)
                self.check(True, f"Task {task_id}: Environment creates")
                
                # Test reset
                obs = env.reset()
                self.check(isinstance(obs, Observation), f"Task {task_id}: reset() returns Observation")
                
                # Test observation structure
                obs_dict = obs.model_dump()
                required_obs_fields = ["task_id", "step", "max_steps", "current_patient", 
                                      "pending_patients", "triaged_patients", "bed_status",
                                      "time_elapsed_minutes", "alerts", "task_description", 
                                      "score_so_far"]
                for field in required_obs_fields:
                    self.check(field in obs_dict, f"Task {task_id}: Observation has {field}")
                
                self.check(obs_dict["task_id"] == task_id, f"Task {task_id}: Observation task_id matches")
                self.check(isinstance(obs_dict["pending_patients"], list), f"Task {task_id}: pending_patients is list")
                self.check(len(obs_dict["pending_patients"]) > 0, f"Task {task_id}: Has pending patients")
                
                # Test state method
                state = env.state()
                self.check(isinstance(state, dict), f"Task {task_id}: state() returns dict")
                self.check("patients" in state, f"Task {task_id}: state() has patients")
                self.check("step" in state, f"Task {task_id}: state() has step")
                
                # Test step method
                if obs_dict["pending_patients"]:
                    patient_id = obs_dict["pending_patients"][0]
                    action = Action(action_type="assign_esi", patient_id=patient_id, 
                                  esi_level=3, routing_zone="fast_track")
                    
                    result = env.step(action)
                    self.check(isinstance(result, StepResult), f"Task {task_id}: step() returns StepResult")
                    self.check(isinstance(result.observation, Observation), f"Task {task_id}: StepResult has Observation")
                    self.check(isinstance(result.reward, Reward), f"Task {task_id}: StepResult has Reward")
                    self.check(isinstance(result.done, bool), f"Task {task_id}: StepResult has done bool")
                    self.check(isinstance(result.info, dict), f"Task {task_id}: StepResult has info dict")
                    
                    # Test reward structure
                    self.check(-1.0 <= result.reward.step_reward <= 1.0, f"Task {task_id}: Step reward in [-1, 1]")
                    self.check(isinstance(result.reward.breakdown, dict), f"Task {task_id}: Reward has breakdown")
                    self.check(isinstance(result.reward.feedback, str), f"Task {task_id}: Reward has feedback")
                
            except Exception as e:
                self.check(False, f"Task {task_id}: API test failed - {e}")
                traceback.print_exc()
        
        return self.failed == 0
    
    def validate_episode_termination(self) -> bool:
        """Validate episode termination and grading."""
        print("\n🔍 Episode Termination & Grading")
        print("-" * 50)
        
        if not MODELS_AVAILABLE:
            return False
        
        try:
            # Run Task 1 to completion
            env = MedicalTriageEnv(task_id=1, seed=42)
            obs = env.reset()
            
            steps = 0
            done = False
            while not done and steps < 20:  # Safety limit
                if obs.pending_patients:
                    patient_id = obs.pending_patients[0]
                    action = Action(action_type="assign_esi", patient_id=patient_id, 
                                  esi_level=2, routing_zone="acute_care")
                else:
                    action = Action(action_type="wait", patient_id="none")
                
                result = env.step(action)
                done = result.done
                obs = result.observation
                steps += 1
            
            # Test grading
            grade = env.grade()
            self.check(isinstance(grade, EpisodeResult), "grade() returns EpisodeResult")
            self.check(0.0 <= grade.final_score <= 1.0, "Final score in [0.0, 1.0]")
            self.check(isinstance(grade.passed, bool), "Grade has passed boolean")
            self.check(grade.task_id == 1, "Grade has correct task_id")
            self.check(grade.steps_taken > 0, "Grade reports steps taken")
            
        except Exception as e:
            self.check(False, f"Episode termination test failed: {e}")
            traceback.print_exc()
        
        return self.failed == 0
    
    def validate_reward_density(self) -> bool:
        """Validate reward density and variation."""
        print("\n🔍 Reward Density Validation")
        print("-" * 50)
        
        if not MODELS_AVAILABLE:
            return False
        
        try:
            env = MedicalTriageEnv(task_id=2, seed=42)
            obs = env.reset()
            
            rewards = []
            for i in range(4):
                if obs.pending_patients:
                    patient_id = obs.pending_patients[0]
                    # Different actions to get different rewards
                    if i == 0:
                        action = Action(action_type="assign_esi", patient_id=patient_id, 
                                      esi_level=1, routing_zone="resuscitation_bay")
                    elif i == 1:
                        action = Action(action_type="assign_esi", patient_id=patient_id, 
                                      esi_level=5, routing_zone="waiting_room")
                    elif i == 2:
                        action = Action(action_type="wait", patient_id=patient_id)
                    else:
                        action = Action(action_type="escalate", patient_id=patient_id)
                else:
                    action = Action(action_type="wait", patient_id="none")
                
                result = env.step(action)
                rewards.append(result.reward.step_reward)
                obs = result.observation
            
            # Check rewards vary (not all same)
            unique_rewards = set(rewards)
            self.check(len(unique_rewards) > 1, f"Rewards vary (found {len(unique_rewards)} unique values)")
            
        except Exception as e:
            self.check(False, f"Reward density test failed: {e}")
            traceback.print_exc()
        
        return self.failed == 0
    
    def validate_loop_detection(self) -> bool:
        """Validate duplicate action detection."""
        print("\n🔍 Loop Detection Validation")
        print("-" * 50)
        
        if not MODELS_AVAILABLE:
            return False
        
        try:
            env = MedicalTriageEnv(task_id=1, seed=42)
            obs = env.reset()
            
            if obs.pending_patients:
                patient_id = obs.pending_patients[0]
                action = Action(action_type="wait", patient_id=patient_id)
                
                # Take same wait action 4 times
                min_reward = None
                for i in range(4):
                    result = env.step(action)
                    if min_reward is None or result.reward.step_reward < min_reward:
                        min_reward = result.reward.step_reward
                
                self.check(min_reward < -0.1, f"Duplicate action penalty applied (min reward: {min_reward})")
            
        except Exception as e:
            self.check(False, f"Loop detection test failed: {e}")
            traceback.print_exc()
        
        return self.failed == 0
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("🏥 TriageNet-RL OpenEnv Validation")
        print("=" * 60)
        
        # Run all validation sections
        self.validate_yaml()
        self.validate_models()
        self.validate_core_api()
        self.validate_episode_termination()
        self.validate_reward_density()
        self.validate_loop_detection()
        
        # Print summary
        total = self.passed + self.failed
        compliance = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ Passed: {self.passed}/{total}")
        print(f"❌ Failed: {self.failed}/{total}")
        print(f"Compliance Score: {compliance:.1f}%")
        
        if self.failed == 0:
            print("✅ All checks passed! Environment is OpenEnv compliant.")
            return True
        else:
            print("❌ Some checks failed. See errors above.")
            if self.errors:
                print("\nFailed checks:")
                for error in self.errors:
                    print(f"  - {error}")
            return False


def main():
    """Run validation and exit with appropriate code."""
    validator = OpenEnvValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
