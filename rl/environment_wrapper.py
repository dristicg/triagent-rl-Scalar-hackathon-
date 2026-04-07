"""
Gymnasium environment wrapper for TriageNet-RL.
Provides standard RL interface with feature extraction and action mapping.
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import gymnasium, but make it optional
try:
    import gymnasium
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    print("Warning: gymnasium not available. Install with: pip install gymnasium")
    GYM_AVAILABLE = False

from env import MedicalTriageEnv
from rl.feature_extractor import FeatureExtractor
from rl.action_mapper import ActionMapper


class TriageGymEnv(gymnasium.Env):
    """Gymnasium wrapper for Medical Triage environment."""
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, task_id: int = 1, seed: int = 42, render_mode: str = "ansi"):
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium is required for TriageGymEnv. Install with: pip install gymnasium")
        
        self._env = MedicalTriageEnv(task_id=task_id, seed=seed)
        self._extractor = FeatureExtractor()
        self._mapper = ActionMapper()
        self.render_mode = render_mode
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=2.0, 
            shape=(30,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(11)
        
        # Track episode info
        self._episode_info = {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        if seed is not None:
            self._env = MedicalTriageEnv(task_id=self._env.task_id, seed=seed)
        
        obs = self._env.reset()
        features = self._extractor.extract(obs.model_dump())
        
        info = {
            "task_id": obs.task_id,
            "task_description": obs.task_description,
            "max_steps": obs.max_steps,
            "pending_patients": len(obs.pending_patients),
            "triaged_patients": len(obs.triaged_patients)
        }
        
        self._episode_info = info.copy()
        
        return features, info
    
    def step(self, action_int: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Get patient ID for action
        obs_dict = self._env._build_observation().model_dump()
        pending = obs_dict.get("pending_patients", [])
        current_patient = obs_dict.get("current_patient")
        
        if not pending and not current_patient:
            # No patient available, force wait action
            action_int = 10
        
        patient_id = current_patient.get("patient_id") if current_patient else "none"
        
        # Convert integer action to Action object
        action = self._mapper.int_to_action(action_int, patient_id)
        
        # Execute step
        result = self._env.step(action)
        
        # Extract features
        features = self._extractor.extract(result.observation.model_dump())
        
        # Prepare return values
        terminated = result.done
        truncated = False
        step_reward = result.reward.step_reward
        
        # Build info dict
        info = {
            "step": result.observation.step,
            "task_id": result.observation.task_id,
            "cumulative_reward": result.reward.cumulative_reward,
            "feedback": result.reward.feedback,
            "breakdown": result.reward.breakdown,
            "pending_patients": len(result.observation.pending_patients),
            "triaged_patients": len(result.observation.triaged_patients),
            "alerts": result.observation.alerts,
            "action_taken": action_int,
            "action_description": self._mapper.describe(action_int),
            "step_reward": step_reward  # Original undampened reward for verification
        }
        
        # Add grade info if episode terminated
        if terminated:
            grade = self._env.grade()
            info.update({
                "final_score": grade.final_score,
                "total_reward": grade.total_reward,
                "steps_taken": grade.steps_taken,
                "correct_esi_count": grade.correct_esi_count,
                "total_patients": grade.total_patients,
                "critical_misses": grade.critical_misses,
                "deteriorations": grade.deteriorations,
                "passed": grade.passed,
                "grade_breakdown": grade.grade_breakdown
            })
        
        # Apply Task 3 reward dampening
        reward = step_reward
        
        # Task 3 reward dampening:
        # The -0.35 deterioration penalty fires constantly during early Task 3
        # training when the agent is still random (15 patients, 1 resus bay,
        # tight time limits). This creates an overwhelmingly negative signal
        # that makes it impossible to find positive reward gradients.
        # Dampening by 30% reduces penalty dominance without removing the signal.
        # This ONLY affects training stability — the grader (env.grade()) still
        # uses the original undampened scores for final evaluation.
        if self._env.task_id == 3:
            reward = reward * 0.7
        
        return features, reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        """Render the environment state."""
        if self.render_mode == "ansi":
            return self._render_ascii()
        elif self.render_mode == "human":
            print(self._render_ascii())
            return None
        else:
            return None
    
    def _render_ascii(self) -> str:
        """Render environment as ASCII string."""
        obs = self._env._build_observation()
        state = self._env.state()
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"Task {obs.task_id}: {obs.task_description}")
        lines.append(f"Step: {obs.step}/{obs.max_steps} | Time: {obs.time_elapsed_minutes} min")
        lines.append(f"Score: {obs.score_so_far:+.3f}")
        lines.append("-" * 60)
        
        if obs.current_patient:
            p = obs.current_patient
            vitals = p.get("vitals", {})
            lines.append(f"Current Patient: {p.get('name', 'Unknown')} ({p.get('patient_id', 'Unknown')})")
            lines.append(f"  Age: {p.get('age', '?')} | Sex: {p.get('gender', '?')}")
            lines.append(f"  Complaint: {p.get('chief_complaint', 'None')}")
            lines.append(f"  Vitals: BP {vitals.get('bp_systolic', '?')}/{vitals.get('bp_diastolic', '?')} | "
                        f"HR {vitals.get('heart_rate', '?')} | SpO2 {vitals.get('spo2', '?')}% | "
                        f"GCS {vitals.get('gcs', '?')} | Pain {vitals.get('pain_scale', '?')}")
            lines.append(f"  Wait time: {p.get('wait_time_minutes', 0)} min | "
                        f"Deteriorated: {'Yes' if p.get('deteriorated', False) else 'No'}")
        else:
            lines.append("No current patient")
        
        lines.append(f"Queue: {len(obs.pending_patients)} pending, {len(obs.triaged_patients)} triaged")
        beds = obs.bed_status
        lines.append(f"Beds: {beds.get('available_beds', 0)} available | "
                    f"Resus: {beds.get('resuscitation_available', 0)} | "
                    f"Acute: {beds.get('acute_care_available', 0)}")
        
        if obs.alerts:
            lines.append("Alerts:")
            for alert in obs.alerts:
                lines.append(f"  🚨 {alert}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_action_mask(self) -> np.ndarray:
        """Get action mask for current state."""
        obs_dict = self._env._build_observation().model_dump()
        return self._mapper.get_action_mask(obs_dict)
    
    def close(self):
        """Close the environment."""
        pass


def register_envs():
    """Register TriageGymEnv environments with gymnasium."""
    if not GYM_AVAILABLE:
        print("Warning: gymnasium not available, cannot register environments")
        return
    
    # Register each task as a separate environment
    for task_id in [1, 2, 3]:
        env_name = f"MedicalTriage-Task{task_id}-v1"
        
        def make_env(task_id=task_id):
            return TriageGymEnv(task_id=task_id)
        
        try:
            gymnasium.register(
                env_name,
                entry_point=make_env,
                kwargs={"task_id": task_id},
                max_episode_steps=1000,
                reward_threshold=0.0,
            )
            print(f"Registered {env_name}")
        except Exception as e:
            print(f"Failed to register {env_name}: {e}")


if __name__ == "__main__":
    # Test the environment wrapper
    if not GYM_AVAILABLE:
        print("gymnasium not available, skipping tests")
        exit(1)
    
    print("TriageGymEnv Test:")
    
    # Register environments
    register_envs()
    
    # Test each task
    for task_id in [1, 2, 3]:
        print(f"\n--- Testing Task {task_id} ---")
        
        env = TriageGymEnv(task_id=task_id, seed=42)
        
        # Test reset
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"Action space size: {env.action_space.n}")
        print(f"Task ID: {info['task_id']}")
        print(f"Max steps: {info['max_steps']}")
        
        # Test action masking
        mask = env.get_action_mask()
        print(f"Action mask: {mask}")
        print(f"Valid actions: {np.where(mask)[0].tolist()}")
        
        # Test a few steps
        for step in range(3):
            # Sample valid action
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 10  # wait
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step + 1}: Action {action} ({env._mapper.describe(action)}) "
                  f"→ Reward {reward:+.3f}")
            
            if terminated:
                print("Episode terminated")
                print(f"Final score: {info.get('final_score', 'N/A')}")
                print(f"Passed: {'✅' if info.get('passed', False) else '❌'}")
                break
            
            # Update mask for next step
            mask = env.get_action_mask()
        
        # Test rendering
        render_output = env.render()
        if render_output:
            print("\nRender output:")
            print(render_output[:200] + "..." if len(render_output) > 200 else render_output)
        
        env.close()
    
    print(f"\n✅ TriageGymEnv test passed!")
