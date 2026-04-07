"""
PPO training with curriculum learning for TriageNet-RL.
Implements progressive difficulty from Task 1 → Task 2 → Task 3.
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import RL libraries
try:
    import numpy as np
    import torch
    import gymnasium
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import tensorboard
    RL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RL libraries not available: {e}")
    print("Install with: pip install -r rl/requirements_rl.txt")
    RL_AVAILABLE = False

from rl.environment_wrapper import TriageGymEnv, register_envs
from rl.feature_extractor import FeatureExtractor
from rl.action_mapper import ActionMapper

# Directory constants
MODELS_DIR = "models"
LOGS_DIR = "logs"


# Curriculum learning configuration
CURRICULUM = {
    1: {"timesteps": 50_000,  "difficulty": "Easy"},
    2: {"timesteps": 150_000, "difficulty": "Medium"},
    3: {"timesteps": 600_000, "difficulty": "Hard"},
}

# PPO hyperparameters
PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="logs/tensorboard/",
    policy_kwargs=dict(net_arch=[128, 128, 64]),
)

# Task 3 specific hyperparameters for improved performance on complex ED surge
TASK3_PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate = 1e-4,     # was 3e-4 — slower fine-tuning preserves
                               # Task 2 knowledge during transfer
    n_steps = 1024,            # was 512 — longer rollouts cover full
                               # 15-patient episodes before updating
    batch_size = 128,          # was 64 — larger batches = more stable
                               # gradients on complex state space
    n_epochs = 10,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.2,
    ent_coef = 0.05,           # was 0.01 — 5x more exploration needed
                               # for 15-patient scheduling problem
    vf_coef = 0.5,
    max_grad_norm = 0.5,
    verbose = 1,
    tensorboard_log = "logs/tensorboard/",
    policy_kwargs = dict(net_arch=[256, 256, 128]),
                               # was [128,128,64] — bigger network for
                               # richer state: 15 patients + beds +
                               # deterioration + time pressure
)


class TriageMetricsCallback(BaseCallback):
    """Custom callback for logging triage-specific metrics during training."""
    
    def __init__(self, eval_env, eval_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_results = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation episodes
            metrics = self._evaluate_agent()
            self.eval_results.append(metrics)
            
            # Log to tensorboard
            self.logger.record("triage/mean_final_score", metrics["mean_final_score"])
            self.logger.record("triage/critical_miss_rate", metrics["critical_miss_rate"])
            self.logger.record("triage/deterioration_rate", metrics["deterioration_rate"])
            self.logger.record("triage/esi_accuracy", metrics["esi_accuracy"])
            self.logger.record("triage/pass_rate", metrics["pass_rate"])
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Score={metrics['mean_final_score']:.3f}, "
                      f"Pass Rate={metrics['pass_rate']:.2f}")
        
        return True
    
    def _evaluate_agent(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent on multiple episodes."""
        final_scores = []
        critical_misses = []
        deteriorations = []
        esi_accuracies = []
        passes = []
        
        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, info = self.eval_env.step(action)
                done = terminated
            
            # Collect episode metrics
            if "final_score" in info:
                final_scores.append(info["final_score"])
                critical_misses.append(info.get("critical_misses", 0))
                deteriorations.append(info.get("deteriorations", 0))
                esi_accuracies.append(info.get("correct_esi_count", 0) / max(1, info.get("total_patients", 1)))
                passes.append(1.0 if info.get("passed", False) else 0.0)
        
        return {
            "mean_final_score": np.mean(final_scores) if final_scores else 0.0,
            "critical_miss_rate": np.mean(critical_misses) if critical_misses else 0.0,
            "deterioration_rate": np.mean(deteriorations) if deteriorations else 0.0,
            "esi_accuracy": np.mean(esi_accuracies) if esi_accuracies else 0.0,
            "pass_rate": np.mean(passes) if passes else 0.0,
        }


def make_env(task_id: int, seed: int = 42, monitor_dir: Optional[str] = None):
    """Create and wrap environment for training."""
    def _init():
        env = TriageGymEnv(task_id=task_id, seed=seed)
        if monitor_dir:
            env = Monitor(env, filename=monitor_dir)
        return env
    
    return _init


def train_task(task_id: int, timesteps: int, pretrained_path: Optional[str] = None, seed: int = 42) -> str:
    """Train PPO agent on a single task."""
    print(f"\n{'='*60}")
    print(f"Training Task {task_id} - {CURRICULUM[task_id]['difficulty']}")
    print(f"Timesteps: {timesteps:,}")
    print(f"{'='*60}")
    
    # Create environments
    train_env = DummyVecEnv([make_env(task_id, seed, "logs/monitor/train")])
    eval_env = DummyVecEnv([make_env(task_id, seed + 1000, None)])
    
    # Normalize observations
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    # Load pretrained model or create new one
    kwargs = TASK3_PPO_KWARGS if task_id == 3 else PPO_KWARGS
    if pretrained_path and os.path.exists(pretrained_path):
        if task_id == 3:
            # Task 3 has different architecture, start fresh
            print("Creating new PPO model for Task 3 (different architecture)")
            model = PPO(env=train_env, **kwargs)
        else:
            print(f"Loading pretrained model from {pretrained_path}")
            model = PPO.load(pretrained_path, env=train_env, **kwargs)
    else:
        print("Creating new PPO model")
        model = PPO(env=train_env, **kwargs)
    
    # Setup callbacks
    metrics_callback = TriageMetricsCallback(eval_env, eval_freq=5000, verbose=1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=f"models/task{task_id}_checkpoints/",
        name_prefix=f"ppo_task{task_id}"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/task{task_id}_best/",
        log_path=f"logs/eval/task{task_id}/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Train model
    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[metrics_callback, checkpoint_callback, eval_callback],
        progress_bar=True,
        log_interval=1000
    )
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = f"models/task{task_id}_final.zip"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save normalization statistics
    train_env.save("models/vec_normalize.pkl")
    
    # Final evaluation
    print("Final Evaluation (20 episodes):")
    final_metrics = TriageMetricsCallback(eval_env, verbose=1)
    final_metrics.model = model  # Set the model reference
    final_metrics = final_metrics._evaluate_agent(n_episodes=20)
    
    print(f"Mean Final Score: {final_metrics['mean_final_score']:.3f} ± {np.std([final_metrics['mean_final_score']]):.3f}")
    print(f"Pass Rate: {final_metrics['pass_rate']:.2f}")
    print(f"ESI Accuracy: {final_metrics['esi_accuracy']:.3f}")
    print(f"Critical Miss Rate: {final_metrics['critical_miss_rate']:.2f}")
    print(f"Deterioration Rate: {final_metrics['deterioration_rate']:.2f}")
    print(f"Training Time: {training_time:.1f}s")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    return final_model_path


def train_curriculum(start_task: int = 1, seed: int = 42):
    """Train PPO agent with curriculum learning."""
    if not RL_AVAILABLE:
        print("RL libraries not available. Install with: pip install -r rl/requirements_rl.txt")
        return
    
    print("🧠 Starting Curriculum Learning for TriageNet-RL")
    print(f"Seed: {seed}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Register environments
    register_envs()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Train each task in sequence
    pretrained_path = None
    training_results = {}
    
    for task_id in range(start_task, 4):
        timesteps = CURRICULUM[task_id]["timesteps"]
        
        # Train task
        model_path = train_task(task_id, timesteps, pretrained_path, seed)
        
        # Store results for next task
        pretrained_path = model_path
        training_results[task_id] = {
            "model_path": model_path,
            "timesteps": timesteps,
            "difficulty": CURRICULUM[task_id]["difficulty"]
        }
    
    # Print training summary
    print(f"\n{'='*60}")
    print("CURRICULUM TRAINING COMPLETE")
    print(f"{'='*60}")
    
    total_timesteps = sum(CURRICULUM[tid]["timesteps"] for tid in range(start_task, 4))
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Final Model: {pretrained_path}")
    
    print("\nTraining Progress:")
    for task_id, result in training_results.items():
        print(f"  Task {task_id} ({result['difficulty']}): {result['timesteps']:,} steps")
    
    print(f"\n📊 View tensorboard logs:")
    print(f"   tensorboard --logdir logs/tensorboard/")
    
    print(f"\n🎯 Test trained agent:")
    print(f"   python rl/evaluate.py --model {pretrained_path}")
    
    return training_results


def retrain_task3_only(seed: int = 42) -> str:
    """
    Retrain Task 3 only, loading Task 2 best model as starting point.
    Use this when Tasks 1 and 2 are already trained and only Task 3
    needs improvement. Skips the full curriculum.
    
    Returns path to best Task 3 model.
    """
    pretrained = f"{MODELS_DIR}/best_task2/best_model"
    
    if not os.path.exists(pretrained + ".zip"):
        print(f"⚠️  Task 2 model not found at {pretrained}.zip")
        print("   Run full curriculum first: python rl/train_ppo.py")
        return ""
    
    print("\n🏥 Retraining Task 3 (Hard) with improved hyperparameters")
    print(f"   Starting from Task 2 weights: {pretrained}")
    print(f"   Timesteps: {CURRICULUM[3]['timesteps']:,}")
    print(f"   Network: [256, 256, 128]")
    print(f"   LR: {TASK3_PPO_KWARGS['learning_rate']}")
    print(f"   Entropy: {TASK3_PPO_KWARGS['ent_coef']}")
    
    return train_task(
        task_id=3,
        timesteps=CURRICULUM[3]["timesteps"],
        pretrained_path=pretrained,
        seed=seed,
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train PPO agent on TriageNet-RL")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], 
                       help="Train single task (default: curriculum)")
    parser.add_argument("--steps", type=int, 
                       help="Override timesteps for single task")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--load", type=str, 
                       help="Load pretrained model")
    parser.add_argument("--curriculum", action="store_true", default=True,
                       help="Use curriculum learning (default)")
    parser.add_argument(
        "--retrain-task3",
        action="store_true",
        help="Retrain Task 3 only using best Task 2 model as base"
    )
    
    args = parser.parse_args()
    
    if not RL_AVAILABLE:
        print("❌ RL libraries not available")
        print("Install with: pip install -r rl/requirements_rl.txt")
        return
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.retrain_task3:
        retrain_task3_only(seed=args.seed)
    elif args.task:
        # Train single task
        timesteps = args.steps or CURRICULUM[args.task]["timesteps"]
        train_task(args.task, timesteps, args.load, args.seed)
    else:
        # Train curriculum
        train_curriculum(1, args.seed)


if __name__ == "__main__":
    main()
