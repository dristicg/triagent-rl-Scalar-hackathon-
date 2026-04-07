"""
Evaluation script for comparing PPO agents against baseline.
Generates comparison tables, learning curves, and detailed metrics.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import RL libraries
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    RL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RL libraries not available: {e}")
    print("Install with: pip install -r rl/requirements_rl.txt")
    RL_AVAILABLE = False

from env import MedicalTriageEnv
from baseline import RuleBasedAgent, run_task
from rl.environment_wrapper import TriageGymEnv, register_envs


def evaluate_ppo_agent(task_id: int, model_path: str, n_episodes: int = 50) -> Dict[str, Any]:
    """Evaluate trained PPO agent on specified task."""
    if not RL_AVAILABLE:
        return {"error": "RL libraries not available"}
    
    print(f"Evaluating PPO agent on Task {task_id}...")
    
    # Load model
    try:
        # Load VecNormalize statistics if available
        normalize_path = "models/vec_normalize.pkl"
        if os.path.exists(normalize_path):
            print(f"Loading VecNormalize statistics from {normalize_path}")
            eval_env = DummyVecEnv([lambda: TriageGymEnv(task_id=task_id, seed=42)])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
            eval_env.load_normalization(normalize_path)
            model = PPO.load(model_path, env=eval_env)
        else:
            print("No VecNormalize statistics found, loading model without normalization")
            model = PPO.load(model_path)
            eval_env = TriageGymEnv(task_id=task_id, seed=42)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}
    
    # Collect metrics
    final_scores = []
    total_rewards = []
    steps_taken = []
    correct_esi_counts = []
    total_patients = []
    critical_misses = []
    deteriorations = []
    passes = []
    esi_accuracies = []
    
    for episode in range(n_episodes):
        if isinstance(eval_env, (DummyVecEnv, VecNormalize)):
            obs = eval_env.reset()
        else:
            obs, _ = eval_env.reset()
        done = False
        episode_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(eval_env, (DummyVecEnv, VecNormalize)):
                obs, reward, terminated, info = eval_env.step(action)
                done = terminated
            else:
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            episode_steps += 1
        
        # Collect episode metrics
        final_scores.append(info.get("final_score", 0.0))
        total_rewards.append(info.get("total_reward", 0.0))
        steps_taken.append(info.get("steps_taken", episode_steps))
        correct_esi_counts.append(info.get("correct_esi_count", 0))
        total_patients.append(info.get("total_patients", 0))
        critical_misses.append(info.get("critical_misses", 0))
        deteriorations.append(info.get("deteriorations", 0))
        passes.append(1.0 if info.get("passed", False) else 0.0)
        
        # Calculate ESI accuracy
        total_pat = info.get("total_patients", 1)
        correct_esi = info.get("correct_esi_count", 0)
        esi_accuracies.append(correct_esi / total_pat if total_pat > 0 else 0.0)
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: Score = {info.get('final_score', 0.0):.3f}")
    
    env.close()
    
    # Calculate statistics
    results = {
        "task_id": task_id,
        "n_episodes": n_episodes,
        "final_score": {
            "mean": np.mean(final_scores),
            "std": np.std(final_scores),
            "min": np.min(final_scores),
            "max": np.max(final_scores),
        },
        "total_reward": {
            "mean": np.mean(total_rewards),
            "std": np.std(total_rewards),
        },
        "steps_taken": {
            "mean": np.mean(steps_taken),
            "std": np.std(steps_taken),
        },
        "esi_accuracy": {
            "mean": np.mean(esi_accuracies),
            "std": np.std(esi_accuracies),
        },
        "critical_misses": {
            "mean": np.mean(critical_misses),
            "std": np.std(critical_misses),
        },
        "deteriorations": {
            "mean": np.mean(deteriorations),
            "std": np.std(deteriorations),
        },
        "pass_rate": np.mean(passes),
    }
    
    return results


def evaluate_baseline(task_id: int, n_episodes: int = 20) -> Dict[str, Any]:
    """Evaluate baseline agent on specified task."""
    print(f"Evaluating baseline agent on Task {task_id}...")
    
    final_scores = []
    total_rewards = []
    steps_taken = []
    correct_esi_counts = []
    total_patients = []
    critical_misses = []
    deteriorations = []
    passes = []
    esi_accuracies = []
    
    for episode in range(n_episodes):
        result = run_task(task_id, seed=episode, verbose=False)
        
        final_scores.append(result["final_score"])
        total_rewards.append(result.get("total_reward", 0.0))
        steps_taken.append(result["steps_taken"])
        correct_esi_counts.append(result["correct_esi"])
        total_patients.append(result["total_patients"])
        critical_misses.append(result["critical_misses"])
        deteriorations.append(result["deteriorations"])
        passes.append(1.0 if result["passed"] else 0.0)
        
        # Calculate ESI accuracy
        total_pat = result["total_patients"]
        correct_esi = result["correct_esi"]
        esi_accuracies.append(correct_esi / total_pat if total_pat > 0 else 0.0)
        
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: Score = {result['final_score']:.3f}")
    
    # Calculate statistics
    results = {
        "task_id": task_id,
        "n_episodes": n_episodes,
        "final_score": {
            "mean": np.mean(final_scores),
            "std": np.std(final_scores),
            "min": np.min(final_scores),
            "max": np.max(final_scores),
        },
        "total_reward": {
            "mean": np.mean(total_rewards),
            "std": np.std(total_rewards),
        },
        "steps_taken": {
            "mean": np.mean(steps_taken),
            "std": np.std(steps_taken),
        },
        "esi_accuracy": {
            "mean": np.mean(esi_accuracies),
            "std": np.std(esi_accuracies),
        },
        "critical_misses": {
            "mean": np.mean(critical_misses),
            "std": np.std(critical_misses),
        },
        "deteriorations": {
            "mean": np.mean(deteriorations),
            "std": np.std(deteriorations),
        },
        "pass_rate": np.mean(passes),
    }
    
    return results


def print_comparison_table(baseline_results: Dict, ppo_results: Dict):
    """Print comparison table between baseline and PPO agents."""
    print(f"\n{'='*80}")
    print("PPO vs BASELINE COMPARISON")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Task':<6} {'Agent':<12} {'Score':<12} {'Std':<8} {'Pass Rate':<10} {'ESI Acc':<10} {'Crit Miss':<10} {'Det':<8}")
    print(f"{'─'*80}")
    
    task_names = {1: "Easy", 2: "Medium", 3: "Hard"}
    
    for task_id in [1, 2, 3]:
        baseline = baseline_results.get(task_id, {})
        ppo = ppo_results.get(task_id, {})
        
        # Baseline row
        baseline_score = baseline.get("final_score", {}).get("mean", 0.0)
        baseline_std = baseline.get("final_score", {}).get("std", 0.0)
        baseline_pass = baseline.get("pass_rate", 0.0)
        baseline_esi = baseline.get("esi_accuracy", {}).get("mean", 0.0)
        baseline_miss = baseline.get("critical_misses", {}).get("mean", 0.0)
        baseline_det = baseline.get("deteriorations", {}).get("mean", 0.0)
        
        print(f"Task {task_id}  {'Baseline':<12} {baseline_score:<12.3f} {baseline_std:<8.3f} "
              f"{baseline_pass:<10.2f} {baseline_esi:<10.3f} {baseline_miss:<10.2f} {baseline_det:<8.2f}")
        
        # PPO row
        if "error" not in ppo:
            ppo_score = ppo.get("final_score", {}).get("mean", 0.0)
            ppo_std = ppo.get("final_score", {}).get("std", 0.0)
            ppo_pass = ppo.get("pass_rate", 0.0)
            ppo_esi = ppo.get("esi_accuracy", {}).get("mean", 0.0)
            ppo_miss = ppo.get("critical_misses", {}).get("mean", 0.0)
            ppo_det = ppo.get("deteriorations", {}).get("mean", 0.0)
            
            # Calculate deltas
            score_delta = ppo_score - baseline_score
            delta_str = f"{score_delta:+.3f}"
            
            print(f"        {'PPO':<12} {ppo_score:<12.3f} {ppo_std:<8.3f} "
                  f"{ppo_pass:<10.2f} {ppo_esi:<10.3f} {ppo_miss:<10.2f} {ppo_det:<8.2f} ({delta_str})")
        else:
            print(f"        {'PPO':<12} {'N/A':<12} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8}")
        
        print(f"{'─'*80}")
    
    # Overall averages
    baseline_avg = np.mean([baseline_results[tid]["final_score"]["mean"] for tid in [1, 2, 3]])
    if all("error" not in ppo_results[tid] for tid in [1, 2, 3]):
        ppo_avg = np.mean([ppo_results[tid]["final_score"]["mean"] for tid in [1, 2, 3]])
        overall_delta = ppo_avg - baseline_avg
        delta_str = f"{overall_delta:+.3f}"
        print(f"{'AVG':<6} {'Baseline':<12} {baseline_avg:<12.3f} {'-':<8} {'-':<10} {'-':<10} {'-':<10} {'-':<8}")
        print(f"        {'PPO':<12} {ppo_avg:<12.3f} {'-':<8} {'-':<10} {'-':<10} {'-':<10} {'-':<8} ({delta_str})")
    
    print(f"{'='*80}")


def plot_learning_curves(log_dir: str = "logs"):
    """Plot learning curves from Monitor CSV logs."""
    if not RL_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    print(f"\nGenerating learning curves from {log_dir}...")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("TriageNet-RL Learning Curves", fontsize=16)
    
    task_names = {1: "Task 1 (Easy)", 2: "Task 2 (Medium)", 3: "Task 3 (Hard)"}
    baseline_scores = {1: 0.68, 2: 0.61, 3: 0.52}  # From baseline results
    
    for task_id in [1, 2, 3]:
        ax = axes[task_id - 1]
        
        # Try to load monitor data
        monitor_file = f"{log_dir}/monitor/train.task{task_id}.monitor.csv"
        
        if os.path.exists(monitor_file):
            try:
                # Read monitor data
                data = pd.read_csv(monitor_file, skiprows=1)
                episodes = data['r'].values  # Episode returns
                timesteps = data['l'].values  # Episode lengths
                
                # Calculate moving average
                window = min(100, len(episodes) // 4)
                if window > 1:
                    moving_avg = pd.Series(episodes).rolling(window=window, min_periods=1).mean()
                    moving_std = pd.Series(episodes).rolling(window=window, min_periods=1).std()
                    
                    x = np.arange(len(moving_avg))
                    ax.plot(x, moving_avg, label='PPO', color='blue', linewidth=2)
                    ax.fill_between(x, moving_avg - moving_std, moving_avg + moving_std, 
                                   alpha=0.3, color='blue')
                else:
                    ax.plot(episodes, label='PPO', color='blue', linewidth=2)
                
                # Add baseline reference line
                ax.axhline(y=baseline_scores[task_id], color='red', linestyle='--', 
                          label=f'Baseline ({baseline_scores[task_id]:.2f})', linewidth=2)
                
            except Exception as e:
                print(f"  Could not load {monitor_file}: {e}")
                ax.text(0.5, 0.5, f"No data available\n{e}", 
                       transform=ax.transAxes, ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f"No monitor data\n{monitor_file}\nnot found", 
                   transform=ax.transAxes, ha='center', va='center')
        
        ax.set_title(task_names[task_id])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/learning_curves.png", dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to results/learning_curves.png")
    
    # Show plot if running interactively
    if os.getenv("DISPLAY"):  # Check if display is available
        plt.show()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate PPO agent vs baseline")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], 
                       help="Evaluate single task (default: all)")
    parser.add_argument("--model", type=str, 
                       help="Path to trained PPO model")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of evaluation episodes for PPO")
    parser.add_argument("--baseline-episodes", type=int, default=20,
                       help="Number of evaluation episodes for baseline")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip learning curve plotting")
    parser.add_argument("--no-json", action="store_true",
                       help="Skip saving results to JSON")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory containing monitor logs")
    
    args = parser.parse_args()
    
    if not RL_AVAILABLE:
        print("❌ RL libraries not available")
        print("Install with: pip install -r rl/requirements_rl.txt")
        return
    
    # Register environments
    register_envs()
    
    # Determine which tasks to evaluate
    tasks = [args.task] if args.task else [1, 2, 3]
    
    # Evaluate baseline
    print("🔍 Evaluating Baseline Agent")
    baseline_results = {}
    for task_id in tasks:
        baseline_results[task_id] = evaluate_baseline(task_id, args.baseline_episodes)
    
    # Evaluate PPO
    ppo_results = {}
    if args.model:
        print("\n🤖 Evaluating PPO Agent")
        for task_id in tasks:
            ppo_results[task_id] = evaluate_ppo_agent(task_id, args.model, args.episodes)
    else:
        # Try to find trained models
        for task_id in tasks:
            model_path = f"models/task{task_id}_final.zip"
            if os.path.exists(model_path):
                print(f"\n🤖 Evaluating PPO Agent (Task {task_id})")
                ppo_results[task_id] = evaluate_ppo_agent(task_id, model_path, args.episodes)
            else:
                print(f"⚠️  No trained model found for Task {task_id} at {model_path}")
                ppo_results[task_id] = {"error": "Model not found"}
    
    # Print comparison
    print_comparison_table(baseline_results, ppo_results)
    
    # Save results
    if not args.no_json:
        os.makedirs("results", exist_ok=True)
        
        combined_results = {
            "baseline": baseline_results,
            "ppo": ppo_results,
            "evaluation_params": {
                "ppo_episodes": args.episodes,
                "baseline_episodes": args.baseline_episodes,
                "model_path": args.model
            }
        }
        
        with open("results/evaluation_results.json", "w") as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to results/evaluation_results.json")
    
    # Plot learning curves
    if not args.no_plot:
        plot_learning_curves(args.log_dir)


if __name__ == "__main__":
    main()
