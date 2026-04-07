#!/usr/bin/env python3
"""Quick test of Task 3 PPO agent performance."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.environment_wrapper import TriageGymEnv

def test_task3():
    print("Testing Task 3 PPO agent...")
    
    # Load VecNormalize statistics
    eval_env = DummyVecEnv([lambda: TriageGymEnv(task_id=3, seed=42)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env = VecNormalize.load("models/vec_normalize.pkl", eval_env)
    
    # Load model
    model = PPO.load("models/task3_final.zip", env=eval_env)
    
    # Test episodes
    scores = []
    for episode in range(5):
        obs = eval_env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = eval_env.step(action)
            done = terminated
            steps += 1
            
            if steps > 100:  # Safety break
                break
        
        # Get final score from the last info dict
        if isinstance(info, dict) and "final_score" in info:
            score = info["final_score"]
        elif isinstance(info, list) and len(info) > 0 and "final_score" in info[0]:
            score = info[0]["final_score"]
        else:
            score = 0.0
            
        scores.append(score)
        print(f"Episode {episode+1}: Score = {score:.3f}, Steps = {steps}")
    
    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Score: {avg_score:.3f}")
    print(f"Target: >0.58, Current: {avg_score:.3f}")
    
    if avg_score > 0.58:
        print("🎉 SUCCESS - Task 3 improved!")
    else:
        print("❌ Still needs improvement")

if __name__ == "__main__":
    test_task3()
