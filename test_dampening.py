from rl.environment_wrapper import TriageGymEnv

print("Testing reward dampening...")

# Test Task 3 (should be dampened)
env = TriageGymEnv(task_id=3, seed=42)
obs, _ = env.reset()
obs, reward, done, _, info = env.step(0)
print(f'Task 3 step reward (dampened): {reward:.4f}')
print(f'Original step reward: {info["step_reward"]:.4f}')

# Test Task 1 (should not be dampened)
env2 = TriageGymEnv(task_id=1, seed=42)
obs, _ = env2.reset()
obs, reward2, _, _, info2 = env2.step(0)
print(f'Task 1 step reward (not dampened): {reward2:.4f}')

if reward != reward2:
    print('Dampening verified ✅')
else:
    print('WARNING: dampening not applied')
