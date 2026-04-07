# TriageNet-RL Reinforcement Learning Module



## Architecture Overview

```
Observation (structured) → FeatureExtractor (30-dim) → PPO MLP(128→128→64) → Discrete(11) → ActionMapper → Action → Environment → Reward
```

The RL module provides a complete reinforcement learning pipeline for TriageNet-RL with:

- **Feature Extraction**: Converts structured observations to 30-dimensional feature vectors
- **Action Mapping**: Maps 11 discrete actions to structured Action objects with masking
- **Environment Wrapper**: Gymnasium-compatible interface for standard RL algorithms
- **PPO Training**: Proximal Policy Optimization with curriculum learning
- **Evaluation**: Comprehensive comparison against baseline agent

## Why PPO?

Proximal Policy Optimization (PPO) is ideal for TriageNet-RL because:

- **On-policy**: Stable learning with clinical decision-making
- **Sample efficient**: Good performance with limited episodes
- **Entropy bonus**: Encourages exploration in complex triage decisions
- **Curriculum compatible**: Progressive difficulty from Task 1 → Task 2 → Task 3
- **Robust**: Handles sparse rewards and partial credit well

## Curriculum Learning Rationale

The curriculum progresses from simple to complex scenarios:

1. **Task 1 (Easy)**: Single patient, basic ESI classification
2. **Task 2 (Medium)**: Multiple patients, priority ordering
3. **Task 3 (Hard)**: Mass casualty, resource constraints, deterioration

This progression allows the agent to:
- Learn basic ESI patterns first
- Master patient prioritization
- Handle complex resource management
- Develop time-pressure strategies

## Setup

### 1. Install Dependencies

```bash
pip install -r rl/requirements_rl.txt
```

### 2. Register Environments

```python
from rl.environment_wrapper import register_envs
register_envs()  # Registers MedicalTriage-Task1-v1, Task2-v1, Task3-v1
```

## Training Commands

### Full Curriculum Training

```bash
python rl/train_ppo.py --curriculum --seed 42
```

This trains sequentially:
- Task 1: 50,000 timesteps
- Task 2: 150,000 timesteps  
- Task 3: 300,000 timesteps
- Total: 500,000 timesteps

### Single Task Training

```bash
# Train Task 1 only
python rl/train_ppo.py --task 1 --steps 100000

# Train Task 2 with pretrained model
python rl/train_ppo.py --task 2 --load models/task1_final.zip

# Train Task 3 with custom seed
python rl/train_ppo.py --task 3 --seed 123 --steps 200000
```

### TensorBoard Monitoring

```bash
tensorboard --logdir logs/tensorboard/
```

Monitor training metrics:
- `triage/mean_final_score`: Average episode score
- `triage/critical_miss_rate`: Critical miss frequency
- `triage/deterioration_rate`: Patient deterioration rate
- `triage/esi_accuracy`: ESI classification accuracy
- `triage/pass_rate`: Episode pass rate

## Evaluation Commands

### Compare PPO vs Baseline

```bash
# Evaluate all tasks
python rl/evaluate.py

# Evaluate specific task with custom model
python rl/evaluate.py --task 2 --model models/task2_final.zip

# Custom evaluation parameters
python rl/evaluate.py --episodes 100 --baseline-episodes 50 --no-plot
```

### Generate Learning Curves

```bash
python rl/evaluate.py --no-json --log-dir logs
```

## Expected Performance

| Task | Baseline | PPO (Expected) | Improvement |
|------|---------|----------------|-------------|
| Task 1 | 0.680 | ~0.85 | +0.17 |
| Task 2 | 0.610 | ~0.75 | +0.14 |
| Task 3 | 0.520 | ~0.65 | +0.13 |
| **Average** | **0.603** | **~0.75** | **+0.15** |

### Training Time Estimates

| Hardware | Task 1 | Task 2 | Task 3 | Total |
|----------|--------|--------|--------|-------|
| CPU (8 cores) | ~5 min | ~15 min | ~30 min | ~50 min |
| GPU (RTX 3080) | ~2 min | ~6 min | ~12 min | ~20 min |

## Feature Engineering

### Feature Vector (30 dimensions)

**Vitals (8 features)**:
- Normalized blood pressure, heart rate, respiratory rate, SpO2, temperature, GCS, pain

**Patient Context (4 features)**:
- Age, gender, wait time, deterioration flag

**Complaint Keywords (10 features)**:
- Binary flags for cardiac, seizure, headache, chest, breath, weakness, bleed, pain, unconscious, stroke

**Queue Status (5 features)**:
- Pending/triaged counts, bed availability, time elapsed

**Step Info (3 features)**:
- Progress, cumulative score, alert activity

### Action Space (11 discrete actions)

| Action | Description | ESI | Routing |
|--------|-------------|-----|---------|
| 0 | assign_esi=1 | 1 | resuscitation_bay |
| 1 | assign_esi=2 | 2 | acute_care |
| 2 | assign_esi=3 | 3 | fast_track |
| 3 | assign_esi=4 | 4 | waiting_room |
| 4 | assign_esi=5 | 5 | waiting_room |
| 5 | request_resources | - | [ecg, iv_access, blood_panel] |
| 6 | request_resources | - | [chest_xray, blood_panel] |
| 7 | escalate | - | - |
| 8 | reassess | - | - |
| 9 | discharge | - | - |
| 10 | wait | - | - |

### Action Masking

The action mapper ensures only valid actions are available:
- ESI assignment only for untriaged patients
- Reassessment only after 10+ minutes or deterioration
- Wait always available as fallback

## Model Architecture

```python
policy_kwargs = dict(net_arch=[128, 128, 64])
```

- **Input**: 30-dimensional feature vector
- **Hidden layers**: 128 → 128 → 64 neurons with ReLU
- **Output**: 11 discrete action logits
- **Activation**: Tanh for final layer (stable training)

## Training Hyperparameters

```python
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}
```

## Advanced Usage

### Custom Feature Extractor

```python
from rl.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract(observation_dict)
print(f"Feature names: {extractor.get_feature_names()}")
```

### Custom Action Mapping

```python
from rl.action_mapper import ActionMapper

mapper = ActionMapper()
mask = mapper.get_action_mask(observation_dict)
action = mapper.sample_valid(observation_dict)
structured_action = mapper.int_to_action(action, patient_id)
```

### Environment Wrapping

```python
from rl.environment_wrapper import TriageGymEnv

env = TriageGymEnv(task_id=2, seed=42)
obs, info = env.reset()
action, _ = model.predict(obs)
next_obs, reward, terminated, truncated, info = env.step(action)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Install RL dependencies with `pip install -r rl/requirements_rl.txt`
2. **CUDA Out of Memory**: Reduce batch size or use CPU training
3. **Slow Training**: Enable GPU or reduce timesteps
4. **Poor Performance**: Increase learning rate or adjust network architecture

### Debug Mode

```bash
# Enable verbose training
python rl/train_ppo.py --task 1 --steps 1000 --verbose 2

# Debug evaluation
python rl/evaluate.py --task 1 --episodes 5 --baseline-episodes 3
```

### Model Checkpoints

Training automatically saves checkpoints:
- `models/task{N}_checkpoints/`: Every 25,000 steps
- `models/task{N}_best/`: Best performing model
- `models/task{N}_final.zip`: Final model after training

## Results Analysis

### Key Metrics

- **Final Score**: 0.0-1.0, higher is better
- **Pass Rate**: Fraction of episodes passing threshold
- **ESI Accuracy**: Correct ESI classification rate
- **Critical Miss Rate**: Dangerous under-triage frequency
- **Deterioration Rate**: Patients who worsened while waiting

### Learning Curves

Monitor learning progress:
- Early plateau: Consider increasing learning rate
- High variance: Increase batch size or reduce learning rate
- No improvement: Check feature extraction or reward shaping

### Comparison Analysis

The evaluation script provides detailed comparisons:
- Score improvements over baseline
- Statistical significance with standard deviations
- Failure mode analysis (critical misses, deteriorations)

## Future Extensions

### Potential Improvements

1. **Advanced Architectures**: Transformer-based policies for sequence modeling
2. **Multi-Objective Learning**: Separate networks for ESI vs routing decisions  
3. **Curriculum Optimization**: Adaptive task selection based on performance
4. **Ensemble Methods**: Multiple agents with voting mechanisms
5. **Transfer Learning**: Pre-training on synthetic data

### Research Directions

1. **Interpretability**: Attention maps for clinical decision reasoning
2. **Safety Constraints**: Formal verification of critical case handling
3. **Curriculum Learning**: Automated difficulty progression
4. **Multi-Agent**: Multiple triage nurses coordination
5. **Real-World Validation**: Clinical simulation center testing

---

**TriageNet-RL RL Module - Advanced triage agent training and evaluation**
