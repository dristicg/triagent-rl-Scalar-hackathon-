---
title: Triagenet Rl
emoji: 🏢
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
app_port: 7860
license: mit
short_description: 'AI agent learns emergency department triage '
---

# 🏥 TriageNet-RL: LLM-Native Emergency Triage Agent

TriageNet-RL is a high-performance, medical triage simulation environment and AI agent designed for the Emergency Severity Index (ESI) standard. It features a complete reinforcement learning pipeline and a production-ready LLM-native inference mode.

## 🚀 Quick Start (Inference Mode)

The primary agent uses Chain-of-Thought reasoning to assess patient vitals and assign ESI levels in real-time.

### Deployment & API
The system is built as a FastAPI + Gradio application, exposing a standard OpenEnv protocol:
- **`POST /reset`**: Initialize a new triage task (Easy, Medium, Hard).
- **`POST /step`**: Submit a triage action and receive the next observation.
- **`GET /ui`**: Interactive Gradio dashboard.

To run locally:
```bash
# Set your Hugging Face Token (if using HF Router)
export HF_TOKEN="your_token_here"
python app.py
```

## 🧠 Architecture Overview

```text
Observation (structured) → FeatureExtractor (30-dim) → PPO MLP(128→128→64) → Discrete(11) → ActionMapper → Action → Environment → Reward
```

The system provides a complete reinforcement learning pipeline with:
- **Feature Extraction**: Converts clinical observations into 30-dimensional vectors.
- **Action Mapping**: Maps discrete agent decisions to structured medical actions.
- **Curriculum Learning**: Progresses from single-patient scenarios to mass casualty surges.

## 📈 Learning Rationale

The agent is trained using **Proximal Policy Optimization (PPO)** because it is stable for clinical decision-making and sample-efficient.

### Curriculum Difficulty:
1. **Task 1 (Easy)**: Single patient, basic ESI classification.
2. **Task 2 (Medium)**: Multiple patients, priority ordering.
3. **Task 3 (Hard)**: Mass casualty surge, resource constraints, and patient deterioration.

## 🛠️ Feature Engineering

The agent processes **30 distinct dimensions** including:
- **Vitals**: BP, Heart Rate, SpO2 (Normalized).
- **History**: Age, gender, and clinical complaints.
- **Keywords**: Flags for critical symptoms (e.g., *cardiac*, *seizure*, *unconscious*).
- **Queue Status**: Wait times and bed availability.

## 📊 Performance Baselines

| Task | Baseline | PPO Trainee | LLM Agent (Expert) |
|------|---------|-------------|-------------------|
| Task 1 | 0.680 | ~0.85 | **1.000** |
| Task 2 | 0.610 | ~0.75 | **0.962** |
| Task 3 | 0.520 | ~0.65 | **0.274** |

## 🧪 Development & Training

If you wish to retrain the PPO models:
```bash
# Install RL dependencies
pip install -r rl/requirements_rl.txt

# Start curriculum training
python rl/train_ppo.py --curriculum --seed 42
```

---
**TriageNet-RL - Advancing AI in Emergency Clinical Logistics**
