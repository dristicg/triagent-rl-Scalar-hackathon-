import os
import json
import traceback
from typing import List, Optional
from openai import OpenAI
from env import MedicalTriageEnv
from schemas import TriageAction as Action

# ── Environment Configuration ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not set. LLM calls will fail. Using placeholder for local testing.")
    HF_TOKEN = "sk-placeholder"

# Initialize OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert ER triage nurse managing a highly constrained environment.

SOP 1: Identify critical patients immediately. If a patient's vitals indicate deterioration (e.g., severe pain, low SpO2, unresponsiveness), assign ESI Level 1 or 2 immediately.
SOP 2: Do not use the 'wait' action if there are untriaged patients or if you have pending resources you can allocate.
SOP 3: Manage beds carefully. Do not route stable patients to the resuscitation_bay.
SOP 4: You MUST fill out the reasoning field in your JSON response first, explaining your logic step-by-step, before outputting the final action.

Valid action types: "assign_esi", "request_resources", "wait".
The output must be EXACTLY this JSON format:
{
  "reasoning": "Briefly explain your step-by-step clinical logic, time management, and bed availability reasoning before deciding on the action.",
  "action_type": "assign_esi",
  "patient_id": "P00X",
  "esi_level": 3,
  "routing_zone": "acute_care",
  "notes": "brief clinical notes"
}
ESI Levels: 1 (Immediate), 2 (Emergent), 3 (Urgent), 4 (Less Urgent), 5 (Non-Urgent).
Routing Zones: resuscitation_bay, acute_care, fast_track, waiting_room."""

def run_episode(env: MedicalTriageEnv, task_name: str):
    """Runs a single episode loop against the environment."""
    print(f"[START] task={task_name} env=MedicalTriage model={MODEL_NAME}")
    
    obs = env.reset()
    step_idx = 0
    done = False
    reward_history = []
    
    while not done:
        step_idx += 1
        obs_dict = obs.model_dump()
        pending_patients = obs_dict.get("pending_patients", [])
        
        if not pending_patients:
            break
            
        current_patient_id = pending_patients[0]
        error_msg = "null"
        action_str = "wait"
        reward_score = 0.0
        
        try:
            # LLM Inference
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current State: {obs.model_dump_json()}\n\nTriage the next patient: {current_patient_id}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            llm_text = completion.choices[0].message.content or "{}"
            llm_dict = json.loads(llm_text)
            
            # Ensure patient_id is correct
            llm_dict["patient_id"] = current_patient_id
            
            # Create Action model
            action = Action(**llm_dict)
            action_str = f"{action.action_type}({action.esi_level or ''})"
            
            # Environment Step
            obs, reward, done, info = env.step(action)
            reward_score = reward.score
            
        except Exception as e:
            # Sanitize error message (no newlines)
            error_msg = str(e).replace("\n", " ")
            # Fallback action
            fallback_action = Action(action_type="wait", patient_id=current_patient_id)
            try:
                obs, reward, done, info = env.step(fallback_action)
                reward_score = reward.score
                action_str = "wait"
            except:
                done = True
        
        reward_history.append(reward_score)
        
        # [STEP] log
        print(f"[STEP] step={step_idx} action={action_str} reward={reward_score:.2f} done={str(done).lower()} error={error_msg}")

    # [END] log
    rewards_str = ",".join([f"{r:.2f}" for r in reward_history])
    success = "true" if env._score > 0 else "false" # Simple success metric or use env.grade()
    print(f"[END] success={success} steps={step_idx} rewards={rewards_str}")


if __name__ == "__main__":
    tasks = [
        (1, "Easy"),
        (2, "Medium"),
        (3, "Hard")
    ]
    
    for task_id, task_name in tasks:
        try:
            env = MedicalTriageEnv(task_id=task_id)
            run_episode(env, task_name)
        except Exception as e:
            # Fatal environment error
            pass