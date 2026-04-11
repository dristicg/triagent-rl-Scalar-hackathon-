import os
import json
import traceback
from typing import List, Optional
from openai import OpenAI
from env import MedicalTriageEnv
from schemas import TriageAction

# ── Environment Configuration ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# The API key for LLM calls is always HF_TOKEN
# routed through API_BASE_URL (HuggingFace router)
API_KEY = HF_TOKEN

TASK_NAMES = {1: "Easy", 2: "Medium", 3: "Hard"}

def get_client():
    """
    Create OpenAI client pointed at HuggingFace router.
    Always uses API_BASE_URL as base, HF_TOKEN as key.
    Never points directly at api.openai.com.
    """
    if not HF_TOKEN:
        return None
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

def get_fallback_action(obs_dict: dict, patient_id: str):
    """
    Rule-based fallback when LLM API is unavailable.
    Uses vitals to make a reasonable triage decision.
    Much better than action=wait which scores -0.08 every step.
    """
    from models import Action
    patient = obs_dict.get("current_patient") or {}
    vitals  = patient.get("vitals", {})

    gcs    = vitals.get("gcs", 15)
    spo2   = vitals.get("spo2", 98)
    hr     = vitals.get("heart_rate", 80)
    bp_sys = vitals.get("bp_systolic", 120)
    pain   = vitals.get("pain_scale", 0)

    # ESI classification from vitals
    if gcs <= 8 or spo2 < 85 or bp_sys < 80 or hr == 0:
        esi, zone = 1, "resuscitation_bay"
    elif spo2 < 92 or hr > 120 or bp_sys > 180 or pain >= 9:
        esi, zone = 2, "acute_care"
    elif pain >= 5 or hr > 100 or bp_sys > 150:
        esi, zone = 3, "fast_track"
    elif pain >= 2:
        esi, zone = 4, "waiting_room"
    else:
        esi, zone = 5, "waiting_room"

    return Action(
        action_type="assign_esi",
        patient_id=patient_id,
        esi_level=esi,
        routing_zone=zone,
        notes="Fallback: vitals-based classification",
    )

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

def run_episode(env: MedicalTriageEnv, task_id: int):
    """Runs a single episode loop against the environment."""
    task_name = TASK_NAMES.get(task_id, "Unknown")
    print(f"[START] task={task_name} env=MedicalTriage model={MODEL_NAME}", flush=True)
    
    obs = env.reset()
    step_idx = 0
    done = False
    reward_history = []
    client = get_client()
    
    while not done:
        step_idx += 1
        obs_dict = obs.model_dump()
        pending_patients = obs_dict.get("pending_patients", [])
        
        if not pending_patients:
            break
            
        current_patient_id = pending_patients[0]
        error_msg = ""
        action_str = "wait"
        reward_score = 0.0
        
        try:
            if not client:
                raise ValueError("No HF_TOKEN provided")
                
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
            from models import Action
            action = Action(**llm_dict)
            action_str = f"{action.action_type}"
            
            # Environment Step
            obs, reward, done, info = env.step(action)
            reward_score = reward.score
            
        except Exception as e:
            error_msg = str(e).replace("\n", " ")
            # Fallback action
            action = get_fallback_action(obs_dict, current_patient_id)
            action_str = action.action_type
            
            try:
                obs, reward, done, info = env.step(action)
                reward_score = reward.score
            except:
                done = True
        
        reward_history.append(reward_score)
        
        # [STEP] log
        print(f"[STEP] step={step_idx} action={action_str} reward={reward_score:.2f} done={str(done).lower()} error={error_msg}", flush=True)

    # [END] log
    rewards_str = ",".join([f"{r:.2f}" for r in reward_history])
    
    # success=true only if final_score > pass_threshold
    try:
        grade = env.grade()
        success = "true" if grade.passed else "false"
    except Exception as e:
        success = "false"
    
    print(f"[END] success={success} steps={step_idx} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    for task_id in [1, 2, 3]:
        try:
            env = MedicalTriageEnv(task_id=task_id)
            run_episode(env, task_id)
        except Exception as e:
            pass