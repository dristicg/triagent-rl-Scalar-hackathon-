import os
import json
import sys
from typing import Optional

from huggingface_hub import get_token
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import MedicalTriageEnv
from models import Action, VALID_ROUTING_ZONES

# Environment variables (never hardcoded)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
# Prefer explicit env var in deployment, then fallback to local HF CLI token.
HF_TOKEN = os.getenv("HF_TOKEN") or get_token()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Constants
MAX_STEPS_PER_TASK = 30
TEMPERATURE = 0.1
MAX_TOKENS = 300
RESULTS_DIR = "results"
BASELINE_SCORES = {1: 0.68, 2: 0.61, 3: 0.52}

# System prompt
SYSTEM_PROMPT = """You are an expert emergency department triage nurse with 15 years of experience. You will be given a patient's clinical information and must make a triage decision using Emergency Severity Index (ESI).

ESI LEVELS:
- ESI 1 (Immediate): Cardiac arrest, unresponsive, GCS ≤ 8, SpO2 < 85%, BP systolic < 80, active seizure, no pulse. Response: NOW.
- ESI 2 (Emergent): SpO2 < 92%, HR > 120, BP > 180, severe chest pain radiating to arm, worst headache of life, stroke symptoms (facial droop, arm weakness, slurred speech), severe asthma. Response: < 15 min.
- ESI 3 (Urgent): Moderate pain 5-8/10, stable vitals but needs workup, abdominal pain, blood in urine, persistent vomiting. Response: < 30 min.
- ESI 4 (Less Urgent): Minor complaint, mild symptoms. Response: < 60 min.
- ESI 5 (Non-Urgent): Prescription refill, minor cut. Response: when available.

ROUTING ZONES:
- resuscitation_bay  → ESI 1 only
- acute_care         → ESI 2
- fast_track         → ESI 3
- waiting_room       → ESI 4 and 5

You MUST respond with ONLY a valid JSON object. No text before or after. No markdown. No explanation. Just JSON:
{
    "action_type": "assign_esi",
    "esi_level": <integer 1-5>,
    "routing_zone": "<one of the four zones above>",
    "notes": "<one sentence of clinical reasoning>"
}"""


def build_patient_prompt(obs_dict: dict) -> str:
    """Takes observation dict from obs.model_dump() and returns a formatted string describing current patient."""
    patient = obs_dict.get("current_patient")
    if patient is None:
        return "No patient currently available."

    vitals = patient.get("vitals", {})

    # Format the patient information string exactly as specified
    prompt = f"""PATIENT TRIAGE REQUEST
─────────────────────────────────────
Patient ID  : {patient.get('patient_id', 'Unknown')}
Name        : {patient.get('name', 'Unknown')}
Age / Gender: {patient.get('age', 'Unknown')} / {patient.get('gender', 'Unknown')}

CHIEF COMPLAINT:
"{patient.get('chief_complaint', 'Unknown')}"

VITAL SIGNS:
Blood Pressure  : {vitals.get('bp_systolic', 'Unknown')}/{vitals.get('bp_diastolic', 'Unknown')} mmHg
Heart Rate      : {vitals.get('heart_rate', 'Unknown')} bpm
Respiratory Rate: {vitals.get('respiratory_rate', 'Unknown')} /min
SpO2            : {vitals.get('spo2', 'Unknown')}%
Temperature     : {vitals.get('temperature', 'Unknown')}°C
GCS             : {vitals.get('gcs', 'Unknown')}/15
Pain Scale      : {vitals.get('pain_scale', 'Unknown')}/10

CLINICAL HISTORY:
Medical History : {', '.join(patient.get('medical_history', [])) or 'None'}
Medications     : {', '.join(patient.get('current_medications', [])) or 'None'}

QUEUE STATUS:
Wait Time       : {obs_dict.get('time_elapsed_minutes', 'Unknown')} minutes
Patients Pending: {len(obs_dict.get('pending_patients', []))}
Alerts          : {'; '.join(obs_dict.get('alerts', [])) if obs_dict.get('alerts') else 'None'}

Resus bays available: {obs_dict.get('bed_status', {}).get('resuscitation_available', '?')}
Acute care beds     : {obs_dict.get('bed_status', {}).get('acute_care_available', '?')}

Based on this patient's presentation, provide your triage decision as JSON."""
    
    return prompt


def parse_llm_response(response_text: str, patient_id: str) -> Action:
    """Parses LLM's JSON response into a structured Action object."""
    try:
        # Strip whitespace from response_text
        response_text = response_text.strip()
        
        # Try to find JSON by looking for first { and last }
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
            
        json_str = response_text[json_start:json_end]
        parsed = json.loads(json_str)
        
        # Extract values with defaults
        esi_level = int(parsed.get("esi_level", 3))
        routing_zone = str(parsed.get("routing_zone", "fast_track"))
        notes = str(parsed.get("notes", ""))
        
        # Validate values
        if esi_level not in [1, 2, 3, 4, 5]:
            esi_level = 3
        if routing_zone not in VALID_ROUTING_ZONES:
            routing_zone = "fast_track"
        
        return Action(
            action_type="assign_esi",
            patient_id=patient_id,
            esi_level=esi_level,
            routing_zone=routing_zone,
            notes=notes
        )
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response text: {response_text[:200]}...")
        # Return fallback action
        return Action(action_type="wait", patient_id=patient_id)


def run_llm_task(task_id: int, client: OpenAI, verbose: bool = True) -> dict:
    """Runs LLM agent on one task and returns a result dict."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = MedicalTriageEnv(task_id=task_id, seed=42)
    obs = env.reset()

    step = 0
    total_reward = 0.0
    api_errors = 0

    # Print header
    task_info = {
        1: ("Easy", 1),
        2: ("Medium", 8), 
        3: ("Hard", 15)
    }
    difficulty, patient_count = task_info[task_id]
    
    print(f"[START] task_id={task_id} difficulty={difficulty} max_steps={MAX_STEPS_PER_TASK}")
    
    if verbose:
        print(f"  Task {task_id} | Difficulty: {difficulty}")
        print(f"  Patients: {patient_count} | Max Steps: {MAX_STEPS_PER_TASK}")
        print(f"  Model: {MODEL_NAME}")
        print("  " + "="*50)

    while not env._done and step < MAX_STEPS_PER_TASK:
        obs_dict = obs.model_dump()
        pending = obs_dict.get("pending_patients", [])

        if not pending:
            break

        patient_id = pending[0]
        patient_prompt = build_patient_prompt(obs_dict)
        step += 1

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": patient_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            if verbose:
                print(f"  Step {step}: API error — {e}")
            api_errors += 1
            response_text = ""

        action = parse_llm_response(response_text, patient_id)
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward.step_reward
        
        print(f"[STEP] step={step} patient_id={patient_id} "
              f"action={action.action_type} esi={action.esi_level or '-'} "
              f"routing={action.routing_zone or '-'} "
              f"reward={result.reward.step_reward:+.3f} "
              f"feedback={result.reward.feedback[:40].strip()}")

        if verbose:
            print(f"  Step {step:02d} | Patient: {patient_id} | "
                  f"ESI: {action.esi_level or '-'} | "
                  f"Zone: {action.routing_zone or '-'} | "
                  f"Reward: {result.reward.step_reward:+.3f} | "
                  f"Feedback: {result.reward.feedback[:60]}")

    grade = env.grade()
    
    print(f"[END] task_id={task_id} score={grade.final_score:.3f} "
          f"passed={grade.passed} steps={step} "
          f"correct_esi={grade.correct_esi_count}/{grade.total_patients} "
          f"critical_misses={grade.critical_misses}")

    if verbose:
        print("  " + "="*50)
        print(f"  Final Score  : {grade.final_score:.3f}")
        print(f"  Correct ESI  : {grade.correct_esi_count}/{grade.total_patients}")
        print(f"  Crit Misses  : {grade.critical_misses}")
        print(f"  Passed       : {'YES' if grade.passed else 'NO'}")
        print(f"  API Errors   : {api_errors}")

    return {
        "task_id": task_id,
        "model": MODEL_NAME,
        "final_score": grade.final_score,
        "total_reward": round(total_reward, 4),
        "correct_esi": grade.correct_esi_count,
        "total_patients": grade.total_patients,
        "critical_misses": grade.critical_misses,
        "deteriorations": grade.deteriorations,
        "passed": grade.passed,
        "steps_taken": step,
        "api_errors": api_errors,
        "grade_breakdown": grade.grade_breakdown,
    }


def print_summary_table(llm_results: dict) -> None:
    """Prints a formatted comparison table."""
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    print("\n" + "="*55)
    print("  TriageNet-RL — LLM Agent vs Baseline Comparison")
    print(f"  Model: {MODEL_NAME}")
    print("="*55)
    print("  Task        Difficulty   Baseline   LLM Agent   Delta")
    print("  " + "-"*45)

    total_baseline = 0
    total_llm = 0
    task_count = 0

    for task_id in [1, 2, 3]:
        difficulty = ["Easy", "Medium", "Hard"][task_id - 1]
        baseline = BASELINE_SCORES[task_id]
        
        if task_id in llm_results:
            llm_score = llm_results[task_id].get("final_score", 0.0)
        else:
            llm_score = 0.0
        
        delta = llm_score - baseline
        
        # Color coding
        if delta >= 0:
            delta_str = f"{GREEN}+{delta:.3f}{RESET}"
        else:
            delta_str = f"{RED}{delta:.3f}{RESET}"
        
        print(f"  Task {task_id}      {difficulty:<11} {baseline:.3f}      {llm_score:.3f}       {delta_str}")
        
        total_baseline += baseline
        total_llm += llm_score
        task_count += 1

    avg_baseline = total_baseline / task_count
    avg_llm = total_llm / task_count
    avg_delta = avg_llm - avg_baseline
    
    if avg_delta >= 0:
        avg_delta_str = f"{GREEN}+{avg_delta:.3f}{RESET}"
    else:
        avg_delta_str = f"{RED}{avg_delta:.3f}{RESET}"

    print("  " + "-"*45)
    print(f"  Average     —            {avg_baseline:.3f}      {avg_llm:.3f}       {avg_delta_str}")
    print("="*55)


def main() -> None:
    """Main function to run LLM inference across all tasks."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Print banner
    print("="*55)
    print("  TriageNet-RL — LLM Inference Script")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  API    : {API_BASE_URL}")
    print("  Tasks  : 3 (Easy, Medium, Hard)")
    print("="*55)

    # Check API key
    if not HF_TOKEN:
        print("WARNING: No API key found.")
        print("Set HF_TOKEN environment variable.")
        print("Running baseline comparison only...")
        print_summary_table({})
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    results = {}
    for task_id in [1, 2, 3]:
        print(f"\n{'='*54}")
        difficulty = ["Easy", "Medium", "Hard"][task_id - 1]
        print(f"  TASK {task_id} — {difficulty}")
        print(f"{'='*54}")
        try:
            results[task_id] = run_llm_task(task_id, client, verbose=True)
        except Exception as e:
            print(f"  Task {task_id} failed: {e}")
            results[task_id] = {"final_score": 0.0, "passed": False, "error": str(e)}

    print_summary_table(results)

    # Save to JSON
    json_path = f"{RESULTS_DIR}/inference_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "results": {str(k): v for k, v in results.items()},
            "baseline_scores": BASELINE_SCORES,
        }, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == "__main__":
    main()
