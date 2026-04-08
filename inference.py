import os
import sys
import json
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import MedicalTriageEnv
from models import Action, VALID_ROUTING_ZONES

# ── Environment variables (EXACT format required by hackathon) ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SYSTEM_PROMPT = """You are an expert emergency department triage nurse.
Given patient information, respond with ONLY valid JSON:
{
  "action_type": "assign_esi",
  "esi_level": <1-5>,
  "routing_zone": "<resuscitation_bay|acute_care|fast_track|waiting_room>",
  "notes": "<brief clinical reasoning>"
}
ESI 1=immediate, 2=emergent, 3=urgent, 4=less urgent, 5=non-urgent."""

def build_patient_prompt(obs_dict):
    patient = obs_dict.get("current_patient")
    if not patient:
        return "No patient available."
    vitals = patient.get("vitals", {})
    return f"""Patient: {patient.get('name')} | Age: {patient.get('age')} {patient.get('gender')}
Complaint: {patient.get('chief_complaint')}
BP: {vitals.get('bp_systolic')}/{vitals.get('bp_diastolic')} | HR: {vitals.get('heart_rate')} | SpO2: {vitals.get('spo2')}% | GCS: {vitals.get('gcs')} | Pain: {vitals.get('pain_scale')}/10
History: {', '.join(patient.get('medical_history', [])) or 'None'}
Provide triage decision as JSON."""

def parse_llm_response(response_text, patient_id):
    try:
        start = response_text.find('{')
        end   = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        parsed       = json.loads(response_text[start:end])
        esi_level    = int(parsed.get("esi_level", 3))
        routing_zone = str(parsed.get("routing_zone", "fast_track"))
        if not (1 <= esi_level <= 5):
            esi_level = 3
        if routing_zone not in VALID_ROUTING_ZONES:
            routing_zone = "fast_track"
        return Action(
            action_type="assign_esi",
            patient_id=patient_id,
            esi_level=esi_level,
            routing_zone=routing_zone,
            notes=str(parsed.get("notes", "")),
        )
    except Exception as e:
        print(f"Parse error: {e} — using fallback")
        return Action(action_type="wait", patient_id=patient_id)

def run_task(task_id, client, verbose=True):
    env  = MedicalTriageEnv(task_id=task_id, seed=42)
    obs  = env.reset()
    step = 0
    total_reward = 0.0
    difficulty = ["", "Easy", "Medium", "Hard"][task_id]

    print(f"[START] task_id={task_id} difficulty={difficulty} max_steps=30")

    while not env._done and step < 30:
        obs_dict = obs.model_dump()
        pending  = obs_dict.get("pending_patients", [])
        if not pending:
            break

        patient_id = pending[0]
        step += 1

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_patient_prompt(obs_dict)},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"API error step {step}: {e}")
            response_text = ""

        action = parse_llm_response(response_text, patient_id)
        result = env.step(action)
        obs    = result.observation
        total_reward += result.reward.step_reward

        print(f"[STEP] step={step} patient_id={patient_id} "
              f"action={action.action_type} esi={action.esi_level or '-'} "
              f"routing={action.routing_zone or '-'} "
              f"reward={result.reward.step_reward:+.3f}")

    grade = env.grade()
    print(f"[END] task_id={task_id} score={grade.final_score:.3f} "
          f"passed={grade.passed} steps={step} "
          f"correct_esi={grade.correct_esi_count}/{grade.total_patients}")

    return {
        "task_id": task_id,
        "final_score": grade.final_score,
        "passed": grade.passed,
        "steps_taken": step,
        "correct_esi": grade.correct_esi_count,
        "total_patients": grade.total_patients,
        "critical_misses": grade.critical_misses,
    }

def main():
    print("=" * 54)
    print("  TriageNet-RL — LLM Inference")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  API    : {API_BASE_URL}")
    print("=" * 54)

    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Set it to run LLM inference.")
        return

    client  = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results = {}

    for task_id in [1, 2, 3]:
        print(f"\n{'='*54}")
        print(f"  TASK {task_id}")
        print(f"{'='*54}")
        try:
            results[task_id] = run_task(task_id, client)
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            results[task_id] = {"final_score": 0.0, "passed": False}

    print("\n" + "="*54)
    print("  RESULTS SUMMARY")
    print("="*54)
    for tid, r in results.items():
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"  Task {tid}: {r.get('final_score', 0):.3f} [{status}]")

if __name__ == "__main__":
    main()