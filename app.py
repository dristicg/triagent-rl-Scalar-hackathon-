import os
import sys
import json
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import MedicalTriageEnv
from models import Action

app = FastAPI(title="TriageNet-RL OpenEnv API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instances — one per task
environments = {
    1: MedicalTriageEnv(task_id=1, seed=42),
    2: MedicalTriageEnv(task_id=2, seed=42),
    3: MedicalTriageEnv(task_id=3, seed=42),
}
current_task_id = 1

class ResetRequest(BaseModel):
    task_id: Optional[int] = 1
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    action_type: str
    patient_id: str
    esi_level: Optional[int] = None
    routing_zone: Optional[str] = None
    resources: Optional[List[str]] = None
    notes: Optional[str] = None
    escalate_reason: Optional[str] = None

@app.get("/")
async def root():
    return {
        "status": "ok",
        "name": "TriageNet-RL",
        "version": "1.0.0",
        "description": "Medical ED Triage OpenEnv",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "tasks": [1, 2, 3]}

@app.post("/reset")
async def reset(request: ResetRequest = None):
    try:
        task_id = 1
        seed = 42
        if request:
            task_id = request.task_id or 1
            seed = request.seed or 42

        env = MedicalTriageEnv(task_id=task_id, seed=seed)
        environments[task_id] = env
        obs = env.reset()

        print(f"[START] task_id={task_id} seed={seed}")

        return {
            "observation": obs.model_dump(),
            "reward": None,
            "done": False,
            "info": {"task_id": task_id, "message": "Environment reset successfully"}
        }
    except Exception as e:
        print(f"Reset error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "observation": None,
            "reward": None,
            "done": False,
            "info": {"error": str(e)}
        }

@app.post("/step")
async def step(request: StepRequest):
    try:
        task_id = 1
        env = environments.get(task_id)
        if env is None:
            env = MedicalTriageEnv(task_id=task_id, seed=42)
            environments[task_id] = env

        action = Action(
            action_type=request.action_type,
            patient_id=request.patient_id,
            esi_level=request.esi_level,
            routing_zone=request.routing_zone,
            resources=request.resources,
            notes=request.notes,
            escalate_reason=request.escalate_reason,
        )

        result = env.step(action)

        print(f"[STEP] action={action.action_type} "
              f"patient={action.patient_id} "
              f"esi={action.esi_level} "
              f"reward={result.reward.step_reward:+.3f}")

        return {
            "observation": result.observation.model_dump(),
            "reward": {
                "step_reward": float(result.reward.step_reward),
                "cumulative_reward": float(result.reward.cumulative_reward),
                "breakdown": result.reward.breakdown,
                "feedback": result.reward.feedback,
            },
            "done": bool(result.done),
            "info": result.info,
        }
    except Exception as e:
        print(f"Step error: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/state")
async def state(task_id: int = 1):
    env = environments.get(task_id)
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    return env.state()

@app.post("/grade")
async def grade(task_id: int = 1):
    env = environments.get(task_id)
    if env is None:
        return {"error": "No active episode"}
    result = env.grade()
    print(f"[END] task_id={task_id} "
          f"score={result.final_score:.3f} "
          f"passed={result.passed}")
    return {
        "task_id": result.task_id,
        "final_score": result.final_score,
        "total_reward": result.total_reward,
        "steps_taken": result.steps_taken,
        "correct_esi_count": result.correct_esi_count,
        "total_patients": result.total_patients,
        "critical_misses": result.critical_misses,
        "deteriorations": result.deteriorations,
        "grade_breakdown": result.grade_breakdown,
        "passed": result.passed,
    }

def create_gradio_app():
    with gr.Blocks(title="TriageNet-RL") as demo:
        gr.Markdown("# 🏥 TriageNet-RL — Medical Triage OpenEnv")
        gr.Markdown(
            "Real-world emergency department triage simulation. "
            "An AI agent acts as a triage nurse assigning ESI levels."
        )

        with gr.Tab("API Info"):
            gr.Markdown("""
            ## Available Endpoints

            | Endpoint | Method | Description |
            |----------|--------|-------------|
            | `/` | GET | API info |
            | `/health` | GET | Health check |
            | `/reset` | POST | Reset environment |
            | `/step` | POST | Take an action |
            | `/state` | GET | Get current state |
            | `/grade` | POST | Get final score |

            ## Quick Test
            POST `/reset` with `{"task_id": 1}` to start.
            POST `/step` with an action to triage a patient.

            ## Tasks
            - Task 1 (Easy): 1 patient, 5 steps
            - Task 2 (Medium): 8 patients, 20 steps
            - Task 3 (Hard): 15 patients, 40 steps
            """)

        with gr.Tab("Live Demo"):
            task_dropdown = gr.Dropdown(
                choices=[1, 2, 3], value=1, label="Task ID"
            )
            reset_btn = gr.Button("Reset Environment", variant="primary")
            output_box = gr.JSON(label="Response")

            def do_reset(task_id):
                env = MedicalTriageEnv(task_id=int(task_id), seed=42)
                environments[int(task_id)] = env
                obs = env.reset()
                return {
                    "observation": obs.model_dump(),
                    "reward": None,
                    "done": False,
                    "info": {"task_id": int(task_id)}
                }

            reset_btn.click(do_reset, inputs=[task_dropdown], outputs=[output_box])

    return demo

# Mount Gradio on FastAPI
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting TriageNet-RL on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
