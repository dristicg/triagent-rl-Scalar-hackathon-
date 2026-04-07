import os

from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI

from env import MedicalTriageEnv
from models import Action


app = FastAPI()

# Required environment variables for OpenEnv / HF deployments.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# OpenAI client configured via the required environment variables.
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

env = MedicalTriageEnv(task_id=int(os.getenv("TASK_ID", "1")))


def _log_start(task_id: int) -> None:
    print(f"[START] task_id={task_id} model={MODEL_NAME}")


def _log_step(step_number: int, action: Action, done: bool) -> None:
    print(
        f"[STEP] step={step_number} action_type={action.action_type} "
        f"esi={action.esi_level or '-'} routing={action.routing_zone or '-'} done={done}"
    )


def _log_end(final_score: float, step_count: int, done: bool) -> None:
    print(f"[END] steps={step_count} final_score={final_score:.3f} done={done}")


@app.get("/")
async def root():
    return {"status": "ok", "message": "TriageNet-RL API is running"}


@app.post("/reset")
async def reset():
    observation = env.reset()
    # Grader expects a dictionary, not a Pydantic object
    _log_start(getattr(env, "task_id", 1))
    return observation.model_dump()


@app.post("/step")
async def step(request: Request):
    data = await request.json()
    try:
        action = Action(**data)
        result = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"step failed: {exc}") from exc

    _log_step(getattr(env, "_step", 0), action, result.done)

    if result.done:
        try:
            grade = env.grade()
            _log_end(grade.final_score, getattr(env, "_step", 0), result.done)
        except Exception:
            _log_end(result.reward.cumulative_reward, getattr(env, "_step", 0), result.done)

    return {
        "observation": result.observation.model_dump(),
        "reward": {
            "step_reward": result.reward.step_reward,
            "cumulative_reward": result.reward.cumulative_reward,
            "breakdown": result.reward.breakdown,
            "feedback": result.reward.feedback,
        },
        "done": result.done,
        "info": result.info,
    }
