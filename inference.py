import os

from fastapi import FastAPI, HTTPException, Request

from env import MedicalTriageEnv
from models import Action


app = FastAPI()
env = MedicalTriageEnv(task_id=int(os.getenv("TASK_ID", "1")))


@app.get("/")
async def root():
    return {"status": "ok", "message": "TriageNet-RL API is running"}


@app.post("/reset")
async def reset():
    observation = env.reset()
    # Grader expects a dictionary, not a Pydantic object
    return observation.model_dump()


@app.post("/step")
async def step(request: Request):
    data = await request.json()
    try:
        action = Action(**data)
        result = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"step failed: {exc}") from exc

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
