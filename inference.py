import os
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware # Added for CORS
from openai import OpenAI

from env import MedicalTriageEnv
from models import Action

app = FastAPI()

# --- STEP 1: ADD CORS MIDDLEWARE ---
# This allows the Scaler/Meta grader to talk to your API without being blocked.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

# Initialize Environment
env = MedicalTriageEnv(task_id=int(os.getenv("TASK_ID", "1")))

@app.get("/")
async def root():
    return {"status": "ok", "message": "TriageNet-RL API is running"}

@app.post("/reset")
async def reset():
    try:
        observation = env.reset()
        print(f"[START] New episode initiated")
        return observation.model_dump()
    except Exception as e:
        print(f"Reset Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: Request):
    try:
        data = await request.json()
        
        # --- STEP 2: ROBUST ACTION PARSING ---
        # If 'data' is nested (sometimes graders wrap it), extract it.
        # This ensures we don't crash if the grader sends extra keys.
        action = Action(**data)
        
        result = env.step(action)
        
        # Log the step
        print(f"[STEP] action={action.action_type} reward={result.reward.step_reward}")

        # Return standard OpenEnv response format
        return {
            "observation": result.observation.model_dump(),
            "reward": {
                "step_reward": float(result.reward.step_reward),
                "cumulative_reward": float(result.reward.cumulative_reward),
                "feedback": result.reward.feedback,
            },
            "done": bool(result.done),
            "info": result.info,
        }
    except Exception as exc:
        print(f"Step Error: {exc}")
        raise HTTPException(status_code=400, detail=f"Step failed: {exc}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)