from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from env import MedicalTriageEnv
from models import Action


app = FastAPI(title="Medical Triage OpenEnv Inference API", version="1.0.0")

# Global environment instance used by /reset and /step.
_env: Optional[MedicalTriageEnv] = None


def _to_jsonable(value: Any) -> Any:
    """Convert dataclass-heavy objects returned by env into JSON-safe structures."""
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


@app.post("/reset")
def reset(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize a new environment episode and return the initial observation."""
    global _env

    payload = payload or {}
    task_id = int(payload.get("task_id", 1))
    seed = payload.get("seed", 42)

    try:
        _env = MedicalTriageEnv(task_id=task_id, seed=seed)
        observation = _env.reset()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"reset failed: {exc}") from exc

    return {
        "observation": _to_jsonable(observation),
    }


@app.post("/step")
def step(action_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one action in the environment and return the resulting transition."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized. Call /reset first.")

    try:
        action = Action(**action_payload)
        result = _env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"step failed: {exc}") from exc

    return _to_jsonable(result)
