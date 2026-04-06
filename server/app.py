from __future__ import annotations
import json
import os
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import OnCallAction
from .environment import OnCallEnvironment

app = FastAPI(
    title="OnCall Engineer Environment",
    description="RL environment: AI on-call engineer fixing production incidents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_NAME = os.getenv("ONCALL_TASK", "easy_crash")



class StepRequest(BaseModel):
    action_type: str
    target_service: str | None = None
    log_filter: str | None = None
    metric_name: str | None = None
    fix_type: str | None = None
    team: str | None = None
    summary_text: str | None = None
    metadata: Dict[str, Any] = {}


def _obs_to_dict(obs, reward, done, state) -> Dict[str, Any]:
    return {
        "observation": {
            "alerts": [
                {
                    "severity": a.severity,
                    "service": a.service,
                    "message": a.message,
                    "fired_at": a.fired_at,
                }
                for a in obs.alerts
            ],
            "services": [
                {
                    "name": s.name,
                    "status": s.status,
                    "error_rate": s.error_rate,
                    "latency_ms": s.latency_ms,
                    "cpu_percent": s.cpu_percent,
                    "memory_percent": s.memory_percent,
                    "last_deployment": s.last_deployment,
                    "recent_log_summary": s.recent_log_summary,
                }
                for s in obs.services
            ],
            "last_action_result": obs.last_action_result,
            "last_action_error": obs.last_action_error,
            "elapsed_minutes": obs.elapsed_minutes,
            "steps_taken": obs.steps_taken,
            "task_description": obs.task_description,
            "task_name": obs.task_name,
            "unlocked_hints": obs.unlocked_hints,
        },
        "reward": reward,
        "done": done,
        "state": {
            "episode_id": state.episode_id,
            "step_count": state.step_count,
            "task_name": state.task_name,
            "task_difficulty": state.task_difficulty,
            "incident_resolved": state.incident_resolved,
            "root_cause_identified": state.root_cause_identified,
            "correct_fix_applied": state.correct_fix_applied,
            "current_score": state.current_score,
            "max_steps": state.max_steps,
            "elapsed_minutes": state.elapsed_minutes,
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy", "task": TASK_NAME}


class ResetRequest(BaseModel):
    task_name: str | None = None


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    task = req.task_name or TASK_NAME
    env = OnCallEnvironment(task_name=task)
    app.state.http_env = env
    obs, reward, done, state = env.reset()
    return _obs_to_dict(obs, reward, done, state)


@app.post("/step")
def step(req: StepRequest):
    env: OnCallEnvironment = getattr(app.state, "http_env", None)
    if env is None:
        return {"error": "Call /reset first"}
    action = OnCallAction(
        action_type=req.action_type,
        target_service=req.target_service,
        log_filter=req.log_filter,
        metric_name=req.metric_name,
        fix_type=req.fix_type,
        team=req.team,
        summary_text=req.summary_text,
        metadata=req.metadata,
    )
    obs, reward, done, state = env.step(action)
    return _obs_to_dict(obs, reward, done, state)


@app.get("/state")
def get_state():
    env: OnCallEnvironment = getattr(app.state, "http_env", None)
    if env is None:
        return {"error": "Call /reset first"}
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "task_name": s.task_name,
        "task_difficulty": s.task_difficulty,
        "current_score": s.current_score,
        "max_steps": s.max_steps,
    }



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket session.
    Each connection gets its own isolated environment instance.
    
    Message format (send):
        {"type": "reset"}
        {"type": "step", "action": { ...OnCallAction fields... }}
        {"type": "state"}
    
    Message format (receive):
        Same as HTTP responses above.
    """
    await websocket.accept()
    env = OnCallEnvironment(task_name=TASK_NAME)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "reset":
                task = data.get("task_name") or TASK_NAME
                env = OnCallEnvironment(task_name=task)
                obs, reward, done, state = env.reset()
                await websocket.send_json(_obs_to_dict(obs, reward, done, state))

            elif msg_type == "step":
                action_data = data.get("action", {})
                action = OnCallAction(**action_data)
                obs, reward, done, state = env.step(action)
                await websocket.send_json(_obs_to_dict(obs, reward, done, state))

            elif msg_type == "state":
                s = env.state
                await websocket.send_json({
                    "episode_id": s.episode_id,
                    "step_count": s.step_count,
                    "task_name": s.task_name,
                    "current_score": s.current_score,
                })

            else:
                await websocket.send_json({"error": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()