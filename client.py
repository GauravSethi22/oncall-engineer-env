from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, Optional

import requests
import websockets

from models import (
    OnCallAction, OnCallObservation, OnCallState,
    Alert, ServiceStatus
)


class StepResult:
    def __init__(
        self,
        observation: OnCallObservation,
        reward: float,
        done: bool,
        state: OnCallState,
    ):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.state = state


# Shared helper used by both OnCallEnv and OnCallEnvAsync.

def _parse_response(data: Dict[str, Any]) -> StepResult:
    obs_data   = data["observation"]
    state_data = data["state"]

    alerts = [Alert(**a) for a in obs_data["alerts"]]
    services = [ServiceStatus(**s) for s in obs_data["services"]]

    obs = OnCallObservation(
        alerts=alerts,
        services=services,
        last_action_result=obs_data["last_action_result"],
        last_action_error=obs_data.get("last_action_error"),
        elapsed_minutes=obs_data["elapsed_minutes"],
        steps_taken=obs_data["steps_taken"],
        task_description=obs_data["task_description"],
        task_name=obs_data["task_name"],
        unlocked_hints=obs_data.get("unlocked_hints", []),
        action_data=obs_data.get("action_data", {}),
    )

    state = OnCallState(
        episode_id=state_data["episode_id"],
        step_count=state_data["step_count"],
        task_name=state_data["task_name"],
        task_difficulty=state_data["task_difficulty"],
        incident_resolved=state_data["incident_resolved"],
        root_cause_identified=state_data["root_cause_identified"],
        correct_fix_applied=state_data["correct_fix_applied"],
        current_score=state_data["current_score"],
        max_steps=state_data["max_steps"],
        elapsed_minutes=state_data["elapsed_minutes"],
    )

    return StepResult(
        observation=obs,
        reward=data["reward"],
        done=data["done"],
        state=state,
    )


class OnCallEnv:
    """
    Synchronous HTTP client.
    Use this in inference.py and quick scripts.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self) -> StepResult:
        r = requests.post(f"{self.base_url}/reset")
        r.raise_for_status()
        return _parse_response(r.json())

    def reset_with_task(self, task_name: str) -> StepResult:
        """Reset and switch to a specific task."""
        r = requests.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name},
        )
        r.raise_for_status()
        return _parse_response(r.json())

    def step(self, action: OnCallAction) -> StepResult:
        payload = {
            "action_type":    action.action_type,
            "target_service": action.target_service,
            "log_filter":     action.log_filter,
            "metric_name":    action.metric_name,
            "fix_type":       action.fix_type,
            "team":           action.team,
            "summary_text":   action.summary_text,
            "metadata":       action.metadata,
        }
        r = requests.post(f"{self.base_url}/step", json=payload)
        r.raise_for_status()
        return _parse_response(r.json())

    def state(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OnCallEnvAsync:
    """
    Async WebSocket client.
    Use this for high-throughput training runs.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ws_url}/ws"
        self._ws = None

    async def connect(self):
        self._ws = await websockets.connect(self.ws_url)

    async def reset(self) -> StepResult:
        await self._ws.send(json.dumps({"type": "reset"}))
        data = json.loads(await self._ws.recv())
        return _parse_response(data)

    async def reset_with_task(self, task_name: str) -> StepResult:
        """Reset and switch to a specific task."""
        await self._ws.send(json.dumps({
            "type": "reset",
            "task_name": task_name,
        }))
        data = json.loads(await self._ws.recv())
        return _parse_response(data)

    async def step(self, action: OnCallAction) -> StepResult:
        payload = {
            "type": "step",
            "action": {
                "action_type":    action.action_type,
                "target_service": action.target_service,
                "log_filter":     action.log_filter,
                "metric_name":    action.metric_name,
                "fix_type":       action.fix_type,
                "team":           action.team,
                "summary_text":   action.summary_text,
                "metadata":       action.metadata or {},
            },
        }
        await self._ws.send(json.dumps(payload))
        data = json.loads(await self._ws.recv())
        return _parse_response(data)

    async def state(self) -> Dict[str, Any]:
        await self._ws.send(json.dumps({"type": "state"}))
        return json.loads(await self._ws.recv())

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()