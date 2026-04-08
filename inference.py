from __future__ import annotations

import os
import json
import textwrap
import time
from typing import List, Optional, Set
import sys

from openai import OpenAI
from client import OnCallEnv, StepResult
from models import OnCallAction

# Fetch Mandatory Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") # Prioritize HF_TOKEN
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000") 

TASKS        = ["easy_crash", "medium_cascade", "hard_corruption"]
MAX_STEPS    = 25
TEMPERATURE  = 0.2
MAX_TOKENS   = 400

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior on-call software engineer responding to production incidents.
Respond with ONE JSON object only. No markdown. Example:
{"action_type": "query_logs", "target_service": "payment_service", "log_filter": "error"}
""").strip()

# ─── MANDATORY LOGGING FORMAT ──────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # ADDED: Mandatory 'score' field to match requirements
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ─── EPISODE RUNNER ───────────────────────────────────────────────────────────

class EpisodeMemory:
    def __init__(self):
        self.queried_services = set()
        self.metrics_checked = set()
        self.fixes_applied = []
        self.escalated = False
        self.summary_written = False
        self.metrics_checked_count = {}

    def record(self, action: OnCallAction):
        if action.action_type == "query_logs": self.queried_services.add(action.target_service)
        if action.action_type == "apply_fix": self.fixes_applied.append(f"{action.fix_type}:{action.target_service}")
        if action.action_type == "escalate": self.escalated = True
        if action.action_type == "write_summary": self.summary_written = True

def run_episode(env: OnCallEnv, client: OpenAI, task_name: str):
    memory = EpisodeMemory()
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env="oncall_engineer", model=MODEL_NAME)

    try:
        result = env.reset_with_task(task_name)
        for step in range(1, MAX_STEPS + 1):
            if result.done: break
            
            # Logic to get action from LLM (using your existing prompt builder/parser)
            action = get_agent_action(client, result, step, memory, task_name)
            memory.record(action)

            # Format action for STEP log
            action_json = json.dumps({"type": action.action_type, "svc": action.target_service})
            
            result = env.step(action)
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_json, reward=reward, done=result.done, error=result.observation.last_action_error)
            if result.done: break

        success = getattr(result.state, "correct_fix_applied", False)
        # Calculate a simple normalized score [0, 1]
        score = sum(rewards) / len(rewards) if rewards else 0.0
        
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    with OnCallEnv(base_url=ENV_URL) as env:
        for task_name in TASKS:
            run_episode(env, client, task_name)

if __name__ == "__main__":
    main()