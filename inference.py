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

# 1. FIXED: Removed hardcoded API defaults and runtime dependencies
# Fetches mandatory environment variables as required by the platform
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")

# 2. FIXED: Localhost Default updated to your direct Space URL
ENV_URL      = os.environ.get("ENV_URL", "https://gaurav206-oncall-engineer-env.hf.space")

TASKS        = ["easy_crash", "medium_cascade", "hard_corruption"]
MAX_STEPS    = 25
TEMPERATURE  = 0.2
MAX_TOKENS   = 400

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior on-call software engineer responding to production incidents.

CRITICAL RULES — follow these exactly:
1. You MUST explore multiple services, not just one.
2. You MUST call write_summary at the end to close the incident. (-20% if missing)
3. You MUST call escalate once before applying a fix. (-10% if missing)
4. NEVER check the same metric or query the same service more than once.
5. NEVER repeat check_metrics on the same service. Move forward after 1 check.
6. Apply a fix only ONCE. Never repeat.

INVESTIGATION STRATEGY:
Step 1-2: Query logs of most affected services
Step 3-4: Check metrics of suspicious services (ONE check per service, never repeat)
Step 5:   Escalate to the correct team
Step 6:   Apply the correct fix
Step 7:   Write summary

AVAILABLE ACTIONS:
  query_logs       → target_service, log_filter (error/timeout/deploy)
  check_metrics    → target_service, metric_name (cpu/memory/latency/error_rate)
  check_deps       → target_service
  apply_fix        → target_service, fix_type (restart/rollback/scale_up/clear_cache)
  escalate         → team (deployment_team/database_team/network_team/data_team)
  write_summary    → summary_text

Respond with ONE JSON object only. No markdown. Example:
{"action_type": "query_logs", "target_service": "payment_service", "log_filter": "error"}
""").strip()


# ─── LOGGING ──────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

# 3. FIXED: Added the mandatory 'score' field to the [END] log
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ─── EPISODE MEMORY ───────────────────────────────────────────────────────────

class EpisodeMemory:
    def __init__(self):
        self.queried_services: Set[str]  = set()
        self.metrics_checked: Set[str]   = set()
        self.fixes_applied: List[str]    = []
        self.escalated: bool             = False
        self.summary_written: bool       = False
        self.action_history: List[str]   = []
        self.metrics_checked_count: dict = {}

    def record(self, action: OnCallAction):
        if action.action_type == "query_logs" and action.target_service:
            self.queried_services.add(action.target_service)
        if action.action_type == "check_metrics" and action.target_service:
            key = f"{action.target_service}.{action.metric_name}"
            self.metrics_checked.add(key)
            svc = action.target_service
            self.metrics_checked_count[svc] = self.metrics_checked_count.get(svc, 0) + 1
        if action.action_type == "apply_fix":
            self.fixes_applied.append(f"{action.fix_type}:{action.target_service}")
        if action.action_type == "escalate":
            self.escalated = True
        if action.action_type == "write_summary":
            self.summary_written = True
        self.action_history.append(action.action_type)

    def times_metric_checked(self, service: str) -> int:
        return self.metrics_checked_count.get(service, 0)


# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────

def build_user_prompt(result: StepResult, step: int, memory: EpisodeMemory, task_name: str) -> str:
    obs = result.observation
    alerts_text = "\n".join(f"  [{a.severity.upper()}] {a.service}: {a.message} ({a.fired_at})" for a in obs.alerts) or "  No alerts."
    services_text = "\n".join(f"  {s.name}: {s.status} | errors={s.error_rate:.0%} latency={s.latency_ms:.0f}ms" for s in obs.services)
    
    return textwrap.dedent(f"""
        TASK: {task_name} | STEP {step}
        ACTIVE ALERTS:
        {alerts_text}
        SERVICE STATUS:
        {services_text}
        LAST RESULT: {obs.last_action_result or 'N/A'}
        ERROR: {obs.last_action_error or 'None'}
        What is your next action? JSON only.
    """).strip()

# ─── ACTION PARSER & HELPERS ──────────────────────────────────────────────────

def get_agent_action(client: OpenAI, result: StepResult, step: int, memory: EpisodeMemory, task_name: str) -> OnCallAction:
    user_prompt = build_user_prompt(result, step, memory, task_name)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = completion.choices[0].message.content.strip().strip("```json").strip("```").strip()
        data = json.loads(text)
        action = OnCallAction(**data)
        
        # Guardrails: block repeated metrics
        if action.action_type == "check_metrics" and memory.times_metric_checked(action.target_service) >= 1:
            return _next_logical_action(memory, task_name)
        return action
    except Exception:
        return _fallback_action(result, memory, task_name)

def _next_logical_action(memory: EpisodeMemory, task_name: str) -> OnCallAction:
    if not memory.escalated:
        return OnCallAction(action_type="escalate", team="deployment_team")
    return OnCallAction(action_type="write_summary", summary_text="Resolved.")

def _fallback_action(result: StepResult, memory: EpisodeMemory, task_name: str) -> OnCallAction:
    return OnCallAction(action_type="query_logs", target_service="payment_service", log_filter="error")

def _step_with_retry(env: OnCallEnv, action: OnCallAction) -> StepResult:
    for _ in range(3):
        try: return env.step(action)
        except Exception: time.sleep(2)
    return env.step(action)

# ─── EPISODE RUNNER ───────────────────────────────────────────────────────────

def run_episode(env: OnCallEnv, client: OpenAI, task_name: str):
    memory = EpisodeMemory()
    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        result = env.reset_with_task(task_name)
        for step in range(1, MAX_STEPS + 1):
            if result.done: break
            action = get_agent_action(client, result, step, memory, task_name)
            memory.record(action)
            
            action_str = json.dumps({"type": action.action_type, "svc": action.target_service})
            result = _step_with_retry(env, action)
            
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=result.done, error=result.observation.last_action_error)
            if result.done: break

        success = getattr(result.state, "correct_fix_applied", False)
    finally:
        # Score calculation: Each task returns score in [0, 1]
        score = 1.0 if success else min(sum(rewards), 1.0)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY required.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in TASKS:
        log_start(task=task_name, env="oncall_engineer", model=MODEL_NAME)

    try:
        with OnCallEnv(base_url=ENV_URL) as env:
            for task_name in TASKS:
                try: run_episode(env, client, task_name)
                except Exception: log_end(success=False, steps=0, score=0.0, rewards=[])
    except Exception:
        for task_name in TASKS: log_end(success=False, steps=0, score=0.0, rewards=[])

if __name__ == "__main__":
    main()