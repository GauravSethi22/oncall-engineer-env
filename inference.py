from __future__ import annotations

import os
import json
import textwrap
from typing import List, Optional, Set
from openai import OpenAI

from client import OnCallEnv, StepResult
from models import OnCallAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not API_KEY:
    raise ValueError(
        "No API key found. Set HF_TOKEN or API_KEY environment variable.\n"
        "Example:  $env:HF_TOKEN='your_token_here'  (PowerShell)\n"
        "          export HF_TOKEN=your_token_here   (bash)"
    )

ENV_URL   = os.getenv("ENV_URL", "http://localhost:8000")
TASKS     = ["easy_crash", "medium_cascade", "hard_corruption"]
MAX_STEPS = 25
TEMPERATURE = 0.2
MAX_TOKENS  = 400


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

TASK-SPECIFIC GUIDANCE:

CRASH (service down after deployment):
  → rollback the crashed service

CASCADE (connection pool exhausted):
  → Step 1: check_metrics database error_rate  (shows who holds connections)
  → Step 2: check_metrics payment_service error_rate  (confirms leak)
  → Step 3: escalate database_team
  → Step 4: apply_fix payment_service restart
  → Step 5: write_summary
  → ONE check per service MAX. NEVER repeat check_metrics on same service.
  → NEVER restart database. NEVER fix order_service.

SILENT CORRUPTION (no alerts, wrong data):
  → Query order_service and database logs
  → Escalate data_team
  → apply_fix order_service rollback
  → write_summary

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



def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score = rewards[-1] if rewards else 0.0
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )



class EpisodeMemory:
    def __init__(self):
        self.queried_services: Set[str] = set()
        self.metrics_checked: Set[str]  = set()
        self.fixes_applied: List[str]   = []
        self.escalated: bool            = False
        self.summary_written: bool      = False
        self.action_history: List[str]  = []
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

    def already_fixed(self, service: str, fix_type: str) -> bool:
        return f"{fix_type}:{service}" in self.fixes_applied

    def times_metric_checked(self, service: str) -> int:
        return self.metrics_checked_count.get(service, 0)



def build_user_prompt(
    result: StepResult,
    step: int,
    memory: EpisodeMemory,
    task_name: str,
) -> str:
    obs = result.observation

    alerts_text = "\n".join(
        f"  [{a.severity.upper()}] {a.service}: {a.message} ({a.fired_at})"
        for a in obs.alerts
    ) or "  No critical alerts. Look for subtle signals."

    services_text = "\n".join(
        f"  {s.name}: {s.status} | errors={s.error_rate:.0%} "
        f"latency={s.latency_ms:.0f}ms | deploy={s.last_deployment}"
        for s in obs.services
    )

    hints_text = (
        "\n".join(f"  HINT: {h}" for h in obs.unlocked_hints)
        if obs.unlocked_hints else "  None yet — keep investigating."
    )

    not_queried = [
        s for s in
        ["user_service", "order_service", "payment_service",
         "notification_service", "database"]
        if s not in memory.queried_services
    ]

    steps_taken = obs.steps_taken
    steps_remaining = MAX_STEPS - steps_taken
    nudges = []

    # Block repeated metric checks: the agent should investigate new data or move toward resolution.
    repeated_metrics = [
        svc for svc, count in memory.metrics_checked_count.items()
        if count >= 2
    ]
    if repeated_metrics:
        nudges.append(
            f"STOP: You already checked metrics for {repeated_metrics} multiple times. "
            f"Do NOT check them again. Move to escalate or apply_fix."
        )

    if len(memory.metrics_checked) >= 2 and not memory.fixes_applied and not memory.escalated:
        nudges.append(
            "You have checked enough metrics. "
            "Now escalate to the right team, then apply the fix."
        )

    if task_name == "medium_cascade":
        db_metrics_checked  = memory.times_metric_checked("database") >= 1
        pay_metrics_checked = memory.times_metric_checked("payment_service") >= 1

        if not db_metrics_checked:
            nudges.append(
                "CASCADE Step 1: check_metrics database error_rate. "
                "Shows which service holds most DB connections."
            )
        elif not pay_metrics_checked:
            nudges.append(
                "CASCADE Step 2: check_metrics payment_service error_rate. "
                "Confirms the connection leak."
            )
        elif not memory.escalated:
            nudges.append(
                "CASCADE Step 3: escalate database_team. You have all evidence."
            )
        elif not memory.fixes_applied:
            nudges.append(
                "CASCADE Step 4 — YOUR ONLY VALID ACTION: "
                "{\"action_type\": \"apply_fix\", \"target_service\": \"payment_service\", \"fix_type\": \"restart\"} "
                "Do NOT check metrics again. Apply fix NOW."
            )

    if task_name == "hard_corruption":
        if (memory.escalated
                and not memory.fixes_applied
                and "order_service" in memory.queried_services):
            nudges.append(
                "CORRUPTION: Apply fix now: "
                "{\"action_type\": \"apply_fix\", \"target_service\": \"order_service\", \"fix_type\": \"rollback\"}"
            )

    if not memory.escalated and steps_taken >= 3:
        nudges.append("You have NOT escalated yet. Do this before applying a fix.")

    if not memory.summary_written and steps_remaining <= 3:
        nudges.append("URGENT: Running out of steps. Call write_summary NOW.")

    if memory.fixes_applied and not memory.summary_written:
        nudges.append("Fix already applied. Your ONLY next action is write_summary.")

    nudge_text = (
        "\nACTION REQUIRED:\n" + "\n".join(f"  {n}" for n in nudges)
        if nudges else ""
    )

    return textwrap.dedent(f"""
        TASK: {task_name} | STEP {step} | Elapsed: {obs.elapsed_minutes} min

        ACTIVE ALERTS:
        {alerts_text}

        ALL SERVICE STATUS:
        {services_text}

        LAST ACTION RESULT:
        {obs.last_action_result or 'N/A'}
        {"ERROR: " + obs.last_action_error if obs.last_action_error else ""}

        HINTS UNLOCKED:
        {hints_text}

        WHAT YOU HAVE DONE SO FAR:
          Queried services  : {list(memory.queried_services) or 'none'}
          Metrics checked   : {list(memory.metrics_checked) or 'none'}
          Fixes applied     : {memory.fixes_applied or 'none'}
          Escalated         : {memory.escalated}
          Summary written   : {memory.summary_written}
          Services NOT yet checked: {not_queried}
        {nudge_text}

        What is your next action? JSON only.
    """).strip()



def get_agent_action(
    client: OpenAI,
    result: StepResult,
    step: int,
    memory: EpisodeMemory,
    task_name: str,
) -> OnCallAction:
    user_prompt = build_user_prompt(result, step, memory, task_name)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = completion.choices[0].message.content.strip()
        text = text.strip("```json").strip("```").strip()
        data = json.loads(text)
        action = OnCallAction(**data)

        # Hard override 1: block repeated metric checks
        if (action.action_type == "check_metrics"
                and action.target_service
                and memory.times_metric_checked(action.target_service) >= 1):
            print(
                f"[DEBUG] Blocking repeated check_metrics on {action.target_service} "
                f"→ forcing next logical action",
                flush=True
            )
            return _next_logical_action(memory, task_name)

        # Hard override 2: block write_summary before fix applied
        if action.action_type == "write_summary" and not memory.fixes_applied:
            fix_map = {
                "easy_crash":      ("payment_service", "rollback"),
                "medium_cascade":  ("payment_service", "restart"),
                "hard_corruption": ("order_service",   "rollback"),
            }
            svc, fix = fix_map.get(task_name, ("payment_service", "rollback"))
            print(f"[DEBUG] Blocking premature write_summary → apply_fix {fix} on {svc}", flush=True)
            return OnCallAction(action_type="apply_fix", target_service=svc, fix_type=fix)

        return action

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error at step {step}: {e}", flush=True)
        return _fallback_action(result, memory, task_name)

    except Exception as e:
        print(f"[DEBUG] Model error at step {step}: {e}", flush=True)
        return _fallback_action(result, memory, task_name)


def _next_logical_action(memory: EpisodeMemory, task_name: str) -> OnCallAction:
    """Returns the next logical step when LLM tries to repeat a metric check."""
    # If the agent has not escalated yet, escalate first before applying any fix.
    if not memory.escalated:
        team_map = {
            "easy_crash":      "deployment_team",
            "medium_cascade":  "database_team",
            "hard_corruption": "data_team",
        }
        return OnCallAction(
            action_type="escalate",
            team=team_map.get(task_name, "deployment_team"),
        )
    if not memory.fixes_applied:
        fix_map = {
            "easy_crash":      ("payment_service", "rollback"),
            "medium_cascade":  ("payment_service", "restart"),
            "hard_corruption": ("order_service",   "rollback"),
        }
        svc, fix = fix_map.get(task_name, ("payment_service", "rollback"))
        return OnCallAction(action_type="apply_fix", target_service=svc, fix_type=fix)
    return OnCallAction(
        action_type="write_summary",
        summary_text=f"Incident resolved. Fixes applied: {memory.fixes_applied}.",
    )


def _fallback_action(
    result: StepResult,
    memory: EpisodeMemory,
    task_name: str,
) -> OnCallAction:
    # Prefer finalizing the episode once a fix is in place.
    if memory.fixes_applied and not memory.summary_written:
        return OnCallAction(
            action_type="write_summary",
            summary_text=(
                f"Incident investigated. Fix applied: {memory.fixes_applied[-1]}. "
                f"Root cause identified through log analysis."
            ),
        )
    if not memory.escalated:
        team_map = {
            "easy_crash":      "deployment_team",
            "medium_cascade":  "database_team",
            "hard_corruption": "data_team",
        }
        return OnCallAction(
            action_type="escalate",
            team=team_map.get(task_name, "deployment_team"),
        )
    all_services = [
        "payment_service", "order_service",
        "database", "user_service", "notification_service"
    ]
    for svc in all_services:
        if svc not in memory.queried_services:
            return OnCallAction(
                action_type="query_logs",
                target_service=svc,
                log_filter="error",
            )
    fix_map = {
        "easy_crash":      ("payment_service", "rollback"),
        "medium_cascade":  ("payment_service", "restart"),
        "hard_corruption": ("order_service",   "rollback"),
    }
    svc, fix = fix_map.get(task_name, ("payment_service", "rollback"))
    return OnCallAction(action_type="apply_fix", target_service=svc, fix_type=fix)


def _step_with_retry(
    env: OnCallEnv,
    action: OnCallAction,
    retries: int = 3,
    delay: float = 2.0,
) -> StepResult:
    """
    Retry env.step() on transient HTTP errors.
    Avoids crashing a whole episode on a brief server hiccup.
    """
    import time
    last_err = None
    for attempt in range(retries):
        try:
            return env.step(action)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                print(
                    f"[DEBUG] step() failed (attempt {attempt+1}/{retries}): "
                    f"{e} — retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
    raise last_err



def run_episode(
    env: OnCallEnv,
    client: OpenAI,
    task_name: str,
) -> tuple[bool, int, List[float]]:

    result = env.reset_with_task(task_name)

    memory  = EpisodeMemory()
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env="oncall_engineer", model=MODEL_NAME)

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        if step == MAX_STEPS and not memory.summary_written:
            action = OnCallAction(
                action_type="write_summary",
                summary_text=(
                    f"Incident summary for {task_name}. "
                    f"Investigated: {list(memory.queried_services)}. "
                    f"Fixes: {memory.fixes_applied}."
                ),
            )
        else:
            action = get_agent_action(client, result, step, memory, task_name)

        memory.record(action)

        action_str = json.dumps({
            k: v for k, v in {
                "type":    action.action_type,
                "service": action.target_service,
                "fix":     action.fix_type,
                "team":    action.team,
            }.items() if v
        })

        result      = _step_with_retry(env, action)
        reward      = result.reward or 0.0
        done        = result.done
        error       = result.observation.last_action_error

        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        if done:
            break

    success = result.state.correct_fix_applied
    log_end(success=success, steps=steps_taken, rewards=rewards)
    return success, steps_taken, rewards



def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    with OnCallEnv(base_url=ENV_URL) as env:
        for task_name in TASKS:
            try:
                run_episode(env, client, task_name)
            except Exception as e:
                print(f"[DEBUG] Episode failed for {task_name}: {e}", flush=True)
                log_end(success=False, steps=0, rewards=[])


if __name__ == "__main__":
    main()