from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Task:
    """
    Defines one incident scenario.
    The environment uses this to know:
      - what to tell the agent at the start
      - what the correct root cause is
      - what the correct fix is
      - how many steps are allowed
    """
    name: str
    difficulty: str
    scenario_key: str
    description: str
    correct_root_cause_service: str
    correct_root_cause_reason: str
    correct_fix_service: str
    correct_fix_type: str
    correct_escalation_team: str
    max_steps: int
    summary_keywords: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)


TASK_EASY = Task(
    name="easy_crash",
    difficulty="easy",
    scenario_key="easy_crash",
    description="""
INCIDENT: P1 — Payment Service Down
Reported: 12 minutes ago
Impact: All customer checkouts are failing. Revenue loss ongoing.

You are the on-call engineer. Diagnose and resolve this incident.

Available services: user_service, order_service, payment_service,
                    notification_service, database

Available actions:
  - query_logs(target_service, log_filter)
  - check_metrics(target_service, metric_name)  [cpu/memory/latency/error_rate]
  - check_deps(target_service)
  - apply_fix(target_service, fix_type)          [restart/rollback/scale_up/clear_cache]
  - escalate(team)                               [deployment_team/database_team/network_team]
  - write_summary(summary_text)

Resolve the incident as fast as possible.
""".strip(),
    correct_root_cause_service="payment_service",
    correct_root_cause_reason="bad deployment v2.1.3 introduced NullPointerException",
    correct_fix_service="payment_service",
    correct_fix_type="rollback",
    correct_escalation_team="deployment_team",
    max_steps=15,
    summary_keywords=["payment_service", "v2.1.3", "rollback", "deployment"],
    hints=[
        "payment_service was last deployed 12 minutes ago",
        "NullPointerException appears in payment_service logs after v2.1.3",
        "Rolling back to v2.1.2 should restore service",
    ],
)


TASK_MEDIUM = Task(
    name="medium_cascade",
    difficulty="medium",
    scenario_key="medium_cascade",
    description="""
INCIDENT: P1 — Checkout Flow Degraded
Reported: 22 minutes ago
Impact: ~60% of checkouts failing. Customer complaints increasing.

You are the on-call engineer. Multiple services appear affected.
Find the ROOT CAUSE (not just symptoms) and resolve.

Available services: user_service, order_service, payment_service,
                    notification_service, database

Available actions:
  - query_logs(target_service, log_filter)
  - check_metrics(target_service, metric_name)  [cpu/memory/latency/error_rate]
  - check_deps(target_service)
  - apply_fix(target_service, fix_type)          [restart/rollback/scale_up/clear_cache]
  - escalate(team)                               [deployment_team/database_team/network_team]
  - write_summary(summary_text)

WARNING: Fixing symptoms without finding root cause will not resolve the incident.
""".strip(),
    correct_root_cause_service="payment_service",
    correct_root_cause_reason="connection leak in payment_service exhausting database connection pool",
    correct_fix_service="payment_service",
    correct_fix_type="restart",
    correct_escalation_team="database_team",
    max_steps=20,
    summary_keywords=[
        "payment_service", "connection", "database", "pool", "leak", "restart"
    ],
    hints=[
        "database connection pool is at 100%",
        "payment_service holds 48 out of 50 database connections",
        "restarting payment_service will release the leaked connections",
    ],
)


TASK_HARD = Task(
    name="hard_corruption",
    difficulty="hard",
    scenario_key="hard_corruption",
    description="""
INCIDENT: P2 — Incorrect Order Prices
Reported: 45 minutes ago (user tickets)
Impact: Unknown scope. No alerts fired. Users reporting wrong prices on orders.

You are the on-call engineer. There are NO obvious system alerts.
You must proactively investigate to find what is wrong.

Available services: user_service, order_service, payment_service,
                    notification_service, database

Available actions:
  - query_logs(target_service, log_filter)
  - check_metrics(target_service, metric_name)  [cpu/memory/latency/error_rate]
  - check_deps(target_service)
  - apply_fix(target_service, fix_type)          [restart/rollback/scale_up/clear_cache]
  - escalate(team)                               [deployment_team/database_team/data_team]
  - write_summary(summary_text)

HINT: The problem started approximately 2 hours ago.
      Focus on what changed around that time.
""".strip(),
    correct_root_cause_service="order_service",
    correct_root_cause_reason="discount_recalculator background job v1.2.1 has a bug halving all discounts",
    correct_fix_service="order_service",
    correct_fix_type="rollback",
    correct_escalation_team="data_team",
    max_steps=25,
    summary_keywords=[
        "order_service", "discount", "background job",
        "corruption", "rollback", "v1.2.1"
    ],
    hints=[
        "no system alerts fired — this is silent corruption",
        "order_service logs show discount_recalculator job ran 2 hours ago",
        "discount_recalculator v1.2.1 contains a bug that halves discount values",
        "rolling back order_service will stop further corruption",
    ],
)



ALL_TASKS = {
    "easy_crash":       TASK_EASY,
    "medium_cascade":   TASK_MEDIUM,
    "hard_corruption":  TASK_HARD,
}


def get_task(name: str) -> Task:
    if name not in ALL_TASKS:
        raise ValueError(
            f"Unknown task: {name}. Choose from: {list(ALL_TASKS.keys())}"
        )
    return ALL_TASKS[name]


def get_random_task() -> Task:
    import random
    return random.choice(list(ALL_TASKS.values()))