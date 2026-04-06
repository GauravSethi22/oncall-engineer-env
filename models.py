from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OnCallAction:
    """
    One action the agent takes per step.
    
    action_type must be one of:
        query_logs       - read logs from a service
        check_metrics    - check CPU/memory/latency/error_rate of a service
        check_deps       - see what services this one depends on
        apply_fix        - attempt a fix (restart/rollback/scale)
        escalate         - page a specialist team
        write_summary    - declare root cause and close the incident
    """
    action_type: str
    target_service: Optional[str] = None
    log_filter: Optional[str] = None
    metric_name: Optional[str] = None
    fix_type: Optional[str] = None
    team: Optional[str] = None
    summary_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceStatus:
    """Health snapshot of one microservice."""
    name: str
    status: str
    error_rate: float
    latency_ms: float
    cpu_percent: float
    memory_percent: float
    last_deployment: str
    recent_log_summary: str


@dataclass
class Alert:
    """A firing alert in the system."""
    severity: str
    service: str
    message: str
    fired_at: str


@dataclass
class OnCallObservation:
    """
    Everything the agent can see at this step.
    Returned by both reset() and step().
    """
    alerts: List[Alert]
    services: List[ServiceStatus]
    last_action_result: str
    last_action_error: Optional[str]
    elapsed_minutes: int
    steps_taken: int
    task_description: str
    task_name: str
    unlocked_hints: List[str]
    action_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnCallState:
    """Episode-level metadata."""
    episode_id: str
    step_count: int
    task_name: str
    task_difficulty: str
    incident_resolved: bool
    root_cause_identified: bool
    correct_fix_applied: bool
    current_score: float
    max_steps: int
    elapsed_minutes: int