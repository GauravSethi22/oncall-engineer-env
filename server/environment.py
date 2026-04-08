from __future__ import annotations
import uuid
from typing import Any, Dict, Optional

from .simulator import ProductionSimulator
from .tasks import get_task, Task
from .graders import EpisodeTracker, get_grader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import (
    OnCallAction, OnCallObservation, OnCallState,
    Alert, ServiceStatus
)


class OnCallEnvironment:
    """
    Main environment class.
    Ties together the simulator, task, and grader.
    Called by the FastAPI server on every reset() and step().
    """

    def __init__(self, task_name: Optional[str] = None):
        self.task_name = task_name or "easy_crash"
        self.task: Optional[Task] = None
        self.simulator: Optional[ProductionSimulator] = None
        self.tracker: Optional[EpisodeTracker] = None
        self.episode_id: str = ""
        self._done: bool = False
        self._current_score: float = 0.0

    def reset(self) -> tuple[OnCallObservation, float, bool, OnCallState]:
        """Start a fresh episode."""
        self.task = get_task(self.task_name)
        self.simulator = ProductionSimulator(scenario=self.task.scenario_key)
        self.tracker = EpisodeTracker(task_name=self.task_name)
        self.episode_id = str(uuid.uuid4())[:8]
        self._done = False
        self._current_score = 0.01

        obs = self._build_observation(
            last_action_result="Incident started. Investigate and resolve.",
            last_action_error=None,
        )
        state = self._build_state()
        return obs, 0.01, False, state

    def step(
        self, action: OnCallAction
    ) -> tuple[OnCallObservation, float, bool, OnCallState]:
        """Execute one agent action and return observation + reward."""

        if self._done:
            obs = self._build_observation(
                last_action_result="Episode already finished.",
                last_action_error="Episode is done. Call reset() to start again.",
            )
            return obs, 0.01, True, self._build_state()

        self.tracker.steps_taken += 1
        self.tracker.actions_taken.append({
            "action_type": action.action_type,
            "target_service": action.target_service,
            "log_filter": action.log_filter,
            "metric_name": action.metric_name,
            "fix_type": action.fix_type,
            "team": action.team,
        })

        result, error = self._handle_action(action)

        grader = get_grader(self.task_name)
        grade = grader.grade(self.tracker)
        self._current_score = grade.score

        # Episode ends when agent writes summary OR max steps reached
        if action.action_type == "write_summary":
            self._done = True
        elif self.tracker.steps_taken >= self.task.max_steps:
            self._done = True
            result += f"\n[MAX STEPS REACHED] Episode ending. Final score: {grade.score:.3f}"

        # Reward = delta in score this step (encourages making progress)
        reward = round(grade.score / max(self.tracker.steps_taken, 1), 4)

        # On final step give full score as reward so the agent is motivated to finish.
        if self._done:
            reward = grade.score

        # SAFETY: Clamp reward to strictly (0, 1) — validator rejects 0.0 and 1.0
        reward = max(0.01, min(0.99, reward))

        obs = self._build_observation(
            last_action_result=result,
            last_action_error=error,
        )
        state = self._build_state(grade_feedback=grade.feedback)
        return obs, reward, self._done, state

    def _handle_action(
        self, action: OnCallAction
    ) -> tuple[str, Optional[str]]:
        """
        Route action to simulator and return (result_text, error_text).
        Also updates the tracker with what was done.
        """
        t = action.action_type

        if t == "query_logs":
            if not action.target_service:
                return "", "target_service is required for query_logs"
            data = self.simulator.query_logs(
                action.target_service, action.log_filter
            )
            if "error" in data:
                return "", data["error"]
            self.tracker.services_queried.append(action.target_service)
            logs_text = "\n".join(data["logs"]) if data["logs"] else "No logs found."
            return (
                f"Logs from {action.target_service} "
                f"({data['time_range']}):\n{logs_text}"
            ), None

        elif t == "check_metrics":
            if not action.target_service or not action.metric_name:
                return "", "target_service and metric_name required for check_metrics"
            data = self.simulator.check_metrics(
                action.target_service, action.metric_name
            )
            if "error" in data:
                return "", data["error"]
            self.tracker.services_queried.append(action.target_service)
            self.tracker.metrics_checked.append(
                f"{action.target_service}.{action.metric_name}"
            )
            extra = {
                k: v for k, v in data.items()
                if k not in ("service", "metric", "current_value", "unit", "trend")
            }
            extra_text = (
                "\n" + "\n".join(f"  {k}: {v}" for k, v in extra.items())
                if extra else ""
            )
            result_text = (
                f"{action.target_service} {action.metric_name}: "
                f"{data['current_value']}{data['unit']} "
                f"(trend: {data['trend']})"
            )

            # Surface the connections_by_service data so the leak is obvious.
            if "connection_pool" in data:
                result_text += f"\nConnection pool: {data['connection_pool']}"
            if "connections_by_service" in data:
                conn = data["connections_by_service"]
                sorted_conn = sorted(conn.items(), key=lambda x: x[1], reverse=True)
                result_text += "\nConnections held by each service:"
                for svc, count in sorted_conn:
                    result_text += f"\n  {svc}: {count} connections"
                top_svc = sorted_conn[0][0]
                result_text += (
                    f"\nCONCLUSION: {top_svc} is holding "
                    f"{sorted_conn[0][1]}/50 connections. "
                    f"Restart {top_svc} to release them."
                )
            if "note" in data:
                result_text += f"\nNOTE: {data['note']}"

            return result_text, None

        elif t == "check_deps":
            if not action.target_service:
                return "", "target_service required for check_deps"
            data = self.simulator.check_dependencies(action.target_service)
            if "error" in data:
                return "", data["error"]
            deps = data["depends_on"]
            dep_text = ", ".join(
                f"{d['service']}({d['status']})" for d in deps
            ) if deps else "none"
            used_by = ", ".join(data["depended_on_by"]) or "none"
            return (
                f"{action.target_service} depends on: {dep_text}\n"
                f"Used by: {used_by}"
            ), None

        elif t == "apply_fix":
            if not action.target_service or not action.fix_type:
                return "", "target_service and fix_type required for apply_fix"
            self.tracker.fixes_attempted.append({
                "service":  action.target_service,
                "fix_type": action.fix_type,
            })
            data = self.simulator.apply_fix(
                action.target_service, action.fix_type
            )
            if data.get("correct"):
                self.tracker.correct_fix_applied = True
            if data.get("result") == "worse":
                self.tracker.made_things_worse = True
            return data["message"], None

        elif t == "escalate":
            if not action.team:
                return "", "team required for escalate"
            self.tracker.escalations.append(action.team)
            data = self.simulator.escalate(action.team)
            return data["message"], None

        elif t == "write_summary":
            if not action.summary_text:
                return "", "summary_text required for write_summary"
            self.tracker.summary_text = action.summary_text
            grader = get_grader(self.task_name)
            grade = grader.grade(self.tracker)
            return (
                f"Incident summary recorded.\n"
                f"Score: {grade.score:.3f}\n"
                f"Feedback: {grade.feedback}"
            ), None

        else:
            return "", f"Unknown action_type: {t}"

    def _build_observation(
        self,
        last_action_result: str,
        last_action_error: Optional[str],
        grade_feedback: str = "",
    ) -> OnCallObservation:

        raw_alerts = self.simulator.get_alerts()
        alerts = [
            Alert(
                severity=a["severity"],
                service=a["service"],
                message=a["message"],
                fired_at=a["fired_at"],
            )
            for a in raw_alerts
        ]

        raw_services = self.simulator.get_service_statuses()
        services = [
            ServiceStatus(
                name=s["name"],
                status=s["status"],
                error_rate=s["error_rate"],
                latency_ms=s["latency_ms"],
                cpu_percent=s["cpu_percent"],
                memory_percent=s["memory_percent"],
                last_deployment=s["last_deployment"],
                recent_log_summary=s["recent_log_summary"],
            )
            for s in raw_services
        ]

        hints = self._get_unlocked_hints()

        result_with_feedback = last_action_result
        if grade_feedback:
            result_with_feedback += f"\n\nGrader feedback: {grade_feedback}"

        return OnCallObservation(
            alerts=alerts,
            services=services,
            last_action_result=result_with_feedback,
            last_action_error=last_action_error,
            elapsed_minutes=self.simulator.elapsed_minutes,
            steps_taken=self.tracker.steps_taken,
            task_description=self.task.description,
            task_name=self.task_name,
            unlocked_hints=hints,
        )

    def _build_state(self, grade_feedback: str = "") -> OnCallState:
        grader = get_grader(self.task_name)
        grade = grader.grade(self.tracker)
        # SAFETY: Clamp score to strictly (0, 1) — validator rejects 0.0 and 1.0
        clamped_score = max(0.01, min(0.99, grade.score))
        return OnCallState(
            episode_id=self.episode_id,
            step_count=self.tracker.steps_taken,
            task_name=self.task_name,
            task_difficulty=self.task.difficulty,
            incident_resolved=grade.incident_resolved,
            root_cause_identified=grade.root_cause_identified,
            correct_fix_applied=grade.correct_fix_applied,
            current_score=clamped_score,
            max_steps=self.task.max_steps,
            elapsed_minutes=self.simulator.elapsed_minutes,
        )

    def _get_unlocked_hints(self) -> list:
        """
        Progressively unlock hints as agent does good investigation.
        """
        hints = self.task.hints
        steps = self.tracker.steps_taken
        unlocked = []
        if steps >= 3 and len(hints) > 0:
            unlocked.append(hints[0])
        if steps >= 6 and len(hints) > 1:
            unlocked.append(hints[1])
        if steps >= 10 and len(hints) > 2:
            unlocked.append(hints[2])
        return unlocked

    @property
    def state(self) -> OnCallState:
        return self._build_state()