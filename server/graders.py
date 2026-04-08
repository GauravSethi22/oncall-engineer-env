from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GradeResult:
    """Returned after every step — running score + what contributed to it."""
    score: float
    breakdown: Dict[str, float]
    feedback: str
    incident_resolved: bool
    root_cause_identified: bool
    correct_fix_applied: bool


@dataclass
class EpisodeTracker:
    """
    Tracks everything the agent has done this episode.
    The grader reads this to compute scores.
    """
    task_name: str
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    services_queried: List[str] = field(default_factory=list)
    metrics_checked: List[str] = field(default_factory=list)
    fixes_attempted: List[Dict[str, str]] = field(default_factory=list)
    escalations: List[str] = field(default_factory=list)
    summary_text: Optional[str] = None
    steps_taken: int = 0
    correct_fix_applied: bool = False
    made_things_worse: bool = False
    root_cause_service_queried: bool = False
    cascading_chain_traced: bool = False
    corruption_source_found: bool = False


class BaseGrader:
    """
    Shared grading utilities used by all three task graders.
    """

    def _score_summary(
        self,
        summary_text: Optional[str],
        keywords: List[str]
    ) -> float:
        """
        Scores the agent's written summary.
        Checks how many required keywords appear.
        Returns 0.0 to 1.0
        """
        if not summary_text:
            return 0.0
        text = summary_text.lower()
        hits = sum(1 for kw in keywords if kw.lower() in text)
        return round(hits / len(keywords), 3) if keywords else 0.0

    def _score_speed(self, steps_taken: int, max_steps: int) -> float:
        """
        Rewards resolving the incident quickly.
        Full score if done in first 40% of steps.
        Zero if took all steps.
        """
        if steps_taken <= 0:
            return 0.0
        ratio = steps_taken / max_steps
        if ratio <= 0.4:
            return 1.0
        elif ratio <= 0.7:
            return 0.5
        elif ratio <= 0.9:
            return 0.2
        return 0.0

    def _score_investigation(
        self,
        services_queried: List[str],
        correct_service: str,
        total_services: int = 5,
    ) -> float:
        """
        Partial credit for investigating the right service.
        Penalises querying too many irrelevant services.
        """
        # Avoid rewarding random exploration; only the correct service earns positive credit.
        if correct_service in services_queried:
            relevant = 1
        else:
            relevant = 0

        # Penalise excessive noise (querying everything blindly)
        noise = len(set(services_queried)) - relevant
        noise_penalty = min(noise * 0.05, 0.3)

        return max(round(relevant * 0.5 - noise_penalty, 3), 0.0)


class EasyCrashGrader(BaseGrader):
    """
    Grades the easy_crash scenario.

    Scoring breakdown:
      30%  Correct fix applied (rollback payment_service)
      25%  Root cause identified (queried payment_service logs)
      20%  Summary quality (keywords: payment_service, v2.1.3, rollback)
      15%  Speed bonus
      10%  Escalated to correct team
    """

    MAX_STEPS = 15
    KEYWORDS = ["payment_service", "v2.1.3", "rollback", "deployment"]

    def grade(self, tracker: EpisodeTracker) -> GradeResult:
        breakdown = {}

        correct_fix = any(
            f["service"] == "payment_service" and f["fix_type"] == "rollback"
            for f in tracker.fixes_attempted
        )
        breakdown["correct_fix"] = 0.30 if correct_fix else 0.0

        # Penalty if made things worse (tried restart instead of rollback first)
        bad_fix = any(
            f["service"] == "payment_service" and f["fix_type"] == "restart"
            for f in tracker.fixes_attempted
        )
        if bad_fix and not correct_fix:
            breakdown["correct_fix"] = -0.10

        queried_root = "payment_service" in tracker.services_queried
        breakdown["root_cause"] = 0.25 if queried_root else 0.0

        checked_deploy_logs = any(
            a.get("action_type") == "query_logs"
            and a.get("target_service") == "payment_service"
            and a.get("log_filter") in ("error", "crash", "deploy", None)
            for a in tracker.actions_taken
        )
        if checked_deploy_logs:
            breakdown["root_cause"] = min(breakdown["root_cause"] + 0.05, 0.25)

        breakdown["summary"] = (
            self._score_summary(tracker.summary_text, self.KEYWORDS) * 0.20
        )

        if correct_fix:
            breakdown["speed"] = self._score_speed(
                tracker.steps_taken, self.MAX_STEPS
            ) * 0.15
        else:
            breakdown["speed"] = 0.0

        correct_escalation = "deployment_team" in tracker.escalations
        breakdown["escalation"] = 0.10 if correct_escalation else 0.0

        total = round(sum(breakdown.values()), 3)
        total = max(0.01, min(0.99, total))

        return GradeResult(
            score=total,
            breakdown=breakdown,
            feedback=self._build_feedback(breakdown, correct_fix, queried_root),
            incident_resolved=correct_fix,
            root_cause_identified=queried_root,
            correct_fix_applied=correct_fix,
        )

    def _build_feedback(
        self,
        breakdown: Dict[str, float],
        correct_fix: bool,
        queried_root: bool,
    ) -> str:
        parts = []
        if correct_fix:
            parts.append("Correctly rolled back payment_service.")
        else:
            parts.append("Did not apply the correct fix (rollback payment_service).")
        if queried_root:
            parts.append("Investigated the right service.")
        else:
            parts.append("Did not investigate payment_service logs.")
        if breakdown.get("summary", 0) > 0.1:
            parts.append("Summary captured key details.")
        else:
            parts.append("Summary was missing or incomplete.")
        return " ".join(parts)




class MediumCascadeGrader(BaseGrader):
    """
    Grades the medium_cascade scenario.

    This task rewards tracing the cascade chain, not just
    fixing the first symptomatic service.

    Scoring breakdown:
      30%  Correct fix applied (restart payment_service)
      25%  Traced the full cascade chain (order→payment→database)
      20%  Summary quality
      15%  Speed bonus
      10%  Correct escalation (database_team)
    """

    MAX_STEPS = 20
    KEYWORDS = ["payment_service", "connection", "database", "pool", "leak", "restart"]
    CHAIN = ["order_service", "payment_service", "database"]

    def grade(self, tracker: EpisodeTracker) -> GradeResult:
        breakdown = {}

        correct_fix = any(
            f["service"] == "payment_service" and f["fix_type"] == "restart"
            for f in tracker.fixes_attempted
        )
        breakdown["correct_fix"] = 0.30 if correct_fix else 0.0

        # Penalty: restarting database is harmful
        db_restart = any(
            f["service"] == "database" and f["fix_type"] == "restart"
            for f in tracker.fixes_attempted
        )
        if db_restart:
            breakdown["correct_fix"] = max(
                breakdown["correct_fix"] - 0.20, -0.20
            )

        chain_queried = [
            svc for svc in self.CHAIN
            if svc in tracker.services_queried
        ]
        chain_score = len(chain_queried) / len(self.CHAIN)

        checked_db_connections = any(
            a.get("action_type") == "check_metrics"
            and a.get("target_service") == "database"
            for a in tracker.actions_taken
        )
        if checked_db_connections:
            chain_score = min(chain_score + 0.15, 1.0)

        breakdown["cascade_trace"] = round(chain_score * 0.25, 3)

        breakdown["summary"] = (
            self._score_summary(tracker.summary_text, self.KEYWORDS) * 0.20
        )

        if correct_fix:
            breakdown["speed"] = self._score_speed(
                tracker.steps_taken, self.MAX_STEPS
            ) * 0.15
        else:
            breakdown["speed"] = 0.0

        breakdown["escalation"] = (
            0.10 if "database_team" in tracker.escalations else 0.0
        )

        total = round(sum(breakdown.values()), 3)
        total = max(0.01, min(0.99, total))

        return GradeResult(
            score=total,
            breakdown=breakdown,
            feedback=self._build_feedback(breakdown, correct_fix, chain_queried, db_restart),
            incident_resolved=correct_fix,
            root_cause_identified=len(chain_queried) == len(self.CHAIN),
            correct_fix_applied=correct_fix,
        )

    def _build_feedback(
        self,
        breakdown: Dict[str, float],
        correct_fix: bool,
        chain_queried: List[str],
        db_restart: bool,
    ) -> str:
        parts = []
        if correct_fix:
            parts.append("Correctly restarted payment_service to release connections.")
        else:
            parts.append("Did not apply correct fix (restart payment_service).")
        if db_restart:
            parts.append("WARNING: Restarting database caused additional outage.")
        parts.append(
            f"Traced {len(chain_queried)}/3 services in cascade chain: {chain_queried}."
        )
        if breakdown.get("summary", 0) > 0.1:
            parts.append("Summary captured key details.")
        return " ".join(parts)




class HardCorruptionGrader(BaseGrader):
    """
    Grades the hard_corruption scenario.

    This is the hardest because there are no obvious alerts.
    The agent must proactively investigate and find silent corruption.

    Scoring breakdown:
      30%  Correct fix applied (rollback order_service)
      25%  Found the corruption source (queried order logs + database job logs)
      20%  Summary quality
      15%  Speed bonus
      10%  Correct escalation (data_team)
    """

    MAX_STEPS = 25
    KEYWORDS = [
        "order_service", "discount", "background job",
        "corruption", "rollback", "v1.2.1"
    ]

    def grade(self, tracker: EpisodeTracker) -> GradeResult:
        breakdown = {}

        correct_fix = any(
            f["service"] == "order_service" and f["fix_type"] == "rollback"
            for f in tracker.fixes_attempted
        )
        breakdown["correct_fix"] = 0.30 if correct_fix else 0.0

        queried_order = "order_service" in tracker.services_queried
        queried_db = "database" in tracker.services_queried

        found_bg_job = any(
            a.get("action_type") == "query_logs"
            and a.get("target_service") in ("order_service", "database")
            and a.get("log_filter") in (
                "discount", "background", "job", "recalculator", None
            )
            for a in tracker.actions_taken
        )

        investigation_score = 0.0
        if queried_order:
            investigation_score += 0.10
        if queried_db:
            investigation_score += 0.08
        if found_bg_job:
            investigation_score += 0.07

        breakdown["corruption_found"] = round(
            min(investigation_score, 0.25), 3
        )

        breakdown["summary"] = (
            self._score_summary(tracker.summary_text, self.KEYWORDS) * 0.20
        )

        if correct_fix:
            breakdown["speed"] = self._score_speed(
                tracker.steps_taken, self.MAX_STEPS
            ) * 0.15
        else:
            breakdown["speed"] = 0.0

        breakdown["escalation"] = (
            0.10 if "data_team" in tracker.escalations else 0.0
        )

        total = round(sum(breakdown.values()), 3)
        total = max(0.01, min(0.99, total))

        return GradeResult(
            score=total,
            breakdown=breakdown,
            feedback=self._build_feedback(
                breakdown, correct_fix, queried_order, queried_db, found_bg_job
            ),
            incident_resolved=correct_fix,
            root_cause_identified=found_bg_job,
            correct_fix_applied=correct_fix,
        )

    def _build_feedback(
        self,
        breakdown: Dict[str, float],
        correct_fix: bool,
        queried_order: bool,
        queried_db: bool,
        found_bg_job: bool,
    ) -> str:
        parts = []
        if correct_fix:
            parts.append("Correctly rolled back order_service.")
        else:
            parts.append("Did not apply correct fix (rollback order_service).")
        if found_bg_job:
            parts.append("Found the background job as corruption source.")
        elif queried_order and queried_db:
            parts.append("Investigated right services but missed the background job.")
        else:
            parts.append("Did not fully investigate the corruption source.")
        if breakdown.get("summary", 0) > 0.1:
            parts.append("Summary captured key details.")
        else:
            parts.append("Summary was missing or weak.")
        return " ".join(parts)



GRADERS = {
    "easy_crash":     EasyCrashGrader(),
    "medium_cascade": MediumCascadeGrader(),
    "hard_corruption": HardCorruptionGrader(),
}


def get_grader(task_name: str) -> BaseGrader:
    if task_name not in GRADERS:
        raise ValueError(f"No grader for task: {task_name}")
    return GRADERS[task_name]