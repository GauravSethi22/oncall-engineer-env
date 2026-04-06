from __future__ import annotations
import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


SERVICES = [
    "user_service",
    "order_service",
    "payment_service",
    "notification_service",
    "database",
]

DEPENDENCIES = {
    "user_service":         [],
    "order_service":        ["user_service", "database"],
    "payment_service":      ["order_service", "database"],
    "notification_service": ["order_service"],
    "database":             [],
}


HEALTHY_LOGS = {
    "user_service": [
        "GET /api/users/123 200 12ms",
        "POST /api/auth/login 200 34ms",
        "GET /api/users/profile 200 8ms",
    ],
    "order_service": [
        "POST /api/orders 201 45ms",
        "GET /api/orders/456 200 22ms",
        "Order 789 status updated to CONFIRMED",
    ],
    "payment_service": [
        "Payment processed for order 456 in 120ms",
        "POST /api/payments 200 115ms",
        "Stripe webhook received: payment_intent.succeeded",
    ],
    "notification_service": [
        "Email sent to user@example.com",
        "SMS notification dispatched",
        "Push notification delivered to device abc123",
    ],
    "database": [
        "Query executed in 3ms: SELECT * FROM orders",
        "Connection pool: 12/50 active",
        "Index scan on orders.user_id: 1.2ms",
    ],
}

CRASH_LOGS = {
    "payment_service": [
        "ERROR NullPointerException at PaymentProcessor.java:142",
        "ERROR Failed to initialize payment gateway client",
        "WARN Retrying payment initialization... attempt 3/3",
        "ERROR Unhandled exception: Cannot read property 'amount' of undefined",
        "FATAL Service crashed after deployment v2.1.3",
    ],
}

CASCADE_LOGS = {
    "database": [
        "WARN Connection pool exhausted: 50/50 active",
        "ERROR New connection refused: pool limit reached",
        "WARN Query timeout after 30000ms",
        "ERROR Too many connections from payment_service (48 connections)",
    ],
    "payment_service": [
        "WARN Database connection attempt 1 failed, retrying...",
        "WARN Database connection attempt 2 failed, retrying...",
        "ERROR Unable to acquire DB connection after 30s",
        "ERROR Payment processing failed: database unavailable",
    ],
    "order_service": [
        "ERROR Timeout waiting for payment_service response (30s)",
        "ERROR POST /api/payments timeout",
        "WARN Circuit breaker OPEN for payment_service",
    ],
}

CORRUPTION_LOGS = {
    "order_service": [
        "POST /api/orders 201 44ms",
        "Order 1001 created with total: $45.00",
        "Order 1002 created with total: $12.50",
        "Discount calculator v1.2.1 applied",
        "Order 1003 created with total: $78.00",
    ],
    "database": [
        "Query executed in 2ms: INSERT INTO orders",
        "Connection pool: 10/50 active",
        "Background job discount_recalculator started",
        "Background job discount_recalculator completed: 847 orders updated",
    ],
}


class ProductionSimulator:
    """
    Simulates a fake e-commerce production system.
    Scenario controls which failure is injected.
    """

    def __init__(self, scenario: str):
        """
        scenario: "easy_crash" | "medium_cascade" | "hard_corruption"
        """
        self.scenario = scenario
        self.elapsed_minutes = random.randint(8, 15)
        self._fixes_applied: List[str] = []
        self._queried_services: List[str] = []

    def get_service_statuses(self) -> List[Dict[str, Any]]:
        statuses = []
        for svc in SERVICES:
            statuses.append(self._get_service_status(svc))
        return statuses

    def _get_service_status(self, service: str) -> Dict[str, Any]:
        """Returns health metrics for one service based on current scenario."""

        base = {
            "name": service,
            "status": "healthy",
            "error_rate": round(random.uniform(0.0, 0.02), 3),
            "latency_ms": round(random.uniform(10, 60), 1),
            "cpu_percent": round(random.uniform(20, 45), 1),
            "memory_percent": round(random.uniform(30, 55), 1),
            "last_deployment": "3 days ago (v2.1.1)",
            "recent_log_summary": "Normal traffic, no issues.",
        }

        if self.scenario == "easy_crash":
            if service == "payment_service":
                base.update({
                    "status": "down",
                    "error_rate": 1.0,
                    "latency_ms": 0.0,
                    "cpu_percent": 0.0,
                    "last_deployment": "12 minutes ago (v2.1.3)",
                    "recent_log_summary": "NullPointerException on startup. Service not responding.",
                })
            elif service == "order_service":
                base.update({
                    "status": "degraded",
                    "error_rate": 0.68,
                    "latency_ms": 4200.0,
                    "recent_log_summary": "Payment service timeouts causing checkout failures.",
                })

        elif self.scenario == "medium_cascade":
            if service == "database":
                base.update({
                    "status": "degraded",
                    "error_rate": 0.35,
                    "latency_ms": 8500.0,
                    "cpu_percent": 88.0,
                    "recent_log_summary": "Connection pool at 100%. New connections refused.",
                })
            elif service == "payment_service":
                base.update({
                    "status": "degraded",
                    "error_rate": 0.72,
                    "latency_ms": 31000.0,
                    "recent_log_summary": "DB connection timeouts. Holding 48 idle connections.",
                })
            elif service == "order_service":
                base.update({
                    "status": "degraded",
                    "error_rate": 0.61,
                    "latency_ms": 35000.0,
                    "recent_log_summary": "Payment service circuit breaker OPEN.",
                })

        elif self.scenario == "hard_corruption":
            if service == "order_service":
                base.update({
                    "status": "healthy",
                    "error_rate": 0.01,
                    "latency_ms": 48.0,
                    "recent_log_summary": "Normal traffic. Orders processing fine.",
                })
            if service == "database":
                base.update({
                    "recent_log_summary": "Background job discount_recalculator ran 2h ago.",
                })

        return base

    def query_logs(self, service: str, log_filter: Optional[str] = None) -> Dict[str, Any]:
        """Returns log lines for a service, optionally filtered."""
        # Track service investigation for grading and hint unlocking.
        self._queried_services.append(service)

        if service not in SERVICES:
            return {"error": f"Unknown service: {service}"}

        if self.scenario == "easy_crash":
            logs = CRASH_LOGS.get(service, HEALTHY_LOGS.get(service, []))
        elif self.scenario == "medium_cascade":
            logs = CASCADE_LOGS.get(service, HEALTHY_LOGS.get(service, []))
        elif self.scenario == "hard_corruption":
            logs = CORRUPTION_LOGS.get(service, HEALTHY_LOGS.get(service, []))
        else:
            logs = HEALTHY_LOGS.get(service, [])

        if log_filter:
            logs = [l for l in logs if log_filter.lower() in l.lower()]

        return {
            "service": service,
            "log_count": len(logs),
            "logs": logs,
            "time_range": f"last {self.elapsed_minutes} minutes",
        }

    def check_metrics(self, service: str, metric: str) -> Dict[str, Any]:
        """Returns a specific metric timeseries for a service."""
        if service not in SERVICES:
            return {"error": f"Unknown service: {service}"}

        status = self._get_service_status(service)
        metric_map = {
            "cpu":        status["cpu_percent"],
            "memory":     status["memory_percent"],
            "latency":    status["latency_ms"],
            "error_rate": status["error_rate"],
        }

        if metric not in metric_map:
            return {"error": f"Unknown metric: {metric}. Choose from: cpu, memory, latency, error_rate"}

        value = metric_map[metric]

        # For cascade scenario, add connection count detail that reveals the leak.
        extra = {}
        if self.scenario == "medium_cascade" and service == "database":
            extra["connection_pool"] = "50/50 (FULL)"
            extra["connections_by_service"] = {
                "payment_service": 48,
                "order_service": 2,
            }
        if self.scenario == "medium_cascade" and service == "payment_service":
            extra["note"] = "Holding 48 DB connections, never releasing them (connection leak)"

        return {
            "service": service,
            "metric": metric,
            "current_value": value,
            "unit": self._metric_unit(metric),
            "trend": self._metric_trend(service, metric),
            **extra,
        }

    def _metric_unit(self, metric: str) -> str:
        return {"cpu": "%", "memory": "%", "latency": "ms", "error_rate": "ratio"}.get(metric, "")

    def _metric_trend(self, service: str, metric: str) -> str:
        if self.scenario == "medium_cascade":
            if service == "database" and metric in ("latency", "error_rate", "cpu"):
                return "rapidly increasing"
            if service == "payment_service" and metric == "latency":
                return "increasing"
        if self.scenario == "easy_crash":
            if service == "payment_service":
                return "N/A (service down)"
        return "stable"

    def check_dependencies(self, service: str) -> Dict[str, Any]:
        if service not in SERVICES:
            return {"error": f"Unknown service: {service}"}
        deps = DEPENDENCIES.get(service, [])
        dep_health = []
        for dep in deps:
            s = self._get_service_status(dep)
            dep_health.append({"service": dep, "status": s["status"]})
        return {
            "service": service,
            "depends_on": dep_health,
            "depended_on_by": [
                s for s, deps in DEPENDENCIES.items() if service in deps
            ],
        }

    def apply_fix(self, service: str, fix_type: str) -> Dict[str, Any]:
        self._fixes_applied.append(f"{fix_type}:{service}")

        if self.scenario == "easy_crash":
            if service == "payment_service" and fix_type == "rollback":
                return {
                    "result": "success",
                    "message": "Rolled back payment_service from v2.1.3 to v2.1.2. Service restarting...",
                    "correct": True,
                }
            elif service == "payment_service" and fix_type == "restart":
                return {
                    "result": "partial",
                    "message": "Service restarted but still running v2.1.3. Same crash on startup.",
                    "correct": False,
                }
            else:
                return {
                    "result": "no_effect",
                    "message": f"{fix_type} on {service} had no effect on the incident.",
                    "correct": False,
                }

        elif self.scenario == "medium_cascade":
            if service == "payment_service" and fix_type == "restart":
                return {
                    "result": "success",
                    "message": "Restarted payment_service. Connection pool released. DB connections: 2/50.",
                    "correct": True,
                }
            elif service == "database" and fix_type == "restart":
                return {
                    "result": "worse",
                    "message": "Database restart caused 2 minute outage. All services now returning 503.",
                    "correct": False,
                }
            else:
                return {
                    "result": "no_effect",
                    "message": f"{fix_type} on {service} did not resolve connection exhaustion.",
                    "correct": False,
                }

        elif self.scenario == "hard_corruption":
            if service == "order_service" and fix_type == "rollback":
                return {
                    "result": "success",
                    "message": "Rolled back order_service discount_calculator to v1.2.0. New orders will have correct pricing.",
                    "correct": True,
                }
            else:
                return {
                    "result": "no_effect",
                    "message": f"{fix_type} on {service} did not address the data corruption.",
                    "correct": False,
                }

        return {"result": "unknown", "message": "Fix had unknown result."}


    def escalate(self, team: str) -> Dict[str, Any]:
        relevant = {
            "easy_crash":       "deployment_team",
            "medium_cascade":   "database_team",
            "hard_corruption":  "data_team",
        }
        correct_team = relevant.get(self.scenario)
        if team == correct_team:
            return {
                "result": "relevant",
                "message": f"{team} paged. They confirm: {self._escalation_hint()}",
            }
        return {
            "result": "not_needed",
            "message": f"{team} responded: this doesn't appear to be our domain.",
        }

    def _escalation_hint(self) -> str:
        hints = {
            "easy_crash":     "v2.1.3 had a known bug in PaymentProcessor. We pushed it by mistake.",
            "medium_cascade": "DB connection pool has been at 100% for 20 minutes. Looks like a connection leak.",
            "hard_corruption":"Background job discount_recalculator in v1.2.1 has a bug. It halves all discounts.",
        }
        return hints.get(self.scenario, "")


    def get_alerts(self) -> List[Dict[str, Any]]:
        alerts = {
            "easy_crash": [
                {"severity": "critical", "service": "payment_service",
                 "message": "Service DOWN — health check failing", "fired_at": "12 minutes ago"},
                {"severity": "critical", "service": "order_service",
                 "message": "Checkout error rate > 60%", "fired_at": "11 minutes ago"},
            ],
            "medium_cascade": [
                {"severity": "critical", "service": "order_service",
                 "message": "Checkout error rate > 55%", "fired_at": "18 minutes ago"},
                {"severity": "warning", "service": "payment_service",
                 "message": "High latency: avg 31s response time", "fired_at": "20 minutes ago"},
                {"severity": "warning", "service": "database",
                 "message": "Connection pool > 95% utilized", "fired_at": "22 minutes ago"},
            ],
            "hard_corruption": [
                {"severity": "info", "service": "order_service",
                 "message": "User reports: order prices appear incorrect (3 tickets)",
                 "fired_at": "45 minutes ago"},
            ],
        }
        return alerts.get(self.scenario, [])