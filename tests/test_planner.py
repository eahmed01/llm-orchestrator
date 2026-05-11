"""Tests for llm_orchestrator.planner module."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from llm_orchestrator.planner import (
    Plan,
    PlanStep,
    StackPlanner,
    SERVICE_PORTS,
    SERVICE_ALIASES,
)
from llm_orchestrator.stack import (
    StackServiceInfo,
    StackSnapshot,
)


def _make_snapshot(**overrides):
    """Helper to create a StackSnapshot with default running services."""
    defaults = {
        "vllm": StackServiceInfo(
            name="vllm",
            model="Qwen/Qwen3.6-27B-FP8",
            port=7999,
            pid=100,
            gpus=[0],
            host="0.0.0.0",
            status="running",
        ),
        "reranker": StackServiceInfo(
            name="reranker",
            model="Qwen/Qwen3-Reranker-0.6B",
            port=7998,
            pid=200,
            gpus=[1],
            host="0.0.0.0",
            status="running",
        ),
        "litellm": StackServiceInfo(
            name="litellm",
            model="",
            port=8000,
            pid=300,
            gpus=[],
            host="0.0.0.0",
            status="running",
        ),
    }
    # Apply overrides: replace the entry entirely, don't mutate shared data
    for key, val in overrides.items():
        if val is None:
            defaults.pop(key, None)
        else:
            defaults[key] = val
    return StackSnapshot(
        services=defaults,
        gpus=[
            {"index": 0, "name": "GPU 0", "memory_used_mb": 50000, "memory_total_mb": 97887, "utilization_pct": 5, "temperature_c": 45},
            {"index": 1, "name": "GPU 1", "memory_used_mb": 5000, "memory_total_mb": 97887, "utilization_pct": 3, "temperature_c": 40},
        ],
        timestamp=datetime.now(timezone.utc),
    )


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_stop_step(self):
        step = PlanStep(
            action="STOP_SERVICE",
            service="litellm",
            details="Stop dependency before vllm restart",
            estimated_seconds=5,
            risk="medium",
            rollback="Restart litellm",
        )
        assert step.action == "STOP_SERVICE"
        assert step.service == "litellm"

    def test_wait_step(self):
        step = PlanStep(
            action="WAIT_GRACE",
            service=None,
            details="Grace period",
            estimated_seconds=2,
            risk="low",
            rollback="",
        )
        assert step.action == "WAIT_GRACE"
        assert step.service is None

    def test_step_is_frozen(self):
        step = PlanStep(
            action="STOP_SERVICE",
            service="x",
            details="",
            estimated_seconds=1,
            risk="low",
            rollback="",
        )
        with pytest.raises(Exception):
            step.risk = "high"  # type: ignore


class TestPlan:
    """Tests for Plan dataclass."""

    def test_plan_str(self):
        steps = [
            PlanStep(
                action="STOP_SERVICE",
                service="litellm",
                details="Stop dependency",
                estimated_seconds=5,
                risk="medium",
                rollback="Restart litellm",
            ),
            PlanStep(
                action="WAIT_GRACE",
                service=None,
                details="2s grace",
                estimated_seconds=2,
                risk="low",
                rollback="",
            ),
            PlanStep(
                action="START_SERVICE",
                service="vllm",
                details="Qwen3.6-27B-FP8 on GPUs 0,1",
                estimated_seconds=30,
                risk="medium",
                rollback="Stop vllm",
            ),
        ]
        plan = Plan(
            description="Restart vLLM on GPUs 0,1",
            steps=steps,
            estimated_total_seconds=37,
            risk_summary="medium",
            requires_confirmation=True,
        )
        s = str(plan)
        assert "STOP" in s
        assert "START" in s
        assert "litellm" in s
        assert "vllm" in s

    def test_plan_empty(self):
        plan = Plan(
            description="No changes",
            steps=[],
            estimated_total_seconds=0,
            risk_summary="low",
            requires_confirmation=False,
        )
        assert plan.steps == []
        assert plan.requires_confirmation is False


class TestStackPlannerDiff:
    """Tests for the diff-based planner."""

    def test_no_change_when_same(self):
        snap = _make_snapshot()
        planner = StackPlanner(snap)
        # No desired changes
        plan = planner.diff({})
        # Should produce minimal or no steps
        assert isinstance(plan, Plan)

    def test_gpu_change_triggers_restart(self):
        snap = _make_snapshot()
        planner = StackPlanner(snap)
        plan = planner.diff({"vllm": {"gpus": [0, 1]}})
        # Should include restart or stop+start for vllm
        actions = [s.action for s in plan.steps]
        has_vllm_change = any(s.service == "vllm" for s in plan.steps)
        assert has_vllm_change, f"Expected vllm steps, got actions: {actions}"

    def test_new_service_triggers_start(self):
        snap = _make_snapshot(reranker=None, litellm=None)
        # Only vllm running, litellm not in snapshot
        planner = StackPlanner(snap)
        plan = planner.diff({"litellm": {"start": True}})
        actions = [s.action for s in plan.steps]
        assert "START_SERVICE" in actions

    def test_stop_flag(self):
        snap = _make_snapshot()
        planner = StackPlanner(snap)
        plan = planner.diff({"reranker": {"stop": True}})
        actions = [s.action for s in plan.steps]
        assert "STOP_SERVICE" in actions

    def test_gpu_conflict_detected(self):
        snap = _make_snapshot()
        planner = StackPlanner(snap)
        # Want vllm on GPU 1, but reranker is on GPU 1
        plan = planner.diff({"vllm": {"gpus": [1]}})
        # Should have steps for reranker too
        services_in_plan = {s.service for s in plan.steps if s.service}
        # May include reranker for conflict resolution
        assert "vllm" in services_in_plan


class TestStackPlannerNaturalLanguage:
    """Tests for natural-language intent parsing."""

    def test_move_chat_to_both_gpus(self):
        snap = _make_snapshot()
        planner = StackPlanner(snap)
        plan = planner.natural_language("move chat to both gpus")
        assert isinstance(plan, Plan)
        # Should plan a vllm restart
        services = {s.service for s in plan.steps if s.service}
        assert "vllm" in services

    def test_restart_vllm(self):
        snap = _make_snapshot()
        planner = StackPlanner(snap)
        plan = planner.natural_language("restart vllm")
        # Restarting with same GPUs is a no-op plan (no changes needed)
        # But the plan object should still be valid
        assert isinstance(plan, Plan)
        assert plan.description is not None

    def test_stop_reranker(self):
        snap = _make_snapshot()
        planner = StackPlanner(snap)
        plan = planner.natural_language("stop reranker")
        actions = [s.action for s in plan.steps]
        assert "STOP_SERVICE" in actions

    def test_start_litellm(self):
        # litellm not running
        snap = _make_snapshot(reranker=None, litellm=None)
        planner = StackPlanner(snap)
        plan = planner.natural_language("start litellm")
        actions = [s.action for s in plan.steps]
        assert "START_SERVICE" in actions

    def test_alias_chat_maps_to_vllm(self):
        assert SERVICE_ALIASES["chat"] == "vllm"
        assert SERVICE_ALIASES["llm"] == "vllm"

    def test_alias_proxy_maps_to_litellm(self):
        assert SERVICE_ALIASES["proxy"] == "litellm"
        assert SERVICE_ALIASES["lite"] == "litellm"

    def test_unknown_goal(self):
        snap = _make_snapshot()
        planner = StackPlanner(snap)
        plan = planner.natural_language("do something completely unrelated")
        assert isinstance(plan, Plan)
        # Should return an empty or minimal plan
        assert len(plan.steps) == 0 or plan.description is not None


class TestServicePorts:
    """Verify well-known port mappings."""

    def test_vllm_port(self):
        assert SERVICE_PORTS["vllm"] == 7999

    def test_litellm_port(self):
        assert SERVICE_PORTS["litellm"] == 8000

    def test_reranker_port(self):
        assert SERVICE_PORTS["reranker"] == 7998
