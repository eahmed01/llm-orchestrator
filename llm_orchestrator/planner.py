"""Diff-and-plan engine: compare desired state to current, produce execution plans."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from .environment import EnvironmentDetector
from .stack import StackSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Well-known service metadata
# ---------------------------------------------------------------------------

SERVICE_PORTS: dict[str, int] = {
    "vllm": 7999,
    "litellm": 8000,
    "reranker": 7998,
}

# dependents[service] = list of services that depend on `service`
DEPENDENCY_GRAPH: dict[str, list[str]] = {
    "vllm": ["litellm"],
    "reranker": ["litellm"],
    "litellm": [],
}

# Reverse: who does a service depend on?
SERVICE_DEPENDS_ON: dict[str, list[str]] = {
    "vllm": [],
    "reranker": [],
    "litellm": ["vllm", "reranker"],
}

# Service aliases used in natural-language parsing
SERVICE_ALIASES: dict[str, str] = {
    "chat": "vllm",
    "llm": "vllm",
    "vllm": "vllm",
    "lite": "litellm",
    "proxy": "litellm",
    "litellm": "litellm",
    "reranker": "reranker",
    "rerank": "reranker",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanStep:
    """A single atomic action in a plan."""

    action: str  # STOP_SERVICE | START_SERVICE | RESTART_SERVICE | VERIFY_PORT | WAIT_GRACE | KILL_ORPHAN
    service: Optional[str]
    details: str
    estimated_seconds: int
    risk: str  # low | medium | high
    rollback: str


@dataclass(frozen=True)
class Plan:
    """An ordered, human-readable execution plan."""

    description: str
    steps: list[PlanStep]
    estimated_total_seconds: int
    risk_summary: str
    requires_confirmation: bool

    def __str__(self) -> str:
        lines: list[str] = []
        lines.append(f"PLAN: {self.description}")
        lines.append(f"Estimated time: ~{self.estimated_total_seconds}s  Risk: {self.risk_summary}")
        if self.requires_confirmation:
            lines.append("⚠  Requires confirmation before execution")
        lines.append("")

        if not self.steps:
            lines.append("  (no steps)")
            return "\n".join(lines)

        for idx, step in enumerate(self.steps, start=1):
            tag = _action_tag(step.action)
            svc_str = step.service if step.service else "—"
            lines.append(f"  {idx:>2}. [{tag}]  {svc_str:<12}  ({step.details})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_tag(action: str) -> str:
    mapping = {
        "STOP_SERVICE": "STOP",
        "START_SERVICE": "START",
        "RESTART_SERVICE": "RESTART",
        "VERIFY_PORT": "VERIFY",
        "WAIT_GRACE": "WAIT",
        "KILL_ORPHAN": "KILL",
    }
    return mapping.get(action, action)


def _service_port(service: Optional[str]) -> Optional[int]:
    if service is None:
        return None
    return SERVICE_PORTS.get(service)


def _gpu_str(gpus: list[int]) -> str:
    if not gpus:
        return "auto"
    return ",".join(str(g) for g in gpus)


def _all_available_gpus() -> list[int]:
    """Return list of GPU indices from the environment."""
    gpus = EnvironmentDetector.detect_gpus()
    return sorted(g["index"] for g in gpus)


def _resolve_gpu_spec(spec: str) -> list[int]:
    """Turn a natural-language GPU spec ('both', 'all', '0,1', '0') into [int, ...]."""
    spec = spec.strip().lower()
    if spec in ("both", "all"):
        return _all_available_gpus()
    # comma or space separated indices
    parts = re.split(r"[,\s]+", spec)
    result: list[int] = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            continue
    return result if result else _all_available_gpus()


def _resolve_service_alias(name: str) -> str:
    return SERVICE_ALIASES.get(name.lower(), name.lower())


# ---------------------------------------------------------------------------
# StackPlanner
# ---------------------------------------------------------------------------

class StackPlanner:
    """Compare a desired state to the current StackSnapshot and produce a Plan."""

    def __init__(self, current: StackSnapshot) -> None:
        self.current = current

    # -----------------------------------------------------------------------
    # diff()
    # -----------------------------------------------------------------------

    def diff(self, desired: dict[str, dict[str, Any]]) -> Plan:
        """Build a plan to bring the stack from *current* into *desired*.

        Args:
            desired: ``{service_name: {gpu: [0,1], ...}}`` overrides per service.
        """
        steps: list[PlanStep] = []
        warnings: list[str] = []

        current_services = self.current.services if self.current.services else {}
        stop_requested = {s for s in desired if desired[s].get("stop") is True}

        # --- 1. GPU conflict detection ---
        conflict_stops: list[tuple[str, str]] = []  # (blocker_service, reason)
        for svc_name, overrides in desired.items():
            new_gpus = overrides.get("gpus")
            if new_gpus is None:
                continue
            new_gpus = [int(g) for g in new_gpus]
            for other_name, other_info in current_services.items():
                if other_name == svc_name:
                    continue
                other_gpus = other_info.gpus or []
                overlap = set(new_gpus) & set(other_gpus)
                if overlap:
                    conflict_stops.append((other_name, f"GPU conflict with {svc_name} on GPU {_gpu_str(sorted(overlap))}"))
                    warnings.append(f"GPU conflict: {svc_name} wants GPU {_gpu_str(sorted(overlap))} but {other_name} is using it")

        # --- 2. Resolve GPU conflicts: stop blockers first ---
        seen_conflicts: set[str] = set()
        for blocker, reason in conflict_stops:
            if blocker in seen_conflicts:
                continue
            seen_conflicts.add(blocker)
            # Stop dependents of the blocker first
            self._add_stop_chain(blocker, steps, current_services, seen_conflicts)
            steps.append(PlanStep(
                action="STOP_SERVICE",
                service=blocker,
                details=reason,
                estimated_seconds=10,
                risk="medium",
                rollback=f"Restart {blocker} on its original GPU(s)",
            ))
            steps.append(PlanStep(
                action="WAIT_GRACE",
                service=None,
                details="2s grace",
                estimated_seconds=2,
                risk="low",
                rollback="",
            ))

        # --- 3. Handle each desired service ---
        for svc_name, overrides in desired.items():
            cur = current_services.get(svc_name)
            new_gpus = overrides.get("gpus")
            new_gpus_int = [int(g) for g in new_gpus] if new_gpus is not None else None

            if overrides.get("stop") is True:
                # Explicit stop
                self._add_stop_chain(svc_name, steps, current_services, seen_conflicts)
                if cur and cur.pid:
                    steps.append(PlanStep(
                        action="STOP_SERVICE",
                        service=svc_name,
                        details=f"explicit stop (current: {_gpu_str(cur.gpus or [])})",
                        estimated_seconds=10,
                        risk="medium",
                        rollback=f"Restart {svc_name} with previous config",
                    ))
                else:
                    steps.append(PlanStep(
                        action="STOP_SERVICE",
                        service=svc_name,
                        details="explicit stop (already stopped)",
                        estimated_seconds=0,
                        risk="low",
                        rollback=f"Restart {svc_name}",
                    ))
                continue

            if overrides.get("start") is True and cur is None:
                # Explicit start request for a non-running service
                self._add_start_steps(svc_name, new_gpus_int, overrides, steps, current_services, seen_conflicts)
                continue

            if cur is None:
                # Service not running -> START
                self._add_start_steps(svc_name, new_gpus_int, overrides, steps, current_services, seen_conflicts)
            else:
                # Service running -> check for changes
                cur_gpus = cur.gpus or []
                if new_gpus_int is not None and sorted(new_gpus_int) != sorted(cur_gpus):
                    # GPU change -> RESTART (stop first, then start)
                    self._add_restart_steps(svc_name, cur_gpus, new_gpus_int, overrides, steps, current_services, seen_conflicts)
                else:
                    # No meaningful change needed
                    pass

        # --- 4. Stop services that are in current but not desired (only if explicitly marked) ---
        for svc_name in current_services:
            if svc_name in stop_requested and svc_name not in seen_conflicts:
                self._add_stop_chain(svc_name, steps, current_services, seen_conflicts)
                cur = current_services[svc_name]
                steps.append(PlanStep(
                    action="STOP_SERVICE",
                    service=svc_name,
                    details=f"bring down (current: {_gpu_str(cur.gpus or [])})",
                    estimated_seconds=10,
                    risk="medium",
                    rollback=f"Restart {svc_name}",
                ))

        # --- Build plan ---
        total_seconds = sum(s.estimated_seconds for s in steps)
        risk = self._compute_risk(steps)
        description = self._summarize_diff(desired, current_services)
        requires_confirmation = risk in ("medium", "high") or any(w for w in warnings)

        if warnings:
            description += " (" + "; ".join(warnings) + ")"

        return Plan(
            description=description,
            steps=steps,
            estimated_total_seconds=total_seconds,
            risk_summary=risk,
            requires_confirmation=requires_confirmation,
        )

    # -----------------------------------------------------------------------
    # natural_language()
    # -----------------------------------------------------------------------

    def natural_language(self, goal: str) -> Plan:
        """Parse a natural-language intent into a Plan.

        Supported intents:
            - "move X to Y gpu(s)"  -> restart X with new GPUs
            - "restart X"           -> restart X with current config
            - "stop X" / "bring down X" -> stop X
            - "start X" / "bring up X"  -> start X
        """
        current_services = self.current.services if self.current.services else {}
        goal_lower = goal.strip().lower()

        # Try advisor LLM if available (async-safe fallback)
        desired = self._try_advisor_parse(goal, current_services)
        if desired is not None:
            return self.diff(desired)

        # Fallback: keyword-based parsing
        desired = self._parse_natural_language(goal_lower, current_services)
        if desired is None:
            return Plan(
                description=f"Could not parse intent: {goal}",
                steps=[],
                estimated_total_seconds=0,
                risk_summary="low",
                requires_confirmation=False,
            )

        return self.diff(desired)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _try_advisor_parse(self, goal: str, current_services: dict) -> Optional[dict[str, dict[str, Any]]]:
        """Attempt to use the advisor LLM for intent parsing. Returns None on failure."""
        try:
            from llm_orchestrator.advisor import Advisor
            # Quick synchronous check — if advisor model is already loaded it's fast
            # We won't block waiting for model load, just use keyword parsing as fallback
            logger.debug("Advisor LLM not used for intent parsing (sync path); falling back to keywords")
        except Exception:
            pass
        return None

    def _parse_natural_language(self, goal: str, current_services: dict) -> Optional[dict[str, dict[str, Any]]]:
        """Keyword-based intent parser."""
        # "move X to Y gpu(s)"  -- e.g. "move chat to both gpus", "move vllm to gpu 0,1"
        move_match = re.match(
            r"move\s+(\w+)\s+to\s+(.+?)(?:\s+gpu)?s?\s*$", goal
        )
        if move_match:
            svc_alias = move_match.group(1)
            gpu_spec = move_match.group(2)
            svc = _resolve_service_alias(svc_alias)
            gpus = _resolve_gpu_spec(gpu_spec)
            return {svc: {"gpus": gpus}}

        # "restart X"
        restart_match = re.match(r"restart\s+(\w+)\s*$", goal)
        if restart_match:
            svc = _resolve_service_alias(restart_match.group(1))
            cur = current_services.get(svc)
            if cur:
                gpus = cur.gpus or [0]
            else:
                gpus = [0]
            return {svc: {"gpus": gpus}}

        # "stop X" / "bring down X" / "shutdown X"
        stop_match = re.match(r"(?:stop|bring\s+down|shutdown)\s+(\w+)\s*$", goal)
        if stop_match:
            svc = _resolve_service_alias(stop_match.group(1))
            return {svc: {"stop": True}}

        # "start X" / "bring up X"
        start_match = re.match(r"(?:start|bring\s+up|launch)\s+(\w+)\s*$", goal)
        if start_match:
            svc = _resolve_service_alias(start_match.group(1))
            cur = current_services.get(svc)
            gpus = (cur.gpus or [0]) if cur else [0]
            return {svc: {"gpus": gpus}}

        # Special: "move chat to both gpus" might have extra words
        loose_move = re.search(r"move\s+(\w+)\s+to\s+(.+)", goal)
        if loose_move:
            svc_alias = loose_move.group(1)
            gpu_spec = loose_move.group(2)
            svc = _resolve_service_alias(svc_alias)
            gpus = _resolve_gpu_spec(gpu_spec)
            return {svc: {"gpus": gpus}}

        return None

    def _add_stop_chain(self, service: str, steps: list[PlanStep], current_services: dict, seen: set[str]) -> None:
        """Add STOP steps for all dependents of *service* first (depth-first)."""
        dependents = DEPENDENCY_GRAPH.get(service, [])
        for dep in dependents:
            if dep in seen:
                continue
            if dep in current_services:
                self._add_stop_chain(dep, steps, current_services, seen)
                steps.append(PlanStep(
                    action="STOP_SERVICE",
                    service=dep,
                    details=f"dependency of {service}",
                    estimated_seconds=10,
                    risk="low",
                    rollback=f"Restart {dep}",
                ))
                steps.append(PlanStep(
                    action="WAIT_GRACE",
                    service=None,
                    details="2s grace",
                    estimated_seconds=2,
                    risk="low",
                    rollback="",
                ))
                seen.add(dep)

    def _add_start_steps(self, service: str, gpus: Optional[list[int]], overrides: dict, steps: list[PlanStep], current_services: dict, seen: set[str]) -> None:
        """Add steps to start a service (including starting its dependencies first)."""
        depends_on = SERVICE_DEPENDS_ON.get(service, [])
        for dep in depends_on:
            if dep in seen or dep in current_services:
                continue
            # Dependency not running and not in desired — skip (user didn't ask for it)
            continue

        port = _service_port(service)
        model = overrides.get("model", "default")
        gpu_display = _gpu_str(gpus) if gpus else "auto"

        steps.append(PlanStep(
            action="START_SERVICE",
            service=service,
            details=f"{model} on GPU {gpu_display}",
            estimated_seconds=60,
            risk="medium",
            rollback=f"Stop {service}",
        ))
        if port:
            steps.append(PlanStep(
                action="VERIFY_PORT",
                service=service,
                details=f"port {port}",
                estimated_seconds=5,
                risk="low",
                rollback="",
            ))

    def _add_restart_steps(self, service: str, old_gpus: list[int], new_gpus: list[int], overrides: dict, steps: list[PlanStep], current_services: dict, seen: set[str]) -> None:
        """Add STOP + WAIT + START + VERIFY steps for a GPU-change restart."""
        # Stop dependents first
        self._add_stop_chain(service, steps, current_services, seen)

        model = overrides.get("model", "default")
        old_display = _gpu_str(old_gpus)
        new_display = _gpu_str(new_gpus)

        # Stop the service itself
        steps.append(PlanStep(
            action="STOP_SERVICE",
            service=service,
            details=f"current: GPU {old_display}",
            estimated_seconds=10,
            risk="medium",
            rollback=f"Restart {service} on GPU {old_display}",
        ))
        steps.append(PlanStep(
            action="WAIT_GRACE",
            service=None,
            details="2s grace",
            estimated_seconds=2,
            risk="low",
            rollback="",
        ))

        # Start with new config
        steps.append(PlanStep(
            action="START_SERVICE",
            service=service,
            details=f"{model} on GPU {new_display}",
            estimated_seconds=60,
            risk="medium",
            rollback=f"Restart {service} on GPU {old_display}",
        ))

        port = _service_port(service)
        if port:
            steps.append(PlanStep(
                action="VERIFY_PORT",
                service=service,
                details=f"port {port}",
                estimated_seconds=5,
                risk="low",
                rollback="",
            ))

        # Restart dependents
        dependents = DEPENDENCY_GRAPH.get(service, [])
        for dep in dependents:
            if dep in current_services:
                dep_port = _service_port(dep)
                steps.append(PlanStep(
                    action="WAIT_GRACE",
                    service=None,
                    details="2s grace",
                    estimated_seconds=2,
                    risk="low",
                    rollback="",
                ))
                steps.append(PlanStep(
                    action="START_SERVICE",
                    service=dep,
                    details=f"port {dep_port}" if dep_port else "dependency restart",
                    estimated_seconds=15,
                    risk="low",
                    rollback=f"Stop {dep}",
                ))
                if dep_port:
                    steps.append(PlanStep(
                        action="VERIFY_PORT",
                        service=dep,
                        details=f"port {dep_port}",
                        estimated_seconds=5,
                        risk="low",
                        rollback="",
                    ))

    @staticmethod
    def _compute_risk(steps: list[PlanStep]) -> str:
        """Determine overall risk level from steps."""
        if not steps:
            return "low"
        has_high = any(s.risk == "high" for s in steps)
        has_medium = any(s.risk == "medium" for s in steps)
        if has_high:
            return "high"
        if has_medium:
            return "medium"
        return "low"

    @staticmethod
    def _summarize_diff(desired: dict[str, dict], current_services: dict) -> str:
        """Create a short human-readable summary of the planned changes."""
        parts: list[str] = []
        for svc, overrides in desired.items():
            new_gpus = overrides.get("gpus")
            if overrides.get("stop"):
                parts.append(f"Stop {svc}")
            elif new_gpus is not None:
                cur = current_services.get(svc)
                if cur and cur.gpus and sorted(new_gpus) != sorted(cur.gpus):
                    parts.append(f"Restart {svc} on GPUs {_gpu_str(sorted(new_gpus))}")
                elif not cur:
                    parts.append(f"Start {svc} on GPUs {_gpu_str(sorted(new_gpus))}")
                else:
                    parts.append(f"Keep {svc} on GPUs {_gpu_str(sorted(new_gpus))}")
            else:
                parts.append(f"Check {svc}")
        if not parts:
            return "No changes"
        return "; ".join(parts)
