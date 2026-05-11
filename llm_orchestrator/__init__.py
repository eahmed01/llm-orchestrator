"""llm-orchestrator: Automate trial-and-error LLM startup with intelligent retry."""

__version__ = "0.2.0"

__all__ = [
    "Advisor",
    "Attempt",
    "EnvironmentDetector",
    "ModelProfile",
    "Monitor",
    "Orchestrator",
    "OrchestratorConfig",
    "Plan",
    "PlanStep",
    "ProfileStore",
    "ServiceConfig",
    "ServiceManager",
    "ServiceResult",
    "StackConfig",
    "StackDetector",
    "StackPlanner",
    "StackServiceInfo",
    "StackSnapshot",
    "UserPreferences",
    "default_stack_configs",
    "get_store",
    "record_event",
    "read_events",
    "history_stats",
]


def __getattr__(name: str):
    """Lazy import to avoid loading heavy deps (transformers/numpy) at package level."""
    if name == "Advisor":
        from llm_orchestrator.advisor import Advisor as _Advisor
        return _Advisor
    elif name == "Attempt":
        from llm_orchestrator.profiles import Attempt as _Attempt
        return _Attempt
    elif name == "EnvironmentDetector":
        from llm_orchestrator.environment import EnvironmentDetector as _EnvironmentDetector
        return _EnvironmentDetector
    elif name == "ModelProfile":
        from llm_orchestrator.profiles import ModelProfile as _ModelProfile
        return _ModelProfile
    elif name == "Monitor":
        from llm_orchestrator.monitor import Monitor as _Monitor
        return _Monitor
    elif name == "Orchestrator":
        from llm_orchestrator.orchestrator import Orchestrator as _Orchestrator
        return _Orchestrator
    elif name in ("OrchestratorConfig", "ServiceConfig", "UserPreferences"):
        import llm_orchestrator.config as _config
        return getattr(_config, name)
    elif name in ("Plan", "PlanStep", "StackPlanner"):
        import llm_orchestrator.planner as _planner
        return getattr(_planner, name)
    elif name in ("ProfileStore", "get_store"):
        import llm_orchestrator.profiles as _profiles
        return getattr(_profiles, name)
    elif name in ("ServiceManager", "ServiceResult"):
        import llm_orchestrator.service as _service
        return getattr(_service, name)
    elif name in ("StackConfig", "StackDetector", "StackServiceInfo", "StackSnapshot", "default_stack_configs"):
        import llm_orchestrator.stack as _stack
        return getattr(_stack, name)
    elif name in ("record_event", "read_events", "history_stats"):
        import llm_orchestrator.history as _history
        return getattr(_history, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
