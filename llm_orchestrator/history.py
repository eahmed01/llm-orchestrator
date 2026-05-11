"""History tracking for LLM service lifecycle events.

Stores structured JSON-lines logs under ~/.llmconf/history/events.jsonl
capturing every start/stop/restart with configuration details.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

HISTORY_DIR = Path.home() / ".llmconf" / "history"
EVENTS_FILE = HISTORY_DIR / "events.jsonl"


def _ensure_dir() -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR


def _capture_env(prefix: str = "VLLM_") -> dict[str, str]:
    """Capture relevant environment variables."""
    return {
        k: v for k, v in os.environ.items()
        if k.startswith(prefix) or k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME")
    }


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def record_event(
    action: str,
    service: str,
    *,
    model: Optional[str] = None,
    args: Optional[dict[str, Any]] = None,
    port: Optional[int] = None,
    gpu: Optional[list[int]] = None,
    interface: Optional[str] = None,
    pid: Optional[int] = None,
    success: bool = True,
    duration_s: Optional[float] = None,
    error: Optional[str] = None,
    note: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> None:
    """Append one JSON-lines event to the history log.

    Args:
        action: ``start``, ``stop``, or ``restart``.
        service: Service name (e.g. ``vllm``, ``reranker``, ``litellm``).
        model: HuggingFace model ID.
        args: vLLM / service arguments.
        port: TCP port the service binds to.
        gpu: GPU indices assigned.
        interface: Network interface bound.
        pid: Process ID if the service started.
        success: Whether the action succeeded.
        duration_s: Elapsed seconds for the action.
        error: Error message on failure.
        note: Free-form note.
        env: Relevant environment variables (VLLM_*, CUDA_*, etc.).
    """
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "service": service,
        "success": success,
    }
    if model is not None:
        entry["model"] = model
    if args is not None:
        entry["args"] = args
    if port is not None:
        entry["port"] = port
    if gpu is not None:
        entry["gpu"] = gpu
    if interface is not None:
        entry["interface"] = interface
    if pid is not None:
        entry["pid"] = pid
    if duration_s is not None:
        entry["duration_s"] = round(duration_s, 2)
    if error is not None:
        entry["error"] = error
    if note is not None:
        entry["note"] = note
    if env is not None:
        entry["env"] = env
    else:
        captured = _capture_env()
        if captured:
            entry["env"] = captured

    try:
        with open(_ensure_dir() / "events.jsonl", "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Failed to write history event: %s", exc)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_events(
    *,
    service: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 50,
    include_older_than_days: int = 90,
) -> list[dict[str, Any]]:
    """Return recent events (newest first), optionally filtered.

    Args:
        service: Only events for this service.
        action: Only ``start`` / ``stop`` / ``restart``.
        limit: Maximum number of events returned.
        include_older_than_days: Drop events older than this many days.
    """
    path = EVENTS_FILE
    if not path.exists():
        return []

    cutoff = datetime.now(timezone.utc).timestamp() - (include_older_than_days * 86400)
    events: list[dict[str, Any]] = []

    try:
        with open(path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Age filter
                ts = ev.get("ts", "")
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt.timestamp() < cutoff:
                        continue
                except (ValueError, OSError):
                    pass

                # Column filters
                if service is not None and ev.get("service") != service:
                    continue
                if action is not None and ev.get("action") != action:
                    continue

                events.append(ev)
    except OSError:
        return []

    events.reverse()
    return events[:limit]


def history_stats() -> dict[str, Any]:
    """Quick stats over the full history file (all ages)."""
    path = EVENTS_FILE
    if not path.exists():
        return {"total": 0, "by_service": {}, "successes": 0, "failures": 0}

    stats: dict[str, Any] = {
        "total": 0,
        "successes": 0,
        "failures": 0,
        "by_service": {},
        "by_action": {},
    }

    try:
        with open(path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                stats["total"] += 1
                if ev.get("success"):
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1

                svc = ev.get("service", "unknown")
                act = ev.get("action", "unknown")

                by_svc = stats.setdefault("by_service", {})
                if svc not in by_svc:
                    by_svc[svc] = 0
                by_svc[svc] += 1

                by_act = stats.setdefault("by_action", {})
                if act not in by_act:
                    by_act[act] = 0
                by_act[act] += 1
    except OSError:
        pass

    return stats
