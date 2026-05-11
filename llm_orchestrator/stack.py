"""Stack detection and snapshot module for the LLM stack orchestrator.

Provides data models and detection logic for identifying running LLM services
(vLLM, LiteLLM, rerankers), their GPU assignments, and overall system state.
"""

import logging
import os
import re
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class StackConfig(BaseModel):
    """Configuration for a named service in the LLM stack."""

    name: str = Field(..., description="Service name (e.g., 'vllm', 'litellm')")
    model: str = Field(..., description="HuggingFace model ID or empty string")
    port: int = Field(..., description="HTTP port the service listens on")
    gpus: Optional[list[int]] = Field(
        default=None, description="GPU indices via CUDA_VISIBLE_DEVICES"
    )
    host: str = Field(default="0.0.0.0", description="Bind address")
    args: dict[str, Any] = Field(default_factory=dict, description="CLI arguments")
    log_file: Optional[str] = Field(None, description="Path to log file")
    pid_file: Optional[str] = Field(None, description="Path to PID file")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Extra env vars")
    venv_path: Optional[str] = Field(None, description="Path to Python venv")
    depends_on: list[str] = Field(
        default_factory=list, description="Service names this depends on"
    )
    # Extra fields for service.py compatibility
    config: Optional[str] = Field(None, description="Config file path (litellm)")
    tools_root: Optional[str] = Field(None, description="PYTHONPATH for custom tools")

    # Keep service_type as an alias of name for backward compat
    @property
    def service_type(self) -> str:
        """Alias of name for service.py compatibility."""
        return self.name

    @property
    def venv(self) -> Optional[str]:
        """Alias of venv_path for service.py compatibility."""
        return self.venv_path

    @property
    def resolved_log_file(self) -> str:
        """Return log_file or a sensible default."""
        if self.log_file:
            return self.log_file
        log_dir = Path("/work/fast3/xeio/logs")
        if self.name == "reranker":
            return str(log_dir / "vllm.reranker.log")
        elif self.name == "vllm":
            return str(log_dir / "vllm.log")
        elif self.name == "litellm":
            return str(log_dir / "litellm.log")
        return str(log_dir / f"{self.name}.log")

    def gpus_str(self) -> str:
        """Return GPU indices as a comma-separated string."""
        if self.gpus is None:
            return "0"  # default
        return ",".join(str(g) for g in self.gpus)


@dataclass(frozen=True)
class StackServiceInfo:
    """Detected information about a single running service."""

    name: str
    model: str
    port: int
    pid: int
    gpus: list[int]
    host: str
    status: Literal["running", "stopped"]
    log_file: Optional[str] = None
    started_at: Optional[datetime] = None


@dataclass(frozen=True)
class StackSnapshot:
    """Immutable snapshot of the entire LLM stack state at a point in time."""

    services: dict[str, StackServiceInfo]
    gpus: list[dict[str, Any]]
    timestamp: datetime


# ---------------------------------------------------------------------------
# Known service definitions (matching the llm-stack startup script)
# ---------------------------------------------------------------------------


def default_stack_configs() -> dict[str, StackConfig]:
    """Return the standard set of stack service configurations.

    These mirror the services defined in the llm-stack startup script.
    """
    LAB_VENV = "/work/fast3/user/xeio/lab6.vllm"
    EPS_VENV = "/home/xeio/dev/epsilon/0"
    TOOLS_ROOT = "/home/xeio/dev/controls-agents/litellm_central_tools"
    LITELLM_CONFIG = "/home/xeio/dev/epsilon/0/controls.trading/bin/litellm_config.yaml"
    LOG_DIR = "/work/fast3/xeio/logs"

    return {
        "reranker": StackConfig(
            name="reranker",
            model="Qwen/Qwen3-Reranker-0.6B",
            port=7998,
            gpus=[1],
            host="0.0.0.0",
            args={
                "--enforce-eager": True,
                "--max-model-len": 1024,
                "--max-num-seqs": 4,
                "--max-num-batched-tokens": 1024,
                "--gpu-memory-utilization": 0.04,
            },
            log_file=f"{LOG_DIR}/vllm.reranker.log",
            venv_path=LAB_VENV,
        ),
        "vllm": StackConfig(
            name="vllm",
            model="Qwen/Qwen3.6-27B-FP8",
            port=7999,
            gpus=None,  # configurable at runtime (defaults to [0])
            host="0.0.0.0",
            args={
                "--max-model-len": 262144,
                "--max-num-seqs": 848,
                "--enable-auto-tool-choice": True,
                "--tool-call-parser": "qwen3_coder",
                "--enable-reasoning": True,
                "--reasoning-parser": "qwen3",
            },
            log_file=f"{LOG_DIR}/vllm.log",
            venv_path=LAB_VENV,
        ),
        "litellm": StackConfig(
            name="litellm",
            model="",
            port=8000,
            gpus=None,
            host="0.0.0.0",
            args={},
            depends_on=["vllm"],
            log_file=f"{LOG_DIR}/litellm.log",
            venv_path=EPS_VENV,
            config=LITELLM_CONFIG,
            tools_root=TOOLS_ROOT,
        ),
    }


# ---------------------------------------------------------------------------
# StackDetector
# ---------------------------------------------------------------------------

_PID_DIR = Path("/tmp/llm-stack-pids")

# Patterns to match known service processes inside `ps aux` command lines
_SERVICE_PATTERNS = [
    # vLLM serve processes
    (
        "vllm",
        re.compile(
            r"\bvllm\s+serve\b.*?--port\s+(\d+).*?(\S+)", re.DOTALL
        ),
    ),
    # Alternative: uvicorn running vllm.entrypoints
    (
        "vllm",
        re.compile(
            r"vllm\.entrypoints.*?--port\s+(\d+).*?(\S+)", re.DOTALL
        ),
    ),
    # LiteLLM processes
    (
        "litellm",
        re.compile(
            r"\blitellm\b.*?--port\s+(\d+)", re.DOTALL
        ),
    ),
    # uvicorn running litellm
    (
        "litellm",
        re.compile(
            r"litellm.*?--port\s+(\d+)", re.DOTALL
        ),
    ),
    # Reranker (vLLM serve with small model on specific port)
    (
        "reranker",
        re.compile(
            r"\bvllm\s+serve\b.*?--port\s+(\d+).*?(Qwen.*?Reranker.*?)(?=\s|$)",
            re.DOTALL,
        ),
    ),
]


class StackDetector:
    """Detect and snapshot the current state of the LLM service stack.

    All methods are static so the class can be used without instantiation.
    """

    @staticmethod
    def detect_running_services() -> dict[str, StackServiceInfo]:
        """Scan running processes to discover active LLM services.

        Combines two sources:
        1. ``ps aux`` parsing for vllm / litellm command lines.
        2. PID files in ``/tmp/llm-stack-pids/`` as fallback.

        Returns:
            Dictionary mapping service name to StackServiceInfo.
        """
        services: dict[str, StackServiceInfo] = {}

        # --- Phase 1: parse `ps aux` -----------------------------------
        try:
            output = subprocess.check_output(
                ["ps", "aux"],
                text=True,
                timeout=10,
            )
        except Exception as e:
            logger.warning("Failed to run ps aux: %s", e)
            output = ""

        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 11:
                continue

            try:
                pid = int(parts[1])
            except (ValueError, IndexError):
                continue

            cmdline = " ".join(parts[11:])

            for svc_name, pattern in _SERVICE_PATTERNS:
                m = pattern.search(cmdline)
                if not m:
                    continue

                try:
                    port = int(m.group(1))
                except (ValueError, IndexError):
                    continue

                model = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                gpus = StackDetector._read_proc_env_gpus(pid)
                log_file = StackDetector._guess_log_file(svc_name, pid)

                services[svc_name] = StackServiceInfo(
                    name=svc_name,
                    model=model,
                    port=port,
                    pid=pid,
                    gpus=gpus,
                    host="0.0.0.0",
                    status="running",
                    log_file=log_file,
                )

        # --- Phase 2: check PID files as fallback ----------------------
        if _PID_DIR.is_dir():
            for pid_file in _PID_DIR.glob("*.pid"):
                svc_name = pid_file.stem
                if svc_name in services:
                    continue  # already found via ps

                try:
                    pid_text = pid_file.read_text().strip()
                    pid = int(pid_text)
                except (ValueError, OSError):
                    continue

                # Verify the PID is still alive
                if not Path(f"/proc/{pid}").is_dir():
                    services[svc_name] = StackServiceInfo(
                        name=svc_name,
                        model="",
                        port=0,
                        pid=pid,
                        gpus=[],
                        host="0.0.0.0",
                        status="stopped",
                    )
                    continue

                # Try to get port from known defaults
                defaults = default_stack_configs()
                known = defaults.get(svc_name)
                port = known.port if known else 0
                model = known.model if known else ""
                gpus = StackDetector._read_proc_env_gpus(pid)

                services[svc_name] = StackServiceInfo(
                    name=svc_name,
                    model=model,
                    port=port,
                    pid=pid,
                    gpus=gpus,
                    host=known.host if known else "0.0.0.0",
                    status="running",
                    log_file=known.log_file if known else None,
                )

        return services

    @staticmethod
    def verify_endpoint(host: str, port: int, timeout: float = 3) -> bool:
        """Check if a service's OpenAI-compatible /v1/models endpoint responds.

        Args:
            host: Hostname or IP address.
            port: TCP port.
            timeout: HTTP request timeout in seconds.

        Returns:
            True if the endpoint returns HTTP 200.
        """
        # Normalise host for the request URL
        request_host = host if host != "0.0.0.0" else "127.0.0.1"
        url = f"http://{request_host}:{port}/v1/models"

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
            TimeoutError,
        ) as exc:
            logger.debug("Endpoint %s:%d not reachable: %s", host, port, exc)
            return False

    @staticmethod
    def get_gpu_usage() -> list[dict[str, Any]]:
        """Query nvidia-smi for per-GPU memory, utilization, and PCIe link status.

        Returns:
            List of dicts with keys: index, name, memory_used_mb,
            memory_total_mb, utilization_pct, temperature_c,
            pcie_link_gen, pcie_link_gen_max, pcie_link_width, pcie_link_width_max.
        Empty list if nvidia-smi is unavailable.
        """
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu="
                    "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,"
                    "pcie.link.gen.current,pcie.link.gen.max,"
                    "pcie.link.width.current,pcie.link.width.max",
                    "--format=csv,nounits,noheader",
                ],
                text=True,
                timeout=10,
            )
        except Exception as e:
            logger.warning("Failed to query GPU usage: %s", e)
            return []

        gpus: list[dict[str, Any]] = []
        for line in output.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                gpu: dict[str, Any] = {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                    "temperature_c": int(parts[5]),
                }
                if len(parts) >= 10:
                    gpu["pcie_link_gen"] = int(parts[6])
                    gpu["pcie_link_gen_max"] = int(parts[7])
                    gpu["pcie_link_width"] = int(parts[8])
                    gpu["pcie_link_width_max"] = int(parts[9])
                gpus.append(gpu)
            except (ValueError, IndexError):
                continue

        return gpus

    @staticmethod
    def capture_snapshot() -> StackSnapshot:
        """Capture a complete, frozen snapshot of the stack's current state.

        Combines service detection, GPU usage, and an endpoint verification
        pass into a single StackSnapshot.

        Returns:
            A StackSnapshot with the current state.
        """
        services = StackDetector.detect_running_services()
        gpus = StackDetector.get_gpu_usage()
        timestamp = datetime.now(timezone.utc)

        # Verify endpoints for running services (non-blocking best-effort)
        verified_services: dict[str, StackServiceInfo] = {}
        for name, info in services.items():
            if info.status == "running":
                reachable = StackDetector.verify_endpoint(info.host, info.port)
                logger.info(
                    "Endpoint check %s (%s:%d): %s",
                    name,
                    info.host,
                    info.port,
                    "reachable" if reachable else "unreachable",
                )
            verified_services[name] = info

        return StackSnapshot(
            services=verified_services,
            gpus=gpus,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_proc_env_gpus(pid: int) -> list[int]:
        """Read CUDA_VISIBLE_DEVICES from /proc/PID/environ.

        Returns an empty list if the file is unreadable or the env var
        is not set.
        """
        environ_path = Path(f"/proc/{pid}/environ")
        if not environ_path.is_file():
            return []
        try:
            raw = environ_path.read_bytes()
            # environ is \0-separated
            for token in raw.split(b"\x00"):
                if token.startswith(b"CUDA_VISIBLE_DEVICES="):
                    val = token.split(b"=", 1)[1].decode()
                    return [
                        int(g.strip())
                        for g in val.split(",")
                        if g.strip().isdigit()
                    ]
        except (OSError, UnicodeDecodeError, ValueError) as e:
            logger.debug("Could not read /proc/%d/environ: %s", pid, e)
        return []

    @staticmethod
    def _guess_log_file(svc_name: str, pid: int) -> Optional[str]:
        """Try to find a log file for a service based on common conventions.

        Checks /tmp/llm-stack-logs/<svc_name>.log first, then falls back
        to nothing.
        """
        log_path = Path(f"/tmp/llm-stack-logs/{svc_name}.log")
        return str(log_path) if log_path.is_file() else None
