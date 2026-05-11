"""Service lifecycle management for the LLM stack."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .stack import StackConfig

logger = logging.getLogger(__name__)

PID_DIR = Path("/tmp/llm-stack-pids")


def _pid_file(name: str) -> Path:
    return PID_DIR / f"{name}.pid"


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ServiceResult:
    """Immutable result from a service operation."""

    success: bool
    name: str
    pid: Optional[int] = None
    message: str = ""
    error: Optional[str] = None


# ── Dependency ordering ─────────────────────────────────────────────────────

def _topological_sort(
    configs: dict[str, StackConfig],
    *,
    reverse: bool = False,
) -> list[str]:
    """Kahn's algorithm on the depends_on graph.

    reverse=False  → dependencies first  (start order)
    reverse=True   → dependents first    (stop order)
    """
    # Build adjacency + in-degree
    names = set(configs.keys())
    in_degree: dict[str, int] = {n: 0 for n in names}
    dependents: dict[str, list[str]] = {n: [] for n in names}

    for name, cfg in configs.items():
        for dep in cfg.depends_on:
            if dep in names:
                dependents[dep].append(name)
                in_degree[name] += 1

    # Seeds: nodes with no incoming edges
    queue = [n for n, d in in_degree.items() if d == 0]
    order: list[str] = []

    while queue:
        # Sort within the same level for deterministic output
        queue.sort()
        node = queue.pop(0)
        order.append(node)
        for dep_name in sorted(dependents[node]):
            in_degree[dep_name] -= 1
            if in_degree[dep_name] == 0:
                queue.append(dep_name)

    if len(order) != len(names):
        logger.warning("Cycle detected in service dependencies; falling back to config order")
        order = list(configs.keys())

    if reverse:
        order.reverse()

    return order


def start_order(configs: dict[str, StackConfig]) -> list[str]:
    """Return service names in dependency-first order (for starting)."""
    return _topological_sort(configs, reverse=False)


def stop_order(configs: dict[str, StackConfig]) -> list[str]:
    """Return service names in dependent-first order (for stopping)."""
    return _topological_sort(configs, reverse=True)


# ── GPU validation ───────────────────────────────────────────────────────────

def check_gpu_vram(gpu_indices: str, required_gb: float = 0) -> tuple[bool, str]:
    """Check nvidia-smi for free VRAM on requested GPUs.

    Args:
        gpu_indices: Comma-separated GPU indices (e.g. "0,1").
        required_gb: Minimum free VRAM in GiB per GPU.  0 = just verify GPUs exist.

    Returns:
        (ok, message)
    """
    try:
        idxs = [int(i.strip()) for i in gpu_indices.split(",") if i.strip()]
    except ValueError:
        return False, f"Invalid GPU indices: {gpu_indices}"

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,nounits,noheader",
            ],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"nvidia-smi unavailable: {exc}"

    gpus = {}
    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx = int(parts[0])
            total_mi = int(parts[1])
            used_mi = int(parts[2])
            gpus[idx] = {"total_mi": total_mi, "used_mi": used_mi, "free_mi": total_mi - used_mi}

    problems: list[str] = []
    for idx in idxs:
        if idx not in gpus:
            problems.append(f"GPU {idx} not found")
            continue
        free_gb = gpus[idx]["free_mi"] / 1024
        if required_gb > 0 and free_gb < required_gb:
            problems.append(
                f"GPU {idx}: {free_gb:.1f} GiB free < {required_gb:.1f} GiB required"
            )

    if problems:
        return False, "; ".join(problems)

    total_free = sum(gpus[i]["free_mi"] / 1024 for i in idxs if i in gpus)
    return True, f"OK — {len(idxs)} GPU(s), {total_free:.1f} GiB total free"


# ── Service Manager ─────────────────────────────────────────────────────────

class ServiceManager:
    """Start, stop, and monitor named services in the LLM stack."""

    def __init__(self, stack_config: dict[str, StackConfig]) -> None:
        self.configs = stack_config
        PID_DIR.mkdir(parents=True, exist_ok=True)

    # ── Command builders ──────────────────────────────────────────────────

    def _build_vllm_command(self, name: str, cfg: StackConfig, overrides: dict[str, Any]) -> str:
        """Build bash command string for vllm or reranker services."""
        model = overrides.get("model", cfg.model)
        port = overrides.get("port", cfg.port)
        host = overrides.get("host", cfg.host)
        gpus_raw = overrides.get("gpus", cfg.gpus)
        # Normalise: list[int] or str → comma-separated string
        if isinstance(gpus_raw, list):
            gpus = ",".join(str(g) for g in gpus_raw)
        else:
            gpus = str(gpus_raw)

        # Tensor-parallel size from GPU count
        gpu_count = len([g for g in gpus.split(",") if g.strip()])
        tp_flag = f"--tensor-parallel-size {gpu_count}" if gpu_count > 1 else ""

        # Context length override
        max_model_len = cfg.args.get("--max-model-len", 262144)
        long_ctx = ""
        if max_model_len > 262144:
            long_ctx = "export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"

        # Reasoning parser
        reasoning = cfg.args.get("--reasoning-parser", "")
        reasoning_flag = f"--reasoning-parser {reasoning}" if reasoning else ""

        # Extra args: serialize as --key value pairs
        extra_parts: list[str] = []
        for k, v in cfg.args.items():
            if k in (
                "--reasoning-parser",
                "--max-model-len",
                "--max-num-seqs",
                "--gpu-memory-utilization",
            ):
                continue  # handled above
            if isinstance(v, bool) and v:
                extra_parts.append(k)
            elif v is not None:
                extra_parts.append(f"{k} {v}")

        extra_args = " ".join(extra_parts)
        # Apply overrides for numeric/string args
        for k, v in overrides.items():
            if k.startswith("--") and k not in ("-"):
                if isinstance(v, bool) and v:
                    extra_args += f" {k}"
                else:
                    extra_args += f" {k} {v}"

        inner = (
            f"export PATH='/usr/local/cuda-12.9/bin:/usr/bin:/usr/local/bin:$PATH'\n"
            f"export LD_LIBRARY_PATH='/usr/local/cuda-12.9/lib64:${{LD_LIBRARY_PATH:-}}'\n"
            f"export CUDA_HOME='/usr/local/cuda-12.9'\n"
            f"export CC=/usr/bin/gcc\n"
            f"{long_ctx}\n"
            f"source '{cfg.venv}'\n"
            f"CUDA_VISIBLE_DEVICES={gpus} vllm serve '{model}' "
            f"--host {host} --port {port} "
            f"--download-dir /work/fast3/shared/hf-cache/hub/ "
            f"--max-model-len {max_model_len} "
            f"{extra_args} "
            f"{reasoning_flag} "
            f"{tp_flag}"
        )
        log = cfg.resolved_log_file
        return f"nohup bash -c '{inner}' > '{log}' 2>&1 & echo $!"

    def _build_litellm_command(self, cfg: StackConfig, overrides: dict[str, Any]) -> str:
        """Build bash command string for litellm service."""
        port = overrides.get("port", cfg.port)
        config_path = overrides.get("config", cfg.config or "")
        verbose = overrides.get("verbose", cfg.args.get("verbose", False))

        debug_env = "LITELLM_LOG=DEBUG" if verbose else ""
        debug_flags = "--detailed_debug" if verbose else ""

        inner = (
            f"source '{cfg.venv}'\n"
            f"PYTHONPATH='{cfg.tools_root}' "
            f"{debug_env} litellm --port {port} "
        )
        if config_path:
            inner += f"--config '{config_path}' "
        inner += f"{debug_flags}"

        log = cfg.resolved_log_file
        return f"nohup bash -c '{inner}' > '{log}' 2>&1 & echo $!"

    def _build_command(self, name: str, cfg: StackConfig, overrides: dict[str, Any]) -> str:
        if cfg.service_type in ("vllm", "reranker"):
            return self._build_vllm_command(name, cfg, overrides)
        elif cfg.service_type == "litellm":
            return self._build_litellm_command(cfg, overrides)
        else:
            raise ValueError(f"Unknown service type: {cfg.service_type}")

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, name: str, **overrides: Any) -> ServiceResult:
        """Start a service by name.

        GPU VRAM is validated before launch.  The process is launched via
        subprocess (nohup pattern from the original bash script).  Its PID
        is saved to /tmp/llm-stack-pids/{name}.pid.
        """
        if name not in self.configs:
            return ServiceResult(
                success=False,
                name=name,
                message=f"Unknown service: {name}",
                error=f"No StackConfig for '{name}'",
            )

        cfg = self.configs[name]

        # Skip if already running
        if self.is_running(name):
            existing_pid = self.get_pid(name)
            return ServiceResult(
                success=True,
                name=name,
                pid=existing_pid,
                message="Already running",
            )

        # GPU validation
        gpu_raw = overrides.get("gpus", cfg.gpus)
        if isinstance(gpu_raw, list):
            gpu_str = ",".join(str(g) for g in gpu_raw)
        else:
            gpu_str = str(gpu_raw)
        if cfg.service_type in ("vllm", "reranker"):
            required_gb = cfg.args.get("--gpu-memory-utilization", 0.9)
            # We don't enforce a hard minimum by default, just a sanity check
            ok, msg = check_gpu_vram(gpu_str, required_gb=0)
            if not ok:
                return ServiceResult(
                    success=False,
                    name=name,
                    message=msg,
                    error="GPU validation failed",
                )
            logger.info("GPU check passed: %s", msg)

        cmd = self._build_command(name, cfg, overrides)
        logger.info("Starting %s: %s", name, cmd[:120])

        try:
            # Run in a login shell so bash -c is interpreted correctly
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            stdout = result.stdout.strip()
            # The command echoes the PID via 'echo $!'
            pid = int(stdout.split("\n")[-1].strip()) if stdout else None

            if pid:
                # Persist PID
                pf = _pid_file(name)
                pf.write_text(str(pid))
                logger.info("Started %s with PID %d", name, pid)
                return ServiceResult(
                    success=True,
                    name=name,
                    pid=pid,
                    message=f"Started with PID {pid}",
                )
            else:
                return ServiceResult(
                    success=False,
                    name=name,
                    message="Could not capture PID",
                    error=result.stderr.strip() or "no PID output",
                )
        except subprocess.TimeoutExpired:
            return ServiceResult(
                success=False,
                name=name,
                message="Launch timed out",
                error="subprocess timed out after 30s",
            )
        except (ValueError, subprocess.CalledProcessError) as exc:
            return ServiceResult(
                success=False,
                name=name,
                message=str(exc),
                error=f"Launch failed: {exc}",
            )

    def stop(self, name: str, timeout: int = 10) -> ServiceResult:
        """Stop a service gracefully (SIGTERM, then SIGKILL after timeout).

        Cleans up the PID file afterward.
        """
        if name not in self.configs:
            return ServiceResult(
                success=False,
                name=name,
                message=f"Unknown service: {name}",
                error=f"No StackConfig for '{name}'",
            )

        pid = self.get_pid(name)
        if pid is None:
            # Clean up stale PID file if present
            pf = _pid_file(name)
            if pf.exists():
                pf.unlink()
            return ServiceResult(
                success=True,
                name=name,
                message="Not running",
            )

        # Send SIGTERM
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to %s (PID %d)", name, pid)
        except ProcessLookupError:
            # Already gone
            pf = _pid_file(name)
            if pf.exists():
                pf.unlink()
            return ServiceResult(
                success=True,
                name=name,
                pid=pid,
                message="Process already exited",
            )
        except PermissionError:
            return ServiceResult(
                success=False,
                name=name,
                pid=pid,
                message="Permission denied",
                error="Cannot kill PID",
            )

        # Wait up to timeout seconds
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)  # Check if alive
                time.sleep(0.2)
            except ProcessLookupError:
                # Process is gone
                pf = _pid_file(name)
                if pf.exists():
                    pf.unlink()
                logger.info("Stopped %s (PID %d) gracefully", name, pid)
                return ServiceResult(
                    success=True,
                    name=name,
                    pid=pid,
                    message="Stopped gracefully",
                )

        # Force kill
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warning("SIGKILL sent to %s (PID %d)", name, pid)
        except ProcessLookupError:
            pass

        pf = _pid_file(name)
        if pf.exists():
            pf.unlink()

        return ServiceResult(
            success=True,
            name=name,
            pid=pid,
            message=f"Force killed after {timeout}s timeout",
        )

    def restart(
        self, name: str, timeout: int = 10, **overrides: Any
    ) -> ServiceResult:
        """Stop then start a service."""
        stop_result = self.stop(name, timeout=timeout)
        if not stop_result.success:
            # Try to start anyway in case it wasn't running
            logger.warning("Stop failed for %s, attempting start: %s", name, stop_result.error)

        return self.start(name, **overrides)

    # ── Helpers ───────────────────────────────────────────────────────────

    def is_running(self, name: str) -> bool:
        """Check if a service is running via PID file, /proc, and port."""
        if name not in self.configs:
            return False

        pid = self.get_pid(name)
        if pid is not None:
            return self._pid_alive(pid)

        # Fallback: check port
        cfg = self.configs[name]
        return self._port_in_use(cfg.port)

    def get_pid(self, name: str) -> Optional[int]:
        """Get PID from file or process matching."""
        if name not in self.configs:
            return None

        # 1. PID file
        pf = _pid_file(name)
        if pf.exists():
            try:
                pid = int(pf.read_text().strip())
                if self._pid_alive(pid):
                    return pid
            except (ValueError, ProcessLookupError):
                pass

        # 2. Port-based fallback (pgrep --port)
        cfg = self.configs[name]
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", f"--port {cfg.port}"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if out.strip():
                return int(out.strip().split("\n")[0])
        except (subprocess.CalledProcessError, ValueError):
            pass

        return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Check if a PID exists and is alive via /proc and kill(0)."""
        # Fast path: /proc/<pid>/stat
        if not Path(f"/proc/{pid}").is_dir():
            return False
        # Confirm with kill(0) — doesn't send a signal, just checks
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    @staticmethod
    def _port_in_use(port: int) -> bool:
        """Check if a TCP port is listening."""
        try:
            out = subprocess.check_output(
                ["ss", "-tlnp"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return f":{port} " in out or f":{port}\n" in out
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Fallback: try binding
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            sock.close()
            return False
        except OSError:
            sock.close()
            return True
