#!/usr/bin/env python3
"""Smoke-test for stack.py and service.py imports and basic logic."""
import sys
sys.path.insert(0, "/home/xeio/dev/controls-agents/llm-orchestrator")

from llm_orchestrator.stack import StackConfig
from llm_orchestrator.service import (
    ServiceResult,
    ServiceManager,
    start_order,
    stop_order,
    check_gpu_vram,
)

# --- ServiceResult is frozen ---
try:
    r = ServiceResult(success=True, name="vllm", pid=123, message="ok")
    r.success = False  # Should raise
    print("FAIL: ServiceResult is not frozen")
except Exception:
    print("OK: ServiceResult is frozen")

# --- StackConfig defaults ---
cfg = StackConfig(service_type="vllm", model="test", port=7999)
assert cfg.resolved_log_file == "/work/fast3/xeio/logs/vllm.log"
print(f"OK: default log_file = {cfg.resolved_log_file}")

# --- Dependency ordering ---
configs = {
    "reranker": StackConfig(service_type="reranker", model="R", port=7998),
    "vllm": StackConfig(service_type="vllm", model="V", port=7999, depends_on=["reranker"]),
    "litellm": StackConfig(service_type="litellm", model="L", port=8000, depends_on=["vllm"]),
}

so = start_order(configs)
print(f"OK: start_order = {so}")
assert so[0] == "reranker", f"reranker should start first, got {so}"
assert so[-1] == "litellm", f"litellm should start last, got {so}"

sto = stop_order(configs)
print(f"OK: stop_order = {sto}")
assert sto[0] == "litellm", f"litellm should stop first, got {sto}"
assert sto[-1] == "reranker", f"reranker should stop last, got {sto}"

# --- ServiceManager instantiation ---
mgr = ServiceManager(configs)
print("OK: ServiceManager created")

# --- get_pid for unknown ---
assert mgr.get_pid("nonexistent") is None
print("OK: get_pid returns None for unknown service")

# --- is_running for unknown ---
assert not mgr.is_running("nonexistent")
print("OK: is_running returns False for unknown service")

# --- stop unknown ---
r = mgr.stop("nonexistent")
assert not r.success
print(f"OK: stop unknown returns success=False")

# --- start unknown ---
r = mgr.start("nonexistent")
assert not r.success
print(f"OK: start unknown returns success=False")

# --- GPU check (best-effort, may fail if no nvidia-smi) ---
ok, msg = check_gpu_vram("bad")
assert not ok
print(f"OK: check_gpu_vram rejects bad input: {msg}")

print("\nAll smoke tests passed!")
