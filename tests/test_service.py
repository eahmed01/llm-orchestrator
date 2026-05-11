"""Tests for llm_orchestrator.service module."""

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_orchestrator.service import (
    ServiceManager,
    ServiceResult,
    start_order,
    stop_order,
    check_gpu_vram,
)
from llm_orchestrator.stack import StackConfig, default_stack_configs


@pytest.fixture
def mgr():
    configs = default_stack_configs()
    return ServiceManager(configs)


class TestDependencyOrdering:
    """Tests for topological sort of service dependencies."""

    def test_start_order_deps_first(self):
        configs = default_stack_configs()
        order = start_order(configs)
        # litellm depends on vllm, so vllm should come before litellm
        vllm_idx = order.index("vllm")
        litellm_idx = order.index("litellm")
        assert vllm_idx < litellm_idx

    def test_stop_order_dependents_first(self):
        configs = default_stack_configs()
        order = stop_order(configs)
        # litellm depends on vllm, so litellm should be stopped before vllm
        vllm_idx = order.index("vllm")
        litellm_idx = order.index("litellm")
        assert litellm_idx < vllm_idx

    def test_start_order_all_services(self):
        configs = default_stack_configs()
        order = start_order(configs)
        assert set(order) == {"reranker", "vllm", "litellm"}

    def test_stop_order_all_services(self):
        configs = default_stack_configs()
        order = stop_order(configs)
        assert set(order) == {"reranker", "vllm", "litellm"}

    def test_single_service(self):
        configs = {
            "x": StackConfig(name="x", model="m", port=1),
        }
        assert start_order(configs) == ["x"]
        assert stop_order(configs) == ["x"]

    def test_no_deps(self):
        configs = {
            "a": StackConfig(name="a", model="m", port=1),
            "b": StackConfig(name="b", model="m", port=2),
        }
        # Alphabetical for deterministic output
        assert start_order(configs) == ["a", "b"]


class TestCheckGpuVram:
    """Tests for GPU VRAM validation."""

    def test_gpu_not_found(self):
        ok, msg = check_gpu_vram("99", 0)
        assert ok is False
        assert "not found" in msg

    def test_invalid_indices(self):
        ok, msg = check_gpu_vram("abc", 0)
        assert ok is False

    def test_no_nvidia_smi(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError()):
            ok, msg = check_gpu_vram("0", 0)
            assert ok is False
            assert "unavailable" in msg

    def test_gpu_available(self):
        fake_output = "0,97887,1000\n1,97887,1000\n"
        with patch("subprocess.check_output", return_value=fake_output):
            ok, msg = check_gpu_vram("0,1", 0)
            assert ok is True
            assert "OK" in msg

    def test_multiple_gpus(self):
        fake_output = "0,97887,5000\n1,97887,5000\n"
        with patch("subprocess.check_output", return_value=fake_output):
            ok, msg = check_gpu_vram("0", 0)
            assert ok is True


class TestServiceResult:
    """Tests for ServiceResult dataclass."""

    def test_success_result(self):
        r = ServiceResult(success=True, name="vllm", pid=1234, message="Started")
        assert r.success is True
        assert r.error is None

    def test_failure_result(self):
        r = ServiceResult(success=False, name="vllm", error="GPU not found")
        assert r.success is False
        assert r.error == "GPU not found"

    def test_result_is_frozen(self):
        r = ServiceResult(success=True, name="x")
        with pytest.raises(Exception):
            r.success = False  # type: ignore


class TestServiceManager:
    """Tests for ServiceManager without launching real processes."""

    @pytest.fixture
    def mgr(self):
        configs = default_stack_configs()
        return ServiceManager(configs)

    def test_init_creates_pid_dir(self):
        configs = default_stack_configs()
        ServiceManager(configs)
        assert Path("/tmp/llm-stack-pids").exists()

    def test_start_unknown_service(self, mgr):
        result = mgr.start("nonexistent")
        assert result.success is False
        assert "Unknown service" in result.message

    def test_stop_unknown_service(self, mgr):
        result = mgr.stop("nonexistent")
        assert result.success is False

    def test_get_pid_unknown(self, mgr):
        assert mgr.get_pid("nonexistent") is None

    def test_is_running_unknown(self, mgr):
        assert mgr.is_running("nonexistent") is False

    def test_stop_not_running(self, mgr):
        with patch.object(mgr, "get_pid", return_value=None):
            result = mgr.stop("litellm")
            assert result.success is True
            assert "Not running" in result.message

    def test_pid_alive_true(self):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("os.kill"):
                assert ServiceManager._pid_alive(12345) is True

    def test_pid_alive_false(self):
        with patch("pathlib.Path.is_dir", return_value=False):
            assert ServiceManager._pid_alive(12345) is False

    def test_port_in_use_false(self):
        """Test port_in_use when port is free."""
        with patch("subprocess.check_output", side_effect=FileNotFoundError()):
            with patch("socket.socket") as mock_sock:
                mock_instance = MagicMock()
                mock_sock.return_value = mock_instance
                # bind succeeds = port free
                result = ServiceManager._port_in_use(19999)
                assert result is False

    def test_port_in_use_true(self):
        """Test port_in_use when port is bound."""
        with patch("subprocess.check_output", side_effect=FileNotFoundError()):
            with patch("socket.socket") as mock_sock:
                mock_instance = MagicMock()
                mock_sock.return_value = mock_instance
                # bind fails = port in use
                mock_instance.bind.side_effect = OSError("Address in use")
                result = ServiceManager._port_in_use(8000)
                assert result is True

    def test_build_vllm_command_has_model(self, mgr):
        cfg = mgr.configs["vllm"]
        overrides = {"gpus": [0, 1]}
        cmd = mgr._build_command("vllm", cfg, overrides)
        assert "Qwen/Qwen3.6-27B-FP8" in cmd
        assert "--port 7999" in cmd

    def test_build_litellm_command(self, mgr):
        cfg = mgr.configs["litellm"]
        cmd = mgr._build_command("litellm", cfg, {})
        assert "litellm" in cmd
        assert "--port 8000" in cmd


class TestServiceManagerRestart:
    """Tests for restart with mocked start/stop."""

    def test_restart_calls_stop_then_start(self):
        configs = default_stack_configs()
        mgr = ServiceManager(configs)

        with patch.object(mgr, "stop", return_value=ServiceResult(success=True, name="vllm", message="Stopped")) as mock_stop:
            with patch.object(mgr, "start", return_value=ServiceResult(success=True, name="vllm", pid=999, message="Started")) as mock_start:
                result = mgr.restart("vllm")
                mock_stop.assert_called_once()
                mock_start.assert_called_once()
                assert result.success is True

    def test_restart_on_stop_failure_still_starts(self):
        configs = default_stack_configs()
        mgr = ServiceManager(configs)

        with patch.object(mgr, "stop", return_value=ServiceResult(success=False, name="vllm", error="timeout")):
            with patch.object(mgr, "start", return_value=ServiceResult(success=True, name="vllm", pid=999, message="Started")):
                result = mgr.restart("vllm")
                assert result.success is True

from llm_orchestrator.service import _topological_sort, ServiceManager


class TestServiceManagerStart:
    """Tests for ServiceManager.start()."""

    def test_start_already_running(self, mgr):
        """Test start when service is already running."""
        with patch.object(mgr, "is_running", return_value=True):
            with patch.object(mgr, "get_pid", return_value=12345):
                result = mgr.start("vllm")
        assert result.success is True
        assert "Already running" in result.message

    def test_start_gpu_validation_failed(self, mgr):
        """Test start when GPU validation fails."""
        with patch.object(mgr, "is_running", return_value=False):
            with patch("llm_orchestrator.service.check_gpu_vram", return_value=(False, "GPU not found")):
                result = mgr.start("vllm")
        assert result.success is False
        assert "GPU validation failed" in result.error

    def test_start_timeout(self, mgr):
        """Test start when subprocess times out."""
        import subprocess as sp
        with patch.object(mgr, "is_running", return_value=False):
            with patch("llm_orchestrator.service.check_gpu_vram", return_value=(True, "OK")):
                with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="test", timeout=30)):
                    result = mgr.start("vllm")
        assert result.success is False
        assert "timed out" in result.message

    def test_start_pid_capture(self, mgr):
        """Test start successfully captures PID."""
        import subprocess as sp
        mock_result = sp.CompletedProcess(args="test", returncode=0, stdout="12345\n", stderr="")
        with patch.object(mgr, "is_running", return_value=False):
            with patch("llm_orchestrator.service.check_gpu_vram", return_value=(True, "OK")):
                with patch("subprocess.run", return_value=mock_result):
                    with patch("pathlib.Path.write_text"):
                        result = mgr.start("vllm")
        assert result.success is True
        assert result.pid == 12345


class TestServiceManagerStop:
    """Tests for ServiceManager.stop()."""

    def test_stop_permission_denied(self, mgr):
        """Test stop when permission is denied."""
        with patch.object(mgr, "get_pid", return_value=12345):
            with patch("os.kill", side_effect=PermissionError()):
                result = mgr.stop("vllm")
        assert result.success is False
        assert "Permission denied" in result.message

    def test_stop_already_gone(self, mgr):
        """Test stop when process already exited."""
        with patch.object(mgr, "get_pid", return_value=12345):
            with patch("os.kill", side_effect=ProcessLookupError()):
                result = mgr.stop("vllm")
        assert result.success is True
        assert "already exited" in result.message

    def test_stop_graceful(self, mgr):
        """Test stop with graceful shutdown."""
        kill_count = [0]
        def mock_kill(pid, sig):
            kill_count[0] += 1
            if kill_count[0] > 1:
                raise ProcessLookupError()
        with patch.object(mgr, "get_pid", return_value=12345):
            with patch("os.kill", side_effect=mock_kill):
                result = mgr.stop("vllm")
        assert result.success is True
        assert "Stopped gracefully" in result.message

    def test_stop_force_kill(self, mgr):
        """Test stop with force kill after timeout."""
        import time as _time
        with patch.object(mgr, "get_pid", return_value=12345):
            with patch("os.kill"):
                with patch("time.monotonic", side_effect=[0, 0, 20, 20]):  # deadline passes
                    with patch("time.sleep"):
                        result = mgr.stop("vllm", timeout=10)
        assert result.success is True
        assert "Force killed" in result.message


class TestServiceManagerIsRunning:
    """Tests for ServiceManager.is_running()."""

    def test_is_running_via_pid(self, mgr):
        """Test is_running when PID file exists and process alive."""
        with patch.object(mgr, "get_pid", return_value=12345):
            with patch.object(ServiceManager, "_pid_alive", return_value=True):
                assert mgr.is_running("vllm") is True

    def test_is_running_via_port(self, mgr):
        """Test is_running when PID not found but port in use."""
        with patch.object(mgr, "get_pid", return_value=None):
            with patch.object(ServiceManager, "_port_in_use", return_value=True):
                assert mgr.is_running("vllm") is True

    def test_is_running_not_running(self, mgr):
        """Test is_running when PID not found and port free."""
        with patch.object(mgr, "get_pid", return_value=None):
            with patch.object(ServiceManager, "_port_in_use", return_value=False):
                assert mgr.is_running("vllm") is False


class TestServiceManagerGetPid:
    """Tests for ServiceManager.get_pid()."""

    def test_get_pid_from_file(self, mgr):
        """Test get_pid reads from PID file."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value="12345"):
                with patch.object(ServiceManager, "_pid_alive", return_value=True):
                    pid = mgr.get_pid("vllm")
        assert pid == 12345

    def test_get_pid_from_port(self, mgr):
        """Test get_pid falls back to pgrep."""
        import subprocess as sp
        with patch("pathlib.Path.exists", return_value=False):
            with patch("subprocess.check_output", return_value="12345\n"):
                pid = mgr.get_pid("vllm")
        assert pid == 12345


class TestTopologicalSort:
    """Tests for _topological_sort internal function."""

    def test_cycle_detected(self):
        """Test that cycles fall back to config order."""
        from llm_orchestrator.service import _topological_sort
        cfg_a = StackConfig(name="a", model="m", port=1, depends_on=["b"])
        cfg_b = StackConfig(name="b", model="m", port=2, depends_on=["a"])
        configs = {"a": cfg_a, "b": cfg_b}
        order = _topological_sort(configs)
        assert len(order) == 2
