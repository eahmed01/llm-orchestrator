"""Tests for llm_orchestrator.environment module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from llm_orchestrator.environment import (
    EnvironmentDetector,
    ask_gpu_preference,
    ask_port_preference,
    ask_interface_preference,
)


class TestDetectGpus:
    """Tests for EnvironmentDetector.detect_gpus()."""

    def test_detect_gpus_success(self):
        fake_output = "0, NVIDIA RTX PRO 6000, 97887, 5000\n1, NVIDIA RTX PRO 6000, 97887, 3000\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = EnvironmentDetector.detect_gpus()
        assert len(gpus) == 2
        assert gpus[0]["index"] == 0
        assert gpus[0]["total_memory_gb"] == 95
        assert gpus[1]["index"] == 1

    def test_detect_gpus_single(self):
        fake_output = "0, NVIDIA A100, 81920, 1024\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = EnvironmentDetector.detect_gpus()
        assert len(gpus) == 1
        assert gpus[0]["total_memory_gb"] == 80

    def test_detect_gpus_no_nvidia(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError()):
            gpus = EnvironmentDetector.detect_gpus()
        assert gpus == []

    def test_detect_gpus_empty_output(self):
        with patch("subprocess.check_output", return_value=""):
            gpus = EnvironmentDetector.detect_gpus()
        assert gpus == []


class TestDetectInterfaces:
    """Tests for EnvironmentDetector.detect_interfaces()."""

    def test_detect_interfaces_always_has_loopback(self):
        interfaces = EnvironmentDetector.detect_interfaces()
        ips = [i["ip"] for i in interfaces]
        assert "127.0.0.1" in ips
        assert "0.0.0.0" in ips

    def test_detect_interfaces_has_types(self):
        interfaces = EnvironmentDetector.detect_interfaces()
        types = {i["type"] for i in interfaces}
        assert "loopback" in types
        assert "all_networks" in types

    def test_detect_interfaces_no_extra_when_socket_fails(self):
        with patch("socket.gethostname", side_effect=Exception("fail")):
            interfaces = EnvironmentDetector.detect_interfaces()
        # Still has defaults
        assert len(interfaces) >= 2


class TestIsPortInUse:
    """Tests for EnvironmentDetector.is_port_in_use()."""

    def test_port_in_use(self):
        with patch("subprocess.check_output", return_value="12345\n"):
            pid = EnvironmentDetector.is_port_in_use(8080)
        assert pid == 12345

    def test_port_free(self):
        with patch("subprocess.check_output", return_value=""):
            pid = EnvironmentDetector.is_port_in_use(9999)
        assert pid is None

    def test_port_free_lsof_not_available(self):
        import subprocess
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "lsof")):
            pid = EnvironmentDetector.is_port_in_use(9999)
        assert pid is None


class TestGetRunningServices:
    """Tests for EnvironmentDetector.get_running_services()."""

    def test_finds_vllm_process(self):
        fake_ps = "user 12345 0.0 0.0 ? Ss 00:00:00 python -m vllm serve --port 7999 /model"
        with patch("subprocess.check_output", return_value=fake_ps):
            services = EnvironmentDetector.get_running_services()
        assert 7999 in services
        assert services[7999]["pid"] == 12345

    def test_no_vllm_process(self):
        fake_ps = "user 12345 0.0 0.0 ? Ss 00:00:00 python some_script.py"
        with patch("subprocess.check_output", return_value=fake_ps):
            services = EnvironmentDetector.get_running_services()
        assert services == {}

    def test_ps_aux_fails(self):
        with patch("subprocess.check_output", side_effect=Exception("fail")):
            services = EnvironmentDetector.get_running_services()
        assert services == {}


class TestAskGpuPreference:
    """Tests for ask_gpu_preference interactive prompts."""

    def test_ask_gpu_valid_selection(self):
        gpus = [
            {"index": 0, "name": "GPU0", "total_memory_gb": 24, "used_memory_mb": 1024},
            {"index": 1, "name": "GPU1", "total_memory_gb": 95, "used_memory_mb": 5120},
        ]
        with patch("builtins.input", return_value="1"):
            result = ask_gpu_preference(gpus)
        assert result == 1

    def test_ask_gpu_auto_select(self):
        gpus = [
            {"index": 0, "name": "GPU0", "total_memory_gb": 24, "used_memory_mb": 1024},
        ]
        with patch("builtins.input", return_value=""):
            result = ask_gpu_preference(gpus)
        assert result is None

    def test_ask_gpu_no_gpus(self):
        result = ask_gpu_preference([])
        assert result is None

    def test_ask_gpu_invalid_then_valid(self):
        gpus = [
            {"index": 0, "name": "GPU0", "total_memory_gb": 24, "used_memory_mb": 1024},
        ]
        with patch("builtins.input", side_effect=["abc", "5", "0"]):
            result = ask_gpu_preference(gpus)
        assert result == 0


class TestAskPortPreference:
    """Tests for ask_port_preference interactive prompts."""

    def test_ask_port_valid(self):
        with patch("builtins.input", return_value="8080"):
            result = ask_port_preference()
        assert result == 8080

    def test_ask_port_default(self):
        with patch("builtins.input", return_value=""):
            result = ask_port_preference()
        assert result == 7999

    def test_ask_port_out_of_range_then_valid(self):
        with patch("builtins.input", side_effect=["80", "99999", "8080"]):
            result = ask_port_preference()
        assert result == 8080


class TestAskInterfacePreference:
    """Tests for ask_interface_preference interactive prompts."""

    def test_ask_interface_valid(self):
        interfaces = [
            {"ip": "127.0.0.1", "type": "loopback"},
            {"ip": "192.168.1.100", "type": "local_network"},
        ]
        with patch("builtins.input", return_value="1"):
            result = ask_interface_preference(interfaces)
        assert result == "192.168.1.100"

    def test_ask_interface_single_returns_first(self):
        interfaces = [{"ip": "127.0.0.1", "type": "loopback"}]
        result = ask_interface_preference(interfaces)
        assert result == "127.0.0.1"

    def test_ask_interface_empty_returns_localhost(self):
        result = ask_interface_preference([])
        assert result == "127.0.0.1"

    def test_ask_interface_default_first(self):
        interfaces = [
            {"ip": "127.0.0.1", "type": "loopback"},
            {"ip": "0.0.0.0", "type": "all"},
        ]
        with patch("builtins.input", return_value=""):
            result = ask_interface_preference(interfaces)
        assert result == "127.0.0.1"


class TestPcieStatus:
    """Tests for EnvironmentDetector PCIe helper methods."""

    def test_pcie_status_optimal(self):
        status = EnvironmentDetector.pcie_status(4, 4, 16, 16)
        assert status == "PCIe 4.0 x16"

    def test_pcie_status_gen_degraded(self):
        status = EnvironmentDetector.pcie_status(1, 4, 16, 16)
        assert "DEGRADED" in status
        assert "PCIe 1.0 x16" in status
        assert "expected 4.0 x16" in status

    def test_pcie_status_width_degraded(self):
        status = EnvironmentDetector.pcie_status(4, 4, 8, 16)
        assert "WIDTH DEGRADED" in status
        assert "PCIe 4.0 x8" in status
        assert "expected x16" in status

    def test_pcie_status_both_degraded(self):
        status = EnvironmentDetector.pcie_status(1, 4, 8, 16)
        assert "DEGRADED" in status
        assert "PCIe 1.0 x8" in status

    def test_pcie_is_degraded_gen(self):
        assert EnvironmentDetector.pcie_is_degraded(1, 4, 16, 16) is True

    def test_pcie_is_degraded_width(self):
        assert EnvironmentDetector.pcie_is_degraded(4, 4, 8, 16) is True

    def test_pcie_is_degraded_optimal(self):
        assert EnvironmentDetector.pcie_is_degraded(4, 4, 16, 16) is False

    def test_pcie_is_degraded_both(self):
        assert EnvironmentDetector.pcie_is_degraded(1, 4, 8, 16) is True

    def test_detect_gpus_with_pcie(self):
        fake_output = "0, NVIDIA RTX PRO 6000, 97887, 50000, 1, 4, 16, 16\n1, NVIDIA RTX PRO 6000, 97887, 5000, 4, 4, 8, 16\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = EnvironmentDetector.detect_gpus()
        assert len(gpus) == 2
        assert gpus[0]["pcie_link_gen"] == 1
        assert gpus[0]["pcie_link_gen_max"] == 4
        assert gpus[0]["pcie_link_width"] == 16
        assert gpus[0]["pcie_link_width_max"] == 16
        assert gpus[1]["pcie_link_gen"] == 4
        assert gpus[1]["pcie_link_width"] == 8
        assert gpus[1]["pcie_link_width_max"] == 16

    def test_detect_gpus_without_pcie_fallback(self):
        """Test that basic fields still work if PCIe fields are missing."""
        fake_output = "0, NVIDIA A100, 81920, 1024\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = EnvironmentDetector.detect_gpus()
        assert len(gpus) == 1
        assert gpus[0]["index"] == 0
        assert "pcie_link_gen" not in gpus[0]
