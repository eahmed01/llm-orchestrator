"""Tests for llm_orchestrator.stack module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_orchestrator.stack import (
    StackConfig,
    StackServiceInfo,
    StackSnapshot,
    StackDetector,
    default_stack_configs,
)


class TestStackConfig:
    """Tests for StackConfig data model."""

    def test_create_vllm_config(self):
        cfg = StackConfig(
            name="vllm",
            model="Qwen/Qwen3.6-27B-FP8",
            port=7999,
            gpus=[0, 1],
        )
        assert cfg.name == "vllm"
        assert cfg.service_type == "vllm"
        assert cfg.gpus == [0, 1]
        assert cfg.gpus_str() == "0,1"

    def test_create_litellm_config(self):
        cfg = StackConfig(
            name="litellm",
            model="",
            port=8000,
            depends_on=["vllm"],
        )
        assert cfg.depends_on == ["vllm"]
        assert cfg.gpus is None
        assert cfg.gpus_str() == "0"  # default

    def test_resolved_log_file(self):
        configs = default_stack_configs()
        assert "vllm.log" in configs["vllm"].resolved_log_file
        assert "reranker" in configs["reranker"].resolved_log_file
        assert "litellm" in configs["litellm"].resolved_log_file

    def test_custom_log_file(self):
        cfg = StackConfig(
            name="test",
            model="test",
            port=9999,
            log_file="/custom/path.log",
        )
        assert cfg.resolved_log_file == "/custom/path.log"

    def test_gpus_str_single(self):
        cfg = StackConfig(name="x", model="m", port=1, gpus=[5])
        assert cfg.gpus_str() == "5"

    def test_service_type_alias(self):
        cfg = StackConfig(name="reranker", model="m", port=1)
        assert cfg.service_type == "reranker"

    def test_venv_alias(self):
        cfg = StackConfig(
            name="x", model="m", port=1, venv_path="/path/to/venv"
        )
        assert cfg.venv == "/path/to/venv"


class TestDefaultStackConfigs:
    """Tests for default_stack_configs()."""

    def test_returns_three_services(self):
        configs = default_stack_configs()
        assert set(configs.keys()) == {"reranker", "vllm", "litellm"}

    def test_reranker_config(self):
        configs = default_stack_configs()
        r = configs["reranker"]
        assert r.port == 7998
        assert r.gpus == [1]
        assert "Reranker" in r.model

    def test_vllm_config(self):
        configs = default_stack_configs()
        v = configs["vllm"]
        assert v.port == 7999
        assert v.gpus is None

    def test_litellm_depends_on_vllm(self):
        configs = default_stack_configs()
        l = configs["litellm"]
        assert "vllm" in l.depends_on

    def test_all_have_venv(self):
        configs = default_stack_configs()
        for name, cfg in configs.items():
            assert cfg.venv_path is not None, f"{name} missing venv_path"

    def test_all_have_log_file(self):
        configs = default_stack_configs()
        for name, cfg in configs.items():
            assert cfg.log_file is not None, f"{name} missing log_file"


class TestStackServiceInfo:
    """Tests for StackServiceInfo dataclass."""

    def test_create_running(self):
        info = StackServiceInfo(
            name="vllm",
            model="Qwen/Qwen3.6-27B-FP8",
            port=7999,
            pid=12345,
            gpus=[0, 1],
            host="0.0.0.0",
            status="running",
        )
        assert info.status == "running"
        assert info.gpus == [0, 1]

    def test_create_stopped(self):
        info = StackServiceInfo(
            name="litellm",
            model="",
            port=8000,
            pid=0,
            gpus=[],
            host="0.0.0.0",
            status="stopped",
        )
        assert info.status == "stopped"


class TestStackSnapshot:
    """Tests for StackSnapshot dataclass."""

    def test_create_snapshot(self):
        services = {
            "vllm": StackServiceInfo(
                name="vllm",
                model="Qwen/Qwen3.6-27B-FP8",
                port=7999,
                pid=12345,
                gpus=[0],
                host="0.0.0.0",
                status="running",
            )
        }
        from datetime import datetime, timezone

        snap = StackSnapshot(
            services=services,
            gpus=[{"index": 0, "name": "Test GPU", "memory_used_mb": 5000, "memory_total_mb": 24576}],
            timestamp=datetime.now(timezone.utc),
        )
        assert "vllm" in snap.services
        assert len(snap.gpus) == 1

    def test_snapshot_inner_mutable(self):
        """StackSnapshot is frozen at the field level (services dict itself can't be reassigned)."""
        services = {}
        from datetime import datetime, timezone

        snap = StackSnapshot(
            services=services,
            gpus=[],
            timestamp=datetime.now(timezone.utc),
        )
        # The field is frozen — you can't reassign snap.services
        # But the dict inside is still Python-mutable (by design)
        with pytest.raises(Exception):  # FrozenInstanceError
            snap.services = {"x": "y"}  # type: ignore


class TestStackDetector:
    """Tests for StackDetector static methods."""

    def test_verify_endpoint_success(self):
        """Test verify_endpoint returns True for working endpoint."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = StackDetector.verify_endpoint("127.0.0.1", 8000)
            assert result is True

    def test_verify_endpoint_failure(self):
        """Test verify_endpoint returns False for unreachable endpoint."""
        import urllib.error

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
            result = StackDetector.verify_endpoint("127.0.0.1", 99999)
            assert result is False

    def test_verify_endpoint_normalizes_0_0_0_0(self):
        """Test that 0.0.0.0 is normalized to 127.0.0.1."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            StackDetector.verify_endpoint("0.0.0.0", 8000)
            call_args = mock_urlopen.call_args
            url = call_args[0][0].full_url if hasattr(call_args[0][0], 'full_url') else str(call_args[0][0])
            assert "127.0.0.1" in url

    def test_detect_running_services_no_ps(self):
        """Test detection when ps aux fails."""
        with patch("subprocess.check_output", side_effect=Exception("no ps")):
            with patch("pathlib.Path.is_dir", return_value=False):
                services = StackDetector.detect_running_services()
                assert isinstance(services, dict)

    def test_get_gpu_usage_no_nvidia(self):
        """Test GPU query when nvidia-smi is unavailable."""
        with patch("subprocess.check_output", side_effect=FileNotFoundError()):
            gpus = StackDetector.get_gpu_usage()
            assert gpus == []

    def test_get_gpu_usage_success(self):
        """Test GPU query with valid nvidia-smi output."""
        fake_output = "0, NVIDIA RTX PRO 6000, 90784, 97887, 5, 45\n1, NVIDIA RTX PRO 6000, 95349, 97887, 3, 47\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = StackDetector.get_gpu_usage()
            assert len(gpus) == 2
            assert gpus[0]["index"] == 0
            assert gpus[1]["index"] == 1

    def test_read_proc_env_gpus_success(self):
        """Test reading CUDA_VISIBLE_DEVICES from /proc."""
        fake_environ = b"PATH=/usr/bin\x00CUDA_VISIBLE_DEVICES=0,1\x00HOME=/home/xeio\x00"
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=fake_environ):
                gpus = StackDetector._read_proc_env_gpus(12345)
                assert gpus == [0, 1]

    def test_read_proc_env_gpus_missing_file(self):
        """Test reading CUDA_VISIBLE_DEVICES when file doesn't exist."""
        with patch("pathlib.Path.is_file", return_value=False):
            gpus = StackDetector._read_proc_env_gpus(12345)
            assert gpus == []

    def test_read_proc_env_gpus_single_gpu(self):
        """Test reading single GPU from /proc."""
        fake_environ = b"CUDA_VISIBLE_DEVICES=1\x00"
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=fake_environ):
                gpus = StackDetector._read_proc_env_gpus(999)
                assert gpus == [1]


class TestStackDetectorCaptureSnapshot:
    """Integration-style tests for capture_snapshot."""

    def test_snapshot_has_timestamp(self):
        """Test that capture_snapshot produces a snapshot with a timestamp."""
        with patch.object(StackDetector, "detect_running_services", return_value={}):
            with patch.object(StackDetector, "get_gpu_usage", return_value=[]):
                snap = StackDetector.capture_snapshot()
                assert snap.timestamp is not None
                assert isinstance(snap.services, dict)
                assert isinstance(snap.gpus, list)

    def test_snapshot_includes_running_services(self):
        """Test that running services appear in the snapshot."""
        fake_service = StackServiceInfo(
            name="vllm",
            model="Qwen/Qwen3.6-27B-FP8",
            port=7999,
            pid=12345,
            gpus=[0],
            host="0.0.0.0",
            status="running",
        )
        with patch.object(
            StackDetector, "detect_running_services",
            return_value={"vllm": fake_service}
        ):
            with patch.object(StackDetector, "get_gpu_usage", return_value=[]):
                with patch.object(StackDetector, "verify_endpoint", return_value=True):
                    snap = StackDetector.capture_snapshot()
                    assert "vllm" in snap.services
                    assert snap.services["vllm"].pid == 12345


class TestStackDetectorPsAux:
    """Tests for StackDetector ps aux parsing paths."""

    def test_detect_running_services_ps_aux_parsing(self):
        """Test that ps aux output is parsed for vllm serve."""
        fake_ps = "user 12345 0.0 0.0 12345 6789 ? Ss May07 0:00 python -m vllm serve --port 7999 Qwen/model\nuser 67890 0.0 0.0 11111 2222 ? Ss May07 0:00 python -m litellm --port 8000"
        with patch("subprocess.check_output", return_value=fake_ps):
            with patch("pathlib.Path.is_dir", return_value=False):
                services = StackDetector.detect_running_services()

        assert "vllm" in services
        assert services["vllm"].port == 7999

    def test_detect_running_services_short_lines(self):
        """Test that short ps aux lines are skipped."""
        fake_ps = "short line\nanother short line"
        with patch("subprocess.check_output", return_value=fake_ps):
            with patch("pathlib.Path.is_dir", return_value=False):
                services = StackDetector.detect_running_services()

        assert services == {}

    def test_detect_running_services_pid_parse_error(self):
        """Test that PID parse errors are handled."""
        fake_ps = "user    abc  0.0  0.0 ?        Ss   00:00:00 python -m vllm serve --port 7999 model"
        with patch("subprocess.check_output", return_value=fake_ps):
            with patch("pathlib.Path.is_dir", return_value=False):
                services = StackDetector.detect_running_services()

        assert services == {}

    def test_detect_running_services_port_parse_error(self):
        """Test that port parse errors are handled."""
        fake_ps = "user     12345  0.0  0.0 ?        Ss   00:00:00 python -m vllm serve --port notanumber model"
        with patch("subprocess.check_output", return_value=fake_ps):
            with patch("pathlib.Path.is_dir", return_value=False):
                services = StackDetector.detect_running_services()

        assert services == {}


class TestStackDetectorPidFiles:
    """Tests for PID file fallback detection."""

    def test_detect_pid_file_alive(self):
        """Test PID file detection when process is alive."""
        fake_ps = ""
        with patch("subprocess.check_output", return_value=fake_ps):
            # Mock PID dir
            with patch("pathlib.Path.is_dir") as mock_is_dir:
                mock_is_dir.side_effect = lambda: True
                with patch("pathlib.Path.glob") as mock_glob:
                    fake_pid_file = MagicMock()
                    fake_pid_file.stem = "vllm"
                    fake_pid_file.read_text.return_value = "12345"
                    mock_glob.return_value = [fake_pid_file]

                    # Mock /proc/12345 exists
                    with patch("pathlib.Path.is_dir") as mock_proc:
                        def side_effect(self):
                            if "12345" in str(self):
                                return True
                            return True  # is_dir for PID_DIR check
                        mock_proc.return_value = True

                        services = StackDetector.detect_running_services()

        # Should find service from PID file
        assert "vllm" in services

    def test_detect_pid_file_dead(self):
        """Test PID file detection when process is dead."""
        fake_ps = ""
        with patch("subprocess.check_output", return_value=fake_ps):
            with patch("llm_orchestrator.stack._PID_DIR") as mock_pid_dir:
                mock_pid_dir.is_dir.return_value = True
                fake_pid_file = MagicMock()
                fake_pid_file.stem = "vllm"
                fake_pid_file.read_text.return_value = "12345"
                mock_pid_dir.glob.return_value = [fake_pid_file]

                with patch("pathlib.Path.is_dir", return_value=False):
                    with patch.object(StackDetector, "_read_proc_env_gpus", return_value=[]):
                        with patch("pathlib.Path.is_file", return_value=False):
                            services = StackDetector.detect_running_services()

        # Should find dead service from PID file
        assert "vllm" in services
        assert services["vllm"].status == "stopped"

    def test_detect_pid_file_parse_error(self):
        """Test PID file with invalid content is skipped."""
        fake_ps = ""
        with patch("subprocess.check_output", return_value=fake_ps):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.glob") as mock_glob:
                    fake_pid_file = MagicMock()
                    fake_pid_file.stem = "vllm"
                    fake_pid_file.read_text.return_value = "not_a_number"
                    mock_glob.return_value = [fake_pid_file]

                    services = StackDetector.detect_running_services()

        assert services == {}

    def test_detect_pid_file_os_error(self):
        """Test PID file read error is skipped."""
        fake_ps = ""
        with patch("subprocess.check_output", return_value=fake_ps):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.glob") as mock_glob:
                    fake_pid_file = MagicMock()
                    fake_pid_file.stem = "vllm"
                    fake_pid_file.read_text.side_effect = OSError("Permission denied")
                    mock_glob.return_value = [fake_pid_file]

                    services = StackDetector.detect_running_services()

        assert services == {}


class TestStackDetectorGuessLogFile:
    """Tests for _guess_log_file."""

    def test_guess_log_file_exists(self):
        with patch("pathlib.Path.is_file", return_value=True):
            result = StackDetector._guess_log_file("vllm", 12345)
        assert "vllm.log" in result

    def test_guess_log_file_not_exists(self):
        with patch("pathlib.Path.is_file", return_value=False):
            result = StackDetector._guess_log_file("vllm", 12345)
        assert result is None


class TestStackDetectorReadProcEnv:
    """Tests for _read_proc_env_gpus edge cases."""

    def test_read_proc_env_gpus_no_cuda_var(self):
        """Test reading when CUDA_VISIBLE_DEVICES not set."""
        fake_environ = b"PATH=/usr/bin\x00HOME=/home/xeio\x00"
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=fake_environ):
                gpus = StackDetector._read_proc_env_gpus(12345)
        assert gpus == []

    def test_read_proc_env_gpus_unicode_error(self):
        """Test reading when environ has unicode error."""
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.read_bytes", side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")):
                gpus = StackDetector._read_proc_env_gpus(12345)
        assert gpus == []

    def test_read_proc_env_gpus_os_error(self):
        """Test reading when environ file read fails."""
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.read_bytes", side_effect=OSError("Permission denied")):
                gpus = StackDetector._read_proc_env_gpus(12345)
        assert gpus == []

    def test_read_proc_env_gpus_non_numeric(self):
        """Test reading when CUDA_VISIBLE_DEVICES has non-numeric values."""
        fake_environ = b"CUDA_VISIBLE_DEVICES=all\x00"
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=fake_environ):
                gpus = StackDetector._read_proc_env_gpus(12345)
        assert gpus == []


class TestStackConfigProperties:
    """Tests for StackConfig properties and methods."""

    def test_resolved_log_file_vllm_default(self):
        cfg = StackConfig(name="vllm", model="test", port=7999)
        assert "vllm.log" in cfg.resolved_log_file

    def test_resolved_log_file_reranker_default(self):
        cfg = StackConfig(name="reranker", model="test", port=7998)
        assert "reranker" in cfg.resolved_log_file

    def test_resolved_log_file_litellm_default(self):
        cfg = StackConfig(name="litellm", model="test", port=8000)
        assert "litellm.log" in cfg.resolved_log_file

    def test_resolved_log_file_unknown_default(self):
        cfg = StackConfig(name="unknown", model="test", port=9999)
        assert "unknown.log" in cfg.resolved_log_file

    def test_gpus_str_none_defaults_to_0(self):
        cfg = StackConfig(name="x", model="m", port=1)
        assert cfg.gpus_str() == "0"

    def test_venv_alias_none(self):
        cfg = StackConfig(name="x", model="m", port=1)
        assert cfg.venv is None


class TestStackDetectorGetGpuUsage:
    """Tests for get_gpu_usage edge cases."""

    def test_get_gpu_usage_empty_lines(self):
        """Test GPU query with empty lines in output."""
        fake_output = "\n0, GPU, 50000, 97887, 5, 45\n\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = StackDetector.get_gpu_usage()
        assert len(gpus) == 1
        assert gpus[0]["index"] == 0

    def test_get_gpu_usage_short_lines(self):
        """Test GPU query with short/malformed lines."""
        fake_output = "0, GPU\n"  # Missing fields
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = StackDetector.get_gpu_usage()
        assert gpus == []

    def test_get_gpu_usage_parse_error(self):
        """Test GPU query with parse errors."""
        fake_output = "abc, GPU, xyz, 97887, 5, 45\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = StackDetector.get_gpu_usage()
        assert gpus == []


class TestStackDetectorCaptureSnapshot:
    """Tests for capture_snapshot with running services."""

    def test_snapshot_empty(self):
        """Test capture_snapshot with no services and no GPUs."""
        with patch.object(StackDetector, "detect_running_services", return_value={}):
            with patch.object(StackDetector, "get_gpu_usage", return_value=[]):
                snap = StackDetector.capture_snapshot()
        assert snap.services == {}
        assert snap.gpus == []
        assert snap.timestamp is not None

    def test_snapshot_stopped_service(self):
        """Test that stopped services are included in snapshot."""
        fake_service = StackServiceInfo(
            name="vllm",
            model="Qwen/Qwen3.6-27B-FP8",
            port=7999,
            pid=12345,
            gpus=[0],
            host="0.0.0.0",
            status="stopped",
        )
        with patch.object(StackDetector, "detect_running_services", return_value={"vllm": fake_service}):
            with patch.object(StackDetector, "get_gpu_usage", return_value=[]):
                snap = StackDetector.capture_snapshot()
        assert "vllm" in snap.services


class TestStackDetectorGpuUsage:
    """Tests for get_gpu_usage with PCIe fields."""

    def test_get_gpu_usage_with_pcie(self):
        """Test GPU query with PCIe fields in output."""
        fake_output = "0, NVIDIA RTX PRO 6000, 90784, 97887, 5, 45, 1, 4, 16, 16\n1, NVIDIA RTX PRO 6000, 95349, 97887, 3, 47, 4, 4, 8, 16\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = StackDetector.get_gpu_usage()
        assert len(gpus) == 2
        assert gpus[0]["pcie_link_gen"] == 1
        assert gpus[0]["pcie_link_gen_max"] == 4
        assert gpus[0]["pcie_link_width"] == 16
        assert gpus[1]["pcie_link_gen"] == 4
        assert gpus[1]["pcie_link_width"] == 8

    def test_get_gpu_usage_without_pcie(self):
        """Test GPU query when PCIe fields are not available (older nvidia-smi)."""
        fake_output = "0, NVIDIA A100, 50000, 97887, 5, 45\n"
        with patch("subprocess.check_output", return_value=fake_output):
            gpus = StackDetector.get_gpu_usage()
        assert len(gpus) == 1
        assert gpus[0]["index"] == 0
        assert "pcie_link_gen" not in gpus[0]
