"""Tests for llm_orchestrator.monitor module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from llm_orchestrator.monitor import Monitor


class TestMonitor:
    """Tests for Monitor."""

    def test_monitor_creation(self):
        """Test creating a Monitor."""
        with tempfile.NamedTemporaryFile() as f:
            monitor = Monitor(f.name)
            assert monitor.position == 0

    def test_reset(self):
        """Test that reset positions to 0."""
        with tempfile.NamedTemporaryFile() as f:
            monitor = Monitor(f.name)
            monitor.position = 100
            monitor.reset()
            assert monitor.position == 0

    @pytest.mark.asyncio
    async def test_monitor_detects_success(self):
        """Test that monitor detects successful startup."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Application startup complete\n")
            f.flush()

            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=5)

            assert success is True
            assert reason is None

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_monitor_detects_uvicorn(self):
        """Test that monitor detects Uvicorn running pattern."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("INFO:     Uvicorn running on http://0.0.0.0:7999\n")
            f.flush()

            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=5)

            assert success is True
            assert reason is None

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_monitor_detects_server_process(self):
        """Test that monitor detects started server process."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Started server process [pid=12345]\n")
            f.flush()

            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=5)

            assert success is True

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_monitor_detects_oom_error(self):
        """Test that monitor detects OOM errors."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("CUDA out of memory\n")
            f.flush()

            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=5)

            assert success is False
            assert "CUDA out of memory" in reason

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_monitor_detects_outofmemoryerror(self):
        """Test that monitor detects OutOfMemoryError."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("torch.OutOfMemoryError: CUDA out of memory\n")
            f.flush()

            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=5)

            assert success is False
            assert "OutOfMemoryError" in reason or "out of memory" in reason.lower()

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_monitor_detects_cuda_runtime_error(self):
        """Test that monitor detects RuntimeError with cuda."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("RuntimeError: CUDA error: device-side assert triggered\n")
            f.flush()

            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=5)

            assert success is False

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_monitor_detects_failed_to_load_model(self):
        """Test that monitor detects failed to load model."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Failed to load model: file not found\n")
            f.flush()

            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=5)

            assert success is False

            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_monitor_timeout(self):
        """Test that monitor times out."""
        with tempfile.NamedTemporaryFile() as f:
            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=1)

            assert success is False
            assert reason == "startup_timeout"

    @pytest.mark.asyncio
    async def test_monitor_exception_reading_log(self):
        """Test that monitor handles exceptions when reading log."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=PermissionError("denied")):
                monitor = Monitor("/tmp/test.log")
                success, reason = await monitor.monitor_startup(timeout_seconds=1)

                assert success is False
                assert reason == "startup_timeout"


class TestHealthCheck:
    """Tests for Monitor.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await Monitor.health_check("127.0.0.1", 8000)
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        import urllib.error
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
            result = await Monitor.health_check("127.0.0.1", 99999)
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_0_0_0_0_normalized(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            await Monitor.health_check("0.0.0.0", 8000)
            call_args = mock_urlopen.call_args
            req = call_args[0][0] if call_args else None
            url = getattr(req, "full_url", str(req)) if req else ""
            assert "127.0.0.1" in url

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError()
            result = await Monitor.health_check("127.0.0.1", 8000)
            assert result is False


class TestWaitForService:
    """Tests for Monitor.wait_for_service()."""

    @pytest.mark.asyncio
    async def test_wait_for_service_becomes_healthy(self):
        with patch.object(Monitor, "health_check", new_callable=lambda: MagicMock()) as mock_hc:
            async def side_effect(*args, **kwargs):
                side_effect.call_count = getattr(side_effect, "call_count", 0) + 1
                return side_effect.call_count >= 2
            mock_hc.side_effect = side_effect

            result = await Monitor.wait_for_service("127.0.0.1", 8000, timeout=10, interval=0.1)
            assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_service_never_healthy(self):
        with patch.object(Monitor, "health_check", return_value=False):
            result = await Monitor.wait_for_service("127.0.0.1", 9999, timeout=0.5, interval=0.1)
            assert result is False


class TestPortInUse:
    """Tests for Monitor.port_in_use()."""

    def test_port_in_use_false(self):
        with patch("socket.socket") as mock_sock:
            mock_instance = MagicMock()
            mock_sock.return_value = mock_instance
            result = Monitor.port_in_use(19999)
            assert result is False

    def test_port_in_use_true(self):
        with patch("socket.socket") as mock_sock:
            mock_instance = MagicMock()
            mock_sock.return_value = mock_instance
            mock_instance.bind.side_effect = OSError("Address in use")
            result = Monitor.port_in_use(8000)
            assert result is True
