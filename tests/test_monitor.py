"""Tests for llm_orchestrator.monitor module."""

import tempfile
from pathlib import Path

import pytest

from llm_orchestrator.monitor import Monitor


class TestMonitor:
    """Tests for Monitor."""

    def test_monitor_creation(self):
        """Test creating a Monitor."""
        with tempfile.NamedTemporaryFile() as f:
            monitor = Monitor(f.name)
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
    async def test_monitor_timeout(self):
        """Test that monitor times out."""
        with tempfile.NamedTemporaryFile() as f:
            monitor = Monitor(f.name)
            success, reason = await monitor.monitor_startup(timeout_seconds=1)

            assert success is False
            assert reason == "startup_timeout"
