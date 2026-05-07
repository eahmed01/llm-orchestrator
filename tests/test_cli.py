"""Tests for llm_orchestrator.cli module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from llm_orchestrator.cli import app
from llm_orchestrator.config import OrchestratorConfig, ServiceConfig

runner = CliRunner()


class TestDiscoverCommand:
    """Tests for discover command."""

    def test_discover_finds_variants(self):
        """Test discover command finds and displays variants."""
        mock_variants = {
            "Qwen/Qwen3.6-27B-Q4": {"size_gb": 13.89, "format": "q4_k_m"},
            "Qwen/Qwen3.6-27B-Q5": {"size_gb": 18.45, "format": "q5_k_m"},
        }

        with patch(
            "llm_orchestrator.cli.ModelDiscovery.find_variants",
            new_callable=AsyncMock,
            return_value=mock_variants,
        ):
            result = runner.invoke(app, ["discover", "Qwen/Qwen3.6-27B"])

        assert result.exit_code == 0
        assert "13.89GB" in result.stdout
        assert "18.45GB" in result.stdout

    def test_discover_no_variants(self):
        """Test discover command when no variants found."""
        with patch(
            "llm_orchestrator.cli.ModelDiscovery.find_variants",
            new_callable=AsyncMock,
            return_value={},
        ):
            result = runner.invoke(app, ["discover", "unknown-model"])

        assert result.exit_code == 0
        assert "No variants found" in result.stdout


class TestStatusCommand:
    """Tests for status command."""

    def test_status_with_config(self):
        """Test status command when config exists."""
        service_config = ServiceConfig(model="Qwen/Qwen3.6-27B-FP8")

        with patch(
            "llm_orchestrator.cli.OrchestratorConfig.load_from_disk"
        ) as mock_load:
            mock_config = MagicMock()
            mock_config.get_service_config.return_value = service_config
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["status", "vllm"])

        assert result.exit_code == 0
        assert "Qwen/Qwen3.6-27B-FP8" in result.stdout
        assert "Service: vllm" in result.stdout

    def test_status_no_config(self):
        """Test status command when no config found."""
        with patch(
            "llm_orchestrator.cli.OrchestratorConfig.load_from_disk"
        ) as mock_load:
            mock_config = MagicMock()
            mock_config.get_service_config.return_value = None
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["status", "vllm"])

        assert result.exit_code == 0
        assert "No configuration found" in result.stdout


class TestConfigCommand:
    """Tests for config command."""

    def test_config_show(self):
        """Test config show action."""
        service_config = ServiceConfig(model="Qwen/Qwen3.6-27B-FP8")

        with patch(
            "llm_orchestrator.cli.OrchestratorConfig.load_from_disk"
        ) as mock_load:
            mock_config = MagicMock()
            mock_config.get_service_config.return_value = service_config
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["config", "vllm", "--action", "show"])

        assert result.exit_code == 0
        assert "Qwen/Qwen3.6-27B-FP8" in result.stdout

    def test_config_delete(self):
        """Test config delete action."""
        with patch(
            "llm_orchestrator.cli.OrchestratorConfig.load_from_disk"
        ) as mock_load:
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["config", "vllm", "--action", "delete"])

        assert result.exit_code == 0
        mock_config.save_to_disk.assert_called_once()
        assert "Deleted config" in result.stdout


class TestStopCommand:
    """Tests for stop command."""

    def test_stop_service(self):
        """Test stop command."""
        result = runner.invoke(app, ["stop", "vllm"])

        assert result.exit_code == 0
        assert "Stopped vllm" in result.stdout


class TestStartCommand:
    """Tests for start command."""

    @pytest.mark.asyncio
    async def test_start_with_model(self):
        """Test start command with model specified."""
        with patch(
            "llm_orchestrator.cli.Orchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.start.return_value = True
            mock_orchestrator_class.return_value = mock_orchestrator

            result = runner.invoke(
                app,
                ["start", "vllm", "--model", "Qwen/Qwen3.6-27B-FP8"],
            )

        assert result.exit_code == 0
        mock_orchestrator.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_failure(self):
        """Test start command when startup fails."""
        with patch(
            "llm_orchestrator.cli.Orchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.start.return_value = False
            mock_orchestrator_class.return_value = mock_orchestrator

            result = runner.invoke(app, ["start", "vllm"])

        assert result.exit_code == 1


class TestHelpCommands:
    """Tests for help output."""

    def test_main_help(self):
        """Test main help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Automate trial-and-error LLM startup" in result.stdout
        assert "start" in result.stdout
        assert "discover" in result.stdout
        assert "status" in result.stdout

    def test_start_help(self):
        """Test start command help."""
        result = runner.invoke(app, ["start", "--help"])

        assert result.exit_code == 0
        assert "Start an LLM service" in result.stdout
        assert "--model" in result.stdout
        assert "--max-retries" in result.stdout
