"""Tests for llm_orchestrator.orchestrator module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_orchestrator.orchestrator import Orchestrator
from llm_orchestrator.config import ServiceConfig


class TestOrchestrator:
    """Tests for Orchestrator."""

    def test_orchestrator_creation(self):
        """Test creating an Orchestrator."""
        orchestrator = Orchestrator(service="vllm", model="Qwen/Qwen3.6-27B-FP8")
        assert orchestrator.service == "vllm"
        assert orchestrator.model == "Qwen/Qwen3.6-27B-FP8"
        assert orchestrator.retry_count == 0

    def test_build_vllm_command(self):
        """Test building vLLM command."""
        orchestrator = Orchestrator(service="vllm")
        cmd = orchestrator._build_vllm_command(
            "Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
        )

        assert "vllm" in cmd
        assert "serve" in cmd
        assert "Qwen/Qwen3.6-27B-FP8" in cmd
        assert "--max-num-seqs" in cmd
        assert "848" in cmd

    def test_build_vllm_command_with_qwen_model(self):
        """Test vLLM command includes Qwen-specific args."""
        orchestrator = Orchestrator(service="vllm")
        cmd = orchestrator._build_vllm_command(
            "Qwen/Qwen3.6-27B-FP8",
            args={},
        )

        assert "--enable-auto-tool-choice" in cmd
        assert "--tool-call-parser" in cmd

    def test_build_vllm_command_non_qwen_model(self):
        """Test vLLM command without Qwen-specific args for non-Qwen models."""
        orchestrator = Orchestrator(service="vllm")
        cmd = orchestrator._build_vllm_command(
            "meta-llama/Llama-3.1-70B",
            args={},
        )

        assert "--enable-auto-tool-choice" not in cmd
        assert "--tool-call-parser" not in cmd

    def test_build_vllm_command_default_args(self):
        """Test that default args are added when not specified."""
        orchestrator = Orchestrator(service="vllm")
        cmd = orchestrator._build_vllm_command(
            "meta-llama/Llama-3.1-70B",
            args={},
        )

        assert "--max-num-seqs" in cmd
        assert "--gpu-memory-utilization" in cmd

    def test_exit_failure(self):
        """Test that _exit_failure returns False."""
        orchestrator = Orchestrator(service="vllm")
        result = orchestrator._exit_failure("Test failure reason")
        assert result is False

    @pytest.mark.asyncio
    async def test_close_calls_advisor_close(self):
        """Test that close() calls advisor.close()."""
        orchestrator = Orchestrator(service="vllm")
        orchestrator.advisor.close = AsyncMock()
        await orchestrator.close()
        orchestrator.advisor.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test start() success path."""
        orchestrator = Orchestrator(service="vllm", model="test/model")
        orchestrator.config = MagicMock()

        with patch.object(orchestrator.advisor, "estimate_viability", new_callable=AsyncMock) as mock_viability:
            mock_viability.return_value = {"viable": True, "confidence": 0.9}
            with patch.object(orchestrator, "_attempt_startup", new_callable=AsyncMock) as mock_startup:
                mock_startup.return_value = (True, None)
                with patch("llm_orchestrator.orchestrator.ModelDiscovery.validate_model_exists", new_callable=AsyncMock) as mock_validate:
                    mock_validate.return_value = True

                    result = await orchestrator.start()

                    assert result is True
                    mock_startup.assert_called_once()
                    orchestrator.config.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_no_model_no_saved_config(self):
        """Test start() fails when no model and no saved config."""
        orchestrator = Orchestrator(service="vllm", model=None)
        orchestrator.config = MagicMock()
        orchestrator.config.get_service_config.return_value = None

        result = await orchestrator.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_with_saved_config(self):
        """Test start() loads model from saved config."""
        orchestrator = Orchestrator(service="vllm", model=None)
        saved = ServiceConfig(model="saved/model", args={})
        orchestrator.config = MagicMock()
        orchestrator.config.get_service_config.return_value = saved

        with patch.object(orchestrator, "_attempt_startup", new_callable=AsyncMock) as mock_startup:
            mock_startup.return_value = (True, None)
            with patch.object(orchestrator.advisor, "estimate_viability", new_callable=AsyncMock) as mock_viability:
                mock_viability.return_value = {"viable": True}
                with patch("llm_orchestrator.orchestrator.ModelDiscovery.validate_model_exists", new_callable=AsyncMock) as mock_validate:
                    mock_validate.return_value = True

                    result = await orchestrator.start()

                    assert result is True
                    assert orchestrator.model == "saved/model"

    @pytest.mark.asyncio
    async def test_start_unknown_failure(self):
        """Test start() returns early on unknown failure."""
        orchestrator = Orchestrator(service="vllm", model="test/model")
        orchestrator.config = MagicMock()

        with patch.object(orchestrator.advisor, "estimate_viability", new_callable=AsyncMock) as mock_viability:
            mock_viability.return_value = {"viable": True}
            with patch.object(orchestrator, "_attempt_startup", new_callable=AsyncMock) as mock_startup:
                mock_startup.return_value = (False, None)  # No failure reason
                with patch("llm_orchestrator.orchestrator.ModelDiscovery.validate_model_exists", new_callable=AsyncMock) as mock_validate:
                    mock_validate.return_value = True

                    result = await orchestrator.start()
                    assert result is False

    @pytest.mark.asyncio
    async def test_start_max_retries(self):
        """Test start() fails after max retries."""
        orchestrator = Orchestrator(service="vllm", model="test/model")
        orchestrator.MAX_RETRIES = 2
        orchestrator.config = MagicMock()

        with patch.object(orchestrator.advisor, "estimate_viability", new_callable=AsyncMock) as mock_viability:
            mock_viability.return_value = {"viable": True}
            with patch.object(orchestrator, "_attempt_startup", new_callable=AsyncMock) as mock_startup:
                mock_startup.return_value = (False, "CUDA out of memory")
                with patch("llm_orchestrator.orchestrator.ModelDiscovery.validate_model_exists", new_callable=AsyncMock) as mock_validate:
                    mock_validate.return_value = True
                    with patch("llm_orchestrator.orchestrator.ModelDiscovery.build_fallback_chain") as mock_chain:
                        mock_chain.return_value = [("model1", None), ("model2", "q4")]
                        with patch.object(orchestrator.advisor, "decide_next_step", new_callable=AsyncMock) as mock_decide:
                            mock_decide.return_value = {"recommendation": "1", "reasoning": "test", "confidence": 0.5}

                            result = await orchestrator.start()
                            assert result is False
                            assert orchestrator.retry_count == 2

    @pytest.mark.asyncio
    async def test_start_invalid_recommendation(self):
        """Test start() fails on invalid advisor recommendation."""
        orchestrator = Orchestrator(service="vllm", model="test/model")
        orchestrator.MAX_RETRIES = 2
        orchestrator.config = MagicMock()

        with patch.object(orchestrator.advisor, "estimate_viability", new_callable=AsyncMock) as mock_viability:
            mock_viability.return_value = {"viable": True}
            with patch.object(orchestrator, "_attempt_startup", new_callable=AsyncMock) as mock_startup:
                mock_startup.return_value = (False, "OOM")
                with patch("llm_orchestrator.orchestrator.ModelDiscovery.validate_model_exists", new_callable=AsyncMock) as mock_validate:
                    mock_validate.return_value = True
                    with patch("llm_orchestrator.orchestrator.ModelDiscovery.build_fallback_chain") as mock_chain:
                        mock_chain.return_value = [("model1", None)]
                        with patch.object(orchestrator.advisor, "decide_next_step", new_callable=AsyncMock) as mock_decide:
                            mock_decide.return_value = {"recommendation": "5", "reasoning": "test", "confidence": 0.5}  # Out of bounds

                            result = await orchestrator.start()
                            assert result is False

    @pytest.mark.asyncio
    async def test_attempt_startup_launch_failure(self):
        """Test _attempt_startup when subprocess creation fails."""
        orchestrator = Orchestrator(service="vllm")

        with patch("asyncio.create_subprocess_exec", side_effect=Exception("launch failed")):
            success, reason = await orchestrator._attempt_startup("test/model", {})
            assert success is False
            assert reason == "launch_failed"

    @pytest.mark.asyncio
    async def test_attempt_startup_monitor_error(self):
        """Test _attempt_startup when monitoring fails."""
        orchestrator = Orchestrator(service="vllm")

        mock_process = AsyncMock()
        mock_process.pid = 12345

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("llm_orchestrator.orchestrator.Monitor") as MockMonitor:
                mock_monitor = MagicMock()
                mock_monitor.monitor_startup = AsyncMock(side_effect=Exception("monitor crash"))
                MockMonitor.return_value = mock_monitor

                success, reason = await orchestrator._attempt_startup("test/model", {})
                assert success is False
                assert reason == "monitor_error"
                mock_process.kill.assert_called()

    @pytest.mark.asyncio
    async def test_attempt_startup_success(self):
        """Test _attempt_startup success path."""
        orchestrator = Orchestrator(service="vllm")

        mock_process = AsyncMock()
        mock_process.pid = 12345

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("llm_orchestrator.orchestrator.Monitor") as MockMonitor:
                mock_monitor = MagicMock()
                mock_monitor.monitor_startup = AsyncMock(return_value=(True, None))
                MockMonitor.return_value = mock_monitor

                success, reason = await orchestrator._attempt_startup("test/model", {})
                assert success is True
                assert reason is None

    @pytest.mark.asyncio
    async def test_attempt_startup_failure_kills_process(self):
        """Test _attempt_startup kills process on failure."""
        orchestrator = Orchestrator(service="vllm")

        mock_process = AsyncMock()
        mock_process.pid = 12345

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("llm_orchestrator.orchestrator.Monitor") as MockMonitor:
                mock_monitor = MagicMock()
                mock_monitor.monitor_startup = AsyncMock(return_value=(False, "OOM error"))
                MockMonitor.return_value = mock_monitor

                success, reason = await orchestrator._attempt_startup("test/model", {})
                assert success is False
                assert reason == "OOM error"
                mock_process.kill.assert_called()
