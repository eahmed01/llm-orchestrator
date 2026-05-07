"""Tests for llm_orchestrator.orchestrator module."""


from llm_orchestrator.orchestrator import Orchestrator


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
