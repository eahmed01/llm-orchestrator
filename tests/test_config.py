"""Tests for llm_orchestrator.config module."""

import tempfile
from pathlib import Path

import pytest

from llm_orchestrator.config import OrchestratorConfig, ServiceConfig


class TestServiceConfig:
    """Tests for ServiceConfig."""

    def test_service_config_creation(self):
        """Test creating a ServiceConfig."""
        config = ServiceConfig(
            model="Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
        )
        assert config.model == "Qwen/Qwen3.6-27B-FP8"
        assert config.args["--max-num-seqs"] == 848

    def test_service_config_with_variant(self):
        """Test ServiceConfig with quantization variant."""
        config = ServiceConfig(
            model="Qwen/Qwen3.6-27B",
            variant="q4_k_m",
        )
        assert config.variant == "q4_k_m"


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_load_empty_config(self):
        """Test loading config when file doesn't exist."""
        config = OrchestratorConfig.load_from_disk()
        assert config.vllm is None

    def test_config_dir_creation(self):
        """Test that config directory is created."""
        config_dir = OrchestratorConfig.config_dir()
        assert config_dir.exists()

    def test_record_success(self):
        """Test recording a successful startup."""
        config = OrchestratorConfig()
        config.record_success(
            "vllm",
            "Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
        )

        assert config.vllm is not None
        assert config.vllm.model == "Qwen/Qwen3.6-27B-FP8"
        assert config.vllm.last_successful is not None
