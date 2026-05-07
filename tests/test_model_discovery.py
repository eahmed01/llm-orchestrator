"""Tests for llm_orchestrator.model_discovery module."""

import pytest

from llm_orchestrator.model_discovery import ModelDiscovery


class TestModelDiscovery:
    """Tests for ModelDiscovery."""

    def test_build_fallback_chain_for_27b(self):
        """Test building fallback chain for 27B model."""
        chain = ModelDiscovery.build_fallback_chain("Qwen/Qwen3.6-27B-FP8")

        assert len(chain) > 0
        assert chain[0] == ("Qwen/Qwen3.6-27B-FP8", None)

    def test_build_fallback_chain_includes_quantized(self):
        """Test that fallback chain includes quantized variants."""
        chain = ModelDiscovery.build_fallback_chain(
            "Qwen/Qwen3.6-27B-FP8",
            prefer_quantized=True,
        )

        # Should include quantized variants
        has_quantized = any("Q4" in model or "q4" in model for model, _ in chain)
        assert has_quantized

    def test_build_fallback_chain_for_7b(self):
        """Test building fallback chain for 7B model."""
        chain = ModelDiscovery.build_fallback_chain("Qwen/Qwen3.6-7B-FP8")

        # Should include smaller models as fallback
        assert len(chain) > 1
