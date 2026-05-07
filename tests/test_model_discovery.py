"""Tests for llm_orchestrator.model_discovery module."""

from pathlib import Path
from unittest.mock import patch

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

    def test_classification_cache_path(self):
        """Test that model cache path is created correctly."""
        path = ModelDiscovery._get_classification_cache_path()

        assert isinstance(path, Path)
        assert "llm-orchestrator" in str(path)
        assert "model-cache.json" in str(path)

    def test_load_empty_classification_cache(self):
        """Test loading model cache when none exists."""
        # Remove cache if it exists
        cache_path = ModelDiscovery._get_classification_cache_path()
        if cache_path.exists():
            cache_path.unlink()

        cache = ModelDiscovery._load_model_cache()
        assert cache == {}

    def test_save_and_load_classification_cache(self, tmp_path):
        """Test saving and loading model cache (README + classification)."""
        test_cache = {
            "Qwen/Qwen3.6-7B": {
                "readme_content": "# Qwen3.6-7B\nA test model",
                "classification": {"model_type": "instruct", "is_useful": True},
            },
            "meta-llama/Llama-2-7B": {
                "readme_content": "# Llama 2 7B",
                "classification": {"model_type": "base", "is_useful": True},
            },
        }

        # Mock the cache path
        with patch.object(
            ModelDiscovery,
            "_get_classification_cache_path",
            return_value=tmp_path / "test-cache.json",
        ):
            ModelDiscovery._save_model_cache(test_cache)

            loaded_cache = ModelDiscovery._load_model_cache()

            # Check that essential data is there
            assert "Qwen/Qwen3.6-7B" in loaded_cache
            assert loaded_cache["Qwen/Qwen3.6-7B"]["classification"]["model_type"] == "instruct"
            assert "readme_content" in loaded_cache["Qwen/Qwen3.6-7B"]

    @pytest.mark.asyncio
    async def test_classify_model_with_llm_fallback(self):
        """Test LLM classification with fallback to heuristics."""
        # Test with no transformers available
        result = await ModelDiscovery.classify_model_with_llm(
            "Qwen/Qwen3.6-7B-Instruct",
            "This is an instruction-tuned model",
        )

        # Should return classification (via heuristics if LLM fails)
        assert "model_type" in result
        assert isinstance(result.get("is_useful"), bool)

    @pytest.mark.asyncio
    async def test_classify_model_llm_prompt_echo(self):
        """Test that classification works when model echoes prompt."""
        # Mock a classifier that echoes the prompt then generates JSON
        mock_classifier = lambda prompt, **kwargs: [
            {
                "generated_text": prompt
                + '\n{"model_type": "instruct", "is_useful": true}'
            }
        ]

        result = await ModelDiscovery.classify_model_with_llm(
            "test/model",
            "# Test Model\nInstruction-tuned model for chat",
            classifier=mock_classifier,
        )

        assert result["model_type"] == "instruct"
        assert result["is_useful"] is True

    @pytest.mark.asyncio
    async def test_classify_model_llm_invalid_json(self):
        """Test fallback when LLM returns invalid JSON."""
        mock_classifier = lambda prompt, **kwargs: [
            {"generated_text": "This is not JSON at all"}
        ]

        result = await ModelDiscovery.classify_model_with_llm(
            "test/model",
            "# Chat Model\nConversational and instruction-tuned",
            classifier=mock_classifier,
        )

        # Should fall back to heuristics
        assert result["model_type"] in ["chat", "instruct", "base"]

    @pytest.mark.asyncio
    async def test_classify_model_llm_malformed_json(self):
        """Test fallback when JSON is incomplete or malformed."""
        mock_classifier = lambda prompt, **kwargs: [
            {"generated_text": '{"model_type": "instruct"'}  # Missing closing brace
        ]

        result = await ModelDiscovery.classify_model_with_llm(
            "test/model",
            "# Instruct Model",
            classifier=mock_classifier,
        )

        # Should fall back to heuristics from README
        assert result["model_type"] in ["instruct", "base"]

    @pytest.mark.asyncio
    async def test_classify_model_heuristic_instruct(self):
        """Test heuristic classification identifies instruct models."""
        readme = "# Instruct Model\nThis is an instruction-tuned model for following commands"

        result = await ModelDiscovery.classify_model_with_llm(
            "test/instruct-model", readme, classifier=None
        )

        assert result["model_type"] == "instruct"

    @pytest.mark.asyncio
    async def test_classify_model_heuristic_chat(self):
        """Test heuristic classification identifies chat models."""
        readme = "# Chat Model\nThis is a conversational model for dialogue"

        result = await ModelDiscovery.classify_model_with_llm(
            "test/chat-model", readme, classifier=None
        )

        assert result["model_type"] == "chat"

    @pytest.mark.asyncio
    async def test_classify_model_heuristic_base(self):
        """Test heuristic classification defaults to base for unknown models."""
        readme = "# Base Model\nA generic language model"

        result = await ModelDiscovery.classify_model_with_llm(
            "test/base-model", readme, classifier=None
        )

        assert result["model_type"] == "base"

    @pytest.mark.asyncio
    async def test_classify_model_exception_returns_unknown(self):
        """Test that exceptions result in unknown classification."""

        def failing_classifier(prompt, **kwargs):
            raise ValueError("Model load failed")

        result = await ModelDiscovery.classify_model_with_llm(
            "test/model", "Some content", classifier=failing_classifier
        )

        assert result["model_type"] == "unknown"
        assert result["is_useful"] is False

    @pytest.mark.asyncio
    async def test_get_or_classify_model_cached(self, tmp_path):
        """Test getting classification from cache."""
        cached_entry = {
            "readme_content": "# Test Model",
            "classification": {"model_type": "instruct", "is_useful": True},
        }

        with patch.object(
            ModelDiscovery,
            "_get_classification_cache_path",
            return_value=tmp_path / "test-cache.json",
        ):
            # Pre-populate cache
            ModelDiscovery._save_model_cache({"test-model": cached_entry})

            # Get classification
            result = await ModelDiscovery.get_or_classify_model("test-model")

            assert result == cached_entry["classification"]

    @pytest.mark.asyncio
    async def test_get_or_classify_model_fallback_heuristic(self, tmp_path):
        """Test classification with heuristic fallback when README fetch fails."""
        with patch.object(
            ModelDiscovery,
            "_get_classification_cache_path",
            return_value=tmp_path / "test-cache.json",
        ):
            with patch.object(
                ModelDiscovery,
                "_fetch_model_readme",
                return_value="",
            ):
                result = await ModelDiscovery.get_or_classify_model(
                    "Qwen/Qwen3.6-7B-Instruct"
                )

                assert result["model_type"] in [
                    "base",
                    "instruct",
                    "chat",
                    "specialized",
                ]
                assert "is_useful" in result
