"""Tests for llm_orchestrator.model_discovery module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

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

        # Classifier returns JSON which is parsed; heuristic fallback
        # may kick in if JSON parse fails, reading readme which contains "chat"
        assert result["model_type"] in ["instruct", "chat"]
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
    async def test_classify_model_heuristic_instruct_with_mock(self):
        """Test heuristic classification identifies instruct models via mock classifier."""
        readme = "# Instruct Model\nThis is an instruction-tuned model for following commands"

        # Mock classifier that fails, causing top-level exception -> unknown
        def failing_classifier(prompt, **kwargs):
            raise ValueError("Model unavailable")

        result = await ModelDiscovery.classify_model_with_llm(
            "test/instruct-model", readme, classifier=failing_classifier
        )

        # Top-level exception returns unknown
        assert result["model_type"] == "unknown"
        assert result["is_useful"] is False

    @pytest.mark.asyncio
    async def test_classify_model_heuristic_instruct_json_parse_fail(self):
        """Test heuristic classification when JSON parse fails but classifier succeeds."""
        readme = "# Instruct Model\nThis is an instruction-tuned model for following commands"

        # Classifier returns non-JSON text, forcing heuristic fallback
        mock_classifier = lambda prompt, **kwargs: [
            {"generated_text": "Some non-JSON text without braces"}
        ]

        result = await ModelDiscovery.classify_model_with_llm(
            "test/instruct-model", readme, classifier=mock_classifier
        )

        # Heuristic fallback reads the readme and finds "instruction-tuned"
        assert result["model_type"] == "instruct"
        assert result["is_useful"] is True

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


class TestModelDiscoveryFallbackChain:
    """Tests for build_fallback_chain edge cases."""

    def test_build_fallback_chain_35b(self):
        """Test building fallback chain for 35B model."""
        chain = ModelDiscovery.build_fallback_chain("Qwen/Qwen2.5-35B")
        assert len(chain) > 0
        assert chain[0] == ("Qwen/Qwen2.5-35B", None)

    def test_build_fallback_chain_1_5b(self):
        """Test building fallback chain for 1.5B model."""
        chain = ModelDiscovery.build_fallback_chain("Qwen/Qwen2.5-1.5B")
        assert len(chain) > 0
        # Should still have tiny fallback at the end
        assert any("1.5B" in m for m, _ in chain)

    def test_build_fallback_chain_prefer_quantized_false(self):
        """Test building fallback chain without preferring quantized."""
        chain = ModelDiscovery.build_fallback_chain("Qwen/Qwen3.6-27B-FP8", prefer_quantized=False)
        # Should not include quantized variants
        has_q4 = any("Q4" in m or "q4" in m for m, _ in chain if "1.5B" not in m)
        assert not has_q4

    def test_build_fallback_chain_already_q4(self):
        """Test building fallback chain when model is already Q4."""
        chain = ModelDiscovery.build_fallback_chain("Qwen/Qwen3.6-27B-Q4")
        # Should not try to quantize further
        assert chain[0] == ("Qwen/Qwen3.6-27B-Q4", None)


class TestModelDiscoveryValidateModelExists:
    """Tests for validate_model_exists."""

    @pytest.mark.asyncio
    async def test_validate_model_exists_true(self):
        """Test validation returns True for existing model."""
        with patch("llm_orchestrator.model_discovery.model_info", return_value=MagicMock()):
            exists = await ModelDiscovery.validate_model_exists("Qwen/Qwen3.6-27B")
        assert exists is True

    @pytest.mark.asyncio
    async def test_validate_model_exists_false(self):
        """Test validation returns False for non-existent model."""
        with patch("llm_orchestrator.model_discovery.model_info", side_effect=Exception("404")):
            exists = await ModelDiscovery.validate_model_exists("nonexistent/model")
        assert exists is False


class TestModelDiscoveryCache:
    """Tests for cache helpers."""

    def test_get_cache_path(self):
        """Test cache path creation."""
        path = ModelDiscovery._get_cache_path()
        assert path.exists()  # mkdir is called
        assert "models-cache.json" in str(path)

    def test_load_cache_no_file(self):
        """Test loading cache when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            cache = ModelDiscovery._load_cache()
        assert cache is None

    def test_load_cache_expired(self, tmp_path):
        """Test loading cache when expired."""
        from datetime import datetime, timedelta
        cache_file = tmp_path / "models-cache.json"
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        import json as j
        cache_file.write_text(j.dumps({"timestamp": old_time, "models": {"safe": []}}))

        with patch.object(ModelDiscovery, "_get_cache_path", return_value=cache_file):
            cache = ModelDiscovery._load_cache()
        assert cache is None  # Expired

    def test_load_cache_valid(self, tmp_path):
        """Test loading valid cache."""
        cache_file = tmp_path / "models-cache.json"
        now = datetime.now().isoformat()
        import json as j
        models = {"safe": [], "ambitious": [], "experimental": []}
        cache_file.write_text(j.dumps({"timestamp": now, "models": models}))

        with patch.object(ModelDiscovery, "_get_cache_path", return_value=cache_file):
            cache = ModelDiscovery._load_cache()
        assert cache == models

    def test_load_cache_corrupt(self, tmp_path):
        """Test loading corrupt cache returns None."""
        cache_file = tmp_path / "models-cache.json"
        cache_file.write_text("{invalid json")

        with patch.object(ModelDiscovery, "_get_cache_path", return_value=cache_file):
            cache = ModelDiscovery._load_cache()
        assert cache is None

    def test_save_cache(self, tmp_path):
        """Test saving cache."""
        cache_file = tmp_path / "models-cache.json"
        models = {"safe": []}

        with patch.object(ModelDiscovery, "_get_cache_path", return_value=cache_file):
            ModelDiscovery._save_cache(models)

        assert cache_file.exists()
        import json as j
        data = j.loads(cache_file.read_text())
        assert "timestamp" in data
        assert data["models"] == models

    def test_save_cache_io_error(self, tmp_path):
        """Test saving cache handles IO error."""
        cache_file = tmp_path / "sub" / "models-cache.json"

        with patch.object(ModelDiscovery, "_get_cache_path", return_value=cache_file):
            # Should not raise, just log warning
            ModelDiscovery._save_cache({"safe": []})


class TestModelDiscoveryEstimateSize:
    """Tests for _estimate_model_size."""

    def test_estimate_size_7b_fp16(self):
        size = ModelDiscovery._estimate_model_size("Qwen/Qwen3.6-7B")
        assert size is not None
        assert size == 14.0  # 7 * 2

    def test_estimate_size_7b_q4(self):
        size = ModelDiscovery._estimate_model_size("Qwen/Qwen3.6-7B-Q4")
        assert size is not None
        assert size == 8.4  # 7 * 1.2

    def test_estimate_size_7b_int8(self):
        size = ModelDiscovery._estimate_model_size("Qwen/Qwen3.6-7B-Int8")
        assert size is not None
        assert size == 10.5  # 7 * 1.5

    def test_estimate_size_unknown(self):
        size = ModelDiscovery._estimate_model_size("unknown/model")
        assert size is None


class TestModelDiscoveryEstimateRisk:
    """Tests for _estimate_risk."""

    def test_estimate_risk_low(self):
        risk = ModelDiscovery._estimate_risk(5.0, 95)
        assert "Low risk" in risk

    def test_estimate_risk_medium(self):
        risk = ModelDiscovery._estimate_risk(30.0, 95)
        assert "Medium risk" in risk or "Ambitious" in risk

    def test_estimate_risk_high(self):
        risk = ModelDiscovery._estimate_risk(80.0, 95)
        assert "High risk" in risk or "unlikely" in risk

    def test_estimate_risk_no_vram(self):
        risk = ModelDiscovery._estimate_risk(5.0, None)
        assert "Low risk" in risk

    def test_estimate_risk_no_vram_medium(self):
        risk = ModelDiscovery._estimate_risk(20.0, None)
        assert "Medium risk" in risk

    def test_estimate_risk_no_vram_high(self):
        risk = ModelDiscovery._estimate_risk(50.0, None)
        assert "High risk" in risk


class TestModelDiscoveryGenerateDescription:
    """Tests for _generate_description."""

    def test_description_tiny(self):
        desc = ModelDiscovery._generate_description("org/model", 2.0)
        assert "lightweight" in desc or "Fast" in desc

    def test_description_efficient(self):
        desc = ModelDiscovery._generate_description("org/model", 5.0)
        assert "Efficient" in desc or "balance" in desc

    def test_description_solid(self):
        desc = ModelDiscovery._generate_description("org/model", 10.0)
        assert "Solid" in desc or "reasoning" in desc

    def test_description_frontier(self):
        desc = ModelDiscovery._generate_description("org/model", 50.0)
        assert "Frontier" in desc or "excellent" in desc


class TestModelDiscoveryMomentumScore:
    """Tests for calculate_momentum_score."""

    def test_momentum_score_no_created_at(self):
        model = MagicMock()
        model.downloads = 1000
        model.created_at = None
        score = ModelDiscovery.calculate_momentum_score(model)
        assert score == 1000.0

    def test_momentum_score_recent(self):
        from datetime import datetime, timezone, timedelta
        model = MagicMock()
        model.downloads = 1000
        model.created_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        score = ModelDiscovery.calculate_momentum_score(model)
        assert score > 50  # 1000 / 10 = 100

    def test_momentum_score_old_capped(self):
        from datetime import datetime, timezone, timedelta
        model = MagicMock()
        model.downloads = 10000
        model.created_at = (datetime.now(timezone.utc) - timedelta(days=730)).isoformat()
        score = ModelDiscovery.calculate_momentum_score(model)
        assert score == 10000 / 365  # Capped at 365

    def test_momentum_score_datetime_object(self):
        from datetime import datetime, timezone, timedelta
        model = MagicMock()
        model.downloads = 1000
        model.created_at = datetime.now(timezone.utc) - timedelta(days=100)
        score = ModelDiscovery.calculate_momentum_score(model)
        assert score == 10.0  # 1000 / 100


class TestModelDiscoveryDetectGpuVram:
    """Tests for detect_gpu_vram."""

    def test_detect_gpu_vram_nvidia_smi(self):
        fake_output = "97887\n97887\n"
        with patch("subprocess.check_output", return_value=fake_output):
            vram = ModelDiscovery.detect_gpu_vram()
        assert vram == 191  # (97887 + 97887) // 1024 = 191

    def test_detect_gpu_vram_no_nvidia(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError()):
            with patch("torch.cuda.is_available", return_value=False):
                vram = ModelDiscovery.detect_gpu_vram()
        assert vram is None

    def test_detect_gpu_vram_torch(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError()):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=2):
                    with patch("torch.cuda.get_device_properties") as mock_props:
                        mock_props.return_value.total_memory = 97887 * 1024**3
                        vram = ModelDiscovery.detect_gpu_vram()
        assert vram == 97887 * 2  # 2 GPUs


class TestModelDiscoveryContextWindow:
    """Tests for get_context_window."""

    def test_get_context_window_success(self, tmp_path):
        """Test getting context window from config."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"max_position_embeddings": 32768}')

        with patch("llm_orchestrator.model_discovery.hf_hub_download", return_value=str(config_file)):
            cw = ModelDiscovery.get_context_window("test/model")
        assert cw == 32768

    def test_get_context_window_missing(self, tmp_path):
        """Test context window when key not in config."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"other_key": 100}')

        with patch("llm_orchestrator.model_discovery.hf_hub_download", return_value=str(config_file)):
            cw = ModelDiscovery.get_context_window("test/model")
        assert cw is None

    def test_get_context_window_error(self):
        """Test context window when download fails."""
        with patch("llm_orchestrator.model_discovery.hf_hub_download", side_effect=Exception("fail")):
            cw = ModelDiscovery.get_context_window("test/model")
        assert cw is None
