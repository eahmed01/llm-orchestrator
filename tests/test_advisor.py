"""Tests for llm_orchestrator.advisor module (in-process transformers model)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_orchestrator.advisor import Advisor


@pytest.fixture
def advisor():
    """Create advisor instance for testing with remote disabled."""
    return Advisor(use_remote=False)


@pytest.fixture
def mock_pipeline():
    """Mock transformers pipeline."""
    return MagicMock()


class TestAdvisorAvailability:
    """Tests for checking advisor availability."""

    @pytest.mark.asyncio
    async def test_check_available_success(self, advisor):
        """Test advisor available when model loads successfully."""
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "test response"}]

        with patch("llm_orchestrator.advisor.pipeline", return_value=mock_pipe):
            available = await advisor.check_available()

        assert available is True
        assert advisor.pipeline is not None

    @pytest.mark.asyncio
    async def test_check_available_model_load_failure(self, advisor):
        """Test advisor unavailable when model fails to load."""
        with patch(
            "llm_orchestrator.advisor.pipeline", side_effect=Exception("Load failed")
        ):
            available = await advisor.check_available()

        assert available is False

    @pytest.mark.asyncio
    async def test_check_available_already_loaded(self, advisor, mock_pipeline):
        """Test advisor returns True immediately if model already loaded."""
        advisor.pipeline = mock_pipeline
        available = await advisor.check_available()
        assert available is True


class TestAdvisorDecisions:
    """Tests for advisor decision-making."""

    @pytest.mark.asyncio
    async def test_decide_next_step_success(self, advisor):
        """Test advisor making a retry decision."""
        mock_pipe = MagicMock()
        options = [
            ("Qwen/Qwen3.6-27B-FP8", None),
            ("Qwen/Qwen3.6-27B-FP8-Q4", "q4_k_m"),
            ("Qwen/Qwen3.6-7B-FP8", None),
        ]

        json_response = json.dumps(
            {
                "recommendation": "2",
                "reasoning": "Try quantized variant",
                "confidence": 0.8,
                "alternatives": ["3"],
                "next_action": "retry",
            }
        )

        mock_pipe.return_value = [{"generated_text": f"prompt text\n{json_response}"}]
        advisor.pipeline = mock_pipe

        decision = await advisor.decide_next_step(
            model="Qwen/Qwen3.6-27B-FP8",
            failure_reason="CUDA out of memory",
            options=options,
        )

        assert decision["recommendation"] == "2"
        assert decision["reasoning"] == "Try quantized variant"
        assert decision["confidence"] == 0.8
        assert 0 <= decision["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_decide_next_step_fallback_when_unavailable(self, advisor):
        """Test fallback decision when advisor is unavailable."""
        options = [
            ("Qwen/Qwen3.6-27B-FP8", None),
            ("Qwen/Qwen3.6-7B-FP8", None),
        ]

        with patch(
            "llm_orchestrator.advisor.pipeline",
            side_effect=Exception("Connection error"),
        ):
            decision = await advisor.decide_next_step(
                model="Qwen/Qwen3.6-27B-FP8",
                failure_reason="OOM",
                options=options,
            )

        assert decision["recommendation"] == "1"
        assert decision["confidence"] < 0.5

    @pytest.mark.asyncio
    async def test_decide_next_step_invalid_json_response(self, advisor):
        """Test fallback when advisor returns invalid JSON."""
        mock_pipe = MagicMock()
        options = [
            ("Qwen/Qwen3.6-27B-FP8", None),
            ("Qwen/Qwen3.6-7B-FP8", None),
        ]

        mock_pipe.return_value = [{"generated_text": "not valid json at all"}]
        advisor.pipeline = mock_pipe

        decision = await advisor.decide_next_step(
            model="Qwen/Qwen3.6-27B-FP8",
            failure_reason="Error",
            options=options,
        )

        assert "recommendation" in decision
        assert decision["confidence"] <= 0.5

    @pytest.mark.asyncio
    async def test_decide_next_step_validates_response(self, advisor):
        """Test that advisor validates and normalizes response."""
        mock_pipe = MagicMock()
        options = [
            ("Model1", None),
            ("Model2", None),
        ]

        json_response = json.dumps(
            {
                "recommendation": "5",  # Out of bounds
                "reasoning": "Test",
                "confidence": 1.5,  # Out of range
            }
        )

        mock_pipe.return_value = [{"generated_text": f"prompt\n{json_response}"}]
        advisor.pipeline = mock_pipe

        decision = await advisor.decide_next_step(
            model="Model1",
            failure_reason="Error",
            options=options,
        )

        assert decision["recommendation"] == "1"  # Clamped to valid range
        assert decision["confidence"] <= 1.0  # Clamped to 0-1


class TestAdvisorRemote:
    """Tests for advisor remote endpoint detection and querying."""

    def test_detect_remote_endpoint_success(self):
        """Test detecting a running remote endpoint."""
        advisor = Advisor(use_remote=True)
        mock_response = json.dumps({"data": [{"id": "test-model"}]}).encode()

        def mock_urlopen(req, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            base, model_id = advisor._detect_remote_endpoint()

        assert base is not None
        assert model_id == "test-model"

    def test_detect_remote_endpoint_not_available(self):
        """Test returning None when no endpoint is available."""
        advisor = Advisor(use_remote=True)
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            base, model_id = advisor._detect_remote_endpoint()

        assert base is None
        assert model_id is None

    def test_detect_remote_endpoint_custom_url(self):
        """Test using custom remote_url."""
        advisor = Advisor(use_remote=True, remote_url="http://custom:9000")
        mock_response = json.dumps({"data": [{"id": "custom-model"}]}).encode()

        def mock_urlopen(req, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            base, model_id = advisor._detect_remote_endpoint()

        assert base == "http://custom:9000"
        assert model_id == "custom-model"

    def test_query_remote_success(self):
        """Test querying remote endpoint successfully."""
        advisor = Advisor(use_remote=True)
        advisor._remote_base = "http://127.0.0.1:8000"

        mock_response = json.dumps({
            "choices": [{"message": {"content": '{"recommendation": "1"}'}}]
        }).encode()

        def mock_urlopen(req, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = advisor._query_remote("test prompt", "system", "model")

        assert '{"recommendation": "1"}' in result

    def test_query_remote_no_base(self):
        """Test query_remote returns None when no base URL set."""
        advisor = Advisor(use_remote=True)
        result = advisor._query_remote("test", "system", "model")
        assert result is None

    def test_query_remote_empty_response(self):
        """Test query_remote handles empty response."""
        advisor = Advisor(use_remote=True)
        advisor._remote_base = "http://127.0.0.1:8000"

        mock_response = json.dumps({
            "choices": [{"message": {"content": ""}}]
        }).encode()

        def mock_urlopen(req, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = advisor._query_remote("test", "system", "model")

        assert result is None

    def test_query_remote_connection_error(self):
        """Test query_remote handles connection errors."""
        advisor = Advisor(use_remote=True)
        advisor._remote_base = "http://127.0.0.1:8000"

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = advisor._query_remote("test", "system", "model")

        assert result is None

    @pytest.mark.asyncio
    async def test_decide_next_step_remote_success(self):
        """Test decide_next_step uses remote when available."""
        advisor = Advisor(use_remote=True)

        def mock_urlopen(req, timeout=None):
            # First call: detect endpoint
            if "/v1/models" in str(getattr(req, "full_url", str(req))):
                mock_resp = MagicMock()
                mock_resp.read.return_value = json.dumps({"data": [{"id": "test-model"}]}).encode()
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp
            # Second call: chat completion
            else:
                mock_resp = MagicMock()
                mock_resp.read.return_value = json.dumps({
                    "choices": [{"message": {"content": '{"recommendation": "2", "confidence": 0.9, "reasoning": "remote"}'}}]
                }).encode()
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            options = [("model1", None), ("model2", "q4")]
            decision = await advisor.decide_next_step(
                model="test/model",
                failure_reason="OOM",
                options=options,
            )

        assert decision["recommendation"] == "2"


class TestAdvisorViability:
    """Tests for viability estimation."""

    @pytest.mark.asyncio
    async def test_estimate_viability_likely_fits(self, advisor):
        """Test estimating model fits on hardware."""
        mock_pipe = MagicMock()

        viability_response = {
            "viable": True,
            "confidence": 0.85,
            "reasoning": "27B-FP8 = ~54GB, 95GB available with 41GB margin",
            "suggestions": [],
        }

        mock_pipe.return_value = [
            {"generated_text": f"prompt\n{json.dumps(viability_response)}"}
        ]
        advisor.pipeline = mock_pipe

        result = await advisor.estimate_viability(
            model="Qwen/Qwen3.6-27B-FP8",
            hardware_vram_gb=95,
        )

        assert result["viable"] is True
        assert result["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_estimate_viability_wont_fit(self, advisor):
        """Test when model won't fit."""
        mock_pipe = MagicMock()

        viability_response = {
            "viable": False,
            "confidence": 0.92,
            "reasoning": "27B exceeds 20GB capacity",
            "suggestions": ["Try 7B model", "Use quantized variant"],
        }

        mock_pipe.return_value = [
            {"generated_text": f"prompt\n{json.dumps(viability_response)}"}
        ]
        advisor.pipeline = mock_pipe

        result = await advisor.estimate_viability(
            model="Qwen/Qwen3.6-27B-FP8",
            hardware_vram_gb=20,
        )

        assert result["viable"] is False
        assert result["confidence"] > 0.5
        assert len(result["suggestions"]) == 2

    @pytest.mark.asyncio
    async def test_estimate_viability_fallback(self, advisor):
        """Test viability estimation fallback on error."""
        with patch("llm_orchestrator.advisor.pipeline", side_effect=Exception("Error")):
            result = await advisor.estimate_viability(
                model="Qwen/Qwen3.6-27B-FP8",
                hardware_vram_gb=95,
            )

        assert "viable" in result
        assert "confidence" in result
        assert result["confidence"] < 0.5

    @pytest.mark.asyncio
    async def test_estimate_viability_fallback_high_vram(self, advisor):
        """Test viability fallback with high VRAM (>50GB)."""
        with patch("llm_orchestrator.advisor.pipeline", side_effect=Exception("Error")):
            result = await advisor.estimate_viability(
                model="Qwen/Qwen3.6-27B-FP8",
                hardware_vram_gb=95,
            )

        assert result["viable"] is True  # >50GB heuristic

    @pytest.mark.asyncio
    async def test_estimate_viability_fallback_low_vram(self, advisor):
        """Test viability fallback with low VRAM (<50GB)."""
        with patch("llm_orchestrator.advisor.pipeline", side_effect=Exception("Error")):
            result = await advisor.estimate_viability(
                model="Qwen/Qwen3.6-27B-FP8",
                hardware_vram_gb=24,
            )

        assert result["viable"] is False  # <=50GB heuristic

    @pytest.mark.asyncio
    async def test_estimate_viability_pipeline_none(self, advisor):
        """Test viability when pipeline is None after _ensure_loaded."""
        advisor.pipeline = None
        advisor._loading = False

        with patch.object(advisor, "_ensure_loaded", new_callable=AsyncMock) as mock_load:
            # _ensure_loaded sets pipeline to None (simulating failure)
            mock_load.side_effect = Exception("Load failed")
            result = await advisor.estimate_viability(
                model="test/model",
                hardware_vram_gb=24,
            )

        assert "viable" in result


class TestAdvisorResponseValidation:
    """Tests for response validation."""

    def test_validate_decision_normalizes_recommendation(self, advisor):
        """Test that recommendation is normalized to valid range."""
        options = [("M1", None), ("M2", None), ("M3", None)]

        decision = {
            "recommendation": "5",
            "reasoning": "Test",
            "confidence": 0.8,
        }

        validated = advisor._validate_decision(decision, options)
        assert validated["recommendation"] == "1"

    def test_validate_decision_clamps_confidence(self, advisor):
        """Test that confidence is clamped to 0-1 range."""
        options = [("M1", None)]

        decision = {
            "recommendation": "1",
            "reasoning": "Test",
            "confidence": 1.5,
        }

        validated = advisor._validate_decision(decision, options)
        assert validated["confidence"] == 1.0

        decision["confidence"] = -0.5
        validated = advisor._validate_decision(decision, options)
        assert validated["confidence"] == 0.0

    def test_validate_decision_ensures_fields(self, advisor):
        """Test that validation ensures all required fields."""
        options = [("M1", None)]

        decision = {"recommendation": "1"}

        validated = advisor._validate_decision(decision, options)
        assert validated["reasoning"] is not None
        assert validated["confidence"] >= 0
        assert validated["alternatives"] is not None

    def test_validate_decision_exception_falls_back(self, advisor):
        """Test that _validate_decision falls back on exception."""
        options = [("M1", None)]
        decision = {"recommendation": "invalid"}

        validated = advisor._validate_decision(decision, options)
        assert validated["recommendation"] == "1"


class TestAdvisorFallback:
    """Tests for fallback decision logic."""

    def test_fallback_decision_with_options(self, advisor):
        """Test fallback chooses first option when available."""
        options = [
            ("M1", None),
            ("M2", None),
            ("M3", None),
        ]

        result = advisor._fallback_decision(options)

        assert result["recommendation"] == "1"
        assert result["confidence"] == 0.3
        assert result["alternatives"] == ["2", "3"]
        assert result["next_action"] == "retry"

    def test_fallback_decision_no_options(self, advisor):
        """Test fallback escalates when no options available."""
        result = advisor._fallback_decision([])

        assert result["recommendation"] == "1"
        assert result["confidence"] == 0.0
        assert result["next_action"] == "escalate"


class TestAdvisorParseJson:
    """Tests for _parse_decision_json."""

    def test_parse_decision_json_no_braces(self, advisor):
        """Test parsing when no JSON braces found."""
        options = [("M1", None)]
        result = advisor._parse_decision_json("just plain text", options)
        assert result["recommendation"] == "1"

    def test_parse_decision_json_invalid(self, advisor):
        """Test parsing when JSON is invalid."""
        options = [("M1", None)]
        result = advisor._parse_decision_json("{invalid json", options)
        assert result["recommendation"] == "1"

    def test_parse_decision_json_not_dict(self, advisor):
        """Test parsing when JSON is not a dict."""
        options = [("M1", None)]
        result = advisor._parse_decision_json("[1, 2, 3]", options)
        assert result["recommendation"] == "1"


class TestAdvisorParseViabilityJson:
    """Tests for _parse_viability_json."""

    def test_parse_viability_json_no_braces(self, advisor):
        """Test parsing when no JSON braces found."""
        result = advisor._parse_viability_json("just plain text")
        assert result is None

    def test_parse_viability_json_invalid(self, advisor):
        """Test parsing when JSON is invalid."""
        result = advisor._parse_viability_json("{invalid json")
        assert result is None

    def test_parse_viability_json_not_dict(self, advisor):
        """Test parsing when JSON is not a dict."""
        result = advisor._parse_viability_json("[1, 2, 3]")
        assert result is None


class TestAdvisorCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_cleans_up_model(self, advisor, mock_pipeline):
        """Test that close properly cleans up model from memory."""
        advisor.pipeline = mock_pipeline
        await advisor.close()
        assert advisor.pipeline is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, advisor):
        """Test that close can be called multiple times safely."""
        await advisor.close()
        await advisor.close()
        assert advisor.pipeline is None
