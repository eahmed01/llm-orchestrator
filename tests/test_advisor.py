"""Tests for llm_orchestrator.advisor module (in-process transformers model)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_orchestrator.advisor import Advisor


@pytest.fixture
def advisor():
    """Create advisor instance for testing."""
    return Advisor()


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
        with patch("llm_orchestrator.advisor.pipeline", side_effect=Exception("Load failed")):
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

        json_response = json.dumps({
            "recommendation": "2",
            "reasoning": "Try quantized variant",
            "confidence": 0.8,
            "alternatives": ["3"],
            "next_action": "retry",
        })

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

        with patch("llm_orchestrator.advisor.pipeline", side_effect=Exception("Connection error")):
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

        json_response = json.dumps({
            "recommendation": "5",  # Out of bounds
            "reasoning": "Test",
            "confidence": 1.5,  # Out of range
        })

        mock_pipe.return_value = [{"generated_text": f"prompt\n{json_response}"}]
        advisor.pipeline = mock_pipe

        decision = await advisor.decide_next_step(
            model="Model1",
            failure_reason="Error",
            options=options,
        )

        assert decision["recommendation"] == "1"  # Clamped to valid range
        assert decision["confidence"] <= 1.0  # Clamped to 0-1


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

        mock_pipe.return_value = [{"generated_text": f"prompt\n{json.dumps(viability_response)}"}]
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

        mock_pipe.return_value = [{"generated_text": f"prompt\n{json.dumps(viability_response)}"}]
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
