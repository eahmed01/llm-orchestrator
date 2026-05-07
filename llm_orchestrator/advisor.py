"""Advisor: uses tiny LLM to make intelligent retry decisions."""

import asyncio
import json
import logging
from typing import Any, Optional

from transformers import pipeline

logger = logging.getLogger(__name__)


class Advisor:
    """Queries Qwen2.5-Coder-1.5B in-process for retry decisions."""

    MODEL = "Qwen/Qwen2.5-Coder-1.5B"

    def __init__(self, endpoint: Optional[str] = None):
        # endpoint parameter kept for compatibility but not used
        self.pipeline: Optional[Any] = None
        self._loading = False

    async def check_available(self) -> bool:
        """Check if advisor model can be loaded."""
        try:
            await self._ensure_loaded()
            return self.pipeline is not None
        except Exception as e:
            logger.warning(f"Advisor model unavailable: {e}")
            return False

    async def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self.pipeline is not None or self._loading:
            return

        self._loading = True
        try:
            logger.info("Loading advisor model (first run only)...")
            self.pipeline = await asyncio.to_thread(
                lambda: pipeline(
                    "text-generation",
                    model=self.MODEL,
                    device="cpu",
                    trust_remote_code=True,
                )
            )
            logger.info("✓ Advisor model loaded")
        except Exception as e:
            logger.error(f"Failed to load advisor model: {e}")
            self._loading = False
            raise
        finally:
            if self.pipeline is None:
                self._loading = False

    async def decide_next_step(
        self,
        model: str,
        failure_reason: str,
        options: list[tuple[str, Optional[str]]],
        hardware_info: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ask advisor what to try next given failure and available options.

        Args:
            model: Current model ID
            failure_reason: Why the launch failed (e.g., "CUDA out of memory")
            options: List of (model_id, quantization_variant) tuples to choose from
            hardware_info: Hardware description (e.g., "RTX Pro 6000 Blackwell (95GB)")

        Returns:
            Dict with keys: recommendation, reasoning, confidence, alternatives.
        """
        options_str = "\n".join(
            f"  {i+1}) {opt[0]}" + (f" ({opt[1]})" if opt[1] else "")
            for i, opt in enumerate(options)
        )

        hardware_str = hardware_info or "Unknown hardware"

        prompt = f"""You are an LLM inference optimizer. Help decide the next step for a failing model launch.

Current model: {model}
Failure: {failure_reason}
Hardware: {hardware_str}

Available options:
{options_str}

Choose which option to try next. Consider:
1. Size constraints (avoid models that are obviously too large)
2. Quantization benefits (Q4 reduces size by ~50%)
3. Fallback strategy (prefer smaller models only if necessary)

Respond in JSON format:
{{
  "recommendation": "<N>",  // Number of recommended option (1-indexed)
  "reasoning": "<brief explanation>",
  "confidence": <0.0-1.0>,
  "alternatives": ["<N>", ...],  // Ordered backup options
  "next_action": "retry|investigate|escalate"
}}"""

        try:
            await self._ensure_loaded()

            pipe = self.pipeline
            if pipe is None:
                raise RuntimeError("Model failed to load")

            response = await asyncio.to_thread(
                lambda: pipe(
                    prompt,
                    max_new_tokens=500,
                    temperature=0.3,
                    do_sample=True,
                )
            )

            response_text = response[0]["generated_text"]
            # Extract only the response after the prompt
            if prompt in response_text:
                response_text = response_text[len(prompt) :].strip()

            # Try to extract JSON from response
            try:
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        return self._validate_decision(result, options)
                else:
                    logger.warning(
                        f"No JSON block found in advisor response: {response_text}"
                    )
                    return self._fallback_decision(options)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse advisor response: {response_text}")
                return self._fallback_decision(options)

            # Fallback if JSON is valid but not a dict
            return self._fallback_decision(options)

        except Exception as e:
            logger.error(f"Advisor query failed: {e}")
            return self._fallback_decision(options)

    async def estimate_viability(
        self,
        model: str,
        hardware_vram_gb: int,
        batch_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """Ask advisor: will this model fit on this hardware?

        Returns:
            Dict with keys: viable, confidence, reasoning, suggestions.
        """
        prompt = f"""You are an LLM inference optimizer.

Model: {model}
Hardware: {hardware_vram_gb}GB VRAM
Batch size: {batch_size or "default"}

Will this model fit during startup and inference? Consider:
1. Model weights (typically 50-150% of parameter count in GB)
2. KV cache (depends on sequence length and batch size)
3. Safety margin (always leave 10-20% headroom)

Respond in JSON:
{{
  "viable": true/false,
  "confidence": <0.0-1.0>,
  "reasoning": "<explanation>",
  "suggestions": ["<suggestion>", ...]
}}"""

        try:
            await self._ensure_loaded()

            pipe = self.pipeline
            if pipe is None:
                raise RuntimeError("Model failed to load")

            response = await asyncio.to_thread(
                lambda: pipe(
                    prompt,
                    max_new_tokens=400,
                    temperature=0.2,
                    do_sample=True,
                )
            )

            response_text = response[0]["generated_text"]
            # Extract only the response after the prompt
            if prompt in response_text:
                response_text = response_text[len(prompt) :].strip()

            try:
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        return result
            except json.JSONDecodeError:
                pass

        except Exception as e:
            logger.error(f"Viability check failed: {e}")

        # Fallback: conservative estimate based on model size heuristics
        return {
            "viable": hardware_vram_gb > 50,
            "confidence": 0.4,
            "reasoning": "Advisor unavailable; using fallback heuristic",
            "suggestions": ["Ensure at least 50GB VRAM", "Try quantized variant if OOM"],
        }

    def _validate_decision(
        self,
        decision: dict[str, Any],
        options: list[tuple[str, Optional[str]]],
    ) -> dict[str, Any]:
        """Validate and normalize advisor decision."""
        try:
            rec = int(decision.get("recommendation", 1))
            if not (1 <= rec <= len(options)):
                rec = 1
            decision["recommendation"] = str(rec)
            decision["confidence"] = max(0, min(1, float(decision.get("confidence", 0.5))))
            decision.setdefault("reasoning", "No explanation provided")
            decision.setdefault("alternatives", [])
            decision.setdefault("next_action", "retry")
            return decision
        except Exception:
            return self._fallback_decision(options)

    def _fallback_decision(self, options: list[tuple[str, Optional[str]]]) -> dict[str, Any]:
        """Fallback decision when advisor is unavailable."""
        if not options:
            return {
                "recommendation": "1",
                "reasoning": "No options available",
                "confidence": 0.0,
                "alternatives": [],
                "next_action": "escalate",
            }

        return {
            "recommendation": "1",
            "reasoning": "Advisor unavailable; using first option",
            "confidence": 0.3,
            "alternatives": [str(i + 1) for i in range(1, len(options))],
            "next_action": "retry",
        }

    async def close(self) -> None:
        """Cleanup resources."""
        self.pipeline = None
