"""Model discovery: find variants on HuggingFace."""

import asyncio
import logging
from typing import Any, Optional

from huggingface_hub import model_info

logger = logging.getLogger(__name__)


class ModelDiscovery:
    """Discover model variants and metadata from HuggingFace."""

    # Quantization format preferences (ordered by size reduction)
    QUANTIZATION_FORMATS = ["q4_k_m", "q4_0", "q5_k_m", "q5_0", "q8_0", "fp8", "awq"]

    @staticmethod
    async def find_variants(model_id: str) -> dict[str, dict[str, Any]]:
        """Find quantized variants of a model on HuggingFace.

        Returns:
            Dict mapping variant name to metadata: {"name": {"size_gb": X, "format": "q4"}}.
        """
        try:
            info = model_info(model_id)
            variants: dict[str, dict[str, Any]] = {}

            # Look through siblings (files in the repo)
            if not info.siblings:
                logger.warning(f"No siblings found for {model_id}")
                return variants

            for sibling in info.siblings:
                filename = sibling.rfilename.lower()
                size_bytes = sibling.size or 0

                # Match quantization formats
                for fmt in ModelDiscovery.QUANTIZATION_FORMATS:
                    if fmt in filename and (".gguf" in filename or ".safetensors" in filename):
                        variant_name = f"{model_id}-{fmt.upper()}"
                        variants[variant_name] = {
                            "size_gb": size_bytes / (1024 ** 3),
                            "format": fmt,
                            "filename": sibling.rfilename,
                        }
                        break

            # Also include the base model if not quantized
            for sibling in info.siblings:
                if sibling.rfilename == "model.safetensors" or (
                    ".safetensors" in sibling.rfilename and "gguf" not in sibling.rfilename
                ):
                    size_bytes = sibling.size or 0
                    variants[model_id] = {
                        "size_gb": size_bytes / (1024 ** 3),
                        "format": "fp16",
                        "filename": sibling.rfilename,
                    }
                    break

            logger.info(f"Found {len(variants)} variants for {model_id}")
            return variants

        except Exception as e:
            logger.error(f"Failed to discover variants for {model_id}: {e}")
            return {}

    @staticmethod
    async def get_model_info(model_id: str) -> dict[str, Any]:
        """Fetch model metadata from HuggingFace.

        Returns:
            Dict with model metadata: {"model_id", "tags", "downloads", "config", etc.}
        """
        try:
            info = model_info(model_id)

            # Extract useful metadata
            return {
                "model_id": getattr(info, "model_id", model_id),
                "author": getattr(info, "author", None),
                "tags": info.tags or [],
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "config": getattr(info, "config", {}),
            }
        except Exception as e:
            logger.error(f"Failed to fetch metadata for {model_id}: {e}")
            return {}

    @staticmethod
    def build_fallback_chain(
        base_model: str,
        available_vram_gb: int = 95,
        prefer_quantized: bool = True,
    ) -> list[tuple[str, Optional[str]]]:
        """Build an intelligent fallback chain.

        Args:
            base_model: Original model to try (e.g., "Qwen/Qwen3.6-27B-FP8")
            available_vram_gb: Available GPU VRAM
            prefer_quantized: Whether to prefer quantized variants

        Returns:
            Ordered list of (model_id, quantization_format) tuples to try.
        """
        chain: list[tuple[str, Optional[str]]] = [
            (base_model, None),  # Try original first
        ]

        # Extract model size from model ID heuristics
        # E.g., "Qwen/Qwen3.6-27B-FP8" -> 27B
        size_hint: Optional[float] = None
        if "-27B" in base_model:
            size_hint = 27
        elif "-35B" in base_model:
            size_hint = 35
        elif "-7B" in base_model:
            size_hint = 7
        elif "-1.5B" in base_model:
            size_hint = 1.5

        # Add quantized variants if base is not quantized
        if prefer_quantized and "q4" not in base_model.lower():
            if size_hint and size_hint >= 20:  # Only for large models
                # Try quantized versions of same size
                base_without_quant = base_model.rsplit("-", 1)[0]
                for fmt in ["q4_k_m", "q5_k_m"]:
                    chain.append((f"{base_without_quant}-{fmt.upper()}", fmt))

        # Add smaller model variants as fallback
        if size_hint:
            # Build fallback sizes
            fallback_sizes: list[float] = []
            if size_hint >= 27:
                fallback_sizes = [7, 1.5]
            elif size_hint >= 7:
                fallback_sizes = [1.5]

            # Add fallback models (simplified; would need mapping)
            for fb_size in fallback_sizes:
                fb_model = base_model.replace(f"{int(size_hint)}B", f"{int(fb_size)}B")
                chain.append((fb_model, None))
                if prefer_quantized:
                    fb_base = fb_model.rsplit("-", 1)[0]
                    chain.append((f"{fb_base}-Q4", "q4_k_m"))

        # Always add tiny fallback model as last resort
        chain.append(("Qwen/Qwen2.5-Coder-1.5B-Instruct", None))
        chain.append(("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "q4_k_m"))

        return chain

    @staticmethod
    async def validate_model_exists(model_id: str) -> bool:
        """Check if a model exists on HuggingFace."""
        try:
            await asyncio.to_thread(model_info, model_id)
            return True
        except Exception:
            return False
