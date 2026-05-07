"""Model discovery: find variants on HuggingFace."""

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import hf_hub_download, list_models, model_info

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

        Note: File sizes may be unavailable (0.00GB) due to API limitations on large models.
        The presence of files (especially .safetensors) indicates the model is available.
        """
        try:
            info = model_info(model_id)
            variants: dict[str, dict[str, Any]] = {}

            # Look through siblings (files in the repo)
            if not info.siblings:
                logger.warning(f"No siblings found for {model_id}")
                return variants

            # Group files by base name (for sharded models)
            file_groups: dict[str, list[Any]] = {}
            for sibling in info.siblings:
                filename = sibling.rfilename.lower()

                # Skip non-model files
                if any(
                    skip in filename
                    for skip in [".gitattribute", ".md", ".txt", ".json", ".jinja"]
                ):
                    continue

                # Extract base name (remove shard numbers like -00001-of-00015)
                if "safetensors" in filename:
                    base_name = (
                        filename.rsplit("-", 2)[0] if "-of-" in filename else filename
                    )
                    if base_name not in file_groups:
                        file_groups[base_name] = []
                    file_groups[base_name].append(sibling)
                elif "gguf" in filename:
                    base_name = filename.rsplit(".", 1)[0]
                    if base_name not in file_groups:
                        file_groups[base_name] = []
                    file_groups[base_name].append(sibling)
                elif "bin" in filename and filename.endswith(".bin"):
                    # PyTorch format (older models)
                    base_name = (
                        filename.rsplit("-", 2)[0] if "-of-" in filename else filename
                    )
                    if base_name not in file_groups:
                        file_groups[base_name] = []
                    file_groups[base_name].append(sibling)

            # Process each file group
            for base_name, siblings in file_groups.items():
                # Calculate total size (may be 0 if API doesn't return sizes)
                total_size = sum(s.size or 0 for s in siblings)
                size_gb = total_size / (1024**3) if total_size > 0 else 0.0

                # Detect quantization format
                filename = siblings[0].rfilename.lower()
                fmt = "fp16"  # default

                for q_fmt in ModelDiscovery.QUANTIZATION_FORMATS:
                    if q_fmt in filename:
                        fmt = q_fmt
                        break

                # Create variant entry
                variant_name = (
                    f"{model_id}-{fmt.upper()}" if fmt != "fp16" else model_id
                )
                variants[variant_name] = {
                    "size_gb": size_gb,
                    "format": fmt,
                    "filename": siblings[0].rfilename,
                    "shards": len(siblings) if len(siblings) > 1 else None,
                }

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

    @staticmethod
    def get_context_window(model_id: str) -> Optional[int]:
        """Get context window size from model config (in tokens).

        Returns:
            Context window in tokens, or None if unavailable.
        """
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                repo_type="model",
                cache_dir=Path.home() / ".cache" / "huggingface" / "hub",
            )
            with open(config_path) as f:
                config = json.load(f)

            # Try common context window keys
            for key in [
                "max_position_embeddings",
                "max_seq_length",
                "context_window",
                "n_ctx",
            ]:
                if key in config:
                    value = config[key]
                    if isinstance(value, int) and value > 0:
                        return value

            return None
        except Exception as e:
            logger.debug(f"Failed to get context window for {model_id}: {e}")
            return None

    @staticmethod
    def _get_cache_path() -> Path:
        """Get path to models cache file."""
        cache_dir = Path.home() / ".cache" / "llm-orchestrator"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "models-cache.json"

    @staticmethod
    def _load_cache() -> Optional[dict[str, Any]]:
        """Load cached models if fresh (< 1 hour old)."""
        cache_file = ModelDiscovery._get_cache_path()
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            timestamp = datetime.fromisoformat(data.get("timestamp", ""))
            if datetime.now() - timestamp < timedelta(hours=1):
                logger.info(f"Using cached models (updated {timestamp})")
                return data.get("models")
        except (json.JSONDecodeError, ValueError, IOError) as e:
            logger.warning(f"Failed to load cache: {e}")

        return None

    @staticmethod
    def _save_cache(models: dict[str, Any]) -> None:
        """Save models to cache."""
        cache_file = ModelDiscovery._get_cache_path()
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "models": models,
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")

    @staticmethod
    async def fetch_trending_models(
        available_vram_gb: Optional[int] = None,
        enable_llm_classification: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch trending text-generation models from HuggingFace, organized by size.

        Caches model list for 1 hour to keep fresh without API spam.
        Risk levels are calculated on-demand based on available VRAM.

        Args:
            available_vram_gb: Available GPU VRAM in GB. If None, auto-detect or use generic risk.
            enable_llm_classification: If True, classify models serially using LLM (slower but smarter).

        Returns:
            Dict with "safe", "ambitious", "experimental" tier lists.
            Each model has: name, desc, vram, risk.
        """
        # Auto-detect VRAM if not provided
        if available_vram_gb is None:
            available_vram_gb = ModelDiscovery.detect_gpu_vram()

        # Try cache first (cache doesn't include risk, so it's reusable)
        # But skip cache if smart classification is enabled (user wants fresh classification)
        cached = ModelDiscovery._load_cache() if not enable_llm_classification else None
        if cached:
            # Recalculate risks based on current VRAM
            for tier in cached.values():
                for model in tier:
                    size_gb = ModelDiscovery._estimate_model_size(model["name"])
                    if size_gb:
                        model["risk"] = ModelDiscovery._estimate_risk(
                            size_gb, available_vram_gb
                        )
            return cached

        logger.info("Fetching trending models from HuggingFace...")

        try:
            models_data = await asyncio.to_thread(
                lambda: list(
                    list_models(
                        filter="text-generation",
                        sort="downloads",
                        limit=300,  # Fetch more to get variety
                        full=True,
                    )
                )
            )
        except Exception as e:
            logger.error(f"Failed to fetch models from HuggingFace: {e}")
            return {"safe": [], "ambitious": [], "experimental": []}

        if enable_llm_classification:
            logger.info("🧠 LLM classification enabled - processing models serially...")
            logger.info("Loading LLM model (first and only time)...")
            from transformers import pipeline

            classifier = await asyncio.to_thread(
                lambda: pipeline(
                    "text-generation",
                    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    device="cpu",
                )
            )
            logger.info("✓ Model loaded")
        else:
            classifier = None

        # Extract model info and categorize by size
        safe_models: list[dict[str, Any]] = []
        ambitious_models: list[dict[str, Any]] = []
        experimental_models: list[dict[str, Any]] = []

        # Score all models by momentum (recent popularity)
        scored_models = []
        for model in models_data:
            momentum = ModelDiscovery.calculate_momentum_score(model)
            scored_models.append((model, momentum))

        # Sort by momentum (higher is better - recent adoption preferred)
        scored_models.sort(key=lambda x: x[1], reverse=True)

        for model, momentum in scored_models:
            model_id = model.modelId
            downloads = getattr(model, "downloads", 0) or 0

            # Skip models older than 12 months (keep proven models)
            created_at = getattr(model, "created_at", None)
            if created_at:
                try:
                    # Parse ISO format date (handle both aware and naive datetimes)
                    if isinstance(created_at, str):
                        created_dt = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                    else:
                        created_dt = created_at

                    # Make comparison timezone-aware
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)

                    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
                    if created_dt < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass

            # Infer model size from name
            size_gb = ModelDiscovery._estimate_model_size(model_id)
            if size_gb is None:
                continue

            # Skip tiny models (< 1B)
            if size_gb < 1:
                continue

            # Use fast heuristics first (name-based filtering)
            is_instruct = any(
                x in model_id for x in ["Instruct", "Chat", "chat", "instruct"]
            )
            is_popular = (
                downloads > 50000
            )  # Much lower threshold; momentum scoring handles recency

            if not (is_instruct or is_popular):
                continue

            # Optionally classify with LLM (serial, one at a time)
            model_type = "unknown"
            if enable_llm_classification:
                try:
                    from huggingface_hub import hf_hub_download

                    logger.info(f"  Classifying {model_id}...")
                    readme_path = await asyncio.to_thread(
                        lambda: hf_hub_download(
                            model_id, "README.md", repo_type="model"
                        )
                    )
                    with open(readme_path) as f:
                        readme_content = f.read()
                    classification = await ModelDiscovery.classify_model_with_llm(
                        model_id, readme_content, classifier=classifier
                    )
                    model_type = classification.get("model_type", "unknown")
                    logger.info(f"    → {model_type}")
                except Exception as e:
                    logger.info(f"  Failed to classify {model_id}: {e}")
                    model_type = "unknown"
            else:
                # Always do at least name-based heuristic classification
                if any(x in model_id for x in ["Instruct", "Chat", "chat", "instruct"]):
                    model_type = "instruct"
                else:
                    model_type = "base"

            # Build better description
            desc = ModelDiscovery._generate_description(model_id, size_gb)

            # Build model entry (risk calculated on-demand based on VRAM)
            entry = {
                "name": model_id,
                "desc": desc,
                "vram": f"{int(size_gb * 2)}GB",  # Rough 2x for inference
                "size_gb": size_gb,  # Store for risk recalculation
                "risk": ModelDiscovery._estimate_risk(size_gb, available_vram_gb),
                "context": ModelDiscovery.get_context_window(
                    model_id
                ),  # Context window in tokens
                "model_type": model_type,  # Classification (or "unknown" if disabled)
            }

            # Categorize by size - more aggressive tiers
            if size_gb <= 8:
                safe_models.append(entry)
            elif size_gb <= 40:
                ambitious_models.append(entry)
            else:
                experimental_models.append(entry)

        # Limit each tier but allow more experimental options
        result = {
            "safe": safe_models[:10],
            "ambitious": ambitious_models[:10],
            "experimental": experimental_models[:15],  # More experimental options
        }

        # Cache result (without risk, so it's reusable with different VRAM configs)
        cache_result = {
            "safe": [
                {k: v for k, v in m.items() if k != "size_gb"} for m in result["safe"]
            ],
            "ambitious": [
                {k: v for k, v in m.items() if k != "size_gb"}
                for m in result["ambitious"]
            ],
            "experimental": [
                {k: v for k, v in m.items() if k != "size_gb"}
                for m in result["experimental"]
            ],
        }
        ModelDiscovery._save_cache(cache_result)
        logger.info(
            f"Fetched {len(safe_models)} safe, {len(ambitious_models)} ambitious, "
            f"{len(experimental_models)} experimental models"
        )
        return result

    @staticmethod
    def _estimate_model_size(model_id: str) -> Optional[float]:
        """Estimate model size in GB from model ID name heuristics."""
        # Extract size hints from model names like "7B", "13B", "70B"
        import re

        size_match = re.search(r"(\d+(?:\.\d+)?)[bB]", model_id)
        if size_match:
            size_b = float(size_match.group(1))
            # Rough estimation: 1B params ≈ 2-3 GB (fp16) or 1-1.5 GB (int8/q4)
            if "q4" in model_id.lower() or "q5" in model_id.lower():
                return size_b * 1.2
            elif "int8" in model_id.lower():
                return size_b * 1.5
            else:
                return size_b * 2

        return None

    @staticmethod
    def get_benchmarks(model_id: str) -> dict[str, Any]:
        """Fetch benchmark scores from model card (with caching).

        Returns:
            Dict with benchmark categories and scores.
        """
        # Check cache first
        cache_dir = Path.home() / ".cache" / "llm-orchestrator" / "benchmarks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{model_id.replace('/', '_')}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if datetime.fromisoformat(
                        data["timestamp"]
                    ) > datetime.now() - timedelta(hours=24):
                        logger.debug(f"Using cached benchmarks for {model_id}")
                        return data["benchmarks"]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        try:
            import re

            from huggingface_hub import hf_hub_download

            readme_path = hf_hub_download(model_id, "README.md", repo_type="model")
            with open(readme_path) as f:
                content = f.read().lower()

            benchmarks: dict[str, Any] = {}

            # More flexible pattern matching for benchmarks
            patterns = {
                "reasoning": [
                    r"mmlu[:\s]+([0-9.]+)",
                    r"arc[:\s]*([0-9.]+)",
                    r"hellaswag[:\s]+([0-9.]+)",
                    r"\|\s*mmlu\s*\|\s*([0-9.]+)",
                    r"(?:mmlu|arc|hellaswag)[^\n]*?([0-9]+\.[0-9]{1,2}|[0-9]{1,3}%)",
                ],
                "math": [
                    r"(?:gsm8k|math|aime)[:\s]+([0-9.]+)",
                    r"mathematics[:\s]+([0-9.]+)",
                    r"(?:gsm8k|math)[^\n]*?([0-9]+\.[0-9]{1,2}|[0-9]{1,3}%)",
                ],
                "coding": [
                    r"humaneval[:\s]+([0-9.]+)",
                    r"mbpp[:\s]+([0-9.]+)",
                    r"coding[:\s]+([0-9.]+)",
                    r"(?:humaneval|mbpp)[^\n]*?([0-9]+\.[0-9]{1,2}|[0-9]{1,3}%)",
                ],
                "tool_use": [
                    r"tool[\s_-]*use[:\s]+([0-9.]+)",
                    r"function[\s_-]*call[:\s]+([0-9.]+)",
                ],
                "agentic": [
                    r"agent[ic]*[:\s]+([0-9.]+)",
                    r"react[:\s]+([0-9.]+)",
                ],
                "engineering": [
                    r"swe[\s_-]*bench[:\s]+([0-9.]+)",
                    r"engineering[:\s]+([0-9.]+)",
                ],
                "medicine": [
                    r"medqa[:\s]+([0-9.]+)",
                    r"medmcqa[:\s]+([0-9.]+)",
                    r"medicine[:\s]+([0-9.]+)",
                ],
            }

            for category, pattern_list in patterns.items():
                scores = []
                for pattern in pattern_list:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    scores.extend([float(m) for m in matches])

                if scores:
                    benchmarks[category] = {
                        "score": round(sum(scores) / len(scores), 1),
                        "count": len(scores),
                    }

            # Cache the result
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "benchmarks": benchmarks,
            }
            try:
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f)
            except IOError:
                pass

            return benchmarks
        except Exception as e:
            logger.debug(f"Could not fetch benchmarks for {model_id}: {e}")
            return {}
        """Fetch context window size in tokens from model config.

        Returns:
            Context window in tokens (e.g., 4096, 8192), or None if unavailable.
        """
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(model_id, "config.json", repo_type="model")
            with open(config_path) as f:
                config = json.load(f)

            # Try common context window field names
            for field in [
                "max_position_embeddings",
                "context_length",
                "max_seq_length",
                "seq_length",
            ]:
                if field in config:
                    return int(config[field])

            return None
        except Exception as e:
            logger.debug(f"Could not fetch context window for {model_id}: {e}")
            return None

    @staticmethod
    def _estimate_risk(size_gb: float, available_vram_gb: Optional[int] = None) -> str:
        """Estimate risk level based on model size and available VRAM.

        Args:
            size_gb: Estimated model size in GB
            available_vram_gb: Available GPU VRAM in GB (if known)
        """
        if available_vram_gb:
            # Adjust risk based on actual hardware (2x overhead for inference)
            estimated_vram_needed = size_gb * 2
            utilization = estimated_vram_needed / available_vram_gb
            if utilization <= 0.6:
                return f"✓ Low risk - fits comfortably ({utilization * 100:.0f}% VRAM)"
            elif utilization <= 0.85:
                return f"⚠ Medium risk - works but may need tuning ({utilization * 100:.0f}% VRAM)"
            elif utilization <= 0.95:
                return f"🟡 Ambitious - likely works with tuning ({utilization * 100:.0f}% VRAM)"
            else:
                return f"❌ High risk - unlikely to fit ({utilization * 100:.0f}% VRAM)"
        else:
            # Generic risk assessment (no hardware info)
            if size_gb <= 10:
                return "✓ Low risk"
            elif size_gb <= 35:
                return "⚠ Medium risk - may OOM on smaller GPUs"
            else:
                return "❌ High risk - requires high-end GPU"

    @staticmethod
    async def classify_model_with_llm(
        model_id: str, readme_content: str, classifier: Optional[Any] = None
    ) -> dict[str, Any]:
        """Use LLM to classify model type and extract details from README.

        Args:
            model_id: Model ID to classify
            readme_content: README content to analyze
            classifier: Optional pre-loaded pipeline to reuse (avoids reloading model)

        Returns:
            Dict with model_type, is_useful.
        """
        try:
            import json
            import re

            # Load model if not provided
            if classifier is None:
                from transformers import pipeline

                classifier = await asyncio.to_thread(
                    lambda: pipeline(
                        "text-generation",
                        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        device="cpu",
                    )
                )

            # Create classification prompt
            prompt = f"""Analyze this model card briefly:

Model ID: {model_id}
Content: {readme_content[:1500]}

Respond ONLY with JSON:
{{"model_type": "base|instruct|reasoning|chat|specialized", "is_useful": true}}"""

            response = await asyncio.to_thread(
                lambda: classifier(prompt, max_new_tokens=200, temperature=0.3)
            )

            full_text = response[0]["generated_text"]
            # Extract only the generated part (after the prompt)
            # Find the last occurrence of the prompt's end marker (the JSON spec line)
            prompt_end = full_text.rfind('is_useful": true')
            if prompt_end == -1:
                # If we can't find the marker, look for the first { after "Respond ONLY"
                respond_idx = full_text.find("Respond ONLY with JSON:")
                if respond_idx != -1:
                    text = full_text[respond_idx + len("Respond ONLY with JSON:"):]
                else:
                    text = full_text
            else:
                text = full_text[prompt_end + len('is_useful": true'):]

            logger.info(f"  Generated: {text[:150]}")
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    logger.info(f"  → {parsed.get('model_type', 'unknown')}")
                    return parsed
                except json.JSONDecodeError as je:
                    logger.info(f"  JSON parse error: {je}")

            # Fallback: simple heuristic classification
            content_lower = readme_content.lower()
            is_chat = any(
                x in content_lower for x in ["chat", "conversational", "dialogue"]
            )
            is_instruct = any(
                x in content_lower
                for x in ["instruction", "instruct", "instruction-tuned"]
            )

            model_type = (
                "chat"
                if is_chat
                else ("instruct" if is_instruct else "base")
            )
            logger.info(f"  → heuristic: {model_type}")
            return {
                "model_type": model_type,
                "is_useful": True,
            }
        except Exception as e:
            logger.info(f"  ✗ Classification failed: {e}")
            return {"model_type": "unknown", "is_useful": False}

    @staticmethod
    def detect_gpu_vram() -> Optional[int]:
        """Detect available GPU VRAM in GB.

        Returns:
            Total VRAM in GB (sum of all GPUs), or None if detection fails.
        """
        # Try nvidia-smi first (NVIDIA GPUs)
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,nounits,noheader",
                ],
                text=True,
            )
            total_mb = sum(
                int(line.strip()) for line in output.strip().split("\n") if line.strip()
            )
            total_gb = total_mb // 1024
            logger.info(f"Detected {total_gb}GB GPU VRAM (via nvidia-smi)")
            return total_gb
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            pass

        # Try PyTorch
        try:
            import torch

            if torch.cuda.is_available():
                total_gb = sum(
                    torch.cuda.get_device_properties(i).total_memory // (1024**3)
                    for i in range(torch.cuda.device_count())
                )
                logger.info(f"Detected {total_gb}GB GPU VRAM (via torch.cuda)")
                return total_gb
        except Exception:
            pass

        logger.warning("Could not detect GPU VRAM. Use --gpu-vram to specify manually.")
        return None

    @staticmethod
    def _generate_description(model_id: str, size_gb: float) -> str:
        """Generate a helpful description from model name and size."""
        org = model_id.split("/")[0] if "/" in model_id else "Unknown"
        if size_gb <= 3:
            return f"{org} - Fast, lightweight model (good for quick testing)"
        elif size_gb <= 8:
            return f"{org} - Efficient model, good balance of speed and quality"
        elif size_gb <= 15:
            return f"{org} - Solid reasoning, higher quality outputs"
        elif size_gb <= 35:
            return f"{org} - Strong reasoning capabilities, better quality"
        else:
            return f"{org} - Frontier model, excellent reasoning but resource-intensive"

    @staticmethod
    def calculate_momentum_score(model: Any) -> float:
        """Calculate download momentum (recent popularity) score.

        Rewards recent models with good adoption over old models with stale downloads.
        Formula: downloads / days_old (caps age at 365 so old models don't dominate)
        """
        downloads = getattr(model, "downloads", 1) or 1
        created_at = getattr(model, "created_at", None)

        if not created_at:
            return float(downloads)

        try:
            if isinstance(created_at, str):
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                created_dt = created_at

            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)

            days_old = max(1, (datetime.now(timezone.utc) - created_dt).days)
            # After 365 days, cap the age so old models don't score disproportionately low
            days_for_scoring = min(days_old, 365)

            momentum = downloads / days_for_scoring
            return momentum
        except Exception:
            return float(downloads)

    @staticmethod
    def _get_classification_cache_path() -> Path:
        """Get path to model cache file (README + classifications)."""
        config_dir = Path.home() / ".config" / "llm-orchestrator"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "model-cache.json"

    @staticmethod
    def _load_model_cache() -> dict[str, dict[str, Any]]:
        """Load model cache (README + classifications, no expiration)."""
        cache_path = ModelDiscovery._get_classification_cache_path()
        if not cache_path.exists():
            return {}

        try:
            with open(cache_path) as f:
                cache_data = json.load(f)
            return cache_data.get("models", {})
        except (json.JSONDecodeError, IOError):
            return {}

    @staticmethod
    def _save_model_cache(models: dict[str, dict[str, Any]]) -> None:
        """Save model cache with README + classifications (indefinite storage)."""
        cache_path = ModelDiscovery._get_classification_cache_path()
        cache_data = {
            "_note": "README + classification cache. No expiration - READMEs don't change.",
            "models": models,
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save model cache: {e}")

    @staticmethod
    async def _fetch_model_readme(model_id: str) -> str:
        """Fetch model README from cache or HuggingFace hub.

        Checks local cache first, then HF hub cache, then fetches fresh.
        """
        # Check our cache first
        cache = ModelDiscovery._load_model_cache()
        if model_id in cache and "readme_content" in cache[model_id]:
            logger.debug(f"README cache hit for {model_id}")
            return cache[model_id]["readme_content"]

        # Fetch from HF hub (which caches locally)
        try:
            readme_path = await asyncio.to_thread(
                lambda: hf_hub_download(
                    repo_id=model_id,
                    filename="README.md",
                    repo_type="model",
                    cache_dir=Path.home() / ".cache" / "huggingface" / "hub",
                )
            )
            with open(readme_path) as f:
                content = f.read()

            # Cache it in our model cache
            if model_id not in cache:
                cache[model_id] = {}
            cache[model_id]["readme_content"] = content
            ModelDiscovery._save_model_cache(cache)

            return content
        except Exception as e:
            logger.debug(f"Failed to fetch README for {model_id}: {e}")
            return ""

    @staticmethod
    async def get_or_classify_model(model_id: str) -> dict[str, Any]:
        """Get model classification from cache or compute via LLM.

        Stores both README and classification together indefinitely.

        Returns:
            Dict with model_type (base/instruct/chat/specialized) and is_useful (bool).
        """
        # Try cache first
        cache = ModelDiscovery._load_model_cache()
        if model_id in cache and "classification" in cache[model_id]:
            logger.debug(f"Classification cache hit for {model_id}")
            return cache[model_id]["classification"]

        logger.debug(f"Computing classification for {model_id}...")

        # Fetch README
        readme_content = await ModelDiscovery._fetch_model_readme(model_id)
        if not readme_content:
            # Fallback to heuristic
            is_instruct = any(
                x in model_id for x in ["Instruct", "Chat", "chat", "instruct"]
            )
            result = {
                "model_type": "instruct" if is_instruct else "base",
                "is_useful": True,
            }
        else:
            # Use LLM classification
            result = await ModelDiscovery.classify_model_with_llm(model_id, readme_content)

        # Update cache with both README and classification
        if result.get("is_useful"):
            if model_id not in cache:
                cache[model_id] = {}
            cache[model_id]["classification"] = result
            # README already cached by _fetch_model_readme
            ModelDiscovery._save_model_cache(cache)

        return result

        return result
