"""Profile index: host-model keyed history of startup attempts and known-good configs.

Tracks everything tried for each model on this host — what worked, what failed,
exact args, env vars, GPU assignments. Enables instant restart reuse and
informed fallback when trying new models.

Stored at ~/.config/llm-orchestrator/profiles.json
"""

import json
import logging
import socket
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from llm_orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)


@dataclass
class Attempt:
    """A single startup attempt for a model on this host."""

    ts: str
    model: str
    args: dict[str, Any]
    env: dict[str, str]
    gpus: list[int]
    port: int
    interface: str
    success: bool
    failure_reason: Optional[str]
    duration_s: Optional[float]
    vllm_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Attempt":
        return cls(**d)


@dataclass
class ModelProfile:
    """Persistent profile for a single model on this host."""

    model: str
    host: str
    size_hint: Optional[float] = None  # e.g. 27.0 for 27B
    quantization: Optional[str] = None  # e.g. "fp8", "q4_k_m"
    family: Optional[str] = None  # e.g. "qwen"
    known_good: Optional[dict[str, Any]] = None  # best known working config
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    updated: str = ""

    def add_attempt(self, attempt: Attempt) -> None:
        """Record an attempt. If successful, update known_good."""
        att_dict = attempt.to_dict()
        self.attempts.append(att_dict)

        # Keep max 100 attempts per profile (trim oldest)
        if len(self.attempts) > 100:
            self.attempts = self.attempts[-100:]

        if attempt.success and self.known_good is None:
            self.known_good = {
                "args": attempt.args,
                "env": attempt.env,
                "gpus": attempt.gpus,
                "port": attempt.port,
                "interface": attempt.interface,
                "ts": attempt.ts,
                "duration_s": attempt.duration_s,
                "vllm_version": attempt.vllm_version,
            }

        self.updated = datetime.now(timezone.utc).isoformat()

    def get_known_good(self) -> Optional[dict[str, Any]]:
        """Return the best known working config, or None."""
        return self.known_good

    def get_recent_attempts(
        self, limit: int = 10, only_failures: bool = False
    ) -> list[Attempt]:
        """Return recent attempts."""
        attempts = [Attempt.from_dict(a) for a in self.attempts]
        attempts.sort(key=lambda a: a.ts, reverse=True)
        if only_failures:
            attempts = [a for a in attempts if not a.success]
        return attempts[:limit]

    def failure_count(self) -> int:
        return sum(1 for a in self.attempts if not a.get("success", True))

    def success_count(self) -> int:
        return sum(1 for a in self.attempts if a.get("success", False))

    def has_known_good(self) -> bool:
        return self.known_good is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelProfile":
        return cls(**d)


def _profile_path() -> Path:
    config_dir = OrchestratorConfig.config_dir()
    return config_dir / "profiles.json"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _extract_model_meta(model_id: str) -> dict[str, Optional[str]]:
    """Extract family and size hint from model ID."""
    import re

    family = None
    size_hint = None
    quantization = None

    # Family detection
    lower = model_id.lower()
    if "qwen" in lower:
        family = "qwen"
    elif "llama" in lower or "meta-llama" in lower:
        family = "llama"
    elif "mistral" in lower:
        family = "mistral"
    elif "deepseek" in lower:
        family = "deepseek"

    # Size extraction
    size_match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
    if size_match:
        size_hint = float(size_match.group(1))

    # Quantization
    for q in ["fp8", "q4_k_m", "q4_0", "q5_k_m", "q8_0", "awq", "gptq"]:
        if q in lower:
            quantization = q
            break

    return {
        "family": family,
        "size_hint": size_hint,
        "quantization": quantization,
    }


class ProfileStore:
    """Persistent store of model profiles, keyed by host:model."""

    def __init__(self) -> None:
        self._path = _profile_path()
        self._host = _hostname()
        self._profiles: dict[str, ModelProfile] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for key, prof_data in data.get("profiles", {}).items():
                self._profiles[key] = ModelProfile.from_dict(prof_data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load profiles: %s", e)

    def save(self) -> None:
        try:
            data = {
                "host": self._host,
                "updated": datetime.now(timezone.utc).isoformat(),
                "profiles": {k: v.to_dict() for k, v in self._profiles.items()},
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.warning("Failed to save profiles: %s", e)

    def _key(self, model: str) -> str:
        return f"{self._host}:{model}"

    def get_profile(self, model: str) -> Optional[ModelProfile]:
        """Get profile for a model on this host."""
        return self._profiles.get(self._key(model))

    def get_or_create(self, model: str) -> ModelProfile:
        """Get or create profile for a model."""
        key = self._key(model)
        if key not in self._profiles:
            meta = _extract_model_meta(model)
            self._profiles[key] = ModelProfile(
                model=model,
                host=self._host,
                size_hint=meta["size_hint"],
                quantization=meta["quantization"],
                family=meta["family"],
                updated=datetime.now(timezone.utc).isoformat(),
            )
        return self._profiles[key]

    def record_attempt(self, attempt: Attempt) -> ModelProfile:
        """Record a startup attempt and update the profile."""
        profile = self.get_or_create(attempt.model)
        profile.add_attempt(attempt)
        self.save()
        return profile

    def get_known_good(self, model: str) -> Optional[dict[str, Any]]:
        """Get known-good config for a model on this host."""
        profile = self.get_profile(model)
        if profile:
            return profile.get_known_good()
        return None

    def find_similar(
        self, model: str, family: Optional[str] = None, size_range: float = 10.0
    ) -> list[tuple[str, ModelProfile]]:
        """Find profiles for similar models (same family, similar size).

        Returns list of (key, profile) tuples sorted by size similarity.
        Only includes profiles with known_good configs.
        """
        meta = _extract_model_meta(model)
        target_family = family or meta["family"]
        target_size = meta["size_hint"]

        results: list[tuple[str, ModelProfile]] = []
        for key, profile in self._profiles.items():
            if profile.model == model:
                continue  # skip exact match
            if not profile.has_known_good():
                continue
            if target_family and profile.family != target_family:
                continue
            if target_size and profile.size_hint:
                if abs(profile.size_hint - target_size) > size_range:
                    continue
            results.append((key, profile))

        # Sort by size similarity
        if target_size:
            results.sort(key=lambda x: abs((x[1].size_hint or 0) - target_size))

        return results

    def get_all_profiles(self) -> dict[str, ModelProfile]:
        """Return all profiles."""
        return dict(self._profiles)

    def get_profile_stats(self) -> dict[str, Any]:
        """Summary statistics across all profiles."""
        total = len(self._profiles)
        with_known_good = sum(1 for p in self._profiles.values() if p.has_known_good())
        total_attempts = sum(len(p.attempts) for p in self._profiles.values())
        total_successes = sum(p.success_count() for p in self._profiles.values())
        total_failures = sum(p.failure_count() for p in self._profiles.values())

        # Group by family
        families: dict[str, int] = {}
        for p in self._profiles.values():
            fam = p.family or "unknown"
            families[fam] = families.get(fam, 0) + 1

        return {
            "host": self._host,
            "total_profiles": total,
            "with_known_good": with_known_good,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "families": families,
            "path": str(self._path),
        }


# Module-level singleton
_store: Optional[ProfileStore] = None


def get_store() -> ProfileStore:
    global _store
    if _store is None:
        _store = ProfileStore()
    return _store


def reset_store() -> None:
    """Reset the singleton (for testing)."""
    global _store
    _store = None
