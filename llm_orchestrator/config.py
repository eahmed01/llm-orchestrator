"""Config: persistence and validation using Pydantic + YAML."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """User preferences for GPU, port, and interface selection."""

    preferred_gpu: Optional[int] = Field(
        None,
        description="Preferred GPU index",
    )
    preferred_port: int = Field(
        default=7999,
        description="Preferred port for vLLM",
    )
    preferred_interface: str = Field(
        default="127.0.0.1",
        description="Preferred network interface",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "preferred_gpu": 1,
                "preferred_port": 7999,
                "preferred_interface": "0.0.0.0",
            }
        }
    }


class ServiceConfig(BaseModel):
    """Configuration for a managed LLM service."""

    model: str = Field(..., description="HuggingFace model ID")
    variant: Optional[str] = Field(
        None, description="Quantization variant (e.g., 'q4', 'fp8')"
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="vLLM startup arguments",
    )
    last_successful: Optional[str] = Field(
        None,
        description="ISO timestamp of last successful startup",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "Qwen/Qwen3.6-27B-FP8",
                "variant": None,
                "args": {
                    "--max-num-seqs": 848,
                    "--gpu-memory-utilization": 0.92,
                },
                "last_successful": "2026-05-07T12:35:00Z",
            }
        }
    }


class OrchestratorConfig(BaseModel):
    """Top-level orchestrator configuration."""

    vllm: Optional[ServiceConfig] = Field(
        None,
        description="Configuration for main vLLM service",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "vllm": {
                    "model": "Qwen/Qwen3.6-27B-FP8",
                    "args": {"--max-num-seqs": 848},
                    "last_successful": "2026-05-07T12:35:00Z",
                }
            }
        }
    }

    @classmethod
    def config_dir(cls) -> Path:
        """Get orchestrator config directory."""
        config_dir = Path.home() / ".config" / "llm-orchestrator"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    @classmethod
    def config_file(cls) -> Path:
        """Get path to working-models.yaml."""
        return cls.config_dir() / "working-models.yaml"

    @classmethod
    def load_from_disk(cls) -> "OrchestratorConfig":
        """Load config from ~/.config/llm-orchestrator/working-models.yaml."""
        config_file = cls.config_file()

        if not config_file.exists():
            return cls(vllm=None)  # Return empty config

        try:
            with open(config_file, "r") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_file}: {e}")

    def save_to_disk(self) -> None:
        """Save config to ~/.config/llm-orchestrator/working-models.yaml."""
        config_file = self.config_file()

        try:
            # Convert to dict for YAML serialization
            data = json.loads(self.model_dump_json())
            with open(config_file, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {config_file}: {e}")

    def get_service_config(self, service: str) -> Optional[ServiceConfig]:
        """Get config for a specific service."""
        if service == "vllm":
            return self.vllm
        raise ValueError(f"Unknown service: {service}")

    def set_service_config(self, service: str, config: ServiceConfig) -> None:
        """Set config for a specific service."""
        if service == "vllm":
            self.vllm = config
        else:
            raise ValueError(f"Unknown service: {service}")

    def record_success(
        self,
        service: str,
        model: str,
        variant: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a successful startup."""
        from datetime import timezone

        config = ServiceConfig(
            model=model,
            variant=variant,
            args=args or {},
            last_successful=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        self.set_service_config(service, config)
        self.save_to_disk()

    @classmethod
    def preferences_file(cls) -> Path:
        """Get path to user-preferences.json."""
        return cls.config_dir() / "user-preferences.json"

    @classmethod
    def load_preferences(cls) -> UserPreferences:
        """Load user preferences from disk."""
        pref_file = cls.preferences_file()

        if not pref_file.exists():
            return UserPreferences()

        try:
            with open(pref_file) as f:
                data = json.load(f)
            return UserPreferences(**data)
        except Exception:
            return UserPreferences()

    @classmethod
    def save_preferences(cls, preferences: UserPreferences) -> None:
        """Save user preferences to disk."""
        pref_file = cls.preferences_file()

        try:
            with open(pref_file, "w") as f:
                json.dump(preferences.model_dump(), f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save preferences: {e}")
