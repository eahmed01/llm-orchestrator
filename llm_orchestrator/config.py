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

    # Backward-compat: single vllm field
    vllm: Optional[ServiceConfig] = Field(
        None,
        description="Configuration for main vLLM service",
    )

    # Multi-service: ordered dict of service name -> ServiceConfig
    services: dict[str, ServiceConfig] = Field(
        default_factory=dict,
        description="Named service configurations",
    )

    # Desired state: persist the desired stack state per service
    desired_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Desired state per service (persisted across runs)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "vllm": {
                    "model": "Qwen/Qwen3.6-27B-FP8",
                    "args": {"--max-num-seqs": 848},
                    "last_successful": "2026-05-07T12:35:00Z",
                },
                "services": {
                    "vllm": {
                        "model": "Qwen/Qwen3.6-27B-FP8",
                        "args": {"--max-num-seqs": 848},
                    },
                    "reranker": {
                        "model": "Qwen/Qwen3-Reranker-0.6B",
                        "args": {"--max-model-len": 1024},
                    },
                },
                "desired_state": {
                    "vllm": {"gpus": [0, 1]},
                    "reranker": {"gpus": [1]},
                },
            }
        }
    }

    # ------------------------------------------------------------------
    # Config file paths
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    @classmethod
    def load_from_disk(cls) -> "OrchestratorConfig":
        """Load config from ~/.config/llm-orchestrator/working-models.yaml."""
        config_file = cls.config_file()

        if not config_file.exists():
            return cls(vllm=None)  # Return empty config

        try:
            with open(config_file, "r") as f:
                data = yaml.safe_load(f) or {}

            # Backward compat: migrate bare vllm key into services if needed
            if "services" not in data and data.get("vllm"):
                try:
                    svc_data = data["vllm"]
                    if isinstance(svc_data, dict):
                        data.setdefault("services", {})["vllm"] = svc_data
                except (TypeError, KeyError):
                    pass

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

    def save(self) -> None:
        """Save config to disk. Alias for save_to_disk()."""
        self.save_to_disk()

    # ------------------------------------------------------------------
    # Service config access (backward compat + multi-service)
    # ------------------------------------------------------------------

    def get_service_config(self, service: str) -> Optional[ServiceConfig]:
        """Get config for a specific service.

        Checks the legacy ``vllm`` field first, then the ``services`` dict.
        """
        if service == "vllm":
            return self.vllm
        return self.services.get(service)

    def set_service_config(self, service: str, config: Optional[ServiceConfig]) -> None:
        """Set config for a specific service.

        Writes to both the legacy ``vllm`` field (if service=="vllm")
        and the ``services`` dict.
        """
        if service == "vllm":
            self.vllm = config
        self.services[service] = config

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

    def list_services(self) -> list[str]:
        """Return all configured service names (vllm + services dict)."""
        names: list[str] = []
        if self.vllm is not None:
            names.append("vllm")
        for name in self.services:
            if name not in names:
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Desired state helpers
    # ------------------------------------------------------------------

    def get_desired(self, service: str) -> dict[str, Any]:
        """Get the persisted desired state for a service.

        Returns an empty dict if nothing is recorded.
        """
        return self.desired_state.get(service, {})

    def set_desired(self, service: str, state: dict[str, Any]) -> None:
        """Set the persisted desired state for a service."""
        self.desired_state[service] = state

    def clear_desired(self, service: str) -> None:
        """Remove the desired state for a service."""
        self.desired_state.pop(service, None)

    def save_desired(self) -> None:
        """Persist the desired_state to disk immediately."""
        self.save()

    # ------------------------------------------------------------------
    # User preferences (unchanged)
    # ------------------------------------------------------------------

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
