"""Tests for llm_orchestrator.config module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest
import yaml

from llm_orchestrator.config import OrchestratorConfig, ServiceConfig, UserPreferences


class TestServiceConfig:
    """Tests for ServiceConfig."""

    def test_service_config_creation(self):
        config = ServiceConfig(
            model="Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
        )
        assert config.model == "Qwen/Qwen3.6-27B-FP8"
        assert config.args["--max-num-seqs"] == 848

    def test_service_config_with_variant(self):
        config = ServiceConfig(
            model="Qwen/Qwen3.6-27B",
            variant="q4_k_m",
        )
        assert config.variant == "q4_k_m"


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_load_empty_config(self):
        with patch(
            "llm_orchestrator.config.OrchestratorConfig.config_file"
        ) as mock_file:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_file.return_value = mock_path

            config = OrchestratorConfig.load_from_disk()
            assert config.vllm is None

    def test_config_dir_creation(self):
        config_dir = OrchestratorConfig.config_dir()
        assert config_dir.exists()

    def test_record_success(self):
        config = OrchestratorConfig()
        config.record_success(
            "vllm",
            "Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
        )

        assert config.vllm is not None
        assert config.vllm.model == "Qwen/Qwen3.6-27B-FP8"
        assert config.vllm.last_successful is not None

    def test_load_backward_compat_migration(self, tmp_path):
        """Test that bare vllm key is migrated to services dict."""
        config_file = tmp_path / "working-models.yaml"
        config_file.write_text(yaml.dump({
            "vllm": {
                "model": "Qwen/Qwen3.6-27B-FP8",
                "args": {"--max-num-seqs": 848},
            }
        }))

        with patch.object(OrchestratorConfig, "config_file", return_value=config_file):
            config = OrchestratorConfig.load_from_disk()
        # Should have migrated to services
        assert config.vllm is not None
        assert config.services.get("vllm") is not None

    def test_load_from_disk_error(self, tmp_path):
        """Test RuntimeError on corrupt config."""
        config_file = tmp_path / "working-models.yaml"
        config_file.write_text("{invalid yaml content [[[")

        with patch.object(OrchestratorConfig, "config_file", return_value=config_file):
            with pytest.raises(RuntimeError, match="Failed to load config"):
                OrchestratorConfig.load_from_disk()

    def test_save_to_disk_and_reload(self, tmp_path):
        """Test save_to_disk writes valid YAML that can be reloaded."""
        config_file = tmp_path / "working-models.yaml"
        config = OrchestratorConfig()
        config.vllm = ServiceConfig(model="test/model")

        with patch.object(OrchestratorConfig, "config_file", return_value=config_file):
            config.save_to_disk()

        assert config_file.exists()
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_save_to_disk_error(self, tmp_path):
        """Test RuntimeError when save fails."""
        config = OrchestratorConfig()
        bad_path = tmp_path / "sub" / "file.yaml"
        with patch.object(OrchestratorConfig, "config_file", return_value=bad_path):
            with patch("builtins.open", side_effect=PermissionError("denied")):
                with pytest.raises(RuntimeError):
                    config.save_to_disk()

    def test_save_alias(self, tmp_path):
        """Test that save() is an alias for save_to_disk()."""
        config_file = tmp_path / "working-models.yaml"
        config = OrchestratorConfig()
        config.vllm = ServiceConfig(model="test/model")

        with patch.object(OrchestratorConfig, "config_file", return_value=config_file):
            config.save()
        assert config_file.exists()

    def test_get_service_config_vllm(self):
        config = OrchestratorConfig()
        config.vllm = ServiceConfig(model="test/model")
        result = config.get_service_config("vllm")
        assert result is not None
        assert result.model == "test/model"

    def test_get_service_config_other(self):
        config = OrchestratorConfig()
        config.services["reranker"] = ServiceConfig(model="reranker/model")
        result = config.get_service_config("reranker")
        assert result is not None
        assert result.model == "reranker/model"

    def test_set_service_config_vllm(self):
        config = OrchestratorConfig()
        svc = ServiceConfig(model="test/model")
        config.set_service_config("vllm", svc)
        assert config.vllm is not None
        assert config.vllm.model == "test/model"
        assert config.services["vllm"].model == "test/model"

    def test_set_service_config_other(self):
        config = OrchestratorConfig()
        svc = ServiceConfig(model="reranker/model")
        config.set_service_config("reranker", svc)
        assert config.services["reranker"].model == "reranker/model"

    def test_list_services_empty(self):
        config = OrchestratorConfig()
        assert config.list_services() == []

    def test_list_services_with_vllm(self):
        config = OrchestratorConfig()
        config.vllm = ServiceConfig(model="test/model")
        names = config.list_services()
        assert "vllm" in names

    def test_list_services_with_multiple(self):
        config = OrchestratorConfig()
        config.vllm = ServiceConfig(model="vllm/model")
        config.services["reranker"] = ServiceConfig(model="reranker/model")
        names = config.list_services()
        assert "vllm" in names
        assert "reranker" in names

    def test_desired_state_helpers(self):
        config = OrchestratorConfig()
        # Initially empty
        assert config.get_desired("vllm") == {}

        # Set desired state
        config.set_desired("vllm", {"gpus": [0, 1]})
        assert config.get_desired("vllm") == {"gpus": [0, 1]}

        # Clear desired state
        config.clear_desired("vllm")
        assert config.get_desired("vllm") == {}

    def test_save_desired(self, tmp_path):
        """Test save_desired persists to disk."""
        config_file = tmp_path / "working-models.yaml"
        config = OrchestratorConfig()
        config.set_desired("vllm", {"gpus": [0]})

        with patch.object(OrchestratorConfig, "config_file", return_value=config_file):
            config.save_desired()
        assert config_file.exists()


class TestUserPreferences:
    """Tests for UserPreferences and config preference helpers."""

    def test_load_preferences_no_file(self, tmp_path):
        """Test loading preferences when file doesn't exist."""
        pref_file = tmp_path / "user-preferences.json"

        with patch.object(OrchestratorConfig, "preferences_file", return_value=pref_file):
            prefs = OrchestratorConfig.load_preferences()

        assert isinstance(prefs, UserPreferences)
        assert prefs.preferred_port == 7999

    def test_load_and_save_preferences_roundtrip(self, tmp_path):
        """Test saving and loading preferences."""
        pref_file = tmp_path / "user-preferences.json"
        prefs = UserPreferences(preferred_gpu=1, preferred_port=8080)

        with patch.object(OrchestratorConfig, "preferences_file", return_value=pref_file):
            OrchestratorConfig.save_preferences(prefs)
            loaded = OrchestratorConfig.load_preferences()

        assert loaded.preferred_gpu == 1
        assert loaded.preferred_port == 8080

    def test_load_preferences_corrupt_file(self, tmp_path):
        """Test loading preferences with corrupt JSON."""
        pref_file = tmp_path / "user-preferences.json"
        pref_file.write_text("{invalid json")

        with patch.object(OrchestratorConfig, "preferences_file", return_value=pref_file):
            prefs = OrchestratorConfig.load_preferences()

        assert isinstance(prefs, UserPreferences)

    def test_save_preferences_error(self, tmp_path):
        """Test RuntimeError when saving preferences fails."""
        pref_file = tmp_path / "sub" / "user-preferences.json"
        prefs = UserPreferences()

        with patch.object(OrchestratorConfig, "preferences_file", return_value=pref_file):
            with patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")):
                with pytest.raises(RuntimeError):
                    OrchestratorConfig.save_preferences(prefs)

    def test_user_preferences_defaults(self):
        prefs = UserPreferences()
        assert prefs.preferred_gpu is None
        assert prefs.preferred_port == 7999
        assert prefs.preferred_interface == "127.0.0.1"
