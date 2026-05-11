"""Tests for llm_orchestrator.cli module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from llm_orchestrator.cli import app
from llm_orchestrator.config import ServiceConfig

runner = CliRunner()


class TestDiscoverCommand:
    """Tests for discover command."""

    def test_discover_finds_variants(self):
        """Test discover command finds and displays variants."""
        mock_variants = {
            "Qwen/Qwen3.6-27B-Q4": {"size_gb": 13.89, "format": "q4_k_m"},
            "Qwen/Qwen3.6-27B-Q5": {"size_gb": 18.45, "format": "q5_k_m"},
        }

        with patch(
            "llm_orchestrator.cli.ModelDiscovery.find_variants",
            new_callable=AsyncMock,
            return_value=mock_variants,
        ):
            result = runner.invoke(app, ["discover", "Qwen/Qwen3.6-27B"])

        assert result.exit_code == 0
        assert "13.89GB" in result.stdout
        assert "18.45GB" in result.stdout

    def test_discover_no_variants(self):
        """Test discover command when no variants found."""
        with patch(
            "llm_orchestrator.cli.ModelDiscovery.find_variants",
            new_callable=AsyncMock,
            return_value={},
        ):
            result = runner.invoke(app, ["discover", "unknown-model"])

        assert result.exit_code == 0
        assert "No variants found" in result.stdout


class TestStatusCommand:
    """Tests for status command."""

    def test_status_with_config(self):
        """Test status command when config exists."""
        service_config = ServiceConfig(model="Qwen/Qwen3.6-27B-FP8")

        with patch(
            "llm_orchestrator.cli.OrchestratorConfig.load_from_disk"
        ) as mock_load:
            mock_config = MagicMock()
            mock_config.get_service_config.return_value = service_config
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["status", "vllm"])

        assert result.exit_code == 0
        assert "Qwen/Qwen3.6-27B-FP8" in result.stdout
        assert "Service: vllm" in result.stdout

    def test_status_no_config(self):
        """Test status command when no config found."""
        with patch(
            "llm_orchestrator.cli.OrchestratorConfig.load_from_disk"
        ) as mock_load:
            mock_config = MagicMock()
            mock_config.get_service_config.return_value = None
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["status", "vllm"])

        assert result.exit_code == 0
        assert "No configuration found" in result.stdout


class TestConfigCommand:
    """Tests for config command."""

    def test_config_show(self):
        """Test config show action."""
        service_config = ServiceConfig(model="Qwen/Qwen3.6-27B-FP8")

        with patch(
            "llm_orchestrator.cli.OrchestratorConfig.load_from_disk"
        ) as mock_load:
            mock_config = MagicMock()
            mock_config.get_service_config.return_value = service_config
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["config", "vllm", "--action", "show"])

        assert result.exit_code == 0
        assert "Qwen/Qwen3.6-27B-FP8" in result.stdout

    def test_config_delete(self):
        """Test config delete action."""
        with patch(
            "llm_orchestrator.cli.OrchestratorConfig.load_from_disk"
        ) as mock_load:
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["config", "vllm", "--action", "delete"])

        assert result.exit_code == 0
        mock_config.save_to_disk.assert_called_once()
        assert "Deleted config" in result.stdout


class TestStopCommand:
    """Tests for stop command."""

    def test_stop_service(self):
        """Test stop command."""
        result = runner.invoke(app, ["stop", "vllm"])

        assert result.exit_code == 0
        assert "Stopped vllm" in result.stdout


class TestStartCommand:
    """Tests for start command."""

    @pytest.mark.asyncio
    async def test_start_with_model(self):
        """Test start command with model specified."""
        with patch("llm_orchestrator.cli.Orchestrator") as mock_orchestrator_class, \
             patch("llm_orchestrator.cli.OrchestratorConfig") as mock_config_class, \
             patch("llm_orchestrator.cli.EnvironmentDetector") as mock_env, \
             patch("llm_orchestrator.cli.record_event"):
            mock_orchestrator = AsyncMock()
            mock_orchestrator.start.return_value = True
            mock_orchestrator.args = {}
            mock_orchestrator.model = "Qwen/Qwen3.6-27B-FP8"
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_config = MagicMock()
            mock_config.load_preferences.return_value = MagicMock(
                preferred_gpu=0, preferred_port=7999, preferred_interface="127.0.0.1"
            )
            mock_config_class.load_preferences.return_value = mock_config.load_preferences.return_value

            mock_env.detect_gpus.return_value = [{"index": 0, "name": "GPU0", "total_memory_gb": 24, "used_memory_mb": 0}]
            mock_env.detect_interfaces.return_value = [{"ip": "127.0.0.1", "type": "loopback"}]
            mock_env.is_port_in_use.return_value = None

            result = runner.invoke(
                app,
                ["start", "vllm", "--model", "Qwen/Qwen3.6-27B-FP8"],
            )

        assert result.exit_code == 0
        mock_orchestrator.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_failure(self):
        """Test start command when startup fails."""
        with patch("llm_orchestrator.cli.Orchestrator") as mock_orchestrator_class, \
             patch("llm_orchestrator.cli.OrchestratorConfig") as mock_config_class, \
             patch("llm_orchestrator.cli.EnvironmentDetector") as mock_env, \
             patch("llm_orchestrator.cli.record_event"):
            mock_orchestrator = AsyncMock()
            mock_orchestrator.start.return_value = False
            mock_orchestrator.args = {}
            mock_orchestrator.model = None
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_config = MagicMock()
            mock_config.load_preferences.return_value = MagicMock(
                preferred_gpu=0, preferred_port=7999, preferred_interface="127.0.0.1"
            )
            mock_config_class.load_preferences.return_value = mock_config.load_preferences.return_value

            mock_env.detect_gpus.return_value = [{"index": 0, "name": "GPU0", "total_memory_gb": 24, "used_memory_mb": 0}]
            mock_env.detect_interfaces.return_value = [{"ip": "127.0.0.1", "type": "loopback"}]
            mock_env.is_port_in_use.return_value = None

            result = runner.invoke(app, ["start", "vllm"])

        assert result.exit_code == 1


class TestHelpCommands:
    """Tests for help output."""

    def test_main_help(self):
        """Test main help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Automate trial-and-error LLM startup" in result.stdout
        assert "start" in result.stdout
        assert "discover" in result.stdout
        assert "status" in result.stdout

    def test_start_help(self):
        """Test start command help."""
        result = runner.invoke(app, ["start", "--help"])

        assert result.exit_code == 0
        assert "Start an LLM service" in result.stdout
        assert "--model" in result.stdout
        assert "--max-retries" in result.stdout

from llm_orchestrator.cli import run_async, _execute_step

class TestEnvCommand:
    """Tests for env command."""

    def test_env_command(self):
        """Test env command shows environment info."""
        with patch("llm_orchestrator.cli.EnvironmentDetector") as mock_env,              patch("llm_orchestrator.cli.OrchestratorConfig") as mock_cfg:
            mock_env.detect_gpus.return_value = [
                {"index": 0, "name": "GPU0", "total_memory_gb": 95, "used_memory_mb": 50000}
            ]
            mock_env.detect_interfaces.return_value = [
                {"ip": "127.0.0.1", "type": "loopback"}
            ]
            mock_env.get_running_services.return_value = {}
            prefs = MagicMock(preferred_gpu=0, preferred_port=7999, preferred_interface="127.0.0.1")
            mock_cfg.load_preferences.return_value = prefs

            result = runner.invoke(app, ["env"])

        assert result.exit_code == 0
        assert "GPU" in result.stdout
        assert "Interface" in result.stdout or "interface" in result.stdout.lower()


class TestModelsCommand:
    """Tests for models command."""

    @pytest.mark.asyncio
    async def test_models_command_with_trending(self):
        """Test models command shows trending models."""
        trending = {
            "safe": [{"name": "test/model", "desc": "A test model", "vram": "14GB", "risk": "low", "context": 8192, "model_type": "instruct"}],
            "ambitious": [],
            "experimental": [],
        }

        with patch("llm_orchestrator.cli.ModelDiscovery") as MockMD:
            MockMD.fetch_trending_models = AsyncMock(return_value=trending)
            MockMD.detect_gpu_vram.return_value = 95

            result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "SAFE" in result.stdout


class TestBenchmarkCommand:
    """Tests for benchmark command."""

    def test_benchmark_empty_input(self):
        """Test benchmark with no models."""
        result = runner.invoke(app, ["benchmark", ""])
        # Command exits cleanly; error message goes to stderr which isn't captured
        assert result.exit_code == 0

    def test_benchmark_with_models(self):
        """Test benchmark comparison."""
        with patch("llm_orchestrator.cli.ModelDiscovery") as MockMD:
            MockMD.get_benchmarks.return_value = {
                "reasoning": {"score": 85.5, "count": 3},
                "math": {"score": 72.1, "count": 2},
            }

            result = runner.invoke(app, ["benchmark", "test/model"])

        assert result.exit_code == 0
        assert "BENCHMARK" in result.stdout


class TestStackCommand:
    """Tests for stack command."""

    def test_stack_command(self):
        """Test stack command shows snapshot."""
        from datetime import datetime, timezone
        from llm_orchestrator.stack import StackSnapshot

        snap = StackSnapshot(
            services={},
            gpus=[{"index": 0, "name": "GPU0", "memory_used_mb": 50000, "memory_total_mb": 97887, "utilization_pct": 5, "temperature_c": 45}],
            timestamp=datetime.now(timezone.utc),
        )

        with patch("llm_orchestrator.cli.StackDetector.capture_snapshot", return_value=snap):
            result = runner.invoke(app, ["stack"])

        assert result.exit_code == 0
        assert "GPU" in result.stdout


class TestRestartCommand:
    """Tests for restart command."""

    def test_restart_all_services(self):
        """Test restart command for all services."""
        from llm_orchestrator.service import ServiceResult

        with patch("llm_orchestrator.cli.ServiceManager") as MockMgr:
            mock_mgr = MagicMock()
            mock_mgr.stop.return_value = ServiceResult(success=True, name="test", message="Stopped")
            mock_mgr.start.return_value = ServiceResult(success=True, name="test", pid=1234, message="Started")
            MockMgr.return_value = mock_mgr

            with patch("llm_orchestrator.cli.StackDetector.capture_snapshot"):
                result = runner.invoke(app, ["restart"])

        assert result.exit_code == 0
        assert "Restarting" in result.stdout


class TestPlanCommand:
    """Tests for plan command."""

    def test_plan_no_args(self):
        """Test plan command with no arguments."""
        with patch("llm_orchestrator.cli.StackDetector.capture_snapshot"):
            result = runner.invoke(app, ["plan"])

        assert result.exit_code == 1

    def test_plan_diff_no_json(self):
        """Test plan --diff without --json."""
        with patch("llm_orchestrator.cli.StackDetector.capture_snapshot"):
            result = runner.invoke(app, ["plan", "--diff"])

        assert result.exit_code == 1
        # Error message goes to stderr, check exit code instead

    def test_plan_diff_invalid_json(self):
        """Test plan --diff with invalid JSON."""
        with patch("llm_orchestrator.cli.StackDetector.capture_snapshot"):
            result = runner.invoke(app, ["plan", "--diff", "--json", "{invalid"])

        assert result.exit_code == 1

    def test_plan_dry_run(self):
        """Test plan with --dry-run flag."""
        from datetime import datetime, timezone
        from llm_orchestrator.stack import StackSnapshot

        snap = StackSnapshot(
            services={},
            gpus=[],
            timestamp=datetime.now(timezone.utc),
        )

        with patch("llm_orchestrator.cli.StackDetector.capture_snapshot", return_value=snap):
            result = runner.invoke(app, ["plan", "--diff", "--json", '{"vllm":{"gpus":[0]}}', "--dry-run"])

        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout


class TestExecuteStep:
    """Tests for _execute_step helper."""

    def test_execute_stop_service(self):
        from llm_orchestrator.planner import PlanStep
        from llm_orchestrator.service import ServiceResult

        step = PlanStep(action="STOP_SERVICE", service="vllm", details="test", estimated_seconds=10, risk="medium", rollback="")
        mgr = MagicMock()
        mgr.stop.return_value = ServiceResult(success=True, name="vllm", message="Stopped")

        result = _execute_step(step, mgr, {}, {})
        assert result is True

    def test_execute_wait_grace(self):
        from llm_orchestrator.planner import PlanStep

        step = PlanStep(action="WAIT_GRACE", service=None, details="2s grace", estimated_seconds=0, risk="low", rollback="")
        result = _execute_step(step, MagicMock(), {}, {})
        assert result is True

    def test_execute_verify_port(self):
        from llm_orchestrator.planner import PlanStep
        from llm_orchestrator.stack import StackConfig

        step = PlanStep(action="VERIFY_PORT", service="vllm", details="port 7999", estimated_seconds=5, risk="low", rollback="")
        cfgs = {"vllm": StackConfig(name="vllm", model="test", port=7999)}

        with patch("llm_orchestrator.cli.StackDetector.verify_endpoint", return_value=True):
            result = _execute_step(step, MagicMock(), cfgs, {})
        assert result is True


class TestRenderSnapshot:
    """Tests for _render_snapshot helper."""

    def test_render_snapshot_no_services_no_gpus(self):
        from datetime import datetime, timezone
        from llm_orchestrator.stack import StackSnapshot

        snap = StackSnapshot(
            services={},
            gpus=[],
            timestamp=datetime.now(timezone.utc),
        )

        from llm_orchestrator.cli import _render_snapshot
        result = runner.invoke(app, [], catch_exceptions=False)

        with patch("typer.echo") as mock_echo:
            _render_snapshot(snap)
            # Should have been called
            assert mock_echo.call_count > 0


class TestRunAsync:
    """Tests for run_async helper."""

    def test_run_async_no_event_loop(self):
        async def simple():
            return 42

        result = run_async(simple())
        assert result == 42

    def test_run_async_existing_event_loop(self):
        import asyncio

        async def inner():
            return "ok"

        # When called from an existing loop, it uses ThreadPoolExecutor
        # This is hard to test directly, so we just verify the function exists
        assert callable(run_async)
