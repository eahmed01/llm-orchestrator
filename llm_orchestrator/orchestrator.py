"""Main orchestrator: retry loop state machine for LLM startup."""

import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from llm_orchestrator.advisor import Advisor
from llm_orchestrator.config import OrchestratorConfig
from llm_orchestrator.model_discovery import ModelDiscovery
from llm_orchestrator.monitor import Monitor
from llm_orchestrator.profiles import (
    Attempt,
    ProfileStore,
    get_store,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Manages the startup lifecycle of an LLM service with adaptive retry."""

    MAX_RETRIES = 5

    def __init__(
        self,
        service: str = "vllm",
        model: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
        advisor_endpoint: Optional[str] = None,
        vllm_log_file: Optional[str] = None,
    ):
        self.service = service
        self.model = model
        self.args = args or {}
        self.advisor = Advisor(endpoint=advisor_endpoint)
        self.vllm_log_file = vllm_log_file or "/work/fast3/xeio/logs/vllm.log"
        self.config = OrchestratorConfig.load_from_disk()
        self.retry_count = 0
        self.profiles = get_store()
        # Interface and port for vLLM binding (can be overridden)
        self.interface = "127.0.0.1"
        self.port = 7999

    async def start(self) -> bool:
        """Start the service with advisor-guided retry logic and profile awareness."""
        logger.info(f"Starting {self.service} with model {self.model or 'default'}")

        # Load model if not specified — try profile first, then config
        if not self.model:
            saved_config = self.config.get_service_config(self.service)
            if saved_config:
                self.model = saved_config.model
                self.args = saved_config.args
                logger.info(f"Using saved config: {self.model}")
            else:
                return self._exit_failure(
                    "No model specified and no saved config found"
                )

        # -- Profile lookup: reuse known-good config for this model --
        known_good = self.profiles.get_known_good(self.model)
        if known_good:
            profile_args = known_good.get("args", {})
            if not self.args or self.args == {"--max-num-seqs": 848, "--gpu-memory-utilization": 0.92}:
                # User didn't provide custom args; reuse what worked
                logger.info(
                    f"Profile: reusing known-good args for {self.model} "
                    f"(success on {known_good.get('ts', '?')})"
                )
                self.args = {**self.args, **profile_args}
                if known_good.get("gpus"):
                    self._gpu_override = known_good["gpus"]
                if known_good.get("interface"):
                    self.interface = known_good["interface"]
                if known_good.get("port"):
                    self.port = known_good["port"]

        # -- New model: borrow args from similar models if nothing set --
        if not self.args:
            similar = self.profiles.find_similar(self.model)
            if similar:
                _, sim_profile = similar[0]
                sim_good = sim_profile.get_known_good()
                if sim_good and sim_good.get("args"):
                    logger.info(
                        f"Profile: borrowing args from similar model "
                        f"{sim_profile.model} (family match)"
                    )
                    self.args = dict(sim_good["args"])

        # Validate model exists
        logger.info(f"Validating model {self.model}...")
        if not await ModelDiscovery.validate_model_exists(self.model):
            logger.warning(f"Model {self.model} not found on HuggingFace")
            # Continue anyway (might be local or in cache)

        # Check viability
        viability = await self.advisor.estimate_viability(
            self.model, hardware_vram_gb=95
        )
        logger.info(f"Viability check: {viability}")

        # Capture env snapshot
        env_snapshot = self._capture_env()

        # Retry loop
        t_start = asyncio.get_event_loop().time()
        while self.retry_count < self.MAX_RETRIES:
            t0 = asyncio.get_event_loop().time()
            success, failure_reason = await self._attempt_startup(self.model, self.args)
            duration = asyncio.get_event_loop().time() - t0

            # Record attempt in profile
            attempt = Attempt(
                ts=datetime.now(timezone.utc).isoformat(),
                model=self.model,
                args=self.args,
                env=env_snapshot,
                gpus=getattr(self, "_gpu_override", []),
                port=self.port,
                interface=self.interface,
                success=success,
                failure_reason=failure_reason,
                duration_s=round(duration, 2),
            )
            self.profiles.record_attempt(attempt)

            if success:
                logger.info("✓ Startup successful!")
                self.config.record_success(self.service, self.model, args=self.args)
                return True

            if not failure_reason:
                logger.error("Startup failed with unknown reason")
                return self._exit_failure("Unknown failure")

            self.retry_count += 1
            logger.error(
                f"Startup failed: {failure_reason} (attempt {self.retry_count}/{self.MAX_RETRIES})"
            )

            if self.retry_count >= self.MAX_RETRIES:
                total_time = asyncio.get_event_loop().time() - t_start
                logger.info(
                    f"Profile: {self.model} has "
                    f"{self.profiles.get_profile(self.model).failure_count()} "
                    f"recorded failures on this host"
                )
                return self._exit_failure(
                    f"Max retries reached ({self.MAX_RETRIES}) in {total_time:.0f}s"
                )

            # Ask advisor what to try next
            fallback_chain = ModelDiscovery.build_fallback_chain(self.model)
            logger.info(f"Considering {len(fallback_chain)} fallback options...")

            decision = await self.advisor.decide_next_step(
                model=self.model,
                failure_reason=failure_reason,
                options=fallback_chain,
            )

            logger.info(f"Advisor recommendation: {decision.get('reasoning')}")
            logger.info(f"Confidence: {decision.get('confidence', 0):.0%}")

            # Extract next model to try
            rec_idx = int(decision.get("recommendation", 1)) - 1
            if 0 <= rec_idx < len(fallback_chain):
                next_model, quant = fallback_chain[rec_idx]
                self.model = next_model
                logger.info(f"Retrying with: {self.model} (variant: {quant})")

                # Reset monitor for next attempt
                monitor = Monitor(self.vllm_log_file)
                monitor.reset()
            else:
                logger.error("Invalid advisor recommendation")
                return self._exit_failure("Advisor gave invalid recommendation")

        return False

    async def _attempt_startup(
        self, model: str, args: dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Attempt to start vLLM with given model and args.

        Returns:
            (success: bool, failure_reason: Optional[str])
        """
        # Build vLLM command
        cmd = self._build_vllm_command(model, args)
        logger.info(f"Launching: {' '.join(cmd)}")

        # Start process
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"vLLM started (PID: {process.pid})")
        except Exception as e:
            logger.error(f"Failed to start vLLM: {e}")
            return False, "launch_failed"

        # Monitor startup
        try:
            monitor = Monitor(self.vllm_log_file)
            monitor.reset()
            success, failure_reason = await monitor.monitor_startup(timeout_seconds=300)

            if not success:
                # Kill the process
                try:
                    process.kill()
                    await asyncio.sleep(1)  # Let it die
                except Exception:
                    pass

            return success, failure_reason

        except Exception as e:
            logger.error(f"Monitor error: {e}")
            try:
                process.kill()
            except Exception:
                pass
            return False, "monitor_error"

    def _capture_env(self) -> dict[str, str]:
        """Capture relevant environment variables."""
        return {
            k: v for k, v in os.environ.items()
            if k.startswith(("VLLM_", "CUDA_", "LD_", "HF_"))
            or k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME", "PATH")
        }

    def _build_vllm_command(self, model: str, args: dict[str, Any]) -> list[str]:
        """Build vLLM command line."""
        cmd = [
            sys.executable,
            "-m",
            "vllm",
            "serve",
            model,
            "--host",
            self.interface,
            "--port",
            str(self.port),
        ]

        # Add default args if not specified
        if "--max-num-seqs" not in args:
            args["--max-num-seqs"] = 848
        if "--gpu-memory-utilization" not in args:
            args["--gpu-memory-utilization"] = 0.92

        # Add tool-choice args for Qwen models
        if "qwen" in model.lower():
            if "--enable-auto-tool-choice" not in args:
                cmd.extend(["--enable-auto-tool-choice"])
            if "--tool-call-parser" not in args:
                cmd.extend(["--tool-call-parser", "qwen3_coder"])
            if "--enable-reasoning" not in args:
                cmd.extend(["--enable-reasoning"])
            if "--reasoning-parser" not in args:
                cmd.extend(["--reasoning-parser", "qwen3"])

        # Add custom args
        for key, value in args.items():
            cmd.append(key)
            if value is not True:  # Skip boolean flags
                cmd.append(str(value))

        return cmd

    def _exit_failure(self, reason: str) -> bool:
        """Log and exit on failure."""
        logger.error(f"Exiting: {reason}")
        return False

    async def close(self) -> None:
        """Cleanup resources."""
        await self.advisor.close()
