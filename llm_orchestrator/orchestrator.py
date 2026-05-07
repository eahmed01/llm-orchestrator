"""Main orchestrator: retry loop state machine for LLM startup."""

import asyncio
import logging
import subprocess
import sys
from typing import Any, Optional

from llm_orchestrator.advisor import Advisor
from llm_orchestrator.config import OrchestratorConfig
from llm_orchestrator.model_discovery import ModelDiscovery
from llm_orchestrator.monitor import Monitor

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

    async def start(self) -> bool:
        """Start the service with advisor-guided retry logic."""
        logger.info(f"Starting {self.service} with model {self.model or 'default'}")

        # Load model if not specified
        if not self.model:
            saved_config = self.config.get_service_config(self.service)
            if saved_config:
                self.model = saved_config.model
                self.args = saved_config.args
                logger.info(f"Using saved config: {self.model}")
            else:
                return self._exit_failure("No model specified and no saved config found")

        # Validate model exists
        logger.info(f"Validating model {self.model}...")
        if not await ModelDiscovery.validate_model_exists(self.model):
            logger.warning(f"Model {self.model} not found on HuggingFace")
            # Continue anyway (might be local or in cache)

        # Check viability
        viability = await self.advisor.estimate_viability(self.model, hardware_vram_gb=95)
        logger.info(f"Viability check: {viability}")

        # Retry loop
        while self.retry_count < self.MAX_RETRIES:
            success, failure_reason = await self._attempt_startup(self.model, self.args)

            if success:
                logger.info("✓ Startup successful!")
                self.config.record_success(self.service, self.model, args=self.args)
                return True

            if not failure_reason:
                logger.error("Startup failed with unknown reason")
                return self._exit_failure("Unknown failure")

            self.retry_count += 1
            logger.error(f"Startup failed: {failure_reason} (attempt {self.retry_count}/{self.MAX_RETRIES})")

            if self.retry_count >= self.MAX_RETRIES:
                return self._exit_failure(f"Max retries reached ({self.MAX_RETRIES})")

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

    async def _attempt_startup(self, model: str, args: dict[str, Any]) -> tuple[bool, Optional[str]]:
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

    def _build_vllm_command(self, model: str, args: dict[str, Any]) -> list[str]:
        """Build vLLM command line."""
        cmd = [
            sys.executable,
            "-m",
            "vllm",
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            "7999",
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
