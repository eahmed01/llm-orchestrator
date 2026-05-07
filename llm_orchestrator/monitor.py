"""Monitor: real-time log tailing and startup detection."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Monitor:
    """Monitor vLLM startup by tailing logs."""

    SUCCESS_PATTERNS = [
        r"Application startup complete",
        r"Uvicorn running on",
        r"Started server process",
    ]

    ERROR_PATTERNS = [
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"out of memory",
        r"RuntimeError.*cuda",
        r"Exception.*model",
        r"Failed to load model",
    ]

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.position = 0

    def reset(self) -> None:
        """Reset log position to start."""
        self.position = 0

    async def monitor_startup(
        self,
        timeout_seconds: int = 300,
    ) -> tuple[bool, Optional[str]]:
        """Monitor logs until startup complete or failure.

        Returns:
            (success: bool, failure_reason: Optional[str])
        """
        start_time = asyncio.get_event_loop().time()
        check_interval = 0.5

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                return False, "startup_timeout"

            # Check for success or failure
            try:
                if self.log_file.exists():
                    with open(self.log_file, "r") as f:
                        f.seek(self.position)
                        lines = f.readlines()
                        self.position = f.tell()

                        for line in lines:
                            # Check success patterns
                            for pattern in self.SUCCESS_PATTERNS:
                                if re.search(pattern, line, re.IGNORECASE):
                                    logger.info(f"✓ Startup success: {line.strip()}")
                                    return True, None

                            # Check error patterns
                            for pattern in self.ERROR_PATTERNS:
                                if re.search(pattern, line, re.IGNORECASE):
                                    logger.error(f"✗ Startup error: {line.strip()}")
                                    return False, line.strip()

            except Exception as e:
                logger.warning(f"Error reading log: {e}")

            await asyncio.sleep(check_interval)
