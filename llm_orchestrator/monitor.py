"""Monitor: real-time log tailing, startup detection, and port health checks."""

import asyncio
import logging
import re
import socket
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Monitor:
    """Monitor vLLM startup by tailing logs and checking port health."""

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

    @staticmethod
    async def health_check(host: str, port: int, timeout: float = 3) -> bool:
        """Check if a service's /v1/models endpoint responds.

        Args:
            host: Hostname or IP (use '127.0.0.1' for localhost).
            port: TCP port.
            timeout: HTTP request timeout in seconds.

        Returns:
            True if the endpoint returns HTTP 200.
        """
        request_host = host if host != "0.0.0.0" else "127.0.0.1"
        url = f"http://{request_host}:{port}/v1/models"

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
            TimeoutError,
        ) as exc:
            logger.debug("Health check %s:%d failed: %s", host, port, exc)
            return False

    @staticmethod
    async def wait_for_service(
        host: str, port: int, timeout: float = 60, interval: float = 1
    ) -> bool:
        """Poll a service's health endpoint until it responds or times out.

        Args:
            host: Hostname or IP.
            port: TCP port.
            timeout: Maximum time to wait in seconds.
            interval: Seconds between polls.

        Returns:
            True if the service became healthy within the timeout.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if await Monitor.health_check(host, port, timeout=3):
                logger.info("Service %s:%d is healthy", host, port)
                return True
            await asyncio.sleep(interval)
        logger.warning("Service %s:%d did not become healthy within %.0fs", host, port, timeout)
        return False

    @staticmethod
    def port_in_use(port: int) -> bool:
        """Check if a TCP port is currently bound/listening."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            sock.close()
            return False
        except OSError:
            sock.close()
            return True
