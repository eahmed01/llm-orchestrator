"""Environment detection: GPUs, network interfaces, running processes."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EnvironmentDetector:
    """Detect and manage environment resources."""

    @staticmethod
    def pcie_status(pcie_link_gen: int, pcie_link_gen_max: int,
                     pcie_link_width: int, pcie_link_width_max: int) -> str:
        """Return a human-readable PCIe link status string.

        Returns status like 'PCIe 4.0 x16', 'PCIe 1.0 x16 (DEGRADED, expected 4.0 x16)', etc.
        """
        if pcie_link_gen < pcie_link_gen_max:
            return (f"PCIe {pcie_link_gen}.0 x{pcie_link_width} "
                    f"(DEGRADED, expected {pcie_link_gen_max}.0 x{pcie_link_width_max})")
        if pcie_link_width < pcie_link_width_max:
            return (f"PCIe {pcie_link_gen}.0 x{pcie_link_width} "
                    f"(WIDTH DEGRADED, expected x{pcie_link_width_max})")
        return f"PCIe {pcie_link_gen}.0 x{pcie_link_width}"

    @staticmethod
    def pcie_is_degraded(pcie_link_gen: int, pcie_link_gen_max: int,
                          pcie_link_width: int, pcie_link_width_max: int) -> bool:
        """Return True if the PCIe link is running below its maximum capability."""
        return pcie_link_gen < pcie_link_gen_max or pcie_link_width < pcie_link_width_max

    @staticmethod
    def detect_gpus() -> list[dict[str, Any]]:
        """Detect available NVIDIA GPUs and their status.

        Returns:
            List of GPU info dicts with keys:
                index, name, total_memory_gb, used_memory_mb,
                pcie_link_gen, pcie_link_gen_max, pcie_link_width, pcie_link_width_max
        """
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,"
                    "pcie.link.gen.current,pcie.link.gen.max,"
                    "pcie.link.width.current,pcie.link.width.max",
                    "--format=csv,nounits,noheader",
                ],
                text=True,
            )

            gpus = []
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu: dict[str, Any] = {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_memory_gb": int(parts[2]) // 1024,
                        "used_memory_mb": int(parts[3]),
                    }
                    if len(parts) >= 8:
                        gpu["pcie_link_gen"] = int(parts[4])
                        gpu["pcie_link_gen_max"] = int(parts[5])
                        gpu["pcie_link_width"] = int(parts[6])
                        gpu["pcie_link_width_max"] = int(parts[7])
                    gpus.append(gpu)
            return gpus
        except Exception as e:
            logger.warning(f"Failed to detect GPUs: {e}")
            return []

    @staticmethod
    def detect_interfaces() -> list[dict[str, str]]:
        """Detect available network interfaces.

        Returns:
            List of interface info dicts: {"name": "eth0", "ip": "192.168.1.100", ...}
        """
        interfaces = []

        # Always include localhost
        interfaces.append({"name": "localhost", "ip": "127.0.0.1", "type": "loopback"})
        interfaces.append({"name": "all", "ip": "0.0.0.0", "type": "all_networks"})

        try:
            import socket

            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if local_ip not in ["127.0.0.1", "0.0.0.0"]:
                interfaces.append(
                    {"name": "local", "ip": local_ip, "type": "local_network"}
                )
        except Exception:
            pass

        return interfaces

    @staticmethod
    def is_port_in_use(port: int) -> Optional[int]:
        """Check if a port is in use and return the PID of the process using it.

        Returns:
            PID of process using the port, or None if port is free.
        """
        try:
            output = subprocess.check_output(
                ["lsof", "-i", f":{port}", "-t"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            pids = output.strip().split("\n")
            if pids and pids[0]:
                return int(pids[0])
        except (subprocess.CalledProcessError, ValueError):
            pass
        return None

    @staticmethod
    def get_running_services() -> dict[int, dict[str, Any]]:
        """Get info about running vLLM processes.

        Returns:
            Dict mapping port to {"pid": int, "model": str, "gpu": int, ...}
        """
        services = {}

        # Try to find vLLM processes
        try:
            output = subprocess.check_output(
                ["ps", "aux"],
                text=True,
            )

            for line in output.split("\n"):
                if "vllm" not in line or "serve" not in line:
                    continue

                # Parse the command line to extract model and port
                parts = line.split()
                try:
                    pid = int(parts[1])
                    # Look for port in command
                    for i, part in enumerate(parts):
                        if part == "--port" and i + 1 < len(parts):
                            port = int(parts[i + 1])
                            services[port] = {"pid": pid, "cmd": " ".join(parts[11:])}
                            break
                except (ValueError, IndexError):
                    continue

        except Exception as e:
            logger.debug(f"Failed to get running services: {e}")

        return services


def ask_gpu_preference(gpus: list[dict[str, Any]]) -> Optional[int]:
    """Ask user to select a GPU.

    Returns:
        GPU index, or None to auto-select.
    """
    if not gpus:
        return None

    print("\n📡 Available GPUs:")
    for gpu in gpus:
        memory_used = gpu["used_memory_mb"] // 1024
        memory_total = gpu["total_memory_gb"]
        usage_pct = (gpu["used_memory_mb"] / (memory_total * 1024)) * 100
        print(
            f"  [{gpu['index']}] {gpu['name']} - {memory_total}GB "
            f"({memory_used}/{memory_total}GB used, {usage_pct:.0f}%)"
        )

    while True:
        try:
            choice = input(f"\nWhich GPU? [0-{len(gpus)-1}]: ").strip()
            if choice == "":
                return None  # Auto-select
            gpu_idx = int(choice)
            if 0 <= gpu_idx < len(gpus):
                return gpus[gpu_idx]["index"]
            print(f"Invalid choice. Please enter 0-{len(gpus)-1}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def ask_port_preference() -> int:
    """Ask user to select a port."""
    while True:
        try:
            choice = input("\nWhich port? [default: 7999]: ").strip()
            if choice == "":
                return 7999
            port = int(choice)
            if 1024 <= port <= 65535:
                return port
            print("Port must be between 1024 and 65535")
        except ValueError:
            print("Invalid input. Please enter a number.")


def ask_interface_preference(interfaces: list[dict[str, str]]) -> str:
    """Ask user to select a network interface.

    Returns:
        IP address of selected interface.
    """
    if not interfaces or len(interfaces) == 1:
        return interfaces[0]["ip"] if interfaces else "127.0.0.1"

    print("\n🌐 Available Interfaces:")
    for i, iface in enumerate(interfaces):
        print(f"  [{i}] {iface['ip']} ({iface['type']})")

    while True:
        try:
            choice = input(f"\nWhich interface? [0-{len(interfaces)-1}]: ").strip()
            if choice == "":
                return interfaces[0]["ip"]
            idx = int(choice)
            if 0 <= idx < len(interfaces):
                return interfaces[idx]["ip"]
            print(f"Invalid choice. Please enter 0-{len(interfaces)-1}")
        except ValueError:
            print("Invalid input. Please enter a number.")
