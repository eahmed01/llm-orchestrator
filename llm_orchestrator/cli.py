"""CLI: Command-line interface for llm-orchestrator."""

import asyncio
import json
import logging
import subprocess
import sys
import time as _time
from pathlib import Path
from typing import Any, Coroutine, Optional, TypeVar

import typer
from typing_extensions import Annotated

from llm_orchestrator.history import record_event
from llm_orchestrator.config import OrchestratorConfig, UserPreferences
from llm_orchestrator.environment import (
    EnvironmentDetector,
    ask_gpu_preference,
    ask_interface_preference,
    ask_port_preference,
)
from llm_orchestrator.model_discovery import ModelDiscovery
from llm_orchestrator.orchestrator import Orchestrator
from llm_orchestrator.stack import (
    StackDetector,
    StackServiceInfo,
    StackSnapshot,
    default_stack_configs,
)
from llm_orchestrator.service import (
    ServiceManager,
    ServiceResult,
    start_order,
    stop_order,
)
from llm_orchestrator.planner import (
    Plan,
    PlanStep,
    StackPlanner,
)

app = typer.Typer(
    help="Automate trial-and-error LLM startup with intelligent retry",
    no_args_is_help=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async function, handling both existing and new event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running; create one
        return asyncio.run(coro)
    else:
        # Event loop already running (e.g., in tests); create task
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


# ============================================================================
#  Help
# ============================================================================

@app.command(name="help")
def help_cmd() -> None:
    """Show help and examples."""
    typer.echo("")
    typer.echo("╔════════════════════════════════════════════════════════════════════════════════╗")
    typer.echo("║                          LLM Orchestrator - Help                               ║")
    typer.echo("╚════════════════════════════════════════════════════════════════════════════════╝")
    typer.echo("")
    typer.echo("🚀 Run massive LLMs locally. Instantly. No trial-and-error.")
    typer.echo("")
    typer.echo("QUICK START:")
    typer.echo("  ./llm-orchestrate start vllm --model Qwen/Qwen3.6-70B-Instruct")
    typer.echo("  → Auto-detects if it fits your GPU")
    typer.echo("  → If OOM: Automatically retries with optimized versions")
    typer.echo("  → Saves working config for next time")
    typer.echo("")
    typer.echo("COMMANDS:")
    typer.echo("")
    typer.echo("  start [SERVICE] [--model MODEL]")
    typer.echo("    Start an LLM service with intelligent retry")
    typer.echo("    Example: ./llm-orchestrate start vllm --model Qwen/Qwen3.6-70B-Instruct")
    typer.echo("")
    typer.echo("  status [SERVICE]")
    typer.echo("    Check what model is configured and when it last worked")
    typer.echo("    Example: ./llm-orchestrate status vllm")
    typer.echo("")
    typer.echo("  models [--gpu-vram GB]")
    typer.echo("    Browse popular models organized by difficulty/risk")
    typer.echo("    Shows context windows and VRAM requirements")
    typer.echo("    Example: ./llm-orchestrate models")
    typer.echo("    Example: ./llm-orchestrate models --gpu-vram 96")
    typer.echo("")
    typer.echo("  discover MODEL")
    typer.echo("    Find quantized variants (Q4, Q5, etc.) for a specific model")
    typer.echo("    Example: ./llm-orchestrate discover Qwen/Qwen3.6-70B")
    typer.echo("")
    typer.echo("  benchmark MODEL1,MODEL2,...")
    typer.echo("    Compare benchmark scores across models side-by-side")
    typer.echo("    Example: ./llm-orchestrate benchmark Qwen/Qwen3.6-70B,meta-llama/Llama-3.1-70B")
    typer.echo("")
    typer.echo("  config [SERVICE]")
    typer.echo("    Show or manage saved configuration")
    typer.echo("    Example: ./llm-orchestrate config vllm")
    typer.echo("")
    typer.echo("  stack")
    typer.echo("    Show full stack state: GPUs, services, ports, pids")
    typer.echo("")
    typer.echo("  restart [SERVICE ...]")
    typer.echo("    Restart one or all services with dependency ordering")
    typer.echo("")
    typer.echo("  plan \"GOAL\" [--diff]")
    typer.echo("    Natural-language planning (e.g., plan 'move chat to both gpus')")
    typer.echo("    --diff mode: specify desired state as JSON")
    typer.echo("")
    typer.echo("KEY FEATURES:")
    typer.echo("  ✓ Zero external services (no Ollama, APIs, or web UIs)")
    typer.echo("  ✓ Intelligent advisor model (built-in AI decides what to try next)")
    typer.echo("  ✓ Automatic model discovery (finds quantized variants)")
    typer.echo("  ✓ Persistent memory (saves working configs)")
    typer.echo("  ✓ Hardware-aware (knows your GPU VRAM)")
    typer.echo("")
    typer.echo("DOCUMENTATION:")
    typer.echo("  Full docs: https://github.com/eahmed01/llm-orchestrator")
    typer.echo("")


# ============================================================================
#  Environment
# ============================================================================

@app.command()
def env() -> None:
    """Show environment status: GPUs, interfaces, running services."""
    # Detect GPUs
    gpus = EnvironmentDetector.detect_gpus()
    interfaces = EnvironmentDetector.detect_interfaces()
    running = EnvironmentDetector.get_running_services()

    # Load user preferences
    prefs = OrchestratorConfig.load_preferences()

    typer.echo("\n📊 Environment Status")
    typer.echo("=" * 80)

    # GPUs
    typer.echo("\n🖥️  GPUs:")
    if gpus:
        for gpu in gpus:
            memory_used = gpu["used_memory_mb"] // 1024
            memory_total = gpu["total_memory_gb"]
            usage_pct = (gpu["used_memory_mb"] / (memory_total * 1024)) * 100
            status = "✓" if usage_pct < 50 else "⚠" if usage_pct < 80 else "❌"
            marker = " (default)" if prefs.preferred_gpu == gpu["index"] else ""

            # PCIe link status
            pcie_str = ""
            if "pcie_link_gen" in gpu:
                pcie_info = EnvironmentDetector.pcie_status(
                    gpu["pcie_link_gen"], gpu["pcie_link_gen_max"],
                    gpu["pcie_link_width"], gpu["pcie_link_width_max"]
                )
                pcie_str = " | " + pcie_info

            typer.echo(
                f"  {status} GPU {gpu['index']}: {gpu['name']} - "
                f"{memory_total}GB ({memory_used}/{memory_total}GB used, {usage_pct:.0f}%){marker}{pcie_str}"
            )
    else:
        typer.echo("  ❌ No GPUs detected (nvidia-smi not available)")

    # Interfaces
    typer.echo("\n🌐 Network Interfaces:")
    for iface in interfaces:
        marker = " (default)" if prefs.preferred_interface == iface["ip"] else ""
        typer.echo(f"  • {iface['ip']} ({iface['type']}){marker}")

    # Running services
    typer.echo("\n🚀 Running Services:")
    if running:
        for port, info in sorted(running.items()):
            typer.echo(f"  • Port {port}: {info.get('cmd', 'vLLM')[:60]}...")
    else:
        typer.echo("  (none)")

    # Default settings
    typer.echo("\n⚙️  Your Preferences:")
    typer.echo(f"  Default GPU: {prefs.preferred_gpu if prefs.preferred_gpu is not None else 'auto'}")
    typer.echo(f"  Default Port: {prefs.preferred_port}")
    typer.echo(f"  Default Interface: {prefs.preferred_interface}")
    typer.echo("")


# ============================================================================
#  Start / Stop / Status / Config (existing commands — unchanged logic)
# ============================================================================

@app.command()
def start(
    service_or_model: Annotated[
        Optional[str],
        typer.Argument(help="Service name (vllm) or model ID (e.g., Qwen/Qwen3.6-70B)"),
    ] = None,
    model: Annotated[Optional[str], typer.Option(help="Model to load (explicit override)")] = None,
    max_retries: Annotated[int, typer.Option(help="Max retry attempts")] = 5,
    gpu: Annotated[Optional[int], typer.Option(help="GPU index to use")] = None,
    interface: Annotated[Optional[str], typer.Option(help="Interface to bind to")] = None,
    port: Annotated[Optional[int], typer.Option(help="Port to bind to")] = None,
    force: Annotated[bool, typer.Option(help="Kill existing process if port is in use")] = False,
) -> None:
    """Start an LLM service with intelligent retry."""
    # Smart argument parsing: if service_or_model looks like a model ID, treat it as such
    service = "vllm"
    actual_model = model

    if service_or_model:
        # If it looks like a model ID (contains "/" or doesn't look like a service), treat as model
        if "/" in service_or_model or (
            service_or_model not in ["vllm", "ollama", "local"]
        ):
            actual_model = service_or_model
        else:
            service = service_or_model

    # Load preferences
    prefs = OrchestratorConfig.load_preferences()
    gpus = EnvironmentDetector.detect_gpus()
    interfaces = EnvironmentDetector.detect_interfaces()

    # Determine GPU to use
    selected_gpu = gpu
    if selected_gpu is None:
        if prefs.preferred_gpu is not None:
            selected_gpu = prefs.preferred_gpu
        else:
            # First time: ask user
            typer.echo("\n🎯 First-time setup:")
            selected_gpu = ask_gpu_preference(gpus)
            if selected_gpu is not None:
                prefs.preferred_gpu = selected_gpu
                OrchestratorConfig.save_preferences(prefs)

    # Determine interface to use
    selected_interface = interface or prefs.preferred_interface
    if selected_interface == "auto":
        typer.echo("\n🌐 Select network interface:")
        selected_interface = ask_interface_preference(interfaces)
        prefs.preferred_interface = selected_interface
        OrchestratorConfig.save_preferences(prefs)

    # Determine port to use
    selected_port = port or prefs.preferred_port

    # Check for port conflicts
    existing_pid = EnvironmentDetector.is_port_in_use(selected_port)
    if existing_pid:
        if force:
            typer.echo(f"🔪 Killing process on port {selected_port} (PID: {existing_pid})...")
            try:
                subprocess.run(["kill", "-9", str(existing_pid)], check=False)
                import time
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Failed to kill process: {e}")
        else:
            typer.echo(
                f"❌ Port {selected_port} is already in use (PID: {existing_pid})\n"
                f"   Use --force to kill the existing process"
            )
            sys.exit(1)

    async def _start() -> None:
        orchestrator = Orchestrator(service=service, model=actual_model)
        orchestrator.MAX_RETRIES = max_retries
        orchestrator.interface = selected_interface
        orchestrator.port = selected_port

        t0 = _time.time()
        try:
            success = await orchestrator.start()
            duration = _time.time() - t0
            if not success:
                record_event(
                    "start", service,
                    model=actual_model or orchestrator.model,
                    args=orchestrator.args, port=selected_port, success=False,
                    duration_s=duration, interface=selected_interface,
                )
                sys.exit(1)
            record_event(
                "start", service,
                model=actual_model or orchestrator.model,
                args=orchestrator.args, port=selected_port, success=True,
                duration_s=duration, interface=selected_interface,
            )
        finally:
            await orchestrator.close()

    run_async(_start())


@app.command()
def stop(
    service: Annotated[str, typer.Argument(help="Service to stop")] = "vllm",
) -> None:
    """Stop a running LLM service."""
    logger.info(f"Stopping {service}...")
    # TODO: implement service stopping
    typer.echo(f"Stopped {service}")


@app.command()
def status(
    service: Annotated[str, typer.Argument(help="Service to check")] = "vllm",
) -> None:
    """Check status of a service."""
    config = OrchestratorConfig.load_from_disk()
    service_config = config.get_service_config(service)

    if service_config:
        typer.echo(f"Service: {service}")
        typer.echo(f"  Model: {service_config.model}")
        typer.echo(f"  Variant: {service_config.variant or 'None'}")
        if service_config.last_successful:
            typer.echo(f"  Last successful: {service_config.last_successful}")
    else:
        typer.echo(f"No configuration found for {service}")


@app.command()
def config(
    service: Annotated[str, typer.Argument(help="Service name")] = "vllm",
    action: Annotated[str, typer.Option(help="Action: show, set, delete")] = "show",
    model: Annotated[Optional[str], typer.Option(help="Model ID")] = None,
) -> None:
    """Manage service configuration."""
    config_obj = OrchestratorConfig.load_from_disk()

    if action == "show":
        service_config = config_obj.get_service_config(service)
        if service_config:
            typer.echo(f"Model: {service_config.model}")
            typer.echo(f"Variant: {service_config.variant}")
            typer.echo(f"Args: {service_config.args}")
        else:
            typer.echo(f"No config for {service}")
    elif action == "delete":
        config_obj.set_service_config(service, None)
        config_obj.save_to_disk()
        typer.echo(f"Deleted config for {service}")


# ============================================================================
#  Models / Benchmark / Discover (existing — unchanged)
# ============================================================================

@app.command()
def models(
    gpu_vram: Annotated[
        Optional[int],
        typer.Option(help="Available GPU VRAM in GB (auto-detect if not set)"),
    ] = None,
    refresh_cache: Annotated[
        bool,
        typer.Option(help="Force refresh cache (clear old trending models)"),
    ] = False,
    smart: Annotated[
        bool,
        typer.Option(help="Use LLM to classify models from READMEs (slower but smarter)"),
    ] = False,
) -> None:
    """Show popular models to try with llm-orchestrate."""

    async def _models() -> None:
        # Clear cache if requested
        if refresh_cache:
            import os
            cache_file = Path.home() / ".cache" / "llm-orchestrator" / "models-cache.json"
            if cache_file.exists():
                os.remove(cache_file)
                typer.echo("🔄 Cache cleared, fetching fresh models...\n")

        # Try to fetch trending models from HuggingFace with hardware awareness
        trending = await ModelDiscovery.fetch_trending_models(
            available_vram_gb=gpu_vram,
            enable_llm_classification=smart,
        )

        # Fall back to models.json if empty
        if not any(trending.values()):
            import json

            models_file = Path(__file__).parent.parent / "models.json"
            if models_file.exists():
                try:
                    with open(models_file) as f:
                        data = json.load(f)
                    trending = data.get("models", {})
                except (json.JSONDecodeError, IOError):
                    pass

        # Tier labels and descriptions
        tier_labels = {
            "safe": "🟢 SAFE & RELIABLE",
            "ambitious": "🟡 AMBITIOUS",
            "experimental": "🔴 EXPERIMENTAL & RISKY",
        }

        tier_descriptions = {
            "safe": "Conservative models - high success rate, fast startup",
            "ambitious": "Larger models - better quality, higher resource needs",
            "experimental": "Large/cutting-edge models - may fail, requires optimization",
        }

        # Show hardware info if detected
        if gpu_vram:
            typer.echo(f"GPU VRAM: {gpu_vram}GB (specified)\n")
        else:
            detected_vram = ModelDiscovery.detect_gpu_vram()
            if detected_vram:
                typer.echo(f"GPU VRAM: {detected_vram}GB (detected)\n")
            else:
                typer.echo("GPU VRAM: Unknown (use --gpu-vram to specify)\n")

        typer.echo("LLM Models by Ambition Level\n")
        typer.echo("Usage: ./llm-orchestrate start vllm --model <model-id>\n")

        # Display each tier
        for tier_name in ["safe", "ambitious", "experimental"]:
            if tier_name not in trending or not trending[tier_name]:
                continue

            tier_label = tier_labels[tier_name]
            tier_desc = tier_descriptions[tier_name]
            models_list = trending[tier_name]

            typer.echo(f"\n{tier_label}")
            typer.echo(f"{tier_desc}")
            typer.echo("-" * 80)

            for i, model in enumerate(models_list, 1):
                typer.echo(f"\n  {i}. {model['name']}")
                typer.echo(f"     {model['desc']}")
                context_str = (
                    f" | Context: {model['context'] // 1024}K tokens"
                    if model.get("context")
                    else ""
                )
                model_type_str = ""
                if model.get("model_type") and model.get("model_type") != "unknown":
                    model_type_str = f" | Type: {model['model_type'].capitalize()}"
                typer.echo(
                    f"     VRAM: {model['vram']}{context_str}{model_type_str}  |  {model['risk']}"
                )

        typer.echo("\n" + "=" * 80)
        typer.echo("STRATEGY:")
        typer.echo("  • Start with 🟢 SAFE models to verify setup")
        typer.echo(
            "  • Try 🟡 AMBITIOUS when comfortable (orchestrator will retry with Q4)"
        )
        typer.echo("  • Attempt 🔴 EXPERIMENTAL at your own risk")
        typer.echo("  • Tip: Use quantization (Q4) for 40-50% memory reduction")
        typer.echo("\nEXAMPLES:")
        typer.echo(
            "  ./llm-orchestrate start vllm --model mistralai/Mistral-7B-Instruct-v0.3"
        )
        typer.echo(
            "  ./llm-orchestrate start vllm --model meta-llama/Llama-3.1-70B-Instruct"
        )
        typer.echo("\nDISCOVER VARIANTS:")
        typer.echo("  ./llm-orchestrate discover Qwen/Qwen2.5-32B-Instruct")
        typer.echo("  (Shows quantized versions: Q4, Q5, etc.)")

    run_async(_models())


@app.command()
def benchmark(
    models_arg: Annotated[
        str,
        typer.Argument(help="Models to compare (comma-separated or space-separated)"),
    ],
) -> None:
    """Compare benchmark scores across models."""

    async def _benchmark() -> None:
        # Parse model IDs
        models_list = [
            m.strip() for m in models_arg.replace(",", " ").split() if m.strip()
        ]

        if not models_list:
            typer.echo("Error: Provide at least one model ID", err=True)
            return

        typer.echo(f"Fetching benchmarks for {len(models_list)} model(s)...\n")

        # Fetch benchmark data for each model
        results: dict[str, dict[str, Any]] = {}
        for model_id in models_list:
            typer.echo(f"  {model_id}...", nl=False)
            benchmarks = ModelDiscovery.get_benchmarks(model_id)
            results[model_id] = benchmarks
            typer.echo(" ✓")

        # Define benchmark categories
        categories = [
            ("reasoning", "Reasoning (MMLU/ARC/HellaSwag)"),
            ("math", "Math (MATH/GSM8K)"),
            ("coding", "Coding (HumanEval/MBPP)"),
            ("tool_use", "Tool Use"),
            ("agentic", "Agentic"),
            ("engineering", "Engineering (SWE-Bench)"),
            ("medicine", "Medicine (MedQA/MedMCQA)"),
        ]

        # Build comparison table
        typer.echo("\n" + "=" * 100)
        typer.echo("BENCHMARK COMPARISON")
        typer.echo("=" * 100)

        # Header row
        header = "Benchmark".ljust(35)
        for model_id in models_list:
            header += f" | {model_id[:28]:28}"
        typer.echo(header)
        typer.echo("-" * 100)

        # Data rows
        for key, label in categories:
            row = label.ljust(35)
            has_data = False
            for model_id in models_list:
                if key in results[model_id]:
                    score = results[model_id][key]["score"]
                    row += f" | {score:>7.2f}%".ljust(30)
                    has_data = True
                else:
                    row += f" | {'—':>7}".ljust(30)

            if has_data:
                typer.echo(row)

        typer.echo("=" * 100)
        typer.echo(
            "\nNote: Scores extracted from model cards. Availability varies by model."
        )
        typer.echo("For comprehensive benchmarks, check:")
        typer.echo(
            "  • Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
        )
        typer.echo(
            "  • LMSYS Chatbot Arena: https://huggingface.co/spaces/lmsys/chatbot-arena-elo"
        )

    run_async(_benchmark())


@app.command()
def discover(
    model: Annotated[str, typer.Argument(help="Model ID to search for")],
) -> None:
    """Discover model variants on HuggingFace."""

    async def _discover() -> None:
        typer.echo(f"Discovering variants for {model}...")
        variants = await ModelDiscovery.find_variants(model)

        if variants:
            for name, info in variants.items():
                size_str = (
                    f"{info['size_gb']:.2f}GB"
                    if info["size_gb"] > 0
                    else "size unknown"
                )
                shard_str = f" ({info['shards']} shards)" if info.get("shards") else ""
                typer.echo(f"  {name}: {size_str} ({info['format']}){shard_str}")
            typer.echo(
                f"\nFound {len(variants)} variant(s). Models are available on HuggingFace."
            )
            if any(v["size_gb"] == 0.0 for v in variants.values()):
                typer.echo(
                    "Note: Exact file sizes unavailable for large models. Files are confirmed to exist."
                )
        else:
            typer.echo("No variants found. Model may not exist or have no model files.")

    run_async(_discover())


# ============================================================================
#  NEW COMMANDS: stack, restart, plan
# ============================================================================

@app.command()
def stack() -> None:
    """Show full stack state: GPUs, services, ports, pids."""
    snapshot = StackDetector.capture_snapshot()
    _render_snapshot(snapshot)


def _render_snapshot(snapshot: StackSnapshot) -> None:
    """Render a StackSnapshot as a terminal table."""
    typer.echo("")
    typer.echo("=" * 80)
    typer.echo("  LLM Stack — Current State")
    typer.echo("=" * 80)

    # ── GPUs ─────────────────────────────────────────────────────────
    typer.echo("")
    typer.echo("  GPUs:")
    typer.echo("  " + "-" * 76)
    has_pcie = any("pcie_link_gen" in g for g in snapshot.gpus) if snapshot.gpus else False
    if has_pcie:
        typer.echo(
            f"  {'GPU':>4}  {'Name':<30}  {'Mem Used':>9}  {'Mem Total':>9}  "
            f"{'Util':>5}  {'Temp':>5}  {'PCIe Link':<30}"
        )
    else:
        typer.echo(
            f"  {'GPU':>4}  {'Name':<35}  {'Mem Used':>9}  {'Mem Total':>9}  "
            f"{'Util':>5}  {'Temp':>5}"
        )
    if snapshot.gpus:
        for gpu in snapshot.gpus:
            idx = gpu.get("index", "?")
            name = gpu.get("name", "?")[:30 if has_pcie else 34]
            used_mb = gpu.get("memory_used_mb", 0)
            total_mb = gpu.get("memory_total_mb", 0)
            util = gpu.get("utilization_pct", 0)
            temp = gpu.get("temperature_c", 0)
            used_gb = f"{used_mb / 1024:.1f}G"
            total_gb = f"{total_mb / 1024:.1f}G"

            if "pcie_link_gen" in gpu:
                pcie_str = EnvironmentDetector.pcie_status(
                    gpu["pcie_link_gen"], gpu["pcie_link_gen_max"],
                    gpu["pcie_link_width"], gpu["pcie_link_width_max"]
                )
            else:
                pcie_str = ""

            if has_pcie:
                typer.echo(
                    f"  {idx:>4}  {name:<30}  {used_gb:>9}  {total_gb:>9}  "
                    f"{util:>4}%  {temp:>3}C  {pcie_str:<30}"
                )
            else:
                typer.echo(
                    f"  {idx:>4}  {name:<35}  {used_gb:>9}  {total_gb:>9}  "
                    f"{util:>4}%  {temp:>3}C"
                )
    else:
        typer.echo("  (no GPU data — nvidia-smi unavailable)")

    # ── Services ─────────────────────────────────────────────────────
    typer.echo("")
    typer.echo("  Services:")
    typer.echo("  " + "-" * 76)
    typer.echo(
        f"  {'Name':<12}  {'Status':<10}  {'Model':<30}  {'Port':>5}  {'PID':>7}  {'GPUs'}"
    )

    all_known = set(default_stack_configs().keys())
    running_names = set()
    for name, info in snapshot.services.items():
        running_names.add(name)
        status_icon = "🟢 running" if info.status == "running" else "🔴 stopped"
        model_display = (info.model or "(none)")[:29]
        gpus_str = ",".join(str(g) for g in info.gpus) if info.gpus else "-"
        typer.echo(
            f"  {name:<12}  {status_icon:<10}  {model_display:<30}  {info.port:>5}  {info.pid:>7}  {gpus_str}"
        )

    # Show known services that are NOT running
    for svc_name in sorted(all_known - running_names):
        known_cfg = default_stack_configs().get(svc_name)
        model_display = (known_cfg.model if known_cfg else "(none)")[:29]
        port = known_cfg.port if known_cfg else "-"
        typer.echo(
            f"  {svc_name:<12}  {'⚪ stopped':<10}  {model_display:<30}  {port:>5}  {'N/A':>7}  -"
        )

    # ── Footer ───────────────────────────────────────────────────────
    typer.echo("")
    typer.echo(f"  Snapshot: {snapshot.timestamp.isoformat()}")
    typer.echo("=" * 80)
    typer.echo("")


@app.command()
def restart(
    services: Annotated[
        Optional[list[str]],
        typer.Argument(help="Service name(s) to restart (default: all)"),
    ] = None,
) -> None:
    """Restart one or all services with dependency ordering."""
    stack_cfgs = default_stack_configs()

    if services:
        target_names = services
    else:
        target_names = list(stack_cfgs.keys())

    # Filter to only known services
    target_names = [n for n in target_names if n in stack_cfgs]
    if not target_names:
        typer.echo("No valid services specified.", err=True)
        sys.exit(1)

    # Build sub-config of only target services for proper ordering
    target_cfgs = {n: stack_cfgs[n] for n in target_names}
    mgr = ServiceManager(target_cfgs)

    typer.echo(f"\n🔄 Restarting services: {', '.join(target_names)}")
    typer.echo("-" * 60)

    t0 = _time.time()

    # Stop in reverse dependency order
    stop_names = stop_order(target_cfgs)
    for name in stop_names:
        typer.echo(f"  ⏹  Stopping {name}... ", nl=False)
        result = mgr.stop(name)
        record_event(
            "stop", name, port=result.pid, success=result.success,
            error=result.error or None,
        )
        if result.success:
            typer.echo(f"✓ {result.message}")
        else:
            typer.echo(f"✗ {result.error or result.message}")

    # Brief grace period
    _time.sleep(2)

    # Start in dependency order
    start_names = start_order(target_cfgs)
    for name in start_names:
        typer.echo(f"  ▶   Starting {name}... ", nl=False)
        cfg = stack_cfgs[name]
        result = mgr.start(name)
        duration = _time.time() - t0
        record_event(
            "start", name,
            model=cfg.model, args=cfg.args, port=cfg.port,
            gpu=cfg.gpus, success=result.success, pid=result.pid,
            error=result.error or None, duration_s=duration,
        )
        if result.success:
            typer.echo(f"✓ {result.message}")
        else:
            typer.echo(f"✗ {result.error or result.message}")

    # Log the restart event
    record_event(
        "restart", target_names[0] if len(target_names) == 1 else "all",
        note=f"services={','.join(target_names)}",
    )

    typer.echo("-" * 60)

    # Show updated state
    snapshot = StackDetector.capture_snapshot()
    _render_snapshot(snapshot)


@app.command()
def plan(
    goal: Annotated[
        Optional[str],
        typer.Argument(help="Natural-language goal (e.g., 'move chat to both gpus')"),
    ] = None,
    diff: Annotated[
        bool,
        typer.Option("--diff", help="Diff mode: specify desired state as JSON or flags"),
    ] = False,
    json_state: Annotated[
        Optional[str],
        typer.Option("--json", help="Desired state JSON (for --diff mode)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show plan only, do not execute"),
    ] = False,
) -> None:
    """Plan and execute stack changes from natural language or diff mode.

    NATURAL LANGUAGE:
        llm-orchestrate plan "move chat to both gpus"
        llm-orchestrate plan "restart vllm"
        llm-orchestrate plan "stop reranker"

    DIFF MODE:
        llm-orchestrate plan --diff --json '{"vllm":{"gpus":[0,1]}}'
    """
    snapshot = StackDetector.capture_snapshot()
    current_services = snapshot.services if snapshot.services else {}

    if diff:
        # ── Diff mode ────────────────────────────────────────────────
        if not json_state:
            typer.echo(
                "Error: --diff requires --json with desired state.\n"
                "  Example: --json '{\"vllm\":{\"gpus\":[0,1]}}'",
                err=True,
            )
            sys.exit(1)
        try:
            desired = json.loads(json_state)
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid JSON: {e}", err=True)
            sys.exit(1)

        planner = StackPlanner(snapshot)
        plan_result = planner.diff(desired)

    elif goal:
        # ── Natural language mode ────────────────────────────────────
        planner = StackPlanner(snapshot)
        plan_result = planner.natural_language(goal)
    else:
        typer.echo(
            "Provide a goal string or use --diff with --json.\n"
            '  llm-orchestrate plan "move chat to both gpus"\n'
            '  llm-orchestrate plan --diff --json \'{"vllm":{"gpus":[0,1]}}\'',
            err=True,
        )
        sys.exit(1)

    # ── Display the plan ─────────────────────────────────────────────
    typer.echo(str(plan_result))
    typer.echo("")

    if not plan_result.steps:
        typer.echo("  No changes needed.")
        return

    # ── Confirmation ─────────────────────────────────────────────────
    if plan_result.requires_confirmation and not dry_run:
        answer = typer.prompt("  Execute this plan?", default="n")
        if answer.lower() not in ("y", "yes"):
            typer.echo("  Plan cancelled.")
            return

    if dry_run:
        typer.echo("  [DRY RUN — no actions taken]")
        return

    # ── Execute the plan ─────────────────────────────────────────────
    stack_cfgs = default_stack_configs()
    mgr = ServiceManager(stack_cfgs)

    total_steps = len(plan_result.steps)
    success_count = 0
    fail_count = 0

    for idx, step in enumerate(plan_result.steps, start=1):
        tag = _cli_action_tag(step.action)
        svc = step.service or "--"
        progress = f"[{idx}/{total_steps}]"
        typer.echo(f"  {progress} [{tag}] {svc}  — {step.details}")

        ok = _execute_step(step, mgr, stack_cfgs, current_services)
        if ok:
            success_count += 1
        else:
            fail_count += 1
            typer.echo(f"         ⚠  Step failed: {step.rollback}")

    typer.echo("")
    typer.echo(
        f"  Done: {success_count}/{total_steps} steps succeeded"
        + (f", {fail_count} failed" if fail_count else "")
    )

    if fail_count:
        typer.echo(
            f"  Rollback hint: review failed steps and run 'llm-orchestrate restart' to recover."
        )

    # Persist desired state if it was from diff mode
    if diff and json_state:
        try:
            desired = json.loads(json_state)
            cfg = OrchestratorConfig.load_from_disk()
            for svc_name, state in desired.items():
                cfg.set_desired(svc_name, state)
            cfg.save()
        except Exception:
            pass

    # Show updated state
    typer.echo("")
    updated = StackDetector.capture_snapshot()
    _render_snapshot(updated)


def _cli_action_tag(action: str) -> str:
    mapping = {
        "STOP_SERVICE": "STOP",
        "START_SERVICE": "START",
        "RESTART_SERVICE": "RESTART",
        "VERIFY_PORT": "VERIFY",
        "WAIT_GRACE": "WAIT",
        "KILL_ORPHAN": "KILL",
    }
    return mapping.get(action, action)


def _execute_step(
    step: PlanStep,
    mgr: ServiceManager,
    stack_cfgs: dict,
    current_services: dict,
) -> bool:
    """Execute a single PlanStep and return True on success."""
    name = step.service
    if not name or name not in stack_cfgs:
        # WAIT, VERIFY without service, etc.
        if step.action == "WAIT_GRACE":
            secs = step.estimated_seconds
            _time.sleep(secs)
            return True
        return True

    if step.action == "STOP_SERVICE":
        result = mgr.stop(name)
        return result.success

    if step.action == "START_SERVICE":
        # Parse GPU overrides from step details if possible
        overrides: dict[str, Any] = {}
        result = mgr.start(name, **overrides)
        return result.success

    if step.action == "RESTART_SERVICE":
        result = mgr.restart(name)
        return result.success

    if step.action == "VERIFY_PORT":
        # Quick port check
        import socket
        cfg = stack_cfgs.get(name)
        if not cfg:
            return True
        port = cfg.port
        reachable = StackDetector.verify_endpoint("127.0.0.1", port, timeout=5)
        return reachable

    if step.action == "WAIT_GRACE":
        _time.sleep(step.estimated_seconds)
        return True

    return True


# ============================================================================
#  History
# ============================================================================

from llm_orchestrator.history import read_events, history_stats, HISTORY_DIR


@app.command()
def history(
    service: Annotated[
        Optional[str],
        typer.Option("--service", "-s", help="Filter by service name"),
    ] = None,
    action: Annotated[
        Optional[str],
        typer.Option("--action", "-a", help="Filter by action (start/stop/restart)"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Number of events to show"),
    ] = 20,
) -> None:
    """Show service start/stop/restart history from ~/.llmconf/history."""
    events = read_events(service=service, action=action, limit=limit)

    stats = history_stats()
    typer.echo("")
    typer.echo("📜 Service History")
    typer.echo("=" * 80)
    typer.echo(
        f"  Total events: {stats['total']}  |  "
        f"Successes: {stats['successes']}  |  "
        f"Failures: {stats['failures']}"
    )
    if stats.get("by_service"):
        svc_parts = [f"{s}={c}" for s, c in stats["by_service"].items()]
        typer.echo(f"  Services: {', '.join(svc_parts)}")
    typer.echo("")

    if not events:
        typer.echo("  (no events found)")
        typer.echo("")
        return

    typer.echo(f"  {'Time (UTC)':<22}  {'Action':<9}  {'Service':<10}  {'Model':<30}  {'Port':>5}  {'GPU':<6}  {'OK':<3}  {'Duration':<8}  {'Env'}")  # noqa
    typer.echo("  " + "-" * 130)

    for ev in events:
        ts = ev.get("ts", "")[:19].replace("T", " ")
        action = ev.get("action", "?")
        svc = ev.get("service", "?")
        model = (ev.get("model") or "")[:29]
        port = ev.get("port", "-")
        gpu = ",".join(str(g) for g in (ev.get("gpu") or [])) or "-"
        ok = "✓" if ev.get("success") else "✗"
        dur = f"{ev['duration_s']:.0f}s" if ev.get("duration_s") else ""
        note = ev.get("note") or (ev.get("error") or "")

        # Build env summary
        env_dict = ev.get("env", {})
        if env_dict:
            env_parts = [f"{k}={v}" for k, v in list(env_dict.items())[:3]]
            env_str = " ".join(env_parts)
            if len(env_dict) > 3:
                env_str += f" (+{len(env_dict) - 3})"
        else:
            env_str = note

        typer.echo(
            f"  {ts:<22}  {action:<9}  {svc:<10}  {model:<30}  {port:>5}  "
            f"{gpu:<6}  {ok:<3}  {dur:<8}  {env_str}"
        )

    typer.echo("")
    typer.echo(f"  Full log: {HISTORY_DIR / 'events.jsonl'}")
    typer.echo("")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
