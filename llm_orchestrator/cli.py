"""CLI: Command-line interface for llm-orchestrator."""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Coroutine, Optional, TypeVar

import typer
from typing_extensions import Annotated

from llm_orchestrator.config import OrchestratorConfig, UserPreferences
from llm_orchestrator.environment import (
    EnvironmentDetector,
    ask_gpu_preference,
    ask_interface_preference,
    ask_port_preference,
)
from llm_orchestrator.model_discovery import ModelDiscovery
from llm_orchestrator.orchestrator import Orchestrator

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
            typer.echo(
                f"  {status} GPU {gpu['index']}: {gpu['name']} - "
                f"{memory_total}GB ({memory_used}/{memory_total}GB used, {usage_pct:.0f}%){marker}"
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

        try:
            success = await orchestrator.start()
            if not success:
                sys.exit(1)
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
                typer.echo(
                    f"     VRAM: {model['vram']}{context_str}  |  {model['risk']}"
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
        typer.echo(
            "Use --gpu-vram to see hardware-aware recommendations for these models.\n"
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


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
