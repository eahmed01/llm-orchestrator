"""CLI: Command-line interface for llm-orchestrator."""

import asyncio
import logging
import sys
from typing import Any, Coroutine, Optional, TypeVar

import typer
from typing_extensions import Annotated

from llm_orchestrator.config import OrchestratorConfig
from llm_orchestrator.model_discovery import ModelDiscovery
from llm_orchestrator.orchestrator import Orchestrator

app = typer.Typer(help="Automate trial-and-error LLM startup with intelligent retry")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async function, handling both existing and new event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running; create one
        return asyncio.run(coro)
    else:
        # Event loop already running (e.g., in tests); create task
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


@app.command()
def start(
    service: Annotated[str, typer.Argument(help="Service to start (e.g., 'vllm'")] = "vllm",
    model: Annotated[Optional[str], typer.Option(help="Model to load")] = None,
    max_retries: Annotated[int, typer.Option(help="Max retry attempts")] = 5,
) -> None:
    """Start an LLM service with intelligent retry."""
    async def _start() -> None:
        orchestrator = Orchestrator(service=service, model=model)
        orchestrator.MAX_RETRIES = max_retries

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
def models() -> None:
    """Show popular models to try with llm-orchestrate."""

    tiers = {
        "🟢 SAFE & RELIABLE": {
            "emoji": "🟢",
            "desc": "Conservative models - high success rate, fast startup",
            "models": [
                {
                    "name": "Qwen/Qwen2.5-7B-Instruct",
                    "desc": "General purpose - fast and reliable",
                    "vram": "14GB",
                    "risk": "✓ Low risk",
                },
                {
                    "name": "mistralai/Mistral-7B-Instruct-v0.1",
                    "desc": "Efficient 7B - proven stable",
                    "vram": "14GB",
                    "risk": "✓ Low risk",
                },
                {
                    "name": "meta-llama/Llama-2-7b-chat-hf",
                    "desc": "Widely compatible, battle-tested",
                    "vram": "14GB",
                    "risk": "✓ Low risk",
                },
            ],
        },
        "🟡 AMBITIOUS": {
            "emoji": "🟡",
            "desc": "Larger models - better quality, higher resource needs",
            "models": [
                {
                    "name": "Qwen/Qwen1.5-14B-Chat",
                    "desc": "14B - significantly better reasoning",
                    "vram": "28GB",
                    "risk": "⚠ Medium risk - may OOM on smaller GPUs",
                },
                {
                    "name": "meta-llama/Llama-2-13b-chat-hf",
                    "desc": "13B - good quality with 26GB requirement",
                    "vram": "26GB",
                    "risk": "⚠ Medium risk - tight memory constraints",
                },
                {
                    "name": "Qwen/Qwen1.5-32B-Chat",
                    "desc": "32B - substantial capability jump",
                    "vram": "64GB",
                    "risk": "⚠ Medium risk - requires high-end GPU",
                },
                {
                    "name": "mistralai/Mistral-34B-Instruct-v0.1",
                    "desc": "34B - excellent reasoning, challenging to fit",
                    "vram": "68GB",
                    "risk": "⚠ Medium risk - needs substantial VRAM",
                },
            ],
        },
        "🔴 EXPERIMENTAL & RISKY": {
            "emoji": "🔴",
            "desc": "Large/cutting-edge models - may fail, requires optimization",
            "models": [
                {
                    "name": "meta-llama/Llama-2-70b-chat-hf",
                    "desc": "70B - state-of-the-art, very memory hungry",
                    "vram": "80GB+",
                    "risk": "❌ High risk - may not fit, may OOM mid-inference",
                },
                {
                    "name": "Qwen/Qwen1.5-72B-Chat",
                    "desc": "72B - cutting edge, requires extreme resources",
                    "vram": "90GB+",
                    "risk": "❌ High risk - orchestrator will try Q4 fallback",
                },
                {
                    "name": "meta-llama/Llama-2-70b-chat-hf",
                    "desc": "70B-Q4 (quantized) - reduced memory, lower quality",
                    "vram": "35GB",
                    "risk": "⚠ Medium-high risk - quality trade-off",
                },
                {
                    "name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                    "desc": "Mixtral 8x7B - MoE model, unpredictable memory",
                    "vram": "60GB+",
                    "risk": "❌ High risk - sparse MoE needs careful tuning",
                },
            ],
        },
    }

    typer.echo("LLM Models by Ambition Level\n")
    typer.echo("Usage: ./llm-orchestrate start vllm --model <model-id>\n")

    for tier_name, tier_info in tiers.items():
        typer.echo(f"\n{tier_name}")
        typer.echo(f"{tier_info['desc']}")
        typer.echo("-" * 80)

        for i, model in enumerate(tier_info["models"], 1):
            typer.echo(f"\n  {i}. {model['name']}")
            typer.echo(f"     {model['desc']}")
            typer.echo(f"     VRAM: {model['vram']}  |  {model['risk']}")

    typer.echo("\n" + "=" * 80)
    typer.echo("STRATEGY:")
    typer.echo("  • Start with 🟢 SAFE models to verify setup")
    typer.echo("  • Try 🟡 AMBITIOUS when comfortable (orchestrator will retry with Q4)")
    typer.echo("  • Attempt 🔴 EXPERIMENTAL at your own risk")
    typer.echo("  • Tip: Use quantization (Q4) for 40-50% memory reduction")
    typer.echo("\nEXAMPLES:")
    typer.echo("  ./llm-orchestrate start vllm --model Qwen/Qwen2.5-7B-Instruct")
    typer.echo("  ./llm-orchestrate start vllm --model meta-llama/Llama-2-70b-chat-hf")
    typer.echo("\nDISCOVER VARIANTS:")
    typer.echo("  ./llm-orchestrate discover Qwen/Qwen1.5-32B-Chat")
    typer.echo("  (Shows quantized versions: Q4, Q5, etc.)")



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
                size_str = f"{info['size_gb']:.2f}GB" if info['size_gb'] > 0 else "size unknown"
                shard_str = f" ({info['shards']} shards)" if info.get('shards') else ""
                typer.echo(f"  {name}: {size_str} ({info['format']}){shard_str}")
            typer.echo(f"\nFound {len(variants)} variant(s). Models are available on HuggingFace.")
            if any(v['size_gb'] == 0.0 for v in variants.values()):
                typer.echo("Note: Exact file sizes unavailable for large models. Files are confirmed to exist.")
        else:
            typer.echo("No variants found. Model may not exist or have no model files.")

    run_async(_discover())


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()


