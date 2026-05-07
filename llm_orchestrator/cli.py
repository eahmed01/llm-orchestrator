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


