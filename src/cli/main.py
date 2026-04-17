"""ragwise CLI — init and serve commands."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="ragwise",
    help="ragwise — Production-grade RAG in 4 lines. Hybrid search on by default.",
)

_CONFIG_TEMPLATE = '''\
from ragwise import RAGConfig

config = RAGConfig(
    embedder="openai/text-embedding-3-small",  # or "local/all-MiniLM-L6-v2" for offline
    store="memory",  # or "lance://./ragwise-index" (persistent) or "postgresql://..."
    llm="openai/gpt-4o-mini",  # or "anthropic/claude-haiku-4-5" or "ollama/llama3"
    chunk_size=512,
    chunk_overlap=64,
)
'''

_DEFAULT_OUTPUT = "ragwise_config.py"


@app.command()
def init(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write the config file."),
    ] = Path(_DEFAULT_OUTPUT),
) -> None:
    """Generate a ragwise_config.py with default settings in the current directory."""
    target = output if output.is_absolute() else Path.cwd() / output
    if target.exists():
        overwrite = typer.confirm(
            f"{target} already exists. Overwrite?", default=False
        )
        if not overwrite:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    target.write_text(_CONFIG_TEMPLATE)
    typer.echo(f"Created {target}")


@app.command()
def serve(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on.")] = 8000,
    host: Annotated[str, typer.Option("--host", help="Host to bind to.")] = "0.0.0.0",
) -> None:
    """Start the ragwise HTTP API server (requires pip install ragwise[serve])."""
    try:
        from cli.serve import run_server
    except ImportError as exc:
        typer.echo("pip install ragwise[serve]")
        raise typer.Exit(1) from exc

    run_server(host=host, port=port)
