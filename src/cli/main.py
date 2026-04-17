"""ragwise CLI — init, serve, and doctor commands."""
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


@app.command()
def doctor(
    store: Annotated[str | None, typer.Option("--store", help="Store spec override.")] = None,
    llm: Annotated[str | None, typer.Option("--llm", help="LLM spec override.")] = None,
    embedder: Annotated[str | None, typer.Option("--embedder", help="Embedder spec override.")] = None,
) -> None:
    """Run a health check: credentials, store, hybrid search, latency."""
    import asyncio

    exit_code = asyncio.run(_run_doctor(store=store, llm=llm, embedder=embedder))
    raise typer.Exit(exit_code)


def _ok(msg: str) -> None:
    typer.echo(f"\u2713 {msg}")


def _fail(msg: str, reason: str) -> None:
    typer.echo(f"\u2717 {msg} \u2014 {reason}")


def _warn(msg: str) -> None:
    typer.echo(f"! {msg}")


async def _run_doctor(
    store: str | None,
    llm: str | None,
    embedder: str | None,
) -> int:
    """Returns 0 (pass) or 1 (fail)."""
    from pathlib import Path

    from ragwise.config import RAGConfig

    # Load config from yaml / env, then apply overrides
    cfg_path = Path("ragwise.yaml")
    try:
        cfg = RAGConfig.from_yaml(cfg_path) if cfg_path.exists() else RAGConfig.from_env()
    except Exception:
        cfg = RAGConfig()

    if store:
        cfg = cfg.model_copy(update={"store": store})
    if llm:
        cfg = cfg.model_copy(update={"llm": llm})
    if embedder:
        cfg = cfg.model_copy(update={"embedder": embedder})

    typer.echo("ragwise doctor — checking configuration")
    typer.echo(f"  LLM: {cfg.llm}  |  Embedder: {cfg.embedder}  |  Store: {cfg.store}")
    typer.echo("")

    failures = 0

    # Check 1 — Embedder reachability
    try:
        import time

        from ragwise.embedding.base import resolve_embedder

        emb = resolve_embedder(cfg.embedder)
        t0 = time.perf_counter()
        vecs = await emb.embed(["ragwise doctor test"])
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if vecs and len(vecs[0]) > 0:
            _ok(f"Embedder reachable — dim={len(vecs[0])}, latency={latency_ms}ms")
        else:
            _fail("Embedder", "returned empty vector")
            failures += 1
    except Exception as exc:
        _fail("Embedder", str(exc))
        failures += 1
        return 1  # can't continue without embedder

    # Check 2 — Store reachability + chunk count
    try:
        from ragwise.pipeline import _resolve_store

        store_inst = _resolve_store(cfg.store)
        sources = await store_inst.list_sources()
        _ok(f"Store reachable — {len(sources)} source(s) indexed")
    except Exception as exc:
        _fail("Store", str(exc))
        failures += 1
        store_inst = None

    # Check 3 — Hybrid search round-trip
    if store_inst is not None:
        try:
            from ragwise.indexing.base import EmbeddedDoc
            from ragwise.retrieval.search import HybridSearcher

            test_doc = EmbeddedDoc(
                id="_doctor_test_chunk",
                text="ragwise doctor test chunk for health verification",
                source="_doctor_test",
                embedding=vecs[0],
                metadata={},
            )
            await store_inst.upsert([test_doc])
            searcher = HybridSearcher(store=store_inst, embedder=emb)
            results = await searcher.search("ragwise doctor", top_k=5)
            await store_inst.delete("_doctor_test")

            hit = any(r.id == "_doctor_test_chunk" for r in results)
            if hit:
                _ok("Hybrid search working — test chunk retrieved successfully")
            else:
                _warn("Hybrid search — test chunk not in top-5 (index may be empty)")
        except Exception as exc:
            _fail("Hybrid search", str(exc))
            failures += 1

    # Check 4 — LLM API key / reachability
    try:
        from ragwise.generation.llm import resolve_llm

        llm_inst = resolve_llm(cfg.llm)
        t0 = time.perf_counter()
        await llm_inst.complete("Reply with exactly one word: ok")
        latency_ms = int((time.perf_counter() - t0) * 1000)
        _ok(f"LLM reachable — latency={latency_ms}ms")
    except Exception as exc:
        _fail("LLM", str(exc))
        failures += 1

    typer.echo("")
    if failures == 0:
        typer.echo("All checks passed.")
    else:
        typer.echo(f"{failures} check(s) failed. Fix the issues above and re-run ragwise doctor.")

    return 1 if failures > 0 else 0
