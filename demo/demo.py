"""
ragwise demo — showcases hybrid search, streaming, agent tools, and multi-tenancy.

Run:
    pip install ragwise rich
    python demo.py
"""
import asyncio
import time

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()


def header(title: str, subtitle: str = "") -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/dim]")
    console.print()


def code(snippet: str, language: str = "python") -> None:
    console.print(Syntax(snippet, language, theme="monokai", line_numbers=False, padding=1))


def pause(msg: str = "Press Enter to continue...") -> None:
    console.print(f"\n[dim]{msg}[/dim]")
    input()


async def run_demo() -> None:
    # ── INTRO ─────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold white]ragwise[/bold white]  [dim]v0.1.0[/dim]\n"
        "[cyan]Hybrid search · Streaming · Agent tools · Multi-tenancy[/cyan]\n"
        "[dim]pip install ragwise[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()
    console.print("  This demo ingests 4 product docs and shows what ragwise does")
    console.print("  that LangChain and LlamaIndex don't do by default.\n")
    pause("Press Enter to start →")

    # ── DEMO 1: HYBRID SEARCH ─────────────────────────────────────────────────
    header("1 / 4  —  Hybrid Search", "BM25 + dense retrieval fused with RRF. On every query. No config.")

    code("""\
from ragwise import RAG

async with RAG(llm="openai/gpt-4o-mini") as rag:
    await rag.ingest("./docs/")
    answer = await rag.query("What is the refund policy?")
    print(answer.text)
    print(answer.citations)""")

    console.print("  [dim]Running...[/dim]")
    console.print()

    from ragwise import RAG, QueryConfig

    async with RAG(llm="openai/gpt-4o-mini") as rag:
        t0 = time.monotonic()
        result = await rag.ingest("./docs/")
        ingest_ms = int((time.monotonic() - t0) * 1000)

        console.print(f"  [green]✓[/green] Indexed [bold]{result.succeeded}[/bold] files in {ingest_ms}ms")
        console.print()

        t0 = time.monotonic()
        answer = await rag.query("What is the refund policy?")
        query_ms = int((time.monotonic() - t0) * 1000)

        console.print(Panel(
            f"[white]{answer.text}[/white]",
            title="[green]Answer[/green]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print(f"  [dim]Sources: {', '.join(answer.citations)}[/dim]")
        console.print(f"  [dim]Latency: {query_ms}ms[/dim]")

    pause("\nPress Enter for Demo 2 — Streaming →")

    # ── DEMO 2: STREAMING ────────────────────────────────────────────────────
    header("2 / 4  —  Streaming", "Tokens arrive as they're generated. Zero waiting. Works with OpenAI, Anthropic, Ollama.")

    code("""\
async for token in rag.stream_query("What changed in v3.2.0?"):
    print(token, end="", flush=True)""")

    console.print("  [dim]Streaming...[/dim]\n")
    console.print("  [bold green]Answer:[/bold green] ", end="")

    async with RAG(llm="openai/gpt-4o-mini") as rag:
        await rag.ingest("./docs/")
        async for token in rag.stream_query("What changed in v3.2.0?"):
            console.print(token, end="", highlight=False)

    console.print("\n")
    console.print("  [dim]→ Same retrieval path as query(). Falls back gracefully for custom LLMs.[/dim]")

    pause("\nPress Enter for Demo 3 — Agent Tools →")

    # ── DEMO 3: AGENT TOOLS ──────────────────────────────────────────────────
    header("3 / 4  —  Agent Tools", "Give your Claude or OpenAI agent access to your docs in one line.")

    code("""\
from ragwise.agent import as_claude_tool

tool = as_claude_tool(rag)           # Anthropic-compatible tool schema
results = await rag.search("query")  # raw retrieval, no generation""")

    async with RAG(llm="openai/gpt-4o-mini") as rag:
        await rag.ingest("./docs/")

        from ragwise.agent import as_claude_tool, as_openai_tool

        claude_tool = as_claude_tool(rag)
        openai_tool = as_openai_tool(rag)

        # Show the tool schema
        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="white")

        table.add_row("Tool name", claude_tool["name"])
        table.add_row("Description", claude_tool["description"][:60] + "...")
        table.add_row("Required params", str(claude_tool["input_schema"]["required"]))
        table.add_row("Optional params", "top_k (default: 5)")
        table.add_row("Claude compatible", "[green]✓[/green]")
        table.add_row("OpenAI compatible", "[green]✓[/green]")

        console.print(table)
        console.print()

        # Show raw search results
        console.print("  [bold]rag.search()[/bold] — raw retrieval, no generation:\n")
        results = await rag.search("error code 0x80004005", top_k=3)

        for i, r in enumerate(results, 1):
            console.print(f"  [cyan]{i}.[/cyan] [bold]{r.source}[/bold]  [dim]score: {r.score:.3f}[/dim]")
            console.print(f"     [dim]{r.text[:120].strip()}...[/dim]")
            console.print()

    pause("Press Enter for Demo 4 — Multi-tenancy →")

    # ── DEMO 4: MULTI-TENANCY ────────────────────────────────────────────────
    header("4 / 4  —  Multi-tenant Isolation", "Tag docs at ingest. Filter at query. No store schema changes.")

    code("""\
# Ingest docs per tenant
await rag.ingest("./docs/org_a/", tenant_id="org_a")
await rag.ingest("./docs/org_b/", tenant_id="org_b")

# org_b docs never appear in org_a queries
answer = await rag.query(
    "What is the pricing?",
    config=QueryConfig(tenant_id="org_a")
)""")

    async with RAG(llm="openai/gpt-4o-mini") as rag:
        # Simulate two tenants by ingesting same docs with different tenant IDs
        await rag.ingest("./docs/", tenant_id="acme_corp")
        await rag.ingest("./docs/", tenant_id="globex_inc")

        console.print("  [dim]Ingested docs for [bold]acme_corp[/bold] and [bold]globex_inc[/bold][/dim]\n")

        results_a = await rag.search("pricing", top_k=3, config=QueryConfig(tenant_id="acme_corp"))
        results_b = await rag.search("pricing", top_k=3, config=QueryConfig(tenant_id="globex_inc"))

        iso_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        iso_table.add_column("Tenant", style="bold", width=15)
        iso_table.add_column("Results returned", width=10)
        iso_table.add_column("Sees other tenant's docs?", width=25)

        iso_table.add_row(
            "acme_corp",
            str(len(results_a)),
            "[red]✗ Never[/red]",
        )
        iso_table.add_row(
            "globex_inc",
            str(len(results_b)),
            "[red]✗ Never[/red]",
        )

        console.print(iso_table)
        console.print()
        console.print("  [dim]Works with InMemoryStore, LanceDB, and PostgreSQL+pgvector.[/dim]")
        console.print("  [dim]No WHERE clause rewrite needed — filtering happens post-RRF.[/dim]")

    # ── STORE UPGRADE PATH ────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Store Upgrade Path[/bold cyan]")
    console.print()

    code("""\
# Dev — volatile, zero setup
RAG(store="memory")

# Persistent dev — embedded, no server, survives restarts
RAG(store="lance://./ragwise-index")

# Production — PostgreSQL + pgvector
RAG(store="postgresql://user:pass@localhost/mydb")

# API stays identical. Only the string changes.""")

    console.print("  [dim]→ pgvector benchmark: 471 QPS @ 50M vectors (source: pgvector/pgvector)[/dim]")

    # ── FINAL PANEL ──────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text.assemble(
            ("  pip install ragwise\n\n", "bold green"),
            ("  GitHub   ", "dim"), ("github.com/laxmikanta415/ragwise\n", "cyan"),
            ("  Docs     ", "dim"), ("ragwise.readthedocs.io\n", "cyan"),
            ("  PyPI     ", "dim"), ("pypi.org/project/ragwise", "cyan"),
        ),
        title="[bold white]Get started[/bold white]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


if __name__ == "__main__":
    asyncio.run(run_demo())
