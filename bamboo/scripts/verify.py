#!/usr/bin/env python
"""Verify that the Bamboo package is correctly installed.

After installation, run from any directory:

    bamboo verify
"""

import asyncio
import importlib.resources
import subprocess
import sys
from pathlib import Path


def _ok(msg: str) -> bool:
    print(f"  ✓ {msg}")
    return True


def _fail(msg: str, hint: str = "") -> bool:
    print(f"  ✗ {msg}")
    if hint:
        print(f"    → {hint}")
    return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> bool:
    print("Python version")
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 10):
        return _ok(label)
    return _fail(label, "Bamboo requires Python 3.10 or higher.")


def check_package_importable() -> bool:
    print("Package import")
    try:
        import bamboo  # noqa: F401

        return _ok(f"bamboo {bamboo.__version__} imported successfully")
    except ImportError as exc:
        return _fail(f"cannot import bamboo: {exc}", "Run: pip install .")


def check_submodule_imports() -> bool:
    print("Sub-module imports")
    modules = [
        "bamboo.config",
        "bamboo.cli",
        "bamboo.models.graph_element",
        "bamboo.models.knowledge_entity",
        "bamboo.llm.llm_client",
        "bamboo.llm.prompts",
        "bamboo.database.base",
        "bamboo.database.factory",
        "bamboo.database.graph_database_client",
        "bamboo.database.vector_database_client",
        "bamboo.database.backends.neo4j_backend",
        "bamboo.database.backends.qdrant_backend",
        "bamboo.extractors.base",
        "bamboo.extractors.factory",
        "bamboo.agents.knowledge_accumulator",
        "bamboo.agents.reasoning_navigator",
        "bamboo.workflows.knowledge_workflow",
        "bamboo.workflows.reasoning_workflow",
    ]
    ok = True
    for mod in modules:
        try:
            __import__(mod)
            _ok(mod)
        except ImportError as exc:
            _fail(mod, str(exc))
            ok = False
    return ok


def check_cli_entry_points() -> bool:
    print("CLI entry points")
    ok = True
    for args in (
        ["bamboo", "--help"],
        ["bamboo", "interactive", "--help"],
        ["bamboo", "populate", "--help"],
        ["bamboo", "analyze", "--help"],
        ["bamboo", "verify", "--help"],
    ):
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
        )
        cmd_str = " ".join(args)
        if result.returncode == 0:
            _ok(f"`{cmd_str}` works")
        else:
            _fail(
                f"`{cmd_str}` not found or errored",
                "Run: pip install .  (entry points are registered on install)",
            )
            ok = False
    return ok


def check_key_dependencies() -> bool:
    print("Key dependencies")
    deps = [
        ("langchain_core", "langchain-core"),
        ("langgraph", "langgraph"),
        ("langchain_openai", "langchain-openai"),
        ("langchain_anthropic", "langchain-anthropic"),
        ("neo4j", "neo4j"),
        ("qdrant_client", "qdrant-client"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("dotenv", "python-dotenv"),
        ("click", "click"),
        ("rich", "rich"),
    ]
    ok = True
    for mod, pkg in deps:
        try:
            __import__(mod)
            _ok(pkg)
        except ImportError:
            _fail(pkg, f"Run: pip install {pkg}")
            ok = False
    return ok


def check_docker() -> bool:
    print("Docker (optional — needed to run databases locally)")
    try:
        r = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        r2 = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, text=True
        )
        _ok(r.stdout.strip())
        _ok(r2.stdout.strip())
        return True
    except FileNotFoundError:
        _fail(
            "Docker not found",
            "Install Docker Desktop: https://www.docker.com/products/docker-desktop/",
        )
        return False


def _env_example_path() -> str:
    """Return the absolute path of the installed ``.env.example`` file."""
    try:
        ref = importlib.resources.files("bamboo.data").joinpath(".env.example")
        return str(ref)
    except Exception:
        return "<bamboo-install-dir>/bamboo/data/.env.example"


def _check_duplicate_env_keys(env_path: str) -> list[str]:
    """Return a list of keys that appear more than once in the .env file."""
    from collections import Counter
    counts: Counter[str] = Counter()
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key = line.split("=", 1)[0].strip()
                counts[key] += 1
    return [key for key, n in counts.items() if n > 1]


def check_api_keys() -> bool:
    print("API keys / settings")

    from bamboo.config import _find_env_file

    env_path = _find_env_file()
    if not env_path:
        example = _env_example_path()
        user_cfg = Path.home() / ".config" / "bamboo" / ".env"
        _fail(
            ".env file not found",
            f"Copy the example and edit it:\n"
            f"    cp {example} .env          # project-local (any parent of cwd)\n"
            f"    # or for a permanent user-level location:\n"
            f"    mkdir -p {user_cfg.parent} && cp {example} {user_cfg}\n"
            f"Then re-run: bamboo verify",
        )
        return False

    _ok(f".env loaded from: {env_path}")

    # Warn about duplicate keys — the last value wins but the first is silently ignored.
    dupes = _check_duplicate_env_keys(env_path)
    if dupes:
        for key in dupes:
            _fail(
                f"Duplicate key in .env: {key}",
                f"Remove the duplicate — only the last value is used.\n"
                f"    Open {env_path} and keep only one {key}= line.",
            )
        return False

    # ------------------------------------------------------------------ #
    # Load settings and report active values                              #
    # ------------------------------------------------------------------ #
    try:
        from bamboo.config import get_settings

        s = get_settings()
    except Exception as exc:
        return _fail(
            f"could not load settings: {exc}",
            f"Check that {env_path} is valid and re-run bamboo verify",
        )

    _ok(
        f"embeddings_provider={s.embeddings_provider!r}  "
        f"embedding_model={s.embedding_model!r}  "
        f"embedding_dimension={s.embedding_dimension}"
    )

    ok = True

    # LLM key — not required for ollama (runs locally)
    if s.llm_provider == "ollama":
        _ok(f"LLM provider: ollama / {s.llm_model}  (no API key required)")
        import urllib.request

        ollama_base = getattr(s, "ollama_base_url", "http://localhost:11434")
        try:
            urllib.request.urlopen(ollama_base, timeout=2)
            _ok(f"Ollama server is reachable at {ollama_base}")
        except Exception:
            _fail(
                f"Ollama server not reachable at {ollama_base}",
                "Run:  ollama serve\n"
                f"    then pull the model:  ollama pull {s.llm_model}",
            )
            ok = False
    else:
        if s.llm_api_key:
            _ok(
                f"LLM_API_KEY is set  (provider: {s.llm_provider}, model: {s.llm_model})"
            )
        else:
            _fail(
                "LLM_API_KEY is not set",
                "Add LLM_API_KEY=<your-key> to your .env file",
            )
            ok = False

    # Embeddings key — only needed when EMBEDDINGS_PROVIDER=openai
    if s.embeddings_provider == "openai":
        if s.effective_embeddings_api_key:
            _ok(f"Embeddings: openai / {s.embedding_model}  (API key is set)")
        else:
            _fail(
                "Embeddings API key is not set",
                "Add EMBEDDINGS_API_KEY=<your-openai-key> to your .env file\n"
                "    (or set EMBEDDINGS_PROVIDER=local to use free local embeddings)",
            )
            ok = False
    else:  # local
        _ok(
            f"Embeddings: local / sentence-transformers / {s.embedding_model}  "
            "(no API key required)"
        )
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            _fail(
                "sentence-transformers is not installed",
                "Run:  pip install sentence-transformers langchain-huggingface",
            )
            ok = False

    return ok


def check_database_connections() -> bool:
    """Connect to Neo4j and Qdrant, creating the vector collection if absent."""
    print("Database connections")

    async def _probe():
        from bamboo.database.graph_database_client import GraphDatabaseClient
        from bamboo.database.vector_database_client import VectorDatabaseClient

        results = {}

        # --- graph database (Neo4j) ---
        neo4j = GraphDatabaseClient()
        try:
            await neo4j.connect()
            results["neo4j"] = True
        except Exception as exc:
            results["neo4j_error"] = str(exc)
            results["neo4j"] = False
        finally:
            try:
                await neo4j.close()
            except Exception:
                pass

        # --- vector database (Qdrant) ---
        qdrant = VectorDatabaseClient()
        try:
            # connect() calls _ensure_collection(), which creates the
            # collection if it does not yet exist — intentional side-effect.
            await qdrant.connect()
            collection_existed = await qdrant.collection_exists()
            results["qdrant"] = True
            results["qdrant_collection_created"] = not collection_existed
            # Smoke-test: run a real search to catch server/client version
            # mismatches early (e.g. qdrant-client ≥ 1.10 uses /query which
            # requires Qdrant server ≥ 1.10; older servers return a bare 404).
            from bamboo.config import get_settings as _gs
            _dim = _gs().embedding_dimension
            await qdrant.search_similar(
                query_embedding=[0.0] * _dim,
                limit=1,
                score_threshold=0.0,
            )
            results["qdrant_search_ok"] = True
        except Exception as exc:
            results["qdrant_error"] = str(exc)
            results["qdrant"] = False
        finally:
            try:
                await qdrant.close()
            except Exception:
                pass

        return results

    try:
        results = asyncio.run(_probe())
    except Exception as exc:
        return _fail(f"database probe failed: {exc}")

    ok = True

    if results.get("neo4j"):
        _ok("graph database (Neo4j) is reachable")
    else:
        _fail(
            f"graph database (Neo4j) not reachable: {results.get('neo4j_error', '?')}",
            "Start Neo4j with:  docker compose up -d",
        )
        ok = False

    if results.get("qdrant"):
        if results.get("qdrant_collection_created"):
            _ok("vector database (Qdrant) is reachable — collection created")
        else:
            _ok("vector database (Qdrant) is reachable — collection already exists")
    else:
        _fail(
            f"vector database (Qdrant) not reachable or search failed: {results.get('qdrant_error', '?')}",
            "If Qdrant is not running:  docker compose up -d\n"
            "    If you see a 404 error: your Qdrant server is too old for the\n"
            "    installed qdrant-client. Update with:  docker compose pull && docker compose up -d",
        )
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("Bamboo Installation Verification")
    print("=" * 70)

    sections = [
        check_python_version,
        check_package_importable,
        check_submodule_imports,
        check_cli_entry_points,
        check_key_dependencies,
        check_api_keys,
        check_database_connections,
    ]

    results = []
    for fn in sections:
        print(f"\n[{fn.__name__.replace('check_', '').replace('_', ' ').title()}]")
        results.append(fn())

    # Docker is optional — run but don't count towards pass/fail
    print(
        f"\n[{check_docker.__name__.replace('check_', '').replace('_', ' ').title()}]"
    )
    check_docker()

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} checks passed — Bamboo is ready to use!")
        print()
        print("Next steps:")
        print("  bamboo populate --task-id <id>   # ingest your first task")
        print("  bamboo interactive               # start the reasoning session")
        print()
        print("Full guide: docs/QUICKSTART.md  (in the project source)")
        return 0
    else:
        print(f"✗ {total - passed} check(s) failed  ({passed}/{total} passed)")
        print()
        print("Fix the issues above, then re-run: bamboo verify")
        return 1


if __name__ == "__main__":
    sys.exit(main())
