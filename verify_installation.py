#!/usr/bin/env python
"""Verify that the Bamboo package is correctly installed.

Can be run from any directory after installation:

    pip install .
    python verify_installation.py

Or directly without changing directory:

    python /path/to/bamboo/verify_installation.py
"""

import subprocess
import sys


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
        return _fail(f"cannot import bamboo: {exc}",
                     "Run: pip install .")


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
    for cmd in ("bamboo", "bamboo-populate", "bamboo-analyze"):
        result = subprocess.run(
            [cmd, "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            _ok(f"`{cmd} --help` works")
        else:
            _fail(f"`{cmd}` not found or errored",
                  "Run: pip install .  (entry points are registered on install)")
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
        r2 = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
        _ok(r.stdout.strip())
        _ok(r2.stdout.strip())
        return True
    except FileNotFoundError:
        _fail("Docker not found",
              "Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
        return False


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
    ]

    results = []
    for fn in sections:
        print(f"\n[{fn.__name__.replace('check_', '').replace('_', ' ').title()}]")
        results.append(fn())

    # Docker is optional — run but don't count towards pass/fail
    print(f"\n[{check_docker.__name__.replace('check_', '').replace('_', ' ').title()}]")
    check_docker()

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} checks passed — Bamboo is ready to use!")
        print()
        print("Next steps:")
        print("  1. cp /path/to/bamboo/examples/.env.example .env")
        print("     Edit .env and add your API keys (OPENAI_API_KEY / ANTHROPIC_API_KEY)")
        print("  2. docker compose -f /path/to/bamboo/docker-compose.yml up -d")
        print("  3. bamboo interactive")
        print()
        print("Full guide: docs/getting-started/QUICKSTART.md  (in the project source)")
        return 0
    else:
        print(f"✗ {total - passed} check(s) failed  ({passed}/{total} passed)")
        print()
        print("Fix the issues above, then re-run: python verify_installation.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())

