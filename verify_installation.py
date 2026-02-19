#!/usr/bin/env python
"""Verification script for Bamboo installation."""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_file_exists(filepath):
    """Check if file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {filepath}")
        return True
    else:
        print(f"✗ {filepath} (missing)")
        return False


def check_directory_exists(dirpath):
    """Check if directory exists."""
    path = Path(dirpath)
    if path.is_dir():
        count = sum(1 for _ in path.rglob('*.py'))
        print(f"✓ {dirpath}/ ({count} Python files)")
        return True
    else:
        print(f"✗ {dirpath}/ (missing)")
        return False


def check_docker():
    """Check if Docker is available."""
    print("\nChecking Docker...")
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Docker not found (optional for local development)")
        return False


def check_docker_compose():
    """Check if Docker Compose is available."""
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Docker Compose not found (optional for local development)")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("Bamboo Installation Verification")
    print("=" * 70)

    all_checks = []

    # Python version
    all_checks.append(check_python_version())

    # Core files
    print("\nChecking core files...")
    all_checks.append(check_file_exists("requirements.txt"))
    all_checks.append(check_file_exists("setup.py"))
    all_checks.append(check_file_exists("pyproject.toml"))
    all_checks.append(check_file_exists("docker-compose.yml"))
    all_checks.append(check_file_exists("examples/.env.example"))

    # Documentation
    print("\nChecking documentation...")
    all_checks.append(check_file_exists("README.md"))
    all_checks.append(check_file_exists("QUICKSTART.md"))
    all_checks.append(check_file_exists("ARCHITECTURE.md"))
    all_checks.append(check_file_exists("DEVELOPMENT.md"))

    # Package structure
    print("\nChecking package structure...")
    all_checks.append(check_directory_exists("bamboo"))
    all_checks.append(check_directory_exists("bamboo/models"))
    all_checks.append(check_directory_exists("bamboo/database"))
    all_checks.append(check_directory_exists("bamboo/llm"))
    all_checks.append(check_directory_exists("bamboo/extractors"))
    all_checks.append(check_directory_exists("bamboo/agents"))
    all_checks.append(check_directory_exists("bamboo/workflows"))
    all_checks.append(check_directory_exists("bamboo/scripts"))
    all_checks.append(check_directory_exists("bamboo/utils"))

    # Examples and tests
    print("\nChecking examples and tests...")
    all_checks.append(check_directory_exists("examples"))
    all_checks.append(check_directory_exists("tests"))

    # Docker (optional)
    check_docker()
    check_docker_compose()

    # Environment check
    print("\nChecking environment configuration...")
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file exists")
        print("  → Make sure to set your API keys!")
    else:
        print("⚠ .env file not found")
        print("  → Run: cp .env.example .env")
        print("  → Then edit .env and add your API keys")

    # Summary
    print("\n" + "=" * 70)
    required_checks = sum(all_checks)
    total_checks = len(all_checks)

    if required_checks == total_checks:
        print(f"✓ All {total_checks} required checks passed!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Start databases: make docker-up")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Try the examples: python -m bamboo.cli interactive")
        print("\nSee QUICKSTART.md for detailed instructions.")
        return 0
    else:
        print(f"✗ {total_checks - required_checks} checks failed")
        print(f"✓ {required_checks}/{total_checks} checks passed")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

