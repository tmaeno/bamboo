"""Tests that enforce consistency between cli.py and the zsh completion script.

These tests are the enforcement mechanism that prevents drift between
``bamboo/data/completions/_bamboo`` and the Click command tree in ``cli.py``.

If any test here fails it means either:
  - A new subcommand was added to cli.py  → add it to _bamboo
  - A new option was added to a command   → add it to _bamboo
  - _bamboo was changed but cli.py wasn't → update cli.py
"""

from pathlib import Path

import importlib.resources as pkg_resources
import click


def _packaged_script(filename: str) -> str:
    ref = pkg_resources.files("bamboo.data.completions").joinpath(filename)
    return ref.read_text(encoding="utf-8")


def _script_lines() -> set[str]:
    return {line.strip() for line in _packaged_script("_bamboo").splitlines()}


class TestZshCompletionConsistency:
    def test_installed_file_matches_package_data(self):
        """~/.zsh/completions/_bamboo must be byte-for-byte identical to the package data."""
        installed = Path.home() / ".zsh" / "completions" / "_bamboo"
        if not installed.exists():
            return  # not installed on this machine (CI / non-zsh) — skip

        packaged = _packaged_script("_bamboo")
        actual = installed.read_text(encoding="utf-8")
        assert actual == packaged, (
            "\n\n~/.zsh/completions/_bamboo is out of sync with "
            "bamboo/data/completions/_bamboo.\n"
            "Run any `bamboo` command to resync, then `exec zsh`.\n"
        )

    def test_package_data_file_is_valid_zsh(self):
        """Basic sanity check that the file is a well-formed zsh compdef script."""
        script = _packaged_script("_bamboo")
        assert script.startswith("#compdef bamboo"), "Must start with '#compdef bamboo'"
        assert "_bamboo()" in script, "Must define _bamboo()"
        assert '_bamboo "$@"' in script, 'Must call _bamboo "$@" at the end'

    def test_all_subcommands_in_completion_script(self):
        """Every subcommand in cli.py must appear in _bamboo.

        Add the subcommand to bamboo/data/completions/_bamboo when this fails.
        """
        from bamboo.cli import cli

        script = _packaged_script("_bamboo")
        missing = [name for name in cli.commands if name not in script]
        assert not missing, (
            f"Subcommand(s) registered in cli.py but missing from "
            f"bamboo/data/completions/_bamboo:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    def test_no_stale_subcommands_in_completion_script(self):
        """Every subcommand in _bamboo must still exist in cli.py.

        Remove the stale entry from bamboo/data/completions/_bamboo when this fails.
        """
        from bamboo.cli import cli

        script = _packaged_script("_bamboo")

        # Extract subcommand names from the commands() array in the script.
        # They appear as lines like: "  'fetch-task:...'"
        import re

        in_script = set(re.findall(r"'([a-z][a-z0-9-]+):[^']*'", script))
        registered = set(cli.commands.keys())
        stale = in_script - registered
        assert not stale, (
            f"Subcommand(s) in bamboo/data/completions/_bamboo no longer exist in cli.py:\n"
            + "\n".join(f"  - {name}" for name in sorted(stale))
        )

    def test_all_options_in_completion_script(self):
        """Every option of every subcommand in cli.py must appear in _bamboo.

        Update bamboo/data/completions/_bamboo when this fails.
        """
        from bamboo.cli import cli

        script = _packaged_script("_bamboo")
        missing: list[str] = []

        for cmd_name, cmd in cli.commands.items():
            if not isinstance(cmd, click.Command):
                continue
            for param in cmd.params:
                if not isinstance(param, click.Option):
                    continue
                # Check at least one of the option's names appears in the script.
                if not any(opt in script for opt in param.opts):
                    missing.append(f"  bamboo {cmd_name}: {'/'.join(param.opts)}")

        assert not missing, (
            f"Option(s) registered in cli.py but missing from "
            f"bamboo/data/completions/_bamboo:\n"
            + "\n".join(missing)
            + "\n\nUpdate bamboo/data/completions/_bamboo to add them."
        )
