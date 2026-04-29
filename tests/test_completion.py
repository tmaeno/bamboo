"""Tests that enforce consistency between cli.py and the packaged completion scripts.

Completion scripts are generated from the Click CLI via `bamboo verify`.
If either test here fails, run `bamboo verify` and commit the updated files in
``bamboo/data/completions/``.
"""

from pathlib import Path

import importlib.resources as pkg_resources


def _packaged_script(filename: str) -> str:
    ref = pkg_resources.files("bamboo.data.completions").joinpath(filename)
    return ref.read_text(encoding="utf-8")


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

    def test_completion_scripts_match_click_output(self):
        """Packaged completion files must match Click's generated output.

        If this fails, run `bamboo verify` to regenerate them, then commit.
        """
        from click.shell_completion import BashComplete, ZshComplete
        from bamboo.cli import cli

        for ShellClass, filename in [
            (ZshComplete, "_bamboo"),
            (BashComplete, "_bamboo.bash"),
        ]:
            expected = ShellClass(cli, {}, "bamboo", "_BAMBOO_COMPLETE").source()
            actual = _packaged_script(filename)
            assert actual == expected, (
                f"bamboo/data/completions/{filename} is out of sync with cli.py.\n"
                f"Run `bamboo verify` to regenerate it, then commit."
            )
