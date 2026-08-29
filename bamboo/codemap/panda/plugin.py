"""PanDA Code Map plugin.

One map covers ``pandaserver`` *and* ``pandajedi``: they are separate Python
packages but a single distribution (``panda-server-source``) that always
deploys at one version.  A JEDI knight decides and a server-side proxy writes,
so the two halves of a single junction routinely sit in different packages --
splitting the map by package would turn those handoffs into cross-system
boundaries and demand version bindings for something that is not another
system.

Extraction is stdlib ``ast`` only.  A census over both versions showed ~92% of
writes to promoted subjects resolve with no interprocedural dataflow at all,
so a heavier analysis framework would buy the remaining few percent at the
cost of a build-time dependency.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

from bamboo.codemap.base import CodeMapPlugin
from bamboo.codemap.models import MapFragment
from bamboo.codemap.panda.recognizers import errorcode

logger = logging.getLogger(__name__)

PACKAGES = ("pandaserver", "pandajedi")


class PandaCodeMapPlugin(CodeMapPlugin):
    """Builds the ``panda`` Code Map from installed or checked-out source."""

    def __init__(self) -> None:
        self._roots: dict[str, Path] = {}
        self._version: str = ""
        # (package, rel_path, tree, source) for every parsed module.
        self._modules: list[tuple[str, str, ast.Module, str]] = []

    @property
    def map_id(self) -> str:
        return "panda"

    def prepare(self, source_root: Optional[Path] = None) -> str:
        """Resolve the source snapshot, parse it, and return the version stamp."""
        self._roots = self._resolve_roots(source_root)
        if not self._roots:
            raise FileNotFoundError(
                "Neither pandaserver nor pandajedi found. Install the 'panda' "
                "extra or pass an explicit source root."
            )
        self._version = self._resolve_version(source_root)
        self._modules = self._parse_modules(self._roots)
        logger.info(
            "PandaCodeMapPlugin: parsed %d module(s) from %s",
            len(self._modules),
            self._version,
        )
        return self._version

    def run(self) -> MapFragment:
        """Extract the fragment.  Requires :meth:`prepare` to have run."""
        if not self._modules:
            raise RuntimeError("prepare() must be called before run()")

        fragment = MapFragment(map_id=self.map_id, derived_from=self._version)
        enums, coverage = errorcode.extract(self._modules, self.map_id, self._version)
        fragment.value_enums.extend(enums)
        fragment.coverage.extend(coverage)
        logger.info(
            "PandaCodeMapPlugin: %d value enumeration(s) across %d file(s)",
            len(enums),
            len(coverage),
        )
        return fragment

    # -- source discovery ------------------------------------------------- #

    @staticmethod
    def _resolve_roots(source_root: Optional[Path]) -> dict[str, Path]:
        """Return ``{package: root}``.

        With *source_root* the packages come from that directory, which is how
        a named release is mapped instead of whatever happens to sit next to
        bamboo.  Otherwise the installed distribution is used, resolved via
        ``importlib.metadata`` rather than ``sys.modules`` so that a
        monkey-patched import cannot redirect it.
        """
        roots: dict[str, Path] = {}
        if source_root is not None:
            base = Path(source_root).expanduser().resolve()
            for pkg in PACKAGES:
                if (base / pkg).is_dir():
                    roots[pkg] = base / pkg
            return roots

        from importlib.metadata import (
            Distribution,
            PackageNotFoundError,
            packages_distributions,
        )

        pkg_to_dist = packages_distributions()
        for pkg in PACKAGES:
            for dist_name in pkg_to_dist.get(pkg, []):
                try:
                    dist = Distribution.from_name(dist_name)
                except PackageNotFoundError:
                    continue
                pkg_dir = Path(dist.locate_file("")).resolve() / pkg
                if pkg_dir.is_dir():
                    roots[pkg] = pkg_dir
                    break
        return roots

    @staticmethod
    def _resolve_version(source_root: Optional[Path]) -> str:
        """Return the stamp recorded on every node of this build."""
        if source_root is not None:
            return f"source-root:{Path(source_root).expanduser().resolve()}"
        from importlib.metadata import (
            Distribution,
            PackageNotFoundError,
            packages_distributions,
        )

        for dist_name in packages_distributions().get(PACKAGES[0], []):
            try:
                return f"{dist_name} {Distribution.from_name(dist_name).version}"
            except PackageNotFoundError:
                continue
        return "unknown"

    @staticmethod
    def _parse_modules(
        roots: dict[str, Path],
    ) -> list[tuple[str, str, ast.Module, str]]:
        """Parse every ``.py`` once; files that will not parse are skipped.

        Recognizers are run over the whole tree rather than a file list, so a
        module added upstream is picked up on the next build with no
        configuration change.
        """
        modules: list[tuple[str, str, ast.Module, str]] = []
        for pkg, root in roots.items():
            for path in sorted(root.rglob("*.py")):
                try:
                    source = path.read_text(errors="replace")
                    tree = ast.parse(source, filename=str(path))
                except SyntaxError:
                    logger.debug("skipping unparseable module: %s", path)
                    continue
                rel = f"{pkg}/{path.relative_to(root)}"
                modules.append((pkg, rel, tree, source))
        return modules
