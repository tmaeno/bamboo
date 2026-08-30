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
import re
from pathlib import Path
from typing import Optional

from bamboo.codemap.base import CodeMapPlugin
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.gitsource import describe as _git_describe
from bamboo.codemap.models import JunctionNode, MapFragment, SourceModule, SubjectNode
from bamboo.codemap.panda.attribution import SpecAttributor, class_bases
from bamboo.codemap.panda.recognizers import boundary, errorcode, progress, sqlwrite

logger = logging.getLogger(__name__)


def _merge_junctions(junctions: list[JunctionNode]) -> list[JunctionNode]:
    """Combine junctions sharing a name, keeping every branch.

    Two recognizers can reach the same write site -- ``updateTaskStatus...``
    sets ``taskSpec.status`` *and* binds ``:status`` -- and both readings are
    real branches of the same junction rather than rival descriptions of it.
    """
    merged: dict[str, JunctionNode] = {}
    for junction in junctions:
        existing = merged.get(junction.name)
        if existing is None:
            merged[junction.name] = junction
            continue
        known = {(b.outcome, tuple(b.path_condition)) for b in existing.branches}
        for branch in junction.branches:
            if (branch.outcome, tuple(branch.path_condition)) in known:
                continue
            branch.order = len(existing.branches)
            existing.branches.append(branch)
        if existing.structural_subject is None:
            existing.structural_subject = junction.structural_subject
    return list(merged.values())


def _unique_subjects(subjects: list[SubjectNode]) -> list[SubjectNode]:
    """Collapse subjects the slices found independently, pooling their evidence."""
    merged: dict[str, SubjectNode] = {}
    for subject in subjects:
        existing = merged.get(subject.name)
        if existing is None:
            merged[subject.name] = subject
            continue
        for criterion in subject.criteria:
            if criterion not in existing.criteria:
                existing.criteria.append(criterion)
        if subject.vocabulary and not existing.vocabulary:
            existing.vocabulary = subject.vocabulary
    return list(merged.values())

PACKAGES = ("pandaserver", "pandajedi")

# Test code is not the system's behaviour.  It writes the same attributes with
# the same literals, so leaving it in files fixtures as junctions and doubles
# the ambiguous attribution cases -- 88 of 376 modules, and 47% of the writes
# whose spec class could not be settled came from them.  A junction pointing at
# a test would send an investigation to code that never runs in production.
_TEST_PATH = re.compile(r"(^|/)(tests?|jeditest)(/|$)", re.IGNORECASE)
_TEST_FILE = re.compile(r"(^test_|_tests?$)", re.IGNORECASE)


def _is_test_module(rel_path: str) -> bool:
    """True when *rel_path* is test code rather than system behaviour."""
    return bool(_TEST_PATH.search(rel_path) or _TEST_FILE.search(Path(rel_path).stem))


class PandaCodeMapPlugin(CodeMapPlugin):
    """Builds the ``panda`` Code Map from installed or checked-out source."""

    def __init__(self) -> None:
        self._roots: dict[str, Path] = {}
        self._version: str = ""
        self._modules: list[SourceModule] = []

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
        enums, enum_coverage = errorcode.extract(self._modules, self.map_id, self._version)
        fragment.value_enums.extend(enums)
        fragment.coverage.extend(enum_coverage)

        boundaries, boundary_coverage = boundary.extract(
            self._modules, self.map_id, self._version
        )
        fragment.boundaries.extend(boundaries)
        fragment.coverage.extend(boundary_coverage)

        subjects, junctions, progress_coverage = progress.extract(
            self._modules, self.map_id, self._version
        )
        fragment.subjects.extend(subjects)
        fragment.junctions.extend(junctions)
        fragment.coverage.extend(progress_coverage)

        # The SQL slice needs an attributor that already knows the table map,
        # which is learned from the whole corpus rather than from one module.
        attributor = SpecAttributor(
            progress.spec_attributes(self._modules), class_bases(self._modules)
        )
        self._table_conflicts = attributor.learn_table_classes(self._modules)
        sql_subjects, sql_junctions, sql_coverage, self._uncovered_tables = sqlwrite.extract(
            self._modules, self.map_id, self._version, attributor
        )
        fragment.subjects.extend(sql_subjects)
        fragment.junctions.extend(sql_junctions)
        fragment.coverage.extend(sql_coverage)

        # A function that writes a status both ways produces one junction from
        # each recognizer under the same name.  Storage merges on the name, so
        # without this the second silently replaces the first's branches.
        fragment.junctions = _merge_junctions(fragment.junctions)
        fragment.subjects = _unique_subjects(fragment.subjects)

        logger.info(
            "PandaCodeMapPlugin: %d enumeration(s), %d boundary/boundaries, "
            "%d subject(s), %d junction(s)",
            len(enums),
            len(boundaries),
            len(fragment.subjects),
            len(fragment.junctions),
        )
        return fragment

    @property
    def uncovered_tables(self) -> set[str]:
        """Tables written by the code that hold no spec class."""
        return getattr(self, "_uncovered_tables", set())

    @property
    def table_conflicts(self) -> dict[str, set[str]]:
        """Tables whose column evidence named more than one spec class."""
        return getattr(self, "_table_conflicts", {})

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
        """Return the stamp recorded on every node of this build.

        The stamp has to identify the *contents* that were analysed.  A
        filesystem path does not: the same path holds different code tomorrow,
        so two builds would carry the same stamp while describing different
        systems -- which defeats the point of stamping at all, and silently
        breaks the version diffing that condition drift depends on.

        For a git checkout the description is used instead.  ``--dirty``
        matters: a map built from a modified working tree cannot be
        reproduced, and a stamp that hides that is worse than no stamp.
        """
        if source_root is not None:
            root = Path(source_root).expanduser().resolve()
            described = _git_describe(root)
            return f"git:{described}" if described else f"source-root:{root}"

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
    def _parse_modules(roots: dict[str, Path]) -> list[SourceModule]:
        """Parse and hash every ``.py`` once; files that will not parse are skipped.

        Recognizers are run over the whole tree rather than a file list, so a
        module added upstream is picked up on the next build with no
        configuration change.

        The content hash is of what was actually read, not of what git has
        committed -- in a modified working tree those differ, and an anchor
        that pointed at the committed bytes would describe code the build never
        saw.
        """
        modules: list[SourceModule] = []
        skipped_tests = 0
        for pkg, root in roots.items():
            for path in sorted(root.rglob("*.py")):
                rel_path = f"{pkg}/{path.relative_to(root)}"
                if _is_test_module(rel_path):
                    skipped_tests += 1
                    continue
                try:
                    source = path.read_text(errors="replace")
                    tree = ast.parse(source, filename=str(path))
                except SyntaxError:
                    logger.debug("skipping unparseable module: %s", path)
                    continue
                modules.append(
                    SourceModule(
                        package=pkg,
                        rel_path=rel_path,
                        tree=tree,
                        source=source,
                        blob_sha=blob_sha(source),
                    )
                )
        if skipped_tests:
            logger.info("skipped %d test module(s)", skipped_tests)
        return modules
