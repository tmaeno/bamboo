"""Code Map plugin contract.

One plugin per *map*, and a map is one **distribution** -- the unit whose
version moves independently -- not one Python package.  ``pandaserver`` and
``pandajedi`` ship together as ``panda-server-source`` and always deploy at the
same version, so they are one map; splitting them by package would turn the
handoffs between them (a JEDI knight decides, a server-side proxy writes) into
cross-system boundaries, which would demand version bindings and interface
contracts for something that is not another system at all.

The contract is stated in terms of what a plugin *produces*, never how it
analyses.  That keeps the door open for a map whose source does not follow the
conventions the current recognizers rely on, without committing the design to
any particular analysis tool now.  Incremental behaviour is left to the
plugin: the contract only says "here is a snapshot, hand back your fragment".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from bamboo.codemap.models import MapFragment


class CodeMapPlugin(ABC):
    """Extracts a Code Map fragment from one target system's source."""

    @property
    @abstractmethod
    def map_id(self) -> str:
        """Stable map identifier, e.g. ``"panda"``.

        Qualifies every node this plugin emits so fragments from different
        maps never collide on a shared name.
        """

    @abstractmethod
    def prepare(self, source_root: Path | None = None) -> str:
        """Resolve the source snapshot and return its version stamp.

        Called before :meth:`run`.  The stamp is recorded on every node as
        ``derived_from``: a root cause explained by code carries no weight
        unless it says which version of that code it came from, and the map is
        routinely built from a snapshot that is not what production is running.

        Args:
            source_root: Analyse this directory instead of whatever is
                installed.  Passing a checkout is how a specific release is
                mapped rather than an accident of the local environment.

        Returns:
            Version stamp, e.g. ``"panda-server-source 1.0.2"``.
        """

    @abstractmethod
    def run(self) -> MapFragment:
        """Extract and return this map's fragment.

        Must be called after :meth:`prepare`.
        """
