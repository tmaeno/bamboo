"""Version stamping and content hashing for an analysed source tree.

Both answer the same question from different angles: *exactly which bytes did
this build read?*  Without that a map's conclusions are unfalsifiable -- an
explanation drawn from code is only as good as knowing which version of the
code it came from, and the map is routinely built from a snapshot that is not
what production is running.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SEC = 10


def describe(root: Path) -> Optional[str]:
    """Return ``git describe --tags --always --dirty`` for *root*, or ``None``.

    ``--dirty`` is not optional here.  A map built from a modified working
    tree cannot be reproduced, and a stamp that quietly omits that is worse
    than having no stamp: it claims a reproducibility the build does not have.

    Returns ``None`` when *root* is not a git checkout (an installed
    distribution, an unpacked tarball), which is not an error -- the caller
    falls back to whatever identity that source form offers.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git describe failed in %s: %s", root, exc)
        return None
    if result.returncode != 0:
        logger.debug("git describe returned %d in %s", result.returncode, root)
        return None
    return result.stdout.strip() or None


def blob_sha(content: str) -> str:
    """Return the git-format blob SHA-1 of *content*.

    Computed rather than read from git so it is available for source that is
    not a checkout at all -- an installed distribution has no object store but
    its files still need a content identity, and a map built from one has to be
    comparable to a map built from the other.

    Hashes the *decoded* text, which is what the parser saw.  That is the
    property an anchor needs: in a modified working tree the committed object
    and the file on disk differ, and the anchor describes the latter.

    Using git's formulation (``blob <len>\\0<content>``) makes the value equal
    to ``git rev-parse HEAD:<path>`` for a clean, LF-terminated file -- which
    is every file in PanDA.  It would diverge for CRLF source, since decoding
    normalises the line endings that git hashes verbatim.
    """
    payload = content.encode()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 -- git's format
