"""The code a reader is given for one junction, chosen by the map.

The map's job here is retrieval, not reading: it says *which function, marked at
which line, against which log line*, and something else makes sense of that.
Keeping the choice here rather than in whoever asks is what makes a reading
reproducible -- the same junction yields the same text, so a wrong answer can be
blamed on the input or on the reader rather than on both at once.

Why a function, measured rather than assumed.  An arm's own guard is small (a
median of eight lines), but only 35 of 927 arms refer to nothing outside it:
96% load names they do not bind, and 52% sit in a loop whose header is outside
the guard, so the region does not even say what is being iterated.  Following
those references reaches a median of 37% of the enclosing function and 96% of
it at the ninetieth percentile.  A slicer would be a mechanism for arriving at
very nearly the function.

Why by line and not by name.  ``panda_source_navigator._read_method`` takes the
last segment of a qualname and returns the first function ``ast.walk`` finds
with that name; 38 files define some name more than once, 99 times over.  It
happens to be right for all but three of 497 junctions, but the map records the
line, so the ambiguity need not be entered into at all.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from bamboo.codemap.gitsource import blob_sha as git_blob_sha

logger = logging.getLogger(__name__)


class Region(BaseModel):
    """One function, as it was when the map was built."""

    owner: str
    file: str
    line_start: int
    line_end: int
    text: str
    blob_sha: str = Field(
        default="",
        description="Hash of the file this text was actually read from.",
    )
    expected_sha: str = Field(
        default="",
        description="Hash the map recorded for it, when the caller supplied one.",
    )
    gloss_key: str = Field(..., description="Hash of `text`; the key of its reading.")

    @property
    def off_version(self) -> bool:
        """Whether the file read is not the file the map was built from.

        The failure this guards is silent and was live the first time this ran:
        a map built from the installed 1.0.2 was read against a checkout, and
        the report printed line 4971 of the wrong version -- plausible code,
        the wrong code.  Line drift is one of the two skew symptoms no gate
        catches, so the only defence is to notice the file changed at all and
        say so.
        """
        return bool(self.expected_sha and self.blob_sha and self.expected_sha != self.blob_sha)

    def marked(self, lines: list[int]) -> str:
        """The text with *lines* marked, for a reader that must pick one arm.

        A marker rather than an excerpt: cutting the text down to the marked
        lines is the thing the measurement above rules out, and cutting it down
        *after* saying the whole function is needed would be the same mistake
        wearing a different name.
        """
        wanted = set(lines)
        out = []
        for offset, line in enumerate(self.text.splitlines()):
            number = self.line_start + offset
            out.append(f"{'>>>' if number in wanted else '   '} {number:5d}  {line}")
        return "\n".join(out)


def key_for(text: str) -> str:
    """The key a reading of *text* is stored under.

    The text, not the position and not the junction's meaning.  A reading is
    only ever about what it was shown, so a function that changed is a
    different key rather than a stale entry -- which is why this is not
    :meth:`JunctionNode.content_hash`, whose whole point is to ignore the
    surrounding source so that a gloss survives edits elsewhere in the file.
    That is the right key for something derived from the branch table and the
    wrong one for something derived from the file.
    """
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _containing(tree: ast.Module, line: int, owner: str) -> Optional[ast.AST]:
    """The function definition *line* falls in.

    Where definitions nest, the one whose name the map already recorded wins
    and the innermost is the fallback.  Neither is a guess about which function
    matters: the map names one, and containment settles it when it does not.
    """
    wanted = owner.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    holders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if not holders:
        return None
    named = [node for node in holders if node.name == wanted]
    pool = named or holders
    return min(pool, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)


def region_in(
    source: str,
    tree: ast.Module,
    *,
    file: str,
    owner: str,
    line: int,
    expected_sha: str = "",
) -> Optional[Region]:
    """The region for *line* in an already-parsed module.

    Split from :func:`region_for` so that a build, which has every module
    parsed already, does not read and re-parse the same file once per junction.
    """
    holder = _containing(tree, line, owner)
    if holder is None:
        return None
    start, end = holder.lineno, holder.end_lineno or holder.lineno
    text = "\n".join(source.splitlines()[start - 1 : end])
    return Region(
        owner=owner,
        file=file,
        line_start=start,
        line_end=end,
        text=text,
        blob_sha=git_blob_sha(source),
        expected_sha=expected_sha,
        gloss_key=key_for(text),
    )


def region_for(
    roots: dict[str, Path],
    *,
    file: str,
    owner: str,
    line: int,
    expected_sha: str = "",
) -> Optional[Region]:
    """The region for *line*, read from a source tree.

    *file* is package-prefixed, as the map records it; *roots* maps a package
    name to the directory holding it, the shape
    ``PandaCodeMapPlugin._resolve_roots`` returns.  Returns ``None`` rather
    than raising when the file is absent or will not parse: a map built from
    one snapshot is routinely read against another, and a missing region is a
    thing to report, not an error to stop on.
    """
    package, _, relative = file.partition("/")
    root = roots.get(package)
    if root is None or not relative:
        return None
    path = root / relative
    try:
        source = path.read_text(errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, ValueError) as exc:
        logger.debug("region_for: %s unreadable (%s)", file, exc)
        return None
    return region_in(
        source, tree, file=file, owner=owner, line=line, expected_sha=expected_sha
    )


def attach(fragment, modules: list) -> int:
    """Give every junction and stage the key of the reading that explains it.

    Eager because the *choice* of input is part of the map -- the same question
    must select the same text however it is asked -- while the reading itself
    stays lazy.  Filling in 213 readings at build time would pay for every
    function on every version, and the distribution is the argument against it:
    the junctions sit in 213 of 3078 functions and an investigation touches a
    handful.

    Returns how many nodes got a key.  A node whose function cannot be located
    keeps an empty one rather than a guess, and the caller reports the count --
    five exist, all writes at module scope, where there is no enclosing
    function to hand anyone.
    """
    parsed = {module.rel_path: module for module in modules}
    keyed = 0
    for node in list(fragment.junctions) + list(fragment.filter_stages):
        anchor = getattr(node, "anchor", None)
        if anchor is None:
            continue
        module = parsed.get(anchor.file)
        if module is None:
            continue
        region = region_in(
            module.source,
            module.tree,
            file=anchor.file,
            owner=node.owner,
            line=anchor.line_start,
            expected_sha=module.blob_sha,
        )
        if region is None:
            continue
        node.gloss_key = region.gloss_key
        keyed += 1
    return keyed
