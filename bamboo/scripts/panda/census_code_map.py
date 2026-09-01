"""Code Map census — survey the source before fixing the slices.

A **throwaway diagnostic**, not production code.  It answers the questions that
must be settled *before* the Code Map slices are fixed:

* Which spec attributes qualify as **subjects** (and by which promotion criteria)?
* Who **writes** each subject, and which meta-pattern explains each write?
* Which **meta-patterns** actually occur, and where?
* What fraction of writers is left **unexplained** (the discovery frontier)?

Slices are *hypotheses*.  This census exists to confirm or refute them, so it is
deliberately slice-agnostic: it sweeps every file and counts what is there.

Everything runs offline against the installed ``pandaserver`` / ``pandajedi``
source using the stdlib ``ast`` module — no external analysis tool, no LLM, no
database.  Only the parts that survive move into ``bamboo/codemap/census.py``.

Usage::

    # Full report
    python -m bamboo.scripts.panda.census_code_map

    # One section only
    python -m bamboo.scripts.panda.census_code_map --section subjects
    python -m bamboo.scripts.panda.census_code_map --section writers
    python -m bamboo.scripts.panda.census_code_map --section metapatterns

    # Machine-readable, for diffing across PanDA releases
    python -m bamboo.scripts.panda.census_code_map --json out.json

    # Analyse a checkout instead of the installed distribution
    python -m bamboo.scripts.panda.census_code_map --source-root /path/to/panda
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import click

# Packages that make up the PanDA map.  Both ship in one distribution
# (``panda-server-source``), so a single version stamps the whole census.
PACKAGES = ("pandaserver", "pandajedi")

# Attribute-declaration names seen in spec classes.  ``_attributes`` is the
# common one; the leading-underscore variant was missed by a first pass that
# only looked for ``attributes`` -- a naming drift no external tool would have
# caught either, which is precisely why the census exists.
ATTR_DECLS = ("attributes", "_attributes")

_STATUS_RE = re.compile(r"status", re.I)


# --------------------------------------------------------------------------- #
# source discovery
# --------------------------------------------------------------------------- #


def resolve_pkg_roots(source_root: Optional[str]) -> dict[str, Path]:
    """Return ``{package_name: root_path}`` for the packages to analyse.

    With *source_root* the packages are taken from that directory, which is how
    a specific release (git ref / tag) is analysed rather than whatever happens
    to be installed alongside bamboo.  Without it the installed distribution is
    used, resolved through ``importlib.metadata`` so a monkey-patched
    ``sys.modules`` cannot mislead it.
    """
    roots: dict[str, Path] = {}
    if source_root:
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


def resolve_version(source_root: Optional[str]) -> str:
    """Return the version stamp recorded on every census result."""
    if source_root:
        return f"source-root:{source_root}"
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


# --------------------------------------------------------------------------- #
# parsed module corpus
# --------------------------------------------------------------------------- #


@dataclass
class Module:
    """One parsed source file, kept in memory for the whole census."""

    pkg: str
    path: Path
    rel: str  # package-prefixed, e.g. "pandajedi/jediorder/JobGenerator.py"
    tree: ast.Module


def load_modules(roots: dict[str, Path]) -> list[Module]:
    """Parse every ``.py`` under *roots* once; unparseable files are skipped.

    Parent links are attached so path conditions can be recovered by walking up
    from an assignment, which ``ast`` does not provide natively.
    """
    modules: list[Module] = []
    for pkg, root in roots.items():
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
            except SyntaxError:
                continue
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    child.parent = parent  # type: ignore[attr-defined]
            modules.append(Module(pkg, path, f"{pkg}/{path.relative_to(root)}", tree))
    return modules


def _ancestors(node: ast.AST) -> Iterator[ast.AST]:
    cur = getattr(node, "parent", None)
    while cur is not None:
        yield cur
        cur = getattr(cur, "parent", None)


def _enclosing_function(node: ast.AST) -> Optional[str]:
    for anc in _ancestors(node):
        if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return anc.name
    return None


def _under_condition(node: ast.AST) -> bool:
    """True when *node* sits inside an ``if``/``elif``/``else`` body.

    Unconditional writes are bookkeeping (``modificationTime=CURRENT_DATE``);
    guarded ones carry a path condition and are candidate junction points.
    """
    prev = node
    for anc in _ancestors(node):
        if isinstance(anc, ast.If) and (prev in anc.body or prev in anc.orelse):
            return True
        prev = anc
    return False


def _string_constants(tree: ast.AST) -> Iterator[ast.Constant]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node


# --------------------------------------------------------------------------- #
# subject census
# --------------------------------------------------------------------------- #


STRONG_WHERE = "1:state-gate-in-where"
STRONG_VOCAB = "2:declared-vocabulary"
STRONG_CLOSED_SET = "3:closed-literal-set"
# Criteria 4 and 5 corroborate but do not promote -- see promote_subjects().
STRONG_CRITERIA = {STRONG_WHERE, STRONG_VOCAB, STRONG_CLOSED_SET}

# Share of an attribute's writes that must come from its literal set for that
# set to count as closed.  A majority keeps identifiers out while tolerating
# the passthrough and computed writes real status fields also have.
CLOSED_SET_SHARE = 0.5


@dataclass
class SubjectCandidate:
    """One ``(spec_class, attribute)`` pair and the criteria it satisfies.

    The key is qualified because ``FileSpec.status`` and ``JediFileSpec.status``
    are different things; the same shape recurs for ``(namespace, code)`` error
    codes and for ``map_id``.
    """

    spec_class: str
    attribute: str
    criteria: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.spec_class}.{self.attribute}"


def collect_spec_attributes(modules: list[Module]) -> dict[str, set[str]]:
    """Return ``{spec_class: {attribute, ...}}`` from ``_attributes`` declarations."""
    out: dict[str, set[str]] = defaultdict(set)
    for mod in modules:
        for cls in (n for n in ast.walk(mod.tree) if isinstance(n, ast.ClassDef)):
            for stmt in cls.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                names = {t.id for t in stmt.targets if isinstance(t, ast.Name)}
                if not names & set(ATTR_DECLS):
                    continue
                if isinstance(stmt.value, (ast.Tuple, ast.List)):
                    out[cls.name].update(
                        e.value
                        for e in stmt.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
    return dict(out)


def collect_declared_vocabularies(modules: list[Module]) -> dict[str, list[str]]:
    """Return ``{ClassName.method: [literal, ...]}`` for declared value sets.

    These are the strongest oracles in the codebase: a classmethod returning a
    literal list of statuses states the vocabulary outright, so extraction can
    be checked against it without touching production data.
    """
    out: dict[str, list[str]] = {}
    for mod in modules:
        for cls in (n for n in ast.walk(mod.tree) if isinstance(n, ast.ClassDef)):
            for fn in cls.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for stmt in ast.walk(fn):
                    if not isinstance(stmt, ast.Return) or stmt.value is None:
                        continue
                    literals = _literal_strings(stmt.value)
                    if len(literals) >= 2:
                        out.setdefault(f"{cls.name}.{fn.name}", literals)
    return out


def _literal_strings(node: ast.AST) -> list[str]:
    """Collect string literals from a list/tuple/dict literal expression."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    if isinstance(node, ast.Dict):
        out: list[str] = []
        for k, v in zip(node.keys, node.values, strict=False):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                out.append(k.value)
            out.extend(_literal_strings(v))
        return out
    return []


# ``WHERE t.status IN ('ready','running')`` -- a field constrained against
# *literals* is a state gate: other components' progress depends on its value,
# which is what makes "why does it have this value" a diagnostic question.
#
# Only quoted literals and ``IN (...)`` lists qualify.  The two rejected forms
# are both keys, not state:
#   ``WHERE PandaID=:PandaID``            -- bind variable, a lookup
#   ``WHERE t.jediTaskID=f.jediTaskID``   -- bare identifier, a join
# Accepting either floods the ranking with identifiers (``jediTaskID`` alone
# contributed 528 writes before this was tightened).
# The ``IN`` list is captured rather than just matched: ``status IN
# ('ready','running')`` is a gate, ``jediTaskID IN (:t1,:t2)`` is a batch
# lookup.  Requiring a quoted literal inside the parentheses separates them.
_WHERE_PRED_RE = re.compile(
    r"(?:WHERE|AND|OR)\s+(?:[a-zA-Z_][\w]*\.)?([a-zA-Z_][\w]*)\s*"
    r"(?:(?:=|<>|!=|<|>)\s*(?P<lit>'[^']*')|\bIN\b\s*\((?P<inlist>[^)]*)\))",
    re.I,
)


def collect_where_fields(modules: list[Module]) -> Counter:
    """Count fields **gated** by a SQL selection predicate against literals."""
    counts: Counter = Counter()
    for mod in modules:
        for const in _string_constants(mod.tree):
            if "WHERE" not in const.value.upper():
                continue
            for match in _WHERE_PRED_RE.finditer(const.value):
                inlist = match.group("inlist")
                if inlist is not None:
                    if "'" not in inlist:
                        continue  # IN over bind variables -> lookup
                    if "SELECT" in inlist.upper():
                        # ``IN (SELECT ... WHERE x='y')`` -- the quote belongs to
                        # the subquery, not to a literal list on this field.
                        continue
                counts[match.group(1)] += 1
    return counts


def collect_logged_fields(modules: list[Module]) -> Counter:
    """Count identifiers the code itself mentions in log f-strings.

    A field the code bothers to log is one its authors considered noteworthy --
    a weak but free signal, and the same templates later become the preferred
    observation channel (they carry the value at decision time).
    """
    counts: Counter = Counter()
    for mod in modules:
        for node in ast.walk(mod.tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            for value in node.values:
                if not isinstance(value, ast.FormattedValue):
                    continue
                target = value.value
                if isinstance(target, ast.Attribute):
                    counts[target.attr] += 1
                elif isinstance(target, ast.Name):
                    counts[target.id] += 1
    return counts


def promote_subjects(
    modules: list[Module],
    spec_attrs: dict[str, set[str]],
    vocabularies: dict[str, list[str]],
    where_fields: Counter,
    logged_fields: Counter,
    literal_writes: dict[str, set[str]],
    guarded_writes: set[str],
) -> list[SubjectCandidate]:
    """Evaluate the five promotion criteria over every declared attribute.

    Hundreds of attributes exist and most are ids, timestamps and counters, so
    the census ranks with evidence attached rather than emitting a flat list --
    a human reviews the ranking instead of authoring it.

    Measured on ``panda-server-source 0.8.1``, criteria 4 and 5 fire on 69% and
    79% of attributes respectively, so they carry almost no signal.  They are
    kept as corroborating evidence but **promotion requires at least one strong
    criterion** (a state gate, a declared vocabulary, or a closed literal set).
    """
    candidates: list[SubjectCandidate] = []
    vocab_classes = {name.split(".")[0] for name in vocabularies}
    for spec_class, attrs in spec_attrs.items():
        for attr in sorted(attrs):
            cand = SubjectCandidate(spec_class, attr)
            if where_fields.get(attr):
                cand.criteria.add(STRONG_WHERE)
            if spec_class in vocab_classes and _STATUS_RE.search(attr):
                cand.criteria.add(STRONG_VOCAB)
            values, n_literal, n_total = literal_writes.get(attr, (set(), 0, 0))
            if 2 <= len(values) <= 40 and n_total and n_literal / n_total >= CLOSED_SET_SHARE:
                cand.criteria.add(STRONG_CLOSED_SET)
            if logged_fields.get(attr):
                cand.criteria.add("4:mentioned-in-logs")
            if attr in guarded_writes:
                cand.criteria.add("5:written-under-condition")
            if cand.criteria & STRONG_CRITERIA:
                candidates.append(cand)
    candidates.sort(key=lambda c: (-len(c.criteria & STRONG_CRITERIA), -len(c.criteria), c.key))
    return candidates


# --------------------------------------------------------------------------- #
# writer census
# --------------------------------------------------------------------------- #

# Meta-patterns that explain a write.  The names double as the coverage
# columns, so a write matching none of them is the discovery frontier.
W_ATTR_LITERAL = "attr = literal"
W_ATTR_CONST = "attr = module const"
W_ATTR_NAME = "attr = local name"
W_ATTR_OTHER = "attr = other expr"
W_BIND_LITERAL = "bind = literal"
W_BIND_CONST = "bind = module const"
W_BIND_NAME = "bind = local name"
W_BIND_OTHER = "bind = other expr"
W_SQL_LITERAL = "SQL SET = literal"
W_SQL_BIND = "SQL SET = :bind"
W_SQL_COLUMN = "SQL SET = column-copy"

# Resolving the outcome needs no dataflow at all for these.
TIER1_PATTERNS = {
    W_ATTR_LITERAL,
    W_ATTR_CONST,
    W_BIND_LITERAL,
    W_BIND_CONST,
    W_SQL_LITERAL,
}
# ... intra-function reaching definitions for these ...
TIER1_LOCAL_PATTERNS = {W_ATTR_NAME, W_BIND_NAME}
# ... and these carry a value from elsewhere, which the model records as a
# passthrough rather than resolving.
PASSTHROUGH_PATTERNS = {W_SQL_COLUMN}


@dataclass
class Writer:
    """One site that writes a subject, plus what explains it."""

    subject: str
    pattern: str
    module: str
    line: int
    enclosing: Optional[str]
    guarded: bool
    snippet: str


def _classify_rhs(value: ast.AST, literal_kind: str, const_kind: str, name_kind: str, other_kind: str) -> str:
    if isinstance(value, ast.Constant):
        return literal_kind
    if isinstance(value, ast.Attribute):
        # ``EventServiceUtils.ST_discarded`` and ``spec.status`` both land here;
        # both resolve without dataflow (constant lookup / passthrough).
        return const_kind
    if isinstance(value, ast.Name):
        return name_kind
    return other_kind


_SQL_SET_RE = re.compile(
    r"\bSET\s+([a-zA-Z_][\w]*)\s*=\s*(:[a-zA-Z_][\w]*|'[^']*'|[a-zA-Z_][\w]*)",
    re.I,
)


def collect_writers(modules: list[Module], subjects: set[str]) -> list[Writer]:
    """Find and classify every write to *subjects* across the corpus.

    Three write forms are covered because PanDA uses all three: attribute
    assignment, SQL bind-variable assignment (``varMap[":status"] = ...``) and
    SQL text (``SET status=...``).  The SQL-text form is the one a naive
    attribute-only sweep misses, and it includes the column-copy variant
    (``SET status=oldStatus``) that carries no literal at all.
    """
    writers: list[Writer] = []
    for mod in modules:
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr in subjects:
                        writers.append(
                            Writer(
                                subject=target.attr,
                                pattern=_classify_rhs(
                                    node.value, W_ATTR_LITERAL, W_ATTR_CONST, W_ATTR_NAME, W_ATTR_OTHER
                                ),
                                module=mod.rel,
                                line=node.lineno,
                                enclosing=_enclosing_function(node),
                                guarded=_under_condition(node),
                                snippet=_snippet(node),
                            )
                        )
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                        and target.slice.value.startswith(":")
                        and target.slice.value[1:] in subjects
                    ):
                        writers.append(
                            Writer(
                                subject=target.slice.value[1:],
                                pattern=_classify_rhs(
                                    node.value, W_BIND_LITERAL, W_BIND_CONST, W_BIND_NAME, W_BIND_OTHER
                                ),
                                module=mod.rel,
                                line=node.lineno,
                                enclosing=_enclosing_function(node),
                                guarded=_under_condition(node),
                                snippet=_snippet(node),
                            )
                        )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "SET " not in node.value.upper():
                    continue
                for column, rhs in _SQL_SET_RE.findall(node.value):
                    if column not in subjects:
                        continue
                    if rhs.startswith(":"):
                        pattern = W_SQL_BIND
                    elif rhs.startswith("'"):
                        pattern = W_SQL_LITERAL
                    else:
                        pattern = W_SQL_COLUMN
                    writers.append(
                        Writer(
                            subject=column,
                            pattern=pattern,
                            module=mod.rel,
                            line=node.lineno,
                            enclosing=_enclosing_function(node),
                            guarded=_under_condition(node),
                            snippet=f"SET {column}={rhs}",
                        )
                    )
    return writers


def _snippet(node: ast.AST, width: int = 80) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on some synthesised nodes
        return ""
    return text if len(text) <= width else text[: width - 1] + "…"


def collect_literal_writes(modules: list[Module]) -> dict[str, tuple[set[str], int, int]]:
    """Return ``{attribute: (literal values, literal writes, total writes)}``.

    Feeds promotion criterion 3.  The write totals matter: an attribute is a
    state variable when a small literal set *dominates* its writes, not merely
    when a few literals exist somewhere.  Counting existence alone promoted
    ``jediTaskID`` (528 writes, a handful of them literal) above every real
    status field.
    """
    values: dict[str, set[str]] = defaultdict(set)
    literal_writes: Counter = Counter()
    total_writes: Counter = Counter()
    for mod in modules:
        for node in ast.walk(mod.tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                total_writes[target.attr] += 1
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    values[target.attr].add(node.value.value)
                    literal_writes[target.attr] += 1
    return {
        attr: (values.get(attr, set()), literal_writes[attr], total)
        for attr, total in total_writes.items()
    }


def collect_guarded_attributes(modules: list[Module]) -> set[str]:
    """Attributes that are written inside at least one conditional branch."""
    out: set[str] = set()
    for mod in modules:
        for node in ast.walk(mod.tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and _under_condition(node):
                    out.add(target.attr)
    return out


# --------------------------------------------------------------------------- #
# meta-pattern inventory
# --------------------------------------------------------------------------- #

# ``criteria=-diskIO`` / ``reason=low_efficiency`` -- a tag embedded in a log
# string names the branch, which is what makes brokerage filters identifiable
# at all (the stages themselves have no name in the code).
_TAG_RE = re.compile(r"\b([a-z_]+)=(-?)([a-zA-Z_][\w]*)")

MP_TAG = "tagged log literal"
MP_VALUE_ENUM = "value enumeration (name <-> value)"
MP_VALUE_SET = "declared value set"
MP_FILTER_CHAIN = "filter-chain reassign idiom"
MP_REQ_ENTRY = "request entry point (req arg)"
MP_CONFIG_KEY = "config key literal"


def _module_constant_name(node: ast.Assign) -> Optional[str]:
    """Return the constant's name if *node* is a module-level CONSTANT = ... ."""
    if not isinstance(getattr(node, "parent", None), ast.Module):
        return None
    for target in node.targets:
        if isinstance(target, ast.Name) and (target.id.isupper() or target.id.startswith(("EC_", "ST_"))):
            return target.id
    return None


def collect_metapatterns(modules: list[Module]) -> dict[str, Counter]:
    """Count each meta-pattern per file.

    Reported per file rather than as a single average: the filter-chain idiom
    varies wildly between sibling files, and a mean would hide that behind the
    patterns that depend only on Python syntax and so match everywhere.
    """
    found: dict[str, Counter] = {
        MP_TAG: Counter(),
        MP_VALUE_ENUM: Counter(),
        MP_VALUE_SET: Counter(),
        MP_FILTER_CHAIN: Counter(),
        MP_REQ_ENTRY: Counter(),
        MP_CONFIG_KEY: Counter(),
    }
    for mod in modules:
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.JoinedStr):
                for value in node.values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        for key, dash, _val in _TAG_RE.findall(value.value):
                            if dash or key in ("criteria", "reason", "action"):
                                found[MP_TAG][mod.rel] += 1
            elif isinstance(node, ast.Assign):
                # Two different things hide under "module constant", and they
                # feed different consumers:
                #   EC_Kill = 100 / ST_ready = 0   -> a name<->value pair, which
                #       is what the (namespace, value) index is built from
                #   FINAL_TASK_STATUSES = [...]    -> a declared vocabulary,
                #       which is a gate oracle, not an index entry
                # Counting them together made ErrorCode.py and DataCarousel.py
                # look like the same kind of file.
                if _module_constant_name(node):
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, (int, str)
                    ):
                        found[MP_VALUE_ENUM][mod.rel] += 1
                    elif _literal_strings(node.value):
                        found[MP_VALUE_SET][mod.rel] += 1
                # ``newScanSiteList = []`` opens a filter stage.
                if (
                    isinstance(node.value, ast.List)
                    and not node.value.elts
                    and any(isinstance(t, ast.Name) and t.id.startswith("new") for t in node.targets)
                ):
                    found[MP_FILTER_CHAIN][mod.rel] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.args.args and node.args.args[0].arg == "req":
                    found[MP_REQ_ENTRY][mod.rel] += 1
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "getConfigValue":
                    if any(isinstance(a, (ast.Constant, ast.JoinedStr)) for a in node.args):
                        found[MP_CONFIG_KEY][mod.rel] += 1
    return found


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _pct(part: int, whole: int) -> str:
    return f"{part * 100 // whole:>3}%" if whole else "  -%"


def report_subjects(candidates: list[SubjectCandidate], spec_attrs: dict[str, set[str]]) -> None:
    total_attrs = sum(len(v) for v in spec_attrs.values())
    click.echo(f"\n=== subjects ===  spec classes: {len(spec_attrs)}   declared attributes: {total_attrs}")
    click.echo(f"promoted candidates: {len(candidates)}\n")
    click.echo(f"{'subject':<40}{'#':>3}  criteria")
    for cand in candidates:
        click.echo(f"{cand.key:<40}{len(cand.criteria):>3}  {', '.join(sorted(cand.criteria))}")


def report_writers(writers: list[Writer]) -> None:
    by_subject: dict[str, Counter] = defaultdict(Counter)
    for w in writers:
        by_subject[w.subject][w.pattern] += 1
    patterns = [
        W_ATTR_LITERAL, W_ATTR_CONST, W_ATTR_NAME, W_ATTR_OTHER,
        W_BIND_LITERAL, W_BIND_CONST, W_BIND_NAME, W_BIND_OTHER,
        W_SQL_LITERAL, W_SQL_BIND, W_SQL_COLUMN,
    ]
    click.echo(f"\n=== writers ===  total sites: {len(writers)}\n")
    header = f"{'subject':<18}" + "".join(f"{p.replace(' = ', '='):>22}" for p in patterns) + f"{'tot':>6}"
    click.echo(header)
    for subject, counts in sorted(by_subject.items(), key=lambda kv: -sum(kv[1].values())):
        row = [counts.get(p, 0) for p in patterns]
        click.echo(f"{subject:<18}" + "".join(f"{v:>22}" for v in row) + f"{sum(row):>6}")

    # Tier split.  The SQL ``:bind`` half of a varMap write is not a separate
    # write, so it is excluded from the denominator rather than double-counted.
    distinct = [w for w in writers if w.pattern != W_SQL_BIND]
    tier1 = sum(1 for w in distinct if w.pattern in TIER1_PATTERNS)
    tier1_local = sum(1 for w in distinct if w.pattern in TIER1_LOCAL_PATTERNS)
    passthrough = sum(1 for w in distinct if w.pattern in PASSTHROUGH_PATTERNS)
    residual = len(distinct) - tier1 - tier1_local - passthrough
    total = len(distinct) or 1
    click.echo(f"\ndistinct writes (excluding the SQL ':bind' half): {len(distinct)}")
    click.echo(f"  no dataflow needed                {tier1:>5}  {_pct(tier1, total)}")
    click.echo(f"  intra-function reaching defs      {tier1_local:>5}  {_pct(tier1_local, total)}")
    click.echo(f"  passthrough (value from elsewhere){passthrough:>5}  {_pct(passthrough, total)}")
    click.echo(f"  residual -> Tier 2                {residual:>5}  {_pct(residual, total)}")
    click.echo(f"  => Tier 1 total                   {tier1 + tier1_local + passthrough:>5}  "
               f"{_pct(tier1 + tier1_local + passthrough, total)}")


def report_metapatterns(found: dict[str, Counter], top: int) -> None:
    click.echo("\n=== meta-patterns ===")
    for name, counts in found.items():
        total = sum(counts.values())
        click.echo(f"\n{name}: {total} occurrence(s) in {len(counts)} file(s)")
        for rel, n in counts.most_common(top):
            click.echo(f"    {n:>5}  {rel}")


def report_unexplained(writers: list[Writer], top: int) -> None:
    """Writers no meta-pattern resolves -- where the next recognizer is needed.

    Reported explicitly because silent truncation reads as "covered everything"
    when it is not: an unexplained write is a known unknown, whereas a write
    neither the recognizers nor this census sees is invisible and only the
    graph invariants and the transition-history check can surface it.
    """
    residual = [
        w
        for w in writers
        if w.pattern
        not in (TIER1_PATTERNS | TIER1_LOCAL_PATTERNS | PASSTHROUGH_PATTERNS | {W_SQL_BIND})
    ]
    click.echo(f"\n=== unresolved outcomes (Tier 2 frontier) === {len(residual)} site(s)")
    for w in residual[:top]:
        where = f"{w.module}:{w.line}"
        click.echo(f"  {w.subject:<16} {w.pattern:<20} {where:<52} {w.snippet}")
    if len(residual) > top:
        click.echo(f"  … {len(residual) - top} more (use --json for the full list)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command("census-code-map")
@click.option("--source-root", default=None, metavar="PATH",
              help="Analyse packages under PATH instead of the installed distribution.")
@click.option("--section", type=click.Choice(["all", "subjects", "writers", "metapatterns"]),
              default="all", show_default=True, help="Limit the report to one section.")
@click.option("--top", default=12, show_default=True, help="Rows shown per per-file listing.")
@click.option("--json", "json_out", default=None, metavar="PATH",
              help="Also write the full result as JSON (for diffing across releases).")
def main(source_root: Optional[str], section: str, top: int, json_out: Optional[str]) -> None:
    """Census the PanDA source for Code Map subjects, writers and meta-patterns."""
    roots = resolve_pkg_roots(source_root)
    if not roots:
        raise click.ClickException(
            "Neither pandaserver nor pandajedi found. Install the 'panda' extra "
            "or pass --source-root."
        )
    version = resolve_version(source_root)
    click.echo(f"source: {', '.join(f'{k} -> {v}' for k, v in roots.items())}")
    click.echo(f"derived_from: {version}")

    modules = load_modules(roots)
    click.echo(f"parsed {len(modules)} module(s)")

    spec_attrs = collect_spec_attributes(modules)
    vocabularies = collect_declared_vocabularies(modules)
    where_fields = collect_where_fields(modules)
    logged_fields = collect_logged_fields(modules)
    literal_writes = collect_literal_writes(modules)
    guarded = collect_guarded_attributes(modules)

    candidates = promote_subjects(
        modules, spec_attrs, vocabularies, where_fields, logged_fields, literal_writes, guarded
    )
    subjects = {c.attribute for c in candidates}
    writers = collect_writers(modules, subjects)
    metapatterns = collect_metapatterns(modules)

    if section in ("all", "subjects"):
        report_subjects(candidates, spec_attrs)
        click.echo(f"\ndeclared vocabularies: {len(vocabularies)}")
        for name in sorted(vocabularies)[:top]:
            click.echo(f"    {name}: {len(vocabularies[name])} value(s)")
    if section in ("all", "writers"):
        report_writers(writers)
        report_unexplained(writers, top)
    if section in ("all", "metapatterns"):
        report_metapatterns(metapatterns, top)

    if json_out:
        payload: dict[str, Any] = {
            "derived_from": version,
            "modules": len(modules),
            "spec_attributes": {k: sorted(v) for k, v in spec_attrs.items()},
            "declared_vocabularies": vocabularies,
            "subjects": [
                {"spec_class": c.spec_class, "attribute": c.attribute, "criteria": sorted(c.criteria)}
                for c in candidates
            ],
            "writers": [
                {
                    "subject": w.subject, "pattern": w.pattern, "module": w.module,
                    "line": w.line, "enclosing": w.enclosing, "guarded": w.guarded,
                    "snippet": w.snippet,
                }
                for w in writers
            ],
            "metapatterns": {name: dict(counts) for name, counts in metapatterns.items()},
        }
        Path(json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
        click.echo(f"\nwrote {json_out}")


if __name__ == "__main__":
    main()
