"""What a write of a computed attribute name on a spec can set -- ``codemap.panda.specwrites``."""

from __future__ import annotations

from pathlib import Path

from bamboo.codemap import calls as calls_mod
from bamboo.codemap.panda import specwrites

PACKAGE = "pkg"

_SPEC = """
class Spec:
    _attributes = ("a", "b", "errCode", "errDiag")

    def __init__(self):
        for attr in self._attributes:
            object.__setattr__(self, attr, None)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __setstate__(self, state):
        for i in range(len(self._attributes)):
            object.__setattr__(self, self._attributes[i], state[i])

    def pack(self, values):
        for i in range(len(self._attributes)):
            attr = self._attributes[i]
            object.__setattr__(self, attr, values[i])

    def from_other(self, other):
        for attr in self._attributes + ("extra",):
            if "err" in attr:
                continue
            self.__setattr__(attr, getattr(other, attr))

    def nulls(self):
        for attr in self._attributes:
            val = getattr(self, attr)
            if val == "NULL":
                object.__setattr__(self, attr, None)


class Plain:
    _attributes = ("x", "y")
    _renamed = {"p": "x", "q": "y"}

    def take(self, other):
        for attr in ("p", "q"):
            if attr in self._renamed:
                renamed = self._renamed[attr]
                setattr(self, renamed, getattr(other, attr))


class NotASpec:
    pass
"""

_USES = """
import copy

from pkg.spec import NotASpec, Spec


def errors(job: Spec):
    for attr in job._attributes:
        if attr.endswith("Code") or attr.endswith("Diag"):
            setattr(job, attr, None)


def by_pattern(job: Spec):
    clone = copy.copy(job)
    for attr in clone._attributes:
        for patt in ["Diag"]:
            if attr.endswith(patt):
                setattr(clone, attr, None)
                break


def from_data(spec: Spec, data):
    for key in data:
        setattr(spec, key, data[key])


def spelled(job: Spec):
    setattr(job, "a", 1)


def unknown(obj, name):
    setattr(obj, name, 1)


def other(thing: NotASpec, name):
    setattr(thing, name, 1)
"""


def _writes(tmp_path: Path) -> dict[tuple[str, int], specwrites.NameWrite]:
    root = tmp_path / PACKAGE
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "spec.py").write_text(_SPEC)
    (root / "uses.py").write_text(_USES)
    types = calls_mod.Types({PACKAGE: root}, {})
    return {
        (write.function, write.line): write for write in specwrites.name_writes(types)
    }


def _one(writes, function):
    found = [write for (name, _line), write in writes.items() if name == function]
    assert len(found) == 1, found
    return found[0]


def test_a_specs_own_methods_take_the_role_python_or_the_declaration_gives_them(
    tmp_path,
):
    writes = _writes(tmp_path)
    every = frozenset({"a", "b", "errCode", "errDiag"})
    assert _one(writes, "Spec.__init__")[5:8] == (specwrites.CONSTRUCT, every, False)
    assert _one(writes, "Spec.__setstate__")[5:8] == (specwrites.RESTORE, every, False)
    # ``pack`` is an assignment of every column, of the row's own value, and
    # goes around ``__setattr__``: a flush of what changed never writes it.
    assert _one(writes, "Spec.pack")[5:8] == (specwrites.READS_ROW, every, False)
    # ``__setattr__`` writes whatever its caller assigned: the caller's write.
    assert _one(writes, "Spec.__setattr__")[5:7] == (specwrites.MECHANISM, None)


def test_names_running_over_a_declaration_are_those_a_test_on_the_name_lets_through(
    tmp_path,
):
    writes = _writes(tmp_path)
    errors = _one(writes, "errors")
    assert (errors.classes, errors.role, errors.names, errors.tracked) == (
        ("Spec",),
        specwrites.NAMES,
        frozenset({"errCode", "errDiag"}),
        True,
    )
    # A loop over literals inside is run through, one value at a time; and a
    # copy is an object of the class it copies.
    by_pattern = _one(writes, "by_pattern")
    assert (by_pattern.classes, by_pattern.names) == (("Spec",), frozenset({"errDiag"}))


def test_a_test_on_anything_but_the_name_keeps_every_name(tmp_path):
    nulls = _one(_writes(tmp_path), "Spec.nulls")
    assert (nulls.role, nulls.names, nulls.tracked) == (
        specwrites.NAMES,
        frozenset({"a", "b", "errCode", "errDiag"}),
        False,
    )


def test_a_value_read_off_a_parameter_by_the_same_name_is_a_copy(tmp_path):
    writes = _writes(tmp_path)
    from_other = _one(writes, "Spec.from_other")
    assert (from_other.role, from_other.names, from_other.tracked) == (
        specwrites.COPY,
        frozenset({"a", "b", "extra"}),
        True,
    )
    # Looked up in a mapping the class declares: its values.  No
    # ``__setattr__`` is defined, so nothing records the change.
    take = _one(writes, "Plain.take")
    assert (take.role, take.names, take.tracked) == (
        specwrites.COPY,
        frozenset({"x", "y"}),
        False,
    )


def test_names_from_a_run_time_value_are_every_column_the_class_declares(tmp_path):
    from_data = _one(_writes(tmp_path), "from_data")
    assert (from_data.role, from_data.names) == (
        specwrites.ANY,
        frozenset({"a", "b", "errCode", "errDiag"}),
    )
    assert "data" in from_data.reason


def test_an_unread_class_is_listed_without_a_role_and_another_class_or_a_spelled_name_is_not(
    tmp_path,
):
    writes = _writes(tmp_path)
    unknown = _one(writes, "unknown")
    assert (unknown.role, unknown.names, unknown.classes) == (
        specwrites.NO_ROLE,
        None,
        (),
    )
    functions = {name for name, _line in writes}
    assert "other" not in functions
    assert "spelled" not in functions
