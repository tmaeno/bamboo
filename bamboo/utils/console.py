"""Environment-aware Rich ``Console`` construction and ASCII fallbacks.

Why this module exists
----------------------
Bamboo's console output is designed for an interactive terminal — colour, rounded
:class:`~rich.panel.Panel` borders, spinners, and glyphs like ``✓ ✗ → …``. When the
same code runs as a **batch job** its stdout is a pipe that a scheduler captures into
a job log, and that log is typically read back through a viewer that decodes it as
cp1252 rather than UTF-8. Every non-ASCII byte then explodes into mojibake
(``╭─`` → ``â•­â”€``), which makes the log unreadable.

So on a non-terminal we switch to **plain mode**:

* :func:`to_ascii` transliterates decorative non-ASCII to ASCII. It is applied at the
  output choke points — :class:`bamboo.utils.logging._AsciiFormatter` for every log
  record, :func:`echo` for ``click`` output, and :meth:`Console.print` on the console
  returned by :func:`make_console` — so individual call sites keep using ``✓``/``→``
  and get an ASCII rendering for free.
* :func:`box_for` swaps Rich's Unicode box drawing for :data:`rich.box.ASCII`. Rich's
  own ``safe_box`` does **not** help here: it only substitutes on legacy Windows.

**Invariant:** :func:`to_ascii` must be applied *before* text is handed to Rich, never
to Rich's rendered output. Rich measures the transliterated string, so replacements
that change length (``…`` → ``...``) cannot shift a panel's right border.

Plain mode is auto-detected from the console not being a terminal, so a batch job
needs no extra flag. ``BAMBOO_PLAIN_OUTPUT=1`` / ``=0`` forces it either way.
"""

from __future__ import annotations

import os
import sys
import unicodedata
from typing import Any

from rich import box as _box
from rich.console import Console

PLAIN_ENV_VAR = "BAMBOO_PLAIN_OUTPUT"
WIDTH_ENV_VAR = "BAMBOO_CONSOLE_WIDTH"

# Rich falls back to 80 columns on a non-terminal, and `apptainer run --cleanenv`
# strips COLUMNS, so a batch job would cram every panel and table into 80 columns.
DEFAULT_PLAIN_WIDTH = 120

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def plain_output(console: Console | None = None) -> bool:
    """Whether output should be plain ASCII (batch/log) rather than rich (terminal).

    ``BAMBOO_PLAIN_OUTPUT`` wins when set to a recognised truthy/falsey value.
    Otherwise the answer is derived from *console* (``not console.is_terminal``) when
    one is given, else from ``sys.stdout``.

    Deriving from the console — rather than a process-wide global — keeps an
    explicitly constructed ``Console(force_terminal=True)`` in rich mode even when the
    process' own stdout is redirected, which is what tests and captured-output helpers
    rely on.
    """
    raw = os.environ.get(PLAIN_ENV_VAR)
    if raw is not None:
        value = raw.strip().lower()
        if value in _TRUTHY:
            return True
        if value in _FALSEY:
            return False
    if console is not None:
        return not console.is_terminal
    return not sys.stdout.isatty()


def plain_width() -> int:
    """Console width to use in plain mode.

    ``BAMBOO_CONSOLE_WIDTH`` first, then ``COLUMNS``, else
    :data:`DEFAULT_PLAIN_WIDTH`. Rich consults ``COLUMNS`` itself, but only while its
    own width is unset — once we pin a width it would be ignored, so honour it here.
    """
    for name in (WIDTH_ENV_VAR, "COLUMNS"):
        raw = os.environ.get(name)
        if raw and raw.strip().isdigit():
            width = int(raw.strip())
            if width > 0:
                return width
    return DEFAULT_PLAIN_WIDTH


# ---------------------------------------------------------------------------
# ASCII transliteration
# ---------------------------------------------------------------------------

# Explicit replacements for the glyphs bamboo actually emits, chosen to stay
# readable ("OK" reads better than "*"). Anything not listed falls through to
# _fallback_char() below.
_ASCII_MAP: dict[str, str] = {
    # dashes and punctuation
    "—": "-", "–": "-", "‒": "-", "―": "-", "−": "-",
    "…": "...", "•": "*", "·": "*", "§": "S",
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
    "«": "<<", "»": ">>", "‹": "<", "›": ">",
    "°": " deg", "™": "(TM)", "®": "(R)", "©": "(C)",
    # arrows
    "→": "->", "⇒": "=>", "↦": "->", "↳": "->", "⟶": "->",
    "←": "<-", "⇐": "<=", "↔": "<->", "⇔": "<=>",
    "↑": "^", "↓": "v", "↻": "(retry)", "⟲": "(retry)",
    # maths / comparison
    "≥": ">=", "≤": "<=", "≠": "!=", "≈": "~", "×": "x", "÷": "/",
    "±": "+/-", "∞": "inf",
    # status glyphs
    "✓": "OK", "✔": "OK", "✅": "OK",
    "✗": "FAIL", "✘": "FAIL", "❌": "FAIL",
    "⚠": "!", "⚡": "!", "◆": "*", "★": "*", "☆": "*", "○": "o",
    # invisible characters that only confuse a log reader
    "\ufe0e": "", "\ufe0f": "",  # variation selectors (text / emoji presentation)
    "\u200b": "", "\u200d": "",  # zero-width space / joiner
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",  # non-breaking / thin spaces
}

# Box drawing (U+2500-U+257F): map the pure line segments to - and |, and every
# corner/junction/joint to +, which is what rich.box.ASCII would have drawn.
_BOX_HORIZONTAL = "─━┄┅┈┉╌╍═╴╶╸╺"
_BOX_VERTICAL = "│┃┆┇┊┋╎╏║╵╷╹╻"
for _ch in _BOX_HORIZONTAL:
    _ASCII_MAP[_ch] = "-"
for _ch in _BOX_VERTICAL:
    _ASCII_MAP[_ch] = "|"
for _cp in range(0x2500, 0x2580):
    _ASCII_MAP.setdefault(chr(_cp), "+")
for _cp in range(0x2580, 0x25A0):  # block elements ▀▄█░▒▓
    _ASCII_MAP.setdefault(chr(_cp), "#")
del _ch, _cp

_TRANSLATION = str.maketrans(_ASCII_MAP)


def _fallback_char(char: str) -> str:
    """ASCII stand-in for a character missing from :data:`_ASCII_MAP`.

    Only *symbols* (which is where the remaining emoji and dingbats live) and
    invisible formatting characters are touched. Letters and marks are left alone —
    an accented name or a CJK log excerpt is genuine content, not decoration, and
    mangling it would lose information.
    """
    if char.isascii():
        return char
    category = unicodedata.category(char)
    if category.startswith("S"):
        return "*"
    if category in ("Cf", "Mn"):
        return ""
    return char


def to_ascii(text: str) -> str:
    """Transliterate decorative non-ASCII in *text* to ASCII.

    Unconditional — callers decide when it applies (see :func:`plain_output`). Cheap
    for the common case: ASCII input is returned unchanged without allocating.
    """
    if text.isascii():
        return text
    translated = text.translate(_TRANSLATION)
    if translated.isascii():
        return translated
    return "".join(_fallback_char(char) for char in translated)


def maybe_ascii(text: str, console: Console | None = None) -> str:
    """:func:`to_ascii` when *console* is in plain mode, else *text* unchanged."""
    return to_ascii(text) if plain_output(console) else text


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


class BambooConsole(Console):
    """A :class:`~rich.console.Console` that renders ASCII when not on a terminal.

    Only the top-level ``str`` arguments of :meth:`print` are transliterated, which
    covers the ordinary one-line output. Text destined for a :class:`~rich.panel.Panel`
    or :class:`~rich.table.Table` must be passed through :func:`to_ascii` by the caller
    *before* construction (see the module invariant) — :func:`panel_for` and
    :func:`table_for` do that.
    """

    @property
    def plain_mode(self) -> bool:
        """Whether this console renders in plain ASCII mode."""
        return plain_output(self)

    def print(self, *objects: Any, **kwargs: Any) -> None:  # noqa: A003 - Rich's API
        if self.plain_mode:
            objects = tuple(
                to_ascii(obj) if isinstance(obj, str) else obj for obj in objects
            )
        super().print(*objects, **kwargs)


def make_console(**kwargs: Any) -> BambooConsole:
    """Build the console bamboo should print through.

    In plain mode the width is pinned to :func:`plain_width` — Rich's non-terminal
    default of 80 columns would otherwise wrap panel and table *content*, which
    ``soft_wrap`` cannot help with (it only applies to text printed as a line). An
    explicit ``width=`` argument always wins.

    Colour needs no handling — Rich already drops ``color_system`` on a non-terminal.
    Emoji *shortcodes* are disabled in plain mode so ``:tada:`` stays literal instead
    of expanding into a glyph that :func:`to_ascii` would then flatten to ``*``.
    """
    console = BambooConsole(**kwargs)
    if console.plain_mode:
        console._emoji = False  # noqa: SLF001 - no public setter in Rich
        if kwargs.get("width") is None:
            console.width = plain_width()
    return console


def box_for(console: Console | None = None):
    """Box style for panels/tables: ASCII in plain mode, Rich's default otherwise."""
    return _box.ASCII if plain_output(console) else _box.ROUNDED


# ---------------------------------------------------------------------------
# Renderable helpers (transliterate before Rich measures)
# ---------------------------------------------------------------------------


def panel_for(console: Console | None, body: str, *, fit: bool = False, **kwargs: Any):
    """A :class:`~rich.panel.Panel` with an ASCII box and body in plain mode.

    *fit* selects :meth:`rich.panel.Panel.fit` (shrink-to-content) instead of the
    full-width constructor.
    """
    from rich.panel import Panel  # noqa: PLC0415 - keep import cost off the CLI path

    if plain_output(console):
        body = to_ascii(body)
        title = kwargs.get("title")
        if isinstance(title, str):
            kwargs["title"] = to_ascii(title)
    kwargs.setdefault("box", box_for(console))
    return Panel.fit(body, **kwargs) if fit else Panel(body, **kwargs)


def table_for(console: Console | None, **kwargs: Any):
    """A :class:`~rich.table.Table` whose borders and title are ASCII in plain mode.

    Cell values are *not* transliterated automatically (``add_row`` is called later);
    pass arbitrary text through :func:`maybe_ascii` if it may contain glyphs.
    """
    from rich.table import Table  # noqa: PLC0415 - keep import cost off the CLI path

    if plain_output(console):
        title = kwargs.get("title")
        if isinstance(title, str):
            kwargs["title"] = to_ascii(title)
    kwargs.setdefault("box", box_for(console))
    return Table(**kwargs)


def echo(message: Any = None, **kwargs: Any) -> None:
    """``click.echo`` with plain-mode ASCII transliteration.

    Mode is decided from ``sys.stdout``/``BAMBOO_PLAIN_OUTPUT`` — ``click.echo`` has no
    console object to consult. ``err=True`` output is transliterated too: a scheduler
    usually merges the two streams into one job log.
    """
    import click  # noqa: PLC0415 - click is always installed, but keep this leaf-cheap

    if isinstance(message, str) and plain_output():
        message = to_ascii(message)
    click.echo(message, **kwargs)
