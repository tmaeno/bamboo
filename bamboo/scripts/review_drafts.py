"""Interactive TUI for reviewing and approving bamboo draft JSON files."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView, Static, TextArea

_PENDING = "○"
_APPROVED = "✓"


class ReviewApp(App[None]):
    TITLE = "Bamboo Draft Reviewer"

    CSS = """
    #main {
        height: 1fr;
    }
    #list-panel {
        width: 30;
        border-right: solid $primary-darken-2;
    }
    #draft-list {
        height: 1fr;
    }
    #detail-panel {
        width: 1fr;
        padding: 1 2;
    }
    .section-label {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }
    #task-info {
        background: $surface;
        padding: 1;
        margin-bottom: 1;
    }
    #review-hint {
        color: $warning;
        margin-bottom: 1;
    }
    TextArea {
        height: 5;
        margin-bottom: 1;
    }
    #procedure-area {
        height: 8;
    }
    #approve-btn {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save_draft", "Save", priority=True),
        Binding("ctrl+d", "approve", "Approve", priority=True),
        Binding("ctrl+q", "quit_app", "Quit", priority=True),
    ]

    def __init__(self, drafts_dir: str) -> None:
        super().__init__()
        self.drafts_dir = Path(drafts_dir)
        self.draft_files: list[Path] = sorted(self.drafts_dir.glob("*.json"))
        self.current_idx: int = -1
        self.current_draft: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="list-panel"):
                yield ListView(id="draft-list")
            with ScrollableContainer(id="detail-panel"):
                yield Static("Select a draft from the list.", id="task-info")
                yield Static("", id="review-hint")
                yield Label("Background", classes="section-label")
                yield TextArea(id="background")
                yield Label("Cause", classes="section-label")
                yield TextArea(id="cause")
                yield Label("Resolution", classes="section-label")
                yield TextArea(id="resolution")
                yield Label("Procedure  (one step per line)", classes="section-label")
                yield TextArea(id="procedure-area")
                yield Button("Approve & Save  [Ctrl+D]", variant="success", id="approve-btn")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild_list()
        if self.draft_files:
            self.query_one("#draft-list", ListView).index = 0

    # ------------------------------------------------------------------
    # List helpers

    def _rebuild_list(self) -> None:
        lst = self.query_one("#draft-list", ListView)
        lst.clear()
        for path in self.draft_files:
            try:
                data = json.loads(path.read_text())
            except Exception:
                data = {}
            icon = _APPROVED if data.get("reviewed") else _PENDING
            lst.append(ListItem(Label(f"{icon} {path.stem}")))

    # ------------------------------------------------------------------
    # Load / save

    def _load_draft(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.draft_files):
            return
        path = self.draft_files[idx]
        try:
            draft = json.loads(path.read_text())
        except Exception as exc:
            self.notify(f"Cannot read {path.name}: {exc}", severity="error")
            return
        self.current_idx = idx
        self.current_draft = draft

        task_ids = ", ".join(str(t) for t in draft.get("task_ids", []))
        status = f"{_APPROVED} REVIEWED" if draft.get("reviewed") else f"{_PENDING} PENDING"
        matched = draft.get("matched_from") or "—"
        error = draft.get("errorDialog_canonical", "")
        self.query_one("#task-info", Static).update(
            f"Tasks : {task_ids}\n"
            f"Status: {status}\n"
            f"Matched: {Path(matched).name if matched != '—' else matched}\n"
            f"Error  : {error[:120]}{'…' if len(error) > 120 else ''}"
        )
        hint = draft.get("review_hint", "")
        self.query_one("#review-hint", Static).update(f"Hint: {hint}" if hint else "")

        body = draft.get("email_body", {})
        self.query_one("#background", TextArea).load_text(body.get("background", ""))
        self.query_one("#cause", TextArea).load_text(body.get("cause", ""))
        self.query_one("#resolution", TextArea).load_text(body.get("resolution", ""))
        self.query_one("#procedure-area", TextArea).load_text(
            "\n".join(body.get("procedure", []))
        )

    def _collect_and_save(self, *, approve: bool) -> None:
        if self.current_draft is None or self.current_idx < 0:
            return
        draft = dict(self.current_draft)
        body = dict(draft.get("email_body", {}))
        procedure_raw = self.query_one("#procedure-area", TextArea).text
        body["background"] = self.query_one("#background", TextArea).text.strip()
        body["cause"] = self.query_one("#cause", TextArea).text.strip()
        body["resolution"] = self.query_one("#resolution", TextArea).text.strip()
        body["procedure"] = [ln for ln in procedure_raw.splitlines() if ln.strip()]
        draft["email_body"] = body
        if approve:
            draft["reviewed"] = True
        path = self.draft_files[self.current_idx]
        path.write_text(json.dumps(draft, indent=2, default=str))
        self.current_draft = draft
        self._rebuild_list()
        self.query_one("#draft-list", ListView).index = self.current_idx

    # ------------------------------------------------------------------
    # Events

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None:
            return
        if self.current_draft is not None and idx != self.current_idx:
            self._collect_and_save(approve=False)
        self._load_draft(idx)

    @on(Button.Pressed, "#approve-btn")
    def on_approve_pressed(self) -> None:
        self.action_approve()

    # ------------------------------------------------------------------
    # Actions

    def action_save_draft(self) -> None:
        self._collect_and_save(approve=False)
        self.notify("Saved.")

    def action_approve(self) -> None:
        self._collect_and_save(approve=True)
        self.notify("Approved!", severity="information")
        next_idx = self._next_pending()
        if next_idx is not None:
            lst = self.query_one("#draft-list", ListView)
            lst.index = next_idx
            self._load_draft(next_idx)

    def action_quit_app(self) -> None:
        self._collect_and_save(approve=False)
        self.exit()

    # ------------------------------------------------------------------
    # Helpers

    def _next_pending(self) -> int | None:
        for i in range(self.current_idx + 1, len(self.draft_files)):
            if not json.loads(self.draft_files[i].read_text()).get("reviewed", False):
                return i
        for i in range(0, self.current_idx):
            if not json.loads(self.draft_files[i].read_text()).get("reviewed", False):
                return i
        return None


def main(drafts_dir: str = "drafts") -> None:
    if not Path(drafts_dir).is_dir():
        print(f"Error: '{drafts_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    ReviewApp(drafts_dir=drafts_dir).run()
