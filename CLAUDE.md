# Working in this repo

## Python interpreter — always use the project virtualenv

This project depends on packages (e.g. `textual`, langchain, neo4j, qdrant-client)
that live in its **virtualenv**, which is **not** inside the repo (`.venv`/`venv`
are gitignored). Run **all** Python tooling — `python`, `pytest`, `ruff`, `mypy`,
the `bamboo` CLI — through that venv's interpreter.

**Never fall back to the system `python` / `python3`.** A system interpreter often
has *most* deps installed, so tests/imports can *appear* to work while silently
running in the wrong environment (this previously made a single failure look
"pre-existing"). If `python` is not on `PATH`:

- locate / activate the project venv and use it, **or** invoke its interpreter by
  explicit path (see `CLAUDE.local.md` for this machine's path);
- do **not** switch to `python3` to "keep going."

A guard in `tests/conftest.py` aborts the test session with an explanatory message
if it detects a missing declared dependency (i.e. the wrong interpreter) — if you
see that, switch to the project venv rather than working around it.

## Tests

- `pytest` reads config from `pyproject.toml` (`testpaths = ["tests"]`,
  `asyncio_mode = "auto"`); run it **from the repo root** via the venv interpreter.
- `test_completion` imports the whole CLI, so it needs every declared dependency
  present — another reason to run under the project venv.
