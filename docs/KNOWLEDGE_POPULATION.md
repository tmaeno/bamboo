# Knowledge Population

Bamboo supports two population paths — choose based on your operational context:

- **Individual** (`bamboo populate`) — ingest a single task immediately.  Suitable
  for ad-hoc incident analysis and for production incidents handled one at a time.
- **Batch** (`seed-drafts → review-drafts → batch-populate`) — pre-populate the
  databases from a large CSV of known problematic tasks before the system goes live,
  or catch up after a database reset.

---

## Individual path: `bamboo populate`

```bash
bamboo populate --task-id <jediTaskID>
```

Fetches the task from PanDA, runs the full Knowledge Accumulation pipeline
(extraction → review → optional MCP enrichment), and stores the results directly
in the graph and vector databases.  Use this for:

- Production incidents: analyze a failing task and immediately add its cause/resolution
  to the knowledge base.
- Spot-checks: verify a single known task is correctly represented before a batch run.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--task-id` | (required) | PanDA jediTaskID |
| `--source` | `panda` | Extraction source (`panda`, `file`, `llm`) |
| `-v` / `--verbose` | off | Enable DEBUG logging |

---

## Batch workflow

`Cause`, `Resolution`, and `Procedure` knowledge graph nodes come exclusively
from human-reviewed email drafts.  The batch workflow reduces the authoring
burden by:

1. **`bamboo seed-drafts`** — reading a CSV of problematic PanDA tasks,
   checking what is already known, and generating LLM-assisted JSON draft files
   for tasks that need human attention.
2. **`bamboo review-drafts`** — an interactive terminal UI for reading, editing,
   and approving each draft.  Alternatively, draft JSON files can be edited
   directly and `"reviewed"` set to `true` by hand.
3. **`bamboo batch-populate`** — reading every approved draft and calling the
   standard `process_knowledge` pipeline to store graphs and vectors.

```
CSV of tasks
     │
     ▼
bamboo seed-drafts
     │
     ├─── DB-covered ──────────────────────────► skip (already in DBs)
     ├─── Approved-matched ────────────────────► drafts/ (pre-filled, quick check)
     └─── New ─────────────────────────────────► drafts/ (LLM draft, full review)
                                                         │
                                          bamboo review-drafts
                                          (or edit JSON by hand)
                                                         │
                                                         ▼
                                              bamboo batch-populate
                                                         │
                                                         ├──► graph + vector databases
                                                         └──► approved_email_drafts/
```

---

## Directory layout

```
project/
├── drafts/                    # Per-batch working area (ephemeral)
│   ├── task_12345.json
│   └── task_12367.json
│
├── approved_email_drafts/     # Permanent library of every reviewed draft
│   ├── task_9001.json         # includes errorDialog_embedding for fast search
│   └── task_9002.json
│
└── problematic_tasks.csv      # Input: exported from PanDA monitoring
```

`drafts/` is the working area for the current batch.  
`approved_email_drafts/` is permanent — it survives database resets and enables
repopulation without re-authoring emails.

---

## Draft file format

```json
{
  "reviewed": false,
  "review_hint": "Generated from scratch — full review required: check all sections carefully.",
  "matched_from": null,
  "task_ids": [12345, 12346, 12347],
  "task_data": {
    "jediTaskID": 12345,
    "status": "exhausted",
    "errorDialog": "<error>TooManyFilesInDataset user.atlas.abc123...</error>",
    "taskName": "user.atlas.abc123",
    "prodSourceLabel": "managed"
  },
  "errorDialog_canonical": "<error>TooManyFilesInDataset [dataset]...</error>",
  "email_body": {
    "background": "The task attempted to process a large dataset but hit a PanDA file-count limit.",
    "cause": "The input dataset contains more files than the configured splitRule limit allows.",
    "resolution": "Split the input dataset into smaller subsets or increase the splitRule parameter.",
    "procedure": [
      "step 1: Check the dataset file count via rucio list-files <dataset>",
      "step 2: Compare against the task's splitRule value",
      "step 3: Either split the dataset or request a higher limit from the site admin"
    ]
  },
  "created_at": "2026-04-27T10:00:00Z"
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `reviewed` | Set to `true` when the draft is approved for population |
| `review_hint` | Plain-language cue for the reviewer |
| `matched_from` | `null` for new drafts; path to source approved entry for pre-filled drafts |
| `task_ids` | Pending task IDs; removed one by one after successful population |
| `task_data` | Full PanDA task snapshot (no live fetch needed at populate time) |
| `errorDialog_canonical` | Canonicalized (instance-agnostic) error string |
| `email_body` | Structured content: background / cause / resolution / procedure |

### Pre-filled draft (`matched_from` set)

```json
{
  "review_hint": "Pre-filled from approved_email_drafts/task_9999.json (errorDialog similarity 0.91). Key differences vs. source: taskType: 'production'→'analysis', site: 'CERN'→'BNL'. Verify the cause is still correct for these differences before approving.",
  "matched_from": "approved_email_drafts/task_9999.json",
  ...
}
```

The `review_hint` shows the similarity score and lists task feature differences
(taskType, prodSourceLabel, site, coreCount, splitRule) so you can quickly judge
whether the pre-filled cause still applies.

### Approved library entry (`approved_email_drafts/`)

Same schema as above, plus:
```json
{
  "reviewed": true,
  "errorDialog_embedding": [0.12, -0.03, ...]
}
```

`errorDialog_embedding` enables fast cosine-similarity matching in subsequent
batches without re-embedding.

---

## Step 1 — Generate drafts: `bamboo seed-drafts`

```bash
bamboo seed-drafts \
  --csv problematic_tasks.csv \
  --output drafts/ \
  --approved approved_email_drafts/
```

**CSV format** (additional columns are ignored):

```
jediTaskID,status
12345,exhausted
12346,broken
12347,exhausted
```

### Classification tiers

| Tier | Condition | Output |
|------|-----------|--------|
| **Skipped** | Live PanDA status ≠ CSV status | Warning printed; no file |
| **DB-covered** | Qdrant has a Symptom vector with similarity ≥ threshold | No file |
| **Approved-matched** | `approved_email_drafts/` has a matching entry ≥ threshold | Pre-filled draft in `drafts/` |
| **New** | No coverage found | LLM-generated draft in `drafts/` |

In a mature system, most tasks fall into the first two tiers and require no
human action.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | (required) | CSV file with jediTaskID and status columns |
| `--output` | `drafts/` | Directory to write draft JSON files |
| `--approved` | `approved_email_drafts/` | Approved email library directory |
| `--similarity-threshold` | `0.85` | Cosine similarity threshold |
| `--concurrency` | `5` | Max concurrent PanDA fetch requests |
| `--skip-existing` | off | Skip drafts that already exist in output |
| `-v` / `--verbose` | off | Enable DEBUG logging |

---

## Step 2 — Review drafts: `bamboo review-drafts`

```bash
bamboo review-drafts --drafts drafts/
```

Opens an interactive terminal UI with:

- **Left panel** — list of all drafts in the directory.  Each entry is prefixed
  with `○` (pending) or `✓` (approved).
- **Right panel** — task metadata (read-only) and four editable fields:
  `Background`, `Cause`, `Resolution`, and `Procedure` (one step per line).

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate the draft list (auto-saves current draft) |
| `Ctrl+D` | Approve current draft and jump to the next pending one |
| `Ctrl+S` | Save edits without approving |
| `Ctrl+Q` | Quit (saves current draft first) |

### Review guidelines

1. Read the `Hint` line at the top of the detail panel.
2. **New drafts** (`Matched: —`): read all four fields carefully.  The LLM
   draft is a starting point — verify or rewrite as needed.
3. **Pre-filled drafts** (`Matched: task_XXXX.json`): the hint lists task
   feature differences (taskType, site, coreCount, …).  If those differences
   do not affect the root cause, a quick read is sufficient.
4. Press `Ctrl+D` when satisfied.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--drafts` | `drafts/` | Directory containing draft JSON files |

### Manual alternative

If you prefer to skip the TUI, edit the JSON files directly and set
`"reviewed": true` in each file you want to approve.  Only drafts with
`"reviewed": true` are processed by `batch-populate`.

---

## Step 3 — Populate databases: `bamboo batch-populate`

```bash
bamboo batch-populate \
  --drafts drafts/ \
  --save-to approved_email_drafts/
```

For each `reviewed: true` draft:

1. Serializes `email_body` to text.
2. Calls `process_knowledge(email_text=..., task_data=...)` for each task_id.
3. On success, removes the task_id from the draft's `task_ids` list.
4. After all task_ids are processed, archives the draft to `approved_email_drafts/`.

**Archival rules:**

| Condition | Action |
|-----------|--------|
| `matched_from: null` (new draft) | Always archive |
| Pre-filled, email_body unchanged | Skip (source already in library) |
| Pre-filled, email_body changed | Archive as new entry (source preserved) |

Failed tasks remain in `task_ids` for the next run.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--drafts` | `drafts/` | Directory with reviewed draft JSON files |
| `--save-to` | `approved_email_drafts/` | Archive destination |
| `--dry-run` | off | Preview without writing to any database |
| `--yes` / `-y` | off | Skip confirmation prompt |
| `--concurrency` | `3` | Max concurrent `process_knowledge` calls |
| `-v` / `--verbose` | off | Enable DEBUG logging |

---

## Incremental batches

The workflow is designed to be run repeatedly.  Across multiple batches:

- Tasks already in the DB are skipped automatically (DB coverage check).
- Tasks whose error pattern matches a prior reviewed draft are pre-filled
  (approved-draft check).  The reviewer only needs a quick sanity check.
- Truly new failure modes (no prior knowledge) get a full LLM draft + review.

As more knowledge accumulates, the proportion of "new" tasks decreases.

---

## Database reset recovery

If the graph and vector databases are reset:

1. Run `bamboo seed-drafts` on the original (or a new) CSV.
2. The DB coverage check finds nothing.
3. The approved-draft check matches all entries in `approved_email_drafts/`.
4. All tasks appear as pre-filled drafts — quick human verify, then
   `bamboo batch-populate` repopulates both databases without re-authoring.

---

## Known limitations

**errorDialog-only matching** — the deduplication and coverage checks use
the canonical errorDialog as the sole matching key.  The same error message
can have different root causes depending on task configuration (site, taskType,
prodSourceLabel, coreCount, splitRule).  The `review_hint` in pre-filled drafts
shows these differences explicitly so the reviewer can catch mismatches.

