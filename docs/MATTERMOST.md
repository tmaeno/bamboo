# Mattermost Integration

Bamboo ships an **ops-facing chat frontend**: a Mattermost bot that runs the same
engines as the CLI, so operators who don't install bamboo locally can drive it
from chat. It supports three things:

- **`investigate`** — live, turn-by-turn co-investigation of an incident.
- **`capture`** — turn an incident discussion thread into curated knowledge.
- **`analyze`** — run a one-shot root-cause analysis on a task and post the
  result card in the thread. (The CLI flag `analyze --post-to-mattermost` is the
  automation/bulk sibling — see [Batch/automation posting](#batchautomation-analysis-posting).)

Interaction is **reply-based**: the bot asks, you reply in the thread. **One
thread = one session.**

---

## Deployment topology — where things run

The bot is **not** a Mattermost plugin and does **not** run inside Mattermost. It
is a standalone process (`bamboo serve-mattermost`) that you run on your own host
or container and that *connects out to* Mattermost:

```
  Mattermost server                bamboo bot process                backends
  (run by your IT)                 (run by you)                      (your infra)
  ┌───────────────┐   WebSocket    ┌────────────────────┐           ┌───────────┐
  │  channels,    │◀──────────────▶│ bamboo             │──────────▶│ Neo4j     │
  │  bot account  │   + REST API   │ serve-mattermost   │           │ Qdrant    │
  └───────────────┘                │  (reads env/.env)  │──────────▶│ LLM API   │
                                   └─────────┬──────────┘           │ PanDA     │
                                             │ device-flow (OIDC)   └───────────┘
                                             ▼
                                            IAM
```

**Configuration is read by the bot process — never sent to Mattermost.** The bot
authenticates *to* Mattermost with `MATTERMOST_TOKEN`. Provide configuration the
same way as any bamboo command:

- a **`.env` file** (searched from the working directory upward, and
  `~/.config/bamboo/.env` — see `bamboo/config.py`), or
- the **shell environment** where you launch the daemon, or
- **systemd** `EnvironmentFile=` / a container `--env-file`.

See [QUICKSTART.md](QUICKSTART.md) for the base environment (LLM keys, Neo4j,
Qdrant) and [PANDA_INTEGRATION.md](PANDA_INTEGRATION.md) for PanDA settings.

---

## Prerequisites

- A working bamboo install with its databases and LLM configured
  ([QUICKSTART.md](QUICKSTART.md)).
- The Mattermost extra:

  ```bash
  pip install 'bamboo[mattermost]'
  ```

- For **per-user PanDA login**, also the PanDA client:

  ```bash
  pip install 'bamboo[panda]'
  ```

---

## Create the bot in Mattermost

1. In the **System Console → Integrations → Bot Accounts**, enable bot accounts
   (and personal access tokens if your instance gates them).
2. Create a bot account for bamboo and **copy its access token** — this is
   `MATTERMOST_TOKEN`.
3. **Add the bot to the channel(s)** the ops team will use. Channel membership is
   your first access-control layer (see [Authorization](#authorization--security)).
4. Get the **channel IDs** for the allow-list: open a channel → *View Info* (the
   ID is shown), or query the Mattermost API. These go in
   `MATTERMOST_ALLOWED_CHANNELS`.

---

## Two distinct tokens

It's important not to conflate the two credentials involved:

| Token | Issued by | Purpose | Where it comes from |
|-------|-----------|---------|---------------------|
| **Mattermost bot token** | Mattermost | Lets the bot talk to Mattermost | `MATTERMOST_TOKEN` (created above) |
| **PanDA OIDC token** | **IAM** (not Mattermost) | Lets bamboo act on PanDA | Service identity *or* per-user `login` (below) |

### PanDA identity: service vs. per-user

- **Service identity** — the bot *process* holds one PanDA OIDC token via the
  usual environment (`PANDA_AUTH=oidc`, `PANDA_AUTH_ID_TOKEN`, `PANDA_AUTH_VO`;
  see [PANDA_INTEGRATION.md](PANDA_INTEGRATION.md)). Every action runs as that
  single identity. This is the default fallback when a user hasn't logged in.

- **Per-user identity** — an operator runs **`login`** in a channel and the bot
  walks them through an OIDC device-code flow. Afterwards *their* PanDA actions
  run under *their* identity. Set `MATTERMOST_REQUIRE_USER_LOGIN=true` to require
  this (no silent service-identity fallback).

  The `login` exchange is sent as a **direct message** from the bot — the
  verification prompt (an "🔐 bamboo login" attachment card with a clickable
  **Authenticate** link to the device-code URL) and the success/failure replies
  go to a private DM with the operator who ran `login`, not the public channel.

  **Device-flow trust model.** The bot process talks to IAM directly to get
  the device code and to poll for the token; the bot posts a verification URL +
  code into the thread; the operator opens that URL in **their own browser** and
  signs in with IAM. Mattermost only relays the URL/code message — neither
  Mattermost nor the bot ever sees the user's IAM credentials, only the issued
  token. The token (incl. refresh token) is stored **on the bot host** under
  `MATTERMOST_TOKEN_DIR` (default `~/.bamboo/mattermost_tokens/<user>/`, dir
  `0700`, files `0600`) — the same trust level as the bot's own service token.

---

## Configure bamboo

Set these in the bot process's environment / `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `MATTERMOST_URL` | yes | Instance base URL, e.g. `https://mattermost.example.org` (optionally `:port`) |
| `MATTERMOST_TOKEN` | yes | Bot account access token |
| `MATTERMOST_ALLOWED_CHANNELS` | yes | Comma-separated channel IDs the bot will act in (empty → it ignores everything) |
| `MATTERMOST_REQUIRE_USER_LOGIN` | no | `true` to require per-user `login` before side-effecting flows (default `false`) |
| `MATTERMOST_TOKEN_DIR` | no | Root for per-user OIDC token stores (default `~/.bamboo/mattermost_tokens`) |

Per-user login additionally needs `PANDA_API_URL_SSL` and `PANDA_AUTH_VO`
(already part of the PanDA setup) to discover the IAM auth config.

---

## Run the bot

```bash
pip install 'bamboo[mattermost]'
bamboo serve-mattermost          # add -v for debug logging
```

It's a long-running service (a single outbound WebSocket + REST connection). A
minimal `systemd` unit:

```ini
[Unit]
Description=bamboo Mattermost bot
After=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/bamboo/bamboo.env
ExecStart=/opt/bamboo/venv/bin/bamboo serve-mattermost
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## Usage

All commands are posted in an **allow-listed** channel. A leading `@bamboo` or
`/bamboo` is accepted but optional.

| Command | What it does |
|---------|--------------|
| `investigate <taskID>` | Start a live co-investigation rooted at this message. Reply in the thread for each turn. |
| `capture [<taskID>]` | After a discussion, ingest the thread as knowledge. |
| `analyze <taskID>` | One-shot root-cause analysis of a task; posts the result card in the thread. |
| `login` | Authenticate as yourself via IAM (per-user PanDA identity). |
| `logout` | Forget your stored token (revert to the service identity). |
| `status` | Check the bot is alive and functional (connection, active sessions, uptime). |
| `help` | Show the list of available commands. |

If you address the bot (`@bamboo …` or `/bamboo …`) with an unrecognized command,
it replies with `help`. Ordinary channel chatter that doesn't address the bot is
ignored.

**Investigate.** Reply in the thread to drive each turn. Meta-commands (as
replies): `/done`, `/abandon`, `/undo`, `/tool <request>`, `/show-graph`,
`/show-tools`. When the bot proposes side-effecting code it asks you to reply
`y`, `N`, or `edit` before running it. At the end it asks for the cause and
resolution, shows the commit diff (a **Mermaid** graph on instances that support
it), and asks you to confirm before writing to the knowledge base.

**Capture.** The bot reads the thread transcript, asks for the cause and
resolution, extracts the knowledge, shows the commit diff for review, and stores
it on your confirmation.

**Analyze.** `analyze <taskID>` fetches the task from PanDA, runs the reasoning
engine, and posts a result card (root cause, confidence, suggested resolution,
plus any novel symptoms / capability gaps) back into the thread. It is read-only
— it queries the knowledge base and PanDA but writes nothing.

**Live progress.** While `investigate`/`capture`/`analyze` run, the bot streams
progress into the thread as a **single live-updating reply**: a head line with an
animated spinner showing the current step and, below it, a **console-style
monospace log** of the last few progress lines, each timestamped. Larger detail
blocks (LLM prompts, log dumps, generated code) are **not** posted to chat —
they're debugging detail that goes to the server log only. When the run finishes
successfully the progress reply is **removed**, leaving just the command and the
result (card / notices). If it fails, the reply is **kept** — frozen to a static
`🔎 <last step>` with the last log lines as a trail. The *full* trace (including
those blocks) is always written under the `bamboo.narration` logger.

**One stream, two views.** Progress narration is a single logging stream
(`bamboo.narration`): the server console and the Mattermost reply render the
*same* records, so what MM shows is always a level-filtered subset of the console
— never a divergent set. Two independent level knobs control the two views:

| Setting | Default | Gates |
|---------|---------|-------|
| `LOG_LEVEL` | `INFO` | the server console / log file (all loggers) |
| `NARRATION_LEVEL` | `INFO` | what narration reaches the Mattermost reply |

So `NARRATION_LEVEL=DEBUG` surfaces verbose detail in chat too (still a subset of
the console at `LOG_LEVEL=DEBUG`); raising it shows less. Operator-significant
warnings/errors are narrated at WARNING/ERROR and appear highlighted (⚠️); ordinary
module logs (`bamboo.agents.*`, db/MCP, …) stay on the console and never reach chat.

The spinner is an **animated custom emoji** (`:bamboo_spinner:`) the bot registers
on startup, because custom emoji render on Mattermost's inline-text path and so
animate reliably (unlike a file-attached GIF, whose inline animation depends on the
client/server preview settings). If the instance disables custom emoji or the bot
lacks permission to create them, the head degrades to a static `🔎` and the status
text still updates live.

### Batch/automation analysis posting

For **scripted/bulk** posting (e.g. a cron job analysing many tasks and pushing
cards to a channel), the CLI exposes the same renderer via `--post-to-mattermost`:

```bash
bamboo analyze --task-id 12345 --post-to-mattermost <channelID>
```

Interactive, single-task analysis does **not** need the CLI — use the in-chat
`analyze <taskID>` command above.

---

## Authorization & security

Two layers:

1. **Who can invoke/approve** — the channel allow-list
   (`MATTERMOST_ALLOWED_CHANNELS`) plus controlled channel membership. The bot
   ignores every channel it isn't allow-listed in, and ignores its own messages.
   *Note:* Mattermost system administrators can read private channels regardless
   of membership.
2. **Whose PanDA identity actions run as** — service identity by default, or
   per-user via `login` (see [above](#panda-identity-service-vs-per-user)). Use
   `MATTERMOST_REQUIRE_USER_LOGIN=true` for genuine per-user attribution of
   side-effecting actions. With this set, if a logged-out user runs
   `investigate`/`capture`/`analyze`, the bot **auto-sends a login link to their
   DM**, waits for them to sign in, then continues the original command under
   their identity — no manual `login` + re-run. (Per-user tokens are checked and
   auto-refreshed each time; expired/near-expiry tokens refresh transparently.)

Side-effecting orchestration code is **never** executed without an explicit reply
confirmation in the thread.

---

## Troubleshooting

- **"requires the optional client" / import error** — install the extra:
  `pip install 'bamboo[mattermost]'` (and `'bamboo[panda]'` for `login`).
- **Bot doesn't respond** — confirm it is a member of the channel, the channel ID
  is in `MATTERMOST_ALLOWED_CHANNELS`, and `MATTERMOST_TOKEN`/`MATTERMOST_URL`
  are correct. Run with `-v` to see WebSocket events.
- **`login` fails** — ensure `PANDA_AUTH_VO` and `PANDA_API_URL_SSL` are set so
  the IAM auth config can be resolved; check the bot host can reach IAM; the
  device code expires after a few minutes, so finish the browser sign-in
  promptly.
- **`login` fails with "certificate verify failed: unable to get local issuer
  certificate"** — the bot's Python has no CA trust store (common on python.org
  framework builds). Run `bamboo verify`: it sets `SSL_CERT_FILE` to the bundled
  `certifi` roots in your `.env`, then restart the bot. Note `PANDA_VERIFY_HOST`
  does **not** affect login — it only applies to PanDA data-plane calls.
- **Live-progress spinner doesn't animate** — the bot registers an animated custom
  emoji (`:bamboo_spinner:`) on startup; if your instance disables custom emoji or
  the bot lacks the "Create Custom Emoji" permission, the head shows a static `🔎`
  instead (the step text and detail still update). Enable custom emoji / grant the
  bot the permission, then restart the bot. The startup log line under
  `bamboo.narration` reports whether the emoji was created/found/refreshed or is
  unavailable.
- **Spinner shows a tiny/old image** — the registered emoji is stale. The bot
  self-heals on startup: it compares the stored emoji image to the bundled
  `spinner.gif` and recreates it on mismatch (log: `spinner emoji refreshed`). If you
  changed the bundled gif, just reinstall (`pip install -U .`) and restart — the new
  image is packaged (the build uses **hatchling**, which always ships current files)
  and the emoji refreshes itself on the next start.
- **Commit diff shows raw `mermaid` code** — the Mattermost instance doesn't have
  Mermaid rendering enabled; the diagram still reads as a code block. Enable
  Mermaid on the instance, or rely on the textual summary line.

---

## Limitations

- Prompts/confirmations are **reply-based** rather than interactive
  buttons/dialogs (which would require the bot to also run an inbound HTTP
  callback server). This is a deliberate trade-off to keep the bot to a single
  outbound connection.
