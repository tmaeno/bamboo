# Mattermost Integration

Bamboo ships an **ops-facing chat frontend**: a Mattermost bot that runs the same
engines as the CLI, so operators who don't install bamboo locally can drive it
from chat. It supports three things:

- **`investigate`** — live, turn-by-turn co-investigation of an incident.
- **`capture`** — turn an incident discussion thread into curated knowledge.
- **`analyze --post-to-mattermost`** — (CLI/automation) post an analysis result
  to a channel.

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
| **PanDA OIDC token** | **CERN IAM** (not Mattermost) | Lets bamboo act on PanDA | Service identity *or* per-user `login` (below) |

### PanDA identity: service vs. per-user

- **Service identity** — the bot *process* holds one PanDA OIDC token via the
  usual environment (`PANDA_AUTH=oidc`, `PANDA_AUTH_ID_TOKEN`, `PANDA_AUTH_VO`;
  see [PANDA_INTEGRATION.md](PANDA_INTEGRATION.md)). Every action runs as that
  single identity. This is the default fallback when a user hasn't logged in.

- **Per-user identity** — an operator runs **`login`** in a channel and the bot
  walks them through a CERN IAM device-login. Afterwards *their* PanDA actions
  run under *their* identity. Set `MATTERMOST_REQUIRE_USER_LOGIN=true` to require
  this (no silent service-identity fallback).

  **Device-flow trust model.** The bot process talks to CERN IAM directly to get
  the device code and to poll for the token; the bot posts a verification URL +
  code into the thread; the operator opens that URL in **their own browser** and
  signs in with CERN IAM. Mattermost only relays the URL/code message — neither
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
| `login` | Authenticate as yourself via CERN IAM (per-user PanDA identity). |
| `logout` | Forget your stored token (revert to the service identity). |

**Investigate.** Reply in the thread to drive each turn. Meta-commands (as
replies): `/done`, `/abandon`, `/undo`, `/tool <request>`, `/show-graph`,
`/show-tools`. When the bot proposes side-effecting code it asks you to reply
`y`, `N`, or `edit` before running it. At the end it asks for the cause and
resolution, shows the commit diff (a **Mermaid** graph on instances that support
it), and asks you to confirm before writing to the knowledge base.

**Capture.** The bot reads the thread transcript, asks for the cause and
resolution, extracts the knowledge, shows the commit diff for review, and stores
it on your confirmation.

**Analysis posting (CLI / automation, not in-chat):**

```bash
bamboo analyze --task-id 12345 --post-to-mattermost <channelID>
```

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
   side-effecting actions.

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
  the IAM auth config can be resolved; check the bot host can reach CERN IAM; the
  device code expires after a few minutes, so finish the browser sign-in
  promptly.
- **Commit diff shows raw `mermaid` code** — the Mattermost instance doesn't have
  Mermaid rendering enabled; the diagram still reads as a code block. Enable
  Mermaid on the instance, or rely on the textual summary line.

---

## Limitations

- Prompts/confirmations are **reply-based** rather than interactive
  buttons/dialogs (which would require the bot to also run an inbound HTTP
  callback server). This is a deliberate trade-off to keep the bot to a single
  outbound connection.
