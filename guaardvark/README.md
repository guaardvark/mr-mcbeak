# Mr. McBeak in Guaardvark

This folder is a Guaardvark extension. Linked into a Guaardvark install it adds:

- a **Mr. McBeak** card on the Plugins page. Its switch starts and stops the
  listener, so it opens and closes this machine's microphone. The card is off
  until you switch it on.
- a **Voice Team** page (sidebar, under Voice): whether the microphone is open,
  who is on the team with each member's kind, voice and permissions, adding
  and removing members, and the settings file.
- `/api/mcbeak` in the Guaardvark server, which the page uses.

Mr. McBeak still runs on its own from the same checkout. Nothing here changes
`mcbeak` in a terminal.

## Link it

1. Set Mr. McBeak up first and check it works alone: `./setup.sh`, then
   `mcbeak --doctor`.
2. Link this folder into the Guaardvark checkout under the name `mcbeak`:

   ```
   ln -s /path/to/mr-mcbeak/guaardvark /path/to/guaardvark/extensions/mcbeak
   ```

3. Restart Guaardvark from its own folder: `./stop.sh`, then `./start.sh`.
   The server loads the extension when it starts, and `start.sh` rebuilds the
   web UI with the new page.
4. Open Plugins and switch **Mr. McBeak** on, or open **Voice Team** and press
   **Start listening**.

If a copy is already listening in a terminal, stop it first (`mcbeak --stop`);
only one copy can hold the microphone.

## What runs where

| Piece | What it does |
|---|---|
| `plugin/plugin.json` | Registers the service: id `mcbeak`, `127.0.0.1:8214`, off by default |
| `plugin/scripts/start.sh` | Runs `mcbeak --listen --status-port 8214` with no terminal and waits for `/health` |
| `plugin/scripts/stop.sh` | Stops that listener and checks it is gone; a copy run in a terminal is left alone |
| `api/mcbeak_api.py` | `GET /api/mcbeak/status`, `GET` and `PUT` `/api/mcbeak/team` and `/api/mcbeak/settings` |
| `bridge.py` | Reads and writes the two files for the API, in its own process with Mr. McBeak's Python |
| `frontend/` | The Voice Team page |

The page edits `~/.config/mcbeak/team.yaml`, the file `mcbeak team add` writes,
and `~/.config/mcbeak/settings.yaml`. A team change goes through the same
checks as `mcbeak team add` (a name that sounds like another member's is
refused, for one) and a running listener picks it up by itself. A settings
change is checked the way a start checks it and applies at the next start.
Both files change only from a browser on the machine itself.

Members with no folder of their own work in your home folder when the listener
is started from Guaardvark; set `MCBEAK_WORKSPACE` in Guaardvark's `.env` to
choose another.

## Privacy

The status endpoint answers on `127.0.0.1` only and reports the status, the
version, whether the microphone is open, the team's names and the mode
(`standalone` or `guaardvark`). The log Guaardvark shows for the plugin holds
start, stop and error lines written by the two scripts. What was heard stays in
`~/.local/state/mcbeak/mcbeak.log`, and the listener's own warnings in
`~/.local/state/mcbeak/plugin-listener.err`; neither is shown in the web UI.

## Another port

`8214` is the default. To move it, set `MCBEAK_STATUS_PORT` in Guaardvark's
`.env`, and put the same port in an untracked `plugin/plugin.local.json`:

```
{ "port": 8300, "config": { "service_url": "http://127.0.0.1:8300" } }
```

## Unlink it

1. Stop the listener from the Plugins page.
2. Remove the link: `rm /path/to/guaardvark/extensions/mcbeak`.
3. Restart Guaardvark.

The team and settings files stay where they are.
