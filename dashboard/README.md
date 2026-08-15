# Command Dashboard (H3)

Single-pane operations view. Renders perception, agent orchestration, ERG zones,
dispatches, and Spark telemetry from one event stream.

No build step, no npm, no CDN, no map tiles, no webfonts — everything renders
offline. That is deliberate: the project claims zero cloud egress, and a tile
server or a Google Fonts call would quietly contradict it.

## Install

```bash
pip install -r requirements.txt        # from repo root
```

## Run

Two independent modes. Both serve the dashboard at http://localhost:8080

**Fixture replay** — no dependency on H1 or H2. Use this to develop UI, and as
the fallback if the pipeline is down during judging.

```bash
cd dashboard
python server.py --fixture fixtures/tanker_i5.jsonl --loop
```

Flags: `--speed 2.0` to run faster, `--loop` to restart at the end,
`--port 8080`, `--host 0.0.0.0` to reach it from another machine on the Spark's
network.

**Live pipeline** — real events from the orchestrator.

```bash
# terminal 1
cd dashboard && python server.py --no-replay

# terminal 2 — MUST run from the repo root, see note below
python dashboard/adapter.py --scenario scenario_1_chemical_tanker
```

Adapter flags: `--scenario` (id from `data/scenarios.json`), `--video` (path,
if not using a scenario), `--wind-from 200` (degrees the wind blows FROM; drives
the protective wedge), `--speed`, `--bus`.

> **Run the adapter from the repo root.** `settings.hazmat_db_path` is the
> relative string `data/hazmat_db.json`, and `HazmatAgent._load_database`
> returns an empty list when the file is missing — silently. From a
> subdirectory the ERG lookup degrades to "Unidentified Hazardous Substance"
> with no error. Until that path is made absolute, working directory matters.

## Video

Drop a clip at `dashboard/clips/tanker.mp4`. The player seeks to each
perception event's `clip_time`. Without a clip the panel shows an empty state
and nothing else is affected. `.mp4` under `dashboard/` is gitignored, so the
file stays local.

## Sending events from your own code

```python
import requests
requests.post("http://localhost:8080/ingest", json=envelope, timeout=1)
```

Schema is in `docs/EVENT_CONTRACT.md`. Wrap the call in try/except — a dead
dashboard must never take down a pipeline. `POST /reset` clears the board.

## Theme

Follows the OS by default; the header toggle overrides and persists. Check
which one reads better on the venue projector before presenting — dark themes
wash out badly on some of them.

## Layout

- **Header** — incident clock, egress attestation, on-premise badge, theme toggle
- **Agent strip** — five tiles (perception, orchestrator, hazmat, traffic, comms)
  showing live state, current task, and cumulative calls/ms
- **Camera** — clip playback with detection overlay and perception readout
- **Map** — hand-drawn SVG: isolation ring, downwind protective wedge oriented
  by wind bearing, detour perimeter, scale bar
- **Ledger** — every perception, reasoning step, tool call and dispatch, in order,
  colour-coded by kind; failures render red
- **Right rail** — ERG 2024 card, dispatch log, Spark telemetry

## Honest labelling

Two things on screen are derived, not measured. Say so if asked:

- The **detour perimeter** is a ring computed from the ERG protective distance.
  `traffic_agent` returns text advisories only, so there is no routed path.
- **Wind bearing** comes from the `--wind-from` flag. Nothing upstream supplies it.

The perception tile appends `(fallback)` when `inference_mode` is
`heuristic_fallback`, meaning the VLM did not answer and canned results were
used. Leave that visible.
