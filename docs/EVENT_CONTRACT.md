# Omni-Responder Event Contract v1

Every component emits **newline-delimited JSON** to the bus. One envelope, five payload types.
H1 and H2: build against this. H3 renders anything conforming to it.

Ship v1 unchanged through code freeze. If you need a field, add it — never rename or remove one.

## Envelope

```json
{
  "event_id": "evt_0007",
  "t": 12.4,
  "ts": "2026-08-15T18:22:04.113Z",
  "kind": "perception",
  "source": "cosmos-reason-2",
  "payload": { }
}
```

| Field      | Type   | Notes |
|------------|--------|-------|
| `event_id` | string | Unique. Monotonic is nice, not required. |
| `t`        | number | **Seconds since incident clock start.** Drives replay + the UI timeline. Required. |
| `ts`       | string | ISO-8601 wall clock. |
| `kind`     | enum   | `perception` \| `reasoning` \| `tool_call` \| `dispatch` \| `telemetry` |
| `source`   | string | Model or agent name. Shown verbatim in the log. |

## `kind: "perception"` — Hacker 1

Emitted when the VLM returns a scene judgment. Include `latency_ms`; the pitch needs it.

```json
{
  "camera_id": "WSDOT-I5-NB-164",
  "location": { "name": "I-5 NB @ NE 145th St", "lat": 47.7341, "lon": -122.3218 },
  "incident_type": "hazmat_release",
  "vehicles_involved": 3,
  "vehicle_types": ["tanker", "sedan", "suv"],
  "hazard_description": "Overturned tanker leaking fluid onto roadway; vapor cloud drifting north.",
  "placard": { "detected": true, "un_number": "1203", "confidence": 0.86 },
  "lanes_blocked": 3,
  "casualties_visible": 2,
  "severity": 3,
  "confidence": 0.91,
  "wind": { "from_deg": 200, "speed_mps": 4.1 },
  "clip_time": 12.4,
  "latency_ms": 840
}
```

- `severity`: 0 none / 1 minor / 2 major / 3 mass-casualty or hazmat.
- `incident_type`: `collision` \| `hazmat_release` \| `fire` \| `smoke` \| `debris` \| `clear`.
- `clip_time` is the offset **into the video file** — the UI seeks the player to it.
- Omit `placard` / `wind` when not applicable. Don't send nulls with fake values.

## `kind: "reasoning"` — Hacker 2

Nemotron's decision step. Keep `text` under ~200 chars; it renders in a log line.

```json
{
  "text": "Placard UN1203 with 3 lanes blocked and casualties visible. Escalating to hazmat protocol before routing.",
  "decision": "escalate_hazmat",
  "confidence": 0.88
}
```

## `kind: "tool_call"` — Hacker 2

One event per sub-agent tool invocation. **Emit it even when it fails** — visible error recovery scores under Technical Execution.

```json
{
  "agent": "hazmat_agent",
  "tool": "erg_lookup",
  "args": { "un_number": "1203" },
  "status": "ok",
  "result": {
    "material": "Gasoline",
    "guide": "128",
    "isolation_m": 50,
    "downwind_day_m": 300,
    "downwind_night_m": 1100,
    "ppe": "Level B"
  },
  "duration_ms": 210
}
```

`status`: `ok` \| `error` \| `retry`. On error, send `"error": "..."` instead of `result`.

**Map-driving keys.** These render as geometry, so use exactly these names:

- `isolation_m` → red hot-zone ring
- `downwind_day_m` / `downwind_night_m` → green protective-action wedge, oriented by `wind.from_deg`
- `route` on a `traffic_agent` call → detour polyline, as `[[lat, lon], ...]`

## `kind: "dispatch"` — Hacker 2

The outbound action. This is the payoff moment in the demo — make the brief read like real CAD.

```json
{
  "channel": "911_cad",
  "priority": 1,
  "recipients": ["Seattle Fire HazMat 1", "WSP District 2"],
  "brief": "HAZMAT / MVC. I-5 NB @ NE 145th. Overturned tanker, UN1203 gasoline...",
  "acknowledged": false
}
```

`channel`: `911_cad` \| `ems` \| `hazmat` \| `vms_sign` \| `public_alert`.

## `kind: "telemetry"` — Hacker 1

Every ~2s. Powers the hardware story.

```json
{
  "gpu_util_pct": 71,
  "unified_mem_used_gb": 96.4,
  "unified_mem_total_gb": 128,
  "streams_active": 3,
  "vlm_calls_total": 14,
  "video_bytes_egressed": 0
}
```

`video_bytes_egressed` must be measured, not hardcoded. A judge may ask how you know.

## Transport

- **Dev:** append to `dashboard/fixtures/*.jsonl`, replay with the mock server.
- **Live:** `POST /ingest` with a single envelope as the body. The dashboard fans it out over WebSocket.

```python
import requests
requests.post("http://localhost:8080/ingest", json=envelope, timeout=1)
```

Wrap that in try/except and never let a dashboard failure kill your pipeline.
