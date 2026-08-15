"""
Omni-Responder event bus + dashboard host.

Two modes, same WebSocket:
  replay  -- plays a .jsonl fixture with real timing, so H3 can build with zero
             dependency on H1/H2 being finished
  live    -- H1/H2 POST envelopes to /ingest, fanned out to every client

    pip install "fastapi>=0.110" "uvicorn[standard]"
    python server.py --fixture fixtures/tanker_i5.jsonl --speed 1.0
    open http://localhost:8080

Flags:
  --speed 2.0     replay faster (demo reruns)
  --loop          restart the fixture when it ends
  --no-replay     live only; wait for /ingest
"""

import argparse
import asyncio
import json
import pathlib
from datetime import datetime, timezone

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

HERE = pathlib.Path(__file__).parent
app = FastAPI(title="Omni-Responder Bus")

clients: set[WebSocket] = set()
history: list[dict] = []          # replayed to late joiners so a reconnect isn't a blank screen
CFG = {"fixture": None, "speed": 1.0, "loop": False, "replay": True}


async def broadcast(event: dict) -> None:
    history.append(event)
    dead = []
    for ws in clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


@app.get("/")
async def index():
    return FileResponse(HERE / "index.html")


@app.post("/ingest")
async def ingest(request: Request):
    """H1/H2 push here. Tolerant on purpose: a bad event must not kill a pipeline."""
    try:
        event = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "body was not JSON"}, status_code=400)

    if not isinstance(event, dict) or "kind" not in event:
        return JSONResponse({"ok": False, "error": "missing 'kind'"}, status_code=400)

    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    event.setdefault("event_id", f"evt_{len(history):04d}")
    event.setdefault("t", round(len(history) * 0.5, 1))
    event.setdefault("source", "unknown")
    event.setdefault("payload", {})

    await broadcast(event)
    return {"ok": True, "event_id": event["event_id"]}


@app.post("/reset")
async def reset():
    history.clear()
    await broadcast({"kind": "control", "payload": {"action": "reset"}})
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        for event in history:          # catch the new client up
            await ws.send_json(event)
        while True:
            await ws.receive_text()    # keepalive; we don't expect inbound
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


async def replay_fixture():
    """Honour each event's `t` so the demo has the pacing of a real incident."""
    path = HERE / CFG["fixture"]
    events = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]
    events.sort(key=lambda e: e.get("t", 0))
    print(f"[replay] {len(events)} events from {path.name} at {CFG['speed']}x")

    while True:
        await asyncio.sleep(1.5)       # let a browser connect before the clock starts
        history.clear()
        await broadcast({"kind": "control", "payload": {"action": "reset"}})
        elapsed = 0.0
        for event in events:
            gap = max(0.0, event.get("t", 0) - elapsed)
            await asyncio.sleep(gap / CFG["speed"])
            elapsed = event.get("t", 0)
            event.setdefault("ts", datetime.now(timezone.utc).isoformat())
            await broadcast(event)
            print(f"  t={elapsed:6.1f}  {event['kind']:<11} {event.get('source','')}")
        print("[replay] complete")
        if not CFG["loop"]:
            return


@app.on_event("startup")
async def startup():
    if CFG["replay"] and CFG["fixture"]:
        asyncio.create_task(replay_fixture())


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="fixtures/tanker_i5.jsonl")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-replay", dest="replay", action="store_false")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    CFG.update(fixture=args.fixture, speed=args.speed, loop=args.loop, replay=args.replay)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
