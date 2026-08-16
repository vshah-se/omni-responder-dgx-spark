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
import contextlib
import json
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if CFG["replay"] and CFG["fixture"]:
        asyncio.create_task(replay_fixture())
    yield

app = FastAPI(title="Omni-Responder Bus", lifespan=lifespan)

clients: set[WebSocket] = set()
history: list[dict] = []          # replayed to late joiners so a reconnect isn't a blank screen
CFG = {"fixture": None, "speed": 1.0, "loop": False, "replay": True, "port": 8080}
RUN = {"proc": None, "video": None, "started": None}      # at most one analysis at a time
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


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


# Video needs a route or the player 404s. Two mounts: dashboard/clips for a clip
# you drop in yourself, and the repo's own clips so no copying is needed at all.
for _url, _dir in (("/clips", HERE / "clips"), ("/video", REPO / "data" / "video_clips")):
    if _dir.is_dir():
        app.mount(_url, StaticFiles(directory=str(_dir)), name=_url.strip("/"))


@app.get("/")
async def index():
    return FileResponse(HERE / "index.html")


@app.get("/clip-manifest")
async def clip_manifest():
    """Which videos are actually available, smallest first. The page probes
    this instead of hardcoding a filename that may not exist.
    All extensions in VIDEO_EXT are included, not just .mp4."""
    out = []
    seen = set()
    for url, d in (("/clips", HERE / "clips"), ("/video", REPO / "data" / "video_clips")):
        if not d.is_dir():
            continue
        files = sorted(
            (p for ext in VIDEO_EXT for p in d.glob(f"*{ext}")),
            key=lambda x: x.stat().st_size
        )
        for f in files:
            if f.name in seen:
                continue
            seen.add(f.name)
            out.append({"url": f"{url}/{f.name}",
                        "name": f.name,
                        "mb": round(f.stat().st_size / 1e6, 1)})
    return {"clips": out}


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


def _safe_video(raw: str) -> pathlib.Path:
    """Resolve a user-supplied path and refuse anything outside the repo.

    The dashboard hands this straight to a subprocess, so an unchecked path
    would let anyone on the network point it at arbitrary files."""
    p = pathlib.Path(raw).expanduser()
    if not p.is_absolute():
        p = (REPO / p)
    p = p.resolve()
    if p.suffix.lower() not in VIDEO_EXT:
        raise ValueError(f"not a video file: {p.suffix or 'no extension'}")
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    if REPO.resolve() not in p.parents:
        raise ValueError("path must be inside the project directory")
    return p


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a video from the browser and drop it in dashboard/clips/."""
    name = pathlib.Path(file.filename or "upload.mp4").name
    if pathlib.Path(name).suffix.lower() not in VIDEO_EXT:
        return JSONResponse({"ok": False, "error": "unsupported file type"}, status_code=400)
    dest = HERE / "clips" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"ok": True, "path": str(dest.relative_to(REPO)),
            "url": f"/clips/{name}", "mb": round(dest.stat().st_size / 1e6, 1)}


@app.get("/run-status")
async def run_status():
    proc = RUN["proc"]
    return {"running": bool(proc and proc.poll() is None),
            "video": RUN["video"], "started": RUN["started"]}


@app.post("/stop")
async def stop():
    proc = RUN["proc"]
    if proc and proc.poll() is None:
        proc.terminate()
        return {"ok": True, "stopped": True}
    return {"ok": True, "stopped": False}


@app.post("/run")
async def run(request: Request):
    """Launch adapter.py against the chosen video. This is what makes the picker
    a control surface rather than just a player."""
    body = await request.json()
    try:
        video = _safe_video(body.get("path", ""))
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    proc = RUN["proc"]
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    history.clear()
    await broadcast({"kind": "control", "payload": {"action": "reset"}})

    cmd = [sys.executable, str(HERE / "adapter.py"),
           "--video", str(video),
           "--bus", f"http://127.0.0.1:{CFG['port']}/ingest"]
    for flag in ("lat", "lon", "wind_from", "location"):
        if body.get(flag) not in (None, ""):
            cmd += [f"--{flag.replace('_', '-')}", str(body[flag])]

    RUN["proc"] = subprocess.Popen(cmd, cwd=str(REPO),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    RUN["video"] = str(video.relative_to(REPO.resolve()))
    RUN["started"] = datetime.now(timezone.utc).isoformat()
    print(f"[run] {' '.join(cmd)}")
    return {"ok": True, "video": RUN["video"]}


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

    CFG.update(fixture=args.fixture, speed=args.speed, loop=args.loop,
               replay=args.replay, port=args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
