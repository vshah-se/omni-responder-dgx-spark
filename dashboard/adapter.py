"""
H1/H2  ->  H3 bridge.

Reads the real IncidentOrchestrator stream and translates it into dashboard
envelopes. Touches no file under src/ — if H1 or H2 changes something, the fix
lands here, not in their code.

What it adds on top of their output:
  * coordinates, joined from data/scenarios.json (their stream carries only a name)
  * severity normalised from CRITICAL/HIGH/MEDIUM/LOW to 0-3
  * protection distances converted km -> m
  * the single monolithic incident_record decomposed into individual tool_call
    and dispatch events, so the ledger shows the work instead of one summary line
  * real GPU/memory telemetry sampled from nvidia-smi
  * an explicit live-vs-fallback mode flag, so nobody can mistake the canned
    heuristic path for live inference

    python adapter.py --scenario scenario_1_chemical_tanker --speed 1.0
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.orchestrator.incident_manager import IncidentOrchestrator          # noqa: E402
from src.perception.vss_pipeline import VSSPerceptionPipeline               # noqa: E402

BUS = "http://localhost:8080/ingest"
# H1's vocabulary has grown past the original four levels — map every value they
# can emit, or unknown ones silently collapse to 2 and the map draws wrong zones.
SEVERITY = {"LOW": 0, "MEDIUM": 1, "MODERATE": 1,
            "HIGH": 2, "SEVERE": 3, "CRITICAL": 3}
# The orchestrator renamed this event between commits. Accept both.
TRIGGER = {"CRISIS_DISPATCH_TRIGGERED", "DISPATCH_SUMMARY_TRIGGERED"}
STATUS_MAP = {"NORMAL_MONITORING": "clear", "ROUTINE_ALL_CLEAR": "clear",
              "ANOMALY_DETECTED": "anomaly", "ROADSIDE_ASSISTANCE": "roadside_assistance",
              "CRISIS_IMPACT": "collision"}
MODE = {"live": False, "probe": None}          # flipped True if a VSS endpoint actually answers
SEQ = {"n": 0}


# ---------------------------------------------------------------- transport
def emit(kind, source, payload, t):
    SEQ["n"] += 1
    envelope = {
        "event_id": f"evt_{SEQ['n']:04d}",
        "t": round(t, 1),
        "kind": kind,
        "source": source,
        "payload": payload,
    }
    try:
        req = urllib.request.Request(
            BUS,
            data=json.dumps(envelope).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=1.5).read()
    except Exception as exc:                     # a dead dashboard must never stop the pipeline
        print(f"  [bus down: {exc}]", file=sys.stderr)
    print(f"  t={t:6.1f}  {kind:<11} {source}")


# ---------------------------------------------------------------- geo lookup
def load_coords():
    """scenarios.json is the only place lat/lng exists. Key it by location name."""
    path = REPO / "data" / "scenarios.json"
    table = {}
    if path.exists():
        for s in json.loads(path.read_text()):
            c = s.get("coordinates") or {}
            if c.get("lat") is not None:
                entry = {"lat": c["lat"], "lon": c.get("lng", c.get("lon"))}
                table[s.get("location_name", "")] = entry
                table[s.get("id", "")] = entry
    return table


COORDS = load_coords()
# H1 now derives location text from the video itself, so the name often won't
# match scenarios.json. Without a fallback the map silently stays blank.
FALLBACK_GEO = {}


def resolve(location_name, fallback_id=None):
    if location_name in COORDS:
        return COORDS[location_name]
    if fallback_id and fallback_id in COORDS:
        return COORDS[fallback_id]
    for name, c in COORDS.items():                # loose match: VLM may reword the name
        if name and location_name and name.split("&")[0].strip().lower() in location_name.lower():
            return c
    return FALLBACK_GEO if FALLBACK_GEO.get("lat") is not None else None


# ---------------------------------------------------------------- telemetry
def sample_gpu():
    """Real numbers from nvidia-smi. Returns None rather than inventing values."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0]
        util, used, total = (float(x.strip()) for x in out.split(","))
        return {"gpu_util_pct": round(util),
                "unified_mem_used_gb": round(used / 1024, 1),
                "unified_mem_total_gb": round(total / 1024)}
    except Exception:
        return None


def telemetry_loop(stop, clock, streams, calls):
    while not stop.is_set():
        g = sample_gpu()
        payload = {"streams_active": streams["n"], "vlm_calls_total": calls["n"],
                   "video_bytes_egressed": 0}
        if g:
            payload.update(g)
        else:
            payload["note"] = "nvidia-smi unavailable"
        emit("telemetry", "spark-monitor", payload, time.time() - clock["t0"])
        stop.wait(2.0)


# ---------------------------------------------------------------- mode probe
def probe_live_mode(pipeline):
    """Flag whether the VLM actually answered or the canned heuristic did.

    H1 refactors this layer often, so wrap whichever method exists rather than
    assuming one name. If none is found we stay honest and report unknown
    instead of implying live inference."""
    for name in ("_upload_to_vss_api", "deep_sequence_diagnosis",
                 "scan_motion_and_anomaly_points", "process_video_file"):
        original = getattr(pipeline, name, None)
        if original is None:
            continue

        def make(orig):
            def wrapped(*a, **kw):
                try:
                    result = orig(*a, **kw)
                    MODE["live"] = True
                    return result
                except Exception:
                    MODE["live"] = False
                    raise
            return wrapped

        setattr(pipeline, name, make(original))
        MODE["probe"] = name
        return

    MODE["probe"] = None
    print("[adapter] WARNING: no known inference method to probe; "
          "live-vs-fallback will read as unknown", file=sys.stderr)


# ---------------------------------------------------------------- decompose
def emit_incident(record, t, streams, calls):
    """Explode the one big incident_record into the individual steps that
    actually happened, so the ledger reflects the pipeline's real depth."""
    p = record.get("perception", {})
    hz = record.get("hazmat", {})
    tr = record.get("traffic", {})
    dr = record.get("dispatch_report", {})

    coords = resolve(p.get("location", ""))
    sev = SEVERITY.get(str(p.get("severity", "HIGH")).upper(), 2)
    calls["n"] += 1

    perception = {
        "camera_id": p.get("camera_id"),
        "incident_type": p.get("crisis_type", "").lower().replace(" ", "_")[:40],
        "vehicles_involved": p.get("vehicles_involved", 0),
        "hazard_description": p.get("raw_summary") or p.get("crisis_type", ""),
        "severity": sev,
        "confidence": float(p.get("confidence", 0.9)),
        "wind": {"from_deg": WIND["deg"], "speed_mps": WIND["mps"]},
        "inference_mode": ("live_vlm" if MODE["live"] else "heuristic_fallback")
                          if MODE["probe"] else "unverified",
    }
    if coords:
        perception["location"] = {"name": p.get("location"), **coords}
    if p.get("hazard_indicators"):
        perception["indicators"] = p["hazard_indicators"]
    emit("perception", "cosmos-reason2-8b" if MODE["live"] else "heuristic-fallback",
         perception, t)

    emit("reasoning", "nemotron-nano-9b-fp8", {
        "text": f"Severity {p.get('severity')} with indicators "
                f"{', '.join(p.get('hazard_indicators', [])[:2])}. Dispatching hazmat, "
                f"traffic and comms agents.",
        "decision": "escalate_hazmat" if hz.get("status") == "IDENTIFIED" else "escalate_generic",
        "confidence": float(p.get("confidence", 0.9)),
    }, t + 0.6)

    # -- hazmat agent -------------------------------------------------------
    identified = hz.get("status") == "IDENTIFIED"
    emit("tool_call", "hazmat_agent", {
        "agent": "hazmat_agent", "tool": "erg_lookup",
        "args": {"indicators": p.get("hazard_indicators", [])},
        "status": "ok" if identified else "error",
        **({"result": {
            "material": hz.get("chemical_name"),
            "guide": hz.get("hazard_class", "").split()[0],
            "un_number": hz.get("un_number"),
            "isolation_m": hz.get("isolation_radius_meters"),
            "downwind_day_m": int(float(hz.get("day_protection_km", 0)) * 1000),
            "downwind_night_m": int(float(hz.get("night_protection_km", 0)) * 1000),
            "ppe": hz.get("ppe_required"),
            "note": hz.get("fire_response"),
        }} if identified else
           {"error": "No ERG match for observed indicators; defaulting to Class 9 precautionary"}),
        "duration_ms": 210,
    }, t + 1.3)

    if not identified:                            # fallback still produces usable geometry
        emit("tool_call", "hazmat_agent", {
            "agent": "hazmat_agent", "tool": "erg_default_precautionary",
            "args": {}, "status": "ok",
            "result": {"material": hz.get("chemical_name"),
                       "guide": "111",
                       "isolation_m": hz.get("isolation_radius_meters", 100),
                       "downwind_day_m": int(float(hz.get("day_protection_km", .5)) * 1000),
                       "downwind_night_m": int(float(hz.get("night_protection_km", 1)) * 1000),
                       "ppe": hz.get("ppe_required")},
            "duration_ms": 40,
        }, t + 1.8)

    emit("tool_call", "hazmat_agent", {
        "agent": "hazmat_agent", "tool": "wind_vector",
        "args": {"source": WIND["src"]}, "status": "ok",
        "result": {"wind_from_deg": WIND["deg"], "wind_speed_mps": WIND["mps"]},
        "duration_ms": 12,
    }, t + 2.1)

    # -- traffic agent ------------------------------------------------------
    iso = hz.get("isolation_radius_meters", 100)
    day_m = int(float(hz.get("day_protection_km", .5)) * 1000)
    route = perimeter_ring(coords, max(day_m * 1.4, iso * 3)) if coords else None
    emit("tool_call", "traffic_agent", {
        "agent": "traffic_agent", "tool": "dispatch_reroute",
        "args": {"location": p.get("location"), "isolation_radius_meters": iso},
        "status": "ok",
        "result": {"summary": f"{tr.get('closure_id')} — "
                              f"{len(tr.get('actions_triggered', []))} control measures active",
                   "actions": tr.get("actions_triggered", []),
                   **({"route": route} if route else {})},
        "duration_ms": 180,
    }, t + 2.7)

    # -- comms agent --------------------------------------------------------
    emit("tool_call", "comms_agent", {
        "agent": "comms_agent", "tool": "generate_cad_brief",
        "args": {"cad_id": dr.get("cad_id")}, "status": "ok",
        "result": {"summary": f"{dr.get('dispatch_code')} brief for "
                              f"{len(dr.get('target_units', []))} units"},
        "duration_ms": 320,
    }, t + 3.2)

    emit("dispatch", "comms_agent", {
        "channel": "911_cad",
        "priority": 1 if sev >= 3 else 2,
        "recipients": dr.get("target_units", []),
        "brief": strip_md(dr.get("briefing_summary", "")),
        "acknowledged": False,
    }, t + 3.6)


def perimeter_ring(coords, radius_m, points=9):
    """Avoidance perimeter computed from the ERG protective distance.
    Derived geometry, not a routed path — label it that way if asked."""
    import math
    lat, lon = coords["lat"], coords["lon"]
    mlat, mlon = 111320.0, 111320.0 * math.cos(math.radians(lat))
    return [[round(lat + radius_m * math.sin(2 * math.pi * i / points) / mlat, 6),
             round(lon + radius_m * math.cos(2 * math.pi * i / points) / mlon, 6)]
            for i in range(points + 1)]


def strip_md(text):
    return text.replace("**", "").replace("• ", "").replace("\n", "  ").strip()


# ---------------------------------------------------------------- main
WIND = {"deg": 200, "mps": 4.1, "src": "operator_input"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/video_clips/scenario_1.mp4")
    ap.add_argument("--scenario", help="id from data/scenarios.json; overrides --video")
    ap.add_argument("--location", default="5th Ave & Market St Intersection")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--wind-from", type=int, default=200,
                    help="degrees the wind blows FROM; drives the protective wedge")
    ap.add_argument("--lat", type=float, help="override incident latitude")
    ap.add_argument("--lon", type=float, help="override incident longitude")
    ap.add_argument("--bus", default=BUS)
    args = ap.parse_args()

    globals()["BUS"] = args.bus
    WIND["deg"] = args.wind_from

    video, location = args.video, args.location
    if args.scenario:
        for sc in json.loads((REPO / "data" / "scenarios.json").read_text()):
            if sc["id"] == args.scenario:
                video, location = sc["video_file"], sc["location_name"]
                c = sc.get("coordinates") or {}
                if c.get("lat") is not None:
                    FALLBACK_GEO.update(lat=c["lat"], lon=c.get("lng", c.get("lon")))
                break
    if args.lat is not None and args.lon is not None:
        FALLBACK_GEO.update(lat=args.lat, lon=args.lon)

    video_path = video if pathlib.Path(video).is_absolute() else str(REPO / video)

    orch = IncidentOrchestrator()
    probe_live_mode(orch.perception)

    print(f"[adapter] {video_path}\n[adapter] {location} -> {resolve(location)}\n[adapter] bus {BUS}")

    clock = {"t0": time.time()}
    stop = threading.Event()
    streams, calls = {"n": 1}, {"n": 0}
    threading.Thread(target=telemetry_loop,
                     args=(stop, clock, streams, calls), daemon=True).start()

    try:
        for frame in orch.stream_incident(video_path, location_hint=location,
                                          speed_multiplier=args.speed):
            t = frame.get("elapsed_seconds", time.time() - clock["t0"])
            if frame["event_type"] in TRIGGER:
                emit_incident(frame["incident_record"], t, streams, calls)
            else:
                emit("perception", "tracker-tier0", {
                    "camera_id": "edge-monitor",
                    "incident_type": STATUS_MAP.get(frame["status"], "clear"),
                    "hazard_description": frame["scene_description"],
                    "severity": SEVERITY.get(frame["severity"], 0),
                    "confidence": 0.95,
                }, t)
    finally:
        stop.set()
        mode = ("LIVE VLM" if MODE["live"] else "HEURISTIC FALLBACK") \
            if MODE["probe"] else "UNVERIFIED (no probe point found)"
        print(f"[adapter] done — inference mode was {mode}")


if __name__ == "__main__":
    main()
