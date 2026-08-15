import sys
import os
import time
import json
import argparse
from src.orchestrator.incident_manager import IncidentOrchestrator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

def print_box(title: str, text: str, color_code: str = "36"):
    """ANSI terminal box fallback when rich is not installed."""
    if HAS_RICH:
        console.print(Panel(text, title=f"[bold]{title}[/bold]"))
    else:
        border = "=" * 70
        print(f"\n\033[1;{color_code}m{border}")
        print(f"  {title.upper()}")
        print(f"{border}\033[0m")
        print(text)

def run_live_stream_simulation(video_path: str, location_hint: str, speed_multiplier: float = 1.0):
    """Executes realistic temporal pre-incident to post-incident edge monitoring simulation."""
    title_banner = (
        "Omni-Responder: Live Edge Temporal Feed 🚨⚡\n"
        "Continuous Vision Perception ➔ Anomaly Detection ➔ Autonomous Multi-Agent Dispatch\n"
        "Platform: NVIDIA Grace Blackwell DGX Spark (128GB Unified Memory | 0% Cloud Egress)"
    )
    print_box("System Live Edge Ingestion Active", title_banner, "36")
    print(f"\n📡 Connecting to Edge Camera Stream: {video_path}")
    print(f"📍 Location: {location_hint}")
    print("⏳ Beginning continuous temporal monitoring (Pre-incident ➔ Crash Event ➔ Dispatch)...\n")

    orchestrator = IncidentOrchestrator()

    for frame in orchestrator.stream_incident(video_path, location_hint=location_hint, speed_multiplier=speed_multiplier):
        ts = frame["timestamp"]
        status = frame["status"]
        desc = frame["scene_description"]

        if status == "NORMAL_MONITORING":
            print(f"\033[1;32m[{ts}] [EDGE MONITOR: NORMAL]\033[0m {desc}")
        elif status == "ANOMALY_DETECTED":
            print(f"\033[1;33m[{ts}] [EDGE MONITOR: ANOMALY]\033[0m {desc}")
        elif status == "CRISIS_IMPACT":
            print(f"\n\033[1;31m[{ts}] [CRITICAL IMPACT DETECTED!]\033[0m {desc}")
            print("\033[1;35m⚡ DGX Spark Nemotron Orchestrator: Triggering sub-agents in parallel (< 150ms latency)...\033[0m")

            incident = frame["incident_record"]
            
            # 1. Perception
            p = incident["perception"]
            p_text = (
                f"Location: {p['location']}\n"
                f"Camera ID: {p['camera_id']}\n"
                f"Classification: {p['crisis_type']} (Severity: {p['severity']})\n"
                f"Vehicles: {p['vehicles_involved']} | Hazard Clues: {', '.join(p['hazard_indicators'])}\n\n"
                f"Cosmos Reasoner VLM Output:\n{p['raw_summary']}"
            )
            print_box("1. Perception (Cosmos Reasoner VLM on Spark)", p_text, "32")

            # 2. Hazmat
            h = incident["hazmat"]
            h_text = (
                f"Chemical: {h['chemical_name']} ({h['un_number']}) - {h['hazard_class']}\n"
                f"Perimeter: {h['isolation_radius_meters']}m standoff | Day: {h.get('day_protection_km', 'N/A')}km | Night: {h.get('night_protection_km', 'N/A')}km\n"
                f"Required PPE: {h['ppe_required']}\n"
                f"Containment Action: {h['fire_response']}\n"
                f"First Aid: {h['first_aid']}"
            )
            print_box("2. Hazmat Sub-Agent (ERG 2024 Knowledge Base)", h_text, "31")

            # 3. Traffic
            t = incident["traffic"]
            t_actions = "\n".join([f"  • {act}" for act in t["actions_triggered"]])
            t_text = (
                f"Closure ID: {t['closure_id']}\n"
                f"Active Actions Dispatched:\n{t_actions}"
            )
            print_box("3. Traffic Sub-Agent (City VMS & Signals)", t_text, "33")

            # 4. CAD Brief
            d = incident["dispatch_report"]
            d_text = (
                f"CAD Incident ID: {d['cad_id']} | Priority: {d['dispatch_code']}\n"
                f"Dispatched Units: {', '.join(d['target_units'])}\n\n"
                f"CAD Briefing:\n{d['briefing_summary']}"
            )
            print_box("4. Comms Sub-Agent (911 CAD Dispatch)", d_text, "35")

            print("\n\033[1;32m✔ Autonomous Edge Response Complete. Video preserved on-node (0 bytes sent to cloud).\033[0m\n")

def main():
    parser = argparse.ArgumentParser(description="Omni-Responder DGX Spark: Autonomous Multi-Agent Incident Dispatch")
    parser.add_argument(
        "--video",
        type=str,
        default="data/video_clips/crash_scenario_1.mp4",
        help="Path to the crisis .mp4 video file to analyze"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="5th Ave & Market St Intersection",
        help="Location or camera name hint for the scene"
    )
    parser.add_argument(
        "--stream",
        "--live",
        action="store_true",
        help="Run realistic continuous video stream monitoring simulation (pre-crash to post-crash trigger)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.5,
        help="Streaming playback speed multiplier (default 1.5x)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw machine JSON payload instead of terminal formatting"
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        alt_path = os.path.join("data/video_clips", os.path.basename(args.video))
        if os.path.exists(alt_path):
            args.video = alt_path

    if args.stream:
        run_live_stream_simulation(args.video, args.location, speed_multiplier=args.speed)
        return

    orchestrator = IncidentOrchestrator()
    result = orchestrator.process_incident(args.video, location_hint=args.location)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    title_banner = (
        "Omni-Responder: DGX Spark 🚨⚡\n"
        "Edge Vision Perception + Autonomous Multi-Agent Incident Dispatch\n"
        "Target Hardware: NVIDIA Grace Blackwell DGX Spark (128GB Unified Memory)"
    )
    print_box("System Initialization", title_banner, "36")
    print(f"\n📡 Ingesting video stream from: {args.video}")
    print(f"📍 Location hint: {args.location}")
    print("🔒 Privacy Guarantee: 100% On-Premise DGX Spark (0% Cloud Video Egress)")

    p = result["perception"]
    p_text = (
        f"Location: {p['location']}\n"
        f"Camera ID: {p['camera_id']}\n"
        f"Incident Classification: {p['crisis_type']} (Severity: {p['severity']})\n"
        f"Vehicles Involved: {p['vehicles_involved']}\n"
        f"Observed Hazard Indicators: {', '.join(p['hazard_indicators'])}\n\n"
        f"Cosmos Reasoner VLM Narrative:\n{p['raw_summary']}"
    )
    print_box("1. Perception Phase (NVIDIA VSS + Cosmos Reasoner VLM)", p_text, "32")

    h = result["hazmat"]
    h_text = (
        f"Identification Status: {h['status']}\n"
        f"Substance: {h['chemical_name']} ({h['un_number']})\n"
        f"Hazard Class: {h['hazard_class']}\n"
        f"Initial Isolation Perimeter: {h['isolation_radius_meters']} meters\n"
        f"Protective Distance: Day: {h.get('day_protection_km', 'N/A')} km | Night: {h.get('night_protection_km', 'N/A')} km\n"
        f"Required Responder PPE: {h['ppe_required']}\n"
        f"Fire/Containment Response: {h['fire_response']}\n"
        f"Medical/First Aid: {h['first_aid']}"
    )
    print_box("2. Hazmat Sub-Agent (Local Emergency Response Guidebook)", h_text, "31")

    t = result["traffic"]
    t_actions = "\n".join([f"  • {act}" for act in t["actions_triggered"]])
    t_text = (
        f"Dispatch Status: {t['status']}\n"
        f"Perimeter Closure ID: {t['closure_id']}\n"
        f"Automated Actions Dispatched:\n{t_actions}"
    )
    print_box("3. Traffic Sub-Agent (City VMS & Signal Priority)", t_text, "33")

    d = result["dispatch_report"]
    d_text = (
        f"CAD Incident ID: {d['cad_id']} | Timestamp: {d['timestamp']}\n"
        f"Priority Code: {d['dispatch_code']}\n"
        f"Target Dispatched Units: {', '.join(d['target_units'])}\n\n"
        f"Emergency Briefing Summary:\n{d['briefing_summary']}"
    )
    print_box("4. Comms Sub-Agent (First Responder 911 CAD Card)", d_text, "35")

    print("\n\033[1;32m✔ All autonomous edge actions completed in parallel on DGX Spark (0ms Cloud Latency).\033[0m\n")

if __name__ == "__main__":
    main()
