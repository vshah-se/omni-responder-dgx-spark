import argparse
import sys
import json
import time
from typing import Dict, Any
from src.orchestrator.incident_manager import IncidentOrchestrator
from src.perception.vss_pipeline import VisualContextSummary
from src.config.settings import settings

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    console = Console()
except ImportError:
    console = None

def get_severity_colors(severity: str, dispatch_code: str = "") -> Dict[str, str]:
    """Returns dynamic terminal color styling based on real severity and dispatch code."""
    sev_upper = severity.upper()
    code_upper = dispatch_code.upper()

    if "RED" in code_upper or sev_upper in ["CRITICAL", "SEVERE"]:
        return {
            "border": "red",
            "title_style": "bold red",
            "badge": "🔴 CRITICAL - CODE RED",
            "icon": "🚨"
        }
    elif "AMBER" in code_upper or sev_upper in ["HIGH", "MEDIUM", "MODERATE"]:
        return {
            "border": "yellow",
            "title_style": "bold yellow",
            "badge": "🟡 CAUTION / SAFETY - CODE AMBER",
            "icon": "⚠️"
        }
    else:
        return {
            "border": "green",
            "title_style": "bold green",
            "badge": "🟢 ROUTINE - CODE GREEN (ALL CLEAR)",
            "icon": "✅"
        }

def print_rich_incident_report(record: Dict[str, Any]):
    """Renders formatted multi-agent report with dynamic color-coding based on severity."""
    perception = record.get("perception", {})
    hazmat = record.get("hazmat", {})
    traffic = record.get("traffic", {})
    dispatch = record.get("dispatch_report", {})

    severity = perception.get("severity", "LOW")
    dispatch_code = dispatch.get("dispatch_code", "CODE GREEN")
    colors = get_severity_colors(severity, dispatch_code)

    if console:
        # Header banner
        header_text = Text()
        header_text.append(f"Omni-Responder: {colors['badge']}\n", style=colors["title_style"])
        header_text.append(f"Scene: {perception.get('crisis_type', 'N/A')} | Location: {perception.get('location', 'N/A')}\n", style="white")
        header_text.append(f"Platform: {record['platform']['hardware']} | {record['platform']['privacy_guarantee']}", style="dim")
        console.print(Panel(header_text, style=colors["border"], border_style=colors["border"]))

        # 1. Perception Panel
        percep_text = Text()
        percep_text.append(f"Location: {perception.get('location')}\n", style="white")
        percep_text.append(f"Camera ID: {perception.get('camera_id')}\n", style="white")
        percep_text.append(f"Classification: {perception.get('crisis_type')} (Severity: {perception.get('severity')})\n", style=colors["title_style"])
        percep_text.append(f"Vehicles Involved: {perception.get('vehicles_involved')} | Hazards: {', '.join(perception.get('hazard_indicators', [])) or 'None'}\n\n", style="dim")
        percep_text.append(f"Cosmos Reasoner VLM Output:\n{perception.get('raw_summary')}", style="italic white")
        console.print(Panel(percep_text, title=f"{colors['icon']} 1. Perception (Cosmos Reasoner VLM on Spark)", border_style=colors["border"]))

        # 2. Hazmat Sub-Agent Panel
        haz_border = "green" if hazmat.get("status") == "NO_HAZARDS_DETECTED" else colors["border"]
        hazmat_text = Text()
        hazmat_text.append(f"Status: {hazmat.get('status')}\n", style="bold white")
        hazmat_text.append(f"Chemical: {hazmat.get('chemical_name')} ({hazmat.get('un_number')}) - {hazmat.get('hazard_class')}\n", style="white")
        hazmat_text.append(f"Perimeter: {hazmat.get('isolation_radius_meters')}m standoff | Day: {hazmat.get('day_protection_km')}km | Night: {hazmat.get('night_protection_km')}km\n", style="white")
        hazmat_text.append(f"Required PPE: {hazmat.get('ppe_required')}\n", style="white")
        hazmat_text.append(f"Containment Action: {hazmat.get('fire_response')}\n", style="dim")
        hazmat_text.append(f"First Aid: {hazmat.get('first_aid')}", style="dim")
        console.print(Panel(hazmat_text, title="🧪 2. Hazmat Sub-Agent (ERG 2024 Knowledge Base)", border_style=haz_border))

        # 3. Traffic Sub-Agent Panel
        traf_border = "green" if traffic.get("isolation_radius_meters", 0) == 0 else colors["border"]
        traffic_text = Text()
        traffic_text.append(f"Status: {traffic.get('status')} (Closure ID: {traffic.get('closure_id')})\n", style="bold white")
        traffic_text.append("Active Actions Dispatched:\n", style="white")
        for action in traffic.get("actions_triggered", []):
            traffic_text.append(f"  • {action}\n", style="cyan")
        console.print(Panel(traffic_text, title="🚦 3. Traffic Sub-Agent (City VMS & Signals)", border_style=traf_border))

        # 4. Comms Sub-Agent CAD Panel
        cad_text = Text()
        cad_text.append(f"CAD Incident ID: {dispatch.get('cad_id')} | Priority: {dispatch.get('dispatch_code')}\n", style=colors["title_style"])
        cad_text.append(f"Status: {dispatch.get('status_description')}\n", style="white")
        cad_text.append(f"Target Units: {', '.join(dispatch.get('target_units', []))}\n\n", style="bold white")
        cad_text.append(f"CAD Briefing:\n{dispatch.get('briefing_summary')}", style="white")
        console.print(Panel(cad_text, title="📻 4. Comms Sub-Agent (911 CAD Dispatch)", border_style=colors["border"]))

        console.print(f"\n{colors['icon']} [bold green]Autonomous Edge Processing Complete.[/bold green] Video preserved on-node (0 bytes sent to cloud).\n")

    else:
        print("\n" + "="*70)
        print(f"  OMNI-RESPONDER: {colors['badge']}")
        print("="*70)
        print(f"Perception: {perception.get('crisis_type')} | Severity: {severity}")
        print(f"Cosmos Output: {perception.get('raw_summary')}")
        print(f"Hazmat: {hazmat.get('chemical_name')} ({hazmat.get('un_number')}) - PPE: {hazmat.get('ppe_required')}")
        print(f"Traffic: {traffic.get('status')} ({traffic.get('closure_id')})")
        print(f"CAD Dispatch: {dispatch.get('dispatch_code')} -> {dispatch.get('status_description')}")
        print("="*70 + "\n")

def run_live_stream_simulation(video_path: str, location: str, speed: float):
    """Simulates live continuous edge video feed with dynamic color coding."""
    orchestrator = IncidentOrchestrator()

    if console:
        console.print(Panel(
            Text.from_markup(
                f"[bold cyan]Omni-Responder: Live Edge Temporal Feed[/bold cyan] 🚨⚡\n"
                f"[white]Continuous Vision Perception ➔ Anomaly Detection ➔ Autonomous Multi-Agent Dispatch[/white]\n"
                f"[dim]Platform: {settings.target_hardware} | 0% Cloud Egress[/dim]"
            ),
            title="System Live Edge Ingestion Active",
            border_style="cyan"
        ))
    else:
        print("\n=== Omni-Responder: Live Edge Ingestion Active ===\n")

    print(f"\n📡 Connecting to Edge Camera Stream: {video_path}")
    print(f"📍 Location: {location}")
    print(f"⏳ Beginning continuous temporal monitoring (Pre-incident ➔ Crash Event ➔ Dispatch)...\n")

    for event in orchestrator.stream_incident(video_path, location_hint=location, speed_multiplier=speed):
        status = event.get("status", "NORMAL_MONITORING")
        ts = event.get("timestamp", "00:00")
        desc = event.get("scene_description", "")

        if status == "ANOMALY_DETECTED":
            prefix = f"\033[1;33m[{ts}] [⚠️ ANOMALY DETECTED]\033[0m"
            print(f"\n{prefix} {desc}")
        elif status == "CRISIS_IMPACT":
            prefix = f"\033[1;31m[{ts}] [🚨 CRISIS IMPACT DETECTED!]\033[0m"
            print(f"\n{prefix} {desc}")
            print("\033[1;36m⚡ DGX Spark Nemotron Orchestrator: Triggering sub-agents in parallel (< 150ms latency)...\033[0m")
            record = event.get("incident_record")
            if record:
                print_rich_incident_report(record)
        elif status == "ROADSIDE_ASSISTANCE":
            prefix = f"\033[1;33m[{ts}] [🟡 ROADSIDE SAFETY ADVISORY ACTIVE]\033[0m"
            print(f"\n{prefix} {desc}")
            print("\033[1;33m⚡ DGX Spark Nemotron Orchestrator: Roadway safety advisory triggered.\033[0m")
            record = event.get("incident_record")
            if record:
                print_rich_incident_report(record)
        elif status == "ROUTINE_ALL_CLEAR":
            prefix = f"\033[1;32m[{ts}] [🟢 ALL CLEAR / ROUTINE MONITORING]\033[0m"
            print(f"\n{prefix} {desc}")
            print("\033[1;32m✔ Scene verified normal. No emergency response required.\033[0m")
            record = event.get("incident_record")
            if record:
                print_rich_incident_report(record)
        else:
            prefix = f"[{ts}] \033[1;32m[EDGE MONITOR: NORMAL]\033[0m"
            print(f"{prefix} {desc}")

def main():
    parser = argparse.ArgumentParser(description="Omni-Responder: Autonomous Emergency Dispatch on NVIDIA DGX Spark")
    parser.add_argument("--video", type=str, default="data/video_clips/crash_scenario_1.mp4", help="Path to input video file")
    parser.add_argument("--location", type=str, default="5th Ave & Market St Intersection", help="Incident location hint")
    parser.add_argument("--stream", action="store_true", help="Run temporal live stream monitoring simulation")
    parser.add_argument("--speed", type=float, default=1.0, help="Stream playback speed multiplier (default: 1.0)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON record only")
    args = parser.parse_args()

    if args.stream:
        run_live_stream_simulation(args.video, args.location, args.speed)
        return

    orchestrator = IncidentOrchestrator()
    record = orchestrator.process_incident(args.video, location_hint=args.location)

    if args.json:
        print(json.dumps(record, indent=2))
    else:
        print_rich_incident_report(record)

if __name__ == "__main__":
    main()
