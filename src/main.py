import sys
import os
import json
from src.orchestrator.incident_manager import IncidentOrchestrator

try:
    from rich.console import Console
    from rich.panel import Panel
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

def run_simulation():
    title_banner = (
        "Omni-Responder: DGX Spark 🚨⚡\n"
        "Edge Vision Perception + Autonomous Multi-Agent Incident Dispatch\n"
        "Target Hardware: NVIDIA Grace Blackwell DGX Spark (128GB Unified Memory)"
    )
    print_box("System Initialization", title_banner, "36")

    print("\n📡 Ingesting video stream from data/video_clips/crash_scenario_1.mp4 (100% On-Premise / 0% Cloud Egress)...")
    
    orchestrator = IncidentOrchestrator()
    # Process crash video through perception + sub-agents
    result = orchestrator.process_incident("data/video_clips/crash_scenario_1.mp4", location_hint="5th Ave & Market St Intersection")

    # 1. Perception Output (Cosmos Reasoner VLM)
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

    # 2. Hazmat Agent Output (ERG 2024 Database)
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

    # 3. Traffic Agent Output
    t = result["traffic"]
    t_actions = "\n".join([f"  • {act}" for act in t["actions_triggered"]])
    t_text = (
        f"Dispatch Status: {t['status']}\n"
        f"Perimeter Closure ID: {t['closure_id']}\n"
        f"Automated Actions Dispatched:\n{t_actions}"
    )
    print_box("3. Traffic Sub-Agent (City VMS & Signal Priority)", t_text, "33")

    # 4. Comms Agent Output (911 CAD Dispatch Brief)
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
    run_simulation()
