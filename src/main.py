import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.orchestrator.incident_manager import IncidentOrchestrator

console = Console()

def run_simulation():
    console.print(Panel.fit(
        "[bold cyan]Omni-Responder: DGX Spark[/bold cyan]\n"
        "[dim]Edge Vision Perception + Autonomous Multi-Agent Crisis Response[/dim]\n"
        "[green]Hardware: NVIDIA Grace Blackwell DGX Spark (128GB Unified Memory)[/green]",
        border_style="cyan"
    ))

    console.print("\n[bold yellow]📡 Ingesting simulated traffic camera feed (Zero Cloud Egress)...[/bold yellow]")
    orchestrator = IncidentOrchestrator()
    result = orchestrator.process_incident(feed_id="feed_camera_402")

    # 1. Perception Output
    p = result["perception"]
    console.print(Panel(
        f"[bold white]Location:[/bold white] {p.location}\n"
        f"[bold white]Crisis Detected:[/bold white] {p.crisis_type}\n"
        f"[bold white]Vehicles Involved:[/bold white] {p.vehicles_involved}\n"
        f"[bold white]VLM Visual Description:[/bold white] {p.raw_summary}",
        title="[bold green]1. Perception (NVIDIA VSS + Local VLM)[/bold green]",
        border_style="green"
    ))

    # 2. Hazmat Agent Output
    h = result["hazmat"]
    console.print(Panel(
        f"[bold white]Status:[/bold white] {h.get('status')}\n"
        f"[bold white]Identified Chemical:[/bold white] {h.get('chemical_name')} ({h.get('un_number')})\n"
        f"[bold white]Hazard Class:[/bold white] {h.get('hazard_class')}\n"
        f"[bold white]Isolation Radius:[/bold white] {h.get('isolation_radius_meters')} meters\n"
        f"[bold white]Required PPE:[/bold white] {h.get('ppe_required')}\n"
        f"[bold white]Fire & Containment Plan:[/bold white] {h.get('recommended_action')}",
        title="[bold red]2. Hazmat Sub-Agent (Local Knowledge Base)[/bold red]",
        border_style="red"
    ))

    # 3. Traffic Agent Output
    t = result["traffic"]
    traffic_actions = "\n".join([f"• {act}" for act in t.get("actions_triggered", [])])
    console.print(Panel(
        f"[bold white]Closure ID:[/bold white] {t.get('closure_id')}\n"
        f"[bold white]Actions Dispatched:[/bold white]\n{traffic_actions}",
        title="[bold yellow]3. Traffic Sub-Agent (City System Mock API)[/bold yellow]",
        border_style="yellow"
    ))

    # 4. Emergency Dispatch Report
    d = result["dispatch_report"]
    console.print(Panel(
        f"[bold white]Priority Code:[/bold white] [bold red]{d.get('dispatch_code')}[/bold red]\n"
        f"[bold white]Dispatched Units:[/bold white] {', '.join(d.get('target_units', []))}\n\n"
        f"{d.get('briefing_summary')}",
        title="[bold magenta]4. Comms Sub-Agent (First Responder Synthesis)[/bold magenta]",
        border_style="magenta"
    ))

    console.print("\n[bold green]✔ All autonomous edge actions completed successfully on DGX Spark.[/bold green]\n")

if __name__ == "__main__":
    run_simulation()
