import datetime
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo

TELEGRAM_ENV_FILE = "config/telegram.env"
DISPLAY_TZ = ZoneInfo("America/Los_Angeles")


def _dispatch_tag(record: Dict[str, Any]) -> str:
    """One-line fingerprint so repeated demo runs are distinguishable at a glance:
    short unique id, wall-clock PDT time (correct even on a UTC-clocked Spark),
    and the source video that produced the incident."""
    short_id = uuid.uuid4().hex[:6]
    stamp = datetime.datetime.now(DISPLAY_TZ).strftime("%b %d %I:%M:%S %p %Z")
    video = record.get("source_video") or "unknown-source"
    return f"#{short_id} · {stamp} · {video}"


def _load_env_file(path: str = TELEGRAM_ENV_FILE) -> Dict[str, str]:
    """Parses a simple KEY=VALUE env file (untracked; holds the bot token locally)."""
    values: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    values[key.strip()] = value.strip()
    return values


class TelegramNotifier:
    """Simulates the call to the needed emergency resource via a Telegram channel.

    In production this would page real CAD/EMS systems; here the dispatch brief is
    delivered to a Telegram group so responders (the demo audience) see the callout.
    Delivery is best-effort: a Telegram failure must never break the pipeline.
    """

    API_BASE = "https://api.telegram.org"

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        file_env = _load_env_file()
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or file_env.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID") or file_env.get("TELEGRAM_CHAT_ID", "")

    @property
    def enabled(self) -> bool:
        # TELEGRAM_DISABLED=1 is the kill-switch (set by tests and --no-telegram runs)
        if os.getenv("TELEGRAM_DISABLED", "").lower() in ("1", "true", "yes"):
            return False
        return bool(self.bot_token and self.chat_id)

    def format_message(self, record: Dict[str, Any]) -> str:
        """Renders the incident record as a plain-text Telegram dispatch brief."""
        perception = record.get("perception", {})
        hazmat = record.get("hazmat", {})
        traffic = record.get("traffic", {})
        dispatch = record.get("dispatch_report", {})

        code = dispatch.get("dispatch_code", "ROUTINE - CODE GREEN").upper()
        location = perception.get("location", "Unknown")
        camera = perception.get("camera_id", "N/A")

        tag = _dispatch_tag(record)

        if "GREEN" in code:
            return (
                f"🟢 ALL CLEAR — {location} ({camera})\n"
                f"{tag}\n"
                f"Routine monitoring. No emergency dispatch required.\n"
                f"Generated on-device — NVIDIA DGX Spark. No raw video egress."
            )

        icon = "🚨" if "RED" in code else "⚠️"
        hazards = ", ".join(perception.get("hazard_indicators", [])) or "None observed"
        units = ", ".join(dispatch.get("target_units", [])) or "None"

        lines = [
            f"{icon} CAD DISPATCH BRIEF — {code}",
            tag,
            "",
            f"Incident ID: {dispatch.get('cad_id', 'N/A')}",
            f"Location: {location}",
            f"Camera: {camera}",
            "",
            f"Scene: {perception.get('crisis_type', 'N/A')} (Severity: {perception.get('severity', 'N/A')})",
            f"Vehicles involved: {perception.get('vehicles_involved', 0)}",
            f"Hazards: {hazards}",
        ]

        if hazmat.get("status") not in (None, "NO_HAZARDS_DETECTED"):
            lines += [
                "",
                f"Hazmat: {hazmat.get('chemical_name')} ({hazmat.get('un_number')})",
                f"Protective Action Zone: {hazmat.get('isolation_radius_meters')}m standoff | PPE: {hazmat.get('ppe_required')}",
            ]

        lines += [
            f"Traffic: {traffic.get('status')} via {traffic.get('closure_id')}",
            "",
            f"📞 Simulated call to: {units}",
            "",
            "Generated on-device — NVIDIA DGX Spark. No raw video egress.",
        ]
        return "\n".join(lines)

    def notify_dispatch(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Sends the dispatch brief. Returns a status dict for the incident record."""
        if not self.enabled:
            return {
                "status": "DISABLED",
                "detail": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not configured",
            }

        text = self.format_message(record)
        # First line of the tag block is what the phone shows; echo it to the terminal
        tag_line = next((l for l in text.splitlines() if l.startswith("#") or " · " in l), "")
        payload = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text}).encode("utf-8")
        url = f"{self.API_BASE}/bot{self.bot_token}/sendMessage"

        try:
            req = urllib.request.Request(url, data=payload)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok"):
                return {
                    "status": "SENT",
                    "chat_id": self.chat_id,
                    "message_id": data.get("result", {}).get("message_id"),
                    "tag": tag_line,
                }
            return {"status": "ERROR", "detail": str(data.get("description", "unknown"))[:200]}
        except urllib.error.HTTPError as e:
            # e.g. 400 bad chat_id, 401 bad token. Never echo the URL (contains the token).
            return {"status": "ERROR", "detail": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"status": "ERROR", "detail": type(e).__name__}
