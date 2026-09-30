"""
Mock MSG91 SMS stub.

Per PROJECT_SPEC.md: SMS is MOCK ONLY. This module never calls a real SMS API —
it only logs what would be sent, for the demo's "Send Alert" flow.
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger("sms_stub")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [SMS_STUB] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def send_alert_sms(officer: str, terminal_id: str, risk_score: float | None = None) -> dict:
    """
    Mock-sends an SMS alert to an officer about a high-risk terminal.
    Never calls a real SMS API (MSG91 or otherwise) — logs only.
    """
    message = f"would send SMS to {officer} about terminal {terminal_id}"
    if risk_score is not None:
        message += f" (risk_score={risk_score:.2f})"
    logger.info(message)

    return {
        "status": "mock_sent",
        "provider": "MSG91_MOCK",
        "officer": officer,
        "terminal_id": terminal_id,
        "risk_score": risk_score,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "detail": message,
    }
