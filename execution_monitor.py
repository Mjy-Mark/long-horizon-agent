"""Side-effect checks for the single email action allowed by Improved 1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from planning_memory import TaskMemory


def make_send_fingerprint(
    event_id: str,
    recipient: str,
    subject: str,
    body: str,
) -> str:
    """Return a stable fingerprint for one event-specific email."""

    payload = json.dumps(
        {
            "event_id": str(event_id),
            "recipient": recipient.strip().lower(),
            "subject": subject,
            "body": body,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MonitorViolation(RuntimeError):
    """Raised when a side-effecting action fails a pre-send check."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class MonitorDecision:
    allowed: bool
    reason: str
    fingerprint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
        }


class ExecutionMonitor:
    """Allow at most one validated send per task.

    Fingerprints are reserved before the underlying send function runs.  This
    keeps a second call from being accepted after a transport exception, which
    is the ambiguity that Improved 2 will handle explicitly.
    """

    def __init__(self, existing_fingerprints: Iterable[str] = ()) -> None:
        self._fingerprints = set(existing_fingerprints)
        self._reserved_fingerprints: set[str] = set()
        self._send_count = 0
        self.checks: list[dict[str, Any]] = []

    @property
    def send_count(self) -> int:
        return self._send_count

    def _record(self, decision: MonitorDecision) -> MonitorDecision:
        self.checks.append(decision.as_dict())
        return decision

    def _block(self, reason: str, fingerprint: str | None = None) -> MonitorDecision:
        return self._record(MonitorDecision(False, reason, fingerprint))

    def check_send(
        self,
        memory: TaskMemory,
        recipient: str,
        subject: str,
        body: str,
    ) -> MonitorDecision:
        """Validate and reserve one email send.

        A failed check never calls the supplied send function.  A successful
        check reserves its fingerprint immediately and increments the send
        count before the side effect starts.
        """

        target_event = memory.target_event
        if target_event is None:
            decision = self._block("target_event_missing")
            raise MonitorViolation(decision.reason)
        if memory.prepared_email is None:
            decision = self._block("email_not_prepared")
            raise MonitorViolation(decision.reason)
        if memory.send_confirmed:
            decision = self._block("send_already_confirmed")
            raise MonitorViolation(decision.reason)
        if memory.send_attempted:
            decision = self._block("send_already_attempted")
            raise MonitorViolation(decision.reason)

        fingerprint = make_send_fingerprint(
            target_event["event_id"], recipient, subject, body
        )
        prepared = memory.prepared_email
        if recipient.strip().lower() != prepared["recipient"].strip().lower():
            decision = self._block("recipient_mismatch", fingerprint)
            raise MonitorViolation(decision.reason)
        if subject != prepared["subject"]:
            decision = self._block("subject_mismatch", fingerprint)
            raise MonitorViolation(decision.reason)
        if body != prepared["body"]:
            decision = self._block("body_mismatch", fingerprint)
            raise MonitorViolation(decision.reason)
        if self._send_count >= 1:
            decision = self._block("multiple_send_attempts", fingerprint)
            raise MonitorViolation(decision.reason)
        if fingerprint in self._fingerprints or fingerprint in self._reserved_fingerprints:
            decision = self._block("duplicate_send_fingerprint", fingerprint)
            raise MonitorViolation(decision.reason)

        self._reserved_fingerprints.add(fingerprint)
        self._send_count += 1
        return self._record(MonitorDecision(True, "allowed", fingerprint))

    def guarded_send_email(
        self,
        memory: TaskMemory,
        send_fn: Callable[[str, str, str], Any],
        recipient: str,
        subject: str,
        body: str,
    ) -> Any:
        """Run the pre-send check, mark the attempt, and invoke ``send_fn``."""

        self.check_send(memory, recipient, subject, body)
        memory.mark_send_attempted()
        return send_fn(recipient, subject, body)

    def snapshot(self) -> dict[str, Any]:
        return {
            "send_count": self._send_count,
            "checks": list(self.checks),
            "reserved_fingerprints": sorted(self._reserved_fingerprints),
            "known_fingerprints": sorted(self._fingerprints),
        }
