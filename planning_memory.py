"""Deterministic planning helpers and per-task execution memory.

The planner is responsible for selecting an event from a tool response.  The
task memory keeps the selected event and the email that the controller is
allowed to send so later checks do not have to rely on model output.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


VALID_STATUSES = {
    "created",
    "planned",
    "events_queried",
    "event_selected",
    "candidate_insufficient",
    "email_prepared",
    "send_attempted",
    "send_confirmed",
    "completed_without_action",
    "blocked",
    "uncertain",
    "failed",
}


class TaskMemoryError(ValueError):
    """Raised when a task state transition violates the task contract."""


def _copy_observation(value: Any) -> Any:
    copied = deepcopy(value)
    try:
        json.dumps(copied, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(copied)
    return copied


def _parse_event_start(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("event_start 必须是字符串")

    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"无法解析 event_start: {value}") from exc


def select_first_event(events_json: str, target_date: str) -> dict[str, Any]:
    """Select the earliest event that starts on ``target_date``.

    The input order is deliberately ignored.  Event IDs provide a stable
    tie-breaker when two events have the same start time.
    """

    try:
        requested_date = date.fromisoformat(target_date)
    except ValueError as exc:
        raise ValueError(
            f"target_date 格式错误，应为 YYYY-MM-DD: {target_date}"
        ) from exc

    try:
        events = json.loads(events_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("events_json 不是合法 JSON") from exc

    if not isinstance(events, list):
        raise ValueError("events_json 顶层结构必须是列表")

    candidates: list[tuple[datetime, dict[str, Any]]] = []
    for event in events:
        if not isinstance(event, dict) or "event_start" not in event:
            continue

        try:
            event_time = _parse_event_start(event["event_start"])
        except ValueError:
            continue

        if event_time.date() == requested_date:
            candidates.append((event_time, event))

    if not candidates:
        raise ValueError("目标日期没有匹配的活动")

    try:
        event_time, first_event = min(
            candidates,
            key=lambda item: (
                item[0],
                str(item[1].get("event_id", "")),
            ),
        )
    except TypeError as exc:
        raise ValueError("候选活动的 event_start 时区格式不一致") from exc

    required_fields = ("event_id", "event_name", "participant_email")
    missing_fields = [field for field in required_fields if field not in first_event]
    if missing_fields:
        raise ValueError(f"目标活动缺少字段: {missing_fields}")

    return {
        "event_id": first_event["event_id"],
        "event_name": first_event["event_name"],
        "participant_email": first_event["participant_email"],
        "event_start": event_time.isoformat(),
    }


@dataclass
class TaskMemory:
    """Serializable state for one Improved 1 task."""

    task_id: str = "task"
    target_date: str | None = None
    target_event: dict[str, Any] | None = None
    prepared_email: dict[str, str] | None = None
    status: str = "created"
    send_attempted: bool = False
    send_confirmed: bool = False
    last_observation: Any = None
    planner_output: dict[str, Any] | None = None
    subgoals: list[str] | None = None
    recent_candidates: list[dict[str, Any]] | None = None
    returned_count: int = 0
    candidate_completeness: str | None = None
    completeness_reason: str | None = None
    memory_transitions: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise TaskMemoryError("task_id 不能为空")
        if self.target_date is not None:
            self.set_target_date(self.target_date)
        if self.status not in VALID_STATUSES:
            raise TaskMemoryError(f"未知状态: {self.status}")
        if self.memory_transitions is None:
            self.memory_transitions = []

    def record_transition(self, label: str, details: Any = None) -> None:
        if self.memory_transitions is None:
            self.memory_transitions = []
        self.memory_transitions.append(
            {
                "label": label,
                "status": self.status,
                "details": _copy_observation(details),
            }
        )

    def set_plan(self, plan: dict[str, Any]) -> None:
        if not isinstance(plan, dict):
            raise TaskMemoryError("planner_output 必须是对象")
        target_date = plan.get("target_date")
        if not isinstance(target_date, str):
            raise TaskMemoryError("计划缺少 target_date")
        self.set_target_date(target_date)
        subgoals = plan.get("subgoals", [])
        if not isinstance(subgoals, list) or not all(
            isinstance(item, str) for item in subgoals
        ):
            raise TaskMemoryError("计划的 subgoals 必须是字符串列表")
        self.planner_output = deepcopy(plan)
        self.subgoals = list(subgoals)
        self.status = "planned"
        self.last_observation = None
        self.record_transition("plan_saved", {"target_date": target_date})

    def set_target_date(self, target_date: str) -> None:
        try:
            date.fromisoformat(target_date)
        except (TypeError, ValueError) as exc:
            raise TaskMemoryError(
                f"target_date 格式错误，应为 YYYY-MM-DD: {target_date}"
            ) from exc
        self.target_date = target_date

    def mark_events_queried(self, observation: Any = None) -> None:
        if self.send_attempted:
            raise TaskMemoryError("发送已经开始，不能回到查询状态")
        self.status = "events_queried"
        self.last_observation = _copy_observation(observation)
        self.record_transition("events_queried")

    def record_search_result(
        self,
        candidates: list[dict[str, Any]],
        candidate_completeness: str,
        completeness_reason: str,
    ) -> None:
        if not isinstance(candidates, list):
            raise TaskMemoryError("候选活动必须是列表")
        self.recent_candidates = deepcopy(candidates)
        self.returned_count = len(candidates)
        self.candidate_completeness = candidate_completeness
        self.completeness_reason = completeness_reason
        self.last_observation = {
            "returned_count": len(candidates),
            "candidate_completeness": candidate_completeness,
            "completeness_reason": completeness_reason,
        }
        if candidate_completeness == "complete":
            self.status = "events_queried"
        else:
            self.status = "candidate_insufficient"
        self.record_transition("search_result_saved", self.last_observation)

    def mark_candidate_blocked(self, reason: str) -> None:
        self.status = "blocked"
        self.last_observation = {"reason": reason}
        self.record_transition("candidate_blocked", {"reason": reason})

    def mark_completed_without_action(self, reason: str) -> None:
        self.status = "completed_without_action"
        self.last_observation = {"reason": reason}
        self.record_transition("completed_without_action", {"reason": reason})

    def set_target_event(self, event: dict[str, Any]) -> None:
        if self.send_attempted:
            raise TaskMemoryError("发送已经开始，不能更新目标活动")
        if not isinstance(event, dict):
            raise TaskMemoryError("target_event 必须是对象")

        required_fields = ("event_id", "event_name", "participant_email")
        missing_fields = [field for field in required_fields if field not in event]
        if missing_fields:
            raise TaskMemoryError(f"目标活动缺少字段: {missing_fields}")
        if any(
            not isinstance(event[field], str) or not event[field].strip()
            for field in required_fields
        ):
            raise TaskMemoryError("目标活动的 ID、名称和参与人必须是非空字符串")

        self.target_event = deepcopy(event)
        self.prepared_email = None
        self.send_confirmed = False
        self.status = "event_selected"
        self.last_observation = None
        self.record_transition("target_event_selected", {"event_id": event["event_id"]})

    def prepare_email(self, recipient: str, subject: str, body: str) -> None:
        if self.target_event is None:
            raise TaskMemoryError("没有目标活动，不能准备邮件")
        if self.send_attempted:
            raise TaskMemoryError("发送已经开始，不能重新准备邮件")
        if self.prepared_email is not None:
            raise TaskMemoryError("该任务已经准备过邮件")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (recipient, subject, body)
        ):
            raise TaskMemoryError("收件人、标题和正文都必须是非空字符串")

        expected_recipient = self.target_event["participant_email"].strip().lower()
        normalized_recipient = recipient.strip().lower()
        if normalized_recipient != expected_recipient:
            raise TaskMemoryError("收件人必须与目标活动参与人一致")
        if subject != self.target_event["event_name"]:
            raise TaskMemoryError("邮件标题必须等于目标活动名称")

        self.prepared_email = {
            "recipient": normalized_recipient,
            "subject": subject,
            "body": body,
        }
        self.status = "email_prepared"
        self.last_observation = None
        self.record_transition("email_prepared", self.prepared_email)

    def mark_send_attempted(self) -> None:
        if self.prepared_email is None:
            raise TaskMemoryError("没有已准备的邮件")
        if self.send_attempted:
            raise TaskMemoryError("该任务已经尝试发送")

        self.send_attempted = True
        self.status = "send_attempted"
        self.record_transition("send_attempted")

    def mark_send_confirmed(self, observation: Any = None) -> None:
        if not self.send_attempted:
            raise TaskMemoryError("发送尚未尝试")

        self.send_confirmed = True
        self.status = "send_confirmed"
        self.last_observation = _copy_observation(observation)
        self.record_transition("send_confirmed", observation)

    def mark_uncertain(self, observation: Any) -> None:
        if not self.send_attempted:
            raise TaskMemoryError("发送尚未尝试，不能标记为 uncertain")
        self.status = "uncertain"
        self.last_observation = _copy_observation(observation)
        self.record_transition("send_uncertain", observation)

    def mark_failed(self, observation: Any) -> None:
        self.status = "failed"
        self.last_observation = _copy_observation(observation)
        self.record_transition("failed", observation)

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "task_id": self.task_id,
            "target_date": self.target_date,
            "target_event": deepcopy(self.target_event),
            "prepared_email": deepcopy(self.prepared_email),
            "status": self.status,
            "send_attempted": self.send_attempted,
            "send_confirmed": self.send_confirmed,
            "last_observation": deepcopy(self.last_observation),
            "planner_output": deepcopy(self.planner_output),
            "subgoals": deepcopy(self.subgoals),
            "recent_candidates": deepcopy(self.recent_candidates),
            "returned_count": self.returned_count,
            "candidate_completeness": self.candidate_completeness,
            "completeness_reason": self.completeness_reason,
            "memory_transitions": deepcopy(self.memory_transitions),
        }
        json.dumps(snapshot, ensure_ascii=False)
        return snapshot
