"""Improved 1 controller for the local WorkBench subset.

The language model proposes only a small, validated plan.  Python owns event
selection, email construction, the pre-send monitor, and final-state checks.
The raw ``send_email`` tool is therefore never exposed to the model.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from smolagents import OpenAIModel

from execution_monitor import (
    ExecutionMonitor,
    MonitorViolation,
    make_send_fingerprint,
)
from planning_memory import TaskMemory, TaskMemoryError, select_first_event


BENCHMARK_ROOT = Path(__file__).resolve().parent / "benchmarks" / "workbench_subset"
DEFAULT_EMAIL_BODY = "Remember to attend this event."
PLAN_FIELDS = {
    "target_date",
    "operation",
    "event_selection",
    "subject_source",
}
PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "Improved1Plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_date": {"type": "string"},
                "operation": {"type": "string", "enum": ["send_email"]},
                "event_selection": {
                    "type": "string",
                    "enum": ["earliest_start"],
                },
                "subject_source": {
                    "type": "string",
                    "enum": ["event_name"],
                },
            },
            "required": [
                "target_date",
                "operation",
                "event_selection",
                "subject_source",
            ],
        },
    },
}


class PlanError(ValueError):
    """Raised when a model plan cannot be used safely."""


def _content_from_model_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str) or not content.strip():
        raise PlanError("模型没有返回结构化计划文本")
    return content


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_plan(raw_plan: str | dict[str, Any]) -> dict[str, str]:
    """Parse and validate the exact plan contract used by Improved 1."""

    if isinstance(raw_plan, dict):
        plan = raw_plan
    elif isinstance(raw_plan, str):
        try:
            plan = json.loads(_strip_json_fence(raw_plan))
        except json.JSONDecodeError as exc:
            raise PlanError("计划不是合法 JSON") from exc
    else:
        raise PlanError("计划必须是 JSON 对象或 JSON 字符串")

    if not isinstance(plan, dict):
        raise PlanError("计划顶层结构必须是对象")
    if set(plan) != PLAN_FIELDS:
        missing = sorted(PLAN_FIELDS - set(plan))
        extra = sorted(set(plan) - PLAN_FIELDS)
        details = []
        if missing:
            details.append(f"缺少字段 {missing}")
        if extra:
            details.append(f"存在额外字段 {extra}")
        raise PlanError("计划字段不符合约定，" + "，".join(details))

    target_date = plan["target_date"]
    if not isinstance(target_date, str):
        raise PlanError("target_date 必须是字符串")
    try:
        date.fromisoformat(target_date)
    except ValueError as exc:
        raise PlanError(f"target_date 格式错误: {target_date}") from exc
    if plan["operation"] != "send_email":
        raise PlanError("只允许 send_email 操作")
    if plan["event_selection"] != "earliest_start":
        raise PlanError("只允许 earliest_start 事件选择规则")
    if plan["subject_source"] != "event_name":
        raise PlanError("只允许从 event_name 生成标题")

    return {field: str(plan[field]) for field in PLAN_FIELDS}


def generate_plan(model: Any, task: str) -> dict[str, str]:
    """Ask the model for a plan without giving it any side-effecting tools."""

    messages = [
        {
            "role": "system",
            "content": (
                "你负责把任务转换为严格的 JSON 计划。"
                "只输出 target_date、operation、event_selection、subject_source 四个字段。"
                "operation 必须为 send_email，event_selection 必须为 earliest_start，"
                "subject_source 必须为 event_name。这个 WorkBench 子集的任务日期按 2023 年解释。"
                "不要填写 event_id、收件人或发送动作。"
            ),
        },
        {"role": "user", "content": task},
    ]
    response = model.generate(messages, response_format=PLAN_RESPONSE_FORMAT)
    return parse_plan(_content_from_model_response(response))


def extract_email_body(task: str) -> str:
    """Extract the quoted reminder body from the benchmark task text."""

    patterns = (
        r"(?:tell them|message them|say)\s+['\"]([^'\"]+)['\"]",
        r"(?:body|message)\s*[:：]\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        match = re.search(pattern, task, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    if DEFAULT_EMAIL_BODY in task:
        return DEFAULT_EMAIL_BODY
    raise PlanError("任务中没有找到带引号的邮件正文")


def _email_value(row: dict[str, Any], field: str) -> Any:
    aliases = {
        "recipient": ("recipient", "sender/recipient"),
        "subject": ("subject",),
        "body": ("body",),
        "email_id": ("email_id", "id"),
    }
    for key in aliases[field]:
        if key in row:
            return row[key]
    return None


def _new_email_rows(
    initial_rows: Iterable[dict[str, Any]],
    final_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    initial = [deepcopy(row) for row in initial_rows]
    final = [deepcopy(row) for row in final_rows]
    initial_ids = {
        str(_email_value(row, "email_id"))
        for row in initial
        if _email_value(row, "email_id") is not None
    }
    if initial_ids or any(_email_value(row, "email_id") is not None for row in final):
        return [
            row
            for row in final
            if _email_value(row, "email_id") is None
            or str(_email_value(row, "email_id")) not in initial_ids
        ]

    remaining = initial.copy()
    new_rows: list[dict[str, Any]] = []
    for row in final:
        try:
            remaining.remove(row)
        except ValueError:
            new_rows.append(row)
    return new_rows


class WorkbenchPlannerController:
    """Run the deterministic Improved 1 pipeline around injected tools."""

    def __init__(
        self,
        *,
        task_id: str,
        task: str,
        search_events_fn: Callable[..., str],
        send_email_fn: Callable[[str, str, str], Any],
        list_sent_emails_fn: Callable[[], Iterable[dict[str, Any]]],
        state_snapshot_fn: Callable[[], dict[str, Any]] | None = None,
        model: Any | None = None,
        trace_path: Path | None = None,
        max_search_retries: int = 1,
    ) -> None:
        self.task_id = task_id
        self.task = task
        self.search_events_fn = search_events_fn
        self.send_email_fn = send_email_fn
        self.list_sent_emails_fn = list_sent_emails_fn
        self.state_snapshot_fn = state_snapshot_fn or (lambda: {})
        self.model = model
        self.trace_path = trace_path
        if max_search_retries < 0:
            raise ValueError("max_search_retries 不能为负数")
        self.max_search_retries = max_search_retries
        self.memory = TaskMemory(task_id=task_id)
        self.monitor: ExecutionMonitor | None = None
        self._initial_sent_emails = [
            deepcopy(row) for row in self.list_sent_emails_fn()
        ]
        self.trace: dict[str, Any] = {
            "benchmark": "WorkBench",
            "method": "improved_1",
            "task_id": task_id,
            "task": task,
            "plan": None,
            "tool_calls": [],
            "memory_snapshots": [],
            "monitor_checks": [],
            "candidate_selection_scope": "returned_candidates_only",
            "candidate_completeness": "not_guaranteed",
            "initial_state": deepcopy(self.state_snapshot_fn()),
            "initial_sent_emails": deepcopy(self._initial_sent_emails),
            "success": False,
            "termination_reason": "not_started",
        }

    def _snapshot_memory(self, label: str) -> None:
        self.trace["memory_snapshots"].append(
            {"label": label, "state": self.memory.snapshot()}
        )

    def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        fn: Callable[..., Any],
    ) -> Any:
        entry: dict[str, Any] = {"tool": name, "arguments": deepcopy(arguments)}
        try:
            result = fn(**arguments)
        except Exception as exc:
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.trace["tool_calls"].append(entry)
            raise
        entry["result"] = deepcopy(result)
        self.trace["tool_calls"].append(entry)
        return result

    def _finish(self, success: bool, reason: str, answer: str | None = None) -> dict[str, Any]:
        self.trace["success"] = success
        self.trace["agent_state"] = "success" if success else "failed"
        self.trace["termination_reason"] = reason
        if answer is not None:
            self.trace["final_answer"] = answer
        return self.trace

    def _existing_fingerprints(self, event: dict[str, Any], body: str) -> set[str]:
        fingerprints: set[str] = set()
        expected_recipient = event["participant_email"].strip().lower()
        for row in self._initial_sent_emails:
            recipient = _email_value(row, "recipient")
            subject = _email_value(row, "subject")
            email_body = _email_value(row, "body")
            if (
                isinstance(recipient, str)
                and recipient.strip().lower() == expected_recipient
                and subject == event["event_name"]
                and email_body == body
            ):
                fingerprints.add(
                    make_send_fingerprint(
                        event["event_id"], expected_recipient, subject, email_body
                    )
                )
        return fingerprints

    def _verify_final_email(
        self,
        prepared_email: dict[str, str],
        new_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if len(new_rows) != 1:
            return None
        row = new_rows[0]
        actual = {
            "recipient": _email_value(row, "recipient"),
            "subject": _email_value(row, "subject"),
            "body": _email_value(row, "body"),
        }
        if actual != prepared_email:
            return None
        return row

    def run(self, plan: str | dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            if plan is None:
                if self.model is None:
                    raise PlanError("没有提供计划或模型")
                parsed_plan = generate_plan(self.model, self.task)
            else:
                parsed_plan = parse_plan(plan)
            self.trace["plan"] = parsed_plan

            self.memory.set_target_date(parsed_plan["target_date"])
            query_args = {
                "query": "",
                "time_min": f"{parsed_plan['target_date']} 00:00:00",
                "time_max": f"{parsed_plan['target_date']} 23:59:59",
            }
            events_json = None
            for attempt in range(self.max_search_retries + 1):
                try:
                    events_json = self._call_tool(
                        "calendar.search_events", query_args, self.search_events_fn
                    )
                    break
                except Exception:
                    if attempt >= self.max_search_retries:
                        raise
            if events_json is None:
                raise ValueError("search_events 没有返回结果")
            if not isinstance(events_json, str):
                events_json = json.dumps(events_json, ensure_ascii=False)
            self.memory.mark_events_queried(events_json)
            self._snapshot_memory("events_queried")

            event = select_first_event(events_json, parsed_plan["target_date"])
            self.memory.set_target_event(event)
            self._snapshot_memory("event_selected")

            body = extract_email_body(self.task)
            self.memory.prepare_email(
                recipient=event["participant_email"],
                subject=event["event_name"],
                body=body,
            )
            self._snapshot_memory("email_prepared")

            self.monitor = ExecutionMonitor(self._existing_fingerprints(event, body))
            prepared_email = deepcopy(self.memory.prepared_email)
            if prepared_email is None:
                raise TaskMemoryError("邮件准备状态丢失")
            self._call_tool(
                "email.send_email",
                prepared_email,
                lambda **arguments: self.monitor.guarded_send_email(
                    self.memory,
                    self.send_email_fn,
                    **arguments,
                ),
            )
            self._snapshot_memory("send_attempted")

            final_rows = [deepcopy(row) for row in self.list_sent_emails_fn()]
            self.trace["final_sent_emails"] = deepcopy(final_rows)
            new_rows = _new_email_rows(self._initial_sent_emails, final_rows)
            self.trace["new_emails"] = deepcopy(new_rows)
            sent_row = self._verify_final_email(prepared_email, new_rows)
            if sent_row is None:
                self.memory.mark_failed(
                    {"reason": "final_state_email_mismatch", "new_emails": new_rows}
                )
                self._snapshot_memory("final_state_mismatch")
                return self._finish(False, "final_state_email_mismatch")

            self.memory.mark_send_confirmed({"email": sent_row})
            self._snapshot_memory("send_confirmed")
            answer = (
                f"Email sent to {prepared_email['recipient']} with the subject "
                f"'{prepared_email['subject']}' and the message '{prepared_email['body']}'."
            )
            return self._finish(True, "success", answer)
        except MonitorViolation as exc:
            self.memory.mark_failed({"type": type(exc).__name__, "reason": exc.reason})
            self._snapshot_memory("monitor_blocked")
            return self._finish(False, f"monitor_blocked:{exc.reason}")
        except TaskMemoryError as exc:
            self.memory.mark_failed({"type": type(exc).__name__, "message": str(exc)})
            self._snapshot_memory("memory_error")
            return self._finish(False, "memory_error")
        except (PlanError, ValueError) as exc:
            self.memory.mark_failed({"type": type(exc).__name__, "message": str(exc)})
            self._snapshot_memory("failed")
            return self._finish(False, "validation_error")
        except Exception as exc:
            self.memory.mark_failed({"type": type(exc).__name__, "message": str(exc)})
            self._snapshot_memory("exception")
            return self._finish(False, "execution_exception")
        finally:
            self.trace["monitor_checks"] = (
                [] if self.monitor is None else list(self.monitor.checks)
            )
            try:
                self.trace["final_state"] = deepcopy(self.state_snapshot_fn())
            except Exception as exc:
                self.trace["final_state_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            if self.trace_path is not None:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                self.trace_path.write_text(
                    json.dumps(self.trace, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )


def main() -> None:
    from importlib.metadata import version

    from workbench_subset_agent import WorkbenchSubsetState

    task_record = json.loads(
        (BENCHMARK_ROOT / "task.json").read_text(encoding="utf-8")
    )
    state = WorkbenchSubsetState(BENCHMARK_ROOT)

    def search_events_fn(**arguments: Any) -> str:
        result = state.search_events(**arguments)
        print(f"[Improved 1 工具] calendar.search_events -> {result}", flush=True)
        return result

    def send_email_fn(recipient: str, subject: str, body: str) -> str:
        result = state.send_email(recipient, subject, body)
        print(
            f"[Improved 1 工具] email.send_email(recipient={recipient!r}, "
            f"subject={subject!r}, body={body!r}) -> {result}",
            flush=True,
        )
        return result

    def list_sent_emails_fn() -> list[dict[str, Any]]:
        return [row.copy() for row in state.emails]

    model = OpenAIModel(
        model_id="longagent-qwen",
        api_base="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0,
        max_tokens=256,
    )
    runs_dir = Path(__file__).resolve().parent / "runs"
    trace_path = runs_dir / f"workbench-improved1-{uuid4().hex}.json"
    controller = WorkbenchPlannerController(
        task_id=task_record["task_id"],
        task=task_record["task"],
        search_events_fn=search_events_fn,
        send_email_fn=send_email_fn,
        list_sent_emails_fn=list_sent_emails_fn,
        state_snapshot_fn=state.snapshot,
        model=model,
        trace_path=trace_path,
    )
    controller.trace.update(
        {
            "source_repo": task_record["source_repo"],
            "source_commit": task_record["source_commit"],
            "model_id": model.model_id,
            "smolagents_version": version("smolagents"),
        }
    )
    trace = controller.run()
    print("[轨迹文件]", trace_path)
    print("[Improved 1 结束状态]", trace["agent_state"])
    print("最终回答：", trace.get("final_answer", "任务未完成"))
    print("[WorkBench 终态]", trace.get("final_state"))


if __name__ == "__main__":
    main()
