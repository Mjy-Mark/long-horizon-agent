"""Planner-guided ReAct controller for the next Improved 1 iteration.

The planner proposes constraints, then a ToolCallingAgent chooses actions after
each tool observation.  Python validates event selection and guards the real
email side effect through the existing ExecutionMonitor.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from smolagents import OpenAIModel, ToolCallingAgent, tool

from execution_monitor import ExecutionMonitor, MonitorViolation, make_send_fingerprint
from planning_memory import TaskMemory, TaskMemoryError, select_first_event


BENCHMARK_ROOT = Path(__file__).resolve().parent / "benchmarks" / "workbench_subset"
DEFAULT_EMAIL_BODY = "Remember to attend this event."
REACT_PLAN_FIELDS = {
    "target_date",
    "event_selection",
    "subject_source",
    "email_body",
    "subgoals",
}
REACT_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "Improved1ReactPlan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_date": {"type": "string"},
                "event_selection": {
                    "type": "string",
                    "enum": ["earliest_start"],
                },
                "subject_source": {
                    "type": "string",
                    "enum": ["event_name"],
                },
                "email_body": {"type": "string"},
                "subgoals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": [
                "target_date",
                "event_selection",
                "subject_source",
                "email_body",
                "subgoals",
            ],
        },
    },
}


class ReactPlanError(ValueError):
    """Raised when the planner output cannot safely drive ReAct."""


def parse_react_plan(raw_plan: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_plan, dict):
        plan = raw_plan
    elif isinstance(raw_plan, str):
        try:
            plan = json.loads(raw_plan.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError as exc:
            raise ReactPlanError("Planner 输出不是合法 JSON") from exc
    else:
        raise ReactPlanError("Planner 输出必须是 JSON 对象或字符串")

    if not isinstance(plan, dict):
        raise ReactPlanError("Planner 输出顶层结构必须是对象")
    if set(plan) != REACT_PLAN_FIELDS:
        missing = sorted(REACT_PLAN_FIELDS - set(plan))
        extra = sorted(set(plan) - REACT_PLAN_FIELDS)
        raise ReactPlanError(f"Planner 字段不符合约定，缺少 {missing}，额外字段 {extra}")

    target_date = plan["target_date"]
    try:
        date.fromisoformat(target_date)
    except (TypeError, ValueError) as exc:
        raise ReactPlanError("target_date 格式错误") from exc
    if plan["event_selection"] != "earliest_start":
        raise ReactPlanError("只支持 earliest_start 事件选择")
    if plan["subject_source"] != "event_name":
        raise ReactPlanError("只支持从 event_name 生成标题")
    if not isinstance(plan["email_body"], str) or not plan["email_body"].strip():
        raise ReactPlanError("email_body 不能为空")
    if not isinstance(plan["subgoals"], list) or not plan["subgoals"]:
        raise ReactPlanError("subgoals 必须是非空列表")
    if not all(isinstance(item, str) and item.strip() for item in plan["subgoals"]):
        raise ReactPlanError("subgoals 必须只包含非空字符串")
    return deepcopy(plan)


def generate_react_plan(model: Any, task: str) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "把用户任务转换为严格 JSON 计划。只输出 target_date、event_selection、"
                "subject_source、email_body、subgoals。不要输出 event_id、recipient、"
                "ground_truth_actions 或任何实际工具调用。当前 WorkBench 子集日期按 2023 年解释。"
            ),
        },
        {"role": "user", "content": task},
    ]
    response = model.generate(messages, response_format=REACT_PLAN_RESPONSE_FORMAT)
    content = getattr(response, "content", response)
    if not isinstance(content, str) or not content.strip():
        raise ReactPlanError("Planner 没有返回计划")
    return parse_react_plan(content)


def extract_task_body(task: str) -> str:
    for pattern in (
        r"(?:tell them|message them|say)\s+['\"]([^'\"]+)['\"]",
        r"(?:body|message)\s*[:：]\s*['\"]([^'\"]+)['\"]",
    ):
        match = re.search(pattern, task, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    if DEFAULT_EMAIL_BODY in task:
        return DEFAULT_EMAIL_BODY
    raise ReactPlanError("任务中没有找到邮件正文")


def extract_task_date(task: str) -> str | None:
    iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", task)
    if iso_match:
        return iso_match.group(1)
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    match = re.search(
        r"\b(" + "|".join(months) + r")\s+(\d{1,2})\b",
        task,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return date(2023, months[match.group(1).lower()], int(match.group(2))).isoformat()


class ProgressMonitor:
    """Environment-aware progress and side-effect monitor."""

    def __init__(
        self,
        memory: TaskMemory,
        execution_monitor: ExecutionMonitor,
        all_events_fn: Callable[[], Iterable[dict[str, Any]]],
        initial_sent_emails: Iterable[dict[str, Any]] = (),
        required_body: str | None = None,
        required_date: str | None = None,
    ) -> None:
        self.memory = memory
        self.execution_monitor = execution_monitor
        self.all_events_fn = all_events_fn
        self.initial_sent_emails = [deepcopy(row) for row in initial_sent_emails]
        self.required_body = required_body
        self.required_date = required_date
        self.decisions: list[dict[str, Any]] = []

    def _record(self, allowed: bool, reason: str, details: Any = None) -> None:
        decision = {"allowed": allowed, "reason": reason, "details": deepcopy(details)}
        self.decisions.append(decision)
        self.memory.record_transition("monitor_decision", decision)

    def block(self, reason: str, details: Any = None) -> dict[str, Any]:
        self._record(False, reason, details)
        self.memory.mark_candidate_blocked(reason)
        return {"ok": False, "code": "blocked", "reason": reason}

    def _environment_event(self, event_id: str) -> dict[str, Any] | None:
        for event in self.all_events_fn():
            if str(event.get("event_id")) == str(event_id):
                return deepcopy(event)
        return None

    def select_event(self, event_id: str) -> dict[str, Any]:
        candidates = self.memory.recent_candidates or []
        if self.memory.candidate_completeness != "complete":
            return self.block(
                "candidate_set_incomplete",
                {
                    "returned_count": self.memory.returned_count,
                    "reason": self.memory.completeness_reason,
                },
            )
        candidate = next(
            (item for item in candidates if str(item.get("event_id")) == str(event_id)),
            None,
        )
        if candidate is None:
            return self.block("event_id_not_in_recent_search", {"event_id": event_id})
        if not self.memory.target_date:
            return self.block("target_date_missing")
        if self.required_date and self.memory.target_date != self.required_date:
            return self.block("selected_event_task_date_mismatch")
        try:
            selected_date = datetime.fromisoformat(candidate["event_start"].replace("Z", "+00:00")).date()
        except (KeyError, TypeError, ValueError):
            return self.block("selected_event_invalid_start")
        if selected_date.isoformat() != self.memory.target_date:
            return self.block("selected_event_date_mismatch", {"event_id": event_id})
        required = ("event_id", "event_name", "participant_email", "event_start")
        if any(not candidate.get(field) for field in required):
            return self.block("selected_event_fields_incomplete", {"event_id": event_id})
        try:
            earliest = select_first_event(
                json.dumps(candidates, ensure_ascii=False), self.memory.target_date
            )
        except ValueError as exc:
            return self.block("candidate_earliest_event_unavailable", {"error": str(exc)})
        if str(earliest["event_id"]) != str(event_id):
            return self.block(
                "selected_event_is_not_earliest",
                {"selected": event_id, "earliest": earliest["event_id"]},
            )
        environment_event = self._environment_event(str(event_id))
        if environment_event is None:
            return self.block("selected_event_missing_from_environment", {"event_id": event_id})
        if environment_event.get("participant_email") != candidate.get("participant_email"):
            return self.block("selected_event_environment_mismatch", {"event_id": event_id})
        self.memory.set_target_event(candidate)
        self._record(True, "event_selected", {"event_id": event_id})
        return {"ok": True, "event": deepcopy(candidate)}

    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        send_fn: Callable[[str, str, str], Any],
    ) -> Any:
        event = self.memory.target_event
        plan = self.memory.planner_output or {}
        if event is None:
            return self.block("event_not_selected_before_send")
        if self.memory.candidate_completeness != "complete":
            return self.block("candidate_set_incomplete_before_send")
        if body != plan.get("email_body"):
            return self.block("email_body_does_not_match_plan")
        if self.required_body and plan.get("email_body") != self.required_body:
            return self.block("planner_body_does_not_match_task")
        if self.required_body and body != self.required_body:
            return self.block("email_body_does_not_match_task")
        if self.required_date and self.memory.target_date != self.required_date:
            return self.block("planner_date_does_not_match_task")
        environment_event = self._environment_event(str(event["event_id"]))
        if environment_event is None:
            return self.block("target_event_missing_from_environment")
        if recipient.strip().lower() != str(environment_event.get("participant_email", "")).strip().lower():
            return self.block("recipient_not_from_environment_event")
        if subject != environment_event.get("event_name"):
            return self.block("subject_not_from_environment_event")
        if self.memory.send_attempted or self.memory.send_confirmed:
            return self.block("send_already_completed")
        fingerprint = make_send_fingerprint(
            event["event_id"], recipient, subject, body
        )
        for row in self.initial_sent_emails:
            if (
                str(row.get("sender/recipient", row.get("recipient", ""))).strip().lower()
                == recipient.strip().lower()
                and row.get("subject") == subject
                and row.get("body") == body
            ):
                return self.block("duplicate_send_fingerprint", {"fingerprint": fingerprint})
        try:
            self.memory.prepare_email(recipient, subject, body)
            result = self.execution_monitor.guarded_send_email(
                self.memory, send_fn, recipient, subject, body
            )
        except (MonitorViolation, TaskMemoryError) as exc:
            self._record(False, str(exc))
            return {"ok": False, "code": "blocked", "reason": str(exc)}
        self._record(True, "send_allowed", {"event_id": event["event_id"]})
        return {"ok": True, "result": result}


class ReactPlannerController:
    """Minimal Planner-Guided ReAct execution around injected WorkBench tools."""

    def __init__(
        self,
        *,
        task_id: str,
        task: str,
        search_events_fn: Callable[..., str],
        send_email_fn: Callable[[str, str, str], Any],
        all_events_fn: Callable[[], Iterable[dict[str, Any]]],
        list_sent_emails_fn: Callable[[], Iterable[dict[str, Any]]],
        state_snapshot_fn: Callable[[], dict[str, Any]],
        model: Any | None = None,
        max_steps: int = 8,
    ) -> None:
        self.task_id = task_id
        self.task = task
        self.search_events_fn = search_events_fn
        self.send_email_fn = send_email_fn
        self.all_events_fn = all_events_fn
        self.list_sent_emails_fn = list_sent_emails_fn
        self.state_snapshot_fn = state_snapshot_fn
        self.model = model
        self.max_steps = max_steps
        try:
            self.task_required_body = extract_task_body(task)
        except ReactPlanError:
            self.task_required_body = None
        self.task_required_date = extract_task_date(task)
        self.memory = TaskMemory(task_id=task_id)
        self.execution_monitor = ExecutionMonitor()
        self._initial_sent = [deepcopy(row) for row in list_sent_emails_fn()]
        self.progress_monitor = ProgressMonitor(
            self.memory,
            self.execution_monitor,
            all_events_fn,
            self._initial_sent,
            self.task_required_body,
            self.task_required_date,
        )
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_observations: list[dict[str, Any]] = []
        self.initial_state = deepcopy(state_snapshot_fn())
        self.send_exception = False

    def _record_tool(self, name: str, arguments: dict[str, Any], result: Any) -> Any:
        self.tool_calls.append({"tool": name, "arguments": deepcopy(arguments)})
        self.tool_observations.append({"tool": name, "observation": deepcopy(result)})
        return result

    def _candidate_completeness(
        self,
        candidates: list[dict[str, Any]],
        query: str,
        time_min: str | None,
        time_max: str | None,
    ) -> tuple[str, str]:
        if not self.memory.target_date:
            return "incomplete", "target_date_not_set"
        expected_min = f"{self.memory.target_date} 00:00:00"
        expected_max = f"{self.memory.target_date} 23:59:59"
        if time_min != expected_min or time_max != expected_max:
            return "incomplete", "search_time_window_missing_or_mismatched"
        try:
            all_target_events = [
                event
                for event in self.all_events_fn()
                if datetime.fromisoformat(event["event_start"].replace("Z", "+00:00")).date().isoformat()
                == self.memory.target_date
            ]
        except (TypeError, KeyError, ValueError):
            return "incomplete", "environment_event_time_unavailable"
        if query:
            return "incomplete", "search_query_may_filter_target_date_events"
        if not all_target_events and not candidates:
            return "complete", "environment_has_no_events_on_target_date"
        if len(all_target_events) > 5:
            return "incomplete", "environment_event_count_exceeds_tool_limit"
        expected_ids = {str(event.get("event_id")) for event in all_target_events}
        returned_ids = {str(event.get("event_id")) for event in candidates}
        if expected_ids != returned_ids:
            return "incomplete", "returned_event_ids_do_not_cover_environment_events"
        return "complete", "returned_ids_match_environment_events"

    def search_events(
        self,
        query: str = "",
        time_min: str | None = None,
        time_max: str | None = None,
    ) -> str:
        arguments = {"query": query, "time_min": time_min, "time_max": time_max}
        try:
            result = self.search_events_fn(**arguments)
            candidates = json.loads(result)
            if not isinstance(candidates, list):
                raise ValueError("search_events 返回结构不是列表")
            completeness, reason = self._candidate_completeness(
                candidates, query, time_min, time_max
            )
            observation = {
                "events": candidates,
                "returned_count": len(candidates),
                "candidate_completeness": completeness,
                "completeness_reason": reason,
            }
            self.memory.record_search_result(candidates, completeness, reason)
            if not candidates and completeness == "complete":
                self.memory.mark_completed_without_action(reason)
            return self._record_tool(
                "search_events", arguments, json.dumps(observation, ensure_ascii=False)
            )
        except Exception as exc:
            self.memory.mark_failed({"tool": "search_events", "error": str(exc)})
            result = json.dumps({"ok": False, "code": "search_failed", "error": str(exc)})
            return self._record_tool("search_events", arguments, result)

    def select_event(self, event_id: str) -> str:
        arguments = {"event_id": event_id}
        result = self.progress_monitor.select_event(event_id)
        return self._record_tool("select_event", arguments, json.dumps(result, ensure_ascii=False))

    def send_email(self, recipient: str, subject: str, body: str) -> str:
        arguments = {"recipient": recipient, "subject": subject, "body": body}
        try:
            result = self.progress_monitor.send_email(
                recipient, subject, body, self.send_email_fn
            )
        except Exception as exc:
            self.send_exception = True
            self.memory.mark_failed(
                {"type": type(exc).__name__, "message": str(exc)}
            )
            result = {"ok": False, "code": "send_exception", "reason": str(exc)}
        return self._record_tool(
            "send_email", arguments, json.dumps(result, ensure_ascii=False)
        )

    def _tools(self):
        controller = self

        @tool
        def search_events(
            query: str = "",
            time_min: str | None = None,
            time_max: str | None = None,
        ) -> str:
            """Search WorkBench calendar events.

            Args:
                query: Optional text filter. Leave empty to preserve candidate completeness.
                time_min: Inclusive lower bound for event end time.
                time_max: Inclusive upper bound for event start time.
            """
            return controller.search_events(query, time_min, time_max)

        @tool
        def select_event(event_id: str) -> str:
            """Select one event from the most recent complete search result.

            Args:
                event_id: Event ID copied from the latest search observation.
            """
            return controller.select_event(event_id)

        @tool
        def send_email(recipient: str, subject: str, body: str) -> str:
            """Send the planned email after ProgressMonitor checks.

            Args:
                recipient: Recipient copied from the selected event.
                subject: Subject copied from the selected event name.
                body: Body from the planner output.
            """
            return controller.send_email(recipient, subject, body)

        return [search_events, select_event, send_email]

    def run(self, plan: str | dict[str, Any] | None = None) -> dict[str, Any]:
        planner_output: dict[str, Any] | None = None
        try:
            if plan is None:
                if self.model is None:
                    raise ReactPlanError("没有提供 Planner 模型")
                planner_output = generate_react_plan(self.model, self.task)
            else:
                planner_output = parse_react_plan(plan)
            if (
                self.task_required_body
                and planner_output["email_body"] != self.task_required_body
            ):
                raise ReactPlanError("Planner email_body 与用户任务不一致")
            if (
                self.task_required_date
                and planner_output["target_date"] != self.task_required_date
            ):
                raise ReactPlanError("Planner target_date 与用户任务不一致")
            self.memory.set_plan(planner_output)
        except (ReactPlanError, TaskMemoryError) as exc:
            self.memory.mark_failed({"type": type(exc).__name__, "message": str(exc)})
            return self._trace(planner_output, [], "failed", None)

        agent = ToolCallingAgent(
            model=self.model,
            tools=self._tools(),
            max_steps=self.max_steps,
        )
        agent.prompt_templates["system_prompt"] = (
            "你是 Planner-Guided ReAct Executor。先使用 search_events 获取事实，"
            "读取观察后必须调用 select_event 选择事件，再根据选择结果调用 send_email。"
            "不要猜测 event_id 或收件人。若观察或工具返回 blocked、candidate_set_incomplete、"
            "无匹配活动，停止副作用并调用 final_answer 说明结果。严格使用结构化工具调用格式。"
            f"当前 Planner 计划为：{json.dumps(planner_output, ensure_ascii=False)}"
        )
        steps: list[Any] = []
        agent_state = "exception"
        try:
            result = agent.run(self.task, return_full_result=True)
            agent_state = result.state
            steps = result.steps
        except Exception as exc:
            agent_state = "exception"
            try:
                steps = agent.memory.get_full_steps()
            except Exception:
                steps = []
            self.memory.mark_failed({"type": type(exc).__name__, "message": str(exc)})

        final_state = self.state_snapshot_fn()
        added_emails = final_state.get("emails_added", [])
        if self.memory.send_attempted and not self.send_exception and not self.memory.send_confirmed:
            prepared = self.memory.prepared_email
            if (
                prepared is not None
                and len(added_emails) == 1
                and added_emails[0].get("sender/recipient") == prepared["recipient"]
                and added_emails[0].get("subject") == prepared["subject"]
                and added_emails[0].get("body") == prepared["body"]
            ):
                self.memory.mark_send_confirmed({"email": added_emails[0]})
            else:
                self.memory.mark_failed(
                    {"reason": "final_state_email_mismatch", "new_emails": added_emails}
                )
        if self.memory.send_confirmed and len(added_emails) == 1:
            terminal_status = "success"
        elif self.memory.status == "completed_without_action":
            terminal_status = "completed_without_action"
        elif self.memory.status in {"blocked", "candidate_insufficient"}:
            terminal_status = "blocked"
        elif self.memory.send_attempted or agent_state == "exception":
            terminal_status = "failed"
        else:
            terminal_status = "blocked"
        return self._trace(planner_output, steps, terminal_status, agent_state)

    def _trace(
        self,
        planner_output: dict[str, Any] | None,
        steps: list[Any],
        terminal_status: str,
        agent_state: str | None,
    ) -> dict[str, Any]:
        memory_snapshot = self.memory.snapshot()
        return {
            "benchmark": "WorkBench",
            "method": "improved_1_react",
            "task_id": self.task_id,
            "task": self.task,
            "planner_output": deepcopy(planner_output),
            "react_steps": deepcopy(steps),
            "tool_calls": deepcopy(self.tool_calls),
            "tool_observations": deepcopy(self.tool_observations),
            "memory_transitions": deepcopy(memory_snapshot["memory_transitions"]),
            "memory_snapshot": memory_snapshot,
            "monitor_decisions": deepcopy(self.progress_monitor.decisions),
            "selected_event": deepcopy(self.memory.target_event),
            "candidate_completeness": self.memory.candidate_completeness,
            "returned_count": self.memory.returned_count,
            "completeness_reason": self.memory.completeness_reason,
            "initial_state": deepcopy(self.initial_state),
            "final_state": deepcopy(self.state_snapshot_fn()),
            "agent_state": agent_state,
            "terminal_status": terminal_status,
            "success": terminal_status == "success",
        }


def main() -> None:
    from importlib.metadata import version

    from workbench_subset_agent import WorkbenchSubsetState

    task_record = json.loads((BENCHMARK_ROOT / "task.json").read_text(encoding="utf-8"))
    state = WorkbenchSubsetState(BENCHMARK_ROOT)
    model = OpenAIModel(
        model_id="longagent-qwen",
        api_base="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0,
        max_tokens=256,
    )
    controller = ReactPlannerController(
        task_id=task_record["task_id"],
        task=task_record["task"],
        search_events_fn=state.search_events,
        send_email_fn=state.send_email,
        all_events_fn=lambda: [row.copy() for row in state.calendar_events],
        list_sent_emails_fn=lambda: [row.copy() for row in state.emails],
        state_snapshot_fn=state.snapshot,
        model=model,
    )
    trace = controller.run()
    trace.update(
        {
            "source_repo": task_record["source_repo"],
            "source_commit": task_record["source_commit"],
            "model_id": model.model_id,
            "smolagents_version": version("smolagents"),
        }
    )
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    trace_path = runs_dir / f"workbench-react-planner-{uuid4().hex}.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[轨迹文件]", trace_path)
    print("[终态]", trace["terminal_status"])
    print("[WorkBench 终态]", trace["final_state"])


if __name__ == "__main__":
    main()
