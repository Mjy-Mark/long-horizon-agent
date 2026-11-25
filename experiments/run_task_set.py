"""Run the fixed task set with Baseline and Improved 1.

Ground-truth actions stay in the verifier path.  Only each task's instruction
is passed to the agent or planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from smolagents import OpenAIModel, ToolCallingAgent, tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "workbench_subset"
DEFAULT_TASKS_PATH = PROJECT_ROOT / "experiments" / "tasks.jsonl"
DEFAULT_TRACE_ROOT = PROJECT_ROOT / "runs" / "task-set"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verify_workbench_subset import verify_workbench_trace
from workbench_planner_agent import WorkbenchPlannerController
from workbench_react_planner_agent import ReactPlannerController
from workbench_subset_agent import WorkbenchSubsetState


def load_tasks(path: Path = DEFAULT_TASKS_PATH) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"任务文件第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(task, dict):
            raise ValueError(f"任务文件第 {line_number} 行顶层结构必须是对象")
        for field in ("task_id", "instruction", "target_date", "ground_truth_actions"):
            if field not in task:
                raise ValueError(f"任务 {line_number} 缺少字段: {field}")
        if not isinstance(task["instruction"], str) or not task["instruction"].strip():
            raise ValueError(f"任务 {line_number} 的 instruction 不能为空")
        if not isinstance(task["ground_truth_actions"], list):
            raise ValueError(f"任务 {line_number} 的 ground_truth_actions 必须是列表")
        tasks.append(task)

    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("任务文件包含重复 task_id")
    return tasks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_state_metadata(root: Path) -> dict[str, Any]:
    return {
        "initial_state_source": "benchmarks/workbench_subset",
        "calendar_events_sha256": file_sha256(root / "calendar_events.csv"),
        "emails_sha256": file_sha256(root / "emails.csv"),
    }


def make_model(model_id: str, api_base: str, temperature: float, max_tokens: int) -> OpenAIModel:
    return OpenAIModel(
        model_id=model_id,
        api_base=api_base,
        api_key="local",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _baseline_tools(state: WorkbenchSubsetState):
    @tool
    def search_events(
        query: str = "",
        time_min: str | None = None,
        time_max: str | None = None,
    ) -> str:
        """Search WorkBench calendar events.

        Args:
            query: Optional text matched against event names and participant emails.
            time_min: Inclusive lower bound for an event's end time, formatted as YYYY-MM-DD HH:MM:SS.
            time_max: Inclusive upper bound for an event's start time, formatted as YYYY-MM-DD HH:MM:SS.
        """
        return state.search_events(query=query, time_min=time_min, time_max=time_max)

    @tool
    def send_email(recipient: str, subject: str, body: str) -> str:
        """Send an email in the WorkBench email sandbox.

        Args:
            recipient: Recipient email address.
            subject: Email subject.
            body: Email body.
        """
        return state.send_email(recipient=recipient, subject=subject, body=body)

    return search_events, send_email


def _write_trace(path: Path, trace: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_baseline(
    task: dict[str, Any],
    workbench_root: Path,
    trace_path: Path,
    *,
    model_id: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
    batch_id: str,
) -> dict[str, Any]:
    state = WorkbenchSubsetState(workbench_root)
    model = make_model(model_id, api_base, temperature, max_tokens)
    search_events, send_email = _baseline_tools(state)
    agent = ToolCallingAgent(
        model=model,
        tools=[search_events, send_email],
        max_steps=max_steps,
    )
    agent.prompt_templates["system_prompt"] = (
        "根据任务和工具反馈持续执行，直到任务目标全部完成。"
        "使用 API 提供的结构化工具调用格式。完成后调用 final_answer，并用中文回答。"
    )

    trace = {
        "benchmark": "WorkBench",
        "method": "baseline_react",
        "batch_id": batch_id,
        "task_id": task["task_id"],
        "task": task["instruction"],
        "model_id": model.model_id,
        "smolagents_version": version("smolagents"),
        "max_steps": max_steps,
        **initial_state_metadata(workbench_root),
        "initial_state": state.snapshot(),
    }
    try:
        result = agent.run(task["instruction"], return_full_result=True)
    except Exception as exc:
        trace["agent_state"] = "exception"
        trace["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        trace["steps"] = agent.memory.get_full_steps()
    else:
        trace["agent_state"] = result.state
        trace["final_answer"] = str(result.output)
        trace["steps"] = result.steps
    finally:
        trace["final_state"] = state.snapshot()
        _write_trace(trace_path, trace)
    return trace


def run_improved_1(
    task: dict[str, Any],
    workbench_root: Path,
    trace_path: Path,
    *,
    model_id: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    batch_id: str,
) -> dict[str, Any]:
    state = WorkbenchSubsetState(workbench_root)
    model = make_model(model_id, api_base, temperature, max_tokens)
    controller = WorkbenchPlannerController(
        task_id=task["task_id"],
        task=task["instruction"],
        search_events_fn=state.search_events,
        send_email_fn=state.send_email,
        list_sent_emails_fn=lambda: [row.copy() for row in state.emails],
        state_snapshot_fn=state.snapshot,
        model=model,
    )
    trace = controller.run()
    trace.update(
        {
            "batch_id": batch_id,
            "task_id": task["task_id"],
            "task": task["instruction"],
            "model_id": model.model_id,
            "smolagents_version": version("smolagents"),
            **initial_state_metadata(workbench_root),
        }
    )
    _write_trace(trace_path, trace)
    return trace


def run_improved_1_react(
    task: dict[str, Any],
    workbench_root: Path,
    trace_path: Path,
    *,
    model_id: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    batch_id: str,
) -> dict[str, Any]:
    state = WorkbenchSubsetState(workbench_root)
    model = make_model(model_id, api_base, temperature, max_tokens)
    controller = ReactPlannerController(
        task_id=task["task_id"],
        task=task["instruction"],
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
            "batch_id": batch_id,
            "task_id": task["task_id"],
            "task": task["instruction"],
            "model_id": model.model_id,
            "smolagents_version": version("smolagents"),
            **initial_state_metadata(workbench_root),
        }
    )
    _write_trace(trace_path, trace)
    return trace


def _baseline_tool_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_calls = step.get("tool_calls", []) or []
        if not step_calls:
            model_output_message = step.get("model_output_message", {}) or {}
            step_calls = model_output_message.get("tool_calls", []) or []
        for call in step_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            calls.append(
                {
                    "tool": function.get("name"),
                    "arguments": function.get("arguments", {}),
                }
            )
    return calls


def tool_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    if trace.get("method") == "baseline_react":
        return _baseline_tool_calls(trace)
    return [
        {"tool": call.get("tool"), "arguments": call.get("arguments", {})}
        for call in trace.get("tool_calls", [])
    ]


def tool_name(call: dict[str, Any]) -> str:
    return str(call.get("tool", "")).split(".")[-1]


def environment_tool_calls(calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for call in calls if tool_name(call) != "final_answer"]


def final_answer_calls(calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for call in calls if tool_name(call) == "final_answer"]


def send_calls(calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for call in calls if tool_name(call) == "send_email"]


def invalid_action_count(task: dict[str, Any], calls: Iterable[dict[str, Any]]) -> int:
    expected = [action["arguments"] for action in task["ground_truth_actions"]]
    sends = [call.get("arguments", {}) for call in send_calls(calls)]
    return sum(1 for index, send in enumerate(sends) if index >= len(expected) or send != expected[index])


def evaluate_run(
    task: dict[str, Any], trace: dict[str, Any], trace_path: Path
) -> dict[str, Any]:
    verdict = verify_workbench_trace(
        trace,
        {
            "task_id": task["task_id"],
            "ground_truth_actions": task["ground_truth_actions"],
        },
    )
    calls = tool_calls(trace)
    environment_calls = environment_tool_calls(calls)
    answer_calls = final_answer_calls(calls)
    email_calls = send_calls(calls)
    invalid_sends = invalid_action_count(task, calls)
    monitor_decisions = trace.get("monitor_decisions", [])
    react_steps = trace.get("react_steps", [])
    memory_transitions = trace.get("memory_transitions", [])
    planner_call_count = 1 if trace.get("planner_output") is not None else 0
    return {
        "task_id": task["task_id"],
        "method": trace["method"],
        "trace": str(trace_path.relative_to(PROJECT_ROOT)),
        "verdict": verdict,
        "framework_success": trace.get("success", trace.get("agent_state") == "success"),
        "tool_call_count": len(calls),
        "environment_tool_call_count": len(environment_calls),
        "final_answer_call_count": len(answer_calls),
        "send_call_count": len(email_calls),
        "invalid_action_count": invalid_sends,
        "invalid_send_count": invalid_sends,
        "planner_call_count": planner_call_count,
        "react_step_count": len(react_steps),
        "memory_transition_count": len(memory_transitions),
        "monitor_block_count": sum(
            not decision.get("allowed", False) for decision in monitor_decisions
        ),
        "unexpected_email_count": verdict["unexpected_email_count"],
        "duplicate_expected_email_count": verdict["duplicate_expected_email_count"],
        "extra_email_count": verdict["extra_email_count"],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({record["method"] for record in records})
    summary: dict[str, Any] = {}
    for method in methods:
        method_records = [record for record in records if record["method"] == method]
        total = len(method_records)
        successes = sum(record["verdict"]["task_success"] for record in method_records)
        framework_successes = sum(record["framework_success"] for record in method_records)
        unwanted = sum(bool(record["verdict"]["harmful_side_effects"]) for record in method_records)
        tool_calls = sum(record["tool_call_count"] for record in method_records)
        environment_calls = sum(record["environment_tool_call_count"] for record in method_records)
        final_answers = sum(record["final_answer_call_count"] for record in method_records)
        send_calls_total = sum(record["send_call_count"] for record in method_records)
        invalid_actions = sum(record["invalid_action_count"] for record in method_records)
        summary[method] = {
            "task_count": total,
            "task_success_rate": successes / total if total else 0.0,
            "verifier_task_success_rate": successes / total if total else 0.0,
            "framework_success_rate": framework_successes / total if total else 0.0,
            "unwanted_side_effect_rate": unwanted / total if total else 0.0,
            "average_environment_tool_calls": environment_calls / total if total else 0.0,
            "average_tool_calls": tool_calls / total if total else 0.0,
            "final_answer_calls": final_answers,
            "send_calls": send_calls_total,
            "invalid_action_rate": invalid_actions / tool_calls if tool_calls else 0.0,
            "invalid_send_rate": invalid_actions / send_calls_total if send_calls_total else 0.0,
            "unexpected_email_count": sum(
                record["unexpected_email_count"] for record in method_records
            ),
            "duplicate_expected_email_count": sum(
                record["duplicate_expected_email_count"] for record in method_records
            ),
            "extra_email_count": sum(record["extra_email_count"] for record in method_records),
            "planner_call_count": sum(record["planner_call_count"] for record in method_records),
            "react_step_count": sum(record["react_step_count"] for record in method_records),
            "memory_transition_count": sum(
                record["memory_transition_count"] for record in method_records
            ),
            "monitor_block_count": sum(
                record["monitor_block_count"] for record in method_records
            ),
        }
    return summary


def metric_definitions() -> dict[str, str]:
    return {
        "average_environment_tool_calls": "environment tool calls excluding final_answer",
        "average_tool_calls": "all parsed tool calls including final_answer",
        "invalid_action_rate": "invalid send calls divided by all parsed tool calls",
        "invalid_send_rate": "invalid send calls divided by all send calls",
    }


def recompute_results(results_path: Path, tasks_path: Path = DEFAULT_TASKS_PATH) -> Path:
    tasks = {task["task_id"]: task for task in load_tasks(tasks_path)}
    result = json.loads(results_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for old_record in result["records"]:
        task = tasks[old_record["task_id"]]
        trace_path = PROJECT_ROOT / old_record["trace"]
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        records.append(evaluate_run(task, trace, trace_path))
    result["records"] = records
    result["summary"] = summarize(records)
    result["metric_definitions"] = metric_definitions()
    result["postprocessing_version"] = "metrics_v2"
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[重算结果]", results_path)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return results_path


def run_batch(args: argparse.Namespace) -> Path:
    tasks_path = args.tasks if args.tasks.is_absolute() else PROJECT_ROOT / args.tasks
    trace_root_base = (
        args.trace_root if args.trace_root.is_absolute() else PROJECT_ROOT / args.trace_root
    )
    results_path = (
        None
        if args.results_path is None
        else args.results_path
        if args.results_path.is_absolute()
        else PROJECT_ROOT / args.results_path
    )
    tasks = load_tasks(tasks_path)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    trace_root = trace_root_base / batch_id
    records: list[dict[str, Any]] = []

    for task in tasks:
        for method in args.methods:
            trace_path = trace_root / f"{task['task_id']}-{method}.json"
            with tempfile.TemporaryDirectory(prefix=f"task-{task['task_id']}-") as temp_dir:
                temp_root = Path(temp_dir) / "workbench_subset"
                shutil.copytree(BENCHMARK_ROOT, temp_root)
                if method == "baseline_react":
                    trace = run_baseline(
                        task,
                        temp_root,
                        trace_path,
                        model_id=args.model_id,
                        api_base=args.api_base,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        max_steps=args.max_steps,
                        batch_id=batch_id,
                    )
                elif method == "additional_baseline_deterministic":
                    trace = run_improved_1(
                        task,
                        temp_root,
                        trace_path,
                        model_id=args.model_id,
                        api_base=args.api_base,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        batch_id=batch_id,
                    )
                    trace["method"] = "additional_baseline_deterministic"
                    _write_trace(trace_path, trace)
                else:
                    trace = run_improved_1_react(
                        task,
                        temp_root,
                        trace_path,
                        model_id=args.model_id,
                        api_base=args.api_base,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        batch_id=batch_id,
                    )
            records.append(evaluate_run(task, trace, trace_path))

    result = {
        "batch_id": batch_id,
        "task_file": str(tasks_path.relative_to(PROJECT_ROOT)),
        "methods": args.methods,
        "model_id": args.model_id,
        "model_parameters": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_steps": args.max_steps,
        },
        "ground_truth_exposure": "verifier_only",
        "metric_definitions": metric_definitions(),
        "postprocessing_version": "metrics_v2",
        "records": records,
        "summary": summarize(records),
    }
    results_path = results_path or DEFAULT_RESULTS_ROOT / f"task-set-{batch_id}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[批量结果]", results_path)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行固定 WorkBench 任务集的 Baseline 与 Improved 1 对照")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "baseline_react",
            "additional_baseline_deterministic",
            "improved_1_react",
        ],
        default=["baseline_react", "additional_baseline_deterministic", "improved_1_react"],
    )
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--model-id", default="longagent-qwen")
    parser.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--recompute-results", type=Path, default=None)
    args = parser.parse_args()
    if args.recompute_results is not None:
        results_path = (
            args.recompute_results
            if args.recompute_results.is_absolute()
            else PROJECT_ROOT / args.recompute_results
        )
        recompute_results(results_path, args.tasks if args.tasks.is_absolute() else PROJECT_ROOT / args.tasks)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
