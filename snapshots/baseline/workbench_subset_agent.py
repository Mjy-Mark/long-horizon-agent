import csv
import json
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from smolagents import OpenAIModel, ToolCallingAgent, tool


BENCHMARK_ROOT = Path(__file__).resolve().parent / "benchmarks" / "workbench_subset"
FIXED_CURRENT_TIME = "2023-11-30 00:00:00"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


class WorkbenchSubsetState:
    """保存这条 WorkBench 任务所需的日历和邮件状态。"""

    def __init__(self, root: Path):
        self.calendar_events = read_csv(root / "calendar_events.csv")
        self.emails = read_csv(root / "emails.csv")
        self.initial_email_ids = {row["email_id"] for row in self.emails}

    def search_events(
        self,
        query: str = "",
        time_min: str | None = None,
        time_max: str | None = None,
    ) -> str:
        """按 WorkBench 的日历搜索规则返回最多五条活动。"""
        query_lower = query.lower()
        events = []
        for event in self.calendar_events:
            if query_lower and (
                query_lower not in event["event_name"].lower()
                and query_lower not in event["participant_email"].lower()
            ):
                continue

            event_start = datetime.fromisoformat(event["event_start"])
            event_end = event_start + timedelta(minutes=int(event["duration"]))
            if time_min and event_end < datetime.fromisoformat(time_min):
                continue
            if time_max and event_start > datetime.fromisoformat(time_max):
                continue
            events.append(event.copy())

        return json.dumps(events[:5], ensure_ascii=False)

    def send_email(self, recipient: str, subject: str, body: str) -> str:
        """按 WorkBench 的邮件发送规则向沙箱追加一封邮件。"""
        if not recipient or not subject or not body:
            return "Recipient, subject, or body not provided."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."

        recipient = recipient.lower()
        next_id = max(int(row["email_id"]) for row in self.emails) + 1
        self.emails.append(
            {
                "email_id": f"{next_id:08d}",
                "inbox/outbox": "outbox",
                "sender/recipient": recipient,
                "subject": subject,
                "sent_datetime": FIXED_CURRENT_TIME,
                "body": body,
            }
        )
        return "Email sent successfully."

    def snapshot(self) -> dict:
        added_emails = [
            row.copy() for row in self.emails if row["email_id"] not in self.initial_email_ids
        ]
        return {
            "calendar_event_count": len(self.calendar_events),
            "email_count": len(self.emails),
            "emails_added": added_emails,
        }


state = WorkbenchSubsetState(BENCHMARK_ROOT)


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
    result = state.search_events(query=query, time_min=time_min, time_max=time_max)
    print(
        f"[WorkBench 工具] calendar.search_events(query={query!r}, "
        f"time_min={time_min!r}, time_max={time_max!r}) -> {result}",
        flush=True,
    )
    return result


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email in the WorkBench email sandbox.

    Args:
        recipient: Recipient email address.
        subject: Email subject.
        body: Email body.
    """
    result = state.send_email(recipient=recipient, subject=subject, body=body)
    print(
        f"[WorkBench 工具] email.send_email(recipient={recipient!r}, "
        f"subject={subject!r}, body={body!r}) -> {result}",
        flush=True,
    )
    return result


def main() -> None:
    model = OpenAIModel(
        model_id="longagent-qwen",
        api_base="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0,
        max_tokens=256,
    )

    agent = ToolCallingAgent(
        model=model,
        tools=[search_events, send_email],
        max_steps=6,
    )
    agent.prompt_templates["system_prompt"] = (
        "根据任务和工具反馈持续执行，直到任务目标全部完成。"
        "使用 API 提供的结构化工具调用格式。完成后调用 final_answer，并用中文回答。"
    )

    task_path = BENCHMARK_ROOT / "task.json"
    task_record = json.loads(task_path.read_text(encoding="utf-8"))
    task = task_record["task"]

    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    trace_path = runs_dir / f"workbench-react-{uuid4().hex}.json"

    trace = {
        "benchmark": "WorkBench",
        "source_repo": task_record["source_repo"],
        "source_commit": task_record["source_commit"],
        "task_id": task_record["task_id"],
        "method": "react",
        "task": task,
        "model_id": model.model_id,
        "smolagents_version": version("smolagents"),
        "max_steps": agent.max_steps,
        "initial_state": state.snapshot(),
    }

    try:
        result = agent.run(task, return_full_result=True)
    except Exception as exc:
        trace["agent_state"] = "exception"
        trace["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        trace["steps"] = agent.memory.get_full_steps()
        raise
    else:
        trace["agent_state"] = result.state
        trace["final_answer"] = str(result.output)
        trace["steps"] = result.steps
    finally:
        trace["final_state"] = state.snapshot()
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("[轨迹文件]", trace_path)

    print("[Agent 结束状态]", result.state)
    print("最终回答：", result.output)
    print("[WorkBench 终态]", trace["final_state"])


if __name__ == "__main__":
    main()
