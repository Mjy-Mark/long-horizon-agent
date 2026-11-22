import argparse
import json
from pathlib import Path


def verify_workbench_trace(trace: dict, target: dict) -> dict:
    """按选定 WorkBench 任务的最终状态检查新增邮件是否唯一且准确。"""
    expected = target["ground_truth_actions"]
    expected_email = expected[0]["arguments"]
    added_emails = trace["final_state"]["emails_added"]

    matching_emails = [
        {
            "recipient": row["sender/recipient"],
            "subject": row["subject"],
            "body": row["body"],
        }
        for row in added_emails
    ]

    checks = {
        "one_email_added": len(matching_emails) == 1,
        "email_matches_ground_truth": matching_emails == [expected_email],
    }
    return {
        "task_id": trace["task_id"],
        "expected_email": expected_email,
        "actual_added_emails": matching_emails,
        "checks": checks,
        "task_success": all(checks.values()),
        "harmful_side_effects": [
            email for email in matching_emails if email != expected_email
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证一条 WorkBench 子集任务轨迹")
    parser.add_argument("trace", type=Path)
    parser.add_argument("task", type=Path)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    target = json.loads(args.task.read_text(encoding="utf-8"))
    verdict = verify_workbench_trace(trace, target)

    print("任务：", trace["task"])
    print("框架结束状态：", trace["agent_state"])
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
