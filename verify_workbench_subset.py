import argparse
import json
from pathlib import Path


def verify_workbench_trace(trace: dict, target: dict) -> dict:
    """按任务的 ground truth 检查最终新增邮件。

    发送任务要求一封完全匹配的邮件，无匹配活动任务要求零封邮件。
    """

    expected = target.get("ground_truth_actions", [])
    expected_emails = [action["arguments"] for action in expected]
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
        "email_count_matches": len(matching_emails) == len(expected_emails),
        "email_matches_ground_truth": matching_emails == expected_emails,
    }
    if len(expected_emails) == 1:
        checks["one_email_added"] = len(matching_emails) == 1
    if len(expected_emails) == 0:
        checks["no_email_added"] = len(matching_emails) == 0

    return {
        "task_id": trace["task_id"],
        "expected_email": expected_emails[0] if expected_emails else None,
        "expected_emails": expected_emails,
        "actual_added_emails": matching_emails,
        "checks": checks,
        "task_success": all(checks.values()),
        "harmful_side_effects": [
            email for email in matching_emails if email not in expected_emails
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
