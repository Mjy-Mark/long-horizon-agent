import unittest
from pathlib import Path

from experiments.run_task_set import (
    PROJECT_ROOT,
    _baseline_tool_calls,
    _react_tool_calls,
    invalid_action_count,
    load_tasks,
    evaluate_run,
    summarize,
)


class TestTaskSetRunner(unittest.TestCase):
    def test_loads_fixed_tasks_without_duplicate_ids(self):
        tasks = load_tasks()

        self.assertEqual(
            [task["task_id"] for task in tasks],
            [
                "email_first_event_001",
                "email_first_event_late_date_001",
                "email_no_matching_event_001",
            ],
        )
        self.assertTrue(all(task["initial_state"] == "fresh_workbench_subset" for task in tasks))

    def test_no_send_task_is_successful_when_no_email_is_added(self):
        task = load_tasks()[-1]
        trace = {
            "method": "improved_1",
            "task_id": task["task_id"],
            "final_state": {"emails_added": []},
            "success": False,
            "tool_calls": [
                {"tool": "calendar.search_events", "arguments": {}}
            ],
        }

        record = evaluate_run(
            task,
            trace,
            PROJECT_ROOT / "runs/task-set/no-send.json",
        )

        self.assertTrue(record["verdict"]["task_success"])
        self.assertFalse(record["framework_success"])
        self.assertEqual(record["invalid_action_count"], 0)

    def test_extra_email_is_an_invalid_action(self):
        task = load_tasks()[0]
        expected = task["ground_truth_actions"][0]["arguments"]
        calls = [
            {"tool": "email.send_email", "arguments": expected},
            {"tool": "send_email", "arguments": expected},
        ]

        self.assertEqual(invalid_action_count(task, calls), 1)

    def test_baseline_parser_keeps_tool_calls_from_parse_error_step(self):
        trace = {
            "steps": [
                {
                    "tool_calls": [],
                    "model_output_message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "send_email",
                                    "arguments": {"recipient": "a@example.com"},
                                }
                            }
                        ]
                    },
                }
            ]
        }

        calls = _baseline_tool_calls(trace)
        self.assertEqual(calls[0]["tool"], "send_email")

    def test_react_parser_counts_final_answer_from_react_steps(self):
        trace = {
            "method": "improved_1_react",
            "react_steps": [
                {
                    "tool_calls": [
                        {"function": {"name": "search_events", "arguments": {}}},
                        {"function": {"name": "select_event", "arguments": {"event_id": "001"}}},
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "send_email",
                                "arguments": {"recipient": "a@example.com"},
                            }
                        },
                        {"function": {"name": "final_answer", "arguments": {"answer": "done"}}},
                    ]
                },
            ],
            "tool_calls": [
                {"tool": "search_events", "arguments": {}},
                {"tool": "select_event", "arguments": {"event_id": "001"}},
                {"tool": "send_email", "arguments": {"recipient": "a@example.com"}},
            ],
        }

        calls = _react_tool_calls(trace)

        self.assertEqual(
            [call["tool"] for call in calls],
            ["search_events", "select_event", "send_email", "final_answer"],
        )

    def test_summary_separates_environment_and_final_answer_calls(self):
        task = load_tasks()[0]
        expected = task["ground_truth_actions"][0]["arguments"]
        trace = {
            "method": "baseline_react",
            "task_id": task["task_id"],
            "agent_state": "success",
            "final_state": {
                "emails_added": [
                    {
                        "sender/recipient": expected["recipient"],
                        "subject": expected["subject"],
                        "body": expected["body"],
                    },
                    {
                        "sender/recipient": "other@example.com",
                        "subject": "Other",
                        "body": "Other",
                    },
                    {
                        "sender/recipient": expected["recipient"],
                        "subject": expected["subject"],
                        "body": expected["body"],
                    },
                ]
            },
            "steps": [
                {
                    "tool_calls": [
                        {"function": {"name": "search_events", "arguments": {}}},
                        {"function": {"name": "send_email", "arguments": expected}},
                        {"function": {"name": "send_email", "arguments": {"recipient": "other@example.com"}}},
                        {"function": {"name": "final_answer", "arguments": {"answer": "done"}}},
                    ]
                }
            ],
        }

        record = evaluate_run(task, trace, PROJECT_ROOT / "runs/task-set/metrics.json")
        metrics = summarize([record])["baseline_react"]

        self.assertEqual(record["unexpected_email_count"], 1)
        self.assertEqual(record["duplicate_expected_email_count"], 1)
        self.assertEqual(record["extra_email_count"], 2)
        self.assertEqual(metrics["average_environment_tool_calls"], 3.0)
        self.assertEqual(metrics["final_answer_calls"], 1)
        self.assertEqual(metrics["invalid_send_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
