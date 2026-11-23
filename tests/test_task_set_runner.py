import unittest
from pathlib import Path

from experiments.run_task_set import (
    PROJECT_ROOT,
    _baseline_tool_calls,
    invalid_action_count,
    load_tasks,
    evaluate_run,
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


if __name__ == "__main__":
    unittest.main()
