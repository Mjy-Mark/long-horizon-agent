import json
import unittest

from workbench_react_planner_agent import (
    ReactPlanError,
    ReactPlannerController,
    parse_react_plan,
)


PLAN = {
    "target_date": "2023-11-30",
    "event_selection": "earliest_start",
    "subject_source": "event_name",
    "email_body": "Remember to attend this event.",
    "subgoals": ["Search events", "Select the earliest event", "Send one email"],
}


EVENTS = [
    {
        "event_id": "001",
        "event_name": "Morning Meeting",
        "participant_email": "a@example.com",
        "event_start": "2023-11-30 09:30:00",
    },
    {
        "event_id": "002",
        "event_name": "Afternoon Meeting",
        "participant_email": "b@example.com",
        "event_start": "2023-11-30 14:00:00",
    },
]


class FakeEnvironment:
    def __init__(self, events=None):
        self.events = list(EVENTS if events is None else events)
        self.emails = []

    def search_events(self, **_arguments):
        return json.dumps(self.events)

    def send_email(self, recipient, subject, body):
        self.emails.append(
            {"sender/recipient": recipient, "subject": subject, "body": body}
        )
        return "Email sent successfully."

    def snapshot(self):
        return {"emails_added": [row.copy() for row in self.emails]}


def make_controller(environment, search_events_fn=None):
    return ReactPlannerController(
        task_id="react-task",
        task="Send the reminder for the first event on November 30.",
        search_events_fn=search_events_fn or environment.search_events,
        send_email_fn=environment.send_email,
        all_events_fn=lambda: [row.copy() for row in environment.events],
        list_sent_emails_fn=lambda: [row.copy() for row in environment.emails],
        state_snapshot_fn=environment.snapshot,
    )


class TestReactPlannerAgent(unittest.TestCase):
    def test_valid_plan_is_parsed(self):
        self.assertEqual(parse_react_plan(json.dumps(PLAN)), PLAN)
        with self.assertRaises(ReactPlanError):
            parse_react_plan({**PLAN, "recipient": "a@example.com"})

    def test_invalid_json_stops_before_send(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)

        trace = controller.run("not json")

        self.assertEqual(trace["terminal_status"], "failed")
        self.assertEqual(environment.emails, [])

    def test_search_observation_can_feed_select_event(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)

        controller.search_events(
            query="",
            time_min="2023-11-30 00:00:00",
            time_max="2023-11-30 23:59:59",
        )
        result = json.loads(controller.select_event("001"))

        self.assertTrue(result["ok"])
        self.assertEqual(controller.memory.target_event["event_id"], "001")

    def test_wrong_event_is_blocked(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)
        controller.search_events()

        result = json.loads(controller.select_event("002"))

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "selected_event_is_not_earliest")

    def test_send_without_select_event_is_blocked(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)

        result = controller.send_email(
            "a@example.com", "Morning Meeting", PLAN["email_body"]
        )

        self.assertEqual(json.loads(result)["code"], "blocked")
        self.assertEqual(environment.emails, [])

    def test_correct_selection_sends_once(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)
        controller.search_events()
        controller.select_event("001")

        result = json.loads(
            controller.send_email("a@example.com", "Morning Meeting", PLAN["email_body"])
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(environment.emails), 1)

    def test_duplicate_send_is_blocked(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)
        controller.search_events()
        controller.select_event("001")
        controller.send_email("a@example.com", "Morning Meeting", PLAN["email_body"])

        result = json.loads(
            controller.send_email("a@example.com", "Morning Meeting", PLAN["email_body"])
        )

        self.assertFalse(result["ok"])
        self.assertEqual(len(environment.emails), 1)

    def test_no_matching_activity_completes_without_action(self):
        environment = FakeEnvironment(events=[])
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)
        controller.search_events()

        self.assertEqual(controller.memory.status, "completed_without_action")
        self.assertEqual(environment.emails, [])

    def test_incomplete_candidate_set_blocks_send(self):
        events = [
            {
                "event_id": f"00{i}",
                "event_name": f"Event {i}",
                "participant_email": f"p{i}@example.com",
                "event_start": f"2023-11-30 {8 + i:02d}:00:00",
            }
            for i in range(1, 7)
        ]
        environment = FakeEnvironment(events=events)
        controller = make_controller(
            environment,
            search_events_fn=lambda **_arguments: json.dumps(events[:5]),
        )
        controller.memory.set_plan(PLAN)
        controller.search_events()

        result = json.loads(controller.select_event("001"))
        self.assertFalse(result["ok"])
        self.assertEqual(controller.memory.candidate_completeness, "incomplete")
        self.assertEqual(environment.emails, [])

    def test_trace_contains_planner_react_memory_and_monitor_fields(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)
        controller.search_events()
        controller.select_event("001")
        controller.send_email("a@example.com", "Morning Meeting", PLAN["email_body"])

        trace = controller._trace(PLAN, [], "success", "success")

        for field in (
            "planner_output",
            "react_steps",
            "tool_calls",
            "tool_observations",
            "memory_transitions",
            "monitor_decisions",
            "selected_event",
            "final_state",
            "terminal_status",
        ):
            self.assertIn(field, trace)
        self.assertEqual(trace["terminal_status"], "success")


if __name__ == "__main__":
    unittest.main()
