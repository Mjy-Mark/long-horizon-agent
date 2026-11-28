import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import workbench_react_planner_agent as react_module
from workbench_react_planner_agent import (
    ReactPlanError,
    ReactPlannerController,
    normalize_time_bound,
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


def search_complete(controller):
    return controller.search_events(
        query="",
        time_min="2023-11-30 00:00:00",
        time_max="2023-11-30 23:59:59",
    )


class TestReactPlannerAgent(unittest.TestCase):
    def test_normalize_time_bound_expands_dates_and_preserves_timestamps(self):
        self.assertEqual(
            normalize_time_bound("2023-11-30", "min"),
            "2023-11-30 00:00:00",
        )
        self.assertEqual(
            normalize_time_bound("2023-11-30", "max"),
            "2023-11-30 23:59:59",
        )
        for timestamp in ("2023-11-30 08:15:00", "2023-11-30T08:15:00Z"):
            self.assertEqual(normalize_time_bound(timestamp, "min"), timestamp)
            self.assertEqual(normalize_time_bound(timestamp, "max"), timestamp)
        self.assertIsNone(normalize_time_bound(None, "min"))

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

    def test_planner_body_must_match_user_task(self):
        environment = FakeEnvironment()
        controller = ReactPlannerController(
            task_id="react-task",
            task="Send the reminder on November 30 and tell them 'Remember to attend this event.'",
            search_events_fn=environment.search_events,
            send_email_fn=environment.send_email,
            all_events_fn=lambda: [row.copy() for row in environment.events],
            list_sent_emails_fn=lambda: [row.copy() for row in environment.emails],
            state_snapshot_fn=environment.snapshot,
        )
        bad_plan = {**PLAN, "email_body": "Wrong body"}

        trace = controller.run(bad_plan)

        self.assertEqual(trace["terminal_status"], "failed")
        self.assertEqual(environment.emails, [])

    def test_planner_date_must_match_user_task(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        bad_plan = {**PLAN, "target_date": "2023-12-21"}

        trace = controller.run(bad_plan)

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

    def test_search_records_requested_and_effective_windows(self):
        environment = FakeEnvironment()
        received = {}

        def search_events(**arguments):
            received.update(arguments)
            return json.dumps(environment.events)

        controller = make_controller(environment, search_events_fn=search_events)
        controller.memory.set_plan(PLAN)

        observation = json.loads(
            controller.search_events(
                query="",
                time_min="2023-11-30",
                time_max="2023-11-30",
            )
        )
        trace = controller._trace(PLAN, [], "blocked", "success")

        expected_requested = {
            "time_min": "2023-11-30",
            "time_max": "2023-11-30",
        }
        expected_effective = {
            "time_min": "2023-11-30 00:00:00",
            "time_max": "2023-11-30 23:59:59",
        }
        self.assertEqual(received, {"query": "", **expected_effective})
        self.assertEqual(observation["requested_window"], expected_requested)
        self.assertEqual(observation["effective_window"], expected_effective)
        self.assertEqual(trace["requested_window"], expected_requested)
        self.assertEqual(trace["effective_window"], expected_effective)

    def test_wrong_event_is_blocked(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)
        search_complete(controller)

        result = json.loads(controller.select_event("002"))

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "selected_event_is_not_earliest")

    def test_selected_event_must_match_environment_facts(self):
        environment = FakeEnvironment()
        tampered_events = [row.copy() for row in environment.events]
        tampered_events[0]["participant_email"] = "spoofed@example.com"
        controller = make_controller(
            environment,
            search_events_fn=lambda **_arguments: json.dumps(tampered_events),
        )
        controller.memory.set_plan(PLAN)
        search_complete(controller)

        result = json.loads(controller.select_event("001"))

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "selected_event_environment_mismatch")
        self.assertIsNone(controller.memory.target_event)

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
        search_complete(controller)
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
        search_complete(controller)
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
        search_complete(controller)

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
        search_complete(controller)

        result = json.loads(controller.select_event("001"))
        self.assertFalse(result["ok"])
        self.assertEqual(controller.memory.candidate_completeness, "incomplete")
        self.assertEqual(environment.emails, [])

    def test_trace_contains_planner_react_memory_and_monitor_fields(self):
        environment = FakeEnvironment()
        controller = make_controller(environment)
        controller.memory.set_plan(PLAN)
        search_complete(controller)
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

    def test_controller_run_executes_fake_react_loop(self):
        environment = FakeEnvironment()
        controller = ReactPlannerController(
            task_id="react-task",
            task="Send the reminder on November 30 and tell them 'Remember to attend this event.'",
            search_events_fn=environment.search_events,
            send_email_fn=environment.send_email,
            all_events_fn=lambda: [row.copy() for row in environment.events],
            list_sent_emails_fn=lambda: [row.copy() for row in environment.emails],
            state_snapshot_fn=environment.snapshot,
            model=object(),
        )

        class FakeAgent:
            def __init__(self, model, tools, max_steps):
                self.tools = {item.name: item for item in tools}
                self.prompt_templates = {}
                self.memory = SimpleNamespace(get_full_steps=lambda: [])

            def run(self, task, return_full_result=True):
                self.tools["search_events"](
                    query="",
                    time_min="2023-11-30 00:00:00",
                    time_max="2023-11-30 23:59:59",
                )
                self.tools["select_event"]("001")
                self.tools["send_email"](
                    "a@example.com", "Morning Meeting", PLAN["email_body"]
                )
                return SimpleNamespace(state="success", steps=[])

        with patch.object(react_module, "ToolCallingAgent", FakeAgent):
            trace = controller.run(PLAN)

        self.assertEqual(trace["terminal_status"], "success")
        self.assertEqual(
            [call["tool"] for call in trace["tool_calls"]],
            ["search_events", "select_event", "send_email"],
        )
        self.assertEqual(trace["max_steps"], 8)
        self.assertEqual(len(environment.emails), 1)


if __name__ == "__main__":
    unittest.main()
