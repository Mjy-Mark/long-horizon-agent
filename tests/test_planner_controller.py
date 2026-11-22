import json
import unittest
from types import SimpleNamespace

from workbench_planner_agent import (
    WorkbenchPlannerController,
    extract_email_body,
    parse_plan,
)


PLAN = {
    "target_date": "2023-11-30",
    "operation": "send_email",
    "event_selection": "earliest_start",
    "subject_source": "event_name",
}


EVENTS = [
    {
        "event_id": "003",
        "event_name": "Late Meeting",
        "participant_email": "late@example.com",
        "event_start": "2023-11-30 14:30:00",
    },
    {
        "event_id": "001",
        "event_name": "Annual Budget Planning Session",
        "participant_email": "fatima.khan@atlas.com",
        "event_start": "2023-11-30 09:30:00",
    },
]


TASK = (
    "Can you send an email to attendees of the first event on November 30? "
    "Title it with the event name and tell them 'Remember to attend this event.'"
)


class FakeWorkbench:
    def __init__(self, events=None):
        self.events = list(EVENTS if events is None else events)
        self.emails = []
        self.search_calls = []
        self.send_calls = []

    def search_events(self, **arguments):
        self.search_calls.append(arguments)
        return json.dumps(self.events)

    def send_email(self, recipient, subject, body):
        self.send_calls.append((recipient, subject, body))
        self.emails.append(
            {
                "email_id": str(len(self.emails) + 1),
                "sender/recipient": recipient,
                "subject": subject,
                "body": body,
            }
        )
        return "Email sent successfully."

    def snapshot(self):
        return {"emails_added": [row.copy() for row in self.emails]}


class FakeModel:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=json.dumps(self.plan))


def make_controller(fake, model=None):
    return WorkbenchPlannerController(
        task_id="email_first_event_001",
        task=TASK,
        search_events_fn=fake.search_events,
        send_email_fn=fake.send_email,
        list_sent_emails_fn=lambda: [row.copy() for row in fake.emails],
        state_snapshot_fn=fake.snapshot,
        model=model,
    )


class TestPlannerController(unittest.TestCase):
    def test_parse_plan_requires_the_fixed_contract(self):
        self.assertEqual(parse_plan(json.dumps(PLAN)), PLAN)
        with self.assertRaises(ValueError):
            parse_plan({**PLAN, "event_id": "001"})

    def test_extract_email_body(self):
        self.assertEqual(extract_email_body(TASK), "Remember to attend this event.")

    def test_controller_sends_once_to_python_selected_event(self):
        fake = FakeWorkbench()
        trace = make_controller(fake).run(PLAN)

        self.assertTrue(trace["success"])
        self.assertEqual(trace["termination_reason"], "success")
        self.assertEqual(len(fake.send_calls), 1)
        self.assertEqual(
            fake.send_calls[0],
            (
                "fatima.khan@atlas.com",
                "Annual Budget Planning Session",
                "Remember to attend this event.",
            ),
        )
        self.assertEqual(trace["memory_snapshots"][-1]["state"]["status"], "send_confirmed")
        self.assertEqual(len(trace["monitor_checks"]), 1)
        self.assertTrue(trace["monitor_checks"][0]["allowed"])
        self.assertEqual(
            fake.search_calls[0]["time_min"], "2023-11-30 00:00:00"
        )

    def test_controller_rejects_invalid_plan_without_sending(self):
        fake = FakeWorkbench()
        invalid_plan = {**PLAN, "event_selection": "first_returned"}
        trace = make_controller(fake).run(invalid_plan)

        self.assertFalse(trace["success"])
        self.assertEqual(trace["termination_reason"], "validation_error")
        self.assertEqual(fake.send_calls, [])
        self.assertEqual(trace["tool_calls"], [])

    def test_controller_stops_when_no_event_matches(self):
        fake = FakeWorkbench(events=[])
        trace = make_controller(fake).run(PLAN)

        self.assertFalse(trace["success"])
        self.assertEqual(trace["termination_reason"], "validation_error")
        self.assertEqual(fake.send_calls, [])
        self.assertEqual(len(trace["tool_calls"]), 1)

    def test_read_only_search_can_retry_once_without_sending(self):
        fake = FakeWorkbench()
        attempts = {"count": 0}

        def flaky_search(**arguments):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary query failure")
            return fake.search_events(**arguments)

        controller = WorkbenchPlannerController(
            task_id="email_first_event_001",
            task=TASK,
            search_events_fn=flaky_search,
            send_email_fn=fake.send_email,
            list_sent_emails_fn=lambda: [row.copy() for row in fake.emails],
            state_snapshot_fn=fake.snapshot,
            max_search_retries=1,
        )
        trace = controller.run(PLAN)

        self.assertTrue(trace["success"])
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(len(fake.send_calls), 1)
        self.assertEqual(len(trace["tool_calls"]), 3)
        self.assertIn("error", trace["tool_calls"][0])

    def test_controller_can_obtain_plan_from_model_without_tools(self):
        fake = FakeWorkbench()
        model = FakeModel(PLAN)
        trace = make_controller(fake, model=model).run()

        self.assertTrue(trace["success"])
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn("tools_to_call_from", model.calls[0][1])
        self.assertIn("response_format", model.calls[0][1])


if __name__ == "__main__":
    unittest.main()
