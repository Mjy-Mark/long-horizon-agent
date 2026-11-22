import json
import unittest

from planning_memory import TaskMemory, TaskMemoryError, select_first_event


class TestPlanningMemory(unittest.TestCase):
    def setUp(self):
        self.events = [
            {
                "event_id": "002",
                "event_name": "Afternoon Meeting",
                "participant_email": "b@example.com",
                "event_start": "2023-11-30 14:00:00",
            },
            {
                "event_id": "001",
                "event_name": "Morning Meeting",
                "participant_email": "a@example.com",
                "event_start": "2023-11-30 09:30:00",
            },
            {
                "event_id": "003",
                "event_name": "Next Day Meeting",
                "participant_email": "c@example.com",
                "event_start": "2023-12-01 08:00:00",
            },
        ]

    def test_select_earliest_event_from_unordered_results(self):
        result = select_first_event(json.dumps(self.events), "2023-11-30")

        self.assertEqual(result["event_id"], "001")
        self.assertEqual(result["event_name"], "Morning Meeting")
        self.assertEqual(result["participant_email"], "a@example.com")

    def test_no_matching_event(self):
        with self.assertRaises(ValueError):
            select_first_event("[]", "2023-11-30")

    def test_same_start_time_is_deterministic(self):
        events = [
            {
                "event_id": "002",
                "event_name": "Event B",
                "participant_email": "b@example.com",
                "event_start": "2023-11-30 09:30:00",
            },
            {
                "event_id": "001",
                "event_name": "Event A",
                "participant_email": "a@example.com",
                "event_start": "2023-11-30 09:30:00",
            },
        ]

        result = select_first_event(json.dumps(events), "2023-11-30")
        self.assertEqual(result["event_id"], "001")

    def test_invalid_event_rows_are_ignored(self):
        events = [{"event_id": "bad", "event_start": "not-a-date"}] + self.events
        result = select_first_event(json.dumps(events), "2023-11-30")
        self.assertEqual(result["event_id"], "001")

    def test_memory_saves_and_updates_target_event(self):
        memory = TaskMemory("task-1", "2023-11-30")
        memory.set_target_event(self.events[0])
        memory.set_target_event(self.events[1])

        self.assertEqual(memory.target_event["event_id"], "001")
        self.assertEqual(memory.status, "event_selected")

    def test_memory_rejects_email_without_target_or_with_wrong_fields(self):
        memory = TaskMemory("task-1", "2023-11-30")
        with self.assertRaises(TaskMemoryError):
            memory.prepare_email("a@example.com", "Morning Meeting", "Reminder")

        memory.set_target_event(self.events[1])
        with self.assertRaises(TaskMemoryError):
            memory.prepare_email("wrong@example.com", "Morning Meeting", "Reminder")
        with self.assertRaises(TaskMemoryError):
            memory.prepare_email("a@example.com", "Wrong subject", "Reminder")

    def test_memory_tracks_email_and_send_state(self):
        memory = TaskMemory("task-1", "2023-11-30")
        memory.set_target_event(self.events[1])
        memory.prepare_email("a@example.com", "Morning Meeting", "Reminder")
        memory.mark_send_attempted()
        memory.mark_send_confirmed({"email_id": "1"})

        snapshot = memory.snapshot()
        json.dumps(snapshot)
        self.assertEqual(snapshot["status"], "send_confirmed")
        self.assertTrue(snapshot["send_attempted"])
        self.assertTrue(snapshot["send_confirmed"])
        self.assertEqual(snapshot["prepared_email"]["recipient"], "a@example.com")

    def test_memory_cannot_prepare_two_emails(self):
        memory = TaskMemory("task-1", "2023-11-30")
        memory.set_target_event(self.events[1])
        memory.prepare_email("a@example.com", "Morning Meeting", "Reminder")

        with self.assertRaises(TaskMemoryError):
            memory.prepare_email("a@example.com", "Morning Meeting", "Another reminder")


if __name__ == "__main__":
    unittest.main()
