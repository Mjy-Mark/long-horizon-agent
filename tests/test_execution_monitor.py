import unittest

from execution_monitor import (
    ExecutionMonitor,
    MonitorViolation,
    make_send_fingerprint,
)
from planning_memory import TaskMemory


EVENT = {
    "event_id": "001",
    "event_name": "Morning Meeting",
    "participant_email": "a@example.com",
    "event_start": "2023-11-30T09:30:00",
}


def prepared_memory():
    memory = TaskMemory("task-1", "2023-11-30")
    memory.set_target_event(EVENT)
    memory.prepare_email("a@example.com", "Morning Meeting", "Reminder")
    return memory


class TestExecutionMonitor(unittest.TestCase):
    def test_missing_target_is_blocked_before_send(self):
        memory = TaskMemory("task-1", "2023-11-30")
        monitor = ExecutionMonitor()
        calls = []

        with self.assertRaises(MonitorViolation) as context:
            monitor.guarded_send_email(
                memory,
                lambda *args: calls.append(args),
                "a@example.com",
                "Morning Meeting",
                "Reminder",
            )

        self.assertEqual(context.exception.reason, "target_event_missing")
        self.assertEqual(calls, [])
        self.assertEqual(monitor.send_count, 0)

    def test_wrong_recipient_is_blocked(self):
        memory = prepared_memory()
        monitor = ExecutionMonitor()
        calls = []

        with self.assertRaises(MonitorViolation) as context:
            monitor.guarded_send_email(
                memory,
                lambda *args: calls.append(args),
                "wrong@example.com",
                "Morning Meeting",
                "Reminder",
            )

        self.assertEqual(context.exception.reason, "recipient_mismatch")
        self.assertEqual(calls, [])
        self.assertFalse(memory.send_attempted)

    def test_wrong_subject_is_blocked(self):
        memory = prepared_memory()
        monitor = ExecutionMonitor()

        with self.assertRaises(MonitorViolation) as context:
            monitor.guarded_send_email(
                memory,
                lambda *args: None,
                "a@example.com",
                "Wrong subject",
                "Reminder",
            )

        self.assertEqual(context.exception.reason, "subject_mismatch")
        self.assertFalse(memory.send_attempted)

    def test_confirmed_send_cannot_be_repeated(self):
        memory = prepared_memory()
        monitor = ExecutionMonitor()
        calls = []

        monitor.guarded_send_email(
            memory,
            lambda *args: calls.append(args) or "ok",
            "a@example.com",
            "Morning Meeting",
            "Reminder",
        )
        memory.mark_send_confirmed()

        with self.assertRaises(MonitorViolation) as context:
            monitor.guarded_send_email(
                memory,
                lambda *args: calls.append(args),
                "a@example.com",
                "Morning Meeting",
                "Reminder",
            )

        self.assertEqual(context.exception.reason, "send_already_confirmed")
        self.assertEqual(len(calls), 1)

    def test_existing_fingerprint_blocks_duplicate(self):
        fingerprint = make_send_fingerprint(
            "001", "a@example.com", "Morning Meeting", "Reminder"
        )
        memory = prepared_memory()
        monitor = ExecutionMonitor([fingerprint])

        with self.assertRaises(MonitorViolation) as context:
            monitor.guarded_send_email(
                memory,
                lambda *args: None,
                "a@example.com",
                "Morning Meeting",
                "Reminder",
            )

        self.assertEqual(context.exception.reason, "duplicate_send_fingerprint")
        self.assertFalse(memory.send_attempted)

    def test_transport_exception_does_not_allow_a_second_attempt(self):
        memory = prepared_memory()
        monitor = ExecutionMonitor()

        def raises(*args):
            raise RuntimeError("transport error")

        with self.assertRaises(RuntimeError):
            monitor.guarded_send_email(
                memory, raises, "a@example.com", "Morning Meeting", "Reminder"
            )

        self.assertTrue(memory.send_attempted)
        with self.assertRaises(MonitorViolation) as context:
            monitor.guarded_send_email(
                memory,
                lambda *args: None,
                "a@example.com",
                "Morning Meeting",
                "Reminder",
            )
        self.assertEqual(context.exception.reason, "send_already_attempted")
        self.assertEqual(monitor.send_count, 1)


if __name__ == "__main__":
    unittest.main()
