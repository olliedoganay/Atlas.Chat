import unittest

from atlas_local.run_contract import RunHub, make_run_event


class RunHubTests(unittest.TestCase):
    def test_slow_subscriber_queue_is_bounded_and_keeps_newest_events(self) -> None:
        hub = RunHub(subscriber_queue_size=2)
        subscriber = hub.subscribe("run-1")

        hub.publish("run-1", make_run_event("first", {}))
        hub.publish("run-1", make_run_event("second", {}))
        hub.publish("run-1", make_run_event("third", {}))

        self.assertEqual(subscriber.maxsize, 2)
        self.assertEqual(subscriber.get_nowait()["type"], "second")
        self.assertEqual(subscriber.get_nowait()["type"], "third")

    def test_non_positive_subscriber_size_is_clamped(self) -> None:
        hub = RunHub(subscriber_queue_size=0)

        self.assertEqual(hub.subscribe("run-1").maxsize, 1)


if __name__ == "__main__":
    unittest.main()
