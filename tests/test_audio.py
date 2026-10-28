import sys
import time
import unittest
from unittest import mock

import context  # noqa: F401
from visionpath.audio import AlertManager, Speaker, build_message
from visionpath.detector import Detection


def det(label="person", pos="left", prox="near", dist=None, conf=0.9):
    return Detection(label, conf, (0, 0, 10, 10), position=pos, proximity=prox, distance_m=dist)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class AlertTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.mgr = AlertManager(cooldown_s=4.0, global_interval_s=1.0, clock=self.clock)

    def test_message_text(self):
        self.assertEqual(build_message(det()), "Person on the left, near")
        self.assertEqual(build_message(det("diningtable", "center", "medium")), "Table ahead, medium distance")
        self.assertEqual(build_message(det(pos="right", dist=1.8)), "Person on the right, about 2 meters")

    def test_same_object_not_repeated_within_cooldown(self):
        self.assertIsNotNone(self.mgr.select([det()]))
        self.clock.t += 2.0
        self.assertIsNone(self.mgr.select([det()]))
        self.clock.t += 3.0                      # 5 s after first alert
        self.assertIsNotNone(self.mgr.select([det()]))

    def test_closer_object_announced_immediately(self):
        self.mgr.select([det(prox="medium")])
        self.clock.t += 1.5
        self.assertEqual(self.mgr.select([det(prox="near")]), "Person on the left, near")

    def test_global_interval(self):
        self.assertIsNotNone(self.mgr.select([det("person", "left")]))
        self.clock.t += 0.5
        self.assertIsNone(self.mgr.select([det("chair", "right")]))
        self.clock.t += 0.6
        self.assertIsNotNone(self.mgr.select([det("chair", "right")]))

    def test_most_urgent_first(self):
        msg = self.mgr.select([det("chair", "right", "medium"), det("person", "center", "near")])
        self.assertTrue(msg.startswith("Person"))

    def test_far_not_announced_by_default(self):
        self.assertIsNone(self.mgr.select([det(prox="far")]))
        mgr = AlertManager(announce_far=True, clock=self.clock)
        self.assertIsNotNone(mgr.select([det(prox="far")]))

    def test_ignored_labels(self):
        mgr = AlertManager(ignore_labels=["Person"], clock=self.clock)
        self.assertIsNone(mgr.select([det()]))

    def test_counts_alerts(self):
        self.mgr.select([det()])
        self.assertEqual(self.mgr.alerts_generated, 1)

    def test_cooldown_adjustable(self):
        self.mgr.select([det()])
        self.mgr.cooldown_s = 0.0
        self.clock.t += 1.1
        self.assertIsNotNone(self.mgr.select([det()]))


class SpeakerTests(unittest.TestCase):
    def test_missing_tts_does_not_crash(self):
        statuses = []
        with mock.patch.dict(sys.modules, {"pyttsx3": None}):
            sp = Speaker(on_status=statuses.append)
            sp.start()
            self.assertFalse(sp.available)
            self.assertFalse(sp.say("hello"))
            sp.stop()
        self.assertTrue(statuses and "Speech unavailable" in statuses[0])

    def test_fake_engine_speaks_and_stops(self):
        spoken = []

        class Engine:
            def setProperty(self, *a): pass
            def say(self, text): spoken.append(text)
            def runAndWait(self): pass
            def stop(self): pass

        fake = mock.Mock()
        fake.init.return_value = Engine()
        with mock.patch.dict(sys.modules, {"pyttsx3": fake}):
            sp = Speaker()
            sp.start()
            self.assertTrue(sp.available)
            sp.say("Person ahead, near")
            deadline = time.time() + 3
            while not spoken and time.time() < deadline:   # stop() discards unspoken messages by design
                time.sleep(0.01)
            sp.stop()
        self.assertEqual(spoken, ["Person ahead, near"])


if __name__ == "__main__":
    unittest.main()
