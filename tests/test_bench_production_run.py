import json
import tempfile
import unittest
from pathlib import Path

from atomwise_ris.bench.production_run import (atomic_write_json, cases_by_map, check_cpu_set,
                                               stop_rule_exceeded)


class StopRule(unittest.TestCase):
    def test_natural_map_of_120_stops_at_three(self):
        self.assertFalse(stop_rule_exceeded(2, 120))
        self.assertTrue(stop_rule_exceeded(3, 120))

    def test_challenge_map_of_23_stops_at_one(self):
        self.assertFalse(stop_rule_exceeded(0, 23))
        self.assertTrue(stop_rule_exceeded(1, 23))

    def test_exactly_two_percent_does_not_stop(self):
        self.assertFalse(stop_rule_exceeded(2, 100))
        self.assertTrue(stop_rule_exceeded(3, 100))

    def test_invalid_counts(self):
        for f, t in ((1, 0), (-1, 10), (11, 10)):
            with self.assertRaises(ValueError):
                stop_rule_exceeded(f, t)


class Helpers(unittest.TestCase):
    def test_atomic_json_roundtrip_and_no_temp_left(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a" / "x.json"
            atomic_write_json(p, {"b": 1})
            self.assertEqual(json.loads(p.read_text()), {"b": 1})
            self.assertEqual([q.name for q in p.parent.iterdir()], ["x.json"])
            with self.assertRaises(ValueError):
                atomic_write_json(p, {"x": float("nan")})
            self.assertEqual(json.loads(p.read_text()), {"b": 1})
            self.assertEqual([q.name for q in p.parent.iterdir()], ["x.json"])

    def test_cases_by_map_refuses_held_out(self):
        g = cases_by_map([{"map_id": "munich", "id": "a"}, {"map_id": "rotterdam", "id": "b"},
                          {"map_id": "munich", "id": "c"}])
        self.assertEqual([c["id"] for c in g["munich"]], ["a", "c"])
        for bad in ("berlin", "utrecht"):
            with self.assertRaises(ValueError):
                cases_by_map([{"map_id": bad, "id": "x"}])

    def test_cpu_set(self):
        self.assertEqual(check_cpu_set([6, 0]), (0, 6))
        for bad in ([], [5], [0, 11]):
            with self.assertRaises(ValueError):
                check_cpu_set(bad)


if __name__ == "__main__":
    unittest.main()
