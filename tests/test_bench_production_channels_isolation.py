"""Lead regression tests (2026-11-10) for the 2026-11-09 production failure: PyNEC result-storage
leak -> memory exhaustion -> spurious case FAILED records and stop-rule trips."""
from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import build_bench_production_channels as prod_ch
from test_bench_production_channels_full import (FakeBasis, fake_physics, make_dummy_case,
                                                 make_fake_frame_fn, write_fake_trace_record)


def real_numpy_memory_error():
    try:
        np.empty(2 ** 45, dtype=np.complex128)  # 512 TiB: numpy raises its own _ArrayMemoryError
    except MemoryError as exc:
        return exc
    raise unittest.SkipTest("host allowed a 512 TiB allocation")


class Clock:
    def __init__(self):
        self.t = 1_000_000.0
        self.sleeps = []

    def time(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class Classification(unittest.TestCase):
    def test_real_numpy_allocation_error_is_transient(self):
        exc = real_numpy_memory_error()
        self.assertIn("Unable to allocate", str(exc))
        self.assertEqual(prod_ch.classify_exception(exc), "transient")

    def test_swig_unknown_exception_is_transient_other_errors_are_not(self):
        self.assertEqual(prod_ch.classify_exception(RuntimeError("Unknown exception")), "transient")
        self.assertEqual(prod_ch.classify_exception(ValueError("dense round-trip failed")), "case")
        self.assertEqual(prod_ch.classify_exception(RuntimeError("NEC geometry error")), "case")

    def test_description_has_type_and_traceback(self):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            d = prod_ch.describe_exception(exc)
        self.assertEqual(d["error_type"], "builtins.ValueError")
        self.assertIn("raise ValueError", d["traceback"])


class ForkIsolation(unittest.TestCase):
    def test_child_success(self):
        self.assertEqual(prod_ch.run_unit_in_child(lambda: None), {"ok": True})

    def test_child_exception_reports_type_and_traceback(self):
        def bad():
            raise ValueError("genuine")
        r = prod_ch.run_unit_in_child(bad)
        self.assertFalse(r["ok"])
        self.assertEqual(r["error_type"], "builtins.ValueError")
        self.assertEqual(r["classification"], "case")
        self.assertIn("genuine", r["traceback"])

    def test_child_numpy_memory_error_is_transient(self):
        r = prod_ch.run_unit_in_child(lambda: np.empty(2 ** 45, dtype=np.complex128))
        self.assertFalse(r["ok"])
        self.assertEqual(r["classification"], "transient")

    def test_child_sigkill_is_transient(self):
        r = prod_ch.run_unit_in_child(lambda: os.kill(os.getpid(), signal.SIGKILL))
        self.assertEqual(r["error_type"], "ChildKilledBySignal")
        self.assertEqual(r["classification"], "transient")

    def test_child_state_does_not_leak_into_parent(self):
        leak = []
        prod_ch.run_unit_in_child(lambda: leak.extend([bytearray(10 ** 6)] * 50))
        self.assertEqual(leak, [])


class WorkerTransientHandling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = Path(self.tmp.name)
        self.trace_out, self.ch_out, self.physics = r / "trace", r / "ch", r / "phys"
        for p in (self.trace_out, self.ch_out, self.physics):
            p.mkdir()
        self.case = make_dummy_case("munich", 0)
        write_fake_trace_record(self.trace_out, "munich", self.case["id"], "h", arrays=(64,))
        (self.trace_out / "munich" / "DONE").write_text("done\n")
        self.args = argparse.Namespace(worker_id="0", arrays="64", cpus="0,6", out=self.ch_out,
                                       physics=self.physics)
        self.unit = self.ch_out / "munich" / "cases" / self.case["id"]

    def tearDown(self):
        self.tmp.cleanup()

    def run_worker(self, frame_fn, mem_fn=lambda: 64 * 2 ** 30):
        clock = Clock()
        with patch.object(prod_ch, "check_mem_available", lambda _m: None), \
                patch.object(prod_ch.time, "time", clock.time):
            prod_ch.production_worker(self.args, {"cases": [self.case]}, "development_full", "h", "f",
                                      trace_out=self.trace_out, basis_builder=FakeBasis,
                                      frame_fn=frame_fn, input_physics_fn=fake_physics,
                                      sleep_fn=clock.sleep, mem_fn=mem_fn)
        return clock

    def test_second_basis_reexecs_worker_and_resume_finishes(self):
        """A second basis in one process hit ulimit -v in production (2026-11-10): the worker must restart."""
        write_fake_trace_record(self.trace_out, "munich", self.case["id"], "h", arrays=(64, 128))
        self.args.arrays = "64,128"

        class Reexec(BaseException):
            pass

        def reexec():
            raise Reexec()

        def run(reexec_fn):
            with patch.object(prod_ch, "check_mem_available", lambda _m: None):
                prod_ch.production_worker(self.args, {"cases": [self.case]}, "development_full", "h", "f",
                                          trace_out=self.trace_out, basis_builder=FakeBasis,
                                          frame_fn=make_fake_frame_fn(), input_physics_fn=fake_physics,
                                          sleep_fn=lambda s: None, reexec_fn=reexec_fn)

        with self.assertRaises(Reexec):
            run(reexec)
        self.assertTrue(prod_ch.is_unit_done(self.unit / "array_64.h5", self.unit / "array_64.json", "h"))
        self.assertFalse((self.unit / "array_128.h5").exists())
        self.assertFalse(list(self.unit.glob("*.FAILED.json")))
        run(reexec)  # the fresh process: 64 is done, so 128 is its first basis and no restart is needed
        self.assertTrue(prod_ch.is_unit_done(self.unit / "array_128.h5", self.unit / "array_128.json", "h"))

    def test_real_numpy_memory_error_retries_then_succeeds_without_failed(self):
        good = make_fake_frame_fn()
        calls = {"n": 0}

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise real_numpy_memory_error()
            return good(**kw)

        clock = self.run_worker(flaky)
        self.assertFalse((self.unit / "array_64.FAILED.json").exists())
        self.assertFalse((self.unit / "array_64.transient.json").exists())
        self.assertTrue(prod_ch.is_unit_done(self.unit / "array_64.h5", self.unit / "array_64.json", "h"))
        self.assertGreaterEqual(sum(clock.sleeps), 300)  # waited out the first backoff
        self.assertFalse((self.ch_out / "munich" / "FAILED").exists())

    def test_transient_exhausted_becomes_counted_failure_with_history(self):
        def always_oom(**kw):
            raise real_numpy_memory_error()

        self.run_worker(always_oom)
        rec = json.loads((self.unit / "array_64.FAILED.json").read_text())
        self.assertEqual(rec["failure_kind"], "transient_retries_exhausted")
        self.assertEqual(len(rec["attempts"]), prod_ch.MAX_TRANSIENT_ATTEMPTS)
        self.assertIn("MemoryError", rec["error_type"])
        self.assertIn("Unable to allocate", rec["error"])

    def test_genuine_error_fails_immediately_with_type_and_traceback(self):
        def bad(**kw):
            raise ValueError("dense round-trip failed")

        self.run_worker(bad)
        rec = json.loads((self.unit / "array_64.FAILED.json").read_text())
        self.assertEqual(rec["failure_kind"], "case_failure")
        self.assertEqual(rec["error_type"], "builtins.ValueError")
        self.assertIn("Traceback", rec["traceback"])
        self.assertEqual(len(rec["attempts"]), 1)

    def test_low_memory_admission_waits_instead_of_failing(self):
        levels = iter([1 * 2 ** 30, 1 * 2 ** 30])
        clock = self.run_worker(make_fake_frame_fn(), mem_fn=lambda: next(levels, 64 * 2 ** 30))
        self.assertIn(120, clock.sleeps)
        self.assertFalse((self.unit / "array_64.FAILED.json").exists())
        self.assertTrue(prod_ch.is_unit_done(self.unit / "array_64.h5", self.unit / "array_64.json", "h"))


if __name__ == "__main__":
    unittest.main()
