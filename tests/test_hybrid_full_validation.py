"""Tests for the full-size P1/P2 validation machinery (declared in docs/hybrid-gates-v2-20261107.md)."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from atomwise_ris.hybrid import full_validation as fv
from atomwise_ris.physics.network import CoupledNetwork
from atomwise_ris.hybrid.embedded import EmbeddedRISResponse
from atomwise_ris.hybrid.evidence import audit_full_size_p1_p2, audit_legacy_gate_evidence
from atomwise_ris.hybrid.operator import HybridCoupledNetwork
from atomwise_ris.hybrid.ray_coupling import (
    build_spherical_dipole_operators, build_spherical_nec_external_operators, external_field_vectors,
    impressed_segment_currents, open_gap_scatter_impedance)

HAS_PYNEC = importlib.util.find_spec("PyNEC") is not None
ROOT = Path(__file__).resolve().parents[1]


def synthetic_response(n=3, s=5, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.03 * j, 1.0] for j in range(n)])
    dl = np.full(n * s, 0.024 / s)
    seg = np.vstack([c + np.column_stack([np.zeros(s), np.zeros(s), (np.arange(s) - s // 2) * 0.024 / s]) for c in centers])
    iz = rng.normal(size=(n * s, n)) + 1j * rng.normal(size=(n * s, n))
    iz[:s, 0] = np.sin(np.pi * (np.arange(s) + 0.5) / s) * (1 + 0.1j)
    z = np.diag(np.full(n, 60 - 60j)) + 0.1 * rng.normal(size=(n, n))
    z = (z + z.T) / 2
    return EmbeddedRISResponse(z_ris=z, y_ris=np.linalg.inv(z), segment_centers=seg, segment_lengths=dl,
                               iy=iz, iz=iz, wire_centers=centers)


class TestNumerics(unittest.TestCase):
    def test_declared_tolerances_are_not_loosened(self):
        # Mirrors docs/hybrid-gates-v2-20261107.md; change both together only before running gates.
        self.assertEqual(fv.THRESHOLDS_P1["P1.1_loaded_current_vector_relative_max"], 1e-6)
        self.assertEqual(fv.THRESHOLDS_P1["P1.2_reduced_channel_relative_max"], 1e-6)
        self.assertEqual(fv.THRESHOLDS_P1["P1.3_woodbury_field_relative_max"], 1e-6)
        self.assertEqual(fv.THRESHOLDS_P1["P1.3_woodbury_se_abs_max_bps_hz"], 1e-4)
        self.assertEqual(fv.THRESHOLDS_P1["P1.4_plane_wave_short_circuit_relative_max"], 1e-3)
        self.assertEqual(fv.THRESHOLDS_P1["P1.5_open_circuit_identity_relative_max"], 1e-5)
        self.assertEqual(fv.THRESHOLDS_P1["P1.6_plane_wave_loaded_current_relative_max"], 1e-3)
        self.assertEqual(fv.THRESHOLDS_P2["P2.1_channel_relative_error_max"], 0.05)
        self.assertEqual(fv.THRESHOLDS_P2["P2.2_worst_user_se_abs_diff_max_bps_hz"], 0.5)
        self.assertEqual(fv.THRESHOLDS_P2["P2.3_spearman_min"], 0.95)
        self.assertEqual(fv.THRESHOLDS_P2["P2.4_regret_max_bps_hz"], 0.5)
        doc = (ROOT / "docs/hybrid-gates-v2-20261107.md").read_text()
        for text in ("<= 1e-6", "max <= 0.05", "max <= 0.5 bit/s/Hz", ">= 0.95", "<= 1e-4 bit/s/Hz"):
            self.assertIn(text, doc)

    def test_spearman_handles_ties_and_monotone(self):
        self.assertAlmostEqual(fv.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(fv.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        np.testing.assert_allclose(fv.rankdata([1.0, 1.0, 3.0]), [1.5, 1.5, 3.0])

    def test_criterion_nan_never_passes(self):
        self.assertFalse(fv.criterion("x", float("nan"), 1.0, "<=")["passed"])
        self.assertTrue(fv.criterion("x", 0.5, 1.0, "<=")["passed"])
        self.assertFalse(fv.criterion("x", 0.0, 0.0, ">")["passed"])

    def test_declared_states_are_deterministic_and_distinct(self):
        a, b = fv.declared_states(64), fv.declared_states(64)
        self.assertEqual(list(a), ["all_off", "all_on", "alternating", "R1", "R2", "R3", "R4", "R5"])
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertEqual(len({a[k].tobytes() for k in a}), len(a))

    def test_network_currents_satisfy_the_circuit_equation(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        t = rng.normal(size=8) + 5.0 + 1j * rng.normal(size=8)
        i = fv.network_currents(z, t, [0, 3, 7])
        v = (z + np.diag(t)) @ i
        np.testing.assert_allclose(v, np.eye(8)[:, [0, 3, 7]], atol=1e-12)

    def test_assembly_reproduces_build_spherical_dipole_operators(self):
        resp = synthetic_response()
        bs = np.array([[-0.5, 0.02 * m, 1.0] for m in range(4)])
        ue = np.array([[-0.4, 0.3, 0.98], [-0.48, 0.38, 1.03]])
        ref = build_spherical_dipole_operators(resp, bs, ue)
        blocks = fv.spherical_blocks(resp, bs, ue)
        mine = fv.assemble_operators(resp.z_ris, blocks["z_re"], fv.z_ee_as_in_code_under_test(blocks))
        for a, b in zip(mine, ref):
            self.assertLess(fv.rel_err(a, b), 1e-12)
        # The full-external variant differs from the code under test (BS-BS coupling is not negligible).
        full = fv.assemble_operators(resp.z_ris, blocks["z_re"], fv.z_ee_full_external(blocks))
        self.assertGreater(fv.rel_err(full[2], ref[2]), 1e-3)

    def test_jc_serialises_complex_and_numpy(self):
        out = fv.jc({"a": np.array([1 + 2j]), "b": np.float64(2.0), "c": (1, 2j)})
        json.dumps(out)
        self.assertEqual(out["a"], {"re": [1.0], "im": [2.0]})


def synthetic_segment_operator(n=3, s=5, seed=3):
    """A symmetric complex segment-level admittance S with its consistent port data (ports = centre segments)."""
    rng = np.random.default_rng(seed)
    m = n * s
    a = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    sym = (a + a.T) / 20.0 + np.diag(rng.normal(size=m) + 5 + 1j)
    ports = np.arange(n) * s + s // 2
    iy = sym[:, ports]
    z_ris = np.linalg.inv(sym[np.ix_(ports, ports)])
    centers = np.array([[0.0, 0.03 * j, 1.0] for j in range(n)])
    seg = np.vstack([c + np.column_stack([np.zeros(s), np.zeros(s), (np.arange(s) - s // 2) * 0.024 / s]) for c in centers])
    resp = EmbeddedRISResponse(z_ris=z_ris, y_ris=sym[np.ix_(ports, ports)], segment_centers=seg,
                               segment_lengths=np.full(m, 0.024 / s), iy=iy, iz=iy @ z_ris, wire_centers=centers)
    return resp, sym, ports


class TestOpenGapScatter(unittest.TestCase):
    def test_open_gap_impedance_matches_constrained_segment_inverse(self):
        resp, sym, ports = synthetic_segment_operator()
        rng = np.random.default_rng(5)
        e = rng.normal(size=(resp.segment_lengths.size, 4)) + 1j * rng.normal(size=(resp.segment_lengths.size, 4))
        u = resp.segment_lengths[:, None] * e
        i_sc = sym @ u
        dz = open_gap_scatter_impedance(resp, e, i_sc)
        # Reference: admittance of the structure with the port currents constrained to zero.
        s_pp = sym[np.ix_(ports, ports)]
        s_open = sym - sym[:, ports] @ np.linalg.solve(s_pp, sym[ports, :])
        ref = -u.T @ s_open @ u
        np.testing.assert_allclose(dz, 0.5 * (ref + ref.T), rtol=1e-10, atol=1e-14)
        # And the induced current has zero port current.
        i_oc = i_sc - resp.iz @ (resp.iy.T @ u)
        np.testing.assert_allclose(i_oc[ports], 0.0, atol=1e-10)

    def test_operators_reduce_to_direct_term_when_ports_open(self):
        resp, sym, _ = synthetic_segment_operator()
        ext = synthetic_response(n=6, s=5, seed=9)
        ext.wire_centers[:] = np.array([[-0.5, 0.02 * m, 1.0] for m in range(6)])
        ext.segment_centers[:] = np.vstack([c + (ext.segment_centers[i * 5:(i + 1) * 5] - ext.wire_centers[0] * 0 - ext.segment_centers[i * 5:(i + 1) * 5].mean(0)) for i, c in enumerate(ext.wire_centers)])
        ext.z_ris[:] = np.diag(np.full(6, 60 - 55j)) + 0.5 * (np.ones((6, 6)) - np.eye(6))
        e = external_field_vectors(resp, ext)
        i_sc = sym @ (resp.segment_lengths[:, None] * e)
        c_in, c_out, h_dir, a, info = build_spherical_nec_external_operators(resp, ext, 4, i_sc)
        self.assertTrue(info["open_gap_scatter_included"])
        net = HybridCoupledNetwork(a, c_out, c_in, h_dir, np.array([1e12, 1e12], dtype=complex))
        np.testing.assert_allclose(net.channel(np.zeros(3, dtype=int)), h_dir, rtol=1e-8, atol=1e-14)
        # Without the scatter term the external block is exactly the NEC external-alone matrix.
        *_, info_a = build_spherical_nec_external_operators(resp, ext, 4, None)
        np.testing.assert_allclose(info_a["z_ee"], ext.z_ris)
        self.assertFalse(info_a["open_gap_scatter_included"])
        self.assertGreater(np.linalg.norm(info["z_ee"] - ext.z_ris), 0.0)

    def test_builders_take_no_truth_argument(self):
        import inspect
        for fn in (build_spherical_nec_external_operators, open_gap_scatter_impedance, external_field_vectors):
            names = " ".join(inspect.signature(fn).parameters)
            self.assertNotIn("z_full", names)
            self.assertNotIn("truth", names)


class TestEvidenceRegistration(unittest.TestCase):
    def _write(self, d, gate, passed, smoke=False, thresholds=None, seg=81):
        sub = Path(d) / "outputs/runs/hybrid_gates_v2_20261107" / f"seg{seg}"
        sub.mkdir(parents=True, exist_ok=True)
        base = fv.THRESHOLDS_P1 if gate == "P1" else fv.THRESHOLDS_P2
        (sub / f"{gate.lower()}.json").write_text(json.dumps({
            "criteria": [{"id": f"{gate}.x", "passed": passed}], "passed": passed, "smoke": smoke,
            "thresholds": base if thresholds is None else thresholds, "segments_per_wire": seg}))

    def test_nothing_run_means_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            audit = audit_full_size_p1_p2(d)
        self.assertEqual((audit["P1"], audit["P2"]), ("not_run", "not_run"))

    def test_failed_smoke_and_loosened_artifacts_never_register_as_pass(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "P1", False)
            self._write(d, "P2", True, smoke=True)
            audit = audit_full_size_p1_p2(d)
        self.assertEqual(audit["P1"], "failed_or_invalid_at_81")
        self.assertEqual(audit["P2"], "failed_or_invalid_at_81")
        loosened = dict(fv.THRESHOLDS_P2, **{"P2.1_channel_relative_error_max": 0.5})
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "P2", True, thresholds=loosened)
            self.assertEqual(audit_full_size_p1_p2(d)["P2"], "failed_or_invalid_at_81")

    def test_pass_requires_honest_artifact_and_reports_mesh_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "P1", True)
            self.assertEqual(audit_full_size_p1_p2(d)["P1"], "passed_at_81_only")
            self._write(d, "P1", True, seg=161)
            self.assertEqual(audit_full_size_p1_p2(d)["P1"], "passed_at_81_and_161")

    def test_fix1_rerun_requires_the_fixed_builder(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "P2", True)
            base = Path(d) / "outputs/runs/hybrid_gates_v2_20261107"
            (base / "seg81").rename(base / "fix1_seg81")
            fixed = "fixed_nec_external_plus_open_gap"
            self.assertEqual(audit_full_size_p1_p2(d, prefix="fix1_", required_p2_builder=fixed)["P2"], "failed_or_invalid_at_81")
            f = base / "fix1_seg81" / "p2.json"
            data = json.loads(f.read_text())
            data["builder"] = fixed
            f.write_text(json.dumps(data))
            self.assertEqual(audit_full_size_p1_p2(d, prefix="fix1_", required_p2_builder=fixed)["P2"], "passed_at_81_only")
            self.assertEqual(audit_full_size_p1_p2(d)["P2"], "not_run")  # un-prefixed audit ignores fix1 dirs

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/runs/hybrid_gates/gate_P1.json').exists(), 'needs the packaged data artifact')
    def test_real_repo_audit_never_authorizes_optimizer(self):
        audit = audit_legacy_gate_evidence(ROOT)
        self.assertFalse(audit["optimizer_authorized"])
        self.assertFalse(audit["training_authorized"])
        self.assertIn(audit["full_size_p1_p2_v2"]["P1"].split("_")[0], ("not", "failed", "passed"))


@unittest.skipUnless(HAS_PYNEC, "PyNEC is required for direct NEC probes")
class TestDirectNecSmall(unittest.TestCase):
    """3x3 RIS + 4 BS + 2 UE, 11 segments: cheap enough to run everywhere PyNEC exists."""

    @classmethod
    def setUpClass(cls):
        from atomwise_ris.physics.nec import geometry
        g = geometry(3)
        cls.centers, cls.lengths = np.asarray(g.centers), np.asarray(g.lengths)
        cls.seg = 11
        cls.n = 9
        cls.z, _ = fv.task_impedance(cls.centers, cls.lengths, fv.RADIUS, fv.FREQUENCY, cls.seg)
        cls.exc = list(range(15))

    def solve(self, term, ports):
        return fv.task_loaded_solve(self.centers, self.lengths, fv.RADIUS, fv.FREQUENCY, self.seg, term, ports)

    def test_loaded_nec_matches_port_network(self):
        bits = fv.declared_states(self.n)["R1"]
        term = fv.terminations_for(bits)
        sol = self.solve(term, self.exc)
        pred = fv.network_currents(self.z, term, self.exc)
        self.assertLess(max(fv.rel_err(sol[:, c], pred[:, c]) for c in range(15)), 1e-9)

    def test_excitations_do_not_accumulate_between_runs(self):
        term = fv.terminations_for(fv.declared_states(self.n)["alternating"])
        both = self.solve(term, [10, 3])
        np.testing.assert_allclose(both[:, 0], self.solve(term, [10])[:, 0], rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(both[:, 1], self.solve(term, [3])[:, 0], rtol=1e-9, atol=1e-15)

    def test_wrong_termination_is_detected(self):
        """Negative control: a missing 50 ohm LD on one BS port must show up as a large error."""
        bits = fv.declared_states(self.n)["R2"]
        term = fv.terminations_for(bits)
        wrong = term.copy()
        wrong[self.n + 1] = 0.0  # drop the BS-1 termination in the NEC deck only
        sol = self.solve(wrong, [self.n])
        pred = fv.network_currents(self.z, term, [self.n])
        self.assertGreater(fv.rel_err(sol[:, 0], pred[:, 0]), 1e-3)

    def test_impressed_sources_superpose_and_reproduce_plane_wave(self):
        from atomwise_ris.hybrid.embedded import extract_ris_response
        cells = self.centers[:self.n]
        resp = extract_ris_response(cells, segments_per_wire=self.seg)
        m = self.n * self.seg
        rng = np.random.default_rng(1)
        v = rng.normal(size=(m, 2)) + 1j * rng.normal(size=(m, 2))
        both = impressed_segment_currents(cells, v, segments_per_wire=self.seg)
        one = impressed_segment_currents(cells, v[:, :1], segments_per_wire=self.seg)
        np.testing.assert_allclose(both[:, 0], one[:, 0], rtol=1e-9, atol=1e-14)
        # superposition across segments: S(v) equals the segment-admittance applied to v via iy columns at ports
        theta, phi = np.deg2rad(70.0), np.deg2rad(30.0)
        k = 2 * np.pi * fv.FREQUENCY / fv.C0 if hasattr(fv, "C0") else 2 * np.pi * fv.FREQUENCY / 299792458.0
        k_arr = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        e_z = -np.sin(theta) * np.exp(1j * k * (resp.segment_centers @ k_arr))
        cur = impressed_segment_currents(cells, (resp.segment_lengths * e_z)[:, None], segments_per_wire=self.seg)[:, 0]
        direct = fv.task_plane_wave(cells, np.full(self.n, fv.WIRE_LENGTH), fv.RADIUS, fv.FREQUENCY, self.seg, None, [(70.0, 30.0)])[:, 0]
        mid = np.arange(self.n) * self.seg + self.seg // 2
        self.assertLess(fv.rel_err(cur[mid], direct), 2e-3)

    def test_fixed_hybrid_beats_legacy_without_using_truth(self):
        from atomwise_ris.hybrid.embedded import extract_ris_response
        from atomwise_ris.hybrid.ray_coupling import build_spherical_dipole_operators
        c = self.centers
        n = self.n
        cells, bs, ue = c[:n], c[n:n + 4], c[n + 4:]
        resp = extract_ris_response(cells, segments_per_wire=self.seg)
        ext = extract_ris_response(c[n:], segments_per_wire=self.seg)
        e = external_field_vectors(resp, ext)
        i_sc = impressed_segment_currents(cells, resp.segment_lengths[:, None] * e, segments_per_wire=self.seg)
        c_in, c_out, h_dir, a, info = build_spherical_nec_external_operators(resp, ext, 4, i_sc)
        fixed = HybridCoupledNetwork(a, c_out, c_in, h_dir, fv.LOADS)
        lc = build_spherical_dipole_operators(resp, bs, ue)
        legacy = HybridCoupledNetwork(lc[3], lc[1], lc[0], lc[2], fv.LOADS)
        truth = CoupledNetwork(self.z, fv.LOADS)
        bits = fv.declared_states(n)["R2"]
        err_fixed = fv.rel_err(fixed.channel(bits), truth.channel(bits))
        err_legacy = fv.rel_err(legacy.channel(bits), truth.channel(bits))
        self.assertLess(err_fixed, 0.1)
        self.assertLess(err_fixed, err_legacy / 3)
        # the open-gap shift of the external block should resemble the truth difference (test-only comparison)
        self.assertLess(fv.rel_err(info["delta_z_open_gap"], self.z[n:, n:] - ext.z_ris), 0.25)

    def test_runner_smoke_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            cmd = [sys.executable, str(ROOT / "scripts/hybrid_gates_v2.py"), "--segments", "11", "--side", "3",
                   "--jobs", "2", "--flip-single", "4", "--flip-pair", "1", "7", "--smoke", "--out", d]
            done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            summary = json.loads((Path(d) / "summary.json").read_text())
            crit = {c["id"]: c for c in summary["P1"]["criteria"]}
            for cid in ("P1.1", "P1.2", "P1.3a", "P1.3b"):
                self.assertTrue(crit[cid]["passed"], crit[cid])
            self.assertTrue(summary["smoke"])
            self.assertFalse(summary["optimizer_authorized"])
            again = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
            self.assertNotEqual(again.returncode, 0)  # refuses to overwrite
            reeval = subprocess.run(cmd[:-2] + ["--out", d, "--reuse-raw", "--reeval-tag", "_re"], env=env,
                                    capture_output=True, text=True, timeout=300)
            self.assertEqual(reeval.returncode, 0, reeval.stderr[-2000:])
            self.assertTrue((Path(d) / "p2_re.json").is_file())
            fixed = subprocess.run(cmd[:-2] + ["--out", d, "--reuse-raw", "--reeval-tag", "_fix", "--builder", "fixed"],
                                   env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(fixed.returncode, 0, fixed.stderr[-2000:])
            p2f = json.loads((Path(d) / "p2_fix.json").read_text())
            self.assertEqual(p2f["builder"], "fixed_nec_external_plus_open_gap")
            self.assertEqual(p2f["thresholds"], fv.THRESHOLDS_P2)
            self.assertTrue((Path(d) / "nec_raw_fix1.npz").is_file())


if __name__ == "__main__":
    unittest.main()
