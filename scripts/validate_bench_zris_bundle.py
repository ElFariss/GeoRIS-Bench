"""Recompute physical diagnostics and check an assembled RIS-only impedance bundle."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read_mesh(root, bundle, n, segments):
    import h5py

    folder = root / str(segments)
    hdf = folder / "impedance.h5"
    metrics = json.loads((folder / "metrics.json").read_text())
    if (metrics["status"], metrics["elements"], metrics["segments_per_wire"], metrics["hdf5_sha256"]) != (
        "complete", n, segments, digest(hdf)
    ):
        raise ValueError(f"matrix provenance mismatch: {folder}")
    with h5py.File(hdf, "r") as f:
        d = f["Z_RIS_ohm"]
        z = d[:]
        if (z.shape, z.dtype, d.attrs["elements"], d.attrs["segments_per_wire"],
                d.attrs["frequency_hz"], d.attrs["units"]) != ((n,n), np.dtype("complex128"), n, segments, 5.53e9, "ohm"):
            raise ValueError(f"matrix metadata mismatch: {hdf}")
    if not np.isfinite(z).all():
        raise ValueError(f"nonfinite matrix: {hdf}")
    np.testing.assert_array_equal(bundle[f"segments_{segments}/Z_RIS_ohm"][:], z)
    hermitian = (z + z.conj().T) / 2
    eig = np.linalg.eigvalsh(hermitian)
    scale = max(1., float(np.max(np.abs(eig))))
    z0 = 50 * np.eye(n)
    s50 = np.linalg.solve((z + z0).T, (z - z0).T).T
    values = {
        "reciprocity_relative_fro": float(np.linalg.norm(z - z.T) / np.linalg.norm(z)),
        "minimum_hermitian_resistance_eigenvalue_ohm": float(eig[0]),
        "maximum_scattering_singular_value": float(np.linalg.svd(s50, compute_uv=False)[0]),
        "condition_number_2": float(np.linalg.cond(z)),
    }
    for k, v in values.items():
        np.testing.assert_allclose(v, metrics[k], rtol=1e-7, atol=1e-11, err_msg=k)
    checks = {
        "reciprocity": values["reciprocity_relative_fro"] <= 1e-6,
        "passivity": values["minimum_hermitian_resistance_eigenvalue_ohm"] >= -1e-6 * scale,
        "scattering_contraction": values["maximum_scattering_singular_value"] <= 1 + 1e-6,
    }
    return z, {**values, "checks": checks, "hdf5_sha256": digest(hdf),
               "metrics_sha256": digest(folder / "metrics.json"),
               "elapsed_wall_s": metrics["elapsed_wall_s"], "peak_rss_bytes": metrics["peak_rss_bytes"]}


def validate(root):
    import h5py

    geo = json.loads((root / "geometry.json").read_text())
    n = int(geo["elements"])
    if n not in (64, 128, 256) or geo["frequency_hz"] != 5.53e9:
        raise ValueError("unexpected geometry")
    combined = root / "impedance.h5"
    compare = root / "mesh_validation.json"
    reported = json.loads(compare.read_text())
    limits = {"relative_frobenius_difference": .05, "maximum_entry_difference_over_fine_frobenius": .10}
    has_161 = (root / "161" / "impedance.h5").exists()
    if n != 256 and not has_161:
        raise ValueError(f"161-segment mesh is mandatory for {n} elements")
    with h5py.File(combined, "r") as h:
        coarse, m81 = read_mesh(root, h, n, 81)
        if has_161:
            fine, m161 = read_mesh(root, h, n, 161)
        elif "segments_161" in h:
            raise ValueError("unexpected segments_161 dataset in coarse-only 256-element bundle")
    if has_161:
        if (reported["status"], reported["elements"], reported["coarse_segments"], reported["fine_segments"]) != ("complete", n, 81, 161):
            raise ValueError("unexpected mesh comparison metadata")
        norm = float(np.linalg.norm(fine))
        delta = fine - coarse
        values = {
            "relative_frobenius_difference": float(np.linalg.norm(delta)) / norm,
            "maximum_entry_difference_over_fine_frobenius": float(np.max(np.abs(delta))) / norm,
        }
        for k, v in values.items():
            np.testing.assert_allclose(v, reported[k], rtol=1e-7, atol=1e-11, err_msg=k)
        ok = all(values[k] <= lim for k, lim in limits.items())
        return {"elements": n, "status": "pass" if ok and all(m81["checks"].values()) and all(m161["checks"].values()) else "fail",
                "mesh": {**values, "limits": limits, "pass": ok}, "matrices": {"81": m81, "161": m161},
                "geometry_sha256": digest(root / "geometry.json"), "combined_sha256": digest(combined),
                "mesh_validation_sha256": digest(compare)}

    if (reported.get("status"), reported.get("elements"), reported.get("coarse_segments"),
            reported.get("fine_segments"), reported.get("fine_segments_extracted")) != (
        "coarse_only_161_not_extracted_memory_cap", 256, 81, None, False
    ):
        raise ValueError("unexpected coarse-only 256-element mesh validation metadata")
    np.testing.assert_allclose(
        m81["reciprocity_relative_fro"], reported["coarse_reciprocity_relative_fro"], rtol=1e-7, atol=1e-11
    )
    trend = reported.get("supporting_mesh_delta_trend", {})
    if set(trend.keys()) != {"64", "128"}:
        raise ValueError("256-element mesh_validation.json must include 64 and 128 supporting_mesh_delta_trend")
    ok = True
    for ref_n in ("64", "128"):
        ref_file = root.parent / f"array_{ref_n}" / "mesh_validation.json"
        if ref_file.exists():
            ref_data = json.loads(ref_file.read_text())
            for k in ("relative_frobenius_difference", "maximum_entry_difference_over_fine_frobenius", "median_diagonal_relative_difference"):
                np.testing.assert_allclose(trend[ref_n][k], ref_data[k], rtol=1e-7, atol=1e-11, err_msg=f"{ref_n}:{k}")
        ok = ok and all(trend[ref_n][k] <= lim for k, lim in limits.items())
    return {
        "elements": n,
        "status": "pass" if ok and all(m81["checks"].values()) else "fail",
        "mesh": {
            "mode": "coarse_only_161_not_extracted_memory_cap",
            "coarse_segments": 81,
            "fine_segments": None,
            "fine_segments_extracted": False,
            "supporting_mesh_delta_trend": trend,
            "limits": limits,
            "pass": ok,
        },
        "matrices": {"81": m81},
        "geometry_sha256": digest(root / "geometry.json"),
        "combined_sha256": digest(combined),
        "mesh_validation_sha256": digest(compare),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array-dir", required=True, type=Path)
    args = p.parse_args()
    result = validate(args.array_dir)
    path = args.array_dir / "numerical_validation.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
