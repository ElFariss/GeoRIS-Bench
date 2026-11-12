"""Compare the two disputed legacy v3 export hulls and stage the geometry audit input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, default=Path(
        "data/raw/deepmimo_scenarios/munich_specular_v3_bs_fullpaths"))
    ap.add_argument("--layout", type=Path, default=Path(
        "data/raw/acquisition/munich_specular_v3/layout.json"))
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--staged-input", type=Path, required=True,
                    help="Private input to send to the tracing host; do not publish.")
    args = ap.parse_args()
    vertices, objects = [args.export / f for f in ("vertices.npz", "objects.json")]
    with np.load(vertices, allow_pickle=False) as z:
        v = z["vertices"]
    objects_data = json.loads(objects.read_text())
    names = ["no-name-6_1165", "no-name-8_1158"]
    entries = [next(obj for obj in objects_data if obj["name"] == name) for name in names]
    faces = [obj["face_vertex_idxs"] for obj in entries]

    def face_signature(f):
        # Independent of index sharing, polygon winding and the choice of first vertex.
        points = [tuple(map(float, v[i])) for i in f]
        orientations = (points, points[::-1])
        return min(tuple(order[k:] + order[:k]) for order in orientations for k in range(len(f)))

    signatures = [sorted(face_signature(f) for f in fs) for fs in faces]
    lower = min(faces[0], key=lambda f: v[f, 2].mean())
    elevations = v[np.concatenate(faces[0]), 2]
    summary = {"names": names, "exact_coordinate_face_multiset_match": signatures[0] == signatures[1],
               "face_counts": list(map(len, faces)),
               "material_ids": [sorted(set(o["face_material_idxs"])) for o in entries],
               "z_range_m": [float(elevations.min()), float(elevations.max())],
               "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (vertices, objects, args.layout)}}
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    args.staged_input.write_text(json.dumps({"layout": json.loads(args.layout.read_text()),
                                            "footprint": v[lower, :2].tolist()}))


if __name__ == "__main__":
    main()
