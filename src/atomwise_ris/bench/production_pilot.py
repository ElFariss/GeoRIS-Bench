"""Registered development pilot and production identities shared by the pipeline."""
import json
from pathlib import Path

from .production_freeze import sha256

PILOT1_HASH = "998083842c02870e45724313151c15c77910754372baba6669776433206eb865"
PILOT2_HASH = "5b129264c6f81d55bffd6d2edfe3db4a58b8938e3484564506779e6228fc5827"
FULL_HASH = "21d54cfbf1db69721aedf89144afb95a38a5c7f0bafc9c0ee24273d445203adf"
CHALLENGE_HASH = "999d221ef4b3ac46ad7a66871cd176470d38cd08faac051159826b4575205f91"

PILOT1_CASES = {"01-munich-natural": "munich", "15-rotterdam-natural": "rotterdam"}
PILOT2_CASES = {**PILOT1_CASES, "39-maastricht-natural": "maastricht"}
FULL_CASES = {
    **{f"munich-v11-{i:04d}": "munich" for i in range(120)},
    **{f"rotterdam-v11-{i:04d}": "rotterdam" for i in range(120)},
    **{f"maastricht-v11-{i:04d}": "maastricht" for i in range(120)},
}
CHALLENGE_CASES = {
    **{f"munich-v11c-{i:04d}": "munich" for i in range(23)},
    **{f"rotterdam-v11c-{i:04d}": "rotterdam" for i in range(23)},
    **{f"maastricht-v11c-{i:04d}": "maastricht" for i in range(23)},
}

PRODUCTION_SCOPES = ("development_full", "development_challenge")


def registered_scopes():
    return PRODUCTION_SCOPES


PILOTS = {PILOT1_HASH: ("two_map_pilot", PILOT1_CASES),
          PILOT2_HASH: ("three_map_pilot2", PILOT2_CASES),
          FULL_HASH: ("development_full", FULL_CASES),
          CHALLENGE_HASH: ("development_challenge", CHALLENGE_CASES)}


def registered_manifest(path: Path, freeze_sha256: str):
    digest = sha256(path)
    if digest not in PILOTS:
        raise ValueError("case manifest hash differs from an approved registration")
    scope, expected = PILOTS[digest]
    manifest = json.loads(path.read_text())
    cases = manifest.get("cases", [])
    if (manifest.get("scope") != "development"
            or manifest.get("freeze_sha256") != freeze_sha256
            or len(cases) != len(expected)
            or {case["id"]: case["map_id"] for case in cases} != expected):
        raise ValueError("invalid or unapproved development manifest identity")
    if scope == "development_full":
        if manifest.get("is_pilot") is not False:
            raise ValueError("development_full manifest cannot be a pilot manifest")
        if any(len(case.get("frames", [])) != 16 for case in cases):
            raise ValueError("development_full requires 16 frames per case")
    elif scope == "development_challenge":
        if manifest.get("is_pilot") is not False:
            raise ValueError("development_challenge manifest cannot be a pilot manifest")
        if manifest.get("subset") != "challenge":
            raise ValueError("development_challenge manifest requires subset == 'challenge'")
        if any(len(case.get("frames", [])) != 16 for case in cases):
            raise ValueError("development_challenge requires 16 frames per case")
        if any(case.get("kind") != "challenge" for case in cases):
            raise ValueError("development_challenge requires every case kind == 'challenge'")
    else:
        if (manifest.get("is_pilot") is not True
                or any([frame["frame"] for frame in case["frames"]] != [0] for case in cases)):
            raise ValueError("invalid or unapproved development pilot identity")
    return manifest, scope, digest
