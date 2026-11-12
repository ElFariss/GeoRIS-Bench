"""Fail-closed check of the user-approved v1.1 generator and its inputs."""
from hashlib import file_digest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK = Path("configs/bench/users-v1.1-freeze-20261109.json")
LOCK_SHA256 = "e152e78cb7b66204ea4080339c29a80a53c94685f8c3743b4eddd0cbab6e0ecd"


def sha256(path):
    with Path(path).open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def verify_freeze(root=ROOT):
    root = Path(root)
    if sha256(root / LOCK) != LOCK_SHA256:
        raise ValueError("unapproved generator freeze lock")
    frozen = json.loads((root / LOCK).read_text())
    wrong = [name for name, expected in frozen["files"].items()
             if not (root / name).is_file() or sha256(root / name) != expected]
    if wrong:
        raise ValueError("frozen v1.1 file mismatch: " + ", ".join(wrong))
    return LOCK_SHA256
