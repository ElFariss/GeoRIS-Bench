#!/usr/bin/env python3
"""Packaging entry point for the ris_relevant_v3 scope.

Runs the unchanged scripts/package_bench_production.py with the registry extended in this process
only (package() imports registered_manifest / PRODUCTION_SCOPES from their modules at call time, so
the module attributes are extended here). The frozen packager writes no split file for a scope it
does not know; this wrapper then writes the declared index split and a sidecar with hashes NEXT TO
the package (the package directory itself is not modified).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_bench_production as packager  # noqa: E402
from atomwise_ris.bench import production_pilot, production_run, ris_relevant_scope_v3  # noqa: E402
from atomwise_ris.bench.production_freeze import sha256, verify_freeze  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[argv.index("--output") + 1])
    manifest_path = Path(argv[argv.index("--manifest") + 1])
    manifest, scope, digest = ris_relevant_scope_v3.registered_manifest(manifest_path, verify_freeze())
    if scope != ris_relevant_scope_v3.SCOPE_NAME:
        raise ValueError(f"this wrapper packages only {ris_relevant_scope_v3.SCOPE_NAME}, got {scope}")
    production_pilot.registered_manifest = ris_relevant_scope_v3.registered_manifest
    production_run.PRODUCTION_SCOPES = ris_relevant_scope_v3.pipeline_scopes()
    packager.main(argv)
    split = ris_relevant_scope_v3.split_document(manifest["cases"])
    split_path = out.parent / "ris_relevant_split.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    sidecar = {"run_scope": scope, "manifest_sha256": digest, "package_path": str(out.resolve()),
               "package_provenance_sha256": sha256(out / "provenance.json") if (out / "provenance.json").is_file() else None,
               "split_file": str(split_path), "split_sha256": sha256(split_path),
               "declaration": ris_relevant_scope_v3.DECLARATION,
               "declaration_prerun_sha256": ris_relevant_scope_v3.DECLARATION_PRERUN_SHA256,
               "note": "package_role reads 'production_development' because the frozen packager has no role for this scope; "
                       "the scope is ris_relevant_pilot and the test split (0041-0047) is SEALED."}
    (out.parent / "ris_relevant_package_sidecar.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print("RIS_RELEVANT_PACKAGE_DONE", out, flush=True)


if __name__ == "__main__":
    main()
