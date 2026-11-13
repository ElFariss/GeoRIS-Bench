#!/usr/bin/env python3
"""Merge ris_relevant_v3 trace shards into one trace directory (docs/bench-ris-relevant-scope-v3-20261113.md s3, s8).

Case records are copied verbatim (they carry the full-manifest hash); the npz directories n64/n128/n256/direct of each
case are symlinked, because the channel builder reads <trace>/<map>/n<N>/<case>/frame-XXXX.npz.
--watch: repeat every 30 s until every listed shard has its map DONE, then (with --final) write the merged
trace_manifest.json, DONE and JOB_DONE. Without --final, a node-local DONE is written once the listed shards are DONE
(so that node's channel workers can finish); --final later replaces it with the full-scope DONE.
"""
import argparse, hashlib, json, os, time
from pathlib import Path

MAP = "rotterdam"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(p, text):
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    tmp.write_text(text); tmp.replace(p)


def merge_once(shards, out):
    md = out / MAP; (md / "cases").mkdir(parents=True, exist_ok=True)
    new = 0
    for s in shards:
        cdir = s / MAP / "cases"
        if not cdir.is_dir():
            continue
        for f in sorted(cdir.glob("*.json")):
            if ".prev." in f.name:
                continue                      # archived by --retry-failed; not part of the merged scope
            dst = md / "cases" / f.name
            if dst.exists():
                continue
            cid = f.name.replace(".FAILED.json", "").replace(".json", "")
            if not f.name.endswith(".FAILED.json"):
                for sub in ("n64", "n128", "n256", "direct"):
                    src = (s / MAP / sub / cid).resolve()
                    if not src.is_dir():
                        raise RuntimeError(f"{cid}: missing {sub} directory in shard {s}")
                    link = md / sub / cid
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if not link.exists():
                        os.symlink(src, link)
            atomic(dst, f.read_text()); new += 1
            stale = md / "cases" / f"{cid}.FAILED.json"
            if not f.name.endswith(".FAILED.json") and stale.exists():
                stale.rename(md / "cases" / f"{cid}.FAILED.superseded.json.txt")
    return new


def finish(shards, out, manifest_sha, final):
    md = out / MAP
    mans = [json.loads((s / MAP / "trace_manifest.json").read_text()) for s in shards]
    cases = sorted((c for m in mans for c in m["cases"]), key=lambda c: c["case_id"])
    failed = sorted(x for m in mans for x in m.get("failed_cases", []))
    if any(m["case_manifest_sha256"] != manifest_sha for m in mans):
        raise RuntimeError("shard trace manifest hash mismatch")
    merged = {**mans[0], "cases": cases, "failed_cases": failed, "status": "complete",
              "rr3_merged_shards": [str(s) for s in shards], "rr3_final": bool(final)}
    atomic(md / "trace_manifest.json", json.dumps(merged, indent=2, sort_keys=True) + "\n")
    atomic(md / "DONE", f"status=complete\nfinished_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
           f"manifest_sha256={sha(md / 'trace_manifest.json')}\ncases_passed={len(cases)}\ncases_failed={len(failed)}\n"
           f"rr3_final={int(final)}\n")
    if final:
        atomic(out / "JOB_DONE", f"exit_status=0\nfinished_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("shards", type=Path, nargs="+")
    a = ap.parse_args()
    while True:
        n = merge_once(a.shards, a.out)
        if n:
            print(time.strftime("%H:%M:%S"), "merged", n, flush=True)
        failed_shard = [s for s in a.shards if (s / MAP / "FAILED").exists()]
        if failed_shard:
            raise SystemExit(f"shard FAILED: {failed_shard}")
        if all((s / MAP / "DONE").exists() for s in a.shards):
            merge_once(a.shards, a.out)
            finish(a.shards, a.out, a.manifest_sha256, a.final)
            print("merged DONE", "final" if a.final else "node-local", flush=True)
            return
        if not a.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
