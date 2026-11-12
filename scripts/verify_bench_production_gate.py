#!/usr/bin/env python3
"""Verify registered development pilot outputs and project the proposed build."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from atomwise_ris.bench.production_freeze import sha256,verify_freeze
from atomwise_ris.bench.production_pilot import registered_manifest
from atomwise_ris.hybrid.operator import HybridCoupledNetwork

ARRAYS=(64,128,256)
LABEL_NAMES=("single_flip_gains.h5","pair_flip_gains.h5","synergy.h5")
CLASSES={"los","reflection_only","diffraction_only","mixed","blocked"}


def h5_bytes(obj):
    size=0
    def visit(_name,x):
        nonlocal size
        if isinstance(x,h5py.Dataset):
            size+=x.id.get_storage_size()
    obj.visititems(visit)
    return size


def check_labels(label_files, key, n):
    groups=[f[key] for f in label_files]
    parents=("all_off","single_greedy")
    if any(set(g)!=set(parents) for g in groups):
        raise ValueError(f"unexpected label parents at {key}")
    for parent in parents:
        sg,pg,yg=[g[parent] for g in groups]
        bits=sg["bits"][:]
        bases=[float(g.attrs["base_rate_bps_hz"]) for g in (sg,pg,yg)]
        single=sg["single_flip_gains"][:]
        pairs=pg["pair_indices"][:]
        count=n*(n-1)//2 if n!=256 or parent=="all_off" else 4096
        pair=pg["pair_flip_gains"][:]
        syn=yg["synergy"][:]
        if (bits.shape!=(n,) or not np.isin(bits,(0,1)).all() or
            (parent=="all_off" and bits.any()) or any(not np.array_equal(g["bits"][:],bits) for g in (pg,yg)) or
            not np.isfinite(bases).all() or bases[0]!=bases[1] or bases[0]!=bases[2] or
            single.shape!=(n,) or pair.shape!=(count,) or syn.shape!=(count,) or pairs.shape!=(count,2) or
            not np.issubdtype(pairs.dtype,np.integer) or not np.array_equal(pairs,yg["pair_indices"][:]) or
            (pairs<0).any() or (pairs>=n).any() or (pairs[:,0]>=pairs[:,1]).any() or
            len(np.unique(pairs[:,0].astype(np.int64)*n+pairs[:,1]))!=count or
            not all(np.isfinite(arr).all() for arr in (single,pair,syn)) or
            not np.allclose(pair-single[pairs[:,0]]-single[pairs[:,1]],syn,rtol=0,atol=1e-12)):
            raise ValueError(f"invalid pilot labels at {key}/{parent}")
    return sum(h5_bytes(g) for g in groups)


def verify_pilot_gate(manifest_path, trace_manifest_path, channel_run_path, metadata_path, package_path, output_path, gpu_peak_mib=None):
    manifest_path, trace_manifest_path, channel_run_path, metadata_path, package_path, output_path = map(Path, (manifest_path, trace_manifest_path, channel_run_path, metadata_path, package_path, output_path))
    freeze=verify_freeze()
    v,scope,mh=registered_manifest(manifest_path,freeze)
    case_maps={c["id"]:c["map_id"] for c in v["cases"]}
    tr=json.loads(trace_manifest_path.read_text())
    cr=json.loads(channel_run_path.read_text())
    if (tr["status"]!="complete" or tr.get("run_scope")!=scope or
            any(x["case_manifest_sha256"]!=mh for x in (tr,cr))):
        raise ValueError("trace/channel sources do not match the pilot")
    geometry={}
    if scope=="three_map_pilot2":
        geo_path=trace_manifest_path.with_name("geometry_preflight.json")
        g=json.loads(geo_path.read_text())
        if (sha256(geo_path)!=tr.get("geometry_preflight_sha256") or g["status"]!="complete"
                or g.get("run_scope")!=scope or g["case_manifest_sha256"]!=mh
                or {row["case_id"]:row["map_id"] for row in g["cases"]}!=case_maps
                or not all(row["status"]=="pass" for row in g["cases"])):
            raise ValueError("pilot2 has no matching successful geometry preflight")
        geometry={row["map_id"]:{"build_seconds":row["geometry_elapsed_s"],
                   "mesh_audit":row["scene"]["mesh_audit"]} for row in g["cases"]}
    z=package_path
    pk=json.loads((z/"provenance.json").read_text())
    cf=z/"channels/channel_realizations.h5"
    if (not (z/"DONE").exists() or pk["package_role"]!="pilot_development" or pk["run_scope"]!=scope or
            pk["pilot_manifest_sha256"]!=mh or sha256(cf)!=cr["hdf5_sha256"]):
        raise ValueError("unverified or incomplete pilot package")
    for nm in LABEL_NAMES:
        if sha256(z/"optimization"/nm)!=pk["label_files_sha256"][nm]:
            raise ValueError(f"label provenance mismatch: {nm}")
    loader_path=z/"examples/load_dataset.py"
    checked_files={loader_path:pk["loader_sha256"],z/"trajectories.parquet":pk["trajectories_sha256"],
        z/"channels/metadata.json":cr["metadata_sha256"]}
    for name in ("within_map_trajectory.json","leave_map_out_development.json"):
        checked_files[z/"splits"/name]=pk["split_files_sha256"][name]
    if any(sha256(path)!=expected for path,expected in checked_files.items()):
        raise ValueError("package metadata, loader, trajectory or split provenance mismatch")
    import pandas as pd
    trajectory=pd.read_parquet(z/"trajectories.parquet")
    row_names=["map_id","case_id","frame","array_n","user_id"]
    rows=[tuple(x) for x in trajectory[row_names].itertuples(index=False,name=None)]
    expected_rows={(c["map_id"],c["id"],f["frame"],n,str(u["user_id"]))
                   for c in v["cases"] for f in c["frames"] for n in ARRAYS for u in f["users"]}
    if set(rows)!=expected_rows or len(rows)!=len(expected_rows) or set(trajectory.split)!={"development"}:
        raise ValueError("trajectory coverage/split differs from registered users")
    spec=importlib.util.spec_from_file_location("bench_pilot_packaged_loader",loader_path)
    loader=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    links=json.loads(metadata_path.read_text())
    if sha256(metadata_path)!=cr["metadata_sha256"]:
        raise ValueError("channel metadata hash mismatch")
    trace_key={f"{x['map_id']}/{x['array_elements']}/{x['case_id']}/{x['frame']}":x
               for x in tr["cases"]}
    keys={f"{c['map_id']}/{n}/{c['id']}/{f['frame']}"
          for c in v["cases"] for n in ARRAYS for f in c["frames"]}
    if (set(links)!=keys or set(trace_key)!=keys or set(pk["channel_keys"])!=keys
            or len(tr["cases"])!=len(keys) or len(pk["channel_keys"])!=len(keys)):
        raise ValueError("pilot coverage mismatch")
    case_data=defaultdict(dict)
    max_rate_error=0.
    max_loader_error=0.
    correct=total=0
    per_map_bs_los=defaultdict(bool)
    per_map_los=defaultdict(lambda: {"matches":0,"total":0})
    class_counts=defaultdict(int)
    with ExitStack() as files:
        h=files.enter_context(h5py.File(cf,"r"))
        label_files=[files.enter_context(h5py.File(z/"optimization"/nm,"r")) for nm in LABEL_NAMES]
        for key in sorted(keys):
            c,n,case,f=key.split("/")
            n=int(n)
            r=h[key]
            names=("H_direct","C_in","C_out","Z_effective","loads","precoder","stored_bits")
            data={nm:r[nm][:] for nm in names}
            k=data["H_direct"].shape[0]
            shapes=((k,4),(n,4),(k,n),(n,n),(2,),(4,k),(n,))
            if any(data[nm].shape!=shape or not np.isfinite(data[nm]).all()
                   for nm,shape in zip(names,shapes,strict=True)):
                raise ValueError(f"invalid channel array at {key}")
            if int(r.attrs["mesh_segments"])!=(81 if n==256 else 161):
                raise ValueError(f"wrong approved wire mesh at {key}")
            ev=HybridCoupledNetwork(data["Z_effective"],data["C_out"],data["C_in"],
                data["H_direct"],data["loads"],w=data["precoder"],noise=r.attrs["noise"])
            rate=float(ev.score(ev.channel(data["stored_bits"])@ev.w))
            stored=float(r.attrs["stored_rate_bps_hz"])
            if not (np.isfinite(stored) and np.isfinite(rate) and np.isfinite(float(r.attrs["noise"]))
                    and float(r.attrs["noise"])>0 and np.isin(data["stored_bits"],(0,1)).all()):
                raise ValueError(f"invalid stored state at {key}")
            max_rate_error=max(max_rate_error,abs(rate-stored))
            loaded=loader.rate_for_bits(cf,key,data["stored_bits"])
            if not np.isfinite(loaded):
                raise ValueError(f"nonfinite loader rate at {key}")
            max_loader_error=max(max_loader_error,abs(loaded-stored))
            label_bytes=check_labels(label_files,key,n)
            m=links[key]
            bs_ris=m["link_classes"]["bs_ris"]
            per_map_bs_los[c]|=bs_ris=="los"
            for nm,val in [("bs_ris",bs_ris),*((kind,u[kind]) for u in m["users"] for kind in ("bs_user","ris_user"))]:
                if val not in CLASSES:
                    raise ValueError(f"unknown path class {val}")
                class_counts[f"{nm}:{val}"]+=1
            for agree in m["geometry_agreement"].values():
                correct+=sum(agree); total+=len(agree)
                per_map_los[c]["matches"]+=sum(agree); per_map_los[c]["total"]+=len(agree)
            t=trace_key[key]
            case_data[case][str(n)]={"city":c,"trace_s":t.get("solver_elapsed_seconds",t["elapsed_seconds"]),
                   "trace_total_s":t["elapsed_seconds"],"trace_export_s":t.get("export_elapsed_seconds"),
                   "scene_load_and_ports_s":t.get("scene_load_and_ports_s"),
                   "label_s":pk["label_frame_seconds"][key],
                   "label_dataset_bytes":label_bytes,
                   "channel_s":m["elapsed_channel_s"],"channel_dataset_bytes":h5_bytes(r),
                   "raw_path_bytes":m["raw_path_bytes"],"paths":m["paths"],
                   "gpu_peak_sampled_mib":t.get("gpu_peak_sampled",{}).get("peak_used_mib"),
                   "gpu_memory_samples":t.get("gpu_peak_sampled",{}).get("samples",0)}
    agree=correct/total if total else 0.
    run_stages={n:float(x["setup_s"]) for n,x in cr["arrays"].items()}
    full_h5_bytes=cf.stat().st_size
    array_count=len(keys)
    raw=sum(x["raw_path_bytes"] for r in case_data.values() for x in r.values())
    trace_s=sum(x["trace_s"] for r in case_data.values() for x in r.values())
    channel_s=sum(x["channel_s"] for r in case_data.values() for x in r.values())
    label_s=sum(x["label_s"] for r in case_data.values() for x in r.values())
    export=[x["trace_export_s"] for r in case_data.values() for x in r.values()]
    export_s=sum(export) if all(x is not None for x in export) else None
    label_sizes={nm:(z/"optimization"/nm).stat().st_size for nm in LABEL_NAMES}
    n_trajectories=120*len(case_maps)
    factor=n_trajectories*16/len(v["cases"])
    label_factor=n_trajectories*4/len(v["cases"])
    prep=[x["scene_load_and_ports_s"] for r in case_data.values() for x in r.values()]
    prep_s=sum(prep) if all(x is not None for x in prep) else None
    gpu=[x["gpu_peak_sampled_mib"] for r in case_data.values() for x in r.values()]
    peak=gpu_peak_mib if gpu_peak_mib is not None else max((x for x in gpu if x is not None),default=None)
    raw_projected=int(raw*factor)
    derived_projected=int(full_h5_bytes*factor)+int(sum(label_sizes.values())*label_factor)
    total_projected=raw_projected+derived_projected
    trace_volume=trace_manifest_path.parent
    free=shutil.disk_usage(z).free
    trace_free=shutil.disk_usage(trace_volume).free
    shared_device=z.stat().st_dev==trace_volume.stat().st_dev
    fits=(total_projected<=.8*min(trace_free,free) if shared_device else
          raw_projected<=.8*trace_free and derived_projected<=.8*free)
    checks={"complete_hdf5_schema":True,"finite":True,
            "loader_dense_1e_10":max(max_rate_error,max_loader_error)<=1e-10,
            "los_geometry_agreement":agree>=.75,
            "bs_ris_has_los_each_map":all(per_map_bs_los.get(c,False) for c in case_maps.values()),
            "gpu_peak_measured":all(x is not None for x in gpu)}
    if scope=="three_map_pilot2":
        checks["mesh_preflight_all_maps"]=set(geometry)==set(case_maps.values())
    result={"status":"pass" if all(checks.values()) else "fail","checks":checks,
        "rate_max_abs_error":max_rate_error,"loader_max_abs_error":max_loader_error,
        "geometry_los_matches":correct,"geometry_los_total":total,
        "geometry_los_agreement":agree,"los_geometry_by_map":dict(per_map_los),
        "bs_ris_has_los_by_map":{c:per_map_bs_los.get(c,False) for c in case_maps.values()},
        "geometry_preflight_by_map":geometry,"class_counts":dict(class_counts),"array_frames":array_count,
        "run_scope":scope,"trajectory_rows":len(rows),
        "gpu_peak_sampled_mib":peak,"per_case":dict(case_data),"nec_setup_s":run_stages,
        "channel_worker_peak_rss_bytes":{n:x.get("peak_rss_bytes") for n,x in cr["arrays"].items()},
        "observed":{"raw_path_bytes":raw,"channel_h5_bytes":full_h5_bytes,
             "label_bytes":label_sizes,"trace_s":trace_s,"trace_export_s":export_s,
             "channel_s":channel_s,"label_s":label_s,"package_s":pk["runtime_s"]},
        "production_projection":{"trajectories":n_trajectories,"map_frames":n_trajectories*16,
            "array_frames":n_trajectories*16*len(ARRAYS),"raw_path_bytes":raw_projected,
            "channel_h5_bytes":int(full_h5_bytes*factor),
            "label_bytes_conservative":int(sum(label_sizes.values())*label_factor),
            "projected_bytes_without_working_overhead":total_projected,
            "trace_volume_free_bytes_at_gate":trace_free,"package_volume_free_bytes_at_gate":free,
            "trace_and_package_share_volume":shared_device,"fits_with_20_percent_headroom":fits,
            "trace_solver_s":trace_s*factor,"trace_export_s":export_s*factor if export_s is not None else None,
            "scene_load_and_ports_s_if_reloaded_each_frame":prep_s*factor if prep_s is not None else None,
            "channel_s":channel_s*factor,"label_s_conservative":label_s*label_factor,
            "one_time_nec_setup_s":sum(run_stages.values()),
            "scope":"one observed placement per registered map; excludes scene rebuild and I/O concurrency; not approval"}}
    temp=output_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result,allow_nan=False,indent=2)+"\n")
    temp.replace(output_path)
    print(result["status"],result["checks"],flush=True)
    return result


def _write_fail_gate(output_path, checks, failing_cases, per_map, reasons, scope="unknown"):
    res = {
        "status": "fail",
        "run_scope": scope,
        "checks": checks,
        "per_map": per_map,
        "failing_cases": failing_cases,
        "reasons": reasons,
    }
    output_path = Path(output_path)
    tmp = output_path.with_suffix(f".json.tmp.{json.__name__}")
    tmp.write_text(json.dumps(res, allow_nan=False, indent=2) + "\n")
    tmp.replace(output_path)
    print("fail", checks, flush=True)
    return res


def verify_production_gate(manifest_path, trace_out, channel_run_path, metadata_path,
                           package_path, output_path, verify_freeze_check=True):
    from atomwise_ris.bench.production_run import PRODUCTION_SCOPES, stop_rule_exceeded

    manifest_path, trace_out, channel_run_path, metadata_path, package_path, output_path = map(
        Path, (manifest_path, trace_out, channel_run_path, metadata_path, package_path, output_path)
    )

    checks = {
        "complete_hdf5_schema": True,
        "finite": True,
        "loader_dense_1e_10": True,
        "los_geometry_agreement": True,
        "bs_ris_link_los": True,
        "mesh_preflight_all_passed": True,
        "stop_rule_not_exceeded": True,
        "provenance_and_splits_valid": True,
        "no_failing_cases": True,
    }
    failing_cases = defaultdict(lambda: {"map_id": "", "reasons": []})
    per_map = {}
    top_reasons = []

    # Fail closed on missing inputs
    for name, p in (("manifest", manifest_path), ("trace_out", trace_out),
                    ("channel_run", channel_run_path), ("metadata", metadata_path),
                    ("package", package_path)):
        if not p.exists():
            checks["provenance_and_splits_valid"] = False
            top_reasons.append(f"missing input path for {name}: {p}")
            return _write_fail_gate(output_path, checks, dict(failing_cases), per_map, top_reasons)

    try:
        if verify_freeze_check:
            freeze = verify_freeze()
            manifest_data, scope, mh = registered_manifest(manifest_path, freeze)
        else:
            manifest_data = json.loads(manifest_path.read_text())
            mh = sha256(manifest_path)
            scope = manifest_data.get("run_scope", manifest_data.get("scope", "development_full"))
            if scope == "development":
                scope = "development_challenge" if manifest_data.get("subset") == "challenge" else "development_full"
    except Exception as exc:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append(f"manifest verification error: {exc}")
        return _write_fail_gate(output_path, checks, dict(failing_cases), per_map, top_reasons)

    if scope not in PRODUCTION_SCOPES:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append(f"unapproved production scope: {scope}")
        return _write_fail_gate(output_path, checks, dict(failing_cases), per_map, top_reasons, scope)

    try:
        cr = json.loads(channel_run_path.read_text())
    except Exception as exc:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append(f"invalid channel_run JSON: {exc}")
        return _write_fail_gate(output_path, checks, dict(failing_cases), per_map, top_reasons, scope)

    if cr.get("case_manifest_sha256") != mh:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append("channel_run case_manifest_sha256 mismatch")
    if cr.get("run_scope") != scope:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append(f"channel_run run_scope mismatch: {cr.get('run_scope')} != {scope}")

    cr_failed_by_map = defaultdict(set)
    for m, m_data in cr.get("maps", {}).items():
        cr_failed_by_map[m].update(m_data.get("failed_cases", []))

    z = package_path
    if not (z / "DONE").exists():
        checks["provenance_and_splits_valid"] = False
        top_reasons.append("package DONE marker missing")

    expected_role = "production_challenge" if scope == "development_challenge" else "production_development"
    try:
        pk = json.loads((z / "provenance.json").read_text())
    except Exception as exc:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append(f"failed reading package provenance.json: {exc}")
        pk = {}

    if pk.get("package_role") != expected_role:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append(f"package_role '{pk.get('package_role')}' != expected '{expected_role}'")
    if pk.get("run_scope") != scope:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append("package run_scope mismatch")
    if pk.get("pilot_manifest_sha256") != mh and pk.get("manifest_sha256") != mh:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append("package manifest hash mismatch")

    cf = z / "channels/channel_realizations.h5"
    if not cf.is_file() or sha256(cf) != cr.get("hdf5_sha256"):
        checks["provenance_and_splits_valid"] = False
        top_reasons.append("package channel HDF5 hash mismatch with channel_run")

    pkg_meta = z / "channels/metadata.json"
    if not pkg_meta.is_file() or sha256(pkg_meta) != cr.get("metadata_sha256") or sha256(metadata_path) != cr.get("metadata_sha256"):
        checks["provenance_and_splits_valid"] = False
        top_reasons.append("metadata sha256 mismatch")

    for nm in LABEL_NAMES:
        lp = z / "optimization" / nm
        if not lp.is_file() or sha256(lp) != pk.get("label_files_sha256", {}).get(nm):
            checks["provenance_and_splits_valid"] = False
            top_reasons.append(f"label file {nm} hash mismatch")

    loader_path = z / "examples/load_dataset.py"
    if not loader_path.is_file() or sha256(loader_path) != pk.get("loader_sha256"):
        checks["provenance_and_splits_valid"] = False
        top_reasons.append("loader hash mismatch")

    traj_path = z / "trajectories.parquet"
    if not traj_path.is_file() or sha256(traj_path) != pk.get("trajectories_sha256"):
        checks["provenance_and_splits_valid"] = False
        top_reasons.append("trajectories.parquet hash mismatch")

    split_dir = z / "splits"
    if scope == "development_full":
        if not (split_dir / "within_map_trajectory.json").is_file() or not (split_dir / "leave_environment_out.json").is_file():
            checks["provenance_and_splits_valid"] = False
            top_reasons.append("development_full missing required split files")
        if (split_dir / "challenge.json").exists():
            checks["provenance_and_splits_valid"] = False
            top_reasons.append("challenge.json must not appear in development_full package")
    elif scope == "development_challenge":
        if not (split_dir / "challenge.json").is_file():
            checks["provenance_and_splits_valid"] = False
            top_reasons.append("development_challenge missing challenge.json")
        if (split_dir / "within_map_trajectory.json").exists() or (split_dir / "leave_environment_out.json").exists():
            checks["provenance_and_splits_valid"] = False
            top_reasons.append("development split files must not appear in challenge package")

    try:
        spec = importlib.util.spec_from_file_location("packaged_loader", loader_path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
    except Exception as exc:
        checks["loader_dense_1e_10"] = False
        top_reasons.append(f"failed to import packaged loader: {exc}")
        loader = None

    import pandas as pd
    try:
        traj_df = pd.read_parquet(traj_path)
        req_cols = {"map_id", "case_id", "trajectory_id", "frame", "time_s", "user_id",
                    "user_x", "user_y", "user_z", "velocity_xyz", "bs_link", "ris_link",
                    "bs_ris_link", "array_n", "split", "kind", "scenario", "trajectory_index", "labelled"}
        if not req_cols.issubset(set(traj_df.columns)):
            checks["provenance_and_splits_valid"] = False
            top_reasons.append(f"trajectories.parquet missing required columns: {req_cols - set(traj_df.columns)}")
    except Exception as exc:
        checks["provenance_and_splits_valid"] = False
        top_reasons.append(f"failed reading trajectories.parquet: {exc}")
        traj_df = None

    metadata = json.loads(metadata_path.read_text())

    # Manifest cases grouped by map
    manifest_by_map = defaultdict(list)
    for c in manifest_data.get("cases", []):
        manifest_by_map[c["map_id"]].append(c)

    max_loader_error = 0.0

    with ExitStack() as stack:
        cf_h5 = stack.enter_context(h5py.File(cf, "r"))
        single_h5 = stack.enter_context(h5py.File(z / "optimization/single_flip_gains.h5", "r"))
        pair_h5 = stack.enter_context(h5py.File(z / "optimization/pair_flip_gains.h5", "r"))
        syn_h5 = stack.enter_context(h5py.File(z / "optimization/synergy.h5", "r"))

        for map_id, cases in manifest_by_map.items():
            total_cases = len(cases)
            failed_case_ids = set(cr_failed_by_map[map_id])
            map_matches = 0
            map_total = 0
            map_bs_ris_los = True
            map_preflight_ok = True

            for case in cases:
                case_id = case["id"]
                traj_idx = case.get("trajectory_index")
                if traj_idx is None:
                    try:
                        traj_idx = int(case_id.split("-")[-1])
                    except Exception:
                        traj_idx = 0

                # If already recorded failed upstream, skip internal checks for it
                if case_id in failed_case_ids:
                    continue

                case_errors = []

                # Trace case record & mesh preflight check
                trace_case_file = trace_out / map_id / "cases" / f"{case_id}.json"
                trace_failed_file = trace_out / map_id / "cases" / f"{case_id}.FAILED.json"
                if trace_failed_file.is_file():
                    case_errors.append("trace record marked FAILED")
                elif not trace_case_file.is_file():
                    case_errors.append("missing trace case record")
                else:
                    try:
                        tr_data = json.loads(trace_case_file.read_text())
                        if tr_data.get("status") != "pass":
                            case_errors.append(f"trace record status '{tr_data.get('status')}' != 'pass'")
                        if tr_data.get("case_manifest_sha256") != mh:
                            case_errors.append("trace record manifest hash mismatch")
                        geom = tr_data.get("geometry", {})
                        preflight = geom.get("preflight_checks") or geom.get("mesh_audit", {}).get("preflight_checks") or tr_data.get("preflight_checks", {})
                        if not isinstance(preflight, dict) or preflight.get("all_passed") is not True:
                            case_errors.append("mesh preflight checks did not pass")
                            map_preflight_ok = False
                    except Exception as exc:
                        case_errors.append(f"error reading trace case record: {exc}")
                        map_preflight_ok = False

                frames = case.get("frames", [])
                if len(frames) != 16:
                    case_errors.append(f"case has {len(frames)} frames; expected 16")

                for fr_idx, fr in enumerate(frames):
                    fr_num = str(int(fr["frame"]))
                    is_labelled = (fr_idx in (0, 5, 10, 15))

                    for n in ARRAYS:
                        key = f"{map_id}/{n}/{case_id}/{fr_num}"
                        if key not in cf_h5:
                            case_errors.append(f"missing channel group {key}")
                            checks["complete_hdf5_schema"] = False
                            continue
                        g = cf_h5[key]

                        # Check schema & finite values
                        ds_names = ("H_direct", "C_in", "C_out", "Z_effective", "loads", "precoder", "stored_bits")
                        for nm in ds_names:
                            if nm not in g:
                                case_errors.append(f"missing dataset {nm} at {key}")
                                checks["complete_hdf5_schema"] = False
                                continue
                            arr = g[nm][()]
                            if not np.isfinite(arr).all():
                                case_errors.append(f"non-finite values in {nm} at {key}")
                                checks["finite"] = False

                        noise = float(g.attrs.get("noise", 0.0))
                        stored_rate = float(g.attrs.get("stored_rate_bps_hz", np.nan))
                        if not (np.isfinite(noise) and noise > 0 and np.isfinite(stored_rate)):
                            case_errors.append(f"invalid noise/stored_rate at {key}")
                            checks["finite"] = False

                        # Check loader dense rate consistency
                        if loader is not None and "stored_bits" in g:
                            try:
                                loaded_rate = loader.rate_for_bits(cf, key, g["stored_bits"][()])
                                if not np.isfinite(loaded_rate):
                                    case_errors.append(f"loader returned non-finite rate at {key}")
                                    checks["finite"] = False
                                err = abs(loaded_rate - stored_rate)
                                max_loader_error = max(max_loader_error, err)
                                if err > 1e-10:
                                    case_errors.append(f"loader dense rate error {err:.3e} > 1e-10 at {key}")
                                    checks["loader_dense_1e_10"] = False
                            except Exception as exc:
                                case_errors.append(f"loader rate error at {key}: {exc}")
                                checks["loader_dense_1e_10"] = False

                        # Metadata LOS sanity & geometry agreement
                        md = metadata.get(key, metadata.get("/" + key))
                        if md is None:
                            case_errors.append(f"missing metadata for {key}")
                        else:
                            bs_ris = md.get("link_classes", {}).get("bs_ris")
                            if bs_ris != "los":
                                case_errors.append(f"BS-RIS link is '{bs_ris}', expected 'los' at {key}")
                                map_bs_ris_los = False
                                checks["bs_ris_link_los"] = False
                            for agree_list in md.get("geometry_agreement", {}).values():
                                map_matches += sum(agree_list)
                                map_total += len(agree_list)

                        # Labels check
                        if not is_labelled:
                            if key in single_h5 or key in pair_h5 or key in syn_h5:
                                case_errors.append(f"unlabelled frame {key} unexpectedly present in label files")
                                checks["complete_hdf5_schema"] = False
                        else:
                            if key not in single_h5 or key not in pair_h5 or key not in syn_h5:
                                case_errors.append(f"labelled frame {key} missing from label files")
                                checks["complete_hdf5_schema"] = False
                            else:
                                sg_k, pg_k, yg_k = single_h5[key], pair_h5[key], syn_h5[key]
                                parents = ("all_off", "single_greedy")
                                if set(sg_k.keys()) != set(parents) or set(pg_k.keys()) != set(parents) or set(yg_k.keys()) != set(parents):
                                    case_errors.append(f"unexpected label parents at {key}")
                                    checks["complete_hdf5_schema"] = False
                                else:
                                    for parent in parents:
                                        sg, pg, yg = sg_k[parent], pg_k[parent], yg_k[parent]
                                        bits = sg["bits"][()]
                                        single = sg["single_flip_gains"][()]
                                        pairs = pg["pair_indices"][()]
                                        pair = pg["pair_flip_gains"][()]
                                        syn = yg["synergy"][()]
                                        bases = [float(target.attrs.get("base_rate_bps_hz", np.nan)) for target in (sg, pg, yg)]
                                        if not all(np.isfinite(b) for b in bases) or bases[0] != bases[1] or bases[0] != bases[2]:
                                            case_errors.append(f"inconsistent base_rate at {key}/{parent}")
                                            checks["finite"] = False

                                        if n in (64, 128):
                                            expected_pairs = n * (n - 1) // 2
                                            expected_cov = "full"
                                        elif n == 256:
                                            if parent == "all_off" and fr_idx == 0 and (traj_idx % 10 == 0):
                                                expected_pairs = 32640
                                                expected_cov = "full"
                                            else:
                                                expected_pairs = 4096
                                                expected_cov = "sampled"
                                        else:
                                            expected_pairs = len(pairs)
                                            expected_cov = pg.attrs.get("pair_coverage")

                                        if len(pairs) != expected_pairs:
                                            case_errors.append(f"pair count {len(pairs)} != {expected_pairs} at {key}/{parent}")
                                            checks["complete_hdf5_schema"] = False
                                        if pg.attrs.get("pair_coverage") != expected_cov:
                                            case_errors.append(f"pair_coverage '{pg.attrs.get('pair_coverage')}' != '{expected_cov}' at {key}/{parent}")
                                            checks["complete_hdf5_schema"] = False
                                        if not all(np.isfinite(arr).all() for arr in (bits, single, pair, syn)):
                                            case_errors.append(f"non-finite values in labels at {key}/{parent}")
                                            checks["finite"] = False
                                        if not np.allclose(pair - single[pairs[:, 0]] - single[pairs[:, 1]], syn, rtol=0, atol=1e-12):
                                            case_errors.append(f"synergy identity violated at {key}/{parent}")
                                            checks["complete_hdf5_schema"] = False

                if case_errors:
                    failing_cases[case_id]["map_id"] = map_id
                    failing_cases[case_id]["reasons"].extend(case_errors)
                    failed_case_ids.add(case_id)

            agree_ratio = map_matches / map_total if map_total > 0 else 0.0
            if agree_ratio < 0.75:
                checks["los_geometry_agreement"] = False
            if not map_bs_ris_los:
                checks["bs_ris_link_los"] = False
            if not map_preflight_ok:
                checks["mesh_preflight_all_passed"] = False

            stop_breach = stop_rule_exceeded(len(failed_case_ids), total_cases)
            if stop_breach:
                checks["stop_rule_not_exceeded"] = False

            per_map[map_id] = {
                "total_cases": total_cases,
                "included_cases": total_cases - len(cr_failed_by_map[map_id]),
                "failed_cases_count": len(failed_case_ids),
                "stop_rule_exceeded": stop_breach,
                "los_geometry_matches": map_matches,
                "los_geometry_total": map_total,
                "los_geometry_agreement": agree_ratio,
                "bs_ris_los": map_bs_ris_los,
                "mesh_preflight_all_passed": map_preflight_ok
            }

    if failing_cases:
        checks["no_failing_cases"] = False

    passed = all(checks.values())
    result = {
        "status": "pass" if passed else "fail",
        "run_scope": scope,
        "checks": checks,
        "per_map": per_map,
        "failing_cases": dict(failing_cases),
        "rate_max_abs_error": max_loader_error
    }
    temp = output_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    temp.replace(output_path)
    print(result["status"], result["checks"], flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production", action="store_true", help="run gate in production mode")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--trace-out", type=Path, help="trace root directory (required with --production)")
    p.add_argument("--trace-manifest", type=Path, help="trace manifest file (required without --production)")
    p.add_argument("--channel-run", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gpu-peak-mib", type=float)
    p.add_argument("--no-verify-freeze", action="store_true", help="diagnostic skip of freeze check")
    a = p.parse_args(argv)

    if a.production:
        if not a.trace_out:
            p.error("--trace-out is required when running with --production")
        verify_production_gate(a.manifest, a.trace_out, a.channel_run, a.metadata,
                               a.package, a.output, verify_freeze_check=not a.no_verify_freeze)
    else:
        if not a.trace_manifest:
            p.error("--trace-manifest is required in pilot mode")
        verify_pilot_gate(a.manifest, a.trace_manifest, a.channel_run, a.metadata,
                          a.package, a.output, gpu_peak_mib=a.gpu_peak_mib)


if __name__ == "__main__":
    main()


if __name__=="__main__":
    main()
