"""Score the registered P4 ray-to-port comparison against independent NEC data."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from atomwise_ris.hybrid.p4_calibration import P4_LIMITS, complex_metrics, cross_polar_leakage
from atomwise_ris.hybrid.p4_rt_bridge import FREQ, response, field_from_jones, isolated_nec_fields
from atomwise_ris.hybrid.p4_vector_mapping import field_to_open_circuit_voltage, direct_channel_with_open_gap
from atomwise_ris.hybrid.ray_coupling import impressed_segment_currents, open_gap_scatter_impedance


def pack(x):
    return np.stack((x.real,x.imag),axis=-1).tolist()


def receive(wires, e):
    xyz = wires.segment_centers
    return field_to_open_circuit_voltage(wires.iz, xyz, np.tile([0.,0.,1.],(len(xyz),1)),
                                        wires.segment_lengths, e)


def fields(mode, ris, ext, ris_wire, external_wires, reflected=False):
    if len(mode)!=2 or any(len(x["rows"])!=65 for x in mode):
        raise ValueError("expected 64 RIS and one opposite external port for each source")
    segment_cols, point_cols, mutual, availability = [], [], [], []
    for i,(source, data) in enumerate(zip(ext,mode,strict=True)):
        sw = external_wires[i]
        dw = external_wires[1-i]
        r = data["rows"]
        availability.append([v is not None for v in r])
        seg = [field_from_jones(r[k],source,p,sw,
             ris_wire.segment_centers[81*k:81*(k+1)],reflected) for k,p in enumerate(ris)]
        segment_cols.append(np.concatenate(seg))
        other = field_from_jones(r[-1],source,ext[1-i],sw,dw.segment_centers,reflected)
        point_cols.append(np.vstack((np.asarray(seg)[:,40,:],other[40,:])))
        mutual.append(receive(dw,other)[0])
    return (np.stack(segment_cols,axis=-1),np.stack(point_cols,axis=0),
            np.asarray(mutual),np.asarray(availability))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace",type=Path,required=True)
    ap.add_argument("--geometry",type=Path,required=True)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--reflection-reference",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    g=json.loads(a.geometry.read_text())
    if sha256(a.reference.read_bytes()).hexdigest()!=g["reference_sha256"]:
        raise ValueError("independent NEC reference hash mismatch")
    trace=json.loads(a.trace.read_text())
    if trace["sionna_rt"]!="2.0.1":
        raise ValueError("unexpected Sionna version")
    ris,ext=np.asarray(g["ris"]),np.asarray(g["external"])
    print("isolated external and RIS-alone current bases",flush=True)
    er=[response(p[None,:]) for p in ext]
    rr=response(ris)
    e_seg, rt_e, rt_z_directional, available = fields(trace["modes"]["munich_los"],ris,ext,rr,er)
    z_ee_rt=np.diag(np.array([w.z_ris[0,0] for w in er]))
    z_ee_rt[0,1]=z_ee_rt[1,0]=rt_z_directional.mean()
    rt_z_re=np.column_stack([receive(rr,e_seg[...,i]) for i in range(2)])
    print("RIS-only impressed current open-gap predictor",flush=True)
    ez=e_seg[:,2,:]
    currents=impressed_segment_currents(ris,rr.segment_lengths[:,None]*ez)
    delta_z=open_gap_scatter_impedance(rr,ez,currents)
    h0,h1=direct_channel_with_open_gap(z_ee_rt,delta_z,1)
    # Reference vectors and monolithic targets are accessed only after prediction.
    print("independent NEC NE-card fields and frozen monolithic targets",flush=True)
    nec_e=np.stack([isolated_nec_fields(s,np.vstack((ris,ext[1-i]))) for i,s in enumerate(ext)])
    with np.load(a.reference,allow_pickle=False) as ref:
        z_full=ref["z_full"]
        nec_z_re=z_full[:64,64:]
        h0_ref=ref["h_direct_without_ris"]
        h1_ref=ref["h_direct_monolithic_open"]
        delta_ref=ref["delta_z_open_structure"]
    point_metrics=[]
    for src in range(2):
        for k in range(65):
            x,y=rt_e[src,k],nec_e[src,k]
            u=y/np.linalg.norm(y)
            stats=complex_metrics(np.array([np.vdot(u,x)]),np.array([np.linalg.norm(y)]))
            stats["cross_polar_leakage"]=cross_polar_leakage(x,u)
            stats["path_present"]=bool(available[src,k])
            stats["receiver"]=(f"RIS{k:02d}" if k<64 else ("UE0" if src==0 else "BS0"))
            stats["source"]="BS0" if src==0 else "UE0"
            point_metrics.append(stats)
    m={"incident_ris_ports":complex_metrics(rt_z_re,nec_z_re),
       "direct_rt_vs_free_nec":complex_metrics(h0,h0_ref),
       "direct_rt_corrected_vs_open_nec":complex_metrics(h1,h1_ref),
       "predicted_delta_z":complex_metrics(delta_z,delta_ref)}
    points_ok=all(row["path_present"] and row["all_scalar_pass"] and
                  row["cross_polar_leakage"]<=P4_LIMITS["cross_polar_leakage_max"] for row in point_metrics)
    p0={"status":"measured","passed":bool(points_ok and all(v["all_scalar_pass"] for v in m.values())),
        "metrics":m,"available_paths":available.sum(axis=1).tolist(),"ports_per_source":65,
        "point_metrics":point_metrics,"rt_incident_v_m_per_feed_a":pack(rt_e),
        "independent_nec_incident_v_m_per_feed_a":pack(nec_e),
        "rt_ris_z_re_ohm":pack(rt_z_re),"rt_z_external_ohm":pack(z_ee_rt),
        "ris_only_delta_z_ohm":pack(delta_z),"rt_direct_h":pack(h0),"rt_corrected_h":pack(h1),
        "matrix_definition":"RT offdiagonal, isolated NEC external self impedances",
        "reference_source":"independent monolithic NEC and isolated-source NE-card field vectors",
        "open_gap_independence":"RIS-only current matrices + RT fields; no reference data used for prediction"}
    out={"criteria":P4_LIMITS,"calibration_fit_applied":False,
         "trace_sha256":sha256(a.trace.read_bytes()).hexdigest(),
         "reference_sha256":g["reference_sha256"],"munich_los":p0,
         "munich_reflection":{"status":"blocked","reason":"no matching monolithic NEC model of the built-in urban reflector geometry",
         "rt_path_counts": [x["path_counts"] for x in trace["modes"]["munich_first_order_inventory"]]},
         "gate":"P4_NOT_PASSED"}
    if a.reflection_reference:
        refl_seg, refl_e, refl_z, refl_avail=fields(trace["modes"]["pec_specular"],ris,ext,rr,er,True)
        with np.load(a.reflection_reference,allow_pickle=False) as f:
            keys=set(f.files)
            if not {"pec_reflected_e_xyz","pec_z_external","free_z_external"}<=keys:
                raise ValueError(f"unexpected reflection reference fields {sorted(keys)}")
            ref_e=f["pec_reflected_e_xyz"]
            dz=f["pec_z_external"]-f["free_z_external"]
        if ref_e.shape != refl_e.shape:
            raise ValueError(f"expected reflected reference {refl_e.shape}, got {ref_e.shape}")
        projs=np.sum(np.conj(ref_e)*refl_e,axis=-1)/np.linalg.norm(ref_e,axis=-1)
        e_m=complex_metrics(projs,np.linalg.norm(ref_e,axis=-1))
        zm=complex_metrics(refl_z,np.array([dz[1,0],dz[0,1]]))
        x=max(cross_polar_leakage(a,b) for a,b in zip(refl_e.reshape(-1,3),ref_e.reshape(-1,3)))
        out["controlled_pec_reflection"]={"status":"measured",
            "reference_sha256":sha256(a.reflection_reference.read_bytes()).hexdigest(),
            "available_paths":refl_avail.sum(axis=1).tolist(),"incident_fields":e_m,
            "mutual_external_reflected_z":zm,"max_cross_polar":x,
            "passed":bool(refl_avail.all() and e_m["all_scalar_pass"] and zm["all_scalar_pass"] and x<=.01),
            "limit":"PEC controlled diagnostic, not a Munich urban reflector reference"}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    tmp.replace(a.out)
    print("LOS pass",p0["passed"],"paths",p0["available_paths"],"metrics",m,flush=True)


if __name__=="__main__":
    main()
