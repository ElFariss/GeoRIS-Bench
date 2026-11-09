"""Complete the registered P4 verdict without substituting a controlled reflection."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from atomwise_ris.hybrid.p4_calibration import P4_LIMITS, complex_metrics, cross_polar_leakage
from atomwise_ris.hybrid.p4_rt_bridge import response, field_from_jones
from p4_score_registered_ports import receive, pack, fields as segmented_fields


def fields(mode, ris, ext, sw, reflected, nec_medium):
    e, z, present = [], [], []
    for i in range(2):
        rows = mode[i]["rows"]
        if len(rows) != 65:
            raise ValueError("missing receiver inventory")
        present.append([x is not None for x in rows])
        e.append([field_from_jones(row, ext[i], pos, sw[i], pos[None,:],
                    reflected,nec_medium)[0] for pos,row in zip(np.vstack((ris,ext[1-i])),rows,strict=True)])
        dw = sw[1-i]
        z.append(receive(dw, field_from_jones(rows[-1],ext[i],ext[1-i],sw[i],
                   dw.segment_centers,reflected,nec_medium))[0])
    return np.asarray(e),np.asarray(z),np.asarray(present)


def compare_fields(actual,expected,present):
    n = np.linalg.norm(expected,axis=-1)
    if (n == 0).any():
        raise ValueError("reference field direction is undefined")
    v = (np.conj(expected)*actual).sum(axis=-1)/n
    proj = complex_metrics(v,n)
    per_port = [complex_metrics(np.asarray([x]),np.asarray([y]))
                for x,y in zip(v.ravel(),n.ravel(),strict=True)]
    cross = max(cross_polar_leakage(a,b) for a,b in zip(actual.reshape(-1,3),expected.reshape(-1,3),strict=True))
    return {"aggregate_projection":proj, "maximum_cross_polar_leakage":cross,
            "worst_relative_port_error":max(x["complex_relative_error"] for x in per_port),
            "worst_phase_rad":max(x["wrapped_phase_error_rad"] for x in per_port),
            "available":present.sum(axis=1).tolist(),
            "passed":bool(present.all() and all(x["all_scalar_pass"] for x in per_port)
                          and cross<=P4_LIMITS["cross_polar_leakage_max"])}


def main():
    p=argparse.ArgumentParser()
    for k in ("trace","geometry","los","reference","out"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--reflection-reference",type=Path)
    a=p.parse_args()
    g=json.loads(a.geometry.read_text())
    los=json.loads(a.los.read_text())
    t=json.loads(a.trace.read_text())
    if (sha256(a.trace.read_bytes()).hexdigest()!=los["trace_sha256"] or
            sha256(a.reference.read_bytes()).hexdigest()!=g["reference_sha256"]):
        raise ValueError("trace or reference provenance mismatch")
    ris,ext=np.asarray(g["ris"]),np.asarray(g["external"])
    sw=[response(c[None,:]) for c in ext]
    phys_e, phys_z, present=fields(t["modes"]["munich_los"],ris,ext,sw,False,False)
    f=np.asarray(los["munich_los"]["independent_nec_incident_v_m_per_feed_a"])
    with np.load(a.reference,allow_pickle=False) as nec:
        z=nec["z_ee_without_ris"]
        z_re_free=nec["z_full"][:64,64:]
        n_z=np.array([z[1,0],z[0,1]])
    out={"criteria":P4_LIMITS,"calibration_fit_applied":False,
        "registered_los_artifact_sha256":sha256(a.los.read_bytes()).hexdigest(),
        "registered_munich_los":{k:v for k,v in los["munich_los"].items()
                              if k in ("status","passed","metrics","available_paths")},
        "physical_si_vs_nec_medium_diagnostic":{"fields":compare_fields(phys_e,f[...,0]+1j*f[...,1],present),
             "external_mutual":complex_metrics(phys_z,n_z),"note":"exact SI propagation; intentionally not converted to the NEC numerical medium"},
        "registered_munich_urban_reflection":los["munich_reflection"],
        "controlled_pec_reflection":{"status":"blocked","reason":"independent PEC reference unavailable"},
        "full_p4":{"status":"blocked","passed":False,"gate":"P4_NOT_PASSED",
            "reason":"A controlled PEC reflector is not an independent reference for the original Munich urban specular paths."}}
    if a.reflection_reference:
        with np.load(a.reflection_reference,allow_pickle=False) as ref:
            np.testing.assert_array_equal(ref["ris_xyz_m"],ris)
            np.testing.assert_array_equal(ref["external_xyz_m"],ext)
            fe=ref["pec_reflected_e_xyz"]
            dz=ref["pec_z_external"]-ref["free_z_external"]
            dz_re=ref["pec_monolithic_z"][:64,64:]-z_re_free
        e,z,present=fields(t["modes"]["pec_specular"],ris,ext,sw,True,True)
        em=compare_fields(e,fe,present)
        zm=complex_metrics(z,np.array([dz[1,0],dz[0,1]]))
        out["controlled_pec_reflection"]={"status":"measured","passed":bool(em["passed"] and zm["all_scalar_pass"]),
             "fields":em,"reflected_mutual_impedance":zm,
             "reference_sha256":sha256(a.reflection_reference.read_bytes()).hexdigest(),
             "rt_reflected_field_v_m_per_feed_a":pack(e),
             "limit":"controlled z=0 PEC-limit plane with the same port centres; not the registered urban material or geometry"}
        # This additional comparison tests single reflected illumination against
        # the full PEC/free difference, which also includes secondary coupling.
        rr=response(ris)
        e_seg, _, _, _ = segmented_fields(t["modes"]["pec_specular"],ris,ext,rr,sw,True)
        z_re = np.column_stack([receive(rr,e_seg[...,i]) for i in range(2)])
        out["controlled_pec_reflection"]["single_bounce_ris_vs_monolithic_pec_difference"] = complex_metrics(z_re,dz_re)
    tmp=a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    tmp.replace(a.out)
    print({k:v.get("passed",v.get("status")) for k,v in out.items() if isinstance(v,dict)},flush=True)


if __name__=="__main__":
    main()
