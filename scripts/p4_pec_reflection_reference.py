"""Independent GN=1 perfect-ground P4 comparison on the registered centres."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from atomwise_ris.physics.nec import WireGeometry
from atomwise_ris.hybrid.p4_rt_bridge import FREQ


def ctx(points, perfect_ground):
    n=WireGeometry(points,np.full(len(points),.024),frequency=FREQ).context(81)
    if perfect_ground:
        n.gn_card(1,0,0.,0.,0.,0.,0.,0.)
    return n


def impedance(points, ground):
    n=ctx(points,ground)
    ports=81*np.arange(len(points))+40
    y=[]
    for i in range(len(points)):
        n.ex_card(0,i+1,41,0,1,0,0,0,0,0)
        n.xq_card(0)
        r=n.get_structure_currents(i)
        if abs(r.get_frequency()-FREQ)>1:
            raise ValueError("NEC frequency mismatch")
        y.append(np.asarray(r.get_current())[ports].copy())
    return np.linalg.inv(np.column_stack(y))


def nec_field(source, points, ground):
    n=ctx(source[None,:],ground)
    n.ex_card(0,1,41,0,1,0,0,0,0,0)
    output=[]
    for i,(x,y,z) in enumerate(points):
        n.ne_card(0,1,1,1,float(x),float(y),float(z),0.,0.,0.)
        field=n.get_near_field_pattern(i)
        loc=[getattr(field,f"get_{a}")()[0] for a in "xyz"]
        np.testing.assert_allclose(loc,[x,y,z],rtol=0,atol=2e-9)
        output.append([getattr(field,f"get_field_{a}")()[0] for a in "xyz"])
    feed=complex(n.get_structure_currents(0).get_current()[40])
    if abs(feed)==0:
        raise ValueError("zero NEC feed current")
    return np.array(output)/feed


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--geometry",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    g=json.loads(a.geometry.read_text())
    ris,ext=np.asarray(g["ris"]),np.asarray(g["external"])
    if ris.shape!=(64,3) or ext.shape!=(2,3) or min(ris[:,2].min(),ext[:,2].min())<=.012:
        raise ValueError("incompatible PEC plane geometry")
    if version("PyNEC")!="2.3.4":
        raise ValueError("requires pinned PyNEC 2.3.4")
    res={}
    for name,ground in (("free",False),("pec",True)):
        print(name,"NEC NE electric fields and external-only ports",flush=True)
        res[name+"_e_xyz"]=np.stack([nec_field(tx,np.vstack((ris,ext[1-i])),ground)
                                   for i,tx in enumerate(ext)])
        res[name+"_z_external"]=impedance(ext,ground)
    print("independent monolithic 66-port NEC with perfect ground",flush=True)
    res["pec_monolithic_z"]=impedance(np.vstack((ris,ext)),True)
    res["pec_reflected_e_xyz"]=res["pec_e_xyz"]-res["free_e_xyz"]
    for k,x in res.items():
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite {k}")
    tmp=a.out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,**res,ris_xyz_m=ris,external_xyz_m=ext)
    tmp.replace(a.out)
    print("reference SHA-256",sha256(a.out.read_bytes()).hexdigest(),flush=True)


if __name__=="__main__":
    main()
