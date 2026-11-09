"""Bounded Sionna 2.0.1 V/H P4 trace of the registered development geometry."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version

import numpy as np

PLATE = """<scene version="3.0.0">
<bsdf type="radio-material" id="p4-ground"><float name="relative_permittivity" value="1"/>
<float name="conductivity" value="1e12"/><float name="thickness" value="1"/></bsdf>
<shape type="rectangle" id="p4-plane"><transform name="to_world"><scale x="1000" y="1000" z="1"/></transform>
<ref id="p4-ground" name="bsdf"/></shape></scene>"""


def trace(rt, scene_name, source, ris, other, *, reflected, urban=False):
    scene = rt.load_scene(str(scene_name), merge_shapes=False)
    scene.frequency = 5.53e9
    scene.tx_array = scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
        vertical_spacing=.5, horizontal_spacing=.5, pattern="iso", polarization="VH")
    scene.add(rt.Transmitter(name="p4_tx", position=source.tolist()))
    for i, pos in enumerate(np.vstack((ris, other))):
        scene.add(rt.Receiver(name=f"p4_rx{i:02d}", position=pos.tolist()))
    paths = rt.PathSolver()(scene, max_depth=int(reflected), los=not reflected or urban,
        specular_reflection=reflected, diffuse_reflection=False,
        diffraction=False, refraction=False, synthetic_array=False,
        samples_per_src=4096 if reflected else 1,
        max_num_paths_per_src=10000 if reflected else 100, seed=7)
    a = np.asarray(paths.a[0]) + 1j*np.asarray(paths.a[1])
    tau = np.asarray(paths.tau)
    if a.ndim != 5 or a.shape[:4] != (65,2,1,2):
        raise ValueError(f"unexpected Sionna paths.a shape {a.shape}")
    if tau.shape != a.shape:
        raise ValueError(f"unexpected Sionna paths.tau shape {tau.shape}")
    counts = []
    rows = []
    for i in range(65):
        mask = np.any(np.abs(a[i,:,0,:,:]) > 0, axis=(0,1))
        choices = np.flatnonzero(mask)
        counts.append(len(choices))
        if urban:
            continue
        if len(choices) != 1:
            rows.append(None)
            continue
        j = choices[0]
        h = a[i,:,0,:,j]
        t = tau[i,:,0,:,j][np.abs(h)>0]
        if not np.isfinite(h).all() or not np.isfinite(t).all() or (t <= 0).any() or np.ptp(t) > 1e-12:
            raise ValueError("nonfinite or non-colocated Jones path")
        rows.append({"delay_s":float(t[0]), "jones":np.stack((h.real,h.imag),axis=-1).tolist()})
    return {"path_counts":counts, "rows":rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--geometry",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    g=json.loads(args.geometry.read_text())
    ris,ext=np.asarray(g["ris"]),np.asarray(g["external"])
    if ris.shape!=(64,3) or ext.shape!=(2,3):
        raise ValueError("expected registered RIS + BS0 + UE0")
    import drjit as dr
    import mitsuba as mi
    mi.set_variant("llvm_ad_mono_polarized")
    dr.set_thread_count(2)
    import sionna.rt as rt
    if version("sionna-rt")!="2.0.1":
        raise RuntimeError("requires pinned Sionna RT 2.0.1")
    plate=args.out.with_name("p4-plane.xml")
    plate.write_text(PLATE)
    out={"sionna_rt":version("sionna-rt"),"mitsuba_variant":mi.variant(),"modes":{}}
    for name,scene,reflect,urban in (
        ("munich_los",rt.scene.munich,False,False),
        ("pec_specular",plate,True,False),
        ("munich_first_order_inventory",rt.scene.munich,True,True)):
        print("trace",name,flush=True)
        out["modes"][name]=[trace(rt,scene,ext[i],ris,ext[1-i],reflected=reflect,urban=urban)
            for i in range(2)]
    tmp=args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out,allow_nan=False)+"\n")
    tmp.replace(args.out)


if __name__=="__main__":
    main()
