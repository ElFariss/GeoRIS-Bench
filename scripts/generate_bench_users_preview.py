#!/usr/bin/env python3
"""Generate the declared v1 50 preview cases and anonymous natural distributions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atomwise_ris.bench import users as u
from atomwise_ris.bench.users_geometry import load_development_maps

NATURAL_PER_MAP = 14
BULK = (1667, 1667, 1666)
CHALLENGES = (
    ("shadow_edge", "munich"), ("shadow_edge", "munich"),
    ("shadow_edge", "rotterdam"), ("shadow_edge_sloped", "maastricht"),
    ("all_no_los", "rotterdam"), ("all_no_los", "maastricht"),
    ("dense_crowd", "rotterdam"), ("elevated_crowd", "munich"),
)


def stream(*keys):
    return np.random.default_rng(np.random.SeedSequence([u.SEED,*keys]))


def dump(p, data):
    tmp = p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    tmp.replace(p)


def table(p, values):
    tmp = p.with_suffix(".parquet.tmp")
    # from_pandas: NaN float values (e.g. slope of elevated/drone users) become nulls
    pq.write_table(pa.table({k:pa.array(v,from_pandas=True) for k,v in values.items()}),tmp,
                   compression="zstd",compression_level=4)
    tmp.replace(p)


def osm_point_hotspots(g,cache):
    """Add actual node POIs; project doorsteps to nearby valid street support."""
    from pyproj import Transformer
    path=Path(cache)/f"{g.map_id}.osm"
    if path.name not in g.city.source_hashes:
        raise ValueError(f"{g.map_id}: OSM hotspot file is not source-verified")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=g.city.source_hashes[path.name]:
        raise ValueError(f"{g.map_id}: OSM hotspot snapshot changed")
    tr=Transformer.from_crs(4326,g.city.crs,always_xy=True)
    counts,weights,coords=Counter(),[],[]
    for node in ET.parse(path).getroot().findall("node"):
        tags={t.attrib["k"]:t.attrib["v"] for t in node.findall("tag")}
        if (tags.get("location")=="underground" or tags.get("tunnel") in {"yes","true","1"}
                or tags.get("access") in {"private","no"}):
            continue
        highway=tags.get("highway")
        if (highway=="bus_stop" or tags.get("public_transport") in {"stop_position","platform","station"}
            or tags.get("railway") in {"tram_stop","subway_entrance","halt","station"}
            or tags.get("amenity")=="bus_station"):
            kind,w="transit_stop",3.5
        elif highway in {"crossing","traffic_signals"} or tags.get("crossing") not in {None,"no"}:
            kind,w="crossing",2.2
        elif tags.get("entrance") not in {None,"no","emergency","exit"}:
            kind,w="entrance",1.3
        elif tags.get("place") in {"square","plaza"} or tags.get("amenity")=="marketplace":
            kind,w="square",3.
        else:
            continue
        xy=np.array(tr.transform(float(node.attrib["lon"]),float(node.attrib["lat"])))
        if not g.within(xy[None])[0] or g.streets.tree.query(xy)[0]>25.:
            counts[kind+"_unusable"]+=1
            continue
        counts[kind]+=1
        coords.append(xy)
        weights.append(w)
    if coords:
        # The base geometry pool is normalized for sampling; restore its raw
        # OSM/junction weights before combining the actual OSM node categories.
        g.hotspots=np.concatenate([g.hotspots,np.asarray(coords)],axis=0)
        g.weights=np.r_[np.asarray(g.city.hotspot_weights),weights]
        g.weights=g.weights/g.weights.sum()
    return dict(counts)


def validated_cities(cache, geometry_gate, out, ground_fill=False, source_manifest=None):
    if not geometry_gate.is_file():
        raise ValueError(f"slope gate required before sampling: {geometry_gate}")
    gate = json.loads(geometry_gate.read_text())
    if set(gate.get("maps",{}))!=set(u.DEVELOPMENT):
        raise ValueError("slope artifact does not cover the three development maps")
    gate_hash = hashlib.sha256(geometry_gate.read_bytes()).hexdigest()
    maps = load_development_maps(cache_root=cache,area_ids=u.DEVELOPMENT,build_sites=False,
                                 manifest_path=source_manifest)
    if set(maps)!=set(u.DEVELOPMENT):
        raise ValueError("expected exactly three approved development maps")
    verdict = {}
    for key,city in maps.items():
        if city.source_hashes!=gate["maps"][key]["checksummed_sources"]:
            raise ValueError(f"{key}: slope gate references a different source snapshot")
        street = np.asarray(city.street_segments)
        z,sl,valid = city.sample_ground(street.reshape(-1,2))
        if not len(street) or not np.asarray(valid).all() or not np.isfinite(z).all():
            raise ValueError(f"{key}: street terrain gate did not pass")
        length=np.linalg.norm(street[:,1]-street[:,0],axis=1)
        grade=np.degrees(np.arctan2(abs(z[1::2]-z[::2]),length))
        verdict[key]={"valid_street_segments":len(street),"valid_street_length_m":float(length.sum()),
                      "geometry_rejections":getattr(city,"geometry_rejections",{}),
                      "street_grade_deg_q50_q90_q95_max":np.percentile(grade,[50,90,95,100]).tolist(),
                      "street_dtm_slope_deg_q50_q90_q95_max":np.percentile(sl,[50,90,95,100]).tolist()}
    if ground_fill:
        from atomwise_ris.bench.ground_fill import prepare
        audit=prepare(maps,cache,out/"ground_fill")
        print("GROUND_FILL",json.dumps({k:v["counts"] for k,v in audit["maps"].items()}),
              "holdout_enabled",audit["interpolated_use_enabled"],flush=True)
    gens={k:u.Generator(v,k,site_rng=stream(5,u.DEVELOPMENT.index(k))) for k,v in maps.items()}
    pool={}
    for k,g in gens.items():
        if len(g.sites)<6:
            raise ValueError(f"{k}: fewer than six planned BS/RIS sites ({len(g.sites)})")
        # Independent recheck of every planned site before any case is drawn.
        bs=np.asarray([s["bs_xyz"] for s in g.sites]); ris=np.asarray([s["ris_xyz"] for s in g.sites])
        vis,_=g.city.los_batch(bs,ris)
        dist=np.linalg.norm(bs-ris,axis=1)
        if not np.asarray(vis).all():
            raise ValueError(f"{k}: planned site pool contains an occluded BS-RIS link")
        if not ((dist>=u.SITE_DISTANCE_M[0])&(dist<=u.SITE_DISTANCE_M[1])).all():
            raise ValueError(f"{k}: planned site outside the declared BS-RIS distance range")
        verdict[k].update({"planned_sites":len(g.sites),"site_stats":g.site_stats,
                          "street_prune":g.street_prune,"roof_support_points":len(g.roofs),
                          "roof_support_rejections":g.roof_rejections,
                          "eligible_osm_point_hotspots":osm_point_hotspots(g,cache),
                          "combined_hotspot_candidates":len(g.hotspots)})
        pool[k]=g.sites
    dump(out/"site_pool.json",{"slope_cap_deg":u.WALK_SLOPE_CAP_DEG,"distance_range_m":list(u.SITE_DISTANCE_M),"maps":pool})
    dump(out/"generator_slope_gate.json",{"status":"PASS","geometry_gate":str(geometry_gate),
                                         "geometry_gate_sha256":gate_hash,"maps":verdict})
    print("SLOPE_PASS",json.dumps({k:{x:v[x] for x in ("planned_sites","site_stats","street_prune")} for k,v in verdict.items()}),flush=True)
    return gens


def natural(g, t, rng, counts):
    for _ in range(35):
        c = g.natural(t,rng,counts)
        if c is not None:
            return c
        counts["empty_natural_crowd"] += 1
    raise RuntimeError(f"{g.map_id}: empty crowd attempts exhausted")


def create_challenge(gens, category, preferred, t, rng, counts):
    for map_id in [preferred,*[k for k in u.DEVELOPMENT if k!=preferred]]:
        g = gens[map_id]
        if category.startswith("shadow"):
            res = g.shadow(t,rng,counts,slope_required=category.endswith("sloped"))
            if res is not None:
                site,c,change=res
                return g,site,c,{"shadow_user_id":0,"first_shadow_transition_frame":change,
                                 "leader_ground_relief_m":float(np.ptp(c.ground[:,0]))}
        elif category=="all_no_los":
            res=g.no_los_group(t,rng,counts)
            if res is not None:
                return g,*res,{}
        elif category=="dense_crowd":
            for _ in range(250):
                c=natural(g,t,rng,counts)
                if c.size>=60:
                    return g,g.choose_site(rng,counts),c,{}
                counts["challenge_density_below_60"] += 1
        elif category=="elevated_crowd":
            c=g.elevated(t,rng,counts)
            if c is not None:
                return g,g.choose_site(rng,counts),c,{}
        counts["challenge_preferred_map_unavailable"] += 1
    raise RuntimeError(f"challenge exhausted under fixed definition: {category}")


def write_case(out,g,site,c,t,case_id,kind,scenario,counts,rng,extra):
    p = out/"cases"/case_id
    p.mkdir(parents=True,exist_ok=True)
    f=len(t)
    labels=u.user_labels(g,site,c,t)
    if not np.isfinite(c.xyz).all() or not np.isfinite(c.ground).all():
        raise ValueError("invalid terrain or coordinates in case")
    ris_frames=labels["ris_los"].reshape(f,c.size)
    reach=u.ris_reachable(ris_frames)
    force=(extra["shadow_user_id"],) if scenario and scenario.startswith("shadow") else ()
    served=u.schedule(reach,rng,counts,force=force)
    labels["served"]=np.tile(served,f)
    if scenario=="all_no_los" and not np.all(labels["bs_link"]==u.NO_LOS):
        raise ValueError("all-no-LOS challenge no longer satisfies the declared label")
    if scenario=="dense_crowd" and c.size<60:
        raise ValueError("dense challenge below declared minimum")
    if scenario=="elevated_crowd" and np.sum(c.height=="elevated")<5:
        raise ValueError("elevated challenge below declared minimum")
    if scenario and scenario.startswith("shadow"):
        leader=labels["bs_link"].reshape(len(t),-1)[:,0]==u.LOS
        if np.all(leader) or not np.any(leader):
            raise ValueError("shadow challenge lost its LOS state change")
    result={"id":case_id,"map_id":g.map_id,"kind":kind,"scenario":scenario,
            "duration_s":float(t[-1]),"dt_s":u.DT,"frame_count":f,"crowd_size":c.size,
            "group_sizes":np.unique(c.group,return_counts=True)[1].tolist(),
            "served_count":int(served.sum()),"served_user_ids":np.flatnonzero(served).tolist(),
            "movement":bool(np.any(c.speed>.015)),
            "users_moving":int(np.any(c.speed>.015,axis=0).sum()),
            "height_class_counts":dict(Counter(c.height.tolist())),
            "initial_link_counts":dict(Counter(labels["bs_link"][:c.size].tolist())),
            "all_frame_link_counts":dict(Counter(labels["bs_link"].tolist())),
            "labels_rows":len(labels["frame"]),"rejections":dict(counts),
            "distance_bs_ris_m":float(np.linalg.norm(site["bs_xyz"]-site["ris_xyz"])),
            "walk_slope_cap_deg":u.WALK_SLOPE_CAP_DEG,"site_distance_range_m":list(u.SITE_DISTANCE_M),
            "ris_los_users_first_frame":int(ris_frames[0].sum()),
            "ris_reachable_users":int(reach.sum()),"served_ris_los_first_frame":int(ris_frames[0][served].sum()),
            "served_ris_reachable":int(reach[served].sum()),
            "max_ground_relief_m":float(np.ptp(c.ground,axis=0).max()),
            "max_speed_mps":float(c.speed.max()),
            "protocol_seed":u.SEED,
            **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in site.items()},**extra}
    table(p/"frames.parquet",labels)
    result["frames_sha256"]=hashlib.sha256((p/"frames.parquet").read_bytes()).hexdigest()
    dump(p/"case.json",result)
    print("CASE",case_id,result["crowd_size"],result["frame_count"],result["initial_link_counts"],
          "ris_los",result["ris_los_users_first_frame"],"served_ris_los",result["served_ris_los_first_frame"],
          "/",result["served_count"],flush=True)
    return result


def existing(out,case_id):
    p=out/"cases"/case_id
    if not (p/"case.json").is_file():
        return None
    m=json.loads((p/"case.json").read_text())
    if (m["protocol_seed"]!=u.SEED or m["id"]!=case_id or
        hashlib.sha256((p/"frames.parquet").read_bytes()).hexdigest()!=m["frames_sha256"]):
        raise ValueError(f"completed case failed identity/hash validation: {case_id}")
    return m


def preview(gens,out):
    rows=[]
    for im,map_id in enumerate(u.DEVELOPMENT):
        g=gens[map_id]
        for k in range(NATURAL_PER_MAP):
            case_id=f"{1+len(rows):02d}-{map_id}-natural"
            if (saved:=existing(out,case_id)) is not None:
                rows.append(saved)
                continue
            rng,counts=stream(1,im,k),Counter()
            t=u.times(rng)
            site=g.choose_site(rng,counts)
            c=natural(g,t,rng,counts)
            rows.append(write_case(out,g,site,c,t,case_id,"natural",None,counts,rng,{}))
    for k,(scenario,preferred) in enumerate(CHALLENGES):
        case_id=f"{1+len(rows):02d}-challenge-{scenario.replace('_','-')}"
        if (saved:=existing(out,case_id)) is not None:
            rows.append(saved)
            continue
        rng,counts=stream(2,k),Counter()
        t=u.times(rng)
        g,site,c,extra=create_challenge(gens,scenario,preferred,t,rng,counts)
        rows.append(write_case(out,g,site,c,t,case_id,"challenge",scenario,counts,rng,extra))
    assert len(rows)==50
    dump(out/"case_manifest.json",{"seed":u.SEED,"retained_cases":len(rows),"cases":rows})
    return rows


def bulk(gens,out):
    if (out/"bulk_summary.json").is_file():
        m=json.loads((out/"bulk_summary.json").read_text())
        if m["draws_sha256"]!=hashlib.sha256((out/"draws.parquet").read_bytes()).hexdigest():
            raise ValueError("bulk draw artifact hash mismatch")
        return m
    records=defaultdict(list)
    stats={}
    for im,(key,n) in enumerate(zip(u.DEVELOPMENT,BULK,strict=True)):
        g=gens[key]
        counts,cl=Counter(),Counter()
        t=np.zeros(1)
        size_hist, group_hist=Counter(),Counter()
        for k in range(n):
            rng=stream(3,im,k)
            site=g.choose_site(rng,counts)
            c=natural(g,t,rng,counts)
            xyz=c.xyz[0]
            links,_,ris_los,_=g.link(site,xyz)
            served=u.schedule(ris_los,rng,counts)
            bdist,_,_=u.angles(site["bs_xyz"],xyz)
            rdist,_,_=u.angles(site["ris_xyz"],xyz)
            cl.update(links.tolist())
            size_hist[c.size]+=1
            _,first,sizes=np.unique(c.group,return_index=True,return_counts=True)
            group_hist.update(sizes.tolist())
            gc=[None]*c.size
            for i,s in zip(first,sizes,strict=True):
                gc[i]=int(s)
            slope=[None if not np.isfinite(x) else float(x) for x in c.slope[0]]
            for name,value in {
                "distribution":["bulk_natural"]*c.size,"map_id":[key]*c.size,
                "crowd_size":[c.size]+[None]*(c.size-1),"group_size":gc,
                "bs_ris_distance_m":[site["bs_ris_distance_m"]]+[None]*(c.size-1),
                "height_agl_m":c.agl[0].tolist(),"height_class":c.height.tolist(),
                "speed_mps":c.speed[0].tolist(),"slope_deg":slope,
                "bs_distance_m":bdist.tolist(),"ris_distance_m":rdist.tolist(),
                "bs_link":links.tolist(),"ris_los":ris_los.tolist(),"served":served.tolist()}.items():
                records[name].extend(value)
            if (k+1)%250==0:
                print("BULK",key,k+1,n,flush=True)
        stats[key]={"cases":n,"rejections":dict(counts),"link_counts":dict(cl),
                    "crowd_size_histogram":dict(size_hist),"group_size_histogram":dict(group_hist)}
    order=stream(4).permutation(len(records["bs_link"]))
    table(out/"draws.parquet",{k:pa.array(v).take(pa.array(order)) for k,v in records.items()})
    m={"status":"PASS","cases":sum(BULK),"user_observations":len(records["bs_link"]),
       "case_files_saved":0,"scope":"anonymous natural snapshot distribution rows only","maps":stats,
       "draws_sha256":hashlib.sha256((out/"draws.parquet").read_bytes()).hexdigest()}
    dump(out/"bulk_summary.json",m)
    return m


def lock_inputs(out,gate,ground_fill,source_manifest=None):
    paths=[Path(__file__),Path(u.__file__),Path(u.__file__).with_name("users_geometry.py"),
           Path("docs/bench-users-v1-design-20261108.md"),gate]
    if ground_fill:
        paths += [Path(u.__file__).with_name("ground_fill.py"),Path("docs/bench-tracks-ground-v1.1-20261108.md")]
    if source_manifest is not None:
        paths.append(source_manifest)
    wanted={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    path=out/"generation_inputs.json"
    if path.exists():
        if json.loads(path.read_text())!=wanted:
            raise ValueError("generation inputs changed; do not mix partial output with another protocol")
    else:
        dump(path,wanted)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--geometry-gate",type=Path,required=True)
    p.add_argument("--source-manifest",type=Path)
    p.add_argument("--ground-fill",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.ground_fill and args.out.resolve().name!="bench-users-v1.1":
        raise ValueError("building-only fills must use separate bench-users-v1.1 output")
    if args.source_manifest and not args.ground_fill:
        raise ValueError("an alternate source manifest is supported only for the separate v1.1 preview")
    lock_inputs(args.out,args.geometry_gate,args.ground_fill,args.source_manifest)
    gens=validated_cities(args.cache,args.geometry_gate,args.out,args.ground_fill,args.source_manifest)
    rows=preview(gens,args.out)
    b=bulk(gens,args.out)
    print("DATA_DONE",len(rows),b["cases"],flush=True)


if __name__=="__main__":
    main()
