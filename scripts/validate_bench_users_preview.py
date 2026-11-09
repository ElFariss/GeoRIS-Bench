#!/usr/bin/env python3
"""Read back every v1 case: DTM stencils, slope cap, null slopes, sites and RIS reach.

Independent of the generator: ground, slope and site-reference checks are
recomputed from the exported native terrain arrays and the case JSON.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEV = ("munich","rotterdam","maastricht")
SLOPE_CAP_DEG = 15.
SITE_RANGE_M = (20.,150.)
RIS_AGL_M = (3.,30.)
NATURAL_SERVED_RIS_LOS_MIN = .5     # declared sanity floor, see the v1 design doc
LINKS = {"los","no_los_reflection_candidate","no_los","no_los_unresolved"}


def quantiles(a):
    a=np.asarray(a,dtype=float)
    a=a[np.isfinite(a)]
    return np.percentile(a,[0,5,50,90,95,99,100]).tolist() if len(a) else []


def load_terrain(root,map_id):
    with np.load(root/"geometry"/f"{map_id}-terrain.npz",allow_pickle=False) as f:
        return tuple(f[k].astype(float) for k in ("x","y","z"))+(f["valid"].astype(bool),)


def _cells(x,y,terrain):
    xs,ys,_,_=terrain
    cx=(x-xs[0])/(xs[1]-xs[0])
    cy=(y-ys[0])/(ys[1]-ys[0])
    i,j=np.floor(cx).astype(int),np.floor(cy).astype(int)
    return i,j,cx-i,cy-j


def stencil_valid(x,y,terrain):
    """True where the FULL 4x4 DTM stencil around each point is inside and valid."""
    xs,ys,_,valid=terrain
    i,j,_,_=_cells(np.atleast_1d(x),np.atleast_1d(y),terrain)
    inside=(i>=1)&(j>=1)&(i+2<len(xs))&(j+2<len(ys))
    ok=inside.copy()
    ii,jj=np.where(inside,i,1),np.where(inside,j,1)
    for ix in (-1,0,1,2):
        for iy in (-1,0,1,2):
            ok&=valid[jj+iy,ii+ix]
    return ok


def terrain_elevation(x,y,terrain):
    _,_,dem,_=terrain
    i,j,dx,dy=_cells(x,y,terrain)
    return (1-dy)*((1-dx)*dem[j,i]+dx*dem[j,i+1])+dy*((1-dx)*dem[j+1,i]+dx*dem[j+1,i+1])


def terrain_slope_deg(x,y,terrain):
    """Slope from the four central-difference gradients of the 2x2 patch (independent code)."""
    xs,ys,dem,_=terrain
    i,j,dx,dy=_cells(x,y,terrain)
    xres,yres=abs(xs[1]-xs[0]),abs(ys[1]-ys[0])

    def grad(jj,ii):
        return (dem[jj,ii+1]-dem[jj,ii-1])/(2*xres),(dem[jj-1,ii]-dem[jj+1,ii])/(2*yres)

    g00,g01,g10,g11=grad(j,i),grad(j,i+1),grad(j+1,i),grad(j+1,i+1)
    out=[]
    for k in (0,1):
        out.append((1-dy)*((1-dx)*g00[k]+dx*g01[k])+dy*((1-dx)*g10[k]+dx*g11[k]))
    return np.degrees(np.arctan(np.hypot(*out)))


def verify_ground(x,y,reported,terrain):
    """Every point has a complete valid 4x4 stencil and the reported bilinear height."""
    x,y,reported=np.atleast_1d(x),np.atleast_1d(y),np.atleast_1d(reported)
    if not len(x):
        return 0.
    ok=stencil_valid(x,y,terrain)
    assert ok.all(),f"label used a no-data or out-of-raster stencil ({int((~ok).sum())} points)"
    err=float(np.max(abs(reported-terrain_elevation(x,y,terrain))))
    assert err<5e-4,f"native elevation disagreement {err}"
    return err


def verify_object_ground(layer, ids, xy, reported, refs, rc):
    """Verify recorded native/filled/fallback values and cell ownership."""
    xy=np.asarray(xy,dtype=float).reshape(-1,2)
    refs=np.asarray(refs)
    rc=np.asarray(rc,dtype=int).reshape(-1,2)
    expected,m,expected_rc=layer.lookup(ids,xy)
    assert np.array_equal(m,refs) and np.array_equal(expected_rc,rc)
    np.testing.assert_allclose(expected,reported,rtol=0,atol=1e-6)
    for j in np.flatnonzero(rc[:,0]>=0):
        row,col=rc[j]
        assert layer.owner[row,col]==layer.by_id[str(ids[j])]
        assert not layer.protected[row,col]
        assert np.linalg.norm(xy[j]-[layer.x[col],layer.y[row]])<=1.5+1e-9


def verify_layers(root, terrains):
    from atomwise_ris.bench.ground_fill import GroundLayer, FILLED, MEASURED
    p=root/"ground_fill"
    audit=json.loads((p/"audit.json").read_text())
    layers={}
    for k,(x,y,z,valid) in terrains.items():
        m=audit["maps"][k]["artifact"]
        assert sha(p/m["raster"])==m["raster_sha256"] and sha(p/m["owners"])==m["owners_sha256"]
        layer=GroundLayer.read(p/m["raster"],p/m["owners"])
        np.testing.assert_array_equal(layer.x,x)
        np.testing.assert_array_equal(layer.y,y)
        np.testing.assert_array_equal(layer.source==MEASURED,valid)
        np.testing.assert_array_equal(layer.z[valid],z[valid])
        fill=layer.source==FILLED
        assert not (fill&(layer.protected|valid|(layer.owner==0))).any()
        assert np.isfinite(layer.z[fill]).all()
        assert int(fill.sum())==audit["maps"][k]["counts"]["filled_cells"]
        if not audit["interpolated_use_enabled"]:
            assert not fill.any()
        for i,b in enumerate(layer.buildings,1):
            row,col=np.nonzero((layer.owner==i)&fill)
            if len(row):
                assert np.all(np.abs(layer.z[row,col]-b["native_ground_m"])<=2.5+1e-9)
        layers[k]=layer
    return layers


def verify_site_ground(case,which,terrain,layer=None):
    """BS / RIS height reference: valid-stencil DTM samples, median, AGL consistency."""
    xyz=np.asarray(case[f"{which}_xyz"],dtype=float)
    samples=np.asarray(case[f"{which}_ground_samples"],dtype=float).reshape(-1,3)
    method=case[f"{which}_ground_method"]
    if layer is not None and method!="dtm_at_position":
        ident=case[f"{which}_building_id"]
        cell=case[f"{which}_ground_cell"]
        ground=float(case[f"{which}_ground_z_m"])
        verify_object_ground(layer,[ident],xyz[None,:2],[ground],[method],[cell])
        if cell[0]>=0:
            assert len(samples)==1
            r,c=cell
            np.testing.assert_allclose(samples[0],[layer.x[c],layer.y[r],ground],rtol=0,atol=1e-9)
        else:
            assert not len(samples)
        assert abs(case[f"{which}_agl_m"]-(xyz[2]-ground))<1e-6 and xyz[2]>ground
        return method
    assert len(samples)>=1
    verify_ground(samples[:,0],samples[:,1],samples[:,2],terrain)
    if method=="dtm_at_position":
        assert np.allclose(samples[:,:2],xyz[:2]),f"{which}: own-XY ground sample moved"
    else:
        assert method=="building_foot_dtm" and len(samples)==3
        d=np.linalg.norm(samples[:,:2]-xyz[:2],axis=1)
        assert (d<400).all()
    ground=float(case[f"{which}_ground_z_m"])
    assert abs(ground-float(np.median(samples[:,2])))<1e-6
    assert abs(case[f"{which}_agl_m"]-(xyz[2]-ground))<1e-6,f"{which}: AGL inconsistent with ground reference"
    assert xyz[2]>ground
    return method


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root):
    manifest=json.loads((root/"case_manifest.json").read_text())
    bulk=json.loads((root/"bulk_summary.json").read_text())
    render=json.loads((root/"render_manifest.json").read_text())
    gate=json.loads((root/"generator_slope_gate.json").read_text())
    rows=manifest["cases"]
    assert len(rows)==50==render["rendered"]==render["cases_found"] and not render["errors"]
    assert set(m["map_id"] for m in rows)==set(DEV)
    assert len({m["id"] for m in rows})==50
    assert bulk["cases"]==5000 and bulk["case_files_saved"]==0
    assert sha(root/"draws.parquet")==bulk["draws_sha256"]
    assert len(list((root/"cases").glob("*/case.json")))==50
    assert Counter(m["kind"] for m in rows)=={"natural":42,"challenge":8}
    for k in DEV:
        assert sum(m["kind"]=="natural" and m["map_id"]==k for m in rows)==14
    assert Counter(m["scenario"] for m in rows if m["kind"]=="challenge")=={
        "shadow_edge":3,"shadow_edge_sloped":1,"all_no_los":2,"dense_crowd":1,"elevated_crowd":1}
    maps={k:load_terrain(root,k) for k in DEV}
    requires_overlay=root.name=="bench-users-v1.1"
    layers=verify_layers(root,maps) if requires_overlay or (root/"ground_fill"/"audit.json").is_file() else {}
    map_meta={k:json.loads((root/"geometry"/f"{k}.json").read_text()) for k in DEV}
    initial=defaultdict(list)
    counts=defaultdict(Counter)
    rejections=defaultdict(Counter)
    ris=defaultdict(Counter)
    slope_by_class=defaultdict(list)
    slope_null_by_class=Counter()
    site_methods=Counter()
    max_ground_residual=max_slope_residual=0.
    n_rows=n_persons=0
    per_case=[]
    for m in rows:
        p=root/"cases"/m["id"]
        assert all((p/f).stat().st_size>0 for f in ("map-3d.png","README.md","frames.parquet"))
        assert sha(p/"frames.parquet")==m["frames_sha256"]
        n,f=m["crowd_size"],m["frame_count"]
        d=pq.read_table(p/"frames.parquet")
        assert len(d)==n*f==m["labels_rows"]
        arr={c:d[c].to_numpy(zero_copy_only=False) for c in d.column_names}
        assert np.array_equal(arr["user_id"],np.tile(np.arange(n),f))
        assert np.array_equal(arr["frame"],np.repeat(np.arange(f),n))
        np.testing.assert_allclose(arr["time_s"],.02*arr["frame"],atol=1e-8,rtol=0)
        served=arr["served"][:n].astype(bool)
        assert np.all(arr["served"].reshape(f,n)==arr["served"][:n])
        assert min(2,n)<=int(served.sum())<=min(4,n)
        map_id=m["map_id"]
        terrain=maps[map_id]
        xmin,xmax,ymin,ymax=map_meta[map_id]["bounds"]
        assert ((arr["x"]>=xmin)&(arr["x"]<xmax)&(arr["y"]>=ymin)&(arr["y"]<ymax)).all()
        # --- sites: declared range, BS-RIS LOS flag, valid ground references
        assert m["bs_ris_los"] is True
        dist=float(np.linalg.norm(np.asarray(m["bs_xyz"])-np.asarray(m["ris_xyz"])))
        assert abs(dist-m["distance_bs_ris_m"])<1e-6 and abs(dist-m["bs_ris_distance_m"])<1e-6
        assert SITE_RANGE_M[0]<=dist<=SITE_RANGE_M[1],f"{m['id']}: BS-RIS distance {dist:.1f} outside declared range"
        for which in ("bs","ris"):
            site_methods[(which,verify_site_ground(m,which,terrain,layers.get(map_id)))]+=1
        assert RIS_AGL_M[0]<=m["ris_agl_m"]<=RIS_AGL_M[1]+1e-9
        # --- per-sample ground: rows standing on DTM must have the full 4x4 stencil
        height=arr["height_class"]
        ref=arr["ground_ref"]
        on_dtm=ref=="dtm_at_position"
        assert np.all(on_dtm|(height=="elevated")),"only rooftop users may use the building-foot reference"
        assert np.all(~on_dtm|(height!="elevated"))
        gt=arr["ground_z_m"]
        if layers:
            roofs=height=="elevated"
            # Roof users are static; check every unique (user, reference), then
            # ensure that all repetitions agree with that validated record.
            ids=arr["ground_building_id"][roofs]
            coords=np.c_[arr["x"][roofs],arr["y"][roofs]]
            rc=np.c_[arr["ground_cell_row"][roofs],arr["ground_cell_col"][roofs]]
            keys=np.c_[arr["user_id"][roofs],coords,gt[roofs],rc]
            _,indices,inv=np.unique(keys,axis=0,return_index=True,return_inverse=True)
            inv=np.ravel(inv)
            assert np.array_equal(ids,ids[indices][inv]) and np.array_equal(ref[roofs],ref[roofs][indices][inv])
            verify_object_ground(layers[map_id],ids[indices],coords[indices],gt[roofs][indices],ref[roofs][indices],rc[indices])
            assert np.all(arr["ground_cell_row"][on_dtm]==-1) and np.all(arr["ground_cell_col"][on_dtm]==-1)
        np.testing.assert_allclose(arr["z"]-gt,arr["height_agl_m"],rtol=0,atol=1e-9)
        max_ground_residual=max(max_ground_residual,
                                verify_ground(arr["x"][on_dtm],arr["y"][on_dtm],gt[on_dtm],terrain))
        # --- slope: pedestrians <= cap and equal to the independent recomputation; others null
        slope=arr["slope_deg"].astype(float)
        ped=height=="pedestrian"
        assert np.isfinite(slope[ped]).all() and (slope[ped]>=0).all()
        assert (slope[ped]<=SLOPE_CAP_DEG+1e-9).all(),f"{m['id']}: pedestrian slope above {SLOPE_CAP_DEG} deg"
        assert np.isnan(slope[~ped]).all(),f"{m['id']}: elevated/drone user carries a DTM slope label"
        if ped.any():
            resid=np.max(abs(terrain_slope_deg(arr["x"][ped],arr["y"][ped],terrain)-slope[ped]))
            assert resid<1e-3,f"{m['id']}: slope label disagrees with native terrain ({resid})"
            max_slope_residual=max(max_slope_residual,float(resid))
        for q in ("pedestrian","elevated","drone"):
            sel=height==q
            slope_by_class[q].append(slope[sel])
            slope_null_by_class[q]+=int(np.isnan(slope[sel]).sum())
        for q in ("pedestrian","drone"):
            mask=height==q
            lo,hi=(1.,1.7) if q=="pedestrian" else (20.,80.)
            assert ((arr["height_agl_m"][mask]>=lo-1e-8)&(arr["height_agl_m"][mask]<=hi+1e-8)).all()
        assert (arr["height_agl_m"][height=="elevated"]>=4.-1e-8).all()
        assert set(arr["bs_link"])<=LINKS
        assert np.all(arr["blocker_id"][arr["bs_link"]!="los"]!=None)  # noqa: E711
        assert np.all(arr["blocker_height_source"][arr["bs_link"]!="los"]!="unresolved_height_source")
        xyz=np.stack([arr[c] for c in "xyz"],axis=-1).reshape(f,n,3)
        if f>1:
            v=np.linalg.norm(np.diff(xyz,axis=0),axis=-1)/.02
            np.testing.assert_allclose(v,arr["speed_mps"].reshape(f,n)[1:],rtol=0,atol=1e-9)
            assert np.linalg.norm(np.diff(xyz[...,:2],axis=0),axis=-1).max()<=.204,"instantaneous walking teleport"
        if m["scenario"]=="all_no_los":
            assert np.all(arr["bs_link"]=="no_los")
        if m["scenario"]=="shadow_edge_sloped":
            assert np.ptp(gt.reshape(f,n)[:,m["shadow_user_id"]])>=.25
        if m["scenario"] and m["scenario"].startswith("shadow"):
            states=arr["bs_link"].reshape(f,n)[:,m["shadow_user_id"]]=="los"
            assert states.any() and not states.all()
            assert served[m["shadow_user_id"]],"shadow-edge user must be in the served group"
        # --- RIS reach
        ris_first=arr["ris_los"].reshape(f,n)[0].astype(bool)
        reach=arr["ris_los"].reshape(f,n).mean(axis=0)>=.5
        assert int(ris_first.sum())==m["ris_los_users_first_frame"]
        assert int(ris_first[served].sum())==m["served_ris_los_first_frame"]
        assert int(reach.sum())==m["ris_reachable_users"] and int(reach[served].sum())==m["served_ris_reachable"]
        kind=m["kind"]
        ris[kind].update(users=n,ris_los=int(ris_first.sum()),served=int(served.sum()),
                         served_ris_los=int(ris_first[served].sum()),reachable=int(reach.sum()),
                         served_reachable=int(reach[served].sum()),cases=1,
                         cases_with_reachable=int(reach.any()),
                         cases_all_served_ris_los=int(ris_first[served].all()))
        per_case.append({"id":m["id"],"map":map_id,"kind":kind,"bs_ris_distance_m":dist,
                         "users":n,"ris_los_users":int(ris_first.sum()),"served":int(served.sum()),
                         "served_ris_los":int(ris_first[served].sum())})
        n_rows+=n*f
        n_persons+=n
        counts[kind].update(arr["bs_link"][:n])
        rejections[map_id].update(m["rejections"])
        if kind=="natural":
            initial["crowd_size"].append(n)
            initial["group_size"].extend(m["group_sizes"])
            initial["bs_ris_distance_m"].append(dist)
            for name in ("bs_distance_m","ris_distance_m","height_agl_m","speed_mps","slope_deg"):
                initial[name].extend(arr[name][:n].tolist())
    nat=ris["natural"]
    served_share_nat=nat["served_ris_los"]/max(1,nat["served"])
    coverage_pass=served_share_nat>=NATURAL_SERVED_RIS_LOS_MIN
    for f in ("distributions.png","link-classes.png"):
        assert (root/"figures"/f).stat().st_size>0
    assert 3<=len(render["movement_strips"])<=5
    assert any("sloped" in row[1] for row in render["movement_strips"])
    assert len({row[1] for row in render["movement_strips"]})==len(render["movement_strips"])
    for row in render["movement_strips"]:
        assert (root/"figures"/row[0]).stat().st_size>0
    index=(root/"index.md").read_text()
    for m in rows:
        assert f"cases/{m['id']}/README.md" in index
    # --- bulk draw
    d=pq.read_table(root/"draws.parquet")
    assert len(d)==bulk["user_observations"]
    assert not set(d.column_names)&{"x","y","z","case_id","frame","user_id"}
    assert d["crowd_size"].null_count==len(d)-5000
    hc=np.asarray(d["height_class"].to_pylist())
    agl=d["height_agl_m"].to_numpy(zero_copy_only=False)
    bs_slope=d["slope_deg"].to_numpy(zero_copy_only=False)
    assert (agl[hc=="elevated"]>=4.-1e-8).all()
    assert np.isfinite(bs_slope[hc=="pedestrian"]).all() and (bs_slope[hc=="pedestrian"]<=SLOPE_CAP_DEG+1e-9).all()
    assert np.isnan(bs_slope[hc!="pedestrian"]).all() and d["slope_deg"].null_count==int((hc!="pedestrian").sum())
    bd=np.asarray([x for x in d["bs_ris_distance_m"].to_pylist() if x is not None])
    assert len(bd)==5000 and (bd>=SITE_RANGE_M[0]).all() and (bd<=SITE_RANGE_M[1]).all()
    assert set(d["bs_link"].to_pylist())<=LINKS
    bris=np.asarray(d["ris_los"].to_pylist(),dtype=bool)
    bserved=np.asarray(d["served"].to_pylist(),dtype=bool)
    bulk_ris={"users":len(d),"ris_los_users":int(bris.sum()),"served_users":int(bserved.sum()),
              "served_ris_los":int((bris&bserved).sum())}
    bulk_stats={}
    for c in ("crowd_size","group_size","bs_ris_distance_m","bs_distance_m","ris_distance_m","height_agl_m","speed_mps","slope_deg"):
        values=[x for x in d[c].to_pylist() if x is not None]
        bulk_stats[c]=quantiles(values)
    def share(a,b): return float(a/b) if b else None
    ris_report={}
    for kind,c in ris.items():
        ris_report[kind]={"cases":c["cases"],"users":c["users"],
                          "users_ris_los_share":share(c["ris_los"],c["users"]),
                          "served":c["served"],"served_ris_los_share":share(c["served_ris_los"],c["served"]),
                          "ris_reachable_users_share":share(c["reachable"],c["users"]),
                          "served_ris_reachable_share":share(c["served_reachable"],c["served"]),
                          "cases_with_a_reachable_user":c["cases_with_reachable"],
                          "cases_all_served_users_ris_los":c["cases_all_served_ris_los"]}
    ris_report["bulk_natural"]={**bulk_ris,"users_ris_los_share":share(bulk_ris["ris_los_users"],bulk_ris["users"]),
                                "served_ris_los_share":share(bulk_ris["served_ris_los"],bulk_ris["served_users"])}
    slope_report={}
    for q in ("pedestrian","elevated","drone"):
        vals=np.concatenate(slope_by_class[q]) if slope_by_class[q] else np.empty(0)
        slope_report[q]={"rows":int(len(vals)),"null_rows":int(slope_null_by_class[q]),
                         "quantiles_0_5_50_90_95_99_100":quantiles(vals)}
        bulk_vals=bs_slope[hc==q]
        slope_report[q]["bulk_users"]=int(len(bulk_vals))
        slope_report[q]["bulk_null"]=int(np.isnan(bulk_vals).sum())
        slope_report[q]["bulk_quantiles"]=quantiles(bulk_vals)
    pr_total=Counter()
    for c in rejections.values(): pr_total.update(c)
    bulk_total=Counter()
    for mp in bulk["maps"].values(): bulk_total.update(mp["rejections"])
    result={"status":"PASS" if coverage_pass else "FAIL","preview_cases":50,"natural_cases":42,"challenge_cases":8,
            "bulk_cases":5000,"labeled_rows":n_rows,"preview_people":n_persons,
            "declared":{"walk_slope_cap_deg":SLOPE_CAP_DEG,"bs_ris_distance_range_m":list(SITE_RANGE_M)},
            "natural_preview_ris_los_gate":{"observed":served_share_nat,"floor":NATURAL_SERVED_RIS_LOS_MIN,
                                            "pass":coverage_pass},
            "max_ground_readback_residual_m":max_ground_residual,
            "max_slope_readback_residual_deg":max_slope_residual,
            "site_ground_methods":{f"{a}:{b}":v for (a,b),v in site_methods.items()},
            "ris_los_shares":ris_report,"slope_by_height_class":slope_report,
            "bs_ris_distance_quantiles":{"preview_natural":quantiles(initial["bs_ris_distance_m"]),
                                         "preview_all":quantiles([c["bs_ris_distance_m"] for c in per_case]),
                                         "bulk_natural":quantiles(bd)},
            "per_case":per_case,"rendered_cases":render["rendered"],
            "movement_strips":render["movement_strips"],"preview_link_counts_at_first_frame":{k:dict(v) for k,v in counts.items()},
            "preview_rejections_by_map":{k:dict(v) for k,v in rejections.items()},
            "preview_rejections_total":dict(pr_total),"bulk_rejections_total":dict(bulk_total),
            "bulk_rejections_by_map":{k:v["rejections"] for k,v in bulk["maps"].items()},
            "geometry_rejections_by_map":{k:v["geometry_rejections"] for k,v in gate["maps"].items()},
            "street_prune_by_map":{k:v["street_prune"] for k,v in gate["maps"].items()},
            "site_stats_by_map":{k:v["site_stats"] for k,v in gate["maps"].items()},
            "quantiles_percent":[0,5,50,90,95,99,100],"natural_preview_quantiles":{k:quantiles(v) for k,v in initial.items()},
            "natural_bulk_quantiles":bulk_stats,"bulk_height_counts":dict(Counter(d["height_class"].to_pylist())),
            "bulk_link_counts":dict(Counter(d["bs_link"].to_pylist())),
            "artifact_sha256":{"cases":sha(root/"case_manifest.json"),"draws":sha(root/"draws.parquet"),
                               "render":sha(root/"render_manifest.json")}}
    (root/"validation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("PREVIEW_VALIDATION",json.dumps({k:v for k,v in result.items() if k!="per_case"}),flush=True)
    if not coverage_pass:
        raise SystemExit(f"served RIS-LOS share {served_share_nat:.4f} below declared floor {NATURAL_SERVED_RIS_LOS_MIN}")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    validate(p.parse_args().root)
