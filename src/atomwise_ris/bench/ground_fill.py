"""Separate footprint terrain overlay; native street sampling is never changed."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError, cKDTree

NO_DATA, MEASURED, FILLED = 0, 1, 2
SOURCE = {NO_DATA:"no_data",MEASURED:"measured",FILLED:"interpolated_under_footprint"}
FALLBACK = "lod2_native_ground_fallback"
HOLDOUT = {"min_count":2000,"min_coverage":.4,"median_abs_max_m":.75,"p95_abs_max_m":2.}
RISK = ("all","large","courtyard","steep","canal_side","underpass")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def protective_shapes(city, osm):
    """Conservative support mask for mapped water and grade separation."""
    import shapely
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize
    from pyproj import Transformer

    root=ET.parse(osm).getroot()
    tr=Transformer.from_crs(4326,city.crs,always_xy=True)
    nodes={n.get("id"):tr.transform(float(n.get("lon")),float(n.get("lat"))) for n in root.findall("node")}
    water=[city.water] if city.water is not None and not city.water.is_empty else []
    corridors=[]
    ways={}

    def tags(e):
        return {t.get("k"):t.get("v") for t in e.findall("tag")}

    def iswater(t):
        return t.get("natural")=="water" or t.get("landuse") in {"reservoir","basin"} or t.get("waterway") in {
            "river","canal","stream","ditch","drain","riverbank"}

    for w in root.findall("way"):
        coord=[nodes[r] for n in w.findall("nd") if (r:=n.get("ref")) in nodes]
        if len(coord)<2:
            continue
        line=LineString(coord)
        ways[w.get("id")]=line
        t=tags(w)
        area=len(coord)>=4 and coord[0]==coord[-1]
        if iswater(t):
            water.append(shapely.make_valid(Polygon(coord)) if area else line.buffer(8.))
        elevated=any(t.get(k) not in {None,"no","false","0"} for k in ("bridge","tunnel","covered"))
        buried=t.get("location")=="underground" or t.get("underground")=="yes"
        if elevated or buried:
            try:
                width=float((t.get("width") or "16").strip().removesuffix("m"))
                if not np.isfinite(width) or width<=0:
                    width=16.
            except ValueError:
                width=16.
            corridors.append(shapely.make_valid(Polygon(coord)) if area else line.buffer(width/2+2.))
    for r in root.findall("relation"):
        if not iswater(tags(r)):
            continue
        parts=[ways[x.get("ref")] for x in r.findall("member")
               if x.get("type")=="way" and x.get("role") in {"","outer"} and x.get("ref") in ways]
        water.extend(polygonize(parts))
    w=shapely.union_all(water)
    return w,shapely.union_all(corridors)


def _raster(shape, geoms, tr, dtype="uint8"):
    from rasterio.features import rasterize
    return rasterize(geoms,out_shape=shape,transform=tr,fill=0,dtype=dtype) if geoms else np.zeros(shape,dtype=dtype)


def errors(values):
    v=np.asarray(values,dtype=float)
    if not len(v):
        return {"n":0}
    a=np.abs(v)
    return {"n":len(v),"bias_m":float(v.mean()),"rmse_m":float(np.sqrt(np.mean(v*v))),
            "median_abs_m":float(np.median(a)),"p90_abs_m":float(np.percentile(a,90)),
            "p95_abs_m":float(np.percentile(a,95)),"max_abs_m":float(a.max())}


def interpolate_xy(source, values, target):
    """Strict convex-hull interpolation plus a bound on distance to support."""
    if len(source)<8 or not len(target):
        return np.full(len(target),np.nan)
    origin=source.mean(axis=0)
    try:
        pred=LinearNDInterpolator(source-origin,values,fill_value=np.nan)(target-origin)
    except QhullError:
        return np.full(len(target),np.nan)
    dist=cKDTree(source).query(target)[0]
    pred[dist>12.]=np.nan
    return pred


class GroundLayer:
    def __init__(self,x,y,z,source,owner,protected,buildings):
        self.x,self.y,self.z,self.source=x,y,z,source
        self.owner,self.protected=owner,protected
        self.buildings=buildings
        self.by_id={b["id"]:i+1 for i,b in enumerate(buildings)}
        self.cache={}

    @staticmethod
    def read(npz,meta):
        with np.load(npz,allow_pickle=False) as a:
            return GroundLayer(*(a[k] for k in ("x","y","z","ground_source","owner","protected")),
                               json.loads(Path(meta).read_text())["buildings"])

    def write(self,out,map_id):
        npz,meta=out/f"{map_id}-footprint-ground.npz",out/f"{map_id}-owners.json"
        np.savez_compressed(npz,x=self.x,y=self.y,z=self.z,ground_source=self.source,
                            owner=self.owner,protected=self.protected)
        meta.write_text(json.dumps({"map_id":map_id,"buildings":self.buildings,"source_codes":SOURCE},
                                   allow_nan=False,indent=2)+"\n")
        return {"raster":npz.name,"raster_sha256":sha(npz),"owners":meta.name,"owners_sha256":sha(meta)}

    def _tree(self,i):
        if i not in self.cache:
            # Limit each support lookup to its footprint's raster bbox.
            b=self.buildings[i-1]
            r0,r1,c0,c1=b["window"]
            r,c=np.nonzero((self.owner[r0:r1,c0:c1]==i)&(self.source[r0:r1,c0:c1]>0)
                           &~self.protected[r0:r1,c0:c1])
            r+=r0; c+=c0
            self.cache[i]=(cKDTree(np.c_[self.x[c],self.y[r]]) if len(r) else None,r,c)
        return self.cache[i]

    def lookup(self,ids,xy):
        xy=np.asarray(xy,dtype=float).reshape(-1,2)
        z=np.full(len(xy),np.nan)
        methods=np.full(len(xy),"unavailable",dtype="<U40")
        cells=np.full((len(xy),2),-1,dtype=np.int32)
        c=np.floor((xy[:,0]-self.x[0])/(self.x[1]-self.x[0])+.5).astype(int)
        r=np.floor((xy[:,1]-self.y[0])/(self.y[1]-self.y[0])+.5).astype(int)
        for j,(ident,p) in enumerate(zip(ids,xy,strict=True)):
            i=self.by_id.get(str(ident))
            if i is None or not (0<=r[j]<len(self.y) and 0<=c[j]<len(self.x)) or self.protected[r[j],c[j]]:
                continue
            tree,rr,cc=self._tree(i)
            if tree is not None:
                d,k=tree.query(p)
                if d<=1.5:
                    # Break equidistant raster-center ties in row-major order.
                    nearby=tree.query_ball_point(p,d+1e-10)
                    if len(nearby)>1:
                        idx=np.asarray(nearby,dtype=int)
                        ds=np.linalg.norm(tree.data[idx]-p,axis=1)
                        k=int(idx[np.abs(ds-d)<=1e-10].min())
                    jj,ii=rr[k],cc[k]
                    z[j]=self.z[jj,ii]
                    methods[j]=SOURCE[int(self.source[jj,ii])]
                    cells[j]=[jj,ii]
                    continue
            ref=self.buildings[i-1]["native_ground_m"]
            if ref is not None:
                z[j]=ref
                methods[j]=FALLBACK
        return z,methods,cells


def prepare_map(city, osm, allow_fill=True):
    import shapely

    tr=city.terrain_transform
    raw=city._dem
    h,w=raw.shape
    x=tr.c+(np.arange(w)+.5)*tr.a
    y=tr.f+(np.arange(h)+.5)*tr.e
    xmin,ymin,xmax,ymax=city.bounds
    inside=(y[:,None]>=ymin)&(y[:,None]<ymax)&(x[None,:]>=xmin)&(x[None,:]<xmax)
    buildings=sorted(city.buildings,key=lambda b:str(b.id))
    polygons=[(b.footprint,i+1) for i,b in enumerate(buildings) if not b.footprint.is_empty]
    owner=_raster(raw.shape,polygons,tr,"int32")
    water,transport=protective_shapes(city,osm)
    protected=_raster(raw.shape,[(g,1) for g in [water.buffer(1.),transport] if not g.is_empty],tr).astype(bool)
    allowed=inside&~protected
    src=np.where(np.isfinite(raw),MEASURED,NO_DATA).astype(np.uint8)
    z=raw.copy()
    windows=[]; counts=defaultdict(int)
    buckets={k:defaultdict(list) for k in RISK}
    subset_eligible=defaultdict(int)
    rows=[]
    for i,b in enumerate(buildings,1):
        p=b.footprint
        native=float(b.ground_m) if b.ground_m is not None and np.isfinite(b.ground_m) else None
        if p.is_empty:
            windows.append({"id":str(b.id),"window":[0,0,0,0],"native_ground_m":native})
            continue
        b0,b1,b2,b3=p.bounds
        rows_y=np.flatnonzero((y>=b1-10.)&(y<=b3+10.))
        cols_x=np.flatnonzero((x>=b0-10.)&(x<=b2+10.))
        if not len(rows_y) or not len(cols_x):
            windows.append({"id":str(b.id),"window":[0,0,0,0],"native_ground_m":native})
            continue
        r0,r1,c0,c1=int(rows_y[0]),int(rows_y[-1]+1),int(cols_x[0]),int(cols_x[-1]+1)
        windows.append({"id":str(b.id),"window":[r0,r1,c0,c1],"native_ground_m":native})
        sl=np.s_[r0:r1,c0:c1]
        mask=(owner[sl]==i)&inside[sl]
        target=mask&allowed[sl]
        vv=raw[sl]
        observed=target&np.isfinite(vv)
        missing=target&~np.isfinite(vv)
        counts["owned_cells"]+=int(mask.sum())
        counts["protected_owned_cells"]+=int((mask&protected[sl]).sum())
        counts["original_missing_under_footprint"]+=int((mask&~np.isfinite(vv)).sum())
        row={"building_id":str(b.id),"area_m2":float(p.area),"native_ground_m":native,
             "measured_cells":int(observed.sum()),"missing_eligible":int(missing.sum()),
             "protected_cells":int((mask&protected[sl]).sum())}
        shapes=list(p.geoms) if p.geom_type=="MultiPolygon" else [p]
        risks=["all"]
        if p.area>=500.: risks.append("large")
        if any(len(getattr(q,"interiors",[])) for q in shapes): risks.append("courtyard")
        if not water.is_empty and p.distance(water)<=15.: risks.append("canal_side")
        if not transport.is_empty and p.intersects(transport): risks.append("underpass")
        limited=p.area>2500. or max(b2-b0,b3-b1)>80.
        if limited: counts["footprints_over_cap"]+=1
        xm,ym=np.meshgrid(x[c0:c1],y[r0:r1])
        seeds=(owner[sl]==0)&allowed[sl]&np.isfinite(vv)
        if seeds.any():
            sr,sc=np.nonzero(seeds)
            near=shapely.distance(shapely.points(xm[sr,sc],ym[sr,sc]),p)<=10.
            sr,sc=sr[near],sc[near]
        else:
            sr,sc=np.array([],dtype=int),np.array([],dtype=int)
        if len(sr)>=8 and np.ptp(np.percentile(vv[sr,sc],[10,90]))/max(1.,np.hypot(b2-b0,b3-b1))>=.0875:
            risks.append("steep")
        row["risk"]=risks[1:]
        row["over_cap"]=bool(limited)
        row["seed_count"]=len(sr)
        if native is not None:
            vals=vv[observed]-native
            row["native_dtm_minus_lod2"]=errors(vals)
            for k in risks: buckets[k]["native_dtm_minus_lod2"].extend(vals.tolist())
        if limited or len(sr)<8:
            counts["footprints_without_estimate"]+=1
            rows.append(row)
            continue
        if city.map_id=="munich":
            for k in risks: subset_eligible[k]+=int(observed.sum())
        evaluate=target if city.map_id=="munich" else missing
        rr,cc=np.nonzero(evaluate)
        pred=interpolate_xy(np.c_[xm[sr,sc],ym[sr,sc]],vv[sr,sc],np.c_[xm[rr,cc],ym[rr,cc]])
        good=np.isfinite(pred)
        held=good&np.isfinite(vv[rr,cc])
        if city.map_id=="munich":
            delta=pred[held]-vv[rr[held],cc[held]]
            row["munich_masked_prediction_minus_provider"]=errors(delta)
            for k in risks: buckets[k]["munich_masked_prediction_minus_provider"].extend(delta.tolist())
        new=good&~np.isfinite(vv[rr,cc])
        row["raw_predicted_missing"]=int(new.sum())
        if native is not None:
            # Munich's masked predictions are also independently compared to its model ground.
            comp=good if city.map_id=="munich" else new
            delta=pred[comp]-native
            row["raw_prediction_minus_lod2"]=errors(delta)
            for k in risks: buckets[k]["raw_prediction_minus_lod2"].extend(delta.tolist())
        accepted=new&(np.abs(pred-native)<=2.5) if native is not None else np.zeros(len(pred),dtype=bool)
        row["accepted_missing"]=int(accepted.sum()) if allow_fill else 0
        counts["raw_predicted_missing"]+=int(new.sum())
        counts["agreement_guard_rejected"]+=int((new&~accepted).sum())
        if allow_fill and accepted.any():
            r,c=rr[accepted]+r0,cc[accepted]+c0
            z[r,c]=pred[accepted]
            src[r,c]=FILLED
            delta=pred[accepted]-native
            row["used_fill_minus_lod2"]=errors(delta)
            for k in risks: buckets[k]["used_fill_minus_lod2"].extend(delta.tolist())
        rows.append(row)
    fill=src==FILLED
    original=np.isfinite(raw)
    assert np.array_equal(z[original],raw[original])
    assert not np.any(fill&(original|protected|~inside|(owner==0)))
    assert np.array_equal(np.isfinite(z),src>0)
    counts["filled_cells"]=int(fill.sum())
    counts["measured_owned_cells"]=int((original&(owner>0)&inside).sum())
    stats={}
    for k in RISK:
        v={m:errors(buckets[k][m]) for m in ("native_dtm_minus_lod2","raw_prediction_minus_lod2",
               "used_fill_minus_lod2","munich_masked_prediction_minus_provider")}
        n=v["munich_masked_prediction_minus_provider"]["n"]
        possible=subset_eligible[k]
        v["masked_eligible_cells"]=possible
        v["masked_prediction_coverage"]=n/possible if possible else None
        stats[k]=v
    hold=stats["all"]["munich_masked_prediction_minus_provider"]
    passed=(hold["n"]>=HOLDOUT["min_count"] and stats["all"]["masked_prediction_coverage"]>=HOLDOUT["min_coverage"]
            and hold["median_abs_m"]<=HOLDOUT["median_abs_max_m"] and hold["p95_abs_m"]<=HOLDOUT["p95_abs_max_m"])
    return GroundLayer(x,y,z,src,owner,protected,windows),{"counts":dict(counts),"subsets":stats,
        "buildings":rows,"holdout_pass":bool(passed) if city.map_id=="munich" else None}


def prepare(cities,cache,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    order=("munich","rotterdam","maastricht")
    if set(cities)!=set(order):
        raise ValueError("ground fill requires exactly the three development maps")
    report={"thresholds":HOLDOUT,"maps":{}}
    approved=None
    for name in order:
        city=cities[name]
        p=Path(cache)/f"{name}.osm"
        if sha(p)!=city.source_hashes[p.name]:
            raise ValueError("protective OSM geometry hash mismatch")
        layer,stats=prepare_map(city,p,allow_fill=approved is not False)
        if name=="munich":
            approved=stats["holdout_pass"]
            if not approved and (layer.source==FILLED).any():
                layer.z[layer.source==FILLED]=np.nan
                layer.source[layer.source==FILLED]=NO_DATA
                stats["counts"]["filled_cells"]=0
                for s in stats["subsets"].values(): s["used_fill_minus_lod2"]={"n":0}
                for b in stats["buildings"]:
                    b["accepted_missing"]=0
                    b.pop("used_fill_minus_lod2",None)
        city.ground_fill=layer
        stats["artifact"]=layer.write(out,name)
        report["maps"][name]=stats
    report["interpolated_use_enabled"]=approved
    (out/"audit.json").write_text(json.dumps(report,allow_nan=False,indent=2)+"\n")
    return report
