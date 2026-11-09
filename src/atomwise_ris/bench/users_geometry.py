"""Audited, geometry-only city support for the bench user preview generator.

All XY coordinates are in the area's projected metric CRS. This module does not
generate RF channels. Missing terrain stays missing and held-out maps are
refused by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEV_IDS = ("munich", "rotterdam", "maastricht")
AREAS = {
    "munich": {"crs": "EPSG:25832", "bounds": (691383., 5334560., 691823., 5335000.), "kind": "munich"},
    "rotterdam": {"crs": "EPSG:28992", "bounds": (92286., 436839., 92726., 437279.), "kind": "netherlands"},
    "maastricht": {"crs": "EPSG:28992", "bounds": (175967., 316952., 176407., 317392.), "kind": "netherlands"},
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Building:
    id: str
    footprint: object
    height_m: float
    ground_m: float | None
    height_source: str
    roof_polygons: list = field(default_factory=list)
    facade_xyz: list = field(default_factory=list)
    triangles_xyz: list = field(default_factory=list)
    wall_triangles: list = field(default_factory=list)

    @property
    def roof_points(self):
        return [np.asarray(t, float).mean(axis=0).tolist() for t in self.roof_polygons]


class CityMap:
    """Map geometry and terrain in projected metres; no channel/propagation API."""
    def __init__(self, map_id, bounds, crs, terrain, buildings, street_segments,
                 hotspots_xy, hotspot_weights, water, source_hashes):
        from shapely.strtree import STRtree
        self.map_id, self.bounds, self.crs = map_id, tuple(bounds), crs
        self.terrain = terrain  # rasterio dataset, kept open for sampling
        self.terrain_transform = terrain.transform
        self.terrain_width, self.terrain_height = terrain.width, terrain.height
        self._dem_cache = terrain.read(1, masked=True).filled(np.nan).astype(float)
        self._valid_cache = np.isfinite(self._dem_cache)
        self.buildings = buildings
        self.building_by_id = {b.id: b for b in buildings}
        self.height_source_by_id = {b.id: b.height_source for b in buildings}
        self._footprints = [b.footprint for b in buildings]
        self._building_tree = STRtree(self._footprints) if self._footprints else None
        if self._footprints:
            from shapely.ops import unary_union
            self._occupied = unary_union(self._footprints)
        else:
            self._occupied = None
        self.water = water
        self.street_segments = np.asarray(street_segments, dtype=float).reshape(-1, 2, 2)
        self.hotspots_xy = np.asarray(hotspots_xy, dtype=float).reshape(-1, 2)
        self.hotspot_weights = np.asarray(hotspot_weights, dtype=float).reshape(-1)
        self.source_hashes = source_hashes
        self.facade_xyz = np.asarray([p for b in buildings for p in b.facade_xyz], dtype=float).reshape(-1, 3)
        self.facade_owner_ids = np.asarray([b.id for b in buildings for _ in b.facade_xyz], dtype=object)
        self.roof_xyz = np.asarray([p for b in buildings for p in b.roof_points], dtype=float).reshape(-1, 3) if buildings else np.empty((0, 3))
        self.roof_owner_ids = np.asarray([b.id for b in buildings for _ in b.roof_points], dtype=object)
        self.wall_triangles = np.asarray([t for b in buildings for t in b.wall_triangles], dtype=float).reshape(-1, 3, 3) if buildings else np.empty((0, 3, 3))
        self.sites = []
        self._ray_intersector = None
        self._ray_face_owner = np.asarray([], dtype=object)
        self._build_ray_scene()
        self._prepare_site_pairs()

    def _build_ray_scene(self):
        """Build one native-mesh Embree scene; partial shells get tagged slabs."""
        triangles=[]; owners=[]
        for b in self.buildings:
            if b.height_source.find("slab fallback") >= 0 or not b.wall_triangles:
                tris=_footprint_slab_triangles(b)
            else:
                tris=b.triangles_xyz
            if tris:
                triangles.extend(tris); owners.extend([b.id]*len(tris))
        if not triangles: return
        try:
            import trimesh
            from trimesh.ray.ray_pyembree import RayMeshIntersector
            tri=np.asarray(triangles,dtype=float)
            mesh=trimesh.Trimesh(vertices=tri.reshape(-1,3),
                faces=np.arange(len(tri)*3,dtype=np.int64).reshape(-1,3),process=False)
            self._ray_intersector=RayMeshIntersector(mesh)
            self._ray_face_owner=np.asarray(owners,dtype=object)
        except (ImportError, ModuleNotFoundError):
            # The exact CPU triangle fallback remains available, but is intended
            # only for small previews; the preparation environment installs
            # trimesh+embreex outside the repository.
            self._ray_triangles=np.asarray(triangles,dtype=float)
            self._ray_face_owner=np.asarray(owners,dtype=object)

    def _pixel_values(self, xy):
        """Bilinear elevation and local slope under ONE validity rule.

        v1 ground rule: a position is valid only if all 16 DTM cells of its
        4x4 stencil (the 2x2 bilinear patch plus the ring needed for the
        central-difference slope) are valid. Elevation is the bilinear
        interpolation of the inner 2x2 cells; slope is the bilinear
        interpolation of the four central-difference gradients at those cell
        centres (each uses its two neighbours across the ring). No-data cells
        are never filled, extrapolated or interpolated across.
        """
        xy = np.asarray(xy, float).reshape(-1, 2)
        inv = ~self.terrain.transform
        cols, rows = inv @ (xy[:, 0], xy[:, 1])
        # Raster coordinates point to cell corners; use cell-centre coordinates.
        cols, rows = np.asarray(cols) - .5, np.asarray(rows) - .5
        c0, r0 = np.floor(cols).astype(int), np.floor(rows).astype(int)
        dx, dy = cols-c0, rows-r0
        h, w = self.terrain_height, self.terrain_width
        arr, mask = self._dem, self._valid
        inside = (c0 >= 1) & (r0 >= 1) & (c0+2 < w) & (r0+2 < h)
        z = np.full(len(xy), np.nan)
        slope = np.full(len(xy), np.nan)
        valid = np.zeros(len(xy), dtype=bool)
        ix = np.flatnonzero(inside)
        if not len(ix):
            return z, slope, valid
        r, c = r0[ix], c0[ix]
        good = np.ones(len(ix), dtype=bool)
        for i in (-1, 0, 1, 2):
            for j in (-1, 0, 1, 2):
                good &= mask[r+i, c+j]
        ii, r, c = ix[good], r[good], c[good]
        if not len(ii):
            return z, slope, valid
        fx, fy = dx[ii], dy[ii]          # column (east) and row (south) fractions
        a, b = arr[r, c], arr[r, c+1]
        d, e = arr[r+1, c], arr[r+1, c+1]
        z[ii] = (1-fy)*((1-fx)*a+fx*b) + fy*((1-fx)*d+fx*e)
        xres, yres = abs(self.terrain_transform.a), abs(self.terrain_transform.e)

        def gradient(rr, cc):
            gx = (arr[rr, cc+1]-arr[rr, cc-1])/(2*xres)
            gy = (arr[rr-1, cc]-arr[rr+1, cc])/(2*yres)     # north positive
            return gx, gy

        g00, g01 = gradient(r, c), gradient(r, c+1)
        g10, g11 = gradient(r+1, c), gradient(r+1, c+1)
        gx = (1-fy)*((1-fx)*g00[0]+fx*g01[0]) + fy*((1-fx)*g10[0]+fx*g11[0])
        gy = (1-fy)*((1-fx)*g00[1]+fx*g01[1]) + fy*((1-fx)*g10[1]+fx*g11[1])
        slope[ii] = np.degrees(np.arctan(np.hypot(gx, gy)))
        valid[ii] = True
        return z, slope, valid

    @property
    def _dem(self):
        return self._dem_cache

    @property
    def _valid(self):
        return self._valid_cache

    def sample_ground(self, xy):
        """Return elevation metres, slope degrees and strict validity arrays."""
        return self._pixel_values(xy)

    FOOT_RING_OFFSETS_M = (3., 4.5, 6.)
    FOOT_RING_SPACING_M = 2.
    FOOT_MIN_SAMPLES = 3
    FOOT_NEAREST = 3

    def _foot_ring(self, building_id):
        """Valid street-level DTM samples on a ring just outside one footprint.

        Building-mounted supports (facade RIS, rooftop BS, rooftop users) sit
        where the bare-earth DTM is usually NO-DATA (AHN4 masks buildings). The
        height of such a support is therefore measured against genuine valid
        DTM samples at the building foot: points 3 m (else 4.5 m, else 6 m)
        outside the footprint, every one passing the same 4x4 stencil and
        outdoor rule as a user position. Nothing is filled or interpolated
        across no-data. Returns an (m,3) array, or None if fewer than
        ``FOOT_MIN_SAMPLES`` valid samples exist at every offset.
        """
        cache = self.__dict__.setdefault("_foot_cache", {})
        if building_id in cache:
            return cache[building_id]
        import shapely
        ring = None
        building = self.building_by_id.get(building_id)
        if building is not None and not building.footprint.is_empty:
            for offset in self.FOOT_RING_OFFSETS_M:
                line = shapely.segmentize(building.footprint.buffer(offset).boundary, self.FOOT_RING_SPACING_M)
                pts = shapely.get_coordinates(line)
                if not len(pts):
                    continue
                z, _, ok = self.sample_ground(pts)
                ok = ok & self.outdoor(pts)
                if ok.sum() >= self.FOOT_MIN_SAMPLES:
                    ring = np.column_stack([pts[ok], z[ok]])
                    break
        cache[building_id] = ring
        return ring

    def building_foot_samples(self, building_id, xy):
        """The ``FOOT_NEAREST`` valid foot-ring samples closest to ``xy`` ((k,3)) or None."""
        ring = self._foot_ring(building_id)
        if ring is None:
            return None
        d = np.linalg.norm(ring[:, :2]-np.asarray(xy, float)[:2], axis=1)
        return ring[np.argsort(d, kind="stable")[:self.FOOT_NEAREST]]

    def building_foot_ground(self, building_ids, xy):
        """Foot-ring ground reference (median of nearest valid samples); NaN if none."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = np.full(len(xy), np.nan)
        for i, (b, p) in enumerate(zip(building_ids, xy)):
            samples = self.building_foot_samples(b, p)
            if samples is not None:
                out[i] = float(np.median(samples[:, 2]))
        return out

    def outdoor(self, xy):
        """True only on valid ground outside modeled footprints and mapped water."""
        import shapely
        pts = np.asarray(xy, float).reshape(-1, 2)
        _, _, ok = self.sample_ground(pts)
        points=shapely.points(pts[:,0],pts[:,1])
        if self._occupied is not None: ok &= ~np.asarray(shapely.covers(self._occupied,points),bool)
        if self.water is not None: ok &= ~np.asarray(shapely.covers(self.water,points),bool)
        return ok

    def los_batch(self, tx_xyz, rx_xyz):
        """Exact 3-D triangle segment test; returns visibility and blocker IDs.

        Endpoints may be one xyz vector or matching (N,3) arrays. Buildings with
        unsupported/empty surfaces fail closed using their footprint and height
        slab as a conservative blocker. `None` means no blocker.
        """
        tx, rx = np.asarray(tx_xyz, float), np.asarray(rx_xyz, float)
        if tx.ndim == 1: tx = tx[None, :]
        if rx.ndim == 1: rx = rx[None, :]
        tx, rx = np.broadcast_arrays(tx, rx)
        if self._ray_intersector is not None:
            delta=rx-tx; length=np.linalg.norm(delta,axis=1)
            valid=length>1e-9; directions=np.zeros_like(delta)
            directions[valid]=delta[valid]/length[valid,None]
            # Facade/roof endpoints may lie on their own triangles. Move only
            # 2 mm into the finite segment to avoid a zero-distance self-hit;
            # retain the original segment end as a strict distance bound.
            eps=0.002
            loc,ray_id,tri_id=self._ray_intersector.intersects_location(
                tx[valid]+eps*directions[valid],directions[valid],multiple_hits=False)
            vis=np.ones(len(tx),dtype=bool); blocker=np.full(len(tx),None,dtype=object)
            if len(ray_id):
                original=np.flatnonzero(valid)[ray_id]
                dist=np.linalg.norm(loc-(tx[original]+eps*directions[original]),axis=1)
                hit=(dist>1e-4)&(dist<length[original]-2*eps)
                for rid,tid in zip(original[hit],tri_id[hit]):
                    vis[rid]=False; blocker[rid]=self._ray_face_owner[tid]
            return vis,blocker
        from shapely.geometry import LineString
        visible = np.ones(len(tx), dtype=bool)
        blocker = np.full(len(tx), None, dtype=object)
        if self._building_tree is None:
            return visible, blocker
        for i, (a, b) in enumerate(zip(tx, rx)):
            line = LineString([a[:2], b[:2]])
            for j in self._building_tree.query(line):
                building = self.buildings[int(j)]
                if not line.intersects(building.footprint):
                    continue
                hit = False
                tris = np.asarray(building.triangles_xyz, float)
                if tris.size:
                    hit = _segment_hits_triangles(a, b, tris)
                if not tris.size or hit is None or not building.wall_triangles:
                    # Conservative fallback when the provider surface could not
                    # be triangulated: any interior footprint passage through
                    # the model's vertical extent blocks.
                    zlo = building.ground_m if building.ground_m is not None else 0.
                    zhi = zlo + building.height_m
                    mid = (line.intersection(building.footprint).representative_point())
                    f = line.project(mid) / max(line.length, 1e-12)
                    z = a[2] + f*(b[2]-a[2])
                    hit = zlo <= z <= zhi
                if hit:
                    visible[i], blocker[i] = False, building.id
                    break
        return visible, blocker

    def _prepare_site_pairs(self):
        # Rooftop/mast anchor candidates: roof points have their provider absolute
        # z; mast sites use ground + 12 m at valid outdoor points. Every pair is
        # kept only if the 3-D geometry has clear LOS.
        if not len(self.street_segments): return
        mids = self.street_segments.mean(axis=1)
        z, _, ok = self.sample_ground(mids)
        mast = np.column_stack([mids[ok], z[ok]+12.0])
        mast=mast[::max(1,len(mast)//48)] if len(mast) else mast
        roofs = self.roof_xyz[::max(1, len(self.roof_xyz)//48)] if len(self.roof_xyz) else np.empty((0,3))
        roof_ids=self.roof_owner_ids[::max(1, len(self.roof_owner_ids)//48)] if len(self.roof_owner_ids) else np.empty(0,dtype=object)
        if len(roofs):
            rg,_,rv=self.sample_ground(roofs[:,:2])
            roof_mask=rv & (roofs[:,2]>rg+1.)
            roofs=roofs[roof_mask]; roof_ids=roof_ids[roof_mask]
        # Interleave supports so the preview site pool contains both masts and
        # rooftop anchors when both are supported by valid terrain.
        anchors=[]
        for idx in range(max(len(mast),len(roofs))):
            if idx<len(roofs): anchors.append((roofs[idx],"rooftop",str(roof_ids[idx])))
            if idx<len(mast): anchors.append((mast[idx],"mast",None))
        facades = self.facade_xyz
        facade_ids=self.facade_owner_ids
        if len(facades):
            fg,_,fvalid=self.sample_ground(facades[:,:2])
            facades=facades[fvalid & (facades[:,2] > fg + 1.0)]
            facade_ids=facade_ids[fvalid & (self.facade_xyz[:,2] > fg+1.0)]
            # Spread the RIS shortlist across the entire development area rather
            # than repeatedly selecting the first facade in source order.
            bins={}
            for p,pid in zip(facades,facade_ids):
                bins.setdefault((int(p[0]//20),int(p[1]//20)),(p,pid))
            values=list(bins.values())
            facades=np.asarray([x[0] for x in values],float).reshape(-1,3)
            facade_ids=np.asarray([x[1] for x in values],dtype=object)
        else: facades=np.empty((0,3))
        if not len(facades) or not len(anchors): return
        count = 0; ris_keys=[]
        anchor_keys = set()
        for bi, (bs,bs_support,bs_building) in enumerate(anchors):
            key=tuple(np.round(bs,2))
            if key in anchor_keys: continue
            anchor_keys.add(key)
            accepted_for_anchor=0
            for ri, ris in enumerate(facades):
                if np.linalg.norm(bs[:2]-ris[:2]) < 12: continue
                if any(np.linalg.norm(ris[:2]-old) < 20 for old in ris_keys): continue
                vis, _ = self.los_batch(bs, ris)
                if vis[0]:
                    ground, _, valid = self.sample_ground([bs[:2], ris[:2]])
                    if not np.all(valid): continue
                    self.sites.append({"site_id": f"{self.map_id}-site-{count:03d}",
                        "bs_xyz": bs.tolist(), "ris_xyz": ris.tolist(),
                        "bs_agl_m": float(bs[2]-ground[0]) if valid[0] else None,
                        "ris_agl_m": float(ris[2]-ground[1]) if valid[1] else None,
                        "bs_support": bs_support,
                        "bs_building_id": bs_building,
                        "ris_building_id": str(facade_ids[ri]),
                        "ris_support": "facade",
                        "bs_ris_los": True})
                    ris_keys.append(ris[:2].copy())
                    count += 1; accepted_for_anchor += 1
                    if accepted_for_anchor >= 2: break
                    if count >= 24: return

    def slope_summary(self):
        """Length-sampled street slope coverage; invalid samples are counted."""
        pts = []
        for seg in self.street_segments:
            n = max(1, int(np.ceil(np.linalg.norm(seg[1]-seg[0]))))
            pts.extend(seg[0] + (seg[1]-seg[0])*t for t in np.linspace(0,1,n+1))
        if not pts: return {"street_sample_count": 0, "valid_count": 0, "invalid_count": 0, "slope_deg": {}}
        _, slope, valid = self.sample_ground(pts)
        v = slope[valid]
        return {"street_sample_count": len(pts), "valid_count": int(valid.sum()),
                "invalid_count": int((~valid).sum()), "valid_fraction": float(valid.mean()),
                "slope_deg": ({"min": float(v.min()), "p50": float(np.percentile(v,50)),
                    "p90": float(np.percentile(v,90)), "p95": float(np.percentile(v,95)),
                    "max": float(v.max()),
                    "count_gt_10": int((v>10).sum()), "count_gt_15": int((v>15).sum()), "count_gt_20": int((v>20).sum()),
                    "count_gt_30": int((v>30).sum())} if len(v) else {})}

    def export_geometry(self, out_dir):
        """Export compact renderer geometry and the native-resolution terrain."""
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        x = self.terrain_transform.c + (np.arange(self.terrain_width)+.5)*self.terrain_transform.a
        y = self.terrain_transform.f + (np.arange(self.terrain_height)+.5)*self.terrain_transform.e
        z = self._dem.astype(np.float32)
        valid = self._valid
        terrain_name = f"{self.map_id}-terrain.npz"
        np.savez_compressed(out_dir/terrain_name, x=x.astype(np.float32), y=y.astype(np.float32), z=z, valid=valid)
        records=[]
        for b in self.buildings:
            if b.footprint.is_empty: continue
            poly = max(getattr(b.footprint, "geoms", [b.footprint]), key=lambda g:g.area) if b.footprint.geom_type=="MultiPolygon" else b.footprint
            records.append({"id": b.id, "polygon_xy": np.asarray(poly.exterior.coords).tolist(),
                            "height_m": b.height_m, "ground_m": b.ground_m,
                            "height_source": b.height_source})
        meta={"map_id":self.map_id,"crs":self.crs,
              "bounds":[self.bounds[0],self.bounds[2],self.bounds[1],self.bounds[3]],
              "buildings":records,"terrain":{"file":terrain_name},
              "sites":self.sites,"street_slope":self.slope_summary(),
              "source_hashes":self.source_hashes}
        (out_dir/f"{self.map_id}.json").write_text(json.dumps(meta,indent=2)+"\n")
        return meta


def _segment_hits_triangles(a, b, triangles):
    """Vectorized Moller-Trumbore finite segment/triangle intersection."""
    v0, v1, v2 = triangles[:,0], triangles[:,1], triangles[:,2]
    d = b-a; e1=v1-v0; e2=v2-v0
    p=np.cross(np.broadcast_to(d,e2.shape),e2); det=np.einsum("ij,ij->i",e1,p)
    good=np.abs(det)>1e-10
    inv=np.zeros_like(det); inv[good]=1/det[good]
    tvec=a-v0; u=np.einsum("ij,ij->i",tvec,p)*inv
    q=np.cross(tvec,e1); v=np.einsum("j,ij->i",d,q)*inv
    t=np.einsum("ij,ij->i",e2,q)*inv
    return bool(np.any(good & (u>=-1e-9) & (v>=-1e-9) & (u+v<=1+1e-9) & (t>1e-8) & (t<1-1e-8)))


def _footprint_slab_triangles(building):
    """Conservative closed extrusion for buildings with incomplete LoD2 shells."""
    from shapely import constrained_delaunay_triangles
    polys=list(building.footprint.geoms) if building.footprint.geom_type=="MultiPolygon" else [building.footprint]
    z0=float(building.ground_m or 0.); z1=z0+float(building.height_m); out=[]
    for poly in polys:
        for t in constrained_delaunay_triangles(poly).geoms:
            xy=np.asarray(t.exterior.coords[:3],float)
            low=np.column_stack([xy,np.full(3,z0)]); high=np.column_stack([xy,np.full(3,z1)])
            out.extend([low,high[[0,2,1]]])
        for ring in [poly.exterior,*poly.interiors]:
            pts=np.asarray(ring.coords,float)
            for a,b in zip(pts[:-1],pts[1:]):
                p0=np.array([a[0],a[1],z0]); p1=np.array([b[0],b[1],z0])
                p2=np.array([b[0],b[1],z1]); p3=np.array([a[0],a[1],z1])
                out.extend([np.asarray([p0,p1,p2]),np.asarray([p0,p2,p3])])
    return out


def _triangulate_face(rings):
    from shapely.geometry import Polygon
    from shapely import constrained_delaunay_triangles
    xyz=[np.asarray(r,float) for r in rings]
    outer=xyz[0]
    if len(outer)<3: return []
    rel=outer-outer[0]
    n=np.cross(rel,np.roll(rel,-1,axis=0)).sum(axis=0)
    if not np.isfinite(n).all() or np.linalg.norm(n)<1e-12: return []
    drop=int(np.argmax(np.abs(n)))
    axes=[i for i in range(3) if i!=drop]
    nonzero=[r for r in xyz if len(r)>=3 and Polygon(r[:,axes]).area>0]
    if not nonzero: return []
    p=Polygon(nonzero[0][:,axes],[r[:,axes] for r in nonzero[1:]])
    if p.is_empty or p.area<=0: return []
    if not p.is_valid: raise ValueError("invalid dominant-projection face")
    point_map={tuple(q[axes]):q for r in nonzero for q in r}
    out=[]
    collection=constrained_delaunay_triangles(p)
    tri_area=sum(t.area for t in collection.geoms)
    if not collection.geoms or abs(tri_area-p.area)>1e-8*max(p.area,1.):
        raise ValueError("constrained triangulation area check failed")
    for tri in collection.geoms:
        q=[tuple(v) for v in list(tri.exterior.coords)[:3]]
        if any(v not in point_map for v in q): raise ValueError("triangulation introduced an unknown vertex")
        v=np.stack([point_map[x] for x in q])
        if np.dot(np.cross(v[1]-v[0],v[2]-v[0]),n)<0: v=v[[0,2,1]]
        out.append(v)
    return out


def _gml_polygon_rings(poly):
    rings=[]
    linear=[x for x in poly.iter() if x.tag.rsplit("}",1)[-1]=="LinearRing"]
    for ring in linear:
        poslist=next((x for x in ring.iter() if x.tag.rsplit("}",1)[-1]=="posList"),None)
        if poslist is not None and poslist.text:
            coords=np.fromstring(poslist.text,sep=" ")
        else:
            positions=[np.fromstring(x.text or "",sep=" ") for x in ring.iter() if x.tag.rsplit("}",1)[-1]=="pos"]
            coords=np.concatenate(positions) if positions else np.asarray([])
        if coords.size%3: continue
        xyz=coords.reshape(-1,3)
        if len(xyz)>1 and np.array_equal(xyz[0],xyz[-1]): xyz=xyz[:-1]
        if len(xyz): xyz=xyz[np.r_[True,np.any(np.diff(xyz,axis=0)!=0,axis=1)]]
        if len(xyz): rings.append(xyz)
    return rings


def _load_osm(path, area, transformer, footprints):
    from shapely.geometry import LineString, Polygon, Point
    root=ET.parse(path).getroot(); nodes={}
    for n in root.findall("node"):
        try: nodes[n.get("id")]=transformer.transform(float(n.get("lon")),float(n.get("lat")))
        except Exception: pass
    road_tags={"residential","living_street","pedestrian","footway","path","cycleway","service","tertiary","secondary","primary","unclassified"}
    ways=[]; water=[]; hotspots=[]; rejected={"bridge_or_tunnel":0,"steps":0}
    for w in root.findall("way"):
        tags={t.get("k"):t.get("v") for t in w.findall("tag")}; ids=[x.get("ref") for x in w.findall("nd")]
        coords=[nodes[x] for x in ids if x in nodes]
        if len(coords)<2: continue
        if "building" in tags or tags.get("building:part"):
            if len(coords)>=4 and coords[0]==coords[-1]:
                try:
                    footprints.append(("osm:"+w.get("id"),Polygon(coords),float(tags.get("height",tags.get("building:height",0)) or 0),"OSM tag/default"))
                except Exception: pass
        if tags.get("highway")=="steps":
            # v1: stairs are never walkable street support (the 1 m DTM
            # cannot represent them); counted, never sampled.
            rejected["steps"]+=1; continue
        if tags.get("highway") in road_tags:
            if any(tags.get(k) in ("yes","true","1") for k in ("tunnel","bridge","covered")) or tags.get("location")=="underground":
                rejected["bridge_or_tunnel"]+=1; continue
            line=LineString(coords)
            if line.is_closed: continue
            # split at OSM vertices. Exclude pieces whose midpoint falls inside
            # any modeled building footprint; final ground check follows.
            for a,b in zip(coords[:-1],coords[1:]):
                if np.linalg.norm(np.subtract(a,b))>0:
                    ways.append(np.asarray([a,b]))
        if tags.get("natural")=="water" or tags.get("waterway") in ("riverbank","canal","river") or tags.get("landuse") in ("reservoir","basin"):
            if coords[0]==coords[-1] and len(coords)>=4: water.append(Polygon(coords))
        if tags.get("amenity") in ("bus_station","marketplace","place_of_worship","library","school","university","cafe","restaurant") or tags.get("public_transport") in ("platform","station") or tags.get("place") in ("square","plaza"):
            if coords: hotspots.append((coords[0],3.0))
    return ways, water, hotspots, rejected


def _netherlands_buildings(files, bounds):
    import json
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely import make_valid
    out=[]; seen=set()
    for path in files:
        opener=gzip.open if path.suffix==".gz" else open
        with opener(path,"rt",encoding="utf8") as f: d=json.load(f)
        if d.get("metadata",{}).get("referenceSystem","").endswith("/7415") is False: raise ValueError("Unexpected CityJSON CRS")
        v=np.asarray(d["vertices"],float)*np.asarray(d["transform"]["scale"])+np.asarray(d["transform"]["translate"])
        objs=d["CityObjects"]
        for ident,obj in objs.items():
            if obj.get("type")!="Building" or ident in seen: continue
            seen.add(ident); attrs=obj.get("attributes",{})
            fp=[]; surfaces=[]
            for k in [ident,*obj.get("children",[])]:
                child=objs[k]
                for g in child.get("geometry",[]):
                    if g.get("lod")=="0":
                        for rings in _all_polys(g.get("boundaries",[])):
                            try: fp.append(Polygon(v[rings[0],:2],[v[r,:2] for r in rings[1:]]))
                            except Exception: pass
                    if g.get("lod")!="2.2": continue
                    semantics=g.get("semantics",{}); sem=semantics.get("values",[])
                    faces=[face for shell in g.get("boundaries",[]) for face in shell]
                    indices=[i for shell in sem for i in (shell if isinstance(shell,list) else [shell])]
                    for face,sid in zip(faces,indices):
                        if sid is None: continue
                        typ=semantics["surfaces"][sid].get("type")
                        rings=[v[r] for r in face]
                        try:
                            tris=_triangulate_face(rings); surfaces.extend((typ,t) for t in tris)
                        except Exception: continue
            if not fp: continue
            footprint=unary_union(fp)
            if footprint.bounds[2]<bounds[0] or footprint.bounds[0]>=bounds[2] or footprint.bounds[3]<bounds[1] or footprint.bounds[1]>=bounds[3]: continue
            roofs=[t for typ,t in surfaces if typ=="RoofSurface"]
            walls=[t for typ,t in surfaces if typ=="WallSurface"]
            grounds=[t for typ,t in surfaces if typ=="GroundSurface"]
            if not roofs or not walls or not grounds: continue
            z0=float(min(t[:,2].min() for t in grounds)); z1=float(max(t[:,2].max() for t in roofs))
            if z1<=z0: continue
            b=Building(ident,footprint,z1-z0,z0,"3DBAG LoD2.2; point-cloud provenance "+str(attrs.get("b3_pw_bron","unknown")),
                       [t for t in roofs],[(t.mean(axis=0)).tolist() for t in walls],
                       [t for _,t in surfaces], [t for typ,t in surfaces if typ=="WallSurface"])
            out.append(b)
    return out


def _all_polys(tree):
    if not tree: return
    if isinstance(tree[0],list) and tree[0] and isinstance(tree[0][0],int):
        yield tree; return
    for x in tree:
        if isinstance(x,list): yield from _all_polys(x)


def _munich_buildings(path, bounds):
    from shapely.geometry import Polygon
    from shapely import make_valid
    from shapely.ops import unary_union
    out=[]; ns={"gml":"http://www.opengis.net/gml"}
    for _,member in ET.iterparse(path,events=("end",)):
        if member.tag.rsplit("}",1)[-1]!="cityObjectMember": continue
        b=next((e for e in member if e.tag.rsplit("}",1)[-1]=="Building"),None)
        if b is None: member.clear(); continue
        ident=b.get("{http://www.opengis.net/gml}id") or f"munich-building-{len(out)}"
        def surfaces(kind):
            ans=[]
            for surf in b.iter():
                if surf.tag.rsplit("}",1)[-1]!=kind: continue
                for poly in surf.iter():
                    if poly.tag.rsplit("}",1)[-1]!="Polygon": continue
                    rings=_gml_polygon_rings(poly)
                    if rings and len(rings[0])>=3: ans.append(rings)
            return ans
        ground,roof,wall=surfaces("GroundSurface"),surfaces("RoofSurface"),surfaces("WallSurface")
        if ground:
            try:
                fps=[make_valid(Polygon(g[0][:,:2],[r[:,:2] for r in g[1:]])) for g in ground]
                fp=unary_union([p for p in fps if not p.is_empty])
                if not (fp.bounds[2]<bounds[0] or fp.bounds[0]>=bounds[2] or fp.bounds[3]<bounds[1] or fp.bounds[1]>=bounds[3]):
                    z0=min(float(r[:,2].min()) for g in ground for r in g)
                    all_xyz=ground+roof+wall
                    z1=max(float(r[:,2].max()) for g in (roof if roof else all_xyz) for r in g)
                    if not np.isfinite(z1-z0) or z1<=z0:
                        z1=z0+12.0
                        height_source="Bavaria LoD2 CityGML footprint; 12m slab fallback (no usable native vertical extent)"
                    else:
                        height_source="Bavaria LoD2 CityGML native surfaces" if roof and wall else "Bavaria LoD2 partial shell; native vertical slab fallback"
                    tris=[]; tri_fail=False
                    for face in ground+roof+wall:
                        # Surface polygons are planar; triangulate in dominant projection.
                        try: tris.extend(_triangulate_face(face))
                        except Exception: tri_fail=True
                    walltris=[]
                    for face in wall:
                        try: walltris.extend(_triangulate_face(face))
                        except Exception: tri_fail=True
                    if tri_fail: height_source += "; slab fallback for incomplete triangulation"
                    # v1: roof supports are triangles (points always lie on the roof);
                    # v0 kept raw rings, so only 3-vertex roof faces were usable.
                    roof_tris=[]
                    for face in roof:
                        try: roof_tris.extend(_triangulate_face(face))
                        except Exception: pass
                    out.append(Building(ident,fp,z1-z0,z0,height_source,
                                        roof_tris,
                                        [np.mean(face[0],axis=0).tolist() for face in wall],tris,walltris))
            except Exception: pass
        member.clear()
    return out


def load_development_maps(cache_root="/tmp/atomwise-phase0b-input", area_ids=DEV_IDS,
                          manifest_path=None, export_dir=None, build_sites=True):
    """Load only checksummed dev maps from the Phase0b cache, exporting geometry."""
    area_ids=tuple(area_ids)
    invalid=set(area_ids)-set(DEV_IDS)
    if invalid: raise ValueError(f"Refusing non-development map: {sorted(invalid)[0]}")
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import Polygon, box, Point
    from shapely.ops import unary_union
    cache=Path(cache_root); manifest_path=Path(manifest_path or ROOT/"outputs/audits/bench-phase0b-20261108/map_downloads.json")
    rows={r["id"]:r for r in json.loads(manifest_path.read_text())}
    cities={}
    for map_id in area_ids:
        row=rows[map_id]; spec=AREAS[map_id]
        entries=[*row["buildings"],row["terrain"],row["osm"]]
        sources={}
        for e in entries:
            p=cache/e["file"]
            if not p.exists() or p.stat().st_size!=e["bytes"] or _sha(p)!=e["sha256"]:
                raise ValueError(f"Phase0b cache integrity mismatch: {p}")
            sources[e["file"]]=e["sha256"]
        bfiles=[cache/e["file"] for e in row["buildings"]]
        # Munich is one 160 MB CityGML file; the streaming parser keeps memory bounded.
        buildings=(_munich_buildings(bfiles[0],spec["bounds"]) if map_id=="munich"
                   else _netherlands_buildings(bfiles,spec["bounds"]))
        trans=Transformer.from_crs("EPSG:4326",spec["crs"],always_xy=True)
        osmways,water,hotspots,osm_rejections=_load_osm(cache/row["osm"]["file"],spec,trans,[])
        water_union=unary_union(water) if water else None
        # Exclude tunnel/bridge roads and all road portions inside modeled
        # footprints; validate every retained edge at <=1m spacing.
        from shapely.geometry import LineString
        provisional=[]; reject={**osm_rejections,"building_interior":0,"water":0,"invalid_terrain":0}
        for seg in osmways:
            if any(b.footprint.intersects(LineString(seg)) for b in buildings):
                reject["building_interior"]+=1; continue
            provisional.append(seg)
        with rasterio.open(cache/row["terrain"]["file"]) as dem:
            city=CityMap(map_id,spec["bounds"],spec["crs"],dem,buildings,[],[],[],water_union,sources)
            for seg in provisional:
                n=max(1,int(np.ceil(np.linalg.norm(seg[1]-seg[0]))))
                pts=seg[0]+(seg[1]-seg[0])*np.linspace(0,1,n+1)[:,None]
                if np.all(city.outdoor(pts)):
                    city.street_segments=np.concatenate([city.street_segments,seg[None,:,:]],axis=0)
                else:
                    _,_,valid=city.sample_ground(pts)
                    if not np.all(valid): reject["invalid_terrain"]+=1
                    elif water_union is not None and any(water_union.covers(Point(*p)) for p in pts): reject["water"]+=1
                    else: reject["other_outdoor"] = reject.get("other_outdoor",0)+1
            # POIs / plazas only retained when outdoor; also add street-junction
            # weighted hotspots so absent POI tags do not erase plausible nodes.
            valid_hot=[]; weights=[]
            for xy,w in hotspots:
                if city.outdoor([xy])[0]: valid_hot.append(xy); weights.append(w)
            if len(city.street_segments):
                ends=city.street_segments.reshape(-1,2)
                rounded=np.round(ends,1); unique,counts=np.unique(rounded,axis=0,return_counts=True)
                for xy,c in zip(unique[counts>=3],counts[counts>=3]):
                    if city.outdoor([xy])[0]: valid_hot.append(xy); weights.append(float(c))
            city.hotspots_xy=np.asarray(valid_hot,float).reshape(-1,2)
            city.hotspot_weights=np.asarray(weights,float)
            city.geometry_rejections=reject
            if build_sites: city._prepare_site_pairs()   # v0 pool; v1 plans sites in users.py
            cities[map_id]=city
            if export_dir is not None: city.export_geometry(export_dir)
    return cities

