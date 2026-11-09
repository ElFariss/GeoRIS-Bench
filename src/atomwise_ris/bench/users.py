"""Terrain-constrained crowd and geometry-only link labels (v1 previews).

v1 changes versus v0 (see docs/bench-users-v1-design-20261108.md):
one DTM-validity rule (4x4 stencil, in users_geometry), a declared walkable
slope cap, DTM slope labels only for ground-supported pedestrians, BS/RIS site
planning with BS-RIS LOS in a declared range and RIS coverage of BS-shadow
streets, RIS-reachability-aware served-group scheduling, and honest link names
(``no_los`` instead of ``blocked``).
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DT = .02
SEED = 202611081501
DEVELOPMENT = ("munich", "rotterdam", "maastricht")
HEIGHTS = np.array(["pedestrian", "elevated", "drone"])

# --- v1 declared parameters -------------------------------------------------
WALK_SLOPE_CAP_DEG = 15.          # pedestrians / walking paths: DTM slope <= cap
MIN_STREET_COMPONENT_NODES = 8    # drop street fragments left after the cap
SITE_DISTANCE_M = (20., 150.)     # 3-D BS-RIS distance, incl. the mast offset
RIS_AGL_M = (3., 30.)             # RIS facade height above local ground
ROOF_BS_MIN_AGL_M = 6.            # rooftop BS support above local ground
MAST_BS_M = 12.                   # street-level mast base height
ROOF_ANTENNA_MIN_M = 1.5          # antenna above a roof surface point (min offset)
PROBE_AGL_M = 1.5                 # geometry-only probe height for site planning
MAX_PROBES = 1200
MAX_PAIR_EVAL = 1000
BS_MIN_LOS_SHARE = .10            # BS must see >= this share of street probes
RIS_MIN_LOS_SHARE = .10           # RIS must see >= this share of street probes
SHADOW_MIN_SHARE = .05            # share of ALL probes: RIS-LOS and BS no-LOS
SHADOW_MIN_POINTS = 25
MAX_SITES = 16
MAX_SITES_PER_BS = 2
MIN_RIS_SEPARATION_M = 25.
SERVED_NEGATIVE_SHARE = .15       # served slots deliberately drawn without RIS LOS
RIS_REACH_MIN_FRACTION = .5       # clip fraction with RIS LOS to count as reachable

# Link classes. "no_los" = no clear direct segment; reflections are NOT traced
# in this preview, so the candidate flag is a bounded one-bounce geometry test.
LOS = "los"
NO_LOS_REFLECTION_CANDIDATE = "no_los_reflection_candidate"
NO_LOS = "no_los"
NO_LOS_UNRESOLVED = "no_los_unresolved"
LINK_CLASSES = (LOS, NO_LOS_REFLECTION_CANDIDATE, NO_LOS, NO_LOS_UNRESOLVED)


class Streets:
    """Continuous paths through the validated OSM street-segment graph."""

    def __init__(self, segments):
        nodes = {}
        neighbors = defaultdict(set)

        def vertex(x):
            k = tuple(np.round(x, 5))
            return nodes.setdefault(k, len(nodes))

        for a,b in np.asarray(segments):
            d = float(np.linalg.norm(a-b))
            if d < 1e-4:
                continue
            points = np.linspace(a,b,max(2,1+int(np.ceil(d/1.75))))
            i = vertex(points[0])
            for p in points[1:]:
                j = vertex(p)
                if j != i:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                i = j
        if not nodes:
            raise ValueError("no terrain-valid OSM street edges")
        self.xy = np.asarray(list(nodes))
        self.tree = cKDTree(self.xy)
        self.neighbors = [np.asarray(sorted(neighbors[i]),dtype=int) for i in range(len(nodes))]

    def prune(self, city, cap_deg, min_component=MIN_STREET_COMPONENT_NODES):
        """Remove graph nodes/edges that fail ground validity or exceed the slope cap.

        Nothing is filled or relocated: the graph just loses the offending
        support, and tiny leftover fragments. Returns a dict of counts.
        """
        n = len(self.xy)
        _, slope, ok = city.sample_ground(self.xy)
        ok = np.asarray(ok, dtype=bool)
        node_ok = ok & (np.asarray(slope) <= cap_deg)
        rows = [(i, int(j)) for i in range(n) for j in self.neighbors[i] if i < j]
        e = np.asarray(rows, dtype=int).reshape(-1, 2)
        counts = {"street_nodes_total": n, "street_nodes_invalid_ground": int((~ok).sum()),
                  "street_nodes_slope_over_cap": int((ok & ~node_ok).sum()),
                  "street_edges_total": len(e)}
        both = node_ok[e[:, 0]] & node_ok[e[:, 1]]
        _, mslope, mok = city.sample_ground((self.xy[e[:, 0]]+self.xy[e[:, 1]])/2)
        mid_ok = np.asarray(mok, dtype=bool) & (np.asarray(mslope) <= cap_deg)
        counts["street_edges_midpoint_slope_or_ground"] = int((both & ~mid_ok).sum())
        e = e[both & mid_ok]
        counts["street_edges_kept_before_fragments"] = len(e)
        adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        k, label = connected_components(adj, directed=False)
        sizes = np.bincount(label, minlength=k)
        keep = (sizes[label] >= min_component) & node_ok
        counts["street_nodes_in_small_fragments"] = int((node_ok & ~keep).sum())
        if not keep.any():
            raise ValueError("no street support left under the slope cap")
        new_id = -np.ones(n, dtype=int)
        new_id[keep] = np.arange(int(keep.sum()))
        neighbors = [set() for _ in range(int(keep.sum()))]
        for a, b in e:
            if keep[a] and keep[b]:
                neighbors[new_id[a]].add(int(new_id[b]))
                neighbors[new_id[b]].add(int(new_id[a]))
        self.xy = self.xy[keep]
        self.tree = cKDTree(self.xy)
        self.neighbors = [np.asarray(sorted(x), dtype=int) for x in neighbors]
        counts["street_nodes_kept"] = int(keep.sum())
        counts["street_edges_kept"] = int(sum(len(x) for x in self.neighbors)//2)
        return counts

    def walk(self, start, length, rng, counts):
        ids = [int(start)]
        distance = [0.]
        while distance[-1] < length:
            now = ids[-1]
            options = self.neighbors[now]
            weights = None
            if len(ids)>1:
                prev = ids[-2]
                if len(options)>1:
                    options = options[options!=prev]
                else:
                    counts["route_dead_end_turns"] += 1
                if len(options)>1:
                    head = self.xy[now]-self.xy[prev]
                    vec = self.xy[options]-self.xy[now]
                    cosine = (vec@head)/(np.linalg.norm(vec,axis=1)*np.linalg.norm(head))
                    weights = np.exp(2*cosine)
                    weights /= weights.sum()
            if len(options)==0 or len(ids)>16000:
                counts["route_graph_failure"] += 1
                return None
            nxt = int(rng.choice(options,p=weights))
            ids.append(nxt)
            distance.append(distance[-1]+float(np.linalg.norm(self.xy[nxt]-self.xy[now])))
        return self.xy[ids], np.asarray(distance)

    @staticmethod
    def position(route, distances):
        xy,s = route
        k = np.minimum(np.searchsorted(s,distances,side="right")-1,len(s)-2)
        k = np.maximum(0,k)
        vec = xy[k+1]-xy[k]
        fraction = (distances-s[k])/(s[k+1]-s[k])
        return xy[k]+fraction[...,None]*vec, vec/np.linalg.norm(vec,axis=-1,keepdims=True)


class Facets:
    """Bounded one-bounce visibility test, without field coefficients or rates."""

    def __init__(self, triangles):
        t = np.asarray(triangles,dtype=float).reshape(-1,3,3)
        e1,e2 = t[:,1]-t[:,0], t[:,2]-t[:,0]
        norms = np.cross(e1,e2)
        size = np.linalg.norm(norms,axis=1)
        good = size>1e-7
        self.t,self.area = t[good],.5*size[good]
        self.norm = norms[good]/size[good,None]
        self.center = self.t.mean(axis=1)
        orientation = np.where(self.norm[:,0]<-1e-6,-1.,
                               np.where(abs(self.norm[:,0])<=1e-6,
                                        np.where(self.norm[:,1]<-1e-6,-1.,1.),1.))
        normalized = self.norm*orientation[:,None]
        keys = np.c_[np.round(normalized,3),np.round((normalized*self.t[:,0]).sum(1),1)]
        self.plane = np.unique(keys,axis=0,return_inverse=True)[1] if len(keys) else np.array([],dtype=int)

    def shortlist(self, source):
        score = self.area/(15.+np.linalg.norm(self.center-source,axis=1))**1.4
        order = np.argsort(-score,kind="stable")
        used = Counter()
        chosen = []
        for i in order:
            k = int(self.plane[i])
            if used[k]<2:
                chosen.append(i)
                used[k]+=1
                if len(chosen)==32:
                    break
        return chosen

    def visible(self, city, source, receivers):
        out = np.zeros(len(receivers),dtype=bool)
        if len(receivers)==0 or len(self.t)==0:
            return out
        for i in self.shortlist(source):
            j = np.flatnonzero(~out)
            if not len(j):
                break
            tri = self.t[i]
            normal = self.norm[i].copy()
            base = tri[0]
            sd = float((source-base)@normal)
            if abs(sd)<.05:
                continue
            if sd<0:
                normal *= -1
                sd *= -1
            rd = (receivers[j]-base)@normal
            keep = rd>.05
            if not keep.any():
                continue
            j,rd = j[keep],rd[keep]
            image = source-2*sd*normal
            hits = image + (receivers[j]-image)*(sd/(rd+sd))[:,None]
            a,b = tri[1]-base,tri[2]-base
            aa,ab,bb = a@a,a@b,b@b
            ha,hb = (hits-base)@a,(hits-base)@b
            det = aa*bb-ab*ab
            u = (bb*ha-ab*hb)/det
            v = (aa*hb-ab*ha)/det
            inside = (u>1e-7)&(v>1e-7)&(u+v<1.-1e-7)
            if not inside.any():
                continue
            j = j[inside]
            probe = hits[inside] + .035*normal
            front,_ = city.los_batch(source,probe)
            front = np.asarray(front,dtype=bool)
            if front.any():
                j,probe = j[front],probe[front]
                back,_ = city.los_batch(probe,receivers[j])
                out[j[np.asarray(back,dtype=bool)]] = True
        return out


@dataclass
class Crowd:
    xyz: np.ndarray
    slope: np.ndarray
    agl: np.ndarray
    speed: np.ndarray
    ground: np.ndarray
    group: np.ndarray
    height: np.ndarray
    ground_ref: np.ndarray
    ground_building: np.ndarray
    ground_cell: np.ndarray

    @property
    def size(self):
        return self.xyz.shape[1]

    @staticmethod
    def join(groups):
        if not groups:
            return None
        return Crowd(*(np.concatenate([getattr(g,k) for g in groups],axis=1 if k in
                                         {"xyz","slope","agl","speed","ground"} else 0)
                       for k in Crowd.__dataclass_fields__))


class Generator:
    def __init__(self, city, map_id, site_rng=None):
        if map_id not in DEVELOPMENT:
            raise ValueError(f"held-out or unknown map refused: {map_id}")
        self.city = city
        self.map_id = map_id
        self.bounds = getattr(city,"bounds",None)
        segments=np.asarray(city.street_segments,dtype=float)
        kept=self.within(segments.reshape(-1,2)).reshape(-1,2).all(axis=1)
        self.streets = Streets(segments[kept])
        # One ground rule + declared walkable slope cap: pruned, never filled.
        self.street_prune = self.streets.prune(city,WALK_SLOPE_CAP_DEG)
        self.facets = Facets(getattr(city,"wall_triangles",[]))
        roofs,roof_ids=[],[]
        if hasattr(city,"buildings"):
            for building in city.buildings:
                for face in building.roof_polygons:
                    tri=np.asarray(face,dtype=float)
                    if tri.shape!=(3,3):
                        continue
                    normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
                    if np.linalg.norm(normal)>1e-9 and abs(normal[2])>=np.cos(np.deg2rad(10.))*np.linalg.norm(normal):
                        roofs.append(tri.mean(axis=0))
                        roof_ids.append(building.id)
        else:
            roofs=city.roof_xyz
            roof_ids=[None]*len(roofs)
        self.roofs = np.asarray(roofs,dtype=float).reshape(-1,3)
        self.roof_ground = np.empty(0)
        self.roof_ref = np.empty(0,dtype="<U40")
        self.roof_ids = np.empty(0,dtype=object)
        self.roof_cells = np.empty((0,2),dtype=int)
        self.roof_rejections={"invalid_ground_or_outside":0,"support_below_3m":0}
        if len(self.roofs):
            if getattr(city,"ground_fill",None) is not None:
                gz,refs,cells=city.ground_fill.lookup(roof_ids,self.roofs[:,:2])
            else:
                gz,ref=self.support_ground(roof_ids,self.roofs,building_mounted=True)
                refs=np.full(len(self.roofs),ref,dtype="<U40")
                cells=np.full((len(self.roofs),2),-1,dtype=int)
            admissible=np.isfinite(gz)&self.within(self.roofs[:,:2])
            high_enough=(self.roofs[:,2]-gz)>=3.
            self.roof_rejections={"invalid_ground_or_outside":int((~admissible).sum()),
                                  "support_below_3m":int((admissible&~high_enough).sum())}
            self.roofs = self.roofs[admissible&high_enough]
            keep=admissible&high_enough
            self.roof_ground = gz[keep]
            self.roof_ref = refs[keep]
            self.roof_ids = np.asarray(roof_ids,dtype=object)[keep]
            self.roof_cells = cells[keep]
        self.site_stats = {}
        if site_rng is None:
            # Fixture/legacy path: sites supplied by the city object.
            self.sites = [s for s in city.sites if self.within(np.asarray([s["bs_xyz"][:2],s["ris_xyz"][:2]])).all()]
        else:
            self.sites, self.site_stats = self.plan_sites(site_rng)
        if len(self.sites)<1:
            raise ValueError(f"no valid BS/RIS sites in {map_id}")
        self.hotspots = np.asarray(city.hotspots_xy,dtype=float).reshape(-1,2)
        self.weights = np.asarray(city.hotspot_weights,dtype=float).copy()
        if len(self.weights):
            self.weights /= self.weights.sum()

    def within(self,xy):
        xy=np.asarray(xy)
        if self.bounds is None:
            return np.ones(len(xy),dtype=bool)
        return ((xy>=self.bounds[:2])&(xy<self.bounds[2:])).all(axis=1)

    def outdoor(self,xy):
        return self.within(xy)&np.asarray(self.city.outdoor(xy),dtype=bool)

    # ---- BS / RIS site planning (geometry only) -----------------------------
    def support_ground(self, ids, xyz, building_mounted):
        """Ground reference z for supports; NaN where no valid ground exists.

        Building-mounted supports use the optional v1.1 footprint layer, or
        the v1 building-foot DTM ring. Everything else samples the original
        DTM at its own XY and requires the full valid 4x4 stencil.
        Returns (z, method).
        """
        xyz = np.asarray(xyz,dtype=float).reshape(-1,3)
        if building_mounted and getattr(self.city,"ground_fill",None) is not None:
            z,methods,_=self.city.ground_fill.lookup(ids,xyz[:,:2])
            return z,methods
        if building_mounted and hasattr(self.city,"building_foot_ground"):
            return self.city.building_foot_ground(ids,xyz[:,:2]),"building_foot_dtm"
        z,_,ok = self.city.sample_ground(xyz[:,:2])
        return np.where(np.asarray(ok,dtype=bool),z,np.nan),"dtm_at_position"

    @staticmethod
    def _thin(points, cell, rng):
        """One random representative per ``cell`` x ``cell`` metre XY bin."""
        order = rng.permutation(len(points))
        keys = np.floor(points[order,:2]/cell).astype(np.int64)
        _, first = np.unique(keys,axis=0,return_index=True)
        return np.sort(order[first])

    def plan_sites(self, rng):
        """Build the eligible site pool for this map.

        A site = BS support + RIS facade point with a clear BS-RIS segment at a
        3-D distance in ``SITE_DISTANCE_M`` and RIS coverage of street probes the
        BS cannot see. Only LOS ray tests are used (never rates).
        """
        city = self.city
        stats = {"ris_candidates":0,"bs_roof_candidates":0,"bs_mast_candidates":0,
                 "pairs_in_range":0,"pairs_bs_ris_los":0,"pairs_evaluated":0,
                 "pairs_failing_coverage":0,"pairs_eligible":0}
        fac = np.asarray(city.facade_xyz,dtype=float).reshape(-1,3)
        fac_ids = np.asarray(getattr(city,"facade_owner_ids",np.full(len(fac),None,dtype=object)),dtype=object)
        fac_g = np.empty(0)
        stats["ris_facade_points_total"] = len(fac)
        if len(fac):
            g,method = self.support_ground(fac_ids,fac,building_mounted=True)
            agl = fac[:,2]-g
            stats["ris_facade_no_valid_ground_reference"] = int((~np.isfinite(g)).sum())
            keep = np.isfinite(g)&self.within(fac[:,:2])&(agl>=RIS_AGL_M[0])&(agl<=RIS_AGL_M[1])
            fac,fac_ids,fac_g = fac[keep],fac_ids[keep],g[keep]
            t = self._thin(fac,10.,rng)
            fac,fac_ids,fac_g = fac[t],fac_ids[t],fac_g[t]
        stats["ris_candidates"] = len(fac)
        roof = np.asarray(city.roof_xyz,dtype=float).reshape(-1,3)
        roof_ids = np.asarray(getattr(city,"roof_owner_ids",np.full(len(roof),None,dtype=object)),dtype=object)
        roof_g = np.empty(0)
        if len(roof):
            g,_ = self.support_ground(roof_ids,roof,building_mounted=True)
            keep = np.isfinite(g)&self.within(roof[:,:2])&((roof[:,2]-g)>=ROOF_BS_MIN_AGL_M)
            roof,roof_ids,roof_g = roof[keep],roof_ids[keep],g[keep]
            t = self._thin(roof,15.,rng)
            roof,roof_ids,roof_g = roof[t],roof_ids[t],roof_g[t]
        mast_xy = self.streets.xy[np.sort(rng.choice(len(self.streets.xy),min(80,len(self.streets.xy)),replace=False))]
        mg,_,mok = city.sample_ground(mast_xy)
        mok = np.asarray(mok,dtype=bool)
        mast = np.c_[mast_xy[mok],mg[mok]+MAST_BS_M]
        stats["bs_roof_candidates"],stats["bs_mast_candidates"] = len(roof),len(mast)
        # Roof anchors are evaluated at the lowest antenna position (roof + 1.5 m):
        # a ray cannot start on the roof surface itself.
        anchors = ([(p+np.array([0.,0.,ROOF_ANTENNA_MIN_M]),"rooftop",str(i),gz) for p,i,gz in zip(roof,roof_ids,roof_g)]+
                   [(p,"mast",None,p[2]-MAST_BS_M) for p in mast])
        if not len(fac) or not anchors:
            return [],stats
        # 1. clear BS-RIS segments in the declared 3-D distance range
        pairs = []
        for a,(bs,_,_,_) in enumerate(anchors):
            d = np.linalg.norm(fac-bs,axis=1)
            near = np.flatnonzero((d>=SITE_DISTANCE_M[0])&(d<=SITE_DISTANCE_M[1]))
            stats["pairs_in_range"] += len(near)
            if len(near):
                vis,_ = city.los_batch(bs,fac[near])
                pairs.extend((a,int(r)) for r in near[np.asarray(vis,dtype=bool)])
        stats["pairs_bs_ris_los"] = len(pairs)
        if not pairs:
            return [],stats
        chosen = np.sort(rng.choice(len(pairs),MAX_PAIR_EVAL,replace=False)) if len(pairs)>MAX_PAIR_EVAL else np.arange(len(pairs))
        # 2. street probes: shadow of the BS that the RIS can see
        pi = np.sort(rng.choice(len(self.streets.xy),min(MAX_PROBES,len(self.streets.xy)),replace=False))
        pxy = self.streets.xy[pi]
        pz,_,pok = city.sample_ground(pxy)
        pok = np.asarray(pok,dtype=bool)
        probes = np.c_[pxy[pok],pz[pok]+PROBE_AGL_M]
        bs_vis,ris_vis={},{}

        def evaluate(pair_indices):
            eligible=[]
            for i in pair_indices:
                a,r=pairs[i]
                stats["pairs_evaluated"]+=1
                if a not in bs_vis:
                    bs_vis[a]=np.asarray(city.los_batch(anchors[a][0],probes)[0],dtype=bool)
                if r not in ris_vis:
                    ris_vis[r]=np.asarray(city.los_batch(fac[r],probes)[0],dtype=bool)
                b,ris=bs_vis[a],ris_vis[r]
                shadow=int((ris&~b).sum())
                if (b.mean()<BS_MIN_LOS_SHARE or ris.mean()<RIS_MIN_LOS_SHARE or
                        shadow<max(SHADOW_MIN_POINTS,SHADOW_MIN_SHARE*len(probes))):
                    stats["pairs_failing_coverage"]+=1
                    continue
                eligible.append((a,r,{"probe_count":len(probes),"bs_los_share":float(b.mean()),
                                      "ris_los_share":float(ris.mean()),
                                      "ris_covers_bs_shadow_share":float(shadow/len(probes)),
                                      "shadow_share_covered_by_ris":float(shadow/max(1,int((~b).sum())))}))
            stats["pairs_eligible"]+=len(eligible)
            return eligible
        # 3. diverse, seeded selection: separated RIS points, few sites per BS anchor
        sites,used,ris_xy = [],Counter(),[]
        def select(eligible,limit):
            for k in rng.permutation(len(eligible)):
                a,r,cov=eligible[int(k)]
                if used[a]>=MAX_SITES_PER_BS or any(np.linalg.norm(fac[r,:2]-q)<MIN_RIS_SEPARATION_M for q in ris_xy):
                    continue
                bs,support,building,bs_ground=anchors[a]
                used[a]+=1
                ris_xy.append(fac[r,:2].copy())
                site_refs={}
                for which,mounted,ident,pos in (("bs",support=="rooftop",building,bs),
                                               ("ris",True,str(fac_ids[r]),fac[r])):
                    site_refs.update(self._ground_record(which,mounted,ident,pos))
                sites.append({"site_id":f"{self.map_id}-v1-site-{len(sites):02d}","bs_xyz":bs.tolist(),
                              "ris_xyz":fac[r].tolist(),"bs_agl_m":float(bs[2]-bs_ground),"ris_agl_m":float(fac[r,2]-fac_g[r]),
                              "bs_ground_z_m":float(bs_ground),"ris_ground_z_m":float(fac_g[r]),
                              **site_refs,
                              "bs_support":support,"bs_building_id":building,"ris_building_id":str(fac_ids[r]),
                              "ris_support":"facade","bs_ris_los":True,
                              "bs_ris_base_distance_m":float(np.linalg.norm(bs-fac[r])),**cov})
                if len(sites)>=limit:
                    break

        select(evaluate(chosen),MAX_SITES)
        if getattr(city,"ground_fill",None) is not None and len(sites)<6:
            unused=np.setdiff1d(np.arange(len(pairs)),chosen,assume_unique=True)
            stats["sites_before_repair"]=len(sites)
            extra=np.sort(rng.choice(unused,min(3000,len(unused)),replace=False))
            stats["repair_pairs_evaluated"]=len(extra)
            select(evaluate(extra),6)
        stats["sites_selected"] = len(sites)
        return sites,stats

    def _ground_record(self, which, mounted, ident, xyz):
        layer=getattr(self.city,"ground_fill",None)
        if mounted and layer is not None:
            z,m,rc=layer.lookup([ident],np.asarray(xyz)[None,:2])
            r,c=map(int,rc[0])
            points=[[float(layer.x[c]),float(layer.y[r]),float(z[0])]] if r>=0 else []
            return {f"{which}_ground_method":str(m[0]),f"{which}_ground_samples":points,
                    f"{which}_ground_cell":[r,c]}
        return {f"{which}_ground_method":"building_foot_dtm" if mounted else "dtm_at_position",
                f"{which}_ground_samples":self._ground_samples("rooftop" if mounted else "mast",ident,xyz)}

    def _ground_samples(self, support, building_id, xyz):
        """The valid DTM samples (x,y,z) a ground reference rests on (for read-back)."""
        xyz = np.asarray(xyz,dtype=float)
        if support=="rooftop" and hasattr(self.city,"building_foot_samples"):
            return np.asarray(self.city.building_foot_samples(building_id,xyz[:2])).tolist()
        z,_,_ = self.city.sample_ground(xyz[None,:2])
        return [[float(xyz[0]),float(xyz[1]),float(z[0])]]

    def choose_site(self, rng, counts):
        for _ in range(25):
            site = self.sites[int(rng.integers(len(self.sites)))]
            bs = np.asarray(site["bs_xyz"],dtype=float).copy()
            ris = np.asarray(site["ris_xyz"],dtype=float).copy()
            offset = min(24.,float(rng.lognormal(np.log(4),.75)))
            if site.get("bs_support")=="rooftop":
                # site bs_xyz already includes the minimum antenna height
                offset = max(0.,offset-ROOF_ANTENNA_MIN_M)
            bs[2] += offset
            distance = float(np.linalg.norm(bs-ris))
            if not SITE_DISTANCE_M[0]<=distance<=SITE_DISTANCE_M[1]:
                counts["site_distance_out_of_range_after_mast_draw"] += 1
                continue
            vis,_ = self.city.los_batch(bs,ris[None])
            if not bool(np.ravel(vis)[0]):
                counts["site_bs_ris_blocked_after_mast_draw"] += 1
                continue
            extra = {k:v for k,v in site.items() if k not in {"bs_xyz","ris_xyz","bs_agl_m","ris_agl_m"}}
            bs_ground = float(site["bs_ground_z_m"]) if "bs_ground_z_m" in site else float(self.city.sample_ground(bs[None,:2])[0][0])
            ris_ground = float(site["ris_ground_z_m"]) if "ris_ground_z_m" in site else float(self.city.sample_ground(ris[None,:2])[0][0])
            return {**extra,"site_id":str(site.get("site_id","site")),"bs_xyz":bs,"ris_xyz":ris,
                    "bs_agl_m":float(bs[2]-bs_ground),"ris_agl_m":float(ris[2]-ris_ground),
                    "bs_ris_distance_m":distance}
        raise RuntimeError(f"{self.map_id}: no eligible BS/RIS after mast draws")

    @staticmethod
    def span(n):
        return min(10.,2.5+.45*max(0,n-4))

    @staticmethod
    def phone(rng,n):
        return 1.+.7*rng.beta(2.5,2,n)

    @staticmethod
    def cruise(rng):
        if rng.random()<.24:
            return 0.
        typ = rng.random()
        if typ<.025:
            return float(rng.uniform(4,7))
        if typ<.125:
            return min(.9,float(rng.lognormal(np.log(.45),.30)))
        return min(3.5,float(rng.lognormal(np.log(1.3),.27)))

    def anchor(self, rng, counts, isolated=False):
        for _ in range(20):
            if isolated or not len(self.hotspots) or rng.random()<.15:
                pt = self.streets.xy[int(rng.integers(len(self.streets.xy)))]
            else:
                pt = self.hotspots[int(rng.choice(len(self.hotspots),p=self.weights))]
            d,i = self.streets.tree.query(pt+rng.normal(0,7,2))
            if d<=25.:
                return int(i)
            counts["anchor_snap_too_far"] += 1
        counts["anchor_retry_exhausted"] += 1
        return None

    @staticmethod
    def travel(t, cruise, rng):
        v = np.full(len(t),cruise)
        if cruise>0 and len(t)>1:
            for start in rng.uniform(0,t[-1],rng.poisson(.12*t[-1])):
                end = start+min(3.,float(rng.exponential(.9)))
                v[(t>=start)&(t<end)] = 0.
        return np.r_[0.,np.cumsum(v[:-1])*DT],v

    def _support(self, loc, ped):
        """(valid, steep) per sample: outdoor ground, and pedestrian slope > cap."""
        f,n,_ = loc.shape
        flat = loc.reshape(-1,2)
        valid = np.asarray(self.outdoor(flat),dtype=bool).reshape(f,n)
        _,slope,_ = self.city.sample_ground(flat)
        steep = ped[None,:] & ~(np.asarray(slope).reshape(f,n)<=WALK_SLOPE_CAP_DEG)
        return valid,steep

    def group_on_route(self, route, distance, velocity, n, group, rng, counts, force_ped=False):
        f = len(distance)
        span = self.span(n)
        shift = rng.uniform(-span,span,n)
        shift[0]=0.
        at = distance[:,None]+span+shift[None,:]
        center, _ = Streets.position(route,at)
        before,_ = Streets.position(route,np.maximum(0.,at-1.))
        after,_ = Streets.position(route,np.minimum(route[1][-1],at+1.))
        head = after-before
        tangent = np.stack([-head[...,1],head[...,0]],axis=-1)
        tangent /= np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-9)
        phase = rng.uniform(0,2*np.pi,n)
        lateral = rng.uniform(-.65,.65,n)[None,:]+.15*np.sin(phase[None,:]+distance[:,None]*.7)
        lateral[:,0]=0.
        local = center+lateral[...,None]*tangent
        codes = np.zeros(n,dtype=int) if force_ped else rng.choice(3,n,p=[.94,.055,.005])
        if not len(self.roofs) and (codes==1).any():
            counts["roof_no_valid_ground_or_support"] += int((codes==1).sum())
            codes[codes==1] = 0
        ped = codes==0
        active = codes!=1          # rooftop users are relocated to a roof below
        good,steep = self._support(local,ped)
        counts["follower_lateral_samples_rejected"] += int(np.count_nonzero(~good[:,active]))
        counts["follower_lateral_slope_cap_samples"] += int(np.count_nonzero(steep[:,active]&good[:,active]))
        reject = (((~good)|steep)&active[None,:]).any(axis=0)
        if f>1:
            # Revert a complete follower track rather than teleporting across
            # the road at a rejected sample or a rapidly turning tangent.
            step = np.linalg.norm(np.diff(local,axis=0),axis=-1)
            speed_guard = (step > 1.45*np.diff(distance)[:,None]+1e-8).any(axis=0)
            counts["follower_lateral_speed_guard_tracks"] += int(speed_guard.sum())
            reject |= speed_guard
        local[:,reject]=center[:,reject]
        good,steep = self._support(local,ped)
        if not good[:,active].all():
            counts["invalid_centreline_routes"] += 1
            return None
        if steep[:,active].any():
            counts["centreline_slope_cap_routes"] += 1
            return None
        agl = self.phone(rng,n)
        roof_z,roof_g,roof_index = {},{},{}
        for j in np.flatnonzero(codes==1):
            # Draw an actual near-level roof close in spatial rank, not a BS/RIS band.
            d = np.linalg.norm(self.roofs[:,:2]-local[0,j],axis=1)
            nearest = np.argsort(d,kind="stable")[:min(6,len(d))]
            k = int(rng.choice(nearest))
            r = self.roofs[k]
            local[:,j] = r[:2]
            roof_z[int(j)] = r[2]+agl[j]
            roof_g[int(j)] = self.roof_ground[k]
            roof_index[int(j)] = k
        for j in np.flatnonzero(codes==2):
            while True:
                v = float(rng.lognormal(np.log(34),.5))
                if 20<=v<=80:
                    agl[j]=v
                    break
                counts["drone_height_outside_20_80"] += 1
        ground, slope, valid = self.city.sample_ground(local.reshape(-1,2))
        valid = np.asarray(valid,dtype=bool).reshape(f,n)
        # Rooftop users have no DTM at their XY: their ground reference is the
        # building-foot DTM ring (declared); every other sample needs valid ground.
        valid[:,codes==1] = True
        if not valid.all():
            counts["route_ground_samples_rejected"] += int(np.count_nonzero(~valid))
            return None
        ground = np.asarray(ground).reshape(f,n).copy()
        for j,gz in roof_g.items():
            ground[:,j] = gz
        slope = np.asarray(slope).reshape(f,n).copy()
        # A DTM slope label belongs only to a user standing on that ground.
        slope[:,codes!=0] = np.nan
        z = ground+agl[None,:]
        for j,v in roof_z.items():
            z[:,j]=v
        xyz = np.concatenate([local,z[...,None]],axis=-1)
        speed = np.zeros((f,n))
        if f>1:
            speed[1:]=np.linalg.norm(np.diff(xyz,axis=0),axis=-1)/DT
            speed[0]=speed[1]
        else:
            speed[0]=velocity[0]
            speed[0,codes==1]=0.
        ref=np.full(n,"dtm_at_position",dtype="<U40")
        ident=np.full(n,None,dtype=object)
        cells=np.full((n,2),-1,dtype=int)
        for j,k in roof_index.items():
            ref[j]=self.roof_ref[k]; ident[j]=self.roof_ids[k]; cells[j]=self.roof_cells[k]
        return Crowd(xyz,slope,z-ground,speed,ground,np.full(n,group),HEIGHTS[codes],ref,ident,cells)

    def natural(self,t,rng,counts):
        intensity = float(np.clip(rng.lognormal(np.log(9),.45),2.,22.))
        g, single = rng.poisson(intensity),rng.poisson(3.)
        groups = []
        for k in range(g+single):
            size = 1
            if k<g:
                size = int(rng.zipf(1.9))
                while size>24:
                    counts["group_size_tail_resampled"] += 1
                    size = int(rng.zipf(1.9))
            cruise = self.cruise(rng)
            dist,vel = self.travel(t,cruise,rng)
            for _ in range(12):
                anchor = self.anchor(rng,counts,isolated=k>=g)
                if anchor is None:
                    break
                route = self.streets.walk(anchor,float(dist[-1]+2*self.span(size)+.01),rng,counts)
                if route is None:
                    continue
                group = self.group_on_route(route,dist,vel,size,k,rng,counts)
                if group is not None:
                    groups.append(group)
                    break
            else:
                counts["groups_route_exhausted"] += 1
        return Crowd.join(groups)

    def positions_static(self,points,z,labels,t,rng,group,ground=None,roof_indices=None):
        """Static users; ``ground`` (building-foot reference) is for rooftop users only."""
        if ground is None:
            ground,slope,good = self.city.sample_ground(points)
            if not np.asarray(good).all():
                raise ValueError("static fixture attempted to fill invalid ground")
            ref = "dtm_at_position"
        else:
            ground = np.asarray(ground,dtype=float)
            ref = "building_foot_dtm"
            slope = np.full(len(points),np.nan)
        if labels!="pedestrian":
            slope = np.full(len(points),np.nan)
        n = len(points)
        return Crowd(np.broadcast_to(np.c_[points,z],(len(t),n,3)).copy(),
                     np.broadcast_to(slope,(len(t),n)).copy(),
                     np.broadcast_to(z-ground,(len(t),n)).copy(),np.zeros((len(t),n)),
                     np.broadcast_to(ground,(len(t),n)).copy(),np.asarray(group),
                     np.full(n,labels),
                     self.roof_ref[roof_indices] if roof_indices is not None else np.full(n,ref),
                     self.roof_ids[roof_indices] if roof_indices is not None else np.full(n,None,dtype=object),
                     self.roof_cells[roof_indices] if roof_indices is not None else np.full((n,2),-1,dtype=int))

    def link(self, site, xyz, include_ris=True):
        bs,ris = site["bs_xyz"],site["ris_xyz"]
        vis, blockers = self.city.los_batch(bs,xyz)
        vis, blockers = np.asarray(vis,dtype=bool),np.asarray(blockers,dtype=object)
        label = np.full(len(xyz),NO_LOS if len(self.facets.t) else NO_LOS_UNRESOLVED,dtype="<U32")
        label[vis]=LOS
        absent = np.flatnonzero(~vis)
        if len(self.facets.t) and len(absent):
            label[absent[self.facets.visible(self.city,bs,xyz[absent])]]=NO_LOS_REFLECTION_CANDIDATE
        if not include_ris:
            return label
        ris_los,ris_blocker = self.city.los_batch(ris,xyz)
        return label, blockers, np.asarray(ris_los,dtype=bool), np.asarray(ris_blocker,dtype=object)

    def shadow(self,t,rng,counts,slope_required):
        for _ in range(950):
            site = self.choose_site(rng,counts)
            anchor = self.anchor(rng,counts)
            if anchor is None:
                continue
            cruise = 1.30
            distance = t*cruise
            route = self.streets.walk(anchor,float(distance[-1]+5.01),rng,counts)
            if route is None:
                continue
            xy,_ = Streets.position(route,distance[::10]+2.5)
            z,_,ok = self.city.sample_ground(xy)
            if not np.asarray(ok).all():
                counts["challenge_shadow_ground_rejected"] += 1
                continue
            if slope_required and np.ptp(z)<.25:
                counts["challenge_slope_relief_rejected"] += 1
                continue
            vis,_ = self.city.los_batch(site["bs_xyz"],np.c_[xy,z+1.35])
            if np.all(vis) or not np.any(vis):
                counts["challenge_no_shadow_edge"] += 1
                continue
            group = self.group_on_route(route,distance,np.full(len(t),cruise),3,0,rng,counts,force_ped=True)
            if group is None:
                continue
            vis,_ = self.city.los_batch(site["bs_xyz"],group.xyz[:,0])
            changes = np.flatnonzero(np.asarray(vis)[1:]!=np.asarray(vis)[:-1])+1
            if not len(changes) or (slope_required and np.ptp(group.ground[:,0])<.25):
                counts["challenge_shadow_fine_rejected"] += 1
                continue
            background = self.natural(t,rng,counts)
            if background is not None:
                background.group = background.group+1
            return site,Crowd.join([x for x in (group,background) if x is not None]),int(changes[0])
        return None

    def no_los_group(self,t,rng,counts):
        """Static pedestrians that all lack a clear BS segment and a reflection candidate."""
        for _ in range(30):
            site = self.choose_site(rng,counts)
            idx = rng.choice(len(self.streets.xy),min(256,len(self.streets.xy)),replace=False)
            xy = self.streets.xy[idx]
            z,_,good = self.city.sample_ground(xy)
            good = np.asarray(good,dtype=bool)&self.outdoor(xy)
            xy,z = xy[good],z[good]
            ht = self.phone(rng,len(xy))
            links = self.link(site,np.c_[xy,z+ht],include_ris=False)
            no_los = np.flatnonzero(links==NO_LOS)
            if len(no_los)<4:
                counts["challenge_too_few_no_los"] += 1
                continue
            take = rng.choice(no_los,min(len(no_los),int(rng.integers(8,21))),replace=False)
            group = self.positions_static(xy[take],z[take]+ht[take],"pedestrian",t,rng,np.arange(len(take)))
            return site,group
        return None

    def elevated(self,t,rng,counts):
        if len(self.roofs)<5:
            counts["challenge_no_rooftop_support"] += 1
            return None
        base = self.natural(t,rng,counts)
        pick = rng.choice(len(self.roofs),5,replace=False)
        roof = self.roofs[pick]
        group = self.positions_static(roof[:,:2],roof[:,2]+self.phone(rng,5),"elevated",t,rng,
                                      (0 if base is None else base.group.max()+1)+np.arange(5),
                                      ground=self.roof_ground[pick],roof_indices=pick)
        return Crowd.join([x for x in (base,group) if x is not None])


def times(rng):
    seconds = float(np.clip(rng.lognormal(np.log(10),.22),6.,16.))
    return DT*np.arange(1+round(seconds/DT))


def ris_reachable(ris_los_frames):
    """Users with RIS LOS in at least ``RIS_REACH_MIN_FRACTION`` of the clip frames."""
    return np.asarray(ris_los_frames,dtype=float).mean(axis=0)>=RIS_REACH_MIN_FRACTION


def schedule(reachable,rng,counts=None,force=()):
    """Seeded served-group choice of 2-4 users, preferring RIS-reachable users.

    Each served slot is drawn from users WITHOUT RIS LOS with probability
    ``SERVED_NEGATIVE_SHARE`` (negative cases) and otherwise from RIS-reachable
    users, uniformly within the pool and with no distance filter. An empty pool
    falls back to the other pool (counted). Unserved users stay in the crowd.
    """
    reachable = np.asarray(reachable,dtype=bool)
    n = len(reachable)
    counts = Counter() if counts is None else counts
    out = np.zeros(n,dtype=bool)
    for j in force:
        out[int(j)] = True
    want = min(n,int(rng.integers(2,5)))
    while out.sum()<want:
        negative = rng.random()<SERVED_NEGATIVE_SHARE
        pools = (~reachable&~out,reachable&~out) if negative else (reachable&~out,~reachable&~out)
        pool = np.flatnonzero(pools[0])
        if not len(pool):
            counts["served_slot_pool_fallback"] += 1
            pool = np.flatnonzero(pools[1])
        out[int(rng.choice(pool))] = True
    return out


def angles(origin,xyz):
    delta = np.asarray(origin)-xyz
    horizontal = np.linalg.norm(delta[:,:2],axis=1)
    return (np.linalg.norm(delta,axis=1),
            np.degrees(np.arctan2(delta[:,1],delta[:,0])),
            np.degrees(np.arctan2(delta[:,2],horizontal)))


def user_labels(g,site,crowd,t,served=None):
    f,n,_ = crowd.xyz.shape
    xyz = crowd.xyz.reshape(-1,3)
    # Classify exact positions once where static users are repeated per frame.
    uniq,inverse = np.unique(xyz,axis=0,return_inverse=True)
    inverse = np.ravel(inverse)
    links, blockers, ris_los, ris_blocker = g.link(site,uniq)
    sources = g.city.height_source_by_id

    def strings(values):
        return [None if v is None else str(v) for v in values[inverse]]

    def provenance(values):
        return [None if v is None else str(sources.get(v,sources.get(str(v),"unresolved_height_source"))) for v in values[inverse]]

    d,az,el = angles(site["bs_xyz"],xyz)
    r,raz,rel = angles(site["ris_xyz"],xyz)
    labels = {"user_id":np.tile(np.arange(n,dtype=np.int32),f),"group_id":np.tile(crowd.group,f),
            "frame":np.repeat(np.arange(f,dtype=np.int32),n),"time_s":np.repeat(t,n),
            "x":xyz[:,0],"y":xyz[:,1],"z":xyz[:,2],"ground_z_m":crowd.ground.ravel(),
            "height_agl_m":crowd.agl.ravel(),"slope_deg":crowd.slope.ravel(),"speed_mps":crowd.speed.ravel(),
            "height_class":np.tile(crowd.height,f),"ground_ref":np.tile(crowd.ground_ref,f),
            "bs_link":links[inverse],
            "ris_los":ris_los[inverse],"blocker_id":strings(blockers),
            "blocker_height_source":provenance(blockers),"ris_blocker_id":strings(ris_blocker),
            "ris_blocker_height_source":provenance(ris_blocker),
            "bs_distance_m":d,"bs_az_deg":az,"bs_el_deg":el,
            "ris_distance_m":r,"ris_az_deg":raz,"ris_el_deg":rel}
    if getattr(g.city,"ground_fill",None) is not None:
        labels.update({"ground_building_id":np.tile(crowd.ground_building,f).tolist(),
                       "ground_cell_row":np.tile(crowd.ground_cell[:,0],f),
                       "ground_cell_col":np.tile(crowd.ground_cell[:,1],f)})
    if served is not None:
        labels["served"] = np.tile(served,f)
    return labels
