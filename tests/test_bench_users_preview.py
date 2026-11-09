from collections import Counter
import unittest
import numpy as np
from atomwise_ris.bench import users


class Hillside:
    street_segments = np.array([[[0.,0.],[30.,0.]]])
    roof_xyz = np.empty((0,3))
    hotspots_xy = np.empty((0,2))
    hotspot_weights = np.empty(0)
    sites = [{"site_id":"hill","bs_xyz":[0,4,25],"ris_xyz":[9,2,7]}]
    height_source_by_id = {}

    def sample_ground(self, xy):
        xy=np.asarray(xy)
        z=100+.2*xy[:,0]
        return z,np.full(len(xy),np.degrees(np.arctan(.2))),np.ones(len(xy),dtype=bool)

    def outdoor(self, xy):
        return self.sample_ground(xy)[2]

    def los_batch(self, tx, rx):
        return np.ones(len(np.atleast_2d(rx)),dtype=bool),np.full(len(np.atleast_2d(rx)),None,dtype=object)


class PreviewTest(unittest.TestCase):
    def test_ground_under_each_user_changes_without_changing_phone_height(self):
        g=users.Generator(Hillside(),"maastricht")
        rng=np.random.default_rng(11)
        t=np.arange(251)*users.DT
        dist,vel=g.travel(t,1.3,np.random.default_rng(12))
        route=(np.array([[0.,0.],[30.,0.]]),np.array([0.,30.]))
        c=g.group_on_route(route,dist,vel,3,0,rng,Counter(),force_ped=True)
        self.assertIsNotNone(c)
        self.assertGreater(np.ptp(c.ground[:,0]),.25)
        np.testing.assert_allclose(c.agl,c.xyz[:,:,2]-c.ground)
        np.testing.assert_allclose(np.ptp(c.agl,axis=0),0,atol=1e-12)
        self.assertGreater(c.speed[:,0].max(),1.3)
        self.assertLess(c.speed.max(),2.)

    def test_missing_ground_is_not_filled(self):
        class Gap(Hillside):
            def sample_ground(self,xy):
                z,s,v=super().sample_ground(xy)
                bad=(np.asarray(xy)[:,0]>4)&(np.asarray(xy)[:,0]<9)
                z[bad]=np.nan
                v[bad]=False
                return z,s,v
        g=users.Generator(Gap(),"maastricht")
        route=(np.array([[0.,0.],[30.,0.]]),np.array([0.,30.]))
        counts=Counter()
        res=g.group_on_route(route,np.linspace(0,10,101),np.ones(101),1,0,np.random.default_rng(3),counts,force_ped=True)
        self.assertIsNone(res)
        self.assertGreater(counts["invalid_centreline_routes"],0)

    def test_one_bounce_requires_both_clear_segments_and_interior_hit(self):
        f=users.Facets(np.array([[[0,-20,-10],[0,20,-10],[0,0,20]]],float))
        points=np.array([[5,0,0],[-1,0,0],[5,0,90]],float)
        hits=f.visible(Hillside(),np.array([5.,0.,0.]),points)
        np.testing.assert_array_equal(hits,[True,False,False])
        class NoBack(Hillside):
            def los_batch(self,tx,rx):
                return np.full(len(np.atleast_2d(rx)),np.asarray(tx).ndim==1,dtype=bool),[None]*len(rx)
        self.assertFalse(f.visible(NoBack(),np.array([5.,0.,0.]),points)[0])

    def test_low_native_roof_does_not_count_as_elevated_support(self):
        class Roofs(Hillside):
            roof_xyz = np.array([[5.,0.,101.4],[5.,0.,105.]])
        g=users.Generator(Roofs(),"maastricht")
        np.testing.assert_allclose(g.roofs,[[5.,0.,105.]])
        self.assertEqual(g.roof_rejections["support_below_3m"],1)

    def test_street_graph_drops_slope_cap_nodes_and_never_fills(self):
        class Ridge(Hillside):
            street_segments = np.array([[[0.,0.],[60.,0.]]])
            def sample_ground(self, xy):
                z,s,v = super().sample_ground(xy)
                s = s.copy()
                s[(np.asarray(xy)[:,0]>28)&(np.asarray(xy)[:,0]<32)] = 40.   # wall / kerb spike
                return z,s,v
        g = users.Generator(Ridge(),"maastricht")
        self.assertGreater(g.street_prune["street_nodes_slope_over_cap"],0)
        x = g.streets.xy[:,0]
        self.assertFalse(((x>28)&(x<32)).any())
        # the ridge splits the street; no graph edge crosses it
        for i,nb in enumerate(g.streets.neighbors):
            for j in nb:
                self.assertFalse((x[i]<28)!=(x[j]<28))

    def test_pedestrian_route_over_steep_cell_is_rejected_not_walked(self):
        class Steep(Hillside):
            def sample_ground(self, xy):
                z,s,v = super().sample_ground(xy)
                s = s.copy()
                s[(np.asarray(xy)[:,0]>14)&(np.asarray(xy)[:,0]<16)] = 30.
                return z,s,v
        g = users.Generator(Steep(),"maastricht")
        route = (np.array([[0.,0.],[30.,0.]]),np.array([0.,30.]))
        counts = Counter()
        res = g.group_on_route(route,np.linspace(0,20,201),np.ones(201),1,0,np.random.default_rng(3),counts,force_ped=True)
        self.assertIsNone(res)
        self.assertGreater(counts["centreline_slope_cap_routes"],0)

    def test_only_pedestrians_carry_slope_labels(self):
        g = users.Generator(Hillside(),"maastricht")
        rng = np.random.default_rng(5)
        t = np.arange(51)*users.DT
        dist,vel = g.travel(t,1.3,np.random.default_rng(6))
        route = (np.array([[0.,0.],[30.,0.]]),np.array([0.,30.]))
        seen = set()
        for k in range(40):
            c = g.group_on_route(route,dist,vel,6,0,rng,Counter())
            if c is None:
                continue
            ped = c.height=="pedestrian"
            seen.update(c.height.tolist())
            self.assertTrue(np.isfinite(c.slope[:,ped]).all())
            self.assertLessEqual(float(c.slope[:,ped].max()),users.WALK_SLOPE_CAP_DEG)
            self.assertTrue(np.isnan(c.slope[:,~ped]).all())
        self.assertIn("pedestrian",seen)

    def test_static_rooftop_users_have_null_slope_and_foot_reference(self):
        g = users.Generator(Hillside(),"maastricht")
        t = np.arange(3)*users.DT
        c = g.positions_static(np.array([[5.,0.],[6.,0.]]),np.array([112.,112.]),"elevated",t,None,
                               np.arange(2),ground=np.array([101.,101.]))
        self.assertTrue(np.isnan(c.slope).all())
        np.testing.assert_allclose(c.agl,11.)
        self.assertEqual(set(c.ground_ref),{"building_foot_dtm"})

    def test_served_group_prefers_ris_reachable_with_declared_negative_share(self):
        reach = np.array([True]*6+[False]*14)
        rng = np.random.default_rng(1)
        counts = Counter()
        hit = total = 0
        for _ in range(4000):
            served = users.schedule(reach,rng,counts)
            self.assertTrue(2<=served.sum()<=4)
            hit += int(reach[served].sum()); total += int(served.sum())
        self.assertAlmostEqual(hit/total,1-users.SERVED_NEGATIVE_SHARE,delta=.03)
        self.assertEqual(counts["served_slot_pool_fallback"],0)
        # no reachable user at all: falls back (counted), never fails
        c2 = Counter()
        served = users.schedule(np.zeros(5,dtype=bool),rng,c2)
        self.assertTrue(2<=served.sum()<=4)
        self.assertGreater(c2["served_slot_pool_fallback"],0)
        # tiny crowd serves everyone present
        self.assertEqual(int(users.schedule(np.array([True]),rng).sum()),1)
        # forced member (shadow-edge user) is always served
        for _ in range(50):
            self.assertTrue(users.schedule(reach,rng,force=(13,))[13])

    def test_ris_reachable_threshold_is_clip_fraction(self):
        frames = np.array([[1,0,1],[1,0,0],[1,0,0],[0,0,1]],dtype=bool)
        np.testing.assert_array_equal(users.ris_reachable(frames),[True,False,True])

    def test_link_names_say_no_los_not_blocked(self):
        class Wall(Hillside):
            def los_batch(self, tx, rx):
                k = len(np.atleast_2d(rx))
                return np.zeros(k,dtype=bool),np.full(k,"b1",dtype=object)
        g = users.Generator(Wall(),"maastricht")
        xyz = np.array([[10.,0.,102.]])
        site = {"bs_xyz":np.array([0.,4.,25.]),"ris_xyz":np.array([9.,2.,7.])}
        self.assertEqual(g.link(site,xyz,include_ris=False).tolist(),[users.NO_LOS_UNRESOLVED])
        g.facets = users.Facets(np.array([[[0,-20,-10],[0,20,-10],[0,0,20]]],float))
        self.assertEqual(g.link(site,xyz,include_ris=False).tolist(),[users.NO_LOS])
        self.assertNotIn("blocked",users.LINK_CLASSES)
        self.assertIn("no_los_reflection_candidate",users.LINK_CLASSES)

    def test_planned_sites_respect_distance_range_los_and_ris_shadow_coverage(self):
        class Scene(Hillside):
            """Two obstacle cylinders; the clear-segment test is exact for them."""
            bounds = np.array([-200.,-200.,200.,200.])
            obstacles = [(np.array([0.,30.]),12.,60.),(np.array([60.,-25.]),15.,60.)]
            street_segments = np.array([[[-150.,0.],[150.,0.]],[[-150.,-60.],[150.,-60.]],[[0.,-60.],[0.,60.]]])
            facade_xyz = np.array([[x,y,8.] for x in (-120,-80,-40,30,70,110) for y in (-20,20,45)],float)
            facade_owner_ids = np.array([f"f{i}" for i in range(18)],dtype=object)
            roof_xyz = np.empty((0,3))
            roof_owner_ids = np.empty(0,dtype=object)
            def sample_ground(self, xy):
                xy = np.asarray(xy)
                return np.zeros(len(xy)),np.zeros(len(xy)),np.ones(len(xy),dtype=bool)
            def outdoor(self, xy):
                return np.ones(len(np.asarray(xy)),dtype=bool)
            def los_batch(self, tx, rx):
                tx = np.atleast_2d(tx); rx = np.atleast_2d(rx)
                tx,rx = np.broadcast_arrays(tx,rx)
                vis = np.ones(len(tx),dtype=bool)
                for i,(a,b) in enumerate(zip(tx,rx)):
                    d = b[:2]-a[:2]; dd = float(d@d)
                    for c,r,zmax in self.obstacles:
                        tt = 0. if dd<1e-9 else float(np.clip((c-a[:2])@d/dd,0,1))
                        if np.linalg.norm(a[:2]+tt*d-c)<r and a[2]+tt*(b[2]-a[2])<zmax:
                            vis[i] = False
                return vis,np.where(vis,None,"cyl").astype(object)
        city = Scene()
        g = users.Generator(city,"maastricht",site_rng=np.random.default_rng(4))
        self.assertGreaterEqual(len(g.sites),1)
        probes_xy = g.streets.xy
        for site in g.sites:
            bs,ris = np.asarray(site["bs_xyz"]),np.asarray(site["ris_xyz"])
            self.assertTrue(users.SITE_DISTANCE_M[0]<=np.linalg.norm(bs-ris)<=users.SITE_DISTANCE_M[1])
            self.assertTrue(city.los_batch(bs,ris[None])[0][0])
            self.assertGreaterEqual(site["ris_covers_bs_shadow_share"],users.SHADOW_MIN_SHARE-1e-9)
            self.assertGreaterEqual(site["bs_los_share"],users.BS_MIN_LOS_SHARE-1e-9)
            self.assertGreaterEqual(site["ris_los_share"],users.RIS_MIN_LOS_SHARE-1e-9)
            self.assertTrue(users.RIS_AGL_M[0]<=site["ris_agl_m"]<=users.RIS_AGL_M[1])
        # reproducible
        g2 = users.Generator(city,"maastricht",site_rng=np.random.default_rng(4))
        self.assertEqual([s["site_id"] for s in g.sites],[s["site_id"] for s in g2.sites])
        np.testing.assert_allclose([s["bs_xyz"] for s in g.sites],[s["bs_xyz"] for s in g2.sites])
        # per-case draw keeps the range and the clear BS-RIS segment
        counts = Counter()
        rng = np.random.default_rng(9)
        for _ in range(60):
            site = g.choose_site(rng,counts)
            d = float(np.linalg.norm(site["bs_xyz"]-site["ris_xyz"]))
            self.assertTrue(users.SITE_DISTANCE_M[0]<=d<=users.SITE_DISTANCE_M[1])
            self.assertAlmostEqual(d,site["bs_ris_distance_m"])
            self.assertTrue(city.los_batch(site["bs_xyz"],site["ris_xyz"][None])[0][0])

    def test_heldout_refused_before_accessing_city(self):
        with self.assertRaisesRegex(ValueError,"held-out"):
            users.Generator(None,"berlin")


if __name__=="__main__":
    unittest.main()
