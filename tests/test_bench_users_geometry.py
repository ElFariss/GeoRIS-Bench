import unittest

import numpy as np

from atomwise_ris.bench.users_geometry import CityMap, _segment_hits_triangles, load_development_maps


class GeometryPrimitiveTests(unittest.TestCase):
    def test_segment_triangle_intersections_are_finite(self):
        tri=np.asarray([[[0,0,0],[1,0,0],[0,1,0]]],float)
        self.assertTrue(_segment_hits_triangles(np.asarray([.2,.2,-1]),np.asarray([.2,.2,1]),tri))
        self.assertFalse(_segment_hits_triangles(np.asarray([1.2,.2,-1]),np.asarray([1.2,.2,1]),tri))

    def test_ground_sampling_keeps_nodata_and_recovers_plane_slope(self):
        try: import rasterio
        except ImportError: self.skipTest("rasterio is available only in the GIS audit environment")
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        rows,cols=np.indices((10,10))
        xx=cols+.5; yy=9.5-rows
        z=xx+2*yy
        z=z.astype(np.float32); z[5,5]=-9999
        mem=MemoryFile()
        ds=mem.open(driver="GTiff",height=10,width=10,count=1,dtype="float32",
                    crs="EPSG:28992",transform=from_origin(0,10,1,1),nodata=-9999)
        ds.write(z,1)
        city=CityMap("test",(0,0,10,10),"EPSG:28992",ds,[],[],[],[],None,{})
        elev,slope,valid=city.sample_ground([[2.5,7.5],[5.5,4.5]])
        self.assertTrue(valid[0]); self.assertFalse(valid[1])
        expected=np.degrees(np.arctan(np.sqrt(5)))
        self.assertAlmostEqual(slope[0],expected,places=4)
        ds.close(); mem.close()

    def test_one_rule_full_4x4_stencil_must_be_valid(self):
        """v1: a point whose 2x2 patch is valid but whose 4x4 ring touches no-data is rejected."""
        try: import rasterio
        except ImportError: self.skipTest("rasterio is available only in the GIS audit environment")
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        rows,cols=np.indices((12,12))
        z=(cols+.5+2*(11.5-rows)).astype(np.float32); z[5,5]=-9999
        mem=MemoryFile()
        ds=mem.open(driver="GTiff",height=12,width=12,count=1,dtype="float32",
                    crs="EPSG:28992",transform=from_origin(0,12,1,1),nodata=-9999)
        ds.write(z,1)
        city=CityMap("test",(0,0,12,12),"EPSG:28992",ds,[],[],[],[],None,{})
        # cell (row 3, col 3)-(4,4): inner 2x2 valid, 4x4 stencil rows 2-5 / cols 2-5 includes (5,5)
        x,y=4.0,8.0             # col coordinate 4.0-.5 -> cell 3; row coordinate 4.0-.5 -> cell 3
        _,_,valid=city.sample_ground([[x,y],[8.5,2.5]])
        self.assertFalse(valid[0]); self.assertTrue(valid[1])
        # same slope plane away from the hole: elevation + slope use the same rule
        elev,slope,valid=city.sample_ground([[8.5,2.5]])
        self.assertAlmostEqual(float(elev[0]),8.5-.0+2*2.5-.0,places=3)
        self.assertAlmostEqual(float(slope[0]),float(np.degrees(np.arctan(np.sqrt(5)))),places=4)
        ds.close(); mem.close()

    def test_loader_refuses_heldout_area_before_io(self):
        with self.assertRaisesRegex(ValueError,"Refusing non-development map"):
            load_development_maps(area_ids=("berlin",))


if __name__=="__main__": unittest.main()
