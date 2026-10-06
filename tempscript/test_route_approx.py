"""Reference geometry regressions; do not certify vehicle motion."""

# 十模块布局：加载旧 API 的薄转发；以下测试/工具逻辑保持原样。
from pathlib import Path as _LayoutPath
import sys as _LayoutSys
_LayoutSys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1] / "src"))
import validator as _layout_compat

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from shapely.geometry import Polygon,box,MultiPolygon,LineString
from route_approx import Navigator,Settings,candidates
from types import SimpleNamespace

class Tests(unittest.TestCase):
    def test_simple_shapes(self):
        shapes=[box(0,0,100,100),Polygon([(0,0),(100,0),(80,100),(20,100)]),Polygon([(0,0),(100,0),(100,30),(30,30),(30,100),(0,100)])]
        for shape in shapes:
            n=Navigator(shape,Settings());g,s=n.connect((10,10),(20,20),(1,0),(-1,0))
            self.assertIsNotNone(g);self.assertTrue(shape.covers(g))
    def test_hole_detour(self):
        p=box(0,0,100,100).difference(box(40,20,60,80));n=Navigator(p,Settings())
        g,s=n.connect((20,50),(80,50));self.assertEqual(s,'GRID_DETOUR');self.assertTrue(p.covers(g));self.assertGreater(g.length,60)
    def test_narrow_fine_grid(self):
        p=box(0,0,20,20).union(box(40,0,60,20)).union(box(19,10.1,41,10.4))
        n=Navigator(p,Settings());g,s=n.connect((10,12),(50,12))
        self.assertIsNotNone(g);self.assertTrue(p.covers(g))
    def test_point_contact(self):
        p=MultiPolygon([box(0,0,10,10),box(10,10,20,20)])
        g,s=Navigator(p,Settings()).connect((5,5),(15,15));self.assertIsNone(g);self.assertEqual(s,'GEOMETRICALLY_DISCONNECTED')
    def test_disconnected(self):
        g,s=Navigator(MultiPolygon([box(0,0,10,10),box(20,0,30,10)]),Settings()).connect((5,5),(25,5));self.assertIsNone(g)
    def test_fragments_once(self):
        ts=[SimpleNamespace(row_index=r,heading_rad=0,task_id=str(i),reference_line=LineString([(x,10+r),(x+5,10+r)])) for i,(r,x) in enumerate([(0,0),(0,10),(1,0),(2,0)])]
        cs=list(candidates(ts));self.assertEqual(len(cs),4)
        for q in cs:self.assertEqual(sorted(t.task_id for t,_ in q),['0','1','2','3'])
    def test_config_validation(self):
        for kw in [dict(grid_m=True),dict(grid_m=0),dict(headland_pass_count=4),dict(max_grid_nodes=False)]:
            with self.assertRaises(ValueError):Settings(**kw)
    def test_search_limited_does_not_cross_hole(self):
        p=box(0,0,100,100).difference(box(40,20,60,80))
        g,reason=Navigator(p,Settings(max_grid_nodes=1)).connect((20,50),(80,50))
        self.assertIsNone(g);self.assertEqual(reason,'SEARCH_LIMITED')
    def test_outside_endpoint(self):
        g,reason=Navigator(box(0,0,10,10),Settings()).connect((-1,5),(5,5))
        self.assertIsNone(g);self.assertEqual(reason,'ENDPOINT_OUTSIDE')
    def test_complete_task_schedule(self):
        from unittest.mock import patch
        from route_approx import solve
        ts=[SimpleNamespace(row_index=i,heading_rad=0,task_id=str(i),region_id='r1',reference_line=LineString([(5,5+i*5),(95,5+i*5)])) for i in range(5)]
        reg=SimpleNamespace(region_id='r1',geometry=box(0,0,100,100))
        job=SimpleNamespace(scene_path='unused',field_id='test',tasks=ts,regions=[reg])
        scene=SimpleNamespace(target=box(0,0,100,100),vehicle=SimpleNamespace(working_width_m=3.75))
        with patch('route_approx.load_scene',return_value=scene):rows,info=solve(job,Settings())
        self.assertEqual(info['reference_route_status'],'COMPLETE_CONNECTED')
        self.assertEqual([r['task_id'] for r in rows if r['kind']=='WORK'],[str(i) for i in range(5)])
        route=[r for r in rows if r['kind']!='HEADLAND']
        for a,b in zip(route,route[1:]):self.assertEqual(a['geometry'].coords[-1],b['geometry'].coords[0])
    def test_mixed_input_directions(self):
        from route_approx import oriented
        ts=[SimpleNamespace(row_index=i,heading_rad=0 if i%2==0 else 3.141592653589793,task_id=str(i),reference_line=LineString([(5,5+i*5),(95,5+i*5)] if i%2==0 else [(95,5+i*5),(5,5+i*5)])) for i in range(4)]
        seq=list(candidates(ts))[0]
        for i,(t,rev) in enumerate(seq):
            coords=oriented(t,rev);self.assertEqual(coords[-1][0]>coords[0][0],i%2==0)
    def test_shared_corner_tasks_disconnected(self):
        from route_approx import solve
        from unittest.mock import patch
        import math
        p1=box(0,0,10,10);p2=box(10,10,20,20)
        ts=[SimpleNamespace(row_index=0,heading_rad=math.pi/4,task_id='a',region_id='r1',reference_line=LineString([(1,1),(10,10)])),SimpleNamespace(row_index=0,heading_rad=math.pi/4,task_id='b',region_id='r2',reference_line=LineString([(10,10),(19,19)]))]
        job=SimpleNamespace(scene_path='unused',field_id='test',tasks=ts,regions=[SimpleNamespace(region_id='r1',geometry=p1),SimpleNamespace(region_id='r2',geometry=p2)])
        scene=SimpleNamespace(target=MultiPolygon([p1,p2]),vehicle=SimpleNamespace(working_width_m=1))
        with patch('route_approx.load_scene',return_value=scene):_,info=solve(job,Settings())
        self.assertEqual(info['reference_route_status'],'GEOMETRICALLY_DISCONNECTED');self.assertEqual(info['route_component_count'],2)
    def test_smooth_endpoint_direction(self):
        import numpy as np
        g,method=Navigator(box(0,0,1100,100),Settings()).connect((1000,5),(10,10),(1,0),(1,0),3.75)
        if method=='SMOOTH_TEMPLATE':
            self.assertGreater(g.coords[1][0]-g.coords[0][0],0)
            self.assertGreater(g.coords[-1][0]-g.coords[-2][0],0)
    def test_execution_groups_around_hole(self):
        from route_approx import execution_blocks,region_sequences
        zone=box(0,0,100,100).difference(box(40,30,60,70))
        ts=[]
        for row,y in enumerate([20,40,60,80]):
            spans=[(5,95)] if y in (20,80) else [(5,40),(60,95)]
            for a,b in spans:ts.append(SimpleNamespace(row_index=row,heading_rad=0,task_id=str(len(ts)),reference_line=LineString([(a,y),(b,y)])))
        blocks=execution_blocks(ts,10,zone)
        self.assertEqual(len(blocks),4)
        for seq in region_sequences(ts,Navigator(zone,Settings()),10):
            self.assertEqual(sorted(t.task_id for t,_ in seq),[str(i) for i in range(6)])
    def test_symmetric_reference_cache(self):
        nav=Navigator(box(0,0,100,100),Settings())
        g,method=nav.connect((10,10),(10,20),(1,0),(-1,0))
        reversed_g,reversed_method=nav.connect((10,20),(10,10),(1,0),(-1,0))
        self.assertEqual(method,reversed_method)
        self.assertEqual(list(reversed(g.coords)),list(reversed_g.coords))
    def test_same_endpoint(self):
        g,s=Navigator(box(0,0,10,10),Settings()).connect((5,5),(5,5));self.assertIsNone(g);self.assertEqual(s,'COINCIDENT')

if __name__=='__main__':unittest.main()
