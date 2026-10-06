"""连续变速解析反例：数值依据闭式公式，不复用生产积分。"""


def _read_config_json(path):
    # 仅适配配置/JSON读取；历史文件仍按原内容，统一配置按显式节选。
    import sys
    from pathlib import Path
    root=next(p for p in Path(__file__).resolve().parents if (p/'src/io_utils.py').is_file())
    if str(root/'src') not in sys.path:sys.path.insert(0,str(root/'src'))
    from io_utils import read_json
    return read_json(path)

import copy,json,math,sys,unittest
from pathlib import Path
from shapely.geometry import LineString
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import calculate_work_time_efficiency as m

CFG=_read_config_json(ROOT/'config.json#efficiency:constant_full')
CFG.update(schema_version=2,speed_classification={'model':'SEGMENT_V1'},speed_transition=dict(
    acceleration_seconds=2.,deceleration_seconds=2.,initial_speed_mps=0.,final_speed_mps=0.,
    transition_time_policy='NOMINAL_FULL_CHANGE',short_distance_policy='REACHABLE_SPEED_ENVELOPE'))

def block(a,z,cap=2.36,sequence=1):return dict(start=a,end=z,cap=cap,sequence=sequence,chain=1,working=True)
def row(sequence,coords,kind='WORK',task='a',**attrs):return dict(sequence=sequence,component=1,kind=kind,task_id=task,geometry=LineString(coords),**attrs)

class SpeedProfileTests(unittest.TestCase):
    def profile(self,blocks,stops=()):return m.reachable_speed_profile(blocks,stops,CFG['speed_transition'])
    def test_long_line_exact_start_and_stop(self):
        p=self.profile([block(0,100)])
        self.assertAlmostEqual(sum(x['seconds'] for x in p),100/2.36+2.,places=10)
        self.assertEqual([x['phase_kind'] for x in p],['ACCEL','CRUISE','DECEL'])
        self.assertAlmostEqual(p[0]['end']-p[0]['start'],2.36)
        self.assertAlmostEqual(p[0]['seconds'],2.)
    def test_short_line_triangular_peak(self):
        length=.5;a=2.36/2;p=self.profile([block(0,length)])
        expected=math.sqrt(a*length)
        self.assertAlmostEqual(max(x['v1'] for x in p),expected)
        self.assertAlmostEqual(sum(x['seconds'] for x in p),2*math.sqrt(length/a))
        self.assertEqual(len(p),2)
    def test_corner_decelerates_before_low_speed_zone(self):
        p=self.profile([block(0,100),block(100,200,.8,2),block(200,300,2.36,3)])
        dec=next(x for x in p if x['sequence']==1 and x['phase_kind']=='DECEL')
        self.assertAlmostEqual(dec['start'],100-3.16)
        self.assertAlmostEqual(dec['seconds'],2.)
        self.assertAlmostEqual(dec['v1'],.8)
        up=next(x for x in p if x['sequence']==3 and x['phase_kind']=='ACCEL')
        self.assertAlmostEqual(up['seconds'],2.)
    def test_equal_speed_artificial_split_invariant(self):
        a=self.profile([block(0,100)])
        b=self.profile([block(0,1),block(1,7,sequence=2),block(7,100,sequence=3)])
        self.assertAlmostEqual(sum(x['seconds'] for x in a),sum(x['seconds'] for x in b),places=10)
    def test_internal_stop_forces_zero_speed(self):
        p=self.profile([block(0,40),block(40,100,sequence=2)],(40,))
        self.assertAlmostEqual(next(x for x in p if x['end']==40)['v1'],0.)
        self.assertAlmostEqual(next(x for x in p if x['start']==40)['v0'],0.)
    def test_many_short_alternating_zones_have_continuity(self):
        p=self.profile([block(i*.2,(i+1)*.2,2.36 if i%2==0 else .8,i+1) for i in range(30)])
        for a,b in zip(p,p[1:]):
            self.assertAlmostEqual(a['end'],b['start']);self.assertAlmostEqual(a['v1'],b['v0'])
        for x in p:self.assertLessEqual(max(x['v0'],x['v1']),x['cap']+1e-10)
        self.assertAlmostEqual(sum(x['end']-x['start'] for x in p),6.)
    def test_invalid_positive_length_at_zero_cap_rejected(self):
        with self.assertRaises(ValueError):self.profile([block(0,100,0)])
    def test_work_time_contains_ramps_and_stops_not_double_counted(self):
        f,s,e=m.compute_geometry_field('x',[row(1,[(0,0),(100,0)])],['a'],CFG,[])
        self.assertAlmostEqual(f['t_work_s'],100/2.36+2.)
        self.assertAlmostEqual(f['t_speed_change_extra_s'],2.)
        self.assertAlmostEqual(f['t_accel_process_s']+f['t_decel_process_s'],4.)
        self.assertAlmostEqual(f['t_total_known_s'],f['t_work_s']+f['t_nonwork_drive_s']+f['t_stop_s'])
        self.assertEqual(f['accel_count'],1);self.assertEqual(f['decel_count'],1)
        self.assertEqual(f['stop_count'],2)
    def test_repeated_sampling_does_not_create_extra_speed_events(self):
        a=m.compute_geometry_field('x',[row(1,[(0,0),(100,0)])],['a'],CFG,[])
        b=m.compute_geometry_field('x',[row(1,[(i,0) for i in range(101)])],['a'],CFG,[])
        self.assertAlmostEqual(a[0]['t_total_known_s'],b[0]['t_total_known_s'])
        self.assertEqual(a[0]['accel_count'],b[0]['accel_count'])
    def test_nonwork_and_work_boundary_ledgers(self):
        rows=[row(1,[(0,0),(20,0)]),row(2,[(20,0),(21,0)],'CONNECTION','',from_task='a',to_task='b'),row(3,[(21,0),(40,0)],task='b')]
        f,s,e=m.compute_geometry_field('x',rows,['a','b'],CFG,[])
        self.assertEqual(f['efficiency_status'],'ESTIMATED')
        self.assertAlmostEqual(f['t_work_s'],sum(x['t_work_s'] for x in s))
        self.assertAlmostEqual(f['t_nonwork_drive_s'],sum(x['t_nonwork_drive_s'] for x in s))
        self.assertTrue(all(math.isfinite(x['duration_s']) for x in e))
    def test_gap_keeps_null_efficiency(self):
        f,_,_=m.compute_geometry_field('x',[row(1,[(0,0),(10,0)]),row(2,[(20,0),(30,0)],task='b')],['a','b'],CFG,[])
        self.assertIsNone(f['efficiency_ratio'])
    def test_same_coordinate_multiple_visits_not_merged(self):
        rows=[row(1,[(0,0),(10,0)]),row(2,[(10,0),(0,0)],task='b'),row(3,[(0,0),(10,0)],task='c')]
        f,_,e=m.compute_geometry_field('x',rows,['a','b','c'],CFG,[])
        stops=[x for x in e if x['event_type']=='STOP']
        self.assertEqual(len({x['event_id'] for x in stops}),len(stops))
        self.assertGreaterEqual(len(stops),4)
    def test_tiny_closed_curve_uses_conservative_low_speed(self):
        cfg=m.load_config(ROOT/'config.json#efficiency:continuous')
        points=[(.15*math.cos(i*math.pi/10),.15*math.sin(i*math.pi/10)) for i in range(21)]
        r=dict(sequence=1,chain=1,kind='HEADLAND',geometry=LineString(points),distance_m=LineString(points).length)
        blocks=m.local_speed_blocks(r,0.,cfg)
        self.assertTrue(all(b['cap']==.8 for b in blocks))
    def test_random_blocks_conserve_distance_and_meet_acceleration(self):
        import random
        rng=random.Random(72)
        for _ in range(40):
            blocks=[];s=0.
            for i in range(15):
                z=s+rng.uniform(.01,20);blocks.append(block(s,z,rng.choice([.8,2.36]),i+1));s=z
            p=self.profile(blocks)
            self.assertAlmostEqual(sum(x['end']-x['start'] for x in p),s,places=7)
            for phase in p:
                length=phase['end']-phase['start']
                self.assertLessEqual(phase['v1']**2-phase['v0']**2,2*phase['accel_bound']*length+1e-8)
                self.assertLessEqual(phase['v0']**2-phase['v1']**2,2*phase['decel_bound']*length+1e-8)
    def test_invalid_new_config_values_rejected(self):
        import tempfile
        parent=ROOT/'outputs/compact_reference_20261006/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            path=Path(tmp)/'bad.json'
            for value in (True,0,-1,float('nan'),float('inf')):
                cfg=copy.deepcopy(CFG);cfg['speed_transition']['acceleration_seconds']=value
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):m.load_config(path)

if __name__=='__main__':unittest.main()
