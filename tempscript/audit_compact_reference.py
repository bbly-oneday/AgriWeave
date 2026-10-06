"""精简参考结果独立审计。

直接读取 SQLite/WKB。速度边界用差分约束的多源 Dijkstra 求最大可达
速度（生产版使用前/后向传播），时间另用闭式公式积分；不导入生产
效率/速度求解函数。覆盖复核显式读取独立封存输入，不自证扫掠。
"""
import argparse,hashlib,heapq,json,math,sqlite3,zlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from pyproj import Transformer,CRS
from shapely import from_wkb,wkt
from shapely.affinity import translate
from shapely.ops import transform,unary_union

ROOT=Path(__file__).resolve().parents[1]
def gpkg(blob):
    if blob is None:return None
    sizes={0:0,1:32,2:48,3:48,4:64};return from_wkb(bytes(blob[8+sizes[(blob[3]>>1)&7]:]))
def close(a,b):return a is not None and b is not None and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-4)

def coverage_area_matches(display_area,metric_area,tolerance_m2):
    # 显示面重投影及 make_valid 的数值面积损失，沿用输入覆盖容差。
    # 主体义务/扫掠面积仍由独立米制输入更严格复算，不放宽主体面积账本。
    return abs(display_area-metric_area)<=max(1e-4,tolerance_m2)

def expected_time(rows,cfg):
    by_chain=defaultdict(list);stops={};previous=None;on=False;progress=0.;chain=0
    def add(row,xy,s,action):
        key=(chain,float(s));e=stops.setdefault(key,dict(chain=chain,start=s,owner=row['sequence'],xy=tuple(xy),actions={}))
        e['actions'][action]=cfg['stops']['stop_steer_seconds' if action=='STOP_STEER' else 'implement_switch_seconds']
    v1_instant=0.
    for row in rows:
        xy=np.asarray(row['geometry'].coords)[:,:2];vectors=np.diff(xy,axis=0);lengths=np.linalg.norm(vectors,axis=1);mask=lengths>1e-8
        if not mask.any():raise ValueError('ZERO_LENGTH_SEGMENT')
        moving=np.vstack([xy[0],xy[1:][mask]]);lengths=lengths[mask];unit=vectors[mask]/lengths[:,None]
        angles=np.degrees(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1,1)))
        working=row['kind'] in ('WORK','HEADLAND')
        if previous is None or row['component']!=previous['component'] or math.dist(previous['xy'],xy[0])>1e-5:
            if previous is not None and on:add(previous,previous['xy'],progress,'IMPLEMENT_OFF')
            chain+=1;progress=0.;on=False
        elif math.degrees(math.acos(float(np.clip(previous['heading']@unit[0],-1,1))))>35:add(row,xy[0],progress,'STOP_STEER')
        if working!=on:add(row,xy[0],progress,'IMPLEMENT_ON' if working else 'IMPLEMENT_OFF')
        cumulative=np.r_[0.,np.cumsum(lengths)]
        for i,angle in enumerate(angles):
            if angle>35:add(row,moving[i+1],progress+cumulative[i+1],'STOP_STEER')
        row['_chain_start']=progress;row['_chain']=chain
        length=float(lengths.sum());high=cfg['speeds']['work_speed_mps' if working else 'transit_speed_mps'];low=min(high,cfg['speeds']['turn_speed_mps'])
        old_cap=(min(cfg['speeds']['work_speed_mps'],cfg['speeds']['turn_speed_mps']) if working else cfg['speeds']['turn_speed_mps']) if np.any(angles>.01) else high
        v1_instant+=length/old_cap
        classification=cfg['speed_classification'];limits=[]
        if classification['model']=='SEGMENT_V1':limits=[(0.,length,old_cap)]
        elif length<classification['window_m']:limits=[(0.,length,min(high,old_cap))]
        else:
            step=classification['sample_step_m'];edges=list(np.arange(0.,length,step))+[length]
            half=classification['window_m']/2
            def point(s):
                s=s%length if row['geometry'].is_ring else min(length,max(0.,s))
                p=row['geometry'].interpolate(s);return np.array([p.x,p.y])
            for a,z in zip(edges,edges[1:]):
                center=(a+z)/2;p=point(center);u=p-point(center-half);v=point(center+half)-p
                norm=float(np.linalg.norm(u)*np.linalg.norm(v))
                theta=math.degrees(math.acos(max(-1.,min(1.,float(u@v)/norm)))) if norm>1e-12 else 0.
                cap=low if theta>=classification['heading_threshold_deg'] else high
                if limits and limits[-1][2]==cap:limits[-1]=(limits[-1][0],z,cap)
                else:limits.append((a,z,cap))
        by_chain[chain].extend(dict(start=progress+a,end=progress+z,cap=cap,sequence=row['sequence'],chain=chain,working=working) for a,z,cap in limits)
        previous=dict(component=row['component'],sequence=row['sequence'],xy=xy[-1],heading=unit[-1]);on=working;progress+=length
    if previous is not None and on:add(previous,previous['xy'],progress,'IMPLEMENT_OFF')
    times={r['sequence']:[0.,0.] for r in rows};phases=[];instant=0.
    for chain,limits in by_chain.items():
        anchors=[e['start'] for e in stops.values() if e['chain']==chain]
        blocks=[]
        for b in limits:
            knots=sorted([b['start'],b['end']]+[s for s in anchors if b['start']+1e-8<s<b['end']-1e-8])
            blocks.extend({**b,'start':a,'end':z} for a,z in zip(knots,knots[1:]))
        n=len(blocks);caps=[b['cap'] for b in blocks];lengths=[b['end']-b['start'] for b in blocks]
        is_stop=lambda s:any(abs(s-x)<=1e-8 for x in anchors)
        bounds=[0.]+[min(caps[i-1],caps[i])**2 for i in range(1,n)]+[0.]
        for i,b in enumerate(blocks):
            if is_stop(b['start']):bounds[i]=0.
            if is_stop(b['end']):bounds[i+1]=0.
        up=[];down=[]
        for i,cap in enumerate(caps):
            before=0. if bounds[i]==0 else caps[i-1];after=0. if bounds[i+1]==0 else caps[i+1]
            up.append((cap-before if cap>before else cap)/cfg['speed_transition']['acceleration_seconds'])
            down.append((cap-after if cap>after else cap)/cfg['speed_transition']['deceleration_seconds'])
        # 所有限速/停车作为多源上界，非负距离权重传播差分约束。
        queue=[(value,i) for i,value in enumerate(bounds)];heapq.heapify(queue)
        while queue:
            value,i=heapq.heappop(queue)
            if value!=bounds[i]:continue
            neighbours=[]
            if i<n:neighbours.append((i+1,2*up[i]*lengths[i]))
            if i>0:neighbours.append((i-1,2*down[i-1]*lengths[i-1]))
            for j,weight in neighbours:
                candidate=value+weight
                if candidate<bounds[j]:bounds[j]=candidate;heapq.heappush(queue,(candidate,j))
        for i,b in enumerate(blocks):
            v0,v1=math.sqrt(bounds[i]),math.sqrt(bounds[i+1]);a,d=up[i],down[i];L=lengths[i]
            peak=min(caps[i],math.sqrt((2*a*d*L+d*v0*v0+a*v1*v1)/(a+d)))
            da=max(0.,(peak*peak-v0*v0)/(2*a));dd=max(0.,(peak*peak-v1*v1)/(2*d));dc=max(0.,L-da-dd)
            duration=(peak-v0)/a+dc/peak+(peak-v1)/d
            times[b['sequence']][0 if b['working'] else 1]+=duration;instant+=L/b['cap']
            cursor=b['start']
            for distance,entry,exit,kind in ((da,v0,peak,'ACCEL'),(dc,peak,peak,'CRUISE'),(dd,peak,v1,'DECEL')):
                if distance>1e-10:
                    phases.append({**b,'start':cursor,'end':cursor+distance,'v0':entry,'v1':exit,'duration':2*distance/(entry+exit),'type':kind});cursor+=distance
    moving=[]
    for p in phases:
        if p['type']=='CRUISE':continue
        if moving and moving[-1]['type']==p['type'] and moving[-1]['chain']==p['chain'] and abs(moving[-1]['end']-p['start'])<1e-8:
            event=moving[-1];event.update(end=p['end'],v1=p['v1']);event['duration']+=p['duration'];event['distance']+=p['end']-p['start']
        else:moving.append(dict(type=p['type'],chain=p['chain'],start=p['start'],end=p['end'],v0=p['v0'],v1=p['v1'],duration=p['duration'],distance=p['end']-p['start'],owner=p['sequence']))
    tw=sum(t[0] for t in times.values());drive=sum(t[1] for t in times.values());stop=sum(max(e['actions'].values(),default=0.) for e in stops.values())
    return dict(t_work_s=tw,t_nonwork_drive_s=drive,t_stop_s=stop,t_break_s=drive+stop,t_total_known_s=tw+drive+stop,
        t_speed_change_extra_s=tw+drive-instant,t_v1_model_difference_s=tw+drive-v1_instant),times,list(stops.values()),moving

def archive_coverage(db,context,fid,check_overlap=False):
    """来源米制 GPKG 逐田读取，覆盖审计不用生产质量派生函数。"""
    required=[];sweeps=[]
    layers=list(db.execute("SELECT table_name,srs_id FROM gpkg_contents WHERE data_type='features'"))
    for name,srs in layers:
        kind='required' if name.startswith('required_main_areas_') else 'sweep' if name.startswith('work_sweeps_') else None
        if kind is None:continue
        source_crs=db.execute('SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=?',(srs,)).fetchone()[0]
        converter=None
        if context['metric_crs']!='LOCAL_METRIC' and CRS.from_user_input(source_crs)!=CRS.from_user_input(context['metric_crs']):converter=Transformer.from_crs(source_crs,context['metric_crs'],always_xy=True)
        for (blob,) in db.execute(f'SELECT geom FROM "{name}" WHERE field_id=?',(fid,)):
            g=gpkg(blob);g=g if converter is None else transform(converter.transform,g);g=translate(g,-context['origin'][0],-context['origin'][1])
            (required if kind=='required' else sweeps).append(g)
    M=unary_union(required);U=unary_union(sweeps);covered=M.intersection(U)
    overlap=[];seen=None
    if check_overlap:
        for g in sweeps:
            g=g.intersection(M)
            if seen is not None:overlap.append(g.intersection(seen))
            seen=g if seen is None else seen.union(g)
    return dict(_missing=M.difference(U),_overlap=unary_union(overlap),body_required_area_m2=M.area,body_covered_area_m2=covered.area,
        body_uncovered_area_m2=M.difference(U).area,body_repeat_excess_area_m2=max(0.,sum(M.intersection(g).area for g in sweeps)-covered.area))


def audit(path,out,*,verify_source=True):
    path=Path(path).resolve();results=[];batch_issues=[];tw=tb=0.;coverage_checked=0;coverage_db=None
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        contract=json.loads(db.execute("SELECT value_json FROM batch_metadata WHERE key='contract'").fetchone()[0]);cfg=contract['efficiency_config']
        archive=Path(contract.get('input_bundle',''))
        if (archive/'swath_results_metric.gpkg').is_file():
            release=archive/'bundle_checksums.json'
            checks=json.loads(release.read_text())
            if hashlib.sha256(release.read_bytes()).hexdigest()!=contract['input_release_sha256']:batch_issues.append('ARCHIVE_RELEASE_HASH')
            metric_file=archive/'swath_results_metric.gpkg'
            digest=hashlib.sha256()
            with metric_file.open('rb') as f:
                for part in iter(lambda:f.read(1024*1024),b''):digest.update(part)
            if digest.hexdigest()!=checks['files']['swath_results_metric.gpkg']['sha256']:batch_issues.append('ARCHIVE_METRIC_HASH')
            coverage_db=sqlite3.connect(f'file:{metric_file.resolve()}?mode=ro',uri=True)
        if contract['schema_version']!='V7_COMPACT_REFERENCE_2':raise ValueError('COMPACT_SCHEMA_REQUIRED')
        if cfg['schema_version']!=2:raise ValueError('SPEED_PROFILE_VERSION_2_REQUIRED')
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':batch_issues.append('SQLITE_INTEGRITY')
        if verify_source:
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}
            if hashes!=contract['source_code_sha256']:batch_issues.append('CURRENT_SOURCE_HASH_MISMATCH')
        ids=[r[0] for r in db.execute('SELECT field_id FROM field_results ORDER BY field_id')]
        manifest=json.loads(db.execute("SELECT value_json FROM batch_metadata WHERE key='input_manifest'").fetchone()[0])
        ordered=[r['field_id'] for r in manifest['fields'] if r['field_id'] in set(ids)]
        if set(ordered)!=set(ids) or hashlib.sha256(json.dumps(ordered,ensure_ascii=False,allow_nan=False).encode()).hexdigest()!=contract['field_ids_sha256']:
            batch_issues.append('INPUT_FIELD_SET_MISMATCH')
        layers=[r[0] for r in db.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'")]
        if not set(layers).issubset({'field_summary','work_regions','route_segments','route_events','coverage_issues'}):batch_issues.append('UNEXPECTED_SPATIAL_LAYER')
        for name in layers:
            unknown=db.execute(f'SELECT COUNT(*) FROM "{name}" s LEFT JOIN field_results f ON s.field_id=f.field_id WHERE f.field_id IS NULL').fetchone()[0]
            if unknown:batch_issues.append('UNKNOWN_FIELD:'+name)
            if db.execute(f'SELECT COUNT(*) FROM "{name}" WHERE geom IS NOT NULL').fetchone()[0]!=db.execute(f'SELECT COUNT(*) FROM "rtree_{name}_geom"').fetchone()[0]:batch_issues.append('SPATIAL_INDEX:'+name)
        for fid in ids:
            issues=[]
            try:
                state=db.execute('SELECT * FROM field_results WHERE field_id=?',(fid,)).fetchone();ctx=json.loads(state['job_json']);detail=json.loads(state['detail_json'])
                summary=db.execute('SELECT * FROM field_summary WHERE field_id=?',(fid,)).fetchone()
                if summary['body_required_area_m2'] is not None:
                    if coverage_db is None:issues.append('COVERAGE_ARCHIVE_MISSING')
                    else:
                        flags=json.loads(summary["quality_flags"] or "[]")
                        coverage=archive_coverage(coverage_db,ctx,fid,check_overlap="UPSTREAM_SEAM_EXCESS_OVERLAP" in flags);coverage_checked+=1
                        expected_missing=coverage.pop("_missing");expected_overlap=coverage.pop("_overlap")
                        for key,value in coverage.items():
                            if not close(value,summary[key]):issues.append('COVERAGE:'+key)
                saved=list(db.execute('SELECT * FROM route_segments WHERE field_id=? ORDER BY sequence',(fid,)));rows=[]
                back=None if ctx['metric_crs']=='LOCAL_METRIC' else Transformer.from_crs('EPSG:4326',ctx['metric_crs'],always_xy=True)
                def local(g):return translate(g if back is None else transform(back.transform,g),-ctx['origin'][0],-ctx['origin'][1])
                if summary['body_required_area_m2'] is not None and coverage_db is not None:
                    present='coverage_issues' in layers
                    evidence=list(db.execute('SELECT * FROM coverage_issues WHERE field_id=?',(fid,))) if present else []
                    for kind,expected in (('BODY_MISSING',expected_missing),('OVERLAP_FOOTPRINT',expected_overlap)):
                        relevant=[r for r in evidence if r['issue_type']==kind]
                        if (kind=='BODY_MISSING' and expected.area>summary['coverage_tolerance_m2']) or (kind=='OVERLAP_FOOTPRINT' and 'UPSTREAM_SEAM_EXCESS_OVERLAP' in flags and expected.area>1e-8):
                            measured=unary_union([local(gpkg(r['geom'])) for r in relevant])
                            if not relevant or measured.symmetric_difference(expected).area>max(1e-4,summary['coverage_tolerance_m2']):issues.append('COVERAGE_EVIDENCE:'+kind)
                        elif relevant:issues.append('UNEXPECTED_COVERAGE_EVIDENCE:'+kind)
                    for r in evidence:
                        if not gpkg(r['geom']).is_valid:issues.append('INVALID_COVERAGE_GEOMETRY')
                        if not coverage_area_matches(local(gpkg(r['geom'])).area,r['area_m2'],r['tolerance_m2']):issues.append('COVERAGE_EVIDENCE_AREA')
                for row in saved:
                    raw=zlib.decompress(row['metric_geometry_blob']);g=from_wkb(raw)
                    if hashlib.sha256(raw).hexdigest()!=row['metric_geometry_sha256']:issues.append('METRIC_HASH')
                    if not local(gpkg(row['geom'])).equals_exact(g,1e-5):issues.append('DISPLAY_PROJECTION')
                    if not local(gpkg(summary['geom'])).buffer(1e-5).covers(g):issues.append('ROUTE_OUTSIDE_FIELD_OR_HOLE')
                    if not gpkg(row['geom']).is_valid:issues.append('INVALID_ROUTE_GEOMETRY')
                    rows.append({**dict(row),'geometry':g})
                if [r['sequence'] for r in rows]!=list(range(1,len(rows)+1)):issues.append('SEQUENCE_GAP_OR_DUPLICATE')
                if Counter(r['task_id'] for r in rows if r['kind']=='WORK')!=Counter(ctx['expected_tasks']):issues.append('BODY_TASK_SET')
                if Counter(r['task_id'] for r in rows if r['kind']=='HEADLAND')!=Counter(ctx['expected_heads']):issues.append('HEADLAND_TASK_SET')
                for i,r in enumerate(rows):
                    if r['kind']=='CONNECTION' and (r['from_task']!=(rows[i-1]['task_id'] if i else 'ENTRY') or r['to_task']!=(rows[i+1]['task_id'] if i+1<len(rows) else 'EXIT')):issues.append('CONNECTION_ORDER')
                    if i and rows[i-1]['component']==r['component'] and math.dist(rows[i-1]['geometry'].coords[-1],r['geometry'].coords[0])>1e-5:issues.append('POSITION_GAP')
                expected,segment_times,stops,moving=expected_time(rows,cfg)
                for key,value in expected.items():
                    if not close(value,summary[key]):issues.append('TIME:'+key)
                for r in rows:
                    values=segment_times[r['sequence']]
                    if not close(values[0],r['t_work_s']) or not close(values[1],r['t_nonwork_drive_s']):issues.append('SEGMENT_TIME')
                    profile=json.loads(json.loads(r['payload_json'])['speed_profile_json'])
                    if profile and (abs(profile[0][0])>1e-5 or abs(profile[-1][1]-r['geometry'].length)>1e-5):issues.append('PROFILE_DISTANCE_COVERAGE')
                    duration=0.
                    for p in profile:
                        a,z,v0,v1,cap,up,down=p[:7];L=z-a
                        if L<=0 or min(v0,v1)<0 or max(v0,v1)>cap+1e-7:issues.append('INVALID_PROFILE_SPEED')
                        if v1*v1-v0*v0>2*up*L+1e-6 or v0*v0-v1*v1>2*down*L+1e-6:issues.append('PROFILE_ACCELERATION')
                        duration+=2*L/(v0+v1)
                    if not close(duration,r['t_work_s']+r['t_nonwork_drive_s']):issues.append('PROFILE_INTEGRAL')
                events=list(db.execute('SELECT * FROM route_events WHERE field_id=? ORDER BY fid',(fid,)))
                stop_rows=[r for r in events if r['event_type']=='STOP']
                if len(stop_rows)!=len(stops):issues.append('STOP_COUNT')
                for source,target in zip(stops,stop_rows):
                    if source['chain']!=target['chain'] or not close(source['start'],target['start_progress_m']) or not close(max(source['actions'].values()),target['duration_s']):issues.append('STOP_EVENT')
                    if math.dist(local(gpkg(target['geom'])).coords[0],source['xy'])>1e-5:issues.append('STOP_POSITION')
                moving_rows=[r for r in events if r['event_type']!='STOP']
                if len(moving_rows)!=len(moving):issues.append('MOVING_EVENT_COUNT')
                by_sequence={r['sequence']:r for r in rows}
                for source,target in zip(moving,moving_rows):
                    if source['type']!=target['event_type'] or source['chain']!=target['chain']:issues.append('MOVING_EVENT_TYPE')
                    owner=by_sequence[source['owner']];point=owner['geometry'].interpolate(source['start']-owner['_chain_start'])
                    if target['owner_sequence']!=source['owner'] or math.dist(local(gpkg(target['geom'])).coords[0],point.coords[0])>1e-5:issues.append('MOVING_EVENT_POSITION')
                    for k,c in (('start','start_progress_m'),('end','end_progress_m'),('duration','duration_s'),('distance','distance_m'),('v0','v_before_mps'),('v1','v_after_mps')):
                        if not close(source[k],target[c]):issues.append('MOVING_EVENT:'+c)
                for kind,prefix in (('ACCEL','accel'),('DECEL','decel')):
                    events_of_type=[e for e in moving if e['type']==kind]
                    if not close(sum(e['duration'] for e in events_of_type),summary['t_'+prefix+'_process_s']) or len(events_of_type)!=summary[prefix+'_count'] or not close(sum(e['distance'] for e in events_of_type),summary[prefix+'_distance_m']):issues.append('SPEED_EVENT_LEDGER:'+kind)
                complete=bool(detail.get('full_reference_connected') and detail.get('export_audit_passed',True) and state['state']=='COMPLETED')
                if complete:
                    ratio=expected['t_work_s']/expected['t_total_known_s']
                    if not close(ratio,summary['efficiency_ratio']):issues.append('EFFICIENCY_RATIO')
                elif summary['efficiency_ratio'] is not None:issues.append('INCOMPLETE_NOT_NULL')
                tw+=expected['t_work_s'];tb+=expected['t_break_s']
            except Exception as exc:issues.append(f'FIELD_AUDIT_ERROR:{type(exc).__name__}:{exc}')
            results.append(dict(field_id=fid,passed=not issues,issues=sorted(set(issues))))
        report=dict(passed=not batch_issues and all(r['passed'] for r in results),field_count=len(ids),
            failed_field_count=sum(not r['passed'] for r in results),batch_issues=batch_issues,checks=results,
            t_work_s=tw,t_break_s=tb,aggregate_efficiency_ratio=tw/(tw+tb) if tw+tb>0 else None,
            independent_method='RAW_METRIC_GEOMETRY_MULTISOURCE_DIJKSTRA_CLOSED_FORM_INTEGRATION',
            source_hash_verified=verify_source,coverage_checked_field_count=coverage_checked,
            scope='ROUTE_TIME_EVENTS_ARCHIVED_BODY_COVERAGE_NOT_PHYSICAL_CERTIFICATION')
    if coverage_db is not None:coverage_db.close()
    Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return {k:v for k,v in report.items() if k!='checks'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpkg',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-historical-source',action='store_true');a=p.parse_args()
    result=audit(a.gpkg,a.out,verify_source=not a.allow_historical_source)
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(not result['passed'])
