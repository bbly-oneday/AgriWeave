"""V7正式参考时间效率入口，不运行规划。统一config.json的efficiency模式与vehicle速度控制时间模型。默认读取自包含GPKG的有序route_segments及原始米制WKB，逐田写work_time_efficiency.gpkg；历史reference_operations及主体运动JSON保留兼容读取。

作业时间t_work只计开机具移动，t_break包括关机具移动和静止停顿，效率=t_work/(t_work+t_break)。continuous模式对同一行驶距离积分连续可达速度，名义加减速各2秒，短段限制峰值，首尾零速；变速过程已包含移动时间，不能另加2秒。一次静止停顿内多个动作按最大耗时计，以行程编号和进度区分同坐标的不同访问。缺任务/缺连接时整田效率为NULL，已知小计仍保存；未知外部转场不补造，配置估算不替代实车认证或GNSS实测。"""
from __future__ import annotations

from io_utils import config_section, config_reference

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.affinity import translate
from shapely.geometry import LineString, Point
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import route_planner as route_io

# 只用于数值往返及类别判断，不用于扩大允许作业或通行区域。
POSITION_EPS_M = 1e-5
MOVEMENT_EPS_M = 1e-8
HEADING_EPS_RAD = 1e-6
SPEED_KEYS = {'work_speed_mps', 'turn_speed_mps', 'transit_speed_mps', 'reverse_speed_mps'}
STOP_KEYS = {'gear_change_seconds', 'implement_switch_seconds', 'stop_steer_seconds'}
BODY_LAYERS = ('body_work', 'body_connections')


def sha256(path):
    """流式取摘要，避免为校验再次把大型 GPKG 全部读入内存。"""
    digest = hashlib.sha256()
    with config_reference(path)[0].open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    """只接受已协商的参数，布尔值、NaN 和拼错的键必须拒绝。"""
    config = config_section(path, "efficiency")
    if not isinstance(config, dict):
        raise ValueError('EFFICIENCY_CONFIG_KEYS_MISMATCH')
    version = config.get('schema_version')
    keys = {'schema_version', 'description', 'scope', 'incomplete_route_policy',
            'stop_overlap_policy', 'speeds', 'stops', 'parameter_status'}
    if version == 2:
        keys |= {'speed_transition', 'speed_classification'}
    if set(config) != keys:
        raise ValueError('EFFICIENCY_CONFIG_KEYS_MISMATCH')
    if type(config['schema_version']) is not int or config['schema_version'] not in (1, 2):
        raise ValueError('UNSUPPORTED_EFFICIENCY_CONFIG_VERSION')
    if config['scope'] not in ('BODY_REFERENCE_ONLY','FULL_REFERENCE'):
        raise ValueError('INVALID_EFFICIENCY_POLICY: scope')
    for key, expected in [('incomplete_route_policy', 'NULL_EFFICIENCY_KEEP_KNOWN_TIMES'),
                         ('stop_overlap_policy', 'MAX_AT_SAME_STOP')]:
        if config[key] != expected:
            raise ValueError(f'INVALID_EFFICIENCY_POLICY: {key}')
    for section, expected in [('speeds', SPEED_KEYS), ('stops', STOP_KEYS)]:
        values = config[section]
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f'EFFICIENCY_CONFIG_KEYS_MISMATCH: {section}')
        for name, value in values.items():
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (section == 'speeds' and value == 0)):
                raise ValueError(f'INVALID_EFFICIENCY_NUMBER: {name}')
    if version == 2:
        if config['scope'] != 'FULL_REFERENCE':
            raise ValueError('SPEED_PROFILE_REQUIRES_FULL_REFERENCE')
        transitions=config['speed_transition']
        expected={'acceleration_seconds','deceleration_seconds','initial_speed_mps','final_speed_mps','transition_time_policy','short_distance_policy'}
        if not isinstance(transitions,dict) or set(transitions)!=expected:
            raise ValueError('SPEED_TRANSITION_KEYS_MISMATCH')
        for name in ('acceleration_seconds','deceleration_seconds'):
            value=transitions[name]
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('INVALID_SPEED_TRANSITION_PARAMETER: '+name)
        if any(type(transitions[k]) not in (int,float) or transitions[k]!=0 for k in ('initial_speed_mps','final_speed_mps')):
            raise ValueError('REFERENCE_ENDPOINT_SPEED_MUST_BE_ZERO')
        if transitions['transition_time_policy']!='NOMINAL_FULL_CHANGE' or transitions['short_distance_policy']!='REACHABLE_SPEED_ENVELOPE':
            raise ValueError('INVALID_SPEED_TRANSITION_POLICY')
        classification=config['speed_classification']
        if classification!={'model':'SEGMENT_V1'}:
            if not isinstance(classification,dict) or set(classification)!={'model','window_m','sample_step_m','heading_threshold_deg'} or classification['model']!='LOCAL_ARCLENGTH_V1':
                raise ValueError('INVALID_SPEED_CLASSIFICATION')
            for name in ('window_m','sample_step_m','heading_threshold_deg'):
                value=classification[name]
                if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                    raise ValueError('INVALID_SPEED_CLASSIFICATION_PARAMETER')
            if not .25<=classification['sample_step_m']<=classification['window_m'] or not classification['heading_threshold_deg']<35:
                raise ValueError('INVALID_SPEED_CLASSIFICATION_RANGE')
    return config


def moving_intervals(points, working, speeds):
    """对每个移动区间分类，倒车时间不能在 t_break 中再加一遍。

    末点的 gear 是终点标记，不表示一个新的移动区间。零长度区间不产生
    行驶时间；停顿由事件账本单独计时。转弯按车辆朝向变化判定，先 wrap
    防止 0/2π 的表示差异被误计为转弯。倒车优先用倒车速度。
    """
    if not isinstance(points, list) or any(not isinstance(row, list) or any(type(v) not in (int, float) for v in row) for row in points):
        raise ValueError('INVALID_MOTION_POINTS')
    p = np.asarray(points, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 4 or len(p) < 2 or not np.isfinite(p).all()
            or not np.isin(p[:, 3], [-1, 1]).all()):
        raise ValueError('INVALID_MOTION_POINTS')
    distance = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    yaw = (np.diff(p[:, 2]) + math.pi) % (2 * math.pi) - math.pi
    reverse = p[:-1, 3] < 0
    turn = np.abs(yaw) > HEADING_EPS_RAD
    labels = np.full(len(distance), 'WORK' if working else 'TRANSIT', dtype=object)
    if not working:
        labels[turn] = 'TURN'
    labels[reverse] = 'WORK_REVERSE' if working else 'REVERSE'
    speed_map = {'WORK': speeds['work_speed_mps'], 'WORK_REVERSE': min(speeds['work_speed_mps'], speeds['reverse_speed_mps']),
                 'TURN': speeds['turn_speed_mps'], 'TRANSIT': speeds['transit_speed_mps'], 'REVERSE': speeds['reverse_speed_mps']}
    moving = distance > MOVEMENT_EPS_M
    result = {}
    for kind in speed_map:
        mask = moving & (labels == kind)
        length = float(distance[mask].sum())
        result[kind] = dict(distance_m=length, seconds=length / speed_map[kind], speed_mps=speed_map[kind])
    return p, distance, result


def steering_anchors(points, events):
    """将停车转向事件定位到轨迹里程，而不是仅用 XY 去重。

    源事件没有采样点编号时，利用事件发生顺序匹配。分别求最早和最晚
    合法匹配；两者不一致就无法确定是哪一次访问，交给调用方置空效率。
    同一点的零长度重复采样视作同一里程，不会制造额外停顿。
    """
    p = np.asarray(points)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
    candidates = []
    for event in events:
        if event.get('kind') != 'STOP_STEER':
            if event.get('kind') != 'GEAR_CHANGE':
                raise ValueError('UNKNOWN_MOTION_EVENT')
            continue  # 换挡从完整相邻移动区间重算，不重复使用单段虚拟前后挡位。
        xy = np.asarray([event['x_m'], event['y_m']], dtype=float)
        yaw = float(event['yaw_rad'])
        if not np.isfinite(xy).all() or not math.isfinite(yaw):
            raise ValueError('INVALID_EVENT_POSE')
        mask = (np.linalg.norm(p[:, :2] - xy, axis=1) <= POSITION_EPS_M)
        mask &= np.abs((p[:, 2] - yaw + math.pi) % (2 * math.pi) - math.pi) <= POSITION_EPS_M
        indices = np.flatnonzero(mask)
        by_progress = {}
        for i in indices:
            by_progress.setdefault(round(float(distance[i]) / POSITION_EPS_M), int(i))
        if not by_progress:
            raise ValueError('EVENT_NOT_ON_MOTION')
        candidates.append(sorted(by_progress.values()))
    earliest, latest = [], []
    previous = -1
    for indices in candidates:
        available = [i for i in indices if i >= previous]
        if not available:
            raise ValueError('EVENT_ORDER_MISMATCH')
        previous = min(available); earliest.append(previous)
    following = len(p)
    for indices in reversed(candidates):
        available = [i for i in indices if i <= following]
        if not available:
            raise ValueError('EVENT_ORDER_MISMATCH')
        following = max(available); latest.append(following)
    latest.reverse()
    if earliest != latest:
        raise ValueError('AMBIGUOUS_STOP_VISIT')
    return earliest


def compute_field(field_id, operations, expected_tasks, config):
    """按顺序重放主体；完整性与已知时间分开，不用缺段的时间报高效率。"""
    issues = set()
    summaries, stops = [], {}
    ordered = sorted(operations, key=lambda r: r['sequence'])
    work_ids = [r['task_id'] for r in ordered if r['kind'] == 'WORK']
    if Counter(work_ids) != Counter(expected_tasks):
        issues.add('BODY_TASK_SET_MISMATCH')
    if not work_ids:
        issues.add('NO_BODY_WORK')
    if len({r['sequence'] for r in ordered}) != len(ordered):
        raise ValueError('DUPLICATE_SEQUENCE')
    if any(r['kind'] not in ('WORK', 'CONNECTION') for r in ordered):
        raise ValueError('UNKNOWN_BODY_OPERATION')
    chain, progress = 0, 0.
    previous_end, previous_component = None, None
    previous_gear, previous_on = None, False
    last_segment = None
    missing_motion = 0

    def event(segment, local_xy, at_progress, kind, seconds):
        # 当前连续访问中的同一停点才能合并；绕一圈回来，里程已经不同。
        key = (chain, round(at_progress / POSITION_EPS_M))
        record = stops.setdefault(key, dict(field_id=field_id, stop_id=f'{field_id}:c{chain}:s{key[1]}',
            chain=chain, progress_m=at_progress, owner_sequence=segment['sequence'], local_x_m=float(local_xy[0]), local_y_m=float(local_xy[1]), actions={}))
        record['actions'][kind] = max(record['actions'].get(kind, 0.), seconds)

    def finish_run():
        if previous_end is not None and previous_on and last_segment is not None:
            event(last_segment, previous_end[:2], progress, 'IMPLEMENT_OFF', config['stops']['implement_switch_seconds'])

    for row in ordered:
        base = dict(field_id=field_id, sequence=row['sequence'], component=row['component'],
                    kind=row['kind'], task_id=row.get('task_id', ''), region_id=row.get('region_id', ''),
                    from_task=row.get('from_task', ''), to_task=row.get('to_task', ''),
                    geometry=row['geometry'], metric_scope=config['scope'])
        try:
            raw = row.get('motion_json')
            if not isinstance(raw, str) or not raw:
                raise ValueError('MISSING_MOTION_INFORMATION')
            if row.get('kinematic_status') != 'PASS':
                issues.add('SOURCE_MOTION_NOT_ACCEPTED')
            p, distance, breakdown = moving_intervals(json.loads(raw), row['kind'] == 'WORK', config['speeds'])
            if row.get('_local_geometry') is not None:
                xy = np.asarray(row['_local_geometry'].coords)
                if xy.shape != p[:, :2].shape or np.max(np.linalg.norm(xy - p[:, :2], axis=1), initial=0.) > POSITION_EPS_M:
                    raise ValueError('MOTION_GEOMETRY_MISMATCH')
        except (ValueError, TypeError, KeyError) as exc:
            issues.add(str(exc));missing_motion += 1
            finish_run();previous_end = None;previous_on = False;previous_gear = None
            base.update(motion_data_status='MISSING_OR_INVALID', distance_m=row.get('_known_length_m'),
                        t_work_s=None, t_nonwork_drive_s=None, breakdown_json='{}', t_stop_allocated_s=0.)
            summaries.append(base)
            continue
        working = row['kind'] == 'WORK'
        continuous = (previous_end is not None and row['component'] == previous_component
            and np.linalg.norm(p[0, :2] - previous_end[:2]) <= POSITION_EPS_M
            and abs((p[0, 2] - previous_end[2] + math.pi) % (2 * math.pi) - math.pi) <= .025)
        if not continuous:
            if previous_end is not None:
                issues.add('BODY_CONNECTION_MISSING_OR_POSE_GAP')
            finish_run();chain += 1;progress = 0.;previous_gear = None;previous_on = False
        cumulative = np.r_[0., np.cumsum(distance)]
        if previous_on != working:
            event(base, p[0, :2], progress, 'IMPLEMENT_ON' if working else 'IMPLEMENT_OFF', config['stops']['implement_switch_seconds'])
        # 每个挡位用于一个非零移动区间；跨段边界连续重放，不能各段强加前进挡。
        for i in np.flatnonzero(distance > MOVEMENT_EPS_M):
            gear = int(p[i, 3])
            if previous_gear is not None and gear != previous_gear:
                event(base, p[i, :2], progress + cumulative[i], 'GEAR_CHANGE', config['stops']['gear_change_seconds'])
            previous_gear = gear
        try:
            events = json.loads(row.get('events_json') or '[]')
            if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
                raise ValueError('INVALID_EVENT_LIST')
            for index in steering_anchors(p, events):
                event(base, p[index, :2], progress + cumulative[index], 'STOP_STEER', config['stops']['stop_steer_seconds'])
        except (ValueError, TypeError, KeyError) as exc:
            issues.add(str(exc))
        work_time = breakdown['WORK']['seconds'] + breakdown['WORK_REVERSE']['seconds']
        off_time = sum(breakdown[k]['seconds'] for k in ('TURN', 'TRANSIT', 'REVERSE'))
        base.update(motion_data_status='AVAILABLE', distance_m=float(distance.sum()),
            t_work_s=work_time, t_nonwork_drive_s=off_time, breakdown_json=json.dumps(breakdown),
            reverse_distance_m=breakdown['REVERSE']['distance_m']+breakdown['WORK_REVERSE']['distance_m'],
            t_stop_allocated_s=0., chain=chain)
        summaries.append(base)
        progress += float(distance.sum());previous_end = p[-1];previous_component = row['component']
        previous_on = working;last_segment = base
    finish_run()
    # 连接必须确实连接所声明的相邻主体任务，不能用“有一条连接线”代替。
    for i, row in enumerate(ordered):
        if row['kind'] == 'CONNECTION':
            if (i == 0 or i == len(ordered)-1 or ordered[i-1]['kind'] != 'WORK'
                    or ordered[i+1]['kind'] != 'WORK' or row.get('from_task') != ordered[i-1]['task_id']
                    or row.get('to_task') != ordered[i+1]['task_id']):
                issues.add('CONNECTION_TASK_ORDER_MISMATCH')
    by_sequence = {r['sequence']: r for r in summaries}
    counts = Counter();nominal = Counter()
    for record in stops.values():
        actions = record.pop('actions')
        duration = max(actions.values(), default=0.)
        record.update(duration_s=duration, unmerged_duration_s=sum(actions.values()),
                      actions_json=json.dumps(actions), overlap_policy=config['stop_overlap_policy'])
        for kind, seconds in actions.items():
            counts[kind] += 1;nominal[kind] += seconds
        by_sequence[record['owner_sequence']]['t_stop_allocated_s'] += duration
    totals = {kind: dict(distance_m=0., seconds=0.) for kind in ('WORK', 'WORK_REVERSE', 'TURN', 'TRANSIT', 'REVERSE')}
    for segment in summaries:
        for kind, item in json.loads(segment['breakdown_json']).items():
            for key in ('distance_m', 'seconds'):
                totals[kind][key] += item[key]
    t_work = totals['WORK']['seconds'] + totals['WORK_REVERSE']['seconds']
    t_drive_off = sum(totals[k]['seconds'] for k in ('TURN', 'TRANSIT', 'REVERSE'))
    t_stop = sum(r['duration_s'] for r in stops.values())
    total = t_work + t_drive_off + t_stop
    if total <= 0:
        issues.add('NO_KNOWN_TIME')
    complete = not issues
    summary = dict(field_id=field_id, metric_scope=config['scope'], efficiency_status='ESTIMATED' if complete else 'INCOMPLETE',
        body_complete=complete, missing_reason=';'.join(sorted(issues)),
        t_work_s=t_work, t_break_s=t_drive_off+t_stop, t_total_known_s=total,
        efficiency_ratio=t_work/total if complete else None,
        efficiency_pct=100*t_work/total if complete else None,
        t_nonwork_drive_s=t_drive_off, t_turn_s=totals['TURN']['seconds'], t_transit_s=totals['TRANSIT']['seconds'],
        t_reverse_nonwork_s=totals['REVERSE']['seconds'], t_reverse_work_s=totals['WORK_REVERSE']['seconds'], t_stop_s=t_stop,
        work_distance_m=totals['WORK']['distance_m']+totals['WORK_REVERSE']['distance_m'],
        turn_distance_m=totals['TURN']['distance_m'], transit_distance_m=totals['TRANSIT']['distance_m'],
        reverse_distance_m=totals['REVERSE']['distance_m']+totals['WORK_REVERSE']['distance_m'],
        body_segment_count=len(summaries), work_task_count=len(work_ids), expected_task_count=len(expected_tasks),
        connection_count=sum(r['kind']=='CONNECTION' for r in ordered), body_run_count=chain,
        missing_motion_segment_count=missing_motion, accounting_completeness='COMPLETE' if complete else 'KNOWN_PART_ONLY', stop_count=len(stops),
        gear_change_count=counts['GEAR_CHANGE'], stop_steer_count=counts['STOP_STEER'],
        implement_switch_count=counts['IMPLEMENT_ON']+counts['IMPLEMENT_OFF'],
        gear_change_nominal_s=nominal['GEAR_CHANGE'], stop_steer_nominal_s=nominal['STOP_STEER'],
        implement_switch_nominal_s=nominal['IMPLEMENT_ON']+nominal['IMPLEMENT_OFF'],
        stop_overlap_saved_s=sum(r['unmerged_duration_s']-r['duration_s'] for r in stops.values()),
        physical_vehicle_certification='NOT_EVALUATED', time_source='CONFIG_ESTIMATE_NOT_FIELD_MEASURED',
        **config['speeds'], **config['stops'])
    return summary, summaries, list(stops.values())


def compute_geometry_field(field_id, operations, expected_tasks, config, expected_heads):
    """几何参考账本：不要求实车运动数据，也不把倒序条带当倒车。

    每段使用局部米制几何。曲线作业取作业/转弯速度较小者；曲线连接
    取转弯速度，直线连接取空驶速度。超过 35 度的离散折角另计停车
    调整，避免将尖角当作无时间的瞬间转向。同一次停顿动作取最大值。
    """
    if config.get('schema_version') == 2:
        return compute_speed_profile_field(field_id,operations,expected_tasks,config,expected_heads)
    rows=sorted(operations,key=lambda r:r['sequence']);issues=set();segments=[];stops={}
    if len({r['sequence'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_SEQUENCE')
    if Counter(r['task_id'] for r in rows if r['kind']=='WORK')!=Counter(expected_tasks):issues.add('BODY_TASK_SET_MISMATCH')
    if Counter(r['task_id'] for r in rows if r['kind']=='HEADLAND')!=Counter(expected_heads):issues.add('HEADLAND_TASK_SET_MISMATCH')
    if not expected_tasks:issues.add('NO_BODY_WORK')
    for i,r in enumerate(rows):
        if r['kind']=='CONNECTION':
            a=rows[i-1]['task_id'] if i else 'ENTRY';b=rows[i+1]['task_id'] if i+1<len(rows) else 'EXIT'
            if r.get('from_task')!=a or r.get('to_task')!=b:issues.add('CONNECTION_TASK_ORDER_MISMATCH')
    previous=None;on=False;progress=0.;chain=0
    def event(row,xy,at,kind):
        key=(chain,float(at) if config.get('_stable_stop_anchors') else round(at/POSITION_EPS_M));e=stops.setdefault(key,dict(field_id=field_id,stop_id=f'{field_id}:c{chain}:s{key[1]}',chain=chain,progress_m=at,owner_sequence=row['sequence'],local_x_m=float(xy[0]),local_y_m=float(xy[1]),actions={}))
        e['actions'][kind]=config['stops']['stop_steer_seconds' if kind=='STOP_STEER' else 'implement_switch_seconds']
    for row in rows:
        if row['kind'] not in ('WORK','HEADLAND','CONNECTION'):raise ValueError('UNKNOWN_REFERENCE_OPERATION')
        g=row.get('_local_geometry',row['geometry']);xy=np.asarray(g.coords);d=np.diff(xy,axis=0);ds=np.linalg.norm(d,axis=1);mask=ds>MOVEMENT_EPS_M
        if not mask.any():issues.add('ZERO_LENGTH_REFERENCE_SEGMENT');continue
        d=d[mask]/ds[mask,None];angles=np.degrees(np.arccos(np.clip(np.sum(d[:-1]*d[1:],axis=1),-1,1)))
        working=row['kind'] in ('WORK','HEADLAND');curve=bool(np.any(angles>0.01))
        if previous is None or previous['component']!=row['component'] or math.dist(previous['_end'],xy[0])>POSITION_EPS_M:
            if previous is not None:
                issues.add('REFERENCE_CONNECTION_MISSING_OR_POSITION_GAP')
                if on:event(previous,previous['_end'],progress,'IMPLEMENT_OFF')
            chain+=1;progress=0.;on=False
        elif math.degrees(math.acos(float(np.clip(previous['_direction']@d[0],-1,1))))>35:
            event(row,xy[0],progress,'STOP_STEER')
        if on!=working:event(row,xy[0],progress,'IMPLEMENT_ON' if working else 'IMPLEMENT_OFF')
        cumulative=np.r_[0.,np.cumsum(ds[mask])];moving_xy=np.vstack([xy[0],xy[1:][mask]])
        for i,angle in enumerate(angles):
            if angle>35:event(row,moving_xy[i+1],progress+cumulative[i+1],'STOP_STEER')
        speed=(min(config['speeds']['work_speed_mps'],config['speeds']['turn_speed_mps']) if curve else config['speeds']['work_speed_mps']) if working else config['speeds']['turn_speed_mps'] if curve else config['speeds']['transit_speed_mps']
        category='WORK_CURVE' if working and curve else 'WORK' if working else 'TURN' if curve else 'TRANSIT'
        length=float(ds.sum());seconds=length/speed
        base={k:row.get(k,'') for k in ('sequence','component','kind','task_id','region_id','from_task','to_task','phase')}
        base.update(field_id=field_id,geometry=row['geometry'],metric_scope=config['scope'],motion_data_status='GEOMETRY_REFERENCE',distance_m=length,t_work_s=seconds if working else 0.,t_nonwork_drive_s=0. if working else seconds,t_stop_allocated_s=0.,breakdown_json=json.dumps({category:dict(distance_m=length,seconds=seconds,speed_mps=speed)}),chain=chain,reverse_distance_m=0.,time_model='FORWARD_GEOMETRY_REFERENCE')
        segments.append(base);progress+=length;on=working;previous={**row,'_end':xy[-1],'_direction':d[-1]}
    if previous is not None and on:event(previous,previous['_end'],progress,'IMPLEMENT_OFF')
    counts=Counter();nominal=Counter();by_seq={r['sequence']:r for r in segments}
    for e in stops.values():
        actions=e.pop('actions');duration=max(actions.values(),default=0.)
        e.update(duration_s=duration,unmerged_duration_s=sum(actions.values()),actions_json=json.dumps(actions),overlap_policy=config['stop_overlap_policy'])
        by_seq[e['owner_sequence']]['t_stop_allocated_s']+=duration
        for kind,seconds in actions.items():counts[kind]+=1;nominal[kind]+=seconds
    tw=sum(r['t_work_s'] for r in segments);drive=sum(r['t_nonwork_drive_s'] for r in segments);stop=sum(e['duration_s'] for e in stops.values());total=tw+drive+stop
    if total<=0:issues.add('NO_KNOWN_TIME')
    complete=not issues
    result=dict(field_id=field_id,metric_scope=config['scope'],efficiency_status='ESTIMATED' if complete else 'INCOMPLETE',body_complete=complete,reference_complete=complete,missing_reason=';'.join(sorted(issues)),t_work_s=tw,t_break_s=drive+stop,t_total_known_s=total,efficiency_ratio=tw/total if complete else None,efficiency_pct=100*tw/total if complete else None,t_nonwork_drive_s=drive,t_stop_s=stop,t_turn_s=sum(r['t_nonwork_drive_s'] for r in segments if 'TURN' in json.loads(r['breakdown_json'])),t_transit_s=sum(r['t_nonwork_drive_s'] for r in segments if 'TRANSIT' in json.loads(r['breakdown_json'])),work_distance_m=sum(r['distance_m'] for r in segments if r['kind']!='CONNECTION'),connection_distance_m=sum(r['distance_m'] for r in segments if r['kind']=='CONNECTION'),headland_work_time_s=sum(r['t_work_s'] for r in segments if r['kind']=='HEADLAND'),headland_task_count=len(expected_heads),work_task_count=len(expected_tasks),expected_task_count=len(expected_tasks),body_segment_count=len(segments),connection_count=sum(r['kind']=='CONNECTION' for r in rows),body_run_count=chain,stop_count=len(stops),gear_change_count=0,stop_steer_count=counts['STOP_STEER'],implement_switch_count=counts['IMPLEMENT_ON']+counts['IMPLEMENT_OFF'],stop_overlap_saved_s=sum(e['unmerged_duration_s']-e['duration_s'] for e in stops.values()),accounting_completeness='COMPLETE' if complete else 'KNOWN_PART_ONLY',physical_vehicle_certification='NOT_EVALUATED',time_source='GEOMETRY_CONFIG_ESTIMATE_NOT_FIELD_MEASURED',gear_model_status='ASSUMED_FORWARD_REFERENCE',**config['speeds'],**config['stops'])
    return result,segments,list(stops.values())


def local_speed_blocks(row, start, config):
    """固定米制弧长窗口识别直/曲段，避免顶点密度决定速度。

    仅划分时间模型的限速区间，几何不简化。不确定或很短的几何
    沿用旧保守分类；转角停车/驾驶质量仍由原规则单独报告。
    """
    classification=config['speed_classification'];working=row['kind'] in ('WORK','HEADLAND')
    high=config['speeds']['work_speed_mps' if working else 'transit_speed_mps']
    low=min(high,config['speeds']['turn_speed_mps'])
    if classification['model']=='SEGMENT_V1':
        cap=next(iter(json.loads(row['breakdown_json']).values()))['speed_mps']
        return [dict(start=start,end=start+row['distance_m'],cap=cap,sequence=row['sequence'],chain=row['chain'],working=working,
            category='WORK_CURVE' if working and cap<high else 'WORK' if working else 'TURN' if cap<high else 'TRANSIT')]
    xy=np.asarray(row['geometry'].coords)[:,:2]
    ds=np.linalg.norm(np.diff(xy,axis=0),axis=1);mask=ds>MOVEMENT_EPS_M
    xy=np.vstack([xy[0],xy[1:][mask]]);progress=np.r_[0.,np.cumsum(ds[mask])];length=float(progress[-1])
    if length<classification['window_m']:
        # 窗口跨越整个短环时可能采到相同位置，不能据此判成直线。
        unit=np.diff(xy,axis=0)/ds[mask,None]
        angles=np.degrees(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1,1)))
        curved=bool(np.any(angles>.01));cap=low if curved else high
        return [dict(start=start,end=start+length,cap=cap,sequence=row['sequence'],chain=row['chain'],working=working,
            category='WORK_CURVE' if working and curved else 'WORK' if working else 'TURN' if curved else 'TRANSIT')]
    edges=np.r_[np.arange(0.,length,classification['sample_step_m']),length]
    center=(edges[:-1]+edges[1:])/2;half=classification['window_m']/2
    def sample(s):
        s=np.mod(s,length) if row['geometry'].is_ring else np.clip(s,0.,length)
        return np.column_stack([np.interp(s,progress,xy[:,0]),np.interp(s,progress,xy[:,1])])
    point=sample(center);before=point-sample(center-half);after=sample(center+half)-point
    norm=np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1)
    dot=np.sum(before*after,axis=1)
    cosine=np.divide(dot,norm,out=np.ones_like(dot),where=norm>1e-12)
    curved=np.degrees(np.arccos(np.clip(cosine,-1,1)))>=classification['heading_threshold_deg']
    blocks=[]
    for a,z,is_curve in zip(edges[:-1],edges[1:],curved):
        cap=low if is_curve else high
        category='WORK_CURVE' if working and is_curve else 'WORK' if working else 'TURN' if is_curve else 'TRANSIT'
        if blocks and blocks[-1]['cap']==cap and blocks[-1]['category']==category:
            blocks[-1]['end']=start+float(z)
        else:blocks.append(dict(start=start+float(a),end=start+float(z),cap=cap,sequence=row['sequence'],chain=row['chain'],working=working,category=category))
    return blocks


def reachable_speed_profile(blocks, stop_positions, transition):
    """速度平方的前向可达/后向制动包络，短段自动形成三角峰值。

    blocks 每项为 start/end/cap/sequence/chain。两秒是完整转换的名义
    时间，先由目标速度差推导加速度，再按实际可达距离求速度，不能
    把所有段都强行跑到限速或在已有时间上加两秒。
    """
    if not blocks:return []
    caps=[b['cap'] for b in blocks];n=len(blocks)
    lengths=[b['end']-b['start'] for b in blocks]
    if any(not math.isfinite(v) or v<=0 for v in caps+lengths):
        raise ValueError('INVALID_SPEED_PROFILE_BLOCK')
    stops=set(stop_positions)
    at_stop=lambda x:any(abs(x-s)<=1e-8 for s in stops)
    velocities=[0.]+[min(caps[i-1],caps[i]) for i in range(1,n)]+[0.]
    for i,b in enumerate(blocks):
        if at_stop(b['start']):velocities[i]=0.
        if at_stop(b['end']):velocities[i+1]=0.
    accel=[];decel=[]
    for i,cap in enumerate(caps):
        before=0. if velocities[i]==0 else (caps[i-1] if i else 0.)
        after=0. if velocities[i+1]==0 else (caps[i+1] if i+1<n else 0.)
        accel.append((cap-before if cap>before else cap)/transition['acceleration_seconds'])
        decel.append((cap-after if cap>after else cap)/transition['deceleration_seconds'])
    for i in range(n):
        velocities[i+1]=min(velocities[i+1],math.sqrt(velocities[i]**2+2*accel[i]*lengths[i]))
    for i in range(n-1,-1,-1):
        velocities[i]=min(velocities[i],math.sqrt(velocities[i+1]**2+2*decel[i]*lengths[i]))
    phases=[]
    for i,block in enumerate(blocks):
        a,d=accel[i],decel[i];v0,v1=velocities[i:i+2];length=lengths[i]
        peak=min(caps[i],math.sqrt(max(0.,(2*a*d*length+d*v0*v0+a*v1*v1)/(a+d))))
        if peak+1e-8<max(v0,v1):raise ValueError('UNREACHABLE_PROFILE_BOUNDARY')
        peak=max(peak,v0,v1)
        up=max(0.,(peak*peak-v0*v0)/(2*a));down=max(0.,(peak*peak-v1*v1)/(2*d))
        cruise=length-up-down
        if cruise < -1e-7:raise ValueError('OVERLAPPING_SPEED_RAMPS')
        cursor=block['start']
        for distance,entry,exit,kind in ((up,v0,peak,'ACCEL'),(max(0.,cruise),peak,peak,'CRUISE'),(down,peak,v1,'DECEL')):
            if distance<=1e-10:continue
            if entry+exit<=0:raise ValueError('ZERO_SPEED_MOVING_INTERVAL')
            seconds=2*distance/(entry+exit)
            phases.append({**block,'start':cursor,'end':cursor+distance,'v0':entry,'v1':exit,
                'seconds':seconds,'phase_kind':kind,'accel_bound':a,'decel_bound':d})
            cursor+=distance
    return phases


def compute_speed_profile_field(field_id, operations, expected_tasks, config, expected_heads):
    """连续变速只改变时间，参考几何/任务不变，非完整行程仍为 NULL。

    停顿锚点沿用既有驾驶/机具动作假设，但使用原始米制进度而非浮点
    取整身份。移动阶段跨任务边界时按 implement 状态分摊，不重复计时。
    """
    legacy={**config,'schema_version':1,'_stable_stop_anchors':True}
    baseline,segments,stops=compute_geometry_field(field_id,operations,expected_tasks,legacy,expected_heads)
    by_chain={};position={};by_seq={r['sequence']:r for r in segments}
    for row in segments:
        chain=row['chain'];start=position.get(chain,0.);end=start+row['distance_m']
        row['_chain_start']=start
        by_chain.setdefault(chain,[]).extend(local_speed_blocks(row,start,config))
        position[chain]=end
    stop_positions={}
    for index,event in enumerate(stops):
        event.update(event_id=f'{field_id}:stop:{event["chain"]}:{event["owner_sequence"]}:{index}',
            event_type='STOP',reason=event['actions_json'],start_progress_m=event['progress_m'],
            end_progress_m=event['progress_m'],distance_m=0.,v_before_mps=0.,v_after_mps=0.)
        stop_positions.setdefault(event['chain'],[]).append(event['progress_m'])
    phases=[];instant_time=0.
    for chain,rows in by_chain.items():
        anchors=sorted(set(stop_positions.get(chain,[])))
        blocks=[]
        for b in rows:
            cuts=[b['start'],*[p for p in anchors if b['start']+1e-8<p<b['end']-1e-8],b['end']]
            blocks.extend({**b,'start':a,'end':z} for a,z in zip(cuts,cuts[1:]) if z>a)
        instant_time+=sum((b['end']-b['start'])/b['cap'] for b in blocks)
        phases.extend(reachable_speed_profile(blocks,anchors,config['speed_transition']))
    for row in segments:
        row.update(t_work_s=0.,t_nonwork_drive_s=0.,speed_profile_json='[]',
            time_model='GEOMETRY_SPEED_PROFILE_V2',entry_speed_mps=None,exit_speed_mps=None,peak_speed_mps=0.)
    profiles={r['sequence']:[] for r in segments};breakdowns={r['sequence']:{} for r in segments};moving_events=[]
    for p in phases:
        row=by_seq[p['sequence']];start=row['_chain_start'];duration=p['seconds']
        row['t_work_s' if p['working'] else 't_nonwork_drive_s']+=duration
        if row['entry_speed_mps'] is None:row['entry_speed_mps']=p['v0']
        row['exit_speed_mps']=p['v1'];row['peak_speed_mps']=max(row['peak_speed_mps'],p['v0'],p['v1'])
        profiles[p['sequence']].append([p['start']-start,p['end']-start,p['v0'],p['v1'],p['cap'],p['accel_bound'],p['decel_bound'],p['category']])
        category=breakdowns[p['sequence']].setdefault(p['category'],dict(distance_m=0.,seconds=0.,speed_limit_mps=p['cap']))
        category['distance_m']+=p['end']-p['start'];category['seconds']+=duration
        if p['phase_kind']=='CRUISE':continue
        if moving_events and moving_events[-1]['event_type']==p['phase_kind'] and moving_events[-1]['chain']==p['chain'] and abs(moving_events[-1]['end_progress_m']-p['start'])<1e-8:
            event=moving_events[-1]
            # 同一类型在真实停顿点两侧不能合并；两侧至少会切换增减速类型。
            event['duration_s']+=duration;event['distance_m']+=p['end']-p['start']
            event['end_progress_m']=p['end'];event['v_after_mps']=p['v1'];event['end_segment_id']=f'{field_id}:segment:{p["sequence"]}'
        else:
            point=row['geometry'].interpolate(p['start']-start)
            event=dict(field_id=field_id,event_id=f'{field_id}:speed:{p["chain"]}:{len(moving_events)}',
                event_type=p['phase_kind'],reason='REACHABLE_SPEED_ENVELOPE',chain=p['chain'],
                owner_sequence=p['sequence'],start_segment_id=f'{field_id}:segment:{p["sequence"]}',
                end_segment_id=f'{field_id}:segment:{p["sequence"]}',start_progress_m=p['start'],
                end_progress_m=p['end'],v_before_mps=p['v0'],v_after_mps=p['v1'],duration_s=duration,
                distance_m=p['end']-p['start'],local_x_m=point.x,local_y_m=point.y,
                t_work_s=0.,t_nonwork_drive_s=0.)
            moving_events.append(event)
        event['t_work_s' if p['working'] else 't_nonwork_drive_s']+=duration
    for row in segments:
        row['speed_profile_json']=json.dumps(profiles[row['sequence']],allow_nan=False)
        row['breakdown_json']=json.dumps(breakdowns[row['sequence']],allow_nan=False)
        row.pop('_chain_start',None)
    tw=sum(r['t_work_s'] for r in segments);drive=sum(r['t_nonwork_drive_s'] for r in segments);stop=baseline['t_stop_s']
    total=tw+drive+stop
    if not all(math.isfinite(x) and x>=0 for x in (tw,drive,stop,total)):raise ValueError('INVALID_TIME_LEDGER')
    v1_moving=baseline['t_work_s']+baseline['t_nonwork_drive_s']
    baseline.update(t_work_s=tw,t_nonwork_drive_s=drive,t_break_s=drive+stop,t_total_known_s=total,
        efficiency_ratio=tw/total if baseline['efficiency_status']=='ESTIMATED' and total>0 else None,
        efficiency_pct=100*tw/total if baseline['efficiency_status']=='ESTIMATED' and total>0 else None,
        time_model_id='GEOMETRY_SPEED_PROFILE_V2'+('_LOCAL1' if config['speed_classification']['model']=='LOCAL_ARCLENGTH_V1' else ''),time_source='GEOMETRY_CONFIG_ESTIMATE_NOT_FIELD_MEASURED',
        t_speed_change_extra_s=tw+drive-instant_time,t_v1_model_difference_s=tw+drive-v1_moving,
        t_turn_s=sum(json.loads(r['breakdown_json']).get('TURN',{}).get('seconds',0.) for r in segments if r['kind']=='CONNECTION'),
        t_transit_s=sum(json.loads(r['breakdown_json']).get('TRANSIT',{}).get('seconds',0.) for r in segments if r['kind']=='CONNECTION'),
        headland_work_time_s=sum(r['t_work_s'] for r in segments if r['kind']=='HEADLAND'),
        headland_work_distance_m=sum(r['distance_m'] for r in segments if r['kind']=='HEADLAND'),
        total_distance_m=sum(r['distance_m'] for r in segments),
        t_accel_process_s=sum(e['duration_s'] for e in moving_events if e['event_type']=='ACCEL'),
        t_decel_process_s=sum(e['duration_s'] for e in moving_events if e['event_type']=='DECEL'),
        accel_distance_m=sum(e['distance_m'] for e in moving_events if e['event_type']=='ACCEL'),
        decel_distance_m=sum(e['distance_m'] for e in moving_events if e['event_type']=='DECEL'),
        accel_count=sum(e['event_type']=='ACCEL' for e in moving_events),decel_count=sum(e['event_type']=='DECEL' for e in moving_events))
    return baseline,segments,stops+moving_events


def compute_reference_efficiency(field_id, detail, expected_tasks, config):
    """逐田纯计算接口，供增量批处理使用；输入/输出几何均为局部米制。

    完整路线缺段时保留已知时间，但效率必须为空。该接口复用原账本，
    不重新规划，也不把分段行程自动拼成跨越孔洞的直线。
    """
    full = config['scope']=='FULL_REFERENCE'
    declared = detail.get('full_reference_operations',[]) if full else [r for r in detail.get('operations',[]) if r.get('phase','BODY')=='BODY' and r['kind']!='HEADLAND']
    rows = [{**{k:v for k,v in r.items() if k!='geometry_wkt'},'geometry':wkt.loads(r['geometry_wkt'])} for r in declared]
    if full:
        heads = [r['task_id'] for r in detail.get('operations',[]) if r['kind']=='HEADLAND']
        result,segments,stops = compute_geometry_field(field_id,rows,expected_tasks,config,heads)
        if not detail.get('full_reference_connected') or detail.get('export_audit_passed') is False:
            result.update(efficiency_status='INCOMPLETE',efficiency_ratio=None,efficiency_pct=None,
                reference_complete=False,body_complete=False,accounting_completeness='KNOWN_PART_ONLY',
                missing_reason=(result['missing_reason']+';SOURCE_FULL_REFERENCE_INCOMPLETE').strip(';'))
    else:
        for r in rows:
            r['_local_geometry']=r['geometry'];r['_known_length_m']=r['geometry'].length
        result,segments,stops=compute_field(field_id,rows,expected_tasks,config)
    result['source_route_status']=detail.get('reference_route_status')
    for key in ('upstream_seam_quality_status','upstream_acceptance_passed','upstream_area_ledger_delta_m2'):
        result[key]=detail.get(key)
    return result,segments,stops


def run_batch(batch, config_path, out, *, resume=False):
    """按协议分发；新参数/精简来源逐田转写，不运行路线搜索。"""
    source=route_io.resolve_reference_gpkg(batch)
    with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as conn:
        incremental=conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='batch_metadata'").fetchone()
        if incremental:
            row=conn.execute("SELECT value_json FROM batch_metadata WHERE key='contract'").fetchone()
            version=json.loads(row[0])['schema_version'] if row else None
            cfg=load_config(config_path)
            if version==route_io.COMPACT_SCHEMA or cfg['schema_version']==2:
                return route_io.convert_reference_batch(source,out,config_path=config_path,resume=resume)
            return route_io.run_incremental_efficiency(source.parent,config_path,out,resume=resume)
    if resume:raise ValueError('LEGACY_EFFICIENCY_BATCH_CANNOT_RESUME')
    return _run_legacy_batch(batch,config_path,out)


def _run_legacy_batch(batch, config_path, out):
    """一次读取＋逐田线性统计；另存新 GPKG，旧图层及源批次均不改。"""
    start = time.perf_counter();batch, out = Path(batch).resolve(), Path(out).resolve()
    config_path = Path(config_path).resolve()
    if not out.is_relative_to((ROOT/'outputs').resolve()) or out.exists():
        raise ValueError('OUTPUT_MUST_BE_NEW_DIRECTORY_IN_V7_OUTPUTS')
    config = load_config(config_path)
    gpkg = batch/'reference_routes.gpkg';source_hash = sha256(gpkg);config_hash = sha256(config_path)
    metadata = json.loads((batch/'summary.json').read_text())
    bundle = Path(metadata['input_bundle']);_, manifest = route_io._verify_bundle(bundle)
    if route_io._verify_bundle_seal(bundle) != metadata['input_release_sha256']:
        raise ValueError('INPUT_RELEASE_MISMATCH')
    jobs = {j.field_id: j for j in route_io._field_rows(bundle, manifest)}
    selected = metadata['input_field_ids']
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected)-jobs.keys():
        raise ValueError('INVALID_FIELD_SELECTION')
    names = set(gpd.list_layers(gpkg)['name'])
    if not {'source_fields','body_work'} <= names:
        raise ValueError('MISSING_REQUIRED_LAYER')
    source = gpd.read_file(gpkg, layer='source_fields')
    if Counter(source['field_id']) != Counter(selected):
        raise ValueError('SOURCE_FIELD_SET_MISMATCH')
    # 所有正式空间层均检查田块引用，未知记录不得在按田分组时被跳过。
    with sqlite3.connect(f'file:{gpkg}?mode=ro', uri=True) as conn:
        for layer in names:
            safe_layer = layer.replace('"','""')
            cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{safe_layer}")')}
            if 'field_id' in cols:
                if any(r[0] not in selected for r in conn.execute(f'SELECT DISTINCT field_id FROM "{safe_layer}"')):
                    raise ValueError(f'UNKNOWN_FIELD_REFERENCE: {layer}')
    grouped = defaultdict(list)
    full=config['scope']=='FULL_REFERENCE'
    if full and ('reference_operations' not in names or not metadata.get('settings',{}).get('assemble_reference')):
        raise ValueError('FULL_REFERENCE_OPERATIONS_REQUIRED')
    for layer in (('reference_operations',) if full else BODY_LAYERS):
        if layer not in names:
            continue
        frame = gpd.read_file(gpkg, layer=layer)
        if frame.crs != source.crs:
            raise ValueError('BODY_LAYER_CRS_MISMATCH')
        for row in frame.to_dict('records'):
            if (not isinstance(row['sequence'], Integral) or isinstance(row['sequence'], bool)
                    or row['sequence'] <= 0 or not isinstance(row['component'], Integral) or isinstance(row['component'], bool) or row['component'] <= 0):
                raise ValueError('INVALID_OPERATION_ORDER')
            if full or row.get('phase', 'BODY') == 'BODY':
                grouped[row['field_id']].append(row)
    read_seconds = time.perf_counter()-start
    fields, segments, stops = [], [], []
    for number, fid in enumerate(selected, 1):
        job = jobs[fid];all_rows = grouped[fid];task_ids = {t.task_id for t in job.tasks}
        # 接头来自头部或入口时，不在“主体条带之间”的统计边界内。
        ordered_body = sorted(all_rows, key=lambda r: r['sequence'])
        work_positions = [i for i, r in enumerate(ordered_body) if r['kind']=='WORK']
        # 只排除首条作业线之前和末条之后的进出连接；主体内部异常连接不能静默丢掉。
        rows = ordered_body if full else ordered_body[work_positions[0]:work_positions[-1]+1] if work_positions else ordered_body
        detail = json.loads((batch/'fields'/fid/'result.json').read_text())
        if detail.get('field_id') != fid:
            raise ValueError('JSON_FIELD_ID_MISMATCH')
        declared = detail.get('full_reference_operations',[]) if full else [r for r in detail['operations'] if r.get('phase','BODY')=='BODY']
        def key(r):return tuple(r[k] for k in ('kind','sequence','component','task_id','from_task','to_task'))
        if Counter(key(r) for r in all_rows) != Counter(key(r) for r in declared):
            raise ValueError(f'OPERATION_EXPORT_SET_MISMATCH: {fid}')
        if len({key(r) for r in declared}) != len(declared):
            raise ValueError(f'DUPLICATE_OPERATION_EXPORT: {fid}')
        lookup = {key(r):r for r in declared}
        back = None if job.metric_crs=='LOCAL_METRIC' else Transformer.from_crs(source.crs,job.metric_crs,always_xy=True)
        def local(geometry):return translate(geometry if back is None else transform(back.transform,geometry),-job.origin[0],-job.origin[1])
        for row in rows:
            geometry = local(row['geometry']);row['_local_geometry'] = geometry;row['_known_length_m'] = geometry.length
            if not geometry.equals_exact(wkt.loads(lookup[key(row)]['geometry_wkt']),POSITION_EPS_M):
                raise ValueError(f'JSON_EXPORT_GEOMETRY_MISMATCH: {fid}')
            for name in ('motion_json','events_json','kinematic_status'):
                if row.get(name) != lookup[key(row)].get(name):
                    raise ValueError(f'JSON_MOTION_METADATA_MISMATCH: {fid}')
        if full:
            expected_heads=[r['task_id'] for r in detail['operations'] if r['kind']=='HEADLAND']
            result,per_segment,per_stop=compute_geometry_field(fid,rows,[t.task_id for t in job.tasks],config,expected_heads)
            if not detail.get('full_reference_connected'):
                result.update(efficiency_status='INCOMPLETE',efficiency_ratio=None,efficiency_pct=None,reference_complete=False,body_complete=False,accounting_completeness='KNOWN_PART_ONLY',missing_reason=(result['missing_reason']+';SOURCE_FULL_REFERENCE_INCOMPLETE').strip(';'))
        else:result, per_segment, per_stop = compute_field(fid, rows, [t.task_id for t in job.tasks], config)
        result['geometry'] = source.loc[source.field_id==fid].iloc[0].geometry
        result['source_route_status'] = detail.get('reference_route_status')
        result['region_count'] = len(job.regions)
        for name in ('upstream_seam_quality_status','upstream_acceptance_passed','upstream_area_ledger_delta_m2'):
            result[name] = detail.get(name)
        # 点事件以原输出 CRS 写入；计算仍用局部米制里程，不能拿经纬度算秒。
        forward = None if back is None else Transformer.from_crs(job.metric_crs,source.crs,always_xy=True)
        for record in per_stop:
            point = translate(Point(record['local_x_m'],record['local_y_m']),*job.origin)
            record['geometry'] = point if forward is None else transform(forward.transform,point)
        fields.append(result);segments.extend(per_segment);stops.extend(per_stop)
        if number % 50 == 0:
            print(f'效率统计 {number}/{len(selected)}',flush=True)
    calculate_seconds = time.perf_counter()-start-read_seconds
    if sha256(gpkg)!=source_hash or sha256(config_path)!=config_hash:
        raise ValueError('INPUT_CHANGED_DURING_CALCULATION')
    out.mkdir();target = out/'work_time_efficiency.gpkg'
    # SQLite backup 保留原图层和元数据，不重新投影或重写原有要素。
    with sqlite3.connect(f'file:{gpkg}?mode=ro',uri=True) as source_db, sqlite3.connect(target) as target_db:
        source_db.backup(target_db)
    for name, records in [('field_efficiency',fields),('efficiency_segments',segments),('efficiency_stops',stops)]:
        if records:
            gpd.GeoDataFrame(records,geometry='geometry',crs=source.crs).to_file(target,layer=name,driver='GPKG',index=False)
    with sqlite3.connect(target) as conn:
        conn.execute('CREATE TABLE efficiency_parameters (id INTEGER PRIMARY KEY, section TEXT, parameter TEXT, value TEXT, unit TEXT)')
        parameters = [('model',k,json.dumps(v,ensure_ascii=False),'') for k,v in config.items() if k not in ('speeds','stops')]
        parameters += [(section,k,str(v),'m/s' if section=='speeds' else 's/event') for section in ('speeds','stops') for k,v in config[section].items()]
        conn.executemany('INSERT INTO efficiency_parameters(section,parameter,value,unit) VALUES(?,?,?,?)',parameters)
        conn.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier,description) VALUES('efficiency_parameters','attributes','efficiency_parameters','User-confirmed reference time model')")
    pd.DataFrame([{k:v for k,v in r.items() if k!='geometry'} for r in fields]).to_csv(out/'field_efficiency.csv',index=False,encoding='utf-8-sig')
    atomic_json(out/'efficiency_config.json',config)
    summary = dict(status='COMPLETED',scope=config['scope'],field_count=len(fields),
        efficiency_status_counts=dict(Counter(r['efficiency_status'] for r in fields)),
        segment_count=len(segments),stop_count=len(stops),input_batch=str(batch),
        source_gpkg_sha256=source_hash,config_sha256=config_hash,script_sha256=sha256(Path(__file__)),
        read_and_verify_seconds=read_seconds,calculation_seconds=calculate_seconds,total_wall_seconds=time.perf_counter()-start,
        physical_vehicle_certification='NOT_EVALUATED',time_source='CONFIG_ESTIMATE_NOT_FIELD_MEASURED',
        notes=['完整有序参考行程含主体、田头和已组装连接；未配置田门时不计外部进出/转场' if full else '主体条带及其相互连接，不含独立田头/进出田块/外部转场',
               'FULL_REFERENCE 包含有序田头，按参考几何估算，不要求实车运动信息',
               '不完整的田块效率为空，时间仅表示已知部分',
               '停顿的分项名义时间不可相加充当实际停顿；实际取同次动作最大值',
               '倒车时间是作业/非作业时间的子项，不能重复计入分母'])
    tw=sum(r['t_work_s'] for r in fields);tb=sum(r['t_break_s'] for r in fields)
    complete=all(r['efficiency_status']=='ESTIMATED' for r in fields)
    summary.update(t_work_seconds_sum=tw,t_break_seconds_sum=tb,aggregate_efficiency_ratio=tw/(tw+tb) if complete and tw+tb>0 else None,aggregate_policy='RATIO_OF_SUMMED_TIMES_NULL_IF_ANY_FIELD_INCOMPLETE')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    """独立效率命令入口，只读取已有路线和所选时间配置；支持续算输出，禁止重新规划来改变输入路线。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route-batch',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batch-resume',action='store_true',help='续算增量 GPKG 的效率输出')
    args=parser.parse_args()
    try:
        result=run_batch(args.route_batch,args.config,args.out,resume=args.batch_resume)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['status']=='COMPLETED' and result['efficiency_status_counts'].get('ESTIMATED',0)==result.get('input_field_count',result.get('field_count',0)) else 1
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print(f'效率统计失败：{type(exc).__name__}: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':sys.exit(main())
