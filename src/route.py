"""作业任务、连接与路线策略。输入 FieldInput 与阶段参数，输出任务候选、路线和状态。
前三类策略有明确边界：严格连接、规则模板、APPROX_CONNECTED 参考路线。
参考策略可选车辆位姿/运动约束组装，仍不等于完整机具安全和田间认证。

分节目录：
1. 严格连接、条带排序与主体行程
2. 田头任务、补作候选与覆盖保护
3. 田头轮廓与温和转角
4. 路线阶段授权的作业空间准备
5. 独立田头参考作业
6. 行程前缀复核、续作与末条恢复
7. 主体和田头任务的有界联合组织
8. 田头剩余目标的补作候选
9. 已作业区域中的中转锚点
10. 规则路线的田头参考与端口准备
11. 统一局部调头模板
12. 规则连续路线策略
13. APPROX_CONNECTED 参考路线策略
"""
from __future__ import annotations

from io_utils import config_section


# ==========================================================================
# 1. 严格连接、条带排序与主体行程
# 输入冻结任务和车辆参数，输出已验证连接或明确失败；此策略保留严格认证语义。
# ==========================================================================

from collections import Counter
from collections import OrderedDict
from dataclasses import replace
import heapq
import math
import time

import numpy as np
import shapely
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.ops import unary_union

import route_planner as io
from scene import Motion
from scene import Pose
from scene import wrap
from validator import MotionChecker


# 严格局部连接器：输入两个有朝向的车辆位姿，输出经过核验的 Motion。
# 连接必须满足该策略的空间与车辆条件；找不到就保留失败原因。
class Connector:
    """原生 F2C 负责车辆曲线；本类负责候选、实际空间验证和复用。"""
    def __init__(self, scene, settings, deadline, *, medial_enabled=False):
        """初始化严格连接器的场景、预算及缓存，只处理当前已开放的通行空间。"""
        self.scene, self.settings, self.deadline = scene, settings, deadline
        self.medial_enabled = medial_enabled
        self.backend = io.F2CBackend(scene)
        # 原生缓存可能复用邻近离散位姿；本地缓存保存归一化曲线，每次仍检查本次真实端点。
        for _, turner in self.backend.turners:
            turner.setUsingCache(False)
        self.checker = MotionChecker(scene)
        self.templates = OrderedDict()
        self.counts = Counter()
        self.timing = Counter()

    def native(self, a, b, index, backward=False, guide=None):
        """调用原生车辆连接并转换运动，后续仍核验曲率、包络和作业空间。"""
        started = time.perf_counter()
        self.counts['native_requests'] += 1
        c, s = math.cos(a.yaw), math.sin(a.yaw)
        dx, dy = b.x-a.x, b.y-a.y
        end = Pose(c*dx+s*dy, -s*dx+c*dy, wrap(b.yaw-a.yaw))
        key = (index, backward, round(end.x, 8), round(end.y, 8),
               round(end.yaw, 10))
        cached = self.templates.get(key) if guide is None else None
        if cached is None:
            self.counts['native_calls'] += 1
            turner = self.backend.turners[index][1]
            shift = math.pi if backward else 0
            start = Pose(0, 0, shift)
            finish = Pose(end.x, end.y, end.yaw+shift)
            local_guide = None
            if guide is not None:
                local_guide = [(c*(x-a.x)+s*(y-a.y), -s*(x-a.x)+c*(y-a.y))
                               for x,y in guide]
            path = self.backend.turn_path(start, finish, turner, local_guide)
            motion = self.backend.convert_path(path, ('from', 'to'), 'turn')
            p = motion.points.copy()
            if backward:
                p[:, 2] -= math.pi
                p[:, 3] *= -1
            if guide is None:
                self.templates[key] = p.copy()
                if len(self.templates)>512:
                    self.templates.popitem(last=False)
        else:
            p = cached.copy()
            self.counts['template_hits'] += 1
        xy = p[:, :2].copy()
        p[:, 0] = a.x+c*xy[:, 0]-s*xy[:, 1]
        p[:, 1] = a.y+s*xy[:, 0]+c*xy[:, 1]
        p[:, 2] += a.yaw
        result = Motion(p, 'turn')
        if not (io._pose_close(io._pose_at(result, False), a, self.scene)
                and io._pose_close(io._pose_at(result, True), b, self.scene)):
            raise ValueError('NATIVE_ENDPOINT_MISMATCH')
        self.timing['native_s'] += time.perf_counter()-started
        return result

    def evaluate(self, motion, method, before, after, ready, worked_by_region):
        """评价候选运动的合法性与代价，失败返回具体原因，不以平滑外观判定可执行。"""
        start = time.perf_counter()
        eps = self.scene.settings.geometry_epsilon_m
        allow = ready.buffer(eps)
        shapely.prepare(allow)
        # 车辆参考点位于车体内；点已经禁入时先廉价拒绝，不必继续生成大量包络面，但点合法不能单独证明车体合法。
        if not np.all(shapely.covers(allow, shapely.points(motion.points[:, :2]))):
            self.counts['REFERENCE_OUTSIDE_READY'] += 1
            return None
        issues, parts = self.checker.physical(motion)
        self.timing['physical_check_s'] += time.perf_counter()-start
        if issues:
            self.counts.update(issues)
            return None
        legs, shifts, reverse_m, events, steer_s = io._gear_and_steering(
            motion, self.scene, self.settings)
        if not math.isfinite(steer_s):
            self.counts['GEAR_OR_REVERSE_LIMIT'] += 1
            return None
        start = time.perf_counter()
        endpoint_area_m2 = 0.0
        if not np.all(shapely.covers(allow, parts)):
            self.counts['UNWORKED_BODY_CROSSING'] += 1
            self.timing['resource_check_s'] += time.perf_counter()-start
            return None
        self.timing['resource_check_s'] += time.perf_counter()-start
        p = motion.points
        ds = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
        v = self.scene.vehicle
        seconds = float(np.sum(ds/np.where(p[:-1,3]<0, v.reverse_speed_mps, v.turn_speed_mps)))
        seconds += shifts*v.gear_change_seconds+steer_s
        for task, kind in ((before,'IMPLEMENT_OFF'),(after,'IMPLEMENT_ON')):
            if task is not None:
                events.append({'kind':kind, 'task_id':task.task_id,
                               'duration_s':v.implement_switch_seconds,
                               'duration_source':'CONFIG_NOMINAL_LAG_UNVERIFIED'})
                seconds += v.implement_switch_seconds
        result = io.Connection(before.task_id if before else None,
                               after.task_id if after else None, motion, method,
                               seconds, reverse_m, legs, shifts, events)
        result.endpoint_unworked_area_m2 = endpoint_area_m2
        return result

    def connect(self, a, b, before, after, ready, worked_by_region, *,
                transfer=False, _allow_portals=True, _allow_medial=True):
        """在有限候选和预算内连接真实位姿，作物时序和已作业空间由当前状态约束。"""
        self.counts['connection_queries'] += 1
        if io._pose_close(a,b,self.scene):
            return io.Connection(before.task_id if before else None,
                                 after.task_id if after else None,None,'CONTIGUOUS',0,0,0,0)
        choices=[]
        direct=io._direct_motion(a,b,self.scene)
        if direct is not None:
            candidate=self.evaluate(direct,'STRAIGHT',before,after,ready,worked_by_region)
            if candidate:
                choices.append(candidate)
        # 紧边界的先倒车HC属于独立运动家族；转换朝向和挡位时必须保持真实刚性车体位姿一致。
        families=[(0,False)]
        if self.scene.vehicle.allow_reverse:
            families += [(1,False),(1,True)]
        for index,backward in families:
            if time.perf_counter()>=self.deadline:
                break
            try:
                motion=self.native(a,b,index,backward)
                candidate=self.evaluate(motion,('BACKWARD_START_' if backward else '')+
                    self.backend.turners[index][0],before,after,ready,worked_by_region)
                if candidate:
                    choices.append(candidate)
            except (ValueError,RuntimeError,IndexError):
                self.counts['NATIVE_CONVERSION_REJECTED']+=1
        if not choices and time.perf_counter()<self.deadline:
            # Retreat/re-approach distances follow vehicle scale, not a dense
            # grid. Both straight pieces are non-working and validated together.
            r=self.scene.vehicle.min_turn_radius_m
            for retreat, approach in ((-r,r),(-r/2,r/2),(r,0),(0,r)):
                if time.perf_counter()>=self.deadline:
                    break
                start=Pose(a.x-retreat*math.cos(a.yaw),a.y-retreat*math.sin(a.yaw),a.yaw)
                end=Pose(b.x-approach*math.cos(b.yaw),b.y-approach*math.sin(b.yaw),b.yaw)
                for idx,back in families:
                    try:
                        central=self.native(start,end,idx,back)
                        p=central.points.copy()
                        if retreat:
                            p=np.vstack(([a.x,a.y,a.yaw,-1 if retreat>0 else 1],p))
                        if approach:
                            p[-1,3]=1
                            p=np.vstack((p,[b.x,b.y,b.yaw,1]))
                        motion=Motion(p,'transit' if transfer else 'turn')
                        candidate=self.evaluate(motion,f'RETREAT_{retreat:g}_{approach:g}_'+
                            ('BACK_' if back else '')+self.backend.turners[idx][0],
                            before,after,ready,worked_by_region)
                        if candidate:
                            choices.append(candidate)
                    except (ValueError,RuntimeError,IndexError):
                        self.counts['NATIVE_CONVERSION_REJECTED']+=1
                if choices:
                    break
        guides=[]
        if not choices and time.perf_counter()<self.deadline:
            allow=ready
            guides=visibility_guides(allow,self.scene,a,b,self.settings.max_guide_trials)
            for guide in guides:
                if time.perf_counter()>=self.deadline:
                    break
                self.counts['guide_trials']+=1
                for idx,back in families:
                    try:
                        motion=self.native(a,b,idx,back,guide)
                        candidate=self.evaluate(motion,'GUIDED_'+('BACK_' if back else '')+
                            self.backend.turners[idx][0],before,after,ready,worked_by_region)
                        if candidate:
                            choices.append(candidate)
                    except (ValueError,RuntimeError,IndexError):
                        self.counts['NATIVE_CONVERSION_REJECTED']+=1
                if choices:
                    break
        if (not choices and transfer and _allow_portals and guides
                and time.perf_counter()<self.deadline):
            # A single F2C curve between swath ends may overshoot a narrow
            # passage. Derive sparse straight portals from the actual free
            # space; F2C still supplies the turns on both sides. The joined
            # motion is rechecked as one rigid-body/dynamic-ready trajectory.
            for left,right in straight_portals(guides,ready,self.scene):
                if time.perf_counter()>=self.deadline:
                    break
                self.counts['portal_trials']+=1
                middle=io._direct_motion(left,right,self.scene)
                if middle is None or self.evaluate(middle,'PORTAL_STRAIGHT',
                                                   None,None,ready,worked_by_region) is None:
                    continue
                incoming=self.connect(a,left,before,None,ready,worked_by_region,
                                      transfer=True,_allow_portals=False)
                if incoming is None or time.perf_counter()>=self.deadline:
                    continue
                outgoing=self.connect(right,b,None,after,ready,worked_by_region,
                                      transfer=True,_allow_portals=False)
                if outgoing is None:
                    continue
                # The central straight leg must remain between the side legs.
                segments=([incoming.motion] if incoming.motion is not None else [])+[
                    middle]+([outgoing.motion] if outgoing.motion is not None else [])
                joined=np.vstack([motion.points[:-1] for motion in segments[:-1]]+
                                 [segments[-1].points])
                candidate=self.evaluate(Motion(joined,'transit'),
                    'STAGED_PORTAL_F2C',before,after,ready,worked_by_region)
                if candidate is not None:
                    self.counts['portal_success']+=1
                    choices.append(candidate)
                    break
        # The expensive multi-bend fallback solves transfer between distinct
        # body regions. Headland task insertion has its own bounded search;
        # running a free-space graph for every rejected ring-to-ring attempt
        # consumes that budget without improving the body connection.
        body_transfer=(before is not None and after is not None
                       and before.region_id!=after.region_id
                       and not before.region_id.startswith('__')
                       and not after.region_id.startswith('__'))
        if (not choices and transfer and _allow_medial and self.medial_enabled
                and body_transfer
                and time.perf_counter()<self.deadline):
            candidate=self._medial_transfer(a,b,before,after,ready,
                                            worked_by_region)
            if candidate is not None:
                choices.append(candidate)
        if not choices:
            return None
        best=min(choices,key=lambda c:(c.seconds,c.gear_shifts,c.reverse_m,c.method))
        if transfer and best.motion is not None:
            best.motion=Motion(best.motion.points,'transfer',implement_on=False,
                               link=(best.from_task or 'FIELD_START',best.to_task or 'FIELD_END'))
        return best

    def _medial_transfer(self,a,b,before,after,ready,worked_by_region):
        """将高净空轴作为原生转场引导；轴线自身不作为已认证运动输出。
        
        Curve a multi-bend free-space axis with F2C, then validate the whole leg.
        
        A single guided curve can overshoot a long bent passage.  This fallback
        only runs after the ordinary and straight-portal connectors fail.  Its
        sparse grid finds *guidance*, never a vehicle trajectory; every local
        curve and the joined motion must pass the normal physical and dynamic
        ready-area checks.  The complete leg also obeys the configured total
        reverse-leg and gear-shift limits."""
        axis=_medial_axis_guide(ready,self.scene,a,b,self.deadline)
        if not axis:
            return None
        radius=self.scene.vehicle.min_turn_radius_m
        for trim in (2.4*radius,3.2*radius):
            if time.perf_counter()>=self.deadline:
                break
            poses=[a]
            for start,end in zip(axis,axis[1:]):
                dx,dy=end[0]-start[0],end[1]-start[1]
                length=math.hypot(dx,dy)
                if length<1e-6:
                    continue
                yaw=math.atan2(dy,dx)
                low=min(trim,length/2)
                high=max(low,length-trim)
                count=max(1,math.ceil((high-low)/(6*radius))+1)
                for distance in np.linspace(low,high,count):
                    poses.append(Pose(start[0]+distance*dx/length,
                                      start[1]+distance*dy/length,yaw))
            poses.append(b)
            if len(poses)>42:
                self.counts['medial_pose_limit']+=1
                continue
            segments=[]
            for first,last in zip(poses,poses[1:]):
                if time.perf_counter()>=self.deadline:
                    break
                local=None
                straight=io._direct_motion(first,last,self.scene)
                if straight is not None:
                    local=self.evaluate(straight,'STRAIGHT',None,None,
                                        ready,worked_by_region)
                if local is None:
                    try:
                        forward=self.native(first,last,0)
                        local=self.evaluate(forward,'dubins_cc',None,None,
                                            ready,worked_by_region)
                    except (ValueError,RuntimeError,IndexError):
                        self.counts['NATIVE_CONVERSION_REJECTED']+=1
                if local is None:
                    local=self.connect(first,last,None,None,ready,worked_by_region,
                                       transfer=False,_allow_portals=False,
                                       _allow_medial=False)
                if local is None or local.motion is None:
                    break
                segments.append(local.motion)
                if len(segments)>1:
                    partial=np.vstack([motion.points[:-1] for motion in segments[:-1]]+
                                      [segments[-1].points])
                    _,shifts,_,_,steer=io._gear_and_steering(
                        Motion(partial,'transit'),self.scene,self.settings)
                    if shifts>self.settings.max_gear_shifts or not math.isfinite(steer):
                        break
            else:
                joined=np.vstack([motion.points[:-1] for motion in segments[:-1]]+
                                 [segments[-1].points])
                result=self.evaluate(Motion(joined,'transit'),
                    'MEDIAL_MULTI_STAGE_F2C',before,after,ready,worked_by_region)
                if result is not None:
                    self.counts['medial_success']+=1
                    return result
            self.counts['medial_rejected']+=1
        return None


def _medial_axis_guide(ready,scene,a,b,deadline):
    """在当前已开放空间搜索有界高净空引导；网格只建议走廊，最终运动仍须独立求解及检查。
    
    Find a bounded high-clearance axis through the *current* ready space.
    
    Grid nodes only suggest a corridor.  A visibility simplification reduces
    them to a few long straight supports; the caller still needs F2C curves
    and an independent check of the complete vehicle motion."""
    vehicle=scene.vehicle
    half=max(vehicle.body_width_m,vehicle.working_width_m)/2+vehicle.safety_margin_m
    free=ready.intersection(scene.travel).buffer(-half)
    parts=[part for part in shapely.get_parts(free) if part.geom_type=='Polygon']
    if not parts:
        return []
    pa,pb=Point(a.x,a.y),Point(b.x,b.y)
    free=min(parts,key=lambda part:part.distance(pa)+part.distance(pb))
    bounds=free.bounds
    step=max(2.0,0.8*vehicle.min_turn_radius_m,
             math.sqrt((bounds[2]-bounds[0])*(bounds[3]-bounds[1])/12000))
    if max(free.distance(pa),free.distance(pb))>max(2*step,2*vehicle.min_turn_radius_m):
        return []
    xs=np.arange(bounds[0],bounds[2]+step,step)
    ys=np.arange(bounds[1],bounds[3]+step,step)
    if len(xs)*len(ys)>16000 or time.perf_counter()>=deadline:
        return []
    gx,gy=np.meshgrid(xs,ys)
    points=shapely.points(gx.ravel(),gy.ravel())
    mask=shapely.covers(free,points).reshape(gx.shape)
    flat=np.flatnonzero(mask.ravel())
    if not len(flat):
        return []
    coords=np.column_stack((gx.ravel(),gy.ravel()))
    clearance=shapely.distance(free.boundary,points).reshape(gx.shape)
    start=flat[np.argmin(np.sum((coords[flat]-[a.x,a.y])**2,axis=1))]
    goal=flat[np.argmin(np.sum((coords[flat]-[b.x,b.y])**2,axis=1))]
    width,height=len(xs),len(ys)
    cost={int(start):0.0};previous={}
    heap=[(math.dist(coords[start],coords[goal]),0.0,int(start))]
    expanded=0
    while heap and expanded<12000:
        _,distance,node=heapq.heappop(heap)
        if distance>cost[node]+1e-9:
            continue
        if node==goal:
            break
        expanded+=1
        if expanded%256==0 and time.perf_counter()>=deadline:
            return []
        row,col=divmod(node,width)
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1),
                      (-1,-1),(-1,1),(1,-1),(1,1)):
            nr,nc=row+dr,col+dc
            if not (0<=nr<height and 0<=nc<width and mask[nr,nc]):
                continue
            other=nr*width+nc
            if not free.covers(LineString((coords[node],coords[other]))):
                continue
            length=step*math.hypot(dr,dc)
            centre=min(clearance[row,col],clearance[nr,nc])
            next_cost=distance+length*(1+2*vehicle.min_turn_radius_m/max(centre,1))
            if next_cost<cost.get(other,math.inf):
                cost[other]=next_cost;previous[other]=node
                priority=next_cost+math.dist(coords[other],coords[goal])
                heapq.heappush(heap,(priority,next_cost,other))
    else:
        return []
    nodes=[int(goal)]
    while nodes[-1]!=start:
        nodes.append(previous[nodes[-1]])
    nodes.reverse()
    path=[tuple(coords[node]) for node in nodes]
    inner=free.buffer(-min(1.0,step/4))
    if inner.is_empty:
        inner=free
    simplified=[path[0]]
    index=0
    while index<len(path)-1:
        next_index=min(len(path)-1,index+60)
        while (next_index>index+1 and not inner.covers(
                LineString((path[index],path[next_index])))):
            next_index-=1
        simplified.append(path[next_index])
        index=next_index
        if len(simplified)>16:
            return []
    return simplified


def visibility_guides(available,scene,a,b,maximum):
    """从侵蚀后的真实空间构建有界可见图，返回引导折线而非车辆轨迹。"""
    v=scene.vehicle
    half=max(v.body_width_m,v.working_width_m)/2+v.safety_margin_m
    guides=[]
    for inset in (half,half+v.min_turn_radius_m):
        free=available.intersection(scene.travel).buffer(-inset)
        if free.is_empty:
            continue
        parts=list(free.geoms) if free.geom_type=='MultiPolygon' else [free]
        part=min(parts,key=lambda p:p.distance(Point(a.x,a.y))+p.distance(Point(b.x,b.y)))
        pa=nearest_points(part,Point(a.x,a.y))[0]
        pb=nearest_points(part,Point(b.x,b.y))[0]
        vertices=[]
        simple=part.simplify(max(.25,inset*.1),preserve_topology=True)
        for ring in [simple.exterior,*simple.interiors]:
            coords=list(ring.coords)[:-1]
            vertices.extend(coords)
        # Sample a finite ring graph. No all-swath-pair graph is constructed.
        if len(vertices)>120:
            vertices=[vertices[i] for i in np.linspace(0,len(vertices)-1,120,dtype=int)]
        points=[(pa.x,pa.y),(pb.x,pb.y),*vertices]
        coords=np.asarray(points)
        i,j=np.triu_indices(len(points),1)
        lines=shapely.linestrings(np.stack((coords[i],coords[j]),axis=1))
        use=shapely.covers(part.buffer(1e-6),lines)
        adjacency=[[] for _ in points]
        for first,last in zip(i[use],j[use]):
            distance=float(np.linalg.norm(coords[first]-coords[last]))
            adjacency[first].append((last,distance));adjacency[last].append((first,distance))
        distances={0:0.0};previous={};heap=[(0.0,0)]
        while heap:
            cost,node=heapq.heappop(heap)
            if node==1:
                indices=[1]
                while indices[-1]!=0:
                    indices.append(previous[indices[-1]])
                guide=[(a.x,a.y),*[points[k] for k in indices[::-1]],(b.x,b.y)]
                # Drop repeated/nearby points before native curve generation.
                cleaned=[guide[0]]
                for point in guide[1:]:
                    if math.dist(cleaned[-1],point)>.05:
                        cleaned.append(point)
                if len(cleaned)>2:
                    guides.append(cleaned)
                break
            if cost>distances[node]+1e-9:
                continue
            for other,length in adjacency[node]:
                new=cost+length
                if new<distances.get(other,math.inf):
                    distances[other]=new;previous[other]=node;heapq.heappush(heap,(new,other))
        if len(guides)>=maximum:
            break
    return guides


def straight_portals(guides,ready,scene):
    """提出少量真实直行通道作为连接候选，整段包络检查后才使用，候选端口不是已完成路线。
    
    Offer at most two geometry-derived straight passages, not a route.
    
    An eroded-space visibility edge is useful only if the entire proposed
    straight centreline, including short extensions into both adjacent open
    areas, lies in the same conservative free space. Narrowness is measured
    at the edge midpoint; boundary-hugging and artificial guide endpoints
    cannot become portals merely because they are long."""
    vehicle=scene.vehicle
    half=max(vehicle.body_width_m,vehicle.working_width_m)/2+vehicle.safety_margin_m
    radius=vehicle.min_turn_radius_m
    free=ready.intersection(scene.travel).buffer(-half)
    if free.is_empty:
        return []
    allowed=free.buffer(1e-6)
    minimum_length=max(8*radius,4*half)
    extension=max(2*radius,vehicle.front_m+vehicle.rear_m,2*half)
    found=[]
    for guide in guides:
        for start,end in zip(guide,guide[1:]):
            length=math.dist(start,end)
            if length<minimum_length:
                continue
            line=LineString((start,end))
            if not allowed.covers(line):
                continue
            midpoint=line.interpolate(.5,normalized=True)
            clearance=midpoint.distance(free.boundary)
            if not 1e-4<clearance<radius/2:
                continue
            dx=(end[0]-start[0])/length;dy=(end[1]-start[1])/length
            # Visibility edges often join opposite corridor walls. Following
            # that slight diagonal can scrape a tool corner even though its
            # midpoint is inside the eroded polygon. Take three perpendicular
            # sections and fit the centres of the same free-space component.
            span=max(free.bounds[2]-free.bounds[0],
                     free.bounds[3]-free.bounds[1],100.0)
            centres=[]
            for fraction in (.25,.5,.75):
                x=start[0]+fraction*(end[0]-start[0])
                y=start[1]+fraction*(end[1]-start[1])
                section=LineString(((x+dy*span,y-dx*span),
                                    (x-dy*span,y+dx*span)))
                pieces=[piece for piece in shapely.get_parts(free.intersection(section))
                        if piece.geom_type=='LineString' and piece.length>1e-6]
                if not pieces:
                    break
                nearest=min(pieces,key=lambda piece:piece.distance(Point(x,y)))
                centres.append(nearest.interpolate(.5,normalized=True))
            if len(centres)!=3:
                continue
            fitted_dx=centres[2].x-centres[0].x
            fitted_dy=centres[2].y-centres[0].y
            fitted_length=math.hypot(fitted_dx,fitted_dy)
            if fitted_length<length/3:
                continue
            fitted_dx/=fitted_length;fitted_dy/=fitted_length
            reach=length/2+extension
            left=(centres[1].x-reach*fitted_dx,
                  centres[1].y-reach*fitted_dy)
            right=(centres[1].x+reach*fitted_dx,
                   centres[1].y+reach*fitted_dy)
            if not allowed.covers(LineString((left,right))):
                continue
            yaw=math.atan2(fitted_dy,fitted_dx)
            found.append((clearance,-length,Pose(*left,yaw),Pose(*right,yaw)))
    found.sort(key=lambda item:(item[0],item[1],item[2].x,item[2].y))
    return [(left,right) for _,_,left,right in found[:2]]


def orders(tasks,scene,settings,backend,entry=None):
    """保留任务身份，使用 F2C 规律排序和按转弯尺度生成的少量隔行顺序。"""
    if len(tasks)<=1:
        if tasks:
            yield 'SINGLE_TASK',list(tasks)
        return
    result=[('FROZEN',sorted(tasks,key=lambda t:(t.suggested_order,t.task_id)))]
    f2c=backend.f2c
    swaths=f2c.Swaths()
    canonical=sorted(tasks,key=lambda t:(t.row_index,t.suggested_order,t.task_id))
    # Consecutive scan rows on the same side of a hole form an execution block.
    # This is an ordering aid; it neither changes frozen partitions nor merges
    # fragments through the obstacle.
    angle=tasks[0].heading_rad % math.pi
    u=np.array([math.cos(angle),math.sin(angle)])
    groups=[];last=[]
    rows={}
    for task in canonical:
        rows.setdefault(task.row_index,[]).append(task)
    threshold=2*scene.vehicle.min_turn_radius_m+2*scene.vehicle.working_width_m
    for row_id,subjects in sorted(rows.items()):
        current=[];used=set()
        for task in subjects:
            interval=sorted(np.asarray(task.reference_line.coords)@u)
            matches=[(max(abs(interval[0]-lo),abs(interval[-1]-hi)),gid)
                     for prev,gid,lo,hi in last if row_id-prev<=2 and gid not in used]
            score,gid=min(matches) if matches else (math.inf,-1)
            if score>threshold:
                gid=len(groups);groups.append([])
            groups[gid].append(task);used.add(gid)
            current.append((row_id,gid,interval[0],interval[-1]))
        last=current
    if 1<len(groups)<max(12,len(tasks)//3):
        remaining=list(groups);grouped=[];end=None
        while remaining:
            choices=[]
            for i,group in enumerate(remaining):
                for reverse in [False,True]:
                    sequence=list(reversed(group)) if reverse else group
                    t=sequence[0]
                    xy=[np.asarray(t.reference_line.coords[0]),np.asarray(t.reference_line.coords[-1])]
                    cost=(min(float(np.linalg.norm(p-end)) for p in xy) if end is not None
                          else float(sequence[0].row_index))
                    choices.append((cost,i,reverse,sequence))
            _,i,_,sequence=min(choices,key=lambda x:x[:3])
            grouped.extend(sequence);end=np.asarray(sequence[-1].reference_line.coords[-1])
            remaining.pop(i)
        result.insert(0,('CONNECTED_ROW_BLOCKS',grouped))
    for index,task in enumerate(canonical):
        line=f2c.LineString();line.importFromWkt(task.reference_line.wkt)
        swaths.push_back(f2c.Swath(line,scene.vehicle.working_width_m,index))
    for name,sorter in [('F2C_BOUSTROPHEDON',f2c.RP_Boustrophedon()),
                        ('F2C_SNAKE',f2c.RP_Snake())]:
        selected=sorter.genSortedSwaths(swaths)
        ids=[int(selected.at(i).getId()) for i in range(selected.size())]
        if sorted(ids)!=list(range(len(tasks))):
            raise ValueError('F2C_ORDER_CHANGED_TASKS')
        result.append((name,[canonical[i] for i in ids]))
    spacing=scene.vehicle.working_width_m*(1-scene.settings.overlap_fraction)
    skip=max(2,math.ceil(2*scene.vehicle.min_turn_radius_m/spacing))
    for k in sorted({skip,skip+1}):
        selected=[canonical[i] for phase in range(min(k,len(tasks)))
                  for i in range(phase,len(tasks),k)]
        result.append((f'ADAPTIVE_SKIP_{k}',selected))
    result.extend((name+'_REVERSE_ORDER',list(reversed(items))) for name,items in list(result)[:2])
    if entry is not None:
        # Start a block at its accessible side. Starting only at the smallest
        # row number can force an unnecessary full-field transfer.
        for reverse in [True,False]:
            base=list(reversed(canonical)) if reverse else canonical
            nearest=min(range(len(base)),key=lambda i:min(
                math.hypot(x-entry.x,y-entry.y) for x,y in base[i].reference_line.coords))
            result.insert(0,('ENTRY_NEAREST_'+str(int(reverse)),base[nearest:]+base[:nearest]))
    # 相邻条带间距小于转弯直径时不能用简单半圆；优先尝试车辆尺度跳行，每个真实连接仍独立检查。
    if (settings.prefer_radius_compatible_orders
            and spacing < 2*scene.vehicle.min_turn_radius_m
            and len({task.row_index for task in canonical}) == len(canonical)
            and all(abs(math.sin(task.heading_rad-canonical[0].heading_rad))
                    < 1e-5 for task in canonical)):
        result.sort(key=lambda item: not item[0].startswith('ADAPTIVE_SKIP_'))
    seen=set();unique=[]
    for name,items in result:
        signature=tuple(t.task_id for t in items)
        if signature not in seen:
            seen.add(signature);unique.append((name,items))
    count=0
    for name,items in unique:
        yield name,items
        count+=1
        if count>=settings.max_order_candidates:
            return
        if count==1 and 8<len(tasks)<=64:
            # This native graph is built only after the first pattern fails.
            # Its one-second OR-tools ordering budget does not perform a grid
            # search over steering angles or permutations in Python.
            try:
                navigation=scene.travel.buffer(-max(
                    scene.vehicle.front_m,scene.vehicle.rear_m,
                    abs(scene.vehicle.implement_offset_m)+scene.vehicle.implement_length_m/2)-
                    scene.vehicle.safety_margin_m-1.0).simplify(1.0,preserve_topology=True)
                # The OR-tools limit does not bound graph construction. Bound
                # its input too; detailed safety geometry is never simplified.
                if shapely.get_num_coordinates(navigation)>160:
                    continue
                cells=backend.cells(navigation)
                grouped=backend.f2c.SwathsByCells();grouped.push_back(swaths)
                native=backend.f2c.RP_RoutePlannerBase().genRoute(
                    cells,grouped,d_tol=0.5,time_limit_seconds=1)
                ids=[]
                for i in range(native.sizeVectorSwaths()):
                    chunk=native.getSwaths(i)
                    ids.extend(int(chunk.at(j).getId()) for j in range(chunk.size()))
                if sorted(ids)==list(range(len(tasks))):
                    yield 'F2C_HEADLAND_GRAPH',[canonical[i] for i in ids]
                    count+=1
                    if count>=settings.max_order_candidates:
                        return
            except (RuntimeError,ValueError,TypeError):
                pass


def candidate_region(region,ordered,mode,variants,scene,connector,headland,
                     completed,worked_by_region,previous,previous_task,force_first):
    """按一组任务顺序规划严格分区路线，逐步更新已作业空间，失败保留具体连接及任务证据。"""
    route=io.RegionRoute(region.region_id,'REGION_ROUTE_NOT_FOUND',order_mode=mode)
    worked=completed
    current=previous
    before=previous_task
    selected_sweeps=[]
    for index,task in enumerate(ordered):
        if time.perf_counter()>=connector.deadline:
            route.reason='SEARCH_BUDGET_EXHAUSTED';break
        options=variants[task.task_id]
        if index==0 and force_first is not None:
            options=[o for o in options if o.reversed==force_first]
        ready=io._ready_area(scene,headland,worked)
        accepted=[]
        if current is not None:
            options=sorted(options,key=lambda o:math.hypot(o.start.x-current.x,o.start.y-current.y))
        for variant in options:
            if current is not None and accepted:
                lower_bound=math.hypot(variant.start.x-current.x,variant.start.y-current.y)/max(
                    scene.vehicle.turn_speed_mps,scene.vehicle.reverse_speed_mps)
                if lower_bound>min(x[0] for x in accepted)*1.05:
                    connector.counts['direction_lower_bound_pruned']+=1
                    continue
            if current is None:
                link=io.Connection(None,task.task_id,None,'FREE_START',0,0,0,0)
            else:
                link=connector.connect(current,variant.start,before,task,ready,
                                       worked_by_region,transfer=(index==0 and before is not None
                                           and before.region_id!=task.region_id))
            route.explored_candidates+=1
            if link is not None:
                work_seconds=(variant.motion.length/scene.vehicle.work_speed_mps
                    if getattr(task,'alternative_motion_points',()) else 0.0)
                accepted.append((link.seconds+work_seconds,variant.reversed,variant,link))
        if not accepted:
            route.reason='NO_LEGAL_CONNECTION_TO:'+task.task_id
            break
        _,_,variant,link=min(accepted,key=lambda x:x[:2])
        if link.motion is not None:
            route.connections.append(link);route.motions.append(link.motion)
        route.motions.append(variant.motion);route.task_order.append(task.task_id)
        route.sweep_error_m2+=variant.sweep_error_m2
        selected_sweeps.append(variant.sweep)
        worked=unary_union([worked,variant.sweep.intersection(scene.target)])
        current=variant.end;before=task
    if len(route.task_order)==len(ordered) and not io._route_continuity_issues(route.motions,scene):
        route.status='REGION_ROUTE_PASS'
        route.completed_sweep=unary_union(selected_sweeps)
        route.connection_seconds=sum(c.seconds for c in route.connections)
        route.reverse_m=sum(c.reverse_m for c in route.connections)
        route.gear_shifts=sum(c.gear_shifts for c in route.connections)
    return route


def candidate_region_dp(region,ordered,mode,variants,scene,connector,headland,
                        completed,worked_by_region,previous,previous_task):
    """固定顺序上的两朝向动态规划。

    一个任务的两种合法朝向具有同一机具扫掠，所以执行前缀相同时，可用
    空间与朝向选择无关。每层只需保留两个到达状态，避免贪心方向造成死路。
    """
    # state = (cost, end pose, path of (work variant, incoming connection))
    states=[(0.0,previous,[])]
    worked=completed;before=previous_task
    calls=0
    for index,task in enumerate(ordered):
        ready=io._ready_area(scene,headland,worked)
        following=[]
        for variant in variants[task.task_id]:
            candidates=[]
            ranked=sorted(states,key=lambda state:state[0]+(
                math.hypot(state[1].x-variant.start.x,state[1].y-variant.start.y)
                /max(scene.vehicle.turn_speed_mps,scene.vehicle.reverse_speed_mps)
                if state[1] else 0))
            for cost,pose,path in ranked:
                if time.perf_counter()>=connector.deadline:
                    break
                lower=cost+(math.hypot(pose.x-variant.start.x,pose.y-variant.start.y)
                    /max(scene.vehicle.turn_speed_mps,scene.vehicle.reverse_speed_mps) if pose else 0)
                if candidates and lower>=min(c[0] for c in candidates):
                    continue
                link=(connector.connect(pose,variant.start,before,task,ready,worked_by_region,
                                        transfer=(index==0 and before is not None
                                            and before.region_id!=task.region_id)) if pose else
                      io.Connection(None,task.task_id,None,'FREE_START',0,0,0,0))
                calls+=1
                if link is not None:
                    candidates.append((cost+link.seconds,variant.end,path+[(variant,link)]))
            if candidates:
                following.append(min(candidates,key=lambda row:row[0]))
        if not following:
            return io.RegionRoute(region.region_id,'REGION_ROUTE_NOT_FOUND',
                order_mode=mode+'_DIRECTION_DP',explored_candidates=calls,
                reason='DIRECTION_DP_NO_CONNECTION_TO:'+task.task_id)
        states=following
        worked=unary_union([worked,variants[task.task_id][0].sweep.intersection(scene.target)])
        before=task
    cost,_,path=min(states,key=lambda row:row[0])
    result=io.RegionRoute(region.region_id,'REGION_ROUTE_PASS',
                           order_mode=mode+'_DIRECTION_DP',connection_seconds=cost,
                           explored_candidates=calls)
    for variant,link in path:
        if link.motion is not None:
            result.motions.append(link.motion);result.connections.append(link)
        result.motions.append(variant.motion);result.task_order.append(variant.task.task_id)
        result.sweep_error_m2+=variant.sweep_error_m2
    result.completed_sweep=unary_union([v.sweep for v,_ in path])
    result.reverse_m=sum(c.reverse_m for c in result.connections)
    result.gear_shifts=sum(c.gear_shifts for c in result.connections)
    return result


def _plan_field_once(job,settings,priority=(),lookahead=False,
                     medial_enabled=False):
    """在本田预算内执行一次严格排序/连接尝试，返回完整或部分结果，不能无限追加搜索。"""
    started=time.perf_counter()
    try:
        scene=io.load_scene(job.scene_path)
        if str(scene.crs)!=job.metric_crs or math.dist(scene.origin,job.origin)>1e-6:
            raise ValueError('SCENE_COORDINATE_MISMATCH')
        validator=io.Validator(scene)
        connector=Connector(scene,settings,started+settings.max_field_seconds,
                            medial_enabled=medial_enabled)
        field_deadline=connector.deadline
        required_body=None;preparation={}
        if settings.replan_headlands:
            pass  # 已合并到本模块，直接使用下方的定义。
            job,required_body,preparation=prepare(
                job,scene,connector,mode=settings.headland_strategy)
        variants={};invalid={}
        for task in job.tasks:
            variants[task.task_id],errors=io._task_variants(task,scene,validator)
            if not variants[task.task_id]:
                invalid[task.task_id]=errors
        adaptation=time.perf_counter()-started
        regions=list(job.regions);tasks={t.task_id:t for t in job.tasks}
        headland=unary_union([r.headland for r in regions])
        completed=GeometryCollection();worked_by_region={}
        previous=scene.start;previous_task=None
        routes=[];sequence=[];failures={};remaining=[r for r in regions if r.task_ids]
        progress_by_region={}
        for region in regions:
            if not region.task_ids:
                routes.append(io.RegionRoute(region.region_id,'HEADLAND_ONLY',
                                             reason='NO_BODY_TASKS_AFTER_HEADLAND_RESERVATION'))
        while remaining and time.perf_counter()<connector.deadline:
            chosen=None
            for region in sorted(remaining,key=lambda r:(0 if r.region_id in priority else 1,
                                                          r.sequence_index)):
                region_tasks=[tasks[t] for t in region.task_ids]
                if not region_tasks or any(t.task_id in invalid for t in region_tasks):
                    failures[region.region_id]='INPUT_TASK_INVALID';continue
                # A difficult first block must not consume every other block's
                # opportunity. Unused time can be used after another block has
                # produced additional worked space.
                now=time.perf_counter()
                share=len(region_tasks)/max(1,sum(len(r.task_ids) for r in remaining))
                connector.deadline=min(field_deadline,now+max(2.0,
                    (field_deadline-now)*max(0.25,share)))
                candidates=[];attempts=0
                for mode,ordered in orders(region_tasks,scene,settings,connector.backend,previous):
                    if time.perf_counter()>=connector.deadline:
                        break
                    max_progress=0
                    for first in variants[ordered[0].task_id]:
                        attempt=candidate_region(region,ordered,mode,variants,scene,connector,
                            headland,completed,worked_by_region,previous,previous_task,first.reversed)
                        attempts+=attempt.explored_candidates
                        max_progress=max(max_progress,len(attempt.task_order))
                        progress_by_region[region.region_id]=max(
                            progress_by_region.get(region.region_id,0),len(attempt.task_order))
                        if attempt.status=='REGION_ROUTE_PASS':
                            candidates.append(attempt)
                            if len(region_tasks)>8 and (not lookahead or len(candidates)>=2):
                                break
                        else:
                            failures[region.region_id]=attempt.reason
                    if (not candidates and max_progress>=max(2,len(ordered)*0.75)
                            and time.perf_counter()<connector.deadline):
                        dp=candidate_region_dp(region,ordered,mode,variants,scene,connector,
                            headland,completed,worked_by_region,previous,previous_task)
                        attempts+=dp.explored_candidates
                        if dp.status=='REGION_ROUTE_PASS':
                            candidates.append(dp)
                    # Feasibility first: additional candidates are searched only
                    # for short blocks. Report the actually explored minimum.
                    if candidates and len(region_tasks)>8 and (not lookahead or len(candidates)>=2):
                        break
                if candidates:
                    fastest=min(c.connection_seconds for c in candidates)
                    eligible=[c for c in candidates if c.connection_seconds<=fastest*(1.20 if lookahead else 1.05)+1e-9]
                    if lookahead:
                        successors=[tasks[tid] for other in remaining
                                    if other.region_id!=region.region_id
                                    for tid in other.task_ids]
                        def exit_cost(candidate):
                            end=io._pose_at(candidate.motions[-1],True)
                            gap=min((math.hypot(x-end.x,y-end.y)
                                     for task in successors
                                     for x,y in task.reference_line.coords),default=0.0)
                            return candidate.connection_seconds+gap/max(
                                scene.vehicle.turn_speed_mps,scene.vehicle.reverse_speed_mps)
                        best=min(eligible,key=lambda c:(exit_cost(c),c.gear_shifts,
                                                        c.reverse_m,c.order_mode))
                    else:
                        best=min(eligible,key=lambda c:(c.irregular_jumps,c.gear_shifts,c.reverse_m,
                                                        c.connection_seconds,c.order_mode))
                    best.explored_candidates=attempts
                    chosen=(region,best)
                    connector.deadline=field_deadline
                    break
                connector.deadline=field_deadline
            if chosen is None:
                break
            region,best=chosen
            routes.append(best);sequence.append(region.region_id)
            completed=unary_union([completed,best.completed_sweep])
            worked_by_region[region.region_id]=best.completed_sweep
            previous=io._pose_at(best.motions[-1],True)
            previous_task=tasks[best.task_order[-1]]
            remaining.remove(region)
        finish_failed=False
        if not remaining and scene.end is not None and routes:
            final=connector.connect(previous,scene.end,previous_task,None,
                io._ready_area(scene,headland,completed),worked_by_region,transfer=True)
            if final is None:
                finish_failed=True
            elif final.motion is not None:
                routes[-1].motions.append(final.motion);routes[-1].connections.append(final)
                routes[-1].connection_seconds+=final.seconds
        for region in remaining:
            reason=('SEARCH_BUDGET_EXHAUSTED' if time.perf_counter()>=connector.deadline
                    else failures.get(region.region_id,'NO_COMPLETE_REGION_ROUTE'))
            routes.append(io.RegionRoute(region.region_id,'REGION_ROUTE_NOT_FOUND',reason=reason))
        body_ok=not preparation or preparation['body_missing_m2']<=scene.settings.coverage_tolerance_m2
        status=('HEADLAND_ONLY' if not job.tasks else
                'FIELD_ROUTE_COMPLETE' if not remaining and not finish_failed and body_ok else
                'PARTIAL_CONTINUOUS_ROUTE' if sequence else 'NO_CONTINUOUS_ROUTE')
        result=io.FieldRoute(job.index,job.field_id,status,sequence,routes,
            [{'field_id':job.field_id,'region_id':r.region_id,'code':r.reason}
             for r in routes if r.status!='REGION_ROUTE_PASS'],
            time.perf_counter()-started,job.source_crs,job.metric_crs,job.origin,
            job.source_geometry,job.upstream_seam_quality_status,job.upstream_acceptance_passed,
            job.upstream_area_ledger_delta_m2)
        result.transfer_status=('ALL_REQUIRED_TRANSFERS_CONNECTED' if len(sequence)==len(regions)
                                else 'PARTIAL_OR_NOT_REACHED')
        if not remaining:
            result.transfer_status='ALL_REQUIRED_TRANSFERS_CONNECTED' if sequence else 'NOT_APPLICABLE'
        result.adapted_job=job
        result.required_body=required_body
        result.preparation=preparation
        if not body_ok:
            result.failures.append({'field_id':job.field_id,'region_id':'__BODY__',
                                    'code':'REPLANNED_BODY_COVERAGE_GAP',
                                    'area_m2':preparation['body_missing_m2']})
        result.statistics={'counts':dict(connector.counts),'stage_seconds':dict(connector.timing),
                           'adaptation_s':adaptation,'physical_cache_hits':connector.checker.cache_hits,
                           'max_progress_by_region':progress_by_region,
                           'search_variant':'FAILED_REGION_FIRST' if priority else 'DEFAULT'}
        if finish_failed:
            result.failures.append({'field_id':job.field_id,'region_id':sequence[-1],
                                    'code':'FIELD_END_UNREACHABLE'})
        return result
    except Exception as exc:
        return io._field_error(job,type(exc).__name__,str(exc),time.perf_counter()-started)


def _choose_attempt(job,attempts,started,triage):
    """先看完整连续覆盖，再比较不完整候选的有效进展和代价，失败不能因较短距离而变为通过。"""
    target=io.load_scene(job.scene_path).target if len(attempts)>1 else None
    def score(result):
        # 完整连续覆盖是硬门槛；比较未完成候选时看真正完成的目标扫掠，不能因选中的主体更小就认为质量更好。
        done=unary_union([r.completed_sweep for r in result.routes
                          if r.status=='REGION_ROUTE_PASS'])
        return (result.status=='FIELD_ROUTE_COMPLETE',
                done.intersection(target).area if target is not None else done.area,
                result.preparation.get('required_body_area_m2',0.0))
    chosen=max(attempts,key=score)
    records=[{
        'mode':candidate.preparation.get('reservation_mode','ERROR'),
        'search_variant':candidate.statistics.get('search_variant','DEFAULT'),
        'status':candidate.status,
        'body_area_m2':candidate.preparation.get('required_body_area_m2',0.0),
        'completed_task_count':sum(len(r.task_order) for r in candidate.routes),
        'elapsed_s':candidate.elapsed_s} for candidate in attempts]
    chosen.elapsed_s=time.perf_counter()-started
    chosen.preparation['headland_attempts']=records
    chosen.preparation['directional_triage']=triage
    return chosen


def _rescue_failed_region(job,settings,result,medial_enabled=False):
    """仅对符合条件的遗漏分区尝试有限恢复，保留原覆盖义务与来源结果。"""
    missing={r.region_id for r in result.routes if r.status=='REGION_ROUTE_NOT_FOUND'}
    if result.status not in {'PARTIAL_CONTINUOUS_ROUTE','NO_CONTINUOUS_ROUTE'} or not missing:
        return None
    priority=tuple(r.region_id for r in sorted(job.regions,
        key=lambda r:(-len(r.task_ids),r.sequence_index)) if r.region_id in missing)
    budget_limited=any('SEARCH_BUDGET_EXHAUSTED' in str(item.get('code',''))
                       for item in result.failures)
    retry_seconds=(max(settings.max_field_seconds,30.0)
                   if budget_limited else settings.max_field_seconds)
    return _plan_field_once(job,replace(settings,headland_strategy='UNIFORM',
                                        max_field_seconds=retry_seconds),
                            priority=priority,lookahead=True,
                            medial_enabled=medial_enabled)


def _replan_last_body_task(routes,sequence,blocks,job,scene,settings,
                           variants,headland,medial_enabled=False):
    """联合检查末条作业与下一固定区块出口方向，防止插入末条后无法重新接回原路线。
    
    Try a last row together with the following block's exit direction.
    
    A one-row insertion can enter a gap yet fail to rejoin the fixed next
    route. Replanning only that next block tests a different exit without
    disturbing earlier or more distant work. Every resulting link and motion
    still needs the same physical and dynamic checks as ordinary work."""
    if len(blocks)!=1 or len(blocks[0][1])!=1 or len(routes)<2:
        return None,{'status':'NOT_APPLICABLE'}
    started=time.perf_counter()
    connector=Connector(scene,settings,
                        started+settings.last_body_replan_seconds,
                        medial_enabled=medial_enabled)
    region,task=blocks[0][0],blocks[0][1][0]
    taskmap={item.task_id:item for item in job.tasks}
    regionmap={item.region_id:item for item in job.regions}
    options=variants[task.task_id]
    done=GeometryCollection();worked={};pose,before=scene.start,None
    boundaries=[]
    for index,route in enumerate(routes):
        first_work=next(motion for motion in route.motions
                        if motion.task_id==route.task_order[0])
        next_pose=io._pose_at(first_work,False)
        priority=min((math.hypot(option.start.x-pose.x,
                                 option.start.y-pose.y) if pose else 0.0)+
                     math.hypot(option.end.x-next_pose.x,
                                option.end.y-next_pose.y)
                     for option in options)
        boundaries.append((priority,index,done,dict(worked),pose,before))
        done=unary_union([done,route.completed_sweep])
        worked[route.region_id]=unary_union([
            worked.get(route.region_id,GeometryCollection()),
            route.completed_sweep])
        pose=io._pose_at(route.motions[-1],True)
        before=taskmap[route.task_order[-1]]
    entries=following_passes=exits=attempts=0
    for _,index,done,worked,pose,before in sorted(boundaries)[:8]:
        if time.perf_counter()>=connector.deadline:
            break
        original=routes[index]
        following_tasks=[taskmap[tid] for tid in original.task_order]
        for direction in sorted({option.reversed for option in options}):
            if time.perf_counter()>=connector.deadline:
                break
            attempts+=1
            inserted=candidate_region(region,[task],'REVISITED_BODY_ROW',
                {task.task_id:options},scene,connector,headland,
                done,worked,pose,before,direction)
            if inserted.status!='REGION_ROUTE_PASS':
                continue
            entries+=1
            done2=unary_union([done,inserted.completed_sweep])
            worked2=dict(worked)
            worked2[region.region_id]=unary_union([
                worked2.get(region.region_id,GeometryCollection()),
                inserted.completed_sweep])
            entry_pose=io._pose_at(inserted.motions[-1],True)
            for order in (following_tasks,list(reversed(following_tasks))):
                if time.perf_counter()>=connector.deadline:
                    break
                first_options=variants[order[0].task_id]
                for first_direction in sorted({v.reversed for v in first_options}):
                    if time.perf_counter()>=connector.deadline:
                        break
                    replanned=candidate_region(
                        regionmap[original.region_id],order,
                        'REVISITED_NEXT_BLOCK',variants,scene,connector,
                        headland,done2,worked2,entry_pose,task,first_direction)
                    if replanned.status!='REGION_ROUTE_PASS':
                        continue
                    following_passes+=1
                    trial=list(routes)
                    trial[index:index+1]=[inserted,replanned]
                    if index+1<len(routes):
                        later=routes[index+1]
                        first_id=later.task_order[0]
                        first_index=next(i for i,motion in enumerate(later.motions)
                                         if motion.task_id==first_id)
                        if first_index>1:
                            continue
                        last_task=taskmap[replanned.task_order[-1]]
                        later_task=taskmap[first_id]
                        done3=unary_union([done2,replanned.completed_sweep])
                        worked3=dict(worked2)
                        worked3[replanned.region_id]=unary_union([
                            worked3.get(replanned.region_id,GeometryCollection()),
                            replanned.completed_sweep])
                        link=connector.connect(
                            io._pose_at(replanned.motions[-1],True),
                            io._pose_at(later.motions[first_index],False),
                            last_task,later_task,
                            io._ready_area(scene,headland,done3),worked3,
                            transfer=last_task.region_id!=later_task.region_id)
                        if link is None:
                            continue
                        prefix=[link.motion] if link.motion is not None else []
                        links=[link] if link.motion is not None else []
                        links+=later.connections[1:] if first_index else later.connections
                        trial[index+2]=replace(later,
                            motions=prefix+later.motions[first_index:],
                            connections=links,
                            connection_seconds=sum(item.seconds for item in links),
                            reverse_m=sum(item.reverse_m for item in links),
                            gear_shifts=sum(item.gear_shifts for item in links))
                    exits+=1
                    if io._route_continuity_issues(
                            [motion for route in trial for motion in route.motions],scene):
                        continue
                    stats={'status':'IMPROVED','task_id':task.task_id,
                           'insertion_index':index,'attempts':attempts,
                           'entry_feasible':entries,
                           'following_replanned':following_passes,
                           'exit_feasible':exits,
                           'elapsed_s':time.perf_counter()-started,
                           'connector_counts':dict(connector.counts)}
                    return (trial,sequence[:index]+[region.region_id]+
                            sequence[index:]),stats
    return None,{'status':'NO_VALID_REPLAN','task_id':task.task_id,
                 'attempts':attempts,'entry_feasible':entries,
                 'following_replanned':following_passes,
                 'exit_feasible':exits,
                 'elapsed_s':time.perf_counter()-started,
                 'search_limited':time.perf_counter()>=connector.deadline,
                 'connector_counts':dict(connector.counts)}


def _work_corridor_groups(ordered, width):
    """按相邻平行行与真实走廊分组，孔洞造成的独立条带碎片不因同一行号而强行合并。
    
    Group adjacent parallel rows without joining across a hole.
    
    A row cut by several obstacles produces independent line fragments with
    the same row index. Consecutive-row fragments belong to one local work
    corridor only when their longitudinal intervals substantially overlap
    and their lengths have a comparable scale. Long rows above/below the
    obstacles do not bridge all of the narrow corridors into one block.
    This changes task *order groups*, never the original task geometries or
    region ownership."""
    if len(ordered) < 2:
        return [ordered]
    angle = ordered[0].heading_rad
    if any(abs(math.sin(task.heading_rad-angle)) > 0.12
           for task in ordered):
        return [ordered]
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    data = []
    rows = {}
    for index, task in enumerate(ordered):
        a = task.reference_line.coords[0]
        b = task.reference_line.coords[-1]
        longitudinal = sorted((a[0]*ux+a[1]*uy,
                               b[0]*ux+b[1]*uy))
        lateral = ((a[0]+b[0])*vx+(a[1]+b[1])*vy)/2
        length = longitudinal[1]-longitudinal[0]
        data.append((longitudinal[0], longitudinal[1], lateral, length))
        rows.setdefault(task.row_index, []).append(index)
    parent = list(range(len(ordered)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        a, b = root(left), root(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    row_keys = sorted(rows)
    for earlier, later in zip(row_keys, row_keys[1:]):
        # A changing fragment count marks an obstacle entering or leaving
        # the row. Keep that transition as a work-block boundary so a long
        # pass cannot join several distinct corridors around the obstacle.
        if len(rows[earlier]) != len(rows[later]):
            continue
        for i in rows[earlier]:
            lo_i, hi_i, y_i, length_i = data[i]
            for j in rows[later]:
                lo_j, hi_j, y_j, length_j = data[j]
                if (min(length_i, length_j) <= 1e-6
                        or abs(y_i-y_j) > 1.6*width
                        or max(length_i, length_j) >
                        2.2*min(length_i, length_j)):
                    continue
                overlap = max(0.0, min(hi_i,hi_j)-max(lo_i,lo_j))
                if overlap >= 0.45*min(length_i,length_j):
                    join(i,j)
    groups = {}
    for index, task in enumerate(ordered):
        groups.setdefault(root(index), []).append(task)
    return sorted(groups.values(),key=lambda group:(
        min(task.row_index for task in group),
        min(task.task_id for task in group)))


def _plan_field_blocks(job,settings,block_size=6,medial_enabled=False,
                       smallest_region_first=False,reservation_mode='UNIFORM'):
    """把各分区条带拆成可重访的连续块，在块间执行真实连接。

    每块完整通过后才计入已作业面积。分区编号和目标归属不变，路线可在
    不同分区的块之间往返；机具关闭时仍只能使用田头或已作业空间。
    """
    started=time.perf_counter()
    try:
        scene=io.load_scene(job.scene_path)
        if str(scene.crs)!=job.metric_crs or math.dist(scene.origin,job.origin)>1e-6:
            raise ValueError('SCENE_COORDINATE_MISMATCH')
        pass  # 已合并到本模块，直接使用下方的定义。
        connector=Connector(scene,settings,started+settings.max_field_seconds,
                            medial_enabled=medial_enabled)
        job,body,preparation=prepare(job,scene,connector,mode=reservation_mode)
        tasks={task.task_id:task for task in job.tasks}
        headland=unary_union([r.headland for r in job.regions])
        blocks=[]
        spatial_work_group_count=0
        for region in job.regions:
            ordered=sorted((tasks[tid] for tid in region.task_ids),
                           key=lambda t:(t.row_index,t.suggested_order,t.task_id))
            polygons=(region.geometry.geoms if
                      region.geometry.geom_type=='MultiPolygon' else
                      [region.geometry])
            hole_count=sum(len(polygon.interiors) for polygon in polygons)
            # Keep the normal fast route for modest fields. Spatial work
            # corridors pay for themselves when several holes split a large
            # number of rows into genuinely separate driving aisles.
            fragmented = len({task.row_index for task in ordered}) < len(ordered)
            groups=(_work_corridor_groups(ordered,scene.vehicle.working_width_m)
                    if ((hole_count>=3 and len(ordered)>=100) or
                        (settings.directional_block_continuation and
                         reservation_mode=='DIRECTIONAL' and
                         fragmented and len(ordered)>=8)) else [ordered])
            spatial_work_group_count+=len(groups)
            for group in groups:
                blocks.extend((region,group[i:i+block_size])
                              for i in range(0,len(group),block_size))
        if smallest_region_first:
            # Complete compact regions while their entry remains accessible;
            # long regions can then be visited in several connected blocks.
            # IDs, geometries and row order within each region do not change.
            blocks.sort(key=lambda item:(len(item[0].task_ids),
                                         item[0].sequence_index))
        validator=io.Validator(scene)
        variants={task.task_id:io._task_variants(task,scene,validator)[0]
                  for task in job.tasks}
        invalid=[task_id for task_id,items in variants.items() if not items]
        if invalid:
            raise ValueError('INVALID_DERIVED_TASK:'+','.join(invalid[:3]))
        adaptation=time.perf_counter()-started
        completed=GeometryCollection();worked_by_region={}
        previous=scene.start;previous_task=None
        routes=[];sequence=[];trials=0;failure='';refinements=0
        field_deadline=connector.deadline
        block_sweep_cache={}
        while blocks and time.perf_counter()<field_deadline:
            def block_priority(row):
                index,(_,block)=row
                if previous is None:
                    return (index,index)
                distance=min((math.hypot(variant.start.x-previous.x,
                                         variant.start.y-previous.y)
                              for task in block
                              for variant in variants[task.task_id]),default=0.0)
                key=tuple(task.task_id for task in block)
                if key not in block_sweep_cache:
                    block_sweep_cache[key]=unary_union(
                        [variants[task.task_id][0].sweep for task in block])
                gain=block_sweep_cache[key].intersection(body).difference(
                    completed).area
                useful_length=gain/max(scene.vehicle.working_width_m,1e-6)
                return ((10.0+distance)/(10.0+useful_length),index)
            ranked=sorted(enumerate(blocks),key=block_priority)
            chosen=None
            for index,(region,block) in ranked:
                if time.perf_counter()>=field_deadline:
                    break
                connector.deadline=min(field_deadline,time.perf_counter()+max(
                    1.0,(field_deadline-time.perf_counter())/max(1,len(blocks))))
                patterns = ([sequence for _,sequence in orders(
                    block,scene,settings,connector.backend,entry=previous)]
                    if settings.directional_block_continuation and
                       reservation_mode=='DIRECTIONAL' else
                    [block,list(reversed(block))])
                for direction in patterns:
                    for first in variants[direction[0].task_id]:
                        candidate=candidate_region(region,direction,'BOUNDED_ROW_BLOCK',
                            variants,scene,connector,headland,completed,
                            worked_by_region,previous,previous_task,first.reversed)
                        trials+=1
                        if candidate.status=='REGION_ROUTE_PASS':
                            chosen=(index,region,candidate)
                            break
                        failure=candidate.reason
                    if chosen or time.perf_counter()>=connector.deadline:
                        break
                if chosen:
                    break
            connector.deadline=field_deadline
            if not chosen:
                # A failed six-row block is not proof that every row in it is
                # unreachable. Keep the completed itinerary and split only a
                # failed block into consecutive smaller work groups. Each new
                # group still needs its own real, fully checked connection.
                splittable=next(((index,region,block)
                    for index,(region,block) in ranked if len(block)>1),None)
                if (splittable is not None and refinements<min(12,len(job.tasks))
                        and time.perf_counter()+0.5<field_deadline):
                    index,region,block=splittable
                    middle=len(block)//2
                    blocks[index:index+1]=[(region,block[:middle]),
                                           (region,block[middle:])]
                    refinements+=1
                    continue
                break
            index,region,candidate=chosen
            routes.append(candidate);sequence.append(region.region_id)
            completed=unary_union([completed,candidate.completed_sweep])
            worked_by_region[region.region_id]=unary_union([
                worked_by_region.get(region.region_id,GeometryCollection()),
                candidate.completed_sweep])
            previous=io._pose_at(candidate.motions[-1],True)
            previous_task=tasks[candidate.task_order[-1]]
            blocks.pop(index)
        replan_stats={'status':'NOT_NEEDED'}
        if len(blocks)==1 and len(blocks[0][1])==1 and routes:
            replacement,replan_stats=_replan_last_body_task(
                routes,sequence,blocks,job,scene,settings,variants,
                headland,medial_enabled)
            if replacement is not None:
                routes,sequence=replacement
                blocks.clear()
                completed=unary_union([route.completed_sweep for route in routes])
                worked_by_region={}
                for route in routes:
                    worked_by_region[route.region_id]=unary_union([
                        worked_by_region.get(route.region_id,GeometryCollection()),
                        route.completed_sweep])
                previous=io._pose_at(routes[-1].motions[-1],True)
                previous_task=tasks[routes[-1].task_order[-1]]
        finish_failed=False
        if not blocks and scene.end is not None and routes:
            final=connector.connect(previous,scene.end,previous_task,None,
                io._ready_area(scene,headland,completed),worked_by_region,
                transfer=True)
            if final is None:
                finish_failed=True
            elif final.motion is not None:
                routes[-1].motions.append(final.motion)
                routes[-1].connections.append(final)
                routes[-1].connection_seconds+=final.seconds
        if blocks:
            reason=('SEARCH_BUDGET_EXHAUSTED' if time.perf_counter()>=field_deadline
                    else failure or 'NO_COMPLETE_ROW_BLOCK')
            for region_id in sorted({region.region_id for region,_ in blocks}):
                routes.append(io.RegionRoute(region_id,'REGION_ROUTE_NOT_FOUND',
                                             reason=reason))
        body_missing=body.difference(completed).area
        body_ok=body_missing<=scene.settings.coverage_tolerance_m2
        status=('HEADLAND_ONLY' if not job.tasks else
                'FIELD_ROUTE_COMPLETE' if not blocks and not finish_failed and body_ok else
                'PARTIAL_CONTINUOUS_ROUTE' if sequence else 'NO_CONTINUOUS_ROUTE')
        result=io.FieldRoute(job.index,job.field_id,status,sequence,routes,
            [{'field_id':job.field_id,'region_id':route.region_id,'code':route.reason}
             for route in routes if route.status!='REGION_ROUTE_PASS'],
            time.perf_counter()-started,job.source_crs,job.metric_crs,job.origin,
            job.source_geometry,job.upstream_seam_quality_status,
            job.upstream_acceptance_passed,job.upstream_area_ledger_delta_m2)
        if not body_ok:
            result.failures.append({'field_id':job.field_id,'region_id':'__BODY__',
                                    'code':'REPLANNED_BODY_COVERAGE_GAP',
                                    'area_m2':body_missing})
        if finish_failed:
            result.failures.append({'field_id':job.field_id,'region_id':sequence[-1],
                                    'code':'FIELD_END_UNREACHABLE'})
        result.transfer_status=('ALL_REQUIRED_TRANSFERS_CONNECTED'
            if not blocks and not finish_failed else 'PARTIAL_OR_NOT_REACHED')
        result.adapted_job=job;result.required_body=body;result.preparation=preparation
        result.statistics={'counts':dict(connector.counts),
                           'stage_seconds':dict(connector.timing),
                           'adaptation_s':adaptation,
                           'physical_cache_hits':connector.checker.cache_hits,
                           'search_variant':'MULTI_REGION_ROW_BLOCKS',
                           'block_size':block_size,'block_trials':trials,
                           'block_seed_order':('SMALLEST_REGION_FIRST' if
                               smallest_region_first else 'SOURCE_REGION_ORDER'),
                           'spatial_work_group_count':spatial_work_group_count,
                           'completed_block_count':len(sequence),
                           'adaptive_split_count':refinements,
                           'final_block_replan':replan_stats,
                           'remaining_block_count':len(blocks)}
        return result
    except Exception as exc:
        return io._field_error(job,type(exc).__name__,str(exc),time.perf_counter()-started)


def _append_block_rescue(job,settings,attempts,medial_enabled=False):
    """对大任务集尝试有限区块恢复；只向候选集合追加真实结果，不改写失败标签。"""
    if (len(job.tasks)>16 and len(job.regions)<=8 and len(job.tasks)<=240
            and all(item.status!='FIELD_ROUTE_COMPLETE' for item in attempts)):
        block_size=6;smallest_region_first=False
        if len(job.regions)>1:
            progress=max((sum(len(route.task_order) for route in item.routes)
                /max(1,len(item.adapted_job.tasks))
                for item in attempts if item.adapted_job is not None),default=0.0)
            if progress<0.70:
                # A very large region can repeatedly approach its far rows
                # without ever committing them: the all-or-nothing region
                # candidate is discarded on the first later failure. Retry
                # only when a real budget failure and several regions make
                # revisitable blocks useful. Small difficult fields keep the
                # existing fast path rather than paying another minute.
                adapted=next((item.adapted_job for item in attempts
                              if item.adapted_job is not None),None)
                sizes=([len(region.task_ids) for region in adapted.regions
                        if region.task_ids] if adapted is not None else [])
                budget_limited=any('SEARCH_BUDGET_EXHAUSTED' in
                    str(failure.get('code','')) for item in attempts
                    for failure in item.failures)
                if (len(sizes)==2 and 50<=len(job.tasks)<100
                        and max(sizes)>=25 and budget_limited):
                    # Spend another block search only when an actual large
                    # body area remains untouched. This catches two-region
                    # fields whose all-or-nothing search discards many legal
                    # rows, without charging every modest field a minute.
                    def executed_body_gap(item):
                        if item.required_body is None:
                            return math.inf
                        done=unary_union([route.completed_sweep for route in item.routes
                                          if route.status=='REGION_ROUTE_PASS'])
                        return item.required_body.difference(done).area
                    if min(map(executed_body_gap,attempts))>=5_000.0:
                        attempts.append(_plan_field_blocks(job,replace(settings,
                            headland_strategy='UNIFORM',
                            max_field_seconds=max(60.0,
                                                  settings.max_field_seconds*2)),
                            block_size=6,smallest_region_first=True,
                            medial_enabled=medial_enabled))
                    return
                if not (len(sizes)>=3 and max(sizes,default=0)>=40
                        and budget_limited):
                    return
                block_size=max(6,min(12,math.ceil(max(sizes)/10)))
                smallest_region_first=True
        attempts.append(_plan_field_blocks(job,replace(settings,
            headland_strategy='UNIFORM',max_field_seconds=max(60.0,
                                                     settings.max_field_seconds*2)),
                                           block_size=block_size,
                                           smallest_region_first=smallest_region_first,
                                           medial_enabled=medial_enabled))


def _append_directional_block_rescue(job,settings,attempts,medial_enabled=False):
    """对符合任务规模的遗漏工作核心尝试顺条带方向的田头/区块组织，保持主体义务不变。
    
    Retry large omitted work cores with strip-aligned headlands and blocks.
    
    A uniform turn reserve can erase every body row in one small region while
    leaving another region's long row sequence unfinished. Replacing only the
    reserve does not solve the latter: the new rows also need revisitable
    blocks and a checked connection between blocks. This bounded retry is
    reserved for fields where the *executed* body gap is substantial."""
    if (not 3<=len(job.regions)<=6 or not 50<=len(job.tasks)<=120
            or any(item.status=='FIELD_ROUTE_COMPLETE' for item in attempts)):
        return
    eligible=False
    for item in attempts:
        if (item.preparation.get('reservation_mode')!='UNIFORM'
                or item.required_body is None or item.adapted_job is None):
            continue
        vanished=any(region.task_ids==() for region in item.adapted_job.regions)
        if not vanished:
            continue
        done=unary_union([route.completed_sweep for route in item.routes
                          if route.status=='REGION_ROUTE_PASS'])
        if item.required_body.difference(done).area>=5_000.0:
            eligible=True
            break
    if not eligible:
        return
    retry_settings=replace(settings,headland_strategy='DIRECTIONAL',
        max_field_seconds=max(60.0,settings.max_field_seconds*2))
    for size in (6,3):
        result=_plan_field_blocks(job,retry_settings,block_size=size,
            smallest_region_first=True,medial_enabled=medial_enabled,
            reservation_mode='DIRECTIONAL')
        attempts.append(result)
        if result.status=='FIELD_ROUTE_COMPLETE':
            break


def _append_compact_directional_block_rescue(job,settings,attempts,
                                              medial_enabled=False):
    """对紧凑两区尝试保留被统一转向预留挤掉的主体；只有找到并复核运动才认可恢复。
    
    Keep a compact two-region body when uniform turns erase most of it.
    
    Whole-region search may abandon all of an otherwise workable region at a
    late connection.  Revisit the two regions in small checked blocks only
    when the directional reserve retains substantially more body than the
    uniform reserve.  This avoids charging ordinary small fields another
    block search and never treats a smaller required body as coverage gain."""
    if len(job.regions)!=2 or not 16<len(job.tasks)<=40:
        return
    if any(item.status=='FIELD_ROUTE_COMPLETE' and (
            not settings.directional_block_continuation or
            item.preparation.get('reservation_mode')=='DIRECTIONAL')
           for item in attempts):
        return
    directional=[item for item in attempts
                 if item.preparation.get('reservation_mode')=='DIRECTIONAL'
                 and item.required_body is not None]
    uniform=[item for item in attempts
             if item.preparation.get('reservation_mode')=='UNIFORM'
             and item.required_body is not None]
    if not directional or not uniform:
        return
    preserved=max(item.required_body.area for item in directional)
    erased=min(item.required_body.area for item in uniform)
    retention_ratio = 1.25 if settings.directional_block_continuation else 4.0
    if preserved<1_000.0 or preserved<retention_ratio*max(erased,1.0):
        return
    attempts.append(_plan_field_blocks(job,replace(settings,
        headland_strategy='DIRECTIONAL',
        max_field_seconds=max(60.0,settings.max_field_seconds*2)),
        block_size=6,smallest_region_first=False,
        reservation_mode='DIRECTIONAL',medial_enabled=medial_enabled))


def _plan_field_body(job,settings,medial_enabled=False):
    """按连接家族分别执行既有主体搜索，田头策略及有界预算由设置控制。
    
    Run the established body search with one connector family at a time."""
    if settings.headland_strategy not in {'AUTO','AUTO_FAST'} or not settings.replan_headlands:
        return _plan_field_once(job,settings,medial_enabled=medial_enabled)
    started=time.perf_counter()
    if settings.headland_strategy=='AUTO_FAST':
        # Extra side coverage is most often connectable in a compact, holeless
        # block.  Route safety, not this cheap triage, still decides acceptance.
        source=job.source_geometry
        components=source.geoms if source.geom_type=='MultiPolygon' else [source]
        hole_count=sum(len(poly.interiors) for poly in components)
        if hole_count or len(job.regions)>2 or len(job.tasks)>40:
            # Several holes can split a large region into distinct aisles.
            # The ordinary whole-region candidate repeatedly discards all
            # progress at the first failed late connection. Give the bounded
            # corridor-block scheduler the first budget in this shape class;
            # retain the established region search as a fallback if needed.
            corridor_first = None
            largest=max((len(region.task_ids) for region in job.regions),
                        default=0)
            # A very large region is also all-or-nothing when there are only
            # two regions and no holes.  If a late row cannot be joined, the
            # ordinary candidate discards every earlier checked row.  Start
            # with bounded revisitable blocks for this general shape class.
            large_multiregion=(len(job.regions)>=2 and len(job.tasks)>=100
                               and largest>=40)
            if ((hole_count>=3 and len(job.tasks)>=100
                 and len(job.regions)>=3) or large_multiregion):
                corridor_first=_plan_field_blocks(job,replace(settings,
                    headland_strategy='UNIFORM',
                    max_field_seconds=max(60.0,settings.max_field_seconds*2)),
                    block_size=max(6,min(12,math.ceil(largest/10))),
                    smallest_region_first=True,
                    medial_enabled=medial_enabled)
                if corridor_first.status=='FIELD_ROUTE_COMPLETE':
                    return _choose_attempt(job,[corridor_first],started,
                                           'CORRIDOR_BLOCKS_FIRST')
            uniform=_plan_field_once(job,replace(settings,headland_strategy='UNIFORM'),
                                     medial_enabled=medial_enabled)
            attempts=([corridor_first,uniform] if corridor_first is not None
                      else [uniform])
            # The fast triage normally skips directional reservation on
            # complex fields.  A zero-area body is a special case: a turn
            # reserve around every edge has erased all main work, so let the
            # strip-aligned reserve recover a body before accepting a
            # headland-only result.  The existing attempt selector still
            # checks the actual connected sweep before using the rescue.
            if (uniform.status == 'HEADLAND_ONLY'
                    and uniform.preparation.get('required_body_area_m2',0.0)
                    <= io.load_scene(job.scene_path).settings.coverage_tolerance_m2):
                attempts.append(_plan_field_once(job,replace(
                    settings,headland_strategy='DIRECTIONAL'),
                    medial_enabled=medial_enabled))
            elif (uniform.status != 'FIELD_ROUTE_COMPLETE'
                    and uniform.adapted_job is not None
                    and 0 < len(uniform.adapted_job.tasks) <= 3
                    and len(job.tasks) > len(uniform.adapted_job.tasks)):
                # A nearly erased core can leave a few disconnected row
                # fragments.  Rebuild a strip-aligned core, then commit only
                # individually connected work passes; the region may be
                # revisited through headland or already worked space.
                attempts.append(_plan_field_blocks(job,settings,
                    block_size=1,medial_enabled=medial_enabled,
                    reservation_mode='DIRECTIONAL'))
            if (uniform.status != 'FIELD_ROUTE_COMPLETE'
                    and uniform.adapted_job is not None
                    and 3 < len(uniform.adapted_job.tasks) <= 40
                    and not any(item.status == 'FIELD_ROUTE_COMPLETE'
                                for item in attempts)):
                # A field can contain several feasible work groups but no
                # feasible single-pass region itinerary.  Before declaring
                # total failure, let the whole-field scheduler revisit each
                # region through checked headland or completed work.  Cap the
                # task count so this rescue stays cheaper than a general
                # search over every row permutation.
                attempts.append(_plan_field_blocks(job,replace(settings,
                    max_field_seconds=max(60.0,settings.max_field_seconds*2)),
                    block_size=1,medial_enabled=medial_enabled))
            if (len(job.regions)>1
                    and not any(item.status == 'FIELD_ROUTE_COMPLETE'
                                for item in attempts)):
                rescue=_rescue_failed_region(job,settings,uniform,medial_enabled)
                if rescue is not None:
                    attempts.append(rescue)
            if corridor_first is None:
                _append_block_rescue(job,settings,attempts,medial_enabled)
            _append_directional_block_rescue(job,settings,attempts,medial_enabled)
            return _choose_attempt(job,attempts,started,'SKIPPED_COMPLEX_OR_LARGE')
    first=_plan_field_once(job,replace(settings,headland_strategy='DIRECTIONAL'),
                           medial_enabled=medial_enabled)
    attempts=[first]
    if first.status!='FIELD_ROUTE_COMPLETE':
        uniform=_plan_field_once(job,replace(settings,headland_strategy='UNIFORM'),
                                 medial_enabled=medial_enabled)
        attempts.append(uniform)
        if uniform.status!='FIELD_ROUTE_COMPLETE' and len(job.regions)>1:
            rescue=_rescue_failed_region(job,settings,uniform,medial_enabled)
            if rescue is not None:
                attempts.append(rescue)
    _append_compact_directional_block_rescue(
        job,settings,attempts,medial_enabled)
    _append_block_rescue(job,settings,attempts,medial_enabled)
    return _choose_attempt(job,attempts,started,'ATTEMPTED')


# 严格策略的单田组织入口，依次组织主体、分区转场和田头任务。
# 是否生成独立田头或联合田头，由路线配置决定，不修改冻结输入。
def plan_field(job,settings):
    """历史严格整田路线分发：依据田头模式选择主体、独立圈或联合组织，不等于当前几何参考入口。"""
    if settings.plan_headland_work:
        if settings.headland_work_mode == 'SEPARATE_PASSES':
            pass  # 已合并到本模块，直接使用下方的定义。
            return plan_separate(job, settings)
        pass  # 已合并到本模块，直接使用下方的定义。
        effective=settings
        components=(job.source_geometry.geoms
                    if job.source_geometry.geom_type=='MultiPolygon'
                    else [job.source_geometry])
        exterior_vertices=sum(len(component.exterior.coords)
                              for component in components)
        if settings.adaptive_headland_simplification and len(job.tasks)>=100:
            # Dense *outer* field edges split the headland into hundreds of
            # tiny pieces. Holes with many circle vertices are excluded from
            # this trigger; the 20-field benchmark shows that globally using
            # 1 m can worsen otherwise simple fields. Static and connected
            # motion checks still decide whether any simplified pass is safe.
            if exterior_vertices>=100:
                effective=replace(settings,
                    headland_simplification_tolerance_m=max(
                        1.0,settings.headland_simplification_tolerance_m))
        result=plan_joint(job,effective)
        result.preparation['effective_headland_simplification_tolerance_m']=(
            effective.headland_simplification_tolerance_m)
        return result
    started=time.perf_counter()
    baseline=_plan_field_body(job,settings)
    # Most fields already have a complete body chain.  Do not spend their
    # headland or body budgets on a multi-bend transfer they do not need.
    if baseline.status in {'FIELD_ROUTE_COMPLETE','HEADLAND_ONLY'}:
        return baseline
    if baseline.statistics.get('search_variant')=='MULTI_REGION_ROW_BLOCKS':
        return baseline
    if sum(bool(region.task_ids) for region in job.regions)<2:
        return baseline
    # A time-limited search has not isolated a failed transfer. Repeating the
    # whole field with a new connector spends time on the wrong failure mode;
    # reserve this retry for a concrete legal-connection counterexample.
    failure_codes=[str(failure.get('code','')) for failure in baseline.failures]
    if (not any('NO_LEGAL_CONNECTION_TO:' in code for code in failure_codes)
            or any('SEARCH_BUDGET_EXHAUSTED' in code for code in failure_codes)):
        return baseline
    retry=_plan_field_body(job,settings,medial_enabled=True)
    scene=io.load_scene(job.scene_path)
    def score(result):
        done=unary_union([r.completed_sweep for r in result.routes
                          if r.status=='REGION_ROUTE_PASS'])
        return (result.status=='FIELD_ROUTE_COMPLETE',
                done.intersection(scene.target).area,
                result.preparation.get('required_body_area_m2',0.0))
    chosen=max((baseline,retry),key=score)
    chosen.elapsed_s=time.perf_counter()-started
    chosen.preparation['medial_retry']={
        'baseline_status':baseline.status,
        'baseline_completed_tasks':sum(len(r.task_order) for r in baseline.routes),
        'retry_status':retry.status,
        'retry_completed_tasks':sum(len(r.task_order) for r in retry.routes),
        'selected':'RETRY' if chosen is retry else 'BASELINE'}
    return chosen

# ==========================================================================
# 2. 田头任务、补作候选与覆盖保护
# 从真实外边界和孔洞建立任务；新增候选必须保留必需目标及真实扫掠账本。
# ==========================================================================

from dataclasses import replace
import math
import warnings

import numpy as np
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.ops import unary_union

import route_planner as io
from validator import conservative_tool_coverage as headland_conservative_tool_coverage
from scene import Motion as headland_Motion
from scene import polygons as headland_polygons


headland_HEADLAND_REGION_ID = "__HEADLAND__"


def _polygons(geometry):
    """提取当前几何的面分量，空对象返回空列表。"""
    if geometry.is_empty:
        return []
    return list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]


def _long_straight_edges(ring, minimum_length, simplification_tolerance_m=0.25):
    # GEOS makes a short vertex chain around each re-entrant corner when the
    # native headland is inset.  Subdividing the *whole* boundary because of
    # one such chain turns a long L-shaped edge into dozens of 8 m jobs.  Keep
    # the original long sides as one job each; the tiny corner remainder stays
    # visible in the target-area ledger.
    """从简化轮廓提取足够长的直边，单位米；只产生候选支撑，不把简化边当真实禁入边界。"""
    simplified = ring.simplify(simplification_tolerance_m,
                               preserve_topology=False)
    points = list(simplified.coords)
    edges = [LineString([a, b]) for a, b in zip(points, points[1:])]
    long_edges = [edge for edge in edges if edge.length >= minimum_length]
    if (long_edges and max(edge.length for edge in long_edges) >= 3*minimum_length
            and sum(edge.length for edge in long_edges) >= 0.6*ring.length):
        return long_edges
    return []


def _hole_motion(ring, task_id):
    """沿平滑偏移孔洞环生成前进作业运动，采样朝向连续，仍须检查真实包络和覆盖。
    
    Follow a smooth offset ring with a continuous, forward work motion."""
    count = max(32, math.ceil(ring.length/0.5))
    core = np.asarray([ring.interpolate(i*ring.length/count).coords[0]
                       for i in range(count)], dtype=float)
    lag = max(1, round(1.5/(ring.length/count)))
    tangent = np.roll(core, -lag, axis=0)-np.roll(core, lag, axis=0)
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    rows = np.column_stack((core, yaw, np.ones(count)))
    return headland_Motion(np.vstack((rows, rows[0])), "work", task_id, True)


# 从真实边界构造田头作业任务，分区公共边界不当成新的真实田界。
# 每个任务带实际参考线与扫掠；后续覆盖保护按实际选中方案记账。
def generate_headland_tasks(job, scene, headland, backend, reservation,
             simplification_tolerance_m=0.25,
             positioning_allowance_m=0.2, contour_connector=None,
             build_reverse_contours=False, ring_count_override=None):
    """生成田头任务及诊断，同时保留原target与冻结主体任务；预留参数是配置假设，不是实测。
    
    Return an augmented job and diagnostics, preserving the original target.
    
    A configurable positioning allowance is added to the configured travel
    and machine safety clearances. It changes candidate placement only, never
    the frozen collision rules. Edge-end trim derives from actual vehicle and
    implement lengths. Missing headland work remains explicit."""
    width = scene.vehicle.working_width_m
    bias = (scene.settings.travel_clearance_m+
            scene.vehicle.safety_margin_m+positioning_allowance_m)
    trim = math.ceil(max(scene.vehicle.front_m/2,
                         scene.vehicle.rear_m/2,
                         scene.vehicle.implement_length_m/2,
                         scene.settings.sampling_step_m*2)*2)/2
    if ring_count_override is not None:
        if type(ring_count_override) is not int or ring_count_override not in (2,3):
            raise ValueError('独立田头圈数只允许整数 2/3')
        ring_count = ring_count_override
    else:
        ring_count = max(1, math.ceil(reservation["headland_width_m"]/width))
    native = backend.f2c.HG_Const_gen().generateHeadlandSwaths(
        backend.cells(scene.target), width, ring_count)
    validator = io.Validator(scene)
    tasks = []
    rejected = []
    candidate_covered = GeometryCollection()
    redundant_access_tasks = []
    for ring_index, cells in enumerate(native):
        candidate = wkt.loads(cells.exportToWkt()).buffer(-bias)
        for polygon_index, polygon in enumerate(_polygons(candidate)):
            outer_edges = _long_straight_edges(
                polygon.exterior, 2*trim+1.0, simplification_tolerance_m)
            if not outer_edges:
                task_id = f"headland_outer_ring_{ring_index:02d}_{polygon_index}"
                motion = _hole_motion(polygon.exterior, task_id)
                issues = validator.motion_issues(motion)
                if issues:
                    rejected.append({"task_id":task_id,
                                     "reason":issues[0]["code"]})
                else:
                    sweep = headland_conservative_tool_coverage(motion, scene)
                    novel = sweep.intersection(headland).difference(candidate_covered).area
                    if novel >= 1.0:
                        path = LineString(motion.points[:, :2])
                        tasks.append(io.FrozenTask(job.field_id, headland_HEADLAND_REGION_ID,
                            task_id, ring_index, polygon_index, path,
                            float(motion.points[0, 2]), sweep, path.length,
                            work_kind="CURVED_HEADLAND",
                            motion_points=motion.points.copy()))
                        candidate_covered = candidate_covered.union(sweep)
            for edge_index, edge in enumerate(outer_edges):
                if edge.length <= 2*trim+1.0:
                    continue
                line = LineString([edge.interpolate(trim),
                                   edge.interpolate(edge.length-trim)])
                if line.length < 1.0:
                    continue
                a, b = line.coords[0], line.coords[-1]
                heading = math.atan2(b[1]-a[1], b[0]-a[0])
                task_id = f"headland_ring_{ring_index:02d}_{polygon_index}_edge_{edge_index:03d}"
                task = io.FrozenTask(job.field_id, headland_HEADLAND_REGION_ID,
                    task_id, ring_index, edge_index, line, heading,
                    GeometryCollection(), line.length,
                    work_kind="STRAIGHT_HEADLAND")
                motion = io._work_motion(task, scene, False)
                issues = validator.motion_issues(motion)
                if issues:
                    rejected.append({"task_id":task_id,
                                     "reason":issues[0]["code"]})
                    continue
                sweep = validator.work_sweep(motion)
                novel = sweep.intersection(headland).difference(candidate_covered).area
                if novel < 1.0:
                    # An inner ring can be the only reachable approach pose
                    # for a later outer pass. Keep it as a provisional access
                    # work candidate and report its redundancy explicitly.
                    redundant_access_tasks.append(task_id)
                tasks.append(replace(task, frozen_sweep=sweep))
                candidate_covered = candidate_covered.union(sweep)
            for hole_index, hole in enumerate(polygon.interiors):
                # A polygonal hole has genuine straight work sides.  Forcing
                # one continuously turning loop across its corners produces
                # a lateral jump/curvature violation, so use separate work
                # tasks and let the connector make the actual turns.
                straight_hole_edges = _long_straight_edges(
                    hole, 2*trim+1.0, simplification_tolerance_m)
                if straight_hole_edges:
                    for edge_index, edge in enumerate(straight_hole_edges):
                        if edge.length <= 2*trim+1.0:
                            continue
                        line = LineString([edge.interpolate(trim),
                                           edge.interpolate(edge.length-trim)])
                        a, b = line.coords[0], line.coords[-1]
                        heading = math.atan2(b[1]-a[1], b[0]-a[0])
                        task_id = (f"headland_hole_ring_{ring_index:02d}_"
                                   f"{polygon_index}_{hole_index}_edge_{edge_index:03d}")
                        task = io.FrozenTask(job.field_id, headland_HEADLAND_REGION_ID,
                            task_id, ring_index, hole_index*100+edge_index,
                            line, heading, GeometryCollection(), line.length,
                            work_kind="STRAIGHT_HEADLAND")
                        motion = io._work_motion(task, scene, False)
                        issues = validator.motion_issues(motion)
                        if issues:
                            rejected.append({"task_id":task_id,
                                             "reason":issues[0]["code"]})
                            continue
                        sweep = validator.work_sweep(motion)
                        if sweep.intersection(headland).difference(candidate_covered).area < 1.0:
                            continue
                        tasks.append(replace(task, frozen_sweep=sweep))
                        candidate_covered = candidate_covered.union(sweep)
                    continue
                task_id = f"headland_hole_ring_{ring_index:02d}_{polygon_index}_{hole_index}"
                motion = _hole_motion(hole, task_id)
                issues = validator.motion_issues(motion)
                if issues:
                    rejected.append({"task_id":task_id,
                                     "reason":issues[0]["code"]})
                    continue
                sweep = headland_conservative_tool_coverage(motion, scene)
                novel = sweep.intersection(headland).difference(candidate_covered).area
                if novel < 1.0:
                    continue
                path = LineString(motion.points[:, :2])
                tasks.append(io.FrozenTask(job.field_id, headland_HEADLAND_REGION_ID,
                    task_id, ring_index, hole_index, path,
                    float(motion.points[0, 2]), sweep, path.length,
                    work_kind="CURVED_HEADLAND",
                    motion_points=motion.points.copy()))
                candidate_covered = candidate_covered.union(sweep)
    source_task_count = len(tasks)
    contour_records = []
    if contour_connector is not None:
        pass  # 已合并到本模块，直接使用下方的定义。
        tasks, contour_records = consolidate(tasks,scene,contour_connector,
                                            build_reverse=build_reverse_contours)
        # 面积账本依据实际连续作业扫掠；吸收原直线任务前必须确认其冻结扫掠仍被覆盖。
        candidate_covered = unary_union([task.frozen_sweep for task in tasks])
    region = io.RegionInput(headland_HEADLAND_REGION_ID, len(job.regions), headland,
                            headland, tuple(task.task_id for task in tasks))
    adjusted = replace(job, regions=(*job.regions, region),
                       tasks=(*job.tasks, *tasks))
    diagnostic = {"headland_task_count":len(tasks),
        "headland_source_task_count":source_task_count,
        "headland_contour_chains":contour_records,
        "headland_straight_task_count":sum(t.work_kind=="STRAIGHT_HEADLAND" for t in tasks),
        "headland_curved_task_count":sum(t.work_kind=="CURVED_HEADLAND" for t in tasks),
        "headland_candidate_missing_m2":headland.difference(candidate_covered).area,
        "headland_candidate_covered_m2":candidate_covered.intersection(headland).area,
        "headland_candidate_rejected":rejected,
        "redundant_access_work_task_ids":redundant_access_tasks,
        "headland_ring_count":len(native),
        "headland_edge_bias_m":bias,
        "headland_positioning_allowance_m":positioning_allowance_m,
        "headland_edge_trim_m":trim,
        "headland_simplification_tolerance_m":simplification_tolerance_m,
        "headland_task_mapping":[{"task_id":t.task_id,
                                  "source_task_id":None,
                                  "region_id":headland_HEADLAND_REGION_ID,
                                  "construction":t.work_kind,
                                  "source_task_ids":next((record["source_task_ids"]
                                      for record in contour_records
                                      if record["task_id"]==t.task_id),[])} for t in tasks]}
    return adjusted, diagnostic


# 残余凹湾和真实外边界的有界补作候选

def propose_bay_tasks(scene, remaining_target, completed, *, max_tasks=12):
    """在狭长残余目标内提出直线补作并验证，几何最小规模用于排除普通角点碎片。
    
    Return physically valid straight-work candidates in elongated gaps.
    
    Minimum rectangle aspect ratio and size exclude ordinary corner slivers.
    A candidate's exact swept polygon must add useful target coverage. Static
    safety is necessary but never treated here as proof of route reachability."""
    vehicle = scene.vehicle
    width = vehicle.working_width_m
    spacing = width * (1 - scene.settings.overlap_fraction)
    validator = io.Validator(scene)
    proposed = []
    proposed_sweeps = GeometryCollection()
    components = sorted(headland_polygons(remaining_target), key=lambda part: (
        -round(part.area, 6), *(round(value, 6) for value in part.bounds)))
    supports = list(components)
    for component in components:
        # A wide omitted work bay may be attached to a hairline around the
        # entire field.  Analysing that whole connected polygon hides the
        # bay's long axis. Remove only the sub-width fringe to identify its
        # broad cores, then restore the supporting bay around each core.
        core = component.buffer(-width / 2)
        for part in headland_polygons(core):
            if part.area >= max(20.0, 2 * width * width):
                supports.append(part.buffer(width / 2).intersection(component))
    for component_index, component in enumerate(supports):
        # Existing long-bay proposals already work well in fields such as
        # E04.  Use opened cores only when the original connected components
        # yielded no proposal; they are a rescue for fringe-attached bays.
        if component_index>=len(components) and proposed:
            break
        if len(proposed) >= max_tasks:
            break
        if component.area < max(100.0, 4 * width * width):
            continue
        # GEOS may emit an oriented-envelope floating-point warning for a
        # valid polygon with an extremely thin boundary sliver. Inspect the
        # resulting rectangle explicitly rather than treating the warning as
        # either a route failure or evidence of valid geometry.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message=".*oriented_envelope.*")
            rectangle = component.minimum_rotated_rectangle
        if rectangle.geom_type != "Polygon" or not rectangle.is_valid:
            continue
        corners = list(rectangle.exterior.coords)[:4]
        if len(corners) != 4 or not all(
                math.isfinite(value) for point in corners for value in point):
            continue
        edges = [
            (math.dist(corners[index], corners[(index + 1) % 4]),
             corners[index], corners[(index + 1) % 4])
            for index in range(4)
        ]
        length, first, second = max(edges, key=lambda item: item[0])
        short = min(item[0] for item in edges)
        if short < 2 * width or length / max(short, 1e-9) < 4:
            continue
        angle = math.atan2(second[1] - first[1], second[0] - first[0])
        aligned = rotate(component, -angle, origin=(0, 0), use_radians=True)
        xmin, ymin, xmax, ymax = aligned.bounds
        trim = max(vehicle.front_m, vehicle.rear_m,
                   abs(vehicle.implement_offset_m) + vehicle.implement_length_m / 2)
        trim += vehicle.safety_margin_m
        if xmax - xmin <= 2 * trim + 10:
            continue
        centers = []
        y = ymin + width
        while y <= ymax - width + spacing / 2:
            centers.append(y)
            y += spacing
        # The interior passes need not reach the two sides of a bay. Try two
        # edge passes only after the ordinary rows, and keep them only when
        # their full envelope is safe and their actual sweep adds coverage.
        edge_offset = max(0.5, min(width / 2, spacing * 0.2))
        centers.extend((ymin + edge_offset, ymax - edge_offset))
        for index, y in enumerate(centers):
            if len(proposed) >= max_tasks:
                break
            aligned_line = LineString([(xmin + trim, y), (xmax - trim, y)])
            line = rotate(aligned_line, angle, origin=(0, 0), use_radians=True)
            task_id = f"residual_bay_{component_index:03d}_{index:03d}"
            task = io.FrozenTask(
                scene.name, headland_HEADLAND_REGION_ID, task_id, component_index,
                index, line, angle, GeometryCollection(), line.length,
                work_kind="STRAIGHT_HEADLAND")
            work = io._work_motion(task, scene, False)
            if not validator.motion_issues(work):
                sweep = validator.work_sweep(work)
                gain = sweep.intersection(remaining_target).difference(
                    unary_union([completed, proposed_sweeps])).area
                if gain >= max(20.0, 5 * width):
                    proposed.append(replace(task, frozen_sweep=sweep))
                    proposed_sweeps = unary_union([proposed_sweeps, sweep])
    return proposed

def propose_outer_edge_tasks(scene, remaining_target, completed, *, max_tasks=18):
    """沿真实长外边界提出合法补作任务；每条候选必须独立通过包络与新增覆盖检查。
    
    Suggest near-boundary work along genuine long exterior field edges.
    
    These are not merely shifts of an existing line. Each candidate is built
    from a real target edge, and complete work motion safety is checked before
    it reaches the joint scheduler. The current field's travel geometry still
    governs every candidate, including explicit obstacles and boundary bays."""
    vehicle = scene.vehicle
    width = vehicle.working_width_m
    validator = io.Validator(scene)
    # The vehicle body keeps its travel clearance, but the tool may work up
    # to the true field edge. The route-local validator checks both hulls
    # separately, including holes; a proposed edge line is never accepted
    # merely because this nominal offset fits the exterior boundary.
    minimum_offset = width / 2 + scene.settings.geometry_epsilon_m
    base_trim = max(vehicle.front_m, vehicle.rear_m,
                    abs(vehicle.implement_offset_m) + vehicle.implement_length_m / 2)
    candidates = []
    edge_number = 0
    for polygon in headland_polygons(scene.target):
        ring = polygon.exterior.simplify(0.25, preserve_topology=False)
        points = list(ring.coords)
        inward_sign = 1 if polygon.exterior.is_ccw else -1
        for first, second in zip(points, points[1:]):
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            if length < max(30.0, 8 * width):
                edge_number += 1
                continue
            ux, uy = dx / length, dy / length
            nx, ny = inward_sign * -uy, inward_sign * ux
            for trim_extra in (3.0, 8.0):
                trim = base_trim + trim_extra
                if length <= 2 * trim + 10:
                    continue
                best = None
                for offset in (minimum_offset, minimum_offset + 0.1,
                               minimum_offset + 0.3,
                               minimum_offset + 0.6):
                    a = (first[0] + ux * trim + nx * offset,
                         first[1] + uy * trim + ny * offset)
                    b = (second[0] - ux * trim + nx * offset,
                         second[1] - uy * trim + ny * offset)
                    line = LineString((a, b))
                    task_id = (f"residual_outer_{edge_number:03d}_"
                               f"trim{round(trim_extra):02d}")
                    task = io.FrozenTask(
                        scene.name, headland_HEADLAND_REGION_ID, task_id, edge_number,
                        round(trim_extra), line, math.atan2(dy, dx),
                        GeometryCollection(), line.length,
                        work_kind="STRAIGHT_HEADLAND")
                    work = io._work_motion(task, scene, False)
                    if validator.motion_issues(work):
                        continue
                    sweep = validator.work_sweep(work)
                    gain = sweep.intersection(remaining_target).difference(completed).area
                    if gain >= max(20.0, 5 * width):
                        if best is None or gain > best[0]:
                            best = gain, replace(task, frozen_sweep=sweep)
                if best is not None:
                    candidates.append(best)
            edge_number += 1
    # Prefer the edges that actually fill the most remaining target. The
    # connector later decides which of these safe work lines is reachable.
    candidates.sort(key=lambda item: (-item[0], item[1].task_id))
    return [task for _, task in candidates[:max_tasks]]


# 按实际田头预留区提议安全补作候选

def _boundary_edges(geometry, minimum_length, tolerance):
    """按长度提取真实边界与预留区接口支撑，只用于补作候选生成。
    
    Return true and reserve-interface straight supports, largest first."""
    edges = []
    for polygon in headland_polygons(geometry):
        for ring in (polygon.exterior, *polygon.interiors):
            reduced = ring.simplify(tolerance, preserve_topology=False)
            points = list(reduced.coords)
            for start, end in zip(points, points[1:]):
                line = LineString((start, end))
                if line.length >= minimum_length:
                    edges.append(line)
    return sorted(edges, key=lambda line: -line.length)

def propose_reserve_gap_tasks(scene, headland, completed,
                              candidate_sweeps=GeometryCollection(), *,
                              max_tasks=10, max_support_edges=48):
    """在仍有遗漏目标的预留边界附近提出有限直线补作，避免对无收益边界全量搜索。
    
    Find useful straight-work candidates along actual reserve boundaries.
    
    Only edges near a still-uncovered target gap are considered. Each tested
    pose uses the real vehicle/implement checker; returned tasks are sorted by
    *marginal* target coverage, not by raw sweep area. The small deterministic
    angle/offset/trim set is a proposal budget, never a physical proof that a
    remaining gap has no solution."""
    width = scene.vehicle.working_width_m
    missing = headland.difference(unary_union([completed, candidate_sweeps]))
    if missing.is_empty or missing.area <= scene.settings.coverage_tolerance_m2:
        return [], {"status": "NO_GAP", "tested_work_lines": 0}
    validator = io.Validator(scene)
    base_trim = max(scene.vehicle.front_m, scene.vehicle.rear_m,
                    abs(scene.vehicle.implement_offset_m)
                    + scene.vehicle.implement_length_m / 2)
    # Full offsets are needed at a re-entrant corner; half-width offsets can
    # also recover a narrow strip left between directional body and true edge.
    offsets = (0.0, -width / 2, width / 2, -width, width,
               -1.5 * width, 1.5 * width)
    trims = (base_trim + 1.0, base_trim + 4.0)
    raw = []
    attempted = safe = 0
    # Work only near spatially relevant edges. Long edges are capped before
    # the offset loop, so a highly detailed field does not cause a full-grid
    # search across its entire boundary.
    edges = [edge for edge in _boundary_edges(
        headland, 2 * trims[0] + max(3.0, width), 0.25)
        if edge.distance(missing) <= 2 * width][:max_support_edges]
    for edge_index, edge in enumerate(edges):
        a, b = edge.coords[0], edge.coords[-1]
        ux, uy = (b[0]-a[0])/edge.length, (b[1]-a[1])/edge.length
        nx, ny = -uy, ux
        heading = math.atan2(uy, ux)
        for offset_index, offset in enumerate(offsets):
            for trim_index, trim in enumerate(trims):
                if edge.length <= 2*trim + max(2.0, width):
                    continue
                start = (a[0]+ux*trim+nx*offset,
                         a[1]+uy*trim+ny*offset)
                end = (b[0]-ux*trim+nx*offset,
                       b[1]-uy*trim+ny*offset)
                line = LineString((start, end))
                if line.distance(missing) > width:
                    continue
                task_id = f"reserve_gap_e{edge_index:03d}_o{offset_index}_t{trim_index}"
                task = io.FrozenTask(scene.name, "__HEADLAND__", task_id,
                    edge_index, offset_index*10+trim_index, line, heading,
                    GeometryCollection(), line.length,
                    work_kind="STRAIGHT_HEADLAND")
                attempted += 1
                motion = io._work_motion(task, scene, False)
                if validator.motion_issues(motion):
                    continue
                safe += 1
                sweep = validator.work_sweep(motion)
                gain = sweep.intersection(missing).area
                if gain >= max(20.0, 5*width):
                    raw.append((gain, replace(task, frozen_sweep=sweep)))
    selected = []
    covered = GeometryCollection()
    while raw and len(selected) < max_tasks:
        _, task = max(raw, key=lambda item: (
            item[1].frozen_sweep.intersection(missing)
            .difference(covered).area, item[0], item[1].task_id))
        novel = task.frozen_sweep.intersection(missing).difference(covered).area
        if novel < max(20.0, 5*width):
            break
        selected.append(task)
        covered = unary_union([covered, task.frozen_sweep])
        raw = [(value, other) for value, other in raw
               if other.row_index != task.row_index]
    return selected, {
        "status": "CANDIDATES" if selected else "NO_SAFE_WORK_LINE",
        "headland_gap_m2": missing.area,
        "support_edge_count": len(edges),
        "tested_work_lines": attempted,
        "safe_work_lines": safe,
        "selected_task_count": len(selected),
        "candidate_gap_reduction_m2": covered.intersection(missing).area,
        "uncovered_after_candidates_m2": missing.difference(covered).area,
    }


# 替代方向的覆盖保护和实际选中扫掠账本

def protect_coverage(adapted,scene,completed):
    """检查田头替代候选能否保持必须覆盖的目标，不能用较顺路线换取静默漏作。"""
    tasks=[task for task in adapted.tasks if task.region_id=='__HEADLAND__']
    choices=[task for task in tasks if task.alternative_motion_points]
    primary=unary_union([task.frozen_sweep for task in tasks])
    common={}
    for task in tasks:
        cover=task.frozen_sweep
        for points in task.alternative_motion_points:
            cover=cover.intersection(headland_conservative_tool_coverage(
                io.Motion(points,'work',task.task_id,True),scene))
        common[task.task_id]=cover
    supported=unary_union([completed,*common.values()])
    lost=primary.intersection(scene.target).difference(supported)
    # Removing every alternative whose primary footprint supplies this gap
    # restores its coverage in one monotone pass. Using *common* coverage
    # avoids circular assumptions that two alternatives cover for each other.
    rejected={task.task_id for task in choices
              if task.frozen_sweep.intersection(lost).area>1e-10}
    guaranteed_parts={task.task_id:(task.frozen_sweep
        if task.task_id in rejected else common[task.task_id]) for task in tasks}
    guaranteed=unary_union([completed,*[
        task.frozen_sweep if task.task_id in rejected else common[task.task_id]
        for task in tasks]])
    missing=primary.intersection(scene.target).difference(guaranteed).area
    if missing>io._sweep_tolerance(primary.area):
        raise ValueError('CONTOUR_ALTERNATIVE_TARGET_COVERAGE_LOST')
    retained=[]
    for task in adapted.tasks:
        if task.task_id in rejected:
            retained.append(replace(task,alternative_motion_points=()))
        elif task.alternative_motion_points:
            other_support=unary_union([completed,*[part for key,part in
                guaranteed_parts.items() if key!=task.task_id]])
            exclusive=task.frozen_sweep.intersection(scene.target).difference(other_support)
            retained.append(replace(task,required_target_sweep=exclusive))
        else:
            retained.append(task)
    return replace(adapted,tasks=tuple(retained)),{
        'proposed_reverse_candidates':len(choices),
        'retained_reverse_candidates':len(choices)-len(rejected),
        'coverage_rejected_task_ids':sorted(rejected),
        'guaranteed_primary_target_missing_m2':missing}

def bind_selected_sweeps(result,scene):
    """将实际选中作业运动的扫掠及必要目标绑定到账本，供严格回读审计。
    
    Store the chosen footprint and its necessary target for strict readback."""
    motions={motion.task_id:motion for route in result.routes
             if route.status=='REGION_ROUTE_PASS' for motion in route.motions
             if motion.implement_on}
    updated=[];records=[]
    for task in result.adapted_job.tasks:
        motion=motions.get(task.task_id)
        if task.required_target_sweep is None or motion is None:
            updated.append(task);continue
        if task.work_kind!='CURVED_HEADLAND' or task.region_id!='__HEADLAND__':
            raise ValueError('TARGET_SWEEP_MINIMUM_OUTSIDE_GENERATED_HEADLAND')
        actual=headland_conservative_tool_coverage(motion,scene)
        lost=task.required_target_sweep.difference(actual).area
        if lost>io._sweep_tolerance(task.required_target_sweep.area):
            raise ValueError('SELECTED_CONTOUR_LOST_REQUIRED_TARGET')
        line=LineString(motion.points[:,:2])
        records.append({'task_id':task.task_id,
            'selected_type':'PRIMARY' if np.array_equal(
                task.motion_points,motion.points) else 'ALTERNATIVE_OR_ENTRY_PHASE',
            'actual_work_length_m':line.length,'required_target_missing_m2':lost,
            'primary_sweep_difference_m2':actual.symmetric_difference(task.frozen_sweep).area})
        updated.append(replace(task,reference_line=line,
            heading_rad=float(motion.points[0,2]),length_m=line.length,
            frozen_sweep=actual,motion_points=motion.points.copy(),
            alternative_motion_points=()))
    result.adapted_job=replace(result.adapted_job,tasks=tuple(updated))
    return records

# ==========================================================================
# 3. 田头轮廓与温和转角
# 把轮廓变成可供连接器尝试的曲线任务，保留原有曲率候选与预算。
# ==========================================================================

from dataclasses import replace
import math
import re
import time
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
from scene import Motion as contours_Motion
from scene import wrap as contours_wrap
import route_planner as io
from validator import conservative_tool_coverage as contours_conservative_tool_coverage

_contours_PATTERN = re.compile(r'^(headland_(?:hole_)?ring_\d+_[\d_]+)_edge_(\d+)$')


def local_forward_corner(motion, net_heading, extra_heading=0.08):
    """排除倒车作业、朝向振荡和绕整圈的局部模板；能求出曲线不等于司机常识合理。
    
    Exclude winding loops, backward work and heading oscillations.
    
    A locally valid Dubins path may contain a full circle. That is a legal
    transit but is not a contour continuation. Compare accumulated heading
    variation with the actual corner angle, independently of line styling."""
    points = motion.points
    if np.any(points[:-1, 3] < 0):
        return False
    dyaw = (np.diff(points[:, 2])+math.pi) % (2*math.pi)-math.pi
    return float(np.abs(dyaw).sum()) <= abs(net_heading)+extra_heading


def gentle_radii(a, b, scene):
    """由端点切向和可用裁切空间提出少量圆滑半径；提案仍须原生运动求解与包络复检。
    
    Two geometric radius proposals for a zero-curvature entry and exit.
    
    Intersect pose tangents to measure usable trim T. A symmetric CC corner
    needs approximately R*tan(theta/2) + 1/(2*sigma*R) of tangent support.
    Its quadratic gives a gentle radius directly; it is only a proposal,
    never a safety proof. The final native curve is checked under the actual
    vehicle limits. Larger R means less steering, not a relaxed constraint."""
    theta = abs(contours_wrap(b.yaw-a.yaw))
    if theta < 1e-5 or theta > math.pi-1e-3:
        return []
    ua = np.array([math.cos(a.yaw), math.sin(a.yaw)])
    ub = np.array([math.cos(b.yaw), math.sin(b.yaw)])
    try:
        support = np.linalg.solve(np.column_stack((ua,ub)),
                                  np.array([b.x-a.x,b.y-a.y]))
    except np.linalg.LinAlgError:
        return []
    trim = float(support.min())
    tangent = math.tan(theta/2)
    sigma = scene.vehicle.max_curvature_rate
    discriminant = trim*trim-2*tangent/sigma
    if trim <= 0 or discriminant <= 0:
        return []
    radius = (trim+math.sqrt(discriminant))/(2*tangent)
    minimum = scene.vehicle.min_turn_radius_m
    return [value for value in (radius, .95*radius)
            if value > minimum*(1+1e-6)]


def gentle_native(connector, a, b, radius):
    """为圆滑候选使用独立原生机器人，不修改冻结后端的车辆配置。
    
    Use an independent F2C robot; never mutate the frozen backend robot."""
    f2c, scene = connector.backend.f2c, connector.scene
    v = scene.vehicle
    robot = f2c.Robot(v.body_width_m,v.working_width_m)
    robot.setMinTurningRadius(radius)
    robot.setMaxDiffCurv(v.max_curvature_rate)
    robot.setCruiseVel(v.transit_speed_mps)
    robot.setTurnVel(v.turn_speed_mps)
    turner = f2c.PP_DubinsCurvesCC()
    turner.setUsingCache(False)
    turner.setDiscretization(scene.settings.sampling_step_m)
    c,s = math.cos(a.yaw),math.sin(a.yaw)
    dx,dy = b.x-a.x,b.y-a.y
    path = turner.createTurn(robot,f2c.Point(0,0),0,
        f2c.Point(c*dx+s*dy,-s*dx+c*dy),contours_wrap(b.yaw-a.yaw))
    local = connector.backend.convert_path(path,('contour','contour'),'work')
    points = local.points.copy()
    xy = points[:,:2].copy()
    points[:,0]=a.x+c*xy[:,0]-s*xy[:,1]
    points[:,1]=a.y+s*xy[:,0]+c*xy[:,1]
    points[:,2]+=a.yaw
    result = contours_Motion(points,'work','__CONTOUR_CORNER__',True)
    if not (io._pose_close(io._pose_at(result,False),a,scene) and
            io._pose_close(io._pose_at(result,True),b,scene)):
        raise ValueError('GENTLE_ENDPOINT_MISMATCH')
    return result


def consolidate(tasks, scene, connector, seconds=10.0, build_reverse=False):
    """在有限预算内整理可替代的田头运动，覆盖和入出连接都要复检，原未完成任务不静默删除。"""
    deadline = min(connector.deadline, time.perf_counter()+seconds)
    groups = {}
    untouched = []
    for task in tasks:
        match = _contours_PATTERN.fullmatch(task.task_id)
        if task.work_kind != 'STRAIGHT_HEADLAND' or match is None:
            untouched.append(task)
        else:
            groups.setdefault(match[1], []).append((int(match[2]), task))
    output = list(untouched)
    records = []

    def corner(left, right):
        a, b = io._pose_at(left, True), io._pose_at(right, False)
        if io._pose_close(a, b, scene):
            return True, None
        if time.perf_counter() >= deadline:
            return False, None
        radii = [None, *gentle_radii(a,b,scene)]
        for radius in radii:
            if time.perf_counter() >= deadline:
                break
            try:
                raw = (connector.native(a,b,0,False) if radius is None else
                       gentle_native(connector,a,b,radius))
            except (RuntimeError, ValueError, IndexError):
                continue
            work = contours_Motion(raw.points,'work','__CONTOUR_CORNER__',True)
            if not local_forward_corner(work,contours_wrap(b.yaw-a.yaw)):
                continue
            issues,_ = connector.checker.physical(work)
            if not issues:
                return True,work
        return False,None

    def finish(prefix, number, originals, segments, close=False):
        if len(originals) < 2:
            output.extend(originals)
            return
        if close:
            accepted, turn = corner(segments[-1], segments[0])
            if accepted and turn is not None:
                segments = [*segments, turn]
        arrays = [seg.points[:-1] for seg in segments[:-1]]+[segments[-1].points]
        points = np.vstack(arrays)
        if np.linalg.norm(points[0,:2]-points[-1,:2]) < 1e-6 and abs(
                contours_wrap(float(points[0,2]-points[-1,2]))) < 1e-6:
            points[-1] = points[0]
        task_id = f'{prefix}_contour_{number:03d}'
        motion = contours_Motion(points, 'work', task_id, True)
        issues, _ = connector.checker.physical(motion)
        if issues:
            output.extend(originals)
            return
        coverage = contours_conservative_tool_coverage(motion, scene)
        required = unary_union([task.frozen_sweep for task in originals])
        lost = required.difference(coverage).area
        if lost > io._sweep_tolerance(required.area):
            output.extend(originals)
            return
        line = LineString(points[:, :2])
        alternatives=[]
        if build_reverse and time.perf_counter()<deadline:
            reverse_segments=[]
            for original in reversed(originals):
                reverse_work=io._work_motion(original,scene,True)
                if connector.checker.physical(reverse_work)[0]:
                    reverse_segments=[];break
                if reverse_segments:
                    accepted,turn=corner(reverse_segments[-1],reverse_work)
                    if not accepted:
                        reverse_segments=[];break
                    if turn is not None:
                        reverse_segments.append(turn)
                reverse_segments.append(reverse_work)
            if reverse_segments:
                closed_original=np.linalg.norm(points[0,:2]-points[-1,:2])<1e-8
                if closed_original:
                    accepted,turn=corner(reverse_segments[-1],reverse_segments[0])
                    if not accepted:
                        reverse_segments=[]
                    elif turn is not None:
                        reverse_segments.append(turn)
                if reverse_segments:
                    reverse_points=np.vstack([seg.points[:-1] for seg in
                        reverse_segments[:-1]]+[reverse_segments[-1].points])
                    reverse_motion=contours_Motion(reverse_points,'work',task_id,True)
                    if not connector.checker.physical(reverse_motion)[0]:
                        reverse_sweep=contours_conservative_tool_coverage(reverse_motion,scene)
                        if required.difference(reverse_sweep).area<=io._sweep_tolerance(required.area):
                            alternatives.append(reverse_points)
            # With a rear-mounted tool, reversing each vertex separately can
            # place the vehicle beyond a tiny bend before it starts turning.
            # For an open contour also propose one smooth native curve from
            # the opposite actual entry to exit. Its footprint is *different*;
            # protect_coverage must certify the whole task set before use.
            if (not alternatives and np.linalg.norm(points[0,:2]-points[-1,:2])>=1e-8
                    and time.perf_counter()<deadline):
                a=io._pose_at(io._work_motion(originals[-1],scene,True),False)
                b=io._pose_at(io._work_motion(originals[0],scene,True),True)
                for radius in [None,*gentle_radii(a,b,scene)]:
                    if time.perf_counter()>=deadline:
                        break
                    try:
                        raw=(connector.native(a,b,0,False) if radius is None else
                             gentle_native(connector,a,b,radius))
                    except (RuntimeError,ValueError,IndexError):
                        continue
                    reverse_motion=contours_Motion(raw.points,'work',task_id,True)
                    if (local_forward_corner(reverse_motion,contours_wrap(b.yaw-a.yaw),.20)
                            and not connector.checker.physical(reverse_motion)[0]):
                        alternatives.append(raw.points.copy());break
        task = io.FrozenTask(originals[0].field_id, originals[0].region_id,
            task_id, originals[0].row_index, originals[0].suggested_order,
            line, float(points[0,2]), coverage, line.length,
            work_kind='CURVED_HEADLAND', motion_points=points.copy(),
            alternative_motion_points=tuple(alternatives),
            required_target_sweep=None)
        output.append(task)
        records.append({'task_id':task_id,
            'source_task_ids':[task.task_id for task in originals],
            # Preserve original straight poses as optional transit support.
            # A continuous working contour restricts its execution direction;
            # its absorbed straight passes may still supply a safe approach
            # after they have actually been worked. They are not extra jobs.
            'source_straight_tasks':[{'task_id':t.task_id,
                'reference_line_wkt':t.reference_line.wkt,
                'heading_rad':t.heading_rad,'sweep_wkt':t.frozen_sweep.wkt,
                'row_index':t.row_index,'suggested_order':t.suggested_order}
                for t in originals],
            'construction':'CONTINUOUS_FORWARD_CONTOUR',
            'closed':bool(np.linalg.norm(points[0,:2]-points[-1,:2]) < 1e-8),
            'source_work_length_m':sum(task.length_m for task in originals),
            'continuous_work_length_m':line.length,
            'reversed_order_candidate':bool(alternatives),
            'source_sweep_missing_m2':lost,
            'additional_work_area_m2':coverage.difference(required).area})

    for prefix, entries in groups.items():
        entries.sort(key=lambda item:item[0])
        source = [task for _, task in entries]
        originals, segments, number = [], [], 0
        for task in source:
            work = io._work_motion(task, scene, False)
            if not originals:
                originals, segments = [task], [work]
                continue
            accepted, turn = corner(segments[-1], work)
            if not accepted:
                finish(prefix, number, originals, segments)
                number += 1
                originals, segments = [task], [work]
            else:
                originals.append(task)
                if turn is not None:
                    segments.append(turn)
                segments.append(work)
        if originals:
            finish(prefix, number, originals, segments,
                   close=len(originals)==len(source))
    return output, records

# ==========================================================================
# 4. 路线阶段授权的作业空间准备
# 生成派生田头和条带输入；冻结输入仍独立保留，原始 target 不被静默修改。
# ==========================================================================

from dataclasses import replace
import math
import numpy as np
from shapely.affinity import rotate
from shapely.affinity import scale
from shapely.affinity import translate
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.ops import unary_union
import route_planner as io
from validator import vehicle_sweep as prepare_vehicle_sweep


def _prepare_lines(geometry):
    """提取正长度独立线段，几何集合递归处理，不跨空白区域拼线。"""
    if geometry.geom_type=='LineString':
        return [geometry] if geometry.length>1e-6 else []
    if hasattr(geometry,'geoms'):
        return [line for child in geometry.geoms for line in _prepare_lines(child)]
    return []


def _directional_inset(geometry, heading_rad, end_m, side_m):
    """沿指定朝向用不同纵横尺度退让，只构造候选参考点空间，最终包络仍需验收。
    
    Erode by a heading-aligned ellipse: end room is large, side room small.
    
    The affine transform is only used to construct a candidate body.  Every
    resulting work motion and connection is still checked against the original
    metric target/travel and full rigid vehicle plus implement envelope."""
    aligned=rotate(geometry,-heading_rad,origin=(0,0),use_radians=True)
    unit=scale(aligned,xfact=1/end_m,yfact=1/side_m,origin=(0,0))
    inset=unit.buffer(-1,quad_segs=8)
    if inset.is_empty:
        return GeometryCollection()
    restored=scale(inset,xfact=end_m,yfact=side_m,origin=(0,0))
    return rotate(restored,heading_rad,origin=(0,0),use_radians=True).intersection(geometry)


# 为严格路线准备派生作业空间和任务，保留原始 target 作为验收基准。
# 派生条带和冻结条带是不同来源，不能覆盖原批次或偷偷改其质量状态。
def prepare(job,scene,connector,mode='UNIFORM'):
    """建立路线阶段派生预留空间与任务上下文；不修改Scene或已发布主体覆盖义务。"""
    v=scene.vehicle
    radius=max(float(np.linalg.norm(rect,axis=1).max()) for rect in v.rectangles())
    staging=radius+v.safety_margin_m+0.25
    spacing=v.working_width_m*(1-scene.settings.overlap_fraction)
    # Use actual permitted CC/HC envelopes, including reverse-leg limits. The
    # chosen nominal family sizes the reserve; actual connections are rechecked.
    demands=[]
    for index in range(len(connector.backend.turners)):
        motion=connector.native(io.Pose(0,0,0),io.Pose(0,spacing,math.pi),index)
        legs,shifts,_,_,duration=io._gear_and_steering(motion,scene,connector.settings)
        envelope=prepare_vehicle_sweep(motion,scene)
        if math.isfinite(duration) and envelope.bounds[0]>=-staging:
            lateral_turn=max(-envelope.bounds[1],envelope.bounds[3]-spacing)
            demands.append((envelope.bounds[2],lateral_turn,
                            connector.backend.turners[index][0]))
    if not demands:
        raise ValueError('NO_NOMINAL_TURN_WITHIN_OPERATION_LIMITS')
    outward,_,family=min(demands)
    width=staging+outward+v.safety_margin_m+scene.settings.travel_clearance_m+0.5+v.working_width_m/2+abs(v.implement_offset_m)
    # Only strip ends need the full manoeuvre room.  At an edge parallel to a
    # strip the machine needs its actual lateral half-envelope plus clearance.
    lateral=max(float(np.abs(rect[:,1]).max()) for rect in v.rectangles())
    # The outermost row also turns toward its neighbour.  Its turn envelope
    # may protrude laterally farther than the implement on a straight pass.
    side_width=max(lateral,max(item[1] for item in demands)) + (
        v.safety_margin_m+scene.settings.travel_clearance_m+0.5)
    side_width=min(width,side_width)
    if mode=='UNIFORM':
        body=scene.target.buffer(-width)
    elif mode=='DIRECTIONAL':
        by_region={r.region_id:next((t.heading_rad for t in job.tasks
                                     if t.region_id==r.region_id),None) for r in job.regions}
        cores=[]
        for region in job.regions:
            heading=by_region[region.region_id]
            if heading is None:
                continue
            cores.append(_directional_inset(scene.target,heading,width,side_width)
                         .intersection(region.geometry))
        body=unary_union(cores) if cores else GeometryCollection()
    else:
        raise ValueError(f'UNKNOWN_HEADLAND_STRATEGY:{mode}')
    interfaces=[]
    for i,first in enumerate(job.regions):
        for second in job.regions[i+1:]:
            if not first.geometry.intersects(second.geometry.buffer(1e-6)):
                continue
            boundary=first.geometry.boundary.intersection(second.geometry.boundary)
            if boundary.length<1e-5:
                boundary=first.geometry.boundary.intersection(second.geometry.boundary.buffer(1e-6))
            if boundary.length>v.working_width_m:
                interfaces.append(boundary)
    corridor=(unary_union(interfaces).buffer(width/2) if interfaces else GeometryCollection())
    body=body.difference(corridor)
    reserve=scene.target.difference(body)
    validator=io.Validator(scene)
    tasks=[];regions=[];rejected=[];mapping=[]
    added_rows=0
    diagonal=math.hypot(scene.target.bounds[2]-scene.target.bounds[0],
                        scene.target.bounds[3]-scene.target.bounds[1])+2*width
    for region in job.regions:
        core=body.intersection(region.geometry)
        region_tasks=[]
        seen_rows=set()
        originals=[t for t in job.tasks if t.region_id==region.region_id]
        rows=[(t,t.task_id) for t in originals]
        if mode=='DIRECTIONAL' and originals and not core.is_empty:
            angle=originals[0].heading_rad % math.pi
            normal=np.array([-math.sin(angle),math.cos(angle)])
            offsets=[float(np.asarray(t.reference_line.centroid.coords[0])@normal)
                     for t in originals]
            first=min(range(len(originals)),key=lambda i:offsets[i])
            template=originals[first]
            base=offsets[first]
            ymin=rotate(core,-angle,origin=(0,0),use_radians=True).bounds[1]
            ymax=rotate(core,-angle,origin=(0,0),use_radians=True).bounds[3]
            half=v.working_width_m/2
            kmin=math.floor((ymin-half-base)/spacing)
            kmax=math.ceil((ymax+half-base)/spacing)
            for k in range(kmin,kmax+1):
                offset=base+k*spacing
                if any(abs(offset-source)<min(0.05,spacing*0.1)
                       for source in offsets):
                    continue
                shift=(offset-base)*normal
                synthetic=replace(template,
                    task_id=f'{region.region_id}_extended_row_{k:+06d}',
                    row_index=template.row_index+k,
                    suggested_order=template.suggested_order+k,
                    reference_line=translate(template.reference_line,
                        xoff=float(shift[0]),yoff=float(shift[1])))
                rows.append((synthetic,None))
                added_rows+=1
        for original,source_id in rows:
            u=np.array([math.cos(original.heading_rad),math.sin(original.heading_rad)])
            center=np.array(original.reference_line.centroid.coords[0])
            angle=original.heading_rad % math.pi
            normal=np.array([-math.sin(angle),math.cos(angle)])
            row_key=(round(angle,7),round(float(center@normal),4))
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            line=LineString([center-diagonal*u,center+diagonal*u])
            # Include edge rows whose implement overlaps the core although its
            # reference line is just outside it; otherwise a thin edge is lost.
            supported=core.buffer(v.working_width_m/2,join_style=2)
            pieces=sorted(_prepare_lines(line.intersection(supported)),key=lambda l:l.centroid.x*u[0]+l.centroid.y*u[1])
            for index,piece in enumerate(pieces):
                a=np.array(piece.coords[0]);b=np.array(piece.coords[-1])
                if np.dot(b-a,u)<0:
                    a,b=b,a
                # Symmetric *implement* endpoints leave comparable room for
                # either driving direction with a longitudinally offset tool.
                a=a-(staging+v.implement_offset_m)*u
                b=b+(staging-v.implement_offset_m)*u
                reference=LineString([a,b])
                if reference.length<1:
                    continue
                tid=f'{original.task_id}_route{index+1:02d}'
                provisional=replace(original,task_id=tid,reference_line=reference,
                                    length_m=reference.length)
                working=io._work_motion(provisional,scene,False)
                issues=validator.motion_issues(working)
                if issues:
                    rejected.append({'task_id':tid,'codes':[x['code'] for x in issues]})
                    continue
                sweep=validator.work_sweep(working)
                region_tasks.append(replace(provisional,frozen_sweep=sweep))
                mapping.append({'task_id':tid,'source_task_id':source_id,
                                'region_id':region.region_id,
                                'construction':'SOURCE_ROW' if source_id else 'EXTENDED_PHASE_ROW'})
        tasks.extend(region_tasks)
        regions.append(replace(region,headland=reserve.intersection(region.geometry),
                               task_ids=tuple(t.task_id for t in region_tasks)))
    # Recover only rounding slivers in the exported partition boundary. The
    # authoritative target/body/reserve ledger stays in Scene's metre geometry.
    remainder=reserve.difference(unary_union([r.headland for r in regions]))
    if regions and not remainder.is_empty:
        regions[0]=replace(regions[0],headland=regions[0].headland.union(remainder))
    sweeps=unary_union([t.frozen_sweep for t in tasks])
    missing=body.difference(sweeps)
    # Empty cores remain explicitly headland-only. They do not disappear from
    # the target denominator or become fake completed work regions.
    diagnostics={'headland_width_m':float(width),'endpoint_staging_m':float(staging),
                 'side_clearance_m':float(side_width),
                 'reservation_mode':mode,
                 'isotropic_body_area_m2':float(scene.target.buffer(-width)
                    .difference(corridor).area),
                 'headland_reference_turn_family':family,
                 'native_turn_envelope_outward_m':float(outward),
                 'required_body_area_m2':float(body.area),
                 'headland_reserve_area_m2':float(reserve.area),
                 'body_missing_m2':float(missing.area),
                 'deferred_headland_work_m2':float(scene.target.difference(sweeps).area),
                 'target_area_m2':float(scene.target.area),
                 'source_task_count':len(job.tasks),'replanned_task_count':len(tasks),
                 'added_phase_rows':added_rows,
                 'rejected_tasks':rejected,'task_mapping':mapping,
                 'authorization':'USER_20260927_REPLAN_HEADLANDS_AND_SWATHS_STRICT_BODY'}
    return replace(job,regions=tuple(regions),tasks=tuple(tasks)),body,diagnostics

# ==========================================================================
# 5. 独立田头参考作业
# 只生成单独田头路线，不声称这些路线全部与主体行程连接。
# ==========================================================================

from dataclasses import replace
import time
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
import route_planner as io
headland_separate_generate = generate_headland_tasks


# 独立田头作业入口：这些线不要求全部与主体行程连在一起。
# 输出合法线段及对应状态，不宣称完整田头覆盖或连续转场。
def plan_separate(job, settings):
    """先组织独立田头圈，再调用既有主体求解；作业任务、连接和各阶段失败分开记录。"""
    pass  # 已合并到本模块，直接使用下方的定义。
    started = time.perf_counter()
    # The existing body solver retains its dynamic worked-space rules and
    # real inter-region transfers. No headland search retries are invoked.
    body = plan_field(job, replace(settings, plan_headland_work=False))
    scene = io.load_scene(job.scene_path)
    adapted = body.adapted_job or job
    required = (body.required_body if body.required_body is not None else
                scene.target.difference(unary_union([r.headland for r in adapted.regions])))
    headland = scene.target.difference(required)
    completed = unary_union([r.completed_sweep for r in body.routes
                             if r.status == 'REGION_ROUTE_PASS'])
    generation_started = time.perf_counter()
    connector = Connector(scene, settings, time.perf_counter()+10.0)
    # Pass the integer explicitly: ceil((3*w)/w) can be 4 with decimal widths
    # such as 0.1 m. Do not alter actual reserve geometry to control ring count.
    try:
        augmented, diagnostic = headland_separate_generate(adapted, scene, headland, connector.backend,
            body.preparation, settings.headland_simplification_tolerance_m,
            settings.headland_positioning_allowance_m, contour_connector=connector,
            ring_count_override=settings.headland_pass_count)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        # A headland failure must not discard a valid, expensive body result.
        record={'mode':'SEPARATE_PASSES','requested_pass_count':settings.headland_pass_count,
            'generated_ring_count':0,'safe_segment_count':0,'segments_per_pass':{},
            'headland_area_m2':headland.area,
            'headland_planned_missing_m2':headland.difference(completed).area,
            'planned_target_missing_m2':scene.target.difference(completed).area,
            'coverage_status':'GENERATION_FAILED',
            'connection_status':'SEPARATE_FROM_BODY_AND_OTHER_SEGMENTS',
            'operational_status':'PARAMETERS_UNVERIFIED',
            'generation_seconds':time.perf_counter()-generation_started,
            'error':f'{type(exc).__name__}: {exc}',
            'no_global_headland_connection_search':True}
        body.preparation['separate_headland']=record
        body.preparation['headland_work_status']='SEPARATE_PASSES_GENERATION_FAILED'
        body.statistics['separate_headland']=record
        body.failures.append({'field_id':job.field_id,'region_id':'__HEADLAND__',
                             'code':'SEPARATE_HEADLAND_GENERATION_FAILED','detail':record['error']})
        body.elapsed_s=time.perf_counter()-started
        return body
    tasks = tuple(t for t in augmented.tasks if t.region_id == '__HEADLAND__')
    # Remove only wholly redundant contour pieces. No arbitrary endpoint
    # truncation or implement switching inside a segment is introduced.
    retained = []
    coverage = completed
    for task in tasks:
        if task.frozen_sweep.intersection(headland).difference(coverage).area <= 1e-8:
            continue
        retained.append(task)
        coverage = coverage.union(task.frozen_sweep)
    tasks = tuple(retained)
    # A separate coverage plan is never inserted into `routes` or task_order.
    # Each entry has its own pose sequence; entry/exit travel is unplanned.
    body.separate_headland_tasks = tasks
    independent_sweep = unary_union([t.frozen_sweep for t in tasks])
    combined = completed.union(independent_sweep)
    missing = headland.difference(combined).area
    counts = {str(i+1):sum(t.row_index == i for t in tasks)
              for i in range(settings.headland_pass_count)}
    record = {
        'mode':'SEPARATE_PASSES', 'requested_pass_count':settings.headland_pass_count,
        'generated_ring_count':diagnostic['headland_ring_count'],
        'safe_segment_count':len(tasks), 'segments_per_pass':counts,
        'headland_area_m2':headland.area,
        'body_already_covered_headland_m2':completed.intersection(headland).area,
        'headland_planned_covered_m2':combined.intersection(headland).area,
        'headland_planned_missing_m2':missing,
        'planned_target_missing_m2':scene.target.difference(combined).area,
        'coverage_status':('COVERAGE_COMPLETE' if missing <= scene.settings.coverage_tolerance_m2
                           else 'COVERAGE_GAP'),
        'connection_status':'SEPARATE_FROM_BODY_AND_OTHER_SEGMENTS',
        'operational_status':'PARAMETERS_UNVERIFIED',
        'generation_seconds':time.perf_counter()-generation_started,
        'headland_source_task_count':diagnostic['headland_source_task_count'],
        'rejected_candidates':diagnostic['headland_candidate_rejected'],
        'contour_chains':diagnostic['headland_contour_chains'],
        'no_global_headland_connection_search':True,
    }
    body.preparation['separate_headland'] = record
    body.preparation['headland_work_status'] = 'SEPARATE_PASSES_WITH_EXPLICIT_GAPS'
    body.statistics['separate_headland'] = record
    body.elapsed_s = time.perf_counter()-started
    return body


def motion(task, scene):
    """读取任务已保存曲线运动，或为直线任务构造运动，保持机具开启标志与任务号。"""
    if task.motion_points is not None:
        from scene import Motion
        return Motion(task.motion_points, 'work', task.task_id, True)
    return io._work_motion(task, scene, False)

# ==========================================================================
# 6. 行程前缀复核、续作与末条恢复
# 先检查已有行程，再尝试有限续接；保留原有主体任务与已认证前缀。
# ==========================================================================

from copy import deepcopy
from dataclasses import replace
import math
import time

import numpy as np
import shapely
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union

import route_planner as io
continue_Connector = Connector
continue_candidate_region = candidate_region
_continue_replan_last_body_task = _replan_last_body_task
from validator import MotionChecker as continue_MotionChecker


# 先从保存采样重建并检查行程前缀，确定已完成任务和真实覆盖。
# 只在可信前缀后继续规划，损坏或重复的前缀不能当成有效起点。
def checked_prefix(result, scene):
    """重新检验真实采样；不能凭成功字段扩大已作业面积。"""
    job = result.adapted_job
    if job is None or result.required_body is None:
        raise ValueError('RESUME_MISSING_ADAPTED_INPUT')
    tasks = {t.task_id: t for t in job.tasks}
    headland = scene.target.difference(result.required_body)
    checker = continue_MotionChecker(scene)
    validator = io.Validator(scene)
    completed = GeometryCollection()
    worked = {}
    seen = set()
    previous = scene.start
    before = None
    for block in result.routes:
        if block.status != 'REGION_ROUTE_PASS':
            continue
        actual_ids = []
        for motion in block.motions:
            if previous is not None and not io._pose_close(
                    previous, io._pose_at(motion, False), scene):
                raise ValueError('RESUME_PREFIX_DISCONTINUITY')
            issues, parts = checker.physical(motion)
            if issues:
                raise ValueError('RESUME_PREFIX_PHYSICS:' + ','.join(issues))
            if motion.implement_on:
                task = tasks.get(motion.task_id)
                if task is None or task.region_id != block.region_id or task.task_id in seen:
                    raise ValueError('RESUME_PREFIX_TASK_CONTRACT')
                if task.work_kind != 'STRAIGHT_BODY':
                    raise ValueError('RESUME_REQUIRES_STRAIGHT_BODY_TASKS')
                sweep = validator.work_sweep(motion)
                if sweep.symmetric_difference(task.frozen_sweep).area > io._sweep_tolerance(sweep.area):
                    raise ValueError('RESUME_PREFIX_SWEEP_CHANGED')
                seen.add(task.task_id)
                actual_ids.append(task.task_id)
                completed = completed.union(sweep.intersection(scene.target))
                worked[task.region_id] = worked.get(task.region_id, GeometryCollection()).union(sweep)
                before = task
            else:
                allowed = io._ready_area(scene, headland, completed).buffer(
                    scene.settings.geometry_epsilon_m)
                shapely.prepare(allowed)
                if not np.all(shapely.covers(allowed, parts)):
                    raise ValueError('RESUME_PREFIX_UNWORKED_BODY_CROSSING')
            previous = io._pose_at(motion, True)
        if actual_ids != block.task_order:
            raise ValueError('RESUME_PREFIX_ORDER_CHANGED')
    if before is None or previous is None:
        raise ValueError('RESUME_REQUIRES_NONEMPTY_PREFIX')
    return headland, completed, worked, seen, previous, before


# 从复核后的末端继续剩余主体任务，保持已经认证的行程不变。
# 连接失败时返回已保留的有效部分，不能用跨障碍直线补齐。
def continue_body(result, settings, *, seconds=60.0, query_seconds=1.25):
    """保留原行程，追加合法的原条带；预算只花在尚未完成的部分。"""
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(query_seconds) or query_seconds <= 0:
        raise ValueError('RESUME_BUDGET_MUST_BE_POSITIVE_FINITE')
    started = time.perf_counter()
    original = result
    result = deepcopy(original)
    job = result.adapted_job
    if job is None:
        raise ValueError('RESUME_MISSING_ADAPTED_INPUT')
    scene = io.load_scene(job.scene_path)
    headland, completed, worked, seen, previous, before = checked_prefix(result, scene)
    remaining = [t for t in job.tasks if t.task_id not in seen]
    if not remaining:
        result.statistics['prefix_continuation'] = {
            'initial_completed_tasks':len(seen),'added_tasks':0,'remaining_tasks':0,
            'queries':0,'trace':[],'counts':{},'stage_seconds':{},
            'search_status':'NO_REMAINING_TASKS','seconds':time.perf_counter()-started,
            'budget_seconds':seconds,'prefix_preserved':True,'tasks_or_target_changed':False}
        return result
    regions = {r.region_id: r for r in job.regions}
    validator = io.Validator(scene)
    variants = {}
    for task in remaining:
        if task.work_kind != 'STRAIGHT_BODY':
            raise ValueError('RESUME_REQUIRES_STRAIGHT_BODY_TASKS')
        variants[task.task_id], issues = io._task_variants(task, scene, validator)
        if not variants[task.task_id]:
            raise ValueError('RESUME_INVALID_TASK:' + task.task_id + ':' + ','.join(issues))
    prefix_count = len(seen)
    result.routes = [r for r in result.routes if r.status == 'REGION_ROUTE_PASS']
    result.failures = []
    deadline = time.perf_counter() + seconds
    connector = continue_Connector(scene, settings, deadline)
    trace = []
    queries = 0
    while remaining and time.perf_counter() < deadline:
        ranked = sorted(remaining, key=lambda t: (
            min(math.hypot(v.start.x-previous.x, v.start.y-previous.y)
                for v in variants[t.task_id]), t.row_index, t.task_id))
        accepted = []
        # Up to two nearby legal next tasks are compared. Only failed queries
        # expand the neighborhood; do not enumerate all possible task orders.
        for medial in (False, True):
            connector.medial_enabled = medial
            for task in ranked:
                if time.perf_counter() >= deadline:
                    break
                connector.deadline = min(deadline, time.perf_counter()+query_seconds*(2 if medial else 1))
                candidate = continue_candidate_region(regions[task.region_id], [task],
                    'AUDITED_PREFIX_CONTINUATION', variants, scene, connector,
                    headland, completed, worked, previous, before, None)
                queries += 1
                if candidate.status == 'REGION_ROUTE_PASS':
                    # The endpoint distance is a cheap one-step lookahead,
                    # not a feasibility claim for the following connection.
                    end = io._pose_at(candidate.motions[-1], True)
                    next_distance = min((math.hypot(v.start.x-end.x, v.start.y-end.y)
                        for other in ranked if other.task_id != task.task_id
                        for v in variants[other.task_id]), default=0.0)
                    accepted.append((candidate.connection_seconds + next_distance /
                        scene.vehicle.turn_speed_mps, task.task_id, task, candidate))
                    if len(accepted) >= 2:
                        break
            if accepted:
                break
        connector.deadline = deadline
        if not accepted:
            break
        _, _, task, candidate = min(accepted, key=lambda x: x[:2])
        result.routes.append(candidate)
        result.region_order.append(task.region_id)
        completed = completed.union(candidate.completed_sweep.intersection(scene.target))
        worked[task.region_id] = worked.get(task.region_id, GeometryCollection()).union(candidate.completed_sweep)
        previous = io._pose_at(candidate.motions[-1], True)
        before = task
        seen.add(task.task_id)
        remaining.remove(task)
        trace.append({'task_id':task.task_id, 'remaining':len(remaining),
                      'body_gap_m2':result.required_body.difference(completed).area})
    finish_ok = scene.end is None
    if not remaining and scene.end is not None:
        connector.deadline = deadline
        final = connector.connect(previous, scene.end, before, None,
            io._ready_area(scene, headland, completed), worked, transfer=True)
        finish_ok = final is not None
        if final is not None and final.motion is not None:
            result.routes[-1].motions.append(final.motion)
            result.routes[-1].connections.append(final)
            result.routes[-1].connection_seconds += final.seconds
    reason = ('RESUME_SEARCH_BUDGET_EXHAUSTED' if time.perf_counter() >= deadline
              else 'RESUME_NO_LEGAL_NEXT_TASK')
    for region_id in sorted({t.region_id for t in remaining}):
        result.routes.append(io.RegionRoute(region_id, 'REGION_ROUTE_NOT_FOUND', reason=reason))
        result.failures.append({'field_id':job.field_id,'region_id':region_id,'code':reason})
    gap = result.required_body.difference(completed).area
    if gap > scene.settings.coverage_tolerance_m2:
        result.failures.append({'field_id':job.field_id,'region_id':'__BODY__',
                               'code':'REPLANNED_BODY_COVERAGE_GAP','area_m2':gap})
    if not remaining and not finish_ok:
        result.failures.append({'field_id':job.field_id,'region_id':'__END__',
                               'code':'FIELD_END_UNREACHABLE'})
    result.status = ('FIELD_ROUTE_COMPLETE' if not remaining and finish_ok and
                     gap <= scene.settings.coverage_tolerance_m2 else 'PARTIAL_CONTINUOUS_ROUTE')
    result.transfer_status = ('ALL_REQUIRED_TRANSFERS_CONNECTED' if not remaining and finish_ok
                              else 'PARTIAL_OR_NOT_REACHED')
    # Independent headland samples are unchanged. Added body work can only
    # reduce their residual gap; there is still no body-to-headland itinerary.
    record = result.preparation.get('separate_headland')
    if record:
        combined = completed.union(unary_union([t.frozen_sweep for t in result.separate_headland_tasks]))
        missing = headland.difference(combined).area
        record.update(body_already_covered_headland_m2=completed.intersection(headland).area,
            headland_planned_covered_m2=combined.intersection(headland).area,
            headland_planned_missing_m2=missing,
            planned_target_missing_m2=scene.target.difference(combined).area,
            coverage_status='COVERAGE_COMPLETE' if missing <= scene.settings.coverage_tolerance_m2 else 'COVERAGE_GAP')
        result.statistics['separate_headland'] = deepcopy(record)
    result.statistics['prefix_continuation'] = {
        'initial_completed_tasks':prefix_count,'added_tasks':len(seen)-prefix_count,
        'remaining_tasks':len(remaining),'queries':queries,'trace':trace,
        'counts':dict(connector.counts),'stage_seconds':dict(connector.timing),
        'search_status':'COMPLETE' if not remaining and finish_ok else reason,
        'seconds':time.perf_counter()-started,'budget_seconds':seconds,
        'prefix_preserved':True,'tasks_or_target_changed':False}
    result.elapsed_s += time.perf_counter()-started
    return result


# 起步恢复：先建立真实长条带作业起点

def seed_and_continue(result, settings, *, seconds=60.0):
    """以已验收路线为前缀尝试有限续作；预算与必须主体义务保持明确，来源不会覆盖。"""
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError('SEED_BUDGET_MUST_BE_POSITIVE_FINITE')
    if result.adapted_job is None or result.required_body is None:
        raise ValueError('SEED_REQUIRES_ADAPTED_INPUT')
    if any(r.status=='REGION_ROUTE_PASS' and r.task_order for r in result.routes):
        return continue_body(result,settings,seconds=seconds)
    started=time.perf_counter();job=result.adapted_job
    scene=io.load_scene(job.scene_path);validator=io.Validator(scene)
    tasks=[t for t in job.tasks if t.work_kind=='STRAIGHT_BODY' and t.length_m>=1.0]
    if len(tasks)!=len(job.tasks) or not tasks:
        raise ValueError('SEED_REQUIRES_NONEMPTY_VALID_BODY_TASKS')
    row_middle=(min(t.row_index for t in tasks)+max(t.row_index for t in tasks))/2
    # Long central rows provide work space on both sides before the shorter
    # corner rows are visited. Equivalent headings keep the same tool sweep.
    ranked=sorted(tasks,key=lambda t:(-t.length_m,abs(t.row_index-row_middle),t.task_id))
    variants=[];seed=None
    for task in ranked:
        variants,_=io._task_variants(task,scene,validator)
        if variants:
            seed=task;break
    if seed is None:raise ValueError('NO_PHYSICALLY_VALID_SEED')
    variants=sorted(variants,key=lambda v:v.reversed)[:2]
    deadline=time.perf_counter()+seconds;attempts=[];records=[]
    for number,variant in enumerate(variants):
        remaining=deadline-time.perf_counter()
        if remaining<=0:break
        candidate=deepcopy(result);candidate.routes=[];candidate.region_order=[];candidate.failures=[]
        headland=scene.target.difference(candidate.required_body)
        motions=[];connections=[]
        if scene.start is not None:
            connector=continue_Connector(scene,settings,min(deadline,time.perf_counter()+2.5))
            incoming=connector.connect(scene.start,variant.start,None,seed,
                io._ready_area(scene,headland,GeometryCollection()),{},transfer=False)
            if incoming is None:
                records.append({'seed_task':seed.task_id,'reversed':variant.reversed,
                                'status':'NO_SAFE_CONFIGURED_START_CONNECTION'})
                continue
            if incoming.motion is not None:motions.append(incoming.motion);connections.append(incoming)
        motions.append(variant.motion)
        route=io.RegionRoute(seed.region_id,'REGION_ROUTE_PASS',
            order_mode='VERIFIED_LONG_ROW_SEED',motions=motions,connections=connections,
            task_order=[seed.task_id],completed_sweep=variant.sweep,
            connection_seconds=sum(c.seconds for c in connections),
            reverse_m=sum(c.reverse_m for c in connections),
            gear_shifts=sum(c.gear_shifts for c in connections),sweep_error_m2=variant.sweep_error_m2)
        candidate.routes=[route];candidate.region_order=[seed.region_id]
        candidate.status='PARTIAL_CONTINUOUS_ROUTE'
        # Reserve half the total budget for the opposite equivalent heading.
        trial_budget=max(.01,(deadline-time.perf_counter())/max(1,len(variants)-number))
        candidate=continue_body(candidate,settings,seconds=trial_budget)
        completed=unary_union([r.completed_sweep for r in candidate.routes if r.status=='REGION_ROUTE_PASS'])
        gap=candidate.required_body.difference(completed).area
        records.append({'seed_task':seed.task_id,'reversed':variant.reversed,
            'status':candidate.status,'body_gap_m2':gap,
            'completed_tasks':sum(len(r.task_order) for r in candidate.routes if r.status=='REGION_ROUTE_PASS')})
        attempts.append((candidate.status=='FIELD_ROUTE_COMPLETE',-gap,candidate))
        if candidate.status=='FIELD_ROUTE_COMPLETE':break
    if not attempts:
        candidate=deepcopy(result)
    else:
        candidate=max(attempts,key=lambda x:x[:2])[2]
    candidate.statistics['long_row_seed']={'attempts':records,
        'free_start_used':scene.start is None,'task_set_changed':False,
        'elapsed_seconds':time.perf_counter()-started,'budget_seconds':seconds}
    # A failed attempt is not an impossibility certificate. Task-set, sweep
    # and field safety are independently checked when this candidate is exported.
    return candidate


# 末条插入：保留已执行任务和实际覆盖

def insert_last_task(result,settings,*,seconds=30.0):
    """复制候选并有界插入末条任务，接入和返回都需验证，不能只补一条展示线。"""
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('INVALID_INSERT_BUDGET')
    started=time.perf_counter();candidate=deepcopy(result);job=candidate.adapted_job
    if job is None or candidate.required_body is None:raise ValueError('MISSING_ADAPTED_INPUT')
    scene=io.load_scene(job.scene_path)
    headland,completed,worked,done,pose,before=checked_prefix(candidate,scene)
    remaining=[t for t in job.tasks if t.task_id not in done]
    if len(remaining)!=1:
        candidate.statistics['terminal_insert']={'status':'NOT_APPLICABLE','remaining_tasks':len(remaining)}
        return candidate
    routes=[r for r in candidate.routes if r.status=='REGION_ROUTE_PASS']
    validator=io.Validator(scene);variants={}
    for task in job.tasks:
        variants[task.task_id],_=io._task_variants(task,scene,validator)
        if not variants[task.task_id]:raise ValueError('INVALID_BODY_VARIANT:'+task.task_id)
    task=remaining[0];region=next(r for r in job.regions if r.region_id==task.region_id)
    replacement,diagnostic=_continue_replan_last_body_task(routes,candidate.region_order,
        [(region,[task])],job,scene,replace(settings,last_body_replan_seconds=seconds),
        variants,headland,medial_enabled=True)
    candidate.statistics['terminal_insert']=diagnostic
    if replacement is None:return candidate
    candidate.routes,candidate.region_order=replacement
    # Independently replay physics and dynamically worked space after the
    # insertion, not just the changed pair's successful connector result.
    headland,completed,worked,done,pose,before=checked_prefix(candidate,scene)
    complete=set(t.task_id for t in job.tasks)==done
    body_gap=candidate.required_body.difference(completed).area
    end_ok=scene.end is None or io._pose_close(pose,scene.end,scene)
    if complete and not end_ok:
        connector=continue_Connector(scene,settings,time.perf_counter()+min(2.5,seconds))
        final=connector.connect(pose,scene.end,before,None,
            io._ready_area(scene,headland,completed),worked,transfer=True)
        end_ok=final is not None
        if final is not None and final.motion is not None:
            candidate.routes[-1].motions.append(final.motion)
            candidate.routes[-1].connections.append(final)
            candidate.routes[-1].connection_seconds+=final.seconds
    candidate.status=('FIELD_ROUTE_COMPLETE' if complete and end_ok and
        body_gap<=scene.settings.coverage_tolerance_m2 else 'PARTIAL_CONTINUOUS_ROUTE')
    candidate.transfer_status='ALL_REQUIRED_TRANSFERS_CONNECTED' if complete and end_ok else 'PARTIAL_OR_NOT_REACHED'
    candidate.failures=[]
    if body_gap>scene.settings.coverage_tolerance_m2:
        candidate.failures.append({'field_id':job.field_id,'region_id':'__BODY__',
            'code':'REPLANNED_BODY_COVERAGE_GAP','area_m2':body_gap})
    if not end_ok:candidate.failures.append({'field_id':job.field_id,'region_id':'__END__','code':'FIELD_END_UNREACHABLE'})
    record=candidate.preparation.get('separate_headland')
    if record:
        combined=completed.union(unary_union([t.frozen_sweep for t in candidate.separate_headland_tasks]))
        gap=headland.difference(combined).area
        record.update(body_already_covered_headland_m2=completed.intersection(headland).area,
            headland_planned_covered_m2=combined.intersection(headland).area,
            headland_planned_missing_m2=gap,planned_target_missing_m2=scene.target.difference(combined).area,
            coverage_status='COVERAGE_COMPLETE' if gap<=scene.settings.coverage_tolerance_m2 else 'COVERAGE_GAP')
        candidate.statistics['separate_headland']=deepcopy(record)
    candidate.elapsed_s+=time.perf_counter()-started
    return candidate


# 主体保留：重放时序，拒绝未真正覆盖的目标

# 覆盖保护依据必需主体目标及实际机具扫掠，而不是某个候选的声明面积。
# 允许已有有效条带承担目标，但不因补作丢失原先必需的任务贡献。
def preserve_required_body(result, baseline_body):
    """要求来源具有完整主体义务，再核对恢复方案仍保留全部必要主体覆盖。"""
    started = time.perf_counter()
    if result.status != 'FIELD_ROUTE_COMPLETE' or result.adapted_job is None:
        raise ValueError('PRESERVE_REQUIRES_COMPLETE_BODY_ROUTE')
    if result.required_body is None or not baseline_body.is_valid:
        raise ValueError('PRESERVE_REQUIRES_VALID_BODY_GEOMETRY')
    scene = io.load_scene(result.adapted_job.scene_path)
    if baseline_body.difference(scene.target).area > 1e-6:
        raise ValueError('BASELINE_BODY_OUTSIDE_TARGET')
    _, completed, _, done, _, _ = checked_prefix(result, scene)
    expected = {task.task_id for task in result.adapted_job.tasks}
    if done != expected:
        raise ValueError('PRESERVE_INCOMPLETE_TASK_SET')
    preserved = result.required_body.union(baseline_body)
    missing = preserved.difference(completed).area
    if missing > scene.settings.coverage_tolerance_m2:
        raise ValueError('BASELINE_BODY_NOT_ACTUALLY_COVERED')

    candidate = deepcopy(result)
    added_area = baseline_body.difference(result.required_body).area
    candidate.required_body = preserved
    headland = scene.target.difference(preserved)
    candidate.adapted_job = replace(candidate.adapted_job, regions=tuple(
        replace(region, headland=region.geometry.intersection(headland))
        for region in candidate.adapted_job.regions))
    # Shrinking the reserve makes implement-off travel more restrictive.
    # A final-area check alone cannot prove this chronological condition.
    checked_prefix(candidate, scene)
    candidate.preparation.update(
        required_body_area_m2=preserved.area,
        headland_reserve_area_m2=headland.area,
        body_missing_m2=missing)
    record = candidate.preparation.get('separate_headland')
    if record is not None:
        combined = completed.union(unary_union([
            task.frozen_sweep for task in candidate.separate_headland_tasks]))
        gap = headland.difference(combined).area
        record.update(
            headland_area_m2=headland.area,
            body_already_covered_headland_m2=completed.intersection(headland).area,
            headland_planned_covered_m2=combined.intersection(headland).area,
            headland_planned_missing_m2=gap,
            planned_target_missing_m2=scene.target.difference(combined).area,
            coverage_status=('COVERAGE_COMPLETE' if gap <=
                             scene.settings.coverage_tolerance_m2 else 'COVERAGE_GAP'))
        candidate.statistics['separate_headland'] = deepcopy(record)
    candidate.statistics['baseline_body_preservation'] = {
        'status': 'ACTUAL_COVERAGE_AND_DYNAMIC_REPLAY_PASS',
        'restored_body_area_m2': added_area,
        'required_body_missing_m2': missing,
        'body_target_shrunk': False,
        'motions_or_tasks_changed': False,
        'elapsed_seconds': time.perf_counter() - started}
    candidate.elapsed_s += time.perf_counter() - started
    return candidate

# ==========================================================================
# 7. 主体和田头任务的有界联合组织
# 在有限候选和时间预算内插入未完成田头任务，不改变验收优先级。
# ==========================================================================

import copy
from dataclasses import replace
import math
import time

from shapely.geometry import GeometryCollection
from shapely.ops import unary_union

import route_planner as io
joint_HEADLAND_REGION_ID = headland_HEADLAND_REGION_ID
joint_generate = generate_headland_tasks


def _rank_tasks(remaining, variants, previous, priority=(), mode="MEAN"):
    """按当前位姿和明确优先任务排序剩余候选，只影响有界搜索次序，不证明可达。"""
    priority = set(priority)
    if previous is None:
        return sorted(remaining,key=lambda t:(t.task_id not in priority,
            t.row_index,t.suggested_order,t.task_id))
    def entry_cost(task):
        options = variants[task.task_id]
        if not options:
            return math.inf
        distances = [(option.start.x-previous.x)**2+
                     (option.start.y-previous.y)**2 for option in options]
        # Keep the established mean rank for normal fields. A bounded retry
        # may use the nearest selectable entry when that headland order stalls.
        return min(distances) if mode == "MIN" else sum(distances)/len(distances)
    return sorted(remaining,key=lambda task:(
        task.task_id not in priority,
        entry_cost(task),
        task.row_index,task.task_id))


def _bounded_headland_search(tasks, variants, region, scene, connector,
                             headland, completed, previous, previous_task,
                             worked_by_region, max_nodes, priority=()):
    """首选续作失败时再试有限次选，节点数与墙钟期限共同控制搜索。
    
    Try a second feasible next task only when the first continuation fails.
    
    The node count and wall-clock deadline both bound the search. A returned
    partial sequence means only that this bounded search failed; it is never a
    proof of physical disconnection."""
    pass  # 已合并到本模块，直接使用下方的定义。

    nodes = 0
    attempts = 0
    best = ([], completed, previous, previous_task, list(tasks))
    best_score = (0, completed.intersection(scene.target).area)

    def visit(remaining, done, pose, before, routes):
        nonlocal nodes, attempts, best, best_score
        score = (len(routes), done.intersection(scene.target).area)
        if score > best_score:
            best_score = score
            best = (list(routes), done, pose, before, list(remaining))
        if not remaining:
            return True
        if nodes >= max_nodes or time.perf_counter() >= connector.deadline:
            return False
        ranked = _rank_tasks(remaining, variants, pose, priority,
                             connector.settings.headland_entry_rank)
        distinct_found = 0
        for task in ranked:
            if nodes >= max_nodes or time.perf_counter() >= connector.deadline:
                break
            if not variants[task.task_id]:
                continue
            task_found = False
            for direction in sorted({v.reversed for v in variants[task.task_id]}):
                if nodes >= max_nodes or time.perf_counter()>=connector.deadline:
                    break
                candidate = candidate_region(region,[task],"JOINT_HEADLAND_BRANCH",
                    variants,scene,connector,headland,done,
                    {**worked_by_region,joint_HEADLAND_REGION_ID:done},
                    pose,before,direction)
                attempts += 1
                if candidate.status != "REGION_ROUTE_PASS":
                    continue
                task_found = True
                nodes += 1
                next_done = unary_union([done,candidate.completed_sweep])
                next_pose = io._pose_at(candidate.motions[-1],True)
                next_remaining = [x for x in remaining if x.task_id!=task.task_id]
                if visit(next_remaining,next_done,next_pose,task,
                         [*routes,candidate]):
                    return True
            if task_found:
                distinct_found += 1
            if distinct_found >= 3:
                break
        return False

    visit(list(tasks),completed,previous,previous_task,[])
    return best, {"branch_nodes":nodes,"branch_attempts":attempts,
                  "branch_limit_reached":nodes>=max_nodes,
                  "branch_time_exhausted":time.perf_counter()>=connector.deadline}


def _split_body_routes(routes, variants, block_size):
    """只在作业运动之间拆分已核查主体行程，保留原运动、任务顺序和分区归属。
    
    Split an audited body chain only between work motions.
    
    Every original motion, task order, and region ID is preserved. The
    previously checked internal turns stay with their work block; only the
    incoming turn of a block may be replaced after a headland insertion."""
    result = []
    for route in routes:
        if route.status != "REGION_ROUTE_PASS":
            continue
        indices = [index for index, motion in enumerate(route.motions)
                   if motion.task_id in route.task_order]
        if len(indices) != len(route.task_order):
            raise ValueError("BODY_WORK_MOTION_COUNT_MISMATCH")
        for first in range(0, len(indices), block_size):
            last = min(first + block_size, len(indices))
            begin_motion = 0 if first == 0 else indices[first - 1] + 1
            end_motion = indices[last - 1] + 1
            motions = route.motions[begin_motion:end_motion]
            motion_ids = {id(motion) for motion in motions}
            links = [link for link in route.connections
                     if link.motion is not None and id(link.motion) in motion_ids]
            task_ids = route.task_order[first:last]
            sweep = unary_union([variants[task_id][0].sweep
                                 for task_id in task_ids])
            result.append(io.RegionRoute(
                region_id=route.region_id, status="REGION_ROUTE_PASS",
                order_mode="AUDITED_BODY_SUBBLOCK", motions=list(motions),
                connections=links, task_order=list(task_ids),
                completed_sweep=sweep,
                connection_seconds=sum(link.seconds for link in links),
                reverse_m=sum(link.reverse_m for link in links),
                gear_shifts=sum(link.gear_shifts for link in links)))
    return result


def _advance(route, taskmap, completed, worked):
    """用已通过分区路线更新完成扫掠与位置，未来任务不能提前提供通行区。"""
    completed = unary_union([completed, route.completed_sweep])
    worked = dict(worked)
    worked[route.region_id] = unary_union([
        worked.get(route.region_id, GeometryCollection()), route.completed_sweep])
    return (completed, worked, io._pose_at(route.motions[-1], True),
            taskmap[route.task_order[-1]])


def _checked_pass_order(routes, region_order):
    """确认记录的执行顺序对应真正通过的作业路线，而非占位失败分区。
    
    Execution order tracks real work routes, not placeholder regions."""
    passed = [route.region_id for route in routes
              if route.status == "REGION_ROUTE_PASS"]
    if passed != list(region_order):
        raise ValueError("JOINT_ROUTE_ORDER_COUNT_MISMATCH")


def _rank_splice_places(task, options, choices, routes, taskmap,
                        *, last_tasks):
    """按真实作业边界及相邻位姿排序插入点，不只用远端端点距离评价邻近田头圈。
    
    Keep nearby work boundaries even when their endpoints are far apart.
    
    Parallel headland passes can be one implement width apart but run in the
    same direction. Their preceding end and the new pass start can therefore
    be a full row length apart. Endpoint-only ranking omitted these useful
    insertion points before the connector was ever asked to validate them."""
    scored = []
    for place in choices:
        _, following_index, _, _, pose, before, next_pose = place
        endpoint = min(
            (math.hypot(option.start.x-pose.x, option.start.y-pose.y)
             if pose is not None else 0.0) +
            (math.hypot(option.end.x-next_pose.x,
                        option.end.y-next_pose.y)
             if next_pose is not None else 0.0)
            for option in options)
        adjacent = []
        if before is not None:
            adjacent.append(task.reference_line.distance(
                before.reference_line))
        if following_index is not None:
            first_id = routes[following_index].task_order[0]
            adjacent.append(task.reference_line.distance(
                taskmap[first_id].reference_line))
        scored.append((min(adjacent, default=math.inf), endpoint, place))
    by_endpoint = sorted(scored, key=lambda row: (row[1], row[0], row[2][0]))
    if not last_tasks:
        return [row[2] for row in by_endpoint[:8]]
    by_neighbor = sorted(scored, key=lambda row: (row[0], row[1], row[2][0]))
    selected = []
    seen = set()
    for row in [*by_neighbor[:8], *by_endpoint[:8]]:
        index = row[2][0]
        if index not in seen:
            selected.append(row[2])
            seen.add(index)
    return selected


def _replace_incoming(original, taskmap, scene, connector, headland,
                      completed, worked, pose, before):
    """替换已检查插入位置的入场连接，保留后继任务顺序，返回仍由上层核对。"""
    first_id = original.task_order[0]
    first_index = next(index for index, motion in enumerate(original.motions)
                       if motion.task_id == first_id)
    if first_index > 1:
        raise ValueError("BODY_BLOCK_PREFIX_UNEXPECTED")
    first_task = taskmap[first_id]
    link = connector.connect(pose,
        io._pose_at(original.motions[first_index], False),
        before, first_task, io._ready_area(scene, headland, completed),
        worked, transfer=before.region_id != first_task.region_id)
    if link is None:
        return None
    prefix = [link.motion] if link.motion is not None else []
    links = [link] if link.motion is not None else []
    tail = original.connections[1:] if first_index else original.connections
    all_links = links + tail
    return replace(original,
        motions=prefix + original.motions[first_index:],
        connections=all_links,
        connection_seconds=sum(item.seconds for item in all_links),
        reverse_m=sum(item.reverse_m for item in all_links),
        gear_shifts=sum(item.gear_shifts for item in all_links))


def _interleaved_attempt(seed, adapted, scene, headland, settings):
    """在保留的主体子区块之间有界插入田头任务，入出两端都需保持连续。
    
    Bounded headland insertions between preserved body work subblocks."""
    pass  # 已合并到本模块，直接使用下方的定义。

    started = time.perf_counter()
    connector = Connector(scene, settings,
                          started + settings.max_headland_seconds)
    taskmap = {task.task_id: task for task in adapted.tasks}
    regions = {region.region_id: region for region in adapted.regions}
    validator = io.Validator(scene)
    variants = {task.task_id: io._task_variants(task, scene, validator)[0]
                for task in adapted.tasks}
    invalid = [task_id for task_id, options in variants.items() if not options]
    if invalid:
        return None, {"status":"INVALID_TASK_VARIANT",
                      "task_ids":invalid[:5]}

    # The body solver may have attached the requested final exit. A joint
    # route may visit that exit only after all headland work is complete.
    if scene.end is not None:
        passed = [route for route in seed.routes
                  if route.status == "REGION_ROUTE_PASS"]
        if passed:
            last = passed[-1]
            if (last.connections and last.connections[-1].to_task is None
                    and last.connections[-1].motion is not None
                    and last.motions[-1] is last.connections[-1].motion):
                last.connections.pop()
                last.motions.pop()
    spacing = max(scene.vehicle.working_width_m, 1e-6)
    block_size = max(3, min(6, round(
        2 * scene.vehicle.min_turn_radius_m / spacing) + 1))
    blocks = _split_body_routes(seed.routes, variants, block_size)
    if not blocks:
        return None, {"status":"NO_BODY_BLOCKS"}
    all_body_ids = [task_id for block in blocks for task_id in block.task_order]
    if len(all_body_ids) != len(set(all_body_ids)):
        raise ValueError("DUPLICATED_BODY_TASK_IN_JOINT_BLOCKS")
    remaining = [task for task in adapted.tasks
                 if task.region_id == joint_HEADLAND_REGION_ID]
    completed = GeometryCollection()
    worked = {}
    pose, before = scene.start, None
    routes = [route for route in seed.routes if route.status == "HEADLAND_ONLY"]
    region_order = []
    inserted = []
    proposals = entered = reentered = 0

    def headland_task(task, current_completed, current_worked,
                      current_pose, previous_task):
        if time.perf_counter() >= connector.deadline:
            return None
        # Evaluate both physical entry directions together. A feasible first
        # direction can still require a much longer loop than the other end.
        directions = ([None] if settings.headland_compare_entry_directions else
                      sorted({v.reversed for v in variants[task.task_id]}))
        for direction in directions:
            attempt = candidate_region(
                regions[joint_HEADLAND_REGION_ID], [task], "INTERLEAVED_HEADLAND",
                variants, scene, connector, headland, current_completed,
                current_worked, current_pose, previous_task, direction)
            if attempt.status == "REGION_ROUTE_PASS":
                return attempt
        return None

    for index, original in enumerate(blocks):
        if time.perf_counter() >= connector.deadline:
            break
        selected = None
        if index and remaining:
            for task in _rank_tasks(remaining, variants, pose,
                                    mode=settings.headland_entry_rank)[:4]:
                proposals += 1
                head = headland_task(task, completed, worked, pose, before)
                if head is None:
                    continue
                entered += 1
                tentative = _advance(head, taskmap, completed, worked)
                following = _replace_incoming(original, taskmap, scene,
                    connector, headland, *tentative)
                if following is not None:
                    reentered += 1
                    selected = (task, head, following)
                    break
        if selected is None:
            segment = original
            if (pose is not None and not io._pose_close(
                    pose, io._pose_at(segment.motions[0], False), scene)):
                raise ValueError("JOINT_BODY_BLOCK_ORIGIN_CHANGED")
        else:
            task, head, segment = selected
            routes.append(head)
            region_order.append(joint_HEADLAND_REGION_ID)
            inserted.append(task.task_id)
            remaining.remove(task)
            completed, worked, pose, before = _advance(
                head, taskmap, completed, worked)
        routes.append(segment)
        region_order.append(segment.region_id)
        completed, worked, pose, before = _advance(
            segment, taskmap, completed, worked)

    finished_body = [task_id for route in routes
                     if route.status == "REGION_ROUTE_PASS"
                     and route.region_id != joint_HEADLAND_REGION_ID
                     for task_id in route.task_order]
    if finished_body != all_body_ids:
        return None, {"status":"BODY_REPLAY_INCOMPLETE",
                      "completed_body_tasks":len(finished_body),
                      "required_body_tasks":len(all_body_ids),
                      "elapsed_s":time.perf_counter()-started}

    while remaining and time.perf_counter() < connector.deadline:
        selected = None
        for task in _rank_tasks(remaining, variants, pose,
                                mode=settings.headland_entry_rank):
            if time.perf_counter() >= connector.deadline:
                break
            head = headland_task(task, completed, worked, pose, before)
            if head is not None:
                selected = (task, head)
                break
        if selected is None:
            break
        task, head = selected
        routes.append(head)
        region_order.append(joint_HEADLAND_REGION_ID)
        remaining.remove(task)
        completed, worked, pose, before = _advance(
            head, taskmap, completed, worked)
    motions = [motion for route in routes if route.status == "REGION_ROUTE_PASS"
               for motion in route.motions]
    continuity = io._route_continuity_issues(motions, scene)
    if continuity:
        raise ValueError("JOINT_INTERLEAVED_DISCONNECTED:"+",".join(continuity[:3]))
    finished = [task_id for route in routes
                if route.region_id == joint_HEADLAND_REGION_ID
                for task_id in route.task_order]
    stats = {"status":"CANDIDATE", "block_size":block_size,
             "body_block_count":len(blocks), "inserted_task_ids":inserted,
             "proposed_insertion_tasks":proposals,
             "headland_entry_feasible":entered,
             "body_reentry_feasible":reentered,
             "completed_headland_tasks":len(finished),
             "remaining_headland_tasks":len(remaining),
             "elapsed_s":time.perf_counter()-started,
             "connector_counts":dict(connector.counts)}
    return (routes, region_order, finished, remaining, completed,
            worked, pose, before), stats


def _register_supplemental_tasks(body, headland, tasks, construction):
    """把通过的补作同时加入任务、分区与面积账本，避免路线里出现无归属作业。
    
    Keep accepted supplemental work in the task, region, and area ledgers."""
    if not tasks:
        return
    adapted = body.adapted_job
    region = replace(adapted.regions[-1],
                     task_ids=(*adapted.regions[-1].task_ids,
                               *(task.task_id for task in tasks)))
    body.adapted_job = replace(adapted,
        tasks=(*adapted.tasks, *tasks),
        regions=(*adapted.regions[:-1], region))
    diagnostic = body.preparation["headland_work"]
    diagnostic["headland_task_count"] += len(tasks)
    diagnostic["headland_straight_task_count"] += len(tasks)
    diagnostic.setdefault("supplemental_task_ids", []).extend(
        task.task_id for task in tasks)
    candidate_sweeps = unary_union([
        task.frozen_sweep for task in body.adapted_job.tasks
        if task.region_id == joint_HEADLAND_REGION_ID])
    diagnostic["headland_candidate_missing_m2"] = (
        headland.difference(candidate_sweeps).area)
    diagnostic["headland_candidate_covered_m2"] = (
        candidate_sweeps.intersection(headland).area)
    mapping = [{
        "task_id": task.task_id,
        "source_task_id": None,
        "region_id": joint_HEADLAND_REGION_ID,
        "construction": construction,
    } for task in tasks]
    diagnostic["headland_task_mapping"].extend(mapping)
    body.preparation["task_mapping"].extend(mapping)


def _omit_headland_work_already_done(adapted, diagnostic, scene, headland,
                                    completed_body, *, retain_access_work=False):
    """移除已被实际主体扫掠覆盖且无新增目标收益的生成田头任务，不删尚未完成的必要覆盖。
    
    Remove generated passes that cannot add target coverage after body work.
    
    The headland generator does not know which body sweeps the selected route
    has actually executed.  Nested headland rings can therefore include
    passes fully covered by the body.  Keep the generated IDs in the audit
    record, but never require a duplicate physical pass to finish the field."""
    tolerance = scene.settings.coverage_tolerance_m2
    headland_tasks = [task for task in adapted.tasks
                      if task.region_id == joint_HEADLAND_REGION_ID]
    redundant = [task for task in headland_tasks
                 if task.frozen_sweep.intersection(scene.target).difference(
                     completed_body).area <= tolerance]
    # A per-task tolerance must not accumulate into a real field omission.
    # When the combined residual is too large, retain every pass with a real
    # positive contribution and omit only numerical-zero footprints.
    redundant_residual=unary_union([task.frozen_sweep for task in redundant]).intersection(
        scene.target).difference(completed_body).area
    if redundant_residual>tolerance:
        redundant=[task for task in redundant
                   if task.frozen_sweep.intersection(scene.target).difference(
                       completed_body).area<=1e-10]
    redundant_ids = {task.task_id for task in redundant}
    useful = [task for task in headland_tasks
              if task.task_id not in redundant_ids]
    useful_sweeps = unary_union([task.frozen_sweep for task in useful])
    anchor_distance = max(10 * scene.settings.geometry_epsilon_m,
                          scene.vehicle.working_width_m * 0.02)
    adjacent = ([task for task in redundant
                if task.frozen_sweep.distance(useful_sweeps) <= anchor_distance]
               if not useful_sweeps.is_empty else [])
    anchors=adjacent if retain_access_work else []
    anchor_ids = {task.task_id for task in anchors}
    omitted = [task for task in redundant if task.task_id not in anchor_ids]
    diagnostic["headland_generated_task_count"] = len(headland_tasks)
    diagnostic["body_redundant_task_ids"] = [task.task_id for task in redundant]
    diagnostic["body_redundant_task_count"] = len(redundant)
    diagnostic["body_redundancy_tolerance_m2"] = tolerance
    diagnostic["body_redundant_access_anchor_ids"] = [
        task.task_id for task in anchors]
    diagnostic['optional_transit_anchor_source_task_ids']=[task.task_id for task in adjacent]
    diagnostic['optional_transit_straight_tasks']=[{
        'task_id':task.task_id,'reference_line_wkt':task.reference_line.wkt,
        'heading_rad':task.heading_rad,'sweep_wkt':task.frozen_sweep.wkt,
        'row_index':task.row_index,'suggested_order':task.suggested_order}
        for task in adjacent if task.motion_points is None]
    diagnostic["body_redundant_omitted_task_ids"] = [
        task.task_id for task in omitted]
    diagnostic["body_redundant_omitted_task_count"] = len(omitted)
    if not omitted:
        return adapted, diagnostic
    omitted_ids = {task.task_id for task in omitted}
    retained = [task for task in headland_tasks
                if task.task_id not in omitted_ids]
    region = replace(adapted.regions[-1],
                     task_ids=tuple(task.task_id for task in retained))
    adapted = replace(adapted,
        tasks=tuple(task for task in adapted.tasks
                    if task.task_id not in omitted_ids),
        regions=(*adapted.regions[:-1], region))
    diagnostic["headland_task_count"] = len(retained)
    diagnostic["headland_straight_task_count"] = sum(
        task.work_kind == "STRAIGHT_HEADLAND" for task in retained)
    diagnostic["headland_curved_task_count"] = sum(
        task.work_kind == "CURVED_HEADLAND" for task in retained)
    diagnostic["headland_task_mapping"] = [row for row in
        diagnostic["headland_task_mapping"]
        if row["task_id"] not in omitted_ids]
    diagnostic["generated_headland_candidate_missing_m2"] = (
        diagnostic["headland_candidate_missing_m2"])
    retained_sweeps = unary_union([task.frozen_sweep for task in retained])
    total_supported=unary_union([retained_sweeps,completed_body])
    before_supported=unary_union([completed_body,*[
        task.frozen_sweep for task in headland_tasks]])
    coverage_loss=before_supported.intersection(scene.target).difference(total_supported).area
    if coverage_loss>tolerance:
        raise ValueError('REDUNDANT_HEADLAND_OMISSION_LOST_TARGET_COVERAGE')
    diagnostic['redundant_omission_target_loss_m2']=coverage_loss
    diagnostic["headland_candidate_missing_m2"] = (
        headland.difference(total_supported).area)
    diagnostic["headland_candidate_covered_m2"] = (
        total_supported.intersection(headland).area)
    return adapted, diagnostic


def _append_residual_bays(body, scene, headland, settings, completed,
                          worked, pose, before, finished):
    """在联合行程后尝试可达的狭长残余补作，只接受真实新增覆盖和合法运动。
    
    Add reachable straight work in omitted long bays after the joint chain.
    
    The generator proposes from the remaining original target. Only a real
    Fields2Cover connection followed by the checked work motion can enter the
    route. A bounded search preserves the existing safe itinerary on failure."""
    pass  # 已合并到本模块，直接使用下方的定义。
    pass  # 已合并到本模块，直接使用下方的定义。

    started = time.perf_counter()
    gap_before = scene.target.difference(completed).area
    proposed = propose_bay_tasks(scene, scene.target.difference(completed),
                                 completed)
    stats = {"status": "NO_LONG_BAY_CANDIDATE", "candidate_count": len(proposed),
             "accepted_task_ids": [], "missing_before_m2": gap_before,
             "missing_after_m2": gap_before, "elapsed_s": 0.0}
    if not proposed:
        return completed, worked, pose, before, stats
    connector = Connector(scene, settings,
                          time.perf_counter() + min(5.0,
                                                    settings.max_headland_seconds * 0.1))
    validator = io.Validator(scene)
    options = {}
    for task in proposed:
        options[task.task_id], _ = io._task_variants(task, scene, validator)
    remaining = [task for task in proposed if options[task.task_id]]
    taskmap = {task.task_id: task for task in body.adapted_job.tasks}
    region = body.adapted_job.regions[-1]
    accepted = []
    while remaining and time.perf_counter() < connector.deadline:
        remaining.sort(key=lambda task: min(
            (option.start.x - pose.x) ** 2 + (option.start.y - pose.y) ** 2
            for option in options[task.task_id]))
        chosen = None
        for task in remaining:
            if time.perf_counter() >= connector.deadline:
                break
            if (task.frozen_sweep.intersection(scene.target)
                    .difference(completed).area < 20.0):
                continue
            for direction in sorted({option.reversed
                                     for option in options[task.task_id]}):
                route = candidate_region(
                    region, [task], "RESIDUAL_BAY_WORK",
                    {task.task_id: options[task.task_id]}, scene, connector,
                    headland, completed, worked, pose, before, direction)
                if route.status == "REGION_ROUTE_PASS":
                    if scene.end is not None:
                        tentative_done = unary_union([completed,
                                                       route.completed_sweep])
                        tentative_worked = dict(worked)
                        tentative_worked[joint_HEADLAND_REGION_ID] = unary_union([
                            worked.get(joint_HEADLAND_REGION_ID,
                                       GeometryCollection()),
                            route.completed_sweep])
                        exit_link = connector.connect(
                            io._pose_at(route.motions[-1], True), scene.end,
                            task, None,
                            io._ready_area(scene, headland, tentative_done),
                            tentative_worked, transfer=False)
                        if exit_link is None:
                            continue
                    chosen = task, route
                    break
            if chosen:
                break
        if chosen is None:
            break
        task, route = chosen
        body.routes.append(route)
        body.region_order.append(joint_HEADLAND_REGION_ID)
        taskmap[task.task_id] = task
        completed, worked, pose, before = _advance(
            route, taskmap, completed, worked)
        accepted.append(task)
        finished.append(task.task_id)
        remaining.remove(task)
    if accepted:
        _register_supplemental_tasks(body, headland, accepted,
                                     "RESIDUAL_BAY_WORK")
    stats.update({"status": "IMPROVED" if accepted else "NO_REACHABLE_BAY",
                  "accepted_task_ids": [task.task_id for task in accepted],
                  "missing_after_m2": scene.target.difference(completed).area,
                  "elapsed_s": time.perf_counter() - started,
                  "search_limited": time.perf_counter() >= connector.deadline,
                  "connector_counts": dict(connector.counts)})
    return completed, worked, pose, before, stats


def _insert_residual_outer_edges(body, scene, headland, settings, completed,
                                 worked, pose, before, finished,
                                 candidate_source="OUTER",
                                 protected_candidates=GeometryCollection()):
    """对真实外边界补作检查接入与接回，两端裁切候选在有限预算中比较。
    
    Insert safe outer-edge work where it can enter *and* rejoin the route.
    
    Each real edge has two endpoint trims. Grouping them prevents an
    inaccessible long variant from exhausting the search before its shorter,
    easier-to-turn alternative is tested at the same itinerary position.
    Existing motion interiors are preserved; only the next route's incoming
    connection is replaced and checked against the enlarged worked area."""
    pass  # 已合并到本模块，直接使用下方的定义。
    pass  # 已合并到本模块，直接使用下方的定义。
    pass  # 已合并到本模块，直接使用下方的定义。

    started = time.perf_counter()
    gap_before = scene.target.difference(completed).area
    if candidate_source == "RESERVE":
        proposed, proposal_diagnostic = propose_reserve_gap_tasks(
            scene, headland, completed, protected_candidates)
        construction = "ACTUAL_RESERVE_GAP_EDGE"
    elif candidate_source == "OUTER":
        proposed = propose_outer_edge_tasks(
            scene, scene.target.difference(completed), completed)
        proposal_diagnostic = {}
        construction = "RESIDUAL_OUTER_EDGE"
    else:
        raise ValueError("UNKNOWN_RESIDUAL_CANDIDATE_SOURCE")
    stats = {"status": "NO_OUTER_EDGE_CANDIDATE",
             "candidate_count": len(proposed), "accepted_task_ids": [],
             "missing_before_m2": gap_before,
             "missing_after_m2": gap_before, "elapsed_s": 0.0,
             "candidate_source": candidate_source,
             "proposal_diagnostic": proposal_diagnostic}
    if not proposed:
        return completed, worked, pose, before, stats
    _checked_pass_order(body.routes, body.region_order)
    deadline = time.perf_counter() + min(6.0,
                                         settings.max_headland_seconds * 0.10)
    connector = Connector(scene, settings, deadline)
    validator = io.Validator(scene)
    variants = {task.task_id: io._task_variants(task, scene, validator)[0]
                for task in proposed}
    by_edge = {}
    for task in proposed:
        if variants[task.task_id]:
            by_edge.setdefault(task.row_index, []).append(task)
    groups = [sorted(group, key=lambda task: -task.suggested_order)
              for group in list(by_edge.values())[:6]]
    taskmap = {task.task_id: task for task in body.adapted_job.tasks}
    accepted = []
    trials = entries = reentries = 0

    def positions():
        passed_indices = [index for index, route in enumerate(body.routes)
                          if route.status == "REGION_ROUTE_PASS"]
        body_indices = [index for index in passed_indices
                        if body.routes[index].region_id != joint_HEADLAND_REGION_ID]
        if not body_indices:
            return []
        last_body_index = max(body_indices)
        done = GeometryCollection()
        by_region = {}
        at, prior = scene.start, None
        choices = []
        body_choices = []
        for order, index in enumerate(passed_indices):
            route = body.routes[index]
            done, by_region, at, prior = _advance(
                route, taskmap, done, by_region)
            following_index = (passed_indices[order + 1]
                               if order + 1 < len(passed_indices) else None)
            if following_index is None and scene.end is not None:
                continue
            next_pose = None
            if following_index is not None:
                following = body.routes[following_index]
                first_id = following.task_order[0]
                first_work = next(motion for motion in following.motions
                                  if motion.task_id == first_id)
                next_pose = io._pose_at(first_work, False)
            position = (index, following_index, done, dict(by_region),
                        at, prior, next_pose)
            (body_choices if index < last_body_index else choices).append(position)
        return choices, body_choices

    while len(accepted) < 4 and time.perf_counter() < connector.deadline:
        places, body_places = positions()
        selected = None
        for group in groups:
            if time.perf_counter() >= connector.deadline:
                break
            if group[0].row_index in {task.row_index for task in accepted}:
                continue
            all_options = [option for task in group
                           for option in variants[task.task_id]]
            ranked = sorted(places, key=lambda place: min(
                math.hypot(option.start.x - place[4].x,
                           option.start.y - place[4].y) +
                (math.hypot(option.end.x - place[6].x,
                            option.end.y - place[6].y)
                 if place[6] is not None else 0.0)
                for option in all_options))
            checked = ranked[:8]
            if (places and places[-1][1] is None
                    and not any(place is places[-1] for place in checked)):
                checked.append(places[-1])
            # The completed-body tail retains its established priority.  If
            # none of those positions can enter and rejoin this edge pass,
            # also try a few nearby body-block boundaries.  The latter are
            # accepted only after rebuilding the next incoming connection and
            # checking continuity in the dynamic already-worked space.
            checked.extend(sorted(body_places, key=lambda place: min(
                math.hypot(option.start.x - place[4].x,
                           option.start.y - place[4].y) +
                (math.hypot(option.end.x - place[6].x,
                            option.end.y - place[6].y)
                 if place[6] is not None else 0.0)
                for option in all_options))[:4])
            for index, following_index, done, by_region, at, prior, _ in checked:
                if time.perf_counter() >= connector.deadline:
                    break
                for task in group:
                    if (task.frozen_sweep.intersection(scene.target)
                            .difference(completed).area < 20.0):
                        continue
                    options = variants[task.task_id]
                    for direction in sorted({option.reversed for option in options}):
                        if time.perf_counter() >= connector.deadline:
                            break
                        trials += 1
                        segment = candidate_region(
                            body.adapted_job.regions[-1], [task],
                            construction, {task.task_id: options},
                            scene, connector, headland, done, by_region,
                            at, prior, direction)
                        if segment.status != "REGION_ROUTE_PASS":
                            continue
                        entries += 1
                        extended_taskmap = {**taskmap, task.task_id: task}
                        tentative = _advance(
                            segment, extended_taskmap, done, by_region)
                        reconnect = (_replace_incoming(
                            body.routes[following_index], extended_taskmap,
                            scene, connector, headland, *tentative)
                            if following_index is not None else None)
                        if following_index is not None and reconnect is None:
                            continue
                        reentries += 1
                        trial_routes = list(body.routes)
                        trial_routes.insert(index + 1, segment)
                        if reconnect is not None:
                            trial_routes[following_index + 1] = reconnect
                        motions = [motion for route in trial_routes
                                   if route.status == "REGION_ROUTE_PASS"
                                   for motion in route.motions]
                        if io._route_continuity_issues(motions, scene):
                            continue
                        actual_gain = (segment.completed_sweep
                                       .intersection(scene.target)
                                       .difference(completed).area)
                        if actual_gain < 20.0:
                            continue
                        selected = task, trial_routes, index
                        break
                    if selected:
                        break
                if selected:
                    break
            if selected:
                break
        if selected is None:
            break
        task, trial_routes, insertion_index = selected
        order_index = sum(route.status == "REGION_ROUTE_PASS"
                          for route in body.routes[:insertion_index + 1])
        body.routes = trial_routes
        body.region_order.insert(order_index, joint_HEADLAND_REGION_ID)
        _checked_pass_order(body.routes, body.region_order)
        taskmap[task.task_id] = task
        accepted.append(task)
        finished.append(task.task_id)
        completed = GeometryCollection()
        worked = {}
        pose, before = scene.start, None
        for route in body.routes:
            if route.status == "REGION_ROUTE_PASS":
                completed, worked, pose, before = _advance(
                    route, taskmap, completed, worked)
    if accepted:
        _register_supplemental_tasks(body, headland, accepted,
                                     construction)
    stats.update({"status": "IMPROVED" if accepted else "NO_REJOINABLE_EDGE",
                  "accepted_task_ids": [task.task_id for task in accepted],
                  "missing_after_m2": scene.target.difference(completed).area,
                  "elapsed_s": time.perf_counter() - started,
                  "search_limited": time.perf_counter() >= connector.deadline,
                  "attempts": trials, "entry_feasible": entries,
                  "reentry_feasible": reentries,
                  "connector_counts": dict(connector.counts)})
    return completed, worked, pose, before, stats


def _splice_unfinished_headland(body, adapted, scene, headland, settings,
                                variants, remaining, finished, completed):
    """将未完成田头任务插入已核查行程中，入出衔接均需通过，不能只附加无法返回的任务。
    
    Insert a required headland pass into a checked itinerary at both ends.
    
    A task that cannot be appended after the body may still be reachable near
    an earlier work block.  Test only the nearest itinerary boundaries, then
    rebuild the next incoming F2C connection.  Nothing is committed unless
    the complete motion chain remains continuous and gains target coverage."""
    pass  # 已合并到本模块，直接使用下方的定义。

    started = time.perf_counter()
    connector = Connector(scene, settings,
                          started + min(12.0, settings.max_headland_seconds*0.2))
    taskmap = {task.task_id: task for task in adapted.tasks}
    attempts = entries = reentries = 0
    accepted = []
    worked = {}
    pose, before = scene.start, None
    for route in body.routes:
        if route.status == "REGION_ROUTE_PASS":
            worked[route.region_id] = unary_union([
                worked.get(route.region_id, GeometryCollection()),
                route.completed_sweep])
            pose = io._pose_at(route.motions[-1], True)
            before = taskmap[route.task_order[-1]]

    def attempt_places(task, options, places):
        nonlocal attempts, entries, reentries
        for index, following_index, done, worked, pose, before, _ in places:
            if time.perf_counter() >= connector.deadline:
                break
            for direction in sorted({option.reversed for option in options}):
                if time.perf_counter() >= connector.deadline:
                    break
                attempts += 1
                segment = candidate_region(
                    adapted.regions[-1], [task], 'SPLICED_HEADLAND',
                    {task.task_id: options}, scene, connector, headland,
                    done, worked, pose, before, direction)
                if segment.status != 'REGION_ROUTE_PASS':
                    continue
                actual_gain = (segment.completed_sweep.intersection(scene.target)
                               .difference(completed).area)
                if actual_gain <= scene.settings.coverage_tolerance_m2:
                    continue
                entries += 1
                tentative = _advance(segment, taskmap, done, worked)
                reconnect = (_replace_incoming(
                    body.routes[following_index], taskmap, scene,
                    connector, headland, *tentative)
                    if following_index is not None else None)
                if following_index is not None and reconnect is None:
                    continue
                reentries += 1
                trial = list(body.routes)
                trial.insert(index + 1, segment)
                if reconnect is not None:
                    trial[following_index + 1] = reconnect
                motions = [motion for route in trial
                           if route.status == 'REGION_ROUTE_PASS'
                           for motion in route.motions]
                if io._route_continuity_issues(motions, scene):
                    continue
                return task, trial, index
        return None

    def positions():
        passed = [i for i, route in enumerate(body.routes)
                  if route.status == 'REGION_ROUTE_PASS']
        done = GeometryCollection()
        worked = {}
        pose, before = scene.start, None
        choices = []
        for order in range(len(passed) + 1):
            previous_index = passed[order - 1] if order else -1
            following_index = passed[order] if order < len(passed) else None
            next_pose = None
            if following_index is not None:
                following = body.routes[following_index]
                first_id = following.task_order[0]
                first_work = next(motion for motion in following.motions
                                  if motion.task_id == first_id)
                next_pose = io._pose_at(first_work, False)
            choices.append((previous_index, following_index, done,
                            dict(worked), pose, before, next_pose))
            if following_index is not None:
                done, worked, pose, before = _advance(
                    body.routes[following_index], taskmap, done, worked)
        return choices

    # A fixed four-task limit stopped simple, audited insertions even when the
    # search had ample time left.  Bound effort by the deadline and the small
    # neighborhood explored per task; each accepted pass must add real sweep.
    while remaining and time.perf_counter() < connector.deadline:
        choices = positions()
        selected = None
        ranked_tasks = sorted(remaining, key=lambda task: -(
            task.frozen_sweep.intersection(scene.target)
            .difference(completed).area))[:12]
        for task in ranked_tasks:
            if time.perf_counter() >= connector.deadline:
                break
            options = variants.get(task.task_id, ())
            if not options:
                continue
            gain = (task.frozen_sweep.intersection(scene.target)
                    .difference(completed).area)
            if gain <= scene.settings.coverage_tolerance_m2:
                continue
            ranked_places = _rank_splice_places(
                task, options, choices, body.routes, taskmap,
                last_tasks=False)
            selected = attempt_places(task, options, ranked_places)
            if selected is not None:
                break
        if selected is None and len(remaining) <= 2:
            # The original nearest-endpoint search keeps its full budget.
            # Only when it fails, spend a small extra budget on adjacent
            # parallel passes whose opposite ends are far apart.
            connector.deadline = min(
                started + min(18.0, settings.max_headland_seconds * 0.3),
                max(connector.deadline, time.perf_counter() + min(
                    6.0, settings.max_headland_seconds * 0.1)))
            for task in ranked_tasks:
                if time.perf_counter() >= connector.deadline:
                    break
                options = variants.get(task.task_id, ())
                if not options:
                    continue
                primary = _rank_splice_places(
                    task, options, choices, body.routes, taskmap,
                    last_tasks=False)
                primary_indices = {place[0] for place in primary}
                alternate = [place for place in _rank_splice_places(
                    task, options, choices, body.routes, taskmap,
                    last_tasks=True) if place[0] not in primary_indices]
                selected = attempt_places(task, options, alternate)
                if selected is not None:
                    break
        if selected is None:
            break
        task, trial, index = selected
        order_index = sum(route.status == 'REGION_ROUTE_PASS'
                          for route in body.routes[:index + 1])
        body.routes = trial
        body.region_order.insert(order_index, joint_HEADLAND_REGION_ID)
        remaining.remove(task)
        finished.append(task.task_id)
        accepted.append(task.task_id)
        completed = GeometryCollection()
        worked = {}
        pose, before = scene.start, None
        for route in body.routes:
            if route.status == 'REGION_ROUTE_PASS':
                completed, worked, pose, before = _advance(
                    route, taskmap, completed, worked)
    stats = {'status': 'IMPROVED' if accepted else 'NO_VALID_SPLICE',
             'accepted_task_ids': accepted, 'attempts': attempts,
             'entry_feasible': entries, 'reentry_feasible': reentries,
             'elapsed_s': time.perf_counter()-started,
             'search_limited': time.perf_counter() >= connector.deadline,
             'connector_counts': dict(connector.counts)}
    return completed, worked, pose, before, stats


def _append_last_headland_via_anchor(body, adapted, scene, headland, settings,
                                     variants, remaining, finished, completed,
                                     worked, pose, before):
    """借助已完成真实直线作业的锚点恢复末条田头通行，不重复宣称锚点作业。
    
    Recover absorbed straight access poses without repeating their work."""
    pass  # 已合并到本模块，直接使用下方的定义。
    pass  # 已合并到本模块，直接使用下方的定义。
    started=time.perf_counter()
    connector=Connector(scene,settings,started+min(
        8.0,settings.max_headland_seconds*.15))
    records=body.preparation['headland_work'].get('headland_contour_chains',[])
    records=[*records,{'source_straight_tasks':body.preparation['headland_work'].get(
        'optional_transit_straight_tasks',[])}]
    anchors=worked_anchors(adapted,records,completed,scene)
    accepted=[]
    taskmap={task.task_id:task for task in adapted.tasks}
    for task in list(remaining):
        if time.perf_counter()>=connector.deadline:
            break
        ready=io._ready_area(scene,headland,completed)
        for variant in variants.get(task.task_id,()):
            link,anchor=connect_via_worked_anchor(connector,pose,variant.start,
                before,task,anchors,ready,worked)
            if link is None:
                continue
            route=io.RegionRoute(joint_HEADLAND_REGION_ID,'REGION_ROUTE_PASS',
                order_mode='WORKED_ACCESS_HEADLAND',
                motions=[link.motion,variant.motion],connections=[link],
                task_order=[task.task_id],completed_sweep=variant.sweep,
                connection_seconds=link.seconds,reverse_m=link.reverse_m,
                gear_shifts=link.gear_shifts,
                sweep_error_m2=variant.sweep_error_m2)
            trial=[motion for prior in body.routes
                   if prior.status=='REGION_ROUTE_PASS' for motion in prior.motions]
            if io._route_continuity_issues([*trial,*route.motions],scene):
                continue
            body.routes.append(route);body.region_order.append(joint_HEADLAND_REGION_ID)
            remaining.remove(task);finished.append(task.task_id)
            completed,worked,pose,before=_advance(route,taskmap,completed,worked)
            accepted.append({'task_id':task.task_id,
                             'worked_access_source_task_id':anchor,
                             'method':link.method,'connection_seconds':link.seconds})
            break
    return completed,worked,pose,before,{
        'status':'IMPROVED' if accepted else 'NO_VALID_WORKED_ACCESS',
        'accepted':accepted,'anchor_count':len(anchors),
        'elapsed_s':time.perf_counter()-started,
        'search_limited':time.perf_counter()>=connector.deadline,
        'connector_counts':dict(connector.counts)}


# 以有限候选和预算联合组织主体与田头任务；不枚举全部排列。
# 新增任务和实际执行扫掠必须进入同一面积账本，失败状态继续保留。
def plan_joint(job, settings):
    """以有效主体为基础有界组织田头与补作，并保留主体、田头、通行与失败的分项证据。
    
    Extend a valid body chain with audited headland work candidates."""
    pass  # 已合并到本模块，直接使用下方的定义。

    started = time.perf_counter()
    body = plan_field(job, replace(settings, plan_headland_work=False))
    if body.status != "FIELD_ROUTE_COMPLETE":
        body.preparation["headland_work_status"] = "NOT_STARTED_BODY_INCOMPLETE"
        body.statistics["joint_stage"] = "BODY_INCOMPLETE"
        return body
    scene = io.load_scene(job.scene_path)
    if body.adapted_job is None or body.required_body is None:
        body.status = "JOINT_INPUT_INCOMPLETE"
        body.failures.append({"field_id":job.field_id,"region_id":"__HEADLAND__",
                              "code":"BODY_ADAPTED_INPUT_MISSING"})
        return body
    headland = scene.target.difference(body.required_body)
    deadline = time.perf_counter()+settings.max_headland_seconds
    connector = Connector(scene, settings, deadline)
    try:
        adapted, diagnostic = joint_generate(body.adapted_job, scene, headland,
                                       connector.backend, body.preparation,
                                       settings.headland_simplification_tolerance_m,
                                       settings.headland_positioning_allowance_m,
                                       contour_connector=(connector if
                                           settings.continuous_headland_contours else None),
                                       build_reverse_contours=settings.bidirectional_headland_contours)
    except (ValueError, RuntimeError, IndexError) as exc:
        body.status = "JOINT_HEADLAND_GENERATION_FAILED"
        body.failures.append({"field_id":job.field_id,
                              "region_id":joint_HEADLAND_REGION_ID,
                              "code":"HEADLAND_GENERATION_FAILED",
                              "detail":str(exc)})
        body.elapsed_s = time.perf_counter()-started
        return body
    executed = [route for route in body.routes if route.status=="REGION_ROUTE_PASS"]
    completed = (unary_union([route.completed_sweep for route in executed])
                 if executed else GeometryCollection())
    adapted, diagnostic = _omit_headland_work_already_done(
        adapted, diagnostic, scene, headland, completed)
    if settings.bidirectional_headland_contours:
        pass  # 已合并到本模块，直接使用下方的定义。
        adapted,choice_diagnostic=protect_coverage(adapted,scene,completed)
        diagnostic['contour_direction_choices']=choice_diagnostic
    body.adapted_job = adapted
    body.preparation["headland_work"] = diagnostic
    body.preparation["task_mapping"].extend(diagnostic["headland_task_mapping"])
    # Greedy headland planning appends routes and may detach a prescribed
    # field exit. Preserve the audited body chain for a later interleaved
    # attempt; that candidate must never inherit partial greedy work.
    audited_body_routes = copy.deepcopy(body.routes)
    taskmap = {task.task_id:task for task in adapted.tasks}
    previous = io._pose_at(executed[-1].motions[-1],True) if executed else scene.start
    previous_task = taskmap[executed[-1].task_order[-1]] if executed else None
    # The body solver may have appended the prescribed field exit.  Postpone
    # that leg until after all headland work, otherwise the exported chain
    # would visit the final exit in the middle of its itinerary.
    if scene.end is not None and executed:
        last = executed[-1]
        if (last.connections and last.connections[-1].to_task is None
                and last.connections[-1].motion is not None
                and last.motions[-1] is last.connections[-1].motion):
            final = last.connections.pop()
            last.motions.pop()
            last.connection_seconds -= final.seconds
            previous = io._pose_at(last.motions[-1],True)
    worked_by_region = {route.region_id:route.completed_sweep for route in executed}
    remaining = [task for task in adapted.tasks
                 if task.region_id==joint_HEADLAND_REGION_ID]
    validator = io.Validator(scene)
    variants = {}
    invalid = {}
    for task in remaining:
        variants[task.task_id],errors = io._task_variants(task,scene,validator)
        if not variants[task.task_id]:
            invalid[task.task_id] = errors
    headland_region = adapted.regions[-1]
    attempts = 0
    finished = []
    initial_completed = completed
    initial_previous = previous
    initial_previous_task = previous_task
    initial_worked_by_region = dict(worked_by_region)
    body_route_count = len(body.routes)
    body_order_count = len(body.region_order)
    while remaining and time.perf_counter()<deadline:
        ranked = _rank_tasks(remaining,variants,previous,
                             mode=settings.headland_entry_rank)
        chosen = None
        for task in ranked:
            if time.perf_counter()>=deadline:
                break
            if task.task_id in invalid:
                continue
            # A closed hole loop offers four distinct entry poses.  The
            # candidate connector evaluates all of them in one call.
            directions = ([None] if settings.headland_compare_entry_directions else
                          sorted({v.reversed for v in variants[task.task_id]}))
            for direction in directions:
                attempt = candidate_region(headland_region,[task],
                    "JOINT_HEADLAND_TASK",variants,scene,connector,headland,
                    completed,worked_by_region,previous,previous_task,direction)
                attempts += 1
                if attempt.status=="REGION_ROUTE_PASS":
                    chosen = (task,attempt)
                    break
            if chosen:
                break
        if chosen is None:
            break
        task,route = chosen
        body.routes.append(route)
        body.region_order.append(joint_HEADLAND_REGION_ID)
        finished.append(task.task_id)
        completed = unary_union([completed,route.completed_sweep])
        worked_by_region[joint_HEADLAND_REGION_ID] = completed
        previous = io._pose_at(route.motions[-1],True)
        previous_task = task
        remaining.remove(task)
    # A missing corner pass may need an absorbed straight pose, not a restart
    # of the entire headland order. Try this bounded local support before the
    # expensive branch/replay stages; all actual work and resource checks are
    # identical to the late fallback.
    access_statistics={'status':'NOT_NEEDED'}
    if (remaining and len(remaining)<=2 and not invalid and
            settings.continuous_headland_contours):
        completed,worked_by_region,previous,previous_task,access_statistics=(
            _append_last_headland_via_anchor(body,adapted,scene,headland,
                settings,variants,remaining,finished,completed,
                worked_by_region,previous,previous_task))
        access_statistics['stage']='AFTER_GREEDY'
    branch_statistics = {}
    if remaining and time.perf_counter()<deadline and not invalid:
        (alternate,branch_statistics) = _bounded_headland_search(
            [task for task in adapted.tasks if task.region_id==joint_HEADLAND_REGION_ID],
            variants,headland_region,scene,connector,headland,
            initial_completed,initial_previous,initial_previous_task,
            initial_worked_by_region,settings.max_headland_branch_nodes,
            priority={task.task_id for task in remaining})
        routes2,completed2,previous2,before2,remaining2 = alternate
        if (len(routes2),completed2.intersection(scene.target).area) > (
                len(finished),completed.intersection(scene.target).area):
            body.routes = body.routes[:body_route_count]+routes2
            body.region_order = body.region_order[:body_order_count]+[
                joint_HEADLAND_REGION_ID]*len(routes2)
            finished = [task_id for route in routes2 for task_id in route.task_order]
            completed,previous,previous_task = completed2,previous2,before2
            remaining = remaining2
            worked_by_region = dict(initial_worked_by_region)
            worked_by_region[joint_HEADLAND_REGION_ID] = completed
    interleaved_statistics = {"status":"NOT_NEEDED"}
    if remaining and not invalid:
        seed = copy.copy(body)
        seed.routes = audited_body_routes
        try:
            alternative, interleaved_statistics = _interleaved_attempt(
                seed, adapted, scene, headland, settings)
        except (ValueError, RuntimeError, IndexError) as exc:
            alternative = None
            interleaved_statistics = {
                "status":"REJECTED", "reason":type(exc).__name__+":"+str(exc)}
        if alternative is not None:
            (routes2, sequence2, finished2, remaining2, completed2,
             worked2, previous2, before2) = alternative
            current_score = (completed.intersection(scene.target).area,
                             len(finished))
            alternative_score = (completed2.intersection(scene.target).area,
                                 len(finished2))
            if alternative_score > current_score:
                replaced_count = len(finished)
                body.routes = routes2
                body.region_order = sequence2
                finished, remaining = finished2, remaining2
                completed, worked_by_region = completed2, worked2
                previous, previous_task = previous2, before2
                interleaved_statistics["status"] = "SELECTED"
                interleaved_statistics["replaced_completed_headland_tasks"] = replaced_count
            else:
                interleaved_statistics["status"] = "NOT_BETTER"
    splice_statistics = {'status': 'NOT_NEEDED'}
    if remaining and not invalid:
        completed, worked_by_region, previous, previous_task, splice_statistics = (
            _splice_unfinished_headland(
                body, adapted, scene, headland, settings, variants,
                remaining, finished, completed))
    if (remaining and len(remaining)<=2 and not invalid and
            settings.continuous_headland_contours):
        early_access=access_statistics
        completed,worked_by_region,previous,previous_task,access_statistics=(
            _append_last_headland_via_anchor(body,adapted,scene,headland,
                settings,variants,remaining,finished,completed,
                worked_by_region,previous,previous_task))
        access_statistics['early_attempt']=early_access
        access_statistics['stage']='AFTER_SPLICE'
    residual_bay_statistics = {"status": "DEFERRED_UNFINISHED_HEADLAND_TASKS"}
    if not remaining and not invalid:
        completed, worked_by_region, previous, previous_task, residual_bay_statistics = (
            _append_residual_bays(body, scene, headland, settings, completed,
                                  worked_by_region, previous, previous_task,
                                  finished))
    outer_edge_statistics = {"status": "DEFERRED_UNFINISHED_HEADLAND_TASKS"}
    if not remaining and not invalid:
        completed, worked_by_region, previous, previous_task, outer_edge_statistics = (
            _insert_residual_outer_edges(
                body, scene, headland, settings, completed,
                worked_by_region, previous, previous_task, finished))
    reserve_gap_statistics = {"status": "DEFERRED_UNFINISHED_HEADLAND_TASKS"}
    if (not remaining or settings.partial_reserve_gap_work) and not invalid:
        unexecuted_sweeps = unary_union([task.frozen_sweep
                                         for task in remaining])
        completed, worked_by_region, previous, previous_task, reserve_gap_statistics = (
            _insert_residual_outer_edges(
                body, scene, headland, settings, completed,
                worked_by_region, previous, previous_task, finished,
                candidate_source="RESERVE",
                protected_candidates=unexecuted_sweeps))
    exit_failed = False
    if not remaining and scene.end is not None and previous is not None:
        final = connector.connect(previous,scene.end,previous_task,None,
            io._ready_area(scene,headland,completed),worked_by_region,
            transfer=False)
        if final is None:
            exit_failed = True
        elif final.motion is not None:
            recipient = body.routes[-1] if finished else executed[-1]
            recipient.motions.append(final.motion)
            recipient.connections.append(final)
            recipient.connection_seconds += final.seconds
    if remaining:
        reason = ("HEADLAND_SEARCH_BUDGET_EXHAUSTED"
                  if time.perf_counter()>=deadline else
                  "HEADLAND_CONNECTION_UNFOUND")
        body.routes.append(io.RegionRoute(joint_HEADLAND_REGION_ID,
            "REGION_ROUTE_NOT_FOUND",reason=reason))
        body.failures.append({"field_id":job.field_id,
                              "region_id":joint_HEADLAND_REGION_ID,
                              "code":reason,
                              "remaining_task_count":len(remaining),
                              "remaining_task_ids":",".join(t.task_id for t in remaining)})
    if exit_failed:
        body.failures.append({"field_id":job.field_id,
                              "region_id":joint_HEADLAND_REGION_ID,
                              "code":"FIELD_END_UNREACHABLE_AFTER_HEADLAND"})
    target_missing = scene.target.difference(completed).area
    body.preparation["headland_work_status"] = (
        "ALL_GENERATED_TASKS_CONNECTED" if not remaining and not exit_failed
        else "PARTIAL_TASK_CHAIN")
    body.preparation["original_target_missing_m2"] = target_missing
    # This only erodes travel by the scalar safety margin. It does not apply
    # the full oriented body shape, and the tool can cover part of the area.
    # Consequently it is a diagnostic, not an unreachable-work lower bound.
    body.preparation["reference_margin_exclusion_area_m2"] = (
        scene.target.difference(
            scene.travel.buffer(-scene.vehicle.safety_margin_m)).area)
    if remaining or exit_failed:
        body.status = "JOINT_TASKS_PARTIAL"
    elif target_missing>scene.settings.coverage_tolerance_m2:
        body.status = "JOINT_TASK_CHAIN_TARGET_GAP"
        body.failures.append({"field_id":job.field_id,
                              "region_id":"__TARGET__",
                              "code":"ORIGINAL_TARGET_UNCOVERED",
                              "area_m2":target_missing})
    else:
        body.status = "FULL_FIELD_ROUTE_COMPLETE"
    body.transfer_status = ("ALL_REQUIRED_TRANSFERS_CONNECTED"
                            if not remaining and not exit_failed else
                            "PARTIAL_OR_NOT_REACHED")
    body.statistics["headland_stage"] = {
        "search_attempts":attempts,
        **branch_statistics,
        "interleaved":interleaved_statistics,
        "splice":splice_statistics,
        "worked_access":access_statistics,
        "residual_bays":residual_bay_statistics,
        "residual_outer_edges":outer_edge_statistics,
        "actual_reserve_gap":reserve_gap_statistics,
        "completed_task_count":len(finished),
        "remaining_task_count":len(remaining),
        "elapsed_s":time.perf_counter()-started-body.elapsed_s,
        "connector_counts":dict(connector.counts)}
    body.elapsed_s = time.perf_counter()-started
    if settings.bidirectional_headland_contours:
        pass  # 已合并到本模块，直接使用下方的定义。
        body.preparation['headland_work']['selected_contour_variants']=(
            bind_selected_sweeps(body,scene))
    return body

# ==========================================================================
# 8. 田头剩余目标的补作候选
# 从实际面积缺口提出直线候选，检查车身和机具包络后才保留。
# ==========================================================================

from collections import Counter
import math
import time

import numpy as np
import shapely
from shapely import wkt
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.ops import unary_union

import route_planner as io
_headland_fill_lines = _prepare_lines


def headland_fill_polygons(g):
    """递归枚举正面积残余面，微小空对象不作为补作田块。"""
    if g.geom_type == 'Polygon':
        if g.area > 1e-10:
            yield g
    elif hasattr(g, 'geoms'):
        for child in g.geoms:
            yield from headland_fill_polygons(child)


def rectangle_inset(g, half_x, half_y):
    """固定轴向矩形的精确面域退让；凹边及孔洞也参与边界扫掠。

    对每条边计算其与中心对称矩形的Minkowski和。参考点必须在原面域
    内，并且不落入任何边界扫掠带。不能用四个角都在田内代替此检查，
    因为矩形的边或内部仍可能覆盖小孔洞。
    """
    edges = []
    for polygon in headland_fill_polygons(g):
        for ring in (polygon.exterior, *polygon.interiors):
            xy = np.asarray(ring.coords)[:, :2]
            edges.append(np.stack((xy[:-1], xy[1:]), axis=1))
    if not edges:
        return GeometryCollection()
    rect = np.array([[-half_x,-half_y], [half_x,-half_y],
                     [half_x,half_y], [-half_x,half_y]])
    points = (np.concatenate(edges)[:, :, None, :] + rect[None, None, :, :]).reshape(-1,8,2)
    bands = shapely.convex_hull(shapely.multipoints(points))
    return g.difference(shapely.union_all(bands))


def reference_space(scene, angle):
    """返回朝向angle下的参考点空间，车体在travel、机具在真实target。"""
    body, tool = scene.vehicle.rectangles()
    safe_travel = scene.travel.buffer(-scene.vehicle.safety_margin_m/math.cos(math.pi/32))
    def inset(geometry, rect):
        aligned = rotate(geometry, -angle, origin=(0,0), use_radians=True)
        lo, hi = rect.min(axis=0), rect.max(axis=0)
        center = (lo+hi)/2
        half = (hi-lo)/2
        eroded = rectangle_inset(aligned, float(half[0]), float(half[1]))
        eroded = translate(eroded, xoff=-float(center[0]), yoff=-float(center[1]))
        return rotate(eroded, angle, origin=(0,0), use_radians=True)
    # Small inward numeric guard prevents a tangent reference edge from
    # becoming a collision through coordinate rotation roundoff.
    return inset(safe_travel, body).intersection(inset(scene.target, tool)).buffer(-1e-5)


def candidate_angles(scene, headings, maximum=8):
    """区域排线方向优先，其次长真实边界方向；固定小集合，不枚举全角度。"""
    ranked = [(1e12-i, float(a)%math.pi) for i,a in enumerate(headings)]
    for p in headland_fill_polygons(scene.target):
        for ring in (p.exterior,*p.interiors):
            for a,b in zip(ring.coords,list(ring.coords)[1:]):
                dx,dy = b[0]-a[0],b[1]-a[1]
                if math.hypot(dx,dy)>scene.vehicle.working_width_m:
                    ranked.append((math.hypot(dx,dy),math.atan2(dy,dx)%math.pi))
    result=[]
    for _,a in sorted(ranked,reverse=True):
        if all(abs((a-b+math.pi/2)%math.pi-math.pi/2)>math.radians(3) for b in result):
            result.append(a)
        if len(result)>=maximum:
            break
    return result or [0.0]


# 根据实际覆盖差集提出田头补作直线，限制方向族和无效短段。
# 这些线只认证局部作业几何，函数不会额外证明进出路线或机具滞后。
def fill_gaps(field_id, scene, covered, headings=(), *, seconds=20.0, max_segments=256):
    """生成真实独立补漏任务和审计记录。搜索耗尽不会把缺口标成不可达。"""
    if isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('HEADLAND_FILL_SECONDS_MUST_BE_POSITIVE_FINITE')
    if type(max_segments) is not int or max_segments<1:
        raise ValueError('HEADLAND_FILL_MAX_SEGMENTS_MUST_BE_POSITIVE_INTEGER')
    started=time.perf_counter();deadline=started+seconds
    initial=scene.target.difference(covered)
    remaining=initial;tasks=[];rejected=Counter();checked=0;seen=set();spaces={}
    validator=io.Validator(scene);backend=io.F2CBackend(scene)
    width=scene.vehicle.working_width_m
    spacing=width*(1-scene.settings.overlap_fraction)
    minimum_novel_area=width*1.0
    minimum_contribution_fraction=.55
    reach=width/2+scene.vehicle.implement_length_m/2
    diagonal=math.hypot(scene.target.bounds[2]-scene.target.bounds[0],
                        scene.target.bounds[3]-scene.target.bounds[1])+width*2
    angles=candidate_angles(scene,headings)
    limited=False;selected_angles=set()
    for round_index in range(2):
        pool=[]
        for angle in angles:
            if time.perf_counter()>=deadline:
                limited=True;break
            if angle not in spaces:
                spaces[angle]=reference_space(scene,angle)
            safe=spaces[angle]
            if safe.is_empty:
                continue
            u=np.array([math.cos(angle),math.sin(angle)])
            # A tool offset changes which reference point can cover a gap.
            wanted=translate(remaining.buffer(reach),
                xoff=-scene.vehicle.implement_offset_m*u[0],
                yoff=-scene.vehicle.implement_offset_m*u[1]).intersection(safe)
            guides=[]
            if round_index==0:
                # Native F2C rows clipped to the allowed reference geometry.
                for polygon in headland_fill_polygons(wanted):
                    # Independent segments need no native snake ordering.
                    # The F2C RP_Snake wrapper can crash on one-row cells.
                    cell=backend.f2c.Cell();cell.importFromWkt(polygon.wkt)
                    native=backend.swath_generator.generateSwaths(angle,spacing,cell)
                    for i in range(native.size()):
                        guides.append(wkt.loads(native[i].getPath().exportToWkt()))
            else:
                # Thin edge remnants can sit between native grid phases.
                # Use one local line per component plus feasible boundary
                # supports, rather than a second full-field angular search.
                for p in headland_fill_polygons(remaining):
                    if p.area>scene.settings.coverage_tolerance_m2:
                        c=np.asarray(p.representative_point().coords[0])-scene.vehicle.implement_offset_m*u
                        guides.append(LineString([c-diagonal*u,c+diagonal*u]))
                for p in headland_fill_polygons(wanted):
                    aligned=rotate(p,-angle,origin=(0,0),use_radians=True)
                    for y in (aligned.bounds[1]+1e-5,aligned.bounds[3]-1e-5):
                        line=LineString([(aligned.bounds[0]-width,y),(aligned.bounds[2]+width,y)])
                        guides.append(rotate(line,angle,origin=(0,0),use_radians=True))
            for guide in guides:
                if time.perf_counter()>=deadline:
                    limited=True;break
                c=np.asarray(guide.centroid.coords[0])
                long=LineString([c-diagonal*u,c+diagonal*u])
                for piece in _headland_fill_lines(long.intersection(wanted)):
                    a,b=np.asarray(piece.coords[0]),np.asarray(piece.coords[-1])
                    if (b-a)@u<0:a,b=b,a
                    if np.linalg.norm(b-a)<1.0:
                        rejected['LESS_THAN_1M']+=1;continue
                    key=(round(angle,7),*np.round(a,5),*np.round(b,5))
                    if key in seen:continue
                    seen.add(key)
                    line=LineString([a,b]);tid=f'{field_id}_headland_fill_{checked:05d}'
                    task=io.FrozenTask(field_id,'__HEADLAND__',tid,2,checked,line,
                        angle,GeometryCollection(),line.length,work_kind='HEADLAND_GAP_FILL')
                    checked+=1
                    work=io._work_motion(task,scene,False)
                    sweep=validator.work_sweep(work)
                    gain=sweep.intersection(remaining).area
                    if gain<max(minimum_novel_area,minimum_contribution_fraction*width*line.length):
                        rejected['LOW_EFFECTIVE_WORK_CONTRIBUTION']+=1;continue
                    issues=validator.motion_issues(work)
                    if issues:
                        rejected[issues[0]['code']]+=1;continue
                    task=io.FrozenTask(field_id,'__HEADLAND__',tid,2,checked,line,
                        angle,sweep,line.length,work_kind='HEADLAND_GAP_FILL',motion_points=work.points.copy())
                    pool.append((gain,task))
        # Prefer large actual area contributions. Recheck novelty after each
        # accepted row, so the same gap cannot justify several duplicate rows.
        for _,task in sorted(pool,key=lambda item:(-item[0],item[1].length_m,item[1].task_id)):
            if task.heading_rad not in selected_angles and len(selected_angles)>=3:
                continue
            gain=remaining.intersection(task.frozen_sweep).area
            if gain<max(minimum_novel_area,minimum_contribution_fraction*width*task.length_m):
                continue
            tasks.append(task);selected_angles.add(task.heading_rad)
            remaining=remaining.difference(task.frozen_sweep)
            if len(tasks)>=max_segments:
                limited=True;break
        if remaining.area<=scene.settings.coverage_tolerance_m2 or limited:break
    total=unary_union([covered,*[t.frozen_sweep for t in tasks]])
    final=scene.target.difference(total)
    diagnostic={'mode':'THREE_BASE_CONTOURS_PLUS_INDEPENDENT_GAP_SEGMENTS',
        'initial_target_missing_m2':initial.area,'final_target_missing_m2':final.area,
        'recovered_m2':initial.area-final.area,'gap_segment_count':len(tasks),
        'gap_work_length_m':sum(t.length_m for t in tasks),'candidate_count':checked,
        'rejected_candidates':dict(rejected),'angles_rad':angles,
        'minimum_novel_area_m2':minimum_novel_area,
        'minimum_contribution_fraction':minimum_contribution_fraction,
        'selected_direction_families_rad':sorted(selected_angles),
        'maximum_direction_families':3,
        'elapsed_seconds':time.perf_counter()-started,'seconds_budget':seconds,
        'search_status':'SEARCH_LIMIT' if limited else 'BOUNDED_CANDIDATES_EXHAUSTED',
        'coverage_status':'COVERAGE_COMPLETE' if final.area<=scene.settings.coverage_tolerance_m2 else 'COVERAGE_GAP',
        'entry_exit_status':'NOT_PLANNED','implement_lag_status':'PARAMETERS_UNVERIFIED',
        'residual_is_physical_impossibility':False,
        'pass_index_semantics':'Supplement uses export group 3; not a fourth contour or proof of three continuous loops'}
    return tuple(tasks),diagnostic

# ==========================================================================
# 9. 已作业区域中的中转锚点
# 为连接提供有限的可选中间位置；不把未作业区域当成已割空间。
# ==========================================================================

import math
import time
import numpy as np
from shapely import wkt
import route_planner as io
from scene import Motion as access_Motion


def worked_anchors(adapted, records, completed, scene):
    """仅从已完成并具有真实覆盖的直线任务提取通行锚点，不借未来条带。
    
    Only real straight source passes whose target footprint is worked."""
    candidates = {task.task_id:task for task in adapted.tasks
                  if task.motion_points is None}
    for record in records:
        for row in record.get('source_straight_tasks',[]):
            line=wkt.loads(row['reference_line_wkt'])
            candidates[row['task_id']]=io.FrozenTask(adapted.field_id,
                '__HEADLAND__',row['task_id'],row['row_index'],
                row['suggested_order'],line,row['heading_rad'],
                wkt.loads(row['sweep_wkt']),line.length,
                work_kind='STRAIGHT_HEADLAND')
    return [task for task in candidates.values()
            if task.length_m>=2*scene.vehicle.min_turn_radius_m and
            task.frozen_sweep.intersection(scene.target).difference(
                completed).area<=scene.settings.coverage_tolerance_m2]


def connect_via_worked_anchor(connector, start, goal, before, after, anchors,
                              ready, worked, max_entries=48):
    """尝试少量已作业锚点位姿，只有整条连接通过检查才返回运动。
    
    Bounded pose proposals; returned geometry is one fully checked link."""
    scene=connector.scene
    validator=io.Validator(scene)
    proposals=[]
    for task in anchors:
        variants,_=io._task_variants(task,scene,validator)
        for variant in variants:
            score=(math.hypot(variant.end.x-goal.x,variant.end.y-goal.y)+
                   .15*math.hypot(variant.start.x-start.x,
                                 variant.start.y-start.y))
            proposals.append((score,task.task_id,variant))
    proposals.sort(key=lambda row:row[:2])
    for _,task_id,variant in proposals[:max_entries]:
        if time.perf_counter()>=connector.deadline:
            break
        # The pose is a proposal only. Even a previously legal work pass may
        # be illegal with its implement closed under the transit rules.
        middle=access_Motion(variant.motion.points.copy(),'turn',implement_on=False)
        if connector.evaluate(middle,'WORKED_ANCHOR_STRAIGHT',None,None,
                              ready,worked) is None:
            continue
        tail=connector.connect(variant.end,goal,None,after,ready,worked,
                               _allow_portals=False,_allow_medial=False)
        if tail is None or time.perf_counter()>=connector.deadline:
            continue
        incoming=connector.connect(start,variant.start,before,None,ready,worked,
                                   _allow_portals=False,_allow_medial=False)
        if incoming is None:
            continue
        segments=([incoming.motion] if incoming.motion is not None else [])+[
            middle]+([tail.motion] if tail.motion is not None else [])
        points=np.vstack([m.points[:-1] for m in segments[:-1]]+
                         [segments[-1].points])
        link=connector.evaluate(access_Motion(points,'turn',implement_on=False),
            'F2C_WORKED_ACCESS:'+task_id,before,after,ready,worked)
        if link is not None:
            return link,task_id
    return None,None

# ==========================================================================
# 10. 规则路线的田头参考与端口准备
# 构造合法田头线和端部空间，为规则条带连接提供局部条件。
# ==========================================================================

import math
import time
import numpy as np
from shapely import wkt
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from scene import Motion as regular_headland_Motion
from scene import Pose as regular_headland_Pose
from scene import wrap as regular_headland_wrap
from validator import conservative_tool_coverage as regular_headland_conservative_tool_coverage
import route_planner as io


def corner_placement(scene,pass_count):
    """估算多圈外侧圆角所需预留，包括车头与后置机具的摆幅，不能只按直线半宽退让。
    
    Analytic corner placement for the outer parallel fillet.
    
    Straight clearance alone misses the front-body and rear-implement swing.
    For a radius R, each rotating corner lies on a circle about the fillet
    center. Its largest radial support gives a conservative placement; final
    complete envelopes still decide safety. This changes only headland width."""
    v=scene.vehicle;radius=v.min_turn_radius_m+(pass_count-1)*v.working_width_m
    body=math.hypot(radius+v.body_width_m/2,max(v.front_m,v.rear_m))-radius
    body+=v.safety_margin_m/math.cos(math.pi/32)+scene.settings.travel_clearance_m
    tool=math.hypot(radius+v.working_width_m/2,
        abs(v.implement_offset_m)+v.implement_length_m/2)-radius
    required=max(body,tool)+.02
    straight_depth=max(v.body_width_m/2+v.safety_margin_m+scene.settings.travel_clearance_m,
        v.working_width_m/2)+.02
    return max(0.,required-straight_depth)


def work_coverage(motion, scene):
    """仅为开启机具的运动计算覆盖；直线和曲线按各自适用几何处理，空驶不记作业。"""
    if not motion.implement_on:
        return GeometryCollection()
    if np.max(np.abs((motion.points[:,2]-motion.points[0,2]+math.pi)%(2*math.pi)-math.pi))<1e-8:
        return io.Validator(scene).work_sweep(motion)
    return regular_headland_conservative_tool_coverage(motion,scene)


def straight(a,b,task_id):
    """由局部米制端点构造前进直线Motion，朝向从端点差得到，机具状态由调用者指定。"""
    a,b=np.array(a,dtype=float),np.array(b,dtype=float)
    yaw=math.atan2(b[1]-a[1],b[0]-a[0])
    return regular_headland_Motion(np.array([[*a,yaw,1],[*b,yaw,1]]),
        'straight_work',task_id,True)


def _attempt(ring, scene, turns, component_id, factor,parallel_pass=None,pass_count=3):
    """对一个规则田头轮廓尝试运动模板并保存失败原因，轮廓存在不等于可转弯。"""
    vertices=np.array(ring.coords[:-1],dtype=float)
    if len(vertices)<3:
        return None,'NO_CONTOUR'
    edges=np.roll(vertices,-1,axis=0)-vertices
    lengths=np.linalg.norm(edges,axis=1)
    if np.any(lengths<1e-4):
        return None,'DEGENERATE_CONTOUR_EDGE'
    units=edges/lengths[:,None]
    angles=np.arctan2(units[:,1],units[:,0])
    radius=scene.vehicle.min_turn_radius_m
    ramp=1/(radius*scene.vehicle.max_curvature_rate)
    deltas=np.array([regular_headland_wrap(angles[i]-angles[i-1]) for i in range(len(vertices))])
    # Current F2C CC family needs longer support than a circular fillet.
    # A01 probes rejected 7--10m quarter-turn support as winding loops;
    # 12m support produced a monotone turn. Every proposal is still checked.
    corner_radii=(radius+np.where(deltas>=0,pass_count-parallel_pass,
        parallel_pass-1)*scene.vehicle.working_width_m if parallel_pass is not None else None)
    trims=(corner_radii*np.abs(np.tan(deltas/2)) if corner_radii is not None
        else factor*(radius*np.abs(np.tan(deltas/2))+2*ramp+.25))
    trims[np.abs(deltas)<(1e-6 if corner_radii is not None else .025)]=0
    if np.any(trims+np.roll(trims,-1)+1.0>lengths):
        return None,'INSUFFICIENT_CORNER_TANGENT_SUPPORT'
    incoming=vertices-trims[:,None]*np.roll(units,1,axis=0)
    outgoing=vertices+trims[:,None]*units
    motions=[]
    for i in range(len(vertices)):
        if time.perf_counter()>=turns.deadline:
            return None,'SEARCH_BUDGET_EXHAUSTED'
        j=(i+1)%len(vertices)
        row=straight(outgoing[i],incoming[j],f'{component_id}_edge{i:03d}')
        issues,_=turns.native.checker.physical(row)
        if issues:
            return None,'EDGE_'+issues[0]
        motions.append(row)
        a=regular_headland_Pose(*incoming[j],float(angles[i]));b=regular_headland_Pose(*outgoing[j],float(angles[j]))
        # A forward cutting corner needs no fictitious previously cut area.
        if corner_radii is not None:
            if io._pose_close(a,b,scene):
                corner=(None,{})
            else:
                corner=turns.working_circle(a,b,float(corner_radii[j]))
        else:
            corner=turns.connect(a,b,GeometryCollection(),work=True)
        if corner is None:
            return None,'NO_LOCAL_WORKING_CORNER'
        if corner[0] is not None:
            corner[0].task_id=f'{component_id}_corner{j:03d}'
            motions.append(corner[0])
    if io._route_continuity_issues(motions,scene):
        return None,'CONTOUR_DISCONTINUITY'
    if not io._pose_close(io._pose_at(motions[0],False),io._pose_at(motions[-1],True),scene):
        return None,'CONTOUR_NOT_CLOSED'
    return motions,None


def _smooth_contour(ring,scene,turns,cid,sigma_m):
    """用周期几何滤波及切向采样提出平滑田头模板，最终仍检查曲率及真实包络。
    
    Periodic Gaussian geometric template, with analytic tangent samples.
    
    F2C supplies the headland offset contour. A periodic filter rounds short
    runs without a combinatorial corner search. It can shrink a hole or move
    a concave corner outside: the *unchanged* physical checker rejects those
    cases. Smoothing is a proposal, never a license to alter the field."""
    line=LineString(ring.coords)
    count=max(64,math.ceil(line.length/scene.settings.sampling_step_m))
    if count>scene.settings.max_motion_samples:
        return None,'CONTOUR_SAMPLE_LIMIT'
    coordinates=np.array([line.interpolate(i/count,normalized=True).coords[0]
                          for i in range(count)])
    step=line.length/count
    frequencies=np.fft.fftfreq(count,d=step)*2*math.pi
    attenuation=np.exp(-.5*(frequencies*sigma_m)**2)
    coefficients=np.fft.fft(coordinates,axis=0)
    smoothed=np.fft.ifft(coefficients*attenuation[:,None],axis=0).real
    tangent=np.fft.ifft(coefficients*(attenuation*frequencies*1j)[:,None],axis=0).real
    yaw=np.arctan2(tangent[:,1],tangent[:,0])
    points=np.column_stack((smoothed,yaw,np.ones(count)))
    points=np.vstack((points,points[0]))
    motion=regular_headland_Motion(points,'work',cid+'_smooth',True)
    issues,_=turns.native.checker.physical(motion)
    if issues:
        return None,issues[0]
    return [motion],None


def generate_regular_headlands(scene, turns, pass_count=3, placement_m=0., *, analytic_only=False, mitre=False):
    """生成规则假设下的2或3圈田头运动，圈数非法拒绝，未通过候选不能算已作业。"""
    if type(pass_count) is not int or pass_count not in (2,3):
        raise ValueError('INVALID_HEADLAND_PASS_COUNT')
    v=scene.vehicle
    # F2C first contour is w/2 inside the true boundary. Add only placement
    # needed for body clearance; the target and original travel do not change.
    extra=max(0.,v.body_width_m/2+v.safety_margin_m+
              scene.settings.travel_clearance_m-v.working_width_m/2)+.02+placement_m
    native=turns.native.backend.f2c.HG_Const_gen().generateHeadlandSwaths(
        turns.native.backend.cells(scene.target),v.working_width_m,pass_count)
    components=[];failures=[]
    for pass_index,cells in enumerate(native,1):
        # The rounded native offset can contain tiny edges at a reflex vertex.
        # A mitre proposal at the same F2C pass depth exposes the two original
        # incident lines so their radius-constrained fillet can be constructed.
        # It changes the candidate route only; original target/travel remain
        # the physical checker's collision and coverage truth.
        candidate=(scene.target.buffer(-((pass_index-.5)*v.working_width_m+extra),
            join_style='mitre') if mitre else wkt.loads(cells.exportToWkt()).buffer(-extra))
        candidate=candidate.simplify(.30,preserve_topology=True)
        polygons=list(candidate.geoms) if hasattr(candidate,'geoms') else [candidate]
        if not polygons or candidate.is_empty:
            failures.append({'pass_index':pass_index,'reason':'EMPTY_INSET_CONTOUR'})
        for polygon_index,polygon in enumerate(polygons):
            if polygon.geom_type!='Polygon':
                continue
            polygon=orient(polygon,sign=1)
            for contour_index,ring in enumerate([polygon.exterior,*polygon.interiors]):
                cid=f'p{pass_index:02d}_poly{polygon_index:02d}_c{contour_index:02d}'
                # Equal-radius independent fillets leave corner gaps between
                # otherwise parallel passes. Convex radii shrink by one tool
                # width per inward pass; concave/hole radii grow. The vehicle
                # minimum is unchanged. Explicit parked steering occurs only
                # at the straight/arc boundaries, without a heading jump.
                selected,reason=_attempt(ring,scene,turns,cid,1.,
                    parallel_pass=pass_index,pass_count=pass_count)
                # Fixed two tangent-support options; no dense angle search.
                if selected is None and not analytic_only:
                    for factor in [1.,1.35]:
                        selected,reason=_attempt(ring,scene,turns,cid,factor)
                        if selected is not None:
                            break
                if selected is None and not analytic_only and time.perf_counter()<turns.deadline:
                    for sigma in [.4*scene.vehicle.min_turn_radius_m,
                                  2.*scene.vehicle.min_turn_radius_m]:
                        selected,reason=_smooth_contour(ring,scene,turns,cid,sigma)
                        if selected is not None:
                            break
                if selected is None:
                    failures.append({'component_id':cid,'pass_index':pass_index,
                        'reason':reason,'contour_wkt':LineString(ring.coords).wkt})
                    continue
                # Every admitted component carries its real stop/gear events,
                # including the geometric smoothing fallback. Missing events
                # must never be invented by the independent replay validator.
                for motion in selected:
                    motion.operation_events=io._gear_and_steering(motion,scene,io.RouteSettings())[3]
                sweep=unary_union([work_coverage(m,scene) for m in selected])
                components.append({'component_id':cid,'pass_index':pass_index,
                    'contour_index':contour_index,'closed':True,
                    'motions':selected,'sweep':sweep})
    worked=unary_union([c['sweep'] for c in components])
    return components,worked,failures


def prepare_headlands(scene,turns,pass_count=3):
    """尝试少量内移预留位置兼顾凸角摆幅与凹角余量，最终运动与目标覆盖必须复检。
    
    Joint staging-band proposal: at most three uniform inward placements.
    
    Convex swing determines the first placement. Concave swing can need more
    room, so try two tool-scaled increments with cheap exact fillets before
    asking a native shortest-path solver to repair individual corners. Every
    pass shares the increment; the returned placement also drives body ports.
    Uncut outer crop is still a target gap, never silently reassigned as cut."""
    base=corner_placement(scene,pass_count)
    trials=[]
    for placement in [base,base+.25*scene.vehicle.working_width_m,
                      base+.5*scene.vehicle.working_width_m]:
        if time.perf_counter()>=turns.deadline:
            break
        heads,worked,failures=generate_regular_headlands(scene,turns,pass_count,placement,
            analytic_only=True,mitre=True)
        trials.append((heads,worked,failures,placement))
        if not failures:
            return trials[-1]
    if not trials:
        return [],GeometryCollection(),[{'reason':'SEARCH_BUDGET_EXHAUSTED'}],base
    best=min(trials,key=lambda r:(len(r[2]),-len(r[0]),-r[1].area,r[3]))
    if time.perf_counter()<turns.deadline:
        fallback=(*generate_regular_headlands(scene,turns,pass_count,best[3]),best[3])
        best=min([best,fallback],key=lambda r:(len(r[2]),-len(r[0]),-r[1].area,r[3]))
    return best

# ==========================================================================
# 11. 统一局部调头模板
# 尝试有限前进、折返与倒车模板，检查模板空间和驾驶操作；不做全排列。
# ==========================================================================

from collections import Counter
import math
import time
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry import GeometryCollection
from scene import Motion as regular_turns_Motion
from scene import wrap as regular_turns_wrap
regular_turns_Connector = Connector
regular_turns_visibility_guides = visibility_guides
regular_turns_gentle_radii = gentle_radii
regular_turns_gentle_native = gentle_native
import route_planner as io


class SampleBoundedBackend(io.F2CBackend):
    """限制原生运动采样规模的后端，超限拒绝，不能偷偷粗化获得通过。"""
    def convert_path(self,path,link,kind):
        """转换原生路径前检查采样上限，再交给既有转换逻辑。"""
        if path.size()>self.scene.settings.max_motion_samples:
            raise ValueError('NATIVE_SAMPLE_LIMIT')
        return super().convert_path(path,link,kind)


def gear_statistics(points):
    """从明确运动挡位统计连续挡位段、换挡和倒车段，不按几何遍历方向推断真实倒挡。"""
    points = np.asarray(points)
    moving = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > 1e-7
    gears = points[:-1, 3][moving].astype(int)
    if not len(gears):
        return [], 0, 0
    runs = [int(gears[0])]
    for gear in gears[1:]:
        if gear != runs[-1]:
            runs.append(int(gear))
    # Work on both sides is forward: include the stationary boundary shifts.
    padded = [1, *runs, 1]
    shifts = sum(a != b for a, b in zip(padded, padded[1:]))
    return runs, shifts, runs.count(-1)


def template_class(points, net_heading):
    """按挡位段和换挡规模判定局部调头模板是否符合规则驾驶假设，超限返回None。"""
    runs, shifts, reverse = gear_statistics(points)
    if shifts > 4 or reverse > 2 or len(runs) > 5:
        return None
    dyaw = (np.diff(points[:, 2]) + math.pi) % (2*math.pi) - math.pi
    variation = float(np.abs(dyaw).sum())
    if not reverse:
        return 'LOCAL_U' if variation <= abs(net_heading) + 0.15 else None
    if variation > 2*math.pi + .15:
        return None
    return 'LOCAL_3LEG' if shifts <= 2 and len(runs) <= 3 else 'LOCAL_5LEG'


def transfer_class(points):
    """判定共享田头转场模板，限制过多换挡和完整绕圈，局部U形另用专门模板约束。
    
    A shared-headland transfer may have several bends, but no full loop.
    
    Local U-turns are still constrained by template_class. Transfers retain
    the same gear limits and all collision/chronological crop checks. A full
    heading winding is rejected rather than hidden as a block transition."""
    runs,shifts,reverse=gear_statistics(points)
    dyaw=(np.diff(points[:,2])+math.pi)%(2*math.pi)-math.pi
    if shifts>4 or reverse>2 or len(runs)>5 or np.abs(dyaw).sum()>2*math.pi-.1:
        return None
    return 'BLOCK_TRANSFER'


def port_window(a, b, scene):
    """构造由车辆尺度决定的有限端口窗口，排除远距离纵向跳跃候选。
    
    A priori finite oriented window; reject far longitudinal endpoint jumps."""
    v = scene.vehicle
    radius = v.min_turn_radius_m
    reach = 2*radius + max(v.front_m, v.rear_m,
        abs(v.implement_offset_m) + v.implement_length_m/2) + v.working_width_m
    u = np.array([math.cos(a.yaw), math.sin(a.yaw)])
    n = np.array([-u[1], u[0]])
    first = np.array([a.x, a.y]);last = np.array([b.x, b.y])
    if abs(float((last-first) @ u)) > reach:
        return GeometryCollection()
    # Skip-row proposals also have finite lateral support.
    if abs(float((last-first) @ n)) > 4*radius + 3*v.working_width_m:
        return GeometryCollection()
    x = [float(first@u), float(last@u)]
    y = [float(first@n), float(last@n)]
    lateral_support=radius+max(v.front_m,v.rear_m,
        abs(v.implement_offset_m)+v.implement_length_m/2)+v.working_width_m/2
    coords = [(xx*u+yy*n).tolist() for xx, yy in
        [(min(x)-reach,min(y)-lateral_support),
         (max(x)+reach,min(y)-lateral_support),
         (max(x)+reach,max(y)+lateral_support),
         (min(x)-reach,max(y)+lateral_support)]]
    return Polygon(coords)


def pose_footprints(pose,scene):
    """按局部位姿变换车体及固定机具矩形，距离米、朝向弧度。"""
    c,s=math.cos(pose.yaw),math.sin(pose.yaw)
    rotation=np.array([[c,-s],[s,c]])
    return [Polygon(rect@rotation.T+np.array([pose.x,pose.y]))
        for rect in scene.vehicle.rectangles()]


def pose_ready(pose,scene,ready):
    """要求位姿对应的全部车体/机具面都位于当前允许空间，不仅检查参考点。"""
    return all(ready.covers(part) for part in pose_footprints(pose,scene))


# 规则策略的统一调头模板库，包含有限的前进、折返和倒车选择。
# 逐模板检查可用端部空间，不能把未检查的模板简单标为可通行。
class LocalTurns:
    """规则策略的有界局部调头候选器，所有模板受截止时间、运动与包络约束。"""
    def __init__(self, scene, deadline):
        """设置局部模板求解的场景与截止时间，所有调用共享本期限。"""
        self.scene = scene
        self.deadline = deadline
        self.native = regular_turns_Connector(scene, io.RouteSettings(), deadline)
        self.native.backend=SampleBoundedBackend(scene)
        for _,turner in self.native.backend.turners:
            turner.setUsingCache(False)
        self.counts = Counter()
        self.errors = []
        self.park_radius=scene.vehicle.min_turn_radius_m
        self.nominal_lateral_m=2*scene.vehicle.min_turn_radius_m
        self.park_templates={}

    def parked_native(self,a,b,backward,radius=None):
        """尝试停车/换挡相关原生模板，必须有明确运动证据，不能用同点朝向跳变替代动作。
        
        F2C circular RS, only legal if all moving steering checks pass.
        
        Wheel steering can change at a genuine stop/gear cusp, as in the
        existing route operation contract. Same-gear moving discontinuities
        receive no exemption. Stop duration is an unmeasured estimate, and
        the vehicle heading never jumps. Actual arc radius is common to the
        attempt and never smaller than its chosen vehicle minimum."""
        radius=self.park_radius if radius is None else radius
        c,s=math.cos(a.yaw),math.sin(a.yaw)
        dx,dy=b.x-a.x,b.y-a.y
        # Reuse the same native family for mirrored, translated row pairs.
        # Subnanometre phase noise at an exactly opposite heading can select
        # a different equal-length RS family. Canonicalise within the existing
        # endpoint tolerance, then still check the actual endpoints below.
        x,y=round(c*dx+s*dy,8),round(-s*dx+c*dy,8)
        angle=regular_turns_wrap(b.yaw-a.yaw)
        if abs(abs(angle)-math.pi)<1e-8:
            angle=math.pi
        reflected=y<0
        end=io.Pose(x,abs(y),-angle if reflected else angle)
        if abs(abs(end.yaw)-math.pi)<1e-8:
            end=io.Pose(end.x,end.y,math.pi)
        key=(radius,backward,end.x,end.y,round(end.yaw,9))
        if key not in self.park_templates:
            f2c=self.native.backend.f2c;v=self.scene.vehicle
            robot=f2c.Robot(v.body_width_m,v.working_width_m)
            robot.setMinTurningRadius(radius);robot.setMaxDiffCurv(v.max_curvature_rate)
            robot.setCruiseVel(v.transit_speed_mps);robot.setTurnVel(v.turn_speed_mps)
            turner=f2c.PP_ReedsSheppCurves()
            turner.setDiscretization(self.scene.settings.sampling_step_m)
            turner.setUsingCache(False)
            shift=math.pi if backward else 0.
            path=turner.createTurn(robot,f2c.Point(0,0),shift,
                f2c.Point(end.x,end.y),end.yaw+shift)
            p=self.native.backend.convert_path(path,('from','to'),'turn').points.copy()
            if backward:
                p[:,2]-=math.pi;p[:,3]*=-1
            self.park_templates[key]=p
            self.counts['parked_native_calls']+=1
        else:
            self.counts['parked_template_hits']+=1
        p=self.park_templates[key].copy()
        if reflected:
            p[:,1]*=-1;p[:,2]*=-1
        xy=p[:,:2].copy()
        p[:,0]=a.x+c*xy[:,0]-s*xy[:,1]
        p[:,1]=a.y+s*xy[:,0]+c*xy[:,1];p[:,2]+=a.yaw
        motion=regular_turns_Motion(p,'turn')
        if not (io._pose_close(io._pose_at(motion,False),a,self.scene) and
                io._pose_close(io._pose_at(motion,True),b,self.scene)):
            raise ValueError('NATIVE_ENDPOINT_MISMATCH')
        return motion

    def working_circle(self,a,b,radius):
        """尝试开启机具的圆弧候选并检查覆盖与包络，不把所有圆弧都认定已作业。
        
        Exact fillet of the F2C offset contour, with stationary steering.
        
        At the exact circle tangency the shortest-Dubins implementation can
        select a winding family because of floating-point degeneracy. The
        geometric circle has no shortest-path ambiguity: its center, sweep
        and tangent are fixed. It is a geometric headland proposal, not a
        claimed native F2C turn. Body connectors use a separately labelled
        exact aligned semicircle or native F2C paths."""
        theta=regular_turns_wrap(b.yaw-a.yaw)
        if radius<self.scene.vehicle.min_turn_radius_m or abs(theta)<1e-6:
            return None
        sign=1 if theta>0 else -1
        center=np.array([a.x,a.y])+sign*radius*np.array([-math.sin(a.yaw),math.cos(a.yaw)])
        alpha=math.atan2(a.y-center[1],a.x-center[0])
        count=max(2,math.ceil(abs(theta)*radius/self.scene.settings.sampling_step_m))
        if count+1>self.scene.settings.max_motion_samples:
            return None
        t=np.linspace(0,1,count+1)
        angle=alpha+theta*t
        p=np.column_stack((center[0]+radius*np.cos(angle),center[1]+radius*np.sin(angle),
            a.yaw+theta*t,np.ones(count+1)))
        motion=regular_turns_Motion(p,'work',implement_on=True)
        if not (io._pose_close(io._pose_at(motion,False),a,self.scene) and
                io._pose_close(io._pose_at(motion,True),b,self.scene)):
            return None
        found=self._check(motion,a,b,GeometryCollection(),port_window(a,b,self.scene),True)
        if found:
            motion.operation_events=found[1]['operations']
            motion.curve_family='GEOMETRIC_PARALLEL_FILLET_OF_F2C_OFFSET'
            self.counts['accepted_WORKING_CIRCLE']+=1
        return found

    def aligned_semicircle(self,a,b,ready,window):
        """尝试满足端口对齐的半圆模板，位置、朝向和车辆尺度都参与约束。
        
        An exact constant-curvature U at aligned opposite row ports.
        
        R = row separation / 2 is fixed by geometry, not a relaxed vehicle
        limit. Native shortest-path degeneracy can introduce microscopic
        straight/reverse pieces at this exact configuration; this explicitly
        geometric template has one arc and no hidden moving steering jumps.
        The existing work/turn boundary stops remain recorded and replayed."""
        dx,dy=b.x-a.x,b.y-a.y
        longitudinal=dx*math.cos(a.yaw)+dy*math.sin(a.yaw)
        lateral=-dx*math.sin(a.yaw)+dy*math.cos(a.yaw)
        radius=abs(lateral)/2
        if (abs(longitudinal)>1e-7 or abs(abs(regular_turns_wrap(b.yaw-a.yaw))-math.pi)>1e-7
                or radius<self.scene.vehicle.min_turn_radius_m or radius==0):
            return None
        theta=math.copysign(math.pi,lateral)
        center=np.array([a.x,a.y])+lateral/2*np.array([-math.sin(a.yaw),math.cos(a.yaw)])
        alpha=math.atan2(a.y-center[1],a.x-center[0])
        count=math.ceil(math.pi*radius/self.scene.settings.sampling_step_m)
        if count+1>self.scene.settings.max_motion_samples:
            return None
        t=np.linspace(0.,1.,count+1);angle=alpha+theta*t
        motion=regular_turns_Motion(np.column_stack((center[0]+radius*np.cos(angle),
            center[1]+radius*np.sin(angle),a.yaw+theta*t,np.ones(count+1))),'turn')
        found=self._check(motion,a,b,ready,window,False)
        if found:
            found[1].update(curve_family='GEOMETRIC_ALIGNED_SEMICIRCLE',actual_arc_radius_m=radius)
            self.counts['accepted_ALIGNED_SEMICIRCLE']+=1
        return found

    def _check(self, motion, a, b, ready, window, work):
        """统一检查局部模板的运动、实际包络及驾驶规则，返回证据或失败。"""
        shape = (transfer_class(motion.points) if motion.kind=='transfer' else
            template_class(motion.points, regular_turns_wrap(b.yaw-a.yaw)))
        if shape is None or (work and np.any(motion.points[:-1, 3] < 0)):
            self.counts['TEMPLATE_REJECTED'] += 1
            return None
        if not work and not np.all(shapely.covers(ready,shapely.points(motion.points[:,:2]))):
            self.counts['UNWORKED_REFERENCE']+=1
            return None
        issues, parts = self.native.checker.physical(motion)
        if issues:
            self.counts.update(issues)
            return None
        if window is not None and not np.all(shapely.covers(
                window.buffer(self.scene.settings.geometry_epsilon_m), parts)):
            self.counts['OUTSIDE_LOCAL_PORT'] += 1
            return None
        if not work and not np.all(shapely.covers(ready, parts)):
            self.counts['UNWORKED_CROSSING'] += 1
            return None
        runs, shifts, reverse = gear_statistics(motion.points)
        ds = np.linalg.norm(np.diff(motion.points[:, :2], axis=0), axis=1)
        seconds = float(np.sum(ds/np.where(motion.points[:-1,3]<0,
            self.scene.vehicle.reverse_speed_mps,self.scene.vehicle.turn_speed_mps)))
        _,_,_,operations,steer_seconds=io._gear_and_steering(motion,self.scene,io.RouteSettings())
        seconds += shifts*self.scene.vehicle.gear_change_seconds+steer_seconds
        return motion, {'template':shape,'length_m':motion.length,
            'gear_runs':runs,'gear_shifts':shifts,'reverse_legs':reverse,
            'reverse_m':float(ds[motion.points[:-1,3]<0].sum()),'seconds':seconds,
            'port_wkt':window.wkt if window is not None else None,
            'operations':operations,'operation_time_status':'UNMEASURED_STOP_STEER_ESTIMATE'}

    def connect(self, a, b, ready, *, work=False, transfer=False):
        """在共享期限内选择合法局部调头，候选未找到不代表田块物理不可达。"""
        if time.perf_counter() >= self.deadline:
            self.counts['SEARCH_BUDGET_EXHAUSTED'] += 1
            return None
        if io._pose_close(a,b,self.scene):
            return None, {'template':'CONTIGUOUS','length_m':0.,'gear_runs':[],
                'gear_shifts':0,'reverse_legs':0,'reverse_m':0.,'seconds':0.,'port_wkt':None}
        if not work:
            ready=ready.buffer(self.scene.settings.geometry_epsilon_m)
            shapely.prepare(ready)
            if not pose_ready(a,self.scene,ready) or not pose_ready(b,self.scene,ready):
                self.counts['PORT_NOT_FULLY_WORKED']+=1
                return None
        window = None if transfer else port_window(a,b,self.scene)
        if window is not None and window.is_empty:
            self.counts['ENDPOINTS_NOT_LOCAL'] += 1
            return None
        if not work and not transfer:
            found=self.aligned_semicircle(a,b,ready,window)
            if found:
                return found
        choices=[]
        # A CC corner at the *minimum* radius can add a full winding loop to
        # reconcile zero-curvature tangents. Derive up to two gentler actual
        # radii from tangent support; the selected vehicle minimum stays fixed.
        if work:
            for radius in regular_turns_gentle_radii(a,b,self.scene):
                if radius>2*self.scene.vehicle.min_turn_radius_m:
                    self.counts['EXCESS_GENTLE_RADIUS_REJECTED']+=1
                    continue
                if time.perf_counter()>=self.deadline:
                    break
                try:
                    curve=regular_turns_gentle_native(self.native,a,b,radius)
                    found=self._check(curve,a,b,ready,window,True)
                    if found:
                        return found
                except (ValueError,RuntimeError,IndexError):
                    self.counts['NATIVE_REJECTED']+=1
        straight=io._direct_motion(a,b,self.scene)
        if straight is not None:
            straight=regular_turns_Motion(straight.points,'work' if work else
                'transfer' if transfer else 'turn',implement_on=work)
            found=self._check(straight,a,b,ready,window,work)
            if found:
                choices.append(found)
        families=[(0,False)] if work else [(0,False),(1,False),(1,True)]
        for index,backward in families:
            if time.perf_counter() >= self.deadline:
                break
            if index >= len(self.native.backend.turners):
                continue
            try:
                curve=self.native.native(a,b,index,backward)
                curve=regular_turns_Motion(curve.points,'work' if work else
                    'transfer' if transfer else 'turn',implement_on=work)
                found=self._check(curve,a,b,ready,window,work)
                if found:
                    choices.append(found)
                    if found[1]['template']=='LOCAL_U' or (transfer and
                            found[1]['reverse_legs']==0):
                        break
            except (ValueError,RuntimeError,IndexError) as exc:
                self.counts['NATIVE_REJECTED']+=1
        if not choices and not work and not transfer:
            lateral=abs(-(b.x-a.x)*math.sin(a.yaw)+(b.y-a.y)*math.cos(a.yaw))
            spacing=self.scene.vehicle.working_width_m*(1-self.scene.settings.overlap_fraction)
            # Residue joins can be *two* rows apart in a three-row skip.
            # Classifying only one-row joins as compact incorrectly asks the
            # regular skip's large arc radius to fit that smaller opening.
            compact=lateral<self.nominal_lateral_m-.1*spacing
            radii=([self.scene.vehicle.min_turn_radius_m,
                    self.scene.vehicle.min_turn_radius_m+self.scene.vehicle.working_width_m/6] if compact else
                [self.park_radius,self.park_radius+self.scene.vehicle.working_width_m/6])
            for radius,backward in [(r,bk) for r in radii for bk in [False,True]]:
                if time.perf_counter()>=self.deadline:
                    break
                try:
                    curve=self.parked_native(a,b,backward,radius)
                    found=self._check(curve,a,b,ready,window,False)
                    if found:
                        found[1]['curve_family']='F2C_PARKED_REEDS_SHEPP'
                        found[1]['actual_arc_radius_m']=radius
                        found[1]['radius_reason']='COMPACT_REMAINDER' if compact else 'REGULAR_SKIP'
                        choices.append(found)
                except (ValueError,RuntimeError,IndexError):
                    self.counts['PARKED_NATIVE_REJECTED']+=1
        if not choices and transfer and time.perf_counter()<self.deadline:
            for guide in regular_turns_visibility_guides(ready,self.scene,a,b,2):
                for index in range(min(2,len(self.native.backend.turners))):
                    if time.perf_counter()>=self.deadline:
                        break
                    try:
                        raw=self.native.native(a,b,index,False,guide)
                        curve=regular_turns_Motion(raw.points,'transfer',implement_on=False)
                        found=self._check(curve,a,b,ready,None,False)
                        if found:
                            found[1]['template']='BLOCK_TRANSFER'
                            choices.append(found)
                    except (ValueError,RuntimeError,IndexError):
                        self.counts['NATIVE_REJECTED']+=1
                if choices:
                    break
        if not choices:
            return None
        found=min(choices,key=lambda c:(c[1]['gear_shifts'],c[1]['seconds']))
        if transfer:
            found[1]['template']='BLOCK_TRANSFER'
        self.counts['accepted_'+found[1]['template']]+=1
        return found

# ==========================================================================
# 12. 规则连续路线策略
# 按有限条带排序和局部转弯候选生成主体行程；保存搜索和认证状态。
# ==========================================================================

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time

import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.ops import unary_union

from scene import load_scene as regular_load_scene
from scene import Pose as regular_Pose
import route_planner as io
regular_straight = straight
regular_work_coverage = work_coverage
regular_prepare_headlands = prepare_headlands
regular_LocalTurns = LocalTurns
regular_pose_ready = pose_ready


@dataclass(frozen=True)
class RegularSettings:
    """规则路线模型的半径候选、田头圈数和预算设置，不替代实测车辆参数。"""
    route_strategy: str = 'REGULAR_HEADLAND_FIRST'
    radius_candidates_m: tuple = (5., 4., 6.)
    headland_pass_count: int = 3
    field_budget_seconds: float = 30.
    skip_candidates: int = 2

    def __post_init__(self):
        """校验规则策略半径、圈数、资源及枚举，布尔值不能冒充合法数值。"""
        if self.route_strategy != 'REGULAR_HEADLAND_FIRST':
            raise ValueError('UNKNOWN_ROUTE_STRATEGY')
        if (not isinstance(self.radius_candidates_m,(list,tuple)) or
                not 1 <= len(self.radius_candidates_m) <= 3 or
                any(type(x) not in (int,float) or not math.isfinite(x) or
                    not 4 <= x <= 6 for x in self.radius_candidates_m)):
            raise ValueError('RADIUS_MUST_BE_FINITE_IN_4_TO_6')
        if len(set(self.radius_candidates_m)) != len(self.radius_candidates_m):
            raise ValueError('DUPLICATE_RADIUS_CANDIDATE')
        if type(self.headland_pass_count) is not int or self.headland_pass_count not in (2,3):
            raise ValueError('HEADLAND_PASSES_MUST_BE_2_OR_3')
        if (type(self.field_budget_seconds) not in (int,float) or
                not math.isfinite(self.field_budget_seconds) or self.field_budget_seconds <= 0):
            raise ValueError('INVALID_FIELD_BUDGET')
        if type(self.skip_candidates) is not int or self.skip_candidates not in (1,2):
            raise ValueError('INVALID_SKIP_CANDIDATES')


def load_settings(path):
    """读取统一路线模式或外部旧规则配置，并按RegularSettings校验。"""
    if path is None:
        return RegularSettings()
    data=config_section(path, "routes")
    if not isinstance(data,dict):
        raise ValueError('ROUTE_CONFIG_MUST_BE_OBJECT')
    unknown=set(data)-{f.name for f in fields(RegularSettings)}
    if unknown:
        raise ValueError(f'UNKNOWN_ROUTE_CONFIG_FIELDS: {sorted(unknown)}')
    return RegularSettings(**data)


def hypotheses_scene(source,radius):
    """在场景副本中使用一组半径假设，原场景不修改，结果须注明假设范围。"""
    return replace(source,vehicle=replace(source.vehicle,min_turn_radius_m=float(radius)))


def reference_area(scene,heading):
    """用车辆轮廓顶点平移交集做必要空间筛查，最终完整扫掠检查不能省略。
    
    Necessary corner constraints only. Final swept-envelope check is mandatory.
    
    Four translated polygon intersections can miss an edge spanning a hole;
    they are a candidate clip, never a physical certificate."""
    c,s=math.cos(heading),math.sin(heading)
    rotation=np.array([[c,-s],[s,c]])
    result=None
    body,tool=scene.vehicle.rectangles()
    body_zone=scene.travel.buffer(-scene.vehicle.safety_margin_m)
    for rectangle,zone in [(body,body_zone),(tool,scene.target)]:
        for x,y in rectangle@rotation.T:
            shifted=translate(zone,-float(x),-float(y))
            result=shifted if result is None else result.intersection(shifted)
    return result


def _regular_lines(geom):
    """提取规则策略需要的独立正长度线段，不跨孔洞补连接。"""
    if geom.is_empty:
        return []
    if geom.geom_type=='LineString':
        return [geom]
    if hasattr(geom,'geoms'):
        return [line for part in geom.geoms for line in _regular_lines(part)]
    return []


def _headland_port_options(candidate,u,heading,need_span,scene,ready):
    """在真实已作业田头内沿行驶方向生成有限端口，位姿的车体和机具须同时可容纳。
    
    Bounded port samples in actually cut headlands, in travel direction.
    
    A point inside a headland is not enough: both rigid footprints must fit.
    An artificial partition edge is not a turning band; the stroke may have
    to extend to the true prepared boundary instead. Final line safety and
    crop coverage are rechecked by body_variants."""
    positions=[]
    origin=np.asarray(candidate.coords[0]);anchor=float(origin@u)
    for part in _regular_lines(candidate.intersection(ready)):
        limits=np.asarray(part.coords)@u
        low,high=float(limits.min()),float(limits.max())
        for fraction in [.05,.2,.35,.5,.65,.8,.95]:
            position=low+(high-low)*fraction
            point=origin+(position-anchor)*u
            if regular_pose_ready(regular_Pose(*point,heading),scene,ready):
                positions.append(position)
    v=scene.vehicle
    entry_limit=float(need_span.min())-(v.implement_offset_m-v.implement_length_m/2)
    exit_limit=float(need_span.max())-(v.implement_offset_m+v.implement_length_m/2)
    entries=[p for p in positions if p<=entry_limit]
    exits=[p for p in positions if p>=exit_limit]
    if not entries or not exits:
        return []
    entries=sorted(set(entries));exits=sorted(set(exits))
    options=[]
    for name,i,j in [('HEADLAND_PORT',-1,0),('HEADLAND_PORT_MID',len(entries)//2,len(exits)//2),
                     ('HEADLAND_PORT_OUTER',0,-1)]:
        low,high=entries[i],exits[j]
        if high-low>=1. and (low,high) not in [p for _,p in options]:
            options.append((name,(low,high)))
    return options


def _headland_ports(candidate,u,heading,need_span,scene,ready):
    """按当前田头和行进方向收集可用端口，供主体衔接的后续运动验证。"""
    options=_headland_port_options(candidate,u,heading,need_span,scene,ready)
    return options[0][1] if options else None


def body_variants(job,scene,heads,turns,headland_pass_count=3,placement_m=0.):
    """为继承直线任务构造保持原扫掠的真实车辆位姿方案，延长车辆参考线不等于改变覆盖义务。
    
    Two genuinely checked poses per inherited straight task.
    
    Extending a reference line is not changing a frozen swath: the derived
    line, actual footprint and source ID are exported. Equivalence is required
    on the source footprint still needing work, not on already cut headlands."""
    areas={};variants={};required={};failures=[]
    head_ready=heads.buffer(scene.settings.geometry_epsilon_m)
    shapely.prepare(head_ready)
    extension=scene.vehicle.working_width_m*3+max(scene.vehicle.front_m,
        abs(scene.vehicle.implement_offset_m)+scene.vehicle.implement_length_m/2)
    # A frozen region seam can be much farther than 3 widths from a real
    # prepared headland. Reach the field's finite bounding span before
    # clipping at safe-space components; never bridge a hole or a gap. The
    # resulting ON stroke can supply neighbouring crop and is exported as
    # derived work rather than falsely inventing an OFF corridor at the seam.
    xmin,ymin,xmax,ymax=scene.travel.bounds
    extension=max(extension,math.hypot(xmax-xmin,ymax-ymin)+extension)
    tolerance=scene.settings.coverage_tolerance_m2
    # The reserve changes the *new phase assignment*, not the original target
    # and not the dynamic ready ledger. Any uncut part of this reserve remains
    # in the final original-target difference and can fail acceptance.
    core=scene.target.buffer(-(headland_pass_count*scene.vehicle.working_width_m+placement_m))
    staging=max(scene.vehicle.front_m,scene.vehicle.rear_m,
        abs(scene.vehicle.implement_offset_m)+scene.vehicle.implement_length_m/2)+.1
    for task_index,task in enumerate(job.tasks):
        if time.perf_counter()>=turns.deadline:
            failures.extend({'task_id':t.task_id,'region_id':t.region_id,
                'reason':'SEARCH_BUDGET_EXHAUSTED_WORK_PREPARATION'} for t in job.tasks[task_index:])
            break
        needed=task.frozen_sweep.intersection(core).difference(heads)
        required[task.task_id]=needed
        if needed.area<=tolerance:
            variants[task.task_id]=[]
            continue
        variants[task.task_id]=[]
        for reverse in [False,True]:
            heading=task.heading_rad+(math.pi if reverse else 0)
            key=round(heading%(2*math.pi),9)
            if key not in areas:
                areas[key]=reference_area(scene,heading).intersection(reference_area(scene,heading+math.pi))
            u=np.array([math.cos(heading),math.sin(heading)])
            coords=np.array(task.reference_line.coords)
            midpoint=coords.mean(axis=0)
            positions=coords@u
            line=LineString([midpoint+(positions.min()-midpoint@u-extension)*u,
                             midpoint+(positions.max()-midpoint@u+extension)*u])
            candidates=_regular_lines(line.intersection(areas[key]))
            need_coords=shapely.get_coordinates(needed)
            need_span=need_coords@u
            desired_min=float(need_span.min())-staging
            # A rear implement reaches the last crop before the full vehicle
            # reaches the outer staging port. Stop as soon as that footprint
            # covers the obligation, keeping room in front for a local turn.
            desired_max=float(need_span.max())+staging
            selected=[]
            for candidate in sorted(candidates,key=lambda g:g.length,reverse=True):
                if candidate.length<1:
                    continue
                p=np.array(candidate.coords)
                order=np.argsort(p@u)
                head_ports=dict(_headland_port_options(candidate,u,heading,need_span,scene,head_ready))
                # Stay just inside the clipping boundary; final body buffer
                # uses a polygonal sagitta compensation, so exact touching is
                # not necessarily safe. This never loosens the final check.
                for mode in [*head_ports,'STAGING','BOUNDARY_PORT','ENTRY_INSET']:
                    if mode in head_ports:
                        low,high=head_ports[mode]
                    elif mode=='STAGING':
                        low=max(float(p[order[0]]@u)+.005,desired_min)
                        # Stop once the unchanged crop obligation is covered.
                        # The outer safe limit alone does not reserve turning
                        # room. Coverage and dynamic OFF access are still
                        # checked separately; this is only a candidate port.
                        high=min(float(p[order[-1]]@u)-.055,desired_max)
                    elif mode=='ENTRY_INSET':
                        # A small inward entry creates room for the front of
                        # the machine during the final reverse/forward leg.
                        # Exit remains at the common outer port. Coverage of
                        # the unchanged task obligation is rechecked below.
                        low=float(p[order[0]]@u)+.055+scene.vehicle.working_width_m/4
                        high=float(p[order[-1]]@u)-.055
                    else:
                        pad=.055
                        low=float(p[order[0]]@u)+pad;high=float(p[order[-1]]@u)-pad
                    a=p[order[0]]+u*(low-float(p[order[0]]@u))
                    b=p[order[-1]]+u*(high-float(p[order[-1]]@u))
                    if (b-a)@u<1:
                        continue
                    motion=regular_straight(a,b,task.task_id)
                    issues,_=turns.native.checker.physical(motion)
                    if issues:
                        continue
                    sweep=regular_work_coverage(motion,scene)
                    gap=needed.difference(sweep).area
                    if gap>tolerance:
                        continue
                    selected.append({'motion':motion,'sweep':sweep,'reversed':reverse,
                        'task':task,'required':needed,'required_gap_m2':gap,'endpoint_mode':mode,
                        'initial_exit_ready':regular_pose_ready(io._pose_at(motion,True),scene,head_ready)})
                if selected:
                    break
            variants[task.task_id].extend(selected)
        if not variants[task.task_id]:
            failures.append({'task_id':task.task_id,'region_id':task.region_id,
                'reason':'NO_SAFE_DERIVED_WORK_VARIANT','required_m2':needed.area})
    return variants,required,failures


def geometry_blocks(job,scene,variants,heads=None):
    """按相邻行的端点匹配形成区块，孔洞碎片和明显形态变化使区块断开，不按固定任务数强分。
    
    Match adjacent rows by endpoints, not an arbitrary task-count chunk.
    
    Disjoint hole fragments and abrupt arm-length changes create separate
    execution blocks. Original region geometries/IDs are preserved."""
    blocks=[]
    head_ready=heads.buffer(scene.settings.geometry_epsilon_m) if heads is not None else None
    if head_ready is not None:
        shapely.prepare(head_ready)
    reach=2*scene.vehicle.min_turn_radius_m+scene.vehicle.working_width_m+max(
        scene.vehicle.front_m,abs(scene.vehicle.implement_offset_m)+scene.vehicle.implement_length_m/2)
    for region in job.regions:
        tasks=[t for t in job.tasks if t.region_id==region.region_id and variants.get(t.task_id)]
        tasks.sort(key=lambda t:(t.row_index,t.suggested_order,t.task_id))
        active=[]
        for task in tasks:
            supported=(head_ready is None or any(
                regular_pose_ready(io._pose_at(v['motion'],False),scene,head_ready) and
                regular_pose_ready(io._pose_at(v['motion'],True),scene,head_ready)
                for v in variants[task.task_id]))
            # Frozen boustrophedon alternates headings by pi. Block geometry
            # must use an unoriented common frame, otherwise opposite rows
            # look mirrored and can spuriously match across the whole field.
            heading=task.heading_rad%math.pi
            if abs(heading-math.pi)<1e-7 or abs(heading)<1e-7:
                heading=0.
            u=np.array([math.cos(heading),math.sin(heading)])
            n=np.array([-u[1],u[0]])
            span=np.sort(np.array(task.reference_line.coords)@u)[[0,-1]]
            lateral=float(np.array(task.reference_line.centroid.coords[0])@n)
            matches=[]
            for block in active:
                previous=block['tasks'][-1]
                if (previous.row_index>=task.row_index or
                        abs(math.sin(previous.heading_rad-task.heading_rad))>1e-5):
                    continue
                if (abs(lateral-block['last_lateral'])>1.6*scene.vehicle.working_width_m or
                    np.max(np.abs(span-block['last_span']))>reach):
                    continue
                if min(span[1],block['last_span'][1])<=max(span[0],block['last_span'][0]):
                    continue
                matches.append((float(np.abs(span-block['last_span']).sum()),block))
            if matches:
                block=min(matches,key=lambda p:(p[0],p[1]['block_id']))[1]
                block['tasks'].append(task)
                block['initial_headland_ports_supported'] &= supported
                if supported:
                    block['headland_supported_task_ids'].append(task.task_id)
                block.update(last_span=span,last_lateral=lateral)
            else:
                block={'block_id':f'{region.region_id}_b{len(blocks)+1:03d}',
                    'region_id':region.region_id,'tasks':[task],
                    'last_span':span,'last_lateral':lateral,
                    'headland_supported_task_ids':[task.task_id] if supported else [],
                    'initial_headland_ports_supported':supported}
                blocks.append(block);active.append(block)
    # Support is a scheduling precondition, never a geometric cut or a reason
    # to move an edge row to the far end of a rectangular field's itinerary.
    return sorted(blocks,key=lambda b:(
        next(r.sequence_index for r in job.regions if r.region_id==b['region_id']),b['block_id']))


def skip_order(tasks,skip):
    """构造有限跳行再补行顺序，全部任务仍保留一次，不全排列搜索。"""
    ordered=[]
    for residue in range(skip):
        group=tasks[residue::skip]
        ordered.extend(group[::-1] if residue%2 else group)
    return ordered


def _aligned_receiving(variant,exit_pose,scene,turns,ready):
    """只沿后继作业直线调整入口，必须仍覆盖未变的目标义务，不为对齐转弯而缩掉必要作业。
    
    Move only the next ON stroke's entry along its own straight line.
    
    Alignment is useful only when the unchanged crop obligation still fits
    and the receiving full pose lies in previously cut crop. Never shorten a
    stroke just to make a prettier connection or open its future work."""
    original=variant['motion'];p=original.points
    if abs(abs(io.wrap(float(p[0,2]-exit_pose.yaw)))-math.pi)>1e-7:
        return None
    u=np.array([math.cos(exit_pose.yaw),math.sin(exit_pose.yaw)])
    start=p[0,:2]+(np.array([exit_pose.x,exit_pose.y])-p[0,:2])@u*u
    heading=p[0,2];forward=np.array([math.cos(heading),math.sin(heading)])
    if (p[-1,:2]-start)@forward<1.:
        return None
    if not regular_pose_ready(regular_Pose(*start,heading),scene,ready.buffer(scene.settings.geometry_epsilon_m)):
        return None
    motion=regular_straight(start,p[-1,:2],original.task_id)
    issues,_=turns.native.checker.physical(motion)
    if issues:
        return None
    sweep=regular_work_coverage(motion,scene)
    if variant['required'].difference(ready.union(sweep)).area>scene.settings.coverage_tolerance_m2:
        return None
    return dict(variant,motion=motion,sweep=sweep,endpoint_mode='HEADLAND_PORT_ALIGNED')


def _borrowed_variants(choices,scene,turns,worked,ready):
    """借已完成邻近作业作为起步空间，保留当前任务原覆盖，不能预借未来任务。
    
    Use already completed neighbours as staging, without recutting them.
    
    At most two new strokes per direction are proposed. Original required
    crop is conserved: its previously covered part plus this ON stroke must
    still cover it. No middle-of-stroke implement switch is introduced."""
    result=[]
    for reversed_direction in [False,True]:
        candidates=[v for v in choices if v['reversed']==reversed_direction]
        if not candidates:
            continue
        base=max(candidates,key=lambda v:v['motion'].length)
        remaining=base['required'].difference(worked)
        if remaining.area<=scene.settings.coverage_tolerance_m2:
            continue
        heading=float(base['motion'].points[0,2]);u=np.array([math.cos(heading),math.sin(heading)])
        candidate=LineString(base['motion'].points[:,:2]);origin=base['motion'].points[0,:2]
        spans=shapely.get_coordinates(remaining)@u
        for _,(low,high) in _headland_port_options(candidate,u,heading,spans,scene,ready)[:2]:
            a=origin+(low-float(origin@u))*u;b=origin+(high-float(origin@u))*u
            motion=regular_straight(a,b,base['motion'].task_id)
            issues,_=turns.native.checker.physical(motion)
            if issues:
                continue
            sweep=regular_work_coverage(motion,scene)
            if base['required'].difference(worked.union(sweep)).area>scene.settings.coverage_tolerance_m2:
                continue
            if motion.length>=base['motion'].length-.01:
                continue
            result.append(dict(base,motion=motion,sweep=sweep,
                endpoint_mode='BORROWED_READY_PORT',initial_exit_ready=True))
    return result


def _body_attempt(blocks,variants,scene,turns,heads,skip,first_reverse,endpoint_mode='STAGING'):
    """按一组排序尝试规则主体路线，逐段更新已作业空间并记录合法连接或失败。"""
    worked=heads;events=[];completed=[];failures=[];previous=None
    allowed_non_target=scene.travel.difference(scene.target)
    spacing=scene.vehicle.working_width_m*(1-scene.settings.overlap_fraction)
    turns.park_radius=max(scene.vehicle.min_turn_radius_m,
        skip*spacing/2+scene.vehicle.working_width_m/4)
    turns.nominal_lateral_m=skip*spacing
    groups=[];deferred_groups=[]
    for block in blocks:
        tasks=block['tasks'][::-1] if first_reverse else block['tasks']
        supported=set(block['headland_supported_task_ids'])
        # Rounded headlands leave corner rows without initial full-vehicle
        # staging space. Cut the regular interior group first, then attempt
        # those rows using the *actual* crop ledger. They remain in the same
        # geometric block and must use a local connection, not a field jump.
        interior=[t for t in tasks if t.task_id in supported]
        deferred=[t for t in tasks if t.task_id not in supported]
        if interior:
            groups.append((block,skip_order(interior,skip)))
        if deferred:
            deferred_groups.append((block,skip_order(deferred,skip)))
    # The neighbour's interior may provide real crop clearance for a corner
    # row. Keep that row's frozen ownership and geometry, but defer it until
    # the regular groups have actually worked; never open future crop.
    groups.extend(deferred_groups)
    promised=None
    for group_index,(block,ordered) in enumerate(groups):
        bank={t.task_id:variants[t.task_id] for t in ordered}
        if group_index>0:
            actual_ready=worked.union(allowed_non_target).buffer(scene.settings.geometry_epsilon_m)
            shapely.prepare(actual_ready)
            bank={t.task_id:[*_borrowed_variants(bank[t.task_id],scene,turns,worked,actual_ready),
                *bank[t.task_id]] for t in ordered}
        canonical=ordered[0].heading_rad%math.pi
        if abs(canonical-math.pi)<1e-7 or abs(canonical)<1e-7:
            canonical=0.
        for rank,task in enumerate(ordered):
            desired=(rank%2==0)==first_reverse
            available=bank[task.task_id]
            if promised is not None:
                available=[promised,*available]
            choices=sorted(available,key=lambda v:(
                promised is not None and v is not promised,
                (math.cos(v['motion'].points[0,2]-canonical)<0)!=desired,
                v['endpoint_mode']!='BORROWED_READY_PORT',
                v['endpoint_mode']!=endpoint_mode))
            found=None;fallback=None;next_promise=None
            for variant in choices:
                motion=variant['motion'];a=io._pose_at(motion,False)
                # ON work can advance the body ahead of its rear implement.
                # Such a legal cutting endpoint can still trap the following
                # OFF move in uncut crop. Reject that trap before committing
                # the stroke, using its actual coverage plus the past ledger.
                last_task=(group_index==len(groups)-1 and rank==len(ordered)-1)
                if not last_task and not variant.get('initial_exit_ready',False):
                    next_ready=worked.union(variant['sweep']).union(allowed_non_target)
                    if not regular_pose_ready(io._pose_at(motion,True),scene,
                            next_ready.buffer(scene.settings.geometry_epsilon_m)):
                        turns.counts['DEAD_END_WORK_PORT']+=1
                        continue
                if previous is None:
                    # Auto-select the legal first work pose; no fictitious
                    # OFF access line from outside the field is inserted.
                    candidate=(variant,None,None)
                else:
                    last_block=events[-1]['block_id']
                    transfer=last_block!=block['block_id']
                    ready=worked.union(allowed_non_target)
                    link=turns.connect(io._pose_at(previous,True),a,ready,transfer=transfer)
                    if link is None:
                        continue
                    candidate=(variant,*link)
                if fallback is None:
                    fallback=candidate
                # One-step port coordination, not permutation search. Only
                # three prepared-headland variants of the *next fixed row*
                # are examined. The ledger includes the current stroke and
                # never the next row's future crop. Preserve the successful
                # receiving port for the next iteration so it is not discarded
                # by the usual direction/phase preference.
                if rank+1<len(ordered):
                    following=ordered[rank+1]
                    next_desired=((rank+1)%2==0)==first_reverse
                    future=[v for v in bank[following.task_id] if
                        (v['endpoint_mode'].startswith('HEADLAND_PORT') or
                         v['endpoint_mode']=='BORROWED_READY_PORT') and
                        (math.cos(v['motion'].points[0,2]-canonical)<0)==next_desired]
                    if not future:
                        found=candidate
                        break
                    ready_next=worked.union(variant['sweep']).union(allowed_non_target)
                    aligned=_aligned_receiving(future[0],io._pose_at(motion,True),scene,turns,ready_next)
                    for receiving in ([aligned] if aligned is not None else [])+future[:3]:
                        if turns.connect(io._pose_at(motion,True),
                                io._pose_at(receiving['motion'],False),ready_next) is not None:
                            found=candidate;next_promise=receiving;break
                    if found is None and time.perf_counter()<turns.deadline:
                        continue
                else:
                    found=candidate
                if found is not None:
                    break
            if found is None:
                found=fallback
            promised=next_promise
            if found is None:
                failures.append({'task_id':task.task_id,'block_id':block['block_id'],
                    'reason':'SEARCH_BUDGET_EXHAUSTED' if time.perf_counter()>=turns.deadline
                        else 'NO_FEASIBLE_REGULAR_CONNECTION'})
                return events,worked,completed,failures
            variant,link,evidence=found
            if link is not None:
                events.append({'motion':link,'phase':'BODY','block_id':block['block_id'],
                    'region_id':block['region_id'],'source_task_id':task.task_id,
                    'turn_evidence':evidence})
            motion=variant['motion']
            events.append({'motion':motion,'phase':'BODY','block_id':block['block_id'],
                'region_id':block['region_id'],'source_task_id':task.task_id,
                'reversed':variant['reversed'],'skip':skip,'endpoint_mode':variant['endpoint_mode'],
                'prior_required_covered_m2':variant['required'].intersection(worked).area})
            worked=worked.union(variant['sweep'])
            completed.append(task.task_id);previous=motion
    return events,worked,completed,failures


# 在有限的统一驾驶假设下组织规则条带和局部连接。
# 车辆假设、路线完成度和参考几何状态分别记录，避免混淆认证范围。
def plan_regular_field(job,settings,checkpoint_path=None):
    """规则整田入口，比较半径/田头/主体候选并保留模型假设、覆盖和运动诊断。"""
    started=time.perf_counter();deadline=started+settings.field_budget_seconds
    source=regular_load_scene(job.scene_path)
    if source.start is not None or source.end is not None:
        raise ValueError('AUTO_ENDPOINT_CONFLICT')
    candidates=[];profiles=[]
    for radius_index,radius in enumerate(settings.radius_candidates_m):
        if time.perf_counter()>=deadline:
            break
        # A failed first radius must not consume the entire shared budget and
        # prevent the other allowed vehicle hypotheses from being examined.
        radius_deadline=min(deadline,started+(radius_index+1)*settings.field_budget_seconds/
            len(settings.radius_candidates_m))
        scene=hypotheses_scene(source,radius);turns=regular_LocalTurns(scene,radius_deadline)
        headstart=time.perf_counter()
        heads,headworked,headfails,placement=regular_prepare_headlands(scene,turns,settings.headland_pass_count)
        heads_seconds=time.perf_counter()-headstart
        prepstart=time.perf_counter()
        variants,required,workfails=body_variants(job,scene,headworked,turns,settings.headland_pass_count,placement)
        blocks=geometry_blocks(job,scene,variants,headworked)
        spacing=scene.vehicle.working_width_m*(1-scene.settings.overlap_fraction)
        skip0=max(1,math.ceil(2*radius/spacing))
        preparation_seconds=time.perf_counter()-prepstart
        body_start=time.perf_counter()
        attempts=[]
        attempt_keys=[(skip,reverse,mode) for skip in range(skip0,skip0+settings.skip_candidates)
            for mode in ['HEADLAND_PORT','STAGING'] for reverse in [False,True]]
        for skip,reverse,mode in attempt_keys:
                if time.perf_counter()>=turns.deadline:
                    break
                events,worked,done,linkfails=_body_attempt(blocks,variants,scene,
                    turns,headworked,skip,reverse,mode)
                gap=scene.target.difference(worked)
                reassigned=[t for t in required if required[t].area<=scene.settings.coverage_tolerance_m2]
                delegated=[t.task_id for t in job.tasks if t.task_id in reassigned and
                    t.frozen_sweep.intersection(scene.target).difference(headworked).area<=scene.settings.coverage_tolerance_m2]
                pending=[t for t in reassigned if t not in delegated]
                failures=[*headfails,*workfails,*linkfails]
                result={'field_id':job.field_id,'radius_m':radius,'scene':scene,
                    'headland_placement_m':placement,
                    'derived_headland_width_m':settings.headland_pass_count*scene.vehicle.working_width_m+placement,
                    'heads':heads,'head_failures':headfails,'events':events,'worked':worked,
                    'target_gap':gap,'completed_task_ids':done,'headland_supplied_task_ids':delegated,
                    'headland_pending_task_ids':pending,
                    'required_task_count':len(job.tasks),'failures':failures,'skip':skip,
                    'derived_body_task_ids':[t for t in variants if variants[t]],
                    'blocks':[{'block_id':b['block_id'],'region_id':b['region_id'],
                        'initial_headland_ports_supported':b['initial_headland_ports_supported'],
                        'headland_supported_task_ids':b['headland_supported_task_ids'],
                        'task_ids':[t.task_id for t in b['tasks']]} for b in blocks],
                    'acceptance_passed':not failures and len(done)+len(delegated)==len(job.tasks)
                        and gap.area<=scene.settings.coverage_tolerance_m2,
                    'input_seam_quality_status':job.upstream_seam_quality_status,
                    'search_status':'BUDGET_EXHAUSTED' if time.perf_counter()>=turns.deadline
                        else 'BOUNDED_CANDIDATES_TESTED'}
                attempts.append(result)
                # Once the actual body obligations are all connected, stop
                # repeating the same schedule to repair an unrelated edge
                # coverage gap. The original-target gap still fails the
                # independent whole-field acceptance below.
                if result['acceptance_passed'] or (not linkfails and not workfails and
                        len(done)==sum(bool(v) for v in variants.values())):
                    break
        if not attempts:
            attempts=[{'field_id':job.field_id,'radius_m':radius,'scene':scene,
                'headland_placement_m':placement,
                'derived_headland_width_m':settings.headland_pass_count*scene.vehicle.working_width_m+placement,
                'heads':heads,'head_failures':headfails,'events':[], 'worked':headworked,
                'target_gap':scene.target.difference(headworked),'completed_task_ids':[],
                'headland_supplied_task_ids':[],'required_task_count':len(job.tasks),
                'headland_pending_task_ids':[],
                'failures':[*headfails,*workfails,{'reason':'SEARCH_BUDGET_EXHAUSTED'}],
                'skip':skip0,'blocks':[],'derived_body_task_ids':[], 'acceptance_passed':False,
                'input_seam_quality_status':job.upstream_seam_quality_status,
                'search_status':'BUDGET_EXHAUSTED'}]
        selected=min(attempts,key=lambda r:(not r['acceptance_passed'],
            -len(r['completed_task_ids'])-len(r['headland_supplied_task_ids']),r['target_gap'].area,
            sum(e['motion'].length for e in r['events'] if not e['motion'].implement_on)))
        candidates.append(selected)
        profiles.append({'radius_m':radius,'headland_seconds':heads_seconds,
            'body_preparation_seconds':preparation_seconds,'body_search_seconds':time.perf_counter()-body_start,
            'candidate_count':len(attempts),'counts':dict(turns.counts),
            'native_counts':dict(turns.native.counts),'native_timing':dict(turns.native.timing)})
        if checkpoint_path:
            # A later hypothesis may stall in C++. Preserve the best result
            # already obtained, not merely the last hypothesis's prefix.
            best=min(candidates,key=_result_key)
            checkpoint=dict(best,profiles=list(profiles),source_scene=source,
                planning_seconds=time.perf_counter()-started)
            _save_checkpoint(Path(checkpoint_path).with_suffix('.partial.pkl'),checkpoint)
        if selected['acceptance_passed'] or (not selected['failures'] and
                set(selected['completed_task_ids'])==set(selected['derived_body_task_ids'])):
            break
    if not candidates:
        raise RuntimeError('NO_RADIUS_ATTEMPT')
    result=min(candidates,key=_result_key)
    result['profiles']=profiles;result['planning_seconds']=time.perf_counter()-started
    result['source_scene']=source
    return result


def _result_key(result):
    """为规则候选排序，完整性及验收条件优先于代价，不把短失败路线误选为完成。"""
    return (not result['acceptance_passed'],
        -len(result['completed_task_ids'])-len(result['headland_supplied_task_ids']),
        result['target_gap'].area,
        sum(e['motion'].length for e in result['events'] if not e['motion'].implement_on))


def _finished_in_time(state,deadline):
    """按子进程实际完成时间戳判断是否在预算内，父进程轮询晚到不应误报超时。
    
    Atomic completion timestamp wins over a late parent polling cycle."""
    finished=state.get('finished_monotonic')
    return (state.get('status')=='COMPLETED' and type(finished) in (int,float)
        and math.isfinite(finished) and finished<=deadline)


def _save_checkpoint(path,result):
    """保存已完成候选检查点，供原生崩溃或超时后读取有效证据。"""
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(result,stream,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _save_state(path,state):
    """原子保存规则计算状态，不能用状态文件代替路线与覆盖审计。"""
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(state));temporary.replace(path)


def _field_process(job,settings,path):
    """隔离单田原生求解；父进程可终止卡住的C++调用并保留故障原因。
    
    One isolated native solve; the parent can kill a stuck C++ call."""
    try:
        result=plan_regular_field(job,settings,path)
        _save_checkpoint(path,result)
        _save_state(path.with_suffix('.state.json'),{'status':'COMPLETED','pid':os.getpid(),
            'finished_monotonic':time.perf_counter()})
    except Exception as exc:
        _save_state(path.with_suffix('.state.json'),{'status':'ERROR',
            'error_type':type(exc).__name__,'message':str(exc),'pid':os.getpid()})


def _failed_result(job,settings,reason,seconds):
    """创建规则单田失败结果，保持田块身份和原因，不能补造零时间成功路线。"""
    source=regular_load_scene(job.scene_path)
    return {'field_id':job.field_id,'radius_m':settings.radius_candidates_m[0],
        'scene':hypotheses_scene(source,settings.radius_candidates_m[0]),'source_scene':source,
        'heads':[],'head_failures':[{'reason':reason}],'events':[],
        'worked':GeometryCollection(),'target_gap':source.target,'completed_task_ids':[],
        'headland_supplied_task_ids':[],'headland_pending_task_ids':[],
        'required_task_count':len(job.tasks),'failures':[{'reason':reason}],
        'derived_body_task_ids':[],
        'skip':None,'blocks':[],'acceptance_passed':False,'profiles':[],
        'planning_seconds':seconds,'search_status':reason,
        'input_seam_quality_status':job.upstream_seam_quality_status}


def _isolated_batch(jobs,settings,workers,out,worker_target=None):
    """管理规则策略隔离进程、硬超时与检查点，独立于默认参考批次的资源自动选核。"""
    context=mp.get_context('spawn');pending=list(jobs);active={};results={};errors=[]
    checkpoints=out/'checkpoints';checkpoints.mkdir()
    try:
        while pending or active:
            while pending and len(active)<workers:
                job=pending.pop(0);path=checkpoints/f'{job.field_id}.pkl'
                child=context.Process(target=worker_target or _field_process,args=(job,settings,path))
                child.start();active[job.field_id]=(child,job,path,time.perf_counter())
            for fid,(child,job,path,began) in list(active.items()):
                elapsed=time.perf_counter()-began
                timeout=elapsed>settings.field_budget_seconds+5.
                state=path.with_suffix('.state.json')
                state_data=json.loads(state.read_text()) if state.exists() else {}
                if path.exists() and _finished_in_time(state_data,began+settings.field_budget_seconds+5.):
                    timeout=False
                if not timeout and child.is_alive() and not state.exists():
                    continue
                if timeout:
                    child.terminate();child.join(1.)
                    if child.is_alive():
                        child.kill();child.join(1.)
                else:
                    child.join(1.)
                    if child.is_alive():
                        child.terminate();child.join(1.)
                code='NATIVE_PROCESS_TIMEOUT' if timeout else 'NATIVE_PROCESS_ERROR'
                available=path if path.exists() else path.with_suffix('.partial.pkl')
                if available.exists():
                    # Only our freshly created, task-local checkpoints are
                    # read. External pickle files are never accepted inputs.
                    with available.open('rb') as stream:
                        result=pickle.load(stream)
                else:
                    result=_failed_result(job,settings,code,elapsed)
                if timeout or not path.exists():
                    result['acceptance_passed']=False
                    result['search_status']=code;result['planning_seconds']=elapsed
                    result['failures'].append({'reason':code})
                    error={'field_id':fid,'error_type':code,'elapsed_seconds':elapsed}
                    if state.exists():
                        error.update(json.loads(state.read_text()))
                    errors.append(error)
                results[fid]=result;del active[fid]
                if len(results)%10==0 or len(results)==len(jobs):
                    print(f'{len(results)}/{len(jobs)} ({len(errors)} process errors/timeouts)',flush=True)
                (out/'progress.json').write_text(json.dumps({'completed':len(results),
                    'field_count':len(jobs),'errors':len(errors),
                    'active':{k:v[0].pid for k,v in active.items()},'latest_field_id':fid}))
            time.sleep(.02)
    finally:
        for child,_,_,_ in active.values():
            if child.is_alive():
                child.terminate();child.join(1.)
                if child.is_alive():
                    child.kill();child.join(1.)
    return results,errors


def run_regular_batch(swath_bundle,out,*,workers=12,field_id=None,route_config=None):
    """历史规则路线批入口，消费可信条带包并保存规则协议结果与独立审计。"""
    from route_planner import export_and_audit
    started=time.perf_counter();settings=load_settings(route_config)
    bundle=Path(swath_bundle).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('OUTPUT_DIRECTORY_MUST_BE_NEW_OR_EMPTY')
    if type(workers) is not int or not 1<=workers<=12:
        raise ValueError('WORKERS_MUST_BE_1_TO_12')
    out.mkdir(parents=True,exist_ok=True)
    source_before={p.name:io._sha256(p) for p in Path(__file__).parent.glob('*.py')}
    _,manifest=io._verify_bundle(bundle)
    jobs=io._field_rows(bundle,manifest)
    if field_id:
        jobs=[j for j in jobs if j.field_id==field_id]
        if not jobs:
            raise ValueError('UNKNOWN_FIELD_ID')
    results={};errors=[];solve_start=time.perf_counter()
    results,errors=_isolated_batch(jobs,settings,workers,out)
    solve_seconds=time.perf_counter()-solve_start
    summary=export_and_audit(jobs,results,out,settings,errors,workers=workers)
    source_after={p.name:io._sha256(p) for p in Path(__file__).parent.glob('*.py')}
    summary.update(input_bundle=str(bundle),workers=workers,settings=asdict(settings),
        solve_wall_seconds=solve_seconds,total_wall_seconds=time.perf_counter()-started,
        source_code_sha256_start=source_before,source_code_sha256_end=source_after,
        source_code_unchanged=source_before==source_after,
        input_field_count=len(jobs))
    summary['acceptance_passed']=bool(summary['acceptance_passed'] and source_before==source_after)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if not k.startswith('source_code')},ensure_ascii=False),flush=True)
    return 0 if summary['acceptance_passed'] else 1

# ==========================================================================
# 13. APPROX_CONNECTED 参考路线策略
# 蛇形顺序、局部 U 形和复用栅格避障；完整几何连接与实车参数认证分开。
# ==========================================================================

import heapq
import html
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import transform
from shapely.affinity import translate
from shapely.prepared import prep
from pyproj import Transformer
from scene import load_scene as approx_load_scene
import route_planner as io


# APPROX_CONNECTED 控制参考搜索、可选车辆位姿组装和最多田头圈数。
# 这些参数不解冻场景、分区或主体条带，也不改变严格策略的车辆约束。
@dataclass(frozen=True)
class ApproxRouteSettings:
    """参考路线设置；完整几何、车辆运动与作物模型细化由独立标志控制，不能混称实车验收。"""
    route_strategy: str = 'APPROX_CONNECTED'
    headland_pass_count: int = 3
    grid_m: float = 1.0
    fine_grid_m: float = 0.25
    max_grid_nodes: int = 150000
    # 数据类默认关闭运动细化，供历史几何策略使用；统一配置的 operational 模式开启。
    # 当前推荐 reference 模式保持关闭，开启 gentle_geometry / assemble_reference 组装完整几何参考。
    motion_refinement: bool = False
    max_motion_queries: int = 6000
    max_headland_seconds: float = 10.0
    operational_refinement: bool = False
    # 几何参考阶段：先连完整行程，再尽量去掉尖角。默认关闭以复现旧批次。
    gentle_geometry: bool = False
    assemble_reference: bool = False
    max_headland_fill_seconds: float = 3.0
    max_headland_fill_segments: int = 24

    def __post_init__(self):
        """校验参考模式标志、田头与网格预算；非法组合不能静默退化成另一策略。"""
        if (type(self.gentle_geometry) is not bool or type(self.assemble_reference) is not bool
                or ((self.gentle_geometry or self.assemble_reference) and self.motion_refinement)):
            raise ValueError('INVALID_GEOMETRY_REFERENCE_MODE')
        if self.route_strategy != 'APPROX_CONNECTED':
            raise ValueError('UNKNOWN_ROUTE_STRATEGY')
        if type(self.headland_pass_count) is not int or self.headland_pass_count not in (2,3):
            raise ValueError('HEADLAND_PASSES_MUST_BE_2_OR_3')
        for name in ('grid_m','fine_grid_m'):
            x=getattr(self,name)
            if type(x) not in (int,float) or not math.isfinite(x) or x<=0:
                raise ValueError('INVALID_GRID_SCALE')
        if self.fine_grid_m>self.grid_m or type(self.max_grid_nodes) is not int or self.max_grid_nodes<1:
            raise ValueError('INVALID_GRID_BUDGET')
        if type(self.motion_refinement) is not bool:
            raise ValueError('INVALID_MOTION_REFINEMENT')
        if type(self.max_motion_queries) is not int or self.max_motion_queries<1:
            raise ValueError('INVALID_MOTION_QUERY_BUDGET')
        if type(self.max_headland_seconds) not in (int,float) or not math.isfinite(self.max_headland_seconds) or self.max_headland_seconds<=0:
            raise ValueError('INVALID_HEADLAND_BUDGET')
        if type(self.operational_refinement) is not bool or (self.operational_refinement and not self.motion_refinement):
            raise ValueError('INVALID_OPERATIONAL_REFINEMENT')
        if type(self.max_headland_fill_seconds) not in (int,float) or not math.isfinite(self.max_headland_fill_seconds) or self.max_headland_fill_seconds<0:
            raise ValueError('INVALID_HEADLAND_FILL_BUDGET')
        if type(self.max_headland_fill_segments) is not int or self.max_headland_fill_segments<1:
            raise ValueError('INVALID_HEADLAND_FILL_SEGMENTS')


def approx_polygons(g):
    """提取参考几何面分量，保留自然断开的田块组成。"""
    if g.geom_type=='Polygon': return [g]
    return [p for x in getattr(g,'geoms',[]) for p in approx_polygons(x)]


def lines(g):
    """递归提取参考线的独立线段，不补造分量间的外部转场。"""
    if g.is_empty: return []
    if g.geom_type in ('LineString','LinearRing'): return [LineString(g.coords)]
    return [p for x in getattr(g,'geoms',[]) for p in lines(x)]


def reference_geometry_angles(g, ua=None, ub=None, closed=False):
    """检查实际折线，避免端点圆滑但曲线内部出现尖点/折返。

    这里只判断参考几何的驾驶观感，不给出实车转弯半径认证。
    零长弦不参与角度计算；闭环的首尾连接也必须检查。
    """
    d=np.diff(np.asarray(g.coords),axis=0);d=d[np.linalg.norm(d,axis=1)>1e-8]
    if not len(d):return dict(max_internal_angle_deg=180.,max_join_angle_deg=180.)
    d=d/np.linalg.norm(d,axis=1)[:,None]
    dot=np.sum(d[:-1]*d[1:],axis=1)
    if closed:dot=np.r_[dot,np.dot(d[-1],d[0])]
    interior=float(np.max(np.degrees(np.arccos(np.clip(dot,-1,1))),initial=0.))
    joins=[angle_degrees(d[0],ua)] if ua is not None else []
    if ub is not None:joins.append(angle_degrees(d[-1],ub))
    return dict(max_internal_angle_deg=interior,max_join_angle_deg=max(joins,default=0.))


def rounded_reference_line(g, zone, width, closed=False):
    """局部二次曲线圆角，逐角核对真实边界，不能跨孔洞抄近路。

    只修改连接和田头参考线，冻结主体线不能缩短。空间不足的角保留并
    由诊断报告；宁可保留联通，也不通过扩大边界伪造一条圆滑线。
    """
    xy=np.asarray(g.coords);xy=xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-8]]
    if closed and np.linalg.norm(xy[0]-xy[-1])<1e-8:xy=xy[:-1]
    if len(xy)<3:return g
    result=[];start=0 if closed else 1;stop=len(xy) if closed else len(xy)-1
    if not closed:result.append(xy[0])
    for i in range(start,stop):
        a,v,b=xy[(i-1)%len(xy)],xy[i],xy[(i+1)%len(xy)]
        left=v-a;right=b-v;ll=np.linalg.norm(left);lr=np.linalg.norm(right)
        cut=min(ll*.42,lr*.42,width*2)
        accepted=None
        if ll>1e-8 and lr>1e-8:
            for fraction in (1.,.5,.25,.125,.0625):
                c=cut*fraction;p=v-left/ll*c;q=v+right/lr*c
                n=max(129 if angle_degrees(left,right)>80 else 17,int(math.ceil(2*c/.2)))
                t=np.unique(np.r_[0.,.0001,np.linspace(0,1,n),.9999,1.])[:,None]
                curve=(1-t)**2*p+2*(1-t)*t*v+t**2*q
                if zone.covers(LineString(curve)) and reference_geometry_angles(LineString(curve))['max_internal_angle_deg']<=20:
                    accepted=curve;break
        result.extend(accepted if accepted is not None else [v])
    if closed:result.append(result[0])
    else:result.append(xy[-1])
    answer=LineString(result)
    return answer if answer.is_simple and zone.covers(answer) else g


# 每田复用一个避障搜索空间，连接先试简单模板，失败后用栅格绕障。
# 所有栅格边仍由原矢量边界核验；点接触和几何断开不能被当成通道。
class Navigator:
    """单田惰性网格导航器；每条搜索边对真实向量边界检查，网格连通不能替代真实几何。
    
    Lazy per-field lattice; every traversed edge is checked against vector truth."""
    def __init__(self,zone,settings):
        """绑定本策略允许空间并建立延迟缓存；几何与运动模式由调用者提供不同的允许空间。"""
        self.zone=zone;self.parts=approx_polygons(zone);self.prepared=[prep(p) for p in self.parts]
        self.settings=settings;self.nodes={};self.edges={};self.cache={};self.expanded=0
        self.bounds=zone.bounds;self.task_components={};self.gentle_guides={}

    def component(self,p):
        """定位点所属真实面分量，未找到时保留不可用结果。"""
        matches=[i for i,g in enumerate(self.prepared) if g.covers(Point(p))]
        return matches[0] if len(matches)==1 else None

    def task_component(self,task):
        """核对任务所属几何分量，任务的独立碎片不因编号相同而自动连通。"""
        if task.task_id not in self.task_components:
            matches=[i for i,p in enumerate(self.prepared) if p.covers(task.reference_line)]
            self.task_components[task.task_id]=matches[0] if len(matches)==1 else None
        return self.task_components[task.task_id]

    def legal(self,xy,component):
        """检查候选线是否处于真实允许空间，用于拒绝越界/穿孔洞连接。"""
        return self.prepared[component].covers(LineString(xy))

    def gentle(self,a,b,ua,ub,width,k,original):
        """保留联通底线，优先寻找端部有切向且内部没有尖点的连接。

        长绕障用原矢量核验的栅格引导。端部先留短直线，再圆角，避免
        将“线连上了”误当成方向也顺畅。候选不足只标为需人工查看。
        """
        if ua is None or ub is None:return original
        ua=np.asarray(ua);ub=np.asarray(ub);zone=self.parts[k]
        def score(g):
            m=reference_geometry_angles(g,ua,ub)
            return max(m.values()),g.length
        choices=[]
        if original[0] is not None:
            choices.append(original)
            if score(original[0])[0]<=15:return original
            rounded=rounded_reference_line(original[0],zone,width)
            if score(rounded)[0]<=15:return rounded,'ROUNDED_REFERENCE'
            choices.append((rounded,'ROUNDED_REFERENCE'))
        # 候选增加横向余量，解决原三次曲线导数接近零形成的内部折返。
        length=math.dist(a,b)
        for depth in (width*.5,width,width*1.5,width*2,max(width,length*.35)):
            c1=np.asarray(a)+ua*depth;c2=np.asarray(b)-ub*depth
            n=min(1501,max(65,int(math.ceil((length+2*depth)/.15))))
            t=np.unique(np.r_[0.,.0001,np.linspace(0,1,n),.9999,1.])[:,None]
            xy=(1-t)**3*np.asarray(a)+3*(1-t)**2*t*c1+3*(1-t)*t**2*c2+t**3*np.asarray(b)
            g=LineString(xy)
            if g.is_simple and g.length<=max(width*10,length*3) and zone.covers(g):
                choices.append((g,'GENTLE_CUBIC_REFERENCE'))
                if score(g)[0]<=15:return g,'GENTLE_CUBIC_REFERENCE'
        # 将端部方向放进绕障引导中，而不是在折线生成后强行改朝向。
        for lead in (width,.5*width,.25*width,.125*width):
            p=np.asarray(a)+ua*lead;q=np.asarray(b)-ub*lead
            if not self.legal([a,p],k) or not self.legal([q,b],k):continue
            if self.legal([p,q],k):middle=LineString([p,q])
            else:middle=self._grid(tuple(p),tuple(q),k,self.settings.grid_m,self.bounds)
            if middle is None:continue
            raw=LineString([a,*middle.coords,b]);g=rounded_reference_line(raw,zone,width)
            if g.is_simple and g.length<=max(width*12,length*4):
                choices.append((g,'TANGENT_DETOUR_REFERENCE'))
                if score(g)[0]<=15:return g,'TANGENT_DETOUR_REFERENCE'
            # 障碍角附近原最短折线可能没有圆角余量。仅为引导搜索预留
            # 小幅空间，不缩减目标义务；最终仍在原面逐线验收。
            for margin in (width*.5,width*.25):
                guidekey=(k,width,margin)
                if guidekey not in self.gentle_guides:
                    inset=zone.buffer(-margin)
                    self.gentle_guides[guidekey]=Navigator(inset,replace(self.settings,gentle_geometry=False,assemble_reference=False)) if not inset.is_empty else None
                guide=self.gentle_guides[guidekey]
                if guide is None:continue
                middle,reason=guide.connect(p,q)
                if middle is None:continue
                raw=LineString([a,*middle.coords,b]);g=rounded_reference_line(raw,zone,width)
                if g.is_simple and zone.covers(g) and g.length<=max(width*12,length*4):
                    choices.append((g,'CLEARANCE_DETOUR_REFERENCE'))
                    if score(g)[0]<=15:return g,'CLEARANCE_DETOUR_REFERENCE'
        return min(choices,key=lambda item:score(item[0])) if choices else original

    def _grid(self,a,b,k,h,window):
        """按需要创建当前分量的有限导航网格，节点预算不是物理可达性证明。"""
        xmin,ymin,xmax,ymax=window
        def xy(n): return (self.bounds[0]+n[0]*h,self.bounds[1]+n[1]*h)
        def valid(n):
            key=(k,h,n)
            if key not in self.nodes:
                p=xy(n);self.nodes[key]=self.prepared[k].contains(Point(p))
            # Window is search-local: do not cache its rejection.
            p=xy(n)
            return xmin<=p[0]<=xmax and ymin<=p[1]<=ymax and self.nodes[key]
        def anchors(p):
            ix=round((p[0]-self.bounds[0])/h);iy=round((p[1]-self.bounds[1])/h)
            ns=sorted(((ix+dx,iy+dy) for dx in range(-2,3) for dy in range(-2,3)),key=lambda n:math.dist(p,xy(n)))
            return [n for n in ns if valid(n) and self.legal([p,xy(n)],k)][:8]
        # Node validity is independent of the local window.
        start=anchors(a);ends=set(anchors(b))
        if not start or not ends: return None
        pq=[];cost={};parent={}
        for n in start:
            cost[n]=math.dist(a,xy(n));parent[n]=None
            heapq.heappush(pq,(cost[n]+math.dist(xy(n),b),cost[n],n))
        count=0;goal=None
        while pq and count<self.settings.max_grid_nodes:
            _,d,n=heapq.heappop(pq)
            if d!=cost[n]: continue
            if n in ends: goal=n;break
            count+=1
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                m=(n[0]+dx,n[1]+dy)
                if not valid(m): continue
                # Both side cells and exact edge prohibit diagonal corner cutting.
                if dx and dy and (not valid((n[0]+dx,n[1])) or not valid((n[0],n[1]+dy))): continue
                ek=(k,h,min(n,m),max(n,m))
                if ek not in self.edges: self.edges[ek]=self.legal([xy(n),xy(m)],k)
                if not self.edges[ek]: continue
                nd=d+h*math.hypot(dx,dy)
                if nd<cost.get(m,math.inf):
                    cost[m]=nd;parent[m]=n;heapq.heappush(pq,(nd+math.dist(xy(m),b),nd,m))
        self.expanded+=count
        if goal is None:return None
        path=[];n=goal
        while n is not None:path.append(xy(n));n=parent[n]
        path=[a]+list(reversed(path))+[b]
        # Linear-time greedy visibility shortening, no all-pairs graph.
        result=[path[0]];i=0
        while i<len(path)-1:
            j=i+1
            while j+1<len(path) and self.legal([path[i],path[j+1]],k):j+=1
            result.append(path[j]);i=j
        return LineString(result)

    def connect(self,a,b,ua=None,ub=None,width=3.75,component_a=None,component_b=None):
        """尝试直达、几何引导和有限网格连接；返回线及状态，未找到时不画未经检查的补线。"""
        a=tuple(a);b=tuple(b)
        key=(a,b,tuple(ua) if ua is not None else None,tuple(ub) if ub is not None else None,width,component_a,component_b)
        if key in self.cache:return self.cache[key]
        ka=self.component(a) if component_a is None else component_a
        kb=self.component(b) if component_b is None else component_b
        if ka is None or kb is None:answer=(None,'ENDPOINT_OUTSIDE')
        elif ka!=kb:answer=(None,'GEOMETRICALLY_DISCONNECTED')
        elif math.dist(a,b)<1e-8:answer=(None,'COINCIDENT')
        else:
            answer=None
            if ua is not None and ub is not None and math.dist(a,b)<=8*width:
                for depth in (width*.5,width,width*1.5):
                    c1=np.array(a)+np.array(ua)*depth;c2=np.array(b)-np.array(ub)*depth
                    n=min(129,max(25,int(math.ceil((math.dist(a,b)+2*depth)/.75))))
                    t=np.linspace(0,1,n)
                    tip=min(.001,depth/max(math.dist(a,b),depth)*.01)
                    t=np.unique(np.r_[0,tip,t,1-tip,1])[:,None]
                    xy=(1-t)**3*np.array(a)+3*(1-t)**2*t*c1+3*(1-t)*t**2*c2+t**3*np.array(b)
                    g=LineString(xy)
                    direction_ok=(angle_degrees(xy[1]-xy[0],ua)<=35 and angle_degrees(xy[-1]-xy[-2],ub)<=35)
                    if direction_ok and g.is_simple and g.length<=max(width*4,math.dist(a,b)*3) and self.prepared[ka].covers(g):
                        answer=(g,'SMOOTH_TEMPLATE');break
            if answer is None and self.legal([a,b],ka):answer=(LineString([a,b]),'DIRECT_FOLD')
            if answer is None:
                g=self._grid(a,b,ka,self.settings.grid_m,self.bounds)
                if g is None:
                    pad=max(10.,math.dist(a,b)*.25)
                    window=(max(self.bounds[0],min(a[0],b[0])-pad),max(self.bounds[1],min(a[1],b[1])-pad),min(self.bounds[2],max(a[0],b[0])+pad),min(self.bounds[3],max(a[1],b[1])+pad))
                    g=self._grid(a,b,ka,self.settings.fine_grid_m,window)
                answer=(g,'GRID_DETOUR' if g is not None else 'SEARCH_LIMITED')
        if self.settings.gentle_geometry and ka is not None and ka==kb:
            answer=self.gentle(a,b,ua,ub,width,ka,answer)
        self.cache[key]=answer
        # Reference motion is reversible: reuse the same geometry, not a new asymmetric search.
        reversekey=(b,a,tuple(-np.asarray(ub)) if ub is not None else None,
                    tuple(-np.asarray(ua)) if ua is not None else None,width,component_b,component_a)
        g,method=answer
        self.cache[reversekey]=(LineString(list(reversed(g.coords))) if g is not None else None,method)
        return answer


def oriented(task,reverse):
    """按选中遍历方向读取任务端点；几何反向不自动代表实车倒挡。"""
    xy=list(task.reference_line.coords)
    return list(reversed(xy)) if reverse else xy


def vector(xy,first):
    """返回线在指定端点的方向向量，供接头转角检查。"""
    a,b=(xy[0],xy[1]) if first else (xy[-2],xy[-1])
    d=np.array(b)-np.array(a);return tuple(d/np.linalg.norm(d))


def angle_degrees(a,b):
    """将两方向向量夹角转换为度，供驾驶风格诊断，不作为转弯半径认证。"""
    a=np.asarray(a);b=np.asarray(b);d=np.linalg.norm(a)*np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a,b)/d,-1,1)))) if d>1e-12 else 0.


# 参考条带只比较蛇形、隔行和两种起始方向，不做全排列搜索。
# 同一条带片段仍保留原编号，不能因排序重复派工或漏掉孔洞两侧片段。
def candidates(tasks):
    """提出有限往复排序，行碎片保持独立任务且不全排列枚举。
    
    Row fragments remain separate tasks. Four fixed choices, never permutations."""
    byrow={}
    for t in tasks:byrow.setdefault(t.row_index,[]).append(t)
    rows=sorted(byrow)
    u=np.array([math.cos(tasks[0].heading_rad),math.sin(tasks[0].heading_rad)])
    backwards={t.task_id:np.dot(np.asarray(t.reference_line.coords[-1])-t.reference_line.coords[0],u)<0 for t in tasks}
    for skip in (1,2):
        order=rows if skip==1 else rows[::2]+rows[1::2]
        for first_reverse in (False,True):
            result=[]
            for i,row in enumerate(order):
                rev=bool(first_reverse^(i%2==1))
                ordered=sorted(byrow[row],key=lambda t:(np.dot(t.reference_line.centroid.coords[0],u),t.task_id),reverse=rev)
                result.extend((t,bool(rev^backwards[t.task_id])) for t in ordered)
            yield result


def execution_blocks(tasks,width,zone):
    """只将相邻行的一对一重叠组织为执行链，绕孔洞的分支不强制塞进同一链。
    
    Only one-to-one overlaps in neighbouring rows form execution chains.
    
    A row splitting around a hole is a branch; do not force both sides into one
    alternating snake. These are scheduling groups, never new work regions."""
    axis=np.array([math.cos(tasks[0].heading_rad),math.sin(tasks[0].heading_rad)])
    byrow={};intervals={};parent={t.task_id:t.task_id for t in tasks}
    for t in tasks:
        byrow.setdefault(t.row_index,[]).append(t)
        xs=[float(np.dot(p,axis)) for p in t.reference_line.coords];intervals[t.task_id]=(min(xs),max(xs))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    rows=sorted(byrow)
    for a,b in zip(rows,rows[1:]):
        links=[];degree=Counter()
        for left in byrow[a]:
            for right in byrow[b]:
                lo=max(intervals[left.task_id][0],intervals[right.task_id][0]);hi=min(intervals[left.task_id][1],intervals[right.task_id][1])
                if hi-lo<=1e-6:continue
                # Exclude gaps and bridges across a hole even if projections overlap.
                mid=LineString([left.reference_line.centroid.coords[0],right.reference_line.centroid.coords[0]])
                if mid.length>3*width or not zone.covers(mid):continue
                links.append((left,right));degree[left.task_id]+=1;degree[right.task_id]+=1
        for left,right in links:
            if degree[left.task_id]==degree[right.task_id]==1:parent[root(right.task_id)]=root(left.task_id)
    groups={}
    for t in tasks:groups.setdefault(root(t.task_id),[]).append(t)
    return sorted(groups.values(),key=lambda ts:(min(intervals[t.task_id][0] for t in ts),min(t.row_index for t in ts),min(t.task_id for t in ts)))


def region_sequences(tasks,nav,width):
    """生成有限分区执行顺序候选，保留任务归属与自然不连通分量。"""
    yield from candidates(tasks)
    blocks=execution_blocks(tasks,width,nav.zone)
    if len(blocks)<=1:return
    local=[list(candidates(block)) for block in blocks]
    for style in range(4):
        sequence=[]
        for variants in local:
            q=variants[style]
            reverse=[(t,not rev) for t,rev in reversed(q)]
            if sequence:
                options=[]
                for directed in (q,reverse):
                    g,method=pair(nav,sequence[-1],directed[0],width)
                    options.append(((int(g is None and method!='COINCIDENT'),g.length if g is not None else 0.),directed))
                q=min(options,key=lambda x:x[0])[1]
            sequence.extend(q)
        yield sequence


def pair(nav,left,right,width):
    """对一对任务读取真实遍历端点并请求连接，连接状态不能由任务邻接代替。"""
    if isinstance(nav,RefinedNavigator):
        a=nav.work_motion(*left);b=nav.work_motion(*right)
        motion,meta=nav.connect_poses(io._pose_at(a,True),io._pose_at(b,False))
        return (LineString(motion.points[:,:2]) if motion is not None else None),meta['method']
    a=oriented(*left);b=oriented(*right)
    ka=nav.task_component(left[0]);kb=nav.task_component(right[0])
    if ka is None or kb is None:return None,'TASK_COMPONENT_AMBIGUOUS'
    return nav.connect(a[-1],b[0],vector(a,False),vector(b,True),width,ka,kb)


def sequence_cost(nav,seq,width):
    """累计当前候选顺序的参考连接代价及失败，用于有界择优，仍需最终完整性核验。"""
    failed=0;length=0.
    for a,b in zip(seq,seq[1:]):
        if isinstance(nav,RefinedNavigator):
            left=io._pose_at(nav.work_motion(*a),True);right=io._pose_at(nav.work_motion(*b),False)
            m,meta=nav.connect_poses(left,right)
            failed+=m is None and meta['method']!='COINCIDENT'
            length+=meta.get('seconds',0.)
            continue
        g,status=pair(nav,a,b,width)
        failed+=g is None and status!='COINCIDENT'
        length+=g.length if g is not None else 0.
        if nav.settings.gentle_geometry:
            left=oriented(*a);right=oriented(*b)
            angle=(max(reference_geometry_angles(g,vector(left,False),vector(right,True)).values())
                   if g is not None else angle_degrees(vector(left,False),vector(right,True)))
            # 联通始终第一；同为联通时，避免为了少走几米选用急折返。
            length+=width*50*int(angle>35)+width*100*int(angle>90)
    return failed,length


def assemble_geometry_reference(rows,scene,nav,width):
    """主体之后接入每条田头参考环；输出唯一有序完整参考行程。

    原主体/独立田头图层保持可追溯，新增完整记录单独导出。田头圆角
    只改变参考线，不宣称覆盖全部田头。缺田门则自由起终，不编造入口。
    """
    result=[dict(r,phase='BODY') for r in sorted(rows,key=lambda r:r['sequence']) if r['kind']!='HEADLAND']
    failures=[];heads=[r for r in rows if r['kind']=='HEADLAND'];pending=list(heads)
    if not result:return [],dict(full_reference_connected=False,full_reference_failures=[],full_reference_status='NO_BODY_TASKS')
    previous=result[-1];component=max(r['component'] for r in result)
    def head_choices(row):
        g=rounded_reference_line(row['geometry'],scene.target,width,closed=True) if nav.settings.gentle_geometry else row['geometry']
        xy=list(g.coords)[:-1]
        p=previous['geometry'].coords[-1]
        # 避免全环每个密集采样点都求解；近端点加均匀锚点足够组织参考顺序。
        nearest=sorted(range(len(xy)),key=lambda i:math.dist(p,xy[i]))[:3]
        anchors=sorted(set(nearest+[int(i) for i in np.linspace(0,len(xy)-1,min(8,len(xy)))]))
        for index in anchors:
            rotated=xy[index:]+xy[:index]+[xy[index]]
            for directed in (rotated,list(reversed(rotated))):yield LineString(directed)
    while pending:
        p=previous['geometry'];ua=vector(list(p.coords),False)
        # 按距离选最近环，然后在该环挑方向/入点；不做全排列搜索。
        row=min(pending,key=lambda r:(r['geometry'].distance(Point(p.coords[-1])),r['task_id']))
        choices=[]
        for g in head_choices(row):
            link,method=nav.connect(p.coords[-1],g.coords[0],ua,vector(list(g.coords),True),width)
            angle=max(reference_geometry_angles(link,ua,vector(list(g.coords),True)).values()) if link is not None else 0. if method=='COINCIDENT' else 180.
            choices.append(((link is None and method!='COINCIDENT',angle>35,angle>90,link.length if link is not None else 0.),g,link,method))
        _,g,link,method=min(choices,key=lambda x:x[0])
        if link is None and method!='COINCIDENT':
            failures.append(dict(from_task=previous['task_id'],to_task=row['task_id'],reason=method));component+=1
        elif link is not None:
            result.append(dict(kind='CONNECTION',region_id='',task_id='',from_task=previous['task_id'],to_task=row['task_id'],component=component,method=method,direction='REFERENCE_FORWARD',geometry=link,phase='HEADLAND'))
        current=dict(row,geometry=g,component=component,phase='HEADLAND',method='ORDERED_ROUNDED_REFERENCE' if nav.settings.gentle_geometry else 'ORDERED_OFFSET_REFERENCE')
        result.append(current);previous=current;pending.remove(row)
    # 显式位姿仅连接位置；朝向是几何参考的切向提示，不冒充实车位姿运动。
    for name,at_start in [('start',True),('end',False)]:
        pose=getattr(scene,name,None)
        if pose is None:continue
        terminal=(pose.x,pose.y);u=(math.cos(pose.yaw),math.sin(pose.yaw))
        work=result[0 if at_start else -1];xy=list(work['geometry'].coords)
        a,b=(terminal,xy[0]) if at_start else (xy[-1],terminal)
        ua,ub=(u,vector(xy,True)) if at_start else (vector(xy,False),u)
        g,method=nav.connect(a,b,ua,ub,width)
        if g is None and method!='COINCIDENT':failures.append(dict(from_task='ENTRY' if at_start else work['task_id'],to_task=work['task_id'] if at_start else 'EXIT',reason=method))
        elif g is not None:
            connection=dict(kind='CONNECTION',region_id='',task_id='',from_task='ENTRY' if at_start else work['task_id'],to_task=work['task_id'] if at_start else 'EXIT',component=work['component'],method=method,direction='REFERENCE_FORWARD',geometry=g,phase='ENTRY' if at_start else 'EXIT')
            result.insert(0,connection) if at_start else result.append(connection)
    joins=[];internal=[]
    for i,r in enumerate(result):
        r['sequence']=i+1
        internal.append(reference_geometry_angles(r['geometry'],closed=r['kind']=='HEADLAND')['max_internal_angle_deg'])
        if i and r['component']==result[i-1]['component']:
            a=list(result[i-1]['geometry'].coords);b=list(r['geometry'].coords)
            joins.append(angle_degrees(vector(a,False),vector(b,True)))
    connected=len({r['component'] for r in result})==1 and not failures
    meta=dict(full_reference_connected=connected,full_reference_status='COMPLETE_CONNECTED' if connected else 'PARTIAL_CONNECTED',full_reference_failures=failures,full_reference_headland_count=len(heads),full_reference_component_count=len({r['component'] for r in result}),full_reference_internal_over35=sum(x>35 for x in internal),full_reference_internal_over90=sum(x>90 for x in internal),full_reference_joins_over35=sum(x>35 for x in joins),full_reference_joins_over90=sum(x>90 for x in joins),full_reference_max_internal_angle_deg=max(internal,default=0.),full_reference_max_join_angle_deg=max(joins,default=0.),full_reference_style_status='PASS' if not any(x>35 for x in internal+joins) else 'REVIEW')
    return [{**{k:v for k,v in r.items() if k!='geometry'},'geometry_wkt':r['geometry'].wkt} for r in result],meta


def style_metrics(rows,taskbook,width):
    """计算主体连接的驾驶风格诊断，范围由记录集合决定；完整行程还须单独检查接头与内部角度。"""
    work={r['task_id']:r for r in rows if r['kind']=='WORK'}
    sharp=0;bad_smooth=0;same_row=0;long_returns=0;internal=0;heading_gaps=0
    def yaw(r,last):
        if r.get('motion_json'):
            p=json.loads(r['motion_json']);return float(p[-1 if last else 0][2])
        xy=list(r['geometry'].coords)
        d=np.asarray(xy[-1])-xy[-2] if last else np.asarray(xy[1])-xy[0]
        return math.atan2(d[1],d[0])
    ordered=sorted((r for r in rows if r.get('sequence',0)>0),key=lambda r:r['sequence'])
    for a,b in zip(ordered,ordered[1:]):
        if a.get('component',1)==b.get('component',1):
            heading_gaps+=abs(wrap(yaw(b,False)-yaw(a,True)))>0.025
    for r in rows:
        if r['kind']!='CONNECTION':continue
        g=r['geometry'];xy=np.asarray(g.coords)
        if r.get('motion_json'):
            p=np.asarray(json.loads(r['motion_json']))
            moving=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)>1e-7
            internal+=int(np.any(moving&(np.abs((np.diff(p[:,2])+math.pi)%(2*math.pi)-math.pi)>math.radians(35))))
            r['start_angle_deg']=0.;r['end_angle_deg']=0.
        else:
            d=np.diff(xy,axis=0);n=np.linalg.norm(d,axis=1);d=d[n>1e-9]/n[n>1e-9,None]
            if len(d)>1:internal+=int(np.any(np.sum(d[:-1]*d[1:],axis=1)<math.cos(math.radians(35))))
        if r['from_task'] not in work or r['to_task'] not in work:continue
        a=work[r['from_task']]['geometry'];b=work[r['to_task']]['geometry']
        start=abs(math.degrees(wrap(yaw(r,False)-yaw(work[r['from_task']],True)))) if r.get('motion_json') else angle_degrees(np.asarray(a.coords[-1])-a.coords[-2],np.asarray(xy[1])-xy[0])
        end=abs(math.degrees(wrap(yaw(work[r['to_task']],False)-yaw(r,True)))) if r.get('motion_json') else angle_degrees(np.asarray(xy[-1])-xy[-2],np.asarray(b.coords[1])-b.coords[0])
        r['start_angle_deg']=start;r['end_angle_deg']=end
        sharp+=max(start,end)>90
        bad_smooth+=r['method']=='SMOOTH_TEMPLATE' and max(start,end)>35
        if r['from_task'] not in taskbook or r['to_task'] not in taskbook:continue
        ta=taskbook[r['from_task']];tb=taskbook[r['to_task']]
        same_row+=ta.region_id==tb.region_id and ta.row_index==tb.row_index and ('GUIDED' in r['method'] or r['method']=='GRID_DETOUR')
        va=np.asarray(a.coords[-1])-a.coords[0];vb=np.asarray(b.coords[-1])-b.coords[0]
        long_returns+=ta.region_id==tb.region_id and ta.row_index!=tb.row_index and angle_degrees(va,vb)<30 and g.length>max(8*width,min(a.length,b.length)*.5)
    status='FAIL' if bad_smooth else 'REVIEW' if sharp or long_returns or same_row>=2 or internal or heading_gaps else 'PASS'
    return dict(reference_style_status=status,sharp_connection_count=int(sharp),invalid_smooth_count=int(bad_smooth),same_row_hole_detour_count=int(same_row),long_same_direction_return_count=int(long_returns),internal_corner_count=int(internal),heading_discontinuity_count=int(heading_gaps))


def reference_motion_metrics(motion,scene):
    """从明确运动采样统计运动约束及停顿/换挡事件，不单独宣称整车包络或实车安全。
    
    Kinematics and explicit stop/gear events; no vehicle-envelope claim."""
    from validator import kinematic_issues
    p=motion.points;ds=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
    dyaw=(np.diff(p[:,2])+math.pi)%(2*math.pi)-math.pi
    moving=ds>1e-7
    k=np.divide(dyaw,ds*p[:-1,3],out=np.zeros_like(ds),where=moving)
    maximum=float(np.abs(k).max(initial=0.))
    ix=np.flatnonzero(moving);rate=0.
    if len(ix)>1:
        distance=np.diff((np.cumsum(ds)-ds/2)[ix])
        same=(p[:-1,3][ix][1:]==p[:-1,3][ix][:-1])&(distance>1e-7)
        rates=np.divide(np.abs(np.diff(k[ix])),distance,out=np.zeros_like(distance),where=same)
        rate=float(rates.max(initial=0.))
    issues=kinematic_issues(motion,scene)
    # A 0.01% allowance covers chord discretization (0.5 mm at a 5 m
    # radius). Native candidates get a separate conservative solver margin;
    # this check does not spend the legacy 15% tolerance on a tighter turn.
    if maximum>(1.0001/scene.vehicle.min_turn_radius_m):issues.append('REFERENCE_RADIUS_LIMIT')
    if scene.settings.check_curvature_rate and rate>scene.vehicle.max_curvature_rate*1.01+.001:
        issues.append('REFERENCE_CURVATURE_RATE')
    legs,shifts,reverse_m,events,steer_s=io._gear_and_steering(motion,scene,io.RouteSettings())
    if not math.isfinite(steer_s):issues.append('GEAR_OR_REVERSE_LIMIT')
    seconds=float(np.sum(ds/np.where(p[:-1,3]<0,scene.vehicle.reverse_speed_mps,scene.vehicle.turn_speed_mps)))
    seconds+=shifts*scene.vehicle.gear_change_seconds+(steer_s if math.isfinite(steer_s) else 0.)
    return dict(issues=sorted(set(issues)),max_curvature_inv_m=maximum,
        maximum_curvature_rate=rate,minimum_radius_m=1/maximum if maximum>1e-10 else None,
        reverse_legs=legs,gear_shifts=shifts,reverse_m=reverse_m,events=events,seconds=seconds)


def continuous_reference_u(a,b,scene):
    """用解析渐变曲率和圆弧生成U形运动，入出曲率为零，按车辆最小半径与横向位移求解。
    
    Exact Fresnel spiral/arc/spiral U, with zero entry/exit curvature.
    
    Solve the lateral displacement for an actual radius >= vehicle minimum.
    Longitudinal mismatch is taken up by forward tangent straights. Unlike a
    cubic interpolation, both curvature and its spatial derivative are known.
    This is an explicitly geometric CC family, not a claimed native F2C turn."""
    if abs(abs(wrap(b.yaw-a.yaw))-math.pi)>1e-6:return None
    from scipy.special import fresnel
    from scipy.optimize import brentq
    c,s=math.cos(a.yaw),math.sin(a.yaw)
    dx,dy=b.x-a.x,b.y-a.y;longitudinal=c*dx+s*dy;lateral=-s*dx+c*dy
    sigma=scene.vehicle.max_curvature_rate
    low=max(scene.vehicle.min_turn_radius_m,math.sqrt(1/(math.pi*sigma))*(1+1e-8))
    def ramp(radius,dist):
        sy,cx=fresnel(np.asarray(dist)*math.sqrt(sigma/math.pi))
        return math.sqrt(math.pi/sigma)*cx,math.sqrt(math.pi/sigma)*sy
    def span(radius):
        length=1/(radius*sigma);x,y=ramp(radius,length)
        angle=1/(2*radius*radius*sigma)
        return float(2*y+2*radius*math.cos(angle))
    distance=abs(lateral)
    if distance<span(low)-1e-7:return None
    high=max(low,distance/2)
    radius=low if abs(span(low)-distance)<1e-7 else brentq(lambda r:span(r)-distance,low,high,xtol=1e-12)
    length=1/(radius*sigma);angle=1/(2*radius*radius*sigma);sign=1 if lateral>=0 else -1
    step=scene.settings.sampling_step_m
    n=max(2,math.ceil(length/step));arc_n=max(2,math.ceil((math.pi-2*angle)*radius/step))
    if 2*n+arc_n+5>scene.settings.max_motion_samples:return None
    t=np.linspace(0,length,n+1);x,y=ramp(radius,t);xr,yr=float(x[-1]),float(y[-1])
    phi=np.linspace(angle,math.pi-angle,arc_n+1)
    first=np.column_stack((x,y,.5*sigma*t*t))
    arc=np.column_stack((xr+radius*(np.sin(phi)-math.sin(angle)),yr+radius*(math.cos(angle)-np.cos(phi)),phi))
    end_t=length-t;end_x,end_y=ramp(radius,end_t)
    last=np.column_stack((end_x,distance-end_y,math.pi-.5*sigma*end_t*end_t))
    path=np.vstack((first,arc[1:],last[1:]));lead=max(0.,longitudinal)
    path[:,0]+=lead;path[:,1]*=sign;path[:,2]*=sign
    if lead>1e-8:path=np.vstack(([0.,0.,0.],path))
    if longitudinal<0:path=np.vstack((path,[longitudinal,lateral,sign*math.pi]))
    xy=path[:,:2].copy();path[:,0]=a.x+c*xy[:,0]-s*xy[:,1];path[:,1]=a.y+s*xy[:,0]+c*xy[:,1];path[:,2]+=a.yaw
    return Motion(np.column_stack((path,np.ones(len(path)))),'turn')


class ReferenceBudgetConnector(Connector):
    """在每次原生回退请求之前执行共享查询上限，避免嵌套回退绕过预算。
    
    Enforce the shared query cap before every native fallback request."""
    def __init__(self,scene,settings,deadline,navigator):
        """绑定共享查询预算与现有原生连接器，不创建独立无限预算。"""
        super().__init__(scene,settings,deadline);self.navigator=navigator

    def native(self,*args,**kwargs):
        """原生求解前消耗共享查询额度，耗尽后保留未验证原因。"""
        self.navigator.consume_query()
        return super().native(*args,**kwargs)


class RefinedNavigator(Navigator):
    """参考位姿细化求解器；网格只提供引导，连接使用真实travel并按配置核验，禁止虚构田外扩张。
    
    Native pose-to-pose refinement; the lattice supplies guides only.
    
    Searches use travel, not a fabricated expansion of the true geometry.
    Work variants preserve frozen implement sweeps. No unverified fold or
    zero-distance heading jump is emitted when bounded refinement fails."""
    def __init__(self,scene,settings):
        """以Scene.travel初始化运动细化导航与缓存，并保存当前作业阶段的空间状态。"""
        super().__init__(scene.travel,settings)
        self.scene=scene;self.variants={};self.pose_cache={};self.motion_queries=0
        # Native HC samples may slightly exceed their requested curvature.
        # Reserve 0.6% radius in the solver, then validate against the actual
        # vehicle limit. Frozen work geometry and coverage are unaffected.
        self.solver_scene=replace(scene,vehicle=replace(scene.vehicle,
            min_turn_radius_m=scene.vehicle.min_turn_radius_m*1.006))
        self.native=None;self.parked=None;self.rejections=Counter()
        self.motion_space=prep(scene.travel.buffer(1e-5))
        self.phase='SELECTION';self.phase_queries=Counter()
        self.model_checker=MotionChecker(scene)
        self.crop_free=None

    def record_work(self,motion):
        """将真实已规划开启机具扫掠加入已作业账本，未来作业不提前登记。"""
        if not self.settings.operational_refinement or not motion.implement_on:return
        # Dilation distributes over unions. Buffer the new sweep once rather
        # than rebuilding every tiny rounded corner of the complete ledger
        # after every task. Inscribed diamond arcs stay within the epsilon.
        sweep=work_coverage(motion,self.scene)
        ready=self.crop_free if self.crop_free is not None else self.scene.travel.difference(self.scene.target).buffer(self.scene.settings.geometry_epsilon_m,quad_segs=1)
        self.crop_free=ready.union(sweep.buffer(self.scene.settings.geometry_epsilon_m,quad_segs=1))
        shapely.prepare(self.crop_free)
        # A failed crop-constrained search may succeed after real work has
        # enlarged access. Neither failure nor a future nominal pass is proof.
        self.pose_cache={k:v for k,v in self.pose_cache.items() if v[0] is not None or v[1]['method'] in ('COINCIDENT','GEOMETRICALLY_DISCONNECTED','ENDPOINT_OUTSIDE')}

    def access_allowed(self,motion):
        """按当前已作业和合法田外空间检查候选通行，不以全田目标默认允许关机具穿越。"""
        if self.crop_free is None:return True
        if self.phase=='TERMINAL' and np.all(motion.points[:,3]>0):return True
        physical,parts=self.model_checker.physical(motion)
        return not physical and parts is not None and np.all(shapely.covers(self.crop_free,parts))

    def endpoint_status(self,a,b):
        """检查端点位姿及空间状态，区分包络不合法与仅预算未解。"""
        if not self.settings.operational_refinement:return None
        for pose in (a,b):
            p=[pose.x,pose.y,pose.yaw,1]
            physical,parts=self.model_checker.physical(Motion(np.array([p,p]),'turn'))
            if physical:return 'ENDPOINT_MODEL_UNSAFE'
            if self.crop_free is not None and self.phase!='TERMINAL' and not np.all(shapely.covers(self.crop_free,parts)):
                return 'CROP_ENDPOINT_UNAVAILABLE'
        return None

    def cutting_entry(self,a,work):
        """为后置机具实际覆盖构造明确开机具入场段，禁止侵占其他未来主体义务。
        
        Lower the implement on a real forward straight before a work port.
        
        The lifted prefix must fit previously cut space. Only the explicitly
        emitted cutting tail can enter the current task; its implement sweep
        must remain in that task or prior work. Future tasks never grant lifted
        access. Work entry adds coverage without moving a frozen obligation."""
        if not self.settings.operational_refinement or self.crop_free is None:return None
        b=io._pose_at(work,False);rects=self.scene.vehicle.rectangles()
        distance=float(np.max(rects[0][:,0])-np.min(rects[1][:,0])+self.scene.vehicle.safety_margin_m)
        goal_sweep=work_coverage(work,self.scene)
        allowed=self.crop_free.union(goal_sweep).union(getattr(self,'entry_work_space',GeometryCollection())).buffer(self.scene.settings.geometry_epsilon_m)
        for factor in (1.,1.5,2.,2.5,3.):
            length=distance*factor
            approach=Pose(b.x-length*math.cos(b.yaw),b.y-length*math.sin(b.yaw),b.yaw)
            tail=Motion(np.array([[approach.x,approach.y,approach.yaw,1],[b.x,b.y,b.yaw,1]]),'straight_work','',True)
            if self.model_checker.physical(tail)[0] or not allowed.covers(work_coverage(tail,self.scene)):continue
            prefix,meta=self.connect_poses(a,approach)
            if prefix is not None or meta['method']=='COINCIDENT':return prefix,tail,meta
        return None

    def cutting_exit(self,work):
        """为后置机具完成当前覆盖构造明确开机具离场段，保持当前与未来任务的作业边界。
        
        Keep cutting across a headland gap until lifting is crop-safe.
        
        An emitted exit may work the non-frozen headland, never another future
        body obligation. Both the whole rigid rig and the final lifted pose
        must be legal; otherwise no exit extension is claimed."""
        a=io._pose_at(work,True)
        if self.endpoint_status(a,a) is None:return None
        rects=self.scene.vehicle.rectangles()
        step=float(np.max(rects[0][:,0])-np.min(rects[1][:,0])+self.scene.vehicle.safety_margin_m)
        allowed=self.crop_free.union(work_coverage(work,self.scene)).union(self.entry_work_space).buffer(self.scene.settings.geometry_epsilon_m)
        for factor in (.5,1.,1.5,2.,2.5,3.):
            b=Pose(a.x+step*factor*math.cos(a.yaw),a.y+step*factor*math.sin(a.yaw),a.yaw)
            m=Motion(np.array([[a.x,a.y,a.yaw,1],[b.x,b.y,b.yaw,1]]),'straight_work','',True)
            if self.model_checker.physical(m)[0] or not allowed.covers(work_coverage(m,self.scene)):continue
            p=[b.x,b.y,b.yaw,1];bad,parts=self.model_checker.physical(Motion(np.array([p,p]),'turn'))
            if not bad and np.all(shapely.covers(self.crop_free.union(work_coverage(m,self.scene)).buffer(self.scene.settings.geometry_epsilon_m),parts)):return m
        return None

    @property
    def query_limit(self):
        """读取当前细化阶段可用的原生查询上限，不把已耗额度重新置零。"""
        if not self.settings.operational_refinement:return self.settings.max_motion_queries
        fraction={'SELECTION':.35,'TERMINAL':.55,'ASSEMBLY':.95,'EXIT':1.}[self.phase]
        return int(self.settings.max_motion_queries*fraction)

    def set_phase(self,phase):
        """切换田头/主体等阶段上下文，使预算及允许作业空间与实际顺序一致。"""
        if phase not in ('SELECTION','ASSEMBLY','TERMINAL','EXIT'):raise ValueError('INVALID_QUERY_PHASE')
        self.phase=phase
        if self.settings.operational_refinement:
            if phase!='SELECTION' and self.crop_free is None:
                self.crop_free=self.scene.travel.difference(self.scene.target).buffer(self.scene.settings.geometry_epsilon_m,quad_segs=1)
                shapely.prepare(self.crop_free)
            # Search failures are conditional evidence, not unreachable-space
            # proofs. Retain successful templates but retry failed queries
            # after the next reserved budget becomes available.
            self.pose_cache={k:v for k,v in self.pose_cache.items() if v[0] is not None or v[1]['method'] in ('COINCIDENT','GEOMETRICALLY_DISCONNECTED','ENDPOINT_OUTSIDE')}

    def consume_query(self):
        """在真正请求原生求解前登记消耗，嵌套回退同样受共享上限约束。"""
        if self.motion_queries>=self.query_limit:
            raise ValueError('MOTION_QUERY_BUDGET_EXHAUSTED')
        self.motion_queries+=1
        self.phase_queries[self.phase]+=1

    def work_motion(self,task,reverse):
        """按选中方向构造车辆作业线并保持冻结机具扫掠，不直接复用反向前的连接端点。"""
        key=(task.task_id,bool(reverse))
        if key not in self.variants:
            m=io._work_motion(task,self.scene,reverse)
            sweep=io.Validator(self.scene).work_sweep(m)
            if sweep.symmetric_difference(task.frozen_sweep).area>io._sweep_tolerance(task.frozen_sweep.area):
                raise ValueError('WORK_SWEEP_MISMATCH:'+task.task_id)
            self.variants[key]=m
        return self.variants[key]

    def _candidate(self,motion,a,b,method):
        """核验一条细化运动候选的状态与约束，未通过的运动不能作为参考成功段。"""
        p=motion.points
        if (np.linalg.norm(p[0,:2]-[a.x,a.y])>1e-5 or
                np.linalg.norm(p[-1,:2]-[b.x,b.y])>1e-5 or
                abs(wrap(p[0,2]-a.yaw))>1e-5 or abs(wrap(p[-1,2]-b.yaw))>1e-5):
            self.rejections['ENDPOINT_MISMATCH']+=1;return None
        meta=reference_motion_metrics(motion,self.scene)
        if meta['issues']:
            self.rejections.update(meta['issues']);return None
        g=LineString(p[:,:2]);ka=self.component((a.x,a.y));kb=self.component((b.x,b.y))
        if ka is None or ka!=kb or not self.parts[ka].buffer(1e-5).covers(g):
            self.rejections['REFERENCE_OUTSIDE_TRAVEL']+=1;return None
        if self.settings.operational_refinement:
            physical,_=self.model_checker.physical(motion)
            if physical:
                self.rejections.update(physical);return None
            if not self.access_allowed(motion):
                self.rejections['UNWORKED_CROP_CROSSING']+=1;return None
        # Avoid full native windings masquerading as a comfortable local turn.
        variation=float(np.abs((np.diff(p[:,2])+math.pi)%(2*math.pi)-math.pi).sum())
        if variation>2*math.pi+.05:
            self.rejections['EXCESS_HEADING_WINDING']+=1;return None
        meta.update(method=method,heading_variation_rad=variation)
        return motion,meta

    def port_safe_order(self,sequence):
        """按端口可达及约束排序有限候选，只排序已保留全部原任务的序列。"""
        result=[]
        for task,reverse in sequence:
            current=self.work_motion(task,reverse)
            if not self.motion_space.covers(LineString(current.points[:,:2])) or (self.settings.operational_refinement and self.model_checker.physical(current)[0]):
                alternate=self.work_motion(task,not reverse)
                if self.motion_space.covers(LineString(alternate.points[:,:2])) and (not self.settings.operational_refinement or not self.model_checker.physical(alternate)[0]):reverse=not reverse
            result.append((task,reverse))
        return result

    def connect_poses(self,a,b):
        """连接真实起终位姿并保存运动、挡位和失败状态，不能用几何折线伪装位姿连续运动。"""
        key=(a,b)
        if key in self.pose_cache:
            cached=self.pose_cache[key]
            if cached[0] is None or self.access_allowed(cached[0]):return cached
        before=self.rejections.copy()
        ka=self.component((a.x,a.y));kb=self.component((b.x,b.y))
        if ka is None or kb is None:answer=(None,dict(method='ENDPOINT_OUTSIDE'))
        elif ka!=kb:answer=(None,dict(method='GEOMETRICALLY_DISCONNECTED'))
        elif math.hypot(a.x-b.x,a.y-b.y)<1e-8 and abs(wrap(a.yaw-b.yaw))<1e-5:
            answer=(None,dict(method='COINCIDENT'))
        elif (endpoint_reason:=self.endpoint_status(a,b)) is not None:
            # Every candidate shares these footprints. Rejecting an unsafe
            # endpoint before native search is exact under the current model,
            # and avoids spending seconds on a provably unavailable port.
            self.rejections[endpoint_reason]+=1
            answer=(None,dict(method=endpoint_reason))
        else:
            choices=[];direct=io._direct_motion(a,b,self.scene)
            if direct is not None:
                found=self._candidate(direct,a,b,'KINEMATIC_STRAIGHT')
                if found:choices.append(found)
            if not choices:
                cc=continuous_reference_u(a,b,self.scene)
                if cc is not None:
                    found=self._candidate(cc,a,b,'GEOMETRIC_CONTINUOUS_U')
                    if found:choices.append(found)
            if not choices and self.settings.operational_refinement:
                # A short straight retreat can put the large forward U inside
                # the real headland. It is an actual reverse leg with explicit
                # stops, never an in-place yaw change or expanded field.
                if self.scene.vehicle.allow_reverse:
                    for distance in (self.scene.vehicle.min_turn_radius_m/2,self.scene.vehicle.min_turn_radius_m,self.scene.vehicle.min_turn_radius_m*1.5):
                        retreat=Pose(a.x-distance*math.cos(a.yaw),a.y-distance*math.sin(a.yaw),a.yaw)
                        cc=continuous_reference_u(retreat,b,self.scene)
                        if cc is None:continue
                        m=Motion(np.vstack(([a.x,a.y,a.yaw,-1],cc.points)),'turn')
                        found=self._candidate(m,a,b,'RETREAT_CONTINUOUS_U')
                        if found:choices.append(found)
            if not choices and self.settings.operational_refinement and self.phase=='SELECTION':
                # Candidate ranking is provisional. Solving every edge in
                # every permutation wastes native queries and biases later
                # permutations once the cap is spent. Use a geometry/time
                # estimate here; clear this negative cache at execution and
                # require full motion, rig and crop checks before any export.
                seconds=math.hypot(a.x-b.x,a.y-b.y)/self.scene.vehicle.turn_speed_mps
                seconds+=self.scene.vehicle.min_turn_radius_m*abs(wrap(b.yaw-a.yaw))/self.scene.vehicle.turn_speed_mps
                answer=(None,dict(method='SELECTION_ESTIMATE',seconds=seconds))
                self.pose_cache[key]=answer;return answer
            if not choices and self.motion_queries>=self.query_limit:
                answer=(None,dict(method='MOTION_QUERY_BUDGET_EXHAUSTED'))
                self.pose_cache[key]=answer;return answer
            if not choices:
                if self.native is None:
                    self.native=ReferenceBudgetConnector(self.solver_scene,io.RouteSettings(),math.inf,self)
                    self.native.backend=SampleBoundedBackend(self.solver_scene)
                    for _,turner in self.native.backend.turners:turner.setUsingCache(False)
                families=[(0,False)]
                if self.scene.vehicle.allow_reverse:families.extend([(1,False),(1,True)])
                for index,backward in families:
                    if self.motion_queries>=self.query_limit:break
                    try:
                        m=self.native.native(a,b,index,backward)
                        found=self._candidate(m,a,b,('BACKWARD_START_' if backward else '')+self.native.backend.turners[index][0].upper())
                        if found:choices.append(found)
                    except (ValueError,RuntimeError,IndexError):self.rejections['NATIVE_REJECTED']+=1
            if not choices and self.scene.vehicle.allow_reverse and self.motion_queries<self.query_limit:
                # Existing parked-RS templates may steer at real gear cusps.
                # Same-gear curvature jumps still fail the exact same checks.
                if self.parked is None:self.parked=LocalTurns(self.solver_scene,math.inf)
                for radius in (self.solver_scene.vehicle.min_turn_radius_m,self.solver_scene.vehicle.min_turn_radius_m+self.scene.vehicle.working_width_m/6):
                    for backward in (False,True):
                        if self.motion_queries>=self.query_limit:break
                        self.consume_query()
                        try:
                            m=self.parked.parked_native(a,b,backward,radius)
                            found=self._candidate(m,a,b,'PARKED_REEDS_SHEPP')
                            if found:choices.append(found)
                        except (ValueError,RuntimeError,IndexError):self.rejections['PARKED_NATIVE_REJECTED']+=1
            if not choices and self.motion_queries<self.query_limit:
                # Coarse/fine A* is never exported as a drivable fold. Its
                # internal corners go back through the native pose solver.
                g=LineString([(a.x,a.y),(b.x,b.y)]) if math.hypot(a.x-b.x,a.y-b.y)>1e-8 else None
                if g is not None and not self.parts[ka].covers(g):
                    guide=self._grid((a.x,a.y),(b.x,b.y),ka,self.settings.grid_m,self.bounds)
                    if guide is None:
                        pad=max(10.,g.length*.25)
                        window=(max(self.bounds[0],min(a.x,b.x)-pad),max(self.bounds[1],min(a.y,b.y)-pad),min(self.bounds[2],max(a.x,b.x)+pad),min(self.bounds[3],max(a.y,b.y)+pad))
                        guide=self._grid((a.x,a.y),(b.x,b.y),ka,self.settings.fine_grid_m,window)
                    if guide is not None:
                        for index,backward in families:
                            if self.motion_queries>=self.query_limit:break
                            try:
                                m=self.native.native(a,b,index,backward,list(guide.coords))
                                found=self._candidate(m,a,b,'GUIDED_'+self.native.backend.turners[index][0].upper())
                                if found:choices.append(found)
                            except (ValueError,RuntimeError,IndexError):self.rejections['GUIDED_NATIVE_REJECTED']+=1
            if not choices and self.motion_queries+8<self.query_limit:
                # Reuse the strict connector's retreat/approach and portal
                # proposals under a small query budget. Passing here never
                # relaxes reference kinematics or fabricates an endpoint.
                old_deadline=self.native.deadline
                self.native.deadline=time.perf_counter()+(.5 if not self.settings.operational_refinement or self.phase=='SELECTION' else 1.0)
                try:
                    ready=self.crop_free if self.crop_free is not None and self.phase!='TERMINAL' else self.scene.travel
                    connection=self.native.connect(a,b,None,None,ready,{},transfer=True,_allow_medial=False)
                    if connection is not None and connection.motion is not None:
                        found=self._candidate(connection.motion,a,b,'REFINED_'+connection.method.upper())
                        if found:choices.append(found)
                finally:
                    self.native.deadline=old_deadline
            if choices:
                fastest=min(c[1]['seconds'] for c in choices)
                shortlist=[c for c in choices if c[1]['seconds']<=fastest*1.05+1e-8]
                answer=min(shortlist,key=lambda c:(c[1]['gear_shifts'],c[1]['reverse_m'],c[1]['seconds']))
            else:
                reason='MOTION_QUERY_BUDGET_EXHAUSTED' if self.motion_queries>=self.query_limit else 'CROP_ACCESS_CONNECTION_NOT_FOUND' if self.rejections['UNWORKED_CROP_CROSSING']>before['UNWORKED_CROP_CROSSING'] else 'MODEL_SAFE_CONNECTION_NOT_FOUND' if any(self.rejections[c]>before[c] for c in ('BODY_COLLISION_OR_BOUNDARY','IMPLEMENT_COLLISION_OR_BOUNDARY')) else 'KINEMATIC_CONNECTION_NOT_FOUND'
                answer=(None,dict(method=reason))
        self.pose_cache[key]=answer;return answer


def refined_candidates(tasks,scene):
    """在原有限排序外补充车辆尺度跳行候选，所有冻结任务仍保持唯一。
    
    Vehicle-scaled skips plus original orders; every task stays unique."""
    yield from candidates(tasks)
    byrow={}
    for t in tasks:byrow.setdefault(t.row_index,[]).append(t)
    rows=sorted(byrow);u=np.array([math.cos(tasks[0].heading_rad),math.sin(tasks[0].heading_rad)])
    # Continuous-curvature forward turns need more room than a 2R circle.
    spacing=scene.vehicle.working_width_m*(1-scene.settings.overlap_fraction)
    ramp=1/(scene.vehicle.min_turn_radius_m*scene.vehicle.max_curvature_rate)
    nominal=max(3,math.ceil((2*scene.vehicle.min_turn_radius_m+2*ramp)/spacing))
    for skip in sorted({min(k,len(rows)) for k in (3,4,nominal,nominal+1)}-{1,2}):
        order=[r for offset in range(skip) for r in rows[offset::skip]]
        for first in (False,True):
            seq=[]
            for i,row in enumerate(order):
                backwards=bool(first^(i%2==1))
                for t in sorted(byrow[row],key=lambda t:(float(np.dot(t.reference_line.centroid.coords[0],u)),t.task_id),reverse=backwards):
                    original_back=np.dot(np.asarray(t.reference_line.coords[-1])-t.reference_line.coords[0],u)<0
                    seq.append((t,bool(backwards^original_back)))
            yield seq


def evaluate_reference_operation(rows,scene):
    """按配置刚性车体/机具回放作业时序、通行和时间；未来条带不能提前供关机具通行，不替代实车验证。
    
    Replay the configured rigid rig, crop ledger, work and stop timing.
    
    Future passes never supply lifted travel access. Cutting motions follow
    the project's work policy: the tractor may enter its current cutting task
    before its rear implement. Lifted travel requires previously cut space or
    explicit travel outside target. Crop access is a model assumption, not a
    measured crop-damage guarantee."""
    checker=MotionChecker(scene);worked=GeometryCollection();body=[];heads=[];issues=[]
    outside=scene.travel.difference(scene.target);eps=scene.settings.geometry_epsilon_m
    ready=outside.buffer(eps,quad_segs=1);shapely.prepare(ready)
    t_work=t_break=0.;previous=None;previous_on=False;timeline=[];components=set()
    for row in sorted(rows,key=lambda r:r['sequence']):
        on=bool(row.get('implement_on',row['kind']=='WORK'))
        kind=row.get('motion_kind','straight_work' if row['kind']=='WORK' else 'work' if on else 'turn')
        m=Motion(np.asarray(json.loads(row['motion_json'])),kind,row.get('task_id',''),on)
        components.add(row['component']);physical,parts=checker.physical(m)
        issues.extend(dict(code=code,sequence=row['sequence'],task_id=row.get('task_id',''),source_stage='FROZEN_BODY' if row['kind']=='WORK' else row.get('phase','HEADLAND')) for code in physical)
        meta=reference_motion_metrics(m,scene)
        issues.extend(dict(code=code,sequence=row['sequence']) for code in meta['issues'])
        # Constant-yaw translation has an exact swept hull. Replacing it with
        # thousands of sampled rectangles adds false slivers and union cost.
        sweep=work_coverage(m,scene)
        if not on:
            if parts is not None and not np.all(shapely.covers(ready,parts)):
                issues.append(dict(code='UNWORKED_CROP_CROSSING',sequence=row['sequence'],task_id=row.get('task_id','')))
        if previous is not None:
            p=np.asarray(json.loads(previous['motion_json']))[-1]
            if previous['component']!=row['component'] or np.linalg.norm(p[:2]-m.points[0,:2])>1e-5 or abs(wrap(p[2]-m.points[0,2]))>1e-5:
                issues.append(dict(code='OPERATION_ITINERARY_GAP',sequence=row['sequence']))
        events=list(meta['events'])
        if on!=previous_on:
            events.append(dict(kind='IMPLEMENT_ON' if on else 'IMPLEMENT_OFF',x_m=float(m.points[0,0]),y_m=float(m.points[0,1]),yaw_rad=float(m.points[0,2]),duration_s=scene.vehicle.implement_switch_seconds,duration_source='CONFIG_NOMINAL_LAG_UNVERIFIED'))
        ds=np.linalg.norm(np.diff(m.points[:,:2],axis=0),axis=1)
        straight_motion=meta['max_curvature_inv_m']<1e-9
        forward_speed=(scene.vehicle.work_speed_mps if straight_motion else min(scene.vehicle.work_speed_mps,scene.vehicle.turn_speed_mps)) if on else scene.vehicle.turn_speed_mps
        moving_s=float(np.sum(ds/np.where(m.points[:-1,3]<0,scene.vehicle.reverse_speed_mps,forward_speed)))
        stop_s=sum(e['duration_s'] for e in events)
        if on:t_work+=moving_s
        else:t_break+=moving_s
        t_break+=stop_s
        timeline.append(dict(sequence=row['sequence'],implement_on=on,moving_seconds=moving_s,stop_seconds=stop_s,operations=events))
        if on:
            worked=worked.union(sweep)
            ready=ready.union(sweep.buffer(eps,quad_segs=1));shapely.prepare(ready)
            (body if row['kind']=='WORK' else heads).append(sweep)
        previous=row;previous_on=on
    if previous_on:
        t_break+=scene.vehicle.implement_switch_seconds
        timeline[-1]['stop_seconds']+=scene.vehicle.implement_switch_seconds
        p=np.asarray(json.loads(previous['motion_json']))[-1]
        timeline[-1]['operations'].append(dict(kind='IMPLEMENT_OFF',x_m=float(p[0]),y_m=float(p[1]),yaw_rad=float(p[2]),duration_s=scene.vehicle.implement_switch_seconds,duration_source='CONFIG_NOMINAL_LAG_UNVERIFIED'))
    target=scene.target;body_sweep=unary_union(body);head_sweep=unary_union(heads)
    gap=target.difference(worked);tol=scene.settings.coverage_tolerance_m2
    access_codes={'UNWORKED_CROP_CROSSING','WORK_OUTSIDE_CUT_CORRIDOR','OPERATION_ITINERARY_GAP'}
    rig=bool(rows) and not any(x['code'] not in access_codes for x in issues)
    crop=bool(rows) and not any(x['code'] in ('UNWORKED_CROP_CROSSING','WORK_OUTSIDE_CUT_CORRIDOR') for x in issues)
    continuous=bool(rows) and len(components)==1 and not any(x['code']=='OPERATION_ITINERARY_GAP' for x in issues)
    gates=scene.start is not None and scene.end is not None
    applied=gates and bool(rows)
    if applied:
        first=np.asarray(json.loads(rows[0]['motion_json']))[0];last=np.asarray(json.loads(rows[-1]['motion_json']))[-1]
        applied=all(np.linalg.norm(p[:2]-[pose.x,pose.y])<1e-5 and abs(wrap(p[2]-pose.yaw))<1e-5 for p,pose in [(first,scene.start),(last,scene.end)])
    return dict(operational_refinement=True,configured_rig_model_status='PASS' if rig else 'FAIL',chronological_crop_access_status='PASS' if crop else 'FAIL',operation_continuity_status='PASS' if continuous else 'FAIL',physical_vehicle_certification='NOT_EVALUATED',gate_data_status='APPLIED' if applied else 'UNCONNECTED' if gates else 'MISSING',configured_operation_passed=bool(rig and crop and continuous and applied and gap.area<=tol and not issues),headland_work_coverage_status='COVERAGE_COMPLETE' if gap.area<=tol else 'COVERAGE_GAP',target_area_m2=target.area,body_work_area_m2=body_sweep.intersection(target).area,headland_work_area_m2=head_sweep.intersection(target).area,headland_additional_area_m2=head_sweep.intersection(target).difference(body_sweep).area,target_planned_work_area_m2=worked.intersection(target).area,target_missing_m2=gap.area,target_planned_coverage_ratio=worked.intersection(target).area/target.area,operation_issues=issues,operation_timing=timeline,operation_coverage_gaps=[dict(area_m2=p.area,geometry_wkt=p.wkt) for p in approx_polygons(gap)],t_work_seconds=t_work,t_break_seconds=t_break,planning_reference_time_efficiency=t_work/(t_work+t_break) if t_work+t_break else None,operation_time_status='CONFIG_ESTIMATE_NOT_FIELD_MEASURED' if continuous else 'PARTIAL_ITINERARY_TIME_ESTIMATE')


def solve_refined_reference(job,settings):
    """组装带车辆运动细化的主体、田头及已配置起终位姿；各模型证据分开，缺真实田门仍标缺失。
    
    Assemble body, headland references and configured gate poses.
    
    Only kinematics and reference geometry are accepted here. Headland work,
    chronological crop access, continuous rig certification and field proof
    remain separate from this reference stage."""
    started=time.perf_counter();scene=approx_load_scene(job.scene_path);nav=RefinedNavigator(scene,settings)
    if settings.operational_refinement:nav.entry_work_space=scene.target.difference(unary_union([t.frozen_sweep for t in job.tasks]))
    width=scene.vehicle.working_width_m;options={};regmap={r.region_id:r for r in job.regions}
    for reg in job.regions:
        ts=[t for t in job.tasks if t.region_id==reg.region_id]
        if not ts:continue
        sequences=list(refined_candidates(ts,scene))
        blocks=execution_blocks(ts,width,scene.target)
        if len(blocks)>1:sequences.extend(list(region_sequences(ts,nav,width))[4:])
        safe_sequences=[];seen=set()
        for q in sequences:
            directed=nav.port_safe_order(q)
            signature=tuple((t.task_id,r) for t,r in directed)
            if signature not in seen:seen.add(signature);safe_sequences.append(directed)
        options[reg.region_id]=sorted(((sequence_cost(nav,q,width),q) for q in safe_sequences),key=lambda v:v[0])
    seq=[];todo=set(options);region_order=[]
    while todo:
        if not seq:
            rid=min(todo,key=lambda r:(regmap[r].geometry.distance(Point(scene.start.x,scene.start.y)) if scene.start else regmap[r].geometry.bounds[0],regmap[r].geometry.bounds[1],r))
            selected=options[rid][0][1]
            if scene.start:
                ranked=[]
                for cost,q in options[rid][:4]:
                    motion,meta=nav.connect_poses(scene.start,io._pose_at(nav.work_motion(*q[0]),False))
                    ranked.append(((int(motion is None and meta['method']!='COINCIDENT')+cost[0],cost[1]+meta.get('seconds',0.)),q))
                selected=min(ranked,key=lambda x:x[0])[1]
        else:
            proposals=[]
            for rid in sorted(todo):
                for cost,q in options[rid]:
                    for directed in (q,[(t,not rev) for t,rev in reversed(q)]):
                        directed=nav.port_safe_order(directed)
                        actual_cost=sequence_cost(nav,directed,width)
                        a=io._pose_at(nav.work_motion(*seq[-1]),True);b=io._pose_at(nav.work_motion(*directed[0]),False)
                        proposals.append(((actual_cost[0],actual_cost[1]+math.hypot(a.x-b.x,a.y-b.y)/scene.vehicle.turn_speed_mps,rid),rid,directed,actual_cost[1]))
            checked=[]
            for rank,r,q,internal in sorted(proposals,key=lambda x:x[0])[:4]:
                m,meta=nav.connect_poses(io._pose_at(nav.work_motion(*seq[-1]),True),io._pose_at(nav.work_motion(*q[0]),False))
                checked.append(((rank[0]+int(m is None and meta['method']!='COINCIDENT'),internal+meta.get('seconds',0.),r),r,q))
            _,rid,selected=min(checked,key=lambda x:x[0])
        todo.remove(rid);seq.extend(selected);region_order.append(rid)
    nav.set_phase('ASSEMBLY')
    rows=[];failures=[];component=1;order=0;previous=scene.start;previous_id='__START__';issues=[]
    def append(m,kind,task_id='',region_id='',phase='BODY',method='',from_task='',to_task='',**extra):
        nonlocal order
        order+=1;meta=reference_motion_metrics(m,scene)
        rows.append(dict(kind=kind,task_id=task_id,region_id=region_id,from_task=from_task,to_task=to_task,sequence=order,component=component,phase=phase,method=method,direction='FORWARD_ORDER' if kind=='WORK' else 'VEHICLE_MOTION',geometry=LineString(m.points[:,:2]),motion_json=json.dumps(m.points.tolist(),separators=(',',':')),events_json=json.dumps(meta['events'],separators=(',',':')),kinematic_status='PASS' if not meta['issues'] else 'FAIL',max_curvature_inv_m=meta['max_curvature_inv_m'],maximum_curvature_rate=meta['maximum_curvature_rate'],gear_shifts=meta['gear_shifts'],reverse_legs=meta['reverse_legs'],implement_on=m.implement_on,motion_kind=m.kind,**extra))
        if meta['issues']:issues.extend(meta['issues'])
        nav.record_work(m)
    def join(goal,goal_id,phase,region_id=''):
        nonlocal component
        if previous is None:return
        m,meta=nav.connect_poses(previous,goal)
        if m is not None:
            # Forward headland links can cut their own corridor. A reverse
            # transfer remains lifted and must pass chronological crop replay.
            if settings.operational_refinement and phase=='HEADLAND' and np.all(m.points[:,3]>0):
                m=Motion(m.points,'work','',True)
            append(m,'CONNECTION',phase=phase,method=meta['method'],region_id=region_id,from_task=previous_id,to_task=goal_id)
        elif meta['method']!='COINCIDENT':
            failures.append(dict(from_task=previous_id,to_task=goal_id,reason=meta['method']));component+=1
    def execute_body():
        nonlocal previous,previous_id
        nav.set_phase('ASSEMBLY')
        for task,rev in seq:
            m=nav.work_motion(task,rev)
            entry=None
            if settings.operational_refinement and previous is not None:
                link,meta=nav.connect_poses(previous,io._pose_at(m,False))
                if link is None and meta['method']!='COINCIDENT':entry=nav.cutting_entry(previous,m)
            if entry is None:join(io._pose_at(m,False),task.task_id,'BODY',task.region_id)
            else:
                prefix,tail,meta=entry;entry_id='entry_'+task.task_id;prefix_id='approach_'+task.task_id
                if prefix is not None:
                    append(prefix,'CONNECTION',prefix_id,task.region_id,method=meta['method'],from_task=previous_id,to_task=entry_id)
                append(tail,'CONNECTION',entry_id,task.region_id,method='EXPLICIT_CUTTING_ENTRY',from_task=prefix_id if prefix is not None else previous_id,to_task=task.task_id)
            append(m,'WORK',task.task_id,task.region_id,method='COVERAGE_PRESERVING_VEHICLE_LINE',reference_direction='REVERSED_ORDER' if rev else 'FORWARD_ORDER',frozen_reference_wkt=task.reference_line.wkt)
            rows[-1]['direction']='REVERSED_ORDER' if rev else 'FORWARD_ORDER'
            previous=io._pose_at(m,True);previous_id=task.task_id
            if settings.operational_refinement:
                exit_work=nav.cutting_exit(m)
                if exit_work is not None:
                    previous_id='exit_'+task.task_id
                    append(exit_work,'CONNECTION',previous_id,task.region_id,method='EXPLICIT_CUTTING_EXIT')
                    previous=io._pose_at(exit_work,True)
    if not settings.operational_refinement:execute_body()
    heads=[];head_failures=[]
    nav.set_phase('TERMINAL')
    if seq:
        turns=LocalTurns(nav.solver_scene,time.perf_counter()+settings.max_headland_seconds)
        placement=corner_placement(nav.solver_scene,settings.headland_pass_count)
        heads,_,head_failures=generate_regular_headlands(nav.solver_scene,turns,settings.headland_pass_count,placement)
        accepted=[]
        for head in heads:
            rejected=sorted({code for m in head['motions'] for code in reference_motion_metrics(m,scene)['issues']})
            if rejected:head_failures.append(dict(component_id=head['component_id'],pass_index=head['pass_index'],reason='STRICT_REFERENCE_KINEMATICS:'+','.join(rejected)))
            else:accepted.append(head)
        heads=accepted
    # Preserve ring direction and internal stopped-steering boundaries. Rotate
    # only at an existing motion boundary; never cut a curved task arbitrarily.
    for head in sorted(heads,key=lambda h:(h['pass_index'],h['component_id'])):
        motions=head['motions']
        indices=sorted(range(len(motions)),key=lambda i:math.hypot(motions[i].points[0,0]-previous.x,motions[i].points[0,1]-previous.y) if previous else i)[:4]
        choices=[]
        for i in indices:
            if previous is None:
                choices.append(((0,0.,i),i));continue
            m,meta=nav.connect_poses(previous,io._pose_at(motions[i],False))
            choices.append(((int(m is None and meta['method']!='COINCIDENT'),m.length if m else 0.,i),i))
        _,i=min(choices);rotated=motions[i:]+motions[:i]
        if settings.operational_refinement and min(choices)[0][0]:
            head_failures.append(dict(component_id=head['component_id'],pass_index=head['pass_index'],reason='HEADLAND_ENTRY_NOT_FOUND'))
            continue
        for n,original in enumerate(rotated):
            task_id=f"head_{head['component_id']}_{n:04d}"
            m=Motion(original.points,original.kind if settings.operational_refinement else 'headland_reference',task_id,settings.operational_refinement)
            join(io._pose_at(m,False),task_id,'HEADLAND')
            part_index=int(head['component_id'].split('_')[1].removeprefix('poly'))+1
            append(m,'HEADLAND',task_id,phase='HEADLAND',method='ORDERED_HEADLAND_REFERENCE',headland_pass_index=head['pass_index'],headland_part_index=part_index,headland_ring_role='EXTERIOR' if head['contour_index']==0 else 'INTERIOR')
            previous=io._pose_at(m,True);previous_id=task_id
    fill_diagnostic={};fill_omissions=[];fill_count=0
    if settings.operational_refinement and seq and settings.max_headland_fill_seconds>0:
        covered=unary_union([t.frozen_sweep for t in job.tasks]+[work_coverage(Motion(np.asarray(json.loads(r['motion_json'])),r['motion_kind'],r['task_id'],r['implement_on']),scene) for r in rows if r['phase']=='HEADLAND' and r['implement_on']])
        fills,fill_diagnostic=fill_gaps(job.field_id,scene,covered,[t.heading_rad for t in job.tasks],seconds=settings.max_headland_fill_seconds,max_segments=settings.max_headland_fill_segments)
        # Supplemental proposals are optional, not new frozen obligations.
        # A failed entry remains an explicit coverage gap and omission record;
        # it must not claim an unexecuted pass or break a usable body itinerary.
        remaining=list(fills)
        while remaining:
            choices=[]
            for task in sorted(remaining,key=lambda t:math.hypot(t.reference_line.centroid.x-previous.x,t.reference_line.centroid.y-previous.y) if previous else t.task_id)[:3]:
                for reverse in (False,True):
                    m=io._work_motion(task,scene,reverse)
                    if reference_motion_metrics(m,scene)['issues']:continue
                    if MotionChecker(scene).physical(m)[0]:continue
                    link,meta=nav.connect_poses(previous,io._pose_at(m,False)) if previous else (None,dict(method='COINCIDENT',seconds=0.))
                    if link is not None or meta['method']=='COINCIDENT':choices.append((meta.get('seconds',0.),task.task_id,task,m))
            if not choices:
                fill_omissions.extend(dict(task_id=t.task_id,reason='SUPPLEMENT_ENTRY_NOT_FOUND',proposed_area_m2=t.frozen_sweep.area) for t in remaining)
                break
            _,_,task,m=min(choices,key=lambda x:x[:2]);remaining.remove(task)
            join(io._pose_at(m,False),task.task_id,'HEADLAND','__HEADLAND__')
            append(m,'HEADLAND',task.task_id,'__HEADLAND__',phase='HEADLAND',method='INDEPENDENT_HEADLAND_GAP_WORK',headland_pass_index=0,headland_part_index=0,headland_ring_role='GAP_FILL')
            previous=io._pose_at(m,True);previous_id=task.task_id;fill_count+=1
    if settings.operational_refinement:execute_body()
    nav.set_phase('EXIT')
    if scene.end is not None and seq:join(scene.end,'__END__','EXIT')
    if settings.operational_refinement:
        # Entry/exit movements have their own IDs. Link them to the actual
        # neighbouring records only after the full ordered itinerary exists.
        for i,r in enumerate(rows):
            if r['kind']=='CONNECTION':
                r['from_task']=rows[i-1]['task_id'] if i else '__START__'
                r['to_task']=rows[i+1]['task_id'] if i+1<len(rows) else '__END__' if scene.end else '__FREE_END__'
    generated={p:sum(r['kind']=='HEADLAND' and r['headland_pass_index']==p for r in rows) for p in range(1,settings.headland_pass_count+1)}
    passes=[dict(pass_index=p,line_count=n,omitted_line_count=sum(f.get('pass_index')==p for f in head_failures),reason='GENERATED' if n else 'NO_KINEMATIC_HEADLAND_REFERENCE') for p,n in generated.items()]
    ids=[r['task_id'] for r in rows if r['kind']=='WORK']
    if Counter(ids)!=Counter(t.task_id for t in job.tasks) or len(ids)!=len(set(ids)):issues.append('TASK_ID_MISMATCH')
    if any(not scene.travel.buffer(1e-5).covers(r['geometry']) for r in rows):issues.append('LINE_OUTSIDE_TRAVEL')
    style=style_metrics(rows,{t.task_id:t for t in job.tasks},width)
    if style['heading_discontinuity_count']:issues.append('JOIN_HEADING_DISCONTINUITY')
    if failures or head_failures:style['reference_style_status']='REVIEW'
    geometry_issues=[i for i in issues if i in ('TASK_ID_MISMATCH','LINE_OUTSIDE_TRAVEL')]
    status='NO_BODY_TASKS' if not seq else 'PARTIAL_CONNECTED' if any(f['reason']!='GEOMETRICALLY_DISCONNECTED' for f in failures) else 'GEOMETRICALLY_DISCONNECTED' if failures else 'COMPLETE_CONNECTED'
    info=dict(field_id=job.field_id,motion_refinement=True,reference_space='TRAVEL',reference_route_status=status,input_task_count=len(job.tasks),output_task_count=len(ids),route_component_count=component if rows else 0,failed_connections=len(failures),failures=failures,region_order=region_order,reference_geometry_passed=not geometry_issues,issues=sorted(set(issues)),connection_length_m=sum(r['geometry'].length for r in rows if r['kind']=='CONNECTION'),max_connection_length_m=max((r['geometry'].length for r in rows if r['kind']=='CONNECTION'),default=0.),grid_expanded_nodes=nav.expanded,motion_queries=nav.motion_queries,motion_rejections=dict(nav.rejections),planning_seconds=time.perf_counter()-started,physical_vehicle_certification='NOT_EVALUATED',implement_switch_lag_status='PENDING_MEASUREMENT',acceptance_passed=False,all_work_sweeps_preserved=True,ordered_reference_motion_passed=bool(seq) and not issues and not any(f['reason']!='GEOMETRICALLY_DISCONNECTED' for f in failures),single_itinerary_connected=bool(seq) and not failures,external_transfer_required=any(f['reason']=='GEOMETRICALLY_DISCONNECTED' for f in failures),search_incomplete=any(f['reason']!='GEOMETRICALLY_DISCONNECTED' for f in failures),upstream_seam_quality_status=getattr(job,'upstream_seam_quality_status','UNKNOWN'),upstream_acceptance_passed=getattr(job,'upstream_acceptance_passed',None),upstream_area_ledger_delta_m2=getattr(job,'upstream_area_ledger_delta_m2',None),requested_headland_pass_count=settings.headland_pass_count,generated_headland_pass_count=sum(n>0 for n in generated.values()),headland_line_count=sum(generated.values()),headland_generation_status='COMPLETE' if all(generated.values()) and not head_failures else 'PARTIAL' if any(generated.values()) else 'EMPTY',headland_passes=passes,headland_failures=head_failures,headland_sequence_status='CONNECTED_REFERENCE' if heads and not failures else 'PARTIAL' if heads else 'EMPTY',headland_work_coverage_status='NOT_EVALUATED',start_pose_status='APPLIED' if scene.start else 'FREE_CHOICE',end_pose_status='APPLIED' if scene.end else 'FREE_CHOICE',operation_time_status='CONFIG_ESTIMATE_NOT_FIELD_MEASURED')
    info.update(style);info['reference_acceptance_passed']=reference_field_accepted(info)
    if settings.operational_refinement:
        if not sum(generated.values()):info['headland_sequence_status']='EMPTY'
        validation_started=time.perf_counter()
        info.update(evaluate_reference_operation(rows,scene))
        info['operation_validation_seconds']=time.perf_counter()-validation_started
        info.update(headland_contour_line_count=info['headland_line_count'],headland_fill_line_count=fill_count,headland_line_count=info['headland_line_count']+fill_count,headland_fill_diagnostic=fill_diagnostic,headland_fill_omissions=fill_omissions,motion_query_phase_counts=dict(nav.phase_queries))
    info['planning_seconds']=time.perf_counter()-started
    for name,pose,at_start in [('start',scene.start,True),('end',scene.end,False)]:
        if pose is not None:
            p=np.asarray(json.loads(rows[0 if at_start else -1]['motion_json']))[0 if at_start else -1] if rows else None
            applied=p is not None and np.linalg.norm(p[:2]-[pose.x,pose.y])<1e-5 and abs(wrap(p[2]-pose.yaw))<1e-5
            info[name+'_pose_status']='APPLIED' if applied else 'UNCONNECTED'
    return rows,info


# 单田参考路线：确定分区和条带顺序，再连接并生成独立田头参考线。
# 必须安排全部主体条带一次；几何断开或搜索受限时如实输出分段状态。
def reference_field_accepted(info):
    """判断本参考策略的完整任务及几何验收；自然断开可保留独立行程，搜索未完成不能认定通过。
    
    Natural multipart references are valid; unfinished searches are not."""
    return (info.get('reference_geometry_passed') is True
            and info.get('input_task_count', 0) > 0
            and info.get('input_task_count') == info.get('output_task_count')
            and info.get('reference_route_status') in
                ('COMPLETE_CONNECTED', 'GEOMETRICALLY_DISCONNECTED')
            and info.get('invalid_smooth_count', 0) == 0
            and (not info.get('motion_refinement') or
                 (info.get('all_work_sweeps_preserved') is True and
                  info.get('ordered_reference_motion_passed') is True)))


def solve(job,settings):
    """参考算法分发：运动细化开启时走位姿模型，否则走完整几何候选；均保持冻结任务，返回有序段与分项诊断。"""
    if settings.motion_refinement:return solve_refined_reference(job,settings)
    start=time.perf_counter();scene=approx_load_scene(job.scene_path)
    nav=Navigator(scene.target,settings);width=scene.vehicle.working_width_m
    region_options={}
    for reg in job.regions:
        ts=[t for t in job.tasks if t.region_id==reg.region_id]
        if ts:region_options[reg.region_id]=sorted(((sequence_cost(nav,q,width),q) for q in region_sequences(ts,nav,width)),key=lambda x:x[0])
    todo=set(region_options);seq=[];region_order=[];regmap={r.region_id:r for r in job.regions}
    while todo:
        if not seq:
            rid=min(todo,key=lambda r:(regmap[r].geometry.bounds[0],regmap[r].geometry.bounds[1],r));chosen=region_options[rid][0][1]
        else:
            previous=regmap[region_order[-1]].geometry
            choices=[]
            for rid in sorted(todo):
                adjacent=previous.distance(regmap[rid].geometry)<.05
                for cost,q in region_options[rid]:
                    # Same snake traversed backwards: no additional permutation search.
                    for directed in (q,[(t,not rev) for t,rev in reversed(q)]):
                        dist=math.dist(oriented(*seq[-1])[-1],oriented(*directed[0])[0])
                        choices.append(((cost[0],not adjacent,cost[1]+dist,rid),rid,directed,cost[1]))
            ranked=sorted(choices,key=lambda x:x[0])[:4]
            checked=[]
            for rank,r,q,internal in ranked:
                g,method=pair(nav,seq[-1],q[0],width)
                failed=int(g is None and method!='COINCIDENT')
                checked.append(((failed,rank[0],rank[1],internal+(g.length if g is not None else 0.),r),r,q))
            _,rid,chosen=min(checked,key=lambda x:x[0])
        todo.remove(rid);seq.extend(chosen);region_order.append(rid)
    rows=[];failures=[];component=1;order=0
    for i,item in enumerate(seq):
        task,rev=item
        if i:
            g,status=pair(nav,seq[i-1],item,width)
            if g is None and status!='COINCIDENT':
                failures.append({'from_task':seq[i-1][0].task_id,'to_task':task.task_id,'reason':status});component+=1
            elif g is not None:
                order+=1;rows.append(dict(kind='CONNECTION',region_id=task.region_id,task_id='',from_task=seq[i-1][0].task_id,to_task=task.task_id,sequence=order,component=component,method=status,direction='REFERENCE_ONLY',geometry=g))
        order+=1;rows.append(dict(kind='WORK',region_id=task.region_id,task_id=task.task_id,from_task='',to_task='',sequence=order,component=component,method='FROZEN_SWATH',direction='REVERSED_ORDER' if rev else 'FORWARD_ORDER',geometry=LineString(oriented(task,rev))))
    headland_passes=[];headland_issues=[]
    for i in range(settings.headland_pass_count):
        offset=scene.target.buffer(-(i+.5)*width,join_style='round')
        generated=0;omitted=0
        parts=sorted(approx_polygons(offset),key=lambda p:(p.bounds,p.area,p.wkb_hex))
        for part_index,p in enumerate(parts,1):
            rings=[('EXTERIOR',p.exterior),*[
                ('INTERIOR',r) for r in sorted(p.interiors,key=lambda r:(r.bounds,r.wkb_hex))]]
            for ring_index,(role,ring) in enumerate(rings,1):
                g=LineString(ring.coords)
                if g.length<=1e-6:
                    omitted+=1;continue
                if not g.is_valid or not scene.target.covers(g):
                    omitted+=1;headland_issues.append('INVALID_HEADLAND_REFERENCE');continue
                generated+=1
                rows.append(dict(kind='HEADLAND',region_id='',
                    task_id=f'head_{i+1:02d}_{part_index:03d}_{ring_index:03d}',
                    headland_pass_index=i+1,headland_part_index=part_index,
                    headland_ring_role=role,from_task='',to_task='',sequence=0,
                    component=0,method='INDEPENDENT_OFFSET',
                    direction='REFERENCE_ONLY',geometry=g))
        headland_passes.append(dict(pass_index=i+1,line_count=generated,
            omitted_line_count=omitted,reason=('OFFSET_EMPTY' if offset.is_empty else
                'REFERENCE_LINES_OMITTED' if omitted else 'GENERATED' if generated else 'NO_VALID_REFERENCE_LINES')))
    ids=[r['task_id'] for r in rows if r['kind']=='WORK'];expected=[t.task_id for t in job.tasks]
    issues=list(headland_issues)
    if Counter(ids)!=Counter(expected) or len(ids)!=len(set(ids)):issues.append('TASK_ID_MISMATCH')
    if any(not scene.target.covers(r['geometry']) for r in rows):issues.append('LINE_OUTSIDE_TARGET')
    reference_ids=[r['task_id'] for r in rows if r['kind'] in ('WORK','HEADLAND')]
    if len(reference_ids)!=len(set(reference_ids)):issues.append('DUPLICATE_REFERENCE_TASK_ID')
    status=('NO_BODY_TASKS' if not seq else 'PARTIAL_CONNECTED' if any(f['reason']!='GEOMETRICALLY_DISCONNECTED' for f in failures) else 'GEOMETRICALLY_DISCONNECTED' if failures else 'COMPLETE_CONNECTED')
    info=dict(field_id=job.field_id,reference_route_status=status,input_task_count=len(expected),output_task_count=len(ids),route_component_count=component if seq else 0,failed_connections=len(failures),failures=failures,region_order=region_order,reference_geometry_passed=not issues,issues=issues,connection_length_m=sum(r['geometry'].length for r in rows if r['kind']=='CONNECTION'),max_connection_length_m=max((r['geometry'].length for r in rows if r['kind']=='CONNECTION'),default=0.),grid_expanded_nodes=nav.expanded,planning_seconds=time.perf_counter()-start,physical_vehicle_certification='NOT_EVALUATED',implement_switch_lag_status='PENDING_MEASUREMENT',acceptance_passed=False)
    info.update(style_metrics(rows,{t.task_id:t for t in job.tasks},width))
    generated_passes=sum(p['line_count']>0 for p in headland_passes)
    info.update(single_itinerary_connected=bool(seq) and not failures,
        external_transfer_required=any(f['reason']=='GEOMETRICALLY_DISCONNECTED' for f in failures),
        search_incomplete=any(f['reason']!='GEOMETRICALLY_DISCONNECTED' for f in failures),
        upstream_seam_quality_status=getattr(job,'upstream_seam_quality_status','UNKNOWN'),
        upstream_acceptance_passed=getattr(job,'upstream_acceptance_passed',None),
        upstream_area_ledger_delta_m2=getattr(job,'upstream_area_ledger_delta_m2',None),
        requested_headland_pass_count=settings.headland_pass_count,
        generated_headland_pass_count=generated_passes,
        headland_line_count=sum(p['line_count'] for p in headland_passes),
        headland_generation_status=('COMPLETE' if generated_passes==settings.headland_pass_count else 'PARTIAL' if generated_passes else 'EMPTY'),
        headland_passes=headland_passes,headland_work_coverage_status='NOT_EVALUATED')
    info['reference_acceptance_passed']=reference_field_accepted(info)
    if settings.assemble_reference:
        full,metadata=assemble_geometry_reference(rows,scene,nav,width)
        info.update(metadata,full_reference_operations=full)
    return rows,info


def plot(job,rows,info,path):
    """绘制参考路线、连接及田头，图像不等于机具覆盖或实车认证。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    from shapely.geometry.polygon import orient
    scene=approx_load_scene(job.scene_path)
    if info.get('full_reference_operations'):
        rows=[{**r,'geometry':wkt.loads(r['geometry_wkt'])} for r in info['full_reference_operations']]
    fig,ax=plt.subplots(figsize=(10,8))
    for p in approx_polygons(scene.target):
        ax.fill(*p.exterior.xy,color='#f1f4ec',zorder=0)
        for ring in p.interiors:ax.fill(*ring.xy,color='white',zorder=1)
        ax.plot(*p.exterior.xy,color='#333333',lw=.8)
        for ring in p.interiors:ax.plot(*ring.xy,color='#333333',lw=.8)
    for reg in job.regions:
        for p in approx_polygons(reg.geometry):ax.plot(*p.exterior.xy,color='#aaaaaa',lw=.5,ls='--')
    for kind,color,lw in [('HEADLAND','#8b60b3',.8),('CONNECTION','#ed8a23',.9),('WORK','#198754',.9)]:
        xy=[list(r['geometry'].coords) for r in rows if r['kind']==kind]
        if xy:ax.add_collection(LineCollection(xy,colors=color,linewidths=lw,label=kind))
    if info.get('motion_refinement'):
        connections=[r for r in rows if r['kind']=='CONNECTION']
        reverse=[]
        for r in connections:
            p=np.asarray(json.loads(r['motion_json']))
            reverse.extend([p[i:i+2,:2] for i in range(len(p)-1) if p[i,3]<0])
        if reverse:ax.add_collection(LineCollection(reverse,colors='#d34343',linewidths=1.6,label='REVERSE'))
        for r in connections[::max(1,len(connections)//30)]:
            g=r['geometry'];a=g.interpolate(.45,normalized=True);b=g.interpolate(.55,normalized=True)
            ax.annotate('',xy=b.coords[0],xytext=a.coords[0],arrowprops=dict(arrowstyle='->',color='#c66a13',lw=1.2))
    work=[r for r in rows if r['kind']=='WORK']
    for r in work[::max(1,len(work)//25)]:
        g=r['geometry'];a=g.interpolate(.45,normalized=True);b=g.interpolate(.55,normalized=True)
        ax.annotate('',xy=b.coords[0],xytext=a.coords[0],arrowprops=dict(arrowstyle='->',color='#12683f',lw=.8))
    itinerary=sorted(rows,key=lambda r:r['sequence'])
    if itinerary:
        ax.scatter(*itinerary[0]['geometry'].coords[0],c='#1769aa',s=35,zorder=5,label='PLANNED START')
        ax.scatter(*itinerary[-1]['geometry'].coords[-1],c='#ce3434',s=35,zorder=5,label='PLANNED END')
    if info.get('operational_refinement'):
        for gap in info['operation_coverage_gaps']:
            for p in approx_polygons(wkt.loads(gap['geometry_wkt'])):
                # Coverage gaps can surround covered islands. Opposite ring
                # winding keeps those holes transparent in the GIS preview.
                p=orient(p,sign=1.);vertices=[];codes=[]
                for ring in [p.exterior,*p.interiors]:
                    xy=list(ring.coords);vertices.extend(xy)
                    codes.extend([PlotPath.MOVETO,*[PlotPath.LINETO]*(len(xy)-2),PlotPath.CLOSEPOLY])
                ax.add_patch(PathPatch(PlotPath(vertices,codes),facecolor='#d74343',edgecolor='none',alpha=.35,zorder=1.5))
        invalid={x['sequence'] for x in info['operation_issues'] if 'COLLISION' in x['code']}
        xy=[list(r['geometry'].coords) for r in rows if r['sequence'] in invalid]
        if xy:ax.add_collection(LineCollection(xy,colors='#b50000',linewidths=2,label='MODEL COLLISION'))
    ax.set_aspect('equal');ax.autoscale();ax.legend(loc='upper right',fontsize=8)
    ax.set_title(f"{job.field_id} | REFERENCE ROUTE\n{info['reference_route_status']} | tasks {info['output_task_count']}/{info['input_task_count']} | components {info['route_component_count']}\nStyle: {info['reference_style_status']}",fontsize=10)
    if info.get('operational_refinement'):
        ax.set_title(ax.get_title()+f"\nConfigured rig: {info['configured_rig_model_status']} | crop access: {info['chronological_crop_access_status']} | uncovered: {info['target_missing_m2']:.1f} m² | gates: {info['gate_data_status']}",fontsize=9)
    ax.set_xlabel('Local x (m)');ax.set_ylabel('Local y (m)');fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def worker(arg):
    """历史参考整批工作入口，保存单田记录及绘图；生产逐田GPKG入口使用route_planner的隔离工作进程。"""
    job,settings,out=arg
    try:
        rows,info=solve(job,settings);directory=Path(out)/'fields'/job.field_id;directory.mkdir(parents=True)
        (directory/'result.json').write_text(json.dumps({**info,'metric_crs':job.metric_crs,'origin':job.origin,'operations':[{**{k:v for k,v in r.items() if k!='geometry'},'geometry_wkt':r['geometry'].wkt} for r in rows]},ensure_ascii=False,indent=2))
        os.environ['MPLCONFIGDIR']=str(Path(out)/'cache'/'matplotlib')
        p=time.perf_counter();plot(job,rows,info,directory/'overview.png');info['plot_seconds']=time.perf_counter()-p
        return job.field_id,rows,info
    except Exception as exc:return job.field_id,[],dict(field_id=job.field_id,reference_route_status='ERROR',error=f'{type(exc).__name__}: {exc}',reference_geometry_passed=False,reference_acceptance_passed=False,upstream_seam_quality_status=job.upstream_seam_quality_status,upstream_acceptance_passed=job.upstream_acceptance_passed,upstream_area_ledger_delta_m2=job.upstream_area_ledger_delta_m2,physical_vehicle_certification='NOT_EVALUATED',headland_work_coverage_status='NOT_EVALUATED',acceptance_passed=False)


# 参考策略的批入口：读取冻结条带，逐田求解和绘图，再导出并独立复核。
# reference_acceptance 与物理车辆认证分开；参考通过不会改写严格 acceptance。
def run_reference_batch(swath_bundle,out,*,workers=None,field_id=None,route_config=None,
        incremental=True,**batch_options):
    """几何参考默认逐田 GPKG 写入；历史整批导出仅用于显式兼容复验。

    运动模型沿用其既有批次协议。大规模生产使用推荐几何配置，启用完整
    参考组装和 FULL_REFERENCE 效率，不隐式降级到历史内存导出。
    """
    data=config_section(route_config, "routes") if route_config else {}
    if incremental and not data.get('motion_refinement',False):
        result=io.run_incremental_reference_batch(swath_bundle,out,workers=0 if workers is None else workers,
            field_id=field_id,route_config=route_config,**batch_options)
        return 0 if result['reference_acceptance_passed'] and result['efficiency_status_counts'].get('ESTIMATED',0)==result['input_field_count'] else 1
    if batch_options:raise ValueError('INCREMENTAL_OPTIONS_REQUIRE_GEOMETRY_REFERENCE')
    return _run_reference_batch_legacy(swath_bundle,out,workers=12 if workers is None else workers,field_id=field_id,route_config=route_config)


def _run_reference_batch_legacy(swath_bundle,out,*,workers=12,field_id=None,route_config=None):
    """历史整批参考导出兼容路径，保留V1图层和JSON供旧数据读取，不用于当前默认精简批量承诺。"""
    import geopandas as gpd
    import pandas as pd
    start=time.perf_counter();out=Path(out).resolve()
    if out.exists() and any(p.name!='cache' for p in out.iterdir()):raise ValueError('OUTPUT_DIRECTORY_MUST_BE_NEW_OR_EMPTY')
    if type(workers) is not int or not 1<=workers<=12:raise ValueError('INVALID_WORKERS')
    data=config_section(route_config, "routes") if route_config else {}
    if not isinstance(data,dict) or set(data)-{f.name for f in fields(ApproxRouteSettings)}:raise ValueError('INVALID_ROUTE_CONFIG')
    settings=ApproxRouteSettings(**data);out.mkdir(parents=True,exist_ok=True)
    (out/'cache'/'matplotlib').mkdir(parents=True,exist_ok=True)
    source_before={p.name:io._sha256(p) for p in Path(__file__).parent.glob('*.py')}
    input_release=io._verify_bundle_seal(Path(swath_bundle))
    _,manifest=io._verify_bundle(Path(swath_bundle));jobs=io._field_rows(Path(swath_bundle),manifest)
    if field_id:
        ids=set(field_id.split(','));jobs=[j for j in jobs if j.field_id in ids]
        if {j.field_id for j in jobs}!=ids:raise ValueError('UNKNOWN_FIELD_ID')
    input_seconds=time.perf_counter()-start;p=time.perf_counter()
    os.environ['MPLCONFIGDIR']=str(out/'cache'/'matplotlib')
    # Build font cache once, before workers, rather than once per process.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    font_manager.findfont('DejaVu Sans')
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(worker,[(j,settings,str(out)) for j in jobs]))
    solve_plot_seconds=time.perf_counter()-p;p=time.perf_counter();result_map={r[0]:r for r in results}
    export=[];sources=[];regions=[];coverage_gaps=[];complete_export=[]
    local=all(j.metric_crs=='LOCAL_METRIC' for j in jobs)
    if any(j.metric_crs=='LOCAL_METRIC' for j in jobs) and not local:raise ValueError('MIXED_LOCAL_AND_PROJECTED_CRS')
    export_crs=jobs[0].source_crs if local else 'EPSG:4326'
    for job in jobs:
        _,rows,info=result_map[job.field_id]
        forward=None if local else Transformer.from_crs(job.metric_crs,export_crs,always_xy=True)
        def world(g):
            g=translate(g,*job.origin)
            return g if local else transform(forward.transform,g)
        scene=approx_load_scene(job.scene_path)
        sources.append({**{k:v for k,v in info.items() if isinstance(v,(str,int,float,bool))},'geometry':world(scene.target)})
        regions.extend(dict(field_id=job.field_id,region_id=r.region_id,geometry=world(r.geometry)) for r in job.regions)
        export.extend({'field_id':job.field_id,**r,'geometry':world(r['geometry'])} for r in rows)
        for r in info.get('full_reference_operations',[]):
            complete_export.append({'field_id':job.field_id,**{k:v for k,v in r.items() if k!='geometry_wkt'},'geometry':world(wkt.loads(r['geometry_wkt']))})
        for n,g in enumerate(info.get('operation_coverage_gaps',[])):
            coverage_gaps.append(dict(field_id=job.field_id,gap_id=n,area_m2=g['area_m2'],geometry=world(wkt.loads(g['geometry_wkt']))))
    gpkg=out/'reference_routes.gpkg'
    gpd.GeoDataFrame(sources,geometry='geometry',crs=export_crs).to_file(gpkg,layer='source_fields',driver='GPKG')
    gpd.GeoDataFrame(regions,geometry='geometry',crs=export_crs).to_file(gpkg,layer='work_regions',driver='GPKG')
    for kind,layer in [('WORK','body_work'),('CONNECTION','body_connections'),('HEADLAND','headland_reference')]:
        rs=[r for r in export if r['kind']==kind]
        if rs:gpd.GeoDataFrame(rs,geometry='geometry',crs=export_crs).to_file(gpkg,layer=layer,driver='GPKG')
    if complete_export:gpd.GeoDataFrame(complete_export,geometry='geometry',crs=export_crs).to_file(gpkg,layer='reference_operations',driver='GPKG')
    if settings.operational_refinement:
        head_work=[r for r in export if r['kind']!='WORK' and r.get('implement_on')]
        if head_work:gpd.GeoDataFrame(head_work,geometry='geometry',crs=export_crs).to_file(gpkg,layer='headland_work',driver='GPKG')
        if coverage_gaps:gpd.GeoDataFrame(coverage_gaps,geometry='geometry',crs=export_crs).to_file(gpkg,layer='target_uncovered',driver='GPKG')
    # One ordered polyline per connected body itinerary, for direct GIS inspection.
    itineraries=[];full_itineraries=[]
    for job in jobs:
        _,rows,info=result_map[job.field_id]
        forward=None if local else Transformer.from_crs(job.metric_crs,export_crs,always_xy=True)
        groups=[(itineraries,[r for r in rows if r.get('phase','BODY')=='BODY' and r['kind']!='HEADLAND'])]
        if settings.motion_refinement:groups.append((full_itineraries,rows))
        elif settings.assemble_reference:
            groups.append((full_itineraries,[{**r,'geometry':wkt.loads(r['geometry_wkt'])} for r in info.get('full_reference_operations',[])]))
        for destination,records in groups:
            for component in sorted({r['component'] for r in records}):
                ordered=sorted((r for r in records if r['component']==component),key=lambda r:r['sequence'])
                xy=[]
                for r in ordered:
                    coords=list(r['geometry'].coords)
                    if xy and math.dist(xy[-1],coords[0])>1e-5:raise ValueError('ITINERARY_ENDPOINT_GAP')
                    xy.extend(coords if not xy else coords[1:])
                if len(xy)>=2:
                    g=translate(LineString(xy),*job.origin)
                    if forward is not None:g=transform(forward.transform,g)
                    destination.append(dict(field_id=job.field_id,component=component,reference_route_status=info['reference_route_status'],physical_vehicle_certification='NOT_EVALUATED',geometry=g))
    if itineraries:gpd.GeoDataFrame(itineraries,geometry='geometry',crs=export_crs).to_file(gpkg,layer='body_itinerary',driver='GPKG')
    if full_itineraries:gpd.GeoDataFrame(full_itineraries,geometry='geometry',crs=export_crs).to_file(gpkg,layer='full_reference_itinerary',driver='GPKG')
    # Independently replay exported geometry in each field's original metric frame.
    saved=[]
    for layer in ('body_work','body_connections','headland_reference'):
        if layer in set(gpd.list_layers(gpkg)['name']):saved.extend(gpd.read_file(gpkg,layer=layer).to_dict('records'))
    byfield={}
    for r in saved:byfield.setdefault(r['field_id'],[]).append(r)
    saved_itineraries=gpd.read_file(gpkg,layer='body_itinerary').to_dict('records') if itineraries else []
    audit=[]
    for job in jobs:
        scene=approx_load_scene(job.scene_path);back=None if local else Transformer.from_crs(export_crs,job.metric_crs,always_xy=True)
        rs=[]
        for r in byfield.get(job.field_id,[]):
            rs.append({**r,'geometry':translate(r['geometry'] if local else transform(back.transform,r['geometry']),-job.origin[0],-job.origin[1])})
        original={t.task_id:t for t in job.tasks};work=[r for r in rs if r['kind']=='WORK'];issues=[]
        if Counter(r['task_id'] for r in work)!=Counter(original.keys()):issues.append('EXPORTED_TASK_MISMATCH')
        for r in work:
            if r['task_id'] in original:
                expected=(LineString(io._work_motion(original[r['task_id']],scene,r['direction']=='REVERSED_ORDER').points[:,:2]) if settings.motion_refinement else original[r['task_id']].reference_line)
                if r['geometry'].hausdorff_distance(expected)>1e-5:issues.append('EXPORTED_SWATH_CHANGED')
        for r in rs:
            # 10 micrometres is a projection roundtrip tolerance, not a routing allowance.
            allowed=scene.travel if settings.motion_refinement else scene.target
            if not allowed.buffer(1e-5).covers(r['geometry']):issues.append('EXPORTED_OUTSIDE_REFERENCE_SPACE')
            if settings.motion_refinement:
                m=Motion(np.asarray(json.loads(r['motion_json'])),'reference')
                if reference_motion_metrics(m,scene)['issues']:issues.append('EXPORTED_KINEMATIC_ISSUE')
        route=sorted((r for r in rs if settings.motion_refinement or r['kind']!='HEADLAND'),key=lambda r:r['sequence'])
        for a,b in zip(route,route[1:]):
            if a['component']==b['component'] and math.dist(a['geometry'].coords[-1],b['geometry'].coords[0])>1e-5:issues.append('EXPORTED_ENDPOINT_GAP')
            if settings.motion_refinement and a['component']==b['component']:
                pa=np.asarray(json.loads(a['motion_json']));pb=np.asarray(json.loads(b['motion_json']))
                if abs(wrap(pa[-1,2]-pb[0,2]))>1e-5:issues.append('EXPORTED_JOIN_HEADING')
        for itinerary in saved_itineraries:
            if itinerary['field_id']!=job.field_id:continue
            g=translate(itinerary['geometry'] if local else transform(back.transform,itinerary['geometry']),-job.origin[0],-job.origin[1])
            if not any(p.buffer(1e-5).covers(g) for p in approx_polygons(scene.travel if settings.motion_refinement else scene.target)):issues.append('ITINERARY_CROSSES_COMPONENTS')
        audit.append(dict(field_id=job.field_id,passed=not issues,issues=sorted(set(issues))))
    # A changed input or source invalidates the field stamps as well as the batch.
    try:
        input_unchanged=io._verify_bundle_seal(Path(swath_bundle))==input_release
    except (OSError,ValueError):
        input_unchanged=False
    sources_unchanged={p.name:io._sha256(p) for p in Path(__file__).parent.glob('*.py')}==source_before
    infos=[r[2] for r in results]
    for info,checked in zip(infos,audit):
        info['export_audit_passed']=checked['passed']
        info['input_integrity_passed']=input_unchanged
        info['source_code_unchanged']=sources_unchanged
        info['reference_acceptance_passed']=reference_field_accepted(info) and checked['passed'] and input_unchanged and sources_unchanged
        if info['reference_route_status']!='ERROR':
            path=out/'fields'/info['field_id']/'result.json'
            stored=json.loads(path.read_text());stored.update(info);path.write_text(json.dumps(stored,ensure_ascii=False,indent=2))
    # Rewrite field metadata after independent export checks so all outputs agree.
    for source,info in zip(sources,infos):
        source.update({k:v for k,v in info.items() if isinstance(v,(str,int,float,bool))})
    gpd.GeoDataFrame(sources,geometry='geometry',crs=export_crs).to_file(gpkg,layer='source_fields',driver='GPKG')
    pd.DataFrame([{k:v for k,v in i.items() if not isinstance(v,(dict,list,tuple))} for i in infos]).to_csv(out/'fields_summary.csv',index=False)
    (out/'export_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    cards=''.join(f'<article><h3>{html.escape(i["field_id"])} — {i["reference_route_status"]}</h3><a href="fields/{i["field_id"]}/overview.png"><img loading="lazy" src="fields/{i["field_id"]}/overview.png"></a></article>' for i in infos if i['reference_route_status']!='ERROR')
    title='完整几何参考路线：主体＋连接＋有序田头；实车约束未认证' if settings.assemble_reference else '车辆位姿参考路线：绿=主体，橙=转弯/连接，紫=有序田头；整车安全与全田作业未认证' if settings.motion_refinement else '参考路线：绿=主体，橙=连接，紫=独立田头；未认证农机运动'
    (out/'gallery.html').write_text('<meta charset="utf-8"><title>参考路线</title><style>body{font-family:sans-serif}main{display:grid;grid-template-columns:repeat(3,1fr)}img{width:100%}article{padding:8px}</style><h1>'+title+'</h1><main>'+cards+'</main>')
    after={p.name:io._sha256(p) for p in Path(__file__).parent.glob('*.py')}
    summary=dict(route_strategy=settings.route_strategy,input_bundle=str(Path(swath_bundle).resolve()),workers=workers,input_field_count=len(jobs),status_counts=dict(Counter(i['reference_route_status'] for i in infos)),style_status_counts=dict(Counter(i.get('reference_style_status','NOT_EVALUATED') for i in infos)),reference_geometry_passed=all(i['reference_geometry_passed'] for i in infos) and all(a['passed'] for a in audit),all_tasks_preserved=all(i['reference_route_status']!='ERROR' and i.get('input_task_count')==i.get('output_task_count') for i in infos),source_code_unchanged=after==source_before,source_code_sha256_start=source_before,source_code_sha256_end=after,settings=asdict(settings),input_seconds=input_seconds,solve_and_plot_wall_seconds=solve_plot_seconds,export_and_audit_seconds=time.perf_counter()-p,total_wall_seconds=time.perf_counter()-start,planning_seconds_sum=sum(i.get('planning_seconds',0) for i in infos),plot_seconds_sum=sum(i.get('plot_seconds',0) for i in infos),physical_vehicle_certification='NOT_EVALUATED',acceptance_passed=False)
    summary['reference_style_passed']=all(i.get('reference_style_status')=='PASS' for i in infos)
    if settings.assemble_reference:
        summary.update(full_reference_connected_count=sum(i.get('full_reference_connected',False) for i in infos),full_reference_status_counts=dict(Counter(i.get('full_reference_status','ERROR') for i in infos)),full_reference_style_counts=dict(Counter(i.get('full_reference_style_status','ERROR') for i in infos)),full_reference_scope='BODY_AND_ORDERED_HEADLANDS_FREE_TERMINALS_WHEN_UNSET')
    if settings.motion_refinement:
        summary['motion_refinement']=True
        summary['ordered_reference_motion_passed']=all(i.get('ordered_reference_motion_passed') for i in infos)
        summary['all_work_sweeps_preserved']=all(i.get('all_work_sweeps_preserved') for i in infos)
        summary['headland_sequence_counts']=dict(Counter(i.get('headland_sequence_status','ERROR') for i in infos))
    summary['input_integrity_passed']=input_unchanged
    summary['input_release_sha256']=input_release
    summary['input_field_ids']=[j.field_id for j in jobs]
    summary['upstream_seam_quality_counts']=dict(Counter(i['upstream_seam_quality_status'] for i in infos))
    summary['upstream_failed_field_count']=sum(i['upstream_acceptance_passed'] is not True for i in infos)
    summary['external_transfer_field_count']=sum(i.get('external_transfer_required',False) for i in infos)
    summary['headland_generation_counts']=dict(Counter(i.get('headland_generation_status','ERROR') for i in infos))
    summary['headland_work_coverage_status']='NOT_EVALUATED'
    if settings.operational_refinement:
        summary.update(operational_refinement=True,configured_rig_model_counts=dict(Counter(i.get('configured_rig_model_status','ERROR') for i in infos)),chronological_crop_access_counts=dict(Counter(i.get('chronological_crop_access_status','ERROR') for i in infos)),configured_operation_pass_count=sum(i.get('configured_operation_passed',False) for i in infos),headland_work_coverage_counts=dict(Counter(i.get('headland_work_coverage_status','ERROR') for i in infos)),gate_data_counts=dict(Counter(i.get('gate_data_status','ERROR') for i in infos)),target_missing_m2_sum=sum(i.get('target_missing_m2',0.) for i in infos),headland_additional_area_m2_sum=sum(i.get('headland_additional_area_m2',0.) for i in infos),headland_work_coverage_status='COVERAGE_COMPLETE' if all(i.get('headland_work_coverage_status')=='COVERAGE_COMPLETE' for i in infos) else 'COVERAGE_GAP')
    summary['reference_acceptance_passed']=summary['input_integrity_passed'] and summary['reference_geometry_passed'] and summary['all_tasks_preserved'] and summary['source_code_unchanged'] and all(i['reference_acceptance_passed'] for i in infos)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if not k.startswith('source_code_sha256')},ensure_ascii=False),flush=True)
    return 0 if summary['reference_acceptance_passed'] else 1
