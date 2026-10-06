"""独立几何验收。输入 Scene、Motion/Plan，输出问题、扫掠与覆盖证据。
先读原标量验证，再读路线向量化包络；两者沿用相同几何规则。
文件末尾的旧 API 转发仅用于兼容历史脚本，算法实现不放在转发层。

分节目录：
1. 原始离散运动与覆盖验收
2. 路线阶段的向量化包络检查
"""
from __future__ import annotations


# ==========================================================================
# 1. 原始离散运动与覆盖验收
# 车体检查 travel，机具覆盖按对应阶段规则检查；只累计 implement_on 的扫掠。
# ==========================================================================

from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from scene import Scene
from scene import Motion
from scene import Plan
from scene import Pose
from scene import Budget
from scene import polygonal
from scene import wrap


# 独立验收的结果结构：运动问题与覆盖问题分别记录。
# passed 由现有运动、覆盖和目标完整性规则决定，不能仅凭导出成功。
@dataclass
class Validation:
    """几何/运动验收结果；是否通过、覆盖是否评估及缺失面积分开记录，未评估不等于没有缺失。"""
    passed: bool
    issues: list[dict[str, Any]]
    coverage_fraction: float
    missing_area_m2: float
    largest_missing_patch_m2: float
    missing: BaseGeometry
    covered: BaseGeometry
    motion_valid: bool
    coverage_evaluated: bool = True

    def summary(self) -> dict[str, Any]:
        """输出可序列化验收摘要；通过字段只对应本验证器实际检查的范围。"""
        return {"passed": self.passed, "motion_valid": self.motion_valid,
                "coverage_evaluated": self.coverage_evaluated,
                "coverage_fraction": self.coverage_fraction,
                "missing_area_m2": self.missing_area_m2,
                "largest_missing_patch_m2": self.largest_missing_patch_m2,
                "issues": self.issues}


def dense_points(points: np.ndarray, step: float, yaw_step: float,
                 max_points: int) -> np.ndarray:
    """分段线性位置+最短角插值；不使用会改变路径拓扑的样条平滑。"""
    result = []
    count = 0
    for a, b in zip(points[:-1], points[1:]):
        dyaw = wrap(float(b[2] - a[2]))
        n = max(1, math.ceil(np.linalg.norm(b[:2] - a[:2]) / step), math.ceil(abs(dyaw) / yaw_step))
        count += n
        if count + 1 > max_points:
            raise ValueError("单段路径采样数量超限；未继续粗化检查")
        t = np.arange(n) / n
        result.append(np.column_stack((a[0] + t * (b[0] - a[0]),
                                       a[1] + t * (b[1] - a[1]),
                                       a[2] + t * dyaw, np.full(n, a[3]))))
    return np.vstack([*result, points[-1:]])


def transform_vertices(local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """按局部位姿旋转平移轮廓顶点；xy单位米，朝向弧度。"""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return local @ np.array([[c, s], [-s, c]]) + pose[:2]


# 按姿态采样构造车体和机具扫掠，并补偿采样之间的旋转误差。
# 返回的两个包络用途不同：车体用于通行核验，机具用于作业覆盖核验。
def swept_polygons(motion: Motion, scene: Scene, work_only: bool = False):
    """恒定朝向直线使用一次精确平移扫掠，避免逐点做大量相同几何运算。"""
    settings = scene.settings
    p = motion.points
    base_rectangles = scene.vehicle.rectangles(work_only)
    delta = p[-1, :2] - p[0, :2]
    length = np.linalg.norm(delta)
    straight = (length > 1e-12 and
                abs(float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()) - length) < 1e-8 and
                np.max(np.abs(np.sin(p[:, 2] - p[0, 2]))) < 1e-9 and
                np.max(np.abs((p[:, 0] - p[0, 0]) * delta[1] -
                              (p[:, 1] - p[0, 1]) * delta[0])) < 1e-8)
    if straight:
        for local in base_rectangles:
            yield MultiPoint(np.vstack([transform_vertices(local, p[0]),
                                        transform_vertices(local, p[-1])])).convex_hull
        return
    dense = dense_points(p, settings.sampling_step_m, settings.heading_step_rad,
                         settings.max_motion_samples)
    for a, b in zip(dense[:-1], dense[1:]):
        for local in base_rectangles:
            hull = MultiPoint(np.vstack([transform_vertices(local, a), transform_vertices(local, b)])).convex_hull
            if not work_only:
                # 对声明的插值运动模型，加上每个刚性顶点旋转轨迹偏离弦的上界。
                radius = float(np.linalg.norm(local, axis=1).max())
                sagitta = radius * (1 - math.cos(abs(wrap(float(b[2] - a[2]))) / 2))
                if sagitta > 1e-12:
                    hull = hull.buffer(sagitta / math.cos(math.pi / 32), quad_segs=8)
            yield hull


class Validator:
    """原完整流程的独立运动与扫掠验证器；安全包络、覆盖义务和资源预算分开检查。"""
    def __init__(self, scene: Scene, budget: Budget | None = None) -> None:
        """准备Scene通行区及本实例缓存，几何epsilon只处理数值误差，不代表额外合法通行空间。"""
        self.scene, self.budget = scene, budget
        self.free = prep(scene.travel.buffer(scene.settings.geometry_epsilon_m))
        self._motion_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._coverage_cache: OrderedDict[str, BaseGeometry] = OrderedDict()

    def _tick(self) -> None:
        """在共享预算存在时检查期限，不把超时转换成几何可行结论。"""
        if self.budget:
            self.budget.check()

    def motion_issues(self, motion: Motion) -> list[dict[str, Any]]:
        """检查一段运动的接头、运动约束和扫掠碰撞，缓存只针对同一场景和运动内容。"""
        self._tick()
        if motion.key in self._motion_cache:
            self._motion_cache.move_to_end(motion.key)
            return self._motion_cache[motion.key]
        cfg, vehicle = self.scene.settings, self.scene.vehicle
        p = motion.points
        issues: list[dict[str, Any]] = []
        def issue(code: str, message: str, index: int = 0) -> None:
            issues.append({"code": code, "message": message, "sample": int(index),
                           "xy_local_m": [float(p[index, 0]), float(p[index, 1])]})
        if not vehicle.allow_reverse and np.any(p[:-1, 3] < 0):
            issue("REVERSE_FORBIDDEN", "当前规则禁止倒车")
        if motion.implement_on and np.any(p[:-1, 3] < 0):
            issue("REVERSE_WORK_UNSUPPORTED", "本版本不支持倒车作业")
        dxy = np.diff(p[:, :2], axis=0)
        ds = np.linalg.norm(dxy, axis=1)
        dyaw = (np.diff(p[:, 2]) + math.pi) % (2 * math.pi) - math.pi
        moving = ds > 1e-7
        zero_turn = (~moving) & (np.abs(dyaw) > cfg.heading_tolerance_rad)
        if np.any(zero_turn):
            issue("HEADING_JUMP", "原地出现车头方向跳变", int(np.flatnonzero(zero_turn)[0]))
        if np.any(moving):
            mid_heading = p[:-1, 2] + dyaw / 2 + np.where(p[:-1, 3] < 0, math.pi, 0)
            bearing = np.arctan2(dxy[:, 1], dxy[:, 0])
            difference = (bearing - mid_heading + math.pi) % (2 * math.pi) - math.pi
            misaligned = moving & (np.abs(difference) > max(0.08, cfg.heading_tolerance_rad))
            if np.any(misaligned):
                issue("LATERAL_JUMP", "位置变化与车身朝向/挡位不一致", int(np.flatnonzero(misaligned)[0]))
            curvature = np.zeros_like(ds)
            curvature[moving] = dyaw[moving] / (ds[moving] * p[:-1, 3][moving])
            bad = np.abs(curvature) > (1 + cfg.curvature_relative_tolerance) / vehicle.min_turn_radius_m
            if np.any(bad):
                issue("CURVATURE_LIMIT", "超过最小转弯半径约束", int(np.flatnonzero(bad)[0]))
            if cfg.check_curvature_rate and len(ds) > 1:
                indices = np.flatnonzero(moving)
                centres = (np.cumsum(ds) - ds / 2)[indices]
                distance = np.diff(centres)
                gears = p[:-1, 3][indices]
                valid = (gears[:-1] == gears[1:]) & (distance > 1e-7)
                rate = np.zeros_like(distance)
                rate[valid] = np.abs(np.diff(curvature[indices])[valid]) / distance[valid]
                limit = vehicle.max_curvature_rate * (1 + cfg.curvature_relative_tolerance) + cfg.curvature_rate_abs_tolerance
                if np.any(rate > limit):
                    issue("CURVATURE_RATE", "转向变化过快，或存在未圆滑拐角", int(np.flatnonzero(rate > limit)[0]))
        # 先拒绝明显运动错误，避免对坏候选执行高成本的整条扫掠检查。
        if not issues:
            for i, swept in enumerate(swept_polygons(motion, self.scene)):
                if i % 64 == 0:
                    self._tick()
                footprint = swept.buffer(vehicle.safety_margin_m / math.cos(math.pi / 32), quad_segs=8) if vehicle.safety_margin_m else swept
                if not self.free.covers(footprint):
                    issue("COLLISION_OR_BOUNDARY", "车体或固定机具扫掠范围碰撞/越界")
                    bad_geometry = footprint.difference(self.scene.travel)
                    if not bad_geometry.is_empty:
                        bad_point = bad_geometry.representative_point()
                        issues[-1]["xy_local_m"] = [bad_point.x, bad_point.y]
                        issues[-1]["swept_piece_index"] = i
                    break
        self._motion_cache[motion.key] = issues
        while len(self._motion_cache) > cfg.connection_cache_size:
            self._motion_cache.popitem(last=False)
        return issues

    def work_sweep(self, motion: Motion) -> BaseGeometry:
        """仅对开启机具的运动计算作业扫掠，关机具行驶不计覆盖。"""
        if not motion.implement_on:
            return GeometryCollection()
        if motion.key not in self._coverage_cache:
            # 当前版本的工作任务全部是直线。不要直接把任意弯曲作业曲线按此近似认证。
            if np.max(np.abs((motion.points[:, 2] - motion.points[0, 2] + math.pi) % (2 * math.pi) - math.pi)) > 1e-7:
                raise ValueError("当前覆盖模型只验收直线作业段；曲线作业需扩展机具扫掠模型")
            self._coverage_cache[motion.key] = unary_union(list(swept_polygons(motion, self.scene, True)))
            while len(self._coverage_cache) > self.scene.settings.connection_cache_size:
                self._coverage_cache.popitem(last=False)
        return self._coverage_cache[motion.key]

    def validate(self, plan: Plan, *, diagnostic_coverage: bool = False) -> Validation:
        """汇总Plan的运动、安全、顺序与覆盖问题，保存未覆盖差集；缺少完整证据时不能输出完整作业通过。"""
        self._tick()
        issues = list(plan.errors)
        actual_ids = {m.task_id for m in plan.motions if m.implement_on}
        expected_ids = {t.task_id for t in plan.tasks}
        missing_ids = sorted(expected_ids - actual_ids)
        if missing_ids:
            issues.append({"code": "MISSING_TASKS", "task_ids": missing_ids})
        if not plan.motions:
            issues.append({"code": "EMPTY_ROUTE", "message": "没有候选路径"})
        for i, motion in enumerate(plan.motions):
            for item in self.motion_issues(motion):
                issues.append({**item, "motion_index": i, "task_id": motion.task_id})
            if i:
                previous = plan.motions[i - 1].points[-1]
                current = motion.points[0]
                if np.linalg.norm(previous[:2] - current[:2]) > self.scene.settings.join_tolerance_m:
                    issues.append({"code": "DISCONNECTED", "motion_index": i, "message": "路径段间存在位置断点"})
                if abs(wrap(float(previous[2] - current[2]))) > self.scene.settings.heading_tolerance_rad:
                    issues.append({"code": "JOIN_HEADING", "motion_index": i, "message": "拼接处朝向不连续"})
                # 跨模块边界的转向变化也检查；不因拆成两个 Motion 而漏掉折角。
                left = plan.motions[i - 1].points[-3:]
                right = motion.points[:3]
                if np.linalg.norm(previous[:2] - current[:2]) <= self.scene.settings.join_tolerance_m:
                    seam = Motion(np.vstack([left, right]), "transit")
                    for item in self.motion_issues(seam):
                        if item["code"] in {"HEADING_JUMP", "CURVATURE_LIMIT", "CURVATURE_RATE", "LATERAL_JUMP"}:
                            issues.append({**item, "motion_index": i, "seam": True})
        if plan.motions:
            for name, pose, endpoint in (("start", self.scene.start, plan.motions[0].points[0]),
                                         ("end", self.scene.end, plan.motions[-1].points[-1])):
                if pose and (np.linalg.norm(endpoint[:2] - [pose.x, pose.y]) > self.scene.settings.join_tolerance_m or
                             abs(wrap(float(endpoint[2] - pose.yaw))) > self.scene.settings.heading_tolerance_rad):
                    issues.append({"code": "ENDPOINT_POSE", "endpoint": name})
        motion_valid = not issues
        coverage_evaluated = motion_valid or diagnostic_coverage
        if coverage_evaluated:
            self._tick()
            covered = polygonal(unary_union([self.work_sweep(m) for m in plan.motions if m.implement_on])
                                .intersection(self.scene.target))
        else:
            covered = GeometryCollection()
        missing = polygonal(self.scene.target.difference(covered))
        from scene import polygons
        largest = max((p.area for p in polygons(missing)), default=0.0)
        missed = float(missing.area)
        fraction = max(0.0, min(1.0, float(covered.area / self.scene.target.area)))
        if coverage_evaluated and missed > self.scene.settings.coverage_tolerance_m2:
            issues.append({"code": "UNCOVERED", "message": "原始目标作业区仍存在漏作", "area_m2": missed})
        return Validation(motion_valid and missed <= self.scene.settings.coverage_tolerance_m2,
                          issues, fraction, missed, float(largest), missing, covered, motion_valid,
                          coverage_evaluated)

# ==========================================================================
# 2. 路线阶段的向量化包络检查
# 保持原采样、旋转补偿和面积规则；这里只加速几何核验，不生成路径。
# ==========================================================================

import math
import numpy as np
import shapely
from shapely.geometry import GeometryCollection
from shapely.geometry import Point
from shapely.geometry import Polygon
geometry_FrozenValidator = Validator
geometry_dense_points = dense_points
geometry_swept_polygons = swept_polygons
from scene import Motion as geometry_Motion
from scene import Scene as geometry_Scene
from scene import Pose as geometry_Pose
from scene import wrap as geometry_wrap


def sweep_part_groups(motion: geometry_Motion, scene: geometry_Scene):
    """分别返回车体与固定机具的采样扫掠，车辆通行区与实际作业目标的约束不同，不能合成一个模糊安全标志。
    
    Return body and fixed-implement sweeps separately.
    
    The body must stay in the vehicle travel zone. The implement may work
    across its edge clearance, but its real hull must remain inside the true
    field target, including the exclusion of interior obstacles."""
    p = motion.points
    delta = p[-1, :2] - p[0, :2]
    length = np.linalg.norm(delta)
    straight = (length > 1e-12 and
                abs(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()-length) < 1e-8
                and np.max(np.abs(np.sin(p[:, 2]-p[0, 2]))) < 1e-9
                and np.max(np.abs((p[:, 0]-p[0, 0])*delta[1]
                                   -(p[:, 1]-p[0, 1])*delta[0])) < 1e-8)
    if straight:
        parts = list(geometry_swept_polygons(motion, scene))
        return (np.array(parts[:1], dtype=object),
                np.array(parts[1:], dtype=object))
    else:
        cfg = scene.settings
        p = geometry_dense_points(p, cfg.sampling_step_m, cfg.heading_step_rad,
                         cfg.max_motion_samples)
        c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
        arrays = []
        for rect in scene.vehicle.rectangles():
            x = p[:, 0, None]+c[:, None]*rect[:, 0]-s[:, None]*rect[:, 1]
            y = p[:, 1, None]+s[:, None]*rect[:, 0]+c[:, None]*rect[:, 1]
            vertices = np.stack((x, y), axis=2)
            pairs = np.concatenate((vertices[:-1], vertices[1:]), axis=1)
            hulls = shapely.convex_hull(shapely.multipoints(pairs))
            yaw = (np.diff(p[:, 2])+math.pi) % (2*math.pi)-math.pi
            sagitta = np.linalg.norm(rect, axis=1).max()*(1-np.cos(np.abs(yaw)/2))
            use = sagitta > 1e-12
            hulls[use] = shapely.buffer(hulls[use], sagitta[use]/math.cos(math.pi/32),
                                        quad_segs=8)
            arrays.append(hulls)
        return arrays[0], arrays[1]


def sweep_parts(motion: geometry_Motion, scene: geometry_Scene, *, margin: bool = False):
    """合并车体和机具扫掠数组；margin按参数增加安全外扩，不能当作实际机具作业覆盖。"""
    parts = np.concatenate(sweep_part_groups(motion, scene))
    if margin and scene.vehicle.safety_margin_m:
        parts = shapely.buffer(parts, scene.vehicle.safety_margin_m/math.cos(math.pi/32),
                               quad_segs=8)
    return parts


def vehicle_sweep(motion: geometry_Motion, scene: geometry_Scene):
    """返回车体与固定机具扫掠并集，不是仅车体，也不是开启机具作业面积。"""
    return shapely.union_all(sweep_parts(motion, scene))


def conservative_tool_coverage(motion: geometry_Motion, scene: geometry_Scene):
    """只计已采样作业位姿实际占用的机具面，是弯曲作业覆盖下界；插值空隙不能自认已作业。
    
    Return only tool footprints actually occupied at sampled work poses.
    
    This is a lower bound on curved-work coverage, so interpolation gaps are
    never silently counted as worked.  It does not certify the motion: callers
    must separately check the full vehicle and fixed implement envelope."""
    if not motion.implement_on:
        return GeometryCollection()
    points = geometry_dense_points(motion.points, scene.settings.sampling_step_m,
                          scene.settings.heading_step_rad,
                          scene.settings.max_motion_samples)
    tool = scene.vehicle.rectangles(work_only=True)[0]
    c = np.cos(points[:, 2, None])
    s = np.sin(points[:, 2, None])
    x = points[:, 0, None] + c * tool[:, 0] - s * tool[:, 1]
    y = points[:, 1, None] + s * tool[:, 0] + c * tool[:, 1]
    return shapely.union_all(shapely.polygons(np.stack((x, y), axis=2)))


def kinematic_issues(motion: geometry_Motion, scene: geometry_Scene):
    """与原运动契约一致的数组检查，不用修改冻结模块放宽物理约束。"""
    p, cfg, v = motion.points, scene.settings, scene.vehicle
    if len(p) < 2 or not np.isfinite(p).all() or not np.isin(p[:, 3], [-1, 1]).all():
        return ["INVALID_MOTION_STATE"]
    dxy = np.diff(p[:, :2], axis=0)
    ds = np.linalg.norm(dxy, axis=1)
    dyaw = (np.diff(p[:, 2])+math.pi) % (2*math.pi)-math.pi
    moving = ds > 1e-7
    problems = []
    if np.any((~moving)&(np.abs(dyaw)>cfg.heading_tolerance_rad)):
        problems.append("HEADING_JUMP")
    bearing = np.arctan2(dxy[:, 1], dxy[:, 0])
    mid = p[:-1, 2]+dyaw/2+np.where(p[:-1, 3]<0, math.pi, 0)
    error = (bearing-mid+math.pi) % (2*math.pi)-math.pi
    if np.any(moving & (np.abs(error)>max(0.08, cfg.heading_tolerance_rad))):
        problems.append("LATERAL_JUMP")
    k = np.divide(dyaw, ds*p[:-1, 3], out=np.zeros_like(ds), where=moving)
    if np.any(np.abs(k)>(1+cfg.curvature_relative_tolerance)/v.min_turn_radius_m):
        problems.append("CURVATURE_LIMIT")
    ix = np.flatnonzero(moving)
    if cfg.check_curvature_rate and len(ix)>1:
        distance = np.diff((np.cumsum(ds)-ds/2)[ix])
        gears = p[:-1, 3][ix]
        valid = (gears[:-1]==gears[1:]) & (distance>1e-7)
        rate = np.divide(np.abs(np.diff(k[ix])), distance,
                         out=np.zeros_like(distance), where=valid)
        limit = v.max_curvature_rate*(1+cfg.curvature_relative_tolerance)+cfg.curvature_rate_abs_tolerance
        if np.any(rate>limit):
            problems.append("CURVATURE_RATE")
    if not v.allow_reverse and np.any(p[:-1, 3]<0):
        problems.append("REVERSE_FORBIDDEN")
    if motion.implement_on and np.any(p[:-1, 3]<0):
        problems.append("REVERSE_WORK_UNSUPPORTED")
    return problems


# 路线阶段的几何检查器：可批量计算包络，但不负责决定条带顺序。
# 缓存只减少重复计算，不能用上一次的通过状态替代当前空间条件。
class MotionChecker:
    """路线阶段分开核验车体通行、机具边界及运动约束；配置模型检查不能替代实车认证。"""
    def __init__(self, scene):
        """准备局部米制车体通行区和真实机具允许目标，保留场景数值容差。"""
        self.scene = scene
        self.free = scene.travel.buffer(scene.settings.geometry_epsilon_m)
        self.tool_free = scene.target.buffer(scene.settings.geometry_epsilon_m)
        shapely.prepare(self.free)
        shapely.prepare(self.tool_free)
        self.cache = {}
        self.calls = self.cache_hits = 0

    def physical(self, motion):
        """检查车辆/机具实际包络对各自允许空间的覆盖关系；外扩余量不能扩大真实作业目标。"""
        self.calls += 1
        if motion.key in self.cache:
            self.cache_hits += 1
            return self.cache[motion.key]
        issues = kinematic_issues(motion, self.scene)
        parts = None
        if not issues:
            body, tool = sweep_part_groups(motion, self.scene)
            parts = np.concatenate((body, tool))
            v = self.scene.vehicle
            safe_body = (shapely.buffer(body, v.safety_margin_m/math.cos(math.pi/32), quad_segs=8)
                         if v.safety_margin_m else body)
            if not np.all(shapely.covers(self.free, safe_body)):
                issues = ["BODY_COLLISION_OR_BOUNDARY"]
            elif not np.all(shapely.covers(self.tool_free, tool)):
                issues = ["IMPLEMENT_COLLISION_OR_BOUNDARY"]
        result = (issues, parts)
        # 失败曲线不用保留几何缓存；成功候选很少且会与不同执行账本再次检查，缓存不能替代新的时序约束。
        if len(self.cache)>512:
            self.cache.clear()
        self.cache[motion.key] = result if not issues else (issues, None)
        return result


# 沿用原 Validator 的验收条件，使用数组几何实现减少小面循环。
# 新策略仍须区分运动有效、覆盖有效和整田结果是否真正通过。
class RouteValidator(geometry_FrozenValidator):
    """路线候选专用验证器，区分车体通行与机具目标约束；不改冻结Scene或原Validator的契约。
    
    Route-local validator using the separately certified body/tool zones.
    
    The frozen scene and upstream validator retain their original contract.
    Route task variants and headland proposals use this subclass so candidate
    generation and final route audit agree on the new work-edge policy."""

    def __init__(self, scene, budget=None):
        """复用原验证器初始化并准备路线阶段的机具允许区。"""
        super().__init__(scene, budget)
        self.motion_checker = MotionChecker(scene)

    def motion_issues(self, motion):
        """先保留既有运动问题，再按路线车体/机具分区规则核验候选。"""
        self._tick()
        codes, _ = self.motion_checker.physical(motion)
        return [{"code": code, "message": code, "sample": 0,
                 "xy_local_m": [float(motion.points[0, 0]),
                                float(motion.points[0, 1])]}
                for code in codes]

# ===========================================================================
# 4. 历史 API 的薄转发：不保存或执行另一份算法源码
# 旧脚本先导入 validator（或主入口）即可继续使用旧模块名。新代码直接使用十一个
# 正式模块。每个别名属性映射到唯一实际定义，旧 mock/patch 也更新该实际绑定。
# ===========================================================================
import importlib as _compat_importlib
import hashlib as _compat_hashlib
import json as _compat_json
import sys as _compat_sys
from pathlib import Path as _CompatPath
from types import ModuleType as _CompatModule

_COMPAT_ROOT = _CompatPath(__file__).resolve().parents[1]
from io_utils import config_section as _compat_section
_COMPAT_LAYOUT_DATA = _compat_section(_COMPAT_ROOT/'config.json', 'compatibility', 'legacy_api')
# 校验规范JSON内容，说明文字/其它参数变化不影响旧API映射；映射改动仍拒绝。
_COMPAT_LAYOUT_BYTES = _compat_json.dumps(_COMPAT_LAYOUT_DATA,sort_keys=True,separators=(',',':')).encode()
if _compat_hashlib.sha256(_COMPAT_LAYOUT_BYTES).hexdigest() != "4b80876d9f1bae3e91934d00665dc2430659e0f1be1ce9e65fdf1729649f6473":
    raise ValueError("旧 API 映射与源码指纹不一致，请恢复匹配的 config.json")
_COMPAT_LAYOUT = _COMPAT_LAYOUT_DATA["modules"]

class _LegacyModule(_CompatModule):
    """旧名称的属性视图；访问和 patch 都定位到唯一的正式模块属性。"""
    def __init__(self, name, record):
        """建立旧模块名到真实实体模块的属性视图，不复制或执行另一份算法。"""
        super().__init__(name)
        self.__dict__["_record"] = record
        self.__dict__["__file__"] = str(_COMPAT_ROOT / "src" / (record["module"] + ".py"))
    def __getattr__(self, name):
        """按受保护映射查找旧属性，缺失属性正常抛出异常。"""
        record = self.__dict__["_record"]
        module = _compat_importlib.import_module(record["module"])
        return getattr(module, record["attributes"].get(name, name))
    def __setattr__(self, name, value):
        """将旧接口patch定位到真实绑定；不能悄悄创建另一份独立实现。"""
        if name.startswith("__") or name == "_record":
            self.__dict__[name] = value
        else:
            record = self.__dict__["_record"]
            setattr(_compat_importlib.import_module(record["module"]),
                    record["attributes"].get(name, name), value)
    def __delattr__(self, name):
        """删除映射指向的真实属性，用于兼容测试的patch恢复机制。"""
        record = self.__dict__["_record"]
        delattr(_compat_importlib.import_module(record["module"]),
                record["attributes"].get(name, name))
    def __dir__(self):
        """列出兼容视图可见属性，便于检查旧接口映射。"""
        return sorted(set(self.__dict__) | set(self.__dict__["_record"]["attributes"]))

def legacy_source_path(path):
    """旧 src 文件名只解析到实际实现文件；不认可任何过期源码摘要。"""
    path = _CompatPath(path)
    if not path.is_file() and path.absolute().parent == (_COMPAT_ROOT / "src").absolute():
        record = _COMPAT_LAYOUT.get(path.stem)
        if record is not None:
            return _COMPAT_ROOT / "src" / (record["module"] + ".py")
    return path

for _old_name, _record in _COMPAT_LAYOUT.items():
    if _old_name != _record["module"]:
        _compat_sys.modules[_old_name] = _LegacyModule(_old_name, _record)
