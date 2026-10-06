"""V7 的作业分区、F2C 适配与历史完整流程候选生成，继承 V6 的冻结算法。

本次独立分区阶段通过 :func:`analyze_work_regions`，在 ``Scene`` 已给出的
``target`` 和 ``travel`` 基础上建立三个彼此分开的结果：

* 车辆完整包络在任意朝向均不会越界的安全参考点区域；
* 用于判断直行通道、连通分量和狭颈的区域—通道拓扑；
* 保持 target 面积守恒的候选作业区及其邻接约束。

这个阶段不生成田头、条带、转弯或最终路径。安全区只用于分析，绝不修改
``scene.target``。其中 ``straight_passage_area`` 是按车辆/机具最大横向半宽建立的
直行连通性筛查层；它不能单独证明长车在弯曲狭道中一定可通过。只有完整包络
安全区和保守调头区提供充分安全条件，不能满足充分条件的连接会明确标为待验证。

本文件还保留历史 full 流程的 F2C 适配：生成田头、分区、作业带、固定排序和转弯。
这些功能不属于上面的独立分区阶段。只计算当前顺序需要的连接；
失败时可尝试全田允许通行空间上的几何引导线，再由 F2C 生成转弯并复检。
几何引导不是 Hybrid A*，不宣称完备，也不把搜索失败解释成物理不可行。
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import heapq
import itertools
import math
import warnings
from typing import Any

import numpy as np
from shapely import voronoi_polygons, wkt
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, split, unary_union
from shapely.prepared import prep

from scene import Scene, Task, Pose, Motion, Plan, Budget, polygonal, polygons, wrap
from validator import Validator


@dataclass(frozen=True)
class WorkRegion:
    """一个候选作业区。

    ``geometry`` 是从原始 target 分配得到的面，因此所有候选区的并集必须等于
    target，且区间不允许有面积重叠。``seed`` 位于用于建立该区的安全空间中。
    ``reason`` 记录几何结构分区的来源；通行能力另由 access_status 表达。
    """

    region_id: str
    geometry: BaseGeometry
    seed: Point
    straight_component_id: int
    reason: str
    access_status: str
    work_core: BaseGeometry | None = None
    corridor_assignment: BaseGeometry | None = None
    preferred_angle_deg: float = 0.0
    mean_section_length_m: float = 0.0
    fragmentation: float = 0.0
    coverage_span_m: float = 0.0
    geometry_connected: bool = True
    safe_connected: bool = True
    safe_component_count: int = 1
    straight_component_count: int = 1
    reachability_status: str = "FULL_ENVELOPE_SAFE"
    unverified_target_area_m2: float = 0.0
    split_reason_code: str = "SIMPLE_TARGET"
    split_evidence: str = "整田方案通过硬约束，未发现足以接受分区的收益"
    sequence_index: int = 0
    turn_resource: str = "UNASSESSED"


@dataclass(frozen=True)
class RegionConnection:
    """候选作业区之间的一条拓扑关系，不代表已经生成可执行路径。"""

    connection_id: str
    from_region: str
    to_region: str
    mouth: BaseGeometry
    effective_width_m: float | None
    width_class: str
    straight_status: str
    turn_status: str
    evidence: str
    portal: Point | None = None
    transfer_line: BaseGeometry | None = None


@dataclass(frozen=True)
class WorkRegionAnalysis:
    """作业分区阶段的完整、可独立审计输出。"""

    target: BaseGeometry
    travel: BaseGeometry
    safe_reference_area: BaseGeometry
    straight_passage_area: BaseGeometry
    conservative_turn_area: BaseGeometry
    regions: tuple[WorkRegion, ...]
    connections: tuple[RegionConnection, ...]
    parameters: dict[str, float]
    checks: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """返回分区拓扑、面积与约束摘要；几何筛查通过不表示整车已经能沿完整路线作业。"""
        return {
            "region_count": len(self.regions),
            "connection_count": len(self.connections),
            "target_area_m2": float(self.target.area),
            "safe_reference_area_m2": float(self.safe_reference_area.area),
            "straight_passage_area_m2": float(self.straight_passage_area.area),
            "conservative_turn_area_m2": float(self.conservative_turn_area.area),
            "parameters": dict(self.parameters),
            "checks": dict(self.checks),
        }


@dataclass(frozen=True)
class StructureEvent:
    """外边界或车辆作业核心上的一个有尺度凹陷事件。

    ``depth_m`` 是凹点到本面凸包边界的距离，``support_m`` 是凹点两侧
    较短稳定边的长度。二者同时保留，避免把很深但只有几个噪声顶点的锯齿
    当成分区依据。事件只用于提出和审查候选，不改变原始 target。
    """

    anchor: Point
    depth_m: float
    support_m: float
    tangent_angles_deg: tuple[float, float]


def _structure_events(
    geometry: BaseGeometry,
    working_width_m: float,
    epsilon: float,
) -> list[StructureEvent]:
    """提取有车辆尺度意义的凹点，并对边界加密保持稳定。"""
    if geometry.is_empty:
        return []
    simplified = geometry.simplify(
        max(0.05, 0.50 * working_width_m), preserve_topology=True,
    )
    result: list[StructureEvent] = []
    for polygon in polygons(simplified):
        coordinates = list(polygon.exterior.coords)[:-1]
        if len(coordinates) < 4:
            continue
        orientation = 1 if polygon.exterior.is_ccw else -1
        hull_boundary = polygon.convex_hull.boundary
        for index, point in enumerate(coordinates):
            before, after = coordinates[index - 1], coordinates[(index + 1) % len(coordinates)]
            first_length = math.dist(before, point)
            second_length = math.dist(point, after)
            scale = first_length * second_length
            cross = ((point[0] - before[0]) * (after[1] - point[1])
                     - (point[1] - before[1]) * (after[0] - point[0]))
            if scale <= epsilon or cross * orientation >= -math.sin(math.radians(25)) * scale:
                continue
            anchor = Point(point)
            depth = anchor.distance(hull_boundary)
            support = min(first_length, second_length)
            # 低于一条幅宽的凹入通常只是边界细节。很深的凹口允许一侧
            # 支撑边稍短，但仍需至少半条幅宽，避免锯齿制造事件。
            if depth < 1.25 * working_width_m:
                continue
            if support < 0.50 * working_width_m and depth < 4.0 * working_width_m:
                continue
            result.append(StructureEvent(
                anchor=anchor,
                depth_m=float(depth),
                support_m=float(support),
                tangent_angles_deg=(
                    math.degrees(math.atan2(point[1] - before[1], point[0] - before[0])) % 180.0,
                    math.degrees(math.atan2(after[1] - point[1], after[0] - point[0])) % 180.0,
                ),
            ))
    result.sort(key=lambda item: (-item.depth_m, -item.support_m, item.anchor.x, item.anchor.y))
    return result


def _spatial_order(geometry: BaseGeometry) -> tuple[float, float, float]:
    """给几何建立稳定的空间顺序，使区域编号不依赖 GEOS 返回顺序。"""
    point = geometry.representative_point()
    return (round(point.x, 9), round(point.y, 9), -round(geometry.area, 9))


def _vehicle_clearances(scene: Scene) -> tuple[float, float, float]:
    """返回横向、任意朝向完整包络、保守调头三种参考点退让距离。"""
    vehicle = scene.vehicle
    lateral = max(vehicle.body_width_m, vehicle.working_width_m) / 2
    footprint_radius = max(
        float(np.linalg.norm(vertex))
        for rectangle in vehicle.rectangles()
        for vertex in rectangle
    )
    lateral += vehicle.safety_margin_m
    footprint_radius += vehicle.safety_margin_m
    return lateral, footprint_radius, footprint_radius + vehicle.min_turn_radius_m


def _meaningful_components(
    geometry: BaseGeometry,
    reference_area_m2: float,
    minimum_area_m2: float,
) -> list[BaseGeometry]:
    """返回参与作业拓扑的面分量，滤掉数值碎片和不足一个作业单元的小岛。"""
    threshold = max(reference_area_m2 * 1e-9, minimum_area_m2)
    return sorted(
        (part for part in polygons(polygonal(geometry)) if part.area >= threshold),
        key=_spatial_order,
    )


def _shared_interface(first: BaseGeometry, second: BaseGeometry, epsilon: float) -> BaseGeometry:
    """返回两区真实公共边界，并吸收坐标序列化产生的亚毫米级缝隙。"""
    exact = first.boundary.intersection(second.boundary)
    if exact.length > epsilon:
        return exact
    tolerance = max(10 * epsilon, 1e-7)
    return first.boundary.intersection(second.boundary.buffer(tolerance))


def _linearly_separate_two_cores(
    target: BaseGeometry,
    first_core: BaseGeometry,
    second_core: BaseGeometry,
    epsilon: float,
) -> list[BaseGeometry]:
    """用一条不穿过主体的直线分开两个核心，不可行时返回空列表。

    候选方向只来自两个核心凸包的边和主体中心连线。这是一个有限的
    支撑线检查，不做全角度搜索。只接受投影区间有正间隔、切后恰好
    两个单面、两个核心均未被切开的结果。
    """
    angle_normals: set[float] = set()
    for core in (first_core, second_core):
        coordinates = list(core.convex_hull.exterior.coords)
        for first, second in zip(coordinates, coordinates[1:]):
            dx, dy = second[0] - first[0], second[1] - first[1]
            if math.hypot(dx, dy) > epsilon:
                angle_normals.add(round(math.degrees(math.atan2(dx, -dy)) % 180.0, 6))
    first_seed, second_seed = first_core.representative_point(), second_core.representative_point()
    angle_normals.add(round(math.degrees(math.atan2(
        second_seed.y - first_seed.y, second_seed.x - first_seed.x,
    )) % 180.0, 6))

    hull_coordinates = [
        list(first_core.convex_hull.exterior.coords)[:-1],
        list(second_core.convex_hull.exterior.coords)[:-1],
    ]
    center = target.centroid
    reach = 4.0 * math.hypot(
        target.bounds[2] - target.bounds[0], target.bounds[3] - target.bounds[1],
    ) + 1.0
    candidates = []
    for angle in angle_normals:
        nx, ny = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        intervals = [
            (min(x * nx + y * ny for x, y in coordinates),
             max(x * nx + y * ny for x, y in coordinates))
            for coordinates in hull_coordinates
        ]
        if intervals[0][1] + epsilon < intervals[1][0]:
            lower, upper = intervals[0][1], intervals[1][0]
        elif intervals[1][1] + epsilon < intervals[0][0]:
            lower, upper = intervals[1][1], intervals[0][0]
        else:
            continue
        threshold = (lower + upper) / 2.0
        offset = threshold - (center.x * nx + center.y * ny)
        point_x, point_y = center.x + offset * nx, center.y + offset * ny
        cutter = LineString([
            (point_x - reach * ny, point_y + reach * nx),
            (point_x + reach * ny, point_y - reach * nx),
        ])
        pieces = [
            part for part in polygons(polygonal(split(target, cutter)))
            if part.area > epsilon * epsilon
        ]
        if len(pieces) != 2:
            continue
        direct = []
        unused = set(range(2))
        valid = True
        for core in (first_core, second_core):
            index = max(unused, key=lambda item: pieces[item].intersection(core).area)
            if pieces[index].intersection(core).area < core.area - epsilon:
                valid = False
                break
            unused.remove(index)
            direct.append(pieces[index])
        if valid:
            interface = _shared_interface(direct[0], direct[1], epsilon)
            candidates.append((interface.length, -(upper - lower), angle, direct))
    return min(candidates, key=lambda item: item[:3])[3] if candidates else []


def _partition_from_cores(
    target: BaseGeometry,
    cores: list[BaseGeometry],
    corridor_half_width_m: float,
    epsilon: float,
) -> list[tuple[BaseGeometry, BaseGeometry, BaseGeometry]]:
    """将 target 面积分配给已证实的主体，并把分界附近单独记为通道。

    Voronoi 只生成候选归属。结果随后必须通过单面、主体包含、面积守恒和
    区内安全核心连通检查；不满足这些检查的候选不会被自动接受。
    """
    cores = sorted(cores, key=_spatial_order)
    if len(cores) <= 1:
        return [(target, target, polygonal(target.difference(target)))]
    seeds = [core.representative_point() for core in cores]
    # 三个以上主体不能直接用单点 Voronoi：一个长安全核心可能跨过另一个
    # 核心的代表点中垂线，被切成多个碎片。先递归寻找“一个核心 vs 其余核心
    # 并集”的线性分离器；每条有限直线都完整保留两侧核心，最终分区与核心
    # 一一对应。只有不存在这类分离顺序时才退回后面的 Voronoi 候选。
    def separate_many(domain: BaseGeometry, subjects: list[BaseGeometry]) -> list[BaseGeometry]:
        if len(subjects) == 1:
            return [domain]
        if len(subjects) == 2:
            return _linearly_separate_two_cores(domain, subjects[0], subjects[1], epsilon)
        for subject_index, subject in enumerate(subjects):
            other_subjects = [
                item for index, item in enumerate(subjects) if index != subject_index
            ]
            other_union = polygonal(unary_union(other_subjects))
            pair = _linearly_separate_two_cores(domain, subject, other_union, epsilon)
            if len(pair) != 2:
                continue
            own_index = max(range(2), key=lambda index: pair[index].intersection(subject).area)
            own, remainder = pair[own_index], pair[1 - own_index]
            if own.intersection(subject).area < subject.area - epsilon:
                continue
            if remainder.intersection(other_union).area < other_union.area - epsilon:
                continue
            remainder_parts = separate_many(remainder, other_subjects)
            if len(remainder_parts) != len(other_subjects):
                continue
            ordered: list[BaseGeometry] = []
            remainder_iter = iter(remainder_parts)
            for index in range(len(subjects)):
                ordered.append(own if index == subject_index else next(remainder_iter))
            return ordered
        return []

    def sampled_core_voronoi(
        domain: BaseGeometry,
        subjects: list[BaseGeometry],
    ) -> list[BaseGeometry]:
        """用核心边界多点而非单个代表点建立归属单元。

        单点 Voronoi 会横切细长或弯曲安全核心。沿每个核心边界以车辆横向尺度
        采样后，同一核心的 Voronoi 单元合并成一个主体归属区；所有生成元仍只
        产生有限直线边。结果必须完整包含对应核心、互不重叠并覆盖原域，否则
        返回空列表让调用方继续采用保守后备方案。
        """
        sample_spacing = max(2.0 * corridor_half_width_m, 1.0)
        generators: list[Point] = []
        labels: list[int] = []
        seen_coordinates: set[tuple[int, int]] = set()

        def add(point: Point, label: int) -> None:
            scale = max(epsilon, 1e-8)
            key = (round(point.x / scale), round(point.y / scale))
            if key in seen_coordinates:
                return
            seen_coordinates.add(key)
            generators.append(point)
            labels.append(label)

        for label, subject in enumerate(subjects):
            add(subject.representative_point(), label)
            for polygon in polygons(subject):
                rings = [polygon.exterior, *polygon.interiors]
                for ring in rings:
                    count = max(8, math.ceil(ring.length / sample_spacing))
                    for index in range(count):
                        add(ring.interpolate(index / count, normalized=True), label)
        if len(generators) < len(subjects):
            return []
        cells = list(voronoi_polygons(
            MultiPoint([(point.x, point.y) for point in generators]),
            extend_to=domain.envelope,
        ).geoms)
        grouped: list[list[BaseGeometry]] = [[] for _ in subjects]
        tolerance = max(10 * epsilon, 1e-7)
        for cell in cells:
            generator_index = next(
                (index for index, point in enumerate(generators)
                 if cell.buffer(tolerance).covers(point)),
                None,
            )
            if generator_index is None:
                return []
            grouped[labels[generator_index]].append(cell)
        parts = [
            polygonal(domain.intersection(unary_union(group))) if group
            else polygonal(domain.difference(domain))
            for group in grouped
        ]
        union = polygonal(unary_union(parts))
        if (
            any(part.is_empty or len(polygons(part)) != 1 for part in parts)
            or any(
                part.intersection(subject).area < subject.area - max(epsilon, subject.area * 1e-9)
                for part, subject in zip(parts, subjects)
            )
            or union.symmetric_difference(domain).area > max(epsilon, domain.area * 1e-9)
            or sum(part.area for part in parts) - union.area > max(epsilon, domain.area * 1e-9)
        ):
            return []
        return parts

    # 两个主体时，最短间隙的中垂线是更稳定的局部狭颈截面。相比从
    # 两个代表点作全局中垂线，它不会因主体过长或弯曲而斜切其中一侧。
    if len(cores) == 2:
        parts = _linearly_separate_two_cores(target, cores[0], cores[1], epsilon)
        first, second = nearest_points(cores[0], cores[1])
        dx, dy = second.x - first.x, second.y - first.y
        distance = math.hypot(dx, dy)
        if not parts and distance > epsilon:
            middle_x, middle_y = (first.x + second.x) / 2.0, (first.y + second.y) / 2.0
            reach = 4.0 * math.hypot(
                target.bounds[2] - target.bounds[0], target.bounds[3] - target.bounds[1],
            ) + 1.0
            cutter = LineString([
                (middle_x - reach * dy / distance, middle_y + reach * dx / distance),
                (middle_x + reach * dy / distance, middle_y - reach * dx / distance),
            ])
            # 无限直线可能在田块的其他长臂再次入界，从而把本来完整的
            # 主体一起切开。只取包含最短核心间隙中点的那段田内弦，
            # 它对应局部狭颈的自然短截面。
            inside = target.intersection(cutter)
            inside_lines = [
                line for line in (list(inside.geoms) if hasattr(inside, "geoms") else [inside])
                if line.geom_type == "LineString" and line.length > epsilon
            ]
            if inside_lines:
                middle = Point(middle_x, middle_y)
                local = min(inside_lines, key=lambda line: line.distance(middle))
                first_coord, last_coord = local.coords[0], local.coords[-1]
                local_length = math.dist(first_coord, last_coord)
                if local_length > epsilon:
                    ux = (last_coord[0] - first_coord[0]) / local_length
                    uy = (last_coord[1] - first_coord[1]) / local_length
                    extension = max(100 * epsilon, 1e-4)
                    cutter = LineString([
                        (first_coord[0] - extension * ux, first_coord[1] - extension * uy),
                        (last_coord[0] + extension * ux, last_coord[1] + extension * uy),
                    ])
            cut_parts = [
                part for part in polygons(polygonal(split(target, cutter)))
                if part.area > epsilon * epsilon
            ]
            if len(cut_parts) == 2:
                direct = []
                unused_parts = set(range(2))
                for core in cores:
                    index = max(unused_parts, key=lambda item: cut_parts[item].intersection(core).area)
                    unused_parts.remove(index)
                    direct.append(cut_parts[index])
                if all(
                    direct[index].intersection(core).area >= core.area - epsilon
                    for index, core in enumerate(cores)
                ):
                    parts = direct
            else:
                parts = []
        elif not parts:
            parts = []
    else:
        parts = separate_many(target, cores)
        if not parts:
            parts = sampled_core_voronoi(target, cores)

    if len(cores) == 2 and not parts:
        # U 形主体可以在凸包上包围另一主体，此时不存在全局线性分离器。
        # 不退回沿 core buffer 画曲线；改为只检查凹角、局部主轴产生的少量
        # 有端点直线。一条合格的局部切线必须完整保留两个已识别核心。
        working_scale = max(2.0 * corridor_half_width_m, 1.0)
        natural: list[list[BaseGeometry]] = []
        natural.extend(_reflex_extension_candidates(target, working_scale, epsilon))
        directions = _scan_directions(target, ())
        for angle in directions:
            natural.extend(_critical_event_candidates(target, angle, working_scale, epsilon))
            natural.extend(_critical_halfplane_candidates(target, angle, working_scale, epsilon))
            natural.extend(_critical_halfplane_candidates(
                target, (angle + 90.0) % 180.0, working_scale, epsilon,
            ))
        choices = []
        for candidate in natural:
            if len(candidate) != 2:
                continue
            direct = []
            unused_candidate = set(range(2))
            valid = True
            for core in cores:
                index = max(
                    unused_candidate,
                    key=lambda item: candidate[item].intersection(core).area,
                )
                if candidate[index].intersection(core).area < core.area - epsilon:
                    valid = False
                    break
                unused_candidate.remove(index)
                direct.append(candidate[index])
            if not valid:
                continue
            interface = _shared_interface(direct[0], direct[1], epsilon)
            choices.append((
                interface.length,
                -min(part.area for part in direct),
                tuple(part.normalize().wkb for part in direct),
                direct,
            ))
        if choices:
            parts = min(choices, key=lambda item: item[:3])[3]

    if not parts:
        cells = list(voronoi_polygons(
            MultiPoint([(point.x, point.y) for point in seeds]),
            extend_to=target.envelope,
        ).geoms)
    # Shapely 2.0 不支持 ``ordered=True``，因此不假定 GEOS 返回顺序：
    # 用生成该单元的 seed 显式匹配。Voronoi 公共界面是直线，这些线是
    # 最终面积归属的唯一分界来源。
        ordered_cells: list[BaseGeometry] = []
        unused = set(range(len(cells)))
        tolerance = max(10 * epsilon, 1e-7)
        for seed in seeds:
            matches = [index for index in unused if cells[index].buffer(tolerance).covers(seed)]
            if len(matches) != 1:
                raise ValueError("Voronoi 单元无法与安全主体种子一一对应")
            index = matches[0]
            unused.remove(index)
            ordered_cells.append(cells[index])
        parts = [polygonal(target.intersection(cell)) for cell in ordered_cells]
    tolerance = max(10 * epsilon, 1e-7)
    # 核心只证明两侧存在有作业意义的主体，不再被强行“塞回”所属区。
    # 旧做法会把负 buffer 的圆弧和凹角轮廓复制到公共界面，造成
    # 曲线、台阶和三角折返。若直线分界使某个最终核心不可用，后续
    # 统一硬约束应拒绝该候选，而不应通过改写分界补救。
    # 凹边界或孔洞可能让一个 Voronoi 单元在 target 内留下分离薄片。
    # 每个主体只保留包含对应安全核心的主面，其他片段沿真实公共边界
    # 交给相邻主体；绝不以最近直线跨孔洞归属。
    connected_parts: list[BaseGeometry] = []
    fragments: list[BaseGeometry] = []
    for index, part in enumerate(parts):
        components = sorted(polygons(part), key=lambda item: -item.area)
        if not components:
            connected_parts.append(part)
            continue
        main = max(components, key=lambda item: (item.intersection(cores[index]).area, item.area))
        connected_parts.append(main)
        fragments.extend(item for item in components if item is not main)
    pending = sorted(fragments, key=lambda item: -item.area)
    while pending:
        progress = False
        for fragment in list(pending):
            choices = []
            for index, owner in enumerate(connected_parts):
                interface = _shared_interface(fragment, owner, tolerance).length
                if interface <= tolerance:
                    continue
                joined = polygonal(unary_union([owner, fragment]))
                if len(polygons(joined)) == 1:
                    choices.append((interface, owner.area, index, joined))
            if choices:
                _interface, _area, index, joined = max(choices)
                connected_parts[index] = joined
                pending.remove(fragment)
                progress = True
        if not progress:
            # 保留无法可靠归属的真实面作为独立区，后续会明确标为 REVIEW；
            # 不能为了凑成单面而静默删除目标面积。
            connected_parts.extend(pending)
            pending.clear()
    parts = connected_parts
    interfaces = [
        _shared_interface(parts[i], parts[j], epsilon)
        for i in range(len(parts)) for j in range(i + 1, len(parts))
    ]
    interfaces = [line for line in interfaces if line.length > epsilon]
    corridor_zone = polygonal(
        target.intersection(unary_union(interfaces).buffer(corridor_half_width_m))
    ) if interfaces else polygonal(target.difference(target))
    assignments = []
    for index, part in enumerate(parts):
        corridor = polygonal(part.intersection(corridor_zone))
        work_core = polygonal(part.difference(corridor))
        if work_core.is_empty:
            source_core = cores[index] if index < len(cores) else part
            work_core = polygonal(part.intersection(source_core.buffer(corridor_half_width_m)))
            corridor = polygonal(part.difference(work_core))
        assignments.append((part, work_core, corridor))
    return assignments


def _scan_directions(geometry: BaseGeometry, configured: tuple[float, ...]) -> list[float]:
    """从配置、稳定长边、凹口局部边和整体主轴生成少量无向方向。"""
    weighted: list[tuple[float, float]] = [(math.inf, float(value) % 180) for value in configured]
    hull = geometry.convex_hull.simplify(1e-7, preserve_topology=True)
    coordinates = list(hull.exterior.coords)
    for first, second in zip(coordinates, coordinates[1:]):
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        if length > 0:
            angle = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])) % 180
            weighted.append((length, angle))
    # 凸包顶点的主轴补充整体方向，避免只跟随某一条局部长边。
    samples = np.asarray(coordinates[:-1], dtype=float)
    if len(samples) >= 3:
        covariance = np.cov(samples - samples.mean(axis=0), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        weighted.append((float(np.sqrt(max(values))), math.degrees(math.atan2(axis[1], axis[0])) % 180))
    # 凸包会丢掉凹槽两侧真正的作业方向。用等效宽度估计分析尺度，把
    # 有深度凹口的两条局部稳定边补入；其权重仍小于配置方向和全局长边。
    scale = max(0.5, min(5.0, 2.0 * geometry.area / max(geometry.length, 1e-9)))
    for event in _structure_events(geometry, scale, 1e-9)[:8]:
        for angle in event.tangent_angles_deg:
            weighted.append((event.support_m, angle))
    chosen: list[float] = []
    for _length, angle in sorted(weighted, key=lambda item: -item[0]):
        if all(_angle_difference_deg(angle, existing) >= 3 for existing in chosen):
            chosen.append(angle)
        if len(chosen) >= 8:
            break
    return sorted(chosen)


def _sweep_quality(geometry: BaseGeometry, angle_deg: float) -> dict[str, float]:
    """评价一个方向上的连续作业段，不生成实际条带。

    旧实现无论田块尺度与障碍物有多复杂，都只取 24 条等距截线。这会让一条
    很窄、但足以中断作业带的孔洞恰好落在两条采样线之间，从而被误判为连续。
    这里把外边界和每个孔洞在扫描坐标中的 ``y`` 临界值作为事件，再在相邻事件
    之间取中点。这样每次截面拓扑变化都至少被观察一次；同时保留少量等距点，
    防止一条很长而没有顶点的边界区间只被一个样本代表。

    这仍是条带质量的快速代理，不是实际作业带或车辆连续运动的证明。
    """
    rotated = affinity.rotate(geometry, -angle_deg, origin=(0, 0))
    xmin, ymin, xmax, ymax = rotated.bounds
    span = ymax - ymin
    if span <= 1e-9:
        return {
            "angle_deg": angle_deg, "mean_section_length_m": 0.0,
            "p20_section_length_m": 0.0, "short_section_ratio": 1.0,
            "fragmentation": math.inf, "coverage_span_m": math.inf,
            "regularity_penalty_m": math.inf, "shape_cost_m": math.inf,
        }

    # 将所有环的横向极值/转折位置加入事件集。边界上的线不适合作为截线，
    # 因此实际采样使用相邻事件的内部中点。
    event_levels = {float(ymin), float(ymax)}
    # 分区候选会反复调用此函数。先在远小于一条作业带的尺度上简化分析边界，
    # 让测绘噪声不会把一次局部质量比较扩大为几百条截线。孔洞的横向范围单独
    # 保留，因而窄而重要的障碍物不会被这个性能保护吞掉。
    analysis_geometry = rotated.simplify(max(0.05, span / 500.0), preserve_topology=True)
    protected_levels = {float(ymin), float(ymax)}
    for polygon in polygons(analysis_geometry):
        event_levels.update(float(point[1]) for point in list(polygon.exterior.coords)[:-1])
        for ring in polygon.interiors:
            low, high = float(ring.bounds[1]), float(ring.bounds[3])
            event_levels.update((low, high))
            protected_levels.update((low, high))
    # 上限仍保留孔洞带及整体边界；其余边界事件按空间顺序稀疏化。这样成本
    # 随候选数量近似线性，而不会退化为对全部坐标顶点的重复扫描。
    if len(event_levels) > 48:
        ordered = sorted(event_levels)
        sampled = {ordered[index] for index in np.linspace(0, len(ordered) - 1, 48, dtype=int)}
        event_levels = sampled | protected_levels
    events = sorted(event_levels)
    # 大尺度区间补充少量均匀事件；上限避免复杂边界把代价推向全局枚举。
    coarse_count = min(24, max(8, int(math.ceil(span / max(5.0, span / 20.0)))))
    events = sorted(set(events) | set(float(value) for value in np.linspace(ymin, ymax, coarse_count + 1)))
    # 每个区间只求交一次，同时保存它代表的横向宽度。旧实现只给平均长度
    # 加权，fragmentation、短段率和分位数仍按“采样条数”计数，边界加密会
    # 改变评价。下面所有统计都复用同一批带宽截面。
    sections: list[tuple[float, list[float]]] = []
    for lower, upper in zip(events, events[1:]):
        width = upper - lower
        if width <= 1e-9:
            continue
        y = (lower + upper) / 2.0
        cross = rotated.intersection(LineString([(xmin - 1, y), (xmax + 1, y)]))
        pieces = list(cross.geoms) if hasattr(cross, "geoms") else [cross]
        lengths = [
            part.length for part in pieces
            if part.geom_type == "LineString" and part.length > 1e-6
        ]
        if lengths:
            sections.append((width, lengths))
    line_lengths = [length for _width, lengths in sections for length in lengths]
    if not line_lengths:
        return {
            "angle_deg": angle_deg, "mean_section_length_m": 0.0,
            "p20_section_length_m": 0.0, "short_section_ratio": 1.0,
            "fragmentation": math.inf, "coverage_span_m": math.inf,
            "regularity_penalty_m": math.inf, "shape_cost_m": math.inf,
        }
    values = np.asarray(line_lengths, dtype=float)
    weighted_lengths = [
        (length, width) for width, lengths in sections for length in lengths
    ]
    total_segment_weight = sum(width * len(lengths) for width, lengths in sections)
    active_width = sum(width for width, _lengths in sections)
    mean = sum(length * weight for length, weight in weighted_lengths) / max(
        total_segment_weight, 1e-9,
    )
    ordered_lengths = sorted(weighted_lengths)
    quantile_target = 0.20 * total_segment_weight
    cumulative = 0.0
    p20 = ordered_lengths[-1][0]
    for length, weight in ordered_lengths:
        cumulative += weight
        if cumulative >= quantile_target:
            p20 = length
            break
    short_limit = max(0.25 * float(values.max()), 2.0)
    short_measure = sum(
        length * weight for length, weight in weighted_lengths
        if length < short_limit
    )
    total_measure = sum(length * weight for length, weight in weighted_lengths)
    fragmentation = total_segment_weight / max(active_width, 1e-9)
    regularity = (geometry.convex_hull.area - geometry.area) / max(math.sqrt(geometry.area), 1e-9)
    coverage_span = geometry.area / max(mean, 1e-9)
    return {
        "angle_deg": float(angle_deg),
        "mean_section_length_m": mean,
        "p20_section_length_m": p20,
        "short_section_ratio": float(short_measure / max(total_measure, 1e-9)),
        "fragmentation": float(fragmentation),
        "coverage_span_m": float(coverage_span),
        "regularity_penalty_m": float(regularity),
        "shape_cost_m": float(coverage_span + 0.35 * regularity),
    }


def _work_shape_quality(geometry: BaseGeometry, directions: list[float] | None = None) -> dict[str, float]:
    """选择该几何最适合的作业方向并返回可审计的条带代理指标。"""
    if geometry.is_empty or geometry.area <= 0:
        return {
            "angle_deg": 0.0, "mean_section_length_m": 0.0,
            "p20_section_length_m": 0.0, "short_section_ratio": 1.0,
            "fragmentation": math.inf, "coverage_span_m": math.inf,
            "regularity_penalty_m": math.inf, "shape_cost_m": math.inf,
        }
    directions = directions or _scan_directions(geometry, (0.0, 90.0))
    choices = [_sweep_quality(geometry, angle) for angle in directions]
    return min(
        choices,
        key=lambda item: (
            item["shape_cost_m"], item["short_section_ratio"],
            item["fragmentation"], -item["p20_section_length_m"],
        ),
    )


def _angle_difference_deg(first: float, second: float) -> float:
    """比较无向条带方向差，单位度；方向相差180度视为同一条带方向。"""
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def _long_axis_angle_deg(geometry: BaseGeometry) -> float:
    """返回最小旋转包围矩形的长轴方向。"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        rectangle = geometry.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon" or rectangle.is_empty:
        # 极小数值碎片的旋转包络会退化为 LineString/Point。它们不是可接受
        # 工作区，但候选评分仍可能短暂遇到；回退到扫描方向以免 GEOS 警告污染
        # 批量日志或产生 NaN。
        return _scan_directions(geometry, (0.0, 90.0))[0]
    coordinates = list(rectangle.exterior.coords)
    edges = []
    for first, second in zip(coordinates, coordinates[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        edges.append((math.hypot(dx, dy), math.degrees(math.atan2(dy, dx)) % 180.0))
    return max(edges)[1]


def _regularity_metrics(
    geometry: BaseGeometry,
    working_width_m: float,
    quality: dict[str, float] | None = None,
) -> dict[str, float]:
    """用无量纲凹性、尺度化反身角和截线碎片度表达子区规则性。

    边界先按作业幅宽的 1/4 简化，避免把测量噪声当成需要分区的
    真实凹口。反身角只计外边界；孔洞对作业的影响由截线碎片度表达。
    """
    simplified = geometry.simplify(max(0.05, 0.50 * working_width_m), preserve_topology=True)
    hull_area = simplified.convex_hull.area
    concavity = (hull_area - simplified.area) / max(hull_area, 1e-9)
    events = _structure_events(geometry, working_width_m, 1e-9)
    reflex_count = 0
    for polygon in polygons(simplified):
        coordinates = list(polygon.exterior.coords)[:-1]
        orientation = 1 if polygon.exterior.is_ccw else -1
        for index, point in enumerate(coordinates):
            before, after = coordinates[index - 1], coordinates[(index + 1) % len(coordinates)]
            cross = ((point[0] - before[0]) * (after[1] - point[1])
                     - (point[1] - before[1]) * (after[0] - point[0]))
            scale = math.dist(before, point) * math.dist(point, after)
            if scale > 1e-9 and cross * orientation < -math.sin(math.radians(25)) * scale:
                reflex_count += 1
    quality = quality or _work_shape_quality(geometry)
    return {
        "concavity_ratio": float(concavity),
        "reflex_vertex_count": float(reflex_count),
        "significant_event_count": float(len(events)),
        "maximum_notch_depth_m": float(max((event.depth_m for event in events), default=0.0)),
        "fragmentation": float(quality["fragmentation"]),
    }


def _terminal_work_shape(metrics: dict[str, float]) -> bool:
    """判断子区是否足够规则；深细槽不能被低面积凹性提前吞掉。"""
    concavity = metrics["concavity_ratio"]
    reflex_count = metrics["reflex_vertex_count"]
    event_count = metrics.get("significant_event_count", 0.0)
    fragmentation = metrics["fragmentation"]
    events_resolved = event_count == 0 or (
        concavity <= 0.05 and fragmentation <= 1.10
    )
    return bool(
        events_resolved
        and (
            (reflex_count <= 1 and concavity <= 0.10 and fragmentation <= 1.15)
            or (concavity <= 0.040 and reflex_count <= 1 and fragmentation <= 1.35)
        )
    )


def _effective_region_width(geometry: BaseGeometry, epsilon: float) -> float:
    """用旋转包围矩形短轴和 4A/P 的较小值拦截细条及尖角碎片。"""
    if geometry.is_empty or geometry.area <= epsilon:
        return 0.0
    hull_coordinates = list(geometry.convex_hull.exterior.coords)
    oriented = math.inf
    for first, second in zip(hull_coordinates, hull_coordinates[1:]):
        if math.hypot(second[0] - first[0], second[1] - first[1]) <= epsilon:
            continue
        angle = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))
        xmin, ymin, xmax, ymax = affinity.rotate(geometry, -angle, origin=(0, 0)).bounds
        oriented = min(oriented, min(xmax - xmin, ymax - ymin))
    if not math.isfinite(oriented):
        oriented = 0.0
    return min(oriented, 4.0 * geometry.area / max(geometry.length, epsilon))


def _region_connectivity(
    geometry: BaseGeometry,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    minimum_core_area_m2: float,
) -> tuple[bool, int, int]:
    """检查候选区域与安全核心、直行筛查层的连通关系；返回筛查结果与分量计数，不能替代转弯认证。"""
    geometry_parts = _meaningful_components(geometry, geometry.area, 0.0)
    safe_parts = _meaningful_components(
        polygonal(geometry.intersection(safe_area)), geometry.area, minimum_core_area_m2,
    )
    straight_parts = _meaningful_components(
        polygonal(geometry.intersection(straight_area)), geometry.area, minimum_core_area_m2,
    )
    return len(geometry_parts) == 1, len(safe_parts), len(straight_parts)


def _materialize_region_core(
    geometry: BaseGeometry,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    minimum_core_area_m2: float,
) -> tuple[BaseGeometry, list[BaseGeometry], list[BaseGeometry]]:
    """从最终区域重新计算真实作业核心，而不继承父区的核心。

    ``_partition_from_cores`` 生成的 corridor 只是一层分区归属提示。它不能替代
    最终区域自己的车辆可达核心：一个递归子区可能恰好把父区的两个安全主体都
    切进来，表面上仍是单一 Polygon，内部可作业核心却已经断开。最终输出必须
    基于当前 ``geometry`` 与安全层的交集重新建立核心。

    返回 ``(work_core, safe_components, straight_components)``。优先使用完整包络
    核心；只有完整包络层为空时才退回基础直行层，并由调用方明确标为待连续运动
    验证。这里不把边界安全带或孔洞周边悄悄并入核心。
    """
    safe_parts = _meaningful_components(
        polygonal(geometry.intersection(safe_area)), geometry.area, minimum_core_area_m2,
    )
    straight_parts = _meaningful_components(
        polygonal(geometry.intersection(straight_area)), geometry.area, minimum_core_area_m2,
    )
    if len(safe_parts) == 1:
        return safe_parts[0], safe_parts, straight_parts
    # 完整包络安全核心分成多块时，基础直行层即使仍然连通，也只能
    # 证明参考点沿某个方向可以通过狭窄区，不能证明同一作业区内可以
    # 任意改变朝向。因此这里保留全部安全主体，让调用方拆分主体并把
    # straight-only 部分记为区域间通道；不能再用直行层伪装成单一安全核心。
    if safe_parts:
        return polygonal(unary_union(safe_parts)), safe_parts, straight_parts
    return polygonal(unary_union(straight_parts)), safe_parts, straight_parts


def _quality_core(
    geometry: BaseGeometry,
    scene: Scene,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
) -> BaseGeometry:
    """返回用于比较作业组织质量的实际车辆作业核心。

    ``work_region`` 仍承担 target 面积守恒：边界安全带、孔洞边缘和连接带都必须
    被明确分配，绝不能在分区时消失。但它们不是车辆能够连续组织条带的主体。
    若用整个外壳计算方向、凹性和扫掠长度，一段分给某主体的狭颈会把一个本来
    已规则的核心误判为 L 形；反过来，宽外壳也会掩盖核心中真正的分支。

    因此候选选择只用完整包络核心比较作业质量；完整包络为空时才使用单连通的
    直行核心，并由最终 ``access_status`` 标明尚需连续运动验证。候选的面积、
    连通性、孔洞保留和区域归属仍由完整 ``work_region`` 单独硬约束。
    """
    minimum_core_area = max(
        2.0 * scene.vehicle.working_width_m ** 2,
        scene.vehicle.working_width_m * scene.vehicle.body_width_m,
    )
    core, _safe_parts, _straight_parts = _materialize_region_core(
        geometry, safe_area, straight_area, minimum_core_area,
    )
    return core if not core.is_empty else geometry


def _split_disconnected_core_assignment(
    geometry: BaseGeometry,
    corridor: BaseGeometry,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    minimum_core_area_m2: float,
    corridor_half_width_m: float,
    epsilon: float,
    _depth: int = 0,
) -> list[tuple[BaseGeometry, BaseGeometry, BaseGeometry]]:
    """把含多个独立车辆核心的表面区域拆回可独立作业的区域。

    这是一个硬几何修复，而不是把问题标记为 review。优先按完整包络核心拆分；
    当该层为空时，再按基础直行核心拆分。若核心本身只有一个连通分量，则返回
    一个已重新计算核心的 assignment。区域之外的 target 面积保留在对应区域，
    因而不会因安全分析而改变覆盖目标。
    """
    core, safe_parts, straight_parts = _materialize_region_core(
        geometry, safe_area, straight_area, minimum_core_area_m2,
    )
    anchors = safe_parts if len(safe_parts) > 1 else (
        straight_parts if not safe_parts and len(straight_parts) > 1 else []
    )
    if not anchors:
        return [(geometry, core, corridor)]
    split_assignments = _partition_from_cores(
        geometry, anchors, corridor_half_width_m, epsilon,
    )
    normalized: list[tuple[BaseGeometry, BaseGeometry, BaseGeometry]] = []
    for child, _old_core, child_corridor in split_assignments:
        child_core, child_safe, _straight = _materialize_region_core(
            child, safe_area, straight_area, minimum_core_area_m2,
        )
        # 保留原通道归属与本次界面通道的并集，work_core 则始终来自实际安全层。
        inherited = polygonal(child.intersection(corridor)) if not corridor.is_empty else corridor
        straight_only = polygonal(child.intersection(straight_area.difference(safe_area)))
        assignment = (
            child,
            child_core,
            polygonal(unary_union([child_corridor, inherited, straight_only])),
        )
        # 一次 Voronoi/自然切线分配可能仍让某个子面包含两个安全主体。
        # 继续在该子面内部拆分，直到每个作业区只有一个完整包络核心；
        # 仅在几何没有实际缩小时停止，防止退化输入造成递归振荡。
        if (
            len(child_safe) > 1 and _depth < 6
            and child.area < geometry.area - max(epsilon, geometry.area * 1e-9)
        ):
            nested = _split_disconnected_core_assignment(
                child, assignment[2], safe_area, straight_area,
                minimum_core_area_m2, corridor_half_width_m, epsilon,
                _depth=_depth + 1,
            )
            if len(nested) > 1 or not nested[0][0].equals(child):
                normalized.extend(nested)
                continue
        normalized.append(assignment)
    return normalized


def _absorb_nonworkable_final_regions(
    assignments: list[tuple[BaseGeometry, BaseGeometry, BaseGeometry, str, str]],
    scene: Scene,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    minimum_core_area_m2: float,
    minimum_core_width_m: float,
) -> list[tuple[BaseGeometry, BaseGeometry, BaseGeometry, str, str]]:
    """在整个 target part 范围吸收不足以独立作业的最终小区。

    候选生成、递归和核心修复都可能新建区域，因此小片规则必须在
    全部后处理结束后再执行一次。吸收只沿真实公共界面进行，且合并后
    必须仍是单面、单作业核心。目标面积不会被删除，corridor 归属也一并保留。
    """
    epsilon = scene.settings.geometry_epsilon_m
    active = list(assignments)
    while len(active) > 1:
        measured = []
        for index, (geometry, _core, _corridor, _reason, _evidence) in enumerate(active):
            core, safe_parts, straight_parts = _materialize_region_core(
                geometry, safe_area, straight_area, minimum_core_area_m2 * 0.05,
            )
            width = _effective_region_width(core, epsilon)
            if (
                core.area < minimum_core_area_m2
                or width < minimum_core_width_m
                or not (len(safe_parts) == 1 or (not safe_parts and len(straight_parts) == 1))
            ):
                measured.append((core.area, width, index))
        if not measured:
            break
        _area, _width, index = min(measured)
        geometry, _core, corridor, reason, evidence = active[index]
        choices = []
        for other_index, other in enumerate(active):
            if other_index == index:
                continue
            other_geometry, _other_core, other_corridor, other_reason, other_evidence = other
            interface = _shared_interface(geometry, other_geometry, epsilon)
            if interface.length <= epsilon:
                continue
            combined = polygonal(unary_union([geometry, other_geometry]))
            if len(polygons(combined)) != 1:
                continue
            combined_core, safe_parts, straight_parts = _materialize_region_core(
                combined, safe_area, straight_area, minimum_core_area_m2 * 0.05,
            )
            if not (len(safe_parts) == 1 or (not safe_parts and len(straight_parts) == 1)):
                continue
            quality = _work_shape_quality(
                combined_core, _scan_directions(combined_core, scene.settings.angles_deg),
            )
            regularity = _regularity_metrics(
                combined_core, scene.vehicle.working_width_m, quality,
            )
            choices.append((
                regularity["concavity_ratio"], quality["fragmentation"],
                -interface.length, other_index, combined, combined_core,
                polygonal(unary_union([corridor, other_corridor])),
                other_reason, other_evidence,
            ))
        if choices:
            (_concavity, _fragmentation, _interface, other_index, combined, combined_core,
             combined_corridor, other_reason, other_evidence) = min(choices)
            merged = (
                combined, combined_core, combined_corridor, "FINAL_SMALL_REGION_MERGED",
                f"吸收不足独立作业的最终小区；原因 {reason}/{other_reason}；"
                f"{evidence}；{other_evidence}",
            )
        else:
            # 小区与邻区合并后如果出现两个不相连的安全核心，不能为了
            # 减少区域数而把它们冒充成一个作业区。这种窄小部分仍归属于
            # 相邻区域，但整体记为通道责任区：不删除 target 面积，也不再
            # 把一个不足车辆尺度的孤立安全块报成可独立作业的核心。
            # 只允许将真正的小残片整体改记为通道。大面积主体即使
            # 核心偏窄，也应继续作为独立作业区保留，留给后续车辆动力学
            # 或共享调头空间验证，不能在这一层把它整块吞掉。
            maximum_corridor_subject_area = max(
                8.0 * minimum_core_area_m2,
                0.02 * scene.target.area,
            )
            if geometry.area > maximum_corridor_subject_area:
                break
            corridor_choices = []
            for other_index, other in enumerate(active):
                if other_index == index:
                    continue
                (other_geometry, _other_core, other_corridor,
                 other_reason, other_evidence) = other
                interface = _shared_interface(geometry, other_geometry, epsilon)
                if interface.length <= epsilon:
                    continue
                combined = polygonal(unary_union([geometry, other_geometry]))
                if len(polygons(combined)) != 1:
                    continue
                combined_corridor = polygonal(unary_union([
                    corridor, other_corridor, geometry,
                ]))
                core_domain = polygonal(combined.difference(combined_corridor))
                combined_core, safe_parts, _straight_parts = _materialize_region_core(
                    core_domain, safe_area, straight_area, minimum_core_area_m2 * 0.05,
                )
                if (
                    len(safe_parts) != 1
                    or combined_core.area < minimum_core_area_m2
                    or _effective_region_width(combined_core, epsilon) < minimum_core_width_m
                ):
                    continue
                quality = _work_shape_quality(
                    combined_core,
                    _scan_directions(combined_core, scene.settings.angles_deg),
                )
                regularity = _regularity_metrics(
                    combined_core, scene.vehicle.working_width_m, quality,
                )
                corridor_choices.append((
                    regularity["concavity_ratio"], quality["fragmentation"],
                    -interface.length, other_index, combined, combined_core,
                    combined_corridor, other_reason, other_evidence,
                ))
            if not corridor_choices:
                break
            (_concavity, _fragmentation, _interface, other_index, combined, combined_core,
             combined_corridor, other_reason, other_evidence) = min(corridor_choices)
            merged = (
                combined, combined_core, combined_corridor,
                "FINAL_NONWORKABLE_SUBJECT_AS_CORRIDOR",
                f"不足独立作业尺度的安全主体改记为邻区通道；"
                f"原因 {reason}/{other_reason}；{evidence}；{other_evidence}",
            )
        active = [
            item for position, item in enumerate(active)
            if position not in {index, other_index}
        ] + [merged]
        active.sort(key=lambda item: _spatial_order(item[0]))
    return active


def _absorb_connector_regions(
    assignments: list[tuple[BaseGeometry, BaseGeometry, BaseGeometry, str, str]],
    scene: Scene,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    minimum_core_area_m2: float,
) -> list[tuple[BaseGeometry, BaseGeometry, BaseGeometry, str, str]]:
    """把夹在两个更大主体之间的细长连接区降级为 corridor。

    扫掠分解会把哑铃形田块的矩形连接带当成第三个规则作业区。司机实际会把
    两端主体分别组织作业，把中间带当成转场/共享空间。这里仅处理满足全部结构
    证据的区域：恰有两个互不直接相邻的邻区、两邻区都明显更大、两邻区质心位于
    候选区两侧、候选区本身细长且无需要独立解决的深凹结构。连接带的 target 面积
    仍归入一个邻区，但从该区 work_core 中扣除并写入 corridor_assignment。
    """
    epsilon = scene.settings.geometry_epsilon_m
    active = list(assignments)
    while len(active) > 2:
        bridge_choices = []
        for index, item in enumerate(active):
            geometry, _core, corridor, reason, evidence = item
            neighbors = [
                other_index for other_index, other in enumerate(active)
                if other_index != index
                and _shared_interface(geometry, other[0], epsilon).length > epsilon
            ]
            if len(neighbors) != 2:
                continue
            first_index, second_index = neighbors
            first, second = active[first_index][0], active[second_index][0]
            if _shared_interface(first, second, epsilon).length > epsilon:
                continue
            if geometry.area > 0.56 * min(first.area, second.area):
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                rectangle = geometry.minimum_rotated_rectangle
            coordinates = list(rectangle.exterior.coords) if rectangle.geom_type == "Polygon" else []
            edges = [math.dist(a, b) for a, b in zip(coordinates, coordinates[1:])]
            if not edges or min(edges) <= epsilon or max(edges) / min(edges) < 1.8:
                continue
            center = geometry.centroid
            first_vector = (first.centroid.x - center.x, first.centroid.y - center.y)
            second_vector = (second.centroid.x - center.x, second.centroid.y - center.y)
            denominator = math.hypot(*first_vector) * math.hypot(*second_vector)
            if denominator <= epsilon:
                continue
            cosine = (
                first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1]
            ) / denominator
            if cosine > -0.50:
                continue
            connector_core = _quality_core(geometry, scene, safe_area, straight_area)
            connector_quality = _work_shape_quality(
                connector_core, _scan_directions(connector_core, scene.settings.angles_deg),
            )
            connector_regularity = _regularity_metrics(
                connector_core, scene.vehicle.working_width_m, connector_quality,
            )
            if not _terminal_work_shape(connector_regularity):
                continue
            merge_options = []
            for other_index in neighbors:
                other_geometry, _other_core, other_corridor, other_reason, other_evidence = active[other_index]
                combined = polygonal(unary_union([geometry, other_geometry]))
                if len(polygons(combined)) != 1:
                    continue
                combined_corridor = polygonal(unary_union([
                    corridor, other_corridor, geometry,
                ]))
                core_domain = polygonal(combined.difference(combined_corridor))
                combined_core, safe_parts, straight_parts = _materialize_region_core(
                    core_domain, safe_area, straight_area, minimum_core_area_m2,
                )
                if len(safe_parts) != 1 or combined_core.area < minimum_core_area_m2:
                    continue
                quality = _work_shape_quality(
                    combined_core, _scan_directions(combined_core, scene.settings.angles_deg),
                )
                merge_options.append((
                    quality["shape_cost_m"], -_shared_interface(geometry, other_geometry, epsilon).length,
                    other_index, combined, combined_core, combined_corridor,
                    other_reason, other_evidence,
                ))
            if merge_options:
                bridge_choices.append((
                    geometry.area / max(min(first.area, second.area), epsilon),
                    index, min(merge_options), reason, evidence,
                ))
        if not bridge_choices:
            break
        _ratio, index, option, reason, evidence = min(bridge_choices)
        (_cost, _interface, other_index, combined, combined_core, combined_corridor,
         other_reason, other_evidence) = option
        merged = (
            combined, combined_core, combined_corridor, "CONNECTOR_ASSIGNED_TO_SUBJECT",
            f"细长中间区改记为两个主体间连接通道；原原因 {reason}/{other_reason}；"
            f"{evidence}；{other_evidence}",
        )
        active = [
            item for position, item in enumerate(active)
            if position not in {index, other_index}
        ] + [merged]
        active.sort(key=lambda item: _spatial_order(item[0]))
    return active


def _critical_sweep_cells(
    geometry: BaseGeometry,
    angle_deg: float,
    epsilon: float,
    minimum_spacing_m: float,
) -> list[BaseGeometry]:
    """仅在明显凹角和孔洞投影极值处切分，避免锯齿边界制造碎区。"""
    rotated = affinity.rotate(geometry, -angle_deg, origin=(0, 0))
    xmin, ymin, xmax, ymax = rotated.bounds
    events = {xmin, xmax}
    coords = list(rotated.exterior.coords)[:-1]
    orientation = 1 if rotated.exterior.is_ccw else -1
    for index, point in enumerate(coords):
        before, after = coords[index - 1], coords[(index + 1) % len(coords)]
        cross = ((point[0] - before[0]) * (after[1] - point[1])
                 - (point[1] - before[1]) * (after[0] - point[0]))
        scale = math.hypot(point[0] - before[0], point[1] - before[1]) * math.hypot(
            after[0] - point[0], after[1] - point[1]
        )
        if scale > epsilon and cross * orientation < -math.sin(math.radians(25)) * scale:
            events.add(point[0])
    for ring in rotated.interiors:
        events.update((ring.bounds[0], ring.bounds[2]))
    ordered = sorted(events)
    cuts = [ordered[0]]
    for value in ordered[1:-1]:
        if value - cuts[-1] >= minimum_spacing_m and ordered[-1] - value >= minimum_spacing_m:
            cuts.append(value)
    cuts.append(ordered[-1])
    cells: list[BaseGeometry] = []
    for left, right in zip(cuts, cuts[1:]):
        slab = rotated.intersection(box(left, ymin - 1, right, ymax + 1))
        cells.extend(
            affinity.rotate(part, angle_deg, origin=(0, 0))
            for part in polygons(polygonal(slab)) if part.area > epsilon * epsilon
        )
    return sorted(cells, key=_spatial_order)


def _reflex_extension_candidates(
    geometry: BaseGeometry,
    working_width_m: float,
    epsilon: float,
) -> list[list[BaseGeometry]]:
    """沿真实凹角的相邻边延长线生成二分候选。

    这类分界继续了凹口本身的几何方向，不会像全局等宽切片那样
    任意横切长边。只返回两个有面积、并且并集仍等于原区的候选。
    """
    simplified = geometry.simplify(max(0.05, 0.50 * working_width_m), preserve_topology=True)
    xmin, ymin, xmax, ymax = geometry.bounds
    reach = 4.0 * math.hypot(xmax - xmin, ymax - ymin) + 1.0
    candidates: list[list[BaseGeometry]] = []
    for polygon in polygons(simplified):
        coordinates = list(polygon.exterior.coords)[:-1]
        orientation = 1 if polygon.exterior.is_ccw else -1
        for index, point in enumerate(coordinates):
            before, after = coordinates[index - 1], coordinates[(index + 1) % len(coordinates)]
            cross = ((point[0] - before[0]) * (after[1] - point[1])
                     - (point[1] - before[1]) * (after[0] - point[0]))
            scale = math.dist(before, point) * math.dist(point, after)
            if scale <= epsilon or cross * orientation >= -math.sin(math.radians(25)) * scale:
                continue
            for neighbor in (before, after):
                dx, dy = point[0] - neighbor[0], point[1] - neighbor[1]
                length = math.hypot(dx, dy)
                if length <= epsilon:
                    continue
                ux, uy = dx / length, dy / length
                cutter = LineString([
                    (point[0] - reach * ux, point[1] - reach * uy),
                    (point[0] + reach * ux, point[1] + reach * uy),
                ])
                pieces = [part for part in polygons(polygonal(split(geometry, cutter)))
                          if part.area > epsilon * epsilon]
                if len(pieces) == 2:
                    candidates.append(sorted(pieces, key=_spatial_order))
    return candidates


def _local_chord(
    target: BaseGeometry,
    anchor: Point,
    angle_deg: float,
    epsilon: float,
) -> LineString | None:
    """返回经过结构事件、且只横断其所在局部田体的有限田内弦。"""
    xmin, ymin, xmax, ymax = target.bounds
    reach = 4.0 * math.hypot(xmax - xmin, ymax - ymin) + 1.0
    ux, uy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    infinite = LineString([
        (anchor.x - reach * ux, anchor.y - reach * uy),
        (anchor.x + reach * ux, anchor.y + reach * uy),
    ])
    inside = target.intersection(infinite)
    lines = [
        line for line in (list(inside.geoms) if hasattr(inside, "geoms") else [inside])
        if line.geom_type == "LineString" and line.length > epsilon
    ]
    if not lines:
        return None
    local = min(lines, key=lambda line: line.distance(anchor))
    first, last = local.coords[0], local.coords[-1]
    length = math.dist(first, last)
    if length <= epsilon:
        return None
    vx, vy = (last[0] - first[0]) / length, (last[1] - first[1]) / length
    extension = max(100.0 * epsilon, 1e-4)
    return LineString([
        (first[0] - extension * vx, first[1] - extension * vy),
        (last[0] + extension * vx, last[1] + extension * vy),
    ])


def _split_by_local_chords(
    geometry: BaseGeometry,
    cutters: tuple[LineString, ...],
    epsilon: float,
) -> list[BaseGeometry]:
    """依次应用少量有来源的局部弦，并验证目标面积没有丢失。"""
    active = [geometry]
    for cutter in cutters:
        updated: list[BaseGeometry] = []
        changed = False
        for part in active:
            pieces = [
                item for item in polygons(polygonal(split(part, cutter)))
                if item.area > epsilon * epsilon
            ]
            if len(pieces) > 1:
                updated.extend(pieces)
                changed = True
            else:
                updated.append(part)
        if not changed:
            return []
        active = updated
    union = polygonal(unary_union(active))
    if geometry.symmetric_difference(union).area > max(epsilon, geometry.area * 1e-9):
        return []
    return sorted(active, key=_spatial_order)


def _structure_event_candidates(
    geometry: BaseGeometry,
    analysis_geometry: BaseGeometry,
    working_width_m: float,
    epsilon: float,
) -> list[list[BaseGeometry]]:
    """由车辆核心上的深凹口提出有限二分和相邻平行多切线候选。

    每个事件只保留一条最短可行局部弦参加组合。组合仅发生在方向近似平行、
    锚点相互分离的事件之间，最多三刀；运行量由事件数而非全角度网格控制。
    """
    raw_events = _structure_events(analysis_geometry, working_width_m, epsilon)
    events: list[StructureEvent] = []
    for event in raw_events:
        if any(event.anchor.distance(previous.anchor) < working_width_m for previous in events):
            continue
        events.append(event)
        if len(events) >= 8:
            break
    candidates: list[list[BaseGeometry]] = []
    event_cutters: list[tuple[StructureEvent, LineString, float]] = []
    for event in events:
        choices: list[tuple[float, LineString, float, list[BaseGeometry]]] = []
        angles: list[float] = []
        for tangent in event.tangent_angles_deg:
            angles.extend((tangent, (tangent + 90.0) % 180.0))
        used_angles: list[float] = []
        for angle in angles:
            if any(_angle_difference_deg(angle, existing) < 3.0 for existing in used_angles):
                continue
            used_angles.append(angle)
            cutter = _local_chord(geometry, event.anchor, angle, epsilon)
            if cutter is None:
                continue
            parts = _split_by_local_chords(geometry, (cutter,), epsilon)
            if len(parts) != 2:
                continue
            choices.append((cutter.length, cutter, angle, parts))
            candidates.append(parts)
        if choices:
            _length, cutter, angle, _parts = min(choices, key=lambda item: item[0])
            event_cutters.append((event, cutter, angle))

    # 梳状/多臂田需要两三条相关切线共同表达；若要求每一刀先独立获益，
    # 中间状态会把真正的组合提前淘汰。只组合近似平行且锚点分离的最短弦。
    for count in (2, 3):
        for group in itertools.combinations(event_cutters[:8], count):
            angles = [item[2] for item in group]
            if max(_angle_difference_deg(angles[0], angle) for angle in angles[1:]) > 12.0:
                continue
            anchors = [item[0].anchor for item in group]
            if min(a.distance(b) for i, a in enumerate(anchors) for b in anchors[i + 1:]) < 2.0 * working_width_m:
                continue
            parts = _split_by_local_chords(
                geometry, tuple(item[1] for item in group), epsilon,
            )
            if len(parts) == count + 1:
                candidates.append(parts)
    return candidates


def _adaptive_erosion_candidates(
    geometry: BaseGeometry,
    working_width_m: float,
    minimum_region_area_m2: float,
    epsilon: float,
) -> list[list[BaseGeometry]]:
    """从田块的多尺度侵蚀骨架提出主体—狭颈分区候选。

    这不是按固定网格把田切碎。把田块逐级向内收缩后，宽主体会保留为较大的
    内核，狭颈和枝干根部会首先消失。只有在同一收缩尺度上出现两个以上足够大
    的主体时，才把这些主体作为 Voronoi 归属的锚点恢复到原 target。因而一个
    ``可通过但不适合连续作业`` 的细长连接可以形成两区和连接带，而一个只是
    边缘锯齿或很小凹口不会触发分区。

    收缩尺度同时含车辆绝对尺度和田块相对尺度，候选数量固定且很小；它是
    近似的 target 骨架，不是对所有切线、角度或组合的蛮力搜索。
    """
    if geometry.is_empty or geometry.area <= minimum_region_area_m2:
        return []
    extent = math.sqrt(max(geometry.area, 1e-9))
    # 车辆尺度控制最小可识别狭颈；相对尺度只提供上限，避免大田完全由
    # 一个 3.75m 幅宽决定。保持 5 个固定等级，运行成本与边界复杂度无关。
    lower = 0.50 * working_width_m
    # 不把上限锁死在三条作业带。对于 400 m 量级的田，30 m 的连接虽然
    # 绝对上很宽，却可能只占两侧主体宽度的一小部分；相对侵蚀仍应能把它
    # 识别成 neck。7 条作业带与 12% 等效尺度共同限制扫描范围；这个上界
    # 避免恰好 60 m 的宽连接在 30 m 腐蚀时被数值上的“刚好闭合”误判为瓶颈。
    upper = min(7.0 * working_width_m, 0.12 * extent)
    if upper <= lower + epsilon:
        return []
    radii = sorted({round(value, 6) for value in np.linspace(lower, upper, 5)})
    candidates: list[list[BaseGeometry]] = []
    previous_count = 1
    for radius in radii:
        eroded = polygonal(geometry.buffer(-radius, quad_segs=16))
        # 内核阈值按原田与车辆作业尺度共同设置，过滤侵蚀产生的角点碎片。
        cores = _meaningful_components(
            eroded, geometry.area, max(minimum_region_area_m2 * 0.30, working_width_m ** 2),
        )
        if len(cores) <= 1:
            previous_count = max(previous_count, len(cores))
            continue
        # 同一尺度出现多个主体才是一次稳定的骨架分叉。若后续尺度继续细分，
        # 仍把候选交给统一质量函数比较，避免把每一级都叠加成机械碎片。
        assignments = _partition_from_cores(
            geometry, cores, max(working_width_m / 2.0, radius / 2.0), epsilon,
        )
        parts = [part for part, _core, _corridor in assignments]
        if len(parts) >= 2:
            candidates.append(sorted(parts, key=_spatial_order))
        previous_count = len(cores)
    return candidates


def _critical_event_candidates(
    geometry: BaseGeometry,
    angle_deg: float,
    working_width_m: float,
    epsilon: float,
) -> list[list[BaseGeometry]]:
    """将每个凹角/孔洞临界位置单独当作候选切线，避免一次切碎复杂田块。"""
    rotated = affinity.rotate(geometry, -angle_deg, origin=(0, 0))
    simplified = rotated.simplify(max(0.05, 0.50 * working_width_m), preserve_topology=True)
    xmin, ymin, xmax, ymax = rotated.bounds
    events: list[float] = []
    for polygon in polygons(simplified):
        coordinates = list(polygon.exterior.coords)[:-1]
        orientation = 1 if polygon.exterior.is_ccw else -1
        for index, point in enumerate(coordinates):
            before, after = coordinates[index - 1], coordinates[(index + 1) % len(coordinates)]
            cross = ((point[0] - before[0]) * (after[1] - point[1])
                     - (point[1] - before[1]) * (after[0] - point[0]))
            scale = math.dist(before, point) * math.dist(point, after)
            if scale > epsilon and cross * orientation < -math.sin(math.radians(25)) * scale:
                events.append(point[0])
        for ring in polygon.interiors:
            events.extend((ring.bounds[0], ring.bounds[2]))
    candidates: list[list[BaseGeometry]] = []
    margin = max(xmax - xmin, ymax - ymin) + 1.0
    for value in sorted(set(round(item, 6) for item in events)):
        if value - xmin < working_width_m or xmax - value < working_width_m:
            continue
        cutter = LineString([(value, ymin - margin), (value, ymax + margin)])
        pieces = [
            affinity.rotate(part, angle_deg, origin=(0, 0))
            for part in polygons(polygonal(split(rotated, cutter)))
            if part.area > epsilon * epsilon
        ]
        if 2 <= len(pieces) <= 4:
            candidates.append(sorted(pieces, key=_spatial_order))
    return candidates


def _critical_halfplane_candidates(
    geometry: BaseGeometry,
    angle_deg: float,
    working_width_m: float,
    epsilon: float,
) -> list[list[BaseGeometry]]:
    """在凹角与孔洞临界位置构造左/右半平面分解。

    与 ``split`` 不同，候选线不必精确穿过原始边界顶点，因而对简化后
    的临界坐标和带孔多边形更稳定。
    """
    rotated = affinity.rotate(geometry, -angle_deg, origin=(0, 0))
    simplified = rotated.simplify(max(0.05, 0.50 * working_width_m), preserve_topology=True)
    xmin, ymin, xmax, ymax = rotated.bounds
    events: list[float] = []
    for polygon in polygons(simplified):
        coordinates = list(polygon.exterior.coords)[:-1]
        orientation = 1 if polygon.exterior.is_ccw else -1
        for index, point in enumerate(coordinates):
            before, after = coordinates[index - 1], coordinates[(index + 1) % len(coordinates)]
            cross = ((point[0] - before[0]) * (after[1] - point[1])
                     - (point[1] - before[1]) * (after[0] - point[0]))
            scale = math.dist(before, point) * math.dist(point, after)
            if scale > epsilon and cross * orientation < -math.sin(math.radians(25)) * scale:
                events.append(point[0])
        for ring in polygon.interiors:
            events.extend((ring.bounds[0], ring.bounds[2]))
    margin = max(xmax - xmin, ymax - ymin) + 1.0
    result: list[list[BaseGeometry]] = []
    for value in sorted(set(round(item, 6) for item in events)):
        if value - xmin < working_width_m or xmax - value < working_width_m:
            continue
        left = polygonal(rotated.intersection(box(xmin - margin, ymin - margin, value, ymax + margin)))
        right = polygonal(rotated.intersection(box(value, ymin - margin, xmax + margin, ymax + margin)))
        pieces = [
            affinity.rotate(part, angle_deg, origin=(0, 0))
            for side in (left, right) for part in polygons(side)
            if part.area > epsilon * epsilon
        ]
        if 2 <= len(pieces) <= 4:
            result.append(sorted(pieces, key=_spatial_order))
    return result


def _merge_candidate_levels(
    initial: list[BaseGeometry],
    epsilon: float,
) -> list[list[BaseGeometry]]:
    """沿真实公共接口合并单元并保留少区层级，收益在候选级统一计算。"""
    active = sorted(initial, key=_spatial_order)
    levels = [list(active)]
    while len(active) > 1:
        options = []
        for i, first in enumerate(active):
            for j in range(i + 1, len(active)):
                second = active[j]
                interface = _shared_interface(first, second, epsilon)
                if interface.length <= epsilon:
                    continue
                combined = polygonal(unary_union([first, second]))
                if len(polygons(combined)) != 1:
                    continue
                hull_area = combined.convex_hull.area
                concavity = (hull_area - combined.area) / max(hull_area, epsilon)
                options.append((concavity, -interface.length, i, j, combined))
        if not options:
            break
        _concavity, _interface, i, j, combined = min(options)
        active = [part for index, part in enumerate(active) if index not in {i, j}] + [combined]
        active.sort(key=_spatial_order)
        levels.append(list(active))
    return levels


def _retain_diverse_candidates(
    candidates: list[list[BaseGeometry]],
    working_width_m: float,
    epsilon: float,
    maximum: int = 72,
) -> list[list[BaseGeometry]]:
    """按互补几何证据剪枝，不再让面积均衡支配全部候选。

    每个区数同时保留面积尺度好、最大残余凹性低、人工接口短和最大主体完整
    的代表。这里只做便宜预筛；车辆核心与统一收益仍在正式评价器中检查。
    """
    descriptors: list[tuple[list[BaseGeometry], tuple[float, ...]]] = []
    for parts in candidates:
        total = sum(part.area for part in parts)
        concavities = []
        event_counts = []
        for part in parts:
            simplified = part.simplify(
                max(0.05, 0.50 * working_width_m), preserve_topology=True,
            )
            hull_area = simplified.convex_hull.area
            concavities.append(
                (hull_area - simplified.area) / max(hull_area, epsilon)
            )
            event_counts.append(len(_structure_events(part, working_width_m, epsilon)))
        interface = sum(
            _shared_interface(parts[i], parts[j], epsilon).length
            for i in range(len(parts)) for j in range(i + 1, len(parts))
        )
        descriptors.append((parts, (
            min(part.area for part in parts) / max(total, epsilon),
            max(concavities),
            float(max(event_counts, default=0)),
            interface,
            max(part.area for part in parts) / max(total, epsilon),
        )))

    selected: list[list[BaseGeometry]] = []
    selected_signatures: set[tuple[bytes, ...]] = set()
    for part_count in sorted({len(parts) for parts, _metrics in descriptors}):
        group = [item for item in descriptors if len(item[0]) == part_count]
        rankings = (
            sorted(group, key=lambda item: (-item[1][0], item[1][1], item[1][3])),
            sorted(group, key=lambda item: (item[1][1], item[1][2], item[1][3])),
            sorted(group, key=lambda item: (item[1][2], item[1][1], item[1][3])),
            sorted(group, key=lambda item: (item[1][3], item[1][1], -item[1][0])),
            sorted(group, key=lambda item: (item[1][4], item[1][1], item[1][3])),
        )
        for ranking in rankings:
            for parts, _metrics in ranking[:6]:
                signature = tuple(sorted(part.normalize().wkb for part in parts))
                if signature in selected_signatures:
                    continue
                selected_signatures.add(signature)
                selected.append(parts)
    return selected[:maximum]


def _absorb_nonworkable_cells(
    initial: list[BaseGeometry],
    minimum_area_m2: float,
    minimum_width_m: float,
    epsilon: float,
) -> list[BaseGeometry]:
    """将扫掠分解产生的小片/细片吸收到真实相邻主体。

    选择使合并后凹性最小的相邻区，比单纯按公共边长更接近
    “分界继续局部边界方向”的几何要求。
    """
    active = sorted(initial, key=_spatial_order)
    while len(active) > 1:
        invalid = [
            (part.area, index) for index, part in enumerate(active)
            if part.area < minimum_area_m2
            or _effective_region_width(part, epsilon) < minimum_width_m
        ]
        if not invalid:
            break
        _area, index = min(invalid)
        part = active[index]
        options = []
        for other_index, other in enumerate(active):
            if other_index == index:
                continue
            interface = _shared_interface(part, other, epsilon)
            if interface.length <= epsilon:
                continue
            combined = polygonal(unary_union([part, other]))
            if len(polygons(combined)) != 1:
                continue
            hull_area = combined.convex_hull.area
            concavity = (hull_area - combined.area) / max(hull_area, epsilon)
            options.append((concavity, -interface.length, other_index, combined))
        if not options:
            break
        _concavity, _interface, other_index, combined = min(options)
        active = [
            item for item_index, item in enumerate(active)
            if item_index not in {index, other_index}
        ] + [combined]
        active.sort(key=_spatial_order)
    return active


def _merge_compatible_work_regions(
    parts: list[BaseGeometry],
    scene: Scene,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    minimum_core_area_m2: float,
    minimum_regions: int = 1,
) -> list[BaseGeometry]:
    """撤销没有作业收益的内部分界，同时保护真实深凹口。

    方向相似不再是必要条件：它会漏掉“同一个规则主体被斜切成两块”。
    每一次合并都在实际车辆作业核心上重新计算代价。若公共界面正在分离深度超过
    四条幅宽的凹口，则保留该界面；唯一例外是不足整体 10% 的同向小附着块，
    它应吸收回相邻主体，不能单独获得作业区号。
    """
    epsilon = scene.settings.geometry_epsilon_m
    working_width = scene.vehicle.working_width_m

    def protects_deep_event(combined_core: BaseGeometry, interface: BaseGeometry) -> bool:
        return any(
            (
                event.depth_m >= 6.0 * working_width
                or (
                    event.depth_m >= 4.0 * working_width
                    and event.support_m >= 3.0 * working_width
                )
                or (
                    event.depth_m >= 4.0 * working_width
                    and event.support_m >= 2.0 * working_width
                    and interface.length <= 8.0 * working_width
                )
            )
            and event.anchor.distance(interface) <= 4.0 * working_width
            for event in _structure_events(combined_core, working_width, epsilon)
        )

    active = sorted(parts, key=_spatial_order)
    while len(active) > max(1, minimum_regions):
        active_core_area = sum(
            _quality_core(part, scene, safe_area, straight_area).area for part in active
        )
        choices = []
        for index, first in enumerate(active):
            first_core = _quality_core(first, scene, safe_area, straight_area)
            first_q = _work_shape_quality(
                first_core, _scan_directions(first_core, scene.settings.angles_deg),
            )
            for other_index, second in enumerate(active[index + 1:], index + 1):
                interface = _shared_interface(first, second, epsilon)
                if interface.length <= epsilon:
                    continue
                second_core = _quality_core(second, scene, safe_area, straight_area)
                second_q = _work_shape_quality(
                    second_core, _scan_directions(second_core, scene.settings.angles_deg),
                )
                combined = polygonal(unary_union([first, second]))
                if len(polygons(combined)) != 1:
                    continue
                connected, safe_count, straight_count = _region_connectivity(
                    combined, safe_area, straight_area, minimum_core_area_m2,
                )
                if not connected or not (
                    safe_count == 1 or (safe_count == 0 and straight_count == 1)
                ):
                    continue
                combined_core = _quality_core(combined, scene, safe_area, straight_area)
                merged_q = _work_shape_quality(
                    combined_core, _scan_directions(combined_core, scene.settings.angles_deg),
                )
                merged_regular = _regularity_metrics(
                    combined_core, working_width, merged_q,
                )
                first_regular = _regularity_metrics(
                    first_core, working_width, first_q,
                )
                second_regular = _regularity_metrics(
                    second_core, working_width, second_q,
                )
                before = first_q["shape_cost_m"] + second_q["shape_cost_m"] + 1.5 * working_width
                cost_ratio = merged_q["shape_cost_m"] / max(before, epsilon)
                direction_difference = _angle_difference_deg(
                    first_q["angle_deg"], second_q["angle_deg"],
                )
                relative_small_area = min(
                    first_core.area, second_core.area,
                ) / max(combined_core.area, epsilon)
                small_appendage = (
                    (
                        relative_small_area <= 0.10
                        and direction_difference <= 10.0
                        and cost_ratio <= 1.12
                    )
                    or (
                        relative_small_area <= 0.15
                        and cost_ratio <= 0.95
                    )
                    or (
                        0.08 <= min(first_core.area, second_core.area)
                        / max(active_core_area, epsilon) <= 0.11
                        and direction_difference >= 30.0
                        and cost_ratio <= 1.35
                    )
                )
                deep_event = protects_deep_event(combined_core, interface)
                coherent_pair_merge = (
                    not deep_event and direction_difference <= 20.0 and cost_ratio <= 1.08
                )
                fragmentation_ok = merged_q["fragmentation"] <= max(
                    first_q["fragmentation"], second_q["fragmentation"],
                ) + (0.75 if coherent_pair_merge else (0.50 if small_appendage else 0.30))
                redundant_interface = not deep_event and cost_ratio <= 0.98
                terminal_merge = (
                    not deep_event and _terminal_work_shape(merged_regular) and cost_ratio <= 1.04
                )
                if not fragmentation_ok or not (
                    small_appendage or redundant_interface or terminal_merge
                    or coherent_pair_merge
                ):
                    continue
                choices.append((
                    cost_ratio, -interface.length, index, other_index, combined,
                ))
        if not choices:
            break
        _cost, _interface, index, other_index, combined = min(choices)
        active = [part for position, part in enumerate(active) if position not in {index, other_index}]
        active.append(combined)
        active.sort(key=_spatial_order)
    return active


def _candidate_metrics(
    parts: list[BaseGeometry],
    directions: list[float],
    whole_quality: dict[str, float],
    scene: Scene,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    minimum_region_area_m2: float,
    minimum_region_width_m: float,
    quality_cache: dict[bytes, dict[str, float]],
) -> dict[str, Any] | None:
    """硬约束过滤后计算收益；返回 None 表示候选不可接受。"""
    epsilon = scene.settings.geometry_epsilon_m
    qualities = []
    quality_parts: list[BaseGeometry] = []
    total_candidate_area = sum(part.area for part in parts)
    for part in parts:
        connected, safe_count, straight_count = _region_connectivity(
            part, safe_area, straight_area, minimum_region_area_m2 * 0.05,
        )
        # 完整包络核心是充分条件，不是必要条件。对细长但基础直行
        # 空间单连通的子区允许保留，后续会标记为待连续车辆运动验证。
        if not connected or not (safe_count == 1 or straight_count == 1):
            return None
        if (part.area < max(minimum_region_area_m2, 0.04 * total_candidate_area)
                or _effective_region_width(part, epsilon) < minimum_region_width_m):
            return None
        quality_part = _quality_core(part, scene, safe_area, straight_area)
        # 最小作业尺度必须检查车辆真正能组织作业的核心，而不是
        # 含安全边带和连接带的分配外壳。否则一个看似足够大的区域可能
        # 只剩下不到两个幅宽的核心，却在最终输出中获得独立区号。
        if (
            quality_part.area < minimum_region_area_m2
            or _effective_region_width(quality_part, epsilon) < minimum_region_width_m
        ):
            return None
        key = quality_part.normalize().wkb
        if key not in quality_cache:
            # 子区必须用自己的车辆作业核心重新选局部方向，不能继承父区
            # 外壳的全局方向集合；否则 L/T 形主体的分解收益会被系统性低估。
            quality_cache[key] = _work_shape_quality(
                quality_part, _scan_directions(quality_part, scene.settings.angles_deg)
            )
        qualities.append(quality_cache[key])
        quality_parts.append(quality_part)
    if len(parts) == 1:
        regularity = _regularity_metrics(
            quality_parts[0], scene.vehicle.working_width_m, qualities[0],
        )
        terminal = _terminal_work_shape(regularity)
        return {
            "parts": parts,
            "qualities": qualities,
            "gain": 1.0,
            "direction_spread": 0.0,
            "total_span": qualities[0]["coverage_span_m"],
            "fragmentation": qualities[0]["fragmentation"],
            "boundary_length": 0.0,
            "boundary_misalignment": 0.0,
            "aligned_run_reward_m": 0.0,
            "regularities": [regularity],
            "terminal_child_count": int(terminal),
            "residual_nonterminal_area_m2": 0.0 if terminal else float(quality_parts[0].area),
            "largest_child_area_ratio": 1.0,
            "maximum_concavity": regularity["concavity_ratio"],
            "maximum_reflex_count": regularity["reflex_vertex_count"],
            "maximum_event_count": regularity["significant_event_count"],
            "maximum_notch_depth_m": regularity["maximum_notch_depth_m"],
            "objective": qualities[0]["shape_cost_m"],
        }
    substantial = [
        quality for part, quality in zip(quality_parts, qualities)
        if part.area >= 0.15 * sum(item.area for item in quality_parts)
    ]
    direction_spread = max(
        (_angle_difference_deg(a["angle_deg"], b["angle_deg"])
         for i, a in enumerate(substantial) for b in substantial[i + 1:]),
        default=0.0,
    )
    total_area = sum(part.area for part in quality_parts)
    total_span = sum(quality["coverage_span_m"] for quality in qualities)
    harmonic_mean = total_area / max(total_span, epsilon)
    gain = harmonic_mean / max(whole_quality["mean_section_length_m"], epsilon)
    fragmentation = max(quality["fragmentation"] for quality in qualities)
    boundary_length = sum(
        _shared_interface(parts[i], parts[j], epsilon).length
        for i in range(len(parts)) for j in range(i + 1, len(parts))
    )
    regularities = [
        _regularity_metrics(part, scene.vehicle.working_width_m, quality)
        for part, quality in zip(quality_parts, qualities)
    ]
    axes = [_long_axis_angle_deg(part) for part in quality_parts]
    axis_lengths = []
    for part in quality_parts:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rectangle = part.minimum_rotated_rectangle
        coordinates = list(rectangle.exterior.coords) if rectangle.geom_type == "Polygon" else []
        axis_lengths.append(
            max((math.dist(a, b) for a, b in zip(coordinates, coordinates[1:])), default=0.0)
        )
    boundary_misalignment = 0.0
    structure_misalignment = 0.0
    unsupported_interface_length = 0.0
    aligned_run_reward = 0.0
    combined_quality = polygonal(unary_union(quality_parts))
    parent_events = _structure_events(
        combined_quality, scene.vehicle.working_width_m, epsilon,
    )
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            interface = _shared_interface(parts[i], parts[j], epsilon)
            lines = list(interface.geoms) if hasattr(interface, "geoms") else [interface]
            for line in lines:
                if line.geom_type != "LineString" or line.length <= epsilon:
                    continue
                first, last = line.coords[0], line.coords[-1]
                angle = math.degrees(math.atan2(last[1] - first[1], last[0] - first[0])) % 180.0
                # 分界应延续至少一侧子区的主轴/长边方向。配合下方
                # “主轴平行分界”候选，可避免在 L 形拐角中任意横切两条长臂。
                error = min(
                    _angle_difference_deg(angle, axes[i]),
                    _angle_difference_deg(angle, axes[j]),
                )
                boundary_misalignment += line.length * math.sin(math.radians(error))
                nearby = [
                    event for event in parent_events
                    if event.anchor.distance(line) <= 3.0 * scene.vehicle.working_width_m
                ]
                if nearby:
                    local_error = min(
                        _angle_difference_deg(angle, tangent)
                        for event in nearby for tangent in event.tangent_angles_deg
                    )
                    structure_misalignment += line.length * math.sin(math.radians(local_error))
                else:
                    # 一条远离任何凹口、狭颈或分支根部的长切线，
                    # 通常只是目标函数的代理产物，不是司机会采用的分界。
                    unsupported_interface_length += line.length
                aligned_run_reward = max([
                    aligned_run_reward,
                    *(
                        axis_lengths[index]
                        for index in (i, j)
                        if _angle_difference_deg(angle, axes[index]) <= 10.0
                    ),
                ])
    # coverage_span 是各区需要的扫掠宽度代理；规则性项惩罚深凹；
    # 新增区和方向不对齐的分界用幅宽无量纲化后计入组织成本。
    objective = (
        total_span
        + 0.35 * sum(quality["regularity_penalty_m"] for quality in qualities)
        + 1.5 * scene.vehicle.working_width_m * (len(parts) - 1)
        + 0.03 * boundary_length
        + 0.15 * boundary_misalignment
        + 0.20 * structure_misalignment
        + 0.08 * unsupported_interface_length
    )
    return {
        "parts": parts,
        "quality_parts": quality_parts,
        "qualities": qualities,
        "gain": gain,
        "direction_spread": direction_spread,
        "total_span": total_span,
        "fragmentation": fragmentation,
        "boundary_length": boundary_length,
        "boundary_misalignment": boundary_misalignment,
        "structure_misalignment": structure_misalignment,
        "unsupported_interface_length": unsupported_interface_length,
        "aligned_run_reward_m": aligned_run_reward,
        "regularities": regularities,
        "terminal_child_count": sum(_terminal_work_shape(item) for item in regularities),
        "residual_nonterminal_area_m2": sum(
            part.area for part, regularity in zip(quality_parts, regularities)
            if not _terminal_work_shape(regularity)
        ),
        # 分子和分母都使用车辆作业核心。旧版用外壳面积除以
        # 核心面积，会产生大于 1 的比率并误杀有效候选。
        "largest_child_area_ratio": max(part.area for part in quality_parts) / max(total_area, epsilon),
        "maximum_concavity": max(item["concavity_ratio"] for item in regularities),
        "maximum_reflex_count": max(item["reflex_vertex_count"] for item in regularities),
        "maximum_event_count": max(item["significant_event_count"] for item in regularities),
        "maximum_notch_depth_m": max(item["maximum_notch_depth_m"] for item in regularities),
        "objective": objective,
    }


def _benefit_partition_once(
    geometry: BaseGeometry,
    scene: Scene,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    turning_clearance_m: float,
) -> tuple[list[BaseGeometry], str]:
    """对一个父区做一层分解；递归由 :func:`_benefit_partition` 管理。"""
    epsilon = scene.settings.geometry_epsilon_m
    polygon = polygons(geometry)[0]
    quality_geometry = _quality_core(geometry, scene, safe_area, straight_area)
    directions = _scan_directions(quality_geometry, scene.settings.angles_deg)
    whole_quality = _work_shape_quality(quality_geometry, directions)
    working_width = scene.vehicle.working_width_m
    whole_regularity = _regularity_metrics(quality_geometry, working_width, whole_quality)
    # 孔洞环本身不自动触发分区；但当实际截线已显示出明显中断时，不能因
    # 外边界近似凸形而在这里提前退出。后面只接受能实质降低碎片度的孔洞候选。
    hole_interruption = bool(polygon.interiors) and whole_quality["fragmentation"] >= 1.15
    if _terminal_work_shape(whole_regularity) and not hole_interruption:
        return [geometry], (
            f"子区已达终止规则性：凹性 {whole_regularity['concavity_ratio']:.3f}，"
            f"反身角 {int(whole_regularity['reflex_vertex_count'])}，"
            f"碎片度 {whole_regularity['fragmentation']:.3f}"
        )
    minimum_region_area = max(6 * working_width ** 2, 2 * turning_clearance_m * working_width)
    minimum_region_width = 2 * working_width
    spacing = max(2 * working_width, scene.vehicle.body_width_m + 2 * scene.vehicle.safety_margin_m)
    candidates: list[list[BaseGeometry]] = [[geometry]]
    candidates.extend(_adaptive_erosion_candidates(
        geometry, working_width, minimum_region_area, epsilon,
    ))
    candidates.extend(_reflex_extension_candidates(geometry, working_width, epsilon))
    # target 外环很平滑时，车辆退让后的真实作业核心仍可能形成深细槽。
    # 这类结构不能被 target 的低面积凹性遮住；切线锚点来自核心，分割仍
    # 作用于原始 target，因此不会修改作业义务或填平孔洞。
    candidates.extend(_structure_event_candidates(
        geometry, quality_geometry, working_width, epsilon,
    ))
    for angle in directions:
        candidates.extend(
            _critical_event_candidates(geometry, angle, working_width, epsilon)
        )
        candidates.extend(
            _critical_halfplane_candidates(geometry, angle, working_width, epsilon)
        )
        # 同时考察与局部主轴平行的分界。上一项的切线与 angle
        # 垂直；旋转 90° 后则生成与长边平行的分支根部候选。
        candidates.extend(
            _critical_halfplane_candidates(
                geometry, (angle + 90.0) % 180.0, working_width, epsilon,
            )
        )
        cells = _critical_sweep_cells(geometry, angle, epsilon, spacing)
        if 1 < len(cells) <= 30:
            cells = _absorb_nonworkable_cells(
                cells, minimum_region_area, minimum_region_width, epsilon,
            )
            candidates.extend(
                level for level in _merge_candidate_levels(cells, epsilon)
                if 2 <= len(level) <= 6
            )
    unique: list[list[BaseGeometry]] = []
    seen: set[tuple[bytes, ...]] = set()
    for parts in candidates:
        signature = tuple(sorted(part.normalize().wkb for part in parts))
        if signature not in seen:
            seen.add(signature)
            unique.append(sorted(parts, key=_spatial_order))
    unique = _retain_diverse_candidates(unique, working_width, epsilon)
    accepted = []
    quality_cache: dict[bytes, dict[str, float]] = {
        quality_geometry.normalize().wkb: whole_quality,
    }
    for parts in unique:
        candidate_union = polygonal(unary_union(parts))
        if candidate_union.symmetric_difference(geometry).area > max(
            epsilon, geometry.area * 1e-9,
        ):
            continue
        metrics = _candidate_metrics(
            parts, directions, whole_quality, scene, safe_area, straight_area,
            minimum_region_area, minimum_region_width, quality_cache,
        )
        if metrics is None or len(parts) == 1:
            continue
        span_gain = metrics["total_span"] / max(whole_quality["coverage_span_m"], epsilon)
        fragmentation_improved = metrics["fragmentation"] <= 0.9 * whole_quality["fragmentation"]
        parent_objective = whole_quality["shape_cost_m"]
        objective_gain = parent_objective / max(metrics["objective"], epsilon)
        directional_benefit = (
            metrics["direction_spread"] >= 15.0
            and objective_gain >= 1.04
            and (span_gain <= 0.97 or fragmentation_improved)
        )
        residual_improved = (
            metrics["maximum_concavity"]
            <= max(0.12, 0.88 * whole_regularity["concavity_ratio"])
            and metrics["maximum_reflex_count"] < whole_regularity["reflex_vertex_count"]
        )
        # 深 L/T/U 形的“子区仍不规则”本身是一项结构性失败。当父区
        # 凹性高于 0.18，且候选把每个子区降到 0.12 以下并减少反身角时，
        # 允许以规则性改善接受；这不再受单一平均截线指标否决。
        terminalizes_parent = (
            not _terminal_work_shape(whole_regularity)
            and metrics["terminal_child_count"] == len(parts)
        )
        structural_benefit = (
            residual_improved and (
            (whole_regularity["concavity_ratio"] >= 0.18
             and metrics["maximum_concavity"]
             <= 0.90 * whole_regularity["concavity_ratio"])
            or objective_gain >= 1.01
            or fragmentation_improved
            ) and (
            # 没有方向差、没有碎片度改善且总代价变差的“规则化”，通常只是把
            # 一个宽连接的整体田块切成多个相同方向的矩形。它不改善司机作业，
            # 也不应借由反身角计数绕过代价约束。
            objective_gain >= 1.0
            or (metrics["direction_spread"] >= 20.0 and fragmentation_improved)
            )
            # 带孔整田对截线中断天然更敏感。孔洞候选若连统一作业代价
            # 都明显变差，不能再借“反身角变少”绕过孔洞屏障的专用门槛。
            and (not polygon.interiors or objective_gain >= 0.90)
        ) or (
            # C/U 形不一定需要改变条带方向，但若少量区域能把每一块都变为
            # 规则且连通的作业核心，总代价没有实质恶化，也应接受这个结构改进。
            terminalizes_parent and objective_gain >= 0.94 and len(parts) <= 3
        )
        hole_barrier_benefit = (
            bool(polygon.interiors)
            and whole_quality["fragmentation"] >= 1.15
            # 孔洞本身不是分区理由。只有它造成的中断被明显消除，
            # 且新区确实形成不同作业方向时，才以孔洞屏障接受。
            and metrics["fragmentation"] <= 0.72 * whole_quality["fragmentation"]
            and metrics["direction_spread"] >= 20.0
            and objective_gain >= 0.84
            and len(parts) <= 4
        )
        branch_root_benefit = (
            whole_regularity["reflex_vertex_count"] >= 3
            and metrics["maximum_reflex_count"] <= whole_regularity["reflex_vertex_count"] - 2
            and objective_gain >= 1.0
            and len(parts) <= 3
        )
        # 连续性改善单独作为结构收益：对作业方向相同、但在局部长度上
        # 持续被中断的田块，只要碎片度实质下降且凹性不变坏，就允许
        # 少量分区。这覆盖了“平行长臂”，避免把方向差当成唯一证据。
        continuity_benefit = (
            metrics["fragmentation"] <= 0.82 * whole_quality["fragmentation"]
            and metrics["maximum_concavity"] <= 1.05 * whole_regularity["concavity_ratio"]
            and metrics["largest_child_area_ratio"] <= 0.88
            and objective_gain >= 0.85
            and len(parts) <= 4
        )
        # 深凹田的两个主作业方向可能使分区后的 coverage-span 之和
        # 变大，但最难区域的凹性会显著降低。这里使用辞典式门槛，
        # 不用一个大权重抵消结构证据。
        deep_concavity_direction_benefit = (
            whole_regularity["concavity_ratio"] >= 0.18
            and metrics["maximum_concavity"] <= 0.60 * whole_regularity["concavity_ratio"]
            and metrics["direction_spread"] >= 20.0
            and objective_gain >= 0.82
            and len(parts) <= 3
        )
        # 平行长臂或同向分支的局部主轴接近，因此方向差可以接近0度；
        # 它们的真实结构证据是多个反身角和长凹槽。对这类父区，只要
        # 少量分区将最难子区的凹性降低至原来的75%以下，且没有一个子区
        # 独占85%以上面积，就视为主体组织改善。
        branch_system_benefit = (
            not polygon.interiors
            and whole_regularity["reflex_vertex_count"] >= 5
            and metrics["maximum_concavity"] <= 0.75 * whole_regularity["concavity_ratio"]
            and metrics["largest_child_area_ratio"] <= 0.85
            and objective_gain >= 0.75
            and len(parts) <= 4
        )
        linked_branch_benefit = (
            whole_regularity["concavity_ratio"] >= 0.20
            and whole_regularity["reflex_vertex_count"] >= 5
            and metrics["maximum_concavity"] <= 0.80 * whole_regularity["concavity_ratio"]
            and metrics["maximum_event_count"] <= math.ceil(
                0.50 * whole_regularity["significant_event_count"],
            )
            and objective_gain >= 0.88
            and len(parts) <= 3
        )
        # 某些深 L/U 形两侧的最佳条带方向相同。此时方向差不是
        # 分区的必要证据；最难子区的凹性和碎片度才是。
        same_direction_shape_benefit = (
            whole_regularity["concavity_ratio"] >= 0.18
            and metrics["maximum_concavity"] <= 0.40 * whole_regularity["concavity_ratio"]
            and metrics["fragmentation"] <= 1.05 * whole_quality["fragmentation"]
            and objective_gain >= 0.88
            and len(parts) <= 3
        )
        # 外边界面积凹性很低时，车辆退让后仍可能出现有尺度的深凹口。
        # 只要候选确实减少了未解凹口，就允许小幅的代理代价波动。
        core_notch_benefit = (
            whole_regularity["significant_event_count"] > 0
            and (
                metrics["maximum_event_count"] < whole_regularity["significant_event_count"]
                or metrics["maximum_notch_depth_m"]
                <= 0.75 * whole_regularity["maximum_notch_depth_m"]
            )
            and metrics["maximum_concavity"] <= max(
                0.12, 1.05 * whole_regularity["concavity_ratio"],
            )
            and (
                objective_gain >= 0.92
                or (
                    whole_regularity["significant_event_count"] <= 2
                    and whole_regularity["maximum_notch_depth_m"] >= 8.0 * working_width
                    and objective_gain >= 0.82
                )
            )
            and len(parts) <= 3
        )
        if (directional_benefit or structural_benefit or hole_barrier_benefit
                or branch_root_benefit or continuity_benefit
                or deep_concavity_direction_benefit or branch_system_benefit
                or linked_branch_benefit or same_direction_shape_benefit
                or core_notch_benefit):
            metrics["objective_gain"] = objective_gain
            accepted.append(metrics)
    if not accepted:
        return [geometry], "分区候选未同时达到方向差、连续作业收益和最小作业规模门槛"
    # 先解决最难子区，再比较整体平均代价。若某些候选已把最大凹性
    # 降到0.05以下，选用其中区数最少的一个，这正是“达到可作业即停止”。
    terminal_concavity = [row for row in accepted if row["maximum_concavity"] <= 0.05]
    if terminal_concavity:
        # 将凹性从“已经足够低”继续压到极小值，不值得额外增加
        # 第五个作业区。若四区以内已把最大残余凹性压到 0.12，
        # 且碎片度没有明显恶化，就先在这些紧凑候选中停止。
        terminal_best = min(
            terminal_concavity,
            key=lambda row: (
                len(row["parts"]), row["maximum_concavity"], row["objective"],
                -row["aligned_run_reward_m"], row["boundary_length"],
            ),
        )
        compact_terminal = [
            row for row in accepted
            if len(row["parts"]) <= 4
            and row["maximum_concavity"] <= 0.12
            and row["fragmentation"] <= 1.07 * whole_quality["fragmentation"]
        ]
        # 三区或四区的低凹性解已经很紧凑，不应用更弱的二区
        # 候选替换；该上限只用于拦截“为追求极小凹性增加第五区”。
        terminal_pool = (
            compact_terminal
            if len(terminal_best["parts"]) > 4 and compact_terminal
            else terminal_concavity
        )
        best = min(
            terminal_pool,
            key=lambda row: (
                len(row["parts"]), row["maximum_concavity"], row["objective"],
                -row["aligned_run_reward_m"], row["boundary_length"],
            ),
        )
    else:
        # 先找到“足够规则”的最小分区，不为消除最后一两个
        # 低影响事件再增加作业区。这是事件解决量的停止条件，
        # 同时要求碎片度不恶化，避免以低凹性换来更多中断。
        sufficient_structure = [
            row for row in accepted
            if row["maximum_event_count"] <= math.ceil(
                0.50 * whole_regularity["significant_event_count"],
            ) and (
                (
                    row["maximum_concavity"] <= 0.19
                    and row["fragmentation"] <= 1.07 * whole_quality["fragmentation"]
                )
                or (
                    row["maximum_concavity"] <= min(
                        0.22, 0.75 * whole_regularity["concavity_ratio"],
                    )
                    and row["fragmentation"] <= whole_quality["fragmentation"]
                )
            )
        ]
        if sufficient_structure:
            # 4–6 个有效事件且整体凹性很高，通常对应三条真实作业臂。
            # 此时先选残余结构最小的候选，再以区数破平；否则“最少区优先”
            # 会把仍含两条臂的主体过早合并成二分方案。
            strong_branch_system = (
                whole_regularity["concavity_ratio"] >= 0.30
                and 4 <= whole_regularity["significant_event_count"] <= 6
            )
            best = min(
                sufficient_structure,
                key=(
                    (lambda row: (
                        row["maximum_event_count"], row["maximum_concavity"],
                        row["fragmentation"], len(row["parts"]), row["objective"],
                        row["boundary_length"],
                    ))
                    if strong_branch_system else
                    (lambda row: (
                        len(row["parts"]), row["maximum_event_count"],
                        row["maximum_concavity"], row["objective"], row["boundary_length"],
                    ))
                ),
            )
        else:
        # 多条平行长臂可能不能一次降到凸形门槛。此时只在已通过
        # branch-system 硬尺度的候选中，选择保留长臂最完整的最多四区方案。
            parallel_arms = [
                row for row in accepted
                if whole_regularity["concavity_ratio"] >= 0.20
                and whole_regularity["reflex_vertex_count"] >= 5
                and row["direction_spread"] <= 5.0
                and row["maximum_concavity"] <= 0.75 * whole_regularity["concavity_ratio"]
                and len(row["parts"]) <= 4
            ]
            if parallel_arms:
                minimum_events = min(row["maximum_event_count"] for row in parallel_arms)
                event_equivalent = [
                    row for row in parallel_arms
                    if row["maximum_event_count"] <= minimum_events
                ]
                best = min(
                    event_equivalent,
                    key=lambda row: (
                        len(row["parts"]), row["maximum_concavity"],
                        -row["aligned_run_reward_m"], row["objective"], row["boundary_length"],
                    ),
                )
            else:
                best_objective = min(row["objective"] for row in accepted)
                near_optimal = [row for row in accepted if row["objective"] <= 1.01 * best_objective]
                best = min(
                    near_optimal,
                    key=lambda row: (
                        len(row["parts"]), -row["aligned_run_reward_m"], row["objective"],
                        row["maximum_concavity"], row["maximum_reflex_count"], row["boundary_length"],
                    ),
                )
    return best["parts"], (
        f"局部方向差 {best['direction_spread']:.1f}°，数学目标改善比 "
        f"{best['objective_gain']:.3f}，最大残余凹性 {best['maximum_concavity']:.3f}"
    )


def _benefit_partition(
    geometry: BaseGeometry,
    scene: Scene,
    safe_area: BaseGeometry,
    straight_area: BaseGeometry,
    turning_clearance_m: float,
    maximum_depth: int = 3,
    maximum_regions: int = 6,
) -> tuple[list[BaseGeometry], str]:
    """以整组叶区为单位递归分解，而不是让每个子区各自贪心切分。

    局部看来有收益的连续切分，合起来可能只增加区域数、缩短全部作业段。
    因此每一轮都把“替换一个叶区后的全部叶区”重新放进同一目标函数，只有
    整体组织成本下降，或未终止的复杂面积显著减少且成本没有明显恶化时，才
    接受切分。``maximum_regions`` 约束的是这一次调用的全部叶区总数。
    """
    epsilon = scene.settings.geometry_epsilon_m
    working_width = scene.vehicle.working_width_m
    minimum_region_area = max(6 * working_width ** 2, 2 * turning_clearance_m * working_width)
    minimum_region_width = 2 * working_width
    quality_geometry = _quality_core(geometry, scene, safe_area, straight_area)
    directions = _scan_directions(quality_geometry, scene.settings.angles_deg)
    whole_quality = _work_shape_quality(quality_geometry, directions)
    whole_regularity = _regularity_metrics(
        quality_geometry, working_width, whole_quality,
    )
    quality_cache: dict[bytes, dict[str, float]] = {
        quality_geometry.normalize().wkb: whole_quality,
    }

    def metrics_for(parts: list[BaseGeometry]) -> dict[str, Any] | None:
        candidate_union = polygonal(unary_union(parts))
        if candidate_union.symmetric_difference(geometry).area > max(
            epsilon, geometry.area * 1e-9,
        ):
            return None
        return _candidate_metrics(
            parts, directions, whole_quality, scene, safe_area, straight_area,
            minimum_region_area, minimum_region_width, quality_cache,
        )

    leaves = [geometry]
    evidence: list[str] = []
    last_rejection_reason = "没有可接受的分区候选"
    current = metrics_for(leaves)
    if current is None:
        return leaves, "整区没有满足车辆尺度和连通约束的候选分区"
    for depth in range(maximum_depth):
        if (
            depth > 0 and len(leaves) >= 2
            and current["maximum_concavity"] <= 0.15
            and current["fragmentation"] <= 1.20
        ):
            evidence.append("当前各区已达规则性停止条件，不再追求消除低影响凹口")
            break
        proposals: list[tuple[tuple[float, ...], list[BaseGeometry], dict[str, Any], str, float, float]] = []
        for index, leaf in enumerate(leaves):
            parts, reason = _benefit_partition_once(
                leaf, scene, safe_area, straight_area, turning_clearance_m,
            )
            last_rejection_reason = reason
            if len(parts) <= 1 or len(leaves) - 1 + len(parts) > maximum_regions:
                continue
            proposal_parts = sorted(leaves[:index] + parts + leaves[index + 1:], key=_spatial_order)
            proposal = metrics_for(proposal_parts)
            if proposal is None:
                continue
            objective_gain = current["objective"] / max(proposal["objective"], epsilon)
            residual_reduction = (
                current["residual_nonterminal_area_m2"] - proposal["residual_nonterminal_area_m2"]
            ) / max(quality_geometry.area, epsilon)
            # 方向或截线收益必须在整组叶区上成立。例外只给予真正消除了
            # 大片非终止 L/T/U 形的分解。该类切分会增加一条分区接口，因而
            # 代理组织成本可以上升，但不得超过约 1/0.70；否则宁可输出
            # ``RESIDUAL`` 供人工复核，也不把田块机械切碎。
            objective_improves = objective_gain >= 1.015
            structural_improves = (
                # 一个候选只要把至少 5% 的原田面积从“仍需分区”的
                # 非终止形状转为终止形状，就有足够的几何意义进入下一
                # 轮全局比较；全局区域上限阻止这种渐进修正规则切碎田块。
                residual_reduction >= 0.05
                and (
                    (
                        objective_gain >= 1.0
                        and (
                            proposal["direction_spread"] >= 15.0
                            or proposal["fragmentation"] <= current["fragmentation"] * 0.90
                        )
                    )
                    or (
                        proposal["terminal_child_count"] == len(proposal_parts)
                        and objective_gain >= 0.94
                        and len(proposal_parts) <= 3
                    )
                    or (
                        proposal["direction_spread"] >= 20.0
                        and proposal["fragmentation"] <= current["fragmentation"] * 0.90
                        and objective_gain >= 0.60
                    )
                    or (
                        proposal["fragmentation"] <= current["fragmentation"] * 0.82
                        and proposal["maximum_concavity"] <= current["maximum_concavity"] * 1.05
                        and proposal["largest_child_area_ratio"] <= 0.88
                        and objective_gain >= 0.85
                    )
                )
            )
            branch_root_improves = (
                current["maximum_reflex_count"] >= 3
                and proposal["maximum_reflex_count"] <= current["maximum_reflex_count"] - 2
                and objective_gain >= 1.0
                and len(proposal_parts) <= 3
            )
            hole_interrupt_improves = (
                any(part.interiors for part in polygons(leaf))
                and current["fragmentation"] >= 1.15
                and proposal["fragmentation"] <= 0.88 * current["fragmentation"]
                and objective_gain >= 0.84
            )
            deep_concavity_improves = (
                current["maximum_concavity"] >= 0.18
                and proposal["maximum_concavity"] <= 0.60 * current["maximum_concavity"]
                and proposal["direction_spread"] >= 20.0
                and objective_gain >= 0.90
                and residual_reduction >= 0.05
                and len(proposal_parts) <= 3
            )
            branch_system_improves = (
                current["maximum_reflex_count"] >= 5
                and proposal["maximum_concavity"] <= 0.75 * current["maximum_concavity"]
                and proposal["largest_child_area_ratio"] <= 0.85
                and objective_gain >= 0.75
                and len(proposal_parts) <= maximum_regions
            )
            same_direction_shape_improves = (
                current["maximum_concavity"] >= 0.18
                and proposal["maximum_concavity"] <= 0.40 * current["maximum_concavity"]
                and proposal["fragmentation"] <= 1.05 * current["fragmentation"]
                and objective_gain >= 0.88
                and len(proposal_parts) <= maximum_regions
            )
            core_notch_improves = (
                current["maximum_event_count"] > 0
                and (
                    proposal["maximum_event_count"] < current["maximum_event_count"]
                    or proposal["maximum_notch_depth_m"]
                    <= 0.75 * current["maximum_notch_depth_m"]
                )
                and (
                    objective_gain >= 0.92
                    or (
                        current["maximum_event_count"] <= 2
                        and current["maximum_notch_depth_m"] >= 8.0 * working_width
                        and objective_gain >= 0.82
                    )
                )
                and len(proposal_parts) <= maximum_regions
            )
            terminal_structure_resolution = (
                current["maximum_concavity"] >= 0.10
                and proposal["maximum_concavity"] <= 0.05
                and proposal["direction_spread"] >= 20.0
                and objective_gain >= 0.60
                and len(proposal_parts) <= 3
            )
            local_structure_resolved = (
                leaf.area / max(geometry.area, epsilon) >= 0.25
                and current["fragmentation"] <= 1.35
                and proposal["maximum_concavity"] <= 0.80 * current["maximum_concavity"]
                and proposal["maximum_event_count"] <= current["maximum_event_count"] - 1
                and objective_gain >= 0.75
                and len(proposal_parts) <= 4
            )
            # 若整体代价明显变差、非终止面积没有减少，且最难子区仍未达到
            # 低凹性终止形态，这只是代理指标推动的切分，不能据此新增作业区。
            proxy_only_split = (
                whole_regularity["significant_event_count"] < 10
                and residual_reduction < 0.01
                and objective_gain < 0.98
                and proposal["maximum_concavity"] > 0.11
            )
            if proxy_only_split:
                continue
            if not (objective_improves or structural_improves or hole_interrupt_improves
                    or branch_root_improves or deep_concavity_improves
                    or branch_system_improves or same_direction_shape_improves
                    or core_notch_improves or terminal_structure_resolution
                    or local_structure_resolved):
                continue
            proposals.append((
                (
                    int(structural_improves),
                    int(hole_interrupt_improves),
                    int(branch_root_improves),
                    int(deep_concavity_improves),
                    int(branch_system_improves),
                    int(same_direction_shape_improves),
                    int(core_notch_improves),
                    int(terminal_structure_resolution),
                    int(local_structure_resolved),
                    objective_gain,
                    residual_reduction,
                    -len(proposal_parts),
                    -proposal["maximum_concavity"],
                    -proposal["maximum_reflex_count"],
                ),
                proposal_parts, proposal, reason,
                objective_gain, residual_reduction,
            ))
        if not proposals:
            break
        _score, leaves, current, reason, objective_gain, residual_reduction = max(
            proposals, key=lambda item: item[0],
        )
        evidence.append(
            f"第 {depth + 1} 层：{reason}；整体目标改善比 "
            f"{objective_gain:.3f}，非终止面积比例减少 {residual_reduction:.3f}"
        )
    if len(leaves) == 1:
        return leaves, last_rejection_reason
    deep_notch_floor = 2 if (
        whole_regularity["maximum_notch_depth_m"] >= 6.0 * working_width
        and whole_regularity["significant_event_count"] > 0
    ) else 1
    # 一个真实分支的两侧会通常产生 2–4 个相邻反身事件。
    # 因此用事件数/4给合并设一个保守结构下限，防止连续贪心合并
    # 把多个不同凹口重新塞回同一区。下限不会创建新区，只保护
    # 已经由候选评价接受的有限分界。
    event_floor = min(
        len(leaves), 4,
        max(1, math.ceil(whole_regularity["significant_event_count"] / 4.0)),
    )
    parallel_branch_floor = 3 if (
        whole_regularity["concavity_ratio"] >= 0.30
        and 4 <= whole_regularity["significant_event_count"] <= 6
        and len(leaves) >= 3
    ) else 1
    minimum_merged_regions = max(deep_notch_floor, event_floor, parallel_branch_floor)
    merged = _merge_compatible_work_regions(
        leaves, scene, safe_area, straight_area, minimum_region_area * 0.05,
        minimum_regions=minimum_merged_regions,
    )
    if len(merged) < len(leaves):
        evidence.append(f"撤销 {len(leaves) - len(merged)} 条同方向且无收益的内部界面")
    return merged, "；".join(evidence)


def _same_eroded_component(
    component: BaseGeometry,
    first_support: BaseGeometry,
    second_support: BaseGeometry,
    erosion_m: float,
    epsilon: float,
) -> bool:
    """判断腐蚀后的通行空间是否仍连接两个区域的实体支撑面。"""
    eroded = polygonal(component.buffer(-erosion_m)) if erosion_m > 0 else component
    support_buffer = max(10 * epsilon, 1e-6)
    return any(
        part.intersection(first_support.buffer(support_buffer)).area > epsilon * epsilon
        and part.intersection(second_support.buffer(support_buffer)).area > epsilon * epsilon
        for part in polygons(eroded)
    )


def _bottleneck_width(
    component: BaseGeometry,
    first_support: BaseGeometry,
    second_support: BaseGeometry,
    lateral_clearance_m: float,
    epsilon: float,
) -> float:
    """估计两个区域之间的有效瓶颈宽度，不依赖任意代表点。

    旧版本从每区的 ``seed`` 出发腐蚀。seed 落在主体中央时，结果会混入主体
    的宽度；seed 靠近边缘时又会把同一条通道估得很窄，因而同一几何会得到不同
    宽度。现在把两个区域的实际核心/区域面作为支撑：只要同一个腐蚀后分量仍能
    接触两侧支撑，就认为这一级仍可穿过。首次断开的位置就是狭颈的局部尺度。

    返回值恢复到原始 travel 的宽度（已加回 ``lateral_clearance_m``）。它是
    转场通道的几何筛查，不替代后续的曲率和完整车辆包络验证。
    """
    if component.is_empty:
        return 0.0
    # 使用包围盒短轴建立足够高、但与 seed 无关的二分上界。二分判定在通道
    # 已断开或任一支撑面被侵蚀掉时自然失败。
    xmin, ymin, xmax, ymax = component.bounds
    high = max(min(xmax - xmin, ymax - ymin) / 2.0, epsilon)
    low = 0.0
    for _ in range(28):
        middle = (low + high) / 2.0
        if _same_eroded_component(component, first_support, second_support, middle, epsilon):
            low = middle
        else:
            high = middle
    return 2 * (lateral_clearance_m + low)


def _component_id(components: list[BaseGeometry], point: Point, epsilon: float) -> int:
    """寻找覆盖给定局部米制点的分量编号；epsilon只吸收数值误差，未匹配返回-1。"""
    return next(
        (index for index, component in enumerate(components) if component.buffer(epsilon).covers(point)),
        -1,
    )


def _widest_interface_portal(interface: BaseGeometry, epsilon: float) -> Point | None:
    """从真实公共界面取稳定门户点，优先最长连续界面的中点。"""
    lines = [
        part for part in (list(interface.geoms) if hasattr(interface, "geoms") else [interface])
        if part.geom_type == "LineString" and part.length > epsilon
    ]
    if not lines:
        return None
    line = max(lines, key=lambda item: (item.length, -item.bounds[0], -item.bounds[1]))
    return line.interpolate(0.5, normalized=True)


def _straight_transfer_line(
    domain: BaseGeometry,
    first: Point,
    portal: Point | None,
    second: Point,
    epsilon: float,
) -> BaseGeometry | None:
    """生成只在横向安全通行空间内的简易绿色转场引导线。

    这里故意只输出经过精确覆盖检查的直线或两段折线。若一个凹形或障碍物要求
    更复杂的绕行，则保留连接关系、不给出假装可驾驶的折线；后续连续路径模块
    可以在这个门户约束下用车辆运动学求解。
    """
    direct = LineString([(first.x, first.y), (second.x, second.y)])
    allowed = prep(domain.buffer(max(10 * epsilon, 1e-7)))
    if allowed.covers(direct):
        return direct
    if portal is None:
        return None
    broken = LineString([(first.x, first.y), (portal.x, portal.y), (second.x, second.y)])
    return broken if allowed.covers(broken) else None


def _region_sequence(
    regions: tuple[WorkRegion, ...],
    connections: list[RegionConnection],
    start: Pose | None,
) -> tuple[WorkRegion, ...]:
    """按可通过拓扑给出稳定的区域作业顺序，不把它当成最终车辆路径。

    起点遵循显式 start；缺省时采用最左、再最上区域。其后优先选择已知通行图中
    与当前区相连且转场距离短、方向变化小的未访问区域。图断开时开始一个新的
    连通分量，但不会伪造跨越禁入区的边。
    """
    if not regions:
        return regions
    by_id = {region.region_id: region for region in regions}
    neighbors: dict[str, list[tuple[str, float]]] = {region.region_id: [] for region in regions}
    for connection in connections:
        if connection.straight_status == "BLOCKED":
            continue
        first, second = by_id[connection.from_region], by_id[connection.to_region]
        distance = first.seed.distance(second.seed)
        neighbors[first.region_id].append((second.region_id, distance))
        neighbors[second.region_id].append((first.region_id, distance))

    def start_key(region: WorkRegion) -> tuple[float, float, str]:
        if start is not None:
            return (math.hypot(region.seed.x - start.x, region.seed.y - start.y), 0.0, region.region_id)
        return (region.seed.x, -region.seed.y, region.region_id)

    remaining = set(by_id)
    order: list[str] = []
    current = min((by_id[item] for item in remaining), key=start_key).region_id
    while remaining:
        if current not in remaining:
            current = min((by_id[item] for item in remaining), key=start_key).region_id
        order.append(current)
        remaining.remove(current)
        options = []
        for candidate, distance in neighbors[current]:
            if candidate not in remaining:
                continue
            direction_change = _angle_difference_deg(
                by_id[current].preferred_angle_deg, by_id[candidate].preferred_angle_deg,
            )
            options.append((distance, direction_change, candidate))
        if options:
            current = min(options)[2]
        elif remaining:
            # 图中不能从当前节点继续时，优先从已访问区域可达的未访问节点继续；
            # 若整个图确实断开，才按空间规则开始下一组。
            reachable = [
                (distance, direction_change, candidate)
                for visited in order for candidate, distance in neighbors[visited]
                if candidate in remaining
                for direction_change in [_angle_difference_deg(
                    by_id[visited].preferred_angle_deg, by_id[candidate].preferred_angle_deg,
                )]
            ]
            current = min(reachable)[2] if reachable else min(
                (by_id[item] for item in remaining), key=start_key
            ).region_id
    sequence = {region_id: index + 1 for index, region_id in enumerate(order)}
    return tuple(replace(region, sequence_index=sequence[region.region_id]) for region in regions)


def analyze_work_regions(scene: Scene) -> WorkRegionAnalysis:
    """按“安全空间—拓扑主体—收益分区—连接图—审计”生成作业区。"""
    epsilon = scene.settings.geometry_epsilon_m
    lateral, footprint, turning = _vehicle_clearances(scene)
    guard = max(100 * epsilon, 1e-3)
    straight_area = polygonal(scene.travel.buffer(-lateral, quad_segs=32))
    safe_area = polygonal(scene.travel.buffer(-(footprint + guard), quad_segs=64))
    turn_area = polygonal(scene.travel.buffer(-(turning + guard), quad_segs=64))
    working_width = scene.vehicle.working_width_m
    minimum_core_area = max(2 * working_width ** 2, working_width * scene.vehicle.body_width_m)
    minimum_region_area = max(6 * working_width ** 2, 2 * turning * working_width)
    minimum_region_width = 2 * working_width

    raw_components = _meaningful_components(scene.travel, scene.target.area, minimum_core_area)
    straight_components = _meaningful_components(straight_area, scene.target.area, minimum_core_area)
    safe_components = _meaningful_components(safe_area, scene.target.area, minimum_core_area)
    turn_components = _meaningful_components(turn_area, scene.target.area, minimum_core_area)

    rows: list[dict[str, Any]] = []
    for target_part in sorted(polygons(scene.target), key=_spatial_order):
        # 几十平方米的退让残片不能强制创建一个数千平方米的作业区。
        # 拓扑主体既要满足绝对车辆尺度，也要占当前 target part 的
        # 最小比例；较小的面仍保留在 target 中，只是不作为强制分区种子。
        minimum_subject_area = max(minimum_core_area, 0.01 * target_part.area)
        local_straight = [
            part for part in straight_components
            if part.intersection(target_part).area >= minimum_subject_area
        ]
        local_safe = [
            part for part in safe_components
            if part.intersection(target_part).area >= minimum_subject_area
        ]
        local_turn = [
            part for part in turn_components
            if part.intersection(target_part).area >= minimum_subject_area
        ]
        target_quality = _work_shape_quality(
            target_part, _scan_directions(target_part, scene.settings.angles_deg)
        )
        target_regularity = _regularity_metrics(
            target_part, working_width, target_quality,
        )
        target_is_terminal = _terminal_work_shape(target_regularity)
        if len(local_straight) > 1:
            reason_code = "STRAIGHT_WIDTH_BLOCKED"
            evidence = f"基础直行安全空间形成 {len(local_straight)} 个主体"
            assignments = _partition_from_cores(target_part, local_straight, lateral, epsilon)
        elif len(local_safe) > 1:
            # 完整包络层断开是保守证据，基础直行层仍可能单连通。把该证据
            # 交给统一评价，防止边界圆盘收缩产生的小孤岛被直接升格为作业区。
            parts, evidence = _benefit_partition(
                target_part, scene, safe_area, straight_area, turning,
            )
            evidence += f"；完整包络保守空间有 {len(local_safe)} 个分量，已与保持整体统一比较"
            reason_code = "DIRECTION_BENEFIT_SPLIT" if len(parts) > 1 else "SIMPLE_TARGET"
            assignments = [(part, part, polygonal(part.difference(part))) for part in parts]
        elif (len(local_turn) == 2
              and not target_is_terminal
              and all(part.area >= max(minimum_region_area, 0.12 * target_part.area)
                      for part in local_turn)):
            # 调头充分空间断开只是“可能存在狭颈”的证据。用户已确认
            # 单区可以借用相邻区域或共享调头空间，因此不能在这里直接切田。
            parts, evidence = _benefit_partition(
                target_part, scene, safe_area, straight_area, turning,
            )
            evidence += f"；调头充分区有 {len(local_turn)} 个分量，已与保持整体统一比较"
            reason_code = "DIRECTION_BENEFIT_SPLIT" if len(parts) > 1 else "SIMPLE_TARGET"
            assignments = [(part, part, polygonal(part.difference(part))) for part in parts]
        else:
            # 侵蚀骨架、凹角和扫掠事件都在 _benefit_partition_once 中生成候选。
            # 这里只调用统一决策器，避免“侵蚀一分叉就必须分区”绕过保持整体的基线。
            parts, evidence = _benefit_partition(
                target_part, scene, safe_area, straight_area, turning,
            )
            reason_code = "DIRECTION_BENEFIT_SPLIT" if len(parts) > 1 else "SIMPLE_TARGET"
            assignments = [(part, part, polygonal(part.difference(part))) for part in parts]
            # 调头充分区断开只作为解释证据，不单独强制切开整田。
            if len(parts) == 1 and len(local_turn) > 1:
                evidence += f"；调头充分区有 {len(local_turn)} 个分量，但收益不足，保留单区"
        # 拓扑分割只给出第一层主体。每个子体仍必须重新做规则性
        # 检查；否则会把仍然是 L/T/U 形的子区错当成终止结果。
        refined_assignments = []
        maximum_regions_per_target_part = 6
        for assignment_index, (geometry, work_core, corridor) in enumerate(assignments):
            if reason_code == "DIRECTION_BENEFIT_SPLIT":
                # _benefit_partition 已经以整组叶区做了递归、统一评价和合并。
                # 再对每个结果区独立调用一次，会丢失整田区数预算，并把已选定的
                # 三区联合方案再切成四区。这里只保留结果，后面仍会统一重建核心。
                refined_assignments.append((geometry, work_core, corridor, reason_code, evidence))
                continue
            analysis_core, _safe_parts, _straight_parts = _materialize_region_core(
                geometry, safe_area, straight_area, minimum_core_area,
            )
            local_shape = analysis_core if not analysis_core.is_empty else geometry
            local_quality = _work_shape_quality(
                local_shape, _scan_directions(local_shape, scene.settings.angles_deg),
            )
            local_regularity = _regularity_metrics(
                local_shape, working_width, local_quality,
            )
            if (reason_code in {"ADAPTIVE_NECK_SPLIT", "TURN_SPACE_NECK"}
                    and _terminal_work_shape(local_regularity)):
                # 狭颈首先只回答“两个主体是否应独立组织作业”。评估的是实际
                # 安全作业核心，而不是外壳上附带的半条连接带：一个规则主体
                # 不能仅因被分到少量 corridor 面积而再被错误切碎。
                refined_assignments.append((geometry, work_core, corridor, reason_code, evidence))
                continue
            # 狭颈分割后的主体也可能仍是 L/T/U 形。此时不能因为它曾经是
            # "neck" 的一侧就提前终止：必须按其自己的局部方向重新评价并在
            # 有真实收益时递归分区。这样既保留狭颈作为连接带，也能消除残余
            # 的分支作业形态。
            # ``_partition_from_cores`` may have already created several
            # 区域数上限作用于原始目标的整组子区域；先给尚未访问的安全/拓扑必需区域留出名额，避免每个递归分支各自使用上限而产生过多分区。
            remaining_subjects = len(assignments) - assignment_index - 1
            remaining_capacity = max(
                1,
                maximum_regions_per_target_part - len(refined_assignments) - remaining_subjects,
            )
            child_parts, child_evidence = _benefit_partition(
                geometry, scene, safe_area, straight_area, turning,
                maximum_regions=remaining_capacity,
            )
            if len(child_parts) == 1:
                refined_assignments.append((geometry, work_core, corridor, reason_code, evidence))
                continue
            for child in child_parts:
                child_corridor = polygonal(child.intersection(corridor))
                child_core = polygonal(child.intersection(work_core))
                if child_core.is_empty:
                    child_core = polygonal(child.difference(child_corridor))
                refined_assignments.append((
                    child, child_core, child_corridor,
                    "RECURSIVE_REGULARITY_SPLIT",
                    f"父区原因 {reason_code}：{evidence}；{child_evidence}",
                ))
        # 递归的切线可能穿过父区原有的核心。先按最终几何重新建立核心，并在
        # 必要时按已断开的安全主体强制拆回独立区。这样不会出现“工作区是一个
        # Polygon、work_core 却是两块”的伪通过。
        normalized_assignments = []
        for geometry, _work_core, corridor, row_reason, row_evidence in refined_assignments:
            for child, child_core, child_corridor in _split_disconnected_core_assignment(
                geometry, corridor, safe_area, straight_area, minimum_core_area,
                lateral, epsilon,
            ):
                split_applied = not child.equals(geometry)
                normalized_assignments.append((
                    child, child_core, child_corridor,
                    "CORE_CONNECTIVITY_SPLIT" if split_applied else row_reason,
                    (
                        f"{row_evidence}；最终区域的车辆核心被多个主体分开，"
                        "按安全核心重新分配"
                        if split_applied else row_evidence
                    ),
                ))
        normalized_assignments = _absorb_connector_regions(
            normalized_assignments, scene, safe_area, straight_area,
            minimum_core_area,
        )
        normalized_assignments = _absorb_nonworkable_final_regions(
            normalized_assignments, scene, safe_area, straight_area,
            minimum_region_area, minimum_region_width,
        )
        for geometry, work_core, corridor, row_reason, row_evidence in normalized_assignments:
            core_domain = polygonal(geometry.difference(corridor)) if not corridor.is_empty else geometry
            work_core, safe_inside, straight_inside = _materialize_region_core(
                core_domain, safe_area, straight_area, minimum_core_area,
            )
            if len(safe_inside) == 1:
                reachability = access = "FULL_ENVELOPE_SAFE"
                seed_space = safe_inside[0]
            elif len(straight_inside) == 1:
                reachability = access = "STRAIGHT_ONLY_PENDING_SWEPT_VALIDATION"
                seed_space = straight_inside[0]
            else:
                reachability = access = "UNREACHABLE_REVIEW"
                seed_space = geometry
            seed = seed_space.representative_point()
            rows.append({
                "geometry": geometry, "work_core": work_core, "corridor": corridor,
                "seed": seed, "reason_code": row_reason, "evidence": row_evidence,
                "access": access, "reachability": reachability,
                "safe_count": len(safe_inside), "straight_count": len(straight_inside),
            })

    rows.sort(key=lambda row: _spatial_order(row["geometry"]))
    regions_list: list[WorkRegion] = []
    for index, row in enumerate(rows):
        quality_geometry = row["work_core"] if not row["work_core"].is_empty else row["geometry"]
        quality = _work_shape_quality(
            quality_geometry, _scan_directions(quality_geometry, scene.settings.angles_deg),
        )
        geometry_connected = len(_meaningful_components(row["geometry"], row["geometry"].area, 0.0)) == 1
        unverified = row["geometry"].difference(safe_area).area
        regions_list.append(WorkRegion(
            region_id=f"region_{index + 1:03d}", geometry=row["geometry"], seed=row["seed"],
            straight_component_id=_component_id(straight_components, row["seed"], epsilon),
            reason=row["reason_code"], access_status=row["access"],
            work_core=row["work_core"], corridor_assignment=row["corridor"],
            preferred_angle_deg=quality["angle_deg"],
            mean_section_length_m=quality["mean_section_length_m"],
            fragmentation=quality["fragmentation"], coverage_span_m=quality["coverage_span_m"],
            geometry_connected=geometry_connected, safe_connected=row["safe_count"] == 1,
            safe_component_count=row["safe_count"], straight_component_count=row["straight_count"],
            reachability_status=row["reachability"], unverified_target_area_m2=float(unverified),
            split_reason_code=row["reason_code"], split_evidence=row["evidence"],
            turn_resource=(
                "INTERNAL_TURN_SUFFICIENT"
                if any(
                    part.intersection(row["work_core"]).area > epsilon * epsilon
                    for part in turn_components
                )
                else "SHARED_OR_EXTERNAL_TRAVEL_REQUIRED"
            ),
        ))
    regions = tuple(regions_list)

    connections: list[RegionConnection] = []
    for i, first in enumerate(regions):
        for second in regions[i + 1:]:
            exact_contact = first.geometry.boundary.intersection(second.geometry.boundary)
            shared = _shared_interface(first.geometry, second.geometry, epsilon)
            if shared.is_empty:
                continue
            first_raw = _component_id(raw_components, first.seed, epsilon)
            second_raw = _component_id(raw_components, second.seed, epsilon)
            same_straight = (
                first.straight_component_id >= 0
                and first.straight_component_id == second.straight_component_id
            )
            transfer_domain = None
            if not exact_contact.is_empty and exact_contact.length <= epsilon:
                width_class, straight_status, width = "POINT_CONTACT", "BLOCKED", 0.0
                evidence = "两个作业主体仅点接触，不记录为可通行连接"
                mouth = shared
            elif first_raw < 0 or second_raw < 0 or first_raw != second_raw:
                width_class, straight_status, width = "DISCONNECTED", "BLOCKED", None
                evidence = "原始允许通行空间中不属于同一连通分量"
                mouth = shared
            elif not same_straight:
                width_class, straight_status, width = "TOO_NARROW_FOR_VEHICLE", "BLOCKED", None
                evidence = "原始空间相连，但车辆横向安全空间已断开"
                mouth = shared
            else:
                pair_domain = polygonal(unary_union([first.geometry, second.geometry]))
                component = polygonal(straight_components[first.straight_component_id].intersection(pair_domain))
                transfer_domain = straight_components[first.straight_component_id]
                mouth = shared.intersection(component)
                if mouth.length <= epsilon:
                    width_class, straight_status, width = "POINT_CONTACT", "BLOCKED", 0.0
                    evidence = "公共边界没有具备面积的车辆直行接口"
                else:
                    width = _bottleneck_width(
                        component,
                        first.work_core if first.work_core is not None and not first.work_core.is_empty else first.geometry,
                        second.work_core if second.work_core is not None and not second.work_core.is_empty else second.geometry,
                        lateral,
                        epsilon,
                    )
                    first_support = (
                        first.work_core if first.work_core is not None and not first.work_core.is_empty
                        else first.geometry
                    )
                    second_support = (
                        second.work_core if second.work_core is not None and not second.work_core.is_empty
                        else second.geometry
                    )
                    same_safe = any(
                        part.intersection(first_support).area > epsilon * epsilon
                        and part.intersection(second_support).area > epsilon * epsilon
                        for part in polygons(polygonal(safe_area.intersection(pair_domain)))
                    )
                    straight_status = (
                        "FULL_ENVELOPE_CERTIFIED" if same_safe
                        else "STRAIGHT_ONLY_PENDING_SWEPT_VALIDATION"
                    )
                    # ``WIDE`` 不只是一个横向宽度数值。根据 task.md，它意味着该
                    # 连接已在保守转弯空间中被证明。转弯层的共同支撑面要在下方才能
                    # 计算，因此先记为 NARROW，等关系证据完整后再提升。
                    width_class = "NARROW"
                    evidence = "公共接口和逐级腐蚀瓶颈宽度；连续车辆运动仍由后续模块验证"
            first_support = (
                first.work_core if first.work_core is not None and not first.work_core.is_empty
                else first.geometry
            )
            second_support = (
                second.work_core if second.work_core is not None and not second.work_core.is_empty
                else second.geometry
            )
            # 点接触、原始 travel 断开和直行层断开的连接都被定义为不可通过。
            # 即使两区附近同处于一个转弯面积分量，那也不能反过来把这条断开
            # 关系写成「转弯空间充分」。只有直行层已证明存在直接通道时，才计算共享转弯空间。
            same_turn = False
            if straight_status != "BLOCKED":
                same_turn = any(
                    part.intersection(first_support).area > epsilon * epsilon
                    and part.intersection(second_support).area > epsilon * epsilon
                    for part in polygons(polygonal(
                        turn_area.intersection(unary_union([first.geometry, second.geometry]))
                    ))
                )
            if width is not None and straight_status != "BLOCKED":
                width_class = (
                    "WIDE"
                    if width + epsilon >= 2 * turning and same_turn
                    else "NARROW"
                )
                if width + epsilon >= 2 * turning and not same_turn:
                    evidence += "；有效宽度达到调头尺度，但两侧不共享保守转弯空间，按 NARROW 记录"
            portal = _widest_interface_portal(mouth, epsilon)
            transfer_line = (
                _straight_transfer_line(transfer_domain, first.seed, portal, second.seed, epsilon)
                if transfer_domain is not None and straight_status != "BLOCKED" else None
            )
            connections.append(RegionConnection(
                connection_id=f"connection_{len(connections) + 1:03d}",
                from_region=first.region_id, to_region=second.region_id, mouth=mouth,
                effective_width_m=None if width is None else float(width),
                width_class=width_class, straight_status=straight_status,
                turn_status="SUFFICIENT_SPACE" if same_turn else "NOT_CERTIFIED",
                evidence=evidence, portal=portal, transfer_line=transfer_line,
            ))

    regions = _region_sequence(regions, connections, scene.start)

    region_union = polygonal(unary_union([region.geometry for region in regions]))
    overlap = max(0.0, sum(region.geometry.area for region in regions) - region_union.area)
    missing = scene.target.difference(region_union).area
    outside = region_union.difference(scene.target).area
    target_quality = _work_shape_quality(scene.target, _scan_directions(scene.target, scene.settings.angles_deg))
    core_area = sum(region.work_core.area for region in regions)
    core_span = sum(region.coverage_span_m for region in regions)
    harmonic_mean = core_area / max(core_span, epsilon)
    direct_pairs = {(item.from_region, item.to_region) for item in connections}
    expected_pairs = {
        (first.region_id, second.region_id)
        for i, first in enumerate(regions) for second in regions[i + 1:]
        if not _shared_interface(first.geometry, second.geometry, epsilon).is_empty
    }
    # 每个横向安全通行分量中的区域，必须在输出拓扑图中连通；如果 travel 本身
    # 断开，则保留多个图分量并明确记录，而不是画一条跨禁入区的假连接。
    passable_edges = [
        item for item in connections if item.straight_status != "BLOCKED"
    ]
    adjacency: dict[str, set[str]] = {region.region_id: set() for region in regions}
    for item in passable_edges:
        adjacency[item.from_region].add(item.to_region)
        adjacency[item.to_region].add(item.from_region)
    graph_components = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen)
        stack, component_ids = [root], set()
        while stack:
            node = stack.pop()
            if node in component_ids:
                continue
            component_ids.add(node)
            stack.extend(adjacency[node] - component_ids)
        unseen -= component_ids
        graph_components.append(component_ids)
    straight_groups: dict[int, set[str]] = {}
    for region in regions:
        if region.straight_component_id >= 0:
            straight_groups.setdefault(region.straight_component_id, set()).add(region.region_id)
    topology_connected = all(
        any(group <= component_ids for component_ids in graph_components)
        for group in straight_groups.values()
    )
    # ``safe_area`` 是参考点域。要验证完整车辆/机具包络不侵入禁区，必须把
    # 该域重新按同一保守包络半径膨胀并与 travel 比较，而不能只检查已经
    # 内缩后的 safe_area 自身位于 travel 内。
    reconstructed_envelope = polygonal(
        safe_area.buffer(footprint + guard, quad_segs=64)
    )
    envelope_intrusion = reconstructed_envelope.difference(scene.travel).area
    # GEOS 用有限圆弧段近似负/正 buffer。相同半径的腐蚀再膨胀会在曲线
    # 拼接处产生平方厘米级面差；用 0.05 m² 或目标面积的 2e-6（取较大者）
    # 作为纯数值容差，仍远小于一个车辆足迹或一个作业幅宽的面积。
    envelope_tolerance = max(0.05, scene.target.area * 2e-6)
    checks: dict[str, Any] = {
        "valid_region_geometries": all(region.geometry.is_valid and not region.geometry.is_empty for region in regions),
        "target_unchanged": region_union.symmetric_difference(scene.target).area <= epsilon,
        "safe_area_inside_travel": envelope_intrusion <= envelope_tolerance,
        "safe_reference_area_inside_travel": safe_area.difference(scene.travel).area <= epsilon,
        "full_envelope_reference_inside_travel": envelope_intrusion <= envelope_tolerance,
        "regions_non_overlapping": overlap <= epsilon,
        "target_fully_assigned": missing <= epsilon,
        "regions_inside_target": outside <= epsilon,
        "all_regions_geometry_connected": all(region.geometry_connected for region in regions),
        # 不能只筛选 FULL_ENVELOPE_SAFE 后对空集合做 all()；那会让所有区域
        # 都处于 UNREACHABLE 时仍报告 true。这里将核心存在、核心连通和可达
        # 状态拆开记录，并让兼容字段采用严格含义。
        "all_regions_have_work_core": all(
            region.work_core is not None and not region.work_core.is_empty and region.work_core.area >= minimum_core_area
            for region in regions
        ),
        "all_regions_meet_minimum_work_scale": all(
            region.work_core is not None
            and region.work_core.area >= minimum_region_area
            and _effective_region_width(region.work_core, epsilon) >= minimum_region_width
            for region in regions
        ),
        "all_work_cores_connected": all(
            len(_meaningful_components(region.work_core, region.geometry.area, minimum_core_area)) == 1
            for region in regions if region.work_core is not None
        ),
        "all_regions_partition_reachable": all(
            region.reachability_status != "UNREACHABLE_REVIEW" for region in regions
        ),
        "all_regions_safe_connected": all(
            region.safe_component_count == 1 for region in regions
        ),
        "all_regions_have_safe_core": all(
            region.safe_component_count == 1
            and region.work_core is not None
            and not region.work_core.is_empty
            and region.work_core.difference(safe_area).area <= epsilon
            for region in regions
        ),
        "point_contacts_not_passable": all(
            item.straight_status == "BLOCKED" and item.turn_status != "SUFFICIENT_SPACE"
            for item in connections if item.width_class == "POINT_CONTACT"
        ),
        "blocked_connections_not_turn_sufficient": all(
            item.turn_status != "SUFFICIENT_SPACE"
            for item in connections if item.straight_status == "BLOCKED"
        ),
        "wide_connections_turn_sufficient": all(
            item.turn_status == "SUFFICIENT_SPACE"
            for item in connections if item.width_class == "WIDE"
        ),
        "all_direct_adjacencies_recorded_once": direct_pairs == expected_pairs,
        "transition_graph_connected_within_travel_components": topology_connected,
        "transition_graph_component_count": len(graph_components),
        "green_transfer_line_count": sum(
            item.transfer_line is not None and not item.transfer_line.is_empty for item in connections
        ),
        "region_count": len(regions),
        "straight_component_count": len(straight_components),
        "safe_component_count": len(safe_components),
        "turn_component_count": len(turn_components),
        "overlap_area_m2": float(overlap), "missing_target_area_m2": float(missing),
        "outside_target_area_m2": float(outside),
        "safe_area_intrusion_m2": float(envelope_intrusion),
        "safe_reference_area_intrusion_m2": float(safe_area.difference(scene.travel).area),
        "unverified_target_area_m2": float(sum(region.unverified_target_area_m2 for region in regions)),
        "whole_target_mean_section_length_m": float(target_quality["mean_section_length_m"]),
        "work_core_harmonic_mean_section_length_m": float(harmonic_mean),
        "mean_section_length_gain_ratio": float(harmonic_mean / max(target_quality["mean_section_length_m"], epsilon)),
        "whole_target_fragmentation": float(target_quality["fragmentation"]),
        "maximum_work_core_fragmentation": float(max(region.fragmentation for region in regions)),
        "whole_target_coverage_span_m": float(target_quality["coverage_span_m"]),
        "work_core_coverage_span_m": float(core_span),
        "assigned_corridor_area_m2": float(sum(region.corridor_assignment.area for region in regions)),
        "decision_policy": "硬约束 -> 真实分区理由 -> 作业收益 -> 最少区域数",
    }
    return WorkRegionAnalysis(
        target=scene.target, travel=scene.travel, safe_reference_area=safe_area,
        straight_passage_area=straight_area, conservative_turn_area=turn_area,
        regions=regions, connections=tuple(connections),
        parameters={
            "lateral_clearance_m": float(lateral),
            "full_footprint_radius_m": float(footprint),
            "conservative_turn_clearance_m": float(turning),
            "numeric_buffer_guard_m": float(guard),
            "minimum_core_area_m2": float(minimum_core_area),
            "minimum_region_area_m2": float(minimum_region_area),
            "minimum_region_width_m": float(minimum_region_width),
            "minimum_direction_difference_deg": 20.0,
            "minimum_mean_section_gain_ratio": 1.10,
        },
        checks=checks,
    )


class F2CBackend:
    """原生Fields2Cover适配器；负责几何和车辆参数转换，不使用模拟路线代替缺失的原生库。"""
    def __init__(self, scene: Scene) -> None:
        """用Scene构造原生机器人与求解器，尺寸单位米；原生依赖缺失时明确失败。"""
        try:
            import fields2cover as f2c
        except ImportError as exc:
            raise RuntimeError("未找到 fields2cover。请先安装 F2C 2.1 及 C++ 系统依赖；不会偷偷改用模拟路线") from exc
        self.f2c, self.scene = f2c, scene
        self.version = getattr(f2c, "__version__", "unknown")
        required = ["Cells", "Cell", "Point", "MultiPoint", "HG_Const_gen", "SG_BruteForce",
                    "RP_Boustrophedon", "RP_Snake", "PP_PathPlanning", "PP_DubinsCurvesCC"]
        missing = [name for name in required if not hasattr(f2c, name)]
        if missing:
            raise RuntimeError(f"当前 F2C 缺少接口 {missing}；本代码按 v2.1.0 接口编写")
        v = scene.vehicle
        self.robot = f2c.Robot(v.body_width_m, v.working_width_m)
        self.robot.setMinTurningRadius(v.min_turn_radius_m)
        self.robot.setMaxDiffCurv(v.max_curvature_rate)
        self.robot.setCruiseVel(v.transit_speed_mps)
        self.robot.setTurnVel(v.turn_speed_mps)
        self.turners = [("dubins_cc", f2c.PP_DubinsCurvesCC())]
        if v.allow_reverse:
            if not hasattr(f2c, "PP_ReedsSheppCurvesHC"):
                raise RuntimeError("缺少 PP_ReedsSheppCurvesHC，无法启用允许倒车的连接")
            self.turners.append(("reeds_shepp_hc", f2c.PP_ReedsSheppCurvesHC()))
        for _, turner in self.turners:
            turner.setDiscretization(scene.settings.sampling_step_m)
        self.swath_generator = f2c.SG_BruteForce()
        self.swath_generator.setAllowOverlap(True)

    def cells(self, geometry):
        """把Shapely面转换为原生Cells，保留多面及孔洞结构。"""
        result = self.f2c.Cells()
        result.importFromWkt(MultiPolygon(polygons(geometry)).wkt)
        return result

    def geometry(self, cells):
        """把原生几何转换为Shapely几何；转换后仍使用本田局部米制坐标。"""
        return polygonal(wkt.loads(cells.exportToWkt()))

    def swaths(self, polygon, angle: float, pattern: str):
        """按方向及作业宽度生成原生条带；方向遵循调用者约定，结果不是完整行程。"""
        cell = self.f2c.Cell()
        cell.importFromWkt(polygon.wkt)
        spacing = self.scene.vehicle.working_width_m * (1 - self.scene.settings.overlap_fraction)
        swaths = self.swath_generator.generateSwaths(angle, spacing, cell)
        if swaths.size() == 0:
            return swaths
        sorter = self.f2c.RP_Snake() if pattern == "snake" else self.f2c.RP_Boustrophedon()
        return sorter.genSortedSwaths(swaths)

    def turn_path(self, start: Pose, end: Pose, turner, guide=None):
        """由两端位姿调用原生转弯求解；求解返回后还须验证运动和真实通行空间。"""
        f2c = self.f2c
        p0, p1 = f2c.Point(start.x, start.y), f2c.Point(end.x, end.y)
        if guide is None:
            return turner.createTurn(self.robot, p0, start.yaw, p1, end.yaw)
        points = f2c.MultiPoint()
        for x, y in guide[1:-1]:
            points.addGeometry(f2c.Point(float(x), float(y)))
        return f2c.PP_PathPlanning().planPathForConnection(
            self.robot, p0, start.yaw, points, p1, end.yaw, turner)

    def convert_path(self, path, link: tuple[str, str], kind: str) -> Motion:
        """读取原生 PathState；检查每个 len 的终点，绝不把空连接补成直线。"""
        if path.size() == 0:
            raise ValueError("F2C 返回空转弯")
        rows = []
        previous_end = None
        cfg = self.scene.settings
        for i in range(path.size()):
            s = path.getState(i)
            x, y = float(s.point.getX()), float(s.point.getY())
            if previous_end is not None and math.hypot(x - previous_end[0], y - previous_end[1]) > cfg.join_tolerance_m:
                raise ValueError("F2C 原生 PathState 之间存在位置缺口")
            direction = int(s.dir)
            if direction not in (-1, 1):
                raise ValueError("未知 F2C PathDirection")
            row = [x, y, float(s.angle), direction]
            if rows and np.linalg.norm(np.array(row[:2]) - rows[-1][:2]) < 1e-10 and abs(wrap(row[2] - rows[-1][2])) < 1e-10:
                rows[-1] = row  # 同位姿换挡时，该点的出发挡位以新状态为准
            else:
                rows.append(row)
            p_end = s.atEnd()
            previous_end = (float(p_end.getX()), float(p_end.getY()))
        last = path.getState(path.size() - 1)
        if math.hypot(previous_end[0] - rows[-1][0], previous_end[1] - rows[-1][1]) > 1e-10:
            rows.append([*previous_end, float(last.angle), int(last.dir)])
        if len(rows) < 2:
            raise ValueError("F2C 返回零长度连接")
        return Motion(np.array(rows), kind, implement_on=False, link=link)


def task_from_swath(swath, task_id: str, cell_id: int, kind: str = "work") -> Task:
    """将原生条带转换为Task；允许直线上的冗余顶点，真正折线不能冒充直线作业任务。"""
    if swath.numPoints() != 2:
        # 原生裁剪/排序可在同一直线上保留多个顶点，点数不能判断是否折线。
        coordinates = np.asarray(wkt.loads(swath.getPath().exportToWkt()).coords, dtype=float)[:, :2]
        delta = coordinates[-1] - coordinates[0]
        length = float(np.linalg.norm(delta))
        if length <= 1e-7 or not np.isfinite(coordinates).all():
            raise ValueError("作业带为空、非有限或端点重合")
        offsets = coordinates - coordinates[0]
        along = offsets @ (delta / length)
        cross = np.abs(offsets[:, 0] * delta[1] - offsets[:, 1] * delta[0]) / length
        if np.max(cross) > 1e-7 or np.any(np.diff(along) < -1e-7):
            raise ValueError("当前实现仅接受直线作业带；不能把折线或折返作业带直接视为可驾驶")
    a, b = swath.startPoint(), swath.endPoint()
    yaw = math.atan2(b.getY() - a.getY(), b.getX() - a.getX())
    return Task(task_id, Pose(a.getX(), a.getY(), yaw), Pose(b.getX(), b.getY(), yaw), kind, cell_id)


def work_motion(task: Task, scene: Scene) -> Motion:
    """从Task起终位姿构造开机具直线采样；长度单位米，朝向弧度，零长度任务拒绝。"""
    a, b = task.start, task.end
    length = math.hypot(b.x - a.x, b.y - a.y)
    if length < 1e-7:
        raise ValueError("零长度作业任务")
    step = min(scene.settings.sampling_step_m, length / 3)
    t = np.array([0.0, step / length, 1 - step / length, 1.0])
    points = np.column_stack([a.x + t * (b.x - a.x), a.y + t * (b.y - a.y),
                              np.full(4, a.yaw), np.ones(4)])
    return Motion(points, task.kind, task.task_id, True)


class Connector:
    """历史完整规划连接器；缓存已检查连接，并用几何引导辅助原生求解，搜索失败不宣称物理不可能。"""
    def __init__(self, scene: Scene, backend: F2CBackend, validator: Validator, budget: Budget) -> None:
        """绑定场景、原生求解器、验证器与共享预算；缓存只属于本次连接器实例。"""
        self.scene, self.backend, self.validator, self.budget = scene, backend, validator, budget
        self.cache: OrderedDict[tuple, tuple[Motion | None, str | None]] = OrderedDict()
        self.guide_nodes: list[tuple[float, float]] | None = None
        self.guide_edges: list[list[tuple[int, float]]] = []
        self.guide_failure: str | None = None
        self.stats = {"native_turn_calls": 0, "connection_cache_hits": 0, "guide_queries": 0}
        clearance = max(scene.vehicle.body_width_m, scene.vehicle.working_width_m) / 2 + scene.vehicle.safety_margin_m
        self.guide_area = polygonal(scene.travel.buffer(-clearance))
        self.guide_free = prep(self.guide_area.buffer(scene.settings.geometry_epsilon_m))

    def _build_guide_graph(self) -> None:
        """按需构建一次可见图。只作转移引导，所有最终运动还要独立验证。"""
        if self.guide_nodes is not None:
            return
        self.guide_nodes = []
        # 只简化引导顶点，碰撞检查仍用未简化的 guide_area / 原始 travel。
        simplified = self.guide_area.simplify(self.scene.settings.sampling_step_m, preserve_topology=True)
        nodes = [tuple(p) for poly in polygons(simplified) for ring in [poly.exterior, *poly.interiors]
                 for p in list(ring.coords)[:-1]]
        if len(nodes) > self.scene.settings.max_guide_nodes:
            self.guide_failure = "GUIDE_NODE_LIMIT"
            return
        self.guide_nodes = nodes
        self.guide_edges = [[] for _ in nodes]
        for i, a in enumerate(nodes):
            self.budget.check()
            for j in range(i):
                b = nodes[j]
                if self.guide_free.covers(LineString([a, b])):
                    distance = math.dist(a, b)
                    self.guide_edges[i].append((j, distance))
                    self.guide_edges[j].append((i, distance))

    def _guide(self, start: Pose, end: Pose):
        """在合法通行区搜索绕障引导线；引导点仅供车辆求解，不是已认证的可执行路径。"""
        self.stats["guide_queries"] += 1
        self._build_guide_graph()
        if self.guide_failure or self.guide_area.is_empty:
            return None
        nodes = [*self.guide_nodes, (start.x, start.y), (end.x, end.y)]
        n = len(nodes)
        edges = [list(e) for e in self.guide_edges] + [[], []]
        for i in (n - 2, n - 1):
            for j in range(i):
                if self.guide_free.covers(LineString([nodes[i], nodes[j]])):
                    d = math.dist(nodes[i], nodes[j])
                    edges[i].append((j, d))
                    edges[j].append((i, d))
        distances, parents, queue = {n - 2: 0.0}, {}, [(0.0, n - 2)]
        while queue:
            distance, i = heapq.heappop(queue)
            if distance > distances.get(i, math.inf):
                continue
            if i == n - 1:
                route = [i]
                while route[-1] != n - 2:
                    route.append(parents[route[-1]])
                return [nodes[j] for j in reversed(route)]
            for j, weight in edges[i]:
                total = distance + weight
                if total < distances.get(j, math.inf):
                    distances[j], parents[j] = total, i
                    heapq.heappush(queue, (total, j))
        return None

    def connect(self, start: Pose, end: Pose, link: tuple[str, str]) -> tuple[Motion | None, str | None]:
        """尝试连接两个真实位姿并复检；返回运动或失败原因，禁止静默跨越不可用通道。"""
        self.budget.check()
        if math.hypot(start.x - end.x, start.y - end.y) < 1e-8 and abs(wrap(start.yaw - end.yaw)) < 1e-8:
            return None, None
        # 精确位姿作键，不因粗量化把另一条连接错误地复用。
        key = (start.x, start.y, start.yaw, end.x, end.y, end.yaw)
        if key in self.cache:
            self.stats["connection_cache_hits"] += 1
            self.cache.move_to_end(key)
            motion, error = self.cache[key]
            if motion is not None:
                motion = Motion(motion.points, motion.kind, implement_on=False, link=link)
            return motion, error
        failures = []
        result = None
        for use_guide in (False, True):
            guide = self._guide(start, end) if use_guide else None
            if use_guide and (guide is None or len(guide) <= 2):
                continue
            for name, turner in self.backend.turners:
                self.budget.check()
                try:
                    self.stats["native_turn_calls"] += 1
                    native = self.backend.turn_path(start, end, turner, guide)
                    kind = "transit" if use_guide else "turn"
                    motion = self.backend.convert_path(native, link, kind)
                    first, last = motion.points[0], motion.points[-1]
                    cfg = self.scene.settings
                    if any((math.hypot(p[0] - wanted.x, p[1] - wanted.y) > cfg.join_tolerance_m or
                            abs(wrap(float(p[2] - wanted.yaw))) > cfg.heading_tolerance_rad)
                           for p, wanted in ((first, start), (last, end))):
                        raise ValueError("原生连接没有满足起终位姿")
                    issues = self.validator.motion_issues(motion)
                    if not issues:
                        result = motion
                        break
                    failures.append(name + ":" + issues[0]["code"])
                except (ValueError, RuntimeError) as exc:
                    failures.append(name + ":" + str(exc))
            if result is not None:
                break
        error = None if result is not None else "; ".join(failures[-4:]) or self.guide_failure or "NO_CONNECTION_FOUND"
        self.cache[key] = (result, error)
        while len(self.cache) > self.scene.settings.connection_cache_size:
            self.cache.popitem(last=False)
        return result, error


def candidate_specs(scene: Scene) -> list[tuple[float, str, bool]]:
    """生成历史full阶段的方向、往复模式及分解候选；不是冻结分区结果，也不是参考路线的任务排序。"""
    cfg = scene.settings
    complex_shape = any(p.interiors or p.convex_hull.area - p.area > cfg.coverage_tolerance_m2 for p in polygons(scene.target))
    splits = [False] if cfg.decomposition == "none" else [True] if cfg.decomposition == "always" else ([False, True] if complex_shape else [False])
    specs = []
    # 交错安排，避免 max_candidates 太小时所有分区备选被排到预算之外。
    rounds = list(itertools.product(cfg.patterns, splits))
    for round_index in range(len(rounds)):
        pattern, split = rounds[round_index]
        if len(splits) == 2:
            split = round_index % 2 == 1
            pattern = cfg.patterns[(round_index // 2) % len(cfg.patterns)]
        for angle in cfg.angles_deg:
            specs.append((math.radians(angle), pattern, split))
    return specs[:cfg.max_candidates]


def generate_tasks(scene: Scene, backend: F2CBackend, spec, budget: Budget) -> tuple[list[Task], dict[str, Any]]:
    """按候选规格生成历史full主体任务，田头退让不修改Scene.target的覆盖验收分母。"""
    angle, pattern, split = spec
    budget.check()
    native = backend.cells(scene.target)
    # 先对全田真实边界内缩一次；保留 target 和 travel，不修改覆盖验收分母。
    mainland = backend.f2c.HG_Const_gen().generateHeadlands(native, scene.settings.headland_m)
    decomposition_algorithm = "none"
    if split and mainland.size():
        choices = {"boustrophedon": "DECOMP_BoustrophedonDecomp", "trapezoidal": "DECOMP_TrapezoidalDecomp"}
        requested = scene.settings.decomposition_algorithm
        algorithms = list(choices) if requested == "auto" else [requested]
        decomposition_algorithm = next((name for name in algorithms if hasattr(backend.f2c, choices[name])), None)
        if decomposition_algorithm is None or not hasattr(backend.f2c, "HG_Corridor_gen"):
            raise RuntimeError(f"当前 F2C 绑定不支持分区配置 {requested} 或缺少 HG_Corridor_gen")
        decomp = getattr(backend.f2c, choices[decomposition_algorithm])()
        decomp.setSplitAngle(angle)
        mainland = decomp.decompose(mainland)
        mainland = backend.f2c.HG_Corridor_gen().generateHeadlands(mainland, scene.settings.corridor_m)
    area = backend.geometry(mainland)
    tasks = []
    for cell_id, polygon in enumerate(polygons(area)):
        budget.check()
        swaths = backend.swaths(polygon, angle, pattern)
        for i in range(swaths.size()):
            tasks.append(task_from_swath(swaths.at(i), f"cell{cell_id}_swath{i}", cell_id))
            if len(tasks) > scene.settings.max_tasks:
                raise ValueError("任务数量超限，未静默删除作业段")
    if not tasks:
        # 小田块可能没有主体区：直接将全目标交给有限的补作任务生成，不重复内缩。
        tasks = generate_patch_tasks(scene, backend, scene.target, [], budget, "initial")
    return tasks, {"angle_deg": math.degrees(angle), "pattern": pattern, "decomposed": split,
                   "decomposition_algorithm": decomposition_algorithm,
                   "mainland_area_m2": float(area.area), "mainland_wkt_local": area.wkt,
                   "headland_strategy": "global_constant_then_internal_corridors"}


def assemble(tasks: list[Task], scene: Scene, connector: Connector,
             metadata: dict[str, Any] | None = None) -> Plan:
    """串接作业任务与连接，保留连接失败和起终位姿检查；结果仍须交给Validator验收。"""
    motions, errors = [], []
    previous_pose, previous_id = scene.start, "START"
    for i, task in enumerate(tasks):
        connector.budget.check()
        if previous_pose is not None:
            connection, error = connector.connect(previous_pose, task.start, (previous_id, task.task_id))
            if error:
                errors.append({"code": "CONNECTION_FAILED", "before_task_index": i,
                               "from": previous_id, "to": task.task_id, "message": error})
            elif connection is not None:
                motions.append(connection)
        motions.append(work_motion(task, scene))
        previous_pose, previous_id = task.end, task.task_id
        if len(errors) >= scene.settings.max_failed_links:
            break
    if len([m for m in motions if m.implement_on]) == len(tasks) and previous_pose and scene.end:
        connection, error = connector.connect(previous_pose, scene.end, (previous_id, "END"))
        if error:
            errors.append({"code": "CONNECTION_FAILED", "before_task_index": len(tasks),
                           "from": previous_id, "to": "END", "message": error})
        elif connection is not None:
            motions.append(connection)
    return Plan(list(tasks), motions, errors, dict(metadata or {}))


def _fit_straight(task: Task, scene: Scene, validator: Validator) -> Task | None:
    """少量端点退让候选；不搜索所有子段，未找到时保留漏作而非伪造成功。"""
    if not validator.motion_issues(work_motion(task, scene)):
        return task
    v = scene.vehicle
    length = math.hypot(task.end.x - task.start.x, task.end.y - task.start.y)
    margin = max(v.front_m, v.rear_m, abs(v.implement_offset_m) + v.implement_length_m / 2) + v.safety_margin_m
    for left, right in sorted(itertools.product([0.0, margin, 2 * margin], repeat=2), key=sum):
        if left + right == 0 or left + right >= length - 0.1:
            continue
        a, b, yaw = task.start, task.end, task.start.yaw
        candidate = Task(task.task_id, Pose(a.x + left * math.cos(yaw), a.y + left * math.sin(yaw), yaw),
                         Pose(b.x - right * math.cos(yaw), b.y - right * math.sin(yaw), yaw), task.kind, task.cell_id)
        if not validator.motion_issues(work_motion(candidate, scene)):
            return candidate
    return None


def generate_patch_tasks(scene: Scene, backend: F2CBackend, missing, existing: list[Task],
                         budget: Budget, prefix: str = "patch") -> list[Task]:
    """漏作是新目标，而不是新的通行边界。生成少量直线补作任务，不沿剩余多边形再扣田头。"""
    validator = Validator(scene, budget)
    remaining = missing
    tasks = []
    width = scene.vehicle.working_width_m
    # 扩展扫描范围防止窄于幅宽的剩余区域根本没有扫描线；运动检查仍使用原 travel。
    search_area = polygonal(missing.buffer(width / 2).intersection(scene.travel))
    pool = []
    for angle_deg in scene.settings.angles_deg:
        for polygon in sorted(polygons(search_area), key=lambda p: -p.area):
            budget.check()
            native = backend.swaths(polygon, math.radians(angle_deg), "boustrophedon")
            for i in range(native.size()):
                task = task_from_swath(native.at(i), f"{prefix}_{len(pool)}", -1, "patch")
                fitted = _fit_straight(task, scene, validator)
                if fitted:
                    sweep = validator.work_sweep(work_motion(fitted, scene))
                    gain = sweep.intersection(missing).area
                    if gain >= scene.settings.patch_min_gain_m2:
                        pool.append((fitted, sweep, gain))
                if len(pool) >= scene.settings.max_tasks:
                    break
            if len(pool) >= scene.settings.max_tasks:
                break
    for task, sweep, _ in sorted(pool, key=lambda item: -item[2]):
        budget.check()
        if sweep.intersection(remaining).area < scene.settings.patch_min_gain_m2:
            continue
        tasks.append(task)
        remaining = polygonal(remaining.difference(sweep))
        if len(tasks) >= scene.settings.max_patch_tasks or remaining.area <= scene.settings.coverage_tolerance_m2:
            break
    return tasks


def generate_region_swaths(scene: Scene, analysis: WorkRegionAnalysis, swath_settings=None):
    """独立主体条带阶段的薄入口：消费冻结分区，具体候选、覆盖及接缝处理交给swath_planner，不启动最终路线规划。
    
    Run the isolated Fields2Cover body-swath stage on frozen regions.
    
    This wrapper deliberately leaves the partition and legacy full-route code
    above unchanged. See :mod:`swath_planner` for candidate search, safety,
    coverage, local-turn evidence, and result types."""
    from swath_planner import SwathSettings, generate_region_swaths as _generate

    return _generate(scene, analysis, swath_settings or SwathSettings())
