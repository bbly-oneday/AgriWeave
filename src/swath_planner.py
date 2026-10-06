"""V7冻结分区之后的主体条带规划与审计。输入Scene和WorkRegionAnalysis，输出候选主体条带、实际机具扫掠、面积账本及局部连接诊断。方向/相位/端部退让有限搜索后，先按全分区最大主体面积的保留门槛筛选，再比较条带数和转向代价。真实边界/孔洞负责退让，人工分区接缝不作真实边界。SWATHS_COMPLETE只表示主体阶段通过，局部转向未认证、田头未作业及完整路线未组装必须另报。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import Counter
import copy
import json
import math
import time
import warnings
from typing import Any, Iterable

import fields2cover as f2c
import numpy as np
from shapely import affinity, force_2d, set_precision, wkt
from shapely.geometry import GeometryCollection, LineString, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, substring, unary_union

from scene import Motion, Pose, Scene, Task, polygons, polygonal
from validator import Validator
from swath_seams import audit_seams, cross_region_assignments, overlapping_work_tasks


@dataclass(frozen=True)
class SwathSettings:
    """主体条带搜索与验收设置，独立于Scene.Settings；预算控制候选规模，不代表所有局部转向均已求解。
    
    Stage-specific bounded search controls; independent of frozen Settings."""

    max_directions: int = 4
    phase_steps: int = 4
    headland_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    area_retention_fraction: float = 0.95
    max_turn_connections: int = 600
    max_turn_order_checks: int = 48
    max_event_phases: int = 8
    max_joint_alignment_candidates: int = 4
    min_work_segment_m: float = 1.0
    max_seam_extra_segments_per_region: int = 1
    max_seam_extra_validation_per_region: int = 4
    max_local_seam_task_pairs: int = 8
    max_local_seam_trials: int = 48
    max_local_seam_rounds: int = 16

    def __post_init__(self) -> None:
        """检查条带阶段枚举、布尔类型与有限数值；拼错/非法参数不能静默采用默认值。"""
        for name in ("max_directions", "phase_steps", "max_turn_connections",
                     "max_turn_order_checks", "max_event_phases"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"SwathSettings.{name} 必须是正整数")
        if (type(self.max_joint_alignment_candidates) is not int
                or self.max_joint_alignment_candidates <= 0):
            raise ValueError("SwathSettings.max_joint_alignment_candidates 必须是正整数")
        if (type(self.max_seam_extra_segments_per_region) is not int
                or not 0 <= self.max_seam_extra_segments_per_region <= 2):
            raise ValueError("SwathSettings.max_seam_extra_segments_per_region 必须为 0–2")
        if (type(self.max_seam_extra_validation_per_region) is not int
                or not 1 <= self.max_seam_extra_validation_per_region <= 8):
            raise ValueError("SwathSettings.max_seam_extra_validation_per_region 必须为 1–8")
        for name, upper in (("max_local_seam_task_pairs", 32),
                            ("max_local_seam_trials", 256),
                            ("max_local_seam_rounds", 16)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"SwathSettings.{name} 必须为 1–{upper}")
        if (isinstance(self.min_work_segment_m, bool)
                or not isinstance(self.min_work_segment_m, (int, float))
                or not math.isfinite(float(self.min_work_segment_m))
                or float(self.min_work_segment_m) < 1.0):
            raise ValueError("SwathSettings.min_work_segment_m 必须不小于 1 m")
        if (not isinstance(self.headland_multipliers, (tuple, list))
                or not self.headland_multipliers):
            raise ValueError("SwathSettings.headland_multipliers 必须是非空数值序列")
        values = []
        for index, value in enumerate(self.headland_multipliers):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"SwathSettings.headland_multipliers[{index}] 必须是有限正数"
                )
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"SwathSettings.headland_multipliers[{index}] 必须是有限正数"
                )
            values.append(value)
        object.__setattr__(self, "headland_multipliers", tuple(values))
        value = self.area_retention_fraction
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or not 0 < float(value) <= 1):
            raise ValueError(
                "SwathSettings.area_retention_fraction 必须满足 0 < 值 <= 1"
            )


@dataclass
class Candidate:
    """一个方向、相位及退让组合的主体候选；保存真实线段和扫掠，转向证据与主体覆盖分开记录。"""
    angle_deg: float
    phase_m: float
    headland_m: float
    side_clearance_m: float
    main_area: BaseGeometry
    corridor: BaseGeometry
    line_geometries: list[tuple[int, LineString]]
    segments: list[dict[str, Any]]
    swept: BaseGeometry
    missing_area_m2: float
    coverage_fraction: float
    coverage_tolerance_passed: bool = False
    turn_checks: list[dict[str, Any]] = field(default_factory=list)
    turn_length_m: float = math.inf
    reverse_turn_count: int = 0
    turn_status: str = "NOT_CHECKED"
    order_mode: str = ""
    geometry_status: str = "REJECTED"

    @property
    def area_m2(self) -> float:
        """候选预定主体面积，单位平方米；不是机具扫掠面积之和。"""
        return float(self.main_area.area)

    @property
    def segment_count(self) -> int:
        """实际独立作业线段数量；孔洞裁断后的不同段分别计数。"""
        return len(self.segments)


def _select_body_candidate_pool(candidates: list[Candidate],
                                retention_fraction: float) -> tuple[float, float, int | None, int | None, list[Candidate]]:
    """先对同一分区全部有效候选建立统一最大主体面积及保留门槛，再比较条带数和局部转向，不能按方向各自降低门槛。
    
    Apply one region-wide area floor before comparing strip counts or turns."""
    valid = [item for item in candidates
             if item.coverage_tolerance_passed and item.geometry_status == "PASS"]
    amax = max((item.area_m2 for item in valid), default=0.0)
    threshold = amax * retention_fraction
    retained = [item for item in valid if item.area_m2 + 1e-8 >= threshold]
    body_min = min((item.segment_count for item in retained), default=None)
    turn_min = min((item.segment_count for item in retained
                    if item.turn_status == "PASS"), default=None)
    feasible = [item for item in retained if item.segment_count == body_min]
    return amax, threshold, body_min, turn_min, feasible


@dataclass
class RegionSwathResult:
    """单分区条带结果与面积账本；主体状态、局部转向及路线状态是不同证据。"""
    field_id: str
    region_id: str
    status: str
    angle_deg: float | None
    phase_m: float | None
    headland_m: float | None
    side_clearance_m: float | None
    segment_count: int
    amax_area_m2: float | None
    retained_fraction: float
    main_area: BaseGeometry
    headland_area: BaseGeometry
    pending_area: BaseGeometry
    segments: list[dict[str, Any]]
    work_sweeps: BaseGeometry
    turn_checks: list[dict[str, Any]]
    candidate_count: int
    f2c_call_count: int
    elapsed_s: float
    failure_reason: str = ""
    search_note: str = ""
    # 主体条带可以覆盖完整，但本阶段未必认证条带之间的转场；两种状态分开报告。
    turn_status: str = "NOT_CHECKED"
    required_main_area_m2: float = 0.0
    required_main_area: BaseGeometry = field(default_factory=GeometryCollection)
    body_feasible_min_segments: int | None = None
    turn_certified_min_segments: int | None = None
    search_candidates_uninspected: int = 0
    turn_search_stop_reason: str = "NOT_STARTED"
    profile: dict[str, float] = field(default_factory=dict)
    order_mode: str = ""
    pre_seam_segment_count: int | None = None

    @property
    def body_status(self) -> str:
        """返回主体条带阶段状态，不能据此声称局部连接或田头已完成。"""
        return "PASS" if self.status == "SWATHS_COMPLETE" else "FAILED"

    @property
    def local_turn_status(self) -> str:
        """返回局部转向检查状态；未认证结果保留待验证原因。"""
        return self.turn_status

    @property
    def route_status(self) -> str:
        """返回条带结果携带的路线相关状态；本模块不组装完整整田行程。"""
        return "NOT_ASSEMBLED"

    @property
    def local_turn_reason(self) -> str:
        """读取局部转向未通过或未完成的具体原因，区分预算限制与已检查失败。"""
        if self.turn_status == "PASS":
            return "ALL_LOCAL_TURNS_CERTIFIED"
        failed_codes = self.selected_turn_failure_codes
        if failed_codes:
            return "SELECTED_DIRECT_TURN_FAILED:" + ",".join(failed_codes)
        if any(row.get("status") == "FAIL" for row in self.turn_checks):
            return "SELECTED_DIRECT_TURN_FAILED:UNKNOWN"
        if any(row.get("status") == "NOT_CHECKED" for row in self.turn_checks):
            return "SELECTED_TURN_NOT_CHECKED"
        if self.turn_status in {"NOT_CHECKED", "SWATHS_NOT_FOUND"}:
            return "SELECTED_TURN_NOT_CHECKED"
        return "SELECTED_TURN_NOT_CERTIFIED"

    @property
    def selected_turn_failure_codes(self) -> list[str]:
        """提取选中候选的转向失败代码，便于追溯，不能用空清单推定实车通过。"""
        return sorted({code for row in self.turn_checks
                       if row.get("status") == "FAIL"
                       for code in row.get("failure_codes", [])})


@dataclass
class SwathStageResult:
    """整田主体条带汇总，包含分区、通道、接缝及面积账本；不等于最终作业路线。"""
    field_id: str
    status: str
    regions: list[RegionSwathResult]
    connections: list[dict[str, Any]]
    continuous_runs: list[dict[str, Any]]
    run_members: list[dict[str, Any]]
    target_area_m2: float
    covered_main_area_m2: float
    pending_area_m2: float
    elapsed_s: float
    f2c_version: str
    checks: dict[str, Any]
    acceptance_passed: bool = False
    whole_field_uncovered: BaseGeometry = field(default_factory=GeometryCollection)
    required_main_area: BaseGeometry = field(default_factory=GeometryCollection)
    work_sweeps: BaseGeometry = field(default_factory=GeometryCollection)
    body_covered_target_fraction: float = 0.0
    headland_reserved_fraction: float = 0.0
    pending_target_fraction: float = 0.0
    area_ledger_delta_m2: float = 0.0
    seam_quality_status: str = "NOT_EVALUATED"
    seam_pairs: list[dict[str, Any]] = field(default_factory=list)
    seam_overlap_footprint: BaseGeometry = field(default_factory=GeometryCollection)
    seam_assignments: list[dict[str, Any]] = field(default_factory=list)
    short_work_review: list[dict[str, Any]] = field(default_factory=list)
    seam_task_overlaps: list[dict[str, Any]] = field(default_factory=list)


def _angle_mod_180(angle_deg: float) -> float:
    """将无向条带角度归一化到0至180度，避免同方向重复搜索。"""
    value = float(angle_deg) % 180.0
    return 0.0 if abs(value - 180.0) < 1e-8 else value


def _stable_angles(scene: Scene, region: Any, settings: SwathSettings) -> list[float]:
    """由分区偏好、工作核心与几何边界提出有界方向集合，兼顾稳定性和搜索成本。
    
    Propose a small direction set from partition evidence and stable geometry."""
    proposals: list[tuple[float, float]] = []
    proposals.append((_angle_mod_180(region.preferred_angle_deg), 1e9))
    for geometry, weight in ((region.work_core, 1e8), (region.geometry, 1e7)):
        if geometry is None or geometry.is_empty or geometry.area <= 1e-8:
            continue
        parts = [part for part in polygons(geometry)
                 if part.area > max(1e-6, scene.vehicle.working_width_m ** 2 * 0.01)]
        if not parts:
            continue
        dominant = max(parts, key=lambda part: part.area)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                rectangle = dominant.minimum_rotated_rectangle
            except RuntimeWarning:
                continue
        if rectangle.is_empty or rectangle.geom_type != "Polygon":
            continue
        coords = list(rectangle.exterior.coords)
        edges = []
        for first, second in zip(coords, coords[1:]):
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            if length > scene.vehicle.working_width_m:
                angle = _angle_mod_180(math.degrees(math.atan2(dy, dx)))
                edges.append((length, angle))
        if edges:
            proposals.append((max(edges)[1], weight))
        for poly in polygons(geometry):
            coords = list(poly.exterior.coords)
            for first, second in zip(coords, coords[1:]):
                dx, dy = second[0] - first[0], second[1] - first[1]
                length = math.hypot(dx, dy)
                if length >= 3.0 * scene.vehicle.working_width_m:
                    proposals.append((_angle_mod_180(math.degrees(math.atan2(dy, dx))), length))
    for angle in scene.settings.angles_deg:
        proposals.append((_angle_mod_180(angle), 1.0))

    # 仅围绕几何证据较强的方向做少量细化，不遍历全部角度；小旋转可能改善凹角碎片。
    proposals.sort(key=lambda item: (-item[1], item[0]))
    bases = []
    for angle, _ in proposals:
        if all(abs((angle - old + 90.0) % 180.0 - 90.0) > 2.0 for old in bases):
            bases.append(angle)
        if len(bases) >= 2:
            break
    geometry_parts = polygons(region.geometry)
    has_holes = any(part.interiors for part in geometry_parts)
    has_multiple_parts = len(geometry_parts) > 1
    hull_area = float(region.geometry.convex_hull.area) if not region.geometry.is_empty else 0.0
    concavity_fraction = max(0.0, (hull_area - float(region.geometry.area)) / hull_area) \
        if hull_area > 0 else 0.0
    # 凸田块已提供稳定边方向；只有凹形、孔洞或碎片明显时才追加局部角度细化，减少短断条带。
    if has_holes or has_multiple_parts or concavity_fraction > 0.03:
        for base in bases[:2]:
            proposals.extend(((_angle_mod_180(base + offset), 0.5)
                              for offset in (-2.5, 2.5)))
    proposals.sort(key=lambda item: (-item[1], item[0]))
    result: list[float] = []
    for angle, _ in proposals:
        if all(abs((angle - old + 90.0) % 180.0 - 90.0) > 1.0 for old in result):
            result.append(angle)
        if len(result) >= settings.max_directions:
            break
    return result


def _phase_values(scene: Scene, analysis: Any, angle_deg: float,
                  spacing: float, steps: int, geometry: BaseGeometry | None = None,
                  max_event_phases: int = 8) -> list[float]:
    """生成共享条带网格与边界事件相位，单位米；几何拓扑改变时补充少量候选，不枚举无限相位。
    
    Combine a shared field grid with bounded geometry-event phases.
    
    Lane topology can change when a lane crosses a projected boundary vertex.
    Midpoints between those events represent stable phases within each interval."""
    radians = math.radians(angle_deg)
    nx, ny = -math.sin(radians), math.cos(radians)
    point = analysis.target.representative_point()
    base = (point.x * nx + point.y * ny) % spacing
    # 保留统一田块基准相位，并在一个条带间距内有限细化；相位仍以同一坐标基准的米计量。
    values = [base]
    values.extend((base + spacing * index / steps) % spacing for index in range(1, steps))
    if geometry is not None and not geometry.is_empty and max_event_phases:
        nx, ny = -math.sin(radians), math.cos(radians)
        events = sorted({round((x * nx + y * ny) % spacing, 9)
                         for part in polygons(geometry)
                         for ring in (part.exterior, *part.interiors)
                         for x, y, *_ in ring.coords})
        if len(events) > 1:
            intervals = []
            for index, left in enumerate(events):
                right = events[(index + 1) % len(events)]
                width = (right - left) % spacing
                if width > 1e-8:
                    intervals.append((width, (left + width / 2.0) % spacing))
            intervals.sort(key=lambda item: (-item[0], item[1]))
            if len(intervals) > max_event_phases:
                selected = [intervals[round(i * (len(intervals) - 1) /
                                           max(1, max_event_phases - 1))]
                            for i in range(max_event_phases)]
            else:
                selected = intervals
            values.extend(phase for _, phase in selected)
    return list(dict.fromkeys(round(value, 9) for value in values))


def _whole_field_coverage_audit(required_areas: Iterable[BaseGeometry],
                                sweep_areas: Iterable[BaseGeometry],
                                tolerance_m2: float) -> dict[str, Any]:
    """独立比较全部主体义务的并集与全部机具扫掠并集；跨分区代作可供覆盖，但不能静默丢失义务。
    
    Audit the union of required body area against the union of all sweeps."""
    required_items = [item for item in required_areas if item is not None and not item.is_empty]
    sweep_items = [item for item in sweep_areas if item is not None and not item.is_empty]
    required = unary_union(required_items) if required_items else GeometryCollection()
    swept = unary_union(sweep_items) if sweep_items else GeometryCollection()
    residual = polygonal(required.difference(swept))
    components = [part for part in polygons(residual) if part.area > 0]
    return {
        "required_area_m2": float(required.area),
        "swept_required_area_m2": float(required.intersection(swept).area),
        "residual_area_m2": float(residual.area),
        "maximum_residual_component_m2": max((float(part.area) for part in components), default=0.0),
        "residual_component_count": len(components),
        "tolerance_m2": float(tolerance_m2),
        "passed": float(residual.area) <= float(tolerance_m2),
        "geometry": residual,
        "required_geometry": required,
        "swept_geometry": swept,
    }


def _area_ledger_audit(scene: Scene, analysis: Any,
                       region_results: list[RegionSwathResult]) -> dict[str, Any]:
    """核对各冻结分区的目标归属及可加面积账本；合并面覆盖通过不能掩盖漏分配或重复分配。
    
    Check both target ownership and additive area, using a stable metric grid.
    
    The three classes are assigned per frozen region. Their *sum* is the area
    statistic; a bare union can silently expand a valid polygon in GEOS and
    make the reported headland fraction overlap body work. We still compare a
    precision-normalized union with the target to detect gaps and overlaps."""
    tolerance = max(float(scene.settings.coverage_tolerance_m2), 1e-8)
    grid = max(float(scene.settings.geometry_epsilon_m) / 100.0, 1e-9)
    by_id = {region.region_id: region.geometry for region in analysis.regions}
    area = {"main": 0.0, "headland": 0.0, "pending": 0.0}
    parts: list[BaseGeometry] = []
    local_delta = 0.0
    outside = 0.0
    pair_overlap = 0.0
    for result in region_results:
        region = set_precision(by_id[result.region_id], grid, mode="valid_output")
        local_parts = []
        for name, geometry in (("main", result.main_area),
                               ("headland", result.headland_area),
                               ("pending", result.pending_area)):
            area[name] += float(geometry.area)
            if geometry.is_empty:
                continue
            normalized = set_precision(geometry, grid, mode="valid_output")
            local_parts.append(normalized)
            parts.append(normalized)
            outside += float(normalized.difference(region).area)
        for index, first in enumerate(local_parts):
            pair_overlap += sum(float(first.intersection(second).area)
                                for second in local_parts[index + 1:])
        local_owned = unary_union(local_parts) if local_parts else GeometryCollection()
        local_delta += float(region.symmetric_difference(local_owned).area)
    owned = unary_union(parts) if parts else GeometryCollection()
    normalized_sum = sum(float(part.area) for part in parts)
    target = set_precision(scene.target, grid, mode="valid_output")
    target_delta = float(target.symmetric_difference(owned).area)
    raw_sum = sum(area.values())
    sum_delta = abs(raw_sum - float(scene.target.area))
    union_area_delta = abs(normalized_sum - float(owned.area))
    passed = all(value <= tolerance for value in (
        target_delta, local_delta, sum_delta, union_area_delta,
        pair_overlap, outside,
    ))
    return {
        "body_area_m2": area["main"],
        "headland_area_m2": area["headland"],
        "pending_area_m2": area["pending"],
        "area_ledger_delta_m2": target_delta,
        "area_sum_delta_m2": sum_delta,
        "area_union_area_delta_m2": union_area_delta,
        "area_region_delta_m2": local_delta,
        "area_pairwise_overlap_m2": pair_overlap,
        "area_outside_region_m2": outside,
        "area_ledger_tolerance_m2": tolerance,
        "area_ledger_precision_m": grid,
        "area_ledger_closed": passed,
    }


def _validate_analysis_source(scene: Scene, analysis: Any) -> bool:
    """校验冻结分区描述的target/travel与当前Scene完全一致，失配立即拒绝。
    
    Reject a frozen analysis snapshot that no longer describes this Scene."""
    if analysis.target.wkb != scene.target.wkb:
        raise ValueError("分区分析 target 与 Scene 不一致")
    if not hasattr(analysis, "travel") or analysis.travel.wkb != scene.travel.wkb:
        raise ValueError("分区分析 travel 与 Scene 不一致；请基于当前 Scene 重新分析")
    try:
        from planner import _vehicle_clearances

        lateral, footprint, turning = _vehicle_clearances(scene)
        epsilon = float(scene.settings.geometry_epsilon_m)
        guard = max(100 * epsilon, 1e-3)
        width = float(scene.vehicle.working_width_m)
        body_width = float(scene.vehicle.body_width_m)
        expected = {
            "lateral_clearance_m": lateral,
            "full_footprint_radius_m": footprint,
            "conservative_turn_clearance_m": turning,
            "numeric_buffer_guard_m": guard,
            "minimum_core_area_m2": max(2 * width ** 2, width * body_width),
            "minimum_region_area_m2": max(6 * width ** 2, 2 * turning * width),
            "minimum_region_width_m": 2 * width,
        }
        for name, value in expected.items():
            recorded = analysis.parameters.get(name)
            if recorded is None or not math.isclose(float(recorded), float(value),
                                                    rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"分区分析与当前车辆/规划参数不一致：{name}")
        current_safe_area = polygonal(scene.travel.buffer(
            -(footprint + guard), quad_segs=64,
        ))
        difference = float(analysis.safe_reference_area.symmetric_difference(
            current_safe_area).area)
        if difference > max(1e-8, epsilon ** 2):
            raise ValueError("分区分析 safe_reference_area 与当前车辆安全参数不一致")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"无法核验分区分析的车辆/规划参数指纹：{exc}") from exc
    return True


def _native_parallel_lines(target: BaseGeometry, angle_deg: float, phase_m: float,
                           spacing: float,
                           call_counter: list[int] | None = None) -> list[tuple[int, LineString]]:
    """在旋转支撑坐标中调用原生固定方向条带生成，并显式控制相位；裁剪后仍回到局部米制坐标。
    
    Generate F2C's fixed-angle lanes in a support frame with an explicit phase."""
    angle = math.radians(angle_deg)
    frame = affinity.rotate(target, -angle_deg, origin=(0.0, 0.0), use_radians=False)
    min_x, min_y, max_x, max_y = frame.bounds
    padding = max(10.0, spacing * 2.0)
    min_x -= padding
    max_x += padding
    # F2C首条带距支撑面下边半个间距；反推支撑边界，使实际网格恰为phase+k×spacing。
    k = math.floor((min_y - padding - phase_m) / spacing)
    first_center = phase_m + k * spacing
    y0 = first_center - spacing / 2.0
    y1 = max_y + padding + spacing
    cell_geometry = box(min_x, y0, max_x, y1)
    cell = f2c.Cell()
    cell.importFromWkt(cell_geometry.wkt)
    generator = f2c.SG_BruteForce()
    generator.setAllowOverlap(True)
    swaths = generator.generateSwaths(0.0, spacing, cell)
    if call_counter is not None:
        call_counter[0] += 1
    result: list[tuple[int, LineString]] = []
    for index in range(swaths.size()):
        path = force_2d(wkt.loads(swaths.at(index).getPath().exportToWkt()))
        if path.geom_type != "LineString" or path.length <= 1e-7:
            continue
        rotated = affinity.rotate(path, angle_deg, origin=(0.0, 0.0), use_radians=False)
        result.append((index, rotated))
    return result


def _line_parts(geometry: BaseGeometry) -> list[LineString]:
    """提取正长度线分量，孔洞或障碍裁断的各段保持独立，不能用端点直线填补间隙。"""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 1e-7 else []
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        parts: list[LineString] = []
        for item in geometry.geoms:
            parts.extend(_line_parts(item))
        return parts
    return []


def _anisotropic_core(geometry: BaseGeometry, angle_deg: float,
                      end_reserve_m: float, side_reserve_m: float) -> BaseGeometry:
    """按方向对真实外边界和障碍物留出端部/侧向空间；人工分区接缝不作为真实禁入边界。
    
    Erode true field/obstacle edges directionally; artificial seams are untouched."""
    rotated = affinity.rotate(geometry, -angle_deg, origin=(0.0, 0.0))
    scaled = affinity.scale(rotated, xfact=1.0 / end_reserve_m,
                            yfact=1.0 / side_reserve_m, origin=(0.0, 0.0))
    eroded = polygonal(scaled.buffer(-1.0, quad_segs=16))
    if eroded.is_empty:
        return eroded
    restored = affinity.scale(eroded, xfact=end_reserve_m,
                              yfact=side_reserve_m, origin=(0.0, 0.0))
    return polygonal(affinity.rotate(restored, angle_deg, origin=(0.0, 0.0)))


def _anisotropic_buffer(geometry: BaseGeometry, angle_deg: float,
                       along_m: float, across_m: float) -> BaseGeometry:
    """在旋转缩放坐标中构造纵横尺度不同的外扩面，沿向和横向距离均为米。"""
    rotated = affinity.rotate(geometry, -angle_deg, origin=(0.0, 0.0))
    scaled = affinity.scale(rotated, xfact=1.0 / along_m,
                            yfact=1.0 / across_m, origin=(0.0, 0.0))
    expanded = polygonal(scaled.buffer(1.0, quad_segs=16))
    if expanded.is_empty:
        return expanded
    restored = affinity.scale(expanded, xfact=along_m,
                              yfact=across_m, origin=(0.0, 0.0))
    return polygonal(affinity.rotate(restored, angle_deg, origin=(0.0, 0.0)))


def _ordered_segments(segments: list[dict[str, Any]], angle_deg: float,
                      mode: str) -> list[dict[str, Any]]:
    """按投影行号及往复模式排序独立线段；排序不合并跨孔洞的任务，也不等于转弯求解。"""
    angle = math.radians(angle_deg)
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    rows: dict[int, list[dict[str, Any]]] = {}
    for segment in segments:
        coords = list(segment["geometry"].coords)
        midx = (coords[0][0] + coords[-1][0]) / 2
        midy = (coords[0][1] + coords[-1][1]) / 2
        row = int(segment["row_index"])
        rows.setdefault(row, []).append({**segment, "_u": midx * ux + midy * uy,
                                         "_v": midx * vx + midy * vy})
    row_ids = sorted(rows, key=lambda key: np.mean([item["_v"] for item in rows[key]]))
    if mode == "skip_row_then_fill":
        ordered_rows = row_ids[::2] + row_ids[1::2]
    else:
        ordered_rows = row_ids
    ordered: list[dict[str, Any]] = []
    for row_position, row_id in enumerate(ordered_rows):
        items = sorted(rows[row_id], key=lambda item: item["_u"],
                       reverse=bool(row_position % 2))
        ordered.extend({**item, "_reverse": bool(row_position % 2)} for item in items)
    return ordered


def _task_motion(segment: dict[str, Any], scene: Scene, *, reverse: bool) -> tuple[Task, Motion]:
    """按指定遍历方向构造该条带的直线作业运动；反向遍历与真实倒挡不是同一概念。"""
    coords = list(segment["geometry"].coords)
    first, last = coords[0], coords[-1]
    if reverse:
        first, last = last, first
    yaw = math.atan2(last[1] - first[1], last[0] - first[0])
    task = Task(segment["task_id"], Pose(first[0], first[1], yaw),
                Pose(last[0], last[1], yaw), "work", segment["region_index"])
    length = math.dist(first, last)
    if length <= 1e-7:
        raise ValueError("零长度连续作业段")
    step = min(scene.settings.sampling_step_m, length / 3.0)
    t = np.asarray([0.0, step / length, 1.0 - step / length, 1.0])
    points = np.column_stack((
        first[0] + t * (last[0] - first[0]),
        first[1] + t * (last[1] - first[1]),
        np.full(4, yaw), np.ones(4),
    ))
    motion = Motion(points, "work", task.task_id, implement_on=True)
    return task, motion


def _check_turn_order(candidate: Candidate, scene: Scene, validator: Validator,
                      backend: Any, region_index: int, mode: str,
                      max_connections: int) -> tuple[Any, ...]:
    """有界检查候选排序中的局部连接，保存每条连接状态/耗时；预算未检查的连接不能当作通过。"""
    ordered = _ordered_segments(candidate.segments, candidate.angle_deg, mode)
    checks: list[dict[str, Any]] = []
    total_length = 0.0
    reverse_count = 0
    work_sweeps: list[BaseGeometry] = []
    for segment in ordered:
        _, work = _task_motion(segment, scene, reverse=segment["_reverse"])
        issues = validator.motion_issues(work)
        if issues:
            checks.append({
                "field_id": scene.name, "region_id": segment["region_id"],
                "check_index": len(checks), "check_type": "straight_swath",
                "from_task": segment["task_id"], "to_task": None,
                "status": "FAIL", "turner": None,
                "failure_codes": sorted({item["code"] for item in issues}),
                "failure_xy_local_m": next(
                    (item.get("xy_local_m") for item in issues
                     if item.get("xy_local_m") is not None), None,
                ),
                "geometry": segment["geometry"], "path_length_m": None,
                "uses_shared_travel": False,
            })
            return False, checks, math.inf, 0, "SWATH_ENVELOPE_INVALID:" + ",".join(
                sorted({item["code"] for item in issues})
            )
        work_sweeps.append(validator.work_sweep(work))
    actual_swept = unary_union(work_sweeps) if work_sweeps else GeometryCollection()
    missing_area = float(candidate.main_area.difference(actual_swept).area)
    if missing_area > scene.settings.coverage_tolerance_m2:
        return False, [], math.inf, 0, f"ORIENTED_SWATH_COVERAGE_GAP:{missing_area:.6f}m2"
    if len(ordered) - 1 > max_connections:
        checks.append({
            "field_id": scene.name, "region_id": candidate.segments[0]["region_id"],
            "check_index": 0, "check_type": "turn_sequence_limit",
            "from_task": None, "to_task": None, "status": "NOT_CHECKED",
            "turner": None, "failure_codes": ["TURN_CHECK_LIMIT_REACHED"],
            "geometry": GeometryCollection(), "path_length_m": None,
            "uses_shared_travel": False,
        })
        return False, checks, math.inf, 0, "TURN_CHECK_LIMIT_REACHED", actual_swept
    for index, (previous, following) in enumerate(zip(ordered, ordered[1:])):
        _, previous_motion = _task_motion(previous, scene, reverse=previous["_reverse"])
        _, next_motion = _task_motion(following, scene, reverse=following["_reverse"])
        end_row = previous_motion.points[-1]
        start_row = next_motion.points[0]
        start_pose = Pose(float(end_row[0]), float(end_row[1]), float(end_row[2]))
        end_pose = Pose(float(start_row[0]), float(start_row[1]), float(start_row[2]))
        chosen = None
        failures: list[str] = []
        failure_points: list[list[float]] = []
        for name, turner in backend.turners:
            try:
                path = backend.turn_path(start_pose, end_pose, turner)
                motion = backend.convert_path(path, (previous["task_id"], following["task_id"]), "turn")
                issues = validator.motion_issues(motion)
                if issues:
                    failures.extend(item["code"] for item in issues)
                    failure_points.extend(
                        item["xy_local_m"] for item in issues
                        if item.get("xy_local_m") is not None
                    )
                    continue
                chosen = (name, motion)
                break
            except Exception as exc:
                failures.append(type(exc).__name__)
        record = {
            "field_id": scene.name, "region_id": candidate.segments[0]["region_id"],
            "check_index": index, "from_task": previous["task_id"],
            "to_task": following["task_id"], "status": "PASS" if chosen else "FAIL",
            "turner": chosen[0] if chosen else None,
            "failure_codes": sorted(set(failures)),
            "failure_xy_local_m": failure_points[0] if failure_points else None,
            "geometry": LineString(chosen[1].points[:, :2]) if chosen else GeometryCollection(),
            "path_length_m": chosen[1].length if chosen else None,
            "uses_shared_travel": bool(chosen and not candidate.segments[0]["region_geometry"].buffer(
                scene.settings.geometry_epsilon_m).covers(LineString(chosen[1].points[:, :2]))),
        }
        checks.append(record)
        if chosen is None:
            # 直线条带安全与覆盖已通过时保留主体；此连接仍须后续绕障求解，直接Dubins/Reeds-Shepp失败不等于完整路线搜索失败。
            return (False, checks, math.inf, reverse_count,
                    "LOCAL_TURN_NOT_VALIDATED", actual_swept)
        total_length += chosen[1].length
        if np.any(chosen[1].points[:-1, 3] < 0):
            reverse_count += 1
    return True, checks, total_length, reverse_count, "PASS", actual_swept


def _evaluate_candidate(scene: Scene, region: Any, region_index: int,
                        analysis: Any, validator: Validator,
                        angle_deg: float, phase_m: float, headland_m: float,
                        side_reserve_m: float, spacing: float,
                        region_id: str, call_counter: list[int] | None = None,
                        field_lines: list[tuple[int, LineString]] | None = None,
                        corridor_mode: str = "directional") -> Candidate:
    """生成并裁剪一个主体候选，检查直线车体安全、实际扫掠和覆盖；局部转向另行验证。"""
    physical_side_clearance = max(scene.vehicle.body_width_m,
                                  scene.vehicle.working_width_m) / 2.0
    physical_side_clearance += (scene.vehicle.safety_margin_m
                                + scene.settings.travel_clearance_m)
    # 安全参考区已经按车体及机具完整包络外接半径、安全余量和数值保护退让；自适应侧向预留只增加额外组织/转向空间。
    additional_side_reserve = max(
        scene.settings.geometry_epsilon_m,
        side_reserve_m - physical_side_clearance,
    )
    center_zone = polygonal(_anisotropic_core(
        analysis.target, angle_deg, headland_m, additional_side_reserve,
    ).intersection(scene.travel).intersection(analysis.safe_reference_area))
    # ``center_zone`` is a legal domain for the vehicle reference point, not the
    # 后置机具在车辆参考点后仍有|offset|+implement_length/2的拖尾；主体需留出双向作业端部空间。横向按实际机具半宽退让，不能按较小搭接间距退让。
    tool_longitudinal_reach = max(
        abs(scene.vehicle.implement_offset_m) + scene.vehicle.implement_length_m / 2,
        scene.vehicle.front_m,
        scene.vehicle.rear_m,
        1e-3,
    )
    implement_half_width = max(scene.vehicle.working_width_m / 2.0, 1e-3)
    main_global = _anisotropic_core(
        center_zone, angle_deg, tool_longitudinal_reach, implement_half_width,
    )
    main_area = polygonal(region.geometry.intersection(main_global))
    if main_area.is_empty or main_area.area <= scene.settings.coverage_tolerance_m2:
        raise ValueError("ADAPTIVE_HEADLAND_LEAVES_NO_MAIN_AREA")
    extension = max(abs(scene.vehicle.implement_offset_m) +
                    scene.vehicle.implement_length_m / 2,
                    scene.vehicle.front_m, scene.vehicle.rear_m)
    # 人工接缝横向只需机具半宽关联范围；过大的各向同性外扩会把远在另一分区内的平行条带误纳入。
    if corridor_mode == "directional":
        support = _anisotropic_buffer(
            region.geometry, angle_deg,
            max(extension, 1e-3) + scene.settings.geometry_epsilon_m,
            implement_half_width + scene.settings.geometry_epsilon_m,
        )
    elif corridor_mode == "legacy_safe_fallback":
        support = region.geometry.buffer(extension + spacing)
    else:
        raise ValueError(f"UNKNOWN_SWATH_CORRIDOR_MODE:{corridor_mode}")
    corridor = polygonal(center_zone.intersection(scene.travel).intersection(support))
    if corridor.is_empty:
        raise ValueError("NO_TRAVEL_CORRIDOR_AROUND_REGION")
    if field_lines is None:
        field_lines = _native_parallel_lines(analysis.target, angle_deg, phase_m,
                                             spacing, call_counter)
    pieces: list[tuple[int, LineString]] = []
    for row_index, line in field_lines:
        for part in _line_parts(line.intersection(corridor)):
            if part.length > max(scene.settings.geometry_epsilon_m, 1e-4):
                pieces.append((row_index, part))
    if not pieces:
        raise ValueError("F2C_GENERATED_NO_CONTINUOUS_SEGMENTS")

    segments: list[dict[str, Any]] = []
    sweeps: list[BaseGeometry] = []
    for serial, (row_index, line) in enumerate(pieces):
        segment_id = f"{region_id}_s{serial + 1:04d}"
        segment = {"field_id": scene.name, "region_id": region_id,
                   "region_index": region_index, "task_id": segment_id,
                   "row_index": row_index, "angle_deg": angle_deg,
                   "phase_m": phase_m, "length_m": float(line.length),
                   "geometry": line, "region_geometry": region.geometry}
        # 构造候选只生成几何，真正选中排序时再检查精确行驶方向及完整包络；避免为未选相位重复检查两种方向。
        segments.append(segment)
    swept = GeometryCollection()
    missing_area = 0.0
    coverage_fraction = 0.0
    return Candidate(angle_deg, phase_m, headland_m, side_reserve_m, main_area, corridor,
                     field_lines, segments, swept, missing_area,
                     coverage_fraction, coverage_tolerance_passed=False,
                     geometry_status="PASS")


def _estimate_main_area(scene: Scene, analysis: Any, region: Any,
                        angle_deg: float, headland_m: float,
                        side_clearance_m: float) -> float:
    """廉价估算主体面积上界，用于跳过明显无竞争力的退让组合；不是最终覆盖证明。
    
    Cheap area upper bound used to prune adaptive side/end-margin batches."""
    physical_side_clearance = max(scene.vehicle.body_width_m,
                                  scene.vehicle.working_width_m) / 2.0
    physical_side_clearance += (scene.vehicle.safety_margin_m
                                + scene.settings.travel_clearance_m)
    additional_side = max(scene.settings.geometry_epsilon_m,
                          side_clearance_m - physical_side_clearance)
    center_zone = polygonal(_anisotropic_core(
        analysis.target, angle_deg, headland_m, additional_side,
    ).intersection(scene.travel).intersection(analysis.safe_reference_area))
    tool_longitudinal_reach = max(
        abs(scene.vehicle.implement_offset_m) + scene.vehicle.implement_length_m / 2,
        scene.vehicle.front_m,
        scene.vehicle.rear_m,
        1e-3,
    )
    implement_half_width = max(scene.vehicle.working_width_m / 2.0, 1e-3)
    main = _anisotropic_core(
        center_zone, angle_deg, tool_longitudinal_reach, implement_half_width,
    )
    return float(region.geometry.intersection(main).area)


def _candidate_orders(candidate: Candidate, scene: Scene, validator: Validator,
                      backend: Any, region_index: int, settings: SwathSettings,
                      cache: dict[tuple[str, str], tuple[Any, ...]],
                      modes: tuple[str, ...] = ("serpentine", "skip_row_then_fill")) -> list[Candidate]:
    """对候选尝试有限往复排序并复用连接检查缓存；不能以转向代价突破统一面积门槛。"""
    variants: list[Candidate] = []
    for mode in modes:
        # 只有角度和相位一致时端点对才会重复；缓存限本分区，候选几何各自保留。
        ordered = _ordered_segments(candidate.segments, candidate.angle_deg, mode)
        cache_key = (repr((candidate.main_area.wkb,
                           [(tuple(item["geometry"].coords), item["_reverse"])
                            for item in ordered])), mode)
        if cache_key not in cache:
            cache[cache_key] = _check_turn_order(
                candidate, scene, validator, backend, region_index, mode,
                settings.max_turn_connections,
            )
        turn_result = cache[cache_key]
        passed, checks, length, reverse_count, reason = turn_result[:5]
        item = Candidate(**{key: value for key, value in candidate.__dict__.items()
                            if key not in {"turn_checks", "turn_length_m", "reverse_turn_count",
                                           "turn_status", "order_mode"}})
        item.turn_checks = checks
        item.turn_length_m = length
        item.reverse_turn_count = reverse_count
        item.turn_status = reason if not passed else "PASS"
        item.order_mode = mode
        unresolved_local_transfer = (
            not passed and len(turn_result) >= 6
            and reason in {"LOCAL_TURN_NOT_VALIDATED", "TURN_CHECK_LIMIT_REACHED"}
        )
        if passed or unresolved_local_transfer:
            item.swept = turn_result[5]
            missing_area = float(item.main_area.difference(item.swept).area)
            item.missing_area_m2 = missing_area
            item.coverage_fraction = max(0.0, min(1.0, 1.0 - missing_area / item.main_area.area))
            item.coverage_tolerance_passed = missing_area <= scene.settings.coverage_tolerance_m2
            if not item.coverage_tolerance_passed:
                item.turn_status = f"ORIENTED_SWATH_COVERAGE_GAP:{missing_area:.6f}m2"
                continue
            variants.append(item)
        else:
            candidate.turn_checks = checks
            candidate.turn_status = reason
    return variants


def _candidate_body_orders(candidate: Candidate, scene: Scene,
                           validator: Validator,
                           modes: tuple[str, ...] = ("serpentine", "skip_row_then_fill"),
                           fallback_reason: str = "TURN_CHECK_BUDGET_EXHAUSTED") -> list[Candidate]:
    """转向预算耗尽时保留已验证直线作业与覆盖的主体候选，明确标注局部转向未认证。
    
    Validate straight work and coverage when local-turn search is exhausted.
    
    This fallback does not claim that any adjacent swaths can be connected. It
    exists so the bounded turn-search budget cannot hide a lower-strip-count
    body plan that satisfies the same safety and coverage checks."""
    variants: list[Candidate] = []
    for mode in modes:
        ordered = _ordered_segments(candidate.segments, candidate.angle_deg, mode)
        work_sweeps: list[BaseGeometry] = []
        invalid = False
        for segment in ordered:
            _, motion = _task_motion(segment, scene, reverse=segment["_reverse"])
            if validator.motion_issues(motion):
                invalid = True
                break
            work_sweeps.append(validator.work_sweep(motion))
        if invalid or not work_sweeps:
            continue
        swept = unary_union(work_sweeps)
        missing_area = float(candidate.main_area.difference(swept).area)
        if missing_area > scene.settings.coverage_tolerance_m2:
            continue
        item = Candidate(**{
            key: value for key, value in candidate.__dict__.items()
            if key not in {"turn_checks", "turn_length_m", "reverse_turn_count",
                           "turn_status", "order_mode"}
        })
        item.swept = swept
        item.missing_area_m2 = missing_area
        item.coverage_fraction = max(
            0.0, min(1.0, 1.0 - missing_area / item.main_area.area),
        )
        item.coverage_tolerance_passed = True
        item.turn_checks = [{
            "field_id": scene.name,
            "region_id": candidate.segments[0]["region_id"],
            "check_index": 0,
            "check_type": "local_turn_sequence",
            "from_task": None,
            "to_task": None,
            "status": "NOT_CHECKED",
            "turner": None,
            "failure_codes": [fallback_reason],
            "geometry": GeometryCollection(),
            "path_length_m": None,
            "uses_shared_travel": False,
        }]
        item.turn_length_m = math.inf
        item.reverse_turn_count = 0
        item.turn_status = "LOCAL_TURN_NOT_VALIDATED"
        item.order_mode = mode
        variants.append(item)
        # 没有认证局部转向时，第二排序不能改善主体数或面积；先用往复，只有直线安全失败时再尝试跳行。
        break
    return variants


def _connection_records(analysis: Any, field_id: str) -> list[dict[str, Any]]:
    """将分区通道拓扑转换为记录，保留通行约束；拓扑邻接本身不构成运动连接。"""
    return [{
        "field_id": field_id, "connection_id": item.connection_id,
        "from_region": item.from_region, "to_region": item.to_region,
        "width_class": item.width_class, "effective_width_m": item.effective_width_m,
        "straight_status": item.straight_status, "turn_status": item.turn_status,
        "evidence": item.evidence, "mouth_wkt": item.mouth.wkt,
        "geometry": (item.portal if item.portal is not None
                     else (item.mouth.representative_point() if not item.mouth.is_empty
                           else Point(0.0, 0.0))),
    } for item in analysis.connections]


def _joint_run_candidates(scene: Scene, shared: BaseGeometry,
                          first: Candidate, second: Candidate,
                          first_mode: str, second_mode: str,
                          validator: Validator) -> list[dict[str, Any]]:
    """寻找两区同网格条带可安全直行衔接的候选，按真实机具扫掠验证。
    
    Return physically checked straight joins shared by two region candidates."""
    first_order = _ordered_segments(first.segments, first.angle_deg, first_mode)
    second_order = _ordered_segments(second.segments, second.angle_deg, second_mode)
    first_sweeps = {}
    second_sweeps = {}
    for segment in first_order:
        _, motion = _task_motion(segment, scene, reverse=segment["_reverse"])
        first_sweeps[segment["task_id"]] = validator.work_sweep(motion)
    for segment in second_order:
        _, motion = _task_motion(segment, scene, reverse=segment["_reverse"])
        second_sweeps[segment["task_id"]] = validator.work_sweep(motion)
    result = []
    tolerance = max(1e-5, scene.settings.geometry_epsilon_m * 10)
    for a in first.segments:
        for b in second.segments:
            if a["row_index"] != b["row_index"] or a["geometry"].distance(b["geometry"]) > tolerance:
                continue
            merged = unary_union([a["geometry"], b["geometry"]])
            if merged.geom_type != "LineString":
                try:
                    merged = linemerge(merged)
                except ValueError:
                    continue
            if merged.geom_type != "LineString":
                continue
            if merged.length <= max(a["geometry"].length, b["geometry"].length) + tolerance:
                continue
            if not shared.is_empty and merged.intersection(shared).is_empty:
                continue
            required = unary_union([first_sweeps[a["task_id"]],
                                    second_sweeps[b["task_id"]]])
            coords = list(merged.coords)
            for reverse in (False, True):
                start, end = (coords[-1], coords[0]) if reverse else (coords[0], coords[-1])
                yaw = math.atan2(end[1] - start[1], end[0] - start[0])
                length = math.dist(start[:2], end[:2])
                step = min(scene.settings.sampling_step_m, length / 3.0)
                t = np.asarray([0.0, step / length, 1.0 - step / length, 1.0])
                points = np.column_stack((
                    start[0] + t * (end[0] - start[0]),
                    start[1] + t * (end[1] - start[1]),
                    np.full(4, yaw), np.ones(4),
                ))
                motion = Motion(points, "work", "joint_alignment_probe", True)
                if validator.motion_issues(motion):
                    continue
                swept = validator.work_sweep(motion)
                if required.difference(swept).area > scene.settings.coverage_tolerance_m2:
                    continue
                result.append({"row_index": int(a["row_index"]),
                               "length_m": float(merged.length),
                               "geometry": merged})
                break
    return result


def _apply_joint_candidate(scene: Scene, region: Any, current: RegionSwathResult,
                           candidate: Candidate, validator: Validator) -> None:
    """安全联合择优后替换一个分区候选，同时更新选中状态与账本；原分区目标不变。
    
    Replace one region's selected candidate after a safe joint comparison."""
    ordered = _ordered_segments(candidate.segments, candidate.angle_deg,
                                candidate.order_mode)
    row_order = {segment["task_id"]: index + 1
                 for index, segment in enumerate(ordered)}
    ordered_by_task = {segment["task_id"]: segment for segment in ordered}
    output_segments = []
    for segment in candidate.segments:
        oriented = ordered_by_task[segment["task_id"]]
        _, motion = _task_motion(segment, scene, reverse=oriented["_reverse"])
        output_segments.append({
            **{key: value for key, value in segment.items()
               if key not in {"region_geometry", "sweep"}},
            "suggested_order": row_order[segment["task_id"]],
            "heading_deg": math.degrees(float(motion.points[0, 2])) % 360.0,
            "implement_on": True, "geometry": segment["geometry"],
            "work_sweep": validator.work_sweep(motion),
        })
    pending = polygonal(candidate.main_area.difference(candidate.swept))
    main = polygonal(candidate.swept.intersection(region.geometry).difference(pending))
    headland = polygonal(region.geometry.difference(unary_union([main, pending])))
    current.angle_deg = candidate.angle_deg
    current.phase_m = candidate.phase_m
    current.headland_m = candidate.headland_m
    current.side_clearance_m = candidate.side_clearance_m
    current.segment_count = candidate.segment_count
    old_amax = current.amax_area_m2 or 0.0
    current.amax_area_m2 = max(current.amax_area_m2 or 0.0, candidate.area_m2)
    current.retained_fraction = (candidate.area_m2 / current.amax_area_m2
                                 if current.amax_area_m2 else 1.0)
    current.main_area, current.headland_area = main, headland
    current.pending_area, current.work_sweeps = pending, candidate.swept
    current.segments, current.turn_checks = output_segments, candidate.turn_checks
    current.turn_status, current.order_mode = candidate.turn_status, candidate.order_mode
    current.required_main_area_m2 = candidate.area_m2
    current.required_main_area = candidate.main_area
    current.body_feasible_min_segments = min(
        current.body_feasible_min_segments or candidate.segment_count,
        candidate.segment_count,
    )
    if current.amax_area_m2 > old_amax + 1e-8:
        # 新检查到的更大主体会抬高统一面积门槛，旧认证候选可能因此不再符合保留条件。
        current.turn_certified_min_segments = None
    if candidate.turn_status == "PASS":
        current.turn_certified_min_segments = min(
            current.turn_certified_min_segments or candidate.segment_count,
            candidate.segment_count,
        )
    current.candidate_count += 1
    current.search_note += "; joint_direction_phase_candidate=selected"


def _coordinate_seam_candidates(scene: Scene, analysis: Any,
                                results: list[RegionSwathResult],
                                pools: dict[str, list[Candidate]],
                                validator: Validator,
                                settings: SwathSettings | None = None) -> dict[str, Any]:
    """在有界候选中比较真实接缝扫掠，先替换同条带数方案，再尝试安全删除完整冗余条带。
    
    Bounded same-count physical replacements, then safe whole-strip removal.
    
    The comparison is always made on actual implement sweeps. A replacement
    must retain the already selected whole-field body target and cannot lower
    a certified local-turn result to an uncertified one."""
    started = time.perf_counter()
    settings = settings or SwathSettings()
    by_id = {item.region_id: item for item in results}
    frozen = {item.region_id: item for item in analysis.regions}
    baseline_required = unary_union([
        item.required_main_area for item in results
        if item.status == "SWATHS_COMPLETE" and not item.required_main_area.is_empty
    ])
    tolerance = scene.settings.coverage_tolerance_m2
    inspected = accepted = removed = pair_inspected = pair_accepted = 0
    stop_reason = "NO_IMPROVEMENT"
    report = audit_seams(scene, analysis, results)

    def keeps_body(trial: RegionSwathResult) -> bool:
        sweep_union = unary_union([
            trial.work_sweeps if item.region_id == trial.region_id else item.work_sweeps
            for item in results if item.status == "SWATHS_COMPLETE"
        ])
        return baseline_required.difference(sweep_union).area <= tolerance

    for round_index in range(3):
        improved = False
        pairs = sorted(report["pairs"],
                       key=lambda row: (-row["overlap_area_m2"], row["from_region"],
                                        row["to_region"]))
        for pair in pairs:
            if pair["overlap_area_m2"] <= tolerance:
                continue
            for region_id in (pair["from_region"], pair["to_region"]):
                current = by_id[region_id]
                if current.status != "SWATHS_COMPLETE":
                    continue
                # 仅删除完整开机具任务；已认证局部转向序列不能无声丢掉其中一条作业。
                if current.turn_status != "PASS" and len(current.segments) > 1:
                    candidates_to_remove = sorted(
                        current.segments,
                        key=lambda segment: (-segment["geometry"].difference(
                            frozen[region_id].geometry).length, segment["task_id"]),
                    )[:8]
                    for segment in candidates_to_remove:
                        if segment["work_sweep"].intersection(
                                by_id[pair["to_region"] if region_id == pair["from_region"]
                                      else pair["from_region"]].work_sweeps).area <= tolerance:
                            continue
                        remaining = [row for row in current.segments
                                     if row["task_id"] != segment["task_id"]]
                        trial = copy.copy(current)
                        trial.segments = remaining
                        trial.work_sweeps = unary_union([row["work_sweep"] for row in remaining])
                        trial.segment_count = len(remaining)
                        inspected += 1
                        if not keeps_body(trial):
                            continue
                        trial_results = [trial if row.region_id == region_id else row
                                         for row in results]
                        measured = audit_seams(scene, analysis, trial_results)
                        if measured["extra_area_m2"] >= report["extra_area_m2"] - tolerance:
                            continue
                        current.segments = remaining
                        for position, row in enumerate(sorted(
                                remaining, key=lambda value: value["suggested_order"]), 1):
                            row["suggested_order"] = position
                        current.work_sweeps = trial.work_sweeps
                        current.segment_count = trial.segment_count
                        current.turn_checks = []
                        current.turn_status = "LOCAL_TURN_NOT_VALIDATED"
                        current.search_note += "; seam_redundant_whole_segment_removed"
                        report = measured
                        removed += 1
                        improved = True
                        break
                # 候选替代先已检查直线运动、实际机具覆盖和统一95%面积门槛，联合替换不能取消这些条件。
                best = None
                for candidate in pools.get(region_id, ()):
                    baseline_count = current.pre_seam_segment_count or current.segment_count
                    if candidate.segment_count > baseline_count + settings.max_seam_extra_segments_per_region:
                        continue
                    if current.turn_status == "PASS" and candidate.turn_status != "PASS":
                        continue
                    if candidate.area_m2 + 1e-8 < (current.amax_area_m2 or 0) * 0.95:
                        continue
                    if (abs(candidate.angle_deg - float(current.angle_deg)) < 1e-8
                            and abs(candidate.phase_m - float(current.phase_m)) < 1e-8
                            and abs(candidate.headland_m - float(current.headland_m)) < 1e-8
                            and abs(candidate.side_clearance_m - float(current.side_clearance_m)) < 1e-8):
                        continue
                    trial = copy.copy(current)
                    trial.work_sweeps = candidate.swept
                    trial.segments = candidate.segments
                    trial.required_main_area = candidate.main_area
                    inspected += 1
                    if not keeps_body(trial):
                        continue
                    trial_results = [trial if row.region_id == region_id else row
                                     for row in results]
                    measured = audit_seams(scene, analysis, trial_results)
                    score = (measured["extra_area_m2"], candidate.segment_count,
                             candidate.angle_deg, candidate.phase_m)
                    added = max(0, candidate.segment_count - current.segment_count)
                    gain = report["extra_area_m2"] - score[0]
                    min_gain = (tolerance if not added else max(
                        2.0 * scene.vehicle.working_width_m ** 2 * added,
                        0.15 * pair["overlap_area_m2"],
                    ))
                    if gain <= min_gain:
                        continue
                    if best is None or score < best[0]:
                        best = (score, candidate, measured)
                if best is not None:
                    _apply_joint_candidate(scene, frozen[region_id], current,
                                           best[1], validator)
                    report = best[2]
                    accepted += 1
                    improved = True
            # 双侧同时调整可走出单侧局部最优；每侧最多少量候选，只对最高超额重叠接缝并在廉价单侧尝试后进行。
            left_id, right_id = pair["from_region"], pair["to_region"]
            current_pair = next((row for row in report["pairs"]
                                 if {row["from_region"], row["to_region"]}
                                 == {left_id, right_id}), None)
            if current_pair is None or current_pair["status"] == "PASS":
                continue

            def joint_options(region_id: str) -> list[Candidate | None]:
                current = by_id[region_id]
                baseline_count = current.pre_seam_segment_count or current.segment_count
                candidates = [item for item in pools.get(region_id, ())
                              if item.segment_count <= baseline_count
                              + settings.max_seam_extra_segments_per_region
                              and item.area_m2 + 1e-8 >= (current.amax_area_m2 or 0.0)
                              * settings.area_retention_fraction
                              and (current.turn_status != "PASS"
                                   or item.turn_status == "PASS")]
                same = [item for item in candidates
                        if item.segment_count <= current.segment_count]
                extra = [item for item in candidates
                         if item.segment_count > current.segment_count]
                return [None] + same[:1] + extra[:2]

            pair_best = None
            for first in joint_options(left_id):
                for second in joint_options(right_id):
                    if first is None and second is None:
                        continue
                    first_trial = copy.copy(by_id[left_id])
                    second_trial = copy.copy(by_id[right_id])
                    if first is not None:
                        first_trial.work_sweeps = first.swept
                        first_trial.segments = first.segments
                    if second is not None:
                        second_trial.work_sweeps = second.swept
                        second_trial.segments = second.segments
                    pair_inspected += 1
                    sweep_union = unary_union([
                        first_trial.work_sweeps if row.region_id == left_id
                        else second_trial.work_sweeps if row.region_id == right_id
                        else row.work_sweeps for row in results
                        if row.status == "SWATHS_COMPLETE"
                    ])
                    if baseline_required.difference(sweep_union).area > tolerance:
                        continue
                    trial_results = [first_trial if row.region_id == left_id
                                     else second_trial if row.region_id == right_id
                                     else row for row in results]
                    measured = audit_seams(scene, analysis, trial_results)
                    gain = report["extra_area_m2"] - measured["extra_area_m2"]
                    added = sum(max(0, option.segment_count - by_id[region_id].segment_count)
                                for option, region_id in ((first, left_id), (second, right_id))
                                if option is not None)
                    minimum_gain = (tolerance if not added else max(
                        2.0 * scene.vehicle.working_width_m ** 2 * added,
                        0.15 * current_pair["overlap_area_m2"],
                    ))
                    if gain <= minimum_gain:
                        continue
                    score = (measured["extra_area_m2"], added,
                             first.angle_deg if first else -1.0,
                             second.angle_deg if second else -1.0)
                    if pair_best is None or score < pair_best[0]:
                        pair_best = (score, first, second, measured)
            if pair_best is not None:
                _, first, second, measured = pair_best
                for region_id, candidate in ((left_id, first), (right_id, second)):
                    if candidate is not None:
                        _apply_joint_candidate(scene, frozen[region_id],
                                               by_id[region_id], candidate, validator)
                report = measured
                pair_accepted += 1
                improved = True
        if not improved:
            stop_reason = "BOUNDED_POOL_NO_IMPROVEMENT"
            break
    else:
        stop_reason = "ROUND_BUDGET_EXHAUSTED"
    return {"rounds": round_index + 1, "candidates_inspected": inspected,
            "replacements": accepted, "segments_removed": removed,
            "pair_combinations_inspected": pair_inspected,
            "pair_replacements": pair_accepted,
            "extra_segments_selected": sum(max(
                0, item.segment_count - (item.pre_seam_segment_count or item.segment_count))
                for item in results),
            "stop_reason": stop_reason, "elapsed_s": time.perf_counter() - started}


def _coordinate_local_seam_tasks(scene: Scene, analysis: Any,
                                 results: list[RegionSwathResult],
                                 validator: Validator,
                                 settings: SwathSettings) -> dict[str, Any]:
    """仅对造成超额接缝重叠的任务尝试有限物理编辑，冻结区域与主体覆盖义务保持不变。
    
    Try bounded physical edits to the tasks causing excess seam overlap.
    
    The region geometries and required body target stay frozen. Every edited
    line is a new implement-on task whose full vehicle envelope and actual
    implement sweep are checked. No mid-line implement switching is assumed."""
    started = time.perf_counter()
    tolerance = scene.settings.coverage_tolerance_m2
    width = scene.vehicle.working_width_m
    reach = max(abs(scene.vehicle.implement_offset_m)
                + scene.vehicle.implement_length_m / 2,
                scene.vehicle.front_m, scene.vehicle.rear_m)
    required = unary_union([item.required_main_area for item in results
                            if item.status == "SWATHS_COMPLETE"
                            and not item.required_main_area.is_empty])
    inspected = accepted = 0
    by_id = {item.region_id: item for item in results}
    operation_counts: Counter[str] = Counter()
    stop_reason = "NO_EXCESS_SEAM"
    initial_report = audit_seams(scene, analysis, results)
    initial_excess = [row for row in initial_report["pairs"]
                      if row["status"] == "EXCESS_OVERLAP"]
    initial_task_pairs = overlapping_work_tasks(
        scene, results, initial_excess, minimum_area_m2=tolerance)
    # 只有多条真实任务重叠的严重接缝才追加局部编辑预算，短孤立接缝保持既有预算。
    round_budget = min(settings.max_local_seam_rounds,
                       max(4, len(initial_task_pairs))) if initial_excess else 0
    rounds_executed = 0

    def new_segment(source: dict[str, Any], geometry: LineString,
                    serial: int) -> dict[str, Any] | None:
        if geometry.geom_type != "LineString" or geometry.length < settings.min_work_segment_m:
            return None
        row = {**source, "task_id": f"{source['task_id']}_seam{serial:04d}",
               "geometry": geometry, "length_m": float(geometry.length)}
        first, last = list(geometry.coords)[0], list(geometry.coords)[-1]
        heading = math.radians(float(source["heading_deg"]))
        reverse = ((last[0] - first[0]) * math.cos(heading)
                   + (last[1] - first[1]) * math.sin(heading)) < 0
        try:
            _, motion = _task_motion(row, scene, reverse=reverse)
            if validator.motion_issues(motion):
                return None
            row["heading_deg"] = math.degrees(float(motion.points[0, 2])) % 360
            row["work_sweep"] = validator.work_sweep(motion)
        except (ValueError, TypeError):
            return None
        return row

    def replacement(original: RegionSwathResult,
                    remove: set[str], add: list[dict[str, Any]],
                    note: str) -> RegionSwathResult:
        trial = copy.copy(original)
        trial.segments = [{**row} for row in original.segments
                          if row["task_id"] not in remove] + add
        trial.segments.sort(key=lambda row: (row["suggested_order"], row["task_id"]))
        for position, row in enumerate(trial.segments, 1):
            row["suggested_order"] = position
        trial.segment_count = len(trial.segments)
        trial.work_sweeps = (unary_union([row["work_sweep"] for row in trial.segments])
                             if trial.segments else GeometryCollection())
        trial.turn_status = "LOCAL_TURN_NOT_VALIDATED"
        trial.turn_checks = []
        trial.search_note += f"; seam_local_task_{note}"
        return trial

    def pair_overlaps(report: dict[str, Any]) -> dict[frozenset[str], float]:
        return {frozenset((row["from_region"], row["to_region"])):
                row["overlap_area_m2"] for row in report["pairs"]}

    for round_index in range(round_budget):
        rounds_executed += 1
        report = audit_seams(scene, analysis, results)
        excess = sorted((row for row in report["pairs"]
                         if row["status"] == "EXCESS_OVERLAP"),
                        key=lambda row: (-(row["overlap_area_m2"] - row["budget_m2"]),
                                         row["from_region"], row["to_region"]))
        if not excess:
            stop_reason = "NO_EXCESS_SEAM"
            break
        task_rows = overlapping_work_tasks(
            scene, results, excess, minimum_area_m2=tolerance)
        grouped: dict[frozenset[str], list[dict[str, Any]]] = {}
        for row in task_rows:
            grouped.setdefault(frozenset((row["from_region"], row["to_region"])),
                               []).append(row)
        # 先轮到每个严重接缝，再给首条长边界上的相似条带继续分配预算，避免其独占搜索。
        selected: list[dict[str, Any]] = []
        for pair in excess:
            key = frozenset((pair["from_region"], pair["to_region"]))
            selected.extend(grouped.get(key, ())[:2])
            if len(selected) >= settings.max_local_seam_task_pairs:
                break
        selected = selected[:settings.max_local_seam_task_pairs]
        if not selected:
            stop_reason = "NO_OVERLAPPING_TASK_PAIR"
            break
        old_pairs = pair_overlaps(report)
        best: tuple[tuple[float, int, str], dict[str, RegionSwathResult],
                    dict[str, Any], str] | None = None
        round_trials = 0
        serial = 0

        def consider(changes: dict[str, RegionSwathResult], kind: str,
                     focus_overlap: float) -> None:
            nonlocal best, inspected, round_trials
            if round_trials >= settings.max_local_seam_trials:
                return
            round_trials += 1
            inspected += 1
            trial_results = [changes.get(item.region_id, item) for item in results]
            if any(item.segment_count < 1 for item in trial_results):
                return
            swept = unary_union([item.work_sweeps for item in trial_results
                                 if item.status == "SWATHS_COMPLETE"])
            if required.difference(swept).area > tolerance:
                return
            measured = audit_seams(scene, analysis, trial_results)
            gain = report["extra_area_m2"] - measured["extra_area_m2"]
            if gain <= max(1.0, 0.02 * focus_overlap):
                return
            new_pairs = pair_overlaps(measured)
            if any(new > old_pairs.get(key, 0.0) + tolerance
                   for key, new in new_pairs.items()):
                return
            old_count = sum(item.segment_count for item in results)
            new_count = sum(item.segment_count for item in trial_results)
            if new_count > old_count + settings.max_seam_extra_segments_per_region:
                return
            score = (measured["extra_area_m2"], new_count, kind)
            if best is None or score < best[0]:
                best = (score, changes, measured, kind)

        for task_pair in selected:
            if round_trials >= settings.max_local_seam_trials:
                break
            left_id, right_id = task_pair["from_region"], task_pair["to_region"]
            left_result, right_result = by_id[left_id], by_id[right_id]
            left = next((row for row in left_result.segments
                         if row["task_id"] == task_pair["from_task"]), None)
            right = next((row for row in right_result.segments
                          if row["task_id"] == task_pair["to_task"]), None)
            if left is None or right is None:
                continue
            focus_overlap = task_pair["overlap_area_m2"]
            for owner, source, other in ((left_result, left, right),
                                         (right_result, right, left)):
                if owner.turn_status == "PASS":
                    continue
                line = source["geometry"]
                # 端部裁短生成新的完整起终任务，不能在同一条连续线内部暗设机具开关。
                max_trim = min(line.length - settings.min_work_segment_m,
                               2 * reach + 2 * width)
                trim_values = sorted({round(value, 6) for value in
                                      (width / 2, width, reach, reach + width,
                                       2 * reach + width)
                                      if 0 < value <= max_trim})
                for distance in trim_values:
                    for start in (True, False):
                        if round_trials >= settings.max_local_seam_trials:
                            break
                        geometry = substring(
                            line, distance if start else 0,
                            line.length if start else line.length - distance)
                        serial += 1
                        candidate = new_segment(source, geometry, serial)
                        if candidate is None:
                            continue
                        consider({owner.region_id: replacement(
                            owner, {source["task_id"]}, [candidate],
                            "trim_start" if start else "trim_end")},
                            "TRIM", focus_overlap)
                # 局部相位移动只调整冲突边行；其余条带仍需覆盖原主体，且新车辆包络必须安全。
                angle = math.radians(float(source["angle_deg"]))
                normal = (-math.sin(angle), math.cos(angle))
                first_mid = line.interpolate(0.5, normalized=True)
                second_mid = other["geometry"].interpolate(0.5, normalized=True)
                separation = ((second_mid.x - first_mid.x) * normal[0]
                              + (second_mid.y - first_mid.y) * normal[1])
                desired = width * (1.0 - scene.settings.overlap_fraction)
                shift = desired - abs(separation)
                if 0.05 < shift < width and abs(separation) > 0.05:
                    away = -math.copysign(shift, separation)
                    geometry = affinity.translate(
                        line, xoff=normal[0] * away, yoff=normal[1] * away)
                    serial += 1
                    candidate = new_segment(source, geometry, serial)
                    if candidate is not None:
                        consider({owner.region_id: replacement(
                            owner, {source["task_id"]}, [candidate], "phase_shift")},
                            "SHIFT", focus_overlap)

                # 中段已由别的任务覆盖时，把两个有用端部变成独立任务；多一段，但不存在未声明的中途机具切换。
                vertices = [point for shape in polygons(task_pair["geometry"])
                            for point in shape.exterior.coords]
                if vertices and line.length > 2 * settings.min_work_segment_m:
                    origin = list(line.coords)[0]
                    end = list(line.coords)[-1]
                    direction = ((end[0] - origin[0]) / line.length,
                                 (end[1] - origin[1]) / line.length)
                    projected = [((x - origin[0]) * direction[0]
                                  + (y - origin[1]) * direction[1])
                                 for x, y in vertices]
                    low = max(0.0, min(projected))
                    high = min(line.length, max(projected))
                    for fraction in (0.0, 0.2, 0.35):
                        if round_trials >= settings.max_local_seam_trials:
                            break
                        start_end = low + fraction * (high - low)
                        end_start = high - fraction * (high - low)
                        if (start_end < settings.min_work_segment_m
                                or line.length - end_start < settings.min_work_segment_m
                                or end_start - start_end < 0.25 * width):
                            continue
                        serial += 1
                        first_piece = new_segment(
                            source, substring(line, 0, start_end), serial)
                        serial += 1
                        second_piece = new_segment(
                            source, substring(line, end_start, line.length), serial)
                        if first_piece is None or second_piece is None:
                            continue
                        consider({owner.region_id: replacement(
                            owner, {source["task_id"]},
                            [first_piece, second_piece], "gap_split")},
                            "SPLIT", focus_overlap)

            # 单侧移动可能产生窄漏作；双侧调整必须作为整体对原始目标检查，不能各自局部自证覆盖。
            if (left_result.turn_status != "PASS"
                    and right_result.turn_status != "PASS"
                    and round_trials < settings.max_local_seam_trials):
                delta = abs((left["angle_deg"] - right["angle_deg"]
                             + 90) % 180 - 90)
                if delta <= 3.0:
                    axis = math.radians(float(left["angle_deg"]))
                    normal = (-math.sin(axis), math.cos(axis))
                    left_mid = left["geometry"].interpolate(0.5, normalized=True)
                    right_mid = right["geometry"].interpolate(0.5, normalized=True)
                    separation = ((right_mid.x - left_mid.x) * normal[0]
                                  + (right_mid.y - left_mid.y) * normal[1])
                    desired = width * (1.0 - scene.settings.overlap_fraction)
                    needed = desired - abs(separation)
                    if 0.05 < needed < width and abs(separation) > 0.05:
                        sign = math.copysign(1.0, separation)
                        for share in (0.25, 0.5, 0.75):
                            if round_trials >= settings.max_local_seam_trials:
                                break
                            left_line = affinity.translate(
                                left["geometry"],
                                xoff=-sign * needed * share * normal[0],
                                yoff=-sign * needed * share * normal[1])
                            right_line = affinity.translate(
                                right["geometry"],
                                xoff=sign * needed * (1.0 - share) * normal[0],
                                yoff=sign * needed * (1.0 - share) * normal[1])
                            serial += 1
                            left_new = new_segment(left, left_line, serial)
                            serial += 1
                            right_new = new_segment(right, right_line, serial)
                            if left_new is None or right_new is None:
                                continue
                            consider({
                                left_id: replacement(left_result, {left["task_id"]},
                                                     [left_new], "joint_shift"),
                                right_id: replacement(right_result, {right["task_id"]},
                                                      [right_new], "joint_shift"),
                            }, "JOINT_SHIFT", focus_overlap)

            if (left_result.turn_status == "PASS"
                    or right_result.turn_status == "PASS"
                    or round_trials >= settings.max_local_seam_trials):
                continue
            # 共线或近平行任务可尝试一个真实直线替代；若跨越受阻点接触，完整车辆包络检查会拒绝。
            angle = math.radians(float(left["angle_deg"]))
            u = (math.cos(angle), math.sin(angle))
            n = (-u[1], u[0])
            angle_delta = abs((left["angle_deg"] - right["angle_deg"]
                               + 90) % 180 - 90)
            if angle_delta > 3.0:
                continue
            left_mid = left["geometry"].interpolate(0.5, normalized=True)
            right_mid = right["geometry"].interpolate(0.5, normalized=True)
            lateral = ((right_mid.x - left_mid.x) * n[0]
                       + (right_mid.y - left_mid.y) * n[1])
            if abs(lateral) >= width:
                continue
            all_points = [point for row in (left, right)
                          for point in (list(row["geometry"].coords)[0],
                                        list(row["geometry"].coords)[-1])]
            low = min(x * u[0] + y * u[1] for x, y in all_points)
            high = max(x * u[0] + y * u[1] for x, y in all_points)
            if high - low < settings.min_work_segment_m:
                continue
            for provider, source, normal_offset in (
                    (left_result, left, 0.0),
                    (right_result, right, lateral),
                    (left_result, left, lateral / 2)):
                if round_trials >= settings.max_local_seam_trials:
                    break
                base = left_mid.x * n[0] + left_mid.y * n[1] + normal_offset
                geometry = LineString([
                    (low * u[0] + base * n[0], low * u[1] + base * n[1]),
                    (high * u[0] + base * n[0], high * u[1] + base * n[1]),
                ])
                serial += 1
                candidate = new_segment(source, geometry, serial)
                if candidate is None:
                    continue
                candidate["task_id"] = (
                    f"{source['task_id']}_joint_{serial:04d}")
                changes = {
                    left_id: replacement(left_result, {left["task_id"]},
                                         [candidate] if provider is left_result else [],
                                         "joint_provider" if provider is left_result
                                         else "joint_assisted"),
                    right_id: replacement(right_result, {right["task_id"]},
                                          [candidate] if provider is right_result else [],
                                          "joint_provider" if provider is right_result
                                          else "joint_assisted"),
                }
                consider(changes, "JOINT", focus_overlap)
        if best is None:
            stop_reason = ("SEARCH_LIMITED" if round_trials >= settings.max_local_seam_trials
                           else "BOUNDED_TASK_OPTIONS_NO_IMPROVEMENT")
            break
        _, changes, _, kind = best
        for index, item in enumerate(results):
            if item.region_id in changes:
                results[index] = changes[item.region_id]
                by_id[item.region_id] = results[index]
        accepted += 1
        operation_counts[kind] += 1
    else:
        stop_reason = ("NO_EXCESS_SEAM" if round_budget == 0 or
                       audit_seams(scene, analysis, results)["status"] == "PASS"
                       else "ROUND_BUDGET_EXHAUSTED")
    return {"candidates_inspected": inspected, "replacements": accepted,
            "operation_counts": dict(operation_counts),
            "stop_reason": stop_reason, "round_budget": round_budget,
            "rounds_executed": rounds_executed,
            "elapsed_s": time.perf_counter() - started}


def _validate_extra_seam_candidates(scene: Scene, analysis: Any,
                                    results: list[RegionSwathResult],
                                    raw: dict[str, list[Candidate]],
                                    pools: dict[str, list[Candidate]],
                                    settings: SwathSettings, validator: Validator,
                                    backend: Any) -> dict[str, Any]:
    """只在超额重叠接缝旁检查少量额外条带候选，防止扩大全田搜索成本。
    
    Inspect a tiny extra-strip shortlist only beside excess-overlap seams."""
    started = time.perf_counter()
    report = audit_seams(scene, analysis, results)
    affected = {name for pair in report["pairs"] if pair["status"] == "EXCESS_OVERLAP"
                for name in (pair["from_region"], pair["to_region"])}
    by_id = {item.region_id: item for item in results}
    region_index = {item.region_id: index for index, item in enumerate(analysis.regions)}
    inspected = accepted = 0
    for region_id in sorted(affected):
        current = by_id[region_id]
        if current.status != "SWATHS_COMPLETE":
            continue
        turn_cache: dict[tuple[str, str], tuple[Any, ...]] = {}
        for candidate in raw.get(region_id, ())[:settings.max_seam_extra_validation_per_region]:
            inspected += 1
            if current.turn_status == "PASS":
                variants = _candidate_orders(
                    candidate, scene, validator, backend, region_index[region_id],
                    settings, turn_cache,
                )
                variants = [item for item in variants if item.turn_status == "PASS"]
            else:
                variants = _candidate_body_orders(
                    candidate, scene, validator,
                    fallback_reason="EXTRA_STRIP_SEAM_TURN_NOT_CERTIFIED",
                )
            for item in variants:
                if (item.geometry_status == "PASS" and item.coverage_tolerance_passed
                        and item.area_m2 + 1e-8 >= (current.amax_area_m2 or 0.0)
                        * settings.area_retention_fraction):
                    pools.setdefault(region_id, []).append(item)
                    accepted += 1
                    break
    return {"affected_region_count": len(affected), "inspected": inspected,
            "accepted": accepted, "elapsed_s": time.perf_counter() - started}


def _remove_short_work_segments(scene: Scene,
                                results: list[RegionSwathResult],
                                minimum_m: float,
                                review_m: float = 5.0) -> dict[str, Any]:
    """删除真正冗余的短完整任务；覆盖必要的短条带保留并报告，不能按长度直接消掉目标。
    
    Remove redundant short whole tasks; retain necessary ones for review."""
    required = unary_union([item.required_main_area for item in results
                            if item.status == "SWATHS_COMPLETE"
                            and not item.required_main_area.is_empty])
    tolerance = scene.settings.coverage_tolerance_m2
    removed: list[str] = []
    unresolved: list[str] = []
    review_rows: list[dict[str, Any]] = []
    for item in results:
        for segment in sorted(list(item.segments),
                              key=lambda row: (row["length_m"], row["task_id"])):
            if segment["length_m"] >= review_m:
                continue
            if item.turn_status == "PASS":
                if segment["length_m"] < minimum_m:
                    unresolved.append(segment["task_id"] + ":CERTIFIED_TURN_SEQUENCE")
                review_rows.append({"region_id": item.region_id,
                                    "task_id": segment["task_id"],
                                    "length_m": segment["length_m"],
                                    "status": "RETAINED_CERTIFIED_TURN_SEQUENCE",
                                    "unique_required_area_m2": None,
                                    "geometry": segment["geometry"]})
                continue
            remaining = [row for row in item.segments
                         if row["task_id"] != segment["task_id"]]
            candidate_sweeps = unary_union([row["work_sweep"] for row in remaining]) \
                if remaining else GeometryCollection()
            all_sweeps = unary_union([
                candidate_sweeps if other is item else other.work_sweeps
                for other in results if other.status == "SWATHS_COMPLETE"
            ])
            unique_required = float(required.difference(all_sweeps).area)
            if unique_required > tolerance:
                if segment["length_m"] < minimum_m:
                    unresolved.append(segment["task_id"] + ":REQUIRED_BODY_GAP")
                review_rows.append({"region_id": item.region_id,
                                    "task_id": segment["task_id"],
                                    "length_m": segment["length_m"],
                                    "status": "RETAINED_REQUIRED_COVERAGE_LAG_PENDING",
                                    "unique_required_area_m2": unique_required,
                                    "geometry": segment["geometry"]})
                continue
            item.segments = remaining
            item.work_sweeps = candidate_sweeps
            item.segment_count = len(remaining)
            item.turn_checks = []
            item.turn_status = "LOCAL_TURN_NOT_VALIDATED"
            for index, row in enumerate(sorted(
                    remaining, key=lambda value: value["suggested_order"]), 1):
                row["suggested_order"] = index
            item.search_note += "; redundant_short_whole_task_removed"
            removed.append(segment["task_id"])
            review_rows.append({"region_id": item.region_id,
                                "task_id": segment["task_id"],
                                "length_m": segment["length_m"],
                                "status": ("REMOVED_SUB_1M" if segment["length_m"] < minimum_m
                                           else "REMOVED_REDUNDANT_SHORT_WORK"),
                                "unique_required_area_m2": unique_required,
                                "geometry": segment["geometry"]})
    return {"status": "PASS" if not unresolved else "SHORT_WORK_UNRESOLVED",
            "removed_task_ids": removed, "unresolved_task_ids": unresolved,
            "minimum_m": minimum_m, "review_m": review_m,
            "review_rows": review_rows}


def _joint_align_regions(scene: Scene, analysis: Any,
                         region_results: list[RegionSwathResult],
                         settings: SwathSettings, validator: Validator,
                         backend: Any, spacing: float) -> dict[str, Any]:
    """尝试相邻冻结分区的少量联合方向/相位；只有共同满足面积、覆盖和安全要求才替换。
    
    Try a small joint direction/phase shortlist on adjacent frozen regions.
    
    A candidate can replace both regional results only when it does not increase
    either region's segment count, remains above each region's original area
    retention threshold, passes the straight-body checks, and creates at least
    one vehicle-envelope-certified continuous straight run across the seam."""
    started = time.perf_counter()
    by_id = {result.region_id: result for result in region_results}
    geometry_by_id = {region.region_id: region for region in analysis.regions}
    index_by_id = {region.region_id: index for index, region in enumerate(analysis.regions)}
    status_by_connection: dict[str, str] = {}
    attempts_by_connection: dict[str, int] = {}
    candidates_by_connection: dict[str, int] = {}
    unattempted_by_connection: dict[str, int] = {}
    failures_by_connection: dict[str, dict[str, int]] = {}
    line_cache: dict[tuple[float, float], list[tuple[int, LineString]]] = {}
    turn_caches: dict[str, dict[tuple[str, str], tuple[Any, ...]]] = {}
    attempts = 0
    aligned_connections = 0
    aligned_runs = 0
    f2c_calls = 0
    reserved: dict[str, tuple[float, float]] = {}
    connection_rows = sorted(
        analysis.connections,
        key=lambda item: (
            0 if (item.width_class == "WIDE"
                  and item.turn_status == "SUFFICIENT_SPACE") else 1,
            0 if item.turn_status == "SUFFICIENT_SPACE" else 1,
            0 if item.width_class == "WIDE" else 1,
            -(item.effective_width_m or 0.0), item.connection_id,
        ),
    )
    for connection in connection_rows:
        connection_id = connection.connection_id
        attempts_by_connection[connection_id] = 0
        candidates_by_connection[connection_id] = 0
        unattempted_by_connection[connection_id] = 0
        failure_reasons: Counter[str] = Counter()
        failures_by_connection[connection_id] = failure_reasons
        connection_started = time.perf_counter()
        if (connection.width_class == "POINT_CONTACT"
                or connection.straight_status == "BLOCKED"):
            status_by_connection[connection.connection_id] = "NOT_INSPECTED_TOPOLOGY_BLOCKED"
            continue
        first = by_id.get(connection.from_region)
        second = by_id.get(connection.to_region)
        if not first or not second:
            status_by_connection[connection.connection_id] = "REGION_SWATH_FAILED"
            continue
        if first.status != "SWATHS_COMPLETE" or second.status != "SWATHS_COMPLETE":
            status_by_connection[connection.connection_id] = "REGION_SWATH_FAILED"
            continue
        angle_delta = abs((float(first.angle_deg) - float(second.angle_deg) + 90.0) % 180.0 - 90.0)
        phase_delta = abs(float(first.phase_m) - float(second.phase_m)) % spacing
        phase_delta = min(phase_delta, spacing - phase_delta)
        if angle_delta <= 1e-6 and phase_delta <= 1e-6:
            status_by_connection[connection.connection_id] = "ALREADY_ALIGNED"
            continue
        first_geometry, second_geometry = (geometry_by_id[first.region_id],
                                           geometry_by_id[second.region_id])
        first_angles = _stable_angles(scene, first_geometry, settings)
        second_angles = _stable_angles(scene, second_geometry, settings)
        # 任一相邻分区提出的方向都可在两侧联合测试，不能只取两个短名单的交集而漏检宽连接。
        joint_angles = list(dict.fromkeys(
            round(_angle_mod_180(angle), 7)
            for angle in (*first_angles, *second_angles,
                          float(first.angle_deg), float(second.angle_deg))
        ))
        joint_angles.sort(key=lambda angle: (
            0 if (any(abs((angle - a + 90.0) % 180.0 - 90.0) <= 1e-6
                      for a in first_angles)
                  and any(abs((angle - b + 90.0) % 180.0 - 90.0) <= 1e-6
                          for b in second_angles)) else 1,
            abs((angle - float(first.angle_deg) + 90.0) % 180.0 - 90.0)
            + abs((angle - float(second.angle_deg) + 90.0) % 180.0 - 90.0), angle,
        ))
        phase_lists: list[tuple[float, list[float]]] = []
        for angle in joint_angles:
            first_phases = _phase_values(
                scene, analysis, angle, spacing, settings.phase_steps,
                first_geometry.geometry, settings.max_event_phases,
            )
            second_phases = _phase_values(
                scene, analysis, angle, spacing, settings.phase_steps,
                second_geometry.geometry, settings.max_event_phases,
            )
            phase_map = {round(value, 8): float(value)
                         for value in first_phases}
            common_phases = [phase_map[round(value, 8)] for value in second_phases
                             if round(value, 8) in phase_map]
            common_phases.extend((float(first.phase_m) % spacing,
                                  float(second.phase_m) % spacing))
            common_phases = list(dict.fromkeys(round(value % spacing, 8)
                                                for value in common_phases))
            common_phases.sort(key=lambda phase: (
                abs((phase - float(first.phase_m) + spacing / 2) % spacing - spacing / 2)
                + abs((phase - float(second.phase_m) + spacing / 2) % spacing - spacing / 2),
                phase,
            ))
            phase_lists.append((angle, common_phases))
        # 按排序方向轮流检查；连接预算决定检查数，不能用隐藏的2×2截断漏掉其它候选。
        pairs = [
            (angle, phases[phase_index])
            for phase_index in range(max((len(phases) for _, phases in phase_lists),
                                         default=0))
            for angle, phases in phase_lists
            if phase_index < len(phases)
        ]
        candidates_by_connection[connection_id] = len(pairs)
        best = None
        for angle, phase in pairs[:settings.max_joint_alignment_candidates]:
            if (first.region_id in reserved
                    and (abs(reserved[first.region_id][0] - angle) > 1e-6
                         or min(abs(reserved[first.region_id][1] - phase) % spacing,
                                spacing - abs(reserved[first.region_id][1] - phase) % spacing) > 1e-6)):
                failure_reasons["RESERVED_ALIGNMENT_CONFLICT"] += 1
                continue
            if (second.region_id in reserved
                    and (abs(reserved[second.region_id][0] - angle) > 1e-6
                         or min(abs(reserved[second.region_id][1] - phase) % spacing,
                                spacing - abs(reserved[second.region_id][1] - phase) % spacing) > 1e-6)):
                failure_reasons["RESERVED_ALIGNMENT_CONFLICT"] += 1
                continue
            attempts += 1
            attempts_by_connection[connection_id] += 1
            key = (round(angle, 7), round(phase, 8))
            if key not in line_cache:
                counter = [0]
                line_cache[key] = _native_parallel_lines(
                    analysis.target, angle, phase, spacing, counter,
                )
                f2c_calls += counter[0]
                first.f2c_call_count += counter[0]
            candidates: list[Candidate] = []
            feasible_pair = True
            for result, region in ((first, first_geometry), (second, second_geometry)):
                original_index = index_by_id[result.region_id]
                if not line_cache[key]:
                    failure_reasons["NO_FIELD_LINES"] += 1
                    feasible_pair = False
                    break
                try:
                    candidate = _evaluate_candidate(
                        scene, region, original_index, analysis, validator,
                        angle, phase, float(result.headland_m),
                        float(result.side_clearance_m), spacing, result.region_id,
                        field_lines=line_cache[key],
                    )
                except Exception:
                    failure_reasons["CANDIDATE_GEOMETRY_ERROR"] += 1
                    feasible_pair = False
                    break
                if (candidate.segment_count > result.segment_count
                        or (result.amax_area_m2 is not None
                            and candidate.area_m2 + 1e-8
                            < result.amax_area_m2 * settings.area_retention_fraction)):
                    failure_reasons["STRIP_COUNT_OR_AREA_LIMIT"] += 1
                    feasible_pair = False
                    break
                cache = turn_caches.setdefault(result.region_id, {})
                modes = (result.order_mode or "serpentine",)
                variants = _candidate_orders(
                    candidate, scene, validator, backend, original_index,
                    settings, cache, modes,
                )
                variants = [item for item in variants
                            if item.coverage_tolerance_passed
                            and item.turn_status in {
                                "PASS", "LOCAL_TURN_NOT_VALIDATED", "TURN_CHECK_LIMIT_REACHED",
                            }]
                if result.turn_status == "PASS":
                    variants = [item for item in variants if item.turn_status == "PASS"]
                if not variants:
                    failure_reasons["BODY_OR_TURN_NOT_CERTIFIED"] += 1
                    feasible_pair = False
                    break
                candidates.append(min(
                    variants, key=lambda item: (
                        0 if item.turn_status == "PASS" else 1,
                        -item.area_m2, item.segment_count,
                    ),
                ))
            if not feasible_pair or len(candidates) != 2:
                continue
            shared = first_geometry.geometry.boundary.intersection(
                second_geometry.geometry.boundary
            )
            joins = _joint_run_candidates(
                scene, shared, candidates[0], candidates[1],
                candidates[0].order_mode, candidates[1].order_mode, validator,
            )
            if not joins:
                failure_reasons["NO_ENVELOPE_CERTIFIED_CONTINUOUS_RUN"] += 1
                continue
            score = (
                -len(joins), candidates[0].segment_count + candidates[1].segment_count,
                -(candidates[0].area_m2 + candidates[1].area_m2),
                0 if all(item.turn_status == "PASS" for item in candidates) else 1,
                angle, phase,
            )
            if best is None or score < best[0]:
                best = (score, angle, phase, candidates, len(joins))
        unattempted_by_connection[connection_id] = (
            len(pairs) - attempts_by_connection[connection_id]
        )
        if best is None:
            status_by_connection[connection.connection_id] = (
                "CHECKED_NO_SAFE_JOINT_CANDIDATE"
                if attempts_by_connection[connection.connection_id]
                else "NOT_INSPECTED_RESERVED_CONFLICT" if pairs
                else "NOT_INSPECTED_NO_DIRECTION_CANDIDATE"
            )
            continue
        _, angle, phase, candidates, join_count = best
        _apply_joint_candidate(scene, first_geometry, first, candidates[0], validator)
        _apply_joint_candidate(scene, second_geometry, second, candidates[1], validator)
        joint_elapsed = time.perf_counter() - connection_started
        first.profile["joint_alignment_s"] = first.profile.get("joint_alignment_s", 0.0) + joint_elapsed
        second.profile["joint_alignment_s"] = second.profile.get("joint_alignment_s", 0.0) + joint_elapsed
        reserved[first.region_id] = (angle, phase)
        reserved[second.region_id] = (angle, phase)
        status_by_connection[connection.connection_id] = "JOINT_CANDIDATE_SELECTED"
        aligned_connections += 1
        aligned_runs += join_count
    return {
        "connection_status": status_by_connection,
        "connection_attempts": attempts_by_connection,
        "connection_candidate_count": candidates_by_connection,
        "connection_unattempted_count": unattempted_by_connection,
        "connection_failure_reasons": failures_by_connection,
        "attempt_count": attempts,
        "budget_per_connection": settings.max_joint_alignment_candidates,
        "budget": settings.max_joint_alignment_candidates,
        "uninspected_connection_count": sum(
            status.startswith("NOT_INSPECTED")
            for status in status_by_connection.values()
        ),
        "inspected_connection_count": sum(
            status in {"ALREADY_ALIGNED", "CHECKED_NO_SAFE_JOINT_CANDIDATE",
                       "JOINT_CANDIDATE_SELECTED"}
            for status in status_by_connection.values()
        ),
        "aligned_connection_count": aligned_connections,
        "candidate_continuous_run_count": aligned_runs,
        "f2c_call_count": f2c_calls,
        "elapsed_s": time.perf_counter() - started,
    }


def _continuous_runs(scene: Scene, analysis: Any,
                     region_results: list[RegionSwathResult],
                     validator: Validator,
                     joint_status: dict[str, str] | None = None,
                     joint_attempts: dict[str, int] | None = None,
                     joint_candidates: dict[str, int] | None = None,
                     joint_unattempted: dict[str, int] | None = None,
                     joint_failures: dict[str, dict[str, int]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """只认证同网格且真实扫掠可跨接缝衔接的直行段，不把所有邻接条带都当连续行程。
    
    Certify only same-grid swaths that actually join across a frozen seam."""
    by_id = {item.region_id: item for item in region_results}
    regions = {item.region_id: item for item in analysis.regions}
    runs: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    connections = _connection_records(analysis, scene.name)
    joint_status = joint_status or {}
    joint_attempts = joint_attempts or {}
    joint_candidates = joint_candidates or {}
    joint_unattempted = joint_unattempted or {}
    joint_failures = joint_failures or {}
    spacing = scene.vehicle.working_width_m * (1.0 - scene.settings.overlap_fraction)
    for connection in connections:
        connection["joint_alignment_status"] = joint_status.get(
            connection["connection_id"], "NOT_ATTEMPTED",
        )
        connection["joint_alignment_attempt_count"] = joint_attempts.get(
            connection["connection_id"], 0,
        )
        connection["joint_alignment_candidate_count"] = joint_candidates.get(
            connection["connection_id"], 0,
        )
        connection["joint_alignment_unattempted_count"] = joint_unattempted.get(
            connection["connection_id"], 0,
        )
        connection["joint_alignment_failure_reasons"] = json.dumps(
            joint_failures.get(connection["connection_id"], {}),
            ensure_ascii=False, sort_keys=True,
        )
        first = by_id.get(connection["from_region"])
        second = by_id.get(connection["to_region"])
        status = "REGION_SWATH_FAILED"
        made = 0
        if first and second and first.status == second.status == "SWATHS_COMPLETE":
            angle_delta = abs((float(first.angle_deg) - float(second.angle_deg) + 90.0) % 180.0 - 90.0)
            phase_delta = abs(float(first.phase_m) - float(second.phase_m)) % spacing
            phase_delta = min(phase_delta, spacing - phase_delta)
            if angle_delta <= 1e-6 and phase_delta <= 1e-6:
                status = "ALIGNED_NO_VALIDATED_RUN"
                shared = regions[first.region_id].geometry.boundary.intersection(
                    regions[second.region_id].geometry.boundary
                )
                for first_segment in first.segments:
                    for second_segment in second.segments:
                        if first_segment["row_index"] != second_segment["row_index"]:
                            continue
                        a, b = first_segment["geometry"], second_segment["geometry"]
                        if a.distance(b) > max(1e-5, scene.settings.geometry_epsilon_m * 10):
                            continue
                        merged = unary_union([a, b])
                        if merged.geom_type != "LineString":
                            try:
                                merged = linemerge(merged)
                            except ValueError:
                                continue
                        if merged.geom_type != "LineString" or merged.length <= max(a.length, b.length) + 1e-5:
                            continue
                        if not shared.is_empty and merged.intersection(shared).is_empty:
                            continue
                        coords = list(merged.coords)
                        for reverse in (False, True):
                            start, end = (coords[-1], coords[0]) if reverse else (coords[0], coords[-1])
                            yaw = math.atan2(end[1] - start[1], end[0] - start[0])
                            length = math.dist(start[:2], end[:2])
                            step = min(scene.settings.sampling_step_m, length / 3.0)
                            t = np.asarray([0.0, step / length, 1.0 - step / length, 1.0])
                            points = np.column_stack((
                                start[0] + t * (end[0] - start[0]),
                                start[1] + t * (end[1] - start[1]),
                                np.full(4, yaw), np.ones(4),
                            ))
                            motion = Motion(points, "work", "continuous_run", True)
                            if validator.motion_issues(motion):
                                continue
                            swept = validator.work_sweep(motion)
                            required = unary_union([first_segment["work_sweep"],
                                                    second_segment["work_sweep"]])
                            if required.difference(swept).area > scene.settings.coverage_tolerance_m2:
                                continue
                            run_id = f"{connection['connection_id']}_row{first_segment['row_index']}"
                            runs.append({
                                "field_id": scene.name, "run_id": run_id,
                                "connection_id": connection["connection_id"],
                                "from_region": first.region_id, "to_region": second.region_id,
                                "angle_deg": first.angle_deg, "phase_m": first.phase_m,
                                "length_m": float(merged.length), "geometry": merged,
                            })
                            for region_id, segment in ((first.region_id, first_segment),
                                                       (second.region_id, second_segment)):
                                members.append({"field_id": scene.name, "run_id": run_id,
                                                "region_id": region_id,
                                                "task_id": segment["task_id"],
                                                "row_index": segment["row_index"],
                                                "geometry": Point(start[0], start[1])})
                            made += 1
                            break
                if made:
                    status = "ALIGNED_CONTINUOUS_RUNS"
            else:
                status = "NOT_ALIGNED"
        connection["swath_alignment_status"] = status
        connection["continuous_run_count"] = made
    return connections, runs, members


def generate_region_swaths(scene: Scene, analysis: Any,
                           swath_settings: SwathSettings | None = None) -> SwathStageResult:
    """完整主体条带入口：保持冻结分区，进行候选搜索、统一面积筛选、覆盖、安全、接缝与失败原因汇总。
    
    Generate audited straight work strips for frozen work regions only."""
    started = time.perf_counter()
    settings = swath_settings or SwathSettings()
    if not analysis.regions:
        raise ValueError("冻结分区结果没有作业区域")
    _validate_analysis_source(scene, analysis)
    spacing = scene.vehicle.working_width_m * (1.0 - scene.settings.overlap_fraction)
    if spacing <= 0:
        raise ValueError("条带间距必须为正")
    from planner import F2CBackend  # avoid a planner/swath_planner import cycle
    backend = F2CBackend(scene)
    validator = Validator(scene)
    region_results: list[RegionSwathResult] = []
    seam_candidate_pools: dict[str, list[Candidate]] = {}
    seam_extra_raw_pools: dict[str, list[Candidate]] = {}
    call_count = 0
    for region_index, region in enumerate(analysis.regions):
        region_started = time.perf_counter()
        region_call_counter = [0]
        failures: Counter[str] = Counter()
        candidate_cache: dict[tuple[float, float, float, float], Candidate | None] = {}
        field_line_cache: dict[tuple[float, float], list[tuple[int, LineString]]] = {}
        evaluated_configs: set[tuple[float, float, float]] = set()
        passing: list[Candidate] = []
        body_feasible_candidates: list[Candidate] = []
        failed_candidates: list[Candidate] = []
        unresolved_candidates: list[Candidate] = []
        turn_order_check_count = 0
        phase_refinement_count = 0
        max_phase_refinements = min(4, max(1, settings.max_event_phases // 2))
        profile = {
            "direction_phase_generation_s": 0.0,
            "f2c_line_generation_s": 0.0,
            "candidate_geometry_s": 0.0,
            "turn_validation_s": 0.0,
            "selection_reconciliation_s": 0.0,
        }
        profile_started = time.perf_counter()
        angles = _stable_angles(scene, region, settings)
        phases_by_angle = {angle: _phase_values(scene, analysis, angle, spacing,
                                                 settings.phase_steps, region.geometry,
                                                 settings.max_event_phases)
                           for angle in angles}
        profile["direction_phase_generation_s"] = time.perf_counter() - profile_started
        turn_cache: dict[tuple[str, str], tuple[Any, ...]] = {}
        tested_variants: dict[tuple[float, float, float, float, int, tuple[str, ...]], list[Candidate]] = {}
        max_longitudinal = max(abs(scene.vehicle.implement_offset_m) +
                               scene.vehicle.implement_length_m / 2,
                               scene.vehicle.front_m, scene.vehicle.rear_m)
        base_headland = (scene.vehicle.min_turn_radius_m + max_longitudinal +
                         scene.vehicle.safety_margin_m)
        base_side = max(scene.vehicle.body_width_m,
                        scene.vehicle.working_width_m) / 2.0
        base_side += scene.settings.travel_clearance_m + scene.vehicle.safety_margin_m
        side_step = max(0.75, 0.3 * scene.vehicle.min_turn_radius_m)
        coarse_side_levels = [round(base_side + side_step * step, 4)
                              for step in (0.0, 1.0, 2.0, 3.0)]
        expanded_side_levels = sorted({
            round(base_side + side_step * step, 4)
            for step in range(0, 7)
        })
        headlands = sorted({round(base_headland * factor, 4)
                            for factor in settings.headland_multipliers})

        def get_candidate(angle_deg: float, phase_m: float, headland_m: float,
                          side_reserve_m: float) -> Candidate | None:
            nonlocal call_count
            key = (round(angle_deg, 7), round(phase_m, 8),
                   round(headland_m, 4), round(side_reserve_m, 4))
            if key in candidate_cache:
                return candidate_cache[key]
            line_key = (key[0], key[1])
            if line_key not in field_line_cache:
                f2c_counter = [0]
                line_started = time.perf_counter()
                try:
                    field_line_cache[line_key] = _native_parallel_lines(
                        analysis.target, angle_deg, phase_m, spacing, f2c_counter,
                    )
                except Exception as exc:
                    failures[f"F2C_LINES:{type(exc).__name__}:{exc}"] += 1
                    field_line_cache[line_key] = []
                profile["f2c_line_generation_s"] += time.perf_counter() - line_started
                region_call_counter[0] += f2c_counter[0]
                call_count += f2c_counter[0]
            try:
                if not field_line_cache[line_key]:
                    raise ValueError("F2C_GENERATED_NO_FIELD_LINES")
                geometry_started = time.perf_counter()
                candidate = _evaluate_candidate(
                    scene, region, region_index, analysis, validator,
                    angle_deg, phase_m, headland_m, side_reserve_m,
                    spacing, region.region_id, field_lines=field_line_cache[line_key],
                )
                profile["candidate_geometry_s"] += time.perf_counter() - geometry_started
                candidate_cache[key] = candidate
                return candidate
            except Exception as exc:
                failures[str(exc)] += 1
                candidate_cache[key] = None
                return None

        def config_area(angle_deg: float, headland_m: float,
                        side_reserve_m: float) -> float:
            try:
                return _estimate_main_area(scene, analysis, region, angle_deg,
                                           headland_m, side_reserve_m)
            except Exception as exc:
                failures[f"AREA_ESTIMATE:{type(exc).__name__}:{exc}"] += 1
                return 0.0

        def config_key(angle_deg: float, headland_m: float,
                       side_reserve_m: float) -> tuple[float, float, float]:
            return (round(angle_deg, 7), round(headland_m, 4),
                    round(side_reserve_m, 4))

        def test_feasible(candidate: Candidate) -> list[Candidate]:
            variants = test_turns(candidate, ("serpentine",))
            if any(item.turn_status == "PASS" for item in variants):
                return variants
            return variants + test_turns(candidate, ("skip_row_then_fill",))

        def try_config(angle_deg: float, headland_m: float,
                       side_reserve_m: float) -> tuple[bool, list[Candidate]]:
            """Try a bounded base batch, then refine phases only after a miss."""
            nonlocal phase_refinement_count
            evaluated_configs.add(config_key(angle_deg, headland_m, side_reserve_m))
            candidates = [get_candidate(angle_deg, phase, headland_m, side_reserve_m)
                          for phase in phases_by_angle[angle_deg]]
            available = sorted((item for item in candidates if item is not None),
                               key=lambda item: (item.segment_count, -item.area_m2,
                                                 item.phase_m))
            batch_passes: list[Candidate] = []
            best_count: int | None = None
            tested_phase_count = 0
            for candidate in available:
                if best_count is not None and candidate.segment_count > best_count:
                    break
                variants = test_feasible(candidate)
                tested_phase_count += 1
                if variants:
                    best_count = candidate.segment_count
                    batch_passes.extend(variants)
                    # 按实际独立段数排序候选；本批首个通过相位已有本批最少段数，不代表所有方向的全局最优。
                    break
                # 粗转向检查只保留少量同段数相位，避免一个退让组合耗尽预算，使后续更保守田头无人检查。
                if tested_phase_count >= 2:
                    break
            if phase_refinement_count >= max_phase_refinements:
                return bool(batch_passes), batch_passes
            if turn_order_check_count >= settings.max_turn_order_checks:
                return False, []

            # 漏掉的粗相位批仅做一次有界12步细化，复用原生角度/相位缓存，不启动全方向相位穷举。
            refined_steps = max(12, settings.phase_steps * 3)
            refined_phases = _phase_values(
                scene, analysis, angle_deg, spacing, refined_steps,
                region.geometry, settings.max_event_phases,
            )
            known_phases = {round(value, 8) for value in phases_by_angle[angle_deg]}
            new_phases = [phase for phase in refined_phases
                          if round(phase, 8) not in known_phases]
            if not new_phases:
                return False, []
            phase_refinement_count += 1
            refined = sorted((item for item in (
                get_candidate(angle_deg, phase, headland_m, side_reserve_m)
                for phase in new_phases
            ) if item is not None), key=lambda item: (
                item.segment_count, -item.area_m2, item.phase_m,
            ))
            refined_best = refined[0].segment_count if refined else None
            refined_tests = 0
            for candidate in refined:
                if refined_best is not None and candidate.segment_count > refined_best:
                    break
                variants = test_feasible(candidate)
                refined_tests += 1
                if variants:
                    batch_passes.extend(variants)
                    break
                if refined_tests >= 2:
                    break
            return bool(batch_passes), batch_passes

        def test_turns(candidate: Candidate,
                       modes: tuple[str, ...] = ("serpentine", "skip_row_then_fill")) -> list[Candidate]:
            nonlocal turn_order_check_count
            key = (round(candidate.angle_deg, 7), round(candidate.phase_m, 8),
                   round(candidate.headland_m, 4), round(candidate.side_clearance_m, 4),
                   candidate.segment_count, modes)
            if key in tested_variants:
                return [item for item in tested_variants[key]
                        if item.turn_status == "PASS"]
            if turn_order_check_count >= settings.max_turn_order_checks:
                return []
            turn_order_check_count += 1
            turn_started = time.perf_counter()
            variants = _candidate_orders(candidate, scene, validator, backend,
                                         region_index, settings, turn_cache, modes)
            profile["turn_validation_s"] += time.perf_counter() - turn_started
            tested_variants[key] = variants
            body_feasible_candidates.extend(
                item for item in variants
                if item.coverage_tolerance_passed and item.geometry_status == "PASS"
            )
            passing.extend(item for item in variants if item.turn_status == "PASS")
            unresolved_candidates.extend(
                item for item in variants
                if item.turn_status in {"LOCAL_TURN_NOT_VALIDATED", "TURN_CHECK_LIMIT_REACHED"}
            )
            if not variants and candidate.turn_status not in {"NOT_CHECKED", "PASS"}:
                failed_candidates.append(candidate)
                failures[candidate.turn_status] += 1
            return [item for item in variants if item.turn_status == "PASS"]

        # 自适应搜索共同增加端部和侧向预留，避免预算浪费在重复而无法解决相同包络碰撞的组合。
        feasible_configurations: list[tuple[float, float, float, float]] = []
        side_levels_by_angle: dict[float, list[float]] = {}
        for angle_deg in angles:
            side_levels_by_angle[angle_deg] = list(expanded_side_levels)
            if turn_order_check_count >= settings.max_turn_order_checks:
                break
            current_amax = max((item.area_m2 for item in passing), default=None)
            if current_amax is not None:
                upper = config_area(angle_deg, headlands[0], base_side)
                if upper + 1e-8 < current_amax * settings.area_retention_fraction:
                    continue
            coarse_configs = sorted(
                ((config_area(angle_deg, headland, side), headland, side)
                 for side in coarse_side_levels for headland in headlands),
                key=lambda item: (-item[0], item[1], item[2]),
            )
            # 沿车辆尺度的少量阶梯增长独立余量，避免侧向×端部×相位×排序的无界组合爆炸。
            ladder_steps = (0, 1, 2, 4, 6)
            ladder = [
                (config_area(angle_deg, headland, expanded_side_levels[step]),
                 headland, expanded_side_levels[step])
                for headland, step in zip(headlands, ladder_steps)
                if step < len(expanded_side_levels)
            ]
            found = False
            tested_keys: set[tuple[float, float, float]] = set()
            schedule = [*ladder]
            schedule.extend(item for item in coarse_configs
                            if config_key(angle_deg, item[1], item[2]) not in {
                                config_key(angle_deg, row[1], row[2]) for row in ladder
                            })
            for area, headland, side in schedule:
                if area <= scene.settings.coverage_tolerance_m2:
                    continue
                current_amax = max((item.area_m2 for item in passing), default=None)
                if current_amax is not None and area + 1e-8 < \
                        current_amax * settings.area_retention_fraction:
                    break
                tested_keys.add(config_key(angle_deg, headland, side))
                passed, variants = try_config(angle_deg, headland, side)
                if passed:
                    feasible_configurations.append((angle_deg, headland, side, area))
                    found = True
                    break
                if turn_order_check_count >= settings.max_turn_order_checks:
                    break
            if not found:
                expanded = sorted(
                    ((config_area(angle_deg, headland, side), headland, side)
                     for side in expanded_side_levels for headland in headlands
                     if config_key(angle_deg, headland, side) not in evaluated_configs
                     and config_key(angle_deg, headland, side) not in tested_keys),
                    key=lambda item: (-item[0], item[1], item[2]),
                )
                for area, headland, side in expanded:
                    if turn_order_check_count >= settings.max_turn_order_checks:
                        break
                    if area <= scene.settings.coverage_tolerance_m2:
                        continue
                    current_amax = max((item.area_m2 for item in passing), default=None)
                    if current_amax is not None and area + 1e-8 < \
                            current_amax * settings.area_retention_fraction:
                        break
                    passed, variants = try_config(angle_deg, headland, side)
                    if passed:
                        feasible_configurations.append((angle_deg, headland, side, area))
                        found = True
                        break

        fallback_body_candidate_count = 0
        if not passing and unresolved_candidates:
            # 未认证局部转向时，仍对全部生成主体重新检查直线包络和机具覆盖，再按同一面积/段数目标择优；连接额度不能暗中决定条带数。
            turn_budget_exhausted = (
                turn_order_check_count >= settings.max_turn_order_checks
            )
            fallback_reason = (
                "TURN_CHECK_BUDGET_EXHAUSTED" if turn_budget_exhausted
                else "NO_LOCAL_TURN_SEQUENCE_CERTIFIED"
            )
            # 转向搜索可能在早期方向停止；仅在明确主体回退中把已探索余量扩展到各方向和相位，防止优待最先访问方向。
            fallback_margins = {
                (round(candidate.headland_m, 4),
                 round(candidate.side_clearance_m, 4))
                for candidate in candidate_cache.values() if candidate is not None
            }
            fallback_margins.add((round(headlands[0], 4), round(base_side, 4)))
            for angle_deg in angles:
                for phase_m in phases_by_angle[angle_deg]:
                    for headland_m, side_reserve_m in sorted(fallback_margins):
                        get_candidate(angle_deg, phase_m, headland_m, side_reserve_m)
            unresolved_by_config: dict[tuple[float, float, float, float, str], Candidate] = {}
            for unresolved in unresolved_candidates:
                unresolved_key = (
                    round(unresolved.angle_deg, 7), round(unresolved.phase_m, 8),
                    round(unresolved.headland_m, 4),
                    round(unresolved.side_clearance_m, 4), unresolved.order_mode,
                )
                unresolved_by_config[unresolved_key] = unresolved
            for candidate in candidate_cache.values():
                if candidate is None:
                    continue
                variants = []
                for mode in ("serpentine", "skip_row_then_fill"):
                    key = (
                        round(candidate.angle_deg, 7), round(candidate.phase_m, 8),
                        round(candidate.headland_m, 4),
                        round(candidate.side_clearance_m, 4), mode,
                    )
                    prior = unresolved_by_config.get(key)
                    if prior is not None:
                        variants.append(prior)
                if not variants:
                    variants = _candidate_body_orders(
                        candidate, scene, validator,
                        fallback_reason=fallback_reason,
                    )
                body_feasible_candidates.extend(variants)
                passing.extend(variants)
                fallback_body_candidate_count += len(variants)
        amax: float | None = max((item.area_m2 for item in body_feasible_candidates),
                                 default=None)
        if amax is None:
            amax = max((item.area_m2 for item in passing), default=None)
        if amax is None:
            empty = GeometryCollection()
            best_failure = max(failed_candidates,
                               key=lambda item: (item.area_m2, -item.segment_count),
                               default=None)
            failure_reason = (
                f"{best_failure.turn_status}; best_checked_candidate="
                f"angle:{best_failure.angle_deg:.3f},phase:{best_failure.phase_m:.3f},"
                f"headland:{best_failure.headland_m:.2f},side:{best_failure.side_clearance_m:.2f}"
                if best_failure else (failures.most_common(1)[0][0] if failures else
                                      "NO_CANDIDATE_PASSED_GEOMETRY_AND_TURN_CHECKS")
            )
            region_results.append(RegionSwathResult(
                field_id=scene.name, region_id=region.region_id,
                status="SWATHS_NOT_FOUND", angle_deg=None, phase_m=None,
                headland_m=None, side_clearance_m=None, segment_count=0,
                amax_area_m2=None, retained_fraction=0.0,
                main_area=empty, headland_area=empty,
                pending_area=region.geometry, segments=[], work_sweeps=empty,
                turn_checks=best_failure.turn_checks if best_failure else [],
                candidate_count=len(candidate_cache),
                f2c_call_count=region_call_counter[0],
                elapsed_s=time.perf_counter() - region_started,
                failure_reason=failure_reason,
                turn_search_stop_reason=(
                    "BUDGET_EXHAUSTED" if turn_order_check_count >= settings.max_turn_order_checks
                    else "CANDIDATES_EXHAUSTED"
                ),
                search_note=(f"directions={len(angles)}; phases={settings.phase_steps}; "
                             f"side_levels={sum(len(values) for values in side_levels_by_angle.values())}; "
                             f"adaptive bounded search; "
                             f"turn_budget_exhausted={turn_order_check_count >= settings.max_turn_order_checks}; "
                             f"failed_order_checks={len(failed_candidates)}"),
            ))
            continue

        # 只在当前可行余量及95%面积范围内的邻近较大余量细化，避免反复裁剪几乎相同的几何组合。
        threshold = amax * settings.area_retention_fraction
        refinement_configs: set[tuple[float, float, float]] = set()
        for angle_deg, headland, side, _ in feasible_configurations:
            local_sides = side_levels_by_angle[angle_deg]
            eligible_larger_sides = [
                value for value in local_sides if value > side
                and config_area(angle_deg, headland, value) + 1e-8 >= threshold
            ]
            side_values = [side]
            if eligible_larger_sides:
                furthest = max(eligible_larger_sides)
                side_values.extend([furthest, (side + furthest) / 2.0])
            side_values = list(dict.fromkeys(round(value, 4) for value in side_values))

            eligible_larger_heads = [
                value for value in headlands if value > headland
                and config_area(angle_deg, value, side) + 1e-8 >= threshold
            ]
            head_values = [headland]
            if eligible_larger_heads:
                nearest = min(eligible_larger_heads)
                head_values.extend([nearest, (headland + nearest) / 2.0])
            head_values = list(dict.fromkeys(round(value, 4) for value in head_values))
            for side_value in side_values:
                for head_value in head_values:
                    if config_area(angle_deg, head_value, side_value) + 1e-8 < threshold:
                        continue
                    refinement_configs.add((angle_deg, head_value, side_value))

        for angle_deg, headland, side in sorted(refinement_configs):
            for phase_m in phases_by_angle[angle_deg]:
                get_candidate(angle_deg, phase_m, headland, side)

        # 细化可能找到比粗搜索Amax更大的主体；冻结95%门槛前须检查全部这种候选。
        higher_area_configs: set[tuple[float, float, float]] = set()
        for candidate in sorted((item for item in candidate_cache.values()
                                 if item is not None and item.area_m2 > amax + 1e-8),
                                key=lambda item: (-item.area_m2, item.segment_count,
                                                  item.angle_deg, item.phase_m)):
            key = config_key(candidate.angle_deg, candidate.headland_m,
                             candidate.side_clearance_m)
            if key in higher_area_configs:
                continue
            if test_feasible(candidate):
                higher_area_configs.add(key)
        amax = max((item.area_m2 for item in body_feasible_candidates), default=amax)
        threshold = amax * settings.area_retention_fraction

        candidate_pool = [item for item in candidate_cache.values()
                          if item is not None and item.area_m2 + 1e-8 >= threshold]
        candidate_pool.sort(key=lambda item: (item.segment_count, -item.area_m2,
                                               item.angle_deg, item.phase_m,
                                               item.side_clearance_m, item.headland_m))
        # 即使较多条带已有转向认证，也比较少量同门槛更少段候选；主体最少段目标与转向证据分开。
        body_candidate_checks = 0
        for candidate in candidate_pool:
            current_min = min((item.segment_count for item in body_feasible_candidates
                               if item.area_m2 + 1e-8 >= threshold), default=None)
            if current_min is not None and candidate.segment_count > current_min:
                break
            candidate_key = (round(candidate.angle_deg, 7), round(candidate.phase_m, 8),
                             round(candidate.headland_m, 4),
                             round(candidate.side_clearance_m, 4),
                             candidate.segment_count,
                             ("serpentine", "skip_row_then_fill"))
            if candidate_key in tested_variants:
                continue
            if turn_order_check_count >= settings.max_turn_order_checks:
                break
            test_turns(candidate)
            body_candidate_checks += 1
            if body_candidate_checks >= 3:
                break
        # 同一分区全部已检查主体候选使用一个Amax；改变田头或侧向退让不能重置95%门槛来获得少条带或转向认证。
        (amax, threshold, selected_count, turn_certified_min,
         feasible) = _select_body_candidate_pool(
            body_feasible_candidates, settings.area_retention_fraction,
        )
        margin_selection = "GLOBAL_BODY_FEASIBLE_POOL"
        fallback_checks = 0
        if not feasible:
            # 旧走廊仅作为方向支撑无法得到安全全覆盖主体时的明确回退，检查规模有上限，最终接缝审计没有豁免。
            fallback_configs = sorted(
                evaluated_configs,
                key=lambda row: (-config_area(*row), row),
            )
            for angle, headland, side in fallback_configs:
                for phase in phases_by_angle.get(angle, ()):
                    if fallback_checks >= 8:
                        break
                    line_key = (round(angle, 7), round(phase, 8))
                    lines = field_line_cache.get(line_key)
                    if not lines:
                        continue
                    fallback_checks += 1
                    try:
                        candidate = _evaluate_candidate(
                            scene, region, region_index, analysis, validator,
                            angle, phase, headland, side, spacing,
                            region.region_id, field_lines=lines,
                            corridor_mode="legacy_safe_fallback",
                        )
                        variants = _candidate_orders(
                            candidate, scene, validator, backend, region_index,
                            settings, turn_cache,
                            ("serpentine", "skip_row_then_fill"),
                        )
                    except Exception as exc:
                        failures[f"LEGACY_FALLBACK:{type(exc).__name__}:{exc}"] += 1
                        continue
                    body_feasible_candidates.extend(
                        item for item in variants if item.coverage_tolerance_passed
                        and item.geometry_status == "PASS"
                    )
                    (amax, threshold, selected_count, turn_certified_min,
                     feasible) = _select_body_candidate_pool(
                        body_feasible_candidates, settings.area_retention_fraction,
                    )
                    if feasible:
                        margin_selection = "LEGACY_SAFE_FALLBACK_SEAM_AUDITED"
                        break
                if feasible or fallback_checks >= 8:
                    break
        candidate_pool = [item for item in candidate_cache.values()
                          if item is not None and item.area_m2 + 1e-8 >= threshold]
        if not feasible:
            empty = GeometryCollection()
            region_results.append(RegionSwathResult(
                field_id=scene.name, region_id=region.region_id,
                status="SWATHS_NOT_FOUND", angle_deg=None, phase_m=None,
                headland_m=None, side_clearance_m=None, segment_count=0,
                amax_area_m2=amax, retained_fraction=0.0,
                main_area=empty, headland_area=empty,
                pending_area=region.geometry, segments=[], work_sweeps=empty,
                turn_checks=(best_failure.turn_checks if (best_failure := max(
                    failed_candidates, key=lambda item: (item.area_m2, -item.segment_count),
                    default=None)) else []),
                candidate_count=len(candidate_cache),
                f2c_call_count=region_call_counter[0],
                elapsed_s=time.perf_counter() - region_started,
                failure_reason="NO_TURN_FEASIBLE_CANDIDATE_WITHIN_RETENTION_THRESHOLD",
                search_note=f"Amax={amax:.6f}; threshold={threshold:.6f}",
                turn_search_stop_reason=(
                    "BUDGET_EXHAUSTED" if turn_order_check_count >= settings.max_turn_order_checks
                    else "CANDIDATES_EXHAUSTED"
                ),
            ))
            continue
        if selected_count is None:
            # 局部转向在预算内未找到认证方案时，明确采用主体回退：保留安全全覆盖主体中最少段数，不声称转弯通过。
            selected = min(feasible, key=lambda item: (
                item.segment_count, -item.area_m2, item.angle_deg, item.phase_m,
                0 if item.order_mode == "serpentine" else 1,
            ))
        else:
            # 条带段数是首要目标；转向认证只在同一最少主体段数的候选间择优。
            selected = min(feasible, key=lambda item: (
                0 if item.turn_status == "PASS" else 1,
                item.reverse_turn_count, item.turn_length_m,
                -item.area_m2, item.angle_deg, item.phase_m,
                0 if item.order_mode == "serpentine" else 1,
            ))
        ranked = sorted(feasible, key=lambda item: (
            item.swept.difference(region.geometry).area,
            0 if item.turn_status == "PASS" else 1,
            -item.area_m2, item.angle_deg, item.phase_m,
        ))
        seam_candidate_pools[region.region_id] = [selected] + [
            item for item in ranked if item is not selected
        ][:7]
        extra_raw = sorted((item for item in candidate_cache.values()
                            if item is not None
                            and selected_count is not None
                            and selected_count < item.segment_count
                            <= selected_count + settings.max_seam_extra_segments_per_region
                            and item.area_m2 + 1e-8 >= threshold),
                           key=lambda item: (
                               item.corridor.difference(region.geometry).area,
                               -item.area_m2, item.segment_count,
                               item.angle_deg, item.phase_m,
                           ))
        diverse_raw: list[Candidate] = []
        seen_grid: set[tuple[float, float]] = set()
        for item in extra_raw:
            grid = (round(item.angle_deg, 6), round(item.phase_m, 5))
            if grid in seen_grid:
                continue
            seen_grid.add(grid)
            diverse_raw.append(item)
            if len(diverse_raw) >= 8:
                break
        seam_extra_raw_pools[region.region_id] = diverse_raw
        selected_margin = (round(selected.headland_m, 4),
                           round(selected.side_clearance_m, 4))
        ordered = _ordered_segments(selected.segments, selected.angle_deg,
                                    selected.order_mode)
        row_order = {segment["task_id"]: index + 1
                     for index, segment in enumerate(ordered)}
        ordered_by_task = {segment["task_id"]: segment for segment in ordered}
        result_segments = []
        for segment in selected.segments:
            row = {key: value for key, value in segment.items()
                   if key not in {"region_geometry", "sweep"}}
            oriented = ordered_by_task[segment["task_id"]]
            _, motion = _task_motion(segment, scene, reverse=oriented["_reverse"])
            row["suggested_order"] = row_order[segment["task_id"]]
            row["heading_deg"] = math.degrees(float(motion.points[0, 2])) % 360.0
            row["implement_on"] = True
            row["geometry"] = segment["geometry"]
            row["work_sweep"] = validator.work_sweep(motion)
            result_segments.append(row)
        # main_area是独立主体覆盖义务，真实机具扫掠可略超该核心；所有扫到的归属目标登记为主体，避免又计入田头。容差残余显式待处理，三类面不重叠且回合冻结分区。
        pending = polygonal(selected.main_area.difference(selected.swept))
        main_area = polygonal(selected.swept.intersection(region.geometry).difference(pending))
        headland_area = polygonal(region.geometry.difference(
            unary_union([main_area, pending])
        ))
        profile["selection_reconciliation_s"] = max(
            0.0, time.perf_counter() - region_started
            - profile["direction_phase_generation_s"]
            - profile["f2c_line_generation_s"]
            - profile["candidate_geometry_s"] - profile["turn_validation_s"],
        )
        validated_signatures = {
            (round(item.angle_deg, 7), round(item.phase_m, 8),
             round(item.headland_m, 4), round(item.side_clearance_m, 4))
            for item in body_feasible_candidates
        }
        uninspected = sum(
            candidate is not None
            and (round(candidate.angle_deg, 7), round(candidate.phase_m, 8),
                 round(candidate.headland_m, 4), round(candidate.side_clearance_m, 4))
            not in validated_signatures
            for candidate in candidate_cache.values()
        )
        region_results.append(RegionSwathResult(
            field_id=scene.name, region_id=region.region_id,
            status="SWATHS_COMPLETE", angle_deg=selected.angle_deg,
            phase_m=selected.phase_m, headland_m=selected.headland_m,
            side_clearance_m=selected.side_clearance_m,
            segment_count=selected.segment_count, amax_area_m2=amax,
            retained_fraction=selected.area_m2 / amax if amax else 1.0,
            main_area=main_area, headland_area=headland_area,
            pending_area=pending, segments=result_segments,
            work_sweeps=selected.swept, turn_checks=selected.turn_checks,
            candidate_count=len(candidate_cache),
            f2c_call_count=region_call_counter[0],
            elapsed_s=time.perf_counter() - region_started, failure_reason="",
            search_note=(f"candidate_pool={len(candidate_pool)}; "
                         f"retention_threshold={threshold:.6f}m2; "
                         f"order={selected.order_mode}; turns={selected.turn_status}; "
                         f"side_levels={len(side_levels_by_angle.get(selected.angle_deg, expanded_side_levels))}; "
                         f"turn_order_checks={turn_order_check_count}/"
                         f"{settings.max_turn_order_checks}; "
                         f"phases_per_direction={max(map(len, phases_by_angle.values()), default=0)}; "
                         f"event_phase_cap={settings.max_event_phases}; "
                         f"phase_refinements={phase_refinement_count}/"
                         f"{max_phase_refinements}; "
                         f"margin_selection={margin_selection}; "
                         f"selected_headland={selected_margin[0]}; "
                         f"selected_side_reserve={selected_margin[1]}; "
                         f"turn_budget_exhausted={turn_order_check_count >= settings.max_turn_order_checks}; "
                         f"body_min={selected_count}; "
                         f"turn_certified_min={turn_certified_min}; "
                         f"uninspected_candidates={uninspected}; "
                         f"body_fallback_candidates={fallback_body_candidate_count}; "
                         f"legacy_corridor_fallback_checks={fallback_checks}; "
                         f"adaptive_refinement={len(refinement_configs)}; "
                         f"feasible_directions={len(feasible_configurations)}/{len(angles)}"),
            turn_status=selected.turn_status,
            required_main_area_m2=selected.area_m2,
            required_main_area=selected.main_area,
            body_feasible_min_segments=selected_count,
            turn_certified_min_segments=turn_certified_min,
            search_candidates_uninspected=uninspected,
            turn_search_stop_reason=(
                "BUDGET_EXHAUSTED" if turn_order_check_count >= settings.max_turn_order_checks
                else "ADAPTIVE_SEARCH_STOPPED" if uninspected
                else "CANDIDATES_EXHAUSTED"
            ),
            profile={**profile, "total_s": time.perf_counter() - region_started},
            order_mode=selected.order_mode,
            pre_seam_segment_count=selected_count,
        ))

    joint_alignment = _joint_align_regions(
        scene, analysis, region_results, settings, validator, backend, spacing,
    )
    call_count += joint_alignment["f2c_call_count"]
    seam_extra_validation = _validate_extra_seam_candidates(
        scene, analysis, region_results, seam_extra_raw_pools,
        seam_candidate_pools, settings, validator, backend,
    )
    # 贪心编辑可能落在不同局部最优；从同一联合起点比较最少条带分支与有限增条带分支，仅在真实减少重叠时允许小幅增加。
    if settings.max_seam_extra_segments_per_region:
        same_count_results = copy.deepcopy(region_results)
        extra_coordination = _coordinate_seam_candidates(
            scene, analysis, region_results, seam_candidate_pools, validator,
            settings,
        )
        extra_work_length = _remove_short_work_segments(
            scene, region_results, settings.min_work_segment_m,
        )
        same_count_coordination = _coordinate_seam_candidates(
            scene, analysis, same_count_results, seam_candidate_pools, validator,
            replace(settings, max_seam_extra_segments_per_region=0),
        )
        same_count_work_length = _remove_short_work_segments(
            scene, same_count_results, settings.min_work_segment_m,
        )
        extra_overlap = audit_seams(scene, analysis, region_results)["extra_area_m2"]
        same_count_overlap = audit_seams(
            scene, analysis, same_count_results)["extra_area_m2"]
        prefer_same_count = (
            same_count_work_length["status"] == "PASS"
            and (extra_work_length["status"] != "PASS"
                 or same_count_overlap <= extra_overlap
                 + scene.settings.coverage_tolerance_m2)
        )
        if prefer_same_count:
            region_results = same_count_results
            work_length = same_count_work_length
            selected_branch = "SAME_COUNT"
        else:
            work_length = extra_work_length
            selected_branch = "BOUNDED_EXTRA"
        seam_coordination = {
            key: same_count_coordination[key] + extra_coordination[key]
            for key in ("rounds", "candidates_inspected", "replacements",
                        "segments_removed", "pair_combinations_inspected",
                        "pair_replacements", "elapsed_s")
        }
        seam_coordination["extra_segments_selected"] = (
            0 if prefer_same_count else extra_coordination["extra_segments_selected"])
        seam_coordination["stop_reason"] = (
            "SELECTED:" + selected_branch
            + ";SAME_COUNT:" + same_count_coordination["stop_reason"]
            + ";EXTRA:" + extra_coordination["stop_reason"]
        )
    else:
        seam_coordination = _coordinate_seam_candidates(
            scene, analysis, region_results, seam_candidate_pools, validator,
            settings,
        )
        work_length = _remove_short_work_segments(
            scene, region_results, settings.min_work_segment_m,
        )
    seam_local = _coordinate_local_seam_tasks(
        scene, analysis, region_results, validator, settings,
    )
    previous_removed = [row for row in work_length["review_rows"]
                        if row["status"].startswith("REMOVED")]
    previous_ids = list(work_length["removed_task_ids"])
    work_length = _remove_short_work_segments(
        scene, region_results, settings.min_work_segment_m,
    )
    work_length["review_rows"] = previous_removed + work_length["review_rows"]
    work_length["removed_task_ids"] = previous_ids + work_length["removed_task_ids"]
    seam_started = time.perf_counter()
    seam = audit_seams(scene, analysis, region_results)
    seam_task_rows = overlapping_work_tasks(
        scene, region_results,
        [row for row in seam["pairs"] if row["status"] == "EXCESS_OVERLAP"],
        minimum_area_m2=scene.settings.coverage_tolerance_m2,
    )
    seam_assignments = cross_region_assignments(scene, analysis, region_results)
    dependent_regions = {row["owner_region_id"] for row in seam_assignments}
    assisted_required_area = sum(row["area_m2"] for row in seam_assignments)
    seam_elapsed = time.perf_counter() - seam_started

    # 按全田扫掠并集重新核对作业归属；跨人工接缝的合法扫掠应计入被覆盖目标的主体账本，不能同时留在田头/待作区域。冻结分区几何不变。
    regions_by_id = {item.region_id: item for item in analysis.regions}
    field_sweeps = unary_union([
        item.work_sweeps for item in region_results
        if not item.work_sweeps.is_empty
    ]) if any(not item.work_sweeps.is_empty for item in region_results) else GeometryCollection()
    for item in region_results:
        region_geometry = regions_by_id[item.region_id].geometry
        covered_in_region = polygonal(field_sweeps.intersection(region_geometry))
        if item.status == "SWATHS_COMPLETE":
            item.pending_area = polygonal(item.required_main_area.difference(field_sweeps))
            item.main_area = polygonal(covered_in_region.difference(item.pending_area))
            item.headland_area = polygonal(region_geometry.difference(
                unary_union([item.main_area, item.pending_area])
            ))
        else:
            item.main_area = covered_in_region
            item.pending_area = polygonal(region_geometry.difference(covered_in_region))
            item.headland_area = GeometryCollection()

    complete = sum(item.status == "SWATHS_COMPLETE" for item in region_results)
    required_main = _whole_field_coverage_audit(
        (item.required_main_area for item in region_results
         if item.status == "SWATHS_COMPLETE"),
        (item.work_sweeps for item in region_results),
        scene.settings.coverage_tolerance_m2,
    )
    if complete == len(region_results) and required_main["passed"]:
        status = "SWATHS_COMPLETE"
    elif complete:
        status = "SWATHS_PARTIAL"
    else:
        status = "SWATHS_NOT_FOUND"
    ledger = _area_ledger_audit(scene, analysis, region_results)
    ledger_delta = ledger["area_ledger_delta_m2"]
    ledger_closed = ledger["area_ledger_closed"]
    if status == "SWATHS_COMPLETE" and not ledger_closed:
        status = "SWATHS_PARTIAL"
    covered_area = ledger["body_area_m2"]
    connections, continuous_runs, run_members = _continuous_runs(
        scene, analysis, region_results, validator,
        joint_alignment["connection_status"],
        joint_alignment["connection_attempts"],
        joint_alignment["connection_candidate_count"],
        joint_alignment["connection_unattempted_count"],
        joint_alignment["connection_failure_reasons"],
    )
    seam_status = seam["status"] if status == "SWATHS_COMPLETE" else "NOT_EVALUATED"
    stage_acceptance = (status == "SWATHS_COMPLETE" and required_main["passed"]
                        and ledger_closed and seam_status == "PASS"
                        and work_length["status"] == "PASS")
    checks = {
        "frozen_partition_region_count": len(analysis.regions),
        "frozen_partition_connections_unchanged": len(analysis.connections),
        "source_target_unchanged": analysis.target.wkb == scene.target.wkb,
        "source_travel_unchanged": analysis.travel.wkb == scene.travel.wkb,
        "analysis_vehicle_parameters_match": True,
        "whole_field_main_coverage_passed": required_main["passed"],
        "whole_field_required_area_m2": required_main["required_area_m2"],
        "whole_field_uncovered_area_m2": required_main["residual_area_m2"],
        "whole_field_max_uncovered_component_m2": required_main["maximum_residual_component_m2"],
        "whole_field_uncovered_component_count": required_main["residual_component_count"],
        "whole_field_coverage_tolerance_m2": required_main["tolerance_m2"],
        "area_ledger_delta_m2": ledger_delta,
        "area_ledger_closed": ledger_closed,
        "area_sum_delta_m2": ledger["area_sum_delta_m2"],
        "area_union_area_delta_m2": ledger["area_union_area_delta_m2"],
        "area_region_delta_m2": ledger["area_region_delta_m2"],
        "area_pairwise_overlap_m2": ledger["area_pairwise_overlap_m2"],
        "area_outside_region_m2": ledger["area_outside_region_m2"],
        "area_ledger_tolerance_m2": ledger["area_ledger_tolerance_m2"],
        "area_ledger_precision_m": ledger["area_ledger_precision_m"],
        "all_segments_are_connected_lines": all(
            segment["geometry"].geom_type == "LineString"
            and segment["geometry"].length > 0
            for result in region_results for segment in result.segments),
        "all_local_turns_checked_for_accepted_regions": all(
            all(check["status"] == "PASS" for check in result.turn_checks)
            for result in region_results if result.status == "SWATHS_COMPLETE"),
        "f2c_generate_swaths_calls": call_count,
        "continuous_run_count": len(continuous_runs),
        "joint_alignment_attempt_count": joint_alignment["attempt_count"],
        "joint_alignment_candidate_count": sum(
            joint_alignment["connection_candidate_count"].values()),
        "joint_alignment_unattempted_candidate_count": sum(
            joint_alignment["connection_unattempted_count"].values()),
        "joint_alignment_candidate_budget": joint_alignment["budget"],
        "joint_alignment_budget_scope": "PER_CONNECTION",
        "joint_inspected_connection_count": joint_alignment["inspected_connection_count"],
        "joint_aligned_connection_count": joint_alignment["aligned_connection_count"],
        "joint_candidate_continuous_run_count": joint_alignment["candidate_continuous_run_count"],
        "joint_uninspected_connection_count": joint_alignment["uninspected_connection_count"],
        "joint_alignment_elapsed_s": joint_alignment["elapsed_s"],
        "seam_extra_area_m2": seam["extra_area_m2"],
        "seam_overlap_footprint_area_m2": seam["overlap_footprint_area_m2"],
        "seam_pair_overlap_sum_m2": seam["pair_overlap_sum_m2"],
        "seam_budget_m2": seam["budget_m2"],
        "seam_centerline_outside_m": seam["centerline_outside_m"],
        "seam_pair_count": len(seam["pairs"]),
        "seam_excess_pair_count": sum(row["status"] != "PASS" for row in seam["pairs"]),
        "seam_audit_elapsed_s": seam_elapsed,
        "seam_coordination_rounds": seam_coordination["rounds"],
        "seam_candidates_inspected": seam_coordination["candidates_inspected"],
        "seam_candidate_replacements": seam_coordination["replacements"],
        "seam_segments_removed": seam_coordination["segments_removed"],
        "seam_pair_combinations_inspected": seam_coordination[
            "pair_combinations_inspected"],
        "seam_pair_replacements": seam_coordination["pair_replacements"],
        "seam_extra_segments_selected": seam_coordination["extra_segments_selected"],
        "seam_search_stop_reason": seam_coordination["stop_reason"],
        "seam_coordination_elapsed_s": seam_coordination["elapsed_s"],
        "seam_local_task_candidates_inspected": seam_local["candidates_inspected"],
        "seam_local_task_replacements": seam_local["replacements"],
        "seam_local_task_operation_counts": seam_local["operation_counts"],
        "seam_local_task_stop_reason": seam_local["stop_reason"],
        "seam_local_task_round_budget": seam_local["round_budget"],
        "seam_local_task_rounds_executed": seam_local["rounds_executed"],
        "seam_local_task_elapsed_s": seam_local["elapsed_s"],
        "seam_task_overlap_count": len(seam_task_rows),
        "seam_extra_affected_region_count": seam_extra_validation["affected_region_count"],
        "seam_extra_candidate_inspected": seam_extra_validation["inspected"],
        "seam_extra_candidate_accepted": seam_extra_validation["accepted"],
        "seam_extra_validation_elapsed_s": seam_extra_validation["elapsed_s"],
        "work_length_status": work_length["status"],
        "minimum_work_segment_m": work_length["minimum_m"],
        "short_work_removed_count": len(work_length["removed_task_ids"]),
        "short_work_removed_sub_1m_count": sum(
            row["status"] == "REMOVED_SUB_1M" for row in work_length["review_rows"]),
        "short_work_review_retained_count": sum(
            row["status"].startswith("RETAINED") for row in work_length["review_rows"]),
        "short_work_review_threshold_m": work_length["review_m"],
        "short_work_removed_task_ids": work_length["removed_task_ids"],
        "short_work_unresolved_count": len(work_length["unresolved_task_ids"]),
        "short_work_unresolved_task_ids": work_length["unresolved_task_ids"],
        "implement_lag_certification": "PENDING_MEASURED_ON_OFF_LAG",
        "dispatch_dependency_count": len(seam_assignments),
        "dispatch_dependent_region_count": len(dependent_regions),
        "dispatch_assisted_required_area_m2": assisted_required_area,
        "independent_dispatch_status": (
            "NEEDS_CROSS_REGION_PROVIDER_TASKS" if seam_assignments
            else "NO_CROSS_REGION_PROVIDER_DEPENDENCY"
        ),
        "scope": "body straight-swath coverage only; headland reserve is not certified as worked; no full route",
    }
    return SwathStageResult(
        scene.name, status, region_results,
        connections, continuous_runs, run_members, float(scene.target.area),
        float(covered_area), float(ledger["pending_area_m2"]),
        time.perf_counter() - started, getattr(f2c, "__version__", "unknown"),
        checks, stage_acceptance, required_main["geometry"],
        required_main["required_geometry"], required_main["swept_geometry"],
        float(ledger["body_area_m2"] / scene.target.area) if scene.target.area else 0.0,
        float(ledger["headland_area_m2"] / scene.target.area) if scene.target.area else 0.0,
        float(ledger["pending_area_m2"] / scene.target.area) if scene.target.area else 0.0,
        ledger_delta,
        seam_status, seam["pairs"], seam["overlap_footprint"], seam_assignments,
        work_length["review_rows"], seam_task_rows,
    )
