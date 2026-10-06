"""路线数据与批次管理。输入冻结条带包及配置，输出可复核的新批次、表格和图库。
算法分发到 route.py；本文件负责来源核验、数据模型、导出和恢复。
各策略的参考几何验收、车辆认证和风格状态分别保留，不能混成一个 PASS。

分节目录：
1. 统一数据模型、输入校验和路线批调度
2. 严格路线结果与采样导出
3. 导出结果回读与独立复核
4. 逐田综合图与图库
5. 并行进程隔离
6. 已保存批次恢复与末端续作
7. 单田有界恢复入口
8. 田头补作批处理
9. 规则路线的独立验收和展示
10. 默认完整几何参考的逐田事务、连续速度账本、独立复算与多核资源调度
"""
from __future__ import annotations


# ==========================================================================
# 1. 统一数据模型、输入校验和路线批调度
# 读取冻结条带、场景及参数，构造 FieldInput，分发策略并管理批次来源指纹。
# ==========================================================================

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import time
from typing import Any
from typing import Iterable

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import affinity
from shapely import force_2d
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.ops import unary_union

from scene import Motion
from scene import Pose
from scene import Scene
from scene import load_scene
from scene import wrap
from planner import F2CBackend
from validator import swept_polygons
from validator import RouteValidator as Validator
from io_utils import atomic_json, config_section, config_reference, read_json, PROJECT_CONFIG


_LEGACY_ROUTE_API = Path(__file__).resolve().parents[1] / "tempscript" / "route_compat"
_LEGACY_ROUTE_NAMES = frozenset({
    "route_seed_continue.py", "route_terminal_insert.py", "route_preserve_body.py",
    "route_residual.py", "route_reserve_candidates.py", "route_contour_choices.py",
})
# 正式模块只导入合并后的实现；这里只为旧实验的模块名提供独立搜索路径。
if str(_LEGACY_ROUTE_API) not in sys.path:
    sys.path.append(str(_LEGACY_ROUTE_API))


FROZEN_FILES = ("scene.py", "planner.py", "swath_planner.py",
                "swath_seams.py", "swath_batch.py")
SCHEMA_VERSION = "route-v3-joint-headland"


# 严格路线阶段参数，和参考策略的 ApproxRouteSettings 分开。
# 预算限制搜索量；不能用扩大预算把未认证的车辆条件改成通过。
@dataclass(frozen=True)
class RouteSettings:
    """历史严格路线的搜索、田头与恢复配置；与ApproxRouteSettings及参考批处理资源预算分开。
    
    Stage-specific search limits and explicit, provisional operation times."""

    max_order_candidates: int = 5
    max_guide_trials: int = 2
    max_field_seconds: float = 30.0
    last_body_replan_seconds: float = 12.0
    max_reverse_legs: int = 2
    max_gear_shifts: int = 4
    stop_steer_seconds: float = 3.0
    time_slack_fraction: float = 0.05
    endpoint_policy: str = "STRICT_WORKED_ONLY"
    replan_headlands: bool = True
    headland_strategy: str = "AUTO_FAST"
    headland_entry_rank: str = "MEAN"
    headland_compare_entry_directions: bool = False
    continuous_headland_contours: bool = False
    bidirectional_headland_contours: bool = False
    directional_block_continuation: bool = False
    prefer_radius_compatible_orders: bool = False
    plan_headland_work: bool = False
    # Standalone passes deliberately do not form part of the body itinerary.
    headland_work_mode: str = "SEPARATE_PASSES"
    headland_pass_count: int = 3
    retry_near_complete: bool = True
    # Safe reserve-gap work can reduce omissions even when some generated
    # headland tasks remain unconnected. Their sweeps stay protected, and
    # each supplemental pass must enter and rejoin the audited route.
    partial_reserve_gap_work: bool = True
    adaptive_headland_entry_rank: bool = True
    max_headland_seconds: float = 60.0
    max_headland_branch_nodes: int = 128
    headland_simplification_tolerance_m: float = 0.25
    headland_positioning_allowance_m: float = 0.2
    adaptive_headland_simplification: bool = True

    @classmethod
    def from_file(cls, path: Path | None) -> "RouteSettings":
        """从统一routes模式或外部旧JSON构造严格设置，拒绝未知键及非法值；None采用本类型默认值。"""
        if path is None:
            return cls()
        data = config_section(path, "routes")
        if not isinstance(data, dict):
            raise ValueError("route config 必须是 JSON 对象")
        allowed = {item.name for item in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"未知 route 参数: {sorted(unknown)}")
        defaults = cls()
        for item in fields(cls):
            if item.name not in data:
                continue
            value = data[item.name]
            default = getattr(defaults, item.name)
            if item.name == "endpoint_policy":
                if value != "STRICT_WORKED_ONLY":
                    raise ValueError("本轮规则禁止未作业主体内的局部调头借道")
                continue
            if item.name == "headland_strategy":
                if type(value) is not str or value not in {"AUTO_FAST", "AUTO", "UNIFORM", "DIRECTIONAL"}:
                    raise ValueError("headland_strategy 必须为 AUTO_FAST/AUTO/UNIFORM/DIRECTIONAL")
                continue
            if item.name == "headland_entry_rank":
                if type(value) is not str or value not in {"MEAN", "MIN"}:
                    raise ValueError("headland_entry_rank 必须为 MEAN/MIN")
                continue
            if item.name == "headland_work_mode":
                if type(value) is not str or value not in {"SEPARATE_PASSES", "JOINT"}:
                    raise ValueError("headland_work_mode 必须为 SEPARATE_PASSES/JOINT")
                continue
            if item.name in {"replan_headlands", "plan_headland_work",
                             "retry_near_complete", "partial_reserve_gap_work",
                             "adaptive_headland_simplification",
                             "adaptive_headland_entry_rank",
                             "headland_compare_entry_directions",
                             "continuous_headland_contours",
                             "bidirectional_headland_contours",
                             "directional_block_continuation",
                             "prefer_radius_compatible_orders"}:
                if type(value) is not bool:
                    raise ValueError(f"{item.name} 必须为布尔值")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"route 参数必须为数值: {item.name}")
            if not math.isfinite(value):
                raise ValueError(f"route 参数非有限: {item.name}")
            if isinstance(default, int):
                if type(value) is not int or value < 1:
                    raise ValueError(f"route 参数必须为正整数: {item.name}")
            elif value <= 0:
                raise ValueError(f"route 参数必须为正数: {item.name}")
        result = cls(**data)
        if result.headland_pass_count not in (2, 3):
            raise ValueError("headland_pass_count 只允许 2 或 3 趟")
        if result.max_reverse_legs != 2 or result.max_gear_shifts != 4:
            raise ValueError("每次连接固定最多两段倒车、四次换挡")
        if abs(result.time_slack_fraction - 0.05) > 1e-12:
            raise ValueError("驾驶偏好时间容差固定为 5%")
        if result.headland_simplification_tolerance_m > 5.0:
            raise ValueError("田头简化容差不得超过 5 m")
        if result.headland_positioning_allowance_m > 1.0:
            raise ValueError("田头候选定位余量不得超过 1 m")
        if result.bidirectional_headland_contours and not result.continuous_headland_contours:
            raise ValueError("双方向田头候选必须启用 continuous_headland_contours")
        return result


# 一个具备原始编号的作业任务，保存参考线、方向及机具扫掠。
# 这里的几何均在当前田块的局部米制坐标中，导出时才转换回世界坐标。
@dataclass(frozen=True)
class FrozenTask:
    """已发布作业任务：保留任务号、分区归属、参考线与独立扫掠；下游不能静默变更覆盖义务。"""
    field_id: str
    region_id: str
    task_id: str
    row_index: int
    suggested_order: int
    reference_line: LineString
    heading_rad: float
    frozen_sweep: BaseGeometry
    length_m: float
    work_kind: str = "STRAIGHT_BODY"
    motion_points: np.ndarray | None = None
    alternative_motion_points: tuple[np.ndarray, ...] = ()
    required_target_sweep: BaseGeometry | None = None


@dataclass(frozen=True)
class RegionInput:
    """路线输入中的一个冻结分区及所属任务号；sequence_index是规划索引，不是GIS的fid。"""
    region_id: str
    sequence_index: int
    geometry: BaseGeometry
    headland: BaseGeometry
    task_ids: tuple[str, ...]


# 完整单田输入：场景来源、分区、冻结任务、拓扑和阶段派生信息。
# target 的归属和实际作业提供方须保持可追踪，跨区代作不能默默消失。
@dataclass(frozen=True)
class FieldInput:
    """一田的路线输入，含来源坐标、局部原点、场景路径和冻结任务；几何计算在该田局部米制坐标进行。"""
    index: int
    field_id: str
    scene_path: str
    source_geometry: BaseGeometry
    source_crs: str
    metric_crs: str
    origin: tuple[float, float]
    regions: tuple[RegionInput, ...]
    tasks: tuple[FrozenTask, ...]
    upstream_seam_quality_status: str
    upstream_acceptance_passed: bool
    upstream_area_ledger_delta_m2: float | None
    required_body: BaseGeometry | None = None


@dataclass(frozen=True)
class WorkVariant:
    """同一冻结任务的一种经过核验的遍历方案；反向时必须保持原机具覆盖，不能只交换车辆端点。"""
    task: FrozenTask
    reversed: bool
    motion: Motion
    sweep: BaseGeometry
    sweep_error_m2: float

    @property
    def start(self) -> Pose:
        """读取本遍历运动的起点局部位姿，而非原条带未经调整的端点。"""
        p = self.motion.points[0]
        return Pose(float(p[0]), float(p[1]), float(p[2]))

    @property
    def end(self) -> Pose:
        """读取本遍历运动的终点局部位姿，供后继连接重新计算。"""
        p = self.motion.points[-1]
        return Pose(float(p[0]), float(p[1]), float(p[2]))


# 一条连接的实际运动、操作代价和认证证据。
# 转移时机具关闭，不能把连接线的几何缓冲误计成已完成作业面积。
@dataclass
class Connection:
    """一次任务间连接及失败证据；motion为None表示未获得有效运动，不能用拓扑邻接补造连接。"""
    from_task: str | None
    to_task: str | None
    motion: Motion | None
    method: str
    seconds: float
    reverse_m: float
    reverse_legs: int
    gear_shifts: int
    events: list[dict[str, Any]] = field(default_factory=list)
    borrowed_area_m2: float = 0.0
    failure_codes: list[str] = field(default_factory=list)
    endpoint_unworked_area_m2: float = 0.0


@dataclass
class RegionRoute:
    """历史严格分区路线的任务顺序、运动与累计作业证据；失败状态必须向整田传递。"""
    region_id: str
    status: str
    order_mode: str = ""
    motions: list[Motion] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    task_order: list[str] = field(default_factory=list)
    completed_sweep: BaseGeometry = field(default_factory=GeometryCollection)
    connection_seconds: float = math.inf
    reverse_m: float = 0.0
    gear_shifts: int = 0
    irregular_jumps: int = 0
    explored_candidates: int = 0
    sweep_error_m2: float = 0.0
    reason: str = ""
    elapsed_s: float = 0.0


# 单田结果对象，包括主体、分区连接、田头、失败原因与各自验收状态。
# 保存部分有效结果有研究价值，但不因此宣称整田连续或实车可执行。
@dataclass
class FieldRoute:
    """历史严格整田路线结果，保留已完成部分、失败原因和恢复来源，不等于默认几何参考的数据结构。"""
    index: int
    field_id: str
    status: str
    region_order: list[str]
    routes: list[RegionRoute]
    failures: list[dict[str, Any]]
    elapsed_s: float
    source_crs: str
    metric_crs: str
    origin: tuple[float, float]
    source_geometry: BaseGeometry
    upstream_seam_quality_status: str
    upstream_acceptance_passed: bool
    upstream_area_ledger_delta_m2: float | None
    implementation_lag_status: str = "PARAMETERS_UNVERIFIED"
    transfer_status: str = "NOT_PLANNED"
    stage_audit_passed: bool = False
    error: str = ""
    statistics: dict[str, Any] = field(default_factory=dict)
    adapted_job: Any = None
    required_body: Any = None
    preparation: dict[str, Any] = field(default_factory=dict)
    separate_headland_tasks: tuple[FrozenTask, ...] = ()


def _sha256(path: Path) -> str:
    """对实际文件计算摘要；已合并的旧 src 名称解析到唯一正式实现。"""
    from validator import legacy_source_path
    return hashlib.sha256(legacy_source_path(path).read_bytes()).hexdigest()


def _source_fingerprint_matches(path: Path, expected: str) -> bool:
    """只认可当前文件的精确字节摘要；冻结模块与其它模块均不再使用旧重构豁免。"""
    path = Path(path)
    try:
        actual = _sha256(path)
    except OSError:
        return False
    if actual == expected:
        return True
    if path.absolute().parent != Path(__file__).absolute().parent or path.name in FROZEN_FILES:
        return False
    # 旧重构豁免对应的实体模块已撤下；当前只认可精确摘要。
    # 删除失效的豁免配置不能顺便放宽冻结来源验证。
    return False


def _crs_equiv(left: str, right: str) -> bool:
    """判断场景米制CRS与来源声明是否兼容；LOCAL_METRIC可与本地工程米制声明对应，不用于转换经纬度。"""
    if left == right:
        return True
    return (left == "LOCAL_METRIC" and right.startswith("LOCAL_CS[")) or (
        right == "LOCAL_METRIC" and left.startswith("LOCAL_CS["))


# 先检查冻结条带包的必要文件、源码及配置指纹，再构造路线输入。
# 未知来源或变化必须拒绝，不以文件能打开作为完整输入验收。
def _bundle_path(bundle: Path, name: str) -> Path:
    """封存清单相对路径以包目录为基准，绝对路径保持原义。
    
    Resolve legacy absolute references or portable, package-relative paths."""
    path = Path(name)
    return path if path.is_absolute() else bundle / path


def _bundle_registry() -> Path:
    """返回唯一项目config.json的位置；可信摘要位于其compatibility分组，不再读独立配置。"""
    return Path(__file__).resolve().parents[1] / "config.json"


def _verify_bundle_seal(bundle: Path) -> str:
    """检查显式发布摘要注册和包内文件校验；缺失、未登记或被改写的包立即拒绝。
    
    Check payload bytes against an explicitly registered release manifest."""
    checksum_path = bundle / "bundle_checksums.json"
    if not checksum_path.is_file():
        raise ValueError("UNSEALED_SWATH_BUNDLE: run tempscript/seal_swath_bundle.py")
    digest = _sha256(checksum_path)
    try:
        registry = config_section(_bundle_registry(), "compatibility", "trusted_swath_bundles")
    except (ValueError,KeyError,TypeError,OSError) as exc:
        raise ValueError('UNTRUSTED_SWATH_RELEASE') from exc
    if not isinstance(registry,dict) or registry.get("version") != "SWATH_RELEASES_V1" or not isinstance(registry.get("releases"),dict) or digest not in registry["releases"]:
        raise ValueError("UNTRUSTED_SWATH_RELEASE")
    checks = json.loads(checksum_path.read_text())
    if not isinstance(checks,dict) or checks.get("version") != "SWATH_BUNDLE_V1" or not isinstance(checks.get("files"), dict):
        raise ValueError("INVALID_BUNDLE_CHECKSUMS")
    required = {"swath_results.gpkg", "swath_results_metric.gpkg",
                "swath_batch_summary.json", "prepared/manifest.json", "field_results.json"}
    if not required <= checks["files"].keys():
        raise ValueError("INCOMPLETE_BUNDLE_CHECKSUMS")
    for relative, record in checks["files"].items():
        if not isinstance(relative,str) or not isinstance(record,dict):
            raise ValueError("INVALID_BUNDLE_CHECKSUMS")
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file():
            raise ValueError(f"BUNDLE_PAYLOAD_MISSING: {relative}")
        if path.stat().st_size != record.get("bytes") or _sha256(path) != record.get("sha256"):
            raise ValueError(f"BUNDLE_PAYLOAD_CHANGED: {relative}")
    manifest = json.loads((bundle / "prepared/manifest.json").read_text())
    for relative in [manifest["input"], manifest["config"],
                     *(e["scene_path"] for e in manifest["fields"])]:
        if Path(relative).is_absolute() or relative not in checks["files"]:
            raise ValueError(f"UNSEALED_BUNDLE_REFERENCE: {relative}")
    return digest


def seal_swath_bundle(source: Path, destination: Path) -> dict[str, Any]:
    """核对包与跨文件一致性后复制发布到新目录并登记摘要，不覆盖源包。
    独立覆盖审核由 tempscript/seal_swath_bundle.py 在调用本函数前执行；
    直接调用本函数不能代替该审核。统一配置只增加可信发布记录。
    
    Explicitly publish a new portable package; never rewrite the source."""
    import shutil
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("SEALED_DESTINATION_MUST_BE_NEW_AND_OUTSIDE_SOURCE")
    summary, manifest = _verify_bundle(source, require_seal=False)
    jobs = _field_rows(source, manifest)
    records = json.loads((source / "field_results.json").read_text())
    by_id = {r["field_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {j.field_id for j in jobs}:
        raise ValueError("SEAL_FIELD_RESULTS_MISMATCH")
    for job in jobs:
        r = by_id[job.field_id]
        if r["seam_quality_status"] != job.upstream_seam_quality_status or bool(r["acceptance_passed"]) != job.upstream_acceptance_passed:
            raise ValueError(f"SEAL_UPSTREAM_STATUS_MISMATCH: {job.field_id}")
    # Cross-file checks prevent registering the original audit's changed status.
    from collections import Counter
    if dict(Counter(j.upstream_seam_quality_status for j in jobs)) != {k:v for k,v in summary["seam_quality_counts"].items() if v}:
        raise ValueError("SEAL_UPSTREAM_SUMMARY_MISMATCH")
    config, original = (_bundle_path(source, manifest[k]) for k in ("config", "input"))
    if not original.is_file():
        raise ValueError("SEAL_ORIGINAL_INPUT_MISSING")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        "cache", ".cache", ".tmp", "matplotlib_config", "bundle_checksums.json"))
    (destination / "input").mkdir(exist_ok=True)
    shutil.copy2(config, destination / "input/config.json")
    shutil.copy2(original, destination / "input/source.gpkg")
    portable = {**manifest, "input": "input/source.gpkg", "config": "input/config.json"}
    portable["fields"] = []
    for i, entry in enumerate(manifest["fields"]):
        relative = f"prepared/scenes/{i:04d}.json"
        shutil.copy2(_bundle_path(source, entry["scene_path"]), destination / relative)
        portable["fields"].append({**entry, "scene_path": relative})
    atomic_json(destination / "prepared/manifest.json", portable)
    payload = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            payload[path.relative_to(destination).as_posix()] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    checks = {"version": "SWATH_BUNDLE_V1", "source_bundle": str(source),
              "source_summary_sha256": _sha256(source / "swath_batch_summary.json"),
              "files": payload}
    atomic_json(destination / "bundle_checksums.json", checks)
    digest = _sha256(destination / "bundle_checksums.json")
    registry_path = _bundle_registry()
    project = read_json(registry_path)
    registry = project["compatibility"]["trusted_swath_bundles"]
    if registry.get("version") != "SWATH_RELEASES_V1":
        raise ValueError("INVALID_RELEASE_REGISTRY")
    registry["releases"][digest] = {"source_bundle": str(source), "field_count": len(jobs)}
    project["compatibility"]["trusted_swath_bundles"] = registry
    atomic_json(registry_path, project)
    _verify_bundle(destination)
    return {"bundle": str(destination), "release_sha256": digest, "field_count": len(jobs)}


def _verify_bundle(bundle: Path, *, require_seal: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """核对发布包、配置、源码与输入清单，返回可靠输入；默认要求封存，不自动补清单认可未知修改。"""
    bundle = bundle.resolve()
    if require_seal:
        _verify_bundle_seal(bundle)
    required = ("swath_results.gpkg", "swath_results_metric.gpkg",
                "swath_batch_summary.json", "prepared/manifest.json")
    for name in required:
        if not (bundle / name).is_file():
            raise ValueError(f"冻结条带输入缺失: {name}")
    summary = json.loads((bundle / "swath_batch_summary.json").read_text())
    manifest = json.loads((bundle / "prepared/manifest.json").read_text())
    src = Path(__file__).resolve().parent
    known = summary.get("source_code_sha256_start", {})
    if not summary.get("source_code_stable") or any(
            summary.get("source_code_sha256_end",{}).get(n)!=known.get(n) for n in FROZEN_FILES):
        raise ValueError("SWATH_SOURCE_CHANGED_DURING_GENERATION")
    changed = [name for name in FROZEN_FILES
               if known.get(name) != _sha256(src / name)]
    if changed:
        raise ValueError(f"冻结模块与输入批次源码指纹不符: {changed}")
    config = _bundle_path(bundle, manifest["config"])
    if not config.is_file() or _sha256(config) != manifest.get("config_sha256"):
        raise ValueError("冻结条带配置指纹与当前文件不符")
    source = _bundle_path(bundle, manifest["input"])
    if source.is_file() and _sha256(source) != manifest.get("source_sha256"):
        raise ValueError("冻结条带原始 GPKG 指纹不符")
    field_ids = [e["field_id"] for e in manifest.get("fields", [])]
    if not field_ids or len(set(field_ids)) != len(field_ids) or len(field_ids) != summary.get("field_count"):
        raise ValueError("manifest 与运行摘要田块数不一致")
    return summary, manifest


def _one_line(geometry: BaseGeometry) -> LineString:
    """要求一个连续LineString；不连续多线任务不能通过合并或补线冒充完整条带。"""
    if geometry.geom_type == "MultiLineString":
        if len(geometry.geoms) != 1:
            raise ValueError("冻结作业段包含不连续多段")
        geometry = geometry.geoms[0]
    if geometry.geom_type != "LineString" or geometry.is_empty or geometry.length <= 0:
        raise ValueError("冻结作业段不是有效直线")
    coords = list(geometry.coords)
    a, z = coords[0], coords[-1]
    length = math.dist(a[:2], z[:2])
    if length <= 0:
        raise ValueError("冻结作业段端点重合")
    ux, uy = (z[0] - a[0]) / length, (z[1] - a[1]) / length
    if any(abs((x-a[0])*uy - (y-a[1])*ux) > 1e-5
           for x, y, *_ in coords):
        raise ValueError("冻结作业段不是直线")
    return LineString([(a[0], a[1]), (z[0], z[1])])


def _to_local(geometry: BaseGeometry, source_crs: str, metric_crs: str,
              origin: tuple[float, float],
              transformers: dict[tuple[str, str], Transformer]) -> BaseGeometry:
    """把来源几何投影到测量CRS并减去本田原点；后续长度/面积计算使用米和平方米。"""
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    if not _crs_equiv(source_crs,metric_crs):
        key = (source_crs, metric_crs)
        if key not in transformers:
            transformers[key] = Transformer.from_crs(*key, always_xy=True)
        geometry = transform(transformers[key].transform, force_2d(geometry))
    return affinity.translate(force_2d(geometry),
                              xoff=-origin[0], yoff=-origin[1])


def _to_world(geometry: BaseGeometry, origin: tuple[float, float],
              work_crs: str, out_crs: str,
              transformers: dict[tuple[str, str], Transformer]) -> BaseGeometry:
    """加回本田原点并投影为导出显示坐标；不得用显示几何替代原始米制验算。"""
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    result = affinity.translate(geometry, xoff=origin[0], yoff=origin[1])
    if not _crs_equiv(work_crs,out_crs):
        key = (work_crs, out_crs)
        if key not in transformers:
            transformers[key] = Transformer.from_crs(*key, always_xy=True)
        result = transform(transformers[key].transform, result)
    return result


def _sweep_tolerance(area_m2: float) -> float:
    # A coordinate round-trip can move sub-millimetre polygon edges. This
    # threshold is for input serialization error, never missing body coverage.
    """按扫掠面积给出数值校验容差，单位平方米；不是允许漏作的比例目标。"""
    return max(2e-4, 1e-8 * area_m2)


def _work_motion(task: FrozenTask, scene: Scene, reverse: bool) -> Motion:
    """由冻结参考线构造遍历运动并按机具偏移调整，最终必须仍符合原始作业扫掠义务。"""
    a = np.asarray(task.reference_line.coords[0][:2], dtype=float)
    z = np.asarray(task.reference_line.coords[-1][:2], dtype=float)
    u = np.array([math.cos(task.heading_rad), math.sin(task.heading_rad)])
    if reverse:
        shift = 2.0 * scene.vehicle.implement_offset_m * u
        a, z = z + shift, a + shift
        yaw = wrap(task.heading_rad + math.pi)
    else:
        yaw = task.heading_rad
    length = float(np.linalg.norm(z-a))
    if length <= 1e-7:
        raise ValueError("零长度冻结条带")
    step = min(scene.settings.sampling_step_m, length / 3)
    t = np.asarray([0, step / length, 1-step / length, 1])
    p = np.column_stack((a[0] + t * (z[0]-a[0]),
                         a[1] + t * (z[1]-a[1]),
                         np.full(4, yaw), np.ones(4)))
    return Motion(p, "work", task.task_id, True)


def _task_variants(task: FrozenTask, scene: Scene,
                   validator: Validator) -> tuple[list[WorkVariant], list[str]]:
    """为冻结任务建立有限合法遍历方案；曲线任务缺失运动证据时拒绝，不猜测车辆朝向。"""
    if task.work_kind == "CURVED_HEADLAND":
        if task.motion_points is None:
            return [], ["CURVED_HEADLAND_MISSING_MOTION"]
        from validator import conservative_tool_coverage
        candidates, errors = [], []
        for reversed_order,points in enumerate((task.motion_points,
                                                *task.alternative_motion_points)):
            closed = len(points)>8 and np.linalg.norm(points[0,:2]-points[-1,:2])<1e-8
            core = points[:-1] if closed else points
            starts = sorted({0,len(core)//4,len(core)//2,3*len(core)//4}) if closed else [0]
            for start in starts:
                rows = np.roll(core,-start,axis=0) if closed else core
                rows = np.vstack((rows,rows[0])) if closed else rows
                motion = Motion(rows,"work",task.task_id,True)
                issues = validator.motion_issues(motion)
                if issues:
                    errors.append(issues[0]["code"])
                    continue
                sweep = conservative_tool_coverage(motion,scene)
                # Only route-generated compound headlands have alternatives.
                # Their exclusive required target is a mandatory minimum;
                # shared coverage is certified by the global common-footprint
                # guard. Extra work follows the selected physical motion. Body
                # and ordinary curved tasks keep exact sweep equivalence.
                if task.required_target_sweep is not None:
                    mismatch=task.required_target_sweep.difference(sweep).area
                    threshold=_sweep_tolerance(task.required_target_sweep.area)
                else:
                    mismatch=sweep.symmetric_difference(task.frozen_sweep).area
                    threshold=_sweep_tolerance(task.frozen_sweep.area)
                if mismatch>threshold:
                    errors.append(f"CURVED_SWEEP_CONTRACT:{mismatch:.9f}")
                    continue
                candidates.append(WorkVariant(task,bool(reversed_order),motion,sweep,mismatch))
        return candidates, errors
    choices = []
    errors = []
    for reverse in (False, True):
        motion = _work_motion(task, scene, reverse)
        issues = validator.motion_issues(motion)
        if issues:
            errors.append(("REVERSED_" if reverse else "ORIGINAL_")
                          + issues[0]["code"])
            continue
        sweep = validator.work_sweep(motion)
        mismatch = float(sweep.symmetric_difference(task.frozen_sweep).area)
        if mismatch > _sweep_tolerance(task.frozen_sweep.area):
            errors.append(("REVERSED_" if reverse else "ORIGINAL_")
                          + f"SWEEP_NOT_EQUIVALENT:{mismatch:.9f}")
            continue
        choices.append(WorkVariant(task, reverse, motion, sweep, mismatch))
    return choices, errors


# 把导出分区与条带转换成每田 FieldInput，恢复局部米制几何。
# 任务编号、所属分区、跨区覆盖依赖和坐标原点都必须保留。
def _field_rows(bundle: Path, manifest: dict[str, Any], *, field_ids=None,
                indexed_inputs=None) -> list[FieldInput]:
    """读取全部或一小片冻结任务；小片模式不得把整批几何装进内存。

    indexed_inputs 仅指向 outputs 内的只读输入副本。原封存输入不建索引、
    不改字节；完整来源检查仍由批入口完成。分片内的重复/缺失校验保留。
    """
    public, metric = (indexed_inputs if indexed_inputs else
                      (bundle / "swath_results.gpkg", bundle / "swath_results_metric.gpkg"))
    selected = set(field_ids) if field_ids is not None else None
    where = ("field_id IN (" + ",".join("'" + fid.replace("'", "''") + "'"
             for fid in sorted(selected)) + ")") if selected else None
    read_options = {"where": where} if where else {}
    entries = [e for e in manifest["fields"] if selected is None or e["field_id"] in selected]
    if selected is not None and (not selected or {e["field_id"] for e in entries} != selected):
        raise ValueError("UNKNOWN_FIELD_ID")
    layers = {}
    for name in ("source_fields", "work_regions", "headland_reserve",
                 "swath_segments"):
        layers[name] = gpd.read_file(public, layer=name, **read_options)
    metric_rows: dict[tuple[str, str, str], tuple[BaseGeometry, str]] = {}
    required_rows: dict[str, list[tuple[BaseGeometry,str]]] = {}
    for layer in gpd.list_layers(metric)["name"]:
        if layer.startswith("required_main_areas_"):
            required = gpd.read_file(metric, layer=layer, **read_options)
            for row in required.itertuples():
                required_rows.setdefault(row.field_id, []).append((row.geometry, required.crs.to_string()))
            continue
        if not layer.startswith("work_sweeps_"):
            continue
        frame = gpd.read_file(metric, layer=layer, **read_options)
        crs = frame.crs.to_string()
        for row in frame.itertuples():
            key = (row.field_id, row.region_id, row.task_id)
            if key in metric_rows:
                raise ValueError(f"米制扫掠任务重复: {key}")
            metric_rows[key] = (row.geometry, crs)
    source = {row.field_id: row for row in layers["source_fields"].itertuples()}
    if len(source) != len(layers["source_fields"]) or set(source) != {e["field_id"] for e in entries}:
        raise ValueError("SOURCE_FIELD_SET_MISMATCH")
    region_rows: dict[str, list[Any]] = {}
    headland_rows: dict[str, list[Any]] = {}
    segment_rows: dict[str, list[Any]] = {}
    for key, container in (("work_regions", region_rows),
                           ("headland_reserve", headland_rows),
                           ("swath_segments", segment_rows)):
        for row in layers[key].itertuples():
            container.setdefault(row.field_id, []).append(row)
    transformers: dict[tuple[str, str], Transformer] = {}
    all_fields = []
    seen_tasks: set[tuple[str, str, str]] = set()
    for entry in entries:
        field_id = entry["field_id"]
        if field_id not in source or field_id not in region_rows:
            raise ValueError(f"冻结田块或分区缺失: {field_id}")
        scene_path = _bundle_path(bundle, entry["scene_path"])
        if not scene_path.is_file():
            raise ValueError(f"Scene 快照缺失: {field_id}")
        expected_crs = entry["work_crs"]
        origins = {(float(r.local_origin_x), float(r.local_origin_y))
                   for r in region_rows[field_id]}
        if len(origins) != 1:
            raise ValueError(f"分区局部原点不一致: {field_id}")
        origin = next(iter(origins))
        if any(r.measurement_crs != expected_crs for r in region_rows[field_id]):
            raise ValueError(f"分区米制 CRS 不一致: {field_id}")
        regions = {}
        for row in region_rows[field_id]:
            if row.region_id in regions:
                raise ValueError(f"重复分区编号: {field_id}/{row.region_id}")
            regions[row.region_id] = RegionInput(
                row.region_id, int(row.sequence_index),
                _to_local(row.geometry, str(layers["work_regions"].crs),
                          expected_crs, origin, transformers),
                GeometryCollection(), ())
        for row in headland_rows.get(field_id, []):
            if row.region_id not in regions:
                raise ValueError(f"田头引用未知分区: {field_id}/{row.region_id}")
            reg = regions[row.region_id]
            h = _to_local(row.geometry, str(layers["headland_reserve"].crs),
                          expected_crs, origin, transformers)
            regions[row.region_id] = RegionInput(
                reg.region_id, reg.sequence_index, reg.geometry, h, reg.task_ids)
        tasks = []
        by_region: dict[str, list[str]] = {name: [] for name in regions}
        for row in segment_rows.get(field_id, []):
            key = (field_id, row.region_id, row.task_id)
            if key in seen_tasks or row.region_id not in regions:
                raise ValueError(f"重复任务或未知分区: {key}")
            seen_tasks.add(key)
            if key not in metric_rows or not _crs_equiv(metric_rows[key][1],expected_crs):
                raise ValueError(f"任务米制扫掠缺失/CRS 错误: {key}")
            geometry = _to_local(
                row.geometry, str(layers["swath_segments"].crs),
                expected_crs, origin, transformers)
            line = _one_line(geometry)
            ux, uy = math.cos(math.radians(row.heading_deg)), math.sin(math.radians(row.heading_deg))
            a, z = line.coords[0], line.coords[-1]
            if (z[0]-a[0])*ux + (z[1]-a[1])*uy < 0:
                line = LineString([z, a])
            length = float(line.length)
            if abs(length - float(row.length_m)) > 1e-3:
                raise ValueError(f"冻结任务长度不一致: {key}")
            base_sweep = _to_local(metric_rows[key][0], expected_crs,
                                   expected_crs, origin, transformers)
            tasks.append(FrozenTask(
                field_id, row.region_id, row.task_id, int(row.row_index),
                int(row.suggested_order), line, math.radians(float(row.heading_deg)),
                base_sweep, length))
            by_region[row.region_id].append(row.task_id)
        for name, reg in list(regions.items()):
            regions[name] = RegionInput(
                reg.region_id, reg.sequence_index, reg.geometry, reg.headland,
                tuple(by_region[name]))
        all_fields.append(FieldInput(
            int(entry["feature_index"]), field_id, str(scene_path),
            source[field_id].geometry, str(layers["source_fields"].crs),
            expected_crs, origin,
            tuple(sorted(regions.values(), key=lambda r: (r.sequence_index,r.region_id))),
            tuple(tasks),
            str(source[field_id].seam_quality_status),
            bool(source[field_id].acceptance_passed),
            float(source[field_id].area_ledger_delta_m2)
            if source[field_id].area_ledger_delta_m2 is not None else None,
            unary_union([_to_local(g,c,expected_crs,origin,transformers) for g,c in required_rows.get(field_id,[])])
            if field_id in required_rows else None))
    if len(seen_tasks) != len(metric_rows):
        missing = set(metric_rows) - seen_tasks
        raise ValueError(f"米制扫掠缺少原任务引用: {len(missing)}")
    return all_fields

def _pose_at(motion: Motion, last: bool) -> Pose:
    """读取运动起点或终点，返回xy米、yaw弧度的Pose。"""
    p = motion.points[-1 if last else 0]
    return Pose(float(p[0]), float(p[1]), float(p[2]))


def _pose_close(left: Pose, right: Pose, scene: Scene) -> bool:
    """按场景位置和朝向容差比较位姿，同坐标不同方向不能算无动作衔接。"""
    return (math.hypot(left.x-right.x, left.y-right.y)
            <= scene.settings.join_tolerance_m and
            abs(wrap(left.yaw-right.yaw))
            <= scene.settings.heading_tolerance_rad)


def _ready_area(scene: Scene, headlands: BaseGeometry,
                completed_work: BaseGeometry) -> BaseGeometry:
    """按当前已作业区、田头及合法田外通行区建立连接候选空间，不借用未完成未来任务。"""
    outside_target = scene.travel.difference(scene.target)
    parts = [g for g in (headlands, completed_work, outside_target)
             if g is not None and not g.is_empty]
    return unary_union(parts) if parts else GeometryCollection()


def _gear_and_steering(motion: Motion, scene: Scene,
                       settings: RouteSettings) -> tuple[int, int, float, list[dict[str, Any]], float]:
    """从明确的运动挡位和朝向提取换挡/停转事件及估计时间，不凭条带排序反向捏造倒挡。"""
    p = motion.points
    directions = [int(v) for v in p[:-1, 3]]
    if not directions:
        return 0, 0, 0.0, [], 0.0
    sequence = [1, *directions, 1]  # adjacent work on both sides is forward
    shifts = sum(a != b for a, b in zip(sequence, sequence[1:]))
    reverse_legs = sum(v == -1 and sequence[i-1] != -1
                       for i, v in enumerate(directions, start=1))
    reverse_m = sum(float(np.linalg.norm(p[i+1, :2]-p[i, :2]))
                    for i, d in enumerate(directions) if d < 0)
    if shifts > settings.max_gear_shifts or reverse_legs > settings.max_reverse_legs:
        return reverse_legs, shifts, reverse_m, [], math.inf

    distance = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    dyaw = (np.diff(p[:, 2]) + math.pi) % (2*math.pi) - math.pi
    kappa = np.divide(dyaw, distance * p[:-1, 3],
                      out=np.zeros_like(distance), where=distance > 1e-8)
    rates = [0.0, *[float(x) for x in kappa], 0.0]
    events: list[dict[str, Any]] = []
    steer_time = 0.0
    for i in range(len(rates)-1):
        # Curvature can change at a stop at either work/turn boundary, or
        # at a genuine gear cusp. Same-gear moving changes remain subject
        # to Validator.motion_issues and do not receive a free stop event.
        is_cusp = (i == 0 or i == len(rates)-2
                   or (i <= len(directions)-1
                       and directions[i-1] != directions[i]))
        if is_cusp and abs(rates[i+1]-rates[i]) > 1e-3:
            pose_index = min(i, len(p)-1)
            events.append({"kind": "STOP_STEER", "x_m": float(p[pose_index,0]),
                           "y_m": float(p[pose_index,1]),
                           "yaw_rad": float(p[pose_index,2]),
                           "duration_s": settings.stop_steer_seconds,
                           "duration_source": "UNMEASURED_PLANNING_ESTIMATE"})
            steer_time += settings.stop_steer_seconds
    for i, (a, b) in enumerate(zip(sequence, sequence[1:])):
        if a != b:
            j = max(0, min(i, len(p)-1))
            events.append({"kind": "GEAR_CHANGE", "x_m": float(p[j,0]),
                           "y_m": float(p[j,1]), "yaw_rad": float(p[j,2]),
                           "from": a, "to": b,
                           "duration_s": scene.vehicle.gear_change_seconds,
                           "duration_source": "CONFIG_NOMINAL"})
    return reverse_legs, shifts, reverse_m, events, steer_time


def _direct_motion(a: Pose, b: Pose, scene: Scene) -> Motion | None:
    """为正长度直线连接生成运动；零长度端点返回None，仍由上层处理朝向差。"""
    length = math.hypot(b.x-a.x, b.y-a.y)
    if length < 1e-8:
        return None
    yaw = math.atan2(b.y-a.y, b.x-a.x)
    if (abs(wrap(yaw-a.yaw)) > scene.settings.heading_tolerance_rad
            or abs(wrap(yaw-b.yaw)) > scene.settings.heading_tolerance_rad):
        return None
    return Motion(np.array([[a.x,a.y,a.yaw,1], [b.x,b.y,b.yaw,1]], dtype=float),
                  "transit", implement_on=False)


def _route_continuity_issues(motions: list[Motion],scene: Scene) -> list[str]:
    """逐段检查路线接头位姿连续性，空路线和位置/朝向失配都保留问题。"""
    if not motions:
        return ["EMPTY_ROUTE"]
    issues=[]
    for a,b in zip(motions,motions[1:]):
        if not _pose_close(_pose_at(a,True),_pose_at(b,False),scene):
            issues.append("DISCONNECTED_OR_HEADING_JUMP")
    return issues


def _field_error(job: FieldInput, code: str, message: str,
                 elapsed: float) -> FieldRoute:
    """构造可对账的单田失败结果，并将分区任务数量及错误原因保留下游。"""
    failures = [{"field_id":job.field_id,"region_id":reg.region_id,
                 "code":code,"detail":message,
                 "task_count":len(reg.task_ids)}
                for reg in job.regions]
    return FieldRoute(
        job.index,job.field_id,"INPUT_ERROR",[],
        [RegionRoute(reg.region_id,"INPUT_ERROR",reason=code+":"+message)
         for reg in job.regions],
        failures,elapsed,job.source_crs,job.metric_crs,job.origin,
        job.source_geometry,job.upstream_seam_quality_status,
        job.upstream_acceptance_passed,job.upstream_area_ledger_delta_m2,
        error=code+":"+message)


def _write_csv(path: Path, rows: list[dict[str,Any]],
               columns: tuple[str,...]) -> None:
    """按固定字段写UTF-8 CSV镜像；权威作业与时间账本以对应协议GPKG为准。"""
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_layer(path: Path, name: str, rows: list[dict[str,Any]],
                 crs: str, columns: tuple[str,...]) -> None:
    """写历史结果空间图层，空结果保留图层结构，不能把空图层理解为阶段通过。"""
    if rows:
        frame=gpd.GeoDataFrame(rows,geometry="geometry",crs=crs)
    else:
        import pandas as pd
        data={key:pd.Series(dtype="object") for key in columns if key!="geometry"}
        frame=gpd.GeoDataFrame(data,geometry=gpd.GeoSeries([],dtype="geometry"),crs=crs)
    frame.to_file(path,layer=name,driver="GPKG",engine="pyogrio")


def _motion_parts(motion: Motion) -> list[tuple[int,int,LineString]]:
    """按真实运动状态拆分可导出的线段，保留原始段位置与挡位，不用显示折线补运动断点。
    
    Split a connection whenever the driving gear changes."""
    p=motion.points
    segments=[]
    begin=0
    for i in range(1,len(p)-1):
        if int(p[i,3])!=int(p[i-1,3]):
            if i>begin:
                segments.append((begin,int(p[begin,3]),
                                 LineString(p[begin:i+1,:2])))
            begin=i
    if len(p)-1>begin:
        segments.append((begin,int(p[begin,3]),LineString(p[begin:,:2])))
    return segments


def plan_field(job: FieldInput, settings: RouteSettings) -> FieldRoute:
    """总调度调用新版真实跨区连接器；旧输入/类型接口保持可读。"""
    from route import plan_field as solve
    return solve(job, settings)


def _needs_low_contention_route(job: FieldInput) -> bool:
    """根据田形与任务规模识别历史严格搜索的资源争用敏感场景，仅用于决定是否重试。
    
    Keep bounded block searches responsive on large multi-region fields."""
    geometry=job.source_geometry
    polygons=geometry.geoms if geometry.geom_type=='MultiPolygon' else [geometry]
    hole_count=sum(len(polygon.interiors) for polygon in polygons)
    largest=max((len(region.task_ids) for region in job.regions),default=0)
    return (len(job.tasks)>=100 and
            ((hole_count>=3 and len(job.regions)>=3)
             or (len(job.regions)>=2 and largest>=40)))


def _needs_fragmented_headland_retry(job: FieldInput, result: FieldRoute,
                                     settings: RouteSettings) -> bool:
    """判断历史田头碎片失败是否符合有限追加搜索条件，不无限重跑全批。
    
    Detect a dense boundary whose generated headland work mostly stalled."""
    if (not settings.plan_headland_work
            or not settings.adaptive_headland_simplification
            or not 50<=len(job.tasks)<100
            or result.status!='JOINT_TASKS_PARTIAL'
            or result.preparation.get(
                'effective_headland_simplification_tolerance_m',1.0)>=1.0):
        return False
    polygons=(job.source_geometry.geoms
              if job.source_geometry.geom_type=='MultiPolygon'
              else [job.source_geometry])
    if sum(len(polygon.exterior.coords) for polygon in polygons)<100:
        return False
    stage=result.statistics.get('headland_stage',{})
    finished=stage.get('completed_task_count',0)
    remaining=stage.get('remaining_task_count',0)
    return finished+remaining>=200 and 2*finished<finished+remaining


def _retry_headland_entry_rank(jobs, results, settings, workers, out):
    """对符合条件的联合田头方案追加入口排序尝试，原始失败和重试证据保留。
    
    Bounded alternative entry ordering; retain both coverage and connectivity.
    
    A failed greedy order is not a physical impossibility certificate. Retry
    only the three largest substantial partial cases, at their already chosen
    geometry tolerance. Complete fields keep their original ordering."""
    if not (settings.plan_headland_work and settings.headland_work_mode == 'JOINT'
            and settings.adaptive_headland_entry_rank
            and settings.headland_entry_rank == "MEAN"):
        return results, []
    job_by_id = {job.field_id: job for job in jobs}
    candidates = []
    for result in results:
        stage = result.statistics.get("headland_stage", {})
        remaining = stage.get("remaining_task_count", 0)
        total = remaining + stage.get("completed_task_count", 0)
        gap = result.preparation.get("original_target_missing_m2", 0.0)
        if (result.status == "JOINT_TASKS_PARTIAL" and remaining >= 20
                and remaining / max(1, total) >= 0.25 and gap >= 2000):
            candidates.append(result)
    candidates.sort(key=lambda r: (
        -r.preparation["original_target_missing_m2"] *
        r.statistics["headland_stage"]["remaining_task_count"] ** 0.5,
        r.field_id))
    chosen = candidates[:3]
    selected = {result.field_id: result for result in results}
    groups = {}
    for prior in chosen:
        tolerance = prior.preparation.get(
            "effective_headland_simplification_tolerance_m",
            settings.headland_simplification_tolerance_m)
        groups.setdefault(tolerance, []).append(job_by_id[prior.field_id])
    records = []
    for tolerance, retry_jobs in sorted(groups.items()):
        retry_settings = replace(settings, headland_entry_rank="MIN",
            adaptive_headland_entry_rank=False,
            headland_simplification_tolerance_m=tolerance,
            adaptive_headland_simplification=False)
        if workers == 1:
            retries = [plan_field(job, retry_settings) for job in retry_jobs]
        else:
            pass  # 已合并到本模块，直接使用下方的定义。
            retry_dir = out / "headland_entry_rank_retry" / f"tol_{tolerance:.3f}"
            retry_dir.mkdir(parents=True)
            retries = list(solve_parallel(retry_jobs, retry_settings,
                                          min(2, workers), retry_dir))
        for candidate in retries:
            prior = selected[candidate.field_id]
            scene = load_scene(job_by_id[candidate.field_id].scene_path)
            def quality(result):
                sweeps = [part.completed_sweep for part in result.routes
                          if part.status == "REGION_ROUTE_PASS"]
                worked = unary_union(sweeps) if sweeps else GeometryCollection()
                body_gap = (result.required_body.difference(worked).area
                            if result.required_body is not None else float("inf"))
                return (body_gap, scene.target.difference(worked).area,
                    result.statistics.get("headland_stage", {}).get(
                        "remaining_task_count", float("inf")))
            before, after = quality(prior), quality(candidate)
            tol = scene.settings.coverage_tolerance_m2
            valid_status = candidate.status in {"JOINT_TASKS_PARTIAL",
                "JOINT_TASK_CHAIN_TARGET_GAP", "FULL_FIELD_ROUTE_COMPLETE"}
            improved = (valid_status and after[0] <= tol
                and after[1] <= before[1] + tol and after[2] <= before[2]
                and (after[1] < before[1] - tol or after[2] < before[2]))
            record = {"field_id": candidate.field_id,
                "selected": "RETRY" if improved else "INITIAL",
                "retry_entry_rank": "MIN", "tolerance_m": tolerance,
                "initial_body_missing_m2": before[0],
                "retry_body_missing_m2": after[0],
                "initial_target_missing_m2": before[1],
                "retry_target_missing_m2": after[1],
                "initial_remaining_task_count": before[2],
                "retry_remaining_task_count": after[2],
                "retry_elapsed_s": candidate.elapsed_s}
            elapsed = prior.elapsed_s + candidate.elapsed_s
            winner = candidate if improved else prior
            winner.elapsed_s = elapsed
            winner.statistics["headland_entry_rank_retry"] = record
            selected[candidate.field_id] = winner
            records.append(record)
            print(f"[entry rank retry] {candidate.field_id}: "
                  f"{record['selected']}, gap {before[1]:.2f}->{after[1]:.2f}, "
                  f"remaining {before[2]}->{after[2]}", flush=True)
    return [selected[result.field_id] for result in results], records


def _retry_fragmented_headlands(jobs, results, settings, workers, out):
    """对碎片化田头候选进行有界复算，不修改来源几何或覆盖义务。
    
    Try coarse headland geometry in a separate, bounded worker phase.
    
    A baseline route plus its full retry can exceed the ordinary worker's
    hard timeout. Preserve the baseline first, then test at most three dense
    fields with one independent 1 m solve each. Compare actual executed work,
    never generated task count, before replacing the original route."""
    job_by_id={job.field_id:job for job in jobs}
    candidates=[result for result in results if
        _needs_fragmented_headland_retry(job_by_id[result.field_id],
                                         result,settings)]
    candidates.sort(key=lambda result:(
        -result.preparation.get('original_target_missing_m2',0.0),
        result.field_id))
    chosen=candidates[:3]
    if not chosen:
        return results,[]
    retry_settings=replace(settings,
        headland_simplification_tolerance_m=1.0,
        adaptive_headland_simplification=False)
    retry_jobs=[job_by_id[result.field_id] for result in chosen]
    if workers==1:
        retries=[plan_field(job,retry_settings) for job in retry_jobs]
    else:
        pass  # 已合并到本模块，直接使用下方的定义。
        retry_dir=out/'fragmented_headland_retry'
        retry_dir.mkdir()
        retries=list(solve_parallel(retry_jobs,retry_settings,
                                    min(2,workers),retry_dir))
    selected={result.field_id:result for result in results}
    records=[]
    for candidate in retries:
        prior=selected[candidate.field_id]
        scene=load_scene(job_by_id[candidate.field_id].scene_path)
        def quality(route):
            sweeps=[part.completed_sweep for part in route.routes
                    if part.status=='REGION_ROUTE_PASS']
            worked=unary_union(sweeps) if sweeps else GeometryCollection()
            body=route.required_body
            body_complete=(body is not None and
                body.difference(worked).area<=scene.settings.coverage_tolerance_m2)
            return body_complete,worked.intersection(scene.target).area
        before,after=quality(prior),quality(candidate)
        improved=(after[0] and not before[0]) or (
            after[0]==before[0] and
            after[1]>before[1]+scene.settings.coverage_tolerance_m2)
        stage=prior.statistics['headland_stage']
        record={
            'field_id':candidate.field_id,
            'selected':'RETRY' if improved else 'INITIAL',
            'initial_tolerance_m':prior.preparation[
                'effective_headland_simplification_tolerance_m'],
            'retry_tolerance_m':1.0,
            'headland_task_count':stage['completed_task_count']+
                                  stage['remaining_task_count'],
            'initial_headland_completed_task_count':stage['completed_task_count'],
            'initial_body_complete':before[0],
            'retry_body_complete':after[0],
            'initial_target_covered_m2':before[1],
            'retry_target_covered_m2':after[1],
            'initial_elapsed_s':prior.elapsed_s,
            'retry_elapsed_s':candidate.elapsed_s}
        winner=candidate if improved else prior
        winner.elapsed_s=prior.elapsed_s+candidate.elapsed_s
        winner.statistics['headland_simplification_retry']=record
        selected[candidate.field_id]=winner
        records.append(record)
        print(f"[headland retry] {candidate.field_id}: "
              f"selected={record['selected']}, {record['retry_elapsed_s']:.2f}s",
              flush=True)
    return [selected[result.field_id] for result in results],records


def _retry_near_complete_body(jobs, results, settings, workers, out):
    """为接近完成的历史主体路线分配有限重试，不能以更改状态代替找到连接。
    
    Recheck only promising incomplete fields with two competing workers.
    
    A twelve-worker wall-clock budget can expire before a near-complete field
    reaches its local block repair. The first pass stays fast; at most three
    fields receive a lower-contention second pass. Keep the first route unless
    the retry covers more of the original target without losing body status."""
    if (not settings.retry_near_complete or settings.plan_headland_work
            or workers <= 2 or len(jobs) <= 1):
        return results,[]
    job_by_id={job.field_id:job for job in jobs}
    candidates=[]
    for result in results:
        if result.status!='PARTIAL_CONTINUOUS_ROUTE' or result.adapted_job is None:
            continue
        total=len(result.adapted_job.tasks)
        done=sum(len(route.task_order) for route in result.routes
                 if route.status=='REGION_ROUTE_PASS')
        ratio=done/max(1,total)
        splits=result.statistics.get('adaptive_split_count',0)
        if ratio>=0.5 and (0<splits<=3 or ratio>=0.7):
            # Near-complete fields come first. Among the others, a small
            # number of local splits is stronger evidence for an exit-order
            # bottleneck than many splits with no complete continuation.
            tier=0 if ratio>=0.7 else 1
            candidates.append((tier,splits,ratio,result.field_id))
    chosen_ids=[field_id for *_,field_id in sorted(candidates,
                key=lambda item:(item[0],-item[2] if item[0]==0 else item[1],
                                 -item[2],item[3]))[:3]]
    if not chosen_ids:
        return results,[]
    retry_dir=out/'low_concurrency_retry'
    retry_dir.mkdir()
    original={result.field_id:result for result in results}
    records=[]
    pass  # 已合并到本模块，直接使用下方的定义。
    retry_jobs=[job_by_id[field_id] for field_id in chosen_ids]
    for candidate in solve_parallel(retry_jobs,settings,2,retry_dir):
        prior=original[candidate.field_id]
        scene=load_scene(job_by_id[candidate.field_id].scene_path)
        def covered(route):
            sweeps=[part.completed_sweep for part in route.routes
                    if part.status=='REGION_ROUTE_PASS']
            return (unary_union(sweeps).intersection(scene.target).area
                    if sweeps else 0.0)
        old_area,new_area=covered(prior),covered(candidate)
        tolerance=scene.settings.coverage_tolerance_m2
        improved=(candidate.status=='FIELD_ROUTE_COMPLETE'
                  and prior.status!='FIELD_ROUTE_COMPLETE'
                  and new_area>=old_area-tolerance) or (
                      new_area>old_area+tolerance
                      and candidate.status in {'FIELD_ROUTE_COMPLETE',
                                               'PARTIAL_CONTINUOUS_ROUTE'})
        initial_elapsed=prior.elapsed_s
        retry_elapsed=candidate.elapsed_s
        selected=candidate if improved else prior
        selected.elapsed_s=initial_elapsed+retry_elapsed
        record={'field_id':candidate.field_id,
                'initial_status':prior.status,'retry_status':candidate.status,
                'selected':'RETRY' if improved else 'INITIAL',
                'initial_target_covered_m2':old_area,
                'retry_target_covered_m2':new_area,
                'initial_elapsed_s':initial_elapsed,
                'retry_elapsed_s':retry_elapsed}
        selected.statistics['low_concurrency_retry']=record
        original[candidate.field_id]=selected
        records.append(record)
        print(f"[retry] {candidate.field_id}: {candidate.status}, "
              f"selected={record['selected']}, {retry_elapsed:.2f}s",
              flush=True)
    return [original[result.field_id] for result in results],records


def _retry_contention_sensitive_joint(jobs, results, settings, low_ids):
    """对并行争用敏感的联合规划追加隔离尝试，保留本次实际预算和结果。
    
    Recheck a few body-incomplete large fields alone before export.
    
    A two-worker wall-clock budget can discard an otherwise connectable long
    body chain. The retry never replaces a safer or more complete first
    attempt; both attempts and their elapsed time remain in the audit record."""
    if not settings.plan_headland_work:
        return results, []
    job_by_id={job.field_id:job for job in jobs}
    selected={result.field_id:result for result in results}
    def directional_block_incomplete(result):
        if (result.status!='PARTIAL_CONTINUOUS_ROUTE'
                or result.required_body is None):
            return False
        attempts=getattr(result,'preparation',{}).get('headland_attempts',[])
        if not any(item.get('mode')=='DIRECTIONAL'
                   and item.get('search_variant')=='MULTI_REGION_ROW_BLOCKS'
                   for item in attempts):
            return False
        sweeps=[route.completed_sweep for route in result.routes
                if route.status=='REGION_ROUTE_PASS']
        worked=unary_union(sweeps) if sweeps else GeometryCollection()
        return result.required_body.difference(worked).area>max(
            1.0,0.01*result.required_body.area)
    candidates=[result for result in results
                if result.status=='SEARCH_LIMIT' or (
                    result.field_id in low_ids and result.status in {
                        'PARTIAL_CONTINUOUS_ROUTE','NO_CONTINUOUS_ROUTE'})
                or directional_block_incomplete(result)]
    candidates.sort(key=lambda result:(
        0 if result.status=='SEARCH_LIMIT' else
        1 if directional_block_incomplete(result) else 2,
        -len(job_by_id[result.field_id].tasks),result.field_id))
    chosen=candidates[:2]
    # Two unrelated worker hard timeouts can occupy both ordinary rechecks.
    # Reserve one additional, evidence-triggered slot for a directional block
    # attempt that left a material body gap under contention. No other partial
    # field is added to this expensive singleton phase.
    directional=next((result for result in candidates[2:]
                      if directional_block_incomplete(result)),None)
    if directional is not None:
        chosen.append(directional)
    records=[]
    for prior in chosen:
        job=job_by_id[prior.field_id]
        scene=load_scene(job.scene_path)
        retry=plan_field(job,settings)
        def quality(result):
            sweeps=[route.completed_sweep for route in result.routes
                    if route.status=='REGION_ROUTE_PASS']
            worked=unary_union(sweeps) if sweeps else GeometryCollection()
            body=result.required_body
            body_complete=(body is not None and
                body.difference(worked).area<=scene.settings.coverage_tolerance_m2)
            target_covered=worked.intersection(scene.target).area
            return (result.status=='FULL_FIELD_ROUTE_COMPLETE',
                    body_complete,target_covered)
        initial_quality,retry_quality=quality(prior),quality(retry)
        improved=retry_quality>initial_quality
        record={'field_id':prior.field_id,
                'initial_status':prior.status,'retry_status':retry.status,
                'selected':'RETRY' if improved else 'INITIAL',
                'initial_body_complete':initial_quality[1],
                'retry_body_complete':retry_quality[1],
                'initial_target_covered_m2':initial_quality[2],
                'retry_target_covered_m2':retry_quality[2],
                'initial_elapsed_s':prior.elapsed_s,
                'retry_elapsed_s':retry.elapsed_s}
        winner=retry if improved else prior
        winner.elapsed_s=prior.elapsed_s+retry.elapsed_s
        winner.statistics['singleton_recheck']=record
        selected[prior.field_id]=winner
        records.append(record)
        print(f"[singleton recheck] {prior.field_id}: {retry.status}, "
              f"selected={record['selected']}, {record['retry_elapsed_s']:.2f}s",
              flush=True)
    return [selected[result.field_id] for result in results],records


def _retry_partial_block_fields(jobs, results, settings):
    """对符合条件的部分区块路线追加有界搜索，所有原任务仍须保留。
    
    Recheck at most three budget-limited two-region block routes alone.
    
    The same bounded block search can finish in isolation but stop early when
    twelve fields contend for CPU. Preserve each safe parallel result unless
    its executed body gap shrinks without reducing original-target coverage.
    This is a search-status retry, not evidence of physical infeasibility."""
    if not settings.plan_headland_work or len(jobs)<=1:
        return results,[]
    job_by_id={job.field_id:job for job in jobs}
    candidates=[]
    for result in results:
        job=job_by_id[result.field_id]
        if (result.status!='PARTIAL_CONTINUOUS_ROUTE'
                or result.statistics.get('search_variant')!='MULTI_REGION_ROW_BLOCKS'
                or len(job.regions)!=2 or not 50<=len(job.tasks)<100
                or result.required_body is None):
            continue
        sweeps=[part.completed_sweep for part in result.routes
                if part.status=='REGION_ROUTE_PASS']
        worked=unary_union(sweeps) if sweeps else GeometryCollection()
        gap=result.required_body.difference(worked).area
        if gap>=100.0:
            candidates.append((gap,result.field_id))
    # A nearly complete body has the strongest chance of becoming a complete
    # task chain. Preserve a hard cap so this phase cannot grow with batch size.
    candidates.sort()
    chosen_ids=[field_id for _,field_id in candidates[:3]]
    selected={result.field_id:result for result in results}
    records=[]
    for field_id in chosen_ids:
        prior=selected[field_id]
        job=job_by_id[field_id]
        scene=load_scene(job.scene_path)
        retry=plan_field(job,settings)
        def quality(route):
            sweeps=[part.completed_sweep for part in route.routes
                    if part.status=='REGION_ROUTE_PASS']
            worked=unary_union(sweeps) if sweeps else GeometryCollection()
            body_gap=(route.required_body.difference(worked).area
                      if route.required_body is not None else math.inf)
            return body_gap,worked.intersection(scene.target).area
        before,after=quality(prior),quality(retry)
        tolerance=scene.settings.coverage_tolerance_m2
        improved=(after[0]<=before[0]+tolerance
                  and after[1]>=before[1]-tolerance
                  and (after[0]<before[0]-tolerance
                       or after[1]>before[1]+tolerance))
        record={'field_id':field_id,
                'selected':'RETRY' if improved else 'INITIAL',
                'initial_status':prior.status,'retry_status':retry.status,
                'initial_body_gap_m2':before[0],
                'retry_body_gap_m2':after[0],
                'initial_target_covered_m2':before[1],
                'retry_target_covered_m2':after[1],
                'initial_elapsed_s':prior.elapsed_s,
                'retry_elapsed_s':retry.elapsed_s}
        winner=retry if improved else prior
        winner.elapsed_s=prior.elapsed_s+retry.elapsed_s
        winner.statistics['block_contention_retry']=record
        selected[field_id]=winner
        records.append(record)
        print(f"[block recheck] {field_id}: {retry.status}, "
              f"selected={record['selected']}, {record['retry_elapsed_s']:.2f}s",
              flush=True)
    return [selected[result.field_id] for result in results],records


def _retry_single_region_budget(jobs, results, settings):
    """对单区预算受限的历史路线追加有限搜索，不把首次超时当物理不可达。
    
    Give a bounded solo budget to incomplete one-region block searches.
    
    A timed block search may report either budget exhaustion or a failed link
    after its bounded candidate list.  Retry the latter only when at least two
    body tasks remain; the one-row case needs a different local replan below.
    Keep the original route unless both executed-body and original-target
    coverage are non-decreasing, with at least one material improvement."""
    if (not settings.plan_headland_work or len(jobs)<=1
            or settings.max_field_seconds>=40.0):
        return results,[]
    job_by_id={job.field_id:job for job in jobs}
    candidates=[]
    for result in results:
        job=job_by_id[result.field_id]
        adapted=result.adapted_job or job
        completed=sum(len(part.task_order) for part in result.routes
                      if part.status=='REGION_ROUTE_PASS')
        unresolved=len(adapted.tasks)-completed
        budget_limited=any('SEARCH_BUDGET_EXHAUSTED' in
                           str(failure.get('code',''))
                           for failure in result.failures)
        failed_link=any(str(failure.get('code','')).startswith(
            'NO_LEGAL_CONNECTION_TO:') for failure in result.failures)
        if (result.status!='PARTIAL_CONTINUOUS_ROUTE'
                or result.statistics.get('search_variant')!='MULTI_REGION_ROW_BLOCKS'
                or len(job.regions)!=1 or not 50<=len(job.tasks)<120
                or result.required_body is None
                or not (budget_limited or failed_link and unresolved>=2)):
            continue
        sweeps=[part.completed_sweep for part in result.routes
                if part.status=='REGION_ROUTE_PASS']
        worked=unary_union(sweeps) if sweeps else GeometryCollection()
        gap=result.required_body.difference(worked).area
        if gap>=100.0:
            candidates.append((gap,result.field_id))
    # Try near-complete bodies first, but never let this phase grow with the
    # number of fields in a large GeoPackage.
    chosen_ids=[field_id for _,field_id in sorted(candidates)[:2]]
    selected={result.field_id:result for result in results}
    retry_settings=replace(settings,max_field_seconds=40.0)
    records=[]
    for field_id in chosen_ids:
        prior=selected[field_id]
        job=job_by_id[field_id]
        scene=load_scene(job.scene_path)
        retry=plan_field(job,retry_settings)
        def quality(route):
            sweeps=[part.completed_sweep for part in route.routes
                    if part.status=='REGION_ROUTE_PASS']
            worked=unary_union(sweeps) if sweeps else GeometryCollection()
            body_gap=(route.required_body.difference(worked).area
                      if route.required_body is not None else math.inf)
            return body_gap,worked.intersection(scene.target).area
        before,after=quality(prior),quality(retry)
        tolerance=scene.settings.coverage_tolerance_m2
        improved=(after[0]<=before[0]+tolerance
                  and after[1]>=before[1]-tolerance
                  and (after[0]<before[0]-tolerance
                       or after[1]>before[1]+tolerance))
        record={'field_id':field_id,
                'selected':'RETRY' if improved else 'INITIAL',
                'initial_status':prior.status,'retry_status':retry.status,
                'initial_body_gap_m2':before[0],
                'retry_body_gap_m2':after[0],
                'initial_target_covered_m2':before[1],
                'retry_target_covered_m2':after[1],
                'initial_elapsed_s':prior.elapsed_s,
                'retry_elapsed_s':retry.elapsed_s,
                'retry_max_field_seconds':retry_settings.max_field_seconds}
        winner=retry if improved else prior
        winner.elapsed_s=prior.elapsed_s+retry.elapsed_s
        winner.statistics['single_region_budget_retry']=record
        selected[field_id]=winner
        records.append(record)
        print(f"[single-region recheck] {field_id}: {retry.status}, "
              f"selected={record['selected']}, {record['retry_elapsed_s']:.2f}s",
              flush=True)
    return [selected[result.field_id] for result in results],records


def _retry_last_body_connection(jobs, results, settings):
    """对末条主体接入失败追加有限预算并重新验证，不能只画末端连线。
    
    Replan one stranded final body row with its following block.
    
    A longer local window is costly, so reserve it for a nearly finished
    single-region body with exactly one unexecuted task.  This is a connection
    search, not a relaxation of the worked-area or envelope rules."""
    if (not settings.plan_headland_work or len(jobs)<=1
            or settings.last_body_replan_seconds>=30.0):
        return results,[]
    job_by_id={job.field_id:job for job in jobs}
    candidates=[]
    for result in results:
        job=job_by_id[result.field_id]
        adapted=result.adapted_job or job
        completed=sum(len(route.task_order) for route in result.routes
                      if route.status=='REGION_ROUTE_PASS')
        if (result.status!='PARTIAL_CONTINUOUS_ROUTE'
                or len(job.regions)!=1 or len(job.tasks)<50
                or result.required_body is None
                or len(adapted.tasks)-completed!=1
                or not any(str(failure.get('code','')).startswith(
                    'NO_LEGAL_CONNECTION_TO:') for failure in result.failures)):
            continue
        worked=unary_union([route.completed_sweep for route in result.routes
                            if route.status=='REGION_ROUTE_PASS'])
        gap=result.required_body.difference(worked).area
        if 50.0<=gap<=500.0:
            candidates.append((gap,result.field_id))
    # At most one expensive local replan per batch, chosen closest to done.
    if not candidates:
        return results,[]
    field_id=min(candidates)[1]
    selected={result.field_id:result for result in results}
    prior=selected[field_id]
    job=job_by_id[field_id]
    scene=load_scene(job.scene_path)
    retry_settings=replace(settings,last_body_replan_seconds=30.0)
    retry=plan_field(job,retry_settings)
    def quality(route):
        worked=unary_union([part.completed_sweep for part in route.routes
                            if part.status=='REGION_ROUTE_PASS'])
        body_gap=(route.required_body.difference(worked).area
                  if route.required_body is not None else math.inf)
        return body_gap,worked.intersection(scene.target).area
    before,after=quality(prior),quality(retry)
    tolerance=scene.settings.coverage_tolerance_m2
    improved=(after[0]<=before[0]+tolerance
              and after[1]>=before[1]-tolerance
              and (after[0]<before[0]-tolerance
                   or after[1]>before[1]+tolerance))
    record={'field_id':field_id,
            'selected':'RETRY' if improved else 'INITIAL',
            'initial_status':prior.status,'retry_status':retry.status,
            'initial_body_gap_m2':before[0],
            'retry_body_gap_m2':after[0],
            'initial_target_covered_m2':before[1],
            'retry_target_covered_m2':after[1],
            'initial_elapsed_s':prior.elapsed_s,
            'retry_elapsed_s':retry.elapsed_s,
            'retry_last_body_replan_seconds':retry_settings.last_body_replan_seconds}
    winner=retry if improved else prior
    winner.elapsed_s=prior.elapsed_s+retry.elapsed_s
    winner.statistics['last_body_connection_retry']=record
    selected[field_id]=winner
    print(f"[last-body recheck] {field_id}: {retry.status}, "
          f"selected={record['selected']}, {record['retry_elapsed_s']:.2f}s",
          flush=True)
    return [selected[result.field_id] for result in results],[record]

# 总路线调度：根据 route_strategy 选择参考、规则或严格求解器。
# 输入核验、并行、导出与实际求解耗时分开统计；结果始终写新目录。
def run_batch(swath_bundle: Path, out: Path, *,
              workers: int=12, field_id: str | None=None,
              route_config: Path | None=None) -> int:
    """历史严格路线批入口：验证封存输入并调度策略、导出和审计；当前推荐几何生产入口是run_incremental_reference_batch。"""
    started=time.perf_counter()
    if workers<1:
        raise ValueError("workers 必须为正整数")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录非空，拒绝覆盖: {out}")
    if not out.resolve().is_relative_to(Path(__file__).resolve().parents[1]/"outputs"):
        raise ValueError("路线运行结果必须保存在 V6/outputs 下")
    settings=RouteSettings.from_file(route_config)
    upstream,manifest=_verify_bundle(swath_bundle)
    all_jobs=_field_rows(swath_bundle,manifest)
    jobs=[job for job in all_jobs if field_id is None or job.field_id==field_id]
    if not jobs:
        raise ValueError(f"没有找到田块: {field_id}")
    out.mkdir(parents=True,exist_ok=True)
    input_seconds=time.perf_counter()-started
    results=[]
    if workers==1:
        for index,job in enumerate(jobs,1):
            result=plan_field(job,settings)
            results.append(result)
            print(f"[{index}/{len(jobs)}] {job.field_id}: {result.status}, "
                  f"{result.elapsed_s:.2f}s",flush=True)
    else:
        pass  # 已合并到本模块，直接使用下方的定义。
        low_contention=([] if not settings.plan_headland_work or workers<=2
                        or settings.headland_work_mode == 'SEPARATE_PASSES'
                        else [job for job in jobs if _needs_low_contention_route(job)])
        low_ids={job.field_id for job in low_contention}
        ordinary=[job for job in jobs if job.field_id not in low_ids]
        batches=[]
        if ordinary:
            batches.append((ordinary,workers,out))
        if low_contention:
            phase=out/'low_contention_fields'
            phase.mkdir()
            batches.append((low_contention,min(2,workers),phase))
        for batch_jobs,batch_workers,batch_out in batches:
            for result in solve_parallel(batch_jobs,settings,batch_workers,batch_out):
                results.append(result)
                print(f"[{len(results)}/{len(jobs)}] {result.field_id}: "
                      f"{result.status}, {result.elapsed_s:.2f}s",flush=True)
    results,headland_retry_records=_retry_fragmented_headlands(
        jobs,results,settings,workers,out)
    results,singleton_records=(_retry_contention_sensitive_joint(
        jobs,results,settings,low_ids) if workers>1 else (results,[]))
    results,block_retry_records=_retry_partial_block_fields(
        jobs,results,settings)
    results,single_region_retry_records=_retry_single_region_budget(
        jobs,results,settings)
    results,last_body_retry_records=_retry_last_body_connection(
        jobs,results,settings)
    results,retry_records=_retry_near_complete_body(
        jobs,results,settings,workers,out)
    results,entry_rank_retry_records=_retry_headland_entry_rank(
        jobs,results,settings,workers,out)
    solve_seconds=time.perf_counter()-started-input_seconds
    ordered=sorted(results,key=lambda row:row.index)
    job_by_id={job.field_id:job for job in jobs}
    expected=sum(len(job.regions) for job in jobs)
    passed=sum(len({region.region_id for region in
                    (result.adapted_job or job_by_id[result.field_id]).regions
                    if region.region_id!='__HEADLAND__'
                    and region.task_ids and set(region.task_ids)<=set(
                        tid for block in result.routes
                        if block.region_id==region.region_id
                        and block.status=='REGION_ROUTE_PASS'
                        for tid in block.task_order)})
               for result in ordered)
    status_counts: dict[str,int]={}
    reason_counts: dict[str,int]={}
    for result in ordered:
        status_counts[result.status]=status_counts.get(result.status,0)+1
        for failure in result.failures:
            code=str(failure.get("code","UNKNOWN")).split(":")[0]
            reason_counts[code]=reason_counts.get(code,0)+1
    summary={
        "schema_version":SCHEMA_VERSION,
        "status":"ROUTES_STAGE_RECORDED",
        "claim":("CONTINUOUS_BODY_ROUTE_AND_SEPARATE_2_3_HEADLAND_PASSES; EXPLICIT_GAPS; NOT_REAL_MACHINE_CERTIFIED"
                 if settings.plan_headland_work and settings.headland_work_mode == 'SEPARATE_PASSES' else
                 "JOINT_BODY_HEADLAND_TASK_CHAIN_WITH_EXPLICIT_TARGET_GAPS; NOT_REAL_MACHINE_CERTIFIED"
                 if settings.plan_headland_work else
                 "CONTINUOUS_MAIN_BODY_ROUTE_WITH_TRANSFERS; HEADLAND_WORK_PENDING; NOT_REAL_MACHINE_CERTIFIED"),
        "input_bundle":str(swath_bundle.resolve()),
        "input_summary_sha256":_sha256(swath_bundle/"swath_batch_summary.json"),
        "input_public_gpkg_sha256":_sha256(swath_bundle/"swath_results.gpkg"),
        "input_metric_gpkg_sha256":_sha256(swath_bundle/"swath_results_metric.gpkg"),
        "frozen_code_sha256":{name:_sha256(Path(__file__).resolve().parent/name)
                              for name in FROZEN_FILES},
        "route_code_sha256":{name:_sha256(Path(__file__).parent/name) for name in
            ("route_planner.py","route.py","route_prepare.py","route_geometry.py",
             "route_joint.py","route_headland.py","route_residual.py",
             "route_reserve_candidates.py",
             "route_results.py","route_rebuild_audit.py","route_workers.py",
             "route_visualization.py","route_contours.py","route_access.py",
             "route_contour_choices.py","route_headland_separate.py")},
        "route_settings":asdict(settings),
        "worker_count":workers,
        "low_contention_field_ids":([job.field_id for job in low_contention]
                                      if workers>1 else []),
        "low_concurrency_retry":retry_records,
        "headland_simplification_retry":headland_retry_records,
        "singleton_recheck":singleton_records,
        "block_contention_retry":block_retry_records,
        "single_region_budget_retry":single_region_retry_records,
        "last_body_connection_retry":last_body_retry_records,
        "headland_entry_rank_retry":entry_rank_retry_records,
        "field_count":len(jobs),"region_count":expected,
        "passed_region_count":passed,
        "status_counts":status_counts,"failure_code_counts":reason_counts,
        "input_seconds":input_seconds,"solve_seconds":solve_seconds,
        "upstream_swath_status":upstream.get("status"),
        "stage_audit_status":"PENDING_INDEPENDENT_AUDIT"}
    export_started=time.perf_counter()
    export = export_route_result
    export(swath_bundle,out,jobs,ordered,summary,settings)
    summary["export_seconds"]=time.perf_counter()-export_started
    audit_started=time.perf_counter()
    pass  # 已合并到本模块，直接使用下方的定义。
    audit=audit_run(swath_bundle,out,workers=workers)
    summary["independent_audit_seconds"]=time.perf_counter()-audit_started
    summary["stage_audit_status"]=audit["status"]
    failed_audit_fields={item["field_id"] for item in audit["issues"]
                         if "field_id" in item}
    field_csv=out/"field_results.csv"
    audited_rows=[]
    with field_csv.open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle)
        headers=tuple(reader.fieldnames or ())
        for row in reader:
            row["audit_status"]=("PASS" if not audit["global_issues"]
                                         and row["field_id"] not in failed_audit_fields else "FAIL")
            audited_rows.append(row)
    _write_csv(field_csv,audited_rows,headers)
    display=gpd.read_file(out/"route_results.gpkg",layer="source_fields")
    statuses={row["field_id"]:row["audit_status"] for row in audited_rows}
    display["audit_status"]=display["field_id"].map(statuses)
    display.to_file(out/"route_results.gpkg",layer="source_fields",driver="GPKG",index=False)
    separate = settings.plan_headland_work and settings.headland_work_mode == 'SEPARATE_PASSES'
    accepted_status=("FULL_FIELD_ROUTE_COMPLETE" if settings.plan_headland_work and not separate
                     else "FIELD_ROUTE_COMPLETE")
    summary["planning_acceptance_passed"]=audit["status"]=="PASS" and all(
        result.status==accepted_status and (not separate or
            result.preparation.get('separate_headland',{}).get('planned_target_missing_m2',math.inf)
            <=load_scene(job_by_id[result.field_id].scene_path).settings.coverage_tolerance_m2)
        for result in ordered)
    summary['body_route_acceptance_passed'] = audit['status']=='PASS' and all(
        result.status==accepted_status for result in ordered)
    if separate:
        summary['separate_headland_pass_count'] = settings.headland_pass_count
        summary['separate_headland_planned_missing_m2'] = sum(
            f.preparation['separate_headland']['headland_planned_missing_m2'] for f in ordered
            if 'separate_headland' in f.preparation)
        summary['separate_headland_unrecorded_field_ids'] = [f.field_id for f in ordered
            if 'separate_headland' not in f.preparation]
    elapsed=np.asarray([result.elapsed_s for result in ordered])
    summary["field_time_seconds"]={"median":float(np.median(elapsed)),
        "p90":float(np.percentile(elapsed,90)),"p95":float(np.percentile(elapsed,95)),
        "max":float(elapsed.max())}
    summary["completed_task_count"]=sum(len(r.task_order) for f in ordered for r in f.routes)
    summary["replanned_task_count"]=sum(len((f.adapted_job or j).tasks) for j,f in zip(jobs,ordered))
    # Saved motion samples are the sole source of the diagnostic pictures.
    # Render after route auditing; no planning or geometry is rerun here.
    pass  # 已合并到本模块，直接使用下方的定义。
    summary["route_overviews"]=export_gallery(out, workers=min(4,workers))
    summary["total_wall_seconds"]=time.perf_counter()-started
    atomic_json(out/"route_batch_summary.json",summary)
    return 0 if summary["planning_acceptance_passed"] else 2


def _parse_args() -> argparse.Namespace:
    """历史路线模块独立命令参数；日常统一入口使用main.py与根配置。"""
    parser=argparse.ArgumentParser(
        description="V6 主体连续路线：重设田头、F2C 调头与真实跨区转场")
    parser.add_argument("--swath-bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--field-id")
    parser.add_argument("--route-config",type=Path)
    return parser.parse_args()

# ==========================================================================
# 2. 严格路线结果与采样导出
# 按真实 Motion 分段导出，记录任务、顺序、机具状态和可复核的采样切片。
# ==========================================================================

from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union

import route_planner as io


# 从真实运动采样导出任务、连接与扫掠，不画跨缺口的伪连续总线。
# 切片偏移、样本数量和执行顺序都是后续独立审计需要的契约。
def export_route_result(bundle,out,jobs,results,summary,settings):
    """导出历史严格路线、运动采样和输入快照供回读审计，不套用精简参考协议的图层承诺。"""
    (out/'inputs').mkdir();(out/'motion_samples').mkdir()
    (out/'headland_pass_samples').mkdir()
    original={j.field_id:j for j in jobs}
    transforms={}
    layers={name:[] for name in ['source_fields','work_regions','frozen_swath_segments',
        'replanned_swath_segments','headland_work_segments','headland_work_region',
        'headland_reserve','required_body','route_work_segments',
        'route_connections','inter_region_transfers','regional_routes','field_routes',
        'joint_route_candidates',
        'route_work_sweeps','route_failures', 'headland_pass_paths',
        'headland_pass_segments','headland_pass_sweeps','headland_pass_gaps']}
    fields=[];regions=[];motions=[];connections=[];events=[];responsibilities=[];lineage=[]
    input_hashes={}; separate_motions=[]
    for result in sorted(results,key=lambda r:r.index):
        source=original[result.field_id]
        job=result.adapted_job or source
        scene=io.load_scene(job.scene_path)
        taskmap={t.task_id:t for t in job.tasks}
        world=lambda g:io._to_world(g,job.origin,job.metric_crs,job.source_crs,transforms)
        headland=unary_union([r.headland for r in job.regions])
        required=(result.required_body if result.required_body is not None else
                  scene.target.difference(headland))
        if result.preparation.get('separate_headland'):
            # Use precisely the same reserve as the standalone generator.
            # Unioning per-region reserves can leave decimal-coordinate
            # slivers along partition seams; it is not the area ledger here.
            headland=scene.target.difference(required)
        snapshot={'field_id':job.field_id,'scene_path':job.scene_path,
            'scene_sha256':io._sha256(Path(job.scene_path)),
            'origin':job.origin,'metric_crs':job.metric_crs,
            'target_wkt':scene.target.wkt,'travel_wkt':scene.travel.wkt,
            'headland_wkt':headland.wkt,'required_body_wkt':required.wkt,
            'regions':[{'region_id':r.region_id,'geometry_wkt':r.geometry.wkt,
                        'task_ids':list(r.task_ids)} for r in job.regions],
            'tasks':[{'task_id':t.task_id,'region_id':t.region_id,
                      'reference_line_wkt':t.reference_line.wkt,'heading_rad':t.heading_rad,
                      'sweep_wkt':t.frozen_sweep.wkt,'row_index':t.row_index,
                      'required_target_sweep_wkt':(
                          t.required_target_sweep.wkt
                          if t.required_target_sweep is not None else None),
                      'work_kind':t.work_kind} for t in job.tasks],
            'preparation':result.preparation,'statistics':result.statistics}
        snapshot['separate_headland_tasks'] = [{'task_id':t.task_id,
            'pass_index':t.row_index+1,'work_kind':t.work_kind,
            'reference_line_wkt':t.reference_line.wkt,'sweep_wkt':t.frozen_sweep.wkt}
            for t in result.separate_headland_tasks]
        path=out/'inputs'/f'{job.field_id}.json'
        io.atomic_json(path,snapshot);input_hashes[job.field_id]=io._sha256(path)
        # A region may be executed in several connected row blocks. Preserve
        # the actual motion order instead of reducing it to one route per ID.
        executed=[r for r in result.routes if r.status=='REGION_ROUTE_PASS']
        completed=[tid for r in executed for tid in r.task_order]
        covered=unary_union([taskmap[t].frozen_sweep for t in completed])
        generated_complete=(set(completed)==set(taskmap)
                            and len(completed)==len(set(completed))
                            and result.status in {'FIELD_ROUTE_COMPLETE',
                                'JOINT_TASK_CHAIN_TARGET_GAP','FULL_FIELD_ROUTE_COMPLETE'})
        fieldrow={'field_id':job.field_id,'field_route_status':result.status,
            'region_count':len(source.regions),
            'planning_region_count':len(job.regions),
            'passed_region_count':len({r.region_id for r in executed
                if r.region_id!='__HEADLAND__'
                and set(next(g.task_ids for g in job.regions if g.region_id==r.region_id))
                   <=set(t for block in executed if block.region_id==r.region_id
                           for t in block.task_order)}),
            'execution_block_count':len(executed),
            'headland_only_region_count':sum(r.status=='HEADLAND_ONLY' for r in result.routes),
            'original_task_count':len(source.tasks),'replanned_task_count':len(job.tasks),
            'completed_task_count':len(completed),'required_body_area_m2':required.area,
            'missing_body_area_m2':required.difference(covered).area,
            'original_target_missing_m2':scene.target.difference(covered).area,
            'reference_margin_exclusion_area_m2':result.preparation.get(
                'reference_margin_exclusion_area_m2'),
            'effective_headland_simplification_tolerance_m':result.preparation.get(
                'effective_headland_simplification_tolerance_m'),
            'generated_task_chain_complete':generated_complete,
            'headland_work_status':result.preparation.get('headland_work_status','NOT_PLANNED'),
            'headland_task_count':sum(t.region_id=='__HEADLAND__' for t in job.tasks),
            'deferred_headland_work_m2':scene.target.difference(covered).intersection(headland).area,
            'target_area_m2':scene.target.area,'headland_width_m':result.preparation.get('headland_width_m'),
            'transfer_status':result.transfer_status,'elapsed_s':result.elapsed_s,
            'upstream_seam_quality_status':source.upstream_seam_quality_status,
            'swath_geometry_changed':settings.replan_headlands,
            'operational_status':'PARAMETERS_UNVERIFIED','audit_status':'PENDING',
            'error':result.error}
        separate = result.preparation.get('separate_headland',{})
        fieldrow.update(headland_work_mode=separate.get('mode','JOINT' if settings.plan_headland_work else 'NOT_PLANNED'),
            headland_pass_count=separate.get('requested_pass_count',0),
            separate_headland_segment_count=len(result.separate_headland_tasks),
            headland_planned_missing_m2=separate.get('headland_planned_missing_m2'),
            planned_target_missing_m2=separate.get('planned_target_missing_m2'),
            headland_coverage_status=separate.get('coverage_status','NOT_EVALUATED'),
            headland_connection_status=separate.get('connection_status','JOINT' if settings.plan_headland_work else 'NOT_PLANNED'))
        from route import motion as separate_motion
        from shapely.geometry import MultiLineString
        separate_arrays=[]; separate_offset=0; pass_lines={}; separate_sweeps=[]
        for index, task in enumerate(result.separate_headland_tasks):
            work = separate_motion(task, scene)
            common={'field_id':job.field_id,'task_id':task.task_id,'pass_index':task.row_index+1,
                'segment_index':index,'sample_offset':separate_offset,'sample_count':len(work.points),
                'work_kind':task.work_kind,'length_m':work.length,
                'connection_status':'ENTRY_EXIT_NOT_PLANNED',
                'operational_status':'PARAMETERS_UNVERIFIED'}
            separate_motions.append(common)
            separate_arrays.append(work.points);separate_offset+=len(work.points)
            line=LineString(work.points[:,:2])
            pass_lines.setdefault(task.row_index+1,[]).append(line)
            separate_sweeps.append(task.frozen_sweep)
            layers['headland_pass_segments'].append({**common,'geometry':world(line)})
            layers['headland_pass_sweeps'].append({**common,'area_m2':task.frozen_sweep.area,
                'geometry':world(task.frozen_sweep)})
        for pass_index, lines in sorted(pass_lines.items()):
            layers['headland_pass_paths'].append({'field_id':job.field_id,'pass_index':pass_index,
                'segment_count':len(lines),'connection_status':'SEGMENTS_NOT_FORCED_CONNECTED',
                'geometry':world(MultiLineString(lines))})
        np.savez_compressed(out/'headland_pass_samples'/f'{job.field_id}.npz',
            points=np.vstack(separate_arrays) if separate_arrays else np.empty((0,4)))
        if separate:
            gap=headland.difference(covered.union(unary_union(separate_sweeps)))
            if not gap.is_empty:
                layers['headland_pass_gaps'].append({'field_id':job.field_id,'area_m2':gap.area,
                    'geometry':world(gap)})
        fields.append(fieldrow)
        layers['source_fields'].append({**fieldrow,'feature_index':job.index,
            'field_intra_route_status':result.status,'geometry':job.source_geometry})
        for name,g in [('headland_reserve',headland),('required_body',required)]:
            if not g.is_empty:
                layers[name].append({'field_id':job.field_id,'area_m2':g.area,'geometry':world(g)})
        for name,subjects in [('frozen_swath_segments',source.tasks),
                              ('replanned_swath_segments',[t for t in job.tasks
                                  if t.region_id!='__HEADLAND__']),
                              ('headland_work_segments',[t for t in job.tasks
                                  if t.region_id=='__HEADLAND__'])]:
            for t in subjects:
                layers[name].append({'field_id':job.field_id,'region_id':t.region_id,
                    'task_id':t.task_id,'work_kind':t.work_kind,
                    'geometry':world(t.reference_line)})
        for region in job.regions:
            blocks=[r for r in executed if r.region_id==region.region_id]
            done={tid for block in blocks for tid in block.task_order}
            missed=next((r for r in result.routes if r.region_id==region.region_id
                         and r.status=='REGION_ROUTE_NOT_FOUND'),None)
            status=('REGION_ROUTE_PASS' if set(region.task_ids)<=done and blocks else
                    'HEADLAND_ONLY' if not region.task_ids else 'REGION_ROUTE_NOT_FOUND')
            row={'field_id':job.field_id,'region_id':region.region_id,'region_route_status':status,
                'execution_order':result.region_order.index(region.region_id)+1
                                  if region.region_id in result.region_order else None,
                'execution_block_count':len(blocks),
                'task_count':len(region.task_ids),'completed_task_count':len(done),
                'order_mode':','.join(dict.fromkeys(r.order_mode for r in blocks)),
                'connection_seconds':sum(r.connection_seconds for r in blocks)
                if blocks else None,'reason':missed.reason if missed else ''}
            regions.append(row)
            layer=('headland_work_region' if region.region_id=='__HEADLAND__'
                   else 'work_regions')
            layers[layer].append({**row,'geometry':world(region.geometry)})
        arrays=[];offset=0;motion_index=0;field_points=[]
        for execution_order,r in enumerate(executed,1):
            by_motion={id(c.motion):c for c in r.connections if c.motion is not None}
            regional_points=[]
            for motion in r.motions:
                key=f'{job.field_id}/{motion_index}'
                common={'field_id':job.field_id,'region_id':r.region_id,
                    'execution_order':execution_order,'motion_index':motion_index,
                    'motion_kind':motion.kind,'implement_on':motion.implement_on,
                    'task_id':motion.task_id or '',
                    'work_kind':taskmap[motion.task_id].work_kind
                    if motion.implement_on else '',
                    'sample_offset':offset,
                    'sample_count':len(motion.points)}
                motions.append(common);arrays.append(motion.points);offset+=len(motion.points)
                field_points.extend(motion.points[:,:2].tolist())
                if motion.kind!='transfer':
                    regional_points.extend(motion.points[:,:2].tolist())
                if motion.implement_on:
                    task=taskmap[motion.task_id]
                    layers['route_work_segments'].append({**common,'length_m':motion.length,
                        'geometry':world(LineString(motion.points[:,:2]))})
                    sweep=task.frozen_sweep
                    layers['route_work_sweeps'].append({**common,'area_m2':sweep.area,
                                                       'geometry':world(sweep)})
                    for owner in job.regions:
                        if owner.region_id=='__HEADLAND__':
                            continue
                        area=sweep.intersection(owner.geometry).area
                        if area>1e-5:
                            responsibilities.append({'field_id':job.field_id,
                                'owner_region_id':owner.region_id,'provider_region_id':r.region_id,
                                'task_id':task.task_id,'area_m2':area,'motion_index':motion_index})
                else:
                    c=by_motion[id(motion)]
                    row={**common,'connection_id':key,'from_task':c.from_task or '',
                        'to_task':c.to_task or '', 'method':c.method,'seconds':c.seconds,
                        'reverse_legs':c.reverse_legs,'gear_shifts':c.gear_shifts,
                        'reverse_m':c.reverse_m,'endpoint_unworked_area_m2':c.endpoint_unworked_area_m2}
                    connections.append(row)
                    for leg,(_,gear,line) in enumerate(io._motion_parts(motion)):
                        record={**row,'leg_index':leg,'gear':gear,'geometry':world(line)}
                        layers['route_connections'].append(record)
                        if motion.kind=='transfer':
                            layers['inter_region_transfers'].append(record)
                    for i,event in enumerate(c.events):
                        anchor=motion.points[0 if event['kind']=='IMPLEMENT_OFF' else -1]
                        events.append({'field_id':job.field_id,'region_id':r.region_id,
                            'motion_index':motion_index,'event_index':i,'kind':event['kind'],
                            'x_m':event.get('x_m',float(anchor[0])),
                            'y_m':event.get('y_m',float(anchor[1])),
                            'yaw_rad':event.get('yaw_rad',float(anchor[2])),
                            'duration_s':event['duration_s'],'duration_source':event['duration_source'],
                            'from_gear':event.get('from'),'to_gear':event.get('to')})
                motion_index+=1
            if len(regional_points)>1:
                layers['regional_routes'].append({'field_id':job.field_id,'region_id':r.region_id,
                    'execution_order':execution_order,'geometry':world(LineString(regional_points))})
        if len(field_points)>1:
            if result.status in {'FIELD_ROUTE_COMPLETE','FULL_FIELD_ROUTE_COMPLETE'}:
                layers['field_routes'].append({'field_id':job.field_id,
                    'task_count':len(completed),
                    'headland_work_status':fieldrow['headland_work_status'],
                    'geometry':world(LineString(field_points))})
            elif settings.plan_headland_work and executed:
                layers['joint_route_candidates'].append({'field_id':job.field_id,
                    'field_route_status':result.status,'task_count':len(completed),
                    'original_target_missing_m2':fieldrow['original_target_missing_m2'],
                    'geometry':world(LineString(field_points))})
        np.savez_compressed(out/'motion_samples'/f'{job.field_id}.npz',
                            points=np.vstack(arrays) if arrays else np.empty((0,4)))
        for f in result.failures:
            reg=next((r for r in job.regions if r.region_id==f['region_id']),None)
            point=(reg.geometry if reg else scene.target).representative_point()
            layers['route_failures'].append({**f,'geometry':world(point)})
        lineage.extend({'field_id':job.field_id,**row}
                       for row in result.preparation.get('task_mapping',[]))
    gpkg=out/'route_results.gpkg'
    for name,rows in layers.items():
        columns=tuple(rows[0]) if rows else ('field_id','region_id','geometry')
        io._write_layer(gpkg,name,rows,jobs[0].source_crs,columns)
    tables={'field_results':fields,'region_results':regions,'route_motions':motions,
            'route_connections':connections,'route_events':events,
            'coverage_responsibilities':responsibilities,'task_lineage':lineage,
            'headland_pass_motions':separate_motions}
    for name,rows in tables.items():
        io._write_csv(out/f'{name}.csv',rows,tuple(rows[0]) if rows else ('field_id',))
    summary.update(display_gpkg=str(gpkg),adapted_input_sha256=input_hashes,
                   export_counts={name:len(rows) for name,rows in layers.items()},
                   sample_contract='motion_samples/<field_id>.npz points[:, x,y,yaw,next_gear]; '
                                   'route_motions.csv provides ordered array slices')
    io.atomic_json(out/'route_batch_summary.json',summary)

# ==========================================================================
# 3. 导出结果回读与独立复核
# 从保存的采样和图层重新计算连续性及面积；不以求解器的成功标志代替证据。
# ==========================================================================

from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import numpy as np
import shapely
from shapely import wkt
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from scene import Motion as rebuild_audit_Motion
from scene import load_scene as rebuild_audit_load_scene
from scene import wrap as rebuild_audit_wrap
from validator import Validator as rebuild_audit_Validator
from validator import MotionChecker as rebuild_audit_MotionChecker
from validator import conservative_tool_coverage as rebuild_audit_conservative_tool_coverage
from io_utils import atomic_json as rebuild_audit_atomic_json


def rebuild_audit_records(path):
    """回读历史CSV运动/连接记录，保留字段字符串用于独立核对。"""
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def audit_field(payload):
    """从已保存输入与运动采样复核单田顺序、连接、安全和覆盖，不只检查导出PASS标签。"""
    out,field,motion_rows,connection_rows,event_rows,separate_rows=payload
    out=Path(out);fid=field['field_id'];issues=[]
    data=json.loads((out/'inputs'/f'{fid}.json').read_text())
    scene=rebuild_audit_load_scene(data['scene_path']);validator=rebuild_audit_Validator(scene);checker=rebuild_audit_MotionChecker(scene)
    expected={t['task_id']:t for t in data['tasks']}
    target=wkt.loads(data['target_wkt']);body=wkt.loads(data['required_body_wkt'])
    headland=wkt.loads(data['headland_wkt'])
    if scene.target.symmetric_difference(target).area>1e-6:
        issues.append('TARGET_CHANGED')
    if scene.travel.symmetric_difference(wkt.loads(data['travel_wkt'])).area>1e-6:
        issues.append('TRAVEL_CHANGED')
    # Same 0.0002 m² serialization allowance as the task-sweep readback. It
    # covers decimal-coordinate slivers, not the old 0.0708 m² ledger issue.
    if (headland.union(body).symmetric_difference(target).area>2e-4
            or headland.intersection(body).area>2e-4):
        issues.append('HEADLAND_BODY_LEDGER')
    points=np.load(out/'motion_samples'/f'{fid}.npz',allow_pickle=False)['points']
    connectors={int(c['motion_index']):c for c in connection_rows}
    events=defaultdict(list)
    for event in event_rows:
        events[int(event['motion_index'])].append(event)
    completed=GeometryCollection();previous=None;task_ids=[];block_order=[]
    offset=0;cross_links=0;previous_task_region=None;pending_transfer=False
    for index,row in enumerate(sorted(motion_rows,key=lambda r:int(r['motion_index']))):
        start=int(row['sample_offset']);count=int(row['sample_count'])
        if int(row['motion_index'])!=index or start!=offset or count<2:
            issues.append('MOTION_INDEX_OR_SLICE');continue
        offset+=count
        p=points[start:start+count]
        if len(p)!=count or not np.isfinite(p).all():
            issues.append('MISSING_SAMPLES');continue
        on=row['implement_on']=='True';tid=row['task_id'] or None
        motion=rebuild_audit_Motion(p,row['motion_kind'],tid,on)
        physics,parts=checker.physical(motion)
        issues.extend('PHYSICS_'+code for code in physics)
        if previous is not None:
            if (np.linalg.norm(previous[-1,:2]-p[0,:2])>scene.settings.join_tolerance_m
                    or abs(rebuild_audit_wrap(previous[-1,2]-p[0,2]))>scene.settings.heading_tolerance_rad):
                issues.append('FIELD_ROUTE_DISCONTINUITY')
        previous=p
        if on:
            if tid not in expected:
                issues.append('UNKNOWN_TASK');continue
            task=expected[tid]
            if task['region_id']!=row['region_id']:
                issues.append('TASK_REGION_CHANGED')
            actual=(rebuild_audit_conservative_tool_coverage(motion,scene)
                    if task.get('work_kind')=='CURVED_HEADLAND'
                    else validator.work_sweep(motion))
            frozen=wkt.loads(task['sweep_wkt'])
            if actual.symmetric_difference(frozen).area>max(2e-4,1e-8*frozen.area):
                issues.append('WORK_SWEEP_CHANGED')
            if task.get('required_target_sweep_wkt'):
                necessary_target=wkt.loads(task['required_target_sweep_wkt'])
                if necessary_target.difference(actual).area>max(2e-4,1e-8*necessary_target.area):
                    issues.append('REQUIRED_HEADLAND_TARGET_LOST')
            task_ids.append(tid)
            if not block_order or block_order[-1]!=row['region_id']:
                if previous_task_region is not None and not pending_transfer:
                    issues.append('MISSING_INTER_REGION_TRANSFER')
                block_order.append(row['region_id'])
            previous_task_region=row['region_id'];pending_transfer=False
            completed=unary_union([completed,actual.intersection(target)])
        else:
            c=connectors.get(index)
            if c is None:
                issues.append('CONNECTION_RECORD_MISSING');continue
            if parts is not None:
                allow=unary_union([headland,completed,scene.travel.difference(target)]).buffer(
                    scene.settings.geometry_epsilon_m)
                shapely.prepare(allow)
                if not np.all(shapely.covers(allow,parts)):
                    issues.append('UNWORKED_BODY_CROSSING')
            direction=[int(x) for x in p[:-1,3]]
            chain=[1,*direction,1]
            shifts=sum(a!=b for a,b in zip(chain,chain[1:]))
            legs=sum(d==-1 and chain[i-1]!=-1 for i,d in enumerate(direction,1))
            if shifts>4 or legs>2 or shifts!=int(c['gear_shifts']) or legs!=int(c['reverse_legs']):
                issues.append('GEAR_OR_REVERSE_LIMIT')
            actual_events=events[index]
            if sum(e['kind']=='GEAR_CHANGE' for e in actual_events)!=shifts:
                issues.append('GEAR_EVENT_MISSING')
            for event in actual_events:
                if float(event['duration_s'])<=0:
                    issues.append('FREE_OR_INVALID_OPERATION_EVENT')
                distances=np.linalg.norm(p[:,:2]-[float(event['x_m']),float(event['y_m'])],axis=1)
                nearest=int(np.argmin(distances))
                if distances[nearest]>scene.settings.join_tolerance_m or abs(rebuild_audit_wrap(
                        p[nearest,2]-float(event['yaw_rad'])))>scene.settings.heading_tolerance_rad:
                    issues.append('EVENT_NOT_ON_ROUTE_POSE')
            kinds={e['kind'] for e in actual_events}
            if ((c['from_task'] and 'IMPLEMENT_OFF' not in kinds)
                    or (c['to_task'] and 'IMPLEMENT_ON' not in kinds)):
                issues.append('IMPLEMENT_EVENT_MISSING')
            ds=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
            seconds=float(np.sum(ds/np.where(p[:-1,3]<0,scene.vehicle.reverse_speed_mps,
                                               scene.vehicle.turn_speed_mps)))
            seconds+=sum(float(e['duration_s']) for e in actual_events)
            if abs(seconds-float(c['seconds']))>1e-6:
                issues.append('CONNECTION_TIME_LEDGER')
            if row['motion_kind']=='transfer':
                pending_transfer=True
                if c['from_task'] and c['to_task']:
                    if expected[c['from_task']]['region_id']==expected[c['to_task']]['region_id']:
                        issues.append('FALSE_INTER_REGION_TRANSFER')
                    else:
                        cross_links+=1
    if offset!=len(points):
        issues.append('UNREFERENCED_SAMPLES')
    if len(task_ids)!=len(set(task_ids)):
        issues.append('DUPLICATE_WORK_TASK')
    if len(task_ids)!=int(field['completed_task_count']):
        issues.append('TASK_COUNT_MISMATCH')
    generated_complete=field.get('generated_task_chain_complete')=='True'
    body_complete=field['field_route_status'] in {
        'FIELD_ROUTE_COMPLETE','JOINT_TASK_CHAIN_TARGET_GAP',
        'JOINT_TASKS_PARTIAL','FULL_FIELD_ROUTE_COMPLETE',
        'JOINT_HEADLAND_GENERATION_FAILED'}
    if generated_complete and set(task_ids)!=set(expected):
        issues.append('INCOMPLETE_TASK_SET')
    if generated_complete and cross_links!=max(0,len(block_order)-1):
        issues.append('TRANSFER_COUNT_MISMATCH')
    missing=body.difference(completed).area
    if body_complete and missing>scene.settings.coverage_tolerance_m2:
        issues.append('MAIN_BODY_UNCOVERED')
    if body_complete:
        body_ids={tid for tid,task in expected.items()
                  if task['region_id']!='__HEADLAND__'}
        if not body_ids.issubset(set(task_ids)):
            issues.append('BODY_TASKS_NOT_COMPLETED')
    if abs(missing-float(field['missing_body_area_m2']))>1e-5:
        issues.append('BODY_AREA_LEDGER')
    target_missing=target.difference(completed).area
    if abs(target_missing-float(field['original_target_missing_m2']))>1e-5:
        issues.append('ORIGINAL_TARGET_AREA_LEDGER')
    separate_record=data['preparation'].get('separate_headland')
    if separate_record is not None:
        auxiliary=np.load(out/'headland_pass_samples'/f'{fid}.npz',allow_pickle=False)['points']
        expected_aux={t['task_id']:t for t in data.get('separate_headland_tasks',[])}
        seen=set();aux_offset=0;aux_sweeps=[]
        if separate_record['requested_pass_count'] not in (2,3):
            issues.append('HEADLAND_PASS_COUNT')
        for index,row in enumerate(separate_rows):
            first=int(row['sample_offset']);count=int(row['sample_count'])
            tid=row['task_id'];task=expected_aux.get(tid)
            if (int(row['segment_index'])!=index or first!=aux_offset or count<2
                    or tid in seen or task is None):
                issues.append('HEADLAND_PASS_SAMPLE_CONTRACT');continue
            aux_offset+=count;seen.add(tid)
            p=auxiliary[first:first+count]
            if len(p)!=count:
                issues.append('HEADLAND_PASS_MISSING_SAMPLES');continue
            # Each segment is checked independently. Deliberately no test of
            # continuity to the body or another independent pass is made.
            work=rebuild_audit_Motion(p,'work',tid,True)
            physics,_=checker.physical(work)
            issues.extend('HEADLAND_PASS_PHYSICS_'+code for code in physics)
            if work.length<1.0:
                issues.append('HEADLAND_PASS_TOO_SHORT')
            if int(row['pass_index'])!=task['pass_index'] or not (
                    1<=task['pass_index']<=separate_record['requested_pass_count']):
                issues.append('HEADLAND_PASS_INDEX')
            actual=(rebuild_audit_conservative_tool_coverage(work,scene) if task['work_kind']=='CURVED_HEADLAND'
                    else validator.work_sweep(work))
            stored=wkt.loads(task['sweep_wkt'])
            if actual.symmetric_difference(stored).area>max(2e-4,1e-8*stored.area):
                issues.append('HEADLAND_PASS_SWEEP_CHANGED')
            if abs(work.length-float(row['length_m']))>1e-5:
                issues.append('HEADLAND_PASS_LENGTH_LEDGER')
            aux_sweeps.append(actual)
        if seen!=set(expected_aux) or aux_offset!=len(auxiliary):
            issues.append('HEADLAND_PASS_TASK_SET')
        combined=completed.union(unary_union(aux_sweeps))
        planned_missing=target.difference(combined).area
        headland_missing=headland.difference(combined).area
        if (abs(planned_missing-float(field['planned_target_missing_m2']))>1e-5 or
                abs(headland_missing-float(field['headland_planned_missing_m2']))>1e-5):
            issues.append('HEADLAND_PASS_AREA_LEDGER')
        if field['headland_coverage_status']=='COVERAGE_COMPLETE' and headland_missing>scene.settings.coverage_tolerance_m2:
            issues.append('FALSE_HEADLAND_COVERAGE_COMPLETE')
    if (field['field_route_status']=='FULL_FIELD_ROUTE_COMPLETE'
            and target_missing>scene.settings.coverage_tolerance_m2):
        issues.append('FALSE_FULL_FIELD_ROUTE_COMPLETE')
    if (field['field_route_status']=='JOINT_TASK_CHAIN_TARGET_GAP'
            and (not generated_complete or
                 target_missing<=scene.settings.coverage_tolerance_m2)):
        issues.append('JOINT_TARGET_GAP_STATUS_INCONSISTENT')
    if len(points):
        for prescribed,actual,label in [(scene.start,points[0],'START'),
                (scene.end if generated_complete else None,points[-1],'END')]:
            if prescribed and (math.hypot(prescribed.x-actual[0],prescribed.y-actual[1])>
                    scene.settings.join_tolerance_m or abs(rebuild_audit_wrap(prescribed.yaw-actual[2]))>
                    scene.settings.heading_tolerance_rad):
                issues.append('FIELD_'+label+'_NOT_MET')
    return {'field_id':fid,'status':'PASS' if not issues else 'FAIL',
            'issue_counts':dict(Counter(issues)),'completed_tasks':len(task_ids),
            'checked_motions':len(motion_rows),'cross_region_transfers':cross_links,
            'main_body_missing_m2':missing,
            'original_target_missing_m2':target_missing}


# 回读保存的采样和图层复算验收，不依赖求解器宣称的成功。
# 源码兼容只接受明确核验过的版本；冻结算法和输入指纹仍严格检查。
def audit_run(bundle,out,workers=12):
    """汇总历史路线批次独立复核与跨文件状态，单田失败必须反映到整批。"""
    pass  # 已合并到本模块，直接使用下方的定义。
    summary=json.loads((out/'route_batch_summary.json').read_text())
    global_issues=[]
    for name,key in [('swath_results.gpkg','input_public_gpkg_sha256'),
                     ('swath_results_metric.gpkg','input_metric_gpkg_sha256'),
                     ('swath_batch_summary.json','input_summary_sha256')]:
        if _sha256(bundle/name)!=summary[key]:
            global_issues.append('UPSTREAM_FINGERPRINT:'+name)
    for name,expected in summary['frozen_code_sha256'].items():
        if _sha256(Path(__file__).parent/name)!=expected:
            global_issues.append('FROZEN_CODE_FINGERPRINT:'+name)
    refactor_matches = []
    for name,expected in summary['route_code_sha256'].items():
        if not _source_fingerprint_matches(Path(__file__).parent/name, expected):
            global_issues.append('ROUTE_CODE_FINGERPRINT:'+name)
        elif _sha256(Path(__file__).parent/name) != expected:
            refactor_matches.append(name)
    fields=rebuild_audit_records(out/'field_results.csv')
    tables=[]
    for name in ['route_motions','route_connections','route_events','headland_pass_motions']:
        by_field=defaultdict(list)
        for row in rebuild_audit_records(out/f'{name}.csv'):
            by_field[row['field_id']].append(row)
        tables.append(by_field)
    payloads=[]
    for f in fields:
        path=out/'inputs'/f"{f['field_id']}.json"
        if _sha256(path)!=summary['adapted_input_sha256'][f['field_id']]:
            global_issues.append('ADAPTED_INPUT_FINGERPRINT:'+f['field_id'])
        snapshot=json.loads(path.read_text())
        if _sha256(Path(snapshot['scene_path']))!=snapshot['scene_sha256']:
            global_issues.append('SCENE_SNAPSHOT_CHANGED:'+f['field_id'])
        payloads.append((str(out),f,*[t[f['field_id']] for t in tables]))
    if workers==1:
        rows=[audit_field(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows=list(pool.map(audit_field,payloads))
    # Verify that the GIS route is the same connected path as the authoritative
    # metre samples, including inter-region transfers (never a display chord).
    import geopandas as gpd
    from shapely.geometry import LineString
    pass  # 已合并到本模块，直接使用下方的定义。
    frame=gpd.read_file(out/'route_results.gpkg',layer='field_routes')
    expected_fields={f['field_id'] for f in fields
                     if f['field_route_status'] in {
                         'FIELD_ROUTE_COMPLETE','FULL_FIELD_ROUTE_COMPLETE'}}
    if set(frame.field_id)!=expected_fields or len(frame)!=len(expected_fields):
        global_issues.append('GPKG_COMPLETE_ROUTE_SET')
    transforms={}
    for r in frame.itertuples():
        snap=json.loads((out/'inputs'/f'{r.field_id}.json').read_text())
        actual=_to_local(r.geometry,str(frame.crs),snap['metric_crs'],tuple(snap['origin']),transforms)
        points=np.load(out/'motion_samples'/f'{r.field_id}.npz')['points']
        coords=np.asarray(actual.coords)[:,:2]
        # Export preserves vertex order. A vertex-wise comparison also catches
        # reordering and takes O(n); Hausdorff on two dense copies was O(n²).
        if len(coords)!=len(points) or np.max(np.linalg.norm(coords-points[:,:2],axis=1))>1e-4:
            global_issues.append('GPKG_ROUTE_GEOMETRY:'+r.field_id)
    candidates=gpd.read_file(out/'route_results.gpkg',layer='joint_route_candidates')
    expected_candidates={f['field_id'] for f in fields
        if summary['route_settings'].get('plan_headland_work')
        and f['field_route_status'] not in {
            'FIELD_ROUTE_COMPLETE','FULL_FIELD_ROUTE_COMPLETE'}
        and int(f['completed_task_count'])>0}
    if set(candidates.field_id)!=expected_candidates or len(candidates)!=len(expected_candidates):
        global_issues.append('GPKG_JOINT_CANDIDATE_SET')
    for row in candidates.itertuples():
        snap=json.loads((out/'inputs'/f'{row.field_id}.json').read_text())
        actual=_to_local(row.geometry,str(candidates.crs),snap['metric_crs'],
                         tuple(snap['origin']),transforms)
        samples=np.load(out/'motion_samples'/f'{row.field_id}.npz')['points']
        coords=np.asarray(actual.coords)[:,:2]
        if len(coords)!=len(samples) or np.max(np.linalg.norm(
                coords-samples[:,:2],axis=1))>1e-4:
            global_issues.append('GPKG_JOINT_CANDIDATE_GEOMETRY:'+row.field_id)
    auxiliary=gpd.read_file(out/'route_results.gpkg',layer='headland_pass_segments')
    aux_rows=rebuild_audit_records(out/'headland_pass_motions.csv')
    lookup={(r['field_id'],r['task_id']):r for r in aux_rows}
    if {(r.field_id,r.task_id) for r in auxiliary.itertuples()}!=set(lookup) or len(auxiliary)!=len(lookup):
        global_issues.append('GPKG_HEADLAND_PASS_SET')
    for fid, group in auxiliary.groupby('field_id'):
        snap=json.loads((out/'inputs'/f'{fid}.json').read_text())
        samples=np.load(out/'headland_pass_samples'/f'{fid}.npz')['points']
        for r in group.itertuples():
            row=lookup[(fid,r.task_id)]
            actual=_to_local(r.geometry,str(auxiliary.crs),snap['metric_crs'],tuple(snap['origin']),transforms)
            coords=np.asarray(actual.coords)[:,:2]
            first,count=int(row['sample_offset']),int(row['sample_count'])
            expected=samples[first:first+count,:2]
            if len(coords)!=len(expected) or np.max(np.linalg.norm(coords-expected,axis=1))>1e-4:
                global_issues.append('GPKG_HEADLAND_PASS_GEOMETRY:'+fid+'/'+r.task_id)
    issues=[{'field_id':r['field_id'],'codes':r['issue_counts']} for r in rows if r['status']=='FAIL']
    result={'status':'PASS' if not global_issues and not issues else 'FAIL',
            'source_refactor_compatibility':refactor_matches,
            'field_count':len(rows),'issues':issues,'global_issues':global_issues,'fields':rows,
            'method':'Independent exported-state reconstruction; shared low-level envelope kernel',
            'headland_work_status':('ATTEMPTED_WITH_EXPLICIT_GAPS'
                if summary['route_settings'].get('plan_headland_work') else 'NOT_PLANNED'),
            'operational_status':'PARAMETERS_UNVERIFIED'}
    rebuild_audit_atomic_json(out/'route_independent_audit.json',result)
    return result

# ==========================================================================
# 4. 逐田综合图与图库
# 统一工作线、连接线及田头线颜色；图片是结果展示，不是车辆安全认证。
# ==========================================================================

import csv
import hashlib
import html
import json
import os as visualization_os
from pathlib import Path
import time

visualization_ROOT = Path(__file__).resolve().parents[1]
visualization_CACHE = visualization_ROOT / 'outputs' / 'route_visualization_cache'
visualization_CACHE.mkdir(parents=True, exist_ok=True)
visualization_os.environ['MPLCONFIGDIR'] = str(visualization_CACHE / 'matplotlib')
visualization_os.environ['XDG_CACHE_HOME'] = str(visualization_CACHE)


def _visualization_records(path):
    """读取绘图所需CSV镜像；只用于展示，不生成新的任务或认证证据。"""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _draw(job):
    """绘制历史单田路线和问题，图形展示不替代运动及面积复核。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    import numpy as np
    from shapely import wkt
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union

    batch, destination, row, motions = job
    batch, destination = Path(batch), Path(destination)
    fid = row['field_id']
    snapshot = json.loads((batch / 'inputs' / f'{fid}.json').read_text())
    points = np.load(batch / 'motion_samples' / f'{fid}.npz',
                     allow_pickle=False)['points']
    target = wkt.loads(snapshot['target_wkt'])
    body = wkt.loads(snapshot['required_body_wkt'])
    taskmap = {task['task_id']: task for task in snapshot['tasks']}
    completed_ids = {m['task_id'] for m in motions
                     if m['implement_on'] == 'True' and m['task_id']}
    covered = unary_union([wkt.loads(taskmap[tid]['sweep_wkt'])
                           for tid in completed_ids if tid in taskmap])
    missing = target.difference(covered)
    font_path = Path('/System/Library/Fonts/STHeiti Medium.ttc')
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    xmin, ymin, xmax, ymax = target.bounds
    ratio = max((xmax-xmin)/max(ymax-ymin, 1e-6), .1)
    fig, axes = plt.subplots(1, 2, figsize=(16, min(13, max(5, 8/ratio+1.7))))

    def polygons(geom, ax, color, alpha=1, edge=None, width=.65):
        if geom.is_empty:
            return
        parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for poly in parts:
            if poly.geom_type != 'Polygon':
                continue
            poly = orient(poly, sign=1)
            vertices, codes = [], []
            for ring in [poly.exterior, *poly.interiors]:
                coordinates = np.asarray(ring.coords)
                vertices.extend(coordinates)
                codes.extend([MPath.MOVETO] + [MPath.LINETO] *
                             (len(coordinates)-2) + [MPath.CLOSEPOLY])
            ax.add_patch(PathPatch(MPath(vertices, codes), facecolor=color,
                edgecolor=edge or 'none', lw=width, alpha=alpha))

    colors = {'body': '#2e7d32', 'headland': '#159da3',
              'turn': '#2563eb', 'transfer': '#a622bb'}
    lines = {key: [] for key in colors}
    backwards = []
    distances = {key: 0.0 for key in colors}
    reverse_length = 0.0
    ordered = sorted(motions, key=lambda m: int(m['motion_index']))
    for motion in ordered:
        start, count = int(motion['sample_offset']), int(motion['sample_count'])
        samples = points[start:start+count]
        if len(samples) < 2:
            continue
        if motion['implement_on'] == 'True':
            kind = 'headland' if ('HEADLAND' in motion['work_kind'] or
                                 'RESERVE' in motion['work_kind']) else 'body'
        else:
            kind = 'transfer' if motion['motion_kind'] == 'transfer' else 'turn'
        lines[kind].append(samples[:, :2])
        step_lengths = np.linalg.norm(np.diff(samples[:, :2], axis=0), axis=1)
        distances[kind] += float(step_lengths.sum())
        reverse = samples[:-1, 3] < 0
        reverse_length += float(step_lengths[reverse].sum())
        for index in np.flatnonzero(reverse):
            backwards.append(samples[index:index+2, :2])
    bounds = target.bounds
    span = max(bounds[2]-bounds[0], bounds[3]-bounds[1])
    for panel, ax in enumerate(axes):
        polygons(target, ax, '#f8f5e9', edge='#3e4145')
        if panel == 1:
            polygons(body, ax, '#e7eddf', alpha=.7)
            polygons(missing, ax, '#f2a1a1', alpha=.85)
        for key, data in lines.items():
            if data:
                is_work = key in {'body', 'headland'}
                ax.add_collection(LineCollection(data, colors=colors[key],
                    linewidths=.65 if is_work else .8,
                    alpha=.16 if panel == 1 and is_work else .82))
        if backwards:
            ax.add_collection(LineCollection(backwards, colors='#e5750c',
                                            linewidths=.9, linestyles='dashed'))
        if len(points):
            ax.scatter(*points[0, :2], s=62, color='#151515', marker='o', zorder=8)
            ax.scatter(*points[-1, :2], s=75, color='#151515', marker='X', zorder=8)
            ax.annotate('S', points[0, :2], xytext=(4, 5),
                        textcoords='offset points', fontsize=10)
            ax.annotate('E', points[-1, :2], xytext=(4, -13),
                        textcoords='offset points', fontsize=10)
        for reg in snapshot['regions']:
            geom = wkt.loads(reg['geometry_wkt'])
            if geom.is_empty:
                continue
            polygons(geom, ax, 'none', edge='#747474', width=.45)
            p = geom.representative_point()
            ax.text(p.x, p.y, reg['region_id'], fontsize=7, color='#444',
                    bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.6})
        # Sparse arrows follow stored sample order; do not manufacture links.
        for motion in ordered[::max(1, len(ordered)//18)]:
            first, count = int(motion['sample_offset']), int(motion['sample_count'])
            if count < 2:
                continue
            middle = first + count//2
            a, b = points[middle-1, :2], points[middle, :2]
            delta = b-a
            norm = float(np.linalg.norm(delta))
            if norm > 1e-8:
                end = a+delta/norm*span*.015
                ax.annotate('', xy=end, xytext=a,
                    arrowprops={'arrowstyle':'->', 'color':'#222', 'lw':.65})
        pad = span*.045
        ax.set_xlim(bounds[0]-pad, bounds[2]+pad)
        ax.set_ylim(bounds[1]-pad, bounds[3]+pad)
        ax.set_aspect('equal'); ax.set_xlabel('local x (m)'); ax.set_ylabel('local y (m)')
    axes[0].set_title('实际作业与完整连接轨迹', fontproperties=font)
    axes[1].set_title('连接诊断：调头、转场、倒车与覆盖缺口', fontproperties=font)
    from matplotlib.lines import Line2D
    labels = [('body','主体作业'),('headland','田头/补充作业'),
              ('turn','调头与区内连接'),('transfer','分区转场')]
    handles = [Line2D([],[],color=colors[key],lw=1.8,label=label)
               for key,label in labels]
    handles += [Line2D([],[],color='#e5750c',ls='--',label='倒车'),
                Line2D([],[],color='#f2a1a1',lw=6,label='未覆盖目标')]
    fig.legend(handles=handles, loc='lower center', ncol=6, prop=font)
    fig.suptitle(f"{fid} | {row['field_route_status']}\n"
        f"task chain complete: {row['generated_task_chain_complete']} | "
        f"target gap: {missing.area:.1f} m² | "
        f"work: {distances['body']+distances['headland']:.0f} m | "
        f"connections: {distances['turn']+distances['transfer']:.0f} m | "
        f"reverse: {reverse_length:.0f} m", fontsize=11)
    fig.tight_layout(rect=(0,.06,1,.91))
    filename = f'{fid}_route_overview.png'
    fig.savefig(destination / filename, dpi=135)
    plt.close(fig)
    # The overview necessarily overlays the whole itinerary. Also show
    # consecutive time windows so a reviewer can follow actual driving order.
    # Every exported motion occurs in exactly one colored panel; preceding
    # motions remain faint context. No path geometry is simplified or added.
    steps_filename = f'{fid}_route_steps.png'
    panel_count = min(6, max(1, (len(ordered)+11)//12))
    columns = min(3, panel_count)
    rows = (panel_count+columns-1)//columns
    step_fig, step_axes = plt.subplots(rows, columns, squeeze=False,
                                      figsize=(6*columns, 4.6*rows))
    windows = np.array_split(np.arange(len(ordered)),panel_count)
    previous_lines = []
    for panel,indices in enumerate(windows):
        ax = step_axes.flat[panel]
        polygons(target,ax,'#f8f5e9',edge='#444')
        if previous_lines:
            ax.add_collection(LineCollection(previous_lines,colors='#b6bac0',
                                             linewidths=.5,alpha=.3))
        stage_work=stage_connection=0.0
        for index in indices:
            motion=ordered[int(index)]
            first,count=int(motion['sample_offset']),int(motion['sample_count'])
            samples=points[first:first+count]
            if len(samples)<2:
                continue
            working=motion['implement_on']=='True'
            kind=('headland' if ('HEADLAND' in motion['work_kind'] or
                                'RESERVE' in motion['work_kind']) else 'body') if working else (
                  'transfer' if motion['motion_kind']=='transfer' else 'turn')
            ax.plot(samples[:,0],samples[:,1],color=colors[kind],lw=.85)
            lengths=np.linalg.norm(np.diff(samples[:,:2],axis=0),axis=1)
            if working:
                stage_work+=float(lengths.sum())
            else:
                stage_connection+=float(lengths.sum())
            reverse=samples[:-1,3]<0
            if np.any(reverse):
                ax.add_collection(LineCollection(
                    [samples[i:i+2,:2] for i in np.flatnonzero(reverse)],
                    colors='#e5750c',linestyles='dashed',linewidths=1.0))
            middle=len(samples)//2
            delta=samples[middle,:2]-samples[middle-1,:2]
            norm=float(np.linalg.norm(delta))
            if norm>1e-8:
                ax.annotate('',xy=samples[middle,:2]+delta/norm*span*.012,
                    xytext=samples[middle,:2],arrowprops={
                        'arrowstyle':'->','color':colors[kind],'lw':.75})
            previous_lines.append(samples[:,:2])
        if len(indices):
            first=ordered[int(indices[0])];last=ordered[int(indices[-1])]
            start=points[int(first['sample_offset']),:2]
            end=points[int(last['sample_offset'])+int(last['sample_count'])-1,:2]
            ax.scatter(*start,s=38,marker='o',color='#111',zorder=8)
            ax.scatter(*end,s=42,marker='X',color='#111',zorder=8)
            ax.annotate('起',start,xytext=(4,4),textcoords='offset points',
                        fontproperties=font,fontsize=8)
            ax.annotate('止',end,xytext=(4,-10),textcoords='offset points',
                        fontproperties=font,fontsize=8)
            ax.set_title(f'阶段 {panel+1}：运动 {int(indices[0])+1}–{int(indices[-1])+1}\n'
                f'作业 {stage_work:.0f} m · 连接 {stage_connection:.0f} m',
                fontproperties=font,fontsize=10)
        pad=span*.045
        ax.set_xlim(bounds[0]-pad,bounds[2]+pad)
        ax.set_ylim(bounds[1]-pad,bounds[3]+pad)
        ax.set_aspect('equal');ax.tick_params(labelsize=7)
    for ax in list(step_axes.flat)[panel_count:]:
        ax.set_visible(False)
    step_fig.suptitle(f'{fid}：按执行先后阅读同一条实际行程\n'
        f"{row['field_route_status']} · 目标缺口 {missing.area:.1f} m²",
        fontproperties=font,fontsize=13)
    step_fig.legend(handles=handles[:-1],loc='lower center',ncol=5,prop=font)
    step_fig.tight_layout(rect=(0,.045,1,.94))
    step_fig.savefig(destination/steps_filename,dpi=135)
    plt.close(step_fig)
    separate_filename = ''
    if snapshot.get('preparation',{}).get('separate_headland'):
        auxiliary = np.load(batch/'headland_pass_samples'/f'{fid}.npz',
                            allow_pickle=False)['points']
        auxiliary_rows=[r for r in _visualization_records(batch/'headland_pass_motions.csv') if r['field_id']==fid]
        pass_lines={}
        for r in auxiliary_rows:
            first,count=int(r['sample_offset']),int(r['sample_count'])
            pass_lines.setdefault(int(r['pass_index']),[]).append(auxiliary[first:first+count,:2])
        planned=covered.union(unary_union([wkt.loads(t['sweep_wkt'])
                                          for t in snapshot['separate_headland_tasks']]))
        remainder=target.difference(planned)
        separate_fig,separate_axes=plt.subplots(1,3,figsize=(18,6))
        palette=['#008b9a','#b04fd4','#df8014']
        for panel,ax in enumerate(separate_axes):
            polygons(target,ax,'#f8f5e9',edge='#414141')
            if panel in (0,2):
                for key,data in lines.items():
                    if data:
                        ax.add_collection(LineCollection(data,colors=colors[key],
                            linewidths=.7 if key=='body' else .8,
                            alpha=.3 if panel==2 else .9))
            if panel in (1,2):
                for pass_index,data in sorted(pass_lines.items()):
                    ax.add_collection(LineCollection(data,colors=palette[(pass_index-1)%3],
                        linewidths=1.25,alpha=.95,label=f'Pass {pass_index}'))
                if panel==1:
                    polygons(wkt.loads(snapshot['headland_wkt']),ax,'#a9c2c6',alpha=.23)
                    ax.legend(loc='upper right',fontsize=8)
                else:
                    polygons(remainder,ax,'#f2a1a1',alpha=.85)
            ax.set_aspect('equal');pad=span*.04
            ax.set_xlim(bounds[0]-pad,bounds[2]+pad);ax.set_ylim(bounds[1]-pad,bounds[3]+pad)
            ax.set_xlabel('local x (m)');ax.set_ylabel('local y (m)')
        for ax,title in zip(separate_axes,['主体条带及真实调头/转场','独立田头路径（断点不补线）','叠加结果及仍未覆盖的目标']):
            ax.set_title(title,fontproperties=font,fontsize=11)
        info=snapshot['preparation']['separate_headland']
        separate_fig.suptitle(f"{fid} | body: {row['field_route_status']} | "
            f"{info['requested_pass_count']} headland passes / {info['safe_segment_count']} safe segments\n"
            f"body connections {distances['turn']+distances['transfer']:.0f} m | "
            f"planned target gap {remainder.area:.1f} m² | headland entry/exit NOT planned",fontsize=12)
        separate_fig.tight_layout(rect=(0,0,1,.9))
        separate_filename=f'{fid}_separate_headland.png'
        separate_fig.savefig(destination/separate_filename,dpi=140);plt.close(separate_fig)
    return {'field_id':fid, 'source_batch':str(batch.resolve()),
        'route_status':row['field_route_status'], 'image':filename,
        'separate_headland_image':separate_filename,
        'execution_steps_image':steps_filename,
        'generated_task_chain_complete':row['generated_task_chain_complete'],
        'target_missing_m2':missing.area, 'work_length_m':distances['body']+distances['headland'],
        'connection_length_m':distances['turn']+distances['transfer'],
        'reverse_length_m':reverse_length}


def export_gallery(batch, out=None, workers=4, field_ids=None):
    """为已生成批次构建可读图库，展示失败及限制，不改写规划结果。
    
    Use saved route samples only; no route solver or safety contract changes."""
    from concurrent.futures import ProcessPoolExecutor
    started = time.perf_counter()
    batch = Path(batch)
    out = Path(out) if out is not None else batch / 'route_overviews'
    if not out.resolve().is_relative_to(visualization_ROOT / 'outputs'):
        raise ValueError('路线图必须保存在 V6/outputs')
    out.mkdir(parents=True, exist_ok=False)
    fields = _visualization_records(batch/'field_results.csv')
    if field_ids is not None:
        fields = [row for row in fields if row['field_id'] in set(field_ids)]
    by_field = {}
    for row in _visualization_records(batch/'route_motions.csv'):
        by_field.setdefault(row['field_id'], []).append(row)
    jobs = [(str(batch), str(out), row, by_field.get(row['field_id'], [])) for row in fields]
    if workers <= 1:
        rows = [_draw(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers,len(jobs) or 1)) as pool:
            rows = list(pool.map(_draw,jobs))
    if rows:
        with (out/'image_index.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]))
            writer.writeheader();writer.writerows(rows)
    cards = ''.join(f'<article><h2>{html.escape(row["field_id"])}</h2>'
        f'<p>{html.escape(row["route_status"])}；目标缺口 {row["target_missing_m2"]:.1f} m²</p>'
        f'<p><a href="{row["execution_steps_image"]}">按实际执行顺序分阶段查看</a></p>'
        f'<a href="{row["separate_headland_image"] or row["image"]}"><img loading="lazy" src="{row["separate_headland_image"] or row["image"]}"></a></article>' for row in rows)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>逐田路线综合图</title><style>body{font-family:sans-serif;margin:24px;}'
        'article{border-top:1px solid #ccc;margin:28px 0}img{width:100%;max-width:1600px}</style>'
        '<h1>逐田路线综合图</h1><p>图来自已导出运动；不补线，不隐藏绕行。S为起点，E为终点。'
        '任务连通不等于原始目标全覆盖。点击图片可查看原图。</p>'+cards,encoding='utf-8')
    summary={'field_count':len(rows),'source_batch':str(batch.resolve()),
        'render_seconds':time.perf_counter()-started,
        'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_motion_csv_sha256':hashlib.sha256((batch/'route_motions.csv').read_bytes()).hexdigest()}
    (out/'gallery_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

# ==========================================================================
# 5. 并行进程隔离
# 管理并行任务和子进程退出；一个田块失败不导致丢失其他田块结果。
# ==========================================================================

import faulthandler
import multiprocessing as mp
from multiprocessing.connection import wait
import time


def _entry(job,settings,pipe,logpath):
    """隔离子进程执行历史路线并保存原生故障日志，供父进程超时/崩溃处理。"""
    from route import plan_field
    with open(logpath,'w') as log:
        faulthandler.enable(file=log)
        try:
            pipe.send(plan_field(job,settings))
        finally:
            pipe.close()


# 在子进程中独立处理田块并保留异常结果，避免一个失败拖垮整批。
# 进程墙钟和各田累计求解时间不同，统计效率时不能混为一谈。
def solve_parallel(jobs,settings,workers,out):
    """用spawn隔离历史严格单田原生求解，父进程管理预算与失败，不等同于推荐参考自动资源调度。"""
    pass  # 已合并到本模块，直接使用下方的定义。
    context=mp.get_context('spawn')
    folder=out/'worker_logs';folder.mkdir()
    pending=iter(jobs);active={};exhausted=False
    try:
        while active or not exhausted:
            while not exhausted and len(active)<workers:
                try:
                    job=next(pending)
                except StopIteration:
                    exhausted=True;break
                receiver,sender=context.Pipe(duplex=False)
                process=context.Process(target=_entry,args=(job,settings,sender,
                    str(folder/f'{job.field_id}.log')))
                process.start();sender.close()
                active[receiver]=(process,job,time.perf_counter())
            if not active:
                break
            ready=set(wait(list(active),timeout=0.1))
            for pipe,(process,job,started) in list(active.items()):
                attempts=3 if settings.headland_strategy in {'AUTO','AUTO_FAST'} and settings.replan_headlands else 1
                block_budget=(max(60.0,settings.max_field_seconds*2)
                    if attempts>1 and len(job.regions)<=8
                    and 16<len(job.tasks)<=240 else 0.0)
                headland_budget=(settings.max_headland_seconds
                    if settings.plan_headland_work else 0.0)
                timeout=(time.perf_counter()-started>
                    max(settings.max_field_seconds,30.0)*attempts
                    +block_budget+headland_budget+20)
                if pipe not in ready and not process.is_alive():
                    # A worker can exit after wait() snapshots the pipes but
                    # before this liveness check. Give its final IPC frame a
                    # brief chance to arrive before declaring a missing result.
                    if pipe.poll(0.5):
                        ready.add(pipe)
                if pipe not in ready and process.is_alive() and not timeout:
                    continue
                result=None
                if pipe in ready:
                    try:
                        result=pipe.recv()
                    except EOFError:
                        pass
                if result is None:
                    if timeout and process.is_alive():
                        process.terminate()
                    process.join(timeout=1)
                    code=('WORKER_HARD_TIMEOUT' if timeout else
                          'WORKER_RESULT_MISSING' if process.exitcode==0 else
                          'NATIVE_WORKER_EXIT')
                    result=_field_error(job,code,f'worker exit={process.exitcode}',
                                        time.perf_counter()-started)
                    result.status='SEARCH_LIMIT' if timeout else 'WORKER_ERROR'
                else:
                    process.join(timeout=1)
                if process.is_alive():
                    process.terminate();process.join(timeout=1)
                pipe.close();del active[pipe]
                yield result
    finally:
        for pipe,(process,_,_) in active.items():
            if process.is_alive():
                process.terminate()
            process.join(timeout=1);pipe.close()

# ==========================================================================
# 6. 已保存批次恢复与末端续作
# 读取来源、核验摘要和参数后恢复行程；新批次记录当前实际源码。
# ==========================================================================

from collections import defaultdict
from collections import Counter
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import numpy as np
from shapely import wkt
from shapely.ops import unary_union

saved_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(saved_ROOT/'src'))
import route_planner as io
from scene import Motion as saved_Motion
def saved_continue_body(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import continue_body as _operation
    return _operation(*args, **kwargs)
saved_export = export_route_result
saved_audit_run = audit_run
saved_export_gallery = export_gallery


def rows(path, fid):
    """从历史CSV读取指定field_id记录，不按显示fid或文件行位置推断田块身份。"""
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [r for r in csv.DictReader(f) if r['field_id']==fid]


def restore(batch, source):
    """保留原采样和操作事件，不根据显示线重新拟合运动。"""
    fid=source.field_id
    d=json.loads((batch/'inputs'/f'{fid}.json').read_text())
    headland=wkt.loads(d['headland_wkt'])
    tasks=tuple(io.FrozenTask(fid,t['region_id'],t['task_id'],int(t['row_index']),i,
        wkt.loads(t['reference_line_wkt']),float(t['heading_rad']),wkt.loads(t['sweep_wkt']),
        wkt.loads(t['reference_line_wkt']).length,t['work_kind'])
        for i,t in enumerate(d['tasks']))
    source_regions={r.region_id:r for r in source.regions}
    regions=tuple(io.RegionInput(r['region_id'],source_regions[r['region_id']].sequence_index,
        wkt.loads(r['geometry_wkt']),wkt.loads(r['geometry_wkt']).intersection(headland),tuple(r['task_ids']))
        for r in d['regions'])
    adapted=replace(source,regions=regions,tasks=tasks)
    taskmap={t.task_id:t for t in tasks}
    samples=np.load(batch/'motion_samples'/f'{fid}.npz',allow_pickle=False)['points']
    eventmap=defaultdict(list)
    for e in rows(batch/'route_events.csv',fid):
        event={'kind':e['kind'],'duration_s':float(e['duration_s']),
            'duration_source':e['duration_source'],
            **{k:float(e[k]) for k in ('x_m','y_m','yaw_rad')}}
        if e['from_gear']:event['from']=int(e['from_gear'])
        if e['to_gear']:event['to']=int(e['to_gear'])
        eventmap[int(e['motion_index'])].append(event)
    connections={int(c['motion_index']):c for c in rows(batch/'route_connections.csv',fid)}
    grouped=[]
    for row in rows(batch/'route_motions.csv',fid):
        key=(row['execution_order'],row['region_id'])
        if not grouped or grouped[-1][0]!=key:grouped.append((key,[]))
        grouped[-1][1].append(row)
    routes=[]
    for (_,rid),mr in grouped:
        route=io.RegionRoute(rid,'REGION_ROUTE_PASS',order_mode='RESTORED_AUDITED_SAMPLES')
        for row in mr:
            offset=int(row['sample_offset']);count=int(row['sample_count'])
            on=row['implement_on']=='True'
            motion=saved_Motion(samples[offset:offset+count].copy(),row['motion_kind'],
                          row['task_id'] or None,on)
            route.motions.append(motion)
            if on:route.task_order.append(row['task_id'])
            else:
                c=connections[int(row['motion_index'])]
                connection=io.Connection(c['from_task'] or None,c['to_task'] or None,
                    motion,c['method'],float(c['seconds']),float(c['reverse_m']),
                    int(c['reverse_legs']),int(c['gear_shifts']),eventmap[int(row['motion_index'])])
                connection.endpoint_unworked_area_m2=float(c['endpoint_unworked_area_m2'])
                route.connections.append(connection)
        route.completed_sweep=unary_union([taskmap[t].frozen_sweep for t in route.task_order])
        route.connection_seconds=sum(c.seconds for c in route.connections)
        route.reverse_m=sum(c.reverse_m for c in route.connections)
        route.gear_shifts=sum(c.gear_shifts for c in route.connections)
        routes.append(route)
    field=rows(batch/'field_results.csv',fid)[0]
    result=io.FieldRoute(source.index,fid,field['field_route_status'],[r.region_id for r in routes],
        routes,[],float(field['elapsed_s']),source.source_crs,source.metric_crs,source.origin,
        source.source_geometry,source.upstream_seam_quality_status,
        source.upstream_acceptance_passed,source.upstream_area_ledger_delta_m2)
    result.adapted_job=adapted;result.required_body=wkt.loads(d['required_body_wkt'])
    result.preparation=d['preparation'];result.statistics=d['statistics']
    result.transfer_status=field['transfer_status']
    aux=np.load(batch/'headland_pass_samples'/f'{fid}.npz',allow_pickle=False)['points']
    auxrows={r['task_id']:r for r in rows(batch/'headland_pass_motions.csv',fid)}
    separate=[]
    for t in d['separate_headland_tasks']:
        row=auxrows[t['task_id']];first=int(row['sample_offset']);count=int(row['sample_count'])
        points=aux[first:first+count].copy();line=wkt.loads(t['reference_line_wkt'])
        separate.append(io.FrozenTask(fid,'__HEADLAND__',t['task_id'],t['pass_index']-1,
            len(separate),line,float(points[0,2]),wkt.loads(t['sweep_wkt']),
            float(row['length_m']),t['work_kind'],points))
    result.separate_headland_tasks=tuple(separate)
    return result


# 从已检查批次恢复任务与采样，再续接末端的剩余作业。
# 来源摘要与参数必须一致；当前输出重新记录当前实际源码版本。
def resume_saved_batch(batch, out, field_ids, seconds):
    """历史路线末端续作，读取审计通过的来源并写新目录；不同于--batch-resume的同批中断续算。"""
    started=time.perf_counter();batch=batch.resolve();out=out.resolve()
    if out.exists() or not out.is_relative_to(saved_ROOT/'outputs'):
        raise ValueError('必须使用 outputs 下的新目录')
    old=json.loads((batch/'route_batch_summary.json').read_text())
    prior_audit=json.loads((batch/'route_independent_audit.json').read_text())
    if prior_audit['status']!='PASS':raise ValueError('来源审计未通过')
    for key in ('route_code_sha256','frozen_code_sha256'):
        for name,digest in old[key].items():
            matches = (io._source_fingerprint_matches(saved_ROOT/'src'/name, digest)
                       if key == 'route_code_sha256' else io._sha256(saved_ROOT/'src'/name) == digest)
            if not matches:raise ValueError('来源源码已变:'+name)
    bundle=Path(old['input_bundle'])
    upstream,manifest=io._verify_bundle(bundle)
    jobs=[j for j in io._field_rows(bundle,manifest) if j.field_id in field_ids]
    if {j.field_id for j in jobs}!=set(field_ids):raise ValueError('田块 ID 缺失')
    results=[]
    settings=None
    for job in jobs:
        fid=job.field_id
        snap=batch/'inputs'/f'{fid}.json'
        if io._sha256(snap)!=old['adapted_input_sha256'][fid]:raise ValueError('输入快照已变化')
        d=json.loads(snap.read_text())
        if io._sha256(Path(d['scene_path']))!=d['scene_sha256']:raise ValueError('scene 已变化')
        data=old.get('route_settings_by_field',{}).get(fid,old['route_settings'])
        settings=io.RouteSettings(**data)
        print(fid,'RESTORE',flush=True)
        prior=restore(batch,job)
        result=saved_continue_body(prior,settings,seconds=seconds)
        results.append(result)
        print(fid,result.status,json.dumps(result.statistics['prefix_continuation'],ensure_ascii=False),flush=True)
    out.mkdir()
    inherited={k:old[k] for k in ('schema_version','input_bundle','input_summary_sha256',
        'input_public_gpkg_sha256','input_metric_gpkg_sha256','frozen_code_sha256') if k in old}
    summary={**inherited,'status':'CONTINUED_SAVED_BODY_ROUTES',
        'claim':'AUDITED_PREFIX_PRESERVED; SEPARATE_HEADLAND_PASSES; EXPLICIT_GAPS',
        'source_batch':str(batch),'source_summary_sha256':io._sha256(batch/'route_batch_summary.json'),
        'continuation_utility_sha256':io._sha256(Path(__file__)),
        'route_code_sha256':{**{name:io._sha256(saved_ROOT/'src'/name) for name in old['route_code_sha256']},'route_continue.py':io._sha256(saved_ROOT/'src'/'route_continue.py'),
            'route_saved.py':io._sha256(saved_ROOT/'src'/'route_saved.py')},
        'route_settings':asdict(settings),'field_count':len(jobs),'worker_count':1,
        'status_counts':dict(Counter(r.status for r in results)),
        'stage_audit_status':'PENDING_INDEPENDENT_AUDIT','planning_acceptance_passed':False,
        'solve_seconds':time.perf_counter()-started,
        'region_count':sum(len(j.regions) for j in jobs),
        'passed_region_count':sum(sum(set(r.task_ids)<=set(t for block in f.routes
            if block.status=='REGION_ROUTE_PASS' for t in block.task_order)
            for r in f.adapted_job.regions) for f in results)}
    summary.pop('route_settings_by_field',None)
    summary.pop('route_overviews',None)
    saved_export(bundle,out,jobs,results,summary,settings)
    io.atomic_json(out/'route_batch_summary.json',summary)
    audit=saved_audit_run(bundle,out,workers=1)
    summary['stage_audit_status']=audit['status']
    failed={item['field_id'] for item in audit['issues'] if 'field_id' in item}
    with (out/'field_results.csv').open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);headers=tuple(reader.fieldnames or ());fields=list(reader)
    for row in fields:
        row['audit_status']='PASS' if not audit['global_issues'] and row['field_id'] not in failed else 'FAIL'
    io._write_csv(out/'field_results.csv',fields,headers)
    import geopandas as gpd
    frame=gpd.read_file(out/'route_results.gpkg',layer='source_fields')
    frame['audit_status']=frame.field_id.map({r['field_id']:r['audit_status'] for r in fields})
    frame.to_file(out/'route_results.gpkg',layer='source_fields',driver='GPKG',index=False)
    summary['body_route_acceptance_passed']=audit['status']=='PASS' and all(r.status=='FIELD_ROUTE_COMPLETE' for r in results)
    summary['planning_acceptance_passed']=False
    if audit['status']=='PASS':summary['route_overviews']=saved_export_gallery(out,workers=1)
    summary['total_wall_seconds']=time.perf_counter()-started
    summary['timing_claim']='Restore, checked-prefix continuation, export, audit and rendering; excludes source solve costs'
    io.atomic_json(out/'route_batch_summary.json',summary)
    print('FINAL',audit['status'],summary['status_counts'],summary['total_wall_seconds'],flush=True)
    return 0 if audit['status']=='PASS' else 2

# ==========================================================================
# 7. 单田有界恢复入口
# 只在明确来源与预算下尝试修复，失败仍导出保留的有效行程和原因。
# ==========================================================================

from dataclasses import asdict
from pathlib import Path
import csv
import json
import math
import time

from shapely import wkt

import route_planner as io
recover_restore = restore
def recover_seed_and_continue(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import seed_and_continue as _operation
    return _operation(*args, **kwargs)
def recover_insert_last_task(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import insert_last_task as _operation
    return _operation(*args, **kwargs)
def recover_preserve_required_body(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import preserve_required_body as _operation
    return _operation(*args, **kwargs)
recover_export = export_route_result
recover_audit_run = audit_run
recover_export_gallery = export_gallery

recover_ROOT = Path(__file__).resolve().parents[1]


def _load_checked_source(batch, fid):
    """加载历史路线来源及独立审计证据，来源未通过或失配不能被恢复入口自动认可。"""
    summary = json.loads((batch/'route_batch_summary.json').read_text())
    audit = json.loads((batch/'route_independent_audit.json').read_text())
    if audit['status'] != 'PASS':
        raise ValueError('RECOVERY_SOURCE_AUDIT_FAILED')
    for key in ('route_code_sha256', 'frozen_code_sha256'):
        for name, digest in summary[key].items():
            matches = (io._source_fingerprint_matches(recover_ROOT/'src'/name, digest)
                       if key == 'route_code_sha256' else io._sha256(recover_ROOT/'src'/name) == digest)
            if not matches:
                raise ValueError('RECOVERY_SOURCE_CODE_CHANGED:'+name)
    settings = io.RouteSettings(**summary.get('route_settings_by_field', {}).get(
        fid, summary['route_settings']))
    if not settings.plan_headland_work or settings.headland_work_mode != 'SEPARATE_PASSES':
        raise ValueError('RECOVERY_REQUIRES_SEPARATE_HEADLAND')
    snapshot = batch/'inputs'/f'{fid}.json'
    if io._sha256(snapshot) != summary['adapted_input_sha256'][fid]:
        raise ValueError('RECOVERY_SNAPSHOT_CHANGED')
    data = json.loads(snapshot.read_text())
    if io._sha256(Path(data['scene_path'])) != data['scene_sha256']:
        raise ValueError('RECOVERY_SCENE_CHANGED')
    return summary, settings, data


# 单田有限恢复入口，执行前必须明确田块、来源与预算。
# 无改进时仍保留原有效行程，并记录未完成位置，而非伪造成功连接。
def recover_saved_batch(batch, out, field_id, seconds=30.0, baseline_batch=None):
    """为指定失败田做有限路线恢复并写新结果，预算必须正有限，原来源不覆盖。"""
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('RECOVERY_BUDGET_MUST_BE_POSITIVE_FINITE')
    started = time.perf_counter()
    batch, out = Path(batch).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(recover_ROOT/'outputs'):
        raise ValueError('必须使用 outputs 下的新目录')
    old, settings, data = _load_checked_source(batch, field_id)
    bundle = Path(old['input_bundle'])
    for name, key in (('swath_results.gpkg', 'input_public_gpkg_sha256'),
                      ('swath_results_metric.gpkg', 'input_metric_gpkg_sha256'),
                      ('swath_batch_summary.json', 'input_summary_sha256')):
        if io._sha256(bundle/name) != old[key]:
            raise ValueError('RECOVERY_UPSTREAM_CHANGED:'+name)
    _, manifest = io._verify_bundle(bundle)
    job = next((j for j in io._field_rows(bundle, manifest) if j.field_id == field_id), None)
    if job is None:
        raise ValueError('RECOVERY_FIELD_NOT_FOUND')
    baseline = None
    if baseline_batch is not None:
        baseline_batch = Path(baseline_batch).resolve()
        _, _, baseline = _load_checked_source(baseline_batch, field_id)
        if any(baseline[k] != data[k] for k in ('scene_sha256', 'origin', 'metric_crs')):
            raise ValueError('RECOVERY_BASELINE_SCENE_MISMATCH')
    prior = recover_restore(batch, job)
    recovery_started = time.perf_counter()
    deadline = recovery_started + seconds
    # Leave a bounded last-row insertion opportunity rather than spending
    # the entire budget on two initial heading variants.
    result = recover_seed_and_continue(prior, settings, seconds=seconds*.8)
    done = {tid for block in result.routes if block.status=='REGION_ROUTE_PASS'
            for tid in block.task_order}
    remaining = [t for t in result.adapted_job.tasks if t.task_id not in done]
    if len(remaining)==1 and done and deadline-time.perf_counter() > .2:
        result = recover_insert_last_task(result, settings,
            seconds=min(12.0, deadline-time.perf_counter()))
    baseline_ok = None
    baseline_error = None
    if baseline is not None:
        baseline_ok = False
        if result.status == 'FIELD_ROUTE_COMPLETE':
            try:
                result = recover_preserve_required_body(result, wkt.loads(baseline['required_body_wkt']))
                baseline_ok = True
            except ValueError as exc:
                baseline_error = str(exc)
    result.elapsed_s = prior.elapsed_s + time.perf_counter()-recovery_started
    record = {
        'source_batch': str(batch),
        'source_summary_sha256': io._sha256(batch/'route_batch_summary.json'),
        'budget_seconds': seconds,
        'actual_seconds': time.perf_counter()-recovery_started,
        'source_status': prior.status, 'result_status': result.status,
        'task_set_changed': False, 'independent_headland_tasks_changed': False,
        'baseline_batch': str(baseline_batch) if baseline_batch else None,
        'baseline_body_acceptance_passed': baseline_ok,
        'baseline_body_rejection': baseline_error,
        'timing_scope': 'All recovery trials included; excludes prior solve cost',
        'failure_is_physical_impossibility': False}
    result.statistics['bounded_recovery'] = record
    out.mkdir()
    summary = {k:old[k] for k in ('schema_version', 'input_bundle', 'input_summary_sha256',
        'input_public_gpkg_sha256', 'input_metric_gpkg_sha256', 'frozen_code_sha256')}
    code = {**{name:io._sha256(recover_ROOT/'src'/name) for name in old['route_code_sha256']}, **{n:io._sha256(recover_ROOT/'src'/n) for n in (
        'route_recover.py', 'route_saved.py', 'route_continue.py', 'route_seed_continue.py',
        'route_terminal_insert.py', 'route_preserve_body.py')}}
    summary.update(status='BOUNDED_SAVED_ROUTE_RECOVERY',
        claim='ACTUAL_TASK_RECOVERY; SEPARATE_HEADLAND; EXPLICIT_GAPS',
        route_code_sha256=code, entry_code_sha256=io._sha256(recover_ROOT/'src/main.py'),
        route_settings=asdict(settings), field_count=1, region_count=len(job.regions),
        worker_count=1, source_batch=str(batch), recovery=record,
        status_counts={result.status:1}, stage_audit_status='PENDING_INDEPENDENT_AUDIT',
        planning_acceptance_passed=False, solve_seconds=time.perf_counter()-recovery_started)
    recover_export(bundle, out, [job], [result], summary, settings)
    io.atomic_json(out/'route_batch_summary.json', summary)
    audit = recover_audit_run(bundle, out, workers=1)
    summary['stage_audit_status'] = audit['status']
    with (out/'field_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); headers = tuple(reader.fieldnames); fields = list(reader)
    for row in fields:
        row['audit_status'] = audit['status']
    io._write_csv(out/'field_results.csv', fields, headers)
    import geopandas as gpd
    frame = gpd.read_file(out/'route_results.gpkg', layer='source_fields')
    frame['audit_status'] = audit['status']
    frame.to_file(out/'route_results.gpkg', layer='source_fields', driver='GPKG', index=False)
    summary['body_route_acceptance_passed'] = audit['status']=='PASS' and result.status=='FIELD_ROUTE_COMPLETE'
    summary['recovery_acceptance_passed'] = summary['body_route_acceptance_passed'] and baseline_ok is not False
    summary['planning_acceptance_passed'] = summary['recovery_acceptance_passed'] and (
        result.preparation['separate_headland']['planned_target_missing_m2'] <=
        io.load_scene(job.scene_path).settings.coverage_tolerance_m2)
    if audit['status']=='PASS':
        summary['route_overviews'] = recover_export_gallery(out, workers=1)
    summary['total_wall_seconds'] = time.perf_counter()-started
    io.atomic_json(out/'route_batch_summary.json', summary)
    print(field_id, result.status, audit['status'], json.dumps(record, ensure_ascii=False), flush=True)
    # Audit PASS includes honestly reported partial routes. It must not become
    # a successful recovery exit code. Full-target acceptance remains separate.
    return 0 if summary['recovery_acceptance_passed'] else 2

# ==========================================================================
# 8. 田头补作批处理
# 逐田提出并复核补作任务，汇总到新目录，不覆盖上游成果。
# ==========================================================================

from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import json
import math
import shutil
import time

import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.ops import unary_union

import route_planner as io
def headland_fill_batch_fill_gaps(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import fill_gaps as _operation
    return _operation(*args, **kwargs)
headland_fill_batch_audit_run = audit_run

headland_fill_batch_ROOT=Path(__file__).resolve().parents[1]


def headland_fill_batch_records(path):
    """回读历史田头补作批次的CSV表头与记录，用于保持协议字段一致。"""
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        return tuple(reader.fieldnames),list(reader)


def _worker(payload):
    """历史田头补作隔离工作入口，消费保存输入并在指定预算内求解。"""
    batch,fid,seconds=payload
    snap=json.loads((batch/'inputs'/f'{fid}.json').read_text())
    scene=io.load_scene(snap['scene_path'])
    body=unary_union([wkt.loads(t['sweep_wkt']) for t in snap['tasks']])
    prior=unary_union([body,*[wkt.loads(t['sweep_wkt']) for t in snap['separate_headland_tasks']]])
    longest={}
    for task in snap['tasks']:
        rid=task['region_id'];length=wkt.loads(task['reference_line_wkt']).length
        if rid not in longest or length>longest[rid][0]:longest[rid]=(length,task['heading_rad'])
    headings=[a for _,a in sorted(longest.values(),reverse=True)]
    tasks,diagnostic=headland_fill_batch_fill_gaps(fid,scene,prior,headings,seconds=seconds)
    return fid,tasks,diagnostic,body.wkt


def fill_headland_batch(batch,out,field_id=None,workers=12,seconds=20.0):
    """在已检查的主体之后追加历史独立田头补作，主体验收与田头新增覆盖仍分别报告。"""
    if type(workers) is not int or workers<1 or workers>64:
        raise ValueError('HEADLAND_FILL_WORKERS_MUST_BE_INTEGER_1_64')
    if isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('HEADLAND_FILL_SECONDS_MUST_BE_POSITIVE_FINITE')
    started=time.perf_counter();batch=Path(batch).resolve();out=Path(out).resolve()
    if out.exists() or not out.is_relative_to(headland_fill_batch_ROOT/'outputs'):
        raise ValueError('必须使用outputs下新目录，不覆盖旧结果')
    old=json.loads((batch/'route_batch_summary.json').read_text())
    audit=json.loads((batch/'route_independent_audit.json').read_text())
    if audit['status']!='PASS':raise ValueError('HEADLAND_FILL_SOURCE_AUDIT_FAILED')
    for key in ('route_code_sha256','frozen_code_sha256'):
        for name,digest in old[key].items():
            matches = (io._source_fingerprint_matches(headland_fill_batch_ROOT/'src'/name, digest)
                       if key == 'route_code_sha256' else io._sha256(headland_fill_batch_ROOT/'src'/name) == digest)
            if not matches:raise ValueError('HEADLAND_FILL_SOURCE_CODE_CHANGED:'+name)
    bundle=Path(old['input_bundle'])
    for name,key in [('swath_results.gpkg','input_public_gpkg_sha256'),
                     ('swath_results_metric.gpkg','input_metric_gpkg_sha256'),
                     ('swath_batch_summary.json','input_summary_sha256')]:
        if io._sha256(bundle/name)!=old[key]:raise ValueError('HEADLAND_FILL_UPSTREAM_CHANGED:'+name)
    field_headers,fields=headland_fill_batch_records(batch/'field_results.csv')
    if field_id is not None:fields=[r for r in fields if r['field_id']==field_id]
    if not fields:raise ValueError('HEADLAND_FILL_FIELD_NOT_FOUND')
    ids={r['field_id'] for r in fields}
    if len(ids)!=len(fields):raise ValueError('HEADLAND_FILL_DUPLICATE_FIELD')
    for row in fields:
        fid=row['field_id'];settings=old.get('route_settings_by_field',{}).get(fid,old['route_settings'])
        if row['field_route_status']!='FIELD_ROUTE_COMPLETE' or settings['headland_work_mode']!='SEPARATE_PASSES':
            raise ValueError('HEADLAND_FILL_REQUIRES_COMPLETE_SEPARATE_BODY:'+fid)
        path=batch/'inputs'/f'{fid}.json'
        if io._sha256(path)!=old['adapted_input_sha256'][fid]:raise ValueError('HEADLAND_FILL_SNAPSHOT_CHANGED:'+fid)
        snap=json.loads(path.read_text())
        if io._sha256(Path(snap['scene_path']))!=snap['scene_sha256']:raise ValueError('HEADLAND_FILL_SCENE_CHANGED:'+fid)
    payloads=[(batch,r['field_id'],seconds) for r in fields]
    solve_start=time.perf_counter()
    if workers==1:results=[_worker(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(_worker,payloads))
    solve_seconds=time.perf_counter()-solve_start
    out.mkdir();(out/'inputs').mkdir();(out/'motion_samples').mkdir();(out/'headland_pass_samples').mkdir()
    # Every body CSV row and pose sample is copied verbatim. Whole-batch CSVs
    # are byte-identical; a single-field selection retains original row order.
    for name in ['field_results','region_results','route_motions','route_connections',
                 'route_events','coverage_responsibilities','task_lineage']:
        if len(ids)==len(headland_fill_batch_records(batch/'field_results.csv')[1]):shutil.copy2(batch/f'{name}.csv',out/f'{name}.csv')
        else:
            header,rs=headland_fill_batch_records(batch/f'{name}.csv');io._write_csv(out/f'{name}.csv',[r for r in rs if r['field_id'] in ids],header)
    shutil.copy2(batch/'route_results.gpkg',out/'route_results.gpkg')
    if field_id is not None:
        import pyogrio
        for layer,_ in pyogrio.list_layers(out/'route_results.gpkg'):
            frame=gpd.read_file(out/'route_results.gpkg',layer=layer)
            frame[frame.field_id.isin(ids)].to_file(out/'route_results.gpkg',layer=layer,driver='GPKG',index=False)
    head_headers,aux=headland_fill_batch_records(batch/'headland_pass_motions.csv');aux=[r for r in aux if r['field_id'] in ids]
    layers={name:gpd.read_file(out/'route_results.gpkg',layer=name) for name in
        ['source_fields','headland_pass_segments','headland_pass_sweeps','headland_pass_paths','headland_pass_gaps']}
    crs=layers['source_fields'].crs;transforms={};add_segments=[];add_sweeps=[];paths=[];gaps=[];input_hashes={};stats=[]
    resultmap={r[0]:r for r in results}
    for row in fields:
        fid=row['field_id'];_,tasks,diag,body_wkt=resultmap[fid]
        snap=json.loads((batch/'inputs'/f'{fid}.json').read_text());scene=io.load_scene(snap['scene_path'])
        body=wkt.loads(body_wkt);headland=scene.target.difference(wkt.loads(snap['required_body_wkt']))
        if abs(diag['initial_target_missing_m2']-float(row['planned_target_missing_m2']))>1e-5:
            raise ValueError('HEADLAND_FILL_SOURCE_SWEEP_LEDGER:'+fid)
        prior_points=np.load(batch/'headland_pass_samples'/f'{fid}.npz',allow_pickle=False)['points']
        arrays=[prior_points];offset=len(prior_points);segment_index=sum(r['field_id']==fid for r in aux)
        world=lambda g:io._to_world(g,tuple(snap['origin']),snap['metric_crs'],str(crs),transforms)
        for task in tasks:
            points=task.motion_points
            common={'field_id':fid,'task_id':task.task_id,'pass_index':3,'segment_index':segment_index,
                'sample_offset':offset,'sample_count':len(points),'work_kind':'HEADLAND_GAP_FILL','length_m':task.length_m,
                'connection_status':'ENTRY_EXIT_NOT_PLANNED','operational_status':'PARAMETERS_UNVERIFIED'}
            aux.append(common);arrays.append(points);offset+=len(points);segment_index+=1
            line=LineString(points[:,:2]);add_segments.append({**common,'geometry':world(line)})
            add_sweeps.append({**common,'area_m2':task.frozen_sweep.area,'geometry':world(task.frozen_sweep)})
            snap['separate_headland_tasks'].append({'task_id':task.task_id,'pass_index':3,'work_kind':task.work_kind,
                'reference_line_wkt':task.reference_line.wkt,'sweep_wkt':task.frozen_sweep.wkt})
        # Explicitly distinguish supplemental lines from baseline contour count.
        diag['baseline_contour_pass_count']=snap['preparation']['separate_headland']['requested_pass_count']
        snap['preparation']['headland_gap_fill']=diag;snap['statistics']['headland_gap_fill']=diag
        combined=unary_union([body,*[wkt.loads(t['sweep_wkt']) for t in snap['separate_headland_tasks']]])
        remaining=scene.target.difference(combined);head_gap=headland.difference(combined)
        info=snap['preparation']['separate_headland']
        info.update(safe_segment_count=len(snap['separate_headland_tasks']),gap_fill_segment_count=len(tasks),
            headland_planned_covered_m2=combined.intersection(headland).area,headland_planned_missing_m2=head_gap.area,
            planned_target_missing_m2=remaining.area,coverage_status=diag['coverage_status'],
            gap_fill_status=diag['search_status'],supplements_are_extra_work_segments=True)
        info['segments_per_pass']=dict(Counter(str(t['pass_index']) for t in snap['separate_headland_tasks']))
        snap['statistics']['separate_headland']=info.copy()
        row.update(separate_headland_segment_count=len(snap['separate_headland_tasks']),
            headland_planned_missing_m2=head_gap.area,planned_target_missing_m2=remaining.area,
            headland_coverage_status=diag['coverage_status'],headland_gap_segment_count=len(tasks),
            headland_gap_recovered_m2=diag['recovered_m2'],headland_gap_search_status=diag['search_status'])
        selector=layers['source_fields'].field_id==fid
        for k in ['separate_headland_segment_count','headland_planned_missing_m2','planned_target_missing_m2',
                  'headland_coverage_status','headland_gap_segment_count','headland_gap_recovered_m2','headland_gap_search_status']:
            layers['source_fields'].loc[selector,k]=row[k]
        if not head_gap.is_empty:gaps.append({'field_id':fid,'area_m2':head_gap.area,'geometry':world(head_gap)})
        for index in range(1,info['requested_pass_count']+1):
            chosen=[world(wkt.loads(t['reference_line_wkt'])) for t in snap['separate_headland_tasks'] if t['pass_index']==index]
            if chosen:paths.append({'field_id':fid,'pass_index':index,'segment_count':len(chosen),
                'connection_status':'BASE_CONTOUR_AND_INDEPENDENT_SUPPLEMENTS' if index==3 and tasks else 'SEGMENTS_NOT_FORCED_CONNECTED',
                'geometry':MultiLineString(chosen)})
        np.savez_compressed(out/'headland_pass_samples'/f'{fid}.npz',points=np.vstack(arrays))
        shutil.copy2(batch/'motion_samples'/f'{fid}.npz',out/'motion_samples'/f'{fid}.npz')
        io.atomic_json(out/'inputs'/f'{fid}.json',snap);input_hashes[fid]=io._sha256(out/'inputs'/f'{fid}.json')
        stats.append({'field_id':fid,**diag})
    for name,rs in [('headland_pass_segments',add_segments),('headland_pass_sweeps',add_sweeps)]:
        if rs:gpd.GeoDataFrame(rs,geometry='geometry',crs=crs).to_file(out/'route_results.gpkg',layer=name,driver='GPKG',mode='a',index=False)
    for name,rs in [('headland_pass_paths',paths),('headland_pass_gaps',gaps),
                    ('headland_gap_fill_segments',add_segments),('headland_gap_fill_sweeps',add_sweeps)]:
        frame=(gpd.GeoDataFrame(rs,geometry='geometry',crs=crs) if rs else
               gpd.GeoDataFrame({'field_id':[],'geometry':gpd.GeoSeries([],crs=crs)},geometry='geometry',crs=crs))
        frame.to_file(out/'route_results.gpkg',layer=name,driver='GPKG',index=False)
    layers['source_fields'].to_file(out/'route_results.gpkg',layer='source_fields',driver='GPKG',index=False)
    new_columns=('headland_gap_segment_count','headland_gap_recovered_m2','headland_gap_search_status')
    io._write_csv(out/'field_results.csv',fields,tuple(dict.fromkeys((*field_headers,*new_columns))))
    io._write_csv(out/'headland_pass_motions.csv',aux,head_headers)
    io.atomic_json(out/'headland_gap_fill_statistics.json',stats)
    summary={**old,'status':'HEADLAND_GAP_FILL_PENDING_AUDIT','field_count':len(fields),'worker_count':workers,
        'claim':'UNCHANGED_CONTINUOUS_BODY; BASE_CONTOURS_PLUS_INDEPENDENT_SUPPLEMENTAL_WORK; EXPLICIT_RESIDUALS',
        'source_batch':str(batch),'source_summary_sha256':io._sha256(batch/'route_batch_summary.json'),
        'route_code_sha256':{**{name:io._sha256(headland_fill_batch_ROOT/'src'/name) for name in old['route_code_sha256']},**{name:io._sha256(headland_fill_batch_ROOT/'src'/name) for name in
            ['route_headland_fill.py','route_headland_fill_batch.py']}},
        'adapted_input_sha256':input_hashes,'status_counts':dict(Counter(r['field_route_status'] for r in fields)),
        'headland_gap_fill_solve_seconds':solve_seconds,'stage_audit_status':'PENDING_INDEPENDENT_AUDIT',
        'planning_acceptance_passed':False,'display_gpkg':str(out/'route_results.gpkg'),
        'headland_gap_segment_count':sum(d['gap_segment_count'] for d in stats),
        'headland_gap_recovered_m2':sum(d['recovered_m2'] for d in stats),
        'planned_target_missing_m2':sum(float(r['planned_target_missing_m2']) for r in fields),
        'completed_task_count':sum(int(r['completed_task_count']) for r in fields),
        'replanned_task_count':sum(int(r['replanned_task_count']) for r in fields),
        'region_count':len(headland_fill_batch_records(out/'region_results.csv')[1]),
        'body_sample_files_identical':all(io._sha256(batch/'motion_samples'/f'{fid}.npz')==
            io._sha256(out/'motion_samples'/f'{fid}.npz') for fid in ids)}
    for key in ['route_overviews','total_wall_seconds','export_counts','source_batches','candidate_selection']:
        summary.pop(key,None)
    if 'route_settings_by_field' in summary:
        summary['route_settings_by_field']={fid:summary['route_settings_by_field'][fid] for fid in ids}
    io.atomic_json(out/'route_batch_summary.json',summary)
    t=time.perf_counter();checked=headland_fill_batch_audit_run(bundle,out,workers=workers)
    summary.update(stage_audit_status=checked['status'],headland_fill_audit_seconds=time.perf_counter()-t,
        body_route_acceptance_passed=checked['status']=='PASS' and summary['body_sample_files_identical'],
        planning_acceptance_passed=checked['status']=='PASS' and all(r['headland_coverage_status']=='COVERAGE_COMPLETE' for r in fields),
        status='HEADLAND_GAP_FILL_AUDIT_'+checked['status'],total_wall_seconds=time.perf_counter()-started)
    io.atomic_json(out/'route_batch_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','field_count','headland_gap_segment_count',
        'headland_gap_recovered_m2','planned_target_missing_m2','total_wall_seconds','planning_acceptance_passed']},ensure_ascii=False),flush=True)
    return 0 if checked['status']=='PASS' else 2

# ==========================================================================
# 9. 规则路线的独立验收和展示
# 核验导出行程、车辆操作、扫掠与账本，并生成规则路线综合图。
# ==========================================================================

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import replace
import csv
import json
import math
import os
from pathlib import Path
import time

import geopandas as gpd
import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.ops import transform
from pyproj import Transformer

from scene import Motion as regular_audit_Motion
from scene import load_scene as regular_audit_load_scene
from scene import wrap as regular_audit_wrap
from validator import dense_points as regular_audit_dense_points
from validator import swept_polygons as regular_audit_swept_polygons
from validator import kinematic_issues as regular_audit_kinematic_issues
from validator import sweep_part_groups as regular_audit_sweep_part_groups
import route_planner as io
def regular_audit_gear_statistics(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import gear_statistics as _operation
    return _operation(*args, **kwargs)
def regular_audit_port_window(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import port_window as _operation
    return _operation(*args, **kwargs)
def regular_audit_template_class(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import template_class as _operation
    return _operation(*args, **kwargs)
def regular_audit_transfer_class(*args, **kwargs):
    """延迟调用策略层，避免批次管理与算法模块循环初始化。"""
    from route import transfer_class as _operation
    return _operation(*args, **kwargs)


def actual_motion_parts(motion):
    """提取正长度实际运动段用于规则路线独立复算，不累计停车点为行驶距离。
    
    A gear cusp at rest is an operation event, not a zero-length route.
    
    Its states remain in NPZ and its gear/steering events remain in JSON.
    A duplicate-point LineString is invalid and may serialize as a Point;
    exporting it as a movement would invent a spatial route part."""
    return [(start,gear,line) for start,gear,line in io._motion_parts(motion) if line.length>1e-7]


def independent_coverage(motion,scene):
    """从开启机具的保存运动独立计算作业覆盖；关闭机具移动返回空覆盖。"""
    if not motion.implement_on:
        return GeometryCollection()
    p=motion.points
    if np.max(np.abs([regular_audit_wrap(float(h-p[0,2])) for h in p[:,2]]))<1e-8:
        return unary_union(list(regular_audit_swept_polygons(motion,scene,work_only=True)))
    p=regular_audit_dense_points(p,scene.settings.sampling_step_m,scene.settings.heading_step_rad,
        scene.settings.max_motion_samples)
    local=scene.vehicle.rectangles(work_only=True)[0]
    # This audit computes occupied rectangles, not planner-generated sweeps
    # and not hulls joining poses (which overclaim curved-work coverage).
    c=np.cos(p[:,2,None]);s=np.sin(p[:,2,None])
    x=p[:,0,None]+c*local[:,0]-s*local[:,1]
    y=p[:,1,None]+s*local[:,0]+c*local[:,1]
    return shapely.union_all(shapely.polygons(np.stack((x,y),axis=2)))


def audit_saved(npz_path,records,source_scene,selected_radius,expected_task_ids,
                head_supplied,head_failures,planner_failures,derived_obligations=None):
    """回放规则路线的保存运动与车辆假设，独立核查包络、时间、任务和覆盖，不提升为实车认证。"""
    started=time.perf_counter()
    issues=[];source_issues=[];worked=GeometryCollection();head_components={}
    scene=replace(source_scene,vehicle=replace(source_scene.vehicle,
        min_turn_radius_m=selected_radius))
    if not 4<=selected_radius<=6 or type(selected_radius) is bool:
        issues.append({'code':'INVALID_MODEL_RADIUS'})
    eps=scene.settings.geometry_epsilon_m;tol=scene.settings.coverage_tolerance_m2
    body_previous=None;body_seen=False;done=[];sums=Counter();head_motions={}
    previous_block=None;region_sweeps={}
    head_worked=GeometryCollection();body_sweep_records=[];borrow_requests=[]
    with np.load(npz_path,allow_pickle=False) as saved:
        if set(saved.files)!={r['array_key'] for r in records}:
            issues.append({'code':'MOTION_ARRAY_MANIFEST_MISMATCH'})
        for record in records:
            motion=regular_audit_Motion(saved[record['array_key']],record['kind'],
                record['source_task_id'],record['implement_on'])
            code=record['motion_id']
            kin=regular_audit_kinematic_issues(motion,scene)
            issues.extend({'code':i,'motion_id':code} for i in kin)
            source_issues.extend({'code':i,'motion_id':code}
                for i in regular_audit_kinematic_issues(motion,source_scene))
            _,_,_,expected_operations,_=io._gear_and_steering(motion,scene,io.RouteSettings())
            recorded_operations=(record.get('turn_evidence') or {}).get('operations',record.get('operations',[]))
            if recorded_operations!=expected_operations:
                issues.append({'code':'STOP_STEER_OPERATION_MISMATCH','motion_id':code})
            body,tool=regular_audit_sweep_part_groups(motion,scene)
            safe_body=shapely.buffer(body,scene.vehicle.safety_margin_m/math.cos(math.pi/32),quad_segs=8)
            if not np.all(shapely.covers(scene.travel.buffer(eps),safe_body)):
                issues.append({'code':'BODY_COLLISION_OR_BOUNDARY','motion_id':code})
            if not np.all(shapely.covers(scene.target.buffer(eps),tool)):
                issues.append({'code':'IMPLEMENT_COLLISION_OR_BOUNDARY','motion_id':code})
            phase=record['phase'];a=io._pose_at(motion,False);b=io._pose_at(motion,True)
            if phase=='HEADLAND':
                if body_seen:
                    issues.append({'code':'HEADLAND_AFTER_BODY','motion_id':code})
                component=record['component_id']
                previous=head_components.get(component)
                if previous is not None and not io._pose_close(previous,a,scene):
                    issues.append({'code':'HEADLAND_DISCONTINUITY','motion_id':code})
                head_components[component]=b
                head_motions.setdefault(component,[]).append(motion)
            elif phase=='BODY':
                body_seen=True
                if body_previous is not None and not io._pose_close(body_previous,a,scene):
                    issues.append({'code':'BODY_DISCONTINUITY','motion_id':code})
                body_previous=b
                if motion.implement_on:
                    done.append(record['source_task_id'])
                elif previous_block is not None and record.get('block_id') is not None:
                    different=previous_block!=record['block_id']
                    if (motion.kind=='transfer')!=different:
                        issues.append({'code':'BLOCK_TRANSFER_DECLARATION_MISMATCH','motion_id':code})
                previous_block=record.get('block_id')
            else:
                issues.append({'code':'UNKNOWN_PHASE','motion_id':code})
            if not motion.implement_on:
                actual_ready=worked.union(scene.travel.difference(scene.target)).buffer(eps)
                if not np.all(shapely.covers(actual_ready,np.concatenate((body,tool)))):
                    issues.append({'code':'UNWORKED_CROSSING','motion_id':code})
                evidence=record.get('turn_evidence') or {}
                shape=(regular_audit_transfer_class(motion.points) if motion.kind=='transfer' else
                    regular_audit_template_class(motion.points,regular_audit_wrap(b.yaw-a.yaw)))
                if shape is None:
                    issues.append({'code':'NON_REGULAR_TURN','motion_id':code})
                runs,shifts,reverse=regular_audit_gear_statistics(motion.points)
                if evidence.get('gear_runs')!=runs or evidence.get('gear_shifts')!=shifts:
                    issues.append({'code':'GEAR_EVIDENCE_MISMATCH','motion_id':code})
                if motion.kind!='transfer':
                    window=regular_audit_port_window(a,b,scene)
                    if window.is_empty or not np.all(shapely.covers(window.buffer(eps),
                            np.concatenate((body,tool)))):
                        issues.append({'code':'OUTSIDE_LOCAL_PORT','motion_id':code})
            sweep=independent_coverage(motion,scene)
            if phase=='BODY' and motion.implement_on and record.get('region_id') is not None:
                region_sweeps.setdefault(record['region_id'],[]).append(sweep)
            if phase=='HEADLAND' and motion.implement_on:
                head_worked=head_worked.union(sweep)
            if phase=='BODY' and motion.implement_on:
                if derived_obligations is not None:
                    obligation=derived_obligations.get(record['source_task_id'],GeometryCollection()).difference(head_worked)
                    prior=obligation.intersection(worked)
                    declared=record.get('prior_required_covered_m2')
                    if (type(declared) not in (int,float) or not math.isfinite(declared) or
                            abs(declared-prior.area)>tol):
                        issues.append({'code':'PRIOR_CROP_SUPPLY_MISMATCH','motion_id':code})
                    if obligation.difference(worked.union(sweep)).area>tol:
                        issues.append({'code':'BODY_TASK_CROP_GAP','motion_id':code})
                    if prior.area>tol:
                        borrow_requests.append({'record_index':len(body_sweep_records),
                            'owner_task_id':record['source_task_id'],'owner_region_id':record.get('region_id'),
                            'geometry':prior})
                body_sweep_records.append({'task_id':record['source_task_id'],
                    'region_id':record.get('region_id'),'geometry':sweep})
            if motion.implement_on:
                sums['work_footprint_sum_m2']+=sweep.intersection(scene.target).area
                worked=worked.union(sweep)
            key='work_m' if motion.implement_on else 'off_m'
            sums[key]+=motion.length
            if not motion.implement_on:
                sums['max_off_m']=max(sums['max_off_m'],motion.length)
    for component,motions in head_motions.items():
        if not io._pose_close(io._pose_at(motions[0],False),io._pose_at(motions[-1],True),scene):
            issues.append({'code':'HEADLAND_NOT_CLOSED','component_id':component})
    expected=set(expected_task_ids);covered_tasks=set(done)|set(head_supplied)
    missing_tasks=sorted(expected-covered_tasks)
    if len(done)!=len(set(done)):
        issues.append({'code':'DUPLICATE_BODY_TASK'})
    if set(done)-expected or set(head_supplied)-expected:
        issues.append({'code':'UNKNOWN_SOURCE_TASK'})
    gap=source_scene.target.difference(worked)
    covered=source_scene.target.intersection(worked)
    ledger_delta=abs(source_scene.target.area-covered.area-gap.area)
    crop_supplies=[]
    if borrow_requests and body_sweep_records:
        tree=shapely.STRtree([r['geometry'] for r in body_sweep_records])
        for request in borrow_requests:
            remaining=request['geometry']
            for index in sorted(tree.query(remaining,predicate='intersects')):
                if index>=request['record_index']:
                    continue
                provider=body_sweep_records[index]
                part=remaining.intersection(provider['geometry'])
                if part.area<=tol:
                    continue
                remaining=remaining.difference(part)
                if provider['region_id']!=request['owner_region_id']:
                    crop_supplies.append({'owner_task_id':request['owner_task_id'],
                        'owner_region_id':request['owner_region_id'],
                        'provider_task_id':provider['task_id'],'provider_region_id':provider['region_id'],
                        'provider_body_order':int(index),'owner_body_order':request['record_index'],
                        'area_m2':part.area,'scope':'DERIVED_CORE_OBLIGATION','geometry_wkt':part.wkt})
    body_by_region=[unary_union(parts).intersection(source_scene.target)
        for parts in region_sweeps.values()]
    body_union=unary_union(body_by_region)
    # Derived ON strokes may reach the real headland across a frozen seam.
    # Recompute the physical extra passes; upstream seam PASS cannot certify
    # these changed strokes. Three-way overlap counts actual extra operations.
    sums['body_cross_region_extra_m2']=max(0.,sum(g.area for g in body_by_region)-body_union.area)
    sums['body_footprint_union_m2']=body_union.area
    coverage_passed=gap.area<=tol
    accounting_passed=ledger_delta<=tol
    acceptance=not issues and not missing_tasks and not head_failures and not planner_failures and coverage_passed and accounting_passed
    codes={i['code'] for i in issues}
    status=lambda relevant:('NOT_EVALUATED' if not records else 'FAIL' if codes.intersection(relevant) else 'PASS')
    return {'acceptance_passed':acceptance,'motion_audit_passed':bool(records) and not issues,
        'motion_audit_scope':'EMITTED_EVENTS_ONLY',
        'headland_self_continuity':'FAIL' if head_failures else
            ('NOT_EVALUATED' if not head_components else status({'HEADLAND_DISCONTINUITY','HEADLAND_NOT_CLOSED'})),
        'body_continuity':'NOT_EVALUATED' if not body_seen else status({'BODY_DISCONTINUITY'}),
        'dynamic_access_status':status({'UNWORKED_CROSSING','HEADLAND_AFTER_BODY'}),
        'turn_regularity_status':status({'NON_REGULAR_TURN','OUTSIDE_LOCAL_PORT','GEAR_EVIDENCE_MISMATCH',
            'BLOCK_TRANSFER_DECLARATION_MISMATCH'}),
        'body_task_coverage_status':('NOT_EVALUATED' if not body_sweep_records or
            derived_obligations is None else status({'BODY_TASK_CROP_GAP','PRIOR_CROP_SUPPLY_MISMATCH'})),
        'body_crop_supply_count':len(crop_supplies),
        'body_crop_supplied_by_neighbours_m2':sum(r['area_m2'] for r in crop_supplies),
        'body_crop_supplies':crop_supplies,
        'design_parameter_check':status({'INVALID_MODEL_RADIUS','BODY_COLLISION_OR_BOUNDARY',
            'IMPLEMENT_COLLISION_OR_BOUNDARY','HEADING_JUMP','LATERAL_JUMP','CURVATURE_LIMIT',
            'CURVATURE_RATE','INVALID_MOTION_STATE','REVERSE_FORBIDDEN','REVERSE_WORK',
            'STOP_STEER_OPERATION_MISMATCH'}),
        'coverage_status':'PASS' if coverage_passed else 'FAIL',
        'source_parameters_motion_passed':bool(records) and not source_issues,
        'source_parameter_issues':source_issues,'issues':issues,'missing_task_ids':missing_tasks,
        'body_completed_count':len(done),'expected_task_count':len(expected),
        'headland_supplied_count':len(head_supplied),'headland_component_count':len(head_components),
        'target_area_m2':source_scene.target.area,'target_gap_m2':gap.area,
        'target_coverage_fraction':covered.area/source_scene.target.area,
        'area_ledger_delta_m2':ledger_delta,'coverage_passed':coverage_passed,
        'area_ledger_passed':accounting_passed,'worked':worked,'gap':gap,
        'audit_seconds':time.perf_counter()-started,**dict(sums)}


def _regular_audit_records(result):
    """将规则结果中的田头与主体运动转换为审计记录，保留数组索引及动作顺序。"""
    records=[];motions=[]
    for component in result['heads']:
        for motion in component['motions']:
            records.append({'phase':'HEADLAND','component_id':component['component_id'],
                'pass_index':component['pass_index'],'block_id':None,'region_id':None,
                'source_task_id':motion.task_id,'kind':motion.kind,'implement_on':motion.implement_on,
                'operations':getattr(motion,'operation_events',[]),
                'curve_family':getattr(motion,'curve_family','F2C_CC_OR_GEOMETRIC_SMOOTH_CONTOUR')})
            motions.append(motion)
    for event in result['events']:
        motion=event['motion']
        record={k:v for k,v in event.items() if k!='motion'}
        record.update(component_id=None,pass_index=None,kind=motion.kind,
            implement_on=motion.implement_on,
            operations=io._gear_and_steering(motion,result['scene'],io.RouteSettings())[3])
        records.append(record);motions.append(motion)
    for i,record in enumerate(records):
        record.update(motion_id=f'm{i:05d}',array_key=f'p{i:05d}',sequence=i)
    return records,motions


def _draw_geom(ax,geom,facecolor,alpha=.5,edgecolor=None):
    """递归绘制面/线等几何，空对象跳过，绘图不参与验收计算。"""
    if geom.is_empty:
        return
    if geom.geom_type=='Polygon':
        x,y=geom.exterior.xy;ax.fill(x,y,color=facecolor,alpha=alpha)
        if edgecolor:
            ax.plot(x,y,color=edgecolor,lw=.6)
        for ring in geom.interiors:
            x,y=ring.xy;ax.fill(x,y,color='white')
    elif hasattr(geom,'geoms'):
        for g in geom.geoms:
            _draw_geom(ax,g,facecolor,alpha,edgecolor)


def draw_field(job,result,audit,path):
    """绘制规则路线及缺失区域，缓存放输出目录，图像只是诊断材料。"""
    cache=Path(path).parents[1]/'plot_cache'
    cache.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    os.environ['XDG_CACHE_HOME']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(2,2,figsize=(13,11),layout='constrained')
    scene=result['scene']
    for ax in axes.flat:
        _draw_geom(ax,scene.target,'#eeeeee',1,'#444444')
        ax.set_aspect('equal');ax.set_xlabel('Local x (m)');ax.set_ylabel('Local y (m)')
    _draw_geom(axes[0,0],scene.travel,'#a1dab4',.6)
    axes[0,0].set_title('1. Scene: target and vehicle travel')
    palette=['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462']
    for i,region in enumerate(job.regions):
        _draw_geom(axes[0,1],region.geometry,palette[i%len(palette)],.7,'#555555')
        center=region.geometry.representative_point()
        axes[0,1].text(center.x,center.y,region.region_id,fontsize=7,ha='center')
    axes[0,1].set_title('2. Frozen partitions')
    for task in job.tasks:
        x,y=task.reference_line.xy;axes[1,0].plot(x,y,c='#367daf',lw=.65)
    axes[1,0].set_title('3. Frozen body swaths (source)')
    ax=axes[1,1]
    for component in result['heads']:
        for motion in component['motions']:
            ax.plot(motion.points[:,0],motion.points[:,1],c='#8451a1',lw=.8)
    for event in result['events']:
        motion=event['motion'];p=motion.points
        color='#1b9e77' if motion.implement_on else '#e69f00'
        ax.plot(p[:,0],p[:,1],c=color,lw=.75)
        if not motion.implement_on:
            for start,gear,part in actual_motion_parts(motion):
                if gear<0:
                    x,y=part.xy
                    ax.plot(x,y,c='#d95f02',lw=1.)
    if result['events']:
        p=result['events'][0]['motion'].points[0]
        ax.scatter(p[0],p[1],marker='o',c='black',s=20)
        p=result['events'][-1]['motion'].points[-1]
        ax.scatter(p[0],p[1],marker='x',c='black',s=25)
    ax.set_title('4. Actual accepted prefix / independent headlands')
    ax.legend(handles=[Line2D([],[],color=c,label=l) for c,l in
        [('#8451a1','Headlands first'),('#1b9e77','Body work'),('#e69f00','OFF turn/transfer'),('#d95f02','Reverse')]],fontsize=7,loc='best')
    label='PASS' if audit['acceptance_passed'] else 'INCOMPLETE'
    fig.suptitle(f"{job.field_id} | {label} | model R={result['radius_m']:g}m | "
        f"body {audit['body_completed_count']}/{audit['expected_task_count']} | "
        f"original target gap {audit['target_gap_m2']:.2f} m2",fontsize=12)
    fig.savefig(path,dpi=140);plt.close(fig)


def _audit_one(job,result,out):
    """保存一田规则运动采样并立即复核，数组归档与任务记录必须能够对应。"""
    folder=out/'fields'/job.field_id;folder.mkdir()
    records,motions=_regular_audit_records(result)
    np.savez_compressed(folder/'motions.npz',**{r['array_key']:m.points for r,m in zip(records,motions)})
    (folder/'motions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    records=json.loads((folder/'motions.json').read_text())
    source_scene=regular_audit_load_scene(job.scene_path)
    width=result.get('derived_headland_width_m')
    valid_width=type(width) in (int,float) and math.isfinite(width) and width>0
    body_core=source_scene.target.buffer(-width) if valid_width else GeometryCollection()
    obligations={t.task_id:t.frozen_sweep.intersection(body_core) for t in job.tasks}
    audit=audit_saved(folder/'motions.npz',records,source_scene,result['radius_m'],
        [t.task_id for t in job.tasks],result['headland_supplied_task_ids'],
        result['head_failures'],result['failures'],derived_obligations=obligations)
    derived=set(result.get('derived_body_task_ids',[]))
    done={r['source_task_id'] for r in records if r['phase']=='BODY' and r['implement_on']}
    # Recounting IDs alone cannot certify the changed strokes' crop coverage.
    # Replay the declared phase boundary against the frozen source footprints
    # and independent actual coverage. Whole original target stays separate.
    required_union=unary_union([g for task_id,g in obligations.items() if task_id in derived])
    audit['body_obligation_gap_m2']=required_union.difference(audit['worked']).area if valid_width else None
    audit['body_obligation_coverage_passed']=(valid_width and audit['body_obligation_gap_m2']<=
        source_scene.settings.coverage_tolerance_m2)
    audit['derived_body_task_count']=len(derived)
    audit['derived_body_complete']=(bool(derived) and derived==done and
        audit['body_continuity']=='PASS' and audit['motion_audit_passed'] and
        audit['body_obligation_coverage_passed'] and
        not any(f.get('reason') in {'NO_SAFE_DERIVED_WORK_VARIANT',
            'SEARCH_BUDGET_EXHAUSTED_WORK_PREPARATION'} for f in result['failures']))
    audit['missing_derived_body_task_ids']=sorted(derived-done)
    draw_field(job,result,audit,out/'images'/f'{job.field_id}_four_stages.png')
    return records,audit


def export_and_audit(jobs,results,out,settings,errors,workers=12):
    """导出规则路线空间结果、时间和独立审计；该历史策略不使用默认精简参考导出协议。"""
    started=time.perf_counter();fieldrows=[];linerows=[];targetrows=[];regionrows=[];gaprows=[];sweeprows=[]
    audits=[];transformers={}
    (out/'fields').mkdir();(out/'images').mkdir()
    audited={}
    # Build one shared font cache before spawning plot workers; otherwise all
    # twelve children scan the same system fonts at the same time.
    cache=out/'plot_cache';cache.mkdir()
    os.environ['MPLCONFIGDIR']=str(cache);os.environ['XDG_CACHE_HOME']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    font_manager.fontManager
    if workers==1:
        for job in jobs:
            if job.field_id in results:
                audited[job.field_id]=_audit_one(job,results[job.field_id],out)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(_audit_one,j,results[j.field_id],out):j.field_id
                for j in jobs if j.field_id in results}
            for future in as_completed(pending):
                fid=pending[future]
                try:
                    audited[fid]=future.result()
                except Exception as exc:
                    errors.append({'field_id':fid,'error_type':type(exc).__name__,
                        'message':str(exc),'stage':'EXPORT_AUDIT'})
                if len(audited)%10==0:
                    print(f'audit/images {len(audited)}/{len(jobs)}',flush=True)
    for index,job in enumerate(jobs):
        if job.field_id not in audited:
            continue
        result=results[job.field_id];folder=out/'fields'/job.field_id
        records,audit=audited[job.field_id]
        _,motions=_regular_audit_records(result)
        # Audit is run on saved arrays, and these are the public values. Do not
        # stamp a planner flag on a physically or geometrically incomplete job.
        row={'field_id':job.field_id,'acceptance_passed':audit['acceptance_passed'],
            'radius_m':result['radius_m'],'source_radius_m':result['source_scene'].vehicle.min_turn_radius_m,
            'model_status':('SOURCE_PARAMETERS' if result['scene'].vehicle==result['source_scene'].vehicle
                else 'HYPOTHETICAL_PARAMETERS'),'skip':result['skip'],
            'headland_placement_m':result.get('headland_placement_m'),
            'derived_headland_width_m':result.get('derived_headland_width_m'),
            'search_status':result['search_status'],'input_seam_quality_status':result['input_seam_quality_status'],
            'switch_lag_status':'PENDING_MEASUREMENT','planning_seconds':result['planning_seconds']}
        row.update({k:v for k,v in audit.items() if type(v) in (bool,int,float,str)})
        fieldrows.append(row)
        detail={**row,'failures':result['failures'],'audit_issues':audit['issues'],
            'source_parameter_issues':audit['source_parameter_issues'],
            'body_crop_supplies':audit['body_crop_supplies'],
            'missing_task_ids':audit['missing_task_ids'],'blocks':result['blocks'],
            'derived_body_task_ids':result.get('derived_body_task_ids',[]),
            'missing_derived_body_task_ids':audit['missing_derived_body_task_ids'],
            'headland_supplied_task_ids':result['headland_supplied_task_ids'],
            'headland_pending_task_ids':result['headland_pending_task_ids'],
            'profiles':result['profiles'],'scene_path':job.scene_path,
            'original_vehicle':asdict(result['source_scene'].vehicle),
            'model_vehicle':asdict(result['scene'].vehicle),
            'phase_connections_status':'HEADLAND_COMPONENTS_INDEPENDENT_BY_USER_RULE'}
        (folder/'result.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
        source=regular_audit_load_scene(job.scene_path)
        def world(g):
            exported=io._to_world(g,job.origin,job.metric_crs,job.source_crs,transformers)
            # Translating thousands of touching curve rectangles can produce
            # a subnanometre self-touch. Repair the serialization geometry;
            # the metric roundtrip audit still checks its real area change.
            return shapely.make_valid(exported) if not exported.is_valid else exported
        targetrows.append({**row,'geometry':world(source.target)})
        for region in job.regions:
            regionrows.append({'field_id':job.field_id,'region_id':region.region_id,
                'geometry':world(region.geometry)})
        if not audit['gap'].is_empty:
            gaprows.append({'field_id':job.field_id,'area_m2':audit['gap'].area,'geometry':world(audit['gap'])})
        if not audit['worked'].is_empty:
            sweeprows.append({'field_id':job.field_id,'area_m2':audit['worked'].area,'geometry':world(audit['worked'])})
        for record,motion in zip(records,motions):
            for start,gear,geometry in actual_motion_parts(motion):
                linerows.append({'field_id':job.field_id,'motion_id':record['motion_id'],
                    'sequence':record['sequence'],'phase':record['phase'],
                    'component_id':record['component_id'],'pass_index':record['pass_index'],
                    'block_id':record['block_id'],'region_id':record['region_id'],
                    'source_task_id':record['source_task_id'],'implement_on':record['implement_on'],
                    'gear':gear,'kind':record['kind'],
                    'part_start_index':start,
                    'length_m':geometry.length,'geometry':world(geometry)})
        audits.append(audit)
        if (index+1)%10==0:
            print(f'audit/export {index+1}/{len(jobs)}',flush=True)
    for layer,rows in [('source_fields',targetrows),('work_regions',regionrows),
                       ('regular_routes',linerows),('actual_worked',sweeprows),('target_gaps',gaprows)]:
        if rows:
            gpd.GeoDataFrame(rows,geometry='geometry',crs=jobs[0].source_crs).to_file(
                out/'regular_route_results.gpkg',layer=layer,driver='GPKG',index=False)
    roundtrip_started=time.perf_counter()
    roundtrip=verify_exported_gpkg(out/'regular_route_results.gpkg',jobs,audited)
    (out/'gpkg_roundtrip_audit.json').write_text(json.dumps(roundtrip,ensure_ascii=False,indent=2))
    if fieldrows:
        columns=sorted(set().union(*(r.keys() for r in fieldrows)))
        with (out/'field_results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,columns);writer.writeheader();writer.writerows(fieldrows)
    (out/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
    (out/'field_results.json').write_text(json.dumps(fieldrows,ensure_ascii=False,indent=2))
    gallery=['<!doctype html><meta charset="utf-8"><title>Regular routes</title>',
        '<h1>Regular headland-first routes</h1><p>INCOMPLETE is not route acceptance.</p>']
    for row in fieldrows:
        fid=row['field_id'];gallery.append(f'<h2>{fid}: {"PASS" if row["acceptance_passed"] else "INCOMPLETE"}</h2>'
            f'<img loading="lazy" width="900" src="images/{fid}_four_stages.png">')
    (out/'gallery.html').write_text('\n'.join(gallery))
    return {'acceptance_passed':not errors and roundtrip['passed'] and len(audits)==len(jobs) and all(a['acceptance_passed'] for a in audits),
        'exported_field_count':len(audits),'error_count':len(errors),
        'accepted_field_count':sum(a['acceptance_passed'] for a in audits),
        'motion_audit_passed_count':sum(a['motion_audit_passed'] for a in audits),
        'body_complete_field_count':sum(not a['missing_task_ids'] for a in audits),
        'derived_body_complete_field_count':sum(a['derived_body_complete'] for a in audits),
        'gpkg_roundtrip_passed':roundtrip['passed'],
        'gpkg_roundtrip_seconds':time.perf_counter()-roundtrip_started,
        'coverage_passed_field_count':sum(a['coverage_passed'] for a in audits),
        'target_gap_total_m2':sum(a['target_gap_m2'] for a in audits),
        'export_audit_wall_seconds':time.perf_counter()-started,
        'audit_cpu_seconds_sum':sum(a['audit_seconds'] for a in audits),
        'design_scope':'HEADLAND_COMPONENTS_FIRST_THEN_CONTINUOUS_BODY',
        'physical_vehicle_certification':'NOT_REAL_VEHICLE_CERTIFIED',
        'implement_switch_lag_status':'PENDING_MEASUREMENT'}


def verify_exported_gpkg(path,jobs,audited):
    """回读规则GPKG空间图层验证字段、几何和任务归属，不能只依据内存结果声明导出正确。
    
    Reproject the public GPKG and compare against saved-pose replay.
    
    This checks coordinate/geometry export and the actual original-target
    difference. Motion continuity/gear/steering are checked from NPZ, because
    a 2D LineString alone cannot establish those properties."""
    available=set(__import__('pyogrio').list_layers(path)[:,0])
    layers={name:gpd.read_file(path,layer=name) for name in
        ['source_fields','work_regions','target_gaps','actual_worked','regular_routes'] if name in available}
    findings=[]
    source=layers['source_fields']
    if len(source)!=len(jobs) or set(source.field_id)!={j.field_id for j in jobs}:
        findings.append({'code':'EXPORTED_SOURCE_FIELD_SET_MISMATCH'})
    lookup={name:{str(fid):part for fid,part in frame.groupby('field_id')} for name,frame in layers.items()}
    route_parts={}
    if 'regular_routes' in layers:
        route_parts={key:frame for key,frame in layers['regular_routes'].groupby(['field_id','motion_id'])}
    rows=[]
    for job in jobs:
        if job.field_id not in audited:
            rows.append({'field_id':job.field_id,'passed':False,'reason':'MISSING_POSE_AUDIT'});continue
        scene=regular_audit_load_scene(job.scene_path);audit=audited[job.field_id][1]
        invalid_count=0
        for name in layers:
            frame=lookup.get(name,{}).get(job.field_id)
            if frame is not None:
                invalid_count+=sum(not g.is_valid for g in frame.geometry)
        local_repair_count=0
        converter=(None if io._crs_equiv(job.source_crs,job.metric_crs) else
            Transformer.from_crs(job.source_crs,job.metric_crs,always_xy=True))
        def local(geometry):
            nonlocal local_repair_count
            if converter is not None:
                geometry=transform(converter.transform,geometry)
            geometry=translate(geometry,-job.origin[0],-job.origin[1])
            if not geometry.is_valid:
                local_repair_count+=1
                geometry=shapely.make_valid(geometry)
            return geometry
        def geometries(layer):
            frame=lookup.get(layer,{}).get(job.field_id)
            return GeometryCollection() if frame is None else unary_union([local(g) for g in frame.geometry])
        target=geometries('source_fields');worked=geometries('actual_worked');gap=geometries('target_gaps')
        recomputed=scene.target.difference(worked)
        target_delta=target.symmetric_difference(scene.target).area
        gap_delta=abs(recomputed.area-audit['target_gap_m2'])
        gap_geometry_delta=gap.symmetric_difference(recomputed).area
        region_delta=max([0.]+[geometries_for_region(job,r,layers['work_regions'],local).symmetric_difference(r.geometry).area for r in job.regions])
        records=audited[job.field_id][0];line_error=0.;part_error=False
        with np.load(path.parent/'fields'/job.field_id/'motions.npz',allow_pickle=False) as arrays:
            for record in records:
                motion=regular_audit_Motion(arrays[record['array_key']],record['kind'],record['source_task_id'],record['implement_on'])
                expected=actual_motion_parts(motion)
                if not expected:
                    continue
                frame=route_parts.get((job.field_id,record['motion_id']))
                if frame is None or len(frame)!=len(expected):
                    part_error=True;continue
                frame=frame.sort_values('part_start_index')
                for (begin,gear,line),(_,exported) in zip(expected,frame.iterrows()):
                    actual=local(exported.geometry)
                    # GPKG preserves vertex order. Comparing paired vertices
                    # is stricter and linear; Hausdorff on a thousands-point
                    # curve can waste quadratic time even for identical lines.
                    actual_points=np.asarray(actual.coords);expected_points=np.asarray(line.coords)
                    if actual_points.shape!=expected_points.shape:
                        part_error=True;continue
                    line_error=max(line_error,float(np.linalg.norm(actual_points-expected_points,axis=1).max()),
                        abs(actual.length-line.length))
                    if exported.part_start_index!=begin or exported.gear!=gear or bool(exported.implement_on)!=motion.implement_on:
                        part_error=True
        tolerance=scene.settings.coverage_tolerance_m2
        passed=(not invalid_count and not part_error and line_error<=scene.settings.geometry_epsilon_m and
            max(target_delta,gap_delta,gap_geometry_delta,region_delta)<=tolerance)
        rows.append({'field_id':job.field_id,'passed':passed,'target_delta_m2':target_delta,
            'gap_area_delta_m2':gap_delta,'gap_geometry_delta_m2':gap_geometry_delta,'region_delta_m2':region_delta,
            'invalid_export_geometry_count':invalid_count,'local_roundoff_repairs':local_repair_count,
            'route_geometry_max_error_m':line_error,'route_part_metadata_error':part_error})
    return {'passed':not findings and all(r['passed'] for r in rows),'findings':findings,'fields':rows,
        'scope':'GPKG_GEOMETRY_ROUNDTRIP_AND_ORIGINAL_TARGET_GAP; MOTIONS_REPLAYED_FROM_NPZ'}


def geometries_for_region(job,region,frame,local):
    """按field_id和region_id筛选并转回局部几何后合并，缺记录返回空面。"""
    selected=frame[(frame.field_id==job.field_id)&(frame.region_id==region.region_id)]
    return unary_union([local(g) for g in selected.geometry]) if len(selected) else GeometryCollection()

# ==========================================================================
# 10. 大批量几何参考：有界输入、独立田块进程、逐田 GPKG 事务及断点续算
# 冻结算法仍由 route.solve 执行。父进程是唯一 SQLite 写者；COMPLETED
# 和所有路线/效率记录同事务提交，断电不能留下“完成但缺少路线”的状态。
# ==========================================================================
import os
import sqlite3
import struct
import multiprocessing as mp
import pickle
import uuid
from collections import Counter, deque
from shapely import from_wkb, to_wkb, wkt

STREAM_SCHEMA = 'V7_INCREMENTAL_REFERENCE_1'
COMPACT_SCHEMA = 'V7_COMPACT_REFERENCE_2'
_OPERATION_COLUMNS = {'kind': 'TEXT', 'region_id': 'TEXT', 'task_id': 'TEXT',
    'from_task': 'TEXT', 'to_task': 'TEXT', 'sequence': 'INTEGER',
    'component': 'INTEGER', 'method': 'TEXT', 'direction': 'TEXT', 'phase': 'TEXT'}
_STREAM_LAYERS = {
    'source_fields': ('GEOMETRY', {'reference_route_status':'TEXT',
        'full_reference_connected':'INTEGER', 'full_reference_style_status':'TEXT',
        'reference_acceptance_passed':'INTEGER', 'input_task_count':'INTEGER',
        'output_task_count':'INTEGER', 'upstream_seam_quality_status':'TEXT',
        'upstream_acceptance_passed':'INTEGER', 'error':'TEXT'}),
    'work_regions': ('GEOMETRY', {'region_id':'TEXT', 'sequence_index':'INTEGER'}),
    'frozen_tasks': ('LINESTRING', {'region_id':'TEXT', 'task_id':'TEXT'}),
    'frozen_work_sweeps': ('GEOMETRY', {'region_id':'TEXT', 'task_id':'TEXT'}),
    **{name: ('LINESTRING', _OPERATION_COLUMNS) for name in
       ('body_work','body_connections','headland_reference','reference_operations')},
    'body_itinerary': ('LINESTRING', {'component':'INTEGER'}),
    'full_reference_itinerary': ('LINESTRING', {'component':'INTEGER'}),
    'field_efficiency': ('GEOMETRY', {'efficiency_status':'TEXT','missing_reason':'TEXT',
        't_work_s':'REAL','t_break_s':'REAL','efficiency_ratio':'REAL','efficiency_pct':'REAL',
        'reference_complete':'INTEGER'}),
    'efficiency_segments': ('LINESTRING', {**_OPERATION_COLUMNS,'distance_m':'REAL',
        't_work_s':'REAL','t_nonwork_drive_s':'REAL','t_stop_allocated_s':'REAL'}),
    'efficiency_stops': ('POINT', {'stop_id':'TEXT','owner_sequence':'INTEGER',
        'duration_s':'REAL','progress_m':'REAL','chain':'INTEGER'}),
}


_SUMMARY_TEXT = ('process_state route_status style_status efficiency_status failure_code '
    'failure_message body_coverage_status headland_coverage_status upstream_seam_quality_status '
    'physical_vehicle_certification time_model_id parameter_set_id accounting_completeness '
    'time_source gear_model_status gate_data_status obstacle_data_status source_record_id '
    'input_release_sha256 source_geometry_sha256 missing_reason quality_flags').split()
_SUMMARY_INTS = ('full_reference_connected reference_acceptance_passed reference_complete '
    'upstream_acceptance_passed body_task_count headland_task_count connection_count stop_count '
    'accel_count decel_count implement_switch_count gear_change_count hole_count '
    'known_obstacle_count component_count').split()
_SUMMARY_REALS = ('t_work_s t_nonwork_drive_s t_stop_s t_break_s t_total_known_s efficiency_ratio '
    'efficiency_pct headland_work_time_s t_turn_nonwork_s t_transit_nonwork_s t_accel_process_s '
    't_decel_process_s t_speed_change_extra_s work_distance_m headland_work_distance_m '
    'connection_distance_m total_distance_m accel_distance_m decel_distance_m '
    't_v1_model_difference_s source_net_area_m2 source_gross_area_m2 effective_target_area_m2 outer_perimeter_m '
    'hole_perimeter_m compactness elongation hole_area_m2 known_obstacle_area_m2 '
    'body_required_area_m2 body_covered_area_m2 body_uncovered_area_m2 body_coverage_ratio '
    'body_repeat_excess_area_m2 coverage_tolerance_m2').split()
_COMPACT_LAYERS = {
    'field_summary': ('GEOMETRY', {**dict.fromkeys(_SUMMARY_TEXT,'TEXT'),
        **dict.fromkeys(_SUMMARY_INTS,'INTEGER'), **dict.fromkeys(_SUMMARY_REALS,'REAL')}),
    'work_regions': ('GEOMETRY', {'region_id':'TEXT','sequence_index':'INTEGER','area_m2':'REAL'}),
    'route_segments': ('LINESTRING', {**_OPERATION_COLUMNS, 'segment_id':'TEXT',
        'implement_on':'INTEGER','distance_m':'REAL','t_work_s':'REAL','t_nonwork_drive_s':'REAL',
        't_stop_allocated_s':'REAL','entry_speed_mps':'REAL','exit_speed_mps':'REAL',
        'peak_speed_mps':'REAL','parameter_set_id':'TEXT','metric_geometry_blob':'BLOB',
        'metric_geometry_sha256':'TEXT'}),
    'route_events': ('POINT', {'event_id':'TEXT','event_type':'TEXT','reason':'TEXT',
        'owner_sequence':'INTEGER','chain':'INTEGER','start_progress_m':'REAL',
        'end_progress_m':'REAL','duration_s':'REAL','distance_m':'REAL',
        'v_before_mps':'REAL','v_after_mps':'REAL','parameter_set_id':'TEXT'}),
}


def _stream_json(value):
    """几何只出现在空间列或明确的 WKT 中，不允许 NaN 冒充有效数值。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def _peak_rss_mib():
    """读取当前进程峰值RSS并统一为MiB；不是所有子进程或整机的峰值内存。"""
    import resource
    # macOS 返回字节，Linux 返回 KiB。这里只报单进程峰值，不冒充整机峰值。
    scale=1024*1024 if sys.platform=='darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/scale


def _gpkg_geometry(geometry, srs_id):
    """编码标准GeoPackage头及二维WKB；空几何返回None，srs_id随图层声明保存。"""
    if geometry is None or geometry.is_empty:
        return None
    # GeoPackage 标准头：little endian，无 envelope，后接二维 WKB。
    return b'GP' + bytes((0, 1)) + struct.pack('<i', srs_id) + to_wkb(geometry, byte_order=1, output_dimension=2)


def _read_gpkg_geometry(blob):
    """校验并解码GeoPackage几何头；返回显示坐标几何，不能直接作为米制长度依据。"""
    if blob is None:
        return None
    if bytes(blob[:2]) != b'GP':
        raise ValueError('INVALID_GPKG_GEOMETRY')
    envelope = (blob[3] >> 1) & 7
    sizes = {0:0, 1:32, 2:48, 3:48, 4:64}
    if envelope not in sizes:
        raise ValueError('INVALID_GPKG_ENVELOPE')
    return from_wkb(bytes(blob[8 + sizes[envelope]:]))


class ReferenceStore:
    """单写者、逐田原子提交。属性 JSON 保存全部字段，常用指标独立成列。

    图层预先建齐，空图层也能被 GIS 打开；field_id 索引用于逐田读取，避免
    效率统计和复验每田再扫描整张表。WAL 支持处理过程中另一连接读取已提交田块。
    """
    def __init__(self, path, *, crs='EPSG:4326', schema=None):
        """创建或打开结果库并取得单写者锁；协议及来源合同须由入口核验后才可续算。"""
        import fcntl
        from pyproj import CRS
        self.path = Path(path)
        existing_schema = None
        if self.path.exists():
            with sqlite3.connect(f'file:{self.path.resolve()}?mode=ro', uri=True) as existing:
                if existing.execute("SELECT 1 FROM sqlite_master WHERE name='batch_metadata'").fetchone():
                    row = existing.execute("SELECT value_json FROM batch_metadata WHERE key='contract'").fetchone()
                    existing_schema = json.loads(row[0]).get('schema_version') if row else None
        self.schema = schema or existing_schema or STREAM_SCHEMA
        if self.schema not in (STREAM_SCHEMA, COMPACT_SCHEMA):
            raise ValueError('UNSUPPORTED_RESULT_SCHEMA')
        if existing_schema and existing_schema != self.schema:
            raise ValueError('RESULT_SCHEMA_MISMATCH')
        self.layers = dict(_COMPACT_LAYERS if self.schema == COMPACT_SCHEMA else _STREAM_LAYERS)
        self.lock = self.path.with_suffix('.gpkg.lock').open('a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError('OUTPUT_ALREADY_RUNNING')
        self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        # 标准 RTree 触发器使用 GeoPackage 几何函数；本写者注册相同函数。
        self.conn.create_function('ST_IsEmpty',1,lambda b: b is None or _read_gpkg_geometry(b).is_empty)
        for name,position in [('ST_MinX',0),('ST_MinY',1),('ST_MaxX',2),('ST_MaxY',3)]:
            self.conn.create_function(name,1,lambda b,i=position: _read_gpkg_geometry(b).bounds[i] if b else None)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=FULL')
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.execute('PRAGMA busy_timeout=60000')
        self.conn.execute('PRAGMA cache_size=-16384')
        self.conn.execute('PRAGMA application_id=1196444487')
        self.conn.execute('PRAGMA user_version=10300')
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys(srs_name TEXT NOT NULL,
              srs_id INTEGER PRIMARY KEY,organization TEXT NOT NULL,
              organization_coordsys_id INTEGER NOT NULL,definition TEXT NOT NULL,description TEXT);
            CREATE TABLE IF NOT EXISTS gpkg_contents(table_name TEXT PRIMARY KEY,
              data_type TEXT NOT NULL,identifier TEXT UNIQUE,description TEXT DEFAULT '',
              last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
              min_x DOUBLE,min_y DOUBLE,max_x DOUBLE,max_y DOUBLE,srs_id INTEGER);
            CREATE TABLE IF NOT EXISTS gpkg_geometry_columns(table_name TEXT NOT NULL,
              column_name TEXT NOT NULL,geometry_type_name TEXT NOT NULL,srs_id INTEGER NOT NULL,
              z INTEGER NOT NULL,m INTEGER NOT NULL,PRIMARY KEY(table_name,column_name));
            CREATE TABLE IF NOT EXISTS gpkg_extensions(table_name TEXT,column_name TEXT,
              extension_name TEXT NOT NULL,definition TEXT NOT NULL,scope TEXT NOT NULL,
              UNIQUE(table_name,column_name,extension_name));
            CREATE TABLE IF NOT EXISTS batch_metadata(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS field_results(field_id TEXT PRIMARY KEY,state TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,route_status TEXT,efficiency_status TEXT,
              error TEXT,elapsed_seconds REAL,detail_json TEXT,scene_json TEXT,job_json TEXT,
              updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
            CREATE TABLE IF NOT EXISTS execution_events(id INTEGER PRIMARY KEY,
              field_id TEXT,event TEXT NOT NULL,attempt INTEGER,elapsed_seconds REAL,detail TEXT,
              created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
            CREATE INDEX IF NOT EXISTS field_results_state ON field_results(state);
        ''')
        for srs_name, srs_id in [('Undefined Cartesian', -1),('Undefined Geographic',0)]:
            self.conn.execute('INSERT OR IGNORE INTO gpkg_spatial_ref_sys VALUES(?,?,?,?,?,?)',
                (srs_name,srs_id,'NONE',srs_id,'undefined',''))
        parsed = CRS.from_user_input(crs)
        self.srs_id = parsed.to_epsg() or 99999
        self.conn.execute('INSERT OR IGNORE INTO gpkg_spatial_ref_sys VALUES(?,?,?,?,?,?)',
            (parsed.name,self.srs_id,'EPSG' if parsed.to_epsg() else 'NONE',
             self.srs_id,parsed.to_wkt(version='WKT1_GDAL'),''))
        for table in ('batch_metadata','field_results','execution_events'):
            self.conn.execute('INSERT OR IGNORE INTO gpkg_contents(table_name,data_type,identifier) VALUES(?,?,?)',
                (table,'attributes',table))
        for name,(kind,columns) in list(self.layers.items()):
            self.ensure_layer(name, kind, columns)
        if self.schema == COMPACT_SCHEMA and self.conn.execute("SELECT 1 FROM gpkg_contents WHERE table_name='coverage_issues'").fetchone():
            self.ensure_layer('coverage_issues','GEOMETRY',{'issue_type':'TEXT','area_m2':'REAL','reason':'TEXT','tolerance_m2':'REAL'})
        self.conn.commit()

    def ensure_layer(self, name, kind, columns):
        """按需建立异常层，与正式空间层使用同一索引/触发器协议。"""
        self.layers[name] = (kind, columns)
        required = {'field_summary':(), 'work_regions':('region_id',), 'route_segments':('sequence','kind','component','segment_id','metric_geometry_blob','metric_geometry_sha256'), 'route_events':('event_id','event_type')} if self.schema == COMPACT_SCHEMA else {}
        definitions = ','.join('"'+key+'" '+value+(' NOT NULL' if key in required.get(name,()) else '') for key,value in columns.items())
        self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{name}" (fid INTEGER PRIMARY KEY,field_id TEXT NOT NULL,{definitions},payload_json TEXT NOT NULL,geom BLOB)')
        self.conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{name}_field" ON "{name}"(field_id)')
        label={'field_summary':'田块结果','work_regions':'作业分区','route_segments':'分段参考路线','route_events':'路线事件','coverage_issues':'覆盖异常'}.get(name,name) if self.schema==COMPACT_SCHEMA else name
        self.conn.execute('INSERT OR IGNORE INTO gpkg_contents(table_name,data_type,identifier,srs_id) VALUES(?,?,?,?)',
            (name,'features',label,self.srs_id))
        self.conn.execute('INSERT OR IGNORE INTO gpkg_geometry_columns VALUES(?,?,?,?,0,0)',(name,'geom',kind,self.srs_id))
        existing_index=self.conn.execute('SELECT 1 FROM sqlite_master WHERE name=?',('rtree_'+name+'_geom',)).fetchone()
        self.conn.execute(f'CREATE VIRTUAL TABLE IF NOT EXISTS "rtree_{name}_geom" USING rtree(id,minx,maxx,miny,maxy)')
        if not existing_index:
            self.conn.execute(f'INSERT INTO "rtree_{name}_geom" SELECT fid,ST_MinX(geom),ST_MaxX(geom),ST_MinY(geom),ST_MaxY(geom) FROM "{name}" WHERE geom IS NOT NULL AND NOT ST_IsEmpty(geom)')
        self.conn.execute('INSERT OR IGNORE INTO gpkg_extensions VALUES(?,?,?,?,?)',
            (name,'geom','gpkg_rtree_index','http://www.geopackage.org/spec/#extension_rtree','write-only'))
        trigger_sql = f'''
            CREATE TRIGGER IF NOT EXISTS "rtree_{name}_geom_insert" AFTER INSERT ON "{name}"
            WHEN NEW.geom IS NOT NULL AND NOT ST_IsEmpty(NEW.geom) BEGIN
              INSERT OR REPLACE INTO "rtree_{name}_geom" VALUES(NEW.fid,ST_MinX(NEW.geom),ST_MaxX(NEW.geom),ST_MinY(NEW.geom),ST_MaxY(NEW.geom)); END;
            CREATE TRIGGER IF NOT EXISTS "rtree_{name}_geom_delete" AFTER DELETE ON "{name}" BEGIN
              DELETE FROM "rtree_{name}_geom" WHERE id=OLD.fid; END;
            CREATE TRIGGER IF NOT EXISTS "rtree_{name}_geom_update" AFTER UPDATE OF geom,fid ON "{name}" BEGIN
              DELETE FROM "rtree_{name}_geom" WHERE id=OLD.fid;
              INSERT OR REPLACE INTO "rtree_{name}_geom" SELECT NEW.fid,ST_MinX(NEW.geom),ST_MaxX(NEW.geom),ST_MinY(NEW.geom),ST_MaxY(NEW.geom)
              WHERE NEW.geom IS NOT NULL AND NOT ST_IsEmpty(NEW.geom); END;
        '''
        import re
        for trigger in re.findall(r'CREATE TRIGGER.*?END;', trigger_sql, re.S):
            self.conn.execute(trigger)
        if self.schema == COMPACT_SCHEMA:
            unique = {'field_summary':('field_id',), 'work_regions':('field_id','region_id'), 'route_segments':('field_id','sequence'), 'route_events':('field_id','event_id')}
            if name in unique:
                self.conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "unique_{name}" ON "{name}" (' + ','.join(unique[name]) + ')')

    def metadata(self, key, default=None):
        """读取批次JSON元数据，不存在时返回调用者给定默认值。"""
        row = self.conn.execute('SELECT value_json FROM batch_metadata WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put_metadata(self, key, value):
        """保存批次元数据，不替代单田空间结果与完成状态的共同事务。"""
        self.conn.execute('INSERT OR REPLACE INTO batch_metadata VALUES(?,?)',(key,_stream_json(value)))

    def event(self, fid, event, attempt=0, seconds=0., detail=''):
        """记录处理事件、尝试次数和程序耗时；elapsed_seconds不属于作业时间。"""
        self.conn.execute('INSERT INTO execution_events(field_id,event,attempt,elapsed_seconds,detail) VALUES(?,?,?,?,?)',
            (fid,event,attempt,seconds,detail))

    def add_features(self, layer, records):
        """在当前事务中插入空间要素、独立字段及剩余payload，校验数值并维护显示范围。"""
        columns = list(self.layers[layer][1])
        keys = ['field_id',*columns,'payload_json','geom']
        sql = f'INSERT INTO "{layer}" (' + ','.join('"'+k+'"' for k in keys) + ') VALUES (' + ','.join('?' for _ in keys) + ')'
        extent = [math.inf,math.inf,-math.inf,-math.inf]
        def values():
            for record in records:
                if self.schema == COMPACT_SCHEMA:
                    if not self.conn.execute('SELECT 1 FROM field_results WHERE field_id=?',(record['field_id'],)).fetchone():
                        raise ValueError('UNKNOWN_FIELD_REFERENCE')
                    for key, sql_type in self.layers[layer][1].items():
                        value = record.get(key)
                        if value is not None and sql_type in ('REAL','INTEGER'):
                            if not isinstance(value,(int,float,bool,np.number)) or not math.isfinite(value):
                                raise ValueError('NONFINITE_OR_NONNUMERIC_ATTRIBUTE: '+key)
                            if value<0 and key!='t_v1_model_difference_s':
                                raise ValueError('NEGATIVE_ATTRIBUTE: '+key)
                            if sql_type=='INTEGER' and float(value)!=int(value):
                                raise ValueError('NONINTEGER_ATTRIBUTE: '+key)
                payload = {k:v for k,v in record.items() if k not in ('geometry','_local_geometry') and not isinstance(v, (bytes, bytearray))
                    and (self.schema != COMPACT_SCHEMA or k not in columns)}
                g=record.get('geometry')
                if g is not None and not g.is_empty:
                    x0,y0,x1,y1=g.bounds
                    extent[:]=[min(extent[0],x0),min(extent[1],y0),max(extent[2],x1),max(extent[3],y1)]
                yield (record['field_id'],*[record.get(k) for k in columns],
                       _stream_json(payload),_gpkg_geometry(record.get('geometry'),self.srs_id))
        self.conn.executemany(sql, values())
        if math.isfinite(extent[0]):
            self.conn.execute("UPDATE gpkg_contents SET last_change=strftime('%Y-%m-%dT%H:%M:%fZ','now'),min_x=MIN(COALESCE(min_x,?),?),min_y=MIN(COALESCE(min_y,?),?),max_x=MAX(COALESCE(max_x,?),?),max_y=MAX(COALESCE(max_y,?),?) WHERE table_name=?",(extent[0],extent[0],extent[1],extent[1],extent[2],extent[2],extent[3],extent[3],layer))

    def records(self, layer, fid):
        """按田读取图层记录；精简独立字段优先于payload，防止重复字段的旧值覆盖权威值。"""
        for row in self.conn.execute(f'SELECT * FROM "{layer}" WHERE field_id=? ORDER BY fid',(fid,)):
            scalars = {k:row[k] for k in self.layers[layer][1]} if self.schema == COMPACT_SCHEMA else {}
            yield {**json.loads(row['payload_json']), **scalars, 'geometry':_read_gpkg_geometry(row['geom'])}

    def close(self):
        """更新图层范围、提交并checkpoint WAL后关闭写者；文件分享前必须完成此步骤。"""
        try:
            for layer in self.layers:
                self.conn.execute(f'UPDATE gpkg_contents SET min_x=(SELECT MIN(minx) FROM "rtree_{layer}_geom"), max_x=(SELECT MAX(maxx) FROM "rtree_{layer}_geom"), min_y=(SELECT MIN(miny) FROM "rtree_{layer}_geom"), max_y=(SELECT MAX(maxy) FROM "rtree_{layer}_geom") WHERE table_name=?',(layer,))
            self.conn.commit()
            self.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        finally:
            self.conn.close()
            self.lock.close()


def _compact_field_quality(job):
    """局部米制测量与整田主体义务差集，不改变场景、分区或条带。

    主体义务必须来自封存 required_main_areas；缺资料标未知，不能用
    当前扫掠反推义务而自证覆盖。重复折算面积和重复唯一地表分开。
    """
    from shapely.geometry import Polygon,shape
    from shapely.strtree import STRtree
    from scene import polygons
    scene=load_scene(job.scene_path)
    source=_to_local(job.source_geometry,job.source_crs,job.metric_crs,job.origin,{})
    parts=polygons(source);holes=[Polygon(r) for p in parts for r in p.interiors]
    outer=sum(p.exterior.length for p in parts);inner=sum(h.length for h in holes)
    gross=unary_union([Polygon(p.exterior) for p in parts])
    rectangle=source.minimum_rotated_rectangle
    edges=[math.dist(a,b) for a,b in zip(rectangle.exterior.coords,list(rectangle.exterior.coords)[1:])] if rectangle.geom_type=='Polygon' else []
    metrics=dict(source_net_area_m2=source.area,source_gross_area_m2=gross.area,
        effective_target_area_m2=scene.target.area,outer_perimeter_m=outer,hole_perimeter_m=inner,
        hole_count=len(holes),hole_area_m2=sum(h.area for h in holes),component_count=len(parts),
        compactness=4*math.pi*source.area/(outer+inner)**2 if outer+inner>0 else None,
        elongation=max(edges)/min(edges) if edges and min(edges)>0 else None,
        known_obstacle_count=None,known_obstacle_area_m2=None,obstacle_data_status='HOLES_ONLY_NO_OBJECT_CLASSIFICATION',
        gate_data_status='CONFIGURED_POSES_NOT_FIELD_VERIFIED' if scene.start is not None and scene.end is not None else 'PARTIAL_CONFIGURED_POSES' if scene.start is not None or scene.end is not None else 'MISSING')
    raw=json.loads(Path(job.scene_path).read_text())
    if raw.get('obstacles') is not None:
        obstacle=wkt.loads(raw['obstacles']) if isinstance(raw['obstacles'],str) else shape(raw['obstacles'])
        obstacle=obstacle.intersection(gross)
        metrics.update(known_obstacle_count=len(polygons(obstacle)),known_obstacle_area_m2=obstacle.area,
            obstacle_data_status='EXPLICIT_SCENE_GEOMETRY_NOT_FIELD_MEASURED')
    required=getattr(job,'required_body',None)
    if required is None:
        metrics.update(body_coverage_status='NOT_EVALUATED',quality_flags=['REQUIRED_BODY_GEOMETRY_MISSING'])
        return dict(metrics=metrics,issues=[])
    tol=scene.settings.coverage_tolerance_m2
    swept=unary_union([t.frozen_sweep for t in job.tasks])
    clipped=[t.frozen_sweep.intersection(required) for t in job.tasks]
    covered=required.intersection(swept);missing=required.difference(swept)
    outside=swept.difference(scene.target)
    metrics.update(body_required_area_m2=required.area,body_covered_area_m2=covered.area,
        body_uncovered_area_m2=missing.area,body_coverage_ratio=covered.area/required.area if required.area>0 else None,
        body_repeat_excess_area_m2=max(0.,sum(g.area for g in clipped)-covered.area),coverage_tolerance_m2=tol)
    flags=[];evidence=[]
    def issue(kind,g,reason):
        evidence.append(dict(issue_type=kind,geometry=g,area_m2=g.area,reason=reason,tolerance_m2=tol))
    if missing.area>tol:
        flags.append('BODY_MISSING');issue('BODY_MISSING',missing,'REQUIRED_MAIN_MINUS_FROZEN_SWEEPS')
    if outside.area>tol:
        flags.append('OUTSIDE_TARGET');issue('OUTSIDE_TARGET',outside,'FROZEN_SWEEP_OUTSIDE_EFFECTIVE_TARGET')
    if job.upstream_seam_quality_status=='EXCESS_OVERLAP':
        flags.append('UPSTREAM_SEAM_EXCESS_OVERLAP')
        tree=STRtree(clipped);overlaps=[]
        for i,g in enumerate(clipped):
            for j in tree.query(g,predicate='intersects'):
                if int(j)>i:
                    overlap=g.intersection(clipped[int(j)])
                    if overlap.area>1e-8:overlaps.append(overlap)
        footprint=unary_union(overlaps)
        if not footprint.is_empty:issue('OVERLAP_FOOTPRINT',footprint,'UPSTREAM_SEAM_BUDGET_EXCEEDED')
    if job.upstream_area_ledger_delta_m2 is not None and abs(job.upstream_area_ledger_delta_m2)>tol:
        flags.append('UPSTREAM_AREA_LEDGER_MISMATCH')
    metrics.update(body_coverage_status='REVIEW' if flags else 'PASS',quality_flags=flags)
    return dict(metrics=metrics,issues=evidence)


def _coverage_display_geometry(geometry, world, local, tolerance_m2):
    """仅修复异常面显示坐标的舍入自交，不改路线或覆盖义务。

    加原点/投影可能把近重合环舍入成自交。make_valid 保留面域；
    若面域变化超过原覆盖容差，拒绝提交。零面积线残片不作为面域。
    不使用 buffer/simplify 悄悄吞掉异常面，也不修改原始面积账本。
    """
    from shapely import make_valid
    from scene import polygons
    displayed=world(geometry)
    if displayed.is_valid:return displayed,'UNCHANGED'
    repaired=unary_union(polygons(make_valid(displayed)))
    restored=local(repaired)
    # make_valid 的零面积线残片不是异常面域；边界比较只针对真实面。
    baseline=unary_union(polygons(make_valid(geometry)))
    restored_faces=unary_union(polygons(make_valid(restored)))
    if repaired.is_empty or not repaired.is_valid or abs(restored.area-geometry.area)>max(1e-6,tolerance_m2) or restored_faces.symmetric_difference(baseline).area>max(1e-8,tolerance_m2):
        raise ValueError('COVERAGE_DISPLAY_REPAIR_EXCEEDS_TOLERANCE')
    return repaired,'DISPLAY_PRECISION_MAKE_VALID'


def _parameter_id(config):
    """以实际序列化参数JSON生成摘要ID；JSON键顺序可影响ID，语义比较须解引用参数内容。"""
    return hashlib.sha256(_stream_json(config).encode()).hexdigest()


def _metric_blob(geometry):
    """将原始局部米制二维WKB压缩，并对未压缩WKB求摘要；这是复算权威坐标，不是显示geom。"""
    import zlib
    raw = to_wkb(geometry, byte_order=1, output_dimension=2)
    return zlib.compress(raw, 6), hashlib.sha256(raw).hexdigest()


def _unpack_metric(blob, digest):
    """解压米制WKB并检查摘要及几何合法性，损坏或不匹配不得降级使用显示坐标。"""
    import zlib
    from shapely.errors import GEOSException
    try:
        decoder=zlib.decompressobj()
        raw=decoder.decompress(blob,128*1024*1024)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('INVALID_OR_OVERSIZED_METRIC_BLOB')
    except zlib.error as exc:
        raise ValueError('INVALID_COMPRESSED_METRIC_BLOB') from exc
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('METRIC_GEOMETRY_HASH_MISMATCH')
    try:g = from_wkb(raw)
    except GEOSException as exc:raise ValueError('INVALID_METRIC_WKB') from exc
    if g.geom_type != 'LineString' or g.is_empty or not g.is_valid:
        raise ValueError('INVALID_METRIC_ROUTE_GEOMETRY')
    return g


def _feature_rows(db, layer, fid):
    """旧/新结果统一只读；有索引，绝不为每田扫描整个图层。"""
    for row in db.execute(f'SELECT * FROM "{layer}" WHERE field_id=? ORDER BY fid', (fid,)):
        record = json.loads(row['payload_json']) if 'payload_json' in row.keys() else {}
        record.update({k:row[k] for k in row.keys() if k not in ('fid','payload_json','geom')})
        record['geometry'] = _read_gpkg_geometry(row['geom'])
        yield record


def resolve_reference_gpkg(source):
    """把结果目录解析为对应参考/效率GPKG，或使用显式文件；输入不存在时明确失败。"""
    source = Path(source).resolve()
    if source.is_file():
        return source
    files = [source/name for name in ('reference_routes.gpkg','work_time_efficiency.gpkg')
             if (source/name).is_file()]
    if len(files) != 1:
        raise ValueError('REFERENCE_GPKG_MISSING_OR_AMBIGUOUS_USE_FILE_PATH')
    return files[0]


def read_reference_field(db, fid):
    """重建仅供计算的局部米制账本；WKT 不再重复持久化。

    预期任务从独立 field_results.job_json 取得，不能从实际导出反推。
    读取新协议同时核验原始坐标摘要和显示坐标，污染必须拒绝复算。
    """
    row = db.execute('SELECT * FROM field_results WHERE field_id=?',(fid,)).fetchone()
    if row is None or row['detail_json'] is None or row['job_json'] is None:
        raise ValueError('SOURCE_FIELD_RESULT_MISSING')
    detail, context = json.loads(row['detail_json']), json.loads(row['job_json'])
    version = json.loads(db.execute("SELECT value_json FROM batch_metadata WHERE key='contract'").fetchone()[0])['schema_version']
    compact = version == COMPACT_SCHEMA
    if version not in (COMPACT_SCHEMA,STREAM_SCHEMA):
        raise ValueError('UNSUPPORTED_RESULT_SCHEMA')
    name = 'route_segments' if compact else 'reference_operations'
    rows = sorted(_feature_rows(db,name,fid), key=lambda r:r['sequence'])
    back = None if context['metric_crs']=='LOCAL_METRIC' else Transformer.from_crs('EPSG:4326',context['metric_crs'],always_xy=True)
    def local(g):
        return affinity.translate(g if back is None else transform(back.transform,g),-context['origin'][0],-context['origin'][1])
    declared = detail.get('full_reference_operations',[])
    def key(r):return tuple(r.get(k,'') for k in ('sequence','component','kind','task_id','from_task','to_task'))
    if not compact and Counter(key(r) for r in rows)!=Counter(key(r) for r in declared):
        raise ValueError('OPERATION_EXPORT_SET_MISMATCH')
    lookup = {key(r):r for r in declared}
    operations = []
    for r in rows:
        g = _unpack_metric(r['metric_geometry_blob'],r['metric_geometry_sha256']) if compact else wkt.loads(lookup[key(r)]['geometry_wkt'])
        if r['geometry'] is None or not local(r['geometry']).equals_exact(g,1e-5):
            raise ValueError('METRIC_DISPLAY_GEOMETRY_MISMATCH')
        attributes = {k:v for k,v in r.items() if k not in ('geometry','metric_geometry_blob','metric_geometry_sha256')}
        operations.append({**attributes,'geometry_wkt':g.wkt})
    if compact:
        detail['operations'] = [{'kind':'HEADLAND','task_id':t} for t in context['expected_heads']]
        detail['full_reference_operations'] = operations
    elif 'expected_heads' not in context:
        context['expected_heads'] = [r['task_id'] for r in detail['operations'] if r['kind']=='HEADLAND']
    sources = list(_feature_rows(db,'field_summary' if compact else 'source_fields',fid))
    if len(sources)!=1:
        raise ValueError('SOURCE_GEOMETRY_COUNT_MISMATCH')
    return detail, context, sources[0]['geometry'], rows


def _compact_event(event, fid, parameter_id, world):
    """转换连续行程进度与动作事件到精简记录，变速事件保留区间和持续时间，不重复加总移动时间。"""
    start = event.get('start_progress_m',event.get('progress_m',0.))
    kind = event.get('event_type','STOP')
    return {**event, 'field_id':fid, 'event_id':event.get('event_id',event.get('stop_id')),
        'event_type':kind, 'reason':event.get('reason',event.get('actions_json','STOP')),
        'start_progress_m':start,'end_progress_m':event.get('end_progress_m',start),
        'distance_m':event.get('distance_m',0.), 'parameter_set_id':parameter_id,
        'geometry':world(Point(event['local_x_m'],event['local_y_m']))}


def _store_compact_field(store, job, result, settings, *, attempt):
    """每田唯一有序路线、事件、质量和完成状态原子提交。

    原始米制 WKB 是计算值，世界几何供 GIS 查看。只保留小型元数据和
    独立预期编号，不保存第二套完整路线、场景或 WKT 字符串。
    """
    import route
    fid = job.field_id
    info = dict(result['info']) if result.get('ok') else _reference_error(fid,result.get('error','UNKNOWN_ERROR'))
    info.update(metric_crs=job.metric_crs,origin=job.origin,
        upstream_seam_quality_status=job.upstream_seam_quality_status,
        upstream_acceptance_passed=job.upstream_acceptance_passed,
        upstream_area_ledger_delta_m2=job.upstream_area_ledger_delta_m2)
    operations = info.get('full_reference_operations',[])
    rows = [{**{k:v for k,v in r.items() if k!='geometry_wkt'},'geometry':wkt.loads(r['geometry_wkt'])} for r in operations]
    expected_heads = [r['task_id'] for r in info.get('operations',[]) if r['kind']=='HEADLAND']
    cfg = store.metadata('contract',{}).get('efficiency_config',{})
    pid = _parameter_id(cfg)
    fallback = (dict(efficiency_status='ERROR' if not result.get('ok') else 'NOT_REQUESTED',
        missing_reason=info.get('error','EFFICIENCY_NOT_REQUESTED'),efficiency_ratio=None,
        efficiency_pct=None,t_work_s=None,t_break_s=None,reference_complete=False),[],[])
    metric, segments, events = result.get('efficiency') or fallback
    metric = dict(metric)
    lookup = {r['sequence']:r for r in segments}
    forward = None if job.metric_crs=='LOCAL_METRIC' else Transformer.from_crs(job.metric_crs,'EPSG:4326',always_xy=True)
    def world(g):
        g = affinity.translate(g,*job.origin)
        return g if forward is None else transform(forward.transform,g)
    back = None if forward is None else Transformer.from_crs('EPSG:4326',job.metric_crs,always_xy=True)
    def local(g):
        return affinity.translate(g if back is None else transform(back.transform,g),-job.origin[0],-job.origin[1])
    quality = result.get('quality') or {}
    quality_values = quality.get('metrics',{})
    issue_geometries = quality.get('issues',[])
    if issue_geometries and 'coverage_issues' not in store.layers:
        with store.conn:
            store.ensure_layer('coverage_issues','GEOMETRY',{'issue_type':'TEXT','area_m2':'REAL','reason':'TEXT','tolerance_m2':'REAL'})
    with store.conn:
        for name in store.layers:
            store.conn.execute(f'DELETE FROM "{name}" WHERE field_id=?',(fid,))
        saved_rows = []
        for row in rows:
            g = row['geometry']; blob, digest = _metric_blob(g)
            timing = {k:v for k,v in lookup.get(row['sequence'],{}).items() if k not in ('geometry','_local_geometry')}
            attrs = {k:v for k,v in row.items() if k!='geometry'}
            saved_rows.append({**attrs,**timing,'field_id':fid,
                'segment_id':f'{fid}:segment:{row["sequence"]}',
                'implement_on':int(row['kind'] in ('WORK','HEADLAND')),
                'distance_m':g.length,'parameter_set_id':pid,'metric_geometry_blob':blob,
                'metric_geometry_sha256':digest,'geometry':world(g)})
        store.add_features('route_segments',saved_rows)
        saved = list(store.records('route_segments',fid))
        issues = []
        if result.get('ok'):
            if Counter(r['task_id'] for r in saved if r['kind']=='WORK')!=Counter(t.task_id for t in job.tasks):
                issues.append('EXPORTED_TASK_MISMATCH')
            if Counter(r['task_id'] for r in saved if r['kind']=='HEADLAND')!=Counter(expected_heads):
                issues.append('EXPORTED_HEADLAND_MISMATCH')
            scene = None if result.get('snapshot_only') else load_scene(job.scene_path)
            expected = {t.task_id:t.reference_line for t in job.tasks}
            for r in saved:
                g = _unpack_metric(r['metric_geometry_blob'],r['metric_geometry_sha256'])
                if not local(r['geometry']).equals_exact(g,1e-5):issues.append('EXPORTED_PROJECTION_MISMATCH')
                if scene is not None and not scene.target.buffer(1e-5).covers(g):issues.append('EXPORTED_OUTSIDE_TARGET')
                if not result.get('snapshot_only') and r['kind']=='WORK' and r['task_id'] in expected and g.hausdorff_distance(expected[r['task_id']])>1e-5:
                    issues.append('EXPORTED_SWATH_CHANGED')
            for a,b in zip(rows,rows[1:]):
                if a['component']==b['component'] and math.dist(a['geometry'].coords[-1],b['geometry'].coords[0])>1e-5:
                    issues.append('EXPORTED_ENDPOINT_GAP')
        info.update(export_audit_passed=not issues,export_audit_issues=sorted(set(issues)),
            input_integrity_passed=True,source_code_unchanged=True)
        info['reference_acceptance_passed'] = bool(result.get('ok') and route.reference_field_accepted(info) and not issues)
        if issues:
            metric.update(efficiency_status='INCOMPLETE',efficiency_ratio=None,efficiency_pct=None,
                missing_reason=';'.join(issues),reference_complete=False)
        state = 'COMPLETED' if result.get('ok') else 'FAILED'
        scalar_info = {k:v for k,v in info.items() if isinstance(v,(str,int,float,bool,type(None)))}
        summary = {**scalar_info,**quality_values,**metric,'field_id':fid,'geometry':job.source_geometry,
            'process_state':state,'route_status':info['reference_route_status'],
            'style_status':info.get('full_reference_style_status','NOT_EVALUATED'),
            'failure_code':info.get('error'),'failure_message':info.get('error'),
            'headland_coverage_status':info.get('headland_work_coverage_status','NOT_EVALUATED'),
            'quality_flags':_stream_json(quality_values.get('quality_flags',[])),
            'body_task_count':len(job.tasks),'headland_task_count':len(expected_heads),
            'source_record_id':str(job.index),'parameter_set_id':pid,
            'time_model_id':metric.get('time_model_id') or 'GEOMETRY_CONSTANT_SPEED_V1',
            'input_release_sha256':store.metadata('contract',{}).get('input_release_sha256'),
            'source_geometry_sha256':hashlib.sha256(to_wkb(job.source_geometry)).hexdigest() if job.source_geometry is not None else None}
        summary['t_turn_nonwork_s'] = metric.get('t_turn_s')
        summary['t_transit_nonwork_s'] = metric.get('t_transit_s')
        store.add_features('field_summary',[summary])
        store.add_features('work_regions',[dict(field_id=fid,region_id=r.region_id,
            sequence_index=r.sequence_index,area_m2=r.geometry.area,geometry=world(r.geometry)) for r in job.regions])
        store.add_features('route_events',[_compact_event(e,fid,pid,world) for e in events])
        if issue_geometries:
            store.ensure_layer('coverage_issues','GEOMETRY',{'issue_type':'TEXT','area_m2':'REAL','reason':'TEXT','tolerance_m2':'REAL'})
            exported_issues=[]
            for record in issue_geometries:
                geometry,repair_status=_coverage_display_geometry(record['geometry'],world,local,record['tolerance_m2'])
                exported_issues.append({**record,'field_id':fid,'geometry':geometry,'geometry_repair_status':record.get('geometry_repair_status',repair_status)})
            store.add_features('coverage_issues',exported_issues)
        light_info = {k:v for k,v in info.items() if k not in ('operations','full_reference_operations','worker_runtime')}
        context = dict(metric_crs=job.metric_crs,source_crs=job.source_crs,origin=job.origin,
            expected_tasks=[t.task_id for t in job.tasks],expected_heads=expected_heads,
            region_count=len(job.regions),scene_path=job.scene_path,source_record_id=str(job.index))
        store.conn.execute("UPDATE field_results SET state=?,attempts=?,route_status=?,efficiency_status=?,error=?,elapsed_seconds=?,detail_json=?,scene_json=NULL,job_json=?,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE field_id=?",
            (state,attempt,info['reference_route_status'],metric['efficiency_status'],info.get('error'),
             result.get('elapsed_seconds',0.),_stream_json(light_info),_stream_json(context),fid))
        store.put_metadata('metric_geometry_encoding','zlib+wkb2d-le-v1')
        store.conn.execute('CREATE TABLE IF NOT EXISTS parameter_sets(parameter_set_id TEXT PRIMARY KEY,config_json TEXT NOT NULL)')
        store.conn.execute('INSERT OR IGNORE INTO parameter_sets VALUES(?,?)',(pid,_stream_json(cfg)))
        store.conn.execute("INSERT OR IGNORE INTO gpkg_contents(table_name,data_type,identifier) VALUES('parameter_sets','attributes','计算参数')")
        runtime=info.get('worker_runtime',{})
        pools=runtime.get('native_thread_pools',[])
        store.event(fid,'COMMITTED',attempt,result.get('elapsed_seconds',0.),_stream_json(dict(state=state,
            parent_peak_rss_mib=_peak_rss_mib(),worker_peak_rss_mib=result.get('worker_peak_rss_mib'),
            native_threads_max=max((p['threads'] for p in pools),default=None),
            native_thread_environment_all_one=all(v=='1' for v in runtime.get('thread_environment',{}).values()) if runtime else None)))
    return info,metric


def _compact_summary(store):
    """从已提交GPKG统计整批状态和时间；效率按汇总时间比值，不取逐田效率的简单平均。"""
    def counts(table,column):
        return dict(store.conn.execute(f'SELECT "{column}",COUNT(*) FROM "{table}" GROUP BY "{column}"').fetchall())
    states=counts('field_results','state'); efficiencies=counts('field_summary','efficiency_status')
    n=store.conn.execute('SELECT COUNT(*) FROM field_results').fetchone()[0]
    stored,connected=store.conn.execute('SELECT COUNT(*),COALESCE(SUM(full_reference_connected),0) FROM field_summary').fetchone()
    tw,tb=store.conn.execute('SELECT COALESCE(SUM(t_work_s),0),COALESCE(SUM(t_break_s),0) FROM field_summary').fetchone()
    return dict(schema_version=COMPACT_SCHEMA,status='COMPLETED' if states.get('COMPLETED',0)+states.get('FAILED',0)==n else 'INTERRUPTED',
        input_field_count=n,stored_field_count=stored,state_counts=states,
        status_counts=counts('field_summary','route_status'),full_reference_connected_count=connected,
        full_reference_style_counts=counts('field_summary','style_status'),efficiency_status_counts=efficiencies,
        layer_counts={name:store.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in store.layers},
        t_work_seconds_sum=tw,t_break_seconds_sum=tb,
        aggregate_efficiency_ratio=tw/(tw+tb) if efficiencies.get('ESTIMATED',0)==n and tw+tb>0 else None,
        aggregate_policy='RATIO_OF_SUMMED_TIMES_NULL_IF_ANY_FIELD_INCOMPLETE',
        physical_vehicle_certification='NOT_EVALUATED',acceptance_passed=False,
        time_source='CONFIG_ESTIMATE_NOT_FIELD_MEASURED')


def _snapshot_job(db, fid, context, geometry, detail, compact):
    """只依赖保存结果的计算上下文；不需要外部 scene 文件。"""
    from types import SimpleNamespace
    back=None if context['metric_crs']=='LOCAL_METRIC' else Transformer.from_crs('EPSG:4326',context['metric_crs'],always_xy=True)
    def local(g):return affinity.translate(g if back is None else transform(back.transform,g),-context['origin'][0],-context['origin'][1])
    original={r['task_id']:wkt.loads(r['geometry_wkt']) for r in detail['full_reference_operations'] if r['kind']=='WORK'}
    return SimpleNamespace(field_id=fid,index=context.get('source_record_id',fid),source_geometry=geometry,
        metric_crs=context['metric_crs'],source_crs=context['source_crs'],origin=context['origin'],
        scene_path=context.get('scene_path',''),
        tasks=[SimpleNamespace(task_id=t,reference_line=original.get(t,LineString())) for t in context['expected_tasks']],
        regions=[SimpleNamespace(region_id=r['region_id'],sequence_index=r['sequence_index'],geometry=local(r['geometry']))
                 for r in _feature_rows(db,'work_regions',fid)],
        upstream_seam_quality_status=detail.get('upstream_seam_quality_status'),
        upstream_acceptance_passed=detail.get('upstream_acceptance_passed'),
        upstream_area_ledger_delta_m2=detail.get('upstream_area_ledger_delta_m2'))


def convert_reference_batch(source, out, *, config_path=None, resume=False, stop_after=None):
    """显式只读迁移或效率重算，逐田转写精简结果；不运行规划。

    config_path=None 严格保留来源时间，用于证明存储等价；指定参数后
    重算。旧协议逐田转写，不整文件复制冗余几何。未知/失败田隔离。
    """
    import fcntl
    from calculate_work_time_efficiency import load_config,compute_reference_efficiency
    source=resolve_reference_gpkg(source);out=Path(out).resolve()
    root=Path(__file__).resolve().parents[1]
    if not out.is_relative_to(root/'outputs'):raise ValueError('OUTPUT_MUST_BE_IN_V7_OUTPUTS')
    target=out/('work_time_efficiency.gpkg' if config_path else 'reference_routes.gpkg')
    if source==target:raise ValueError('SOURCE_OUTPUT_MUST_DIFFER')
    if not resume and out.exists():raise ValueError('OUTPUT_MUST_BE_NEW_DIRECTORY_IN_V7_OUTPUTS')
    if resume and not target.is_file():raise ValueError('RESUME_DATABASE_MISSING')
    lockpath=source.with_suffix('.gpkg.lock')
    lock=lockpath.open('rb') if lockpath.exists() else None
    try:
        if lock is not None:
            try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('SOURCE_ROUTE_BATCH_STILL_RUNNING')
        with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            original=json.loads(db.execute("SELECT value_json FROM batch_metadata WHERE key='contract'").fetchone()[0])
            compact=original['schema_version']==COMPACT_SCHEMA
            if original['schema_version'] not in (STREAM_SCHEMA,COMPACT_SCHEMA):raise ValueError('UNSUPPORTED_RESULT_SCHEMA')
            cfg=load_config(config_path) if config_path else original['efficiency_config']
            if cfg['scope']!='FULL_REFERENCE':raise ValueError('COMPACT_REQUIRES_FULL_REFERENCE')
            hashes={p.name:_sha256(p) for p in (root/'src').glob('*.py')}
            contract={**original,'schema_version':COMPACT_SCHEMA,'efficiency_config':cfg,
                'source_code_sha256':hashes,'source_gpkg_sha256':_sha256(source),
                'source_gpkg':str(source),'operation':'RECOMPUTE' if config_path else 'STORAGE_MIGRATION'}
            source_layer='field_summary' if compact else 'source_fields'
            crs=db.execute('SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=(SELECT srs_id FROM gpkg_contents WHERE table_name=?)',(source_layer,)).fetchone()[0]
            out.mkdir(parents=True,exist_ok=True)
            store=ReferenceStore(target,crs=crs,schema=COMPACT_SCHEMA)
            try:
                if resume:
                    if store.metadata('contract')!=contract:raise ValueError('RESUME_CONTRACT_MISMATCH')
                else:
                    with store.conn:
                        store.put_metadata('contract',contract)
                        store.put_metadata('source_generation_contract',original)
                        store.put_metadata('source_archive',{'source_gpkg':str(source),'sha256':contract['source_gpkg_sha256'],
                            'input_bundle':original.get('input_bundle'),'input_release_sha256':original.get('input_release_sha256')})
                        for key in ('input_manifest','input_swath_summary'):
                            row=db.execute('SELECT value_json FROM batch_metadata WHERE key=?',(key,)).fetchone()
                            if row:store.put_metadata(key,json.loads(row[0]))
                        store.conn.executemany("INSERT INTO field_results(field_id,state) VALUES(?,'PENDING')",((r[0],) for r in db.execute('SELECT field_id FROM field_results ORDER BY field_id')))
                import route
                settings=route.ApproxRouteSettings(**original.get('route_settings',{}))
                manifest=store.metadata('input_manifest',{'fields':[]})
                source_indices={e['field_id']:str(e['feature_index']) for e in manifest['fields']}
                pending=[r[0] for r in store.conn.execute("SELECT field_id FROM field_results WHERE state='PENDING' ORDER BY field_id")]
                processed=0;started=time.perf_counter()
                for fid in pending:
                    try:
                        detail,context,geometry,rows=read_reference_field(db,fid)
                        context.setdefault('source_record_id',source_indices.get(fid,fid))
                        job=_snapshot_job(db,fid,context,geometry,detail,compact)
                        state=db.execute('SELECT state,error FROM field_results WHERE field_id=?',(fid,)).fetchone()
                        old_summary=list(_feature_rows(db,'field_summary' if compact else 'field_efficiency',fid))[0]
                        if config_path:
                            try:efficiency=compute_reference_efficiency(fid,detail,context['expected_tasks'],cfg)
                            except (ValueError,KeyError,TypeError) as exc:
                                efficiency=({**old_summary,'efficiency_status':'ERROR','efficiency_ratio':None,'efficiency_pct':None,
                                    't_work_s':None,'t_break_s':None,'missing_reason':f'{type(exc).__name__}: {exc}'},[],[])
                        else:
                            segments=list(_feature_rows(db,'route_segments' if compact else 'efficiency_segments',fid))
                            events=list(_feature_rows(db,'route_events' if compact else 'efficiency_stops',fid))
                            efficiency=(old_summary,segments,events)
                        quality={'metrics':{k:old_summary.get(k) for k in _SUMMARY_REALS+_SUMMARY_INTS+_SUMMARY_TEXT
                            if k.startswith(('source_','body_','hole_','known_obstacle','outer_perimeter','compactness','elongation','effective_target','obstacle_data','coverage_tolerance','component_count','gate_data_status'))},'issues':[]}
                        flags=old_summary.get('quality_flags',[])
                        quality['metrics']['quality_flags']=json.loads(flags) if isinstance(flags,str) else flags or []
                        if compact and db.execute("SELECT 1 FROM gpkg_contents WHERE table_name='coverage_issues'").fetchone():
                            inverse=None if context['metric_crs']=='LOCAL_METRIC' else Transformer.from_crs(crs,context['metric_crs'],always_xy=True)
                            for evidence in _feature_rows(db,'coverage_issues',fid):
                                g=evidence['geometry']
                                g=g if inverse is None else transform(inverse.transform,g)
                                quality['issues'].append({**evidence,'geometry':affinity.translate(g,-context['origin'][0],-context['origin'][1])})
                        if state['state']!='COMPLETED':
                            efficiency=({**efficiency[0],'efficiency_status':'ERROR','efficiency_ratio':None,'efficiency_pct':None,'missing_reason':state['error']},efficiency[1],efficiency[2])
                        result=dict(ok=state['state']=='COMPLETED',info=detail,efficiency=efficiency,
                            quality=quality,snapshot_only=True,elapsed_seconds=0.,error=state['error'])
                    except (ValueError,KeyError,TypeError,OSError) as exc:
                        detail=_reference_error(fid,f'SOURCE_FIELD_INVALID: {type(exc).__name__}: {exc}')
                        try:
                            context=json.loads(db.execute('SELECT job_json FROM field_results WHERE field_id=?',(fid,)).fetchone()[0])
                        except (ValueError,TypeError):
                            context=dict(metric_crs='LOCAL_METRIC',source_crs=crs,origin=(0.,0.),expected_tasks=[],expected_heads=[],scene_path='')
                        sources=list(_feature_rows(db,source_layer,fid));geometry=sources[0]['geometry'] if sources else None
                        job=_snapshot_job(db,fid,context,geometry,detail,compact)
                        result=dict(ok=False,error=detail['error'],snapshot_only=True,elapsed_seconds=0.)
                    _store_reference_field(store,job,result,settings,attempt=1)
                    # 异常证据随单田路线/时间/状态在同一事务转写。
                    processed+=1
                    if processed%50==0:print(f'精简逐田提交 {processed} | {fid}',flush=True)
                    if stop_after is not None and processed>=stop_after:break
                if (_sha256(source)!=contract['source_gpkg_sha256'] or
                    hashes!={p.name:_sha256(p) for p in (root/'src').glob('*.py')} or
                    (config_path and load_config(config_path)!=cfg)):
                    invalidate_compact_efficiency(store,'INPUT_CHANGED_DURING_CALCULATION')
                    invalid=_compact_summary(store);invalid.update(status='INPUT_CHANGED',input_integrity_passed=False,reference_acceptance_passed=False)
                    with store.conn:store.put_metadata('summary',invalid)
                    atomic_json(out/'summary.json',invalid)
                    raise ValueError('INPUT_CHANGED_DURING_CALCULATION')
                summary=_compact_summary(store)
                summary.update(committed_this_run=processed,total_wall_seconds=time.perf_counter()-started,
                    input_integrity_passed=True,source_code_unchanged=True,
                    efficiency_checkpoint_counts=summary['state_counts'])
                summary['reference_acceptance_passed']=summary['status']=='COMPLETED' and not summary['state_counts'].get('FAILED') and store.conn.execute('SELECT COUNT(*) FROM field_summary WHERE reference_acceptance_passed=0').fetchone()[0]==0
                with store.conn:store.put_metadata('summary',summary)
                atomic_json(out/'summary.json',summary)
                return summary
            finally:store.close()
    finally:
        if lock is not None:lock.close()


def invalidate_compact_efficiency(store, reason):
    """发生来源/参数不稳定时清空完整效率并标记不完整，已知诊断仍保留，不能发布假完整比值。"""
    with store.conn:
        store.conn.execute("UPDATE field_summary SET efficiency_status='INCOMPLETE',efficiency_ratio=NULL,efficiency_pct=NULL,reference_complete=0,missing_reason=?,reference_acceptance_passed=0",(reason,))
        store.conn.execute("UPDATE field_results SET efficiency_status='INCOMPLETE',detail_json=json_set(detail_json,'$.reference_acceptance_passed',json('false'))")


def full_route(source, field_id):
    """按需重建局部米制完整参考线；多个行程分别返回，绝不跨断点补线。"""
    with sqlite3.connect(f'file:{resolve_reference_gpkg(source)}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        db.execute('BEGIN')  # 几何和任务账本使用同一个只读快照。
        detail,_,_,_=read_reference_field(db,field_id)
        rows=[{**r,'geometry':wkt.loads(r['geometry_wkt'])} for r in detail['full_reference_operations']]
        return list(_reference_itineraries(rows))


def _indexed_reference_inputs(bundle, out, release):
    """为分片查询建立带 field_id 索引的缓存副本；不改变封存输入。

    cache 缺失时可重建。复制＋索引成功后才原子改名，杀进程不会把半份缓存
    当作已完成；SQLite backup 逐页复制，内存不随文件大小增长。
    """
    cache = out/'cache'/'inputs';cache.mkdir(parents=True,exist_ok=True)
    paths = []
    for name in ('swath_results.gpkg','swath_results_metric.gpkg'):
        target = cache/name
        stamp = cache/(name+'.sha256.json')
        if target.exists():
            if (not stamp.is_file() or json.loads(stamp.read_text()) !=
                    {'release':release,'sha256':_sha256(target)}):
                raise ValueError('INDEXED_INPUT_CACHE_CHANGED')
        if not target.exists():
            temp = cache/(name+'.building')
            temp.unlink(missing_ok=True)
            with sqlite3.connect(f'file:{bundle/name}?mode=ro',uri=True) as source, sqlite3.connect(temp) as dest:
                source.backup(dest)
                for (table,) in dest.execute('SELECT table_name FROM gpkg_contents WHERE data_type="features"').fetchall():
                    quoted = '"'+table.replace('"','""')+'"'
                    if 'field_id' in {r[1] for r in dest.execute('PRAGMA table_info('+quoted+')')}:
                        index = '"stream_field_'+table.replace('"','""')+'"'
                        dest.execute(f'CREATE INDEX IF NOT EXISTS {index} ON {quoted}(field_id)')
                dest.execute('PRAGMA quick_check').fetchone()
            os.replace(temp,target)
            atomic_json(stamp,{'release':release,'sha256':_sha256(target)})
        paths.append(target)
    return tuple(paths)


@dataclass(frozen=True)
class ReferenceInputFailure:
    """保存无法装载单田输入的任务上下文与错误，使坏田能记账而不污染其余田块。"""
    job: FieldInput
    error: str


def _reference_input_slice(bundle, manifest, entries, indexed):
    """分片坏数据先缩小到单田；合法田块不因同片的一条坏记录被丢弃。"""
    try:
        yield from _field_rows(bundle,{**manifest,'fields':entries},
            field_ids=[e['field_id'] for e in entries],indexed_inputs=indexed)
    except (ValueError,KeyError,TypeError,OSError) as original:
        for entry in entries:
            try:
                yield from _field_rows(bundle,{**manifest,'fields':[entry]},
                    field_ids=[entry['field_id']],indexed_inputs=indexed)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                fid=entry['field_id'];where="field_id='"+fid.replace("'","''")+"'"
                source=gpd.read_file(indexed[0],layer='source_fields',where=where)
                geometry=source.iloc[0].geometry if len(source)==1 else None
                row=source.iloc[0] if len(source)==1 else {}
                job=FieldInput(int(entry['feature_index']),fid,str(_bundle_path(bundle,entry['scene_path'])),
                    geometry,str(source.crs),entry['work_crs'],(0.,0.),(),(),
                    str(row.get('seam_quality_status','UNKNOWN')),bool(row.get('acceptance_passed',False)),
                    row.get('area_ledger_delta_m2'))
                yield ReferenceInputFailure(job,f'INPUT_FIELD_INVALID: {type(exc).__name__}: {exc}')


def iter_reference_jobs(bundle, manifest, out, release, *, chunk_size=64, field_ids=None):
    """只保留一片几何；field_ids 集合仅保存轻量编号，不保存路线对象。"""
    indexed = _indexed_reference_inputs(bundle,out,release)
    selected = set(field_ids) if field_ids is not None else None
    entries = []
    for entry in manifest['fields']:
        if selected is not None and entry['field_id'] not in selected:
            continue
        entries.append(entry)
        if len(entries) == chunk_size:
            yield from _reference_input_slice(bundle,manifest,entries,indexed)
            entries = []
    if entries:
        yield from _reference_input_slice(bundle,manifest,entries,indexed)


def _reference_process(job, settings, efficiency_config, path, plot_enabled=False):
    """子进程只计算并写一个临时结果；绝不并发写最终 GPKG。

    文件传递避免大路线通过 Pipe 时阻塞发送导致父进程误判超时。父进程只在
    exitcode==0 且原子结果存在时读取；原生库崩溃由父进程独立记录并重试。
    """
    import route
    # 环境设置负责新库；threadpoolctl 同时限制已经载入的 BLAS/OpenMP 库。
    # 保持控制器活到子进程结束，避免计算过程中恢复多线程造成 CPU 过订阅。
    native_limit=None;native_pools=[]
    try:
        from threadpoolctl import threadpool_limits,threadpool_info
        native_limit=threadpool_limits(limits=1)
        native_pools=[dict(api=p['internal_api'],threads=p['num_threads']) for p in threadpool_info()]
    except ImportError:pass
    started = time.perf_counter()
    try:
        rows, info = route.solve(job,settings)
        info['worker_runtime']=dict(pid=os.getpid(),native_thread_pools=native_pools,
            thread_environment={key:os.environ.get(key) for key in _NATIVE_THREAD_ENV})
        info.update(metric_crs=job.metric_crs,origin=job.origin,
            operations=[{**{k:v for k,v in r.items() if k!='geometry'},'geometry_wkt':r['geometry'].wkt} for r in rows])
        efficiency = None
        if efficiency_config is not None:
            from calculate_work_time_efficiency import compute_reference_efficiency
            try:
                efficiency = compute_reference_efficiency(job.field_id,info,[t.task_id for t in job.tasks],efficiency_config)
            except (ValueError,KeyError,TypeError) as exc:
                efficiency = (dict(field_id=job.field_id,efficiency_status='ERROR',
                    missing_reason=f'{type(exc).__name__}: {exc}',efficiency_ratio=None,
                    efficiency_pct=None,t_work_s=None,t_break_s=None,reference_complete=False),[],[])
        if plot_enabled:
            route.plot(job,rows,info,Path(path).with_suffix('.png'))
        quality = None
        if efficiency_config and efficiency_config.get('schema_version')==2:
            quality = _compact_field_quality(job)
        result = dict(ok=True,info=info,efficiency=efficiency,quality=quality,elapsed_seconds=time.perf_counter()-started,
            worker_peak_rss_mib=_peak_rss_mib())
    except Exception as exc:
        result = dict(ok=False,error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-started)
    temporary = str(path)+'.tmp'
    with open(temporary,'wb') as stream:
        pickle.dump(result,stream,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary,path)


def _reference_error(fid, reason):
    """构造参考失败记录，明确路线未通过、效率不可用，不能用空路线当作零时间成功。"""
    return dict(field_id=fid,reference_route_status='ERROR',error=reason,
        reference_geometry_passed=False,reference_acceptance_passed=False,
        full_reference_connected=False,full_reference_status='ERROR',
        full_reference_style_status='NOT_EVALUATED',acceptance_passed=False,
        physical_vehicle_certification='NOT_EVALUATED',headland_work_coverage_status='NOT_EVALUATED',
        operations=[],full_reference_operations=[])


def _reference_itineraries(records):
    """按sequence分组独立行程，断开分量分别保留，不补造外部转场。"""
    grouped = {}
    for record in sorted(records,key=lambda r:r['sequence']):
        grouped.setdefault(record['component'],[]).append(record)
    for component, ordered in grouped.items():
        xy = []
        for record in ordered:
            coords = list(record['geometry'].coords)
            if xy and math.dist(xy[-1],coords[0])>1e-5:
                raise ValueError('ITINERARY_ENDPOINT_GAP')
            xy.extend(coords if not xy else coords[1:])
        if len(xy)>=2:
            yield dict(component=component,geometry=LineString(xy))


def _store_reference_field(store, job, result, settings, *, attempt, scene_json=None):
    """路线、回读检查、效率及完成标记同事务写入；任何失败全部回滚。

    正常的自然不连通可以保留多个行程，但全田效率为空。计算错误田块仍保存
    source_fields 与 field_efficiency 一行，便于输入输出对账，不伪造成功路线。
    """
    if store.schema == COMPACT_SCHEMA:
        return _store_compact_field(store, job, result, settings, attempt=attempt)
    import route
    fid = job.field_id
    info = result['info'] if result.get('ok') else _reference_error(fid,result.get('error','UNKNOWN_ERROR'))
    info.update(metric_crs=job.metric_crs,origin=job.origin,
        upstream_seam_quality_status=job.upstream_seam_quality_status,
        upstream_acceptance_passed=job.upstream_acceptance_passed,
        upstream_area_ledger_delta_m2=job.upstream_area_ledger_delta_m2)
    world_transform = None if job.metric_crs=='LOCAL_METRIC' else Transformer.from_crs(job.metric_crs,'EPSG:4326',always_xy=True)
    def world(geometry):
        geometry = affinity.translate(geometry,*job.origin)
        return geometry if world_transform is None else transform(world_transform.transform,geometry)
    def export(records):
        return ({**r,'field_id':fid,'geometry':world(r['geometry'])} for r in records)
    rows = [{**{k:v for k,v in r.items() if k!='geometry_wkt'},'geometry':wkt.loads(r['geometry_wkt'])} for r in info['operations']]
    full = [{**{k:v for k,v in r.items() if k!='geometry_wkt'},'geometry':wkt.loads(r['geometry_wkt'])} for r in info.get('full_reference_operations',[])]
    efficiency = result.get('efficiency')
    if efficiency is None:
        efficiency = (dict(field_id=fid,efficiency_status='ERROR' if not result.get('ok') else 'NOT_REQUESTED',
            missing_reason=info.get('error','EFFICIENCY_NOT_REQUESTED'),efficiency_ratio=None,efficiency_pct=None,
            t_work_s=None,t_break_s=None,reference_complete=False),[],[])
    metric,segments,stops = efficiency
    with store.conn:
        for layer in _STREAM_LAYERS:
            store.conn.execute(f'DELETE FROM "{layer}" WHERE field_id=?',(fid,))
        store.add_features('source_fields',[dict(field_id=fid,geometry=job.source_geometry,**{k:v for k,v in info.items() if k!='field_id' and isinstance(v,(str,int,float,bool,type(None)))})])
        store.add_features('work_regions',export([dict(region_id=r.region_id,sequence_index=r.sequence_index,geometry=r.geometry) for r in job.regions]))
        store.add_features('frozen_tasks',export([dict(region_id=t.region_id,task_id=t.task_id,geometry=t.reference_line) for t in job.tasks]))
        store.add_features('frozen_work_sweeps',export([dict(region_id=t.region_id,task_id=t.task_id,geometry=t.frozen_sweep) for t in job.tasks]))
        for kind,layer in [('WORK','body_work'),('CONNECTION','body_connections'),('HEADLAND','headland_reference')]:
            store.add_features(layer,export([r for r in rows if r['kind']==kind]))
        store.add_features('reference_operations',export(full))
        body = [r for r in rows if r['kind']!='HEADLAND' and r.get('phase','BODY')=='BODY']
        store.add_features('body_itinerary',export(list(_reference_itineraries(body))))
        store.add_features('full_reference_itinerary',export(list(_reference_itineraries(full))))
        # 独立读出持久化空间列，检查米制往返、冻结任务、边界与同分量接头。
        scene = load_scene(job.scene_path) if result.get('ok') else None
        back = None if world_transform is None else Transformer.from_crs('EPSG:4326',job.metric_crs,always_xy=True)
        def local(g):
            return affinity.translate(g if back is None else transform(back.transform,g),-job.origin[0],-job.origin[1])
        saved = list(store.records('reference_operations' if settings.assemble_reference else 'body_work',fid))
        issues = []
        expected = {t.task_id:t.reference_line for t in job.tasks}
        if result.get('ok') and Counter(r['task_id'] for r in saved if r['kind']=='WORK')!=Counter(expected.keys()):
            issues.append('EXPORTED_TASK_MISMATCH')
        for r in saved:
            g = local(r['geometry'])
            if not scene.target.buffer(1e-5).covers(g):issues.append('EXPORTED_OUTSIDE_TARGET')
            if r['kind']=='WORK' and r['task_id'] in expected and g.hausdorff_distance(expected[r['task_id']])>1e-5:
                issues.append('EXPORTED_SWATH_CHANGED')
        for a,b in zip(saved,saved[1:]) if settings.assemble_reference else []:
            if a['component']==b['component'] and math.dist(local(a['geometry']).coords[-1],local(b['geometry']).coords[0])>1e-5:
                issues.append('EXPORTED_ENDPOINT_GAP')
        info.update(export_audit_passed=not issues,export_audit_issues=sorted(set(issues)),
            input_integrity_passed=True,source_code_unchanged=True)
        info['reference_acceptance_passed']=bool(result.get('ok') and route.reference_field_accepted(info) and not issues)
        if issues:
            metric.update(efficiency_status='INCOMPLETE',efficiency_ratio=None,efficiency_pct=None,
                missing_reason=';'.join(issues),reference_complete=False)
        # 非数值属性全部保存在 payload_json，汇总列直接供 GIS 制图和统计。
        store.add_features('field_efficiency',[{**metric,'field_id':fid,'geometry':job.source_geometry}])
        store.add_features('efficiency_segments',export(segments))
        local_stops = [{**s,'geometry':Point(s['local_x_m'],s['local_y_m'])} for s in stops]
        store.add_features('efficiency_stops',export(local_stops))
        # source_fields 必须反映回读后状态，不保留检查前的乐观标记。
        store.conn.execute('DELETE FROM source_fields WHERE field_id=?',(fid,))
        store.add_features('source_fields',[dict(field_id=fid,geometry=job.source_geometry,**{k:v for k,v in info.items() if k!='field_id' and isinstance(v,(str,int,float,bool,type(None)))})])
        job_info = dict(metric_crs=job.metric_crs,source_crs=job.source_crs,origin=job.origin,
            expected_tasks=[t.task_id for t in job.tasks],region_count=len(job.regions),scene_path=job.scene_path)
        state = 'COMPLETED' if result.get('ok') else 'FAILED'
        store.conn.execute("UPDATE field_results SET state=?,attempts=?,route_status=?,efficiency_status=?,error=?,elapsed_seconds=?,detail_json=?,scene_json=?,job_json=?,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE field_id=?",
            (state,attempt,info['reference_route_status'],metric['efficiency_status'],info.get('error'),
             result.get('elapsed_seconds',0.),_stream_json(info),scene_json or (Path(job.scene_path).read_text() if Path(job.scene_path).exists() else '{}'),_stream_json(job_info),fid))
        store.event(fid,'COMMITTED',attempt,result.get('elapsed_seconds',0.),
            _stream_json(dict(state=state,parent_peak_rss_mib=_peak_rss_mib(),worker_peak_rss_mib=result.get('worker_peak_rss_mib'))))
    return info,metric


def _incremental_summary(store):
    """按实际存储协议生成汇总，V1历史层和V2精简层分别统计，避免重复累计。"""
    if store.schema == COMPACT_SCHEMA:
        return _compact_summary(store)
    states = dict(store.conn.execute('SELECT state,COUNT(*) FROM field_results GROUP BY state').fetchall())
    routes = dict(store.conn.execute('SELECT route_status,COUNT(*) FROM field_results WHERE route_status IS NOT NULL GROUP BY route_status').fetchall())
    styles = dict(store.conn.execute('SELECT full_reference_style_status,COUNT(*) FROM source_fields GROUP BY full_reference_style_status').fetchall())
    efficiencies = dict(store.conn.execute('SELECT efficiency_status,COUNT(*) FROM field_efficiency GROUP BY efficiency_status').fetchall())
    n,connected = store.conn.execute('SELECT COUNT(*),COALESCE(SUM(full_reference_connected),0) FROM source_fields').fetchone()
    tw,tb = store.conn.execute('SELECT COALESCE(SUM(t_work_s),0),COALESCE(SUM(t_break_s),0) FROM field_efficiency').fetchone()
    selected_count = store.conn.execute('SELECT COUNT(*) FROM field_results').fetchone()[0]
    complete = states.get('COMPLETED',0)+states.get('FAILED',0)==selected_count
    all_estimated = efficiencies.get('ESTIMATED',0)==selected_count
    layer_counts = {layer:store.conn.execute(f'SELECT COUNT(*) FROM "{layer}"').fetchone()[0] for layer in _STREAM_LAYERS}
    return dict(schema_version=STREAM_SCHEMA,status='COMPLETED' if complete else 'INTERRUPTED',
        input_field_count=selected_count,stored_field_count=n,state_counts=states,status_counts=routes,
        full_reference_connected_count=connected,full_reference_style_counts=styles,
        efficiency_status_counts=efficiencies,layer_counts=layer_counts,
        t_work_seconds_sum=tw,t_break_seconds_sum=tb,
        aggregate_efficiency_ratio=tw/(tw+tb) if all_estimated and tw+tb>0 else None,
        aggregate_policy='RATIO_OF_SUMMED_TIMES_NULL_IF_ANY_FIELD_INCOMPLETE',
        physical_vehicle_certification='NOT_EVALUATED',acceptance_passed=False,
        time_source='CONFIG_ESTIMATE_NOT_FIELD_MEASURED')


# 每田使用一个 Python 进程。CPU 上限取可见核、进程 affinity 和 Linux
# cgroup 配额的最小值；容器只能使用配给自己的资源。内存预算是调度估计，
# 不是操作系统的硬 RSS 限额：大田仍需监测，不能宣称彻底消除 OOM。
_NATIVE_THREAD_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')


def _cgroup_resource_directories():
    """常规 Linux cgroup v1/v2 挂载，连同当前分组的祖先限制一起读取。"""
    if sys.platform != 'linux':return []
    root=Path('/sys/fs/cgroup');locations=[]
    try:lines=Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:lines=[]
    for line in lines:
        parts=line.split(':',2)
        if len(parts)!=3:continue
        _,controllers,relative=parts
        names=controllers.split(',') if controllers else ['unified']
        bases=[root] if not controllers else [root/controllers,*[root/n for n in names]]
        for base in bases:
            # 容器路径可能含 '..'；不得读出 cgroup 挂载点以外。
            candidate=base/relative.lstrip('/')
            if '..' in candidate.parts or not candidate.is_dir():candidate=base
            while True:
                for name in names:locations.append((name,candidate))
                if candidate==base:break
                candidate=candidate.parent
    locations.extend([('unified',root),('cpu',root/'cpu'),
        ('cpu',root/'cpu,cpuacct'),('memory',root/'memory')])
    return list(dict.fromkeys(locations))


def _resource_text(path):
    """读取系统资源声明文本；不存在或不可解码返回None，由上层采用保守策略。"""
    try:return path.read_text().strip()
    except (OSError,UnicodeError):return None


def _cpu_idle_capacity(allowed_cpu_ids=None):
    """启动时采样0.25秒，估算可用的整核槽位；只统计进程允许使用的CPU。

    这是一次负载快照，不是实时扩缩容或CPU配额保证。满载时仍留一个进程
    使任务能推进；探测失败返回未知，由上层使用CPU硬上限和内存预算。
    """
    try:
        import psutil
        samples=psutil.cpu_percent(interval=.25,percpu=True)
        ids=sorted(allowed_cpu_ids) if allowed_cpu_ids is not None else list(range(len(samples)))
        if not ids or any(i<0 or i>=len(samples) for i in ids):return None
        values=[samples[i] for i in ids]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=100 for v in values):return None
        idle=sum((100-v)/100 for v in values)
        return dict(cpu_idle_slots=max(1,math.floor(idle)),
            cpu_utilization_pct=sum(values)/len(values),cpu_idle_source='psutil.percpu_startup_sample',
            cpu_sample_seconds=.25)
    except (ImportError,OSError,AttributeError,ValueError,TypeError):return None


def batch_resource_limits():
    """资源探测失败显式返回未知；不能把未知内存误报成无限内存。"""
    cpu=max(1,os.cpu_count() or 1);counts={'visible_logical_cpus':cpu};allowed_cpu_ids=None
    try:
        allowed_cpu_ids=os.sched_getaffinity(0)
        affinity=len(allowed_cpu_ids);counts['affinity_cpus']=affinity
        if affinity>0:cpu=min(cpu,affinity)
    except (AttributeError,OSError):pass
    available=None;memory_source=None
    try:
        import psutil
        available=int(psutil.virtual_memory().available);memory_source='psutil.available'
    except (ImportError,OSError,AttributeError):
        try:
            available=int(os.sysconf('SC_AVPHYS_PAGES'))*int(os.sysconf('SC_PAGE_SIZE'))
            memory_source='sysconf.available_pages'
        except (ValueError,OSError,AttributeError):pass
    quotas=[];headrooms=[]
    for controller,directory in _cgroup_resource_directories():
        try:
            if controller=='unified':
                maximum=_resource_text(directory/'cpu.max')
                if maximum:
                    quota,period=maximum.split()
                    if quota!='max' and int(quota)>0 and int(period)>0:quotas.append(int(quota)/int(period))
                limit=_resource_text(directory/'memory.max');usage=_resource_text(directory/'memory.current')
            elif controller=='cpu':
                quota=_resource_text(directory/'cpu.cfs_quota_us');period=_resource_text(directory/'cpu.cfs_period_us')
                if quota and period and int(quota)>0 and int(period)>0:quotas.append(int(quota)/int(period))
                continue
            elif controller=='memory':
                limit=_resource_text(directory/'memory.limit_in_bytes');usage=_resource_text(directory/'memory.usage_in_bytes')
            else:continue
            if limit and usage and limit!='max' and 0<int(limit)<2**60:
                headrooms.append(max(0,int(limit)-int(usage)))
        except (ValueError,TypeError):continue  # 未支持的格式保留其它可用探测结果。
    if quotas:
        counts['cgroup_cpu_quota']=min(quotas)
        cpu=min(cpu,max(1,math.floor(min(quotas))))
    if headrooms:
        available=min([*headrooms,*([available] if available is not None else [])])
        memory_source=(memory_source+' + ' if memory_source else '')+'cgroup.headroom'
    idle=_cpu_idle_capacity(allowed_cpu_ids)
    counts.update(idle or dict(cpu_idle_slots=None,cpu_utilization_pct=None,cpu_idle_source=None,cpu_sample_seconds=None))
    return dict(**counts,usable_cpus=cpu,available_memory_mib=None if available is None else available/2**20,
        memory_source=memory_source)


def reference_worker_plan(workers=0,*,field_count,worker_memory_mib=768.,memory_budget_mib=0.):
    """0 自动选核；正数是请求上限，可超过12，仍受CPU空闲快照/硬限制、内存及任务数约束。

    默认使用当前空闲内存的 75%，先留 512 MiB 给父进程，再按每子进程预算
    计算并行槽位。显式预算同样不能超过已探测的空闲内存。768 MiB 是保守
    起点，绝非所有田块的内存保证；大田/复杂孔洞应增大该预算。
    """
    if type(workers) is not int or workers<0:raise ValueError('INVALID_WORKERS')
    if type(field_count) is not int or field_count<0:raise ValueError('INVALID_FIELD_COUNT')
    for value in (worker_memory_mib,memory_budget_mib):
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('INVALID_MEMORY_BUDGET')
    if worker_memory_mib<=0:raise ValueError('INVALID_WORKER_MEMORY_BUDGET')
    limits=batch_resource_limits();available=limits['available_memory_mib']
    if memory_budget_mib>0:
        budget=min(memory_budget_mib,available) if available is not None else memory_budget_mib
    else:budget=available*.75 if available is not None else None
    memory_slots=None if budget is None else max(0,math.floor((budget-512.)/worker_memory_mib))
    requested=workers or limits['usable_cpus']
    effective=min(requested,limits['usable_cpus'],field_count)
    if limits.get('cpu_idle_slots') is not None:effective=min(effective,limits['cpu_idle_slots'])
    if memory_slots is not None:effective=min(effective,memory_slots)
    else:effective=min(effective,1)  # 无法探测RAM且无显式预算时，任何请求都保守运行。
    if field_count and effective<1:raise ValueError('INSUFFICIENT_MEMORY_BUDGET_INCREASE_BUDGET_OR_REDUCE_WORKER_MEMORY_ESTIMATE')
    return dict(**limits,requested_workers=workers,effective_workers=effective,
        worker_memory_mib=worker_memory_mib,memory_budget_mib=budget,memory_slots=memory_slots,
        memory_policy='ESTIMATED_NOT_HARD_RSS_LIMIT',native_threads_per_worker=1)


def _start_reference_process(process):
    """spawn 在导入 NumPy/GDAL 前继承单线程设置；启动后恢复父进程环境。"""
    previous={key:os.environ.get(key) for key in _NATIVE_THREAD_ENV}
    try:
        for key in _NATIVE_THREAD_ENV:os.environ[key]='1'
        process.start()
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def run_incremental_reference_batch(swath_bundle,out,*,workers=0,field_id=None,
        route_config=None,efficiency_config=None,resume=False,field_timeout=300.,
        retries=1,chunk_size=64,plot_enabled=False,stop_after=None,worker_target=None,
        worker_memory_mib=768.,memory_budget_mib=0.):
    """生产批处理入口。stop_after 仅供中断/续算验证，按已提交田数停止。

    每个 worker 只持有一田，父进程持有一块输入及在途任务（受 chunk_size
    与 workers 共同限制）；一个完成结果提交后
    立即释放。重启只重算 PENDING/RUNNING 和预算尚未耗尽的 FAILED。
    输入摘要、源码、路线及效率参数不一致必须拒绝续跑，禁止混合版本。
    """
    import route
    from calculate_work_time_efficiency import load_config
    start = time.perf_counter();bundle = Path(swath_bundle).resolve();out = Path(out).resolve()
    root = Path(__file__).resolve().parents[1]
    if not out.is_relative_to(root/'outputs'):raise ValueError('OUTPUT_MUST_BE_IN_V7_OUTPUTS')
    if type(workers) is not int or workers<0:raise ValueError('INVALID_WORKERS')
    if type(chunk_size) is not int or not 1<=chunk_size<=2000:raise ValueError('INVALID_CHUNK_SIZE')
    if type(retries) is not int or not 0<=retries<=5:raise ValueError('INVALID_RETRY_LIMIT')
    if type(field_timeout) not in (int,float) or not math.isfinite(field_timeout) or field_timeout<=0:raise ValueError('INVALID_FIELD_TIMEOUT')
    if stop_after is not None and (type(stop_after) is not int or stop_after<1):raise ValueError('INVALID_STOP_AFTER')
    if type(plot_enabled) is not bool or type(resume) is not bool:raise ValueError('INVALID_BATCH_FLAG')
    data = config_section(route_config, "routes") if route_config else {}
    settings = route.ApproxRouteSettings(**data)
    if settings.motion_refinement or settings.operational_refinement:
        raise ValueError('INCREMENTAL_REQUIRES_GEOMETRY_REFERENCE_PROFILE')
    cfg = load_config(efficiency_config or root/'config.json')
    if cfg['scope']!='FULL_REFERENCE' or not settings.assemble_reference:
        raise ValueError('INCREMENTAL_REQUIRES_FULL_REFERENCE_AND_EFFICIENCY')
    # verify_bundle 本身执行 seal 检查，不对巨型包做两次重复校验。
    _,manifest = _verify_bundle(bundle)
    release = _sha256(bundle/'bundle_checksums.json')
    selected = set(field_id.split(',')) if field_id else None
    entries = [e for e in manifest['fields'] if selected is None or e['field_id'] in selected]
    if not entries or (selected is not None and {e['field_id'] for e in entries}!=selected):raise ValueError('UNKNOWN_FIELD_ID')
    hashes = {p.name:_sha256(p) for p in Path(__file__).parent.glob('*.py')}
    output_schema = COMPACT_SCHEMA if cfg['schema_version'] == 2 else STREAM_SCHEMA
    contract = dict(schema_version=output_schema,input_bundle=str(bundle),input_release_sha256=release,
        route_settings=asdict(settings),efficiency_config=cfg,source_code_sha256=hashes,
        field_ids_sha256=hashlib.sha256(_stream_json([e['field_id'] for e in entries]).encode()).hexdigest())
    database = out/'reference_routes.gpkg'
    if resume:
        if not database.is_file():raise ValueError('RESUME_DATABASE_MISSING')
    elif out.exists() and any(p.name!='cache' for p in out.iterdir()):
        raise ValueError('OUTPUT_DIRECTORY_MUST_BE_NEW_OR_EMPTY_USE_BATCH_RESUME')
    out.mkdir(parents=True,exist_ok=True)
    temp = out/'cache'/'workers';temp.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'cache'/'matplotlib')
    # 世界坐标图层全批使用同一 CRS。LOCAL_METRIC 不可与投影田混写。
    local = all(e['work_crs']=='LOCAL_METRIC' for e in entries)
    if any(e['work_crs']=='LOCAL_METRIC' for e in entries) and not local:raise ValueError('MIXED_LOCAL_AND_PROJECTED_CRS')
    # 读取 CRS 只需一个轻量要素，避免某些 GDAL 引擎将 rows=0 解释为全层。
    source_crs = gpd.read_file(bundle/'swath_results.gpkg',layer='source_fields',rows=1).crs
    if resume:
        with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as saved:
            existing=saved.execute("SELECT value_json FROM batch_metadata WHERE key='contract'").fetchone()
            if existing is None or json.loads(existing[0])!=contract:raise ValueError('RESUME_CONTRACT_MISMATCH')
    store = ReferenceStore(database,crs=source_crs if local else 'EPSG:4326',schema=output_schema)
    active = {};pending = deque();committed_this_run = 0;halt = False;peak_active_workers=0
    # spawn 避免继承父进程 SQLite/GDAL 句柄；每田独立，原生崩溃不会破坏其他田。
    context = mp.get_context('spawn');target = worker_target or _reference_process
    import signal
    previous_sigterm=signal.getsignal(signal.SIGTERM)
    def stop_signal(signum,frame):raise InterruptedError('BATCH_TERMINATED_COMMITTED_FIELDS_RETAINED')
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        if resume:
            if store.metadata('contract')!=contract:raise ValueError('RESUME_CONTRACT_MISMATCH')
            with store.conn:
                store.conn.execute("UPDATE field_results SET state='PENDING' WHERE state='RUNNING'")
                store.event(None,'RESUMED')
        else:
            with store.conn:
                store.put_metadata('contract',contract)
                store.put_metadata('input_manifest',manifest)
                store.put_metadata('input_swath_summary',json.loads((bundle/'swath_batch_summary.json').read_text()))
                store.conn.executemany("INSERT INTO field_results(field_id,state) VALUES(?,'PENDING')",((e['field_id'],) for e in entries))
                store.event(None,'STARTED')
        remaining = {r[0] for r in store.conn.execute("SELECT field_id FROM field_results WHERE state='PENDING' OR (state='FAILED' AND attempts<?)",(retries+1,))}
        # 幂等续算没有待算任务时不要求启动资源；内存不足不能让已完成结果失效。
        worker_plan=reference_worker_plan(workers,field_count=len(remaining),worker_memory_mib=worker_memory_mib,memory_budget_mib=memory_budget_mib)
        workers=worker_plan['effective_workers']
        print(f'并行调度: 请求 {worker_plan["requested_workers"]} (0=自动), CPU 上限 {worker_plan["usable_cpus"]}, 空闲估计 {worker_plan.get("cpu_idle_slots")}, 内存槽位 {worker_plan["memory_slots"]}, 实际进程 {workers}, 每进程原生线程 1',flush=True)
        with store.conn:
            store.event(None,'WORKER_PLAN',detail=_stream_json(worker_plan))
        manifest_slice = {**manifest,'fields':entries}
        jobs = iter_reference_jobs(bundle,manifest_slice,out,release,chunk_size=chunk_size,field_ids=remaining)
        exhausted = not remaining
        while active or pending or not exhausted:
            while not halt and len(active)<workers:
                if not pending:
                    try:pending.append(next(jobs))
                    except StopIteration:exhausted=True;break
                job = pending.popleft()
                if isinstance(job,ReferenceInputFailure):
                    failure=job;job=failure.job
                    attempt=store.conn.execute('SELECT attempts FROM field_results WHERE field_id=?',(job.field_id,)).fetchone()[0]+1
                    _store_reference_field(store,job,dict(ok=False,error=failure.error,elapsed_seconds=0.),settings,attempt=attempt)
                    committed_this_run+=1
                    if stop_after is not None and committed_this_run>=stop_after:halt=True;break
                    continue
                attempts = store.conn.execute('SELECT attempts FROM field_results WHERE field_id=?',(job.field_id,)).fetchone()[0]+1
                artifact = temp/(uuid.uuid4().hex+'.pkl')
                process = context.Process(target=target,args=(job,settings,cfg,str(artifact),plot_enabled))
                with store.conn:
                    store.conn.execute("UPDATE field_results SET state='RUNNING',attempts=? WHERE field_id=?",(attempts,job.field_id))
                    store.event(job.field_id,'STARTED',attempts)
                _start_reference_process(process)
                active[job.field_id]=(job,process,time.monotonic(),artifact,attempts)
                peak_active_workers=max(peak_active_workers,len(active))
            for fid,(job,process,started,artifact,attempt) in list(active.items()):
                elapsed = time.monotonic()-started
                if process.is_alive() and elapsed<field_timeout:continue
                timed_out = process.is_alive()
                if timed_out:
                    process.terminate();process.join(timeout=1)
                    if process.is_alive():process.kill();process.join(timeout=1)
                else:process.join(timeout=1)
                if timed_out:result=dict(ok=False,error='FIELD_HARD_TIMEOUT',elapsed_seconds=elapsed)
                elif process.exitcode!=0:result=dict(ok=False,error=f'WORKER_CRASH_EXIT_{process.exitcode}',elapsed_seconds=elapsed)
                elif not artifact.is_file():result=dict(ok=False,error='WORKER_RESULT_MISSING',elapsed_seconds=elapsed)
                else:
                    try:
                        with artifact.open('rb') as stream:result=pickle.load(stream)
                    except Exception as exc:result=dict(ok=False,error=f'WORKER_RESULT_INVALID: {exc}',elapsed_seconds=elapsed)
                artifact.unlink(missing_ok=True);Path(str(artifact)+'.tmp').unlink(missing_ok=True)
                del active[fid];process.close()
                if not result.get('ok') and attempt<retries+1:
                    with store.conn:store.event(fid,'RETRY',attempt,elapsed,result['error'])
                    pending.append(job)
                    continue
                try:
                    info,metric = _store_reference_field(store,job,result,settings,attempt=attempt)
                except Exception as exc:
                    # 单田转换/导出失败也隔离。数据库 I/O/磁盘满不能伪装为田块失败。
                    if isinstance(exc,sqlite3.Error):raise
                    failure=dict(ok=False,error=f'FIELD_EXPORT_ERROR: {type(exc).__name__}: {exc}',elapsed_seconds=elapsed)
                    info,metric=_store_reference_field(store,job,failure,settings,attempt=attempt)
                committed_this_run+=1
                if committed_this_run%10==0 or stop_after==committed_this_run:
                    print(f'逐田提交 GPKG {committed_this_run} | {fid} | {info["reference_route_status"]} | {metric["efficiency_status"]}',flush=True)
                if plot_enabled:
                    # 图件是可选展示；原始结果与模型参数均已进入 GPKG。
                    png=artifact.with_suffix('.png')
                    if png.exists():
                        with store.conn:
                            store.conn.execute('CREATE TABLE IF NOT EXISTS field_artifacts(field_id TEXT,kind TEXT,content BLOB,PRIMARY KEY(field_id,kind))')
                            store.conn.execute('INSERT OR REPLACE INTO field_artifacts VALUES(?,?,?)',(fid,'overview.png',png.read_bytes()))
                            store.conn.execute("INSERT OR IGNORE INTO gpkg_contents(table_name,data_type,identifier) VALUES('field_artifacts','attributes','field_artifacts')")
                        png.unlink()
                if stop_after is not None and committed_this_run>=stop_after:
                    halt=True;break
            if halt:break
            if active:time.sleep(.05)
        # 整批完成后复核来源。发生变化则撤回通过标记，续跑也会拒绝旧合同。
        unchanged = hashes=={p.name:_sha256(p) for p in Path(__file__).parent.glob('*.py')}
        try:input_unchanged = _verify_bundle_seal(bundle)==release
        except (ValueError,OSError):input_unchanged=False
        try:
            parameters_unchanged = load_config(efficiency_config or root/'config.json') == cfg and (route_config is None or config_section(route_config, "routes") == data)
        except (ValueError,OSError):parameters_unchanged=False
        if not unchanged or not input_unchanged or not parameters_unchanged:
            if store.schema == COMPACT_SCHEMA:
                invalidate_compact_efficiency(store,'BATCH_INPUT_OR_SOURCE_CHANGED')
            else:
                with store.conn:
                    store.conn.execute("UPDATE source_fields SET reference_acceptance_passed=0,payload_json=json_set(payload_json,'$.reference_acceptance_passed',json('false'),'$.source_code_unchanged',json(?),'$.input_integrity_passed',json(?))",('true' if unchanged else 'false','true' if input_unchanged else 'false'))
                    store.conn.execute("UPDATE field_results SET detail_json=json_set(detail_json,'$.reference_acceptance_passed',json('false'),'$.source_code_unchanged',json(?),'$.input_integrity_passed',json(?)),efficiency_status='INCOMPLETE' WHERE detail_json IS NOT NULL",('true' if unchanged else 'false','true' if input_unchanged else 'false'))
                    store.conn.execute("UPDATE field_efficiency SET efficiency_status='INCOMPLETE',efficiency_ratio=NULL,efficiency_pct=NULL,reference_complete=0,missing_reason='BATCH_INPUT_OR_SOURCE_CHANGED',payload_json=json_set(payload_json,'$.efficiency_status','INCOMPLETE','$.efficiency_ratio',NULL,'$.efficiency_pct',NULL,'$.reference_complete',json('false'),'$.missing_reason','BATCH_INPUT_OR_SOURCE_CHANGED')")
        summary = _incremental_summary(store)
        summary.update(input_integrity_passed=input_unchanged,source_code_unchanged=unchanged,parameters_unchanged=parameters_unchanged,
            workers=workers,worker_plan=worker_plan,peak_active_workers=peak_active_workers,chunk_size=chunk_size,field_timeout_seconds=field_timeout,retry_limit=retries,
            plots_enabled=plot_enabled,committed_this_run=committed_this_run,
            total_wall_seconds=time.perf_counter()-start,settings=asdict(settings),input_bundle=str(bundle),
            input_release_sha256=release,source_code_sha256_start=hashes)
        summary['parent_peak_rss_mib']=_peak_rss_mib()
        summary['reference_acceptance_passed']=(summary['status']=='COMPLETED' and not summary['state_counts'].get('FAILED')
            and input_unchanged and unchanged and parameters_unchanged and store.conn.execute('SELECT COUNT(*) FROM '+('field_summary' if store.schema==COMPACT_SCHEMA else 'source_fields')+' WHERE reference_acceptance_passed=0').fetchone()[0]==0)
        with store.conn:
            store.put_metadata('summary',summary)
            store.event(None,'STOPPED_AFTER_COMMIT' if halt else 'FINISHED',seconds=summary['total_wall_seconds'])
        atomic_json(out/'summary.json',summary)  # 可读镜像；完整正式结果在 GPKG 内。
        return summary
    except BaseException as exc:
        # SIGTERM / Ctrl-C 同样保留已提交结果，并明确记录未完成。SIGKILL 或
        # 断电由 SQLite 恢复事务，RUNNING 在下次 --batch-resume 重置为待算。
        try:
            summary=_incremental_summary(store)
            summary.update(status='INTERRUPTED',interruption_reason=f'{type(exc).__name__}: {exc}',
                committed_this_run=committed_this_run,reference_acceptance_passed=False)
            with store.conn:
                store.put_metadata('summary',summary)
                store.event(None,'INTERRUPTED',detail=summary['interruption_reason'])
            atomic_json(out/'summary.json',summary)
        except Exception:pass  # 磁盘满时保留原始异常，不能宣称记录成功。
        raise
    finally:
        for job,process,started,artifact,attempt in active.values():
            if process.is_alive():process.terminate()
            process.join(timeout=1)
            if process.is_alive():process.kill();process.join(timeout=1)
            process.close();artifact.unlink(missing_ok=True);Path(str(artifact)+'.tmp').unlink(missing_ok=True)
        store.close()
        signal.signal(signal.SIGTERM,previous_sigterm)


def run_incremental_efficiency(batch, config_path, out, *, resume=False, stop_after=None):
    """读取关闭写者后的快照；共享锁阻止参考批次同时续写源数据库。"""
    import fcntl
    lock_path=Path(batch)/'reference_routes.gpkg.lock'
    source_lock=lock_path.open('rb') if lock_path.exists() else None
    try:
        if source_lock is not None:
            try:fcntl.flock(source_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('SOURCE_ROUTE_BATCH_STILL_RUNNING')
        return _run_incremental_efficiency_locked(batch,config_path,out,resume=resume,stop_after=stop_after)
    finally:
        if source_lock is not None:source_lock.close()


def _run_incremental_efficiency_locked(batch, config_path, out, *, resume=False, stop_after=None):
    """从自包含参考 GPKG 逐田复算；坏田隔离，停止后可继续同一输出。

    不加载整批几何，也不再依赖几十万个外部 result.json。原路线数据库只读；
    首次用 SQLite 页式 backup 保留路线，之后仅替换当前田的效率记录。
    """
    from calculate_work_time_efficiency import load_config, compute_reference_efficiency, sha256
    root = Path(__file__).resolve().parents[1]
    batch,out = Path(batch).resolve(),Path(out).resolve()
    source = batch/'reference_routes.gpkg';target=out/'work_time_efficiency.gpkg'
    config=load_config(config_path)
    if not out.is_relative_to(root/'outputs'):raise ValueError('OUTPUT_MUST_BE_IN_V7_OUTPUTS')
    contract=dict(source_gpkg_sha256=sha256(source),config=config,
        script_sha256=sha256(root/'src/calculate_work_time_efficiency.py'),
        io_sha256=sha256(Path(__file__)))
    if resume:
        if not target.is_file():raise ValueError('RESUME_DATABASE_MISSING')
    else:
        if out.exists():raise ValueError('OUTPUT_MUST_BE_NEW_DIRECTORY_IN_V7_OUTPUTS')
        out.mkdir(parents=True)
        temp=out/'efficiency.building.gpkg'
        with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as src,sqlite3.connect(temp) as dst:
            src.backup(dst)
        os.replace(temp,target)
    with sqlite3.connect(target) as conn:
        crs=conn.execute('SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=(SELECT srs_id FROM gpkg_contents WHERE table_name="source_fields")').fetchone()[0]
    store=ReferenceStore(target,crs=crs)
    started=time.perf_counter();processed=0
    try:
        if resume:
            if store.metadata('efficiency_contract')!=contract:raise ValueError('RESUME_CONTRACT_MISMATCH')
        else:
            with store.conn:
                store.put_metadata('efficiency_contract',contract)
                if store.conn.execute("SELECT 1 FROM sqlite_master WHERE name='independent_validation'").fetchone():
                    store.conn.execute('DELETE FROM independent_validation')
                store.conn.execute("DELETE FROM batch_metadata WHERE key='independent_validation'")
                store.conn.execute('CREATE TABLE efficiency_checkpoint(field_id TEXT PRIMARY KEY,state TEXT,error TEXT)')
                store.conn.execute("INSERT INTO efficiency_checkpoint SELECT field_id,'PENDING',NULL FROM field_results")
                store.conn.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier) VALUES('efficiency_checkpoint','attributes','efficiency_checkpoint')")
        # 游标流式取编号；每田空间查询有 field_id 索引，不扫描整层。
        cursor=store.conn.execute("SELECT field_id FROM efficiency_checkpoint WHERE state='PENDING' ORDER BY field_id")
        while True:
            record=cursor.fetchone()
            if record is None:break
            fid=record[0]
            row=store.conn.execute('SELECT detail_json,job_json FROM field_results WHERE field_id=?',(fid,)).fetchone()
            sources=list(store.records('source_fields',fid));geometry=sources[0]['geometry'] if sources else None
            try:
                if row[0] is None or row[1] is None:raise ValueError('SOURCE_FIELD_RESULT_MISSING')
                detail,job=json.loads(row[0]),json.loads(row[1])
                full=config['scope']=='FULL_REFERENCE'
                operations=list(store.records('reference_operations',fid)) if full else [*store.records('body_work',fid),*store.records('body_connections',fid)]
                operations.sort(key=lambda r:r['sequence'])
                declared=detail.get('full_reference_operations',[]) if full else [r for r in detail['operations'] if r['kind']!='HEADLAND' and r.get('phase','BODY')=='BODY']
                def key(r):return tuple(r.get(k,'') for k in ('kind','sequence','component','task_id','from_task','to_task'))
                if Counter(key(r) for r in operations)!=Counter(key(r) for r in declared):raise ValueError('OPERATION_EXPORT_SET_MISMATCH')
                lookup={key(r):r for r in declared}
                back=None if job['metric_crs']=='LOCAL_METRIC' else Transformer.from_crs('EPSG:4326',job['metric_crs'],always_xy=True)
                for op in operations:
                    g=affinity.translate(op['geometry'] if back is None else transform(back.transform,op['geometry']),-job['origin'][0],-job['origin'][1])
                    if not g.equals_exact(wkt.loads(lookup[key(op)]['geometry_wkt']),1e-5):raise ValueError('JSON_EXPORT_GEOMETRY_MISMATCH')
                result,segments,stops=compute_reference_efficiency(fid,detail,job['expected_tasks'],config)
                forward=None if back is None else Transformer.from_crs(job['metric_crs'],'EPSG:4326',always_xy=True)
                def world(g):
                    g=affinity.translate(g,*job['origin'])
                    return g if forward is None else transform(forward.transform,g)
                segments=[{**r,'geometry':world(r['geometry'])} for r in segments]
                stops=[{**r,'geometry':world(Point(r['local_x_m'],r['local_y_m']))} for r in stops]
                state='COMPLETED';error=None
            except (ValueError,KeyError,TypeError,OSError) as exc:
                error=f'{type(exc).__name__}: {exc}';state='FAILED'
                result=dict(field_id=fid,efficiency_status='ERROR',efficiency_ratio=None,efficiency_pct=None,
                    missing_reason=error,t_work_s=None,t_break_s=None,reference_complete=False)
                segments=[];stops=[]
            with store.conn:
                for layer in ('field_efficiency','efficiency_segments','efficiency_stops'):
                    store.conn.execute(f'DELETE FROM "{layer}" WHERE field_id=?',(fid,))
                store.add_features('field_efficiency',[{**result,'field_id':fid,'geometry':geometry}])
                store.add_features('efficiency_segments',segments);store.add_features('efficiency_stops',stops)
                store.conn.execute('UPDATE efficiency_checkpoint SET state=?,error=? WHERE field_id=?',(state,error,fid))
                store.conn.execute('UPDATE field_results SET efficiency_status=? WHERE field_id=?',(result['efficiency_status'],fid))
                store.event(fid,'EFFICIENCY_COMMITTED',detail=state)
            processed+=1
            if processed%50==0:print(f'效率逐田提交 {processed} | {fid}',flush=True)
            if stop_after is not None and processed>=stop_after:break
        cursor.close()
        try:
            source_stable=(sha256(source)==contract['source_gpkg_sha256'] and
                load_config(config_path)==config and sha256(root/'src/calculate_work_time_efficiency.py')==contract['script_sha256']
                and sha256(Path(__file__))==contract['io_sha256'])
        except (OSError,ValueError):source_stable=False
        if not source_stable:
            with store.conn:
                store.conn.execute("UPDATE field_efficiency SET efficiency_status='INCOMPLETE',efficiency_ratio=NULL,efficiency_pct=NULL,reference_complete=0,missing_reason='INPUT_CHANGED_DURING_CALCULATION',payload_json=json_set(payload_json,'$.efficiency_status','INCOMPLETE','$.efficiency_ratio',NULL,'$.efficiency_pct',NULL,'$.reference_complete',json('false'),'$.missing_reason','INPUT_CHANGED_DURING_CALCULATION')")
                store.conn.execute("UPDATE field_results SET efficiency_status='INCOMPLETE'")
                store.conn.execute("UPDATE efficiency_checkpoint SET state='FAILED',error='INPUT_CHANGED_DURING_CALCULATION'")
                invalid=_incremental_summary(store);invalid.update(status='INPUT_CHANGED',input_integrity_passed=False)
                store.put_metadata('efficiency_summary',invalid)
            atomic_json(out/'summary.json',invalid)
            raise ValueError('INPUT_CHANGED_DURING_CALCULATION')
        summary=_incremental_summary(store)
        counts=dict(store.conn.execute('SELECT state,COUNT(*) FROM efficiency_checkpoint GROUP BY state').fetchall())
        summary.update(status='INTERRUPTED' if counts.get('PENDING') else 'COMPLETED',
            efficiency_checkpoint_counts=counts,scope=config['scope'],committed_this_run=processed,
            input_batch=str(batch),source_gpkg_sha256=contract['source_gpkg_sha256'],
            total_wall_seconds=time.perf_counter()-started)
        with store.conn:store.put_metadata('efficiency_summary',summary)
        atomic_json(out/'summary.json',summary)
        return summary
    finally:store.close()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(run_batch(args.swath_bundle, args.out, workers=args.workers,
        field_id=args.field_id, route_config=args.route_config))
