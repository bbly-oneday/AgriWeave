"""V7主体条带阶段的单田/GPKG批处理、导出与验收。只调用场景、冻结分区及主体条带，不组装最终路线。输出swath_results.gpkg、米制验算几何与逐田状态，协议不同于下游reference_routes.gpkg的精简四层。本模块源码实现数据准备，生成的场景/结果写入outputs，不写入src。"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time
import traceback
from typing import Any

import numpy as np
from pyproj import CRS, Transformer
from shapely import affinity, force_2d, make_valid, set_precision
from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                              MultiPoint, MultiPolygon, Point, Polygon, mapping)
from shapely.ops import transform
from shapely.validation import explain_validity

PROJECT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io_utils import atomic_json  # noqa: E402
from planner import analyze_work_regions, generate_region_swaths  # noqa: E402
from scene import load_scene, normalize_input_polygon, polygons  # noqa: E402


_RUNTIME_SOURCES = (
    "main.py", "swath_batch.py", "swath_planner.py", "validator.py",
    "planner.py", "scene.py", "io_utils.py", "swath_seams.py",
)


def _runtime_source_hashes() -> dict[str, str]:
    """记录本批真正执行的源码摘要；源码变更后不能沿用旧批次通过结论。
    
    Pin the executable source snapshot used by a batch acceptance claim."""
    return {name: hashlib.sha256((SRC / name).read_bytes()).hexdigest()
            for name in _RUNTIME_SOURCES}


def _safe_value(value: Any) -> Any:
    """将来源属性转换为可序列化值，保留原始标识供逐田对账。"""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _acceptance_exit_code(status_counts: dict[str, int]) -> int:
    """只有非空批次且全部主体条带通过才返回0；退出码不证明整田路线可执行。
    
    Return zero only when at least one field exists and every field passed."""
    total = sum(int(value) for value in status_counts.values())
    accepted = int(status_counts.get("SWATHS_COMPLETE", 0))
    return 0 if total > 0 and accepted == total else 2


def _prepare_fields(input_path: Path, config_path: Path, prepared: Path,
                    layer: str | None) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """将原始GIS田块转换为逐田米制Scene文件，写到输出目录；这里实现数据准备，不把生成文件写进src。
    
    Build per-field metric Scene JSON files inside this formal src module."""
    import geopandas as gpd
    input_path, config_path = input_path.resolve(), config_path.resolve()
    prepared.mkdir(parents=True, exist_ok=True)
    if any(prepared.iterdir()):
        raise ValueError(f"准备目录非空，拒绝覆盖：{prepared}")
    layers = gpd.list_layers(input_path)
    spatial_layers = layers.loc[layers.geometry_type.notna(), "name"].tolist()
    if layer is None:
        if len(spatial_layers) != 1:
            raise ValueError(f"必须用 --layer 指定输入图层，可选：{spatial_layers}")
        layer = spatial_layers[0]
    if layer not in spatial_layers:
        raise ValueError(f"不是可用空间图层：{layer}")
    frame = gpd.read_file(input_path, layer=layer)
    if frame.empty or frame.crs is None:
        raise ValueError("输入图层为空或缺少 CRS")
    if "field_id" in frame:
        if frame.field_id.isna().any() or frame.field_id.astype(str).duplicated().any():
            raise ValueError("field_id 不能为空或重复")
    source_crs = CRS.from_user_input(frame.crs)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("crs") == "LOCAL_METRIC":
        raise ValueError("GIS 输入不能用 LOCAL_METRIC 覆盖原 CRS")
    if not config.get("crs") and any(config.get(key) is not None
                                      for key in ("travel", "obstacles", "start", "end")):
        raise ValueError("配置含空间坐标时必须显式提供其米制 CRS")
    config_crs = CRS.from_user_input(config["crs"]) if config.get("crs") else None
    source_uses_metres = (
        source_crs.axis_info is not None
        and len(source_crs.axis_info) >= 2
        and all(abs(axis.unit_conversion_factor - 1.0) <= 1e-9
                for axis in source_crs.axis_info[:2])
    )
    # 工程坐标可能已经以米计量，但PROJ不将其分类为投影CRS；只有配置声明同一CRS时才直接保留，不能误当经纬度转换。
    local_metric = bool(source_uses_metres and config_crs is not None
                        and source_crs.equals(config_crs))
    entries = []
    normalized = []
    scenes_dir = prepared / "scenes"
    scenes_dir.mkdir()
    for index, (_, row) in enumerate(frame.iterrows()):
        geometry = normalize_input_polygon(row.geometry, f"第 {index} 行")
        normalized.append(geometry)
        field_id = str(row["field_id"]) if "field_id" in frame else f"feature_{index}"
        source_geometry_series = gpd.GeoSeries([geometry], crs=frame.crs)
        if local_metric:
            projected = geometry
            # 场景读取器接受明确的本地米制标记，包括PROJ未分类为投影坐标的工程米制CRS。
            metric_crs = "LOCAL_METRIC"
        else:
            metric = config_crs if config_crs is not None else source_geometry_series.estimate_utm_crs()
            if (metric is None or not metric.is_projected
                    or any(abs(axis.unit_conversion_factor - 1.0) > 1e-9
                           for axis in metric.axis_info[:2])):
                raise ValueError(f"第 {index} 行无法确定米制投影")
            metric_crs = metric.to_string()
            projected = source_geometry_series.to_crs(metric).iloc[0]
        if projected.is_empty or not projected.is_valid or projected.area <= 0:
            raise ValueError(f"第 {index} 行投影后几何无效")
        scene_data = dict(config)
        scene_data.update(name=field_id, crs=metric_crs, target=mapping(projected))
        filename = f"{index:04d}.json"
        (scenes_dir / filename).write_text(
            json.dumps(scene_data, ensure_ascii=False, allow_nan=False), encoding="utf-8",
        )
        hole_count = sum(len(part.interiors) for part in polygons(projected))
        entries.append({"feature_index": index, "field_id": field_id,
                        "scene_path": str(scenes_dir / filename),
                        "work_crs": metric_crs, "target_area_m2": float(projected.area),
                        "hole_count": hole_count})
    frame = frame.copy()
    frame.geometry = [MultiPolygon(polygons(geometry)) for geometry in normalized]
    frame.to_file(prepared / "source_fields.gpkg", layer="source_fields",
                  driver="GPKG", index=False)
    manifest = {
        "input": str(input_path), "input_layer": layer,
        "source_crs": source_crs.to_string(),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "projection_policy": "input_local_metric" if local_metric else "per_feature_local_utm_or_config_crs",
        "fields": entries,
    }
    atomic_json(prepared / "manifest.json", manifest)
    return frame, manifest


def _process_scene(task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    """单田加载场景、执行冻结分区及主体条带阶段，异常转换为该田失败记录，保留原因。"""
    index, entry = task
    started = time.perf_counter()
    try:
        scene = load_scene(entry["scene_path"])
        partition = analyze_work_regions(scene)
        swaths = generate_region_swaths(scene, partition)
        return {
            "index": index, "field_id": entry["field_id"], "status": swaths.status,
            "elapsed_s": time.perf_counter() - started, "scene": scene,
            "partition": partition, "swaths": swaths, "error": "",
        }
    except Exception as exc:
        return {"index": index, "field_id": entry["field_id"],
                "status": "SWATHS_WORKER_ERROR", "elapsed_s": time.perf_counter() - started,
                "scene": None, "partition": None, "swaths": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8)}


def _output_geometry(geometry, scene, source_crs: str):
    """将局部几何加回原点并转换成显示坐标；用于导出，米制验算仍使用原始局部几何。"""
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    geometry = force_2d(geometry)
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        grid = max(scene.settings.geometry_epsilon_m / 100.0, 1e-9)
        geometry = set_precision(geometry, grid, mode="valid_output")
    restored = affinity.translate(geometry, xoff=scene.origin[0], yoff=scene.origin[1])
    if not restored.is_valid and restored.geom_type in {"Polygon", "MultiPolygon"}:
        fixed = make_valid(restored)
        if abs(fixed.area - restored.area) > scene.settings.geometry_epsilon_m:
            raise ValueError("输出坐标恢复时几何修复超出数值容差")
        restored = fixed
    if scene.crs == "LOCAL_METRIC" or CRS.from_user_input(scene.crs).equals(
            CRS.from_user_input(source_crs)):
        result = restored
    else:
        transformer = Transformer.from_crs(scene.crs, source_crs, always_xy=True)
        result = transform(transformer.transform, restored)
    if not result.is_valid and result.geom_type in {"Polygon", "MultiPolygon"}:
        result = make_valid(result)
    if not result.is_valid:
        raise ValueError("转换到输入 CRS 后存在无效几何")
    if result.geom_type in {"Polygon", "MultiPolygon"}:
        return MultiPolygon(polygons(result))
    if result.geom_type in {"LineString", "MultiLineString"}:
        parts = [result] if result.geom_type == "LineString" else list(result.geoms)
        return MultiLineString(parts)
    return result


def _field_rows(result: dict[str, Any], source_crs: str,
                feature_row: Any, field_index: int) -> dict[str, Any]:
    """汇总单田来源属性、分区、条带与约束记录；计算耗时和作业时间不能混用。"""
    scene, analysis, stage = result["scene"], result["partition"], result["swaths"]
    attrs = {key: _safe_value(value) for key, value in feature_row.items()
             if key != "geometry"}
    status = result["status"]
    if stage is None:
        attrs.update(swath_status=status, swath_regions=0, swath_segments=0,
                     swath_failed_regions=0, swath_elapsed_s=result["elapsed_s"],
                     acceptance_passed=False, seam_quality_status="NOT_EVALUATED")
        return {"source": attrs, "regions": [], "mains": [], "required_mains": [],
                "headlands": [],
                "pending": [], "segments": [], "sweeps": [], "turns": [],
                "seam_footprints": [], "seam_pairs": [], "seam_task_overlaps": [],
                "seam_assignments": [],
                "short_work_review": [],
                "metric_required": [], "metric_sweeps": [],
                "connections": [], "runs": [], "run_members": [],
                "field": {"field_index": field_index, "field_id": result["field_id"],
                          "status": status, "acceptance_passed": False,
                          "failed_regions": 0, "elapsed_s": result["elapsed_s"],
                          "error": result.get("error", "")}}
    successful = sum(region.status == "SWATHS_COMPLETE" for region in stage.regions)
    failed = len(stage.regions) - successful
    segment_count = sum(region.segment_count for region in stage.regions)
    attrs.update(swath_status=stage.status, swath_regions=len(stage.regions),
                 swath_segments=segment_count, swath_failed_regions=failed,
                 swath_elapsed_s=stage.elapsed_s,
                 acceptance_passed=bool(stage.acceptance_passed),
                 seam_quality_status=stage.seam_quality_status,
                 seam_extra_area_m2=stage.checks.get("seam_extra_area_m2"),
                 seam_overlap_footprint_area_m2=stage.checks.get("seam_overlap_footprint_area_m2"),
                 seam_centerline_outside_m=stage.checks.get("seam_centerline_outside_m"),
                 seam_budget_m2=stage.checks.get("seam_budget_m2"),
                 seam_local_task_replacements=stage.checks.get(
                     "seam_local_task_replacements"),
                 seam_task_overlap_count=stage.checks.get(
                     "seam_task_overlap_count"),
                 dispatch_dependency_count=stage.checks.get("dispatch_dependency_count"),
                 dispatch_assisted_required_area_m2=stage.checks.get(
                     "dispatch_assisted_required_area_m2"),
                 independent_dispatch_status=stage.checks.get(
                     "independent_dispatch_status"),
                 work_length_status=stage.checks.get("work_length_status"),
                 short_work_removed_count=stage.checks.get("short_work_removed_count"),
                 body_covered_target_fraction=stage.body_covered_target_fraction,
                 headland_reserved_fraction=stage.headland_reserved_fraction,
                 pending_target_fraction=stage.pending_target_fraction,
                 required_main_area_fraction=(
                     float(stage.checks.get("whole_field_required_area_m2", 0.0))
                     / stage.target_area_m2 if stage.target_area_m2 else 0.0
                 ),
                 whole_field_required_area_m2=stage.checks.get(
                     "whole_field_required_area_m2"),
                 whole_field_uncovered_area_m2=stage.checks.get(
                     "whole_field_uncovered_area_m2"),
                 area_ledger_delta_m2=stage.area_ledger_delta_m2,
                 area_sum_delta_m2=stage.checks.get("area_sum_delta_m2"),
                 area_pairwise_overlap_m2=stage.checks.get("area_pairwise_overlap_m2"),
                 joint_alignment_attempt_count=stage.checks.get(
                     "joint_alignment_attempt_count", 0),
                 joint_aligned_connection_count=stage.checks.get(
                     "joint_aligned_connection_count", 0),
                 joint_uninspected_connection_count=stage.checks.get(
                     "joint_uninspected_connection_count", 0),
                 measurement_crs=str(scene.crs), output_crs=str(source_crs))
    rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("regions", "mains", "required_mains", "headlands",
                              "pending", "segments", "sweeps", "turns",
                              "seam_footprints", "seam_pairs", "seam_task_overlaps",
                              "seam_assignments",
                              "short_work_review",
                              "metric_required", "metric_sweeps", "connections",
                              "runs", "run_members")
    }
    output_metadata = {
        "measurement_crs": str(scene.crs),
        "output_crs": str(source_crs),
        "geometry_precision_m": max(scene.settings.geometry_epsilon_m / 100.0, 1e-9),
        "local_origin_x": float(scene.origin[0]),
        "local_origin_y": float(scene.origin[1]),
        "geometry_transform_policy": "local metric geometry; restore local origin; transform to output CRS",
    }
    metric_geometry_crs = (
        source_crs if scene.crs == "LOCAL_METRIC" and source_crs != "LOCAL_METRIC"
        else str(scene.crs)
    )
    longitudinal_reach = max(
        abs(scene.vehicle.implement_offset_m) + scene.vehicle.implement_length_m / 2,
        scene.vehicle.front_m, scene.vehicle.rear_m,
    )
    headland_base = (scene.vehicle.min_turn_radius_m + longitudinal_reach
                     + scene.vehicle.safety_margin_m)
    for region in analysis.regions:
        rows["regions"].append({"field_index": field_index, "field_id": scene.name,
                                "region_id": region.region_id,
                                "area_m2": float(region.geometry.area),
                                **output_metadata,
                                "sequence_index": int(region.sequence_index),
                                "reason": region.reason,
                                "preferred_angle_deg": float(region.preferred_angle_deg),
                                "geometry": _output_geometry(region.geometry, scene, source_crs)})
    for region in stage.regions:
        common = {"field_index": field_index, "field_id": scene.name,
                  "region_id": region.region_id, "status": region.status,
                  "angle_deg": region.angle_deg, "phase_m": region.phase_m,
                  "headland_m": region.headland_m,
                  "headland_base_m": float(headland_base),
                  "headland_multiplier": (
                      float(region.headland_m / headland_base)
                      if region.headland_m is not None and headland_base > 0 else None
                  ),
                  "headland_width_source": (
                      "min_turn_radius + max(|implement_offset| + implement_length/2, front, rear) + safety_margin, times selected stage multiplier"
                  ),
                  **output_metadata,
                  "side_clearance_m": region.side_clearance_m,
                  "segment_count": region.segment_count,
                  "main_area_m2": float(region.main_area.area),
                  "required_main_area_m2": float(region.required_main_area.area),
                  "headland_area_m2": float(region.headland_area.area),
                  "pending_area_m2": float(region.pending_area.area),
                  "failure_reason": region.failure_reason,
                  "body_status": region.body_status,
                  "local_turn_status": region.local_turn_status,
                  "local_turn_reason": region.local_turn_reason,
                  "turn_search_stop_reason": region.turn_search_stop_reason,
                  "selected_turn_failure_codes": ",".join(region.selected_turn_failure_codes),
                  "route_status": region.route_status,
                  "body_feasible_min_segments": region.body_feasible_min_segments,
                  "pre_seam_min_segments": region.pre_seam_segment_count,
                  "final_selected_segments": region.segment_count,
                  "turn_certified_min_segments": region.turn_certified_min_segments,
                  "search_candidates_uninspected": region.search_candidates_uninspected,
                  }
        if not region.required_main_area.is_empty:
            rows["required_mains"].append({
                **common, "geometry": _output_geometry(
                    region.required_main_area, scene, source_crs,
                ),
            })
            rows["metric_required"].append({
                **common, "metric_crs": metric_geometry_crs,
                "geometry": affinity.translate(
                    region.required_main_area, xoff=scene.origin[0], yoff=scene.origin[1],
                ),
            })
        if not region.main_area.is_empty:
            rows["mains"].append({**common, "geometry": _output_geometry(region.main_area, scene, source_crs)})
        if not region.headland_area.is_empty:
            rows["headlands"].append({**common, "geometry": _output_geometry(region.headland_area, scene, source_crs)})
        if not region.pending_area.is_empty:
            rows["pending"].append({**common, "geometry": _output_geometry(region.pending_area, scene, source_crs)})
        for segment in region.segments:
            segment_row = {"field_index": field_index, "field_id": scene.name,
                           "region_id": region.region_id, "task_id": segment["task_id"],
                           "row_index": int(segment["row_index"]),
                           "suggested_order": int(segment["suggested_order"]),
                           "angle_deg": float(segment["angle_deg"]),
                           "heading_deg": float(segment["heading_deg"]),
                           "phase_m": float(segment["phase_m"]),
                           "length_m": float(segment["length_m"]),
                           "implement_on": bool(segment["implement_on"]),
                           "geometry": _output_geometry(segment["geometry"], scene, source_crs)}
            rows["segments"].append(segment_row)
            rows["sweeps"].append({"field_index": field_index, "field_id": scene.name,
                                   "region_id": region.region_id, "task_id": segment["task_id"],
                                   "area_m2": float(segment["work_sweep"].area),
                                   **output_metadata,
                                   "geometry": _output_geometry(segment["work_sweep"], scene, source_crs)})
            rows["metric_sweeps"].append({
                "field_index": field_index, "field_id": scene.name,
                "region_id": region.region_id, "task_id": segment["task_id"],
                "area_m2": float(segment["work_sweep"].area),
                "metric_crs": metric_geometry_crs,
                "geometry": affinity.translate(
                    segment["work_sweep"], xoff=scene.origin[0], yoff=scene.origin[1],
                ),
            })
        for turn in region.turn_checks:
            turn_row = {key: value for key, value in turn.items() if key != "geometry"}
            turn_row.update(field_index=field_index,
                            geometry=_output_geometry(turn["geometry"], scene, source_crs))
            turn_row["failure_codes"] = ",".join(turn_row.get("failure_codes", []))
            if turn_row.get("failure_xy_local_m") is not None:
                turn_row["failure_xy_local_m"] = json.dumps(
                    turn_row["failure_xy_local_m"], separators=(",", ":"),
                )
            rows["turns"].append(turn_row)
    if not stage.seam_overlap_footprint.is_empty:
        rows["seam_footprints"].append({
            "field_index": field_index, "field_id": scene.name,
            "area_m2": float(stage.seam_overlap_footprint.area),
            "seam_quality_status": stage.seam_quality_status,
            **output_metadata,
            "geometry": _output_geometry(stage.seam_overlap_footprint, scene, source_crs),
        })
    for pair in stage.seam_pairs:
        if pair["geometry"].is_empty:
            continue
        rows["seam_pairs"].append({
            **{key: value for key, value in pair.items() if key != "geometry"},
            "field_index": field_index, "field_id": scene.name,
            **output_metadata,
            "geometry": _output_geometry(pair["geometry"], scene, source_crs),
        })
    for pair in stage.seam_task_overlaps:
        rows["seam_task_overlaps"].append({
            **{key: value for key, value in pair.items() if key != "geometry"},
            "field_index": field_index, "field_id": scene.name,
            **output_metadata,
            "geometry": _output_geometry(pair["geometry"], scene, source_crs),
        })
    for assignment in stage.seam_assignments:
        rows["seam_assignments"].append({
            **{key: value for key, value in assignment.items() if key != "geometry"},
            "field_index": field_index, "field_id": scene.name,
            "assignment_scope": "REQUIRED_MAIN_ONLY",
            **output_metadata,
            "geometry": _output_geometry(assignment["geometry"], scene, source_crs),
        })
    for review in stage.short_work_review:
        rows["short_work_review"].append({
            **{key: value for key, value in review.items() if key != "geometry"},
            "field_index": field_index, "field_id": scene.name,
            **output_metadata,
            "geometry": _output_geometry(review["geometry"], scene, source_crs),
        })
    for connection in stage.connections:
        row = {key: value for key, value in connection.items() if key != "geometry"}
        row["field_index"] = field_index
        row["geometry"] = _output_geometry(connection["geometry"], scene, source_crs)
        rows["connections"].append(row)
    for row in stage.continuous_runs:
        converted = {key: value for key, value in row.items() if key != "geometry"}
        converted["field_index"] = field_index
        converted["geometry"] = _output_geometry(row["geometry"], scene, source_crs)
        rows["runs"].append(converted)
    for row in stage.run_members:
        converted = {key: value for key, value in row.items() if key != "geometry"}
        converted["field_index"] = field_index
        converted["geometry"] = _output_geometry(row["geometry"], scene, source_crs)
        rows["run_members"].append(converted)
    field_summary = {
        "field_index": field_index, "field_id": scene.name, "status": stage.status,
        "acceptance_passed": bool(stage.acceptance_passed),
        "body_status": "PASS" if stage.status == "SWATHS_COMPLETE" else "PARTIAL_OR_FAILED",
        "seam_quality_status": stage.seam_quality_status,
        "seam_extra_area_m2": stage.checks.get("seam_extra_area_m2"),
        "seam_overlap_footprint_area_m2": stage.checks.get("seam_overlap_footprint_area_m2"),
        "seam_pair_overlap_sum_m2": stage.checks.get("seam_pair_overlap_sum_m2"),
        "seam_centerline_outside_m": stage.checks.get("seam_centerline_outside_m"),
        "seam_budget_m2": stage.checks.get("seam_budget_m2"),
        "seam_excess_pair_count": stage.checks.get("seam_excess_pair_count"),
        "seam_audit_elapsed_s": stage.checks.get("seam_audit_elapsed_s"),
        "seam_coordination_elapsed_s": stage.checks.get("seam_coordination_elapsed_s"),
        "seam_candidates_inspected": stage.checks.get("seam_candidates_inspected"),
        "seam_candidate_replacements": stage.checks.get("seam_candidate_replacements"),
        "seam_segments_removed": stage.checks.get("seam_segments_removed"),
        "seam_pair_combinations_inspected": stage.checks.get(
            "seam_pair_combinations_inspected"),
        "seam_pair_replacements": stage.checks.get("seam_pair_replacements"),
        "seam_extra_segments_selected": stage.checks.get(
            "seam_extra_segments_selected"),
        "seam_extra_candidate_inspected": stage.checks.get(
            "seam_extra_candidate_inspected"),
        "seam_extra_candidate_accepted": stage.checks.get(
            "seam_extra_candidate_accepted"),
        "seam_search_stop_reason": stage.checks.get("seam_search_stop_reason"),
        "seam_local_task_candidates_inspected": stage.checks.get(
            "seam_local_task_candidates_inspected"),
        "seam_local_task_replacements": stage.checks.get(
            "seam_local_task_replacements"),
        "seam_local_task_operation_counts": json.dumps(stage.checks.get(
            "seam_local_task_operation_counts", {}), sort_keys=True),
        "seam_local_task_stop_reason": stage.checks.get(
            "seam_local_task_stop_reason"),
        "seam_local_task_round_budget": stage.checks.get(
            "seam_local_task_round_budget"),
        "seam_local_task_rounds_executed": stage.checks.get(
            "seam_local_task_rounds_executed"),
        "seam_local_task_elapsed_s": stage.checks.get(
            "seam_local_task_elapsed_s"),
        "seam_task_overlap_count": stage.checks.get("seam_task_overlap_count"),
        "dispatch_dependency_count": stage.checks.get("dispatch_dependency_count"),
        "dispatch_dependent_region_count": stage.checks.get(
            "dispatch_dependent_region_count"),
        "dispatch_assisted_required_area_m2": stage.checks.get(
            "dispatch_assisted_required_area_m2"),
        "independent_dispatch_status": stage.checks.get(
            "independent_dispatch_status"),
        "work_length_status": stage.checks.get("work_length_status"),
        "minimum_work_segment_m": stage.checks.get("minimum_work_segment_m"),
        "short_work_removed_count": stage.checks.get("short_work_removed_count"),
        "short_work_removed_sub_1m_count": stage.checks.get(
            "short_work_removed_sub_1m_count"),
        "short_work_review_retained_count": stage.checks.get(
            "short_work_review_retained_count"),
        "short_work_review_threshold_m": stage.checks.get(
            "short_work_review_threshold_m"),
        "short_work_unresolved_count": stage.checks.get("short_work_unresolved_count"),
        "implement_lag_certification": stage.checks.get("implement_lag_certification"),
        "local_turn_status": (
            "PASS" if successful and all(
                region.turn_status == "PASS" for region in stage.regions
                if region.status == "SWATHS_COMPLETE"
            ) else "NOT_FULLY_CERTIFIED"
        ),
        "unverified_turn_count": sum(
            region.status == "SWATHS_COMPLETE" and region.turn_status != "PASS"
            for region in stage.regions
        ),
        "selected_turn_failed_region_count": sum(
            region.status == "SWATHS_COMPLETE"
            and region.local_turn_reason.startswith("SELECTED_DIRECT_TURN_FAILED")
            for region in stage.regions
        ),
        "selected_turn_not_checked_region_count": sum(
            region.status == "SWATHS_COMPLETE"
            and region.local_turn_reason == "SELECTED_TURN_NOT_CHECKED"
            for region in stage.regions
        ),
        "route_status": "NOT_ASSEMBLED",
        "operational_review_status": "OPERATIONAL_REVIEW_REQUIRED",
        "operational_review_reason": "measured implement on/off lag and complete transfer route are unavailable",
        "target_area_m2": stage.target_area_m2,
        "region_count": len(stage.regions), "successful_regions": successful,
        "failed_regions": failed, "regional_segment_count": segment_count,
        "continuous_run_count": len(stage.continuous_runs),
        "joint_alignment_attempt_count": stage.checks.get("joint_alignment_attempt_count", 0),
        "joint_alignment_candidate_count": stage.checks.get("joint_alignment_candidate_count", 0),
        "joint_alignment_unattempted_candidate_count": stage.checks.get(
            "joint_alignment_unattempted_candidate_count", 0),
        "joint_alignment_candidate_budget": stage.checks.get("joint_alignment_candidate_budget", 0),
        "joint_aligned_connection_count": stage.checks.get("joint_aligned_connection_count", 0),
        "joint_candidate_continuous_run_count": stage.checks.get(
            "joint_candidate_continuous_run_count", 0),
        "joint_uninspected_connection_count": stage.checks.get(
            "joint_uninspected_connection_count", 0),
        "joint_alignment_elapsed_s": stage.checks.get("joint_alignment_elapsed_s", 0.0),
        "covered_main_area_m2": stage.covered_main_area_m2,
        "body_covered_target_fraction": stage.body_covered_target_fraction,
        "headland_reserved_fraction": stage.headland_reserved_fraction,
        "required_main_area_fraction": (
            float(stage.checks.get("whole_field_required_area_m2", 0.0))
            / stage.target_area_m2 if stage.target_area_m2 else 0.0
        ),
        "whole_field_required_area_m2": stage.checks.get(
            "whole_field_required_area_m2"),
        "pending_area_m2": stage.pending_area_m2,
        "pending_target_fraction": stage.pending_target_fraction,
        "whole_field_uncovered_area_m2": stage.checks.get("whole_field_uncovered_area_m2"),
        "whole_field_max_uncovered_component_m2": stage.checks.get(
            "whole_field_max_uncovered_component_m2"),
        "whole_field_coverage_tolerance_m2": stage.checks.get(
            "whole_field_coverage_tolerance_m2"),
        "area_ledger_delta_m2": stage.area_ledger_delta_m2,
        "area_ledger_closed": stage.checks.get("area_ledger_closed", False),
        "area_sum_delta_m2": stage.checks.get("area_sum_delta_m2"),
        "area_pairwise_overlap_m2": stage.checks.get("area_pairwise_overlap_m2"),
        "area_union_area_delta_m2": stage.checks.get("area_union_area_delta_m2"),
        "area_outside_region_m2": stage.checks.get("area_outside_region_m2"),
        "measurement_crs": str(scene.crs),
        "output_crs": str(source_crs),
        "geometry_precision_m": output_metadata["geometry_precision_m"],
        "geometry_transform_policy": output_metadata["geometry_transform_policy"],
        "elapsed_s": stage.elapsed_s,
        "candidate_count": sum(region.candidate_count for region in stage.regions),
        "f2c_call_count": sum(region.f2c_call_count for region in stage.regions),
        "shortest_segment_m": min(
            (segment["length_m"] for region in stage.regions for segment in region.segments),
            default=None,
        ),
        "segments_shorter_than_1m": sum(
            segment["length_m"] < 1.0 for region in stage.regions for segment in region.segments
        ),
        "segments_shorter_than_5m": sum(
            segment["length_m"] < 5.0 for region in stage.regions for segment in region.segments
        ),
        "profile_direction_phase_generation_s": sum(
            region.profile.get("direction_phase_generation_s", 0.0) for region in stage.regions
        ),
        "profile_candidate_geometry_s": sum(
            region.profile.get("candidate_geometry_s", 0.0) for region in stage.regions
        ),
        "profile_f2c_line_generation_s": sum(
            region.profile.get("f2c_line_generation_s", 0.0) for region in stage.regions
        ),
        "profile_turn_validation_s": sum(
            region.profile.get("turn_validation_s", 0.0) for region in stage.regions
        ),
        "profile_selection_reconciliation_s": sum(
            region.profile.get("selection_reconciliation_s", 0.0) for region in stage.regions
        ),
        "search_candidates_uninspected": sum(
            region.search_candidates_uninspected for region in stage.regions
        ),
        "frozen_partition_checks_passed": bool(
            analysis.checks.get("target_unchanged", False)
            and analysis.checks.get("target_fully_assigned", False)
            and analysis.checks.get("regions_non_overlapping", False)
        ),
        "source_target_unchanged": stage.checks["source_target_unchanged"],
        "source_travel_unchanged": stage.checks["source_travel_unchanged"],
        "error": "",
    }
    return {"source": attrs, **rows, "field": field_summary,
            "metric_crs": str(scene.crs),
        "regions_summary": [{
                "field_index": field_index, "field_id": scene.name,
                "region_id": region.region_id, "status": region.status,
                "turn_status": region.turn_status,
                "body_status": region.body_status,
                "local_turn_status": region.local_turn_status,
                "local_turn_reason": region.local_turn_reason,
                "turn_search_stop_reason": region.turn_search_stop_reason,
                "selected_turn_failure_codes": ",".join(region.selected_turn_failure_codes),
                "route_status": region.route_status,
                "angle_deg": region.angle_deg, "phase_m": region.phase_m,
                "headland_m": region.headland_m,
                "headland_base_m": float(headland_base),
                "headland_multiplier": (
                    float(region.headland_m / headland_base)
                    if region.headland_m is not None and headland_base > 0 else None
                ),
                "headland_width_source": "vehicle turn radius + longitudinal body/tool reach + safety margin, times selected stage multiplier",
                "side_clearance_m": region.side_clearance_m,
                "segment_count": region.segment_count,
                "amax_area_m2": region.amax_area_m2,
                "required_main_area_m2": region.required_main_area_m2,
                "body_feasible_min_segments": region.body_feasible_min_segments,
                "pre_seam_min_segments": region.pre_seam_segment_count,
                "final_selected_segments": region.segment_count,
                "turn_certified_min_segments": region.turn_certified_min_segments,
                "search_candidates_uninspected": region.search_candidates_uninspected,
                "operational_review_status": "OPERATIONAL_REVIEW_REQUIRED",
                "operational_review_reason": "measured implement on/off lag and complete transfer route are unavailable",
                "retained_fraction": region.retained_fraction,
                "main_area_m2": float(region.main_area.area),
                "headland_area_m2": float(region.headland_area.area),
                "pending_area_m2": float(region.pending_area.area),
                "main_uncovered_area_m2": (
                    float(region.pending_area.area)
                    if region.status == "SWATHS_COMPLETE" else None),
                "shortest_segment_m": min((seg["length_m"] for seg in region.segments), default=None),
                "segments_shorter_than_1m": sum(seg["length_m"] < 1.0 for seg in region.segments),
                "segments_shorter_than_5m": sum(seg["length_m"] < 5.0 for seg in region.segments),
                "turn_checks_passed": sum(row.get("status") == "PASS" for row in region.turn_checks),
                "turn_checks_failed": sum(row.get("status") == "FAIL" for row in region.turn_checks),
                "turn_checks_not_checked": sum(
                    row.get("status") == "NOT_CHECKED" for row in region.turn_checks),
                "turn_checks_recorded": len(region.turn_checks),
                "candidate_count": region.candidate_count,
                "f2c_call_count": region.f2c_call_count,
                "elapsed_s": region.elapsed_s,
                "failure_reason": region.failure_reason,
                "search_note": region.search_note,
                **{f"profile_{key}": value for key, value in region.profile.items()},
            } for region in stage.regions]}


def _geometry_placeholder(kind: str):
    """为空图层创建时提供类型样例，随后不作为真实作业要素保留。"""
    if kind == "polygon":
        return MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])])
    if kind == "line":
        return MultiLineString([LineString([(0, 0), (1, 0)])])
    return Point(0, 0)


def _write_layer(path: Path, layer: str, records: list[dict[str, Any]],
                 crs: str, geometry_kind: str,
                 columns: list[tuple[str, Any]]) -> None:
    """写入指定GPKG图层及字段；空图层保持声明结构，不能用样例几何冒充田块结果。"""
    import geopandas as gpd
    output_crs = None if crs == "LOCAL_METRIC" else crs
    if records:
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs=output_crs)
    else:
        defaults = {name: value for name, value in columns}
        defaults["geometry"] = _geometry_placeholder(geometry_kind)
        frame = gpd.GeoDataFrame([defaults], geometry="geometry", crs=output_crs)
    mode = "w" if not path.exists() else "a"
    frame.to_file(path, layer=layer, driver="GPKG", mode=mode, index=False)
    if not records:
        # 为零要素结果保留真实、定型的GPKG图层；空图层不能插入样例要素冒充结果。
        with sqlite3.connect(path) as connection:
            quoted = '"' + layer.replace('"', '""') + '"'
            connection.execute(f"DELETE FROM {quoted}")
            connection.commit()


def _export_gpkg(out: Path, source_fields: gpd.GeoDataFrame,
                 results: list[dict[str, Any]], source_crs: str) -> dict[str, Any]:
    """导出主体条带显示GPKG与逐田结果，保留任务归属和失败状态；不是默认参考路线的精简输出协议。"""
    import geopandas as gpd
    gpkg = out / "swath_results.gpkg"
    field_rows = []
    for index, result in enumerate(results):
        attrs = dict(result["source"])
        attrs["geometry"] = source_fields.geometry.iloc[index]
        field_rows.append(attrs)
    if field_rows:
        fields = gpd.GeoDataFrame(
            field_rows, geometry="geometry",
            crs=None if source_crs == "LOCAL_METRIC" else source_crs,
        )
        fields.to_file(gpkg, layer="source_fields", driver="GPKG", index=False)
    layer_specs = [
        ("work_regions", "polygon", "regions", [("field_id", ""), ("region_id", ""), ("area_m2", 0.0), ("geometry", None)]),
        ("main_work_areas", "polygon", "mains", [("field_id", ""), ("region_id", ""), ("status", ""), ("geometry", None)]),
        ("required_main_areas", "polygon", "required_mains", [("field_id", ""), ("region_id", ""), ("required_main_area_m2", 0.0), ("geometry", None)]),
        ("headland_reserve", "polygon", "headlands", [("field_id", ""), ("region_id", ""), ("status", ""), ("geometry", None)]),
        ("pending_target_areas", "polygon", "pending", [("field_id", ""), ("region_id", ""), ("status", ""), ("geometry", None)]),
        ("swath_segments", "line", "segments", [("field_id", ""), ("region_id", ""), ("task_id", ""), ("geometry", None)]),
        ("work_sweeps", "polygon", "sweeps", [("field_id", ""), ("region_id", ""), ("task_id", ""), ("geometry", None)]),
        ("seam_overlap_footprints", "polygon", "seam_footprints", [("field_id", ""), ("area_m2", 0.0), ("geometry", None)]),
        ("seam_pair_overlaps", "polygon", "seam_pairs", [("field_id", ""), ("from_region", ""), ("to_region", ""), ("geometry", None)]),
        ("seam_task_overlaps", "polygon", "seam_task_overlaps", [("field_id", ""), ("from_task", ""), ("to_task", ""), ("geometry", None)]),
        ("seam_work_assignments", "polygon", "seam_assignments", [("field_id", ""), ("owner_region_id", ""), ("provider_region_id", ""), ("task_id", ""), ("geometry", None)]),
        ("short_work_review", "line", "short_work_review", [("field_id", ""), ("region_id", ""), ("task_id", ""), ("status", ""), ("geometry", None)]),
        ("continuous_runs", "line", "runs", [("field_id", ""), ("run_id", ""), ("geometry", None)]),
        ("run_members", "point", "run_members", [("field_id", ""), ("run_id", ""), ("region_id", ""), ("geometry", None)]),
        ("local_turn_checks", "line", "turns", [("field_id", ""), ("region_id", ""), ("status", ""), ("geometry", None)]),
        ("connection_records", "point", "connections", [("field_id", ""), ("connection_id", ""), ("geometry", None)]),
    ]
    for layer, kind, bucket, columns in layer_specs:
        records = [record for result in results for record in result.get(bucket, [])]
        _write_layer(gpkg, layer, records, source_crs, kind, columns)
    with sqlite3.connect(gpkg) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    counts = {row["name"]: len(gpd.read_file(gpkg, layer=row["name"]))
              for _, row in gpd.list_layers(gpkg).iterrows()}
    if integrity != "ok" or counts.get("source_fields") != len(results):
        raise RuntimeError(f"GeoPackage 回读校验失败：integrity={integrity}, counts={counts}")
    expected = {spec[0] for spec in layer_specs} | {"source_fields"}
    if not expected.issubset(counts):
        raise RuntimeError(f"GeoPackage 缺少图层：{sorted(expected - set(counts))}")
    metric = _export_metric_gpkg(out, results)
    return {"path": str(gpkg), "integrity_check": integrity,
            "layer_counts": counts, "crs": source_crs,
            "metric_geometry": metric}


def _export_metric_gpkg(out: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    """导出验算用米制几何；每田测量CRS应随结果保留，不能把米制长度按经纬度解读。
    
    Write check geometries in their actual per-field measurement CRS."""
    import re
    import geopandas as gpd

    gpkg = out / "swath_results_metric.gpkg"
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        for bucket, stem in (("metric_required", "required_main_areas"),
                             ("metric_sweeps", "work_sweeps")):
            for row in result.get(bucket, []):
                crs = str(row.get("metric_crs", "LOCAL_METRIC"))
                suffix = re.sub(r"[^A-Za-z0-9]+", "_", crs).strip("_") or "LOCAL_METRIC"
                if crs == "LOCAL_METRIC":
                    suffix += f"_field_{row.get('field_index', 0):04d}"
                grouped.setdefault((stem, suffix), []).append(row)
    if not grouped:
        return {"path": None, "integrity_check": "not_created", "layer_counts": {}}
    for (stem, suffix), records in sorted(grouped.items()):
        crs = str(records[0].get("metric_crs", "LOCAL_METRIC"))
        _write_layer(
            gpkg, f"{stem}_{suffix}", records, crs, "polygon",
            [("field_id", ""), ("region_id", ""), ("geometry", None)],
        )
    with sqlite3.connect(gpkg) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    counts = {row["name"]: len(gpd.read_file(gpkg, layer=row["name"]))
              for _, row in gpd.list_layers(gpkg).iterrows()}
    if integrity != "ok":
        raise RuntimeError(f"米制 GeoPackage 完整性检查失败：{integrity}")
    return {
        "path": str(gpkg), "integrity_check": integrity, "layer_counts": counts,
        "geometry_semantics": "measurement CRS coordinates with scene origin restored; one layer per CRS",
    }


def _write_plots(out: Path, source_fields: gpd.GeoDataFrame,
                 results: list[dict[str, Any]], source_crs: str) -> list[str]:
    """按结果绘制条带检查图，图片只是辅助诊断，不替代面积/安全审计。"""
    import os
    mpl_config = out / "matplotlib_config"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    os.environ.setdefault("XDG_CACHE_HOME", str(out / "cache"))
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    overview = out / "swath_overview_350.png"
    plot_crs = None if source_crs == "LOCAL_METRIC" else source_crs
    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    source_fields.boundary.plot(ax=ax, color="#777777", linewidth=0.22, alpha=0.55)
    main_rows = [record for result in results for record in result.get("mains", [])]
    if main_rows:
        gpd.GeoDataFrame(main_rows, geometry="geometry", crs=plot_crs).plot(
            ax=ax, color="#d8ecdf", edgecolor="none", alpha=0.38,
        )
    headland_rows = [record for result in results for record in result.get("headlands", [])]
    if headland_rows:
        gpd.GeoDataFrame(headland_rows, geometry="geometry", crs=plot_crs).plot(
            ax=ax, color="#f2d48d", edgecolor="none", alpha=0.30,
        )
    line_rows = [record for result in results for record in result.get("segments", [])]
    if line_rows:
        lines = gpd.GeoDataFrame(line_rows, geometry="geometry", crs=plot_crs)
        lines.plot(ax=ax, color="#12805c", linewidth=0.28, alpha=0.75)
    ax.set_title("Main-work swaths after partitioning (source CRS)")
    ax.set_axis_off()
    fig.savefig(overview, dpi=180)
    plt.close(fig)

    gallery = out / "swath_examples_first12.png"
    count = min(12, len(source_fields))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.set_axis_off()
            continue
        gpd.GeoSeries([source_fields.geometry.iloc[index].boundary], crs=plot_crs).plot(
            ax=ax, color="#333333", linewidth=0.7,
        )
        if results[index].get("mains"):
            gpd.GeoDataFrame(results[index]["mains"], geometry="geometry",
                             crs=plot_crs).plot(
                ax=ax, color="#d8ecdf", edgecolor="none", alpha=0.5,
            )
        if results[index].get("headlands"):
            gpd.GeoDataFrame(results[index]["headlands"], geometry="geometry",
                             crs=plot_crs).plot(
                ax=ax, color="#f2d48d", edgecolor="none", alpha=0.45,
            )
        rows = results[index].get("segments", [])
        if rows:
            gpd.GeoDataFrame(rows, geometry="geometry", crs=plot_crs).plot(
                ax=ax, color="#16855f", linewidth=0.45,
            )
        field_id = results[index]["field"].get("field_id", index)
        state = results[index]["field"].get("status", "ERROR")
        ax.set_title(f"{field_id} | {state}", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    fig.savefig(gallery, dpi=180)
    plt.close(fig)
    return [str(overview), str(gallery)]


def _write_summaries(out: Path, results: list[dict[str, Any]],
                     manifest: dict[str, Any], prep_s: float,
                     worker_s: float, total_s: float,
                     gpkg_result: dict[str, Any], plots: list[str],
                     workers: int,
                     source_hashes_start: dict[str, str],
                     source_hashes_end: dict[str, str]) -> dict[str, Any]:
    """保存批次状态、时间、参数和源码摘要；只认可本批实际执行的主体阶段。"""
    fields = [result["field"] for result in results]
    regions = [region for result in results for region in result.get("regions_summary", [])]
    body_ranked = sorted(
        (field for field in fields if field.get("body_covered_target_fraction") is not None),
        key=lambda field: (float(field["body_covered_target_fraction"]), field["field_id"]),
    )
    for rank, field in enumerate(body_ranked, start=1):
        field["body_coverage_review_rank"] = rank
        field["lowest_body_coverage_in_batch"] = rank == 1
    counts: dict[str, int] = {}
    for field in fields:
        counts[field["status"]] = counts.get(field["status"], 0) + 1
    runtimes = [float(field["elapsed_s"]) for field in fields]
    ordered_runtimes = sorted(runtimes)

    def percentile(fraction: float) -> float | None:
        if not ordered_runtimes:
            return None
        index = min(len(ordered_runtimes) - 1,
                    max(0, int(math.ceil(fraction * len(ordered_runtimes))) - 1))
        return ordered_runtimes[index]

    record = {
        "input": manifest.get("input"), "input_layer": manifest.get("input_layer"),
        "source_sha256": manifest.get("source_sha256"), "config": manifest.get("config"),
        "config_sha256": manifest.get("config_sha256"),
        "field_count": len(fields), "status_counts": counts,
        "seam_quality_counts": {status: sum(
            field.get("seam_quality_status") == status for field in fields)
            for status in ("PASS", "EXCESS_OVERLAP", "NOT_EVALUATED")},
        "seam_extra_area_m2": sum(float(field.get("seam_extra_area_m2") or 0.0)
                                  for field in fields),
        "seam_local_task_replacements": sum(int(
            field.get("seam_local_task_replacements") or 0) for field in fields),
        "seam_centerline_outside_m": sum(float(field.get("seam_centerline_outside_m") or 0.0)
                                         for field in fields),
        "dispatch_dependent_region_count": sum(int(
            field.get("dispatch_dependent_region_count") or 0) for field in fields),
        "dispatch_assisted_required_area_m2": sum(float(
            field.get("dispatch_assisted_required_area_m2") or 0.0) for field in fields),
        "short_work_removed_count": sum(int(
            field.get("short_work_removed_count") or 0) for field in fields),
        "short_work_review_retained_count": sum(int(
            field.get("short_work_review_retained_count") or 0) for field in fields),
        "retained_fraction_definition": (
            "selected required-main candidate area divided by the largest inspected "
            "body-feasible candidate area Amax in the same region; not whole-field completion"
        ),
        "implement_lag_certification": "PENDING_MEASURED_ON_OFF_LAG",
        "route_certification": "NOT_ASSEMBLED",
        "acceptance_passed": (source_hashes_start == source_hashes_end
                              and bool(fields) and all(
            bool(field.get("acceptance_passed", False)) for field in fields
        )),
        "source_code_stable": source_hashes_start == source_hashes_end,
        "source_code_sha256_start": source_hashes_start,
        "source_code_sha256_end": source_hashes_end,
        "failed_field_count": sum(
            not bool(field.get("acceptance_passed", False)) for field in fields
        ),
        "lowest_body_coverage_field_id": (
            body_ranked[0].get("field_id") if body_ranked else None
        ),
        "lowest_body_coverage_fraction": (
            body_ranked[0].get("body_covered_target_fraction") if body_ranked else None
        ),
        "region_count": len(regions),
        "successful_region_count": sum(region["status"] == "SWATHS_COMPLETE" for region in regions),
        "failed_region_count": sum(region["status"] != "SWATHS_COMPLETE" for region in regions),
        "regional_segment_count": sum(field.get("regional_segment_count", 0) for field in fields),
        "continuous_run_count": sum(field.get("continuous_run_count", 0) for field in fields),
        "f2c_call_count": sum(field.get("f2c_call_count", 0) for field in fields),
        "unverified_turn_count": sum(field.get("unverified_turn_count", 0) for field in fields),
        "operational_review_field_count": sum(
            field.get("operational_review_status") == "OPERATIONAL_REVIEW_REQUIRED"
            for field in fields
        ),
        "search_candidates_uninspected": sum(
            field.get("search_candidates_uninspected", 0) for field in fields
        ),
        "search_profile_s": {
            "direction_phase_generation": sum(
                field.get("profile_direction_phase_generation_s", 0.0) for field in fields
            ),
            "candidate_geometry": sum(
                field.get("profile_candidate_geometry_s", 0.0) for field in fields
            ),
            "f2c_line_generation": sum(
                field.get("profile_f2c_line_generation_s", 0.0) for field in fields
            ),
            "turn_validation": sum(
                field.get("profile_turn_validation_s", 0.0) for field in fields
            ),
            "selection_reconciliation": sum(
                field.get("profile_selection_reconciliation_s", 0.0) for field in fields
            ),
            "joint_alignment": sum(
                field.get("joint_alignment_elapsed_s", 0.0) for field in fields
            ),
            "seam_coordination": sum(
                field.get("seam_coordination_elapsed_s", 0.0) for field in fields
            ),
            "seam_local_tasks": sum(
                field.get("seam_local_task_elapsed_s", 0.0) for field in fields
            ),
            "seam_audit": sum(
                field.get("seam_audit_elapsed_s", 0.0) for field in fields
            ),
        },
        "worker_count": workers, "preparation_elapsed_s": prep_s,
        "worker_elapsed_wall_s": worker_s, "total_elapsed_s": total_s,
        "field_runtime_s_all_statuses": {
            "count": len(runtimes),
            "median": statistics.median(runtimes) if runtimes else None,
            "p90": percentile(0.90), "p95": percentile(0.95),
            "max": max(runtimes) if runtimes else None,
        },
        "gpkg": gpkg_result, "plots": plots,
        "metric_geometry_policy": gpkg_result.get("metric_geometry"),
        "geometry_measurement_note": "areas are measured in the per-field metric CRS; source-CRS geometry is transformed and carries its CRS and precision metadata",
        "scope": "scene loading, frozen partition, body swath geometry and coverage; local turns are separate evidence, and no complete route or headland work is claimed",
    }
    atomic_json(out / "swath_batch_summary.json", record)
    atomic_json(out / "field_results.json", fields)
    seam_pairs = [{key: value for key, value in row.items() if key != "geometry"}
                  for result in results for row in result.get("seam_pairs", [])]
    atomic_json(out / "seam_pairs.json", seam_pairs)
    task_overlaps = [{key: value for key, value in row.items() if key != "geometry"}
                     for result in results for row in result.get("seam_task_overlaps", [])]
    atomic_json(out / "seam_task_overlaps.json", task_overlaps)
    dependencies = [{key: value for key, value in row.items() if key != "geometry"}
                    for result in results for row in result.get("seam_assignments", [])]
    atomic_json(out / "dispatch_dependencies.json", dependencies)
    short_review = [{key: value for key, value in row.items() if key != "geometry"}
                    for result in results for row in result.get("short_work_review", [])]
    atomic_json(out / "short_work_review.json", short_review)
    with (out / "short_work_review.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        import csv
        names = list(short_review[0]) if short_review else [
            "field_id", "region_id", "task_id", "length_m", "status",
            "unique_required_area_m2"]
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(short_review)
    with (out / "dispatch_dependencies.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        import csv
        names = list(dependencies[0]) if dependencies else [
            "field_id", "owner_region_id", "provider_region_id", "task_id",
            "area_m2", "execution_rule"]
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dependencies)
    with (out / "seam_pairs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        import csv
        fieldnames = list(seam_pairs[0]) if seam_pairs else [
            "field_id", "from_region", "to_region", "overlap_area_m2", "budget_m2", "status"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(seam_pairs)
    with (out / "seam_task_overlaps.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        import csv
        names = list(task_overlaps[0]) if task_overlaps else [
            "field_id", "from_region", "to_region", "from_task", "to_task",
            "overlap_area_m2"]
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(task_overlaps)
    with (out / "field_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        import csv
        fieldnames = list(fields[0]) if fields else ["field_id", "status"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fields)
    with (out / "region_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        import csv
        fieldnames = list(regions[0]) if regions else ["field_id", "region_id", "status"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(regions)
    return record


def run_batch(input_path: Path, config_path: Path, out: Path, *,
              layer: str | None = None, workers: int = 12) -> int:
    """主体条带GPKG批入口，使用有界进程并行；输出本阶段专用结果，不套用下游参考路线的四层协议。
    
    Run the swath-only stage over every field with bounded process parallelism."""
    started = time.perf_counter()
    input_path, config_path, out = input_path.resolve(), config_path.resolve(), out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"输出目录非空，拒绝覆盖：{out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    source_hashes_start = _runtime_source_hashes()
    prep_started = time.perf_counter()
    frame, manifest = _prepare_fields(input_path, config_path, out / "prepared", layer)
    prep_s = time.perf_counter() - prep_started
    tasks = [(index, entry) for index, entry in enumerate(manifest["fields"])]
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    worker_started = time.perf_counter()
    atomic_json(out / "batch_progress.json", {"completed": 0, "total": len(tasks), "workers": workers})
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    os.environ.update(env)
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_process_scene, task): task[0] for task in tasks}
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                raw = future.result()
            except Exception as exc:
                raw = {"index": index, "field_id": manifest["fields"][index]["field_id"],
                       "status": "SWATHS_WORKER_ERROR", "elapsed_s": 0.0,
                       "scene": None, "partition": None, "swaths": None,
                       "error": f"{type(exc).__name__}: {exc}"}
            results[index] = _field_rows(raw, manifest["source_crs"],
                                         frame.iloc[index], index)
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                atomic_json(out / "batch_progress.json", {
                    "completed": completed, "total": len(tasks), "workers": workers,
                    "last_field": manifest["fields"][index]["field_id"],
                    "last_status": raw["status"],
                    "elapsed_s": time.perf_counter() - worker_started,
                })
                print(f"[{completed}/{len(tasks)}] {manifest['fields'][index]['field_id']} {raw['status']}", flush=True)
    worker_s = time.perf_counter() - worker_started
    final_results = [item for item in results if item is not None]
    export_started = time.perf_counter()
    gpkg_result = _export_gpkg(out, frame, final_results, manifest["source_crs"])
    gpkg_s = time.perf_counter() - export_started
    plot_started = time.perf_counter()
    plots = _write_plots(out, frame, final_results, manifest["source_crs"])
    plot_s = time.perf_counter() - plot_started
    total_s = time.perf_counter() - started
    source_hashes_end = _runtime_source_hashes()
    summary = _write_summaries(out, final_results, manifest, prep_s,
                               worker_s, total_s, gpkg_result, plots, workers,
                               source_hashes_start, source_hashes_end)
    summary["gpkg_export_elapsed_s"] = gpkg_s
    summary["plot_export_elapsed_s"] = plot_s
    atomic_json(out / "swath_batch_summary.json", summary)
    print(json.dumps({"status_counts": summary["status_counts"],
                      "regions": summary["region_count"],
                      "segments": summary["regional_segment_count"],
                      "total_elapsed_s": total_s,
                      "gpkg": gpkg_result["path"]}, ensure_ascii=False, indent=2))
    return (0 if summary["acceptance_passed"] else 2)


def run_single(scene, out: Path, *, source_crs: str | None = None,
               field_path: Path | None = None, layer: str | None = None,
               feature_index: int = 0) -> int:
    """单场景或单GIS要素的主体条带入口；采用新输出目录，不覆盖已有批次。"""
    import geopandas as gpd
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"输出目录非空，拒绝覆盖：{out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    source_hashes_start = _runtime_source_hashes()
    if field_path:
        frame = gpd.read_file(field_path, layer=layer)
        if not 0 <= feature_index < len(frame):
            raise ValueError("feature_index 超出输入图层范围")
        row = frame.iloc[feature_index]
        source_crs = frame.crs.to_string()
        one = gpd.GeoDataFrame([row], geometry="geometry", crs=frame.crs)
        one.geometry = [MultiPolygon(polygons(one.geometry.iloc[0]))]
        row = one.iloc[0]
        manifest = {"input": str(field_path.resolve()), "input_layer": layer,
                    "source_crs": source_crs, "fields": [{"field_id": scene.name}]}
        source_fields = one
    else:
        source_crs = source_crs or scene.crs
        geometry = _output_geometry(scene.target, scene, source_crs)
        source_fields = gpd.GeoDataFrame([{"field_id": scene.name, "geometry": geometry}],
                                         geometry="geometry",
                                         crs=None if source_crs == "LOCAL_METRIC" else source_crs)
        row = source_fields.iloc[0]
        manifest = {"input": scene.name, "input_layer": None,
                    "source_crs": source_crs, "fields": [{"field_id": scene.name}]}
    started = time.perf_counter()
    try:
        partition = analyze_work_regions(scene)
        stage = generate_region_swaths(scene, partition)
        raw = {"index": 0, "field_id": scene.name, "status": stage.status,
               "elapsed_s": time.perf_counter() - started,
               "scene": scene, "partition": partition, "swaths": stage, "error": ""}
    except Exception as exc:
        raw = {"index": 0, "field_id": scene.name, "status": "SWATHS_WORKER_ERROR",
               "elapsed_s": time.perf_counter() - started, "scene": None,
               "partition": None, "swaths": None, "error": f"{type(exc).__name__}: {exc}"}
    result = _field_rows(raw, source_crs, row, 0)
    result_list = [result]
    gpkg_result = _export_gpkg(out, source_fields, result_list, source_crs)
    plots = _write_plots(out, source_fields, result_list, source_crs)
    summary = _write_summaries(out, result_list, manifest, 0.0,
                     raw["elapsed_s"], time.perf_counter() - started,
                     gpkg_result, plots, 1,
                     source_hashes_start, _runtime_source_hashes())
    print(json.dumps({"field_id": scene.name, "status": raw["status"],
                      "output": str(out), "gpkg": gpkg_result["path"],
                      "elapsed_s": raw["elapsed_s"]}, ensure_ascii=False, indent=2))
    return (0 if summary["acceptance_passed"] else 2)
