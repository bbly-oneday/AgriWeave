"""总入口与流程控制。输入命令行、GPKG/场景及配置，输出各阶段的新批次。
这里只组织阶段、进程与预算，不把参考路线结果伪装成严格认证。
阅读顺序：命令行 main → 阶段调用 → 输出；旧完整流程的修补和效率在后两节。

分节目录：
1. 总调度入口与原完整流程
2. 原始 GPKG 准备与逐田调度
3. 原完整规划的一轮局部修补
4. 已验收路线的规划参考效率
"""
from __future__ import annotations


# ==========================================================================
# 1. 总调度入口与原完整流程
# 根据命令行决定阶段；场景、分区、主体条带与路线各自负责自己的结果。
# ==========================================================================

import argparse
import csv
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import numpy as np
from shapely import affinity
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry import mapping
from shapely.geometry import GeometryCollection
from shapely.ops import transform
from pyproj import Transformer

from scene import Scene
from scene import Vehicle
from scene import Settings
from scene import Pose
from scene import Task
from scene import Motion
from scene import Plan
from scene import Budget
from scene import load_scene
from scene import make_config
from scene import polygons
from scene import wrap
from planner import F2CBackend
from planner import Connector
from planner import candidate_specs
from planner import generate_tasks
from planner import assemble
from planner import work_motion
from validator import Validator
pass  # 已合并到本模块，直接使用下方的定义。
pass  # 已合并到本模块，直接使用下方的定义。
from io_utils import atomic_json, PROJECT_CONFIG, config_section, config_reference, project_config, scene_config, resolve_input_path, validate_execution


def output_geometry(geometry, scene: Scene):
    """将本田局部米制几何加回原点并转换为显示坐标；LOCAL_METRIC保留米制，不伪装成经纬度。"""
    projected = affinity.translate(geometry, *scene.origin)
    if scene.crs == "LOCAL_METRIC":
        return projected
    transformer = Transformer.from_crs(scene.crs, "EPSG:4326", always_xy=True)
    return transform(transformer.transform, projected)


def export_result(out: Path, scene: Scene, plan: Plan, report, *, accepted: bool) -> dict[str, str]:
    """每个 Motion 单独导出，失败方案绝不画一根跨缺口的总 LineString。"""
    prefix = "route" if accepted else "candidate_route"
    suffix = ".geojson" if scene.crs != "LOCAL_METRIC" else ".local.json"
    route_features = []
    for i, motion in enumerate(plan.motions):
        line = LineString(motion.points[:, :2])
        route_features.append({"type": "Feature", "geometry": mapping(output_geometry(line, scene)),
                               "properties": {"sequence": i, "kind": motion.kind, "task_id": motion.task_id,
                                              "implement_on": motion.implement_on, "accepted": accepted,
                                              "length_m": motion.length, "link": motion.link}})
    collection = {"type": "FeatureCollection", "features": route_features}
    if scene.crs == "LOCAL_METRIC":
        collection["coordinate_note"] = "LOCAL_METRIC metres; not RFC 7946 geographic GeoJSON"
    atomic_json(out / (prefix + suffix), collection)
    features = []
    for name, geometry in (("target", scene.target), ("travel", scene.travel),
                           ("covered", report.covered), ("uncovered", report.missing)):
        features.append({"type": "Feature", "geometry": mapping(output_geometry(geometry, scene)),
                         "properties": {"kind": name, "area_m2": geometry.area,
                                        "coverage_evaluated": report.coverage_evaluated}})
    coverage_collection = {"type": "FeatureCollection", "features": features}
    if scene.crs == "LOCAL_METRIC":
        coverage_collection["coordinate_note"] = "LOCAL_METRIC metres; not RFC 7946 geographic GeoJSON"
    coverage_name = ("coverage" if accepted else "candidate_coverage") + suffix
    atomic_json(out / coverage_name, coverage_collection)
    csv_path = out / (prefix + ".csv")
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["segment", "sample", "kind", "task_id", "x_m", "y_m", "heading_rad",
                         "outgoing_direction", "implement_on", "segment_terminal"])
        for i, motion in enumerate(plan.motions):
            for j, row in enumerate(motion.points):
                writer.writerow([i, j, motion.kind, motion.task_id, row[0] + scene.origin[0],
                                 row[1] + scene.origin[1], row[2], int(row[3]), int(motion.implement_on),
                                 int(j == len(motion.points) - 1)])
    os.replace(temporary, csv_path)
    atomic_json(out / ("validation.json" if accepted else "candidate_validation.json"), report.summary())
    return {"route": prefix + suffix, "samples": prefix + ".csv", "coverage": coverage_name}


def _scene_from_args(args) -> Scene:
    """读取单田输入并应用显式full预算覆盖；不能在这里替换Scene.target或改变冻结条带义务。"""
    scene = load_scene(args.scene or args.config, field_path=args.field,
                       feature_index=args.feature_index, layer=args.layer)
    if args.seconds is not None:
        scene.settings = replace(scene.settings, wall_time_seconds=args.seconds)
    return scene


# 只执行场景构建与几何核验，方便单独检查 target、travel 和孔洞。
# 此入口不会顺带运行分区、条带或路线搜索。
def run_scene_check(args) -> int:
    """只调用 scene.py 加载和检查场景，不进入任何路径规划模块。"""
    out = Path(args.out)
    try:
        scene = _scene_from_args(args)
        target_holes = sum(len(polygon.interiors) for polygon in polygons(scene.target))
        summary = {
            "status": "SCENE_VALID",
            "module_under_test": "src/scene.py:load_scene",
            "planner_executed": False,
            "repair_executed": False,
            "validator_executed": False,
            "name": scene.name,
            "crs": scene.crs,
            "origin": list(scene.origin),
            "travel_source": scene.travel_source,
            "travel_clearance_m": scene.settings.travel_clearance_m,
            "target_geometry_type": scene.target.geom_type,
            "travel_geometry_type": scene.travel.geom_type,
            "target_valid": bool(scene.target.is_valid),
            "travel_valid": bool(scene.travel.is_valid),
            "target_area_m2": float(scene.target.area),
            "travel_area_m2": float(scene.travel.area),
            "target_only_area_m2": float(scene.target.difference(scene.travel).area),
            "travel_only_area_m2": float(scene.travel.difference(scene.target).area),
            "target_hole_count": target_holes,
            "start": asdict(scene.start) if scene.start else None,
            "end": asdict(scene.end) if scene.end else None,
        }
        atomic_json(out / "scene_check.json", summary)
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        summary = {
            "status": "SCENE_INPUT_ERROR",
            "module_under_test": "src/scene.py:load_scene",
            "planner_executed": False,
            "repair_executed": False,
            "validator_executed": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        atomic_json(out / "scene_check.json", summary)
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


# 原完整规划的单田执行器：有限候选 → 连接 → 独立验证 → 最多有限轮修补。
# 每得到一个有效结果就保存检查点，后续原生库超时不会抹掉已保存成果。
def run_worker(args) -> int:
    """隔离执行历史full候选搜索、验证及有限修补；成功检查点保留，失败不会冒充当前推荐参考路线通过。"""
    out = Path(args.out)
    scene = _scene_from_args(args)
    budget = Budget(scene.settings.wall_time_seconds)
    audit = []
    best_failed = None
    best_passed = None
    def progress(stage, **detail):
        atomic_json(out / "progress.json", {"stage": stage, "elapsed_s": budget.elapsed, **detail})
        print(f"[{budget.elapsed:7.2f}s] {stage} {detail}", flush=True)
    progress("load_f2c")
    backend = F2CBackend(scene)
    validator = Validator(scene, budget)
    connector = Connector(scene, backend, validator, budget)
    termination = "CANDIDATES_EXHAUSTED"
    for number, spec in enumerate(candidate_specs(scene)):
        try:
            budget.check()
            progress("generate_tasks", candidate=number + 1, angle_deg=math.degrees(spec[0]),
                     pattern=spec[1], decomposed=spec[2])
            tasks, metadata = generate_tasks(scene, backend, spec, budget)
            progress("connect_tasks", candidate=number + 1, task_count=len(tasks),
                     decomposition_algorithm=metadata.get("decomposition_algorithm", "none"))
            plan = assemble(tasks, scene, connector, metadata)
            seen_orders = set()
            for round_index in range(scene.settings.max_repair_rounds + 1):
                budget.check()
                progress("validate", candidate=number + 1, repair_round=round_index)
                report = validator.validate(plan)
                record = {"candidate": number + 1, "repair_round": round_index,
                          "task_count": len(plan.tasks), "elapsed_s": budget.elapsed,
                          "candidate_spec": {k: v for k, v in plan.metadata.items() if k != "mainland_wkt_local"},
                          "validation": report.summary()}
                audit.append(record)
                atomic_json(out / "audit.json", audit)
                if report.passed:
                    efficiency = calculate_efficiency(plan, scene, report)
                    if best_passed is None or efficiency["estimated_total_time_s"] < best_passed[2]["estimated_total_time_s"]:
                        best_passed = (plan, report, efficiency)
                        files = export_result(out, scene, plan, report, accepted=True)
                        atomic_json(out / "efficiency.json", efficiency)
                        atomic_json(out / "summary.json", {"status": "PASSED", "validation": report.summary(),
                                                           "efficiency": efficiency, "files": files,
                                                           "f2c_version": backend.version,
                                                           "work_crs": scene.crs, "local_origin_m": scene.origin,
                                                           "vehicle": asdict(scene.vehicle),
                                                           "planning": asdict(scene.settings)})
                    break
                score = (int(report.motion_valid), report.coverage_fraction,
                         -sum(e.get("code") != "UNCOVERED" for e in report.issues))
                if best_failed is None or score > best_failed[0]:
                    best_failed = (score, plan, report)
                    # 在新的原生调用之前保留诊断；超时/崩溃也不会丢掉已经完成的部分。
                    export_result(out, scene, plan, report, accepted=False)
                if round_index >= scene.settings.max_repair_rounds:
                    break
                signature = tuple((t.task_id, t.start.x, t.start.y, t.end.x, t.end.y) for t in plan.tasks)
                if signature in seen_orders:
                    break
                seen_orders.add(signature)
                progress("repair", candidate=number + 1, repair_round=round_index + 1)
                updated, detail = repair_once(plan, report, scene, backend, connector, validator,
                                              budget, round_index)
                record["repair"] = detail
                atomic_json(out / "audit.json", audit)
                if not detail["changed"]:
                    break
                plan = updated
            if best_passed and scene.settings.first_feasible:
                termination = "FIRST_FEASIBLE"
                break
        except TimeoutError as exc:
            termination = "TIME_BUDGET"
            audit.append({"code": termination, "message": str(exc)})
            break
        except (ValueError, RuntimeError, IndexError) as exc:
            audit.append({"candidate": number + 1, "code": "CANDIDATE_ERROR", "message": str(exc)})
            progress("candidate_error", message=str(exc))
    atomic_json(out / "audit.json", audit)
    summary = {"scene": scene.name, "f2c_version": backend.version, "termination": termination,
               "elapsed_s": budget.elapsed, "connection_statistics": connector.stats,
               "work_crs": scene.crs, "local_origin_m": scene.origin,
               "vehicle": asdict(scene.vehicle), "planning": asdict(scene.settings),
               "validation_scope": "Sampled rigid-body vehicle and fixed-implement model; not real-machine certification"}
    if best_passed:
        plan, report, efficiency = best_passed
        files = export_result(out, scene, plan, report, accepted=True)
        atomic_json(out / "efficiency.json", efficiency)
        summary.update(status="PASSED", validation=report.summary(), efficiency=efficiency, files=files)
        code = 0
    elif best_failed:
        _, plan, report = best_failed
        # 已有运动失败时，不花额外预算计算可能误导的完整覆盖率。
        files = export_result(out, scene, plan, report, accepted=False)
        summary.update(status="NO_VALID_PLAN_FOUND", validation=report.summary(), efficiency=None, files=files,
                       message="预算内未找到合格方案，不等于物理不可行；candidate 文件仅供诊断")
        code = 2
    else:
        summary.update(status="NO_CANDIDATE", efficiency=None, message="没有生成可供验收的候选，见 audit.json")
        code = 2
    atomic_json(out / "summary.json", summary)
    progress("finished", status=summary["status"])
    return code


def run_self_tests(out: Path) -> int:
    """不依赖 F2C 的单元自检；合成路线绝不冒充 F2C 的端到端运行结果。"""
    results = []
    def check(name, function):
        try:
            function()
            results.append({"test": name, "passed": True})
        except Exception as exc:
            results.append({"test": name, "passed": False, "error": str(exc)})
    def require(value, message="断言失败"):
        if not value:
            raise AssertionError(message)
    v = Vehicle(implement_offset_m=0.0, safety_margin_m=0.0)
    settings = Settings(check_curvature_rate=True, coverage_tolerance_m2=0.01)
    scene = Scene(box(0, -3, 100, 3), box(-20, -20, 120, 20), v, settings)
    task = Task("one", Pose(0, 0, 0), Pose(100, 0, 0))
    good = Plan([task], [work_motion(task, scene)])
    validator = Validator(scene)
    report = validator.validate(good)
    check("straight_full_coverage", lambda: require(report.passed and report.missing_area_m2 < 1e-8))
    check("positive_reference_efficiency", lambda: require(0 < calculate_efficiency(good, scene, report)["field_efficiency_pct"] <= 100))
    check("working_distance_exact", lambda: require(abs(calculate_efficiency(good, scene, report)["work_m"] - 100) < 1e-8))
    check("implement_switch_time_counted", lambda: require(calculate_efficiency(good, scene, report)["estimated_stop_time_s"] == 2))
    short = Task("short", Pose(0, 0, 0), Pose(40, 0, 0))
    short_plan = Plan([short], [work_motion(short, scene)])
    short_report = validator.validate(short_plan)
    check("uncovered_is_failure", lambda: require(not short_report.passed and short_report.missing_area_m2 > 300))
    def no_efficiency():
        try:
            calculate_efficiency(short_plan, scene, short_report)
        except ValueError:
            return
        raise AssertionError("失败路线竟然输出了效率")
    check("invalid_plan_cannot_get_efficiency", no_efficiency)
    crossing_scene = Scene(scene.target, scene.travel.difference(box(49.99, -0.1, 50.01, 0.1)), v, settings)
    check("tiny_obstacle_between_samples_detected", lambda: require(any(e["code"] == "COLLISION_OR_BOUNDARY" for e in Validator(crossing_scene).motion_issues(good.motions[0]))))
    boundary_scene = Scene(scene.target, box(0, -3, 100, 3), v, settings)
    check("vehicle_front_rear_boundary_detected", lambda: require(bool(Validator(boundary_scene).motion_issues(good.motions[0]))))
    reverse = Motion(np.array([[10, 0, 0, -1], [0, 0, 0, -1]]), "transit")
    check("reverse_forbidden_detected", lambda: require(any(e["code"] == "REVERSE_FORBIDDEN" for e in validator.motion_issues(reverse))))
    jump = Motion(np.array([[0, 0, 0, 1], [0, 5, 0, 1]]), "transit")
    check("lateral_motion_detected", lambda: require(any(e["code"] == "LATERAL_JUMP" for e in validator.motion_issues(jump))))
    rotate = Motion(np.array([[0, 0, 0, 1], [0, 0, math.pi / 2, 1]]), "transit")
    check("stationary_heading_jump_detected", lambda: require(any(e["code"] == "HEADING_JUMP" for e in validator.motion_issues(rotate))))
    displaced = Task("two", Pose(80, 0, 0), Pose(100, 0, 0))
    broken = Plan([short, displaced], [work_motion(short, scene), work_motion(displaced, scene)])
    check("inter_segment_gap_detected", lambda: require(any(e["code"] == "DISCONNECTED" for e in validator.validate(broken).issues)))
    tool_off = Motion(good.motions[0].points, "transit", implement_on=False)
    check("transit_is_not_coverage", lambda: require(validator.validate(Plan([], [tool_off])).coverage_fraction == 0))
    check("missing_task_detected", lambda: require(any(e["code"] == "MISSING_TASKS" for e in validator.validate(Plan([task, displaced], good.motions)).issues)))
    check("task_reverse_preserves_id", lambda: require(task.reverse().task_id == task.task_id and abs(abs(task.reverse().start.yaw) - math.pi) < 1e-9))
    def holes_preserved():
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "scene.json"
            geometry = Polygon([(0, 0), (100, 0), (100, 80), (0, 80)], holes=[[(20, 20), (30, 20), (30, 30), (20, 30)]])
            p.write_text(json.dumps({"crs": "LOCAL_METRIC", "target": mapping(geometry)}), encoding="utf-8")
            loaded = load_scene(p)
            require(len(loaded.target.interiors) == 1 and abs(loaded.target.area - 7900) < 1e-8)
            restored = affinity.translate(loaded.target, *loaded.origin)
            require(restored.equals(geometry))
    check("holes_and_coordinate_origin_preserved", holes_preserved)
    def rejects_degrees():
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "scene.json"
            p.write_text(json.dumps({"crs": "EPSG:4326", "target": mapping(box(110, 30, 111, 31))}), encoding="utf-8")
            try:
                load_scene(p)
            except ValueError:
                return
            raise AssertionError("接受了经纬度直接规划")
    check("geographic_coordinates_rejected", rejects_degrees)
    def timeout_works():
        try:
            subprocess.run([sys.executable, "-c", "import time; time.sleep(3)"], timeout=0.1, check=False)
        except subprocess.TimeoutExpired:
            return
        raise AssertionError("子进程未按限时结束")
    check("subprocess_hard_timeout", timeout_works)
    check("angle_wrap", lambda: require(abs(wrap(math.radians(359)) - math.radians(-1)) < 1e-9))
    reverse_scene = Scene(scene.target, scene.travel, replace(v, allow_reverse=True), settings)
    repeated_task = Task("again", task.start, task.end)
    repeated = Plan([task, repeated_task], [work_motion(task, reverse_scene), reverse,
                                          work_motion(repeated_task, reverse_scene)])
    # 使用完整的100m返程，而非上面单独测试禁止倒车的10m段。
    repeated.motions[1] = Motion(np.array([[100, 0, 0, -1], [0, 0, 0, -1]]), "transit")
    repeated_report = Validator(reverse_scene).validate(repeated)
    check("repeat_coverage_is_union_not_sum", lambda: require(repeated_report.passed and abs(repeated_report.covered.area - 600) < 1e-8))
    check("repeated_work_reduces_efficiency", lambda: require(calculate_efficiency(repeated, reverse_scene, repeated_report)["field_efficiency_pct"] < calculate_efficiency(good, scene, report)["field_efficiency_pct"]))
    backtrack = Motion(np.array([[0, 0, 0, 1], [10, 0, 0, -1], [5, 0, 0, -1]]), "transit")
    backtrack_scene = Scene(scene.target, scene.travel.difference(box(8.9, -0.1, 9.1, 0.1)), replace(v, allow_reverse=True), settings)
    check("backtracking_sweep_uses_full_excursion", lambda: require(any(e["code"] == "COLLISION_OR_BOUNDARY" for e in Validator(backtrack_scene).motion_issues(backtrack))))
    def explicit_travel_does_not_fill_holes():
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            target = Polygon([(0, 0), (100, 0), (100, 80), (0, 80)], holes=[[(20, 20), (30, 20), (30, 30), (20, 30)]])
            path.write_text(json.dumps({"crs": "LOCAL_METRIC", "target": mapping(target), "travel": mapping(box(-10, -10, 110, 90))}), encoding="utf-8")
            loaded = load_scene(path)
            from shapely.geometry import Point
            hole_point = Point(25 - loaded.origin[0], 25 - loaded.origin[1])
            require(not loaded.travel.covers(hole_point))
    check("explicit_travel_cannot_fill_target_hole", explicit_travel_does_not_fill_holes)
    # 以下桩对象只用于单元自检，不用于正常规划，也不替代原生F2C联调。
    class StraightBackendStub:
        turners = [("synthetic_straight", object())]
        def turn_path(self, start, end, turner, guide=None):
            return start, end
        def convert_path(self, native, link, kind):
            a, b = native
            return Motion(np.array([[a.x, a.y, a.yaw, 1], [b.x, b.y, b.yaw, 1]]), kind, link=link)
    def cache_reused():
        connector = Connector(scene, StraightBackendStub(), Validator(scene), Budget(5))
        first, error = connector.connect(Pose(0, 0, 0), Pose(10, 0, 0), ("a", "b"))
        second, error2 = connector.connect(Pose(0, 0, 0), Pose(10, 0, 0), ("c", "d"))
        require(first is not None and second is not None and error is None and error2 is None)
        require(connector.stats["native_turn_calls"] == 1 and connector.stats["connection_cache_hits"] == 1)
        require(second.link == ("c", "d"))
    check("unchanged_connection_is_cached", cache_reused)
    def local_insert_preserves_order():
        pass  # 已合并到本模块，直接使用下方的定义。
        a = Task("a", Pose(0, 0, 0), Pose(10, 0, 0))
        b = Task("b", Pose(20, 0, 0), Pose(30, 0, 0))
        patch = Task("patch", Pose(10, 0, 0), Pose(20, 0, 0), "patch")
        validator2 = Validator(scene)
        connector = Connector(scene, StraightBackendStub(), validator2, Budget(5))
        result = _insert_one([a, b], patch, scene, connector, validator2)
        require(result is not None and [t.task_id for t in result] == ["a", "patch", "b"])
    check("local_patch_preserves_other_tasks", local_insert_preserves_order)
    def one_round_is_not_recursive():
        a = Task("a", Pose(0, 0, 0), Pose(10, 0, 0))
        b = Task("b", Pose(20, 0, 0), Pose(30, 0, 0))
        broken = Plan([a, b], [], [{"code": "CONNECTION_FAILED", "before_task_index": 1}])
        budget2 = Budget(5)
        validator2 = Validator(scene)
        backend = StraightBackendStub()
        connector = Connector(scene, backend, validator2, budget2)
        updated, detail = repair_once(broken, short_report, scene, backend, connector, validator2, budget2, 0)
        require(detail["changed"] and updated.tasks[0] == a and updated.tasks[1] == b.reverse())
        require(len(updated.tasks) == 2 and scene.target.area == 600)
    check("repair_is_one_local_operation", one_round_is_not_recursive)
    def native_state_conversion():
        class NativePoint:
            def __init__(self, x, y): self.x, self.y = x, y
            def getX(self): return self.x
            def getY(self): return self.y
        class NativeState:
            def __init__(self, x):
                self.point, self.angle, self.dir, self.len = NativePoint(x, 0), 0, 1, 5
            def atEnd(self): return NativePoint(self.point.x + self.len, 0)
        class NativePath:
            def __init__(self, second=5): self.states = [NativeState(0), NativeState(second)]
            def size(self): return len(self.states)
            def getState(self, index): return self.states[index]
        adapter = object.__new__(F2CBackend)
        adapter.scene = scene
        motion = adapter.convert_path(NativePath(), ("a", "b"), "transit")
        require(abs(motion.length - 10) < 1e-8 and motion.points[-1, 0] == 10)
        try:
            adapter.convert_path(NativePath(6), ("a", "b"), "transit")
        except ValueError:
            return
        raise AssertionError("转换时静默连接了原生路径断点")
    check("native_state_adapter_keeps_end_and_rejects_gaps", native_state_conversion)
    files = export_result(out, scene, good, report, accepted=True)
    check("artifact_export_exists", lambda: require(all((out / value).exists() for value in files.values())))
    def invalid_input_rejected():
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            invalid = Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)])
            path.write_text(json.dumps({"crs": "LOCAL_METRIC", "target": mapping(invalid)}), encoding="utf-8")
            try:
                load_scene(path)
            except ValueError:
                return
            raise AssertionError("无效几何没有拒绝")
    check("self_intersection_rejected_not_silently_repaired", invalid_input_rejected)
    def accepted_checkpoint_not_overwritten():
        export_result(out, scene, short_plan, short_report, accepted=False)
        accepted = json.loads((out / "validation.json").read_text(encoding="utf-8"))
        rejected = json.loads((out / "candidate_validation.json").read_text(encoding="utf-8"))
        require(accepted["passed"] and not rejected["passed"])
    check("accepted_checkpoint_survives_failed_candidate", accepted_checkpoint_not_overwritten)
    theta = np.linspace(0, math.pi / 2, 40)
    tight_turn = Motion(np.column_stack([np.cos(theta), np.sin(theta), theta + math.pi / 2, np.ones(40)]), "turn")
    check("too_small_turn_radius_rejected", lambda: require(any(e["code"] == "CURVATURE_LIMIT" for e in validator.motion_issues(tight_turn))))
    def geographic_export_transforms_units():
        from shapely.geometry import Point
        projected = Scene(scene.target, scene.travel, v, settings, crs="EPSG:32650", origin=(500000, 4000000))
        point = output_geometry(Point(0, 0), projected)
        require(100 < point.x < 130 and 20 < point.y < 50)
    check("projected_export_is_real_longitude_latitude", geographic_export_transforms_units)
    data = {"passed": sum(r["passed"] for r in results), "total": len(results), "tests": results,
            "f2c_native_integration_tested": False,
            "scope": "Synthetic unit checks for scene, validation, metrics, exports and subprocess timeout only"}
    atomic_json(out / "self_test_results.json", data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


def parser() -> argparse.ArgumentParser:
    """声明正式命令行参数与阶段选择；具体组合限制由main检查，默认值可来自统一配置。"""
    p = argparse.ArgumentParser(description="F2C 2.1 覆盖规划、有限修补与规划参考效率；离线研究工具")
    p.add_argument("--input", help="原始多田块GPKG；default使用config.json中input.default_gpkg")
    p.add_argument("--stage", choices=("full", "swaths", "routes", "efficiency"), default="full",
                   help="完整规划、主体条带，或消费冻结条带结果生成区内连接路线")
    p.add_argument("--workers", type=int, default=None,
                   help="推荐参考/多田条带默认上限12，按启动时CPU空闲量和内存预算缩减；0自动，正整数可超过12；历史阶段按指定数")
    p.add_argument("--worker-memory-mib", type=float, default=None, help="推荐参考/多田条带每个子进程的预估内存 MiB，非硬内存限额")
    p.add_argument("--memory-budget-mib", type=float, default=None, help="推荐参考/多田条带总内存预算 MiB；0 使用当前可用内存75%%，留512 MiB给主进程")
    p.add_argument("--swath-bundle", help="--stage routes 的冻结条带结果目录")
    p.add_argument("--field-id", help="--stage routes 时仅处理指定 field_id")
    p.add_argument("--route-config", help="兼容旧路线JSON；默认从--config的routes读取")
    p.add_argument("--route-profile", choices=("reference","geometry","operational","strict","separate_headland"), help="覆盖统一配置中routes.active_profile")
    p.add_argument("--efficiency-profile", choices=("continuous","constant_full","constant_body"), help="覆盖统一配置中efficiency.active_profile")
    p.add_argument("--check-config", action="store_true", help="校验统一配置全部模式及运行参数，不运行规划")
    p.add_argument("--route-batch", help="--stage efficiency 的参考批次目录或明确 GPKG 文件（可重算结果再重算）")
    p.add_argument("--efficiency-config", default=None, help="时间效率参数 JSON，默认完整参考路线配置")
    p.add_argument("--batch-resume", action="store_true", help="增量参考/效率批次续跑同一 --out；跳过已提交田块")
    p.add_argument("--field-timeout", type=float, default=None, help="增量参考每田硬超时秒数，含计算及子进程启动")
    p.add_argument("--retry-failed", type=int, default=None, help="每田最多追加重试次数，默认 1")
    p.add_argument("--input-chunk-size", type=int, default=None, help="每次最多读取的冻结输入田块数")
    p.add_argument("--plot-fields", action=argparse.BooleanOptionalAction, default=None, help="增量批次生成并嵌入逐田展示 PNG，默认关闭")
    p.add_argument("--stop-after", type=int, default=None, help="验证断点续算：本次提交指定田数后暂停")
    p.add_argument("--resume-route-batch", help="从已复核路线批次的末端续作，需 --stage routes --field-id --workers 1")
    p.add_argument("--recover-route-batch", help="恢复已复核失败候选：长条带起步、末端续作和末条插入；需 routes/field-id/workers 1")
    p.add_argument("--fill-route-headland-batch", help="保留已复核主体路线，仅增加独立田头补漏段；需 --stage routes")
    p.add_argument("--headland-fill-seconds", type=float, default=None, help="每田独立补漏候选预算，默认20秒")
    p.add_argument("--recover-seconds", type=float, default=None, help="有界路线恢复预算，默认30秒")
    p.add_argument("--preserve-body-batch", help="路线恢复时必须保回的原主体基准批次")
    p.add_argument("--resume-seconds", type=float, default=None,
                   help="末端续作的追加搜索预算（默认 60 秒），不含恢复、复核和绘图")
    p.add_argument("--scene", help="含 target/travel 的 JSON 场景")
    p.add_argument("--field", help="SHP/GPKG/GeoJSON 输入，一次选择一个田块")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"),
                   help="唯一项目配置，默认V7/config.json；命令行显式参数优先")
    p.add_argument("--feature-index", type=int, default=0)
    p.add_argument("--layer", default=None)
    p.add_argument("--out", default=None, help="新建的空输出目录；不覆盖已有结果")
    p.add_argument("--seconds", type=float, default=None,
                   help="覆盖完整规划运行预算；swaths 阶段当前不支持该参数并会明确拒绝")
    p.add_argument("--check-env", action="store_true")
    p.add_argument("--check-scene", action="store_true",
                   help="只调用 scene.py 加载与检查场景，不执行规划、修补或验证")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p


# 统一入口：先拒绝冲突参数，再按 stage 选择场景、条带或路线流程。
# 批次导出必须使用新目录；某条策略的几何通过不代表其他策略也已认证。
def main() -> int:
    """读取唯一配置并分发阶段；显式参数优先，拒绝冲突输入，默认reference路线采用逐田GPKG与效率同批处理。"""
    args = parser().parse_args()
    if (args.route_profile and args.stage!='routes') or (args.efficiency_profile and args.stage not in ('routes','efficiency')):
        print('配置模式只用于对应路线/效率阶段',file=sys.stderr);return 2
    try:
        configuration_path=config_reference(args.config)[0]
        raw=json.loads(configuration_path.read_text())
        cfg=project_config(configuration_path) if raw.get('_format') else project_config()
        if args.check_config:
            cfg=project_config(configuration_path)
            from calculate_work_time_efficiency import load_config
            from route import ApproxRouteSettings
            from route_planner import RouteSettings
            make_config(Vehicle,cfg['vehicle']);make_config(Settings,cfg['planning'])
            for name in cfg['routes']['profiles']:
                selected=config_section(configuration_path,'routes',name)
                if selected.get('route_strategy')=='APPROX_CONNECTED':ApproxRouteSettings(**selected)
                else:RouteSettings.from_file(Path(str(configuration_path)+'#routes:'+name))
            for name in cfg['efficiency']['profiles']:load_config(Path(str(configuration_path)+'#efficiency:'+name))
            validate_execution(cfg['execution'])
            print(json.dumps(dict(status='CONFIG_VALID',config=str(configuration_path),route_profiles=list(cfg['routes']['profiles']),efficiency_profiles=list(cfg['efficiency']['profiles'])),ensure_ascii=False,indent=2))
            return 0
        validate_execution(cfg['execution'])
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(f'配置错误：{type(exc).__name__}: {exc}',file=sys.stderr);return 2
    if args.input=='default':args.input=str(resolve_input_path(cfg['input']['default_gpkg'],configuration_path))
    if (args.input or args.field) and args.layer is None:args.layer=cfg['input']['default_layer']
    reference_workers=cfg['execution']['workers'] if args.workers is None else args.workers
    if args.workers is None:args.workers=cfg['execution']['other_stage_workers']
    for name in ('worker_memory_mib','memory_budget_mib','field_timeout','retry_failed','input_chunk_size','plot_fields'):
        if getattr(args,name) is None:setattr(args,name,cfg['execution'][name])
    if args.stage=='routes' and not (args.resume_route_batch or args.recover_route_batch or args.fill_route_headland_batch):
        args.route_config=args.route_config or args.config
        if args.route_profile:args.route_config=str(config_reference(args.route_config)[0])+'#routes:'+args.route_profile
    if args.stage=='efficiency' or (args.stage=='routes' and args.efficiency_profile):
        args.efficiency_config=args.efficiency_config or args.config
        if args.efficiency_profile:args.efficiency_config=str(config_reference(args.efficiency_config)[0])+'#efficiency:'+args.efficiency_profile

    if args.workers<0:
        print('--workers 必须是非负整数；0 用于推荐几何参考/多田条带的自动调度',file=sys.stderr)
        return 2
    if ((args.stage != 'efficiency' and args.route_batch) or
            (args.stage not in ('routes','efficiency') and args.efficiency_config)):
        print('--route-batch 只用于效率阶段；--efficiency-config 用于参考路线或效率阶段',file=sys.stderr)
        return 2
    explicit_batch_flags={'--field-timeout','--retry-failed','--input-chunk-size','--worker-memory-mib','--memory-budget-mib','--plot-fields','--no-plot-fields'}
    batch_options_used=(args.batch_resume or args.stop_after is not None or any(x.split('=')[0] in explicit_batch_flags for x in sys.argv[1:]))
    if args.stage not in ('routes','efficiency') and batch_options_used:
        print('增量批处理参数只用于 routes / efficiency',file=sys.stderr)
        return 2
    if args.stage=='efficiency' and (args.stop_after is not None or any(x.split('=')[0] in explicit_batch_flags for x in sys.argv[1:])):
        print('效率阶段只接受 --batch-resume；超时、重试、绘图及输入分片参数用于参考路线阶段',file=sys.stderr)
        return 2
    if args.stage=='routes' and (args.resume_route_batch or args.recover_route_batch or args.fill_route_headland_batch) and (batch_options_used or args.efficiency_config or args.efficiency_profile):
        print('增量批处理不能与历史恢复/末端续作入口混用',file=sys.stderr)
        return 2
    if args.headland_fill_seconds is not None and not args.fill_route_headland_batch:
        print("--headland-fill-seconds 只用于 --fill-route-headland-batch", file=sys.stderr)
        return 2
    if args.fill_route_headland_batch and (args.stage != "routes" or args.recover_route_batch or args.resume_route_batch):
        print("独立田头补漏需 --stage routes，不能与恢复或续作混用", file=sys.stderr)
        return 2
    if (args.recover_seconds is not None or args.preserve_body_batch) and not args.recover_route_batch:
        print("--recover-seconds / --preserve-body-batch 只用于 --recover-route-batch", file=sys.stderr)
        return 2
    if args.recover_route_batch and (args.stage != "routes" or args.resume_route_batch):
        print("--recover-route-batch 只用于 --stage routes，不能与 --resume-route-batch 混用", file=sys.stderr)
        return 2
    if args.resume_seconds is not None and not args.resume_route_batch:
        print("--resume-seconds 只用于 --resume-route-batch", file=sys.stderr)
        return 2
    if args.resume_route_batch and args.stage != "routes":
        print("--resume-route-batch 只用于 --stage routes", file=sys.stderr)
        return 2
    if args.stage == "swaths" and args.seconds is not None:
        print(
            "--stage swaths 不支持 --seconds：条带候选搜索不读取全规划时间预算；"
            "请移除此参数后运行，避免把未执行的预算误当成生效。",
            file=sys.stderr,
        )
        return 2
    if args.check_env:
        missing = []
        for module in ["numpy", "shapely", "pyproj", "fields2cover"]:
            available = importlib.util.find_spec(module) is not None
            print(f"{module}: {'available' if available else 'MISSING'}")
            if not available:
                missing.append(module)
        return 2 if missing else 0
    if not args.out:
        args.out = str(Path("outputs") / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"))
    out = Path(args.out).resolve()
    args.out = str(out)
    if args.stage == 'efficiency':
        if (not args.route_batch or args.input or args.scene or args.field or args.swath_bundle
                or args.route_config or args.field_id or args.seconds is not None or args.self_test
                or args.check_scene or args._worker or args.resume_route_batch or args.recover_route_batch
                or args.fill_route_headland_batch):
            print('效率阶段需要 --route-batch，不能混用规划或恢复输入',file=sys.stderr)
            return 2
        try:
            from calculate_work_time_efficiency import run_batch as calculate_time_batch
            cfg=Path(args.efficiency_config) if args.efficiency_config else Path(args.config)
            result=calculate_time_batch(Path(args.route_batch),cfg,out,resume=args.batch_resume)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result.get('status')=='COMPLETED' and result.get('efficiency_status_counts',{}).get('ESTIMATED',0)==result.get('input_field_count',result.get('field_count',0)) else 1
        except (ValueError,KeyError,OSError,TypeError) as exc:
            print(f'效率计算失败：{type(exc).__name__}: {exc}',file=sys.stderr)
            return 2
    if args.stage == "routes":
        if args.fill_route_headland_batch:
            if (args.swath_bundle or args.route_config or args.input or args.scene or args.field
                    or args.self_test or args.check_scene or args._worker or args.seconds is not None):
                print("独立田头补漏从来源批次读取参数，不能混用其他输入或 --seconds", file=sys.stderr)
                return 2
            try:
                from route_planner import fill_headland_batch as fill_headland
                return fill_headland(Path(args.fill_route_headland_batch), out, args.field_id,
                    args.workers, args.headland_fill_seconds if args.headland_fill_seconds is not None else 20.0)
            except Exception as exc:
                print(f"独立田头补漏失败：{type(exc).__name__}: {exc}", file=sys.stderr)
                return 2
        if args.recover_route_batch:
            if (not args.field_id or args.workers != 1 or args.swath_bundle or args.route_config
                    or args.input or args.scene or args.field or args.self_test or args.check_scene
                    or args._worker or args.seconds is not None):
                print("有界恢复需要 --field-id --workers 1；不能混用其他输入或 --seconds", file=sys.stderr)
                return 2
            try:
                from route_planner import recover_saved_batch as recover_route
                return recover_route(Path(args.recover_route_batch), out, args.field_id,
                    args.recover_seconds if args.recover_seconds is not None else 30.0,
                    Path(args.preserve_body_batch) if args.preserve_body_batch else None)
            except Exception as exc:
                print(f"有界路线恢复失败：{type(exc).__name__}: {exc}", file=sys.stderr)
                return 2
        if args.resume_route_batch:
            if (not args.field_id or args.workers != 1 or args.swath_bundle or args.route_config
                    or args.input or args.scene or args.field or args.self_test or args.check_scene
                    or args._worker or args.seconds is not None):
                print("末端续作需要 --field-id --workers 1；上游和参数从来源批次读取，不能混用其他输入", file=sys.stderr)
                return 2
            try:
                from route_planner import resume_saved_batch as resume_route
                return resume_route(Path(args.resume_route_batch),out,[args.field_id],
                    args.resume_seconds if args.resume_seconds is not None else 60.0)
            except Exception as exc:
                print(f"末端续作失败：{type(exc).__name__}: {exc}",file=sys.stderr)
                return 2
        if (not args.swath_bundle or args.input or args.scene or args.field
                or args.self_test or args.check_scene or args._worker
                or args.seconds is not None):
            print("--stage routes 需要 --swath-bundle，不能混用旧规划/条带输入或 --seconds",
                  file=sys.stderr)
            return 2
        try:
            # 新策略使用自己的配置契约；历史严格设置和冻结上游算法保持原义。
            if args.route_config:
                selected_config=config_section(args.route_config,"routes")
                if isinstance(selected_config,dict) and selected_config.get('route_strategy')=='APPROX_CONNECTED':
                    if not selected_config.get('motion_refinement',False) and selected_config.get('assemble_reference',False):
                        from route_planner import run_incremental_reference_batch
                        result=run_incremental_reference_batch(Path(args.swath_bundle),out,
                            workers=reference_workers,field_id=args.field_id,route_config=Path(args.route_config),
                            worker_memory_mib=args.worker_memory_mib,memory_budget_mib=args.memory_budget_mib,
                            efficiency_config=Path(args.efficiency_config or args.config),
                            resume=args.batch_resume,field_timeout=args.field_timeout,retries=args.retry_failed,
                            chunk_size=args.input_chunk_size,plot_enabled=args.plot_fields,stop_after=args.stop_after)
                        print(json.dumps(result,ensure_ascii=False,indent=2))
                        return 0 if result['reference_acceptance_passed'] and result['efficiency_status_counts'].get('ESTIMATED',0)==result['input_field_count'] else 1
                    if batch_options_used or "--efficiency-config" in sys.argv or args.efficiency_profile:
                        raise ValueError('INCREMENTAL_FLAGS_REQUIRE_GEOMETRY_REFERENCE_PROFILE')
                    from route import run_reference_batch as run_approx_batch
                    return run_approx_batch(Path(args.swath_bundle),out,workers=args.workers,
                        field_id=args.field_id,route_config=Path(args.route_config))
                if isinstance(selected_config,dict) and selected_config.get('route_strategy')=='REGULAR_HEADLAND_FIRST':
                    if batch_options_used or "--efficiency-config" in sys.argv or args.efficiency_profile:raise ValueError('INCREMENTAL_FLAGS_REQUIRE_GEOMETRY_REFERENCE_PROFILE')
                    from route import run_regular_batch
                    return run_regular_batch(Path(args.swath_bundle),out,
                        workers=args.workers,field_id=args.field_id,
                        route_config=Path(args.route_config))
            from route_planner import run_batch as run_route_batch
            if batch_options_used or "--efficiency-config" in sys.argv or args.efficiency_profile:raise ValueError('INCREMENTAL_FLAGS_REQUIRE_GEOMETRY_REFERENCE_PROFILE')
            return run_route_batch(
                Path(args.swath_bundle), out, workers=args.workers,
                field_id=args.field_id,
                route_config=Path(args.route_config) if args.route_config else None)
        except Exception as exc:
            print(f"区内路线阶段失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    if args._worker:
        try:
            return run_worker(args)
        except Exception as exc:
            traceback.print_exc()
            old = {}
            if (out / "summary.json").exists():
                old = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            if old.get("status") == "PASSED":
                old["interrupted_after_valid_checkpoint"] = str(exc)
                atomic_json(out / "summary.json", old)
                return 0
            atomic_json(out / "summary.json", {"status": "ERROR", "efficiency": None,
                                               "error_type": type(exc).__name__, "message": str(exc)})
            return 2
    if args.input:
        if raw.get('_format'):
            # 下游冻结入口要求输出目录为空；启动快照先置于outputs缓存。
            # 唯一可编辑配置仍是根config.json，派生快照只用于重现本次输入。
            import uuid
            snapshot_dir=PROJECT_CONFIG.parent/'outputs/.cache/project_configs'/uuid.uuid4().hex
            effective=scene_config(args.config)
            probe=gpd.read_file(args.input,**({'layer':args.layer} if args.layer else {}),rows=1)
            if probe.crs is not None and probe.crs.is_engineering and not effective.get('crs'):
                effective['crs']=probe.crs.to_wkt()
            atomic_json(snapshot_dir/'effective_scene_config.json',effective)
            atomic_json(snapshot_dir/'project_config_snapshot.json',raw)
            args.config=str(snapshot_dir/'effective_scene_config.json')
        if args.scene or args.field or args.self_test or args.check_scene:
            print("--input 不能与 --scene、--field、--self-test 或 --check-scene 同时使用",
                  file=sys.stderr)
            return 2
        if args.stage == "swaths":
            # 只在调度入口限流，保持冻结条带算法及单田输出合同不变。
            if out.exists() and any(out.iterdir()):
                print(f"输出目录非空，拒绝覆盖：{out}",file=sys.stderr)
                return 2
            import pyogrio
            from route_planner import reference_worker_plan
            from swath_batch import run_batch as run_swath_batch
            try:
                info=pyogrio.read_info(args.input,layer=args.layer,force_feature_count=True)
                plan=reference_worker_plan(args.workers,field_count=int(info['features']),
                    worker_memory_mib=args.worker_memory_mib,memory_budget_mib=args.memory_budget_mib)
                if plan['effective_workers']<1:raise ValueError('EMPTY_INPUT')
            except (ValueError,OSError) as exc:
                print(f'条带调度资源检查失败: {exc}',file=sys.stderr)
                return 2
            print(f'条带并行调度: 请求 {args.workers} (0=自动), CPU上限 {plan["usable_cpus"]}, '
                  f'空闲估计 {plan.get("cpu_idle_slots")}, 内存槽位 {plan["memory_slots"]}, '
                  f'实际进程 {plan["effective_workers"]}',flush=True)
            result=run_swath_batch(Path(args.input),Path(args.config),out,
                                   layer=args.layer,workers=plan['effective_workers'])
            if out.is_dir():atomic_json(out/'worker_plan.json',plan)
            return result
        # 延迟导入：批处理会通过 --scene 回调本文件；run_gpkg 不再反向导入 main。
        run_batch = run_gpkg_batch
        return run_batch(Path(args.input), Path(args.config), out, layer=args.layer,
                         seconds=args.seconds, planner_entry=Path(__file__).resolve())
    if out.exists() and any(out.iterdir()):
        print(f"输出目录非空，拒绝覆盖：{out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        return run_self_tests(out)
    if not args.scene and not args.field:
        print("请提供 --scene 或 --field；也可以先运行 --self-test", file=sys.stderr)
        return 2
    if args.check_scene:
        return run_scene_check(args)
    if args.stage == "swaths":
        if args.self_test or args.check_env or args._worker or not (args.scene or args.field):
            print("--stage swaths 需要 --scene 或 --field，且不能与内部 worker/自检选项组合",
                  file=sys.stderr)
            return 2
        try:
            from swath_batch import run_single as run_single_swaths
            scene = _scene_from_args(args)
            return run_single_swaths(
                scene, out, source_crs=scene.crs,
                field_path=Path(args.field) if args.field else None,
                layer=args.layer, feature_index=args.feature_index,
            )
        except Exception as exc:
            atomic_json(out / "swath_batch_summary.json", {
                "status": "SWATHS_WORKER_ERROR", "error_type": type(exc).__name__,
                "message": str(exc), "traceback": traceback.format_exc(limit=8),
            })
            print(f"条带阶段失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    try:
        # 父进程只读取轻量配置，不重复读取整个 GIS 图层；场景读取也在受限工作进程内。
        configuration = json.loads(Path(args.scene or args.config).read_text(encoding="utf-8"))
        limits = make_config(Settings, configuration.get("planning", {}))
        if args.seconds is not None:
            limits = replace(limits, wall_time_seconds=args.seconds)
    except Exception as exc:
        atomic_json(out / "summary.json", {"status": "INPUT_ERROR", "efficiency": None, "message": str(exc)})
        print(str(exc), file=sys.stderr)
        return 2
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker", "--out", str(out),
               "--config", str(Path(args.config).resolve()), "--feature-index", str(args.feature_index)]
    for name in ("scene", "field", "layer", "seconds"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name, str(Path(value).resolve()) if name in {"scene", "field"} else str(value)]
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    code, interrupted = 2, None
    with (out / "run.log").open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       timeout=limits.wall_time_seconds, check=False)
            code = completed.returncode
            if code < 0:
                interrupted = f"原生工作进程被信号 {-code} 终止"
        except subprocess.TimeoutExpired:
            code, interrupted = 124, "达到总时间预算，已终止工作进程；未声称物理不可行"
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if interrupted or not summary:
        if summary.get("status") == "PASSED":
            summary["interrupted_after_valid_checkpoint"] = interrupted
            code = 0
        else:
            summary.update(status="TIMEOUT" if code == 124 else "WORKER_FAILURE",
                           efficiency=None, message=interrupted or "工作进程没有输出总结，查看 run.log")
        atomic_json(summary_path, summary)
    print(json.dumps({"status": summary.get("status"), "output_directory": str(out),
                      "message": summary.get("message", "见 summary.json、audit.json 与 run.log")}, ensure_ascii=False, indent=2))
    return code if code >= 0 else 2

# ==========================================================================
# 2. 原始 GPKG 准备与逐田调度
# 输入多田块文件与配置，输出逐田结果和原完整流程汇总；不在这里计算几何分区。
# ==========================================================================

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import geopandas as gpd
from shapely.geometry import shape

from io_utils import atomic_json as run_gpkg_atomic_json

run_gpkg_PROJECT = Path(__file__).resolve().parents[1]


def export_gpkg(out: Path, manifest: dict, rows: list[dict]) -> dict:
    """汇总历史full批次为results.gpkg，字段v6_status属于历史协议，不等于当前参考路线状态。"""
    path = out / "results.gpkg"
    fields = gpd.read_file(out / "prepared/input_fields.gpkg", layer="input_fields")
    fields["v6_status"] = [r["status"] for r in rows]
    fields["v6_return_code"] = [r["return_code"] for r in rows]
    fields.to_file(path, layer="input_fields", driver="GPKG", index=False)
    records = {"routes": [], "coverage": []}
    for field, result in zip(manifest["fields"], rows):
        folder = out / field["directory"]
        prefix = "" if result["status"] == "PASSED" else "candidate_"
        for kind, filename in (("routes", f"{prefix}route.geojson"), ("coverage", f"{prefix}coverage.geojson")):
            artifact = folder / filename
            if not artifact.exists():
                continue
            for feature in json.loads(artifact.read_text(encoding="utf-8"))["features"]:
                geometry = shape(feature["geometry"])
                if geometry.is_empty:
                    continue
                properties = dict(feature["properties"])
                if isinstance(properties.get("link"), list):
                    properties["link"] = json.dumps(properties["link"])
                properties.update(field_id=field["field_id"], status=result["status"],
                                  accepted=result["status"] == "PASSED", geometry=geometry)
                records[kind].append(properties)
    for layer, features in records.items():
        if features:
            gpd.GeoDataFrame(features, crs="EPSG:4326").to_crs(manifest["source_crs"]).to_file(
                path, layer=layer, driver="GPKG", index=False)
    with sqlite3.connect(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    counts = {row["name"]: len(gpd.read_file(path, layer=row["name"])) for _, row in gpd.list_layers(path).iterrows()}
    if integrity != "ok" or counts["input_fields"] != len(rows):
        raise RuntimeError("GPKG 回读验证失败")
    return {"file": path.name, "integrity_check": integrity, "layer_counts": counts,
            "crs": manifest["source_crs"]}


# 多田块原始输入入口：调用准备工具，再以 --scene 逐田启动总入口。
# 这里只有数据准备和调度；真实规划仍由单田工作器完成。
def run_gpkg_batch(input_path: Path, config_path: Path, out: Path, *,
              layer: str | None = None, seconds: float | None = None,
              planner_entry: Path | None = None) -> int:
    """从原始多田块 GPKG 开始运行完整批处理。"""
    input_path = input_path.resolve()
    config_path = config_path.resolve()
    out = out.resolve()
    planner_entry = (planner_entry or run_gpkg_PROJECT / "src/main.py").resolve()
    if out.exists() and any(out.iterdir()):
        print(f"输出目录非空，拒绝覆盖：{out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    prepare = [sys.executable, str(run_gpkg_PROJECT / "tempscript/prepare_gpkg.py"),
               "--input", str(input_path), "--config", str(config_path),
               "--out", str(out / "prepared")]
    if layer:
        prepare += ["--layer", layer]
    subprocess.run(prepare, check=True)
    manifest = json.loads((out / "prepared/manifest.json").read_text(encoding="utf-8"))
    results = []
    for field in manifest["fields"]:
        folder = out / field["directory"]
        command = [sys.executable, str(planner_entry), "--scene",
                   str(out / "prepared" / field["scene"]), "--out", str(folder)]
        if seconds is not None:
            command += ["--seconds", str(seconds)]
        started = time.monotonic()
        print(f"Running {field['field_id']} ({field['work_crs']})", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        folder.mkdir(exist_ok=True)
        (folder / "launcher.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        validation = summary.get("validation", {})
        # 硬超时后主入口可能只留下候选检查点；明确采用该检查点，而不伪造最终验收。
        checkpoint = folder / "candidate_validation.json"
        validation_source = "summary"
        if not validation and checkpoint.exists():
            validation = json.loads(checkpoint.read_text(encoding="utf-8"))
            validation_source = "candidate_checkpoint"
        evaluated = validation.get("coverage_evaluated", False)
        record = {"field_id": field["field_id"], "work_crs": field["work_crs"],
                  "target_area_m2": field["target_area_m2"], "status": summary.get("status", "WORKER_FAILURE"),
                  "return_code": completed.returncode, "elapsed_s": time.monotonic() - started,
                  "motion_valid": validation.get("motion_valid"), "coverage_evaluated": evaluated,
                  "coverage_fraction": validation.get("coverage_fraction") if evaluated else None,
                  "missing_area_m2": validation.get("missing_area_m2") if evaluated else None,
                  "issue_codes": sorted({i["code"] for i in validation.get("issues", [])}),
                  "validation_source": validation_source, "directory": field["directory"]}
        results.append(record)
        run_gpkg_atomic_json(out / "batch_progress.json", {"completed": len(results), "total": len(manifest["fields"]), "fields": results})
        print(json.dumps(record, ensure_ascii=False), flush=True)
    gpkg = export_gpkg(out, manifest, results)
    counts = dict(Counter(row["status"] for row in results))
    run_gpkg_atomic_json(out / "batch_summary.json", {"input": manifest["source"], "config": manifest["config"],
                                            "status_counts": counts, "fields": results, "gpkg": gpkg})
    with (out / "batch_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"status_counts": counts, "gpkg": gpkg}, ensure_ascii=False), flush=True)
    return 0 if all(row["status"] == "PASSED" and row["return_code"] == 0 for row in results) else 2


def gpkg_cli_parser() -> argparse.ArgumentParser:
    """声明原始GPKG历史兼容入口；新用户通过main的--input使用同一正式实现。"""
    result = argparse.ArgumentParser(
        description="兼容入口；推荐通过 main.py --input 运行原始多田块 GPKG"
    )
    result.add_argument("--input", type=Path,
                        default=run_gpkg_PROJECT.parent.parent / "data/fields2cover_regular_size_5samples.gpkg")
    result.add_argument("--config", type=Path, default=run_gpkg_PROJECT / "config.json")
    result.add_argument("--layer")
    result.add_argument("--seconds", type=float, help="每田预算；省略则保持 config.json")
    result.add_argument("--out", type=Path, required=True)
    return result


def gpkg_cli_main() -> int:
    """将历史GPKG参数交给原始批处理，不另建一套规划算法。"""
    args = gpkg_cli_parser().parse_args()
    return run_gpkg_batch(args.input, args.config, args.out, layer=args.layer, seconds=args.seconds)

# ==========================================================================
# 3. 原完整规划的一轮局部修补
# 只提出有限修补候选；是否继续修补、是否验收由上层预算和独立验证决定。
# ==========================================================================

import math
from typing import Any

from scene import Scene as repair_Scene
from scene import Plan as repair_Plan
from scene import Task as repair_Task
from scene import Budget as repair_Budget
from planner import F2CBackend as repair_F2CBackend
from planner import Connector as repair_Connector
from planner import assemble as repair_assemble
from planner import generate_patch_tasks as repair_generate_patch_tasks
from planner import work_motion as repair_work_motion
from validator import Validation as repair_Validation
from validator import Validator as repair_Validator


def _distance(a, b) -> float:
    """计算两个位姿的平面距离，单位米；缺少任一位姿时返回0供调用者的可选端点处理。"""
    return 0.0 if a is None or b is None else math.hypot(a.x - b.x, a.y - b.y)


def _insert_one(tasks: list[repair_Task], task: repair_Task, scene: repair_Scene,
                connector: repair_Connector, validator: repair_Validator) -> list[repair_Task] | None:
    """只尝试距离较近的三个插入位置及两个作业方向，不重排其他任务。"""
    candidates = []
    for i in range(len(tasks) + 1):
        left = scene.start if i == 0 else tasks[i - 1].end
        right = scene.end if i == len(tasks) else tasks[i].start
        estimate = _distance(left, task.start) + _distance(task.end, right) - _distance(left, right)
        candidates.append((estimate, i))
    for _, i in sorted(candidates)[:3]:
        connector.budget.check()
        for oriented in (task, task.reverse()):
            if validator.motion_issues(repair_work_motion(oriented, scene)):
                continue
            left = scene.start if i == 0 else tasks[i - 1].end
            right = scene.end if i == len(tasks) else tasks[i].start
            left_id = "START" if i == 0 else tasks[i - 1].task_id
            right_id = "END" if i == len(tasks) else tasks[i].task_id
            if left is not None:
                _, error = connector.connect(left, oriented.start, (left_id, oriented.task_id))
                if error:
                    continue
            if right is not None:
                _, error = connector.connect(oriented.end, right, (oriented.task_id, right_id))
                if error:
                    continue
            return tasks[:i] + [oriented] + tasks[i:]
    return None


# 修补只做一轮，不在函数内部无限递归搜索。
# 缺口候选仍必须由独立验证器核验；修补预算由总调度控制。
def repair_once(plan: repair_Plan, report: repair_Validation, scene: repair_Scene, backend: repair_F2CBackend,
                connector: repair_Connector, validator: repair_Validator, budget: repair_Budget,
                round_index: int) -> tuple[repair_Plan, dict[str, Any]]:
    """在剩余预算中进行一轮历史full候选修补，保留检查结果；不会自动解冻已发布条带。"""
    budget.check()
    tasks = list(plan.tasks)
    detail: dict[str, Any] = {"round": round_index + 1, "changed": False}
    failures = [e for e in plan.errors if e.get("code") == "CONNECTION_FAILED"]
    if failures and tasks:
        i = min(int(failures[0]["before_task_index"]), len(tasks) - 1)
        # 一轮一种确定性修复。方向调整不是倒车作业；每次都重新验证机具位置。
        if round_index % 2 == 0 or i + 1 >= len(tasks):
            tasks[i] = tasks[i].reverse()
            detail.update(action="reverse_task_entry", task_id=tasks[i].task_id)
        else:
            tasks[i], tasks[i + 1] = tasks[i + 1], tasks[i]
            detail.update(action="swap_adjacent_tasks", index=i)
        detail["changed"] = True
        return repair_assemble(tasks, scene, connector, plan.metadata), detail
    if not report.motion_valid:
        # 处理工作段本身越界时，只尝试调整这一个任务的进入方向。
        # 不放大 travel、不裁剪目标，也不把所有工作段全局重生成。
        task_id = next((e.get("task_id") for e in report.issues if e.get("task_id")), None)
        if task_id:
            for i, task in enumerate(tasks):
                if task.task_id == task_id:
                    alternative = task.reverse()
                    if not validator.motion_issues(repair_work_motion(alternative, scene)):
                        tasks[i] = alternative
                        detail.update(changed=True, action="reverse_invalid_work", task_id=task_id)
                        return repair_assemble(tasks, scene, connector, plan.metadata), detail
        detail.update(action="stop_this_candidate", reason="不是此有限局部修复器能解决的运动错误")
        return plan, detail
    if report.missing_area_m2 > scene.settings.coverage_tolerance_m2:
        patches = repair_generate_patch_tasks(scene, backend, report.missing, tasks, budget,
                                       prefix=f"patch_round{round_index + 1}")
        inserted = []
        for patch in patches:
            budget.check()
            if len(tasks) >= scene.settings.max_tasks:
                break
            result = _insert_one(tasks, patch, scene, connector, validator)
            if result is not None:
                tasks = result
                inserted.append(patch.task_id)
        detail.update(changed=bool(inserted), action="insert_missing_coverage_tasks",
                      proposed=len(patches), inserted=inserted)
        if inserted:
            return repair_assemble(tasks, scene, connector, plan.metadata), detail
        detail["reason"] = "当前补作候选或连接不可行；未删除漏作区域"
    return plan, detail

# ==========================================================================
# 4. 已验收路线的规划参考效率
# 根据实际行程累加距离与时间；这些是规划估计，不是真实农机作业日志。
# ==========================================================================

import math
import numpy as np

from scene import Plan as efficiency_Plan
from scene import Scene as efficiency_Scene
from validator import Validation as efficiency_Validation


# 只对已验收行程计算距离、时间和规划参考作业能力。
# 倒车、调头和停顿按现有参数累加，不把这些估算解释成实车测量。
def calculate_efficiency(plan: efficiency_Plan, scene: efficiency_Scene, report: efficiency_Validation) -> dict:
    """历史full专用时间统计，先要求运动和覆盖完整验收；与正式参考路线效率模块的输入/认证范围不同。"""
    if not report.passed or not report.motion_valid or not report.coverage_evaluated:
        raise ValueError("路线未通过完整验收，禁止计算正式规划参考效率")
    v = scene.vehicle
    distances = {"work_m": 0.0, "turn_m": 0.0, "transit_m": 0.0, "reverse_m": 0.0}
    times = {"work_s": 0.0, "turn_s": 0.0, "transit_s": 0.0}
    gear_changes = 0
    switches = 0
    turn_connections = 0
    last_gear = None
    implement_was_on = False
    for motion in plan.motions:
        if motion.implement_on != implement_was_on:
            switches += 1
            implement_was_on = motion.implement_on
        p = motion.points
        ds = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
        da = np.abs((np.diff(p[:, 2]) + math.pi) % (2 * math.pi) - math.pi)
        has_turn = False
        for i, length in enumerate(ds):
            if length < 1e-9:
                continue
            gear = int(p[i, 3])
            if last_gear is not None and last_gear != gear:
                gear_changes += 1
            last_gear = gear
            if motion.implement_on:
                kind, speed = "work", v.work_speed_mps
            elif da[i] > 1e-6:
                kind, speed = "turn", v.turn_speed_mps
                has_turn = True
            else:
                kind, speed = "transit", v.transit_speed_mps
            if gear < 0:
                speed = min(speed, v.reverse_speed_mps)
                distances["reverse_m"] += float(length)
            distances[kind + "_m"] += float(length)
            times[kind + "_s"] += float(length / speed)
        turn_connections += int(has_turn)
    if implement_was_on:
        switches += 1  # 到终点关闭机具，也按统一规则计入
    stop_time = gear_changes * v.gear_change_seconds + switches * v.implement_switch_seconds
    total_time = sum(times.values()) + stop_time
    if total_time <= 0:
        raise ValueError("总时间非正，无法计算效率")
    area = float(scene.target.area)
    theoretical = v.working_width_m * v.work_speed_mps * 0.36  # ha/h
    effective = area / 10000 / (total_time / 3600)
    field_efficiency = effective / theoretical * 100
    if field_efficiency > 100.0 + 1e-4:
        raise ValueError("效率大于100%，请核对覆盖起止、有限机具长度与时间模型；未强行截断为100%")
    total_distance = distances["work_m"] + distances["turn_m"] + distances["transit_m"]
    return {"metric_name": "planning_reference_field_efficiency",
            "field_efficiency_pct": field_efficiency,
            "effective_capacity_ha_per_h": effective,
            "theoretical_capacity_ha_per_h": theoretical,
            "target_area_m2": area, "estimated_total_time_s": total_time,
            "estimated_stop_time_s": stop_time, "total_distance_m": total_distance,
            "working_distance_ratio_pct": 100 * distances["work_m"] / total_distance,
            "turn_connection_count": turn_connections, "gear_change_count": gear_changes,
            "implement_switch_count": switches, **distances, **times,
            "definition": "100 * target_area_m2 / (working_width_m * work_speed_mps * estimated_total_time_s)",
            "scope": "Rigid vehicle + fixed implement, static scene, configured speeds and stops; not measured field efficiency",
            "start_end_mode": "specified" if scene.start and scene.end else "one_or_both_endpoints_free",
            "assumptions": ["No slope, soil, wheel slip, acceleration or loading delays are modelled",
                            "All work is forward; reverse is only a transit/turn option when enabled",
                            "Turning speed is assigned to sampled intervals with nonzero heading change",
                            "Distances are computed from the sampled trajectory, not inferred from task count"]}

if __name__ == "__main__":
    raise SystemExit(main())
