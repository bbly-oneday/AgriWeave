"""Independently recompute required-main minus work-sweep coverage from GPKG."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import geopandas as gpd
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union


def _union(frame: gpd.GeoDataFrame):
    geometries = [geometry for geometry in frame.geometry
                  if geometry is not None and not geometry.is_empty]
    return unary_union(geometries) if geometries else GeometryCollection()


def audit(gpkg: Path, summary_path: Path) -> dict:
    metric_path = gpkg.with_name("swath_results_metric.gpkg")
    if not metric_path.exists():
        raise FileNotFoundError(f"缺少米制独立复核文件：{metric_path}")
    with sqlite3.connect(gpkg) as connection:
        output_integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    with sqlite3.connect(metric_path) as connection:
        metric_integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if output_integrity != "ok" or metric_integrity != "ok":
        raise AssertionError("GeoPackage PRAGMA integrity_check 未通过")

    main_layers = set(gpd.list_layers(gpkg)["name"])
    if "required_main_areas" not in main_layers:
        raise AssertionError("swath_results.gpkg 缺少 required_main_areas 图层")
    main_required = gpd.read_file(gpkg, layer="required_main_areas")
    if not main_required.empty and not main_required.geometry.is_valid.all():
        raise AssertionError("required_main_areas 存在无效几何")

    layers = gpd.list_layers(metric_path)["name"].tolist()
    required_layers = [name for name in layers if name.startswith("required_main_areas_")]
    sweep_layers = [name for name in layers if name.startswith("work_sweeps_")]
    required_by_field: dict[str, list] = {}
    sweeps_by_field: dict[str, list] = {}
    for layer in required_layers:
        frame = gpd.read_file(metric_path, layer=layer)
        for field_id, group in frame.groupby("field_id"):
            required_by_field.setdefault(str(field_id), []).extend(
                geometry for geometry in group.geometry
                if geometry is not None and not geometry.is_empty
            )
            if not group.geometry.is_valid.all():
                raise AssertionError(f"{field_id} 的米制 required 几何无效")
    for layer in sweep_layers:
        frame = gpd.read_file(metric_path, layer=layer)
        for field_id, group in frame.groupby("field_id"):
            sweeps_by_field.setdefault(str(field_id), []).extend(
                geometry for geometry in group.geometry
                if geometry is not None and not geometry.is_empty
            )
            if not group.geometry.is_valid.all():
                raise AssertionError(f"{field_id} 的米制 sweep 几何无效")

    fields = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = []
    failures = []
    for field in fields:
        field_id = str(field["field_id"])
        required_parts = required_by_field.get(field_id, [])
        sweep_parts = sweeps_by_field.get(field_id, [])
        required = unary_union(required_parts) if required_parts else GeometryCollection()
        swept = unary_union(sweep_parts) if sweep_parts else GeometryCollection()
        residual = required.difference(swept)
        residual_area = float(residual.area)
        recorded = field.get("whole_field_uncovered_area_m2")
        tolerance = float(field.get("whole_field_coverage_tolerance_m2") or 0.01)
        area_difference = (
            abs(residual_area - float(recorded)) if recorded is not None else None
        )
        passed = (
            recorded is not None and area_difference <= 1e-7
            and (not bool(field.get("acceptance_passed")) or residual_area <= tolerance)
        )
        record = {
            "field_id": field_id,
            "required_main_area_m2": float(required.area),
            "swept_required_area_m2": float(required.intersection(swept).area),
            "recomputed_uncovered_area_m2": residual_area,
            "reported_uncovered_area_m2": recorded,
            "uncovered_area_difference_m2": area_difference,
            "tolerance_m2": tolerance,
            "acceptance_passed": bool(field.get("acceptance_passed")),
            "passed": bool(passed),
        }
        checks.append(record)
        if not passed:
            failures.append(record)
    return {
        "passed": not failures,
        "field_count": len(checks),
        "failed_field_count": len(failures),
        "output_gpkg_integrity": output_integrity,
        "metric_gpkg_integrity": metric_integrity,
        "required_main_feature_count": len(main_required),
        "checks": checks,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpkg", required=True, type=Path)
    parser.add_argument("--field-results", required=True, type=Path,
                        help="field_results.json from the same run")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.gpkg, args.field_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "field_count": result["field_count"],
        "failed_field_count": result["failed_field_count"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
