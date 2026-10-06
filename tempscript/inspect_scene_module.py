"""只运行并导出 V6 scene.py 的中间结果，供人工审核几何解析。

本脚本不调用 planner.py、repair.py、validator.py 或 efficiency.py。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from shapely import affinity
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import nearest_points

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene import load_scene, polygons  # noqa: E402
from prepare_gpkg import prepare  # noqa: E402


def _hole_polygons(geometry):
    return [Polygon(ring) for poly in polygons(geometry) for ring in poly.interiors]


def _project_one(geometry, source_crs, target_crs):
    return gpd.GeoSeries([geometry], crs=source_crs).to_crs(target_crs).iloc[0]


def _draw_polygonal(ax, geometry, facecolor, edgecolor, alpha=0.75,
                    linewidth=1.2, linestyle="-"):
    """Draw polygonal geometry while keeping interior holes visible."""
    for poly in polygons(geometry):
        x, y = poly.exterior.xy
        ax.fill(x, y, facecolor=facecolor, alpha=alpha, zorder=1)
        ax.plot(x, y, color=edgecolor, linewidth=linewidth,
                linestyle=linestyle, zorder=3)
        for ring in poly.interiors:
            hx, hy = ring.xy
            ax.fill(hx, hy, facecolor="white", alpha=1.0, zorder=2)
            ax.plot(hx, hy, color="#b91c1c", linewidth=1.0, zorder=4)


def _format_comparison_axis(ax, bounds):
    minx, miny, maxx, maxy = bounds
    span = max(maxx - minx, maxy - miny, 1.0)
    margin = span * 0.06
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("local x (m)", fontsize=8)
    ax.set_ylabel("local y (m)", fontsize=8)


def _longest_segment_midpoint(ring) -> Point:
    coordinates = list(ring.coords)
    a, b = max(
        zip(coordinates[:-1], coordinates[1:]),
        key=lambda pair: (pair[1][0] - pair[0][0]) ** 2 + (pair[1][1] - pair[0][1]) ** 2,
    )
    return Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _draw_clearance_marker(ax, boundary_point: Point, travel):
    travel_point = nearest_points(boundary_point, travel.boundary)[1]
    ax.plot(
        [boundary_point.x, travel_point.x], [boundary_point.y, travel_point.y],
        color="#111827", linewidth=2.2, marker="o", markersize=4, zorder=10,
    )
    midpoint = ((boundary_point.x + travel_point.x) / 2,
                (boundary_point.y + travel_point.y) / 2)
    ax.annotate(
        f"{boundary_point.distance(travel_point):.3f} m",
        midpoint, xytext=(8, 8), textcoords="offset points",
        fontsize=10, weight="bold", color="#111827",
    )


def inspect(input_path: Path, config_path: Path, out: Path, layer: str) -> dict:
    input_path = input_path.resolve()
    config_path = config_path.resolve()
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录非空，拒绝覆盖：{out}")
    out.mkdir(parents=True, exist_ok=True)

    prepared_dir = out / "prepared"
    manifest = prepare(input_path, config_path, prepared_dir, layer)
    source = gpd.read_file(input_path, layer=layer)
    layer_names = set(gpd.list_layers(input_path)["name"])
    obstacles = (
        gpd.read_file(input_path, layer="obstacles")
        if "obstacles" in layer_names else None
    )

    summaries = []
    target_rows = []
    travel_rows = []
    hole_rows = []
    detail_candidate = None
    output_dir = out / "loaded_scenes"
    output_dir.mkdir()

    field_count = len(manifest["fields"])
    visual_count = min(field_count, 10)
    visual_indices = (
        list(range(field_count))
        if field_count <= 10
        else sorted({round(i * (field_count - 1) / 9) for i in range(10)})
    )
    visual_positions = {field_index: position
                        for position, field_index in enumerate(visual_indices)}

    fig, axes = plt.subplots(2, 5, figsize=(20, 9), constrained_layout=True)
    comparison_fig, comparison_axes = plt.subplots(
        visual_count, 3,
        figsize=(15, max(4 * visual_count, 8)),
        constrained_layout=True,
        squeeze=False,
    )
    for field_index, entry in enumerate(manifest["fields"]):
        scene_path = prepared_dir / entry["scene"]
        scene = load_scene(scene_path)
        source_row = source.iloc[entry["feature_index"]]
        source_geometry = source_row.geometry
        projected_source = _project_one(source_geometry, source.crs, scene.crs)

        restored_target = affinity.translate(scene.target, *scene.origin)
        restored_travel = affinity.translate(scene.travel, *scene.origin)
        target_source_crs = _project_one(restored_target, scene.crs, source.crs)
        travel_source_crs = _project_one(restored_travel, scene.crs, source.crs)
        holes_centered = _hole_polygons(scene.target)
        holes_restored = [affinity.translate(h, *scene.origin) for h in holes_centered]
        hole_area_m2 = float(sum(h.area for h in holes_centered))
        outer_area_m2 = float(sum(Polygon(p.exterior).area for p in polygons(scene.target)))
        if (obstacles is not None
                and {"scenario_id", "obstacle_area_m2"}.issubset(obstacles.columns)):
            obstacle_rows = obstacles.loc[
                obstacles["scenario_id"].astype(str) == scene.name
            ]
            declared_obstacle_area_m2 = float(obstacle_rows["obstacle_area_m2"].sum())
        else:
            declared_obstacle_area_m2 = 0.0
        source_area_value = source_row.get("area_m2")
        source_area_attribute_m2 = (
            float(source_area_value)
            if source_area_value is not None and pd.notna(source_area_value)
            else float(projected_source.area)
        )
        symmetric_difference_m2 = float(restored_target.symmetric_difference(projected_source).area)
        target_travel_difference_m2 = float(scene.target.symmetric_difference(scene.travel).area)
        target_only = scene.target.difference(scene.travel)
        travel_only = scene.travel.difference(scene.target)
        cx, cy = scene.target.centroid.coords[0]

        checks = {
            "target_valid": bool(scene.target.is_valid),
            "travel_valid": bool(scene.travel.is_valid),
            "travel_relation_valid": bool(
                scene.travel_source == "explicit"
                or scene.target.buffer(scene.settings.geometry_epsilon_m).covers(scene.travel)
            ),
            "source_geometry_preserved": symmetric_difference_m2 <= 1e-6,
            "target_area_matches_manifest": abs(scene.target.area - entry["target_area_m2"]) <= 1e-6,
            "hole_count_matches_manifest": len(holes_centered) == entry["hole_count"],
            "centroid_shifted_to_origin": abs(cx) <= 1e-7 and abs(cy) <= 1e-7,
        }
        record = {
            "feature_index": entry["feature_index"],
            "field_id": scene.name,
            "source_crs": manifest["source_crs"],
            "work_crs": scene.crs,
            "origin_easting_m": scene.origin[0],
            "origin_northing_m": scene.origin[1],
            "centered_centroid_x_m": cx,
            "centered_centroid_y_m": cy,
            "target_geometry_type": scene.target.geom_type,
            "travel_geometry_type": scene.travel.geom_type,
            "travel_source": scene.travel_source,
            "travel_clearance_m": scene.settings.travel_clearance_m,
            "target_area_m2": float(scene.target.area),
            "travel_area_m2": float(scene.travel.area),
            "outer_area_m2": outer_area_m2,
            "hole_count": len(holes_centered),
            "hole_area_m2": hole_area_m2,
            "source_area_attribute_m2": source_area_attribute_m2,
            "declared_obstacle_area_m2": declared_obstacle_area_m2,
            "target_plus_holes_minus_outer_m2": float(scene.target.area + hole_area_m2 - outer_area_m2),
            "hole_minus_declared_obstacle_m2": hole_area_m2 - declared_obstacle_area_m2,
            "source_symmetric_difference_m2": symmetric_difference_m2,
            "target_only_area_m2": float(target_only.area),
            "travel_only_area_m2": float(travel_only.area),
            "target_travel_symmetric_difference_m2": target_travel_difference_m2,
            "target_travel_boundary_min_distance_m": float(
                scene.target.boundary.distance(scene.travel.boundary)
            ),
            **checks,
        }
        summaries.append(record)

        loaded_payload = {
            "module": "src/scene.py:load_scene",
            "input_scene_json": str(scene_path),
            "name": scene.name,
            "crs": scene.crs,
            "origin": list(scene.origin),
            "start": asdict(scene.start) if scene.start else None,
            "end": asdict(scene.end) if scene.end else None,
            "vehicle": asdict(scene.vehicle),
            "settings": asdict(scene.settings),
            "travel_source": scene.travel_source,
            "target_centered": mapping(scene.target),
            "travel_centered": mapping(scene.travel),
            "metrics": record,
        }
        (output_dir / f"{entry['feature_index']:03d}_{scene.name}.json").write_text(
            json.dumps(loaded_payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        target_rows.append({"field_id": scene.name, "work_crs": scene.crs,
                            "target_area_m2": scene.target.area, "geometry": target_source_crs})
        travel_rows.append({"field_id": scene.name, "work_crs": scene.crs,
                            "travel_area_m2": scene.travel.area, "geometry": travel_source_crs})
        for hole_index, hole in enumerate(holes_restored, start=1):
            hole_rows.append({
                "field_id": scene.name,
                "hole_index": hole_index,
                "hole_area_m2": hole.area,
                "geometry": _project_one(hole, scene.crs, source.crs),
            })

        if holes_centered and (
            detail_candidate is None
            or len(holes_centered) > len(detail_candidate[3])
        ):
            detail_candidate = (
                scene.name, scene.target, scene.travel, holes_centered,
                scene.settings.travel_clearance_m,
            )

        if field_index in visual_positions:
            visual_index = visual_positions[field_index]
            ax = axes.flat[visual_index]
            for poly in polygons(scene.target):
                x, y = poly.exterior.xy
                ax.fill(x, y, color="#d9ead3", alpha=0.9)
                ax.plot(x, y, color="#274e13", linewidth=1.3,
                        label="target/travel boundary")
                for ring in poly.interiors:
                    hx, hy = ring.xy
                    ax.fill(hx, hy, color="#e06666", alpha=0.85)
                    ax.plot(hx, hy, color="#990000", linewidth=1.0)
            ax.scatter([0], [0], color="#1155cc", s=20, zorder=5)
            ax.set_title(
                f"{scene.name}\narea={scene.target.area:.1f} m²; "
                f"holes={len(holes_centered)}",
                fontsize=9,
            )
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
            ax.set_xlabel("centered local x (m)", fontsize=8)
            ax.set_ylabel("centered local y (m)", fontsize=8)

            target_ax, travel_ax, overlay_ax = comparison_axes[visual_index]
            common_bounds = scene.target.union(scene.travel).bounds

            _draw_polygonal(target_ax, scene.target, "#86c98a", "#216e39", alpha=0.82)
            target_ax.scatter([0], [0], color="#111827", s=12, zorder=6)
            target_ax.set_title(
                f"Target work area\n{scene.target.area:.2f} m²",
                fontsize=10,
            )

            _draw_polygonal(travel_ax, scene.travel, "#8ecae6", "#075985", alpha=0.82)
            travel_ax.scatter([0], [0], color="#111827", s=12, zorder=6)
            travel_ax.set_title(
                f"Allowed travel area\n{scene.travel.area:.2f} m²",
                fontsize=10,
            )

            _draw_polygonal(overlay_ax, scene.target, "#86c98a", "#216e39", alpha=0.55)
            _draw_polygonal(
                overlay_ax, scene.travel, "none", "#0369a1",
                alpha=0.0, linewidth=1.5, linestyle="--",
            )
            if not target_only.is_empty:
                _draw_polygonal(
                    overlay_ax, target_only, "#f59e0b", "#b45309", alpha=0.85
                )
            if not travel_only.is_empty:
                _draw_polygonal(
                    overlay_ax, travel_only, "#a855f7", "#6b21a8", alpha=0.75
                )
            overlay_ax.scatter([0], [0], color="#111827", s=12, zorder=6)
            overlay_ax.set_title(
                "Overlay / difference\n"
                f"sym. diff.={target_travel_difference_m2:.6f} m²",
                fontsize=10,
            )

            for comparison_ax in (target_ax, travel_ax, overlay_ax):
                _format_comparison_axis(comparison_ax, common_bounds)
            target_ax.set_ylabel(f"{scene.name}\nlocal y (m)", fontsize=8)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "scene_summary.csv", index=False)
    (out / "scene_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    fig.suptitle(
        f"V6 scene.py output: {visual_count} representative fields of {field_count}; "
        "green=target/travel, red=holes",
        fontsize=14,
    )
    fig.savefig(out / "scene_preview.png", dpi=180)
    plt.close(fig)

    comparison_fig.suptitle(
        f"V6 scene.py: target vs travel ({visual_count} representative fields "
        f"of {field_count})",
        fontsize=15,
    )
    comparison_fig.legend(
        handles=[
            Patch(facecolor="#86c98a", edgecolor="#216e39", label="target work area"),
            Patch(facecolor="#8ecae6", edgecolor="#075985", label="allowed travel area"),
            Line2D([0], [0], color="#0369a1", linestyle="--", label="travel boundary in overlay"),
            Patch(facecolor="#f59e0b", edgecolor="#b45309", label="target only"),
            Patch(facecolor="#a855f7", edgecolor="#6b21a8", label="travel only"),
            Line2D([0], [0], color="#b91c1c", label="hole boundary"),
        ],
        loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.995),
    )
    comparison_fig.savefig(out / "target_travel_comparison.png", dpi=180)
    plt.close(comparison_fig)

    statistics_fig, statistics_axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    statistics_axes[0].scatter(
        summary["target_area_m2"], summary["travel_area_m2"],
        s=18, alpha=0.65, color="#2563eb",
    )
    area_min = float(min(summary["target_area_m2"].min(), summary["travel_area_m2"].min()))
    area_max = float(max(summary["target_area_m2"].max(), summary["travel_area_m2"].max()))
    statistics_axes[0].plot([area_min, area_max], [area_min, area_max],
                            color="#b91c1c", linestyle="--", linewidth=1.2)
    statistics_axes[0].set(
        title="Target area vs allowed travel area",
        xlabel="target area (m²)", ylabel="travel area (m²)",
    )
    statistics_axes[0].grid(alpha=0.2)
    statistics_axes[1].plot(
        summary["feature_index"], summary["target_travel_symmetric_difference_m2"],
        color="#7c3aed", linewidth=1.0,
    )
    statistics_axes[1].set(
        title="Target-travel symmetric difference for every field",
        xlabel="feature index", ylabel="symmetric difference (m²)",
    )
    statistics_axes[1].grid(alpha=0.2)
    statistics_fig.savefig(out / "target_travel_statistics.png", dpi=180)
    plt.close(statistics_fig)

    clearance_detail_path = None
    if detail_candidate is not None:
        detail_name, detail_target, detail_travel, detail_holes, detail_clearance = detail_candidate
        detail_difference = detail_target.difference(detail_travel)
        detail_fig, detail_axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for detail_ax in detail_axes:
            _draw_polygonal(detail_ax, detail_target, "#86c98a", "#216e39", alpha=0.55)
            _draw_polygonal(detail_ax, detail_travel, "#8ecae6", "#0369a1", alpha=0.65)
            _draw_polygonal(detail_ax, detail_difference, "#f59e0b", "#b45309", alpha=0.95)
            detail_ax.set_aspect("equal", adjustable="box")
            detail_ax.grid(alpha=0.2)
            detail_ax.set_xlabel("local x (m)")
            detail_ax.set_ylabel("local y (m)")

        _format_comparison_axis(detail_axes[0], detail_target.bounds)
        detail_axes[0].set_title(
            f"Full field: {detail_name}\norange exclusion band={detail_difference.area:.2f} m²"
        )

        largest_polygon = max(polygons(detail_target), key=lambda polygon: polygon.area)
        outer_point = _longest_segment_midpoint(largest_polygon.exterior)
        detail_axes[1].set_xlim(outer_point.x - 2, outer_point.x + 2)
        detail_axes[1].set_ylim(outer_point.y - 2, outer_point.y + 2)
        _draw_clearance_marker(detail_axes[1], outer_point, detail_travel)
        detail_axes[1].set_title("Outer boundary detail")

        largest_hole = max(detail_holes, key=lambda hole: hole.area)
        hole_point = _longest_segment_midpoint(largest_hole.exterior)
        detail_axes[2].set_xlim(hole_point.x - 2, hole_point.x + 2)
        detail_axes[2].set_ylim(hole_point.y - 2, hole_point.y + 2)
        _draw_clearance_marker(detail_axes[2], hole_point, detail_travel)
        detail_axes[2].set_title("Hole boundary detail")

        detail_fig.suptitle(
            f"Derived travel clearance: configured {detail_clearance:.2f} m",
            fontsize=15,
        )
        detail_fig.legend(
            handles=[
                Patch(facecolor="#86c98a", edgecolor="#216e39", label="target"),
                Patch(facecolor="#8ecae6", edgecolor="#0369a1", label="travel"),
                Patch(facecolor="#f59e0b", edgecolor="#b45309", label="excluded buffer"),
            ],
            loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.93),
        )
        clearance_detail_path = out / "travel_clearance_detail.png"
        detail_fig.savefig(clearance_detail_path, dpi=220)
        plt.close(detail_fig)

    gpkg = out / "scene_outputs.gpkg"
    shutil.copy2(prepared_dir / "input_fields.gpkg", gpkg)
    gpd.GeoDataFrame(target_rows, geometry="geometry", crs=source.crs).to_file(
        gpkg, layer="scene_target", driver="GPKG", index=False
    )
    gpd.GeoDataFrame(travel_rows, geometry="geometry", crs=source.crs).to_file(
        gpkg, layer="scene_travel", driver="GPKG", index=False
    )
    # 无孔洞数据集时 hole_rows 是空列表，GeoDataFrame 无法从中推断
    # geometry 列。此时直接省略空图层，其余审计结果仍是完整的。
    if hole_rows:
        gpd.GeoDataFrame(hole_rows, geometry="geometry", crs=source.crs).to_file(
            gpkg, layer="scene_holes", driver="GPKG", index=False
        )

    all_checks_pass = bool(summary[list(checks)].all(axis=None))
    run_result = {
        "module_under_test": "src/scene.py:load_scene",
        "planner_called": False,
        "source": str(input_path),
        "source_layer": layer,
        "field_count": len(summary),
        "visualized_field_count": visual_count,
        "visualized_feature_indices": visual_indices,
        "different_field_count": int(
            (summary["target_travel_symmetric_difference_m2"] > 1e-6).sum()
        ),
        "scene_holes_layer_written": bool(hole_rows),
        "all_checks_pass": all_checks_pass,
        "outputs": {
            "summary_csv": str(out / "scene_summary.csv"),
            "summary_json": str(out / "scene_summary.json"),
            "loaded_scene_json_directory": str(output_dir),
            "preview_png": str(out / "scene_preview.png"),
            "target_travel_comparison_png": str(out / "target_travel_comparison.png"),
            "target_travel_statistics_png": str(out / "target_travel_statistics.png"),
            "travel_clearance_detail_png": (
                str(clearance_detail_path) if clearance_detail_path else None
            ),
            "review_gpkg": str(gpkg),
            "prepared_scene_directory": str(prepared_dir / "scenes"),
        },
    }
    (out / "run_result.json").write_text(
        json.dumps(run_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--layer", default="fields")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.input, args.config, args.out, args.layer),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
