"""将多田块 GPKG 拆为 V6 场景；逐田选择当地 UTM，并清理零面积边界毛刺。"""
from __future__ import annotations


def _read_config_json(path):
    # 仅适配配置/JSON读取；历史文件仍按原内容，统一配置按显式节选。
    import sys
    from pathlib import Path
    root=next(p for p in Path(__file__).resolve().parents if (p/'src/io_utils.py').is_file())
    if str(root/'src') not in sys.path:sys.path.insert(0,str(root/'src'))
    from io_utils import read_json
    return read_json(path)


import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import mapping
from shapely.validation import explain_validity

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene import normalize_input_polygon  # noqa: E402


def prepare(input_path: Path, config_path: Path, out: Path, layer: str | None = None) -> dict:
    input_path, config_path, out = input_path.resolve(), config_path.resolve(), out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录非空，拒绝覆盖：{out}")
    layers = gpd.list_layers(input_path)
    spatial = layers.loc[layers.geometry_type.notna(), "name"].tolist()
    if layer is None:
        if len(spatial) != 1:
            raise ValueError(f"必须用 --layer 指定图层，可选：{spatial}")
        layer = spatial[0]
    if layer not in spatial:
        raise ValueError(f"不是可用的空间图层：{layer}")
    frame = gpd.read_file(input_path, layer=layer)
    if frame.empty or frame.crs is None:
        raise ValueError("输入为空或缺少 CRS")
    from io_utils import scene_config
    config = scene_config(config_path)
    if config.get("crs") == "LOCAL_METRIC":
        raise ValueError("GIS 输入不能用 LOCAL_METRIC 覆盖真实 CRS")
    if not config.get("crs") and any(config.get(k) is not None for k in ("travel", "obstacles", "start", "end")):
        raise ValueError("配置含空间坐标时必须显式提供其米制 crs，不能逐田猜测坐标系")
    if "field_id" in frame:
        if frame.field_id.isna().any() or frame.field_id.astype(str).duplicated().any():
            raise ValueError("field_id 不能为空或重复")
    source_crs = CRS.from_user_input(frame.crs)
    # 标准基准样本使用 LOCAL_CS 米制工程坐标，不存在可推断的 UTM。它与
    # 真实 GIS 的经纬度输入不同：单位已经是米，因此可直接作为 Scene 的
    # LOCAL_METRIC 坐标运行，导出时也无需反投影。
    local_metric_source = (
        not source_crs.is_projected
        and source_crs.axis_info
        and all(abs(axis.unit_conversion_factor - 1.0) <= 1e-9 for axis in source_crs.axis_info[:2])
    )
    entries, scenes, normalized_geometries = [], [], []
    for index, (_, row) in enumerate(frame.iterrows()):
        source_geometry = row.geometry
        if (source_geometry is None or source_geometry.is_empty
                or source_geometry.geom_type not in {"Polygon", "MultiPolygon"}):
            raise ValueError(f"第 {index} 行不是非空 Polygon/MultiPolygon")
        source_validity = explain_validity(source_geometry)
        try:
            geometry = normalize_input_polygon(source_geometry, f"第 {index} 行")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        geometry_repaired = not source_geometry.is_valid
        normalized_geometries.append(geometry)
        one = gpd.GeoSeries([geometry], crs=frame.crs)
        metric_crs = (
            None if local_metric_source else
            (CRS.from_user_input(config["crs"]) if config.get("crs") else one.estimate_utm_crs())
        )
        if not local_metric_source and (
            metric_crs is None or not metric_crs.is_projected
            or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in metric_crs.axis_info[:2])
        ):
            raise ValueError(f"第 {index} 行无法确定米制投影")
        projected = geometry if local_metric_source else one.to_crs(metric_crs).iloc[0]
        if projected.is_empty or not projected.is_valid:
            raise ValueError(f"第 {index} 行投影后几何无效")
        source_projected_area = (
            source_geometry.area if local_metric_source else
            gpd.GeoSeries([source_geometry], crs=frame.crs).to_crs(metric_crs).iloc[0].area
        )
        field_id = str(row["field_id"]) if "field_id" in frame else f"feature_{index}"
        slug = f"{index:03d}_" + (re.sub(r"[^\w.-]+", "_", field_id)[:100] or "field")
        scene = copy.deepcopy(config)
        scene.update(
            name=field_id,
            crs="LOCAL_METRIC" if local_metric_source else metric_crs.to_string(),
            target=mapping(projected),
        )
        scenes.append((slug, scene))
        entries.append({"field_id": field_id, "feature_index": index, "directory": slug,
                        "scene": f"scenes/{slug}.json",
                        "work_crs": "LOCAL_METRIC" if local_metric_source else metric_crs.to_string(),
                        "target_area_m2": projected.area,
                        "hole_count": sum(len(p.interiors) for p in (projected.geoms if projected.geom_type == "MultiPolygon" else [projected])),
                        "geometry_repaired": geometry_repaired,
                        "source_validity": source_validity,
                        "repair_area_change_m2": projected.area - source_projected_area})
    # 所有几何审核完成后才写出，避免半份 manifest 被误用。
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    for slug, scene in scenes:
        (out / "scenes" / f"{slug}.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    frame = frame.copy()
    frame.geometry = normalized_geometries
    frame.to_file(out / "input_fields.gpkg", layer="input_fields", driver="GPKG", index=False)
    manifest = {"source": str(input_path), "source_layer": layer, "source_crs": frame.crs.to_string(),
                "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "config": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "projection_policy": (
                    "input_local_metric" if local_metric_source else
                    ("explicit_config_crs" if config.get("crs") else "per_feature_local_utm")
                ),
                "geometry_repaired": any(entry["geometry_repaired"] for entry in entries),
                "geometry_repair_policy": "zero_area_artifacts_only_same_part_count_and_area",
                "fields": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--layer")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.input, args.config, args.out, args.layer)
    print(json.dumps({"fields": len(manifest["fields"]), "manifest": str(args.out.resolve() / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
