"""将用户田块显式标准化到EPSG:4326；不修改原始输入或规划算法。"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()) or output.exists():
        parser.error("输出必须是本项目outputs中尚未存在的文件")
    frame = gpd.read_file(args.input, layer=args.layer)
    if frame.crs is None or frame.empty:
        parser.error("输入必须有明确CRS且至少包含一个田块；不猜测坐标系")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        parser.error("输入存在空几何，请先处理")
    if not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        parser.error("输入必须全部为Polygon或MultiPolygon")
    converted = frame.to_crs("EPSG:4326")
    bounds = converted.total_bounds
    if not np.isfinite(converted.bounds.to_numpy()).all() or bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -90 or bounds[3] > 90:
        parser.error("转换后坐标非法，请核对源CRS")
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.to_file(output, layer="fields", driver="GPKG", index=False)
    print(f"已标准化{len(converted)}田；输出图层fields，CRS EPSG:4326：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
