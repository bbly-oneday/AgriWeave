"""构造人工田块验证公开流程；不是实测田块或GNSS数据。"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.affinity import translate
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/demo_geographic/fields.gpkg")
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("人工样例必须写入本项目outputs目录")
    if output.exists():
        parser.error("目标已存在，请选用新的outputs路径")
    shapes = [
        box(0, 0, 100, 160),
        Polygon([(0, 0), (140, 0), (140, 65), (80, 65), (80, 160), (0, 160)]),
        box(0, 0, 140, 160).difference(box(50, 55, 90, 105)),
    ]
    frame = gpd.GeoDataFrame(
        {"field_id": ["synthetic_rectangle", "synthetic_concave", "synthetic_hole"],
         "data_kind": ["SYNTHETIC_NOT_OBSERVED"] * 3},
        geometry=[translate(shape, 500000 + i * 300, 4000000)
                  for i, shape in enumerate(shapes)],
        crs="EPSG:32650",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # 内部以米构造真实尺寸；公开输入统一转EPSG:4326，避开当前精简
    # 导出协议中field_summary直接复用输入几何的投影坐标限制。
    frame.to_crs("EPSG:4326").to_file(output, layer="fields", driver="GPKG", index=False)
    print(f"生成3个人工田块（米制构造，EPSG:4326存储）：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
