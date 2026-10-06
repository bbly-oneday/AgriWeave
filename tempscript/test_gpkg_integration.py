"""真实 GPKG 坐标/边界回归及原生 F2C 分区回归；不替代整田验收。"""


def _read_config_json(path):
    # 仅适配配置/JSON读取；历史文件仍按原内容，统一配置按显式节选。
    import sys
    from pathlib import Path
    root=next(p for p in Path(__file__).resolve().parents if (p/'src/io_utils.py').is_file())
    if str(root/'src') not in sys.path:sys.path.insert(0,str(root/'src'))
    from io_utils import read_json
    return read_json(path)

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import geopandas as gpd
from shapely import affinity
from shapely.geometry import Point, Polygon, box, mapping

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from prepare_gpkg import prepare
from inspect_scene_module import inspect
from scene import Scene, Settings, Vehicle, Budget, load_scene, normalize_input_polygon
from planner import F2CBackend, generate_tasks, task_from_swath


class GpkgIntegration(unittest.TestCase):
    def test_zero_area_spikes_are_normalized_but_bowties_remain_rejected(self):
        # The boundary goes into the field and immediately returns along the same
        # segment. It changes no area and should not reject an otherwise usable field.
        spiked = Polygon([
            (0, 0), (10, 0), (10, 10), (6, 10),
            (6, 7), (6, 10), (0, 10), (0, 0),
        ])
        self.assertFalse(spiked.is_valid)
        normalized = normalize_input_polygon(spiked, "target")
        self.assertTrue(normalized.is_valid)
        self.assertAlmostEqual(normalized.area, spiked.area, places=12)

        bowtie = Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)])
        with self.assertRaisesRegex(ValueError, "改变面域或面数量"):
            normalize_input_polygon(bowtie, "target")

    def test_scene_rejects_unknown_top_level_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps({
                "crs": "LOCAL_METRIC",
                "target": mapping(box(0, 0, 10, 10)),
                "obstcles": mapping(box(4, 4, 6, 6)),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知顶层字段.*obstcles"):
                load_scene(path)

    def test_present_travel_and_obstacles_cannot_use_falsey_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            base = {"crs": "LOCAL_METRIC", "target": mapping(box(0, 0, 10, 10))}
            for field in ("travel", "obstacles"):
                for invalid in ({}, False, None, []):
                    with self.subTest(field=field, invalid=invalid):
                        path.write_text(json.dumps({**base, field: invalid}), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, field):
                            load_scene(path)

    def test_scene_root_and_config_sections_must_be_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            for invalid_root in (None, False, [], "scene"):
                with self.subTest(root=invalid_root):
                    path.write_text(json.dumps(invalid_root), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "根节点.*JSON 对象"):
                        load_scene(path)

            base = {"crs": "LOCAL_METRIC", "target": mapping(box(0, 0, 10, 10))}
            for section in ("vehicle", "planning"):
                with self.subTest(section=section):
                    path.write_text(json.dumps({**base, section: None}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"{section}.*JSON 对象"):
                        load_scene(path)

    def test_explicit_travel_buffers_boundary_intersecting_obstacle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps({
                "crs": "LOCAL_METRIC",
                "target": mapping(box(0, 0, 10, 10)),
                "travel": mapping(box(-5, -5, 15, 15)),
                "obstacles": mapping(box(-1, 4, 1, 6)),
                "planning": {"travel_clearance_m": 0.3},
            }), encoding="utf-8")
            scene = load_scene(path)
            near_obstacle = Point(1.1 - scene.origin[0], 5.0 - scene.origin[1])
            beyond_clearance = Point(1.31 - scene.origin[0], 5.0 - scene.origin[1])
            self.assertFalse(scene.travel.covers(near_obstacle))
            self.assertTrue(scene.travel.covers(beyond_clearance))

    def test_boolean_numeric_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            base = {"crs": "LOCAL_METRIC", "target": mapping(box(0, 0, 10, 10))}
            invalid_planning = [
                {"angles_deg": [True]},
                {"wall_time_seconds": True},
                {"headland_m": True},
            ]
            for planning in invalid_planning:
                with self.subTest(planning=planning):
                    path.write_text(json.dumps({**base, "planning": planning}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_scene(path)
            path.write_text(json.dumps({**base, "start": [True, False, True]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "start/end"):
                load_scene(path)

    def test_explicit_travel_and_endpoint_spatial_relationships_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            target = mapping(box(0, 0, 10, 10))
            path.write_text(json.dumps({
                "crs": "LOCAL_METRIC", "target": target,
                "travel": mapping(box(100, 100, 110, 110)),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "travel.*target"):
                load_scene(path)

            for endpoint in ("start", "end"):
                with self.subTest(endpoint=endpoint):
                    path.write_text(json.dumps({
                        "crs": "LOCAL_METRIC", "target": target,
                        "travel": mapping(box(-5, -5, 15, 15)),
                        endpoint: [1000, 1000, 0],
                    }), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, endpoint):
                        load_scene(path)

    def test_scene_inspection_handles_dataset_without_holes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "no_holes.gpkg"
            geometry = Polygon([
                (5.0000, 52.0000), (5.0020, 52.0000),
                (5.0020, 52.0010), (5.0000, 52.0010),
            ])
            gpd.GeoDataFrame({"field_id": ["plain"]}, geometry=[geometry], crs=4326).to_file(
                source, layer="fields", driver="GPKG", index=False
            )
            result = inspect(source, PROJECT / "config.json", root / "inspection", "fields")
            gpkg = Path(result["outputs"]["review_gpkg"])
            self.assertTrue(gpkg.exists())
            self.assertEqual(
                set(gpd.list_layers(gpkg)["name"]),
                {"input_fields", "scene_target", "scene_travel"},
            )

    def test_main_is_the_public_gpkg_batch_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "one_field.gpkg"
            output = root / "result"
            geometry = Polygon([
                (5.0000, 52.0000), (5.0020, 52.0000),
                (5.0020, 52.0010), (5.0000, 52.0010),
            ])
            gpd.GeoDataFrame({"field_id": ["one"]}, geometry=[geometry], crs=4326).to_file(
                source, layer="fields", driver="GPKG", index=False
            )
            completed = subprocess.run(
                [
                    sys.executable, str(PROJECT / "src/main.py"),
                    "--input", str(source), "--layer", "fields",
                    "--config", str(PROJECT / "config.json"),
                    "--seconds", "0.2", "--out", str(output),
                ],
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            self.assertIn(completed.returncode, {0, 2}, completed.stdout + completed.stderr)
            self.assertNotIn("unrecognized arguments", completed.stderr)
            self.assertTrue((output / "prepared/manifest.json").exists())
            self.assertTrue((output / "batch_summary.json").exists())
            summary = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input"], str(source.resolve()))
            self.assertEqual([row["field_id"] for row in summary["fields"]], ["one"])

    def test_config_preserves_documented_vehicle_parameter_basis(self):
        # 独立固定的参数依据；删除重复V5 YAML后仍检查迁移未改变机具和参考点定义。
        config=_read_config_json(PROJECT/'config.json')
        expected=dict(body_width_m=2.43,front_m=3.54,rear_m=1.,working_width_m=3.75,
            implement_length_m=2.5,implement_offset_m=-2.25,min_turn_radius_m=5.,
            max_curvature_rate=.05,work_speed_mps=2.36,turn_speed_mps=.8,
            transit_speed_mps=2.36,reverse_speed_mps=.5,allow_reverse=True,
            safety_margin_m=.5,gear_change_seconds=3.)
        for name,value in expected.items():self.assertEqual(config['vehicle'][name],value,name)
        self.assertEqual(config['_meta']['parameter_profile'],'JD6R140_PROLANDER400R')
        self.assertNotIn('profile',config)

    def test_all_five_fields_preserve_crs_geometry_and_identity(self):
        source = PROJECT / "../data/fields2cover_regular_size_5samples.gpkg"
        original = gpd.read_file(source)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = prepare(source, PROJECT / "config.json", out)
            self.assertEqual(len(manifest["fields"]), 5)
            self.assertEqual([x["work_crs"] for x in manifest["fields"]],
                             ["EPSG:32635", "EPSG:32631", "EPSG:32635", "EPSG:32632", "EPSG:32632"])
            restored = gpd.read_file(out / "input_fields.gpkg")
            self.assertEqual(restored.field_id.tolist(), original.field_id.tolist())
            self.assertEqual(restored.crs, original.crs)
            for field in manifest["fields"]:
                scene = load_scene(out / field["scene"])
                source_geometry = original.geometry.iloc[field["feature_index"]]
                expected = gpd.GeoSeries([source_geometry], crs=original.crs).to_crs(scene.crs).iloc[0]
                actual = affinity.translate(scene.target, *scene.origin)
                actual_travel = affinity.translate(scene.travel, *scene.origin)
                self.assertLess(actual.symmetric_difference(expected).area, 1e-6)
                expected_travel = expected.buffer(
                    -scene.settings.travel_clearance_m, quad_segs=32
                )
                self.assertLess(actual_travel.symmetric_difference(expected_travel).area, 1e-6)
                self.assertTrue(scene.target.covers(scene.travel))
                self.assertFalse(scene.target.equals(scene.travel))
                self.assertEqual(scene.travel_source, "derived_from_target_buffer")
                self.assertEqual(scene.name, field["field_id"])

    def test_holes_survive_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poly = Polygon([(500000, 5000000), (500100, 5000000), (500100, 5000100), (500000, 5000100)],
                           holes=[[(500040, 5000040), (500060, 5000040), (500060, 5000060), (500040, 5000060)]])
            gpd.GeoDataFrame({"field_id": ["with_hole"]}, geometry=[poly], crs=32631).to_file(root / "hole.gpkg")
            manifest = prepare(root / "hole.gpkg", PROJECT / "config.json", root / "prepared")
            entry = manifest["fields"][0]
            scene = load_scene(root / "prepared" / entry["scene"])
            self.assertEqual(entry["hole_count"], 1)
            self.assertEqual(len(scene.target.interiors), 1)
            self.assertAlmostEqual(scene.target.area, 9600, places=5)
            self.assertAlmostEqual(
                scene.travel.symmetric_difference(
                    scene.target.buffer(
                        -scene.settings.travel_clearance_m, quad_segs=32
                    )
                ).area,
                0.0,
                places=7,
            )
            self.assertGreater(scene.target.difference(scene.travel).area, 0.0)

    def test_invalid_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = Polygon([(0, 0), (10, 10), (0, 10), (10, 0)])
            gpd.GeoDataFrame(geometry=[invalid], crs=32631).to_file(root / "bad.gpkg")
            with self.assertRaisesRegex(ValueError, "几何无效"):
                prepare(root / "bad.gpkg", PROJECT / "config.json", root / "prepared")
            self.assertFalse((root / "prepared/manifest.json").exists())

    def test_spatial_config_requires_explicit_crs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"start": [1, 2, 0]}))
            with self.assertRaisesRegex(ValueError, "显式"):
                prepare(PROJECT / "../data/fields2cover_regular_size_5samples.gpkg", config, root / "prepared")

    def test_real_native_decomposition_produces_tasks(self):
        polygon = Polygon([(0, 0), (100, 0), (100, 30), (60, 30), (60, 60), (100, 60), (100, 90), (0, 90)])
        scene = Scene(polygon, box(-20, -20, 120, 110), Vehicle(),
                      Settings(headland_m=2, decomposition_algorithm="trapezoidal"))
        backend = F2CBackend(scene)
        tasks, metadata = generate_tasks(scene, backend, (0.0, "snake", True), Budget(15))
        self.assertTrue(tasks)
        self.assertEqual(metadata["decomposition_algorithm"], "trapezoidal")
        self.assertGreater(len({t.cell_id for t in tasks}), 1)

    def test_native_collinear_vertices_remain_a_straight_task(self):
        import fields2cover as f2c
        line = f2c.LineString()
        line.importFromWkt("LINESTRING (75 99,24 99,12 99,6 99,-22 99)")
        task = task_from_swath(f2c.Swath(line, 6.0), "collinear", 0)
        self.assertEqual((task.start.x, task.end.x), (75, -22))

    def test_native_bent_and_backtracking_swaths_still_rejected(self):
        import fields2cover as f2c
        for text in ("LINESTRING (0 0,5 1,10 0)", "LINESTRING (0 0,12 0,10 0)"):
            line = f2c.LineString()
            line.importFromWkt(text)
            with self.assertRaisesRegex(ValueError, "折线或折返"):
                task_from_swath(f2c.Swath(line, 6.0), "invalid", 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
