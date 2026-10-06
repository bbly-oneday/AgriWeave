"""V7跨分区接缝的真实作业审计。所有几何为本田局部米制，面积为平方米。以实际机具扫掠计算重叠和跨区代作，先按提供分区合并，避免把区内条带搭接算成跨区接缝；目标所属区和实际提供作业区分别保留。"""
from __future__ import annotations

from typing import Any

from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from shapely.strtree import STRtree


def _lines(geometry):
    """递归提取接缝审计所需线几何，空几何返回空列表。"""
    if geometry.is_empty:
        return []
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [geometry]
    return [line for part in getattr(geometry, "geoms", ()) for line in _lines(part)]


def audit_seams(scene: Any, analysis: Any, results: list[Any]) -> dict[str, Any]:
    """按实际扫掠计算跨区重复作业次数及成对预算；每提供分区先合并扫掠，区内搭接不误计为跨区接缝。
    
    Return area by work multiplicity, pair budgets and physical footprints."""
    region_geometry = {region.region_id: region.geometry for region in analysis.regions}
    active = [item for item in results if item.region_id in region_geometry]
    swept = {}
    outside = {}
    for item in active:
        swept[item.region_id] = item.work_sweeps.intersection(scene.target)
        outside[item.region_id] = sum(
            segment["geometry"].difference(region_geometry[item.region_id]).length
            for segment in item.segments if segment.get("implement_on", True)
        )
    names = [item.region_id for item in active]
    geoms = [swept[name] for name in names]
    tree = STRtree(geoms) if geoms else None
    w = float(scene.vehicle.working_width_m)
    p = float(scene.settings.overlap_fraction)
    reach = max(abs(scene.vehicle.implement_offset_m) + scene.vehicle.implement_length_m / 2,
                scene.vehicle.front_m, scene.vehicle.rear_m)
    tolerance = float(scene.settings.coverage_tolerance_m2)
    pairs = []
    intersections = []
    for i, left in enumerate(names):
        if geoms[i].is_empty:
            continue
        for j in tree.query(geoms[i]):
            j = int(j)
            if j <= i:
                continue
            right = names[j]
            overlap = geoms[i].intersection(geoms[j])
            shared = region_geometry[left].boundary.intersection(
                region_geometry[right].boundary)
            boundary_lines = _lines(shared)
            length = sum(line.length for line in boundary_lines)
            meaningful = sum(line.length >= w for line in boundary_lines)
            budget = w * p * length + 2.0 * meaningful * w * reach
            area = float(overlap.area)
            if area <= 0 and length <= 0:
                continue
            pairs.append({"from_region": left, "to_region": right,
                          "overlap_area_m2": area, "shared_boundary_m": length,
                          "boundary_component_count": meaningful,
                          "budget_m2": budget + tolerance,
                          "status": "PASS" if area <= budget + tolerance else "EXCESS_OVERLAP",
                          "reason": ("WITHIN_MACHINE_SCALE_BUDGET" if area <= budget + tolerance
                                     else "ACTUAL_CROSS_REGION_SWEEP_EXCEEDS_BUDGET"),
                          "geometry": overlap})
            if area > 0:
                intersections.append(overlap)
    summed = sum(geometry.area for geometry in geoms)
    union = unary_union([geometry for geometry in geoms if not geometry.is_empty]) if geoms else GeometryCollection()
    extra = max(0.0, float(summed - union.area))
    footprint = unary_union(intersections) if intersections else GeometryCollection()
    # 成对重叠预算只是诊断；三个分区重叠不能被重复算三遍，真实作业次数损失另行检查。
    total_budget = sum(row["budget_m2"] - tolerance for row in pairs) + tolerance
    status = ("PASS" if all(row["status"] == "PASS" for row in pairs)
              and extra <= total_budget else "EXCESS_OVERLAP")
    return {"status": status, "extra_area_m2": extra,
            "overlap_footprint_area_m2": float(footprint.area),
            "overlap_footprint": footprint, "pair_overlap_sum_m2": sum(
                row["overlap_area_m2"] for row in pairs),
            "budget_m2": total_budget, "pairs": pairs,
            "centerline_outside_m": sum(outside.values()),
            "centerline_outside_by_region_m": outside,
            "swept_by_region": swept}


def overlapping_work_tasks(scene: Any, results: list[Any],
                           region_pairs: list[dict[str, Any]],
                           *, minimum_area_m2: float = 0.0) -> list[dict[str, Any]]:
    """只在指定分区对内定位真正重叠的作业任务，输出米制交集面积，避免无关任务的全量两两比较。
    
    Locate the actual task pairs behind a region-level seam overlap.
    
    Only the supplied region pairs are inspected. An STRtree on the smaller
    task set avoids an all-field Cartesian product, including for point-contact
    regions whose implement footprints overlap without a shared boundary line."""
    by_region = {item.region_id: item for item in results}
    target = scene.target
    rows: list[dict[str, Any]] = []
    for pair in region_pairs:
        left_id, right_id = pair["from_region"], pair["to_region"]
        left = [item for item in by_region[left_id].segments
                if item.get("implement_on", True)
                and not item["work_sweep"].is_empty]
        right = [item for item in by_region[right_id].segments
                 if item.get("implement_on", True)
                 and not item["work_sweep"].is_empty]
        if not left or not right:
            continue
        smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
        tree = STRtree([item["work_sweep"] for item in smaller])
        for other in larger:
            for index in tree.query(other["work_sweep"]):
                own = smaller[int(index)]
                first, second = ((own, other) if len(left) <= len(right)
                                 else (other, own))
                overlap = first["work_sweep"].intersection(
                    second["work_sweep"]).intersection(target)
                if overlap.area <= minimum_area_m2:
                    continue
                rows.append({"from_region": left_id, "to_region": right_id,
                             "from_task": first["task_id"],
                             "to_task": second["task_id"],
                             "overlap_area_m2": float(overlap.area),
                             "geometry": overlap})
    rows.sort(key=lambda row: (-row["overlap_area_m2"], row["from_region"],
                               row["to_region"], row["from_task"], row["to_task"]))
    return rows


def cross_region_assignments(scene: Any, analysis: Any,
                             results: list[Any]) -> list[dict[str, Any]]:
    """优先用本区作业覆盖归属目标，仅把剩余义务登记为跨区代作，不能改变目标所属分区。
    
    Record required body area that only a foreign work task supplies.
    
    Own-region work gets priority. A foreign provider is recorded only for the
    remaining uncovered target, never as an accounting-only deduplication."""
    owners = {item.region_id: item.required_main_area.intersection(scene.target)
              for item in results if item.status == "SWATHS_COMPLETE"}
    partition_order = {region.region_id: int(region.sequence_index)
                       for region in analysis.regions}
    task_rank = {}
    rank = 0
    for provider in sorted(results, key=lambda item: (
            partition_order.get(item.region_id, 10**9), item.region_id)):
        for segment in sorted(provider.segments,
                              key=lambda row: (row["suggested_order"], row["task_id"])):
            rank += 1
            task_rank[segment["task_id"]] = rank
    rows = []
    for owner_id, target in owners.items():
        own = next((item for item in results if item.region_id == owner_id), None)
        allocated = own.work_sweeps.intersection(target) if own else GeometryCollection()
        for provider in results:
            if provider.region_id == owner_id:
                continue
            for segment in provider.segments:
                contribution = segment["work_sweep"].intersection(target).difference(allocated)
                if contribution.area <= scene.settings.coverage_tolerance_m2:
                    continue
                rows.append({"owner_region_id": owner_id,
                             "provider_region_id": provider.region_id,
                             "task_id": segment["task_id"],
                             "provider_suggested_order": segment["suggested_order"],
                             "provider_task_rank": task_rank[segment["task_id"]],
                             "owner_sequence_index": partition_order[owner_id],
                             "provider_sequence_index": partition_order[provider.region_id],
                             "execution_rule": "PROVIDER_TASK_BEFORE_OWNER_COMPLETION",
                             "dispatch_order_status": "TASK_PRECEDENCE_RECORDED_TRANSFER_UNCERTIFIED",
                             "area_m2": float(contribution.area),
                             "geometry": contribution})
                allocated = allocated.union(contribution)
    last_own_rank = {owner_id: max((task_rank[segment["task_id"]]
                                   for provider in results
                                   if provider.region_id == owner_id
                                   for segment in provider.segments), default=0)
                     for owner_id in owners}
    completion_rank = dict(last_own_rank)
    for row in rows:
        owner_id = row["owner_region_id"]
        completion_rank[owner_id] = max(
            completion_rank[owner_id], row["provider_task_rank"])
    for row in rows:
        row["owner_completion_not_before_rank"] = completion_rank[
            row["owner_region_id"]]
    return rows
