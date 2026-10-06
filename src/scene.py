"""V7场景入口与共享模型，继承V6场景契约。读取田块/机具参数，将几何整理为本田局部米制Scene；target是不可静默改变的作业义务，travel是允许通行空间。起终位姿的输入角度为度，内部yaw为弧度。场景检查不生成分区或路线。"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import hashlib
import json
import math
import time
from typing import Any

import numpy as np
from pyproj import CRS
from shapely import affinity
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity, make_valid
from shapely.ops import unary_union


# 场景文件是几何和车辆安全参数的统一输入入口。顶层字段拼错时不能继续使用默认值，
# 尤其是 obstacles/travel；否则本应禁止通行的区域可能完全不进入后续几何计算。
SCENE_JSON_FIELDS = frozenset({
    "profile", "name", "crs", "target", "travel", "obstacles",
    "vehicle", "planning", "start", "end",
})


def _is_finite_number(value: Any) -> bool:
    """只接受真正的有限数值；JSON 布尔值不能冒充 0 或 1。"""
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value))


@dataclass(frozen=True)
class Vehicle:
    """规划和安全验证共同使用的车辆/机具参数。

    路径中的 ``(x, y, yaw)`` 表示车辆参考点及车身朝向。``front_m`` 和
    ``rear_m`` 是参考点到车体前后端的纵向距离，不是轴距。机具被近似为一个
    与车身刚性连接的矩形；``implement_offset_m`` 表示机具中心相对参考点的
    纵向偏移，负值表示位于参考点后方。

    ``working_width_m`` 在当前版本同时表示机具实体宽度和有效作业幅宽。
    ``safety_margin_m`` 不改变 target 或 travel，而是在 validator 中加到车体和
    机具的扫掠包络外侧。数据类设置为 frozen，避免规划途中被意外修改。
    """

    body_width_m: float = 2.5
    front_m: float = 3.0          # 路径参考点至车体前端；不是轴距
    rear_m: float = 1.0
    working_width_m: float = 6.0
    implement_length_m: float = 0.6
    implement_offset_m: float = -1.5  # 刚性机具中心相对参考点，后方为负
    min_turn_radius_m: float = 5.0
    max_curvature_rate: float = 0.15  # dκ/ds，单位 1/m²
    work_speed_mps: float = 2.0
    turn_speed_mps: float = 1.0
    transit_speed_mps: float = 2.0
    reverse_speed_mps: float = 0.6
    allow_reverse: bool = False
    safety_margin_m: float = 0.05
    gear_change_seconds: float = 3.0
    implement_switch_seconds: float = 1.0

    def __post_init__(self) -> None:
        """尽早拒绝缺失、非有限或物理范围明显错误的车辆参数。"""
        nonnegative = {"rear_m", "safety_margin_m", "gear_change_seconds", "implement_switch_seconds"}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "allow_reverse":
                if type(value) is not bool:
                    raise ValueError("allow_reverse 必须为 JSON 布尔值")
                continue
            if not _is_finite_number(value):
                raise ValueError(f"非法车辆参数: {f.name}")
            if f.name == "implement_offset_m":
                continue
            if value < 0 or (value == 0 and f.name not in nonnegative):
                raise ValueError(f"车辆参数范围错误: {f.name}")

    def rectangles(self, work_only: bool = False) -> list[np.ndarray]:
        """返回车辆局部坐标系中的矩形轮廓。

        每个数组按顺序保存四个 ``[纵向坐标, 横向坐标]`` 顶点。validator 会按
        每个路径姿态旋转、平移这些矩形，再构造连续扫掠包络。``work_only=True``
        时只返回机具矩形，用于计算实际作业覆盖；否则同时返回车体和机具，
        用于碰撞及越界检查。
        """

        def rectangle(x0: float, x1: float, half_width: float) -> np.ndarray:
            return np.array([[x0, -half_width], [x1, -half_width],
                             [x1, half_width], [x0, half_width]], dtype=float)
        tool = rectangle(self.implement_offset_m - self.implement_length_m / 2,
                         self.implement_offset_m + self.implement_length_m / 2,
                         self.working_width_m / 2)
        if work_only:
            return [tool]
        return [rectangle(-self.rear_m, self.front_m, self.body_width_m / 2), tool]


@dataclass(frozen=True)
class Settings:
    """控制候选生成、数值离散、资源上限和验收容差的规划参数。

    这些值来自场景 JSON 的 ``planning`` 节点。距离单位为米，面积单位为平方米，
    时间单位为秒；``angles_deg`` 使用度，其余角度步长和容差使用弧度。这里集中
    校验参数，是为了让 planner、repair 和 validator 看到完全相同的配置，避免
    各模块分别解释 JSON。
    """

    # 候选方向、往复模式和田块分解策略。
    angles_deg: tuple[float, ...] = (0.0, 90.0)
    patterns: tuple[str, ...] = ("boustrophedon", "snake")
    decomposition: str = "auto"       # none / auto / always
    decomposition_algorithm: str = "auto"  # auto / boustrophedon / trapezoidal

    # 田头、连接走廊以及候选/修补搜索规模。
    headland_m: float = 15.0
    corridor_m: float = 6.0
    max_candidates: int = 4
    max_repair_rounds: int = 3
    max_tasks: int = 350
    max_patch_tasks: int = 16
    max_failed_links: int = 3
    max_guide_nodes: int = 120
    connection_cache_size: int = 1500

    # 墙钟时间和路径离散上限。步长越小，扫掠近似通常越紧，计算量也越大。
    wall_time_seconds: float = 90.0
    sampling_step_m: float = 0.10
    heading_step_rad: float = 0.04
    join_tolerance_m: float = 0.02
    heading_tolerance_rad: float = 0.025
    curvature_relative_tolerance: float = 0.15
    curvature_rate_abs_tolerance: float = 0.03
    check_curvature_rate: bool = True

    # 几何容差只用于吸收浮点误差，不代表允许真实越界。
    geometry_epsilon_m: float = 1e-5
    # target 保留扣除障碍物后的覆盖义务；缺省 travel 据此内缩外边界并外扩孔洞。
    travel_clearance_m: float = 0.30
    coverage_tolerance_m2: float = 0.01  # 默认只容忍数值量级残余，而不是默认允许漏 1%

    # 补作收益、运动采样上限及当前版本支持的作业规则。
    overlap_fraction: float = 0.02
    patch_min_gain_m2: float = 0.02
    max_motion_samples: int = 30000
    allow_revisit: bool = True
    first_feasible: bool = True

    def __post_init__(self) -> None:
        """检查枚举值、布尔类型、计数上限和所有数值范围。"""
        if not self.angles_deg or not all(_is_finite_number(a) for a in self.angles_deg):
            raise ValueError("angles_deg 不能为空且必须为有限数")
        if not self.patterns or any(p not in {"snake", "boustrophedon"} for p in self.patterns):
            raise ValueError("patterns 只支持 snake、boustrophedon")
        if self.decomposition not in {"none", "auto", "always"}:
            raise ValueError("decomposition 必须为 none / auto / always")
        if self.decomposition_algorithm not in {"auto", "boustrophedon", "trapezoidal"}:
            raise ValueError("decomposition_algorithm 必须为 auto / boustrophedon / trapezoidal")
        for name in ("allow_revisit", "first_feasible", "check_curvature_rate"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须为 JSON 布尔值")
        if not self.allow_revisit:
            raise ValueError("此版本仅支持允许重复通行的静态作业空间；不支持收获后才开放通道")
        for name in ("max_candidates", "max_tasks", "max_patch_tasks", "max_failed_links",
                     "max_guide_nodes", "connection_cache_size", "max_motion_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} 必须为正整数")
        if type(self.max_repair_rounds) is not int or self.max_repair_rounds < 0:
            raise ValueError("max_repair_rounds 必须为非负整数")
        for name in ("wall_time_seconds", "sampling_step_m", "heading_step_rad", "join_tolerance_m",
                     "heading_tolerance_rad", "geometry_epsilon_m", "patch_min_gain_m2"):
            value = getattr(self, name)
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(f"{name} 必须为正有限数")
        for name in ("headland_m", "corridor_m", "travel_clearance_m", "coverage_tolerance_m2",
                     "curvature_relative_tolerance", "curvature_rate_abs_tolerance"):
            value = getattr(self, name)
            if not _is_finite_number(value) or value < 0:
                raise ValueError(f"{name} 必须为非负有限数")
        if (not _is_finite_number(self.overlap_fraction)
                or not 0 <= self.overlap_fraction < 0.5):
            raise ValueError("overlap_fraction 必须在 [0, 0.5) 内")


def make_config(cls: type, values: dict[str, Any]) -> Any:
    """把 JSON 字典安全地转换成配置数据类。

    未知字段通常意味着拼写错误或旧版本配置。如果静默忽略，程序可能使用默认值
    继续运行并产生难以发现的实验偏差，因此这里直接报错。JSON 数组在 Settings
    中转换成元组，使 frozen 配置保持不可变。
    """
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{cls.__name__} 含未知参数: {sorted(unknown)}")
    if cls is Settings:
        values = dict(values)
        for key in ("angles_deg", "patterns"):
            if key in values:
                values[key] = tuple(values[key])
    return cls(**values)


@dataclass(frozen=True)
class Pose:
    """车辆参考点的二维位姿；x/y 单位为米，yaw 单位为弧度。"""

    x: float
    y: float
    yaw: float


@dataclass
class Scene:
    """一个可以直接交给规划器的单田场景。

    ``target`` 和 ``travel`` 已转换到同一个局部米制坐标系；其中 ``target`` 已
    扣除显式障碍物。``origin`` 保存局部原点在原投影坐标中的位置，导出结果时
    必须加回。``travel_source`` 用于区分用户显式提供的通行区和由 target 缓冲
    生成的通行区，方便审计结果来源。
    ``start``/``end`` 为 None 时表示规划器可以自由选择起终位姿，不代表已经识别
    出真实田间入口。
    """

    target: BaseGeometry
    travel: BaseGeometry
    vehicle: Vehicle
    settings: Settings
    start: Pose | None = None
    end: Pose | None = None
    crs: str = "LOCAL_METRIC"
    origin: tuple[float, float] = (0.0, 0.0)
    name: str = "field"
    travel_source: str = "explicit"


@dataclass(frozen=True)
class Task:
    """一条尚未连接进完整路线的直线作业任务。

    ``cell_id`` 记录任务来自哪个分解单元；``task_id`` 在排序、连接、修补和导出
    之间保持稳定，用于追踪某条作业带是否真正出现在最终路径中。
    """

    task_id: str
    start: Pose
    end: Pose
    kind: str = "work"
    cell_id: int = 0

    def reverse(self) -> Task:
        """交换起终点并反转朝向，同时保留任务身份和所属分区。"""
        return Task(self.task_id,
                    Pose(self.end.x, self.end.y, wrap(self.end.yaw + math.pi)),
                    Pose(self.start.x, self.start.y, wrap(self.start.yaw + math.pi)),
                    self.kind, self.cell_id)


@dataclass
class Motion:
    """规划器和验证器之间传递的一段连续车辆运动。

    ``points`` 是 N×4 数组，四列依次为 x、y、车身朝向和下一小段的行驶方向。
    行驶方向只能为 ``+1``（前进）或 ``-1``（倒车）；最后一行没有后继线段，
    因而其方向值不参与运动解释。``kind`` 用于区分 work、headland、patch、turn
    和 transit。只有 ``implement_on=True`` 的运动才计入覆盖面积。

    初始化时会复制数组并设为只读，随后根据坐标和机具状态生成稳定的 ``key``。
    validator 使用该 key 缓存高成本的扫掠检查；如果调用方原地修改 points，缓存
    就会失真，所以这里明确禁止修改。
    """
    points: np.ndarray
    kind: str                  # work / headland / patch / turn / transit
    task_id: str | None = None
    implement_on: bool = False
    link: tuple[str, str] | None = None
    key: str = field(init=False)

    def __post_init__(self) -> None:
        """规范化运动数组，检查形状和挡位，并生成内容哈希。"""
        p = np.array(self.points, dtype=float, copy=True)
        if p.ndim != 2 or p.shape[1] != 4 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError("Motion 需要至少两个有限的 [x,y,yaw,direction] 状态")
        if not np.isin(p[:, 3], [-1, 1]).all():
            raise ValueError("行驶方向只允许 +1/-1")
        p.setflags(write=False)
        self.points = p
        self.key = hashlib.blake2b(p.tobytes() + str(self.implement_on).encode(), digest_size=16).hexdigest()

    @property
    def length(self) -> float:
        """返回相邻采样点欧氏距离之和，单位为米。"""
        return float(np.linalg.norm(np.diff(self.points[:, :2], axis=0), axis=1).sum())


@dataclass
class Plan:
    """一个候选规划结果，包括任务、连续运动以及生成阶段发现的错误。

    tasks 表示计划完成的作业任务，motions 表示车辆实际执行的顺序运动。二者分开
    保存，使 validator 能发现“任务已生成但没有进入最终路线”的缺失情况。
    metadata 只记录候选规格和诊断信息，不参与几何验收。
    """

    tasks: list[Task]
    motions: list[Motion]
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Budget:
    """多个规划模块共享的单调时钟时间预算。

    使用 ``time.monotonic``，避免系统时钟校准导致剩余时间突然增加或减少。
    各个可能长时间循环的模块主动调用 ``check``；超时表示本次搜索预算耗尽，
    不等同于证明该田块不存在可行路线。
    """

    def __init__(self, seconds: float) -> None:
        """创建墙钟预算，seconds单位为秒；后续check检查期限，预算耗尽不等于田块物理不可达。"""
        self.started = time.monotonic()
        self.deadline = self.started + seconds

    def check(self) -> None:
        """预算用尽时立即抛出 TimeoutError，由上层记录终止原因。"""
        if time.monotonic() >= self.deadline:
            raise TimeoutError("本次规划时间预算已用完；这不代表物理不可行")

    @property
    def elapsed(self) -> float:
        """返回从预算创建到当前时刻的墙钟时间，单位为秒。"""
        return time.monotonic() - self.started


def wrap(angle: float) -> float:
    """把任意弧度角规范到 [-pi, pi) 区间。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def polygons(geometry: BaseGeometry) -> list[Polygon]:
    """递归提取几何对象中的全部 Polygon，忽略线和点。

    Shapely 的 difference、intersection 和 make_valid 可能返回 MultiPolygon 或
    GeometryCollection。规划代码只接受有面积的多边形部分，因此统一通过这个
    小函数展开，避免每个调用处重复判断几何类型。
    """
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if hasattr(geometry, "geoms"):
        return [p for g in geometry.geoms for p in polygons(g)]
    return []


def polygonal(geometry: BaseGeometry) -> BaseGeometry:
    """合并几何中的多边形部分；没有面时返回空 GeometryCollection。"""
    parts = polygons(geometry)
    return unary_union(parts) if parts else GeometryCollection()


def normalize_input_polygon(g: BaseGeometry, name: str) -> BaseGeometry:
    """清理不改变面域的零面积毛刺，但拒绝会改变真实边界的自动修复。

    一些来源文件含连续重复点，或沿同一条线出去后立即原路返回。这些线状毛刺
    没有面积，也不代表车辆可进入的空间。``make_valid`` 会把它们分离为线，并
    保留原来的面。只有修复前后面数量相同且面积在数值精度内不变时才接受；
    蝴蝶结交叉、面被拆分或面积改变仍交给人工审核。
    """
    if not isinstance(g, (Polygon, MultiPolygon)) or g.is_empty:
        raise ValueError(f"{name} 必须为非空 Polygon/MultiPolygon")
    if g.is_valid:
        if g.area <= 0:
            raise ValueError(f"{name} 必须为有面积的 Polygon/MultiPolygon")
        return g

    reason = explain_validity(g)
    repaired = polygonal(make_valid(g))
    original_parts = len(polygons(g))
    repaired_parts = len(polygons(repaired))
    area_unchanged = math.isclose(
        repaired.area, g.area, rel_tol=1e-9, abs_tol=1e-12
    )
    if (not isinstance(repaired, (Polygon, MultiPolygon))
            or repaired.is_empty or not repaired.is_valid
            or repaired_parts != original_parts or not area_unchanged):
        raise ValueError(
            f"{name} 几何无效: {reason}；自动修复会改变面域或面数量，请人工审核"
        )
    return repaired


def _valid_polygon(g: BaseGeometry, name: str) -> BaseGeometry:
    """确认几何是有效且面积为正的 Polygon/MultiPolygon。

    这里故意不调用 ``make_valid`` 或 ``buffer(0)``。自动修复可能改变边界、拆分
    地块或删除细小结构，这些变化必须在数据准备阶段由研究人员单独审核。
    """
    if not isinstance(g, (Polygon, MultiPolygon)) or g.is_empty or g.area <= 0:
        raise ValueError(f"{name} 必须为非空 Polygon/MultiPolygon")
    if not g.is_valid:
        raise ValueError(f"{name} 几何无效: {explain_validity(g)}。请审核后修复，程序不自动改边界")
    return g


def _json_object(value: Any, name: str) -> dict[str, Any]:
    """要求 JSON 节点是对象，避免 ``null`` 等值泄漏成内部类型错误。"""
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须为 JSON 对象")
    return value


def _json_polygon(data: dict[str, Any], name: str) -> BaseGeometry:
    """读取一个明确提供的 GeoJSON 面，并把解析异常统一为输入校验错误。"""
    value = data[name]
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须为 GeoJSON 几何对象")
    try:
        geometry = shape(value)
    except Exception as exc:
        raise ValueError(f"{name} 不是有效的 GeoJSON 几何对象") from exc
    return _valid_polygon(normalize_input_polygon(geometry, name), name)


def load_scene(path: str | Path, *, field_path: str | None = None,
               feature_index: int = 0, layer: str | None = None) -> Scene:
    """读取、校验并标准化一个单田场景。

    ``path`` 指向 JSON 场景或合并后的 config/scene 文件。调用方也可以通过
    ``field_path`` 指定 SHP/GPKG；这时只读取 ``feature_index`` 对应的一个要素，
    用它覆盖 JSON 中的 target，但仍沿用 JSON 中的车辆和规划参数。函数不会把
    整个图层合并成一块田。

    处理顺序是：读取配置 -> 读取 target -> 检查米制 CRS -> 扣除显式障碍物 ->
    处理显式或派生 travel -> 校验几何 -> 平移到局部原点 -> 返回 Scene。
    返回的几何可以直接用于面积、距离、buffer、规划和安全验证。
    """

    # 先验证 JSON 结构和字段名。关键约束字段拼错时必须立即失败，不能把它当成
    # “未提供”并继续派生 travel 或忽略障碍物。
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = _json_object(data, "场景 JSON 根节点")
    unknown = set(data) - SCENE_JSON_FIELDS
    if unknown:
        raise ValueError(f"场景含未知顶层字段: {sorted(unknown)}")
    for object_name in ("profile", "vehicle", "planning"):
        if object_name in data:
            _json_object(data[object_name], object_name)

    # 配置对象继续完成内部字段名和数值范围检查，后续代码无需再处理原始字典。
    vehicle = make_config(Vehicle, data.get("vehicle", {}))
    settings = make_config(Settings, data.get("planning", {}))
    crs = data.get("crs")

    # GIS 模式只负责替换 target。批处理器已经逐田调用本函数，因此这里不能把
    # 多个要素 dissolve；否则会丢失田块身份并把田间空隙误当成同一作业场景。
    if field_path:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise RuntimeError("读取 SHP/GPKG 需安装 geopandas；JSON 场景不需要") from exc
        frame = gpd.read_file(field_path, **({"layer": layer} if layer else {}))
        if not 0 <= feature_index < len(frame):
            raise ValueError("feature_index 超出图层范围")
        if frame.crs is None:
            raise ValueError("输入图层缺少 CRS，不能猜测坐标单位")
        source_crs = CRS.from_user_input(frame.crs)
        if crs and crs != "LOCAL_METRIC":
            source_crs = CRS.from_user_input(crs)
            frame = frame.to_crs(source_crs)
        elif crs == "LOCAL_METRIC":
            raise ValueError("GIS 图层不能通过 LOCAL_METRIC 忽略原 CRS")
        crs = source_crs.to_string()
        target = normalize_input_polygon(frame.geometry.iloc[feature_index], "target")
    else:
        if "target" not in data:
            raise ValueError("场景缺少 target；或使用 --field 指定矢量文件")
        target = _json_polygon(data, "target")

    # buffer、车辆宽度和转弯半径都以米为单位。经纬度的“1”是角度而不是1米，
    # 英尺投影同样会产生系统性尺度错误，所以两者都在入口处拒绝。
    if not crs:
        raise ValueError("必须声明米制投影 CRS 或明确使用 LOCAL_METRIC")
    if crs != "LOCAL_METRIC":
        parsed = CRS.from_user_input(crs)
        if not parsed.is_projected or any(abs(a.unit_conversion_factor - 1.0) > 1e-9
                                           for a in parsed.axis_info[:2]):
            raise ValueError("请先转换为当地米制投影（如 UTM）；禁止用经纬度或英尺直接规划")

    # 输入 target 扣除障碍物后成为 Scene.target，也是覆盖率分母。障碍物安全缓冲
    # 只进一步约束 travel，不再缩小 Scene.target，避免用安全带制造虚假的高覆盖率。
    target = _valid_polygon(target, "target")
    explicit_travel = "travel" in data
    travel = _json_polygon(data, "travel") if explicit_travel else target
    if "obstacles" in data:
        obstacles = _json_polygon(data, "obstacles")
        target = polygonal(target.difference(obstacles))
        if explicit_travel and settings.travel_clearance_m:
            # 直接缓冲原始障碍物，不依赖 difference 后它是否仍表现为孔洞。
            # 因此，与 target 外边界相交的障碍物也有完整的基础禁入带。
            blocked = obstacles.buffer(settings.travel_clearance_m, quad_segs=32)
        else:
            blocked = obstacles
        travel = polygonal(travel.difference(blocked))
    target = _valid_polygon(target, "target")
    if explicit_travel:
        # 显式 travel 适合表达合法的田外掉头区、入口或连接通道，因此不能简单裁剪
        # 到 target 内部。不过 target 中的孔洞代表禁入障碍，即使显式 travel 覆盖了
        # 孔洞，也必须把孔洞及其安全带重新扣除。
        holes = [Polygon(ring) for poly in polygons(target) for ring in poly.interiors]
        if holes:
            hole_obstacles = unary_union(holes).difference(target)
            if settings.travel_clearance_m:
                hole_obstacles = hole_obstacles.buffer(
                    settings.travel_clearance_m, quad_segs=32
                )
            travel = polygonal(travel.difference(hole_obstacles))
        travel_source = "explicit"
    else:
        # Shapely 负 buffer 同时将外边界向内移、孔洞边界向外移，正好形成车辆不能
        # 进入的基础安全带。quad_segs 提高圆角近似精度；窄小地块若被完全侵蚀，
        # 后面的严格校验会报错，而不会返回一个看似可规划的空场景。
        travel = polygonal(
            target.buffer(-settings.travel_clearance_m, quad_segs=32)
        )
        travel_source = "derived_from_target_buffer"
    target, travel = _valid_polygon(target, "target"), _valid_polygon(travel, "travel")

    # 对自动派生的 travel，检查它没有因几何运算跑到 target 外。显式 travel 允许
    # 包含合法田外区域，但至少要与 target 存在正面积交集。
    if not explicit_travel and not target.buffer(settings.geometry_epsilon_m).covers(travel):
        raise ValueError("由 target 派生的 travel 必须位于目标作业区内")
    if explicit_travel and target.intersection(travel).area <= settings.geometry_epsilon_m ** 2:
        raise ValueError("显式 travel 必须与 target 存在正面积交集")
    # UTM 坐标通常达到几十万或几百万米。以 target 质心为局部原点可以减小浮点
    # 运算中的有效数字损失，同时让调试图和局部路径坐标更直观。origin 会随 Scene
    # 保存，main.py 导出时负责加回，因而不会改变数据的真实空间位置。
    ox, oy = target.centroid.coords[0]

    def read_pose(value: Any) -> Pose | None:
        """把 JSON 的 [投影x, 投影y, 角度] 转为局部米坐标和弧度朝向。"""
        if value is None:
            return None
        if (not isinstance(value, (list, tuple)) or len(value) != 3
                or not all(_is_finite_number(x) for x in value)):
            raise ValueError("start/end 应为 [x_m, y_m, heading_deg]")
        return Pose(float(value[0]) - ox, float(value[1]) - oy, math.radians(float(value[2])))

    # target、travel 和起终位姿使用同一次平移，保证模块间不存在坐标基准差异。
    local_target = affinity.translate(target, -ox, -oy)
    local_travel = affinity.translate(travel, -ox, -oy)
    start, end = read_pose(data.get("start")), read_pose(data.get("end"))
    for pose_name, pose in (("start", start), ("end", end)):
        if (pose is not None
                and not local_travel.buffer(settings.geometry_epsilon_m).covers(
                    Point(pose.x, pose.y)
                )):
            raise ValueError(f"{pose_name} 必须位于允许通行区 travel 内")
    return Scene(local_target, local_travel,
                 vehicle, settings, start, end,
                 str(crs), (ox, oy), str(data.get("name", Path(path).stem)), travel_source)
