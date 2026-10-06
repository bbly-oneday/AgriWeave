# AgriWeave 1.0.0（内核V7.0.10） 配置说明

更新日期：2026-10-06。唯一可编辑项目配置是根目录 [config.json](../config.json)；本页数值是本次核对快照，运行以文件实际值为准。JSON不能写//注释，_help保存中文说明：★★常用、★条件生效。输入场景和outputs中的不可变参数快照不属于第二份日常项目配置。

## 分组与推荐设置

| 分组 | 新用户需要理解的内容 |
|---|---|
| vehicle | 车体、固定后置机具及唯一效率速度来源；幅宽3.75m，作业速度2.36m/s，弯道0.8m/s |
| planning | 场景、冻结分区/条带、历史full约束；条件生效参数不能仅因默认reference不读而删除 |
| routes | active_profile=reference；各模式参数放profiles内部，不能混在vehicle/planning |
| efficiency | active_profile=continuous；stops共享停顿，连续变速参数在continuous内 |
| execution | workers=12默认请求上限，按启动时CPU空闲和内存预算缩减，单田硬超时300s，追加重试1次，输入分片64田，默认不绘图 |
| input | default_gpkg=outputs/demo_geographic/fields.gpkg，default_layer=fields；--input default才使用它 |
| compatibility | 旧API映射及可信封存摘要，程序维护，不能手工造摘要绕过发布检查 |

距离m、面积m²、速度m/s、时间s；angles_deg为度，其它rad字段为弧度。尺寸和时间是已有配置假设，没有实测资料时不标为实车值。

## 模式和参数来源

reference是完整几何参考；geometry是历史主体对照；operational是车辆/作物配置模型；strict与separate_headland用于历史严格研究。continuous是完整参考连续变速；constant_full是完整恒速对照；constant_body读取历史主体运动账本。主体几何对照没有完整行程，不能通过切换scope字符串获得整田效率。

命令行--route-profile/--efficiency-profile覆盖active_profile；显式--workers等覆盖execution。效率速度由vehicle提取，停顿由efficiency.stops提取；profiles不能重复定义speeds/stops。

路线消费封存场景中的车辆尺寸和规划参数。改根配置的尺寸不会修改已封存包；独立效率复算会使用当前所选速度和停顿模型。历史严格/运动搜索还读取vehicle.gear_change_seconds与implement_switch_seconds作为求解/模型代价，独立效率读取efficiency.stops，二者需要分别解释。

显式--input路径相对运行目录；run.sh会先切到本仓库根目录。配置default_gpkg相对config.json所在目录。--layer可显式选图层，未指定使用input.default_layer；多图层来源应明确选择。

--scene输入可能已包含车辆/规划参数，场景自带值具有其输入语义；不能假定修改根配置一定覆盖现成场景。已有GIS或生成场景应检查其有效参数快照。详细优先级由scene.load_scene实际读取决定。

## 连续速度和停顿

continuous.speed_transition.acceleration_seconds/deceleration_seconds名义各2秒，行程首尾速度必须0。短距离采用可达速度包络，加减速过程替换同距离恒速时间，已经包含在移动时间中，不额外再加2秒。speed_classification使用LOCAL_ARCLENGTH_V1、2m窗口、1m采样、1度方向差阈值，只影响时间限速，不改变几何路线。

stops中换挡2秒、机具开关1秒、停车转向2秒；同一停点多动作按最大耗时合并。几何条带逆序不等于真实倒挡，不据此造换挡次数。配置允许修改估计，但必须在新结果中保留参数依据。

## 改什么需要重算哪一步

| 修改项 | 需要的操作 |
|---|---|
| 幅宽、车体/机具尺寸、通行退让、主体/分区约束 | 从原始GIS重新生成条带→独立审计→新目录封存→新路线 |
| 路线模式或路线搜索参数 | 消费仍兼容的真实可信条带，写新的路线批次 |
| 作业/弯道速度、停顿、加减速估计 | 独立efficiency复算已有完整路线，写新效率GPKG |
| 并行数或调度内存估计 | 可在匹配合同的--batch-resume时调整，不改变算法参数 |
| 源码，包括可能影响字节摘要的注释 | 不改旧合同；核对兼容检查并使用新批次证据 |

编辑后执行./run.sh --check-config。它校验所有配置模式，包括未激活的模式；新增参数需同步实际消费者、校验、_help、回归和文档，仅添加键名不会生效。planning.allow_revisit是固定true的版本契约，false会拒绝，不是可调的作物时序开关。

## 车辆、规划和调度参数快照

| 参数路径 | 当前值 | 用途/范围 |
|---|---|---|
| `vehicle.body_width_m` | `2.43` | 车体宽度；通行/碰撞包络。 |
| `vehicle.front_m` | `3.54` | 车辆参考点到车头，非轴距。 |
| `vehicle.rear_m` | `1.0` | 参考点到车尾。 |
| `vehicle.working_width_m` | `3.75` | ★★ 有效机具幅宽：决定条带间距和扫掠覆盖。 |
| `vehicle.implement_length_m` | `2.5` | 机具纵向长度；扫掠包络。 |
| `vehicle.implement_offset_m` | `-2.25` | 机具中心相对车辆参考点；后置为负，不是机具尾端。 |
| `vehicle.min_turn_radius_m` | `5.0` | ★ 最小转弯半径；安全核心及运动模型使用，几何参考不做实车认证。 |
| `vehicle.max_curvature_rate` | `0.05` | ★ 曲率沿距离变化上限1/m²，仅运动/严格验证；几何参考不据此认证。 |
| `vehicle.work_speed_mps` | `2.36` | ★★ 开机具直线作业速度；效率模型唯一速度来源。 |
| `vehicle.turn_speed_mps` | `0.8` | ★★ 弯道速度；连续模型在局部弯道减速。 |
| `vehicle.transit_speed_mps` | `2.36` | ★★ 关机具直线空驶速度。 |
| `vehicle.reverse_speed_mps` | `0.5` | ★ 模型有明确倒挡证据才使用，不因条带逆序自动套用。 |
| `vehicle.allow_reverse` | `true` | ★ 严格/运动连接允许倒车；参考几何不等于挡位计划。 |
| `vehicle.safety_margin_m` | `0.5` | ★ 包络安全外扩，不能当条带覆盖幅宽。 |
| `vehicle.gear_change_seconds` | `3.0` | ★ 配置车辆运动/严格路径搜索的换挡代价秒数，当前3；效率账本使用efficiency.stops中的2秒估计，两阶段各有消费者。 |
| `vehicle.implement_switch_seconds` | `1.0` | ★ 车辆运动/严格及历史full的机具升降代价秒数；独立效率账本用efficiency.stops。 |
| `planning.angles_deg` | `[0.0, 90.0]` | ★ 额外候选方向，单位度；分区形状评估、主体候选生成及历史full搜索均读取，不是只作用于full。 |
| `planning.patterns` | `["boustrophedon", "snake"]` | ★ 历史full往复顺序候选。 |
| `planning.decomposition` | `"auto"` | ★ 历史full几何分解：none/auto/always；不替代冻结作业分区。 |
| `planning.headland_m` | `15.0` | ★★ 场景/严格田头宽度；条带/分区读取时按阶段判断，不能直接解释成已作业。 |
| `planning.corridor_m` | `6.0` | ★ 历史严格连接走廊。 |
| `planning.max_candidates` | `4` | ★ 历史full最多候选。 |
| `planning.max_repair_rounds` | `3` | ★ 历史full修补轮数。 |
| `planning.max_tasks` | `350` | ★ 历史full任务上限。 |
| `planning.max_patch_tasks` | `16` | ★ 历史full单轮补条带数。 |
| `planning.max_failed_links` | `3` | ★ 历史连接失败上限。 |
| `planning.max_guide_nodes` | `120` | ★ 严格绕障引导节点上限。 |
| `planning.connection_cache_size` | `1500` | ★ 严格连接缓存容量。 |
| `planning.wall_time_seconds` | `90.0` | ★ 历史full搜索秒数，不是条带或参考批次的单田硬超时。 |
| `planning.sampling_step_m` | `0.1` | ★ 运动/包络距离采样，越小越慢。 |
| `planning.heading_step_rad` | `0.04` | ★ 运动/包络朝向采样步长。 |
| `planning.join_tolerance_m` | `0.02` | ★ 严格运动接头位置容差；参考有独立容差，不等同。 |
| `planning.heading_tolerance_rad` | `0.025` | ★ 严格接头朝向容差。 |
| `planning.curvature_relative_tolerance` | `0.15` | ★ 严格采样曲率相对容差。 |
| `planning.curvature_rate_abs_tolerance` | `0.03` | ★ 严格曲率变化容差，单位1/m²。 |
| `planning.check_curvature_rate` | `true` | ★ 严格曲率变化检查开关。 |
| `planning.geometry_epsilon_m` | `1e-05` | ★ 几何运算数值容差，不是安全退让。 |
| `planning.travel_clearance_m` | `0.3` | ★★ 场景通行边界/障碍物退让，不改变target覆盖义务。 |
| `planning.coverage_tolerance_m2` | `0.01` | ★★ 覆盖/面积账本允许的绝对数值容差，不是95%阈值。 |
| `planning.overlap_fraction` | `0.02` | ★★ 规划搭接比例，0≤值<0.5；主体部分候选间距和接缝预算、历史full条带、模型田头条带读取；并不保证所有接缝均满足预算。 |
| `planning.patch_min_gain_m2` | `0.02` | ★ 历史修补最小新增覆盖面积。 |
| `planning.max_motion_samples` | `30000` | ★ 每段运动采样上限，超限拒绝而非偷偷粗化。 |
| `planning.allow_revisit` | `true` | 固定版本策略，必须true，false会被场景构造器拒绝；不是可以切换的无压苗/作物通行时序开关。保留作为兼容契约说明。 |
| `planning.first_feasible` | `true` | ★ 历史full找到首个通过候选就停止。 |
| `execution.workers` | `12` | ★★ 推荐几何参考默认请求上限12；按启动时0.25秒CPU空闲采样、CPU配额/亲和性、可用内存及待算田数取最小值。0=资源允许范围内自动；正整数可超过12。进程数在本批次内固定，不是实时扩缩容。 |
| `execution.other_stage_workers` | `12` | ★ 多田条带默认请求上限12，同样按启动时CPU空闲和内存预算缩减；历史非推荐路线阶段使用指定进程数，必须正整数。 |
| `execution.worker_memory_mib` | `768.0` | ★★ 推荐参考/多田条带每子进程预估MiB；复杂田可调大，非硬RSS限制。 |
| `execution.memory_budget_mib` | `0.0` | ★★ 0=可用内存的75%并留512MiB给父进程；正值为总预算。 |
| `execution.field_timeout` | `300.0` | ★★ 每田计算+进程启动硬超时秒数；不含全批运行时间。 |
| `execution.retry_failed` | `1` | ★ 失败后的追加重试次数，0~5。 |
| `execution.input_chunk_size` | `64` | ★★ 每次读取最多田块数，1~2000；限制几何常驻内存。 |
| `execution.plot_fields` | `false` | ★ 默认false；仅有需要才生成嵌入图，增加磁盘及时间。 |

## 兼容与追溯

旧--route-config/--efficiency-config继续支持外部真实旧JSON。仅本项目已删除的已知configs路径可转发到根配置的同名模式；这不恢复对应历史参数，也不恢复原始封存文件。

legacy_api规范JSON摘要受源码保护；trusted_swath_bundles由显式封存工具登记。注册表存在条目不表示磁盘文件还在。历史outputs备份目前缺失，后续备份、参数快照及封存输入应单独妥善保存，不能靠文档重新生成旧证据。

## 并行数与资源预算

推荐参考与多田条带按以下最小值选进程：请求上限（默认12）、CPU硬上限、启动时空闲CPU整核估计、内存槽位、待算田数。空闲CPU以0.25秒的逐核负载采样估计，只统计亲和性允许使用的核；忙碌时至少保留一个进程推进任务，不能据此保证独占CPU。采样失败回退CPU硬上限；内存探测失败且没有显式预算时回退一个进程。调度数在本批次内固定；重启/续算会重新探测。

默认内存槽位为floor((可用内存MiB×0.75−512)/768)。因此默认预算要启动12进程，至少需约12.67GiB可用内存，还要满足CPU与任务数条件。内存不足一个槽位时拒绝启动，避免把不足预算默认为充足。该估计不是操作系统硬限额，复杂田块应提高每进程估计。服务器可用--workers 0自动或--workers 64提高上限；不要为凑12进程而任意降低内存预算。历史非推荐路线入口维持原指定进程数语义。
