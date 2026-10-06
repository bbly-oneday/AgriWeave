# V7.0.9 结果读取与时间口径

本页根据当前route_planner的V7_COMPACT_REFERENCE_2声明整理，作为持久使用说明，不依赖已缺失的历史字段字典。只适用于默认完整几何参考+continuous；历史严格/规则/V1输出另有图层和契约，不能重复累计。

## 哪些层和表需要保留

| 名称 | 类型 | 用途 |
|---|---|---|
| field_summary | 常规空间层 | 每田面积、孔洞、路线、质量和时间效率 |
| work_regions | 常规空间层 | 分区面、编号与规划索引，不给分区硬分摊效率 |
| route_segments | 常规空间层 | 唯一完整有序路线账本：主体、田头和连接 |
| route_events | 常规空间层 | STOP/ACCEL/DECEL的点及其行程进度区间 |
| coverage_issues | 条件空间层 | 有覆盖/接缝异常面时建立，保留原因和数值容差 |
| field_results | 属性表 | PENDING/RUNNING/COMPLETED/FAILED、输入上下文及轻量结果状态 |
| parameter_sets | 属性表 | 去重有效时间配置及parameter_set_id，不是第二份日常配置 |
| batch_metadata | 属性表 | 输入发布、源码/参数合同及整批汇总 |
| execution_events | 属性表 | 重试、提交、资源计划和程序耗时 |

独立效率复算还保存相应来源与效率检查点；可选绘图/归档表属于附加材料。geom用于GIS显示，metric_geometry_blob是zlib压缩的本田局部米制二维WKB，metric_geometry_sha256校验解压WKB。复算以米制值为准。

## 路线顺序与状态

用field_id选择田块，再按route_segments.sequence从1排序；不能按fid或并行提交顺序连线。component记录自然独立分量，时间chain记录连续进度。WORK为主体作业，HEADLAND为有序田头，CONNECTION为连接；机具开启状态和动作证据必须与对应策略相符。多个分量之间不补造未知外部转场。

process_state=COMPLETED只代表结果已经提交。full_reference_connected、reference_acceptance_passed、style_status、body_coverage_status、headland_coverage_status、upstream_seam_quality_status、physical_vehicle_certification分别描述不同证据。REVIEW不是自动丢弃结果，也不等于完整实车通过；NOT_EVALUATED不是缺失面积为0。

## 整田时间与效率

| 字段 | 单位和含义 |
|---|---|
| t_work_s | 秒，机具开启移动，含主体、有序田头及相应变速过程 |
| t_nonwork_drive_s | 秒，机具关闭移动，含连接空驶/转弯及相应变速 |
| t_stop_s | 秒，静止动作；同一次停顿取最长动作耗时 |
| t_break_s | 秒，非作业移动+静止停顿 |
| t_total_known_s | 秒，t_work_s+t_break_s；不包含未知外部转场 |
| efficiency_ratio / efficiency_pct | 整田时间比值t_work/(t_work+t_break)，百分数为比值×100 |
| t_accel_process_s / t_decel_process_s | 已包含在移动时间的加/减速过程诊断，不能再加到总时间 |
| t_speed_change_extra_s | 相对同一局部限速分类的瞬时切换基线的时间差 |
| t_v1_model_difference_s | 与旧整段恒速模型的时间差，允许为负，与上一项不同 |
| work_distance_m / headland_work_distance_m | 米，作业移动距离及其中田头距离，不能把田头再重复累加 |
| connection_distance_m / total_distance_m | 米，连接距离及全部有序移动距离 |
| accel_count / decel_count | 变速过程次数，诊断不代表额外停车次数 |
| gear_change_count | 有明确挡位证据时计数；几何条带逆序不自动算倒挡 |
| planning_seconds / execution_events.elapsed_seconds | 程序运行耗时，不是农机作业时间 |

整批加权效率按Σt_work/(Σt_work+Σt_break)计算，不取逐田效率平均。缺完整任务/连接、处理失败或有效时间不足时整田效率NULL，missing_reason保留原因；已知时间小计仍可诊断，但不能代替整田比值。time_source为配置估算，不是实测。

## 面积与障碍物

source_net_area_m2保留来源孔洞；source_gross_area_m2填孔计外轮廓面积；effective_target_area_m2是Scene实际目标。body_required_area_m2来自独立封存主体义务，body_covered_area_m2和body_uncovered_area_m2对该义务计算，不由当前扫掠反推义务来“自证”覆盖。

body_repeat_excess_area_m2是各扫掠面积和减并集面积，重复次数累计；不等于唯一重叠地表面积。coverage_tolerance_m2是数值容差，不是允许漏作比例。已生成田头参考线不表示真实田头全部已作业。

hole_count/hole_area_m2描述来源孔洞；known_obstacle_count/known_obstacle_area_m2只针对明确障碍物资料，未知为NULL，不把每个孔洞自动当作分类障碍物。面积m²，周长/距离m；compactness与elongation为无量纲形状指标。

## 保存、分享与审计

路线阶段逐田提交reference_routes.gpkg，单独时间复算逐田提交work_time_efficiency.gpkg；summary.json为镜像。运行中可只读已提交结果，分享前等待写者关闭并checkpoint WAL。

原始封存输入、源码/参数快照和独立审计也需保管，注册表摘要不能代替实际文件。按README命令用audit_compact_reference.py回读米制坐标、事件、积分时间及主体义务；审计通过不等于实车认证。完整线full_route仅按需导出，默认不重复存储。
