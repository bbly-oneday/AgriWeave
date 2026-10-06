# AgriWeave · 田织

**Farmland reference routes and work-time efficiency, woven from field geometry.**

AgriWeave根据田块形态、孔洞和农机配置，生成分区、主体条带及完整几何参考路线，估算作业、非作业行驶、停顿和加减速时间，并逐田保存到GeoPackage。适用于农田宜机化研究、田块方案比较和批量参考效率评价。

首个公开版本为 **1.0.0研究版**，内核来自V7.0.10。十一正式模块与350田验证版本逐字节相同，公开发布仅整理启动、样例、配置中的本地登记信息和文档。它输出规划参考结果，尚未完成实车运动认证或田头真实覆盖认证。

## 能做什么

- 保留原始作业目标，分析通行空间、分区和主体条带。
- 组织有序条带、区域间连接及田头参考行驶，保留驾驶风格和覆盖告警。
- 按连续变速模型计算 `efficiency = t_work / (t_work + t_break)`；缺少完整路线时效率为NULL。
- 多田并行计算，默认请求上限12，按启动时空闲CPU、容器配额和内存缩减。
- 路线阶段逐田事务写GPKG，支持超时、失败记录及相同合同下的断点续算。

## 安装

验证环境为Python 3.12、Fields2Cover 2.1.0。Fields2Cover是原生依赖；`requirements.txt`仅锁定已验证的Python运行库。请先按[Fields2Cover官方安装说明](https://fields2cover.github.io/source/installation.html)安装GDAL、GEOS、OR-tools等依赖及Python绑定，并确认目标解释器能`import fields2cover`。不要假定任意新版本原生绑定兼容。

```bash
git clone https://github.com/bbly-oneday/AgriWeave.git
cd AgriWeave
# 复用已经能导入Fields2Cover的Python环境，或按官方说明创建本项目.venv。
export AGRIWEAVE_PYTHON=/absolute/path/to/python
"$AGRIWEAVE_PYTHON" -m pip install -r requirements.txt
./run.sh --check-env
./run.sh --check-config
```

`run.sh`依次选择`AGRIWEAVE_PYTHON`、兼容变量`V7_PYTHON`、本仓库`.venv/bin/python`、PATH中的python3。它将缓存和临时文件放在outputs。独立脚本的运行环境设置如下：

```bash
# 没有显式指定解释器时，请使用与run.sh一致且已安装原生依赖的python。
export AGRIWEAVE_PYTHON=/absolute/path/to/python
mkdir -p outputs/.tmp outputs/.cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/outputs/.tmp"
export MPLCONFIGDIR="$PWD/outputs/.cache/matplotlib"
```

首次发布已经在维护者原生环境完成验证；尚未验证全新Linux/macOS机器的一键安装。环境要求和常见问题见[安装说明](docs/INSTALLATION.md)。

## 五步运行人工样例

仓库不包含真实田块或运行产物。样例脚本以米制尺寸构造矩形、凹形和带孔洞的三个**人工田块**，再以EPSG:4326保存输入，全部产物位于outputs。以下命令各运行一次；重复运行请使用新的批次目录名。

```bash
# 1. 生成样例；仅为流程验证，不是实测资料
"$AGRIWEAVE_PYTHON" -B tempscript/create_demo_fields.py
# 2. 主体条带；default指向上述人工样例
./run.sh --stage swaths --input default --out outputs/demo_swaths
# 3. 独立审核并封存；工具在config.json登记本地可信包
"$AGRIWEAVE_PYTHON" -B tempscript/seal_swath_bundle.py \
  --source outputs/demo_swaths --out outputs/demo_swaths_sealed
# 4. 完整参考路线与连续变速效率；逐田写GPKG
./run.sh --stage routes --swath-bundle outputs/demo_swaths_sealed \
  --out outputs/demo_routes
# 5. 独立核验有序路线、事件、时间和封存主体覆盖
"$AGRIWEAVE_PYTHON" -B tempscript/audit_compact_reference.py \
  --gpkg outputs/demo_routes/reference_routes.gpkg \
  --out outputs/demo_routes/audit.json
```

主体批次返回码2可能表示已有结果但接缝或质量检查未通过；先查看`swath_batch_summary.json`及具体问题，不能改摘要强行封存。独立封存检查主体覆盖，接缝告警仍须保留，不能称为全部质量通过。

## 使用自己的田块

公开1.0推荐输入为**EPSG:4326的GPKG多边形图层**，推荐唯一的`field_id`；面内孔洞视为障碍。入口按田块转换为米制坐标计算。当前精简导出将field_summary中的原始面直接标为EPSG:4326，投影原始输入须先用以下工具标准化，否则整田面与路线显示坐标不一致，独立审计会失败。这是保留内核中的已知限制，不能仅靠GIS重标CRS解决。

```bash
# 对投影坐标输入或其它地理CRS，先显式转换；不修改原始文件。
"$AGRIWEAVE_PYTHON" -B tempscript/normalize_field_input.py \
  --input /path/to/fields.gpkg --layer parcels \
  --out outputs/input_normalized.gpkg
```

下面使用标准化输出，图层名为fields。已有EPSG:4326数据可以直接使用自己的路径。复杂Scene JSON和历史策略需按各自协议另行验证，见[配置说明](docs/CONFIGURATION.md)。

```bash
./run.sh --stage swaths --input outputs/input_normalized.gpkg --layer fields \
  --out outputs/my_swaths
"$AGRIWEAVE_PYTHON" -B tempscript/seal_swath_bundle.py \
  --source outputs/my_swaths --out outputs/my_swaths_sealed
./run.sh --stage routes --swath-bundle outputs/my_swaths_sealed \
  --workers 0 --out outputs/my_routes
# 中断后，仅续算同版本、同输入和同算法参数的批次
./run.sh --stage routes --swath-bundle outputs/my_swaths_sealed \
  --workers 0 --out outputs/my_routes --batch-resume
```

只编辑根目录[config.json](config.json)。`_help`提供中文说明：★★为常调参数，★为条件生效参数。工作幅宽、退让、分区或条带参数变化后重新生成并封存；仅速度/停顿参数变化可独立重算效率：

```bash
./run.sh --stage efficiency --route-batch outputs/my_routes \
  --out outputs/my_efficiency
```

加速、减速名义各2秒，短段限制可达峰值；变速过程已计入移动时间，不再额外重复加2秒。所有速度和时间均为配置估计。

## 输出与并行

最终权威文件为`outputs/<批次>/reference_routes.gpkg`。

| 图层/表 | 用途 |
|---|---|
| field_summary | 每田路线状态、质量告警、距离、时间和效率 |
| work_regions | 作业分区 |
| route_segments | 按field_id和sequence排序的完整参考路线段 |
| route_events | 停顿、加速及减速事件 |
| coverage_issues | 存在异常时记录问题面 |
| field_results、parameter_sets、batch_metadata、execution_events | 完成/失败状态、参数及追溯证据 |

不能按GPKG的fid或并行提交次序串路线。压缩米制WKB是计算依据，显示geom用于GIS。分享前等待写者退出，不删除运行中的WAL。

默认每工作进程估计768MiB，并预留512MiB给父进程；实际进程数可能低于12。`--workers 0`按资源自动选择，可超过12。单田内部串行；路线阶段每田默认硬超时300秒、追加重试1次、读取分片64田。历史`--stage full`是不同的旧入口，不能用来宣称推荐路线阶段的并行或持久化行为。详细口径见[结果说明](docs/RESULTS.md)与[架构](docs/ARCHITECTURE.md)。

## 验证与已知限制

源内核最近一次350田全流程验证：350/350完整参考联通、350/350配置估算效率、350/350独立审计通过，生产流程29分32秒；85田保留接缝重叠告警，21田保留驾驶风格REVIEW。公开仓库提供可复跑的人工样例和源码摘要，不分发该真实数据集、坐标或原始GPKG。详见[验证记录](docs/VALIDATION.md)和[剩余限制](docs/LIMITATIONS.md)。

参考连接通过不等于整体质量全部通过，也不等于车辆可以实际转过所有连接。田头参考线不证明田头已覆盖；没有真实田门、轴距、GNSS及作业日志时不能宣称实测效率。几十万田块尚未完成规模验证；先分片并进行代表性服务器压力测试。

开发检查、贡献规则见[开发说明](docs/DEVELOPMENT.md)与[贡献指南](CONTRIBUTING.md)。仓库自带CI仅检查源码摘要、语法与发布文件完整性，不能替代原生GIS回归或实车验收。

## 许可与引用

本项目代码采用[MIT许可](LICENSE)，Fields2Cover及其它依赖保留各自许可，见[第三方说明](THIRD_PARTY_NOTICES.md)。人工样例由本仓库代码生成；真实数据不在本许可授权范围内。引用本项目请使用GitHub的Cite this repository入口，并说明具体版本与配置估算范围。
