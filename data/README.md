# 350田块样本

本目录按维护者于2026-10-06的明确要求，发布V7/data中的原始fields2cover_350fields.gpkg。文件原样复制，没有重投影、简化或修改字段；与内核350田验证输入一致。原始运行结果不在本目录。

| 属性 | 值 |
|---|---|
| 文件 | fields2cover_350fields.gpkg |
| 大小 | 507,904字节，约496KiB |
| 图层 | fields |
| CRS | EPSG:4326 |
| 田块数 | 350 |
| 唯一field_id数 | 350 |
| 字段 | fid、geom、field_id、source_file、country、is_valid |
| SQLite完整性 | ok |
| SHA-256 | f7386c198ceb6906e4ea999e132bbbce006efc39682cc515ab9b7c7f742549e0 |

source_file和country是源数据的追溯信息，原样保留；本次未增加新的来源或实测声明。代码的MIT许可不自动变更数据及其原始来源的许可条件，本次未为数据另行指定许可。

完整运行命令见[README](../README.md#运行随仓库发布的350田块)。配置默认输入仍为人工三田，使用这份数据须显式指定`--input data/fields2cover_350fields.gpkg --layer fields`。样本包含此前记录的接缝和驾驶风格复核案例，不能将运行完成视为实车认证。

机器可读摘要见[manifest.json](manifest.json)。CI校验文件大小、SHA-256、数据库完整性、图层、CRS及田块数量，只证明发布数据一致性，不执行350田完整规划。
