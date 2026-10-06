# AgriWeave 开发规则

- 正式实现位于src的十一模块；测试、审计和数据准备位于tempscript。先阅读README、docs/ARCHITECTURE.md、docs/CONFIGURATION.md和docs/RESULTS.md。
- 根目录config.json是唯一日常项目配置。新增参数时同步消费者、校验、_help、文档和回归。输出参数快照和docs/source_manifest.json是追溯证据，不是额外用户配置。
- 用户数据通过--input提供；不提交真实田块、运行输出、凭据或虚拟环境。所有测试、日志、临时文件、缓存和运行产物放outputs。
- 发布1.0.0的十一模块与V7.0.10逐字节相同。scene.py、planner.py、swath_planner.py、swath_seams.py和swath_batch.py保持算法冻结，修改需获得维护者明确授权。
- 默认reference+continuous输出完整几何参考路线及配置估算效率。参考联通、驾驶风格、主体覆盖、田头真实覆盖和实车认证分开报告。不能把独立GPKG一致性审计当成实车认证。
- 修改前保存基线及反例，修改后运行相关回归；不能只替换哈希使未验证代码通过。更新源码后必须由真实输入建立新条带包、独立审计和显式封存。
- 新批次用新目录；--batch-resume仅续算相同源码、输入和算法合同的批次，调度资源可调整。父进程唯一写GPKG，逐田结果和状态同事务提交；未知效率保持NULL；分享前等待关库，不丢失WAL。
- 缺少真实田门、车辆参数或外部转场时不得编造。人工样例、350田验证、十万级压力验证和实车验证分别说明。
- 若本仓库存在.codegraph，先用CodeGraph定位代码；未建立索引时使用rg，不自行建立索引。
