# 开发与回归

只在config.json维护日常配置；正式源码十一模块，测试、审计和样例准备放tempscript，所有运行产物放outputs。AGENTS.md记录冻结算法和结果范围。

## 定向检查

```bash
export AGRIWEAVE_PYTHON=/absolute/path/to/python
CHECK_DIR="outputs/check_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECK_DIR" outputs/.tmp outputs/.cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/outputs/.tmp"
export MPLCONFIGDIR="$PWD/outputs/.cache/matplotlib"
export PYTHONPATH="$PWD/tempscript:$PWD/src"
./run.sh --check-env > "$CHECK_DIR/environment.log" 2>&1
./run.sh --check-config > "$CHECK_DIR/config.log" 2>&1
"$AGRIWEAVE_PYTHON" -B tempscript/check_release.py > "$CHECK_DIR/release.log" 2>&1
"$AGRIWEAVE_PYTHON" -B -m unittest -v \
  test_route_approx test_speed_profile test_gpkg_integration test_multicore_batch \
  > "$CHECK_DIR/regressions.log" 2>&1
./run.sh --self-test --out "$CHECK_DIR/selftest" > "$CHECK_DIR/selftest.log" 2>&1
```

80项回归包括16项参考几何、15项连续速度、17项GPKG和32项多核调度；另有32项自检。可选的真实16进程隔离测试需要至少20田可信封存包，通过V7_TEST_SWATH_BUNDLE指定；未提供时明确跳过一项，不算通过，也不代表服务器性能验证。

测试自行构造人工几何；GPKG和自检需要真正可用的Fields2Cover原生环境。完整端到端流程按README运行人工三田，并执行audit_compact_reference.py。仓库未保留需要私人历史输出的测试，不能宣称已重跑全部历史实验。

## 修改与发布

修改前保存源码、配置、反例和结果。中文注释解释目的、坐标与单位、返回状态及检查范围。注释也会改变源码哈希，旧包合同不能靠改摘要通过。改变源码后重新生成、独立审核并封存条带，不能只改source_manifest.json。

更新版本时同步发布文档、CITATION.cff、源码清单和真实验证记录；内部config._meta.project_version保留内核版本，避免把包装发布号冒充内核修改。新批次必须记录其实际源码及输入哈希。服务器吞吐和GNSS实车验证分开开展。

config.json中的trusted_swath_bundles由封存工具在本机更新；向GitHub同步配置时审阅并移除本地绝对路径，不能上传私人包目录。
