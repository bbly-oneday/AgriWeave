# 环境安装与检查

验证组合：Python 3.12、Fields2Cover 2.1.0，以及requirements.txt中的版本。安装可复用已有原生环境；全新机器的安装时间与系统库兼容性未纳入本次发布验证。

## 原生库先于Python项目运行

Fields2Cover提供C++库与Python绑定，依赖GDAL、GEOS、OR-tools等。遵循[官方安装说明](https://fields2cover.github.io/source/installation.html)或[官方仓库](https://github.com/Fields2Cover/Fields2Cover)。选择与本项目已验证API兼容的2.1.0，绑定须能被你实际运行本项目的Python解释器导入。安装到其它解释器的绑定无效。

可以用已有原生Python创建环境；不要直接搬迁虚拟环境目录。独立安装时按官方说明选择CMake的Python目标和依赖位置。仓库不提供未经验证的一键原生安装脚本。

```bash
export AGRIWEAVE_PYTHON=/absolute/path/to/python
"$AGRIWEAVE_PYTHON" -c 'import fields2cover; print(fields2cover.__version__)'
"$AGRIWEAVE_PYTHON" -m pip install -r requirements.txt
./run.sh --check-env
./run.sh --check-config
```

requirements.txt为实际验证版本记录；其它Python、原生库或库版本需要重新运行定向回归及人工完整流程，不能自动继承本版本验收结果。

## 常见故障

- `ModuleNotFoundError: fields2cover`：检查AGRIWEAVE_PYTHON和绑定安装目标是否一致；requirements.txt不会安装原生绑定。
- 原生动态库加载失败：按官方安装说明核对GDAL、GEOS及OR-tools的ABI和动态库查找路径。
- `--input default`找不到文件：先运行人工样例生成脚本，或在唯一config.json中设置自己的输入路径。
- 旧条带包源码合同失败：从对应原始输入重新生成、审核、封存。不要修改摘要绕过校验。
- 进程数低于12：查看worker_plan.json或路线summary.json中的资源计划；可能受可用内存、空闲CPU或任务数限制。
