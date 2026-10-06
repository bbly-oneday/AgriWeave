#!/bin/sh
# 从任意目录启动；只设置运行环境，不改变规划算法。
set -eu
AGRIWEAVE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${AGRIWEAVE_PYTHON:-}" ]; then
    AGRIWEAVE_RUNTIME=$AGRIWEAVE_PYTHON
elif [ -n "${V7_PYTHON:-}" ]; then
    AGRIWEAVE_RUNTIME=$V7_PYTHON
elif [ -x "$AGRIWEAVE_ROOT/.venv/bin/python" ]; then
    AGRIWEAVE_RUNTIME="$AGRIWEAVE_ROOT/.venv/bin/python"
else
    AGRIWEAVE_RUNTIME=$(command -v python3 || true)
fi
if [ -z "$AGRIWEAVE_RUNTIME" ] || [ ! -x "$AGRIWEAVE_RUNTIME" ]; then
    printf '%s\n' '找不到Python。请创建.venv或设置AGRIWEAVE_PYTHON为解释器绝对路径。' >&2
    exit 1
fi
cd "$AGRIWEAVE_ROOT"
mkdir -p outputs/.cache/matplotlib outputs/.tmp
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$AGRIWEAVE_ROOT/outputs/.cache/matplotlib"
export XDG_CACHE_HOME="$AGRIWEAVE_ROOT/outputs/.cache"
export TMPDIR="$AGRIWEAVE_ROOT/outputs/.tmp"
exec "$AGRIWEAVE_RUNTIME" -B src/main.py "$@"
