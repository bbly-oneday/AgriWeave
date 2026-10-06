"""检查公开发布的语法、源码摘要和文件范围；不宣称算法或实车验收。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check() -> dict:
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    expected = manifest["source_sha256"]
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "src").glob("*.py")}
    if actual != expected:
        raise ValueError("RELEASE_SOURCE_HASH_MISMATCH")
    for path in [*(ROOT / "src").glob("*.py"), *(ROOT / "tempscript").glob("*.py")]:
        ast.parse(path.read_text(), filename=str(path))
    config = json.loads((ROOT / "config.json").read_text())
    if config["compatibility"]["trusted_swath_bundles"]["releases"]:
        raise ValueError("PUBLIC_TEMPLATE_MUST_NOT_CONTAIN_LOCAL_RELEASE_REGISTRY")
    required = ["README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md",
                "CONTRIBUTING.md", "docs/VALIDATION.md", "docs/LIMITATIONS.md"]
    for name in required:
        if not (ROOT / name).is_file():
            raise ValueError("MISSING_RELEASE_FILE: " + name)
    # 若有Git仓库，仅检查实际跟踪文件；outputs内的验证产物不属于公开内容。
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                             capture_output=True, check=False)
    if tracked.returncode == 0:
        files = [ROOT / p for p in tracked.stdout.decode().split("\0") if p]
    else:
        files = [p for p in ROOT.rglob("*") if p.is_file()
                 and not p.relative_to(ROOT).parts[0].startswith(".")
                 and p.relative_to(ROOT).parts[0] not in {"outputs", "data"}]
    forbidden = re.compile(r"/Users/|BaiduNetDisk|gh[pousr]_[A-Za-z0-9]{20,}|"
                           r"github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"outputs", "data", ".venv"} or path.suffix in {".gpkg", ".pkl"}:
            raise ValueError("PRIVATE_OR_GENERATED_FILE_TRACKED: " + str(relative))
        # 本检查器自身包含禁止字符串的匹配表达式。
        if path != Path(__file__).resolve() and forbidden.search(path.read_text()):
            raise ValueError("PRIVATE_PATH_OR_SECRET_PATTERN: " + str(relative))
    return {"status": "PASS", "scope": "PACKAGING_SYNTAX_SOURCE_HASHES_ONLY",
            "source_modules": len(actual), "release": manifest["release"]}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))

