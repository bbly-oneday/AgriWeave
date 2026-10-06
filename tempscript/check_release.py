"""检查公开发布的语法、源码摘要和文件范围；不宣称算法或实车验收。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sqlite3
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
                "CONTRIBUTING.md", "docs/VALIDATION.md", "docs/LIMITATIONS.md",
                "data/README.md", "data/manifest.json"]
    for name in required:
        if not (ROOT / name).is_file():
            raise ValueError("MISSING_RELEASE_FILE: " + name)
    # 只放行维护者授权的350田原始输入；其它GPKG与outputs仍不公开。
    data_manifest = json.loads((ROOT / "data/manifest.json").read_text())
    approved_data = {entry["path"]: entry for entry in data_manifest["files"]}
    if set(approved_data) != {"data/fields2cover_350fields.gpkg"} or len(data_manifest["files"]) != 1:
        raise ValueError("UNAPPROVED_PUBLIC_DATA_FILE")
    for name, entry in approved_data.items():
        path = ROOT / name
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("PUBLIC_DATA_HASH_MISMATCH: " + name)
        if entry["layer"] != "fields" or entry["crs"] != "EPSG:4326":
            raise ValueError("PUBLIC_DATA_MANIFEST_SCHEMA_MISMATCH")
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("PUBLIC_DATA_SQLITE_INTEGRITY")
            count, unique = db.execute("SELECT COUNT(*),COUNT(DISTINCT field_id) FROM fields").fetchone()
            if count != entry["feature_count"] or unique != entry["unique_field_id_count"] or count != 350 or unique != 350:
                raise ValueError("PUBLIC_DATA_FIELD_COUNT")
            crs = db.execute("SELECT organization,organization_coordsys_id FROM gpkg_spatial_ref_sys "
                             "WHERE srs_id=(SELECT srs_id FROM gpkg_contents WHERE table_name='fields')").fetchone()
            if crs != ("EPSG", 4326):
                raise ValueError("PUBLIC_DATA_CRS")
    # 若有Git仓库，仅检查实际跟踪文件；outputs内的验证产物不属于公开内容。
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                             capture_output=True, check=False)
    if tracked.returncode == 0:
        files = [ROOT / p for p in tracked.stdout.decode().split("\0") if p]
    else:
        files = [p for p in ROOT.rglob("*") if p.is_file()
                 and not p.relative_to(ROOT).parts[0].startswith(".")
                 and p.relative_to(ROOT).parts[0] != "outputs"]
    forbidden = re.compile(r"/Users/|BaiduNetDisk|gh[pousr]_[A-Za-z0-9]{20,}|"
                           r"github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.as_posix() in approved_data:
            continue  # 指定二进制原始输入已用哈希、数据库、CRS和田块数检查。
        if (relative.parts[0] in {"outputs", ".venv"} or path.suffix in {".gpkg", ".pkl"}
                or (relative.parts[0] == "data" and relative.as_posix() not in {"data/README.md", "data/manifest.json"})):
            raise ValueError("PRIVATE_OR_GENERATED_FILE_TRACKED: " + str(relative))
        # 本检查器自身包含禁止字符串的匹配表达式。
        if path != Path(__file__).resolve() and forbidden.search(path.read_text()):
            raise ValueError("PRIVATE_PATH_OR_SECRET_PATTERN: " + str(relative))
    return {"status": "PASS", "scope": "PACKAGING_SYNTAX_SOURCE_HASHES_DATA_INTEGRITY_ONLY",
            "source_modules": len(actual), "release": manifest["release"],
            "public_data_files": len(approved_data), "public_data_fields": 350}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
