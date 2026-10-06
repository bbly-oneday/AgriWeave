"""Audit and explicitly publish a portable swath package to a new directory."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import route_planner as io
from audit_swath_result_gpkg import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output_root = Path(__file__).resolve().parents[1] / "outputs"
    if not args.out.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("SEAL_OUTPUT_MUST_BE_IN_V7_OUTPUTS")
    checked = audit(args.source / "swath_results.gpkg", args.source / "field_results.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = args.out.with_name(args.out.name + ".seal_audit.json")
    if report.exists():
        raise ValueError("SEAL_AUDIT_OUTPUT_MUST_BE_NEW")
    report.write_text(json.dumps(checked, ensure_ascii=False, indent=2))
    if not checked["passed"]:
        raise ValueError("SEAL_INDEPENDENT_COVERAGE_AUDIT_FAILED")
    print(json.dumps(io.seal_swath_bundle(args.source, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
