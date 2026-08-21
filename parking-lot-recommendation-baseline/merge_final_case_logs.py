from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_logging_pipeline import (  # noqa: E402
    CANONICAL_RESULTS_HEADER,
    canonical_row_from_results,
    normalize_case_name,
)


MASTER_NAME = {
    "heuristics": "master_results_heuristic.csv",
    "aco": "master_results_aco.csv",
    "ga": "master_results_ga.csv",
    "pso": "master_results_pso.csv",
}


RESULT_SOURCES = {
    "heuristics": [
        ROOT / "HEURISTICS" / "data" / "final_cases" / "results_heuristic.csv",
    ],
    "aco": [
        ROOT / "ACO" / "data" / "final_cases" / "results_aco.csv",
        ROOT / "ACO" / "data" / "final_cases" / "results_aco_aco_n1500.csv",
        ROOT / "ACO" / "data" / "final_cases" / "results_aco_aco_n2000.csv",
        ROOT / "ACO" / "data" / "final_cases" / "results_aco_aco_n2500.csv",
    ],
    "ga": [
        ROOT / "GA" / "data" / "final_cases" / "results_ga.csv",
        ROOT / "GA" / "data" / "final_cases" / "results_ga_ga_n1500.csv",
        ROOT / "GA" / "data" / "final_cases" / "results_ga_ga_n2000.csv",
        ROOT / "GA" / "data" / "final_cases" / "results_ga_ga_n2500.csv",
    ],
    "pso": [
        ROOT / "PSO" / "data" / "final_cases" / "results_pso.csv",
        ROOT / "PSO" / "data" / "final_cases" / "results_pso_before_N2500_C4_resume.csv",
        ROOT / "PSO" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "results_pso.csv",
        ROOT / "PSO" / "data" / "final_cases" / "results_pso_pso_n1500.csv",
        ROOT / "PSO" / "data" / "final_cases" / "results_pso_pso_n2000.csv",
        ROOT / "PSO" / "data" / "final_cases" / "results_pso_pso_n2500.csv",
    ],
}


PRUNE_DIRS = {
    "heuristics": [ROOT / "HEURISTICS" / "data" / "final_cases"],
    "aco": [ROOT / "ACO" / "data" / "final_cases"],
    "ga": [ROOT / "GA" / "data" / "final_cases"],
    "pso": [ROOT / "PSO" / "data" / "final_cases"],
}


def _iter_direct_missing() -> list[Path]:
    direct_dir = ROOT / "PSO" / "data" / "final_cases" / "direct_missing"
    if not direct_dir.exists():
        return []
    return sorted(direct_dir.glob("*.csv"))


def _read_canonical_rows(method: str, path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if not any(str(v).strip() for v in raw.values()):
                continue
            row = canonical_row_from_results(method, raw)
            row["dat_file"] = normalize_case_name(row.get("dat_file", ""))
            if row["dat_file"]:
                rows.append({key: row.get(key, "") for key in CANONICAL_RESULTS_HEADER})
    return rows


def _all_sources(method: str) -> list[Path]:
    sources = list(RESULT_SOURCES[method])
    if method == "pso":
        sources.extend(_iter_direct_missing())
    return [path for path in sources if path.exists()]


def merge_method(method: str) -> tuple[Path, int]:
    if method == "heuristics":
        out_dir = ROOT / "HEURISTICS" / "data"
    elif method == "aco":
        out_dir = ROOT / "ACO" / "data"
    elif method == "ga":
        out_dir = ROOT / "GA" / "data"
    elif method == "pso":
        out_dir = ROOT / "PSO" / "data"
    else:
        raise ValueError(method)

    merged: dict[str, dict] = {}
    for path in _all_sources(method):
        for row in _read_canonical_rows(method, path):
            merged[row["dat_file"]] = row

    rows = sorted(merged.values(), key=lambda row: row["dat_file"])
    out_path = out_dir / MASTER_NAME[method]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return out_path, len(rows)


def prune_method(method: str) -> int:
    removed = 0
    master_name = MASTER_NAME[method]
    for base_dir in PRUNE_DIRS[method]:
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.rglob("*.csv")):
            if path.name == master_name:
                continue
            path.unlink()
            removed += 1
        for path in sorted(base_dir.rglob("*.pdf")):
            continue
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge final-case CSV logs down to one master file per method.")
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=["heuristics", "aco", "ga", "pso"],
        default=["heuristics", "aco", "ga", "pso"],
    )
    parser.add_argument("--prune-old", action="store_true")
    args = parser.parse_args()

    for method in args.methods:
        out_path, count = merge_method(method)
        print(f"{method}: wrote {count} rows to {out_path}")
        if args.prune_old:
            removed = prune_method(method)
            print(f"{method}: removed {removed} old CSV files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
