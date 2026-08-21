from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data_logging_pipeline import CANONICAL_RESULTS_HEADER, normalize_results_csv


ROOT = Path(__file__).resolve().parent

METHODS = {
    "heuristics": ROOT / "HEURISTICS" / "data",
    "aco": ROOT / "ACO" / "data",
    "pso": ROOT / "PSO" / "data",
    "ga": ROOT / "GA" / "data",
}

MASTER_NAME = {
    "heuristics": "master_results_heuristic.csv",
    "aco": "master_results_aco.csv",
    "pso": "master_results_pso.csv",
    "ga": "master_results_ga.csv",
}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_result_logs(method: str, data_dir: Path) -> list[Path]:
    master_path = data_dir / MASTER_NAME[method]
    paths: list[Path] = []
    for path in sorted(data_dir.rglob("*.csv")):
        name = path.name.lower()
        rel = path.relative_to(data_dir)
        rel_text = str(rel).lower()
        if path == master_path:
            continue
        if "backup" in rel_text or "backups" in rel_text:
            continue
        if "before_" in name or "resume" in name:
            continue
        if not name.startswith("results_"):
            continue
        paths.append(path)
    return paths


def consolidate_method(method: str, prune_old: bool = False) -> tuple[Path, int, int]:
    data_dir = METHODS[method]
    master_path = data_dir / MASTER_NAME[method]
    source_logs = _candidate_result_logs(method, data_dir)

    unique: dict[tuple[str, ...], dict] = {}
    for path in source_logs:
        normalize_results_csv(method, path)
        for row in _read_rows(path):
            canonical = {key: row.get(key, "") for key in CANONICAL_RESULTS_HEADER}
            canonical["method"] = canonical.get("method") or method
            key = tuple(canonical.get(col, "") for col in CANONICAL_RESULTS_HEADER)
            unique[key] = canonical

    rows = sorted(
        unique.values(),
        key=lambda row: (
            row.get("dat_file", ""),
            row.get("method", ""),
            row.get("runtime_ms", ""),
            row.get("parking_node", ""),
        ),
    )
    _write_rows(master_path, rows, CANONICAL_RESULTS_HEADER)

    removed = 0
    if prune_old:
        for path in source_logs:
            if path.exists():
                path.unlink()
                removed += 1
        for path in sorted(data_dir.rglob("routes_*.csv")):
            rel_text = str(path.relative_to(data_dir)).lower()
            if "backup" in rel_text or "backups" in rel_text:
                continue
            path.unlink()
            removed += 1

    return master_path, len(rows), removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Collapse method logs to one master CSV per method.")
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=list(METHODS.keys()),
        default=list(METHODS.keys()),
        help="Methods to consolidate.",
    )
    parser.add_argument(
        "--prune-old",
        action="store_true",
        help="Delete duplicate results_*.csv and routes_*.csv files after creating master logs.",
    )
    args = parser.parse_args()

    for method in args.methods:
        master_path, row_count, removed = consolidate_method(method, prune_old=args.prune_old)
        print(f"{method}: wrote {row_count} rows to {master_path}")
        if args.prune_old:
            print(f"{method}: removed {removed} old CSV logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
