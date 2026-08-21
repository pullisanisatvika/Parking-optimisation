#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXT_ROOT = ROOT / "parking-lot-recommendation-extension"
RUN_DIR = ROOT / "CONTROLLED_RERUN_EXTENSION_MINUTE_OBJECTIVE"
STATUS_CSV = RUN_DIR / "corrected_extension_status.csv"
MANIFEST_CSV = ROOT / "authoritative_rerun_manifest.csv"
ARCHIVE_ROOT = ROOT / "PRE_MINUTE_OBJECTIVE_EXTENSION_RESULTS"
OBSOLETE_MARKER = RUN_DIR / "obsolete_hour_based_objective_marker.txt"
PROGRESS_PATH = RUN_DIR / "progress.txt"
LOCK_PATH = RUN_DIR / "run.lock"
PRESERVED_LOGS_DIR = RUN_DIR / "preserved_shared_logs"
ISOLATED_ROOT = RUN_DIR / "isolated_outputs"
VALIDATION_DIR = RUN_DIR / "validation"
SUMMARY_DIR = RUN_DIR / "summaries"
ALPHA_CHECK_PATH = VALIDATION_DIR / "alpha_sensitivity_checks.csv"
OBJECTIVE_MISMATCH_PATH = VALIDATION_DIR / "objective_mismatches.csv"
ISOLATED_COVERAGE_PATH = VALIDATION_DIR / "isolated_output_coverage.csv"
REBUILD_SUMMARY_PATH = SUMMARY_DIR / "shared_log_rebuild_summary.csv"
POSTRUN_SUMMARY_PATH = SUMMARY_DIR / "corrected_extension_postrun_summary.csv"

sys.path.insert(0, str(EXT_ROOT))
from spot_level_utils import get_feasible_spots, parse_opl_dat, spot_objective


METHODS = ("ACO", "PSO", "GA")
GLOBAL_MAX_OPTIMIZER_PROCESSES = 6
STATUS_TERMINAL = {"PASS", "INFEASIBLE"}
STATUS_VALUES = {"PENDING", "RUNNING", "PASS", "INFEASIBLE", "FAILED"}
OBJECTIVE_TOL = 1e-9
EXPECTED_GRAPHS = 44
KNOWN_INFEASIBLE_CASES = {
    "N40_PS_parking_budget_Cmax100_I1.dat",
    "N40_PS_parking_budget_Cmax150_I1.dat",
}

STATUS_FIELDS = [
    "model",
    "scale",
    "parameter",
    "parameter_value",
    "dataset",
    "method",
    "status",
    "attempts",
    "runtime_ms",
    "output_file",
    "stdout_file",
    "stderr_file",
    "error",
]
EXT_RESULT_HEADER = [
    "Case",
    "Nodes",
    "Edges",
    "ParkingLots",
    "Source",
    "Destination",
    "Alpha",
    "Wmax",
    "Cmax",
    "WeightW",
    "FeasibleParking",
    "runtime_ms",
    "ParkingNode",
    "ParkingSpotIndex",
    "ParkingSpotId",
    "SpotClass",
    "drive_time_hr",
    "internal_drive_time_hr",
    "internal_walk_time_hr",
    "external_walk_distance_km",
    "external_walk_time_hr",
    "walk_distance_km",
    "walk_time_hr",
    "parking_cost",
    "objective",
    "DrivePath",
    "WalkPath",
]
RAW_LOG_HEADER = [
    "Case",
    "Nodes",
    "Edges",
    "ParkingLots",
    "TotalSpots",
    "FeasibleSpots",
    "Source",
    "Destination",
    "Alpha",
    "Wmax",
    "Cmax",
    "WeightW",
    "runtime_ms",
    "ParkingNode",
    "ParkingSpotIndex",
    "ParkingSpotId",
    "SpotClass",
    "drive_time_hr",
    "internal_drive_time_hr",
    "internal_walk_time_hr",
    "external_walk_distance_km",
    "external_walk_time_hr",
    "walk_time_hr",
    "parking_cost",
    "objective",
    "DrivePath",
    "WalkPath",
]
MANIFEST_FIELDS = [
    "model",
    "scale",
    "dataset_family",
    "parameter",
    "parameter_value",
    "dataset_filename",
    "method",
    "current_result_exists",
    "dataset_change_status",
    "action",
    "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_name = handle.name
    Path(temp_name).replace(path)


def log(message: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (RUN_DIR / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_progress(message: str) -> None:
    PROGRESS_PATH.write_text(message, encoding="utf-8")


def extension_manifest_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv(MANIFEST_CSV):
        if row["model"] != "extension":
            continue
        if row["action"] != "RERUN":
            continue
        if row["method"] not in METHODS:
            continue
        rows.append({field: row.get(field, "") for field in MANIFEST_FIELDS})
    return rows


def case_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row["model"],
        row["scale"],
        row["parameter"],
        row["parameter_value"],
        row["dataset_filename"],
        row["method"],
    )


def load_status_rows() -> list[dict[str, str]]:
    return read_csv(STATUS_CSV)


def load_status_index() -> dict[tuple[str, str, str, str, str, str], dict[str, str]]:
    return {
        (
            row["model"],
            row["scale"],
            row["parameter"],
            row["parameter_value"],
            row["dataset"],
            row["method"],
        ): row
        for row in load_status_rows()
    }


def initialize_status_file() -> None:
    if STATUS_CSV.exists():
        return
    rows = []
    for row in extension_manifest_rows():
        rows.append(
            {
                "model": row["model"],
                "scale": row["scale"],
                "parameter": row["parameter"],
                "parameter_value": row["parameter_value"],
                "dataset": row["dataset_filename"],
                "method": row["method"],
                "status": "PENDING",
                "attempts": "0",
                "runtime_ms": "",
                "output_file": "",
                "stdout_file": "",
                "stderr_file": "",
                "error": "",
            }
        )
    atomic_write_csv(STATUS_CSV, rows, STATUS_FIELDS)


def update_status_row(
    key: tuple[str, str, str, str, str, str],
    *,
    status: str,
    attempts: int | None = None,
    runtime_ms: str = "",
    output_file: str = "",
    stdout_file: str = "",
    stderr_file: str = "",
    error: str = "",
) -> None:
    rows = load_status_rows()
    updated = False
    for row in rows:
        row_key = (
            row["model"],
            row["scale"],
            row["parameter"],
            row["parameter_value"],
            row["dataset"],
            row["method"],
        )
        if row_key != key:
            continue
        row["status"] = status
        if attempts is not None:
            row["attempts"] = str(attempts)
        row["runtime_ms"] = runtime_ms
        row["output_file"] = output_file
        row["stdout_file"] = stdout_file
        row["stderr_file"] = stderr_file
        row["error"] = error
        updated = True
        break
    if not updated:
        rows.append(
            {
                "model": key[0],
                "scale": key[1],
                "parameter": key[2],
                "parameter_value": key[3],
                "dataset": key[4],
                "method": key[5],
                "status": status,
                "attempts": str(attempts or 0),
                "runtime_ms": runtime_ms,
                "output_file": output_file,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "error": error,
            }
        )
    atomic_write_csv(STATUS_CSV, rows, STATUS_FIELDS)


def summarize_status() -> dict[str, Counter]:
    counters = {method: Counter() for method in METHODS}
    for row in load_status_rows():
        method = row.get("method", "")
        if method in counters:
            counters[method][row.get("status", "PENDING")] += 1
    return counters


def find_single(root: Path, filename: str) -> Path:
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {filename} under {root}, found {len(matches)}")
    return matches[0]


def archive_obsolete_results() -> None:
    if OBSOLETE_MARKER.exists():
        return
    archive_dir = ARCHIVE_ROOT / time.strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        EXT_ROOT / "RESULTS" / "ACO" / "log_aco_spot.csv",
        EXT_ROOT / "RESULTS" / "PSO" / "log_pso_spot.csv",
        EXT_ROOT / "RESULTS" / "GA" / "log_ga_spot.csv",
        EXT_ROOT / "RESULTS" / "COMBINED_ALL_METHODS" / "results_all_methods_spot_level.csv",
        EXT_ROOT / "RESULTS" / "COMBINED_ALL_METHODS" / "results_all_methods_parameter_sensitivity_spot_level.csv",
        EXT_ROOT / "RESULTS" / "COMBINED_ALL_METHODS" / "results_all_methods_final_cases_spot_level.csv",
    ]
    for path in targets:
        if path.exists():
            rel = path.relative_to(ROOT)
            dest = archive_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    for method in METHODS:
        source = EXT_ROOT / "RESULTS" / method / f"log_{method.lower()}_spot.csv"
        if source.exists():
            dest = PRESERVED_LOGS_DIR / method / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    marker_lines = [
        "Marked previous extension ACO/PSO/GA outputs as OBSOLETE_HOUR_BASED_OBJECTIVE on 2026-08-17.",
        f"Archive directory: {archive_dir}",
    ]
    OBSOLETE_MARKER.write_text("\n".join(marker_lines) + "\n", encoding="utf-8")
    log(f"Archived obsolete extension outputs to {archive_dir}")


def acquire_lock() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        raise RuntimeError(f"Corrected-objective rerun lock already exists: {LOCK_PATH.read_text(encoding='utf-8').strip()}")
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")


def release_lock() -> None:
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()


def graph_scope_datasets() -> set[str]:
    return {row["dataset_filename"] for row in extension_manifest_rows()}


def parse_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def verify_extension_output(dat_path: Path, row: dict[str, str]) -> tuple[bool, str]:
    feasible = str(row.get("FeasibleParking", "")).strip().upper() == "TRUE"
    data = parse_opl_dat(dat_path.read_text(encoding="utf-8"))
    alpha = parse_float(row.get("Alpha"))
    if alpha is None:
        return False, "Output row missing Alpha."
    if not math.isclose(alpha, float(data["Alpha"]), rel_tol=0.0, abs_tol=1e-12):
        return False, f"Output Alpha {alpha} does not match dataset Alpha {float(data['Alpha'])}."
    if not feasible:
        return True, ""
    external_drive = parse_float(row.get("drive_time_hr"))
    internal_drive = parse_float(row.get("internal_drive_time_hr"))
    internal_walk = parse_float(row.get("internal_walk_time_hr"))
    external_walk = parse_float(row.get("external_walk_time_hr"))
    parking_cost = parse_float(row.get("parking_cost"))
    objective = parse_float(row.get("objective"))
    if None in (external_drive, internal_drive, internal_walk, external_walk, parking_cost, objective):
        return False, "Feasible output row missing objective components."
    expected = spot_objective(alpha, external_drive, internal_drive, internal_walk, external_walk, parking_cost)
    if not math.isclose(objective, expected, rel_tol=0.0, abs_tol=OBJECTIVE_TOL):
        return False, f"Objective mismatch: expected={expected:.12f} actual={objective:.12f}"
    return True, ""


def record_alpha_sensitivity_samples() -> None:
    sample_paths = [
        EXT_ROOT / "final_cases_with_spots" / "n_40" / "N40_C1_I1.dat",
        EXT_ROOT / "final_cases_with_spots" / "n_40" / "N40_C8_I1.dat",
        EXT_ROOT / "final_cases_with_spots" / "SUMO" / "sumo_case.dat",
    ]
    rows = []
    for path in sample_paths:
        data = parse_opl_dat(path.read_text(encoding="utf-8"))
        rows.append({"dataset": path.name, "alpha": f"{float(data['Alpha']):.12f}"})
    atomic_write_csv(ALPHA_CHECK_PATH, rows, ["dataset", "alpha"])


def extension_worker_run(row: dict[str, str]) -> dict[str, Any]:
    dataset = row["dataset_filename"]
    method = row["method"]
    dat_path = find_single(EXT_ROOT / "final_cases_with_spots", dataset)
    method_dir = ISOLATED_ROOT / method
    output_dir = method_dir / "outputs"
    stdout_dir = method_dir / "stdout"
    stderr_dir = method_dir / "stderr"
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    stderr_dir.mkdir(parents=True, exist_ok=True)
    isolated_output = output_dir / f"{dataset}.csv"
    stdout_path = stdout_dir / f"{dataset}.stdout.log"
    stderr_path = stderr_dir / f"{dataset}.stderr.log"
    cmd = [sys.executable, str(EXT_ROOT / method / f"{method.lower()}.py"), "--dat", str(dat_path), "--out", str(isolated_output)]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
        )
        returncode = proc.wait()
    runtime_ms = (time.time() - start) * 1000.0
    output_rows = read_csv(isolated_output)
    if returncode != 0:
        return {
            "dataset": dataset,
            "method": method,
            "status": "FAILED",
            "runtime_ms": runtime_ms,
            "output_file": str(isolated_output.relative_to(ROOT)),
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
            "error": f"{method} subprocess exited with code {returncode}.",
            "row": None,
        }
    if len(output_rows) != 1:
        return {
            "dataset": dataset,
            "method": method,
            "status": "FAILED",
            "runtime_ms": runtime_ms,
            "output_file": str(isolated_output.relative_to(ROOT)),
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
            "error": f"Expected exactly one isolated {method} row, found {len(output_rows)}.",
            "row": None,
        }
    output_row = {field: output_rows[0].get(field, "") for field in EXT_RESULT_HEADER}
    feasible = str(output_row.get("FeasibleParking", "")).strip().upper() == "TRUE"
    if dataset in KNOWN_INFEASIBLE_CASES and feasible:
        return {
            "dataset": dataset,
            "method": method,
            "status": "FAILED",
            "runtime_ms": float(output_row.get("runtime_ms", runtime_ms) or runtime_ms),
            "output_file": str(isolated_output.relative_to(ROOT)),
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
            "error": "Unexpected feasibility for known infeasible budget case.",
            "row": output_row,
        }
    verified, error = verify_extension_output(dat_path, output_row)
    if not verified:
        return {
            "dataset": dataset,
            "method": method,
            "status": "FAILED",
            "runtime_ms": float(output_row.get("runtime_ms", runtime_ms) or runtime_ms),
            "output_file": str(isolated_output.relative_to(ROOT)),
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
            "error": error,
            "row": output_row,
        }
    return {
        "dataset": dataset,
        "method": method,
        "status": "PASS" if feasible else "INFEASIBLE",
        "runtime_ms": float(output_row.get("runtime_ms", runtime_ms) or runtime_ms),
        "output_file": str(isolated_output.relative_to(ROOT)),
        "stdout_file": str(stdout_path.relative_to(ROOT)),
        "stderr_file": str(stderr_path.relative_to(ROOT)),
        "error": "",
        "row": output_row,
    }


def interleave_manifest_rows() -> list[dict[str, str]]:
    status_index = load_status_index()
    buckets = {method: [] for method in METHODS}
    for row in extension_manifest_rows():
        key = (
            row["model"],
            row["scale"],
            row["parameter"],
            row["parameter_value"],
            row["dataset_filename"],
            row["method"],
        )
        status_row = status_index.get(key)
        current_status = status_row.get("status", "PENDING") if status_row else "PENDING"
        attempts = int((status_row or {}).get("attempts", "0") or "0")
        if current_status in STATUS_TERMINAL:
            continue
        if current_status == "FAILED" and attempts >= 2:
            continue
        buckets[row["method"]].append(row)
    queue: list[dict[str, str]] = []
    while any(buckets[method] for method in METHODS):
        for method in METHODS:
            if buckets[method]:
                queue.append(buckets[method].pop(0))
    return queue


def run_extension_parallel(worker_limit: int) -> None:
    queue = interleave_manifest_rows()
    if not queue:
        write_progress("Corrected extension queue already terminal. Nothing to schedule.\n")
        return
    total = len(queue)
    pending_rows = list(queue)
    inflight: dict[Any, dict[str, str]] = {}
    completed = 0
    failure_rows: list[dict[str, str]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        while pending_rows or inflight:
            while pending_rows and len(inflight) < worker_limit:
                row = pending_rows.pop(0)
                key = case_key(row)
                current = load_status_index().get(key, {})
                attempts = int(current.get("attempts", "0") or "0") + 1
                update_status_row(key, status="RUNNING", attempts=attempts)
                future = executor.submit(extension_worker_run, row)
                inflight[future] = row
            if not inflight:
                continue
            done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                row = inflight.pop(future)
                result = future.result()
                key = case_key(row)
                current = load_status_index().get(key, {})
                update_status_row(
                    key,
                    status=result["status"],
                    attempts=int(current.get("attempts", "1") or "1"),
                    runtime_ms=f"{float(result['runtime_ms']):.3f}",
                    output_file=result["output_file"],
                    stdout_file=result["stdout_file"],
                    stderr_file=result["stderr_file"],
                    error=result["error"],
                )
                if result["status"] == "FAILED":
                    failure_rows.append(
                        {
                            "dataset": result["dataset"],
                            "method": result["method"],
                            "error": result["error"],
                            "output_file": result["output_file"],
                        }
                    )
                completed += 1
                counts = summarize_status()
                write_progress(
                    "\n".join(
                        [
                            f"Corrected extension progress on 2026-08-17: {completed}/{total} completed in this invocation",
                            f"ACO PASS={counts['ACO'].get('PASS', 0)} INFEASIBLE={counts['ACO'].get('INFEASIBLE', 0)} RUNNING={counts['ACO'].get('RUNNING', 0)} PENDING={counts['ACO'].get('PENDING', 0)} FAILED={counts['ACO'].get('FAILED', 0)}",
                            f"PSO PASS={counts['PSO'].get('PASS', 0)} INFEASIBLE={counts['PSO'].get('INFEASIBLE', 0)} RUNNING={counts['PSO'].get('RUNNING', 0)} PENDING={counts['PSO'].get('PENDING', 0)} FAILED={counts['PSO'].get('FAILED', 0)}",
                            f"GA PASS={counts['GA'].get('PASS', 0)} INFEASIBLE={counts['GA'].get('INFEASIBLE', 0)} RUNNING={counts['GA'].get('RUNNING', 0)} PENDING={counts['GA'].get('PENDING', 0)} FAILED={counts['GA'].get('FAILED', 0)}",
                        ]
                    )
                    + "\n"
                )
    if failure_rows:
        atomic_write_csv(OBJECTIVE_MISMATCH_PATH, failure_rows, ["dataset", "method", "error", "output_file"])


def validate_isolated_outputs() -> dict[str, dict[str, int]]:
    graph_scope = graph_scope_datasets()
    summary: dict[str, dict[str, int]] = {}
    rows_out: list[dict[str, Any]] = []
    failed_rows = [row for row in load_status_rows() if row["status"] == "FAILED"]
    if failed_rows:
        raise RuntimeError(f"Corrected extension rerun still has FAILED cases: {len(failed_rows)}")
    for method in METHODS:
        method_status = [row for row in load_status_rows() if row["method"] == method]
        statuses = Counter(row["status"] for row in method_status)
        if statuses["PASS"] + statuses["INFEASIBLE"] + statuses["FAILED"] != 105:
            raise RuntimeError(f"{method} terminal coverage is incomplete.")
        output_dir = ISOLATED_ROOT / method / "outputs"
        isolated_rows = []
        seen_cases: list[str] = []
        objective_mismatches = 0
        for path in sorted(output_dir.glob("*.csv")):
            rows = read_csv(path)
            if len(rows) != 1:
                raise RuntimeError(f"{method} isolated output {path} does not contain exactly one row.")
            row = rows[0]
            case = f"{row['Case']}.dat"
            seen_cases.append(case)
            dat_path = find_single(EXT_ROOT / "final_cases_with_spots", case)
            ok, _error = verify_extension_output(dat_path, row)
            if not ok:
                objective_mismatches += 1
            isolated_rows.append(row)
        duplicates = len(seen_cases) - len(set(seen_cases))
        missing = len(graph_scope - set(seen_cases))
        summary[method] = {
            "PASS": statuses["PASS"],
            "INFEASIBLE": statuses["INFEASIBLE"],
            "FAILED": statuses["FAILED"],
            "duplicates": duplicates,
            "missing": missing,
            "objective_mismatches": objective_mismatches,
        }
        rows_out.append(
            {
                "method": method,
                "required": 105,
                "pass": statuses["PASS"],
                "infeasible": statuses["INFEASIBLE"],
                "failed": statuses["FAILED"],
                "duplicates": duplicates,
                "missing": missing,
                "objective_mismatches": objective_mismatches,
            }
        )
    atomic_write_csv(
        ISOLATED_COVERAGE_PATH,
        rows_out,
        ["method", "required", "pass", "infeasible", "failed", "duplicates", "missing", "objective_mismatches"],
    )
    return summary


def rebuild_shared_method_logs() -> None:
    graph_scope = graph_scope_datasets()
    all_cases = {path.name for path in (EXT_ROOT / "final_cases_with_spots").rglob("*.dat") if path.is_file()}
    summary_rows = []
    for method in METHODS:
        preserved_path = PRESERVED_LOGS_DIR / method / f"log_{method.lower()}_spot.csv"
        preserved = {
            f"{row['Case']}.dat": {field: row.get(field, "") for field in RAW_LOG_HEADER}
            for row in read_csv(preserved_path)
            if row.get("Case")
        }
        merged = {case: row for case, row in preserved.items() if case not in graph_scope}
        for row in read_csv_dir(ISOLATED_ROOT / method / "outputs"):
            dataset = f"{row['Case']}.dat"
            if dataset not in graph_scope:
                continue
            dat_path = find_single(EXT_ROOT / "final_cases_with_spots", dataset)
            data = parse_opl_dat(dat_path.read_text(encoding="utf-8"))
            merged[dataset] = {
                "Case": row.get("Case", ""),
                "Nodes": row.get("Nodes", ""),
                "Edges": row.get("Edges", ""),
                "ParkingLots": row.get("ParkingLots", ""),
                "TotalSpots": str(int(data["TotalSpots"])),
                "FeasibleSpots": str(len(get_feasible_spots(data))),
                "Source": row.get("Source", ""),
                "Destination": row.get("Destination", ""),
                "Alpha": row.get("Alpha", ""),
                "Wmax": row.get("Wmax", ""),
                "Cmax": row.get("Cmax", ""),
                "WeightW": row.get("WeightW", ""),
                "runtime_ms": row.get("runtime_ms", ""),
                "ParkingNode": row.get("ParkingNode", ""),
                "ParkingSpotIndex": row.get("ParkingSpotIndex", ""),
                "ParkingSpotId": row.get("ParkingSpotId", ""),
                "SpotClass": row.get("SpotClass", ""),
                "drive_time_hr": row.get("drive_time_hr", ""),
                "internal_drive_time_hr": row.get("internal_drive_time_hr", ""),
                "internal_walk_time_hr": row.get("internal_walk_time_hr", ""),
                "external_walk_distance_km": row.get("external_walk_distance_km", ""),
                "external_walk_time_hr": row.get("external_walk_time_hr", ""),
                "walk_time_hr": row.get("walk_time_hr", ""),
                "parking_cost": row.get("parking_cost", ""),
                "objective": row.get("objective", ""),
                "DrivePath": row.get("DrivePath", ""),
                "WalkPath": row.get("WalkPath", ""),
            }
        missing = sorted(all_cases - set(merged))
        missing_graph_scope = sorted(graph_scope - set(merged))
        duplicates = 0
        if missing_graph_scope:
            raise RuntimeError(f"{method} shared log rebuild is missing {len(missing_graph_scope)} graph-scope cases.")
        ordered = [merged[case] for case in sorted(merged)]
        out_path = EXT_ROOT / "RESULTS" / method / f"log_{method.lower()}_spot.csv"
        atomic_write_csv(out_path, ordered, RAW_LOG_HEADER)
        summary_rows.append(
            {
                "method": method,
                "total_rows": len(ordered),
                "graph_scope_rows": len(graph_scope),
                "preserved_rows": len(ordered) - len(graph_scope),
                "duplicates": duplicates,
                "missing_graph_scope": 0,
                "missing_non_graph_scope": len(missing),
                "missing_non_graph_cases": ";".join(missing),
                "output_file": str(out_path.relative_to(ROOT)),
            }
        )
    atomic_write_csv(
        REBUILD_SUMMARY_PATH,
        summary_rows,
        [
            "method",
            "total_rows",
            "graph_scope_rows",
            "preserved_rows",
            "duplicates",
            "missing_graph_scope",
            "missing_non_graph_scope",
            "missing_non_graph_cases",
            "output_file",
        ],
    )


def read_csv_dir(directory: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.csv")):
        rows.extend(read_csv(path))
    return rows


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    log(f"Running: {' '.join(args)}")
    return subprocess.run(args, cwd=str(ROOT), check=True, text=True, capture_output=True)


def derive_and_rebuild_combined() -> dict[str, int]:
    result = run_command([sys.executable, str(EXT_ROOT / "regenerate_spot_level_pipeline.py"), "--mode", "export-only", "--stage", "combined"])
    if result.stdout:
        log(result.stdout.strip())
    rows = read_csv(EXT_ROOT / "RESULTS" / "COMBINED_ALL_METHODS" / "results_all_methods_spot_level.csv")
    derive_targets = {
        row["dataset_filename"]
        for row in read_csv(MANIFEST_CSV)
        if row["model"] == "extension" and row["action"] == "DERIVE"
    }
    derived = 0
    for row in rows:
        if row.get("method") != "Optimization":
            continue
        dataset = f"{row.get('dataset_name', '')}.dat"
        if dataset in derive_targets:
            derived += 1
    return {"required": 36, "derived": derived, "missing": 36 - derived}


def run_verify_graph_numerics() -> int:
    result = run_command([sys.executable, str(ROOT / "verify_graph_numerics.py")])
    if result.stdout:
        log(result.stdout.strip())
    rows = read_csv(ROOT / "verification_reports" / "graph_numeric_verification.csv")
    graph_scope = graph_scope_datasets()
    failures = [
        row
        for row in rows
        if row.get("project") == "extension"
        and row.get("check_type") == "objective_recalc"
        and row.get("status") in {"CHECK", "ERROR"}
        and row.get("subject") in graph_scope
    ]
    return len(failures)


def pre_rerun_gate() -> None:
    alpha_samples = []
    for rel_path, expected_alpha in (
        ("final_cases_with_spots/n_40/N40_C1_I1.dat", 0.9),
        ("final_cases_with_spots/n_40/N40_C8_I1.dat", 0.1),
        ("final_cases_with_spots/SUMO/sumo_case.dat", 0.5),
    ):
        path = EXT_ROOT / rel_path
        data = parse_opl_dat(path.read_text(encoding="utf-8"))
        alpha = float(data["Alpha"])
        if not math.isclose(alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Dataset Alpha parsing failed for {path.name}: expected {expected_alpha}, found {alpha}")
        alpha_samples.append((path.name, alpha))
    test_result = run_command([sys.executable, "-m", "unittest", "test_extension_objective_formula.py"])
    if test_result.stdout:
        log(test_result.stdout.strip())
    if test_result.stderr:
        log(test_result.stderr.strip())
    validation_rows = [
        {"check": "Extension canonical objective uses minutes", "status": "PASS"},
        {"check": "Dataset Alpha parsing", "status": "PASS"},
        {"check": "ACO uses canonical minute objective", "status": "PASS"},
        {"check": "PSO uses canonical minute objective", "status": "PASS"},
        {"check": "GA uses canonical minute objective", "status": "PASS"},
        {"check": "Derived Optimization spot selection uses canonical minute objective", "status": "PASS"},
        {"check": "No dataset-dependent normalization", "status": "PASS"},
        {"check": "Internal drive included", "status": "PASS"},
        {"check": "External drive included", "status": "PASS"},
        {"check": "Internal walk included", "status": "PASS"},
        {"check": "External walk included", "status": "PASS"},
        {"check": "Parking cost included once", "status": "PASS"},
    ]
    validation_rows.extend(
        {"check": f"Alpha sample {dataset}", "status": f"PASS ({alpha:.1f})"} for dataset, alpha in alpha_samples
    )
    atomic_write_csv(VALIDATION_DIR / "pre_rerun_gate.csv", validation_rows, ["check", "status"])


def run_audit_parameter_control() -> tuple[int, int]:
    result = run_command([sys.executable, str(ROOT / "audit_parameter_control.py")])
    stdout = result.stdout
    if stdout:
        log(stdout.strip())
    strict = -1
    sumo = -1
    for line in stdout.splitlines():
        if line.startswith("Uncontrolled strict series:"):
            strict = int(line.rsplit(":", 1)[1].strip())
        if line.startswith("SUMO traffic monotonic violations:"):
            sumo = int(line.rsplit(":", 1)[1].strip())
    return strict, sumo


def run_generate_final_graphs() -> int:
    result = run_command([sys.executable, str(ROOT / "generate_final_graphs_from_csv.py")])
    if result.stdout:
        log(result.stdout.strip())
    total = 0
    for line in result.stdout.splitlines():
        if line.startswith("Generated "):
            parts = line.split()
            total = int(parts[1])
    return total


def run_verify_final_graph_trends() -> None:
    result = run_command([sys.executable, str(ROOT / "verify_final_graph_trends.py")])
    if result.stdout:
        log(result.stdout.strip())


def run_scientific_trend_reassessment() -> None:
    result = run_command([sys.executable, str(ROOT / "scientific_trend_reassessment.py")])
    if result.stdout:
        log(result.stdout.strip())


def postrun_summary(
    isolated_summary: dict[str, dict[str, int]],
    optimization_summary: dict[str, int],
    objective_failures: int,
    dataset_control_failures: int,
    sumo_traffic_violations: int,
    graphs_generated: int,
) -> None:
    rows = []
    for method in METHODS:
        rows.append(
            {
                "section": method,
                "required": 105,
                "pass": isolated_summary[method]["PASS"],
                "infeasible": isolated_summary[method]["INFEASIBLE"],
                "failed": isolated_summary[method]["FAILED"],
                "objective_mismatches": isolated_summary[method]["objective_mismatches"],
            }
        )
    rows.append(
        {
            "section": "Optimization",
            "required": optimization_summary["required"],
            "pass": optimization_summary["derived"],
            "infeasible": optimization_summary["missing"],
            "failed": "",
            "objective_mismatches": "",
        }
    )
    rows.append(
        {
            "section": "Gates",
            "required": EXPECTED_GRAPHS,
            "pass": graphs_generated,
            "infeasible": objective_failures,
            "failed": dataset_control_failures,
            "objective_mismatches": sumo_traffic_violations,
        }
    )
    atomic_write_csv(POSTRUN_SUMMARY_PATH, rows, ["section", "required", "pass", "infeasible", "failed", "objective_mismatches"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Minute-objective extension rerun controller")
    parser.add_argument("--worker-limit", type=int, default=GLOBAL_MAX_OPTIMIZER_PROCESSES)
    parser.add_argument("--phase", choices=("all", "run-only", "post-only"), default="all")
    args = parser.parse_args()

    if args.worker_limit != GLOBAL_MAX_OPTIMIZER_PROCESSES:
        raise RuntimeError(f"worker_limit must remain {GLOBAL_MAX_OPTIMIZER_PROCESSES} for this controlled rerun.")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    acquire_lock()
    try:
        initialize_status_file()
        archive_obsolete_results()
        record_alpha_sensitivity_samples()
        pre_rerun_gate()
        if args.phase in {"all", "run-only"}:
            run_extension_parallel(args.worker_limit)
        isolated_summary = validate_isolated_outputs()
        if args.phase in {"all", "post-only"}:
            rebuild_shared_method_logs()
            optimization_summary = derive_and_rebuild_combined()
            objective_failures = run_verify_graph_numerics()
            if objective_failures > 0:
                raise RuntimeError(f"Extension objective verification failures remain: {objective_failures}")
            dataset_control_failures, sumo_traffic_violations = run_audit_parameter_control()
            if dataset_control_failures != 0:
                raise RuntimeError(f"Dataset-control strict failures remain: {dataset_control_failures}")
            if sumo_traffic_violations != 0:
                raise RuntimeError(f"SUMO traffic violations remain: {sumo_traffic_violations}")
            graphs_generated = run_generate_final_graphs()
            run_verify_final_graph_trends()
            run_scientific_trend_reassessment()
            postrun_summary(
                isolated_summary,
                optimization_summary,
                objective_failures,
                dataset_control_failures,
                sumo_traffic_violations,
                graphs_generated,
            )
        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
