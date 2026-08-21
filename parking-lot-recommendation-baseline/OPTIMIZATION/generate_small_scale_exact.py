#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HEURISTICS.heuristics import parse_opl_dat


DATASET_ROOT = ROOT / "final_cases"
OUTPUT_CSV = ROOT / "OPTIMIZATION" / "optimisation_results_summary_exact_small.csv"
VERIFICATION_CSV = ROOT / "RESULTS" / "VALIDATION" / "exact_vs_opl_small.csv"
HISTORICAL_OPL_CSV = ROOT / "OPTIMIZATION" / "optimisation_results_summary_solved_small.csv"
MANIFEST_CSV = ROOT / "RESULTS" / "VALIDATION" / "parameter_sensitivity_manifest.csv"

TARGET_NODE_SCALES = {10, 20, 30, 40, 50}
TOL = 1e-9
CSV_COMPARE_TOL = 1e-4

HISTORICAL_RUNTIME_OVERRIDES = {
    "N30_C1_I1.dat": {
        "runtime_ms": "00:09:18:46",
        "solver_status": "runtime_only_user_supplied",
        "feasible": "",
    },
    "N40_C1_I1.dat": {
        "runtime_ms": "51:52:15:79",
        "solver_status": "runtime_only_user_supplied",
        "feasible": "",
    },
    "N50_C1_I1.dat": {
        "runtime_ms": "69:37:18:72",
        "solver_status": "crashed_memory_limit_user_supplied",
        "feasible": "",
    },
}

CASE_METADATA = {
    "C1": ("Baseline", "baseline", "Baseline"),
    "C2": ("Dense", "edge_density", "Dense"),
    "C3": ("Higher supply", "parking_supply", "Higher"),
    "C4": ("1.00 km", "maximum_walking_distance", "1.0"),
    "C5": ("Medium traffic", "traffic_level", "Medium"),
    "C6": ("High traffic", "traffic_level", "High"),
    "C7": ("Alpha 0.80", "objective_weight_alpha", "0.8"),
    "C8": ("Alpha 0.20", "objective_weight_alpha", "0.2"),
    "C9": ("Cmax 15.0", "parking_budget_cmax", "15.0"),
    "C10": ("15.0-30.0", "parking_cost_variation_mode", "15.0-30.0"),
}

RESULT_HEADER = [
    "scenario_id",
    "scenario_name",
    "parameter_name",
    "parameter_value",
    "method",
    "dat_file",
    "nodes",
    "edges",
    "parking_lots",
    "source",
    "destination",
    "S",
    "D",
    "Alpha",
    "Wmax",
    "Cmax",
    "w",
    "feasible",
    "solver_status",
    "parking_node",
    "driving_path",
    "walking_edge",
    "full_path",
    "selected_edges",
    "drive_time_hr",
    "walk_distance_km",
    "walk_time_hr",
    "travel_time_hr",
    "parking_cost",
    "objective",
    "path_reconstruction_status",
    "runtime_ms",
    "drive_path",
    "walk_path",
    "case_name",
    "OptimizationSource",
    "HistoricalOPLRuntime",
    "HistoricalOPLStatus",
    "OPLVerificationStatus",
]

VERIFICATION_HEADER = [
    "dat_file",
    "historical_solver_status",
    "historical_feasible",
    "historical_runtime_ms",
    "python_solver_status",
    "python_feasible",
    "python_runtime_ms",
    "historical_parking_node",
    "python_parking_node",
    "historical_drive_time_hr",
    "python_drive_time_hr",
    "historical_walk_time_hr",
    "python_walk_time_hr",
    "historical_parking_cost",
    "python_parking_cost",
    "historical_objective",
    "python_objective",
    "verification_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merged_historical_rows() -> dict[str, dict[str, str]]:
    rows = {row["dat_file"]: row for row in read_csv(HISTORICAL_OPL_CSV)}
    for dat_file, override in HISTORICAL_RUNTIME_OVERRIDES.items():
        merged = dict(rows.get(dat_file, {}))
        merged["dat_file"] = dat_file
        merged.update(override)
        rows[dat_file] = merged
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_existing_rows(
    path: Path,
    incoming_rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
    key_field: str = "dat_file",
) -> list[dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        key = str(row.get(key_field, "")).strip()
        if key:
            existing[key] = {field: row.get(field, "") for field in fieldnames}
    for row in incoming_rows:
        key = str(row.get(key_field, "")).strip()
        if key:
            existing[key] = {field: row.get(field, "") for field in fieldnames}
    return sorted(existing.values(), key=lambda row: str(row.get(key_field, "")))


def case_sort_key(path: Path) -> tuple[int, int]:
    stem = path.stem
    node_count = int(stem.split("_", 1)[0][1:])
    case_id = int(stem.split("_C", 1)[1].split("_", 1)[0])
    return node_count, case_id


def small_scale_case_paths() -> list[Path]:
    paths: list[Path] = []
    for subdir in sorted(DATASET_ROOT.glob("n_*")):
        for path in sorted(subdir.glob("N*_C*_I*.dat")):
            node_count = int(path.stem.split("_", 1)[0][1:])
            if node_count in TARGET_NODE_SCALES:
                paths.append(path)
    return sorted(paths, key=case_sort_key)


def manifest_rows() -> list[dict[str, str]]:
    return read_csv(MANIFEST_CSV)


def small_scale_parameter_case_paths() -> list[Path]:
    manifest_index = {
        row["dataset_name"]: row
        for row in manifest_rows()
        if row.get("scale_category") == "small"
    }
    base_dir = DATASET_ROOT / "parameter_sensitivity" / "small"
    paths: list[Path] = []
    for path in sorted(base_dir.glob("N40_PS_*.dat")):
        if path.name in manifest_index:
            paths.append(path)
    return paths


def filter_case_paths(paths: list[Path], case_names: set[str] | None) -> list[Path]:
    if not case_names:
        return paths
    normalized = {name if name.endswith(".dat") else f"{name}.dat" for name in case_names}
    return [path for path in paths if path.name in normalized]


def build_drive_adj(data: dict[str, Any]) -> dict[int, list[tuple[int, float]]]:
    destination = int(data["D"])
    parking_nodes = {int(value) for value in data["L"]}
    adj: dict[int, list[tuple[int, float]]] = {}
    for (u, v), edge_time in zip(data["E"], data["EdgeTime"]):
        u_i = int(u)
        v_i = int(v)
        if v_i == destination and u_i in parking_nodes:
            continue
        adj.setdefault(u_i, []).append((v_i, float(edge_time)))
    return adj


def shortest_path(adj: dict[int, list[tuple[int, float]]], source: int, target: int) -> tuple[list[int], float]:
    heap: list[tuple[float, int]] = [(0.0, source)]
    dist = {source: 0.0}
    prev: dict[int, int] = {}
    while heap:
        curr_dist, node = heapq.heappop(heap)
        if curr_dist > dist.get(node, math.inf) + TOL:
            continue
        if node == target:
            break
        for nxt, weight in adj.get(node, []):
            cand = curr_dist + float(weight)
            if cand + TOL < dist.get(nxt, math.inf):
                dist[nxt] = cand
                prev[nxt] = node
                heapq.heappush(heap, (cand, nxt))
    if target not in dist:
        return [], math.inf
    path = [target]
    while path[-1] != source:
        parent = prev.get(path[-1])
        if parent is None:
            return [], math.inf
        path.append(parent)
    path.reverse()
    return path, dist[target]


def edge_maps(data: dict[str, Any]) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    distance_by_edge: dict[tuple[int, int], float] = {}
    time_by_edge: dict[tuple[int, int], float] = {}
    for (u, v), distance, edge_time in zip(data["E"], data["Distance"], data["EdgeTime"]):
        edge = (int(u), int(v))
        distance_by_edge[edge] = float(distance)
        time_by_edge[edge] = float(edge_time)
    return distance_by_edge, time_by_edge


def format_num(value: float) -> str:
    return f"{float(value):.12g}"


def opl_objective(*, alpha: float, drive_time_hr: float, walk_time_hr: float, parking_cost: float) -> float:
    return float(alpha) * (60.0 * (float(drive_time_hr) + float(walk_time_hr))) + (1.0 - float(alpha)) * float(parking_cost)


def compare_float_text(left: str, right: str, tol: float = CSV_COMPARE_TOL) -> bool:
    if not left and not right:
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)
    except ValueError:
        return False


def exact_result_for_case(path: Path, historical_rows: dict[str, dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = parse_opl_dat(text)
    stem = path.stem
    node_count = int(data["Nodes"])
    case_id = "C" + stem.split("_C", 1)[1].split("_", 1)[0]
    scenario_name, parameter_name, parameter_value = CASE_METADATA[case_id]
    source = int(data["S"])
    destination = int(data["D"])
    alpha = float(data["Alpha"])
    budget_limit = float(data["Cmax"])
    walk_limit = float(data["W_km"])
    distance_by_edge, time_by_edge = edge_maps(data)
    adj = build_drive_adj(data)
    historical = historical_rows.get(path.name, {})

    base = {
        "scenario_id": case_id,
        "scenario_name": scenario_name,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "method": "Optimization",
        "dat_file": path.name,
        "nodes": str(node_count),
        "edges": str(len(data["E"])),
        "parking_lots": str(int(data["Q"])),
        "source": str(source),
        "destination": str(destination),
        "S": str(source),
        "D": str(destination),
        "Alpha": format_num(alpha),
        "Wmax": format_num(walk_limit),
        "Cmax": format_num(budget_limit),
        "w": format_num(alpha),
        "case_name": stem,
        "OptimizationSource": "ExactShortestPathEnumeration",
        "HistoricalOPLRuntime": historical.get("runtime_ms", ""),
        "HistoricalOPLStatus": historical.get("solver_status", ""),
    }

    start = time.perf_counter()
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int, tuple[int, ...]] | None = None
    for lot, cost in zip(data["L"], data["C"]):
        lot_node = int(lot)
        parking_cost = float(cost)
        walk_edge = (lot_node, destination)
        walk_time = time_by_edge.get(walk_edge)
        walk_distance = distance_by_edge.get(walk_edge)
        if walk_time is None or walk_distance is None:
            continue
        if parking_cost > budget_limit + TOL or walk_distance > walk_limit + TOL:
            continue
        drive_path, drive_time = shortest_path(adj, source, lot_node)
        if not drive_path or not math.isfinite(drive_time):
            continue
        travel_time = drive_time + walk_time
        objective = opl_objective(
            alpha=alpha,
            drive_time_hr=drive_time,
            walk_time_hr=walk_time,
            parking_cost=parking_cost,
        )
        key = (objective, travel_time, parking_cost, lot_node, tuple(drive_path))
        if best is None or key < best_key:
            best = {
                "parking_node": lot_node,
                "drive_path_nodes": drive_path,
                "drive_time_hr": drive_time,
                "walk_distance_km": walk_distance,
                "walk_time_hr": walk_time,
                "travel_time_hr": travel_time,
                "parking_cost": parking_cost,
                "objective": objective,
            }
            best_key = key
    runtime_ms = (time.perf_counter() - start) * 1000.0

    if best is None:
        row = {field: "" for field in RESULT_HEADER}
        row.update(
            {
                **base,
                "feasible": "FALSE",
                "solver_status": "infeasible_no_feasible_parking_lot",
                "path_reconstruction_status": "not_applicable",
                "runtime_ms": f"{runtime_ms:.3f}",
            }
        )
        verification = build_verification_row(row, historical)
        return row, verification

    drive_path_nodes = [int(node) for node in best["drive_path_nodes"]]
    drive_path_text = "->".join(map(str, drive_path_nodes))
    drive_edges = [f"{a}->{b}" for a, b in zip(drive_path_nodes[:-1], drive_path_nodes[1:])]
    walk_edge_text = f"{int(best['parking_node'])}->{destination}"
    full_path_nodes = drive_path_nodes + [destination]

    row = {field: "" for field in RESULT_HEADER}
    row.update(
        {
            **base,
            "feasible": "TRUE",
            "solver_status": "optimal_exact_shortest_path_enumeration",
            "parking_node": str(int(best["parking_node"])),
            "driving_path": drive_path_text,
            "walking_edge": walk_edge_text,
            "full_path": "->".join(map(str, full_path_nodes)),
            "selected_edges": ";".join(drive_edges + [walk_edge_text]),
            "drive_time_hr": format_num(best["drive_time_hr"]),
            "walk_distance_km": format_num(best["walk_distance_km"]),
            "walk_time_hr": format_num(best["walk_time_hr"]),
            "travel_time_hr": format_num(best["travel_time_hr"]),
            "parking_cost": format_num(best["parking_cost"]),
            "objective": format_num(best["objective"]),
            "path_reconstruction_status": "shortest_path",
            "runtime_ms": f"{runtime_ms:.3f}",
            "drive_path": ";".join(drive_edges),
            "walk_path": walk_edge_text,
        }
    )
    verification = build_verification_row(row, historical)
    row["OPLVerificationStatus"] = verification["verification_status"]
    return row, verification


def exact_result_for_parameter_case(
    path: Path,
    manifest_row: dict[str, str],
    historical_rows: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = parse_opl_dat(text)
    source = int(data["S"])
    destination = int(data["D"])
    alpha = float(data["Alpha"])
    budget_limit = float(data["Cmax"])
    walk_limit = float(data["W_km"])
    distance_by_edge, time_by_edge = edge_maps(data)
    adj = build_drive_adj(data)
    historical = historical_rows.get(path.name, {})

    base = {
        "scenario_id": "",
        "scenario_name": "",
        "parameter_name": manifest_row["varied_parameter"],
        "parameter_value": manifest_row["active_parameter_value"],
        "method": "Optimization",
        "dat_file": path.name,
        "nodes": str(int(data["Nodes"])),
        "edges": str(len(data["E"])),
        "parking_lots": str(int(data["Q"])),
        "source": str(source),
        "destination": str(destination),
        "S": str(source),
        "D": str(destination),
        "Alpha": format_num(alpha),
        "Wmax": format_num(walk_limit),
        "Cmax": format_num(budget_limit),
        "w": format_num(alpha),
        "case_name": path.stem,
        "OptimizationSource": "ExactShortestPathEnumeration",
        "HistoricalOPLRuntime": historical.get("runtime_ms", ""),
        "HistoricalOPLStatus": historical.get("solver_status", ""),
    }

    start = time.perf_counter()
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int, tuple[int, ...]] | None = None
    for lot, cost in zip(data["L"], data["C"]):
        lot_node = int(lot)
        parking_cost = float(cost)
        walk_edge = (lot_node, destination)
        walk_time = time_by_edge.get(walk_edge)
        walk_distance = distance_by_edge.get(walk_edge)
        if walk_time is None or walk_distance is None:
            continue
        if parking_cost > budget_limit + TOL or walk_distance > walk_limit + TOL:
            continue
        drive_path, drive_time = shortest_path(adj, source, lot_node)
        if not drive_path or not math.isfinite(drive_time):
            continue
        travel_time = drive_time + walk_time
        objective = opl_objective(
            alpha=alpha,
            drive_time_hr=drive_time,
            walk_time_hr=walk_time,
            parking_cost=parking_cost,
        )
        key = (objective, travel_time, parking_cost, lot_node, tuple(drive_path))
        if best is None or key < best_key:
            best = {
                "parking_node": lot_node,
                "drive_path_nodes": drive_path,
                "drive_time_hr": drive_time,
                "walk_distance_km": walk_distance,
                "walk_time_hr": walk_time,
                "travel_time_hr": travel_time,
                "parking_cost": parking_cost,
                "objective": objective,
            }
            best_key = key
    runtime_ms = (time.perf_counter() - start) * 1000.0

    if best is None:
        row = {field: "" for field in RESULT_HEADER}
        row.update(
            {
                **base,
                "feasible": "FALSE",
                "solver_status": "infeasible_no_feasible_parking_lot",
                "path_reconstruction_status": "not_applicable",
                "runtime_ms": f"{runtime_ms:.3f}",
            }
        )
        verification = build_verification_row(row, historical)
        return row, verification

    drive_path_nodes = [int(node) for node in best["drive_path_nodes"]]
    drive_path_text = "->".join(map(str, drive_path_nodes))
    drive_edges = [f"{a}->{b}" for a, b in zip(drive_path_nodes[:-1], drive_path_nodes[1:])]
    walk_edge_text = f"{int(best['parking_node'])}->{destination}"
    full_path_nodes = drive_path_nodes + [destination]

    row = {field: "" for field in RESULT_HEADER}
    row.update(
        {
            **base,
            "feasible": "TRUE",
            "solver_status": "optimal_exact_shortest_path_enumeration",
            "parking_node": str(int(best["parking_node"])),
            "driving_path": drive_path_text,
            "walking_edge": walk_edge_text,
            "full_path": "->".join(map(str, full_path_nodes)),
            "selected_edges": ";".join(drive_edges + [walk_edge_text]),
            "drive_time_hr": format_num(best["drive_time_hr"]),
            "walk_distance_km": format_num(best["walk_distance_km"]),
            "walk_time_hr": format_num(best["walk_time_hr"]),
            "travel_time_hr": format_num(best["travel_time_hr"]),
            "parking_cost": format_num(best["parking_cost"]),
            "objective": format_num(best["objective"]),
            "path_reconstruction_status": "shortest_path",
            "runtime_ms": f"{runtime_ms:.3f}",
            "drive_path": ";".join(drive_edges),
            "walk_path": walk_edge_text,
        }
    )
    verification = build_verification_row(row, historical)
    row["OPLVerificationStatus"] = verification["verification_status"]
    return row, verification


def build_verification_row(python_row: dict[str, Any], historical_row: dict[str, str]) -> dict[str, Any]:
    historical_status = historical_row.get("solver_status", "")
    historical_feasible = historical_row.get("feasible", "")
    python_status = python_row.get("solver_status", "")
    python_feasible = python_row.get("feasible", "")

    verification_status = "no_historical_reference"
    if historical_row:
        if historical_status == "solved" and historical_feasible.upper() == "TRUE":
            exact_match = (
                python_feasible == "TRUE"
                and python_row.get("parking_node", "") == historical_row.get("parking_node", "")
                and compare_float_text(str(python_row.get("drive_time_hr", "")), historical_row.get("drive_time_hr", ""))
                and compare_float_text(str(python_row.get("walk_time_hr", "")), historical_row.get("walk_time_hr", ""))
                and compare_float_text(str(python_row.get("parking_cost", "")), historical_row.get("parking_cost", ""))
                and compare_float_text(str(python_row.get("objective", "")), historical_row.get("objective", ""))
            )
            verification_status = "exact_match" if exact_match else "mismatch"
        elif historical_status == "time_limit":
            verification_status = "historical_time_limit"
        else:
            verification_status = f"historical_{historical_status or 'unknown'}"

    return {
        "dat_file": python_row.get("dat_file", ""),
        "historical_solver_status": historical_status,
        "historical_feasible": historical_feasible,
        "historical_runtime_ms": historical_row.get("runtime_ms", ""),
        "python_solver_status": python_status,
        "python_feasible": python_feasible,
        "python_runtime_ms": python_row.get("runtime_ms", ""),
        "historical_parking_node": historical_row.get("parking_node", ""),
        "python_parking_node": python_row.get("parking_node", ""),
        "historical_drive_time_hr": historical_row.get("drive_time_hr", ""),
        "python_drive_time_hr": python_row.get("drive_time_hr", ""),
        "historical_walk_time_hr": historical_row.get("walk_time_hr", ""),
        "python_walk_time_hr": python_row.get("walk_time_hr", ""),
        "historical_parking_cost": historical_row.get("parking_cost", ""),
        "python_parking_cost": python_row.get("parking_cost", ""),
        "historical_objective": historical_row.get("objective", ""),
        "python_objective": python_row.get("objective", ""),
        "verification_status": verification_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lot-level exact optimizer for small baseline cases.")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only the specified .dat case name or stem (repeatable).",
    )
    args = parser.parse_args()

    requested_cases = set(args.cases or [])
    case_paths = filter_case_paths(small_scale_case_paths(), requested_cases)
    parameter_paths = filter_case_paths(small_scale_parameter_case_paths(), requested_cases)
    if requested_cases and not case_paths and not parameter_paths:
        raise SystemExit(f"No small-scale cases matched: {sorted(requested_cases)}")

    historical_rows = merged_historical_rows()
    manifest_index = {row["dataset_name"]: row for row in manifest_rows()}

    result_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    for path in case_paths:
        result_row, verification_row = exact_result_for_case(path, historical_rows)
        result_rows.append(result_row)
        verification_rows.append(verification_row)
    for path in parameter_paths:
        result_row, verification_row = exact_result_for_parameter_case(path, manifest_index[path.name], historical_rows)
        result_rows.append(result_row)
        verification_rows.append(verification_row)

    if requested_cases:
        result_rows = merge_existing_rows(OUTPUT_CSV, result_rows, fieldnames=RESULT_HEADER)
        verification_rows = merge_existing_rows(VERIFICATION_CSV, verification_rows, fieldnames=VERIFICATION_HEADER)

    write_csv(OUTPUT_CSV, result_rows, RESULT_HEADER)
    write_csv(VERIFICATION_CSV, verification_rows, VERIFICATION_HEADER)

    matched = sum(1 for row in verification_rows if row["verification_status"] == "exact_match")
    mismatched = sum(1 for row in verification_rows if row["verification_status"] == "mismatch")
    print(f"Wrote {len(result_rows)} exact rows to {OUTPUT_CSV}")
    print(f"Wrote {len(verification_rows)} verification rows to {VERIFICATION_CSV}")
    print(f"Historical OPL exact matches: {matched}")
    print(f"Historical OPL mismatches: {mismatched}")


if __name__ == "__main__":
    main()
