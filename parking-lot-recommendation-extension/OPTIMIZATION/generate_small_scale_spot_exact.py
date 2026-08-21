#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spot_level_utils import (
    build_spot_normalization_context,
    evaluate_spot,
    get_feasible_spots,
    normalized_spot_objective,
    parse_opl_dat,
)


DATASET_ROOT = ROOT / "final_cases_with_spots"
OPTIMIZATION_CSV = ROOT / "OPTIMIZATION" / "optimisation_results_summary.csv"
CANONICAL_OPTIMIZATION_CSV = ROOT / "RESULTS" / "Optimization" / "log_optimization_spot.csv"
MANIFEST_CSV = ROOT / "RESULTS" / "VALIDATION" / "parameter_sensitivity_manifest.csv"
GA_LOG = ROOT / "RESULTS" / "GA" / "log_ga_spot.csv"
PSO_LOG = ROOT / "RESULTS" / "PSO" / "log_pso_spot.csv"
ACO_LOG = ROOT / "RESULTS" / "ACO" / "log_aco_spot.csv"

TARGET_NODE_SCALES = {10, 20, 30, 40}
CASE_ORDER = [f"C{i}" for i in range(1, 11)]
TOL = 1e-9
SMALL_PARAM_TARGETS = {"parking_spot_availability", "spots_per_lot"}

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

BASE_HEADER = [
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
]

EXTRA_HEADER = [
    "case_name",
    "OptimizationSource",
    "HistoricalOPLRuntime",
    "parking_spot_index",
    "parking_spot_id",
    "spot_class",
    "selected_spot_available",
    "internal_drive_time_hr",
    "internal_walk_time_hr",
    "external_walk_time_hr",
    "total_walk_time_hr",
    "status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def case_sort_key(path: Path) -> tuple[int, int]:
    stem = path.stem
    match_nodes = re.search(r"N(\d+)", stem)
    node_count = int(match_nodes.group(1)) if match_nodes else 0
    match_case = re.search(r"_C(\d+)", stem)
    case_id = int(match_case.group(1)) if match_case else 0
    return node_count, case_id


def small_scale_case_paths() -> list[Path]:
    paths: list[Path] = []
    for subdir in sorted(DATASET_ROOT.glob("n_*")):
        for path in sorted(subdir.glob("N*_C*_I*.dat")):
            node_count = int(path.stem.split("_", 1)[0][1:])
            if node_count in TARGET_NODE_SCALES:
                paths.append(path)
    return sorted(paths, key=case_sort_key)


def filter_case_paths(paths: list[Path], case_names: set[str] | None) -> list[Path]:
    if not case_names:
        return paths
    normalized = {name if name.endswith(".dat") else f"{name}.dat" for name in case_names}
    return [path for path in paths if path.name in normalized]


def final_case_paths(scope: str) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(DATASET_ROOT.glob("n_*/*.dat")):
        if "_C" not in path.stem:
            continue
        if scope == "small":
            node_count = int(path.stem.split("_", 1)[0][1:])
            if node_count not in TARGET_NODE_SCALES:
                continue
        paths.append(path)
    if scope == "all":
        paths.extend(path for path in sorted((DATASET_ROOT / "SUMO").glob("*.dat")) if "_C" in path.stem)
    return sorted(paths, key=lambda path: (path.parent.name, case_sort_key(path)))


def manifest_rows() -> list[dict[str, str]]:
    return read_csv(MANIFEST_CSV)


def parameter_case_paths(scope: str) -> list[Path]:
    base_dir = DATASET_ROOT / "parameter_sensitivity"
    if not base_dir.exists():
        return []
    if scope == "small":
        return sorted((base_dir / "small").glob("N40_PS_*.dat"))
    return sorted(base_dir.rglob("*.dat"))


def parameter_metadata_from_data(data: dict[str, Any], path: Path) -> dict[str, str]:
    parameter_name = str(data.get("VariedParameter") or data.get("ParameterLabel") or "").strip()
    parameter_value = str(data.get("ActiveParameterValue") or "").strip()
    if not parameter_name:
        stem = path.stem
        if "_PS_" in stem:
            parameter_name = stem.split("_PS_", 1)[1]
        else:
            parameter_name = "parameter_sensitivity"
    if not parameter_value:
        parameter_value = path.stem
    scale_category = str(data.get("ScaleCategory") or path.parent.name).strip()
    return {
        "dataset_name": path.name,
        "varied_parameter": parameter_name,
        "active_parameter_value": parameter_value,
        "scale_category": scale_category,
    }


def build_adj(edges: list[tuple[int, int]], edge_times: list[float]) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = {}
    for (u, v), t in zip(edges, edge_times):
        adj.setdefault(int(u), []).append((int(v), float(t)))
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


def exact_result_for_case(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = parse_opl_dat(text)
    stem = path.stem
    node_count = int(data["Nodes"])
    case_id = "C" + stem.split("_C", 1)[1].split("_", 1)[0]
    scenario_name, parameter_name, parameter_value = CASE_METADATA[case_id]
    adj = build_adj([(int(u), int(v)) for u, v in data["E"]], [float(x) for x in data["EdgeTime"]])
    feasible_spots = get_feasible_spots(data)
    summary = {
        "case_name": stem,
        "nodes": node_count,
        "parking_lots": int(data["Q"]),
        "total_spots": int(data["TotalSpots"]),
        "feasible_spots": len(feasible_spots),
        "alpha": float(data["Alpha"]),
        "Wmax": float(data["W_km"]),
        "Cmax": float(data["Cmax"]),
        "status": "Infeasible" if not feasible_spots else "Optimal",
    }
    if not feasible_spots:
        row = {key: "" for key in BASE_HEADER + EXTRA_HEADER}
        row.update(
            {
                "scenario_id": case_id,
                "scenario_name": scenario_name,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "method": "Optimization",
                "dat_file": path.name,
                "case_name": stem,
                "nodes": str(node_count),
                "edges": str(len(data["E"])),
                "parking_lots": str(int(data["Q"])),
                "source": str(int(data["S"])),
                "destination": str(int(data["D"])),
                "S": str(int(data["S"])),
                "D": str(int(data["D"])),
                "Alpha": str(float(data["Alpha"])),
                "Wmax": str(float(data["W_km"])),
                "Cmax": str(float(data["Cmax"])),
                "w": str(float(data["Alpha"])),
                "feasible": "FALSE",
                "solver_status": "infeasible_no_feasible_spot",
                "path_reconstruction_status": "not_applicable",
                "OptimizationSource": "ExhaustiveEnumeration",
                "status": "Infeasible",
                "selected_spot_available": "",
                "runtime_ms": "0",
            }
        )
        return row, summary

    start = time.perf_counter()
    shortest_by_lot: dict[int, tuple[list[int], float]] = {}
    for lot in sorted({int(spot["parking_node"]) for spot in feasible_spots}):
        shortest_by_lot[lot] = shortest_path(adj, int(data["S"]), lot)

    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int, int] | None = None
    best_path: list[int] = []
    for spot in feasible_spots:
        lot = int(spot["parking_node"])
        drive_path, drive_time = shortest_by_lot[lot]
        if not drive_path or not math.isfinite(drive_time):
            continue
        evaluated = evaluate_spot(data, drive_time, int(spot["parking_spot_index"]), alpha=float(data["Alpha"]))
        if evaluated is None:
            continue
        key = (
            float(evaluated["objective"]),
            float(evaluated["travel_time_hr"]),
            float(evaluated["parking_cost"]),
            int(evaluated["parking_node"]),
            int(evaluated["parking_spot_index"]),
        )
        if best is None or key < best_key:
            best = evaluated
            best_key = key
            best_path = drive_path
    runtime_ms = (time.perf_counter() - start) * 1000.0

    if best is None:
        summary["status"] = "Infeasible"
        row = {key: "" for key in BASE_HEADER + EXTRA_HEADER}
        row.update(
            {
                "scenario_id": case_id,
                "scenario_name": scenario_name,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "method": "Optimization",
                "dat_file": path.name,
                "case_name": stem,
                "nodes": str(node_count),
                "edges": str(len(data["E"])),
                "parking_lots": str(int(data["Q"])),
                "source": str(int(data["S"])),
                "destination": str(int(data["D"])),
                "S": str(int(data["S"])),
                "D": str(int(data["D"])),
                "Alpha": str(float(data["Alpha"])),
                "Wmax": str(float(data["W_km"])),
                "Cmax": str(float(data["Cmax"])),
                "w": str(float(data["Alpha"])),
                "feasible": "FALSE",
                "solver_status": "infeasible_unreachable_lots",
                "path_reconstruction_status": "unreachable",
                "OptimizationSource": "ExhaustiveEnumeration",
                "status": "Infeasible",
                "runtime_ms": f"{runtime_ms:.3f}",
            }
        )
        return row, summary

    total_walk = float(best["internal_walk_time_hr"]) + float(best["external_walk_time_hr"])
    total_travel = float(best["drive_time_hr"]) + float(best["internal_drive_time_hr"]) + total_walk
    objective_context = build_spot_normalization_context(data)
    objective = normalized_spot_objective(total_travel, float(best["parking_cost"]), context=objective_context)
    if not math.isclose(total_walk, float(best["walk_time_hr"]), rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"Walk-time mismatch for {stem}")
    if not math.isclose(total_travel, float(best["travel_time_hr"]), rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"Travel-time mismatch for {stem}")
    if not math.isclose(objective, float(best["objective"]), rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"Objective mismatch for {stem}")

    row = {key: "" for key in BASE_HEADER + EXTRA_HEADER}
    row.update(
        {
            "scenario_id": case_id,
            "scenario_name": scenario_name,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "method": "Optimization",
            "dat_file": path.name,
            "case_name": stem,
            "nodes": str(node_count),
            "edges": str(len(data["E"])),
            "parking_lots": str(int(data["Q"])),
            "source": str(int(data["S"])),
            "destination": str(int(data["D"])),
            "S": str(int(data["S"])),
            "D": str(int(data["D"])),
            "Alpha": str(float(data["Alpha"])),
            "Wmax": str(float(data["W_km"])),
            "Cmax": str(float(data["Cmax"])),
            "w": str(float(data["Alpha"])),
            "feasible": "TRUE",
            "solver_status": "optimal_exhaustive_enumeration",
            "parking_node": str(int(best["parking_node"])),
            "driving_path": "->".join(map(str, best_path)),
            "walking_edge": f"{int(best['parking_node'])}->{int(data['D'])}",
            "full_path": "->".join(map(str, best_path + [int(data['D'])])),
            "selected_edges": ";".join(f"{a}->{b}" for a, b in zip(best_path[:-1], best_path[1:])),
            "drive_time_hr": f"{float(best['drive_time_hr']):.12g}",
            "walk_distance_km": f"{float(best['external_walk_distance_km']):.12g}",
            "walk_time_hr": f"{float(best['walk_time_hr']):.12g}",
            "travel_time_hr": f"{float(best['travel_time_hr']):.12g}",
            "parking_cost": f"{float(best['parking_cost']):.12g}",
            "objective": f"{float(best['objective']):.12g}",
            "path_reconstruction_status": "shortest_path",
            "runtime_ms": f"{runtime_ms:.3f}",
            "drive_path": "->".join(map(str, best_path)),
            "walk_path": f"{int(best['parking_node'])}->{int(data['D'])}",
            "OptimizationSource": "ExhaustiveEnumeration",
            "HistoricalOPLRuntime": "",
            "parking_spot_index": str(int(best["parking_spot_index"])),
            "parking_spot_id": str(best["parking_spot_id"]),
            "spot_class": str(best["spot_class"]),
            "selected_spot_available": "1",
            "internal_drive_time_hr": f"{float(best['internal_drive_time_hr']):.12g}",
            "internal_walk_time_hr": f"{float(best['internal_walk_time_hr']):.12g}",
            "external_walk_time_hr": f"{float(best['external_walk_time_hr']):.12g}",
            "total_walk_time_hr": f"{float(best['walk_time_hr']):.12g}",
            "status": "Optimal",
        }
    )
    summary.update(
        {
            "SelectedLot": int(best["parking_node"]),
            "SelectedSpot": str(best["parking_spot_id"]),
            "ParkingSpotIndex": int(best["parking_spot_index"]),
            "ObjectiveValue": float(best["objective"]),
            "RuntimeMs": runtime_ms,
        }
    )
    return row, summary


def exact_result_for_parameter_case(path: Path, manifest_row: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = parse_opl_dat(text)
    manifest_row = manifest_row or parameter_metadata_from_data(data, path)
    feasible_spots = get_feasible_spots(data)
    summary = {
        "case_name": path.stem,
        "parameter_name": manifest_row["varied_parameter"],
        "parameter_value": manifest_row["active_parameter_value"],
        "nodes": int(data["Nodes"]),
        "parking_lots": int(data["Q"]),
        "total_spots": int(data["TotalSpots"]),
        "feasible_spots": len(feasible_spots),
        "alpha": float(data["Alpha"]),
        "Wmax": float(data["W_km"]),
        "Cmax": float(data["Cmax"]),
        "status": "Infeasible" if not feasible_spots else "Optimal",
    }

    base = {
        "scenario_id": "",
        "scenario_name": "",
        "parameter_name": manifest_row["varied_parameter"],
        "parameter_value": manifest_row["active_parameter_value"],
        "method": "Optimization",
        "dat_file": path.name,
        "case_name": path.stem,
        "nodes": str(int(data["Nodes"])),
        "edges": str(len(data["E"])),
        "parking_lots": str(int(data["Q"])),
        "source": str(int(data["S"])),
        "destination": str(int(data["D"])),
        "S": str(int(data["S"])),
        "D": str(int(data["D"])),
        "Alpha": str(float(data["Alpha"])),
        "Wmax": str(float(data["W_km"])),
        "Cmax": str(float(data["Cmax"])),
        "w": str(float(data["Alpha"])),
        "OptimizationSource": "ExhaustiveEnumeration",
    }

    if not feasible_spots:
        row = {key: "" for key in BASE_HEADER + EXTRA_HEADER}
        row.update(
            {
                **base,
                "feasible": "FALSE",
                "solver_status": "infeasible_no_feasible_spot",
                "path_reconstruction_status": "not_applicable",
                "status": "Infeasible",
                "runtime_ms": "0",
            }
        )
        return row, summary

    adj = build_adj([(int(u), int(v)) for u, v in data["E"]], [float(x) for x in data["EdgeTime"]])
    start = time.perf_counter()
    shortest_by_lot: dict[int, tuple[list[int], float]] = {}
    for lot in sorted({int(spot["parking_node"]) for spot in feasible_spots}):
        shortest_by_lot[lot] = shortest_path(adj, int(data["S"]), lot)

    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int, int] | None = None
    best_path: list[int] = []
    for spot in feasible_spots:
        lot = int(spot["parking_node"])
        drive_path, drive_time = shortest_by_lot[lot]
        if not drive_path or not math.isfinite(drive_time):
            continue
        evaluated = evaluate_spot(data, drive_time, int(spot["parking_spot_index"]), alpha=float(data["Alpha"]))
        if evaluated is None:
            continue
        key = (
            float(evaluated["objective"]),
            float(evaluated["travel_time_hr"]),
            float(evaluated["parking_cost"]),
            int(evaluated["parking_node"]),
            int(evaluated["parking_spot_index"]),
        )
        if best is None or key < best_key:
            best = evaluated
            best_key = key
            best_path = drive_path
    runtime_ms = (time.perf_counter() - start) * 1000.0

    if best is None:
        summary["status"] = "Infeasible"
        row = {key: "" for key in BASE_HEADER + EXTRA_HEADER}
        row.update(
            {
                **base,
                "feasible": "FALSE",
                "solver_status": "infeasible_unreachable_lots",
                "path_reconstruction_status": "unreachable",
                "status": "Infeasible",
                "runtime_ms": f"{runtime_ms:.3f}",
            }
        )
        return row, summary

    total_walk = float(best["internal_walk_time_hr"]) + float(best["external_walk_time_hr"])
    total_travel = float(best["drive_time_hr"]) + float(best["internal_drive_time_hr"]) + total_walk
    objective_context = build_spot_normalization_context(data)
    objective = normalized_spot_objective(total_travel, float(best["parking_cost"]), context=objective_context)
    if not math.isclose(total_walk, float(best["walk_time_hr"]), rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"Walk-time mismatch for {path.stem}")
    if not math.isclose(total_travel, float(best["travel_time_hr"]), rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"Travel-time mismatch for {path.stem}")
    if not math.isclose(objective, float(best["objective"]), rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"Objective mismatch for {path.stem}")

    row = {key: "" for key in BASE_HEADER + EXTRA_HEADER}
    row.update(
        {
            **base,
            "feasible": "TRUE",
            "solver_status": "optimal_exhaustive_enumeration",
            "parking_node": str(int(best["parking_node"])),
            "driving_path": "->".join(map(str, best_path)),
            "walking_edge": f"{int(best['parking_node'])}->{int(data['D'])}",
            "full_path": "->".join(map(str, best_path + [int(data['D'])])),
            "selected_edges": ";".join(f"{a}->{b}" for a, b in zip(best_path[:-1], best_path[1:])),
            "drive_time_hr": f"{float(best['drive_time_hr']):.12g}",
            "walk_distance_km": f"{float(best['external_walk_distance_km']):.12g}",
            "walk_time_hr": f"{float(best['walk_time_hr']):.12g}",
            "travel_time_hr": f"{float(best['travel_time_hr']):.12g}",
            "parking_cost": f"{float(best['parking_cost']):.12g}",
            "objective": f"{float(best['objective']):.12g}",
            "path_reconstruction_status": "shortest_path",
            "runtime_ms": f"{runtime_ms:.3f}",
            "drive_path": "->".join(map(str, best_path)),
            "walk_path": f"{int(best['parking_node'])}->{int(data['D'])}",
            "HistoricalOPLRuntime": "",
            "parking_spot_index": str(int(best["parking_spot_index"])),
            "parking_spot_id": str(best["parking_spot_id"]),
            "spot_class": str(best["spot_class"]),
            "selected_spot_available": "1",
            "internal_drive_time_hr": f"{float(best['internal_drive_time_hr']):.12g}",
            "internal_walk_time_hr": f"{float(best['internal_walk_time_hr']):.12g}",
            "external_walk_time_hr": f"{float(best['external_walk_time_hr']):.12g}",
            "total_walk_time_hr": f"{float(best['walk_time_hr']):.12g}",
            "status": "Optimal",
        }
    )
    summary.update(
        {
            "SelectedLot": int(best["parking_node"]),
            "SelectedSpot": str(best["parking_spot_id"]),
            "ParkingSpotIndex": int(best["parking_spot_index"]),
            "ObjectiveValue": float(best["objective"]),
            "RuntimeMs": runtime_ms,
        }
    )
    return row, summary


def merge_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    all_fields = list(dict.fromkeys(BASE_HEADER + EXTRA_HEADER + [field for row in existing_rows for field in row.keys()]))
    keep_existing: list[dict[str, Any]] = []
    target_cases = {row["dat_file"] for row in new_rows}
    for row in existing_rows:
        if row.get("method") == "Optimization" and row.get("dat_file") in target_cases:
            continue
        keep_existing.append({field: row.get(field, "") for field in all_fields})
    merged = keep_existing + [{field: row.get(field, "") for field in all_fields} for row in new_rows]
    merged.sort(key=lambda row: (row.get("dat_file", ""), row.get("method", "")))
    return merged, all_fields


def load_method_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {row["Case"]: row for row in rows}


def compare_with_methods(exact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ga_rows = load_method_rows(GA_LOG)
    pso_rows = load_method_rows(PSO_LOG)
    aco_rows = load_method_rows(ACO_LOG)
    comparisons: list[dict[str, Any]] = []
    for row in exact_rows:
        case = row["case_name"]
        status = row["status"]
        entry: dict[str, Any] = {"Case": case, "Status": status}
        if status != "Optimal":
            comparisons.append(entry)
            continue
        exact_obj = float(row["objective"])
        for method_name, method_rows in (("GA", ga_rows), ("PSO", pso_rows), ("ACO", aco_rows)):
            method_row = method_rows.get(case)
            method_obj = float(method_row["objective"]) if method_row and method_row.get("objective") else math.inf
            entry[f"{method_name}Objective"] = method_obj
            entry[f"Optimization<={method_name}"] = exact_obj <= method_obj + TOL
        comparisons.append(entry)
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact spot-level optimization cases for the extension dataset.")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only the specified .dat case name or stem (repeatable).",
    )
    parser.add_argument(
        "--scope",
        choices=("small", "all"),
        default="small",
        help="`small` keeps the original small-scale behavior; `all` runs the broader extension dataset.",
    )
    args = parser.parse_args()

    requested_cases = set(args.cases or [])

    case_paths = filter_case_paths(final_case_paths(args.scope), requested_cases)
    if requested_cases and not case_paths and not parameter_case_paths(args.scope):
        raise SystemExit(f"No cases matched: {sorted(requested_cases)}")
    exact_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in case_paths:
        row, summary = exact_result_for_case(path)
        exact_rows.append(row)
        summaries.append(summary)

    manifest_index = {row["dataset_name"]: row for row in manifest_rows()}
    param_paths = filter_case_paths(parameter_case_paths(args.scope), requested_cases)
    param_rows: list[dict[str, Any]] = []
    param_summaries: list[dict[str, Any]] = []
    for path in param_paths:
        row, summary = exact_result_for_parameter_case(path, manifest_index.get(path.name))
        param_rows.append(row)
        param_summaries.append(summary)

    all_new_rows = exact_rows + param_rows

    existing_rows = read_csv(OPTIMIZATION_CSV)
    merged_rows, fieldnames = merge_rows(existing_rows, all_new_rows)
    write_csv(OPTIMIZATION_CSV, merged_rows, fieldnames)

    canonical_rows = [{field: row.get(field, "") for field in fieldnames} for row in all_new_rows]
    canonical_rows.sort(key=lambda row: (row.get("dat_file", ""), row.get("method", "")))
    if args.scope == "small":
        write_csv(CANONICAL_OPTIMIZATION_CSV, canonical_rows, fieldnames)

    comparisons = compare_with_methods(all_new_rows)
    feasible = [row for row in all_new_rows if row["status"] == "Optimal"]
    infeasible = [row["case_name"] for row in all_new_rows if row["status"] != "Optimal"]

    print(f"Generated exact optimization rows: {len(all_new_rows)}")
    print(f"Feasible: {len(feasible)}")
    print(f"Infeasible: {len(infeasible)}")
    for summary in summaries + param_summaries:
        print(
            f"{summary['case_name']},{summary['status']},"
            f"{summary.get('SelectedLot', '')},{summary.get('SelectedSpot', '')},"
            f"{summary.get('ObjectiveValue', '')},{summary.get('RuntimeMs', '')}"
        )
    print("Comparisons:")
    for row in comparisons:
        print(row)


if __name__ == "__main__":
    main()
