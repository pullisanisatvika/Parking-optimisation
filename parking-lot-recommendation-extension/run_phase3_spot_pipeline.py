#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ACO import aco as aco_mod
from GA import ga as ga_mod
from PSO import pso as pso_mod
from regenerate_spot_level_pipeline import (
    ACO_LOG,
    ALL_SPOT_CSV,
    CANONICAL_FIELDS,
    COMBINED_DIR,
    FINAL_SPOT_CSV,
    GA_LOG,
    PARAM_SPOT_CSV,
    PSO_LOG,
    RAW_LOG_FIELDS,
    RESULTS_DIR,
    SPOT_DIR,
    canonical_row_from_raw,
    dataset_metadata,
    read_csv,
    write_csv,
)
from spot_level_utils import get_feasible_spots, parse_opl_dat, spot_objective


VALIDATION_DIR = RESULTS_DIR / "VALIDATION"
SUMMARY_JSON = VALIDATION_DIR / "phase3_spot_summary.json"
SUMMARY_TXT = VALIDATION_DIR / "phase3_spot_summary.txt"

METHODS = ("GA", "PSO", "ACO")
TOL = 1e-9


def build_dataset_index() -> dict[str, tuple[Path, str, dict[str, Any], dict[str, Any]]]:
    index: dict[str, tuple[Path, str, dict[str, Any], dict[str, Any]]] = {}
    for path in sorted(SPOT_DIR.rglob("*.dat")):
        text = path.read_text(encoding="utf-8")
        data = parse_opl_dat(text)
        metadata = dataset_metadata(path, text, data)
        index[path.stem] = (path, text, data, metadata)
    return index


def scenario_bucket(metadata: dict[str, Any]) -> str:
    if metadata["dataset_group"] == "parameter_sensitivity":
        rel = metadata["dataset_relative_path"].split("/")
        scale = rel[1] if len(rel) > 1 else "unknown"
        return f"parameter_sensitivity/{scale}/{metadata['parameter_name']}"
    rel = metadata["dataset_relative_path"].split("/")
    return f"final_cases/{rel[0]}"


def parse_path_text(text: str) -> list[int]:
    if not text:
        return []
    try:
        return [int(part) for part in text.split("->") if part]
    except ValueError:
        return []


def make_raw_row(
    case_name: str,
    data: dict[str, Any],
    runtime_ms: float,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "Case": case_name,
        "Nodes": int(data["Nodes"]),
        "Edges": len(data["E"]),
        "ParkingLots": int(data["Q"]),
        "TotalSpots": int(data["TotalSpots"]),
        "FeasibleSpots": len(get_feasible_spots(data)),
        "Source": int(data["S"]),
        "Destination": int(data["D"]),
        "Alpha": float(data["Alpha"]),
        "Wmax": float(data["W_km"]),
        "Cmax": float(data["Cmax"]),
        "WeightW": float(data["Alpha"]),
        "runtime_ms": round(runtime_ms, 3),
        "ParkingNode": "",
        "ParkingSpotIndex": "",
        "ParkingSpotId": "",
        "SpotClass": "",
        "drive_time_hr": "",
        "internal_drive_time_hr": "",
        "internal_walk_time_hr": "",
        "external_walk_distance_km": "",
        "external_walk_time_hr": "",
        "walk_time_hr": "",
        "parking_cost": "",
        "objective": "",
        "DrivePath": "",
        "WalkPath": "",
    }
    if not result or not result.get("feasible"):
        return row
    row.update(
        {
            "ParkingNode": result.get("parking_node", ""),
            "ParkingSpotIndex": result.get("parking_spot_index", ""),
            "ParkingSpotId": result.get("parking_spot_id", ""),
            "SpotClass": result.get("spot_class", ""),
            "drive_time_hr": result.get("drive_time_hr", result.get("drive_time", "")),
            "internal_drive_time_hr": result.get("internal_drive_time_hr", ""),
            "internal_walk_time_hr": result.get("internal_walk_time_hr", ""),
            "external_walk_distance_km": result.get("external_walk_distance_km", ""),
            "external_walk_time_hr": result.get("external_walk_time_hr", ""),
            "walk_time_hr": result.get("walk_time_hr", result.get("walk_time", "")),
            "parking_cost": result.get("parking_cost", ""),
            "objective": result.get("objective", ""),
            "DrivePath": "->".join(map(str, result.get("drive_path", []))),
            "WalkPath": "->".join(map(str, result.get("walk_path", []))),
        }
    )
    return row


def run_ga(data: dict[str, Any], seed: int = 42) -> dict[str, Any] | None:
    nodes = list(range(1, int(data["Nodes"]) + 1))
    edges = [(int(u), int(v)) for (u, v) in data["E"]]
    edge_time = ga_mod.make_edge_time_map(edges, data["EdgeTime"])
    edge_distance = ga_mod.make_edge_distance_map(edges, data["Distance"])
    adj = ga_mod.build_adj(edges, edge_time)
    lots = [int(x) for x in data["L"]]
    costs = [float(x) for x in data["C"]]
    walk_time_map = {p: t for p, t in zip(lots, data.get("WalkTimeToDest", []))}
    walk_distance_map = {p: d for p, d in zip(lots, data.get("WalkDistanceToDest", []))}
    feasible_spots = [int(spot["parking_spot_index"]) for spot in get_feasible_spots(data)]
    return ga_mod.solve_ga_joint_route_and_parking(
        data=data,
        nodes=nodes,
        src=int(data["S"]),
        dst=int(data["D"]),
        feasible_spots=feasible_spots,
        adj=adj,
        edge_time=edge_time,
        edge_distance=edge_distance,
        cost_map={p: c for p, c in zip(lots, costs)},
        w=float(data["Alpha"]),
        walk_time_map=walk_time_map,
        walk_distance_map=walk_distance_map,
        population_size=30,
        generations=80,
        parking_mutation_rate=0.10,
        priority_mutation_rate=0.05,
        elitism=True,
        seed=seed,
    )


def run_pso(data: dict[str, Any], seed: int = 42) -> dict[str, Any] | None:
    nodes = list(range(1, int(data["Nodes"]) + 1))
    edges = [(int(u), int(v)) for (u, v) in data["E"]]
    edge_time = pso_mod.make_edge_time_map(edges, data["EdgeTime"])
    edge_distance = pso_mod.make_edge_distance_map(edges, data["Distance"])
    adj = pso_mod.build_adj(edges, edge_time)
    lots = [int(x) for x in data["L"]]
    costs = [float(x) for x in data["C"]]
    walk_time_map = {p: t for p, t in zip(lots, data.get("WalkTimeToDest", []))}
    walk_distance_map = {p: d for p, d in zip(lots, data.get("WalkDistanceToDest", []))}
    feasible_spots = [int(spot["parking_spot_index"]) for spot in get_feasible_spots(data)]
    swarm_size, iterations = pso_mod.recommended_search_budget(int(data["Nodes"]))
    return pso_mod.solve_pso_joint_route_and_parking(
        data=data,
        nodes=nodes,
        src=int(data["S"]),
        dst=int(data["D"]),
        feasible_spots=feasible_spots,
        adj=adj,
        edge_time=edge_time,
        edge_distance=edge_distance,
        cost_map={p: c for p, c in zip(lots, costs)},
        w=float(data["Alpha"]),
        walk_time_map=walk_time_map,
        walk_distance_map=walk_distance_map,
        swarm_size=swarm_size,
        iterations=iterations,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=seed,
    )


def run_aco(data: dict[str, Any], seed: int = 42) -> dict[str, Any] | None:
    edges = [(int(u), int(v)) for (u, v) in data["E"]]
    edge_time = aco_mod.make_edge_time_map(edges, data["EdgeTime"])
    edge_distance = aco_mod.make_edge_distance_map(edges, data["Distance"])
    adj = aco_mod.build_adj(edges, edge_time)
    lots = [int(x) for x in data["L"]]
    costs = [float(x) for x in data["C"]]
    walk_time_map = {p: t for p, t in zip(lots, data.get("WalkTimeToDest", []))}
    walk_distance_map = {p: d for p, d in zip(lots, data.get("WalkDistanceToDest", []))}
    feasible_spots = [int(spot["parking_spot_index"]) for spot in get_feasible_spots(data)]
    return aco_mod.solve_aco_joint_route_and_parking(
        data=data,
        src=int(data["S"]),
        dst=int(data["D"]),
        feasible_spots=feasible_spots,
        adj=adj,
        edge_time=edge_time,
        edge_distance=edge_distance,
        cost_map={p: c for p, c in zip(lots, costs)},
        alpha_weight=float(data["Alpha"]),
        walk_time_map=walk_time_map,
        walk_distance_map=walk_distance_map,
        ants=30,
        iterations=80,
        alpha_pher=1.0,
        beta_heur=2.0,
        evaporation=0.3,
        Q=1.0,
        seed=seed,
    )


def classify_exception(method: str, exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "missing" in text or "parse" in text:
        return "parser/data issue"
    if "math domain" in text or "overflow" in text:
        return "numerical issue"
    if "timeout" in text:
        return "runtime issue"
    return f"{method} bug"


def run_method(method: str, data: dict[str, Any]) -> tuple[dict[str, Any] | None, float]:
    t0 = time.time()
    if method == "GA":
        result = run_ga(data)
    elif method == "PSO":
        result = run_pso(data)
    else:
        result = run_aco(data)
    return result, (time.time() - t0) * 1000.0


def validate_row(
    row: dict[str, Any],
    raw_row: dict[str, Any],
    data: dict[str, Any],
    counters: Counter,
) -> bool:
    if row["method"] == "Heuristic":
        counters["unexpected_heuristic_rows"] += 1
    if row["method"] not in METHODS:
        return False
    counters["total_rows_generated"] += 1

    solved = str(row.get("feasible", "")).upper() == "TRUE"
    if not solved:
        counters["total_rows_validated"] += 1
        return True

    counters["total_rows_validated"] += 1
    spot_index_text = str(row.get("parking_spot_index", "")).strip()
    lot_text = str(row.get("parking_node", "")).strip()
    if not spot_index_text:
        counters["missing_selected_spot"] += 1
        return False
    if not lot_text:
        counters["missing_selected_lot"] += 1
        return False

    spot_index = int(float(spot_index_text))
    lot = int(float(lot_text))

    try:
        total_walk = float(row["total_walk_time_hr"])
        internal_walk = float(row["internal_walk_time_hr"])
        external_walk = float(row["external_walk_time_hr"])
        if not math.isclose(total_walk, internal_walk + external_walk, rel_tol=0.0, abs_tol=TOL):
            counters["walking_time_mismatches"] += 1
    except Exception:
        counters["walking_time_mismatches"] += 1

    try:
        total_travel = float(row["total_travel_time_hr"])
        external_drive = float(row["external_drive_time_hr"])
        internal_drive = float(row["internal_drive_time_hr"])
        internal_walk = float(row["internal_walk_time_hr"])
        external_walk = float(row["external_walk_time_hr"])
        if not math.isclose(
            total_travel,
            external_drive + internal_drive + internal_walk + external_walk,
            rel_tol=0.0,
            abs_tol=TOL,
        ):
            counters["travel_time_mismatches"] += 1
    except Exception:
        counters["travel_time_mismatches"] += 1

    try:
        objective = float(row["objective"])
        expected_objective = spot_objective(
            float(row["Alpha"]),
            float(row["external_drive_time_hr"]),
            float(row["internal_drive_time_hr"]),
            float(row["internal_walk_time_hr"]),
            float(row["external_walk_time_hr"]),
            float(row["parking_cost"]),
        )
        if not math.isclose(objective, expected_objective, rel_tol=0.0, abs_tol=TOL):
            counters["objective_mismatches"] += 1
    except Exception:
        counters["objective_mismatches"] += 1

    if spot_index < 1 or spot_index > int(data["TotalSpots"]):
        counters["missing_selected_spot"] += 1
        return False
    if lot not in {int(value) for value in data["L"]}:
        counters["missing_selected_lot"] += 1
        return False
    if int(data["SpotAvailable"][spot_index - 1]) != 1:
        counters["availability_violations"] += 1
    if int(data["SpotLotNode"][spot_index - 1]) != lot:
        counters["spot_lot_mismatches"] += 1
    if float(row["total_walk_time_hr"]) > float(data["W"]) + TOL:
        counters["wmax_violations"] += 1
    if float(row["parking_cost"]) > float(data["Cmax"]) + TOL:
        counters["cmax_violations"] += 1

    drive_path = parse_path_text(str(raw_row.get("DrivePath", "")))
    if not drive_path or drive_path[-1] != lot:
        counters["selected_lot_reachability_violations"] += 1

    if str(raw_row.get("ParkingSpotIndex", "")) != str(row.get("parking_spot_index", "")):
        counters["post_hoc_replacements"] += 1
    if str(raw_row.get("ParkingNode", "")) != str(row.get("parking_node", "")):
        counters["post_hoc_replacements"] += 1
    return True


def load_existing_raw_logs() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "GA": {row["Case"]: row for row in read_csv(GA_LOG)},
        "PSO": {row["Case"]: row for row in read_csv(PSO_LOG)},
        "ACO": {row["Case"]: row for row in read_csv(ACO_LOG)},
    }


def finalize_from_existing_logs(dataset_index: dict[str, tuple[Path, str, dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    discovered_counts = Counter()
    processed_counts = Counter()
    combined_rows: list[dict[str, Any]] = []
    candidate_checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    infeasible_cases: list[dict[str, Any]] = []

    for _, (_path, _text, _data, metadata) in sorted(dataset_index.items()):
        discovered_counts[scenario_bucket(metadata)] += 1

    representative = [
        "n_40/N40_C1_I1.dat",
        "parameter_sensitivity/medium_large/N2500_PS_spot_availability_A20_I1.dat",
    ]
    for rel in representative:
        path = SPOT_DIR / rel
        if not path.exists():
            continue
        data = parse_opl_dat(path.read_text(encoding="utf-8"))
        ga_set = {spot["parking_spot_id"] for spot in ga_mod.get_feasible_spots(data)}
        pso_set = {spot["parking_spot_id"] for spot in pso_mod.get_feasible_spots(data)}
        aco_set = {spot["parking_spot_id"] for spot in aco_mod.get_feasible_spots(data)}
        candidate_checks.append(
            {
                "scenario": rel,
                "ga_count": len(ga_set),
                "pso_count": len(pso_set),
                "aco_count": len(aco_set),
                "identical": ga_set == pso_set == aco_set,
            }
        )

    raw_logs = load_existing_raw_logs()
    for dataset_name, (_path, _text, data, metadata) in sorted(dataset_index.items()):
        bucket = scenario_bucket(metadata)
        processed_counts[bucket] += 1
        for method in METHODS:
            raw_row = raw_logs[method].get(dataset_name)
            if raw_row is None:
                failures.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "classification": "output writer issue",
                        "exception": "Missing raw log row",
                    }
                )
                continue
            if not raw_row.get("objective"):
                infeasible_cases.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "classification": "no feasible spots",
                    }
                )
            combined_rows.append(
                canonical_row_from_raw(
                    method=method,
                    raw=raw_row,
                    metadata=metadata,
                    data=data,
                    result_source=f"RESULTS/{method}/log_{method.lower()}_spot.csv",
                )
            )

    combined_rows.sort(key=lambda row: (int(row["nodes"]), row["dataset_group"], row["dataset_name"], row["method"]))
    write_csv(ALL_SPOT_CSV, combined_rows, CANONICAL_FIELDS)
    write_csv(PARAM_SPOT_CSV, [row for row in combined_rows if row["dataset_group"] == "parameter_sensitivity"], CANONICAL_FIELDS)
    write_csv(FINAL_SPOT_CSV, [row for row in combined_rows if row["dataset_group"] == "final_case"], CANONICAL_FIELDS)

    validation_counts = Counter()
    for row in combined_rows:
        _path, _text, data, _metadata = dataset_index[row["dataset_name"]]
        validate_row(row, raw_logs[row["method"]][row["dataset_name"]], data, validation_counts)

    comparison = Counter()
    rows_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in combined_rows:
        rows_by_case[row["dataset_name"]][row["method"]] = row
    for dataset_name, method_rows in rows_by_case.items():
        if set(method_rows) != set(METHODS):
            continue
        if any(str(method_rows[m]["feasible"]).upper() != "TRUE" for m in METHODS):
            continue
        spots = {method_rows[m]["parking_spot_id"] for m in METHODS}
        lots = {method_rows[m]["parking_node"] for m in METHODS}
        if len(spots) == 1:
            comparison["identical_spots"] += 1
        elif len(lots) == 1:
            comparison["different_spots_same_lot"] += 1
            comparison["all_same_lot_different_spots"] += 1
        else:
            comparison["different_lots"] += 1

    return {
        "commands_used": [
            "python3 parking-lot-recommendation-extension/GA/ga.py --dat-dir parking-lot-recommendation-extension/final_cases_with_spots --out parking-lot-recommendation-extension/RESULTS/GA/log_ga_spot.csv",
            "python3 parking-lot-recommendation-extension/PSO/pso.py --dat-dir parking-lot-recommendation-extension/final_cases_with_spots --out parking-lot-recommendation-extension/RESULTS/PSO/log_pso_spot.csv",
            "python3 parking-lot-recommendation-extension/ACO/aco.py --dat-dir parking-lot-recommendation-extension/final_cases_with_spots --out parking-lot-recommendation-extension/RESULTS/ACO/log_aco_spot.csv",
            "python3 parking-lot-recommendation-extension/run_phase3_spot_pipeline.py --mode finalize-only",
        ],
        "authoritative_runner": "GA/PSO/ACO entry points for raw execution, regenerate_spot_level_pipeline.py helpers for canonicalization",
        "result_folders_regenerated": [
            str(GA_LOG.parent.relative_to(RESULTS_DIR.parent)),
            str(PSO_LOG.parent.relative_to(RESULTS_DIR.parent)),
            str(ACO_LOG.parent.relative_to(RESULTS_DIR.parent)),
            str(COMBINED_DIR.relative_to(RESULTS_DIR.parent)),
            str(VALIDATION_DIR.relative_to(RESULTS_DIR.parent)),
        ],
        "dataset_coverage": {
            bucket: {
                "discovered": discovered_counts[bucket],
                "processed": processed_counts[bucket],
            }
            for bucket in sorted(discovered_counts)
        },
        "method_counts": {
            "GA": len(raw_logs["GA"]),
            "PSO": len(raw_logs["PSO"]),
            "ACO": len(raw_logs["ACO"]),
        },
        "candidate_set_consistency": candidate_checks,
        "validation_summary": dict(validation_counts),
        "failures": failures,
        "infeasible_cases": infeasible_cases,
        "spot_selection_comparison": dict(comparison),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 spot-level runner/finalizer")
    parser.add_argument("--mode", choices=("run", "finalize-only"), default="run")
    args = parser.parse_args()
    dataset_index = build_dataset_index()
    if args.mode == "finalize-only":
        summary = finalize_from_existing_logs(dataset_index)
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        lines = [
            f"Authoritative runner: {summary['authoritative_runner']}",
            "Commands used:",
            *[f"  {command}" for command in summary["commands_used"]],
            "Method counts:",
            *[f"  {method}: {count}" for method, count in summary["method_counts"].items()],
            "Candidate-set consistency:",
            *[
                f"  {item['scenario']}: GA={item['ga_count']} PSO={item['pso_count']} ACO={item['aco_count']} identical={item['identical']}"
                for item in summary["candidate_set_consistency"]
            ],
            "Validation summary:",
            *[f"  {key}: {value}" for key, value in sorted(summary["validation_summary"].items())],
            "Spot-selection comparison:",
            *[f"  {key}: {value}" for key, value in sorted(summary["spot_selection_comparison"].items())],
            f"Failures: {len(summary['failures'])}",
            f"Infeasible cases: {len(summary['infeasible_cases'])}",
        ]
        SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(SUMMARY_TXT.read_text(encoding="utf-8"), end="")
        return 0

    discovered_counts = Counter()
    processed_counts = Counter()
    raw_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    combined_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    infeasible_cases: list[dict[str, Any]] = []
    candidate_checks: list[dict[str, Any]] = []

    for _, (path, _text, data, metadata) in sorted(dataset_index.items()):
        discovered_counts[scenario_bucket(metadata)] += 1

    representative = [
        "n_40/N40_C1_I1.dat",
        "parameter_sensitivity/medium_large/N2500_PS_spot_availability_A20_I1.dat",
    ]
    for rel in representative:
        path = SPOT_DIR / rel
        if not path.exists():
            continue
        data = parse_opl_dat(path.read_text(encoding="utf-8"))
        ga_set = {spot["parking_spot_id"] for spot in ga_mod.get_feasible_spots(data)}
        pso_set = {spot["parking_spot_id"] for spot in pso_mod.get_feasible_spots(data)}
        aco_set = {spot["parking_spot_id"] for spot in aco_mod.get_feasible_spots(data)}
        identical = ga_set == pso_set == aco_set
        candidate_checks.append(
            {
                "scenario": rel,
                "ga_count": len(ga_set),
                "pso_count": len(pso_set),
                "aco_count": len(aco_set),
                "identical": identical,
            }
        )

    for method, log_path in (("GA", GA_LOG), ("PSO", PSO_LOG), ("ACO", ACO_LOG)):
        if log_path.exists():
            log_path.unlink()

    for dataset_name, (path, _text, data, metadata) in sorted(dataset_index.items()):
        bucket = scenario_bucket(metadata)
        processed_counts[bucket] += 1
        for method in METHODS:
            try:
                feasible_spots = get_feasible_spots(data)
                if not feasible_spots:
                    result = None
                    runtime_ms = 0.0
                    infeasible_cases.append(
                        {
                            "dataset": dataset_name,
                            "method": method,
                            "classification": "no feasible spots",
                        }
                    )
                else:
                    result, runtime_ms = run_method(method, data)
            except Exception as exc:
                runtime_ms = 0.0
                failures.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "classification": classify_exception(method, exc),
                        "exception": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                result = None

            raw_row = make_raw_row(dataset_name, data, runtime_ms, result)
            raw_logs[method].append(raw_row)
            combined_rows.append(
                canonical_row_from_raw(
                    method=method,
                    raw=raw_row,
                    metadata=metadata,
                    data=data,
                    result_source=f"RESULTS/{method}/log_{method.lower()}_spot.csv",
                )
            )

    write_csv(GA_LOG, raw_logs["GA"], RAW_LOG_FIELDS)
    write_csv(PSO_LOG, raw_logs["PSO"], RAW_LOG_FIELDS)
    write_csv(ACO_LOG, raw_logs["ACO"], RAW_LOG_FIELDS)

    combined_rows.sort(key=lambda row: (int(row["nodes"]), row["dataset_group"], row["dataset_name"], row["method"]))
    write_csv(ALL_SPOT_CSV, combined_rows, CANONICAL_FIELDS)
    write_csv(PARAM_SPOT_CSV, [row for row in combined_rows if row["dataset_group"] == "parameter_sensitivity"], CANONICAL_FIELDS)
    write_csv(FINAL_SPOT_CSV, [row for row in combined_rows if row["dataset_group"] == "final_case"], CANONICAL_FIELDS)

    validation_counts = Counter()
    raw_by_method_case = {
        method: {row["Case"]: row for row in rows}
        for method, rows in raw_logs.items()
    }
    for row in combined_rows:
        path, _text, data, _metadata = dataset_index[row["dataset_name"]]
        _ = path
        validate_row(row, raw_by_method_case[row["method"]][row["dataset_name"]], data, validation_counts)

    comparison = Counter()
    rows_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in combined_rows:
        rows_by_case[row["dataset_name"]][row["method"]] = row
    for dataset_name, method_rows in rows_by_case.items():
        if set(method_rows) != set(METHODS):
            continue
        if any(str(method_rows[m]["feasible"]).upper() != "TRUE" for m in METHODS):
            continue
        ga_row = method_rows["GA"]
        pso_row = method_rows["PSO"]
        aco_row = method_rows["ACO"]
        spots = {ga_row["parking_spot_id"], pso_row["parking_spot_id"], aco_row["parking_spot_id"]}
        lots = {ga_row["parking_node"], pso_row["parking_node"], aco_row["parking_node"]}
        if len(spots) == 1:
            comparison["identical_spots"] += 1
        elif len(lots) == 1:
            comparison["different_spots_same_lot"] += 1
            comparison["all_same_lot_different_spots"] += 1
        else:
            comparison["different_lots"] += 1

    summary = {
        "commands_used": [
            "python3 parking-lot-recommendation-extension/run_phase3_spot_pipeline.py",
        ],
        "authoritative_runner": "regenerate_spot_level_pipeline.py helpers plus Phase 3 wrapper for run-only execution",
        "result_folders_regenerated": [
            str(GA_LOG.parent.relative_to(RESULTS_DIR.parent)),
            str(PSO_LOG.parent.relative_to(RESULTS_DIR.parent)),
            str(ACO_LOG.parent.relative_to(RESULTS_DIR.parent)),
            str(COMBINED_DIR.relative_to(RESULTS_DIR.parent)),
            str(VALIDATION_DIR.relative_to(RESULTS_DIR.parent)),
        ],
        "dataset_coverage": {
            bucket: {
                "discovered": discovered_counts[bucket],
                "processed": processed_counts[bucket],
            }
            for bucket in sorted(discovered_counts)
        },
        "method_counts": {
            "GA": len(raw_logs["GA"]),
            "PSO": len(raw_logs["PSO"]),
            "ACO": len(raw_logs["ACO"]),
        },
        "candidate_set_consistency": candidate_checks,
        "validation_summary": dict(validation_counts),
        "failures": failures,
        "infeasible_cases": infeasible_cases,
        "spot_selection_comparison": dict(comparison),
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"Authoritative runner: {summary['authoritative_runner']}",
        "Commands used:",
        *[f"  {command}" for command in summary["commands_used"]],
        "Method counts:",
        *[f"  {method}: {count}" for method, count in summary["method_counts"].items()],
        "Candidate-set consistency:",
        *[
            f"  {item['scenario']}: GA={item['ga_count']} PSO={item['pso_count']} ACO={item['aco_count']} identical={item['identical']}"
            for item in candidate_checks
        ],
        "Validation summary:",
        *[f"  {key}: {value}" for key, value in sorted(validation_counts.items())],
        "Spot-selection comparison:",
        *[f"  {key}: {value}" for key, value in sorted(comparison.items())],
        f"Failures: {len(failures)}",
        f"Infeasible cases: {len(infeasible_cases)}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_TXT.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
