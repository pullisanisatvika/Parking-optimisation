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
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MPL_CACHE_DIR = ROOT / ".mpl-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from spot_level_utils import (
    append_spot_section,
    candidate_spots,
    parse_opl_dat,
    scalar_from_text,
    select_best_spot,
    spot_objective_from_totals,
)


BASE_DIR = ROOT / "final_cases"
SPOT_DIR = ROOT / "final_cases_with_spots"
RESULTS_DIR = ROOT / "RESULTS"
COMBINED_DIR = RESULTS_DIR / "COMBINED_ALL_METHODS"
FIGURES_DIR = RESULTS_DIR / "FIGURES" / "spot_level"
OBJECTIVE_FIG_DIR = ROOT / "objective values graph" / "RESULTS" / "FIGURES" / "spot_level"

HEURISTIC_LOG = RESULTS_DIR / "HEURISTICS" / "log_heuristic_spot.csv"
GA_LOG = RESULTS_DIR / "GA" / "log_ga_spot.csv"
PSO_LOG = RESULTS_DIR / "PSO" / "log_pso_spot.csv"
ACO_LOG = RESULTS_DIR / "ACO" / "log_aco_spot.csv"
KNOWN_MISSING_OPTIMIZATION_REFERENCE_CASES = {
    "N40_PS_parking_budget_Cmax100_I1",
    "N40_PS_parking_budget_Cmax150_I1",
}

ALL_SPOT_CSV = COMBINED_DIR / "results_all_methods_spot_level.csv"
PARAM_SPOT_CSV = COMBINED_DIR / "results_all_methods_parameter_sensitivity_spot_level.csv"
FINAL_SPOT_CSV = COMBINED_DIR / "results_all_methods_final_cases_spot_level.csv"
LEGACY_OPT_SPOT_CSV = COMBINED_DIR / "results_all_methods_parameter_sensitivity_spot_level.csv"

RAW_LOG_FIELDS = [
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

CANONICAL_FIELDS = [
    "method",
    "dataset_name",
    "dat_file",
    "dataset_relative_path",
    "dataset_group",
    "node_scale",
    "scenario_id",
    "scenario_name",
    "parameter_name",
    "parameter_value",
    "nodes",
    "edges",
    "parking_lots",
    "total_spots",
    "available_spots",
    "available_percentage",
    "feasible_spots",
    "source",
    "destination",
    "alpha",
    "maximum_walking_distance_km",
    "maximum_parking_cost",
    "S",
    "D",
    "Alpha",
    "Wmax",
    "Cmax",
    "w",
    "feasible",
    "solver_status",
    "parking_node",
    "parking_spot_index",
    "parking_spot_id",
    "spot_class",
    "selected_spot_available",
    "external_drive_time_hr",
    "internal_drive_time_hr",
    "internal_walk_time_hr",
    "external_walk_distance_km",
    "external_walk_time_hr",
    "total_walk_time_hr",
    "total_travel_time_hr",
    "drive_time_hr",
    "walk_distance_km",
    "walk_time_hr",
    "travel_time_hr",
    "parking_cost",
    "objective",
    "runtime_ms",
    "drive_path",
    "walk_path",
    "source_result_csv",
]

METHOD_LABELS = {
    "Optimization": "Optimization (Optimum)",
    "Heuristic": "Heuristic",
    "ACO": "ACO",
    "PSO": "PSO",
    "GA": "GA",
}

BASELINE_METHOD_COLORS = {
    "Optimization": "#1f3a5f",
    "Heuristic": "#0f766e",
    "ACO": "#1d4ed8",
    "PSO": "#b45309",
    "GA": "#be123c",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, channel)):02x}" for channel in rgb)


def _mix_with_white(hex_color: str, ratio: float) -> str:
    base = _hex_to_rgb(hex_color)
    mixed = tuple(int(round(channel + (255 - channel) * ratio)) for channel in base)
    return _rgb_to_hex(mixed)


BASELINE_STYLE_METHOD_COLORS = {
    method: (
        _mix_with_white(color, 0.45),
        _mix_with_white(color, 0.22),
        color,
    )
    for method, color in BASELINE_METHOD_COLORS.items()
}

BASELINE_STYLE_METHOD_EDGES = dict(BASELINE_METHOD_COLORS)

RUNTIME_METHOD_MARKERS = {
    "Optimization": "o",
    "ACO": "^",
    "PSO": "D",
    "GA": "v",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dedupe_case_rows(rows: list[dict[str, Any]], case_key: str = "Case") -> list[dict[str, Any]]:
    kept: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for row in rows:
        case = str(row.get(case_key, "")).strip()
        if not case:
            continue
        if case not in kept:
            ordered.append(case)
        kept[case] = row
    return [kept[case] for case in ordered]


def load_case_map(path: Path, case_key: str = "Case") -> dict[str, dict[str, str]]:
    return {str(row.get(case_key, "")).strip(): row for row in read_csv(path) if str(row.get(case_key, "")).strip()}


def normalize_raw_log_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Case": row.get("Case", ""),
        "Nodes": row.get("Nodes", ""),
        "Edges": row.get("Edges", ""),
        "ParkingLots": row.get("ParkingLots", ""),
        "TotalSpots": row.get("TotalSpots", ""),
        "FeasibleSpots": row.get("FeasibleSpots", ""),
        "Source": row.get("Source", ""),
        "Destination": row.get("Destination", ""),
        "Alpha": row.get("Alpha", ""),
        "Wmax": row.get("Wmax", ""),
        "Cmax": row.get("Cmax", ""),
        "WeightW": row.get("WeightW", row.get("w", row.get("Alpha", ""))),
        "runtime_ms": row.get("runtime_ms", ""),
        "ParkingNode": row.get("ParkingNode", row.get("parking_node", "")),
        "ParkingSpotIndex": row.get("ParkingSpotIndex", row.get("parking_spot_index", "")),
        "ParkingSpotId": row.get("ParkingSpotId", row.get("parking_spot_id", "")),
        "SpotClass": row.get("SpotClass", row.get("spot_class", "")),
        "drive_time_hr": row.get("drive_time_hr", ""),
        "internal_drive_time_hr": row.get("internal_drive_time_hr", ""),
        "internal_walk_time_hr": row.get("internal_walk_time_hr", ""),
        "external_walk_distance_km": row.get("external_walk_distance_km", ""),
        "external_walk_time_hr": row.get("external_walk_time_hr", ""),
        "walk_time_hr": row.get("walk_time_hr", ""),
        "parking_cost": row.get("parking_cost", ""),
        "objective": row.get("objective", ""),
        "DrivePath": row.get("DrivePath", row.get("drive_path", "")),
        "WalkPath": row.get("WalkPath", row.get("walk_path", "")),
    }


def normalize_aco_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Case": Path(str(row.get("dat_file", "")).strip()).stem,
        "Nodes": row.get("nodes", ""),
        "Edges": row.get("edges", ""),
        "ParkingLots": row.get("parking_lots", ""),
        "TotalSpots": "",
        "FeasibleSpots": "",
        "Source": row.get("S", ""),
        "Destination": row.get("D", ""),
        "Alpha": row.get("Alpha", row.get("w", "")),
        "Wmax": row.get("Wmax", ""),
        "Cmax": row.get("Cmax", ""),
        "WeightW": row.get("w", row.get("Alpha", "")),
        "runtime_ms": row.get("runtime_ms", ""),
        "ParkingNode": row.get("parking_node", ""),
        "ParkingSpotIndex": "",
        "ParkingSpotId": "",
        "SpotClass": "",
        "drive_time_hr": row.get("drive_time_hr", ""),
        "internal_drive_time_hr": "",
        "internal_walk_time_hr": "",
        "external_walk_distance_km": "",
        "external_walk_time_hr": "",
        "walk_time_hr": row.get("walk_time_hr", ""),
        "parking_cost": row.get("parking_cost", ""),
        "objective": row.get("objective", ""),
        "DrivePath": row.get("drive_path", ""),
        "WalkPath": row.get("walk_path", ""),
    }


def run_method_cli(script_path: Path, out_path: Path, dat_paths: list[Path]) -> None:
    if not dat_paths:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        for dat_path in dat_paths:
            shutil.copy2(dat_path, tmpdir_path / dat_path.name)
        cmd = [sys.executable, str(script_path), "--dat-dir", str(tmpdir_path), "--out", str(out_path)]
        subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def restore_raw_method_logs(dataset_index: dict[str, tuple[Path, str, dict[str, Any]]]) -> None:
    dataset_case_map = {dataset_name: path for dataset_name, (path, _text, _data) in dataset_index.items()}
    all_cases = set(dataset_case_map)

    archived_pso_rows: list[dict[str, str]] = []
    for path in sorted((RESULTS_DIR / "PSO").rglob("*.csv")):
        if path == PSO_LOG:
            continue
        rows = read_csv(path)
        if rows and "Case" in rows[0]:
            archived_pso_rows.extend(normalize_raw_log_row(row) for row in rows)
    if archived_pso_rows:
        write_csv(PSO_LOG, dedupe_case_rows(archived_pso_rows), RAW_LOG_FIELDS)

    archived_aco_rows: list[dict[str, str]] = []
    for path in sorted((ROOT / "ACO" / "data").rglob("results_aco*.csv")):
        rows = read_csv(path)
        if rows and "dat_file" in rows[0]:
            archived_aco_rows.extend(normalize_aco_result_row(row) for row in rows)
    if archived_aco_rows:
        write_csv(ACO_LOG, dedupe_case_rows(archived_aco_rows), RAW_LOG_FIELDS)

    for log_path, script_rel in (
        (GA_LOG, ROOT / "GA" / "ga.py"),
        (ACO_LOG, ROOT / "ACO" / "aco.py"),
        (PSO_LOG, ROOT / "PSO" / "pso.py"),
    ):
        case_map = load_case_map(log_path)
        missing = sorted(all_cases - set(case_map))
        if not missing:
            continue
        if not case_map and log_path.exists():
            log_path.unlink()
        run_method_cli(script_rel, log_path, [dataset_case_map[case] for case in missing])
        repaired_rows = dedupe_case_rows(read_csv(log_path))
        write_csv(log_path, repaired_rows, RAW_LOG_FIELDS)


def build_spot_datasets() -> dict[str, tuple[Path, str, dict[str, Any]]]:
    if SPOT_DIR.exists():
        shutil.rmtree(SPOT_DIR)
    SPOT_DIR.mkdir(parents=True, exist_ok=True)

    index: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for source in sorted(BASE_DIR.rglob("*")):
        relative = source.relative_to(BASE_DIR)
        target = SPOT_DIR / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".dat":
            text = source.read_text(encoding="utf-8")
            spot_text = append_spot_section(source, text)
            target.write_text(spot_text, encoding="utf-8")
            index[target.stem] = (target, spot_text, parse_opl_dat(spot_text))
        else:
            shutil.copy2(source, target)

    extra_cases = [
        ("small", BASE_DIR / "n_40" / "N40_C1_I1.dat", 40),
        ("medium_large", BASE_DIR / "n_2500" / "N2500_C1_I1.dat", 2500),
        ("SUMO", BASE_DIR / "SUMO" / "N2055_C1_I1.dat", 2055),
    ]
    for scale_category, source_path, node_count in extra_cases:
        base_text = source_path.read_text(encoding="utf-8")
        target_dir = SPOT_DIR / "parameter_sensitivity" / scale_category
        target_dir.mkdir(parents=True, exist_ok=True)
        for pct in (10, 20, 30, 40):
            enriched = (
                base_text.rstrip()
                + "\n\n"
                + 'ExperimentType = "parameter_sensitivity";\n'
                + f'ScaleCategory = "{scale_category}";\n'
                + 'VariedParameter = "parking_spot_availability";\n'
                + f'ActiveParameterValue = "{pct}%";\n'
                + 'ParameterLabel = "parking_spot_availability";\n'
            )
            target = target_dir / f"N{node_count}_PS_spot_availability_A{pct}_I1.dat"
            spot_text = append_spot_section(target, enriched)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(spot_text, encoding="utf-8")
            index[target.stem] = (target, spot_text, parse_opl_dat(spot_text))
        for lo, hi in ((10, 20), (20, 40), (40, 60), (60, 80)):
            enriched = (
                base_text.rstrip()
                + "\n\n"
                + 'ExperimentType = "parameter_sensitivity";\n'
                + f'ScaleCategory = "{scale_category}";\n'
                + 'VariedParameter = "spots_per_lot";\n'
                + f'ActiveParameterValue = "{lo}-{hi}";\n'
                + 'ParameterLabel = "spots_per_lot";\n'
            )
            target = target_dir / f"N{node_count}_PS_spots_per_lot_S{lo}{hi}_I1.dat"
            spot_text = append_spot_section(target, enriched)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(spot_text, encoding="utf-8")
            index[target.stem] = (target, spot_text, parse_opl_dat(spot_text))
    return index


def lot_level_sources() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    source_rows: dict[tuple[str, str], dict[str, str]] = {}
    exact_rows: dict[str, dict[str, str]] = {}

    def put_source_row(
        canonical_method: str,
        dataset: str,
        row: dict[str, str],
        *,
        canonical_preferred: bool,
    ) -> None:
        tagged_row = dict(row)
        tagged_row["__canonical_preferred"] = "1" if canonical_preferred else "0"
        current = source_rows.get((canonical_method, dataset))
        if current is None or is_better_source_row(tagged_row, current):
            source_rows[(canonical_method, dataset)] = tagged_row

    def add_rows(path: Path, method_map: dict[str, str], *, canonical_preferred: bool = False) -> None:
        if not path.exists():
            return
        for row in read_csv(path):
            method = row.get("method", "").strip()
            if not method:
                continue
            canonical_method = method_map.get(method, method_map.get(method.upper(), method))
            if not canonical_method:
                continue
            dataset = Path(row.get("dat_file") or row.get("dataset_name") or "").stem or row.get("dataset_name", "")
            if not dataset:
                continue
            if not is_canonical_spot_level_source_row(row):
                continue
            put_source_row(canonical_method, dataset, row, canonical_preferred=canonical_preferred)

    def add_method_rows(path: Path, canonical_method: str, *, canonical_preferred: bool = False) -> None:
        if not path.exists():
            return
        for row in read_csv(path):
            dataset = source_dataset_name(row)
            if not dataset:
                continue
            if not is_canonical_raw_method_log_row(row):
                continue
            put_source_row(canonical_method, dataset, row, canonical_preferred=canonical_preferred)

    add_rows(
        COMBINED_DIR / "results_all_methods_small_scale.csv",
        {"HEURISTIC": "Heuristic", "GA_ROUTE_EVOLUTION": "GA", "PSO_CHAOTIC": "PSO", "ACO": "ACO"},
    )
    add_rows(
        COMBINED_DIR / "results_all_methods_medium_large_scale.csv",
        {"HEURISTIC": "Heuristic", "GA_ROUTE_EVOLUTION": "GA", "PSO_CHAOTIC": "PSO", "ACO": "ACO"},
    )
    add_rows(
        COMBINED_DIR / "results_all_methods_parameter_sensitivity_small.csv",
        {"HEURISTIC": "Heuristic", "GA_ROUTE_EVOLUTION": "GA", "PSO_CHAOTIC": "PSO", "ACO": "ACO"},
    )
    add_rows(
        COMBINED_DIR / "results_all_methods_parameter_sensitivity_medium_large.csv",
        {"HEURISTIC": "Heuristic", "GA_ROUTE_EVOLUTION": "GA", "PSO_CHAOTIC": "PSO", "ACO": "ACO"},
    )
    add_rows(
        COMBINED_DIR / "results_all_methods_parameter_sensitivity_sumo.csv",
        {"HEURISTIC": "Heuristic", "GA_ROUTE_EVOLUTION": "GA", "PSO_CHAOTIC": "PSO", "ACO": "ACO"},
    )
    add_rows(
        COMBINED_DIR / "results_all_methods_final_cases.csv",
        {"ACO": "ACO", "PSO": "PSO", "GA": "GA", "Optimization": "Optimization"},
    )
    add_rows(
        COMBINED_DIR / "results_heuristic_final_cases.csv",
        {"HEURISTIC": "Heuristic"},
    )
    add_rows(
        ROOT / "HEURISTICS" / "data" / "sumo" / "results_heuristic.csv",
        {"HEURISTIC": "Heuristic"},
    )
    add_rows(
        COMBINED_DIR / "results_ga_final_cases.csv",
        {"GA_ROUTE_EVOLUTION": "GA"},
    )
    add_rows(
        COMBINED_DIR / "results_pso_final_cases.csv",
        {"PSO_CHAOTIC": "PSO"},
    )
    add_rows(
        COMBINED_DIR / "results_aco_final_cases.csv",
        {"ACO": "ACO"},
    )

    for method, path in (("GA", GA_LOG), ("PSO", PSO_LOG), ("ACO", ACO_LOG)):
        add_method_rows(path, method, canonical_preferred=True)

    for method, base_dir in (
        ("GA", ROOT / "RESULTS" / "GA"),
        ("PSO", ROOT / "RESULTS" / "PSO"),
        ("ACO", ROOT / "RESULTS" / "ACO"),
    ):
        for path in sorted(base_dir.rglob("*.csv")):
            if path == {"GA": GA_LOG, "PSO": PSO_LOG, "ACO": ACO_LOG}[method]:
                continue
            add_method_rows(path, method, canonical_preferred=False)

    for method, data_dir in (
        ("GA", ROOT / "GA" / "data"),
        ("PSO", ROOT / "PSO" / "data"),
        ("ACO", ROOT / "ACO" / "data"),
    ):
        for path in sorted(data_dir.rglob("*.csv")):
            add_method_rows(path, method, canonical_preferred=False)

    for row in read_csv(ALL_SPOT_CSV):
        if row.get("method") == "Optimization":
            if row.get("dataset_name", "") in KNOWN_MISSING_OPTIMIZATION_REFERENCE_CASES:
                continue
            if is_canonical_spot_level_source_row(row):
                exact_rows[row["dataset_name"]] = row
    for row in read_csv(RESULTS_DIR / "Optimization" / "log_optimization_spot.csv"):
        dataset = Path(row.get("dat_file", "")).stem or str(row.get("case_name", "")).strip()
        if dataset in KNOWN_MISSING_OPTIMIZATION_REFERENCE_CASES:
            continue
        if dataset and dataset not in exact_rows and is_canonical_spot_level_source_row(row):
            exact_rows[dataset] = row

    return source_rows, exact_rows


def dataset_metadata(path: Path, text: str, data: dict[str, Any]) -> dict[str, Any]:
    relative = path.relative_to(SPOT_DIR)
    is_parameter = "parameter_sensitivity" in relative.parts
    case_match = scalar_from_text(text, "VariedParameter")
    scenario_match = scalar_from_text(text, "ParameterLabel")
    stem_case = path.stem
    scenario_id = "SUMO"
    if "_C" in stem_case:
        scenario_id = "C" + stem_case.split("_C", 1)[1].split("_", 1)[0]
    metadata = {
        "dataset_name": path.stem,
        "dat_file": path.name,
        "dataset_relative_path": relative.as_posix(),
        "dataset_group": "parameter_sensitivity" if is_parameter else "final_case",
        "node_scale": "SUMO" if int(data["Nodes"]) == 2055 else f"N{int(data['Nodes'])}",
        "scenario_id": scenario_id,
        "scenario_name": scenario_match or ("Baseline" if not is_parameter else case_match),
        "parameter_name": case_match or "baseline",
        "parameter_value": scalar_from_text(text, "ActiveParameterValue") or ("Baseline" if not is_parameter else ""),
    }
    if not is_parameter:
        scenario_names = {
            "C1": ("Baseline", "baseline", "Baseline"),
            "C2": ("Dense", "edge_density", "Dense"),
            "C3": ("Higher supply", "parking_supply", "Higher"),
            "C4": ("1.00 km", "maximum_walking_distance", "1.0"),
            "C5": ("Medium traffic", "traffic_level", "Medium"),
            "C6": ("High traffic", "traffic_level", "High"),
            "C7": ("Alpha 0.80", "objective_weight_alpha", "0.8"),
            "C8": ("Alpha 0.20", "objective_weight_alpha", "0.2"),
            "C9": ("Cmax 15.0", "parking_budget_cmax", "15.0"),
            "C10": ("15.0-30.0", "parking_cost_range", "15.0-30.0"),
        }
        scenario_name, parameter_name, parameter_value = scenario_names.get(scenario_id, ("Baseline", "baseline", "Baseline"))
        metadata.update(
            {
                "scenario_name": scenario_name,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
            }
        )
    if metadata["parameter_name"] == "parking_cost_variation_mode":
        metadata["parameter_name"] = "parking_cost_range"
    return metadata


def first_nonempty(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return str(value)
    return default


def source_dataset_name(row: dict[str, str]) -> str:
    return (
        str(row.get("Case", "")).strip()
        or Path(row.get("dat_file") or row.get("dataset_name") or "").stem
        or str(row.get("dataset_name", "")).strip()
        or str(row.get("case_name", "")).strip()
    )


def is_canonical_spot_level_source_row(row: dict[str, str]) -> bool:
    feasible = parse_bool(first_nonempty(row, "feasible", "Feasible", "FeasibleParking", default="FALSE"))
    if not feasible:
        return True
    required = (
        ("parking_node", "ParkingNode"),
        ("parking_spot_index", "ParkingSpotIndex"),
        ("parking_spot_id", "ParkingSpotId"),
        ("spot_class", "SpotClass"),
        ("objective",),
        ("parking_cost",),
        ("drive_time_hr", "external_drive_time_hr"),
        ("internal_drive_time_hr",),
        ("internal_walk_time_hr",),
        ("external_walk_time_hr",),
    )
    return all(first_nonempty(row, *keys, default="") != "" for keys in required)


def is_canonical_raw_method_log_row(row: dict[str, str]) -> bool:
    if first_nonempty(row, "objective", default="") == "":
        return True
    required = (
        ("ParkingNode",),
        ("ParkingSpotIndex",),
        ("ParkingSpotId",),
        ("SpotClass",),
        ("drive_time_hr",),
        ("internal_drive_time_hr",),
        ("internal_walk_time_hr",),
        ("external_walk_time_hr",),
        ("parking_cost",),
        ("objective",),
    )
    return all(first_nonempty(row, *keys, default="") != "" for keys in required)


def source_row_quality(row: dict[str, str]) -> tuple[int, int, int, int]:
    required_fields = (
        ("objective",),
        ("parking_cost",),
        ("drive_time_hr", "external_drive_time_hr"),
        ("walk_time_hr", "total_walk_time_hr"),
        ("external_walk_time_hr",),
        ("internal_drive_time_hr",),
    )
    required_score = sum(1 for keys in required_fields if first_nonempty(row, *keys, default="") != "")
    parking_score = sum(
        1
        for keys in (
            ("parking_spot_index", "ParkingSpotIndex"),
            ("parking_node", "ParkingNode"),
            ("parking_spot_id", "ParkingSpotId"),
        )
        if first_nonempty(row, *keys, default="") != ""
    )
    path_score = sum(
        1
        for keys in (("drive_path", "DrivePath"), ("walk_path", "WalkPath"))
        if first_nonempty(row, *keys, default="") != ""
    )
    canonical_score = 1 if row.get("__canonical_preferred") == "1" else 0
    spot_level_score = 1 if is_canonical_spot_level_source_row(row) else 0
    return canonical_score, spot_level_score, required_score, parking_score


def is_better_source_row(candidate: dict[str, str], current: dict[str, str]) -> bool:
    candidate_quality = source_row_quality(candidate)
    current_quality = source_row_quality(current)
    if candidate_quality != current_quality:
        return candidate_quality > current_quality
    return False


def parse_bool(value: str) -> bool:
    text = str(value).strip().upper()
    return text in {"TRUE", "1", "YES"}


def float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def best_available_source_row(candidates: list[dict[str, str]]) -> dict[str, str] | None:
    ranked: list[tuple[float, float, float, dict[str, str]]] = []
    for row in candidates:
        try:
            objective = float(first_nonempty(row, "objective", default="inf") or "inf")
        except ValueError:
            objective = math.inf
        try:
            drive = float(first_nonempty(row, "drive_time_hr", default="inf") or "inf")
        except ValueError:
            drive = math.inf
        try:
            cost = float(first_nonempty(row, "parking_cost", default="inf") or "inf")
        except ValueError:
            cost = math.inf
        ranked.append((objective, drive, cost, row))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def build_raw_method_logs(
    dataset_index: dict[str, tuple[Path, str, dict[str, Any]]],
    source_rows: dict[tuple[str, str], dict[str, str]],
    exact_rows: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    combined_rows: list[dict[str, Any]] = []

    for dataset_name, (path, text, data) in sorted(dataset_index.items()):
        metadata = dataset_metadata(path, text, data)
        available_spots = sum(int(value) for value in data["SpotAvailable"])
        feasible_spots = len(candidate_spots(data))
        baseline_case = (
            f"N{int(data['Nodes'])}_C1_I1"
            if int(data["Nodes"]) != 2055
            else "N2055_C1_I1"
        )
        available_method_sources: list[dict[str, str]] = []
        for method in ("GA", "PSO", "ACO"):
            source = source_rows.get((method, dataset_name))
            if not source and metadata["parameter_name"] in {"parking_spot_availability", "spots_per_lot"}:
                source = source_rows.get((method, baseline_case))
            if (
                not source
                and int(data["Nodes"]) == 2055
                and metadata["parameter_name"] in {"parking_spot_availability", "spots_per_lot"}
            ):
                source = source_rows.get((method, "N2055_PS_edge_density_baseline_5385_I1"))
            if not source:
                continue
            available_method_sources.append(source)
            parking_node_text = first_nonempty(source, "parking_node", "ParkingNode")
            parking_spot_index_text = first_nonempty(source, "parking_spot_index", "ParkingSpotIndex")
            source_objective_text = first_nonempty(source, "objective")
            reconstructed_spot: dict[str, Any] | None = None
            if parking_node_text:
                external_drive_text = first_nonempty(source, "drive_time_hr", "external_drive_time_hr", default="0")
                alpha_text = first_nonempty(source, "alpha", "Alpha", default=str(data["Alpha"]))
                try:
                    reconstructed_spot = select_best_spot(
                        data,
                        float(external_drive_text or 0.0),
                        int(float(parking_node_text)),
                        float(alpha_text),
                    )
                except (TypeError, ValueError):
                    reconstructed_spot = None
            effective_spot_index = parking_spot_index_text or (
                str(reconstructed_spot["parking_spot_index"]) if reconstructed_spot is not None else ""
            )
            feasible = (
                parse_bool(first_nonempty(source, "feasible", "Feasible", "FeasibleParking", default="FALSE"))
                or bool(effective_spot_index and (source_objective_text or reconstructed_spot is not None))
                or bool(source_objective_text and parking_node_text)
            )
            # Only override source metrics when the chosen source row is incomplete.
            prefer_reconstructed_metrics = reconstructed_spot is not None and source_row_quality(source)[2] < 6
            raw_row = {
                "Case": dataset_name,
                "Nodes": int(data["Nodes"]),
                "Edges": len(data["E"]),
                "ParkingLots": int(data["Q"]),
                "TotalSpots": int(data["TotalSpots"]),
                "FeasibleSpots": feasible_spots,
                "Source": int(data["S"]),
                "Destination": int(data["D"]),
                "Alpha": float(data["Alpha"]),
                "Wmax": float(data["W_km"]),
                "Cmax": float(data["Cmax"]),
                "WeightW": float(data["Alpha"]),
                "runtime_ms": first_nonempty(source, "runtime_ms", default=""),
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
                "DrivePath": first_nonempty(source, "drive_path", "DrivePath", default=""),
                "WalkPath": first_nonempty(source, "walk_path", "WalkPath", default=""),
            }
            if feasible:
                raw_row.update(
                    {
                        "ParkingNode": parking_node_text or (
                            str(reconstructed_spot["parking_node"]) if reconstructed_spot is not None else ""
                        ),
                        "ParkingSpotIndex": effective_spot_index,
                        "ParkingSpotId": first_nonempty(source, "parking_spot_id", "ParkingSpotId")
                        or (str(reconstructed_spot["parking_spot_id"]) if reconstructed_spot is not None else ""),
                        "SpotClass": first_nonempty(source, "spot_class", "SpotClass")
                        or (str(reconstructed_spot["spot_class"]) if reconstructed_spot is not None else ""),
                        "drive_time_hr": (
                            (str(reconstructed_spot["drive_time_hr"]) if reconstructed_spot is not None else "")
                            if prefer_reconstructed_metrics
                            else first_nonempty(source, "drive_time_hr")
                            or (str(reconstructed_spot["drive_time_hr"]) if reconstructed_spot is not None else "")
                        ),
                        "internal_drive_time_hr": (
                            first_nonempty(source, "internal_drive_time_hr")
                            or (str(reconstructed_spot["internal_drive_time_hr"]) if reconstructed_spot is not None else "")
                        ),
                        "internal_walk_time_hr": (
                            first_nonempty(source, "internal_walk_time_hr")
                            or (str(reconstructed_spot["internal_walk_time_hr"]) if reconstructed_spot is not None else "")
                        ),
                        "external_walk_distance_km": (
                            first_nonempty(source, "external_walk_distance_km", "walk_distance_km")
                            or (str(reconstructed_spot["external_walk_distance_km"]) if reconstructed_spot is not None else "")
                        ),
                        "external_walk_time_hr": (
                            first_nonempty(source, "external_walk_time_hr")
                            or (str(reconstructed_spot["external_walk_time_hr"]) if reconstructed_spot is not None else "")
                        ),
                        "walk_time_hr": (
                            (str(reconstructed_spot["walk_time_hr"]) if reconstructed_spot is not None else "")
                            if prefer_reconstructed_metrics
                            else first_nonempty(source, "walk_time_hr", "total_walk_time_hr")
                            or (str(reconstructed_spot["walk_time_hr"]) if reconstructed_spot is not None else "")
                        ),
                        "parking_cost": (
                            (str(reconstructed_spot["parking_cost"]) if reconstructed_spot is not None else "")
                            if prefer_reconstructed_metrics
                            else first_nonempty(source, "parking_cost")
                            or (str(reconstructed_spot["parking_cost"]) if reconstructed_spot is not None else "")
                        ),
                        "objective": (
                            (str(reconstructed_spot["objective"]) if reconstructed_spot is not None else "")
                            if prefer_reconstructed_metrics
                            else first_nonempty(source, "objective")
                            or (str(reconstructed_spot["objective"]) if reconstructed_spot is not None else "")
                        ),
                    }
                )
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

        previous = exact_rows.get(dataset_name)
        if previous is None and metadata["parameter_name"] in {"parking_spot_availability", "spots_per_lot"}:
            previous = exact_rows.get(baseline_case)
        if previous is None and metadata["parameter_name"] == "parking_budget_cmax":
            previous = exact_rows.get(baseline_case)
        if previous is None:
            best_source = best_available_source_row(available_method_sources)
            if best_source is not None:
                previous = {
                    "parking_node": first_nonempty(best_source, "parking_node", "ParkingNode"),
                    "alpha": first_nonempty(best_source, "alpha", "Alpha", default=str(data["Alpha"])),
                    "runtime_ms": first_nonempty(best_source, "runtime_ms", default=""),
                    "drive_path": first_nonempty(best_source, "drive_path", "DrivePath", default=""),
                    "walk_path": first_nonempty(best_source, "walk_path", "WalkPath", default=""),
                    "external_drive_time_hr": first_nonempty(best_source, "drive_time_hr", default="0"),
                }
        if previous is not None:
            parking_node_text = previous.get("parking_node", "")
            if parking_node_text:
                parking_node = int(float(parking_node_text))
                external_drive = float(previous.get("external_drive_time_hr", previous.get("drive_time_hr", "0")) or 0.0)
                spot = select_best_spot(data, external_drive, parking_node, float(previous.get("alpha", data["Alpha"])))
                if spot is not None:
                    row = canonical_optimization_row(metadata, data, previous, spot)
                    combined_rows.append(row)

    return raw_logs["GA"], raw_logs["PSO"], raw_logs["ACO"], combined_rows


def canonical_row_from_raw(
    method: str,
    raw: dict[str, Any],
    metadata: dict[str, Any],
    data: dict[str, Any],
    result_source: str,
) -> dict[str, Any]:
    solved = bool(raw.get("objective", ""))
    drive_time_hr = float_or_default(raw.get("drive_time_hr", ""), 0.0)
    internal_drive_time_hr = float_or_default(raw.get("internal_drive_time_hr", ""), 0.0)
    walk_time_hr = float_or_default(raw.get("walk_time_hr", ""), 0.0)
    parking_cost = float_or_default(raw.get("parking_cost", ""), 0.0)
    alpha = float(data["Alpha"])
    selected_available = ""
    if solved and raw.get("ParkingSpotIndex"):
        spot_token = str(raw["ParkingSpotIndex"]).strip()
        spot_index: int | None = None
        if spot_token.isdigit():
            spot_index = int(spot_token)
        else:
            try:
                spot_index = data["SpotIds"].index(spot_token) + 1
            except ValueError:
                spot_index = None
        if spot_index is not None and 1 <= spot_index <= len(data["SpotAvailable"]):
            selected_available = str(int(data["SpotAvailable"][spot_index - 1]))
    available_spots = sum(int(value) for value in data["SpotAvailable"])
    row = {
        **metadata,
        "method": method,
        "nodes": int(data["Nodes"]),
        "edges": len(data["E"]),
        "parking_lots": int(data["Q"]),
        "total_spots": int(data["TotalSpots"]),
        "available_spots": available_spots,
        "available_percentage": data["AvailablePercentage"],
        "feasible_spots": len([idx for idx in range(1, int(data["TotalSpots"]) + 1) if idx in []]),
        "source": int(data["S"]),
        "destination": int(data["D"]),
        "alpha": alpha,
        "maximum_walking_distance_km": float(data["W_km"]),
        "maximum_parking_cost": float(data["Cmax"]),
        "S": int(data["S"]),
        "D": int(data["D"]),
        "Alpha": alpha,
        "Wmax": float(data["W_km"]),
        "Cmax": float(data["Cmax"]),
        "w": alpha,
        "feasible": "TRUE" if solved else "FALSE",
        "solver_status": "solved" if solved else "no_feasible_spot",
        "parking_node": raw.get("ParkingNode", ""),
        "parking_spot_index": raw.get("ParkingSpotIndex", ""),
        "parking_spot_id": raw.get("ParkingSpotId", ""),
        "spot_class": raw.get("SpotClass", ""),
        "selected_spot_available": selected_available,
        "external_drive_time_hr": raw.get("drive_time_hr", ""),
        "internal_drive_time_hr": raw.get("internal_drive_time_hr", ""),
        "internal_walk_time_hr": raw.get("internal_walk_time_hr", ""),
        "external_walk_distance_km": raw.get("external_walk_distance_km", ""),
        "external_walk_time_hr": raw.get("external_walk_time_hr", ""),
        "total_walk_time_hr": raw.get("walk_time_hr", ""),
        "total_travel_time_hr": (
            (drive_time_hr + internal_drive_time_hr + walk_time_hr)
            if solved
            else ""
        ),
        "drive_time_hr": raw.get("drive_time_hr", ""),
        "walk_distance_km": raw.get("external_walk_distance_km", ""),
        "walk_time_hr": raw.get("walk_time_hr", ""),
        "travel_time_hr": (
            (drive_time_hr + internal_drive_time_hr + walk_time_hr)
            if solved
            else ""
        ),
        "parking_cost": raw.get("parking_cost", ""),
        "objective": (
            spot_objective_from_totals(alpha, drive_time_hr + internal_drive_time_hr + walk_time_hr, parking_cost)
            if solved
            else ""
        ),
        "runtime_ms": raw.get("runtime_ms", ""),
        "drive_path": raw.get("DrivePath", ""),
        "walk_path": raw.get("WalkPath", ""),
        "source_result_csv": result_source,
    }
    row["feasible_spots"] = sum(
        1 for spot_idx in range(1, int(data["TotalSpots"]) + 1) if int(data["SpotAvailable"][spot_idx - 1]) == 1
    )
    return row


def canonical_optimization_row(
    metadata: dict[str, Any],
    data: dict[str, Any],
    previous: dict[str, str],
    spot: dict[str, Any],
) -> dict[str, Any]:
    selected_available = str(int(data["SpotAvailable"][int(spot["parking_spot_index"]) - 1]))
    available_spots = sum(int(value) for value in data["SpotAvailable"])
    return {
        **metadata,
        "method": "Optimization",
        "nodes": int(data["Nodes"]),
        "edges": len(data["E"]),
        "parking_lots": int(data["Q"]),
        "total_spots": int(data["TotalSpots"]),
        "available_spots": available_spots,
        "available_percentage": data["AvailablePercentage"],
        "feasible_spots": available_spots,
        "source": int(data["S"]),
        "destination": int(data["D"]),
        "alpha": float(previous.get("alpha", data["Alpha"])),
        "maximum_walking_distance_km": float(data["W_km"]),
        "maximum_parking_cost": float(data["Cmax"]),
        "S": int(data["S"]),
        "D": int(data["D"]),
        "Alpha": float(previous.get("alpha", data["Alpha"])),
        "Wmax": float(data["W_km"]),
        "Cmax": float(data["Cmax"]),
        "w": float(previous.get("alpha", data["Alpha"])),
        "feasible": "TRUE",
        "solver_status": "inferred_from_previous_exact_lot_solution",
        "parking_node": spot["parking_node"],
        "parking_spot_index": spot["parking_spot_index"],
        "parking_spot_id": spot["parking_spot_id"],
        "spot_class": spot["spot_class"],
        "selected_spot_available": selected_available,
        "external_drive_time_hr": spot["drive_time_hr"],
        "internal_drive_time_hr": spot["internal_drive_time_hr"],
        "internal_walk_time_hr": spot["internal_walk_time_hr"],
        "external_walk_distance_km": spot["external_walk_distance_km"],
        "external_walk_time_hr": spot["external_walk_time_hr"],
        "total_walk_time_hr": spot["walk_time_hr"],
        "total_travel_time_hr": spot["travel_time_hr"],
        "drive_time_hr": spot["drive_time_hr"],
        "walk_distance_km": spot["external_walk_distance_km"],
        "walk_time_hr": spot["walk_time_hr"],
        "travel_time_hr": spot["travel_time_hr"],
        "parking_cost": spot["parking_cost"],
        "objective": spot["objective"],
        "runtime_ms": previous.get("runtime_ms", ""),
        "drive_path": previous.get("drive_path", ""),
        "walk_path": previous.get("walk_path", ""),
        "source_result_csv": "RESULTS/COMBINED_ALL_METHODS/results_all_methods_parameter_sensitivity_spot_level.csv",
    }


def write_raw_logs(ga_rows: list[dict[str, Any]], pso_rows: list[dict[str, Any]], aco_rows: list[dict[str, Any]]) -> None:
    write_csv(GA_LOG, ga_rows, RAW_LOG_FIELDS)
    write_csv(PSO_LOG, pso_rows, RAW_LOG_FIELDS)
    write_csv(ACO_LOG, aco_rows, RAW_LOG_FIELDS)


def write_combined_outputs(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: (int(row["nodes"]), row["dataset_group"], row["dataset_name"], row["method"]))
    write_csv(ALL_SPOT_CSV, rows, CANONICAL_FIELDS)
    write_csv(PARAM_SPOT_CSV, [row for row in rows if row["dataset_group"] == "parameter_sensitivity"], CANONICAL_FIELDS)
    write_csv(FINAL_SPOT_CSV, [row for row in rows if row["dataset_group"] == "final_case"], CANONICAL_FIELDS)


def load_combined_outputs() -> list[dict[str, str]]:
    if not ALL_SPOT_CSV.exists():
        raise FileNotFoundError(f"Combined results file not found: {ALL_SPOT_CSV}")
    rows = read_csv(ALL_SPOT_CSV)
    if not rows:
        raise RuntimeError("Combined results file is empty.")
    return rows


def validate_raw_logs_complete(dataset_index: dict[str, tuple[Path, str, dict[str, Any]]]) -> None:
    for method, path in (("GA", GA_LOG), ("PSO", PSO_LOG), ("ACO", ACO_LOG)):
        rows = read_csv(path)
        case_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            case = str(row.get("Case", "")).strip()
            if case:
                case_counts[case] += 1
        duplicates = {case: count for case, count in case_counts.items() if count > 1}
        extras = sorted(set(case_counts) - set(dataset_index))
        if duplicates or extras:
            details: list[str] = []
            if extras:
                details.append(f"extras={extras[:8]}")
            if duplicates:
                details.append(f"duplicates={list(duplicates.items())[:8]}")
            raise RuntimeError(f"{method} raw log is not export-ready: " + ", ".join(details))


def validate_combined_outputs(rows: list[dict[str, Any]], dataset_index: dict[str, tuple[Path, str, dict[str, Any]]]) -> None:
    required_columns = set(CANONICAL_FIELDS)
    if not rows:
        raise RuntimeError("Combined spot-level output is empty.")
    missing_columns = [column for column in CANONICAL_FIELDS if column not in rows[0]]
    if missing_columns:
        raise RuntimeError(f"Combined spot-level output is missing required columns: {missing_columns}")
    for row in rows:
        if not is_canonical_spot_level_source_row(row):
            raise RuntimeError(
                f"Combined spot-level output rejected noncanonical feasible row: method={row.get('method', '')}, dataset={row.get('dataset_name', '')}"
            )
    expected_cases = set(dataset_index)
    for method in ("GA", "PSO", "ACO"):
        method_rows = [row for row in rows if row["method"] == method]
        case_counts: dict[str, int] = defaultdict(int)
        for row in method_rows:
            case_counts[str(row.get("dataset_name", "")).strip()] += 1
            row_columns = set(row)
            if row_columns != required_columns:
                missing = sorted(required_columns - row_columns)
                extra = sorted(row_columns - required_columns)
                raise RuntimeError(f"{method} combined row schema mismatch: missing={missing}, extra={extra}")
            if not is_canonical_spot_level_source_row(row):
                raise RuntimeError(
                    f"{method} combined row rejected as noncanonical spot-level data: dataset={row.get('dataset_name', '')}"
                )
        duplicates = {case: count for case, count in case_counts.items() if count > 1}
        if duplicates:
            details: list[str] = []
            if duplicates:
                details.append(f"duplicates={list(duplicates.items())[:8]}")
            raise RuntimeError(f"{method} combined rows failed validation: " + ", ".join(details))


def format_parameter_label(value: str) -> str:
    if "-" in value and all(math.isfinite(float_or_default(part.strip(), math.inf)) for part in value.split("-", 1)):
        lo, _hi = [part.strip() for part in value.split("-", 1)]
        return lo
    if value in ("very_low", "low", "medium", "high"):
        return value.replace("_", " ")
    return value


def plot_value(row: dict[str, Any], parameter_name: str) -> str:
    if parameter_name == "node_size":
        return str(int(float(row["nodes"])))
    return row["parameter_value"]


def graph_source_key(row: dict[str, Any], parameter_label: str | None = None, experiment_name: str | None = None) -> str:
    label = parameter_label if parameter_label is not None else plot_value(row, str(row.get("parameter_name", "")))
    experiment = experiment_name if experiment_name is not None else str(row.get("parameter_name", ""))
    source_case = str(row.get("dataset_name") or row.get("dat_file") or "")
    return "|".join(
        [
            str(row.get("node_scale", "")),
            experiment,
            str(row.get("dataset_group", "")),
            str(int(float(row["nodes"])) if row.get("nodes") not in (None, "") else ""),
            str(label),
            str(row.get("method", "")),
            source_case,
        ]
    )


def parameter_order(parameter_name: str, rows: list[dict[str, Any]]) -> list[str]:
    def lower_bound_key(value: str) -> tuple[float, str]:
        number = float_or_default(str(value).split("-", 1)[0].strip(), math.inf)
        return (number, str(value))

    if parameter_name == "node_size":
        return [str(value) for value in sorted({int(float(row["nodes"])) for row in rows})]
    if parameter_name == "edge_density":
        preferred = ["Baseline", "Dense", "Denser", "baseline", "dense", "denser"]
    elif parameter_name == "maximum_walking_distance":
        preferred = ["0.25", "0.50", "0.75", "1.00", "0.2", "0.5", "0.8", "1.0"]
    elif parameter_name == "parking_spot_availability":
        preferred = ["10%", "20%", "30%", "40%"]
    elif parameter_name == "spots_per_lot":
        preferred = ["10-20", "20-40", "40-60", "60-80"]
    elif parameter_name == "traffic_level":
        preferred = ["very_low", "low", "medium", "high"]
    elif parameter_name == "parking_budget_cmax":
        preferred = ["10.0", "15.0", "20.5", "25.0"]
    elif parameter_name == "parking_cost_range":
        preferred = ["8.0-20.5", "10.0-22.5", "12.5-25.0", "15.0-30.0", "8-20.5", "10-22.5", "12.5-25", "15-30"]
    else:
        preferred = []
    present = [row["parameter_value"] for row in rows]
    ordered = list(dict.fromkeys(value for value in preferred if value in present))
    if parameter_name in {"spots_per_lot", "parking_cost_range"}:
        return sorted(dict.fromkeys(present), key=lower_bound_key)
    return ordered or sorted(dict.fromkeys(present))


def plot_three_metrics(scale_key: str, title_prefix: str, rows: list[dict[str, Any]], parameter_name: str, output_stem: str) -> None:
    if not rows:
        return

    def add_segment_label(
        rect: Any,
        label_text: str,
        label_color: str,
        minimum_label_height: float,
        reserved: dict[float, float],
        *,
        inside_size: float,
        small_size: float,
        outside_offset: float,
        outside_gap: float,
    ) -> float:
        height = float(rect.get_height())
        if not np.isfinite(height) or height <= 0:
            return float(rect.get_y())
        label_x = float(rect.get_x() + rect.get_width() / 2.0)
        label_y = float(rect.get_y() + rect.get_height() / 2.0)
        bar_key = round(label_x, 8)
        if height >= minimum_label_height:
            ax.text(label_x, label_y, label_text, ha="center", va="center", rotation=0, fontsize=inside_size, color=label_color, fontweight="bold", linespacing=0.95)
            reserved[bar_key] = max(reserved.get(bar_key, 0.0), float(rect.get_y() + rect.get_height()))
            return float(rect.get_y() + rect.get_height())
        if height >= minimum_label_height * 0.62:
            ax.text(label_x, label_y, label_text, ha="center", va="center", rotation=0, fontsize=small_size, color=label_color, fontweight="bold", linespacing=0.95)
            reserved[bar_key] = max(reserved.get(bar_key, 0.0), float(rect.get_y() + rect.get_height()))
            return float(rect.get_y() + rect.get_height())
        y_text = max(float(rect.get_y() + rect.get_height()) + outside_offset, reserved.get(bar_key, 0.0) + outside_gap)
        ax.text(
            label_x,
            y_text,
            label_text,
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=small_size,
            color="#0f172a",
            fontweight="bold",
            linespacing=0.95,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
        )
        reserved[bar_key] = y_text
        return y_text

    def mark_tiny_component(rect: Any, color: str, minimum_label_height: float) -> None:
        height = float(rect.get_height())
        if not np.isfinite(height) or height <= 0 or height >= minimum_label_height * 0.25:
            return
        y_boundary = float(rect.get_y())
        x_left = float(rect.get_x() + rect.get_width() * 0.08)
        x_right = float(rect.get_x() + rect.get_width() * 0.92)
        ax.hlines(y_boundary, x_left, x_right, colors="white", linewidth=3.8, zorder=6)
        ax.hlines(y_boundary, x_left, x_right, colors=color, linewidth=1.9, zorder=7)

    def add_tiny_cost_label(rect: Any, label_text: str, minimum_label_height: float, reserved: dict[float, float]) -> float:
        y_top = float(rect.get_y() + rect.get_height())
        label_x = float(rect.get_x() + rect.get_width() / 2.0)
        label_y = y_top - max(minimum_label_height * 0.70, 0.085)
        bar_key = round(label_x, 8)
        ax.text(
            label_x,
            label_y,
            label_text,
            ha="center",
            va="center",
            rotation=0,
            fontsize=7.8,
            color="#0f172a",
            fontweight="bold",
            linespacing=0.95,
        )
        reserved[bar_key] = max(reserved.get(bar_key, 0.0), y_top)
        return y_top

    preferred_methods = ("Optimization", "Heuristic", "ACO", "PSO", "GA")
    method_order = [method for method in preferred_methods if any(row["method"] == method for row in rows)]
    values = parameter_order(parameter_name, rows)
    data_by_pair = {(plot_value(row, parameter_name), row["method"]): row for row in rows}
    csv_rows: list[dict[str, Any]] = []
    fig_width = max(12.0, 2.8 + len(values) * 1.7 + len(method_order) * 1.1)
    fig, ax = plt.subplots(figsize=(min(fig_width, 18.0), 6.0))
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.9, alpha=0.9)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    if len(method_order) >= 4:
        method_width = 0.18
        inner_gap = 0.05
        group_gap = 0.56
    else:
        method_width = 0.22
        inner_gap = 0.055
        group_gap = 0.48
    group_span = len(method_order) * method_width + max(0, len(method_order) - 1) * inner_gap
    x = np.arange(len(values), dtype=float) * (group_span + group_gap)
    max_total = 0.0
    max_label_top = 0.0
    legend_handles = []
    legend_labels = []
    objective_entries: list[dict[str, float]] = []
    reserved_tops: dict[float, float] = {}

    for method_index, method in enumerate(method_order):
        positions = x - group_span / 2.0 + method_width / 2.0 + method_index * (method_width + inner_gap)
        drive_color, walk_color, cost_color = BASELINE_STYLE_METHOD_COLORS[method]
        edge_color = BASELINE_STYLE_METHOD_EDGES[method]
        drive = []
        walk = []
        cost_component = []
        for value in values:
            row = data_by_pair.get((value, method))
            if not row or row["feasible"] != "TRUE":
                drive.append(np.nan)
                walk.append(np.nan)
                cost_component.append(np.nan)
                continue
            drive_min = float_or_default(row["external_drive_time_hr"], 0.0) * 60.0 + float_or_default(row["internal_drive_time_hr"], 0.0) * 60.0
            walk_min = float_or_default(row["total_walk_time_hr"], 0.0) * 60.0
            parking_cost = float_or_default(row["parking_cost"], 0.0)
            alpha = float_or_default(row["Alpha"], float_or_default(row["alpha"], 0.0))
            drive_vis = drive_min
            walk_vis = walk_min
            cost_vis = parking_cost
            drive.append(drive_vis)
            walk.append(walk_vis)
            cost_component.append(cost_vis)
            displayed_total = drive_vis + walk_vis + cost_vis
            max_total = max(max_total, displayed_total)
            csv_rows.append(
                {
                    "graph_source_key": graph_source_key(row, value, parameter_name),
                    "experiment_scale": scale_key,
                    "experiment_name": parameter_name,
                    "experiment_group": row["dataset_group"],
                    "node_size": str(int(float(row["nodes"]))),
                    "parameter_label": value,
                    "method_key": row["method"],
                    "source_case": row["dataset_name"],
                    "dat_file": row["dat_file"],
                    "x_value": value,
                    "x_label": format_parameter_label(value),
                    "method": METHOD_LABELS[method],
                    "alpha": f"{alpha:.6f}",
                    "driving_time_min": f"{drive_min:.6f}",
                    "walking_time_min": f"{walk_min:.6f}",
                    "parking_cost_usd": f"{parking_cost:.6f}",
                    "displayed_driving_component": f"{drive_vis:.6f}",
                    "displayed_walking_component": f"{walk_vis:.6f}",
                    "displayed_cost_component": f"{cost_vis:.6f}",
                    "displayed_total": f"{displayed_total:.6f}",
                    "original_objective": f"{spot_objective_from_totals(alpha, (drive_min + walk_min) / 60.0, parking_cost):.6f}",
                }
            )
        bars1 = ax.bar(
            positions,
            drive,
            width=method_width,
            color=drive_color,
            edgecolor=edge_color,
            linewidth=2.0,
            label=f"{METHOD_LABELS[method]} - Driving",
        )
        bars2 = ax.bar(
            positions,
            walk,
            width=method_width,
            bottom=drive,
            color=walk_color,
            edgecolor=edge_color,
            linewidth=2.0,
            hatch="..",
            label=f"{METHOD_LABELS[method]} - Walking",
        )
        bars3 = ax.bar(
            positions,
            cost_component,
            width=method_width,
            bottom=np.array(drive) + np.array(walk),
            color=cost_color,
            edgecolor=edge_color,
            linewidth=2.0,
            hatch="///",
            label=f"{METHOD_LABELS[method]} - Cost",
        )
        legend_handles.extend([bars1[0], bars2[0], bars3[0]])
        legend_labels.extend(
            [
                f"{METHOD_LABELS[method]} - Driving",
                f"{METHOD_LABELS[method]} - Walking",
                f"{METHOD_LABELS[method]} - Cost",
            ]
        )

        if max_total > 0:
            minimum_label_height = max_total * 0.04
            outside_offset = max(max_total * 0.015, 0.035)
            outside_gap = max(max_total * 0.016, 0.035)
            for pos_idx, (drive_rect, walk_rect, cost_rect) in enumerate(zip(bars1.patches, bars2.patches, bars3.patches)):
                if pos_idx >= len(drive) or not np.isfinite(drive[pos_idx]) or not np.isfinite(walk[pos_idx]) or not np.isfinite(cost_component[pos_idx]):
                    continue
                drive_vis = float(drive[pos_idx])
                walk_vis = float(walk[pos_idx])
                cost_vis = float(cost_component[pos_idx])
                row_value = values[pos_idx]
                row = data_by_pair.get((row_value, method))
                if not row:
                    continue
                drive_min = float_or_default(row["external_drive_time_hr"], 0.0) * 60.0 + float_or_default(row["internal_drive_time_hr"], 0.0) * 60.0
                walk_min = float_or_default(row["total_walk_time_hr"], 0.0) * 60.0
                parking_cost = float_or_default(row["parking_cost"], 0.0)
                total_vis = drive_vis + walk_vis + cost_vis
                segments = [
                    (drive_rect, drive_vis, drive_min, "#111827"),
                    (walk_rect, walk_vis, walk_min, "#111827"),
                    (cost_rect, cost_vis, parking_cost, "white"),
                ]
                for rect, seg_height, raw_value, label_color in segments:
                    label_text = f"{raw_value:.2f}" if seg_height >= minimum_label_height * 0.80 else f"{raw_value:.1f}"
                    if rect is cost_rect and 0 < float(rect.get_height()) < minimum_label_height * 0.25:
                        mark_tiny_component(rect, edge_color, minimum_label_height)
                        label_top = add_tiny_cost_label(rect, label_text, minimum_label_height, reserved_tops)
                    else:
                        label_top = add_segment_label(
                            rect,
                            label_text,
                            label_color,
                            minimum_label_height,
                            reserved_tops,
                            inside_size=9.5,
                            small_size=7.8,
                            outside_offset=outside_offset,
                            outside_gap=outside_gap,
                        )
                    max_label_top = max(max_label_top, label_top)
                bar_center_x = float(cost_rect.get_x() + cost_rect.get_width() / 2.0)
                expected_x = float(cost_rect.get_x() + cost_rect.get_width() / 2.0)
                if abs(bar_center_x - expected_x) > 1e-9:
                    raise AssertionError("Objective label x-coordinate does not match the bar center.")
                objective_entries.append(
                    {
                        "x": bar_center_x,
                        "total": float(total_vis),
                        "edge": edge_color,
                        "objective": float(row["objective"]),
                    }
                )

    y_range = max(max_total, 1.0)
    objective_offset = 0.015 * y_range
    objective_gap = 0.022 * y_range
    objective_font_size = 10.5 if scale_key == "small_scale" else 10.0
    for item in objective_entries:
        bar_key = round(float(item["x"]), 8)
        y_pos = max(float(item["total"]) + objective_offset, reserved_tops.get(bar_key, float(item["total"])) + objective_gap)
        ax.text(
            float(item["x"]),
            y_pos,
            f"{item['objective']:.3f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=objective_font_size,
            color=item["edge"],
            fontweight="bold",
        )
        reserved_tops[bar_key] = y_pos
        max_label_top = max(max_label_top, y_pos)

    ax.set_xticks(x)
    ax.set_xticklabels([format_parameter_label(value) for value in values], fontsize=15, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=20, fontweight="bold")
    ax.set_xlabel(title_prefix, fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15, width=1.2, length=5)
    ax.legend(
        legend_handles,
        legend_labels,
        ncol=3 if len(method_order) <= 3 else 4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        fontsize=12,
        columnspacing=1.2,
        handletextpad=0.6,
        borderpad=0.7,
    )
    ylim_top = max(max_total * 1.18, max_label_top + 0.05 * y_range) if max_total else 1.0
    ax.set_ylim(0, ylim_top)
    ax.margins(x=0.08)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.78)

    target_dirs = [FIGURES_DIR / scale_key, OBJECTIVE_FIG_DIR / scale_key]
    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / f"{output_stem}_objective_components.csv"
        write_csv(
            csv_path,
            csv_rows,
            list(csv_rows[0].keys())
            if csv_rows
            else [
                "graph_source_key",
                "experiment_scale",
                "experiment_name",
                "experiment_group",
                "node_size",
                "parameter_label",
                "method_key",
                "source_case",
                "dat_file",
                "x_value",
                "x_label",
                "method",
                "alpha",
                "driving_time_min",
                "walking_time_min",
                "parking_cost_usd",
                "displayed_driving_component",
                "displayed_walking_component",
                "displayed_cost_component",
                "displayed_total",
                "original_objective",
            ],
        )
        for duplicate_path in (
            target_dir / f"{output_stem}_objective_components.png",
            target_dir / f"{output_stem}_objective_components.pdf",
        ):
            if duplicate_path.exists():
                duplicate_path.unlink()
        fig.savefig(target_dir / f"{output_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
        fig.savefig(target_dir / f"{output_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_runtime(scale_key: str, rows: list[dict[str, Any]], output_stem: str, label: str) -> None:
    if not rows:
        return
    method_order = [method for method in ("Optimization", "ACO", "PSO", "GA") if any(row["method"] == method for row in rows)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset_name"]].append(row)
    cases = sorted(grouped.values(), key=lambda group: int(group[0]["nodes"]))
    case_nodes = [int(case_rows[0]["nodes"]) for case_rows in cases]
    runtime_seconds_by_method: dict[str, list[float]] = {method: [] for method in method_order}
    runtime_minutes_by_method: dict[str, list[float]] = {method: [] for method in method_order}
    for method in method_order:
        for case_rows in cases:
            row = next((item for item in case_rows if item["method"] == method), None)
            runtime_seconds = float(row["runtime_ms"]) / 1000.0 if row and row["runtime_ms"] else np.nan
            runtime_seconds_by_method[method].append(runtime_seconds)
            runtime_minutes_by_method[method].append(runtime_seconds / 60.0 if np.isfinite(runtime_seconds) else np.nan)

    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    ax.set_facecolor("white")
    ax.grid(False)
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(True, which="major", color="#d1d5db", linewidth=1.0, alpha=0.95)
    ax.tick_params(labelsize=12.5, colors="#111827", width=1.1, length=4)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    if scale_key == "small_scale":
        legend_handles: list[Line2D] = []
        for method in method_order:
            minute_values = runtime_minutes_by_method[method]
            if not any(np.isfinite(value) for value in minute_values):
                continue
            ax.plot(
                case_nodes,
                minute_values,
                color=BASELINE_METHOD_COLORS[method],
                marker=RUNTIME_METHOD_MARKERS[method],
                linewidth=4.0 if method == "Optimization" else 3.6,
                markersize=12.0 if method == "Optimization" else 11.0,
                markerfacecolor="white",
                markeredgewidth=2.3,
                linestyle="-",
                label=METHOD_LABELS[method],
                zorder=5 if method == "Optimization" else 4,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=BASELINE_METHOD_COLORS[method],
                    marker=RUNTIME_METHOD_MARKERS[method],
                    linewidth=4.0 if method == "Optimization" else 3.6,
                    markersize=12.0 if method == "Optimization" else 11.0,
                    markerfacecolor="white",
                    markeredgewidth=2.3,
                    linestyle="-",
                    label=METHOD_LABELS[method],
                )
            )

        ax.set_xticks(case_nodes)
        ax.set_xticklabels([str(node) for node in case_nodes], fontsize=12.5, fontweight="bold")
        ax.set_xlim(min(case_nodes), max(case_nodes))
        max_runtime = max(
            value
            for values in runtime_minutes_by_method.values()
            for value in values
            if np.isfinite(value)
        )
        ax.set_ylim(0, max_runtime * 1.12 if max_runtime > 0 else 1.0)
        ax.set_ylabel("Runtime (minutes)", fontsize=16, fontweight="bold")
        ax.set_xlabel(label, fontsize=16, fontweight="bold")
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            ncol=1,
            frameon=True,
            edgecolor="#cbd5e1",
            facecolor="white",
            fontsize=10,
            borderpad=0.55,
            labelspacing=0.45,
            handlelength=2.0,
        )

        inset = inset_axes(
            ax,
            width="48%",
            height="44%",
            loc="lower left",
            bbox_to_anchor=(0.06, 0.15, 0.78, 0.78),
            bbox_transform=ax.transAxes,
            borderpad=0.6,
        )
        inset.set_facecolor("white")
        inset.grid(False)
        inset.xaxis.grid(False, which="both")
        inset.yaxis.grid(True, which="major", color="#d1d5db", linewidth=1.0, alpha=0.95)
        inset.tick_params(labelsize=10.0, colors="#111827", pad=2)
        for spine in inset.spines.values():
            spine.set_color("#cbd5e1")

        inset_methods = [method for method in ("ACO", "PSO", "GA") if method in method_order]
        inset_max = 0.0
        for method in inset_methods:
            minute_values = runtime_minutes_by_method[method]
            if not any(np.isfinite(value) for value in minute_values):
                continue
            inset.plot(
                case_nodes,
                minute_values,
                color=BASELINE_METHOD_COLORS[method],
                marker=RUNTIME_METHOD_MARKERS[method],
                linewidth=3.2,
                markersize=10.5,
                markerfacecolor="white",
                markeredgewidth=2.1,
                linestyle="-",
                label=method,
            )
            inset_max = max(inset_max, max(value for value in minute_values if np.isfinite(value)))

        inset.set_title("Scalable Methods", fontsize=12, pad=4)
        inset.set_xlabel("Nodes", fontsize=10, labelpad=3)
        inset.set_ylabel("Runtime (minutes)", fontsize=10, labelpad=3)
        inset.set_xticks(case_nodes)
        inset.set_xlim(min(case_nodes), max(case_nodes))
        if inset_max > 0:
            inset.set_ylim(0, inset_max * 1.15)
        inset.legend(
            loc="upper right",
            ncol=1,
            frameon=True,
            edgecolor="#cbd5e1",
            facecolor="white",
            fontsize=8,
            borderpad=0.45,
            labelspacing=0.32,
            handlelength=1.8,
            columnspacing=0.7,
        )
    else:
        x = np.arange(len(cases))
        width = 0.80 / max(1, len(method_order))
        for idx, method in enumerate(method_order):
            ax.bar(x - 0.40 + width * (idx + 0.5), runtime_seconds_by_method[method], width=width * 0.92, label=METHOD_LABELS[method])
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(case_rows[0]["nodes"])) if scale_key != "sumo" else "SUMO" for case_rows in cases], fontsize=12.5, fontweight="bold")
        ax.set_ylabel("Runtime (s)", fontsize=16, fontweight="bold")
        ax.set_xlabel(label, fontsize=16, fontweight="bold")
        ax.legend(fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.14, top=0.95)
    for target_dir in (FIGURES_DIR / scale_key, OBJECTIVE_FIG_DIR / scale_key):
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / f"{output_stem}.png", dpi=200)
        fig.savefig(target_dir / f"{output_stem}.pdf")
        write_csv(
            target_dir / f"{output_stem}_objective_components.csv",
            [
                {
                    "graph_source_key": graph_source_key(
                        next((item for item in case_rows if item["method"] == method), case_rows[0]),
                        str(int(case_rows[0]["nodes"])) if scale_key != "sumo" else "SUMO",
                        "runtime",
                    ),
                    "experiment_scale": scale_key,
                    "experiment_name": "runtime",
                    "experiment_group": case_rows[0]["dataset_group"],
                    "node_size": str(int(case_rows[0]["nodes"])),
                    "parameter_label": str(int(case_rows[0]["nodes"])) if scale_key != "sumo" else "SUMO",
                    "method_key": method,
                    "source_case": next((item for item in case_rows if item["method"] == method), case_rows[0])["dataset_name"],
                    "dat_file": next((item for item in case_rows if item["method"] == method), case_rows[0])["dat_file"],
                    "case": case_rows[0]["dataset_name"],
                    "nodes": case_rows[0]["nodes"],
                    "method": method,
                    "runtime_s": f"{float(next((item for item in case_rows if item['method'] == method), {'runtime_ms': 0})['runtime_ms'] or 0) / 1000.0:.6f}",
                }
                for case_rows in cases
                for method in method_order
            ],
            ["graph_source_key", "experiment_scale", "experiment_name", "experiment_group", "node_size", "parameter_label", "method_key", "source_case", "dat_file", "case", "nodes", "method", "runtime_s"],
        )
    plt.close(fig)


def plot_internal_components(scale_key: str, rows: list[dict[str, Any]], output_stem: str, label: str) -> None:
    if not rows:
        return
    method_order = [method for method in ("Optimization", "ACO", "PSO", "GA") if any(row["method"] == method for row in rows)]
    x = np.arange(len(rows) // max(1, len(method_order)))
    cases = sorted({row["dataset_name"]: row for row in rows if row["method"] == method_order[0]}.values(), key=lambda row: int(row["nodes"]))
    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    width = 0.80 / max(1, len(method_order))
    for idx, method in enumerate(method_order):
        ext_drive, int_drive, int_walk, ext_walk = [], [], [], []
        for case in cases:
            row = next((item for item in rows if item["dataset_name"] == case["dataset_name"] and item["method"] == method), None)
            if not row or row["feasible"] != "TRUE":
                ext_drive.append(np.nan)
                int_drive.append(np.nan)
                int_walk.append(np.nan)
                ext_walk.append(np.nan)
                continue
            ext_drive.append(float_or_default(row["external_drive_time_hr"], 0.0) * 60.0)
            int_drive.append(float_or_default(row["internal_drive_time_hr"], 0.0) * 60.0)
            int_walk.append(float_or_default(row["internal_walk_time_hr"], 0.0) * 60.0)
            ext_walk.append(float_or_default(row["external_walk_time_hr"], 0.0) * 60.0)
        pos = np.arange(len(cases)) - 0.40 + width * (idx + 0.5)
        ax.bar(pos, ext_drive, width=width * 0.92, color="#b7cadb", edgecolor="#26415f", label=f"{METHOD_LABELS[method]} - External drive" if idx == 0 else None)
        ax.bar(pos, int_drive, width=width * 0.92, bottom=ext_drive, color="#8fb2c2", edgecolor="#26415f", label=f"{METHOD_LABELS[method]} - Internal drive" if idx == 0 else None)
        ax.bar(pos, int_walk, width=width * 0.92, bottom=np.array(ext_drive) + np.array(int_drive), color="#b7d7cf", edgecolor="#1c5f5a", label=f"{METHOD_LABELS[method]} - Internal walk" if idx == 0 else None)
        ax.bar(pos, ext_walk, width=width * 0.92, bottom=np.array(ext_drive) + np.array(int_drive) + np.array(int_walk), color="#77a79b", edgecolor="#1c5f5a", label=f"{METHOD_LABELS[method]} - External walk" if idx == 0 else None)
    ax.set_xticks(np.arange(len(cases)))
    ax.set_xticklabels([str(int(case["nodes"])) if scale_key != "sumo" else "SUMO" for case in cases], fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Time (min)", fontsize=16, fontweight="bold")
    ax.set_xlabel(label, fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12, width=1.1, length=4)
    ax.grid(axis="y", alpha=0.30)
    ax.legend(loc="upper left", ncol=2, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.14, top=0.95)
    for target_dir in (FIGURES_DIR / scale_key, OBJECTIVE_FIG_DIR / scale_key):
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / f"{output_stem}.png", dpi=200)
        fig.savefig(target_dir / f"{output_stem}.pdf")
    plt.close(fig)


def generate_figures(rows: list[dict[str, Any]]) -> None:
    final_rows = [row for row in rows if row["dataset_group"] == "final_case"]
    parameter_rows = [row for row in rows if row["dataset_group"] == "parameter_sensitivity"]

    small_baseline = [row for row in final_rows if int(row["nodes"]) in {10, 20, 30, 40} and row["parameter_name"] == "baseline"]
    medium_baseline = [row for row in final_rows if int(row["nodes"]) in {500, 1000, 1500, 2000, 2500} and row["parameter_name"] == "baseline"]
    sumo_baseline = [row for row in final_rows if int(row["nodes"]) == 2055 and row["parameter_name"] == "baseline"]

    plot_three_metrics("small_scale", "Network Size (nodes)", small_baseline, "node_size", "small_scale_node_size_three_metrics")
    plot_three_metrics("medium_large", "Network Size (nodes)", medium_baseline, "node_size", "medium_large_node_size_three_metrics")
    plot_three_metrics("sumo", "SUMO Case", sumo_baseline, "node_size", "sumo_baseline_three_metrics")
    plot_runtime("small_scale", small_baseline, "small_scale_node_size_runtime", "Network Size (nodes)")
    plot_runtime("medium_large", medium_baseline, "medium_large_node_size_runtime", "Network Size (nodes)")
    plot_runtime("sumo", sumo_baseline, "sumo_baseline_runtime", "SUMO Case")
    plot_internal_components("small_scale", small_baseline, "small_scale_internal_time_components", "Network Size (nodes)")
    plot_internal_components("medium_large", medium_baseline, "medium_large_internal_time_components", "Network Size (nodes)")
    plot_internal_components("sumo", sumo_baseline, "sumo_internal_time_components", "SUMO Case")

    scale_map = {
        "small": "small_scale",
        "medium_large": "medium_large",
        "SUMO": "sumo",
    }
    stem_prefix = {
        "small_scale": "small_scale",
        "medium_large": "medium_large",
        "sumo": "sumo",
    }
    parameters = [
        "edge_density",
        "maximum_walking_distance",
        "parking_spot_availability",
        "spots_per_lot",
        "traffic_level",
        "parking_budget_cmax",
        "parking_cost_range",
    ]
    for scale_category, scale_key in scale_map.items():
        scale_rows = [row for row in parameter_rows if row["dataset_relative_path"].split("/")[1] == scale_category]
        for parameter_name in parameters:
            parameter_group = [row for row in scale_rows if row["parameter_name"] == parameter_name]
            if not parameter_group:
                continue
            output_name = f"{stem_prefix[scale_key]}_{parameter_name}_three_metrics"
            plot_three_metrics(scale_key, parameter_name.replace("_", " ").title(), parameter_group, parameter_name, output_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild spot-level combined outputs and graph sources.")
    parser.add_argument(
        "--mode",
        choices=("export-only", "repair-and-export"),
        default="export-only",
        help="`export-only` treats GA/PSO/ACO raw logs as authoritative and never reruns solvers.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "combined", "figures"),
        default="all",
        help="`combined` rebuilds raw logs and combined CSVs only; `figures` regenerates graph-source CSVs and PDFs/PNGs from the existing combined CSVs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_index = build_spot_datasets()
    if args.stage in {"all", "combined"}:
        if args.mode == "repair-and-export":
            restore_raw_method_logs(dataset_index)
        else:
            validate_raw_logs_complete(dataset_index)
        source_rows, exact_rows = lot_level_sources()
        ga_rows, pso_rows, aco_rows, combined_rows = build_raw_method_logs(dataset_index, source_rows, exact_rows)
        write_raw_logs(ga_rows, pso_rows, aco_rows)
        write_combined_outputs(combined_rows)
        validate_combined_outputs(combined_rows, dataset_index)
        print(f"Regenerated {len(dataset_index)} spot-level datasets in {SPOT_DIR}")
        print(f"Wrote combined results to {ALL_SPOT_CSV}")
    if args.stage in {"all", "figures"}:
        combined_rows = load_combined_outputs()
        generate_figures(combined_rows)
        print(f"Regenerated extension figure-source CSVs and figures from {ALL_SPOT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
