#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
BASELINE_MASTER_CSV = ROOT / "parking-lot-recommendation-baseline" / "RESULTS" / "master_results_all_methods_for_graphs.csv"
BASELINE_PS_MANIFEST_CSV = ROOT / "parking-lot-recommendation-baseline" / "RESULTS" / "VALIDATION" / "parameter_sensitivity_manifest.csv"
BASELINE_EXACT_SMALL_CSV = ROOT / "parking-lot-recommendation-baseline" / "OPTIMIZATION" / "optimisation_results_summary_exact_small.csv"
EXTENSION_ROOT = ROOT / "parking-lot-recommendation-extension"
EXTENSION_ALL_SPOT_CSV = EXTENSION_ROOT / "RESULTS" / "COMBINED_ALL_METHODS" / "results_all_methods_spot_level.csv"
EXTENSION_PS_MANIFEST_CSV = EXTENSION_ROOT / "RESULTS" / "VALIDATION" / "parameter_sensitivity_manifest.csv"
EXTENSION_SOURCE_DIR = ROOT / "parking-lot-recommendation-extension" / "RESULTS" / "FIGURES" / "spot_level"
FINAL_BASELINE_DIR = ROOT / "FINAL_GRAPHS" / "baseline"
FINAL_EXTENSION_DIR = ROOT / "FINAL_GRAPHS" / "extension"
FINAL_GRAPH_SOURCE_AUDIT_CSV = ROOT / "FINAL_GRAPHS" / "final_graph_source_mapping_audit.csv"
sys.path.insert(0, str(EXTENSION_ROOT))
sys.path.insert(0, str(ROOT))

from spot_level_utils import spot_objective
from run_corrected_extension_objective_rerun import graph_scope_datasets

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
TRAFFIC_ORDER = {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3}

METHOD_ORDER = ["Optimization", "Heuristic", "ACO", "PSO", "GA"]
BASELINE_METHOD_ORDER_BY_SCALE = {
    "small": ["Optimization", "Heuristic"],
    "medium_large": ["Heuristic", "ACO", "PSO", "GA"],
    "sumo": ["Heuristic", "ACO", "PSO", "GA"],
}
EXTENSION_METHOD_ORDER_BY_SCALE = {
    "small_scale": ["Optimization", "ACO", "PSO", "GA"],
    "medium_large": ["ACO", "PSO", "GA"],
    "sumo": ["ACO", "PSO", "GA"],
}
METHOD_LABELS = {
    "Optimization": "Optimization (Optimum)",
    "Heuristic": "Heuristic",
    "ACO": "ACO",
    "PSO": "PSO",
    "GA": "GA",
}
METHOD_COLORS = {
    "Optimization": "#1f3a5f",
    "Heuristic": "#0f766e",
    "ACO": "#1d4ed8",
    "PSO": "#b45309",
    "GA": "#be123c",
}
COMPONENT_STYLES = {
    "Driving": {"hatch": None, "alpha": 0.96},
    "Walking": {"hatch": "..", "alpha": 0.82},
    "Cost": {"hatch": "///", "alpha": 0.68},
}
EXPERIMENT_LABELS = {
    "network_size": "Network Size (Nodes)",
    "node_size": "Network Size",
    "edge_density": "Edge Density",
    "parking_supply": "Parking Supply",
    "parking_spot_availability": "Parking Spot Availability",
    "spot_availability": "Spot Availability",
    "parking_budget": "Parking Budget",
    "parking_budget_cmax": "Parking Budget",
    "parking_cost_range": "Parking Cost Values",
    "parking_cost_variation_mode": "Parking Cost Values",
    "spots_per_lot": "Spots per Lot",
    "maximum_walking_distance": "Maximum Walking Distance (km)",
    "traffic_level": "Traffic Level",
    "objective_weight_alpha": "Objective Weight Alpha",
    "baseline": "Baseline",
}
CSV_FIELDS = [
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
    "objective_value",
]

AUTHORITATIVE_EXTENSION_GRAPH_SCOPE = graph_scope_datasets()


def parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def canonical_method(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    upper = text.upper()
    if "OPT" in upper:
        return "Optimization"
    if "HEUR" in upper:
        return "Heuristic"
    if "ACO" in upper:
        return "ACO"
    if "PSO" in upper:
        return "PSO"
    if "GA" in upper:
        return "GA"
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extension_parameter_value_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in read_csv(EXTENSION_PS_MANIFEST_CSV):
        if str(row.get("experiment_type", "")).strip() != "parameter_sensitivity":
            continue
        dataset_name = str(row.get("dataset_name", "")).strip()
        active_value = str(row.get("active_parameter_value", "")).strip()
        if dataset_name and active_value:
            mapping[dataset_name] = active_value
    return mapping


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def lower_bound_label(value: object) -> str:
    numbers = re.findall(FLOAT_RE, str(value or ""))
    return numbers[0] if numbers else str(value or "")


def sort_key(experiment: str, value: str) -> tuple:
    if experiment == "edge_density":
        nums = [float(num) for num in re.findall(FLOAT_RE, value)]
        return tuple(nums or [-math.inf])
    if experiment in {"network_size", "node_size", "parking_supply", "spot_availability", "parking_spot_availability", "spots_per_lot"}:
        nums = [float(num) for num in re.findall(FLOAT_RE, value)]
        return tuple(nums or [math.inf])
    if experiment in {"maximum_walking_distance", "parking_budget", "parking_budget_cmax", "objective_weight_alpha"}:
        number = parse_float(lower_bound_label(value))
        return (number if number is not None else math.inf,)
    if experiment in {"parking_cost_range", "parking_cost_variation_mode"}:
        nums = [float(num) for num in re.findall(FLOAT_RE, value)]
        return tuple(nums or [math.inf])
    if experiment == "traffic_level":
        normalized = str(value).replace("_", " ").strip().title()
        return (TRAFFIC_ORDER.get(normalized, 99),)
    return (str(value),)


def display_x_label(experiment: str, value: str, fallback: str) -> str:
    if experiment in {"parking_cost_range", "parking_cost_variation_mode", "spots_per_lot"}:
        return lower_bound_label(fallback or value)
    return fallback or value


def baseline_ps_expected_values() -> dict[tuple[str, str], list[str]]:
    if not BASELINE_PS_MANIFEST_CSV.exists():
        return {}
    scale_map = {"small": "small", "medium_large": "medium_large", "SUMO": "sumo"}
    parameter_map = {
        "edge_density": "edge_density",
        "parking_supply": "parking_supply",
        "maximum_walking_distance": "maximum_walking_distance",
        "traffic_level": "traffic_level",
        "parking_budget_cmax": "parking_budget",
        "parking_cost_variation_mode": "parking_cost_range",
    }
    expected: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in read_csv(BASELINE_PS_MANIFEST_CSV):
        scale = scale_map.get(str(row.get("scale_category", "")).strip())
        experiment = parameter_map.get(str(row.get("varied_parameter", "")).strip())
        if not scale or not experiment:
            continue
        value = str(row.get("active_parameter_value", "")).strip()
        if not value:
            continue
        label = value
        if experiment == "parking_supply":
            label = value
        elif experiment == "maximum_walking_distance":
            label = f"{float(value):.2f}"
        elif experiment == "traffic_level":
            label = value.replace("_", " ").title()
        elif experiment == "parking_budget":
            label = value
        elif experiment == "parking_cost_range":
            label = value
        expected[(scale, experiment)].append(display_x_label(experiment, value, label))
    return {key: sorted(set(values), key=lambda item: sort_key(key[1], item)) for key, values in expected.items()}


def validate_baseline_small_wmax_exact_coverage() -> None:
    expected_cases = {
        "N40_PS_walking_distance_W025_I1.dat",
        "N40_PS_walking_distance_W050_I1.dat",
        "N40_PS_walking_distance_W075_I1.dat",
        "N40_PS_walking_distance_W100_I1.dat",
    }
    exact_rows = {
        str(row.get("dat_file", "")).strip()
        for row in read_csv(BASELINE_EXACT_SMALL_CSV)
        if str(row.get("method", "")).strip() == "Optimization"
        and str(row.get("parameter_name", "")).strip() == "maximum_walking_distance"
    }
    missing = sorted(expected_cases - exact_rows)
    if missing:
        raise RuntimeError(
            "Missing baseline exact small maximum_walking_distance coverage: "
            f"{missing}"
        )


def baseline_graph_rows() -> dict[tuple[str, str], list[dict[str, str]]]:
    rows = read_csv(BASELINE_MASTER_CSV)
    grouped: dict[tuple[str, str], dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if str(row.get("feasible", "")).upper() != "TRUE":
            continue
        scale = str(row.get("scale", "")).strip()
        experiment = str(row.get("experiment_name", "")).strip()
        method = canonical_method(row.get("method"))
        if not scale or not experiment or method is None:
            continue
        if method not in BASELINE_METHOD_ORDER_BY_SCALE.get(scale, METHOD_ORDER):
            continue
        dataset_group = str(row.get("dataset_group", "")).strip()
        if experiment == "network_size":
            if dataset_group != "baseline":
                continue
        elif dataset_group != "parameter_sensitivity":
            continue

        alpha = parse_float(row.get("alpha_used")) or parse_float(row.get("Alpha")) or parse_float(row.get("w"))
        drive = parse_float(row.get("drive_time_min_used"))
        walk = parse_float(row.get("walk_time_min_used"))
        cost = parse_float(row.get("parking_cost"))
        if alpha is None or drive is None or walk is None or cost is None:
            continue
        drive_component = alpha * drive
        walk_component = alpha * walk
        cost_component = (1.0 - alpha) * cost
        total = drive_component + walk_component + cost_component
        x_value = str(row.get("parameter_value_key") or row.get("parameter_label") or "")
        x_label = str(row.get("parameter_label") or x_value)
        if not x_value:
            continue
        grouped[(scale, experiment)][(x_label, method)] = {
            "x_value": x_value,
            "x_label": display_x_label(experiment, x_value, x_label),
            "method": method,
            "alpha": f"{alpha:.6f}",
            "driving_time_min": f"{drive:.6f}",
            "walking_time_min": f"{walk:.6f}",
            "parking_cost_usd": f"{cost:.6f}",
            "displayed_driving_component": f"{drive_component:.6f}",
            "displayed_walking_component": f"{walk_component:.6f}",
            "displayed_cost_component": f"{cost_component:.6f}",
            "displayed_total": f"{total:.6f}",
            "objective_value": f"{total:.6f}",
            "source_dat_file": str(row.get("dat_file", "")).strip(),
            "source_dataset_group": dataset_group,
        }

    output: dict[tuple[str, str], list[dict[str, str]]] = {}
    for key, by_pair in grouped.items():
        output[key] = list(by_pair.values())

    expected = baseline_ps_expected_values()
    labels = expected.get(("small", "maximum_walking_distance"), [])
    rows_for_graph = output.get(("small", "maximum_walking_distance"), [])
    present = {(row["x_label"], row["method"]) for row in rows_for_graph}
    missing = [label for label in labels if (label, "Optimization") not in present]
    if missing:
        raise RuntimeError(
            "Missing baseline Optimization parameter-sensitivity coverage for "
            f"small/maximum_walking_distance: {missing}"
        )
    return output


def extension_graph_rows() -> dict[tuple[str, str], list[dict[str, str]]]:
    if not EXTENSION_ALL_SPOT_CSV.exists():
        return {}

    grouped: dict[tuple[str, str], dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    manifest_values = extension_parameter_value_map()
    scale_map = {
        "N10": "small_scale",
        "N20": "small_scale",
        "N30": "small_scale",
        "N40": "small_scale",
        "N50": "small_scale",
        "N500": "medium_large",
        "N1000": "medium_large",
        "N1500": "medium_large",
        "N2000": "medium_large",
        "N2500": "medium_large",
        "SUMO": "sumo",
    }
    experiment_map = {
        "spot_availability": "parking_spot_availability",
        "parking_budget": "parking_budget_cmax",
        "parking_cost_variation_mode": "parking_cost_range",
    }
    allowed_experiments = {
        "node_size",
        "edge_density",
        "parking_spot_availability",
        "spots_per_lot",
        "maximum_walking_distance",
        "traffic_level",
        "parking_budget_cmax",
        "parking_cost_range",
    }
    for row in read_csv(EXTENSION_ALL_SPOT_CSV):
        if str(row.get("feasible", "")).upper() != "TRUE":
            continue
        dat_file = str(row.get("dat_file", "")).strip()
        if AUTHORITATIVE_EXTENSION_GRAPH_SCOPE and dat_file not in AUTHORITATIVE_EXTENSION_GRAPH_SCOPE:
            continue
        method = canonical_method(row.get("method"))
        if method is None:
            continue
        scale = scale_map.get(str(row.get("node_scale", "")).strip())
        if scale is None:
            continue

        dataset_group = str(row.get("dataset_group", "")).strip()
        parameter_name = experiment_map.get(str(row.get("parameter_name", "")).strip(), str(row.get("parameter_name", "")).strip())
        if dataset_group == "final_case" and str(row.get("scenario_id", "")).strip() == "C1" and scale != "sumo":
            experiment = "node_size"
            x_value = str(int(parse_float(row.get("nodes")) or 0))
            x_label = x_value
        elif dataset_group == "parameter_sensitivity":
            experiment = parameter_name
            if experiment == "edge_density":
                x_value = (
                    manifest_values.get(dat_file, "")
                    or str(row.get("parameter_value", "")).strip()
                    or str(row.get("edges", "")).strip()
                    or str(row.get("scenario_name", "")).strip()
                )
            else:
                x_value = str(row.get("parameter_value", "")).strip() or str(row.get("scenario_name", "")).strip()
            x_label = x_value
        else:
            continue

        if experiment not in allowed_experiments:
            continue

        if not x_value:
            continue

        alpha = parse_float(row.get("alpha"))
        if alpha is None:
            alpha = parse_float(row.get("Alpha"))
        if alpha is None:
            continue

        external_drive_hr = parse_float(row.get("external_drive_time_hr")) or parse_float(row.get("drive_time_hr")) or 0.0
        internal_drive_hr = parse_float(row.get("internal_drive_time_hr")) or 0.0
        internal_walk_hr = parse_float(row.get("internal_walk_time_hr")) or 0.0
        external_walk_hr = parse_float(row.get("external_walk_time_hr")) or 0.0
        parking_cost = parse_float(row.get("parking_cost")) or 0.0

        drive_min = (external_drive_hr + internal_drive_hr) * 60.0
        walk_min = (internal_walk_hr + external_walk_hr) * 60.0
        objective = spot_objective(
            alpha,
            external_drive_hr,
            internal_drive_hr,
            internal_walk_hr,
            external_walk_hr,
            parking_cost,
        )

        grouped[(scale, experiment)][(display_x_label(experiment, x_value, x_label), method)] = {
            "x_value": x_value,
            "x_label": display_x_label(experiment, x_value, x_label),
            "method": method,
            "alpha": f"{alpha:.6f}",
            "driving_time_min": f"{drive_min:.6f}",
            "walking_time_min": f"{walk_min:.6f}",
            "parking_cost_usd": f"{parking_cost:.6f}",
            "displayed_driving_component": f"{drive_min:.6f}",
            "displayed_walking_component": f"{walk_min:.6f}",
            "displayed_cost_component": f"{parking_cost:.6f}",
            "displayed_total": f"{(drive_min + walk_min + parking_cost):.6f}",
            "objective_value": f"{objective:.6f}",
            "source_dat_file": dat_file,
            "source_dataset_group": dataset_group,
        }

    return {key: list(by_pair.values()) for key, by_pair in grouped.items()}


def validate_edge_density_graphs(
    baseline_rows: dict[tuple[str, str], list[dict[str, str]]],
    extension_rows: dict[tuple[str, str], list[dict[str, str]]],
) -> None:
    specs = [
        ("BASELINE", "small", baseline_rows.get(("small", "edge_density"), [])),
        ("BASELINE", "medium_large", baseline_rows.get(("medium_large", "edge_density"), [])),
        ("BASELINE", "sumo", baseline_rows.get(("sumo", "edge_density"), [])),
        ("EXTENSION", "small_scale", extension_rows.get(("small_scale", "edge_density"), [])),
        ("EXTENSION", "medium_large", extension_rows.get(("medium_large", "edge_density"), [])),
        ("EXTENSION", "sumo", extension_rows.get(("sumo", "edge_density"), [])),
    ]
    failures: list[str] = []
    for model, scale, rows in specs:
        values = sorted({str(row.get("x_label", "")).strip() for row in rows if str(row.get("x_label", "")).strip()}, key=lambda item: sort_key("edge_density", item))
        source_families = sorted({str(row.get("source_dataset_group", "")).strip() for row in rows if str(row.get("source_dataset_group", "")).strip()})
        status = "PASS" if len(values) == 4 and source_families == ["parameter_sensitivity"] else "FAIL"
        print(f"{model} {scale}: values={values} count={len(values)} source_family={source_families or ['MISSING']} status={status}")
        if status != "PASS":
            failure_rows = []
            for row in sorted(rows, key=lambda item: (sort_key("edge_density", str(item.get("x_label", ""))), str(item.get("method", "")))):
                failure_rows.append(
                    f"dat={row.get('source_dat_file','')} family={row.get('source_dataset_group','')} value={row.get('x_label','')} method={row.get('method','')}"
                )
            failures.append(f"{model} {scale}: " + "; ".join(failure_rows or ["no rows"]))
    if failures:
        raise RuntimeError(
            "Edge-density graph validation failed. Expected exactly 4 parameter_sensitivity values per scale.\n"
            + "\n".join(failures)
        )


def audit_final_graph_source_mapping(
    baseline_rows: dict[tuple[str, str], list[dict[str, str]]],
    extension_rows: dict[tuple[str, str], list[dict[str, str]]],
) -> list[dict[str, str]]:
    audit_rows: list[dict[str, str]] = []

    baseline_expected_family = {
        "network_size": "baseline",
        "edge_density": "parameter_sensitivity",
        "parking_supply": "parameter_sensitivity",
        "maximum_walking_distance": "parameter_sensitivity",
        "traffic_level": "parameter_sensitivity",
        "parking_budget": "parameter_sensitivity",
        "parking_cost_range": "parameter_sensitivity",
    }
    for (scale, parameter), rows in sorted(baseline_rows.items()):
        expected_family = baseline_expected_family.get(parameter, "parameter_sensitivity")
        for row in rows:
            actual_dataset = str(row.get("source_dat_file", "")).strip()
            actual_family = str(row.get("source_dataset_group", "")).strip()
            if not actual_dataset or not actual_family:
                audit_rows.append(
                    {
                        "model": "baseline",
                        "scale": scale,
                        "parameter": parameter,
                        "parameter_value": str(row.get("x_label", "")),
                        "method": str(row.get("method", "")),
                        "expected_dataset_family": expected_family,
                        "actual_dataset": "",
                        "actual_dataset_family": "",
                        "status": "MISSING",
                    }
                )
                continue
            status = "PASS" if actual_family == expected_family else "WRONG_DATASET_FAMILY"
            audit_rows.append(
                {
                    "model": "baseline",
                    "scale": scale,
                    "parameter": parameter,
                    "parameter_value": str(row.get("x_label", "")),
                    "method": str(row.get("method", "")),
                    "expected_dataset_family": expected_family,
                    "actual_dataset": actual_dataset,
                    "actual_dataset_family": actual_family,
                    "status": status,
                }
            )

    extension_allowed_families = {
        "node_size": {"final_case"},
        "edge_density": {"parameter_sensitivity"},
        "maximum_walking_distance": {"parameter_sensitivity"},
        "traffic_level": {"parameter_sensitivity"},
        "parking_budget_cmax": {"parameter_sensitivity"},
        "parking_cost_range": {"parameter_sensitivity"},
        "parking_spot_availability": {"parameter_sensitivity"},
        "spots_per_lot": {"parameter_sensitivity"},
    }
    for (scale, parameter), rows in sorted(extension_rows.items()):
        allowed_families = extension_allowed_families.get(parameter, {"final_case"})
        expected_family = "|".join(sorted(allowed_families))
        for row in rows:
            actual_dataset = str(row.get("source_dat_file", "")).strip()
            actual_family = str(row.get("source_dataset_group", "")).strip()
            if not actual_dataset or not actual_family:
                audit_rows.append(
                    {
                        "model": "extension",
                        "scale": scale,
                        "parameter": parameter,
                        "parameter_value": str(row.get("x_label", "")),
                        "method": str(row.get("method", "")),
                        "expected_dataset_family": expected_family,
                        "actual_dataset": "",
                        "actual_dataset_family": "",
                        "status": "MISSING",
                    }
                )
                continue
            status = "PASS" if actual_family in allowed_families else "WRONG_DATASET_FAMILY"
            audit_rows.append(
                {
                    "model": "extension",
                    "scale": scale,
                    "parameter": parameter,
                    "parameter_value": str(row.get("x_label", "")),
                    "method": str(row.get("method", "")),
                    "expected_dataset_family": expected_family,
                    "actual_dataset": actual_dataset,
                    "actual_dataset_family": actual_family,
                    "status": status,
                }
            )

    write_csv(
        FINAL_GRAPH_SOURCE_AUDIT_CSV,
        audit_rows,
        [
            "model",
            "scale",
            "parameter",
            "parameter_value",
            "method",
            "expected_dataset_family",
            "actual_dataset",
            "actual_dataset_family",
            "status",
        ],
    )
    return audit_rows


def extension_graph_sources() -> list[tuple[str, str, Path]]:
    sources: list[tuple[str, str, Path]] = []
    for path in sorted(EXTENSION_SOURCE_DIR.glob("*/*_objective_components.csv")):
        if "_runtime_" in path.name:
            continue
        scale = path.parent.name
        stem = path.stem.removesuffix("_objective_components")
        experiment = stem
        for prefix in ("small_scale_", "medium_large_", "sumo_"):
            if experiment.startswith(prefix):
                experiment = experiment.removeprefix(prefix)
        experiment = experiment.removesuffix("_three_metrics")
        if experiment == "baseline":
            continue
        sources.append((scale, experiment, path))
    return sources


def normalized_graph_rows_from_csv(path: Path, experiment: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(path):
        method = canonical_method(row.get("method"))
        if method is None:
            continue
        x_value = str(row.get("x_value") or row.get("parameter_value") or row.get("x_label") or "").strip()
        x_label = str(row.get("x_label") or x_value).strip()
        output = {field: str(row.get(field, "")).strip() for field in CSV_FIELDS}
        output["method"] = method
        output["x_value"] = x_value
        output["x_label"] = display_x_label(experiment, x_value, x_label)
        rows.append(output)
    return rows


def extension_dense_edge_rows(scale: str) -> list[dict[str, str]]:
    node_scale_by_output_scale = {
        "small_scale": "N40",
        "medium_large": "N2500",
        "sumo": "SUMO",
    }
    target_node_scale = node_scale_by_output_scale.get(scale)
    if target_node_scale is None or not EXTENSION_ALL_SPOT_CSV.exists():
        return []

    rows: list[dict[str, str]] = []
    for row in read_csv(EXTENSION_ALL_SPOT_CSV):
        method = canonical_method(row.get("method"))
        if method is None:
            continue
        if row.get("dataset_group") != "final_case":
            continue
        if row.get("node_scale") != target_node_scale:
            continue
        if row.get("parameter_name") != "edge_density" or row.get("scenario_name") != "Dense":
            continue

        edge_value = str(row.get("edges") or row.get("parameter_value") or row.get("scenario_name") or "").strip()
        if not edge_value or edge_value == "Dense":
            continue
        drive_hr = (parse_float(row.get("external_drive_time_hr")) or 0.0) + (parse_float(row.get("internal_drive_time_hr")) or 0.0)
        walk_hr = parse_float(row.get("total_walk_time_hr")) or parse_float(row.get("walk_time_hr")) or 0.0
        drive_min = drive_hr * 60.0
        walk_min = walk_hr * 60.0
        parking_cost = parse_float(row.get("parking_cost")) or 0.0
        alpha = parse_float(row.get("alpha"))
        if alpha is None:
            alpha = parse_float(row.get("Alpha"))
        if alpha is None:
            continue
        drive_component = drive_min
        walk_component = walk_min
        cost_component = parking_cost
        total = drive_component + walk_component + cost_component
        objective = parse_float(row.get("objective"))
        if objective is None:
            objective = spot_objective(
                alpha,
                parse_float(row.get("external_drive_time_hr")) or 0.0,
                parse_float(row.get("internal_drive_time_hr")) or 0.0,
                parse_float(row.get("internal_walk_time_hr")) or 0.0,
                parse_float(row.get("external_walk_time_hr")) or 0.0,
                parking_cost,
            )
        rows.append(
            {
                "x_value": edge_value,
                "x_label": edge_value,
                "method": method,
                "alpha": f"{alpha:.6f}",
                "driving_time_min": f"{drive_min:.6f}",
                "walking_time_min": f"{walk_min:.6f}",
                "parking_cost_usd": f"{parking_cost:.6f}",
                "displayed_driving_component": f"{drive_component:.6f}",
                "displayed_walking_component": f"{walk_component:.6f}",
                "displayed_cost_component": f"{cost_component:.6f}",
                "displayed_total": f"{total:.6f}",
                "objective_value": f"{objective:.6f}",
            }
        )
    return rows


def write_graph_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in CSV_FIELDS} for row in rows])


def layout_for(method_count: int) -> tuple[float, float, float]:
    if method_count >= 5:
        return 0.28, 0.09, 0.76
    if method_count == 4:
        return 0.34, 0.10, 0.72
    if method_count == 3:
        return 0.42, 0.12, 0.66
    return 0.52, 0.14, 0.60


def add_label(ax: plt.Axes, rect: Any, value: float, color: str, ymax: float) -> float:
    height = float(rect.get_height())
    if not math.isfinite(height) or height <= 0 or abs(value) <= 1e-12:
        return float(rect.get_y() + rect.get_height())
    x = float(rect.get_x() + rect.get_width() / 2.0)
    y_mid = float(rect.get_y() + rect.get_height() / 2.0)
    y_top = float(rect.get_y() + rect.get_height())
    relative_height = height / max(ymax, 1e-12)
    if relative_height >= 0.040:
        fontsize = 9.0
    elif relative_height >= 0.015:
        fontsize = 8.0
    else:
        fontsize = 7.0
    label = ax.text(
        x,
        y_mid,
        f"{value:.2f}",
        ha="center",
        va="center",
        rotation=0,
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        zorder=8,
        clip_on=True,
    )
    label.set_clip_path(rect)
    return y_top


def clean_tick_interval(ymax: float) -> float:
    if ymax <= 0:
        return 1.0
    target_interval = ymax / 8.0
    magnitude = 10 ** math.floor(math.log10(target_interval))
    for multiplier in (1, 2, 5, 10):
        interval = multiplier * magnitude
        if interval >= target_interval:
            return interval
    return 10 * magnitude


def render_graph(project: str, scale: str, experiment: str, rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    method_order = (
        BASELINE_METHOD_ORDER_BY_SCALE.get(scale, METHOD_ORDER)
        if project == "baseline"
        else EXTENSION_METHOD_ORDER_BY_SCALE.get(scale, METHOD_ORDER)
    )
    methods = [method for method in method_order if any(canonical_method(row.get("method")) == method for row in rows)]
    if not rows or not methods:
        return []

    x_values = sorted({str(row["x_label"]) for row in rows}, key=lambda value: sort_key(experiment, value))
    by_pair = {(str(row["x_label"]), canonical_method(row.get("method"))): row for row in rows}
    bar_width, inner_gap, group_gap = layout_for(len(methods))
    group_span = len(methods) * bar_width + max(0, len(methods) - 1) * inner_gap
    group_step = group_span + group_gap
    group_positions = np.arange(len(x_values), dtype=float) * group_step

    width = max(13.0, 3.0 + len(x_values) * 2.45 + len(methods) * 0.55)
    fig, ax = plt.subplots(figsize=(width, 12.0), dpi=300, constrained_layout=False)
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.9, alpha=0.95)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    max_total = 0.0
    for row in rows:
        total = parse_float(row.get("displayed_total")) or parse_float(row.get("objective_value")) or 0.0
        max_total = max(max_total, total)
    ymax = max(max_total * 1.06, 1.0)
    max_label_top = 0.0
    objective_labels: list[tuple[float, float, float, str]] = []
    ax.set_ylim(0, ymax)

    for x_label, group_center in zip(x_values, group_positions):
        present_methods = [method for method in methods if (x_label, method) in by_pair]
        if not present_methods:
            continue
        present_group_span = (
            len(present_methods) * bar_width
            + max(0, len(present_methods) - 1) * inner_gap
        )
        x_positions = [
            group_center - present_group_span / 2.0 + bar_width / 2.0 + idx * (bar_width + inner_gap)
            for idx in range(len(present_methods))
        ]
        for method, x in zip(present_methods, x_positions):
            row = by_pair[(x_label, method)]
            heights = [
                parse_float(row.get("displayed_driving_component")) or 0.0,
                parse_float(row.get("displayed_walking_component")) or 0.0,
                parse_float(row.get("displayed_cost_component")) or 0.0,
            ]
            raw_values = [
                parse_float(row.get("driving_time_min")) or 0.0,
                parse_float(row.get("walking_time_min")) or 0.0,
                parse_float(row.get("parking_cost_usd")) or 0.0,
            ]
            bottom = 0.0
            total_bar_top = 0.0
            first_rect = None
            for component, height, raw_value in zip(("Driving", "Walking", "Cost"), heights, raw_values):
                style = COMPONENT_STYLES[component]
                bars = ax.bar(
                    x,
                    height,
                    width=bar_width,
                    bottom=bottom,
                    color=METHOD_COLORS[method],
                    edgecolor="#475569",
                    linewidth=1.25,
                    hatch=style["hatch"],
                    alpha=style["alpha"],
                    zorder=3,
                )
                rect = bars.patches[0]
                first_rect = first_rect or rect
                text_color = "#0f172a" if component != "Cost" else "#111827"
                max_label_top = max(max_label_top, add_label(ax, rect, raw_value, text_color, ymax))
                bottom += height
                total_bar_top = bottom
            if first_rect is not None:
                bar_center = float(first_rect.get_x() + first_rect.get_width() / 2.0)
                objective = parse_float(row.get("objective_value")) or parse_float(row.get("displayed_total")) or total_bar_top
                objective_labels.append((bar_center, total_bar_top, objective, METHOD_COLORS[method]))

    objective_offset = ymax * 0.006
    objective_fontsize = 8.0 if len(methods) >= 4 else 8.8
    placed_objectives: list[tuple[float, float]] = []
    objective_x_collision = (bar_width + inner_gap) * 1.45
    objective_y_collision = ymax * 0.020
    objective_stagger = ymax * 0.022
    for x, total, objective, color in sorted(objective_labels, key=lambda item: (item[0], item[1])):
        expected_x = x
        y = total + objective_offset
        while any(abs(x - px) < objective_x_collision and abs(y - py) < objective_y_collision for px, py in placed_objectives):
            y += objective_stagger
        ax.text(
            expected_x,
            y,
            f"{objective:.3f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=objective_fontsize,
            color=color,
            fontweight="bold",
            zorder=8,
        )
        placed_objectives.append((expected_x, y))
        max_label_top = max(max_label_top, y)

    ax.set_xticks(group_positions)
    ax.set_xticklabels(x_values, fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_ylabel("Objective Value", fontsize=17, fontweight="bold")
    ax.set_xlabel(EXPERIMENT_LABELS.get(experiment, experiment.replace("_", " ").title()), fontsize=17, fontweight="bold")
    if max_label_top > ymax * 0.985:
        ymax = max(max_total * 1.08, max_label_top + max_total * 0.006, 1.0)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(clean_tick_interval(ymax)))
    ax.margins(x=0.08)

    handles: list[Patch] = []
    for method in methods:
        for component, label in (("Driving", "Driving Time"), ("Walking", "Walking Time"), ("Cost", "Parking Cost")):
            style = COMPONENT_STYLES[component]
            handles.append(
                Patch(
                    facecolor=METHOD_COLORS[method],
                    edgecolor="#475569",
                    hatch=style["hatch"],
                    alpha=style["alpha"],
                    label=f"{METHOD_LABELS[method]} - {label}",
                )
            )
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(2, min(4, len(methods))),
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        fontsize=10.2,
        borderpad=0.85,
        labelspacing=0.65,
        handletextpad=0.7,
        columnspacing=1.2,
    )
    legend.get_frame().set_edgecolor("#94a3b8")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.82)
    if (
        os.environ.get("FINAL_GRAPH_DIAGNOSTICS") == "1"
        and project == "baseline"
        and scale == "medium_large"
        and experiment == "traffic_level"
    ):
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        ymin, ymax_actual = ax.get_ylim()
        pixels_per_y_unit = bbox.height / max(ymax_actual - ymin, 1e-12)
        print("Representative graph: Traffic Level")
        print(f"Figure size: {fig.get_figwidth():.2f} x {fig.get_figheight():.2f}")
        print(f"Actual axes height: {bbox.height:.2f} pixels")
        print(f"Y-axis range: {ymin:.3f} to {ymax_actual:.3f}")
        print(f"Pixels per Y unit: {pixels_per_y_unit:.2f}")
        print(f"Pixels from 0 to 10: {pixels_per_y_unit * 10:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scale}_{experiment}_three_metrics"
    csv_path = output_dir / f"{stem}_objective_components.csv"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    sorted_rows = [by_pair[(x_label, method)] for x_label in x_values for method in methods if (x_label, method) in by_pair]
    write_graph_csv(csv_path, sorted_rows)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [csv_path, png_path, pdf_path]


def clean_final_outputs() -> None:
    for root in (FINAL_BASELINE_DIR, FINAL_EXTENSION_DIR):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".csv"}:
                path.unlink()


def main() -> None:
    validate_baseline_small_wmax_exact_coverage()
    baseline_rows = baseline_graph_rows()
    extension_rows = extension_graph_rows()
    validate_edge_density_graphs(baseline_rows, extension_rows)
    audit_rows = audit_final_graph_source_mapping(baseline_rows, extension_rows)
    wrong_family = sum(1 for row in audit_rows if row["status"] == "WRONG_DATASET_FAMILY")
    missing = sum(1 for row in audit_rows if row["status"] == "MISSING")
    if wrong_family or missing:
        raise RuntimeError(
            f"Final graph source mapping audit failed: wrong_family={wrong_family}, missing={missing}. "
            f"See {FINAL_GRAPH_SOURCE_AUDIT_CSV}"
        )

    clean_final_outputs()
    generated: list[Path] = []
    for (scale, experiment), rows in sorted(baseline_rows.items()):
        generated.extend(render_graph("baseline", scale, experiment, rows, FINAL_BASELINE_DIR / scale))

    for (scale, experiment), rows in sorted(extension_rows.items()):
        generated.extend(render_graph("extension", scale, experiment, rows, FINAL_EXTENSION_DIR / scale))

    print(f"Generated {sum(1 for path in generated if path.suffix == '.pdf')} final PDF figures.")
    print(f"Baseline output: {FINAL_BASELINE_DIR}")
    print(f"Extension output: {FINAL_EXTENSION_DIR}")
    print(f"Final graph source audit: {FINAL_GRAPH_SOURCE_AUDIT_CSV}")


if __name__ == "__main__":
    main()
