from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lot_objective_utils import build_lot_normalization_context, normalized_lot_objective, parse_lot_dat_for_normalization


ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = ROOT / "SUMO"
SUMO_CASES_DIR = SUMO_DIR / "solver_cases_10"
CANONICAL_SUMO_CASES_DIR = ROOT / "final_cases" / "SUMO"
PARAMETER_SUMO_CASES_DIR = ROOT / "final_cases" / "parameter_sensitivity" / "SUMO"
RESULTS_DIR = ROOT / "COMPARISIONS" / "sumo_comparison_results"
LOGS_DIR = RESULTS_DIR / "logs"
SEPARATE_PLOTS_DIR = RESULTS_DIR / "separate_metric_plots"
PANELS_DIR = RESULTS_DIR / "three_metric_panels"
COMPARISON_DATA_CSV = RESULTS_DIR / "sumo_method_comparison_data.csv"
SUMMARY_CSV = RESULTS_DIR / "sumo_method_summary.csv"

WALKING_SPEED_KMPH = 5.0

METHOD_LABELS = {
    "HEURISTIC": "Heuristic",
    "ACO": "ACO",
    "PSO_CHAOTIC": "PSO",
    "GA_ROUTE_EVOLUTION": "GA",
}

METHOD_ORDER = ["Heuristic", "ACO", "PSO", "GA"]

METHOD_COLORS = {
    "Heuristic": "#0f766e",
    "ACO": "#1d4ed8",
    "PSO": "#b45309",
    "GA": "#be123c",
}

def _prefer_parameter_sensitivity(path: Path, fallback: Path) -> Path:
    return path if path.exists() else fallback


SOURCE_RESULT_LOGS = {
    "Heuristic": _prefer_parameter_sensitivity(
        ROOT / "heuristics" / "data" / "parameter_sensitivity_sumo" / "results_heuristic.csv",
        ROOT / "heuristics" / "data" / "sumo" / "results_heuristic.csv",
    ),
    "ACO": _prefer_parameter_sensitivity(
        ROOT / "aco" / "data" / "parameter_sensitivity_sumo" / "results_aco.csv",
        ROOT / "aco" / "data" / "sumo" / "results_aco.csv",
    ),
    "PSO": _prefer_parameter_sensitivity(
        ROOT / "pso" / "data" / "parameter_sensitivity_sumo" / "results_pso.csv",
        ROOT / "pso" / "data" / "sumo" / "results_pso.csv",
    ),
    "GA": _prefer_parameter_sensitivity(
        ROOT / "ga" / "data" / "parameter_sensitivity_sumo" / "results_ga.csv",
        ROOT / "ga" / "data" / "sumo" / "results_ga.csv",
    ),
}

SOURCE_ROUTE_LOGS = {
    "Heuristic": _prefer_parameter_sensitivity(
        ROOT / "heuristics" / "data" / "parameter_sensitivity_sumo" / "routes_heuristic.csv",
        ROOT / "heuristics" / "data" / "sumo" / "routes_heuristic.csv",
    ),
    "ACO": _prefer_parameter_sensitivity(
        ROOT / "aco" / "data" / "parameter_sensitivity_sumo" / "routes_aco.csv",
        ROOT / "aco" / "data" / "sumo" / "routes_aco.csv",
    ),
    "PSO": _prefer_parameter_sensitivity(
        ROOT / "pso" / "data" / "parameter_sensitivity_sumo" / "routes_pso.csv",
        ROOT / "pso" / "data" / "sumo" / "routes_pso.csv",
    ),
    "GA": _prefer_parameter_sensitivity(
        ROOT / "ga" / "data" / "parameter_sensitivity_sumo" / "routes_ga.csv",
        ROOT / "ga" / "data" / "sumo" / "routes_ga.csv",
    ),
}

TARGET_LOGS = {
    "Heuristic": LOGS_DIR / "log_heuristic_sumo.csv",
    "ACO": LOGS_DIR / "log_aco_sumo.csv",
    "PSO": LOGS_DIR / "log_pso_sumo.csv",
    "GA": LOGS_DIR / "log_ga_sumo.csv",
}

CASE_VARIANTS = {
    "C1": {
        "scenario_name": "Backbone baseline",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Baseline",
        "traffic_level": "Low",
        "traffic_speed_range": "50-70 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C2": {
        "scenario_name": "Dense baseline",
        "edge_mode": "Dense",
        "parking_supply_mode": "Baseline",
        "traffic_level": "Low",
        "traffic_speed_range": "50-70 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C3": {
        "scenario_name": "Higher parking supply",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Higher",
        "traffic_level": "Low",
        "traffic_speed_range": "50-70 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C4": {
        "scenario_name": "Higher walking limit",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Baseline",
        "traffic_level": "Low",
        "traffic_speed_range": "50-70 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C5": {
        "scenario_name": "Medium traffic",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Baseline",
        "traffic_level": "Medium",
        "traffic_speed_range": "30-49 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C6": {
        "scenario_name": "High traffic",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Baseline",
        "traffic_level": "High",
        "traffic_speed_range": "10-29 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C7": {
        "scenario_name": "High alpha",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Baseline",
        "traffic_level": "Low",
        "traffic_speed_range": "50-70 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C8": {
        "scenario_name": "Low alpha",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Baseline",
        "traffic_level": "Low",
        "traffic_speed_range": "50-70 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C9": {
        "scenario_name": "Lower parking budget",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Baseline",
        "traffic_level": "Low",
        "traffic_speed_range": "50-70 km/h",
        "parking_cost_mode": "Baseline",
        "parking_cost_variation_mode": "8.0-20.5",
    },
    "C10": {
        "scenario_name": "High parking-cost variation",
        "edge_mode": "Backbone",
        "parking_supply_mode": "Baseline",
        "traffic_level": "Low",
        "traffic_speed_range": "50-70 km/h",
        "parking_cost_mode": "High",
        "parking_cost_variation_mode": "15.0-30.0",
    },
}

PARAMETER_GROUPS = {
    "node_size": {
        "cases": {"C1"},
        "title": "Effect of SUMO Network Size on Algorithm Performance",
        "x_label": "Network Size",
        "value_key": "nodes",
    },
    "edge_density": {
        "cases": {"C1", "C2"},
        "title": "Effect of Edge Density on Algorithm Performance",
        "x_label": "Number of Directed Edges",
        "value_key": "edges",
    },
    "parking_supply": {
        "cases": {"C1", "C3"},
        "title": "Effect of Parking Supply on Algorithm Performance",
        "x_label": "Parking Lots",
        "value_key": "parking_lots",
    },
    "maximum_walking_distance": {
        "cases": {"C1", "C4"},
        "title": "Effect of Maximum Walking Distance on Algorithm Performance",
        "x_label": "Maximum Walking Distance (km)",
        "value_key": "maximum_walking_distance",
    },
    "traffic_level": {
        "cases": {"C1", "C5", "C6"},
        "title": "Effect of Traffic Level on Algorithm Performance",
        "x_label": "Traffic Level",
        "value_key": "traffic_level",
    },
    "objective_weight_alpha": {
        "cases": {"C1", "C7", "C8"},
        "title": "Effect of Objective Weight Alpha on Algorithm Performance",
        "x_label": "Objective Weight Alpha",
        "value_key": "objective_weight_alpha",
    },
    "parking_budget_cmax": {
        "cases": {"C1", "C9"},
        "title": "Effect of Parking Budget Cmax on Algorithm Performance",
        "x_label": "Parking Budget Cmax",
        "value_key": "parking_budget_cmax",
    },
    "parking_cost_variation_mode": {
        "cases": {"C1", "C10"},
        "title": "Effect of Parking-Cost Range on Algorithm Performance",
        "x_label": "Parking Cost Range",
        "value_key": "parking_cost_variation_mode",
    },
}

REQUIRED_NORMALIZED_COLUMNS = [
    "case",
    "scenario_id",
    "scenario_group",
    "parameter_name",
    "parameter_value",
    "method",
    "nodes",
    "edges",
    "parking_lots",
    "source",
    "destination",
    "alpha",
    "wmax",
    "cmax",
    "traffic_level",
    "edge_mode",
    "parking_supply_mode",
    "parking_cost_mode",
    "feasible",
    "parking_node",
    "drive_time_hr",
    "walking_distance_km",
    "walking_time_hr",
    "travel_time_hr",
    "parking_cost",
    "objective",
    "runtime_ms",
    "drive_path",
    "walk_path",
]


def ensure_output_dirs() -> None:
    for directory in [RESULTS_DIR, LOGS_DIR, SEPARATE_PLOTS_DIR, PANELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_case_code(name: str) -> str:
    match = re.search(r"_C(\d+)_", name)
    if not match:
        if "_PS_" in name:
            return "PS"
        raise ValueError(f"Could not parse case code from {name}")
    return f"C{int(match.group(1))}"


def safe_float(value: str | float | int | None, default: float | None = None) -> float:
    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError("Missing numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value}")
    return result


def safe_int(value: str | float | int | None, default: int | None = None) -> int:
    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError("Missing integer value")
    return int(float(value))


def parse_feasible(value: str | int | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "feasible"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def method_label(raw_method: str) -> str:
    return METHOD_LABELS.get(raw_method, raw_method)


def scale_category(nodes: int) -> str:
    if nodes <= 100:
        return "Small"
    if nodes <= 1000:
        return "Medium"
    return "Large"


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_sumo_dat_paths() -> list[Path]:
    dat_paths = sorted(SUMO_CASES_DIR.glob("N*_C*_I*.dat"))
    if not dat_paths:
        dat_paths = sorted(CANONICAL_SUMO_CASES_DIR.glob("N*_C*_I*.dat"))
    parameter_paths = sorted(PARAMETER_SUMO_CASES_DIR.glob("N*_PS_*_I*.dat"))
    return dat_paths + parameter_paths


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_dat_metadata(dat_path: Path) -> dict:
    text = dat_path.read_text(encoding="utf-8", errors="replace")

    def get_int(name: str) -> int | None:
        match = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", text)
        return int(match.group(1)) if match else None

    def get_float(name: str) -> float | None:
        match = re.search(rf"\b{name}\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*;", text)
        return float(match.group(1)) if match else None

    def get_float_list(name: str) -> list[float]:
        match = re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]\s*;", text, flags=re.DOTALL)
        if not match:
            return []
        return [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", match.group(1))]

    def get_int_list(name: str) -> list[int]:
        match = re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]\s*;", text, flags=re.DOTALL)
        if not match:
            return []
        return [int(item) for item in re.findall(r"\d+", match.group(1))]

    def get_string(name: str) -> str | None:
        match = re.search(rf'\b{name}\s*=\s*"([^"]*)"\s*;', text)
        return match.group(1).strip() if match else None

    lots = get_int_list("L")
    costs = get_float_list("C")
    walk_km = get_float_list("WalkDistanceToDest")
    walk_hr = get_float_list("WalkTimeToDest")
    edge_count = len(re.findall(r"<\s*\d+\s*,\s*\d+\s*>", text))
    return {
        "experiment_type": get_string("ExperimentType"),
        "scale_category": get_string("ScaleCategory"),
        "varied_parameter": get_string("VariedParameter"),
        "active_parameter_value": get_string("ActiveParameterValue"),
        "traffic_level_meta": get_string("TrafficLevel"),
        "traffic_speed_range_meta": get_string("TrafficSpeedRange"),
        "parking_cost_range_meta": get_string("ParkingCostRange"),
        "parking_supply_pct_meta": get_float("ParkingSupplyPct"),
        "parameter_label": get_string("ParameterLabel"),
        "nodes": get_int("Nodes"),
        "source": get_int("S"),
        "destination": get_int("D"),
        "alpha": get_float("Alpha"),
        "wmax_km": get_float("W_km"),
        "wmax_time_hr": get_float("W"),
        "cmax": get_float("Cmax"),
        "parking_lots": get_int("Q") or len(lots),
        "lot_nodes": lots,
        "parking_costs": costs,
        "walk_distance_values_km": walk_km,
        "walk_time_values_hr": walk_hr,
        "walk_distance_by_node": {node: value for node, value in zip(lots, walk_km)},
        "walk_time_by_node": {node: value for node, value in zip(lots, walk_hr)},
        "dat_edges": edge_count,
    }


def infer_speed_range_from_traffic(traffic_level: str | None) -> str | None:
    mapping = {
        "Very Low": "60-80 km/h",
        "Low": "50-70 km/h",
        "Medium": "30-49 km/h",
        "High": "10-29 km/h",
    }
    return mapping.get(traffic_level or "")


def _title_case_traffic_level(label: str | None) -> str | None:
    if not label:
        return None
    normalized = str(label).replace("_", " ").replace("-", " ").strip().lower()
    mapping = {
        "very low": "Very Low",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
    }
    return mapping.get(normalized, str(label))


def _scenario_name_from_parameter(dat_path: Path, dat_meta: dict) -> str:
    parameter = dat_meta.get("varied_parameter") or "parameter_sensitivity"
    active_value = dat_meta.get("active_parameter_value") or dat_path.stem
    display = {
        "edge_density": f"Edge density {active_value}",
        "parking_supply": f"Parking supply {active_value}",
        "maximum_walking_distance": f"Walking distance {active_value} km",
        "parking_budget_cmax": f"Parking budget {active_value}",
        "parking_cost_variation_mode": f"Parking cost range {active_value}",
        "traffic_level": f"Traffic level {_title_case_traffic_level(active_value) or active_value}",
    }
    return display.get(parameter, f"{parameter}: {active_value}")


def infer_case_metadata(case_code: str, dat_path: Path | None = None) -> dict:
    if dat_path is None:
        dat_path = SUMO_CASES_DIR / f"N2055_{case_code}_I1.dat"
        if not dat_path.exists():
            dat_path = CANONICAL_SUMO_CASES_DIR / f"N2055_{case_code}_I1.dat"
    dat_meta = parse_dat_metadata(dat_path)
    if case_code == "PS" or dat_meta.get("experiment_type") == "parameter_sensitivity":
        traffic_level = _title_case_traffic_level(dat_meta.get("traffic_level_meta")) or "Low"
        speed_range = dat_meta.get("traffic_speed_range_meta") or infer_speed_range_from_traffic(traffic_level)
        parking_ratio = 0.0
        if dat_meta["parking_lots"] and dat_meta["nodes"]:
            parking_ratio = dat_meta["parking_lots"] / dat_meta["nodes"]
        parking_supply_pct = dat_meta.get("parking_supply_pct_meta")
        if parking_supply_pct is None:
            parking_supply_pct = parking_ratio * 100.0
        parking_supply_mode = f"{int(round(parking_supply_pct))}%"
        parking_cost_variation_mode = dat_meta.get("parking_cost_range_meta") or dat_meta.get("active_parameter_value") or "8.0-20.5"
        edge_mode = dat_meta.get("active_parameter_value") if dat_meta.get("varied_parameter") == "edge_density" else "Baseline"
        return {
            "case_id": case_code,
            "scenario_name": _scenario_name_from_parameter(dat_path, dat_meta),
            "dat_path": dat_path,
            "nodes": dat_meta.get("nodes"),
            "edges": dat_meta.get("dat_edges"),
            "parking_lots": dat_meta.get("parking_lots"),
            "source": dat_meta.get("source"),
            "destination": dat_meta.get("destination"),
            "alpha": dat_meta.get("alpha"),
            "wmax": dat_meta.get("wmax_km"),
            "wmax_time_hr": dat_meta.get("wmax_time_hr"),
            "cmax": dat_meta.get("cmax"),
            "traffic_level": traffic_level,
            "traffic_speed_range": speed_range,
            "edge_mode": edge_mode,
            "parking_cost_mode": "Baseline" if parking_cost_variation_mode == "8.0-20.5" else "Varied",
            "parking_supply_mode": parking_supply_mode,
            "maximum_walking_distance": dat_meta.get("wmax_km"),
            "objective_weight_alpha": dat_meta.get("alpha"),
            "parking_budget_cmax": dat_meta.get("cmax"),
            "parking_cost_variation_mode": parking_cost_variation_mode,
            "parking_supply_pct": parking_supply_pct,
            "graph_node_count": dat_meta.get("nodes"),
            "graph_edge_count": dat_meta.get("dat_edges"),
            "sumo_source_node": dat_meta.get("source"),
            "sumo_destination_node": dat_meta.get("destination"),
            "selected_parking_count": dat_meta.get("parking_lots"),
            "requested_parking_count": dat_meta.get("parking_lots"),
            "feasible_parking_count": len(dat_meta.get("walk_distance_by_node", {})),
            "infeasible_parking_count": 0,
            "source_root": "final_cases/parameter_sensitivity/SUMO",
            "walk_distance_by_node": dat_meta.get("walk_distance_by_node", {}),
            "walk_time_by_node": dat_meta.get("walk_time_by_node", {}),
            "experiment_type": "parameter_sensitivity",
            "varied_parameter": dat_meta.get("varied_parameter"),
            "active_parameter_value": dat_meta.get("active_parameter_value"),
        }

    parking_meta = read_json(SUMO_DIR / "_variation_work" / case_code / "parking" / "parking_metadata.json")
    graph_meta = read_json(SUMO_DIR / "_variation_work" / case_code / "graph" / "graph_metadata.json")
    case_defaults = CASE_VARIANTS.get(case_code, {}).copy()
    traffic_level = case_defaults.get("traffic_level")
    speed_range = infer_speed_range_from_traffic(traffic_level)
    parking_cost_mode = case_defaults.get("parking_cost_mode", "Baseline")
    if parking_meta.get("cost_mode"):
        parking_cost_mode = str(parking_meta["cost_mode"]).strip().capitalize()
    parking_supply_mode = case_defaults.get("parking_supply_mode", "Baseline")
    if dat_meta["parking_lots"] and dat_meta["nodes"]:
        baseline_ratio = 103 / 2055
        parking_ratio = dat_meta["parking_lots"] / dat_meta["nodes"]
        parking_supply_mode = "Higher" if parking_ratio > baseline_ratio + 1e-9 else "Baseline"
    edge_mode = case_defaults.get("edge_mode", "Backbone")
    parking_cost_variation_mode = case_defaults.get("parking_cost_variation_mode", "8.0-20.5")
    cost_range = parking_meta.get("parking_cost_range")
    if isinstance(cost_range, list) and len(cost_range) == 2:
        parking_cost_variation_mode = f"{float(cost_range[0]):.1f}-{float(cost_range[1]):.1f}"
    elif parking_cost_mode.lower() == "high":
        parking_cost_variation_mode = "15.0-30.0"

    return {
        "case_id": case_code,
        "scenario_name": case_defaults.get("scenario_name", dat_path.stem),
        "dat_path": dat_path,
        "nodes": dat_meta.get("nodes"),
        "edges": dat_meta.get("dat_edges"),
        "parking_lots": dat_meta.get("parking_lots"),
        "source": dat_meta.get("source"),
        "destination": dat_meta.get("destination"),
        "alpha": dat_meta.get("alpha"),
        "wmax": dat_meta.get("wmax_km"),
        "wmax_time_hr": dat_meta.get("wmax_time_hr"),
        "cmax": dat_meta.get("cmax"),
        "traffic_level": traffic_level,
        "traffic_speed_range": speed_range,
        "edge_mode": edge_mode,
        "parking_cost_mode": parking_cost_mode,
        "parking_supply_mode": parking_supply_mode,
        "maximum_walking_distance": dat_meta.get("wmax_km"),
        "objective_weight_alpha": dat_meta.get("alpha"),
        "parking_budget_cmax": dat_meta.get("cmax"),
        "parking_cost_variation_mode": parking_cost_variation_mode,
        "parking_supply_pct": (parking_meta.get("parking_pct", 0.0) or 0.0) * 100.0,
        "graph_node_count": graph_meta.get("node_count"),
        "graph_edge_count": graph_meta.get("edge_count"),
        "sumo_source_node": parking_meta.get("source_node"),
        "sumo_destination_node": parking_meta.get("destination_node"),
        "selected_parking_count": parking_meta.get("selected_parking_count"),
        "requested_parking_count": parking_meta.get("requested_parking_count"),
        "feasible_parking_count": parking_meta.get("feasible_parking_count"),
        "infeasible_parking_count": parking_meta.get("infeasible_parking_count"),
        "source_root": "SUMO/solver_cases_10",
        "walk_distance_by_node": dat_meta.get("walk_distance_by_node", {}),
        "walk_time_by_node": dat_meta.get("walk_time_by_node", {}),
        "experiment_type": "legacy_controlled_cases",
        "varied_parameter": None,
        "active_parameter_value": None,
    }


def scenario_metadata_rows() -> list[dict]:
    rows: list[dict] = []
    for dat_path in list_sumo_dat_paths():
        case_code = parse_case_code(dat_path.name)
        meta = infer_case_metadata(case_code, dat_path)
        rows.append(meta)
    rows.sort(key=lambda row: (row.get("experiment_type", ""), row["scenario_name"]))
    return rows


def inventory_sumo_assets() -> dict[str, list[str]]:
    return {
        "SUMO .net.xml files": sorted(str(path.relative_to(ROOT)) for path in SUMO_DIR.rglob("*.net.xml")),
        "Converted .dat files": sorted(str(path.relative_to(ROOT)) for path in SUMO_DIR.rglob("*.dat")),
        "Scenario folders": sorted(
            str(path.relative_to(ROOT))
            for path in sorted(SUMO_DIR.glob("*"))
            if path.is_dir()
        ),
        "Result/log CSVs": sorted(
            str(path.relative_to(ROOT))
            for path in (
                list(SUMO_DIR.rglob("*.csv"))
                + [path for path in SOURCE_RESULT_LOGS.values() if path.exists()]
                + [path for path in SOURCE_ROUTE_LOGS.values() if path.exists()]
            )
        ),
        "Parking candidate files": sorted(
            str(path.relative_to(ROOT))
            for path in list(SUMO_DIR.rglob("parking_nodes.csv")) + list(SUMO_DIR.rglob("parking_metadata.json"))
        ),
        "Preprocessing scripts": sorted(
            str(path.relative_to(ROOT))
            for path in [SUMO_DIR / "preprocess_sumo_network.py", SUMO_DIR / "build_parking_layer.py"]
            if path.exists()
        ),
        "Variant indicators (C1-C10)": sorted(
            str(path.relative_to(ROOT))
            for path in list(SUMO_CASES_DIR.glob("N2055_C*_I1.dat")) + list((SUMO_DIR / "_variation_work").glob("C*"))
        ),
    }


def print_inventory() -> None:
    inventory = inventory_sumo_assets()
    print("=" * 60)
    print("SUMO ASSET INVENTORY")
    print("=" * 60)
    for heading, values in inventory.items():
        print(f"{heading}:")
        if not values:
            print("  (none)")
        else:
            for value in values:
                print(f"  - {value}")
        print()


def print_scenario_interpretation() -> None:
    rows = scenario_metadata_rows()
    print("=" * 60)
    print("SUMO SCENARIO INTERPRETATION")
    print("=" * 60)
    for row in rows:
        print(
            f"{row['case_id']} | {row['scenario_name']} | "
            f"nodes={row['nodes']} edges={row['edges']} parking={row['parking_lots']} "
            f"S={row['source']} D={row['destination']} alpha={row['alpha']} "
            f"wmax={row['wmax']} cmax={row['cmax']} traffic={row['traffic_level']} "
            f"speed={row['traffic_speed_range']} edge_mode={row['edge_mode']} "
            f"parking_supply={row['parking_supply_mode']} cost_mode={row['parking_cost_mode']}"
        )
    print()


def normalize_result_row(row: dict) -> dict:
    def first_value(*keys: str, default=None):
        for key in keys:
            if key in row and row[key] not in {"", None}:
                return row[key]
        return default

    dat_file = first_value("dat_file", "Case")
    if dat_file is None:
        raise KeyError("dat_file")
    if not str(dat_file).endswith(".dat") and re.match(r"N\d+_C\d+_I\d+$", str(dat_file)):
        dat_file = f"{dat_file}.dat"
    case_code = parse_case_code(dat_file)
    dat_path = Path(dat_file)
    if not dat_path.exists():
        candidate_paths = [
            CANONICAL_SUMO_CASES_DIR / dat_path.name,
            SUMO_CASES_DIR / dat_path.name,
            PARAMETER_SUMO_CASES_DIR / dat_path.name,
        ]
        dat_path = next((candidate for candidate in candidate_paths if candidate.exists()), dat_path)
    meta = infer_case_metadata(case_code, dat_path)
    parking_node_raw = first_value("parking_node", "ParkingNode", default="")
    parking_node = str(parking_node_raw).strip()
    parking_node_int = None
    if parking_node:
        try:
            parking_node_int = safe_int(parking_node)
            parking_node = str(parking_node_int)
        except Exception:
            parking_node_int = None

    dat_walk_distance = meta["walk_distance_by_node"].get(parking_node_int) if parking_node_int is not None else None
    dat_walk_time = meta["walk_time_by_node"].get(parking_node_int) if parking_node_int is not None else None

    raw_walk_distance = first_value("walk_distance_km")
    raw_walk_time = first_value("walk_time_hr")
    walking_distance_km = safe_float(raw_walk_distance, dat_walk_distance if dat_walk_distance is not None else 0.0)
    walking_time_hr = safe_float(raw_walk_time, dat_walk_time if dat_walk_time is not None else walking_distance_km / WALKING_SPEED_KMPH)
    if raw_walk_time in {"", None} and dat_walk_distance is not None and dat_walk_time is not None:
        walking_distance_km = dat_walk_distance
        walking_time_hr = dat_walk_time
    drive_time_hr = safe_float(first_value("drive_time_hr"), 0.0)
    travel_time_hr = drive_time_hr + walking_time_hr
    parking_cost = safe_float(first_value("parking_cost"), 0.0)
    alpha = safe_float(first_value("Alpha"), meta.get("objective_weight_alpha", 0.5))
    dat_text = dat_path.read_text(encoding="utf-8", errors="replace")
    objective_context = build_lot_normalization_context(parse_lot_dat_for_normalization(dat_text))
    objective = safe_float(first_value("objective"), normalized_lot_objective(travel_time_hr, parking_cost, context=objective_context))
    cmax = safe_float(first_value("Cmax"), meta.get("parking_budget_cmax", 0.0))
    return {
        "case": case_code,
        "scenario_id": Path(dat_file).stem,
        "scenario_name": meta["scenario_name"],
        "dat_file": dat_file,
        "source_dataset": meta["source_root"],
        "method": method_label(first_value("method", default="HEURISTIC")),
        "raw_method": first_value("method", default="HEURISTIC"),
        "nodes": safe_int(first_value("nodes", "Nodes"), meta.get("nodes")),
        "edges": safe_int(first_value("edges", "Edges"), meta.get("edges", 0)),
        "parking_lots": safe_int(first_value("parking_lots", "ParkingLots"), meta.get("parking_lots", 0)),
        "source": safe_int(first_value("S", "Source"), meta.get("source", 0)),
        "destination": safe_int(first_value("D", "Destination"), meta.get("destination", 0)),
        "alpha": alpha,
        "wmax": safe_float(first_value("Wmax"), meta.get("wmax", 0.0)),
        "wmax_time_hr": safe_float(meta.get("wmax_time_hr"), 0.0),
        "cmax": cmax,
        "traffic_level": meta["traffic_level"],
        "traffic_speed_range": meta["traffic_speed_range"],
        "edge_mode": meta["edge_mode"],
        "parking_supply_mode": meta["parking_supply_mode"],
        "parking_supply_pct": safe_float(meta.get("parking_supply_pct"), 0.0),
        "parking_cost_mode": meta["parking_cost_mode"],
        "maximum_walking_distance": safe_float(meta["maximum_walking_distance"]),
        "objective_weight_alpha": safe_float(meta["objective_weight_alpha"]),
        "parking_budget_cmax": safe_float(meta["parking_budget_cmax"]),
        "parking_cost_variation_mode": meta["parking_cost_variation_mode"],
        "experiment_type": meta.get("experiment_type", "legacy_controlled_cases"),
        "varied_parameter": meta.get("varied_parameter"),
        "active_parameter_value": meta.get("active_parameter_value"),
        "scale": scale_category(safe_int(first_value("nodes", "Nodes"), meta.get("nodes"))),
        "feasible": 1 if parse_feasible(first_value("feasible", "FeasibleParking", "Feasible")) else 0,
        "parking_node": parking_node,
        "drive_time_hr": drive_time_hr,
        "walking_distance_km": walking_distance_km,
        "walking_time_hr": walking_time_hr,
        "travel_time_hr": travel_time_hr,
        "parking_cost": parking_cost,
        "objective": objective,
        "runtime_ms": safe_float(first_value("runtime_ms"), 0.0),
        "drive_path": first_value("drive_path", "DrivePath", default=""),
        "walk_path": first_value("walk_path", "WalkPath", default=""),
    }


def expand_comparison_rows(base_row: dict) -> list[dict]:
    def build_parameter_row(parameter_name: str, config: dict) -> dict:
        row = dict(base_row)
        row["parameter_name"] = parameter_name
        row["parameter_value"] = row[config["value_key"]]
        if parameter_name == "edge_density":
            row["scenario_group"] = row["edge_mode"]
        elif parameter_name == "parking_supply":
            row["scenario_group"] = row["parking_supply_mode"]
        else:
            row["scenario_group"] = parameter_name
        return row

    if base_row.get("experiment_type") == "parameter_sensitivity":
        varied_parameter = base_row.get("varied_parameter")
        config = PARAMETER_GROUPS.get(varied_parameter)
        if not varied_parameter or config is None:
            return []
        return [build_parameter_row(varied_parameter, config)]

    case_code = base_row["case"]
    expanded: list[dict] = []
    for parameter_name, config in PARAMETER_GROUPS.items():
        if case_code not in config["cases"]:
            continue
        expanded.append(build_parameter_row(parameter_name, config))
    return expanded


def normalize_logs_to_comparison_rows(log_paths: dict[str, Path] | None = None) -> list[dict]:
    log_paths = log_paths or TARGET_LOGS
    rows: list[dict] = []
    for method in METHOD_ORDER:
        path = log_paths[method]
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            normalized = normalize_result_row(row)
            rows.extend(expand_comparison_rows(normalized))
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["parameter_name"], str(row["parameter_value"]), row["method"])
        grouped[key].append(row)

    summary_rows: list[dict] = []
    for (parameter_name, parameter_value, method), members in sorted(grouped.items()):
        summary_rows.append(
            {
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "method": method,
                "sample_size": len(members),
                "mean_travel_time_hr": mean([member["travel_time_hr"] for member in members]),
                "mean_drive_time_hr": mean([member["drive_time_hr"] for member in members]),
                "mean_walking_time_hr": mean([member["walking_time_hr"] for member in members]),
                "mean_walking_distance_km": mean([member["walking_distance_km"] for member in members]),
                "mean_parking_cost": mean([member["parking_cost"] for member in members]),
                "mean_objective": mean([member["objective"] for member in members]),
                "mean_runtime_ms": mean([member["runtime_ms"] for member in members]),
                "selected_parking_nodes": "|".join(sorted({str(member["parking_node"]) for member in members})),
            }
        )
    return summary_rows
