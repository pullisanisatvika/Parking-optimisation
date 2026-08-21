from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "RESULTS"
OUTPUT_DIRS = {
    "small": RESULTS_DIR / "small_scale",
    "medium_large": RESULTS_DIR / "medium_scale",
    "sumo": RESULTS_DIR / "sumo",
}
MASTER_CSV = RESULTS_DIR / "master_results_all_methods_for_graphs.csv"

SOURCE_FILES = {
    "Optimization": ROOT / "OPTIMIZATION" / "optimisation_results_summary_exact_small.csv",
    "Heuristic": ROOT / "HEURISTICS" / "data" / "master_results_heuristic.csv",
    "ACO": ROOT / "ACO" / "data" / "master_results_aco.csv",
    "GA": ROOT / "GA" / "data" / "master_results_ga.csv",
    "PSO": ROOT / "PSO" / "data" / "master_results_pso.csv",
}

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

OBJECTIVE_FIGSIZE = (36.0, 20.5)
OBJECTIVE_DPI = 300
OBJECTIVE_SAVEFIG_KWARGS = {"dpi": OBJECTIVE_DPI, "bbox_inches": "tight", "pad_inches": 0.15}
SHOW_OBJECTIVE_COMPONENT_LABELS = True
OBJECTIVE_AXIS_LABEL_SIZE = 40
OBJECTIVE_TICK_LABEL_SIZE = 34
OBJECTIVE_TOTAL_LABEL_SIZE = 28
OBJECTIVE_LEGEND_SIZE = 24
OBJECTIVE_COMPONENT_LABEL_SIZE = 20
OBJECTIVE_GRID = {"axis": "y", "color": "#cbd5e1", "linewidth": 1.15, "alpha": 0.95}
OBJECTIVE_COMPONENT_STYLES = {
    "Driving": {"hatch": None, "alpha": 0.96},
    "Walking": {"hatch": "..", "alpha": 0.82},
    "Cost": {"hatch": "///", "alpha": 0.68},
}
OBJECTIVE_EDGE_COLOR = "#475569"
OBJECTIVE_BAR_LEGEND_KWARGS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, 1.19),
    "ncol": 3,
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.95,
    "fontsize": OBJECTIVE_LEGEND_SIZE,
    "borderpad": 1.05,
    "labelspacing": 0.9,
    "handletextpad": 1.05,
    "columnspacing": 2.0,
}

RUNTIME_FIGSIZE = (24.0, 15.0)
RUNTIME_FIG_DPI = 300
RUNTIME_PNG_DPI = 600
RUNTIME_FONT_FAMILY = "Times New Roman"
RUNTIME_AXIS_LABEL_SIZE = 40
RUNTIME_TICK_LABEL_SIZE = 35
RUNTIME_LEGEND_SIZE = 34
RUNTIME_INSET_LABEL_SIZE = 32
RUNTIME_TITLE_SIZE = 38

METHOD_COLORS = {
    "Optimization": "#1f3a5f",
    "Heuristic": "#0f766e",
    "ACO": "#1d4ed8",
    "PSO": "#b45309",
    "GA": "#be123c",
}
METHOD_MARKERS = {
    "Optimization": "o",
    "Heuristic": "s",
    "ACO": "^",
    "PSO": "D",
    "GA": "v",
}

METHOD_ORDER_BY_SCALE = {
    "small": ["Optimization", "Heuristic", "ACO", "PSO", "GA"],
    "medium_large": ["Heuristic", "ACO", "PSO", "GA"],
    "sumo": ["Optimization", "Heuristic", "ACO", "PSO", "GA"],
}
RUNTIME_METHOD_ORDER = ["Optimization", "Heuristic", "ACO", "PSO", "GA"]
RUNTIME_MAIN_METHODS = {
    "small": ["Optimization", "Heuristic"],
    "medium_large": ["Heuristic", "ACO", "PSO", "GA"],
    "sumo": ["Heuristic", "ACO", "PSO", "GA"],
}

TRAFFIC_ORDER = {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3}
EXPERIMENT_LABELS = {
    "network_size": "Network Size (Nodes)",
    "edge_density": "Edge Density",
    "parking_supply": "Parking Supply",
    "maximum_walking_distance": "Maximum Walking Distance (km)",
    "traffic_level": "Traffic Level",
    "parking_budget": "Parking Budget",
    "parking_cost_range": "Parking Cost Values",
    "spot_availability": "Spot Availability",
    "spots_per_lot": "Spots per Lot",
    "objective_weight_alpha": "Objective Weight Alpha",
}

FINAL_CASE_PARAM_MAP = {
    "C1": ("network_size", "baseline"),
    "C2": ("edge_density", "final_case_variant"),
    "C3": ("parking_supply", "final_case_variant"),
    "C4": ("maximum_walking_distance", "final_case_variant"),
    "C5": ("traffic_level", "final_case_variant"),
    "C6": ("traffic_level", "final_case_variant"),
    "C7": ("objective_weight_alpha", "final_case_variant"),
    "C8": ("objective_weight_alpha", "final_case_variant"),
    "C9": ("parking_budget", "final_case_variant"),
    "C10": ("parking_cost_range", "final_case_variant"),
}

COMBINED_HEADER = [
    "source_file",
    "method_raw",
    "method",
    "dat_file",
    "nodes",
    "edges",
    "parking_lots",
    "S",
    "D",
    "Alpha",
    "w",
    "alpha_used",
    "Wmax",
    "Cmax",
    "feasible",
    "solver_status",
    "parking_node",
    "drive_time_hr",
    "walk_time_hr",
    "travel_time_hr",
    "walk_distance_km",
    "parking_cost",
    "objective",
    "objective_value",
    "runtime_ms",
    "historical_runtime_status",
    "drive_path",
    "walk_path",
    "drive_time_min_used",
    "walk_time_min_used",
    "dataset_group",
    "scale",
    "experiment_name",
    "parameter_value_key",
    "parameter_label",
    "scenario_code",
]


def parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_runtime_to_ms(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    direct = parse_float(text)
    if direct is not None:
        return direct
    parts = text.split(":")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        return None
    hours, minutes, seconds, centiseconds = (int(part) for part in parts)
    total_seconds = hours * 3600 + minutes * 60 + seconds + (centiseconds / 100.0)
    return total_seconds * 1000.0


def parse_int(value: object) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    return int(round(number))


def is_truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "solved", "optimal"}


def normalize_method_name(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text in {"OPT", "OPTIMUM", "OPTIMIZATION", "OPTIMISATION"}:
        return "Optimization"
    if text in {"HEURISTIC", "PROPOSED HEURISTIC"}:
        return "Heuristic"
    if text == "ACO":
        return "ACO"
    if text in {"PSO", "PSO_CHAOTIC", "CHAOTIC_PSO"}:
        return "PSO"
    if text in {"GA", "GA_ROUTE_EVOLUTION"}:
        return "GA"
    return None


def format_number(value: float, *, decimals: int = 2) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def detect_scale(dat_file: str, nodes: int | None) -> str | None:
    if dat_file == "sumo_case.dat":
        return "sumo"
    if dat_file.startswith("N2055_") or (nodes == 2055):
        return "sumo"
    if nodes is not None and nodes <= 50:
        return "small"
    if nodes in {500, 1000, 1500, 2000, 2500}:
        return "medium_large"
    return None


def parse_case_code(dat_file: str) -> str | None:
    match = re.search(r"_C(\d+)_", dat_file)
    if not match:
        return None
    return f"C{int(match.group(1))}"


def sort_key_for_cost_range(label: str) -> tuple[float, float]:
    numbers = [float(value) for value in re.findall(FLOAT_RE, label)]
    if len(numbers) >= 2:
        return (numbers[0], numbers[1])
    if numbers:
        return (numbers[0], numbers[0])
    return (math.inf, math.inf)


def parse_ps_experiment(dat_file: str) -> tuple[str, object, str] | None:
    if "_PS_" not in dat_file:
        return None
    suffix = dat_file.split("_PS_", 1)[1].removesuffix(".dat")
    suffix = re.sub(r"_I\d+$", "", suffix)

    if suffix.startswith("edge_density_"):
        value = int(re.findall(r"(\d+)", suffix)[-1])
        return ("edge_density", value, str(value))
    if suffix.startswith("parking_supply_P"):
        value = int(re.findall(r"(\d+)", suffix)[0])
        return ("parking_supply", value, f"{value}%")
    if suffix.startswith("walking_distance_W"):
        code = re.findall(r"W(\d+)", suffix)[0]
        value = float(code) / 100.0
        return ("maximum_walking_distance", value, f"{value:.2f}")
    if suffix.startswith("traffic_"):
        level = suffix.removeprefix("traffic_").replace("_", " ").title()
        if level == "Very Low":
            return ("traffic_level", "Very Low", "Very Low")
        if level == "Low":
            return ("traffic_level", "Low", "Low")
        if level == "Medium":
            return ("traffic_level", "Medium", "Medium")
        if level == "High":
            return ("traffic_level", "High", "High")
    if suffix.startswith("parking_budget_Cmax"):
        raw = re.findall(r"Cmax(\d+)", suffix)[0]
        value = float(raw) / 10.0
        return ("parking_budget", value, format_number(value, decimals=1))
    if suffix.startswith("parking_cost_"):
        low, high = re.findall(r"parking_cost_(\d+)_(\d+)", suffix)[0]
        label = f"{float(low) / 10.0:.1f}-{float(high) / 10.0:.1f}"
        return ("parking_cost_range", label, label)
    if suffix.startswith("spot_availability_"):
        value = int(re.findall(r"(\d+)", suffix)[0])
        return ("spot_availability", value, f"{value}%")
    if suffix.startswith("spots_per_lot_"):
        numbers = re.findall(r"(\d+)", suffix)
        if len(numbers) >= 2:
            label = f"{int(numbers[0])}-{int(numbers[1])}"
            return ("spots_per_lot", label, label)
        return ("spots_per_lot", suffix.removeprefix("spots_per_lot_"), suffix.removeprefix("spots_per_lot_"))
    return None


def parse_final_case_experiment(dat_file: str, row: dict[str, str]) -> tuple[str, object, str, str] | None:
    case_code = parse_case_code(dat_file)
    if not case_code or case_code not in FINAL_CASE_PARAM_MAP:
        return None

    experiment_name, dataset_group = FINAL_CASE_PARAM_MAP[case_code]
    if case_code == "C1":
        nodes = parse_int(row.get("nodes"))
        if nodes is None:
            return None
        return (experiment_name, nodes, str(nodes), dataset_group)
    if case_code == "C2":
        return (experiment_name, "Dense", "Dense", dataset_group)
    if case_code == "C3":
        lots = parse_int(row.get("parking_lots"))
        if lots is None:
            return None
        baseline_guess = max(1, round(lots / 2))
        return (experiment_name, lots, f"{lots} lots", dataset_group if lots != baseline_guess else "baseline")
    if case_code == "C4":
        value = parse_float(row.get("Wmax"))
        if value is None:
            return None
        return (experiment_name, value, f"{value:.2f}", dataset_group)
    if case_code in {"C5", "C6"}:
        label = "Medium" if case_code == "C5" else "High"
        return (experiment_name, label, label, dataset_group)
    if case_code in {"C7", "C8"}:
        alpha = parse_float(row.get("Alpha"))
        if alpha is None:
            return None
        return (experiment_name, alpha, format_number(alpha, decimals=2), dataset_group)
    if case_code == "C9":
        value = parse_float(row.get("Cmax"))
        if value is None:
            return None
        return (experiment_name, value, format_number(value, decimals=1), dataset_group)
    if case_code == "C10":
        cost = parse_float(row.get("parking_cost"))
        if cost is None:
            return None
        return (experiment_name, "high_cost_variant", "15.0-30.0", dataset_group)
    return None


def parse_experiment_metadata(row: dict[str, str]) -> tuple[str | None, object | None, str | None, str | None]:
    dat_file = str(row.get("dat_file", ""))
    ps_meta = parse_ps_experiment(dat_file)
    if ps_meta is not None:
        return (ps_meta[0], ps_meta[1], ps_meta[2], "parameter_sensitivity")

    final_meta = parse_final_case_experiment(dat_file, row)
    if final_meta is not None:
        return final_meta

    return (None, None, None, None)


def canonical_alpha(row: dict[str, str]) -> float | None:
    return parse_float(row.get("Alpha")) or parse_float(row.get("w"))


def drive_time_minutes(row: dict[str, str]) -> float | None:
    return parse_float(row.get("non_normalized_drive_time_min")) or (
        parse_float(row.get("drive_time_hr")) * 60.0 if parse_float(row.get("drive_time_hr")) is not None else None
    )


def walk_time_minutes(row: dict[str, str]) -> float | None:
    return parse_float(row.get("non_normalized_walk_time_min")) or (
        parse_float(row.get("walk_time_hr")) * 60.0 if parse_float(row.get("walk_time_hr")) is not None else None
    )


def plotted_objective_value(alpha: float | None, drive_min: float | None, walk_min: float | None, parking_cost: float | None) -> float | None:
    if alpha is None or drive_min is None or walk_min is None or parking_cost is None:
        return None
    return alpha * drive_min + alpha * walk_min + (1.0 - alpha) * parking_cost


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def standardize_source_row(row: dict[str, str], *, source_file: Path) -> dict[str, str] | None:
    method = normalize_method_name(row.get("method"))
    if method is None:
        return None

    dat_file = str(row.get("dat_file", "")).strip()
    nodes = parse_int(row.get("nodes"))
    scale = detect_scale(dat_file, nodes)
    experiment_name, parameter_value_key, parameter_label, dataset_group = parse_experiment_metadata(row)
    alpha = canonical_alpha(row)
    drive_hr = parse_float(row.get("drive_time_hr"))
    walk_hr = parse_float(row.get("walk_time_hr"))
    drive_min = drive_time_minutes(row)
    walk_min = walk_time_minutes(row)
    parking_cost = parse_float(row.get("parking_cost"))
    objective_value = plotted_objective_value(alpha, drive_min, walk_min, parking_cost)
    runtime_ms = parse_runtime_to_ms(row.get("runtime_ms"))
    historical_runtime_status = ""
    if method == "Optimization":
        historical_runtime_ms = parse_runtime_to_ms(row.get("HistoricalOPLRuntime"))
        if historical_runtime_ms is not None:
            runtime_ms = historical_runtime_ms
        historical_runtime_status = str(row.get("HistoricalOPLStatus", ""))

    if row.get("travel_time_hr"):
        travel_hr = parse_float(row.get("travel_time_hr"))
    elif drive_hr is not None and walk_hr is not None:
        travel_hr = drive_hr + walk_hr
    else:
        travel_hr = None

    return {
        "source_file": str(source_file.relative_to(ROOT)),
        "method_raw": str(row.get("method", "")),
        "method": method,
        "dat_file": dat_file,
        "nodes": "" if nodes is None else str(nodes),
        "edges": str(row.get("edges", "")),
        "parking_lots": str(row.get("parking_lots", "")),
        "S": str(row.get("S", row.get("source", ""))),
        "D": str(row.get("D", row.get("destination", ""))),
        "Alpha": str(row.get("Alpha", "")),
        "w": str(row.get("w", "")),
        "alpha_used": "" if alpha is None else f"{alpha:.10g}",
        "Wmax": str(row.get("Wmax", "")),
        "Cmax": str(row.get("Cmax", "")),
        "feasible": "TRUE" if is_truthy(row.get("feasible")) else "FALSE",
        "solver_status": str(row.get("solver_status", "")),
        "parking_node": str(row.get("parking_node", "")),
        "drive_time_hr": "" if drive_hr is None else f"{drive_hr:.10g}",
        "walk_time_hr": "" if walk_hr is None else f"{walk_hr:.10g}",
        "travel_time_hr": "" if travel_hr is None else f"{travel_hr:.10g}",
        "walk_distance_km": str(row.get("walk_distance_km", "")),
        "parking_cost": str(row.get("parking_cost", "")),
        "objective": str(row.get("objective", "")),
        "objective_value": "" if objective_value is None else f"{objective_value:.10g}",
        "runtime_ms": "" if runtime_ms is None else f"{runtime_ms:.10g}",
        "historical_runtime_status": historical_runtime_status,
        "drive_path": str(row.get("drive_path", row.get("driving_path", ""))),
        "walk_path": str(row.get("walk_path", row.get("walking_edge", ""))),
        "drive_time_min_used": "" if drive_min is None else f"{drive_min:.10g}",
        "walk_time_min_used": "" if walk_min is None else f"{walk_min:.10g}",
        "dataset_group": str(dataset_group or ""),
        "scale": str(scale or ""),
        "experiment_name": str(experiment_name or ""),
        "parameter_value_key": "" if parameter_value_key is None else str(parameter_value_key),
        "parameter_label": str(parameter_label or ""),
        "scenario_code": str(parse_case_code(dat_file) or ""),
    }


def build_master_csv() -> list[dict[str, str]]:
    combined_rows: list[dict[str, str]] = []
    for _, source_file in SOURCE_FILES.items():
        if not source_file.exists():
            continue
        for row in load_csv_rows(source_file):
            standardized = standardize_source_row(row, source_file=source_file)
            if standardized is not None:
                combined_rows.append(standardized)

    combined_rows.sort(
        key=lambda row: (
            {"small": 0, "medium_large": 1, "sumo": 2}.get(row["scale"], 9),
            row["experiment_name"],
            parse_int(row["nodes"]) or 999999,
            row["dat_file"],
            row["method"],
        )
    )

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_HEADER)
        writer.writeheader()
        writer.writerows(combined_rows)
    return combined_rows


def validate_master_rows(rows: list[dict[str, str]]) -> None:
    required = ["method", "dat_file", "nodes", "Alpha", "drive_time_hr", "walk_time_hr", "parking_cost", "runtime_ms"]
    missing = [column for column in required if not any(str(row.get(column, "")).strip() for row in rows)]
    if missing:
        raise RuntimeError(f"Combined master CSV is missing required usable columns: {', '.join(missing)}")


def remove_old_graph_outputs() -> list[str]:
    deleted: list[str] = []
    for directory in OUTPUT_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
                path.unlink()
                deleted.append(str(path.relative_to(ROOT)))
    return deleted


def sort_parameter_values(experiment_name: str, values: Iterable[str]) -> list[str]:
    values = list(dict.fromkeys(values))
    if experiment_name == "network_size":
        return sorted(values, key=lambda value: int(float(value)))
    if experiment_name == "edge_density":
        return sorted(values, key=lambda value: int(float(value)))
    if experiment_name in {"parking_supply", "spot_availability"}:
        return sorted(values, key=lambda value: int(re.findall(r"\d+", value)[0]))
    if experiment_name == "maximum_walking_distance":
        return sorted(values, key=lambda value: float(value))
    if experiment_name == "traffic_level":
        return sorted(values, key=lambda value: TRAFFIC_ORDER.get(value, 99))
    if experiment_name == "parking_budget":
        return sorted(values, key=lambda value: float(value))
    if experiment_name == "parking_cost_range":
        return sorted(values, key=sort_key_for_cost_range)
    if experiment_name == "spots_per_lot":
        return sorted(values, key=lambda value: [int(number) for number in re.findall(r"\d+", value)] or [999])
    if experiment_name == "objective_weight_alpha":
        return sorted(values, key=lambda value: float(value))
    return sorted(values)


def display_parameter_labels(experiment_name: str, values: list[str]) -> list[str]:
    if experiment_name in {"parking_cost_range", "spots_per_lot"}:
        labels = []
        for value in values:
            numbers = re.findall(FLOAT_RE, str(value))
            labels.append(numbers[0] if numbers else str(value))
        return labels
    return values


def objective_rows_for_scale(rows: list[dict[str, str]], scale: str) -> dict[str, list[dict[str, str]]]:
    method_order = METHOD_ORDER_BY_SCALE[scale]
    eligible = [
        row
        for row in rows
        if row["scale"] == scale
        and row["method"] in method_order
        and row["feasible"] == "TRUE"
        and row["experiment_name"]
    ]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        if row["experiment_name"] == "network_size":
            if row["dataset_group"] != "baseline":
                continue
            grouped["network_size"].append(row)
            continue

        if row["dataset_group"] != "parameter_sensitivity":
            continue
        grouped[row["experiment_name"]].append(row)

    return grouped


def complete_method_rows(
    rows: list[dict[str, str]],
    *,
    method_order: list[str],
    experiment_name: str,
) -> tuple[list[str], dict[tuple[str, str], dict[str, str]], dict[str, list[str]]]:
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    parameter_values: list[str] = []
    for row in rows:
        key = (row["parameter_label"], row["method"])
        by_pair[key] = row
        parameter_values.append(row["parameter_label"])

    ordered_values = sort_parameter_values(experiment_name, parameter_values)
    included_values: list[str] = []
    available_methods: dict[str, list[str]] = {}
    for value in ordered_values:
        present_methods = [method for method in method_order if (value, method) in by_pair]
        if present_methods:
            included_values.append(value)
            available_methods[value] = present_methods
    return included_values, by_pair, available_methods


def plot_objective_graph(
    *,
    scale: str,
    experiment_name: str,
    rows: list[dict[str, str]],
    generated_files: list[str],
) -> list[str]:
    def layout_for_methods(count: int) -> tuple[float, float, float]:
        if count >= 5:
            return 0.26, 0.08, 0.70
        if count == 4:
            return 0.30, 0.09, 0.64
        if count == 3:
            return 0.36, 0.10, 0.58
        return 0.46, 0.12, 0.52

    def add_component_text(
        rect: Any,
        label_text: str,
        color: str,
        min_height: float,
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
        if height >= min_height:
            ax.text(label_x, label_y, label_text, ha="center", va="center", rotation=0, fontsize=inside_size, color=color, fontweight="bold")
            reserved[bar_key] = max(reserved.get(bar_key, 0.0), float(rect.get_y() + rect.get_height()))
            return float(rect.get_y() + rect.get_height())
        if height >= min_height * 0.62:
            ax.text(label_x, label_y, label_text, ha="center", va="center", rotation=0, fontsize=small_size, color=color, fontweight="bold")
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
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
        )
        reserved[bar_key] = y_text
        return y_text

    def mark_tiny_component(rect: Any, color: str, min_height: float) -> None:
        height = float(rect.get_height())
        if not np.isfinite(height) or height <= 0 or height >= min_height * 0.25:
            return
        y_boundary = float(rect.get_y())
        x_left = float(rect.get_x() + rect.get_width() * 0.08)
        x_right = float(rect.get_x() + rect.get_width() * 0.92)
        ax.hlines(y_boundary, x_left, x_right, colors="white", linewidth=5.2, zorder=6)
        ax.hlines(y_boundary, x_left, x_right, colors=color, linewidth=2.4, zorder=7)

    def add_tiny_cost_text(rect: Any, label_text: str, min_height: float, reserved: dict[float, float]) -> float:
        y_top = float(rect.get_y() + rect.get_height())
        label_x = float(rect.get_x() + rect.get_width() / 2.0)
        label_y = y_top - max(min_height * 0.70, 0.62)
        bar_key = round(label_x, 8)
        ax.text(
            label_x,
            label_y,
            label_text,
            ha="center",
            va="center",
            rotation=0,
            fontsize=15.0,
            color="#0f172a",
            fontweight="bold",
        )
        reserved[bar_key] = max(reserved.get(bar_key, 0.0), y_top)
        return y_top

    method_order = METHOD_ORDER_BY_SCALE[scale]
    values, by_pair, available_methods = complete_method_rows(
        rows,
        method_order=method_order,
        experiment_name=experiment_name,
    )
    if not values:
        return []

    fig_width = max(24.0, 7.0 + len(values) * 3.0 + len(method_order) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 18.0), dpi=OBJECTIVE_DPI)
    ax.set_facecolor("white")
    ax.grid(True, **OBJECTIVE_GRID)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    bar_width, inner_gap, group_gap = layout_for_methods(len(method_order))
    group_span = len(method_order) * bar_width + max(0, len(method_order) - 1) * inner_gap
    group_step = group_span + group_gap
    x = np.arange(len(values), dtype=float) * group_step
    max_total = 0.0
    max_label_top = 0.0
    reserved_tops: dict[float, float] = {}
    objective_entries: list[dict[str, float | str]] = []

    for method_index, method in enumerate(method_order):
        positions = x - group_span / 2.0 + bar_width / 2.0 + method_index * (bar_width + inner_gap)
        driving = []
        walking = []
        cost = []
        totals = []
        raw_driving = []
        raw_walking = []
        raw_cost = []
        for value in values:
            if method not in available_methods.get(value, []):
                driving.append(np.nan)
                walking.append(np.nan)
                cost.append(np.nan)
                totals.append(np.nan)
                raw_driving.append(np.nan)
                raw_walking.append(np.nan)
                raw_cost.append(np.nan)
                continue

            row = by_pair[(value, method)]
            alpha = canonical_alpha(row)
            drive_min = parse_float(row["drive_time_min_used"])
            walk_min = parse_float(row["walk_time_min_used"])
            parking_cost = parse_float(row["parking_cost"])
            if alpha is None or drive_min is None or walk_min is None or parking_cost is None:
                driving.append(np.nan)
                walking.append(np.nan)
                cost.append(np.nan)
                totals.append(np.nan)
                raw_driving.append(np.nan)
                raw_walking.append(np.nan)
                raw_cost.append(np.nan)
                continue
            drive_component = alpha * drive_min
            walk_component = alpha * walk_min
            cost_component = (1.0 - alpha) * parking_cost
            total = drive_component + walk_component + cost_component
            driving.append(drive_component)
            walking.append(walk_component)
            cost.append(cost_component)
            totals.append(total)
            raw_driving.append(drive_min)
            raw_walking.append(walk_min)
            raw_cost.append(parking_cost)
            max_total = max(max_total, total)

        drive_bars = ax.bar(
            positions,
            driving,
            width=bar_width,
            color=METHOD_COLORS[method],
            edgecolor=OBJECTIVE_EDGE_COLOR,
            linewidth=1.8,
            alpha=OBJECTIVE_COMPONENT_STYLES["Driving"]["alpha"],
        )
        walk_bars = ax.bar(
            positions,
            walking,
            width=bar_width,
            bottom=driving,
            color=METHOD_COLORS[method],
            edgecolor=OBJECTIVE_EDGE_COLOR,
            linewidth=1.8,
            hatch=OBJECTIVE_COMPONENT_STYLES["Walking"]["hatch"],
            alpha=OBJECTIVE_COMPONENT_STYLES["Walking"]["alpha"],
        )
        cost_bars = ax.bar(
            positions,
            cost,
            width=bar_width,
            bottom=np.array(driving) + np.array(walking),
            color=METHOD_COLORS[method],
            edgecolor=OBJECTIVE_EDGE_COLOR,
            linewidth=1.8,
            hatch=OBJECTIVE_COMPONENT_STYLES["Cost"]["hatch"],
            alpha=OBJECTIVE_COMPONENT_STYLES["Cost"]["alpha"],
        )

        if SHOW_OBJECTIVE_COMPONENT_LABELS:
            min_component_height = max(0.20, max_total * 0.050)
            outside_offset = max(0.08, max_total * 0.016)
            outside_gap = max(0.08, max_total * 0.017)
            for drive_rect, walk_rect, cost_rect, total, drive_value, walk_value, cost_value in zip(
                drive_bars.patches,
                walk_bars.patches,
                cost_bars.patches,
                totals,
                raw_driving,
                raw_walking,
                raw_cost,
            ):
                if np.isfinite(drive_value):
                    max_label_top = max(
                        max_label_top,
                        add_component_text(
                            drive_rect,
                            f"D {drive_value:.2f}",
                            "#0f172a",
                            min_component_height,
                            reserved_tops,
                            inside_size=18.0,
                            small_size=15.0,
                            outside_offset=outside_offset,
                            outside_gap=outside_gap,
                        ),
                    )
                if np.isfinite(walk_value):
                    max_label_top = max(
                        max_label_top,
                        add_component_text(
                            walk_rect,
                            f"W {walk_value:.2f}",
                            "#0f172a",
                            min_component_height,
                            reserved_tops,
                            inside_size=18.0,
                            small_size=15.0,
                            outside_offset=outside_offset,
                            outside_gap=outside_gap,
                        ),
                    )
                if np.isfinite(cost_value):
                    cost_height = float(cost_rect.get_height())
                    if np.isfinite(cost_height) and cost_height > 0 and cost_height < min_component_height * 0.25:
                        mark_tiny_component(cost_rect, OBJECTIVE_EDGE_COLOR, min_component_height)
                        max_label_top = max(
                            max_label_top,
                            add_tiny_cost_text(cost_rect, f"C {cost_value:.2f}", min_component_height, reserved_tops),
                        )
                    else:
                        max_label_top = max(
                            max_label_top,
                            add_component_text(
                                cost_rect,
                                f"C {cost_value:.2f}",
                                "#0f172a",
                                min_component_height,
                                reserved_tops,
                                inside_size=18.0,
                                small_size=15.0,
                                outside_offset=outside_offset,
                                outside_gap=outside_gap,
                            ),
                        )
                if not np.isfinite(total):
                    continue
                bar_center_x = float(cost_rect.get_x() + cost_rect.get_width() / 2.0)
                expected_x = float(cost_rect.get_x() + cost_rect.get_width() / 2.0)
                if abs(bar_center_x - expected_x) > 1e-9:
                    raise AssertionError("Objective label x-coordinate does not match the bar center.")
                objective_entries.append(
                    {
                        "x": bar_center_x,
                        "total": float(total),
                        "edge": OBJECTIVE_EDGE_COLOR,
                        "objective": float(total),
                    }
                )
        for total in totals:
            if not np.isfinite(total):
                continue

    y_range = max(max_total, 1.0)
    objective_offset = 0.015 * y_range
    objective_gap = 0.024 * y_range
    objective_font_size = 24 if len(method_order) <= 3 else 22
    for item in objective_entries:
        bar_key = round(float(item["x"]), 8)
        objective_y = max(float(item["total"]) + objective_offset, reserved_tops.get(bar_key, float(item["total"])) + objective_gap)
        ax.text(
            float(item["x"]),
            objective_y,
            f"{float(item['objective']):.3f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=objective_font_size,
            color=str(item["edge"]),
            fontweight="bold",
        )
        reserved_tops[bar_key] = objective_y
        max_label_top = max(max_label_top, objective_y)

    xtick_labels = display_parameter_labels(experiment_name, values)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=OBJECTIVE_TICK_LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=OBJECTIVE_TICK_LABEL_SIZE, width=1.8, length=7)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.set_ylabel("Objective Value", fontsize=OBJECTIVE_AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_xlabel(
        EXPERIMENT_LABELS.get(experiment_name, experiment_name.replace("_", " ").title()),
        fontsize=OBJECTIVE_AXIS_LABEL_SIZE,
        fontweight="bold",
    )
    legend_handles = []
    component_labels = {
        "Driving": "Driving Time",
        "Walking": "Walking Time",
        "Cost": "Parking Cost",
    }
    for method in method_order:
        legend_handles.append(
            Patch(
                facecolor=METHOD_COLORS[method],
                edgecolor=OBJECTIVE_EDGE_COLOR,
                alpha=OBJECTIVE_COMPONENT_STYLES["Driving"]["alpha"],
                label=f"{method} - {component_labels['Driving']}",
            )
        )
        legend_handles.append(
            Patch(
                facecolor=METHOD_COLORS[method],
                edgecolor=OBJECTIVE_EDGE_COLOR,
                hatch=OBJECTIVE_COMPONENT_STYLES["Walking"]["hatch"],
                alpha=OBJECTIVE_COMPONENT_STYLES["Walking"]["alpha"],
                label=f"{method} - {component_labels['Walking']}",
            )
        )
        legend_handles.append(
            Patch(
                facecolor=METHOD_COLORS[method],
                edgecolor=OBJECTIVE_EDGE_COLOR,
                hatch=OBJECTIVE_COMPONENT_STYLES["Cost"]["hatch"],
                alpha=OBJECTIVE_COMPONENT_STYLES["Cost"]["alpha"],
                label=f"{method} - {component_labels['Cost']}",
            )
        )

    bar_legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3 if len(method_order) <= 3 else 4,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        fontsize=OBJECTIVE_LEGEND_SIZE,
        borderpad=1.05,
        labelspacing=0.9,
        handletextpad=1.05,
        columnspacing=2.0,
    )
    bar_legend.get_frame().set_edgecolor("#94a3b8")
    bar_legend.get_frame().set_linewidth(1.0)
    for text in bar_legend.get_texts():
        text.set_fontweight("bold")
    ax.set_ylim(0, max(max_total * 1.18, max_label_top + 0.05 * y_range) if max_total else 1.0)
    ax.margins(x=0.08)
    fig.subplots_adjust(left=0.085, right=0.988, bottom=0.16, top=0.77)

    stem = f"{experiment_name}_objective_value"
    output_dir = OUTPUT_DIRS[scale]
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, **OBJECTIVE_SAVEFIG_KWARGS)
    fig.savefig(pdf_path, **OBJECTIVE_SAVEFIG_KWARGS)
    plt.close(fig)

    generated_files.extend([str(png_path.relative_to(ROOT)), str(pdf_path.relative_to(ROOT))])
    return [str(png_path.relative_to(ROOT)), str(pdf_path.relative_to(ROOT))]


def configure_runtime_rcparams() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": RUNTIME_FONT_FAMILY,
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": RUNTIME_AXIS_LABEL_SIZE,
            "xtick.labelsize": RUNTIME_TICK_LABEL_SIZE,
            "ytick.labelsize": RUNTIME_TICK_LABEL_SIZE,
            "legend.fontsize": RUNTIME_LEGEND_SIZE,
            "axes.titlesize": RUNTIME_TITLE_SIZE,
            "figure.titlesize": RUNTIME_TITLE_SIZE,
        }
    )


def style_runtime_axes(ax: plt.Axes, *, y_grid: bool = False) -> None:
    ax.grid(False)
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(y_grid, which="major", color="#d1d5db", linewidth=1.0, alpha=0.95)
    ax.tick_params(labelsize=RUNTIME_TICK_LABEL_SIZE, colors="#111827")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def runtime_rows_for_scale(rows: list[dict[str, str]], scale: str) -> tuple[dict[str, dict[int, float]], dict[int, str]]:
    runtime_by_method: dict[str, dict[int, float]] = {method: {} for method in RUNTIME_METHOD_ORDER}
    optimization_status_by_node: dict[int, str] = {}
    for row in rows:
        if row["scale"] != scale or row["dataset_group"] != "baseline":
            continue
        if row["experiment_name"] != "network_size":
            continue
        if row["method"] not in runtime_by_method:
            continue
        nodes = parse_int(row["nodes"])
        runtime_ms = parse_float(row["runtime_ms"])
        if nodes is None or runtime_ms is None:
            continue
        runtime_by_method[row["method"]][nodes] = runtime_ms / 1000.0 / 60.0
        if row["method"] == "Optimization":
            optimization_status_by_node[nodes] = str(row.get("historical_runtime_status", "")).strip().lower()
    return runtime_by_method, optimization_status_by_node


def generate_scalable_runtime_graph(
    scale: str,
    runtime_by_method: dict[str, dict[int, float]],
    target_nodes: list[int],
    generated_files: list[str],
) -> list[str]:
    scalable_methods = [method for method in ["ACO", "PSO", "GA"] if runtime_by_method.get(method)]
    if not scalable_methods:
        return []

    fig, ax = plt.subplots(figsize=RUNTIME_FIGSIZE, dpi=RUNTIME_FIG_DPI, constrained_layout=True)
    style_runtime_axes(ax, y_grid=True)
    legend_handles: list[Line2D] = []
    max_runtime = 0.0

    for method in scalable_methods:
        values = runtime_by_method.get(method, {})
        valid_nodes = sorted(node for node in target_nodes if node in values)
        if not valid_nodes:
            continue
        y_values = [values[node] for node in valid_nodes]
        ax.plot(
            valid_nodes,
            y_values,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=4.0,
            markersize=12.0,
            markerfacecolor="white",
            markeredgewidth=2.3,
            linestyle="-",
            label=method,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=4.0,
                markersize=12.0,
                markerfacecolor="white",
                markeredgewidth=2.3,
                linestyle="-",
                label=method,
            )
        )
        max_runtime = max(max_runtime, max(y_values))

    ax.set_xlabel("Node Size")
    ax.set_ylabel("Runtime (minutes)")
    ax.set_xticks(target_nodes)
    ax.set_xticklabels([str(node) for node in target_nodes])
    ax.set_xlim(min(target_nodes), max(target_nodes))
    if max_runtime > 0:
        ax.set_ylim(0, max_runtime * 1.12)
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        edgecolor="#cbd5e1",
        facecolor="white",
        fontsize=RUNTIME_LEGEND_SIZE,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=2.0,
    )

    output_dir = OUTPUT_DIRS[scale]
    png_path = output_dir / "runtime_hours_scalable_methods.png"
    pdf_path = output_dir / "runtime_hours_scalable_methods.pdf"
    fig.savefig(png_path, dpi=RUNTIME_PNG_DPI, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    generated_files.extend([str(png_path.relative_to(ROOT)), str(pdf_path.relative_to(ROOT))])
    return [str(png_path.relative_to(ROOT)), str(pdf_path.relative_to(ROOT))]


def generate_runtime_graph(scale: str, rows: list[dict[str, str]], generated_files: list[str]) -> list[str]:
    configure_runtime_rcparams()
    runtime_by_method, optimization_status_by_node = runtime_rows_for_scale(rows, scale)
    target_nodes = sorted({node for values in runtime_by_method.values() for node in values.keys()})
    if not target_nodes:
        return []

    fig, ax = plt.subplots(figsize=RUNTIME_FIGSIZE, dpi=RUNTIME_FIG_DPI, constrained_layout=True)
    style_runtime_axes(ax, y_grid=True)
    legend_handles: list[Line2D] = []
    for method in RUNTIME_MAIN_METHODS[scale]:
        values = runtime_by_method.get(method, {})
        if method == "Optimization":
            valid_nodes = sorted(
                node
                for node in target_nodes
                if node in values and "crashed" not in optimization_status_by_node.get(node, "")
            )
        else:
            valid_nodes = sorted(node for node in target_nodes if node in values)
        if not valid_nodes:
            continue
        y_values = [values[node] for node in valid_nodes]
        ax.plot(
            valid_nodes,
            y_values,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=4.0 if method == "Optimization" else 3.6,
            markersize=12.0 if method == "Optimization" else 11.0,
            markerfacecolor="white",
            markeredgewidth=2.3,
            linestyle="-",
            label=method,
            zorder=5 if method == "Optimization" else 4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=4.0 if method == "Optimization" else 3.6,
                markersize=12.0 if method == "Optimization" else 11.0,
                markerfacecolor="white",
                markeredgewidth=2.3,
                linestyle="-",
                label=method,
            )
        )

        if method == "Optimization":
            crashed_nodes = sorted(
                node for node, status in optimization_status_by_node.items() if "crashed" in status and node in values
            )
            if crashed_nodes and valid_nodes:
                crash_node = crashed_nodes[0]
                prior_nodes = [node for node in valid_nodes if node < crash_node]
                if prior_nodes:
                    last_valid = max(prior_nodes)
                    crash_y = values[crash_node]
                    ax.plot(
                        [last_valid, crash_node],
                        [values[last_valid], crash_y],
                        color=METHOD_COLORS[method],
                        linestyle=":",
                        linewidth=2.6,
                        label="_nolegend_",
                    )
                    ax.scatter(
                        [crash_node],
                        [crash_y],
                        marker="x",
                        s=180,
                        linewidths=3.2,
                        color=METHOD_COLORS[method],
                        label="_nolegend_",
                        zorder=6,
                    )
                    ax.annotate(
                        "After\n69 hours,\ncrashed due\nto memory\nlimit",
                        xy=(crash_node, crash_y),
                        xytext=(crash_node, crash_y * 0.88),
                        textcoords="data",
                        ha="center",
                        va="top",
                        fontsize=30,
                        color="#7f1d1d",
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.9,
                        },
                        zorder=7,
                    )
            else:
                last_valid = max(valid_nodes)
                next_missing = next((node for node in target_nodes if node > last_valid), None)
                if next_missing is not None:
                    ax.plot(
                        [last_valid, next_missing],
                        [values[last_valid], values[last_valid] * 1.10],
                        color=METHOD_COLORS[method],
                        linestyle=":",
                        linewidth=2.6,
                        label="_nolegend_",
                    )
                    ax.scatter(
                        [next_missing],
                        [values[last_valid] * 1.10],
                        marker="x",
                        s=180,
                        linewidths=3.2,
                        color=METHOD_COLORS[method],
                        label="_nolegend_",
                        zorder=6,
                    )

    ax.set_xlabel("Node Size")
    ax.set_ylabel("Runtime (minutes)")
    ax.set_xticks(target_nodes)
    ax.set_xticklabels([str(node) for node in target_nodes])
    ax.set_xlim(min(target_nodes), max(target_nodes) + (2 if len(target_nodes) > 1 else 50))
    all_runtime_values = [value for values in runtime_by_method.values() for value in values.values()]
    if all_runtime_values:
        ax.set_ylim(0, max(all_runtime_values) * 1.08)
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        edgecolor="#cbd5e1",
        facecolor="white",
        fontsize=RUNTIME_LEGEND_SIZE,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=2.0,
    )

    if scale == "small":
        inset = inset_axes(
            ax,
            width="48%",
            height="44%",
            loc="lower left",
            bbox_to_anchor=(0.06, 0.15, 0.78, 0.78),
            bbox_transform=ax.transAxes,
            borderpad=0.6,
        )
        style_runtime_axes(inset, y_grid=True)
        inset.set_facecolor("white")
        inset.patch.set_alpha(1.0)
        inset.set_zorder(10)

        inset_methods = ["ACO", "PSO", "GA"]
        inset_max = 0.0
        for method in inset_methods:
            values = runtime_by_method.get(method, {})
            valid_nodes = sorted(node for node in target_nodes if node in values)
            if not valid_nodes:
                continue
            y_values = [values[node] for node in valid_nodes]
            inset.plot(
                valid_nodes,
                y_values,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=3.2,
                markersize=10.5,
                markerfacecolor="white",
                markeredgewidth=2.1,
                linestyle="-",
                label=method,
            )
            inset_max = max(inset_max, max(y_values))

        inset.set_title("Scalable Methods", fontsize=RUNTIME_TITLE_SIZE, pad=4)
        inset.set_xlabel("Nodes", fontsize=RUNTIME_INSET_LABEL_SIZE, labelpad=3)
        inset.set_ylabel("Runtime (minutes)", fontsize=RUNTIME_INSET_LABEL_SIZE, labelpad=3)
        inset.set_xticks([node for node in target_nodes if node != 50])
        inset.set_xlim(10, 50)
        inset.tick_params(labelsize=RUNTIME_INSET_LABEL_SIZE, pad=2)
        if inset_max > 0:
            inset.set_ylim(0, inset_max * 1.15)
        inset.legend(
            loc="upper right",
            ncol=1,
            frameon=True,
            edgecolor="#cbd5e1",
            facecolor="white",
            fontsize=RUNTIME_INSET_LABEL_SIZE - 2,
            borderpad=0.45,
            labelspacing=0.32,
            handlelength=1.8,
            columnspacing=0.7,
        )

    output_dir = OUTPUT_DIRS[scale]
    png_path = output_dir / "runtime_hours.png"
    pdf_path = output_dir / "runtime_hours.pdf"
    fig.savefig(png_path, dpi=RUNTIME_PNG_DPI, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    generated_files.extend([str(png_path.relative_to(ROOT)), str(pdf_path.relative_to(ROOT))])
    outputs = [str(png_path.relative_to(ROOT)), str(pdf_path.relative_to(ROOT))]
    outputs.extend(generate_scalable_runtime_graph(scale, runtime_by_method, target_nodes, generated_files))
    return outputs


def validate_one_objective_example(rows: list[dict[str, str]]) -> dict[str, float | str] | None:
    for row in rows:
        if row["feasible"] != "TRUE":
            continue
        alpha = canonical_alpha(row)
        drive_min = parse_float(row["drive_time_min_used"])
        walk_min = parse_float(row["walk_time_min_used"])
        parking_cost = parse_float(row["parking_cost"])
        if alpha is None or drive_min is None or walk_min is None or parking_cost is None:
            continue
        if not math.isclose(alpha, 0.9, abs_tol=1e-9):
            continue
        weighted_drive = alpha * drive_min
        weighted_walk = alpha * walk_min
        weighted_cost = (1.0 - alpha) * parking_cost
        return {
            "dat_file": row["dat_file"],
            "method": row["method"],
            "alpha": alpha,
            "raw_driving_min": drive_min,
            "raw_walking_min": walk_min,
            "raw_cost": parking_cost,
            "weighted_driving": weighted_drive,
            "weighted_walking": weighted_walk,
            "weighted_cost": weighted_cost,
            "plotted_total": weighted_drive + weighted_walk + weighted_cost,
        }
    return None


def main() -> int:
    combined_rows = build_master_csv()
    validate_master_rows(combined_rows)
    deleted_files = remove_old_graph_outputs()

    plot_rows = load_csv_rows(MASTER_CSV)
    generated_files: list[str] = []
    objective_counts = {"small": 0, "medium_large": 0, "sumo": 0}
    skipped: list[str] = []

    for scale in ["small", "medium_large", "sumo"]:
        experiments = objective_rows_for_scale(plot_rows, scale)
        for experiment_name, rows in sorted(experiments.items()):
            outputs = plot_objective_graph(
                scale=scale,
                experiment_name=experiment_name,
                rows=rows,
                generated_files=generated_files,
            )
            if outputs:
                objective_counts[scale] += 1
            else:
                skipped.append(f"{scale}:{experiment_name}")

    runtime_counts = {"small": 0, "medium_large": 0, "sumo": 0}
    runtime_outputs: list[str] = []
    for scale in ["small", "medium_large", "sumo"]:
        outputs = generate_runtime_graph(scale, plot_rows, generated_files)
        if outputs:
            runtime_counts[scale] += 1
            runtime_outputs.extend(outputs)
    validation = validate_one_objective_example(plot_rows)

    print(f"Combined master CSV: {MASTER_CSV}")
    print("Objective graphs generated:")
    print(f"  Small-scale: {objective_counts['small']}")
    print(f"  Medium/Large: {objective_counts['medium_large']}")
    print(f"  SUMO: {objective_counts['sumo']}")
    print("Runtime graphs generated:")
    print(f"  Small-scale: {runtime_counts['small']}")
    print(f"  Medium/Large: {runtime_counts['medium_large']}")
    print(f"  SUMO: {runtime_counts['sumo']}")
    print("Skipped:")
    print("  driving-only")
    print("  walking-only")
    print("  cost-only")
    if skipped:
        for item in skipped:
            print(f"  incomplete objective coverage: {item}")
    print(f"Output directory: {RESULTS_DIR}")
    print("Generated files:")
    for path in generated_files:
        print(f"  {path}")
    print("Deleted old graph files:")
    if deleted_files:
        for path in deleted_files:
            print(f"  {path}")
    else:
        print("  none")
    if validation:
        print("Validation sample:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
