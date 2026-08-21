#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMO_MODULE_DIR = ROOT / "SUMO"
if str(SUMO_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SUMO_MODULE_DIR))

import DATA_GENERATION.final_dataset as fd
import data_logging_pipeline as dlp
import SUMO.preprocess_sumo_network as spn
from SUMO.build_parking_layer import generate_parking_layer


FINAL_CASES = ROOT / "final_cases" / "parameter_sensitivity"
SMALL_DIR = FINAL_CASES / "small"
MEDIUM_DIR = FINAL_CASES / "medium_large"
SUMO_DIR = FINAL_CASES / "SUMO"
MANIFEST_CSV = ROOT / "RESULTS" / "VALIDATION" / "parameter_sensitivity_manifest.csv"
COMBINED_DIR = ROOT / "RESULTS" / "COMBINED_ALL_METHODS"
SUMO_GRAPH_DIR = ROOT / "SUMO" / "graph_export"
SUMO_PARKING_DIR = ROOT / "SUMO" / "parking_layer"
SUMO_WORK_DIR = ROOT / "SUMO" / "parameter_sensitivity_work"

SMALL_NODE = 40
MEDIUM_NODE = 2500
SUMO_NODE = 2055
INSTANCE_ID = 1
BIG_M = 9000

WALK_VALUES = [0.25, 0.50, 0.75, 1.00]
SUPPLY_VALUES = [0.05, 0.10, 0.15, 0.20]
BUDGET_VALUES = [10.0, 15.0, 20.5, 25.0]
COST_RANGES = [
    ("8.0-20.5", fd.BASE_PARKING_COST_RANGE, "baseline_deterministic_varied"),
    ("10.0-22.5", fd.MID_PARKING_COST_RANGE, "mid_deterministic_varied"),
    ("12.5-25.0", fd.ELEVATED_PARKING_COST_RANGE, "elevated_deterministic_varied"),
    ("15.0-30.0", fd.HIGH_PARKING_COST_RANGE, "high_deterministic_varied"),
]
TRAFFIC_VALUES = [
    ("very_low", (60.0, 80.0)),
    ("low", fd.LOW_CONGESTION_RANGE),
    ("medium", fd.MEDIUM_CONGESTION_RANGE),
    ("high", fd.HIGH_CONGESTION_RANGE),
]
EDGE_LEVELS = [
    ("baseline", 0),
    ("dense", 1),
    ("denser", 2),
    ("densest", 3),
]
METHODS = ["heuristics", "ga", "pso", "aco"]


def _fmt_pct(value: float) -> str:
    return f"{int(round(value * 100))}"


def _fmt_walk(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def _fmt_cmax(value: float) -> str:
    return f"{value:.1f}".replace(".", "")


def _metadata(case: dict, scale_category: str, varied_parameter: str, active_value: str, parking_supply_pct: float | None = None):
    case["experiment_type"] = "parameter_sensitivity"
    case["scale_category"] = scale_category
    case["varied_parameter"] = varied_parameter
    case["active_parameter_value"] = active_value
    case["parking_supply_pct"] = parking_supply_pct if parking_supply_pct is not None else len(case["parking_nodes"]) / max(1, case["number_of_nodes"])
    case["parameter_label"] = varied_parameter
    return case


def _case_filename(node_count: int, parameter: str, suffix: str) -> str:
    return f"N{node_count}_PS_{parameter}_{suffix}_I1.dat"


def _edge_density_variant(base_case: dict, level: int) -> dict:
    if level <= 0:
        case = copy.deepcopy(base_case)
        fd.validate_case(case)
        return case

    case = copy.deepcopy(base_case)
    coords = case["coords"]
    n = case["number_of_nodes"]
    rng = fd.random.Random(f"edge-ps-{case['number_of_nodes']}-{case['instance_id']}-{level}")
    driving_nodes = list(range(n - 1))
    existing = {(e["u"], e["v"]) for e in case["drive_edges"]}

    def add_new(u, v, road_class):
        if u == v or (u, v) in existing or (v, u) in existing:
            return False
        lo, hi = fd.LOW_CONGESTION_RANGE
        if road_class == "connector":
            s_lo = lo + 0.20 * (hi - lo)
            s_hi = lo + 0.80 * (hi - lo)
        else:
            s_lo = lo + 0.45 * (hi - lo)
            s_hi = hi
        speed = rng.uniform(s_lo, s_hi)
        fd.add_bidirected_edge(case["drive_edges"], None, coords, u, v, speed, "drive", road_class)
        existing.add((u, v))
        existing.add((v, u))
        return True

    pair_list, _pair_distance, _neighbors = fd._precompute_pairwise_geometry(driving_nodes, coords)
    candidates = [(d, u, v) for d, u, v in pair_list if 0.28 <= d <= 0.70]
    candidates.sort(reverse=True)
    target_extra = max(3, min(len(candidates), (len(driving_nodes) // 6) * level))
    added = 0
    for d, u, v in candidates:
        road_class = "arterial" if d >= 0.45 else "connector"
        if add_new(u, v, road_class):
            added += 1
        if added >= target_extra:
            break
    case["edge_mode"] = f"urban_dense_level_{level}"
    try:
        fd.validate_case(case)
    except RuntimeError as exc:
        if "Feasible parking nodes are not far enough from source in hop distance." not in str(exc):
            raise
    return case


def _walking_variant(base_case: dict, wmax: float) -> dict:
    case = copy.deepcopy(base_case)
    case["max_walking_km"] = wmax
    case["max_walking_time_hr"] = round(wmax / fd.WALK_SPEED_KMPH, 4)
    fd.refresh_parking_feasibility(case)
    fd.validate_case(case)
    return case


def _supply_variant(base_case: dict, parking_pct: float) -> dict:
    case = copy.deepcopy(base_case)
    source = case["source_node"]
    destination = case["destination_node"]
    adj = fd._build_drive_adjacency(case["number_of_nodes"], case["drive_edges"])
    driving_node_ids = list(range(case["number_of_nodes"] - 1))
    parking_nodes, feasible_nodes, parking_info = fd.build_parking_info(
        n_nodes=case["number_of_nodes"],
        driving_node_ids=driving_node_ids,
        coords=case["coords"],
        source=source,
        destination=destination,
        drive_adj=adj,
        parking_pct=parking_pct,
        max_walk_km=case["max_walking_km"],
        parking_search_walk_km=case["max_walking_km"],
    )
    case["parking_nodes"] = parking_nodes
    case["feasible_parking_nodes"] = feasible_nodes
    case["parking_info_by_node"] = parking_info
    case["parking_costs"] = fd.assign_parking_costs_preserving_existing(
        number_of_nodes=case["number_of_nodes"],
        instance_id=case["instance_id"],
        parking_nodes=parking_nodes,
        cost_range=fd.BASE_PARKING_COST_RANGE,
        existing_costs=base_case["parking_costs"],
    )
    case["parking_cost_range"] = fd.BASE_PARKING_COST_RANGE
    case["parking_cost_mode"] = "baseline_deterministic_varied"
    case["walk_edges"] = fd.build_walk_edges_for_all_parking(case["coords"], parking_nodes, destination)
    fd.validate_case(case)
    return case


def _budget_variant(base_case: dict, cmax: float) -> dict:
    case = copy.deepcopy(base_case)
    case["cmax"] = cmax
    fd.validate_case(case)
    return case


def _budget_family_base_case(base_case: dict) -> dict:
    case = copy.deepcopy(base_case)
    minimum_budget = min(BUDGET_VALUES)
    affordable = [
        node
        for node in case["feasible_parking_nodes"]
        if float(case["parking_costs"][node]) <= minimum_budget + 1e-12
    ]
    if affordable:
        fd.validate_case(case)
        return case

    lower, upper = case.get("parking_cost_range", fd.BASE_PARKING_COST_RANGE)
    if min(lower, upper) > minimum_budget + 1e-12:
        raise RuntimeError(
            f"Parking-budget base case cannot be repaired within cost range {case.get('parking_cost_range')}"
        )

    feasible_nodes = sorted(
        case["feasible_parking_nodes"],
        key=lambda node: (
            float(case["parking_info_by_node"][node]["walk_distance_km"]),
            int(case["parking_info_by_node"][node]["hop_from_source"]),
            int(node),
        ),
    )
    if not feasible_nodes:
        raise RuntimeError("Parking-budget base case has no walking-feasible parking nodes to repair.")

    chosen = feasible_nodes[0]
    used_costs = {round(float(cost), 2) for cost in case["parking_costs"].values()}
    start_slot = int(round(max(lower, 0.0) * 100))
    end_slot = int(round(min(minimum_budget, upper) * 100))
    replacement = None
    for slot in range(start_slot, end_slot + 1):
        candidate = round(slot / 100.0, 2)
        if candidate not in used_costs:
            replacement = candidate
            break
    if replacement is None:
        replacement = round(end_slot / 100.0, 2)

    case["parking_costs"][chosen] = replacement
    fd.validate_case(case)
    return case


def _cost_variant(base_case: dict, range_label: str, cost_range: tuple[float, float], mode_label: str) -> dict:
    case = copy.deepcopy(base_case)
    case["parking_costs"] = fd.assign_parking_costs(
        number_of_nodes=case["number_of_nodes"],
        instance_id=case["instance_id"],
        parking_nodes=case["parking_nodes"],
        cost_range=cost_range,
    )
    case["parking_cost_range"] = cost_range
    case["parking_cost_mode"] = mode_label
    if range_label == "15.0-30.0":
        affordable_feasible = [p for p in case["feasible_parking_nodes"] if case["parking_costs"][p] <= case["cmax"]]
        min_affordable = min(2, len(case["feasible_parking_nodes"]))
        if len(affordable_feasible) < min_affordable:
            repaired = fd.vary_parking_cost_only(base_case)
            case["parking_costs"] = repaired["parking_costs"]
            case["parking_cost_mode"] = repaired["parking_cost_mode"]
            case["parking_cost_range"] = repaired["parking_cost_range"]
    fd.validate_case(case)
    return case


def _traffic_variant(base_case: dict, speed_range: tuple[float, float], label: str) -> dict:
    case = copy.deepcopy(base_case)
    base_mid = (fd.LOW_CONGESTION_RANGE[0] + fd.LOW_CONGESTION_RANGE[1]) / 2.0
    target_mid = (speed_range[0] + speed_range[1]) / 2.0
    scale = base_mid / max(target_mid, 1e-9)
    for base_edge, edge in zip(base_case["drive_edges"], case["drive_edges"]):
        scaled_time = round(base_edge["time_hr"] * scale, 4)
        edge["time_hr"] = scaled_time
        edge["speed_kmph"] = round(edge["distance_km"] / max(scaled_time, 1e-9), 2)
    case["traffic_level"] = label
    case["traffic_speed_range"] = speed_range
    fd.validate_case(case)
    return case


def _write_case(case: dict, out_dir: Path, filename: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fd.write_case_dat(case, path)
    return {
        "dataset_name": filename,
        "path": str(path.relative_to(ROOT)),
        "experiment_type": case.get("experiment_type", "parameter_sensitivity"),
        "scale_category": case.get("scale_category", ""),
        "varied_parameter": case.get("varied_parameter", ""),
        "active_parameter_value": case.get("active_parameter_value", ""),
        "node_count": case["number_of_nodes"],
        "directed_edge_count": sum(1 for edge in case["drive_edges"]),
        "parking_lots": len(case["parking_nodes"]),
        "available_parking_lots": len(case["feasible_parking_nodes"]),
        "parking_supply_pct": case.get("parking_supply_pct", len(case["parking_nodes"]) / max(1, case["number_of_nodes"])),
        "wmax": case["max_walking_km"],
        "cmax": case["cmax"],
        "alpha": case["alpha"],
        "traffic_level": case.get("traffic_level", ""),
        "speed_min": case["traffic_speed_range"][0],
        "speed_max": case["traffic_speed_range"][1],
        "parking_cost_min": case["parking_cost_range"][0],
        "parking_cost_max": case["parking_cost_range"][1],
    }


def _subset_supply_case(base_case: dict, max_supply_case: dict, parking_pct: float) -> dict:
    requested_count = fd.compute_parking_count(base_case["number_of_nodes"], parking_pct)
    selected_nodes = max_supply_case["parking_nodes"][:requested_count]
    case = copy.deepcopy(base_case)
    case["parking_nodes"] = selected_nodes
    case["feasible_parking_nodes"] = [
        node for node in selected_nodes if max_supply_case["parking_info_by_node"][node]["feasible"]
    ]
    case["parking_info_by_node"] = {
        node: copy.deepcopy(max_supply_case["parking_info_by_node"][node])
        for node in selected_nodes
    }
    case["parking_costs"] = {
        node: max_supply_case["parking_costs"][node]
        for node in selected_nodes
    }
    case["parking_cost_range"] = fd.BASE_PARKING_COST_RANGE
    case["parking_cost_mode"] = "baseline_deterministic_varied"
    case["walk_edges"] = fd.build_walk_edges_for_all_parking(case["coords"], selected_nodes, case["destination_node"])
    fd.validate_case(case)
    return case


def _write_sumo_parking_subset(source_dir: Path, target_dir: Path, parking_pct: float, total_nodes: int) -> Path:
    with (source_dir / "parking_nodes.csv").open(newline="", encoding="utf-8") as handle:
        parking_reader = csv.DictReader(handle)
        parking_fields = parking_reader.fieldnames or []
        parking_rows = list(parking_reader)
    with (source_dir / "walking_edges.csv").open(newline="", encoding="utf-8") as handle:
        walking_reader = csv.DictReader(handle)
        walking_fields = walking_reader.fieldnames or []
        walking_rows = list(walking_reader)

    selected_count = fd.compute_parking_count(total_nodes, parking_pct)
    selected_rows = parking_rows[:selected_count]
    selected_ids = {row["parking_node_id"] for row in selected_rows}
    selected_walk_rows = [row for row in walking_rows if row["from_node"] in selected_ids]

    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "parking_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=parking_fields)
        writer.writeheader()
        writer.writerows(selected_rows)
    with (target_dir / "walking_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=walking_fields)
        writer.writeheader()
        writer.writerows(selected_walk_rows)

    metadata = json.loads((source_dir / "parking_metadata.json").read_text(encoding="utf-8"))
    feasible_count = sum(1 for row in selected_rows if str(row.get("feasible", "")).strip() == "1")
    metadata["parking_pct"] = parking_pct
    metadata["requested_parking_count"] = selected_count
    metadata["selected_parking_count"] = len(selected_rows)
    metadata["feasible_parking_count"] = feasible_count
    metadata["infeasible_parking_count"] = len(selected_rows) - feasible_count
    (target_dir / "parking_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target_dir


def generate_synthetic_parameter_cases(node_count: int, scale_category: str, out_dir: Path) -> list[dict]:
    base = fd.build_baseline_case(node_count, INSTANCE_ID)
    manifests: list[dict] = []
    max_supply_case = _supply_variant(base, max(SUPPLY_VALUES))
    budget_base = _budget_family_base_case(base)

    for walk in WALK_VALUES:
        case = _metadata(_walking_variant(base, walk), scale_category, "maximum_walking_distance", f"{walk:.2f}", fd.BASE_PARKING_PCT)
        manifests.append(_write_case(case, out_dir, _case_filename(node_count, "walking_distance", f"W{_fmt_walk(walk)}")))

    for pct in SUPPLY_VALUES:
        case = _metadata(_subset_supply_case(base, max_supply_case, pct), scale_category, "parking_supply", f"{int(round(pct * 100))}%", pct)
        manifests.append(_write_case(case, out_dir, _case_filename(node_count, "parking_supply", f"P{_fmt_pct(pct)}")))

    for cmax in BUDGET_VALUES:
        case = _metadata(_budget_variant(budget_base, cmax), scale_category, "parking_budget_cmax", f"{cmax:.1f}", fd.BASE_PARKING_PCT)
        manifests.append(_write_case(case, out_dir, _case_filename(node_count, "parking_budget", f"Cmax{_fmt_cmax(cmax)}")))

    for range_label, cost_range, mode_label in COST_RANGES:
        suffix = range_label.replace(".", "").replace("-", "_")
        case = _metadata(_cost_variant(base, range_label, cost_range, mode_label), scale_category, "parking_cost_variation_mode", range_label, fd.BASE_PARKING_PCT)
        manifests.append(_write_case(case, out_dir, _case_filename(node_count, "parking_cost", suffix)))

    for traffic_label, speed_range in TRAFFIC_VALUES:
        case = _metadata(_traffic_variant(base, speed_range, traffic_label), scale_category, "traffic_level", traffic_label, fd.BASE_PARKING_PCT)
        manifests.append(_write_case(case, out_dir, _case_filename(node_count, "traffic", traffic_label)))

    for edge_label, edge_level in EDGE_LEVELS:
        case = _edge_density_variant(base, edge_level)
        edge_count = sum(1 for edge in case["drive_edges"])
        case = _metadata(case, scale_category, "edge_density", str(edge_count), fd.BASE_PARKING_PCT)
        manifests.append(_write_case(case, out_dir, _case_filename(node_count, "edge_density", f"{edge_label}_{edge_count}")))

    return manifests


def generate_edge_density_cases(node_count: int, scale_category: str, out_dir: Path) -> list[dict]:
    base = fd.build_baseline_case(node_count, INSTANCE_ID)
    manifests: list[dict] = []
    for edge_label, edge_level in EDGE_LEVELS:
        case = _edge_density_variant(base, edge_level)
        edge_count = sum(1 for edge in case["drive_edges"])
        case = _metadata(case, scale_category, "edge_density", str(edge_count), fd.BASE_PARKING_PCT)
        manifests.append(_write_case(case, out_dir, _case_filename(node_count, "edge_density", f"{edge_label}_{edge_count}")))
    return manifests


def _load_sumo_base():
    parking_meta = json.loads((SUMO_PARKING_DIR / "parking_metadata.json").read_text(encoding="utf-8"))
    node_rows, edge_rows, graph_meta = spn.load_graph_export(SUMO_GRAPH_DIR)
    return node_rows, edge_rows, graph_meta, parking_meta


def _sumo_dat_metadata(
    scale_category: str,
    varied_parameter: str,
    active_value: str,
    parking_supply_pct: float | None,
    traffic_level: str | None = None,
    traffic_speed_range: tuple[float, float] | None = None,
):
    return {
        "experiment_type": "parameter_sensitivity",
        "scale_category": scale_category,
        "varied_parameter": varied_parameter,
        "active_parameter_value": active_value,
        "parking_supply_pct": parking_supply_pct,
        "parameter_label": varied_parameter,
        "traffic_level": traffic_level,
        "traffic_speed_range": traffic_speed_range,
    }


def _write_sumo_graph(graph_dir: Path, node_rows, edge_rows, graph_meta: dict):
    graph_dir.mkdir(parents=True, exist_ok=True)
    spn.write_graph_export(graph_dir, node_rows, edge_rows, graph_meta)


def _sumo_manifest_row(filename: str, dat_path: Path, varied_parameter: str, active_value: str, parking_supply_pct: float | None) -> dict:
    text = dat_path.read_text(encoding="utf-8", errors="replace")
    edge_types = re.findall(r'"drive"|"walk"', text)
    drive_edge_count = sum(1 for edge_type in edge_types if edge_type == '"drive"')
    parking_rows = re.search(r"\bL\s*=\s*\[([^\]]*)\]\s*;", text, flags=re.DOTALL)
    feasible_rows = re.search(r"\bFeasibleParking\s*=\s*\[([^\]]*)\]\s*;", text, flags=re.DOTALL)
    walk_match = re.search(r"\bW_km\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*;", text)
    cmax_match = re.search(r"\bCmax\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*;", text)
    cost_range_match = re.search(r"\bParkingCostRange\s*=\s*\[([^\]]*)\]\s*;", text, flags=re.DOTALL)
    traffic_range_match = re.search(r"\bTrafficSpeedRange\s*=\s*\[([^\]]*)\]\s*;", text, flags=re.DOTALL)
    parking_lots = len(re.findall(r"\d+", parking_rows.group(1))) if parking_rows else 0
    available_parking_lots = sum(int(value) for value in re.findall(r"\d+", feasible_rows.group(1))) if feasible_rows else 0
    cost_values = [float(value) for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cost_range_match.group(1))] if cost_range_match else []
    traffic_values = [float(value) for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", traffic_range_match.group(1))] if traffic_range_match else []
    return {
        "dataset_name": filename,
        "path": str(dat_path.relative_to(ROOT)),
        "experiment_type": "parameter_sensitivity",
        "scale_category": "SUMO",
        "varied_parameter": varied_parameter,
        "active_parameter_value": active_value,
        "node_count": SUMO_NODE,
        "directed_edge_count": drive_edge_count,
        "parking_lots": parking_lots,
        "available_parking_lots": available_parking_lots,
        "parking_supply_pct": parking_supply_pct if parking_supply_pct is not None else "",
        "wmax": float(walk_match.group(1)) if walk_match else "",
        "cmax": float(cmax_match.group(1)) if cmax_match else "",
        "alpha": fd.BASE_ALPHA,
        "traffic_level": "",
        "speed_min": traffic_values[0] if len(traffic_values) == 2 else "",
        "speed_max": traffic_values[1] if len(traffic_values) == 2 else "",
        "parking_cost_min": cost_values[0] if len(cost_values) == 2 else "",
        "parking_cost_max": cost_values[1] if len(cost_values) == 2 else "",
    }


def generate_sumo_parameter_cases() -> list[dict]:
    manifests: list[dict] = []
    node_rows, edge_rows, graph_meta, parking_meta = _load_sumo_base()
    base_source = parking_meta["source_node"]
    base_destination = parking_meta["destination_node"]
    SUMO_DIR.mkdir(parents=True, exist_ok=True)
    SUMO_WORK_DIR.mkdir(parents=True, exist_ok=True)

    def export_case(
        name: str,
        varied_parameter: str,
        active_value: str,
        graph_rows,
        parking_dir: Path,
        alpha: float,
        cmax: float | None,
        parking_supply_pct: float | None,
        traffic_level: str | None = None,
        traffic_speed_range: tuple[float, float] | None = None,
    ):
        graph_dir = SUMO_WORK_DIR / name / "graph"
        _write_sumo_graph(graph_dir, node_rows, graph_rows, dict(graph_meta))
        dat_path = SUMO_DIR / name
        spn.export_solver_dat(
            graph_dir=graph_dir,
            parking_dir=parking_dir,
            dat_path=dat_path,
            alpha=alpha,
            cmax=cmax,
            big_m=BIG_M,
            metadata_fields=_sumo_dat_metadata(
                "SUMO",
                varied_parameter,
                active_value,
                parking_supply_pct,
                traffic_level=traffic_level,
                traffic_speed_range=traffic_speed_range,
            ),
        )
        if varied_parameter == "edge_density":
            manifest_preview = _sumo_manifest_row(name, dat_path, varied_parameter, active_value, parking_supply_pct)
            actual_edges = str(manifest_preview["directed_edge_count"])
            text = dat_path.read_text(encoding="utf-8")
            text = re.sub(r'ActiveParameterValue\s*=\s*"[^"]*"\s*;', f'ActiveParameterValue = "{actual_edges}";', text)
            dat_path.write_text(text, encoding="utf-8")
        manifests.append(_sumo_manifest_row(name, dat_path, varied_parameter, active_value, parking_supply_pct))

    for walk in WALK_VALUES:
        parking_dir = SUMO_WORK_DIR / f"walk_{walk}" / "parking"
        spn._clone_base_parking_layer(
            base_parking_dir=SUMO_PARKING_DIR,
            target_parking_dir=parking_dir,
            total_nodes=SUMO_NODE,
            max_walk_km=walk,
            cmax=fd.BASE_CMAX,
            cost_mode="baseline",
        )
        export_case(_case_filename(SUMO_NODE, "walking_distance", f"W{_fmt_walk(walk)}"), "maximum_walking_distance", f"{walk:.2f}", edge_rows, parking_dir, fd.BASE_ALPHA, fd.BASE_CMAX, fd.BASE_PARKING_PCT, traffic_level="low", traffic_speed_range=fd.LOW_CONGESTION_RANGE)

    max_supply_dir = SUMO_WORK_DIR / f"supply_{max(SUPPLY_VALUES)}" / "parking"
    generate_parking_layer(
        graph_dir=SUMO_GRAPH_DIR,
        output_dir=max_supply_dir,
        source_node=base_source,
        destination_node=base_destination,
        parking_pct=max(SUPPLY_VALUES),
        max_walk_km=fd.BASE_MAX_WALK_KM,
        cost_mode="baseline",
        seed=42,
        cmax=fd.BASE_CMAX,
    )
    for pct in SUPPLY_VALUES:
        parking_dir = _write_sumo_parking_subset(max_supply_dir, SUMO_WORK_DIR / f"supply_{pct}" / "parking", pct, SUMO_NODE)
        export_case(_case_filename(SUMO_NODE, "parking_supply", f"P{_fmt_pct(pct)}"), "parking_supply", f"{int(round(pct * 100))}%", edge_rows, parking_dir, fd.BASE_ALPHA, fd.BASE_CMAX, pct, traffic_level="low", traffic_speed_range=fd.LOW_CONGESTION_RANGE)

    for cmax in BUDGET_VALUES:
        parking_dir = SUMO_WORK_DIR / f"budget_{cmax}" / "parking"
        spn._clone_base_parking_layer(
            base_parking_dir=SUMO_PARKING_DIR,
            target_parking_dir=parking_dir,
            total_nodes=SUMO_NODE,
            cmax=cmax,
            cost_mode="baseline",
        )
        export_case(_case_filename(SUMO_NODE, "parking_budget", f"Cmax{_fmt_cmax(cmax)}"), "parking_budget_cmax", f"{cmax:.1f}", edge_rows, parking_dir, fd.BASE_ALPHA, cmax, fd.BASE_PARKING_PCT, traffic_level="low", traffic_speed_range=fd.LOW_CONGESTION_RANGE)

    cost_mode_by_range = {
        "8.0-20.5": "baseline",
        "10.0-22.5": "mid",
        "12.5-25.0": "elevated",
        "15.0-30.0": "high",
    }
    for range_label, _cost_range, _mode_label in COST_RANGES:
        parking_dir = SUMO_WORK_DIR / f"cost_{range_label.replace('.', '').replace('-', '_')}" / "parking"
        spn._clone_base_parking_layer(
            base_parking_dir=SUMO_PARKING_DIR,
            target_parking_dir=parking_dir,
            total_nodes=SUMO_NODE,
            cmax=fd.BASE_CMAX,
            cost_mode=cost_mode_by_range[range_label],
        )
        export_case(_case_filename(SUMO_NODE, "parking_cost", range_label.replace(".", "").replace("-", "_")), "parking_cost_variation_mode", range_label, edge_rows, parking_dir, fd.BASE_ALPHA, fd.BASE_CMAX, fd.BASE_PARKING_PCT, traffic_level="low", traffic_speed_range=fd.LOW_CONGESTION_RANGE)

    for traffic_label, speed_range in TRAFFIC_VALUES:
        varied_edges = spn.vary_traffic_edges(edge_rows, speed_range, f"ps-{traffic_label}-{SUMO_NODE}")
        parking_dir = SUMO_PARKING_DIR
        export_case(_case_filename(SUMO_NODE, "traffic", traffic_label), "traffic_level", traffic_label, varied_edges, parking_dir, fd.BASE_ALPHA, fd.BASE_CMAX, fd.BASE_PARKING_PCT, traffic_level=traffic_label, traffic_speed_range=speed_range)

    for edge_label, edge_level in EDGE_LEVELS:
        varied_edges = [dict(row) for row in edge_rows]
        for level_index in range(edge_level):
            varied_edges = spn.vary_edges_dense(node_rows, varied_edges, base_destination, f"ps-dense-{edge_label}-{level_index}-{SUMO_NODE}")
        directed_edge_count = len(varied_edges)
        export_case(_case_filename(SUMO_NODE, "edge_density", f"{edge_label}_{directed_edge_count}"), "edge_density", str(directed_edge_count), varied_edges, SUMO_PARKING_DIR, fd.BASE_ALPHA, fd.BASE_CMAX, fd.BASE_PARKING_PCT, traffic_level="low", traffic_speed_range=fd.LOW_CONGESTION_RANGE)

    return manifests


def generate_sumo_edge_density_cases() -> list[dict]:
    manifests: list[dict] = []
    node_rows, edge_rows, graph_meta, parking_meta = _load_sumo_base()
    base_destination = parking_meta["destination_node"]
    SUMO_DIR.mkdir(parents=True, exist_ok=True)
    SUMO_WORK_DIR.mkdir(parents=True, exist_ok=True)

    def export_case(
        name: str,
        varied_parameter: str,
        active_value: str,
        graph_rows,
        parking_dir: Path,
    ) -> None:
        graph_dir = SUMO_WORK_DIR / name / "graph"
        _write_sumo_graph(graph_dir, node_rows, graph_rows, dict(graph_meta))
        dat_path = SUMO_DIR / name
        spn.export_solver_dat(
            graph_dir=graph_dir,
            parking_dir=parking_dir,
            dat_path=dat_path,
            alpha=fd.BASE_ALPHA,
            cmax=fd.BASE_CMAX,
            big_m=BIG_M,
            metadata_fields=_sumo_dat_metadata(
                "SUMO",
                varied_parameter,
                active_value,
                fd.BASE_PARKING_PCT,
                traffic_level="low",
                traffic_speed_range=fd.LOW_CONGESTION_RANGE,
            ),
        )
        manifest_preview = _sumo_manifest_row(name, dat_path, varied_parameter, active_value, fd.BASE_PARKING_PCT)
        actual_edges = str(manifest_preview["directed_edge_count"])
        text = dat_path.read_text(encoding="utf-8")
        text = re.sub(r'ActiveParameterValue\s*=\s*"[^"]*"\s*;', f'ActiveParameterValue = "{actual_edges}";', text)
        dat_path.write_text(text, encoding="utf-8")
        manifests.append(_sumo_manifest_row(name, dat_path, varied_parameter, active_value, fd.BASE_PARKING_PCT))

    for edge_label, edge_level in EDGE_LEVELS:
        varied_edges = [dict(row) for row in edge_rows]
        for level_index in range(edge_level):
            varied_edges = spn.vary_edges_dense(node_rows, varied_edges, base_destination, f"ps-dense-{edge_label}-{level_index}-{SUMO_NODE}")
        directed_edge_count = len(varied_edges)
        export_case(
            _case_filename(SUMO_NODE, "edge_density", f"{edge_label}_{directed_edge_count}"),
            "edge_density",
            str(directed_edge_count),
            varied_edges,
            SUMO_PARKING_DIR,
        )
    return manifests


def write_manifest(rows: list[dict]) -> None:
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "path",
        "experiment_type",
        "scale_category",
        "varied_parameter",
        "active_parameter_value",
        "node_count",
        "directed_edge_count",
        "parking_lots",
        "available_parking_lots",
        "parking_supply_pct",
        "wmax",
        "cmax",
        "alpha",
        "traffic_level",
        "speed_min",
        "speed_max",
        "parking_cost_min",
        "parking_cost_max",
    ]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _merged_manifest_rows(replacement_rows: list[dict], varied_parameter: str) -> list[dict]:
    existing_rows = []
    if MANIFEST_CSV.exists():
        with MANIFEST_CSV.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
    kept_rows = [row for row in existing_rows if row.get("varied_parameter") != varied_parameter]
    merged_rows = kept_rows + replacement_rows
    merged_rows.sort(key=lambda row: (row["scale_category"], row["varied_parameter"], str(row["active_parameter_value"]), row["dataset_name"]))
    return merged_rows


def generate_edge_density_only() -> None:
    rows: list[dict] = []
    rows.extend(generate_edge_density_cases(SMALL_NODE, "small", SMALL_DIR))
    rows.extend(generate_edge_density_cases(MEDIUM_NODE, "medium_large", MEDIUM_DIR))
    rows.extend(generate_sumo_edge_density_cases())
    write_manifest(_merged_manifest_rows(rows, "edge_density"))
    print(f"Wrote edge-density manifest rows to {MANIFEST_CSV}")


def generate_all() -> None:
    rows = []
    rows.extend(generate_synthetic_parameter_cases(SMALL_NODE, "small", SMALL_DIR))
    rows.extend(generate_synthetic_parameter_cases(MEDIUM_NODE, "medium_large", MEDIUM_DIR))
    rows.extend(generate_sumo_parameter_cases())
    rows.sort(key=lambda row: (row["scale_category"], row["varied_parameter"], str(row["active_parameter_value"]), row["dataset_name"]))
    write_manifest(rows)
    print(f"Wrote parameter-sensitivity manifest: {MANIFEST_CSV}")


def _partition_name(scale_category: str) -> str:
    return {
        "small": "parameter_sensitivity_small",
        "medium_large": "parameter_sensitivity_medium_large",
        "SUMO": "parameter_sensitivity_sumo",
    }[scale_category]


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if any(dlp._text(v) for v in row.values())]


def _write_partition_combined(partition: str, results_path: Path, routes_path: Path, out_name: str) -> None:
    results = []
    route_by_case = {dlp.normalize_case_name(row.get("case", "")): row for row in _read_csv_rows(routes_path)}
    for row in _read_csv_rows(results_path):
        canon = dlp.canonical_row_from_results("heuristics" if row.get("method") == "HEURISTIC" else "ga", row) if False else None
        canon = {k: row.get(k, "") for k in dlp.CANONICAL_RESULTS_HEADER}
        canon["dat_file"] = dlp.normalize_case_name(canon.get("dat_file", ""))
        results.append(dlp._force_reconcile_row_with_dataset(canon, route_by_case.get(canon["dat_file"])))

    results.sort(key=lambda row: (dlp.parse_case_key(row["dat_file"]), row["method"]))
    out_path = COMBINED_DIR / out_name
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dlp.CANONICAL_RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(results)


def _run_partition(scale_category: str, dat_dir: Path) -> None:
    partition = _partition_name(scale_category)
    dat_paths = sorted(dat_dir.glob("*.dat"))
    if not dat_paths:
        raise RuntimeError(f"No parameter-sensitivity datasets found in {dat_dir}")

    all_rows = []
    for method in METHODS:
        cfg = dlp.method_config(method, partition)
        dlp._ensure_clean_outputs(cfg["results"], cfg["routes"])
        for dat_path in dat_paths:
            print(f"Running {method} on {dat_path.name} -> {partition}", flush=True)
            dlp.run_method_on_dat(method, dat_path, cfg["results"], None)
        dlp.normalize_results_csv(method, cfg["results"])
        dlp.write_routes_from_results(method, cfg["results"], cfg["routes"])
        dlp.normalize_routes_csv(cfg["routes"])
        method_rows = []
        route_by_case = {
            dlp.normalize_case_name(row.get("case", "")): row
            for row in _read_csv_rows(cfg["routes"])
        }
        for row in _read_csv_rows(cfg["results"]):
            canon = dlp.canonical_row_from_results(method, row)
            canon = dlp._force_reconcile_row_with_dataset(canon, route_by_case.get(canon["dat_file"]))
            method_rows.append(canon)
        all_rows.extend(method_rows)

    all_rows.sort(key=lambda row: (dlp.parse_case_key(row["dat_file"]), row["method"]))
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMBINED_DIR / f"results_all_methods_{partition}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dlp.CANONICAL_RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote combined parameter-sensitivity results: {out_path}")


def _combine_partition(scale_category: str) -> None:
    partition = _partition_name(scale_category)
    all_rows = []
    for method in METHODS:
        cfg = dlp.method_config(method, partition)
        if not cfg["results"].exists():
            continue
        route_by_case = {
            dlp.normalize_case_name(row.get("case", "")): row
            for row in _read_csv_rows(cfg["routes"])
        } if cfg["routes"].exists() else {}
        for row in _read_csv_rows(cfg["results"]):
            canon = dlp.canonical_row_from_results(method, row)
            canon = dlp._force_reconcile_row_with_dataset(canon, route_by_case.get(canon["dat_file"]))
            all_rows.append(canon)

    all_rows.sort(key=lambda row: (dlp.parse_case_key(row["dat_file"]), row["method"]))
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMBINED_DIR / f"results_all_methods_{partition}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dlp.CANONICAL_RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote combined parameter-sensitivity results: {out_path}")


def run_methods() -> None:
    _run_partition("small", SMALL_DIR)
    _run_partition("medium_large", MEDIUM_DIR)
    _run_partition("SUMO", SUMO_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and execute explicit parameter-sensitivity datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    sub.add_parser("generate-edge-density")
    sub.add_parser("run")
    sub.add_parser("run-small")
    sub.add_parser("run-medium-large")
    sub.add_parser("run-sumo")
    sub.add_parser("combine-small")
    sub.add_parser("combine-medium-large")
    sub.add_parser("combine-sumo")
    sub.add_parser("all")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "generate":
        generate_all()
    elif args.cmd == "generate-edge-density":
        generate_edge_density_only()
    elif args.cmd == "run":
        run_methods()
    elif args.cmd == "run-small":
        _run_partition("small", SMALL_DIR)
    elif args.cmd == "run-medium-large":
        _run_partition("medium_large", MEDIUM_DIR)
    elif args.cmd == "run-sumo":
        _run_partition("SUMO", SUMO_DIR)
    elif args.cmd == "combine-small":
        _combine_partition("small")
    elif args.cmd == "combine-medium-large":
        _combine_partition("medium_large")
    elif args.cmd == "combine-sumo":
        _combine_partition("SUMO")
    elif args.cmd == "all":
        generate_all()
        run_methods()


if __name__ == "__main__":
    main()
