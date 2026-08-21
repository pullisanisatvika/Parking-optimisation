#!/usr/bin/env python3
"""Convert a SUMO .net.xml file into a clean graph export for optimization.

The script writes:
- nodes.csv: node_id, x, y
- edges.csv: edge_id, from_node, to_node, length_m, speed_mps, drive_time_s, num_lanes
- graph_metadata.json: summary counts and units

By default, internal SUMO connector edges are excluded because optimization
models usually want the road graph, not junction-internal movement links.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from build_parking_layer import (
    BASE_PARKING_COST_RANGE,
    MID_PARKING_COST_RANGE,
    ELEVATED_PARKING_COST_RANGE,
    HIGH_PARKING_COST_RANGE,
    assign_costs,
    generate_parking_layer,
    load_csv,
    repair_high_cost_feasibility,
    write_csv,
)


try:
    import sumolib  # type: ignore
except ImportError:  # pragma: no cover
    sumolib = None


def positive_floats(values):
    return [float(v) for v in values if v is not None and float(v) > 0.0]


def keep_largest_connected_component(nodes, edges):
    undirected = defaultdict(set)
    for edge in edges:
        u = edge["from_node"]
        v = edge["to_node"]
        undirected[u].add(v)
        undirected[v].add(u)

    remaining = set(undirected)
    largest_component = set()
    while remaining:
        start = remaining.pop()
        component = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in undirected[node]:
                if neighbor not in component:
                    component.add(neighbor)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                    queue.append(neighbor)
        if len(component) > len(largest_component):
            largest_component = component

    filtered_nodes = [node for node in nodes if node["node_id"] in largest_component]
    filtered_edges = [
        edge
        for edge in edges
        if edge["from_node"] in largest_component and edge["to_node"] in largest_component
    ]
    return filtered_nodes, filtered_edges, len(largest_component)


def extract_with_sumolib(net_path: Path, include_internal: bool, vehicle_class: str):
    net = sumolib.net.readNet(str(net_path), withInternal=include_internal)

    edges = []
    used_nodes = {}

    for edge in net.getEdges():
        if not include_internal and edge.getFunction() == "internal":
            continue
        if vehicle_class and not edge.allows(vehicle_class):
            continue

        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        if from_node is None or to_node is None:
            continue

        lanes = edge.getLanes()
        lane_lengths = positive_floats([lane.getLength() for lane in lanes])
        lane_speeds = positive_floats([lane.getSpeed() for lane in lanes])

        length_m = edge.getLength() if edge.getLength() > 0 else (max(lane_lengths) if lane_lengths else 0.0)
        speed_mps = edge.getSpeed() if edge.getSpeed() > 0 else (max(lane_speeds) if lane_speeds else 0.0)
        drive_time_s = (length_m / speed_mps) if speed_mps > 0 else None

        used_nodes[from_node.getID()] = from_node
        used_nodes[to_node.getID()] = to_node
        edges.append(
            {
                "edge_id": edge.getID(),
                "from_node": from_node.getID(),
                "to_node": to_node.getID(),
                "length_m": length_m,
                "speed_mps": speed_mps,
                "drive_time_s": drive_time_s,
                "num_lanes": len(lanes),
            }
        )

    nodes = []
    for node_id, node in used_nodes.items():
        x, y = node.getCoord()
        nodes.append({"node_id": node_id, "x": x, "y": y})

    return nodes, edges


def edge_is_allowed_for_vehicle(edge_elem, vehicle_class: str):
    if not vehicle_class:
        return True

    allow = set(edge_elem.attrib.get("allow", "").split())
    disallow = set(edge_elem.attrib.get("disallow", "").split())
    if vehicle_class in disallow:
        return False
    if allow:
        return vehicle_class in allow

    lanes = edge_elem.findall("lane")
    lane_allows = set()
    saw_lane_allow = False
    for lane in lanes:
        lane_disallow = set(lane.attrib.get("disallow", "").split())
        if vehicle_class in lane_disallow:
            continue
        lane_allow = set(lane.attrib.get("allow", "").split())
        if lane_allow:
            saw_lane_allow = True
            lane_allows |= lane_allow
        else:
            return True
    return (vehicle_class in lane_allows) if saw_lane_allow else True


def extract_with_xml(net_path: Path, include_internal: bool, vehicle_class: str):
    root = ET.parse(net_path).getroot()

    node_map = {}
    for junction in root.findall("junction"):
        node_id = junction.attrib["id"]
        node_type = junction.attrib.get("type", "")
        if not include_internal and node_type == "internal":
            continue
        node_map[node_id] = {
            "node_id": node_id,
            "x": float(junction.attrib["x"]),
            "y": float(junction.attrib["y"]),
        }

    edges = []
    used_node_ids = set()
    for edge in root.findall("edge"):
        if not include_internal and edge.attrib.get("function") == "internal":
            continue

        edge_id = edge.attrib.get("id", "")
        if not include_internal and edge_id.startswith(":"):
            continue
        if not edge_is_allowed_for_vehicle(edge, vehicle_class):
            continue

        from_node = edge.attrib.get("from")
        to_node = edge.attrib.get("to")
        if not from_node or not to_node:
            continue

        lanes = edge.findall("lane")
        lane_lengths = positive_floats([lane.attrib.get("length") for lane in lanes if lane.attrib.get("length")])
        lane_speeds = positive_floats([lane.attrib.get("speed") for lane in lanes if lane.attrib.get("speed")])
        length_m = max(lane_lengths) if lane_lengths else 0.0
        speed_mps = max(lane_speeds) if lane_speeds else 0.0
        drive_time_s = (length_m / speed_mps) if speed_mps > 0 else None

        used_node_ids.add(from_node)
        used_node_ids.add(to_node)
        edges.append(
            {
                "edge_id": edge_id,
                "from_node": from_node,
                "to_node": to_node,
                "length_m": length_m,
                "speed_mps": speed_mps,
                "drive_time_s": drive_time_s,
                "num_lanes": len(lanes),
            }
        )

    nodes = [node_map[node_id] for node_id in sorted(used_node_ids) if node_id in node_map]
    return nodes, edges


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def round_rows(rows):
    rounded = []
    for row in rows:
        new_row = dict(row)
        for key in ("x", "y", "length_m", "speed_mps", "drive_time_s"):
            if key in new_row and new_row[key] is not None:
                new_row[key] = round(float(new_row[key]), 4)
        rounded.append(new_row)
    return rounded


def load_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_num(x):
    if isinstance(x, float):
        s = f"{x:.4f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(x)


def fmt_edge(pair):
    return f"<{pair[0]},{pair[1]}>"


def chunk_lines(items, per_line=4, fmt=str):
    if not items:
        return ""
    lines = []
    for i in range(0, len(items), per_line):
        chunk = items[i:i + per_line]
        lines.append(" , ".join(fmt(x) for x in chunk) + " ,")
    return "\n".join(lines)


BASE_PARKING_PCT = 0.05
VAR_PARKING_PCT = 0.10
BASE_MAX_WALK_KM = 0.50
VAR_MAX_WALK_KM = 1.00
BASE_ALPHA = 0.90
VAR_ALPHA_HIGH = 0.90
VAR_ALPHA_LOW = 0.10
BASE_CMAX = 20.5
VAR_CMAX = 15.0
LOW_CONGESTION_RANGE = (50.0, 70.0)
MEDIUM_CONGESTION_RANGE = (30.0, 49.0)
HIGH_CONGESTION_RANGE = (10.0, 29.0)

def export_solver_dat(
    graph_dir: Path,
    parking_dir: Path,
    dat_path: Path,
    alpha: float,
    cmax: float | None,
    big_m: int,
    metadata_fields: dict | None = None,
):
    node_rows = load_csv_rows(graph_dir / "nodes.csv")
    edge_rows = load_csv_rows(graph_dir / "edges.csv")
    parking_rows = load_csv_rows(parking_dir / "parking_nodes.csv")
    walking_rows = load_csv_rows(parking_dir / "walking_edges.csv")
    parking_meta = json.loads((parking_dir / "parking_metadata.json").read_text(encoding="utf-8"))

    ordered_node_ids = [row["node_id"] for row in node_rows]
    node_id_to_int = {node_id: idx for idx, node_id in enumerate(ordered_node_ids, start=1)}
    source_node = parking_meta["source_node"]
    destination_node = parking_meta["destination_node"]
    if source_node not in node_id_to_int or destination_node not in node_id_to_int:
        raise ValueError("Source or destination missing from node mapping.")

    drive_edge_map = {}
    for row in edge_rows:
        from_node = row["from_node"]
        to_node = row["to_node"]
        if from_node == destination_node or to_node == destination_node:
            continue
        edge = {
            "u": node_id_to_int[from_node],
            "v": node_id_to_int[to_node],
            "distance_km": float(row["length_m"]) / 1000.0,
            "time_hr": float(row["drive_time_s"]) / 3600.0,
            "type": "drive",
        }
        key = (edge["u"], edge["v"])
        best = drive_edge_map.get(key)
        if best is None or edge["time_hr"] < best["time_hr"]:
            drive_edge_map[key] = edge
    drive_edges = list(drive_edge_map.values())

    walk_lookup = {row["from_node"]: row for row in walking_rows}
    walk_edge_map = {}
    for parking_row in parking_rows:
        parking_node = parking_row["parking_node_id"]
        if parking_node not in walk_lookup:
            continue
        walk_row = walk_lookup[parking_node]
        p_int = node_id_to_int[parking_node]
        d_int = node_id_to_int[destination_node]
        dist_km = float(walk_row["walk_distance_km"])
        time_hr = float(walk_row["walk_time_hr"])
        walk_edge_map[(p_int, d_int)] = {"u": p_int, "v": d_int, "distance_km": dist_km, "time_hr": time_hr, "type": "walk"}
        walk_edge_map[(d_int, p_int)] = {"u": d_int, "v": p_int, "distance_km": dist_km, "time_hr": time_hr, "type": "walk"}
    walk_edges = list(walk_edge_map.values())

    all_edges = sorted(drive_edges + walk_edges, key=lambda e: (e["u"], e["v"], e["type"]))
    E = [(e["u"], e["v"]) for e in all_edges]
    Distance = [e["distance_km"] for e in all_edges]
    EdgeTime = [e["time_hr"] for e in all_edges]
    EdgeType = [e["type"] for e in all_edges]

    parking_rows_sorted = parking_rows
    L = [node_id_to_int[row["parking_node_id"]] for row in parking_rows_sorted]
    C = [float(row["parking_cost"]) for row in parking_rows_sorted]
    feasible_flags = [int(row["feasible"]) for row in parking_rows_sorted]
    walk_distance = [float(row["walk_distance_km"]) for row in parking_rows_sorted]
    walk_time = [float(row["walk_time_hr"]) for row in parking_rows_sorted]
    q = len(L)
    w_hr = float(parking_meta["max_walk_time_hr"])
    w_km = float(parking_meta["max_walk_km"])
    cmax_value = max(C) if cmax is None and C else (cmax if cmax is not None else 0.0)
    parking_cost_range = parking_meta.get("parking_cost_range")
    cost_mode = parking_meta.get("cost_mode", "baseline")

    dat_path.parent.mkdir(parents=True, exist_ok=True)
    with dat_path.open("w", encoding="utf-8") as f:
        f.write("/*********************************************\n")
        f.write(" * SUMO-derived urban-road dataset\n")
        f.write(" *********************************************/\n\n")
        f.write(f"Nodes = {len(ordered_node_ids)};\n\n")
        f.write("E = {\n")
        f.write(chunk_lines(E, per_line=4, fmt=fmt_edge))
        f.write("\n};\n\n")
        f.write(f"S = {node_id_to_int[source_node]};\t\t// Source of EV\n")
        f.write(f"D = {node_id_to_int[destination_node]};\t\t// Destination node\n\n")
        f.write(f"Q = {q};\t\t// Number of parking lots (all urban parking facilities)\n")
        f.write(f"L = [{','.join(map(str, L))}];\t\t// Parking lot nodes\n")
        f.write(f"C = [{','.join(map(fmt_num, C))}];\t// Parking costs\n")
        f.write(f"FeasibleParking = [{','.join(map(str, feasible_flags))}];\t// 1 if parking lot is within W, else 0\n")
        f.write(f"WalkDistanceToDest = [{','.join(map(fmt_num, walk_distance))}];\t// Walking distance from each parking lot to destination\n")
        f.write(f"WalkTimeToDest = [{','.join(map(fmt_num, walk_time))}];\t// Walking time from each parking lot to destination\n\n")
        f.write(f"W = {fmt_num(w_hr)};\t\t// Maximum walking time (hr)\n")
        f.write(f"W_km = {fmt_num(w_km)};\t\t// Maximum walking distance (km), reference only\n")
        f.write(f"Cmax = {fmt_num(cmax_value)};\t\t// Maximum allowed parking cost\n\n")
        f.write(f"B = {big_m};\t\t// Large constant\n\n")
        f.write(f"Alpha = {fmt_num(alpha)};\t// Objective weight\n")
        f.write(f"w = {fmt_num(alpha)};\t\t// Solver weight, aligned with Alpha\n\n")
        f.write(f'ParkingCostMode = "{cost_mode}";\n')
        if metadata_fields and metadata_fields.get("traffic_level"):
            f.write(f'TrafficLevel = "{metadata_fields["traffic_level"]}";\n')
        if metadata_fields and metadata_fields.get("traffic_speed_range") and len(metadata_fields["traffic_speed_range"]) == 2:
            f.write(
                f"TrafficSpeedRange = [{fmt_num(float(metadata_fields['traffic_speed_range'][0]))},"
                f"{fmt_num(float(metadata_fields['traffic_speed_range'][1]))}];\n"
            )
        if parking_cost_range and len(parking_cost_range) == 2:
            f.write(
                f"ParkingCostRange = [{fmt_num(float(parking_cost_range[0]))},"
                f"{fmt_num(float(parking_cost_range[1]))}];\n"
            )
        if metadata_fields:
            if metadata_fields.get("experiment_type"):
                f.write(f'ExperimentType = "{metadata_fields["experiment_type"]}";\n')
            if metadata_fields.get("scale_category"):
                f.write(f'ScaleCategory = "{metadata_fields["scale_category"]}";\n')
            if metadata_fields.get("varied_parameter"):
                f.write(f'VariedParameter = "{metadata_fields["varied_parameter"]}";\n')
            if metadata_fields.get("active_parameter_value") is not None:
                f.write(f'ActiveParameterValue = "{metadata_fields["active_parameter_value"]}";\n')
            if metadata_fields.get("parking_supply_pct") is not None:
                f.write(f'ParkingSupplyPct = {fmt_num(float(metadata_fields["parking_supply_pct"]))};\n')
            if metadata_fields.get("parameter_label"):
                f.write(f'ParameterLabel = "{metadata_fields["parameter_label"]}";\n')
            f.write("\n")
        f.write("Distance = [\n")
        f.write(chunk_lines(Distance, per_line=4, fmt=fmt_num))
        f.write("\n];\n\n")
        f.write("EdgeTime = [\n")
        f.write(chunk_lines(EdgeTime, per_line=4, fmt=fmt_num))
        f.write("\n];\n\n")
        f.write("EdgeType = [\n")
        f.write(chunk_lines(EdgeType, per_line=8, fmt=lambda x: f'\"{x}\"'))
        f.write("\n];\n")

    map_rows = [
        {"solver_node": idx, "sumo_node_id": node_id}
        for node_id, idx in node_id_to_int.items()
    ]
    write_csv(dat_path.with_name(dat_path.stem + "_node_map.csv"), map_rows, ["solver_node", "sumo_node_id"])

    return {
        "dat_path": str(dat_path),
        "node_map_path": str(dat_path.with_name(dat_path.stem + "_node_map.csv")),
        "nodes": len(ordered_node_ids),
        "edges": len(E),
        "parking_lots": q,
        "source_solver_id": node_id_to_int[source_node],
        "destination_solver_id": node_id_to_int[destination_node],
    }


def write_graph_export(graph_dir: Path, nodes, edges, metadata):
    graph_dir.mkdir(parents=True, exist_ok=True)
    write_csv(graph_dir / "nodes.csv", nodes, ["node_id", "x", "y"])
    write_csv(
        graph_dir / "edges.csv",
        edges,
        ["edge_id", "from_node", "to_node", "length_m", "speed_mps", "drive_time_s", "num_lanes"],
    )
    with (graph_dir / "graph_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_graph_export(graph_dir: Path):
    nodes = load_csv_rows(graph_dir / "nodes.csv")
    edges = load_csv_rows(graph_dir / "edges.csv")
    metadata = json.loads((graph_dir / "graph_metadata.json").read_text(encoding="utf-8"))
    return nodes, edges, metadata


def vary_traffic_edges(edge_rows, speed_range_kmph, seed_label):
    base_mid_kmph = 60.0
    target_mid_kmph = (float(speed_range_kmph[0]) + float(speed_range_kmph[1])) / 2.0
    scale = base_mid_kmph / max(target_mid_kmph, 1e-9)
    varied = []
    for row in edge_rows:
        new_row = dict(row)
        length_m = float(new_row["length_m"])
        base_drive_time_s = float(new_row["drive_time_s"])
        drive_time_s = round(base_drive_time_s * scale, 4)
        speed_mps = length_m / max(drive_time_s, 1e-9)
        new_row["speed_mps"] = round(speed_mps, 4)
        new_row["drive_time_s"] = drive_time_s
        varied.append(new_row)
    return varied


def _read_parking_layer(parking_dir: Path):
    parking_rows = load_csv(parking_dir / "parking_nodes.csv")
    walking_rows = load_csv(parking_dir / "walking_edges.csv")
    metadata = json.loads((parking_dir / "parking_metadata.json").read_text(encoding="utf-8"))
    return parking_rows, walking_rows, metadata


def _write_parking_layer(parking_dir: Path, parking_rows, walking_rows, metadata):
    parking_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        parking_dir / "parking_nodes.csv",
        parking_rows,
        [
            "parking_node_id",
            "x",
            "y",
            "parking_cost",
            "walk_distance_m",
            "walk_distance_km",
            "walk_time_hr",
            "feasible",
            "hop_from_source",
            "source_node",
            "destination_node",
        ],
    )
    write_csv(
        parking_dir / "walking_edges.csv",
        walking_rows,
        ["from_node", "to_node", "walk_distance_m", "walk_distance_km", "walk_time_hr", "feasible"],
    )
    with (parking_dir / "parking_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _clone_base_parking_layer(
    base_parking_dir: Path,
    target_parking_dir: Path,
    *,
    total_nodes: int,
    max_walk_km: float | None = None,
    cmax: float | None = None,
    cost_mode: str | None = None,
):
    parking_rows, walking_rows, metadata = _read_parking_layer(base_parking_dir)
    parking_rows = [dict(row) for row in parking_rows]
    walking_rows = [dict(row) for row in walking_rows]
    metadata = dict(metadata)

    if max_walk_km is not None:
        max_walk_time_hr = round(max_walk_km / 5.0, 4)
        metadata["max_walk_km"] = max_walk_km
        metadata["max_walk_time_hr"] = max_walk_time_hr
        feasible_count = 0
        for row in parking_rows:
            feasible = float(row["walk_distance_km"]) <= max_walk_km + 1e-9
            row["feasible"] = int(feasible)
            if feasible:
                feasible_count += 1
        for row in walking_rows:
            row["feasible"] = int(float(row["walk_distance_km"]) <= max_walk_km + 1e-9)
        metadata["feasible_parking_count"] = feasible_count
        metadata["infeasible_parking_count"] = len(parking_rows) - feasible_count

    if cost_mode is not None:
        range_by_mode = {
            "baseline": BASE_PARKING_COST_RANGE,
            "mid": MID_PARKING_COST_RANGE,
            "elevated": ELEVATED_PARKING_COST_RANGE,
            "high": HIGH_PARKING_COST_RANGE,
        }
        active_range = range_by_mode[cost_mode]
        parking_nodes = [row["parking_node_id"] for row in parking_rows]
        seed = int(metadata.get("seed", 42))
        costs = assign_costs(
            parking_nodes=parking_nodes,
            total_nodes=total_nodes,
            cost_range=active_range,
            rng_seed=seed,
        )
        if cost_mode == "high" and cmax is not None:
            selected_rows = [
                {
                    "node_id": row["parking_node_id"],
                    "walk_distance_km": float(row["walk_distance_km"]),
                    "hop_from_source": int(row["hop_from_source"]),
                    "feasible": bool(int(row["feasible"])),
                }
                for row in parking_rows
            ]
            costs = repair_high_cost_feasibility(selected_rows, costs, cmax, seed)
        for row in parking_rows:
            row["parking_cost"] = costs[row["parking_node_id"]]
        metadata["cost_mode"] = cost_mode
        metadata["parking_cost_range"] = [round(active_range[0], 2), round(active_range[1], 2)]

    _write_parking_layer(target_parking_dir, parking_rows, walking_rows, metadata)


def vary_edges_dense(node_rows, edge_rows, destination_node, seed_label):
    import random
    import math

    rng = random.Random(seed_label)
    coords = {row["node_id"]: (float(row["x"]), float(row["y"])) for row in node_rows}
    existing = {(row["from_node"], row["to_node"]) for row in edge_rows}
    undirected = {tuple(sorted((row["from_node"], row["to_node"]))) for row in edge_rows}
    node_ids = list(coords.keys())
    candidates = []
    for i, u in enumerate(node_ids):
        if u == destination_node:
            continue
        x1, y1 = coords[u]
        for v in node_ids[i + 1:]:
            if v == destination_node:
                continue
            if tuple(sorted((u, v))) in undirected:
                continue
            x2, y2 = coords[v]
            d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if 280.0 <= d <= 700.0:
                candidates.append((d, u, v))
    candidates.sort(reverse=True)

    target_extra = max(3, min(len(candidates), max(10, len(node_ids) // 6)))
    added_pairs = []
    used = set()
    for d, u, v in candidates:
        if len(added_pairs) >= target_extra:
            break
        if u in used and v in used:
            continue
        added_pairs.append((d, u, v))
        used.add(u)
        used.add(v)

    varied = [dict(row) for row in edge_rows]
    next_idx = 1
    for d, u, v in added_pairs:
        speed_mps = rng.uniform(LOW_CONGESTION_RANGE[0], LOW_CONGESTION_RANGE[1]) / 3.6
        drive_time_s = d / speed_mps
        edge_base = f"sumo_dense_{next_idx}"
        next_idx += 1
        varied.append(
            {
                "edge_id": edge_base,
                "from_node": u,
                "to_node": v,
                "length_m": round(d, 4),
                "speed_mps": round(speed_mps, 4),
                "drive_time_s": round(drive_time_s, 4),
                "num_lanes": 1,
            }
        )
        varied.append(
            {
                "edge_id": edge_base + "_rev",
                "from_node": v,
                "to_node": u,
                "length_m": round(d, 4),
                "speed_mps": round(speed_mps, 4),
                "drive_time_s": round(drive_time_s, 4),
                "num_lanes": 1,
            }
        )
    return varied


def generate_ten_variations(
    script_dir: Path,
    base_graph_dir: Path,
    base_parking_dir: Path,
    nodes_count: int,
    big_m: int,
):
    node_rows, edge_rows, graph_meta = load_graph_export(base_graph_dir)
    base_parking_meta = json.loads((base_parking_dir / "parking_metadata.json").read_text(encoding="utf-8"))
    source_node = base_parking_meta["source_node"]
    destination_node = base_parking_meta["destination_node"]
    variations_root = script_dir.parent / "final_cases" / "SUMO"
    working_root = script_dir / "_variation_work"
    variations_root.mkdir(parents=True, exist_ok=True)
    working_root.mkdir(parents=True, exist_ok=True)

    case_specs = {
        "C1": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": BASE_ALPHA, "cmax": BASE_CMAX, "cost_mode": "baseline", "edge_mode": "baseline", "traffic": None},
        "C2": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": BASE_ALPHA, "cmax": BASE_CMAX, "cost_mode": "baseline", "edge_mode": "dense", "traffic": None},
        "C3": {"parking_pct": VAR_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": BASE_ALPHA, "cmax": BASE_CMAX, "cost_mode": "baseline", "edge_mode": "baseline", "traffic": None},
        "C4": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": VAR_MAX_WALK_KM, "alpha": BASE_ALPHA, "cmax": BASE_CMAX, "cost_mode": "baseline", "edge_mode": "baseline", "traffic": None},
        "C5": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": BASE_ALPHA, "cmax": BASE_CMAX, "cost_mode": "baseline", "edge_mode": "baseline", "traffic": MEDIUM_CONGESTION_RANGE},
        "C6": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": BASE_ALPHA, "cmax": BASE_CMAX, "cost_mode": "baseline", "edge_mode": "baseline", "traffic": HIGH_CONGESTION_RANGE},
        "C7": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": VAR_ALPHA_HIGH, "cmax": BASE_CMAX, "cost_mode": "baseline", "edge_mode": "baseline", "traffic": None},
        "C8": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": VAR_ALPHA_LOW, "cmax": BASE_CMAX, "cost_mode": "baseline", "edge_mode": "baseline", "traffic": None},
        "C9": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": BASE_ALPHA, "cmax": VAR_CMAX, "cost_mode": "baseline", "edge_mode": "baseline", "traffic": None},
        "C10": {"parking_pct": BASE_PARKING_PCT, "max_walk_km": BASE_MAX_WALK_KM, "alpha": BASE_ALPHA, "cmax": BASE_CMAX, "cost_mode": "high", "edge_mode": "baseline", "traffic": None},
    }

    for case_name, spec in case_specs.items():
        case_graph_dir = working_root / case_name / "graph"
        case_parking_dir = working_root / case_name / "parking"
        case_edges = [dict(row) for row in edge_rows]
        if spec["edge_mode"] == "dense":
            case_edges = vary_edges_dense(node_rows, case_edges, destination_node, f"dense-{nodes_count}")
        if spec["traffic"] is not None:
            case_edges = vary_traffic_edges(case_edges, spec["traffic"], f"{case_name}-{nodes_count}")
        case_meta = dict(graph_meta)
        case_meta["variation_case"] = case_name
        write_graph_export(case_graph_dir, node_rows, case_edges, case_meta)

        if case_name == "C3":
            generate_parking_layer(
                graph_dir=case_graph_dir,
                output_dir=case_parking_dir,
                source_node=source_node,
                destination_node=destination_node,
                parking_pct=spec["parking_pct"],
                max_walk_km=spec["max_walk_km"],
                cost_mode=spec["cost_mode"],
                seed=42,
                cmax=spec["cmax"],
            )
        elif case_name == "C4":
            _clone_base_parking_layer(
                base_parking_dir=base_parking_dir,
                target_parking_dir=case_parking_dir,
                total_nodes=nodes_count,
                max_walk_km=spec["max_walk_km"],
                cmax=spec["cmax"],
                cost_mode="baseline",
            )
        elif case_name == "C10":
            _clone_base_parking_layer(
                base_parking_dir=base_parking_dir,
                target_parking_dir=case_parking_dir,
                total_nodes=nodes_count,
                cmax=spec["cmax"],
                cost_mode=spec["cost_mode"],
            )
        else:
            _clone_base_parking_layer(
                base_parking_dir=base_parking_dir,
                target_parking_dir=case_parking_dir,
                total_nodes=nodes_count,
                cmax=spec["cmax"],
                cost_mode="baseline",
            )

        dat_path = variations_root / f"N{nodes_count}_{case_name}_I1.dat"
        export_solver_dat(
            graph_dir=case_graph_dir,
            parking_dir=case_parking_dir,
            dat_path=dat_path,
            alpha=spec["alpha"],
            cmax=spec["cmax"],
            big_m=big_m,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net-file",
        default="sumo.net.xml",
        help="Path to the SUMO .net.xml file. Defaults to sumo.net.xml in the SUMO folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="graph_export",
        help="Directory where the exported graph files will be written.",
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Include SUMO internal connector edges and nodes.",
    )
    parser.add_argument(
        "--vehicle-class",
        default="passenger",
        help="Only keep edges drivable by this SUMO vehicle class. Default: passenger.",
    )
    parser.add_argument(
        "--skip-largest-component",
        action="store_true",
        help="Keep all retained components instead of restricting to the largest connected one.",
    )
    parser.add_argument(
        "--skip-parking-layer",
        action="store_true",
        help="Only export the cleaned driving graph and skip parking-node selection.",
    )
    parser.add_argument(
        "--parking-output-dir",
        default="parking_layer",
        help="Directory where parking layer files will be written.",
    )
    parser.add_argument("--source-node", help="Optional source node id for parking selection.")
    parser.add_argument("--destination-node", help="Optional destination node id for parking selection.")
    parser.add_argument(
        "--parking-pct",
        type=float,
        default=0.05,
        help="Parking share target for parking selection. Use 0.05 baseline or 0.10 for supply variation.",
    )
    parser.add_argument(
        "--max-walk-km",
        type=float,
        default=0.50,
        help="Walking distance threshold in km for parking feasibility.",
    )
    parser.add_argument(
        "--parking-cost-mode",
        choices=["baseline", "mid", "elevated", "high"],
        default="baseline",
        help="Parking cost mode for the selected parking nodes.",
    )
    parser.add_argument(
        "--parking-seed",
        type=int,
        default=42,
        help="Seed used when parking cost mode is high.",
    )
    parser.add_argument(
        "--skip-dat-export",
        action="store_true",
        help="Skip exporting the solver-ready OPL-style .dat case file.",
    )
    parser.add_argument(
        "--dat-output",
        default="../final_cases/SUMO/sumo_case.dat",
        help="Path for the solver-ready .dat file, relative to the SUMO folder.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.50,
        help="Objective weight Alpha/w to write into the solver-ready .dat file.",
    )
    parser.add_argument(
        "--cmax",
        type=float,
        default=None,
        help="Maximum allowed parking cost to write into the .dat file. Defaults to max selected parking cost.",
    )
    parser.add_argument(
        "--big-m",
        type=int,
        default=9000,
        help="Large constant B to write into the solver-ready .dat file.",
    )
    parser.add_argument(
        "--skip-10-variations",
        action="store_true",
        help="Skip generating the full C1-C10 SUMO variation set.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    net_path = (script_dir / args.net_file).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    if not net_path.exists():
        parser.error(f"SUMO network file not found: {net_path}")

    if sumolib is not None:
        nodes, edges = extract_with_sumolib(
            net_path,
            include_internal=args.include_internal,
            vehicle_class=args.vehicle_class,
        )
        parser_name = "sumolib"
    else:
        nodes, edges = extract_with_xml(
            net_path,
            include_internal=args.include_internal,
            vehicle_class=args.vehicle_class,
        )
        parser_name = "xml"

    original_node_count = len(nodes)
    original_edge_count = len(edges)
    largest_component_node_count = None
    if not args.skip_largest_component:
        nodes, edges, largest_component_node_count = keep_largest_connected_component(nodes, edges)

    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = sorted(round_rows(nodes), key=lambda row: row["node_id"])
    edges = sorted(round_rows(edges), key=lambda row: row["edge_id"])

    write_csv(
        output_dir / "nodes.csv",
        nodes,
        fieldnames=["node_id", "x", "y"],
    )
    write_csv(
        output_dir / "edges.csv",
        edges,
        fieldnames=["edge_id", "from_node", "to_node", "length_m", "speed_mps", "drive_time_s", "num_lanes"],
    )

    metadata = {
        "net_file": str(net_path),
        "parser": parser_name,
        "include_internal": args.include_internal,
        "vehicle_class": args.vehicle_class,
        "largest_component_only": not args.skip_largest_component,
        "pre_component_node_count": original_node_count,
        "pre_component_edge_count": original_edge_count,
        "largest_component_node_count": largest_component_node_count,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "units": {
            "x": "network coordinate units from SUMO",
            "y": "network coordinate units from SUMO",
            "length_m": "meters",
            "speed_mps": "meters/second",
            "drive_time_s": "seconds",
        },
    }
    with (output_dir / "graph_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Exported {len(nodes)} nodes and {len(edges)} edges to {output_dir}")
    print(f"Parser used: {parser_name}")

    if not args.skip_parking_layer:
        parking_output_dir = (script_dir / args.parking_output_dir).resolve()
        parking_metadata = generate_parking_layer(
            graph_dir=output_dir,
            output_dir=parking_output_dir,
            source_node=args.source_node,
            destination_node=args.destination_node,
            parking_pct=args.parking_pct,
            max_walk_km=args.max_walk_km,
            cost_mode=args.parking_cost_mode,
            seed=args.parking_seed,
        )
        print(
            f"Selected {parking_metadata['selected_parking_count']} parking nodes; "
            f"feasible={parking_metadata['feasible_parking_count']}, "
            f"infeasible={parking_metadata['infeasible_parking_count']}"
        )
        print(f"Parking layer written to {parking_output_dir}")

        if not args.skip_dat_export:
            dat_output_path = (script_dir / args.dat_output).resolve()
            dat_metadata = export_solver_dat(
                graph_dir=output_dir,
                parking_dir=parking_output_dir,
                dat_path=dat_output_path,
                alpha=args.alpha,
                cmax=args.cmax,
                big_m=args.big_m,
            )
            print(f"Solver-ready dataset written to {dat_metadata['dat_path']}")
            print(f"Node-id map written to {dat_metadata['node_map_path']}")

        if not args.skip_10_variations:
            generate_ten_variations(
                script_dir=script_dir,
                base_graph_dir=output_dir,
                base_parking_dir=parking_output_dir,
                nodes_count=len(nodes),
                big_m=args.big_m,
            )
            print(f"Generated C1-C10 SUMO variation set in {script_dir.parent / 'final_cases' / 'SUMO'}")


if __name__ == "__main__":
    sys.exit(main())
