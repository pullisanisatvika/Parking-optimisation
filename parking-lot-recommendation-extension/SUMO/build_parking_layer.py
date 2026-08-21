#!/usr/bin/env python3
"""Build a parking layer on top of the cleaned SUMO driving graph export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from pathlib import Path


BASE_PARKING_COST_RANGE = (8.0, 20.5)
MID_PARKING_COST_RANGE = (10.0, 22.5)
ELEVATED_PARKING_COST_RANGE = (12.5, 25.0)
HIGH_PARKING_COST_RANGE = (15.0, 30.0)
WALK_SPEED_KMPH = 5.0
DESTINATION_PARKING_RADIUS_CAP_KM = 3.0
DESTINATION_PARKING_RADIUS_DIAG_FRAC = 0.30
DESTINATION_PARKING_RADIUS_WALK_MULT = 2.50
LARGE_CASE_NODE_THRESHOLD = 500
LARGE_CASE_MAX_INFEASIBLE_RATIO = 0.20
LARGE_CASE_MIN_FEASIBLE_RATIO = 0.80
LARGE_CASE_MIN_FEASIBLE_COUNT = 8
BASE_MAX_WALK_KM = 0.50


def load_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_parking_count(total_nodes: int, parking_pct: float):
    raw = int(round(total_nodes * parking_pct))
    upper = max(2, int(math.floor(0.20 * total_nodes)))
    return max(2, min(upper, raw))


def euclidean_m(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def bfs_hops(adj, start):
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in adj.get(node, []):
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


def destination_parking_radius_km(coords, driving_node_ids, destination, max_walk_km):
    xs = [coords[node_id][0] for node_id in driving_node_ids]
    ys = [coords[node_id][1] for node_id in driving_node_ids]
    width_km = (max(xs) - min(xs)) / 1000.0
    height_km = (max(ys) - min(ys)) / 1000.0
    diag_km = math.sqrt(width_km * width_km + height_km * height_km)
    radius = max(
        max_walk_km * DESTINATION_PARKING_RADIUS_WALK_MULT,
        DESTINATION_PARKING_RADIUS_DIAG_FRAC * diag_km,
    )
    return min(radius, DESTINATION_PARKING_RADIUS_CAP_KM)


def _parking_cost_fraction(total_nodes: int, seed: int, parking_node_id: str) -> float:
    rng = random.Random(f"sumo-parking-cost-{total_nodes}-{seed}-{parking_node_id}")
    return rng.random()

def assign_costs(parking_nodes, total_nodes: int, cost_range: tuple[float, float], rng_seed: int):
    lower, upper = cost_range
    costs = {}
    for node_id in sorted(parking_nodes):
        fraction = _parking_cost_fraction(total_nodes, rng_seed, str(node_id))
        costs[node_id] = round(lower + fraction * (upper - lower), 2)

    if len(parking_nodes) > 1 and len(set(costs.values())) == 1:
        ordered_nodes = sorted(parking_nodes)
        denominator = max(1, len(ordered_nodes) - 1)
        for index, node_id in enumerate(ordered_nodes):
            fraction = 0.5 if denominator == 0 else index / denominator
            costs[node_id] = round(lower + fraction * (upper - lower), 2)

    return costs


def repair_high_cost_feasibility(selected_rows, costs, cmax, rng_seed):
    feasible_nodes = [row["node_id"] for row in selected_rows if row["feasible"]]
    affordable_feasible = [node_id for node_id in feasible_nodes if costs[node_id] <= cmax]
    min_affordable_feasible = min(2, len(feasible_nodes))
    if len(affordable_feasible) >= min_affordable_feasible:
        return costs

    rng = random.Random(rng_seed)
    repair_candidates = sorted(
        [row for row in selected_rows if row["feasible"]],
        key=lambda row: (row["walk_distance_km"], row["hop_from_source"]),
    )
    lower = HIGH_PARKING_COST_RANGE[0]
    upper = min(cmax, HIGH_PARKING_COST_RANGE[1])
    needed = min_affordable_feasible - len(affordable_feasible)
    repaired = 0
    for row in repair_candidates:
        node_id = row["node_id"]
        if costs[node_id] <= cmax:
            continue
        costs[node_id] = round(rng.uniform(lower, upper), 2) if lower <= upper else round(cmax, 2)
        repaired += 1
        if repaired >= needed:
            break
    return costs


def collect_candidates(
    driving_node_ids,
    coords,
    source,
    destination,
    drive_adj,
    max_walk_km,
    search_radius_km,
):
    hops = bfs_hops(drive_adj, source)
    feasible = []
    infeasible = []

    for node_id in driving_node_ids:
        if node_id in {source, destination}:
            continue
        if node_id not in hops:
            continue

        walk_distance_m = euclidean_m(coords[node_id], coords[destination])
        walk_distance_km = walk_distance_m / 1000.0
        if walk_distance_km > search_radius_km:
            continue

        item = {
            "node_id": node_id,
            "hop_from_source": hops[node_id],
            "walk_distance_m": walk_distance_m,
            "walk_distance_km": round(walk_distance_km, 4),
            "walk_time_hr": round(walk_distance_km / WALK_SPEED_KMPH, 4),
            "feasible": walk_distance_km <= max_walk_km,
        }
        if item["feasible"]:
            feasible.append(item)
        else:
            infeasible.append(item)

    if max_walk_km > BASE_MAX_WALK_KM:
        feasible.sort(key=lambda row: (row["walk_distance_km"], -row["hop_from_source"]))
    else:
        feasible.sort(key=lambda row: (row["hop_from_source"], -row["walk_distance_km"]), reverse=True)

    infeasible.sort(key=lambda row: (row["hop_from_source"], row["walk_distance_km"]), reverse=True)

    if max_walk_km > BASE_MAX_WALK_KM:
        near_threshold = [row for row in infeasible if row["walk_distance_km"] <= max_walk_km + 0.15]
        if near_threshold:
            infeasible = near_threshold

    return feasible, infeasible


def build_parking_selection(
    total_nodes,
    driving_node_ids,
    coords,
    source,
    destination,
    drive_adj,
    parking_pct,
    max_walk_km,
):
    search_radius_km = destination_parking_radius_km(coords, driving_node_ids, destination, max_walk_km)
    feasible, infeasible = collect_candidates(
        driving_node_ids,
        coords,
        source,
        destination,
        drive_adj,
        max_walk_km,
        search_radius_km,
    )

    if len(feasible) < 2:
        feasible, infeasible = collect_candidates(
            driving_node_ids,
            coords,
            source,
            destination,
            drive_adj,
            max_walk_km,
            search_radius_km=float("inf"),
        )

    if len(feasible) < 2:
        raise RuntimeError("Need at least 2 walking-feasible parking nodes.")

    requested = compute_parking_count(total_nodes, parking_pct)
    target_infeasible = 0
    if infeasible and requested >= 4:
        if total_nodes >= LARGE_CASE_NODE_THRESHOLD:
            target_infeasible = min(
                int(math.floor(requested * LARGE_CASE_MAX_INFEASIBLE_RATIO)),
                len(infeasible),
            )
        else:
            target_infeasible = min(max(1, requested // 4), len(infeasible))

    target_feasible = requested - target_infeasible
    if target_feasible < 2:
        target_feasible = 2
        target_infeasible = min(requested - target_feasible, len(infeasible))

    if total_nodes >= LARGE_CASE_NODE_THRESHOLD:
        min_feasible_target = max(
            2,
            min(len(feasible), LARGE_CASE_MIN_FEASIBLE_COUNT),
            int(math.ceil(requested * LARGE_CASE_MIN_FEASIBLE_RATIO)),
        )
        if target_feasible < min_feasible_target:
            target_feasible = min_feasible_target
            target_infeasible = min(max(0, requested - target_feasible), len(infeasible))
            target_feasible = requested - target_infeasible

    selected = feasible[:target_feasible] + infeasible[:target_infeasible]
    if len(selected) < requested:
        used = {row["node_id"] for row in selected}
        selected.extend([row for row in feasible if row["node_id"] not in used][: requested - len(selected)])
    if len(selected) < requested:
        used = {row["node_id"] for row in selected}
        selected.extend([row for row in infeasible if row["node_id"] not in used][: requested - len(selected)])

    if len(selected) < 2:
        raise RuntimeError("Parking construction did not produce enough parking nodes.")

    feasible_count = sum(1 for row in selected if row["feasible"])
    if feasible_count < 2:
        raise RuntimeError("Need at least 2 feasible parking nodes after selection.")

    return selected, search_radius_km


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", default="graph_export", help="Directory containing nodes.csv and edges.csv.")
    parser.add_argument("--output-dir", default="parking_layer", help="Directory for parking export files.")
    parser.add_argument("--source-node", help="Source node id in the cleaned driving graph.")
    parser.add_argument("--destination-node", help="Destination node id in the cleaned driving graph.")
    parser.add_argument("--parking-pct", type=float, default=0.05, help="Parking share target, e.g. 0.05 or 0.10.")
    parser.add_argument("--max-walk-km", type=float, default=0.50, help="Maximum walking distance threshold in km.")
    parser.add_argument(
        "--cost-mode",
        choices=["baseline", "mid", "elevated", "high"],
        default="baseline",
        help="Parking-cost range mode for the selected parking nodes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for higher random parking costs.")
    return parser


def auto_select_source_destination(coords, max_walk_km: float):
    node_ids = list(coords.keys())
    if len(node_ids) < 2:
        raise RuntimeError("Need at least 2 nodes to choose source and destination.")

    walk_radius_m = max_walk_km * 1000.0
    destination_scores = []
    for dst in node_ids:
        feasible_neighbors = 0
        farthest_src = None
        farthest_distance = -1.0
        for src in node_ids:
            if src == dst:
                continue
            distance = euclidean_m(coords[src], coords[dst])
            if distance <= walk_radius_m:
                feasible_neighbors += 1
            if distance > farthest_distance:
                farthest_distance = distance
                farthest_src = src
        if feasible_neighbors >= 2 and farthest_src is not None:
            destination_scores.append((farthest_distance, feasible_neighbors, farthest_src, dst))

    if destination_scores:
        destination_scores.sort(reverse=True)
        _, _, src, dst = destination_scores[0]
        return src, dst

    best_pair = None
    best_distance = -1.0
    for i, src in enumerate(node_ids):
        for dst in node_ids[i + 1:]:
            distance = euclidean_m(coords[src], coords[dst])
            if distance > best_distance:
                best_distance = distance
                best_pair = (src, dst)
    return best_pair


def generate_parking_layer(
    graph_dir: Path,
    output_dir: Path,
    source_node: str | None = None,
    destination_node: str | None = None,
    parking_pct: float = 0.05,
    max_walk_km: float = 0.50,
    cost_mode: str = "baseline",
    seed: int = 42,
    cmax: float | None = None,
):
    nodes_path = graph_dir / "nodes.csv"
    edges_path = graph_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Expected nodes.csv and edges.csv under {graph_dir}")

    node_rows = load_csv(nodes_path)
    edge_rows = load_csv(edges_path)
    coords = {row["node_id"]: (float(row["x"]), float(row["y"])) for row in node_rows}
    driving_node_ids = list(coords.keys())
    drive_adj = {}
    for node_id in driving_node_ids:
        drive_adj[node_id] = []
    for row in edge_rows:
        drive_adj.setdefault(row["from_node"], []).append(row["to_node"])

    if not source_node or not destination_node:
        source_node, destination_node = auto_select_source_destination(coords, max_walk_km)

    if source_node not in coords:
        raise ValueError(f"Source node not found in cleaned graph: {source_node}")
    if destination_node not in coords:
        raise ValueError(f"Destination node not found in cleaned graph: {destination_node}")
    if source_node == destination_node:
        raise ValueError("Source and destination must be different.")

    selected, search_radius_km = build_parking_selection(
        total_nodes=len(driving_node_ids),
        driving_node_ids=driving_node_ids,
        coords=coords,
        source=source_node,
        destination=destination_node,
        drive_adj=drive_adj,
        parking_pct=parking_pct,
        max_walk_km=max_walk_km,
    )

    parking_nodes = [row["node_id"] for row in selected]
    cost_ranges = {
        "baseline": BASE_PARKING_COST_RANGE,
        "mid": MID_PARKING_COST_RANGE,
        "elevated": ELEVATED_PARKING_COST_RANGE,
        "high": HIGH_PARKING_COST_RANGE,
    }
    active_cost_range = cost_ranges[cost_mode]
    costs = assign_costs(parking_nodes, len(driving_node_ids), active_cost_range, seed)
    if cost_mode == "high" and cmax is not None:
        costs = repair_high_cost_feasibility(selected, costs, cmax, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    parking_rows = []
    walking_rows = []
    for row in selected:
        node_id = row["node_id"]
        parking_rows.append(
            {
                "parking_node_id": node_id,
                "x": round(coords[node_id][0], 4),
                "y": round(coords[node_id][1], 4),
                "parking_cost": costs[node_id],
                "walk_distance_m": round(row["walk_distance_m"], 4),
                "walk_distance_km": row["walk_distance_km"],
                "walk_time_hr": row["walk_time_hr"],
                "feasible": int(row["feasible"]),
                "hop_from_source": row["hop_from_source"],
                "source_node": source_node,
                "destination_node": destination_node,
            }
        )
        walking_rows.append(
            {
                "from_node": node_id,
                "to_node": destination_node,
                "walk_distance_m": round(row["walk_distance_m"], 4),
                "walk_distance_km": row["walk_distance_km"],
                "walk_time_hr": row["walk_time_hr"],
                "feasible": int(row["feasible"]),
            }
        )

    parking_rows.sort(key=lambda row: (row["feasible"], row["walk_distance_km"], row["parking_node_id"]), reverse=True)
    walking_rows.sort(key=lambda row: row["from_node"])

    write_csv(
        output_dir / "parking_nodes.csv",
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
        output_dir / "walking_edges.csv",
        walking_rows,
        ["from_node", "to_node", "walk_distance_m", "walk_distance_km", "walk_time_hr", "feasible"],
    )

    metadata = {
        "graph_dir": str(graph_dir),
        "source_node": source_node,
        "destination_node": destination_node,
        "parking_pct": parking_pct,
        "requested_parking_count": compute_parking_count(len(driving_node_ids), parking_pct),
        "selected_parking_count": len(parking_rows),
        "feasible_parking_count": sum(row["feasible"] for row in parking_rows),
        "infeasible_parking_count": len(parking_rows) - sum(row["feasible"] for row in parking_rows),
        "max_walk_km": max_walk_km,
        "max_walk_time_hr": round(max_walk_km / WALK_SPEED_KMPH, 4),
        "search_radius_km": round(search_radius_km, 4) if math.isfinite(search_radius_km) else None,
        "cost_mode": cost_mode,
        "parking_cost_range": [round(active_cost_range[0], 2), round(active_cost_range[1], 2)],
        "seed": seed,
    }
    with (output_dir / "parking_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    graph_dir = (script_dir / args.graph_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    try:
        metadata = generate_parking_layer(
            graph_dir=graph_dir,
            output_dir=output_dir,
            source_node=args.source_node,
            destination_node=args.destination_node,
            parking_pct=args.parking_pct,
            max_walk_km=args.max_walk_km,
            cost_mode=args.cost_mode,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(
        f"Selected {metadata['selected_parking_count']} parking nodes; "
        f"feasible={metadata['feasible_parking_count']}, "
        f"infeasible={metadata['infeasible_parking_count']}"
    )
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    main()
