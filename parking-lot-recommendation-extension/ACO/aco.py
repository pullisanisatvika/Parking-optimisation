import re
import argparse
import math
import time
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import csv
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spot_level_utils import evaluate_spot, get_feasible_spots, parse_opl_dat as parse_spot_opl_dat

Node = int
Edge = Tuple[int, int]


# -----------------------------
# OPL .dat parsing
# -----------------------------
def parse_opl_dat(dat_text: str) -> Dict[str, Any]:
    FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

    def get_int(name: str) -> int:
        m = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", dat_text)
        if not m:
            raise ValueError(f"Missing integer: {name}")
        return int(m.group(1))

    def get_float(name: str) -> float:
        m = re.search(rf"\b{name}\s*=\s*({FLOAT_RE})\s*;", dat_text)
        if not m:
            raise ValueError(f"Missing float: {name}")
        return float(m.group(1))

    def get_float_optional(name: str, default: float) -> float:
        m = re.search(rf"\b{name}\s*=\s*({FLOAT_RE})\s*;", dat_text)
        if not m:
            return default
        return float(m.group(1))

    Nodes = get_int("Nodes")
    S = get_int("S")
    D = get_int("D")
    Q = get_int("Q")
    dat_w = get_float_optional("w", get_float_optional("Alpha", 0.0))
    W = get_float("W")
    Cmax = get_float_optional("Cmax", math.inf)
    W_km = get_float_optional("W_km", W)

    mE = re.search(r"\bE\s*=\s*\{(.*?)\};", dat_text, flags=re.DOTALL)
    if not mE:
        raise ValueError("Missing E={...};")
    E_pairs = re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", mE.group(1))
    E: List[Tuple[int, int]] = [(int(a), int(b)) for a, b in E_pairs]
    if not E:
        raise ValueError("No edges found in E={...};")

    mL = re.search(r"\bL\s*=\s*\[([^\]]+)\]\s*;", dat_text)
    if not mL:
        raise ValueError("Missing L=[...];")
    L = [int(x) for x in re.findall(r"\d+", mL.group(1))]

    mC = re.search(r"\bC\s*=\s*\[([^\]]+)\]\s*;", dat_text)
    if not mC:
        raise ValueError("Missing C=[...];")
    C = [float(x) for x in re.findall(FLOAT_RE, mC.group(1))]

    def _get_list(name: str, cast=float):
        m = re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]\s*;", dat_text, flags=re.DOTALL)
        if not m:
            return []
        vals = re.findall(FLOAT_RE if cast is float else r"\d+", m.group(1))
        return [cast(v) for v in vals]

    WalkDistanceToDest = _get_list("WalkDistanceToDest", float)
    WalkTimeToDest = _get_list("WalkTimeToDest", float)

    if len(L) != Q:
        raise ValueError(f"L has {len(L)} entries but Q={Q}")
    if len(C) != Q:
        raise ValueError(f"C has {len(C)} entries but Q={Q}")

    def parse_weight_block(name: str) -> Tuple[Dict[Tuple[int, int], float], List[float]]:
        m = re.search(rf"\b{name}\s*=\s*\[(.*?)\];", dat_text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Missing {name}=[...];")
        body = m.group(1)

        map_entries = re.findall(rf"<\s*(\d+)\s*,\s*(\d+)\s*>\s*({FLOAT_RE})", body)
        if map_entries:
            mp: Dict[Tuple[int, int], float] = {}
            for a, b, val in map_entries:
                mp[(int(a), int(b))] = float(val)
            return mp, []

        nums = re.findall(FLOAT_RE, body)
        return {}, [float(x) for x in nums]

    dist_mp, dist_lst = parse_weight_block("Distance")
    time_mp, time_lst = parse_weight_block("EdgeTime")

    def align_weights(
        name: str,
        E: List[Tuple[int, int]],
        mp: Dict[Tuple[int, int], float],
        lst: List[float]
    ) -> List[float]:
        if mp:
            missing = [e for e in E if e not in mp]
            if missing:
                raise ValueError(f"{name} map is missing weights for edges: {missing[:10]} ...")
            return [mp[e] for e in E]
        if len(lst) != len(E):
            raise ValueError(f"{name} length {len(lst)} != |E| {len(E)}")
        return lst

    Distance = align_weights("Distance", E, dist_mp, dist_lst)
    EdgeTime = align_weights("EdgeTime", E, time_mp, time_lst)

    return {
        "Nodes": Nodes,
        "S": S,
        "D": D,
        "Q": Q,
        "w": dat_w,
        "Alpha": dat_w,
        "W": W,
        "W_km": W_km,
        "Cmax": Cmax,
        "L": L,
        "C": C,
        "E": E,
        "Distance": Distance,
        "EdgeTime": EdgeTime,
        "WalkDistanceToDest": WalkDistanceToDest,
        "WalkTimeToDest": WalkTimeToDest,
    }


def make_edge_time_map(E: List[Edge], EdgeTime: List[float]) -> Dict[Edge, float]:
    return {(u, v): float(t) for (u, v), t in zip(E, EdgeTime)}

def make_edge_distance_map(E: List[Edge], Distance: List[float]) -> Dict[Edge, float]:
    return {(u, v): float(d) for (u, v), d in zip(E, Distance)}


def build_adj(edges: List[Edge], edge_time: Dict[Edge, float]) -> Dict[Node, List[Tuple[Node, float]]]:
    adj: Dict[Node, List[Tuple[Node, float]]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append((v, edge_time[(u, v)]))
    return adj

import heapq

def dijkstra_to_goal(nodes: List[Node], adj: Dict[Node, List[Tuple[Node, float]]], goal: Node) -> Dict[Node, float]:
    """
    Reverse Dijkstra: finds shortest time from all nodes to 'goal'.
    """
    rev_adj: Dict[Node, List[Tuple[Node, float]]] = {}
    for u, neighbors in adj.items():
        for v, t in neighbors:
            rev_adj.setdefault(v, []).append((u, t))

    dist = {n: math.inf for n in nodes}
    dist[goal] = 0.0
    pq = [(0.0, goal)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, t in rev_adj.get(u, []):
            if dist[u] + t < dist[v]:
                dist[v] = dist[u] + t
                heapq.heappush(pq, (dist[v], v))
    return dist


# -----------------------------
# Candidate parking filtering
# -----------------------------
def candidate_parking_lots(
    lots: List[Node],
    dst: Node,
    edge_time: Dict[Edge, float],
    cost_map: Dict[Node, float],
    availability: Dict[Node, int],
    Wmax: float,
    Cmax: float,
    walk_time_map: Dict[Node, float],
) -> List[Node]:
    feasible: List[Node] = []
    for p in lots:
        if availability.get(p, 0) <= 0:
            continue
        # Use explicit WalkTimeToDest map if available, otherwise fallback to direct edge
        walk_time = walk_time_map.get(p, edge_time.get((p, dst), math.inf))
        if not math.isfinite(walk_time) or walk_time > Wmax:
            continue
        if cost_map.get(p, math.inf) > Cmax:
            continue
        feasible.append(p)
    return feasible


# -----------------------------
# Ant route construction
# -----------------------------
def choose_next_node(
    current: Node,
    target: Node,
    candidates: List[Tuple[Node, float]],
    pheromone: Dict[Edge, float],
    alpha_pher: float,
    beta_heur: float,
    guidance_map: Optional[Dict[Node, float]] = None,
) -> Tuple[Optional[Node], float]:
    if not candidates:
        return None, math.inf

    desirabilities = []
    total = 0.0

    for nxt, edge_t in candidates:
        tau = pheromone.get((current, nxt), 1.0)
        # heuristic: inverse of (edge_time + distance_to_goal)
        # We use a large penalty if guidance_map is missing or node is unreachable
        dist_to_goal = guidance_map.get(nxt, 999999) if guidance_map else 0
        eta = 1.0 / (edge_t + dist_to_goal + 1e-9)
        val = (tau ** alpha_pher) * (eta ** beta_heur)
        desirabilities.append((nxt, edge_t, val))
        total += val

    if total <= 0.0:
        # Fallback to pure guidance if everything is zero
        if guidance_map:
            nxt, edge_t, _ = min(desirabilities, key=lambda x: (guidance_map.get(x[0], 999999), x[1]))
        else:
            nxt, edge_t, _ = min(desirabilities, key=lambda x: (x[1], x[0]))
        return nxt, edge_t

    r = random.random() * total
    acc = 0.0
    for nxt, edge_t, val in desirabilities:
        acc += val
        if acc >= r:
            return nxt, edge_t

    nxt, edge_t, _ = desirabilities[-1]
    return nxt, edge_t


def construct_ant_route(
    src: Node,
    parking_node: Node,
    adj: Dict[Node, List[Tuple[Node, float]]],
    pheromone: Dict[Edge, float],
    alpha_pher: float,
    beta_heur: float,
    guidance_map: Optional[Dict[Node, float]] = None,
    max_steps: int = 5000,
) -> Tuple[List[Node], float]:
    """
    Build a driving route from src to parking_node using a Backtracking Search (DFS).
    Guided by pheromones and distance-to-goal heuristic.
    """
    if src == parking_node:
        return [src], 0.0

    # Stack stores (current_node, current_path, current_time)
    stack: List[Tuple[Node, List[Node], float]] = [(src, [src], 0.0)]
    visited = {src}
    
    steps = 0
    while stack and steps < max_steps:
        steps += 1
        curr, path, p_time = stack.pop()
        
        if curr == parking_node:
            return path, p_time

        visited.add(curr)
        
        neighbors = adj.get(curr, [])
        scored_candidates = []
        for v, t in neighbors:
            if v in visited or v in path:
                continue
            
            tau = pheromone.get((curr, v), 1.0)
            h_val = guidance_map.get(v, 999999) if guidance_map else 0
            eta = 1.0 / (t + h_val + 1e-9)
            val = (tau ** alpha_pher) * (eta ** beta_heur)
            scored_candidates.append((v, t, val))
            
        # Sort by desirability (higher last for stack.pop() to get it first)
        scored_candidates.sort(key=lambda x: x[2])
        
        for v, t, _ in scored_candidates:
            stack.append((v, path + [v], p_time + t))

    return [], math.inf


# -----------------------------
# Ant solution evaluation
# -----------------------------
def evaluate_ant_solution(
    data: Dict[str, Any],
    spot_index: int,
    src: Node,
    dst: Node,
    adj: Dict[Node, List[Tuple[Node, float]]],
    edge_time: Dict[Edge, float],
    edge_distance: Dict[Edge, float],
    cost_map: Dict[Node, float],
    pheromone: Dict[Edge, float],
    alpha_weight: float,
    alpha_pher: float,
    beta_heur: float,
    walk_time_map: Dict[Node, float],
    walk_distance_map: Dict[Node, float],
    guidance_map: Optional[Dict[Node, float]] = None,
) -> Dict[str, Any]:
    parking_node = int(data["SpotLotNode"][spot_index - 1])
    drive_path, drive_time = construct_ant_route(
        src=src,
        parking_node=parking_node,
        adj=adj,
        pheromone=pheromone,
        alpha_pher=alpha_pher,
        beta_heur=beta_heur,
        guidance_map=guidance_map,
    )

    if not drive_path or not math.isfinite(drive_time):
        return {
            "feasible": False,
            "objective": math.inf,
            "parking_node": parking_node,
            "parking_spot_index": spot_index,
            "drive_path": [],
            "walk_path": [],
            "drive_time": math.inf,
            "walk_time": math.inf,
            "travel_time": math.inf,
            "parking_cost": math.inf,
        }

    spot = evaluate_spot(data, drive_time, spot_index, alpha=alpha_weight)
    if spot is None:
        return {
            "feasible": False,
            "objective": math.inf,
            "parking_node": parking_node,
            "parking_spot_index": spot_index,
            "drive_path": [],
            "walk_path": [],
            "drive_time": math.inf,
            "walk_time": math.inf,
            "travel_time": math.inf,
            "parking_cost": math.inf,
        }

    return {
        **spot,
        "feasible": True,
        "drive_path": drive_path,
        "walk_path": [parking_node, dst],
        "drive_time": spot["drive_time_hr"],
        "walk_time": spot["walk_time_hr"],
        "walk_distance": spot["external_walk_distance_km"],
        "travel_time": spot["travel_time_hr"],
    }


# -----------------------------
# ACO solver
# -----------------------------
def solve_aco_joint_route_and_parking(
    data: Dict[str, Any],
    src: Node,
    dst: Node,
    feasible_spots: List[int],
    adj: Dict[Node, List[Tuple[Node, float]]],
    edge_time: Dict[Edge, float],
    edge_distance: Dict[Edge, float],
    cost_map: Dict[Node, float],
    alpha_weight: float,
    walk_time_map: Dict[Node, float],
    walk_distance_map: Dict[Node, float],
    ants: int = 30,
    iterations: int = 80,
    alpha_pher: float = 1.0,
    beta_heur: float = 2.0,
    evaporation: float = 0.3,
    Q: float = 1.0,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    if not feasible_spots:
        return None

    random.seed(seed)

    # Precalculate guidance maps for each parent lot referenced by a feasible spot.
    guidance_maps: Dict[Node, Dict[Node, float]] = {}
    nodes = list(adj.keys())
    for p in sorted({int(data["SpotLotNode"][spot_index - 1]) for spot_index in feasible_spots}):
        guidance_maps[p] = dijkstra_to_goal(nodes, adj, p)

    pheromone: Dict[Edge, float] = {}
    for u in adj:
        for v, _ in adj[u]:
            pheromone[(u, v)] = 1.0
    spot_pheromone = {spot_index: 1.0 for spot_index in feasible_spots}
    static_spots = {int(spot["parking_spot_index"]): spot for spot in get_feasible_spots(data)}

    global_best: Optional[Dict[str, Any]] = None

    for _ in range(iterations):
        iteration_results: List[Dict[str, Any]] = []

        for _ant in range(ants):
            weighted_spots: list[tuple[int, float]] = []
            total_weight = 0.0
            for spot_index in feasible_spots:
                spot = static_spots[spot_index]
                static_score = (
                    alpha_weight * (
                        float(spot["internal_drive_time_hr"])
                        + float(spot["internal_walk_time_hr"])
                        + float(spot["external_walk_time_hr"])
                    )
                    + (1.0 - alpha_weight) * float(spot["parking_cost"])
                )
                tau = spot_pheromone.get(spot_index, 1.0)
                eta = 1.0 / (static_score + 1e-9)
                weight = (tau ** alpha_pher) * (eta ** beta_heur)
                weighted_spots.append((spot_index, weight))
                total_weight += weight
            if total_weight <= 0.0:
                spot_index = random.choice(feasible_spots)
            else:
                draw = random.random() * total_weight
                acc = 0.0
                spot_index = feasible_spots[-1]
                for candidate_index, weight in weighted_spots:
                    acc += weight
                    if acc >= draw:
                        spot_index = candidate_index
                        break
            parking_node = int(data["SpotLotNode"][spot_index - 1])
            g_map = guidance_maps.get(parking_node)

            res = evaluate_ant_solution(
                data=data,
                spot_index=spot_index,
                src=src,
                dst=dst,
                adj=adj,
                edge_time=edge_time,
                edge_distance=edge_distance,
                cost_map=cost_map,
                pheromone=pheromone,
                alpha_weight=alpha_weight,
                alpha_pher=alpha_pher,
                beta_heur=beta_heur,
                walk_time_map=walk_time_map,
                walk_distance_map=walk_distance_map,
                guidance_map=g_map,
            )
            iteration_results.append(res)

            if global_best is None or res["objective"] < global_best["objective"]:
                global_best = res

        # Evaporation
        for e in list(pheromone.keys()):
            pheromone[e] *= (1.0 - evaporation)
            if pheromone[e] < 1e-6:
                pheromone[e] = 1e-6
        for spot_index in list(spot_pheromone.keys()):
            spot_pheromone[spot_index] *= (1.0 - evaporation)
            if spot_pheromone[spot_index] < 1e-6:
                spot_pheromone[spot_index] = 1e-6

        # Deposit pheromone from feasible solutions
        for res in iteration_results:
            if not res["feasible"] or not math.isfinite(res["objective"]) or res["objective"] <= 0:
                continue
            deposit = Q / res["objective"]
            path = res["drive_path"]
            for a, b in zip(path[:-1], path[1:]):
                pheromone[(a, b)] = pheromone.get((a, b), 1.0) + deposit
            spot_pheromone[int(res["parking_spot_index"])] = spot_pheromone.get(int(res["parking_spot_index"]), 1.0) + deposit

    return global_best


# -----------------------------
# Logging
# -----------------------------
def write_log_table(
    out_prefix: str,
    case_name: str,
    Nodes: int,
    Edges: int,
    TotalSpots: int,
    FeasibleSpots: int,
    Source: int,
    Destination: int,
    Alpha: float,
    W: float,
    Cmax: float,
    weight_w: float,
    parking_lots: int,
    feasible_parking: bool,
    runtime_ms: float,
    selected_result: Optional[Dict[str, Any]],
) -> None:
    if out_prefix and os.path.splitext(out_prefix)[1].lower() == ".csv":
        filename = out_prefix
    else:
        filename = os.path.join(out_prefix, "log_aco.csv") if out_prefix else "log_aco.csv"

    file_exists = os.path.isfile(filename)
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Case", "Nodes", "Edges", "ParkingLots", "TotalSpots", "FeasibleSpots",
            "Source", "Destination", "Alpha", "Wmax", "Cmax", "WeightW", "runtime_ms",
            "ParkingNode", "ParkingSpotIndex", "ParkingSpotId", "SpotClass",
            "drive_time_hr", "internal_drive_time_hr", "internal_walk_time_hr",
            "external_walk_distance_km", "external_walk_time_hr", "walk_time_hr",
            "parking_cost", "objective", "DrivePath", "WalkPath"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        result_ok = (
            selected_result is not None
            and bool(selected_result.get("feasible", True))
            and math.isfinite(float(selected_result.get("drive_time", math.inf)))
            and math.isfinite(float(selected_result.get("walk_time", math.inf)))
            and math.isfinite(float(selected_result.get("parking_cost", math.inf)))
            and math.isfinite(float(selected_result.get("objective", math.inf)))
            and bool(selected_result.get("drive_path"))
            and bool(selected_result.get("walk_path"))
        )

        if not result_ok:
            writer.writerow({
                "Case": case_name,
                "Nodes": Nodes,
                "Edges": Edges,
                "ParkingLots": parking_lots,
                "TotalSpots": TotalSpots,
                "FeasibleSpots": FeasibleSpots,
                "Source": Source,
                "Destination": Destination,
                "Alpha": Alpha,
                "Wmax": W,
                "Cmax": Cmax,
                "WeightW": weight_w,
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
            })
        else:
            writer.writerow({
                "Case": case_name,
                "Nodes": Nodes,
                "Edges": Edges,
                "ParkingLots": parking_lots,
                "TotalSpots": TotalSpots,
                "FeasibleSpots": FeasibleSpots,
                "Source": Source,
                "Destination": Destination,
                "Alpha": Alpha,
                "Wmax": W,
                "Cmax": Cmax,
                "WeightW": weight_w,
                "runtime_ms": round(runtime_ms, 3),
                "ParkingNode": selected_result["parking_node"],
                "ParkingSpotIndex": selected_result.get("parking_spot_index", ""),
                "ParkingSpotId": selected_result.get("parking_spot_id", ""),
                "SpotClass": selected_result.get("spot_class", ""),
                "drive_time_hr": selected_result["drive_time"],
                "internal_drive_time_hr": selected_result.get("internal_drive_time_hr", ""),
                "internal_walk_time_hr": selected_result.get("internal_walk_time_hr", ""),
                "external_walk_distance_km": selected_result.get("external_walk_distance_km", ""),
                "external_walk_time_hr": selected_result.get("external_walk_time_hr", ""),
                "walk_time_hr": selected_result.get("walk_time_hr", selected_result.get("walk_time", "")),
                "parking_cost": selected_result["parking_cost"],
                "objective": selected_result["objective"],
                "DrivePath": "->".join(map(str, selected_result["drive_path"])),
                "WalkPath": "->".join(map(str, selected_result["walk_path"])),
            })


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Joint ACO for routing optimization and parking selection")
    parser.add_argument("--dat", help="Path to one OPL case file (.dat or .txt)")
    parser.add_argument("--dat-dir", default="formatted_cases", help="Directory containing OPL case files")
    parser.add_argument("--out", default="aco_results/log_aco.csv", help="Output CSV path")
    parser.add_argument("--w", type=float, default=None, help="Weight w; defaults to value from .dat")
    parser.add_argument("--cmax", type=float, default=None, help="Maximum parking cost threshold; defaults to .dat if present")
    parser.add_argument("--ants", type=int, default=30, help="Number of ants")
    parser.add_argument("--iterations", type=int, default=80, help="Number of ACO iterations")
    parser.add_argument("--alpha-pher", type=float, default=1.0, help="Pheromone exponent")
    parser.add_argument("--beta-heur", type=float, default=2.0, help="Heuristic exponent")
    parser.add_argument("--evaporation", type=float, default=0.3, help="Evaporation rate")
    parser.add_argument("--Q", type=float, default=1.0, help="Pheromone deposit constant")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.dat:
        dat_paths = [Path(args.dat)]
    else:
        root = Path(args.dat_dir)
        dat_paths = sorted(root.rglob("*.dat"))

    if not dat_paths:
        raise RuntimeError(f"No .dat files found in {args.dat_dir}.")

    for dat_path in dat_paths:
        dat_text = dat_path.read_text(encoding="utf-8", errors="replace")
        data = parse_spot_opl_dat(dat_text)

        Nodes = int(data["Nodes"])
        S = int(data["S"])
        D = int(data["D"])
        lots = [int(x) for x in data["L"]]
        costs = [float(x) for x in data["C"]]
        dat_w = float(data["w"])
        Alpha = float(data["Alpha"])
        Wmax = float(data["W"])
        Wmax_km = float(data.get("W_km", Wmax))
        Cmax = args.cmax if args.cmax is not None else float(data["Cmax"])

        edges: List[Edge] = [(int(u), int(v)) for (u, v) in data["E"]]
        edge_time = make_edge_time_map(edges, data["EdgeTime"])
        edge_distance = make_edge_distance_map(edges, data["Distance"])
        adj = build_adj(edges, edge_time)

        cost_map: Dict[Node, float] = {p: c for p, c in zip(lots, costs)}
        use_w = args.w if args.w is not None else dat_w

        walk_time_map = {p: t for p, t in zip(lots, data.get("WalkTimeToDest", []))}
        walk_distance_map = {p: d for p, d in zip(lots, data.get("WalkDistanceToDest", []))}

        feasible_spots = [int(spot["parking_spot_index"]) for spot in get_feasible_spots(data)]

        t0 = time.time()
        best = solve_aco_joint_route_and_parking(
            data=data,
            src=S,
            dst=D,
            feasible_spots=feasible_spots,
            adj=adj,
            edge_time=edge_time,
            edge_distance=edge_distance,
            cost_map=cost_map,
            alpha_weight=use_w,
            walk_time_map=walk_time_map,
            walk_distance_map=walk_distance_map,
            ants=args.ants,
            iterations=args.iterations,
            alpha_pher=args.alpha_pher,
            beta_heur=args.beta_heur,
            evaporation=args.evaporation,
            Q=args.Q,
            seed=args.seed,
        )
        runtime_ms = (time.time() - t0) * 1000.0

        # OPL-like output
        print("Status: solved" if best is not None and best["feasible"] else "Status: not solved")
        print(f"Objective value: {best['objective'] if best is not None else math.inf}")

        if best is not None and best["feasible"]:
            print(f"Travel time T[D]: {best['travel_time']}")
            print(f"Parking cost: {best['parking_cost']}")
        else:
            print("Travel time T[D]: inf")
            print("Parking cost: inf")

        print("\nSelected route edges:")
        if best is not None and best["feasible"]:
            full_path = list(best["drive_path"]) + [D]
            for a, b in zip(full_path[:-1], full_path[1:]):
                print(f"  {a} -> {b}   time={edge_time.get((a, b), math.inf)}")

        print("\nSelected parking lot:")
        if best is not None and best["feasible"]:
            p_idx = lots.index(best["parking_node"]) + 1
            print(
                f"  p={p_idx}  node={best['parking_node']}  cost={best['parking_cost']}  "
                f"finalWalkTime={best['walk_time']}"
            )
            print(
                f"Selected parking spot: index={best['parking_spot_index']}  "
                f"id={best.get('parking_spot_id', '')}  class={best.get('spot_class', '')}"
            )

        write_log_table(
            out_prefix=args.out,
            case_name=dat_path.stem,
            Nodes=Nodes,
            Edges=len(edges),
            TotalSpots=int(data["TotalSpots"]),
            FeasibleSpots=len(feasible_spots),
            Source=S,
            Destination=D,
            Alpha=Alpha,
            W=Wmax_km,
            Cmax=Cmax,
            weight_w=use_w,
            parking_lots=len(lots),
            feasible_parking=(best is not None and best["feasible"]),
            runtime_ms=runtime_ms,
            selected_result=best,
        )


if __name__ == "__main__":
    main()
