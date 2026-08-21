import re
import argparse
import math
import time
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import csv
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spot_level_utils import evaluate_spot, get_feasible_spots, parse_opl_dat as parse_spot_opl_dat

Node = int
Edge = Tuple[int, int]


@dataclass
class Particle:
    parking_pos: float
    parking_vel: float
    priorities: Dict[Node, float]
    priority_vels: Dict[Node, float]


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
# Parking candidate filtering
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
# PSO route decoder
# -----------------------------
def decode_spot_index(parking_pos: float, feasible_spots: List[int]) -> int:
    idx = int(round(parking_pos))
    idx = max(0, min(idx, len(feasible_spots) - 1))
    return feasible_spots[idx]


def decode_route_from_particle(
    src: Node,
    parking_node: Node,
    adj: Dict[Node, List[Tuple[Node, float]]],
    priorities: Dict[Node, float],
    guidance_map: Optional[Dict[Node, float]] = None,
    max_steps: int = 5000
) -> Tuple[List[Node], float]:
    """
    Build a driving route from src to parking_node using a Backtracking Search (DFS).
    Guided by particle priorities and distance-to-goal heuristic.
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
            
            h_val = guidance_map.get(v, 999999) if guidance_map else 0
            score = h_val + 0.1 * priorities.get(v, 0.5)
            scored_candidates.append((v, t, score))
            
        # DFS: Best candidates last -> first pop
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        for v, t, _ in scored_candidates:
            stack.append((v, path + [v], p_time + t))

    return [], math.inf


# -----------------------------
# Fitness evaluation
# -----------------------------
def evaluate_particle(
    particle: Particle,
    data: Dict[str, Any],
    feasible_spots: List[int],
    src: Node,
    dst: Node,
    adj: Dict[Node, List[Tuple[Node, float]]],
    edge_time: Dict[Edge, float],
    edge_distance: Dict[Edge, float],
    cost_map: Dict[Node, float],
    w: float,
    walk_time_map: Dict[Node, float],
    walk_distance_map: Dict[Node, float],
    guidance_maps: Optional[Dict[Node, Dict[Node, float]]] = None,
) -> Dict[str, Any]:
    spot_index = decode_spot_index(particle.parking_pos, feasible_spots)
    parking_node = int(data["SpotLotNode"][spot_index - 1])
    g_map = guidance_maps.get(parking_node) if guidance_maps else None

    drive_path, drive_time = decode_route_from_particle(
        src=src,
        parking_node=parking_node,
        adj=adj,
        priorities=particle.priorities,
        guidance_map=g_map,
    )
    # ... rest of evaluated_particle remains same ...

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

    spot = evaluate_spot(data, drive_time, spot_index, alpha=w)
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
# Particle initialization
# -----------------------------
def random_particle(feasible_spots: List[int], nodes: List[Node]) -> Particle:
    return Particle(
        parking_pos=random.uniform(0, max(0, len(feasible_spots) - 1)),
        parking_vel=random.uniform(-1.0, 1.0),
        priorities={n: random.random() for n in nodes},
        priority_vels={n: random.uniform(-0.1, 0.1) for n in nodes},
    )


def recommended_search_budget(node_count: int) -> Tuple[int, int]:
    """
    Scale the PSO search budget for larger graphs where the default budget
    is prone to premature convergence.
    """
    if node_count >= 2500:
        return 100, 260
    if node_count >= 2000:
        return 80, 220
    if node_count >= 1500:
        return 60, 160
    return 30, 80


# -----------------------------
# PSO solver
# -----------------------------
def solve_pso_joint_route_and_parking(
    data: Dict[str, Any],
    nodes: List[Node],
    src: Node,
    dst: Node,
    feasible_spots: List[int],
    adj: Dict[Node, List[Tuple[Node, float]]],
    edge_time: Dict[Edge, float],
    edge_distance: Dict[Edge, float],
    cost_map: Dict[Node, float],
    w: float,
    walk_time_map: Dict[Node, float],
    walk_distance_map: Dict[Node, float],
    swarm_size: int = 30,
    iterations: int = 80,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    if not feasible_spots:
        return None

    random.seed(seed)

    # Precalculate guidance maps for each parent lot referenced by a feasible spot.
    guidance_maps: Dict[Node, Dict[Node, float]] = {}
    for p in sorted({int(data["SpotLotNode"][spot_index - 1]) for spot_index in feasible_spots}):
        guidance_maps[p] = dijkstra_to_goal(nodes, adj, p)

    swarm = [random_particle(feasible_spots, nodes) for _ in range(swarm_size)]

    personal_best_particles: List[Particle] = []
    personal_best_results: List[Dict[str, Any]] = []

    global_best_particle: Optional[Particle] = None
    global_best_result: Optional[Dict[str, Any]] = None

    # initialize personal/global bests
    for p in swarm:
        res = evaluate_particle(p, data, feasible_spots, src, dst, adj, edge_time, edge_distance, cost_map, w, walk_time_map, walk_distance_map, guidance_maps)
        personal_best_particles.append(
            Particle(
                parking_pos=p.parking_pos,
                parking_vel=p.parking_vel,
                priorities=dict(p.priorities),
                priority_vels=dict(p.priority_vels),
            )
        )
        personal_best_results.append(res)

        if global_best_result is None or res["objective"] < global_best_result["objective"]:
            global_best_result = res
            global_best_particle = Particle(
                parking_pos=p.parking_pos,
                parking_vel=p.parking_vel,
                priorities=dict(p.priorities),
                priority_vels=dict(p.priority_vels),
            )

    for _ in range(iterations):
        for i, particle in enumerate(swarm):
            r1 = random.random()
            r2 = random.random()

            # parking dimension update
            pb_parking = personal_best_particles[i].parking_pos
            gb_parking = global_best_particle.parking_pos if global_best_particle is not None else particle.parking_pos

            particle.parking_vel = (
                inertia * particle.parking_vel
                + cognitive * r1 * (pb_parking - particle.parking_pos)
                + social * r2 * (gb_parking - particle.parking_pos)
            )
            particle.parking_pos += particle.parking_vel

            # clamp parking position
            particle.parking_pos = max(0.0, min(float(len(feasible_spots) - 1), particle.parking_pos))

            # priority dimensions update
            for n in nodes:
                r1n = random.random()
                r2n = random.random()

                pb_val = personal_best_particles[i].priorities[n]
                gb_val = global_best_particle.priorities[n] if global_best_particle is not None else particle.priorities[n]

                particle.priority_vels[n] = (
                    inertia * particle.priority_vels[n]
                    + cognitive * r1n * (pb_val - particle.priorities[n])
                    + social * r2n * (gb_val - particle.priorities[n])
                )
                particle.priorities[n] += particle.priority_vels[n]

                # clamp priorities to [0,1]
                particle.priorities[n] = max(0.0, min(1.0, particle.priorities[n]))

            res = evaluate_particle(particle, data, feasible_spots, src, dst, adj, edge_time, edge_distance, cost_map, w, walk_time_map, walk_distance_map, guidance_maps)

            if res["objective"] < personal_best_results[i]["objective"]:
                personal_best_results[i] = res
                personal_best_particles[i] = Particle(
                    parking_pos=particle.parking_pos,
                    parking_vel=particle.parking_vel,
                    priorities=dict(particle.priorities),
                    priority_vels=dict(particle.priority_vels),
                )

            if global_best_result is None or res["objective"] < global_best_result["objective"]:
                global_best_result = res
                global_best_particle = Particle(
                    parking_pos=particle.parking_pos,
                    parking_vel=particle.parking_vel,
                    priorities=dict(particle.priorities),
                    priority_vels=dict(particle.priority_vels),
                )

    return global_best_result


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
        filename = os.path.join(out_prefix, "log_pso.csv") if out_prefix else "log_pso.csv"

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
    parser = argparse.ArgumentParser(description="Joint PSO for routing optimization and parking selection")
    parser.add_argument("--dat", help="Path to one OPL case file (.dat or .txt)")
    parser.add_argument("--dat-dir", default="formatted_cases", help="Directory containing OPL case files")
    parser.add_argument("--out", default="pso_results/log_pso.csv", help="Output CSV path")
    parser.add_argument("--w", type=float, default=None, help="Weight w; defaults to value from .dat")
    parser.add_argument("--cmax", type=float, default=None, help="Maximum parking cost threshold; defaults to .dat if present")
    parser.add_argument(
        "--swarm-size",
        type=int,
        default=None,
        help="Swarm size; defaults to a node-size-aware budget when omitted",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of PSO iterations; defaults to a node-size-aware budget when omitted",
    )
    parser.add_argument("--inertia", type=float, default=0.7, help="Inertia weight")
    parser.add_argument("--cognitive", type=float, default=1.5, help="Cognitive coefficient")
    parser.add_argument("--social", type=float, default=1.5, help="Social coefficient")
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
        nodes: List[Node] = list(range(1, Nodes + 1))
        edge_time = make_edge_time_map(edges, data["EdgeTime"])
        edge_distance = make_edge_distance_map(edges, data["Distance"])
        adj = build_adj(edges, edge_time)

        cost_map: Dict[Node, float] = {p: c for p, c in zip(lots, costs)}
        use_w = args.w if args.w is not None else dat_w
        default_swarm_size, default_iterations = recommended_search_budget(Nodes)
        swarm_size = args.swarm_size if args.swarm_size is not None else default_swarm_size
        iterations = args.iterations if args.iterations is not None else default_iterations

        walk_time_map = {p: t for p, t in zip(lots, data.get("WalkTimeToDest", []))}
        walk_distance_map = {p: d for p, d in zip(lots, data.get("WalkDistanceToDest", []))}

        feasible_spots = [int(spot["parking_spot_index"]) for spot in get_feasible_spots(data)]

        t0 = time.time()
        best = solve_pso_joint_route_and_parking(
            data=data,
            nodes=nodes,
            src=S,
            dst=D,
            feasible_spots=feasible_spots,
            adj=adj,
            edge_time=edge_time,
            edge_distance=edge_distance,
            cost_map=cost_map,
            w=use_w,
            walk_time_map=walk_time_map,
            walk_distance_map=walk_distance_map,
            swarm_size=swarm_size,
            iterations=iterations,
            inertia=args.inertia,
            cognitive=args.cognitive,
            social=args.social,
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
