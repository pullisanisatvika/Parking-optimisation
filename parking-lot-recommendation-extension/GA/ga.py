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
class Chromosome:
    spot_index: int
    priorities: Dict[Node, float]


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
    Since the graph might be directed, we actually need to find paths TO the goal.
    In this simple model, we'll assume we can use the adj list logic.
    Actually, to find shortest path TO goal, we need the reverse adjacency.
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
# Basic filtering for candidate parking lots
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
# GA route decoder
# -----------------------------
def decode_route_from_chromosome(
    src: Node,
    parking_node: Node,
    adj: Dict[Node, List[Tuple[Node, float]]],
    priorities: Dict[Node, float],
    guidance_map: Optional[Dict[Node, float]] = None,
    max_steps: int = 5000
) -> Tuple[List[Node], float]:
    """
    Build a driving route from src to parking_node using a Backtracking Search (DFS).
    Guided by chromosome priorities and distance-to-goal heuristic.
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
            
            # Heuristic value (Distance-to-goal)
            h_val = guidance_map.get(v, 999999) if guidance_map else 0
            # Higher priority = less desirable (so we use 0.1 * priority)
            score = h_val + 0.1 * priorities.get(v, 0.5)
            scored_candidates.append((v, t, score))
            
        # DFS: Pull best candidates last to pop them first
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        for v, t, _ in scored_candidates:
            stack.append((v, path + [v], p_time + t))

    return [], math.inf


# -----------------------------
# Fitness evaluation
# -----------------------------
def evaluate_chromosome(
    chrom: Chromosome,
    data: Dict[str, Any],
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
    parking_node = int(data["SpotLotNode"][chrom.spot_index - 1])
    g_map = guidance_maps.get(parking_node) if guidance_maps else None

    drive_path, drive_time = decode_route_from_chromosome(
        src=src,
        parking_node=parking_node,
        adj=adj,
        priorities=chrom.priorities,
        guidance_map=g_map,
    )

    if not drive_path or not math.isfinite(drive_time):
        return {
            "feasible": False,
            "objective": math.inf,
            "parking_node": parking_node,
            "parking_spot_index": chrom.spot_index,
            "drive_path": [],
            "walk_path": [],
            "drive_time": math.inf,
            "walk_time": math.inf,
            "travel_time": math.inf,
            "parking_cost": math.inf,
        }

    spot = evaluate_spot(data, drive_time, chrom.spot_index, alpha=w)
    if spot is None:
        return {
            "feasible": False,
            "objective": math.inf,
            "parking_node": parking_node,
            "parking_spot_index": chrom.spot_index,
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
# GA operators
# -----------------------------
def random_chromosome(feasible_spots: List[int], nodes: List[Node]) -> Chromosome:
    return Chromosome(
        spot_index=random.choice(feasible_spots),
        priorities={n: random.random() for n in nodes}
    )


def tournament_select(population: List[Chromosome], fitness_list: List[Dict[str, Any]], k: int = 3) -> Chromosome:
    idxs = random.sample(range(len(population)), k=min(k, len(population)))
    best_idx = min(idxs, key=lambda i: fitness_list[i]["objective"])
    return population[best_idx]


def crossover(parent1: Chromosome, parent2: Chromosome, nodes: List[Node]) -> Chromosome:
    child_spot = parent1.spot_index if random.random() < 0.5 else parent2.spot_index
    child_priorities: Dict[Node, float] = {}
    for n in nodes:
        child_priorities[n] = parent1.priorities[n] if random.random() < 0.5 else parent2.priorities[n]
    return Chromosome(spot_index=child_spot, priorities=child_priorities)


def mutate(
    chrom: Chromosome,
    feasible_spots: List[int],
    nodes: List[Node],
    parking_mutation_rate: float,
    priority_mutation_rate: float
) -> Chromosome:
    new_spot = chrom.spot_index
    if random.random() < parking_mutation_rate:
        new_spot = random.choice(feasible_spots)

    new_priorities = dict(chrom.priorities)
    for n in nodes:
        if random.random() < priority_mutation_rate:
            new_priorities[n] = random.random()

    return Chromosome(spot_index=new_spot, priorities=new_priorities)


# -----------------------------
# GA solver
# -----------------------------
def solve_ga_joint_route_and_parking(
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
    population_size: int = 30,
    generations: int = 80,
    parking_mutation_rate: float = 0.10,
    priority_mutation_rate: float = 0.05,
    elitism: bool = True,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    if not feasible_spots:
        return None

    random.seed(seed)

    # Precalculate guidance maps for each parent lot referenced by a feasible spot.
    guidance_maps: Dict[Node, Dict[Node, float]] = {}
    for p in sorted({int(data["SpotLotNode"][spot_index - 1]) for spot_index in feasible_spots}):
        guidance_maps[p] = dijkstra_to_goal(nodes, adj, p)

    population = [random_chromosome(feasible_spots, nodes) for _ in range(population_size)]
    best_result: Optional[Dict[str, Any]] = None
    best_chrom: Optional[Chromosome] = None

    for _ in range(generations):
        fitness_list = [
            evaluate_chromosome(ch, data, src, dst, adj, edge_time, edge_distance, cost_map, w, walk_time_map, walk_distance_map, guidance_maps)
            for ch in population
        ]

        gen_best_idx = min(range(len(population)), key=lambda i: fitness_list[i]["objective"])
        gen_best = fitness_list[gen_best_idx]

        if best_result is None or gen_best["objective"] < best_result["objective"]:
            best_result = gen_best
            best_chrom = population[gen_best_idx]

        new_population: List[Chromosome] = []

        if elitism and best_chrom is not None:
            new_population.append(best_chrom)

        while len(new_population) < population_size:
            p1 = tournament_select(population, fitness_list)
            p2 = tournament_select(population, fitness_list)
            child = crossover(p1, p2, nodes)
            child = mutate(child, feasible_spots, nodes, parking_mutation_rate, priority_mutation_rate)
            new_population.append(child)

        population = new_population

    return best_result


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
        filename = os.path.join(out_prefix, "log_ga.csv") if out_prefix else "log_ga.csv"

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
    parser = argparse.ArgumentParser(description="Joint GA for routing optimization and parking selection")
    parser.add_argument("--dat", help="Path to one OPL case file (.dat or .txt)")
    parser.add_argument("--dat-dir", default="formatted_cases", help="Directory containing OPL case files")
    parser.add_argument("--out", default="ga_results/log_ga.csv", help="Output CSV path")
    parser.add_argument("--w", type=float, default=None, help="Weight w; defaults to value from .dat")
    parser.add_argument("--cmax", type=float, default=None, help="Maximum parking cost threshold; defaults to .dat if present")
    parser.add_argument("--pop-size", type=int, default=30, help="Population size")
    parser.add_argument("--generations", type=int, default=80, help="Number of generations")
    parser.add_argument("--parking-mutation-rate", type=float, default=0.10, help="Parking gene mutation probability")
    parser.add_argument("--priority-mutation-rate", type=float, default=0.05, help="Node-priority mutation probability")
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

        walk_time_map = {p: t for p, t in zip(lots, data.get("WalkTimeToDest", []))}
        walk_distance_map = {p: d for p, d in zip(lots, data.get("WalkDistanceToDest", []))}
        feasible_spots = [int(spot["parking_spot_index"]) for spot in get_feasible_spots(data)]

        t0 = time.time()
        best = solve_ga_joint_route_and_parking(
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
            population_size=args.pop_size,
            generations=args.generations,
            parking_mutation_rate=args.parking_mutation_rate,
            priority_mutation_rate=args.priority_mutation_rate,
            elitism=True,
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
