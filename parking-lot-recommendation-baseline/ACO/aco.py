import re
import argparse
import math
import time
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lot_objective_utils import raw_lot_objective

Node = int
Edge = Tuple[int, int]

MASTER_RESULTS_HEADER = [
    "method",
    "dat_file",
    "nodes",
    "edges",
    "parking_lots",
    "S",
    "D",
    "Alpha",
    "Wmax",
    "Cmax",
    "w",
    "feasible",
    "parking_node",
    "drive_time_hr",
    "walk_distance_km",
    "walk_time_hr",
    "parking_cost",
    "objective",
    "non_normalized_drive_time_min",
    "non_normalized_walk_time_min",
    "non_normalized_travel_time_min",
    "non_normalized_parking_cost",
    "non_normalized_objective",
    "runtime_ms",
    "drive_path",
    "walk_path",
]


def _normalize_case_name(text: str) -> str:
    value = str(text or "").strip()
    if value and not value.lower().endswith(".dat"):
        return f"{value}.dat"
    return value


def _master_results_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "master_results_aco.csv"


def _load_master_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csvfile:
        return [
            {field: row.get(field, "") for field in MASTER_RESULTS_HEADER}
            for row in csv.DictReader(csvfile)
            if any(str(value).strip() for value in row.values())
        ]


def _write_master_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=MASTER_RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _sort_case_key(dat_file: str) -> Tuple[int, int, str]:
    match = re.search(r"N(\d+)(?:_C(\d+))?", dat_file or "", re.IGNORECASE)
    if match:
        case_num = int(match.group(2)) if match.group(2) else 0
        return int(match.group(1)), case_num, dat_file
    return 10**9, 10**9, dat_file or ""


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
    parking_node: Node,
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
    objective_context: Dict[str, float],
    guidance_map: Optional[Dict[Node, float]] = None,
) -> Dict[str, Any]:
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
            "drive_path": [],
            "walk_path": [],
            "drive_time": math.inf,
            "walk_time": math.inf,
            "travel_time": math.inf,
            "parking_cost": math.inf,
        }

    # Use explicit WalkTimeToDest map if available, otherwise fallback to direct edge
    walk_time = walk_time_map.get(parking_node, edge_time.get((parking_node, dst), math.inf))
    walk_distance = walk_distance_map.get(parking_node, edge_distance.get((parking_node, dst), math.inf))

    if not math.isfinite(walk_time):
        return {
            "feasible": False,
            "objective": math.inf,
            "parking_node": parking_node,
            "drive_path": [],
            "walk_path": [],
            "drive_time": math.inf,
            "walk_time": math.inf,
            "travel_time": math.inf,
            "parking_cost": math.inf,
        }

    parking_cost = cost_map.get(parking_node, math.inf)
    travel_time = drive_time + walk_time
    objective = raw_lot_objective(
        drive_time * 60.0,
        walk_time * 60.0,
        parking_cost,
        weight_w=alpha_weight,
    )

    return {
        "feasible": True,
        "objective": objective,
        "parking_node": parking_node,
        "drive_path": drive_path,
        "walk_path": [parking_node, dst],
        "drive_time": drive_time,
        "walk_time": walk_time,
        "walk_distance": walk_distance,
        "walk_time_hr": walk_time,
        "travel_time": travel_time,
        "parking_cost": parking_cost,
    }


# -----------------------------
# ACO solver
# -----------------------------
def solve_aco_joint_route_and_parking(
    src: Node,
    dst: Node,
    feasible_parking: List[Node],
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
    if not feasible_parking:
        return None

    random.seed(seed)
    # Precalculate guidance maps for each feasible parking lot
    guidance_maps: Dict[Node, Dict[Node, float]] = {}
    nodes = list(adj.keys())
    for p in feasible_parking:
        guidance_maps[p] = dijkstra_to_goal(nodes, adj, p)

    pheromone: Dict[Edge, float] = {}
    for u in adj:
        for v, _ in adj[u]:
            pheromone[(u, v)] = 1.0

    global_best: Optional[Dict[str, Any]] = None

    for _ in range(iterations):
        iteration_results: List[Dict[str, Any]] = []

        for _ant in range(ants):
            parking_node = random.choice(feasible_parking)
            g_map = guidance_maps.get(parking_node)

            res = evaluate_ant_solution(
                parking_node=parking_node,
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
                objective_context=None,
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

        # Deposit pheromone from feasible solutions
        for res in iteration_results:
            if not res["feasible"] or not math.isfinite(res["objective"]) or res["objective"] <= 0:
                continue
            deposit = Q / res["objective"]
            path = res["drive_path"]
            for a, b in zip(path[:-1], path[1:]):
                pheromone[(a, b)] = pheromone.get((a, b), 1.0) + deposit

    return global_best


# -----------------------------
# Logging
# -----------------------------
def write_log_table(
    out_prefix: str,
    case_name: str,
    Nodes: int,
    Edges: int,
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
    del out_prefix

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

    normalized_case_name = _normalize_case_name(case_name)
    existing_rows = _load_master_rows(_master_results_path())
    row_by_case = {
        _normalize_case_name(row.get("dat_file", "")): row
        for row in existing_rows
        if row.get("dat_file", "")
    }

    row_by_case[normalized_case_name] = {
        "method": "ACO",
        "dat_file": normalized_case_name,
        "nodes": str(Nodes),
        "edges": str(Edges),
        "parking_lots": str(parking_lots),
        "S": str(Source),
        "D": str(Destination),
        "Alpha": str(Alpha),
        "Wmax": str(W),
        "Cmax": str(Cmax),
        "w": str(weight_w),
        "feasible": "1" if result_ok and feasible_parking else "0",
        "parking_node": "" if not result_ok else str(selected_result["parking_node"]),
        "drive_time_hr": "" if not result_ok else str(selected_result["drive_time"]),
        "walk_distance_km": "" if not result_ok else str(selected_result.get("walk_distance", "")),
        "walk_time_hr": "" if not result_ok else str(selected_result.get("walk_time_hr", selected_result.get("walk_time", ""))),
        "parking_cost": "" if not result_ok else str(selected_result["parking_cost"]),
        "objective": "" if not result_ok else str(selected_result["objective"]),
        "non_normalized_drive_time_min": "" if not result_ok else str(float(selected_result["drive_time"]) * 60.0),
        "non_normalized_walk_time_min": "" if not result_ok else str(float(selected_result.get("walk_time_hr", selected_result.get("walk_time", ""))) * 60.0),
        "non_normalized_travel_time_min": "" if not result_ok else str(
            (float(selected_result["drive_time"]) + float(selected_result.get("walk_time_hr", selected_result.get("walk_time", "")))) * 60.0
        ),
        "non_normalized_parking_cost": "" if not result_ok else str(selected_result["parking_cost"]),
        "non_normalized_objective": "" if not result_ok else str(
            raw_lot_objective(
                float(selected_result["drive_time"]) * 60.0,
                float(selected_result.get("walk_time_hr", selected_result.get("walk_time", ""))) * 60.0,
                selected_result["parking_cost"],
                weight_w=weight_w,
            )
        ),
        "runtime_ms": str(round(runtime_ms, 3)),
        "drive_path": "" if not result_ok else "->".join(map(str, selected_result["drive_path"])),
        "walk_path": "" if not result_ok else "->".join(map(str, selected_result["walk_path"])),
    }
    rows = sorted(row_by_case.values(), key=lambda row: _sort_case_key(row.get("dat_file", "")))
    _write_master_rows(_master_results_path(), rows)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Joint ACO for routing optimization and parking selection")
    parser.add_argument("--dat", help="Path to one OPL case file (.dat or .txt)")
    parser.add_argument("--dat-dir", default="formatted_cases", help="Directory containing OPL case files")
    parser.add_argument("--out", default="", help="Deprecated; results always write to data/master_results_aco.csv")
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
        dat_paths = sorted(set(root.rglob("*.dat")) | set(root.rglob("*.txt")))

    if not dat_paths:
        raise RuntimeError(f"No .dat or .txt files found in {args.dat_dir}.")

    for dat_path in dat_paths:
        dat_text = dat_path.read_text(encoding="utf-8", errors="replace")
        data = parse_opl_dat(dat_text)

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
        availability: Dict[Node, int] = {p: 1 for p in lots}
        use_w = args.w if args.w is not None else dat_w

        walk_time_map = {p: t for p, t in zip(lots, data.get("WalkTimeToDest", []))}
        walk_distance_map = {p: d for p, d in zip(lots, data.get("WalkDistanceToDest", []))}

        feasible_parking = candidate_parking_lots(
            lots=lots,
            dst=D,
            edge_time=edge_time,
            cost_map=cost_map,
            availability=availability,
            Wmax=Wmax,
            Cmax=Cmax,
            walk_time_map=walk_time_map,
        )

        t0 = time.time()
        best = solve_aco_joint_route_and_parking(
            src=S,
            dst=D,
            feasible_parking=feasible_parking,
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

        write_log_table(
            out_prefix=args.out,
            case_name=dat_path.stem,
            Nodes=Nodes,
            Edges=len(edges),
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
