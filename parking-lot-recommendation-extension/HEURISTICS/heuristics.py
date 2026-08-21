import re
import argparse
import math
import time
import heapq
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import csv
import os

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


def make_edge_weight_map(E: List[Edge], weights: List[float]) -> Dict[Edge, float]:
    return {(u, v): float(w) for (u, v), w in zip(E, weights)}


# -----------------------------
# Dijkstra shortest path
# -----------------------------
def dijkstra_shortest_path_parents(
    nodes: List[Node],
    edges: List[Edge],
    edge_weight: Dict[Edge, float],
    start: Node
) -> Tuple[Dict[Node, float], Dict[Node, Optional[Node]]]:
    adj = {n: [] for n in nodes}
    for u, v in edges:
        w = edge_weight.get((u, v), math.inf)
        if math.isfinite(w):
            adj[u].append((v, w))

    dist = {n: math.inf for n in nodes}
    parent: Dict[Node, Optional[Node]] = {n: None for n in nodes}
    dist[start] = 0.0
    heap: List[Tuple[float, Node]] = [(0.0, start)]

    while heap:
        cur_dist, u = heapq.heappop(heap)
        if cur_dist > dist[u]:
            continue
        for v, w in adj[u]:
            nd = cur_dist + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, parent


def reconstruct_path(parent: Dict[Node, Optional[Node]], start: Node, end: Node) -> List[Node]:
    if start == end:
        return [start]
    if parent[end] is None:
        return []
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]
    if path[-1] != start:
        return []
    path.reverse()
    return path


# -----------------------------
# Feasibility filtering
# -----------------------------
def feasibility_filter(
    lots: List[Node],
    availability: Dict[Node, int],
    walk_time: Dict[Node, float],
    cost: Dict[Node, float],
    Wmax: float,
    Cmax: float
) -> List[Node]:
    feasible = []
    for p in lots:
        if availability.get(p, 0) <= 0:
            continue
        if walk_time.get(p, math.inf) > Wmax:
            continue
        if cost.get(p, math.inf) > Cmax:
            continue
        feasible.append(p)
    return feasible


# -----------------------------
# Dominance pruning
# -----------------------------
def dominance_prune(
    candidates: List[Node],
    drive_time: Dict[Node, float],
    walk_time: Dict[Node, float],
    cost: Dict[Node, float]
) -> List[Node]:
    kept = []
    for p in candidates:
        dominated = False
        for q in candidates:
            if q == p:
                continue
            if (
                drive_time.get(q, math.inf) <= drive_time.get(p, math.inf)
                and walk_time.get(q, math.inf) <= walk_time.get(p, math.inf)
                and cost.get(q, math.inf) <= cost.get(p, math.inf)
                and (
                    drive_time.get(q, math.inf) < drive_time.get(p, math.inf)
                    or walk_time.get(q, math.inf) < walk_time.get(p, math.inf)
                    or cost.get(q, math.inf) < cost.get(p, math.inf)
                )
            ):
                dominated = True
                break
        if not dominated:
            kept.append(p)
    return kept


# -----------------------------
# Heuristic selection
# -----------------------------
def heuristic_select_parking(
    nodes: List[Node],
    edges: List[Edge],
    A: Node,
    D: Node,
    lots: List[Node],
    edge_time: Dict[Edge, float],
    distance_map: Dict[Edge, float],
    cost: Dict[Node, float],
    availability: Dict[Node, int],
    Wmax: float,
    Cmax: float,
    w: float,
    walk_time_map: Dict[Node, float],
) -> Dict[str, object]:
    # Stage 1: shortest driving route to each parking lot
    drive_dist, drive_parent = dijkstra_shortest_path_parents(nodes, edges, edge_time, A)

    drive_time_map: Dict[Node, float] = {}
    walk_time_map: Dict[Node, float] = {}
    drive_path_map: Dict[Node, List[Node]] = {}
    walk_path_map: Dict[Node, List[Node]] = {}

    for p in lots:
        drive_time_map[p] = drive_dist.get(p, math.inf)
        drive_path_map[p] = reconstruct_path(drive_parent, A, p)

        # Use WalkTimeToDest map for walking time
        tw = walk_time_map.get(p, edge_time.get((p, D), math.inf))
        if math.isfinite(tw):
            walk_time_map[p] = tw
            walk_path_map[p] = [p, D]
        else:
            walk_time_map[p] = math.inf
            walk_path_map[p] = []

    # Stage 2: feasibility filtering
    feasible = feasibility_filter(lots, availability, walk_time_map, cost, Wmax, Cmax)

    # Also require a valid drive path to the lot
    feasible = [p for p in feasible if drive_path_map[p] and math.isfinite(drive_time_map[p])]

    if not feasible:
        return {
            "selected": None,
            "reason": "No feasible parking lots after availability, walking, and cost filtering.",
            "candidates": [],
        }

    # Stage 3: dominance pruning
    pruned = dominance_prune(feasible, drive_time_map, walk_time_map, cost)

    if not pruned:
        return {
            "selected": None,
            "reason": "All candidates removed during dominance pruning.",
            "candidates": [],
        }

    # Stage 4: raw-score selection (no normalization)
    score_map: Dict[Node, float] = {}
    travel_time_map: Dict[Node, float] = {}

    for p in pruned:
        total_travel = drive_time_map[p] + walk_time_map[p]
        travel_time_map[p] = total_travel
        score_map[p] = w * total_travel + (1.0 - w) * cost[p]

    best_parking = min(pruned, key=lambda p: (score_map[p], travel_time_map[p], cost[p], p))

    return {
        "selected": best_parking,
        "score": score_map[best_parking],
        "drive_time": drive_time_map[best_parking],
        "walk_time": walk_time_map[best_parking],
        "cost": cost[best_parking],
        "drive_path": drive_path_map[best_parking],
        "walk_path": walk_path_map[best_parking],
        "all_scores": score_map,
        "candidates": pruned,
        "drive_time_map": drive_time_map,
        "walk_time_map": walk_time_map,
        "cost_map": cost,
        "score_map": score_map,
        "feasible_set": set(pruned),
        "travel_time_map": travel_time_map,
    }


# -----------------------------
# Logging function
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
    parking_rows: List[Tuple[int, float, float, float, Optional[float], bool]],
    best_parking: Optional[int],
    drive_path: Optional[List[int]] = None,
    walk_path: Optional[List[int]] = None,
) -> None:
    if out_prefix and os.path.splitext(out_prefix)[1].lower() == ".csv":
        filename = out_prefix
    else:
        filename = os.path.join(out_prefix, "log.csv") if out_prefix else "log.csv"

    file_exists = os.path.isfile(filename)
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Case", "Nodes", "Edges", "ParkingLots", "Source", "Destination", "Alpha", "Wmax", "Cmax",
            "WeightW", "FeasibleParking", "runtime_ms",
            "ParkingNode", "drive_time_hr", "walk_distance_km", "parking_cost",
            "objective", "Feasible", "Selected", "DrivePath", "WalkPath"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        selected_row = None
        if best_parking is not None:
            for row in parking_rows:
                if row[0] == best_parking:
                    selected_row = row
                    break

        if selected_row is None:
            p, drive_t, walk_t, cost_v, score_v, feasible = (-1, math.inf, math.inf, math.inf, math.inf, False)
        else:
            p, drive_t, walk_t, cost_v, score_v, feasible = selected_row

        writer.writerow({
            "Case": case_name,
            "Nodes": Nodes,
            "Edges": Edges,
            "ParkingLots": parking_lots,
            "Source": Source,
            "Destination": Destination,
            "Alpha": Alpha,
            "Wmax": W,
            "Cmax": Cmax,
            "WeightW": weight_w,
            "FeasibleParking": feasible_parking,
            "runtime_ms": round(runtime_ms, 3),
            "ParkingNode": p,
            "drive_time_hr": drive_t,
            "walk_distance_km": walk_t,
            "parking_cost": cost_v,
            "objective": score_v if score_v is not None else "",
            "Feasible": feasible,
            "Selected": (p == best_parking),
            "DrivePath": "->".join(map(str, drive_path or [])),
            "WalkPath": "->".join(map(str, walk_path or [])),
        })


# -----------------------------
# CLI / main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-style heuristic on OPL case files")
    parser.add_argument("--dat", help="Path to one OPL case file (.dat or .txt)")
    parser.add_argument("--dat-dir", default="formatted_cases", help="Directory containing OPL case files (.dat/.txt)")
    parser.add_argument("--out", default="heuristics/log.csv", help="Output CSV path")
    parser.add_argument("--w", type=float, default=None, help="Heuristic weight w. Defaults to value from .dat file.")
    parser.add_argument("--cmax", type=float, default=None, help="Cost threshold Cmax; defaults to value from .dat")
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
        nodes: List[Node] = list(range(1, Nodes + 1))

        edge_time = make_edge_time_map(edges, data["EdgeTime"])
        distance_map = make_edge_weight_map(edges, data["Distance"])
        cost_map: Dict[Node, float] = {p: c for p, c in zip(lots, costs)}
        availability: Dict[Node, int] = {p: 1 for p in lots}

        walk_time_map_data = {p: t for p, t in zip(lots, data.get("WalkTimeToDest", []))}

        use_w = args.w if args.w is not None else dat_w

        t0 = time.time()
        result = heuristic_select_parking(
            nodes=nodes,
            edges=edges,
            A=S,
            D=D,
            lots=lots,
            edge_time=edge_time,
            distance_map=distance_map,
            cost=cost_map,
            availability=availability,
            Wmax=Wmax,
            Cmax=Cmax,
            w=use_w,
            walk_time_map=walk_time_map_data,
        )
        runtime_ms = (time.time() - t0) * 1000.0

        print("Status: solved" if result.get("selected") is not None else "Status: not solved")
        print(f"Objective value: {result.get('score', math.inf)}")

        if result.get("selected") is not None:
            total_time = result["drive_time"] + result["walk_time"]
            print(f"Travel time T[D]: {total_time}")
            print(f"Parking cost: {result['cost']}")
        else:
            print("Travel time T[D]: inf")
            print("Parking cost: inf")

        print("\nSelected route edges:")
        if result.get("selected") is not None:
            drive_path = result.get("drive_path", [])
            walk_path = result.get("walk_path", [])
            full_path = list(drive_path)
            if walk_path:
                full_path.extend(walk_path[1:])
            for a, b in zip(full_path[:-1], full_path[1:]):
                edge_val = edge_time.get((a, b), distance_map.get((a, b), math.inf))
                print(f"  {a} -> {b}   time={edge_val}")

        print("\nSelected parking lot:")
        if result.get("selected") is not None:
            p = result["selected"]
            print(
                f"  node={p}  cost={result['cost']}  "
                f"walkToDest={result['walk_time']}"
            )

        all_scores: Dict[Node, float] = result.get("all_scores", {})
        if all_scores:
            print("\nCandidate scores (lower is better):")
            print("  P      score       drive_time   walk_time   cost")
            for p, sc in sorted(all_scores.items(), key=lambda kv: kv[1]):
                d = result["drive_time_map"].get(p, math.inf)
                wv = result["walk_time_map"].get(p, math.inf)
                c = result["cost_map"].get(p, math.inf)
                sel_mark = "*" if p == result.get("selected") else " "
                print(f"{sel_mark} P={p:>4}  score={sc:>10.6f}  drive={d:>9.4f}  walk={wv:>9.4f}  cost={c:>7.4f}")

        parking_rows = []
        score_map = result.get("score_map", {})
        feasible_set = result.get("feasible_set", set())

        for p in lots:
            drive_t = result.get("drive_time_map", {}).get(p, math.inf)
            walk_t = result.get("walk_time_map", {}).get(p, math.inf)
            cost_v = cost_map.get(p, math.inf)
            score_v = score_map.get(p) if p in feasible_set else None
            feasible = p in feasible_set
            parking_rows.append((p, drive_t, walk_t, cost_v, score_v, feasible))

        write_log_table(
            out_prefix=args.out,
            case_name=dat_path.stem,
            Nodes=Nodes,
            Edges=len(edges),
            parking_lots=len(lots),
            Source=S,
            Destination=D,
            Alpha=Alpha,
            W=Wmax_km,
            Cmax=Cmax,
            weight_w=use_w,
            feasible_parking=(result.get("selected") is not None),
            runtime_ms=runtime_ms,
            parking_rows=parking_rows,
            best_parking=result.get("selected"),
            drive_path=result.get("drive_path"),
            walk_path=result.get("walk_path"),
        )


if __name__ == "__main__":
    main()
