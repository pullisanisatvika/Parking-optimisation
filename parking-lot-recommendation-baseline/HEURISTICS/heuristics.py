import re
import argparse
import math
import time
import heapq
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    return Path(__file__).resolve().parent / "data" / "master_results_heuristic.csv"


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
    walk_time_input: Dict[Node, float],
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
        # Use the precomputed parking-to-destination walking times from the case data.
        tw = walk_time_input.get(p, edge_time.get((p, D), math.inf))
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

    # Stage 4: approximate preselection.
    # Use a drive-biased proxy to keep only a limited shortlist, then evaluate the
    # exact objective within that shortlist. This keeps the method lightweight and
    # intentionally approximate rather than equivalent to full enumeration.
    score_map: Dict[Node, float] = {}
    proxy_score_map: Dict[Node, float] = {}
    travel_time_map: Dict[Node, float] = {}

    for p in pruned:
        total_travel = drive_time_map[p] + walk_time_map[p]
        travel_time_map[p] = total_travel
        score_map[p] = w * (total_travel * 60.0) + (1.0 - w) * cost[p]
        proxy_score_map[p] = (0.85 * w) * (drive_time_map[p] * 60.0) + (0.15 * w) * (walk_time_map[p] * 60.0) + (1.0 - w) * cost[p]

    shortlist_size = max(1, int(math.sqrt(len(pruned))))
    shortlist = sorted(
        pruned,
        key=lambda p: (proxy_score_map[p], drive_time_map[p], walk_time_map[p], cost[p], p),
    )[:shortlist_size]

    best_parking = min(shortlist, key=lambda p: (score_map[p], travel_time_map[p], cost[p], p))

    return {
        "selected": best_parking,
        "score": score_map[best_parking],
        "proxy_score": proxy_score_map[best_parking],
        "drive_time": drive_time_map[best_parking],
        "walk_time": walk_time_map[best_parking],
        "cost": cost[best_parking],
        "drive_path": drive_path_map[best_parking],
        "walk_path": walk_path_map[best_parking],
        "all_scores": score_map,
        "proxy_score_map": proxy_score_map,
        "candidates": pruned,
        "shortlist": shortlist,
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
    del out_prefix

    selected_row = None
    if best_parking is not None:
        for row in parking_rows:
            if row[0] == best_parking:
                selected_row = row
                break

    if selected_row is None:
        p, drive_t, walk_t, cost_v, score_v, feasible = ("", "", "", "", "", False)
    else:
        p, drive_t, walk_t, cost_v, score_v, feasible = selected_row

    walk_distance = ""
    raw_objective = ""
    raw_drive_min = ""
    raw_walk_min = ""
    raw_travel_min = ""
    if selected_row is not None:
        if walk_t not in ("", None):
            walk_distance = str(float(walk_t) * 5.0)
            raw_walk_min = str(float(walk_t) * 60.0)
        if drive_t not in ("", None):
            raw_drive_min = str(float(drive_t) * 60.0)
        if drive_t not in ("", None) and walk_t not in ("", None):
            raw_travel_min = str((float(drive_t) + float(walk_t)) * 60.0)
        if score_v not in ("", None):
            raw_objective = str(
                weight_w * ((float(drive_t) + float(walk_t)) * 60.0) + (1.0 - weight_w) * float(cost_v)
            )

    walk_path_text = "->".join(map(str, walk_path or []))
    existing_rows = _load_master_rows(_master_results_path())
    row_by_case = {
        _normalize_case_name(row.get("dat_file", "")): row
        for row in existing_rows
        if row.get("dat_file", "")
    }
    normalized_case_name = _normalize_case_name(case_name)
    row_by_case[normalized_case_name] = {
        "method": "HEURISTIC",
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
        "feasible": "1" if feasible_parking else "0",
        "parking_node": str(p),
        "drive_time_hr": str(drive_t),
        "walk_distance_km": walk_distance,
        "walk_time_hr": str(walk_t),
        "parking_cost": str(cost_v),
        "objective": "" if score_v in ("", None) else str(score_v),
        "non_normalized_drive_time_min": raw_drive_min,
        "non_normalized_walk_time_min": raw_walk_min,
        "non_normalized_travel_time_min": raw_travel_min,
        "non_normalized_parking_cost": str(cost_v),
        "non_normalized_objective": raw_objective,
        "runtime_ms": str(round(runtime_ms, 3)),
        "drive_path": "->".join(map(str, drive_path or [])),
        "walk_path": walk_path_text,
    }
    rows = sorted(row_by_case.values(), key=lambda row: _sort_case_key(row.get("dat_file", "")))
    _write_master_rows(_master_results_path(), rows)


# -----------------------------
# CLI / main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-style heuristic on OPL case files")
    parser.add_argument("--dat", help="Path to one OPL case file (.dat or .txt)")
    parser.add_argument("--dat-dir", default="formatted_cases", help="Directory containing OPL case files (.dat/.txt)")
    parser.add_argument("--out", default="", help="Deprecated; results always write to data/master_results_heuristic.csv")
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
        try:
            data = parse_opl_dat(dat_text)
        except ValueError as exc:
            print(f"Skipping {dat_path}: {exc}")
            continue

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
            walk_time_input=walk_time_map_data,
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
                f"walkToDest={result['walk_time']}  proxyScore={result.get('proxy_score', math.inf)}"
            )

        all_scores: Dict[Node, float] = result.get("all_scores", {})
        proxy_scores: Dict[Node, float] = result.get("proxy_score_map", {})
        shortlist = set(result.get("shortlist", []))
        if all_scores:
            print("\nCandidate scores (proxy selects shortlist, objective ranks shortlist):")
            print("  P      objective    proxy        drive_time   walk_time   cost")
            for p, sc in sorted(all_scores.items(), key=lambda kv: (proxy_scores.get(kv[0], math.inf), kv[1])):
                d = result["drive_time_map"].get(p, math.inf)
                wv = result["walk_time_map"].get(p, math.inf)
                c = result["cost_map"].get(p, math.inf)
                sel_mark = "*" if p == result.get("selected") else " "
                shortlist_mark = "s" if p in shortlist else " "
                proxy = proxy_scores.get(p, math.inf)
                print(
                    f"{sel_mark}{shortlist_mark} P={p:>4}  objective={sc:>10.6f}  "
                    f"proxy={proxy:>10.6f}  drive={d:>9.4f}  walk={wv:>9.4f}  cost={c:>7.4f}"
                )

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
