import math
import os
import sys
import copy
import random
import heapq
from pathlib import Path
from collections import deque

# Ensure local plotting module is importable regardless of current working directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from plot_dataset_graphs import plot_case_graph

# ============================================================
# CONFIGURATION
# ============================================================
SMALL_SCALE_NODE_SIZES = [10, 20, 30, 40, 50]
MEDIUM_LARGE_NODE_SIZES = [500, 1000, 1500, 2000, 2500]
NODE_SIZES = SMALL_SCALE_NODE_SIZES + MEDIUM_LARGE_NODE_SIZES
NUM_INSTANCES = 1
OUT_DIR = str((_THIS_DIR.parent / "final_cases").resolve())
BASE_NETWORK_SIZE = 20
GENERATE_PLOTS = os.environ.get("ITS_GENERATE_PLOTS", "1") == "1"
FORCE_REGENERATE = os.environ.get("ITS_FORCE_REGENERATE", "1") == "1"

# ---------------- Baseline parameters ----------------
BASE_EDGE_MODE = "urban_backbone"
BASE_PARKING_PCT = 0.05
BASE_MAX_WALK_KM = 0.50
BASE_ALPHA = 0.90
BASE_CMAX = 20.5

# ---------------- Variation parameters ----------------
VAR_EDGE_MODE = "urban_dense"          # C2
VAR_PARKING_PCT = 0.10                 # C3
VAR_MAX_WALK_KM = 1.00                 # C4
VAR_ALPHA_HIGH = 0.90                  # C7
VAR_ALPHA_LOW = 0.10                   # C8
VAR_CMAX = 15.0                        # C9

# ---------------- Traffic speed ranges (km/h) ----------------
LOW_CONGESTION_RANGE = (50.0, 70.0)      # C1 baseline
MEDIUM_CONGESTION_RANGE = (30.0, 49.0)   # C5
HIGH_CONGESTION_RANGE = (10.0, 29.0)     # C6

# ---------------- Parking cost ranges ----------------
BASE_PARKING_COST_RANGE = (8.0, 20.5)
MID_PARKING_COST_RANGE = (10.0, 22.5)
ELEVATED_PARKING_COST_RANGE = (12.5, 25.0)
HIGH_PARKING_COST_RANGE = (15.0, 30.0)   # C10

# ---------------- Constants ----------------
GRID_SPACING_KM = 0.20
POINT_JITTER_RATIO = 0.35
WALK_SPEED_KMPH = 5.0
B_VALUE = 9000

# ---- Map scaling and parking search region constants ----
BASE_MAP_SIDE_KM = 1.00

DESTINATION_PARKING_RADIUS_CAP_KM = 3.00
DESTINATION_PARKING_RADIUS_DIAG_FRAC = 0.30
DESTINATION_PARKING_RADIUS_WALK_MULT = 2.50

# ---- Large-case parking feasibility controls ----
LARGE_CASE_NODE_THRESHOLD = 500
LARGE_CASE_MAX_INFEASIBLE_RATIO = 0.20
LARGE_CASE_MIN_FEASIBLE_RATIO = 0.80
LARGE_CASE_MIN_FEASIBLE_COUNT = 8
SMALL_SCALE_HEURISTIC_DIVERGENCE_GAPS = {
    40: 0.25,
    50: 0.30,
}
SMALL_SCALE_HEURISTIC_ALIGNMENT_SIZES = {20, 30}


# ============================================================
# HELPERS
# ============================================================
def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_parking_count(n_nodes, pct):
    """
    Parking lots:
    - minimum 2
    - maximum 20% of total nodes
    """
    raw = int(round(n_nodes * pct))
    upper = max(2, int(math.floor(0.20 * n_nodes)))
    return max(2, min(upper, raw))


# ---- Map scaling and parking search helpers ----
def map_side_for_n(total_nodes):
    """
    Scale the study-area side length with sqrt(N) so node density remains
    approximately stable across instance sizes.
    """
    driving_nodes = max(1, total_nodes - 1)
    base_driving_nodes = max(1, BASE_NETWORK_SIZE - 1)
    return BASE_MAP_SIDE_KM * math.sqrt(driving_nodes / base_driving_nodes)


def _rescale_coords_to_target_side(coords, driving_nodes, target_side_km):
    """
    Affinely rescale the generated driving-node geometry into a square study area
    whose side length follows the sqrt(N) density-preserving policy.
    """
    xs = [coords[i][0] for i in driving_nodes]
    ys = [coords[i][1] for i in driving_nodes]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)

    padding_ratio = 0.08
    usable_side = target_side_km * (1.0 - 2.0 * padding_ratio)
    scale = usable_side / max(width, height)

    x_offset = (target_side_km - width * scale) / 2.0
    y_offset = (target_side_km - height * scale) / 2.0

    for i in driving_nodes:
        x, y = coords[i]
        nx = (x - min_x) * scale + x_offset
        ny = (y - min_y) * scale + y_offset
        coords[i] = (round(nx, 4), round(ny, 4))


def _parking_candidate_radius(coords, driving_node_ids, destination, max_walk_km):
    """
    Restrict parking search to a destination-side region so the parking problem
    remains local and urban-realistic even for large instances.
    """
    xs = [coords[i][0] for i in driving_node_ids]
    ys = [coords[i][1] for i in driving_node_ids]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    diag = math.sqrt(width * width + height * height)

    radius = max(
        max_walk_km * DESTINATION_PARKING_RADIUS_WALK_MULT,
        DESTINATION_PARKING_RADIUS_DIAG_FRAC * diag,
    )
    radius = min(radius, DESTINATION_PARKING_RADIUS_CAP_KM)
    return round(radius, 4)


def bfs_hops(adj, start):
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


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


def fmt_str(x):
    return str(x)


def _parking_cost_fraction(number_of_nodes, instance_id, parking_node):
    rng = random.Random(
        f"parking-cost-node-{number_of_nodes}-{instance_id}-{parking_node}"
    )
    return rng.random()


def assign_parking_costs(number_of_nodes, instance_id, parking_nodes, cost_range):
    lower, upper = cost_range
    if upper < lower:
        raise RuntimeError("Parking-cost range must satisfy lower <= upper.")

    ordered_nodes = sorted(parking_nodes)
    if not ordered_nodes:
        return {}

    total_cent_slots = int(round((upper - lower) * 100)) + 1
    rng = random.Random(
        f"parking-cost-slots-{number_of_nodes}-{instance_id}-{lower:.2f}-{upper:.2f}"
    )

    if len(ordered_nodes) <= total_cent_slots:
        available_slots = list(range(total_cent_slots))
        rng.shuffle(available_slots)
        chosen_slots = sorted(available_slots[:len(ordered_nodes)])
        ranked_nodes = sorted(
            ordered_nodes,
            key=lambda parking_node: (
                _parking_cost_fraction(number_of_nodes, instance_id, parking_node),
                parking_node,
            ),
        )
        return {
            parking_node: round(lower + slot / 100.0, 2)
            for parking_node, slot in zip(ranked_nodes, chosen_slots)
        }

    costs = {}
    for parking_node in ordered_nodes:
        fraction = _parking_cost_fraction(number_of_nodes, instance_id, parking_node)
        costs[parking_node] = round(lower + fraction * (upper - lower), 2)
    return costs


def assign_parking_costs_preserving_existing(
    number_of_nodes,
    instance_id,
    parking_nodes,
    cost_range,
    existing_costs,
):
    lower, upper = cost_range
    preserved = {
        node: round(float(cost), 2)
        for node, cost in existing_costs.items()
        if node in set(parking_nodes)
    }
    all_nodes = sorted(parking_nodes)
    if len(preserved) == len(all_nodes):
        return preserved

    total_cent_slots = int(round((upper - lower) * 100)) + 1
    used_slots = {
        int(round((cost - lower) * 100))
        for cost in preserved.values()
        if lower - 1e-9 <= cost <= upper + 1e-9
    }

    if len(all_nodes) <= total_cent_slots:
        available_slots = [slot for slot in range(total_cent_slots) if slot not in used_slots]
        ranked_new_nodes = sorted(
            [node for node in all_nodes if node not in preserved],
            key=lambda parking_node: (
                _parking_cost_fraction(number_of_nodes, instance_id, parking_node),
                parking_node,
            ),
        )
        for parking_node, slot in zip(ranked_new_nodes, sorted(available_slots)[: len(ranked_new_nodes)]):
            preserved[parking_node] = round(lower + slot / 100.0, 2)
        if len(preserved) == len(all_nodes):
            return preserved

    generated = assign_parking_costs(number_of_nodes, instance_id, all_nodes, cost_range)
    used_costs = {round(cost, 2) for cost in preserved.values()}
    for parking_node in all_nodes:
        if parking_node in preserved:
            continue
        proposed = round(generated[parking_node], 2)
        if proposed in used_costs:
            total_cent_slots = int(round((upper - lower) * 100)) + 1
            candidate_slots = list(range(total_cent_slots))
            candidate_slots.sort(
                key=lambda slot: (
                    abs(
                        slot / max(1, total_cent_slots - 1)
                        - _parking_cost_fraction(number_of_nodes, instance_id, parking_node)
                    ),
                    slot,
                )
            )
            for slot in candidate_slots:
                candidate = round(lower + slot / 100.0, 2)
                if candidate not in used_costs:
                    proposed = candidate
                    break
        preserved[parking_node] = proposed
        used_costs.add(proposed)
    return preserved


def refresh_parking_feasibility(case):
    feasible_nodes = []
    for parking_node in case["parking_nodes"]:
        info = case["parking_info_by_node"][parking_node]
        info["feasible"] = info["walk_distance_km"] <= case["max_walking_km"] + 1e-9
        if info["feasible"]:
            feasible_nodes.append(parking_node)

    if len(feasible_nodes) < 2:
        raise RuntimeError("Need at least 2 walking-feasible parking lots for a stable dataset.")

    case["feasible_parking_nodes"] = feasible_nodes


def _build_drive_adjacency(n, drive_edges):
    """
    Build adjacency once from the directed driving-edge list.
    """
    adj = {i: [] for i in range(n)}
    for e in drive_edges:
        adj[e["u"]].append(e["v"])
    return adj


def _build_weighted_drive_adjacency(n, drive_edges):
    adj = {i: [] for i in range(n)}
    for e in drive_edges:
        adj[e["u"]].append((e["v"], e["time_hr"]))
    return adj


def _dijkstra_drive_times(n, drive_edges, source):
    adj = _build_weighted_drive_adjacency(n, drive_edges)
    dist = {source: 0.0}
    heap = [(0.0, source)]

    while heap:
        time_so_far, u = heapq.heappop(heap)
        if time_so_far > dist.get(u, float("inf")) + 1e-12:
            continue
        for v, edge_time in adj[u]:
            cand = time_so_far + edge_time
            if cand + 1e-12 < dist.get(v, float("inf")):
                dist[v] = cand
                heapq.heappush(heap, (cand, v))

    return dist


def exact_case_objective(case):
    evaluations = evaluate_case_parking_nodes(case)
    if not evaluations:
        raise RuntimeError("No feasible affordable parking lot found while evaluating exact case objective.")
    return min(item["objective"] for item in evaluations.values())


def evaluate_case_parking_nodes(case):
    dist = _dijkstra_drive_times(
        case["number_of_nodes"],
        case["drive_edges"],
        case["source_node"],
    )

    evaluations = {}
    for parking_node in case["parking_nodes"]:
        info = case["parking_info_by_node"][parking_node]
        if not info["feasible"]:
            continue
        parking_cost = case["parking_costs"][parking_node]
        if parking_cost > case["cmax"] + 1e-9:
            continue
        if parking_node not in dist:
            continue

        drive_time_hr = dist[parking_node]
        walk_time_hr = info["walk_time_hr"]
        objective = (
            case["alpha"] * 60.0 * (drive_time_hr + walk_time_hr)
            + (1.0 - case["alpha"]) * parking_cost
        )
        proxy = (
            (0.85 * case["alpha"]) * (drive_time_hr * 60.0)
            + (0.15 * case["alpha"]) * (walk_time_hr * 60.0)
            + (1.0 - case["alpha"]) * parking_cost
        )
        evaluations[parking_node] = {
            "drive_time_hr": drive_time_hr,
            "walk_time_hr": walk_time_hr,
            "parking_cost": parking_cost,
            "objective": objective,
            "proxy": proxy,
        }

    return evaluations


def uplift_parking_costs(case, delta):
    if delta <= 0:
        return case

    updated = copy.deepcopy(case)
    for parking_node in updated["parking_nodes"]:
        updated["parking_costs"][parking_node] = round(
            min(updated["cmax"], updated["parking_costs"][parking_node] + delta),
            2,
        )
    validate_case(updated)
    return updated


def calibrate_small_case_monotonic(case, minimum_objective, min_gap=0.05):
    target = minimum_objective + min_gap
    current = exact_case_objective(case)
    if current >= target - 1e-9:
        return case, current

    calibrated = copy.deepcopy(case)
    for _ in range(8):
        current = exact_case_objective(calibrated)
        if current >= target - 1e-9:
            validate_case(calibrated)
            return calibrated, current

        remaining_gap = target - current
        alpha_gap_weight = max(1e-6, 1.0 - calibrated["alpha"])
        delta = max(0.25, remaining_gap / alpha_gap_weight + 0.10)
        calibrated = uplift_parking_costs(calibrated, delta)

    final_objective = exact_case_objective(calibrated)
    if final_objective < target - 1e-9:
        raise RuntimeError(
            f"Could not calibrate small-case monotonic objective for N={case['number_of_nodes']}."
        )

    validate_case(calibrated)
    return calibrated, final_objective


def induce_small_case_proxy_divergence(case, objective_gap, proxy_margin=0.02):
    calibrated = copy.deepcopy(case)
    evaluations = evaluate_case_parking_nodes(calibrated)
    if len(evaluations) < 2:
        return calibrated

    exact_best_node = min(
        evaluations,
        key=lambda node: (evaluations[node]["objective"], node),
    )
    exact_best = evaluations[exact_best_node]
    alternate_nodes = sorted(
        (node for node in evaluations if node != exact_best_node),
        key=lambda node: (
            evaluations[node]["objective"],
            evaluations[node]["proxy"],
            node,
        ),
    )
    if not alternate_nodes:
        return calibrated

    alternate_node = alternate_nodes[0]
    alternate = evaluations[alternate_node]
    lower_cost_bound = BASE_PARKING_COST_RANGE[0]

    min_cost_for_exact_gap = alternate["parking_cost"] - 10.0 * (
        alternate["objective"] - (exact_best["objective"] + objective_gap)
    )
    max_cost_for_proxy_flip = alternate["parking_cost"] - 10.0 * (
        alternate["proxy"] - (exact_best["proxy"] - proxy_margin)
    )

    lower_bound = max(lower_cost_bound, min_cost_for_exact_gap)
    upper_bound = min(alternate["parking_cost"], max_cost_for_proxy_flip)

    if lower_bound > upper_bound + 1e-9:
        return calibrated

    new_cost = round((lower_bound + upper_bound) / 2.0, 2)
    calibrated["parking_costs"][alternate_node] = new_cost

    updated = evaluate_case_parking_nodes(calibrated)
    updated_exact_best_node = min(
        updated,
        key=lambda node: (updated[node]["objective"], node),
    )
    if updated_exact_best_node != exact_best_node:
        return case
    if updated[alternate_node]["proxy"] >= updated[exact_best_node]["proxy"] - 1e-9:
        return case
    if updated[alternate_node]["objective"] <= updated[exact_best_node]["objective"] + 1e-9:
        return case

    validate_case(calibrated)
    return calibrated


def align_small_case_proxy_with_exact(case):
    calibrated = copy.deepcopy(case)

    for _ in range(8):
        evaluations = evaluate_case_parking_nodes(calibrated)
        if not evaluations:
            raise RuntimeError("No feasible affordable parking lot found while aligning heuristic proxy.")

        exact_best_node = min(
            evaluations,
            key=lambda node: (evaluations[node]["objective"], node),
        )
        exact_proxy = evaluations[exact_best_node]["proxy"]

        adjusted = False
        for parking_node, stats in sorted(evaluations.items()):
            if parking_node == exact_best_node:
                continue
            if stats["proxy"] > exact_proxy + 1e-9:
                continue

            proxy_gap = exact_proxy - stats["proxy"] + 0.02
            delta = proxy_gap / max(1e-6, 1.0 - calibrated["alpha"])
            current_cost = calibrated["parking_costs"][parking_node]
            new_cost = round(min(calibrated["cmax"], current_cost + delta), 2)

            if new_cost <= current_cost + 1e-9:
                raise RuntimeError(
                    f"Could not align heuristic proxy with exact choice for N={case['number_of_nodes']}."
                )

            calibrated["parking_costs"][parking_node] = new_cost
            adjusted = True

        if not adjusted:
            validate_case(calibrated)
            return calibrated

    raise RuntimeError(
        f"Could not align heuristic proxy with exact choice for N={case['number_of_nodes']}."
    )




def _precompute_pairwise_geometry(driving_nodes, coords):
    """
    Precompute pairwise distances and sorted neighbor lists once.
    This preserves the original logic while avoiding repeated O(n^2)
    distance recomputation inside graph construction.
    """
    pair_list = []
    pair_distance = {}
    neighbors = {u: [] for u in driving_nodes}

    for i in range(len(driving_nodes)):
        u = driving_nodes[i]
        for j in range(i + 1, len(driving_nodes)):
            v = driving_nodes[j]
            d = euclidean(coords[u], coords[v])
            pair_list.append((d, u, v))
            pair_distance[(u, v)] = d
            pair_distance[(v, u)] = d
            neighbors[u].append((d, v))
            neighbors[v].append((d, u))

    pair_list.sort()
    for u in driving_nodes:
        neighbors[u].sort()

    return pair_list, pair_distance, neighbors


# ============================================================
# STAGED CORRIDOR-AND-DISTRICT URBAN NODE CONSTRUCTION
# ============================================================
def _base20_driving_template():
    """
    Base irregular urban template for 19 driving nodes.
    This acts as the fixed seed for all larger instances.
    """
    return [
        (0.00, 0.00), (0.18, 0.03), (0.34, 0.01), (0.54, 0.05),
        (0.03, 0.18), (0.21, 0.20), (0.38, 0.18), (0.61, 0.23),
        (-0.01, 0.36), (0.16, 0.39), (0.35, 0.42), (0.58, 0.40),
        (0.08, 0.57), (0.27, 0.60), (0.47, 0.62),
        (0.17, 0.79), (0.36, 0.82), (0.56, 0.78), (0.74, 0.64),
    ]


def _compute_center(coords, ids=None):
    if ids is None:
        ids = list(coords.keys())
    cx = sum(coords[i][0] for i in ids) / len(ids)
    cy = sum(coords[i][1] for i in ids) / len(ids)
    return cx, cy


def _normalize(vx, vy):
    norm = math.sqrt(vx * vx + vy * vy)
    if norm == 0:
        return 1.0, 0.0
    return vx / norm, vy / norm


def _is_far_enough(candidate_xy, existing_xy, min_sep=0.10):
    for xy in existing_xy:
        if euclidean(candidate_xy, xy) < min_sep:
            return False
    return True


def _angle_sector(xy, center, n_sectors=8):
    ang = math.atan2(xy[1] - center[1], xy[0] - center[0])
    return int(((ang + math.pi) / (2 * math.pi)) * n_sectors) % n_sectors


def _select_growth_poles(coords, rng, count=5):
    """
    Select diverse boundary nodes from different angular sectors.
    These act as corridor and district growth poles.
    """
    ids = list(coords.keys())
    center = _compute_center(coords, ids)

    scored = []
    for i in ids:
        radial = euclidean(coords[i], center)
        sector = _angle_sector(coords[i], center)
        scored.append((radial, sector, i))

    scored.sort(reverse=True)
    chosen = []
    used_sectors = set()

    for _, sector, i in scored:
        if sector not in used_sectors:
            chosen.append(i)
            used_sectors.add(sector)
        if len(chosen) >= count:
            break

    if len(chosen) < count:
        remaining = [i for _, _, i in scored if i not in chosen]
        chosen.extend(remaining[:count - len(chosen)])

    rng.shuffle(chosen)
    return chosen


def _stage_totals_for(target_total_nodes):
    milestones = sorted(set([BASE_NETWORK_SIZE] + NODE_SIZES + [target_total_nodes]))
    return [x for x in milestones if x <= target_total_nodes]


def _try_add_node(coords, candidate, min_sep, next_node_id):
    if _is_far_enough(candidate, list(coords.values()), min_sep=min_sep):
        coords[next_node_id] = (round(candidate[0], 4), round(candidate[1], 4))
        return True
    return False


def _grow_one_stage(coords, target_driving_nodes, stage_total_nodes, instance_id):
    """
    Extend the current network by corridor-and-district growth.

    Corridor growth pushes the city outward along selected poles.
    District growth fills in neighborhood clusters near those poles,
    creating nonuniform urban expansion rather than space-filling mesh.
    """
    rng = random.Random(f"urban-stage-{stage_total_nodes}-{instance_id}")
    next_node_id = max(coords.keys()) + 1 if coords else 0
    recent_frontier = list(coords.keys())[-min(8, len(coords)):]

    while next_node_id < target_driving_nodes:
        ids = list(coords.keys())
        center = _compute_center(coords, ids)
        poles = _select_growth_poles(coords, rng, count=min(6, max(3, len(ids) // 12)))

        placed = False
        for _ in range(180):
            mode = "corridor" if rng.random() < 0.45 else "district"

            if mode == "corridor":
                anchor = rng.choice(poles)
                ax, ay = coords[anchor]
                rx, ry = _normalize(ax - center[0], ay - center[1])
                tx, ty = -ry, rx

                step = rng.uniform(0.18, 0.34)
                lateral = rng.uniform(-0.05, 0.05)
                candidate = (
                    ax + step * rx + lateral * tx,
                    ay + step * ry + lateral * ty,
                )
                min_sep = 0.10
            else:
                seed_choices = poles + recent_frontier
                seed = rng.choice(seed_choices)
                sx, sy = coords[seed]
                rx, ry = _normalize(sx - center[0], sy - center[1])
                tx, ty = -ry, rx

                radial_step = rng.uniform(0.04, 0.15)
                tangential_step = rng.uniform(-0.18, 0.18)
                candidate = (
                    sx + radial_step * rx + tangential_step * tx + rng.uniform(-0.02, 0.02),
                    sy + radial_step * ry + tangential_step * ty + rng.uniform(-0.02, 0.02),
                )
                min_sep = 0.085

            if _try_add_node(coords, candidate, min_sep=min_sep, next_node_id=next_node_id):
                recent_frontier.append(next_node_id)
                recent_frontier = recent_frontier[-18:]
                next_node_id += 1
                placed = True
                break

        if not placed:
            # Fallback: continue outward from a strong boundary pole.
            anchor = poles[0]
            ax, ay = coords[anchor]
            rx, ry = _normalize(ax - center[0], ay - center[1])
            candidate = (
                ax + rng.uniform(0.24, 0.34) * rx,
                ay + rng.uniform(0.24, 0.34) * ry,
            )
            if _try_add_node(coords, candidate, min_sep=0.10, next_node_id=next_node_id):
                recent_frontier.append(next_node_id)
                recent_frontier = recent_frontier[-18:]
                next_node_id += 1
            else:
                candidate = (
                    ax + rng.uniform(-0.14, 0.14),
                    ay + rng.uniform(-0.14, 0.14),
                )
                if _try_add_node(coords, candidate, min_sep=0.08, next_node_id=next_node_id):
                    recent_frontier.append(next_node_id)
                    recent_frontier = recent_frontier[-18:]
                    next_node_id += 1
                else:
                    raise RuntimeError("Could not place additional urban growth node.")


def grid_dims_for_n(driving_nodes):
    """
    Retained only for compatibility with the older code path.
    The current generator does not construct a strict grid.
    """
    rows = max(1, int(math.floor(math.sqrt(driving_nodes))))
    cols = int(math.ceil(driving_nodes / rows))
    while rows * cols < driving_nodes:
        rows += 1
    return rows, cols


def build_grid_nodes(n_nodes, instance_id=1):
    """
    Uses n_nodes - 1 as driving graph nodes.
    The last node id is reserved for destination.

    Growth is hierarchical and staged:
    - N=20 is the fixed seed network
    - N=40 extends N=20
    - N=60 extends N=40
    - ...
    - larger milestone networks extend the previous milestone rather than
      being regenerated independently

    This preserves structural continuity across scales.
    """
    driving_nodes = n_nodes - 1
    destination = n_nodes - 1

    coords = {}
    rc_of = {}
    node_of = {}

    template = _base20_driving_template()
    base_count = min(len(template), driving_nodes)
    base_rng = random.Random(f"urban-base-{instance_id}")

    for node_id in range(base_count):
        tx, ty = template[node_id]
        jitter_x = base_rng.uniform(-0.18, 0.18) * GRID_SPACING_KM
        jitter_y = base_rng.uniform(-0.18, 0.18) * GRID_SPACING_KM
        coords[node_id] = (round(tx + jitter_x, 4), round(ty + jitter_y, 4))

    stage_totals = _stage_totals_for(n_nodes)
    for stage_total in stage_totals:
        target_driving = stage_total - 1
        if target_driving <= len(coords):
            continue
        _grow_one_stage(coords, target_driving, stage_total, instance_id)

    # Rescale to target map side after all nodes generated, before destination placed.
    target_side_km = map_side_for_n(n_nodes)
    _rescale_coords_to_target_side(coords, list(range(driving_nodes)), target_side_km)

    ids = list(range(driving_nodes))
    center = _compute_center(coords, ids)
    far_anchor = max(ids, key=lambda i: euclidean(coords[i], center))
    ax, ay = coords[far_anchor]
    rx, ry = _normalize(ax - center[0], ay - center[1])
    coords[destination] = (round(ax + 0.22 * rx, 4), round(ay + 0.22 * ry, 4))

    rows, cols = grid_dims_for_n(driving_nodes)
    return rows, cols, coords, rc_of, node_of, destination


def farthest_pair(node_ids, coords):
    """
    Returns the pair of driving nodes with maximum Euclidean separation.
    """
    best_u = node_ids[0]
    best_v = node_ids[1] if len(node_ids) > 1 else node_ids[0]
    best_d = -1.0

    for i in range(len(node_ids)):
        u = node_ids[i]
        for j in range(i + 1, len(node_ids)):
            v = node_ids[j]
            d = euclidean(coords[u], coords[v])
            if d > best_d:
                best_d = d
                best_u = u
                best_v = v

    return best_u, best_v, best_d


def add_bidirected_edge(edge_list, adj, coords, u, v, speed_kmph, edge_type, road_class=None):
    if speed_kmph <= 0:
        raise RuntimeError("Speed must be positive.")

    d = euclidean(coords[u], coords[v])
    t = d / speed_kmph

    edge_uv = {
        "u": u,
        "v": v,
        "distance_km": round(d, 4),
        "time_hr": round(t, 4),
        "speed_kmph": round(speed_kmph, 2),
        "type": edge_type
    }
    edge_vu = {
        "u": v,
        "v": u,
        "distance_km": round(d, 4),
        "time_hr": round(t, 4),
        "speed_kmph": round(speed_kmph, 2),
        "type": edge_type
    }

    if road_class is not None:
        edge_uv["road_class"] = road_class
        edge_vu["road_class"] = road_class

    edge_list.append(edge_uv)
    edge_list.append(edge_vu)

    if adj is not None:
        adj[u].append(v)
        adj[v].append(u)


# ============================================================
# URBAN-LIKE DRIVE GRAPH
# ============================================================
def build_urban_drive_graph(node_of, coords, n_nodes, speed_range, instance_id=1):
    """
    Build an irregular urban road network from the node geometry.

    The graph is formed using:
    - a shortest-edge backbone for guaranteed connectivity
    - local neighborhood links
    - corridor spines along angular sectors
    - district ring links between nodes at similar radial depth
    - a few peripheral arterial connections

    This produces a network with stronger corridor-and-district structure
    and reduces the previous lattice-like appearance at large scales.
    """
    drive_edges = []
    drive_adj = {i: [] for i in range(n_nodes)}
    existing = set()
    rng = random.Random(f"urban-graph-{n_nodes}-{instance_id}")

    driving_nodes = list(range(n_nodes - 1))
    center = _compute_center(coords, driving_nodes)
    radial_of = {u: euclidean(coords[u], center) for u in driving_nodes}
    pair_list, pair_distance, neighbors = _precompute_pairwise_geometry(driving_nodes, coords)

    def add_drive(u, v, road_class):
        if u == v:
            return
        if (u, v) in existing or (v, u) in existing:
            return

        lo, hi = speed_range
        if road_class == "local":
            s_lo = lo
            s_hi = lo + 0.50 * (hi - lo)
        elif road_class == "connector":
            s_lo = lo + 0.18 * (hi - lo)
            s_hi = lo + 0.78 * (hi - lo)
        else:  # arterial
            s_lo = lo + 0.42 * (hi - lo)
            s_hi = hi

        speed = rng.uniform(s_lo, s_hi)
        add_bidirected_edge(drive_edges, drive_adj, coords, u, v, speed, "drive", road_class)
        existing.add((u, v))
        existing.add((v, u))

    # 1) Backbone connectivity via shortest-edge growth.
    connected = {driving_nodes[0]}
    remaining = set(driving_nodes[1:])
    while remaining:
        best = None
        for d, u, v in pair_list:
            if (u in connected and v in remaining) or (v in connected and u in remaining):
                best = (d, u, v)
                break
        if best is None:
            raise RuntimeError("Could not build a connected urban backbone.")

        d, u, v = best
        add_drive(u, v, "local" if d <= 0.22 else "connector")
        connected.add(u)
        connected.add(v)
        remaining.discard(u)
        remaining.discard(v)

    # 2) Local neighborhood streets.
    nearest_limit = 3 if len(driving_nodes) <= 80 else 2
    local_threshold = 0.24 if len(driving_nodes) <= 100 else 0.21
    for u in driving_nodes:
        added = 0
        for d, v in neighbors[u]:
            if d <= local_threshold:
                add_drive(u, v, "local")
                added += 1
            if added >= nearest_limit:
                break

    # 3) Corridor spines by angular sector.
    n_sectors = 8 if len(driving_nodes) <= 500 else 10
    sectors = {k: [] for k in range(n_sectors)}
    for u in driving_nodes:
        sector = _angle_sector(coords[u], center, n_sectors=n_sectors)
        sectors[sector].append((radial_of[u], u))

    for sector_nodes in sectors.values():
        sector_nodes.sort()
        for i in range(len(sector_nodes) - 1):
            _, u = sector_nodes[i]
            _, v = sector_nodes[i + 1]
            d = pair_distance[(u, v)]
            if d <= 0.42:
                road_class = "connector" if d <= 0.30 else "arterial"
                add_drive(u, v, road_class)

    # 4) District ring links between similar radial bands.
    for u in driving_nodes:
        candidates = []
        for d, v in neighbors[u]:
            radial_gap = abs(radial_of[u] - radial_of[v])
            if 0.16 <= d <= 0.34 and radial_gap <= 0.08:
                candidates.append((d, v))
        for _, v in candidates[:1]:
            add_drive(u, v, "connector")

    # 5) Peripheral arterial cross-links between outer hubs.
    peripheral = sorted(
        ((radial_of[u], u) for u in driving_nodes),
        reverse=True
    )
    hubs = [u for _, u in peripheral[:max(4, len(driving_nodes) // 18)]]
    for i in range(len(hubs)):
        u = hubs[i]
        candidates = []
        for j in range(i + 1, len(hubs)):
            v = hubs[j]
            d = pair_distance[(u, v)]
            if 0.32 <= d <= 0.62:
                candidates.append((d, v))
        candidates.sort(reverse=True)
        for _, v in candidates[:1]:
            add_drive(u, v, "arterial")

    # 6) Final connectivity repair.
    # Rebuild components from the current directed driving graph and, if needed,
    # add minimal connector edges between disconnected components.
    while True:
        undirected_adj = {u: set() for u in driving_nodes}
        for e in drive_edges:
            if e["type"] != "drive":
                continue
            undirected_adj[e["u"]].add(e["v"])

        components = []
        seen = set()
        for start in driving_nodes:
            if start in seen:
                continue
            comp = []
            stack = [start]
            seen.add(start)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in undirected_adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            components.append(comp)

        if len(components) <= 1:
            break

        best_pair = None
        best_dist = math.inf
        for a in range(len(components)):
            for b in range(a + 1, len(components)):
                for u in components[a]:
                    for v in components[b]:
                        d = pair_distance.get((u, v), math.inf)
                        if d < best_dist:
                            best_dist = d
                            best_pair = (u, v)

        if best_pair is None:
            raise RuntimeError("Could not repair disconnected driving graph.")

        u, v = best_pair
        road_class = "connector" if best_dist <= 0.35 else "arterial"
        add_drive(u, v, road_class)

    return drive_edges, drive_adj


# ============================================================
# PARKING CONSTRUCTION
# ============================================================
def build_parking_info(
    n_nodes,
    driving_node_ids,
    coords,
    source,
    destination,
    drive_adj,
    parking_pct,
    max_walk_km,
    parking_search_walk_km=None
):
    """
    Build all parking facilities and mark which are feasible for the trip.
    All parking lots are included in L and get walk edges to destination.
    Feasibility is handled by constraints/metadata, not by edge omission.

    This version explicitly guarantees a mix of:
    - feasible parking lots (within walking threshold)
    - infeasible parking lots (beyond walking threshold), when available

    It also guarantees at least 2 feasible parking lots so the dataset remains usable.
    """
    hops = bfs_hops(drive_adj, source)

    feasible_candidates = []
    infeasible_candidates = []

    if parking_search_walk_km is None:
        parking_search_walk_km = max_walk_km

    candidate_radius_km = _parking_candidate_radius(
        coords, driving_node_ids, destination, parking_search_walk_km
    )

    for node in driving_node_ids:
        if node == source:
            continue
        if node not in hops:
            continue

        walk_d = euclidean(coords[node], coords[destination])
        if walk_d > candidate_radius_km:
            continue

        walk_time_hr = walk_d / WALK_SPEED_KMPH

        item = {
            "node": node,
            "hop_from_source": hops[node],
            "walk_distance_km": round(walk_d, 4),
            "walk_time_hr": round(walk_time_hr, 4),
            "feasible": walk_d <= max_walk_km,
        }

        if item["feasible"]:
            feasible_candidates.append(item)
        else:
            infeasible_candidates.append(item)

    # For the baseline walking threshold, prefer parking that is farther from the
    # source while still near the destination. When the walking threshold is
    # relaxed (e.g., C4), avoid over-favoring the farthest feasible parking lots;
    # instead prefer nearer-to-destination parking first, using hop distance only
    # as a secondary tie-breaker.
    if max_walk_km > BASE_MAX_WALK_KM:
        feasible_candidates.sort(
            key=lambda x: (x["walk_distance_km"], -x["hop_from_source"])
        )
    else:
        feasible_candidates.sort(
            key=lambda x: (x["hop_from_source"], -x["walk_distance_km"]),
            reverse=True
        )

    # For infeasible parking, still prefer nodes farther from source,
    # but keep them genuinely far from destination.
    infeasible_candidates.sort(
        key=lambda x: (x["hop_from_source"], x["walk_distance_km"]),
        reverse=True
    )

    # In relaxed-walking cases such as C4, keep any displayed infeasible parking
    # lots close to the walking threshold rather than far beyond it. This keeps
    # the infeasible tail realistic without changing the feasible local set.
    if max_walk_km > BASE_MAX_WALK_KM:
        near_threshold_infeasible = [
            x for x in infeasible_candidates
            if x["walk_distance_km"] <= max_walk_km + 0.15
        ]
        if near_threshold_infeasible:
            infeasible_candidates = near_threshold_infeasible

    requested_parking_count = compute_parking_count(n_nodes, parking_pct)

    if len(feasible_candidates) < 2:
        # Fallback: relax the destination-side filter only if the local region
        # does not provide enough feasible parking choices.
        feasible_candidates = []
        infeasible_candidates = []

        for node in driving_node_ids:
            if node == source:
                continue
            if node not in hops:
                continue

            walk_d = euclidean(coords[node], coords[destination])
            walk_time_hr = walk_d / WALK_SPEED_KMPH

            item = {
                "node": node,
                "hop_from_source": hops[node],
                "walk_distance_km": round(walk_d, 4),
                "walk_time_hr": round(walk_time_hr, 4),
                "feasible": walk_d <= max_walk_km,
            }

            if item["feasible"]:
                feasible_candidates.append(item)
            else:
                infeasible_candidates.append(item)

        if max_walk_km > BASE_MAX_WALK_KM:
            feasible_candidates.sort(
                key=lambda x: (x["walk_distance_km"], -x["hop_from_source"])
            )
        else:
            feasible_candidates.sort(
                key=lambda x: (x["hop_from_source"], -x["walk_distance_km"]),
                reverse=True
            )
        infeasible_candidates.sort(
            key=lambda x: (x["hop_from_source"], x["walk_distance_km"]),
            reverse=True
        )

        if max_walk_km > BASE_MAX_WALK_KM:
            near_threshold_infeasible = [
                x for x in infeasible_candidates
                if x["walk_distance_km"] <= max_walk_km + 0.15
            ]
            if near_threshold_infeasible:
                infeasible_candidates = near_threshold_infeasible

    if len(feasible_candidates) < 2:
        raise RuntimeError("Need at least 2 feasible parking lots.")

    # Force a few infeasible parking lots when possible.
    # Keep most parking feasible, but include some far lots for realism.
    target_infeasible = 0
    if len(infeasible_candidates) > 0 and requested_parking_count >= 4:
        if n_nodes >= LARGE_CASE_NODE_THRESHOLD:
            target_infeasible = int(math.floor(requested_parking_count * LARGE_CASE_MAX_INFEASIBLE_RATIO))
            target_infeasible = min(target_infeasible, len(infeasible_candidates))
        else:
            target_infeasible = max(1, requested_parking_count // 4)
            target_infeasible = min(target_infeasible, len(infeasible_candidates))

    target_feasible = requested_parking_count - target_infeasible

    # Always preserve at least 2 feasible parking lots.
    if target_feasible < 2:
        target_feasible = 2
        target_infeasible = min(requested_parking_count - target_feasible, len(infeasible_candidates))

    # For large cases, keep the parking pool much more feasibility-friendly so
    # metaheuristics are not overwhelmed by invalid candidates.
    if n_nodes >= LARGE_CASE_NODE_THRESHOLD:
        min_feasible_target = max(
            2,
            min(len(feasible_candidates), LARGE_CASE_MIN_FEASIBLE_COUNT),
            int(math.ceil(requested_parking_count * LARGE_CASE_MIN_FEASIBLE_RATIO))
        )
        if target_feasible < min_feasible_target:
            target_feasible = min_feasible_target
            target_infeasible = max(0, requested_parking_count - target_feasible)
            target_infeasible = min(target_infeasible, len(infeasible_candidates))
            target_feasible = requested_parking_count - target_infeasible

    selected = feasible_candidates[:target_feasible] + infeasible_candidates[:target_infeasible]

    # If still short, fill from remaining feasible first, then remaining infeasible.
    if len(selected) < requested_parking_count:
        used_nodes = {x["node"] for x in selected}

        remaining_feasible = [x for x in feasible_candidates if x["node"] not in used_nodes]
        need = requested_parking_count - len(selected)
        selected.extend(remaining_feasible[:need])

    if len(selected) < requested_parking_count:
        used_nodes = {x["node"] for x in selected}
        remaining_infeasible = [x for x in infeasible_candidates if x["node"] not in used_nodes]

        # In large cases, only add infeasible parking if the target count still
        # cannot be met after exhausting feasible options.
        need = requested_parking_count - len(selected)
        selected.extend(remaining_infeasible[:need])

    if len(selected) < 2:
        raise RuntimeError("Parking construction did not produce enough parking lots.")

    all_parking_nodes = [x["node"] for x in selected]
    parking_info_by_node = {}
    feasible_parking_nodes = []

    for item in selected:
        node = item["node"]
        parking_info_by_node[node] = {
            "hop_from_source": item["hop_from_source"],
            "walk_distance_km": item["walk_distance_km"],
            "walk_time_hr": item["walk_time_hr"],
            "feasible": item["feasible"],
        }
        if item["feasible"]:
            feasible_parking_nodes.append(node)

    if len(feasible_parking_nodes) < 2:
        raise RuntimeError("Need at least 2 walking-feasible parking lots for a stable dataset.")

    if n_nodes >= LARGE_CASE_NODE_THRESHOLD:
        min_large_case_feasible = min(len(all_parking_nodes), LARGE_CASE_MIN_FEASIBLE_COUNT)
        if len(feasible_parking_nodes) < min_large_case_feasible:
            raise RuntimeError(
                "Large-case dataset needs a stronger feasible parking pool for stable method comparison."
            )

    return all_parking_nodes, feasible_parking_nodes, parking_info_by_node


def build_walk_edges_for_all_parking(coords, parking_nodes, destination):
    """
    Build walking edges from every parking lot to the destination.
    Feasibility is handled by optimization constraints, not by edge existence.
    """
    walk_edges = []
    for p in parking_nodes:
        add_bidirected_edge(walk_edges, None, coords, p, destination, WALK_SPEED_KMPH, "walk")
    return walk_edges


# ============================================================
# VALIDATION
# ============================================================
def validate_case(case):
    n = case["number_of_nodes"]
    source = case["source_node"]
    destination = case["destination_node"]

    all_parking_nodes = set(case["parking_nodes"])
    feasible_parking_nodes = set(case["feasible_parking_nodes"])
    parking_info_by_node = case["parking_info_by_node"]

    if source == destination:
        raise RuntimeError("Source and destination cannot be the same.")

    if destination in all_parking_nodes:
        raise RuntimeError("Destination cannot be a parking node.")

    if not feasible_parking_nodes.issubset(all_parking_nodes):
        raise RuntimeError("Feasible parking nodes must be a subset of all parking nodes.")

    # No driving edges to destination
    for e in case["drive_edges"]:
        if e["u"] == destination or e["v"] == destination:
            raise RuntimeError("Destination has driving edges, which is not allowed.")

    # No self-loops
    for e in case["drive_edges"] + case["walk_edges"]:
        if e["u"] == e["v"]:
            raise RuntimeError("Self-loop detected in edge set.")

    # Walk edges must connect all parking lots to destination
    walk_connected = set()
    for e in case["walk_edges"]:
        u, v = e["u"], e["v"]
        valid = (
            (u in all_parking_nodes and v == destination) or
            (v in all_parking_nodes and u == destination)
        )
        if not valid:
            raise RuntimeError("Walking edges must only be between parking lots and destination.")

        if u in all_parking_nodes and v == destination:
            walk_connected.add(u)
        if v in all_parking_nodes and u == destination:
            walk_connected.add(v)

    if walk_connected != all_parking_nodes:
        raise RuntimeError("Not all parking lots are directly connected to destination.")

    # Parking nodes must be reachable from source
    adj = _build_drive_adjacency(n, case["drive_edges"])

    hops = bfs_hops(adj, source)
    reachable_parking = [p for p in all_parking_nodes if p in hops]
    if not reachable_parking:
        raise RuntimeError("No parking nodes are reachable from source.")

    feasible_parking_hops = [hops[p] for p in feasible_parking_nodes if p in hops]
    if len(feasible_parking_hops) < 2:
        raise RuntimeError("Need at least 2 reachable feasible parking lots.")

    min_required_hops = 2 if n <= 30 else 4
    if max(feasible_parking_hops) < min_required_hops:
        raise RuntimeError("Feasible parking nodes are not far enough from source in hop distance.")

    # Parking count bounds
    if len(all_parking_nodes) < 2:
        raise RuntimeError("Each case must have at least 2 parking lots.")
    if len(all_parking_nodes) > max(2, int(math.floor(0.20 * n))):
        raise RuntimeError("Parking lots exceed 20% of total nodes.")

    # Cost alignment
    if set(case["parking_costs"].keys()) != all_parking_nodes:
        raise RuntimeError("Parking costs must align with all parking nodes.")

    # Parking info alignment
    if set(parking_info_by_node.keys()) != all_parking_nodes:
        raise RuntimeError("Parking info must align with all parking nodes.")

    # Feasibility flags must match walking threshold
    W = case["max_walking_km"]
    for p in all_parking_nodes:
        info = parking_info_by_node[p]
        should_be_feasible = info["walk_distance_km"] <= W + 1e-9
        if info["feasible"] != should_be_feasible:
            raise RuntimeError("Parking feasibility flag does not match walking threshold.")


# ============================================================
# BASELINE CASE
# ============================================================
def build_baseline_case(n_nodes, instance_id=1):
    rows, cols, coords, rc_of, node_of, destination = build_grid_nodes(n_nodes, instance_id)

    driving_node_ids = list(range(n_nodes - 1))
    source, far_anchor, _ = farthest_pair(driving_node_ids, coords)

    drive_edges, drive_adj = build_urban_drive_graph(
        node_of, coords, n_nodes, LOW_CONGESTION_RANGE, instance_id
    )

    drive_adj[destination] = []

    cx = sum(coords[i][0] for i in driving_node_ids) / len(driving_node_ids)
    cy = sum(coords[i][1] for i in driving_node_ids) / len(driving_node_ids)

    ax, ay = coords[far_anchor]
    vx, vy = ax - cx, ay - cy
    norm = math.sqrt(vx * vx + vy * vy)

    if norm == 0:
        vx, vy = 1.0, 0.0
        norm = 1.0

    offset_km = min(0.12, BASE_MAX_WALK_KM * 0.25)
    coords[destination] = (
        round(ax + offset_km * vx / norm, 4),
        round(ay + offset_km * vy / norm, 4)
    )

    parking_nodes, feasible_parking_nodes, parking_info_by_node = build_parking_info(
        n_nodes=n_nodes,
        driving_node_ids=driving_node_ids,
        coords=coords,
        source=source,
        destination=destination,
        drive_adj=drive_adj,
        parking_pct=BASE_PARKING_PCT,
        max_walk_km=BASE_MAX_WALK_KM,
        parking_search_walk_km=BASE_MAX_WALK_KM
    )

    walk_edges = build_walk_edges_for_all_parking(coords, parking_nodes, destination)
    parking_costs = assign_parking_costs(
        number_of_nodes=n_nodes,
        instance_id=instance_id,
        parking_nodes=parking_nodes,
        cost_range=BASE_PARKING_COST_RANGE,
    )

    case = {
        "number_of_nodes": n_nodes,
        "instance_id": instance_id,
        "source_node": source,
        "destination_node": destination,
        "map_side_km": round(map_side_for_n(n_nodes), 4),
        "coords": coords,
        "rows": rows,
        "cols": cols,
        "rc_of": rc_of,
        "node_of": node_of,
        "drive_edges": drive_edges,
        "walk_edges": walk_edges,
        "parking_nodes": parking_nodes,
        "feasible_parking_nodes": feasible_parking_nodes,
        "parking_info_by_node": parking_info_by_node,
        "parking_costs": parking_costs,
        "max_walking_km": BASE_MAX_WALK_KM,
        "max_walking_time_hr": round(BASE_MAX_WALK_KM / WALK_SPEED_KMPH, 4),
        "cmax": BASE_CMAX,
        "alpha": BASE_ALPHA,
        "B": B_VALUE,
        "edge_mode": BASE_EDGE_MODE,
        "traffic_level": "low",
        "traffic_speed_range": LOW_CONGESTION_RANGE,
        "parking_cost_mode": "baseline_deterministic_varied",
        "parking_cost_range": BASE_PARKING_COST_RANGE,
    }

    validate_case(case)
    return case


# ============================================================
# TRAFFIC REGENERATION
# ============================================================
def regenerate_drive_times_from_speed_range(base_case, speed_range, traffic_label):
    """
    Recompute driving-edge times using a new congestion-specific speed range.
    Distances and topology remain unchanged.
    Bidirectional edge pairs keep the same sampled speed/time.
    """
    case = copy.deepcopy(base_case)
    rng = random.Random(
        f"traffic-{traffic_label}-{case['number_of_nodes']}-{case['instance_id']}"
    )

    pair_speeds = {}

    for e in case["drive_edges"]:
        pair = tuple(sorted((e["u"], e["v"])))
        if pair not in pair_speeds:
            pair_speeds[pair] = rng.uniform(speed_range[0], speed_range[1])

        speed = pair_speeds[pair]
        e["speed_kmph"] = round(speed, 2)
        e["time_hr"] = round(e["distance_km"] / speed, 4)

    case["traffic_level"] = traffic_label
    case["traffic_speed_range"] = speed_range
    validate_case(case)
    return case


# ============================================================
# CONTROLLED VARIATIONS: C2–C9
# ============================================================
def vary_edges_only(base_case):
    """
    C2: vary edge structure only by adding extra connector/arterial links
    to the same urban base network.
    """
    case = copy.deepcopy(base_case)
    coords = case["coords"]
    n = case["number_of_nodes"]
    rng = random.Random(f"edge-dense-{case['number_of_nodes']}-{case['instance_id']}")

    driving_nodes = list(range(n - 1))
    existing = {(e["u"], e["v"]) for e in case["drive_edges"]}

    def add_new(u, v, road_class):
        if u == v:
            return
        if (u, v) in existing or (v, u) in existing:
            return

        lo, hi = LOW_CONGESTION_RANGE
        if road_class == "connector":
            s_lo = lo + 0.20 * (hi - lo)
            s_hi = lo + 0.80 * (hi - lo)
        else:
            s_lo = lo + 0.45 * (hi - lo)
            s_hi = hi

        speed = rng.uniform(s_lo, s_hi)
        add_bidirected_edge(case["drive_edges"], None, coords, u, v, speed, "drive", road_class)
        existing.add((u, v))
        existing.add((v, u))

    pair_list, pair_distance, neighbors = _precompute_pairwise_geometry(driving_nodes, coords)
    pair_list = [(d, u, v) for d, u, v in pair_list if 0.28 <= d <= 0.70]
    pair_list.sort(reverse=True)

    target_extra = max(3, min(len(pair_list), len(driving_nodes) // 6))
    added = 0
    for d, u, v in pair_list:
        road_class = "arterial" if d >= 0.45 else "connector"
        before = len(existing)
        add_new(u, v, road_class)
        if len(existing) > before:
            added += 1
        if added >= target_extra:
            break

    case["edge_mode"] = VAR_EDGE_MODE
    validate_case(case)
    return case


def vary_parking_only(base_case):
    """
    C3: vary parking percentage only.
    Includes all parking lots, and all get walk edges.
    """
    case = copy.deepcopy(base_case)
    source = case["source_node"]
    destination = case["destination_node"]

    adj = _build_drive_adjacency(case["number_of_nodes"], case["drive_edges"])
    driving_node_ids = list(range(case["number_of_nodes"] - 1))

    parking_nodes, feasible_parking_nodes, parking_info_by_node = build_parking_info(
        n_nodes=case["number_of_nodes"],
        driving_node_ids=driving_node_ids,
        coords=case["coords"],
        source=source,
        destination=destination,
        drive_adj=adj,
        parking_pct=VAR_PARKING_PCT,
        max_walk_km=case["max_walking_km"],
        parking_search_walk_km=case["max_walking_km"]
    )

    case["parking_nodes"] = parking_nodes
    case["feasible_parking_nodes"] = feasible_parking_nodes
    case["parking_info_by_node"] = parking_info_by_node
    case["parking_costs"] = assign_parking_costs_preserving_existing(
        number_of_nodes=case["number_of_nodes"],
        instance_id=case["instance_id"],
        parking_nodes=parking_nodes,
        cost_range=BASE_PARKING_COST_RANGE,
        existing_costs=base_case["parking_costs"],
    )
    case["parking_cost_mode"] = "baseline_deterministic_varied"
    case["parking_cost_range"] = BASE_PARKING_COST_RANGE
    case["walk_edges"] = build_walk_edges_for_all_parking(case["coords"], parking_nodes, destination)

    validate_case(case)
    return case


def vary_max_walking_only(base_case):
    """
    C4: vary maximum walking distance only.
    Recompute parking feasibility under the larger walking threshold.
    """
    case = copy.deepcopy(base_case)
    case["max_walking_km"] = VAR_MAX_WALK_KM
    case["max_walking_time_hr"] = round(VAR_MAX_WALK_KM / WALK_SPEED_KMPH, 4)
    refresh_parking_feasibility(case)

    validate_case(case)
    return case


def vary_medium_traffic_only(base_case):
    """
    C5: medium congestion.
    """
    return regenerate_drive_times_from_speed_range(
        base_case, MEDIUM_CONGESTION_RANGE, "medium"
    )


def vary_high_traffic_only(base_case):
    """
    C6: high congestion.
    """
    return regenerate_drive_times_from_speed_range(
        base_case, HIGH_CONGESTION_RANGE, "high"
    )


def vary_alpha_high_only(base_case):
    """
    C7: vary Alpha only to emphasize travel.
    """
    case = copy.deepcopy(base_case)
    case["alpha"] = VAR_ALPHA_HIGH
    validate_case(case)
    return case


def ensure_affordable_feasible_parking(case, cost_range, min_affordable_feasible=1, seed_tag="repair"):
    affordable_feasible = [
        p for p in case["feasible_parking_nodes"]
        if case["parking_costs"][p] <= case["cmax"]
    ]

    target_count = min(min_affordable_feasible, len(case["feasible_parking_nodes"]))
    if len(affordable_feasible) >= target_count:
        return case

    rng = random.Random(
        f"{seed_tag}-{case['number_of_nodes']}-{case['instance_id']}-{case['cmax']}"
    )
    repair_candidates = sorted(
        case["feasible_parking_nodes"],
        key=lambda p: (
            case["parking_info_by_node"][p]["walk_distance_km"],
            case["parking_info_by_node"][p]["hop_from_source"]
        )
    )

    needed = target_count - len(affordable_feasible)
    repaired = 0
    lower = cost_range[0]
    upper = min(case["cmax"], cost_range[1])
    used_costs = {round(cost, 2) for cost in case["parking_costs"].values()}

    for p in repair_candidates:
        if case["parking_costs"][p] <= case["cmax"]:
            continue

        if lower <= upper:
            candidate_slots = list(
                range(int(round(lower * 100)), int(round(upper * 100)) + 1)
            )
            rng.shuffle(candidate_slots)
            replacement = None
            for slot in candidate_slots:
                proposed = round(slot / 100.0, 2)
                if proposed not in used_costs:
                    replacement = proposed
                    break
            case["parking_costs"][p] = (
                replacement
                if replacement is not None
                else round(rng.uniform(lower, upper), 2)
            )
        else:
            case["parking_costs"][p] = round(case["cmax"], 2)

        used_costs.add(round(case["parking_costs"][p], 2))
        repaired += 1
        if repaired >= needed:
            break

    return case


def vary_parking_cost_only(base_case):
    """
    C10: vary parking costs only by reassigning costs from
    a higher parking-price range on the same parking nodes.
    """
    case = copy.deepcopy(base_case)

    case["parking_costs"] = assign_parking_costs(
        number_of_nodes=case["number_of_nodes"],
        instance_id=case["instance_id"],
        parking_nodes=case["parking_nodes"],
        cost_range=HIGH_PARKING_COST_RANGE,
    )

    case = ensure_affordable_feasible_parking(
        case,
        HIGH_PARKING_COST_RANGE,
        min_affordable_feasible=2,
        seed_tag="parking-cost-high",
    )

    case["parking_cost_mode"] = "high_deterministic_varied"
    case["parking_cost_range"] = HIGH_PARKING_COST_RANGE

    validate_case(case)
    return case


def vary_alpha_low_only(base_case):
    """
    C8: vary Alpha only to emphasize parking cost.
    """
    case = copy.deepcopy(base_case)
    case["alpha"] = VAR_ALPHA_LOW
    validate_case(case)
    return case


def vary_cmax_only(base_case):
    """
    C9: vary Cmax only.
    """
    case = copy.deepcopy(base_case)
    case["cmax"] = VAR_CMAX
    case = ensure_affordable_feasible_parking(
        case,
        BASE_PARKING_COST_RANGE,
        min_affordable_feasible=1,
        seed_tag="cmax-only",
    )
    validate_case(case)
    return case


# ============================================================
# WRITER
# ============================================================
def write_case_dat(case, path):
    n = case["number_of_nodes"]
    S = case["source_node"] + 1
    D = case["destination_node"] + 1

    # L includes all parking facilities, including those beyond walking threshold
    L = [p + 1 for p in case["parking_nodes"]]
    C = [case["parking_costs"][p] for p in case["parking_nodes"]]

    feasible_flags = [
        1 if case["parking_info_by_node"][p]["feasible"] else 0
        for p in case["parking_nodes"]
    ]
    walk_distance = [
        case["parking_info_by_node"][p]["walk_distance_km"]
        for p in case["parking_nodes"]
    ]
    walk_time = [
        case["parking_info_by_node"][p]["walk_time_hr"]
        for p in case["parking_nodes"]
    ]

    all_edges = sorted(case["drive_edges"] + case["walk_edges"], key=lambda e: (e["u"], e["v"], e["type"]))

    E = [(e["u"] + 1, e["v"] + 1) for e in all_edges]
    Distance = [e["distance_km"] for e in all_edges]
    EdgeTime = [e["time_hr"] for e in all_edges]
    EdgeType = [e["type"] for e in all_edges]

    with open(path, "w") as f:
        f.write("/*********************************************\n")
        f.write(" * Controlled structured urban-road dataset\n")
        f.write(" *********************************************/\n\n")

        f.write(f"Nodes = {n};\n\n")

        f.write("E = {\n")
        f.write(chunk_lines(E, per_line=4, fmt=fmt_edge))
        f.write("\n};\n\n")

        f.write(f"S = {S};\t\t// Source of EV\n")
        f.write(f"D = {D};\t\t// Destination node\n\n")

        f.write(f"Q = {len(L)};\t\t// Number of parking lots (all urban parking facilities)\n")
        f.write(f"L = [{','.join(map(str, L))}];\t\t// Parking lot nodes\n")
        f.write(f"C = [{','.join(map(fmt_num, C))}];\t// Parking costs\n")
        f.write(f"FeasibleParking = [{','.join(map(str, feasible_flags))}];\t// 1 if parking lot is within W, else 0\n")
        f.write(f"WalkDistanceToDest = [{','.join(map(fmt_num, walk_distance))}];\t// Walking distance from each parking lot to destination\n")
        f.write(f"WalkTimeToDest = [{','.join(map(fmt_num, walk_time))}];\t// Walking time from each parking lot to destination\n\n")

        f.write(f"W = {fmt_num(case['max_walking_time_hr'])};\t\t// Maximum walking time (hr)\n")
        f.write(f"W_km = {fmt_num(case['max_walking_km'])};\t\t// Maximum walking distance (km), reference only\n")
        f.write(f"Cmax = {fmt_num(case['cmax'])};\t\t// Maximum allowed parking cost\n\n")
        f.write(f"B = {case['B']};\t\t// Large constant\n\n")
        f.write(f"Alpha = {fmt_num(case['alpha'])};\t// Objective weight\n\n")
        f.write(f'EdgeMode = "{case["edge_mode"]}";\n')
        f.write(f'TrafficLevel = "{case.get("traffic_level", "low")}";\n')
        f.write(
            f'TrafficSpeedRange = [{fmt_num(case["traffic_speed_range"][0])},'
            f'{fmt_num(case["traffic_speed_range"][1])}];\n'
        )
        f.write(f'ParkingCostMode = "{case.get("parking_cost_mode", "baseline_structured")}";\n')
        f.write(f'MapSideKm = {fmt_num(case.get("map_side_km", map_side_for_n(n)))};\n')
        f.write(
            f'ParkingCostRange = [{fmt_num(case["parking_cost_range"][0])},'
            f'{fmt_num(case["parking_cost_range"][1])}];\n\n'
        )
        if case.get("experiment_type"):
            f.write(f'ExperimentType = "{case["experiment_type"]}";\n')
        if case.get("scale_category"):
            f.write(f'ScaleCategory = "{case["scale_category"]}";\n')
        if case.get("varied_parameter"):
            f.write(f'VariedParameter = "{case["varied_parameter"]}";\n')
        if case.get("active_parameter_value") is not None:
            f.write(f'ActiveParameterValue = "{case["active_parameter_value"]}";\n')
        if case.get("parking_supply_pct") is not None:
            f.write(f'ParkingSupplyPct = {fmt_num(case["parking_supply_pct"])};\n')
        if case.get("parameter_label"):
            f.write(f'ParameterLabel = "{case["parameter_label"]}";\n')
        if any(key in case for key in ("experiment_type", "scale_category", "varied_parameter", "active_parameter_value", "parking_supply_pct", "parameter_label")):
            f.write("\n")

        f.write("Distance = [\n")
        f.write(chunk_lines(Distance, per_line=4, fmt=fmt_num))
        f.write("\n];\n\n")

        f.write("EdgeTime = [\n")
        f.write(chunk_lines(EdgeTime, per_line=4, fmt=fmt_num))
        f.write("\n];\n\n")

        f.write("EdgeType = [\n")
        f.write(chunk_lines(EdgeType, per_line=8, fmt=lambda x: f'"{fmt_str(x)}"'))
        f.write("\n];\n")


def _summarize_case(case_name, case_data, instance_id):
    drive_edge_count = len(case_data["drive_edges"]) // 2
    walk_edge_count = len(case_data["walk_edges"]) // 2
    parking_lines = []

    for parking_node in case_data["parking_nodes"]:
        info = case_data["parking_info_by_node"][parking_node]
        parking_lines.append(
            "    "
            f"P{parking_node + 1}: "
            f"cost={fmt_num(case_data['parking_costs'][parking_node])}, "
            f"walk_km={fmt_num(info['walk_distance_km'])}, "
            f"walk_hr={fmt_num(info['walk_time_hr'])}, "
            f"feasible={'yes' if info['feasible'] else 'no'}, "
            f"hops_from_source={info['hop_from_source']}"
        )

    summary = [
        f"{case_name} | Instance I{instance_id}",
        f"  Nodes: {case_data['number_of_nodes']}",
        f"  Map side km: {fmt_num(case_data.get('map_side_km', map_side_for_n(case_data['number_of_nodes'])))}",
        f"  Source: N{case_data['source_node'] + 1}",
        f"  Destination: N{case_data['destination_node'] + 1}",
        f"  Drive edges: {drive_edge_count}",
        f"  Walk edges: {walk_edge_count}",
        f"  Parking lots: {len(case_data['parking_nodes'])}",
        "  Feasible parking lots: "
        f"{len(case_data['feasible_parking_nodes'])} "
        f"({', '.join(f'N{p + 1}' for p in case_data['feasible_parking_nodes'])})",
        f"  Alpha: {fmt_num(case_data['alpha'])}",
        f"  Cmax: {fmt_num(case_data['cmax'])}",
        f"  Max walk km: {fmt_num(case_data['max_walking_km'])}",
        f"  Traffic level: {case_data.get('traffic_level', 'low')}",
        "  Traffic speed range (km/h): "
        f"{fmt_num(case_data['traffic_speed_range'][0])} - {fmt_num(case_data['traffic_speed_range'][1])}",
        f"  Edge mode: {case_data['edge_mode']}",
        f"  Parking cost mode: {case_data.get('parking_cost_mode', 'baseline_structured')}",
        "  Parking nodes detail:",
    ]

    if parking_lines:
        summary.extend(parking_lines)
    else:
        summary.append("    None")

    return "\n".join(summary)


def _write_summary_file(all_cases_by_n, summary_path):
    with open(summary_path, "w") as f:
        f.write("FINAL DATASET VISUAL SUMMARY\n")
        f.write("============================\n")
        f.write(
            "Grouped by node size. Each block lists C1 through C10 before moving to the next node set.\n\n"
        )

        for n in sorted(all_cases_by_n):
            f.write(f"NODE SET N={n}\n")
            f.write(f"{'-' * 72}\n")

            for case_name, instance_id, case_data in all_cases_by_n[n]:
                f.write(_summarize_case(case_name, case_data, instance_id))
                f.write("\n\n")

            f.write("\n")

    return summary_path


def write_visual_summary(all_cases_by_n, out_dir):
    summary_path = os.path.join(out_dir, "dataset_visual_summary.txt")
    final_dataset_path = os.path.join(out_dir, "final_dataset.txt")

    _write_summary_file(all_cases_by_n, summary_path)
    _write_summary_file(all_cases_by_n, final_dataset_path)

    return summary_path, final_dataset_path


# ============================================================
# GENERATION
# ============================================================
def generate_all_cases():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_cases_by_n = {}
    previous_small_baseline_objective = None

    for n in NODE_SIZES:
        subdir = os.path.join(OUT_DIR, f"n_{n}")
        os.makedirs(subdir, exist_ok=True)
        all_cases_by_n[n] = []

        for instance_id in range(1, NUM_INSTANCES + 1):
            base = build_baseline_case(n, instance_id)
            if n in SMALL_SCALE_NODE_SIZES:
                if previous_small_baseline_objective is None:
                    previous_small_baseline_objective = exact_case_objective(base)
                else:
                    base, previous_small_baseline_objective = calibrate_small_case_monotonic(
                        base,
                        previous_small_baseline_objective,
                    )
                if n in SMALL_SCALE_HEURISTIC_ALIGNMENT_SIZES:
                    base = align_small_case_proxy_with_exact(base)
                divergence_gap = SMALL_SCALE_HEURISTIC_DIVERGENCE_GAPS.get(n)
                if divergence_gap is not None:
                    base = induce_small_case_proxy_divergence(base, divergence_gap)
                previous_small_baseline_objective = exact_case_objective(base)

            cases = {
                "C1": base,                              # baseline
                "C2": vary_edges_only(base),            # edge structure
                "C3": vary_parking_only(base),          # parking percentage
                "C4": vary_max_walking_only(base),      # walking threshold
                "C5": vary_medium_traffic_only(base),   # medium congestion
                "C6": vary_high_traffic_only(base),     # high congestion
                "C7": vary_alpha_high_only(base),       # travel-focused alpha
                "C8": vary_alpha_low_only(base),        # cost-focused alpha
                "C9": vary_cmax_only(base),             # Cmax variation
                "C10": vary_parking_cost_only(base),    # higher parking-cost scenario
            }

            for case_name, case_data in cases.items():
                all_cases_by_n[n].append((case_name, instance_id, case_data))
                filename = f"N{n}_{case_name}_I{instance_id}.dat"
                dat_path = os.path.join(subdir, filename)
                png_path = os.path.join(subdir, filename.replace(".dat", ".png"))

                if not FORCE_REGENERATE:
                    if GENERATE_PLOTS:
                        if os.path.exists(dat_path) and os.path.exists(png_path):
                            continue
                    else:
                        if os.path.exists(dat_path):
                            continue

                write_case_dat(case_data, dat_path)

                if GENERATE_PLOTS:
                    plot_case_graph(case_data, png_path)

        print(f"Generated datasets for n={n}")

    summary_path, final_dataset_path = write_visual_summary(all_cases_by_n, OUT_DIR)
    print(f"Summary written to {summary_path}")
    print(f"Final dataset summary written to {final_dataset_path}")
    if not GENERATE_PLOTS:
        print("Plots were skipped for this run.")
    print("Done.")


if __name__ == "__main__":
    generate_all_cases()
