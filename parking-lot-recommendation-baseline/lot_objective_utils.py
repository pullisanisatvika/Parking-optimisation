from __future__ import annotations

import heapq
import math
import re
from typing import Any


TRAVEL_WEIGHT = 0.9
COST_WEIGHT = 0.1
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def _build_adjacency(data: dict[str, Any]) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = {}
    for (u, v), edge_time in zip(data["E"], data["EdgeTime"]):
        adj.setdefault(int(u), []).append((int(v), float(edge_time)))
    return adj


def _dijkstra_from_source(adj: dict[int, list[tuple[int, float]]], source: int) -> dict[int, float]:
    dist: dict[int, float] = {int(source): 0.0}
    pq: list[tuple[float, int]] = [(0.0, int(source))]
    while pq:
        current_dist, node = heapq.heappop(pq)
        if current_dist > dist.get(node, math.inf):
            continue
        for nxt, edge_time in adj.get(node, []):
            candidate = current_dist + float(edge_time)
            if candidate < dist.get(nxt, math.inf):
                dist[nxt] = candidate
                heapq.heappush(pq, (candidate, nxt))
    return dist


def build_lot_normalization_context(data: dict[str, Any]) -> dict[str, float]:
    adj = _build_adjacency(data)
    dist_from_source = _dijkstra_from_source(adj, int(data["S"]))
    alpha = float(data.get("w", data.get("Alpha", TRAVEL_WEIGHT)))
    alpha = min(max(alpha, 0.0), 1.0)

    walk_limit = float(data.get("W", math.inf))
    budget_limit = float(data.get("Cmax", math.inf))
    lots = [int(value) for value in data["L"]]
    costs = [float(value) for value in data["C"]]
    walk_times = [float(value) for value in data.get("WalkTimeToDest", [])]

    feasible_travel_times: list[float] = []
    feasible_costs: list[float] = []
    for lot, cost, walk_time in zip(lots, costs, walk_times):
        drive_time = dist_from_source.get(lot, math.inf)
        if not math.isfinite(drive_time) or not math.isfinite(walk_time):
            continue
        if walk_time > walk_limit + 1e-12 or cost > budget_limit + 1e-12:
            continue
        feasible_travel_times.append(drive_time + walk_time)
        feasible_costs.append(cost)

    fallback_travel = [value for value in dist_from_source.values() if math.isfinite(value)]
    fallback_walk = [value for value in walk_times if math.isfinite(value)]
    fallback_costs = [value for value in costs if math.isfinite(value)]
    if math.isfinite(budget_limit):
        fallback_costs.append(budget_limit)

    travel_scale = max(feasible_travel_times, default=0.0)
    if travel_scale <= 0.0:
        travel_scale = max(fallback_travel, default=0.0) + max(fallback_walk, default=0.0)
    if travel_scale <= 0.0:
        travel_scale = 1.0

    cost_scale = max(feasible_costs, default=0.0)
    if cost_scale <= 0.0:
        cost_scale = max(fallback_costs, default=0.0)
    if cost_scale <= 0.0:
        cost_scale = 1.0

    return {
        "travel_scale": float(travel_scale),
        "cost_scale": float(cost_scale),
        "travel_weight": alpha,
        "cost_weight": 1.0 - alpha,
    }


def parse_lot_dat_for_normalization(dat_text: str) -> dict[str, Any]:
    def get_int(name: str) -> int:
        match = re.search(rf"\b{re.escape(name)}\s*=\s*(\d+)\s*;", dat_text)
        if not match:
            raise ValueError(f"Missing integer: {name}")
        return int(match.group(1))

    def get_float(name: str, default: float | None = None, *, required: bool = True) -> float | None:
        match = re.search(rf"\b{re.escape(name)}\s*=\s*({FLOAT_RE})\s*;", dat_text)
        if match:
            return float(match.group(1))
        if not required:
            return default
        if default is not None:
            return default
        raise ValueError(f"Missing float: {name}")

    def get_number_list(name: str) -> list[float]:
        match = re.search(rf"\b{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", dat_text, flags=re.DOTALL)
        if not match:
            return []
        return [float(token) for token in re.findall(FLOAT_RE, match.group(1))]

    edge_match = re.search(r"\bE\s*=\s*\{(.*?)\};", dat_text, flags=re.DOTALL)
    if not edge_match:
        raise ValueError("Missing edge set E")
    edges = [(int(a), int(b)) for a, b in re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", edge_match.group(1))]

    lots = [int(token) for token in re.findall(r"\d+", re.search(r"\bL\s*=\s*\[([^\]]+)\]\s*;", dat_text).group(1))]
    costs = get_number_list("C")
    edge_times = get_number_list("EdgeTime")
    walk_times = get_number_list("WalkTimeToDest")

    walk_limit = get_float("W", None, required=False)
    if walk_limit is None:
        walk_limit = get_float("W_km", math.inf)

    return {
        "S": get_int("S"),
        "L": lots,
        "C": costs,
        "E": edges,
        "EdgeTime": edge_times,
        "WalkTimeToDest": walk_times,
        "W": walk_limit,
        "Cmax": get_float("Cmax", math.inf),
    }


def normalized_lot_objective(
    travel_time_hr: float,
    parking_cost: float,
    *,
    context: dict[str, float],
) -> float:
    travel_scale = max(float(context["travel_scale"]), 1e-12)
    cost_scale = max(float(context["cost_scale"]), 1e-12)
    return (
        float(context.get("travel_weight", TRAVEL_WEIGHT)) * (float(travel_time_hr) / travel_scale)
        + float(context.get("cost_weight", COST_WEIGHT)) * (float(parking_cost) / cost_scale)
    )


def raw_lot_objective(
    drive_time_hr: float,
    walk_time_hr: float,
    parking_cost: float,
    *,
    weight_w: float,
) -> float:
    travel_weight = min(max(float(weight_w), 0.0), 1.0)
    cost_weight = 1.0 - travel_weight
    return travel_weight * (float(drive_time_hr) + float(walk_time_hr)) + cost_weight * float(parking_cost)
