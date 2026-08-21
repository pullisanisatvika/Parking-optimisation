from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path
from typing import Any


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def scalar_from_text(text: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(?:\"([^\"]*)\"|([^;]+))\s*;", text)
    if not match:
        return ""
    value = match.group(1) if match.group(1) is not None else match.group(2)
    return value.strip()


def parse_string_list(text: str, name: str) -> list[str]:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", text, flags=re.DOTALL)
    if not match:
        return []
    return re.findall(r'"([^"]*)"', match.group(1))


def parse_number_list(text: str, name: str, cast=float) -> list[Any]:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", text, flags=re.DOTALL)
    if not match:
        return []
    tokens = re.findall(FLOAT_RE if cast is float else r"\d+", match.group(1))
    return [cast(token) for token in tokens]


def parse_opl_dat(dat_text: str) -> dict[str, Any]:
    def get_int(name: str, default: int | None = None) -> int:
        match = re.search(rf"\b{re.escape(name)}\s*=\s*(\d+)\s*;", dat_text)
        if match:
            return int(match.group(1))
        if default is not None:
            return default
        raise ValueError(f"Missing integer: {name}")

    def get_float(name: str, default: float | None = None) -> float:
        match = re.search(rf"\b{re.escape(name)}\s*=\s*({FLOAT_RE})\s*;", dat_text)
        if match:
            return float(match.group(1))
        if default is not None:
            return default
        raise ValueError(f"Missing float: {name}")

    nodes = get_int("Nodes")
    source = get_int("S")
    destination = get_int("D")
    lots = parse_number_list(dat_text, "L", int)
    costs = parse_number_list(dat_text, "C", float)
    q = get_int("Q", len(lots))
    if len(lots) != q or len(costs) != q:
        raise ValueError("Parking lot arrays do not match Q")

    match_edges = re.search(r"\bE\s*=\s*\{(.*?)\};", dat_text, flags=re.DOTALL)
    if not match_edges:
        raise ValueError("Missing edge set E")
    edges = [(int(a), int(b)) for a, b in re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", match_edges.group(1))]
    distance = parse_number_list(dat_text, "Distance", float)
    edge_time = parse_number_list(dat_text, "EdgeTime", float)
    if len(distance) != len(edges) or len(edge_time) != len(edges):
        raise ValueError("Edge weights do not align with E")

    walk_distance = parse_number_list(dat_text, "WalkDistanceToDest", float)
    walk_time = parse_number_list(dat_text, "WalkTimeToDest", float)
    feasible_parking = parse_number_list(dat_text, "FeasibleParking", int)

    alpha = get_float("Alpha")
    solver_weight = get_float("w", alpha)

    data: dict[str, Any] = {
        "Nodes": nodes,
        "S": source,
        "D": destination,
        "Q": q,
        "L": lots,
        "C": costs,
        "E": edges,
        "Distance": distance,
        "EdgeTime": edge_time,
        "WalkDistanceToDest": walk_distance,
        "WalkTimeToDest": walk_time,
        "FeasibleParking": feasible_parking,
        "Alpha": alpha,
        "w": solver_weight,
        "W": get_float("W"),
        "W_km": get_float("W_km", get_float("W")),
        "Cmax": get_float("Cmax", math.inf),
        "VariedParameter": scalar_from_text(dat_text, "VariedParameter"),
        "ActiveParameterValue": scalar_from_text(dat_text, "ActiveParameterValue"),
        "ParameterLabel": scalar_from_text(dat_text, "ParameterLabel"),
        "ScaleCategory": scalar_from_text(dat_text, "ScaleCategory"),
        "ExperimentType": scalar_from_text(dat_text, "ExperimentType"),
    }

    data["AvailablePercentage"] = get_float("AvailablePercentage", 0.10)
    data["SpotCountRangeMin"] = get_int("SpotCountRangeMin", 20)
    data["SpotCountRangeMax"] = get_int("SpotCountRangeMax", 40)
    data["TotalSpots"] = get_int("TotalSpots", 0)
    data["SpotsPerLot"] = parse_number_list(dat_text, "SpotsPerLot", int)
    data["AvailableSpotsPerLot"] = parse_number_list(dat_text, "AvailableSpotsPerLot", int)
    data["SpotIds"] = parse_string_list(dat_text, "SpotIds")
    data["SpotLotNode"] = parse_number_list(dat_text, "SpotLotNode", int)
    data["SpotLotIndex"] = parse_number_list(dat_text, "SpotLotIndex", int)
    data["SpotClass"] = parse_string_list(dat_text, "SpotClass")
    data["SpotAvailable"] = parse_number_list(dat_text, "SpotAvailable", int)
    data["SpotInternalDriveTime"] = parse_number_list(dat_text, "SpotInternalDriveTime", float)
    data["SpotInternalWalkTime"] = parse_number_list(dat_text, "SpotInternalWalkTime", float)
    data["SpotParentCost"] = parse_number_list(dat_text, "SpotParentCost", float)
    data["SpotCost"] = parse_number_list(dat_text, "SpotCost", float)
    return data


def parking_lot_maps(data: dict[str, Any]) -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    walk_distance = {lot: dist for lot, dist in zip(data["L"], data.get("WalkDistanceToDest", []))}
    walk_time = {lot: dist for lot, dist in zip(data["L"], data.get("WalkTimeToDest", []))}
    feasible = {lot: flag for lot, flag in zip(data["L"], data.get("FeasibleParking", []))}
    return walk_distance, walk_time, feasible


def get_feasible_spots(data: dict[str, Any], parking_node: int | None = None) -> list[dict[str, Any]]:
    if not data.get("SpotIds"):
        return []
    walk_distance_map, external_walk_map, _ = parking_lot_maps(data)
    feasible: list[dict[str, Any]] = []
    for spot_index, lot_node in enumerate(data["SpotLotNode"], start=1):
        if parking_node is not None and lot_node != parking_node:
            continue
        if int(data["SpotAvailable"][spot_index - 1]) != 1:
            continue
        external_walk = float(external_walk_map.get(lot_node, math.inf))
        if not math.isfinite(external_walk):
            continue
        internal_drive = float(data["SpotInternalDriveTime"][spot_index - 1])
        internal_walk = float(data["SpotInternalWalkTime"][spot_index - 1])
        total_walk = internal_walk + external_walk
        spot_cost = float(data["SpotCost"][spot_index - 1])
        if total_walk > float(data["W"]) + 1e-12 or spot_cost > float(data["Cmax"]) + 1e-12:
            continue
        feasible.append(
            {
                "parking_spot_index": spot_index,
                "parking_spot_id": data["SpotIds"][spot_index - 1],
                "spot_class": data["SpotClass"][spot_index - 1],
                "parking_node": int(lot_node),
                "internal_drive_time_hr": internal_drive,
                "internal_walk_time_hr": internal_walk,
                "external_walk_distance_km": float(walk_distance_map.get(lot_node, 0.0)),
                "external_walk_time_hr": external_walk,
                "walk_time_hr": total_walk,
                "parking_cost": spot_cost,
            }
        )
    return feasible


def candidate_spots(data: dict[str, Any], parking_node: int | None = None) -> list[int]:
    return [int(spot["parking_spot_index"]) for spot in get_feasible_spots(data, parking_node)]


def spot_objective_from_totals(alpha: float, total_travel_hr: float, parking_cost: float) -> float:
    alpha_value = float(alpha)
    total_travel_min = 60.0 * float(total_travel_hr)
    return alpha_value * total_travel_min + (1.0 - alpha_value) * float(parking_cost)


def spot_objective(
    alpha: float,
    external_drive_hr: float,
    internal_drive_hr: float,
    internal_walk_hr: float,
    external_walk_hr: float,
    parking_cost: float,
) -> float:
    total_travel_hr = (
        float(external_drive_hr)
        + float(internal_drive_hr)
        + float(internal_walk_hr)
        + float(external_walk_hr)
    )
    return spot_objective_from_totals(alpha, total_travel_hr, parking_cost)


def build_spot_normalization_context(data: dict[str, Any]) -> dict[str, float]:
    alpha = float(data["Alpha"])
    return {
        "alpha": alpha,
        "travel_scale": 1.0,
        "cost_scale": 1.0,
        "travel_weight": alpha,
        "cost_weight": 1.0 - alpha,
    }


def normalized_spot_objective(
    total_travel_hr: float,
    parking_cost: float,
    *,
    context: dict[str, float],
) -> float:
    alpha = float(context.get("alpha", context.get("travel_weight", 0.5)))
    return spot_objective_from_totals(alpha, total_travel_hr, parking_cost)


def evaluate_spot(
    data: dict[str, Any],
    external_drive_time_hr: float,
    spot_index: int,
    alpha: float | None = None,
    normalization_context: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    for spot in get_feasible_spots(data):
        if int(spot["parking_spot_index"]) != int(spot_index):
            continue
        alpha_value = float(data["Alpha"] if alpha is None else alpha)
        total_travel = float(external_drive_time_hr) + float(spot["internal_drive_time_hr"]) + float(spot["walk_time_hr"])
        objective = spot_objective(
            alpha_value,
            external_drive_time_hr,
            float(spot["internal_drive_time_hr"]),
            float(spot["internal_walk_time_hr"]),
            float(spot["external_walk_time_hr"]),
            float(spot["parking_cost"]),
        )
        return {
            **spot,
            "drive_time_hr": float(external_drive_time_hr),
            "travel_time_hr": total_travel,
            "objective": objective,
        }
    return None


def select_best_spot(
    data: dict[str, Any],
    external_drive_time_hr: float,
    parking_node: int,
    alpha: float | None = None,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for spot_index in candidate_spots(data, parking_node):
        row = evaluate_spot(data, external_drive_time_hr, spot_index, alpha=alpha)
        if row is None:
            continue
        if best is None or (row["objective"], row["travel_time_hr"], row["parking_cost"], row["parking_spot_index"]) < (
            best["objective"],
            best["travel_time_hr"],
            best["parking_cost"],
            best["parking_spot_index"],
        ):
            best = row
    return best


def parse_spot_parameters(path: Path, text: str) -> tuple[float, tuple[int, int]]:
    available_pct = 0.10
    count_range = (20, 40)
    varied = scalar_from_text(text, "VariedParameter")
    active = scalar_from_text(text, "ActiveParameterValue")
    filename = path.stem

    if varied == "parking_spot_availability" or "_spot_availability_" in filename:
        match = re.search(r"(\d+)", active or filename)
        if match:
            available_pct = max(0.01, min(1.0, int(match.group(1)) / 100.0))
    if varied == "spots_per_lot" or "_spots_per_lot_" in filename:
        match = re.search(r"(\d+)\D+(\d+)", active or filename)
        if match:
            lo = int(match.group(1))
            hi = int(match.group(2))
            count_range = (min(lo, hi), max(lo, hi))
    return available_pct, count_range


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            text = f"{value:.6f}".rstrip("0").rstrip(".")
            return text if text else "0"
        return "0"
    return str(value)


def _format_array(name: str, values: list[Any], per_line: int = 8) -> str:
    lines = [f"{name} = ["]
    if values:
        for start in range(0, len(values), per_line):
            chunk = values[start:start + per_line]
            lines.append(", ".join(_format_value(value) for value in chunk))
    lines.append("];")
    return "\n".join(lines)


def build_spot_section(path: Path, text: str, data: dict[str, Any]) -> str:
    available_pct, count_range = parse_spot_parameters(path, text)
    spot_count_min, spot_count_max = count_range
    varied_parameter = scalar_from_text(text, "VariedParameter")
    lots = data["L"]
    parent_costs = data["C"]
    feasible_flags = data.get("FeasibleParking", [1] * len(lots))
    walk_distance_map, external_walk_map, _ = parking_lot_maps(data)
    # Keep Wmax parameter sweeps on a fixed canonical spot layout. The walking
    # constraint changes at the lot level; spot attributes must remain stable.
    reference_walk_limit_hr = 0.10 if varied_parameter == "maximum_walking_distance" else float(data["W"])

    spot_ids: list[str] = []
    spot_lot_node: list[int] = []
    spot_lot_index: list[int] = []
    spot_class: list[str] = []
    spot_available: list[int] = []
    spot_internal_drive: list[float] = []
    spot_internal_walk: list[float] = []
    spot_parent_cost: list[float] = []
    spot_cost: list[float] = []
    spots_per_lot: list[int] = []
    available_spots_per_lot: list[int] = []

    for lot_index, (lot_node, parent_cost) in enumerate(zip(lots, parent_costs), start=1):
        if varied_parameter == "spots_per_lot":
            max_rng = random.Random(stable_seed(data["Nodes"], lot_node, "spot-count-max"))
            max_count = max_rng.randint(60, 80)
            count = max(spot_count_min, min(spot_count_max, int(round(max_count * (spot_count_max / 80.0)))))
            count = max(count, min(spot_count_max, spot_count_min))
            canonical_count = max_count
            capacity_factor = max(0.0, min(1.0, (canonical_count - 20) / 60.0))
        else:
            rng = random.Random(stable_seed(data["Nodes"], lot_node, spot_count_min, spot_count_max))
            count = rng.randint(spot_count_min, spot_count_max)
            canonical_count = count
            capacity_factor = 0.35
        rng = random.Random(stable_seed(data["Nodes"], lot_node, "spot-layout"))
        ext_walk = float(external_walk_map.get(lot_node, 0.0))
        ext_walk_km = float(walk_distance_map.get(lot_node, 0.0))
        walk_pressure = min(1.0, ext_walk / max(1e-9, reference_walk_limit_hr))
        price_factor = max(0.0, min(1.0, (float(parent_cost) - 8.0) / 12.5))
        spots_per_lot.append(count)

        lot_rows: list[dict[str, Any]] = []
        for offset in range(canonical_count):
            quality = offset / max(1, canonical_count - 1)
            if quality <= 0.25:
                label = "near"
            elif quality <= 0.65:
                label = "mid"
            else:
                label = "far"

            drive_low = 0.0018 + 0.0008 * walk_pressure - 0.0013 * capacity_factor
            drive_high = 0.0165 + 0.0015 * walk_pressure - 0.0062 * capacity_factor
            walk_low = 0.0024 + 0.0040 * walk_pressure - 0.0015 * capacity_factor
            walk_high = 0.0300 + 0.0250 * walk_pressure - 0.0100 * capacity_factor
            wiggle = (rng.random() - 0.5) * 0.00012
            internal_drive = max(0.0010, drive_low + quality * (drive_high - drive_low) + wiggle)
            internal_walk = max(0.0018, walk_low + quality * (walk_high - walk_low) + wiggle)

            cost_discount = 1.45 * capacity_factor + 0.55 * walk_pressure
            convenience_markup = 0.22 * quality + 0.20 * price_factor
            internal_cost = max(4.5, parent_cost * (0.88 + convenience_markup) - cost_discount + (rng.random() - 0.5) * 0.05)

            lot_rows.append(
                {
                    "id": f"{lot_node}_S{offset + 1:03d}",
                    "lot_node": lot_node,
                    "lot_index": lot_index,
                    "class": label,
                    "available": 0,
                    "internal_drive": round(internal_drive, 6),
                    "internal_walk": round(internal_walk, 6),
                    "parent_cost": round(float(parent_cost), 2),
                    "cost": round(internal_cost, 2),
                }
            )

        active_rows = lot_rows[:count]
        if varied_parameter == "spots_per_lot":
            available_score = {
                idx: stable_seed(data["Nodes"], lot_node, lot_rows[idx]["id"], "spot-availability")
                for idx in range(canonical_count)
            }
            chosen: set[int] = set()
            base_available = max(1, int(round(canonical_count * available_pct))) if canonical_count else 0
            for idx in sorted(range(canonical_count), key=lambda item: (available_score[item], item))[:base_available]:
                chosen.add(idx)
            feasible_indices = [
                idx
                for idx, row in enumerate(lot_rows)
                if row["cost"] <= float(data["Cmax"]) + 1e-12 and feasible_flags[lot_index - 1] == 1
            ]
            if feasible_indices and not any(idx in chosen for idx in feasible_indices):
                chosen.add(min(feasible_indices))
        else:
            available_count = max(1, int(round(count * available_pct))) if count else 0
            available_count = min(count, available_count)
            feasible_indices = [
                idx
                for idx, row in enumerate(active_rows)
                if row["cost"] <= float(data["Cmax"]) + 1e-12 and feasible_flags[lot_index - 1] == 1
            ]
            chosen = set(range(min(available_count, count)))
            if not any(idx in chosen for idx in feasible_indices) and feasible_indices and available_count > 0:
                worst_chosen = max(chosen) if chosen else None
                if worst_chosen is not None:
                    chosen.remove(worst_chosen)
                chosen.add(min(feasible_indices))

        for idx, row in enumerate(active_rows):
            row["available"] = 1 if idx in chosen else 0
            spot_ids.append(row["id"])
            spot_lot_node.append(row["lot_node"])
            spot_lot_index.append(row["lot_index"])
            spot_class.append(row["class"])
            spot_available.append(row["available"])
            spot_internal_drive.append(row["internal_drive"])
            spot_internal_walk.append(row["internal_walk"])
            spot_parent_cost.append(row["parent_cost"])
            spot_cost.append(row["cost"])

        available_spots_per_lot.append(sum(int(row["available"]) for row in active_rows))

    lines = [
        "",
        "/*********************************************",
        " * Spot-level parking extension",
        " *********************************************/",
        "",
        f"AvailablePercentage = {_format_value(round(available_pct, 4))};",
        f"SpotCountRangeMin = {spot_count_min};",
        f"SpotCountRangeMax = {spot_count_max};",
        f"TotalSpots = {len(spot_ids)};",
        _format_array("SpotsPerLot", spots_per_lot),
        _format_array("AvailableSpotsPerLot", available_spots_per_lot),
        _format_array("SpotIds", spot_ids),
        _format_array("SpotLotNode", spot_lot_node),
        _format_array("SpotLotIndex", spot_lot_index),
        _format_array("SpotClass", spot_class),
        _format_array("SpotAvailable", spot_available),
        _format_array("SpotInternalDriveTime", spot_internal_drive),
        _format_array("SpotInternalWalkTime", spot_internal_walk),
        _format_array("SpotParentCost", spot_parent_cost),
        _format_array("SpotCost", spot_cost),
        "",
    ]
    return "\n".join(lines)


def append_spot_section(path: Path, text: str) -> str:
    marker = "/*********************************************\n * Spot-level parking extension"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    data = parse_opl_dat(text)
    return text.rstrip() + "\n" + build_spot_section(path, text, data)
