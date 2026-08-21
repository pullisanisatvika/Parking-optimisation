#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = ROOT / "parking-lot-recommendation-baseline"
EXTENSION_ROOT = ROOT / "parking-lot-recommendation-extension"

BASELINE_MANIFEST = BASELINE_ROOT / "RESULTS" / "VALIDATION" / "parameter_sensitivity_manifest.csv"
EXTENSION_MANIFEST = EXTENSION_ROOT / "RESULTS" / "VALIDATION" / "parameter_sensitivity_manifest.csv"

EXPECTED_COUNTS = {
    "maximum_walking_distance": 4,
    "parking_supply": 4,
    "parking_budget_cmax": 4,
    "parking_cost_variation_mode": 4,
    "traffic_level": 4,
    "edge_density": 4,
}
STABLE_FIELDS_BY_PARAMETER = {
    "maximum_walking_distance": {"node_count", "directed_edge_count", "parking_supply_pct", "cmax", "alpha", "speed_min", "speed_max", "parking_cost_min", "parking_cost_max"},
    "parking_supply": {"node_count", "directed_edge_count", "wmax", "cmax", "alpha", "speed_min", "speed_max", "parking_cost_min", "parking_cost_max"},
    "parking_budget_cmax": {"node_count", "directed_edge_count", "parking_supply_pct", "wmax", "alpha", "speed_min", "speed_max", "parking_cost_min", "parking_cost_max"},
    "parking_cost_variation_mode": {"node_count", "directed_edge_count", "parking_supply_pct", "wmax", "cmax", "alpha", "speed_min", "speed_max"},
    "traffic_level": {"node_count", "directed_edge_count", "parking_supply_pct", "wmax", "cmax", "alpha", "parking_cost_min", "parking_cost_max"},
    "edge_density": {"node_count", "parking_supply_pct", "wmax", "cmax", "alpha", "speed_min", "speed_max", "parking_cost_min", "parking_cost_max"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def dat_drive_edges(path: Path) -> set[tuple[int, int]]:
    text = path.read_text(encoding="utf-8")
    edge_block = re.search(r"\bE\s*=\s*\{(.*?)\}\s*;", text, flags=re.DOTALL)
    type_block = re.search(r"\bEdgeType\s*=\s*\[(.*?)\]\s*;", text, flags=re.DOTALL)
    if not edge_block or not type_block:
        raise RuntimeError(f"Missing E/EdgeType block in {path}")
    edges = [(int(u), int(v)) for u, v in re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", edge_block.group(1))]
    edge_types = re.findall(r'"([^"]+)"', type_block.group(1))
    if len(edges) != len(edge_types):
        raise RuntimeError(f"Edge/type length mismatch in {path}: edges={len(edges)} types={len(edge_types)}")
    return {edge for edge, edge_type in zip(edges, edge_types) if edge_type == "drive"}


def uncontrolled_strict_failures(rows: list[dict[str, str]], manifest_name: str) -> list[str]:
    failures: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scale_category"], row["varied_parameter"])].append(row)
    for (scale, parameter), group_rows in sorted(grouped.items()):
        expected_count = EXPECTED_COUNTS.get(parameter)
        if expected_count is not None and len(group_rows) != expected_count:
            failures.append(f"{manifest_name}:{scale}:{parameter}: expected {expected_count} rows, found {len(group_rows)}")
        stable_fields = STABLE_FIELDS_BY_PARAMETER.get(parameter, set())
        for field in sorted(stable_fields):
            values = {str(row.get(field, "")).strip() for row in group_rows}
            if len(values) > 1:
                failures.append(f"{manifest_name}:{scale}:{parameter}: field {field} varies unexpectedly -> {sorted(values)}")
    return failures


def edge_density_rows(rows: list[dict[str, str]], scale: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["scale_category"] == scale and row["varied_parameter"] == "edge_density"]
    selected.sort(key=lambda row: int(float(row["active_parameter_value"])))
    return selected


def edge_density_dat_dir(model: str, scale: str) -> Path:
    base = BASELINE_ROOT if model == "baseline" else EXTENSION_ROOT
    family_root = "final_cases_with_spots" if model == "extension" else "final_cases"
    return base / family_root / "parameter_sensitivity" / scale


def check_edge_density_family(model: str, display_scale: str, manifest_scale: str, rows: list[dict[str, str]]) -> tuple[bool, list[str], list[str]]:
    failures: list[str] = []
    values = [str(int(float(row["active_parameter_value"]))) for row in rows]
    if len(values) != 4:
        failures.append(f"{model}:{display_scale}: expected 4 edge-density levels, found {len(values)} -> {values}")
    dat_dir = edge_density_dat_dir(model, manifest_scale)
    dat_paths = [dat_dir / row["dataset_name"] for row in rows]
    if not all(path.exists() for path in dat_paths):
        missing = [str(path) for path in dat_paths if not path.exists()]
        failures.append(f"{model}:{display_scale}: missing datasets {missing}")
        return False, values, failures

    edge_sets = [dat_drive_edges(path) for path in dat_paths]
    for idx in range(len(edge_sets) - 1):
        if not edge_sets[idx].issubset(edge_sets[idx + 1]):
            failures.append(
                f"{model}:{display_scale}: nesting failure between {dat_paths[idx].name} and {dat_paths[idx + 1].name}"
            )
    return not failures, values, failures


def traffic_violations(rows: list[dict[str, str]], manifest_name: str) -> list[str]:
    failures: list[str] = []
    order = ["very_low", "low", "medium", "high"]
    selected = [row for row in rows if row["scale_category"] == "SUMO" and row["varied_parameter"] == "traffic_level"]
    by_label = {row["active_parameter_value"]: row for row in selected}
    speed_mins = []
    speed_maxs = []
    for label in order:
        row = by_label.get(label)
        if row is None:
            failures.append(f"{manifest_name}:SUMO:traffic_level missing {label}")
            continue
        speed_mins.append(parse_float(row.get("speed_min", "")))
        speed_maxs.append(parse_float(row.get("speed_max", "")))
    if not failures:
        for earlier, later in zip(speed_mins, speed_mins[1:]):
            if earlier is None or later is None or earlier < later:
                failures.append(f"{manifest_name}:SUMO:traffic_level speed_min monotonicity failure {speed_mins}")
                break
        for earlier, later in zip(speed_maxs, speed_maxs[1:]):
            if earlier is None or later is None or earlier < later:
                failures.append(f"{manifest_name}:SUMO:traffic_level speed_max monotonicity failure {speed_maxs}")
                break
    return failures


def main() -> int:
    baseline_rows = read_csv(BASELINE_MANIFEST)
    extension_rows = read_csv(EXTENSION_MANIFEST)

    strict_failures = uncontrolled_strict_failures(baseline_rows, "baseline_manifest")
    strict_failures.extend(uncontrolled_strict_failures(extension_rows, "extension_manifest"))

    traffic_failures = traffic_violations(baseline_rows, "baseline_manifest")
    traffic_failures.extend(traffic_violations(extension_rows, "extension_manifest"))

    edge_specs = [
        ("baseline", "small", "small", baseline_rows),
        ("baseline", "medium_large", "medium_large", baseline_rows),
        ("baseline", "sumo", "SUMO", baseline_rows),
        ("extension", "small_scale", "small", extension_rows),
        ("extension", "medium_large", "medium_large", extension_rows),
        ("extension", "sumo", "SUMO", extension_rows),
    ]

    edge_failures: list[str] = []
    print("EDGE_DENSITY_CONTROL")
    for model, display_scale, manifest_scale, rows in edge_specs:
        selected = edge_density_rows(rows, manifest_scale)
        ok, values, failures = check_edge_density_family(model, display_scale, manifest_scale, selected)
        print(f"{model} {display_scale}: values={values} count={len(values)} nested={'PASS' if ok else 'FAIL'}")
        edge_failures.extend(failures)

    print(f"Uncontrolled strict series = {len(strict_failures)}")
    print(f"TYPE A failures = {len(strict_failures)}")
    print(f"SUMO traffic violations = {len(traffic_failures)}")

    if strict_failures or traffic_failures or edge_failures:
        print("FAILURES")
        for failure in strict_failures + traffic_failures + edge_failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
