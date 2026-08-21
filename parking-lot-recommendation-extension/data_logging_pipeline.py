from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path
from functools import lru_cache

from spot_level_utils import build_spot_normalization_context, normalized_spot_objective, parse_opl_dat as parse_spot_opl_dat


ROOT = Path(__file__).resolve().parent
FINAL_CASES = ROOT / "final_cases"
SUMO_CASES = FINAL_CASES / "SUMO"
PROJECT_SUMO_DIR = ROOT / "SUMO"
PROJECT_SUMO_CASES = PROJECT_SUMO_DIR / "solver_cases_10"
COMBINED_DIR = ROOT / "RESULTS" / "COMBINED_ALL_METHODS"

SMALL_SCALE_NODES = [10, 20, 30, 40, 50]
MEDIUM_LARGE_NODES = [500, 1000, 1500, 2000, 2500]
ACTIVE_SYNTHETIC_NODES = SMALL_SCALE_NODES + MEDIUM_LARGE_NODES
PARTITION_ORDER = ["small_scale", "medium_large_scale"]
METHOD_ORDER = ["heuristics", "ga", "pso", "aco"]

CANONICAL_RESULTS_HEADER = [
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
    "runtime_ms",
    "drive_path",
    "walk_path",
]

ROUTE_HEADER = ["case", "parking_node", "feasible", "drive_path", "walk_path", "full_route"]

LEGACY_RESULT_HEADERS = {
    "heuristics": [
        "Case",
        "Nodes",
        "Edges",
        "ParkingLots",
        "Source",
        "Destination",
        "Alpha",
        "Wmax",
        "Cmax",
        "WeightW",
        "FeasibleParking",
        "runtime_ms",
        "ParkingNode",
        "drive_time_hr",
        "walk_distance_km",
        "parking_cost",
        "objective",
        "Feasible",
        "Selected",
        "DrivePath",
        "WalkPath",
    ],
    "ga": [
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
        "feasible",
        "parking_node",
        "drive_time_hr",
        "walk_distance_km",
        "parking_cost",
        "objective",
        "runtime_ms",
        "drive_path",
        "walk_path",
    ],
    "pso": [
        "method",
        "dat_file",
        "nodes",
        "edges",
        "parking_lots",
        "S",
        "D",
        "w_dat",
        "Wmax",
        "Cmax",
        "w",
        "feasible",
        "parking_node",
        "drive_time_hr",
        "walk_distance_km",
        "parking_cost",
        "objective",
        "runtime_ms",
        "drive_path",
        "walk_path",
    ],
    "aco": [
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
        "parking_cost",
        "objective",
        "runtime_ms",
        "drive_path",
        "walk_path",
    ],
}

LEGACY_CSV_FILES = [
    Path("heuristics/log.csv"),
    Path("heuristics/smallscale_heuristic/log_smallscale_heuristic.csv"),
    Path("heuristics/routes_heuristic.csv"),
    Path("heuristics/smallscale_heuristic/routes_heuristic.csv"),
    Path("ga/data/results_ga.csv"),
    Path("ga/data/smallscale_ga/results_ga_smallscale.csv"),
    Path("ga/data/routes_ga.csv"),
    Path("ga/data/smallscale_ga/routes_ga.csv"),
    Path("pso/data/results_pso.csv"),
    Path("pso/data/smallscale_pso/results_pso_smallscale.csv"),
    Path("pso/data/routes_pso.csv"),
    Path("pso/data/smallscale_pso/routes_pso.csv"),
    Path("aco/data/results_aco.csv"),
    Path("aco/data/smallscale_aco/results_aco_smallscale.csv"),
    Path("aco/data/routes_aco.csv"),
    Path("aco/data/smallscale_aco/routes_aco.csv"),
]


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_ANNOUNCED_SUMO_DAT_SOURCES: set[str] = set()
_WARNED_DAT_LOOKUPS: set[str] = set()


def _announce_sumo_dat_source(path: Path) -> None:
    key = str(path.resolve())
    if key in _ANNOUNCED_SUMO_DAT_SOURCES:
        return
    _ANNOUNCED_SUMO_DAT_SOURCES.add(key)
    try:
        display = path.relative_to(ROOT)
    except ValueError:
        display = path
    print(f"Using SUMO .dat source: {display}")


def _warn_missing_dat_once(message: str) -> None:
    if message in _WARNED_DAT_LOOKUPS:
        return
    _WARNED_DAT_LOOKUPS.add(message)
    print(message)


def _find_dat_path(dat_file: str) -> Path | None:
    dat_name = normalize_case_name(dat_file)
    if not dat_name:
        return None
    is_sumo_case = dat_name.upper().startswith("N2055_C")
    if is_sumo_case:
        direct_sumo = PROJECT_SUMO_CASES / dat_name
        if direct_sumo.exists():
            _announce_sumo_dat_source(direct_sumo)
            return direct_sumo

        recursive_sumo_matches = sorted(PROJECT_SUMO_DIR.rglob(dat_name))
        if len(recursive_sumo_matches) == 1:
            _announce_sumo_dat_source(recursive_sumo_matches[0])
            return recursive_sumo_matches[0]
        if len(recursive_sumo_matches) > 1:
            preferred = [path for path in recursive_sumo_matches if path.parent == PROJECT_SUMO_CASES]
            if preferred:
                _announce_sumo_dat_source(preferred[0])
                return preferred[0]
            print(
                f"[ERROR] Ambiguous SUMO dataset file lookup for {dat_name}: "
                + ", ".join(str(p) for p in recursive_sumo_matches)
            )
            return None

    m = re.match(r"N(\d+)(?:_|\.dat)", dat_name, re.IGNORECASE)
    if m:
        n = m.group(1)
        exact = FINAL_CASES / f"n_{n}" / dat_name
        if exact.exists():
            return exact
        ps_matches = sorted((FINAL_CASES / "parameter_sensitivity").rglob(dat_name))
        if len(ps_matches) == 1:
            return ps_matches[0]
        if len(ps_matches) > 1:
            print(f"[ERROR] Ambiguous parameter-sensitivity dataset file lookup for {dat_name}: " + ", ".join(str(p) for p in ps_matches))
            return None
        matches = sorted(FINAL_CASES.rglob(dat_name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"[ERROR] Ambiguous dataset file lookup for {dat_name}: " + ", ".join(str(p) for p in matches))
            return None
        _warn_missing_dat_once(f"[WARN] .dat file not found: {exact}")
        return None
    matches = sorted(FINAL_CASES.rglob(dat_name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[ERROR] Ambiguous dataset file lookup for {dat_name}: " + ", ".join(str(p) for p in matches))
        return None
    _warn_missing_dat_once(f"[WARN] .dat file not found: {dat_name}")
    return None


def _parse_synthetic_node_size_from_name(dat_name: str) -> int | None:
    m = re.match(r"N(\d+)(?:_|\.dat)", dat_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _is_active_synthetic_dat_name(dat_name: str) -> bool:
    node_size = _parse_synthetic_node_size_from_name(normalize_case_name(dat_name))
    return node_size in ACTIVE_SYNTHETIC_NODES if node_size is not None else False


def _is_active_synthetic_dat_path(dat_path: Path) -> bool:
    return _is_active_synthetic_dat_name(dat_path.name)

    matches = sorted(FINAL_CASES.rglob(dat_name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous dataset file lookup for {dat_name}: "
            + ", ".join(str(p) for p in matches)
        )
    return None


@lru_cache(maxsize=None)
def _parse_dat_metadata(dat_file: str):
    dat_path = _find_dat_path(dat_file)
    if dat_path is None or not dat_path.exists():
        return None

    text = dat_path.read_text(encoding="utf-8", errors="replace")

    def _get_int(name: str):
        m = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", text)
        return int(m.group(1)) if m else None

    def _get_float(name: str):
        m = re.search(rf"\b{name}\s*=\s*({FLOAT_RE})\s*;", text)
        return float(m.group(1)) if m else None

    def _get_list(name: str, cast=float):
        m = re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]\s*;", text, flags=re.DOTALL)
        if not m:
            return []
        vals = re.findall(FLOAT_RE if cast is float else r"\d+", m.group(1))
        return [cast(v) for v in vals]

    def _get_str_list(name: str):
        m = re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]\s*;", text, flags=re.DOTALL)
        if not m:
            return []
        return re.findall(r'"([^"]+)"', m.group(1))

    q = _get_int("Q")
    lots = _get_list("L", int)
    costs = _get_list("C", float)
    walk_km = _get_list("WalkDistanceToDest", float)
    walk_hr = _get_list("WalkTimeToDest", float)

    edge_matches = re.findall(r"<\s*\d+\s*,\s*\d+\s*>", text)
    edge_types = _get_str_list("EdgeType")
    if edge_types and len(edge_types) == len(edge_matches):
        directed_drive_edges = sum(1 for edge_type in edge_types if edge_type == "drive")
    else:
        directed_drive_edges = len(edge_matches)
    parking_cost_by_node = {node: cost for node, cost in zip(lots, costs)}
    walk_km_by_node = {node: dist for node, dist in zip(lots, walk_km)}
    walk_hr_by_node = {node: wt for node, wt in zip(lots, walk_hr)}

    alpha = _get_float("Alpha")
    return {
        "dat_path": dat_path,
        "nodes": _get_int("Nodes"),
        "edges": directed_drive_edges,
        "parking_lots": q if q is not None else len(lots),
        "S": _get_int("S"),
        "D": _get_int("D"),
        "Alpha": alpha,
        "w": alpha,
        "Wmax": _get_float("W_km"),
        "W_time": _get_float("W"),
        "Cmax": _get_float("Cmax"),
        "lot_nodes": lots,
        "lot_costs": costs,
        "parking_cost_by_node": parking_cost_by_node,
        "walk_km_by_node": walk_km_by_node,
        "walk_hr_by_node": walk_hr_by_node,
    }


@lru_cache(maxsize=None)
def _spot_objective_context_for_dat(dat_file: str):
    meta = _parse_dat_metadata(dat_file)
    if meta is None:
        return None
    text = meta["dat_path"].read_text(encoding="utf-8", errors="replace")
    data = parse_spot_opl_dat(text)
    return build_spot_normalization_context(data)


def _resolve_selected_node(row: dict, meta: dict) -> int | None:
    selected_from_path = _selected_node_from_paths(row)
    if selected_from_path is not None:
        if selected_from_path in meta["parking_cost_by_node"]:
            return selected_from_path
        return None

    parking_node_text = _text(row.get("parking_node", ""))
    if parking_node_text:
        p = _to_float_or_none(parking_node_text)
        if p is not None and math.isfinite(p):
            p_int = int(round(p))
            if p_int in meta["parking_cost_by_node"]:
                return p_int
            return None

    return None


# --- Inserted helper function ---
def _force_reconcile_row_with_dataset(row: dict, route_row: dict | None = None) -> dict:
    out = dict(row)
    if route_row:
        out["drive_path"] = _text(route_row.get("drive_path", out.get("drive_path", "")))
        out["walk_path"] = _text(route_row.get("walk_path", out.get("walk_path", "")))
        if _text(route_row.get("parking_node", "")):
            out["parking_node"] = _text(route_row.get("parking_node", out.get("parking_node", "")))
        if _text(route_row.get("feasible", "")):
            out["feasible"] = _normalize_feasible(route_row.get("feasible", out.get("feasible", "0")))

    meta = _parse_dat_metadata(out.get("dat_file", ""))
    if meta is None:
        return _sanitize_canonical_row(out)

    for key in ("nodes", "edges", "parking_lots", "S", "D"):
        value = meta.get(key)
        if value is not None:
            out[key] = str(value)

    if meta.get("Alpha") is not None:
        out["Alpha"] = str(meta["Alpha"])
        out["w"] = str(meta["w"])

    if meta.get("Wmax") is not None:
        out["Wmax"] = str(meta["Wmax"])
    if meta.get("Cmax") is not None:
        out["Cmax"] = str(meta["Cmax"])

    resolved_node = _resolve_selected_node(out, meta)
    if resolved_node is not None and resolved_node in meta["parking_cost_by_node"]:
        out["parking_node"] = str(resolved_node)
        out["parking_cost"] = str(meta["parking_cost_by_node"][resolved_node])
        out["walk_distance_km"] = str(meta["walk_km_by_node"].get(resolved_node, ""))
        out["walk_time_hr"] = str(meta["walk_hr_by_node"].get(resolved_node, ""))
    else:
        out["parking_node"] = ""
        out["parking_cost"] = ""
        out["walk_distance_km"] = ""
        out["walk_time_hr"] = ""
        out["feasible"] = "0"

    wmax_km = meta.get("Wmax")
    cmax = meta.get("Cmax")
    walk_distance_km = _to_float_or_none(out.get("walk_distance_km", ""))
    parking_cost_val = _to_float_or_none(out.get("parking_cost", ""))
    if (
        _normalize_feasible(out.get("feasible", "0")) == "1"
        and (
            (wmax_km is not None and walk_distance_km is not None and walk_distance_km > wmax_km + 1e-9)
            or (cmax is not None and parking_cost_val is not None and parking_cost_val > cmax + 1e-9)
        )
    ):
        out["feasible"] = "0"
        out["parking_node"] = ""
        out["parking_cost"] = ""
        out["walk_distance_km"] = ""
        out["walk_time_hr"] = ""
        out["objective"] = ""

    alpha = _to_float_or_none(out.get("Alpha", ""))
    drive_time = _to_float_or_none(out.get("drive_time_hr", ""))
    walk_time = _to_float_or_none(out.get("walk_time_hr", ""))
    parking_cost = _to_float_or_none(out.get("parking_cost", ""))
    if (
        _normalize_feasible(out.get("feasible", "0")) == "1"
        and alpha is not None
        and drive_time is not None
        and walk_time is not None
        and parking_cost is not None
    ):
        total_time = drive_time + walk_time
        context = _spot_objective_context_for_dat(out.get("dat_file", ""))
        if context is not None:
            out["objective"] = str(normalized_spot_objective(total_time, parking_cost, context=context))

    return _sanitize_canonical_row(out)

def _enrich_row_from_dataset(row: dict) -> dict:
    out = dict(row)
    meta = _parse_dat_metadata(out.get("dat_file", ""))
    if meta is None:
        return out

    for key in ("nodes", "edges", "parking_lots", "S", "D"):
        value = meta.get(key)
        if value is not None:
            out[key] = str(value)

    if meta.get("Alpha") is not None:
        out["Alpha"] = str(meta["Alpha"])
        if not _text(out.get("w", "")):
            out["w"] = str(meta["w"])

    if meta.get("Wmax") is not None:
        out["Wmax"] = str(meta["Wmax"])

    if meta.get("Cmax") is not None:
        out["Cmax"] = str(meta["Cmax"])

    resolved_node = _resolve_selected_node(out, meta)
    if resolved_node is not None:
        out["parking_node"] = str(resolved_node)
        out["parking_cost"] = str(meta["parking_cost_by_node"][resolved_node])
        if resolved_node in meta["walk_km_by_node"]:
            out["walk_distance_km"] = str(meta["walk_km_by_node"][resolved_node])

    return out


def _text(v) -> str:
    return "" if v is None else str(v).strip()


def _is_nonfinite_text(v: str) -> bool:
    s = _text(v).lower()
    return s in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan"}


def _to_float_or_none(v: str):
    s = _text(v)
    if not s or _is_nonfinite_text(s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# --- Helper functions for extracting node from path(s) ---

def _extract_path_nodes(path_text: str) -> list[int]:
    text = _text(path_text)
    if not text:
        return []
    nums = re.findall(r"\d+", text)
    return [int(x) for x in nums]


def _selected_node_from_paths(row: dict) -> int | None:
    walk_nodes = _extract_path_nodes(row.get("walk_path", ""))
    if walk_nodes:
        return walk_nodes[0]

    drive_nodes = _extract_path_nodes(row.get("drive_path", ""))
    if drive_nodes:
        return drive_nodes[-1]

    return None


def _normalize_feasible(v) -> str:
    s = _text(v).lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return "1"
    if s in {"0", "false", "f", "no", "n", "", "-1", "none", "null"}:
        return "0"
    fv = _to_float_or_none(s)
    if fv is None:
        return "0"
    return "1" if fv > 0 else "0"


def _clean_numeric_text(v: str) -> str:
    s = _text(v)
    if not s or _is_nonfinite_text(s):
        return ""
    return s


def normalize_case_name(text: str) -> str:
    text = _text(text)
    if text and not text.lower().endswith(".dat"):
        return f"{text}.dat"
    return text


def parse_case_key(text: str):
    match = re.search(r"N(\d+)(?:_C(\d+))?", text or "", re.IGNORECASE)
    if match:
        case_num = int(match.group(2)) if match.group(2) else 0
        return int(match.group(1)), case_num, text
    return 10**9, 10**9, text or ""

def case_partition_from_dat_name(dat_file: str) -> str:
    n, _, _ = parse_case_key(dat_file)
    if n in SMALL_SCALE_NODES:
        return "small_scale"
    if n in MEDIUM_LARGE_NODES:
        return "medium_large_scale"
    return "other"


def _partition_case_rows(rows: list[dict], case_key_name: str):
    grouped = {"small_scale": [], "medium_large_scale": [], "other": []}
    for row in rows:
        grouped[case_partition_from_dat_name(row.get(case_key_name, ""))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: parse_case_key(r.get(case_key_name, "")))
    return grouped

def parse_case_id(text: str):
    match = re.search(r"_C(\d+)(?:_I\d+)?(?:\.dat)?$", _text(text), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _is_ga_shifted_row(row: dict) -> bool:
    feasible = _text(row.get("feasible", ""))
    parking_node = _text(row.get("parking_node", ""))
    runtime_ms = _text(row.get("runtime_ms", ""))
    drive_path = _text(row.get("drive_path", ""))
    walk_path = _text(row.get("walk_path", ""))
    if feasible in {"0", "1", "true", "false", "True", "False"}:
        return False
    if _to_float_or_none(feasible) is None:
        return False
    if _to_float_or_none(parking_node) is None:
        return False
    if "->" not in runtime_ms:
        return False
    if "->" not in drive_path:
        return False
    if walk_path and "->" in walk_path:
        return False
    return True


def _repair_ga_shifted_row(row: dict) -> dict:
    fixed = dict(row)
    fixed["walk_path"] = _text(row.get("drive_path", ""))
    fixed["drive_path"] = _text(row.get("runtime_ms", ""))
    fixed["runtime_ms"] = _text(row.get("objective", ""))
    fixed["objective"] = _text(row.get("parking_cost", ""))
    fixed["parking_cost"] = _text(row.get("walk_distance_km", ""))
    fixed["walk_distance_km"] = _text(row.get("drive_time_hr", ""))
    fixed["drive_time_hr"] = _text(row.get("parking_node", ""))
    fixed["parking_node"] = _text(row.get("feasible", ""))
    fixed["feasible"] = "1"
    return fixed


def _is_ga_case_in_method_row(row: dict) -> bool:
    method = _text(row.get("method", ""))
    dat_file = _text(row.get("dat_file", ""))
    return bool(
        re.fullmatch(r"N\d+_C\d+_I\d+", method, re.IGNORECASE)
        and re.fullmatch(r"\d+\.dat", dat_file, re.IGNORECASE)
    )


def _repair_ga_case_in_method_row(row: dict) -> dict:
    fixed = dict(row)
    fixed["method"] = "GA_ROUTE_EVOLUTION"
    fixed["dat_file"] = normalize_case_name(_text(row.get("method", "")))

    case_match = re.match(r"N(\d+)_C\d+_I\d+", _text(row.get("method", "")), re.IGNORECASE)
    fixed["nodes"] = case_match.group(1) if case_match else _text(row.get("nodes", ""))
    fixed["edges"] = _text(row.get("nodes", ""))
    fixed["parking_lots"] = _text(row.get("edges", ""))
    fixed["S"] = _text(row.get("parking_lots", ""))
    fixed["D"] = _text(row.get("S", ""))
    fixed["Alpha"] = _text(row.get("D", ""))
    fixed["Wmax"] = _text(row.get("Alpha", ""))
    fixed["Cmax"] = _text(row.get("Wmax", ""))
    fixed["w"] = _text(row.get("Cmax", ""))
    fixed["feasible"] = _text(row.get("w", ""))
    fixed["parking_node"] = _text(row.get("drive_time_hr", ""))
    fixed["drive_time_hr"] = _text(row.get("walk_distance_km", ""))
    fixed["walk_distance_km"] = _text(row.get("parking_cost", ""))
    fixed["walk_time_hr"] = _text(row.get("walk_time_hr", ""))
    fixed["parking_cost"] = _text(row.get("objective", ""))
    fixed["objective"] = _text(row.get("runtime_ms", ""))
    fixed["runtime_ms"] = _text(row.get("parking_node", ""))
    return fixed


def _sanitize_canonical_row(row: dict) -> dict:
    row = dict(row)
    row["feasible"] = _normalize_feasible(row.get("feasible", "0"))

    if not _text(row.get("w", "")):
        row["w"] = _text(row.get("Alpha", ""))

    if row["feasible"] == "0":
        row["parking_node"] = ""
        row["drive_path"] = ""
        row["walk_path"] = ""
        for k in ("drive_time_hr", "walk_distance_km", "walk_time_hr", "parking_cost", "objective"):
            row[k] = ""
    else:
        for k in ("drive_time_hr", "walk_distance_km", "parking_cost", "objective", "runtime_ms", "walk_time_hr"):
            if k in row:
                row[k] = _clean_numeric_text(row.get(k, ""))
        if _text(row.get("parking_node", "")):
            p = _to_float_or_none(row["parking_node"])
            if p is not None and math.isfinite(p):
                row["parking_node"] = str(int(p)) if abs(p - round(p)) < 1e-9 else str(p)

    row["Alpha"] = _clean_numeric_text(row.get("Alpha", ""))
    row["Wmax"] = _clean_numeric_text(row.get("Wmax", ""))
    row["Cmax"] = _clean_numeric_text(row.get("Cmax", ""))
    row["w"] = _clean_numeric_text(row.get("w", ""))
    row["runtime_ms"] = _clean_numeric_text(row.get("runtime_ms", ""))
    return row


def method_config(method: str, partition: str):
    partition_base = "scale_wise_evaluation" if partition in {"small_scale", "medium_large_scale"} else ""
    if method == "heuristics":
        base = ROOT / "heuristics" / "data" / partition_base / partition if partition_base else ROOT / "heuristics" / "data" / partition
        return {
            "script": ROOT / "heuristics" / "heuristics.py",
            "results": base / "results_heuristic.csv",
            "routes": base / "routes_heuristic.csv",
        }
    if method == "ga":
        base = ROOT / "ga" / "data" / partition_base / partition if partition_base else ROOT / "ga" / "data" / partition
        return {
            "script": ROOT / "ga" / "ga.py",
            "results": base / "results_ga.csv",
            "routes": base / "routes_ga.csv",
        }
    if method == "pso":
        base = ROOT / "pso" / "data" / partition_base / partition if partition_base else ROOT / "pso" / "data" / partition
        return {
            "script": ROOT / "pso" / "pso.py",
            "results": base / "results_pso.csv",
            "routes": base / "routes_pso.csv",
        }
    if method == "aco":
        base = ROOT / "aco" / "data" / partition_base / partition if partition_base else ROOT / "aco" / "data" / partition
        return {
            "script": ROOT / "aco" / "aco.py",
            "results": base / "results_aco.csv",
            "routes": base / "routes_aco.csv",
        }
    raise ValueError(f"Unknown method: {method}")


def method_data_root(method: str) -> Path:
    if method == "heuristics":
        return ROOT / "heuristics" / "data"
    if method == "ga":
        return ROOT / "ga" / "data"
    if method == "pso":
        return ROOT / "pso" / "data"
    if method == "aco":
        return ROOT / "aco" / "data"
    raise ValueError(f"Unknown method: {method}")


def canonical_row_from_results(method: str, row: dict):
    if method == "heuristics":
        canon = {
            "method": "HEURISTIC",
            "dat_file": normalize_case_name(row.get("Case", row.get("dat_file", ""))),
            "nodes": row.get("Nodes", row.get("nodes", "")),
            "edges": row.get("Edges", row.get("edges", "")),
            "parking_lots": row.get("ParkingLots", row.get("parking_lots", "")),
            "S": row.get("Source", row.get("S", "")),
            "D": row.get("Destination", row.get("D", "")),
            "Alpha": row.get("Alpha", ""),
            "Wmax": row.get("Wmax", ""),
            "Cmax": row.get("Cmax", ""),
            "w": row.get("WeightW", row.get("w", "")),
            "feasible": row.get("Feasible", row.get("feasible", "")),
            "parking_node": row.get("ParkingNode", row.get("parking_node", "")),
            "drive_time_hr": row.get("drive_time_hr", ""),
            "walk_distance_km": row.get("walk_distance_km", ""),
            "walk_time_hr": row.get("walk_time_hr", ""),
            "parking_cost": row.get("parking_cost", ""),
            "objective": row.get("objective", ""),
            "runtime_ms": row.get("runtime_ms", row.get("RuntimeMs", "")),
            "drive_path": row.get("DrivePath", row.get("drive_path", "")),
            "walk_path": row.get("WalkPath", row.get("walk_path", "")),
        }
        return _sanitize_canonical_row(canon)
    if method == "ga":
        src = dict(row)
        if _is_ga_shifted_row(src):
            src = _repair_ga_shifted_row(src)
        if _is_ga_case_in_method_row(src):
            src = _repair_ga_case_in_method_row(src)
        canon = {
            "method": src.get("method", "GA_ROUTE_EVOLUTION"),
            "dat_file": normalize_case_name(src.get("Case", src.get("dat_file", ""))),
            "nodes": src.get("Nodes", src.get("nodes", "")),
            "edges": src.get("Edges", src.get("edges", "")),
            "parking_lots": src.get("ParkingLots", src.get("parking_lots", "")),
            "S": src.get("Source", src.get("S", "")),
            "D": src.get("Destination", src.get("D", "")),
            "Alpha": src.get("Alpha", ""),
            "Wmax": src.get("Wmax", ""),
            "Cmax": src.get("Cmax", src.get("cmax", "")),
            "w": src.get("WeightW", src.get("w", src.get("Alpha", ""))),
            "feasible": src.get("FeasibleParking", src.get("feasible", "")),
            "parking_node": src.get("ParkingNode", src.get("parking_node", "")),
            "drive_time_hr": src.get("drive_time_hr", ""),
            "walk_distance_km": src.get("walk_time", src.get("walk_distance_km", "")),
            "walk_time_hr": src.get("walk_time_hr", src.get("walk_time", "")),
            "parking_cost": src.get("parking_cost", ""),
            "objective": src.get("objective", ""),
            "runtime_ms": src.get("runtime_ms", ""),
            "drive_path": src.get("DrivePath", src.get("drive_path", "")),
            "walk_path": src.get("WalkPath", src.get("walk_path", "")),
        }
        return _sanitize_canonical_row(canon)
    if method == "pso":
        alpha = row.get("Alpha", "")
        if not _text(alpha):
            alpha = row.get("w_dat", row.get("w", ""))
        canon = {
            "method": row.get("method", "PSO_CHAOTIC"),
            "dat_file": normalize_case_name(row.get("Case", row.get("dat_file", ""))),
            "nodes": row.get("Nodes", row.get("nodes", "")),
            "edges": row.get("Edges", row.get("edges", "")),
            "parking_lots": row.get("ParkingLots", row.get("parking_lots", "")),
            "S": row.get("Source", row.get("S", "")),
            "D": row.get("Destination", row.get("D", "")),
            "Alpha": alpha,
            "Wmax": row.get("Wmax", ""),
            "Cmax": row.get("Cmax", ""),
            "w": row.get("WeightW", row.get("w", "")),
            "feasible": row.get("FeasibleParking", row.get("feasible", "")),
            "parking_node": row.get("ParkingNode", row.get("parking_node", "")),
            "drive_time_hr": row.get("drive_time_hr", ""),
            "walk_distance_km": row.get("walk_time", row.get("walk_distance_km", "")),
            "walk_time_hr": row.get("walk_time_hr", row.get("walk_time", "")),
            "parking_cost": row.get("parking_cost", ""),
            "objective": row.get("objective", ""),
            "runtime_ms": row.get("runtime_ms", ""),
            "drive_path": row.get("DrivePath", row.get("drive_path", "")),
            "walk_path": row.get("WalkPath", row.get("walk_path", "")),
        }
        return _sanitize_canonical_row(canon)
    if method == "aco":
        canon = {
            "method": row.get("method", "ACO"),
            "dat_file": normalize_case_name(row.get("Case", row.get("dat_file", ""))),
            "nodes": row.get("Nodes", row.get("nodes", "")),
            "edges": row.get("Edges", row.get("edges", "")),
            "parking_lots": row.get("ParkingLots", row.get("parking_lots", "")),
            "S": row.get("Source", row.get("S", "")),
            "D": row.get("Destination", row.get("D", "")),
            "Alpha": row.get("Alpha", ""),
            "Wmax": row.get("Wmax", ""),
            "Cmax": row.get("Cmax", ""),
            "w": row.get("WeightW", row.get("w", "")),
            "feasible": row.get("FeasibleParking", row.get("feasible", "")),
            "parking_node": row.get("ParkingNode", row.get("parking_node", "")),
            "drive_time_hr": row.get("drive_time_hr", ""),
            "walk_distance_km": row.get("walk_time", row.get("walk_distance_km", "")),
            "walk_time_hr": row.get("walk_time_hr", row.get("walk_time", "")),
            "parking_cost": row.get("parking_cost", ""),
            "objective": row.get("objective", ""),
            "runtime_ms": row.get("runtime_ms", ""),
            "drive_path": row.get("DrivePath", row.get("drive_path", "")),
            "walk_path": row.get("WalkPath", row.get("walk_path", "")),
        }
        return _sanitize_canonical_row(canon)
    raise ValueError(method)


def normalize_results_csv(method: str, results_path: Path):
    rows = []
    with results_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not any(str(v).strip() for v in row.values()):
                continue
            try:
                canon = canonical_row_from_results(method, row)
                canon["dat_file"] = normalize_case_name(canon["dat_file"])
                canon = _force_reconcile_row_with_dataset(canon)
                rows.append(canon)
            except Exception as ex:
                print(f"[WARN] Skipping row due to error: {ex}\\nRow: {row}")
    deduped = {}
    for row in rows:
        key = row.get("dat_file", None)
        if not key:
            print(f"[WARN] Row missing dat_file, skipping: {row}")
            continue
        deduped[key] = row
    rows = list(deduped.values())
    rows.sort(key=lambda r: parse_case_key(r["dat_file"]))
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def normalize_routes_csv(routes_path: Path):
    rows = []
    with routes_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not any(str(v).strip() for v in row.values()):
                continue
            normalized = {k: row.get(k, "") for k in ROUTE_HEADER}
            normalized["case"] = normalize_case_name(normalized["case"])
            normalized["feasible"] = _normalize_feasible(normalized.get("feasible", "0"))
            if normalized["feasible"] == "0":
                normalized["parking_node"] = ""
                normalized["drive_path"] = ""
                normalized["walk_path"] = ""
                normalized["full_route"] = ""
            rows.append(normalized)

    deduped = {}
    for row in rows:
        deduped[row["case"]] = row
    rows = list(deduped.values())
    rows.sort(key=lambda r: parse_case_key(r["case"]))

    with routes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROUTE_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _combine_route_text(drive_path: str, walk_path: str) -> str:
    drive_parts = [p for p in _text(drive_path).split("->") if p]
    walk_parts = [p for p in _text(walk_path).split("->") if p]
    if drive_parts and walk_parts and drive_parts[-1] == walk_parts[0]:
        walk_parts = walk_parts[1:]
    return "->".join(drive_parts + walk_parts)


def write_routes_from_results(method: str, results_path: Path, routes_path: Path):
    rows = []
    with results_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not any(_text(v) for v in row.values()):
                continue
            rows.append({
                "case": normalize_case_name(row.get("dat_file", row.get("Case", ""))),
                "parking_node": _text(row.get("parking_node", row.get("ParkingNode", ""))),
                "feasible": _normalize_feasible(row.get("feasible", row.get("Feasible", ""))),
                "drive_path": _text(row.get("drive_path", row.get("DrivePath", ""))),
                "walk_path": _text(row.get("walk_path", row.get("WalkPath", ""))),
                "full_route": _combine_route_text(
                    row.get("drive_path", row.get("DrivePath", "")),
                    row.get("walk_path", row.get("WalkPath", "")),
                ),
            })

    routes_path.parent.mkdir(parents=True, exist_ok=True)
    with routes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROUTE_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _method_short_name(method: str) -> str:
    return "heuristic" if method == "heuristics" else method


def _load_casewise_partition_rows(method: str, partition: str):
    casewise_root = method_data_root(method) / "case_wise" / partition
    method_short = _method_short_name(method)
    result_pattern = f"C*/results_{method_short}_C*.csv"
    route_pattern = f"C*/routes_{method_short}_C*.csv"
    if not casewise_root.exists():
        return None, None

    result_rows = []
    route_rows = []

    for path in sorted(casewise_root.glob(result_pattern)):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if any(_text(v) for v in row.values()) and _is_active_synthetic_dat_name(row.get("dat_file", "")):
                    result_rows.append(row)

    for path in sorted(casewise_root.glob(route_pattern)):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if any(_text(v) for v in row.values()) and _is_active_synthetic_dat_name(row.get("case", "")):
                    route_rows.append(row)

    if not result_rows:
        return None, None
    return result_rows, route_rows


def export_casewise_method_csvs(method: str, partition: str, results_path: Path, routes_path: Path):
    """
    Create method-wise case CSVs (C1..C10 when present) under each method folder:
      <method data>/<partition>/case_wise/Ck/results_<method>_Ck.csv
      <method data>/<partition>/case_wise/Ck/routes_<method>_Ck.csv
    """
    method_short = _method_short_name(method)
    out_root = method_data_root(method) / "case_wise" / partition
    out_root.mkdir(parents=True, exist_ok=True)

    with results_path.open(newline="", encoding="utf-8") as f:
        result_rows = [r for r in csv.DictReader(f) if any(_text(v) for v in r.values())]
    with routes_path.open(newline="", encoding="utf-8") as f:
        route_rows = [r for r in csv.DictReader(f) if any(_text(v) for v in r.values())]

    grouped_results = {k: [] for k in range(1, 11)}
    grouped_routes = {k: [] for k in range(1, 11)}

    for row in result_rows:
        case_id = parse_case_id(row.get("dat_file", ""))
        if case_id is not None and 1 <= case_id <= 10:
            grouped_results[case_id].append(row)

    for row in route_rows:
        case_id = parse_case_id(row.get("case", ""))
        if case_id is not None and 1 <= case_id <= 10:
            grouped_routes[case_id].append(row)

    for case_id in range(1, 11):
        case_dir = out_root / f"C{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)

        case_results = grouped_results[case_id]
        case_routes = grouped_routes[case_id]
        case_results.sort(key=lambda r: parse_case_key(r.get("dat_file", "")))
        case_routes.sort(key=lambda r: parse_case_key(r.get("case", "")))

        out_results = case_dir / f"results_{method_short}_C{case_id}.csv"
        out_routes = case_dir / f"routes_{method_short}_C{case_id}.csv"

        with out_results.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANONICAL_RESULTS_HEADER)
            writer.writeheader()
            writer.writerows(case_results)

        with out_routes.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ROUTE_HEADER)
            writer.writeheader()
            writer.writerows(case_routes)


def run_method_on_dat(method: str, dat_path: Path, results_path: Path, cmax: float):
    cfg = method_config(method, partition="")
    cmd = [
        sys.executable,
        str(cfg["script"]),
        "--dat",
        str(dat_path),
        "--out",
        str(results_path),
    ]
    if cmax is not None:
        cmd.extend(["--cmax", str(cmax)])
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Subprocess failed for {method} on {dat_path}: {e}")
        print(e.output)
        raise


def run_method_on_dat_dir(method: str, dat_dir: Path, results_path: Path, cmax: float):
    cfg = method_config(method, partition="")
    cmd = [
        sys.executable,
        str(cfg["script"]),
        "--dat-dir",
        str(dat_dir),
        "--out",
        str(results_path),
    ]
    if cmax is not None:
        cmd.extend(["--cmax", str(cmax)])
    subprocess.run(cmd, check=True, cwd=ROOT)


def _ensure_clean_outputs(results_path: Path, routes_path: Path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()
    if routes_path.exists():
        routes_path.unlink()


def run_partitioned_c1(cmax: float):
    node_by_partition = {
        "small_scale": SMALL_SCALE_NODES,
        "medium_large_scale": MEDIUM_LARGE_NODES,
    }

    for partition in PARTITION_ORDER:
        node_sizes = node_by_partition[partition]
        dat_paths = []
        for n in node_sizes:
            path = FINAL_CASES / f"n_{n}" / f"N{n}_C1_I1.dat"
            if not path.exists():
                raise FileNotFoundError(f"Missing dataset: {path}")
            dat_paths.append(path)

        for method in METHOD_ORDER:
            cfg = method_config(method, partition)
            _ensure_clean_outputs(cfg["results"], cfg["routes"])

            for dat_path in dat_paths:
                print(f"Running {method} on {dat_path.name} -> {partition}", flush=True)
                run_method_on_dat(method, dat_path, cfg["results"], cmax)

            normalize_results_csv(method, cfg["results"])
            normalize_routes_csv(cfg["routes"])
            export_casewise_method_csvs(method, partition, cfg["results"], cfg["routes"])
            print(f"Normalized {cfg['results']}", flush=True)
            print(f"Normalized {cfg['routes']}", flush=True)

    print("Partitioned C1 run complete.", flush=True)


def run_final_cases(cmax: float):
    if not FINAL_CASES.exists():
        raise RuntimeError(f"Missing dataset directory: {FINAL_CASES}")

    dat_paths = sorted(
        path for path in FINAL_CASES.rglob("N*_C*_I*.dat")
        if _is_active_synthetic_dat_path(path)
    )
    if not dat_paths:
        raise RuntimeError(f"No .dat files found under {FINAL_CASES}")

    for method in METHOD_ORDER:
        cfg = method_config(method, "final_cases")
        _ensure_clean_outputs(cfg["results"], cfg["routes"])
        for dat_path in dat_paths:
            print(f"Running {method} on {dat_path.name} -> final_cases", flush=True)
            run_method_on_dat(method, dat_path, cfg["results"], cmax)
        normalize_results_csv(method, cfg["results"])
        write_routes_from_results(method, cfg["results"], cfg["routes"])
        normalize_routes_csv(cfg["routes"])
        export_casewise_method_csvs(method, "final_cases", cfg["results"], cfg["routes"])
        print(f"Normalized {cfg['results']}", flush=True)
        print(f"Normalized {cfg['routes']}", flush=True)

    write_combined_final_cases()
    print(f"Updated combined files in: {COMBINED_DIR}", flush=True)


def run_sumo_cases(cmax: float):
    if not SUMO_CASES.exists():
        raise RuntimeError(f"Missing SUMO dataset directory: {SUMO_CASES}")

    dat_paths = sorted(SUMO_CASES.glob("N*_C*_I*.dat"))
    if not dat_paths:
        raise RuntimeError(f"No SUMO .dat files found under {SUMO_CASES}")

    for method in METHOD_ORDER:
        cfg = method_config(method, "sumo")
        _ensure_clean_outputs(cfg["results"], cfg["routes"])
        for dat_path in dat_paths:
            print(f"Running {method} on {dat_path.name} -> sumo", flush=True)
            run_method_on_dat(method, dat_path, cfg["results"], cmax)
        normalize_results_csv(method, cfg["results"])
        write_routes_from_results(method, cfg["results"], cfg["routes"])
        normalize_routes_csv(cfg["routes"])
        print(f"Normalized {cfg['results']}", flush=True)
        print(f"Normalized {cfg['routes']}", flush=True)


def run_all_node_major(cmax: float):
    if not FINAL_CASES.exists():
        raise RuntimeError(f"Missing dataset directory: {FINAL_CASES}")

    all_nodes = SMALL_SCALE_NODES + MEDIUM_LARGE_NODES
    
    # Initialize/Clean outputs first
    for method in METHOD_ORDER:
        cfg = method_config(method, "final_cases")
        _ensure_clean_outputs(cfg["results"], cfg["routes"])

    for n in all_nodes:
        dat_dir = FINAL_CASES / f"n_{n}"
        if not dat_dir.exists():
            print(f"[WARN] Missing node directory: {dat_dir}")
            continue
        
        dat_paths = sorted(dat_dir.glob("*.dat"))
        if not dat_paths:
            continue

        print(f"\n--- STARTING NODE SIZE N={n} ---", flush=True)
        for method in METHOD_ORDER:
            cfg = method_config(method, "final_cases")
            print(f"  > Method: {method.upper()} (N={n})", flush=True)
            for dat_path in dat_paths:
                run_method_on_dat(method, dat_path, cfg["results"], cmax)
            
            # Incremental normalization and route writing
            normalize_results_csv(method, cfg["results"])
            write_routes_from_results(method, cfg["results"], cfg["routes"])
            normalize_routes_csv(cfg["routes"])
            export_casewise_method_csvs(method, "final_cases", cfg["results"], cfg["routes"])

    print("\n--- FINALIZING COMBINED RESULTS ---", flush=True)
    write_combined_final_cases()
    print(f"Updated combined files in: {COMBINED_DIR}", flush=True)


def write_combined_final_cases():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    all_result_rows = []
    all_route_rows = []

    for method in METHOD_ORDER:
        cfg = method_config(method, "final_cases")
        method_name = "heuristic" if method == "heuristics" else method
        casewise_results, casewise_routes = _load_casewise_partition_rows(method, "final_cases")

        result_rows = []
        if casewise_results is not None:
            for row in casewise_results:
                canon = {k: row.get(k, "") for k in CANONICAL_RESULTS_HEADER}
                canon["dat_file"] = normalize_case_name(canon.get("dat_file", ""))
                if not _is_active_synthetic_dat_name(canon["dat_file"]):
                    continue
                canon = _force_reconcile_row_with_dataset(canon)
                result_rows.append(canon)
            route_rows = casewise_routes or []
        else:
            with cfg["results"].open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not any(_text(v) for v in row.values()):
                        continue
                    canon = {k: row.get(k, "") for k in CANONICAL_RESULTS_HEADER}
                    canon["dat_file"] = normalize_case_name(canon.get("dat_file", ""))
                    if not _is_active_synthetic_dat_name(canon["dat_file"]):
                        continue
                    canon = _force_reconcile_row_with_dataset(canon)
                    result_rows.append(canon)

            with cfg["routes"].open(newline="", encoding="utf-8") as f:
                route_rows = [
                    r for r in csv.DictReader(f)
                    if any(_text(v) for v in r.values()) and _is_active_synthetic_dat_name(r.get("case", ""))
                ]

        route_by_case = {
            normalize_case_name(r.get("case", "")): r
            for r in route_rows
            if normalize_case_name(r.get("case", ""))
        }

        for i, row in enumerate(result_rows):
            route_row = route_by_case.get(row["dat_file"])
            result_rows[i] = _force_reconcile_row_with_dataset(row, route_row)

        dedup_results = {}
        for row in result_rows:
            dedup_results[row["dat_file"]] = _force_reconcile_row_with_dataset(
                row, route_by_case.get(row["dat_file"])
            )
        result_rows = list(dedup_results.values())
        result_rows.sort(key=lambda r: parse_case_key(r["dat_file"]))

        dedup_routes = {}
        for row in route_rows:
            key = normalize_case_name(row.get("case", ""))
            if not key:
                continue
            rr = {k: row.get(k, "") for k in ROUTE_HEADER}
            rr["case"] = key
            if not _is_active_synthetic_dat_name(rr["case"]):
                continue
            rr["feasible"] = _normalize_feasible(rr.get("feasible", "0"))
            if rr["feasible"] == "0":
                rr["parking_node"] = ""
                rr["drive_path"] = ""
                rr["walk_path"] = ""
                rr["full_route"] = ""
            dedup_routes[key] = rr
        route_rows = list(dedup_routes.values())
        route_rows.sort(key=lambda r: parse_case_key(r["case"]))

        result_out = COMBINED_DIR / f"results_{method_name}_final_cases.csv"
        with result_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANONICAL_RESULTS_HEADER)
            writer.writeheader()
            writer.writerows(result_rows)

        route_header = ROUTE_HEADER + ["method"]
        route_rows_with_method = []
        for row in route_rows:
            r = dict(row)
            r["method"] = method_name.upper()
            route_rows_with_method.append(r)

        route_out = COMBINED_DIR / f"routes_{method_name}_final_cases.csv"
        with route_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=route_header)
            writer.writeheader()
            writer.writerows(route_rows_with_method)

        all_result_rows.extend(result_rows)
        all_route_rows.extend(route_rows_with_method)

    dedup_results = {}
    for row in all_result_rows:
        dedup_results[(row["method"], row["dat_file"])] = _force_reconcile_row_with_dataset(row)
    all_result_rows = list(dedup_results.values())
    all_result_rows.sort(key=lambda r: (parse_case_key(r["dat_file"]), r["method"]))

    dedup_routes = {}
    for row in all_route_rows:
        dedup_routes[(row["method"], row["case"])] = row
    all_route_rows = list(dedup_routes.values())
    all_route_rows.sort(key=lambda r: (parse_case_key(r["case"]), r["method"]))

    out_results = COMBINED_DIR / "results_all_methods_final_cases.csv"
    with out_results.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(all_result_rows)

    out_routes = COMBINED_DIR / "routes_all_methods_final_cases.csv"
    route_header = ROUTE_HEADER + ["method"]
    with out_routes.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=route_header)
        writer.writeheader()
        writer.writerows(all_route_rows)

    result_partitions = _partition_case_rows(all_result_rows, "dat_file")
    route_partitions = _partition_case_rows(all_route_rows, "case")

    for partition in ("small_scale", "medium_large_scale"):
        part_results = COMBINED_DIR / f"results_all_methods_{partition}.csv"
        with part_results.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANONICAL_RESULTS_HEADER)
            writer.writeheader()
            writer.writerows(result_partitions[partition])

        part_routes = COMBINED_DIR / f"routes_all_methods_{partition}.csv"
        with part_routes.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=route_header)
            writer.writeheader()
            writer.writerows(route_partitions[partition])

def merge_partitioned_method_data():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    result_files = {
        "heuristic": [
            ROOT / "heuristics" / "data" / "scale_wise_evaluation" / "small_scale" / "results_heuristic.csv",
            ROOT / "heuristics" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "results_heuristic.csv",
        ],
        "ga": [
            ROOT / "ga" / "data" / "scale_wise_evaluation" / "small_scale" / "results_ga.csv",
            ROOT / "ga" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "results_ga.csv",
        ],
        "pso": [
            ROOT / "pso" / "data" / "scale_wise_evaluation" / "small_scale" / "results_pso.csv",
            ROOT / "pso" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "results_pso.csv",
        ],
        "aco": [
            ROOT / "aco" / "data" / "scale_wise_evaluation" / "small_scale" / "results_aco.csv",
            ROOT / "aco" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "results_aco.csv",
        ],
    }

    route_files = {
        "heuristic": [
            ROOT / "heuristics" / "data" / "scale_wise_evaluation" / "small_scale" / "routes_heuristic.csv",
            ROOT / "heuristics" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "routes_heuristic.csv",
        ],
        "ga": [
            ROOT / "ga" / "data" / "scale_wise_evaluation" / "small_scale" / "routes_ga.csv",
            ROOT / "ga" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "routes_ga.csv",
        ],
        "pso": [
            ROOT / "pso" / "data" / "scale_wise_evaluation" / "small_scale" / "routes_pso.csv",
            ROOT / "pso" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "routes_pso.csv",
        ],
        "aco": [
            ROOT / "aco" / "data" / "scale_wise_evaluation" / "small_scale" / "routes_aco.csv",
            ROOT / "aco" / "data" / "scale_wise_evaluation" / "medium_large_scale" / "routes_aco.csv",
        ],
    }

    all_result_rows = []
    all_route_rows = []

    for method in ["heuristic", "ga", "pso", "aco"]:
        rows = []
        for path in result_files[method]:
            part = path.parent.name
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    r = dict(row)
                    if not _is_active_synthetic_dat_name(r.get("dat_file", "")):
                        continue
                    r["scale_partition"] = part
                    rows.append(r)
        if rows:
            dedup = {r["dat_file"]: r for r in rows}
            merged = list(dedup.values())
            merged.sort(key=lambda r: parse_case_key(r["dat_file"]))
            out = COMBINED_DIR / f"results_{method}_combined.csv"
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
                writer.writeheader()
                writer.writerows(merged)
            all_result_rows.extend(merged)

        r_rows = []
        for path in route_files[method]:
            part = path.parent.name
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    r = dict(row)
                    if not _is_active_synthetic_dat_name(r.get("case", "")):
                        continue
                    r["scale_partition"] = part
                    r["method"] = method.upper()
                    r_rows.append(r)
        if r_rows:
            dedup = {r["case"]: r for r in r_rows}
            merged = list(dedup.values())
            merged.sort(key=lambda r: parse_case_key(r["case"]))
            out = COMBINED_DIR / f"routes_{method}_combined.csv"
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
                writer.writeheader()
                writer.writerows(merged)
            all_route_rows.extend(merged)

    if all_result_rows:
        all_result_rows.sort(key=lambda r: (parse_case_key(r["dat_file"]), r["method"]))
        out = COMBINED_DIR / "results_all_methods_combined.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_result_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_result_rows)

    if all_route_rows:
        all_route_rows.sort(key=lambda r: (parse_case_key(r["case"]), r["method"]))
        out = COMBINED_DIR / "routes_all_methods_combined.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_route_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_route_rows)

    print(f"Wrote merged files to: {COMBINED_DIR}", flush=True)


def _detect_legacy_method(path: Path) -> str:
    text = path.as_posix()
    if text.startswith("heuristics/"):
        return "heuristics"
    if text.startswith("ga/"):
        return "ga"
    if text.startswith("pso/"):
        return "pso"
    if text.startswith("aco/"):
        return "aco"
    raise ValueError(f"Cannot detect method for {path}")


def _expected_legacy_header(path: Path):
    if "routes_" in path.name:
        return ROUTE_HEADER
    return LEGACY_RESULT_HEADERS[_detect_legacy_method(path)]


def _looks_like_header(row, expected):
    return [x.strip() for x in row] == [x.strip() for x in expected]


def _case_text_from_legacy_row(path: Path, row: dict) -> str:
    if "routes_" in path.name:
        return _text(row.get("case", ""))
    if _detect_legacy_method(path) == "heuristics":
        return _text(row.get("Case", ""))
    return _text(row.get("dat_file", ""))


def _row_as_expected_dict(expected, raw_row):
    out = {}
    for i, col in enumerate(expected):
        out[col] = raw_row[i].strip() if i < len(raw_row) else ""
    return out


def normalize_legacy_files():
    processed = []

    for rel_path in LEGACY_CSV_FILES:
        path = ROOT / rel_path
        expected = _expected_legacy_header(rel_path)
        if not path.exists():
            continue

        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            data_rows = []
        else:
            start = 1 if _looks_like_header(rows[0], expected) else 0
            deduped = {}
            for raw in rows[start:]:
                if not any(cell.strip() for cell in raw):
                    continue
                row = _row_as_expected_dict(expected, raw)
                key = _case_text_from_legacy_row(rel_path, row)
                deduped[key] = row
            data_rows = list(deduped.values())
            data_rows.sort(key=lambda r: parse_case_key(_case_text_from_legacy_row(rel_path, r)))

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=expected)
            writer.writeheader()
            for row in data_rows:
                writer.writerow({c: row.get(c, "") for c in expected})

        processed.append((rel_path.as_posix(), len(data_rows)))

    for p, n in processed:
        print(f"{p}: {n} data rows", flush=True)


def export_casewise_from_existing():
    for method in METHOD_ORDER:
        for partition in PARTITION_ORDER + ["final_cases", "sumo"]:
            cfg = method_config(method, partition)
            if cfg["results"].exists() and cfg["routes"].exists():
                export_casewise_method_csvs(method, partition, cfg["results"], cfg["routes"])
                print(f"Case-wise exported: {method_data_root(method) / 'case_wise' / partition}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Unified data logging pipeline for ITS method outputs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("run-partitioned-c1", help="Run all methods for C1 partitioned nodes and normalize logs.")
    p1.add_argument("--cmax", type=float, default=None)
    p2 = sub.add_parser("run-final-cases", help="Run all methods on final_cases dir and rebuild combined final-cases files.")
    p2.add_argument("--cmax", type=float, default=None)
    p2s = sub.add_parser("run-sumo-cases", help="Run all methods on final_cases/SUMO and write normalized CSVs under each method/data/sumo.")
    p2s.add_argument("--cmax", type=float, default=None)
    p3 = sub.add_parser("run-all-node-major", help="Run all methods node-by-node (N=10..2500 active synthetic nodes only) and rebuild combined files.")
    p3.add_argument("--cmax", type=float, default=None)
    sub.add_parser("merge-partitioned", help="Merge existing small/medium partition logs into combined outputs.")
    sub.add_parser("normalize-legacy", help="Header+dedupe+sort pass on legacy method CSV logs.")
    sub.add_parser("export-casewise", help="Create C1..C10 case-wise method CSVs from existing logs.")
    args = parser.parse_args()
    try:
        if args.cmd == "run-partitioned-c1":
            run_partitioned_c1(args.cmax)
        elif args.cmd == "run-final-cases":
            run_final_cases(args.cmax)
        elif args.cmd == "run-sumo-cases":
            run_sumo_cases(args.cmax)
        elif args.cmd == "run-all-node-major":
            run_all_node_major(args.cmax)
        elif args.cmd == "merge-partitioned":
            merge_partitioned_method_data()
        elif args.cmd == "normalize-legacy":
            normalize_legacy_files()
        elif args.cmd == "export-casewise":
            export_casewise_from_existing()
        else:
            raise RuntimeError(f"Unknown command: {args.cmd}")
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
