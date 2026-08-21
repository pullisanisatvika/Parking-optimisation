#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lot_objective_utils import parse_lot_dat_for_normalization, raw_lot_objective


ROOT = Path(__file__).resolve().parents[1]
OPT_DIR = ROOT / "OPTIMIZATION"
MODEL_PATH = OPT_DIR / "Satvika1.mod"
SUMMARY_CSV = OPT_DIR / "optimisation_results_summary.csv"
SOURCE_CASES_DIR = ROOT / "final_cases"
CONVERTED_CASES_DIR = OPT_DIR / "final_cases_opl"

CASE_METADATA = {
    "C1": ("Baseline", "baseline", "Baseline"),
    "C2": ("Dense", "edge_density", "Dense"),
    "C3": ("Higher supply", "parking_supply", "Higher"),
    "C4": ("1.00 km", "maximum_walking_distance", "1.0"),
    "C5": ("Medium traffic", "traffic_level", "Medium"),
    "C6": ("High traffic", "traffic_level", "High"),
    "C7": ("Alpha 0.80", "objective_weight_alpha", "0.8"),
    "C8": ("Alpha 0.20", "objective_weight_alpha", "0.2"),
    "C9": ("Cmax 15.0", "parking_budget_cmax", "15.0"),
    "C10": ("15.0-30.0", "parking_cost_variation_mode", "15.0-30.0"),
}

ALLOWED_DATA_KEYS = [
    "Nodes",
    "E",
    "S",
    "D",
    "Q",
    "L",
    "C",
    "B",
    "Alpha",
    "W_km",
    "Cmax",
    "Distance",
    "EdgeTime",
]

SUMMARY_HEADER = [
    "scenario_id",
    "scenario_name",
    "parameter_name",
    "parameter_value",
    "method",
    "dat_file",
    "nodes",
    "edges",
    "parking_lots",
    "source",
    "destination",
    "S",
    "D",
    "Alpha",
    "Wmax",
    "Cmax",
    "w",
    "feasible",
    "solver_status",
    "parking_node",
    "driving_path",
    "walking_edge",
    "full_path",
    "selected_edges",
    "drive_time_hr",
    "walk_distance_km",
    "walk_time_hr",
    "travel_time_hr",
    "parking_cost",
    "objective",
    "path_reconstruction_status",
    "runtime_ms",
    "drive_path",
    "walk_path",
]


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def split_statements(text: str) -> list[str]:
    cleaned = strip_comments(text)
    statements: list[str] = []
    current: list[str] = []
    depth_square = 0
    depth_curly = 0
    depth_angle = 0
    in_string = False

    for ch in cleaned:
        current.append(ch)
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
        elif ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle -= 1
        elif ch == ";" and depth_square == 0 and depth_curly == 0 and depth_angle == 0:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def parse_dat_assignments(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    assignments: dict[str, str] = {}
    for statement in split_statements(text):
        if "=" not in statement:
            continue
        key, value = statement.split("=", 1)
        assignments[key.strip()] = value.strip().rstrip(";").strip()
    assignments["__raw_text__"] = text
    return assignments


def count_edges(edge_expr: str) -> int:
    return len(re.findall(r"<\s*\d+\s*,\s*\d+\s*>", edge_expr))


def scalar_value(raw: str) -> str:
    return raw.strip().strip('"')


def format_optional_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def parse_case_code(dat_name: str) -> str:
    match = re.search(r"_C(\d+)", dat_name)
    if not match:
        return "C1"
    return f"C{int(match.group(1))}"


def case_metadata(dat_name: str) -> tuple[str, str, str, str]:
    case_id = parse_case_code(dat_name)
    scenario_name, parameter_name, parameter_value = CASE_METADATA.get(
        case_id,
        ("Baseline", "baseline", "Baseline"),
    )
    return case_id, scenario_name, parameter_name, parameter_value


def write_converted_dat(source_path: Path, target_path: Path) -> dict[str, str]:
    assignments = parse_dat_assignments(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "/*********************************************",
        " * Converted final-case dataset for Satvika1.mod",
        f" * Source: {source_path.relative_to(ROOT)}",
        " *********************************************/",
        "",
    ]
    for key in ALLOWED_DATA_KEYS:
        if key in assignments:
            lines.append(f"{key} = {assignments[key]};")
            lines.append("")

    target_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return assignments


def find_oplrun(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return shutil.which("oplrun")


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_summary_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def dat_needs_run(dat_name: str, current_rows: list[dict[str, str]]) -> bool:
    matching = [row for row in current_rows if row.get("dat_file") == dat_name]
    if not matching:
        return True
    return not any(str(row.get("feasible", "")).upper() == "TRUE" for row in matching)


def parse_selected_edges(stdout: str) -> list[tuple[int, int]]:
    return [
        (int(a), int(b))
        for a, b in re.findall(r">>> X\[(\d+)->(\d+)\]", stdout)
    ]


def extract_float(stdout: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.+-eE]+)", stdout)
    return float(match.group(1)) if match else None


def extract_int(stdout: str, label: str) -> int | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", stdout)
    return int(match.group(1)) if match else None


def selected_edges_text(edge_list: list[tuple[int, int]]) -> str:
    return ";".join(f"{a}->{b}" for a, b in edge_list)


def build_paths(
    edge_list: list[tuple[int, int]],
    source: int,
    parking_node: int,
    dest: int,
) -> tuple[str, str, str, str, str]:
    succ: dict[int, int] = {}
    for a, b in edge_list:
        if a not in succ:
            succ[a] = b

    walking_edge = f"{parking_node}->{dest}"
    ordered_nodes = [source]
    seen = {source}
    current = source
    reconstruction_status = "failed"

    while current in succ:
        nxt = succ[current]
        ordered_nodes.append(nxt)
        if nxt == parking_node:
            reconstruction_status = "success"
            break
        if nxt in seen:
            break
        seen.add(nxt)
        current = nxt

    drive_edges = ";".join(f"{ordered_nodes[i]}->{ordered_nodes[i + 1]}" for i in range(len(ordered_nodes) - 1))
    driving_path = "->".join(str(node) for node in ordered_nodes)
    if reconstruction_status == "success":
        full_path = "->".join(str(node) for node in ordered_nodes + [dest])
    else:
        full_path = ""
    return drive_edges, driving_path, walking_edge, full_path, reconstruction_status


def infer_solver_status(returncode: int, stdout: str, stderr: str, timed_out: bool = False) -> str:
    if timed_out:
        return "time_limit"
    combined = f"{stdout}\n{stderr}".lower()
    if returncode == 0 and "parking node:" in combined:
        return "solved"
    if "memory" in combined or "out of memory" in combined:
        return "memory_limit"
    if "time limit" in combined or "timed out" in combined:
        return "time_limit"
    if "infeasible" in combined or "no solution" in combined:
        return "no_solution"
    return "solver_error"


def base_summary_row(dat_name: str, assignments: dict[str, str]) -> dict[str, str]:
    case_id, scenario_name, parameter_name, parameter_value = case_metadata(dat_name)
    nodes = scalar_value(assignments.get("Nodes", ""))
    source = scalar_value(assignments.get("S", ""))
    dest = scalar_value(assignments.get("D", ""))
    return {
        "scenario_id": case_id,
        "scenario_name": scenario_name,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "method": "Optimization",
        "dat_file": dat_name,
        "nodes": nodes,
        "edges": str(count_edges(assignments["E"])) if "E" in assignments else "",
        "parking_lots": scalar_value(assignments.get("Q", "")),
        "source": source,
        "destination": dest,
        "S": source,
        "D": dest,
        "Alpha": scalar_value(assignments.get("Alpha", "")),
        "Wmax": scalar_value(assignments.get("W_km", "")),
        "Cmax": scalar_value(assignments.get("Cmax", "")),
        "w": scalar_value(assignments.get("W", "")),
        "feasible": "FALSE",
        "solver_status": "",
        "parking_node": "",
        "driving_path": "",
        "walking_edge": "",
        "full_path": "",
        "selected_edges": "",
        "drive_time_hr": "",
        "walk_distance_km": "",
        "walk_time_hr": "",
        "travel_time_hr": "",
        "parking_cost": "",
        "objective": "",
        "path_reconstruction_status": "",
        "runtime_ms": "",
        "drive_path": "",
        "walk_path": "",
    }


def build_summary_row(
    dat_name: str,
    assignments: dict[str, str],
    stdout: str,
    runtime_ms: int | None,
    solver_status: str,
) -> dict[str, str]:
    row = base_summary_row(dat_name, assignments)
    source = int(scalar_value(assignments["S"]))
    dest = int(scalar_value(assignments["D"]))
    parking_node = extract_int(stdout, "Parking node")
    parking_cost = extract_float(stdout, "Parking cost")
    drive_time = extract_float(stdout, "Driving time to parking lot")
    walk_distance = extract_float(stdout, "Walking distance to destination")
    walk_time = extract_float(stdout, "Walking time to destination")

    if parking_node is None or parking_cost is None or drive_time is None or walk_distance is None or walk_time is None:
        raise ValueError(f"Could not parse full Optimization output for {dat_name}")

    travel_time = drive_time + walk_time
    parse_lot_dat_for_normalization(assignments["__raw_text__"])
    objective = raw_lot_objective(
        drive_time * 60.0,
        walk_time * 60.0,
        parking_cost,
        weight_w=float(scalar_value(assignments.get("Alpha", "0.9"))),
    )
    selected_edges = parse_selected_edges(stdout)
    drive_path, driving_path, walking_edge, full_path, path_status = build_paths(
        selected_edges,
        source,
        parking_node,
        dest,
    )

    row.update(
        {
            "feasible": "TRUE",
            "solver_status": solver_status,
            "parking_node": str(parking_node),
            "driving_path": driving_path,
            "walking_edge": walking_edge,
            "full_path": full_path,
            "selected_edges": selected_edges_text(selected_edges),
            "drive_time_hr": format_optional_number(drive_time),
            "walk_distance_km": format_optional_number(walk_distance),
            "walk_time_hr": format_optional_number(walk_time),
            "travel_time_hr": format_optional_number(travel_time),
            "parking_cost": format_optional_number(parking_cost),
            "objective": format_optional_number(objective),
            "path_reconstruction_status": path_status,
            "runtime_ms": format_optional_number(runtime_ms),
            "drive_path": drive_path,
            "walk_path": walking_edge,
        }
    )
    return row


def build_unsolved_row(
    dat_name: str,
    assignments: dict[str, str],
    solver_status: str,
    runtime_ms: int | None,
) -> dict[str, str]:
    row = base_summary_row(dat_name, assignments)
    row["solver_status"] = solver_status
    row["runtime_ms"] = format_optional_number(runtime_ms)
    return row


def upsert_summary_row(rows: list[dict[str, str]], new_row: dict[str, str]) -> list[dict[str, str]]:
    replaced = False
    updated: list[dict[str, str]] = []
    for row in rows:
        if row.get("dat_file") == new_row["dat_file"]:
            updated.append(new_row)
            replaced = True
        else:
            updated.append(row)
    if not replaced:
        updated.append(new_row)
    return updated


def convert_all_cases(source_dir: Path, target_dir: Path) -> dict[str, dict[str, str]]:
    converted: dict[str, dict[str, str]] = {}
    for source_path in sorted(source_dir.rglob("*.dat")):
        relative = source_path.relative_to(source_dir)
        target_path = target_dir / relative
        converted[source_path.name] = write_converted_dat(source_path, target_path)
    return converted


def run_missing_cases(
    converted_dir: Path,
    summary_path: Path,
    model_path: Path,
    oplrun_path: str | None,
    include_patterns: list[str],
    timeout_s: int | None,
    collect_runtime: bool,
) -> tuple[int, int, list[dict[str, str]], list[dict[str, str]]]:
    oplrun = find_oplrun(oplrun_path)
    if not oplrun:
        print("`oplrun` was not found on PATH. Converted files were created, but no Optimization runs were executed.")
        return 0, 0, [], []

    rows = read_summary_rows(summary_path)
    attempted = 0
    succeeded = 0
    solved_rows: list[dict[str, str]] = []
    unsolved_rows: list[dict[str, str]] = []

    for dat_path in sorted(converted_dir.rglob("*.dat")):
        dat_name = dat_path.name
        if include_patterns and not any(pattern in dat_name for pattern in include_patterns):
            continue
        if not dat_needs_run(dat_name, rows):
            continue

        assignments = parse_dat_assignments(dat_path)
        attempted += 1
        print(f"Running Optimization on {dat_name}")
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [oplrun, str(model_path), str(dat_path)],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            runtime_ms = int(round((time.perf_counter() - start) * 1000)) if collect_runtime else None
            row = build_unsolved_row(dat_name, assignments, "time_limit", runtime_ms)
            rows = upsert_summary_row(rows, row)
            write_summary_rows(summary_path, rows)
            unsolved_rows.append(row)
            print(f"  solver_status=time_limit")
            continue
        runtime_ms = int(round((time.perf_counter() - start) * 1000)) if collect_runtime else None
        solver_status = infer_solver_status(result.returncode, result.stdout, result.stderr)

        if result.returncode != 0:
            row = build_unsolved_row(dat_name, assignments, solver_status, runtime_ms)
            rows = upsert_summary_row(rows, row)
            write_summary_rows(summary_path, rows)
            unsolved_rows.append(row)
            print(f"  failed with return code {result.returncode}")
            print(f"  solver_status={solver_status}")
            if result.stderr.strip():
                print(result.stderr.strip())
            continue

        try:
            row = build_summary_row(dat_name, assignments, result.stdout, runtime_ms, solver_status)
        except Exception as exc:
            fallback_status = "no_solution" if solver_status == "solved" else solver_status
            row = build_unsolved_row(dat_name, assignments, fallback_status, runtime_ms)
            rows = upsert_summary_row(rows, row)
            write_summary_rows(summary_path, rows)
            unsolved_rows.append(row)
            print(f"  could not parse output for {dat_name}: {exc}")
            print(f"  solver_status={fallback_status}")
            continue

        rows = upsert_summary_row(rows, row)
        write_summary_rows(summary_path, rows)
        succeeded += 1
        solved_rows.append(row)
        print(f"  logged result for {dat_name}")
        print(
            f"  parking_node={row['parking_node']} driving_path={row['driving_path']} "
            f"walking_edge={row['walking_edge']}"
        )

    return attempted, succeeded, solved_rows, unsolved_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert final_cases .dat files into Satvika1.mod-compatible OPL inputs and run missing Optimization cases."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_CASES_DIR)
    parser.add_argument("--output-dir", type=Path, default=CONVERTED_CASES_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert final-case .dat files; do not attempt to run OPL.",
    )
    parser.add_argument(
        "--oplrun",
        default=None,
        help="Absolute path to oplrun. If omitted, PATH lookup is used.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Only run dat files whose names contain one of these substrings.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=None,
        help="Optional per-case timeout in seconds for OPL execution.",
    )
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Do not collect wall-clock runtime_ms during OPL execution.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.source.exists():
        print(f"Missing source directory: {args.source}", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"Missing model file: {args.model}", file=sys.stderr)
        return 1

    converted = convert_all_cases(args.source, args.output_dir)
    print(f"Converted {len(converted)} .dat files into {args.output_dir}")

    if args.convert_only:
        return 0

    attempted, succeeded, solved_rows, unsolved_rows = run_missing_cases(
        args.output_dir,
        args.summary,
        args.model,
        args.oplrun,
        args.include,
        args.timeout_s,
        not args.skip_runtime,
    )
    print(f"Attempted Optimization runs: {attempted}")
    print(f"Successfully logged runs: {succeeded}")
    print("Solved cases:")
    for row in solved_rows:
        print(
            f"  {row['dat_file']} -> parking_node={row['parking_node']}, "
            f"driving_path={row['driving_path']}, walking_edge={row['walking_edge']}"
        )
    print("Unsolved or unavailable cases:")
    for row in unsolved_rows:
        print(f"  {row['dat_file']} -> {row['solver_status']}")
    print(f"Normalized Optimization CSV: {args.summary}")
    print(f"runtime_ms collection: {'skipped' if args.skip_runtime else 'enabled'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
