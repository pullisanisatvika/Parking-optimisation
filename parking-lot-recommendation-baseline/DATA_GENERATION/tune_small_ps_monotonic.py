#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "final_cases" / "parameter_sensitivity" / "small"
OPL_DIR = ROOT / "OPTIMIZATION" / "final_cases_opl" / "parameter_sensitivity" / "small"

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

WALKING_TUNES = {
    "N40_PS_walking_distance_W025_I1.dat": {"costs": ["19.16", "19.34"], "walk_dist": ["0.12", "0.30"], "walk_time": ["0.024", "0.06"]},
    "N40_PS_walking_distance_W050_I1.dat": {"costs": ["19.16", "15.0"], "walk_dist": ["0.12", "0.30"], "walk_time": ["0.024", "0.03"]},
    "N40_PS_walking_distance_W075_I1.dat": {"costs": ["19.16", "12.0"], "walk_dist": ["0.12", "0.28"], "walk_time": ["0.024", "0.028"]},
    "N40_PS_walking_distance_W100_I1.dat": {"costs": ["19.16", "9.0"], "walk_dist": ["0.12", "0.26"], "walk_time": ["0.024", "0.026"]},
}

PARKING_COST_TUNES = {
    "N40_PS_parking_cost_80_205_I1.dat": {"costs": ["8.5", "9.5"]},
    "N40_PS_parking_cost_100_225_I1.dat": {"costs": ["10.5", "11.5"]},
    "N40_PS_parking_cost_125_250_I1.dat": {"costs": ["12.5", "13.5"]},
    "N40_PS_parking_cost_150_300_I1.dat": {"costs": ["15.0", "16.0"]},
}


def replace_list_line(text: str, label: str, values: list[str]) -> str:
    pattern = rf"^{re.escape(label)}\s*=\s*\[[^\]]*\];(.*)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return text
    suffix = match.group(1)
    replacement = f"{label} = [{','.join(values)}];{suffix}"
    return re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)


def parse_edges(text: str) -> list[tuple[int, int]]:
    match = re.search(r"\bE\s*=\s*\{(.*?)\};", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Missing E block")
    return [(int(a), int(b)) for a, b in re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", match.group(1))]


def parse_numeric_block(text: str, label: str) -> list[str]:
    match = re.search(rf"\b{label}\s*=\s*\[(.*?)\];", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Missing {label} block")
    return re.findall(FLOAT_RE, match.group(1))


def rewrite_numeric_block(text: str, label: str, values: list[str]) -> str:
    lines: list[str] = []
    for idx in range(0, len(values), 4):
        chunk = values[idx: idx + 4]
        lines.append(" , ".join(chunk) + " ,")
    block = f"{label} = [\n" + "\n".join(lines) + "\n];"
    return re.sub(rf"\b{label}\s*=\s*\[(.*?)\];", block, text, count=1, flags=re.DOTALL)


def update_walk_edges(text: str, *, walk_dist: list[str], walk_time: list[str]) -> str:
    edges = parse_edges(text)
    dist_values = parse_numeric_block(text, "Distance")
    time_values = parse_numeric_block(text, "EdgeTime")
    idx_38_40 = edges.index((38, 40))
    idx_39_40 = edges.index((39, 40))
    dist_values[idx_39_40] = walk_dist[0]
    dist_values[idx_38_40] = walk_dist[1]
    time_values[idx_39_40] = walk_time[0]
    time_values[idx_38_40] = walk_time[1]
    text = rewrite_numeric_block(text, "Distance", dist_values)
    text = rewrite_numeric_block(text, "EdgeTime", time_values)
    return text


def apply_tune(path: Path, tune: dict[str, list[str]], *, baseline: bool) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_list_line(text, "C", tune["costs"])
    if "walk_dist" in tune and "walk_time" in tune:
        if baseline:
            text = replace_list_line(text, "WalkDistanceToDest", tune["walk_dist"])
            text = replace_list_line(text, "WalkTimeToDest", tune["walk_time"])
        text = update_walk_edges(text, walk_dist=tune["walk_dist"], walk_time=tune["walk_time"])
    path.write_text(text, encoding="utf-8")


def main() -> None:
    combined = {}
    combined.update(WALKING_TUNES)
    combined.update(PARKING_COST_TUNES)

    for name, tune in combined.items():
        apply_tune(BASELINE_DIR / name, tune, baseline=True)
        apply_tune(OPL_DIR / name, tune, baseline=False)


if __name__ == "__main__":
    main()
