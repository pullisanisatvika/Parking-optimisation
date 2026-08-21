#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = ROOT / "final_cases" / "n_10"
OPL_DIR = ROOT / "OPTIMIZATION" / "final_cases_opl" / "n_10"

THIRD_LOT_DISTANCE = "0.3165"
THIRD_LOT_WALK_TIME = "0.0633"
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def patch_text(text: str, *, c_line: str, include_walk_metadata: bool, include_edge_type: bool) -> str:
    replacements = [
        ("<7,8> ,", "<7,8> , <7,10> ,"),
        ("Q = 2;\t\t// Number of parking lots (all urban parking facilities)", "Q = 3;\t\t// Number of parking lots (all urban parking facilities)"),
        ("L = [5,6];\t\t// Parking lot nodes", "L = [5,6,7];\t\t// Parking lot nodes"),
        ("0.2661 , 0.3339 ,", f"0.2661 , 0.3339 , {THIRD_LOT_DISTANCE} ,"),
        ("0.0532 , 0.0668 ,", f"0.0532 , 0.0668 , {THIRD_LOT_WALK_TIME} ,"),
    ]
    if include_walk_metadata:
        replacements.extend(
            [
                ("FeasibleParking = [1,1];", "FeasibleParking = [1,1,1];"),
                ("WalkDistanceToDest = [0.2661,0.3339];", f"WalkDistanceToDest = [0.2661,0.3339,{THIRD_LOT_DISTANCE}];"),
                ("WalkTimeToDest = [0.0532,0.0668];", f"WalkTimeToDest = [0.0532,0.0668,{THIRD_LOT_WALK_TIME}];"),
            ]
        )
    if include_edge_type:
        replacements.append(
            ('"walk" , "walk" ,', '"walk" , "walk" , "walk" ,')
        )

    if "C = [" in text:
        start = text.index("C = [")
        end = text.index("];", start) + 2
        prefix = text[:start]
        suffix = text[end:]
        text = prefix + c_line + suffix

    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            raise ValueError(f"Expected pattern not found: {old}")
        text = text.replace(old, new, 1)
    return text


def _rewrite_numeric_block(text: str, name: str, values: list[str]) -> str:
    formatted = []
    for i in range(0, len(values), 4):
        formatted.append(", ".join(values[i:i + 4]) + " ,")
    block = f"{name} = [\n" + "\n".join(formatted) + "\n];"
    return re.sub(rf"{name}\s*=\s*\[(.*?)\];", block, text, count=1, flags=re.DOTALL)


def _rewrite_quoted_block(text: str, name: str, values: list[str]) -> str:
    formatted = []
    for i in range(0, len(values), 8):
        formatted.append(" , ".join(values[i:i + 8]) + " ,")
    block = f"{name} = [\n" + "\n".join(formatted) + "\n];"
    return re.sub(rf"{name}\s*=\s*\[(.*?)\];", block, text, count=1, flags=re.DOTALL)


def repair_array_alignment(text: str, *, include_edge_type: bool) -> str:
    edges_match = re.search(r"\bE\s*=\s*\{(.*?)\};", text, flags=re.DOTALL)
    if not edges_match:
        return text
    edges = [(int(a), int(b)) for a, b in re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", edges_match.group(1))]
    if (7, 10) not in edges or (7, 8) not in edges:
        return text

    insert_at = edges.index((7, 8)) + 1

    distance_match = re.search(r"\bDistance\s*=\s*\[(.*?)\];", text, flags=re.DOTALL)
    edge_time_match = re.search(r"\bEdgeTime\s*=\s*\[(.*?)\];", text, flags=re.DOTALL)
    if not distance_match or not edge_time_match:
        return text

    distance_values = re.findall(FLOAT_RE, distance_match.group(1))
    edge_time_values = re.findall(FLOAT_RE, edge_time_match.group(1))

    if distance_values and distance_values[-1] == THIRD_LOT_DISTANCE:
        distance_values.insert(insert_at, distance_values.pop())
    if edge_time_values and edge_time_values[-1] == THIRD_LOT_WALK_TIME:
        edge_time_values.insert(insert_at, edge_time_values.pop())

    text = _rewrite_numeric_block(text, "Distance", distance_values)
    text = _rewrite_numeric_block(text, "EdgeTime", edge_time_values)

    if include_edge_type:
        edge_type_match = re.search(r"\bEdgeType\s*=\s*\[(.*?)\];", text, flags=re.DOTALL)
        if edge_type_match:
            edge_types = re.findall(r'"[^"]+"', edge_type_match.group(1))
            if edge_types and edge_types[-1] == '"walk"':
                edge_types.insert(insert_at, edge_types.pop())
            text = _rewrite_quoted_block(text, "EdgeType", edge_types)

    return text


def main() -> None:
    for path in sorted(BASELINE_DIR.glob("N10_C*_I1.dat")):
        cost_line = "C = [15.39,18.62,15];" if path.name == "N10_C10_I1.dat" else "C = [10.77,19.98,8];"
        original = path.read_text(encoding="utf-8")
        updated = original if "L = [5,6,7]" in original else patch_text(
            original,
            c_line=cost_line,
            include_walk_metadata=True,
            include_edge_type=True,
        )
        updated = repair_array_alignment(updated, include_edge_type=True)
        path.write_text(updated, encoding="utf-8")

    for path in sorted(OPL_DIR.glob("N10_C*_I1.dat")):
        cost_line = "C = [15.39,18.62,15];" if path.name == "N10_C10_I1.dat" else "C = [10.77,19.98,8];"
        original = path.read_text(encoding="utf-8")
        updated = original if "L = [5,6,7]" in original else patch_text(
            original,
            c_line=cost_line,
            include_walk_metadata=False,
            include_edge_type=False,
        )
        updated = repair_array_alignment(updated, include_edge_type=False)
        path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
