import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_CASES_DIR = ROOT / "final_cases"

BASE_MIN_COST = 8.0
BASE_MAX_COST = 20.5


def parse_list(text: str, key: str) -> list[float]:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*\[([^\]]*)\]\s*;", text)
    if not match:
        raise ValueError(f"Missing {key} list")
    body = match.group(1).strip()
    if not body:
        return []
    return [float(item.strip()) for item in body.split(",") if item.strip()]


def parse_scalar(text: str, key: str) -> float:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*([0-9.]+)\s*;", text)
    if not match:
        raise ValueError(f"Missing {key} scalar")
    return float(match.group(1))


def parse_case_code(path: Path) -> str:
    match = re.search(r"_C(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not parse case code from {path.name}")
    return f"C{int(match.group(1))}"


def format_cost(value: float) -> str:
    rounded = round(value, 2)
    if math.isclose(rounded, round(rounded)):
        return str(int(round(rounded)))
    text = f"{rounded:.2f}"
    if text.endswith("0"):
        text = text[:-1]
    return text


def rank_based_costs(
    walk_distances: list[float], case_code: str, cmax: float
) -> list[float]:
    count = len(walk_distances)
    if count == 0:
        return []
    max_cost = min(BASE_MAX_COST, cmax)
    if count == 1:
        return [round(max_cost, 2)]

    ordered_indices = sorted(range(count), key=lambda idx: (walk_distances[idx], idx))
    costs = [0.0] * count

    for rank, idx in enumerate(ordered_indices):
        t = rank / (count - 1)

        if case_code == "C10":
            # Keep C10 as the parking-cost-variation scenario with a steeper spread.
            score = 1.0 - math.pow(t, 0.55)
        else:
            # For the rest of the dataset, make closer parking systematically costlier.
            score = 1.0 - t

        costs[idx] = BASE_MIN_COST + score * (max_cost - BASE_MIN_COST)

    return [round(value, 2) for value in costs]


def rewrite_cost_vector(path: Path) -> tuple[str, int]:
    text = path.read_text(encoding="utf-8")
    parking_nodes = parse_list(text, "L")
    walk_distances = parse_list(text, "WalkDistanceToDest")
    cmax = parse_scalar(text, "Cmax")

    if len(parking_nodes) != len(walk_distances):
        raise ValueError(
            f"{path.name}: parking node count {len(parking_nodes)} does not match "
            f"walk-distance count {len(walk_distances)}"
        )

    case_code = parse_case_code(path)
    new_costs = rank_based_costs(walk_distances, case_code, cmax)
    replacement = f"C = [{','.join(format_cost(cost) for cost in new_costs)}];"

    updated_text, replacements = re.subn(
        r"C\s*=\s*\[[^\]]*\]\s*;",
        replacement,
        text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"{path.name}: expected exactly one C assignment")

    path.write_text(updated_text, encoding="utf-8")
    return case_code, len(new_costs)


def main() -> None:
    dat_paths = sorted(FINAL_CASES_DIR.glob("n_*/*.dat"))
    if not dat_paths:
        raise RuntimeError(f"No final-case .dat files found under {FINAL_CASES_DIR}")

    for path in dat_paths:
        case_code, parking_count = rewrite_cost_vector(path)
        print(f"Updated {path.relative_to(ROOT)}: {case_code}, {parking_count} parking costs")


if __name__ == "__main__":
    main()
