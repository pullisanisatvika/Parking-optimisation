import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 10,
    }
)

LOW_CONGESTION_RANGE = (50.0, 70.0)


def fmt_str(x):
    return f"{x:.2f}"


def _drive_style_for_plot(road_class):
    if road_class == "arterial":
        return {
            "casing_color": "#99a4af",
            "fill_color": "#f5f6f7",
            "casing_width": 7.0,
            "fill_width": 4.8,
            "zorder": 2,
        }
    if road_class == "connector":
        return {
            "casing_color": "#aab3bd",
            "fill_color": "#fbfbfb",
            "casing_width": 5.2,
            "fill_width": 3.2,
            "zorder": 2,
        }
    return {
        "casing_color": "#bcc4cc",
        "fill_color": "#ffffff",
        "casing_width": 3.5,
        "fill_width": 2.0,
        "zorder": 2,
    }


def _node_text_size(n):
    if n <= 25:
        return 12
    if n <= 100:
        return 10
    if n <= 500:
        return 8
    if n <= 1000:
        return 7
    return 0


def _node_style(node, source, destination, feasible_parking, infeasible_parking):
    if node == source:
        return {"color": "#157a6e", "size": 170, "edge": "#0f5c52", "zorder": 5}
    if node == destination:
        return {"color": "#cc4b42", "size": 185, "edge": "#8e2e2b", "zorder": 5}
    if node in feasible_parking:
        return {"color": "#f0c04f", "size": 120, "edge": "#9b7c1e", "zorder": 5}
    if node in infeasible_parking:
        return {"color": "#90a6b4", "size": 95, "edge": "#607887", "zorder": 4}
    return {"color": "#f7f8f9", "size": 10, "edge": "#d9dfe5", "zorder": 3}


def plot_case_graph(case, out_path):
    coords = case["coords"]
    source = case["source_node"]
    destination = case["destination_node"]

    parking = set(case["parking_nodes"])
    feasible_parking = set(case["feasible_parking_nodes"])
    infeasible_parking = parking - feasible_parking

    fig = plt.figure(figsize=(14.5, 10.5))
    ax = fig.add_axes([0.05, 0.05, 0.70, 0.90])
    fig.patch.set_facecolor("#edf1ec")
    ax.set_facecolor("#edf1ec")

    xs = [xy[0] for xy in coords.values()]
    ys = [xy[1] for xy in coords.values()]
    xpad = (max(xs) - min(xs) or 1.0) * 0.08
    ypad = (max(ys) - min(ys) or 1.0) * 0.08

    # Driving network rendered as road casings + road fill,
    # so the image reads like a road map rather than a graph.
    seen_drive = set()
    for e in case["drive_edges"]:
        u, v = e["u"], e["v"]
        pair = tuple(sorted((u, v)))
        if pair in seen_drive:
            continue
        seen_drive.add(pair)

        x1, y1 = coords[u]
        x2, y2 = coords[v]
        style = _drive_style_for_plot(e.get("road_class", "local"))

        ax.plot(
            [x1, x2], [y1, y2],
            color=style["casing_color"],
            linewidth=style["casing_width"],
            solid_capstyle="round",
            alpha=1.0,
            zorder=style["zorder"],
        )
        ax.plot(
            [x1, x2], [y1, y2],
            color=style["fill_color"],
            linewidth=style["fill_width"],
            solid_capstyle="round",
            alpha=1.0,
            zorder=style["zorder"] + 0.1,
        )

    # Walking links remain visually secondary.
    seen_walk = set()
    for e in case["walk_edges"]:
        u, v = e["u"], e["v"]
        pair = tuple(sorted((u, v)))
        if pair in seen_walk:
            continue
        seen_walk.add(pair)

        x1, y1 = coords[u]
        x2, y2 = coords[v]
        ax.plot(
            [x1, x2], [y1, y2],
            color="#e19947",
            linewidth=1.4,
            linestyle=(0, (2.5, 3.0)),
            alpha=0.85,
            zorder=3.5,
        )

    # Nodes are rendered as intersections/markers rather than dominant graph points.
    for node, (x, y) in coords.items():
        style = _node_style(node, source, destination, feasible_parking, infeasible_parking)

        ax.scatter(
            x,
            y,
            s=style["size"],
            c=style["color"],
            edgecolors=style["edge"],
            linewidths=0.8,
            zorder=style["zorder"],
        )

        label_size = _node_text_size(case["number_of_nodes"])
        if label_size > 0 and (node == source or node == destination or node in parking):
            ax.text(
                x,
                y,
                str(node + 1),
                ha="center",
                va="center",
                fontsize=label_size,
                fontweight="bold",
                color="#111111",
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )

    ax.plot([], [], color="#f5f6f7", linewidth=4.8, label="Arterial road")
    ax.plot([], [], color="#fbfbfb", linewidth=3.2, label="Connector road")
    ax.plot([], [], color="#ffffff", linewidth=2.0, label="Local road")
    ax.plot([], [], color="#e39a52", linewidth=1.3, linestyle=(0, (3, 3)), label="Walking access")
    ax.scatter([], [], s=90, c="#1b8a7a", edgecolors="#0f5c52", label="Source")
    ax.scatter([], [], s=95, c="#cf4b43", edgecolors="#8e2e2b", label="Destination")
    ax.scatter([], [], s=85, c="#f0c04f", edgecolors="#9b7c1e", label="Feasible parking")
    ax.scatter([], [], s=75, c="#91a7b5", edgecolors="#5d7380", label="Infeasible parking")

    traffic_level = case.get("traffic_level", "low")
    speed_range = case.get("traffic_speed_range", LOW_CONGESTION_RANGE)

    max_walk_km = case.get("max_walking_km", 0.0)
    max_walk_time_hr = case.get("max_walking_time_hr", max_walk_km / 5.0 if max_walk_km else 0.0)
    cmax = case.get("cmax", None)
    parking_cost_mode = case.get("parking_cost_mode", "baseline_structured")

    title_parts = [
        f"Urban Road Network: N={case['number_of_nodes']}",
        f"alpha={fmt_str(case['alpha'])}",
        f"W={fmt_str(max_walk_km)} km",
        f"Wtime={fmt_str(max_walk_time_hr)} hr",
        f"traffic={traffic_level} ({fmt_str(speed_range[0])}-{fmt_str(speed_range[1])} km/h)",
    ]

    if cmax is not None:
        title_parts.append(f"Cmax={fmt_str(cmax)}")

    if parking_cost_mode:
        title_parts.append(f"cost_mode={parking_cost_mode}")

    # Titles are omitted because the paper caption carries the figure context.

    ax.set_aspect("equal")
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.78, 0.93),
        frameon=True,
        borderaxespad=0.0,
        fontsize=10,
        labelspacing=0.7,
        borderpad=0.8,
        handlelength=2.2,
        handletextpad=0.8,
    )
    legend.get_frame().set_facecolor("#f7f8f6")
    legend.get_frame().set_edgecolor("#c7cec6")
    legend.get_frame().set_linewidth(1.0)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="#edf1ec")
    plt.close(fig)
