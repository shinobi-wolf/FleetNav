"""Route visualisation utilities."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


def build_plot_coordinates(
    physical_dist: np.ndarray,
    instance: dict[str, Any],
    physical_coords: np.ndarray | None = None,
) -> np.ndarray:
    """Build plotting coordinates from SUMO coordinates when available, otherwise MDS."""
    if physical_coords is not None:
        base_coords = np.array(physical_coords, dtype=float)
    else:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            n_init=4,
            normalized_stress="auto",
        )
        base_coords = mds.fit_transform(physical_dist)

    node_coords = np.zeros((len(instance["V"]), 2), dtype=float)
    parent_seen_count: dict[int, int] = defaultdict(int)

    for node, parent in enumerate(instance["parent_index"]):
        coord = base_coords[parent].copy()
        node_type = instance["node_types"][node]
        if node_type in {"afs_dummy", "depot_dummy"}:
            parent_seen_count[parent] += 1
            k = parent_seen_count[parent]
            angle = (2.0 * math.pi * k) / 6.0
            radius = 5.0 + 1.5 * (k - 1)
            coord += np.array([radius * math.cos(angle), radius * math.sin(angle)])
        node_coords[node] = coord

    return node_coords


def plot_routes(
    routes: list[dict[str, Any]],
    coords: np.ndarray,
    node_types: list[str],
    out_path: str | None = None,
) -> None:
    """Visualize node classes and routes in 2D."""
    if len(routes) == 0:
        print("No routes to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 8))

    class_styles = {
        "depot": {"marker": "s", "color": "black", "label": "Depot", "size": 130},
        "depot_dummy": {
            "marker": "s",
            "color": "0.45",
            "label": "Depot dummy",
            "size": 85,
        },
        "customer": {
            "marker": "o",
            "color": "tab:blue",
            "label": "Customer",
            "size": 95,
        },
        "afs": {"marker": "^", "color": "tab:green", "label": "AFS", "size": 130},
        "afs_dummy": {
            "marker": "^",
            "color": "tab:olive",
            "label": "AFS dummy",
            "size": 85,
        },
    }

    for node_class, style in class_styles.items():
        idxs = [i for i, t in enumerate(node_types) if t == node_class]
        if not idxs:
            continue
        ax.scatter(
            coords[idxs, 0],
            coords[idxs, 1],
            marker=style["marker"],
            s=style["size"],
            c=style["color"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label=style["label"],
            zorder=3,
        )

    route_palette = plt.colormaps.get_cmap("tab10")
    for ridx, route in enumerate(routes):
        nodes = route["nodes"]
        x_line = [coords[n, 0] for n in nodes]
        y_line = [coords[n, 1] for n in nodes]
        ax.plot(
            x_line,
            y_line,
            color=route_palette(ridx % 10),
            linewidth=2.4,
            alpha=0.95,
            label=f"Vehicle {route['vehicle']} ({route['distance']:.1f})",
            zorder=2,
        )

    for node_idx in range(coords.shape[0]):
        ax.text(
            coords[node_idx, 0] + 1.2,
            coords[node_idx, 1] + 1.2,
            str(node_idx),
            fontsize=8,
            color="0.2",
        )

    ax.set_title("G-VRP solution routes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.28)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()

    if "agg" in plt.get_backend().lower():
        out_file = out_path or "gvrp_routes.png"
        plt.savefig(out_file, dpi=200)
        print(f"Plot saved to {out_file} (non-GUI backend detected).")
        plt.close(fig)
    else:
        plt.show()
