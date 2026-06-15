"""Save G-VRP problem instances to JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_instance_json(
    config: dict[str, Any],
    physical_dist: np.ndarray,
    physical_meta: dict[str, Any],
    generation_info: dict[str, Any],
) -> None:
    output_cfg = config.get("output", {})
    out_path = output_cfg.get("save_instance_json", None)
    if not out_path:
        return

    payload = {
        "distance_unit": "meters",
        "physical_node_names": physical_meta["physical_node_names"],
        "physical_node_types": physical_meta["physical_node_types"],
        "physical_sumo_ids": physical_meta["physical_sumo_ids"],
        "distance_matrix": physical_dist.tolist(),
        "generation_info": {
            "seed": generation_info["seed"],
            "net_xml_path": generation_info["net_xml_path"],
            "depot_sumo_id": generation_info["depot_sumo_id"],
            "customer_sumo_ids": generation_info["customer_sumo_ids"],
            "afs_sumo_ids": generation_info["afs_sumo_ids"],
        },
    }

    out_file = Path(out_path)
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved generated instance to {out_file}")
