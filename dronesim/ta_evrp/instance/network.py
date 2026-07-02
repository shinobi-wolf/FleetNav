"""SUMO network loader for the TA-EVRP Instance block.

Native implementation — does not import from dronesim.evrp.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

try:
    import sumolib  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover
    sumolib = None
    _SUMOLIB_IMPORT_ERROR = exc
else:
    _SUMOLIB_IMPORT_ERROR = None


def _resolve_net_xml_path(net_path: str) -> Path:
    p = Path(net_path)
    if not p.exists():
        raise FileNotFoundError(f"Network path not found: {net_path}")

    if p.suffixes[-2:] == [".net", ".xml"]:
        return p.resolve()

    if p.suffix == ".sumocfg":
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Cannot parse SUMO config {p}: {exc}") from exc
        net_el = root.find("./input/net-file") or root.find(".//net-file")
        if net_el is None:
            raise ValueError(f"No <net-file> element found in {p}")
        value = net_el.get("value")
        if not value:
            raise ValueError(f"Empty <net-file> value in {p}")
        net_xml = (p.parent / value).resolve()
        if not net_xml.exists():
            raise FileNotFoundError(f"net-file referenced by {p} not found: {net_xml}")
        return net_xml

    raise ValueError(
        f"Unsupported network path format (expected .net.xml or .sumocfg): {net_path}"
    )


def _node_position(node_obj: Any) -> tuple[float, float]:
    shape = node_obj.getShape()
    if shape:
        return float(np.mean([pt[0] for pt in shape])), float(np.mean([pt[1] for pt in shape]))
    coord = node_obj.getCoord()
    return float(coord[0]), float(coord[1])


class Network:
    """Loaded SUMO network as an nx.DiGraph.

    Built once from a .sumocfg or .net.xml path and injected into Traffic
    (for edge IDs) and Problem (for graph topology and node positions).

    ``ped_mode`` builds the *pedestrian* graph for delivery-robot (ADR) runs:
    the net is read with pedestrian connections, edges without a pedestrian
    lane are dropped, and each remaining edge carries a ``sidewalk_lane``
    attribute (the lane the robot walks on). This mirrors the TRSP training
    env / eval graph so the ped policy sees the same topology end-to-end.
    """

    def __init__(self, net_path: str, ped_mode: bool = False) -> None:
        if sumolib is None:
            raise RuntimeError(f"sumolib import failed: {_SUMOLIB_IMPORT_ERROR}")

        self.ped_mode: bool = bool(ped_mode)
        net_xml = _resolve_net_xml_path(net_path)
        self.net_xml_path: str = str(net_xml)
        # sumolib requires withInternal when pedestrian connections are requested;
        # internal (':'-prefixed) edges are filtered out of the graph below either way.
        self.net: Any = sumolib.net.readNet(
            str(net_xml), withInternal=self.ped_mode, withPedestrianConnections=self.ped_mode
        )

        g = nx.DiGraph()
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":"):
                continue

            if self.ped_mode and not any(
                lane.allows("pedestrian") for lane in edge.getLanes()
            ):
                continue

            from_node = edge.getFromNode()
            to_node   = edge.getToNode()
            from_id   = from_node.getID()
            to_id     = to_node.getID()

            fx, fy = _node_position(from_node)
            tx, ty = _node_position(to_node)

            g.add_node(from_id, x=fx, y=fy)
            g.add_node(to_id,   x=tx, y=ty)

            length = float(edge.getLength())
            if length <= 0.0:
                length = float(math.hypot(tx - fx, ty - fy))

            if self.ped_mode:
                sidewalk_lane = next(
                    (lane.getID() for lane in edge.getLanes() if lane.allows("pedestrian")),
                    None,
                )
                g.add_edge(from_id, to_id, id=edge_id, length=length, sidewalk_lane=sidewalk_lane)
            else:
                g.add_edge(from_id, to_id, id=edge_id, length=length)

        if len(g.nodes) == 0:
            raise ValueError(f"No non-internal nodes found in {net_xml}")

        self.graph: nx.DiGraph = g
        self.node_positions: dict[str, tuple[float, float]] = {
            n: (float(g.nodes[n]["x"]), float(g.nodes[n]["y"]))
            for n in g.nodes
        }

    @property
    def edge_ids(self) -> list[str]:
        return [d["id"] for _, _, d in self.graph.edges(data=True)]

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()
