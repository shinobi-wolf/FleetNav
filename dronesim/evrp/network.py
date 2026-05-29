"""Shared SUMO network loader — builds NetworkContext once per inference run."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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


def resolve_net_xml_path(net_path: str) -> Path:
    """Resolve the .net.xml file from either a .sumocfg or .net.xml path.

    Parses the .sumocfg XML to follow the <net-file> pointer so the correct
    file is returned even when the config references a differently-named .net.xml.
    """
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

    raise ValueError(f"Unsupported network path format (expected .net.xml or .sumocfg): {net_path}")


def _node_position(node_obj: Any) -> tuple[float, float]:
    shape = node_obj.getShape()
    if shape:
        return float(np.mean([pt[0] for pt in shape])), float(np.mean([pt[1] for pt in shape]))
    coord = node_obj.getCoord()
    return float(coord[0]), float(coord[1])


@dataclass
class NetworkContext:
    """Loaded SUMO network in all forms needed by the inference pipeline.

    Build once with build_network_context() and inject into both the planner
    and the simulator so the network is read from disk exactly once per run.
    """
    net_xml_path: str                               # resolved .net.xml path
    net: Any                                        # sumolib.net.Net
    graph: nx.DiGraph                               # NX routing graph (nodes carry x, y)
    node_positions: dict[str, tuple[float, float]]  # {node_id: (x, y)}


def build_network_context(net_path: str) -> NetworkContext:
    """Load the SUMO network once and return a shared NetworkContext.

    Resolves .sumocfg paths by parsing the XML rather than doing a naive
    suffix swap, so the returned context always references the correct file.
    """
    if sumolib is None:
        raise RuntimeError(f"sumolib import failed: {_SUMOLIB_IMPORT_ERROR}")

    net_xml = resolve_net_xml_path(net_path)
    net = sumolib.net.readNet(str(net_xml), withInternal=False)

    g = nx.DiGraph()
    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(":"):
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

        g.add_edge(from_id, to_id, id=edge_id, length=length)

    if len(g.nodes) == 0:
        raise ValueError(f"No non-internal nodes found in {net_xml}")

    node_positions = {
        n: (float(g.nodes[n]["x"]), float(g.nodes[n]["y"]))
        for n in g.nodes
    }

    return NetworkContext(
        net_xml_path=str(net_xml),
        net=net,
        graph=g,
        node_positions=node_positions,
    )
