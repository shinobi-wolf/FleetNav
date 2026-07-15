#!/usr/bin/env python3
"""
gen_traffic.py — Generate weighted random car traffic for
rand_grid_road_2.net.xml.  Car-only network: no pedestrian demand.

Trips are sampled so that OD pairs whose midpoints are closest to the
network centroid are far more frequent than those near the perimeter
(Gaussian weight centred on the network centre). Because duarouter tends
to route centrally-weighted OD pairs through the network core, this
concentrates traffic volume — and therefore congestion — on the central
junctions/edges while leaving fringe junctions/edges comparatively free
flowing.

The edge topology is parsed directly from the .net.xml file, so this
script works for any SUMO network derived from rand_grid_road_2.

Outputs (all in the same directory as this script):
  vehicles.trips.xml       — raw car OD trips fed to duarouter
  vehicles.rou.xml         — duarouter-routed car routes
  rand_grid_road_2.sumocfg — SUMO config (net + vehicles)
"""

import math
import random
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# ── Parameters ──────────────────────────────────────────────────────────────
SEED         = 42
N_VEHICLES   = 4500   # sole traffic source on this network (no pedestrians)
SIM_DURATION = 7200   # seconds
SIGMA        = 250.0  # Gaussian falloff in metres (network spans ~900 m)

HERE     = Path(__file__).parent
NET_FILE = "rand_grid_road_2.net.xml"


# ── Topology helpers ─────────────────────────────────────────────────────────

def load_network(net_path: Path):
    """
    Parse *net_path* and return:
      junction_xy  : {junction_id: (x, y)}  — non-internal junctions only
      normal_edges : list of (edge_id, from_junction, to_junction) tuples —
                     non-internal edges vehicles can use as from/to edges
    """
    tree = ET.parse(str(net_path))
    root = tree.getroot()

    junction_xy = {}
    for junc in root.findall("junction"):
        jtype = junc.get("type", "")
        if jtype == "internal":
            continue
        jid = junc.get("id")
        x   = float(junc.get("x", 0))
        y   = float(junc.get("y", 0))
        junction_xy[jid] = (x, y)

    normal_edges = []
    for edge in root.findall("edge"):
        func = edge.get("function", "")
        if func == "internal":
            continue
        eid    = edge.get("id")
        from_j = edge.get("from")
        to_j   = edge.get("to")
        # Only keep edges whose endpoints we know (guards against dead fringe)
        if from_j in junction_xy and to_j in junction_xy:
            normal_edges.append((eid, from_j, to_j))

    return junction_xy, normal_edges


def network_centroid(junction_xy: dict) -> tuple:
    """Return the arithmetic centre of all junction coordinates."""
    xs = [v[0] for v in junction_xy.values()]
    ys = [v[1] for v in junction_xy.values()]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def centrality_weight(from_j, to_j, junction_xy, centroid, sigma):
    """
    Gaussian weight based on the Euclidean distance from the edge midpoint
    to the network centroid. Central edges get weight ~1, fringe edges
    decay towards 0.
    """
    x0, y0 = junction_xy[from_j]
    x1, y1 = junction_xy[to_j]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    cx, cy = centroid
    d2 = (mx - cx) ** 2 + (my - cy) ** 2
    return math.exp(-d2 / sigma ** 2)


def sample_od_pair(edges, weights, rng):
    """Sample two distinct edges weighted by centrality."""
    from_edge = rng.choices(edges, weights=weights, k=1)[0]
    to_edge   = rng.choices(edges, weights=weights, k=1)[0]
    while to_edge[0] == from_edge[0]:
        to_edge = rng.choices(edges, weights=weights, k=1)[0]
    return from_edge, to_edge


# ── XML helpers ──────────────────────────────────────────────────────────────

def indent(elem, level=0):
    """Add pretty-print indentation in-place."""
    pad = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def write_xml(root, path):
    indent(root)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)


def write_sumocfg(cfg_path, route_file):
    """Write a sumocfg referencing the shared net + vehicle routes."""
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file").set("value",    NET_FILE)
    ET.SubElement(inp, "route-files").set("value", route_file)
    time_elem = ET.SubElement(cfg, "time")
    ET.SubElement(time_elem, "begin").set("value", "0")
    ET.SubElement(time_elem, "end").set("value",   str(SIM_DURATION))
    rep = ET.SubElement(cfg, "report")
    ET.SubElement(rep, "verbose").set("value",     "true")
    ET.SubElement(rep, "no-step-log").set("value", "true")
    write_xml(cfg, cfg_path)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    net_path = HERE / NET_FILE
    junction_xy, normal_edges = load_network(net_path)

    print(f"Loaded {len(junction_xy)} junctions and {len(normal_edges)} routable edges "
          f"from {NET_FILE}")

    centroid = network_centroid(junction_xy)
    print(f"Network centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}) m")

    weights = [
        centrality_weight(fj, tj, junction_xy, centroid, SIGMA)
        for _, fj, tj in normal_edges
    ]

    # Sanity-check a handful of central vs. fringe edges by sorting on weight
    ranked = sorted(zip(normal_edges, weights), key=lambda p: -p[1])
    print("\nMost central edges (edge_id → weight):")
    for (eid, fj, tj), w in ranked[:5]:
        print(f"  {eid:8s}  ({fj}→{tj})  w={w:.4f}")
    print("Most fringe edges (edge_id → weight):")
    for (eid, fj, tj), w in ranked[-5:]:
        print(f"  {eid:8s}  ({fj}→{tj})  w={w:.4f}")

    rng = random.Random(SEED)

    # ── Vehicle trips ─────────────────────────────────────────────────────────
    trips = []
    for i in range(N_VEHICLES):
        fe, te = sample_od_pair(normal_edges, weights, rng)
        depart = round(rng.uniform(0, SIM_DURATION), 2)
        trips.append((depart, i, fe[0], te[0]))
    trips.sort()

    veh_root = ET.Element("routes")
    veh_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    vtype = ET.SubElement(veh_root, "vType")
    vtype.set("id",     "car")
    vtype.set("vClass", "passenger")
    for depart, i, from_e, to_e in trips:
        t = ET.SubElement(veh_root, "trip")
        t.set("id",     f"car_{i}")
        t.set("type",   "car")
        t.set("depart", str(depart))
        t.set("from",   from_e)
        t.set("to",     to_e)

    trips_path = HERE / "vehicles.trips.xml"
    write_xml(veh_root, trips_path)
    print(f"\nWrote {N_VEHICLES} vehicle trips → {trips_path.name}")

    # ── duarouter: trips → routes ────────────────────────────────────────────
    rou_path = HERE / "vehicles.rou.xml"
    cmd = [
        "duarouter",
        "--net",          str(net_path),
        "--route-files",  str(trips_path),
        "--output-file",  str(rou_path),
        "--ignore-errors",
        "--no-step-log",
    ]
    print(f"\nRunning duarouter …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("duarouter stderr:\n", result.stderr)
        raise RuntimeError("duarouter failed")
    print(f"Wrote routed vehicle routes → {rou_path.name}")

    # ── Sanity check: routed edge-visit counts, central vs. fringe ───────────
    central_ids = {eid for (eid, _, _), _ in ranked[: len(ranked) // 5]}
    fringe_ids  = {eid for (eid, _, _), _ in ranked[-len(ranked) // 5 :]}

    rou_tree = ET.parse(str(rou_path))
    visit_counts = Counter()
    for vehicle in rou_tree.getroot().findall("vehicle"):
        route = vehicle.find("route")
        if route is None:
            continue
        for eid in route.get("edges", "").split():
            visit_counts[eid] += 1

    central_visits = sum(visit_counts[e] for e in central_ids)
    fringe_visits  = sum(visit_counts[e] for e in fringe_ids)
    central_avg = central_visits / max(len(central_ids), 1)
    fringe_avg  = fringe_visits / max(len(fringe_ids), 1)
    print(f"\nRouted edge-visit check (top/bottom quintile by centrality):")
    print(f"  central edges: {len(central_ids)}, avg visits/edge = {central_avg:.2f}")
    print(f"  fringe  edges: {len(fringe_ids)}, avg visits/edge = {fringe_avg:.2f}")
    if fringe_avg > 0:
        print(f"  central/fringe ratio = {central_avg / fringe_avg:.2f}x")

    # ── sumocfg ────────────────────────────────────────────────────────────
    cfg_path = HERE / "rand_grid_road_2.sumocfg"
    write_sumocfg(cfg_path, "vehicles.rou.xml")
    print(f"\nWrote SUMO config → {cfg_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
