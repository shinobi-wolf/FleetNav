#!/usr/bin/env python3
"""
gen_traffic.py — Generate weighted random car and pedestrian traffic for
rand_grid_road_ped.net.xml.  Trips are sampled so that OD pairs whose
midpoints are closest to the network centroid are far more frequent than
those near the perimeter (Gaussian weight centred on the network centre).

The edge topology is parsed directly from the .net.xml file, so this
script works for any SUMO network derived from rand_grid_road_ped.

Outputs (all in the same directory as this script):
  vehicles.trips.xml   — raw car OD trips fed to duarouter
  vehicles.rou.xml     — duarouter-routed car routes
  pedestrians.rou.xml  — person+walk elements (SUMO routes at runtime)
  rand_grid_road_ped.sumocfg
"""

import math
import random
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Parameters ──────────────────────────────────────────────────────────────
SEED          = 42
N_VEHICLES    = 500
N_PEDESTRIANS = 6000
SIM_DURATION  = 7200   # seconds
SIGMA         = 250.0  # Gaussian falloff in metres (network spans ~900 m)

# ── Custom pedestrian vType dimensions ───────────────────────────────────────
PED_WIDTH  = 1.5   # metres  (default SUMO pedestrian: ~0.48)
PED_LENGTH = 0.8   # metres  (default SUMO pedestrian: ~0.20)

HERE     = Path(__file__).parent
NET_FILE = "rand_grid_road_ped.net.xml"


# ── Topology helpers ─────────────────────────────────────────────────────────

def load_network(net_path: Path):
    """
    Parse *net_path* and return:
      junction_xy  : {junction_id: (x, y)}  — non-internal junctions only
      normal_edges : list of edge_id strings — non-internal, non-walkingarea,
                     non-crossing edges that vehicles can use as from/to edges
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
        if func in ("internal", "crossing", "walkingarea"):
            continue
        eid  = edge.get("id")
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
    to the network centroid.
    """
    x0, y0 = junction_xy[from_j]
    x1, y1 = junction_xy[to_j]
    mx, my  = (x0 + x1) / 2, (y0 + y1) / 2
    cx, cy  = centroid
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

    # Sanity-check a handful of representative edges
    print("\nEdge centrality weight sample (edge_id → weight):")
    for eid, fj, tj in normal_edges[:6]:
        w = centrality_weight(fj, tj, junction_xy, centroid, SIGMA)
        print(f"  {eid:8s}  ({fj}→{tj})  w={w:.4f}")

    rng = random.Random(SEED)

    # ── Vehicle trips ────────────────────────────────────────────────────────
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

    # ── Pedestrian routes ────────────────────────────────────────────────────
    peds = []
    for i in range(N_PEDESTRIANS):
        fe, te = sample_od_pair(normal_edges, weights, rng)
        depart = round(rng.uniform(0, SIM_DURATION), 2)
        peds.append((depart, i, fe[0], te[0]))
    peds.sort()

    ped_root = ET.Element("routes")
    ped_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    # Base type — inherits all SUMO pedestrian defaults
    base_pvtype = ET.SubElement(ped_root, "vType")
    base_pvtype.set("id",     "pedestrian")
    base_pvtype.set("vClass", "pedestrian")
    # Custom type — same as base but wider and longer
    pvtype = ET.SubElement(ped_root, "vType")
    pvtype.set("id",     "custom_pedestrian")
    pvtype.set("vClass", "pedestrian")
    pvtype.set("width",  str(PED_WIDTH))
    pvtype.set("length", str(PED_LENGTH))
    for depart, i, from_e, to_e in peds:
        person = ET.SubElement(ped_root, "person")
        person.set("id",     f"ped_{i}")
        person.set("type",   "custom_pedestrian")
        person.set("depart", str(depart))
        walk = ET.SubElement(person, "walk")
        walk.set("from", from_e)
        walk.set("to",   to_e)

    ped_path = HERE / "pedestrians.rou.xml"
    write_xml(ped_root, ped_path)
    print(f"Wrote {N_PEDESTRIANS} pedestrian persons → {ped_path.name}")

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

    # ── sumocfg ──────────────────────────────────────────────────────────────
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file").set("value",    NET_FILE)
    ET.SubElement(inp, "route-files").set("value", "vehicles.rou.xml,pedestrians.rou.xml")
    time_elem = ET.SubElement(cfg, "time")
    ET.SubElement(time_elem, "begin").set("value", "0")
    ET.SubElement(time_elem, "end").set("value",   str(SIM_DURATION))
    rep = ET.SubElement(cfg, "report")
    ET.SubElement(rep, "verbose").set("value",     "true")
    ET.SubElement(rep, "no-step-log").set("value", "true")

    cfg_path = HERE / "rand_grid_road_ped.sumocfg"
    write_xml(cfg, cfg_path)
    print(f"Wrote SUMO config → {cfg_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
