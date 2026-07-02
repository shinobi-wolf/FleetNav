#!/usr/bin/env python3
"""
gen_traffic.py — Generate weighted random car and pedestrian traffic for
grid_road_ped.net.xml.  Trips are sampled so that OD pairs on center edges
are far more frequent than perimeter edges (Gaussian weight centered at C2).

Outputs (all in the same directory as this script):
  vehicles.trips.xml   — raw car OD trips fed to duarouter
  vehicles.rou.xml     — duarouter-routed car routes
  pedestrians.rou.xml  — person+walk elements (SUMO routes at runtime)
  grid_road_ped.sumocfg
"""

import math
import random
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Parameters ──────────────────────────────────────────────────────────────
SEED          = 42
N_VEHICLES    = 500
N_PEDESTRIANS = 4000
SIM_DURATION  = 7200   # seconds
SIGMA         = 1.5    # Gaussian falloff in grid-distance units

# ── Custom pedestrian vType dimensions ───────────────────────────────────────
PED_WIDTH  = 1.5   # metres  (default SUMO pedestrian: ~0.48)
PED_LENGTH = 0.8   # metres  (default SUMO pedestrian: ~0.20)

HERE     = Path(__file__).parent
NET_FILE = "rand_grid_road_ped.net.xml"

# ── Grid topology ────────────────────────────────────────────────────────────
COLS    = list("ABCDE")
COL_IDX = {c: i for i, c in enumerate(COLS)}
ROWS    = list(range(5))
CENTER  = (2, 2)   # grid-unit coords of junction C2


def build_edges():
    """Return list of (edge_id, col_f, row_f, col_t, row_t) for all 80 directed edges."""
    edges = []
    # Horizontal (east / west)
    for row in ROWS:
        for ci in range(len(COLS) - 1):
            cf, ct = COLS[ci], COLS[ci + 1]
            fj, tj = cf + str(row), ct + str(row)
            edges.append((fj + tj, COL_IDX[cf], row, COL_IDX[ct], row))
            edges.append((tj + fj, COL_IDX[ct], row, COL_IDX[cf], row))
    # Vertical (north / south)
    for col in COLS:
        ci = COL_IDX[col]
        for row in range(len(ROWS) - 1):
            fj, tj = col + str(row), col + str(row + 1)
            edges.append((fj + tj, ci, row, ci, row + 1))
            edges.append((tj + fj, ci, row + 1, ci, row))
    return edges


def centrality_weight(cf, rf, ct, rt):
    mid_c = (cf + ct) / 2
    mid_r = (rf + rt) / 2
    d2 = (mid_c - CENTER[0]) ** 2 + (mid_r - CENTER[1]) ** 2
    return math.exp(-d2 / SIGMA ** 2)


def sample_od_pair(edges, weights, rng):
    """Sample two distinct edges weighted by centrality."""
    from_edge = rng.choices(edges, weights=weights, k=1)[0]
    to_edge   = rng.choices(edges, weights=weights, k=1)[0]
    while to_edge[0] == from_edge[0]:
        to_edge = rng.choices(edges, weights=weights, k=1)[0]
    return from_edge, to_edge


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
    rng   = random.Random(SEED)
    edges = build_edges()
    assert len(edges) == 80, f"Expected 80 edges, got {len(edges)}"

    weights = [centrality_weight(cf, rf, ct, rt) for _, cf, rf, ct, rt in edges]

    # Print weight summary for sanity check
    print("Edge centrality weight sample (edge_id → weight):")
    for eid, *_ in [("C2D2", 2, 2, 3, 2), ("B1C1", 1, 1, 2, 1),
                    ("A0A1", 0, 0, 0, 1), ("E3E4", 4, 3, 4, 4)]:
        cf, rf, ct, rt = (int(v) for v in [_[0], _[1], _[2], _[3]])
        w = centrality_weight(cf, rf, ct, rt)
        print(f"  {eid:8s}  w={w:.4f}")

    # ── Vehicle trips ────────────────────────────────────────────────────────
    trips = []
    for i in range(N_VEHICLES):
        fe, te = sample_od_pair(edges, weights, rng)
        depart = round(rng.uniform(0, SIM_DURATION), 2)
        trips.append((depart, i, fe[0], te[0]))
    trips.sort()

    veh_root = ET.Element("routes")
    veh_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    vtype = ET.SubElement(veh_root, "vType")
    vtype.set("id", "car")
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
        fe, te = sample_od_pair(edges, weights, rng)
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
        "--net",          str(HERE / NET_FILE),
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
    ET.SubElement(inp, "net-file").set("value", NET_FILE)
    ET.SubElement(inp, "route-files").set("value", "vehicles.rou.xml,pedestrians.rou.xml")
    time_elem = ET.SubElement(cfg, "time")
    ET.SubElement(time_elem, "begin").set("value", "0")
    ET.SubElement(time_elem, "end").set("value",   str(SIM_DURATION))
    rep = ET.SubElement(cfg, "report")
    ET.SubElement(rep, "verbose").set("value",      "true")
    ET.SubElement(rep, "no-step-log").set("value",  "true")

    cfg_path = HERE / "rand_grid_road_ped.sumocfg"
    write_xml(cfg, cfg_path)
    print(f"Wrote SUMO config → {cfg_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
