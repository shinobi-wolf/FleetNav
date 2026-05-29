import sumolib
import networkx as nx
import subprocess
import os
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────
# STEP 1: Download and save raw OSM (unchanged)
# ─────────────────────────────────────────────
import osmnx as ox

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.all_oneway = True

G = ox.graph_from_bbox(
    [-93.61485, 42.01818, -93.67373, 42.04203],
    custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary"]["area"!~"yes"]["access"!~"private"]',
    network_type='drive', simplify=False, retain_all=True, truncate_by_edge=True,
)
wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
G_largest = G.subgraph(wcc[0]).copy()
ox.io.save_graph_xml(G_largest, filepath="ames_campus.osm")
print(f"Step 1 done — raw nodes: {len(G_largest.nodes())}")

# ─────────────────────────────────────────────
# STEP 2: Standardize roads via osmfilter (unchanged)
# ─────────────────────────────────────────────
subprocess.run([
    "osmfilter", "ames_campus.osm",
    "--drop-tags=maxspeed=* maxspeed:forward=* maxspeed:backward=* "
                "lanes=* lanes:forward=* lanes:backward=* "
                "turn:lanes=* surface=* name=* ref=*",
    "-o=ames_campus_clean.osm"
], check=True)
print("Step 2 done — osmfilter complete")

# ─────────────────────────────────────────────
# STEP 3 (moved): netconvert — join edges and junctions
# ─────────────────────────────────────────────
sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
type_files = (f"{sumo_home}/data/typemap/osmNetconvert.typ.xml,"
              f"{sumo_home}/data/typemap/osmNetconvertUrbanDe.typ.xml")

subprocess.run([
    "netconvert",
    "--osm-files",           "ames_campus_clean.osm",
    "--output-file",         "ames_campus_raw.net.xml",   # intermediate output
    "--type-files",          type_files,
    "--geometry.remove",
    "--edges.join",
    "--junctions.join",
    "--junctions.join-dist", "50",
    "--roundabouts.guess",   "false",
    "--tls.discard-loaded",  "true",
    "--tls.guess",           "false",
    "--tls.guess-signals",   "false",
    "--sidewalks.guess",     "false",
    "--crossings.guess",     "false",
    "--walkingareas",        "false",
    "--keep-edges.by-vclass","passenger",
    "--no-turnarounds",
], check=True)
print("Step 3 done — SUMO net written: ames_campus_raw.net.xml")

# ─────────────────────────────────────────────
# STEP 4 (moved): Remove degree-2 nodes from SUMO net
#   — uses sumolib instead of osmnx, reads .net.xml directly
# ─────────────────────────────────────────────

# Parse SUMO net into undirected networkx graph via sumolib
net = sumolib.net.readNet("ames_campus_raw.net.xml")

G_sumo = nx.Graph()
for junction in net.getNodes():
    if junction.getType() != "internal":          # skip SUMO-internal junctions
        G_sumo.add_node(junction.getID())

for edge in net.getEdges():
    if not edge.getID().startswith(":"):          # skip SUMO-internal edges
        u = edge.getFromNode().getID()
        v = edge.getToNode().getID()
        if u in G_sumo and v in G_sumo:
            G_sumo.add_edge(u, v)

print(f"Before degree-2 removal: {G_sumo.number_of_nodes()} nodes")

# Remove degree-2 nodes iteratively (same logic as before)
def remove_degree2_nodes(G):
    G = G.copy()
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes()):
            if G.degree(node) == 2:
                u, v = list(G.neighbors(node))
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
                G.remove_node(node)
                changed = True
                break
    return G

G_simplified = remove_degree2_nodes(G_sumo)
surviving_nodes = set(G_simplified.nodes())
nodes_to_remove = set(G_sumo.nodes()) - surviving_nodes
print(f"Removing {len(nodes_to_remove)} degree-2 nodes")
print(f"After degree-2 removal: {G_simplified.number_of_nodes()} nodes")

# Build coordinate and directed-edge lookups from raw net
full_pos = {j.getID(): j.getCoord()
            for j in net.getNodes() if j.getType() != "internal"}
sumo_dir_edges = {(e.getFromNode().getID(), e.getToNode().getID()): e
                  for e in net.getEdges() if not e.getID().startswith(":")}

def trace_path(u, v):
    """DFS from u to v through removed degree-2 nodes to recover intermediate shape."""
    stack = [(u, [u])]
    visited = {u}
    while stack:
        curr, path = stack.pop()
        for nbr in G_sumo.neighbors(curr):
            if nbr in visited:
                continue
            new_path = path + [nbr]
            if nbr == v:
                return new_path
            if nbr in nodes_to_remove:
                visited.add(nbr)
                stack.append((nbr, new_path))
    return [u, v]

# Write plain SUMO node XML — only surviving junctions
nod_root = ET.Element("nodes")
for nid in G_simplified.nodes():
    x, y = full_pos[nid]
    ET.SubElement(nod_root, "node", id=nid, x=f"{x:.2f}", y=f"{y:.2f}", type="priority")
ET.ElementTree(nod_root).write("tmp_nodes.xml", encoding="unicode", xml_declaration=True)

# Write plain SUMO edge XML — one directed edge per direction in G_simplified
edg_root = ET.Element("edges")
written = set()
for u, v in G_simplified.edges():
    path_uv = trace_path(u, v)
    path_vu = list(reversed(path_uv))
    for a, b, path in [(u, v, path_uv), (v, u, path_vu)]:
        if (a, b) in written:
            continue
        # Find first directed SUMO edge along this path to inherit attributes
        ref = next((sumo_dir_edges[(x, y)]
                    for x, y in zip(path[:-1], path[1:])
                    if (x, y) in sumo_dir_edges), None)
        if ref is None:
            continue
        e_elem = ET.SubElement(edg_root, "edge")
        e_elem.set("id",       f"{a}_to_{b}")
        e_elem.set("from",     a)
        e_elem.set("to",       b)
        e_elem.set("numLanes", str(ref.getLaneNumber()))
        e_elem.set("speed",    str(ref.getSpeed()))
        e_elem.set("priority", str(ref.getPriority()))
        e_elem.set("shape",    " ".join(f"{full_pos[n][0]:.2f},{full_pos[n][1]:.2f}"
                                        for n in path if n in full_pos))
        written.add((a, b))
ET.ElementTree(edg_root).write("tmp_edges.xml", encoding="unicode", xml_declaration=True)

subprocess.run([
    "netconvert",
    "--node-files",  "tmp_nodes.xml",
    "--edge-files",  "tmp_edges.xml",
    "--output-file", "ames_campus.net.xml",
    "--no-turnarounds",
], check=True)
for f in ("tmp_nodes.xml", "tmp_edges.xml"):
    os.remove(f)
print("Step 4 done — final SUMO net: ames_campus.net.xml")