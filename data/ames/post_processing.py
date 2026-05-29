import sumolib
import networkx as nx
import subprocess
import os
import xml.etree.ElementTree as ET
from collections import Counter

# ─────────────────────────────────────────────
# Load post-surgery SUMO network
# ─────────────────────────────────────────────
net      = sumolib.net.readNet("ames_campus.net.xml")
junc_all = [j for j in net.getNodes() if j.getType() != "internal"]
edge_all = [e for e in net.getEdges() if not e.getID().startswith(":")]

# Build undirected graph
G = nx.Graph()
for j in junc_all:
    G.add_node(j.getID())
for e in edge_all:
    u, v = e.getFromNode().getID(), e.getToNode().getID()
    if u in G and v in G:
        G.add_edge(u, v)

print(f"Input:  {G.number_of_nodes()} junctions, {G.number_of_edges()} undirected edges")
print("Degree distribution:", dict(sorted(Counter(d for _, d in G.degree()).items())))

# ─────────────────────────────────────────────
# Remove leaf (degree < 2) nodes iteratively
# ─────────────────────────────────────────────
def remove_leaf_nodes(G):
    G = G.copy()
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes()):
            if G.degree(node) < 2:
                G.remove_node(node)
                changed = True
                break
    return G

G_clean = remove_leaf_nodes(G)
surviving = set(G_clean.nodes())
removed   = set(G.nodes()) - surviving
print(f"Removed {len(removed)} leaf nodes")
print(f"Output: {G_clean.number_of_nodes()} junctions, {G_clean.number_of_edges()} undirected edges")

# ─────────────────────────────────────────────
# Reconstruct SUMO net from surviving nodes
# ─────────────────────────────────────────────
full_pos = {j.getID(): j.getCoord() for j in junc_all}

nod_root = ET.Element("nodes")
for nid in surviving:
    x, y = full_pos[nid]
    ET.SubElement(nod_root, "node", id=nid, x=f"{x:.2f}", y=f"{y:.2f}", type="priority")
ET.ElementTree(nod_root).write("tmp_nodes.xml", encoding="unicode", xml_declaration=True)

edg_root = ET.Element("edges")
for e in edge_all:
    u, v = e.getFromNode().getID(), e.getToNode().getID()
    if u not in surviving or v not in surviving:
        continue
    e_elem = ET.SubElement(edg_root, "edge")
    e_elem.set("id",       e.getID())
    e_elem.set("from",     u)
    e_elem.set("to",       v)
    e_elem.set("numLanes", str(e.getLaneNumber()))
    e_elem.set("speed",    str(e.getSpeed()))
    e_elem.set("priority", str(e.getPriority()))
    shape = e.getShape()
    if shape:
        e_elem.set("shape", " ".join(f"{sx:.2f},{sy:.2f}" for sx, sy in shape))
ET.ElementTree(edg_root).write("tmp_edges.xml", encoding="unicode", xml_declaration=True)

subprocess.run([
    "netconvert",
    "--node-files",  "tmp_nodes.xml",
    "--edge-files",  "tmp_edges.xml",
    "--output-file", "ames_campus_final.net.xml",
    "--no-turnarounds",
], check=True)
for f in ("tmp_nodes.xml", "tmp_edges.xml"):
    os.remove(f)
print("Done — ames_campus_final.net.xml written")
