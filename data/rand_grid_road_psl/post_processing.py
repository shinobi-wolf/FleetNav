import argparse
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def strip_xml_namespace(root):
    """
    SUMO XML usually has no namespace, but this makes tag matching safer.
    """
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def parse_nodes(nod_file):
    tree = ET.parse(nod_file)
    root = tree.getroot()
    strip_xml_namespace(root)

    nodes = {}
    for node in root.findall("node"):
        node_id = node.attrib["id"]
        nodes[node_id] = node

    return tree, root, nodes


def parse_edges(edg_file):
    tree = ET.parse(edg_file)
    root = tree.getroot()
    strip_xml_namespace(root)

    edges = []
    for edge in root.findall("edge"):
        # Ignore special/internal edges if any appear
        if edge.attrib.get("function") == "internal":
            continue

        from_node = edge.attrib.get("from")
        to_node = edge.attrib.get("to")

        if from_node is None or to_node is None:
            continue

        edges.append(edge)

    return tree, root, edges


def compute_undirected_degrees(edges, active_nodes):
    neighbors = defaultdict(set)

    for edge in edges:
        u = edge.attrib["from"]
        v = edge.attrib["to"]

        if u in active_nodes and v in active_nodes:
            neighbors[u].add(v)
            neighbors[v].add(u)

    degrees = {node: len(neighbors[node]) for node in active_nodes}
    return degrees


def prune_leaf_nodes(nodes, edges):
    active_nodes = set(nodes.keys())
    active_edges = list(edges)

    total_removed = set()
    iteration = 0

    while True:
        iteration += 1

        degrees = compute_undirected_degrees(active_edges, active_nodes)

        leaf_nodes = {node for node, deg in degrees.items() if deg <= 1}

        if not leaf_nodes:
            break

        print(f"Iteration {iteration}: removing {len(leaf_nodes)} leaf/isolated nodes")

        total_removed.update(leaf_nodes)
        active_nodes -= leaf_nodes

        active_edges = [
            edge for edge in active_edges
            if edge.attrib["from"] in active_nodes and edge.attrib["to"] in active_nodes
        ]

    print(f"Total removed nodes: {len(total_removed)}")
    print(f"Remaining nodes: {len(active_nodes)}")
    print(f"Remaining edges: {len(active_edges)}")

    return active_nodes, active_edges, total_removed


def write_clean_nodes(original_node_root, active_nodes, output_nod_file):
    new_root = ET.Element(original_node_root.tag, original_node_root.attrib)

    for child in original_node_root:
        if child.tag == "node":
            if child.attrib["id"] in active_nodes:
                new_root.append(child)
        else:
            new_root.append(child)

    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="    ")
    tree.write(output_nod_file, encoding="UTF-8", xml_declaration=True)


def write_clean_edges(original_edge_root, active_edges, output_edg_file):
    active_edge_ids = {edge.attrib["id"] for edge in active_edges}

    new_root = ET.Element(original_edge_root.tag, original_edge_root.attrib)

    for child in original_edge_root:
        if child.tag == "edge":
            if child.attrib.get("id") in active_edge_ids:
                new_root.append(child)
        else:
            new_root.append(child)

    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="    ")
    tree.write(output_edg_file, encoding="UTF-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(
        description="Remove dead-end/leaf nodes from a SUMO .net.xml network."
    )

    parser.add_argument(
        "--input-net",
        required=True,
        help="Input SUMO .net.xml file"
    )

    parser.add_argument(
        "--output-net",
        required=True,
        help="Output cleaned SUMO .net.xml file"
    )

    parser.add_argument(
        "--keep-plain",
        action="store_true",
        help="Keep intermediate .nod.xml and .edg.xml files"
    )

    args = parser.parse_args()

    input_net = os.path.abspath(args.input_net)
    output_net = os.path.abspath(args.output_net)

    with tempfile.TemporaryDirectory() as tmpdir:
        plain_prefix = os.path.join(tmpdir, "plain")

        # Step 1: Convert .net.xml into plain node/edge XML files
        run([
            "netconvert",
            "--sumo-net-file", input_net,
            "--plain-output-prefix", plain_prefix
        ])

        nod_file = plain_prefix + ".nod.xml"
        edg_file = plain_prefix + ".edg.xml"

        # Step 2: Load plain XML files
        _, node_root, nodes = parse_nodes(nod_file)
        _, edge_root, edges = parse_edges(edg_file)

        print(f"Original nodes: {len(nodes)}")
        print(f"Original edges: {len(edges)}")

        # Step 3: Iteratively remove degree-1 nodes
        active_nodes, active_edges, removed_nodes = prune_leaf_nodes(nodes, edges)

        clean_nod_file = os.path.join(tmpdir, "clean.nod.xml")
        clean_edg_file = os.path.join(tmpdir, "clean.edg.xml")

        # Step 4: Write cleaned plain files
        write_clean_nodes(node_root, active_nodes, clean_nod_file)
        write_clean_edges(edge_root, active_edges, clean_edg_file)

        # Step 5: Rebuild SUMO network
        run([
            "netconvert",
            "--node-files", clean_nod_file,
            "--edge-files", clean_edg_file,
            "--output-file", output_net,
            "--no-turnarounds", "true"
        ])

        if args.keep_plain:
            base = os.path.splitext(output_net)[0]
            final_nod = base + ".clean.nod.xml"
            final_edg = base + ".clean.edg.xml"

            import shutil
            shutil.copy(clean_nod_file, final_nod)
            shutil.copy(clean_edg_file, final_edg)

            print(f"Saved cleaned nodes: {final_nod}")
            print(f"Saved cleaned edges: {final_edg}")

    print(f"Done. Cleaned network written to: {output_net}")


if __name__ == "__main__":
    main()