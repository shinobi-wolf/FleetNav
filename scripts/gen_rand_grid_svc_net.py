"""Generate data/rand_grid_road_svc — a randomized-grid road network sized for the
SVC (EV) DQN-vs-Dijkstra benchmark.

Why a new net: the original rand_grid_road has 81-212 m edges (median 184 m). At that
scale the per-hop cost (insertion, junction crossing) is the same order as the
free-flow traversal itself, so even a congested edge only costs ~11 s more than its
detour — the traffic signal drowns in hop noise and the measured Dijkstra-vs-optimal
headroom caps at ~32% (grid_road: 47%, ames: 45%). DSP (whole-graph live snapshot)
still extracts it, but a 1-hop-observing DQN cannot.

Recipe (deterministic given SEED):
  1. 6x6 lattice, 600 m spacing, node positions jittered +-JITTER m.
  2. Randomly delete DELETE_LINKS of the 60 lattice links, resampling until the
     remaining undirected graph is connected and has <= MAX_BRIDGES bridges
     (bridges = OD pairs with no detour, useless for the benchmark).
  3. Iteratively prune degree-1 nodes (same policy as data/rand_grid_road).
  4. Emit plain .nod.xml/.edg.xml (all edges bidirectional, 1 lane, 30 m/s) and
     netconvert --no-turnarounds into rand_grid_road_svc.net.xml (+ .sumocfg).

The chosen SEED/DELETE_LINKS were screened (seeds 0-19 x deletions {10,14,18}) for:
connected, bridge count, 24-34 nodes, and max analytic Dijkstra-vs-optimal headroom
of a 12-edge hot set (scripts/gen_svc_hot_traffic.py design model).

Usage:  python scripts/gen_rand_grid_svc_net.py [--screen]
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRID_N = 6
SPACING = 600.0
JITTER = 120.0
DELETE_LINKS = 10
MAX_BRIDGES = 2
SEED = 8            # picked by --screen: 31 nodes, 45 und. edges, 56.2% design headroom
OUT_DIR = 'data/rand_grid_road_svc'
NET_NAME = 'rand_grid_road_svc'


def build_lattice(seed, delete_links):
    """Return (positions, undirected edge list) after jitter+deletion+leaf-pruning,
    or None if constraints fail."""
    rng = random.Random(seed)
    nodes = [(i, j) for i in range(GRID_N) for j in range(GRID_N)]
    pos = {(i, j): (i * SPACING + rng.uniform(-JITTER, JITTER),
                    j * SPACING + rng.uniform(-JITTER, JITTER))
           for (i, j) in nodes}
    links = []
    for (i, j) in nodes:
        if i + 1 < GRID_N:
            links.append(((i, j), (i + 1, j)))
        if j + 1 < GRID_N:
            links.append(((i, j), (i, j + 1)))
    doomed = set(rng.sample(range(len(links)), delete_links))
    kept = [l for k, l in enumerate(links) if k not in doomed]

    G = nx.Graph(kept)
    if not nx.is_connected(G):
        return None
    # prune degree-1 nodes iteratively (same as rand_grid_road post-processing)
    while True:
        leaves = [v for v in G if G.degree(v) <= 1]
        if not leaves:
            break
        G.remove_nodes_from(leaves)
    if G.number_of_nodes() < 24 or not nx.is_connected(G):
        return None
    if len(list(nx.bridges(G))) > MAX_BRIDGES:
        return None
    return pos, list(G.edges())


def write_net(pos, edges, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nid = lambda ij: f"n{ij[0]}_{ij[1]}"
    used = {u for e in edges for u in e}
    with tempfile.TemporaryDirectory() as tmp:
        nod = os.path.join(tmp, 'n.nod.xml')
        edg = os.path.join(tmp, 'e.edg.xml')
        with open(nod, 'w') as f:
            f.write('<nodes>\n')
            for ij in sorted(used):
                x, y = pos[ij]
                f.write(f'  <node id="{nid(ij)}" x="{x:.2f}" y="{y:.2f}" type="priority"/>\n')
            f.write('</nodes>\n')
        with open(edg, 'w') as f:
            f.write('<edges>\n')
            for k, (a, b) in enumerate(sorted(edges)):
                f.write(f'  <edge id="e{k}" from="{nid(a)}" to="{nid(b)}" '
                        f'numLanes="1" speed="30.0"/>\n')
                f.write(f'  <edge id="-e{k}" from="{nid(b)}" to="{nid(a)}" '
                        f'numLanes="1" speed="30.0"/>\n')
            f.write('</edges>\n')
        net_path = os.path.join(out_dir, f'{NET_NAME}.net.xml')
        subprocess.run(['netconvert', '--node-files', nod, '--edge-files', edg,
                        '--output-file', net_path, '--no-turnarounds', 'true'],
                       check=True, capture_output=True, text=True)
    cfg_path = os.path.join(out_dir, f'{NET_NAME}.sumocfg')
    with open(cfg_path, 'w') as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>

<sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{NET_NAME}.net.xml"/>
        <!-- SVC background traffic is attached via traffic.route_file -->
    </input>

</sumoConfiguration>
''')
    return net_path


def screen():
    """Print candidate stats incl. the analytic hot-set headroom, best last."""
    import sumolib
    from eval_trsp import build_graph
    from scripts.gen_svc_hot_traffic import select_hot_edges
    rows = []
    for deletions in (10, 14, 18):
        for seed in range(20):
            r = build_lattice(seed, deletions)
            if r is None:
                continue
            pos, edges = r
            with tempfile.TemporaryDirectory() as tmp:
                net_path = write_net(pos, edges, tmp)
                gi = build_graph(net_path)
                net = sumolib.net.readNet(net_path)
                hot, pp, (td, to, h) = select_hot_edges(gi, net, 12)
            n = gi['G'].number_of_nodes()
            print(f"del={deletions} seed={seed}: nodes={n} und_edges={len(edges)} "
                  f"design t_dij={td:.0f}s t_opt={to:.0f}s headroom={h*100:.1f}%")
            rows.append((h, deletions, seed, n))
    rows.sort()
    print("\nBest:", [(f"{h*100:.1f}%", f"del={d}", f"seed={s}", f"n={n}")
                      for h, d, s, n in rows[-5:]])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--screen', action='store_true',
                    help='screen seeds/deletion counts instead of writing the net')
    args = ap.parse_args()
    if args.screen:
        screen()
    else:
        r = build_lattice(SEED, DELETE_LINKS)
        assert r is not None, "chosen SEED/DELETE_LINKS violates constraints"
        pos, edges = r
        net_path = write_net(pos, edges, OUT_DIR)
        print(f"wrote {net_path}")
