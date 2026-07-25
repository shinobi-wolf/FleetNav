"""Generate DQN-favoring SVC background traffic (svc_hot_42) for the TRSP networks.

Design goal: maximize the makespan gap between the static greedy-Dijkstra baseline
and a traffic-aware router on the exact 100 seed-42 eval OD pairs of eval_trsp.py.

Key insight: the Dijkstra baseline's walk is traffic-independent (it always follows
best_dir = argmin neighbor SPL), so its edge set per OD pair is FIXED. We therefore
select the set of directed "hot" edges that maximizes
    mean(t_dijkstra) - mean(t_time_optimal)
under a congested-speed model, via greedy set growth — congesting exactly the edges
the baseline must cross while leaving cheap detours untouched.

Congestion mechanism: continuous flows of slow vehicles (vType maxSpeed = V_HOT) on
explicit routes over the hot-edge chains. On 1-lane roads they are un-overtakable
moving bottlenecks -> persistent ~V_HOT edge speeds with no queue spillback and no
gridlock (they never stop). Only 1-lane edges are eligible as hot edges. A light
ambient randomTrips layer is added so neighbor_speeds stays informative off-corridor.

Outputs per network (run from repo root):
    data/<net>/traffic_routes/svc_hot_42/rvt_routes.xml   (vTypes + hot flows + ambient)
    data/<net>/traffic_routes/svc_hot_42/hot_edges.txt    (selected directed edges)
    data/<net>/traffic_routes/svc_hot_42/params.yaml      (written by hand, see repo)

Usage:
    python scripts/gen_svc_hot_traffic.py [grid_road|rand_grid_road|ames|all]
"""
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import networkx as nx
import sumolib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_trsp import build_graph, select_od_pairs, INF, N_PAIRS, SEED  # noqa: E402

V_FREE = 30.0        # m/s — lane speed limit on all three nets (post lane-speed fix)
V_HOT = 5.0          # m/s — slow-vehicle max speed (moving bottleneck)
HOP_OVERHEAD = 3.0   # s   — per-hop insertion/accel cost used in the design model
HORIZON = 20000      # s   — flows/trips end time (covers eval max_time 7200s with slack)
MIN_PERIOD = 10.0    # s — lower bound on flow insertion period (short-edge chains)

RANDOM_TRIPS = '/usr/share/sumo/tools/randomTrips.py'

NETWORKS = {
    'grid_road': dict(
        net='data/grid_road/grid_road.net.xml',
        n_hot=10,           # analytic headroom ~51%
        occupancy=2.0,      # 990m edges: tight spacing so mid-gap entries get caught
        ambient_period=4.0, # light ambient: ~25 concurrent on 80 edges
        ambient_min_dist=1500.0,
    ),
    'rand_grid_road': dict(
        net='data/rand_grid_road/rand_grid_road_final.net.xml',
        n_hot=16,           # short edges -> more of them needed; ~45%
        occupancy=1.0,      # 81-212m edges store few queued cars — higher occupancy
                            # plus dense ambient gridlocked the whole net (349 teleports)
        ambient_period=8.0,
        ambient_min_dist=300.0,
    ),
    'ames': dict(
        net='data/ames/ames_campus_final.net.xml',
        n_hot=8,            # long arterials dominate; ~53%
        occupancy=2.0,
        ambient_period=5.0, # matches old svc_42 density
        ambient_min_dist=500.0,
    ),
    'rand_grid_road_svc': dict(
        net='data/rand_grid_road_svc/rand_grid_road_svc.net.xml',
        n_hot=12,           # jittered 6x6 lattice, 386-806m edges; design ~56%
        occupancy=2.0,      # long edges again — tight bottleneck spacing is safe
        ambient_period=5.0,
        ambient_min_dist=1200.0,
    ),
}


# ── Hot-set selection (mirrors the greedy-Dijkstra eval baseline exactly) ────

def greedy_walk(G_dir, node2idx, idx2node, spl_dir, start_node, target_node,
                num_actions=5, max_steps=50):
    curr = start_node
    edges = []
    for _ in range(max_steps):
        if curr == target_node:
            return edges
        nb_idxs = sorted(node2idx[nb] for nb in G_dir.neighbors(curr))[:num_actions]
        if not nb_idxs:
            return None
        s2d = [spl_dir.get(idx2node[i], {}).get(target_node, INF) for i in nb_idxs]
        nxt = idx2node[nb_idxs[int(np.argmin(s2d))]]
        edges.append((curr, nxt))
        curr = nxt
    return None if curr != target_node else edges


def edge_time(G_dir, u, v, hot):
    L = G_dir.edges[u, v]['length']
    return L / (V_HOT if (u, v) in hot else V_FREE) + HOP_OVERHEAD


def headroom(G_dir, pair_paths, hot):
    H = nx.DiGraph()
    for u, v in G_dir.edges():
        H.add_edge(u, v, w=edge_time(G_dir, u, v, hot))
    t_dij, t_opt = [], []
    for walk in pair_paths.values():
        td = sum(edge_time(G_dir, u, v, hot) for u, v in walk)
        try:
            to = nx.dijkstra_path_length(H, walk[0][0], walk[-1][1], weight='w')
        except nx.NetworkXNoPath:
            continue
        t_dij.append(td)
        t_opt.append(to)
    td, to = float(np.mean(t_dij)), float(np.mean(t_opt))
    return td, to, 1.0 - to / td


def select_hot_edges(gi, net, n_hot):
    """Greedy headroom-maximizing growth over 1-lane directed edges used >=3 times."""
    G_dir, node2idx, idx2node = gi['G_dir'], gi['node2idx'], gi['idx2node']
    spl_dir = gi['all_shortest_path_lengths_dir']
    pairs = select_od_pairs(len(idx2node), seed=SEED, n_pairs=N_PAIRS)

    pair_paths = {}
    usage = {}
    for (o, d) in pairs:
        w = greedy_walk(G_dir, node2idx, idx2node, spl_dir, idx2node[o], idx2node[d])
        if w is None:
            continue
        pair_paths[(o, d)] = w
        for e in w:
            usage[e] = usage.get(e, 0) + 1

    def one_lane(u, v):
        return len(net.getEdge(G_dir.edges[u, v]['id']).getLanes()) == 1

    cands = [e for e, c in usage.items() if c >= 3 and one_lane(*e)]
    hot = set()
    for _ in range(n_hot):
        best = None
        for e in cands:
            if e in hot:
                continue
            _, _, h = headroom(G_dir, pair_paths, hot | {e})
            if best is None or h > best[1]:
                best = (e, h)
        if best is None:
            break
        hot.add(best[0])
    td, to, h = headroom(G_dir, pair_paths, hot)
    return hot, pair_paths, (td, to, h)


# ── Flow construction ────────────────────────────────────────────────────────

def build_chains(G_dir, hot):
    """Group the directed hot edges into maximal simple chains (flow routes)."""
    out_map = {u: v for (u, v) in hot}   # may overwrite on branching — handled below
    # Build adjacency restricted to hot edges
    succ = {}
    pred_count = {}
    for (u, v) in hot:
        succ.setdefault(u, []).append(v)
        pred_count[v] = pred_count.get(v, 0) + 1
    remaining = set(hot)
    chains = []
    # chain starts: hot edges whose from-node is not the to-node of another hot edge,
    # or branch points
    while remaining:
        starts = [(u, v) for (u, v) in remaining if pred_count.get(u, 0) == 0]
        if not starts:  # pure cycle — break it anywhere
            starts = [sorted(remaining)[0]]
        u, v = sorted(starts)[0]
        chain = [(u, v)]
        remaining.discard((u, v))
        while True:
            nxts = [w for w in succ.get(v, []) if (v, w) in remaining]
            if len(nxts) != 1:
                break
            w = nxts[0]
            if w == chain[-1][0]:
                # U-turn continuation (…->X->v then v->X): nets built with
                # --no-turnarounds have no such connection — keep as separate flows
                break
            chain.append((v, w))
            remaining.discard((v, w))
            v = w
        chains.append(chain)
    return chains


def emit_hot_flows(G_dir, chains, occupancy):
    """Return (route_elems, flow_elems). Period keeps ~occupancy slow veh per edge."""
    routes, flows = [], []
    for i, chain in enumerate(chains):
        edge_ids = [G_dir.edges[u, v]['id'] for (u, v) in chain]
        lens = [G_dir.edges[u, v]['length'] for (u, v) in chain]
        # spacing so that each edge of the chain holds ~occupancy vehicles
        period = max(MIN_PERIOD, round(min(lens) / V_HOT / occupancy, 1))
        routes.append((f'hot_route_{i}', ' '.join(edge_ids)))
        flows.append((f'hot_flow_{i}', f'hot_route_{i}', period))
    return routes, flows


def gen_ambient(net_file, out_dir, period, min_dist):
    """Light ambient traffic via randomTrips + duarouter (seed 42, validated)."""
    trips = os.path.join(out_dir, 'ambient_trips.xml')
    routes = os.path.join(out_dir, 'ambient_routes.xml')
    cmd = [sys.executable, RANDOM_TRIPS,
           '-n', net_file, '-e', str(HORIZON), '-p', str(period),
           '--min-distance', str(min_dist), '--fringe-factor', '2',
           '-s', str(SEED), '-o', trips, '-r', routes, '--validate']
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return routes


def merge_route_file(out_path, hot_routes, hot_flows, ambient_routes_path):
    """Single sorted route file: vTypes, hot routes+flows (begin=0), ambient vehicles."""
    root = ET.Element('routes')
    ET.SubElement(root, 'vType', id='ambient_car', vClass='passenger')
    ET.SubElement(root, 'vType', id='slow_veh', vClass='passenger',
                  maxSpeed=str(V_HOT), speedFactor='1', speedDev='0',
                  color='255,0,255', length='6.0')
    for rid, edges in hot_routes:
        ET.SubElement(root, 'route', id=rid, edges=edges)
    for fid, rid, period in hot_flows:
        ET.SubElement(root, 'flow', id=fid, route=rid, type='slow_veh',
                      begin='0', end=str(HORIZON), period=str(period),
                      departSpeed='max')
    amb = ET.parse(ambient_routes_path).getroot()
    for veh in amb.iter('vehicle'):
        veh.set('type', 'ambient_car')
        root.append(veh)
    ET.indent(root)
    ET.ElementTree(root).write(out_path, xml_declaration=True, encoding='UTF-8')


def generate(name):
    cfg = NETWORKS[name]
    net_file = cfg['net']
    out_dir = os.path.join(os.path.dirname(net_file), 'traffic_routes', 'svc_hot_42')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {name} ===")
    gi = build_graph(net_file)
    net = sumolib.net.readNet(net_file)
    hot, pair_paths, (td, to, h) = select_hot_edges(gi, net, cfg['n_hot'])
    G_dir = gi['G_dir']
    print(f"hot edges ({len(hot)}): "
          f"{[G_dir.edges[u, v]['id'] for (u, v) in sorted(hot)]}")
    print(f"design model: t_dij={td:.1f}s  t_opt={to:.1f}s  headroom={h*100:.1f}%")

    chains = build_chains(G_dir, hot)
    hot_routes, hot_flows = emit_hot_flows(G_dir, chains, cfg['occupancy'])
    for (fid, rid, period), chain in zip(hot_flows, chains):
        eids = [G_dir.edges[u, v]['id'] for (u, v) in chain]
        print(f"  {fid}: period={period}s  edges={eids}")

    ambient = gen_ambient(net_file, out_dir, cfg['ambient_period'], cfg['ambient_min_dist'])
    out_path = os.path.join(out_dir, 'rvt_routes.xml')
    merge_route_file(out_path, hot_routes, hot_flows, ambient)

    with open(os.path.join(out_dir, 'hot_edges.txt'), 'w') as f:
        for (u, v) in sorted(hot):
            f.write(f"{G_dir.edges[u, v]['id']}\t{u}\t{v}\n")
    print(f"wrote {out_path}")
    return out_path


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    targets = list(NETWORKS) if which == 'all' else [which]
    for t in targets:
        generate(t)
