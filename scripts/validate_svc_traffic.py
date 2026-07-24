"""Validate an SVC traffic route file against the eval_trsp.py benchmark design.

Runs SUMO (no UGV) for the eval horizon with the candidate route file, collects
per-edge aggregated travel times (edgeData), then replays the 100 seed-42 greedy-
Dijkstra walks under the MEASURED travel times and compares with the time-optimal
paths under the same measurements:

    headroom = 1 - mean(t_optimal) / mean(t_dijkstra)

This is the ceiling any traffic-aware router (DQN included) can reach vs the static
Dijkstra baseline. Also reports hot-edge speeds, detour-edge speeds and teleports
(gridlock indicator).

Usage:
    python scripts/validate_svc_traffic.py <grid_road|rand_grid_road|ames> [tag]
"""
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_trsp import build_graph, select_od_pairs, INF, N_PAIRS, SEED  # noqa: E402
from scripts.gen_svc_hot_traffic import greedy_walk, NETWORKS  # noqa: E402

SUMO = 'sumo'
SIM_END = 7200          # eval session horizon (simulation.max_time = 2 h)
WARMUP = 300            # ignore the first window while flows fill in
AGG_PERIOD = 600        # edgeData aggregation window
V_FREE = 30.0
HOP_OVERHEAD = 3.0


def run_sumo(net_file, route_file, out_dir):
    add = os.path.join(out_dir, 'edgedata_add.xml')
    out = os.path.join(out_dir, 'edgedata.xml')
    stats = os.path.join(out_dir, 'stats.xml')
    with open(add, 'w') as f:
        f.write(f'<additional>\n  <edgeData id="ed" file="{out}" period="{AGG_PERIOD}" '
                f'begin="{WARMUP}" end="{SIM_END}" excludeEmpty="false" withInternal="false"/>\n'
                f'</additional>\n')
    cmd = [SUMO, '-n', net_file, '--route-files', route_file,
           '--additional-files', add, '--end', str(SIM_END),
           '--step-length', '1.0', '--no-step-log', '--duration-log.statistics',
           '--statistic-output', stats]
    res = subprocess.run(cmd, capture_output=True, text=True)
    errs = [l for l in (res.stderr or '').splitlines() if 'Error' in l]
    warns = [l for l in (res.stderr or '').splitlines() if 'teleport' in l.lower()]
    tele = 0
    if os.path.exists(stats):
        st = ET.parse(stats).getroot()
        ts = st.find('teleports')
        if ts is not None:
            tele = int(ts.get('total', 0))
    return out, errs, warns, tele


def measured_edge_times(edgedata_path, G_dir):
    """Per directed edge: mean travel time across aggregation windows (weighted by
    sampled vehicles); empty windows fall back to free-flow."""
    id2uv = {d['id']: (u, v) for u, v, d in G_dir.edges(data=True)}
    tt_sum = {}
    tt_n = {}
    for interval in ET.parse(edgedata_path).getroot().iter('interval'):
        for e in interval.iter('edge'):
            eid = e.get('id')
            if eid not in id2uv:
                continue
            tt = e.get('traveltime')
            n = float(e.get('sampledSeconds') or 0)
            if tt is None or n <= 0:
                continue
            tt_sum[eid] = tt_sum.get(eid, 0.0) + float(tt) * n
            tt_n[eid] = tt_n.get(eid, 0.0) + n
    times = {}
    for eid, (u, v) in id2uv.items():
        L = G_dir.edges[u, v]['length']
        free = L / V_FREE
        times[(u, v)] = max(tt_sum[eid] / tt_n[eid], free) if tt_n.get(eid) else free
    return times


def main(name, tag='svc_hot_42'):
    cfg = NETWORKS[name]
    net_file = cfg['net']
    base = os.path.dirname(net_file)
    route_file = os.path.join(base, 'traffic_routes', tag, 'rvt_routes.xml')
    hot_path = os.path.join(base, 'traffic_routes', tag, 'hot_edges.txt')
    hot_ids = set()
    if os.path.exists(hot_path):
        with open(hot_path) as f:
            hot_ids = {l.split('\t')[0] for l in f if l.strip()}

    out_dir = tempfile.mkdtemp(prefix=f'svc_val_{name}_')
    print(f"[{name}] running SUMO {SIM_END}s with {route_file} ...")
    edgedata, errs, warns, tele = run_sumo(net_file, route_file, out_dir)
    for l in errs[:10]:
        print("  ERROR:", l)
    print(f"  teleports: {tele}  (warnings mentioning teleport: {len(warns)})")

    gi = build_graph(net_file)
    G_dir, node2idx, idx2node = gi['G_dir'], gi['node2idx'], gi['idx2node']
    spl_dir = gi['all_shortest_path_lengths_dir']
    times = measured_edge_times(edgedata, G_dir)

    # hot vs non-hot measured speeds
    hs, cs = [], []
    for (u, v), t in times.items():
        L = G_dir.edges[u, v]['length']
        (hs if G_dir.edges[u, v]['id'] in hot_ids else cs).append(L / t)
    if hs:
        print(f"  hot edges   ({len(hs)}): mean speed {np.mean(hs):5.2f} m/s "
              f"(min {min(hs):.2f}, max {max(hs):.2f})")
    print(f"  other edges ({len(cs)}): mean speed {np.mean(cs):5.2f} m/s "
          f"(min {min(cs):.2f})")

    # headroom under measured times
    H = nx.DiGraph()
    for (u, v), t in times.items():
        H.add_edge(u, v, w=t + HOP_OVERHEAD)
    pairs = select_od_pairs(len(idx2node), seed=SEED, n_pairs=N_PAIRS)
    t_dij, t_opt, n_hit = [], [], 0
    for (o, d) in pairs:
        walk = greedy_walk(G_dir, node2idx, idx2node, spl_dir, idx2node[o], idx2node[d])
        if walk is None:
            continue
        td = sum(times[(u, v)] + HOP_OVERHEAD for u, v in walk)
        to = nx.dijkstra_path_length(H, walk[0][0], walk[-1][1], weight='w')
        t_dij.append(td)
        t_opt.append(to)
        if any(G_dir.edges[u, v]['id'] in hot_ids for u, v in walk):
            n_hit += 1
    t_dij, t_opt = np.array(t_dij), np.array(t_opt)
    hr = 1 - t_opt.mean() / t_dij.mean()
    print(f"  OD pairs whose Dijkstra path crosses >=1 hot edge: {n_hit}/{len(t_dij)}")
    print(f"  MEASURED: t_dij={t_dij.mean():6.1f}s  t_opt={t_opt.mean():6.1f}s  "
          f"headroom={hr*100:5.1f}%")
    return hr


if __name__ == '__main__':
    name = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else 'svc_hot_42'
    main(name, tag)
