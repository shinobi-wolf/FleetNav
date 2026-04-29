"""TRSP-based local planner for EVRP edge routing."""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import networkx as nx
import gymnasium as gym
import yaml

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration

INF = 100000
_MAX_PLAN_STEPS = 500


class TRSPLocalPlanner:
    """
    Runs a trained TRSP RL agent on the road graph (no SUMO) to produce an
    edge ID list from src_node to dst_node.  Falls back to Dijkstra if the
    agent cycles or reaches max steps without arriving.

    Only SARSA-style Q-table policies (policy.pkl = dict) are supported.
    For other agents the planner degrades gracefully to Dijkstra.
    """

    def __init__(self, policy_id: str) -> None:
        config_path = f"runs/trsp/{policy_id}/config.yaml"
        with open(config_path) as f:
            trsp_config = yaml.safe_load(f)

        # Instantiate env to extract graph and traffic model; SUMO is NOT started.
        wrapped = gym.make("SumoTRSPEnv-v0", config=trsp_config)
        env = wrapped.unwrapped

        self.G: nx.Graph = env.G                    # undirected
        self.G_dir: nx.DiGraph = env.G_dir          # directed
        self.idx2node: dict = env.idx2node
        self.node2idx: dict = env.node2idx
        self.all_spl: dict = env.all_shortest_path_lengths  # directed SPL
        self.num_actions: int = env.num_actions
        self.traffic = env.traffic

        policy_path = f"runs/trsp/{policy_id}/policy.pkl"
        self._Q: dict = {}
        if os.path.exists(policy_path):
            with open(policy_path, "rb") as f:
                self._Q = pickle.load(f)

        print(f"TRSPLocalPlanner: loaded policy '{policy_id}'  Q-states={len(self._Q)}")

    def plan(self, src_node: str, dst_node: str, sim_time_s: float) -> list[str]:
        """Return SUMO edge ID list from src_node to dst_node using the RL policy."""
        edges, _nodes, _meta = self.plan_edges_and_nodes(src_node, dst_node, sim_time_s)
        return edges

    def plan_edges_and_nodes(
        self, src_node: str, dst_node: str, sim_time_s: float
    ) -> tuple[list[str], list[str], dict]:
        """Return (edge_ids, node_ids, meta) for the leg.

        - edge_ids: SUMO edge IDs in traversal order
        - node_ids: SUMO node/junction IDs in traversal order (includes src and dst)
        - meta: {traffic_slot, fallback_reason}
        """
        if src_node == dst_node:
            slot = self.traffic.get_slot(sim_time_s) if self.traffic.enabled else 0
            return [], [src_node], {"traffic_slot": int(slot), "fallback_reason": None}

        slot = self.traffic.get_slot(sim_time_s) if self.traffic.enabled else 0
        curr_node = src_node
        curr_pos = self.node2idx[curr_node]
        target_idx = self.node2idx[dst_node]
        visited: set[str] = set()

        edges: list[str] = []
        nodes: list[str] = [src_node]

        for _ in range(_MAX_PLAN_STEPS):
            if curr_node == dst_node:
                return edges, nodes, {"traffic_slot": int(slot), "fallback_reason": None}

            if curr_node in visited:
                return self._dijkstra_edges_and_nodes(src_node, dst_node, slot=slot, reason="cycle_detected")
            visited.add(curr_node)

            nb_nodes = list(self.G.neighbors(curr_node))
            if not nb_nodes:
                return self._dijkstra_edges_and_nodes(src_node, dst_node, slot=slot, reason="dead_end")
            nb_idxs = sorted([self.node2idx[nb] for nb in nb_nodes])[: self.num_actions]

            steps = [
                self.all_spl.get(self.idx2node[ni], {}).get(dst_node, INF)
                for ni in nb_idxs
            ]
            steps += [INF] * (self.num_actions - len(steps))
            best_dir = int(np.argmin(steps))

            state = (curr_pos, target_idx, slot)
            if state in self._Q:
                q = self._Q[state][: len(nb_idxs)]
                action = int(np.argmax(q))
            else:
                action = best_dir

            if action >= len(nb_idxs):
                action = best_dir

            next_pos_idx = nb_idxs[action]
            next_node = self.idx2node[next_pos_idx]

            edge_data = self.G_dir.get_edge_data(curr_node, next_node)
            if edge_data is None:
                return self._dijkstra_edges_and_nodes(src_node, dst_node, slot=slot, reason="missing_edge_data")
            edges.append(edge_data["id"])
            nodes.append(next_node)

            curr_node = next_node
            curr_pos = next_pos_idx

        return self._dijkstra_edges_and_nodes(src_node, dst_node, slot=slot, reason="max_steps_exceeded")

    def _dijkstra_edges(self, src_node: str, dst_node: str) -> list[str]:
        edges, _nodes, _meta = self._dijkstra_edges_and_nodes(src_node, dst_node, slot=0, reason=None)
        return edges

    def _dijkstra_edges_and_nodes(
        self, src_node: str, dst_node: str, *, slot: int, reason: Optional[str]
    ) -> tuple[list[str], list[str], dict]:
        try:
            node_path = nx.shortest_path(self.G_dir, src_node, dst_node, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], [src_node], {"traffic_slot": int(slot), "fallback_reason": reason or "no_path"}

        edges: list[str] = []
        for u, v in zip(node_path[:-1], node_path[1:]):
            edata = self.G_dir.get_edge_data(u, v)
            if edata:
                edges.append(edata["id"])

        meta = {"traffic_slot": int(slot), "fallback_reason": reason}
        return edges, list(node_path), meta
