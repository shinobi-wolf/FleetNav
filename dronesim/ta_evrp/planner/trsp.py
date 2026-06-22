"""TRSP-based local planner for EVRP edge routing."""
from __future__ import annotations

from typing import Optional

import numpy as np
import networkx as nx
import gymnasium as gym
import yaml

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration

INF = 100000
_MAX_PLAN_STEPS = 500

# SARSA-family: predict() returns index into sorted(nb_idxs[:num_actions])
_NEIGHBOR_IDX_ALGOS = {"sarsa", "sarsa_lambda"}
# Deep RL + DSP: predict() returns global node index masked to valid neighbours
_NODE_IDX_ALGOS = {"dqn", "a2c", "ppo", "dqn_svc", "dsp"}


def _build_agent(algo: str, trsp_config: dict):
    """Instantiate + load the correct agent.  Returns (agent, action_mode) or (None, None)."""
    if algo == "sarsa":
        from dronesim.trsp.agents import SARSAAgent
        return SARSAAgent(trsp_config), "neighbor"

    if algo == "dqn":
        from dronesim.trsp.agents import DQNAgent
        return DQNAgent(trsp_config), "node"

    if algo == "a2c":
        from dronesim.trsp.agents import A2CAgent
        return A2CAgent(trsp_config), "node"

    if algo == "ppo":
        from dronesim.trsp.agents import PPOAgent
        return PPOAgent(trsp_config), "node"

    if algo == "dqn_svc":
        from dronesim.trsp.agents import DQNSVCAgent
        return DQNSVCAgent(trsp_config), "node"

    return None, None


class LocalPlanner:
    """
    Routes node-to-node via a trained TRSP RL agent (graph-only for PSL; TraCI-backed for SVC).
    Falls back to Dijkstra on cycles, dead-ends, invalid actions, or missing policy.

    Supported algos: sarsa, dqn, a2c, ppo, dqn_svc, dsp.

    `dsp` (Dynamic Shortest Path) is a non-learning, traffic-aware baseline: it
    queries live TraCI edge speeds once per leg, re-weights the graph by travel
    time, and follows the resulting Dijkstra path hop-by-hop. Like dqn_svc, it
    must run inside the SUMO session (LocalPlanner is called from Simulator.run).
    """

    def __init__(self, policy_id: str) -> None:
        config_path = f"runs/trsp/{policy_id}/config.yaml"
        with open(config_path) as f:
            trsp_config = yaml.safe_load(f)

        algo = (trsp_config.get("agent") or {}).get("algo", "sarsa")
        self.algo = algo

        self._agent = None
        self._action_mode: str = "neighbor"

        if algo == "dsp":
            # DSP is non-learning: it needs the env's graph maps at construction
            # time (it has no internal env) and queries live TraCI edge speeds
            # once per leg. Build the env first, then hand its graph to the agent.
            from dronesim.trsp.agents import DSPAgent
            wrapped = gym.make("SumoTRSPEnv-v0", config=trsp_config)
            env = wrapped.unwrapped
            self._agent = DSPAgent(0, env.G_dir, env.node2idx, env.idx2node, env.ugv_speed)
            self._action_mode = "node"
        else:
            try:
                self._agent, self._action_mode = _build_agent(algo, trsp_config)
            except Exception as exc:
                print(f"LocalPlanner: failed to load '{algo}' agent — {exc}; using Dijkstra fallback")

            if self._agent is not None:
                env = self._agent.env.unwrapped
            else:
                # No agent (unknown algo or load failure) — still need graph for Dijkstra
                wrapped = gym.make("SumoTRSPEnv-v0", config=trsp_config)
                env = wrapped.unwrapped

        self.G: nx.Graph              = env.G
        self.G_dir: nx.DiGraph        = env.G_dir
        self.idx2node: dict           = env.idx2node
        self.node2idx: dict           = env.node2idx
        self.all_spl: dict            = env.all_shortest_path_lengths
        self.num_actions: int         = env.num_actions
        self.traffic                  = env.traffic

        # DQN-SVC needs TraCI neighbor speeds; store max_degree and ugv_speed for obs building
        if algo == "dqn_svc" and self._agent is not None:
            self.max_degree: int   = self._agent.max_degree
            self.ugv_speed: float  = float(
                trsp_config.get("fleet", {}).get("max_speed", 110.0)
            ) / 3.6   # m/s

        mode = "Dijkstra" if self._agent is None else f"{algo.upper()} ({self._action_mode}-index)"
        print(f"TRSPLocalPlanner: policy='{policy_id}'  algo={algo}  mode={mode}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, src_node: str, dst_node: str, sim_time_s: float) -> list[str]:
        """Return SUMO edge ID list from src_node to dst_node."""
        edges, _nodes, _meta = self.plan_edges_and_nodes(src_node, dst_node, sim_time_s)
        return edges

    def plan_edges_and_nodes(
        self, src_node: str, dst_node: str, sim_time_s: float
    ) -> tuple[list[str], list[str], dict]:
        """Return (edge_ids, node_ids, meta) for the leg.

        edge_ids : SUMO edge IDs in traversal order
        node_ids : SUMO junction IDs in traversal order (includes src and dst)
        meta     : {traffic_slot, fallback_reason}
        """
        slot = self.traffic.get_slot(sim_time_s) if self.traffic.enabled else 0

        if src_node == dst_node:
            return [], [src_node], {"traffic_slot": int(slot), "fallback_reason": None}

        if self._agent is None:
            return self._dijkstra_edges_and_nodes(src_node, dst_node, slot=slot, reason="no_agent")

        # DSP-family agents plan the full leg up-front from live edge speeds;
        # the hop-by-hop rollout below then just follows that planned path.
        if hasattr(self._agent, "start_leg"):
            self._agent.start_leg(src_node, dst_node)

        curr_node  = src_node
        curr_pos   = self.node2idx[curr_node]
        target_idx = self.node2idx[dst_node]
        visited: set[str] = set()
        edges: list[str]  = []
        nodes: list[str]  = [src_node]

        for _ in range(_MAX_PLAN_STEPS):
            if curr_node == dst_node:
                return edges, nodes, {"traffic_slot": int(slot), "fallback_reason": None}

            if curr_node in visited:
                return self._dijkstra_edges_and_nodes(
                    src_node, dst_node, slot=slot, reason="cycle_detected"
                )
            visited.add(curr_node)

            nb_nodes = list(self.G.neighbors(curr_node))
            if not nb_nodes:
                return self._dijkstra_edges_and_nodes(
                    src_node, dst_node, slot=slot, reason="dead_end"
                )

            if self.algo == "dqn_svc":
                ns  = self._neighbor_speeds(curr_node)
                obs = {"graph_node": curr_pos, "target_node": target_idx, "neighbor_speeds": ns}
            else:
                obs = {"graph_node": curr_pos, "target_node": target_idx, "time_slot": int(slot)}
            next_pos_idx = self._select_next(obs, nb_nodes, dst_node)

            next_node = self.idx2node[next_pos_idx]
            edge_data = self.G_dir.get_edge_data(curr_node, next_node)
            if edge_data is None:
                return self._dijkstra_edges_and_nodes(
                    src_node, dst_node, slot=slot, reason="missing_edge_data"
                )
            edges.append(edge_data["id"])
            nodes.append(next_node)
            curr_node = next_node
            curr_pos  = next_pos_idx

        return self._dijkstra_edges_and_nodes(
            src_node, dst_node, slot=slot, reason="max_steps_exceeded"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_next(self, obs: dict, nb_nodes: list, dst_node: str) -> int:
        """Return the global node index of the next hop chosen by the agent."""
        all_nb_idxs = sorted([self.node2idx[nb] for nb in nb_nodes])

        if self._action_mode == "neighbor":
            # SARSA: action is an index into the first num_actions sorted neighbours
            nb_idxs = all_nb_idxs[: self.num_actions]
            steps   = [
                self.all_spl.get(self.idx2node[ni], {}).get(dst_node, INF)
                for ni in nb_idxs
            ]
            steps  += [INF] * (self.num_actions - len(steps))
            best    = int(np.argmin(steps))
            try:
                action = int(self._agent.predict(obs))
            except Exception:
                action = best
            if action >= len(nb_idxs):
                action = best
            return nb_idxs[action]

        else:
            # DQN / A2C / PPO: action is a global node index for any valid neighbour
            nb_idxs_set = set(all_nb_idxs)
            steps       = [
                self.all_spl.get(self.idx2node[ni], {}).get(dst_node, INF)
                for ni in all_nb_idxs
            ]
            best_nb = all_nb_idxs[int(np.argmin(steps))]
            try:
                action = int(self._agent.predict(obs))
            except Exception:
                action = best_nb
            return action if action in nb_idxs_set else best_nb

    def _neighbor_speeds(self, curr_node: str) -> np.ndarray:
        """Query TraCI for outgoing-edge mean speeds from curr_node.

        Called only when algo == 'dqn_svc'; SUMO must be running (called from
        inside Simulator.run()). Mirrors how SumoTRSPEnv builds neighbor_speeds:
        sorted by node2idx, normalized by ugv_speed, clamped to [0, 2], padded
        with 1.0 (free-flow) for missing slots.
        """
        import traci  # noqa: PLC0415 — intentional late import; only used for SVC
        speeds    = np.ones(self.max_degree, dtype=np.float32)
        nb_sorted = sorted(self.G.neighbors(curr_node), key=lambda n: self.node2idx[n])
        for i, nb in enumerate(nb_sorted[: self.max_degree]):
            edge_data = self.G_dir.get_edge_data(curr_node, nb)
            if edge_data:
                raw = traci.edge.getLastStepMeanSpeed(edge_data["id"])
                speeds[i] = float(np.clip(raw / self.ugv_speed, 0.0, 2.0)) if raw >= 0.0 else 1.0
        return speeds

    def _dijkstra_edges_and_nodes(
        self, src_node: str, dst_node: str, *, slot: int, reason: Optional[str]
    ) -> tuple[list[str], list[str], dict]:
        try:
            node_path = nx.shortest_path(self.G_dir, src_node, dst_node, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], [src_node], {"traffic_slot": int(slot), "fallback_reason": reason or "no_path"}

        edges = [
            self.G_dir.get_edge_data(u, v)["id"]
            for u, v in zip(node_path[:-1], node_path[1:])
            if self.G_dir.get_edge_data(u, v)
        ]
        return edges, list(node_path), {"traffic_slot": int(slot), "fallback_reason": reason}
