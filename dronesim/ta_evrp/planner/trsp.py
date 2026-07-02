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
_NEIGHBOR_IDX_ALGOS = {"sarsa", "sarsa_lambda", "sarsa_svc", "sarsa_ped"}
# Deep RL + DSP: predict() returns global node index masked to valid neighbours
_NODE_IDX_ALGOS = {"dqn", "a2c", "ppo", "dqn_svc", "a2c_svc", "ppo_svc", "dsp",
                   "dqn_ped", "dsp_ped"}
# SVC agents: obs includes neighbor_speeds (live vehicular) instead of time_slot
_SVC_ALGOS = {"dqn_svc", "sarsa_svc", "a2c_svc", "ppo_svc"}
# PED agents: obs includes neighbor_speeds (live Weidmann sidewalk) instead of time_slot
_PED_OBS_ALGOS = {"dqn_ped", "sarsa_ped"}
# Any agent whose observation carries a live neighbor_speeds vector
_LIVE_OBS_ALGOS = _SVC_ALGOS | _PED_OBS_ALGOS


def _build_agent(algo: str, trsp_config: dict):
    """Instantiate + load the correct agent.  Returns (agent, action_mode) or (None, None)."""
    if algo == "sarsa":
        from dronesim.trsp.agents import SARSAAgent
        return SARSAAgent(trsp_config), "neighbor"

    if algo == "sarsa_svc":
        from dronesim.trsp.agents import SARSASVCAgent
        return SARSASVCAgent(trsp_config), "neighbor"

    if algo == "sarsa_ped":
        from dronesim.trsp.agents import SARSAPEDAgent
        return SARSAPEDAgent(trsp_config), "neighbor"

    if algo == "dqn":
        from dronesim.trsp.agents import DQNAgent
        return DQNAgent(trsp_config), "node"

    if algo == "dqn_svc":
        from dronesim.trsp.agents import DQNSVCAgent
        return DQNSVCAgent(trsp_config), "node"

    if algo == "dqn_ped":
        from dronesim.trsp.agents import DQNPEDAgent
        return DQNPEDAgent(trsp_config), "node"

    if algo == "a2c":
        from dronesim.trsp.agents import A2CAgent
        return A2CAgent(trsp_config), "node"

    if algo == "a2c_svc":
        from dronesim.trsp.agents import A2CSVCAgent
        return A2CSVCAgent(trsp_config), "node"

    if algo == "ppo":
        from dronesim.trsp.agents import PPOAgent
        return PPOAgent(trsp_config), "node"

    if algo == "ppo_svc":
        from dronesim.trsp.agents import PPOSVCAgent
        return PPOSVCAgent(trsp_config), "node"

    return None, None


class LocalPlanner:
    """
    Routes node-to-node via a trained TRSP RL agent (graph-only for PSL; TraCI-backed
    for SVC/PED). Falls back to Dijkstra on cycles, dead-ends, invalid actions, or
    missing policy.

    Supported algos: sarsa, sarsa_svc, sarsa_ped, dqn, dqn_svc, dqn_ped, a2c, a2c_svc,
    ppo, ppo_svc, dsp, dsp_ped.

    `dsp` / `dsp_ped` (Dynamic Shortest Path) are non-learning, traffic-aware baselines:
    they query live TraCI speeds once per leg (vehicular edge speeds for `dsp`; the
    robot's Weidmann sidewalk speed for `dsp_ped`), re-weight the graph by travel time,
    and follow the resulting Dijkstra path hop-by-hop. Like the *_svc / *_ped agents,
    they must run inside the SUMO session (LocalPlanner is called from Simulator.run).
    """

    def __init__(self, policy_id: str) -> None:
        config_path = f"runs/trsp/{policy_id}/config.yaml"
        with open(config_path) as f:
            trsp_config = yaml.safe_load(f)

        algo = (trsp_config.get("agent") or {}).get("algo", "sarsa")
        self.algo = algo

        self._agent = None
        self._action_mode: str = "neighbor"
        self._lane_area_cache: dict[str, float] = {}

        if algo in ("dsp", "dsp_ped"):
            # DSP is non-learning: it needs the env's graph maps at construction time
            # (it has no internal env) and queries live TraCI speeds once per leg.
            # Build the env first, then hand its graph to the agent.
            wrapped = gym.make("SumoTRSPEnv-v0", config=trsp_config)
            env = wrapped.unwrapped
            if algo == "dsp_ped":
                from dronesim.trsp.agents.dsp_ped import DSPPEDAgent
                pso = (trsp_config.get("traffic", {}) or {}).get("ped_speed_override", {})
                self._agent = DSPPEDAgent(
                    0, env.G_dir, env.node2idx, env.idx2node, env.ugv_speed,
                    gamma=pso.get("gamma", 1.913),
                    density_jam=pso.get("density_jam", 5.4),
                    v_min=pso.get("v_min", 0.1),
                )
            else:
                from dronesim.trsp.agents import DSPAgent
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

        # Live-speed agents (svc/ped) need per-hop neighbor speeds; store the graph
        # fan-out and the free-flow speed used to normalize them. For ped agents,
        # capture the Weidmann override params so obs speeds match the training env.
        self._ped: bool = algo in _PED_OBS_ALGOS or algo == "dsp_ped"
        self._ped_params: Optional[tuple] = None
        if algo in _LIVE_OBS_ALGOS and self._agent is not None:
            self.max_degree: int  = int(trsp_config["net"]["max_degree"])
            self.ugv_speed: float = float(
                trsp_config.get("fleet", {}).get("max_speed", 110.0)
            ) / 3.6   # m/s
            if algo in _PED_OBS_ALGOS:
                pso = (trsp_config.get("traffic", {}) or {}).get("ped_speed_override", {})
                if bool(pso.get("enabled", False)):
                    self._ped_params = (
                        pso.get("gamma", 1.913),
                        pso.get("density_jam", 5.4),
                        pso.get("v_min", 0.1),
                    )

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

            if self.algo in _LIVE_OBS_ALGOS:
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
        """Query TraCI for outgoing-edge speeds from curr_node.

        Called for all live-speed algos; SUMO must be running (called from
        inside Simulator.run()). Mirrors how SumoTRSPEnv builds neighbor_speeds:
        sorted by node2idx, normalized by ugv_speed, clamped to [0, 2], padded
        with 1.0 (free-flow) for missing slots. For ped agents the neighbour
        speed is the robot's Weidmann sidewalk speed (the speed it will actually
        get), not the vehicular edge mean speed.
        """
        import traci  # noqa: PLC0415 — intentional late import; only used for live models
        speeds    = np.ones(self.max_degree, dtype=np.float32)
        nb_sorted = sorted(self.G.neighbors(curr_node), key=lambda n: self.node2idx[n])
        for i, nb in enumerate(nb_sorted[: self.max_degree]):
            edge_data = self.G_dir.get_edge_data(curr_node, nb)
            if not edge_data:
                continue
            if self._ped:
                lane_id = edge_data.get("sidewalk_lane")
                if not lane_id:
                    raw = self.ugv_speed
                elif self._ped_params is not None:
                    raw = self._ped_sidewalk_speed(lane_id)
                else:
                    raw = traci.lane.getLastStepMeanSpeed(lane_id)
            else:
                raw = traci.edge.getLastStepMeanSpeed(edge_data["id"])
            speeds[i] = float(np.clip(raw / self.ugv_speed, 0.0, 2.0)) if raw >= 0.0 else 1.0
        return speeds

    def _ped_sidewalk_speed(self, lane_id: str) -> float:
        """Robot Weidmann sidewalk speed (m/s) from live person counts on the edge."""
        import traci  # noqa: PLC0415 — intentional late import; only used for ped models
        from dronesim.trsp.ped_speed import weidmann_speed
        gamma, density_jam, v_min = self._ped_params
        n = len(traci.edge.getLastStepPersonIDs(traci.lane.getEdgeID(lane_id)))
        area = self._lane_area_cache.get(lane_id)
        if area is None:
            area = traci.lane.getLength(lane_id) * traci.lane.getWidth(lane_id)
            self._lane_area_cache[lane_id] = area
        return weidmann_speed(n, area, self.ugv_speed, gamma, density_jam, v_min)

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
