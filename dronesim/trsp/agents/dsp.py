import traci
import networkx as nx
from dronesim.trsp.agents.agent_psl import Agent

MIN_SPEED_MS = 0.1  # floor to avoid division by zero on fully jammed edges


class DSPAgent(Agent):
    """
    Dynamic Shortest Path agent.

    At the start of each OD leg, queries TraCI for the current mean speed on
    every edge, reweights the graph by travel time (length / speed), and runs
    Dijkstra once to obtain the full path.  The path is followed hop-by-hop
    for the rest of the leg without re-querying SUMO.

    Returns actions in node_id mode (global node index).
    """

    def __init__(self, agent_id, G_dir, node2idx, idx2node, ugv_speed):
        super().__init__()
        self.agent_id = agent_id
        self.G_dir = G_dir
        self.node2idx = node2idx
        self.idx2node = idx2node
        self.ugv_speed = ugv_speed

        self._planned_path: list[str] = []
        self._path_pos: int = 0

    # ------------------------------------------------------------------ leg API

    def start_leg(self, src_node: str, dst_node: str):
        """Query live edge speeds, plan the full path from src to dst via Dijkstra."""
        weights = self._build_weights()
        self._planned_path = self._dijkstra(src_node, dst_node, weights)
        self._path_pos = 0

    # ------------------------------------------------------------------ agent API

    def predict(self, observation, **kwargs):
        curr_node = self.idx2node[observation['graph_node']]

        if not self._planned_path or self._path_pos >= len(self._planned_path) - 1:
            return self._fallback(curr_node, observation, **kwargs)

        if self._planned_path[self._path_pos] != curr_node:
            try:
                self._path_pos = self._planned_path.index(curr_node)
            except ValueError:
                return self._fallback(curr_node, observation, **kwargs)

        next_node = self._planned_path[self._path_pos + 1]
        self._path_pos += 1
        return self.node2idx[next_node]

    def learn(self, *args, **kwargs):
        pass

    def _observation_to_state(self, observation):
        pass

    # ------------------------------------------------------------------ internals

    def _build_weights(self) -> dict:
        """Return {(u, v): travel_time_s} using live TraCI mean speeds."""
        weights = {}
        for u, v, data in self.G_dir.edges(data=True):
            eid = data['id']
            speed = traci.edge.getLastStepMeanSpeed(eid)
            if speed <= 0:
                speed = self.ugv_speed
            speed = max(speed, MIN_SPEED_MS)
            weights[(u, v)] = data['length'] / speed
        return weights

    def _dijkstra(self, src: str, dst: str, weights: dict) -> list[str]:
        def weight_fn(u, v, data):
            return weights.get((u, v), data['length'] / max(self.ugv_speed, MIN_SPEED_MS))
        try:
            return nx.dijkstra_path(self.G_dir, src, dst, weight=weight_fn)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _fallback(self, curr_node: str, observation: dict, **kwargs) -> int:
        """Return the node_id of the statically best neighbour (best_dir hint)."""
        G = kwargs.get('G')
        node2idx = kwargs.get('node2idx', self.node2idx)
        best_dir = int(observation.get('best_dir', 0))
        if G is not None:
            nb_sorted = sorted([node2idx[nb] for nb in G.neighbors(curr_node)])
            if nb_sorted:
                idx = best_dir if best_dir < len(nb_sorted) else 0
                return nb_sorted[idx]
        # Last resort: stay in place
        return observation['graph_node']
