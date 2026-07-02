import traci

from dronesim.trsp.agents.dsp import DSPAgent, MIN_SPEED_MS
from dronesim.trsp.ped_speed import weidmann_speed


class DSPPEDAgent(DSPAgent):
    """Dynamic Shortest Path for pedestrian (delivery-robot) mode.

    Same leg-level Dijkstra replanning as DSPAgent, but edge weights use the robot's
    Weidmann density-based *sidewalk* speed — the same speed the env's override imposes —
    instead of vehicular road mean speed. So the planned path minimises the robot's real
    travel time under the current pedestrian crowding, making this a proper traffic-aware
    baseline for the ped RL agents.

    Weidmann params default to the env's defaults; the eval/infer pipelines pass the
    resolved `traffic.ped_speed_override` values so planner and simulator stay consistent.
    """

    def __init__(self, agent_id, G_dir, node2idx, idx2node, ugv_speed,
                 gamma=1.913, density_jam=5.4, v_min=0.1):
        super().__init__(agent_id, G_dir, node2idx, idx2node, ugv_speed)
        self.gamma = gamma
        self.density_jam = density_jam
        self.v_min = v_min
        self._lane_area_cache: dict[str, float] = {}

    def _lane_area(self, lane_id: str) -> float:
        a = self._lane_area_cache.get(lane_id)
        if a is None:
            a = traci.lane.getLength(lane_id) * traci.lane.getWidth(lane_id)
            self._lane_area_cache[lane_id] = a
        return a

    def _sidewalk_speed(self, lane_id: str) -> float:
        n = len(traci.edge.getLastStepPersonIDs(traci.lane.getEdgeID(lane_id)))
        return weidmann_speed(n, self._lane_area(lane_id), self.ugv_speed,
                              self.gamma, self.density_jam, self.v_min)

    def _build_weights(self) -> dict:
        """Return {(u, v): travel_time_s} using the robot's Weidmann sidewalk speed."""
        weights = {}
        for u, v, data in self.G_dir.edges(data=True):
            lane_id = data.get('sidewalk_lane')
            speed = self._sidewalk_speed(lane_id) if lane_id else self.ugv_speed
            speed = max(speed, MIN_SPEED_MS)
            weights[(u, v)] = data['length'] / speed
        return weights
