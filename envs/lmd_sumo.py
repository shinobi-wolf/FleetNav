import math
import random

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
import pygame
import sumolib
import traci

from envs.fleet.UGVSumo import UGVSumo

INF = 100000.0


def get_centroid(shape):
    """Return centroid of a list of (x, y) points."""
    if not shape:
        return 0.0, 0.0
    x_coords = [p[0] for p in shape]
    y_coords = [p[1] for p in shape]
    return float(np.mean(x_coords)), float(np.mean(y_coords))


class SumoLMDEnv(gym.Env):
    metadata = {"render_modes": ["human", "print", "quiet"], "render_fps": 30}

    def __init__(self, config, render_mode=None):
        super(SumoLMDEnv, self).__init__()

        # Network and depot settings
        self.net_path = config["net"]["net_path"]
        self.warehouse_choice = config["net"].get("warehouse", "center")

        # Task settings
        self.task_density = float(config["tasks"]["task_density"])

        # Fleet settings (lmd.yaml units)
        self.num_ugvs = int(config["fleet"]["size"])
        self.max_ugv_range = float(config["fleet"]["range"]) * 1000.0  # km -> m
        self.ugv_speed = float(config["fleet"]["max_speed"]) * 1000.0 / 3600.0  # km/h -> m/s
        self.num_actions = int(config["net"]["max_degree"])
        self.idle_action = self.num_actions

        # SUMO settings
        self.step_length = float(config["sumo"]["step_length"])
        self.traffic_scale = float(config["sumo"].get("traffic_scale", 1))
        self.delay = float(config["sumo"].get("delay", 0))

        # Simulation settings
        self.max_time = float(config["simulation"]["max_time"]) * 3600.0  # hours -> seconds
        self.max_steps = int(config["simulation"]["max_steps"])
        self.render_mode = render_mode if render_mode is not None else config["simulation"].get("render_mode", "quiet")
        self.render_fps = int(config["simulation"].get("render_fps", self.metadata["render_fps"]))
        self.warmup_time = float(config["simulation"].get("warm_up_time", 0))
        self.vehicle_type = config["simulation"].get("vehicle_type", "car")
        self.sumo_vehicle_type = None
        self.metadata["render_fps"] = self.render_fps

        # Global piecewise traffic settings
        self.traffic_speed_limits = tuple(np.array(config["traffic"]["speed_limits"], dtype=np.float32) * 1000.0 / 3600.0)
        self.traffic_speed_step = float(config["traffic"]["speed_step"]) * 1000.0 / 3600.0
        self.traffic_seg_duration = float(config["traffic"]["seg_duration"]) * 60.0  # min -> s
        self.traffic_seed = int(config["traffic"]["seed"])

        min_speed, max_speed = self.traffic_speed_limits
        if self.traffic_speed_step <= 0:
            raise ValueError("traffic.speed_step must be > 0")
        if max_speed < min_speed:
            raise ValueError("traffic.speed_limits must satisfy min <= max")

        self.traffic_speed_range = np.arange(min_speed, max_speed + 1e-9, self.traffic_speed_step, dtype=np.float32)
        num_segments = max(1, int(np.ceil(self.max_time / self.traffic_seg_duration)))
        traffic_rng = np.random.default_rng(self.traffic_seed)
        self.traffic_speed_segments = traffic_rng.choice(self.traffic_speed_range, size=num_segments)

        print(
            f"Traffic segment duration: {self.traffic_seg_duration:.0f} s "
            f"({self.traffic_seg_duration / 60.0:.1f} min) "
            f"| Number of segments: {num_segments}"
        )
        print("First 10 traffic speed segments:")
        for seg_idx, speed_ms in enumerate(self.traffic_speed_segments[:10]):
            print(f"  seg {seg_idx}: {speed_ms:.2f} m/s ({self.ms_to_kmh(speed_ms):.1f} km/h)")

        self.speed_changed_this_step = False
        self.last_speed_change_time = None
        self.last_speed_change_from = None
        self.last_speed_change_to = None
        self.last_speed_change_segment_from = None
        self.last_speed_change_segment_to = None
        self.current_speed = float(self.traffic_speed_segments[0])

        # Build once from net.xml and keep stable indexing
        self.build_graph()
        self.num_nodes = len(self.node2idx)
        self.warehouse_node_idx = self._resolve_warehouse_node_idx(self.warehouse_choice)

        # Gym spaces: explicit IDLE action and valid non-negative observation bounds
        self.action_space = spaces.MultiDiscrete([self.num_actions + 1] * self.num_ugvs)

        self.observation_space = spaces.Dict(
            {
                "ugv_status": spaces.MultiDiscrete([3] * self.num_ugvs),
                "steps2dest": spaces.Box(
                    low=0.0,
                    high=INF,
                    shape=(self.num_ugvs * self.num_actions,),
                    dtype=np.float32,
                ),
                "graph_node": spaces.MultiDiscrete([self.num_nodes] * self.num_ugvs),
                "target_node": spaces.MultiDiscrete([self.num_nodes] * self.num_ugvs),
                "nb_traffic": spaces.MultiDiscrete([3] * (self.num_ugvs * self.num_actions)),
            }
        )

        # Runtime state
        self.screen = None
        self.clock = None
        self.ugv_states = []
        self.route_spawn_counter = 0
        self.current_timestep = 0
        self.terminated = False
        self.truncated = False
        self.time_elapsed = 0.0
        self.num_tasks = 0
        self.num_tasks_completed = 0
        self.task_completion_times = []
        self.total_ev_distance = 0.0
        self.task_list = []
        self.info = {}
        self.moved_this_step = set()

    def ms_to_kmh(self, speed_ms):
        if speed_ms is None:
            return None
        return float(speed_ms) * 3.6

    def format_speed(self, speed_ms):
        if speed_ms is None:
            return "N/A"
        return f"{speed_ms:.2f} m/s ({self.ms_to_kmh(speed_ms):.1f} km/h)"

    def get_current_segment_index(self):
        if len(self.traffic_speed_segments) == 0:
            return -1
        current_time = max(0.0, self.time_elapsed)
        seg_idx = int(current_time / self.traffic_seg_duration)
        return min(seg_idx, len(self.traffic_speed_segments) - 1)

    def get_current_speed(self):
        seg_idx = self.get_current_segment_index()
        if seg_idx < 0:
            return float(self.traffic_speed_segments[0])
        return float(self.traffic_speed_segments[seg_idx])

    def _node_position(self, node_obj):
        shape = node_obj.getShape()
        if shape:
            return get_centroid(shape)
        coord = node_obj.getCoord()
        return float(coord[0]), float(coord[1])

    def build_graph(self):
        net_path = self.net_path.replace(".sumocfg", ".net.xml")
        net = sumolib.net.readNet(net_path, withInternal=False)

        g_dir = nx.DiGraph()

        for edge in net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":"):
                continue

            from_node = edge.getFromNode()
            to_node = edge.getToNode()
            from_id = from_node.getID()
            to_id = to_node.getID()

            fx, fy = self._node_position(from_node)
            tx, ty = self._node_position(to_node)

            g_dir.add_node(from_id, x=fx, y=fy)
            g_dir.add_node(to_id, x=tx, y=ty)

            edge_length = float(edge.getLength())
            if edge_length <= 0.0:
                edge_length = float(math.hypot(tx - fx, ty - fy))

            g_dir.add_edge(from_id, to_id, id=edge_id, length=edge_length)

        if len(g_dir.nodes) == 0:
            raise ValueError("No non-internal edges/nodes found while building SUMO graph")

        self.G_dir = g_dir
        self.G = g_dir
        self.all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(self.G_dir, weight="length"))

        sorted_nodes = sorted(self.G.nodes)
        self.node2idx = {node: idx for idx, node in enumerate(sorted_nodes)}
        self.idx2node = {idx: node for node, idx in self.node2idx.items()}

    def _resolve_warehouse_node_idx(self, warehouse_choice):
        nodes = list(self.G.nodes)
        if not nodes:
            raise ValueError("Cannot resolve warehouse: graph has no nodes")

        if isinstance(warehouse_choice, str):
            choice = warehouse_choice.strip()
            lower_choice = choice.lower()

            if lower_choice == "center":
                xs = np.array([self.G.nodes[n]["x"] for n in nodes], dtype=np.float32)
                ys = np.array([self.G.nodes[n]["y"] for n in nodes], dtype=np.float32)
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                best_node = min(nodes, key=lambda n: (self.G.nodes[n]["x"] - cx) ** 2 + (self.G.nodes[n]["y"] - cy) ** 2)
                return self.node2idx[best_node]

            if lower_choice == "random":
                return self.node2idx[random.choice(nodes)]

            if choice in self.node2idx:
                return self.node2idx[choice]

        # Fallback to center semantics if value is unknown
        xs = np.array([self.G.nodes[n]["x"] for n in nodes], dtype=np.float32)
        ys = np.array([self.G.nodes[n]["y"] for n in nodes], dtype=np.float32)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        best_node = min(nodes, key=lambda n: (self.G.nodes[n]["x"] - cx) ** 2 + (self.G.nodes[n]["y"] - cy) ** 2)
        return self.node2idx[best_node]

    def _safe_close_traci(self):
        try:
            if traci.isLoaded():
                traci.close(False)
        except Exception:
            pass

    def _reset_speed_change_tracking(self):
        self.speed_changed_this_step = False
        self.last_speed_change_time = None
        self.last_speed_change_from = None
        self.last_speed_change_to = None
        self.last_speed_change_segment_from = None
        self.last_speed_change_segment_to = None

    def _resolve_sumo_vehicle_type(self):
        """Pick a valid SUMO vehicle type, preferring the configured type."""
        try:
            type_ids = list(traci.vehicletype.getIDList())
        except traci.TraCIException:
            return None

        if not type_ids:
            return None

        if self.vehicle_type in type_ids:
            return self.vehicle_type

        if "DEFAULT_VEHTYPE" in type_ids:
            return "DEFAULT_VEHTYPE"

        return type_ids[0]

    def _initialize_tasks(self):
        candidate_nodes = [idx for idx in range(self.num_nodes) if idx != self.warehouse_node_idx]
        if not candidate_nodes:
            candidate_nodes = [self.warehouse_node_idx]

        num_tasks = max(1, int((self.task_density * len(candidate_nodes)) // 100))
        replace = num_tasks > len(candidate_nodes)
        sampled = self.np_random.choice(np.array(candidate_nodes, dtype=np.int32), size=num_tasks, replace=replace)
        self.task_list = [int(v) for v in sampled.tolist()]

        # Shuffle task order for fair distribution
        perm = self.np_random.permutation(len(self.task_list))
        self.task_list = [self.task_list[int(i)] for i in perm]
        self.num_tasks = len(self.task_list)

        base = self.num_tasks // self.num_ugvs
        remainder = self.num_tasks % self.num_ugvs

        start = 0
        for ugv_id, ugv in enumerate(self.ugv_states):
            count = base + (1 if ugv_id < remainder else 0)
            ugv.task_list = self.task_list[start : start + count]
            start += count
            if len(ugv.task_list) == 0:
                ugv.status = 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._safe_close_traci()

        self.current_timestep = 0
        self.terminated = False
        self.truncated = False
        self.time_elapsed = 0.0
        self.route_spawn_counter = 0
        self.current_speed = self.get_current_speed()
        self._reset_speed_change_tracking()

        # Pygame state
        self.screen = None
        self.clock = None

        sumo_mode = "sumo-gui" if self.render_mode == "human" else "sumo"
        delay = self.delay if self.render_mode == "human" else 0.0
        sumo_config = [
            sumo_mode,
            "-c",
            self.net_path,
            "--step-length",
            str(self.step_length),
            "--scale",
            str(self.traffic_scale),
            "--start",
            "--delay",
            str(delay),
            "--no-warnings",
        ]
        traci.start(sumo_config)
        self.sumo_vehicle_type = self._resolve_sumo_vehicle_type()

        while self.time_elapsed < self.warmup_time:
            prev_segment = self.get_current_segment_index()
            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()
            self._update_speed_if_needed(prev_segment)

        self.ugv_states = [
            UGVSumo(
                ugv_id=i,
                base_position=self.warehouse_node_idx,
                max_range=self.max_ugv_range,
                max_speed=self.ugv_speed,
            )
            for i in range(self.num_ugvs)
        ]

        for ugv in self.ugv_states:
            ugv.position = self.warehouse_node_idx
            ugv.base_point = self.warehouse_node_idx
            ugv.current_range = ugv.max_range
            ugv.current_range_percent = 1.0
            ugv.status = 1
            ugv.veh_id = None

        self._initialize_tasks()

        self.num_tasks_completed = 0
        self.task_completion_times = []
        self.total_ev_distance = 0.0
        self.moved_this_step = set()

        obs = self._get_observation()
        self.info = self.analyzer()
        return obs, self.info

    def _get_nb_traffic(self, nb_count):
        nb_traffic = []
        v_limit = max(self.traffic_speed_limits)
        v_mean = self.current_speed

        congestion = max(0.0, min(1.0 - v_mean / max(v_limit, 1e-6), 1.0)) * 100.0
        if congestion < 33.0:
            bin_val = 0
        elif congestion < 66.0:
            bin_val = 1
        else:
            bin_val = 2

        for _ in range(nb_count):
            nb_traffic.append(bin_val)
        while len(nb_traffic) < self.num_actions:
            nb_traffic.append(2)

        return nb_traffic[: self.num_actions]

    def _safe_distance(self, src_node, dst_node):
        src_dist = self.all_shortest_path_lengths.get(src_node, None)
        if src_dist is None:
            return INF
        return float(src_dist.get(dst_node, INF))

    def _get_observation(self):
        steps2dest = []
        nb_traffic = []
        ugv_status = []
        graph_node = []
        target_node = []

        for ugv in self.ugv_states:
            ugv_node = self.idx2node[ugv.position]
            neighbors = sorted([self.node2idx[n] for n in self.G.neighbors(ugv_node)])[: self.num_actions]

            if ugv.task_list:
                task_idx = ugv.task_list[0]
                task_node = self.idx2node[task_idx]
                steps = [self._safe_distance(self.idx2node[nb_idx], task_node) for nb_idx in neighbors]
                steps += [INF] * (self.num_actions - len(steps))
                target_idx = task_idx
            else:
                steps = [INF] * self.num_actions
                target_idx = self.warehouse_node_idx

            steps2dest.append(steps)
            nb_traffic.append(self._get_nb_traffic(len(neighbors)))
            ugv_status.append(int(ugv.status))
            graph_node.append(int(ugv.position))
            target_node.append(int(target_idx))

        obs_dict = {
            "steps2dest": np.array(steps2dest, dtype=np.float32).flatten(),
            "nb_traffic": np.array(nb_traffic, dtype=np.int32).flatten(),
            "ugv_status": np.array(ugv_status, dtype=np.int32).flatten(),
            "graph_node": np.array(graph_node, dtype=np.int32).flatten(),
            "target_node": np.array(target_node, dtype=np.int32).flatten(),
        }
        return obs_dict

    def _get_active_vehicle_ids(self):
        try:
            active_ids = set(traci.vehicle.getIDList())
        except traci.TraCIException:
            return []

        return [ugv.veh_id for ugv in self.ugv_states if ugv.veh_id and ugv.veh_id in active_ids]

    def _update_speed_if_needed(self, prev_segment_idx):
        current_segment_idx = self.get_current_segment_index()
        if current_segment_idx == prev_segment_idx:
            return

        prev_speed = self.current_speed
        self.current_speed = self.get_current_speed()
        self.speed_changed_this_step = True
        self.last_speed_change_time = self.time_elapsed
        self.last_speed_change_from = prev_speed
        self.last_speed_change_to = self.current_speed
        self.last_speed_change_segment_from = prev_segment_idx
        self.last_speed_change_segment_to = current_segment_idx

        for veh_id in self._get_active_vehicle_ids():
            try:
                traci.vehicle.setSpeed(veh_id, self.current_speed)
            except traci.TraCIException:
                continue

    def _refresh_ugv_states_from_sumo(self):
        try:
            active_ids = set(traci.vehicle.getIDList())
        except traci.TraCIException:
            active_ids = set()

        for ugv in self.ugv_states:
            if ugv.veh_id and ugv.veh_id in active_ids:
                ugv.status = 0
                continue

            if ugv.veh_id and ugv.veh_id not in active_ids:
                ugv.veh_id = None

            if ugv.task_list:
                ugv.status = 1
            else:
                ugv.status = 2

            if ugv.position == self.warehouse_node_idx and ugv.status != 0:
                ugv.recharge()

    def _spawn_vehicle_for_move(self, ugv, edge_id):
        route_id = f"ugv_{ugv.agent_id}_{self.route_spawn_counter}_route"
        veh_id = f"ugv_{ugv.agent_id}_{self.route_spawn_counter}"
        self.route_spawn_counter += 1

        traci.route.add(route_id, [edge_id])
        if self.sumo_vehicle_type is not None:
            traci.vehicle.add(veh_id, route_id, typeID=self.sumo_vehicle_type, departSpeed=self.current_speed)
        else:
            traci.vehicle.add(veh_id, route_id, departSpeed=self.current_speed)

        traci.vehicle.setSpeed(veh_id, self.current_speed)
        traci.vehicle.setColor(veh_id, (255, 0, 0, 255))
        ugv.veh_id = veh_id
        ugv.status = 0

    def ugv_handler(self, action):
        actions = np.asarray(action, dtype=np.int32).flatten()
        if len(actions) != self.num_ugvs:
            raise ValueError(f"Expected {self.num_ugvs} actions, got {len(actions)}")

        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.status == 2:
                continue

            if ugv.veh_id and ugv.veh_id in self._get_active_vehicle_ids():
                ugv.status = 0
                continue

            if not ugv.task_list:
                ugv.status = 2
                continue

            ugv.status = 1
            ugv.last_position = ugv.position
            act = int(actions[ugv_id])

            # Explicit IDLE action
            if act == self.idle_action:
                if ugv.position == self.warehouse_node_idx:
                    ugv.recharge()
                continue

            move_flag, edge_id, edge_dir = ugv.move(
                act,
                self.time_elapsed,
                self.G,
                self.G_dir,
                self.node2idx,
                self.idx2node,
            )

            if not move_flag:
                continue

            self.moved_this_step.add(ugv_id)
            self._spawn_vehicle_for_move(ugv, edge_id)

    def _advance_simulation_until_idle(self):
        active_vehicle_ids = self._get_active_vehicle_ids()

        if not active_vehicle_ids:
            prev_segment = self.get_current_segment_index()
            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()
            self._update_speed_if_needed(prev_segment)
            self._refresh_ugv_states_from_sumo()
            return

        while active_vehicle_ids and self.time_elapsed < self.max_time:
            prev_segment = self.get_current_segment_index()
            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()
            self._update_speed_if_needed(prev_segment)
            active_vehicle_ids = self._get_active_vehicle_ids()

        self._refresh_ugv_states_from_sumo()

    def task_handler(self):
        truncated = False

        for ugv in self.ugv_states:
            if not ugv.task_list:
                if ugv.status != 0:
                    ugv.status = 2
                continue

            if ugv.status == 0:
                continue

            if ugv.position == ugv.task_list[0]:
                truncated = True
                ugv.task_list.pop(0)
                ugv.task_completed_this_step = True
                self.num_tasks_completed += 1
                self.task_completion_times.append(self.time_elapsed)

                if not ugv.task_list:
                    ugv.status = 2

            if ugv.position == self.warehouse_node_idx and ugv.status != 0:
                ugv.recharge()

        return truncated

    def _get_reward(self):
        total_reward = 0.0

        completion_reward = 50.0
        invalid_action_penalty = -15.0
        move_time_penalty_scale = -0.1

        for ugv_id, ugv in enumerate(self.ugv_states):
            ugv_reward = 0.0

            if ugv.task_completed_this_step:
                ugv_reward += completion_reward
                ugv.task_completed_this_step = False

            if ugv.invalid_action_this_step:
                ugv_reward += invalid_action_penalty
                ugv.invalid_action_this_step = False

            if ugv_id in self.moved_this_step and ugv.last_move_time is not None:
                move_time = max(0.0, self.time_elapsed - ugv.last_move_time)
                ugv_reward += move_time_penalty_scale * move_time

            total_reward += ugv_reward

        return total_reward

    def _check_termination_condition(self):
        done_status_flag = all((len(ugv.task_list) == 0) for ugv in self.ugv_states)
        if (
            self.time_elapsed >= self.max_time
            or self.current_timestep >= self.max_steps
            or self.num_tasks_completed >= self.num_tasks
            or done_status_flag
        ):
            self.terminated = True
        else:
            self.terminated = False

    def get_actual_vehicle_speeds(self):
        speeds = []
        try:
            active_ids = set(traci.vehicle.getIDList())
        except traci.TraCIException:
            active_ids = set()

        for ugv in self.ugv_states:
            if ugv.veh_id and ugv.veh_id in active_ids:
                try:
                    speed_ms = traci.vehicle.getSpeed(ugv.veh_id)
                except traci.TraCIException:
                    speed_ms = None
            else:
                speed_ms = None

            speeds.append(speed_ms)

        return speeds

    def analyzer(self):
        self.total_ev_distance = float(sum(ugv.distance_traveled for ugv in self.ugv_states))

        completion_diffs = [
            self.task_completion_times[i] - self.task_completion_times[i - 1]
            for i in range(1, len(self.task_completion_times))
        ]
        avg_task_completion_time = float(np.mean(completion_diffs)) if completion_diffs else 0.0

        ugv_positions_idx = [int(ugv.position) for ugv in self.ugv_states]
        ugv_positions = [self.idx2node[ugv.position] for ugv in self.ugv_states]

        active_targets_idx = [int(ugv.task_list[0]) if ugv.task_list else None for ugv in self.ugv_states]
        active_targets = [self.idx2node[target] if target is not None else None for target in active_targets_idx]

        ugv_status = [int(ugv.status) for ugv in self.ugv_states]
        ugv_ranges = [float(ugv.current_range) for ugv in self.ugv_states]

        vehicle_speed_actual_ms = self.get_actual_vehicle_speeds()
        vehicle_speed_actual_kmh = [self.ms_to_kmh(v) if v is not None else None for v in vehicle_speed_actual_ms]

        traffic_speed_cmd_ms = float(self.current_speed)

        info = {
            "time_elapsed": float(self.time_elapsed),
            "current_timestep": int(self.current_timestep),
            "current_traffic_segment_idx": int(self.get_current_segment_index()),
            "traffic_speed_cmd_ms": traffic_speed_cmd_ms,
            "traffic_speed_cmd_kmh": self.ms_to_kmh(traffic_speed_cmd_ms),
            "ugv_pos": ugv_positions,
            "ugv_pos_idx": ugv_positions_idx,
            "active_tasks": active_targets,
            "active_tasks_idx": active_targets_idx,
            "ugv_status": ugv_status,
            "ugv_remaining_range_m": ugv_ranges,
            "vehicle_speed_actual_ms": vehicle_speed_actual_ms,
            "vehicle_speed_actual_kmh": vehicle_speed_actual_kmh,
            "num_tasks_completed": int(self.num_tasks_completed),
            "total_tasks": int(self.num_tasks),
            "total_ev_distance": self.total_ev_distance,
            "avg_task_completion_time": avg_task_completion_time,
            "speed_changed_this_step": bool(self.speed_changed_this_step),
            "last_speed_change_time": self.last_speed_change_time,
            "last_speed_change_from": self.last_speed_change_from,
            "last_speed_change_to": self.last_speed_change_to,
            "last_speed_change_segment_from": self.last_speed_change_segment_from,
            "last_speed_change_segment_to": self.last_speed_change_segment_to,
            "warehouse_node": self.idx2node[self.warehouse_node_idx],
            "warehouse_node_idx": int(self.warehouse_node_idx),
        }
        return info

    def step(self, action):
        if self.terminated:
            return None, None, self.terminated, None, self.info

        self.speed_changed_this_step = False
        self.moved_this_step = set()

        for ugv in self.ugv_states:
            ugv.task_completed_this_step = False
            ugv.invalid_action_this_step = False

        self.ugv_handler(action)
        self.current_timestep += 1

        self._advance_simulation_until_idle()
        truncated = self.task_handler()

        reward = self._get_reward()
        self._check_termination_condition()

        obs = self._get_observation()
        self.info = self.analyzer()

        return obs, reward, self.terminated, truncated, self.info

    def _status_text(self, status):
        if status == 0:
            return "InTransit"
        if status == 1:
            return "Ready"
        return "Done"

    def render(self):
        if self.render_mode == "quiet":
            return

        info = self.analyzer()

        if self.render_mode in ("human", "print"):
            print("\n" + "=" * 80)
            print(
                f"Time: {info['time_elapsed']:.2f}s | Step: {info['current_timestep']}/{self.max_steps} "
                f"| Tasks: {info['num_tasks_completed']}/{info['total_tasks']}"
            )
            print(
                f"Traffic seg: {info['current_traffic_segment_idx']} | "
                f"Cmd speed: {self.format_speed(info['traffic_speed_cmd_ms'])} | "
                f"Speed changed: {'YES' if info['speed_changed_this_step'] else 'NO'}"
            )
            if info["last_speed_change_time"] is not None:
                print(
                    "Last speed change: "
                    f"t={info['last_speed_change_time']:.1f}s | "
                    f"seg {info['last_speed_change_segment_from']} -> {info['last_speed_change_segment_to']} | "
                    f"{self.format_speed(info['last_speed_change_from'])} -> "
                    f"{self.format_speed(info['last_speed_change_to'])}"
                )
            print(
                f"Warehouse: {info['warehouse_node']} (idx={info['warehouse_node_idx']}) | "
                f"Total distance: {info['total_ev_distance']:.1f}m"
            )

            for ugv_id, ugv in enumerate(self.ugv_states):
                pos_idx = info["ugv_pos_idx"][ugv_id]
                pos_node = info["ugv_pos"][ugv_id]
                target_idx = info["active_tasks_idx"][ugv_id]
                target_node = info["active_tasks"][ugv_id]
                speed_ms = info["vehicle_speed_actual_ms"][ugv_id]
                speed_str = self.format_speed(speed_ms) if speed_ms is not None else "N/A"
                print(
                    f"UGV {ugv_id}: status={self._status_text(ugv.status)} | "
                    f"pos={pos_node} (idx={pos_idx}) | "
                    f"target={target_node} (idx={target_idx}) | "
                    f"range={ugv.current_range:.1f}/{self.max_ugv_range:.1f}m | "
                    f"speed={speed_str}"
                )

            print(f"Terminated: {self.terminated}")
            print("=" * 80)

        if self.render_mode == "print":
            return

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen_width = 1100
                self.screen_height = 800
                self.clock = pygame.time.Clock()
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("SUMO LMD Environment")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            self.screen.fill((245, 245, 245))

            pos_x = [self.G.nodes[n]["x"] for n in self.G.nodes]
            pos_y = [self.G.nodes[n]["y"] for n in self.G.nodes]
            min_x, max_x = min(pos_x), max(pos_x)
            min_y, max_y = min(pos_y), max(pos_y)

            margin = 50
            panel_width = 380
            map_width = self.screen_width - panel_width - margin
            scale_x = (map_width - 2 * margin) / max(1.0, max_x - min_x)
            scale_y = (self.screen_height - 2 * margin) / max(1.0, max_y - min_y)
            scale = min(scale_x, scale_y)

            def transform(x, y):
                px = int(margin + (x - min_x) * scale)
                py = int(margin + (max_y - y) * scale)
                return px, py

            for u, v in self.G_dir.edges():
                sx, sy = transform(self.G.nodes[u]["x"], self.G.nodes[u]["y"])
                ex, ey = transform(self.G.nodes[v]["x"], self.G.nodes[v]["y"])
                pygame.draw.line(self.screen, (165, 165, 165), (sx, sy), (ex, ey), 2)

            node_radius = 11
            font_node = pygame.font.Font(None, 16)
            for node in self.G.nodes:
                x, y = transform(self.G.nodes[node]["x"], self.G.nodes[node]["y"])
                pygame.draw.circle(self.screen, (85, 85, 85), (x, y), node_radius)
                node_idx = self.node2idx[node]
                idx_text = font_node.render(str(node_idx), True, (255, 255, 255))
                text_rect = idx_text.get_rect(center=(x, y))
                self.screen.blit(idx_text, text_rect)

            warehouse_node = self.idx2node[self.warehouse_node_idx]
            wx, wy = transform(self.G.nodes[warehouse_node]["x"], self.G.nodes[warehouse_node]["y"])
            pygame.draw.circle(self.screen, (0, 80, 220), (wx, wy), 12)

            task_radius = 7
            ugv_radius = 9
            for ugv in self.ugv_states:
                if ugv.task_list:
                    task_node = self.idx2node[ugv.task_list[0]]
                    tx, ty = transform(self.G.nodes[task_node]["x"], self.G.nodes[task_node]["y"])
                    task_surface = pygame.Surface((task_radius * 2, task_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(task_surface, (220, 30, 30, 120), (task_radius, task_radius), task_radius)
                    self.screen.blit(task_surface, (tx - task_radius, ty - task_radius))

                ugv_node = self.idx2node[ugv.position]
                ux, uy = transform(self.G.nodes[ugv_node]["x"], self.G.nodes[ugv_node]["y"])
                if ugv.status == 0:
                    color = (0, 180, 0)
                elif ugv.status == 1:
                    color = (255, 180, 0)
                else:
                    color = (130, 130, 130)

                ugv_surface = pygame.Surface((ugv_radius * 2, ugv_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(ugv_surface, (*color, 170), (ugv_radius, ugv_radius), ugv_radius)
                self.screen.blit(ugv_surface, (ux - ugv_radius, uy - ugv_radius))

            panel_x = self.screen_width - panel_width + 20
            y = 18
            font_big = pygame.font.Font(None, 30)
            font_small = pygame.font.Font(None, 22)

            lines = [
                f"Time: {info['time_elapsed']:.1f}s",
                f"Step: {info['current_timestep']}/{self.max_steps}",
                f"Tasks: {info['num_tasks_completed']}/{info['total_tasks']}",
                f"Traffic seg: {info['current_traffic_segment_idx']}",
                f"Cmd speed: {self.format_speed(info['traffic_speed_cmd_ms'])}",
                f"Speed changed: {'YES' if info['speed_changed_this_step'] else 'NO'}",
                f"Distance: {info['total_ev_distance']:.1f}m",
                "",
                "UGV STATUS",
            ]

            for i, line in enumerate(lines):
                text = font_big.render(line, True, (15, 15, 15)) if i < 2 else font_small.render(line, True, (15, 15, 15))
                self.screen.blit(text, (panel_x, y))
                y += 30 if i < 2 else 24

            for ugv_id, ugv in enumerate(self.ugv_states):
                speed_ms = info["vehicle_speed_actual_ms"][ugv_id]
                speed_kmh = info["vehicle_speed_actual_kmh"][ugv_id]
                target_idx = info["active_tasks_idx"][ugv_id]
                target_str = "None" if target_idx is None else str(target_idx)
                speed_str = "N/A" if speed_ms is None else f"{speed_ms:.1f} m/s ({speed_kmh:.1f} km/h)"

                ugv_lines = [
                    f"UGV {ugv_id} {self._status_text(ugv.status)}",
                    f"  pos idx: {info['ugv_pos_idx'][ugv_id]} | target idx: {target_str}",
                    f"  range: {ugv.current_range:.1f}/{self.max_ugv_range:.1f}m",
                    f"  speed: {speed_str}",
                ]
                for line in ugv_lines:
                    text = font_small.render(line, True, (30, 30, 30))
                    self.screen.blit(text, (panel_x, y))
                    y += 22
                y += 6

            pygame.display.flip()
            self.clock.tick(self.render_fps)

    def close(self):
        if hasattr(self, "screen") and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

        self._safe_close_traci()
