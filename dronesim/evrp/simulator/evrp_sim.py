from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

import networkx as nx

try:
    import traci  # type: ignore[import-not-found]
    import sumolib  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(f"traci/sumolib required for simulation: {e}") from e


from ..routes.base import Route
from ..traffic.segmented import PSLTrafficModel, build_traffic_model
from .pygame_renderer import PygameRenderer

# Per-vehicle SUMO colors (match pygame renderer palette)
_SUMO_COLORS = [
    (255, 140,   0, 255),
    (  0, 180, 220, 255),
    (220,   0, 180, 255),
    (  0, 200,  80, 255),
    (255,  60,  60, 255),
    ( 60,  60, 255, 255),
    (200, 200,   0, 255),
    (  0, 200, 200, 255),
]

@dataclass
class _VehicleState:
    vehicle_id: int
    current_fuel: float        # remaining range in metres (named fuel for renderer compat)
    curr_net_node: str
    curr_net_edge: str | None
    curr_stop_idx: int
    status: str                # enroute | waiting | stranded | done
    distance_m: float = 0.0
    customers_served: int = 0
    num_recharges: int = 0
    finish_time_s: float = 0.0
    wait_until_s: float = 0.0  # sim_time at which waiting ends
    inserted: bool = False     # True once SUMO confirms the vehicle is in the simulation
    curr_traci_veh_id: str = ""     # active SUMO vehicle ID for this vehicle
    emg_refuel_active: bool = False # True while en route to depot for emergency refuel
    emg_resume_dest: str = ""       # planned next stop to visit after emergency refuel
    _emg_serial: int = 0            # monotonic counter for unique emergency leg SUMO IDs


def _resolve_net_xml_path(net_path: str) -> Optional[Path]:
    """Best-effort resolve a SUMO `.net.xml` path from a config net_path."""
    p = Path(net_path)
    if not p.exists():
        return None

    if p.suffixes[-2:] == [".net", ".xml"]:
        return p

    if p.suffix == ".sumocfg":
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            return None
        net_el = root.find("./input/net-file") or root.find(".//net-file")
        if net_el is None:
            return None
        value = net_el.get("value")
        if not value:
            return None
        net_xml = (p.parent / value).resolve()
        return net_xml if net_xml.exists() else None

    return None


# ---------------------------------------------------------------------------
# Concurrent simulator
# ---------------------------------------------------------------------------

class EVRPSimulator:
    """Time-step-driven simulator: all vehicles advance concurrently each step."""

    def __init__(self, config: dict[str, Any], local_planner: Optional[Any] = None):
        self.config = config
        self.local_planner = local_planner
        self.instance_cfg = config.get("instance", {})
        self.traffic_cfg = config.get("traffic", {})
        self.net_cfg = config.get("net", {})
        self.hps = config.get("hps", {})

        self.render_mode = self.hps["sumo"]["render_mode"]
        self.net_xml_path = _resolve_net_xml_path(self.net_cfg.get("net_path", ""))
        self.max_time = self.hps["stop_time"]
        self.ugv_max_speed = self.instance_cfg["ugv_speed"]
        self.sumocfg_path = self.net_cfg["net_path"]
        self.step_length = self.hps["sumo"]["step_length"]
        self.traffic_scale = self.hps["sumo"].get("traffic_scale", 1.0)
        self.gui_delay = self.hps["sumo"]["delay"]
        self._traffic_v_min = self.traffic_cfg["v_min"]
        self._traffic_v_max = self.traffic_cfg["v_max"]
        self.ugv_range = self.instance_cfg["ugv_range"]
        self.customer_service_time_s = self.instance_cfg.get("customer_service_time_s", 0.0)
        self.recharge_time_s = self.instance_cfg.get("recharge_time_s", 0.0)
        self.runtime_refuel = bool(self.instance_cfg.get("runtime_refuel", False))

        if self.net_xml_path is None:
            raise FileNotFoundError(
                f"Cannot locate .net.xml for SUMO config: {self.net_cfg.get('net_path')}"
            )
        self._net = sumolib.net.readNet(self.net_xml_path, withInternal=False)
        self._routing_graph = self._build_routing_graph()

    def _build_routing_graph(self) -> nx.DiGraph:
        """Build directed graph from the loaded SUMO net.

        Each non-internal SUMO edge becomes a DiGraph edge weighted by its length.
        This graph is used by _find_edge_route so simulation distances match the
        planner's shortest-path distances.
        """
        g = nx.DiGraph()
        for edge in self._net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":"):
                continue
            from_id = edge.getFromNode().getID()
            to_id = edge.getToNode().getID()
            length = float(edge.getLength())
            if length <= 0.0:
                length = 1.0
            g.add_edge(from_id, to_id, id=edge_id, length=length)
        return g

    def run(self, routes: list[Route]) -> dict[str, Any]:
        all_edge_ids = [
            e.getID() for e in self._net.getEdges() if not e.getID().startswith(":")
        ]
        traffic = build_traffic_model(
            self.config,
            free_flow_ms=self.ugv_max_speed,
            max_time_s=self.max_time,
            edge_ids=all_edge_ids,
        )

        routes_by_id: dict[int, Route] = {r.vehicle_id: r for r in routes}
        veh_states: dict[int, _VehicleState] = {}
        sim_stats: dict[int, dict] = {}
        served_customers: set[str] = set()
        vehicle_positions: dict[int, str] = {}
        vehicle_speeds: dict[int, float] = {}

        renderer: PygameRenderer | None = None
        if self.render_mode == "human":
            node_roles = self._collect_node_roles(routes)
            vehicle_ids = [r.vehicle_id for r in routes]
            renderer = PygameRenderer(
                self._net, node_roles, vehicle_ids,
                v_min_ms=self._traffic_v_min,
                v_max_ms=self._traffic_v_max,
            )

        sumo_mode = "sumo-gui" if self.render_mode == "human" else "sumo"
        sumo_cmd = [
            sumo_mode,
            "-c", self.sumocfg_path,
            "--step-length", str(self.step_length),
            "--scale", str(self.traffic_scale),
            "--no-warnings",
            "--no-step-log",
            "--start",
        ]
        if self.render_mode == "human":
            sumo_cmd += ["--delay", str(self.gui_delay)]

        traci.start(sumo_cmd)
        sim_time = 0.0
        traffic_slot = traffic.get_slot(sim_time)

        try:
            # Initialise vehicle states and spawn first leg for each route
            for route in routes:
                sim_stats[route.vehicle_id] = {
                    "distance_covered": 0.0,
                    "num_recharges": 0,
                    "cust_times": [],
                    "num_customers_served": 0,
                    "finish_time": 0.0,
                }
                if len(route.stops) < 2:
                    continue

                veh_states[route.vehicle_id] = _VehicleState(
                    vehicle_id=route.vehicle_id,
                    current_fuel=self.ugv_range,
                    curr_net_node=route.stops[0].sumo_node_id,
                    curr_net_edge=None,
                    curr_stop_idx=0,
                    status="enroute",
                )
                vehicle_positions[route.vehicle_id] = route.stops[0].sumo_node_id

                src_node = route.stops[0].sumo_node_id
                dest_node = route.stops[1].sumo_node_id
                edge_route, path_dist = self._resolve_route(src_node, dest_node, sim_time)

                if edge_route is None:
                    raise ValueError(
                        f"No SUMO route found from {src_node} to {dest_node} "
                        f"for vehicle {route.vehicle_id}"
                    )

                if veh_states[route.vehicle_id].current_fuel < path_dist:
                    veh_states[route.vehicle_id].status = "stranded"
                    if self.render_mode != "quiet":
                        print(
                            f"[{sim_time:.0f}s] V{route.vehicle_id}: STRANDED on first leg"
                            f"  (have: {veh_states[route.vehicle_id].current_fuel:.0f}m,"
                            f" need: {path_dist:.0f}m)"
                        )
                    continue

                veh_states[route.vehicle_id].current_fuel -= path_dist
                veh_states[route.vehicle_id].distance_m += path_dist
                sim_stats[route.vehicle_id]["distance_covered"] += path_dist

                traci_veh_id = f"ev_{route.vehicle_id}_leg0"
                route_id = f"route_{route.vehicle_id}_leg0"
                traci.route.add(route_id, edge_route)
                speed = traffic.get_speed(edge_route[0], traffic_slot)
                traci.vehicle.add(traci_veh_id, route_id, departSpeed=speed)
                traci.vehicle.setColor(traci_veh_id, _SUMO_COLORS[route.vehicle_id % len(_SUMO_COLORS)])
                veh_states[route.vehicle_id].curr_traci_veh_id = traci_veh_id

                if self.render_mode != "quiet":
                    print(
                        f"[{sim_time:.0f}s] V{route.vehicle_id}: "
                        f"departing {route.stops[0].node_name} → {route.stops[1].node_name}"
                        f"  (range: {veh_states[route.vehicle_id].current_fuel:.0f}m)"
                    )
                
                if renderer is not None:
                    edge_speeds = (
                        {eid: traffic.get_speed(eid, traffic_slot) for eid in all_edge_ids}
                        if traffic.enabled else None
                    )
                    renderer.draw(
                        sim_time, edge_speeds, vehicle_positions,
                        served_customers, veh_states, vehicle_speeds,
                    )

            # Main simulation loop
            while sim_time < self.max_time:
                traci.simulationStep()
                sim_time = traci.simulation.getTime()

                next_slot = traffic.get_slot(sim_time)
                slot_changed = next_slot != traffic_slot
                traffic_slot = next_slot

                active_traci_ids  = set(traci.vehicle.getIDList())
                arrived_this_step = set(traci.simulation.getArrivedIDList())

                for veh_id, veh_state in veh_states.items():
                    if veh_state.status in {"stranded", "done"}:
                        continue

                    route = routes_by_id[veh_id]

                    # Resume from waiting if service/recharge time has elapsed
                    if veh_state.status == "waiting":
                        if sim_time >= veh_state.wait_until_s:
                            depot_node = route.stops[0].sumo_node_id
                            if veh_state.emg_refuel_active and veh_state.emg_resume_dest:
                                # dwell after deciding emergency detour; now head to depot
                                self._spawn_emg_to_depot_leg(
                                    veh_id, veh_state, depot_node,
                                    sim_time, traffic, traffic_slot, sim_stats,
                                )
                            elif not veh_state.emg_refuel_active and veh_state.emg_resume_dest:
                                # dwell after emergency depot refuel; now resume planned route
                                self._spawn_emg_resume_leg(
                                    veh_id, veh_state,
                                    sim_time, traffic, traffic_slot, sim_stats,
                                )
                            elif veh_state.curr_stop_idx < len(route.stops) - 1:
                                next_stop = route.stops[veh_state.curr_stop_idx + 1]
                                if self._needs_emergency_refuel(veh_state, next_stop, depot_node):
                                    veh_state.emg_refuel_active = True
                                    veh_state.emg_resume_dest = next_stop.sumo_node_id
                                    self._spawn_emg_to_depot_leg(
                                        veh_id, veh_state, depot_node,
                                        sim_time, traffic, traffic_slot, sim_stats,
                                    )
                                else:
                                    if self.render_mode != "quiet":
                                        curr_stop = route.stops[veh_state.curr_stop_idx]
                                        print(
                                            f"[{sim_time:.0f}s] V{veh_id}: dwell complete at"
                                            f" {curr_stop.node_name}, departing → {next_stop.node_name}"
                                        )
                                    self._spawn_next_leg(
                                        veh_id, veh_state, route,
                                        sim_time, traffic, traffic_slot, sim_stats,
                                    )
                            veh_state.status = "enroute"
                        continue  # always skip transit-check when status was "waiting"

                    traci_veh_id = veh_state.curr_traci_veh_id

                    if traci_veh_id in active_traci_ids:
                        # Vehicle confirmed in SUMO — update position and apply traffic
                        veh_state.inserted = True
                        new_edge = traci.vehicle.getRoadID(traci_veh_id)
                        vehicle_positions[veh_id] = traci.vehicle.getPosition(traci_veh_id)
                        vehicle_speeds[veh_id] = traci.vehicle.getSpeed(traci_veh_id)
                        edge_changed = new_edge != veh_state.curr_net_edge
                        veh_state.curr_net_edge = new_edge
                        if edge_changed and not new_edge.startswith(":"):
                            traci.vehicle.setSpeed(traci_veh_id, traffic.get_speed(new_edge, traffic_slot))
                        if self.render_mode != "quiet" and edge_changed and not new_edge.startswith(":"):
                            try:
                                from_node = self._net.getEdge(new_edge).getFromNode().getID()
                            except Exception:
                                from_node = "?"
                            print(
                                f"[{sim_time:.0f}s] V{veh_id}: "
                                f"crossing node {from_node} → edge {new_edge}"
                                f"  (range: {veh_state.current_fuel:.0f}m)"
                            )
                    elif traci_veh_id in arrived_this_step:
                        # Emergency depot arrival — refuel and resume planned route
                        if veh_state.emg_refuel_active:
                            veh_state.emg_refuel_active = False
                            depot_node = route.stops[0].sumo_node_id
                            veh_state.curr_net_node = depot_node
                            vehicle_positions[veh_id] = depot_node
                            veh_state.current_fuel = self.ugv_range
                            veh_state.num_recharges += 1
                            sim_stats[veh_id]["num_recharges"] += 1
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: emergency refuel at depot"
                                    f"  (range restored: {self.ugv_range:.0f}m)"
                                )
                            if self.recharge_time_s > 0.0:
                                veh_state.wait_until_s = sim_time + self.recharge_time_s
                                veh_state.status = "waiting"
                                # emg_resume_dest still set; waiting block will use it
                                if self.render_mode != "quiet":
                                    print(
                                        f"[{sim_time:.0f}s] V{veh_id}: dwell at depot"
                                        f" for {self.recharge_time_s:.0f}s"
                                    )
                            else:
                                self._spawn_emg_resume_leg(
                                    veh_id, veh_state,
                                    sim_time, traffic, traffic_slot, sim_stats,
                                )
                                veh_state.status = "enroute"
                            continue

                        # Normal stop arrival — advance planned stop index
                        veh_state.curr_stop_idx += 1

                        if veh_state.curr_stop_idx >= len(route.stops):
                            veh_state.status = "done"
                            veh_state.finish_time_s = sim_time
                            sim_stats[veh_id]["finish_time"] = sim_time
                            vehicle_positions[veh_id] = route.stops[-1].sumo_node_id
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: route complete"
                                    f"  (dist: {veh_state.distance_m:.0f}m,"
                                    f" served: {veh_state.customers_served},"
                                    f" recharges: {veh_state.num_recharges})"
                                )
                            continue

                        recent_stop = route.stops[veh_state.curr_stop_idx]
                        veh_state.curr_net_node = recent_stop.sumo_node_id
                        vehicle_positions[veh_id] = recent_stop.sumo_node_id

                        if self.render_mode != "quiet":
                            print(
                                f"[{sim_time:.0f}s] V{veh_id}: arrived {recent_stop.node_name}"
                                f" ({recent_stop.node_type})  (range: {veh_state.current_fuel:.0f}m)"
                            )

                        # Process stop type
                        if recent_stop.node_type == "customer":
                            veh_state.customers_served += 1
                            sim_stats[veh_id]["num_customers_served"] += 1
                            sim_stats[veh_id]["cust_times"].append(sim_time)
                            served_customers.add(recent_stop.sumo_node_id)
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: served "
                                    f"{recent_stop.node_name} (customer #{veh_state.customers_served})"
                                    f"  (range: {veh_state.current_fuel:.0f}m)"
                                )
                        elif recent_stop.node_type in {"afs", "afs_dummy", "depot_dummy"}:
                            veh_state.current_fuel = self.ugv_range
                            veh_state.num_recharges += 1
                            sim_stats[veh_id]["num_recharges"] += 1
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: refueled at {recent_stop.node_name}"
                                    f"  (range restored: {self.ugv_range:.0f}m)"
                                )

                        # Check if this was the last stop
                        if veh_state.curr_stop_idx >= len(route.stops) - 1:
                            veh_state.status = "done"
                            veh_state.finish_time_s = sim_time
                            sim_stats[veh_id]["finish_time"] = sim_time
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: route complete"
                                    f"  (dist: {veh_state.distance_m:.0f}m,"
                                    f" served: {veh_state.customers_served},"
                                    f" recharges: {veh_state.num_recharges})"
                                )
                            continue

                        depot_node = route.stops[0].sumo_node_id
                        next_stop = route.stops[veh_state.curr_stop_idx + 1]

                        # Apply dwell time before spawning next leg
                        wait_s = 0.0
                        if recent_stop.node_type == "customer":
                            wait_s = self.customer_service_time_s
                        elif recent_stop.node_type in {"afs", "afs_dummy", "depot_dummy"}:
                            wait_s = self.recharge_time_s

                        # Decide if emergency depot detour is needed for next stop
                        emg_needed = self._needs_emergency_refuel(
                            veh_state, next_stop, depot_node
                        )
                        if emg_needed:
                            veh_state.emg_refuel_active = True
                            veh_state.emg_resume_dest = next_stop.sumo_node_id

                        if wait_s > 0.0:
                            veh_state.wait_until_s = sim_time + wait_s
                            veh_state.status = "waiting"
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: dwell at"
                                    f" {recent_stop.node_name} for {wait_s:.0f}s"
                                )
                            continue

                        if emg_needed:
                            self._spawn_emg_to_depot_leg(
                                veh_id, veh_state, depot_node,
                                sim_time, traffic, traffic_slot, sim_stats,
                            )
                        else:
                            # Spawn next leg immediately
                            self._spawn_next_leg(
                                veh_id, veh_state, route,
                                sim_time, traffic, traffic_slot, sim_stats,
                            )
                    # else: vehicle not yet inserted into SUMO — skip until confirmed

                # Early exit when every vehicle has finished or stranded
                if veh_states and all(vs.status in {"done", "stranded"} for vs in veh_states.values()):
                    break

                # Render
                if renderer is not None:
                    edge_speeds = (
                        {eid: traffic.get_speed(eid, traffic_slot) for eid in all_edge_ids}
                        if traffic.enabled else None
                    )
                    renderer.draw(
                        sim_time, edge_speeds, vehicle_positions,
                        served_customers, veh_states, vehicle_speeds,
                    )

        finally:
            traci.close()

        return self._build_metrics(sim_stats)

    def _spawn_next_leg(
        self,
        veh_id: int,
        veh_state: _VehicleState,
        route: Route,
        sim_time: float,
        traffic: PSLTrafficModel,
        traffic_slot: int,
        sim_stats: dict,
    ) -> None:
        """Plan and add the SUMO vehicle for the next leg of veh_state."""
        src_node = route.stops[veh_state.curr_stop_idx].sumo_node_id
        dest_node = route.stops[veh_state.curr_stop_idx + 1].sumo_node_id
        edge_route, path_dist = self._resolve_route(src_node, dest_node, sim_time)

        if edge_route is None or veh_state.current_fuel < path_dist:
            veh_state.status = "stranded"
            sim_stats[veh_id]["finish_time"] = sim_time
            if self.render_mode != "quiet":
                print(
                    f"[{sim_time:.0f}s] V{veh_id}: STRANDED"
                    f"  (have: {veh_state.current_fuel:.0f}m, need: {path_dist:.0f}m)"
                )
            return

        veh_state.current_fuel -= path_dist
        veh_state.distance_m += path_dist
        sim_stats[veh_id]["distance_covered"] += path_dist

        leg_idx = veh_state.curr_stop_idx
        traci_veh_id = f"ev_{veh_id}_leg{leg_idx}"
        route_id = f"route_{veh_id}_leg{leg_idx}"
        traci.route.add(route_id, edge_route)
        speed = traffic.get_speed(edge_route[0], traffic_slot)
        traci.vehicle.add(traci_veh_id, route_id, departSpeed=speed)
        traci.vehicle.setColor(traci_veh_id, _SUMO_COLORS[veh_id % len(_SUMO_COLORS)])
        veh_state.curr_traci_veh_id = traci_veh_id

        if self.render_mode != "quiet":
            current_stop = route.stops[veh_state.curr_stop_idx]
            next_stop = route.stops[veh_state.curr_stop_idx + 1]
            print(
                f"[{sim_time:.0f}s] V{veh_id}: "
                f"departing {current_stop.node_name} → {next_stop.node_name}"
                f"  (range: {veh_state.current_fuel:.0f}m, leg dist: {path_dist:.0f}m)"
            )

    def _needs_emergency_refuel(
        self,
        veh_state: _VehicleState,
        next_stop: Any,
        depot_node: str,
    ) -> bool:
        """Return True if vehicle can't reach next_stop and return to depot on current range."""
        if not self.runtime_refuel or veh_state.emg_refuel_active:
            return False
        # No need to detour if the next stop is itself a refueling point
        if next_stop.node_type in {"afs", "afs_dummy", "depot", "depot_dummy"}:
            return False
        _, d_to_next = self._find_edge_route(veh_state.curr_net_node, next_stop.sumo_node_id)
        _, d_next_to_depot = self._find_edge_route(next_stop.sumo_node_id, depot_node)
        return (d_to_next + d_next_to_depot) > veh_state.current_fuel

    def _spawn_emg_to_depot_leg(
        self,
        veh_id: int,
        veh_state: _VehicleState,
        depot_node: str,
        sim_time: float,
        traffic: PSLTrafficModel,
        traffic_slot: int,
        sim_stats: dict,
    ) -> None:
        """Spawn an emergency leg back to the depot for refueling."""
        edge_route, path_dist = self._resolve_route(
            veh_state.curr_net_node, depot_node, sim_time
        )
        if edge_route is None or veh_state.current_fuel < path_dist:
            veh_state.status = "stranded"
            sim_stats[veh_id]["finish_time"] = sim_time
            if self.render_mode != "quiet":
                print(
                    f"[{sim_time:.0f}s] V{veh_id}: STRANDED — insufficient range to reach depot"
                    f"  (have: {veh_state.current_fuel:.0f}m, need: {path_dist:.0f}m)"
                )
            return

        veh_state.current_fuel -= path_dist
        veh_state.distance_m += path_dist
        sim_stats[veh_id]["distance_covered"] += path_dist

        veh_state._emg_serial += 1
        traci_veh_id = f"ev_{veh_id}_emg{veh_state._emg_serial}d"
        route_id = f"route_{veh_id}_emg{veh_state._emg_serial}d"
        traci.route.add(route_id, edge_route)
        speed = traffic.get_speed(edge_route[0], traffic_slot)
        traci.vehicle.add(traci_veh_id, route_id, departSpeed=speed)
        traci.vehicle.setColor(traci_veh_id, _SUMO_COLORS[veh_id % len(_SUMO_COLORS)])
        veh_state.curr_traci_veh_id = traci_veh_id

        if self.render_mode != "quiet":
            print(
                f"[{sim_time:.0f}s] V{veh_id}: EMERGENCY REFUEL — diverting to depot"
                f"  (range: {veh_state.current_fuel:.0f}m, detour dist: {path_dist:.0f}m)"
            )

    def _spawn_emg_resume_leg(
        self,
        veh_id: int,
        veh_state: _VehicleState,
        sim_time: float,
        traffic: PSLTrafficModel,
        traffic_slot: int,
        sim_stats: dict,
    ) -> None:
        """After emergency refuel at depot, spawn leg to the originally planned next stop."""
        dest_node = veh_state.emg_resume_dest
        veh_state.emg_resume_dest = ""

        edge_route, path_dist = self._resolve_route(
            veh_state.curr_net_node, dest_node, sim_time
        )
        if edge_route is None or veh_state.current_fuel < path_dist:
            veh_state.status = "stranded"
            sim_stats[veh_id]["finish_time"] = sim_time
            if self.render_mode != "quiet":
                print(
                    f"[{sim_time:.0f}s] V{veh_id}: STRANDED after emergency refuel"
                    f"  (have: {veh_state.current_fuel:.0f}m, need: {path_dist:.0f}m)"
                )
            return

        veh_state.current_fuel -= path_dist
        veh_state.distance_m += path_dist
        sim_stats[veh_id]["distance_covered"] += path_dist

        veh_state._emg_serial += 1
        traci_veh_id = f"ev_{veh_id}_emg{veh_state._emg_serial}r"
        route_id = f"route_{veh_id}_emg{veh_state._emg_serial}r"
        traci.route.add(route_id, edge_route)
        speed = traffic.get_speed(edge_route[0], traffic_slot)
        traci.vehicle.add(traci_veh_id, route_id, departSpeed=speed)
        traci.vehicle.setColor(traci_veh_id, _SUMO_COLORS[veh_id % len(_SUMO_COLORS)])
        veh_state.curr_traci_veh_id = traci_veh_id

        if self.render_mode != "quiet":
            print(
                f"[{sim_time:.0f}s] V{veh_id}: resuming planned route after emergency refuel"
                f"  (range: {veh_state.current_fuel:.0f}m, leg dist: {path_dist:.0f}m)"
            )

    def _build_metrics(self, sim_stats: dict) -> dict:
        all_cust_times = sorted(t for v in sim_stats.values() for t in v["cust_times"])
        makespan_s = max((v["finish_time"] for v in sim_stats.values()), default=0.0)
        total_dist = sum(v["distance_covered"] for v in sim_stats.values())
        gaps = [
            all_cust_times[i + 1] - all_cust_times[i]
            for i in range(len(all_cust_times) - 1)
        ]
        per_vehicle = {
            str(vid): {
                "distance_m": v["distance_covered"],
                "num_recharges": v["num_recharges"],
                "customers_served": v["num_customers_served"],
                "finish_time_s": v["finish_time"],
            }
            for vid, v in sim_stats.items()
        }
        return {
            "makespan_s": makespan_s,
            "makespan_h": makespan_s / 3600.0,
            "total_distance_m": total_dist,
            "total_distance_km": total_dist / 1000.0,
            "total_recharges": sum(v["num_recharges"] for v in sim_stats.values()),
            "customers_served": sum(v["num_customers_served"] for v in sim_stats.values()),
            "customer_completion_times_s": all_cust_times,
            "avg_inter_customer_gap_s": (sum(gaps) / len(gaps) if gaps else None),
            "per_vehicle": per_vehicle,
        }

    @staticmethod
    def _collect_node_roles(routes: list[Route]) -> dict[str, str]:
        """Build {sumo_node_id: 'depot'|'customer'|'afs'} from all route stops."""
        roles: dict[str, str] = {}
        for route in routes:
            for stop in route.stops:
                base_type = stop.node_type.replace("_dummy", "")
                roles[stop.sumo_node_id] = base_type
        return roles

    def _find_edge_route(self, src_node: str, dst_node: str) -> tuple[list[str] | None, float]:
        """Return (edge_ids, path_length) for the shortest path from src to dst.

        Uses the nx routing graph so simulation distances are consistent with
        planning distances. Returns (None, 0.0) when no path exists.
        """
        try:
            node_path = nx.shortest_path(
                self._routing_graph, src_node, dst_node, weight="length"
            )
            path_length = nx.shortest_path_length(
                self._routing_graph, src_node, dst_node, weight="length"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, 0.0

        edge_ids = []
        for u, v in zip(node_path[:-1], node_path[1:]):
            edge_data = self._routing_graph.get_edge_data(u, v)
            if edge_data:
                edge_ids.append(edge_data["id"])
        return edge_ids, path_length

    def _resolve_route(
        self, src_node: str, dst_node: str, sim_time_s: float
    ) -> tuple[list[str] | None, float]:
        """Dispatch leg routing to the local planner (if set) or Dijkstra.

        Distance returned is always the shortest-path distance so fuel accounting
        is consistent with the EVRP planner's distance matrix, regardless of which
        physical path the TRSP agent drives.

        The _RouteLogger in infer_ta_evrp.py hooks this method to log every leg.
        """
        if self.local_planner is not None:
            edge_ids = self.local_planner.plan(src_node, dst_node, sim_time_s)
            if edge_ids:
                sp_edges, sp_dist = self._find_edge_route(src_node, dst_node)
                if sp_edges is not None:
                    return edge_ids, sp_dist
                # No SP found; fall back to actual route length
                actual_dist = sum(
                    self._net.getEdge(eid).getLength()
                    for eid in edge_ids
                    if not eid.startswith(":")
                )
                return edge_ids, actual_dist
        return self._find_edge_route(src_node, dst_node)
