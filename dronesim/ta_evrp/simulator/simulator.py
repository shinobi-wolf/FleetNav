"""TA-EVRP concurrent simulator.

Time-step-driven: all vehicles advance concurrently each SUMO step.
Consumes an Instance object; does not import from dronesim.evrp or dronesim.trsp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import networkx as nx

try:
    import traci    # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(f"traci required for simulation: {e}") from e

from .renderer import PygameRenderer

if TYPE_CHECKING:
    from dronesim.ta_evrp.instance.instance import Instance
    from dronesim.ta_evrp.instance.traffic import PSLTrafficModel

# Per-vehicle SUMO colors (rgba) — matches renderer palette
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
    current_fuel: float         # remaining range in metres
    curr_net_node: str
    curr_net_edge: Optional[str]
    curr_stop_idx: int
    status: str                 # enroute | waiting | stranded | done
    distance_m: float = 0.0
    customers_served: int = 0
    num_recharges: int = 0
    finish_time_s: float = 0.0
    wait_until_s: float = 0.0
    inserted: bool = False
    spawn_time_s: float = -1.0  # sim time when curr SUMO vehicle was added; -1 = not yet spawned
    curr_traci_veh_id: str = ""
    emg_refuel_active: bool = False
    emg_resume_dest: str = ""
    _emg_serial: int = field(default=0, repr=False)


class Simulator:
    """Concurrent TA-EVRP simulator.

    Parameters
    ----------
    instance : Instance
        Fully built TA-EVRP problem instance.
    local_planner : LocalPlanner | None
        TRSP RL agent for inter-stop leg routing.
        When None, legs are routed via Dijkstra on the road graph.
    """

    def __init__(self, instance: "Instance", local_planner: Optional[Any] = None) -> None:
        self.local_planner = local_planner
        self.sim = self  # lets route_logger.attach(simulator.sim) hook _resolve_route

        sim_cfg  = instance.sim_cfg
        sumo_cfg = sim_cfg["hps"]["sumo"]

        self.render_mode            = str(sumo_cfg.get("render_mode", "quiet"))
        self.max_time               = float(sim_cfg["hps"]["stop_time"])
        self.step_length            = float(sumo_cfg.get("step_length", 1.0))
        self.warmup_time            = float(sumo_cfg.get("warm_up_time", 0.0))
        self.gui_delay              = int(sumo_cfg.get("delay", 0))
        self.traffic_scale          = float(sim_cfg["traffic"].get("traffic_scale", 1.0))
        self.sumocfg_path           = str(sim_cfg["net"]["net_path"])
        self.ugv_range              = float(sim_cfg["instance"]["ugv_range"])
        self.ugv_max_speed          = float(sim_cfg["instance"]["ugv_speed"])
        self.customer_service_time_s = float(sim_cfg["instance"]["customer_service_time_s"])
        self.recharge_time_s        = float(sim_cfg["instance"]["recharge_time_s"])
        self.runtime_refuel         = bool(sim_cfg["instance"]["runtime_refuel"])
        trsp_cfg = sim_cfg.get("model", {}).get("trsp", {})
        self.raps_enabled: bool = bool(trsp_cfg.get("raps_enabled", False))
        self._raps_log: list[dict] = []
        self._v_min_ms              = float(sim_cfg["traffic"]["v_min"])
        self._v_max_ms              = float(sim_cfg["traffic"]["v_max"])
        self.traffic_model          = str(sim_cfg["traffic"].get("model", "psl"))
        self.svc_route_file         = sim_cfg["traffic"].get("route_file")
        # psl and td are both per-edge speed-limit models applied via SUMO
        # setMaxSpeed/setSpeed; svc instead spawns background vehicles.
        self._psl_like = self.traffic_model in ("psl", "td")

        self._net            = instance.network.net    # sumolib Net
        self._routing_graph  = instance.network.graph  # nx.DiGraph (edge attrs: id, length)
        self._traffic: "PSLTrafficModel" = instance.traffic  # already built

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, routes: list) -> dict[str, Any]:
        traffic       = self._traffic
        all_edge_ids  = [d["id"] for _, _, d in self._routing_graph.edges(data=True)]
        traffic_log: list[dict] = []
        traffic_profile: dict = dict(sorted(traffic.profiles.items())) if traffic.enabled else {}

        routes_by_id: dict[int, Any]        = {r.vehicle_id: r for r in routes}
        veh_states:   dict[int, _VehicleState] = {}
        sim_stats:    dict[int, dict]        = {}
        served_customers: set[str]           = set()
        vehicle_positions: dict[int, Any]    = {}
        vehicle_speeds:    dict[int, float]  = {}

        renderer: PygameRenderer | None = None
        if self.render_mode == "human":
            renderer = PygameRenderer(
                self._net,
                self._collect_node_roles(routes),
                self._collect_node_labels(routes),
                [r.vehicle_id for r in routes],
                v_min_ms=self._v_min_ms,
                v_max_ms=self._v_max_ms,
            )

        sumo_mode = "sumo-gui" if self.render_mode == "human" else "sumo"
        sumo_cmd  = [
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
        if self.traffic_model == "svc" and self.svc_route_file:
            sumo_cmd += ["--route-files", self.svc_route_file]

        traci.start(sumo_cmd)
        sim_time     = 0.0

        # Warm up background traffic before spawning delivery vehicles
        while sim_time < self.warmup_time:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

        traffic_slot = traffic.get_slot(sim_time)

        try:
            # Apply initial edge speeds for slot 0 (PSL only)
            if self._psl_like and traffic.enabled:
                for eid in all_edge_ids:
                    traci.edge.setMaxSpeed(eid, traffic.get_speed(eid, traffic_slot))

            # Initialise vehicle states and spawn first leg for each route
            for route in routes:
                sim_stats[route.vehicle_id] = {
                    "distance_covered":    0.0,
                    "num_recharges":       0,
                    "cust_times":          [],
                    "num_customers_served": 0,
                    "finish_time":         0.0,
                }
                if len(route.stops) < 2:
                    continue

                veh_states[route.vehicle_id] = _VehicleState(
                    vehicle_id    = route.vehicle_id,
                    current_fuel  = self.ugv_range,
                    curr_net_node = route.stops[0].sumo_node_id,
                    curr_net_edge = None,
                    curr_stop_idx = 0,
                    status        = "enroute",
                )
                vehicle_positions[route.vehicle_id] = route.stops[0].sumo_node_id

                src_node  = route.stops[0].sumo_node_id
                dest_node = route.stops[1].sumo_node_id
                raps_kwargs: dict = {}
                if self.raps_enabled and self.local_planner is not None:
                    nsi = 1
                    ridx = self._find_next_refuel_idx(route.stops, nsi)
                    min_after = (
                        route.stops[nsi].planned_fuel - route.stops[ridx].planned_fuel
                        if ridx is not None and ridx > nsi else 0.0
                    )
                    raps_kwargs = dict(
                        raps_fuel=self.ugv_range,
                        raps_min_after=min_after,
                        raps_veh_id=route.vehicle_id,
                        raps_leg_idx=0,
                    )
                edge_route, path_dist = self._resolve_route(src_node, dest_node, sim_time, **raps_kwargs)

                if edge_route is None:
                    raise ValueError(
                        f"No SUMO route from {src_node} to {dest_node} "
                        f"for vehicle {route.vehicle_id}"
                    )

                vs = veh_states[route.vehicle_id]
                if vs.current_fuel < path_dist:
                    vs.status = "stranded"
                    if self.render_mode != "quiet":
                        print(
                            f"[{sim_time:.0f}s] V{route.vehicle_id}: STRANDED on first leg"
                            f"  (have: {vs.current_fuel:.0f}m, need: {path_dist:.0f}m)"
                        )
                    continue

                vs.current_fuel -= path_dist
                vs.distance_m   += path_dist
                sim_stats[route.vehicle_id]["distance_covered"] += path_dist

                traci_veh_id = f"ev_{route.vehicle_id}_leg0"
                route_id     = f"route_{route.vehicle_id}_leg0"
                traci.route.add(route_id, edge_route)
                depart_speed = traffic.get_speed(edge_route[0], traffic_slot) if self._psl_like else 0
                traci.vehicle.add(traci_veh_id, route_id, departSpeed=depart_speed)
                traci.vehicle.setColor(traci_veh_id, _SUMO_COLORS[route.vehicle_id % len(_SUMO_COLORS)])
                vs.curr_traci_veh_id = traci_veh_id
                vs.spawn_time_s = sim_time

                if self.render_mode != "quiet":
                    print(
                        f"[{sim_time:.0f}s] V{route.vehicle_id}: "
                        f"departing {route.stops[0].node_name} → {route.stops[1].node_name}"
                        f"  (range: {vs.current_fuel:.0f}m)"
                    )

            if renderer is not None:
                edge_speeds = {eid: traffic.get_speed(eid, traffic_slot) for eid in all_edge_ids} if traffic.enabled else None
                renderer.draw(sim_time, edge_speeds, vehicle_positions, served_customers, veh_states, vehicle_speeds, traffic_slot=traffic_slot)

            # Main simulation loop
            while sim_time < self.max_time:
                traci.simulationStep()
                sim_time = traci.simulation.getTime()

                next_slot    = traffic.get_slot(sim_time)
                slot_changed = next_slot != traffic_slot
                if slot_changed:
                    if self.render_mode != "quiet":
                        print(f"[{sim_time:.0f}s] SLOT CHANGE: {traffic_slot} → {next_slot}")
                    traffic_log.append({
                        "t": sim_time, "type": "slot_change",
                        "old_slot": traffic_slot, "new_slot": next_slot,
                    })
                    if self._psl_like and traffic.enabled:
                        for eid in all_edge_ids:
                            traci.edge.setMaxSpeed(eid, traffic.get_speed(eid, next_slot))
                traffic_slot = next_slot

                active_traci_ids  = set(traci.vehicle.getIDList())
                arrived_this_step = set(traci.simulation.getArrivedIDList())

                # Detect SUMO vehicles that were spawned but never appeared or arrived.
                # SUMO silently drops vehicles whose departure edge is invalid for the
                # vehicle type (e.g. wrong lane type), leaving the vehicle invisible forever.
                force_arrived: set[str] = {
                    vs.curr_traci_veh_id
                    for vs in veh_states.values()
                    if (
                        vs.status == "enroute"
                        and vs.spawn_time_s >= 0.0
                        and (sim_time - vs.spawn_time_s) > self._STUCK_TIMEOUT_S
                        and vs.curr_traci_veh_id not in active_traci_ids
                        and vs.curr_traci_veh_id not in arrived_this_step
                    )
                }
                if force_arrived and self.render_mode != "quiet":
                    for vs in veh_states.values():
                        if vs.curr_traci_veh_id in force_arrived:
                            print(
                                f"[{sim_time:.0f}s] V{vs.vehicle_id}: WARNING — SUMO vehicle"
                                f" {vs.curr_traci_veh_id} invisible for"
                                f" {sim_time - vs.spawn_time_s:.0f}s, force-arriving at destination"
                            )

                for veh_id, vs in veh_states.items():
                    if vs.status in {"stranded", "done"}:
                        continue

                    route = routes_by_id[veh_id]

                    # Resume from dwell when service/recharge time has elapsed
                    if vs.status == "waiting":
                        if sim_time >= vs.wait_until_s:
                            depot_node = route.stops[0].sumo_node_id
                            if vs.emg_refuel_active and vs.emg_resume_dest:
                                self._spawn_emg_to_depot_leg(
                                    veh_id, vs, depot_node,
                                    sim_time, traffic, traffic_slot, sim_stats,
                                )
                            elif not vs.emg_refuel_active and vs.emg_resume_dest:
                                self._spawn_emg_resume_leg(
                                    veh_id, vs,
                                    sim_time, traffic, traffic_slot, sim_stats,
                                )
                            elif vs.curr_stop_idx < len(route.stops) - 1:
                                next_stop = route.stops[vs.curr_stop_idx + 1]
                                if self._needs_emergency_refuel(vs, next_stop, depot_node):
                                    vs.emg_refuel_active = True
                                    vs.emg_resume_dest   = next_stop.sumo_node_id
                                    self._spawn_emg_to_depot_leg(
                                        veh_id, vs, depot_node,
                                        sim_time, traffic, traffic_slot, sim_stats,
                                    )
                                else:
                                    if self.render_mode != "quiet":
                                        curr_stop = route.stops[vs.curr_stop_idx]
                                        print(
                                            f"[{sim_time:.0f}s] V{veh_id}: dwell complete at"
                                            f" {curr_stop.node_name}, departing → {next_stop.node_name}"
                                        )
                                    self._spawn_next_leg(
                                        veh_id, vs, route,
                                        sim_time, traffic, traffic_slot, sim_stats,
                                    )
                            vs.status = "enroute"
                        continue

                    traci_veh_id = vs.curr_traci_veh_id

                    if traci_veh_id in active_traci_ids:
                        vs.inserted = True
                        new_edge = traci.vehicle.getRoadID(traci_veh_id)
                        vehicle_positions[veh_id] = traci.vehicle.getPosition(traci_veh_id)
                        vehicle_speeds[veh_id]    = traci.vehicle.getSpeed(traci_veh_id)
                        edge_changed = new_edge != vs.curr_net_edge
                        vs.curr_net_edge = new_edge
                        if self._psl_like and (edge_changed or slot_changed) and not new_edge.startswith(":"):
                            new_speed = traffic.get_speed(new_edge, traffic_slot)
                            traci.vehicle.setSpeed(traci_veh_id, new_speed)
                            trigger = (
                                "slot" if (slot_changed and not edge_changed) else
                                "edge" if (edge_changed and not slot_changed) else "both"
                            )
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: speed [{trigger}]"
                                    f" edge={new_edge}  {new_speed:.2f} m/s"
                                )
                            traffic_log.append({
                                "t": sim_time, "type": "speed_update",
                                "vehicle": veh_id, "edge": new_edge,
                                "slot": traffic_slot,
                                "speed_ms": round(new_speed, 4),
                                "trigger": trigger,
                            })
                        if self.render_mode != "quiet" and edge_changed and not new_edge.startswith(":"):
                            try:
                                from_node = self._net.getEdge(new_edge).getFromNode().getID()
                            except Exception:
                                from_node = "?"
                            print(
                                f"[{sim_time:.0f}s] V{veh_id}: "
                                f"crossing node {from_node} → edge {new_edge}"
                                f"  (range: {vs.current_fuel:.0f}m)"
                            )

                    elif traci_veh_id in arrived_this_step or traci_veh_id in force_arrived:
                        # Emergency depot arrival — refuel and resume
                        if vs.emg_refuel_active:
                            vs.emg_refuel_active = False
                            depot_node            = route.stops[0].sumo_node_id
                            vs.curr_net_node      = depot_node
                            vehicle_positions[veh_id] = depot_node
                            vs.current_fuel        = self.ugv_range
                            vs.num_recharges      += 1
                            sim_stats[veh_id]["num_recharges"] += 1
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: emergency refuel at depot"
                                    f"  (range restored: {self.ugv_range:.0f}m)"
                                )
                            if self.recharge_time_s > 0.0:
                                vs.wait_until_s = sim_time + self.recharge_time_s
                                vs.status = "waiting"
                                if self.render_mode != "quiet":
                                    print(f"[{sim_time:.0f}s] V{veh_id}: dwell at depot for {self.recharge_time_s:.0f}s")
                            else:
                                self._spawn_emg_resume_leg(
                                    veh_id, vs,
                                    sim_time, traffic, traffic_slot, sim_stats,
                                )
                                vs.status = "enroute"
                            continue

                        # Normal stop arrival
                        vs.curr_stop_idx += 1

                        if vs.curr_stop_idx >= len(route.stops):
                            vs.status        = "done"
                            vs.finish_time_s = sim_time
                            sim_stats[veh_id]["finish_time"] = sim_time
                            vehicle_positions[veh_id] = route.stops[-1].sumo_node_id
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: route complete"
                                    f"  (dist: {vs.distance_m:.0f}m, served: {vs.customers_served},"
                                    f" recharges: {vs.num_recharges})"
                                )
                            continue

                        recent_stop = route.stops[vs.curr_stop_idx]
                        vs.curr_net_node = recent_stop.sumo_node_id
                        vehicle_positions[veh_id] = recent_stop.sumo_node_id

                        if self.render_mode != "quiet":
                            print(
                                f"[{sim_time:.0f}s] V{veh_id}: arrived {recent_stop.node_name}"
                                f" ({recent_stop.node_type})  (range: {vs.current_fuel:.0f}m)"
                            )

                        if recent_stop.node_type == "customer":
                            vs.customers_served += 1
                            sim_stats[veh_id]["num_customers_served"] += 1
                            sim_stats[veh_id]["cust_times"].append(sim_time)
                            served_customers.add(recent_stop.sumo_node_id)
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: served"
                                    f" {recent_stop.node_name} (customer #{vs.customers_served})"
                                    f"  (range: {vs.current_fuel:.0f}m)"
                                )
                        elif recent_stop.node_type in {"afs", "afs_dummy", "depot_dummy"}:
                            vs.current_fuel = self.ugv_range
                            vs.num_recharges += 1
                            sim_stats[veh_id]["num_recharges"] += 1
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: refueled at {recent_stop.node_name}"
                                    f"  (range restored: {self.ugv_range:.0f}m)"
                                )

                        if vs.curr_stop_idx >= len(route.stops) - 1:
                            vs.status        = "done"
                            vs.finish_time_s = sim_time
                            sim_stats[veh_id]["finish_time"] = sim_time
                            if self.render_mode != "quiet":
                                print(
                                    f"[{sim_time:.0f}s] V{veh_id}: route complete"
                                    f"  (dist: {vs.distance_m:.0f}m, served: {vs.customers_served},"
                                    f" recharges: {vs.num_recharges})"
                                )
                            continue

                        depot_node = route.stops[0].sumo_node_id
                        next_stop  = route.stops[vs.curr_stop_idx + 1]

                        wait_s = 0.0
                        if recent_stop.node_type == "customer":
                            wait_s = self.customer_service_time_s
                        elif recent_stop.node_type in {"afs", "afs_dummy", "depot_dummy"}:
                            wait_s = self.recharge_time_s

                        emg_needed = self._needs_emergency_refuel(vs, next_stop, depot_node)
                        if emg_needed:
                            vs.emg_refuel_active = True
                            vs.emg_resume_dest   = next_stop.sumo_node_id

                        if wait_s > 0.0:
                            vs.wait_until_s = sim_time + wait_s
                            vs.status = "waiting"
                            if self.render_mode != "quiet":
                                print(f"[{sim_time:.0f}s] V{veh_id}: dwell at {recent_stop.node_name} for {wait_s:.0f}s")
                            continue

                        if emg_needed:
                            self._spawn_emg_to_depot_leg(
                                veh_id, vs, depot_node,
                                sim_time, traffic, traffic_slot, sim_stats,
                            )
                        else:
                            self._spawn_next_leg(
                                veh_id, vs, route,
                                sim_time, traffic, traffic_slot, sim_stats,
                            )

                if veh_states and all(vs.status in {"done", "stranded"} for vs in veh_states.values()):
                    break

                if renderer is not None:
                    edge_speeds = {eid: traffic.get_speed(eid, traffic_slot) for eid in all_edge_ids} if traffic.enabled else None
                    renderer.draw(sim_time, edge_speeds, vehicle_positions, served_customers, veh_states, vehicle_speeds, traffic_slot=traffic_slot)

        finally:
            traci.close()
            if renderer is not None:
                renderer.close()

        return self._build_metrics(sim_stats, traffic_log, traffic_profile)

    # ------------------------------------------------------------------
    # Leg spawning
    # ------------------------------------------------------------------

    def _spawn_next_leg(
        self,
        veh_id: int,
        vs: _VehicleState,
        route: Any,
        sim_time: float,
        traffic: "PSLTrafficModel",
        traffic_slot: int,
        sim_stats: dict,
    ) -> None:
        src_node  = route.stops[vs.curr_stop_idx].sumo_node_id
        dest_node = route.stops[vs.curr_stop_idx + 1].sumo_node_id
        raps_kwargs: dict = {}
        if self.raps_enabled and self.local_planner is not None:
            nsi = vs.curr_stop_idx + 1
            ridx = self._find_next_refuel_idx(route.stops, nsi)
            min_after = (
                route.stops[nsi].planned_fuel - route.stops[ridx].planned_fuel
                if ridx is not None and ridx > nsi else 0.0
            )
            raps_kwargs = dict(
                raps_fuel=vs.current_fuel,
                raps_min_after=min_after,
                raps_veh_id=veh_id,
                raps_leg_idx=vs.curr_stop_idx,
            )
        edge_route, path_dist = self._resolve_route(src_node, dest_node, sim_time, **raps_kwargs)

        if edge_route is None or vs.current_fuel < path_dist:
            vs.status = "stranded"
            sim_stats[veh_id]["finish_time"] = sim_time
            if self.render_mode != "quiet":
                print(
                    f"[{sim_time:.0f}s] V{veh_id}: STRANDED"
                    f"  (have: {vs.current_fuel:.0f}m, need: {path_dist:.0f}m)"
                )
            return

        vs.current_fuel -= path_dist
        vs.distance_m   += path_dist
        sim_stats[veh_id]["distance_covered"] += path_dist

        leg_idx      = vs.curr_stop_idx
        traci_veh_id = f"ev_{veh_id}_leg{leg_idx}"
        route_id     = f"route_{veh_id}_leg{leg_idx}"
        traci.route.add(route_id, edge_route)
        depart_speed = traffic.get_speed(edge_route[0], traffic_slot) if self._psl_like else 0
        traci.vehicle.add(traci_veh_id, route_id, departSpeed=depart_speed)
        traci.vehicle.setColor(traci_veh_id, _SUMO_COLORS[veh_id % len(_SUMO_COLORS)])
        vs.curr_traci_veh_id = traci_veh_id
        vs.spawn_time_s = sim_time

        if self.render_mode != "quiet":
            curr = route.stops[vs.curr_stop_idx]
            nxt  = route.stops[vs.curr_stop_idx + 1]
            print(
                f"[{sim_time:.0f}s] V{veh_id}: "
                f"departing {curr.node_name} → {nxt.node_name}"
                f"  (range: {vs.current_fuel:.0f}m, leg dist: {path_dist:.0f}m)"
            )

    def _spawn_emg_to_depot_leg(
        self,
        veh_id: int,
        vs: _VehicleState,
        depot_node: str,
        sim_time: float,
        traffic: "PSLTrafficModel",
        traffic_slot: int,
        sim_stats: dict,
    ) -> None:
        edge_route, path_dist = self._resolve_route(vs.curr_net_node, depot_node, sim_time)
        if edge_route is None or vs.current_fuel < path_dist:
            vs.status = "stranded"
            sim_stats[veh_id]["finish_time"] = sim_time
            if self.render_mode != "quiet":
                print(
                    f"[{sim_time:.0f}s] V{veh_id}: STRANDED — cannot reach depot"
                    f"  (have: {vs.current_fuel:.0f}m, need: {path_dist:.0f}m)"
                )
            return

        vs.current_fuel -= path_dist
        vs.distance_m   += path_dist
        sim_stats[veh_id]["distance_covered"] += path_dist

        vs._emg_serial  += 1
        traci_veh_id = f"ev_{veh_id}_emg{vs._emg_serial}d"
        route_id     = f"route_{veh_id}_emg{vs._emg_serial}d"
        traci.route.add(route_id, edge_route)
        depart_speed = traffic.get_speed(edge_route[0], traffic_slot) if self._psl_like else 0
        traci.vehicle.add(traci_veh_id, route_id, departSpeed=depart_speed)
        traci.vehicle.setColor(traci_veh_id, _SUMO_COLORS[veh_id % len(_SUMO_COLORS)])
        vs.curr_traci_veh_id = traci_veh_id
        vs.spawn_time_s = sim_time

        if self.render_mode != "quiet":
            print(
                f"[{sim_time:.0f}s] V{veh_id}: EMERGENCY REFUEL — diverting to depot"
                f"  (range: {vs.current_fuel:.0f}m, detour dist: {path_dist:.0f}m)"
            )

    def _spawn_emg_resume_leg(
        self,
        veh_id: int,
        vs: _VehicleState,
        sim_time: float,
        traffic: "PSLTrafficModel",
        traffic_slot: int,
        sim_stats: dict,
    ) -> None:
        dest_node        = vs.emg_resume_dest
        vs.emg_resume_dest = ""

        edge_route, path_dist = self._resolve_route(vs.curr_net_node, dest_node, sim_time)
        if edge_route is None or vs.current_fuel < path_dist:
            vs.status = "stranded"
            sim_stats[veh_id]["finish_time"] = sim_time
            if self.render_mode != "quiet":
                print(
                    f"[{sim_time:.0f}s] V{veh_id}: STRANDED after emergency refuel"
                    f"  (have: {vs.current_fuel:.0f}m, need: {path_dist:.0f}m)"
                )
            return

        vs.current_fuel -= path_dist
        vs.distance_m   += path_dist
        sim_stats[veh_id]["distance_covered"] += path_dist

        vs._emg_serial  += 1
        traci_veh_id = f"ev_{veh_id}_emg{vs._emg_serial}r"
        route_id     = f"route_{veh_id}_emg{vs._emg_serial}r"
        traci.route.add(route_id, edge_route)
        depart_speed = traffic.get_speed(edge_route[0], traffic_slot) if self._psl_like else 0
        traci.vehicle.add(traci_veh_id, route_id, departSpeed=depart_speed)
        traci.vehicle.setColor(traci_veh_id, _SUMO_COLORS[veh_id % len(_SUMO_COLORS)])
        vs.curr_traci_veh_id = traci_veh_id
        vs.spawn_time_s = sim_time

        if self.render_mode != "quiet":
            print(
                f"[{sim_time:.0f}s] V{veh_id}: resuming route after emergency refuel"
                f"  (range: {vs.current_fuel:.0f}m, leg dist: {path_dist:.0f}m)"
            )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_route(
        self,
        src_node: str,
        dst_node: str,
        sim_time_s: float,
        *,
        raps_fuel: float | None = None,
        raps_min_after: float | None = None,
        raps_veh_id: int | None = None,
        raps_leg_idx: int | None = None,
    ) -> tuple[list[str] | None, float]:
        if self.local_planner is None:
            return self._find_edge_route(src_node, dst_node)

        trsp_edges = self.local_planner.plan(src_node, dst_node, sim_time_s)
        sp_edges, sp_dist = self._find_edge_route(src_node, dst_node)

        if not trsp_edges:
            return sp_edges, sp_dist

        if self.raps_enabled and raps_fuel is not None and raps_min_after is not None:
            trsp_dist = sum(
                self._net.getEdge(eid).getLength()
                for eid in trsp_edges if not eid.startswith(":")
            )
            extra_dist = trsp_dist - sp_dist
            surplus    = raps_fuel - sp_dist - raps_min_after
            use_trsp   = extra_dist <= surplus
            self._raps_log.append({
                "vehicle_id":      raps_veh_id,
                "leg_idx":         raps_leg_idx,
                "from_node":       src_node,
                "to_node":         dst_node,
                "dijkstra_dist_m": round(sp_dist, 3),
                "trsp_dist_m":     round(trsp_dist, 3),
                "extra_dist_m":    round(extra_dist, 3),
                "surplus_m":       round(surplus, 3),
                "used_trsp":       use_trsp,
            })
            return (trsp_edges, trsp_dist) if use_trsp else (sp_edges, sp_dist)

        # Legacy: TRSP path, SP distance charged
        if sp_edges is not None:
            return trsp_edges, sp_dist
        actual_dist = sum(
            self._net.getEdge(eid).getLength()
            for eid in trsp_edges if not eid.startswith(":")
        )
        return trsp_edges, actual_dist

    def _find_edge_route(
        self, src_node: str, dst_node: str
    ) -> tuple[list[str] | None, float]:
        """Dijkstra on the routing graph. Returns (edge_ids, path_length) or (None, 0.0)."""
        try:
            node_path   = nx.shortest_path(self._routing_graph, src_node, dst_node, weight="length")
            path_length = nx.shortest_path_length(self._routing_graph, src_node, dst_node, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, 0.0

        edge_ids = []
        for u, v in zip(node_path[:-1], node_path[1:]):
            edata = self._routing_graph.get_edge_data(u, v)
            if edata:
                edge_ids.append(edata["id"])
        return edge_ids, path_length

    def _needs_emergency_refuel(
        self,
        vs: _VehicleState,
        next_stop: Any,
        depot_node: str,
    ) -> bool:
        if not self.runtime_refuel or vs.emg_refuel_active:
            return False
        if next_stop.node_type in {"afs", "afs_dummy", "depot", "depot_dummy"}:
            return False
        _, d_to_next    = self._find_edge_route(vs.curr_net_node, next_stop.sumo_node_id)
        _, d_to_depot   = self._find_edge_route(next_stop.sumo_node_id, depot_node)
        return (d_to_next + d_to_depot) > vs.current_fuel

    _REFUEL_TYPES: frozenset[str] = frozenset({"afs", "afs_dummy", "depot", "depot_dummy"})
    _STUCK_TIMEOUT_S: float = 30.0  # seconds before an invisible spawned vehicle is force-arrived

    @staticmethod
    def _find_next_refuel_idx(stops: list, from_idx: int) -> int | None:
        for i in range(from_idx, len(stops)):
            if stops[i].node_type in Simulator._REFUEL_TYPES:
                return i
        return None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _build_metrics(
        self,
        sim_stats: dict,
        traffic_log: list,
        traffic_profile: dict,
    ) -> dict[str, Any]:
        all_cust_times = sorted(t for v in sim_stats.values() for t in v["cust_times"])
        makespan_s     = max((v["finish_time"] for v in sim_stats.values()), default=0.0)
        total_dist     = sum(v["distance_covered"] for v in sim_stats.values())
        gaps = [
            all_cust_times[i + 1] - all_cust_times[i]
            for i in range(len(all_cust_times) - 1)
        ]
        per_vehicle = {
            str(vid): {
                "distance_m":       v["distance_covered"],
                "num_recharges":    v["num_recharges"],
                "customers_served": v["num_customers_served"],
                "finish_time_s":    v["finish_time"],
            }
            for vid, v in sim_stats.items()
        }
        return {
            "makespan_s":                   makespan_s,
            "makespan_h":                   makespan_s / 3600.0,
            "total_distance_m":             total_dist,
            "total_distance_km":            total_dist / 1000.0,
            "total_recharges":              sum(v["num_recharges"] for v in sim_stats.values()),
            "customers_served":             sum(v["num_customers_served"] for v in sim_stats.values()),
            "customer_completion_times_s":  all_cust_times,
            "avg_inter_customer_gap_s":     (sum(gaps) / len(gaps) if gaps else None),
            "per_vehicle":                  per_vehicle,
            "traffic_log":                  traffic_log,
            "traffic_profile":              traffic_profile,
            "raps_log":                     list(self._raps_log),
        }

    # ------------------------------------------------------------------
    # Renderer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_node_roles(routes: list) -> dict[str, str]:
        roles: dict[str, str] = {}
        for route in routes:
            for stop in route.stops:
                roles[stop.sumo_node_id] = stop.node_type.replace("_dummy", "")
        return roles

    @staticmethod
    def _collect_node_labels(routes: list) -> dict[str, str]:
        labels: dict[str, str] = {}
        for route in routes:
            for stop in route.stops:
                base_type = stop.node_type.replace("_dummy", "")
                if base_type not in {"depot", "customer", "afs"}:
                    continue
                if stop.sumo_node_id in labels:
                    continue
                label = getattr(stop, "node_name", None)
                if not isinstance(label, str) or not label:
                    continue
                if "_d" in label:
                    label = label.split("_d", 1)[0]
                labels[stop.sumo_node_id] = label
        return labels
