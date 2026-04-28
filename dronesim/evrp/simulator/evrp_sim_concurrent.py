"""Concurrent (time-step-driven) EVRP simulator.

All vehicles advance simultaneously within each SUMO time step.

Workflow
--------
1. Receive list[Route] from any planner.
2. Start SUMO (headless or GUI).
3. Main loop advances one SUMO step at a time. For each vehicle every step:
     a. If at an instance node and waiting → tick down service/refuel timer.
     b. If at an instance node and ready   → compute edge path to next stop,
        spawn a SUMO leg vehicle, flip the flag to IN_TRANSIT.
     c. If IN_TRANSIT → query live XY position; detect arrival when the SUMO
        vehicle disappears from the simulation, flip flag back to AT_NODE.
4. Apply global piecewise traffic model; broadcast new speed to all active
   leg vehicles each time the segment changes.
5. Render positions in pygame every step (human mode).
6. Collect metrics and return them.

Vehicle state machine (per vehicle)
------------------------------------
  AT_NODE / waiting  : wait_remaining_s > 0
  AT_NODE / ready    : wait_remaining_s == 0, done == False
  IN_TRANSIT         : at_instance_node == False, sumo_veh_id set
  DONE               : done == True

Fuel constraints
----------------
Same hard-fuel-gate and optional range-callback (trackback) logic as the
sequential simulator, evaluated at the moment of dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import traci  # type: ignore[import-not-found]
    import sumolib  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(f"traci/sumolib required for simulation: {e}") from e

from ..routes.base import Route, Stop
from ..traffic.segmented import SegmentedSpeedModel, build_traffic_model
from .evrp_sim_sequential import EVRPSequentialSimulator, _VehicleState
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


# ---------------------------------------------------------------------------
# Extended per-vehicle state for concurrent execution
# ---------------------------------------------------------------------------

@dataclass
class _ConcVehicleState(_VehicleState):
    leg_idx: int = 0                # index into route.stops of current position
    at_instance_node: bool = True   # False while a SUMO leg vehicle is active
    wait_remaining_s: float = 0.0   # service / refuel countdown at current stop
    sumo_veh_id: str = ""           # active SUMO leg vehicle ID (empty at node)
    pending_leg_dist_m: float = 0.0 # planned distance for current leg (consumed on arrival)
    returning_to_depot: bool = False # True during a runtime-refuel trackback
    done: bool = False


# ---------------------------------------------------------------------------
# Concurrent simulator
# ---------------------------------------------------------------------------

class EVRPConcurrentSimulator(EVRPSequentialSimulator):
    """Time-step-driven simulator: all vehicles advance concurrently each step.

    Inherits config parsing, routing graph, and metrics from
    EVRPSequentialSimulator; overrides run() and _execute().
    """

    def run(self, routes: list[Route]) -> dict[str, Any]:
        self._net = sumolib.net.readNet(self.net_xml_path, withInternal=False)
        self._routing_graph = self._build_routing_graph()

        traffic = build_traffic_model(
            self.config,
            free_flow_ms=self.free_flow_ms,
            max_time_s=self.max_time_s,
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

        renderer: PygameRenderer | None = None
        if self.render_mode == "human":
            node_roles = self._collect_node_roles(routes)
            vehicle_ids = [r.vehicle_id for r in routes]
            renderer = PygameRenderer(self._net, node_roles, vehicle_ids)

        traci.start(sumo_cmd)
        try:
            metrics = self._execute(routes, traffic, renderer)
        finally:
            try:
                traci.close(False)
            except Exception:
                pass
            if renderer is not None:
                renderer.close()

        return metrics

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        routes: list[Route],
        traffic: SegmentedSpeedModel,
        renderer: PygameRenderer | None,
    ) -> dict[str, Any]:
        # Initialize vehicle states
        vehicle_states: dict[int, _ConcVehicleState] = {}
        for route in routes:
            vs = _ConcVehicleState(
                vehicle_id=route.vehicle_id,
                veh_id=f"ev_{route.vehicle_id}",
                current_fuel=self.fuel_capacity,
                fuel_capacity=self.fuel_capacity,
                consumption_rate=self.consumption_rate_per_m,
            )
            vs.done = len(route.stops) < 2
            vehicle_states[route.vehicle_id] = vs

        # vehicle_positions: sumo_node_id when at a node, (x, y) when in transit
        vehicle_positions: dict[int, str | tuple[float, float]] = {
            r.vehicle_id: r.stops[0].sumo_node_id for r in routes if r.stops
        }
        served_customers: set[str] = set()
        customer_completion_times: list[float] = []

        total_customers = sum(
            sum(1 for s in r.stops if s.node_type == "customer") for r in routes
        )

        sim_time_s = float(traci.simulation.getTime())
        traffic.update(sim_time_s)

        fleet_size = len(routes)

        if renderer is not None:
            renderer.draw(sim_time_s, traffic.commanded_speed_ms,
                          vehicle_positions, served_customers, vehicle_states)

        # Main time-step loop
        while sim_time_s < self.max_time_s:
            traci.simulationStep()
            sim_time_s = float(traci.simulation.getTime())

            if traffic.update(sim_time_s):
                self._broadcast_speed(vehicle_states, traffic.commanded_speed_ms)
                if self.render_mode in ("print", "human"):
                    print(
                        f"  traffic speed -> {traffic.commanded_speed_ms * 3.6:.1f} km/h "
                        f"at t={sim_time_s:.1f}s"
                    )

            sumo_active = set(traci.vehicle.getIDList())
            all_done = True

            for route in routes:
                vs = vehicle_states[route.vehicle_id]
                if vs.done:
                    continue
                all_done = False
                self._step_vehicle(
                    vs, route, sim_time_s, traffic, sumo_active,
                    vehicle_positions, served_customers, customer_completion_times,
                )

            if renderer is not None:
                renderer.draw(sim_time_s, traffic.commanded_speed_ms,
                              vehicle_positions, served_customers, vehicle_states)

            # Terminate early: all vehicles finished/stranded, or all customers served
            if all_done:
                break
            if total_customers > 0 and len(served_customers) >= total_customers:
                if self.render_mode in ("print", "human"):
                    print(f"  all {total_customers} customers served at t={sim_time_s:.1f}s, stopping")
                break

        # Cap any vehicles that hit max_time without finishing
        for vs in vehicle_states.values():
            if not vs.done:
                vs.finish_time_s = sim_time_s
                vs.done = True

        traci.simulationStep()
        return self._build_metrics(vehicle_states, customer_completion_times)

    # ------------------------------------------------------------------
    # Per-vehicle state machine
    # ------------------------------------------------------------------

    def _step_vehicle(
        self,
        vs: _ConcVehicleState,
        route: Route,
        sim_time_s: float,
        traffic: SegmentedSpeedModel,
        sumo_active: set[str],
        vehicle_positions: dict[int, str | tuple[float, float]],
        served_customers: set[str],
        customer_completion_times: list[float],
    ) -> None:
        if vs.at_instance_node:
            if vs.wait_remaining_s > 0:
                vs.wait_remaining_s = max(0.0, vs.wait_remaining_s - self.step_length)
            else:
                self._dispatch_leg(
                    vs, route.stops, sim_time_s, traffic, vehicle_positions,
                )
        else:
            if vs.sumo_veh_id not in sumo_active:
                self._on_arrival(
                    vs, route.stops, sim_time_s,
                    vehicle_positions, served_customers, customer_completion_times,
                )
            else:
                try:
                    xy = traci.vehicle.getPosition(vs.sumo_veh_id)
                    vehicle_positions[vs.vehicle_id] = (float(xy[0]), float(xy[1]))
                except traci.TraCIException:
                    pass

    def _dispatch_leg(
        self,
        vs: _ConcVehicleState,
        stops: list[Stop],
        sim_time_s: float,
        traffic: SegmentedSpeedModel,
        vehicle_positions: dict[int, str | tuple[float, float]],
    ) -> None:
        """Evaluate fuel constraints and spawn a SUMO leg vehicle toward next stop."""
        # After trackback: vehicle is now at depot; update the source stop record
        if vs.returning_to_depot:
            vs.returning_to_depot = False
            depot_node = stops[0].sumo_node_id
            stops[vs.leg_idx] = Stop(
                sumo_node_id=depot_node,
                node_name="D",
                node_type="depot",
                planned_fuel=vs.current_fuel,
                planned_time=0.0,
            )

        if vs.leg_idx >= len(stops) - 1:
            vs.done = True
            vs.finish_time_s = sim_time_s
            return

        src_node = stops[vs.leg_idx].sumo_node_id
        dst_stop = stops[vs.leg_idx + 1]
        dst_node = dst_stop.sumo_node_id

        # Same-node dummy stop (AFS / depot copy at same location)
        if src_node == dst_node:
            vs.refuel()
            vs.leg_idx += 1
            vs.wait_remaining_s = self.refuel_time_s
            return

        dist_to_next = self._route_distance_m(src_node, dst_node)

        # Hard fuel gate
        if vs.range_m < dist_to_next:
            if self.render_mode in ("print", "human"):
                print(
                    f"  vehicle {vs.vehicle_id} stranded at {stops[vs.leg_idx].node_name}: "
                    f"range={vs.range_m:.1f}m < {dist_to_next:.1f}m to "
                    f"{dst_stop.node_name}, ending run"
                )
            vs.done = True
            vs.finish_time_s = sim_time_s
            return

        # Optional range callback: trackback to depot if round-trip is infeasible
        depot_node = stops[0].sumo_node_id
        if self.runtime_refuel:
            dist_dst_to_depot = self._route_distance_m(dst_node, depot_node)
            if vs.range_m < dist_to_next + dist_dst_to_depot:
                dist_to_depot = self._route_distance_m(src_node, depot_node)
                if vs.range_m < dist_to_depot:
                    if self.render_mode in ("print", "human"):
                        print(
                            f"  vehicle {vs.vehicle_id} stranded at "
                            f"{stops[vs.leg_idx].node_name}: cannot reach depot, ending run"
                        )
                    vs.done = True
                    vs.finish_time_s = sim_time_s
                    return

                edge_list = self._find_edge_route(src_node, depot_node)
                if not edge_list:
                    vs.done = True
                    vs.finish_time_s = sim_time_s
                    return

                vs.returning_to_depot = True
                vs.pending_leg_dist_m = dist_to_depot
                try:
                    vs.sumo_veh_id = self._spawn_leg(vs, edge_list, traffic)
                    vs.at_instance_node = False
                except traci.TraCIException:
                    vs.returning_to_depot = False
                    vs.done = True
                    vs.finish_time_s = sim_time_s
                return

        # Normal dispatch to next planned stop
        edge_list = self._resolve_route(src_node, dst_node, sim_time_s)
        if not edge_list:
            if self.render_mode in ("print", "human"):
                print(
                    f"  [warn] vehicle {vs.vehicle_id}: no route "
                    f"{src_node} -> {dst_node}, skipping leg"
                )
            vs.leg_idx += 1
            return

        vs.pending_leg_dist_m = dist_to_next
        try:
            vs.sumo_veh_id = self._spawn_leg(vs, edge_list, traffic)
            vs.at_instance_node = False
        except traci.TraCIException:
            vs.leg_idx += 1  # spawn failed; skip this leg

    def _on_arrival(
        self,
        vs: _ConcVehicleState,
        stops: list[Stop],
        sim_time_s: float,
        vehicle_positions: dict[int, str | tuple[float, float]],
        served_customers: set[str],
        customer_completion_times: list[float],
    ) -> None:
        """Vehicle's SUMO leg finished. Consume fuel, set wait timer."""
        vs.consume(vs.pending_leg_dist_m)
        vs.pending_leg_dist_m = 0.0
        vs.active = False
        vs.sumo_veh_id = ""

        if vs.returning_to_depot:
            # Trackback leg completed — now at depot waiting to refuel
            depot_node = stops[0].sumo_node_id
            vehicle_positions[vs.vehicle_id] = depot_node
            vs.refuel()
            vs.wait_remaining_s = self.refuel_time_s
            vs.at_instance_node = True
            if self.render_mode in ("print", "human"):
                print(
                    f"  vehicle {vs.vehicle_id} trackback refuel at depot "
                    f"t={sim_time_s:.1f}s  fuel={vs.current_fuel:.3f}gal"
                )
            return

        # Normal arrival: advance to next stop
        vs.leg_idx += 1
        dst_stop = stops[vs.leg_idx]
        vehicle_positions[vs.vehicle_id] = dst_stop.sumo_node_id

        ntype = dst_stop.node_type
        if ntype == "customer":
            vs.customers_served += 1
            served_customers.add(dst_stop.sumo_node_id)
            customer_completion_times.append(sim_time_s)
            vs.wait_remaining_s = self.service_time_s
            if self.render_mode in ("print", "human"):
                print(
                    f"  vehicle {vs.vehicle_id} served {dst_stop.node_name} "
                    f"at t={sim_time_s:.1f}s  fuel={vs.current_fuel:.3f}gal"
                )
        elif ntype in ("afs", "afs_dummy"):
            vs.refuel()
            vs.wait_remaining_s = self.refuel_time_s
            if self.render_mode in ("print", "human"):
                print(
                    f"  vehicle {vs.vehicle_id} refuelled at {dst_stop.node_name} "
                    f"t={sim_time_s:.1f}s"
                )
        elif ntype in ("depot", "depot_dummy"):
            vs.refuel()
            vs.wait_remaining_s = self.refuel_time_s
            if self.render_mode in ("print", "human"):
                print(
                    f"  vehicle {vs.vehicle_id} refuelled at depot "
                    f"t={sim_time_s:.1f}s"
                )

        vs.at_instance_node = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spawn_leg(
        self,
        vs: _ConcVehicleState,
        edge_list: list[str],
        traffic: SegmentedSpeedModel,
    ) -> str:
        """Add one SUMO vehicle for a single leg. Returns its SUMO veh_id."""
        veh_id = f"{vs.veh_id}_leg{self._spawn_counter}"
        self._spawn_counter += 1

        route_id = f"route_{veh_id}"
        traci.route.add(route_id, edge_list)

        speed = traffic.commanded_speed_ms
        try:
            vtype_list = list(traci.vehicletype.getIDList())
            vtype = (
                "DEFAULT_VEHTYPE" if "DEFAULT_VEHTYPE" in vtype_list
                else (vtype_list[0] if vtype_list else None)
            )
        except traci.TraCIException:
            vtype = None

        if vtype:
            traci.vehicle.add(veh_id, route_id, typeID=vtype, departSpeed=speed)
        else:
            traci.vehicle.add(veh_id, route_id, departSpeed=speed)

        traci.vehicle.setMaxSpeed(veh_id, speed)
        color = _SUMO_COLORS[vs.vehicle_id % len(_SUMO_COLORS)]
        traci.vehicle.setColor(veh_id, color)
        vs.active = True

        return veh_id

    def _broadcast_speed(
        self,
        vehicle_states: dict[int, _ConcVehicleState],
        new_speed: float,
    ) -> None:
        """Push new traffic speed to all in-transit SUMO vehicles."""
        for vs in vehicle_states.values():
            if not vs.at_instance_node and vs.sumo_veh_id:
                try:
                    traci.vehicle.setMaxSpeed(vs.sumo_veh_id, new_speed)
                    traci.vehicle.setSpeed(vs.sumo_veh_id, new_speed)
                except traci.TraCIException:
                    pass
