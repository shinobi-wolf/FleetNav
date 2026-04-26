"""SUMO TraCI simulator for EVRP route execution.

Workflow
--------
1. Receive list[Route] from any planner.
2. Start SUMO (headless or GUI).
3. For each vehicle, replay stops in order:
     depot → [customer | AFS]* → depot
   - Between stops: call traci.simulation.findRoute() to get the edge
     sequence, add a vehicle, step until it arrives.
   - At AFS stops: vehicle "recharges" (fuel reset to Q, wait refuel_time).
4. Apply global piecewise traffic model every seg_duration minutes.
5. Collect metrics and return them.

Render modes
------------
  quiet  — no output beyond final metrics.
  print  — log each vehicle event (customer served, refuel, traffic change).
  human  — sumo-gui + pygame window; sumo.delay is applied between steps.

Traffic model
-------------
  - A seeded RNG draws one speed value per time segment from the range
    [speed_min_kmh, speed_max_kmh] in steps of speed_step_kmh.
  - Every seg_duration simulated minutes the commanded speed is updated
    for all active EV vehicles via traci.vehicle.setMaxSpeed().
  - When traffic.enabled is false, vehicles use the free-flow speed from
    the GVRP params (speed_mph converted to m/s).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

try:
    import traci  # type: ignore[import-not-found]
    import sumolib  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(f"traci/sumolib required for simulation: {e}") from e

from ..routes.base import Route, Stop
from ..traffic.base import BaseTrafficModel
from ..traffic.segmented import SegmentedSpeedModel, build_traffic_model, _mph_to_ms
from .pygame_renderer import PygameRenderer


def _resolve_net_xml(net_path: str) -> str:
    p = Path(net_path)
    if p.suffix == ".sumocfg":
        return str(p.with_suffix(".net.xml"))
    return str(p)


def _node_to_incoming_edge(net, node_id: str) -> str | None:
    """Return any edge whose toNode is node_id (used as arrival proxy)."""
    node = net.getNode(node_id)
    if node is None:
        return None
    incoming = node.getIncoming()
    if not incoming:
        return None
    # pick non-internal edge
    for e in incoming:
        if not e.getID().startswith(":"):
            return e.getID()
    return None


def _node_to_outgoing_edge(net, node_id: str) -> str | None:
    """Return any edge whose fromNode is node_id (used as departure proxy)."""
    node = net.getNode(node_id)
    if node is None:
        return None
    outgoing = node.getOutgoing()
    if not outgoing:
        return None
    for e in outgoing:
        if not e.getID().startswith(":"):
            return e.getID()
    return None


# ---------------------------------------------------------------------------
# Per-vehicle runtime state
# ---------------------------------------------------------------------------

@dataclass
class _VehicleState:
    vehicle_id: int
    veh_id: str                  # SUMO id string
    current_fuel: float          # gallons
    fuel_capacity: float
    consumption_rate: float      # gallons per metre
    distance_m: float = 0.0
    num_recharges: int = 0
    customers_served: int = 0
    finish_time_s: float = 0.0
    active: bool = False         # currently in SUMO simulation
    energy_kwh: float = 0.0

    # 1 gallon of gasoline ≈ 33.7 kWh; adjust if you switch to kWh/km
    _GAL_TO_KWH: float = field(default=33.7, init=False, repr=False)

    def consume(self, dist_m: float) -> None:
        used = self.consumption_rate * dist_m
        self.current_fuel = max(0.0, self.current_fuel - used)
        self.distance_m += dist_m
        self.energy_kwh += used * self._GAL_TO_KWH

    @property
    def range_m(self) -> float:
        """Remaining range in metres given current fuel."""
        if self.consumption_rate <= 0:
            return float("inf")
        return self.current_fuel / self.consumption_rate

    def refuel(self) -> None:
        self.current_fuel = self.fuel_capacity
        self.num_recharges += 1


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class EVRPSimulator:
    """Execute a planned list[Route] inside SUMO and collect metrics."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._sumo_cfg = config.get("sumo", {})
        self._sim_cfg = config.get("simulation", {})
        self._gvrp_cfg = config.get("gvrp", {})
        self._net_cfg = config.get("net", {})

        net_path = str(self._net_cfg.get("net_path", ""))
        self.net_xml_path = _resolve_net_xml(net_path)
        self.sumocfg_path = net_path if net_path.endswith(".sumocfg") else net_path

        step_length = float(self._sumo_cfg.get("step_length", 1.0))
        self.step_length = step_length

        speed_mph = float(self._gvrp_cfg.get("speed_mph", 40.0))
        self.free_flow_ms = _mph_to_ms(speed_mph)

        max_time_h = float(self._sim_cfg.get("max_time", 12.0))
        self.max_time_s = max_time_h * 3600.0

        self.fuel_capacity = float(self._gvrp_cfg.get("fuel_capacity", 7.0))
        # consumption_rate in config is gal/mile; convert to gal/metre
        gal_per_mile = float(self._gvrp_cfg.get("consumption_rate", 1.0))
        self.consumption_rate_per_m = gal_per_mile / 1609.344

        self.refuel_time_s = float(self._gvrp_cfg.get("refuel_time_hours", 0.25)) * 3600.0
        self.service_time_s = float(self._gvrp_cfg.get("customer_service_time_hours", 0.5)) * 3600.0

        self.runtime_refuel = bool(self._sim_cfg.get("runtime_refuel", True))
        self.use_planned_wait_times = bool(self._sim_cfg.get("use_planned_wait_times", False))
        self.render_mode = str(self._sim_cfg.get("render_mode", "quiet")).strip().lower()
        valid_render_modes = {"quiet", "print", "human"}
        if self.render_mode not in valid_render_modes:
            valid_modes_text = ", ".join(sorted(valid_render_modes))
            raise ValueError(
                f"Invalid simulation.render_mode '{self.render_mode}'. "
                f"Valid modes: {valid_modes_text}"
            )
        self.traffic_scale = float(self._sumo_cfg.get("traffic_scale", 0.0))
        self.gui_delay = float(self._sumo_cfg.get("delay", 0))

        self._net = None  # loaded on run()
        self._spawn_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, routes: list[Route]) -> dict[str, Any]:
        """Simulate routes and return metrics dict."""
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

        # human mode: sumo-gui + pygame window both active
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

    def _execute(self, routes: list[Route], traffic: BaseTrafficModel, renderer: PygameRenderer | None) -> dict[str, Any]:
        vehicle_states: dict[int, _VehicleState] = {}
        for route in routes:
            vs = _VehicleState(
                vehicle_id=route.vehicle_id,
                veh_id=f"ev_{route.vehicle_id}",
                current_fuel=self.fuel_capacity,
                fuel_capacity=self.fuel_capacity,
                consumption_rate=self.consumption_rate_per_m,
            )
            vehicle_states[route.vehicle_id] = vs

        # All vehicles start at the depot
        vehicle_positions: dict[int, str] = {
            r.vehicle_id: r.stops[0].sumo_node_id for r in routes if r.stops
        }
        served_customers: set[str] = set()

        sim_time_s = float(traci.simulation.getTime())
        traffic.update(sim_time_s)

        # Initial draw so the pygame window appears before any vehicle moves.
        if renderer is not None:
            renderer.draw(
                sim_time_s=sim_time_s,
                traffic_speed_ms=traffic.commanded_speed_ms,
                vehicle_positions=vehicle_positions,
                served_customers=served_customers,
                vehicle_states=vehicle_states,
            )

        customer_completion_times: list[float] = []

        for route in routes:
            vs = vehicle_states[route.vehicle_id]
            sim_time_s = self._execute_route(
                route, vs, traffic, sim_time_s, customer_completion_times,
                renderer, vehicle_positions, served_customers, vehicle_states,
            )
            vs.finish_time_s = sim_time_s

        traci.simulationStep()  # drain any remaining

        return self._build_metrics(vehicle_states, customer_completion_times)

    def _execute_route(
        self,
        route: Route,
        vs: _VehicleState,
        traffic: BaseTrafficModel,
        sim_time_s: float,
        customer_completion_times: list[float],
        renderer: PygameRenderer | None,
        vehicle_positions: dict[int, str],
        served_customers: set[str],
        vehicle_states: dict[int, _VehicleState],
    ) -> float:
        """Drive one vehicle through its planned stops. Returns updated sim time.

        Fuel constraints enforced at runtime
        -------------------------------------
        Before driving to the next planned stop the simulator checks:

        1. **Hard fuel gate** — is there enough fuel to physically reach the
           next stop?  If not the vehicle's run ends where it is.
        2. **Range callback** — is there enough fuel to reach the next stop
           *and* return to the depot from there?  If not the vehicle drives
           back to the depot first, refuels, then resumes its route.
        3. **Depot recharge** — any depot visit (planned or unplanned
           trackback) fully recharges the vehicle.
        """
        stops = route.stops
        if len(stops) < 2:
            return sim_time_s

        # Identify the depot node from the first stop.
        depot_node = stops[0].sumo_node_id

        leg_idx = 0
        while leg_idx < len(stops) - 1:
            src_stop = stops[leg_idx]
            dst_stop = stops[leg_idx + 1]

            src_node = src_stop.sumo_node_id
            dst_node = dst_stop.sumo_node_id

            # ── Same-node dummy stops (AFS / depot copies) ────────────
            if src_node == dst_node:
                wait_s = (
                    dst_stop.planned_wait_s
                    if self.use_planned_wait_times and dst_stop.planned_wait_s > 0
                    else self.refuel_time_s
                )
                sim_time_s = self._wait(sim_time_s, wait_s, traffic)
                vs.refuel()
                leg_idx += 1
                continue

            dist_to_next = self._route_distance_m(src_node, dst_node)

            # ── Hard fuel gate ─────────────────────────────────────────
            # Not enough fuel to even reach the next stop → end run.
            if vs.range_m < dist_to_next:
                if self.render_mode in ("print", "human"):
                    print(
                        f"  vehicle {vs.vehicle_id} stranded at {src_stop.node_name}: "
                        f"range={vs.range_m:.1f}m < {dist_to_next:.1f}m to "
                        f"{dst_stop.node_name}, ending run"
                    )
                break

            # ── Range callback (optional) ──────────────────────────────
            # Check: can the vehicle reach dst *and* still return to depot?
            dist_dst_to_depot = self._route_distance_m(dst_node, depot_node)
            if self.runtime_refuel and vs.range_m < dist_to_next + dist_dst_to_depot:
                # Not enough range for the round-trip — trackback to depot.
                dist_to_depot = self._route_distance_m(src_node, depot_node)

                if vs.range_m < dist_to_depot:
                    # Cannot even reach the depot from here → end run.
                    if self.render_mode in ("print", "human"):
                        print(
                            f"  vehicle {vs.vehicle_id} stranded at "
                            f"{src_stop.node_name}: cannot reach depot, ending run"
                        )
                    break

                # Drive back to depot, refuel, then resume from this leg_idx.
                edge_list = self._find_edge_route(src_node, depot_node)
                if not edge_list:
                    if self.render_mode in ("print", "human"):
                        print(
                            f"  [warn] vehicle {vs.vehicle_id}: no route to depot "
                            f"from {src_node}, ending run"
                        )
                    break

                dist_m, sim_time_s = self._drive(vs, edge_list, sim_time_s, traffic)
                vs.consume(dist_m)
                vehicle_positions[vs.vehicle_id] = depot_node

                # Refuel at depot
                sim_time_s = self._wait(sim_time_s, self.refuel_time_s, traffic)
                vs.refuel()
                if self.render_mode in ("print", "human"):
                    print(
                        f"  vehicle {vs.vehicle_id} trackback refuel at depot "
                        f"t={sim_time_s:.1f}s  fuel={vs.current_fuel:.3f}gal"
                    )

                if renderer is not None:
                    renderer.draw(
                        sim_time_s=sim_time_s,
                        traffic_speed_ms=traffic.commanded_speed_ms,
                        vehicle_positions=vehicle_positions,
                        served_customers=served_customers,
                        vehicle_states=vehicle_states,
                    )

                # After refuel, update src to depot and retry this leg.
                # Replace the source stop with the depot for the next
                # iteration so _find_edge_route starts from depot.
                stops[leg_idx] = Stop(
                    sumo_node_id=depot_node,
                    node_name="D",
                    node_type="depot",
                    planned_fuel=vs.current_fuel,
                    planned_time=0.0,
                )
                continue  # retry leg_idx (now from depot)

            # ── Drive to next stop ─────────────────────────────────────
            edge_list = self._find_edge_route(src_node, dst_node)
            if not edge_list:
                if self.render_mode in ("print", "human"):
                    print(
                        f"  [warn] vehicle {vs.vehicle_id}: no SUMO route "
                        f"{src_node} -> {dst_node}, skipping leg"
                    )
                leg_idx += 1
                continue

            dist_m, sim_time_s = self._drive(vs, edge_list, sim_time_s, traffic)
            vs.consume(dist_m)

            vehicle_positions[vs.vehicle_id] = dst_node
            ntype = dst_stop.node_type

            if ntype == "customer":
                wait_s = (
                    dst_stop.planned_wait_s
                    if self.use_planned_wait_times and dst_stop.planned_wait_s > 0
                    else self.service_time_s
                )
                sim_time_s = self._wait(sim_time_s, wait_s, traffic)
                vs.customers_served += 1
                served_customers.add(dst_node)
                customer_completion_times.append(sim_time_s)
                if self.render_mode in ("print", "human"):
                    print(
                        f"  vehicle {vs.vehicle_id} served {dst_stop.node_name} "
                        f"at t={sim_time_s:.1f}s  fuel={vs.current_fuel:.3f}gal"
                    )
            elif ntype in ("afs", "afs_dummy"):
                wait_s = (
                    dst_stop.planned_wait_s
                    if self.use_planned_wait_times and dst_stop.planned_wait_s > 0
                    else self.refuel_time_s
                )
                sim_time_s = self._wait(sim_time_s, wait_s, traffic)
                vs.refuel()
                if self.render_mode in ("print", "human"):
                    print(
                        f"  vehicle {vs.vehicle_id} refuelled at {dst_stop.node_name} "
                        f"t={sim_time_s:.1f}s"
                    )
            elif ntype in ("depot", "depot_dummy"):
                wait_s = (
                    dst_stop.planned_wait_s
                    if self.use_planned_wait_times and dst_stop.planned_wait_s > 0
                    else self.refuel_time_s
                )
                sim_time_s = self._wait(sim_time_s, wait_s, traffic)
                vs.refuel()
                if self.render_mode in ("print", "human"):
                    print(
                        f"  vehicle {vs.vehicle_id} refuelled at depot "
                        f"t={sim_time_s:.1f}s"
                    )

            if renderer is not None:
                renderer.draw(
                    sim_time_s=sim_time_s,
                    traffic_speed_ms=traffic.commanded_speed_ms,
                    vehicle_positions=vehicle_positions,
                    served_customers=served_customers,
                    vehicle_states=vehicle_states,
                )

            leg_idx += 1

        return sim_time_s

    def _drive(
        self,
        vs: _VehicleState,
        edge_list: list[str],
        sim_time_s: float,
        traffic: BaseTrafficModel,
    ) -> tuple[float, float]:
        """Add vehicle to SUMO, step until done. Returns (dist_m, new_sim_time_s)."""
        veh_id = f"{vs.veh_id}_leg{self._spawn_counter}"
        self._spawn_counter += 1

        route_id = f"route_{veh_id}"
        traci.route.add(route_id, edge_list)

        speed = traffic.commanded_speed_ms
        try:
            vtype_list = list(traci.vehicletype.getIDList())
            vtype = "DEFAULT_VEHTYPE" if "DEFAULT_VEHTYPE" in vtype_list else (vtype_list[0] if vtype_list else None)
        except traci.TraCIException:
            vtype = None

        if vtype:
            traci.vehicle.add(veh_id, route_id, typeID=vtype, departSpeed=speed)
        else:
            traci.vehicle.add(veh_id, route_id, departSpeed=speed)

        traci.vehicle.setMaxSpeed(veh_id, speed)
        traci.vehicle.setColor(veh_id, (255, 80, 0, 255))
        vs.active = True

        dist_m = 0.0
        guard = 0
        max_steps = int(self.max_time_s / self.step_length) + 1000

        while guard < max_steps:
            traci.simulationStep()
            sim_time_s = traci.simulation.getTime()
            guard += 1

            # Update traffic
            if traffic.update(sim_time_s):
                new_speed = traffic.commanded_speed_ms
                try:
                    active = set(traci.vehicle.getIDList())
                    if veh_id in active:
                        traci.vehicle.setMaxSpeed(veh_id, new_speed)
                        traci.vehicle.setSpeed(veh_id, new_speed)
                except traci.TraCIException:
                    pass
                if self.render_mode in ("print", "human"):
                    print(
                        f"  traffic speed -> {traffic.commanded_speed_ms*3.6:.1f} km/h "
                        f"at t={sim_time_s:.1f}s"
                    )

            # Check if vehicle still exists
            try:
                active = set(traci.vehicle.getIDList())
            except traci.TraCIException:
                active = set()

            if veh_id in active:
                try:
                    cur_dist = traci.vehicle.getDistance(veh_id)
                    dist_m = max(dist_m, cur_dist)
                except traci.TraCIException:
                    pass
            else:
                # Vehicle finished route and was removed by SUMO
                vs.active = False
                break

            if sim_time_s >= self.max_time_s:
                # Force remove
                try:
                    traci.vehicle.remove(veh_id)
                except traci.TraCIException:
                    pass
                vs.active = False
                break

        return dist_m, sim_time_s

    def _wait(
        self, sim_time_s: float, duration_s: float, traffic: BaseTrafficModel
    ) -> float:
        """Advance simulation by duration_s without spawning a vehicle."""
        target = sim_time_s + duration_s
        while sim_time_s < target:
            traci.simulationStep()
            sim_time_s = traci.simulation.getTime()
            traffic.update(sim_time_s)
        return sim_time_s

    def _find_sumo_edge_route(self, src_node: str, dst_node: str) -> list[str]:
        """Use TraCI's built-in router to get edge list from node to node."""
        src_edge = _node_to_outgoing_edge(self._net, src_node)
        dst_edge = _node_to_incoming_edge(self._net, dst_node)

        if not src_edge or not dst_edge:
            return []

        try:
            result = traci.simulation.findRoute(src_edge, dst_edge)
            edges = list(result.edges)
            if not edges:
                return []
            return edges
        except traci.TraCIException:
            return []
        
    def _build_routing_graph(self) -> nx.DiGraph:
        """Build directed graph from the loaded SUMO net (mirrors load_sumo_graph).

        Each non-internal SUMO edge becomes a DiGraph edge weighted by its length
        and labelled with its SUMO edge ID.  This graph is used by _find_edge_route
        so that simulation distances match the planner's shortest-path distances.
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

    def _find_edge_route(self, src_node: str, dst_node: str) -> list[str]:
        """Return SUMO edge ID list for the shortest path from src to dst.

        Uses the nx routing graph (same graph / same weights as the planner)
        so simulation distances are consistent with planning distances.
        """
        try:
            node_path = nx.shortest_path(
                self._routing_graph, src_node, dst_node, weight="length"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        edge_ids = []
        for u, v in zip(node_path[:-1], node_path[1:]):
            edge_data = self._routing_graph.get_edge_data(u, v)
            if edge_data:
                edge_ids.append(edge_data["id"])
        return edge_ids

    def _route_distance_m(self, src_node: str, dst_node: str) -> float:
        """Shortest-path distance in metres between two nodes on the routing graph."""
        try:
            return float(
                nx.shortest_path_length(
                    self._routing_graph, src_node, dst_node, weight="length"
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    @staticmethod
    def _collect_node_roles(routes: list[Route]) -> dict[str, str]:
        """Build {sumo_node_id: 'depot'|'customer'|'afs'} from all route stops."""
        roles: dict[str, str] = {}
        for route in routes:
            for stop in route.stops:
                base_type = stop.node_type.replace("_dummy", "")
                # depot_dummy → depot, afs_dummy → afs; customer stays customer
                roles[stop.sumo_node_id] = base_type
        return roles

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _build_metrics(
        self,
        vehicle_states: dict[int, _VehicleState],
        customer_completion_times: list[float],
    ) -> dict[str, Any]:
        all_vs = list(vehicle_states.values())

        makespan_s = max((vs.finish_time_s for vs in all_vs), default=0.0)
        total_dist_m = sum(vs.distance_m for vs in all_vs)
        total_energy_kwh = sum(vs.energy_kwh for vs in all_vs)
        total_recharges = sum(vs.num_recharges for vs in all_vs)
        total_customers = sum(vs.customers_served for vs in all_vs)

        per_vehicle = {
            vs.vehicle_id: {
                "distance_m": vs.distance_m,
                "energy_kwh": vs.energy_kwh,
                "num_recharges": vs.num_recharges,
                "customers_served": vs.customers_served,
                "finish_time_s": vs.finish_time_s,
            }
            for vs in all_vs
        }

        avg_gap_s = None
        if len(customer_completion_times) > 1:
            gaps = [
                customer_completion_times[i] - customer_completion_times[i - 1]
                for i in range(1, len(customer_completion_times))
            ]
            avg_gap_s = float(sum(gaps) / len(gaps))

        metrics = {
            "makespan_s": makespan_s,
            "makespan_h": makespan_s / 3600.0,
            "total_distance_m": total_dist_m,
            "total_distance_km": total_dist_m / 1000.0,
            "total_energy_kwh": total_energy_kwh,
            "total_recharges": total_recharges,
            "customers_served": total_customers,
            "customer_completion_times_s": customer_completion_times,
            "avg_inter_customer_gap_s": avg_gap_s,
            "per_vehicle": per_vehicle,
        }

        return metrics
