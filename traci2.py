#!/usr/bin/env python3
# Spawn one blue vehicle, assign it a route, and report how long it took to finish.
# Uses your same SUMO config style and TraCI.

import random
import sys
from typing import List, Tuple

import traci


def choose_reachable_route_edges(candidates: List[str], max_tries: int = 200) -> Tuple[str, str, List[str]]:
    """Pick two non-internal edges and let SUMO compute a shortest path route between them.
    Returns (from_edge, to_edge, route_edge_list). Raises SystemExit if no route found."""
    for _ in range(max_tries):
        frm, to = random.sample(candidates, 2)
        # Ask SUMO for a shortest path route between frm and to
        route = traci.simulation.findRoute(frm, to)  # returns a Route-like object or dict
        # Normalize return format across SUMO versions
        if isinstance(route, dict):
            edges = route.get("edges", [])
            travel_time = route.get("travelTime", None)
            length = route.get("length", None)
        else:
            # Older TraCI returned an object with attributes
            edges = getattr(route, "edges", [])
            travel_time = getattr(route, "travelTime", None)
            length = getattr(route, "length", None)

        if edges and len(edges) >= 1:
            # Route exists
            print(f"Selected route from '{frm}' to '{to}' "
                  f"({len(edges)} edges, length={length}, tt_est={travel_time})")
            return frm, to, edges
    raise SystemExit("Could not find a reachable edge pair to build a route after many tries.")


def main():
    # ---- Your original SUMO config style ----
    Sumo_config = [
        'sumo-gui',
        '-c', 'data/grid/grid.sumocfg',
        '--step-length', '0.05',
        '--delay', '500',
        '--lateral-resolution', '0.1'
    ]

    # Start SUMO+TraCI
    traci.start(Sumo_config)

    try:
        # Get list of usable (non-internal) edges
        all_edges = traci.edge.getIDList()
        edge_candidates = [e for e in all_edges if not e.startswith(':')]
        if len(edge_candidates) < 2:
            raise SystemExit("Network has fewer than 2 non-internal edges; cannot create a route.")

        # Make selection reproducible
        random.seed(42)

        # Find a reachable route between two edges
        frm, to, route_edges = choose_reachable_route_edges(edge_candidates)

        # Create a TraCI route and add a vehicle
        route_id = "r_demo"
        traci.route.add(route_id, route_edges)

        veh_id = "veh_blue_0"
        # Add the vehicle; depart='now' so it spawns as soon as possible
        traci.vehicle.add(veh_id, route_id)

        # Color it BLUE (RGBA, 0..255)
        traci.vehicle.setColor(veh_id, (0, 0, 255, 255))

        # (Optional) cap its max speed if you want deterministic-ish runs
        # traci.vehicle.setMaxSpeed(veh_id, 20.0)

        print(f"Spawned vehicle '{veh_id}' on route '{route_id}'.")
        print(f"Route edges: {route_edges}")

        start_time = None
        end_time = None

        # Run the simulation until our vehicle has arrived (or safety cap)
        safety_cap_steps = 200000  # avoid infinite loops on weird nets (with step-length=0.05, this is 10k s)
        steps = 0

        while steps < safety_cap_steps:
            traci.simulationStep()
            sim_t = traci.simulation.getTime()

            # When the vehicle first appears in the network
            if start_time is None and veh_id in traci.vehicle.getIDList():
                start_time = sim_t

            # When the vehicle is no longer in the network AND it has arrived
            if start_time is not None and veh_id not in traci.vehicle.getIDList():
                end_time = sim_t
                break

            steps += 1

        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            print("\n=== Result ===")
            print(f"Vehicle:      {veh_id}")
            print(f"Start edge:   {frm}")
            print(f"End edge:     {to}")
            print(f"Steps:        {steps}")
            print(f"Route time:   {duration:.2f} s (sim time)")
        else:
            print("\nVehicle did not complete its route (or never entered). "
                  "Try a different network or check for network connectivity.")

    finally:
        traci.close()


if __name__ == "__main__":
    # Ensure SUMO_HOME is set and TraCI is importable in your environment.
    try:
        main()
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        try:
            traci.close(False)
        except Exception:
            pass
        sys.exit(1)
