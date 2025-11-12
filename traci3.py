#!/usr/bin/env python3
# Interactive SUMO simulation: Place vehicle at random junction, route to another,
# wait for user input (Enter to continue, Escape to exit).

import random
import sys
from typing import List, Tuple, Optional
import networkx as nx

import traci

def build_graph():
    """Build a NetworkX directed graph from the SUMO network."""
    G = nx.DiGraph()

    # Get all edges (skip internal)
    edge_ids = [e for e in traci.edge.getIDList() if not e.startswith(":")]

    for eid in edge_ids:
        try:
            from_node = traci.edge.getFromJunction(eid)
            to_node = traci.edge.getToJunction(eid)

            # Get edge length (try lane first, fallback to edge)
            length = 0.0
            try:
                # Get first lane of the edge
                lanes = traci.edge.getLaneNumber(eid)
                if lanes > 0:
                    lid = f"{eid}_0"
                    length = traci.lane.getLength(lid)
                else:
                    # Fallback: use edge length if available
                    length = traci.edge.getLength(eid)
            except:
                # If lane doesn't exist, try to get edge length directly
                try:
                    length = traci.edge.getLength(eid)
                except:
                    length = 100.0  # Default fallback

            # Get speed (use max speed if current speed is 0)
            try:
                speed = traci.edge.getLastStepMeanSpeed(eid)
                if speed <= 0:
                    # Use max allowed speed for the edge
                    speed = traci.edge.getMaxSpeed(eid)
                    if speed <= 0:
                        speed = 13.89  # Default ~50 km/h in m/s
            except:
                speed = 13.89  # Default fallback

            # Add nodes (junctions)
            G.add_node(from_node)
            G.add_node(to_node)

            # Calculate weight (travel time: length / speed)
            weight = length / speed if speed > 0 else float("inf")

            # Add directed edge with attributes
            G.add_edge(
                from_node,
                to_node,
                id=eid,
                length=length,
                speed=speed,
                weight=weight
            )
        except traci.exceptions.TraCIException:
            # Skip edges that cause errors
            continue

    return G

def get_junction_edges(G: nx.DiGraph, junction_id: str) -> Tuple[List[str], List[str]]:
    """Get outgoing and incoming edge IDs for a junction using the graph."""
    if junction_id not in G:
        return [], []
    
    # Get outgoing edges (edges from this junction)
    outgoing = [G[u][v]['id'] for u, v in G.out_edges(junction_id)]
    
    # Get incoming edges (edges to this junction)
    incoming = [G[u][v]['id'] for u, v in G.in_edges(junction_id)]
    
    return outgoing, incoming


def choose_random_junction(G: nx.DiGraph) -> str:
    """Return a random junction ID from the graph."""
    nodes = list(G.nodes())
    if len(nodes) < 1:
        raise SystemExit("Network must have at least 1 junction.")
    return random.choice(nodes)


def find_route_to_junction(G: nx.DiGraph, veh_id: str, target_junction: str) -> Optional[Tuple[str, List[str]]]:
    """
    Find a route from the vehicle's current position to the target junction using the graph.
    Returns (end_edge, route_edges) or None if no route found.
    """
    # Get vehicle's current edge
    if veh_id not in traci.vehicle.getIDList():
        return None
    
    try:
        current_edge_id = traci.vehicle.getRoadID(veh_id)
    except traci.exceptions.TraCIException:
        return None
    
    # Find which junction the current edge leads to (to_node)
    # Try TraCI method first, then fallback to graph lookup
    current_junction = None
    try:
        current_junction = traci.edge.getToJunction(current_edge_id)
    except traci.exceptions.TraCIException:
        # Fallback: find the junction from the graph
        for u, v, data in G.edges(data=True):
            if data['id'] == current_edge_id:
                # Use the 'to' node (v) as the current junction
                current_junction = v
                break
    
    if current_junction is None:
        # If still not found, try to find from_node (edge starts from)
        for u, v, data in G.edges(data=True):
            if data['id'] == current_edge_id:
                current_junction = u  # Use from node as fallback
                break
    
    if current_junction is None:
        return None
    
    # Check if target junction is reachable
    if target_junction not in G:
        return None
    
    # Use NetworkX to find shortest path from current junction to target junction
    try:
        if current_junction == target_junction:
            # Already at target, just need to get an incoming edge to stay there
            _, incoming = get_junction_edges(G, target_junction)
            if incoming:
                return incoming[0], [incoming[0]]
            return None
        
        # Check if path exists
        if not nx.has_path(G, current_junction, target_junction):
            return None
        
        # Find shortest path using NetworkX (weighted by travel time)
        path_nodes = nx.shortest_path(G, current_junction, target_junction, weight='weight')
        
        if not path_nodes or len(path_nodes) < 2:
            return None
        
        # Convert node path to edge IDs
        route_edges = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if G.has_edge(u, v):
                edge_id = G[u][v]['id']
                route_edges.append(edge_id)
        
        if not route_edges:
            return None
        
        # The last edge is the one that leads to the target junction
        end_edge = route_edges[-1]
        
        return end_edge, route_edges
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def place_vehicle_at_junction(G: nx.DiGraph, veh_id: str, junction_id: str) -> bool:
    """
    Place a vehicle at a junction by selecting an outgoing edge and creating a route.
    Uses the graph to find outgoing edges.
    Returns True if successful, False otherwise.
    """
    outgoing, _ = get_junction_edges(G, junction_id)
    if not outgoing:
        print(f"Warning: Junction '{junction_id}' has no outgoing edges.")
        return False
    
    # Pick a random outgoing edge and create a minimal route
    start_edge = random.choice(outgoing)
    
    # Create a route with just this edge (vehicle will be placed at the start)
    route_id = f"r_{junction_id}"
    try:
        traci.route.add(route_id, [start_edge])
        traci.vehicle.add(veh_id, route_id, depart="now")
        traci.vehicle.setStop(veh_id, start_edge, G.edges[start_edge]['length']-1, duration=99999999)
        traci.vehicle.setColor(veh_id, (0, 0, 255, 255))  # Blue
        return True
    except traci.exceptions.TraCIException as e:
        # Route might already exist or vehicle might already exist
        try:
            traci.route.delete(route_id)
            traci.route.add(route_id, [start_edge])
        except:
            pass
        try:
            traci.vehicle.remove(veh_id)
            traci.vehicle.add(veh_id, route_id, depart="now")
            traci.vehicle.setColor(veh_id, (0, 0, 255, 255))
        except:
            pass
        return True


def wait_for_user_input(veh_id: str = None):
    """
    Wait for user to press Enter while optionally running simulation steps.
    Returns True if Enter pressed, False if Escape pressed or user quits.
    """
    try:
        import select
        import termios
        import tty
        import time
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to cbreak mode (character by character, no line buffering)
            tty.setcbreak(sys.stdin.fileno())
            
            print("Press Enter to route to next junction, Escape to quit: ", end='', flush=True)
            
            last_step_time = time.time()
            step_interval = 0.05  # Run simulation step every 50ms (matches step-length)
            
            while True:
                # Check if input is available (non-blocking with timeout)
                timeout = 0.1
                ready, _, _ = select.select([sys.stdin], [], [], timeout)
                
                if ready:
                    key = sys.stdin.read(1)
                    
                    if key == '\x1b':  # Possible Escape key or start of escape sequence
                        # Check if there are more characters (escape sequence)
                        # If no more characters within a short time, it's Escape key
                        if not select.select([sys.stdin], [], [], 0.05)[0]:
                            # No more characters, it's an Escape key
                            print("\nEscape pressed. Exiting...")
                            return False
                        # Otherwise, it's an escape sequence (like arrow keys), ignore it
                        # Read the rest of the sequence to clear it
                        try:
                            while select.select([sys.stdin], [], [], 0.01)[0]:
                                sys.stdin.read(1)
                        except:
                            pass
                    elif key == '\n' or key == '\r':  # Enter key
                        print()
                        return True
                    elif key == 'q' or key == 'Q':  # 'q' key (quick quit)
                        print("\nQuit requested. Exiting...")
                        return False
                    elif ord(key) == 3:  # Ctrl+C
                        raise KeyboardInterrupt
                else:
                    # No input available, run simulation step if enough time has passed
                    current_time = time.time()
                    if current_time - last_step_time >= step_interval:
                        try:
                            traci.simulationStep()
                            last_step_time = current_time
                        except:
                            pass
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
    except (ImportError, OSError, AttributeError):
        # Fallback for systems without termios (e.g., Windows) or if stdin is not a TTY
        # In fallback mode, we can't run simulation while waiting, so simulation will pause
        try:
            user_input = input("Press Enter to continue (or type 'q' to quit): ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                return False
            return True
        except (EOFError, KeyboardInterrupt):
            return False


def run_simulation_step():
    """Run one simulation step."""
    traci.simulationStep()


def main():
    # ---- SUMO config style from traci2.py ----
    Sumo_config = [
        'sumo-gui',
        '-c', 'data/grid/grid.sumocfg',
        '--step-length', '0.05',
        '--delay', '500',
        '--lateral-resolution', '0.1'
    ]
    
    # Start SUMO+TraCI
    traci.start(Sumo_config)
    
    veh_id = "veh_blue_0"
    route_counter = 0
    
    try:
        # Build the NetworkX graph from the SUMO network
        print("Building network graph...")
        G = build_graph()
        
        # Get list of junctions from the graph
        junctions = list(G.nodes())
        if len(junctions) < 1:
            raise SystemExit("Network has no junctions.")
        
        print(f"Graph built: {len(junctions)} junctions, {G.number_of_edges()} edges")
        print("=== Interactive SUMO Simulation ===")
        print("Press Enter to route vehicle to next random junction")
        print("Type 'q' and press Enter to quit\n")
        
        # Initial placement: choose random junction and place vehicle there
        print("Step 1: Placing vehicle at random junction...")
        start_junction = choose_random_junction(G)
        print(f"Selected junction: {start_junction}")
        
        if not place_vehicle_at_junction(G, veh_id, start_junction):
            raise SystemExit(f"Failed to place vehicle at junction '{start_junction}'")
        
        # Run simulation a few steps to let vehicle appear
        for _ in range(10):
            run_simulation_step()
        
        print(f"Vehicle '{veh_id}' placed at junction '{start_junction}'")
        print("Press Enter to route to next junction...")
        
        # Main loop: wait for user, then route to next junction
        while True:
            # Wait for user input (simulation continues running in background)
            if not wait_for_user_input(veh_id):
                print("\nExiting simulation...")
                break
            
            # Choose random destination junction
            target_junction = choose_random_junction(G)
            print(f"\nRouting vehicle to junction: {target_junction}")
            
            # Find route to target junction using the graph
            route_result = find_route_to_junction(G, veh_id, target_junction)
            
            if route_result is None:
                print(f"Warning: Could not find route to junction '{target_junction}'. Trying another...")
                continue
            
            end_edge, route_edges = route_result
            
            # Update vehicle's route
            route_id = f"r_{route_counter}"
            route_counter += 1
            
            try:
                # Remove old route if it exists (cleanup)
                try:
                    traci.route.delete(route_id)
                except:
                    pass
                
                # Add new route
                traci.route.add(route_id, route_edges)
                
                # Change vehicle's route to the new route
                # If vehicle is already in the simulation, use changeTarget or setRoute
                if veh_id in traci.vehicle.getIDList():
                    # Vehicle is already in simulation, change its target/route
                    try:
                        # Try to change the route directly
                        traci.vehicle.setRoute(veh_id, route_edges)
                    except traci.exceptions.TraCIException:
                        # Fallback: change target edge
                        try:
                            traci.vehicle.changeTarget(veh_id, end_edge)
                        except traci.exceptions.TraCIException:
                            # If that fails, try rerouting
                            traci.vehicle.rerouteTraveltime(veh_id)
                            traci.vehicle.changeTarget(veh_id, end_edge)
                else:
                    # Vehicle not in simulation yet, should not happen but handle it
                    print("Warning: Vehicle not found in simulation")
                    continue
                
                print(f"Route created: {len(route_edges)} edges")
                print(f"Vehicle is now routing to junction '{target_junction}'")
                print("(Simulation continues running. Press Enter when ready for next route...)")
                    
            except traci.exceptions.TraCIException as e:
                print(f"Error updating route: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except SystemExit as e:
        print(str(e), file=sys.stderr)
    finally:
        try:
            traci.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        try:
            traci.close(False)
        except Exception:
            pass
        sys.exit(1)


