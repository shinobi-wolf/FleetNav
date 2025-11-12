#!/usr/bin/env python3
"""
Visualize SUMO Simulation from Server Log

This script takes a policy name, reads the server log and config,
builds the SUMO network graph, creates routes for each UGV based on
their logged positions, and replays the simulation in SUMO.

Usage:
    python visualize_sumo_simulation.py <policy_name>
    python visualize_sumo_simulation.py dijkstra_sumo
"""

import argparse
import yaml
import os
import re
from collections import defaultdict
import traci
import networkx as nx
import time


def parse_server_log(log_path):
    """
    Parse the server log file to extract UGV positions at each timestep.
    
    Returns:
        config_path: Path to the config file
        ugv_trajectories: Dict mapping ugv_id -> list of (step, position, status) tuples
    """
    ugv_trajectories = defaultdict(list)
    config_path = None
    
    with open(log_path, 'r') as f:
        current_step = None
        for line in f:
            line = line.strip()
            
            # Parse config path
            if line.startswith('Config:'):
                config_path = line.split('Config:')[1].strip()
            
            # Parse step number
            elif line.startswith('Step'):
                match = re.match(r'Step (\d+):', line)
                if match:
                    current_step = int(match.group(1))
            
            # Parse UGV state
            elif line.startswith('UGV'):
                # Format: UGV 0: Position=E0, Active Task=B0, Status=1
                match = re.match(r'UGV (\d+): Position=(\w+), Active Task=(\w+|None), Status=(\d+)', line)
                if match and current_step is not None:
                    ugv_id = int(match.group(1))
                    position = match.group(2)
                    active_task = match.group(3) if match.group(3) != 'None' else None
                    status = int(match.group(4))
                    
                    ugv_trajectories[ugv_id].append({
                        'step': current_step,
                        'position': position,
                        'active_task': active_task,
                        'status': status
                    })
    
    return config_path, ugv_trajectories


def build_graph():
    """Build a NetworkX graph from the SUMO network using TraCI."""
    G = nx.DiGraph()
    
    # Get all edges (skip internal)
    edge_ids = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    
    for eid in edge_ids:
        from_node = traci.edge.getFromJunction(eid)
        to_node = traci.edge.getToJunction(eid)
        
        lid = f"{eid}_0"  # Assume single lane per edge for simplicity
        length = traci.lane.getLength(lid)
        
        # Add nodes (junctions) with positions
        fx, fy = traci.junction.getPosition(from_node)
        tx, ty = traci.junction.getPosition(to_node)
        G.add_node(from_node, x=fx, y=fy)
        G.add_node(to_node, x=tx, y=ty)
        
        # Add directed edge with attributes
        G.add_edge(from_node, to_node, id=eid, length=length)
    
    return G


def convert_node_path_to_edges(node_path, graph):
    """Convert a sequence of nodes to a sequence of edge IDs."""
    edge_path = []
    for i in range(len(node_path) - 1):
        from_node = node_path[i]
        to_node = node_path[i + 1]
        
        if graph.has_edge(from_node, to_node):
            edge_id = graph[from_node][to_node]['id']
            edge_path.append(edge_id)
        else:
            # If no direct edge, try to find shortest path
            try:
                intermediate_path = nx.shortest_path(graph, from_node, to_node)
                for j in range(len(intermediate_path) - 1):
                    edge_id = graph[intermediate_path[j]][intermediate_path[j + 1]]['id']
                    edge_path.append(edge_id)
            except nx.NetworkXNoPath:
                print(f"Warning: No path found from {from_node} to {to_node}")
    
    return edge_path


def extract_ugv_routes(ugv_trajectories, graph):
    """
    Extract edge-based routes for each UGV from their node trajectories.
    
    Returns:
        Dict mapping ugv_id -> list of edge IDs
    """
    ugv_routes = {}
    
    for ugv_id, trajectory in ugv_trajectories.items():
        # Extract unique positions in order (remove duplicates while preserving order)
        node_path = []
        prev_position = None
        for entry in trajectory:
            position = entry['position']
            if position != prev_position:
                node_path.append(position)
                prev_position = position
        
        # Convert node path to edge path
        if len(node_path) > 1:
            edge_path = convert_node_path_to_edges(node_path, graph)
            ugv_routes[ugv_id] = edge_path
            print(f"UGV {ugv_id}: {len(node_path)} nodes -> {len(edge_path)} edges")
            print(f"  Node path: {' -> '.join(node_path)}")
        else:
            print(f"UGV {ugv_id}: No movement detected (stayed at {node_path[0] if node_path else 'unknown'})")
            ugv_routes[ugv_id] = []
    
    return ugv_routes


def draw_circle(id_prefix, center, color, radius=5, num_points=16):
    """Draw a circle polygon in SUMO using TraCI."""
    import math
    cx, cy = center
    points = [
        (cx + radius * math.cos(2 * math.pi * i / num_points),
         cy + radius * math.sin(2 * math.pi * i / num_points))
        for i in range(num_points)
    ]
    traci.polygon.add(
        f"{id_prefix}",
        shape=points,
        color=color,
        fill=True,
        layer=10
    )


def visualize_simulation(policy_name):
    """
    Main visualization function.
    
    Args:
        policy_name: Name of the policy (e.g., 'dijkstra_sumo')
    """
    # Construct paths
    run_dir = f"runs/{policy_name}"
    log_path = os.path.join(run_dir, "server_log.txt")
    
    if not os.path.exists(log_path):
        print(f"Error: Server log not found at {log_path}")
        return
    
    print(f"Reading server log from: {log_path}")
    
    # Parse the log
    config_path, ugv_trajectories = parse_server_log(log_path)
    
    if not config_path:
        print("Error: Could not find config path in server log")
        return
    
    # Handle relative path in config
    if not os.path.isabs(config_path):
        # If the config path is relative, it's relative to the current directory
        if not os.path.exists(config_path):
            # Try relative to run directory
            config_path = os.path.join(run_dir, os.path.basename(config_path))
    
    if not os.path.exists(config_path):
        # Fall back to the run directory's config
        config_path = os.path.join(run_dir, "config.yaml")
    
    print(f"Loading config from: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    net_path = config["net"]["net_path"]
    step_length = config['simulation']['step_length']
    
    print(f"Network path: {net_path}")
    print(f"Found {len(ugv_trajectories)} UGVs in log")
    
    # Use only the network file, not the full config to avoid route conflicts
    # net_file = net_path.replace('.sumocfg', '.net.xml')
    
    sumo_config = [
        'sumo-gui',  # Use GUI for visualization
        '-c', net_path,
        '--step-length', str(step_length),
        '--delay', '500',
        '--start',  # Auto-start simulation
        '--quit-on-end',  # Close when done
    ]
    
    print("\nStarting SUMO...")
    traci.start(sumo_config)
    
    try:
        # Build graph
        print("Building network graph...")
        G = build_graph()
        print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        # Extract routes from trajectories
        print("\nExtracting UGV routes...")
        ugv_routes = extract_ugv_routes(ugv_trajectories, G)
        
        # Visualize task locations and routes
        # Collect all unique task locations from the trajectories
        all_tasks = set()
        for ugv_id, trajectory in ugv_trajectories.items():
            for entry in trajectory:
                if entry['active_task'] and entry['active_task'] != 'None':
                    all_tasks.add(entry['active_task'])
        
        # Draw task locations (red circles)
        print(f"\nVisualizing {len(all_tasks)} task locations...")
        for i, task_node in enumerate(sorted(all_tasks)):
            if task_node in G.nodes:
                pos = traci.junction.getPosition(task_node)
                draw_circle(f"task_{task_node}", pos, (255, 0, 0, 255), radius=5)
        
        # Find warehouse (start position of UGV 0)
        if ugv_trajectories:
            warehouse_node = ugv_trajectories[0][0]['position']
            print(f"Warehouse node: {warehouse_node}")
            if warehouse_node in G.nodes:
                warehouse_pos = traci.junction.getPosition(warehouse_node)
                draw_circle("warehouse", warehouse_pos, (0, 255, 0, 255), radius=8)
        
        # Create routes and vehicles
        print("\nCreating SUMO routes and vehicles...")
        for ugv_id, edge_path in ugv_routes.items():
            if edge_path:  # Only create route if there are edges
                route_id = f"ugv_{ugv_id}_route"
                traci.route.add(route_id, edge_path)
                
                veh_id = f"ugv_{ugv_id}"
                traci.vehicle.add(veh_id, route_id)
                
                # Color vehicles blue
                traci.vehicle.setColor(veh_id, (0, 0, 255, 255))
                
                print(f"  Created route for UGV {ugv_id}: {len(edge_path)} edges")
        
        # Run simulation
        print("\n" + "="*60)
        print("Starting simulation replay...")
        print("Close SUMO window to exit")
        print("="*60)
        
        step = 0
        active_vehicles = set(ugv_routes.keys())
        
        while active_vehicles:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            
            # Check which vehicles have finished
            current_vehicles = set()
            for ugv_id in active_vehicles:
                veh_id = f"ugv_{ugv_id}"
                if veh_id in traci.vehicle.getIDList():
                    current_vehicles.add(ugv_id)
                else:
                    print(f"UGV {ugv_id} completed route at step {step} (t={sim_time:.1f}s)")
            
            active_vehicles = current_vehicles
            
            # Print progress every 10 steps
            if step % 10 == 0 and active_vehicles:
                print(f"Step {step} (t={sim_time:.1f}s): {len(active_vehicles)} vehicles active")
            
            step += 1
            
            # Safety limit
            if step > 10000:
                print("Reached step limit (10000), stopping...")
                break
        
        print(f"\nSimulation complete! Total steps: {step}")
        
    except traci.exceptions.FatalTraCIError as e:
        print(f"\nTraCI Error: {e}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            traci.close()
            print("TraCI connection closed")
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Visualize SUMO simulation from server log',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_sumo_simulation.py dijkstra_sumo
  python visualize_sumo_simulation.py dqn_sumo
        """
    )
    parser.add_argument(
        'policy_name',
        type=str,
        help='Name of the policy (e.g., dijkstra_sumo, dqn_sumo)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SUMO Simulation Visualizer")
    print("="*60)
    print(f"Policy: {args.policy_name}\n")
    
    visualize_simulation(args.policy_name)


if __name__ == "__main__":
    main()
