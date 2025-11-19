#!/usr/bin/env python3
"""
Visualize SUMO Simulation from Server Log

This script takes a policy name, reads the server log and config,
builds the SUMO network graph, creates routes for each UGV based on
their logged positions, and replays the simulation in SUMO.

It can also compare two policies side-by-side with metrics and color-coded routes.

Usage:
    python visualize_sumo_simulation.py <policy_name>
    python visualize_sumo_simulation.py dijkstra_sumo
    python visualize_sumo_simulation.py <policy1> --compare <policy2>
    python visualize_sumo_simulation.py dijkstra --compare dqn
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


def extract_ugv_routes(ugv_trajectories, graph, verbose=True):
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
            if verbose:
                print(f"UGV {ugv_id}: {len(node_path)} nodes -> {len(edge_path)} edges")
                print(f"  Node path: {' -> '.join(node_path)}")
        else:
            if verbose:
                print(f"UGV {ugv_id}: No movement detected (stayed at {node_path[0] if node_path else 'unknown'})")
            ugv_routes[ugv_id] = []
    
    return ugv_routes


def calculate_route_metrics(ugv_routes, ugv_trajectories, graph):
    """
    Calculate metrics for a set of routes.
    
    Returns:
        Dict with metrics: total_distance, total_steps, max_steps, num_ugvs, tasks_completed
    """
    total_distance = 0.0
    total_steps = 0
    max_steps = 0
    tasks_completed = set()
    
    for ugv_id, edge_path in ugv_routes.items():
        # Calculate distance using graph edge lengths
        for edge_id in edge_path:
            # Find the edge in the graph by matching edge IDs
            for u, v, data in graph.edges(data=True):
                if data.get('id') == edge_id:
                    total_distance += data.get('length', 0.0)
                    break
        
        # Get steps from trajectory
        if ugv_id in ugv_trajectories:
            trajectory = ugv_trajectories[ugv_id]
            if trajectory:
                steps = trajectory[-1]['step'] - trajectory[0]['step'] + 1
                total_steps += steps
                max_steps = max(max_steps, steps)
                
                # Count unique tasks
                for entry in trajectory:
                    if entry['active_task'] and entry['active_task'] != 'None':
                        tasks_completed.add(entry['active_task'])
    
    return {
        'total_distance': total_distance,
        'total_steps': total_steps,
        'max_steps': max_steps,
        'num_ugvs': len(ugv_routes),
        'tasks_completed': len(tasks_completed),
        'avg_distance_per_ugv': total_distance / len(ugv_routes) if ugv_routes else 0,
        'avg_steps_per_ugv': total_steps / len(ugv_routes) if ugv_routes else 0
    }


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


def print_comparison_table(metrics1, metrics2, policy1_name, policy2_name):
    """Print a formatted comparison table of metrics."""
    print("\n" + "="*80)
    print("POLICY COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {policy1_name:<25} {policy2_name:<25} {'Difference':<15}")
    print("-"*80)
    
    metrics_to_compare = [
        ('Total Distance (m)', 'total_distance', '{:.2f}'),
        ('Total Steps', 'total_steps', '{:d}'),
        ('Max Steps (any UGV)', 'max_steps', '{:d}'),
        ('Tasks Completed', 'tasks_completed', '{:d}'),
        ('Avg Distance/UGV (m)', 'avg_distance_per_ugv', '{:.2f}'),
        ('Avg Steps/UGV', 'avg_steps_per_ugv', '{:.1f}'),
    ]
    
    for label, key, fmt in metrics_to_compare:
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)
        diff = val1 - val2
        diff_str = f"{diff:+.2f}" if isinstance(diff, float) else f"{diff:+d}"
        
        print(f"{label:<30} {fmt.format(val1):<25} {fmt.format(val2):<25} {diff_str:<15}")
    
    print("="*80)
    
    # Determine winner
    better_policy = None
    if metrics1['total_distance'] < metrics2['total_distance']:
        better_policy = f"{policy1_name} (shorter distance)"
    elif metrics2['total_distance'] < metrics1['total_distance']:
        better_policy = f"{policy2_name} (shorter distance)"
    
    if metrics1['total_steps'] < metrics2['total_steps']:
        if better_policy:
            better_policy += f" and {policy1_name} (fewer steps)"
        else:
            better_policy = f"{policy1_name} (fewer steps)"
    elif metrics2['total_steps'] < metrics1['total_steps']:
        if better_policy:
            better_policy += f" and {policy2_name} (fewer steps)"
        else:
            better_policy = f"{policy2_name} (fewer steps)"
    
    if better_policy:
        print(f"\nBest Policy: {better_policy}")
    else:
        print("\nPolicies are similar in performance")
    print()


def visualize_simulation(policy_name, compare_policy=None):
    """
    Main visualization function.
    
    Args:
        policy_name: Name of the first policy (e.g., 'dijkstra_sumo')
        compare_policy: Optional name of second policy to compare
    """
    # Load first policy
    run_dir1 = f"runs/{policy_name}"
    log_path1 = os.path.join(run_dir1, "server_log.txt")
    
    if not os.path.exists(log_path1):
        print(f"Error: Server log not found at {log_path1}")
        return
    
    print(f"Reading server log from: {log_path1}")
    config_path1, ugv_trajectories1 = parse_server_log(log_path1)
    
    if not config_path1:
        print("Error: Could not find config path in server log")
        return
    
    # Handle relative path in config
    if not os.path.isabs(config_path1):
        if not os.path.exists(config_path1):
            config_path1 = os.path.join(run_dir1, os.path.basename(config_path1))
    
    if not os.path.exists(config_path1):
        config_path1 = os.path.join(run_dir1, "config.yaml")
    
    print(f"Loading config from: {config_path1}")
    with open(config_path1, 'r') as f:
        config1 = yaml.safe_load(f)
    
    net_path = config1["net"]["net_path"]
    step_length = config1['simulation']['step_length']
    
    # Load second policy if comparing
    ugv_trajectories2 = None
    if compare_policy:
        run_dir2 = f"runs/{compare_policy}"
        log_path2 = os.path.join(run_dir2, "server_log.txt")
        
        if not os.path.exists(log_path2):
            print(f"Error: Comparison server log not found at {log_path2}")
            return
        
        print(f"Reading comparison server log from: {log_path2}")
        config_path2, ugv_trajectories2 = parse_server_log(log_path2)
    
    print(f"Network path: {net_path}")
    print(f"Policy 1: Found {len(ugv_trajectories1)} UGVs in log")
    if compare_policy:
        print(f"Policy 2: Found {len(ugv_trajectories2)} UGVs in log")
    
    # Use only the network file, not the full config to avoid route conflicts
    # net_file = net_path.replace('.sumocfg', '.net.xml')
    
    sumo_config = [
        'sumo-gui',  # Use GUI for visualization
        '-c', net_path,
        '--step-length', str(step_length),
        '--delay', '200',
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
        print("\nExtracting UGV routes for Policy 1...")
        ugv_routes1 = extract_ugv_routes(ugv_trajectories1, G, verbose=not compare_policy)
        
        # Calculate metrics for policy 1
        metrics1 = calculate_route_metrics(ugv_routes1, ugv_trajectories1, G)
        
        # Extract routes for policy 2 if comparing
        ugv_routes2 = None
        metrics2 = None
        if compare_policy and ugv_trajectories2:
            print("\nExtracting UGV routes for Policy 2...")
            ugv_routes2 = extract_ugv_routes(ugv_trajectories2, G, verbose=False)
            metrics2 = calculate_route_metrics(ugv_routes2, ugv_trajectories2, G)
            
            # Print comparison table
            print_comparison_table(metrics1, metrics2, policy_name, compare_policy)
        
        # Visualize task locations and routes
        # Collect all unique task locations from both trajectories
        all_tasks = set()
        for ugv_id, trajectory in ugv_trajectories1.items():
            for entry in trajectory:
                if entry['active_task'] and entry['active_task'] != 'None':
                    all_tasks.add(entry['active_task'])
        
        if compare_policy and ugv_trajectories2:
            for ugv_id, trajectory in ugv_trajectories2.items():
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
        if ugv_trajectories1:
            warehouse_node = ugv_trajectories1[0][0]['position']
            print(f"Warehouse node: {warehouse_node}")
            if warehouse_node in G.nodes:
                warehouse_pos = traci.junction.getPosition(warehouse_node)
                draw_circle("warehouse", warehouse_pos, (0, 255, 0, 255), radius=8)
        
        # Create routes and vehicles for policy 1 (blue)
        print("\nCreating SUMO routes and vehicles for Policy 1 (BLUE)...")
        for ugv_id, edge_path in ugv_routes1.items():
            if edge_path:  # Only create route if there are edges
                veh_id = f"p1_ugv_{ugv_id}"
                first_edge = edge_path[0]
                traci.person.add(veh_id, first_edge, 0.0)
                traci.person.setColor(veh_id, (255, 0, 0, 255))
                
                traci.person.appendWalkingStage(veh_id, edge_path, 
                                                arrivalPos=-1, speed=3.0)

                
                if not compare_policy:
                    print(f"  Created route for UGV {ugv_id}: {len(edge_path)} edges")
        
        # Create routes and vehicles for policy 2 (red/orange) if comparing
        if compare_policy and ugv_routes2:
            print("\nCreating SUMO routes and vehicles for Policy 2 (RED/ORANGE)...")
            for ugv_id, edge_path in ugv_routes2.items():
                if edge_path:  # Only create route if there are edges
                    veh_id = f"p2_ugv_{ugv_id}"
                    first_edge = edge_path[0]  
                    traci.person.add(veh_id, first_edge, 0.0)
                    traci.person.setColor(veh_id, (255, 255, 0, 255))

                    traci.person.appendWalkingStage(veh_id, edge_path, 
                                                    arrivalPos=-1, speed=3.0)
        
        # Update active vehicles set
        active_vehicles = set(ugv_routes1.keys())
        if compare_policy and ugv_routes2:
            active_vehicles.update(ugv_routes2.keys())
        
        # Run simulation
        print("\n" + "="*60)
        if compare_policy:
            print(f"Starting simulation replay...")
            print(f"Policy 1 ({policy_name}): BLUE vehicles")
            print(f"Policy 2 ({compare_policy}): RED/ORANGE vehicles")
        else:
            print("Starting simulation replay...")
        print("Close SUMO window to exit")
        print("="*60)
        
        step = 0
        completed_p1 = set()
        completed_p2 = set()
        
        while active_vehicles:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            
            # Check which vehicles have finished
            current_vehicles = set()
            for ugv_id in ugv_routes1.keys():
                veh_id = f"p1_ugv_{ugv_id}"
                if veh_id in traci.person.getIDList():
                    current_vehicles.add(ugv_id)
                elif ugv_id not in completed_p1:
                    completed_p1.add(ugv_id)
                    print(f"Policy 1 - UGV {ugv_id} completed route at step {step} (t={sim_time:.1f}s)")
            
            if compare_policy and ugv_routes2:
                for ugv_id in ugv_routes2.keys():
                    veh_id = f"p2_ugv_{ugv_id}"
                    if veh_id in traci.person.getIDList():
                        current_vehicles.add(ugv_id)
                    elif ugv_id not in completed_p2:
                        completed_p2.add(ugv_id)
                        print(f"Policy 2 - UGV {ugv_id} completed route at step {step} (t={sim_time:.1f}s)")
            
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
        if compare_policy:
            print(f"Policy 1 completed: {len(completed_p1)}/{len(ugv_routes1)} UGVs")
            if ugv_routes2:
                print(f"Policy 2 completed: {len(completed_p2)}/{len(ugv_routes2)} UGVs")
        
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
  python visualize_sumo_simulation.py dijkstra
  python visualize_sumo_simulation.py dqn
  python visualize_sumo_simulation.py dijkstra --compare dqn
        """
    )
    parser.add_argument(
        'policy_name',
        type=str,
        help='Name of the first policy (e.g., dijkstra, dqn)'
    )
    parser.add_argument(
        '--compare',
        type=str,
        default=None,
        help='Name of second policy to compare (e.g., --compare dqn)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SUMO Simulation Visualizer")
    print("="*60)
    if args.compare:
        print(f"Comparing Policies:")
        print(f"  Policy 1: {args.policy_name} (BLUE)")
        print(f"  Policy 2: {args.compare} (RED/ORANGE)")
    else:
        print(f"Policy: {args.policy_name}")
    print()
    
    visualize_simulation(args.policy_name, args.compare)


if __name__ == "__main__":
    main()
