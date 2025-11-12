import argparse
import yaml
import os
import random
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import traci

def draw_circle(id_prefix, center, color, radius=5, num_points=16):
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

def build_graph():
    G = nx.DiGraph()

    # Get all edges (skip internal)
    edge_ids = [e for e in traci.edge.getIDList() if not e.startswith(":")]

    for eid in edge_ids:
        from_node = traci.edge.getFromJunction(eid)
        to_node = traci.edge.getToJunction(eid)

        lid = f"{eid}_0"  # Assume single lane per edge for simplicity
        length = traci.lane.getLength(lid)
        speed = traci.edge.getLastStepMeanSpeed(eid)

        # Add nodes (junctions)
        fx, fy = traci.junction.getPosition(from_node)
        tx, ty = traci.junction.getPosition(to_node)
        G.add_node(from_node, x=fx, y=fy)
        G.add_node(to_node, x=tx, y=ty)

        # Add directed edge with attributes
        G.add_edge(
            from_node,
            to_node,
            id=eid,
            length=length,
            speed=speed,
            weight=length / speed if speed > 0 else float("inf")
        )

    return G

def distribute_tasks(task_list, num_ugvs):
    num_tasks_per_ugv = max(1, len(task_list) // num_ugvs)
    task_list_dist = []
    for i in range(num_ugvs):
        start_idx = i * num_tasks_per_ugv
        end_idx = start_idx + num_tasks_per_ugv
        task_list_dist.append(task_list[start_idx:end_idx])
    return task_list_dist

if __name__ == "__main__":
    config_path = 'config/sumo_lmd.yaml'
    config = yaml.safe_load(open(config_path, 'r'))

    net_path = config["net"]["net_path"]
    warehouse_choice = config["net"]["warehouse"]

    # Task vars
    demand_per = config['task']['demand_per']

    # Agent vars
    num_ugvs = config["fleet"]["count"]
    max_ugv_range = config["fleet"]['range']
    ugv_speed = config["fleet"]['max_speed']

    # Simulation vars
    max_time = config['simulation']['max_time']
    max_steps = config['simulation']['max_steps']
    render_mode = config['simulation']['render_mode']  # Render mode from config
    render_fps = config['simulation'].get('render_fps', 30)  # Default to 30 if not specified
    step_length = config['simulation']['step_length']

    # Use only the network file, not the full config to avoid route conflicts
    net_file = net_path.replace('.sumocfg', '.net.xml')
    Sumo_config = [
        'sumo',
        '--net-file', net_file,
        '--step-length', str(step_length),
        '--lateral-resolution', '0.1',
        # '--delay', '500'
    ]

    traci.start(Sumo_config)

    random.seed(786)
    np.random.seed(786)
    G = build_graph() # Rebuild graph with new maze
    # pos = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes}
    # plt.figure(figsize=(8, 8))
    # nx.draw(G, pos=pos, node_size=10, width=0.5, with_labels=False)
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig('sumo_graph.png', dpi=200)
    # plt.close()
    warehouse_node = random.choice(list(G.nodes))
    num_tasks = (demand_per * len(G.nodes))//100
    task_list = random.choices(list(G.nodes), k=num_tasks) # Randomly select tasks from graph nodes
    task_list_dist = distribute_tasks(task_list, num_ugvs) # Distribute tasks to UGVs

    # Highlight warehouse
    warehouse_pos = traci.junction.getPosition(warehouse_node)
    draw_circle("warehouse", warehouse_pos, (0, 255, 0, 255), radius=8)  # green circle

    # Highlight all task locations
    for i, t in enumerate(task_list):
        pos = traci.junction.getPosition(t)
        draw_circle(f"task_{i}", pos, (255, 0, 0, 255), radius=5)  # red circles

    print(f"Warehouse at junction: {warehouse_node}")
    print(f"Total tasks: {num_tasks}, Tasks per UGV: {len(task_list_dist[0])}")
    print(f"Task list distribution among UGVs: {task_list_dist}")

    ugv_edge_routes = []
    for ugv_id in range(num_ugvs):
        curr_node = warehouse_node
        edge_path = []
        while task_list_dist[ugv_id]:
            task_node = task_list_dist[ugv_id][0]
            path = nx.dijkstra_path(G, curr_node, task_node)

            # Convert node path to edge path
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                edge_id = G[from_node][to_node]['id']
                edge_path.append(edge_id)

            curr_node = task_node
            task_list_dist[ugv_id].pop(0)

        ugv_edge_routes.append(edge_path)

        # Create a TraCI route and add a vehicle
        route_id = f"ugv_{ugv_id}_route"
        traci.route.add(route_id, edge_path)

        veh_id = f"ugv_{ugv_id}"
        traci.vehicle.add(veh_id, route_id)
        traci.vehicle.setColor(veh_id, (0, 0, 255, 255))

    print("UGV routes have been created and vehicles added to the simulation.")
    print("UGV Edge Routes:")
    for i, route in enumerate(ugv_edge_routes):
        print(f"UGV {i}: {route}")

    start_time = None
    end_time = None

    # Run the simulation until our vehicle has arrived (or safety cap)
    safety_cap_steps = max_steps  # Use configured max_steps
    steps = 0

    try:
        while True:
            traci.simulationStep()
            sim_t = traci.simulation.getTime()

            # When the vehicle first appears in the network
            if start_time is None and "ugv_0" in traci.vehicle.getIDList():
                start_time = sim_t
                print(f"Vehicle ugv_0 started at time {start_time:.2f}s")

            # When the vehicle is no longer in the network AND it has arrived
            if start_time is not None and "ugv_0" not in traci.vehicle.getIDList():
                end_time = sim_t
                print(f"Vehicle ugv_0 finished at time {end_time:.2f}s")
                break

            # # Check if simulation should end due to time limit
            # if sim_t >= max_time:
            #     print(f"Simulation reached time limit of {max_time}s")
            #     break

            steps += 1

        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            print("\n=== Result ===")
            print(f"Vehicle:      ugv_0")
            print(f"Steps:        {steps}")
            print(f"Route time:   {duration:.2f} s (sim time)")
        else:
            print("\n=== Warning ===")
            print("Vehicle did not complete its route within the time/step limits")
            print(f"Steps completed: {steps}")
            print(f"Simulation time: {traci.simulation.getTime():.2f}s")

    except traci.exceptions.FatalTraCIError as e:
        print(f"\nTraCI Error: {e}")
        print("This usually means SUMO simulation ended unexpectedly")
    except Exception as e:
        print(f"\nUnexpected error during simulation: {e}")
    finally:
        # Always close TraCI connection
        try:
            traci.close()
            print("TraCI connection closed successfully")
        except:
            pass