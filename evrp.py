import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyamaze import maze
import networkx as nx
import tkinter as tk
tk.Tk.state = lambda self, s=None: self.wm_state('normal' if s == 'zoomed' else s)
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random


def generate_df_maze(rows=20, cols=20, lp=80):
    # Create a Pyamaze maze
    m = maze(rows, cols)
    m.CreateMaze(loopPercent=lp)  # No loops to match a grid-like maze

    # List to store cell data
    maze_data = []
    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            cell = (x, y)
            cell_walls = m.maze_map[cell]  # Get wall info for each cell

            E = 1 if cell_walls['E'] else 0
            W = 1 if cell_walls['W'] else 0
            N = 1 if cell_walls['N'] else 0
            S = 1 if cell_walls['S'] else 0

            maze_data.append([f"({x},{y})", E, W, N, S])
    df = pd.DataFrame(maze_data, columns=["  cell  ", "E", "W", "N", "S"])
    if '  cell  ' in df.columns:
        df[['row', 'col']] = (
            df['  cell  '].astype(str)
            .str.replace(r'[()]', '', regex=True)
            .str.split(',', expand=True)
            .astype(int)
        )
    return df

def build_graph(rows=10, cols=10, lp=80, df_maze=None):
    df_maze = generate_df_maze(rows, cols, lp)
    G = nx.Graph()
    for _, row in df_maze.iterrows():
        r, c = int(row['row']), int(row['col'])
        G.add_node((r, c))
        if row['E'] == 1:
            G.add_edge((r, c), (r, c+1), traffic=0)
        if row['W'] == 1:
            G.add_edge((r, c), (r, c - 1), traffic=0)
        if row['N'] == 1:
            G.add_edge((r, c), (r - 1, c), traffic=0)
        if row['S'] == 1:
            G.add_edge((r, c), (r + 1,c), traffic=0)

    all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    return G, all_shortest_path_lengths
# Convert the graph to a distance matrix for OR-Tools
def create_distance_matrix(G, tasks):
    nodes = sorted(tasks)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    dist_matrix = np.zeros((n, n), dtype=int)
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j:
                if nx.has_path(G, node_i, node_j):
                    dist = nx.shortest_path_length(G, node_i, node_j)
                    dist_matrix[i, j] = dist * 50  # Scale for better numeric stability
                else:
                    dist_matrix[i, j] = 999999  # No path exists
    
    return dist_matrix, nodes, node_to_idx

# Create a VRP solution using OR-Tools
def solve_vrp(G, tasks, num_vehicles=3, fuel_capacity=3000, depot_node=(5,5)):
    distance_matrix, nodes, node_to_idx = create_distance_matrix(G, tasks)
    depot_idx = node_to_idx[depot_node]
    
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return distance_matrix[from_node][to_node]
    
    transit_callback_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

    num_nodes = manager.GetNumberOfNodes()
    slacks = [0] * num_nodes
    
    
    routing.AddDimension(
        transit_callback_idx,
        0,  # no slack
        fuel_capacity,  # vehicle fuel capacity
        True,  # start cumul to zero
        'Fuel'
    )
    distance_dimension = routing.GetDimensionOrDie('Fuel')
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        return solution, routing, manager, nodes
    else:
        return None, None, None, nodes

# Visualize the solution
def visualize_solution(G, solution, routing, manager, nodes):
    if not solution:
        print("No solution found.")
        return
    
    plt.figure(figsize=(10, 10))
    pos = {node: (node[1], -node[0]) for node in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=200)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    
    depot_node = (5, 5)
    nx.draw_networkx_nodes(G, pos, nodelist=[depot_node], node_color='red', node_size=300, label='Depot/Charging')
    
    colors = ['blue', 'green', 'orange']
    
    for vehicle_id in range(routing.vehicles()):
        route = []
        index = routing.Start(vehicle_id)
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            route.append(nodes[node_idx])
            index = solution.Value(routing.NextVar(index))
            
        node_idx = manager.IndexToNode(index)
        route.append(nodes[node_idx])
        
        edge_list = []
        for i in range(len(route)-1):
            if route[i] != route[i+1]:
                path = nx.shortest_path(G, route[i], route[i+1])
                for j in range(len(path)-1):
                    edge_list.append((path[j], path[j+1]))
        
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_list, 
            edge_color=colors[vehicle_id], 
            width=2, label=f"Vehicle {vehicle_id+1}"
        )
    
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.legend()
    plt.title("Vehicle Routing Problem Solution with 3 Vehicles")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    for vehicle_id in range(routing.vehicles()):
        print(f"Route for Vehicle {vehicle_id+1}:")
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_string = ""
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            route_string += f" {nodes[node_idx]} ->"
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
        node_idx = manager.IndexToNode(index)
        route_string += f" {nodes[node_idx]}"
        print(f"Distance: {route_distance/100}\nRoute: {route_string}\n")


G,_ = build_graph(10,10,60)
tasks = random.sample(list(G.nodes), k=10)
if (5,5) not in tasks:
    tasks.append((5,5))
# Solve and visualize the VRP
solution, routing, manager, nodes = solve_vrp(G, tasks, num_vehicles=3, fuel_capacity=3000, depot_node=(5,5))
visualize_solution(G, solution, routing, manager, nodes)