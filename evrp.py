import numpy as np
import pandas as pd
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random

def create_distance_matrix(G):
        nodes = sorted(G.nodes())
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

G = nx.read_gml('data/maze.gml')
depot_node = '0'
num_vehicles = 5
fuel_capacity = 10000000

distance_matrix, nodes, node_to_idx = create_distance_matrix(G)
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
ugv_task_list = []
if solution:
    for vehicle_id in range(routing.vehicles()):
        route = []
        index = routing.Start(vehicle_id)
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            route.append(nodes[node_idx])
            index = solution.Value(routing.NextVar(index))
            
        node_idx = manager.IndexToNode(index)
        route.append(nodes[node_idx])
        ugv_task_list.append(route[1:])  # Exclude depot from tasks

print(ugv_task_list)