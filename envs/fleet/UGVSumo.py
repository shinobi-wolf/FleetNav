import networkx as nx
import numpy as np

class UGVSumo:
    def __init__(self,ugv_id, base_position, max_range=1000000, max_speed=1):
        super().__init__()
        self.agent_id = ugv_id
        self.position = base_position  # e.g., (row, col) or (x, y)
        self.base_point = base_position  # Base station position
        self.max_range = max_range       # maximum battery capacity or range
        self.max_speed = max_speed

        self.current_range = self.max_range    # start fully charged
        self.current_range_percent = 1.0
        self.status = 1

        self.distance_traveled = 0.0

        self.task_list = []            # List of tasks (nodes) to complete
        self.edge_route=[]
        self.veh_id=None
        
        # Reward tracking variables
        self.task_completed_this_step = False  # Flag for task completion in current step
        self.invalid_action_this_step = False  # Flag for invalid action in last step
        self.last_move_time = None  # Start time of the last move
        self.last_position = None
        
    def move(self, action, current_time, graph, graph_dir, node2idx, idx2node):
        self.task_completed_this_step = False
        self.invalid_action_this_step = False
        if self.status != 1:
            # Not in moving state, action is invalid
            self.last_move_time = None
            return False, None, None

        self.last_move_time = current_time
        
        current_node = idx2node[self.position]
        nb_nodes = list(graph.neighbors(current_node))
        neighbors = sorted([node2idx[nb] for nb in nb_nodes])

        if action >= len(neighbors):
            # Action out of bounds
            self.invalid_action_this_step = True
            return False, None, None

        next_node = idx2node[neighbors[action]]
        edge_id = graph.edges[(current_node, next_node)]['id']
        edge_dir = 0
        if (current_node, next_node) not in graph_dir.edges:
            edge_dir = 1  # reverse direction
        edge_length = graph.edges[(current_node, next_node)]['length']
        edge_type = graph.edges[(current_node, next_node)]['type']
        # if edge_type =='crossing':
        #     self.last_move_time = -1
        self.current_range -= edge_length
        self.current_range_percent = self.current_range / self.max_range
        if self.current_range_percent <= 0:
            self.invalid_action_this_step = True
            self.status = 2
            return False, None, None
        self.distance_traveled += edge_length
        self.position = node2idx[next_node]
        self.edge_route.append(edge_id)
        return True, edge_id, edge_dir

    def recharge(self):
        """
        Recharge the agent to its maximum range.
        """
        self.current_range = self.max_range
        self.current_range_percent = 1.0
        self.charging_status = 0

    def get_actual_position(self, global_time):
        delta_t = self.local_time - global_time
        if delta_t <= 0:
            return self.position
        curr_speed = self.max_speed / (1 + self.traffic_delay_factor*self.current_traffic)  # Speed reduces with increasing traffic
        delta_d = curr_speed * delta_t/self.cell_dist
        moves = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
        }
        hr,hc = moves[self.current_heading]
        (ar, ac) = self.position[0] - hr*delta_d, self.position[1] - hc*delta_d
        return (ar, ac)


    def __str__(self):
        return (f"UGV Agent ID: {self.agent_id}, "
                f"Position: {self.position}, "
                f"Active Task List: {self.active_task_list}, "
                f"Dest Task: {self.task_list[0]}, "
                f"Current Range: {self.current_range}, "
                f"Local Time: {self.local_time}, "
                f"Distance Traveled: {self.distance_traveled}, "
                f"Energy Consumed: {self.energy_consumed}, "
                f"Status: {self.status}, "
                f"Charging Status: {'Charging' if self.charging_status else 'Not Charging'}")

# Example usage:
if __name__ == "__main__":
    # Create a UGVAgent with ID 1, initial position (5, 5) and a max range of 100.
    ugv = UGV(agent_id=1, initial_position=(5, 5), max_range=100)
    
    # Print out the agent and its action space.
    print(ugv)
    print("Action Space:", ugv.action_space)