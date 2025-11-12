import networkx as nx
import numpy as np

class UGVSumo:
    def __init__(self,ugv_id, base_position, max_range, max_speed, G):
        super().__init__()
        self.agent_id = ugv_id
        self.position = base_position  # e.g., (row, col) or (x, y)
        self.base_point = base_position  # Base station position
        self.max_range = max_range       # maximum battery capacity or range
        self.G = G
        self.max_speed = max_speed

        self.current_range = self.max_range    # start fully charged
        self.current_range_percent = 1.0
        self.status = 1

        self.distance_traveled = 0.0

        self.task_list = []            # List of tasks (nodes) to complete
        self.edge_route=[]
        self.veh_id=None
        
    def move(self, action, graph):
        if self.status==1:
            current_node = self.position
            neighbors = sorted(list(graph.neighbors(current_node)))

            if action < len(neighbors):
                next_node = neighbors[action]
                edge_id = graph.edges[(current_node, next_node)]['id']
                edge_length = graph.edges[(current_node, next_node)]['length']
                self.distance_traveled += edge_length
                self.current_range -= edge_length
                self.current_range_percent = self.current_range / self.max_range
                self.position = next_node
                if self.current_range >= 0:
                    self.edge_route.append(edge_id)
                    return True, edge_id
                else:
                    self.status = 2
        return False, None

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