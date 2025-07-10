import networkx as nx

class UGV:
    def __init__(self, ugv_id, base_position, cell_dist, max_range, drain_rate, max_speed, traffic_delay_factor, G):
        super().__init__()
        self.agent_id = ugv_id
        self.position = base_position  # e.g., (row, col) or (x, y)
        self.base_point = base_position  # Base station position
        self.max_range = max_range       # maximum battery capacity or range
        self.drain_rate= drain_rate
        self.G = G
        self.max_speed = max_speed
        self.cell_dist = cell_dist
        self.traffic_delay_factor = traffic_delay_factor

        self.current_range = self.max_range    # start fully charged
        self.current_range_percent = 1.0
        self.prev_position = None
        self.local_time = 0.0
        self.current_heading = 1
        self.current_traffic = 0
        self.act_response = None
        self.move_time = None
        
        self.distance_traveled = 0.0  # distance traveled by the UGV
        self.energy_consumed = 0.0

        self.task_list = []
        self.active_task_list = []
        self.status = 1
        self.charging_status = 0  # 0: not charging, 1: charging
        self.task_completed = False

        self.path2warehouse = None  # Path to the warehouse for charging


    def move(self, action, graph):
        """
        Move the agent based on action number:
        0: Up
        1: Right
        2: Down
        3: Left
        4: Stay
        Returns True if move was successful, False if out of range
        """
        if self.charging_status == 1:
            if self.path2warehouse is None:
                self.path2warehouse = nx.shortest_path(self.G, source=self.position, target=self.base_point)
            if self.path2warehouse:
                next_position = self.path2warehouse.pop(0)
                # Determine direction of the next step towards the warehouse
                curr_row, curr_col = self.position
                next_row, next_col = next_position
                if next_row < curr_row:
                    action = 0  # Up
                elif next_row > curr_row:
                    action = 2  # Down
                elif next_col > curr_col:
                    action = 1  # Right
                elif next_col < curr_col:
                    action = 3  # Left
                else:
                    action = 4  # Stay
        if self.status:
            action = int(action)
            if action ==4:
                self.current_traffic = 0
                self.prev_position = self.position
                self.move_time = self.cell_dist/self.max_speed
                self.local_time += self.move_time
                self.act_response = 1
            else:
                self.current_traffic = 0
                range_left = self.current_range - self.drain_rate * self.cell_dist
                if range_left<0:
                    self.current_traffic = 0
                    self.move_time = self.cell_dist/self.max_speed
                    self.local_time += self.move_time
                    self.act_response = 0
                else:
                    moves = {
                        0: (-1, 0),  # Up
                        1: (0, 1),   # Right
                        2: (1, 0),   # Down
                        3: (0, -1),  # Left
                    }
                        
                    dy, dx = moves[action]
                    self.prev_position = self.position
                    tmp_position = (self.position[0] + dy, self.position[1] + dx)
                    if tmp_position in self.G.nodes and (tmp_position, self.position) in self.G.edges:
                        self.position = tmp_position
                        self.distance_traveled += self.cell_dist
                        self.energy_consumed += self.drain_rate * self.cell_dist
                        traffic = graph.edges[(self.position, self.prev_position)]['traffic']
                        curr_speed = self.max_speed / (1 + self.traffic_delay_factor*traffic)  # Speed reduces with increasing traffic
                        self.move_time = self.cell_dist / curr_speed
                        if self.charging_status == 0:
                            self.current_range = range_left
                            self.current_range_percent = self.current_range / self.max_range
                        self.local_time += self.move_time

                        self.current_traffic = traffic
                        self.current_heading = action

                        self.act_response = 1
                    else:
                        self.current_traffic = 0
                        self.move_time = self.cell_dist/self.max_speed
                        self.local_time += self.move_time
                        self.act_response = 0

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