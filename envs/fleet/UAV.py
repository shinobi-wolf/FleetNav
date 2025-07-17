import numpy as np
import random

class UAV:
    def __init__(self, agent_id, base_position, cell_dist, max_range, max_speed, G):
        self.agent_id = agent_id
        self.base_station = np.array(base_position)  # Base station position
        self.position = np.array(base_position)
        self.offset = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])  # Random offset for actual position
        self.actual_position = base_position + self.offset
        self.cell_dist = cell_dist
        self.max_range = max_range
        self.max_speed = max_speed
        self.G = G

        self.status = 0
        self.current_range = max_range  # Start fully charged
        self.current_range_percent = 1.0
        self.local_time = 0.0
        self.active_task = None
        self.assigned_ugv = None

    def update(self, end_time):
        reached_flag = False

        dt = (end_time - self.local_time)/self.cell_dist
        if self.status == 0:
            self.actual_position = self.base_station + self.offset
        else:
            uav_pos = np.array(self.position)
            if self.status == 1:
                dest_pos = np.array(self.active_task)
            else:
                dest_pos = np.array(self.base_station)

            dx_ = np.linalg.norm(dest_pos - uav_pos)
            if dx_ > self.max_speed * dt:
                direction = (dest_pos - uav_pos) / dx_
                self.position = uav_pos + direction * self.max_speed * dt
                self.actual_position = self.position
            else:
                self.position = dest_pos
                if self.status == 1:
                    self.actual_position = self.position
                    reached_flag = True
                else:
                    self.actual_position = self.position + self.offset
                    self.status = 0
        self.local_time = end_time
        return reached_flag, self.assigned_ugv


    def __str__(self):
        return f"UAV {self.agent_id}: Position={self.position}, Actual Position={self.actual_position}, Status={self.status}, Local Time={self.local_time}"