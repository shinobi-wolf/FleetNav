import time
import yaml
# from console import startup
from server import startup

if __name__ == '__main__':
    # startup()
    with open("results_evrp_bms.csv", "w") as f:
        f.write("num_evs,num_tasks,ev_range,maze_loop_percentage,maze_size,bms_enabled,final_frame,num_tasks_completed,ev_distance_traveled,drone_distance_traveled\n")
    with open("config/world1.yaml", "r") as f:
        config = yaml.safe_load(f)
    startup(config, output_file="results_evrp_bms.csv")
