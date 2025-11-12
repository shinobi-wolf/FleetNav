# SUMO Simulation Visualization

## Overview

The `visualize_sumo_simulation.py` script replays a simulation from a server log file. It:

1. Parses the server log to extract UGV trajectories
2. Loads the network configuration from the config file
3. Builds a graph representation of the SUMO network
4. Converts node-based paths to edge-based routes
5. Creates SUMO vehicles and routes
6. Runs the simulation with visualization

## Requirements

- SUMO with GUI support (`sumo-gui`)
- Python packages: `traci`, `networkx`, `pyyaml`

## Usage

### Basic Usage

```bash
python visualize_sumo_simulation.py <policy_name>
```

### Example

To visualize the `dijkstra_sumo` policy:

```bash
python visualize_sumo_simulation.py dijkstra_sumo
```

This will:
- Read from `runs/dijkstra_sumo/server_log.txt`
- Load configuration from `runs/dijkstra_sumo/config.yaml`
- Launch SUMO GUI with the replay

### Directory Structure

The script expects the following structure:

```
runs/
  <policy_name>/
    server_log.txt    # Required: Contains UGV trajectories
    config.yaml       # Required: Contains network and simulation config
```

## Server Log Format

The server log should have the following format:

```
Config: runs/dijkstra_sumo//config.yaml
Step 0:
UGV 0: Position=E0, Active Task=B0, Status=1
Step 1:
UGV 0: Position=D0, Active Task=B0, Status=1
...
```

- **Config line**: Path to the configuration file
- **Step N**: Timestep number
- **UGV ID**: Vehicle ID and its state
  - Position: Junction/node in the SUMO network
  - Active Task: Target location (or None)
  - Status: 0=idle, 1=moving, 2=completed

## Visualization Features

- **Green circle**: Warehouse/base station
- **Red circles**: Task locations
- **Blue vehicles**: UGVs following their logged routes
- **Gray lines**: Road network

## Output

The script prints:
- Network statistics (nodes, edges)
- UGV route information
- Simulation progress
- Completion status for each vehicle

## Troubleshooting

### "Server log not found"
Make sure the policy name matches a directory in `runs/` with a `server_log.txt` file.

### "Could not find config path"
Ensure `config.yaml` exists in the policy's run directory.

### "No path found from X to Y"
The logged trajectory may contain jumps between non-adjacent nodes. The script attempts to find shortest paths to bridge these gaps.

### SUMO doesn't start
- Ensure `sumo-gui` is installed and in your PATH
- Check that the network file specified in config.yaml exists
- Verify SUMO installation: `sumo-gui --version`

## Advanced Usage

You can modify the script to:
- Use `sumo` instead of `sumo-gui` for headless mode (line 202)
- Adjust visualization delay with `--delay` parameter
- Change vehicle colors or sizes in the `draw_circle` function
- Add additional visualization elements (routes, paths, etc.)

## Example Run

```bash
$ python visualize_sumo_simulation.py dijkstra_sumo
============================================================
SUMO Simulation Visualizer
============================================================
Policy: dijkstra_sumo

Reading server log from: runs/dijkstra_sumo/server_log.txt
Loading config from: runs/dijkstra_sumo/config.yaml
Network path: data/grid/grid.sumocfg
Found 1 UGVs in log

Starting SUMO...
Building network graph...
Graph: 25 nodes, 80 edges

Extracting UGV routes...
UGV 0: 6 nodes -> 5 edges
  Node path: E0 -> D0 -> C0 -> B0 -> B1 -> B2 -> B3 -> C3 -> D3

Visualizing 1 task locations...
Warehouse node: E0
...
```
