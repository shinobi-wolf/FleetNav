#!/usr/bin/env python3
# TraCI-only edge monitor. Connects to an already running SUMO (started with --remote-port).
# Reports: per-step veh count, mean speed, occupancy; plus cumulative entrants and flow (veh/h).

import argparse
import time
from statistics import mean

import traci

def avg_edge_occupancy(edge_id: str) -> float:
    """Return average occupancy (0..1) across all lanes of edge."""
    lane_ids = traci.edge.getLaneIDs(edge_id)
    if not lane_ids:
        return 0.0
    vals = [traci.lane.getLastStepOccupancy(l) for l in lane_ids]  # fraction 0..1
    return sum(vals) / len(vals)

def main():
    ap = argparse.ArgumentParser(description="Monitor traffic on a SUMO edge via TraCI.")
    ap.add_argument("--lane", required=True, help="Target lane ID (not an internal ':' edge)")
    ap.add_argument("--duration", type=int, default=1000, help="Seconds to monitor")
    ap.add_argument("--print-every", type=int, default=1, help="Print interval (steps)")
    args = ap.parse_args()

    Sumo_config = [
    'sumo-gui',
    '-c', 'data/grid/grid.sumocfg',
    '--step-length', '0.05',
    '--delay', '100',
    '--lateral-resolution', '0.1'
    ]

    # Step 5: Open connection between SUMO and Traci
    traci.start(Sumo_config)

    ll = traci.lane.getIDList()
    print("Available lanes:", ll)

    # Validate lane
    if args.lane not in traci.lane.getIDList():
        raise SystemExit(f"Lane '{args.lane}' not found in network.")

    step = 0

    # Optional: read step length to scale results correctly
    # (you set it with --step-length when starting SUMO)
    # We’ll assume 1s if not available.
    try:
        step_length = traci.simulation.getDeltaT() / 1000.0  # ms -> s
    except Exception:
        step_length = 1.0

    print(f"Monitoring lane '{args.lane}' for {args.duration}s (step={step_length}s) ...")

    # Advance until we cover the desired window
    end_time = traci.simulation.getTime() + args.duration
    while traci.simulation.getTime() < end_time or traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        avg_trav_time = traci.lane.getTraveltime(args.lane)

        if (step % args.print_every) == 0:
            print(
                f"[t={int(traci.simulation.getTime()):4d}] "
                f"average travel time on lane '{args.lane}': {avg_trav_time:.2f} s"
            )
        step += 1

        # If you started SUMO with a very small --step-length, this small sleep
        # makes the log readable; otherwise you can remove it.
        # time.sleep(0.0)

    print("\n=== Summary ===")
    print(f"Lane:         {args.lane}")
    print(f"Duration:     {args.duration} s")
    print(f"Step length:  {step_length} s")

    traci.close()

if __name__ == "__main__":
    main()
