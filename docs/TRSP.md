# TRSP Module — Current State

The TRSP module is the **inner navigation layer**: given a source node and a destination node on the road network, route a ground robot through the graph in a traffic-aware manner using a trained RL policy. Each EVRP stop-to-stop leg is handled by a TRSP agent.

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `train_trsp.py` | Train a TRSP policy (currently: SARSA). Saves under `runs/trsp/<policy_id>/` |
| `eval_trsp.py -p <policy_id>` | Pairwise evaluation over all (start, goal) node pairs |
| `infer_trsp.py -p <policy_id>` | Interactive inference for N episodes |

---

## Architecture

```
dronesim/trsp/
├── __init__.py              # gym env registration (SumoTRSPEnv-v0)
├── envs/
│   └── sumo_trsp_env.py     # SumoTRSPEnv(gym.Env) — SUMO-backed RL environment
├── fleet/
│   └── UGVSumo.py           # Ground robot movement/state model
├── traffic.py               # PSLTrafficModel — per-edge piecewise speed profiles
├── agents/
│   ├── __init__.py          # get_agent() registry
│   ├── dijkstra.py          # Greedy shortest-path baseline
│   ├── sarsa.py             # SARSA tabular Q-learning ✅
│   ├── sarsa_l.py           # SARSA(λ) with eligibility traces (skeleton)
│   ├── dqn.py               # DQN (CleanRL-style replay buffer) (skeleton)
│   └── ppo.py               # PPO (stub)
└── trainer.py               # lmd_trainer() — training loop

config/trsp/train.yaml       # Training config
runs/trsp/<policy_id>/       # Saved policies and logs
```

---

## Environment: `SumoTRSPEnv-v0`

Registered in `dronesim/trsp/__init__.py`. The environment wraps a SUMO simulation; one SUMO process runs for the lifetime of the environment (across episodes). SUMO is only started on `env.reset()`, not on `gym.make()`.

### Graph

Built at `__init__` time from the SUMO net file via `sumolib` (no SUMO process needed). Stored as:
- `self.G_dir` — `nx.DiGraph` (directed edges)
- `self.G` — `G_dir.to_undirected()` (used for neighbor enumeration)
- `self.node2idx / idx2node` — sorted node index maps
- `self.junc2idx / idx2junc / junction_list` — junction index maps (1-to-1 with nodes for simple nets)
- `self.all_shortest_path_lengths` — all-pairs directed SPL; used for reward shaping and `best_dir`
- `self.edge_ids` — all non-internal SUMO edge IDs

### Observation Space

```python
spaces.Dict({
    'graph_node':  Discrete(num_nodes),   # current node index
    'target_node': Discrete(num_nodes),   # goal node index
    'time_slot':   Discrete(K),           # current PSL slot index
    'best_dir':    Discrete(max_degree),  # argmin of steps-to-dest per neighbour
})
```

`time_slot` is always 0 when `traffic.K = 1` (constant per-edge speed).

### Action Space

`Discrete(max_degree)` — index into the **sorted** (by node index) neighbour list at the current node. Actions beyond the actual degree of the current node are flagged as invalid.

### Reward

| Component | Value | Condition |
|-----------|-------|-----------|
| Time penalty | `−1.0 × move_time_s` | Every valid step |
| Progress shaping | `+0.05 × Δdist_m` | `Δdist = directed_SPL_before − directed_SPL_after` |
| Completion bonus | `+50.0` | Arrived at target junction |
| Truncation penalty | `−10.0` | `max_steps` exceeded without arrival |
| Invalid action | `−10.0` | Chosen neighbour index ≥ actual degree |

### Episode Flow

```
env.reset() → start SUMO → warm-up steps → sample (start, goal) pair → return obs
env.step(a) → ugv_handler (move vehicle) → task_handler (check arrival/timeout)
            → compute reward → return obs, r, done, trunc, info
```

`task_handler` truncates and calls `_reset_episode()` on timeout or arrival — SUMO stays open across episodes within one environment. `_check_termination` fires only when the wall-clock budget (`max_time` hours) is exhausted.

### Render Modes

| Mode | Behaviour |
|------|-----------|
| `quiet` | No output |
| `print` | One-line table per step: time, slot, step, position, target, distance, reward |
| `human` | SUMO GUI + pygame overlay with edge coloring; waits for Return/numpad keypress |

---

## Traffic Model: `PSLTrafficModel`

**File:** `dronesim/trsp/traffic.py`

Per-edge piecewise speed profiles — each edge gets its own independent speed for each time slot. The TA-EVRP simulators now use the same PSL approach (`dronesim/evrp/traffic/segmented.py`) with a unified config key schema.

### Config Keys

```yaml
traffic:
  enabled: true
  seed: 42
  T: 7200.0              # total horizon in seconds
  K: 1                   # number of time slots
  v_min: 40.0            # km/h — minimum speed level
  v_max: 100.0           # km/h — maximum speed level
  speed_step: 60.0       # km/h — step between levels
  speed_profile_type: random  # random | function | file
```

With `K=1` every edge has a single fixed speed for the entire simulation (set at construction via seeded RNG). The slot index is always 0.

### Speed Profile Generation

- `random`: each edge independently draws from a discrete set of speed levels spanning `[v_min, v_max]` in increments of `speed_step` (inclusive of `v_max` when it lands exactly). Same seed + same net file → identical profiles across training, eval, and inference.
- `function`: `fn(edge_id, slot_idx) → km/h` callable.
- `file`: JSON/YAML mapping `edge_id → [v_slot0, v_slot1, ...]` in km/h.

Internally, the model stores and returns speeds in **m/s**; config values are in km/h.

### Usage in the Env

At each SUMO step, the env calls `traffic.get_slot(sim_time_s)` to get the active slot. When the slot changes, the env updates the active vehicle’s speed via `traci.vehicle.setSpeed()` using the per-edge speed for the current road.

### Edge Coloring (human render mode)

Edges are drawn in the pygame overlay with a red→green gradient based on their speed in the current slot: `v_min → red (255, 0, 0)`, `v_max → green (0, 255, 0)`. Opposing directions are drawn with a 3px perpendicular offset so both colors are visible on two-way roads.

---

## Agents

### `DijkstraAgent` ✅ Done

Always picks the neighbour with the smallest `steps2dest` (greedy shortest path). Used as the baseline. No learning.

### `SARSAAgent` ✅ Done

- **State key:** `(graph_node, target_node, time_slot)` — 3-tuple
- **Q-table:** `dict[state → np.ndarray[max_degree]]`. Initialised with `uniform(-1e-5, 1e-5)` noise to prevent argmax always returning index 0.
- **Action selection:** slice Q to `[:num_neighbours]` — only valid neighbour indices are considered.
- **Update:** `Q(s,a) += α [r + γ Q(s',a') − Q(s,a)]`; terminal/truncation steps use `r` only (no bootstrap).
- **ε-greedy** with multiplicative decay each episode.
- Policies saved as `policy.pkl` (Python pickle of the Q-table dict).

Policy-loading safeguard:
- When `agent.policy` is set (loading an existing `policy.pkl`), the agent runs a post-pass that detects greedy 2-cycles (A→B→A for the same target/slot) and breaks them by boosting a Dijkstra-consistent alternative action.

### `SARSALambdaAgent` ⚠️ Partial

Eligibility traces support added to file structure; training loop not yet wired.

### `DQNAgent` ⚠️ Experimental

CleanRL-style MLP Q-network with replay buffer. File exists; not fully integrated.

### `PPOAgent` ⚠️ Experimental / incomplete

Not yet implemented.

---

## Training Config (`config/trsp/train.yaml`)

```yaml
net:
  net_path: data/grid_road/grid_road.sumocfg
  max_degree: 4

fleet:
  max_speed: 110      # km/h

traffic:
  enabled: true
  K: 1                # constant per-edge speed (single slot)
  T: 7200.0
  v_min: 40.0         # km/h
  v_max: 100.0        # km/h
  speed_step: 60.0    # gives exactly two speed levels: {40, 100} km/h
  seed: 42
  speed_profile_type: random

simulation:
  max_steps: 50
  random_sg_pair: true

agent:
  algo: sarsa

training:
  sarsa:
    num_episodes: 5000
    gamma: 0.99
    alpha: 0.1
    epsilon: 0.9
    epsilon_decay: 0.9995
```

---

## Evaluation (`eval_trsp.py`)

Iterates over all (start, goal) node pairs. For each pair:
1. Plans a route using the agent's `predict()` (graph-only, no SUMO) — same observation building as training.
2. Starts a SUMO instance with a configurable warm-up time.
3. Spawns a SUMO vehicle on the planned edge list and steps until arrival.
4. Records travel time, path length, traffic slot at start/end, and edge speeds.

Results written to `runs/trsp/<policy_id>/test_<N>/simulation_results.csv`.

Current limitation: the *planning* phase in `eval_trsp.py` uses `time_slot = get_slot(0.0)` (i.e., slot 0) rather than a simulated clock. The *SUMO execution* phase uses the real `sim_time` slot when overriding speeds.

Traffic consistency across training/eval/inference: `PSLTrafficModel` uses an isolated `random.Random(seed)` — never affects the global RNG — so the same seed + same net file produces identical per-edge profiles everywhere.

---

## Saved Policy Structure

```
runs/trsp/<policy_id>/
├── config.yaml           # Config snapshot at training time
├── policy.pkl            # Q-table (SARSA) or network weights (DQN)
├── train_log.csv         # Per-episode: total reward, epsilon
└── training_results.pkl  # State/SA visit counts, new-states-explored per episode
    test_<N>/
    ├── config.yaml
    └── simulation_results.csv

Note: only SARSA training is currently wired in `train_trsp.py` (it always instantiates `SARSAAgent`). DQN/PPO code exists, but training those policies requires additional runner wiring.
```

---

## Known Issues / Future Work

- `SARSALambdaAgent` and `DQNAgent` training loops not yet wired.
- `PPOAgent` is a stub.
- Pairwise evaluation in `eval_trsp.py` is O(N²) SUMO starts — slow for large networks.
- `K > 1` training has not been tested end-to-end; state space grows by factor K.
- TRSP's `PSLTrafficModel` and EVRP's `PSLTrafficModel` are independent instances sharing the same config format. Speed values are sampled from the same discrete levels but with separate RNG instances — they will agree only if given identical seeds and the same edge set. Config key names differ: TRSP uses `v_min`/`v_max`/`speed_step` (km/h), EVRP uses `v_min_kmh`/`v_max_kmh`/`speed_step_kmh`.

Runner limitation:
- `infer_trsp.py` currently treats an episode as done only when `terminated` (wall-clock `max_time`) becomes true; it does not break on `truncated` even though truncation is how the env ends “one routing task”. Use `simulation.max_time` to bound runs, or adjust the runner if you want one route per `env.reset()`.
