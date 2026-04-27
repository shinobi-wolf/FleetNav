# Phase 2: Traffic-Responsive Shortest Path (TRSP)

## Goal

Develop a local navigation layer for ground robots executing Level-1 EVRP plans. Each robot must route itself from one network node to another using an RL policy trained entirely inside SUMO. Phase 2 targets a no-traffic baseline first; time-varying traffic (piecewise speed limits) is deferred to Phase 3.

## Problem

Given a road network (SUMO `.net.xml`) with fixed edge speeds:

- **State**: `(node_idx, target_idx)` — current junction index and goal junction index
- **Action**: integer in `[0, max_degree)` — index into the sorted list of neighbours at the current node
- **Reward**: time penalty + progress shaping ± terminal bonus (see below)
- **Episode**: single source→destination trip; truncates at `max_steps` or on arrival; resets with a new random (start, goal) pair within the same SUMO session

## Architecture

```
dronesim/trsp/
├── __init__.py              # gym env registration (SumoTRSPEnv-v0)
├── envs/
│   └── sumo_trsp_env.py     # SumoTRSPEnv(gym.Env) — SUMO-backed environment
├── fleet/
│   └── UGVSumo.py           # Ground robot state/movement model
├── agents/
│   ├── agent.py             # Abstract base
│   ├── dijkstra.py          # Shortest-path baseline (no learning)
│   ├── sarsa.py             # SARSA tabular Q-learning
│   ├── sarsa_l.py           # SARSA(λ) with eligibility traces
│   ├── dqn.py               # DQN (CleanRL-style, replay buffer)
│   └── ppo.py               # PPO (stub)
└── trainer.py               # lmd_trainer() — core training loop

train_trsp.py                # Runner: train a policy
eval_trsp.py                 # Runner: evaluate a saved policy over all (s,g) pairs
run_trsp_sim.py              # Runner: run a saved policy interactively for N episodes
config/trsp/train.yaml       # Training config
runs/trsp/<policy_id>/       # Saved policies, logs, training stats
```

## Environment

### Observation space

```python
spaces.Dict({
    'graph_node':  spaces.Discrete(num_nodes),   # current node index
    'target_node': spaces.Discrete(num_nodes),   # goal node index
    'steps2dest':  spaces.Box(shape=(max_degree,)),  # shortest-path dist per neighbour
    'best_dir':    spaces.Discrete(max_degree),  # argmin of steps2dest
})
```

SARSA uses only `(graph_node, target_node)` as its state key. The `steps2dest` and `best_dir` fields are available for Dijkstra and future deep agents.

### Action space

`spaces.Discrete(max_degree)` — index into the **sorted** (by node index) neighbour list at the current node. Actions beyond the actual degree are flagged as invalid.

### Reward

| Component | Value | Condition |
|---|---|---|
| Time penalty | `-1.0 × move_time_s` | Every valid step |
| Progress shaping | `+0.05 × Δdist_m` | Every valid step (`Δdist = dist_before − dist_after`) |
| Completion bonus | `+50.0` | Arrived at target junction |
| Truncation penalty | `−10.0` | `max_steps` exceeded without arrival |
| Invalid action | `−10.0` | Chosen neighbour index ≥ actual degree |

Progress shaping gives a positive signal when the agent moves toward the goal and a negative signal when it moves away, preventing the policy from getting trapped in cycles between tie-valued Q-entries.

### Episode flow

```
reset() → SUMO launch → warm-up → sample (start, goal) → return obs
step(a) → ugv_handler → task_handler → _get_reward → _check_termination → return obs, r, done, trunc, info
```

`task_handler` truncates and calls `_reset_episode()` on timeout or arrival; SUMO stays open across episodes. `_check_termination` fires only when the SUMO wall-clock budget (`max_time` hours) is exhausted.

### Render modes

| Mode | Behaviour |
|---|---|
| `quiet` | No output |
| `print` | One-line table per step: time, step, position, target, distance, reward |
| `human` | SUMO GUI (sumo-gui) + pygame overlay; waits for Return/numpad keypress between steps |

## Agents

| Agent | Type | State key | Notes |
|---|---|---|---|
| `DijkstraAgent` | Greedy baseline | `best_dir` obs field | Always picks the neighbour with smallest `steps2dest` |
| `SARSAAgent` | Tabular on-policy TD | `(node_idx, target_idx)` | ε-greedy, Q initialised with small uniform noise to break argmax ties |
| `SARSALambdaAgent` | Tabular + eligibility traces | `(node_idx, target_idx)` | Faster credit assignment |
| `DQNAgent` | Deep RL | Full obs dict (flattened) | MLP Q-network, replay buffer |
| `PPOAgent` | Deep RL | — | Stub — not yet implemented |

### SARSA details

- Q-table: `dict[(node_idx, target_idx)] → np.ndarray[max_degree]`
- Init: `np.random.uniform(-1e-5, 1e-5, max_degree)` — tiny noise prevents argmax always returning index 0 when all values are equal
- Action selection: slice Q to `[:num_neighbours]` before argmax/random — only valid actions are ever chosen
- Update rule: `Q(s,a) += α [r + γ Q(s',a') − Q(s,a)]` (terminal/truncation steps use `r` only)
- ε decays multiplicatively each episode

## Running

```bash
# Train
python train_trsp.py -c config/trsp/train.yaml

# Evaluate a saved policy (pairwise over all start/goal combinations)
python eval_trsp.py -p <policy_id>

# Run interactively (inference only)
python run_trsp_sim.py -p <policy_id> -n 10 -r print
```

Policies and logs are saved under `runs/trsp/<policy_id>/`:

| File | Contents |
|---|---|
| `policy.pkl` | Q-table (pickle) |
| `config.yaml` | Config snapshot at training time |
| `train_log.csv` | Per-episode reward and ε |
| `training_results.pkl` | State/SA visit counts, new-states-explored per episode |

## Known issues / Phase 3 TODO

- No traffic yet — edge speeds are fixed at `fleet.max_speed`. Phase 3 adds piecewise speed segments and restores `time_slot` to the state.
- `SARSALambdaAgent` and `DQNAgent` exist as skeletons; training loops not yet wired.
- `eval_trsp.py` pairwise evaluation is slow for large networks — batching planned.
