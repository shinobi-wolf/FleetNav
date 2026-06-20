# TRSP Module — Current State

The TRSP module is the **inner navigation layer**: given a source node and a destination node on the road network, route a ground robot through the graph in a traffic-aware manner using a trained RL policy. Each EVRP stop-to-stop leg is handled by a TRSP agent.

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `train_trsp.py -c <config>` | Train a TRSP policy. Saves under `runs/trsp/<policy_id>/` |
| `eval_trsp2.py -p <policy_id> [--dijkstra]` | Sequential evaluation over 100 OD pairs within a single SUMO session (K>1 compatible). Optionally runs Dijkstra baseline on the same pairs. |
| `eval_trsp.py -p <policy_id>` | Legacy pairwise evaluation (O(N²) SUMO starts; use `eval_trsp2.py` instead) |
| `infer_trsp.py -p <policy_id>` | Interactive inference for N episodes |
| `train_a2c_all.sh` | Train all 6 A2C (network × traffic) configs sequentially and eval each |

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
│   ├── __init__.py          # get_agent() registry; dispatches 'dqn_svc' to DQNSVCAgent
│   ├── agent_psl.py         # BasePSLAgent — shared interface for PSL agents
│   ├── dijkstra_psl.py      # Greedy shortest-path baseline ✅
│   ├── sarsa_psl.py         # SARSA tabular Q-learning ✅
│   ├── sarsa_l_psl.py       # SARSA(λ) with eligibility traces (skeleton)
│   ├── dqn_psl.py           # Deep Q-Network — PSL traffic ✅
│   ├── ppo_psl.py           # Proximal Policy Optimization — PSL traffic ✅
│   ├── a2c_psl.py           # Advantage Actor-Critic — PSL traffic ✅
│   └── dqn_svc.py           # Deep Q-Network — SVC traffic ✅
└── trainer.py               # lmd_trainer() — training loop

config/trsp/
├── train.yaml               # Default PSL training config (editable algo/net/traffic)
├── train_svc_dqn.yaml       # SVC DQN training config — grid_road template
├── a2c_grid_road_K1.yaml    # A2C — grid_road, K=1
├── a2c_grid_road_K2.yaml    # A2C — grid_road, K=2
├── a2c_rand_grid_K1.yaml    # A2C — rand_grid_road, K=1
├── a2c_rand_grid_K2.yaml    # A2C — rand_grid_road, K=2
├── a2c_ames_K1.yaml         # A2C — ames, K=1
└── a2c_ames_K2.yaml         # A2C — ames, K=2

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

Mode depends on `traffic.model` and `agent.action_mode`:

**PSL traffic, `node_id` mode** (DQN-PSL, PPO, A2C):
```python
spaces.Dict({
    'graph_node':  Discrete(num_nodes),   # current node index
    'target_node': Discrete(num_nodes),   # goal node index
    'time_slot':   Discrete(K),           # current PSL slot index
})
```

**PSL traffic, `neighbor_index` mode** (SARSA, Dijkstra):
```python
spaces.Dict({
    'graph_node':  Discrete(num_nodes),
    'target_node': Discrete(num_nodes),
    'time_slot':   Discrete(K),
    'best_dir':    Discrete(max_degree),  # argmin of steps-to-dest per neighbour
})
```

**SVC traffic, `node_id` mode** (DQN-SVC):
```python
spaces.Dict({
    'graph_node':      Discrete(num_nodes),
    'target_node':     Discrete(num_nodes),
    'neighbor_speeds': Box(0.0, 2.0, shape=(max_degree,)),  # real-time TraCI speeds
})
```

`neighbor_speeds[i]` = `traci.edge.getLastStepMeanSpeed(edge_i) / ugv_speed`, clamped to [0, 2]. Padded to `max_degree` with 1.0 (free-flow) for nodes with fewer neighbours. Missing-vehicle edges (TraCI returns -1) are also treated as free-flow = 1.0.

`time_slot` is always 0 when `traffic.K = 1` (constant per-edge speed).

### Action Space

- **`node_id`** (deep RL): `Discrete(num_nodes)` — any node, masked to valid neighbours at inference.
- **`neighbor_index`** (tabular): `Discrete(max_degree)` — index into sorted neighbour list. Actions beyond actual node degree incur the invalid-action penalty.

### Reward

| Component | Value | Condition |
|-----------|-------|-----------|
| Time penalty | `−1.0 × move_time_s` | Every valid step |
| Progress shaping | `+0.05 × Δdist_m` | `Δdist = directed_SPL_before − directed_SPL_after` |
| Completion bonus | `+50.0` | Arrived at target junction |
| Truncation penalty | `−10.0` | `max_steps` exceeded without arrival |
| Invalid action | `−10.0` | Chosen action is not a valid neighbour |

### Episode Flow

```
env.reset() → start SUMO → warm-up steps → sample (start, goal) pair → return obs
env.step(a) → ugv_handler (move vehicle) → task_handler (check arrival/timeout)
            → compute reward → return obs, r, terminated, truncated, info
```

- `truncated=True`: target reached OR `max_steps` exceeded. Env calls `_reset_episode()` internally; SUMO stays open across episodes.
- `terminated=True`: wall-clock budget (`max_time` hours) exhausted. Signals agents to call `env.close()` + `env.reset()`.

### Render Modes

| Mode | Behaviour |
|------|-----------|
| `quiet` | No output |
| `print` | One-line table per step: time, slot, step, position, target, distance, reward |
| `human` | SUMO GUI + pygame overlay with edge coloring; waits for Return/numpad keypress |

---

## Traffic Models

### PSL — `PSLTrafficModel`

**File:** `dronesim/trsp/traffic.py`

Per-edge piecewise speed profiles — each edge independently draws a speed for each time slot.

```yaml
traffic:
  model: psl               # default when 'model' key absent
  enabled: true
  seed: 42
  T: 7200.0                # total horizon in seconds
  K: 1                     # number of time slots (1–4 tested)
  v_min: 40.0              # km/h
  v_max: 100.0             # km/h
  speed_step: 60.0         # km/h — step between levels
  speed_profile_type: random
```

With `K=1` every edge has a single fixed speed for the entire simulation. With `K=2` and `speed_step=60.0`, exactly two speed levels exist: `{40, 100}` km/h.

### SVC — Stochastic Vehicular Congestion

**Mechanism:** Real SUMO background vehicles pre-generated with `randomTrips.py` + `duarouter`. Injected into the same simulation as the UGV via `--route-files`. Congestion is emergent — not scripted per-edge — so the traffic pattern varies with time in a naturalistic way.

```yaml
traffic:
  model: svc
  route_file: data/<network>/traffic_routes/<tag>/rvt_routes.xml

sumo:
  traffic_scale: 1         # enable background vehicle insertion
simulation:
  warm_up_time: 60         # seconds before sampling OD pairs (traffic builds up)
```

The route file is appended to the SUMO command via `--route-files` on each `env.reset()`. The PSL model is disabled when `model: svc`.

**Congestion parameters (generated files):**

| Network | Route file | Period | Vehicles | Notes |
|---------|------------|--------|----------|-------|
| `grid_road` | `traffic_routes/svc_123/rvt_routes.xml` | 1.5s | 13,334 | Dense — ~2 vehicles/edge |
| `rand_grid_road` | `traffic_routes/svc_42/rvt_routes.xml` | 3.0s | 6,667 | Moderate — ~0.14 avg, hotspots via fringe-factor 5 |
| `ames` | `traffic_routes/svc_42/rvt_routes.xml` | 5.0s | 4,000 | Light — ~0.27 avg; campus bottlenecks |

All route files generated with `seed=42`, `--fringe-factor 5`, `--validate`; see `params.yaml` in each directory for the exact reproduction command.

---

## Agents

### `DijkstraAgent` ✅ Done (`dijkstra_psl.py`)

Always picks the neighbour with the smallest `steps2dest` (greedy shortest path). Used as the baseline. No learning. Works for both PSL and SVC evaluation since it ignores traffic state.

### `SARSAAgent` ✅ Done (`sarsa_psl.py`)

- **State key:** `(graph_node, target_node, time_slot)` — 3-tuple
- **Q-table:** `dict[state → np.ndarray[max_degree]]`. Initialised with `uniform(-1e-5, 1e-5)` noise to break argmax ties.
- **Action selection:** slice Q to `[:num_neighbours]` — only valid neighbour indices are considered.
- **Update:** `Q(s,a) += α [r + γ Q(s',a') − Q(s,a)]`; terminal/truncation steps use `r` only.
- **ε-greedy** with multiplicative decay each episode.
- Policies saved as `policy.pkl` (pickle of the Q-table dict).
- Post-load 2-cycle detection: detects greedy A→B→A loops and breaks them via a Dijkstra-consistent nudge.

### `SARSALambdaAgent` ⚠️ Partial (`sarsa_l_psl.py`)

Eligibility traces support added to file structure; training loop not yet wired.

### `DQNAgent` ✅ Done (`dqn_psl.py`) — PSL traffic

CleanRL-style MLP Q-network with experience replay.

- **Observation encoding:** one-hot(graph_node) ++ one-hot(target_node) ++ one-hot(time_slot); `obs_dim = 2·num_nodes + K`
- **Network:** 3-layer MLP (ReLU, hidden=64)
- **Training:** ε-greedy, replay buffer (50K), target network (sync every 500 steps), TD(0) MSE loss

### `DQNSVCAgent` ✅ Done (`dqn_svc.py`) — SVC traffic

DQN variant for the SVC traffic model. Replaces the global `time_slot` with real-time local traffic observations.

- **Observation encoding:** one-hot(graph_node) ++ one-hot(target_node) ++ `neighbor_speeds[max_degree]`; `obs_dim = 2·num_nodes + max_degree` (e.g. 55 for grid_road with max_degree=5)
- `neighbor_speeds` queried live from `traci.edge.getLastStepMeanSpeed()` at each step, normalized by `ugv_speed`, sorted by `node2idx` to match action ordering
- **Network:** 3-layer MLP (ReLU, hidden=128) — wider than PSL DQN to handle the larger obs vector
- **Hardware:** CUDA-aware; moves networks and tensors to GPU if available
- **Training:** same ε-greedy / replay / target-net scheme as DQN-PSL
- **Policy pickle keys:** `state_dict`, `num_nodes`, `max_degree`, `obs_dim`, `hidden_dim`

### `PPOAgent` ✅ Done (`ppo_psl.py`)

Single-env CleanRL-style PPO with GAE advantage estimation.

- **Observation encoding:** same as DQN-PSL — `obs_dim = 2·num_nodes + K`
- **Network:** separate actor and critic, 2-layer Tanh MLP (hidden=64), orthogonal init
- **Training:** `num_steps`-step rollouts, GAE (λ=0.95), `update_epochs` epochs of minibatch PPO (clipped ratio + clipped value loss + entropy bonus)

### `A2CAgent` ✅ Done (`a2c_psl.py`)

Simpler sibling of PPO — same network, same rollout, same GAE — but a single unclipped gradient step per rollout.

- **Policy loss:** plain policy gradient `−log_prob · advantage`
- **Value loss:** unclipped `0.5 · MSE(value, return)`
- **No clipping, no minibatches, no importance-sampling**

---

## Training Config

Default PSL: `config/trsp/train.yaml` — edit `agent.algo` and `net.net_path` / `traffic.K`.
SVC template: `config/trsp/train_svc_dqn.yaml` — copy and update `net.net_path` and `traffic.route_file` for each network.

### Key Hyperparameters

| Param | SARSA | DQN-PSL | DQN-SVC | PPO | A2C |
|-------|-------|---------|---------|-----|-----|
| Episodes / Iterations | 5000 ep | 5000 ep | 5000 ep | 500 iter | 500 iter |
| Steps per rollout | — | — | — | 512 | 512 |
| Learning rate | 0.1 (α) | 1e-3 | 1e-3 | 2.5e-4 | 7e-4 |
| Hidden dim | — | 64 | 128 | 64 | 64 |
| Gamma | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| Warm-up time | — | — | 60s | — | — |

---

## Evaluation: `eval_trsp2.py`

Runs 100 OD pairs sequentially within a **single continuous SUMO session** — critical for K>1 policies since traffic slots evolve naturally across pairs, and for SVC since background vehicles persist between evaluations.

1. Builds graph from the saved policy's `config.yaml`.
2. Detects `traffic.model` (`psl` or `svc`) and configures the env accordingly.
3. For SVC: appends `--route-files` to the SUMO command; queries `traci.edge.getLastStepMeanSpeed()` for obs.
4. Plans each route using the agent's `predict()` at current sim state.
5. Injects a vehicle on the planned edge list, steps SUMO until arrival or timeout.
6. Writes `runs/trsp/<policy_id>/test_<N>/simulation_results.csv`.

With `--dijkstra`: reruns the same 100 pairs with `DijkstraAgent` on the live SVC/PSL session and saves results to `runs/trsp/dijkstra/test_<N>/`.

---

## Saved Policy Structure

```
runs/trsp/<policy_id>/
├── config.yaml           # Config snapshot (with action_mode and agent.policy set)
├── policy.pkl            # Weights: Q-table (SARSA) | state_dict dict (DQN/PPO/A2C)
├── train_log.csv         # Per-episode reward + epsilon (SARSA/DQN) or
│                         #   per-iteration pg_loss/v_loss/mean_ep_ret (A2C/PPO)
└── test_<N>/
    ├── config.yaml
    └── simulation_results.csv
```

**Policy pickle schema:**

| Agent | Keys in `policy.pkl` |
|-------|----------------------|
| SARSA | Q-table dict |
| DQN-PSL | `state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |
| DQN-SVC | `state_dict`, `num_nodes`, `max_degree`, `obs_dim`, `hidden_dim` |
| PPO | `actor_state_dict`, `critic_state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |
| A2C | `actor_state_dict`, `critic_state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |

---

## Networks Tested

| ID | Directory | Nodes | Edges | Notes |
|----|-----------|-------|-------|-------|
| N1 | `data/grid_road/` | ~25 | ~40 | Regular grid |
| N2 | `data/rand_grid_road/` | 26 | 39 | Randomized grid |
| N3 | `data/ames/` | ~45 | ~80 | Ames campus road network |

All networks: `max_degree: 5`.

---

## Results Summary

### PSL Traffic (all networks, K=1–4)

DQN, SARSA, and Dijkstra achieve near-identical success rates (~98–100%) on N1/N2/N3 with K=1. PPO and A2C converge to faster mean traversal times but at significantly lower success rates (8–30%), indicating incomplete policy convergence. SARSA and DQN consistently match or beat Dijkstra on mean makespan.

### SVC Traffic (DQN-SVC, all three networks)

| Network | Traffic | Policy | Mean time | vs Dijkstra | Success |
|---------|---------|--------|-----------|-------------|---------|
| N1 (grid_road) | SVC-svc_123 | `b3ef6da6` | 253.6s | −33% ✓ | 96% |
| N2 (rand_grid_road) | SVC-svc_42 | `2553013e` | 59.7s | −27% ✓ | 75% |
| N3 (ames) | SVC-svc_42 | `964ad609` | 146.1s | −3.6% | 95% |

Dijkstra baselines: N1=378s (94%), N2=82s (99%), N3=152s (98%).

DQN-SVC outperforms Dijkstra on mean traversal time on all three networks. The N2 success-rate drop (75% vs 99%) suggests the policy sometimes fails to reach the target within `max_steps` — likely due to the short edges (mean 152m) requiring more graph steps per route. N3 (Ames) shows a near-tie in mean time; the real-world campus topology is less structured, making traffic-avoidance harder to exploit.

---

## Known Issues / Future Work

- `SARSALambdaAgent` training loop not yet wired.
- DQN-SVC N2 success rate (75%) is noticeably below Dijkstra (99%) — consider longer training or a higher `max_steps` limit for the short-edge rand_grid_road network.
- `infer_trsp.py` treats an episode as done only when `terminated` (wall-clock) fires; it does not break on `truncated`. Use `simulation.max_time` to bound runs, or adjust the runner for per-route inference.
- SVC policies (DQN-SVC) are not yet supported by `LocalPlanner` in the TA-EVRP integration — `plan()` is graph-only and cannot call TraCI for `neighbor_speeds`. PSL agents (SARSA, DQN-PSL) are the current choice for end-to-end TA-EVRP runs.
- TRSP's `PSLTrafficModel` and EVRP's `PSLTrafficModel` are independent instances. Config key names differ: TRSP uses `v_min`/`v_max`/`speed_step` (km/h), EVRP uses `v_min_kmh`/`v_max_kmh`/`speed_step_kmh`.
