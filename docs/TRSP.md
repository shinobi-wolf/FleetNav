# TRSP Module — Current State

The TRSP module is the **inner navigation layer**: given a source node and a destination node on the road network, route a ground robot through the graph in a traffic-aware manner using a trained RL policy. Each EVRP stop-to-stop leg is handled by a TRSP agent.

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `train_trsp.py -c <config>` | Train a TRSP policy. Saves under `runs/trsp/<policy_id>/` |
| `eval_trsp2.py -p <policy_id>` | Sequential evaluation over 100 OD pairs within a single SUMO session (K>1 compatible). Always runs Dijkstra and DSP baselines automatically after the policy eval; saves all three CSVs co-located in `test_N/`. |
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
│   ├── __init__.py          # get_agent() registry
│   ├── agent_psl.py         # BasePSLAgent — shared interface for PSL agents
│   ├── dijkstra_psl.py      # Greedy shortest-path baseline ✅
│   ├── dsp.py               # Dynamic Shortest Path — live-speed Dijkstra ✅
│   ├── sarsa_psl.py         # SARSA tabular Q-learning ✅
│   ├── sarsa_l_psl.py       # SARSA(λ) with eligibility traces (skeleton)
│   ├── dqn_psl.py           # Deep Q-Network — PSL traffic ✅
│   ├── ppo_psl.py           # Proximal Policy Optimization — PSL traffic ✅
│   ├── a2c_psl.py           # Advantage Actor-Critic — PSL traffic ✅
│   ├── dqn_svc.py           # Deep Q-Network — SVC traffic ✅
│   ├── sarsa_svc.py         # SARSA tabular Q-learning — SVC traffic ✅
│   ├── a2c_svc.py           # Advantage Actor-Critic — SVC traffic ✅
│   └── ppo_svc.py           # Proximal Policy Optimization — SVC traffic ✅
└── trainer.py               # lmd_trainer() — training loop

config/trsp/
├── train.yaml               # Default PSL training config (editable algo/net/traffic)
├── train_svc_dqn.yaml       # SVC DQN training config — grid_road template
├── a2c_grid_road_K1.yaml    # A2C — grid_road, K=1
├── a2c_grid_road_K2.yaml    # A2C — grid_road, K=2
├── a2c_rand_grid_K1.yaml    # A2C — rand_grid_road, K=1
├── a2c_rand_grid_K2.yaml    # A2C — rand_grid_road, K=2
├── a2c_ames_K1.yaml         # A2C — ames, K=1
├── a2c_ames_K2.yaml         # A2C — ames, K=2
├── svc/                     # SVC training configs — one per algo × network
│   ├── sarsa_grid_road.yaml
│   ├── sarsa_rand_grid.yaml
│   ├── sarsa_ames.yaml
│   ├── dqn_grid_road.yaml   # (train_svc_dqn.yaml is the grid_road template)
│   ├── a2c_grid_road.yaml
│   ├── a2c_rand_grid.yaml
│   ├── a2c_ames.yaml
│   ├── ppo_grid_road.yaml
│   ├── ppo_rand_grid.yaml
│   └── ppo_ames.yaml
└── td/                      # TD training configs — one per algo × network
    ├── sarsa_grid_road.yaml
    ├── sarsa_rand_grid.yaml
    ├── sarsa_ames.yaml
    ├── dqn_ames.yaml
    ├── dqn_grid_road.yaml
    ├── a2c_ames.yaml
    ├── a2c_grid_road.yaml
    ├── a2c_rand_grid.yaml
    ├── ppo_ames.yaml
    ├── ppo_grid_road.yaml
    └── ppo_rand_grid.yaml

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

### TD — `TDTrafficModel` (3-period congestion)

**File:** `dronesim/trsp/traffic.py` (subclass of `PSLTrafficModel`)

Three-period time-of-day congestion model (Lu et al. 2020): `K=3` with a **global** congestion speed `v_c` in slots 0 & 2 and a free-flow speed `v_f` in slot 1 (every edge gets the identical profile `[v_c, v_f, v_c]`). Only `get_slot()` differs from PSL — the active slot is `0` for `t<T1`, `1` for `T1≤t<T2`, else `2`. `time_slot ∈ {0,1,2}` enters the observation exactly like PSL with `K=3`.

```yaml
traffic:
  model: td
  enabled: true
  seed: 42
  v_congestion: 30.0       # km/h — slots 0 & 2 (peak)
  v_freeflow: 90.0         # km/h — slot 1 (free-flow); must exceed v_congestion
  T1: 2400.0               # seconds — morning-peak end
  T2: 4800.0               # seconds — evening-peak start
```

Single-horizon (non-cyclic): once `t ≥ T2` the slot stays `2`. Set `T1`/`T2` within the per-session `simulation.max_time` so each SUMO session traverses all three slots. Training config: `config/trsp/train_td.yaml`.

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

Always picks the neighbour with the smallest `steps2dest` (greedy shortest path). Used as a static baseline. No learning. Works for both PSL and SVC evaluation since it ignores traffic state.

### `DSPAgent` ✅ Done (`dsp.py`)

**Dynamic Shortest Path** — a non-learning, traffic-aware baseline that sits between static Dijkstra and RL.

At the **start of each OD leg**, queries `traci.edge.getLastStepMeanSpeed()` for every edge in `G_dir`, reweights the graph by travel time (`length / speed`), and runs `nx.dijkstra_path` once. The resulting node sequence is followed hop-by-hop for the rest of the leg without any further TraCI queries.

- **Action mode:** `node_id` — returns a global node index directly.
- **Speed fallback:** edges with zero/negative TraCI speed (no vehicles yet) use `ugv_speed`; floor of `0.1 m/s` prevents division by zero on fully jammed edges.
- **Leg hook:** `start_leg(src_node, dst_node)` is called by `plan_route()` in `eval_trsp2.py` via `hasattr(agent, 'start_leg')` before the routing loop.
- **Fallback:** if the planned path is empty (disconnected graph) or the agent ends up off-path, it falls back to the static best-direction neighbour from `obs['best_dir']`.
- **Construction:** `DSPAgent(agent_id, G_dir, node2idx, idx2node, ugv_speed)` — requires the directed graph and node maps at construction time; undirected `G` passed via `**kwargs` from `plan_route()` for the fallback only.

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

### `SARSASVCAgent` ✅ Done (`sarsa_svc.py`) — SVC traffic

SARSA tabular Q-learning adapted for SVC. Discretizes the continuous `neighbor_speeds` vector into bins to keep the state space finite.

- **State key:** `(graph_node, target_node, speed_bins_tuple)` — the speed vector is digitized into `num_speed_bins` bins via `np.digitize`
- **Action space:** `neighbor_index` — same as SARSA-PSL; action is an index into sorted neighbour list up to `max_degree`
- **Q-table:** `dict[state → np.ndarray[max_degree]]`; initialized with uniform noise; updates identical to SARSA-PSL
- **Config key:** `training.sarsa.num_speed_bins` (default 3)
- **Policy pickle:** same Q-table dict format as SARSA-PSL
- **Limitation:** the state explosion from combining positional state with speed bins can slow convergence on networks with many neighbour-speed combinations; DQN-SVC handles this better via continuous function approximation

### `DQNSVCAgent` ✅ Done (`dqn_svc.py`) — SVC traffic

DQN variant for the SVC traffic model. Replaces the global `time_slot` with real-time local traffic observations.

- **Observation encoding:** one-hot(graph_node) ++ one-hot(target_node) ++ `neighbor_speeds[max_degree]`; `obs_dim = 2·num_nodes + max_degree` (e.g. 55 for grid_road with max_degree=5)
- `neighbor_speeds` queried live from `traci.edge.getLastStepMeanSpeed()` at each step, normalized by `ugv_speed`, sorted by `node2idx` to match action ordering
- **Network:** 3-layer MLP (ReLU, hidden=128) — wider than PSL DQN to handle the larger obs vector
- **Hardware:** CUDA-aware; moves networks and tensors to GPU if available
- **Training:** same ε-greedy / replay / target-net scheme as DQN-PSL
- **Policy pickle keys:** `state_dict`, `num_nodes`, `max_degree`, `obs_dim`, `hidden_dim`

### `A2CSVCAgent` ✅ Done (`a2c_svc.py`) — SVC traffic

A2C variant for SVC traffic. Architecture and training loop identical to `A2CAgent` (PSL) but replaces `time_slot` with live `neighbor_speeds` in the observation.

- **Observation encoding:** one-hot(graph_node) ++ one-hot(target_node) ++ `neighbor_speeds[max_degree]`; `obs_dim = 2·num_nodes + max_degree`
- **Network:** separate actor and critic, 2-layer Tanh MLP (hidden=128), orthogonal init
- **Action space:** `node_id` — global node index, masked to valid neighbours at inference
- **Policy pickle keys:** `actor_state_dict`, `critic_state_dict`, `num_nodes`, `max_degree`, `obs_dim`, `hidden_dim`

### `PPOSVCAgent` ✅ Done (`ppo_svc.py`) — SVC traffic

PPO variant for SVC traffic. Architecture and training loop identical to `PPOAgent` (PSL) but uses `neighbor_speeds` observation.

- **Observation encoding:** same as A2C-SVC — `obs_dim = 2·num_nodes + max_degree`
- **Network:** separate actor and critic, 2-layer Tanh MLP (hidden=128), orthogonal init
- **Training:** `num_steps`-step rollouts, GAE (λ=0.95), `update_epochs` epochs of minibatch PPO (clipped ratio + entropy bonus)
- **Policy pickle keys:** `actor_state_dict`, `critic_state_dict`, `num_nodes`, `max_degree`, `obs_dim`, `hidden_dim`

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

| Param | SARSA | DQN-PSL | DQN-SVC | PPO | A2C | SARSA-SVC | A2C-SVC | PPO-SVC |
|-------|-------|---------|---------|-----|-----|-----------|---------|---------|
| Episodes / Iterations | 5000 ep | 5000 ep | 5000 ep | 500 iter | 500 iter | 5000 ep | 100 iter | 100 iter |
| Steps per rollout | — | — | — | 512 | 512 | — | 512 | 512 |
| Learning rate | 0.1 (α) | 1e-3 | 1e-3 | 2.5e-4 | 7e-4 | 0.1 (α) | 7e-4 | 2.5e-4 |
| Hidden dim | — | 64 | 128 | 64 | 64 | — | 128 | 128 |
| Gamma | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| Warm-up time | — | — | 60s | — | — | 60s | 60s | 60s |
| Speed bins | — | — | — | — | — | 3 | — | — |

---

## Evaluation: `eval_trsp2.py`

Runs 100 OD pairs sequentially within a **single continuous SUMO session** — critical for K>1 policies since traffic slots evolve naturally across pairs, and for SVC since background vehicles persist between evaluations.

### Policy evaluation

1. Builds graph from the saved policy's `config.yaml`.
2. Detects `traffic.model` (`psl` or `svc`) and configures the env accordingly.
3. For SVC: appends `--route-files` to the SUMO command; queries `traci.edge.getLastStepMeanSpeed()` for obs.
4. For DSP: calls `agent.start_leg(src, dst)` via `hasattr` hook before each route's planning loop.
5. Plans each route using the agent's `predict()` at current sim state.
6. Injects a vehicle on the planned edge list, steps SUMO until arrival or timeout.
7. Writes `runs/trsp/<policy_id>/test_<N>/simulation_results.csv`.

### Automatic baselines

After the policy eval, **Dijkstra and DSP baselines are always run automatically** on the same 100 OD pair indices, same config, and same traffic model — each in an independent SUMO session. Results are co-located in the same `test_N/` directory:

| File | Agent |
|------|-------|
| `simulation_results.csv` | Trained policy |
| `simulation_results_dijkstra.csv` | `DijkstraAgent` |
| `simulation_results_dsp.csv` | `DSPAgent` (live-speed Dijkstra) |

The `--dijkstra` and `--dsp` CLI flags have been removed — baselines are unconditional.

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
    ├── simulation_results.csv              # Policy results
    ├── simulation_results_dijkstra.csv     # Dijkstra baseline (same OD pairs)
    └── simulation_results_dsp.csv          # DSP baseline (same OD pairs)
```

**Policy pickle schema:**

| Agent | Keys in `policy.pkl` |
|-------|----------------------|
| SARSA / SARSA-SVC | Q-table dict |
| DQN-PSL | `state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |
| DQN-SVC | `state_dict`, `num_nodes`, `max_degree`, `obs_dim`, `hidden_dim` |
| PPO-PSL | `actor_state_dict`, `critic_state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |
| A2C-PSL | `actor_state_dict`, `critic_state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |
| A2C-SVC / PPO-SVC | `actor_state_dict`, `critic_state_dict`, `num_nodes`, `max_degree`, `obs_dim`, `hidden_dim` |

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

### SVC Traffic — All Agents (all three networks, SVC-svc_42)

All SVC agents trained with `traffic_scale` per network (N1: 1.5, N2: 2.5, N3: 2.0), `warm_up_time: 60s`.

#### DQN-SVC

| Network | Policy | Mean time | Dijkstra | DSP | vs Dijkstra | Success |
|---------|--------|-----------|----------|-----|-------------|---------|
| N1 (grid_road) | `19acc44f` | 112.3s | 152.2s | 136.5s | −26% ✓ | 99% |
| N2 (rand_grid_road) | `2553013e` | 59.7s | 82.0s | 68.3s | −27% ✓ | 75% |
| N3 (ames) | `964ad609` | 146.1s | 151.5s | 145.2s | −3.5% | 95% |

DQN-SVC achieves the best balance of mean traversal time and success rate across all networks. N2 success rate (75%) is lower due to short edges requiring more hops; N3 shows near-tie with DSP and Dijkstra on this structured campus network.

#### SARSA-SVC

| Network | Policy | Mean time | Dijkstra | DSP | vs Dijkstra | Success |
|---------|--------|-----------|----------|-----|-------------|---------|
| N1 (grid_road) | `e6703a2a` | 258.2s | 156.7s | 138.2s | +65% ✗ | 72% |
| N2 (rand_grid_road) | `38ac1d0a` | 52.1s | 59.5s | 62.8s | −12% ✓ | 58% |
| N3 (ames) | `d3230dca` | 156.8s | 158.9s | 147.1s | −1.3% | 94% |

SARSA-SVC's discretized speed-bin state space works reasonably on N2 and N3 but fails on N1: the grid_road topology (regular grid, high degree, many similar-length edges) creates too many state collisions, pushing mean traversal time well above Dijkstra. The 58% success rate on N2 reflects the same short-edge problem as DQN-SVC.

#### A2C-SVC

| Network | Policy | Mean time | Dijkstra | DSP | vs Dijkstra | Success |
|---------|--------|-----------|----------|-----|-------------|---------|
| N1 (grid_road) | `1ae65214` | 117.4s | 114.6s | 124.7s | +2.4% | 19% |
| N2 (rand_grid_road) | `6fe5d2d5` | 13.4s | 45.8s | 51.4s | −71% ✓ | 10% |
| N3 (ames) | `ccbf5610` | 109.8s | 152.3s | 148.9s | −28% ✓ | 6% |

A2C-SVC mirrors the PSL pattern: extremely fast on successful routes (N2: 13.4s vs 45.8s Dijkstra, N3: 109.8s vs 152.3s) but success rates are very low (6–19%). The policy converges to a fast strategy that works on a small fraction of OD pairs while failing to generalise. N1 essentially ties Dijkstra (the grid's regular structure limits gain).

#### PPO-SVC

| Network | Policy | Mean time | Dijkstra | DSP | vs Dijkstra | Success |
|---------|--------|-----------|----------|-----|-------------|---------|
| N1 (grid_road) | `7ed3a4eb` | 69.4s | 114.6s | 124.7s | −39% ✓ | 16% |
| N2 (rand_grid_road) | `17301777` | 20.7s | 45.8s | 51.4s | −55% ✓ | 7% |
| N3 (ames) | `2c00863a` | 66.5s | 152.3s | 148.9s | −56% ✓ | 13% |

PPO-SVC shows the largest mean-time improvement when successful (N3: 66.5s vs 152.3s, N1: 69.4s vs 114.6s) but at the lowest success rates (7–16%). Same fast-but-narrow convergence as A2C-SVC, with PPO's clipped objective providing slightly higher per-route speed gains. The low success rates make PPO-SVC unsuitable for production TA-EVRP use without further training or reward shaping.

**Summary across SVC agents:**
- **DQN-SVC** is the recommended SVC agent: high success rate (75–99%), consistent improvement over Dijkstra across all networks.
- **SARSA-SVC** is viable for N2 and N3 but degrades on N1 due to state-space aliasing.
- **A2C-SVC / PPO-SVC** have strong per-route speed gains but low success rates (~6–19%); suitable for research comparison, not for end-to-end TA-EVRP deployment without improvement.

### TD Traffic — All Agents (N1–N3, v_c=10 km/h, v_f=40 km/h)

Three-period congestion model: `K=3`, `T1=100s`, `T2=200s`, `v_congestion=10.0 km/h`, `v_freeflow=40.0 km/h`. Training config: `config/trsp/train_td.yaml`. Policy IDs in `logs/td/policy_ids.txt`.

#### SARSA-TD

| Network | Policy | Mean time | Dijkstra | DSP | vs Dijkstra | Success |
|---------|--------|-----------|----------|-----|-------------|---------|
| N1 (grid_road)      | `53255479` | 673.2s | 562.5s | 562.5s | +19.7% ✗ | 74% |
| N2 (rand_grid_road) | `fab1cbeb` | 102.6s | 108.8s | 106.0s | −5.7% ✓ | 43% |
| N3 (ames)           | `14cd2751` | 288.5s | 360.8s | 355.7s | −20.1% ✓ | 47% |

#### DQN-TD

| Network | Policy | Mean time | Dijkstra | DSP | vs Dijkstra | Success |
|---------|--------|-----------|----------|-----|-------------|---------|
| N2 (rand_grid_road) | `c365a5c3` | 128.4s | 130.7s | 127.7s | −1.8% | 99% |
| N3 (ames)           | `22781c4a` | 354.6s | 360.8s | 355.7s | −1.7% | 94% |

DQN achieves the highest success rates under TD traffic (94–99%), closely matching Dijkstra on traversal time. No N1 policy trained.

#### A2C-TD

| Network | Policy | Mean time | Dijkstra | DSP | vs Dijkstra | Success |
|---------|--------|-----------|----------|-----|-------------|---------|
| N2 (rand_grid_road) | `7d5b6d27` |  50.0s | 108.8s | 106.0s | −54.0% ✓ | 10% |
| N3 (ames)           | `8f6c4be4` | 205.8s | 360.8s | 355.7s | −43.0% ✓ | 12% |

#### PPO-TD

| Network | Policy | Mean time | Dijkstra | DSP | vs Dijkstra | Success |
|---------|--------|-----------|----------|-----|-------------|---------|
| N2 (rand_grid_road) | `f1682c4c` |  73.4s | 108.8s | 106.0s | −32.6% ✓ | 14% |
| N3 (ames)           | `0a2ff4c1` | 173.2s | 360.8s | 355.7s | −52.0% ✓ | 12% |

**Summary across TD agents:**
- **DQN-TD** is the recommended TD agent: near-100% success rate on N2 and N3, marginally faster than Dijkstra.
- **SARSA-TD** is viable on N2 and N3 but degrades on N1 (same regular-grid aliasing issue as PSL); lower success rate than DQN (43–74%).
- **A2C-TD / PPO-TD** mirror the PSL/SVC pattern: fast on successful routes (up to −54% makespan) but very low success rates (10–14%). Not suitable for end-to-end TA-EVRP use without further training.

---

## Analysis Notebook

`notebooks/trsp_testing_analysis.ipynb` — three sections:

| Section | What it does |
|---------|-------------|
| **Policy Evaluation** | Set `policy_id`; loads the latest `test_N/` directory; compares policy vs Dijkstra vs DSP on common successful OD pairs (box, mean bar, histogram, route-index plots, route map highlights). |
| **Policy vs Policy** | Set `algo1`/`algo2` (policy IDs or `"dsp"`/`"dijkstra"`); compares any two runs head-to-head using the same analysis. |
| **Aggregated Results** | Loads `results/trsp_results_psl.csv` (pre-computed PSL sweep), then scans `runs/trsp/` for SVC and TD policies whose latest `test_N/` has all three co-located CSVs. Combines PSL + SVC + TD into `results/trsp_results.csv` and plots mean makespan and success rate. Saves `notebooks/aggregate_makespan.png` and `notebooks/aggregate_success_rate.png`. |

---

## Known Issues / Future Work

- `SARSALambdaAgent` training loop not yet wired.
- DQN-SVC / SARSA-SVC N2 (rand_grid_road) success rates (75% / 58%) are below Dijkstra (99%) — the short-edge topology requires more hops per route; consider increasing `max_steps` or training longer.
- A2C-SVC and PPO-SVC success rates are low (6–19%): these agents converge fast but narrowly. Reward shaping, longer training, or curriculum (start with easy OD pairs) may improve generalisation.
- SARSA-SVC degrades on N1 (grid_road) relative to Dijkstra: the regular grid creates too many state collisions under binned speed discretisation.
- `infer_trsp.py` treats an episode as done only when `terminated` (wall-clock) fires; it does not break on `truncated`. Use `simulation.max_time` to bound runs, or adjust the runner for per-route inference.
- TRSP's `PSLTrafficModel` and EVRP's `PSLTrafficModel` are independent instances. Config key names differ: TRSP uses `v_min`/`v_max`/`speed_step` (km/h), EVRP uses `v_min_kmh`/`v_max_kmh`/`speed_step_kmh`.
