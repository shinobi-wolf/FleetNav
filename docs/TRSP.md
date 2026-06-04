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
│   ├── __init__.py          # get_agent() registry
│   ├── dijkstra.py          # Greedy shortest-path baseline ✅
│   ├── sarsa.py             # SARSA tabular Q-learning ✅
│   ├── sarsa_l.py           # SARSA(λ) with eligibility traces (skeleton)
│   ├── dqn.py               # Deep Q-Network (CleanRL-style) ✅
│   ├── ppo.py               # Proximal Policy Optimization ✅
│   └── a2c.py               # Advantage Actor-Critic ✅
└── trainer.py               # lmd_trainer() — training loop

config/trsp/
├── train.yaml               # Default training config (editable algo/net/traffic)
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

Two modes depending on `agent.action_mode`:

**`node_id` mode** (DQN, PPO, A2C):
```python
spaces.Dict({
    'graph_node':  Discrete(num_nodes),   # current node index
    'target_node': Discrete(num_nodes),   # goal node index
    'time_slot':   Discrete(K),           # current PSL slot index
})
```

**`neighbor_index` mode** (SARSA, Dijkstra):
```python
spaces.Dict({
    'graph_node':  Discrete(num_nodes),
    'target_node': Discrete(num_nodes),
    'time_slot':   Discrete(K),
    'best_dir':    Discrete(max_degree),  # argmin of steps-to-dest per neighbour
})
```

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

## Traffic Model: `PSLTrafficModel`

**File:** `dronesim/trsp/traffic.py`

Per-edge piecewise speed profiles — each edge gets its own independent speed for each time slot.

### Config Keys

```yaml
traffic:
  enabled: true
  seed: 42
  T: 7200.0              # total horizon in seconds
  K: 1                   # number of time slots (1 or 2 tested)
  v_min: 40.0            # km/h — minimum speed level
  v_max: 100.0           # km/h — maximum speed level
  speed_step: 60.0       # km/h — step between levels
  speed_profile_type: random  # random | function | file
```

With `K=1` every edge has a single fixed speed for the entire simulation. With `K=2` and `speed_step=60.0`, exactly two speed levels exist: `{40, 100}` km/h; each edge independently draws a speed per slot.

### Speed Profile Generation

- `random`: each edge independently draws from a discrete set of speed levels spanning `[v_min, v_max]` in increments of `speed_step`. Same seed + same net file → identical profiles across training, eval, and inference.
- `function`: `fn(edge_id, slot_idx) → km/h` callable.
- `file`: JSON/YAML mapping `edge_id → [v_slot0, v_slot1, ...]` in km/h.

Internally, the model stores and returns speeds in **m/s**; config values are in km/h.

---

## Agents

### `DijkstraAgent` ✅ Done

Always picks the neighbour with the smallest `steps2dest` (greedy shortest path). Used as the baseline. No learning.

### `SARSAAgent` ✅ Done

- **State key:** `(graph_node, target_node, time_slot)` — 3-tuple
- **Q-table:** `dict[state → np.ndarray[max_degree]]`. Initialised with `uniform(-1e-5, 1e-5)` noise to break argmax ties.
- **Action selection:** slice Q to `[:num_neighbours]` — only valid neighbour indices are considered.
- **Update:** `Q(s,a) += α [r + γ Q(s',a') − Q(s,a)]`; terminal/truncation steps use `r` only.
- **ε-greedy** with multiplicative decay each episode.
- Policies saved as `policy.pkl` (pickle of the Q-table dict).
- Post-load 2-cycle detection: detects greedy A→B→A loops and breaks them via a Dijkstra-consistent nudge.

### `SARSALambdaAgent` ⚠️ Partial

Eligibility traces support added to file structure; training loop not yet wired.

### `DQNAgent` ✅ Done

CleanRL-style MLP Q-network with experience replay.

- **Observation encoding:** one-hot(graph_node) ++ one-hot(target_node) ++ one-hot(time_slot); `obs_dim = 2·num_nodes + K`
- **Action space:** `node_id` mode — masked to valid neighbours via `-inf` on logits before `argmax`
- **Network:** 3-layer MLP (ReLU, hidden=64), separate Q-value per node
- **Training:** ε-greedy exploration, replay buffer (capacity 50K), target network (sync every 500 steps), TD(0) loss (MSE)
- **Checkpoints:** every 100 episodes; final save at end

### `PPOAgent` ✅ Done

Single-env CleanRL-style PPO with GAE advantage estimation.

- **Observation encoding:** same as DQN — one-hot vector, `obs_dim = 2·num_nodes + K`
- **Action space:** `node_id` mode with `masked_fill(-inf)` before `Categorical`
- **Network:** separate actor and critic, 2-layer Tanh MLP (hidden=64), orthogonal init
- **Training:** collects `num_steps`-step rollouts, computes GAE (λ=0.95), runs `update_epochs` epochs of minibatch PPO updates (clipped ratio + clipped value loss + entropy bonus)
- **Episode handling:** `truncated` → episode boundary (env auto-resets); `terminated` → wall-clock exhausted → full `env.close()` + `env.reset()` (takes priority over `truncated`)
- **Checkpoints:** every `save_interval` iterations

### `A2CAgent` ✅ Done

Simpler sibling of PPO — same network, same rollout collection, same GAE — but a single unclipped gradient step per rollout.

- **Observation encoding / action space / network:** identical to PPO
- **Training:** collects `num_steps`-step rollouts, computes GAE, performs **one** gradient step over the full rollout batch (no epochs, no minibatches)
- **Policy loss:** plain policy gradient `−log_prob · advantage` (no importance-sampling ratio or clipping)
- **Value loss:** unclipped `0.5 · MSE(value, return)`
- **Logging:** per-iteration `pg_loss`, `v_loss`, `mean_ep_ret` (episode rewards accumulated within each rollout window)
- **Checkpoints:** every `save_interval` iterations

---

## Training Config

Default: `config/trsp/train.yaml` — edit `agent.algo` and `net.net_path` / `traffic.K` to switch configurations.

Dedicated A2C configs for all 6 (network × traffic) combinations live in `config/trsp/a2c_*.yaml`. Run all 6 sequentially with eval via:

```bash
./train_a2c_all.sh
```

### Key Hyperparameters

| Param | SARSA | DQN | PPO | A2C |
|-------|-------|-----|-----|-----|
| Episodes / Iterations | 5000 ep | 5000 ep | 500 iter | 500 iter |
| Steps per rollout | — | — | 512 | 512 |
| Learning rate | 0.1 (α) | 1e-3 | 2.5e-4 | 7e-4 |
| Gamma | 0.99 | 0.99 | 0.99 | 0.99 |
| GAE λ | — | — | 0.95 | 0.95 |
| Entropy coef | — | — | 0.01 | 0.01 |
| Value coef | — | — | 0.5 | 0.5 |
| Update epochs | — | — | 4 | 1 (full batch) |

---

## Evaluation: `eval_trsp2.py`

Runs 100 OD pairs sequentially within a **single continuous SUMO session** — critical for K>1 policies since traffic slots evolve naturally across pairs.

1. Builds graph and PSL model from the saved policy's `config.yaml`.
2. Selects 101 nodes with a fixed seed (→ 100 pairs, reproducible across all algos).
3. Plans each route using the agent's `predict()` (graph-level, no SUMO) at the current sim-time traffic slot.
4. Injects a vehicle on the planned edge list, steps SUMO until arrival or timeout.
5. Writes `runs/trsp/<policy_id>/test_<N>/simulation_results.csv`.

With `--dijkstra`: reruns the same 100 pairs with `DijkstraAgent` and saves results to `runs/trsp/dijkstra/test_<N>/`.

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
| DQN | `state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |
| PPO | `actor_state_dict`, `critic_state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |
| A2C | `actor_state_dict`, `critic_state_dict`, `num_nodes`, `num_slots`, `obs_dim`, `hidden_dim` |

---

## Networks Tested

| ID | Directory | Nodes | Edges | Notes |
|----|-----------|-------|-------|-------|
| N1 | `data/grid_road/` | ~25 | ~40 | Regular grid |
| N2 | `data/rand_grid_road/` | 26 | 39 | Randomized grid |
| N3 | `data/ames/` | larger | larger | Ames campus road network |

All networks: `max_degree: 5`.

---

## Known Issues / Future Work

- `SARSALambdaAgent` training loop not yet wired.
- `infer_trsp.py` treats an episode as done only when `terminated` (wall-clock) fires; it does not break on `truncated`. Use `simulation.max_time` to bound runs, or adjust the runner for per-route inference.
- TRSP's `PSLTrafficModel` and EVRP's `PSLTrafficModel` are independent instances. Config key names differ: TRSP uses `v_min`/`v_max`/`speed_step` (km/h), EVRP uses `v_min_kmh`/`v_max_kmh`/`speed_step_kmh`.
