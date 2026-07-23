import math
import json
import random as _random

import numpy as np


class PSLTrafficModel:
    """
    Piecewise Speed Limit (PSL) traffic model.

    Each edge e has an independent speed profile v[e][0..K-1].
    At simulation time t the active slot is j = floor(t / delta_t) % K (cyclic).
    All speeds are stored internally in m/s; config values are given in km/h.

    Special case K=1: single constant-speed slot per edge, no runtime updates.
    """

    def __init__(self, config: dict, edge_ids: list, seed=None):
        """
        Args:
            config:   traffic config dict (keys: enabled, T, K, v_min, v_max, speed_step,
                      optionally speed_profile_type, speed_profile_fn, speed_profile_path)
            edge_ids: list of non-internal SUMO edge IDs
            seed:     random seed for reproducible random profiles
        """
        self.enabled = bool(config.get('enabled', False))
        if not self.enabled:
            self.K = 1
            self.profiles = {}
            return

        self.T = float(config['T'])
        self.K = int(config['K'])
        if self.K < 1:
            raise ValueError("traffic.K must be >= 1")
        self.delta_t = float(config.get('slot_duration', self.T / self.K))

        self.v_min = round(float(config['v_min']) / 3.6, 8)    # km/h → m/s, rounded to match _speed_levels()
        self.v_max = round(float(config['v_max']) / 3.6, 8)
        self.speed_step = round(float(config['speed_step']) / 3.6, 8)
        if self.v_min > self.v_max:
            raise ValueError("traffic.v_min must be <= v_max")
        if self.speed_step <= 0:
            raise ValueError("traffic.speed_step must be > 0")

        # Optional per-level sampling weights for random profiles. When set, each
        # slot draws a speed level via weighted rng.choices instead of uniform
        # rng.choice; length must match the number of speed levels. Example: two
        # levels (v_min, v_max) with speed_probs [0.3, 0.7] gives p_slow = 0.3.
        self.speed_probs = config.get('speed_probs', None)

        rng = _random.Random(seed)
        profile_type = config.get('speed_profile_type', 'random')

        if profile_type == 'random':
            self._generate_random(edge_ids, rng)
        elif profile_type == 'function':
            self._from_function(edge_ids, config['speed_profile_fn'])
        elif profile_type == 'file':
            self._from_file(edge_ids, config['speed_profile_path'])
        else:
            raise ValueError(f"Unknown speed_profile_type: '{profile_type}'")

        self._validate(edge_ids)

    # ------------------------------------------------------------------ builders

    def _speed_levels(self):
        levels, v = [], self.v_min
        while v <= self.v_max + 1e-9:
            levels.append(round(v, 8))
            v += self.speed_step
        return levels or [self.v_max]

    def _generate_random(self, edge_ids, rng):
        levels = self._speed_levels()
        probs = self.speed_probs
        if probs is not None:
            if len(probs) != len(levels):
                raise ValueError(
                    f"traffic.speed_probs has {len(probs)} weights; expected "
                    f"{len(levels)} (one per speed level {[round(v, 2) for v in levels]})"
                )
            draw = lambda: rng.choices(levels, weights=probs, k=1)[0]
        else:
            draw = lambda: rng.choice(levels)
        self.profiles = {
            eid: [draw() for _ in range(self.K)]
            for eid in sorted(edge_ids)
        }

    def _from_function(self, edge_ids, fn):
        """fn(edge_id: str, slot_idx: int) -> float (km/h)"""
        self.profiles = {
            eid: [float(np.clip(fn(eid, j) / 3.6, self.v_min, self.v_max))
                  for j in range(self.K)]
            for eid in edge_ids
        }

    def _from_file(self, edge_ids, path):
        """File maps edge_id -> [v_slot0_kmh, v_slot1_kmh, ...]"""
        with open(path) as f:
            if path.endswith(('.yaml', '.yml')):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        self.profiles = {
            eid: [float(np.clip(s / 3.6, self.v_min, self.v_max))
                  for s in data.get(eid, [self.v_max * 3.6] * self.K)]
            for eid in edge_ids
        }

    def _validate(self, edge_ids):
        for eid in edge_ids:
            profile = self.profiles.setdefault(eid, [self.v_max] * self.K)
            if len(profile) != self.K:
                raise ValueError(
                    f"Edge '{eid}' has {len(profile)} speed slots; expected {self.K}"
                )
            self.profiles[eid] = [
                float(np.clip(s, self.v_min, self.v_max)) for s in profile
            ]

    # ------------------------------------------------------------------ runtime

    def get_slot(self, t: float) -> int:
        """Active time slot index for simulation time t (seconds)."""
        if self.K == 1:
            return 0
        return int(math.floor(t / self.delta_t)) % self.K

    def get_speed(self, edge_id: str, slot: int) -> float:
        """Speed in m/s for edge_id at the given slot."""
        return self.profiles[edge_id][slot]


class TDTrafficModel(PSLTrafficModel):
    """Three-period time-of-day congestion model (Lu et al. 2020, Fig. 1).

    Slot 0 = morning peak  [0,  T1)  at congestion speed v_c
    Slot 1 = free-flow     [T1, T2)  at free-flow speed  v_f  (> v_c)
    Slot 2 = evening peak  [T2, ...] at congestion speed v_c  (== slot 0)

    Speeds are GLOBAL: every edge gets the identical profile [v_c, v_f, v_c].
    Config keys (under `traffic`): v_congestion, v_freeflow (km/h); T1, T2 (seconds).
    """

    def __init__(self, config: dict, edge_ids: list, seed=None):
        self.enabled = bool(config.get('enabled', False))
        if not self.enabled:
            self.K = 1
            self.profiles = {}
            return

        self.K = 3
        self.v_min = round(float(config['v_congestion']) / 3.6, 8)   # km/h → m/s
        self.v_max = round(float(config['v_freeflow']) / 3.6, 8)
        if not (0.0 < self.v_min < self.v_max):
            raise ValueError("traffic: require 0 < v_congestion < v_freeflow")

        self.T1 = float(config['T1'])   # seconds
        self.T2 = float(config['T2'])   # seconds
        self.T = float(config['T'])
        if not (0.0 < self.T1 < self.T2):
            raise ValueError("traffic: require 0 < T1 < T2")

        self.profiles = {
            eid: [self.v_min, self.v_max, self.v_min] for eid in sorted(edge_ids)
        }

    def get_slot(self, t: float) -> int:
        """Active period index for simulation time t (seconds): 0|1|2 by T1/T2."""
        t = t % self.T  # wrap around T
        if t < self.T1:
            return 0
        if t < self.T2:
            return 1
        return 2
