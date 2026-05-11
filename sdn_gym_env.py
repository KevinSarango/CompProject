import csv
import os

import numpy as np

from config import (
    ACTION_TO_PATH,
    BASE_REWARD,
    DEFAULT_FLOW_SIZE_KB,
    DEMAND_BINS,
    GAMMA_ACTION_IMPACT,
    GAMMA_DELAY,
    GAMMA_IMBALANCE,
    GAMMA_PACKET_LOSS,
    GAMMA_SWITCHING,
    GAMMA_THROUGHPUT,
    LOAD_DECAY_FACTOR,
    MAX_DELAY_SCORE,
    MAX_FLOW_SIZE_KB,
    NUM_PATHS,
    PATH_CAPACITY_KB,
    PATH_DELAY_FACTOR,
    STATE_BINS,
)


class SimpleSDNEnv:
    """
    Demand-driven SDN environment for tabular Q-learning.

    Multipath ratio-state representation:
        state = least_utilized_path_utilization_spread_bin_delay_spread_bin_demand_bin_previous_action

    This generalizes the PDF's two-path ratio state:
        St = [delta_U, delta_D, lambda_in]

    For more than two paths, there is no single delta between only path A and B.
    Instead, this environment uses compact multipath summaries:
        - least_utilized_path: path index with the lowest current utilization
        - utilization_spread_bin: discretized max(utilization) - min(utilization)
        - delay_spread_bin: discretized max(delay_score) - min(delay_score)
        - demand_bin: discretized current incoming flow size
        - previous_action: previous selected path index, retained to discourage flapping

    The implementation uses numpy arrays so the same code works for diamond
    and three_path topologies.
    """

    def __init__(self, demand_file="data/trafpy_demands.csv"):
        self.demand_file = demand_file
        self.demands = self.load_demands()

        self.current_index = 0
        self.path_loads = np.zeros(NUM_PATHS, dtype=float)
        self.path_capacities = np.array(PATH_CAPACITY_KB, dtype=float)
        self.path_delay_factors = np.array(PATH_DELAY_FACTOR, dtype=float)
        self.previous_action = 0
        self.decay_factor = LOAD_DECAY_FACTOR

    def load_demands(self):
        if not os.path.exists(self.demand_file):
            raise FileNotFoundError(
                f"{self.demand_file} not found. Run python3 generate_trafpy_demands.py first."
            )

        demands = []

        with open(self.demand_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                demands.append({
                    "flow_id": int(row["flow_id"]),
                    "src": row["src"],
                    "dst": row["dst"],
                    "start_time": float(row["start_time"]),
                    "size_kb": float(row["size_kb"]),
                })

        return demands


    def set_demands(self, demands):
        """Replace the active traffic trace used by the next episode.

        Training can call this once per episode to avoid replaying the exact
        same demand sequence for all episodes. Each demand dictionary should
        contain flow_id, src, dst, start_time, and size_kb.
        """
        self.demands = list(demands)
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        self.path_loads = np.zeros(NUM_PATHS, dtype=float)
        self.previous_action = 0
        return self.get_state(self.get_current_flow_size())

    def get_current_flow_size(self):
        if self.current_index >= len(self.demands):
            return DEFAULT_FLOW_SIZE_KB

        return self.demands[self.current_index]["size_kb"]

    def normalize(self, value, max_value):
        if max_value == 0:
            return 0.0

        value = max(0.0, min(float(value), float(max_value)))
        return value / max_value

    def discretize(self, value, min_value, max_value, bins):
        """Convert a continuous value into a bounded integer bin."""
        if bins <= 1 or max_value <= min_value:
            return 0

        value = max(min_value, min(float(value), max_value))
        scaled = (value - min_value) / (max_value - min_value)
        index = int(scaled * bins)
        return min(index, bins - 1)

    def get_path_utilizations(self):
        """Return per-path utilization as load / path-specific capacity."""
        return self.path_loads / self.path_capacities

    def get_path_delay_scores(self):
        """
        Return normalized per-path delay scores.

        Delay is still simulated, not measured from the switch. It combines
        utilization and the path-specific delay factor from config.py.
        """
        utilizations = self.get_path_utilizations()
        raw_delay_scores = utilizations * 100.0 * self.path_delay_factors
        return np.clip(raw_delay_scores / MAX_DELAY_SCORE, 0.0, 1.0)

    def get_least_utilized_path(self):
        return int(np.argmin(self.get_path_utilizations()))

    def get_utilization_spread_bin(self):
        utilizations = self.get_path_utilizations()
        spread = float(np.max(utilizations) - np.min(utilizations))
        return self.discretize(spread, 0.0, 1.0, STATE_BINS)

    def get_delay_spread_bin(self):
        delay_scores = self.get_path_delay_scores()
        spread = float(np.max(delay_scores) - np.min(delay_scores))
        return self.discretize(spread, 0.0, 1.0, STATE_BINS)

    def get_demand_bin(self, flow_size_kb):
        return self.discretize(
            value=flow_size_kb,
            min_value=0.0,
            max_value=MAX_FLOW_SIZE_KB,
            bins=DEMAND_BINS,
        )

    def get_state(self, flow_size_kb):
        least_path = self.get_least_utilized_path()
        util_spread_bin = self.get_utilization_spread_bin()
        delay_spread_bin = self.get_delay_spread_bin()
        demand_bin = self.get_demand_bin(flow_size_kb)

        return (
            f"{least_path}_"
            f"{util_spread_bin}_"
            f"{delay_spread_bin}_"
            f"{demand_bin}_"
            f"{self.previous_action}"
        )

    def calculate_action_impact(self, action, selected_util):
        """
        Positive if the selected path is less utilized than the average
        alternative path. Negative if the selected path is more utilized.
        """
        utilizations = self.get_path_utilizations()
        other_utils = np.delete(utilizations, action)

        if other_utils.size == 0:
            return 0.0

        avg_other_util = float(np.mean(other_utils))
        impact = avg_other_util - float(selected_util)
        return max(-1.0, min(1.0, impact))

    def step(self, action):
        if self.current_index >= len(self.demands):
            return self.get_state(DEFAULT_FLOW_SIZE_KB), 0.0, True, {}

        if action < 0 or action >= NUM_PATHS:
            raise ValueError(f"Invalid action={action}. Expected 0..{NUM_PATHS - 1}.")

        demand = self.demands[self.current_index]
        flow_size = demand["size_kb"]

        # Simulate old flows completing over time.
        self.path_loads *= self.decay_factor

        selected_path = ACTION_TO_PATH[str(action)]
        self.path_loads[action] += flow_size

        utilizations = self.get_path_utilizations()
        delay_scores = self.get_path_delay_scores()

        selected_load = float(self.path_loads[action])
        selected_capacity = float(self.path_capacities[action])
        selected_delay_factor = float(self.path_delay_factors[action])
        selected_util = float(utilizations[action])

        # Path-aware delay model: combines utilization and path-specific delay factor.
        raw_delay = selected_util * 100.0 * selected_delay_factor

        # Path-aware packet loss: loss begins when selected path exceeds capacity.
        raw_packet_loss = max(0.0, selected_util - 1.0) * 100.0

        # Path-aware throughput: remaining capacity on the selected path.
        raw_throughput = max(0.0, selected_capacity - selected_load)

        normalized_delay = self.normalize(raw_delay, MAX_DELAY_SCORE)
        normalized_packet_loss = self.normalize(raw_packet_loss, 100.0)
        normalized_throughput = self.normalize(raw_throughput, selected_capacity)

        action_impact = self.calculate_action_impact(
            action=action,
            selected_util=selected_util,
        )

        imbalance = float(np.max(utilizations) - np.min(utilizations))
        imbalance = max(0.0, min(1.0, imbalance))

        switching_penalty = 1.0 if action != self.previous_action else 0.0

        reward = (
            BASE_REWARD
            - GAMMA_PACKET_LOSS * normalized_packet_loss
            - GAMMA_DELAY * normalized_delay
            + GAMMA_THROUGHPUT * normalized_throughput
            + GAMMA_ACTION_IMPACT * action_impact
            - GAMMA_IMBALANCE * imbalance
            - GAMMA_SWITCHING * switching_penalty
        )

        self.previous_action = action
        self.current_index += 1

        next_flow_size = self.get_current_flow_size()
        next_state = self.get_state(next_flow_size)
        done = self.current_index >= len(self.demands)

        info = {
            "flow_id": demand["flow_id"],
            "src": demand["src"],
            "dst": demand["dst"],
            "start_time": demand["start_time"],
            "size_kb": flow_size,
            "selected_path": selected_path,
            "path_loads": self.path_loads.tolist(),
            "path_utilizations": utilizations.tolist(),
            "path_delay_scores": delay_scores.tolist(),
            "selected_load": selected_load,
            "selected_capacity": selected_capacity,
            "selected_delay_factor": selected_delay_factor,
            "selected_util": selected_util,
            "raw_delay": raw_delay,
            "raw_packet_loss": raw_packet_loss,
            "raw_throughput": raw_throughput,
            "normalized_delay": normalized_delay,
            "normalized_packet_loss": normalized_packet_loss,
            "normalized_throughput": normalized_throughput,
            "action_impact": action_impact,
            "imbalance": imbalance,
            "switching_penalty": switching_penalty,
            "reward": reward,
        }

        return next_state, reward, done, info
