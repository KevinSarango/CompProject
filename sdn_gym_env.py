import csv
import os

from config import (
    BASE_REWARD,
    DEFAULT_FLOW_SIZE_KB,
    GAMMA_ACTION_IMPACT,
    GAMMA_DELAY,
    GAMMA_IMBALANCE,
    GAMMA_PACKET_LOSS,
    GAMMA_SWITCHING,
    GAMMA_THROUGHPUT,
    LOAD_BALANCE_LIMIT_KB,
    LOAD_DECAY_FACTOR,
    MEDIUM_FLOW_KB,
    SMALL_FLOW_KB,
    UTILIZATION_DIFF_THRESHOLD,
)


class SimpleSDNEnv:
    """
    Demand-driven SDN environment for tabular Q-learning.

    The agent trains on the same TrafPy-style demand trace that is replayed
    later in Mininet.

    Actions:
        0 = upper path: s1 -> s2 -> s4
        1 = lower path: s1 -> s3 -> s4

    18-state representation:
        state = utilization_bin_demand_bin_previous_action

    utilization_bin:
        0 = upper path is less utilized
        1 = paths are similarly utilized
        2 = lower path is less utilized

    demand_bin:
        0 = small incoming flow
        1 = medium incoming flow
        2 = large incoming flow

    previous_action:
        0 = previous flow used upper path
        1 = previous flow used lower path

    Total states:
        3 * 3 * 2 = 18
    """

    def __init__(self, demand_file="data/trafpy_demands.csv"):
        self.demand_file = demand_file
        self.demands = self.load_demands()

        self.current_index = 0
        self.upper_load = 0.0
        self.lower_load = 0.0
        self.previous_action = 0

        self.load_balance_limit_kb = LOAD_BALANCE_LIMIT_KB
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

    def reset(self):
        self.current_index = 0
        self.upper_load = 0.0
        self.lower_load = 0.0
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

    def get_utilization_bin(self):
        upper_util = self.upper_load / self.load_balance_limit_kb
        lower_util = self.lower_load / self.load_balance_limit_kb

        diff = upper_util - lower_util

        if diff < -UTILIZATION_DIFF_THRESHOLD:
            return 0  # upper is less utilized

        if diff > UTILIZATION_DIFF_THRESHOLD:
            return 2  # lower is less utilized

        return 1  # similar utilization

    def get_demand_bin(self, flow_size_kb):
        if flow_size_kb <= SMALL_FLOW_KB:
            return 0

        if flow_size_kb <= MEDIUM_FLOW_KB:
            return 1

        return 2

    def get_state(self, flow_size_kb):
        utilization_bin = self.get_utilization_bin()
        demand_bin = self.get_demand_bin(flow_size_kb)
        return f"{utilization_bin}_{demand_bin}_{self.previous_action}"

    def calculate_action_impact(self, selected_util, other_util):
        """
        Positive if selected path is less utilized than the alternative.
        Negative if selected path is more utilized than the alternative.
        """
        impact = other_util - selected_util
        return max(-1.0, min(1.0, impact))

    def step(self, action):
        if self.current_index >= len(self.demands):
            return self.get_state(DEFAULT_FLOW_SIZE_KB), 0.0, True, {}

        demand = self.demands[self.current_index]
        flow_size = demand["size_kb"]

        # Simulate old flows completing over time.
        self.upper_load *= self.decay_factor
        self.lower_load *= self.decay_factor

        if action == 0:
            selected_path = "upper"
            self.upper_load += flow_size
            selected_load = self.upper_load
            other_load = self.lower_load
        else:
            selected_path = "lower"
            self.lower_load += flow_size
            selected_load = self.lower_load
            other_load = self.upper_load

        selected_util = selected_load / self.load_balance_limit_kb
        other_util = other_load / self.load_balance_limit_kb

        raw_delay = selected_util * 100.0
        raw_packet_loss = max(0.0, selected_util - 1.0) * 100.0
        raw_throughput = max(0.0, self.load_balance_limit_kb - selected_load)

        normalized_delay = self.normalize(raw_delay, 100.0)
        normalized_packet_loss = self.normalize(raw_packet_loss, 100.0)
        normalized_throughput = self.normalize(raw_throughput, self.load_balance_limit_kb)

        action_impact = self.calculate_action_impact(
            selected_util=selected_util,
            other_util=other_util,
        )

        imbalance = abs(self.upper_load - self.lower_load) / self.load_balance_limit_kb
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
            "upper_load": self.upper_load,
            "lower_load": self.lower_load,
            "selected_load": selected_load,
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
