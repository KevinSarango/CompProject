import csv
import os
from itertools import product
from config import *

class SimpleSDNEnv:
    """
    54-state demand-driven Q-learning environment.
    State key: util_bin_delay_bin_demand_bin_previous_action
    util/delay bins: 0=upper better, 1=similar, 2=lower better
    demand bins: 0=small, 1=medium, 2=large
    previous_action: 0=upper, 1=lower
    """
    def __init__(self, demand_file="data/trafpy_demands.csv"):
        self.demand_file = demand_file
        self.demands = self.load_demands()
        self.limit = LOAD_BALANCE_LIMIT_KB
        self.decay = LOAD_DECAY_FACTOR
        self.reset()

    @staticmethod
    def all_state_keys():
        return [f"{u}_{d}_{s}_{p}" for u, d, s, p in product(range(3), range(3), range(3), range(2))]

    def load_demands(self):
        if not os.path.exists(self.demand_file):
            raise FileNotFoundError(f"{self.demand_file} not found. Run generate_trafpy_demands.py first.")
        out = []
        with open(self.demand_file) as f:
            for row in csv.DictReader(f):
                out.append({
                    "flow_id": int(row["flow_id"]),
                    "src": row["src"],
                    "dst": row["dst"],
                    "start_time": float(row["start_time"]),
                    "size_kb": float(row["size_kb"]),
                })
        return out

    def reset(self):
        self.current_index = 0
        self.upper_load = 0.0
        self.lower_load = 0.0
        self.previous_action = 0
        return self.get_state()

    @staticmethod
    def normalize(value, max_value):
        if max_value == 0:
            return 0.0
        return max(0.0, min(float(value), float(max_value))) / max_value

    @staticmethod
    def difference_bin(upper_value, lower_value, threshold):
        diff = upper_value - lower_value
        if diff < -threshold:
            return 0
        if diff > threshold:
            return 2
        return 1

    @staticmethod
    def demand_bin(size_kb):
        if size_kb <= SMALL_FLOW_KB:
            return 0
        if size_kb <= LARGE_FLOW_KB:
            return 1
        return 2

    def get_state(self):
        size_kb = 0.0 if self.current_index >= len(self.demands) else self.demands[self.current_index]["size_kb"]
        upper_util = self.upper_load / self.limit
        lower_util = self.lower_load / self.limit
        util_bin = self.difference_bin(upper_util, lower_util, UTIL_DIFF_THRESHOLD)
        delay_bin = self.difference_bin(upper_util, lower_util, DELAY_DIFF_THRESHOLD)
        demand_bin = self.demand_bin(size_kb)
        return f"{util_bin}_{delay_bin}_{demand_bin}_{self.previous_action}"

    def step(self, action):
        if self.current_index >= len(self.demands):
            return self.get_state(), 0.0, True, {}

        demand = self.demands[self.current_index]
        flow_size = demand["size_kb"]
        old_previous_action = self.previous_action

        self.upper_load *= self.decay
        self.lower_load *= self.decay

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

        selected_util = selected_load / self.limit
        other_util = other_load / self.limit
        raw_delay = selected_util * 100.0
        raw_packet_loss = max(0.0, selected_util - 1.0) * 100.0
        raw_throughput = max(0.0, self.limit - selected_load)
        normalized_delay = self.normalize(raw_delay, 100.0)
        normalized_packet_loss = self.normalize(raw_packet_loss, 100.0)
        normalized_throughput = self.normalize(raw_throughput, self.limit)
        upper_util_after = self.upper_load / self.limit
        lower_util_after = self.lower_load / self.limit
        path_imbalance = abs(upper_util_after - lower_util_after)
        action_change = 1.0 if action != old_previous_action else 0.0
        action_impact = other_util - selected_util

        reward = (
            BASE_REWARD
            + W_THROUGHPUT * normalized_throughput
            - W_DELAY * normalized_delay
            - W_PACKET_LOSS * normalized_packet_loss
            - W_IMBALANCE * path_imbalance
            - W_ACTION_CHANGE * action_change
            + action_impact
        )
        reward = max(-2.0, min(2.0, reward))

        self.previous_action = action
        self.current_index += 1
        next_state = self.get_state()
        done = self.current_index >= len(self.demands)

        info = {
            "flow_id": demand["flow_id"], "src": demand["src"], "dst": demand["dst"],
            "start_time": demand["start_time"], "size_kb": flow_size,
            "selected_path": selected_path, "previous_action": old_previous_action,
            "upper_load": self.upper_load, "lower_load": self.lower_load,
            "selected_load": selected_load, "raw_delay": raw_delay,
            "raw_packet_loss": raw_packet_loss, "raw_throughput": raw_throughput,
            "normalized_delay": normalized_delay, "normalized_packet_loss": normalized_packet_loss,
            "normalized_throughput": normalized_throughput, "path_imbalance": path_imbalance,
            "action_change": action_change, "action_impact": action_impact, "reward": reward,
        }
        return next_state, reward, done, info
