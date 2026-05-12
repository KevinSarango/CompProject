import csv
import os

from config import (
    ACTION_TO_PATH,
    BASE_REWARD,
    DEFAULT_FLOW_SIZE_KB,
    GAMMA_ACTION_IMPACT,
    GAMMA_DELAY,
    GAMMA_IMBALANCE,
    GAMMA_PACKET_LOSS,
    GAMMA_SWITCHING,
    GAMMA_THROUGHPUT,
    LOAD_DECAY_FACTOR,
    NUM_PATHS,
    PATH_CAPACITY_KB,
    PATH_DELAY_FACTOR,
    SMALL_FLOW_KB,
    MEDIUM_FLOW_KB,
    UTILIZATION_DIFF_THRESHOLD,
)


class SimpleSDNEnv:
    """
    Demand-driven SDN environment.

    State:
        least_utilized_path_bin_demand_bin_previous_action

    For diamond:
        NUM_PATHS = 2
        actions = upper, lower

    For three_path:
        NUM_PATHS = 3
        actions = low_delay, balanced, high_bw

    The reward is now path-aware:
        - each path has its own capacity
        - each path has its own delay factor
    """

    def __init__(self, demand_file="data/trafpy_demands.csv"):
        self.demand_file = demand_file
        self.demands = self.load_demands()

        self.current_index = 0
        self.path_loads = [0.0 for _ in range(NUM_PATHS)]
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
        """Replace the active demand trace for a new training episode."""
        self.demands = list(demands)
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        self.path_loads = [0.0 for _ in range(NUM_PATHS)]
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

    def get_demand_bin(self, flow_size_kb):
        if flow_size_kb <= SMALL_FLOW_KB:
            return 0

        if flow_size_kb <= MEDIUM_FLOW_KB:
            return 1

        return 2

    def get_path_utilizations(self):
        utilizations = []

        for index, load in enumerate(self.path_loads):
            capacity = PATH_CAPACITY_KB[index]
            utilizations.append(load / capacity)

        return utilizations

    def get_least_utilized_path_bin(self):
        utilizations = self.get_path_utilizations()

        min_util = min(utilizations)
        max_util = max(utilizations)

        if max_util - min_util <= UTILIZATION_DIFF_THRESHOLD:
            if NUM_PATHS == 3:
                return 1
            return 0

        return utilizations.index(min_util)

    def get_state(self, flow_size_kb):
        least_utilized_bin = self.get_least_utilized_path_bin()
        demand_bin = self.get_demand_bin(flow_size_kb)
        return f"{least_utilized_bin}_{demand_bin}_{self.previous_action}"

    def calculate_action_impact(self, action, selected_util):
        """
        Positive if selected path is less utilized than the other available paths.
        Negative if selected path is more utilized.
        """

        utilizations = self.get_path_utilizations()

        other_utils = [
            util
            for index, util in enumerate(utilizations)
            if index != action
        ]

        if not other_utils:
            return 0.0

        avg_other_util = sum(other_utils) / len(other_utils)
        impact = avg_other_util - selected_util

        return max(-1.0, min(1.0, impact))

    def step(self, action):
        if self.current_index >= len(self.demands):
            return self.get_state(DEFAULT_FLOW_SIZE_KB), 0.0, True, {}

        demand = self.demands[self.current_index]
        flow_size = demand["size_kb"]

        # Simulate old flows completing over time.
        self.path_loads = [
            load * self.decay_factor
            for load in self.path_loads
        ]

        selected_path = ACTION_TO_PATH[str(action)]

        self.path_loads[action] += flow_size

        selected_load = self.path_loads[action]
        selected_capacity = PATH_CAPACITY_KB[action]
        selected_delay_factor = PATH_DELAY_FACTOR[action]

        selected_util = selected_load / selected_capacity

        # Path-aware delay model:
        # high_bw has more capacity but a higher base delay factor.
        raw_delay = selected_util * 100.0 * selected_delay_factor

        # Path-aware packet loss:
        # loss begins when selected load exceeds selected path capacity.
        raw_packet_loss = max(0.0, selected_util - 1.0) * 100.0

        # Path-aware throughput:
        # larger-capacity paths provide more available throughput.
        raw_throughput = max(0.0, selected_capacity - selected_load)

        normalized_delay = self.normalize(raw_delay, 150.0)
        normalized_packet_loss = self.normalize(raw_packet_loss, 100.0)
        normalized_throughput = self.normalize(raw_throughput, selected_capacity)

        action_impact = self.calculate_action_impact(
            action=action,
            selected_util=selected_util,
        )

        utilizations = self.get_path_utilizations()
        imbalance = max(utilizations) - min(utilizations)
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
            "path_loads": list(self.path_loads),
            "path_utilizations": utilizations,
            "selected_load": selected_load,
            "selected_capacity": selected_capacity,
            "selected_delay_factor": selected_delay_factor,
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
