import csv
import os


class SimpleSDNEnv:
    """
    Demand-driven SDN environment.

    The RL agent trains on the same traffic demand trace that is later replayed
    in Mininet.

    Actions:
        0 = upper path: s1 -> s2 -> s4
        1 = lower path: s1 -> s3 -> s4

    States:
        0 = balanced
        1 = upper path busy
        2 = lower path busy
        3 = both paths busy

    Reward:
        Uses normalized packet loss, delay, throughput, and action impact.
    """

    def __init__(self, demand_file="data/trafpy_demands.csv"):
        self.demand_file = demand_file
        self.demands = self.load_demands()

        self.current_index = 0

        self.upper_load = 0.0
        self.lower_load = 0.0

        # Capacity is now in KB because the demand file uses size_kb.
        # Lower capacity makes congestion show up with 100–1000 KB flows.
        self.path_capacity_kb = 2500.0

        # Simulates old flows completing over time.
        self.decay_factor = 0.85

        # Reward weights.
        self.gamma_packet_loss = 2.0
        self.gamma_delay = 1.5
        self.gamma_throughput = 1.0
        self.gamma_action_impact = 1.0

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
        return self.get_state()

    def normalize(self, value, max_value):
        if max_value == 0:
            return 0.0

        value = max(0.0, min(float(value), float(max_value)))
        return value / max_value

    def get_state(self):
        upper_util = self.upper_load / self.path_capacity_kb
        lower_util = self.lower_load / self.path_capacity_kb

        busy_threshold = 0.70
        balanced_threshold = 0.15

        upper_busy = upper_util >= busy_threshold
        lower_busy = lower_util >= busy_threshold

        if upper_busy and lower_busy:
            return 3

        if upper_busy:
            return 1

        if lower_busy:
            return 2

        if abs(upper_util - lower_util) <= balanced_threshold:
            return 0

        if upper_util > lower_util:
            return 1

        return 2

    def calculate_action_impact(self, selected_load, other_load):
        impact = (other_load - selected_load) / self.path_capacity_kb

        if impact > 1.0:
            return 1.0

        if impact < -1.0:
            return -1.0

        return impact

    def step(self, action):
        if self.current_index >= len(self.demands):
            return self.get_state(), 0.0, True, {}

        demand = self.demands[self.current_index]
        flow_size = demand["size_kb"]

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

        selected_util = selected_load / self.path_capacity_kb

        raw_delay = selected_util * 100.0
        raw_packet_loss = max(0.0, selected_util - 1.0) * 100.0
        raw_throughput = max(0.0, self.path_capacity_kb - selected_load)

        normalized_delay = self.normalize(raw_delay, 100.0)
        normalized_packet_loss = self.normalize(raw_packet_loss, 100.0)
        normalized_throughput = self.normalize(raw_throughput, self.path_capacity_kb)

        action_impact = self.calculate_action_impact(
            selected_load=selected_load,
            other_load=other_load,
        )

        reward = (
            -self.gamma_packet_loss * normalized_packet_loss
            -self.gamma_delay * normalized_delay
            +self.gamma_throughput * normalized_throughput
            +self.gamma_action_impact * action_impact
        )

        self.current_index += 1

        next_state = self.get_state()
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
            "reward": reward,
        }

        return next_state, reward, done, info
