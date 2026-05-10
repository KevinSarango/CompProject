import csv
import os
from config import LOAD_BALANCE_LIMIT_KB, LOAD_DECAY_FACTOR, DEFAULT_FLOW_SIZE_KB
from ryu_common import BaseMultipathController
from rl_policy import choose_path, build_state_key, set_network_state

class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "RL"
    METRICS_FILE = "data/rl_metrics.csv"

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.upper_load = 0.0
        self.lower_load = 0.0
        self.previous_action = 0
        self.limit = LOAD_BALANCE_LIMIT_KB
        self.decay = LOAD_DECAY_FACTOR
        self.default_flow_size_kb = DEFAULT_FLOW_SIZE_KB
        self.flow_sizes = self.load_flow_sizes()

    def load_flow_sizes(self):
        demand_file = "data/trafpy_demands.csv"
        if not os.path.exists(demand_file):
            return {}
        sizes = {}
        with open(demand_file) as f:
            for row in csv.DictReader(f):
                sizes[int(row["flow_id"])] = float(row["size_kb"])
        return sizes

    def get_flow_size_from_ports(self, flow_info):
        if not flow_info:
            return self.default_flow_size_kb
        dst_port = int(flow_info.get("dst_port", 0))
        src_port = int(flow_info.get("src_port", 0))
        if dst_port >= 5001:
            return self.flow_sizes.get(dst_port - 5001, self.default_flow_size_kb)
        if src_port >= 5001:
            return self.flow_sizes.get(src_port - 5001, self.default_flow_size_kb)
        return self.default_flow_size_kb

    def choose_path(self, src, dst, flow_info=None):
        flow_size_kb = self.get_flow_size_from_ports(flow_info)
        self.upper_load *= self.decay
        self.lower_load *= self.decay
        state_key = build_state_key(self.upper_load, self.lower_load, flow_size_kb, self.previous_action)
        set_network_state(state_key)
        path = choose_path(src, dst, state_key=state_key)

        if path == "upper":
            self.upper_load += flow_size_kb
            self.previous_action = 0
        else:
            self.lower_load += flow_size_kb
            self.previous_action = 1

        self.logger.info(
            "[RL] decision=%s src=%s dst=%s state=%s path=%s size=%.1fKB upper_load=%.2f lower_load=%.2f",
            self.flow_counter, src, dst, state_key, path, flow_size_kb, self.upper_load, self.lower_load
        )
        self.flow_counter += 1
        return path
