import csv
import os

import numpy as np

from config import (
    DEFAULT_FLOW_SIZE_KB,
    LOAD_DECAY_FACTOR,
    NUM_PATHS,
    PATH_CAPACITY_KB,
    PATH_TO_ACTION,
    RL_METRICS_FILE,
)
from ryu_common import BaseMultipathController
from rl_policy import build_state, choose_path, set_network_state


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "RL"
    METRICS_FILE = RL_METRICS_FILE

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)

        self.path_loads = np.zeros(NUM_PATHS, dtype=float)
        self.path_capacity_kb = np.array(PATH_CAPACITY_KB, dtype=float)
        self.decay_factor = LOAD_DECAY_FACTOR
        self.default_flow_size_kb = DEFAULT_FLOW_SIZE_KB
        self.previous_action = 0
        self.port_to_size = self.load_demand_sizes_by_port()

    def load_demand_sizes_by_port(self):
        demand_file = "data/trafpy_demands.csv"
        mapping = {}

        if not os.path.exists(demand_file):
            return mapping

        with open(demand_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    flow_id = int(row["flow_id"])
                    size_kb = float(row["size_kb"])
                    port = 5001 + flow_id
                    mapping[port] = size_kb
                except (KeyError, ValueError):
                    continue

        return mapping

    def extract_flow_size(self, flow_info=None):
        if not isinstance(flow_info, dict):
            return self.default_flow_size_kb

        for key in ["tcp_dst", "tcp_src"]:
            try:
                port = int(flow_info.get(key))
            except (TypeError, ValueError):
                continue

            if port in self.port_to_size:
                return self.port_to_size[port]

        return self.default_flow_size_kb

    def choose_path(self, src, dst, flow_info=None):
        # Decay estimated loads so older flows gradually stop affecting state.
        self.path_loads *= self.decay_factor

        flow_size_kb = self.extract_flow_size(flow_info)

        state = build_state(
            path_loads=self.path_loads,
            flow_size_kb=flow_size_kb,
            previous_action=self.previous_action,
        )

        set_network_state(state)
        path = choose_path(src, dst, state=state)
        action = int(PATH_TO_ACTION[path])

        self.path_loads[action] += flow_size_kb
        self.previous_action = action

        path_utils = self.path_loads / self.path_capacity_kb

        self.logger.info(
            "[RL] decision=%s src=%s dst=%s state=%s path=%s "
            "flow_size=%.2f loads=%s utils=%s",
            self.flow_counter,
            src,
            dst,
            state,
            path,
            flow_size_kb,
            [round(load, 2) for load in self.path_loads.tolist()],
            [round(util, 3) for util in path_utils.tolist()],
        )

        self.flow_counter += 1
        return path
