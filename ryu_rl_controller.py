import csv
import os

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


DEMAND_FILE = "data/trafpy_demands.csv"
IPERF_BASE_PORT = 5001


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "RL"
    METRICS_FILE = RL_METRICS_FILE

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)

        self.path_loads = [0.0 for _ in range(NUM_PATHS)]
        self.path_capacity_kb = PATH_CAPACITY_KB
        self.decay_factor = LOAD_DECAY_FACTOR
        self.default_flow_size_kb = DEFAULT_FLOW_SIZE_KB
        self.previous_action = 0

        self.demand_file_mtime = None
        self.port_to_size = {}
        self.reload_demand_sizes_if_needed(force=True)

    def reload_demand_sizes_if_needed(self, force=False):
        """
        The automated tests rewrite data/trafpy_demands.csv for each evaluation
        seed. Reloading on mtime change lets Ryu use the real flow size for the
        current test trace rather than DEFAULT_FLOW_SIZE_KB.
        """
        if not os.path.exists(DEMAND_FILE):
            self.port_to_size = {}
            self.demand_file_mtime = None
            return

        mtime = os.path.getmtime(DEMAND_FILE)

        if not force and self.demand_file_mtime == mtime:
            return

        mapping = {}

        with open(DEMAND_FILE, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    flow_id = int(row["flow_id"])
                    size_kb = float(row["size_kb"])
                except (KeyError, TypeError, ValueError):
                    continue

                port = IPERF_BASE_PORT + flow_id
                mapping[port] = size_kb

        self.port_to_size = mapping
        self.demand_file_mtime = mtime
        self.logger.info(
            "[RL] Loaded %s flow sizes from %s",
            len(self.port_to_size),
            DEMAND_FILE,
        )

    def extract_flow_size(self, flow_info=None):
        self.reload_demand_sizes_if_needed()

        if not isinstance(flow_info, dict):
            return self.default_flow_size_kb

        # For the client-to-server SYN, tcp_dst is 5001 + flow_id.
        # For the reverse direction, tcp_src may be 5001 + flow_id.
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
        self.path_loads = [
            load * self.decay_factor
            for load in self.path_loads
        ]

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

        path_utils = [
            self.path_loads[i] / self.path_capacity_kb[i]
            for i in range(NUM_PATHS)
        ]

        self.logger.info(
            "[RL] decision=%s src=%s dst=%s state=%s path=%s "
            "flow_size=%.2f loads=%s utils=%s",
            self.flow_counter,
            src,
            dst,
            state,
            path,
            flow_size_kb,
            [round(load, 2) for load in self.path_loads],
            [round(util, 3) for util in path_utils],
        )

        self.flow_counter += 1
        return path
