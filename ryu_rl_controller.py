from config import DEFAULT_FLOW_SIZE_KB, LOAD_BALANCE_LIMIT_KB, LOAD_DECAY_FACTOR
from ryu_common import BaseMultipathController
from rl_policy import build_state, choose_path, set_network_state


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "RL"
    METRICS_FILE = "data/rl_metrics.csv"

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)

        self.upper_load = 0.0
        self.lower_load = 0.0
        self.load_balance_limit_kb = LOAD_BALANCE_LIMIT_KB
        self.decay_factor = LOAD_DECAY_FACTOR
        self.default_flow_size_kb = DEFAULT_FLOW_SIZE_KB
        self.previous_action = 0

    def extract_flow_size(self, flow_info=None):
        """
        If ryu_common passes flow_info with size_kb, use it.
        Otherwise use the shared default estimate.
        """
        if isinstance(flow_info, dict):
            try:
                return float(flow_info.get("size_kb", self.default_flow_size_kb))
            except (TypeError, ValueError):
                return self.default_flow_size_kb

        return self.default_flow_size_kb

    def choose_path(self, src, dst, flow_info=None):
        self.upper_load *= self.decay_factor
        self.lower_load *= self.decay_factor

        flow_size_kb = self.extract_flow_size(flow_info)

        state = build_state(
            upper_load=self.upper_load,
            lower_load=self.lower_load,
            flow_size_kb=flow_size_kb,
            previous_action=self.previous_action,
        )

        set_network_state(state)
        path = choose_path(src, dst, state=state)

        if path == "upper":
            self.upper_load += flow_size_kb
            self.previous_action = 0
        else:
            self.lower_load += flow_size_kb
            self.previous_action = 1

        self.logger.info(
            "[RL] decision=%s src=%s dst=%s state=%s path=%s flow_size=%.2f upper_load=%.2f lower_load=%.2f",
            self.flow_counter,
            src,
            dst,
            state,
            path,
            flow_size_kb,
            self.upper_load,
            self.lower_load,
        )

        self.flow_counter += 1
        return path
