from config import (
    LOAD_BALANCE_LIMIT_KB,
    LOAD_DECAY_FACTOR,
    DEFAULT_FLOW_SIZE_KB,
)
from ryu_common import BaseMultipathController
from rl_policy import choose_path, set_network_state


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

        # If one path is clearly more loaded, prefer the less-loaded path.
        # This keeps the deployed RL policy from overloading one side.
        self.override_threshold = 0.15

    def estimate_state(self):
        upper_util = self.upper_load / self.load_balance_limit_kb
        lower_util = self.lower_load / self.load_balance_limit_kb

        busy_threshold = 0.70
        balanced_threshold = 0.15

        upper_busy = upper_util >= busy_threshold
        lower_busy = lower_util >= busy_threshold

        if upper_busy and lower_busy:
            return "3"

        if upper_busy:
            return "1"

        if lower_busy:
            return "2"

        if abs(upper_util - lower_util) <= balanced_threshold:
            return "0"

        if upper_util > lower_util:
            return "1"

        return "2"

    def less_loaded_path(self):
        upper_util = self.upper_load / self.load_balance_limit_kb
        lower_util = self.lower_load / self.load_balance_limit_kb

        if upper_util <= lower_util:
            return "upper"

        return "lower"

    def choose_path(self, src, dst, tcp_dst=None, flow_size_kb=None):
        if flow_size_kb is None:
            flow_size_kb = self.default_flow_size_kb

        # Decay current estimated loads to simulate older flows completing.
        self.upper_load *= self.decay_factor
        self.lower_load *= self.decay_factor

        state = self.estimate_state()
        set_network_state(state)

        qtable_path = choose_path(src, dst)
        load_path = self.less_loaded_path()

        upper_util = self.upper_load / self.load_balance_limit_kb
        lower_util = self.lower_load / self.load_balance_limit_kb
        util_gap = abs(upper_util - lower_util)

        # RL first, but if the estimated load is clearly imbalanced,
        # use the less-loaded path to avoid making the network worse.
        if util_gap >= self.override_threshold:
            final_path = load_path
            reason = "load_override"
        else:
            final_path = qtable_path
            reason = "q_table"

        if final_path == "upper":
            self.upper_load += flow_size_kb
        else:
            self.lower_load += flow_size_kb

        self.logger.info(
            "[RL] decision=%s src=%s dst=%s tcp_dst=%s size=%.1fKB "
            "state=%s qtable=%s final=%s reason=%s "
            "upper_load=%.1f lower_load=%.1f limit=%.1f",
            self.flow_counter,
            src,
            dst,
            tcp_dst,
            flow_size_kb,
            state,
            qtable_path,
            final_path,
            reason,
            self.upper_load,
            self.lower_load,
            self.load_balance_limit_kb,
        )

        self.flow_counter += 1

        return final_path
