from ryu_common import BaseMultipathController
from rl_policy import choose_path, set_network_state


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "RL"
    METRICS_FILE = "data/rl_metrics.csv"

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)

        self.upper_load = 0.0
        self.lower_load = 0.0

        # Match the training environment approximately.
        self.path_capacity_kb = 3000.0
        self.decay_factor = 0.85

        # Since current controller still decides per host-pair, this is an
        # estimated average flow demand.
        self.default_flow_size_kb = 500.0

    def estimate_state(self):
        upper_util = self.upper_load / self.path_capacity_kb
        lower_util = self.lower_load / self.path_capacity_kb

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

    def choose_path(self, src, dst):
        """
        Estimate current congestion state from controller-side path load,
        then use the trained Q-table to select a path.

        This is better than cycling states artificially.
        """

        # Decay previous estimated load.
        self.upper_load *= self.decay_factor
        self.lower_load *= self.decay_factor

        state = self.estimate_state()
        set_network_state(state)

        path = choose_path(src, dst)

        if path == "upper":
            self.upper_load += self.default_flow_size_kb
        else:
            self.lower_load += self.default_flow_size_kb

        self.logger.info(
            "[RL] flow=%s src=%s dst=%s state=%s path=%s upper_load=%.2f lower_load=%.2f",
            self.flow_counter,
            src,
            dst,
            state,
            path,
            self.upper_load,
            self.lower_load,
        )

        self.flow_counter += 1

        return path
