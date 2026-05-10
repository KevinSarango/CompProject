from ryu_common import BaseMultipathController
from rl_policy import choose_path, set_network_state


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "RL"
    METRICS_FILE = "data/rl_metrics.csv"

    def choose_path(self, src, dst):
        """
        Deployment approximation.

        The trained Q-table expects one of four states:
            0 = balanced
            1 = upper busy
            2 = lower busy
            3 = both busy

        For this project version, we cycle through those states as flows are
        assigned. This avoids hardcoding states to hosts and lets the controller
        exercise the full learned Q-table.
        """

        state_cycle = ["0", "1", "2", "3"]
        state = state_cycle[self.flow_counter % len(state_cycle)]

        set_network_state(state)

        path = choose_path(src, dst)

        self.logger.info(
            "[RL] flow=%s src=%s dst=%s state=%s path=%s",
            self.flow_counter,
            src,
            dst,
            state,
            path,
        )

        self.flow_counter += 1

        return path
