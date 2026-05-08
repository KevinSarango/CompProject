from ryu_common import BaseMultipathController
from rl_policy import choose_path


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "RL"
    METRICS_FILE = "data/rl_metrics.csv"

    def choose_path(self, src, dst):
        return choose_path(src, dst)
