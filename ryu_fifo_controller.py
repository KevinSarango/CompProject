from ryu_common import BaseMultipathController


class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "FIFO"
    METRICS_FILE = "data/fifo_metrics.csv"

    def choose_path(self, src, dst):
        """
        FIFO-style baseline:
        Each new flow alternates between upper and lower path.
        This gives us a comparison point against RL.
        """
        path = "upper" if self.flow_counter % 2 == 0 else "lower"
        self.flow_counter += 1
        return path
