from ryu_common import BaseMultipathController

class SimpleSwitch13(BaseMultipathController):
    POLICY_NAME = "FIFO"
    METRICS_FILE = "data/fifo_metrics.csv"

    def choose_path(self, src, dst, flow_info=None):
        path = "upper" if self.flow_counter % 2 == 0 else "lower"
        self.flow_counter += 1
        return path
